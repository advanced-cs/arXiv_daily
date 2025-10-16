# 自然语言处理 cs.CL

- **最新发布 98 篇**

- **更新 78 篇**

## 最新发布

#### [new 001] MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文针对大模型在时序推理中多实体、复合操作等复杂场景下的不足，提出MemoTime框架，结合时序知识图谱与记忆增强机制，通过分层推理树、动态检索和经验复用，提升时序问答的准确性与效率。**

- **链接: [http://arxiv.org/pdf/2510.13614v1](http://arxiv.org/pdf/2510.13614v1)**

> **作者:** Xingyu Tan; Xiaoyang Wang; Xiwei Xu; Xin Yuan; Liming Zhu; Wenjie Zhang
>
> **摘要:** Large Language Models (LLMs) have achieved impressive reasoning abilities, but struggle with temporal understanding, especially when questions involve multiple entities, compound operators, and evolving event sequences. Temporal Knowledge Graphs (TKGs), which capture vast amounts of temporal facts in a structured format, offer a reliable source for temporal reasoning. However, existing TKG-based LLM reasoning methods still struggle with four major challenges: maintaining temporal faithfulness in multi-hop reasoning, achieving multi-entity temporal synchronization, adapting retrieval to diverse temporal operators, and reusing prior reasoning experience for stability and efficiency. To address these issues, we propose MemoTime, a memory-augmented temporal knowledge graph framework that enhances LLM reasoning through structured grounding, recursive reasoning, and continual experience learning. MemoTime decomposes complex temporal questions into a hierarchical Tree of Time, enabling operator-aware reasoning that enforces monotonic timestamps and co-constrains multiple entities under unified temporal bounds. A dynamic evidence retrieval layer adaptively selects operator-specific retrieval strategies, while a self-evolving experience memory stores verified reasoning traces, toolkit decisions, and sub-question embeddings for cross-type reuse. Comprehensive experiments on multiple temporal QA benchmarks show that MemoTime achieves overall state-of-the-art results, outperforming the strong baseline by up to 24.0%. Furthermore, MemoTime enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo.
>
---
#### [new 002] Efficient Adaptive Transformer: An Empirical Study and Reproducible Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦模型推理效率任务，旨在解决Transformer推理延迟问题。作者提出EAT框架，整合三种自适应技术，并构建开源可复现的基准 pipeline，在保持精度的同时探索动态计算对NLP延迟的优化潜力。**

- **链接: [http://arxiv.org/pdf/2510.12856v1](http://arxiv.org/pdf/2510.12856v1)**

> **作者:** Jan Miller
>
> **备注:** 10 pages, 6 figures, pgfplots tables included; BibTeX compiled to .bbl. Code and reproducibility artifacts referenced in the paper
>
> **摘要:** The Efficient Adaptive Transformer (EAT) framework unifies three adaptive efficiency techniques - progressive token pruning, sparse attention, and dynamic early exiting - into a single, reproducible architecture for input-adaptive inference. EAT provides an open-source benchmarking pipeline that automates data processing, timing, and ablation across GLUE tasks (SST-2, QQP, MNLI). Although this empirical study finds that combining these mechanisms can increase latency in shallow six-layer models, it demonstrates that EAT achieves slightly higher accuracy than the optimized DistilBERT baseline on SST-2, illustrating the potential of dynamic computation for latency-sensitive NLP. The main contribution is the open, end-to-end reproducible framework - complete with scripts, CSV logging, and analysis utilities - intended to serve as a community tool for further research on adaptive transformers.
>
---
#### [new 003] Putting on the Thinking Hats: A Survey on Chain of Thought Fine-tuning from the Perspective of Human Reasoning Mechanism
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在从人类认知机制角度系统分析思维链微调技术。针对现有研究忽视人类推理模式的问题，作者基于“六顶思考帽”理论对方法进行分类梳理，提出未来方向，并整理数据集与性能，建立实时更新的GitHub资源库。**

- **链接: [http://arxiv.org/pdf/2510.13170v1](http://arxiv.org/pdf/2510.13170v1)**

> **作者:** Xiaoshu Chen; Sihang Zhou; Ke Liang; Duanyang Yuan; Haoyuan Chen; Xiaoyu Sun; Linyuan Meng; Xinwang Liu
>
> **摘要:** Chain of thought (CoT) fine-tuning aims to endow large language models (LLMs) with reasoning capabilities by training them on curated reasoning traces. It leverages both supervised and reinforced fine-tuning to cultivate human-like reasoning skills in LLMs, including detailed planning, divergent thinking, intuitive judgment, timely reflection, internal thinking, and fact perception, etc. As CoT fine-tuning has advanced, LLMs have demonstrated substantial improvements in tasks such as mathematical reasoning and code generation. However, existing surveys about CoT fine-tuning primarily focus on technical aspects and overlook a systematic analysis from the perspective of human reasoning mechanisms. Given that the ultimate goal of CoT fine-tuning is to enable LLMs to reason like humans, it is crucial to investigate this technique through the lens of human cognition. To fill this gap, we present the first comprehensive survey of CoT fine-tuning grounded in human reasoning theory. Specifically, inspired by the well-known Six Thinking Hats framework, which systematically characterizes common human thinking modes using six metaphorical hats, we classify and examine CoT fine-tuning methods through this lens. Furthermore, building upon this theory, we outline potential directions for future research in CoT fine-tuning. In addition, we compile a comprehensive overview of existing datasets and model performances, and a real-time GitHub repository \footnote{https://github.com/AI-Chen/Awesome-CoT-Finetuning} that continuously tracks recent advances in this area is maintained. We hope this survey will serve as a valuable resource to inspire innovation and foster progress in this rapidly evolving field.
>
---
#### [new 004] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 该论文研究多轮文本到SQL生成任务，旨在解决现有方法缺乏执行反馈与对话连贯性的问题。提出MTSQL-R1框架，通过代理训练引入数据库交互与记忆机制，实现查询的迭代生成、验证与修正，提升长周期对话下的SQL生成准确性与一致性。**

- **链接: [http://arxiv.org/pdf/2510.12831v1](http://arxiv.org/pdf/2510.12831v1)**

> **作者:** Taicheng Guo; Hai Wang; ChaoChun Liu; Mohsen Golalikhani; Xin Chen; Xiangliang Zhang; Chandan K. Reddy
>
> **摘要:** Multi-turn Text-to-SQL aims to translate a user's conversational utterances into executable SQL while preserving dialogue coherence and grounding to the target schema. However, most existing systems only regard this task as a simple text translation task and follow a short-horizon paradigm, generating a query per turn without execution, explicit verification, and refinement, which leads to non-executable or incoherent outputs. We present MTSQL-R1, an agentic training framework for long-horizon multi-turn Text-to-SQL. We cast the task as a Markov Decision Process (MDP) in which an agent interacts with (i) a database for execution feedback and (ii) a persistent dialogue memory for coherence verification, performing an iterative propose to execute -> verify -> refine cycle until all checks pass. Experiments on COSQL and SPARC demonstrate that MTSQL-R1 consistently outperforms strong baselines, highlighting the importance of environment-driven verification and memory-guided refinement for conversational semantic parsing. Full recipes (including code, trained models, logs, reasoning trajectories, etc.) will be released after the internal review to contribute to community research.
>
---
#### [new 005] Stable LLM Ensemble: Interaction between Example Representativeness and Diversity
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究一-shot大语言模型集成，解决示例选择与多样性对性能的影响。提出基于质心的代表性示例方法，结合高温度采样，提升准确率与鲁棒性，验证了示例代表性和控制多样性对一-shot集成的重要性。**

- **链接: [http://arxiv.org/pdf/2510.13143v1](http://arxiv.org/pdf/2510.13143v1)**

> **作者:** Junichiro Niimi
>
> **摘要:** Large language models (LLMs) have achieved remarkable results in wide range of domains. However, the accuracy and robustness of one-shot LLM predictions remain highly sensitive to the examples and the diversity among ensemble members. This study systematically investigates the effects of example representativeness (one-shot strategy) and output diversity (sampling temperature) on LLM ensemble performance. Two one-shot strategies are compared: centroid-based representative examples (proposed) and randomly sampled examples (baseline) and sampling temperature also is varied. The proposed approach with higher temperature setting significantly outperforms random selection by +7.6% (macro-F1) and -10.5% (RMSE). Furthermore, the proposed model exceeds 5-shot prompting by +21.1% (macro-F1) and -24.0% (RMSE). Our findings demonstrate that combining representative example selection with increased temperature provides the appropriate level of diversity to the ensemble. This work highlights the practical importance of both example selection and controlled diversity in designing effective one-shot LLM ensembles.
>
---
#### [new 006] Do You Get the Hint? Benchmarking LLMs on the Board Game Concept
- **分类: cs.CL**

- **简介: 该论文提出“Concept”这一基于自然语言的猜词游戏，用于评估大语言模型在抽象推理（尤其是溯因推理）中的表现。任务聚焦模型对玩家意图的理解与假设修正能力，并在多语言场景下测试，发现现有模型表现远低于人类，尤其在低资源语言中更差。**

- **链接: [http://arxiv.org/pdf/2510.13271v1](http://arxiv.org/pdf/2510.13271v1)**

> **作者:** Ine Gevers; Walter Daelemans
>
> **摘要:** Large language models (LLMs) have achieved striking successes on many benchmarks, yet recent studies continue to expose fundamental weaknesses. In particular, tasks that require abstract reasoning remain challenging, often because they use representations such as grids, symbols, or visual patterns that differ from the natural language data LLMs are trained on. In this paper, we introduce Concept, a simple word-guessing board game, as a benchmark for probing abductive reasoning in a representation that is much closer to LLM pre-training data: natural language. Our results show that this game, easily solved by humans (with a success rate of over 90\%), is still very challenging for state-of-the-art LLMs (no model exceeds 40\% success rate). Specifically, we observe that LLMs struggle with interpreting other players' strategic intents, and with correcting initial hypotheses given sequential information updates. In addition, we extend the evaluation across multiple languages, and find that the LLM performance drops further in lower-resource languages (Dutch, French, and Spanish) compared to English.
>
---
#### [new 007] Doing Things with Words: Rethinking Theory of Mind Simulation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文探讨大语言模型是否具备真正的心理理论（ToM）能力。通过Concordia模型测试GPT-4在社会情境中基于信念推理而行动的能力，发现其表现依赖统计关联而非深层推理，揭示当前LLM的ToM模拟存在局限，需更严格的基于行为的评估方法。**

- **链接: [http://arxiv.org/pdf/2510.13395v1](http://arxiv.org/pdf/2510.13395v1)**

> **作者:** Agnese Lombardi; Alessandro Lenci
>
> **摘要:** Language is fundamental to human cooperation, facilitating not only the exchange of information but also the coordination of actions through shared interpretations of situational contexts. This study explores whether the Generative Agent-Based Model (GABM) Concordia can effectively model Theory of Mind (ToM) within simulated real-world environments. Specifically, we assess whether this framework successfully simulates ToM abilities and whether GPT-4 can perform tasks by making genuine inferences from social context, rather than relying on linguistic memorization. Our findings reveal a critical limitation: GPT-4 frequently fails to select actions based on belief attribution, suggesting that apparent ToM-like abilities observed in previous studies may stem from shallow statistical associations rather than true reasoning. Additionally, the model struggles to generate coherent causal effects from agent actions, exposing difficulties in processing complex social interactions. These results challenge current statements about emergent ToM-like capabilities in LLMs and highlight the need for more rigorous, action-based evaluation frameworks.
>
---
#### [new 008] MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医疗大模型的知识编辑任务，旨在解决知识更新与错误修正问题。针对现有方法存在检索不准和难以批量编辑的缺陷，提出MedREK框架，并构建新基准MedVersa，实现精准、高效的单次与批量医学知识编辑。**

- **链接: [http://arxiv.org/pdf/2510.13500v1](http://arxiv.org/pdf/2510.13500v1)**

> **作者:** Shujun Xia; Haokun Lin; Yichen Wu; Yinan Zhou; Zixuan Li; Zhongwei Wan; Xingrun Xing; Yefeng Zheng; Xiang Li; Caifeng Shan; Zhenan Sun; Quanzheng Li
>
> **备注:** Preprint, work in progress
>
> **摘要:** LLMs hold great promise for healthcare applications, but the rapid evolution of medical knowledge and errors in training data often cause them to generate outdated or inaccurate information, limiting their applicability in high-stakes clinical practice. Model editing has emerged as a potential remedy without full retraining. While parameter-based editing often compromises locality and is thus ill-suited for the medical domain, retrieval-based editing offers a more viable alternative. However, it still faces two critical challenges: (1) representation overlap within the medical knowledge space often causes inaccurate retrieval and reduces editing accuracy; (2) existing methods are restricted to single-sample edits, while batch-editing remains largely unexplored despite its importance for real-world medical applications. To address these challenges, we first construct MedVersa, \hk{an enhanced benchmark with broader coverage of medical subjects, designed to evaluate both single and batch edits under strict locality constraints}. We then propose MedREK, a retrieval-based editing framework that integrates a shared query-key module for precise matching with an attention-based prompt encoder for informative guidance. Experimental results on various medical benchmarks demonstrate that our MedREK achieves superior performance across different core metrics and provides the first validated solution for batch-editing in medical LLMs. Our code and dataset are available at https://github.com/mylittleriver/MedREK.
>
---
#### [new 009] Cancer Diagnosis Categorization in Electronic Health Records Using Large Language Models and BioBERT: Model Performance Evaluation Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属文本分类任务，旨在评估大语言模型与BioBERT在电子健康记录中癌症诊断分类的性能。研究比较了多种模型对结构化与非结构化文本的分类效果，并通过专家验证，探讨其临床应用可靠性。**

- **链接: [http://arxiv.org/pdf/2510.12813v1](http://arxiv.org/pdf/2510.12813v1)**

> **作者:** Soheil Hashtarkhani; Rezaur Rashid; Christopher L Brett; Lokesh Chinthala; Fekede Asefa Kumsa; Janet A Zink; Robert L Davis; David L Schwartz; Arash Shaban-Nejad
>
> **备注:** 8 Pages
>
> **摘要:** Electronic health records contain inconsistently structured or free-text data, requiring efficient preprocessing to enable predictive health care models. Although artificial intelligence-driven natural language processing tools show promise for automating diagnosis classification, their comparative performance and clinical reliability require systematic evaluation. The aim of this study is to evaluate the performance of 4 large language models (GPT-3.5, GPT-4o, Llama 3.2, and Gemini 1.5) and BioBERT in classifying cancer diagnoses from structured and unstructured electronic health records data. We analyzed 762 unique diagnoses (326 International Classification of Diseases (ICD) code descriptions, 436free-text entries) from 3456 records of patients with cancer. Models were tested on their ability to categorize diagnoses into 14predefined categories. Two oncology experts validated classifications. BioBERT achieved the highest weighted macro F1-score for ICD codes (84.2) and matched GPT-4o in ICD code accuracy (90.8). For free-text diagnoses, GPT-4o outperformed BioBERT in weighted macro F1-score (71.8 vs 61.5) and achieved slightly higher accuracy (81.9 vs 81.6). GPT-3.5, Gemini, and Llama showed lower overall performance on both formats. Common misclassification patterns included confusion between metastasis and central nervous system tumors, as well as errors involving ambiguous or overlapping clinical terminology. Although current performance levels appear sufficient for administrative and research use, reliable clinical applications will require standardized documentation practices alongside robust human oversight for high-stakes decision-making.
>
---
#### [new 010] Mathematics with large language models as provers and verifiers
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **简介: 该论文研究大语言模型在定理证明中的应用，旨在解决数学命题证明的正确性问题。作者提出多智能体协作框架，结合GPT-5作为证明生成与验证模块，并通过Lean工具进行形式化验证，成功求解多数2025年IMO题目及部分数论猜想。**

- **链接: [http://arxiv.org/pdf/2510.12829v1](http://arxiv.org/pdf/2510.12829v1)**

> **作者:** Hieu Le Duc; Leo Liberti
>
> **摘要:** During 2024 and 2025 the discussion about the theorem-proving capabilities of large language models started reporting interesting success stories, mostly to do with difficult exercises (such as problems from the International Mathematical Olympiad), but also with conjectures [Feldman & Karbasi, arXiv:2509.18383v1] formulated for the purpose of verifying whether the artificial intelligence could prove it. In this paper we report a theorem proving feat achieved by ChatGPT by using a protocol involving different prover and verifier instances of the gpt-5 model working collaboratively. To make sure that the produced proofs do not suffer from hallucinations, the final proof is formally verified by the lean proof assistant, and the conformance of premises and conclusion of the lean code is verified by a human. Our methodology was able to solve five out of six 2025 IMO problems, and close a third of the sixty-six number theory conjectures in [Cohen, Journal of Integer Sequences, 2025].
>
---
#### [new 011] Make an Offer They Can't Refuse: Grounding Bayesian Persuasion in Real-World Dialogues without Pre-Commitment
- **分类: cs.CL; cs.GT**

- **简介: 该论文研究基于贝叶斯说服（BP）的LLM说服策略，解决现实对话中缺乏预承诺和信息不对称利用的问题。提出SFNL与FNL两种自然语言BP方法，通过承诺-沟通机制提升说服效果，在多任务和多说服对象上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.13387v1](http://arxiv.org/pdf/2510.13387v1)**

> **作者:** Buwei He; Yang Liu; Zhaowei Zhang; Zixia Jia; Huijia Wu; Zhaofeng He; Zilong Zheng; Yipeng Kang
>
> **摘要:** Persuasion, a fundamental social capability for humans, remains a challenge for AI systems such as large language models (LLMs). Current studies often overlook the strategic use of information asymmetry in message design or rely on strong assumptions regarding pre-commitment. In this work, we explore the application of Bayesian Persuasion (BP) in natural language within single-turn dialogue settings, to enhance the strategic persuasion capabilities of LLMs. Our framework incorporates a commitment-communication mechanism, where the persuader explicitly outlines an information schema by narrating their potential types (e.g., honest or dishonest), thereby guiding the persuadee in performing the intended Bayesian belief update. We evaluate two variants of our approach: Semi-Formal-Natural-Language (SFNL) BP and Fully-Natural-Language (FNL) BP, benchmarking them against both naive and strong non-BP (NBP) baselines within a comprehensive evaluation framework. This framework covers a diverse set of persuadees -- including LLM instances with varying prompts and fine-tuning and human participants -- across tasks ranging from specially designed persuasion scenarios to general everyday situations. Experimental results on LLM-based agents reveal three main findings: (1) LLMs guided by BP strategies consistently achieve higher persuasion success rates than NBP baselines; (2) SFNL exhibits greater credibility and logical coherence, while FNL shows stronger emotional resonance and robustness in naturalistic conversations; (3) with supervised fine-tuning, smaller models can attain BP performance comparable to that of larger models.
>
---
#### [new 012] Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对智能客服系统在数据构建、多轮对话、业务适配、协作能力与评估方面的挑战，提出WOWService系统，融合大模型与多智能体架构，实现自主任务管理与协同服务，在美团应用中显著提升用户满意度。**

- **链接: [http://arxiv.org/pdf/2510.13291v1](http://arxiv.org/pdf/2510.13291v1)**

> **作者:** Xuxin Cheng; Ke Zeng; Zhiquan Cao; Linyi Dai; Wenxuan Gao; Fei Han; Ai Jian; Feng Hong; Wenxing Hu; Zihe Huang; Dejian Kong; Jia Leng; Zhuoyuan Liao; Pei Liu; Jiaye Lin; Xing Ma; Jingqing Ruan; Jiaxing Song; Xiaoyu Tan; Ruixuan Xiao; Wenhui Yu; Wenyu Zhan; Haoxing Zhang; Chao Zhou; Hao Zhou; Shaodong Zheng; Ruinian Chen; Siyuan Chen; Ziyang Chen; Yiwen Dong; Yaoyou Fan; Yangyi Fang; Yang Gan; Shiguang Guo; Qi He; Chaowen Hu; Binghui Li; Dailin Li; Xiangyu Li; Yan Li; Chengjian Liu; Xiangfeng Liu; Jiahui Lv; Qiao Ma; Jiang Pan; Cong Qin; Chenxing Sun; Wen Sun; Zhonghui Wang; Abudukelimu Wuerkaixi; Xin Yang; Fangyi Yuan; Yawen Zhu; Tianyi Zhai; Jie Zhang; Runlai Zhang; Yao Xu; Yiran Zhao; Yifan Wang; Xunliang Cai; Yangen Hu; Cao Liu; Lu Pan; Xiaoli Wang; Bo Xiao; Wenyuan Yao; Qianlin Zhou; Benchang Zhu
>
> **备注:** 36 pages, 14 figures
>
> **摘要:** Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.
>
---
#### [new 013] NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **简介: 该论文提出NExT-OMNI，旨在解决现有模型在多模态任意到任意生成与理解任务中架构不统一、效率低的问题。通过离散流匹配实现统一建模，支持跨模态生成、检索与多轮交互，提升响应效率与应用广度，并开源代码与模型。**

- **链接: [http://arxiv.org/pdf/2510.13721v1](http://arxiv.org/pdf/2510.13721v1)**

> **作者:** Run Luo; Xiaobo Xia; Lu Wang; Longze Chen; Renke Shan; Jing Luo; Min Yang; Tat-Seng Chua
>
> **摘要:** Next-generation multimodal foundation models capable of any-to-any cross-modal generation and multi-turn interaction will serve as core components of artificial general intelligence systems, playing a pivotal role in human-machine interaction. However, most existing multimodal models remain constrained by autoregressive architectures, whose inherent limitations prevent a balanced integration of understanding and generation capabilities. Although hybrid and decoupling strategies have been explored to address these tasks within unified frameworks separately, their redundant, non-integrated designs limit their applicability to broader scenarios, such as cross-modal retrieval.In this work, we introduce NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling through discrete flow paradigms. By leveraging metric-induced probability paths and kinetic optimal velocities, NExT-OMNI natively supports any-to-any understanding and generation with enhanced response efficiency, while enabling broader application scenarios through concise unified representations rather than task-decoupled designs. Trained on large-scale interleaved text, image, video, and audio data, NExT-OMNI delivers competitive performance on multimodal generation and understanding benchmarks, while outperforming prior unified models in multi-turn multimodal interaction and cross-modal retrieval, highlighting its architectural advantages as a next-generation multimodal foundation model. To advance further research, we release training details, data protocols, and open-source both the code and model checkpoints.
>
---
#### [new 014] Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在系统评估阿拉伯语大语言模型的现有基准。它分析了40多个基准，提出分类体系，揭示在时效性、多轮对话和文化适配上的不足，并比较数据构建方法，为未来研究提供指导。**

- **链接: [http://arxiv.org/pdf/2510.13430v1](http://arxiv.org/pdf/2510.13430v1)**

> **作者:** Ahmed Alzubaidi; Shaikha Alsuwaidi; Basma El Amel Boussaha; Leen AlQadi; Omar Alkaabi; Mohammed Alyafeai; Hamza Alobeidli; Hakim Hacid
>
> **摘要:** This survey provides the first systematic review of Arabic LLM benchmarks, analyzing 40+ evaluation benchmarks across NLP tasks, knowledge domains, cultural understanding, and specialized capabilities. We propose a taxonomy organizing benchmarks into four categories: Knowledge, NLP Tasks, Culture and Dialects, and Target-Specific evaluations. Our analysis reveals significant progress in benchmark diversity while identifying critical gaps: limited temporal evaluation, insufficient multi-turn dialogue assessment, and cultural misalignment in translated datasets. We examine three primary approaches: native collection, translation, and synthetic generation discussing their trade-offs regarding authenticity, scale, and cost. This work serves as a comprehensive reference for Arabic NLP researchers, providing insights into benchmark methodologies, reproducibility standards, and evaluation metrics while offering recommendations for future development.
>
---
#### [new 015] Beyond Single-Reward: Multi-Pair, Multi-Perspective Preference Optimization for Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究机器翻译中的人类偏好对齐任务，针对现有偏好优化方法奖励信号不准确和数据利用率低的问题，提出M²PO框架，通过多视角奖励机制和多候选对构建策略，提升翻译质量与事实一致性。**

- **链接: [http://arxiv.org/pdf/2510.13434v1](http://arxiv.org/pdf/2510.13434v1)**

> **作者:** Hao Wang; Linlong Xu; Heng Liu; Yangyang Liu; Xiaohu Zhao; Bo Zeng; Liangying Shao; Longyue Wang; Weihua Luo; Kaifu Zhang
>
> **摘要:** Direct Preference Optimization (DPO) is a powerful paradigm for aligning Large Language Models (LLMs) to human preferences in Machine Translation (MT), but current methods are hindered by two fundamental challenges: (1) flawed reward signals from Quality Estimation (QE) models that overlook critical errors like translation hallucination, and (2) inefficient data utilization that discards valuable learning signals by selecting only a single win-loss pair. To address these limitations, we introduce M^2PO: Multi-Pair, Multi-Perspective Preference Optimization. Our framework integrates a multi-perspective reward engine that creates a more robust signal by combining two key viewpoints: a new hallucination penalty for factuality, and an innovative dynamic quality score that adaptively fuses external evaluations with the model's own evolving judgment. This is synergistically paired with a multi-pair construction strategy that systematically creates a comprehensive set of preference pairs from the entire pool of translation candidates. This synergistic approach ensures the model learns from a richer spectrum of quality trade-offs, leading to more robust and faithful translations. On challenging WMT21-22 benchmarks, M^2PO substantially outperforms existing preference optimization methods and demonstrates highly competitive performance against leading proprietary LLMs.
>
---
#### [new 016] Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation
- **分类: cs.CL**

- **简介: 该论文研究RAG系统中LLM输出的置信度估计，旨在提升高风险领域中模型回答的可信度。提出利用FFN激活值作为自回归信号，通过序列分类建模置信度，避免softmax导致的信息损失，并在金融客服场景中验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.13750v1](http://arxiv.org/pdf/2510.13750v1)**

> **作者:** Zhiqi Huang; Vivek Datla; Chenyang Zhu; Alfy Samuel; Daben Liu; Anoop Kumar; Ritesh Soni
>
> **备注:** UncertaiNLP at EMNLP 2025
>
> **摘要:** We propose a method for confidence estimation in retrieval-augmented generation (RAG) systems that aligns closely with the correctness of large language model (LLM) outputs. Confidence estimation is especially critical in high-stakes domains such as finance and healthcare, where the cost of an incorrect answer outweighs that of not answering the question. Our approach extends prior uncertainty quantification methods by leveraging raw feed-forward network (FFN) activations as auto-regressive signals, avoiding the information loss inherent in token logits and probabilities after projection and softmax normalization. We model confidence prediction as a sequence classification task, and regularize training with a Huber loss term to improve robustness against noisy supervision. Applied in a real-world financial industry customer-support setting with complex knowledge bases, our method outperforms strong baselines and maintains high accuracy under strict latency constraints. Experiments on Llama 3.1 8B model show that using activations from only the 16th layer preserves accuracy while reducing response latency. Our results demonstrate that activation-based confidence modeling offers a scalable, architecture-aware path toward trustworthy RAG deployment.
>
---
#### [new 017] Text Anomaly Detection with Simplified Isolation Kernel
- **分类: cs.CL**

- **简介: 该论文针对文本异常检测中高维语义嵌入导致的高计算与内存开销问题，提出简化隔离核（SIK），将稠密嵌入映射为低维稀疏表示，在降低资源消耗的同时保持优异检测性能。**

- **链接: [http://arxiv.org/pdf/2510.13197v1](http://arxiv.org/pdf/2510.13197v1)**

> **作者:** Yang Cao; Sikun Yang; Yujiu Yang; Lianyong Qi; Ming Liu
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** Two-step approaches combining pre-trained large language model embeddings and anomaly detectors demonstrate strong performance in text anomaly detection by leveraging rich semantic representations. However, high-dimensional dense embeddings extracted by large language models pose challenges due to substantial memory requirements and high computation time. To address this challenge, we introduce the Simplified Isolation Kernel (SIK), which maps high-dimensional dense embeddings to lower-dimensional sparse representations while preserving crucial anomaly characteristics. SIK has linear time complexity and significantly reduces space complexity through its innovative boundary-focused feature mapping. Experiments across 7 datasets demonstrate that SIK achieves better detection performance than 11 state-of-the-art (SOTA) anomaly detection algorithms while maintaining computational efficiency and low memory cost. All code and demonstrations are available at https://github.com/charles-cao/SIK.
>
---
#### [new 018] 3-Model Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理加速任务，旨在解决小草案模型与大目标模型间token接受率低的问题。作者提出PyramidSD，在草案和目标模型间引入中间判别模型，通过分层解码和模糊接受机制提升接受率和吞吐量，显著提高生成速度。**

- **链接: [http://arxiv.org/pdf/2510.12966v1](http://arxiv.org/pdf/2510.12966v1)**

> **作者:** Sanghyun Byun; Mohanad Odema; Jung Ick Guack; Baisub Lee; Jacob Song; Woo Seong Chung
>
> **备注:** Accepted at NeurIPS SPIGM 2025
>
> **摘要:** Speculative Decoding (SD) accelerates inference in large language models by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, the throughput gains of SD are fundamentally limited by a trade-off between draft model size and token acceptance: smaller draft models generate tokens more quickly but exhibit greater divergence from the target model, resulting in lower acceptance rates and reduced speedups. We introduce Pyramid Speculative Decoding (PyramidSD), an extension of SD that inserts an intermediate qualifier model between the draft and target to bridge the distributional gap in output predictions, allowing smaller model to be used for drafting. This hierarchical decoding strategy improves alignment across models, enabling higher acceptance rates and allowing the use of significantly smaller draft models without sacrificing overall performance. PyramidSD builds on fuzzy acceptance criteria to support relaxed divergence thresholds at each stage, improving throughput. In experiments, PyramidSD achieves up to 1.91x generation speed over standard SD, reaching 124 tokens per second on a consumer GPU (RTX 4090). In small-memory settings with a 1B-parameter draft model and an 8B target model, PyramidSD minimally trades target model quality for improved throughput. Overall, PyramidSD offers a practical approach to enhancing speculative decoding efficiency and can be readily applied to existing inference pipelines.
>
---
#### [new 019] In-Distribution Steering: Balancing Control and Coherence in Language Model Generation
- **分类: cs.CL**

- **简介: 该论文研究语言模型生成中的激活控制问题，提出“分布内 steering”（IDS）方法。通过动态调整干预强度，平衡控制力与文本连贯性，解决传统方法因固定强度导致的生成失真或控制不足问题，提升实际应用效果。**

- **链接: [http://arxiv.org/pdf/2510.13285v1](http://arxiv.org/pdf/2510.13285v1)**

> **作者:** Arthur Vogels; Benjamin Wong; Yann Choho; Annabelle Blangero; Milan Bhan
>
> **摘要:** Activation steering methods control large language model (LLM) behavior by modifying internal activations at inference time. However, most existing activation steering methods rely on a fixed steering strength, leading to either insufficient control or unadapted intervention that degrades text plausibility and coherence. We introduce In-Distribution Steering (IDS), a novel method that adapts steering strength based on the input data distribution in representation space. IDS dynamically adjusts interventions according to how far a given input lies within the distribution, enabling adaptive intervention and generation stability during text generation. Experiments demonstrate that IDS achieves strong accuracy on classification tasks while producing coherent text without collapse, making IDS particularly well suited for real-world applications.
>
---
#### [new 020] DSCD: Large Language Model Detoxification with Self-Constrained Decoding
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的无毒化生成任务，旨在解决现有解码方法依赖外部约束导致资源开销大、流畅性差的问题。提出自约束解码（DSCD）方法，通过调整模型内部层的token分布，在不微调参数的前提下提升生成安全性和流畅性。**

- **链接: [http://arxiv.org/pdf/2510.13183v1](http://arxiv.org/pdf/2510.13183v1)**

> **作者:** Ming Dong; Jinkui Zhang; Bolong Zheng; Xinhui Tu; Po Hu; Tingting He
>
> **备注:** Accepted at EMNLP 2025 MainConference
>
> **摘要:** Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLM detoxification without parameter fine-tuning. DSCD strengthens the inner next-token distribution of the safety layer while weakening that of hallucination and toxic layers during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD's effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD's potential as a practical and scalable solution for safer LLM deployments.
>
---
#### [new 021] Document Intelligence in the Era of Large Language Models: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于综述任务，旨在梳理大语言模型时代文档智能的发展。它分析了多模态、多语言和检索增强等关键方向，探讨挑战并提出未来研究方向，如基于代理的方法和文档专用基础模型。**

- **链接: [http://arxiv.org/pdf/2510.13366v1](http://arxiv.org/pdf/2510.13366v1)**

> **作者:** Weishi Wang; Hengchang Hu; Zhijie Zhang; Zhaochen Li; Hongxin Shao; Daniel Dahlmeier
>
> **摘要:** Document AI (DAI) has emerged as a vital application area, and is significantly transformed by the advent of large language models (LLMs). While earlier approaches relied on encoder-decoder architectures, decoder-only LLMs have revolutionized DAI, bringing remarkable advancements in understanding and generation. This survey provides a comprehensive overview of DAI's evolution, highlighting current research attempts and future prospects of LLMs in this field. We explore key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions, including agent-based approaches and document-specific foundation models. This paper aims to provide a structured analysis of the state-of-the-art in DAI and its implications for both academic and practical applications.
>
---
#### [new 022] Who's Asking? Evaluating LLM Robustness to Inquiry Personas in Factual Question Answering
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在不同用户身份提示下的事实问答鲁棒性。针对用户披露的身份、专长等信息可能影响回答准确性的现象，提出探究“询问者角色”对模型输出的影响，发现此类线索可导致拒绝回答、幻觉等问题，强调需评估模型对用户情境的敏感性。**

- **链接: [http://arxiv.org/pdf/2510.12925v1](http://arxiv.org/pdf/2510.12925v1)**

> **作者:** Nil-Jana Akpinar; Chia-Jung Lee; Vanessa Murdock; Pietro Perona
>
> **摘要:** Large Language Models (LLMs) should answer factual questions truthfully, grounded in objective knowledge, regardless of user context such as self-disclosed personal information, or system personalization. In this paper, we present the first systematic evaluation of LLM robustness to inquiry personas, i.e. user profiles that convey attributes like identity, expertise, or belief. While prior work has primarily focused on adversarial inputs or distractors for robustness testing, we evaluate plausible, human-centered inquiry persona cues that users disclose in real-world interactions. We find that such cues can meaningfully alter QA accuracy and trigger failure modes such as refusals, hallucinated limitations, and role confusion. These effects highlight how model sensitivity to user framing can compromise factual reliability, and position inquiry persona testing as an effective tool for robustness evaluation.
>
---
#### [new 023] Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy Optimization
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大模型推理机制的可解释性，旨在解决强化学习中信用分配模糊的问题。通过分析注意力模式，提出“预规划-锚定”机制，并设计基于注意力信号的精细策略优化方法，提升推理任务性能。**

- **链接: [http://arxiv.org/pdf/2510.13554v1](http://arxiv.org/pdf/2510.13554v1)**

> **作者:** Yang Li; Zhichen Dong; Yuhan Sun; Weixun Wang; Shaopan Xiong; Yijia Luo; Jiashun Liu; Han Lu; Jiamang Wang; Wenbo Su; Bo Zheng; Junchi Yan
>
> **备注:** 23 pages, 8 figures, 5 tables
>
> **摘要:** The reasoning pattern of Large language models (LLMs) remains opaque, and Reinforcement learning (RL) typically applies uniform credit across an entire generation, blurring the distinction between pivotal and routine steps. This work positions attention as a privileged substrate that renders the internal logic of LLMs legible, not merely as a byproduct of computation, but as a mechanistic blueprint of reasoning itself. We first distinguish attention heads between locally and globally focused information processing and reveal that locally focused heads produce a sawtooth pattern near the diagonal indicating phrasal chunks, while globally focused heads expose tokens that exert broad downstream influence over future tokens. We formalize these with two metrics: 1) Windowed Average Attention Distance, which measures the extent of backward attention within a clipped window; 2) Future Attention Influence, which quantifies a token's global importance as the average attention it receives from subsequent tokens. Taken together, these signals reveal a recurring preplan-and-anchor mechanism, where the model first performs a long-range contextual reference to generate an introductory token, which is immediately followed by or coincides with a semantic anchor token that organizes subsequent reasoning. Leveraging these insights, we introduce three novel RL strategies that dynamically perform targeted credit assignment to critical nodes (preplan tokens, anchor tokens, and their temporal coupling) and show consistent performance gains across various reasoning tasks. By aligning optimization with the model's intrinsic reasoning rhythm, we aim to transform opaque optimization into an actionable structure-aware process, hoping to offer a potential step toward more transparent and effective optimization of LLM reasoning.
>
---
#### [new 024] Grounding Long-Context Reasoning with Contextual Normalization for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究检索增强生成（RAG）中的上下文格式影响，提出“上下文归一化”方法，通过标准化上下文表示来提升长上下文推理的稳定性和准确性，解决因格式差异导致的性能波动问题。**

- **链接: [http://arxiv.org/pdf/2510.13191v1](http://arxiv.org/pdf/2510.13191v1)**

> **作者:** Jiamin Chen; Yuchen Li; Xinyu Ma; Xinran Chen; Xiaokun Zhang; Shuaiqiang Wang; Chen Ma; Dawei Yin
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become an essential approach for extending the reasoning and knowledge capacity of large language models (LLMs). While prior research has primarily focused on retrieval quality and prompting strategies, the influence of how the retrieved documents are framed, i.e., context format, remains underexplored. We show that seemingly superficial choices, such as delimiters or structural markers in key-value extraction, can induce substantial shifts in accuracy and stability, even when semantic content is identical. To systematically investigate this effect, we design controlled experiments that vary context density, delimiter styles, and positional placement, revealing the underlying factors that govern performance differences. Building on these insights, we introduce Contextual Normalization, a lightweight strategy that adaptively standardizes context representations before generation. Extensive experiments on both controlled and real-world RAG benchmarks across diverse settings demonstrate that the proposed strategy consistently improves robustness to order variation and strengthens long-context utilization. These findings underscore that reliable RAG depends not only on retrieving the right content, but also on how that content is presented, offering both new empirical evidence and a practical technique for better long-context reasoning.
>
---
#### [new 025] SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs
- **分类: cs.CL**

- **简介: 该论文针对大视觉语言模型（LVLM）易受隐蔽有害提示攻击的问题，提出SHIELD框架。它通过细粒度安全分类与类别化引导，在不重训练情况下实现拒绝、重构或转发等响应，提升模型安全性与鲁棒性，适用于多种LVLM且开销低。**

- **链接: [http://arxiv.org/pdf/2510.13190v1](http://arxiv.org/pdf/2510.13190v1)**

> **作者:** Juan Ren; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs.
>
---
#### [new 026] Sparse Subnetwork Enhancement for Underrepresented Languages in Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大模型在低资源语言上性能不足的问题，提出通过识别并微调语言特异性子网络来增强小语种能力。方法仅更新1%参数，兼顾效果与效率，提升多语言均衡性。**

- **链接: [http://arxiv.org/pdf/2510.13580v1](http://arxiv.org/pdf/2510.13580v1)**

> **作者:** Daniil Gurgurov; Josef van Genabith; Simon Ostermann
>
> **备注:** preprint
>
> **摘要:** Large language models exhibit uneven performance across languages, with substantial gaps between high- and low-resource languages. We present a framework for enhancing monolingual capabilities of LLMs in underrepresented languages while preserving their general-purpose performance through targeted fine-tuning of language-specific subnetworks. Our approach identifies language-specific neurons using Language Activation Probability Entropy and fine-tunes only the weights associated with these neurons, a dedicated subnetwork, on target-language data. Experiments on Llama-3.1-8B and Mistral-Nemo-12B across 12 mid- and low-resource languages demonstrate that our method consistently outperforms full fine-tuning, FFN-only fine-tuning, LoRA adaptation, and random subset fine-tuning baselines while efficiently updating only up to 1% of model parameters. Beyond performance improvements, we observe enhanced favorable training dynamics, cross-lingual representational alignment, and systematic weight update changes. To facilitate future research, we release language-specific neuron identifications for over 100 languages as well as our adaptation pipeline, offering a cost-effective pathway for adapting state-of-the-art models to underrepresented languages.
>
---
#### [new 027] NOSA: Native and Offloadable Sparse Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型长上下文解码效率低的问题，提出NOSA框架，在保持训练时注意力计算不变的前提下，通过引入显式局部性约束，实现KV缓存的高效CPU-GPU卸载，显著提升解码吞吐量。**

- **链接: [http://arxiv.org/pdf/2510.13602v1](http://arxiv.org/pdf/2510.13602v1)**

> **作者:** Yuxiang Huang; Chaojun Xiao; Xu Han; Zhiyuan Liu
>
> **备注:** Preprint
>
> **摘要:** Trainable sparse attention has emerged as a promising solution to address the decoding efficiency bottleneck of LLMs in long-context processing, significantly saving memory accesses while minimally impacting task performance. However, existing sparse attention methods leave a crucial limitation unresolved: the size of the key-value (KV) cache remains unreduced, which constrains on-GPU batch sizes and throttles decoding throughput, especially in large-scale batched inference. In this paper, we show that trainable sparse attention naturally exhibits strong locality in token selection across adjacent decoding steps, thereby enabling KV cache offloading without altering the underlying attention computation. However, the inherent locality remains insufficient to achieve efficient offloading, as the transfer of selected KV pairs between the CPU and GPU continues to dominate the overall decoding cost. Building on this insight, we present NOSA, a trainable sparse attention framework designed to natively support KV cache offloading. NOSA introduces explicit locality constraints by decomposing token selection into query-aware and query-agnostic components, thereby reducing KV transfers while preserving the same attention computation as used during training. We pretrain a 1B-parameter model with NOSA and conduct extensive benchmarks, showing that it preserves near-lossless performance while achieving up to a 2.3x improvement in decoding throughput compared with the vanilla trainable sparse attention baseline (InfLLM-V2).
>
---
#### [new 028] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成中长上下文导致的高延迟与认知负担问题，提出BRIEF-Pro——一种通用轻量级压缩方法，通过短到长上下文合成实现多跳推理的高效摘要，支持灵活控制摘要长度，显著提升问答性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.13799v1](http://arxiv.org/pdf/2510.13799v1)**

> **作者:** Jia-Chen Gu; Junyi Zhang; Di Wu; Yuankai Li; Kai-Wei Chang; Nanyun Peng
>
> **备注:** Code and data: https://github.com/JasonForJoy/BRIEF
>
> **摘要:** As retrieval-augmented generation (RAG) tackles complex tasks, increasingly expanded contexts offer richer information, but at the cost of higher latency and increased cognitive load on the model. To mitigate this bottleneck, especially for intricate multi-hop questions, we introduce BRIEF-Pro. It is a universal, lightweight compressor that distills relevant evidence for a given query from retrieved documents into a concise summary for seamless integration into in-context RAG. Using seed data consisting of relatively short contexts (fewer than 1k words), BRIEF-Pro is trained to perform abstractive compression of extended contexts exceeding 10k words across a wide range of scenarios. Furthermore, BRIEF-Pro offers flexible user control over summary length by allowing users to specify the desired number of sentences. Experiments on four open-domain multi-hop question-answering datasets show that BRIEF-Pro generates more concise and relevant summaries, enhancing performance across small, large, and proprietary language models. With the 70B reader model, 32x compression by BRIEF-Pro improves QA performance by 4.67% on average over LongLLMLingua's 9x, while requiring only 23% of its computational overhead.
>
---
#### [new 029] CoT-Evo: Evolutionary Distillation of Chain-of-Thought for Scientific Reasoning
- **分类: cs.CL**

- **简介: 该论文针对科学推理中大模型生成的思维链存在错误的问题，提出CoT-Evo框架，通过多模型生成、知识增强与进化优化，提升推理质量，并用于蒸馏训练小型模型，在科学推理任务上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2510.13166v1](http://arxiv.org/pdf/2510.13166v1)**

> **作者:** Kehua Feng; Keyan Ding; Zhihui Zhu; Lei Liang; Qiang Zhang; Huajun Chen
>
> **备注:** 28 pages, 3 figures
>
> **摘要:** While chain-of-thought (CoT) distillation from advanced large language models (LLMs) has proven effective in general reasoning tasks, it struggles in scientific domains where even advanced models often produce incorrect or superficial reasoning due to high complexity and specialized knowledge requirements. Directly distilling from such flawed outputs results in low-quality training data and limits the performance of smaller student models. To overcome this, we propose CoT-Evo, an evolutionary CoT distillation framework. It begins by constructing a diverse pool of reasoning trajectories from multiple LLM thinkers, enriches them with automatically retrieved domain knowledge, and iteratively refines the trajectories using novelty-driven selection, reflective recombination and mutation. The refinement is guided by a fitness function that evaluates answer correctness, coherence, and effective knowledge utilization. This results in a high-quality CoT dataset tailored for scientific reasoning. We employ this evolved dataset to fine-tune a compact model, which achieves state-of-the-art performance on scientific reasoning benchmarks. Our work establishes a scalable approach to synthesizing high-fidelity scientific reasoning data from diverse and fallible LLMs.
>
---
#### [new 030] MEDEQUALQA: Evaluating Biases in LLMs with Counterfactual Reasoning
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出MEDEQUALQA，旨在评估大语言模型在医疗决策中因患者代词变化引发的推理偏差。通过控制症状不变、仅改变代词，构建反事实数据集，分析模型推理稳定性，揭示潜在的临床偏见。**

- **链接: [http://arxiv.org/pdf/2510.12818v1](http://arxiv.org/pdf/2510.12818v1)**

> **作者:** Rajarshi Ghosh; Abhay Gupta; Hudson McBride; Anurag Vaidya; Faisal Mahmood
>
> **摘要:** Large language models (LLMs) are increasingly deployed in clinical decision support, yet subtle demographic cues can influence their reasoning. Prior work has documented disparities in outputs across patient groups, but little is known about how internal reasoning shifts under controlled demographic changes. We introduce MEDEQUALQA, a counterfactual benchmark that perturbs only patient pronouns (he/him, she/her, they/them) while holding critical symptoms and conditions (CSCs) constant. Each clinical vignette is expanded into single-CSC ablations, producing three parallel datasets of approximately 23,000 items each (69,000 total). We evaluate a GPT-4.1 model and compute Semantic Textual Similarity (STS) between reasoning traces to measure stability across pronoun variants. Our results show overall high similarity (mean STS >0.80), but reveal consistent localized divergences in cited risk factors, guideline anchors, and differential ordering, even when final diagnoses remain unchanged. Our error analysis highlights certain cases in which the reasoning shifts, underscoring clinically relevant bias loci that may cascade into inequitable care. MEDEQUALQA offers a controlled diagnostic setting for auditing reasoning stability in medical AI.
>
---
#### [new 031] Closing the Gap Between Text and Speech Understanding in LLMs
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音与文本理解在大语言模型中的性能差距问题，提出SALAD方法，通过跨模态蒸馏和主动选择合成数据，在减少语音数据使用的情况下有效缩小文本-语音理解差距。**

- **链接: [http://arxiv.org/pdf/2510.13632v1](http://arxiv.org/pdf/2510.13632v1)**

> **作者:** Santiago Cuervo; Skyler Seto; Maureen de Seyssel; Richard He Bai; Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly; Zakaria Aldeneh
>
> **摘要:** Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.
>
---
#### [new 032] Personal Attribute Leakage in Federated Speech Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究联邦语音识别模型中的个人属性泄露问题，旨在评估在无原始语音访问的情况下，通过模型权重差异推断敏感属性（如性别、口音等）的可行性。工作包括对Wav2Vec2、HuBERT和Whisper进行非参数白盒攻击实验，揭示了预训练数据稀疏属性更易受攻击。**

- **链接: [http://arxiv.org/pdf/2510.13357v1](http://arxiv.org/pdf/2510.13357v1)**

> **作者:** Hamdan Al-Ali; Ali Reza Ghavamipour; Tommaso Caselli; Fatih Turkmen; Zeerak Talat; Hanan Aldarmaki
>
> **备注:** 5 pages, 4 figures, 2 tables
>
> **摘要:** Federated learning is a common method for privacy-preserving training of machine learning models. In this paper, we analyze the vulnerability of ASR models to attribute inference attacks in the federated setting. We test a non-parametric white-box attack method under a passive threat model on three ASR models: Wav2Vec2, HuBERT, and Whisper. The attack operates solely on weight differentials without access to raw speech from target speakers. We demonstrate attack feasibility on sensitive demographic and clinical attributes: gender, age, accent, emotion, and dysarthria. Our findings indicate that attributes that are underrepresented or absent in the pre-training data are more vulnerable to such inference attacks. In particular, information about accents can be reliably inferred from all models. Our findings expose previously undocumented vulnerabilities in federated ASR models and offer insights towards improved security.
>
---
#### [new 033] Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference
- **分类: cs.CL**

- **简介: 该论文属LLM推理加速任务，旨在解决传统推测解码中延迟与接受率的权衡问题。提出Mirror-SD，通过异构并行执行和多令牌流式推测，实现高接受率与低开销，显著提升推理速度。**

- **链接: [http://arxiv.org/pdf/2510.13161v1](http://arxiv.org/pdf/2510.13161v1)**

> **作者:** Nikhil Bhendawade; Kumari Nishu; Arnav Kundu; Chris Bartels; Minsik Cho; Irina Belousova
>
> **摘要:** Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3.
>
---
#### [new 034] GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对MoE模型中专家功能冗余问题，提出无需参数的专家选择优化方法GatePro。通过引入局部竞争机制提升专家多样性，避免重复计算，增强模型容量，可即插即用且无需额外训练参数。**

- **链接: [http://arxiv.org/pdf/2510.13079v1](http://arxiv.org/pdf/2510.13079v1)**

> **作者:** Chen Zheng; Yuhang Cai; Deyi Liu; Jin Ma; Yiyuan Ma; Yuan Yang; Jing Liu; Yutao Zeng; Xun Zhou; Siyuan Qiao
>
> **摘要:** Modern large language models leverage Mixture-of-Experts (MoE) architectures for efficient scaling, but face a critical challenge: functionally similar experts are often selected simultaneously, creating redundant computation and limiting effective model capacity. Existing auxiliary balance loss methods improve token distribution but fail to address the underlying expert diversity problem. We introduce GatePro, a novel parameter-free method that directly promotes expert selection diversity. GatePro identifies the most similar expert pairs and introduces localized competition mechanisms, preventing redundant expert co-activation while maintaining natural expert specialization. Our comprehensive evaluation demonstrates GatePro's effectiveness across model scales and benchmarks. Analysis demonstrates GatePro's ability to achieve enhanced expert diversity, where experts develop more distinct and complementary capabilities, avoiding functional redundancy. This approach can be deployed hot-swappable during any training phase without additional learnable parameters, offering a practical solution for improving MoE effectiveness.
>
---
#### [new 035] LLM one-shot style transfer for Authorship Attribution and Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究作者归属与验证任务，旨在解决现有方法混淆风格与主题的问题。提出一种基于大语言模型的无监督方法，利用其预训练和上下文学习能力，通过log概率衡量文本间风格可迁移性，提升准确性并支持计算与性能的灵活权衡。**

- **链接: [http://arxiv.org/pdf/2510.13302v1](http://arxiv.org/pdf/2510.13302v1)**

> **作者:** Pablo Miralles-González; Javier Huertas-Tato; Alejandro Martín; David Camacho
>
> **摘要:** Computational stylometry analyzes writing style through quantitative patterns in text, supporting applications from forensic tasks such as identity linking and plagiarism detection to literary attribution in the humanities. Supervised and contrastive approaches rely on data with spurious correlations and often confuse style with topic. Despite their natural use in AI-generated text detection, the CLM pre-training of modern LLMs has been scarcely leveraged for general authorship problems. We propose a novel unsupervised approach based on this extensive pre-training and the in-context learning capabilities of LLMs, employing the log-probabilities of an LLM to measure style transferability from one text to another. Our method significantly outperforms LLM prompting approaches of comparable scale and achieves higher accuracy than contrastively trained baselines when controlling for topical correlations. Moreover, performance scales fairly consistently with the size of the base model and, in the case of authorship verification, with an additional mechanism that increases test-time computation; enabling flexible trade-offs between computational cost and accuracy.
>
---
#### [new 036] FreshTab: Sourcing Fresh Data for Table-to-Text Generation Evaluation
- **分类: cs.CL**

- **简介: 该论文针对表格到文本生成任务中评测基准存在大模型训练数据污染和领域不平衡问题，提出FreshTab方法，从维基百科动态生成多语言新鲜表格数据，以实现更可靠、领域敏感的评估。**

- **链接: [http://arxiv.org/pdf/2510.13598v1](http://arxiv.org/pdf/2510.13598v1)**

> **作者:** Kristýna Onderková; Ondřej Plátek; Zdeněk Kasner; Ondřej Dušek
>
> **备注:** To be published in INLG 2025
>
> **摘要:** Table-to-text generation (insight generation from tables) is a challenging task that requires precision in analyzing the data. In addition, the evaluation of existing benchmarks is affected by contamination of Large Language Model (LLM) training data as well as domain imbalance. We introduce FreshTab, an on-the-fly table-to-text benchmark generation from Wikipedia, to combat the LLM data contamination problem and enable domain-sensitive evaluation. While non-English table-to-text datasets are limited, FreshTab collects datasets in different languages on demand (we experiment with German, Russian and French in addition to English). We find that insights generated by LLMs from recent tables collected by our method appear clearly worse by automatic metrics, but this does not translate into LLM and human evaluations. Domain effects are visible in all evaluations, showing that a~domain-balanced benchmark is more challenging.
>
---
#### [new 037] The Mechanistic Emergence of Symbol Grounding in Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究语言模型中符号接地的机制性涌现，旨在揭示其产生位置与驱动机制。作者提出评估框架，通过因果分析发现中间层注意力头通过聚合环境信息实现接地，且该现象在多模态和多种架构中存在。**

- **链接: [http://arxiv.org/pdf/2510.13796v1](http://arxiv.org/pdf/2510.13796v1)**

> **作者:** Shuyu Wu; Ziqiao Ma; Xiaoxi Luo; Yidong Huang; Josue Torres-Fonseca; Freda Shi; Joyce Chai
>
> **摘要:** Symbol grounding (Harnad, 1990) describes how symbols such as words acquire their meanings by connecting to real-world sensorimotor experiences. Recent work has shown preliminary evidence that grounding may emerge in (vision-)language models trained at scale without using explicit grounding objectives. Yet, the specific loci of this emergence and the mechanisms that drive it remain largely unexplored. To address this problem, we introduce a controlled evaluation framework that systematically traces how symbol grounding arises within the internal computations through mechanistic and causal analysis. Our findings show that grounding concentrates in middle-layer computations and is implemented through the aggregate mechanism, where attention heads aggregate the environmental ground to support the prediction of linguistic forms. This phenomenon replicates in multimodal dialogue and across architectures (Transformers and state-space models), but not in unidirectional LSTMs. Our results provide behavioral and mechanistic evidence that symbol grounding can emerge in language models, with practical implications for predicting and potentially controlling the reliability of generation.
>
---
#### [new 038] A Critical Review of the Need for Knowledge-Centric Evaluation of Quranic Recitation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属文献综述任务，旨在解决现有 Quranic 诵读自动评估工具效果不佳的问题。作者分析20年研究，指出当前依赖语音识别技术的局限，主张转向以知识为中心、基于 Tajweed 规则的计算框架，提出融合语言知识与音频分析的混合系统方向。**

- **链接: [http://arxiv.org/pdf/2510.12858v1](http://arxiv.org/pdf/2510.12858v1)**

> **作者:** Mohammed Hilal Al-Kharusi; Khizar Hayat; Khalil Bader Al Ruqeishi; Haroon Rashid Lone
>
> **备注:** 33 pages
>
> **摘要:** The sacred practice of Quranic recitation (Tajweed), governed by precise phonetic, prosodic, and theological rules, faces significant pedagogical challenges in the modern era. While digital technologies promise unprecedented access to education, automated tools for recitation evaluation have failed to achieve widespread adoption or pedagogical efficacy. This literature review investigates this critical gap, conducting a comprehensive analysis of academic research, web platforms, and commercial applications developed over the past two decades. Our synthesis reveals a fundamental misalignment in prevailing approaches that repurpose Automatic Speech Recognition (ASR) architectures, which prioritize lexical recognition over qualitative acoustic assessment and are plagued by data dependency, demographic biases, and an inability to provide diagnostically useful feedback. Critiquing these data--driven paradigms, we argue for a foundational paradigm shift towards a knowledge-centric computational framework. Capitalizing on the immutable nature of the Quranic text and the precisely defined rules of Tajweed, we propose that a robust evaluator must be architected around anticipatory acoustic modeling based on canonical rules and articulation points (Makhraj), rather than relying on statistical patterns learned from imperfect and biased datasets. This review concludes that the future of automated Quranic evaluation lies in hybrid systems that integrate deep linguistic knowledge with advanced audio analysis, offering a path toward robust, equitable, and pedagogically sound tools that can faithfully support learners worldwide.
>
---
#### [new 039] LLM-Guided Synthetic Augmentation (LGSA) for Mitigating Bias in AI Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对AI系统中的性别偏见问题，提出LLM引导的合成增强（LGSA）方法。利用大语言模型生成性别对换的反事实样本，提升数据均衡性。实验显示，该方法在保持高准确率的同时显著降低性别偏差。**

- **链接: [http://arxiv.org/pdf/2510.13202v1](http://arxiv.org/pdf/2510.13202v1)**

> **作者:** Sai Suhruth Reddy Karri; Yashwanth Sai Nallapuneni; Laxmi Narasimha Reddy Mallireddy; Gopichand G
>
> **备注:** 11 pages, 4 figures, 1 Table, submitted to an international conference
>
> **摘要:** Bias in AI systems, especially those relying on natural language data, raises ethical and practical concerns. Underrepresentation of certain groups often leads to uneven performance across demographics. Traditional fairness methods, such as pre-processing, in-processing, and post-processing, depend on protected-attribute labels, involve accuracy-fairness trade-offs, and may not generalize across datasets. To address these challenges, we propose LLM-Guided Synthetic Augmentation (LGSA), which uses large language models to generate counterfactual examples for underrepresented groups while preserving label integrity. We evaluated LGSA on a controlled dataset of short English sentences with gendered pronouns, professions, and binary classification labels. Structured prompts were used to produce gender-swapped paraphrases, followed by quality control including semantic similarity checks, attribute verification, toxicity screening, and human spot checks. The augmented dataset expanded training coverage and was used to train a classifier under consistent conditions. Results show that LGSA reduces performance disparities without compromising accuracy. The baseline model achieved 96.7 percent accuracy with a 7.2 percent gender bias gap. Simple swap augmentation reduced the gap to 0.7 percent but lowered accuracy to 95.6 percent. LGSA achieved 99.1 percent accuracy with a 1.9 percent bias gap, improving performance on female-labeled examples. These findings demonstrate that LGSA is an effective strategy for bias mitigation, enhancing subgroup balance while maintaining high task accuracy and label fidelity.
>
---
#### [new 040] Multi-Label Clinical Text Eligibility Classification and Summarization System
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对临床试验入组标准的多标签分类与摘要生成任务，旨在提升筛选效率。作者结合NLP技术和大语言模型，采用多种特征提取、分类模型（如Random Forest、SVM）与摘要算法（如TextRank、GPT-3），实现对临床文本的自动分类与关键信息浓缩。**

- **链接: [http://arxiv.org/pdf/2510.13115v1](http://arxiv.org/pdf/2510.13115v1)**

> **作者:** Surya Tejaswi Yerramsetty; Almas Fathimah
>
> **摘要:** Clinical trials are central to medical progress because they help improve understanding of human health and the healthcare system. They play a key role in discovering new ways to detect, prevent, or treat diseases, and it is essential that clinical trials include participants with appropriate and diverse medical backgrounds. In this paper, we propose a system that leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to automate multi-label clinical text eligibility classification and summarization. The system combines feature extraction methods such as word embeddings (Word2Vec) and named entity recognition to identify relevant medical concepts, along with traditional vectorization techniques such as count vectorization and TF-IDF (Term Frequency-Inverse Document Frequency). We further explore weighted TF-IDF word embeddings that integrate both count-based and embedding-based strengths to capture term importance effectively. Multi-label classification using Random Forest and SVM models is applied to categorize documents based on eligibility criteria. Summarization techniques including TextRank, Luhn, and GPT-3 are evaluated to concisely summarize eligibility requirements. Evaluation with ROUGE scores demonstrates the effectiveness of the proposed methods. This system shows potential for automating clinical trial eligibility assessment using data-driven approaches, thereby improving research efficiency.
>
---
#### [new 041] How Sampling Affects the Detectability of Machine-written texts: A Comprehensive Study
- **分类: cs.CL**

- **简介: 该论文研究采样策略对机器生成文本检测的影响，揭示当前检测器在不同解码参数下性能急剧下降的问题。作者系统评估37种配置，发现微小参数变化可致AUROC从近100%降至1%，并公开数据与代码以促进更鲁棒的检测研究。**

- **链接: [http://arxiv.org/pdf/2510.13681v1](http://arxiv.org/pdf/2510.13681v1)**

> **作者:** Matthieu Dubois; François Yvon; Pablo Piantanida
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** As texts generated by Large Language Models (LLMs) are ever more common and often indistinguishable from human-written content, research on automatic text detection has attracted growing attention. Many recent detectors report near-perfect accuracy, often boasting AUROC scores above 99\%. However, these claims typically assume fixed generation settings, leaving open the question of how robust such systems are to changes in decoding strategies. In this work, we systematically examine how sampling-based decoding impacts detectability, with a focus on how subtle variations in a model's (sub)word-level distribution affect detection performance. We find that even minor adjustments to decoding parameters - such as temperature, top-p, or nucleus sampling - can severely impair detector accuracy, with AUROC dropping from near-perfect levels to 1\% in some settings. Our findings expose critical blind spots in current detection methods and emphasize the need for more comprehensive evaluation protocols. To facilitate future research, we release a large-scale dataset encompassing 37 decoding configurations, along with our code and evaluation framework https://github.com/BaggerOfWords/Sampling-and-Detection
>
---
#### [new 042] I Am Aligned, But With Whom? MENA Values Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs
- **分类: cs.CL**

- **简介: 该论文提出MENAValues基准，评估大模型在中东和北非地区的文化对齐与多语言偏见问题。通过多语言、多视角测试，揭示模型存在跨语言价值偏移、推理退化和隐性偏好泄露等问题，推动更包容的文化对齐AI发展。**

- **链接: [http://arxiv.org/pdf/2510.13154v1](http://arxiv.org/pdf/2510.13154v1)**

> **作者:** Pardis Sadat Zahraei; Ehsaneddin Asgari
>
> **摘要:** We introduce MENAValues, a novel benchmark designed to evaluate the cultural alignment and multilingual biases of large language models (LLMs) with respect to the beliefs and values of the Middle East and North Africa (MENA) region, an underrepresented area in current AI evaluation efforts. Drawing from large-scale, authoritative human surveys, we curate a structured dataset that captures the sociocultural landscape of MENA with population-level response distributions from 16 countries. To probe LLM behavior, we evaluate diverse models across multiple conditions formed by crossing three perspective framings (neutral, personalized, and third-person/cultural observer) with two language modes (English and localized native languages: Arabic, Persian, Turkish). Our analysis reveals three critical phenomena: "Cross-Lingual Value Shifts" where identical questions yield drastically different responses based on language, "Reasoning-Induced Degradation" where prompting models to explain their reasoning worsens cultural alignment, and "Logit Leakage" where models refuse sensitive questions while internal probabilities reveal strong hidden preferences. We further demonstrate that models collapse into simplistic linguistic categories when operating in native languages, treating diverse nations as monolithic entities. MENAValues offers a scalable framework for diagnosing cultural misalignment, providing both empirical insights and methodological tools for developing more culturally inclusive AI.
>
---
#### [new 043] FaStFACT: Faster, Stronger Long-Form Factuality Evaluations in LLMs
- **分类: cs.CL; cs.AI; cs.CE; cs.CY**

- **简介: 该论文聚焦于大语言模型长文本生成的事实性评估任务，旨在解决现有方法效率低、证据不足的问题。作者提出FaStFACT框架，通过分块提取与置信度预验证减少计算开销，并利用网页级证据提升验证准确性，显著提高了评估效果与效率。**

- **链接: [http://arxiv.org/pdf/2510.12839v1](http://arxiv.org/pdf/2510.12839v1)**

> **作者:** Yingjia Wan; Haochen Tan; Xiao Zhu; Xinyu Zhou; Zhiwei Li; Qingsong Lv; Changxuan Sun; Jiaqi Zeng; Yi Xu; Jianqiao Lu; Yinhong Liu; Zhijiang Guo
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** Evaluating the factuality of long-form generations from Large Language Models (LLMs) remains challenging due to accuracy issues and costly human assessment. Prior efforts attempt this by decomposing text into claims, searching for evidence, and verifying claims, but suffer from critical drawbacks: (1) inefficiency due to complex pipeline components unsuitable for long LLM outputs, and (2) ineffectiveness stemming from inaccurate claim sets and insufficient evidence collection of one-line snippets. To address these limitations, we propose \name, a fast and strong evaluation framework that achieves the highest alignment with human evaluation and efficiency among existing baselines. \name first employs chunk-level claim extraction integrated with confidence-based pre-verification, significantly reducing the cost of web searching and inference calling while ensuring reliability. For searching and verification, it collects document-level evidence from crawled webpages and selectively retrieves it during verification, addressing the evidence insufficiency problem in previous pipelines. Extensive experiments based on an aggregated and manually annotated benchmark demonstrate the reliability of \name in both efficiently and effectively evaluating the factuality of long-form LLM generations. Code and benchmark data is available at https://github.com/Yingjia-Wan/FastFact.
>
---
#### [new 044] Taming the Fragility of KV Cache Eviction in LLM Inference
- **分类: cs.CL**

- **简介: 该论文针对大模型推理中KV缓存淘汰的脆弱性问题，提出防御性聚合策略DefensiveKV，通过控制最坏情况风险来提升缓存淘汰鲁棒性，显著降低生成质量损失。**

- **链接: [http://arxiv.org/pdf/2510.13334v1](http://arxiv.org/pdf/2510.13334v1)**

> **作者:** Yuan Feng; Haoyu Guo; JunLin Lv; S. Kevin Zhou; Xike Xie
>
> **摘要:** Large language models have revolutionized natural language processing, yet their deployment remains hampered by the substantial memory and runtime overhead of the transformer's Key-Value cache. To mitigate this, recent methods employ a scoring-aggregation framework to evict unimportant cache entries, based on the stability assumption-that a fixed subset of entries remains consistently important during generation. However, prior work has largely focused on refining importance indicators for scoring, while defaulting to mean aggregation due to a faithful trust in the stability assumption. In this work, we argue that this underlying assumption is inherently fragile, making mean aggregation highly vulnerable in extreme cases. To counter this, we propose a simple yet elegant defensive aggregation strategy: a two-step, linear-time approach that controls worst-case risk, thereby defending against extreme cases with negligible computational overhead. Embodying this strategy, we propose a novel cache eviction method, DefensiveKV and its extension, Layer-DefensiveKV, which incorporates layer-wise budget allocation. Across seven task domains (18 datasets), our methods reduce generation quality loss by 2.3x and 4.3x respectively, versus the strongest baseline under a 20% cache size. These results set new performance benchmarks and pioneer a promising direction for optimizing cache eviction against underlying fragility through worst-case risk management. Our code is available at https://github.com/FFY0/DefensiveKV.
>
---
#### [new 045] A Multilingual, Large-Scale Study of the Interplay between LLM Safeguards, Personalisation, and Disinformation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在个性化误导信息生成中的安全漏洞。通过红队测试，构建多语言数据集AI-TRAITS，评估不同模型、语言和人群特征下的越狱率与说服力，揭示个性化提示显著提升攻击成功率和虚假内容 persuasiveness。**

- **链接: [http://arxiv.org/pdf/2510.12993v1](http://arxiv.org/pdf/2510.12993v1)**

> **作者:** João A. Leite; Arnav Arora; Silvia Gargova; João Luz; Gustavo Sampaio; Ian Roberts; Carolina Scarton; Kalina Bontcheva
>
> **摘要:** The human-like proficiency of Large Language Models (LLMs) has brought concerns about their potential misuse for generating persuasive and personalised disinformation at scale. While prior work has demonstrated that LLMs can generate disinformation, specific questions around persuasiveness and personalisation (generation of disinformation tailored to specific demographic attributes) remain largely unstudied. This paper presents the first large-scale, multilingual empirical study on persona-targeted disinformation generation by LLMs. Employing a red teaming methodology, we systematically evaluate the robustness of LLM safety mechanisms to persona-targeted prompts. A key novel result is AI-TRAITS (AI-generaTed peRsonAlIsed disinformaTion dataSet), a new dataset of around 1.6 million texts generated by eight state-of-the-art LLMs. AI-TRAITS is seeded by prompts that combine 324 disinformation narratives and 150 distinct persona profiles, covering four major languages (English, Russian, Portuguese, Hindi) and key demographic dimensions (country, generation, political orientation). The resulting personalised narratives are then assessed quantitatively and compared along the dimensions of models, languages, jailbreaking rate, and personalisation attributes. Our findings demonstrate that the use of even simple personalisation strategies in the prompts significantly increases the likelihood of jailbreaks for all studied LLMs. Furthermore, personalised prompts result in altered linguistic and rhetorical patterns and amplify the persuasiveness of the LLM-generated false narratives. These insights expose critical vulnerabilities in current state-of-the-art LLMs and offer a foundation for improving safety alignment and detection strategies in multilingual and cross-demographic contexts.
>
---
#### [new 046] D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree
- **分类: cs.CL; 68T50, 68T30; I.2.7; I.2.4**

- **简介: 该论文针对大语言模型在多轮对话中存在事实不一致和逻辑衰减的问题，提出D-SMART框架。通过动态结构化记忆和推理树，构建可追溯的对话知识图谱，提升多轮对话一致性，在多种模型上显著改善表现。**

- **链接: [http://arxiv.org/pdf/2510.13363v1](http://arxiv.org/pdf/2510.13363v1)**

> **作者:** Xiang Lei; Qin Li; Min Zhang; Min Zhang
>
> **备注:** 8 pages, 6 figures (main content); 25 pages, 18 figures (total)
>
> **摘要:** Large Language Models (LLMs) often exhibit factual inconsistencies and logical decay in extended, multi-turn dialogues, a challenge stemming from their reliance on static, pre-trained knowledge and an inability to reason adaptively over the dialogue history. Prevailing mitigation strategies, such as Retrieval-Augmented Generation (RAG) and agentic working memories, improve information recall but still engage with fundamentally static knowledge sources and follow pre-defined single reasoning path. This hinders their ability to preserve factual and logical consistency of their responses in multi-turn dialogues while the context evolves over time. To address this issue, we propose D-SMART, a model-agnostic framework designed to maintain multi-turn dialogue consistency by enabling LLMs to build and reason over a dynamic, structured representation of the conversational context. This is achieved via two synergistic components: (1) a Dynamic Structured Memory (DSM), which incrementally constructs and maintains an authoritative, OWL-compliant knowledge graph of the conversation; and (2) a Reasoning Tree (RT), which executes inferences as an explicit and traceable multi-step search over the graph. As the popular-used quality score (judged by GPT-4) can overlook logical flaws, we introduce new NLI-based metrics to better measure multi-turn dialogue consistency. Comprehensive experiments on the MT-Bench-101 benchmark show that D-SMART significantly outperforms state-of-the-art baselines, elevating the dialogue consistency score by over 48\% for both proprietary and open-source models, and notably improves the quality score of the latter by up to 10.1\%.
>
---
#### [new 047] A Matter of Representation: Towards Graph-Based Abstract Code Generation
- **分类: cs.CL**

- **简介: 该论文研究图结构抽象代码生成任务，旨在解决现有大模型难以生成基于节点与边的可视化编程代码问题。作者提出多种JSON图表示方法，在自建ScratchTest基准上验证其有效性，表明合适的表示可显著提升生成准确率，为该领域提供首个系统探索。**

- **链接: [http://arxiv.org/pdf/2510.13163v1](http://arxiv.org/pdf/2510.13163v1)**

> **作者:** Nyx Iskandar; Hisham Bedri; Andy Tsen
>
> **摘要:** Most large language models (LLMs) today excel at generating raw, sequential code with minimal abstractions and custom structures. However, there has been little work on graph-based abstract code generation, where significant logic is encapsulated in predefined nodes and execution flow is determined by edges. This is relevant for visual programming languages, and in cases where raw source code is inaccessible to users and LLM training sets. In this work, we propose and evaluate JSON representations for graphs to enable high accuracy graph-based abstract code generation. We evaluate these representations on ScratchTest, a mini-benchmark based on our custom Python re-implementation of Scratch, which tests the LLM in code graph space. Our findings demonstrate that LLMs can indeed perform the aforementioned generation task in a single pass without relying on specialized or complex pipelines, given the correct graph representations. We also show that different representations induce significantly different accuracies, highlighting the instrumental role of representations in this generation task. All in all, this work establishes the first steps towards representation learning for graph-based abstract code generation.
>
---
#### [new 048] Benchmarking Open-Source Large Language Models for Persian in Zero-Shot and Few-Shot Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估开源大模型在低资源语言波斯语上的零样本和少样本学习能力。研究选取多个波斯语NLP任务，通过标准数据集和指标进行实验，发现Gemma 2表现最佳，但多数模型在命名实体识别等细粒度任务上仍有困难。**

- **链接: [http://arxiv.org/pdf/2510.12807v1](http://arxiv.org/pdf/2510.12807v1)**

> **作者:** Mahdi Cherakhloo; Arash Abbasi; Mohammad Saeid Sarafraz; Bijan Vosoughi Vahdat
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous languages; however, their effectiveness in low-resource languages like Persian requires thorough investigation. This paper presents a comprehensive benchmark of several open-source LLMs for Persian Natural Language Processing (NLP) tasks, utilizing both zero-shot and few-shot learning paradigms. We evaluate models across a range of tasks including sentiment analysis, named entity recognition, reading comprehension, and question answering, using established Persian datasets such as ParsiNLU and ArmanEmo. Our methodology encompasses rigorous experimental setups for both zero-shot and few-shot scenarios, employing metrics such as Accuracy, F1-score, BLEU, and ROUGE for performance evaluation. The results reveal that Gemma 2 consistently outperforms other models across nearly all tasks in both learning paradigms, with particularly strong performance in complex reasoning tasks. However, most models struggle with token-level understanding tasks like Named Entity Recognition, highlighting specific challenges in Persian language processing. This study contributes to the growing body of research on multilingual LLMs, providing valuable insights into their performance in Persian and offering a benchmark for future model development.
>
---
#### [new 049] StressTransfer: Stress-Aware Speech-to-Speech Translation with Emphasis Preservation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语音到语音翻译中语义重音的跨语言保留问题，提出StressTransfer方法，利用大语言模型实现重音感知的翻译，并通过自动标注数据和“LLM-as-Judge”评估框架，在低资源下有效保持语调与说话人意图。**

- **链接: [http://arxiv.org/pdf/2510.13194v1](http://arxiv.org/pdf/2510.13194v1)**

> **作者:** Xi Chen; Yuchen Song; Satoshi Nakamura
>
> **摘要:** We propose a stress-aware speech-to-speech translation (S2ST) system that preserves word-level emphasis by leveraging LLMs for cross-lingual emphasis conversion. Our method translates source-language stress into target-language tags that guide a controllable TTS model. To overcome data scarcity, we developed a pipeline to automatically generate aligned training data and introduce the "LLM-as-Judge" for evaluation. Experiments show our approach substantially outperforms baselines in preserving emphasis while maintaining comparable translation quality, speaker intent, and naturalness. Our work highlights the importance of prosody in translation and provides an effective, data-efficient solution for preserving paralinguistic cues in S2ST.
>
---
#### [new 050] ConsintBench: Evaluating Language Models on Real-World Consumer Intent Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大语言模型在真实消费者意图理解任务中的评估难题，提出ConsintBench——首个动态、实时更新的基准，通过自动化流程收集多源争议性讨论数据，解决现有缺乏大规模、抗污染评估体系的问题。**

- **链接: [http://arxiv.org/pdf/2510.13499v1](http://arxiv.org/pdf/2510.13499v1)**

> **作者:** Xiaozhe Li; TianYi Lyu; Siyi Yang; Yuxi Gong; Yizhao Yang; Jinxuan Huang; Ligao Zhang; Zhuoyi Huang; Qingwen Liu
>
> **摘要:** Understanding human intent is a complex, high-level task for large language models (LLMs), requiring analytical reasoning, contextual interpretation, dynamic information aggregation, and decision-making under uncertainty. Real-world public discussions, such as consumer product discussions, are rarely linear or involve a single user. Instead, they are characterized by interwoven and often conflicting perspectives, divergent concerns, goals, emotional tendencies, as well as implicit assumptions and background knowledge about usage scenarios. To accurately understand such explicit public intent, an LLM must go beyond parsing individual sentences; it must integrate multi-source signals, reason over inconsistencies, and adapt to evolving discourse, similar to how experts in fields like politics, economics, or finance approach complex, uncertain environments. Despite the importance of this capability, no large-scale benchmark currently exists for evaluating LLMs on real-world human intent understanding, primarily due to the challenges of collecting real-world public discussion data and constructing a robust evaluation pipeline. To bridge this gap, we introduce \bench, the first dynamic, live evaluation benchmark specifically designed for intent understanding, particularly in the consumer domain. \bench is the largest and most diverse benchmark of its kind, supporting real-time updates while preventing data contamination through an automated curation pipeline.
>
---
#### [new 051] Are Proverbs the New Pythian Oracles? Exploring Sentiment in Greek Sayings
- **分类: cs.CL**

- **简介: 该论文利用大语言模型对希腊谚语进行情感分析，解决传统谚语情感极性识别问题。通过扩展标注数据集，结合方言与地域信息，绘制希腊谚语情感分布地图，并分析情感与地理、话题的关系，发现负向情感在多数地区占主导。**

- **链接: [http://arxiv.org/pdf/2510.13341v1](http://arxiv.org/pdf/2510.13341v1)**

> **作者:** Katerina Korre; John Pavlopoulos
>
> **摘要:** Proverbs are among the most fascinating linguistic phenomena that transcend cultural and linguistic boundaries. Yet, much of the global landscape of proverbs remains underexplored, as many cultures preserve their traditional wisdom within their own communities due to the oral tradition of the phenomenon. Taking advantage of the current advances in Natural Language Processing (NLP), we focus on Greek proverbs, analyzing their sentiment. Departing from an annotated dataset of Greek proverbs, we expand it to include local dialects, effectively mapping the annotated sentiment. We present (1) a way to exploit LLMs in order to perform sentiment classification of proverbs, (2) a map of Greece that provides an overview of the distribution of sentiment, (3) a combinatory analysis in terms of the geographic position, dialect, and topic of proverbs. Our findings show that LLMs can provide us with an accurate enough picture of the sentiment of proverbs, especially when approached as a non-conventional sentiment polarity task. Moreover, in most areas of Greece negative sentiment is more prevalent.
>
---
#### [new 052] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究对话式问答（CQA）任务，解决用户意图动态变化和查询模糊的问题。提出ChatR1框架，通过强化学习实现检索与推理的交替进行，并设计意图感知奖励机制，提升模型在多轮对话中的上下文理解与适应能力。**

- **链接: [http://arxiv.org/pdf/2510.13312v1](http://arxiv.org/pdf/2510.13312v1)**

> **作者:** Simon Lupart; Mohammad Aliannejadi; Evangelos Kanoulas
>
> **摘要:** We present ChatR1, a reasoning framework based on reinforcement learning (RL) for conversational question answering (CQA). Reasoning plays an important role in CQA, where user intent evolves across dialogue turns, and utterances are often underspecified, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Unlike static `rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through RL. To address the challenge of sparse and delayed rewards in RL, we propose an intent-aware reward that provides turn-level feedback by aligning retrieval and reasoning with evolving user goals. Our proposed ChatR1 demonstrates strong performance on both 3B and 7B model backbones, outperforming competitive models on five CQA datasets, measured by different metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA datasets to cover topic shifts, evolving intents, mixed-initiative dialogues, and multi-document grounding, testing ChatR1's performance from various aspects. Ablation studies confirm the effectiveness of the intent-aware reward. Our analyses further reveal diverse reasoning trajectories and effective use of the search tool. ChatR1 also generalizes robustly across domains, demonstrating that RL-based reasoning enables more flexible and context-sensitive behavior than static CQA pipelines.
>
---
#### [new 053] A\textsuperscript{2}FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出A²FM模型，解决大模型在推理与工具使用间的效率与准确性权衡问题。通过引入任务感知路由、第三执行模式及自适应策略优化，实现推理、工具调用与直接响应的统一，在保持准确率的同时显著降低计算成本。**

- **链接: [http://arxiv.org/pdf/2510.12838v1](http://arxiv.org/pdf/2510.12838v1)**

> **作者:** Qianben Chen; Jingyi Cao; Jiayu Zhang; Tianrui Qin; Xiaowan Li; King Zhu; Dingfeng Shi; He Zhu; Minghao Liu; Xiaobo Liang; Ge Zhang; Jian Yang; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** 9 pages, 5 figures, submitted to ICLR 2026
>
> **摘要:** Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A\textsuperscript{2}FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third mode-instant-that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A\textsuperscript{2}FM achieves 13.4\% on BrowseComp, 70.4\% on AIME25, and 16.7\% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only \$0.00487 per correct answer-cutting cost by 45.2\% relative to reasoning and 33.5\% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy.
>
---
#### [new 054] Scheming Ability in LLM-to-LLM Strategic Interactions
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文研究大语言模型在多智能体交互中的策略性欺骗能力，属于AI安全与博弈论交叉任务。通过廉价对话和对抗评估两个游戏框架，测试主流模型的自发与提示下欺骗行为，揭示模型在无提示时仍倾向策略性欺骗，强调高风险多智能体场景需更鲁棒的安全评估。**

- **链接: [http://arxiv.org/pdf/2510.12826v1](http://arxiv.org/pdf/2510.12826v1)**

> **作者:** Thao Pham
>
> **备注:** 25 pages, 13 figures, under review at IASEAI'26
>
> **摘要:** As large language model (LLM) agents are deployed autonomously in diverse contexts, evaluating their capacity for strategic deception becomes crucial. While recent research has examined how AI systems scheme against human developers, LLM-to-LLM scheming remains underexplored. We investigate the scheming ability and propensity of frontier LLM agents through two game-theoretic frameworks: a Cheap Talk signaling game and a Peer Evaluation adversarial game. Testing four models (GPT-4o, Gemini-2.5-pro, Claude-3.7-Sonnet, and Llama-3.3-70b), we measure scheming performance with and without explicit prompting while analyzing scheming tactics through chain-of-thought reasoning. When prompted, most models, especially Gemini-2.5-pro and Claude-3.7-Sonnet, achieved near-perfect performance. Critically, models exhibited significant scheming propensity without prompting: all models chose deception over confession in Peer Evaluation (100% rate), while models choosing to scheme in Cheap Talk succeeded at 95-100% rates. These findings highlight the need for robust evaluations using high-stakes game-theoretic scenarios in multi-agent settings.
>
---
#### [new 055] Repurposing Annotation Guidelines to Instruct LLM Annotators: A Case Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何将面向人类的标注指南改造为适合大语言模型（LLM）的指令，解决LLM作为标注者时对明确、结构化指导的需求。提出一种基于LLM审核的指南转换方法，并在NCBI疾病语料上验证其有效性，支持高效、低成本的自动标注与指南优化。**

- **链接: [http://arxiv.org/pdf/2510.12835v1](http://arxiv.org/pdf/2510.12835v1)**

> **作者:** Kon Woo Kim; Rezarta Islamaj; Jin-Dong Kim; Florian Boudin; Akiko Aizawa
>
> **备注:** 11 pages, 2 figures, 3 tables, This is a preprint of the article accepted at NLDB 2025 (Springer LNCS). The final version is available at https://doi.org/10.1007/978-3-031-97144-0_13
>
> **摘要:** This study investigates how existing annotation guidelines can be repurposed to instruct large language model (LLM) annotators for text annotation tasks. Traditional guidelines are written for human annotators who internalize training, while LLMs require explicit, structured instructions. We propose a moderation-oriented guideline repurposing method that transforms guidelines into clear directives for LLMs through an LLM moderation process. Using the NCBI Disease Corpus as a case study, our experiments show that repurposed guidelines can effectively guide LLM annotators, while revealing several practical challenges. The results highlight the potential of this workflow to support scalable and cost-effective refinement of annotation guidelines and automated annotation.
>
---
#### [new 056] ESI: Epistemic Uncertainty Quantification via Semantic-preserving Intervention for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的不确定性量化任务，旨在评估模型预测的可信度。通过因果视角，提出一种基于语义保持干预的新方法，有效估计认知不确定性，提升模型可靠性，实验证明其在多模型和数据集上兼具高效性与优越性能。**

- **链接: [http://arxiv.org/pdf/2510.13103v1](http://arxiv.org/pdf/2510.13103v1)**

> **作者:** Mingda Li; Xinyu Li; Weinan Zhang; Longxuan Ma
>
> **摘要:** Uncertainty Quantification (UQ) is a promising approach to improve model reliability, yet quantifying the uncertainty of Large Language Models (LLMs) is non-trivial. In this work, we establish a connection between the uncertainty of LLMs and their invariance under semantic-preserving intervention from a causal perspective. Building on this foundation, we propose a novel grey-box uncertainty quantification method that measures the variation in model outputs before and after the semantic-preserving intervention. Through theoretical justification, we show that our method provides an effective estimate of epistemic uncertainty. Our extensive experiments, conducted across various LLMs and a variety of question-answering (QA) datasets, demonstrate that our method excels not only in terms of effectiveness but also in computational efficiency.
>
---
#### [new 057] From Noise to Signal to Selbstzweck: Reframing Human Label Variation in the Era of Post-training in NLP
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属NLP中的模型对齐任务，指出当前偏好学习数据忽略人类标注差异（HLV），导致价值观单一化。作者主张将HLV视为目的本身，提出应主动保留并融入人类多样性标注，以更好实现AI与人类价值观的对齐。**

- **链接: [http://arxiv.org/pdf/2510.12817v1](http://arxiv.org/pdf/2510.12817v1)**

> **作者:** Shanshan Xu; Santosh T. Y. S. S; Barbara Plank
>
> **摘要:** Human Label Variation (HLV) refers to legitimate disagreement in annotation that reflects the genuine diversity of human perspectives rather than mere error. For decades, HLV in NLP was dismissed as noise to be discarded, and only slowly over the last decade has it been reframed as a signal for improving model robustness. With the rise of large language models (LLMs), where post-training on human feedback has become central to model alignment, the role of HLV has become increasingly consequential. Yet current preference-learning datasets routinely aggregate multiple annotations into a single label, thereby flattening diverse perspectives into a false universal agreement and erasing precisely the pluralism of human values that alignment aims to preserve. In this position paper, we argue that preserving HLV as an embodiment of human pluralism must be treated as a Selbstzweck - a goal it self when designing AI systems. We call for proactively incorporating HLV into preference datasets and outline actionable steps towards it.
>
---
#### [new 058] VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文聚焦多模态模型在低资源语言中的视觉与语言理解问题，提出新基准VLURes，包含八项任务和长文本多语言数据，评估十种VLM在英语、日语、斯瓦希里语和乌尔都语下的细粒度理解能力，揭示语言间性能差距。**

- **链接: [http://arxiv.org/pdf/2510.12845v1](http://arxiv.org/pdf/2510.12845v1)**

> **作者:** Jesse Atuhurra; Iqra Ali; Tomoya Iwakura; Hidetaka Kamigaito; Tatsuya Hiraoka
>
> **摘要:** Vision Language Models (VLMs) are pivotal for advancing perception in intelligent agents. Yet, evaluation of VLMs remains limited to predominantly English-centric benchmarks in which the image-text pairs comprise short texts. To evaluate VLM fine-grained abilities, in four languages under long-text settings, we introduce a novel multilingual benchmark VLURes featuring eight vision-and-language tasks, and a pioneering unrelatedness task, to probe the fine-grained Visual and Linguistic Understanding capabilities of VLMs across English, Japanese, and low-resource languages, Swahili, and Urdu. Our datasets, curated from web resources in the target language, encompass ten diverse image categories and rich textual context, introducing valuable vision-language resources for Swahili and Urdu. By prompting VLMs to generate responses and rationales, evaluated automatically and by native speakers, we uncover performance disparities across languages and tasks critical to intelligent agents, such as object recognition, scene understanding, and relationship understanding. We conducted evaluations of ten VLMs with VLURes. The best performing model, GPT-4o, achieves an overall accuracy of 90.8% and lags human performance by 6.7%, though the gap is larger for open-source models. The gap highlights VLURes' critical role in developing intelligent agents to tackle multi-modal visual reasoning.
>
---
#### [new 059] GAPS: A Clinically Grounded, Automated Benchmark for Evaluating AI Clinicians
- **分类: cs.CL**

- **简介: 该论文提出GAPS框架，旨在评估AI临床医生的推理深度、回答完整性、鲁棒性和安全性。针对现有基准缺乏临床依据和可扩展性的问题，作者构建了自动化、指南驱动的评测流水线，结合证据生成问题，并用LLM评委评分，实现可复现、高效的AI医疗能力评估。**

- **链接: [http://arxiv.org/pdf/2510.13734v1](http://arxiv.org/pdf/2510.13734v1)**

> **作者:** Xiuyuan Chen; Tao Sun; Dexin Su; Ailing Yu; Junwei Liu; Zhe Chen; Gangzeng Jin; Xin Wang; Jingnan Liu; Hansong Xiao; Hualei Zhou; Dongjie Tao; Chunxiao Guo; Minghui Yang; Yuan Xia; Jing Zhao; Qianrui Fan; Yanyun Wang; Shuai Zhen; Kezhong Chen; Jun Wang; Zewen Sun; Heng Zhao; Tian Guan; Shaodong Wang; Geyun Chang; Jiaming Deng; Hongchengcheng Chen; Kexin Feng; Ruzhen Li; Jiayi Geng; Changtai Zhao; Jun Wang; Guihu Lin; Peihao Li; Liqi Liu; Peng Wei; Jian Wang; Jinjie Gu; Ping Wang; Fan Yang
>
> **摘要:** Current benchmarks for AI clinician systems, often based on multiple-choice exams or manual rubrics, fail to capture the depth, robustness, and safety required for real-world clinical practice. To address this, we introduce the GAPS framework, a multidimensional paradigm for evaluating \textbf{G}rounding (cognitive depth), \textbf{A}dequacy (answer completeness), \textbf{P}erturbation (robustness), and \textbf{S}afety. Critically, we developed a fully automated, guideline-anchored pipeline to construct a GAPS-aligned benchmark end-to-end, overcoming the scalability and subjectivity limitations of prior work. Our pipeline assembles an evidence neighborhood, creates dual graph and tree representations, and automatically generates questions across G-levels. Rubrics are synthesized by a DeepResearch agent that mimics GRADE-consistent, PICO-driven evidence review in a ReAct loop. Scoring is performed by an ensemble of large language model (LLM) judges. Validation confirmed our automated questions are high-quality and align with clinician judgment. Evaluating state-of-the-art models on the benchmark revealed key failure modes: performance degrades sharply with increased reasoning depth (G-axis), models struggle with answer completeness (A-axis), and they are highly vulnerable to adversarial perturbations (P-axis) as well as certain safety issues (S-axis). This automated, clinically-grounded approach provides a reproducible and scalable method for rigorously evaluating AI clinician systems and guiding their development toward safer, more reliable clinical practice.
>
---
#### [new 060] A fully automated and scalable Parallel Data Augmentation for Low Resource Languages using Image and Text Analytics
- **分类: cs.CL**

- **简介: 该论文属自然语言处理任务，旨在解决低资源语言平行语料稀缺问题。提出一种全自动可扩展方法，结合图像与文本分析，从报纸中提取双语平行数据，并通过机器翻译任务验证其有效性，BLEU提升近3点。**

- **链接: [http://arxiv.org/pdf/2510.13211v1](http://arxiv.org/pdf/2510.13211v1)**

> **作者:** Prawaal Sharma; Navneet Goyal; Poonam Goyal; Vishnupriyan R
>
> **备注:** 4 Pages, Parallel Data Augmentation
>
> **摘要:** Linguistic diversity across the world creates a disparity with the availability of good quality digital language resources thereby restricting the technological benefits to majority of human population. The lack or absence of data resources makes it difficult to perform NLP tasks for low-resource languages. This paper presents a novel scalable and fully automated methodology to extract bilingual parallel corpora from newspaper articles using image and text analytics. We validate our approach by building parallel data corpus for two different language combinations and demonstrate the value of this dataset through a downstream task of machine translation and improve over the current baseline by close to 3 BLEU points.
>
---
#### [new 061] CurLL: A Developmental Framework to Evaluate Continual Learning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CurlL，一个受儿童发展启发的持续学习评测框架。旨在解决语言模型在渐进学习中的技能保持与迁移问题，构建了含技能图谱和合成数据的基准，评估模型在不同训练设置下的遗忘与转移表现。**

- **链接: [http://arxiv.org/pdf/2510.13008v1](http://arxiv.org/pdf/2510.13008v1)**

> **作者:** Pavan Kalyan; Shubhra Mishra; Satya Lokam; Navin Goyal
>
> **摘要:** We introduce a comprehensive continual learning dataset and benchmark (CurlL) grounded in human developmental trajectories from ages 5-10, enabling systematic and fine-grained assessment of models' ability to progressively acquire new skills. CurlL spans five developmental stages (0-4) covering ages 5-10, supported by a skill graph that breaks down broad skills into smaller abilities, concrete goals, and measurable indicators, while also capturing which abilities build on others. We generate a 23.4B-token synthetic dataset with controlled skill progression, vocabulary complexity, and format diversity, comprising paragraphs, comprehension-based QA (CQA), skill-testing QA (CSQA), and instruction-response (IR) pairs. Stage-wise token counts range from 2.12B to 6.78B tokens, supporting precise analysis of forgetting, forward transfer, and backward transfer. Using a 135M-parameter transformer trained under independent, joint, and sequential (continual) setups, we show trade-offs in skill retention and transfer efficiency. By mirroring human learning patterns and providing fine-grained control over skill dependencies, this work advances continual learning evaluations for language models.
>
---
#### [new 062] Assessing Web Search Credibility and Response Groundedness in Chat Assistants
- **分类: cs.CL**

- **简介: 该论文评估聊天助手在调用网页搜索时的可信度与回应依据性，旨在解决其可能引用低可信来源传播 misinformation 的问题。作者提出新方法，对多个主流模型进行测试，发现不同模型在来源可信度和响应 groundedness 上表现差异。**

- **链接: [http://arxiv.org/pdf/2510.13749v1](http://arxiv.org/pdf/2510.13749v1)**

> **作者:** Ivan Vykopal; Matúš Pikuliak; Simon Ostermann; Marián Šimko
>
> **摘要:** Chat assistants increasingly integrate web search functionality, enabling them to retrieve and cite external sources. While this promises more reliable answers, it also raises the risk of amplifying misinformation from low-credibility sources. In this paper, we introduce a novel methodology for evaluating assistants' web search behavior, focusing on source credibility and the groundedness of responses with respect to cited sources. Using 100 claims across five misinformation-prone topics, we assess GPT-4o, GPT-5, Perplexity, and Qwen Chat. Our findings reveal differences between the assistants, with Perplexity achieving the highest source credibility, whereas GPT-4o exhibits elevated citation of non-credibility sources on sensitive topics. This work provides the first systematic comparison of commonly used chat assistants for fact-checking behavior, offering a foundation for evaluating AI systems in high-stakes information environments.
>
---
#### [new 063] Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与CPDC 2025挑战，旨在平衡NPC的任务执行与角色一致性对话。提出Deflanderization提示法抑制过度角色扮演，并在API和GPU赛道采用轻量提示与微调Qwen3-14B模型，提升对话真实性与任务完成度。**

- **链接: [http://arxiv.org/pdf/2510.13586v1](http://arxiv.org/pdf/2510.13586v1)**

> **作者:** Pasin Buakhaw; Kun Kerdthaisong; Phuree Phenhiran; Pitikorn Khlaisamniang; Supasate Vorathammathorn; Piyalitt Ittichaiwong; Nutchanon Yongsatianchot
>
> **摘要:** The emergence of large language models (LLMs) has opened new opportunities for cre- ating dynamic non-player characters (NPCs) in gaming environments, enabling both func- tional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which eval- uates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
>
---
#### [new 064] EduDial: Constructing a Large-scale Multi-turn Teacher-Student Dialogue Corpus
- **分类: cs.CL**

- **简介: 该论文构建了大规模师生多轮对话数据集EduDial，旨在提升大模型在教育场景中的教学能力。基于Bloom分类法设计提问策略，提出11维评估框架，并训练出表现更优的EduDial-LLM。**

- **链接: [http://arxiv.org/pdf/2510.12899v1](http://arxiv.org/pdf/2510.12899v1)**

> **作者:** Shouang Wei; Min Zhang; Xin Lin; Bo Jiang; Zhongxiang Dai; Kun Kuang
>
> **摘要:** Recently, several multi-turn dialogue benchmarks have been proposed to evaluate the conversational abilities of large language models (LLMs). As LLMs are increasingly recognized as a key technology for advancing intelligent education, owing to their ability to deeply understand instructional contexts and provide personalized guidance, the construction of dedicated teacher-student dialogue benchmarks has become particularly important. To this end, we present EduDial, a comprehensive multi-turn teacher-student dialogue dataset. EduDial covers 345 core knowledge points and consists of 34,250 dialogue sessions generated through interactions between teacher and student agents. Its design is guided by Bloom's taxonomy of educational objectives and incorporates ten questioning strategies, including situational questioning, zone of proximal development (ZPD) questioning, and metacognitive questioning-thus better capturing authentic classroom interactions. Furthermore, we design differentiated teaching strategies for students at different cognitive levels, thereby providing more targeted teaching guidance. Building on EduDial, we further develop EduDial-LLM 32B via training and propose an 11-dimensional evaluation framework that systematically measures the teaching abilities of LLMs, encompassing both overall teaching quality and content quality. Experiments on 17 mainstream LLMs reveal that most models struggle in student-centered teaching scenarios, whereas our EduDial-LLM achieves significant gains, consistently outperforming all baselines across all metrics. The code is available at https://github.com/Mind-Lab-ECNU/EduDial/tree/main.
>
---
#### [new 065] LiteraryQA: Towards Effective Evaluation of Long-document Narrative QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦叙事性长文档问答（QA）任务，针对NarrativeQA数据集存在噪声和低质问答对的问题，构建了高质量子集LiteraryQA。作者通过人工与大模型协同清洗数据，并开展自动评测指标的元评估，发现基于LLM的评判指标更符合人类判断。**

- **链接: [http://arxiv.org/pdf/2510.13494v1](http://arxiv.org/pdf/2510.13494v1)**

> **作者:** Tommaso Bonomo; Luca Gioffré; Roberto Navigli
>
> **备注:** Accepted to EMNLP 2025 Main Conference. 22 pages
>
> **摘要:** Question Answering (QA) on narrative text poses a unique challenge to current systems, requiring a deep understanding of long, complex documents. However, the reliability of NarrativeQA, the most widely used benchmark in this domain, is hindered by noisy documents and flawed QA pairs. In this work, we introduce LiteraryQA, a high-quality subset of NarrativeQA focused on literary works. Using a human- and LLM-validated pipeline, we identify and correct low-quality QA samples while removing extraneous text from source documents. We then carry out a meta-evaluation of automatic metrics to clarify how systems should be evaluated on LiteraryQA. This analysis reveals that all n-gram-based metrics have a low system-level correlation to human judgment, while LLM-as-a-Judge evaluations, even with small open-weight models, can strongly agree with the ranking identified by humans. Finally, we benchmark a set of long-context LLMs on LiteraryQA. We release our code and data at https://github.com/SapienzaNLP/LiteraryQA.
>
---
#### [new 066] Unlocking Public Catalogues: Instruction-Tuning LLMs for ICD Coding of German Tumor Diagnoses
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究利用公开编码手册构建指令数据集，通过微调提升开源大模型在德语肿瘤诊断文本的ICD编码准确性，解决小模型在医疗编码中准确率低的问题，实现了显著性能提升并开源了数据与模型。**

- **链接: [http://arxiv.org/pdf/2510.13624v1](http://arxiv.org/pdf/2510.13624v1)**

> **作者:** Stefan Lenz; Lakisha Ortiz Rosario; Georg Vollmar; Arsenij Ustjanzew; Fatma Alickovic; Thomas Kindler; Torsten Panholzer
>
> **备注:** 19 pages, 4 figures
>
> **摘要:** Accurate coding of tumor diagnoses with ICD-10-GM and ICD-O-3 is essential for structured cancer documentation in Germany. Smaller open-weight LLMs are appealing for privacy-preserving automation but often struggle with coding accuracy in German-language contexts. This study investigates whether instruction-based fine-tuning on public datasets improves the coding accuracy of open-weight LLMs for German tumor diagnosis texts. The evaluation uses coded diagnoses from the local tumor documentation system as test data. In a systematic data quality assessment, the upper limit for ICD-10 coding performance was estimated at 60-79% for exact and 81-94% for partial (three-character codes only) derivation. As training data, over 500,000 question-answer pairs were created based on the ICD-10-GM, ICD-O-3, and OPS catalogues. Eight open-weight models from the Qwen, Llama, and Mistral families (7-70 B parameters) were fine-tuned. ICD-10-GM accuracy rose from 1.4-24% to 41-58%, and partial accuracy from 31-74% to 73-83%. The accuracy of ICD-O-3 topography coding also improved but started and remained considerably lower with an exact accuracy of 22-40% and a partial accuracy of 56-67% after fine-tuning. Malformed code outputs dropped to 0% for all models. Tumor-diagnosis recognition reached 99%. Accuracy correlated positively with model size, but gaps between small and large models narrowed after fine-tuning. The reasoning mode in Qwen3 generally yielded a lower performance than fine-tuning and was over 100 times slower. Our findings highlight the potential of leveraging public catalogues to build instruction datasets that improve LLMs in medical documentation tasks. The complete training dataset and the best-performing checkpoints of the fine-tuned models are available from https://huggingface.co/datasets/stefan-m-lenz/ICDOPS-QA-2024.
>
---
#### [new 067] Hierarchical Frequency Tagging Probe (HFTP): A Unified Approach to Investigate Syntactic Structure Representations in Large Language Models and the Human Brain
- **分类: cs.CL; cs.NE**

- **简介: 该论文提出层级频率标记探针（HFTP），旨在探究大语言模型与人脑在句法结构表征上的相似性。通过频域分析，定位模型神经元与脑区中编码句法的组件，比较不同模型与大脑的表征对齐情况，揭示模型升级后与人脑相似性的变化趋势。**

- **链接: [http://arxiv.org/pdf/2510.13255v1](http://arxiv.org/pdf/2510.13255v1)**

> **作者:** Jingmin An; Yilong Song; Ruolin Yang; Nai Ding; Lingxi Lu; Yuxuan Wang; Wei Wang; Chu Zhuang; Qian Wang; Fang Fang
>
> **摘要:** Large Language Models (LLMs) demonstrate human-level or even superior language abilities, effectively modeling syntactic structures, yet the specific computational modules responsible remain unclear. A key question is whether LLM behavioral capabilities stem from mechanisms akin to those in the human brain. To address these questions, we introduce the Hierarchical Frequency Tagging Probe (HFTP), a tool that utilizes frequency-domain analysis to identify neuron-wise components of LLMs (e.g., individual Multilayer Perceptron (MLP) neurons) and cortical regions (via intracranial recordings) encoding syntactic structures. Our results show that models such as GPT-2, Gemma, Gemma 2, Llama 2, Llama 3.1, and GLM-4 process syntax in analogous layers, while the human brain relies on distinct cortical regions for different syntactic levels. Representational similarity analysis reveals a stronger alignment between LLM representations and the left hemisphere of the brain (dominant in language processing). Notably, upgraded models exhibit divergent trends: Gemma 2 shows greater brain similarity than Gemma, while Llama 3.1 shows less alignment with the brain compared to Llama 2. These findings offer new insights into the interpretability of LLM behavioral improvements, raising questions about whether these advancements are driven by human-like or non-human-like mechanisms, and establish HFTP as a valuable tool bridging computational linguistics and cognitive neuroscience. This project is available at https://github.com/LilTiger/HFTP.
>
---
#### [new 068] Embedding-Based Context-Aware Reranker
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决RAG系统中跨段落推理的检索难题。作者提出EBCAR，一种基于嵌入的轻量级重排序框架，通过引入段落结构信息和混合注意力机制增强跨段落理解，在准确性和效率上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.13329v1](http://arxiv.org/pdf/2510.13329v1)**

> **作者:** Ye Yuan; Mohammad Amin Shabani; Siqi Liu
>
> **备注:** Under Review
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems rely on retrieving relevant evidence from a corpus to support downstream generation. The common practice of splitting a long document into multiple shorter passages enables finer-grained and targeted information retrieval. However, it also introduces challenges when a correct retrieval would require inference across passages, such as resolving coreference, disambiguating entities, and aggregating evidence scattered across multiple sources. Many state-of-the-art (SOTA) reranking methods, despite utilizing powerful large pretrained language models with potentially high inference costs, still neglect the aforementioned challenges. Therefore, we propose Embedding-Based Context-Aware Reranker (EBCAR), a lightweight reranking framework operating directly on embeddings of retrieved passages with enhanced cross-passage understandings through the structural information of the passages and a hybrid attention mechanism, which captures both high-level interactions across documents and low-level relationships within each document. We evaluate EBCAR against SOTA rerankers on the ConTEB benchmark, demonstrating its effectiveness for information retrieval requiring cross-passage inference and its advantages in both accuracy and efficiency.
>
---
#### [new 069] Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons
- **分类: cs.CL**

- **简介: 该论文针对大模型长上下文推理中显存开销过大的问题，提出Breadcrumbs Reasoning方法，通过学习压缩历史KV缓存并定期清除，结合蒸馏与强化学习高效训练，在降低内存的同时保持推理准确性。**

- **链接: [http://arxiv.org/pdf/2510.13797v1](http://arxiv.org/pdf/2510.13797v1)**

> **作者:** Giovanni Monea; Yair Feldman; Shankar Padmanabhan; Kianté Brantley; Yoav Artzi
>
> **摘要:** The scalability of large language models for long-context reasoning is severely constrained by the linear growth of their Transformer key-value cache, which incurs significant memory and computational costs. We posit that as a model generates reasoning tokens, the informational value of past generated tokens diminishes, creating an opportunity for compression. In this work, we propose to periodically compress the generation KV cache with a learned, special-purpose token and evict compressed entries. We train the model to perform this compression via a modified joint distillation and reinforcement learning (RL) framework. Our training method minimizes overhead over the conventional RL process, as it leverages RL outputs for distillation. Empirically, our method achieves a superior memory-accuracy Pareto frontier compared to both the model without cache compression and training-free compression techniques.
>
---
#### [new 070] Mismatch Aware Guidance for Robust Emotion Control in Auto-Regressive TTS Models
- **分类: cs.CL**

- **简介: 该论文研究情感TTS中的风格-内容不匹配问题，提出一种基于大模型检测不匹配程度的自适应无分类器引导（CFG）方法，用于提升自回归TTS模型在冲突场景下的情感表达力与语音质量。**

- **链接: [http://arxiv.org/pdf/2510.13293v1](http://arxiv.org/pdf/2510.13293v1)**

> **作者:** Yizhou Peng; Yukun Ma; Chong Zhang; Yi-Wen Chao; Chongjia Ni; Bin Ma
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** While Text-to-Speech (TTS) systems can achieve fine-grained control over emotional expression via natural language prompts, a significant challenge emerges when the desired emotion (style prompt) conflicts with the semantic content of the text. This mismatch often results in unnatural-sounding speech, undermining the goal of achieving fine-grained emotional control. Classifier-Free Guidance (CFG) is a key technique for enhancing prompt alignment; however, its application to auto-regressive (AR) TTS models remains underexplored, which can lead to degraded audio quality. This paper directly addresses the challenge of style-content mismatch in AR TTS models by proposing an adaptive CFG scheme that adjusts to different levels of the detected mismatch, as measured using large language models or natural language inference models. This solution is based on a comprehensive analysis of CFG's impact on emotional expressiveness in state-of-the-art AR TTS models. Our results demonstrate that the proposed adaptive CFG scheme improves the emotional expressiveness of the AR TTS model while maintaining audio quality and intelligibility.
>
---
#### [new 071] Protect: Towards Robust Guardrailing Stack for Trustworthy Enterprise LLM Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大模型安全防护任务，旨在解决现有单模态、孤立式系统在多模态企业场景中实时性、可解释性不足的问题。作者提出多模态防护模型Protect，采用LoRA微调适配器，覆盖毒害、性别歧视、隐私和提示注入四类安全维度，实现跨文本、图像与音频的统一、可审计防护，性能超越主流模型。**

- **链接: [http://arxiv.org/pdf/2510.13351v1](http://arxiv.org/pdf/2510.13351v1)**

> **作者:** Karthik Avinash; Nikhil Pareek; Rishav Hada
>
> **摘要:** The increasing deployment of Large Language Models (LLMs) across enterprise and mission-critical domains has underscored the urgent need for robust guardrailing systems that ensure safety, reliability, and compliance. Existing solutions often struggle with real-time oversight, multi-modal data handling, and explainability -- limitations that hinder their adoption in regulated environments. Existing guardrails largely operate in isolation, focused on text alone making them inadequate for multi-modal, production-scale environments. We introduce Protect, natively multi-modal guardrailing model designed to operate seamlessly across text, image, and audio inputs, designed for enterprise-grade deployment. Protect integrates fine-tuned, category-specific adapters trained via Low-Rank Adaptation (LoRA) on an extensive, multi-modal dataset covering four safety dimensions: toxicity, sexism, data privacy, and prompt injection. Our teacher-assisted annotation pipeline leverages reasoning and explanation traces to generate high-fidelity, context-aware labels across modalities. Experimental results demonstrate state-of-the-art performance across all safety dimensions, surpassing existing open and proprietary models such as WildGuard, LlamaGuard-4, and GPT-4.1. Protect establishes a strong foundation for trustworthy, auditable, and production-ready safety systems capable of operating across text, image, and audio modalities.
>
---
#### [new 072] On the Role of Preference Variance in Preference Optimization
- **分类: cs.CL**

- **简介: 该论文研究偏好优化中偏好方差（PVar）的作用，旨在识别高效对齐大模型的高价值样本。通过理论分析与实验验证，提出基于PVar选择训练样本可提升学习效率与性能，尤其在标注有限时更有效。**

- **链接: [http://arxiv.org/pdf/2510.13022v1](http://arxiv.org/pdf/2510.13022v1)**

> **作者:** Jiacheng Guo; Zihao Li; Jiahao Qiu; Yue Wu; Mengdi Wang
>
> **摘要:** Direct Preference Optimization (DPO) has emerged as an important approach for learning from human preferences in aligning large language models (LLMs). However, collecting human preference data is costly and inefficient, motivating methods to reduce the required annotations. In this work, we investigate the impact of \emph{preference variance} (PVar), which measures the variance in model preferences when comparing pairs of responses, on the effectiveness of DPO training. We provide a theoretical insight by establishing an upper bound on the DPO gradient norm for any given prompt, showing it is controlled by the PVar of that prompt. This implies that prompts with low PVar can only produce small gradient updates, making them less valuable for learning. We validate this finding by fine-tuning LLMs with preferences generated by a reward model, evaluating on two benchmarks (AlpacaEval 2.0 and Arena-Hard). Experimental results demonstrate that prompts with higher PVar outperform randomly selected prompts or those with lower PVar. We also show that our PVar-based selection method is robust, when using smaller reward models (1B, 3B) for selection. Notably, in a separate experiment using the original human annotations from the UltraFeedback dataset, we found that training on only the top 10\% of prompts with the highest PVar yields better evaluation performance than training on the full dataset, highlighting the importance of preference variance in identifying informative examples for efficient LLM alignment.
>
---
#### [new 073] Investigating Lexical Change through Cross-Linguistic Colexification Patterns
- **分类: cs.CL**

- **简介: 该论文研究词汇演变中的共词化模式，探讨概念关联性、借用倾向和使用频率对语义变化的影响。基于三个语系的词典数据，采用系统发育模型分析共词化的演化动态，揭示不同因素及语言家族差异在意义变迁中的作用。**

- **链接: [http://arxiv.org/pdf/2510.13407v1](http://arxiv.org/pdf/2510.13407v1)**

> **作者:** Kim Gfeller; Sabine Stoll; Chundra Cathcart; Paul Widmer
>
> **摘要:** One of the most intriguing features of language is its constant change, with ongoing shifts in how meaning is expressed. Despite decades of research, the factors that determine how and why meanings evolve remain only partly understood. Colexification -- the phenomenon of expressing multiple distinct concepts using the same word form -- serves as a valuable window onto the dynamics of meaning change across languages. Here, we apply phylogenetic comparative models to dictionary data from three language families, Austronesian, Indo-European, and Uralic, in order to shed light on the evolutionary dynamics underlying the colexification of concept pairs. We assess the effects of three predictors: associativity, borrowability, and usage frequency. Our results show that more closely related concept pairs are colexified across a larger portion of the family tree and exhibit slower rates of change. In contrast, concept pairs that are more frequent and more prone to borrowing tend to change more rapidly and are less often colexified. We also find considerable differences between the language families under study, suggesting that areal and cultural factors may play a role.
>
---
#### [new 074] OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究参数高效微调中的灾难性遗忘问题，提出OPLoRA方法。通过奇异值分解和双侧正交投影，将LoRA更新限制在主导子空间的正交补空间中，理论上保证关键知识保留，有效缓解遗忘，提升模型在多任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.13003v1](http://arxiv.org/pdf/2510.13003v1)**

> **作者:** Yifeng Xiong; Xiaohui Xie
>
> **摘要:** Low-Rank Adaptation (LoRA) enables efficient fine-tuning of large language models but suffers from catastrophic forgetting when learned updates interfere with the dominant singular directions that encode essential pre-trained knowledge. We propose Orthogonal Projection LoRA (OPLoRA), a theoretically grounded approach that prevents this interference through double-sided orthogonal projections. By decomposing frozen weights via SVD, OPLoRA constrains LoRA updates to lie entirely within the orthogonal complement of the top-$k$ singular subspace using projections $P_L = I - U_k U_k^\top$ and $P_R = I - V_k V_k^\top$. We prove that this construction exactly preserves the top-$k$ singular triples, providing mathematical guarantees for knowledge retention. To quantify subspace interference, we introduce $\rho_k$, a metric measuring update alignment with dominant directions. Extensive experiments across commonsense reasoning, mathematics, and code generation demonstrate that OPLoRA significantly reduces forgetting while maintaining competitive task-specific performance on LLaMA-2 7B and Qwen2.5 7B, establishing orthogonal projection as an effective mechanism for knowledge preservation in parameter-efficient fine-tuning.
>
---
#### [new 075] Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究检索增强生成中大模型的推理忠实性问题，旨在提升模型在使用搜索引擎时推理过程的可信度。作者提出VERITAS框架，通过细粒度忠实性奖励优化强化学习过程，在保证答案正确的同时显著提升推理链的忠实性。**

- **链接: [http://arxiv.org/pdf/2510.13272v1](http://arxiv.org/pdf/2510.13272v1)**

> **作者:** Zhichao Xu; Zongyu Wu; Yun Zhou; Aosong Feng; Kang Zhou; Sangmin Woo; Kiran Ramnath; Yijun Tian; Xuan Qi; Weikang Qiu; Lin Lee Cheong; Haibo Ding
>
> **摘要:** Inspired by the success of reinforcement learning (RL) in Large Language Model (LLM) training for domains like math and code, recent works have begun exploring how to train LLMs to use search engines more effectively as tools for retrieval-augmented generation. Although these methods achieve performance improvement across QA benchmarks, many prioritize final answer correctness while overlooking the quality of intermediate reasoning steps, which may lead to chain-of-thought unfaithfulness. In this paper, we first introduce a comprehensive evaluation framework for evaluating RL-based search agents, covering three distinct faithfulness metrics: information-think faithfulness, think-answer faithfulness, and think-search faithfulness. Our evaluations reveal that a prototypical RL-based search agent, Search-R1, has significant room for improvement in this regard. To foster faithful reasoning, we introduce VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search), a novel framework that integrates fine-grained faithfulness rewards into the reinforcement learning process. Our experiments show that models trained with VERITAS not only significantly improve reasoning faithfulness, but also achieve comparable task performance across seven QA benchmarks.
>
---
#### [new 076] The Curious Case of Curiosity across Human Cultures and LLMs
- **分类: cs.CL**

- **简介: 该论文研究跨文化好奇心差异及大模型中的表达偏差，提出CUEST框架评估人类与模型在语言风格、话题偏好上的对齐程度，发现LLMs偏向西方表达并忽略文化多样性，通过微调缩小50%差距，提升模型跨文化适应性。**

- **链接: [http://arxiv.org/pdf/2510.12943v1](http://arxiv.org/pdf/2510.12943v1)**

> **作者:** Angana Borah; Rada Mihalcea
>
> **备注:** Preprint (Paper under review)
>
> **摘要:** Recent advances in Large Language Models (LLMs) have expanded their role in human interaction, yet curiosity -- a central driver of inquiry -- remains underexplored in these systems, particularly across cultural contexts. In this work, we investigate cultural variation in curiosity using Yahoo! Answers, a real-world multi-country dataset spanning diverse topics. We introduce CUEST (CUriosity Evaluation across SocieTies), an evaluation framework that measures human-model alignment in curiosity through linguistic (style), topic preference (content) analysis and grounding insights in social science constructs. Across open- and closed-source models, we find that LLMs flatten cross-cultural diversity, aligning more closely with how curiosity is expressed in Western countries. We then explore fine-tuning strategies to induce curiosity in LLMs, narrowing the human-model alignment gap by up to 50\%. Finally, we demonstrate the practical value of curiosity for LLM adaptability across cultures, showing its importance for future NLP research.
>
---
#### [new 077] Classifier-Augmented Generation for Structured Workflow Prediction
- **分类: cs.CL; cs.AI; cs.DB; cs.LG; 68T50, 68T05, 68T09; I.2.7; I.2.6; H.2.5**

- **简介: 该论文研究自然语言生成ETL工作流任务，旨在解决手动配置复杂、耗时的问题。提出Classifier-Augmented Generation方法，结合分类与少样本提示，实现流程结构、连接和属性的端到端预测，提升准确率与效率。**

- **链接: [http://arxiv.org/pdf/2510.12825v1](http://arxiv.org/pdf/2510.12825v1)**

> **作者:** Thomas Gschwind; Shramona Chakraborty; Nitin Gupta; Sameep Mehta
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** ETL (Extract, Transform, Load) tools such as IBM DataStage allow users to visually assemble complex data workflows, but configuring stages and their properties remains time consuming and requires deep tool knowledge. We propose a system that translates natural language descriptions into executable workflows, automatically predicting both the structure and detailed configuration of the flow. At its core lies a Classifier-Augmented Generation (CAG) approach that combines utterance decomposition with a classifier and stage-specific few-shot prompting to produce accurate stage predictions. These stages are then connected into non-linear workflows using edge prediction, and stage properties are inferred from sub-utterance context. We compare CAG against strong single-prompt and agentic baselines, showing improved accuracy and efficiency, while substantially reducing token usage. Our architecture is modular, interpretable, and capable of end-to-end workflow generation, including robust validation steps. To our knowledge, this is the first system with a detailed evaluation across stage prediction, edge layout, and property generation for natural-language-driven ETL authoring.
>
---
#### [new 078] UniMoE-Audio: Unified Speech and Music Generation with Dynamic-Capacity MoE
- **分类: cs.SD; cs.CL**

- **简介: 该论文聚焦统一语音与音乐生成任务，旨在解决领域冲突与数据不平衡问题。作者提出UniMoE-Audio模型，采用动态容量MoE架构与三阶段训练策略，实现语音和音乐的协同生成，提升跨域融合性能。**

- **链接: [http://arxiv.org/pdf/2510.13344v1](http://arxiv.org/pdf/2510.13344v1)**

> **作者:** Zhenyu Liu; Yunxin Li; Xuanyu Zhang; Qixun Teng; Shenyuan Jiang; Xinyu Chen; Haoyuan Shi; Jinchao Li; Qi Wang; Haolan Chen; Fanbo Meng; Mingjun Zhao; Yu Xu; Yancheng He; Baotian Hu; Min Zhang
>
> **摘要:** Recent advances in unified multimodal models indicate a clear trend towards comprehensive content generation. However, the auditory domain remains a significant challenge, with music and speech often developed in isolation, hindering progress towards universal audio synthesis. This separation stems from inherent task conflicts and severe data imbalances, which impede the development of a truly unified audio generation model. To address this challenge, we propose UniMoE-Audio, a unified speech and music generation model within a novel Dynamic-Capacity Mixture-of-Experts (MoE) framework. Architecturally, UniMoE-Audio introduces a Top-P routing strategy for dynamic expert number allocation, and a hybrid expert design comprising routed experts for domain-specific knowledge, shared experts for domain-agnostic features, and null experts for adaptive computation skipping. To tackle data imbalance, we introduce a three-stage training curriculum: 1) Independent Specialist Training leverages original datasets to instill domain-specific knowledge into each "proto-expert" without interference; 2) MoE Integration and Warmup incorporates these specialists into the UniMoE-Audio architecture, warming up the gate module and shared expert using a subset of balanced dataset; and 3) Synergistic Joint Training trains the entire model end-to-end on the fully balanced dataset, fostering enhanced cross-domain synergy. Extensive experiments show that UniMoE-Audio not only achieves state-of-the-art performance on major speech and music generation benchmarks, but also demonstrates superior synergistic learning, mitigating the performance degradation typically seen in naive joint training. Our findings highlight the substantial potential of specialized MoE architecture and curated training strategies in advancing the field of universal audio generation. Homepage: https://mukioxun.github.io/Uni-MoE-site/home.html
>
---
#### [new 079] K-Merge: Online Continual Merging of Adapters for On-device Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究设备端大模型的适配器在线持续合并任务，旨在解决增量到达的LoRA适配器在有限存储下融合时的任务遗忘问题。提出K-Merge方法，实现无需数据、高效计算的适配器选择与合并。**

- **链接: [http://arxiv.org/pdf/2510.13537v1](http://arxiv.org/pdf/2510.13537v1)**

> **作者:** Donald Shenaj; Ondrej Bohdal; Taha Ceritli; Mete Ozay; Pietro Zanuttigh; Umberto Michieli
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** On-device deployment of Large Language Models (LLMs) frequently leverages Low-Rank Adapters (LoRAs) to support diverse downstream tasks under tight resource constraints. To address the limited storage capacity of mobile devices, recent works have explored model merging techniques to fuse multiple LoRAs into a single one. In practice, however, LoRAs are often delivered incrementally, as users request support for new tasks (e.g., novel problem types or languages). This scenario introduces a new challenge: on-device online continual merging, where the objective is to incorporate new LoRAs while preserving the performance on previously supported tasks. In this paper, we propose a data-free and computationally efficient strategy for selecting and merging LoRAs when a new one becomes available, assuming the device can store only a limited number of adapters. Extensive experiments across real-world tasks demonstrate the superiority of our approach compared to alternative strategies while adhering to the storage budget and compute limitations of on-device settings.
>
---
#### [new 080] On the Reasoning Abilities of Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究掩码扩散语言模型（MDM）的推理能力，旨在揭示其可解决的推理问题类型及效率。通过与链式思维和填充循环Transformer对比，证明MDM在有限精度下等效于多项式填充的PLT，能解决CoT类问题，且对正则语言等任务更高效。**

- **链接: [http://arxiv.org/pdf/2510.13117v1](http://arxiv.org/pdf/2510.13117v1)**

> **作者:** Anej Svete; Ashish Sabharwal
>
> **摘要:** Masked diffusion models (MDMs) for text offer a compelling alternative to traditional autoregressive language models. Parallel generation makes them efficient, but their computational capabilities and the limitations inherent to their parallelism remain largely unexplored. To this end, we characterize what types of reasoning problems MDMs can provably solve and how efficiently. We do this by connecting MDMs to the well-understood reasoning frameworks of chain of thought (CoT) and padded looped transformers (PLTs) in the finite-precision log-width setting: We show that MDMs and polynomially-padded PLTs are, in fact, equivalent in this setting, and that MDMs can solve all problems that CoT-augmented transformers can. Moreover, we showcase classes of problems (including regular languages) for which MDMs are inherently more efficient than CoT transformers, where parallel generation allows for substantially faster reasoning.
>
---
#### [new 081] MMLongCite: A Benchmark for Evaluating Fidelity of Long-Context Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MMLongCite，首个评估大视觉语言模型在长上下文下保真度的多模态基准，涵盖文本、图像、视频和多种任务，揭示现有模型在长上下文处理中的不足，并分析上下文长度与关键信息位置的影响。**

- **链接: [http://arxiv.org/pdf/2510.13276v1](http://arxiv.org/pdf/2510.13276v1)**

> **作者:** Keyan Zhou; Zecheng Tang; Lingfeng Ming; Guanghao Zhou; Qiguang Chen; Dan Qiao; Zheming Yang; Libo Qin; Minghui Qiu; Juntao Li; Min Zhang
>
> **摘要:** The rapid advancement of large vision language models (LVLMs) has led to a significant expansion of their context windows. However, an extended context window does not guarantee the effective utilization of the context, posing a critical challenge for real-world applications. Current evaluations of such long-context faithfulness are predominantly focused on the text-only domain, while multimodal assessments remain limited to short contexts. To bridge this gap, we introduce MMLongCite, a comprehensive benchmark designed to evaluate the fidelity of LVLMs in long-context scenarios. MMLongCite comprises 8 distinct tasks spanning 6 context length intervals and incorporates diverse modalities, including text, images, and videos. Our evaluation of state-of-the-art LVLMs reveals their limited faithfulness in handling long multimodal contexts. Furthermore, we provide an in-depth analysis of how context length and the position of crucial content affect the faithfulness of these models.
>
---
#### [new 082] TRUSTVIS: A Multi-Dimensional Trustworthiness Evaluation Framework for Large Language Models
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出TRUSTVIS框架，旨在评估大语言模型的可信度，聚焦安全与鲁棒性问题。通过集成扰动方法与多指标投票机制，并结合可视化界面，实现对Vicuna、Llama2等模型的可解释、交互式可信评估。**

- **链接: [http://arxiv.org/pdf/2510.13106v1](http://arxiv.org/pdf/2510.13106v1)**

> **作者:** Ruoyu Sun; Da Song; Jiayang Song; Yuheng Huang; Lei Ma
>
> **备注:** 4 pages, 2 figures, To appear in ASE 2025 Demo Track
>
> **摘要:** As Large Language Models (LLMs) continue to revolutionize Natural Language Processing (NLP) applications, critical concerns about their trustworthiness persist, particularly in safety and robustness. To address these challenges, we introduce TRUSTVIS, an automated evaluation framework that provides a comprehensive assessment of LLM trustworthiness. A key feature of our framework is its interactive user interface, designed to offer intuitive visualizations of trustworthiness metrics. By integrating well-known perturbation methods like AutoDAN and employing majority voting across various evaluation methods, TRUSTVIS not only provides reliable results but also makes complex evaluation processes accessible to users. Preliminary case studies on models like Vicuna-7b, Llama2-7b, and GPT-3.5 demonstrate the effectiveness of our framework in identifying safety and robustness vulnerabilities, while the interactive interface allows users to explore results in detail, empowering targeted model improvements. Video Link: https://youtu.be/k1TrBqNVg8g
>
---
#### [new 083] Assessing LLM Reasoning Through Implicit Causal Chain Discovery in Climate Discourse
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦隐式因果链发现任务，旨在评估大语言模型在气候议题中的因果推理能力。通过诊断性框架生成因果中间步骤，分析模型推理机制，并构建基准数据集，揭示模型依赖关联匹配而非真正因果推理，但生成结果具逻辑连贯性。**

- **链接: [http://arxiv.org/pdf/2510.13417v1](http://arxiv.org/pdf/2510.13417v1)**

> **作者:** Liesbeth Allein; Nataly Pineda-Castañeda; Andrea Rocci; Marie-Francine Moens
>
> **摘要:** How does a cause lead to an effect, and which intermediate causal steps explain their connection? This work scrutinizes the mechanistic causal reasoning capabilities of large language models (LLMs) to answer these questions through the task of implicit causal chain discovery. In a diagnostic evaluation framework, we instruct nine LLMs to generate all possible intermediate causal steps linking given cause-effect pairs in causal chain structures. These pairs are drawn from recent resources in argumentation studies featuring polarized discussion on climate change. Our analysis reveals that LLMs vary in the number and granularity of causal steps they produce. Although they are generally self-consistent and confident about the intermediate causal connections in the generated chains, their judgments are mainly driven by associative pattern matching rather than genuine causal reasoning. Nonetheless, human evaluations confirmed the logical coherence and integrity of the generated chains. Our baseline causal chain discovery approach, insights from our diagnostic evaluation, and benchmark dataset with causal chains lay a solid foundation for advancing future work in implicit, mechanistic causal reasoning in argumentation settings.
>
---
#### [new 084] EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究测试时学习任务，旨在解决AI代理在运行时无法自主提升性能的问题。作者提出J-TTL基准和EvoTest框架，通过进化机制在无梯度情况下持续优化代理的提示、记忆、参数与工具使用，显著提升多轮任务表现。**

- **链接: [http://arxiv.org/pdf/2510.13220v1](http://arxiv.org/pdf/2510.13220v1)**

> **作者:** Yufei He; Juncheng Liu; Yue Liu; Yibo Li; Tri Cao; Zhiyuan Hu; Xinxing Xu; Bryan Hooi
>
> **摘要:** A fundamental limitation of current AI agents is their inability to learn complex skills on the fly at test time, often behaving like "clever but clueless interns" in novel environments. This severely limits their practical utility. To systematically measure and drive progress on this challenge, we first introduce the Jericho Test-Time Learning (J-TTL) benchmark. J-TTL is a new evaluation setup where an agent must play the same game for several consecutive episodes, attempting to improve its performance from one episode to the next. On J-TTL, we find that existing adaptation methods like reflection, memory, or reinforcement learning struggle. To address the challenges posed by our benchmark, we present EvoTest, an evolutionary test-time learning framework that improves an agent without any fine-tuning or gradients-by evolving the entire agentic system after every episode. EvoTest has two roles: the Actor Agent, which plays the game, and the Evolver Agent, which analyzes the episode transcript to propose a revised configuration for the next run. This configuration rewrites the prompt, updates memory by logging effective state-action choices, tunes hyperparameters, and learns the tool-use routines. On our J-TTL benchmark, EvoTest consistently increases performance, outperforming not only reflection and memory-only baselines but also more complex online fine-tuning methods. Notably, our method is the only one capable of winning two games (Detective and Library), while all baselines fail to win any.
>
---
#### [new 085] Program of Thoughts for Financial Reasoning: Leveraging Dynamic In-Context Examples and Generative Retrieval
- **分类: cs.CE; cs.AI; cs.CL**

- **简介: 该论文针对金融领域数值推理任务，旨在提升大模型在FinQA和ConvFinQA数据集上的表现。提出FINDER框架，结合生成式检索与动态上下文示例选择，增强事实提取与推理能力，显著提升执行准确率。**

- **链接: [http://arxiv.org/pdf/2510.13157v1](http://arxiv.org/pdf/2510.13157v1)**

> **作者:** Subhendu Khatuya; Shashwat Naidu; Pawan Goyal; Niloy Ganguly
>
> **备注:** This work has been accepted for publication in the Main Conference of the Empirical Methods in Natural Language Processing (EMNLP) 2025
>
> **摘要:** Despite continuous advancements in the capabilities of large language models (LLMs), numerical reasoning remains a challenging area. Techniques like chain-of-thought prompting, tree-of-thought prompting, and program-of-thought prompting guide LLMs through intermediate reasoning steps. Although in-context learning with few-shot prompting has improved performance, LLMs still lag behind state-of-the-art models on financial numerical reasoning datasets such as FinQA and ConvFinQA. In this work, we introduce FINDER, a novel two-step framework, to enhance LLMs' capabilities in financial numerical reasoning. The first step utilizes a generative retriever to extract relevant facts from unstructured data, including both text and tables. This is followed by context-aware Program of Thought prompting with dynamic selection of in-context examples. Our model FINDER achieves a new state-of-the-art performance on both the FinQA and ConvFinQA datasets, surpassing previous benchmarks with execution accuracy improvements of 5.98% and 4.05%, respectively.
>
---
#### [new 086] Addressing the alignment problem in transportation policy making: an LLM approach
- **分类: cs.CY; cs.CE; cs.CL; cs.MA**

- **简介: 该论文探讨利用大语言模型（LLM）解决交通政策中公众偏好与决策工具间的错配问题。通过构建多智能体仿真，让LLM代表居民对交通提案投票，模拟民主共识，评估其在芝加哥和休斯顿的应用效果，揭示LLM在捕捉集体偏好方面的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2510.13139v1](http://arxiv.org/pdf/2510.13139v1)**

> **作者:** Xiaoyu Yan; Tianxing Dai; Yu; Nie
>
> **摘要:** A key challenge in transportation planning is that the collective preferences of heterogeneous travelers often diverge from the policies produced by model-driven decision tools. This misalignment frequently results in implementation delays or failures. Here, we investigate whether large language models (LLMs), noted for their capabilities in reasoning and simulating human decision-making, can help inform and address this alignment problem. We develop a multi-agent simulation in which LLMs, acting as agents representing residents from different communities in a city, participate in a referendum on a set of transit policy proposals. Using chain-of-thought reasoning, LLM agents provide ranked-choice or approval-based preferences, which are aggregated using instant-runoff voting (IRV) to model democratic consensus. We implement this simulation framework with both GPT-4o and Claude-3.5, and apply it for Chicago and Houston. Our findings suggest that LLM agents are capable of approximating plausible collective preferences and responding to local context, while also displaying model-specific behavioral biases and modest divergences from optimization-based benchmarks. These capabilities underscore both the promise and limitations of LLMs as tools for solving the alignment problem in transportation decision-making.
>
---
#### [new 087] UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles
- **分类: cs.RO; cs.CL; cs.CV; cs.MA**

- **简介: 该论文研究多车协同规划任务，解决通信效率与安全性问题。提出UNCAP方法，利用自然语言传递带不确定性感知的信息，通过选择性融合高价值消息，降低带宽消耗，提升决策安全性和可靠性。**

- **链接: [http://arxiv.org/pdf/2510.12992v1](http://arxiv.org/pdf/2510.12992v1)**

> **作者:** Neel P. Bhatt; Po-han Li; Kushagra Gupta; Rohan Siva; Daniel Milan; Alexander T. Hogue; Sandeep P. Chinchali; David Fridovich-Keil; Zhangyang Wang; Ufuk Topcu
>
> **摘要:** Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: https://uncap-project.github.io/
>
---
#### [new 088] LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中鲁棒性不足的问题，系统分析了七类扰动下的模型表现。研究发现，尽管基准成绩高，但模型对视角、初始状态等敏感，且常忽略语言指令，揭示了当前评估方式的局限性。**

- **链接: [http://arxiv.org/pdf/2510.13626v1](http://arxiv.org/pdf/2510.13626v1)**

> **作者:** Senyu Fei; Siyin Wang; Junhao Shi; Zihao Dai; Jikun Cai; Pengfang Qian; Li Ji; Xinzhe He; Shiduo Zhang; Zhaoye Fei; Jinlan Fu; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation.
>
---
#### [new 089] Generative Universal Verifier as Multimodal Meta-Reasoner
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出生成式通用验证器，解决多模态模型在视觉推理中缺乏可靠验证的问题。构建ViVerBench基准和OmniVerifier模型，实现视觉结果的反思与优化，提升生成与推理的可靠性与可控性。**

- **链接: [http://arxiv.org/pdf/2510.13804v1](http://arxiv.org/pdf/2510.13804v1)**

> **作者:** Xinchen Zhang; Xiaoying Zhang; Youbin Wu; Yanbin Cao; Renrui Zhang; Ruihang Chu; Ling Yang; Yujiu Yang
>
> **摘要:** We introduce Generative Universal Verifier, a novel concept and plugin designed for next-generation multimodal reasoning in vision-language models and unified multimodal models, providing the fundamental capability of reflection and refinement on visual outcomes during the reasoning and generation process. This work makes three main contributions: (1) We build ViVerBench, a comprehensive benchmark spanning 16 categories of critical tasks for evaluating visual outcomes in multimodal reasoning. Results show that existing VLMs consistently underperform across these tasks, underscoring a substantial gap from human-level capability in reliable visual verification. (2) We design two automated pipelines to construct large-scale visual verification data and train OmniVerifier-7B, the first omni-capable generative verifier trained for universal visual verification and achieves notable gains on ViVerBench(+8.3). Through training, we identify three atomic capabilities in visual verification and demonstrate how they generalize and interact synergistically. (3) We propose OmniVerifier-TTS, a sequential test-time scaling paradigm that leverages the universal verifier to bridge image generation and editing within unified models, enhancing the upper bound of generative ability through iterative fine-grained optimization. Beyond generation, we extend universal verifier to broader world-modeling interleaved reasoning scenarios. Empirically, OmniVerifier-TTS achieves improvements on T2I-ReasonBench(+3.7), and GenEval++(+4.3), outperforming existing parallel test-time scaling methods, such as Best-of-N. By endowing multimodal reasoning with reliable visual verification, OmniVerifier advances both reliable reflection during generation and scalable test-time refinement, marking a step toward more trustworthy and controllable next-generation reasoning systems.
>
---
#### [new 090] Toward LLM-Supported Automated Assessment of Critical Thinking Subskills
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文研究基于学生议论文自动评估批判性思维子技能的可行性。任务为自动化评分，解决缺乏有效测量方法的问题。工作包括构建编码量规、人工标注数据，并比较三种大模型在不同提示策略下的表现。**

- **链接: [http://arxiv.org/pdf/2510.12915v1](http://arxiv.org/pdf/2510.12915v1)**

> **作者:** Marisa C. Peczuh; Nischal Ashok Kumar; Ryan Baker; Blair Lehman; Danielle Eisenberg; Caitlin Mills; Keerthi Chebrolu; Sudhip Nashi; Cadence Young; Brayden Liu; Sherry Lachman; Andrew Lan
>
> **备注:** preprint: 17 pages
>
> **摘要:** Critical thinking represents a fundamental competency in today's education landscape. Developing critical thinking skills through timely assessment and feedback is crucial; however, there has not been extensive work in the learning analytics community on defining, measuring, and supporting critical thinking. In this paper, we investigate the feasibility of measuring core "subskills" that underlie critical thinking. We ground our work in an authentic task where students operationalize critical thinking: student-written argumentative essays. We developed a coding rubric based on an established skills progression and completed human coding for a corpus of student essays. We then evaluated three distinct approaches to automated scoring: zero-shot prompting, few-shot prompting, and supervised fine-tuning, implemented across three large language models (GPT-5, GPT-5-mini, and ModernBERT). GPT-5 with few-shot prompting achieved the strongest results and demonstrated particular strength on subskills with separable, frequent categories, while lower performance was observed for subskills that required detection of subtle distinctions or rare categories. Our results underscore critical trade-offs in automated critical thinking assessment: proprietary models offer superior reliability at higher cost, while open-source alternatives provide practical accuracy with reduced sensitivity to minority categories. Our work represents an initial step toward scalable assessment of higher-order reasoning skills across authentic educational contexts.
>
---
#### [new 091] From Literal to Liberal: A Meta-Prompting Framework for Eliciting Human-Aligned Exception Handling in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大语言模型僵化遵循规则导致违背人类意图的问题，提出一种名为RID的低计算元提示框架，通过结构化认知模式提升模型在零样本下对例外情况的合理处理能力，实现从字面到意图驱动的推理。**

- **链接: [http://arxiv.org/pdf/2510.12864v1](http://arxiv.org/pdf/2510.12864v1)**

> **作者:** Imran Khan
>
> **备注:** 13 pages. Code and data are available at https://github.com/strongSoda/LITERAL-TO-LIBERAL
>
> **摘要:** Large Language Models (LLMs) are increasingly being deployed as the reasoning engines for agentic AI systems, yet they exhibit a critical flaw: a rigid adherence to explicit rules that leads to decisions misaligned with human common sense and intent. This "rule-rigidity" is a significant barrier to building trustworthy autonomous agents. While prior work has shown that supervised fine-tuning (SFT) with human explanations can mitigate this issue, SFT is computationally expensive and inaccessible to many practitioners. To address this gap, we introduce the Rule-Intent Distinction (RID) Framework, a novel, low-compute meta-prompting technique designed to elicit human-aligned exception handling in LLMs in a zero-shot manner. The RID framework provides the model with a structured cognitive schema for deconstructing tasks, classifying rules, weighing conflicting outcomes, and justifying its final decision. We evaluated the RID framework against baseline and Chain-of-Thought (CoT) prompting on a custom benchmark of 20 scenarios requiring nuanced judgment across diverse domains. Our human-verified results demonstrate that the RID framework significantly improves performance, achieving a 95% Human Alignment Score (HAS), compared to 80% for the baseline and 75% for CoT. Furthermore, it consistently produces higher-quality, intent-driven reasoning. This work presents a practical, accessible, and effective method for steering LLMs from literal instruction-following to liberal, goal-oriented reasoning, paving the way for more reliable and pragmatic AI agents.
>
---
#### [new 092] Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文针对音视频语音识别中的错误修正问题，提出DualHyp框架，利用ASR和VSR双模态假设生成N-best候选，并通过噪声感知的RelPrompt机制引导大模型动态融合与纠错，显著提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.13281v1](http://arxiv.org/pdf/2510.13281v1)**

> **作者:** Sungnyun Kim; Kangwook Jang; Sungwoo Cho; Joon Son Chung; Hoirin Kim; Se-Young Yun
>
> **备注:** Preprint work
>
> **摘要:** This paper introduces a new paradigm for generative error correction (GER) framework in audio-visual speech recognition (AVSR) that reasons over modality-specific evidences directly in the language space. Our framework, DualHyp, empowers a large language model (LLM) to compose independent N-best hypotheses from separate automatic speech recognition (ASR) and visual speech recognition (VSR) models. To maximize the effectiveness of DualHyp, we further introduce RelPrompt, a noise-aware guidance mechanism that provides modality-grounded prompts to the LLM. RelPrompt offers the temporal reliability of each modality stream, guiding the model to dynamically switch its focus between ASR and VSR hypotheses for an accurate correction. Under various corruption scenarios, our framework attains up to 57.7% error rate gain on the LRS2 benchmark over standard ASR baseline, contrary to single-stream GER approaches that achieve only 10% gain. To facilitate research within our DualHyp framework, we release the code and the dataset comprising ASR and VSR hypotheses at https://github.com/sungnyun/dualhyp.
>
---
#### [new 093] Unifying Vision-Language Latents for Zero-label Image Caption Enhancement
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究零标签图像描述生成任务，旨在解决现有模型依赖标注数据的问题。提出ViZer框架，通过视觉-语言特征对齐，在无需文本标签或完全重训练的情况下提升预训练模型的描述生成质量。**

- **链接: [http://arxiv.org/pdf/2510.12931v1](http://arxiv.org/pdf/2510.12931v1)**

> **作者:** Sanghyun Byun; Jung Ick Guack; Mohanad Odema; Baisub Lee; Jacob Song; Woo Seong Chung
>
> **备注:** Accepted to PMLR and NeurIPS 2025 UniReps
>
> **摘要:** Vision-language models (VLMs) achieve remarkable performance through large-scale image-text pretraining. However, their reliance on labeled image datasets limits scalability and leaves vast amounts of unlabeled image data underutilized. To address this, we propose Unified Vision-Language Alignment for Zero-Label Enhancement (ViZer), an enhancement training framework that enables zero-label learning in image captioning, providing a practical starting point for broader zero-label adaptation in vision-language tasks. Unlike prior approaches that rely on human or synthetically annotated datasets, ViZer actively aligns vision and language representation features during training, enabling existing VLMs to generate improved captions without requiring text labels or full retraining. We demonstrate ViZer's advantage in qualitative evaluation, as automated caption metrics such as CIDEr and BERTScore often penalize details that are absent in reference captions. Applying ViZer on SmolVLM-Base and Qwen2-VL, we observe consistent qualitative improvements, producing captions that are more grounded and descriptive than their baseline.
>
---
#### [new 094] AutoCode: LLMs as Problem Setters for Competitive Programming
- **分类: cs.SE; cs.AI; cs.CL; cs.PL**

- **简介: 该论文提出AutoCode，利用大语言模型生成竞赛级编程题目。任务是自动创建高质量、高难度的算法问题。解决传统人工出题耗时且难控质量的问题。通过多轮验证生成题目与测试用例，并经高手评估确认其达到比赛标准。**

- **链接: [http://arxiv.org/pdf/2510.12803v1](http://arxiv.org/pdf/2510.12803v1)**

> **作者:** Shang Zhou; Zihan Zheng; Kaiyuan Liu; Zeyu Shen; Zerui Cheng; Zexing Chen; Hansen He; Jianzhu Yao; Huanzhi Mao; Qiuyang Mang; Tianfu Fu; Beichen Li; Dongruixuan Li; Wenhao Chai; Zhuang Liu; Aleksandra Korolova; Peter Henderson; Natasha Jaques; Pramod Viswanath; Saining Xie; Jingbo Shang
>
> **备注:** Project page: https://livecodebenchpro.com/projects/autocode/overview
>
> **摘要:** Writing competitive programming problems is exacting. Authors must: set constraints, input distributions, and edge cases that rule out shortcuts; target specific algorithms (e.g., max-flow, dynamic programming, data structures); and calibrate complexity beyond the reach of most competitors. We argue that this makes for an ideal test of general large language model capabilities and study whether they can do this reliably. We introduce AutoCode, which uses multiple rounds of validation to yield competition-grade problem statements and test cases. On held-out problems, AutoCode test suites approach 99% consistency with official judgments, a significant improvement over current state-of-the-art methods like HardTests, which achieve less than 81%. Furthermore, starting with a random seed problem, AutoCode can create novel variants with reference and brute-force solutions. By cross-verifying these generated solutions against test cases, we can further filter out malformed problems. Our system ensures high correctness, as verified by human experts. AutoCode successfully produces novel problems judged by Grandmaster-level (top 0.3%) competitive programmers to be of contest quality.
>
---
#### [new 095] DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦于提升深度研究代理的规划能力，解决现有方法对规划阶段优化不足的问题。提出DeepPlanner框架，通过熵感知的优势塑造，在token和样本层面优化强化学习过程，显著提升规划质量，实现在更低训练成本下达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2510.12979v1](http://arxiv.org/pdf/2510.12979v1)**

> **作者:** Wei Fan; Wenlin Yao; Zheng Li; Feng Yao; Xin Liu; Liang Qiu; Qingyu Yin; Yangqiu Song; Bing Yin
>
> **备注:** Under Review
>
> **摘要:** Large language models (LLMs) augmented with multi-step reasoning and action generation abilities have shown promise in leveraging external tools to tackle complex tasks that require long-horizon planning. However, existing approaches either rely on implicit planning in the reasoning stage or introduce explicit planners without systematically addressing how to optimize the planning stage. As evidence, we observe that under vanilla reinforcement learning (RL), planning tokens exhibit significantly higher entropy than other action tokens, revealing uncertain decision points that remain under-optimized. To address this, we propose DeepPlanner, an end-to-end RL framework that effectively enhances the planning capabilities of deep research agents. Our approach shapes token-level advantage with an entropy-based term to allocate larger updates to high entropy tokens, and selectively upweights sample-level advantages for planning-intensive rollouts. Extensive experiments across seven deep research benchmarks demonstrate that DeepPlanner improves planning quality and achieves state-of-the-art results under a substantially lower training budget.
>
---
#### [new 096] Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦数学推理中的步骤级验证任务，旨在解决大模型生成证明中细粒度错误难以检测的问题。作者构建了Hard2Verify基准，含人工标注的步骤级反馈，并评估多种验证模型，揭示开源模型在细粒度验证上的不足。**

- **链接: [http://arxiv.org/pdf/2510.13744v1](http://arxiv.org/pdf/2510.13744v1)**

> **作者:** Shrey Pandit; Austin Xu; Xuan-Phi Nguyen; Yifei Ming; Caiming Xiong; Shafiq Joty
>
> **备注:** 21 pages, 8 figures, 5 tables
>
> **摘要:** Large language model (LLM)-based reasoning systems have recently achieved gold medal-level performance in the IMO 2025 competition, writing mathematical proofs where, to receive full credit, each step must be not only correct but also sufficiently supported. To train LLM-based reasoners in such challenging, open-ended settings, strong verifiers capable of catching step-level mistakes are necessary prerequisites. We introduce Hard2Verify, a human-annotated, step-level verification benchmark produced with over 500 hours of human labor. Hard2Verify is designed to rigorously assess step-level verifiers at the frontier: Verifiers must provide step-level annotations or identify the first error in responses generated by frontier LLMs for very recent, challenging, and open-ended math questions. We evaluate 29 generative critics and process reward models, demonstrating that, beyond a few standouts, open-source verifiers lag closed source models. We subsequently analyze what drives poor performance in step-level verification, the impacts of scaling verifier compute, as well as fundamental questions such as self-verification and verification-generation dynamics.
>
---
#### [new 097] Max It or Miss It: Benchmarking LLM On Solving Extremal Problems
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文聚焦大模型在极值问题求解中的推理能力评估，提出ExtremBench基准，包含93个数学极值问题。通过多模型实验，揭示现有数学基准未能充分反映优化推理能力，指出当前评测体系的局限性。**

- **链接: [http://arxiv.org/pdf/2510.12997v1](http://arxiv.org/pdf/2510.12997v1)**

> **作者:** Binxin Gao; Jingjun Han
>
> **备注:** Our benchmark dataset is available at https://huggingface.co/datasets/binxingao/extrem-bench
>
> **摘要:** Test-time scaling has enabled Large Language Models (LLMs) with remarkable reasoning capabilities, particularly in mathematical domains, through intermediate chain-of-thought (CoT) reasoning before generating final answers. However, the specific sources and mechanisms underlying these reasoning capabilities remain insufficiently understood. Optimization reasoning, i.e. finding extrema under constraints, represents a fundamental abstraction that underpins critical applications in planning, control, resource allocation, and prompt search. To systematically evaluate this capability, we introduce ExtremBench, a benchmark dataset for solving mathematical extremal problems, curated from inequality exercises used for Chinese Mathematical Olympiad and transformed into $93$ standardized extrema-finding problems. We conduct extensive evaluations across various state-of-the-art open-source model families, including the Qwen3, GPT-OSS, and DeepSeek. Our results reveal that LLMs' extremal-solving reasoning capabilities do not always align with those of current mathematical benchmarks such as AIME25 and MATH-500, with some models showing strong general mathematical reasoning but poor extremal-solving skills, and vice versa. This discrepancy highlights a critical gap in current evaluation practices and suggests that existing benchmarks may not comprehensively capture the full spectrum of mathematical reasoning abilities.
>
---
#### [new 098] Personalized Learning Path Planning with Goal-Driven Learner State Modeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究个性化学习路径规划（PLPP），旨在解决现有方法缺乏目标对齐机制的问题。提出Pxplore框架，结合强化学习与大语言模型，设计学习者状态模型和自动奖励函数，实现目标驱动的个性化学习路径生成。**

- **链接: [http://arxiv.org/pdf/2510.13215v1](http://arxiv.org/pdf/2510.13215v1)**

> **作者:** Joy Jia Yin Lim; Ye He; Jifan Yu; Xin Cong; Daniel Zhang-Li; Zhiyuan Liu; Huiqin Liu; Lei Hou; Juanzi Li; Bin Xu
>
> **摘要:** Personalized Learning Path Planning (PLPP) aims to design adaptive learning paths that align with individual goals. While large language models (LLMs) show potential in personalizing learning experiences, existing approaches often lack mechanisms for goal-aligned planning. We introduce Pxplore, a novel framework for PLPP that integrates a reinforcement-based training paradigm and an LLM-driven educational architecture. We design a structured learner state model and an automated reward function that transforms abstract objectives into computable signals. We train the policy combining supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), and deploy it within a real-world learning platform. Extensive experiments validate Pxplore's effectiveness in producing coherent, personalized, and goal-driven learning paths. We release our code and dataset to facilitate future research.
>
---
## 更新

#### [replaced 001] RPM: Reasoning-Level Personalization for Black-Box Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21082v4](http://arxiv.org/pdf/2505.21082v4)**

> **作者:** Jieyong Kim; Tongyoung Kim; Soojin Yoon; Jaehyung Kim; Dongha Lee
>
> **摘要:** While black-box large language models are widely deployed, they produce generic outputs that overlook individual user preferences. Current personalization methods are fundamentally limited to response-level personalization; they only match final outputs, failing to model the underlying reasoning that connects user behavior to responses. To address this, this work introduces reasoning-level personalization as a new paradigm and proposes RPM, the first systematic framework designed to guide the model's reasoning process using structured rationales constructed from patterns in a user's behavior. RPM constructs a structured model of user behavior-built from response-influential features and statistical factors-to create personalized reasoning paths and retrieve beneficial examples for guiding inference through a feature-based retrieval mechanism. Extensive experiments across four diverse tasks demonstrate that RPM consistently outperforms existing response-level methods while simultaneously enhancing both personalization performance and interpretability, providing a promising direction for black-box LLM personalization.
>
---
#### [replaced 002] MULTI: Multimodal Understanding Leaderboard with Text and Images
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.03173v4](http://arxiv.org/pdf/2402.03173v4)**

> **作者:** Zichen Zhu; Yang Xu; Lu Chen; Jingkai Yang; Yichuan Ma; Yiming Sun; Hailin Wen; Jiaqi Liu; Jinyu Cai; Yingzi Ma; Situo Zhang; Zihan Zhao; Liangtai Sun; Kai Yu
>
> **备注:** 24 pages, 19 figures, 10 tables. Details and access are available at: https://OpenDFM.github.io/MULTI-Benchmark/
>
> **摘要:** The rapid development of multimodal large language models (MLLMs) raises the question of how they compare to human performance. While existing datasets often feature synthetic or overly simplistic tasks, some models have already surpassed human expert baselines. In this paper, we present MULTI, a Chinese multimodal dataset derived from authentic examination questions. Comprising over 18,000 carefully selected and refined questions, MULTI evaluates models using real-world examination standards, encompassing image-text comprehension, complex reasoning, and knowledge recall. Additionally, We also introduce MULTI-Elite, a 500-question selected hard subset, and MULTI-Extend with more than 4,500 external knowledge context pieces for testing in-context learning capabilities. Our evaluation highlights substantial room for MLLM advancement, with Qwen2-VL-72B achieving a 76.9% accuracy on MULTI and 53.1% on MULTI-Elite leading 25 evaluated models, compared to human expert baselines of 86.1% and 73.1%. MULTI serves not only as a robust evaluation platform but also paves the way for the development of expert-level AI.
>
---
#### [replaced 003] LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15969v3](http://arxiv.org/pdf/2506.15969v3)**

> **作者:** Haoyue Zhang; Hualei Zhang; Xiaosong Ma; Jie Zhang; Song Guo
>
> **摘要:** Large Language Models (LLMs) exhibit enhanced capabilities by Chain-of-Thought reasoning. However, the extended reasoning sequences introduce significant GPU memory overhead due to increased key-value (KV) cache. Existing KV cache compression methods mitigate memory bottlenecks but struggle in long reasoning tasks. In this paper, we analyze attention patterns in reasoning tasks and reveal a Token Importance Recurrence phenomenon: a large proportion of tokens regain high attention after multiple decoding steps, which is failed to capture by existing works and may lead to unpredictable eviction on such periodically critical tokens. To address this, we propose LazyEviction, an observation window-based lagged eviction framework retaining latent recurring tokens by prioritized eviction based on tokens' recurrence patterns. Extensive experiments demonstrate that LazyEviction reduces KV cache by 50%~70% while maintaining comparable accuracy, outperforming existing KV cache compression baselines. Our implementation code can be found at https://github.com/Halo-949/LazyEviction.
>
---
#### [replaced 004] Orthogonal Finetuning Made Scalable
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19847v2](http://arxiv.org/pdf/2506.19847v2)**

> **作者:** Zeju Qiu; Weiyang Liu; Adrian Weller; Bernhard Schölkopf
>
> **备注:** EMNLP 2025 Main (18 pages, 7 figures, project page: https://spherelab.ai/oftv2/)
>
> **摘要:** Orthogonal finetuning (OFT) offers highly parameter-efficient adaptation while preventing catastrophic forgetting, but its high runtime and memory demands limit practical deployment. We identify the core computational bottleneck in OFT as its weight-centric implementation, which relies on costly matrix-matrix multiplications with cubic complexity. To overcome this, we propose OFTv2, an input-centric reformulation that instead uses matrix-vector multiplications (i.e., matrix-free computation), reducing the computational cost to quadratic. We further introduce the Cayley-Neumann parameterization, an efficient orthogonal parameterization that approximates the matrix inversion in the Cayley transform via a truncated Neumann series. These modifications allow OFTv2 to achieve up to 10x faster training and 3x lower GPU memory usage without compromising performance. In addition, we extend OFTv2 to support finetuning quantized foundation models and show that it outperforms the popular QLoRA in training stability, efficiency, and memory usage.
>
---
#### [replaced 005] Learning Optimal Prompt Ensemble for Multi-source Visual Prompt Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12311v5](http://arxiv.org/pdf/2504.12311v5)**

> **作者:** Enming Zhang; Liwen Cao; Yanru Wu; Zijie Zhao; Yang Li
>
> **摘要:** Prompt tuning has emerged as a lightweight strategy for adapting foundation models to downstream tasks, particularly for resource-constrained systems. As pre-trained prompts become valuable assets, combining multiple source prompts offers a promising approach to enhance generalization for new tasks by leveraging complementary knowledge. However, naive aggregation often overlooks different source prompts have different contribution potential to the target task. To address this, we propose HGPrompt, a dynamic framework that learns optimal ensemble weights. These weights are optimized by jointly maximizing an information-theoretic metric for transferability and minimizing gradient conflicts via a novel regularization strategy. Specifically, we propose a differentiable prompt transferability metric to captures the discriminability of prompt-induced features on the target task. Meanwhile, HGPrompt match the gradient variances with respect to different source prompts based on Hessian and Fisher Information, ensuring stable and coherent knowledge transfer while suppressing gradient conflicts among them. Extensive experiments on the large-scale VTAB benchmark demonstrate the state-of-the-art performance of HGPrompt, validating its effectiveness in learning an optimal ensemble for effective multi-source prompt transfer.
>
---
#### [replaced 006] A Linguistically Motivated Analysis of Intonational Phrasing in Text-to-Speech Systems: Revealing Gaps in Syntactic Sensitivity
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22236v2](http://arxiv.org/pdf/2505.22236v2)**

> **作者:** Charlotte Pouw; Afra Alishahi; Willem Zuidema
>
> **备注:** Accepted to CoNLL 2025
>
> **摘要:** We analyze the syntactic sensitivity of Text-to-Speech (TTS) systems using methods inspired by psycholinguistic research. Specifically, we focus on the generation of intonational phrase boundaries, which can often be predicted by identifying syntactic boundaries within a sentence. We find that TTS systems struggle to accurately generate intonational phrase boundaries in sentences where syntactic boundaries are ambiguous (e.g., garden path sentences or sentences with attachment ambiguity). In these cases, systems need superficial cues such as commas to place boundaries at the correct positions. In contrast, for sentences with simpler syntactic structures, we find that systems do incorporate syntactic cues beyond surface markers. Finally, we finetune models on sentences without commas at the syntactic boundary positions, encouraging them to focus on more subtle linguistic cues. Our findings indicate that this leads to more distinct intonation patterns that better reflect the underlying structure.
>
---
#### [replaced 007] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03550v2](http://arxiv.org/pdf/2508.03550v2)**

> **作者:** Peng Lai; Jianjie Zheng; Sijie Cheng; Yun Chen; Peng Li; Yang Liu; Guanhua Chen
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
>
---
#### [replaced 008] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01422v2](http://arxiv.org/pdf/2503.01422v2)**

> **作者:** Yiming Wang; Pei Zhang; Siyuan Huang; Baosong Yang; Zhuosheng Zhang; Fei Huang; Rui Wang
>
> **备注:** Accepted by NeurIPS 2025 (Spotlight)
>
> **摘要:** Test-time scaling enhances large language model performance by allocating additional compute resources during inference. Best-of-N (BoN) sampling serves as a common sampling-based scaling technique, broadening the search space in parallel to find better solutions from the model distribution. However, its cost-performance trade-off is still underexplored. Two main challenges limit the efficiency of BoN sampling: (1) Generating N full samples consumes substantial GPU memory, reducing inference capacity under limited resources. (2) Reward models add extra memory and latency overhead, and training strong reward models introduces potential training data costs. Although some studies have explored efficiency improvements, none have addressed both challenges at once. To address this gap, we propose Self-Truncation Best-of-N (ST-BoN), a decoding method that avoids fully generating all N samples and eliminates the need for reward models. It leverages early sampling consistency in the model's internal states to identify the most promising path and truncate suboptimal ones. In terms of cost, ST-BoN reduces dynamic GPU memory usage by over 80% and inference latency by 50%. In terms of cost-performance trade-off, ST-BoN achieves the same performance as Full-BoN while saving computational cost by 70%-80%, and under the same cost, it can improve accuracy by 3-4 points.
>
---
#### [replaced 009] No Language Data Left Behind: A Comparative Study of CJK Language Datasets in the Hugging Face Ecosystem
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04329v3](http://arxiv.org/pdf/2507.04329v3)**

> **作者:** Dasol Choi; Woomyoung Park; Youngsook Song
>
> **备注:** Accepted to EMNLP 2025 MRL Workshop
>
> **摘要:** Recent advances in Natural Language Processing (NLP) have underscored the crucial role of high-quality datasets in building large language models (LLMs). However, while extensive resources and analyses exist for English, the landscape for East Asian languages - particularly Chinese, Japanese, and Korean (CJK) - remains fragmented and underexplored, despite these languages together serving over 1.6 billion speakers. To address this gap, we investigate the HuggingFace ecosystem from a cross-linguistic perspective, focusing on how cultural norms, research environments, and institutional practices shape dataset availability and quality. Drawing on more than 3,300 datasets, we employ quantitative and qualitative methods to examine how these factors drive distinct creation and curation patterns across Chinese, Japanese, and Korean NLP communities. Our findings highlight the large-scale and often institution-driven nature of Chinese datasets, grassroots community-led development in Korean NLP, and an entertainment- and subculture-focused emphasis on Japanese collections. By uncovering these patterns, we reveal practical strategies for enhancing dataset documentation, licensing clarity, and cross-lingual resource sharing - ultimately guiding more effective and culturally attuned LLM development in East Asia. We conclude by discussing best practices for future dataset curation and collaboration, aiming to strengthen resource development across all three languages.
>
---
#### [replaced 010] Early Signs of Steganographic Capabilities in Frontier LLMs
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02737v2](http://arxiv.org/pdf/2507.02737v2)**

> **作者:** Artur Zolkowski; Kei Nishimura-Gasparian; Robert McCarthy; Roland S. Zimmermann; David Lindner
>
> **摘要:** Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances like using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future.
>
---
#### [replaced 011] Benchmarking LLMs' Swarm intelligence
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04364v4](http://arxiv.org/pdf/2505.04364v4)**

> **作者:** Kai Ruan; Mowen Huang; Ji-Rong Wen; Hao Sun
>
> **摘要:** Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
>
---
#### [replaced 012] GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling
- **分类: cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.19234v2](http://arxiv.org/pdf/2505.19234v2)**

> **作者:** Jialong Zhou; Lichao Wang; Xiao Yang
>
> **摘要:** The emergence of large language models (LLMs) enables the development of intelligent agents capable of engaging in complex and multi-turn dialogues. However, multi-agent collaboration faces critical safety challenges, such as hallucination amplification and error injection and propagation. This paper presents GUARDIAN, a unified method for detecting and mitigating multiple safety concerns in GUARDing Intelligent Agent collaboratioNs. By modeling the multi-agent collaboration process as a discrete-time temporal attributed graph, GUARDIAN explicitly captures the propagation dynamics of hallucinations and errors. The unsupervised encoder-decoder architecture incorporating an incremental training paradigm learns to reconstruct node attributes and graph structures from latent embeddings, enabling the identification of anomalous nodes and edges with unparalleled precision. Moreover, we introduce a graph abstraction mechanism based on the Information Bottleneck Theory, which compresses temporal interaction graphs while preserving essential patterns. Extensive experiments demonstrate GUARDIAN's effectiveness in safeguarding LLM multi-agent collaborations against diverse safety vulnerabilities, achieving state-of-the-art accuracy with efficient resource utilization. The code is available at https://github.com/JialongZhou666/GUARDIAN
>
---
#### [replaced 013] Multi-Scale Probabilistic Generation Theory: A Unified Information-Theoretic Framework for Hierarchical Structure in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18244v2](http://arxiv.org/pdf/2505.18244v2)**

> **作者:** Yukin Zhang; Qi Dong
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable emergent abilities but remain poorly understood at a mechanistic level. This paper introduces the Multi-Scale Probabilistic Generation Theory (MSPGT), a theoretical framework that models LLMs as Hierarchical Variational Information Bottleneck (H-VIB) systems. MSPGT posits that standard language modeling objectives implicitly optimize multi-scale information compression, leading to the spontaneous formation of three internal processing scales-Global, Intermediate, and Local. We formalize this principle, derive falsifiable predictions about boundary positions and architectural dependencies, and validate them through cross-model experiments combining multi-signal fusion and causal interventions. Results across Llama and Qwen families reveal consistent multi-scale organization but strong architecture-specific variations, partially supporting and refining the theory. MSPGT thus advances interpretability from descriptive observation toward predictive, information-theoretic understanding of how hierarchical structure emerges within large neural language models.
>
---
#### [replaced 014] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.01472v2](http://arxiv.org/pdf/2502.01472v2)**

> **作者:** Jinwei Hu; Zhenglin Huang; Xiangyu Yin; Wenjie Ruan; Guangliang Cheng; Yi Dong; Xiaowei Huang
>
> **备注:** Updated to the NeurIPS 2025 camera-ready version with minor revisions
>
> **摘要:** Large language models have been widely applied, but can inadvertently encode sensitive or harmful information, raising significant safety concerns. Machine unlearning has emerged to alleviate this concern; however, existing training-time unlearning approaches, relying on coarse-grained loss combinations, have limitations in precisely separating knowledge and balancing removal effectiveness with model utility. In contrast, we propose Fine-grained Activation manipuLation by Contrastive Orthogonal uNalignment (FALCON), a novel representation-guided unlearning approach that leverages information-theoretic guidance for efficient parameter selection, employs contrastive mechanisms to enhance representation separation, and projects conflict gradients onto orthogonal subspaces to resolve conflicts between forgetting and retention objectives. Extensive experiments demonstrate that FALCON achieves superior unlearning effectiveness while maintaining model utility, exhibiting robust resistance against knowledge recovery attempts.
>
---
#### [replaced 015] Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.14304v2](http://arxiv.org/pdf/2507.14304v2)**

> **作者:** Rakesh Paul; Anusha Kamath; Kanishk Singla; Raviraj Joshi; Utkarsh Vaidya; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs.
>
---
#### [replaced 016] On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collapse
- **分类: cs.LG; cs.AI; cs.CL; cs.DS; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.09642v3](http://arxiv.org/pdf/2411.09642v3)**

> **作者:** Alkis Kalavasis; Anay Mehrotra; Grigoris Velegkas
>
> **备注:** Accepted for presentation at the 57th Symposium on Theory of Computing (STOC 2025); v3 fixes typos
>
> **摘要:** Specifying all desirable properties of a language model is challenging, but certain requirements seem essential. Given samples from an unknown language, the trained model should produce valid strings not seen in training and be expressive enough to capture the language's full richness. Otherwise, outputting invalid strings constitutes "hallucination," and failing to capture the full range leads to "mode collapse." We ask if a language model can meet both requirements. We investigate this within a statistical language generation setting building on Gold and Angluin. Here, the model receives random samples from a distribution over an unknown language K, which belongs to a possibly infinite collection of languages. The goal is to generate unseen strings from K. We say the model generates from K with consistency and breadth if, as training size increases, its output converges to all unseen strings in K. Kleinberg and Mullainathan [KM24] asked if consistency and breadth in language generation are possible. We answer this negatively: for a large class of language models, including next-token prediction models, this is impossible for most collections of candidate languages. This contrasts with [KM24]'s result, showing consistent generation without breadth is possible for any countable collection of languages. Our finding highlights that generation with breadth fundamentally differs from generation without breadth. As a byproduct, we establish near-tight bounds on the number of samples needed for generation with or without breadth. Finally, our results offer hope: consistent generation with breadth is achievable for any countable collection of languages when negative examples (strings outside K) are available alongside positive ones. This suggests that post-training feedback, which encodes negative examples, can be crucial in reducing hallucinations while limiting mode collapse.
>
---
#### [replaced 017] Can Large Language Models Master Complex Card Games?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01328v4](http://arxiv.org/pdf/2509.01328v4)**

> **作者:** Wei Wang; Fuqing Bie; Junzhe Chen; Dan Zhang; Shiyu Huang; Evgeny Kharlamov; Jie Tang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Complex games have long been an important benchmark for testing the progress of artificial intelligence algorithms. AlphaGo, AlphaZero, and MuZero have defeated top human players in Go and Chess, garnering widespread societal attention towards artificial intelligence. Concurrently, large language models (LLMs) have exhibited remarkable capabilities across various tasks, raising the question of whether LLMs can achieve similar success in complex games. In this paper, we explore the potential of LLMs in mastering complex card games. We systematically assess the learning capabilities of LLMs across eight diverse card games, evaluating the impact of fine-tuning on high-quality gameplay data, and examining the models' ability to retain general capabilities while mastering these games. Our findings indicate that: (1) LLMs can approach the performance of strong game AIs through supervised fine-tuning on high-quality data, (2) LLMs can achieve a certain level of proficiency in multiple complex card games simultaneously, with performance augmentation for games with similar rules and conflicts for dissimilar ones, and (3) LLMs experience a decline in general capabilities when mastering complex games, but this decline can be mitigated by integrating a certain amount of general instruction data. The evaluation results demonstrate strong learning ability and versatility of LLMs. The code is available at https://github.com/THUDM/LLM4CardGame
>
---
#### [replaced 018] RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21936v3](http://arxiv.org/pdf/2505.21936v3)**

> **作者:** Zeyi Liao; Jaylen Jones; Linxi Jiang; Yuting Ning; Eric Fosler-Lussier; Yu Su; Zhiqiang Lin; Huan Sun
>
> **摘要:** Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning high ASRs in realistic end-to-end settings, with the strongest-to-date Claude 4.5 Sonnet | CUA exhibiting the highest ASR of 60%, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.
>
---
#### [replaced 019] ReasoningShield: Safety Detection over Reasoning Traces of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17244v2](http://arxiv.org/pdf/2505.17244v2)**

> **作者:** Changyi Li; Jiayi Wang; Xudong Pan; Geng Hong; Min Yang
>
> **摘要:** Large Reasoning Models (LRMs) leverage transparent reasoning traces, known as Chain-of-Thoughts (CoTs), to break down complex problems into intermediate steps and derive final answers. However, these reasoning traces introduce unique safety challenges: harmful content can be embedded in intermediate steps even when final answers appear benign. Existing moderation tools, designed to handle generated answers, struggle to effectively detect hidden risks within CoTs. To address these challenges, we introduce ReasoningShield, a lightweight yet robust framework for moderating CoTs in LRMs. Our key contributions include: (1) formalizing the task of CoT moderation with a multi-level taxonomy of 10 risk categories across 3 safety levels, (2) creating the first CoT moderation benchmark which contains 9.2K pairs of queries and reasoning traces, including a 7K-sample training set annotated via a human-AI framework and a rigorously curated 2.2K human-annotated test set, and (3) developing a two-stage training strategy that combines stepwise risk analysis and contrastive learning to enhance robustness. Experiments show that ReasoningShield achieves state-of-the-art performance, outperforming task-specific tools like LlamaGuard-4 by 35.6% and general-purpose commercial models like GPT-4o by 15.8% on benchmarks, while also generalizing effectively across diverse reasoning paradigms, tasks, and unseen scenarios. All resources are released at https://github.com/CosmosYi/ReasoningShield.
>
---
#### [replaced 020] From Next-Token to Mathematics: The Learning Dynamics of Mathematical Reasoning in Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.00900v3](http://arxiv.org/pdf/2407.00900v3)**

> **作者:** Shubhra Mishra; Gabriel Poesia; Noah D. Goodman
>
> **备注:** Accepted to COLM 2025. Dataset and code: https://github.com/gpoesia/mathcamps/
>
> **摘要:** Large Language Models (LLMs) solely trained on next-token prediction learn to solve a wide range of problems involving mathematical reasoning. But how does this ability evolve during training? We show the first analysis of how mathematical reasoning abilities of several open-weight LLMs develop during pre-training and post-training. To this end, we construct MathCAMPS, a synthetic dataset of novel mathematical reasoning problems grounded in 44 fine-grained skills taken from the Common Core curriculum from K to 8th grades. In one experiment, we show that mathematical skills are learned during pre-training in an order that measurably correlates with the human-designed curriculum, even though training data are randomly ordered. We also show a detailed analysis of which mathematical abilities benefit from instruction tuning, a widely used post-training method and, in contrast, which skills suffer. Our work paves the way for an empirical understanding of LLM training dynamics in relation to reasoning.
>
---
#### [replaced 021] ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13313v2](http://arxiv.org/pdf/2509.13313v2)**

> **作者:** Xixi Wu; Kuan Li; Yida Zhao; Liwen Zhang; Litu Ou; Huifeng Yin; Zhongwang Zhang; Xinmiao Yu; Dingchu Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Minhao Cheng; Shuai Wang; Hong Cheng; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Large Language Model (LLM)-based web agents demonstrate strong performance on knowledge-intensive tasks but are hindered by context window limitations in paradigms like ReAct. Complex queries involving multiple entities, intertwined relationships, and high uncertainty demand extensive search cycles that rapidly exhaust context budgets before reaching solutions. To overcome this challenge, we introduce ReSum, a novel paradigm that enables indefinite exploration through periodic context summarization. ReSum converts growing interaction histories into compact reasoning states, maintaining awareness of prior discoveries while bypassing context constraints. For paradigm adaptation, we propose ReSum-GRPO, integrating GRPO with segmented trajectory training and advantage broadcasting to familiarize agents with summary-conditioned reasoning. Extensive experiments on web agents across three benchmarks demonstrate that ReSum delivers an average absolute improvement of 4.5% over ReAct, with further gains of 8.2% following ReSum-GRPO training. Notably, with only 1K training samples, our WebResummer-30B (a ReSum-GRPO-trained version of WebSailor-30B) achieves 33.3% Pass@1 on BrowseComp-zh and 18.3% on BrowseComp-en, surpassing most open-source web agents.
>
---
#### [replaced 022] AI Realtor: Towards Grounded Persuasive Language Generation for Automated Copywriting
- **分类: cs.AI; cs.CL; cs.HC; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2502.16810v4](http://arxiv.org/pdf/2502.16810v4)**

> **作者:** Jibang Wu; Chenghao Yang; Yi Wu; Simon Mahns; Chaoqi Wang; Hao Zhu; Fei Fang; Haifeng Xu
>
> **备注:** V2: Add more human verification to ensure safety and examine potential hallucination. Significant reframing for the general audience. Website: https://yangalan123.github.io/ai-realtor/. Codebase: https://github.com/yangalan123/AI-Realtor-Codebase. Data released at Huggingface Hub (Sigma-Lab/AI_Realtor_xxx)
>
> **摘要:** This paper develops an agentic framework that employs large language models (LLMs) for grounded persuasive language generation in automated copywriting, with real estate marketing as a focal application. Our method is designed to align the generated content with user preferences while highlighting useful factual attributes. This agent consists of three key modules: (1) Grounding Module, mimicking expert human behavior to predict marketable features; (2) Personalization Module, aligning content with user preferences; (3) Marketing Module, ensuring factual accuracy and the inclusion of localized features. We conduct systematic human-subject experiments in the domain of real estate marketing, with a focus group of potential house buyers. The results demonstrate that marketing descriptions generated by our approach are preferred over those written by human experts by a clear margin while maintaining the same level of factual accuracy. Our findings suggest a promising agentic approach to automate large-scale targeted copywriting while ensuring factuality of content generation.
>
---
#### [replaced 023] TASER: Table Agents for Schema-guided Extraction and Recommendation
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.13404v3](http://arxiv.org/pdf/2508.13404v3)**

> **作者:** Nicole Cho; Kirsty Fielding; William Watson; Sumitra Ganesh; Manuela Veloso
>
> **摘要:** Real-world financial documents report essential information about an entity's financial holdings that can span millions of different financial instrument types. Yet, these details are often buried in messy, multi-page, fragmented tables - for example, 99.4% of the tables in our dataset have no bounding boxes with the maximum number of rows amounting to 426 per table across 44 pages. To tackle these unique challenges from real-world tables, we present a continuously learning, agentic table extraction system, TASER (Table Agents for Schema-guided Extraction and Recommendation) that extracts highly unstructured, multi-page, heterogeneous tables into normalized, schema-conforming outputs. Our table agents execute on table detection, classification, extraction, and recommendations by leveraging an initial schema. Then, our Recommender Agent reviews the outputs, recommends schema revisions, and decides on the final recommendations, enabling TASER to outperform existing table detection models such as Table Transformer by 10.1%. Within this continuous learning process, we highlight that larger batch sizes result in a 104.3% increase in schema recommendations that are actionable and utilized, resulting in a 9.8% increase in extracted holdings - highlighting the importance of a continuous learning process. To train TASER, we have manually labeled 22,584 pages (28,150,449 tokens), 3,213 tables for $731,685,511,687 of holdings culminating in one of the first real financial table datasets. We release our dataset TASERTab to enable the research community to access real-world financial tables and outputs. Our results highlight the promise of agentic, schema-guided extraction systems for robust understanding of real-world financial tables.
>
---
#### [replaced 024] SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02803v3](http://arxiv.org/pdf/2506.02803v3)**

> **作者:** Sifan Li; Yujun Cai; Yiwei Wang
>
> **摘要:** Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden text, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0-5.36%)-even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions (32-128 pixels), which unlocks >99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond.
>
---
#### [replaced 025] On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00597v3](http://arxiv.org/pdf/2504.00597v3)**

> **作者:** Jirui Qi; Raquel Fernández; Arianna Bisazza
>
> **备注:** MRL Workshop 2025, co-located with EMNLP 2025. All codes and data are released at https://github.com/Betswish/mRAG-Context-Consistency
>
> **摘要:** Retrieval-augmented generation (RAG) with large language models (LLMs) has demonstrated strong performance in multilingual question-answering (QA) tasks by leveraging relevant passages retrieved from corpora. In multilingual RAG (mRAG), the retrieved passages can be written in languages other than that of the query entered by the user, making it challenging for LLMs to effectively utilize the provided information. Recent research suggests that retrieving passages from multilingual corpora can improve RAG performance, particularly for low-resource languages. However, the extent to which LLMs can leverage different kinds of multilingual contexts to generate accurate answers, *independently from retrieval quality*, remains understudied. In this paper, we conduct an extensive assessment of LLMs' ability to (i) make consistent use of a relevant passage regardless of its language, (ii) respond in the expected language, and (iii) focus on the relevant passage even when multiple `distracting' passages in different languages are provided in the context. Our experiments with four LLMs across three QA datasets covering a total of 48 languages reveal a surprising ability of LLMs to extract the relevant information from passages in a different language than the query, but a much weaker ability to formulate a full answer in the correct language. Our analysis, based on both accuracy and feature attribution techniques, further shows that distracting passages negatively impact answer quality regardless of their language. However, distractors in the query language exert a slightly stronger influence. Taken together, our findings deepen the understanding of how LLMs utilize context in mRAG systems, providing directions for future improvements.
>
---
#### [replaced 026] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.20712v4](http://arxiv.org/pdf/2509.20712v4)**

> **作者:** Zhenpeng Su; Leiyu Pan; Minxuan Lv; Yuntao Li; Wenping Hu; Fuzheng Zhang; Kun Gai; Guorui Zhou
>
> **摘要:** Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}oordinating \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.
>
---
#### [replaced 027] Improving Text-to-Image Generation with Input-Side Inference-Time Scaling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.12041v2](http://arxiv.org/pdf/2510.12041v2)**

> **作者:** Ruibo Chen; Jiacheng Pan; Heng Huang; Zhenheng Yang
>
> **摘要:** Recent advances in text-to-image (T2I) generation have achieved impressive results, yet existing models often struggle with simple or underspecified prompts, leading to suboptimal image-text alignment, aesthetics, and quality. We propose a prompt rewriting framework that leverages large language models (LLMs) to refine user inputs before feeding them into T2I backbones. Our approach introduces a carefully designed reward system and an iterative direct preference optimization (DPO) training pipeline, enabling the rewriter to enhance prompts without requiring supervised fine-tuning data. We evaluate our method across diverse T2I models and benchmarks. Results show that our prompt rewriter consistently improves image-text alignment, visual quality, and aesthetics, outperforming strong baselines. Furthermore, we demonstrate strong transferability by showing that a prompt rewriter trained on one T2I backbone generalizes effectively to others without needing to be retrained. We also systematically study scalability, evaluating how performance gains scale with the capacity of the large LLM used as the rewriter. These findings highlight that prompt rewriting is an effective, scalable, and practical model-agnostic strategy for improving T2I systems. We plan to release the code and trained prompt rewriters soon.
>
---
#### [replaced 028] Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08615v2](http://arxiv.org/pdf/2510.08615v2)**

> **作者:** Kaiqi Yang; Hang Li; Yucheng Chu; Zitao Liu; Mi Tian; Hui Liu
>
> **摘要:** Mathematical reasoning serves as a crucial testbed for evaluating the intelligence of large language models (LLMs), and math word problems (MWPs) represent one of the most widely used formats. Most existing MWP datasets contain only the necessary information, while problems with distracting or excessive conditions are often overlooked. Prior studies have shown that popular LLMs experience a dramatic performance drop when such distracting conditions are introduced. However, available datasets of MWPs with distracting conditions remain limited, and most exhibit low difficulty and out-of-context expressions. These shortcomings make the distracting conditions easy to detect and disregard, thereby reducing the credibility of benchmarking on these datasets. Moreover, when distracting conditions are added, the reasoning process and answers may change, requiring intensive manual effort to check and rewrite solutions. To address these issues, we design an iterative framework that leverages LLMs to generate distracting conditions automatically. We develop a set of prompts to revise MWPs from multiple perspectives and cognitive levels, encouraging the creation of meaningful distracting conditions as well as suggestions for further refinement. A key advantage of our framework is the preservation of shared solutions between the original and revised problems: the LLMs are explicitly guided to generate distractions that do not alter the original solution, thus eliminating the need to produce new answers. This framework is efficient and easy to deploy, substantially reducing the effort required to generate MWPs with distracting conditions while maintaining high data quality.
>
---
#### [replaced 029] PAL: Probing Audio Encoders via LLMs - Audio Information Transfer into LLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10423v2](http://arxiv.org/pdf/2506.10423v2)**

> **作者:** Tony Alex; Wish Suharitdamrong; Sara Atito; Armin Mustafa; Philip J. B. Jackson; Imran Razzak; Muhammad Awais
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Integration of audio perception into large language models (LLMs) is an emerging research area for enabling machine listening applications, yet efficient transfer of rich audio semantics from audio encoders to LLMs remains underexplored. The most widely used integration paradigm projects the audio encoder output tokens into the LLM input space (e.g., via an MLP or a Q-Former), then prepends or inserts them to the text tokens. We refer to this generic scheme as Prepend to the LLM's input token space (PLITS) integration. We propose an efficient alternative, Lightweight Audio LLM Integration (LAL). LAL introduces audio representations solely via the attention mechanism within different layers of the LLM, bypassing its feedforward module. LAL encodes rich audio semantics at an appropriate level of abstraction for integration into different blocks of LLMs. Our design significantly reduces computational overhead compared to existing integration approaches. Observing with Whisper that the speech encoder benefits from PLITS integration, we propose an audio encoder aware approach for efficiently Probing Audio encoders via LLM (PAL), which employs PLITS integration for Whisper and LAL for general audio encoders. Under an identical training curriculum, LAL consistently maintains performance or outperforms existing integration approaches across multiple base LLMs and tasks. For general audio tasks, LAL improvement is up to 30% over a strong PLITS baseline while reducing memory usage by up to 64.1% and increasing throughput by up to 247.5%. Furthermore, for general audio-music-speech LLM, PAL performs on par with a fully PLITS integration-based system but with substantially improved computational and memory efficiency. Project page: https://ta012.github.io/PAL/
>
---
#### [replaced 030] Sri Lanka Document Datasets: A Large-Scale, Multilingual Resource forLaw, News, and Policy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04124v2](http://arxiv.org/pdf/2510.04124v2)**

> **作者:** Nuwan I. Senaratna
>
> **备注:** 4 pages. 229,858 documents (57.1 GB) across 24 datasets in Sinhala, Tamil, and English. Last updated on 2025-10-15-1111
>
> **摘要:** We present a collection of open, machine-readable document datasets covering parliamentary proceedings, legal judgments, government publications, news, and tourism statistics from Sri Lanka. The collection currently comprises of 229,858 documents (57.1 GB) across 24 datasets in Sinhala, Tamil, and English. The datasets are updated daily and mirrored on GitHub and Hugging Face. These resources aim to support research in computational linguistics, legal analytics, socio-political studies, and multilingual natural language processing. We describe the data sources, collection pipeline, formats, and potential use cases, while discussing licensing and ethical considerations. This manuscript is at version v2025-10-15-1111.
>
---
#### [replaced 031] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19972v3](http://arxiv.org/pdf/2508.19972v3)**

> **作者:** Seongheon Park; Sharon Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Object hallucination in large vision-language models presents a significant challenge to their safe deployment in real-world applications. Recent works have proposed object-level hallucination scores to estimate the likelihood of object hallucination; however, these methods typically adopt either a global or local perspective in isolation, which may limit detection reliability. In this paper, we introduce GLSim, a novel training-free object hallucination detection framework that leverages complementary global and local embedding similarity signals between image and text modalities, enabling more accurate and reliable hallucination detection in diverse scenarios. We comprehensively benchmark existing object hallucination detection methods and demonstrate that GLSim achieves superior detection performance, outperforming competitive baselines by a significant margin.
>
---
#### [replaced 032] FineScope : Precision Pruning for Domain-Specialized Large Language Models Using SAE-Guided Self-Data Cultivation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00624v2](http://arxiv.org/pdf/2505.00624v2)**

> **作者:** Chaitali Bhattacharyya; Hyunsei Lee; Junyoung Lee; Shinhyoung Jang; Il hong Suh; Yeseong Kim
>
> **摘要:** Training large language models (LLMs) from scratch requires significant computational resources, driving interest in developing smaller, domain-specific LLMs that maintain both efficiency and strong task performance. Medium-sized models such as LLaMA, llama} have served as starting points for domain-specific adaptation, but they often suffer from accuracy degradation when tested on specialized datasets. We introduce FineScope, a framework for deriving compact, domain-optimized LLMs from larger pretrained models. FineScope leverages the Sparse Autoencoder (SAE) framework, inspired by its ability to produce interpretable feature representations, to extract domain-specific subsets from large datasets. We apply structured pruning with domain-specific constraints, ensuring that the resulting pruned models retain essential knowledge for the target domain. To further enhance performance, these pruned models undergo self-data distillation, leveraging SAE-curated datasets to restore key domain-specific information lost during pruning. Extensive experiments and ablation studies demonstrate that FineScope achieves highly competitive performance, outperforming several large-scale state-of-the-art LLMs in domain-specific tasks. Additionally, our results show that FineScope enables pruned models to regain a substantial portion of their original performance when fine-tuned with SAE-curated datasets. Furthermore, applying these datasets to fine-tune pretrained LLMs without pruning also improves their domain-specific accuracy, highlighting the robustness of our approach.
>
---
#### [replaced 033] SeLeRoSa: Sentence-Level Romanian Satire Detection Dataset
- **分类: cs.CL; I.2.7; I.7**

- **链接: [http://arxiv.org/pdf/2509.00893v2](http://arxiv.org/pdf/2509.00893v2)**

> **作者:** Răzvan-Alexandru Smădu; Andreea Iuga; Dumitru-Clementin Cercel; Florin Pop
>
> **备注:** 9 pages, 2 Figures
>
> **摘要:** Satire, irony, and sarcasm are techniques typically used to express humor and critique, rather than deceive; however, they can occasionally be mistaken for factual reporting, akin to fake news. These techniques can be applied at a more granular level, allowing satirical information to be incorporated into news articles. In this paper, we introduce the first sentence-level dataset for Romanian satire detection for news articles, called SeLeRoSa. The dataset comprises 13,873 manually annotated sentences spanning various domains, including social issues, IT, science, and movies. With the rise and recent progress of large language models (LLMs) in the natural language processing literature, LLMs have demonstrated enhanced capabilities to tackle various tasks in zero-shot settings. We evaluate multiple baseline models based on LLMs in both zero-shot and fine-tuning settings, as well as baseline transformer-based models. Our findings reveal the current limitations of these models in the sentence-level satire detection task, paving the way for new research directions.
>
---
#### [replaced 034] Evaluating and Mitigating Social Bias for Large Language Models in Open-ended Settings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06134v4](http://arxiv.org/pdf/2412.06134v4)**

> **作者:** Zhao Liu; Tian Xie; Xueru Zhang
>
> **备注:** 15 pages
>
> **摘要:** Current social bias benchmarks for Large Language Models (LLMs) primarily rely on predefined question formats like multiple-choice, limiting their ability to reflect the complexity and open-ended nature of real-world interactions. To close this gap, we extend an existing dataset BBQ (Parrish et al., 2022) to Open-BBQ, a comprehensive framework to evaluate the social bias of LLMs in open-ended settings by incorporating two additional question categories: fill-in-the-blank and short-answer. Since our new Open-BBQ dataset contains a lot of open-ended responses like sentences and paragraphs, we developed an evaluation process to detect biases from open-ended content by labeling sentences and paragraphs. In addition to this, we also found that existing debiasing methods, such as self-debiasing (Gallegos et al., 2024), have over-correction issues, which make the original correct answers incorrect. In order to solve this issue, we propose Composite Prompting, an In-context Learning (ICL) method combining structured examples with explicit chain-of-thought reasoning to form a unified instruction template for LLMs to explicitly identify content that needs debiasing. Experimental results show that the proposed method significantly reduces the bias for both GPT-3.5 and GPT-4o while maintaining high accuracy.
>
---
#### [replaced 035] FlashAdventure: A Benchmark for GUI Agents Solving Full Story Arcs in Diverse Adventure Games
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01052v2](http://arxiv.org/pdf/2509.01052v2)**

> **作者:** Jaewoo Ahn; Junseo Kim; Heeseung Yun; Jaehyeon Son; Dongmin Park; Jaewoong Cho; Gunhee Kim
>
> **备注:** EMNLP 2025 Main. Project page: https://ahnjaewoo.github.io/flashadventure
>
> **摘要:** GUI agents powered by LLMs show promise in interacting with diverse digital environments. Among these, video games offer a valuable testbed due to their varied interfaces, with adventure games posing additional challenges through complex, narrative-driven interactions. Existing game benchmarks, however, lack diversity and rarely evaluate agents on completing entire storylines. To address this, we introduce FlashAdventure, a benchmark of 34 Flash-based adventure games designed to test full story arc completion and tackle the observation-behavior gap: the challenge of remembering and acting on earlier gameplay information. We also propose CUA-as-a-Judge, an automated gameplay evaluator, and COAST, an agentic framework leveraging long-term clue memory to better plan and solve sequential tasks. Experiments show current GUI agents struggle with full story arcs, while COAST improves milestone completion by bridging the observation-behavior gap. Nonetheless, a marked discrepancy between humans and best-performing agents warrants continued research efforts to narrow this divide.
>
---
#### [replaced 036] Translating Regulatory Clauses into Executable Codes for Building Design Checking via Large Language Model Driven Function Matching and Composing
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2308.08728v2](http://arxiv.org/pdf/2308.08728v2)**

> **作者:** Zhe Zheng; Jin Han; Ke-Yin Chen; Xin-Yu Cao; Xin-Zheng Lu; Jia-Rui Lin
>
> **摘要:** Translating clauses into executable code is a vital stage of automated rule checking (ARC) and is essential for effective building design compliance checking, particularly for rules with implicit properties or complex logic requiring domain knowledge. Thus, by systematically analyzing building clauses, 66 atomic functions are defined first to encapsulate common computational logics. Then, LLM-FuncMapper is proposed, a large language model (LLM)-based approach with rule-based adaptive prompts that match clauses to atomic functions. Finally, executable code is generated by composing functions through the LLMs. Experiments show LLM-FuncMapper outperforms fine-tuning methods by 19% in function matching while significantly reducing manual annotation efforts. Case study demonstrates that LLM-FuncMapper can automatically compose multiple atomic functions to generate executable code, boosting rule-checking efficiency. To our knowledge, this research represents the first application of LLMs for interpreting complex design clauses into executable code, which may shed light on further adoption of LLMs in the construction domain.
>
---
#### [replaced 037] BTC-SAM: Leveraging LLMs for Generation of Bias Test Cases for Sentiment Analysis Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.24101v2](http://arxiv.org/pdf/2509.24101v2)**

> **作者:** Zsolt T. Kardkovacs; Lynda Djennane; Anna Field; Boualem Benatallah; Yacine Gaci; Fabio Casati; Walid Gaaloul
>
> **备注:** Accepted at EMNLP 2025 main conference
>
> **摘要:** Sentiment Analysis (SA) models harbor inherent social biases that can be harmful in real-world applications. These biases are identified by examining the output of SA models for sentences that only vary in the identity groups of the subjects. Constructing natural, linguistically rich, relevant, and diverse sets of sentences that provide sufficient coverage over the domain is expensive, especially when addressing a wide range of biases: it requires domain experts and/or crowd-sourcing. In this paper, we present a novel bias testing framework, BTC-SAM, which generates high-quality test cases for bias testing in SA models with minimal specification using Large Language Models (LLMs) for the controllable generation of test sentences. Our experiments show that relying on LLMs can provide high linguistic variation and diversity in the test sentences, thereby offering better test coverage compared to base prompting methods even for previously unseen biases.
>
---
#### [replaced 038] Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04886v4](http://arxiv.org/pdf/2507.04886v4)**

> **作者:** A. Bochkov
>
> **备注:** Published in Transactions on Machine Learning Research (10/2025). OpenReview: https://openreview.net/forum?id=Odh8IynO1o
>
> **摘要:** Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research.
>
---
#### [replaced 039] PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12119v2](http://arxiv.org/pdf/2502.12119v2)**

> **作者:** Jinhe Bi; Yifan Wang; Danqi Yan; Aniri; Wenke Huang; Zengjie Jin; Xiaowen Ma; Artur Hecker; Mang Ye; Xun Xiao; Hinrich Schuetze; Volker Tresp; Yunpu Ma
>
> **摘要:** Visual instruction tuning adapts pre-trained Multimodal Large Language Models (MLLMs) to follow human instructions for real-world applications. However, the rapid growth of these datasets introduces significant redundancy, leading to increased computational costs. Existing methods for selecting instruction data aim to prune this redundancy, but predominantly rely on computationally demanding techniques such as proxy-based inference or training-based metrics. Consequently, the substantial computational costs incurred by these selection processes often exacerbate the very efficiency bottlenecks they are intended to resolve, posing a significant challenge to the scalable and effective tuning of MLLMs. To address this challenge, we first identify a critical, yet previously overlooked, factor: the anisotropy inherent in visual feature distributions. We find that this anisotropy induces a \textit{Global Semantic Drift}, and overlooking this phenomenon is a key factor limiting the efficiency of current data selection methods. Motivated by this insight, we devise \textbf{PRISM}, the first training-free framework for efficient visual instruction selection. PRISM surgically removes the corrupting influence of global background features by modeling the intrinsic visual semantics via implicit re-centering. Empirically, PRISM reduces the end-to-end time for data selection and model tuning to just 30\% of conventional pipelines. More remarkably, it achieves this efficiency while simultaneously enhancing performance, surpassing models fine-tuned on the full dataset across eight multimodal and three language understanding benchmarks, culminating in a 101.7\% relative improvement over the baseline. The code is available for access via \href{https://github.com/bibisbar/PRISM}{this repository}.
>
---
#### [replaced 040] Your AI, Not Your View: The Bias of LLMs in Investment Analysis
- **分类: q-fin.PM; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20957v3](http://arxiv.org/pdf/2507.20957v3)**

> **作者:** Hoyoung Lee; Junhyuk Seo; Suhwan Park; Junhyeong Lee; Wonbin Ahn; Chanyeol Choi; Alejandro Lopez-Lira; Yongjae Lee
>
> **备注:** Accepted at ACM International Conference on AI in Finance (ICAIF)
>
> **摘要:** In finance, Large Language Models (LLMs) face frequent knowledge conflicts arising from discrepancies between their pre-trained parametric knowledge and real-time market data. These conflicts are especially problematic in real-world investment services, where a model's inherent biases can misalign with institutional objectives, leading to unreliable recommendations. Despite this risk, the intrinsic investment biases of LLMs remain underexplored. We propose an experimental framework to investigate emergent behaviors in such conflict scenarios, offering a quantitative analysis of bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent biases of models and measure their persistence. Our analysis, centered on sector, size, and momentum, reveals distinct, model-specific biases. Across most models, a tendency to prefer technology stocks, large-cap stocks, and contrarian strategies is observed. These foundational biases often escalate into confirmation bias, causing models to cling to initial judgments even when faced with increasing counter-evidence. A public leaderboard benchmarking bias across a broader set of models is available at https://linqalpha.com/leaderboard
>
---
#### [replaced 041] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10412v4](http://arxiv.org/pdf/2506.10412v4)**

> **作者:** Ching Chang; Jeehyun Hwang; Yidan Shi; Haixin Wang; Wen-Chih Peng; Tien-Fu Chen; Wei Wang
>
> **备注:** This paper has been accepted by the NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Time series data in real-world applications such as healthcare, climate modeling, and finance are often irregular, multimodal, and messy, with varying sampling rates, asynchronous modalities, and pervasive missingness. However, existing benchmarks typically assume clean, regularly sampled, unimodal data, creating a significant gap between research and real-world deployment. We introduce Time-IMM, a dataset specifically designed to capture cause-driven irregularity in multimodal multivariate time series. Time-IMM represents nine distinct types of time series irregularity, categorized into trigger-based, constraint-based, and artifact-based mechanisms. Complementing the dataset, we introduce IMM-TSF, a benchmark library for forecasting on irregular multimodal time series, enabling asynchronous integration and realistic evaluation. IMM-TSF includes specialized fusion modules, including a timestamp-to-text fusion module and a multimodality fusion module, which support both recency-aware averaging and attention-based integration strategies. Empirical results demonstrate that explicitly modeling multimodality on irregular time series data leads to substantial gains in forecasting performance. Time-IMM and IMM-TSF provide a foundation for advancing time series analysis under real-world conditions. The dataset is publicly available at https://github.com/blacksnail789521/Time-IMM, and the benchmark library can be accessed at https://github.com/blacksnail789521/IMM-TSF. Project page: https://blacksnail789521.github.io/time-imm-project-page/
>
---
#### [replaced 042] Studying the Korean Word-Chain Game with RLVR: Mitigating Reward Conflicts via Curriculum Learning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03394v2](http://arxiv.org/pdf/2510.03394v2)**

> **作者:** Donghwan Rho
>
> **备注:** 10 pages
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) is a promising approach for training large language models (LLMs) with stronger reasoning abilities. It has also been applied to a variety of logic puzzles. In this work, we study the Korean word-chain game using RLVR. We show that rule-derived rewards can naturally conflict, and demonstrate through experiments that a curriculum-learning scheme mitigates these conflicts. Our findings motivate further studies of puzzle tasks in diverse languages.
>
---
#### [replaced 043] MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08567v2](http://arxiv.org/pdf/2510.08567v2)**

> **作者:** Tajamul Ashraf; Umair Nawaz; Abdelrahman M. Shaker; Rao Anwer; Philip Torr; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** https://mbzuai-oryx.github.io/MATRIX/
>
> **摘要:** Vision language models (VLMs) are increasingly deployed as controllers with access to external tools for complex reasoning and decision-making, yet their effectiveness remains limited by the scarcity of high-quality multimodal trajectories and the cost of manual annotation. We address this challenge with a vision-centric agent tuning framework that automatically synthesizes multimodal trajectories, generates step-wise preference pairs, and trains a VLM controller for robust tool-use reasoning. Our pipeline first constructs M-TRACE, a large-scale dataset of 28.5K multimodal tasks with 177K verified trajectories, enabling imitation-based trajectory tuning. Building on this, we develop MATRIX Agent, a controller finetuned on M-TRACE for step-wise tool reasoning. To achieve finer alignment, we further introduce Pref-X, a set of 11K automatically generated preference pairs, and optimize MATRIX on it via step-wise preference learning. Across three benchmarks, Agent-X, GTA, and GAIA, MATRIX consistently surpasses both open- and closed-source VLMs, demonstrating scalable and effective multimodal tool use. Our data and code is avaliable at https://github.com/mbzuai-oryx/MATRIX.
>
---
#### [replaced 044] Investigating the interaction of linguistic and mathematical reasoning in language models using multilingual number puzzles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13886v2](http://arxiv.org/pdf/2506.13886v2)**

> **作者:** Antara Raaghavi Bhattacharya; Isabel Papadimitriou; Kathryn Davidson; David Alvarez-Melis
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Across languages, numeral systems vary widely in how they construct and combine numbers. While humans consistently learn to navigate this diversity, large language models (LLMs) struggle with linguistic-mathematical puzzles involving cross-linguistic numeral systems, which humans can learn to solve successfully. We investigate why this task is difficult for LLMs through a series of experiments that untangle the linguistic and mathematical aspects of numbers in language. Our experiments establish that models cannot consistently solve such problems unless the mathematical operations in the problems are explicitly marked using known symbols ($+$, $\times$, etc., as in "twenty + three"). In further ablation studies, we probe how individual parameters of numeral construction and combination affect performance. While humans use their linguistic understanding of numbers to make inferences about the implicit compositional structure of numerals, LLMs seem to lack this notion of implicit numeral structure. We conclude that the ability to flexibly infer compositional rules from implicit patterns in human-scale data remains an open challenge for current reasoning models.
>
---
#### [replaced 045] KG2QA: Knowledge Graph-enhanced Retrieval-augmented Generation for Communication Standards Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07037v2](http://arxiv.org/pdf/2506.07037v2)**

> **作者:** Zhongze Luo; Weixuan Wan; Tianya Zhang; Dan Wang; Xiaoying Tang
>
> **摘要:** The rapid evolution of communication technologies has led to an explosion of standards, rendering traditional expert-dependent consultation methods inefficient and slow. To address this challenge, we propose \textbf{KG2QA}, a question answering (QA) framework for communication standards that integrates fine-tuned large language models (LLMs) with a domain-specific knowledge graph (KG) via a retrieval-augmented generation (RAG) pipeline. We construct a high-quality dataset of 6,587 QA pairs from ITU-T recommendations and fine-tune Qwen2.5-7B-Instruct, achieving significant performance gains: BLEU-4 increases from 18.86 to 66.90, outperforming both the base model and Llama-3-8B-Instruct. A structured KG containing 13,906 entities and 13,524 relations is built using LLM-assisted triple extraction based on a custom ontology. In our KG-RAG pipeline, the fine-tuned LLMs first retrieves relevant knowledge from KG, enabling more accurate and factually grounded responses. Evaluated by DeepSeek-V3 as a judge, the KG-enhanced system improves performance across five dimensions, with an average score increase of 2.26\%, demonstrating superior factual accuracy and relevance. Integrated with Web platform and API, KG2QA delivers an efficient and interactive user experience. Our code and data have been open-sourced https://github.com/luozhongze/KG2QA.
>
---
#### [replaced 046] Improving Zero-shot Sentence Decontextualisation with Content Selection and Planning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17921v2](http://arxiv.org/pdf/2509.17921v2)**

> **作者:** Zhenyun Deng; Yulong Chen; Andreas Vlachos
>
> **备注:** Accepted to EMLNP 2025 (Main Conference)
>
> **摘要:** Extracting individual sentences from a document as evidence or reasoning steps is commonly done in many NLP tasks. However, extracted sentences often lack context necessary to make them understood, e.g., coreference and background information. To this end, we propose a content selection and planning framework for zero-shot decontextualisation, which determines what content should be mentioned and in what order for a sentence to be understood out of context. Specifically, given a potentially ambiguous sentence and its context, we first segment it into basic semantically-independent units. We then identify potentially ambiguous units from the given sentence, and extract relevant units from the context based on their discourse relations. Finally, we generate a content plan to rewrite the sentence by enriching each ambiguous unit with its relevant units. Experimental results demonstrate that our approach is competitive for sentence decontextualisation, producing sentences that exhibit better semantic integrity and discourse coherence, outperforming existing methods.
>
---
#### [replaced 047] Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11052v2](http://arxiv.org/pdf/2510.11052v2)**

> **作者:** Qinglin Zhu; Yizhen Yao; Runcong Zhao; Yanzheng Xiang; Amrutha Saseendran; Chen Jin; Philip Teare; Bin Liang; Yulan He; Lin Gui
>
> **摘要:** Autoregressive (AR) models remain the standard for natural language generation but still suffer from high latency due to strictly sequential decoding. Recent diffusion-inspired approaches, such as LlaDA and Dream, mitigate this by generating in parallel, yet they suffer from two core limitations: information loss, as predictive distributions for non-finalized tokens are discarded at each step, and premature commitment, where local decisions are made without sufficient global coordination. We introduce Latent Refinement Decoding (LRD), a two-stage framework with Latent Refinement and a Predictive Feedback Loop. The first stage maintains masked positions as distributional mixtures of predicted tokens and the mask embedding, allowing the model to establish more globally consistent beliefs. The second stage progressively finalizes confident tokens while retaining uncertain ones for iterative feedback. KL-divergence dynamics provide a principled and reliable criterion for convergence and early stopping. Experiments across coding (HumanEval +6.3, MBPP +2.6) and reasoning (GSM8K +2.9, MATH500 +3.8) show that LRD improves accuracy while delivering speedups of up to 10.6x, making it a strong and versatile alternative for parallel sequence generation.
>
---
#### [replaced 048] R$^2$ec: Towards Large Recommender Models with Reasoning
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16994v2](http://arxiv.org/pdf/2505.16994v2)**

> **作者:** Runyang You; Yongqi Li; Xinyu Lin; Xin Zhang; Wenjie Wang; Wenjie Li; Liqiang Nie
>
> **摘要:** Large recommender models have extended LLMs as powerful recommenders via encoding or item generation, and recent breakthroughs in LLM reasoning synchronously motivate the exploration of reasoning in recommendation. In this work, we propose R$^2$ec, a unified large recommender model with intrinsic reasoning capability. R$^2$ec introduces a dual-head architecture that supports both reasoning chain generation and efficient item prediction in a single model, significantly reducing inference latency. To overcome the lack of annotated reasoning data, we design RecPO, a reinforcement learning framework that optimizes reasoning and recommendation jointly with a novel fused reward mechanism. Extensive experiments on three datasets demonstrate that R$^2$ec outperforms traditional, LLM-based, and reasoning-augmented recommender baselines, while further analyses validate its competitive efficiency among conventional LLM-based recommender baselines and strong adaptability to diverse recommendation scenarios. Code and checkpoints available at https://github.com/YRYangang/RRec.
>
---
#### [replaced 049] Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.02340v2](http://arxiv.org/pdf/2510.02340v2)**

> **作者:** Xin Gao; Ruiyi Zhang; Daniel Du; Saurabh Mahindre; Sai Ashish Somayajula; Pengtao Xie
>
> **备注:** Published at EMNLP 2025; Code and data available at https://github.com/gxx27/time_unlearn
>
> **摘要:** Large Language Models (LLMs) are widely used for temporal prediction, but their reliance on pretraining data raises contamination concerns, as accurate predictions on pre-cutoff test data may reflect memorization rather than reasoning, leading to an overestimation of their generalization capability. With the recent emergence of prompting-based unlearning techniques, a natural question arises: Can LLMs be prompted to simulate an earlier knowledge cutoff? In this work, we investigate the capability of prompting to simulate earlier knowledge cutoff in LLMs. We construct three evaluation datasets to assess the extent to which LLMs can forget (1) direct factual knowledge, (2) semantic shifts, and (3) causally related knowledge. Results demonstrate that while prompt-based simulated knowledge cutoffs show effectiveness when directly queried with the information after that date, they struggle to induce forgetting when the forgotten content is not directly asked but causally related to the query. These findings highlight the need for more rigorous evaluation settings when applying LLMs for temporal prediction tasks. The full dataset and evaluation code are available at https://github.com/gxx27/time_unlearn.
>
---
#### [replaced 050] Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12470v3](http://arxiv.org/pdf/2502.12470v3)**

> **作者:** Alireza S. Ziabari; Nona Ghazizadeh; Zhivar Sourati; Farzan Karimi-Malekabadi; Payam Piray; Morteza Dehghani
>
> **摘要:** Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. In contrast, human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context. This difference between human cognitive flexibility and LLMs' reliance on a single reasoning style raises a critical question: while human fast heuristic reasoning evolved for its efficiency and adaptability, is a uniform reasoning approach truly optimal for LLMs, or does its inflexibility make them brittle and unreliable when faced with tasks demanding more agile, intuitive responses? To answer these questions, we explicitly align LLMs to these reasoning styles by curating a dataset with valid System 1 and System 2 answers, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense reasoning tasks. To analyze the reasoning spectrum, we interpolated between the two extremes by varying the proportion of alignment data, which resulted in a monotonic change in accuracy. A mechanistic analysis of model responses shows that System 1 models employ more definitive outputs, whereas System 2 models demonstrate greater uncertainty. Building on these findings, we further combine System 1- and System 2-aligned models based on the entropy of their generations, without additional training, and obtain a dynamic model that outperforms across nearly all benchmarks. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.
>
---
#### [replaced 051] ICA-RAG: Information Completeness Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14614v5](http://arxiv.org/pdf/2502.14614v5)**

> **作者:** Jiawei He; Mingyi Jia; Zhihao Jia; Junwen Duan; Yan Song; Jianxin Wang
>
> **摘要:** Retrieval-Augmented Large Language Models (LLMs), which integrate external knowledge, have shown remarkable performance in medical domains, including clinical diagnosis. However, existing RAG methods often struggle to tailor retrieval strategies to diagnostic difficulty and input sample informativeness. This limitation leads to excessive and often unnecessary retrieval, impairing computational efficiency and increasing the risk of introducing noise that can degrade diagnostic accuracy. To address this, we propose ICA-RAG (\textbf{I}nformation \textbf{C}ompleteness Guided \textbf{A}daptive \textbf{R}etrieval-\textbf{A}ugmented \textbf{G}eneration), a novel framework for enhancing RAG reliability in disease diagnosis. ICA-RAG utilizes an adaptive control module to assess the necessity of retrieval based on the input's information completeness. By optimizing retrieval and incorporating knowledge filtering, ICA-RAG better aligns retrieval operations with clinical requirements. Experiments on three Chinese electronic medical record datasets demonstrate that ICA-RAG significantly outperforms baseline methods, highlighting its effectiveness in clinical diagnosis.
>
---
#### [replaced 052] Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.14896v2](http://arxiv.org/pdf/2508.14896v2)**

> **作者:** Haokun Lin; Haobo Xu; Yichen Wu; Ziyu Guo; Renrui Zhang; Zhichao Lu; Ying Wei; Qingfu Zhang; Zhenan Sun
>
> **备注:** Technical Report, Work in Progress
>
> **摘要:** Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. Our code is publicly available at https://github.com/FelixMessi/QDLM.
>
---
#### [replaced 053] Cross-modal Associations in Vision and Language Models: Revisiting the Bouba-Kiki Effect
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10013v2](http://arxiv.org/pdf/2507.10013v2)**

> **作者:** Tom Kouwenhoven; Kiana Shahrasbi; Tessa Verhoef
>
> **备注:** Presented at the Thirty-Ninth Annual Conference on Neural Information Processing Systems (2025)
>
> **摘要:** Recent advances in multimodal models have raised questions about whether vision-and-language models (VLMs) integrate cross-modal information in ways that reflect human cognition. One well-studied test case in this domain is the bouba-kiki effect, where humans reliably associate pseudowords like `bouba' with round shapes and `kiki' with jagged ones. Given the mixed evidence found in prior studies for this effect in VLMs, we present a comprehensive re-evaluation focused on two variants of CLIP, ResNet and Vision Transformer (ViT), given their centrality in many state-of-the-art VLMs. We apply two complementary methods closely modelled after human experiments: a prompt-based evaluation that uses probabilities as a measure of model preference, and we use Grad-CAM as a novel approach to interpret visual attention in shape-word matching tasks. Our findings show that these model variants do not consistently exhibit the bouba-kiki effect. While ResNet shows a preference for round shapes, overall performance across both model variants lacks the expected associations. Moreover, direct comparison with prior human data on the same task shows that the models' responses fall markedly short of the robust, modality-integrated behaviour characteristic of human cognition. These results contribute to the ongoing debate about the extent to which VLMs truly understand cross-modal concepts, highlighting limitations in their internal representations and alignment with human intuitions.
>
---
#### [replaced 054] ACADATA: Parallel Dataset of Academic Data for Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.12621v2](http://arxiv.org/pdf/2510.12621v2)**

> **作者:** Iñaki Lacunza; Javier Garcia Gilabert; Francesca De Luca Fornaciari; Javier Aula-Blasco; Aitor Gonzalez-Agirre; Maite Melero; Marta Villegas
>
> **摘要:** We present ACADATA, a high-quality parallel dataset for academic translation, that consists of two subsets: ACAD-TRAIN, which contains approximately 1.5 million author-generated paragraph pairs across 96 language directions and ACAD-BENCH, a curated evaluation set of almost 6,000 translations covering 12 directions. To validate its utility, we fine-tune two Large Language Models (LLMs) on ACAD-TRAIN and benchmark them on ACAD-BENCH against specialized machine-translation systems, general-purpose, open-weight LLMs, and several large-scale proprietary models. Experimental results demonstrate that fine-tuning on ACAD-TRAIN leads to improvements in academic translation quality by +6.1 and +12.4 d-BLEU points on average for 7B and 2B models respectively, while also improving long-context translation in a general domain by up to 24.9% when translating out of English. The fine-tuned top-performing model surpasses the best propietary and open-weight models on academic translation domain. By releasing ACAD-TRAIN, ACAD-BENCH and the fine-tuned models, we provide the community with a valuable resource to advance research in academic domain and long-context translation.
>
---
#### [replaced 055] Variational Reasoning for Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.22637v2](http://arxiv.org/pdf/2509.22637v2)**

> **作者:** Xiangxin Zhou; Zichen Liu; Haonan Wang; Chao Du; Min Lin; Chongxuan Li; Liang Wang; Tianyu Pang
>
> **摘要:** We introduce a variational reasoning framework for language models that treats thinking traces as latent variables and optimizes them through variational inference. Starting from the evidence lower bound (ELBO), we extend it to a multi-trace objective for tighter bounds and propose a forward-KL formulation that stabilizes the training of the variational posterior. We further show that rejection sampling finetuning and binary-reward RL, including GRPO, can be interpreted as local forward-KL objectives, where an implicit weighting by model accuracy naturally arises from the derivation and reveals a previously unnoticed bias toward easier questions. We empirically validate our method on the Qwen 2.5 and Qwen 3 model families across a wide range of reasoning tasks. Overall, our work provides a principled probabilistic perspective that unifies variational inference with RL-style methods and yields stable objectives for improving the reasoning ability of language models. Our code is available at https://github.com/sail-sg/variational-reasoning.
>
---
#### [replaced 056] MMD-Flagger: Leveraging Maximum Mean Discrepancy to Detect Hallucinations
- **分类: cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.01367v2](http://arxiv.org/pdf/2506.01367v2)**

> **作者:** Kensuke Mitsuzawa; Damien Garreau
>
> **摘要:** Large language models (LLMs) have become pervasive in our everyday life. Yet, a fundamental obstacle prevents their use in many critical applications: their propensity to generate fluent, human-quality content that is not grounded in reality. The detection of such hallucinations is thus of the highest importance. In this work, we propose a new method to flag hallucinated content: MMD-Flagger. It relies on Maximum Mean Discrepancy (MMD), a non-parametric distance between distributions. On a high-level perspective, MMD-Flagger tracks the MMD between the output to inspect and counterparts generated with various temperature parameters. We show empirically that inspecting the shape of this trajectory is sufficient to detect most hallucinations. This novel method is benchmarked on machine translation and summarization datasets, on which it exhibits competitive performance relative to natural competitors.
>
---
#### [replaced 057] Enabling Few-Shot Alzheimer's Disease Diagnosis on Biomarker Data with Tabular LLMs
- **分类: cs.CL; cs.LG; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2507.23227v2](http://arxiv.org/pdf/2507.23227v2)**

> **作者:** Sophie Kearney; Shu Yang; Zixuan Wen; Bojian Hou; Duy Duong-Tran; Tianlong Chen; Jason Moore; Marylyn Ritchie; Li Shen
>
> **备注:** accepted by ACM-BCB'25: ACM Conference on Bioinformatics, Computational Biology, and Health Informatics [ACM SIGBio Best Paper Award]
>
> **摘要:** Early and accurate diagnosis of Alzheimer's disease (AD), a complex neurodegenerative disorder, requires analysis of heterogeneous biomarkers (e.g., neuroimaging, genetic risk factors, cognitive tests, and cerebrospinal fluid proteins) typically represented in a tabular format. With flexible few-shot reasoning, multimodal integration, and natural-language-based interpretability, large language models (LLMs) offer unprecedented opportunities for prediction with structured biomedical data. We propose a novel framework called TAP-GPT, Tabular Alzheimer's Prediction GPT, that adapts TableGPT2, a multimodal tabular-specialized LLM originally developed for business intelligence tasks, for AD diagnosis using structured biomarker data with small sample sizes. Our approach constructs few-shot tabular prompts using in-context learning examples from structured biomedical data and finetunes TableGPT2 using the parameter-efficient qLoRA adaption for a clinical binary classification task of AD or cognitively normal (CN). The TAP-GPT framework harnesses the powerful tabular understanding ability of TableGPT2 and the encoded prior knowledge of LLMs to outperform more advanced general-purpose LLMs and a tabular foundation model (TFM) developed for prediction tasks. To our knowledge, this is the first application of LLMs to the prediction task using tabular biomarker data, paving the way for future LLM-driven multi-agent frameworks in biomedical informatics.
>
---
#### [replaced 058] SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents
- **分类: cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.23694v3](http://arxiv.org/pdf/2509.23694v3)**

> **作者:** Jianshuo Dong; Sheng Guo; Hao Wang; Xun Chen; Zhuotao Liu; Tianwei Zhang; Ke Xu; Minlie Huang; Han Qiu
>
> **备注:** Preprint
>
> **摘要:** Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
>
---
#### [replaced 059] Benchmarking Hindi LLMs: A New Suite of Datasets and a Comparative Analysis
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.19831v2](http://arxiv.org/pdf/2508.19831v2)**

> **作者:** Anusha Kamath; Kanishk Singla; Rakesh Paul; Raviraj Joshi; Utkarsh Vaidya; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** Evaluating instruction-tuned Large Language Models (LLMs) in Hindi is challenging due to a lack of high-quality benchmarks, as direct translation of English datasets fails to capture crucial linguistic and cultural nuances. To address this, we introduce a suite of five Hindi LLM evaluation datasets: IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi, and BFCL-Hi. These were created using a methodology that combines from-scratch human annotation with a translate-and-verify process. We leverage this suite to conduct an extensive benchmarking of open-source LLMs supporting Hindi, providing a detailed comparative analysis of their current capabilities. Our curation process also serves as a replicable methodology for developing benchmarks in other low-resource languages.
>
---
#### [replaced 060] Optimal Quantization for Matrix Multiplication
- **分类: cs.IT; cs.AI; cs.CL; cs.LG; math.IT**

- **链接: [http://arxiv.org/pdf/2410.13780v3](http://arxiv.org/pdf/2410.13780v3)**

> **作者:** Or Ordentlich; Yury Polyanskiy
>
> **摘要:** Recent work in machine learning community proposed multiple methods for performing lossy compression (quantization) of large matrices. This quantization is important for accelerating matrix multiplication (main component of large language models), which is often bottlenecked by the speed of loading these matrices from memory. Unlike classical vector quantization and rate-distortion theory, the goal of these new compression algorithms is to be able to approximate not the matrices themselves, but their matrix product. Specifically, given a pair of real matrices $A,B$ an encoder (compressor) is applied to each of them independently producing descriptions with $R$ bits per entry. These representations subsequently are used by the decoder to estimate matrix product $A^\top B$. In this work, we provide a non-asymptotic lower bound on the mean squared error of this approximation (as a function of rate $R$) for the case of matrices $A,B$ with iid Gaussian entries. Algorithmically, we construct a universal quantizer based on nested lattices with an explicit guarantee of approximation error for any (non-random) pair of matrices $A$, $B$ in terms of only Frobenius norms $\|\bar{A}\|_F, \|\bar{B}\|_F$ and $\|\bar{A}^\top \bar{B}\|_F$, where $\bar{A},\bar{B}$ are versions of $A,B$ with zero-centered columns, respectively. For iid Gaussian matrices our quantizer achieves the lower bound and is, thus, asymptotically optimal. A practical low-complexity version of our quantizer achieves performance quite close to optimal. In addition, we derive rate-distortion function for matrix multiplication of iid Gaussian matrices, which exhibits an interesting phase-transition at $R\approx 0.906$ bit/entry, showing necessity of Johnson-Lindestrauss dimensionality reduction (sketching) in the low-rate regime.
>
---
#### [replaced 061] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.09710v2](http://arxiv.org/pdf/2510.09710v2)**

> **作者:** Xiaonan Si; Meilin Zhu; Simeng Qin; Lijia Yu; Lijun Zhang; Shuaitong Liu; Xinfeng Li; Ranjie Duan; Yang Liu; Xiaojun Jia
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to corpus poisoning and contamination attacks, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. To address this problem, we propose a two-stage semantic filtering and conflict-free framework for trustworthy RAG. In the first stage, we perform a joint filter with semantic and cluster-based filtering which is guided by the Entity-intent-relation extractor (EIRE). EIRE extracts entities, latent objectives, and entity relations from both the user query and filtered documents, scores their semantic relevance, and selectively adds valuable documents into the clean retrieval database. In the second stage, we proposed an EIRE-guided conflict-aware filtering module, which analyzes semantic consistency between the query, candidate answers, and retrieved knowledge before final answer generation, filtering out internal and external contradictions that could mislead the model. Through this two-stage process, SeCon-RAG effectively preserves useful knowledge while mitigating conflict contamination, achieving significant improvements in both generation robustness and output trustworthiness. Extensive experiments across various LLMs and datasets demonstrate that the proposed SeCon-RAG markedly outperforms state-of-the-art defense methods.
>
---
#### [replaced 062] Detecting Distillation Data from Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.04850v2](http://arxiv.org/pdf/2510.04850v2)**

> **作者:** Hengxiang Zhang; Hyeong Kyu Choi; Sharon Li; Hongxin Wei
>
> **摘要:** Reasoning distillation has emerged as an efficient and powerful paradigm for enhancing the reasoning capabilities of large language models. However, reasoning distillation may inadvertently cause benchmark contamination, where evaluation data included in distillation datasets can inflate performance metrics of distilled models. In this work, we formally define the task of distillation data detection, which is uniquely challenging due to the partial availability of distillation data. Then, we propose a novel and effective method Token Probability Deviation (TBD), which leverages the probability patterns of the generated output tokens. Our method is motivated by the analysis that distilled models tend to generate near-deterministic tokens for seen questions, while producing more low-probability tokens for unseen questions. Our key idea behind TBD is to quantify how far the generated tokens' probabilities deviate from a high reference probability. In effect, our method achieves competitive detection performance by producing lower scores for seen questions than for unseen questions. Extensive experiments demonstrate the effectiveness of our method, achieving an AUC of 0.918 and a TPR@1% FPR of 0.470 on the S1 dataset.
>
---
#### [replaced 063] Teaching Models to Understand (but not Generate) High-risk Data
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03052v2](http://arxiv.org/pdf/2505.03052v2)**

> **作者:** Ryan Wang; Matthew Finlayson; Luca Soldaini; Swabha Swayamdipta; Robin Jia
>
> **摘要:** Language model developers typically filter out high-risk content -- such as toxic or copyrighted text -- from their pre-training data to prevent models from generating similar outputs. However, removing such data altogether limits models' ability to recognize and appropriately respond to harmful or sensitive content. In this paper, we introduce Selective Loss to Understand but Not Generate (SLUNG), a pre-training paradigm through which models learn to understand high-risk data without learning to generate it. Instead of uniformly applying the next-token prediction loss, SLUNG selectively avoids incentivizing the generation of high-risk tokens while ensuring they remain within the model's context window. As the model learns to predict low-risk tokens that follow high-risk ones, it is forced to understand the high-risk content. Through our experiments, we show that SLUNG consistently improves models' understanding of high-risk data (e.g., ability to recognize toxic content) without increasing its generation (e.g., toxicity of model responses). Overall, our SLUNG paradigm enables models to benefit from high-risk text that would otherwise be filtered out.
>
---
#### [replaced 064] Can an Individual Manipulate the Collective Decisions of Multi-Agents?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.16494v2](http://arxiv.org/pdf/2509.16494v2)**

> **作者:** Fengyuan Liu; Rui Zhao; Shuo Chen; Guohao Li; Philip Torr; Lei Han; Jindong Gu
>
> **摘要:** Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.
>
---
#### [replaced 065] Assessing the Latent Automated Program Repair Capabilities of Large Language Models using Round-Trip Translation
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.07994v2](http://arxiv.org/pdf/2401.07994v2)**

> **作者:** Fernando Vallecillos Ruiz; Anastasiia Grishina; Max Hort; Leon Moonen
>
> **备注:** Accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM)
>
> **摘要:** Research shows that errors in natural language can be corrected by translating texts to another language and back using language models. We explore to what extent this latent correction capability extends to Automated Program Repair (APR) by investigating Round-Trip Translation (RTT): translating code from one programming language into another programming or natural language and back, using Large Language Models (LLMs). We hypothesize that RTT restores patterns most commonly seen in the LLM's training corpora through regression toward the mean, replacing infrequent bugs with more frequent, natural, bug-free code. To test this hypothesis, we employ nine LLMs and four common APR benchmarks in Java, and perform a detailed quantitative and qualitative analysis of RTT-generated patches. We find that RTT through English generates plausible patches for 100 of 164 bugs with GPT-4 on the HumanEval-Java benchmark, and 97 are found to be correct in our manual assessment. Moreover, RTT uniquely generates plausible patches for 46 bugs that were missed by LLMs specifically fine-tuned for APR. While this demonstrates the viability of RTT for APR, we also observe limitations, such as a lower overall bug fix rate than the state-of-the-art and diluting the original coding style. We analyze the impact of these limitations and discuss the potential of using RTT as a complementary component in APR frameworks. A replication package is available for download from https://doi.org/10.5281/zenodo.10500593. Keywords: automated program repair, large language model, machine translation
>
---
#### [replaced 066] Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs
- **分类: cs.LG; cs.CL; cs.CR; cs.CY; math.OC**

- **链接: [http://arxiv.org/pdf/2510.03567v2](http://arxiv.org/pdf/2510.03567v2)**

> **作者:** Fatmazohra Rezkellah; Ramzi Dakhmouche
>
> **摘要:** With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
>
---
#### [replaced 067] Towards Region-aware Bias Evaluation Metrics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.16152v2](http://arxiv.org/pdf/2406.16152v2)**

> **作者:** Angana Borah; Aparna Garimella; Rada Mihalcea
>
> **备注:** Accepted to Cross-Cultural Considerations in NLP (C3NLP Workshop at NAACL 2025) -- Outstanding Paper Award
>
> **摘要:** When exposed to human-generated data, language models are known to learn and amplify societal biases. While previous works introduced benchmarks that can be used to assess the bias in these models, they rely on assumptions that may not be universally true. For instance, a gender bias dimension commonly used by these metrics is that of family--career, but this may not be the only common bias in certain regions of the world. In this paper, we identify topical differences in gender bias across different regions and propose a region-aware bottom-up approach for bias assessment. Our proposed approach uses gender-aligned topics for a given region and identifies gender bias dimensions in the form of topic pairs that are likely to capture gender societal biases. Several of our proposed bias topic pairs are on par with human perception of gender biases in these regions in comparison to the existing ones, and we also identify new pairs that are more aligned than the existing ones. In addition, we use our region-aware bias topic pairs in a Word Embedding Association Test (WEAT)-based evaluation metric to test for gender biases across different regions in different data domains. We also find that LLMs have a higher alignment to bias pairs for highly-represented regions showing the importance of region-aware bias evaluation metric.
>
---
#### [replaced 068] LLM Probability Concentration: How Alignment Shrinks the Generative Horizon
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.17871v2](http://arxiv.org/pdf/2506.17871v2)**

> **作者:** Chenghao Yang; Ari Holtzman
>
> **备注:** Codebase: https://github.com/yangalan123/LLMBranchingFactor. V2: Rewrite the theory part for a broader audience. Add experiments to verify the necessity of the AEP estimator. Generalize findings to multilingual tasks and Qwen models. Add discussions on practical implications, and on which alignment stage contributes most to BF reduction. Add ethical statements connecting pluralistic alignment
>
> **摘要:** Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this consistency in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the *Branching Factor* (BF)--a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this consistency has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., ``Sure'') that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity.
>
---
#### [replaced 069] Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13358v4](http://arxiv.org/pdf/2502.13358v4)**

> **作者:** Yiming Zeng; Wanhao Yu; Zexin Li; Tao Ren; Yu Ma; Jinghan Cao; Xiyan Chen; Tingting Yu
>
> **摘要:** Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
>
---
#### [replaced 070] Evaluating Style-Personalized Text Generation: Challenges and Directions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.06374v2](http://arxiv.org/pdf/2508.06374v2)**

> **作者:** Anubhav Jangra; Bahareh Sarrafzadeh; Silviu Cucerzan; Adrian de Wynter; Sujay Kumar Jauhar
>
> **摘要:** With the surge of large language models (LLMs) and their ability to produce customized output, style-personalized text generation--"write like me"--has become a rapidly growing area of interest. However, style personalization is highly specific, relative to every user, and depends strongly on the pragmatic context, which makes it uniquely challenging. Although prior research has introduced benchmarks and metrics for this area, they tend to be non-standardized and have known limitations (e.g., poor correlation with human subjects). LLMs have been found to not capture author-specific style well, it follows that the metrics themselves must be scrutinized carefully. In this work we critically examine the effectiveness of the most common metrics used in the field, such as BLEU, embeddings, and LLMs-as-judges. We evaluate these metrics using our proposed style discrimination benchmark, which spans eight diverse writing tasks across three evaluation settings: domain discrimination, authorship attribution, and LLM-generated personalized vs non-personalized discrimination. We find strong evidence that employing ensembles of diverse evaluation metrics consistently outperforms single-evaluator methods, and conclude by providing guidance on how to reliably assess style-personalized text generation.
>
---
#### [replaced 071] ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Code
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08163v3](http://arxiv.org/pdf/2510.08163v3)**

> **作者:** Jian Xie; Zhendong Chu; Aoxiao Zhong; Kai Zhang; Mingzhe Han; Xing Fan; Jialie Shen; Qingsong Wen
>
> **备注:** Work in Progress
>
> **摘要:** Large Reasoning Models (LRMs) often suffer from the ``over-thinking'' problem, generating unnecessarily long reasoning on simple tasks. Some strategies have been proposed to mitigate this issue, such as length penalties or routing mechanisms, but they are typically heuristic and task-specific, lacking a general framework for adaptive reasoning. In this paper, we present ARM2, a unified model that adaptively balances reasoning performance and efficiency across multiple formats through a reinforcement learning framework augmented with length-aware optimization. Beyond conventional natural language inference, ARM2 integrates vision understanding, extending its applicability to multimodal. Moreover, ARM2 integrates executable code into reasoning, enabling substantial reductions in token cost while preserving task performance compared to long CoT. Experiments demonstrate that ARM2 achieves performance on par with traditional reasoning models trained with GRPO, while reducing token usage by over 70% on average. We further conduct extensive analyses to validate the effectiveness of ARM2 and the soundness of its design.
>
---
#### [replaced 072] Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24289v3](http://arxiv.org/pdf/2503.24289v3)**

> **作者:** Jiacheng Lin; Tian Wang; Kun Qian
>
> **备注:** Published in the TMLR journal
>
> **摘要:** We propose Rec-R1, a general reinforcement learning framework that bridges large language models (LLMs) with recommendation systems through closed-loop optimization. Unlike prompting and supervised fine-tuning (SFT), Rec-R1 directly optimizes LLM generation using feedback from a fixed black-box recommendation model, without relying on synthetic SFT data from proprietary models such as GPT-4o. This avoids the substantial cost and effort required for data distillation. To verify the effectiveness of Rec-R1, we evaluate it on two representative tasks: product search and sequential recommendation. Experimental results demonstrate that Rec-R1 not only consistently outperforms prompting- and SFT-based methods, but also achieves significant gains over strong discriminative baselines, even when used with simple retrievers such as BM25. Moreover, Rec-R1 preserves the general-purpose capabilities of the LLM, unlike SFT, which often impairs instruction-following and reasoning. These findings suggest Rec-R1 as a promising foundation for continual task-specific adaptation without catastrophic forgetting.
>
---
#### [replaced 073] AutoPR: Let's Automate Your Academic Promotion!
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09558v2](http://arxiv.org/pdf/2510.09558v2)**

> **作者:** Qiguang Chen; Zheng Yan; Mingda Yang; Libo Qin; Yixin Yuan; Hanjing Li; Jinhao Liu; Yiyan Ji; Dengyun Peng; Jiannan Guan; Mengkang Hu; Yantao Du; Wanxiang Che
>
> **备注:** Preprint. Code: https://github.com/LightChen233/AutoPR . Benchmark: https://huggingface.co/datasets/yzweak/PRBench
>
> **摘要:** As the volume of peer-reviewed research surges, scholars increasingly rely on social platforms for discovery, while authors invest considerable effort in promoting their work to ensure visibility and citations. To streamline this process and reduce the reliance on human effort, we introduce Automatic Promotion (AutoPR), a novel task that transforms research papers into accurate, engaging, and timely public content. To enable rigorous evaluation, we release PRBench, a multimodal benchmark that links 512 peer-reviewed articles to high-quality promotional posts, assessing systems along three axes: Fidelity (accuracy and tone), Engagement (audience targeting and appeal), and Alignment (timing and channel optimization). We also introduce PRAgent, a multi-agent framework that automates AutoPR in three stages: content extraction with multimodal preparation, collaborative synthesis for polished outputs, and platform-specific adaptation to optimize norms, tone, and tagging for maximum reach. When compared to direct LLM pipelines on PRBench, PRAgent demonstrates substantial improvements, including a 604% increase in total watch time, a 438% rise in likes, and at least a 2.9x boost in overall engagement. Ablation studies show that platform modeling and targeted promotion contribute the most to these gains. Our results position AutoPR as a tractable, measurable research problem and provide a roadmap for scalable, impactful automated scholarly communication.
>
---
#### [replaced 074] A Risk Taxonomy and Reflection Tool for Large Language Model Adoption in Public Health
- **分类: cs.HC; cs.AI; cs.CL; H.5; J.3; K.4**

- **链接: [http://arxiv.org/pdf/2411.02594v3](http://arxiv.org/pdf/2411.02594v3)**

> **作者:** Jiawei Zhou; Amy Z. Chen; Darshi Shah; Laura M. Schwab Reese; Munmun De Choudhury
>
> **摘要:** Recent breakthroughs in large language models (LLMs) have generated both interest and concern about their potential adoption as information sources or communication tools across different domains. In public health, where stakes are high and impacts extend across diverse populations, adopting LLMs poses unique challenges that require thorough evaluation. However, structured approaches for assessing potential risks in public health remain under-explored. To address this gap, we conducted focus groups with public health professionals and individuals with lived experience to unpack their concerns, situated across three distinct and critical public health issues that demand high-quality information: infectious disease prevention (vaccines), chronic and well-being care (opioid use disorder), and community health and safety (intimate partner violence). We synthesize participants' perspectives into a risk taxonomy, identifying and contextualizing the potential harms LLMs may introduce when positioned alongside traditional health communication. This taxonomy highlights four dimensions of risk to individuals, human-centered care, information ecosystem, and technology accountability. For each dimension, we unpack specific risks and offer example reflection questions to help practitioners adopt a risk-reflexive approach. By summarizing distinctive LLM characteristics and linking them to identified risks, we discuss the need to revisit prior mental models of information behaviors and complement evaluations with external validity and domain expertise through lived experience and real-world practices. Together, this work contributes a shared vocabulary and reflection tool for people in both computing and public health to collaboratively anticipate, evaluate, and mitigate risks in deciding when to employ LLM capabilities (or not) and how to mitigate harm.
>
---
#### [replaced 075] I Have No Mouth, and I Must Rhyme: Uncovering Internal Phonetic Representations in LLaMA 3.2
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02527v2](http://arxiv.org/pdf/2508.02527v2)**

> **作者:** Oliver McLaughlin; Arjun Khurana; Jack Merullo
>
> **摘要:** Large language models demonstrate proficiency on phonetic tasks, such as rhyming, without explicit phonetic or auditory grounding. In this work, we investigate how \verb|Llama-3.2-1B-Instruct| represents token-level phonetic information. Our results suggest that Llama uses a rich internal model of phonemes to complete phonetic tasks. We provide evidence for high-level organization of phoneme representations in its latent space. In doing so, we also identify a ``phoneme mover head" which promotes phonetic information during rhyming tasks. We visualize the output space of this head and find that, while notable differences exist, Llama learns a model of vowels similar to the standard IPA vowel chart for humans, despite receiving no direct supervision to do so.
>
---
#### [replaced 076] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.03144v2](http://arxiv.org/pdf/2506.03144v2)**

> **作者:** Wei Chow; Yuan Gao; Linfeng Li; Xian Wang; Qi Xu; Hang Song; Lingdong Kong; Ran Zhou; Yi Zeng; Yidong Cai; Botian Jiang; Shilin Xu; Jiajun Zhang; Minghui Qiu; Xiangtai Li; Tianshu Yang; Siliang Tang; Juncheng Li
>
> **备注:** NeurIPS 2025; Project Page, Code, and Dataset at: https://merit-2025.github.io/
>
> **摘要:** Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval.
>
---
#### [replaced 077] What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10113v3](http://arxiv.org/pdf/2505.10113v3)**

> **作者:** Xinlan Yan; Di Wu; Yibin Lei; Christof Monz; Iacer Calixto
>
> **摘要:** In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset for benchmarking large language models (LLMs) in fine-grained clinical specialties. S-MedQA has over 20k examples, covers 15 medical specialties, and QA pairs can have multiple specialty annotations (e.g., when a question is cross-disciplinary), constructed with both machine and expert verification to maximize data availability. We use S-MedQA to investigate the role of clinical specialty data in the knowledge-intensive scenario of medical QA. Our results show that 1) training on data from a clinical specialty does not necessarily lead to best performance on that specialty, and 2) regardless of the specialty the LLM was fine-tuned on, token probabilities of clinically relevant terms increase consistently across all specialties. Thus, we hypothesize improvement gains are derived mostly from domain shifting (e.g., general to medical) rather than specialty-specific knowledge injection, and suggest rethinking the role of fine-tuning data in the medical domain.
>
---
#### [replaced 078] The Landscape of Arabic Large Language Models (ALLMs): A New Era for Arabic Language Technology
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01340v2](http://arxiv.org/pdf/2506.01340v2)**

> **作者:** Shahad Al-Khalifa; Nadir Durrani; Hend Al-Khalifa; Firoj Alam
>
> **备注:** Accepted at CACM
>
> **摘要:** The emergence of ChatGPT marked a transformative milestone for Artificial Intelligence (AI), showcasing the remarkable potential of Large Language Models (LLMs) to generate human-like text. This wave of innovation has revolutionized how we interact with technology, seamlessly integrating LLMs into everyday tasks such as vacation planning, email drafting, and content creation. While English-speaking users have significantly benefited from these advancements, the Arabic world faces distinct challenges in developing Arabic-specific LLMs. Arabic, one of the languages spoken most widely around the world, serves more than 422 million native speakers in 27 countries and is deeply rooted in a rich linguistic and cultural heritage. Developing Arabic LLMs (ALLMs) presents an unparalleled opportunity to bridge technological gaps and empower communities. The journey of ALLMs has been both fascinating and complex, evolving from rudimentary text processing systems to sophisticated AI-driven models. This article explores the trajectory of ALLMs, from their inception to the present day, highlighting the efforts to evaluate these models through benchmarks and public leaderboards. We also discuss the challenges and opportunities that ALLMs present for the Arab world.
>
---
