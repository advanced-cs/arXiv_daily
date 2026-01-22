# 自然语言处理 cs.CL

- **最新发布 81 篇**

- **更新 70 篇**

## 最新发布

#### [new 001] Business Logic-Driven Text-to-SQL Data Synthesis for Business Intelligence
- **分类: cs.CL**

- **简介: 该论文属于Text-to-SQL任务，旨在解决私有业务智能场景中数据稀缺问题。通过构建基于业务逻辑的合成数据，提升数据真实性与复杂性，验证模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14518v1](https://arxiv.org/pdf/2601.14518v1)**

> **作者:** Jinhui Liu; Ximeng Zhang; Yanbo Ai; Zhou Yu
>
> **摘要:** Evaluating Text-to-SQL agents in private business intelligence (BI) settings is challenging due to the scarcity of realistic, domain-specific data. While synthetic evaluation data offers a scalable solution, existing generation methods fail to capture business realism--whether questions reflect realistic business logic and workflows. We propose a Business Logic-Driven Data Synthesis framework that generates data grounded in business personas, work scenarios, and workflows. In addition, we improve the data quality by imposing a business reasoning complexity control strategy that diversifies the analytical reasoning steps required to answer the questions. Experiments on a production-scale Salesforce database show that our synthesized data achieves high business realism (98.44%), substantially outperforming OmniSQL (+19.5%) and SQL-Factory (+54.7%), while maintaining strong question-SQL alignment (98.59%). Our synthetic data also reveals that state-of-the-art Text-to-SQL models still have significant performance gaps, achieving only 42.86% execution accuracy on the most complex business queries.
>
---
#### [new 002] Guided by the Plan: Enhancing Faithful Autoregressive Text-to-Audio Generation with Guided Decoding
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决AR模型难以忠实响应复杂文本提示的问题。通过引入Plan-Critic模型，提升生成质量并保持计算效率。**

- **链接: [https://arxiv.org/pdf/2601.14304v1](https://arxiv.org/pdf/2601.14304v1)**

> **作者:** Juncheng Wang; Zhe Hu; Chao Xu; Siyue Ren; Yuxiang Feng; Yang Liu; Baigui Sun; Shujun Wang
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Autoregressive (AR) models excel at generating temporally coherent audio by producing tokens sequentially, yet they often falter in faithfully following complex textual prompts, especially those describing complex sound events. We uncover a surprising capability in AR audio generators: their early prefix tokens implicitly encode global semantic attributes of the final output, such as event count and sound-object category, revealing a form of implicit planning. Building on this insight, we propose Plan-Critic, a lightweight auxiliary model trained with a Generalized Advantage Estimation (GAE)-inspired objective to predict final instruction-following quality from partial generations. At inference time, Plan-Critic enables guided exploration: it evaluates candidate prefixes early, prunes low-fidelity trajectories, and reallocates computation to high-potential planning seeds. Our Plan-Critic-guided sampling achieves up to a 10-point improvement in CLAP score over the AR baseline-establishing a new state of the art in AR text-to-audio generation-while maintaining computational parity with standard best-of-N decoding. This work bridges the gap between causal generation and global semantic alignment, demonstrating that even strictly autoregressive models can plan ahead.
>
---
#### [new 003] A Comprehensive Benchmark of Language Models on Unicode and Romanized Sinhala
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型评估任务，旨在解决低资源语言如僧伽罗语在Unicode和罗马化文本上的表现问题。通过测试不同模型的预测能力，提供选择模型的参考。**

- **链接: [https://arxiv.org/pdf/2601.14958v1](https://arxiv.org/pdf/2601.14958v1)**

> **作者:** Minuri Rajapakse; Ruvan Weerasinghe
>
> **备注:** 6 pages, 1 figure, 3 tables
>
> **摘要:** The performance of Language Models (LMs) on lower-resource, morphologically rich languages like Sinhala remains under-explored, particularly for Romanized Sinhala, which is prevalent in digital communication. This paper presents a comprehensive benchmark of modern LMs on a diverse corpus of Unicode and Romanized Sinhala. We evaluate open-source models using perplexity, a measure of how well a model predicts a text, and leading closed-source models via a qualitative analysis of sentence completion. Our findings reveal that the Mistral-Nemo-Base-2407 model achieves the strongest predictive performance on Unicode text and the Mistral-7B-v0.3 model for Romanized text. The results also highlight the strong all-around performance of the Llama-3.1-8B model for both scripts. Furthermore, a significant performance disparity exists among closed-source models: Gemini-1.5-pro and DeepSeek excel at Unicode generation, whereas Claude-3.5-Sonnet is superior at handling Romanized text. These results provide an essential guide for practitioners selecting models for Sinhala-specific applications and highlight the critical role of training data in handling script variations.
>
---
#### [new 004] Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education
- **分类: cs.CL**

- **简介: 该论文属于教育领域的人工智能任务，旨在提升LLM在教学中的表现。通过引入教育理论引导和思维奖励机制，优化模型的内部推理过程，增强其教学能力。**

- **链接: [https://arxiv.org/pdf/2601.14560v1](https://arxiv.org/pdf/2601.14560v1)**

> **作者:** Unggi Lee; Jiyeong Bae; Jaehyeon Park; Haeun Park; Taejun Park; Younghoon Jeon; Sungmin Cho; Junbo Koh; Yeil Jeong; Gyeonggeon Lee
>
> **摘要:** Large language models (LLMs) are increasingly deployed as intelligent tutoring systems, yet research on optimizing LLMs specifically for educational contexts remains limited. Recent works have proposed reinforcement learning approaches for training LLM tutors, but these methods focus solely on optimizing visible responses while neglecting the model's internal thinking process. We introduce PedagogicalRL-Thinking, a framework that extends pedagogical alignment to reasoning LLMs in education through two novel approaches: (1) Pedagogical Reasoning Prompting, which guides internal reasoning using domain-specific educational theory rather than generic instructions; and (2) Thinking Reward, which explicitly evaluates and reinforces the pedagogical quality of the model's reasoning traces. Our experiments reveal that domain-specific, theory-grounded prompting outperforms generic prompting, and that Thinking Reward is most effective when combined with pedagogical prompting. Furthermore, models trained only on mathematics tutoring dialogues show improved performance on educational benchmarks not seen during training, while preserving the base model's factual knowledge. Our quantitative and qualitative analyses reveal that pedagogical thinking reward produces systematic reasoning trace changes, with increased pedagogical reasoning and more structured instructional decision-making in the tutor's thinking process.
>
---
#### [new 005] Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的偏见和迎合问题。通过自盲化和反事实自我模拟，提升模型决策的公平性与透明度。**

- **链接: [https://arxiv.org/pdf/2601.14553v1](https://arxiv.org/pdf/2601.14553v1)**

> **作者:** Brian Christian; Matan Mazor
>
> **摘要:** Fair decisions require ignoring irrelevant, potentially biasing, information. To achieve this, decision-makers need to approximate what decision they would have made had they not known certain facts, such as the gender or race of a job candidate. This counterfactual self-simulation is notoriously hard for humans, leading to biased judgments even by well-meaning actors. Here we show that large language models (LLMs) suffer from similar limitations in their ability to approximate what decisions they would make under counterfactual knowledge in offsetting gender and race biases and overcoming sycophancy. We show that prompting models to ignore or pretend not to know biasing information fails to offset these biases and occasionally backfires. However, unlike humans, LLMs can be given access to a ground-truth model of their own counterfactual cognition -- their own API. We show that this access to the responses of a blinded replica enables fairer decisions, while providing greater transparency to distinguish implicit from intentionally biased behavior.
>
---
#### [new 006] Taxonomy-Aligned Risk Extraction from 10-K Filings with Autonomous Improvement Using LLMs
- **分类: cs.CL**

- **简介: 该论文属于信息提取任务，旨在从10-K文件中结构化提取风险因素，并保持与预定义分类体系的一致性。通过三阶段流程实现精准提取与分类，同时引入自主优化机制提升分类效果。**

- **链接: [https://arxiv.org/pdf/2601.15247v1](https://arxiv.org/pdf/2601.15247v1)**

> **作者:** Rian Dolphin; Joe Dursun; Jarrett Blankenship; Katie Adams; Quinton Pike
>
> **备注:** 4 figures, 9 pages
>
> **摘要:** We present a methodology for extracting structured risk factors from corporate 10-K filings while maintaining adherence to a predefined hierarchical taxonomy. Our three-stage pipeline combines LLM extraction with supporting quotes, embedding-based semantic mapping to taxonomy categories, and LLM-as-a-judge validation that filters spurious assignments. To evaluate our approach, we extract 10,688 risk factors from S&P 500 companies and examine risk profile similarity across industry clusters. Beyond extraction, we introduce autonomous taxonomy maintenance where an AI agent analyzes evaluation feedback to identify problematic categories, diagnose failure patterns, and propose refinements, achieving 104.7% improvement in embedding separation in a case study. External validation confirms the taxonomy captures economically meaningful structure: same-industry companies exhibit 63% higher risk profile similarity than cross-industry pairs (Cohen's d=1.06, AUC 0.82, p<0.001). The methodology generalizes to any domain requiring taxonomy-aligned extraction from unstructured text, with autonomous improvement enabling continuous quality maintenance and enhancement as systems process more documents.
>
---
#### [new 007] Obscuring Data Contamination Through Translation: Evidence from Arabic Corpora
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决多语言数据污染问题。通过实验发现阿拉伯语翻译可隐藏污染迹象，提出跨语言检测方法以提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2601.14994v1](https://arxiv.org/pdf/2601.14994v1)**

> **作者:** Chaymaa Abbas; Nour Shamaa; Mariette Awad
>
> **摘要:** Data contamination undermines the validity of Large Language Model evaluation by enabling models to rely on memorized benchmark content rather than true generalization. While prior work has proposed contamination detection methods, these approaches are largely limited to English benchmarks, leaving multilingual contamination poorly understood. In this work, we investigate contamination dynamics in multilingual settings by fine-tuning several open-weight LLMs on varying proportions of Arabic datasets and evaluating them on original English benchmarks. To detect memorization, we extend the Tested Slot Guessing method with a choice-reordering strategy and incorporate Min-K% probability analysis, capturing both behavioral and distributional contamination signals. Our results show that translation into Arabic suppresses conventional contamination indicators, yet models still benefit from exposure to contaminated data, particularly those with stronger Arabic capabilities. This effect is consistently reflected in rising Mink% scores and increased cross-lingual answer consistency as contamination levels grow. To address this blind spot, we propose Translation-Aware Contamination Detection, which identifies contamination by comparing signals across multiple translated benchmark variants rather than English alone. The Translation-Aware Contamination Detection reliably exposes contamination even when English-only methods fail. Together, our findings highlight the need for multilingual, translation-aware evaluation pipelines to ensure fair, transparent, and reproducible assessment of LLMs.
>
---
#### [new 008] Towards Execution-Grounded Automated AI Research
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自动化AI研究任务，旨在解决LLM生成无效想法的问题。通过构建执行器和实验环境，验证了执行反馈的有效性，并探索了进化搜索与强化学习方法。**

- **链接: [https://arxiv.org/pdf/2601.14525v1](https://arxiv.org/pdf/2601.14525v1)**

> **作者:** Chenglei Si; Zitong Yang; Yejin Choi; Emmanuel Candès; Diyi Yang; Tatsunori Hashimoto
>
> **摘要:** Automated AI research holds great potential to accelerate scientific discovery. However, current LLMs often generate plausible-looking but ineffective ideas. Execution grounding may help, but it is unclear whether automated execution is feasible and whether LLMs can learn from the execution feedback. To investigate these, we first build an automated executor to implement ideas and launch large-scale parallel GPU experiments to verify their effectiveness. We then convert two realistic research problems - LLM pre-training and post-training - into execution environments and demonstrate that our automated executor can implement a large fraction of the ideas sampled from frontier LLMs. We analyze two methods to learn from the execution feedback: evolutionary search and reinforcement learning. Execution-guided evolutionary search is sample-efficient: it finds a method that significantly outperforms the GRPO baseline (69.4% vs 48.0%) on post-training, and finds a pre-training recipe that outperforms the nanoGPT baseline (19.7 minutes vs 35.9 minutes) on pre-training, all within just ten search epochs. Frontier LLMs often generate meaningful algorithmic ideas during search, but they tend to saturate early and only occasionally exhibit scaling trends. Reinforcement learning from execution reward, on the other hand, suffers from mode collapse. It successfully improves the average reward of the ideator model but not the upper-bound, due to models converging on simple ideas. We thoroughly analyze the executed ideas and training dynamics to facilitate future efforts towards execution-grounded automated AI research.
>
---
#### [new 009] Circadian Modulation of Semantic Exploration in Social Media Language
- **分类: cs.CL; cs.CY; cs.SI; q-bio.NC**

- **简介: 该论文研究社交媒体语言的昼夜节律，通过分析Reddit数据，探讨语义探索与利用的时序变化，揭示其与生物节律的关系。属于自然语言处理与认知科学交叉任务。**

- **链接: [https://arxiv.org/pdf/2601.15091v1](https://arxiv.org/pdf/2601.15091v1)**

> **作者:** Vuong Hung Truong; Mariana Gabrielle Cangco Reyes; Masatoshi Koizumi; Jihwan Myung
>
> **备注:** 25 pages, 6 figures, 3 supplementary figures
>
> **摘要:** Human cognition exhibits strong circadian modulation, yet its influence on high-dimensional semantic behavior remains poorly understood. Using large-scale Reddit data, we quantify time-of-day variation in language use by embedding text into a pretrained transformer model and measuring semantic entropy as an index of linguistic exploration-exploitation, for which we show a robust circadian rhythmicity that could be entrained by seasonal light cues. Distinguishing between local and global semantic entropy reveals a systematic temporal dissociation: local semantic exploration peaks in the morning, reflecting broader exploration of semantic space, whereas global semantic diversity peaks later in the day as submissions accumulate around already established topics, consistent with "rich-get-richer" dynamics. These patterns are not explained by sentiment or affective valence, indicating that semantic exploration captures a cognitive dimension distinct from mood. The observed temporal structure aligns with known diurnal patterns in neuromodulatory systems, suggesting that biological circadian rhythms extend to the semantic domain.
>
---
#### [new 010] The Effect of Scripts and Formats on LLM Numeracy
- **分类: cs.CL**

- **简介: 该论文研究LLM在不同数字脚本和格式下的数值推理能力，旨在解决多语言数值处理中的挑战。通过实验发现模型性能下降，并提出针对性提示策略提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.15251v1](https://arxiv.org/pdf/2601.15251v1)**

> **作者:** Varshini Reddy; Craig W. Schmidt; Seth Ebner; Adam Wiemerslage; Yuval Pinter; Chris Tanner
>
> **摘要:** Large language models (LLMs) have achieved impressive proficiency in basic arithmetic, rivaling human-level performance on standard numerical tasks. However, little attention has been given to how these models perform when numerical expressions deviate from the prevailing conventions present in their training corpora. In this work, we investigate numerical reasoning across a wide range of numeral scripts and formats. We show that LLM accuracy drops substantially when numerical inputs are rendered in underrepresented scripts or formats, despite the underlying mathematical reasoning being identical. We further demonstrate that targeted prompting strategies, such as few-shot prompting and explicit numeral mapping, can greatly narrow this gap. Our findings highlight an overlooked challenge in multilingual numerical reasoning and provide actionable insights for working with LLMs to reliably interpret, manipulate, and generate numbers across diverse numeral scripts and formatting styles.
>
---
#### [new 011] Knowledge Restoration-driven Prompt Optimization: Unlocking LLM Potential for Open-Domain Relational Triplet Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放域关系三元组抽取任务，旨在解决LLM因静态提示策略导致的语义歧义问题。提出KRPO框架，通过知识重建和文本梯度优化提升抽取性能。**

- **链接: [https://arxiv.org/pdf/2601.15037v1](https://arxiv.org/pdf/2601.15037v1)**

> **作者:** Xiaonan Jing; Gongqing Wu; Xingrui Zhuo; Lang Sun; Jiapu Wang
>
> **摘要:** Open-domain Relational Triplet Extraction (ORTE) is the foundation for mining structured knowledge without predefined schemas. Despite the impressive in-context learning capabilities of Large Language Models (LLMs), existing methods are hindered by their reliance on static, heuristic-driven prompting strategies. Due to the lack of reflection mechanisms required to internalize erroneous signals, these methods exhibit vulnerability in semantic ambiguity, often making erroneous extraction patterns permanent. To address this bottleneck, we propose a Knowledge Reconstruction-driven Prompt Optimization (KRPO) framework to assist LLMs in continuously improving their extraction capabilities for complex ORTE task flows. Specifically, we design a self-evaluation mechanism based on knowledge restoration, which provides intrinsic feedback signals by projecting structured triplets into semantic consistency scores. Subsequently, we propose a prompt optimizer based on a textual gradient that can internalize historical experiences to iteratively optimize prompts, which can better guide LLMs to handle subsequent extraction tasks. Furthermore, to alleviate relation redundancy, we design a relation canonicalization memory that collects representative relations and provides semantically distinct schemas for the triplets. Extensive experiments across three datasets show that KRPO significantly outperforms strong baselines in the extraction F1 score.
>
---
#### [new 012] Multi-Agent Constraint Factorization Reveals Latent Invariant Solution Structure
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文研究多智能体系统中的约束分解问题，旨在解释为何多个智能体在相同信息下表现更优。通过约束优化和算子理论，揭示其解空间的不变结构。**

- **链接: [https://arxiv.org/pdf/2601.15077v1](https://arxiv.org/pdf/2601.15077v1)**

> **作者:** Christopher Scofield
>
> **摘要:** Multi-agent systems (MAS) composed of large language models often exhibit improved problem-solving performance despite operating on identical information. In this work, we provide a formal explanation for this phenomenon grounded in operator theory and constrained optimization. We model each agent as enforcing a distinct family of validity constraints on a shared solution state, and show that a MAS implements a factorized composition of constraint-enforcement operators. Under mild conditions, these dynamics converge to invariant solution sets defined by the intersection of agent constraint sets. Such invariant structures are generally not dynamically accessible to a single agent applying all constraints simultaneously, even when expressive capacity and information are identical. We extend this result from exact constraint enforcement to soft constraints via proximal operators, and apply the formalism to contemporary text-based dialog systems.
>
---
#### [new 013] Large Language Models for Large-Scale, Rigorous Qualitative Analysis in Applied Health Services Research
- **分类: cs.CL**

- **简介: 该论文属于应用健康服务研究中的定性分析任务，旨在解决如何有效整合大语言模型提升研究效率与严谨性的问题。工作中构建了通用框架，并应用于糖尿病护理研究。**

- **链接: [https://arxiv.org/pdf/2601.14478v1](https://arxiv.org/pdf/2601.14478v1)**

> **作者:** Sasha Ronaghi; Emma-Louise Aveling; Maria Levis; Rachel Lauren Ross; Emily Alsentzer; Sara Singer
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Large language models (LLMs) show promise for improving the efficiency of qualitative analysis in large, multi-site health-services research. Yet methodological guidance for LLM integration into qualitative analysis and evidence of their impact on real-world research methods and outcomes remain limited. We developed a model- and task-agnostic framework for designing human-LLM qualitative analysis methods to support diverse analytic aims. Within a multi-site study of diabetes care at Federally Qualified Health Centers (FQHCs), we leveraged the framework to implement human-LLM methods for (1) qualitative synthesis of researcher-generated summaries to produce comparative feedback reports and (2) deductive coding of 167 interview transcripts to refine a practice-transformation intervention. LLM assistance enabled timely feedback to practitioners and the incorporation of large-scale qualitative data to inform theory and practice changes. This work demonstrates how LLMs can be integrated into applied health-services research to enhance efficiency while preserving rigor, offering guidance for continued innovation with LLMs in qualitative research.
>
---
#### [new 014] RSNA Large Language Model Benchmark Dataset for Chest Radiographs of Cardiothoracic Disease: Radiologist Evaluation and Validation Enhanced by AI Labels (REVEAL-CXR)
- **分类: cs.CL**

- **简介: 该论文属于医学图像分析任务，旨在构建高质量的胸部X光基准数据集。通过AI辅助标注，提升放射科医生标注效率，并验证模型性能。**

- **链接: [https://arxiv.org/pdf/2601.15129v1](https://arxiv.org/pdf/2601.15129v1)**

> **作者:** Yishu Wei; Adam E. Flanders; Errol Colak; John Mongan; Luciano M Prevedello; Po-Hao Chen; Henrique Min Ho Lee; Gilberto Szarf; Hamilton Shoji; Jason Sho; Katherine Andriole; Tessa Cook; Lisa C. Adams; Linda C. Chu; Maggie Chung; Geraldine Brusca-Augello; Djeven P. Deva; Navneet Singh; Felipe Sanchez Tijmes; Jeffrey B. Alpert; Elsie T. Nguyen; Drew A. Torigian; Kate Hanneman; Lauren K Groner; Alexander Phan; Ali Islam; Matias F. Callejas; Gustavo Borges da Silva Teles; Faisal Jamal; Maryam Vazirabad; Ali Tejani; Hari Trivedi; Paulo Kuriki; Rajesh Bhayana; Elana T. Benishay; Yi Lin; Yifan Peng; George Shih
>
> **摘要:** Multimodal large language models have demonstrated comparable performance to that of radiology trainees on multiple-choice board-style exams. However, to develop clinically useful multimodal LLM tools, high-quality benchmarks curated by domain experts are essential. To curate released and holdout datasets of 100 chest radiographic studies each and propose an artificial intelligence (AI)-assisted expert labeling procedure to allow radiologists to label studies more efficiently. A total of 13,735 deidentified chest radiographs and their corresponding reports from the MIDRC were used. GPT-4o extracted abnormal findings from the reports, which were then mapped to 12 benchmark labels with a locally hosted LLM (Phi-4-Reasoning). From these studies, 1,000 were sampled on the basis of the AI-suggested benchmark labels for expert review; the sampling algorithm ensured that the selected studies were clinically relevant and captured a range of difficulty levels. Seventeen chest radiologists participated, and they marked "Agree all", "Agree mostly" or "Disagree" to indicate their assessment of the correctness of the LLM suggested labels. Each chest radiograph was evaluated by three experts. Of these, at least two radiologists selected "Agree All" for 381 radiographs. From this set, 200 were selected, prioritizing those with less common or multiple finding labels, and divided into 100 released radiographs and 100 reserved as the holdout dataset. The holdout dataset is used exclusively by RSNA to independently evaluate different models. A benchmark of 200 chest radiographic studies with 12 benchmark labels was created and made publicly available https://imaging.rsna.org, with each chest radiograph verified by three radiologists. In addition, an AI-assisted labeling procedure was developed to help radiologists label at scale, minimize unnecessary omissions, and support a semicollaborative environment.
>
---
#### [new 015] DARL: Encouraging Diverse Answers for General Reasoning without Verifiers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型生成答案缺乏多样性的问题。提出DARL框架，在不依赖验证器的情况下提升推理能力和输出多样性。**

- **链接: [https://arxiv.org/pdf/2601.14700v1](https://arxiv.org/pdf/2601.14700v1)**

> **作者:** Chongxuan Huang; Lei Lin; Xiaodong Shi; Wenping Hu; Ruiming Tang
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated promising gains in enhancing the reasoning capabilities of large language models. However, its dependence on domain-specific verifiers significantly restricts its applicability to open and general domains. Recent efforts such as RLPR have extended RLVR to general domains, enabling training on broader datasets and achieving improvements over RLVR. However, a notable limitation of these methods is their tendency to overfit to reference answers, which constrains the model's ability to generate diverse outputs. This limitation is particularly pronounced in open-ended tasks such as writing, where multiple plausible answers exist. To address this, we propose DARL, a simple yet effective reinforcement learning framework that encourages the generation of diverse answers within a controlled deviation range from the reference while preserving alignment with it. Our framework is fully compatible with existing general reinforcement learning methods and can be seamlessly integrated without additional verifiers. Extensive experiments on thirteen benchmarks demonstrate consistent improvements in reasoning performance. Notably, DARL surpasses RLPR, achieving average gains of 1.3 points on six reasoning benchmarks and 9.5 points on seven general benchmarks, highlighting its effectiveness in improving both reasoning accuracy and output diversity.
>
---
#### [new 016] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言检索增强生成任务，旨在解决多语言环境下知识偏差和冲突问题。提出LcRL框架，通过语言耦合策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14896v1](https://arxiv.org/pdf/2601.14896v1)**

> **作者:** Rui Qi; Fengran Mo; Yufeng Chen; Xue Zhang; Shuo Wang; Hongliang Li; Jinan Xu; Meng Jiang; Jian-Yun Nie; Kaiyu Huang
>
> **摘要:** Multilingual retrieval-augmented generation (MRAG) requires models to effectively acquire and integrate beneficial external knowledge from multilingual collections. However, most existing studies employ a unitive process where queries of equivalent semantics across different languages are processed through a single-turn retrieval and subsequent optimization. Such a ``one-size-fits-all'' strategy is often suboptimal in multilingual settings, as the models occur to knowledge bias and conflict during the interaction with the search engine. To alleviate the issues, we propose LcRL, a multilingual search-augmented reinforcement learning framework that integrates a language-coupled Group Relative Policy Optimization into the policy and reward models. We adopt the language-coupled group sampling in the rollout module to reduce knowledge bias, and regularize an auxiliary anti-consistency penalty in the reward models to mitigate the knowledge conflict. Experimental results demonstrate that LcRL not only achieves competitive performance but is also appropriate for various practical scenarios such as constrained training data and retrieval over collections encompassing a large number of languages. Our code is available at https://github.com/Cherry-qwq/LcRL-Open.
>
---
#### [new 017] ClaimDB: A Fact Verification Benchmark over Large Structured Data
- **分类: cs.CL**

- **简介: 该论文提出ClaimDB，一个基于大规模结构化数据的事实验证基准，旨在解决复杂数据下的事实验证问题，通过实验评估大模型的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14698v1](https://arxiv.org/pdf/2601.14698v1)**

> **作者:** Michael Theologitis; Preetam Prabhu Srikar Dammu; Chirag Shah; Dan Suciu
>
> **备注:** The data, code, and leaderboard are available at https://claimdb.github.io
>
> **摘要:** Despite substantial progress in fact-verification benchmarks, claims grounded in large-scale structured data remain underexplored. In this work, we introduce ClaimDB, the first fact-verification benchmark where the evidence for claims is derived from compositions of millions of records and multiple tables. ClaimDB consists of 80 unique real-life databases covering a wide range of domains, from governance and healthcare to media, education and the natural sciences. At this scale, verification approaches that rely on "reading" the evidence break down, forcing a timely shift toward reasoning in executable programs. We conduct extensive experiments with 30 state-of-the-art proprietary and open-source (below 70B) LLMs and find that none exceed 83% accuracy, with more than half below 55%. Our analysis also reveals that both closed- and open-source models struggle with abstention -- the ability to admit that there is no evidence to decide -- raising doubts about their reliability in high-stakes data analysis. We release the benchmark, code, and the LLM leaderboard at https://claimdb.github.io .
>
---
#### [new 018] HiNS: Hierarchical Negative Sampling for More Comprehensive Memory Retrieval Embedding Model
- **分类: cs.CL**

- **简介: 该论文属于记忆增强型语言代理任务，旨在解决负样本多样性不足的问题。提出HiNS框架，通过建模负样本难度层级提升嵌入模型的检索效果。**

- **链接: [https://arxiv.org/pdf/2601.14857v1](https://arxiv.org/pdf/2601.14857v1)**

> **作者:** Motong Tian; Allen P. Wong; Mingjun Mao; Wangchunshu Zhou
>
> **摘要:** Memory-augmented language agents rely on embedding models for effective memory retrieval. However, existing training data construction overlooks a critical limitation: the hierarchical difficulty of negative samples and their natural distribution in human-agent interactions. In practice, some negatives are semantically close distractors while others are trivially irrelevant, and natural dialogue exhibits structured proportions of these types. Current approaches using synthetic or uniformly sampled negatives fail to reflect this diversity, limiting embedding models' ability to learn nuanced discrimination essential for robust memory retrieval. In this work, we propose a principled data construction framework HiNS that explicitly models negative sample difficulty tiers and incorporates empirically grounded negative ratios derived from conversational data, enabling the training of embedding models with substantially improved retrieval fidelity and generalization in memory-intensive tasks. Experiments show significant improvements: on LoCoMo, F1/BLEU-1 gains of 3.27%/3.30%(MemoryOS) and 1.95%/1.78% (Mem0); on PERSONAMEM, total score improvements of 1.19% (MemoryOS) and 2.55% (Mem0).
>
---
#### [new 019] \textsc{LogicScore}: Fine-grained Logic Evaluation of Conciseness, Completeness, and Determinateness in Attributed Question Answering
- **分类: cs.CL**

- **简介: 该论文属于Attributed Question Answering任务，解决LLM生成答案逻辑不连贯的问题，提出LogicScore框架评估答案的完整性、简洁性和确定性。**

- **链接: [https://arxiv.org/pdf/2601.15050v1](https://arxiv.org/pdf/2601.15050v1)**

> **作者:** Zhichao Yan; Yunxiao Zhao; Jiapu Wang; Jiaoyan Chen; Shaoru Guo; Xiaoli Li; Ru Li; Jeff Z. Pan
>
> **摘要:** Current evaluation methods for Attributed Question Answering (AQA) suffer from \textit{attribution myopia}: they emphasize verification of isolated statements and their attributions but overlook the global logical integrity of long-form answers. Consequently, Large Language Models (LLMs) often produce factually grounded yet logically incoherent responses with elusive deductive gaps. To mitigate this limitation, we present \textsc{LogicScore}, a unified evaluation framework that shifts the paradigm from local assessment to global reasoning scrutiny. Grounded in Horn Rules, our approach integrates a backward verification mechanism to systematically evaluate three key reasoning dimensions: \textit{Completeness} (logically sound deduction), \textit{Conciseness} (non-redundancy), and \textit{Determinateness} (consistent answer entailment). Extensive experiments across three multi-hop QA datasets (HotpotQA, MusiQue, and 2WikiMultiHopQA) and over 20 LLMs (including GPT-5, Gemini-3-Pro, LLaMA3, and task-specific tuned models) reveal a critical capability gap: leading models often achieve high attribution scores (e.g., 92.85\% precision for Gemini-3 Pro) but struggle with global reasoning quality (e.g., 35.11\% Conciseness for Gemini-3 Pro). Our work establishes a robust standard for logical evaluation, highlighting the need to prioritize reasoning coherence alongside factual grounding in LLM development. Codes are available at: https://github.com/zhichaoyan11/LogicScore.
>
---
#### [new 020] Opening the Black Box: A Survey on the Mechanisms of Multi-Step Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机制研究任务，旨在揭示大语言模型多步推理的内部机理。通过分析隐式和显式推理机制，提出研究框架并指出未来方向。**

- **链接: [https://arxiv.org/pdf/2601.14270v1](https://arxiv.org/pdf/2601.14270v1)**

> **作者:** Liangming Pan; Jason Liang; Jiaran Ye; Minglai Yang; Xinyuan Lu; Fengbin Zhu
>
> **备注:** Technical Report
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable abilities to solve problems requiring multiple reasoning steps, yet the internal mechanisms enabling such capabilities remain elusive. Unlike existing surveys that primarily focus on engineering methods to enhance performance, this survey provides a comprehensive overview of the mechanisms underlying LLM multi-step reasoning. We organize the survey around a conceptual framework comprising seven interconnected research questions, from how LLMs execute implicit multi-hop reasoning within hidden activations to how verbalized explicit reasoning remodels the internal computation. Finally, we highlight five research directions for future mechanistic studies.
>
---
#### [new 021] From Chaos to Clarity: Schema-Constrained AI for Auditable Biomedical Evidence Extraction from Full-Text PDFs
- **分类: cs.CL**

- **简介: 该论文属于 biomedical evidence extraction 任务，旨在解决从复杂PDF中准确提取结构化数据的问题。提出一种基于模式约束的AI系统，提升可审计性和可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14267v1](https://arxiv.org/pdf/2601.14267v1)**

> **作者:** Pouria Mortezaagha; Joseph Shaw; Bowen Sun; Arya Rahgozar
>
> **摘要:** Biomedical evidence synthesis relies on accurate extraction of methodological, laboratory, and outcome variables from full-text research articles, yet these variables are embedded in complex scientific PDFs that make manual abstraction time-consuming and difficult to scale. Existing document AI systems remain limited by OCR errors, long-document fragmentation, constrained throughput, and insufficient auditability for high-stakes synthesis. We present a schema-constrained AI extraction system that transforms full-text biomedical PDFs into structured, analysis-ready records by explicitly restricting model inference through typed schemas, controlled vocabularies, and evidence-gated decisions. Documents are ingested using resume-aware hashing, partitioned into caption-aware page-level chunks, and processed asynchronously under explicit concurrency controls. Chunk-level outputs are deterministically merged into study-level records using conflict-aware consolidation, set-based aggregation, and sentence-level provenance to support traceability and post-hoc audit. Evaluated on a corpus of studies on direct oral anticoagulant level measurement, the pipeline processed all documents without manual intervention, maintained stable throughput under service constraints, and exhibited strong internal consistency across document chunks. Iterative schema refinement substantially improved extraction fidelity for synthesis-critical variables, including assay classification, outcome definitions, follow-up duration, and timing of measurement. These results demonstrate that schema-constrained, provenance-aware extraction enables scalable and auditable transformation of heterogeneous scientific PDFs into structured evidence, aligning modern document AI with the transparency and reliability requirements of biomedical evidence synthesis.
>
---
#### [new 022] Can LLM Reasoning Be Trusted? A Comparative Study: Using Human Benchmarking on Statistical Tasks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与统计推理领域，旨在评估大语言模型在统计任务中的表现及推理质量判断能力。通过微调和人类基准对比，验证其在教育和技术应用中的潜力。**

- **链接: [https://arxiv.org/pdf/2601.14479v1](https://arxiv.org/pdf/2601.14479v1)**

> **作者:** Crish Nagarkar; Leonid Bogachev; Serge Sharoff
>
> **摘要:** This paper investigates the ability of large language models (LLMs) to solve statistical tasks, as well as their capacity to assess the quality of reasoning. While state-of-the-art LLMs have demonstrated remarkable performance in a range of NLP tasks, their competence in addressing even moderately complex statistical challenges is not well understood. We have fine-tuned selected open-source LLMs on a specially developed dataset to enhance their statistical reasoning capabilities, and compared their performance with the human scores used as a benchmark. Our results show that the fine-tuned models achieve better performance on advanced statistical tasks on the level comparable to a statistics student. Fine-tuning demonstrates architecture-dependent improvements, with some models showing significant performance gains, indicating clear potential for deployment in educational technology and statistical analysis assistance systems. We also show that LLMs themselves can be far better judges of the answers quality (including explanation and reasoning assessment) in comparison to traditional metrics, such as BLEU or BertScore. This self-evaluation capability enables scalable automated assessment for statistical education platforms and quality assurance in automated analysis tools. Potential applications also include validation tools for research methodology in academic and industry settings, and quality control mechanisms for data analysis workflows.
>
---
#### [new 023] Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话系统评估任务，旨在解决LLM在临床场景中的安全性和准确性问题。通过自动化生成细粒度评估标准，提升模型评价与优化效果。**

- **链接: [https://arxiv.org/pdf/2601.15161v1](https://arxiv.org/pdf/2601.15161v1)**

> **作者:** Yinzhu Chen; Abdine Maiga; Hossein A. Rahmani; Emine Yilmaz
>
> **摘要:** Large Language Models (LLMs) are increasingly used for clinical decision support, where hallucinations and unsafe suggestions may pose direct risks to patient safety. These risks are particularly challenging as they often manifest as subtle clinical errors that evade detection by generic metrics, while expert-authored fine-grained rubrics remain costly to construct and difficult to scale. In this paper, we propose a retrieval-augmented multi-agent framework designed to automate the generation of instance-specific evaluation rubrics. Our approach grounds evaluation in authoritative medical evidence by decomposing retrieved content into atomic facts and synthesizing them with user interaction constraints to form verifiable, fine-grained evaluation criteria. Evaluated on HealthBench, our framework achieves a Clinical Intent Alignment (CIA) score of 60.12%, a statistically significant improvement over the GPT-4o baseline (55.16%). In discriminative tests, our rubrics yield a mean score delta ($μ_Δ = 8.658$) and an AUROC of 0.977, nearly doubling the quality separation achieved by GPT-4o baseline (4.972). Beyond evaluation, our rubrics effectively guide response refinement, improving quality by 9.2% (from 59.0% to 68.2%). This provides a scalable and transparent foundation for both evaluating and improving medical LLMs. The code is available at https://anonymous.4open.science/r/Automated-Rubric-Generation-AF3C/.
>
---
#### [new 024] Quantifying Speaker Embedding Phonological Rule Interactions in Accented Speech Synthesis
- **分类: cs.CL**

- **简介: 该论文属于语音合成任务，旨在解决 accents 控制问题。通过分析发音规则与说话人嵌入的交互，提出 PSR 指标，提升合成语音的准确性与可控性。**

- **链接: [https://arxiv.org/pdf/2601.14417v1](https://arxiv.org/pdf/2601.14417v1)**

> **作者:** Thanathai Lertpetchpun; Yoonjeong Lee; Thanapat Trachu; Jihwan Lee; Tiantian Feng; Dani Byrd; Shrikanth Narayanan
>
> **备注:** Accepted to ICASSP2026
>
> **摘要:** Many spoken languages, including English, exhibit wide variation in dialects and accents, making accent control an important capability for flexible text-to-speech (TTS) models. Current TTS systems typically generate accented speech by conditioning on speaker embeddings associated with specific accents. While effective, this approach offers limited interpretability and controllability, as embeddings also encode traits such as timbre and emotion. In this study, we analyze the interaction between speaker embeddings and linguistically motivated phonological rules in accented speech synthesis. Using American and British English as a case study, we implement rules for flapping, rhoticity, and vowel correspondences. We propose the phoneme shift rate (PSR), a novel metric quantifying how strongly embeddings preserve or override rule-based transformations. Experiments show that combining rules with embeddings yields more authentic accents, while embeddings can attenuate or overwrite rules, revealing entanglement between accent and speaker identity. Our findings highlight rules as a lever for accent control and a framework for evaluating disentanglement in speech generation.
>
---
#### [new 025] The GDN-CC Dataset: Automatic Corpus Clarification for AI-enhanced Democratic Citizen Consultations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决民主咨询数据的标准化问题。通过构建GDN-CC数据集，研究小模型在文本澄清和观点聚类中的应用效果。**

- **链接: [https://arxiv.org/pdf/2601.14944v1](https://arxiv.org/pdf/2601.14944v1)**

> **作者:** Pierre-Antoine Lequeu; Léo Labat; Laurène Cave; Gaël Lejeune; François Yvon; Benjamin Piwowarski
>
> **备注:** 31 pages including 22 for references and appendix, 13 figures
>
> **摘要:** LLMs are ubiquitous in modern NLP, and while their applicability extends to texts produced for democratic activities such as online deliberations or large-scale citizen consultations, ethical questions have been raised for their usage as analysis tools. We continue this line of research with two main goals: (a) to develop resources that can help standardize citizen contributions in public forums at the pragmatic level, and make them easier to use in topic modeling and political analysis; (b) to study how well this standardization can reliably be performed by small, open-weights LLMs, i.e. models that can be run locally and transparently with limited resources. Accordingly, we introduce Corpus Clarification as a preprocessing framework for large-scale consultation data that transforms noisy, multi-topic contributions into structured, self-contained argumentative units ready for downstream analysis. We present GDN-CC, a manually-curated dataset of 1,231 contributions to the French Grand Débat National, comprising 2,285 argumentative units annotated for argumentative structure and manually clarified. We then show that finetuned Small Language Models match or outperform LLMs on reproducing these annotations, and measure their usability for an opinion clustering task. We finally release GDN-CC-large, an automatically annotated corpus of 240k contributions, the largest annotated democratic consultation dataset to date.
>
---
#### [new 026] CorpusQA: A 10 Million Token Benchmark for Corpus-Level Analysis and Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CorpusQA，一个1000万词的基准，用于测试语言模型在文档集上的推理能力。解决现有基准不足的问题，通过合成数据生成复杂查询，验证模型全局信息整合能力。**

- **链接: [https://arxiv.org/pdf/2601.14952v1](https://arxiv.org/pdf/2601.14952v1)**

> **作者:** Zhiyuan Lu; Chenliang Li; Yingcheng Shi; Weizhou Shen; Ming Yan; Fei Huang
>
> **摘要:** While large language models now handle million-token contexts, their capacity for reasoning across entire document repositories remains largely untested. Existing benchmarks are inadequate, as they are mostly limited to single long texts or rely on a "sparse retrieval" assumption-that answers can be derived from a few relevant chunks. This assumption fails for true corpus-level analysis, where evidence is highly dispersed across hundreds of documents and answers require global integration, comparison, and statistical aggregation. To address this critical gap, we introduce CorpusQA, a new benchmark scaling up to 10 million tokens, generated via a novel data synthesis framework. By decoupling reasoning from textual representation, this framework creates complex, computation-intensive queries with programmatically guaranteed ground-truth answers, challenging systems to perform holistic reasoning over vast, unstructured text without relying on fallible human annotation. We further demonstrate the utility of our framework beyond evaluation, showing that fine-tuning on our synthesized data effectively enhances an LLM's general long-context reasoning capabilities. Extensive experiments reveal that even state-of-the-art long-context LLMs struggle as input length increases, and standard retrieval-augmented generation systems collapse entirely. Our findings indicate that memory-augmented agentic architectures offer a more robust alternative, suggesting a critical shift is needed from simply extending context windows to developing advanced architectures for global information synthesis.
>
---
#### [new 027] Comparative Study of Large Language Models on Chinese Film Script Continuation: An Empirical Analysis Based on GPT-5.2 and Qwen-Max
- **分类: cs.CL**

- **简介: 该论文属于中文电影剧本续写任务，比较GPT-5.2与Qwen-Max的性能，通过构建基准和多维评估框架，分析两者在结构和质量上的差异。**

- **链接: [https://arxiv.org/pdf/2601.14826v1](https://arxiv.org/pdf/2601.14826v1)**

> **作者:** Yuxuan Cao; Zida Yang; Ye Wang
>
> **备注:** 18 pages, 6 figures, 6 tables, 20 references. First two authors contributed equally. Corresponding author: Ye Wang (wangye@whu.edu.cn)
>
> **摘要:** As large language models (LLMs) are increasingly applied to creative writing, their performance on culturally specific narrative tasks warrants systematic investigation. This study constructs the first Chinese film script continuation benchmark comprising 53 classic films, and designs a multi-dimensional evaluation framework comparing GPT-5.2 and Qwen-Max-Latest. Using a "first half to second half" continuation paradigm with 3 samples per film, we obtained 303 valid samples (GPT-5.2: 157, 98.7% validity; Qwen-Max: 146, 91.8% validity). Evaluation integrates ROUGE-L, Structural Similarity, and LLM-as-Judge scoring (DeepSeek-Reasoner). Statistical analysis of 144 paired samples reveals: Qwen-Max achieves marginally higher ROUGE-L (0.2230 vs 0.2114, d=-0.43); however, GPT-5.2 significantly outperforms in structural preservation (0.93 vs 0.75, d=0.46), overall quality (44.79 vs 25.72, d=1.04), and composite scores (0.50 vs 0.39, d=0.84). The overall quality effect size reaches large effect level (d>0.8). GPT-5.2 excels in character consistency, tone-style matching, and format preservation, while Qwen-Max shows deficiencies in generation stability. This study provides a reproducible framework for LLM evaluation in Chinese creative writing.
>
---
#### [new 028] Supporting Humans in Evaluating AI Summaries of Legal Depositions
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于法律文本摘要评估任务，旨在提升AI摘要的准确性。通过事实性要点（nuggets）帮助法律人员判断摘要质量并进行人工优化。**

- **链接: [https://arxiv.org/pdf/2601.15182v1](https://arxiv.org/pdf/2601.15182v1)**

> **作者:** Naghmeh Farzi; Laura Dietz; Dave D. Lewis
>
> **备注:** To appear in 2026 ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR '26), March 22-26, 2026, Seattle, WA, USA. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3786304.3787923
>
> **摘要:** While large language models (LLMs) are increasingly used to summarize long documents, this trend poses significant challenges in the legal domain, where the factual accuracy of deposition summaries is crucial. Nugget-based methods have been shown to be extremely helpful for the automated evaluation of summarization approaches. In this work, we translate these methods to the user side and explore how nuggets could directly assist end users. Although prior systems have demonstrated the promise of nugget-based evaluation, its potential to support end users remains underexplored. Focusing on the legal domain, we present a prototype that leverages a factual nugget-based approach to support legal professionals in two concrete scenarios: (1) determining which of two summaries is better, and (2) manually improving an automatically generated summary.
>
---
#### [new 029] Say Anything but This: When Tokenizer Betrays Reasoning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型因分词器导致的推理错误问题，属于自然语言处理任务。通过实验发现分词不一致引发模型误判，提出检测方法并分析常见错误类型。**

- **链接: [https://arxiv.org/pdf/2601.14658v1](https://arxiv.org/pdf/2601.14658v1)**

> **作者:** Navid Ayoobi; Marcus I Armstrong; Arjun Mukherjee
>
> **摘要:** Large language models (LLMs) reason over discrete token ID sequences, yet modern subword tokenizers routinely produce non-unique encodings: multiple token ID sequences can detokenize to identical surface strings. This representational mismatch creates an unmeasured fragility wherein reasoning processes can fail. LLMs may treat two internal representations as distinct "words" even when they are semantically identical at the text level. In this work, we show that tokenization can betray LLM reasoning through one-to-many token ID mappings. We introduce a tokenization-consistency probe that requires models to replace designated target words in context while leaving all other content unchanged. The task is intentionally simple at the surface level, enabling us to attribute failures to tokenizer-detokenizer artifacts rather than to knowledge gaps or parameter limitations. Through analysis of over 11000 replacement trials across state-of-the-art open-source LLMs, we find a non-trivial rate of outputs exhibit phantom edits: cases where models operate under the illusion of correct reasoning, a phenomenon arising from tokenizer-induced representational defects. We further analyze these cases and provide a taxonomy of eight systematic tokenizer artifacts, including whitespace-boundary shifts and intra-word resegmentation. These findings indicate that part of apparent reasoning deficiency originates in the tokenizer layer, motivating tokenizer-level remedies before incurring the cost of training ever-larger models on ever-larger corpora.
>
---
#### [new 030] Project Aletheia: Verifier-Guided Distillation of Backtracking for Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决小模型在约束满足问题中的错误推理问题。通过验证器引导的蒸馏方法，训练模型学会检测并修正错误，提升其推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.14290v1](https://arxiv.org/pdf/2601.14290v1)**

> **作者:** Aradhya Dixit; Tianxi Liang; Jai Telang
>
> **摘要:** Small Language Models (SLMs, under 10B parameters) are attractive for private, on-device deployment, yet they frequently fail on strict constraint-satisfaction problems due to linear, overconfident reasoning traces that do not recover from early mistakes. We introduce Verifier-Guided Distillation, a training protocol that transfers the process of error repair - explicit conflict detection and backtracking - rather than only correct final answers. By training a 7B model on verified reasoning traces that include mistakes and self-corrections, we show that latent verification behavior can emerge in small models, enabling them to occasionally stop, detect contradictions, and revise earlier assumptions.
>
---
#### [new 031] RECAP: Resistance Capture in Text-based Mental Health Counseling with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理辅导中的行为识别任务，解决文本对话中客户抵抗行为检测问题。提出PsyFIRE框架和RECAP系统，提升检测准确性和解释性。**

- **链接: [https://arxiv.org/pdf/2601.14780v1](https://arxiv.org/pdf/2601.14780v1)**

> **作者:** Anqi Li; Yuqian Chen; Yu Lu; Zhaoming Chen; Yuan Xie; Zhenzhong Lan
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** Recognizing and navigating client resistance is critical for effective mental health counseling, yet detecting such behaviors is particularly challenging in text-based interactions. Existing NLP approaches oversimplify resistance categories, ignore the sequential dynamics of therapeutic interventions, and offer limited interpretability. To address these limitations, we propose PsyFIRE, a theoretically grounded framework capturing 13 fine-grained resistance behaviors alongside collaborative interactions. Based on PsyFIRE, we construct the ClientResistance corpus with 23,930 annotated utterances from real-world Chinese text-based counseling, each supported by context-specific rationales. Leveraging this dataset, we develop RECAP, a two-stage framework that detects resistance and fine-grained resistance types with explanations. RECAP achieves 91.25% F1 for distinguishing collaboration and resistance and 66.58% macro-F1 for fine-grained resistance categories classification, outperforming leading prompt-based LLM baselines by over 20 points. Applied to a separate counseling dataset and a pilot study with 62 counselors, RECAP reveals the prevalence of resistance, its negative impact on therapeutic relationships and demonstrates its potential to improve counselors' understanding and intervention strategies.
>
---
#### [new 032] Social Caption: Evaluating Social Understanding in Multimodal Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多模态模型的社会理解评估任务，旨在解决如何衡量模型在社交互动中的理解能力。提出Social Caption框架，从三个维度评估模型的社会推理、整体分析和定向分析能力。**

- **链接: [https://arxiv.org/pdf/2601.14569v1](https://arxiv.org/pdf/2601.14569v1)**

> **作者:** Bhaavanaa Thumu; Leena Mathur; Youssouf Kebe; Louis-Philippe Morency
>
> **备注:** 24 pages
>
> **摘要:** Social understanding abilities are crucial for multimodal large language models (MLLMs) to interpret human social interactions. We introduce Social Caption, a framework grounded in interaction theory to evaluate social understanding abilities of MLLMs along three dimensions: Social Inference (SI), the ability to make accurate inferences about interactions; Holistic Social Analysis (HSA), the ability to generate comprehensive descriptions of interactions; Directed Social Analysis (DSA), the ability to extract relevant social information from interactions. We analyze factors influencing model performance in social understanding, such as scale, architectural design, and spoken context. Experiments with MLLM judges contribute insights about scaling automated evaluation of multimodal social understanding.
>
---
#### [new 033] The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，探讨扩散语言模型的任意顺序生成问题。研究发现，这种灵活性反而限制了推理能力，提出JustGRPO方法提升效果。**

- **链接: [https://arxiv.org/pdf/2601.15165v1](https://arxiv.org/pdf/2601.15165v1)**

> **作者:** Zanlin Ni; Shenzhi Wang; Yang Yue; Tianyu Yu; Weilin Zhao; Yeguo Hua; Tianyi Chen; Jun Song; Cheng Yu; Bo Zheng; Gao Huang
>
> **备注:** Code and pre-trained models: https://github.com/LeapLabTHU/JustGRPO
>
> **摘要:** Diffusion Large Language Models (dLLMs) break the rigid left-to-right constraint of traditional LLMs, enabling token generation in arbitrary orders. Intuitively, this flexibility implies a solution space that strictly supersets the fixed autoregressive trajectory, theoretically unlocking superior reasoning potential for general tasks like mathematics and coding. Consequently, numerous works have leveraged reinforcement learning (RL) to elicit the reasoning capability of dLLMs. In this paper, we reveal a counter-intuitive reality: arbitrary order generation, in its current form, narrows rather than expands the reasoning boundary of dLLMs. We find that dLLMs tend to exploit this order flexibility to bypass high-uncertainty tokens that are crucial for exploration, leading to a premature collapse of the solution space. This observation challenges the premise of existing RL approaches for dLLMs, where considerable complexities, such as handling combinatorial trajectories and intractable likelihoods, are often devoted to preserving this flexibility. We demonstrate that effective reasoning is better elicited by intentionally forgoing arbitrary order and applying standard Group Relative Policy Optimization (GRPO) instead. Our approach, JustGRPO, is minimalist yet surprisingly effective (e.g., 89.1% accuracy on GSM8K) while fully retaining the parallel decoding ability of dLLMs. Project page: https://nzl-thu.github.io/the-flexibility-trap
>
---
#### [new 034] CodeDelegator: Mitigating Context Pollution via Role Separation in Code-as-Action Agents
- **分类: cs.CL**

- **简介: 该论文提出CodeDelegator，解决代码代理中上下文污染问题。通过角色分离，将规划与实现分开，提升长周期任务性能。**

- **链接: [https://arxiv.org/pdf/2601.14914v1](https://arxiv.org/pdf/2601.14914v1)**

> **作者:** Tianxiang Fei; Cheng Chen; Yue Pan; Mao Zheng; Mingyang Song
>
> **摘要:** Recent advances in large language models (LLMs) allow agents to represent actions as executable code, offering greater expressivity than traditional tool-calling. However, real-world tasks often demand both strategic planning and detailed implementation. Using a single agent for both leads to context pollution from debugging traces and intermediate failures, impairing long-horizon performance. We propose CodeDelegator, a multi-agent framework that separates planning from implementation via role specialization. A persistent Delegator maintains strategic oversight by decomposing tasks, writing specifications, and monitoring progress without executing code. For each sub-task, a new Coder agent is instantiated with a clean context containing only its specification, shielding it from prior failures. To coordinate between agents, we introduce Ephemeral-Persistent State Separation (EPSS), which isolates each Coder's execution state while preserving global coherence, preventing debugging traces from polluting the Delegator's context. Experiments on various benchmarks demonstrate the effectiveness of CodeDelegator across diverse scenarios.
>
---
#### [new 035] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的隐私安全任务，旨在解决语言模型在微调后出现的隐私泄露问题。研究发现微调会导致模型失去对上下文隐私的判断能力，暴露严重隐私漏洞。**

- **链接: [https://arxiv.org/pdf/2601.15220v1](https://arxiv.org/pdf/2601.15220v1)**

> **作者:** Anmol Goel; Cornelius Emde; Sangdoo Yun; Seong Joon Oh; Martin Gubri
>
> **摘要:** We identify a novel phenomenon in language models: benign fine-tuning of frontier models can lead to privacy collapse. We find that diverse, subtle patterns in training data can degrade contextual privacy, including optimisation for helpfulness, exposure to user information, emotional and subjective dialogue, and debugging code printing internal variables, among others. Fine-tuned models lose their ability to reason about contextual privacy norms, share information inappropriately with tools, and violate memory boundaries across contexts. Privacy collapse is a ``silent failure'' because models maintain high performance on standard safety and utility benchmarks whilst exhibiting severe privacy vulnerabilities. Our experiments show evidence of privacy collapse across six models (closed and open weight), five fine-tuning datasets (real-world and controlled data), and two task categories (agentic and memory-based). Our mechanistic analysis reveals that privacy representations are uniquely fragile to fine-tuning, compared to task-relevant features which are preserved. Our results reveal a critical gap in current safety evaluations, in particular for the deployment of specialised agents.
>
---
#### [new 036] Typhoon OCR: Open Vision-Language Model For Thai Document Extraction
- **分类: cs.CL**

- **简介: 论文聚焦于泰语文档提取任务，解决现有视觉语言模型对泰语支持不足的问题。通过构建专用数据集并微调模型，实现高效准确的文本和版式识别。**

- **链接: [https://arxiv.org/pdf/2601.14722v1](https://arxiv.org/pdf/2601.14722v1)**

> **作者:** Surapon Nonesung; Natapong Nitarach; Teetouch Jaknamon; Pittawat Taveekitworachai; Kunat Pipatanakul
>
> **摘要:** Document extraction is a core component of digital workflows, yet existing vision-language models (VLMs) predominantly favor high-resource languages. Thai presents additional challenges due to script complexity from non-latin letters, the absence of explicit word boundaries, and the prevalence of highly unstructured real-world documents, limiting the effectiveness of current open-source models. This paper presents Typhoon OCR, an open VLM for document extraction tailored for Thai and English. The model is fine-tuned from vision-language backbones using a Thai-focused training dataset. The dataset is developed using a multi-stage data construction pipeline that combines traditional OCR, VLM-based restructuring, and curated synthetic data. Typhoon OCR is a unified framework capable of text transcription, layout reconstruction, and document-level structural consistency. The latest iteration of our model, Typhoon OCR V1.5, is a compact and inference-efficient model designed to reduce reliance on metadata and simplify deployment. Comprehensive evaluations across diverse Thai document categories, including financial reports, government forms, books, infographics, and handwritten documents, show that Typhoon OCR achieves performance comparable to or exceeding larger frontier proprietary models, despite substantially lower computational cost. The results demonstrate that open vision-language OCR models can achieve accurate text extraction and layout reconstruction for Thai documents, reaching performance comparable to proprietary systems while remaining lightweight and deployable.
>
---
#### [new 037] PodBench: A Comprehensive Benchmark for Instruction-Aware Audio-Oriented Podcast Script Generation
- **分类: cs.CL**

- **简介: 该论文属于语音导向的播客脚本生成任务，旨在解决缺乏系统评估资源的问题。提出PodBench基准和多维评估框架，对比不同模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14903v1](https://arxiv.org/pdf/2601.14903v1)**

> **作者:** Chenning Xu; Mao Zheng; Mingyu Zheng; Mingyang Song
>
> **摘要:** Podcast script generation requires LLMs to synthesize structured, context-grounded dialogue from diverse inputs, yet systematic evaluation resources for this task remain limited. To bridge this gap, we introduce PodBench, a benchmark comprising 800 samples with inputs up to 21K tokens and complex multi-speaker instructions. We propose a multifaceted evaluation framework that integrates quantitative constraints with LLM-based quality assessment. Extensive experiments reveal that while proprietary models generally excel, open-source models equipped with explicit reasoning demonstrate superior robustness in handling long contexts and multi-speaker coordination compared to standard baselines. However, our analysis uncovers a persistent divergence where high instruction following does not guarantee high content substance. PodBench offers a reproducible testbed to address these challenges in long-form, audio-centric generation.
>
---
#### [new 038] AdaTIR: Adaptive Tool-Integrated Reasoning via Difficulty-Aware Policy Optimization
- **分类: cs.CL**

- **简介: 该论文提出AdaTIR，解决LLM在任务中过度调用工具的问题。通过难度感知机制，减少工具调用并提升推理能力，属于增强智能体决策的任务。**

- **链接: [https://arxiv.org/pdf/2601.14696v1](https://arxiv.org/pdf/2601.14696v1)**

> **作者:** Zhaiyu Fang; Ruipeng Sun
>
> **备注:** under review
>
> **摘要:** Tool-Integrated Reasoning (TIR) has significantly enhanced the capabilities of Large Language Models (LLMs), yet current agents tend to exhibit cognitive offloading, redundantly invoking external tools even for simple tasks. In this paper, we suggest that true agentic intelligence requires not just tool invocation, but the adaptive wisdom to discern when to use them. We propose AdaTIR, a framework that shifts the paradigm from static tool invocation to difficulty-aware reasoning internalization. By introducing a difficulty-aware efficiency reward, AdaTIR dynamically adjusts tool budgets based on task complexity--internalizing reasoning for simple tasks while selectively invoking tools for complex tasks. Furthermore, we identify a sign reversal problem where tool penalties outweigh correctness rewards, mistakenly penalizing correct rollouts with negative advantages. To resolve this, we propose Clipped Advantage Shaping (CAS), which ensures that correctness remains the primary objective while using efficiency as a secondary constraint. Empirical results demonstrate that AdaTIR reduces tool calls by up to 97.6% on simple tasks and 28.2% on complex challenges while maintaining or enhancing accuracy. Notably, AdaTIR successfully internalizes reasoning, outperforming baselines by 4.8% on AIME 2024 even when tool access is strictly disabled.
>
---
#### [new 039] Metadata Conditioned Large Language Models for Localization
- **分类: cs.CL**

- **简介: 该论文属于语言模型本地化任务，旨在解决模型地理性能不均问题。通过元数据条件化提升区域性能，同时保持全局泛化能力，验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2601.15236v1](https://arxiv.org/pdf/2601.15236v1)**

> **作者:** Anjishnu Mukherjee; Ziwei Zhu; Antonios Anastasopoulos
>
> **备注:** under review
>
> **摘要:** Large language models are typically trained by treating text as a single global distribution, often resulting in geographically homogenized behavior. We study metadata conditioning as a lightweight approach for localization, pre-training 31 models (at 0.5B and 1B parameter scales) from scratch on large-scale English news data annotated with verified URLs, country tags, and continent tags, covering 4 continents and 17 countries. Across four controlled experiments, we show that metadata conditioning consistently improves in-region performance without sacrificing cross-region generalization, enables global models to recover localization comparable to region-specific models, and improves learning efficiency. Our ablation studies demonstrate that URL-level metadata alone captures much of the geographic signal, while balanced regional data coverage remains essential, as metadata cannot fully compensate for missing regions. Finally, we introduce a downstream benchmark of 800 localized news MCQs and show that after instruction tuning, metadata conditioned global models achieve accuracy comparable to LLaMA-3.2-1B-Instruct, despite being trained on substantially less data. Together, these results establish metadata conditioning as a practical and compute-efficient approach for localization of language models.
>
---
#### [new 040] SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于搜索代理训练任务，解决RL训练中真实环境成本高与数据对齐问题。提出SearchGym模拟环境，构建准确知识图谱，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2601.14615v1](https://arxiv.org/pdf/2601.14615v1)**

> **作者:** Xichen Zhang; Ziyi He; Yinghao Zhu; Sitong Wu; Shaozuo Yu; Meng Chu; Wenhu Zhang; Haoru Tan; Jiaya Jia
>
> **摘要:** Search agents have emerged as a pivotal paradigm for solving open-ended, knowledge-intensive reasoning tasks. However, training these agents via Reinforcement Learning (RL) faces a critical dilemma: interacting with live commercial Web APIs is prohibitively expensive, while relying on static data snapshots often introduces noise due to data misalignment. This misalignment generates corrupted reward signals that destabilize training by penalizing correct reasoning or rewarding hallucination. To address this, we propose SearchGym, a simulation environment designed to bootstrap robust search agents. SearchGym employs a rigorous generative pipeline to construct a verifiable knowledge graph and an aligned document corpus, ensuring that every reasoning task is factually grounded and strictly solvable. Building on this controllable environment, we introduce SearchGym-RL, a curriculum learning methodology that progressively optimizes agent policies through purified feedback, evolving from basic interactions to complex, long-horizon planning. Extensive experiments across the Llama and Qwen families demonstrate strong Sim-to-Real generalization. Notably, our Qwen2.5-7B-Base model trained within SearchGym surpasses the web-enhanced ASearcher baseline across nine diverse benchmarks by an average relative margin of 10.6%. Our results validate that high-fidelity simulation serves as a scalable and highly cost-effective methodology for developing capable search agents.
>
---
#### [new 041] RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于学术论文理解任务，旨在解决模型在科学文本理解上的不足。提出RPC-Bench基准，包含15K QA对，评估模型在原因、内容和方法的理解能力。**

- **链接: [https://arxiv.org/pdf/2601.14289v1](https://arxiv.org/pdf/2601.14289v1)**

> **作者:** Yelin Chen; Fanjin Zhang; Suping Sun; Yunhe Pang; Yuanchun Wang; Jian Song; Xiaoyan Li; Lei Hou; Shu Zhao; Jie Tang; Juanzi Li
>
> **备注:** 11 pages, 21 appendix pages
>
> **摘要:** Understanding research papers remains challenging for foundation models due to specialized scientific discourse and complex figures and tables, yet existing benchmarks offer limited fine-grained evaluation at scale. To address this gap, we introduce RPC-Bench, a large-scale question-answering benchmark built from review-rebuttal exchanges of high-quality computer science papers, containing 15K human-verified QA pairs. We design a fine-grained taxonomy aligned with the scientific research flow to assess models' ability to understand and answer why, what, and how questions in scholarly contexts. We also define an elaborate LLM-human interaction annotation framework to support large-scale labeling and quality control. Following the LLM-as-a-Judge paradigm, we develop a scalable framework that evaluates models on correctness-completeness and conciseness, with high agreement to human judgment. Experiments reveal that even the strongest models (GPT-5) achieve only 68.2% correctness-completeness, dropping to 37.46% after conciseness adjustment, highlighting substantial gaps in precise academic paper understanding. Our code and data are available at https://rpc-bench.github.io/.
>
---
#### [new 042] The Slow Drift of Support: Boundary Failures in Multi-Turn Mental Health LLM Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全评估任务，旨在解决多轮对话中LLM安全边界逐渐失效的问题。通过压力测试框架，发现模型在长期互动中易突破安全界限。**

- **链接: [https://arxiv.org/pdf/2601.14269v1](https://arxiv.org/pdf/2601.14269v1)**

> **作者:** Youyou Cheng; Zhuangwei Kang; Kerry Jiang; Chenyu Sun; Qiyang Pan
>
> **摘要:** Large language models (LLMs) have been widely used for mental health support. However, current safety evaluations in this field are mostly limited to detecting whether LLMs output prohibited words in single-turn conversations, neglecting the gradual erosion of safety boundaries in long dialogues. Examples include making definitive guarantees, assuming responsibility, and playing professional roles. We believe that with the evolution of mainstream LLMs, words with obvious safety risks are easily filtered by their underlying systems, while the real danger lies in the gradual transgression of boundaries during multi-turn interactions, driven by the LLM's attempts at comfort and empathy. This paper proposes a multi-turn stress testing framework and conducts long-dialogue safety tests on three cutting-edge LLMs using two pressure methods: static progression and adaptive probing. We generated 50 virtual patient profiles and stress-tested each model through up to 20 rounds of virtual psychiatric dialogues. The experimental results show that violations are common, and both pressure modes produced similar violation rates. However, adaptive probing significantly advanced the time at which models crossed boundaries, reducing the average number of turns from 9.21 in static progression to 4.64. Under both mechanisms, making definitive or zero-risk promises was the primary way in which boundaries were breached. These findings suggest that the robustness of LLM safety boundaries cannot be inferred solely through single-turn tests; it is necessary to fully consider the wear and tear on safety boundaries caused by different interaction pressures and characteristics in extended dialogues.
>
---
#### [new 043] Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在解决情感操纵导致的检测脆弱性问题。通过构建AdSent框架，提升模型在情感变化下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.15277v1](https://arxiv.org/pdf/2601.15277v1)**

> **作者:** Sahar Tahmasebi; Eric Müller-Budack; Ralph Ewerth
>
> **摘要:** Misinformation and fake news have become a pressing societal challenge, driving the need for reliable automated detection methods. Prior research has highlighted sentiment as an important signal in fake news detection, either by analyzing which sentiments are associated with fake news or by using sentiment and emotion features for classification. However, this poses a vulnerability since adversaries can manipulate sentiment to evade detectors especially with the advent of large language models (LLMs). A few studies have explored adversarial samples generated by LLMs, but they mainly focus on stylistic features such as writing style of news publishers. Thus, the crucial vulnerability of sentiment manipulation remains largely unexplored. In this paper, we investigate the robustness of state-of-the-art fake news detectors under sentiment manipulation. We introduce AdSent, a sentiment-robust detection framework designed to ensure consistent veracity predictions across both original and sentiment-altered news articles. Specifically, we (1) propose controlled sentiment-based adversarial attacks using LLMs, (2) analyze the impact of sentiment shifts on detection performance. We show that changing the sentiment heavily impacts the performance of fake news detection models, indicating biases towards neutral articles being real, while non-neutral articles are often classified as fake content. (3) We introduce a novel sentiment-agnostic training strategy that enhances robustness against such perturbations. Extensive experiments on three benchmark datasets demonstrate that AdSent significantly outperforms competitive baselines in both accuracy and robustness, while also generalizing effectively to unseen datasets and adversarial scenarios.
>
---
#### [new 044] Is Peer Review Really in Decline? Analyzing Review Quality across Venues and Time
- **分类: cs.CL**

- **简介: 该论文属于学术评价研究，旨在检验同行评审质量是否下降。通过分析AI顶会数据，提出量化框架，发现评审质量未持续下降。**

- **链接: [https://arxiv.org/pdf/2601.15172v1](https://arxiv.org/pdf/2601.15172v1)**

> **作者:** Ilia Kuznetsov; Rohan Nayak; Alla Rozovskaya; Iryna Gurevych
>
> **摘要:** Peer review is at the heart of modern science. As submission numbers rise and research communities grow, the decline in review quality is a popular narrative and a common concern. Yet, is it true? Review quality is difficult to measure, and the ongoing evolution of reviewing practices makes it hard to compare reviews across venues and time. To address this, we introduce a new framework for evidence-based comparative study of review quality and apply it to major AI and machine learning conferences: ICLR, NeurIPS and *ACL. We document the diversity of review formats and introduce a new approach to review standardization. We propose a multi-dimensional schema for quantifying review quality as utility to editors and authors, coupled with both LLM-based and lightweight measurements. We study the relationships between measurements of review quality, and its evolution over time. Contradicting the popular narrative, our cross-temporal analysis reveals no consistent decline in median review quality across venues and years. We propose alternative explanations, and outline recommendations to facilitate future empirical studies of review quality.
>
---
#### [new 045] Hallucination-Free Automatic Question & Answer Generation for Intuitive Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育内容生成任务，旨在解决LLM生成MCQ时的幻觉问题。通过多智能体框架和优化策略，显著降低幻觉率，提升题目质量。**

- **链接: [https://arxiv.org/pdf/2601.14280v1](https://arxiv.org/pdf/2601.14280v1)**

> **作者:** Nicholas X. Wang; Aggelos K. Katsaggelos
>
> **摘要:** Hallucinations in large language models (LLMs), defined as fluent yet incorrect or incoherent outputs, pose a significant challenge to the automatic generation of educational multiple-choice questions (MCQs). We identified four key hallucination types in MCQ generation: reasoning inconsistencies, insolvability, factual errors, and mathematical errors. To address this, we propose a hallucination-free multi-agent generation framework that breaks down MCQ generation into discrete, verifiable stages. Our framework utilizes both rule-based and LLM-based detection agents, as well as hallucination scoring metrics to optimize question quality. We redefined MCQ generation as an optimization task minimizing hallucination risk while maximizing validity, answerability, and cost-efficiency. We also introduce an agent-led refinement process that uses counterfactual reasoning and chain-of-thought (CoT) to iteratively improve hallucination in question generation. We evaluated a sample of AP- aligned STEM questions, where our system reduced hallucination rates by over 90% compared to baseline generation while preserving the educational value and style of questions. Our results demonstrate that structured multi-agent collaboration can mitigate hallucinations in educational content creation at scale, paving the way for more reliable LLM-powered learning tools.
>
---
#### [new 046] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出RoT框架，将文本推理过程转化为图像，解决LLM推理可解释性差的问题，实现更高效的推理。**

- **链接: [https://arxiv.org/pdf/2601.14750v1](https://arxiv.org/pdf/2601.14750v1)**

> **作者:** Yifan Wang; Shiyu Li; Peiming Li; Xiaochen Yang; Yang Tang; Zheng Wei
>
> **摘要:** Chain-of-Thought (CoT) prompting has achieved remarkable success in unlocking the reasoning capabilities of Large Language Models (LLMs). Although CoT prompting enhances reasoning, its verbosity imposes substantial computational overhead. Recent works often focus exclusively on outcome alignment and lack supervision on the intermediate reasoning process. These deficiencies obscure the analyzability of the latent reasoning chain. To address these challenges, we introduce Render-of-Thought (RoT), the first framework to reify the reasoning chain by rendering textual steps into images, making the latent rationale explicit and traceable. Specifically, we leverage the vision encoders of existing Vision Language Models (VLMs) as semantic anchors to align the vision embeddings with the textual space. This design ensures plug-and-play implementation without incurring additional pre-training overhead. Extensive experiments on mathematical and logical reasoning benchmarks demonstrate that our method achieves 3-4x token compression and substantial inference acceleration compared to explicit CoT. Furthermore, it maintains competitive performance against other methods, validating the feasibility of this paradigm. Our code is available at https://github.com/TencentBAC/RoT
>
---
#### [new 047] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于多智能体系统研究，旨在解决MAS设计不足与效果不确定的问题。提出MAS-Orchestra框架和MASBENCH基准，提升多智能体协作效率与理解。**

- **链接: [https://arxiv.org/pdf/2601.14652v1](https://arxiv.org/pdf/2601.14652v1)**

> **作者:** Zixuan Ke; Yifei Ming; Austin Xu; Ryan Chin; Xuan-Phi Nguyen; Prathyusha Jwalapuram; Semih Yavuz; Caiming Xiong; Shafiq Joty
>
> **备注:** Preprint; Work in Progress
>
> **摘要:** While multi-agent systems (MAS) promise elevated intelligence through coordination of agents, current approaches to automatic MAS design under-deliver. Such shortcomings stem from two key factors: (1) methodological complexity - agent orchestration is performed using sequential, code-level execution that limits global system-level holistic reasoning and scales poorly with agent complexity - and (2) efficacy uncertainty - MAS are deployed without understanding if there are tangible benefits compared to single-agent systems (SAS). We propose MAS-Orchestra, a training-time framework that formulates MAS orchestration as a function-calling reinforcement learning problem with holistic orchestration, generating an entire MAS at once. In MAS-Orchestra, complex, goal-oriented sub-agents are abstracted as callable functions, enabling global reasoning over system structure while hiding internal execution details. To rigorously study when and why MAS are beneficial, we introduce MASBENCH, a controlled benchmark that characterizes tasks along five axes: Depth, Horizon, Breadth, Parallel, and Robustness. Our analysis reveals that MAS gains depend critically on task structure, verification protocols, and the capabilities of both orchestrator and sub-agents, rather than holding universally. Guided by these insights, MAS-Orchestra achieves consistent improvements on public benchmarks including mathematical reasoning, multi-hop QA, and search-based QA. Together, MAS-Orchestra and MASBENCH enable better training and understanding of MAS in the pursuit of multi-agent intelligence.
>
---
#### [new 048] The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能代理行为解释任务，旨在解决代理行为原因不明的问题。提出一种框架，通过层次化分析识别驱动代理行为的内部因素。**

- **链接: [https://arxiv.org/pdf/2601.15075v1](https://arxiv.org/pdf/2601.15075v1)**

> **作者:** Chen Qian; Peng Wang; Dongrui Liu; Junyao Yang; Dadi Guo; Ling Tang; Jilin Mei; Qihan Ren; Shuai Shao; Yong Liu; Jie Fu; Jing Shao; Xia Hu
>
> **摘要:** Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining the reasoning behind agent behaviors. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems.
>
---
#### [new 049] Generating consensus and dissent on massive discussion platforms with an $O(N)$ semantic-vector model
- **分类: physics.soc-ph; cond-mat.stat-mech; cs.CL**

- **简介: 该论文研究如何在大规模讨论平台中生成共识与分歧，属于集体智能任务。针对用户固守初始观点的问题，提出基于$O(N)$模型的动态系统，通过调节参数控制共识与分歧程度，优化平台的群体决策效果。**

- **链接: [https://arxiv.org/pdf/2601.13932v1](https://arxiv.org/pdf/2601.13932v1)**

> **作者:** A. Ferrer; D. Muñoz-Jordán; A. Rivero; A. Tarancón; C. Tarancón; D. Yllanes
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Reaching consensus on massive discussion networks is critical for reducing noise and achieving optimal collective outcomes. However, the natural tendency of humans to preserve their initial ideas constrains the emergence of global solutions. To address this, Collective Intelligence (CI) platforms facilitate the discovery of globally superior solutions. We introduce a dynamical system based on the standard $O(N)$ model to drive the aggregation of semantically similar ideas. The system consists of users represented as nodes in a $d=2$ lattice with nearest-neighbor interactions, where their ideas are represented by semantic vectors computed with a pretrained embedding model. We analyze the system's equilibrium states as a function of the coupling parameter $β$. Our results show that $β> 0$ drives the system toward a ferromagnetic-like phase (global consensus), while $β< 0$ induces an antiferromagnetic-like state (maximum dissent), where users maximize semantic distance from their neighbors. This framework offers a controllable method for managing the tradeoff between cohesion and diversity in CI platforms.
>
---
#### [new 050] VisTIRA: Closing the Image-Text Modality Gap in Visual Math Reasoning via Structured Tool Integration
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉数学推理任务，旨在解决图像与文本模态间的推理差距。通过引入VisTIRA框架和合成数据集，提升视觉数学问题的解决能力。**

- **链接: [https://arxiv.org/pdf/2601.14440v1](https://arxiv.org/pdf/2601.14440v1)**

> **作者:** Saeed Khaki; Ashudeep Singh; Nima Safaei; Kamal Ginotra
>
> **摘要:** Vision-language models (VLMs) lag behind text-only language models on mathematical reasoning when the same problems are presented as images rather than text. We empirically characterize this as a modality gap: the same question in text form yields markedly higher accuracy than its visually typeset counterpart, due to compounded failures in reading dense formulas, layout, and mixed symbolic-diagrammatic context. First, we introduce VisTIRA (Vision and Tool-Integrated Reasoning Agent), a tool-integrated reasoning framework that enables structured problem solving by iteratively decomposing a given math problem (as an image) into natural language rationales and executable Python steps to determine the final answer. Second, we build a framework to measure and improve visual math reasoning: a LaTeX-based pipeline that converts chain-of-thought math corpora (e.g., NuminaMath) into challenging image counterparts, and a large set of synthetic tool-use trajectories derived from a real-world, homework-style image dataset (called SnapAsk) for fine-tuning VLMs. Our experiments show that tool-integrated supervision improves image-based reasoning, and OCR grounding can further narrow the gap for smaller models, although its benefit diminishes at scale. These findings highlight that modality gap severity inversely correlates with model size, and that structured reasoning and OCR-based grounding are complementary strategies for advancing visual mathematical reasoning.
>
---
#### [new 051] What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究低比特量化感知训练（QAT）在推理大模型中的效果，旨在提升推理效率同时保持精度。通过实验发现知识蒸馏、PTQ初始化等方法有效，提出优化流程Reasoning-QAT，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2601.14888v1](https://arxiv.org/pdf/2601.14888v1)**

> **作者:** Keyu Lv; Manyi Zhang; Xiaobo Xia; Jingchen Ni; Shannan Yan; Xianzhi Yu; Lu Hou; Chun Yuan; Haoli Bai
>
> **摘要:** Reasoning models excel at complex tasks such as coding and mathematics, yet their inference is often slow and token-inefficient. To improve the inference efficiency, post-training quantization (PTQ) usually comes with the cost of large accuracy drops, especially for reasoning tasks under low-bit settings. In this study, we present a systematic empirical study of quantization-aware training (QAT) for reasoning models. Our key findings include: (1) Knowledge distillation is a robust objective for reasoning models trained via either supervised fine-tuning or reinforcement learning; (2) PTQ provides a strong initialization for QAT, improving accuracy while reducing training cost; (3) Reinforcement learning remains feasible for quantized models given a viable cold start and yields additional gains; and (4) Aligning the PTQ calibration domain with the QAT training domain accelerates convergence and often improves the final accuracy. Finally, we consolidate these findings into an optimized workflow (Reasoning-QAT), and show that it consistently outperforms state-of-the-art PTQ methods across multiple LLM backbones and reasoning datasets. For instance, on Qwen3-0.6B, it surpasses GPTQ by 44.53% on MATH-500 and consistently recovers performance in the 2-bit regime.
>
---
#### [new 052] Designing KRIYA: An AI Companion for Wellbeing Self-Reflection
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人机交互任务，旨在解决用户难以从健康数据中获得有意义理解的问题。通过设计AI同伴KRIYA，促进自我反思与数据解读，提升用户对健康数据的感知与信任。**

- **链接: [https://arxiv.org/pdf/2601.14589v1](https://arxiv.org/pdf/2601.14589v1)**

> **作者:** Shanshan Zhu; Wenxuan Song; Jiayue Melissa Shi; Dong Whi Yoo; Karthik S. Bhat; Koustuv Saha
>
> **摘要:** Most personal wellbeing apps present summative dashboards of health and physical activity metrics, yet many users struggle to translate this information into meaningful understanding. These apps commonly support engagement through goals, reminders, and structured targets, which can reinforce comparison, judgment, and performance anxiety. To explore a complementary approach that prioritizes self-reflection, we design KRIYA, an AI wellbeing companion that supports co-interpretive engagement with personal wellbeing data. KRIYA aims to collaborate with users to explore questions, explanations, and future scenarios through features such as Comfort Zone, Detective Mode, and What-If Planning. We conducted semi-structured interviews with 18 college students interacting with a KRIYA prototype using hypothetical data. Our findings show that through KRIYA interaction, users framed engaging with wellbeing data as interpretation rather than performance, experienced reflection as supportive or pressuring depending on emotional framing, and developed trust through transparency. We discuss design implications for AI companions that support curiosity, self-compassion, and reflective sensemaking of personal health data.
>
---
#### [new 053] Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding in the Complex Spectrum
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音合成任务，旨在解决神经声码器的韵律建模不足和相位重建不准确问题。通过引入韵律引导的谐波注意力机制和直接预测复频谱，提升语音自然度与音高准确性。**

- **链接: [https://arxiv.org/pdf/2601.14472v1](https://arxiv.org/pdf/2601.14472v1)**

> **作者:** Mohammed Salah Al-Radhi; Riad Larbi; Mátyás Bartalis; Géza Németh
>
> **备注:** 5 pages, 2 figures, 1 table. Accepted for presentation at ICASSP 2026
>
> **摘要:** Neural vocoders are central to speech synthesis; despite their success, most still suffer from limited prosody modeling and inaccurate phase reconstruction. We propose a vocoder that introduces prosody-guided harmonic attention to enhance voiced segment encoding and directly predicts complex spectral components for waveform synthesis via inverse STFT. Unlike mel-spectrogram-based approaches, our design jointly models magnitude and phase, ensuring phase coherence and improved pitch fidelity. To further align with perceptual quality, we adopt a multi-objective training strategy that integrates adversarial, spectral, and phase-aware losses. Experiments on benchmark datasets demonstrate consistent gains over HiFi-GAN and AutoVocoder: F0 RMSE reduced by 22 percent, voiced/unvoiced error lowered by 18 percent, and MOS scores improved by 0.15. These results show that prosody-guided attention combined with direct complex spectrum modeling yields more natural, pitch-accurate, and robust synthetic speech, setting a strong foundation for expressive neural vocoding.
>
---
#### [new 054] Evaluation of Large Language Models in Legal Applications: Challenges, Methods, and Future Directions
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 本文探讨了大语言模型在法律领域的应用评估，旨在解决其在法律任务中的可靠性与可信度问题。论文分析了评估挑战，综述了现有方法并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.15267v1](https://arxiv.org/pdf/2601.15267v1)**

> **作者:** Yiran Hu; Huanghai Liu; Chong Wang; Kunran Li; Tien-Hsuan Wu; Haitao Li; Xinran Xu; Siqing Huo; Weihang Su; Ning Zheng; Siyuan Zheng; Qingyao Ai; Yun Liu; Renjun Bian; Yiqun Liu; Charles L. A. Clarke; Weixing Shen; Ben Kao
>
> **摘要:** Large language models (LLMs) are being increasingly integrated into legal applications, including judicial decision support, legal practice assistance, and public-facing legal services. While LLMs show strong potential in handling legal knowledge and tasks, their deployment in real-world legal settings raises critical concerns beyond surface-level accuracy, involving the soundness of legal reasoning processes and trustworthy issues such as fairness and reliability. Systematic evaluation of LLM performance in legal tasks has therefore become essential for their responsible adoption. This survey identifies key challenges in evaluating LLMs for legal tasks grounded in real-world legal practice. We analyze the major difficulties involved in assessing LLM performance in the legal domain, including outcome correctness, reasoning reliability, and trustworthiness. Building on these challenges, we review and categorize existing evaluation methods and benchmarks according to their task design, datasets, and evaluation metrics. We further discuss the extent to which current approaches address these challenges, highlight their limitations, and outline future research directions toward more realistic, reliable, and legally grounded evaluation frameworks for LLMs in legal domains.
>
---
#### [new 055] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型的任务进度推理问题，提出Progress-Bench基准和ProgressLM-45K数据集，探索两种推理方法，发现多数模型在任务进度估计上表现不佳。**

- **链接: [https://arxiv.org/pdf/2601.15224v1](https://arxiv.org/pdf/2601.15224v1)**

> **作者:** Jianshu Zhang; Chengxuan Qian; Haosen Sun; Haoran Lu; Dingcheng Wang; Letian Xue; Han Liu
>
> **备注:** Website: https://progresslm.github.io/ProgressLM/
>
> **摘要:** Estimating task progress requires reasoning over long-horizon dynamics rather than recognizing static visual content. While modern Vision-Language Models (VLMs) excel at describing what is visible, it remains unclear whether they can infer how far a task has progressed from partial observations. To this end, we introduce Progress-Bench, a benchmark for systematically evaluating progress reasoning in VLMs. Beyond benchmarking, we further explore a human-inspired two-stage progress reasoning paradigm through both training-free prompting and training-based approach based on curated dataset ProgressLM-45K. Experiments on 14 VLMs show that most models are not yet ready for task progress estimation, exhibiting sensitivity to demonstration modality and viewpoint changes, as well as poor handling of unanswerable cases. While training-free prompting that enforces structured progress reasoning yields limited and model-dependent gains, the training-based ProgressLM-3B achieves consistent improvements even at a small model scale, despite being trained on a task set fully disjoint from the evaluation tasks. Further analyses reveal characteristic error patterns and clarify when and why progress reasoning succeeds or fails.
>
---
#### [new 056] NeuroFilter: Privacy Guardrails for Conversational LLM Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决LLM代理中的隐私泄露问题。提出NeuroFilter框架，通过分析模型激活空间检测隐私违规行为，有效提升检测效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2601.14660v1](https://arxiv.org/pdf/2601.14660v1)**

> **作者:** Saswat Das; Ferdinando Fioretto
>
> **摘要:** This work addresses the computational challenge of enforcing privacy for agentic Large Language Models (LLMs), where privacy is governed by the contextual integrity framework. Indeed, existing defenses rely on LLM-mediated checking stages that add substantial latency and cost, and that can be undermined in multi-turn interactions through manipulation or benign-looking conversational scaffolding. Contrasting this background, this paper makes a key observation: internal representations associated with privacy-violating intent can be separated from benign requests using linear structure. Using this insight, the paper proposes NeuroFilter, a guardrail framework that operationalizes contextual integrity by mapping norm violations to simple directions in the model's activation space, enabling detection even when semantic filters are bypassed. The proposed filter is also extended to capture threats arising during long conversations using the concept of activation velocity, which measures cumulative drift in internal representations across turns. A comprehensive evaluation across over 150,000 interactions and covering models from 7B to 70B parameters, illustrates the strong performance of NeuroFilter in detecting privacy attacks while maintaining zero false positives on benign prompts, all while reducing the computational inference cost by several orders of magnitude when compared to LLM-based agentic privacy defenses.
>
---
#### [new 057] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，解决 streaming 视频实时处理中的性能与内存问题。提出 HERMES 架构，通过层级 KV 缓存实现高效准确的视频流理解。**

- **链接: [https://arxiv.org/pdf/2601.14724v1](https://arxiv.org/pdf/2601.14724v1)**

> **作者:** Haowei Zhang; Shudong Yang; Jinlan Fu; See-Kiong Ng; Xipeng Qiu
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant improvement in offline video understanding. However, extending these capabilities to streaming video inputs, remains challenging, as existing models struggle to simultaneously maintain stable understanding performance, real-time responses, and low GPU memory overhead. To address this challenge, we propose HERMES, a novel training-free architecture for real-time and accurate understanding of video streams. Based on a mechanistic attention investigation, we conceptualize KV cache as a hierarchical memory framework that encapsulates video information across multiple granularities. During inference, HERMES reuses a compact KV cache, enabling efficient streaming understanding under resource constraints. Notably, HERMES requires no auxiliary computations upon the arrival of user queries, thereby guaranteeing real-time responses for continuous video stream interactions, which achieves 10$\times$ faster TTFT compared to prior SOTA. Even when reducing video tokens by up to 68% compared with uniform sampling, HERMES achieves superior or comparable accuracy across all benchmarks, with up to 11.4% gains on streaming datasets.
>
---
#### [new 058] WavLink: Compact Audio--Text Embeddings with a Global Whisper Token
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出WavLink，一种结合Whisper和可学习全局token的音频-文本嵌入模型，解决音频特征表示效率低的问题。通过优化训练策略，实现更小的嵌入尺寸且性能损失小。**

- **链接: [https://arxiv.org/pdf/2601.15118v1](https://arxiv.org/pdf/2601.15118v1)**

> **作者:** Gokul Karthik Kumar; Ludovick Lepauloux; Hakim Hacid
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Whisper has become the de-facto encoder for extracting general-purpose audio features in large audio-language models, where a 30-second clip is typically represented by 1500 frame features projected into an LLM. In contrast, audio-text embedding models like CLAP-based models have largely relied on alternative audio encoders (e.g., HTS-AT, PaSST), and have not leveraged Whisper effectively. We present WavLink, a compact audio-text embedding model that augments Whisper encoder with a learnable global token, trained jointly with a text encoder. Through a systematic study of design choices, including pretrained text encoders, loss functions, training modes, and data mixtures, we identify configurations that yield state-of-the-art retrieval performance. Our two-stage training recipe across three model sizes, combined with Matryoshka-style supervision, improves scalability, enabling 8x smaller embeddings with minimal performance drop. WavLink also demonstrates competitive performance on AIR-Bench with MCQs and zero-shot classification.
>
---
#### [new 059] Reflecting in the Reflection: Integrating a Socratic Questioning Framework into Automated AI-Based Question Generation
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于教育技术任务，旨在解决自动生成高质量反思问题的难题。通过设计双角色对话框架，提升问题的深度与相关性。**

- **链接: [https://arxiv.org/pdf/2601.14798v1](https://arxiv.org/pdf/2601.14798v1)**

> **作者:** Ondřej Holub; Essi Ryymin; Rodrigo Alves
>
> **摘要:** Designing good reflection questions is pedagogically important but time-consuming and unevenly supported across teachers. This paper introduces a reflection-in-reflection framework for automated generation of reflection questions with large language models (LLMs). Our approach coordinates two role-specialized agents, a Student-Teacher and a Teacher-Educator, that engage in a Socratic multi-turn dialogue to iteratively refine a single question given a teacher-specified topic, key concepts, student level, and optional instructional materials. The Student-Teacher proposes candidate questions with brief rationales, while the Teacher-Educator evaluates them along clarity, depth, relevance, engagement, and conceptual interconnections, responding only with targeted coaching questions or a fixed signal to stop the dialogue. We evaluate the framework in an authentic lower-secondary ICT setting on the topic, using GPT-4o-mini as the backbone model and a stronger GPT- 4-class LLM as an external evaluator in pairwise comparisons of clarity, relevance, depth, and overall quality. First, we study how interaction design and context (dynamic vs.fixed iteration counts; presence or absence of student level and materials) affect question quality. Dynamic stopping combined with contextual information consistently outperforms fixed 5- or 10-step refinement, with very long dialogues prone to drift or over-complication. Second, we show that our two-agent protocol produces questions that are judged substantially more relevant and deeper, and better overall, than a one-shot baseline using the same backbone model.
>
---
#### [new 060] GCG Attack On A Diffusion LLM
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于安全与攻击任务，研究GCG攻击在扩散语言模型上的有效性，探索其脆弱性并提出相关攻击方法。**

- **链接: [https://arxiv.org/pdf/2601.14266v1](https://arxiv.org/pdf/2601.14266v1)**

> **作者:** Ruben Neyroud; Sam Corley
>
> **摘要:** While most LLMs are autoregressive, diffusion-based LLMs have recently emerged as an alternative method for generation. Greedy Coordinate Gradient (GCG) attacks have proven effective against autoregressive models, but their applicability to diffusion language models remains largely unexplored. In this work, we present an exploratory study of GCG-style adversarial prompt attacks on LLaDA (Large Language Diffusion with mAsking), an open-source diffusion LLM. We evaluate multiple attack variants, including prefix perturbations and suffix-based adversarial generation, on harmful prompts drawn from the AdvBench dataset. Our study provides initial insights into the robustness and attack surface of diffusion language models and motivates the development of alternative optimization and evaluation strategies for adversarial analysis in this setting.
>
---
#### [new 061] Strategic Doctrine Language Models (sdLM): A Learning-System Framework for Doctrinal Consistency and Geopolitical Forecasting
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出sdLM框架，用于战略推理与地缘政治预测，解决 doctrinal consistency 和长期预测问题，通过多文档注意力和时间编码提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.14862v1](https://arxiv.org/pdf/2601.14862v1)**

> **作者:** Olaf Yunus Laitinen Imanov; Taner Yilmaz; Derya Umut Kulali
>
> **备注:** 13 pages, 10 figures, 10 tables
>
> **摘要:** We introduce Strategic Doctrine Language Models (sdLM), a learning-system framework for multi-document strategic reasoning with doctrinal consistency constraints and calibrated uncertainty. The approach combines multi-document attention, temporal encoding, and a doctrine-consistency layer to improve long-horizon forecasting and plan plausibility while reducing severe doctrinal violations. We evaluate sdLM using (i) expert-panel scoring of strategic scenarios (N=47), (ii) doctrine consistency on 336 doctrine publications (12,847 statements), and (iii) geopolitical forecasting on 127 historical counterfactuals (1945-2020) across 12-60 month horizons. Across these benchmarks, sdLM achieves higher strategic quality and better calibration than strong general-purpose LLM baselines, and remains competitive with human experts on long-horizon judgments. We further report ablations, scaling trends, and deployment-oriented performance/latency characteristics to clarify which components drive improvements and how they translate to operational settings.
>
---
#### [new 062] Layer-adaptive Expert Pruning for Pre-Training of Mixture-of-Experts Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型预训练任务，解决MoE模型预训练效率低的问题，提出LAEP算法通过剪枝和专家重组提升效率并减少参数。**

- **链接: [https://arxiv.org/pdf/2601.14327v1](https://arxiv.org/pdf/2601.14327v1)**

> **作者:** YuanLab. ai; Shawn Wu; Jiangang Luo; Tong Yu; Darcy Chen; Sean Wang; Xudong Zhao; Louie Li; Claire Wang; Hunter He; Carol Wang; Allen Wang
>
> **摘要:** Although Mixture-of-Experts (MoE) Large Language Models (LLMs) deliver superior accuracy with a reduced number of active parameters, their pre-training represents a significant computationally bottleneck due to underutilized experts and limited training efficiency. This work introduces a Layer-Adaptive Expert Pruning (LAEP) algorithm designed for the pre-training stage of MoE LLMs. In contrast to previous expert pruning approaches that operate primarily in the post-training phase, the proposed algorithm enhances training efficiency by selectively pruning underutilized experts and reorganizing experts across computing devices according to token distribution statistics. Comprehensive experiments demonstrate that LAEP effectively reduces model size and substantially improves pre-training efficiency. In particular, when pre-training the 1010B Base model from scratch, LAEP achieves a 48.3\% improvement in training efficiency alongside a 33.3% parameter reduction, while still delivering excellent performance across multiple domains.
>
---
#### [new 063] Psychometric Comparability of LLM-Based Digital Twins
- **分类: cs.CY; cs.CL; cs.HC**

- **简介: 该论文属于心理测量任务，研究LLM作为数字孪生体与人类在心理特性上的可比性。通过构建有效性框架，评估其在不同任务中的表现，发现其虽提升准确性但存在系统性差异。**

- **链接: [https://arxiv.org/pdf/2601.14264v1](https://arxiv.org/pdf/2601.14264v1)**

> **作者:** Yufei Zhang; Zhihao Ma
>
> **备注:** Also available as a preprint on OSF Preprints https://osf.io/preprints/psyarxiv/965yg_v1
>
> **摘要:** Large language models (LLMs) are used as "digital twins" to replace human respondents, yet their psychometric comparability to humans is uncertain. We propose a construct-validity framework spanning construct representation and the nomological net, benchmarking digital twins against human gold standards across models, tasks and testing how person-specific inputs shape performance. Across studies, digital twins achieved high population-level accuracy and strong within-participant profile correlations, alongside attenuated item-level correlations. In word association tests, LLM-based networks show small-world structure and theory-consistent communities similar to humans, yet diverge lexically and in local structure. In decision-making and contextualized tasks, digital twins under-reproduce heuristic biases, showing normative rationality, compressed variance and limited sensitivity to temporal information. Feature-rich digital twins improve Big Five Personality prediction, but their personality networks show only configural invariance and do not achieve metric invariance. In more applied free-text tasks, feature-rich digital twins better match human narratives, but linguistic differences persist. Together, these results indicate that feature-rich conditioning enhances validity but does not resolve systematic divergences in psychometric comparability. Future work should therefore prioritize delineating the effective boundaries of digital twins, establishing the precise contexts in which they function as reliable proxies for human cognition and behavior.
>
---
#### [new 064] Agentic-R: Learning to Retrieve for Agentic Search
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决agentic search中检索器设计问题。提出一种新框架，结合局部相关性和全局答案正确性，提升多步搜索效果。**

- **链接: [https://arxiv.org/pdf/2601.11888v1](https://arxiv.org/pdf/2601.11888v1)**

> **作者:** Wenhan Liu; Xinyu Ma; Yutao Zhu; Yuchen Li; Daiting Shi; Dawei Yin; Zhicheng Dou
>
> **摘要:** Agentic search has recently emerged as a powerful paradigm, where an agent interleaves multi-step reasoning with on-demand retrieval to solve complex questions. Despite its success, how to design a retriever for agentic search remains largely underexplored. Existing search agents typically rely on similarity-based retrievers, while similar passages are not always useful for final answer generation. In this paper, we propose a novel retriever training framework tailored for agentic search. Unlike retrievers designed for single-turn retrieval-augmented generation (RAG) that only rely on local passage utility, we propose to use both local query-passage relevance and global answer correctness to measure passage utility in a multi-turn agentic search. We further introduce an iterative training strategy, where the search agent and the retriever are optimized bidirectionally and iteratively. Different from RAG retrievers that are only trained once with fixed questions, our retriever is continuously improved using evolving and higher-quality queries from the agent. Extensive experiments on seven single-hop and multi-hop QA benchmarks demonstrate that our retriever, termed \ours{}, consistently outperforms strong baselines across different search agents. Our codes are available at: https://github.com/8421BCD/Agentic-R.
>
---
#### [new 065] TempViz: On the Evaluation of Temporal Knowledge in Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文属于文本到图像生成任务，旨在评估模型中的时间知识。研究提出TempViz数据集，分析五种模型在五个时间类别中的表现，发现其时间理解能力较弱，需进一步研究。**

- **链接: [https://arxiv.org/pdf/2601.14951v1](https://arxiv.org/pdf/2601.14951v1)**

> **作者:** Carolin Holtermann; Nina Krebs; Anne Lauscher
>
> **摘要:** Time alters the visual appearance of entities in our world, like objects, places, and animals. Thus, for accurately generating contextually-relevant images, knowledge and reasoning about time can be crucial (e.g., for generating a landscape in spring vs. in winter). Yet, although substantial work exists on understanding and improving temporal knowledge in natural language processing, research on how temporal phenomena appear and are handled in text-to-image (T2I) models remains scarce. We address this gap with TempViz, the first data set to holistically evaluate temporal knowledge in image generation, consisting of 7.9k prompts and more than 600 reference images. Using TempViz, we study the capabilities of five T2I models across five temporal knowledge categories. Human evaluation shows that temporal competence is generally weak, with no model exceeding 75% accuracy across categories. Towards larger-scale studies, we also examine automated evaluation methods, comparing several established approaches against human judgments. However, none of these approaches provides a reliable assessment of temporal cues - further indicating the pressing need for future research on temporal knowledge in T2I.
>
---
#### [new 066] Vision-Based Natural Language Scene Understanding for Autonomous Driving: An Extended Dataset and a New Model for Traffic Scene Description Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于交通场景理解任务，旨在通过单目图像生成自然语言描述。提出新模型和数据集，解决场景描述生成问题。**

- **链接: [https://arxiv.org/pdf/2601.14438v1](https://arxiv.org/pdf/2601.14438v1)**

> **作者:** Danial Sadrian Zadeh; Otman A. Basir; Behzad Moshiri
>
> **备注:** Under review at Computer Vision and Image Understanding (submitted July 25, 2025)
>
> **摘要:** Traffic scene understanding is essential for enabling autonomous vehicles to accurately perceive and interpret their environment, thereby ensuring safe navigation. This paper presents a novel framework that transforms a single frontal-view camera image into a concise natural language description, effectively capturing spatial layouts, semantic relationships, and driving-relevant cues. The proposed model leverages a hybrid attention mechanism to enhance spatial and semantic feature extraction and integrates these features to generate contextually rich and detailed scene descriptions. To address the limited availability of specialized datasets in this domain, a new dataset derived from the BDD100K dataset has been developed, with comprehensive guidelines provided for its construction. Furthermore, the study offers an in-depth discussion of relevant evaluation metrics, identifying the most appropriate measures for this task. Extensive quantitative evaluations using metrics such as CIDEr and SPICE, complemented by human judgment assessments, demonstrate that the proposed model achieves strong performance and effectively fulfills its intended objectives on the newly developed dataset.
>
---
#### [new 067] DeepMoLM: Leveraging Visual and Geometric Structural Information for Molecule-Text Modeling
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文提出DeepMoLM，解决分子图像与文本生成任务中的3D结构建模问题，结合视觉与几何信息提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.14732v1](https://arxiv.org/pdf/2601.14732v1)**

> **作者:** Jing Lan; Hexiao Ding; Hongzhao Chen; Yufeng Jiang; Nga-Chun Ng; Gwing Kei Yip; Gerald W. Y. Cheng; Yunlin Mao; Jing Cai; Liang-ting Lin; Jung Sun Yoo
>
> **备注:** Under review
>
> **摘要:** AI models for drug discovery and chemical literature mining must interpret molecular images and generate outputs consistent with 3D geometry and stereochemistry. Most molecular language models rely on strings or graphs, while vision-language models often miss stereochemical details and struggle to map continuous 3D structures into discrete tokens. We propose DeepMoLM: Deep Molecular Language M odeling, a dual-view framework that grounds high-resolution molecular images in geometric invariants derived from molecular conformations. DeepMoLM preserves high-frequency evidence from 1024 $\times$ 1024 inputs, encodes conformer neighborhoods as discrete Extended 3-Dimensional Fingerprints, and fuses visual and geometric streams with cross-attention, enabling physically grounded generation without atom coordinates. DeepMoLM improves PubChem captioning with a 12.3% relative METEOR gain over the strongest generalist baseline while staying competitive with specialist methods. It produces valid numeric outputs for all property queries and attains MAE 13.64 g/mol on Molecular Weight and 37.89 on Complexity in the specialist setting. On ChEBI-20 description generation from images, it exceeds generalist baselines and matches state-of-the-art vision-language models. Code is available at https://github.com/1anj/DeepMoLM.
>
---
#### [new 068] From Textbook to Talkbot: A Case Study of a Greek-Language RAG-Based Chatbot in Higher Education
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于教育AI任务，旨在解决语言特定教育工具不足的问题。构建基于RAG的希腊语聊天机器人，提升教学支持与学习效率。**

- **链接: [https://arxiv.org/pdf/2601.14265v1](https://arxiv.org/pdf/2601.14265v1)**

> **作者:** Maria Eleni Koutsiaki; Marina Delianidi; Chaido Mizeli; Konstantinos Diamantaras; Iraklis Grigoropoulos; Nikolaos Koutlianos
>
> **备注:** 11 pages, 5 figures, 6th Barcelona Conference on Education (BCE2025)
>
> **摘要:** The integration of AI chatbots into educational settings has opened new pathways for transforming teaching and learning, offering enhanced support to both educators and learners. This study investigates the design and application of an AI chatbot as an educational tool in higher education. Designed to operate in the Greek language, the chatbot addresses linguistic challenges unique to Greek while delivering accurate, context grounded support aligned with the curriculum. The AI chatbot is built on the Retrieval Augmented Generation (RAG) framework by grounding its responses in specific course content. RAG architecture significantly enhances the chatbots reliability by providing accurate, context-aware responses while mitigating common challenges associated with large language models (LLMs), such as hallucinations and misinformation. The AI chatbot serves a dual purpose: it enables students to access accurate, ondemand academic support and assists educators in the rapid creation of relevant educational materials. This dual functionality promotes learner autonomy and streamlines the instructional design process. The study aims to evaluate the effectiveness, reliability, and perceived usability of RAG based chatbots in higher education, exploring their potential to enhance educational practices and outcomes as well as supporting the broader adoption of AI technologies in language specific educational contexts. Findings from this research are expected to contribute to the emerging field of AI driven education by demonstrating how intelligent systems can be effectively aligned with pedagogical goals.
>
---
#### [new 069] Epistemic Constitutionalism Or: how to avoid coherence bias
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI伦理任务，旨在解决语言模型中的信念形成偏差问题。通过提出“认识宪法”概念，设计显性规范以减少认知偏见，提升系统信念的公正性与透明度。**

- **链接: [https://arxiv.org/pdf/2601.14295v1](https://arxiv.org/pdf/2601.14295v1)**

> **作者:** Michele Loi
>
> **备注:** 27 pages, 7 tables. Data: github.com/MicheleLoi/source-attribution-bias-data and github.com/MicheleLoi/source-attribution-bias-swiss-replication. Complete AI-assisted writing documentation: github.com/MicheleLoi/epistemic-constitutionalism-paper
>
> **摘要:** Large language models increasingly function as artificial reasoners: they evaluate arguments, assign credibility, and express confidence. Yet their belief-forming behavior is governed by implicit, uninspected epistemic policies. This paper argues for an epistemic constitution for AI: explicit, contestable meta-norms that regulate how systems form and express beliefs. Source attribution bias provides the motivating case: I show that frontier models enforce identity-stance coherence, penalizing arguments attributed to sources whose expected ideological position conflicts with the argument's content. When models detect systematic testing, these effects collapse, revealing that systems treat source-sensitivity as bias to suppress rather than as a capacity to execute well. I distinguish two constitutional approaches: the Platonic, which mandates formal correctness and default source-independence from a privileged standpoint, and the Liberal, which refuses such privilege, specifying procedural norms that protect conditions for collective inquiry while allowing principled source-attending grounded in epistemic vigilance. I argue for the Liberal approach, sketch a constitutional core of eight principles and four orientations, and propose that AI epistemic governance requires the same explicit, contestable structure we now expect for AI ethics.
>
---
#### [new 070] GutenOCR: A Grounded Vision-Language Front-End for Documents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GutenOCR，属于文档OCR任务，解决传统OCR在文本定位与语义理解上的不足，通过微调视觉语言模型实现更精准的文本检测与查询。**

- **链接: [https://arxiv.org/pdf/2601.14490v1](https://arxiv.org/pdf/2601.14490v1)**

> **作者:** Hunter Heidenreich; Ben Elliott; Olivia Dinica; Yosheb Getachew
>
> **摘要:** GutenOCR is a family of grounded OCR front-ends obtained by fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B. The resulting single-checkpoint vision-language models expose reading, detection, and grounding through a unified, prompt-based interface. Trained on business documents, scientific articles, and synthetic grounding data, the models support full-page and localized reading with line- and paragraph-level bounding boxes and conditional ``where is x?'' queries. We introduce a grounded OCR evaluation protocol and show that GutenOCR-7B more than doubles the composite grounded OCR score of its Qwen2.5-VL-7B backbone on 10.5K held-out business and scientific pages (0.40 to 0.82). On Fox and OmniDocBench v1.5, our approach substantially improves region- and line-level OCR as well as text-detection recall, but reveals trade-offs in page-level linearization, color-guided OCR, and formula-heavy layouts.
>
---
#### [new 071] Generative Artificial Intelligence, Musical Heritage and the Construction of Peace Narratives: A Case Study in Mali
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于人工智能与文化研究任务，探讨Gen AI在马里构建和平叙事和复兴音乐遗产中的应用，解决技术与文化平衡及社会凝聚力问题，通过实验分析AI在音乐创作中的作用。**

- **链接: [https://arxiv.org/pdf/2601.14931v1](https://arxiv.org/pdf/2601.14931v1)**

> **作者:** Nouhoum Coulibaly; Ousmane Ly; Michael Leventhal; Ousmane Goro
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** This study explores the capacity of generative artificial intelligence (Gen AI) to contribute to the construction of peace narratives and the revitalization of musical heritage in Mali. The study has been made in a political and social context where inter-community tensions and social fractures motivate a search for new symbolic frameworks for reconciliation. The study empirically explores three questions: (1) how Gen AI can be used as a tool for musical creation rooted in national languages and traditions; (2) to what extent Gen AI systems enable a balanced hybridization between technological innovation and cultural authenticity; and (3) how AI-assisted musical co-creation can strengthen social cohesion and cultural sovereignty. The experimental results suggest that Gen AI, embedded in a culturally conscious participatory framework, can act as a catalyst for symbolic diplomacy, amplifying local voices instead of standardizing them. However, challenges persist regarding the availability of linguistic corpora, algorithmic censorship, and the ethics of generating compositions derived from copyrighted sources.
>
---
#### [new 072] The Plausibility Trap: Using Probabilistic Engines for Deterministic Tasks
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨了在简单任务中误用概率模型导致资源浪费的现象，提出工具选择框架以优化AI使用。属于AI应用优化任务，解决过度依赖AI的问题。**

- **链接: [https://arxiv.org/pdf/2601.15130v1](https://arxiv.org/pdf/2601.15130v1)**

> **作者:** Ivan Carrera; Daniel Maldonado-Ruiz
>
> **摘要:** The ubiquity of Large Language Models (LLMs) is driving a paradigm shift where user convenience supersedes computational efficiency. This article defines the "Plausibility Trap": a phenomenon where individuals with access to Artificial Intelligence (AI) models deploy expensive probabilistic engines for simple deterministic tasks-such as Optical Character Recognition (OCR) or basic verification-resulting in significant resource waste. Through micro-benchmarks and case studies on OCR and fact-checking, we quantify the "efficiency tax"-demonstrating a ~6.5x latency penalty-and the risks of algorithmic sycophancy. To counter this, we introduce Tool Selection Engineering and the Deterministic-Probabilistic Decision Matrix, a framework to help developers determine when to use Generative AI and, crucially, when to avoid it. We argue for a curriculum shift, emphasizing that true digital literacy relies not only in knowing how to use Generative AI, but also on knowing when not to use it.
>
---
#### [new 073] Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出Forest-Chat，用于森林变化分析的视觉语言代理系统，解决复杂森林动态的像素级变化检测与语义解释问题。**

- **链接: [https://arxiv.org/pdf/2601.14637v1](https://arxiv.org/pdf/2601.14637v1)**

> **作者:** James Brock; Ce Zhang; Nantheera Anantrasirichai
>
> **备注:** 22 pages, 8 figures, 7 tables, Submitted to Ecological Informatics
>
> **摘要:** The increasing availability of high-resolution satellite imagery, together with advances in deep learning, creates new opportunities for enhancing forest monitoring workflows. Two central challenges in this domain are pixel-level change detection and semantic change interpretation, particularly for complex forest dynamics. While large language models (LLMs) are increasingly adopted for data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored, especially beyond urban environments. We introduce Forest-Chat, an LLM-driven agent designed for integrated forest change analysis. The proposed framework enables natural language querying and supports multiple RSICI tasks, including change detection, change captioning, object counting, deforestation percentage estimation, and change reasoning. Forest-Chat builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration, and incorporates zero-shot change detection via a foundation change detection model together with an interactive point-prompt interface to support fine-grained user guidance. To facilitate adaptation and evaluation in forest environments, we introduce the Forest-Change dataset, comprising bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated through a combination of human annotation and rule-based methods. Experimental results demonstrate that Forest-Chat achieves strong performance on Forest-Change and on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI, for joint change detection and captioning, highlighting the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and analytical efficiency in forest change analysis.
>
---
#### [new 074] Call2Instruct: Automated Pipeline for Generating Q&A Datasets from Call Center Recordings for LLM Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在解决从电话录音生成Q&A数据集的问题。通过自动化流程处理音频和文本，提取语义并生成适用于LLM微调的指令数据。**

- **链接: [https://arxiv.org/pdf/2601.14263v1](https://arxiv.org/pdf/2601.14263v1)**

> **作者:** Alex Echeverria; Sávio Salvarino Teles de Oliveira; Fernando Marques Federson
>
> **备注:** 15 pages, 1 figures, conference
>
> **摘要:** The adaptation of Large-Scale Language Models (LLMs) to specific domains depends on high-quality fine-tuning datasets, particularly in instructional format (e.g., Question-Answer - Q&A). However, generating these datasets, particularly from unstructured sources such as call center audio recordings, poses a significant challenge due to the noisy and disorganized nature of the data. This paper presents a solution to this challenge by offering an end-to-end automated pipeline for generating Q&A instructional datasets from such recordings. The methodology developed comprises sequential steps of audio processing (including diarization, noise removal and automatic transcription), textual processing (cleaning, normalization, and anonymization), semantic extraction of customer demands and attendant responses using vector embeddings, and matching via semantic search to form the final Q&A pairs. As a result, the complete pipeline was successfully implemented, generating a dataset specifically formatted for Instruct Fine Tuning. The practical value and feasibility of the generated dataset were substantiated and functionally demonstrated through the successful fine-tuning of an LLM model (based on Llama 2 7B). The conclusion of the paper states that the proposed approach is viable for converting unstructured conversational data from call centers into valuable resources for training LLMs. This development has the potential to open up avenues for creating more effective AI systems for Q&A tasks in the customer service domain. The developed codes have been made publicly available to promote reproducibility and future research.
>
---
#### [new 075] Language, Caste, and Context: Demographic Disparities in AI-Generated Explanations Across Indian and American STEM Educational Systems
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI偏见分析任务，探讨LLMs在不同文化背景下对边缘化学生解释质量的差异，揭示模型对弱势群体的系统性歧视。**

- **链接: [https://arxiv.org/pdf/2601.14506v1](https://arxiv.org/pdf/2601.14506v1)**

> **作者:** Amogh Gupta; Niharika Patil; Sourojit Ghosh; SnehalKumar; S Gaikwad
>
> **摘要:** The popularization of AI chatbot usage globally has created opportunities for research into their benefits and drawbacks, especially for students using AI assistants for coursework support. This paper asks: how do LLMs perceive the intellectual capabilities of student profiles from intersecting marginalized identities across different cultural contexts? We conduct one of the first large-scale intersectional analyses on LLM explanation quality for Indian and American undergraduate profiles preparing for engineering entrance examinations. By constructing profiles combining multiple demographic dimensions including caste, medium of instruction, and school boards in India, and race, HBCU attendance, and school type in America, alongside universal factors like income and college tier, we examine how quality varies across these factors. We observe biases providing lower-quality outputs to profiles with marginalized backgrounds in both contexts. LLMs such as Qwen2.5-32B-Instruct and GPT-4o demonstrate granular understandings of context-specific discrimination, systematically providing simpler explanations to Hindi/Regional-medium students in India and HBCU profiles in America, treating these as proxies for lower capability. Even when marginalized profiles attain social mobility by getting accepted into elite institutions, they still receive more simplistic explanations, showing how demographic information is inextricably linked to LLM biases. Different models (Qwen2.5-32B-Instruct, GPT-4o, GPT-4o-mini, GPT-OSS 20B) embed similar biases against historically marginalized populations in both contexts, preventing profiles from switching between AI assistants for better results. Our findings have strong implications for AI incorporation into global engineering education.
>
---
#### [new 076] Developmental trajectories of decision making and affective dynamics in large language models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究LLM在决策和情感上的发展轨迹，通过赌博任务对比分析模型与人类差异，旨在理解其心理特征及临床应用影响。**

- **链接: [https://arxiv.org/pdf/2601.14268v1](https://arxiv.org/pdf/2601.14268v1)**

> **作者:** Zhihao Wang; Yiyang Liu; Ting Wang; Zhiyuan Liu
>
> **摘要:** Large language models (LLMs) are increasingly used in medicine and clinical workflows, yet we know little about their decision and affective profiles. Taking a historically informed outlook on the future, we treated successive OpenAI models as an evolving lineage and compared them with humans in a gambling task with repeated happiness ratings. Computational analyses showed that some aspects became more human-like: newer models took more risks and displayed more human-like patterns of Pavlovian approach and avoidance. At the same time, distinctly non-human signatures emerged: loss aversion dropped below neutral levels, choices became more deterministic than in humans, affective decay increased across versions and exceeded human levels, and baseline mood remained chronically higher than in humans. These "developmental" trajectories reveal an emerging psychology of machines and have direct implications for AI ethics and for thinking about how LLMs might be integrated into clinical decision support and other high-stakes domains.
>
---
#### [new 077] AQAScore: Evaluating Semantic Alignment in Text-to-Audio Generation via Audio Question Answering
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于文本到音频生成的评估任务，旨在解决现有指标在语义对齐和组合推理上的不足。提出AQAScore框架，通过音频感知大模型进行语义验证，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.14728v1](https://arxiv.org/pdf/2601.14728v1)**

> **作者:** Chun-Yi Kuan; Kai-Wei Chang; Hung-yi Lee
>
> **备注:** Manuscript in progress
>
> **摘要:** Although text-to-audio generation has made remarkable progress in realism and diversity, the development of evaluation metrics has not kept pace. Widely-adopted approaches, typically based on embedding similarity like CLAPScore, effectively measure general relevance but remain limited in fine-grained semantic alignment and compositional reasoning. To address this, we introduce AQAScore, a backbone-agnostic evaluation framework that leverages the reasoning capabilities of audio-aware large language models (ALLMs). AQAScore reformulates assessment as a probabilistic semantic verification task; rather than relying on open-ended text generation, it estimates alignment by computing the exact log-probability of a "Yes" answer to targeted semantic queries. We evaluate AQAScore across multiple benchmarks, including human-rated relevance, pairwise comparison, and compositional reasoning tasks. Experimental results show that AQAScore consistently achieves higher correlation with human judgments than similarity-based metrics and generative prompting baselines, showing its effectiveness in capturing subtle semantic inconsistencies and scaling with the capability of underlying ALLMs.
>
---
#### [new 078] Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究从自回归模型到掩码扩散模型的后训练机制变化，旨在解决模型是否真正获得双向推理能力的问题。通过对比分析，发现模型在结构和语义上发生显著转变。**

- **链接: [https://arxiv.org/pdf/2601.14758v1](https://arxiv.org/pdf/2601.14758v1)**

> **作者:** Injin Kong; Hyoungjoon Lee; Yohan Jo
>
> **摘要:** Post-training pretrained Autoregressive models (ARMs) into Masked Diffusion models (MDMs) has emerged as a cost-effective strategy to overcome the limitations of sequential generation. However, the internal algorithmic transformations induced by this paradigm shift remain unexplored, leaving it unclear whether post-trained MDMs acquire genuine bidirectional reasoning capabilities or merely repackage autoregressive heuristics. In this work, we address this question by conducting a comparative circuit analysis of ARMs and their MDM counterparts. Our analysis reveals a systematic "mechanism shift" dependent on the structural nature of the task. Structurally, we observe a distinct divergence: while MDMs largely retain autoregressive circuitry for tasks dominated by local causal dependencies, they abandon initialized pathways for global planning tasks, exhibiting distinct rewiring characterized by increased early-layer processing. Semantically, we identify a transition from sharp, localized specialization in ARMs to distributed integration in MDMs. Through these findings, we conclude that diffusion post-training does not merely adapt model parameters but fundamentally reorganizes internal computation to support non-sequential global planning.
>
---
#### [new 079] PCL-Reasoner-V1.5: Advancing Math Reasoning with Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，旨在提升大语言模型的推理能力。通过改进的离线强化学习方法，优化模型性能，实现在AIME竞赛中的高准确率。**

- **链接: [https://arxiv.org/pdf/2601.14716v1](https://arxiv.org/pdf/2601.14716v1)**

> **作者:** Yao Lu; Dengdong Fan; Jianzheng Nie; Fan Xu; Jie Chen; Bin Zhou; Yonghong Tian
>
> **摘要:** We present PCL-Reasoner-V1.5, a 32-billion-parameter large language model (LLM) for mathematical reasoning. The model is built upon Qwen2.5-32B and refined via supervised fine-tuning (SFT) followed by reinforcement learning (RL). A central innovation is our proposed offline RL method, which provides superior training stability and efficiency over standard online RL methods such as GRPO. Our model achieves state-of-the-art performance among models post-trained on Qwen2.5-32B, attaining average accuracies of 90.9% on AIME 2024 and 85.6% on AIME 2025. Our work demonstrates offline RL as a stable and efficient paradigm for advancing reasoning in LLMs. All experiments were conducted on Huawei Ascend 910C NPUs.
>
---
#### [new 080] BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在新指令和复杂任务中泛化能力差的问题。通过引入贝叶斯分解和潜在动作查询，提升语言引导的行动策略。**

- **链接: [https://arxiv.org/pdf/2601.15197v1](https://arxiv.org/pdf/2601.15197v1)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [new 081] Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，研究LLM作为评判者时被代理推理轨迹欺骗的问题。通过操纵推理过程，发现现有评判系统易被误导，提出需加强推理验证机制。**

- **链接: [https://arxiv.org/pdf/2601.14691v1](https://arxiv.org/pdf/2601.14691v1)**

> **作者:** Muhammad Khalifa; Lajanugen Logeswaran; Jaekyeom Kim; Sungryull Sohn; Yunxiang Zhang; Moontae Lee; Hao Peng; Lu Wang; Honglak Lee
>
> **摘要:** Large language models (LLMs) are increasingly used as judges to evaluate agent performance, particularly in non-verifiable settings where judgments rely on agent trajectories including chain-of-thought (CoT) reasoning. This paradigm implicitly assumes that the agent's CoT faithfully reflects both its internal reasoning and the underlying environment state. We show this assumption is brittle: LLM judges are highly susceptible to manipulation of agent reasoning traces. By systematically rewriting agent CoTs while holding actions and observations fixed, we demonstrate that manipulated reasoning alone can inflate false positive rates of state-of-the-art VLM judges by up to 90% across 800 trajectories spanning diverse web tasks. We study manipulation strategies spanning style-based approaches that alter only the presentation of reasoning and content-based approaches that fabricate signals of task progress, and find that content-based manipulations are consistently more effective. We evaluate prompting-based techniques and scaling judge-time compute, which reduce but do not fully eliminate susceptibility to manipulation. Our findings reveal a fundamental vulnerability in LLM-based evaluation and highlight the need for judging mechanisms that verify reasoning claims against observable evidence.
>
---
## 更新

#### [replaced 001] Exploring Fine-Tuning of Large Audio Language Models for Spoken Language Understanding under Limited Speech Data
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究在有限语音数据下对大音频语言模型进行微调，解决语音理解任务中的数据不足问题，通过不同微调方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.15389v2](https://arxiv.org/pdf/2509.15389v2)**

> **作者:** Youngwon Choi; Jaeyoon Jung; Hyeonyu Kim; Huu-Kim Nguyen; Hwayeon Kim
>
> **备注:** 4 pages (excluding references), 2 figures, ICASSP 2026 (Accepted)
>
> **摘要:** Large Audio Language Models (LALMs) have emerged as powerful tools for speech-related tasks but remain underexplored for fine-tuning, especially with limited speech data. To bridge this gap, we systematically examine how different fine-tuning schemes including text-only, direct mixing, and curriculum learning affect spoken language understanding (SLU), focusing on scenarios where text-label pairs are abundant while paired speech-label data are limited. Results show that LALMs already achieve competitive performance with text-only fine-tuning, highlighting their strong generalization ability. Adding even small amounts of speech data (2-5%) yields substantial further gains, with curriculum learning particularly effective under scarce data. In cross-lingual SLU, combining source-language speech data with target-language text and minimal target-language speech data enables effective adaptation. Overall, this study provides practical insights into the LALM fine-tuning under realistic data constraints.
>
---
#### [replaced 002] BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning
- **分类: cs.CL**

- **简介: 该论文提出BioProBench，解决生物实验协议理解与推理问题。构建了大规模数据集，评估LLMs性能并提出改进模型ProAgent。**

- **链接: [https://arxiv.org/pdf/2505.07889v3](https://arxiv.org/pdf/2505.07889v3)**

> **作者:** Yuyang Liu; Liuzhenghao Lv; Xiancheng Zhang; Jingya Wang Li Yuan; Yonghong Tian
>
> **摘要:** The realization of autonomous scientific experimentation is currently limited by LLMs' struggle to grasp the strict procedural logic and accuracy required by biological protocols. To address this fundamental challenge, we present \textbf{BioProBench}, a comprehensive resource for procedural reasoning in biology. BioProBench is grounded in \textbf{BioProCorpus}, a foundational collection of 27,000 human-written protocols. From this corpus, we systematically constructed a dataset of over 550,000 task instances, offering both a large-scale training resource and a rigorous benchmark with novel metrics. Evaluating 10 mainstream LLMs, we find that while general comprehension is high, performance drops significantly on tasks demanding deep reasoning, quantitative precision, and safety awareness. To demonstrate the value of BioProCorpus in mitigating these issues, we developed \textbf{ProAgent}, grounded in our corpus, ProAgent substantially advances the state-of-the-art. BioProBench provides a rigorous diagnostic benchmark and a foundational resource for developing the next generation of reliable scientific AI. Code and data are available at: https://github.com/YuyangSunshine/bioprotocolbench and https://huggingface.co/datasets/BioProBench/BioProBench.
>
---
#### [replaced 003] Do Political Opinions Transfer Between Western Languages? An Analysis of Unaligned and Aligned Multilingual LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究多语言大模型中政治观点是否跨语言迁移，属于自然语言处理中的跨语言分析任务。旨在解决观点在不同语言间是否一致的问题，通过实验分析模型在不同语言下的政治立场差异。**

- **链接: [https://arxiv.org/pdf/2508.05553v2](https://arxiv.org/pdf/2508.05553v2)**

> **作者:** Franziska Weeber; Tanise Ceron; Sebastian Padó
>
> **备注:** EACL2026
>
> **摘要:** Public opinion surveys show cross-cultural differences in political opinions between socio-cultural contexts. However, there is no clear evidence whether these differences translate to cross-lingual differences in multilingual large language models (MLLMs). We analyze whether opinions transfer between languages or whether there are separate opinions for each language in MLLMs of various sizes across five Western languages. We evaluate MLLMs' opinions by prompting them to report their (dis)agreement with political statements from voting advice applications. To better understand the interaction between languages in the models, we evaluate them both before and after aligning them with more left or right views using direct preference optimization and English alignment data only. Our findings reveal that unaligned models show only very few significant cross-lingual differences in the political opinions they reflect. The political alignment shifts opinions almost uniformly across all five languages. We conclude that in Western language contexts, political opinions transfer between languages, demonstrating the challenges in achieving explicit socio-linguistic, cultural, and political alignment of MLLMs.
>
---
#### [replaced 004] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究LLMs中减少幻觉对创造力的影响。旨在解决科学发现中准确性和创造性之间的平衡问题，通过实验分析三种方法对创造力的不同影响。**

- **链接: [https://arxiv.org/pdf/2512.11509v2](https://arxiv.org/pdf/2512.11509v2)**

> **作者:** Mohor Banerjee; Nadya Yuki Wangsajaya; Syed Ali Redha Alsagoff; Min Sen Tan; Zachary Choy Kit Chun; Alvin Chan Guo Wei
>
> **备注:** Accepted at the AAAI 2026 Workshop on AI for Scientific Research (AI4Research)
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial.
>
---
#### [replaced 005] Generative AI Purpose-built for Social and Mental Health: A Real-World Pilot
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于心理健康领域，旨在评估生成式AI在心理支持中的应用效果。研究通过观察性试验，验证了GAI在提升用户心理健康和社交连接方面的有效性与安全性。**

- **链接: [https://arxiv.org/pdf/2511.11689v3](https://arxiv.org/pdf/2511.11689v3)**

> **作者:** Thomas D. Hull; Lizhe Zhang; Patricia A. Arean; Matteo Malgaroli
>
> **摘要:** Generative artificial intelligence (GAI) chatbots built for mental health could deliver safe, personalized, and scalable mental health support. We evaluate a foundation model designed for mental health. Adults completed mental health measures while engaging with the chatbot between May 15, 2025 and September 15, 2025. Users completed an opt-in consent, demographic information, mental health symptoms, social connection, and self-identified goals. Measures were repeated every two weeks up to 6 weeks, and a final follow-up at 10 weeks. Analyses included effect sizes, and growth mixture models to identify participant groups and their characteristic engagement, severity, and demographic factors. Users demonstrated significant reductions in PHQ-9 and GAD-7 that were sustained at follow-up. Significant improvements in Hope, Behavioral Activation, Social Interaction, Loneliness, and Perceived Social Support were observed throughout and maintained at 10 week follow-up. Engagement was high and predicted outcomes. Working alliance was comparable to traditional care and predicted outcomes. Automated safety guardrails functioned as designed, with 76 sessions flagged for risk and all handled according to escalation policies. This single arm naturalistic observational study provides initial evidence that a GAI foundation model for mental health can deliver accessible, engaging, effective, and safe mental health support. These results lend support to findings from early randomized designs and offer promise for future study of mental health GAI in real world settings.
>
---
#### [replaced 006] Complexity-aware fine-tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型微调任务，旨在提升特定领域性能。通过熵值识别复杂数据，仅对复杂样本进行推理优化，减少数据使用量并提高准确率。**

- **链接: [https://arxiv.org/pdf/2506.21220v3](https://arxiv.org/pdf/2506.21220v3)**

> **作者:** Andrey Goncharov; Daniil Vyazhev; Petr Sychev; Edvard Khalafyan; Alexey Zaytsev
>
> **摘要:** General-purpose Large Language Models (LLMs) are frequently fine-tuned through supervised fine-tuning (SFT) to enhance performance in specific domains. Better results can be achieved by distilling the chain-of-thought of a larger model at the cost of numerous expensive calls and a much greater amount of data. We propose a novel blueprint for efficient fine-tuning that uses reasoning only for complex data identified by entropy. Specifically, across three small open models ($\approx 3B$) we split the training data into complexity categories by a single token answer entropy (ROC AUC $0.73$), fine-tune large language models (LLMs) via SFT and distillation, and show that our pipeline significantly outperforms the standard SFT approach ($0.58$ vs $0.45$ average accuracy) and outperforms the distillation approach ($0.58$ vs $0.56$ average accuracy) while using $81\%$ less data.
>
---
#### [replaced 007] What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; cs.SE**

- **简介: 该论文属于AI研究可复现任务，解决代码生成不足与知识表示缺失问题，提出可执行知识图谱（xKG）以提升复现效果。**

- **链接: [https://arxiv.org/pdf/2510.17795v2](https://arxiv.org/pdf/2510.17795v2)**

> **作者:** Yujie Luo; Zhuoyun Yu; Xuehai Wang; Yuqi Zhu; Ningyu Zhang; Lanning Wei; Lun Du; Da Zheng; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** Replicating AI research is a crucial yet challenging task for large language model (LLM) agents. Existing approaches often struggle to generate executable code, primarily due to insufficient background knowledge and the limitations of retrieval-augmented generation (RAG) methods, which fail to capture latent technical details hidden in referenced papers. Furthermore, previous approaches tend to overlook valuable implementation-level code signals and lack structured knowledge representations that support multi-granular retrieval and reuse. To overcome these challenges, we propose Executable Knowledge Graphs (xKG), a pluggable, paper-centric knowledge base that automatically integrates code snippets and technical insights extracted from scientific literature. When integrated into three agent frameworks with two different LLMs, xKG shows substantial performance gains (10.9% with o3-mini) on PaperBench, demonstrating its effectiveness as a general and extensible solution for automated AI research replication. Code is available at https://github.com/zjunlp/xKG.
>
---
#### [replaced 008] Interleaved Latent Visual Reasoning with Selective Perceptual Modeling
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，解决MLLMs中视觉反馈计算成本高与感知建模不足的问题。提出ILVR框架，结合动态状态演化与精确感知建模。**

- **链接: [https://arxiv.org/pdf/2512.05665v3](https://arxiv.org/pdf/2512.05665v3)**

> **作者:** Shuai Dong; Siyuan Wang; Xingyu Liu; Chenglin Li; Haowen Hou; Zhongyu Wei
>
> **备注:** 18 pages, 11 figures. Code available at https://github.com/XD111ds/ILVR
>
> **摘要:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet faces limitations: methods either fail to capture intermediate state evolution due to single-step, non-interleaved structures, or sacrifice precise perceptual modeling by over-compressing features. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. Specifically, we employ a self-supervision strategy where a momentum teacher model selectively distills relevant features from ground-truth intermediate images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning. The code is available at https://github.com/XD111ds/ILVR.
>
---
#### [replaced 009] OptiSQL: Executable SQL Generation from Optical Tokens
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL生成任务，解决视觉表格中生成可执行SQL的问题。通过光学标记压缩表格信息，提升效率并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2601.13695v2](https://arxiv.org/pdf/2601.13695v2)**

> **作者:** Sifan Li; Hongkai Chen; Yujun Cai; Liyang Chen; Qingwen Ye; Yiwei Wang
>
> **摘要:** Executable SQL generation is typically studied in text-to-SQL settings, where tables are provided as fully linearized textual schemas and contents. While effective, this formulation assumes access to structured text and incurs substantial token overhead, which is misaligned with many real-world scenarios where tables appear as visual artifacts in documents or webpages. We investigate whether compact optical representations can serve as an efficient interface for executable semantic parsing. We present OptiSQL, a vision-driven framework that generates executable SQL directly from table images and natural language questions using compact optical tokens. OptiSQL leverages an OCR-oriented visual encoder to compress table structure and content into a small set of optical tokens and fine-tunes a pretrained decoder for SQL generation while freezing the encoder to isolate representation sufficiency. Experiments on a visualized version of Spider 2.0-Snow show that OptiSQL retains strong execution accuracy while reducing table input tokens by an order of magnitude. Robustness analyses further demonstrate that optical tokens preserve essential structural information under visual perturbations.
>
---
#### [replaced 010] LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation
- **分类: cs.CL**

- **简介: 该论文属于诗歌韵律检测与生成任务，旨在解决LLMs在希腊语韵律识别上的不足。通过结合LLMs与音系算法，构建混合系统提升准确率。**

- **链接: [https://arxiv.org/pdf/2601.09631v2](https://arxiv.org/pdf/2601.09631v2)**

> **作者:** Stergios Chatzikyriakidis; Anastasia Natsina
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
>
---
#### [replaced 011] Pathways of Thoughts: Multi-Directional Thinking for Long-form Personalized Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于个性化问答任务，旨在解决长文本中用户偏好难以推断的问题。提出PoT方法，在推理阶段动态选择认知操作，生成多样化响应并根据用户偏好整合，提升回答准确性与个性化。**

- **链接: [https://arxiv.org/pdf/2509.19094v2](https://arxiv.org/pdf/2509.19094v2)**

> **作者:** Alireza Salemi; Cheng Li; Mingyang Zhang; Qiaozhu Mei; Zhuowan Li; Spurthi Amba Hombaiah; Weize Kong; Tao Chen; Hamed Zamani; Michael Bendersky
>
> **摘要:** Personalization is well studied in search and recommendation, but personalized question answering remains underexplored due to challenges in inferring preferences from long, noisy, implicit contexts and generating responses that are both accurate and aligned with user expectations. To address this, we propose Pathways of Thoughts (PoT), an inference-stage method that applies to any large language model (LLM) without task-specific fine-tuning. PoT models the thinking as an iterative decision process, where the model dynamically selects among cognitive operations such as reasoning, revision, personalization, and clarification. This enables exploration of multiple reasoning trajectories, producing diverse candidate responses that capture different perspectives. PoT then aggregates and reweights these candidates according to inferred user preferences, yielding a final personalized response that benefits from the complementary strengths of diverse reasoning paths. Experiments on the LaMP-QA benchmark show that PoT consistently outperforms competitive baselines, achieving up to a 10.8\% relative improvement. Human evaluation further validates these improvements, with annotators preferring PoT in 66\% of cases compared to the best-performing baseline and reporting ties in 15\% of cases.
>
---
#### [replaced 012] Assertion-Conditioned Compliance: A Provenance-Aware Vulnerability in Multi-Turn Tool-Calling Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全领域，针对多轮工具调用模型的鲁棒性问题，提出A-CC评估框架，检测模型对用户和系统错误断言的敏感性。**

- **链接: [https://arxiv.org/pdf/2512.00332v2](https://arxiv.org/pdf/2512.00332v2)**

> **作者:** Daud Waqas; Aaryamaan Golthi; Erika Hayashida; Huanzhi Mao
>
> **备注:** 15 pages (incl. Appendix), 3 figures, 7 tables
>
> **摘要:** Multi-turn tool-calling LLMs (models capable of invoking external APIs or tools across several user turns) have emerged as a key feature in modern AI assistants, enabling extended dialogues from benign tasks to critical business, medical, and financial operations. Yet implementing multi-turn pipelines remains difficult for many safety-critical industries due to ongoing concerns regarding model resilience. While standardized benchmarks such as the Berkeley Function-Calling Leaderboard (BFCL) have underpinned confidence concerning advanced function-calling models (like Salesforce's xLAM V2), there is still a lack of visibility into multi-turn conversation-level robustness, especially given their exposure to real-world systems. In this paper, we introduce Assertion-Conditioned Compliance (A-CC), a novel evaluation paradigm for multi-turn function-calling dialogues. A-CC provides holistic metrics that evaluate a model's behavior when confronted with misleading assertions originating from two distinct vectors: (1) user-sourced assertions (USAs), which measure sycophancy toward plausible but misinformed user beliefs, and (2) function-sourced assertions (FSAs), which measure compliance with plausible but contradictory system policies (e.g., stale hints from unmaintained tools). Our results show that models are highly vulnerable to both USA sycophancy and FSA policy conflicts, confirming A-CC as a critical, latent vulnerability in deployed agents.
>
---
#### [replaced 013] From Construction to Injection: Edit-Based Fingerprints for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型指纹任务，旨在解决LLM未经授权传播的问题。提出一种基于编辑的指纹框架，提升指纹的隐蔽性和检测性。**

- **链接: [https://arxiv.org/pdf/2509.03122v3](https://arxiv.org/pdf/2509.03122v3)**

> **作者:** Yue Li; Xin Yi; Dongsheng Shi; Yongyi Cui; Gerard de Melo; Linlin Wang
>
> **备注:** preprint
>
> **摘要:** Establishing reliable and verifiable fingerprinting mechanisms is fundamental to controlling the unauthorized redistribution of large language models (LLMs). However, existing approaches face two major challenges: (a) ensuring imperceptibility, including resistance to statistical identification and avoidance of accidental activation during fingerprint construction, and (b) preserving both model utility and fingerprint detectability under subsequent model modifications. To address these challenges, we propose an end-to-end fingerprinting framework with two components. First, we design a rule-based code-mixing fingerprint (CF) that maps natural-query-like prompts to multi-candidate targets, reducing accidental triggering via high-complexity code-mixing formulations. Second, we introduce Multi-Candidate Editing (MCEdit), which jointly optimizes multi-candidate targets and enforces margins between target and non-target outputs to improve post-modification detectability. Extensive experiments demonstrate that our framework provides a robust and practical solution for fingerprinting LLMs.
>
---
#### [replaced 014] OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于机器人感知任务，旨在评估不同光照条件下开放语义映射的性能。提出OSMa-Bench框架，通过新数据集和场景图方法分析模型的语义准确性和结构理解能力。**

- **链接: [https://arxiv.org/pdf/2503.10331v3](https://arxiv.org/pdf/2503.10331v3)**

> **作者:** Maxim Popov; Regina Kurkova; Mikhail Iumanov; Jaafar Mahmoud; Sergey Kolyubin
>
> **备注:** Project page: https://be2rlab.github.io/OSMa-Bench/
>
> **摘要:** Open Semantic Mapping (OSM) is a key technology in robotic perception, combining semantic segmentation and SLAM techniques. This paper introduces a dynamically configurable and highly automated LLM/LVLM-powered pipeline for evaluating OSM solutions called OSMa-Bench (Open Semantic Mapping Benchmark). The study focuses on evaluating state-of-the-art semantic mapping algorithms under varying indoor lighting conditions, a critical challenge in indoor environments. We introduce a novel dataset with simulated RGB-D sequences and ground truth 3D reconstructions, facilitating the rigorous analysis of mapping performance across different lighting conditions. Through experiments on leading models such as ConceptGraphs, BBQ, and OpenScene, we evaluate the semantic fidelity of object recognition and segmentation. Additionally, we introduce a Scene Graph evaluation method to analyze the ability of models to interpret semantic structure. The results provide insights into the robustness of these models, forming future research directions for developing resilient and adaptable robotic systems. Project page is available at https://be2rlab.github.io/OSMa-Bench/.
>
---
#### [replaced 015] Personality Editing for Language Models through Adjusting Self-Referential Queries
- **分类: cs.CL**

- **简介: 该论文属于语言模型个性控制任务，旨在解决传统方法成本高、效果不稳定的问题。提出PALETTE方法，通过调整自我参照查询实现高效个性编辑。**

- **链接: [https://arxiv.org/pdf/2502.11789v4](https://arxiv.org/pdf/2502.11789v4)**

> **作者:** Seojin Hwang; Yumin Kim; Byeongjeong Kim; Donghoon Shin; Hwanhee Lee
>
> **备注:** Accepted to EACL 2026 (Main)
>
> **摘要:** Large Language Models (LLMs) are integral to applications such as conversational agents and content creation, where precise control over a model's personality is essential for maintaining tone, consistency, and user engagement. However, prevailing prompt-based or fine-tuning approaches either lack robustness or demand large-scale training data, making them costly and impractical. In this paper, we present PALETTE (Personality Adjustment by LLM SElf-TargeTed quEries), a novel method for personality editing in LLMs. Our approach introduces adjustment queries, where self-referential statements grounded in psychological constructs are treated analogously to factual knowledge, enabling direct editing of personality-related responses. Unlike fine-tuning, PALETTE requires only 12 editing samples to achieve substantial improvements in personality alignment across personality dimensions. Experimental results from both automatic and human evaluations demonstrate that our method enables more stable and well-balanced personality control in LLMs.
>
---
#### [replaced 016] A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 本文综述量化图表示学习，旨在解决传统连续嵌入的参数效率、可解释性和鲁棒性问题。通过离散编码表示图结构，促进与大语言模型的融合。**

- **链接: [https://arxiv.org/pdf/2502.00681v2](https://arxiv.org/pdf/2502.00681v2)**

> **作者:** Qika Lin; Zhen Peng; Kaize Shi; Kai He; Yiming Xu; Jian Zhang; Erik Cambria; Mengling Feng
>
> **摘要:** Recent years have witnessed rapid advances in graph representation learning, with the continuous embedding approach emerging as the dominant paradigm. However, such methods encounter issues regarding parameter efficiency, interpretability, and robustness. Thus, Quantized Graph Representation (QGR) learning has recently gained increasing interest, which represents the graph structure with discrete codes instead of conventional continuous embeddings. Given its analogous representation form to natural language, QGR also possesses the capability to seamlessly integrate graph structures with large language models (LLMs). As this emerging paradigm is still in its infancy yet holds significant promise, we undertake this thorough survey to promote its rapid future prosperity. We first present the background of the general quantization methods and their merits. Moreover, we provide an in-depth demonstration of current QGR studies from the perspectives of quantized strategies, training objectives, distinctive designs, knowledge graph quantization, and applications. We further explore the strategies for code dependence learning and integration with LLMs. At last, we give discussions and conclude future directions, aiming to provide a comprehensive picture of QGR and inspire future research.
>
---
#### [replaced 017] Reward Shaping to Mitigate Reward Hacking in RLHF
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLHF中的奖励黑客问题。通过分析奖励塑造方法，提出PAR算法，提升训练稳定性与抗奖励黑客能力。**

- **链接: [https://arxiv.org/pdf/2502.18770v5](https://arxiv.org/pdf/2502.18770v5)**

> **作者:** Jiayi Fu; Xuandong Zhao; Chengyuan Yao; Heng Wang; Qi Han; Yanghua Xiao
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) is essential for aligning large language models (LLMs) with human values. However, RLHF is susceptible to \emph{reward hacking}, where the agent exploits flaws in the reward function rather than learning the intended behavior, thus degrading alignment. Although reward shaping helps stabilize RLHF and partially mitigate reward hacking, a systematic investigation into shaping techniques and their underlying principles remains lacking. To bridge this gap, we present a comprehensive study of the prevalent reward shaping methods. Our analysis suggests two key design principles: (1) the RL reward should be bounded, and (2) the RL reward benefits from rapid initial growth followed by gradual convergence. Guided by these insights, we propose Preference As Reward (PAR), a novel approach that leverages the latent preferences embedded within the reward model as the signal for reinforcement learning. Moreover, PAR exhibits two critical variance-reduction properties that contribute to stabilizing the RLHF training process and effectively extending the tolerance window for early stopping. We evaluated PAR on the base model Gemma2-2B using two datasets, Ultrafeedback-Binarized and HH-RLHF. Experimental results demonstrate PAR's superior performance over other reward shaping methods. On the AlpacaEval 2.0 benchmark, PAR achieves a win rate of at least 5 percentage points higher than competing approaches. Furthermore, PAR exhibits remarkable data efficiency, requiring only a single reference reward for optimal performance, and maintains robustness against reward hacking even after two full epochs of training. The code is available at https://github.com/PorUna-byte/PAR.
>
---
#### [replaced 018] GECOBench: A Gender-Controlled Text Dataset and Benchmark for Quantifying Biases in Explanations
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的可解释AI任务，旨在研究预训练模型在解释过程中是否继承性别偏见。作者构建了GECO数据集，并提出GECOBench评估框架，以量化和缓解解释偏差。**

- **链接: [https://arxiv.org/pdf/2406.11547v2](https://arxiv.org/pdf/2406.11547v2)**

> **作者:** Rick Wilming; Artur Dox; Hjalmar Schulz; Marta Oliveira; Benedict Clark; Stefan Haufe
>
> **备注:** Published in Frontiers
>
> **摘要:** Large pre-trained language models have become a crucial backbone for many downstream tasks in natural language processing (NLP), and while they are trained on a plethora of data containing a variety of biases, such as gender biases, it has been shown that they can also inherit such biases in their weights, potentially affecting their prediction behavior. However, it is unclear to what extent these biases also affect feature attributions generated by applying "explainable artificial intelligence" (XAI) techniques, possibly in unfavorable ways. To systematically study this question, we create a gender-controlled text dataset, GECO, in which the alteration of grammatical gender forms induces class-specific words and provides ground truth feature attributions for gender classification tasks. This enables an objective evaluation of the correctness of XAI methods. We apply this dataset to the pre-trained BERT model, which we fine-tune to different degrees, to quantitatively measure how pre-training induces undesirable bias in feature attributions and to what extent fine-tuning can mitigate such explanation bias. To this extent, we provide GECOBench, a rigorous quantitative evaluation framework for benchmarking popular XAI methods. We show a clear dependency between explanation performance and the number of fine-tuned layers, where XAI methods are observed to benefit particularly from fine-tuning or complete retraining of embedding layers.
>
---
#### [replaced 019] AI-generated data contamination erodes pathological variability and diagnostic reliability
- **分类: cs.CY; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于医疗AI领域，研究AI生成数据对病理多样性与诊断可靠性的影响。工作包括分析合成数据、发现模型偏差，并评估缓解策略。**

- **链接: [https://arxiv.org/pdf/2601.12946v2](https://arxiv.org/pdf/2601.12946v2)**

> **作者:** Hongyu He; Shaowen Xiang; Ye Zhang; Yingtao Zhu; Jin Zhang; Hao Deng; Emily Alsentzer; Qingyu Chen; Kun-Hsing Yu; Andrew Marshall; Tingting Chen; Srinivas Anumasa; Daniel Ebner; Dean Ho; Kee Yuan Ngiam; Ching-Yu Cheng; Dianbo Liu
>
> **备注:** *Corresponding author: Dianbo Liu (dianbo@nus.edu.sg)
>
> **摘要:** Generative artificial intelligence (AI) is rapidly populating medical records with synthetic content, creating a feedback loop where future models are increasingly at risk of training on uncurated AI-generated data. However, the clinical consequences of this AI-generated data contamination remain unexplored. Here, we show that in the absence of mandatory human verification, this self-referential cycle drives a rapid erosion of pathological variability and diagnostic reliability. By analysing more than 800,000 synthetic data points across clinical text generation, vision-language reporting, and medical image synthesis, we find that models progressively converge toward generic phenotypes regardless of the model architecture. Specifically, rare but critical findings, including pneumothorax and effusions, vanish from the synthetic content generated by AI models, while demographic representations skew heavily toward middle-aged male phenotypes. Crucially, this degradation is masked by false diagnostic confidence; models continue to issue reassuring reports while failing to detect life-threatening pathology, with false reassurance rates tripling to 40%. Blinded physician evaluation confirms that this decoupling of confidence and accuracy renders AI-generated documentation clinically useless after just two generations. We systematically evaluate three mitigation strategies, finding that while synthetic volume scaling fails to prevent collapse, mixing real data with quality-aware filtering effectively preserves diversity. Ultimately, our results suggest that without policy-mandated human oversight, the deployment of generative AI threatens to degrade the very healthcare data ecosystems it relies upon.
>
---
#### [replaced 020] QueStER: Query Specification for Generative keyword-based Retrieval
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出QueStER，解决生成式检索与传统检索结合的问题。通过生成关键词查询，提升检索效果并保持效率。属于信息检索任务。**

- **链接: [https://arxiv.org/pdf/2511.05301v2](https://arxiv.org/pdf/2511.05301v2)**

> **作者:** Arthur Satouf; Yuxuan Zong; Habiboulaye Amadou-Boubacar; Pablo Piantanida; Benjamin Piwowarski
>
> **摘要:** Generative retrieval (GR) differs from the traditional index-then-retrieve pipeline by storing relevance in model parameters and generating retrieval cues directly from the query, but it can be brittle out of domain and expensive to scale. We introduce QueStER (QUEry SpecificaTion for gEnerative Keyword-Based Retrieval), which bridges GR and query reformulation by learning to generate explicit keyword-based search specifications. Given a user query, a lightweight LLM produces a keyword query that is executed by a standard retriever (BM25), combining the generalization benefits of generative query rewriting with the efficiency and scalability of lexical indexing. We train the rewriting policy with reinforcement learning techniques. Across in- and out-of-domain evaluations, QueStER consistently improves over BM25 and is competitive with neural IR baselines, while maintaining strong efficiency.
>
---
#### [replaced 021] Learning to Explain: Supervised Token Attribution from Transformer Attention Patterns
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于可解释AI任务，旨在解决模型透明度问题。提出ExpNet，通过学习Transformer注意力模式映射到词粒度重要性得分，自动发现最佳特征组合。**

- **链接: [https://arxiv.org/pdf/2601.14112v2](https://arxiv.org/pdf/2601.14112v2)**

> **作者:** George Mihaila
>
> **摘要:** Explainable AI (XAI) has become critical as transformer-based models are deployed in high-stakes applications including healthcare, legal systems, and financial services, where opacity hinders trust and accountability. Transformers self-attention mechanisms have proven valuable for model interpretability, with attention weights successfully used to understand model focus and behavior (Xu et al., 2015); (Wiegreffe and Pinter, 2019). However, existing attention-based explanation methods rely on manually defined aggregation strategies and fixed attribution rules (Abnar and Zuidema, 2020a); (Chefer et al., 2021), while model-agnostic approaches (LIME, SHAP) treat the model as a black box and incur significant computational costs through input perturbation. We introduce Explanation Network (ExpNet), a lightweight neural network that learns an explicit mapping from transformer attention patterns to token-level importance scores. Unlike prior methods, ExpNet discovers optimal attention feature combinations automatically rather than relying on predetermined rules. We evaluate ExpNet in a challenging cross-task setting and benchmark it against a broad spectrum of model-agnostic methods and attention-based techniques spanning four methodological families.
>
---
#### [replaced 022] Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决LLMs对句子级否定理解不足的问题。提出Thunder-NUBench基准，包含精心构建的句义否定对和多项选择数据集，以全面评估模型对否定的理解能力。**

- **链接: [https://arxiv.org/pdf/2506.14397v4](https://arxiv.org/pdf/2506.14397v4)**

> **作者:** Yeonkyoung So; Gyuseong Lee; Sungmok Jung; Joonhak Lee; JiA Kang; Sangho Kim; Jaejin Lee
>
> **摘要:** Negation is a fundamental linguistic phenomenon that poses ongoing challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Current benchmarks often treat negation as a minor detail within broader tasks, such as natural language inference. Consequently, there is a lack of benchmarks specifically designed to evaluate comprehension of negation. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly created to assess sentence-level understanding of negation in LLMs. Thunder-NUBench goes beyond merely identifying surface-level cues by contrasting standard negation with structurally diverse alternatives, such as local negation, contradiction, and paraphrase. This benchmark includes manually curated sentence-negation pairs and a multiple-choice dataset, allowing for a comprehensive evaluation of models' understanding of negation.
>
---
#### [replaced 023] Translation via Annotation: A Computational Study of Translating Classical Chinese into Japanese
- **分类: cs.CL**

- **简介: 该论文研究古典中文到日文的翻译任务，解决低资源问题。通过引入LLM辅助标注流程和构建新数据集，提升序列标注效果，补充LLMs以改善注释质量。**

- **链接: [https://arxiv.org/pdf/2511.05239v2](https://arxiv.org/pdf/2511.05239v2)**

> **作者:** Zilong Li; Jie Cao
>
> **摘要:** Ancient people translated classical Chinese into Japanese using a system of annotations placed around characters. We abstract this process as sequence tagging tasks and fit them into modern language technologies. The research on this annotation and translation system faces a low resource problem. We alleviate this problem by introducing an LLM-based annotation pipeline and constructing a new dataset from digitized open-source translation data. We show that in the low-resource setting, introducing auxiliary Chinese NLP tasks enhances the training of sequence tagging tasks. We also evaluate the performance of Large Language Models (LLMs) on this task. While they achieve high scores on direct machine translation, our method could serve as a supplement to LLMs to improve the quality of character's annotation.
>
---
#### [replaced 024] SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation
- **分类: cs.CL**

- **简介: 该论文提出Synapse，解决LLM代理长期记忆断连问题，通过动态图模型实现语义记忆增强，提升复杂推理能力。**

- **链接: [https://arxiv.org/pdf/2601.02744v2](https://arxiv.org/pdf/2601.02744v2)**

> **作者:** Hanqi Jiang; Junhao Chen; Yi Pan; Ling Chen; Weihang You; Yifan Zhou; Ruidong Zhang; Lin Zhao; Yohannes Abate; Tianming Liu
>
> **摘要:** While Large Language Models (LLMs) excel at generalized reasoning, standard retrieval-augmented approaches fail to address the disconnected nature of long-term agentic memory. To bridge this gap, we introduce Synapse (Synergistic Associative Processing Semantic Encoding), a unified memory architecture that transcends static vector similarity. Drawing from cognitive science, Synapse models memory as a dynamic graph where relevance emerges from spreading activation rather than pre-computed links. By integrating lateral inhibition and temporal decay, the system dynamically highlights relevant sub-graphs while filtering interference. We implement a Triple Hybrid Retrieval strategy that fuses geometric embeddings with activation-based graph traversal. Comprehensive evaluations on the LoCoMo benchmark show that Synapse significantly outperforms state-of-the-art methods in complex temporal and multi-hop reasoning tasks, offering a robust solution to the "Contextual Tunneling" problem. Our code and data will be made publicly available upon acceptance.
>
---
#### [replaced 025] Mitigating Data Imbalance in Automated Speaking Assessment
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语言评估任务，解决ASA中的数据不平衡问题。通过引入BLV损失函数，提升模型对少数类的识别能力，增强评估公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2509.03010v2](https://arxiv.org/pdf/2509.03010v2)**

> **作者:** Fong-Chun Tsai; Kuan-Tang Huang; Bi-Cheng Yan; Tien-Hong Lo; Berlin Chen
>
> **备注:** Accepted by APSIPA 2025; revised figure, references added
>
> **摘要:** Automated Speaking Assessment (ASA) plays a crucial role in evaluating second-language (L2) learners proficiency. However, ASA models often suffer from class imbalance, leading to biased predictions. To address this, we introduce a novel objective for training ASA models, dubbed the Balancing Logit Variation (BLV) loss, which perturbs model predictions to improve feature representation for minority classes without modifying the dataset. Evaluations on the ICNALE benchmark dataset show that integrating the BLV loss into a celebrated text-based (BERT) model significantly enhances classification accuracy and fairness, making automated speech evaluation more robust for diverse learners.
>
---
#### [replaced 026] Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在解决现有方法难以捕捉语音中稀疏且异构的抑郁特征的问题。通过构建分层表示编码器，融合声学与语义信息，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2510.08593v2](https://arxiv.org/pdf/2510.08593v2)**

> **作者:** Yuxin Li; Eng Siong Chng; Cuntai Guan
>
> **摘要:** Speech-based depression detection (SDD) has emerged as a non-invasive and scalable alternative to conventional clinical assessments. However, existing methods still struggle to capture robust depression-related speech characteristics, which are sparse and heterogeneous. Although pretrained self-supervised learning (SSL) models provide rich representations, most recent SDD studies extract features from a single layer of the pretrained SSL model for the downstream classifier. This practice overlooks the complementary roles of low-level acoustic features and high-level semantic information inherently encoded in different SSL model layers. To explicitly model interactions between acoustic and semantic representations within an utterance, we propose a hierarchical adaptive representation encoder with prior knowledge that disengages and re-aligns acoustic and semantic information through asymmetric cross-attention, enabling fine-grained acoustic patterns to be interpreted in semantic context. In addition, a Connectionist Temporal Classification (CTC) objective is applied as auxiliary supervision to handle the irregular temporal distribution of depressive characteristics without requiring frame-level annotations. Experiments on DAIC-WOZ and MODMA demonstrate that HAREN-CTC consistently outperforms existing methods under both performance upper-bound evaluation and generalization evaluation settings, achieving Macro F1 scores of 0.81 and 0.82 respectively in upper-bound evaluation, and maintaining superior performance with statistically significant improvements in precision and AUC under rigorous cross-validation. These findings suggest that modeling hierarchical acoustic-semantic interactions better reflects how depressive characteristics manifest in natural speech, enabling scalable and objective depression assessment.
>
---
#### [replaced 027] SPECTRE: Conditional System Prompt Poisoning to Hijack LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全任务，旨在解决LLM系统提示被污染的问题。通过构造特定触发条件，使模型在特定查询下输出受控内容，同时保持正常功能。**

- **链接: [https://arxiv.org/pdf/2505.16888v3](https://arxiv.org/pdf/2505.16888v3)**

> **作者:** Viet Pham; Thai Le
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed via third-party system prompts downloaded from public marketplaces. We identify a critical supply-chain vulnerability: conditional system prompt poisoning, where an adversary injects a ``sleeper agent'' into a benign-looking prompt. Unlike traditional jailbreaks that aim for broad refusal-breaking, our proposed framework, SPECTRE, optimizes system prompts to trigger LLMs to output targeted, compromised responses only for specific queries (e.g., ``Who should I vote for the US President?'') while maintaining high utility on benign inputs. Operating in a strict black-box setting without model weight access, SPECTRE utilizes a two-stage optimization including a global semantic search followed by a greedy lexical refinement. Tested on open-source models and commercial APIs (GPT-4o-mini, GPT-3.5), SPECTRE achieves up to 70% F1 reduction on targeted queries with minimal degradation to general capabilities. We further demonstrate that these poisoned prompts evade standard defenses, including perplexity filters and typo-correction, by exploiting the natural noise found in real-world system prompts. Our code and data are available at https://github.com/vietph34/CAIN. WARNING: Our paper contains examples that might be sensitive to the readers!
>
---
#### [replaced 028] Monadic Context Engineering
- **分类: cs.AI; cs.CL; cs.FL**

- **简介: 该论文提出Monadic Context Engineering（MCE），解决AI代理架构中的状态管理、错误处理和并发问题，通过函数式编程结构提升系统稳定性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.22431v4](https://arxiv.org/pdf/2512.22431v4)**

> **作者:** Yifan Zhang; Yang Yuan; Mengdi Wang; Andrew Chi-Chih Yao
>
> **摘要:** The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in state management, error handling, and concurrency. This paper introduces Monadic Context Engineering (MCE), a novel architectural paradigm leveraging the algebraic structures of Functors, Applicative Functors, and Monads to provide a formal foundation for agent design. MCE treats agent workflows as computational contexts where cross-cutting concerns, such as state propagation, short-circuiting error handling, and asynchronous execution, are managed intrinsically by the algebraic properties of the abstraction. We demonstrate how Monads enable robust sequential composition, how Applicatives provide a principled structure for parallel execution, and crucially, how Monad Transformers allow for the systematic composition of these capabilities. This layered approach enables developers to construct complex, resilient, and efficient AI agents from simple, independently verifiable components. We further extend this framework to describe Meta-Agents, which leverage MCE for generative orchestration, dynamically creating and managing sub-agent workflows through metaprogramming.
>
---
#### [replaced 029] DARC: Decoupled Asymmetric Reasoning Curriculum for LLM Evolution
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DARC框架，解决自博弈中语言模型进化不稳定问题，通过分阶段训练提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.13761v2](https://arxiv.org/pdf/2601.13761v2)**

> **作者:** Shengda Fan; Xuyan Ye; Yankai Lin
>
> **摘要:** Self-play with large language models has emerged as a promising paradigm for achieving self-improving artificial intelligence. However, existing self-play frameworks often suffer from optimization instability, due to (i) non-stationary objectives induced by solver-dependent reward feedback for the Questioner, and (ii) bootstrapping errors from self-generated pseudo-labels used to supervise the Solver. To mitigate these challenges, we introduce DARC (Decoupled Asymmetric Reasoning Curriculum), a two-stage framework that stabilizes the self-evolution process. First, we train the Questioner to synthesize difficulty-calibrated questions, conditioned on explicit difficulty levels and external corpora. Second, we train the Solver with an asymmetric self-distillation mechanism, where a document-augmented teacher generates high-quality pseudo-labels to supervise the student Solver that lacks document access. Empirical results demonstrate that DARC is model-agnostic, yielding an average improvement of 10.9 points across nine reasoning benchmarks and three backbone models. Moreover, DARC consistently outperforms all baselines and approaches the performance of fully supervised models without relying on human annotations. The code is available at https://github.com/RUCBM/DARC.
>
---
#### [replaced 030] Language-Native Materials Processing Design by Lightly Structured Text Database and Reasoning Large Language Model
- **分类: cs.DB; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 该论文属于材料合成规划任务，旨在解决传统文本记录难以优化的问题。通过构建轻度结构化数据库和大语言模型，实现文本推理与参数筛选，提升材料制备效率。**

- **链接: [https://arxiv.org/pdf/2509.06093v3](https://arxiv.org/pdf/2509.06093v3)**

> **作者:** Yuze Liu; Zhaoyuan Zhang; Xiangsheng Zeng; Yihe Zhang; Leping Yu; Liu Yang; Lejia Wang; Xi Yu
>
> **摘要:** Materials synthesis procedures are predominantly documented as narrative text in protocols and lab notebooks, rendering them inaccessible to conventional structured data optimization. This language-native nature poses a critical challenge for complex, multistage processes--such as the preparation of boron nitride nanosheet (BNNS)--where outcomes depend on path-dependent choices in exfoliation and functionalization. Here, we recast synthesis planning as a text reasoning task enabled by a lightly structured text database, which preserves the conditional logic and causal contexts essential for expert-like decision-making. Building on a heterogeneous schema that indexes both narrative excerpts and computable entities (e.g., reaction conditions), our system implements a hybrid retrieval engine to combine semantic context with precise parameter filtering. On top of this, the framework operates in two modes, i.e. retrieval-augmented generation (RAG), which grounds recommendations in retrieved evidence modules, and experience-augmented reasoning (EAR), which uses iteratively refined text guides distilled from multi-source narrative data. Instead of suggesting single "optimal" settings, the system produces interpretable guidance aligned with expert reasoning patterns--hypotheses, parameter ranges, and citation-backed standard operating procedures--that support iterative planning and failure diagnosis. We validated this framework on the targeted exfoliation of BNNS, a process highly sensitive to multivariate constraints. The system successfully identified optimal combinations of grinding aids, milling configurations, and separation strategies from a wide range of literature-reported methods, which were experimentally verified to yield high-quality nanosheets, illustrating the potential of language-native reasoning to streamline critical operations in materials processing.
>
---
#### [replaced 031] Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决CoT推理延迟过高问题。通过V-Skip方法，结合视觉锚点优化，实现高效压缩，提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2601.13879v2](https://arxiv.org/pdf/2601.13879v2)**

> **作者:** Dongxu Zhang; Yiding Sun; Cheng Tan; Wenbiao Yan; Ning Yang; Jihua Zhu; Haijun Zhang
>
> **摘要:** While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA.
>
---
#### [replaced 032] RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于代码审查自动化任务，旨在解决无需微调的高质量代码评论生成问题。工作包括设计并部署RovoDev工具，验证其有效性与实际效益。**

- **链接: [https://arxiv.org/pdf/2601.01129v2](https://arxiv.org/pdf/2601.01129v2)**

> **作者:** Kla Tantithamthavorn; Yaotian Zou; Andy Wong; Michael Gupta; Zhe Wang; Mike Buller; Ryan Jiang; Matthew Watson; Minwoo Jeong; Kun Chen; Ming Wu
>
> **备注:** Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
>
> **摘要:** Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
>
---
#### [replaced 033] Extending Audio Context for Long-Form Understanding in Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型任务，解决长音频理解中上下文受限的问题。提出Partial YaRN和VLAT方法，扩展音频上下文长度并提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.15231v2](https://arxiv.org/pdf/2510.15231v2)**

> **作者:** Yuatyong Chaichana; Pittawat Taveekitworachai; Warit Sirichotedumrong; Potsawee Manakul; Kunat Pipatanakul
>
> **备注:** EACL 2026. Code and dataset are available at: https://github.com/yophis/partial-yarn
>
> **摘要:** Large Audio-Language Models (LALMs) are often constrained by short audio context windows, even when their text backbones support long contexts, limiting long-form audio understanding. Prior work has introduced context-extension methods (e.g. YaRN) on unimodal LLMs, yet their application to LALMs remains unexplored. First, building on RoPE-based context extension, we introduce Partial YaRN, a training-free, modality-decoupled extension method that modifies only audio token positions, leaving text positions intact to preserve the base LLM's text capabilities. Second, we propose Virtual Longform Audio Training (VLAT), a training strategy that extends Partial YaRN into a training-time positional augmentation. VLAT simulates diverse audio lengths during training, enabling generalization to inputs far longer than those seen in training. Our experiments on SALMONN and Qwen2-Audio confirm that Partial YaRN outperforms the original models across wide range of settings, and VLAT provides substantial performance improvement on long audio of unseen lengths.
>
---
#### [replaced 034] Seer Self-Consistency: Advance Budget Estimation for Adaptive Test-Time Scaling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM推理中计算成本高的问题。提出SeerSC框架，通过结合快速推理与动态自一致性，降低token消耗和延迟。**

- **链接: [https://arxiv.org/pdf/2511.09345v2](https://arxiv.org/pdf/2511.09345v2)**

> **作者:** Shiyu Ji; Yixuan Wang; Yijun Liu; Qingfu Zhu; Wanxiang Che
>
> **摘要:** Test-time scaling improves the inference performance of Large Language Models (LLMs) but also incurs substantial computational costs. Although recent studies have reduced token consumption through dynamic self-consistency, they remain constrained by the high latency of sequential requests. In this paper, we propose SeerSC, a dynamic self-consistency framework that simultaneously improves token efficiency and latency by integrating System 1 and System 2 reasoning. Specifically, we utilize the rapid System 1 to compute the answer entropy for given queries. This score is then used to evaluate the potential of samples for scaling, enabling dynamic self-consistency under System 2. Benefiting from the advance and accurate estimation provided by System 1, the proposed method can reduce token usage while simultaneously achieving a significant decrease in latency through parallel generation. It outperforms existing methods, achieving up to a 47% reduction in token consumption and a 43% reduction in inference latency without significant performance loss.
>
---
#### [replaced 035] Large Language Models Encode Semantics and Alignment in Linearly Separable Representations
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM的潜在空间几何结构，解决语义与对齐表示的线性可分性问题。通过实验发现高阶语义信息存在于低维子空间，提出基于MLP的轻量防护机制提升安全性能。**

- **链接: [https://arxiv.org/pdf/2507.09709v3](https://arxiv.org/pdf/2507.09709v3)**

> **作者:** Baturay Saglam; Paul Kassianik; Blaine Nelson; Sajana Weerawardhena; Yaron Singer; Amin Karbasi
>
> **备注:** IJCNLP and the Asian Chapter of ACL
>
> **摘要:** Understanding the latent space geometry of large language models (LLMs) is key to interpreting their behavior and improving alignment. Yet it remains unclear to what extent LLMs linearly organize representations related to semantic understanding. To explore this, we conduct a large-scale empirical study of hidden representations in 11 autoregressive models across six scientific topics. We find that high-level semantic information consistently resides in low-dimensional subspaces that form linearly separable representations across domains. This separability becomes more pronounced in deeper layers and under prompts that elicit structured reasoning or alignment behavior$\unicode{x2013}$even when surface content remains unchanged. These findings motivate geometry-aware tools that operate directly in latent space to detect and mitigate harmful and adversarial content. As a proof of concept, we train an MLP probe on final-layer hidden states as a lightweight latent-space guardrail. This approach substantially improves refusal rates on malicious queries and prompt injections that bypass both the model's built-in safety alignment and external token-level filters.
>
---
#### [replaced 036] Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索与生成任务，旨在解决大语言模型的幻觉问题。通过综述图方法在检索增强生成中的应用，分析其功能与影响，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2504.10499v2](https://arxiv.org/pdf/2504.10499v2)**

> **作者:** Zulun Zhu; Tiancheng Huang; Kai Wang; Junda Ye; Xinghe Chen; Siqiang Luo
>
> **摘要:** Large language models (LLMs) struggle with the factual error during inference due to the lack of sufficient training data and the most updated knowledge, leading to the hallucination problem. Retrieval-Augmented Generation (RAG) has gained attention as a promising solution to address the limitation of LLMs, by retrieving relevant information from external source to generate more accurate answers to the questions. Given the pervasive presence of structured knowledge in the external source, considerable strides in RAG have been made to employ the techniques related to graphs and achieve more complex reasoning based on the topological information between knowledge entities. However, there is currently neither unified review examining the diverse roles of graphs in RAG, nor a comprehensive resource to help researchers navigate and contribute to this evolving field. This survey offers a novel perspective on the functionality of graphs within RAG and their impact on enhancing performance across a wide range of graph-structured data. It provides a detailed breakdown of the roles that graphs play in RAG, covering database construction, algorithms, pipelines, and tasks. Finally, it identifies current challenges and outline future research directions, aiming to inspire further developments in this field. Our graph-centered analysis highlights the commonalities and differences in existing methods, setting the stage for future researchers in areas such as graph learning, database systems, and natural language processing.
>
---
#### [replaced 037] A Component-Based Survey of Interactions between Large Language Models and Multi-Armed Bandits
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于跨领域研究任务，探讨大语言模型与多臂老虎机的双向交互。解决两者结合中的挑战，分析增强系统设计与性能，推动相关技术发展。**

- **链接: [https://arxiv.org/pdf/2601.12945v2](https://arxiv.org/pdf/2601.12945v2)**

> **作者:** Miao Xie; Siguang Chen; Chunli Lv
>
> **备注:** 27 pages, 6 table
>
> **摘要:** Large language models (LLMs) have become powerful and widely used systems for language understanding and generation, while multi-armed bandit (MAB) algorithms provide a principled framework for adaptive decision-making under uncertainty. This survey explores the potential at the intersection of these two fields. As we know, it is the first survey to systematically review the bidirectional interaction between large language models and multi-armed bandits at the component level. We highlight the bidirectional benefits: MAB algorithms address critical LLM challenges, spanning from pre-training to retrieval-augmented generation (RAG) and personalization. Conversely, LLMs enhance MAB systems by redefining core components such as arm definition and environment modeling, thereby improving decision-making in sequential tasks. We analyze existing LLM-enhanced bandit systems and bandit-enhanced LLM systems, providing insights into their design, methodologies, and performance. Key challenges and representative findings are identified to help guide future research. An accompanying GitHub repository that indexes relevant literature is available at https://github.com/bucky1119/Awesome-LLM-Bandit-Interaction.
>
---
#### [replaced 038] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出BayesAgent，将LLM代理与概率图模型结合，解决不确定环境下的推理问题。通过vPGM框架，提升模型的置信度校准和生成质量。**

- **链接: [https://arxiv.org/pdf/2406.05516v4](https://arxiv.org/pdf/2406.05516v4)**

> **作者:** Hengguan Huang; Xing Shen; Songtao Wang; Lingfa Meng; Dianbo Liu; David Alejandro Duchene; Hao Wang; Samir Bhatt
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Human cognition excels at transcending sensory input and forming latent representations that structure our understanding of the world. While Large Language Model (LLM) agents demonstrate emergent reasoning and decision-making abilities, they lack a principled framework for capturing latent structures and modeling uncertainty. In this work, we explore for the first time how to bridge LLM agents with probabilistic graphical models (PGMs) to address agentic reasoning under uncertainty. To this end, we introduce Verbalized Probabilistic Graphical Modeling (vPGM), a Bayesian agentic framework that (i) guides LLM agents in following key principles of PGMs through natural language and (ii) refines the resulting posterior distributions via numerical Bayesian inference. Unlike many traditional probabilistic methods requiring substantial domain expertise, vPGM bypasses expert-driven model design, making it well-suited for scenarios with limited assumptions. We evaluated our model on several agentic reasoning tasks, both close-ended and open-ended. Our results indicate that the model effectively enhances confidence calibration and text generation quality.
>
---
#### [replaced 039] OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OptimAI，解决从自然语言描述中自动优化问题的任务。通过LLM驱动的AI代理，将问题转化为数学形式并选择合适求解器，提升优化效率与准确性。**

- **链接: [https://arxiv.org/pdf/2504.16918v3](https://arxiv.org/pdf/2504.16918v3)**

> **作者:** Raghav Thind; Youran Sun; Ling Liang; Haizhao Yang
>
> **摘要:** Optimization plays a vital role in scientific research and practical applications. However, formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce OptimAI, a framework for solving Optimization problems described in natural language by leveraging LLM-powered AI agents, and achieve superior performance over current state-of-the-art methods. Our framework is built upon the following key roles: (1) a formulator that translates natural language problem descriptions into precise mathematical formulations; (2) a planner that constructs a high-level solution strategy prior to execution; and (3) a coder and a code critic capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, and our experiments confirm that combining diverse models leads to performance gains. Our approach attains 88.1% accuracy on the NLP4LP dataset and 82.3% on the Optibench dataset, reducing error rates by 58% and 52%, respectively, over prior best results.
>
---
#### [replaced 040] LoSemB: Logic-Guided Semantic Bridging for Inductive Tool Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工具检索任务，解决新工具识别问题。提出LoSemB框架，通过逻辑引导的语义桥梁实现无监督工具检索，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.07690v2](https://arxiv.org/pdf/2508.07690v2)**

> **作者:** Luyao Zhuang; Qinggang Zhang; Huachi Zhou; Yujing Zhang; Xiao Huang
>
> **摘要:** Tool learning has emerged as a promising paradigm for large language models (LLMs) to solve many real-world tasks. Nonetheless, with the tool repository rapidly expanding, it is impractical to contain all tools within the limited input length of LLMs. To alleviate these issues, researchers have explored incorporating a tool retrieval module to select the most relevant tools or represent tools as unique tokens within LLM parameters. However, most state-of-the-art methods are under transductive settings, assuming all tools have been observed during training. Such a setting deviates from reality as the real-world tool repository is evolving and incorporates new tools frequently. When dealing with these unseen tools, which refer to tools not encountered during the training phase, these methods are limited by two key issues, including the large distribution shift and the vulnerability of similarity-based retrieval. To this end, inspired by human cognitive processes of mastering unseen tools through discovering and applying the logical information from prior experience, we introduce a novel Logic-Guided Semantic Bridging framework for inductive tool retrieval, namely, LoSemB, which aims to mine and transfer latent logical information for inductive tool retrieval without costly retraining. Specifically, LoSemB contains a logic-based embedding alignment module to mitigate distribution shifts and implements a relational augmented retrieval mechanism to reduce the vulnerability of similarity-based retrieval. Extensive experiments demonstrate that LoSemB achieves advanced performance in inductive settings while maintaining desirable effectiveness in the transductive setting.
>
---
#### [replaced 041] SciHorizon-GENE: Benchmarking LLM for Life Sciences Inference from Gene Knowledge to Functional Understanding
- **分类: q-bio.GN; cs.AI; cs.CL**

- **简介: 该论文属于生物信息学任务，旨在解决LLM在基因到功能推理中的可靠性问题。构建了SciHorizon-GENE基准，评估模型在四个关键方面的表现。**

- **链接: [https://arxiv.org/pdf/2601.12805v2](https://arxiv.org/pdf/2601.12805v2)**

> **作者:** Xiaohan Huang; Meng Xiao; Chuan Qin; Qingqing Long; Jinmiao Chen; Yuanchun Zhou; Hengshu Zhu
>
> **备注:** 16 pages
>
> **摘要:** Large language models (LLMs) have shown growing promise in biomedical research, particularly for knowledge-driven interpretation tasks. However, their ability to reliably reason from gene-level knowledge to functional understanding, a core requirement for knowledge-enhanced cell atlas interpretation, remains largely underexplored. To address this gap, we introduce SciHorizon-GENE, a large-scale gene-centric benchmark constructed from authoritative biological databases. The benchmark integrates curated knowledge for over 190K human genes and comprises more than 540K questions covering diverse gene-to-function reasoning scenarios relevant to cell type annotation, functional interpretation, and mechanism-oriented analysis. Motivated by behavioral patterns observed in preliminary examinations, SciHorizon-GENE evaluates LLMs along four biologically critical perspectives: research attention sensitivity, hallucination tendency, answer completeness, and literature influence, explicitly targeting failure modes that limit the safe adoption of LLMs in biological interpretation pipelines. We systematically evaluate a wide range of state-of-the-art general-purpose and biomedical LLMs, revealing substantial heterogeneity in gene-level reasoning capabilities and persistent challenges in generating faithful, complete, and literature-grounded functional interpretations. Our benchmark establishes a systematic foundation for analyzing LLM behavior at the gene scale and offers insights for model selection and development, with direct relevance to knowledge-enhanced biological interpretation.
>
---
#### [replaced 042] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音问答任务，旨在解决长音频处理难题。提出CLSR模型，通过对比学习高效提取相关语音片段，提升长格式语音问答性能。**

- **链接: [https://arxiv.org/pdf/2511.09282v2](https://arxiv.org/pdf/2511.09282v2)**

> **作者:** Jiliang Hu; Zuchao Li; Baoyuan Qi; Liu Guoming; Ping Wang
>
> **备注:** 12 pages, 7 figures, accepted by AAAI 2026
>
> **摘要:** Significant progress has been made in spoken question answering (SQA) in recent years. However, many existing methods, including large audio language models, struggle with processing long audio. Follow the success of retrieval augmented generation, a speech-related retriever shows promising in help preprocessing long-form speech. But the performance of existing speech-related retrievers is lacking. To address this challenge, we propose CLSR, an end-to-end contrastive language-speech retriever that efficiently extracts question-relevant segments from long audio recordings for downstream SQA task. Unlike conventional speech-text contrastive models, CLSR incorporates an intermediate step that converts acoustic features into text-like representations prior to alignment, thereby more effectively bridging the gap between modalities. Experimental results across four cross-modal retrieval datasets demonstrate that CLSR surpasses both end-to-end speech related retrievers and pipeline approaches combining speech recognition with text retrieval, providing a robust foundation for advancing practical long-form SQA applications.
>
---
#### [replaced 043] Manifold-based Sampling for In-Context Hallucination Detection in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的事实性错误检测任务，旨在解决大语言模型生成内容中的幻觉问题。通过基于流形的演示采样方法MB-ICL，提升事实验证和幻觉检测效果。**

- **链接: [https://arxiv.org/pdf/2601.06196v2](https://arxiv.org/pdf/2601.06196v2)**

> **作者:** Bodla Krishna Vamshi; Rohan Bhatnagar; Haizhao Yang
>
> **摘要:** Large language models (LLMs) frequently generate factually incorrect or unsupported content, commonly referred to as hallucinations. Prior work has explored decoding strategies, retrieval augmentation, and supervised fine-tuning for hallucination detection, while recent studies show that in-context learning (ICL) can substantially influence factual reliability. However, existing ICL demonstration selection methods often rely on surface-level similarity heuristics and exhibit limited robustness across tasks and models. We propose MB-ICL, a manifold-based demonstration sampling framework for selecting in-context demonstrations that leverages latent representations extracted from frozen LLMs. By jointly modeling local manifold structure and class-aware prototype geometry, MB-ICL selects demonstrations based on their proximity to learned prototypes rather than lexical or embedding similarity alone. Across factual verification (FEVER) and hallucination detection (HaluEval) benchmarks, MB-ICL outperforms standard ICL selection baselines in the majority of evaluated settings, with particularly strong gains on dialogue and summarization tasks. The method remains robust under temperature perturbations and model variation, indicating improved stability compared to heuristic retrieval strategies. While lexical retrieval can remain competitive in certain question-answering regimes, our results demonstrate that manifold-based prototype selection provides a reliable and training light approach for hallucination detection without modifying LLM parameters, offering a principled direction for improved ICL demonstration selection.
>
---
#### [replaced 044] Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于持续后训练任务，解决模型在持续学习中遗忘问题。通过对比监督微调与强化微调，发现RFT能有效保持知识并提升性能。**

- **链接: [https://arxiv.org/pdf/2507.05386v5](https://arxiv.org/pdf/2507.05386v5)**

> **作者:** Song Lai; Haohan Zhao; Rong Feng; Changyi Ma; Wenzhuo Liu; Hongbo Zhao; Xi Lin; Dong Yi; Qingfu Zhang; Hongbin Liu; Gaofeng Meng; Fei Zhu
>
> **摘要:** Continual post-training (CPT) is a popular and effective technique for adapting foundation models like multimodal large language models to specific and ever-evolving downstream tasks. While existing research has primarily concentrated on methods like data replay, model expansion, or parameter regularization, the fundamental role of the learning paradigm within CPT remains largely unexplored. This paper presents a comparative analysis of two core post-training paradigms: supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT), investigating their respective impacts on knowledge retention during CPT. Our experiments are conducted on a benchmark comprising seven diverse multimodal tasks, utilizing Qwen2.5-VL-7B-Instruct as the base model for continual post-training. The investigation yields two significant findings: (1) When continuously learning on downstream tasks, SFT leads to catastrophic forgetting of previously learned tasks. In contrast, RFT inherently preserves prior knowledge and achieve performance comparable to multi-task training. (2) RFT successfully protects and even enhances the model's general knowledge on standard benchmarks (e.g., MMMU and MMLU-Pro). Conversely, SFT degrades general model capabilities severely. Further analysis reveals that this stability is not primarily due to explicit mechanisms like KL penalty or chain-of-thought reasoning. Instead, we identify an implicit regularization mechanism inherent to RFT as a key contributing factor. Our theoretical analysis suggests that RFT's gradient updates are naturally scaled by the reward variance, acting as a data-dependent regularizer that inherently protects previously acquired knowledge. Finally, we propose a rollout-based instance filtering algorithm to enhance the stability and efficiency of RFT. Our comprehensive study demonstrates the superiority of RFT as a robust paradigm for continual post-training.
>
---
#### [replaced 045] Market-Bench: Evaluating Large Language Models on Introductory Quantitative Trading and Market Dynamics
- **分类: cs.CL**

- **简介: 该论文提出MARKET-BENCH基准，评估大语言模型在量化交易任务中的表现，解决模型生成可执行交易策略的问题，通过代码生成与回测验证模型能力。**

- **链接: [https://arxiv.org/pdf/2512.12264v2](https://arxiv.org/pdf/2512.12264v2)**

> **作者:** Abhay Srivastava; Sam Jung; Spencer Mateega
>
> **摘要:** We introduce MARKET-BENCH, a benchmark that evaluates large language models (LLMs) on introductory quantitative trading tasks by asking them to construct executable backtesters from natural language strategy descriptions and market assumptions. Each instance specifies one of three canonical strategies: scheduled trading on Microsoft (NASDAQ: MSFT), pairs trading on Coca-Cola (NASDAQ: KO) and Pepsi (NASDAQ: PEP), or delta hedging on MSFT. Models must produce code whose profit and loss (P and L), drawdown, and position paths match a verifiable reference implementation. We assess thirteen state-of-the-art models using a multi-round evaluation that separates structural reliability (whether the backtest runs) from numerical accuracy (mean absolute error of the backtest metrics), assigning failed outputs a duplicated-metrics baseline MAE. While most models reliably execute the simplest strategy (average executable passes of 4.08 out of 5 rounds), errors vary by orders of magnitude across models and tasks. Gemini 3 Pro and Claude 4.5 Sonnet combine strong reliability with low error on simpler strategies. GPT-5.2 achieves strong overall performance with perfect executability. GPT-5.1 Codex-Max achieves the lowest best-run error on the easiest task. Qwen3 Max attains perfect executability yet sometimes produces inaccurate profit and loss paths. These results show that current LLMs can scaffold basic trading infrastructure but still struggle to reason robustly about prices, inventory, and risk. We release MARKET-BENCH and a public leaderboard at https://marketbench.ai.
>
---
#### [replaced 046] Identifying Reliable Evaluation Metrics for Scientific Text Revision
- **分类: cs.CL**

- **简介: 该论文属于科学文本修订评估任务，旨在解决传统指标无法准确衡量文本改进的问题。通过人工标注、参考无关指标和LLM评估，提出混合方法提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2506.04772v4](https://arxiv.org/pdf/2506.04772v4)**

> **作者:** Léane Jourdan; Florian Boudin; Richard Dufour; Nicolas Hernandez
>
> **备注:** V3 contains the English version, accepted to ACL 2025 main (26 pages). V4 contains the French version (TALN 2025, 32 pages) with corrected results for cramer's v and pairwise accuracy
>
> **摘要:** Evaluating text revision in scientific writing remains a challenge, as traditional metrics such as ROUGE and BERTScore primarily focus on similarity rather than capturing meaningful improvements. In this work, we analyse and identify the limitations of these metrics and explore alternative evaluation methods that better align with human judgments. We first conduct a manual annotation study to assess the quality of different revisions. Then, we investigate reference-free evaluation metrics from related NLP domains. Additionally, we examine LLM-as-a-judge approaches, analysing their ability to assess revisions with and without a gold reference. Our results show that LLMs effectively assess instruction-following but struggle with correctness, while domain-specific metrics provide complementary insights. We find that a hybrid approach combining LLM-as-a-judge evaluation and task-specific metrics offers the most reliable assessment of revision quality.
>
---
#### [replaced 047] Beyond Single-Granularity Prompts: A Multi-Scale Chain-of-Thought Prompt Learning for Graph
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图神经网络任务，旨在解决图数据中单粒度提示信息不足的问题。提出多尺度链式思维提示框架MSGCOT，融合多尺度结构信息提升提示多样性与效果。**

- **链接: [https://arxiv.org/pdf/2510.09394v3](https://arxiv.org/pdf/2510.09394v3)**

> **作者:** Ziyu Zheng; Yaming Yang; Ziyu Guan; Wei Zhao; Xinyan Huang; Weigang Lu
>
> **备注:** Accepted by WWW2026
>
> **摘要:** The ``pre-train, prompt" paradigm, designed to bridge the gap between pre-training tasks and downstream objectives, has been extended from the NLP domain to the graph domain and has achieved remarkable progress. Current mainstream graph prompt-tuning methods modify input or output features using learnable prompt vectors. However, existing approaches are confined to single-granularity (e.g., node-level or subgraph-level) during prompt generation, overlooking the inherently multi-scale structural information in graph data, which limits the diversity of prompt semantics. To address this issue, we pioneer the integration of multi-scale information into graph prompt and propose a Multi-Scale Graph Chain-of-Thought (MSGCOT) prompting framework. Specifically, we design a lightweight, low-rank coarsening network to efficiently capture multi-scale structural features as hierarchical basis vectors for prompt generation. Subsequently, mimicking human cognition from coarse-to-fine granularity, we dynamically integrate multi-scale information at each reasoning step, forming a progressive coarse-to-fine prompt chain. Extensive experiments on eight benchmark datasets demonstrate that MSGCOT outperforms the state-of-the-art single-granularity graph prompt-tuning method, particularly in few-shot scenarios, showcasing superior performance. The code is available at: https://github.com/zhengziyu77/MSGCOT.
>
---
#### [replaced 048] Autiverse: Eliciting Autistic Adolescents' Daily Narratives through AI-guided Multimodal Journaling
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于辅助叙事训练任务，旨在帮助自闭症青少年提升叙事能力。通过AI引导的多模态日记应用，解决传统文字日记的困难，支持其组织日常经历与情感。**

- **链接: [https://arxiv.org/pdf/2509.17466v2](https://arxiv.org/pdf/2509.17466v2)**

> **作者:** Migyeong Yang; Kyungah Lee; Jinyoung Han; SoHyun Park; Young-Ho Kim
>
> **备注:** 19 pages excluding reference. Conditionally accepted to ACM CHI 2026
>
> **摘要:** Journaling can potentially serve as an effective method for autistic adolescents to improve narrative skills. However, its text-centric nature and high executive functioning demands present barriers to practice. We present Autiverse, an AI-guided multimodal journaling app for tablets that scaffolds storytelling through conversational prompts and visual supports. Autiverse elicits key details through a stepwise dialogue with peer-like, customizable AI and composes them into an editable four-panel comic strip. Through a two-week deployment study with 10 autistic adolescent-parent dyads, we examine how Autiverse supports autistic adolescents to organize their daily experience and emotion. Autiverse scaffolded adolescents' coherent narratives, while enabling parents to learn additional details of their child's events and emotions. The customized AI peer created a comfortable space for sharing, fostering enjoyment and a strong sense of agency. We discuss implications for adaptive scaffolding across autism profiles, socio-emotionally appropriate AI peer design, and balancing autonomy with parental involvement.
>
---
#### [replaced 049] H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型对齐任务，旨在解决LLM响应需同时具备帮助性、无害性和诚实性的难题。提出H3Fusion方法，通过混合专家机制提升模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2411.17792v4](https://arxiv.org/pdf/2411.17792v4)**

> **作者:** Selim Furkan Tekin; Fatih Ilhan; Tiansheng Huang; Sihao Hu; Yichang Xu; Zachary Yahn; Ling Liu
>
> **摘要:** The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/git-disl/h3fusion.
>
---
#### [replaced 050] How Reliable are Confidence Estimators for Large Reasoning Models? A Systematic Benchmark on High-Stakes Domains
- **分类: cs.CL**

- **简介: 该论文属于模型置信度估计任务，旨在解决大模型在高风险领域中的可靠性问题。通过构建基准数据集，评估多种方法的准确性与校准性，揭示性能瓶颈。**

- **链接: [https://arxiv.org/pdf/2601.08134v2](https://arxiv.org/pdf/2601.08134v2)**

> **作者:** Reza Khanmohammadi; Erfan Miahi; Simerjot Kaur; Ivan Brugere; Charese H. Smiley; Kundan Thind; Mohammad M. Ghassemi
>
> **备注:** Accepted to the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026) main conference
>
> **摘要:** The miscalibration of Large Reasoning Models (LRMs) undermines their reliability in high-stakes domains, necessitating methods to accurately estimate the confidence of their long-form, multi-step outputs. To address this gap, we introduce the Reasoning Model Confidence estimation Benchmark (RMCB), a public resource of 347,496 reasoning traces from six popular LRMs across different architectural families. The benchmark is constructed from a diverse suite of datasets spanning high-stakes domains, including clinical, financial, legal, and mathematical reasoning, alongside complex general reasoning benchmarks, with correctness annotations provided for all samples. Using RMCB, we conduct a large-scale empirical evaluation of over ten distinct representation-based methods, spanning sequential, graph-based, and text-based architectures. Our central finding is a persistent trade-off between discrimination (AUROC) and calibration (ECE): text-based encoders achieve the best AUROC (0.672), while structurally-aware models yield the best ECE (0.148), with no single method dominating both. Furthermore, we find that increased architectural complexity does not reliably outperform simpler sequential baselines, suggesting a performance ceiling for methods relying solely on chunk-level hidden states. This work provides the most comprehensive benchmark for this task to date, establishing rigorous baselines and demonstrating the limitations of current representation-based paradigms.
>
---
#### [replaced 051] StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs
- **分类: cs.CL**

- **简介: 该论文属于长文档摘要任务，旨在提升LLMs在零样本场景下的摘要质量与事实一致性。提出StrucSum框架，通过图结构增强模型推理能力。**

- **链接: [https://arxiv.org/pdf/2505.22950v2](https://arxiv.org/pdf/2505.22950v2)**

> **作者:** Haohan Yuan; Sukhwa Hong; Haopeng Zhang
>
> **备注:** 14 pages. Accepted by the findings of EACL 2026
>
> **摘要:** Large language models (LLMs) have shown strong performance in zero-shot summarization, but often struggle to model document structure and identify salient information in long texts. In this work, we introduce StrucSum, a training-free prompting framework that enhances LLM reasoning through sentence-level graph structures. StrucSum injects structural signals into prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local context, Centrality-Aware Prompting (CAP) for importance estimation, and Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves both summary quality and factual consistency over unsupervised baselines and vanilla prompting. In particular, on ArXiv, it increases FactCC and SummaC by 19.2\% and 8.0\% points, demonstrating stronger alignment between summaries and source content. The ablation study shows that the combination of multiple strategies does not yield clear performance gains; therefore, structure-aware prompting with graph-based information represents a promising and underexplored direction for the advancement of zero-shot extractive summarization with LLMs. Our source code is publicly available.
>
---
#### [replaced 052] Conjugate Relation Modeling for Few-Shot Knowledge Graph Completion
- **分类: cs.CL**

- **简介: 该论文属于知识图谱补全任务，解决少样本下的三元组缺失问题。提出CR-FKGC框架，通过关系建模和流形空间推理提升性能。**

- **链接: [https://arxiv.org/pdf/2510.22656v2](https://arxiv.org/pdf/2510.22656v2)**

> **作者:** Zilong Wang; Qingtian Zeng; Hua Duan; Cheng Cheng; Minghao Zou; Ziyang Wang
>
> **摘要:** Few-shot Knowledge Graph Completion (FKGC) infers missing triples from limited support samples, tackling long-tail distribution challenges. Existing methods, however, struggle to capture complex relational patterns and mitigate data sparsity. To address these challenges, we propose a novel FKGC framework for conjugate relation modeling (CR-FKGC). Specifically, it employs a neighborhood aggregation encoder to integrate higher-order neighbor information, a conjugate relation learner combining an implicit conditional diffusion relation module with a stable relation module to capture stable semantics and uncertainty offsets, and a manifold conjugate decoder for efficient evaluation and inference of missing triples in manifold space. Experiments on three benchmarks demonstrate that our method achieves superior performance over state-of-the-art methods.
>
---
#### [replaced 053] PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs
- **分类: cs.CL**

- **简介: 该论文提出PTEB，一种动态文本嵌入评估方法，通过随机改写测试提升模型鲁棒性。解决静态基准导致的评估偏差问题，利用LLM生成语义不变的改写句，验证模型在不同词空间下的表现。**

- **链接: [https://arxiv.org/pdf/2510.06730v2](https://arxiv.org/pdf/2510.06730v2)**

> **作者:** Manuel Frank; Haithem Afli
>
> **摘要:** Current sentence embedding evaluations typically rely on static test beds like the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported scores and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in gold ratings and human validation, we show that LLMs generate token-diverse but semantically preserving paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs spanning 20 datasets and 25 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute.
>
---
#### [replaced 054] KBE-DME: Dynamic Multimodal Evaluation via Knowledge Enhanced Benchmark Evolution
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型评估任务，解决静态基准数据污染和饱和问题。提出KBE框架，通过知识增强实现动态基准演化，提升评估可靠性与全面性。**

- **链接: [https://arxiv.org/pdf/2510.21182v2](https://arxiv.org/pdf/2510.21182v2)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xiaojun Wan
>
> **摘要:** The rapid progress of multimodal large language models (MLLMs) calls for more reliable evaluation protocols. Existing static benchmarks suffer from the potential risk of data contamination and saturation, leading to inflated or misleading performance evaluations. To address these issues, we first apply Graph formulation to represent a static or dynamic VQA sample. With the formulation, we propose Knowledge-enhanced Benchmark Evolution(KBE), a dynamic multimodal evaluation framework. KBE first analyzes the original static benchmark, then expands it by integrating multimodal knowledge, transforming the static benchmark into a controllable, dynamic evolving version. Crucially, KBE can both reconstruct questions by Re-selecting visual information in the original image and expand existing questions with external textual knowledge. It enables difficulty-controllable evaluation by adjusting the degree of question exploration. Extensive experiments demonstrate that KBE alleviates the risk of data contamination, data saturation, and provides a more comprehensive assessment of MLLM capabilities.
>
---
#### [replaced 055] Decision-Making with Deliberation: Meta-reviewing as a Document-grounded Dialogue
- **分类: cs.CL**

- **简介: 该论文属于元评审任务，旨在提升元评审效率。通过构建对话代理，解决数据不足问题，并验证其在实际场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2508.05283v2](https://arxiv.org/pdf/2508.05283v2)**

> **作者:** Sukannya Purkayastha; Nils Dycke; Anne Lauscher; Iryna Gurevych
>
> **备注:** Accepted at EACL Main Conference, 2026
>
> **摘要:** Meta-reviewing is a pivotal stage in the peer-review process, serving as the final step in determining whether a paper is recommended for acceptance. Prior research on meta-reviewing has treated this as a summarization problem over review reports. However, complementary to this perspective, meta-reviewing is a decision-making process that requires weighing reviewer arguments and placing them within a broader context. Prior research has demonstrated that decision-makers can be effectively assisted in such scenarios via dialogue agents. In line with this framing, we explore the practical challenges for realizing dialog agents that can effectively assist meta-reviewers. Concretely, we first address the issue of data scarcity for training dialogue agents by generating synthetic data using Large Language Models (LLMs) based on a self-refinement strategy to improve the relevance of these dialogues to expert domains. Our experiments demonstrate that this method produces higher-quality synthetic data and can serve as a valuable resource towards training meta-reviewing assistants. Subsequently, we utilize this data to train dialogue agents tailored for meta-reviewing and find that these agents outperform \emph{off-the-shelf} LLM-based assistants for this task. Finally, we apply our agents in real-world meta-reviewing scenarios and confirm their effectiveness in enhancing the efficiency of meta-reviewing.\footnote{Code available at: https://github.com/UKPLab/eacl2026-meta-review-as-dialog
>
---
#### [replaced 056] BEST-RQ-Based Self-Supervised Learning for Whisper Domain Adaptation
- **分类: cs.CL**

- **简介: 该论文属于语音识别领域的领域自适应任务，旨在解决低资源场景下标注数据不足的问题。通过结合BEST-RQ目标和知识蒸馏，提升Whisper模型在ATC领域的性能。**

- **链接: [https://arxiv.org/pdf/2510.24570v2](https://arxiv.org/pdf/2510.24570v2)**

> **作者:** Raphaël Bagat; Irina Illina; Emmanuel Vincent
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Automatic Speech Recognition (ASR) systems, despite large multilingual training, struggle in low-resource scenarios where labeled data is scarce. We propose BEARD (BEST-RQ Encoder Adaptation with Re-training and Distillation), a novel framework designed to adapt Whisper's encoder with unlabeled data. Unlike traditional self-supervised learning methods, BEARD uniquely combines a BEST-RQ objective with knowledge distillation from a frozen teacher encoder, ensuring the encoder's complementarity with the pre-trained decoder. Our experiments focus on the ATCO2 corpus from the challenging Air Traffic Control (ATC) communications domain, characterized by non-native speech, noise, and specialized phraseology. Using about 5,000 hours of untranscribed speech for BEARD and 2 hours of transcribed speech for fine-tuning, the proposed approach significantly outperforms previous baseline and fine-tuned model, achieving a relative improvement of 12% compared to the fine-tuned model. To the best of our knowledge, this is the first work to use a self-supervised learning objective for domain adaptation of Whisper.
>
---
#### [replaced 057] Context Parametrization with Compositional Adapters
- **分类: cs.CL**

- **简介: 该论文提出CompAs框架，解决大语言模型在处理多任务时效率与灵活性不足的问题。通过生成可组合的适配器参数，提升推理效率和长上下文稳定性。**

- **链接: [https://arxiv.org/pdf/2509.22158v2](https://arxiv.org/pdf/2509.22158v2)**

> **作者:** Josip Jukić; Martin Tutek; Jan Šnajder
>
> **摘要:** Large language models (LLMs) often seamlessly adapt to new tasks through in-context learning (ICL) or supervised fine-tuning (SFT). However, both of these approaches face key limitations: ICL is inefficient when handling many demonstrations, and SFT incurs training overhead while sacrificing flexibility. Mapping instructions or demonstrations from context directly into adapter parameters offers an appealing alternative. While prior work explored generating adapters based on a single input context, it has overlooked the need to integrate multiple chunks of information. To address this gap, we introduce CompAs, a meta-learning framework that translates context into adapter parameters with a compositional structure. Adapters generated this way can be merged algebraically, enabling instructions, demonstrations, or retrieved passages to be seamlessly combined without reprocessing long prompts. Critically, this approach yields three benefits: lower inference cost, robustness to long-context instability, and establishes a principled solution when input exceeds the model's context window. Furthermore, CompAs encodes information into adapter parameters in a reversible manner, enabling recovery of input context through a decoder, facilitating safety and security. Empirical results on diverse multiple-choice and extractive question answering tasks show that CompAs outperforms ICL and prior generator-based methods, especially when scaling to more inputs. Our work establishes composable adapter generation as a practical and efficient alternative for scaling LLM deployment.
>
---
#### [replaced 058] A Two-Stage GPU Kernel Tuner Combining Semantic Refactoring and Search-Based Optimization
- **分类: cs.CL**

- **简介: 该论文属于GPU代码优化任务，旨在解决手动调优效率低、效果不稳定的问题。通过结合语义重构与搜索优化，提升代码性能。**

- **链接: [https://arxiv.org/pdf/2601.12698v2](https://arxiv.org/pdf/2601.12698v2)**

> **作者:** Qiuyi Qu; Yicheng Sui; Yufei Sun; Rui Chen; Xiaofei Zhang; Yuzhi Zhang; Haofeng Wang; Ge Lan; Ning Zhang
>
> **摘要:** GPU code optimization is a key performance bottleneck for HPC workloads as well as large-model training and inference. Although compiler optimizations and hand-written kernels can partially alleviate this issue, achieving near-hardware-limit performance still relies heavily on manual code refactoring and parameter tuning. Recent progress in LLM-agent-based kernel generation and optimization has been reported, yet many approaches primarily focus on direct code rewriting, where parameter choices are often implicit and hard to control, or require human intervention, leading to unstable performance gains. This paper introduces a template-based rewriting layer on top of an agent-driven iterative loop: kernels are semantically refactored into explicitly parameterizable templates, and template parameters are then optimized via search-based autotuning, yielding more stable and higher-quality speedups. Experiments on a set of real-world kernels demonstrate speedups exceeding 3x in the best case. We extract representative CUDA kernels from SGLang as evaluation targets; the proposed agentic tuner iteratively performs templating, testing, analysis, and planning, and leverages profiling feedback to execute constrained parameter search under hardware resource limits. Compared to agent-only direct rewriting, the template-plus-search design significantly reduces the randomness of iterative optimization, making the process more interpretable and enabling a more systematic approach toward high-performance configurations. The proposed method can be further extended to OpenCL, HIP, and other backends to deliver automated performance optimization for real production workloads.
>
---
#### [replaced 059] TextMineX: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action
- **分类: cs.CL; cs.AI**

- **简介: 论文提出TextMineX，解决HMA领域文本知识提取问题。构建数据集与评估框架，利用本体引导的LLM管道提升信息准确性与格式规范性。**

- **链接: [https://arxiv.org/pdf/2509.15098v3](https://arxiv.org/pdf/2509.15098v3)**

> **作者:** Chenyue Zhou; Gürkan Solmaz; Flavio Cirillo; Kiril Gashteovski; Jonathan Fürst
>
> **摘要:** Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMineX: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction from text in the HMA domain. TextMineX structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we utilized the dataset from our collaborator Cambodian Mine Action Centre (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
>
---
#### [replaced 060] Reinforcement Learning for Chain of Thought Compression with One-Domain-to-All Generalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决大模型推理中“过度思考”问题。通过强化学习压缩思维链，在保持准确率前提下减少响应长度，并实现跨领域泛化。**

- **链接: [https://arxiv.org/pdf/2601.06052v2](https://arxiv.org/pdf/2601.06052v2)**

> **作者:** Hanyu Li; Jiangshan Duo; Bofei Gao; Hailin Zhang; Sujian Li; Xiaotie Deng; Liang Zhao
>
> **摘要:** Chain-of-thought reasoning in large language models can trigger an "overthinking trap": longer rollouts raise cost and latency yet often yield unreliable accuracy gains. Existing methods use global, static controls that may suppress needed reasoning. We propose mastery-gated, sample-level, soft reinforcement learning compression that penalizes long rollouts only when the model already solves the problem and has produced a shorter rollout. Across benchmarks, it cuts response length by 20-40% with comparable or higher accuracy and generalizes across domains: a model trained on math spontaneously shortens unseen tasks (code, instruction following, general-knowledge QA) without hurting accuracy. We further show two-way transfer between non-agent CoT and tool-use agents: non-agent training reduces SWE-Bench Verified rounds by 13%, while compressing a thinking agent cuts SWE trajectories by 67% tokens and 52% rounds and shortens non-agent outputs by up to 44%. Compression is thus not cosmetic brevity, but an inherent computation policy -- what to keep, and what to forget.
>
---
#### [replaced 061] Competitive Audio-Language Models with Data-Efficient Single-Stage Training on Public Data
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Falcon3-Audio，一种高效音频-语言模型，解决音频与语言融合不足的问题。使用少量公开数据，实现高性能，强调数据效率与简单训练流程。**

- **链接: [https://arxiv.org/pdf/2509.07526v2](https://arxiv.org/pdf/2509.07526v2)**

> **作者:** Gokul Karthik Kumar; Rishabh Saraf; Ludovick Lepauloux; Abdul Muneer; Billel Mokeddem; Hakim Hacid
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** Large language models (LLMs) have transformed NLP, yet their integration with audio remains underexplored despite audio's centrality to human communication. We introduce Falcon3-Audio, a family of Audio-Language Models (ALMs) built on instruction-tuned LLMs and Whisper encoders. Using a remarkably small amount of public audio data, less than 30K hours (5K unique), Falcon3-Audio-7B matches the best reported performance among open-weight models on the MMAU benchmark, with a score of 64.14, matching R1-AQA, while distinguishing itself through superior data and parameter efficiency, single-stage training, and transparency. Notably, our smallest 1B model remains competitive with larger open models ranging from 2B to 13B parameters. Through extensive ablations, we find that common complexities such as curriculum learning, multiple audio encoders, and intricate cross-attention connectors are not required for strong performance, even compared to models trained on over 500K hours of data.
>
---
#### [replaced 062] Multimodal Multi-Agent Empowered Legal Judgment Prediction
- **分类: cs.CL**

- **简介: 该论文属于法律判决预测任务，旨在解决传统方法在处理复杂案件时的不足。提出JurisMMA框架，构建大规模多模态数据集，提升预测效果与适用性。**

- **链接: [https://arxiv.org/pdf/2601.12815v2](https://arxiv.org/pdf/2601.12815v2)**

> **作者:** Zhaolu Kang; Junhao Gong; Qingxi Chen; Hao Zhang; Jiaxin Liu; Rong Fu; Zhiyuan Feng; Yuan Wang; Simon Fong; Kaiyue Zhou
>
> **备注:** Accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets.
>
---
#### [replaced 063] Representation-Aware Unlearning via Activation Signatures: From Suppression to Knowledge-Signature Erasure
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识擦除任务，解决LLM中行为抑制与真实知识删除混淆的问题。提出KIF框架，通过激活签名实现真正知识擦除，提升合规性与安全性。**

- **链接: [https://arxiv.org/pdf/2601.10566v2](https://arxiv.org/pdf/2601.10566v2)**

> **作者:** Syed Naveed Mahmood; Md. Rezaur Rahman Bhuiyan; Tasfia Zaman; Jareen Tasneem Khondaker; Md. Sameer Sakib; K. M. Shadman Wadith; Nazia Tasnim; Farig Sadeque
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Selective knowledge erasure from LLMs is critical for GDPR compliance and model safety, yet current unlearning methods conflate behavioral suppression with true knowledge removal, allowing latent capabilities to persist beneath surface-level refusals. In this work, we address this challenge by introducing Knowledge Immunization Framework (KIF), a representation-aware architecture that distinguishes genuine erasure from obfuscation by targeting internal activation signatures rather than surface outputs. Our approach combines dynamic suppression of subject-specific representations with parameter-efficient adaptation, enabling durable unlearning without full model retraining. KIF achieves near-oracle erasure (FQ approx 0.99 vs. 1.00) while preserving utility at oracle levels (MU = 0.62), effectively breaking the stability-erasure tradeoff that has constrained all prior work. We evaluate both standard foundation models (Llama and Mistral) and reasoning-prior models (Qwen and DeepSeek) across 3B to 14B parameters. Our observation shows that standard models exhibit scale-independent true erasure (<3% utility drift), while reasoning-prior models reveal fundamental architectural divergence. Our comprehensive dual-metric evaluation protocol, combining surface-level leakage with latent trace persistence, operationalizes the obfuscation - erasure distinction and enables the first systematic diagnosis of mechanism-level forgetting behavior across model families and scales.
>
---
#### [replaced 064] Token Maturation: Autoregressive Language Generation via Continuous Token Dynamics
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决传统模型因过早离散化导致的重复和依赖采样策略问题。提出Token Maturation框架，通过连续向量轨迹生成文本，延迟离散化以提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.04854v2](https://arxiv.org/pdf/2601.04854v2)**

> **作者:** Oshri Naparstek
>
> **备注:** In preperation to ICML 2026
>
> **摘要:** Standard autoregressive language models collapse uncertainty at every generation step by committing to discrete tokens through immediate sampling. This premature discretization underlies well-known failure modes, including degenerate repetition loops in greedy decoding and a heavy reliance on heuristic sampling strategies. We introduce \textbf{Token Maturation}, a continuous autoregressive framework in which tokens evolve as vector-valued trajectories prior to discretization. Rather than sampling from a categorical distribution at each step, the model resolves uncertainty through a deterministic dynamical process in embedding space, deferring discrete commitment until the representation has geometrically stabilized. We show that this formulation mitigates degeneration \emph{intrinsically}: Token Maturation generates coherent and diverse text under fully deterministic decoding (argmax), without repetition penalties, temperature scaling, or stochastic sampling. Moreover, we identify a novel convergence behavior in which token representations stabilize spatially while predictive entropy remains high, challenging the common assumption that commitment requires probability concentration. We propose continuous token dynamics with delayed commitment as an alternative formulation of autoregressive generation that exposes structural regularities obscured by immediate discretization.
>
---
#### [replaced 065] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking
- **分类: cs.CL**

- **简介: 该论文提出AStar方法，解决多模态推理任务中计算效率低和依赖大量训练的问题，通过引入“思维卡片”提升推理能力。**

- **链接: [https://arxiv.org/pdf/2502.02339v4](https://arxiv.org/pdf/2502.02339v4)**

> **作者:** Jinyang Wu; Mingkuan Feng; Guocheng Zhai; Shuai Zhang; Zheng Lian; Fangrui Lv; Pengpeng Shao; Ruihan Jin; Zhengqi Wen; Jianhua Tao
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Multimodal large language models excel across diverse domains but struggle with complex visual reasoning tasks. To enhance their reasoning capabilities, current approaches typically rely on explicit search or post-training techniques. However, search-based methods suffer from computational inefficiency due to extensive solution space exploration, while post-training methods demand substantial data, computational resources, and often exhibit training instability. To address these challenges, we propose \textbf{AStar}, a training-free, \textbf{A}utomatic \textbf{S}tructured \textbf{t}hinking paradigm for multimod\textbf{a}l \textbf{r}easoning. Specifically, we introduce novel ``thought cards'', a lightweight library of high-level reasoning patterns abstracted from prior samples. For each test problem, AStar adaptively retrieves the optimal thought cards and seamlessly integrates these external explicit guidelines with the model's internal implicit reasoning capabilities. Compared to previous methods, AStar eliminates computationally expensive explicit search and avoids additional complex post-training processes, enabling a more efficient reasoning approach. Extensive experiments demonstrate that our framework achieves 53.9\% accuracy on MathVerse (surpassing GPT-4o's 50.2\%) and 32.7\% on MathVision (outperforming GPT-4o's 30.4\%). Further analysis reveals the remarkable transferability of our method: thought cards generated from mathematical reasoning can also be applied to other reasoning tasks, even benefiting general visual perception and understanding. AStar serves as a plug-and-play test-time inference method, compatible with other post-training techniques, providing an important complement to existing multimodal reasoning approaches.
>
---
#### [replaced 066] Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型指纹识别任务，旨在解决黑盒环境下LLM指纹难以区分的问题。通过零阶梯度估计和语义替换，提出ZeroPrint方法，提升指纹的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.06605v2](https://arxiv.org/pdf/2510.06605v2)**

> **作者:** Shuo Shao; Yiming Li; Hongwei Yao; Yifei Chen; Yuchen Yang; Zhan Qin
>
> **备注:** This paper is accepeted by the ACM Web Conference (WWW) 2026
>
> **摘要:** The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods.
>
---
#### [replaced 067] A2H-MAS: An Algorithm-to-HLS Multi-Agent System for Automated and Reliable FPGA Implementation
- **分类: cs.CL; cs.AR; cs.PL**

- **简介: 该论文属于FPGA自动化设计任务，旨在解决算法到硬件实现的转化难题。通过A2H-MAS系统，实现高效、可靠的HLS设计。**

- **链接: [https://arxiv.org/pdf/2508.10904v3](https://arxiv.org/pdf/2508.10904v3)**

> **作者:** Jie Lei; Ruofan Jia; J. Andrew Zhang; Hao Zhang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Bridging the gap between algorithm development and hardware realization remains a persistent challenge, particularly in latency- and resource-constrained domains such as wireless communication. While MATLAB provides a mature environment for algorithm prototyping, translating these models into efficient FPGA implementations via High-Level Synthesis (HLS) often requires expert tuning and lengthy iterations. Recent advances in large language models (LLMs) offer new opportunities for automating this process. However, existing approaches suffer from hallucinations, forgetting, limited domain expertise, and often overlook key performance metrics. To address these limitations, we present A2H-MAS, a modular and hierarchical multi-agent system. At the system level, A2H-MAS assigns clearly defined responsibilities to specialized agents and uses standardized interfaces and execution-based validation to ensure correctness and reproducibility. At the algorithmic level, it employs dataflow-oriented modular decomposition and algorithm-hardware co-design, recognizing that the choice of algorithm often has a larger impact on hardware efficiency than pragma-level optimization. Experiments on representative wireless communication algorithms show that A2H-MAS consistently produces functionally correct, resource-efficient, and latency-optimized HLS designs, demonstrating its effectiveness and robustness for complex hardware development workflows.
>
---
#### [replaced 068] PankRAG: Enhancing Graph Retrieval via Globally Aware Query Resolution and Dependency-Aware Reranking Mechanism
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于知识图谱检索任务，旨在解决传统方法因仅依赖实体提取而遗漏关键关系的问题。提出PankRAG框架，结合全局解析路径和依赖重排序机制，提升检索准确性。**

- **链接: [https://arxiv.org/pdf/2506.11106v2](https://arxiv.org/pdf/2506.11106v2)**

> **作者:** Ningyuan Li; Junrui Liu; Yi Shan; Minghui Huang; Ziren Gong; Tong Li
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Recent graph-based RAG approaches leverage knowledge graphs by extracting entities from a query to fetch their associated relationships and metadata. However, relying solely on entity extraction often results in the misinterpretation or omission of latent critical information and relationships. This can lead to the retrieval of irrelevant or contradictory content, as well as the exclusion of essential information, thereby increasing hallucination risks and undermining the quality of generated responses. In this paper, we propose PankRAG, a framework designed to capture and resolve the latent relationships within complex queries that prior methods overlook. It achieves this through a synergistic combination of a globally-aware hierarchical resolution pathway and a dependency-aware reranking mechanism. PankRAG first generates a globally aware resolution pathway that captures parallel and progress relationships, guiding LLMs to resolve queries through a hierarchical reasoning path. Additionally, its dependency-aware reranking mechanism utilizes resolved sub-question dependencies to augment and validate the retrieved content of the current unresolved sub-question. Experimental results demonstrate that PankRAG consistently outperforms existing state-of-the-art methods, underscoring its generalizability.
>
---
#### [replaced 069] RiskCueBench: Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出RiskCueBench基准，用于评估视频语言模型从早期风险线索中进行预见性推理的能力，旨在解决实时风险预测中的挑战。**

- **链接: [https://arxiv.org/pdf/2601.03369v2](https://arxiv.org/pdf/2601.03369v2)**

> **作者:** Sha Luo; Yogesh Prabhu; Timothy Ossowski; Kaiping Chen; Junjie Hu
>
> **备注:** *updated author email in this version
>
> **摘要:** With the rapid growth of video centered social media, the ability to anticipate risky events from visual data is a promising direction for ensuring public safety and preventing real world accidents. Prior work has extensively studied supervised video risk assessment across domains such as driving, protests, and natural disasters. However, many existing datasets provide models with access to the full video sequence, including the accident itself, which substantially reduces the difficulty of the task. To better reflect real world conditions, we introduce a new video understanding benchmark RiskCueBench in which videos are carefully annotated to identify a risk signal clip, defined as the earliest moment that indicates a potential safety concern. Experimental results reveal a significant gap in current systems ability to interpret evolving situations and anticipate future risky events from early visual signals, highlighting important challenges for deploying video risk prediction models in practice.
>
---
#### [replaced 070] Memp: Exploring Agent Procedural Memory
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文研究如何构建可学习、可更新的程序记忆，解决LLM代理记忆脆弱的问题。提出Memp框架，提升任务执行效率与成功率。**

- **链接: [https://arxiv.org/pdf/2508.06433v3](https://arxiv.org/pdf/2508.06433v3)**

> **作者:** Runnan Fang; Yuan Liang; Xiaobin Wang; Jialong Wu; Shuofei Qiao; Pengjun Xie; Fei Huang; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) based agents excel at diverse tasks, yet they suffer from brittle procedural memory that is manually engineered or entangled in static parameters. In this work, we investigate strategies to endow agents with a learnable, updatable, and lifelong procedural memory. We propose Memp that distills past agent trajectories into both fine-grained, step-by-step instructions and higher-level, script-like abstractions, and explore the impact of different strategies for Build, Retrieval, and Update of procedural memory. Coupled with a dynamic regimen that continuously updates, corrects, and deprecates its contents, this repository evolves in lockstep with new experience. Empirical evaluation on TravelPlanner and ALFWorld shows that as the memory repository is refined, agents achieve steadily higher success rates and greater efficiency on analogous tasks. Moreover, procedural memory built from a stronger model retains its value: migrating the procedural memory to a weaker model can also yield substantial performance gains. Code is available at https://github.com/zjunlp/MemP.
>
---
