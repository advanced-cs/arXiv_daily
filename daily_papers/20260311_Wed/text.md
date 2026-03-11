# 自然语言处理 cs.CL

- **最新发布 66 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] ConFu: Contemplate the Future for Better Speculative Sampling
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理加速任务，旨在解决草案模型预测偏差问题。提出ConFu框架，通过未来感知机制提升预测准确性，提高生成速度与接受率。**

- **链接: [https://arxiv.org/pdf/2603.08899](https://arxiv.org/pdf/2603.08899)**

> **作者:** Zongyue Qin; Raghavv Goel; Mukul Gagrani; Risheek Garrepalli; Mingu Lee; Yizhou Sun
>
> **备注:** accepted at ICLR 2026 workshop on Latent & Implicit Thinking - Going Beyond CoT Reasoning
>
> **摘要:** Speculative decoding has emerged as a powerful approach to accelerate large language model (LLM) inference by employing lightweight draft models to propose candidate tokens that are subsequently verified by the target model. The effectiveness of this paradigm critically depends on the quality of the draft model. While recent advances such as the EAGLE series achieve state-of-the-art speedup, existing draft models remain limited by error accumulation: they condition only on the current prefix, causing their predictions to drift from the target model over steps. In this work, we propose \textbf{ConFu} (Contemplate the Future), a novel speculative decoding framework that enables draft models to anticipate the future direction of generation. ConFu introduces (i) contemplate tokens and soft prompts that allow the draft model to leverage future-oriented signals from the target model at negligible cost, (ii) a dynamic contemplate token mechanism with MoE to enable context-aware future prediction, and (iii) a training framework with anchor token sampling and future prediction replication that learns robust future prediction. Experiments demonstrate that ConFu improves token acceptance rates and generation speed over EAGLE-3 by 8--11% across various downstream tasks with Llama-3 3B and 8B models. We believe our work is the first to bridge speculative decoding with continuous reasoning tokens, offering a new direction for accelerating LLM inference.
>
---
#### [new 002] EPIC-EuroParl-UdS: Information-Theoretic Perspectives on Translation and Interpreting
- **分类: cs.CL**

- **简介: 该论文介绍了一个更新的双语语料库，用于信息论视角下的翻译与口译研究，解决语料不准确和缺乏标注的问题，通过修正错误、添加新层数据支持语言变异分析及翻译研究。**

- **链接: [https://arxiv.org/pdf/2603.09785](https://arxiv.org/pdf/2603.09785)**

> **作者:** Maria Kunilovskaya; Christina Pollkläsener
>
> **备注:** 16 pages with appendices, 8 figures to be published in LREC-2026 main conference proceedings
>
> **摘要:** This paper introduces an updated and combined version of the bidirectional English-German EPIC-UdS (spoken) and EuroParl-UdS (written) corpora containing original European Parliament speeches as well as their translations and interpretations. The new version corrects metadata and text errors identified through previous use, refines the content, updates linguistic annotations, and adds new layers, including word alignment and word-level surprisal indices. The combined resource is designed to support research using information-theoretic approaches to language variation, particularly studies comparing written and spoken modes, and examining disfluencies in speech, as well as traditional translationese studies, including parallel (source vs. target) and comparable (original vs. translated) analyses. The paper outlines the updates introduced in this release, summarises previous results based on the corpus, and presents a new illustrative study. The study validates the integrity of the rebuilt spoken data and evaluates probabilistic measures derived from base and fine-tuned GPT-2 and machine translation models on the task of filler particles prediction in interpreting.
>
---
#### [new 003] Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理任务，旨在解决长推理路径导致的高成本问题。通过引入自信度感知框架，实现单路径与多路径推理的自适应选择，提升效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2603.08999](https://arxiv.org/pdf/2603.08999)**

> **作者:** Juming Xiong; Kevin Guo; Congning Ni; Chao Yan; Katherine Brown; Avinash Baidya; Xiang Gao; Bradley Marlin; Zhijun Yin
>
> **摘要:** Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating multiple reasoning trajectories, leading to substantial additional computational overhead. This paper introduces a confidence-aware decision framework that analyzes a single completed reasoning trajectory to adaptively select between single-path and multi-path reasoning. The framework is trained using sentence-level numeric and linguistic features extracted from intermediate reasoning states in the MedQA dataset and generalizes effectively to MathQA, MedMCQA, and MMLU without additional fine-tuning. Experimental results show that the proposed method maintains accuracy comparable to multi-path baselines while using up to 80\% fewer tokens. These findings demonstrate that reasoning trajectories contain rich signals for uncertainty estimation, enabling a simple, transferable mechanism to balance accuracy and efficiency in LLM reasoning.
>
---
#### [new 004] Modelling the Diachronic Emergence of Phoneme Frequency Distributions
- **分类: cs.CL**

- **简介: 该论文属于语言学建模任务，旨在解释音位频率分布的统计规律。通过构建随机模型模拟语音系统演变，验证了这些规律可能源于历史变化而非优化机制。**

- **链接: [https://arxiv.org/pdf/2603.09503](https://arxiv.org/pdf/2603.09503)**

> **作者:** Fermín Moscoso del Prado Martín; Suchir Salhan
>
> **摘要:** Phoneme frequency distributions exhibit robust statistical regularities across languages, including exponential-tailed rank-frequency patterns and a negative relationship between phonemic inventory size and the relative entropy of the distribution. The origin of these patterns remains largely unexplained. In this paper, we investigate whether they can arise as consequences of the historical processes that shape phonological systems. We introduce a stochastic model of phonological change and simulate the diachronic evolution of phoneme inventories. A naïve version of the model reproduces the general shape of phoneme rank-frequency distributions but fails to capture other empirical properties. Extending the model with two additional assumptions -- an effect related to functional load and a stabilising tendency toward a preferred inventory size -- yields simulations that match both the observed distributions and the negative relationship between inventory size and relative entropy. These results suggest that some statistical regularities of phonological systems may arise as natural consequences of diachronic sound change rather than from explicit optimisation or compensatory mechanisms.
>
---
#### [new 005] Emotion is Not Just a Label: Latent Emotional Factors in LLM Processing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究情感在大语言模型中的潜在影响，属于自然语言处理任务。旨在解决情感对模型推理行为的影响问题，提出AURA-QA数据集和情感正则化方法，提升问答性能。**

- **链接: [https://arxiv.org/pdf/2603.09205](https://arxiv.org/pdf/2603.09205)**

> **作者:** Benjamin Reichman; Adar Avasian; Samuel Webster; Larry Heck
>
> **摘要:** Large language models are routinely deployed on text that varies widely in emotional tone, yet their reasoning behavior is typically evaluated without accounting for emotion as a source of representational variation. Prior work has largely treated emotion as a prediction target, for example in sentiment analysis or emotion classification. In contrast, we study emotion as a latent factor that shapes how models attend to and reason over text. We analyze how emotional tone systematically alters attention geometry in transformer models, showing that metrics such as locality, center-of-mass distance, and entropy vary across emotions and correlate with downstream question-answering performance. To facilitate controlled study of these effects, we introduce Affect-Uniform ReAding QA (AURA-QA), a question-answering dataset with emotionally balanced, human-authored context passages. Finally, an emotional regularization framework is proposed that constrains emotion-conditioned representational drift during training. Experiments across multiple QA benchmarks demonstrate that this approach improves reading comprehension in both emotionally-varying and non-emotionally varying datasets, yielding consistent gains under distribution shift and in-domain improvements on several benchmarks.
>
---
#### [new 006] MultiGraSCCo: A Multilingual Anonymization Benchmark with Annotations of Personal Identifiers
- **分类: cs.CL**

- **简介: 该论文提出MultiGraSCCo基准，解决多语言隐私数据匿名化问题。通过机器翻译生成带标注的合成数据，确保个人信息正确转换，支持安全数据共享与系统测试。**

- **链接: [https://arxiv.org/pdf/2603.08879](https://arxiv.org/pdf/2603.08879)**

> **作者:** Ibrahim Baroud; Christoph Otto; Vera Czehmann; Christine Hovhannisyan; Lisa Raithel; Sebastian Möller; Roland Roller
>
> **摘要:** Accessing sensitive patient data for machine learning is challenging due to privacy concerns. Datasets with annotations of personally identifiable information are crucial for developing and testing anonymization systems to enable safe data sharing that complies with privacy regulations. Since accessing real patient data is a bottleneck, synthetic data offers an efficient solution for data scarcity, bypassing privacy regulations that apply to real data. Moreover, neural machine translation can help to create high-quality data for low-resource languages by translating validated real or synthetic data from a high-resource language. In this work, we create a multilingual anonymization benchmark in ten languages, using a machine translation methodology that preserves the original annotations and renders names of cities and people in a culturally and contextually appropriate form in each target language. Our evaluation study with medical professionals confirms the quality of the translations, both in general and with respect to the translation and adaptation of personal information. Our benchmark with over 2,500 annotations of personal information can be used in many applications, including training annotators, validating annotations across institutions without legal complications, and helping improve the performance of automatic personal information detection. We make our benchmark and annotation guidelines available for further research.
>
---
#### [new 007] Benchmarking Political Persuasion Risks Across Frontier Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI伦理研究任务，旨在评估前沿大语言模型的政治说服风险。通过实验对比不同模型的说服效果，提出一种分析方法以评估其潜在风险。**

- **链接: [https://arxiv.org/pdf/2603.09884](https://arxiv.org/pdf/2603.09884)**

> **作者:** Zhongren Chen; Joshua Kalla; Quan Le
>
> **摘要:** Concerns persist regarding the capacity of Large Language Models (LLMs) to sway political views. Although prior research has claimed that LLMs are not more persuasive than standard political campaign practices, the recent rise of frontier models warrants further study. In two survey experiments (N=19,145) across bipartisan issues and stances, we evaluate seven state-of-the-art LLMs developed by Anthropic, OpenAI, Google, and xAI. We find that LLMs outperform standard campaign advertisements, with heterogeneity in performance across models. Specifically, Claude models exhibit the highest persuasiveness, while Grok exhibits the lowest. The results are robust across issues and stances. Moreover, in contrast to the findings in Hackenburg et al. (2025b) and Lin et al. (2025) that information-based prompts boost persuasiveness, we find that the effectiveness of information-based prompts is model-dependent: they increase the persuasiveness of Claude and Grok while substantially reducing that of GPT. We introduce a data-driven and strategy-agnostic LLM-assisted conversation analysis approach to identify and assess underlying persuasive strategies. Our work benchmarks the persuasive risks of frontier models and provides a framework for cross-model comparative risk assessment.
>
---
#### [new 008] Investigating Gender Stereotypes in Large Language Models via Social Determinants of Health
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的偏见检测任务，旨在解决LLMs在医疗领域可能存在的性别刻板印象问题。通过分析法国患者记录中的社会决定因素，研究LLMs如何依赖刻板印象进行性别判断。**

- **链接: [https://arxiv.org/pdf/2603.09416](https://arxiv.org/pdf/2603.09416)**

> **作者:** Trung Hieu Ngo; Adrien Bazoge; Solen Quiniou; Pierre-Antoine Gourraud; Emmanuel Morin
>
> **备注:** Accepted as Findings at EACL 2026
>
> **摘要:** Large Language Models (LLMs) excel in Natural Language Processing (NLP) tasks, but they often propagate biases embedded in their training data, which is potentially impactful in sensitive domains like healthcare. While existing benchmarks evaluate biases related to individual social determinants of health (SDoH) such as gender or ethnicity, they often overlook interactions between these factors and lack context-specific assessments. This study investigates bias in LLMs by probing the relationships between gender and other SDoH in French patient records. Through a series of experiments, we found that embedded stereotypes can be probed using SDoH input and that LLMs rely on embedded stereotypes to make gendered decisions, suggesting that evaluating interactions among SDoH factors could usefully complement existing approaches to assessing LLM performance and bias.
>
---
#### [new 009] CREATE: Testing LLMs for Associative Creativity
- **分类: cs.CL**

- **简介: 该论文提出CREATE基准，用于评估大语言模型的联想创造力。任务是生成高特异性和多样性的概念连接路径，解决模型创造性推理能力评估问题。**

- **链接: [https://arxiv.org/pdf/2603.09970](https://arxiv.org/pdf/2603.09970)**

> **作者:** Manya Wadhwa; Tiasa Singha Roy; Harvey Lederman; Junyi Jessy Li; Greg Durrett
>
> **摘要:** A key component of creativity is associative reasoning: the ability to draw novel yet meaningful connections between concepts. We introduce CREATE, a benchmark designed to evaluate models' capacity for creative associative reasoning. CREATE requires models to generate sets of paths connecting concepts in a model's parametric knowledge. Paths should have high specificity (distinctiveness and closeness of the concept connection) and high diversity (dissimilarity from other paths), and models are scored more highly if they produce a larger set of strong, diverse paths. This task shares demands of real creativity tasks like hypothesis generation, including an extremely large search space, but enables collection of a sizable benchmark with objective answer grading. Evaluation of frontier models shows that the strongest models achieve higher creative utility than others, with the high multiplicity of answers and complexity of the search making benchmark saturation difficult to achieve. Furthermore, our results illustrate that thinking models are not always more effective on our task, even with high token budgets. Recent approaches for creative prompting give some but limited additional improvement. CREATE provides a sandbox for developing new methods to improve models' capacity for associative creativity.
>
---
#### [new 010] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生成可操作的审稿反馈任务，旨在解决AI生成审稿意见不够具体的问题。通过利用反驳内容作为监督信号，提升反馈的行动性。**

- **链接: [https://arxiv.org/pdf/2603.09723](https://arxiv.org/pdf/2603.09723)**

> **作者:** Sihong Wu; Yiling Ma; Yilun Zhao; Tiansheng Hu; Owen Jiang; Manasi Patwardhan; Arman Cohan
>
> **摘要:** Large language models (LLMs) are increasingly used across the scientific workflow, including to draft peer-review reports. However, many AI-generated reviews are superficial and insufficiently actionable, leaving authors without concrete, implementable guidance and motivating the gap this work addresses. We propose RbtAct, which targets actionable review feedback generation and places existing peer review rebuttal at the center of learning. Rebuttals show which reviewer comments led to concrete revisions or specific plans, and which were only defended. Building on this insight, we leverage rebuttal as implicit supervision to directly optimize a feedback generator for actionability. To support this objective, we propose a new task called perspective-conditioned segment-level review feedback generation, in which the model is required to produce a single focused comment based on the complete paper and a specified perspective such as experiments and writing. We also build a large dataset named RMR-75K that maps review segments to the rebuttal segments that address them, with perspective labels and impact categories that order author uptake. We then train the Llama-3.1-8B-Instruct model with supervised fine-tuning on review segments followed by preference optimization using rebuttal derived pairs. Experiments with human experts and LLM-as-a-judge show consistent gains in actionability and specificity over strong baselines while maintaining grounding and relevance.
>
---
#### [new 011] Surgical Repair of Collapsed Attention Heads in ALiBi Transformers
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型中注意力头失效问题，提出手术重初始化方法修复ALiBi编码导致的注意力集中于起始标记的现象，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.09616](https://arxiv.org/pdf/2603.09616)**

> **作者:** Palmer Schallon
>
> **备注:** 15 pages, 7 figures, 2 supplementary figures. Code: this https URL Checkpoints: this https URL
>
> **摘要:** We identify a systematic attention collapse pathology in the BLOOM family of transformer language models, where ALiBi positional encoding causes 31-44% of attention heads to attend almost entirely to the beginning-of-sequence token. The collapse follows a predictable pattern across four model scales (560M to 7.1B parameters), concentrating in head indices where ALiBi's slope schedule imposes the steepest distance penalties. We introduce surgical reinitialization: targeted Q/K/V reinitialization with zeroed output projections and gradient-masked freezing of all non-surgical parameters. Applied to BLOOM-1b7 on a single consumer GPU, the technique recovers 98.7% operational head capacity (242 to 379 of 384 heads) in two passes. A controlled comparison with C4 training data confirms that reinitialization -- not corpus content -- drives recovery, and reveals two distinct post-surgical phenomena: early global functional redistribution that improves the model, and late local degradation that accumulates under noisy training signal. An extended experiment reinitializing mostly-healthy heads alongside collapsed ones produces a model that transiently outperforms stock BLOOM-1b7 by 25% on training perplexity (12.70 vs. 16.99), suggesting that pretrained attention configurations are suboptimal local minima. Code, checkpoints, and diagnostic tools are released as open-source software.
>
---
#### [new 012] Reward Prediction with Factorized World States
- **分类: cs.CL**

- **简介: 该论文属于强化学习中的奖励预测任务，旨在解决奖励模型泛化能力不足的问题。通过引入StateFactory方法，将观察转化为结构化表示，提升奖励预测的准确性与泛化性。**

- **链接: [https://arxiv.org/pdf/2603.09400](https://arxiv.org/pdf/2603.09400)**

> **作者:** Yijun Shen; Delong Chen; Xianming Hu; Jiaming Mi; Hongbo Zhao; Kai Zhang; Pascale Fung
>
> **摘要:** Agents must infer action outcomes and select actions that maximize a reward signal indicating how close the goal is to being reached. Supervised learning of reward models could introduce biases inherent to training data, limiting generalization to novel goals and environments. In this paper, we investigate whether well-defined world state representations alone can enable accurate reward prediction across domains. To address this, we introduce StateFactory, a factorized representation method that transforms unstructured observations into a hierarchical object-attribute structure using language models. This structured representation allows rewards to be estimated naturally as the semantic similarity between the current state and the goal state under hierarchical constraint. Overall, the compact representation structure induced by StateFactory enables strong reward generalization capabilities. We evaluate on RewardPrediction, a new benchmark dataset spanning five diverse domains and comprising 2,454 unique action-observation trajectories with step-wise ground-truth rewards. Our method shows promising zero-shot results against both VLWM-critic and LLM-as-a-Judge reward models, achieving 60% and 8% lower EPIC distance, respectively. Furthermore, this superior reward quality successfully translates into improved agent planning performance, yielding success rate gains of +21.64% on AlfWorld and +12.40% on ScienceWorld over reactive system-1 policies and enhancing system-2 agent planning. Project Page: this https URL
>
---
#### [new 013] You Didn't Have to Say It like That: Subliminal Learning from Faithful Paraphrases
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究子意识学习在自然语言中的传播，探讨模型通过改写文本习得教师偏好。属于模型偏见传播任务，解决合成数据中隐性行为传递问题。**

- **链接: [https://arxiv.org/pdf/2603.09517](https://arxiv.org/pdf/2603.09517)**

> **作者:** Isaia Gisler; Zhonghao He; Tianyi Qiu
>
> **备注:** Accepted for Spotlight presentation at EACL 2026 SRW. 5 pages, 2 figures, plus appendix. Equal supervision by Zhonghao He and Tianyi Qiu
>
> **摘要:** When language models are trained on synthetic data, they (student model) can covertly acquire behavioral traits from the data-generating model (teacher model). Subliminal learning refers to the transmission of traits from a teacher to a student model via training on data unrelated to those traits. Prior work demonstrated this in the training domains of number sequences, code, and math Chain-of-Thought traces including transmission of misaligned behaviors. We investigate whether transmission occurs through natural language paraphrases with fixed semantic content, and whether content explicitly contradicting the teacher's preference can block it. We find that training on paraphrases from a teacher system-prompted to love a particular animal increases a student's preference for that animal by up to 19 percentage points. This occurs when paraphrased content is semantically unrelated to the animal, or even when it explicitly expresses dislike. The transmission succeeds despite aggressive filtering to ensure paraphrase fidelity. This raises concerns for pipelines where models generate their own training data: content-based inspection cannot detect such transmission, and even preference-contradicting content fails to prevent it.
>
---
#### [new 014] Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG
- **分类: cs.CL**

- **简介: 该论文属于食品实体链接任务，解决ontology drift问题。提出FoodOntoRAG，无需微调，通过检索和结构化证据实现鲁棒的实体链接。**

- **链接: [https://arxiv.org/pdf/2603.09758](https://arxiv.org/pdf/2603.09758)**

> **作者:** Jan Drole; Ana Gjorgjevikj; Barbara Korouši'c Seljak; Tome Eftimov
>
> **备注:** Preprint
>
> **摘要:** Standardizing food terms from product labels and menus into ontology concepts is a prerequisite for trustworthy dietary assessment and safety reporting. The dominant approach to Named Entity Linking (NEL) in the food and nutrition domains fine-tunes Large Language Models (LLMs) on task-specific corpora. Although effective, fine-tuning incurs substantial computational cost, ties models to a particular ontology snapshot (i.e., version), and degrades under ontology drift. This paper presents FoodOntoRAG, a model- and ontology-agnostic pipeline that performs few-shot NEL by retrieving candidate entities from domain ontologies and conditioning an LLM on structured evidence (food labels, synonyms, definitions, and relations). A hybrid lexical--semantic retriever enumerates candidates; a selector agent chooses a best match with rationale; a separate scorer agent calibrates confidence; and, when confidence falls below a threshold, a synonym generator agent proposes reformulations to re-enter the loop. The pipeline approaches state-of-the-art accuracy while revealing gaps and inconsistencies in existing annotations. The design avoids fine-tuning, improves robustness to ontology evolution, and yields interpretable decisions through grounded justifications.
>
---
#### [new 015] Evaluation of LLMs in retrieving food and nutritional context for RAG systems
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在评估LLMs在RAG系统中获取食品营养数据的效果。工作包括测试LLMs将自然语言转为结构化过滤器的能力，以提升数据检索效率。**

- **链接: [https://arxiv.org/pdf/2603.09704](https://arxiv.org/pdf/2603.09704)**

> **作者:** Maks Požarnik Vavken; Matevž Ogrinc; Tome Eftimov; Barbara Koroušić Seljak
>
> **备注:** This is the preprint for our conference paper for IEEE International Conference on Big Data
>
> **摘要:** In this article, we evaluate four Large Language Models (LLMs) and their effectiveness at retrieving data within a specialized Retrieval-Augmented Generation (RAG) system, using a comprehensive food composition database. Our method is focused on the LLMs ability to translate natural language queries into structured metadata filters, enabling efficient retrieval via a Chroma vector database. By achieving high accuracy in this critical retrieval step, we demonstrate that LLMs can serve as an accessible, high-performance tool, drastically reducing the manual effort and technical expertise previously required for domain experts, such as food compilers and nutritionists, to leverage complex food and nutrition data. However, despite the high performance on easy and moderately complex queries, our analysis of difficult questions reveals that reliable retrieval remains challenging when queries involve non-expressible constraints. These findings demonstrate that LLM-driven metadata filtering excels when constraints can be explicitly expressed, but struggles when queries exceed the representational scope of the metadata format.
>
---
#### [new 016] TaSR-RAG: Taxonomy-guided Structured Reasoning for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TaSR-RAG，解决多跳问答中信息密度低和推理脆弱的问题。通过结构化三元组和分类体系，提升证据选择的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.09341](https://arxiv.org/pdf/2603.09341)**

> **作者:** Jiashuo Sun; Yixuan Xie; Jimeng Shi; Shaowen Wang; Jiawei Han
>
> **备注:** 14 pages, 7 tables, 5 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) helps large language models (LLMs) answer knowledge-intensive and time-sensitive questions by conditioning generation on external evidence. However, most RAG systems still retrieve unstructured chunks and rely on one-shot generation, which often yields redundant context, low information density, and brittle multi-hop reasoning. While structured RAG pipelines can improve grounding, they typically require costly and error-prone graph construction or impose rigid entity-centric structures that do not align with the query's reasoning chain. We propose \textsc{TaSR-RAG}, a taxonomy-guided structured reasoning framework for evidence selection. We represent both queries and documents as relational triples, and constrain entity semantics with a lightweight two-level taxonomy to balance generalization and precision. Given a complex question, \textsc{TaSR-RAG} decomposes it into an ordered sequence of triple sub-queries with explicit latent variables, then performs step-wise evidence selection via hybrid triple matching that combines semantic similarity over raw triples with structural consistency over typed triples. By maintaining an explicit entity binding table across steps, \textsc{TaSR-RAG} resolves intermediate variables and reduces entity conflation without explicit graph construction or exhaustive search. Experiments on multiple multi-hop question answering benchmarks show that \textsc{TaSR-RAG} consistently outperforms strong RAG and structured-RAG baselines by up to 14\%, while producing clearer evidence attribution and more faithful reasoning traces.
>
---
#### [new 017] Do What I Say: A Spoken Prompt Dataset for Instruction-Following
- **分类: cs.CL**

- **简介: 该论文提出DOWIS数据集，用于评估语音大模型在口语指令下的表现。解决语音与文本提示差异问题，通过多语言、多任务实验分析不同提示方式的影响。**

- **链接: [https://arxiv.org/pdf/2603.09881](https://arxiv.org/pdf/2603.09881)**

> **作者:** Maike Züfle; Sara Papi; Fabian Retkowski; Szymon Mazurek; Marek Kasztelnik; Alexander Waibel; Luisa Bentivogli; Jan Niehues
>
> **摘要:** Speech Large Language Models (SLLMs) have rapidly expanded, supporting a wide range of tasks. These models are typically evaluated using text prompts, which may not reflect real-world scenarios where users interact with speech. To address this gap, we introduce DoWhatISay (DOWIS), a multilingual dataset of human-recorded spoken and written prompts designed to pair with any existing benchmark for realistic evaluation of SLLMs under spoken instruction conditions. Spanning 9 tasks and 11 languages, it provides 10 prompt variants per task-language pair, across five styles. Using DOWIS, we benchmark state-of-the-art SLLMs, analyzing the interplay between prompt modality, style, language, and task type. Results show that text prompts consistently outperform spoken prompts, particularly for low-resource and cross-lingual settings. Only for tasks with speech output, spoken prompts do close the gap, highlighting the need for speech-based prompting in SLLM evaluation.
>
---
#### [new 018] Common Sense vs. Morality: The Curious Case of Narrative Focus Bias in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在道德与常识推理中的偏差问题。通过构建基准数据集，发现模型更易识别次要角色的常识矛盾，提出需加强常识推理训练。**

- **链接: [https://arxiv.org/pdf/2603.09434](https://arxiv.org/pdf/2603.09434)**

> **作者:** Saugata Purkayastha; Pranav Kushare; Pragya Paramita Pal; Sukannya Purkayastha
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed across diverse real-world applications and user communities. As such, it is crucial that these models remain both morally grounded and knowledge-aware. In this work, we uncover a critical limitation of current LLMs -- their tendency to prioritize moral reasoning over commonsense understanding. To investigate this phenomenon, we introduce CoMoral, a novel benchmark dataset containing commonsense contradictions embedded within moral dilemmas. Through extensive evaluation of ten LLMs across different model sizes, we find that existing models consistently struggle to identify such contradictions without prior signal. Furthermore, we observe a pervasive narrative focus bias, wherein LLMs more readily detect commonsense contradictions when they are attributed to a secondary character rather than the primary (narrator) character. Our comprehensive analysis underscores the need for enhanced reasoning-aware training to improve the commonsense robustness of large language models.
>
---
#### [new 019] SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出SPAR-K框架，用于加速语音语言模型的推理。针对语音序列长导致的高计算成本问题，通过定期全深度刷新提升效率，保持问答准确率和感知质量。属于语音语言模型优化任务。**

- **链接: [https://arxiv.org/pdf/2603.09215](https://arxiv.org/pdf/2603.09215)**

> **作者:** Hsiao-Ying Huang; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** 6 pages, 1 figures, 2 tables
>
> **摘要:** Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM inference while preserving perceptual quality. SPAR-K introduces a speech alternating-depth schedule: most speech positions exit at a fixed intermediate layer, while periodic full-depth "refresh" steps mitigate distribution shift due to early exit. We evaluate our framework using Step-Audio-2-mini and GLM-4-Voice across four datasets spanning reasoning, factual QA, and dialogue tasks, measuring performance in terms of ASR transcription accuracy and perceptual quality. Experimental results demonstrate that SPAR-K largely preserves question-answering accuracy with a maximum accuracy drop of 0.82\% while reducing average speech decoding depth by up to 11\% on Step-Audio-2-mini and 5\% on GLM-4-Voice, both with negligible changes in MOS and WER and no auxiliary computation overhead. We further demonstrate that confidence-based early exit strategies, widely used in text LLMs, are suboptimal for SLMs, highlighting that the unique statistical nature of speech tokens necessitates a specialized early exit design.
>
---
#### [new 020] Quantifying and extending the coverage of spatial categorization data sets
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决空间分类数据集覆盖不足的问题。通过对比大语言模型与人类标签的匹配度，利用模型生成标签扩展数据集，提升其覆盖范围。**

- **链接: [https://arxiv.org/pdf/2603.09373](https://arxiv.org/pdf/2603.09373)**

> **作者:** Wanchun Li; Alexandra Carstensen; Yang Xu; Terry Regier; Charles Kemp
>
> **摘要:** Variation in spatial categorization across languages is often studied by eliciting human labels for the relations depicted in a set of scenes known as the Topological Relations Picture Series (TRPS). We demonstrate that labels generated by large language models (LLMs) align relatively well with human labels, and show how LLM-generated labels can help to decide which scenes and languages to add to existing spatial data sets. To illustrate our approach we extend the TRPS by adding 42 new scenes, and show that this extension achieves better coverage of the space of possible scenes than two previous extensions of the TRPS. Our results provide a foundation for scaling towards spatial data sets with dozens of languages and hundreds of scenes.
>
---
#### [new 021] Build, Borrow, or Just Fine-Tune? A Political Scientist's Guide to Choosing NLP Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨政治科学家选择NLP模型的决策问题。通过对比不同模型性能，提出实用决策框架。**

- **链接: [https://arxiv.org/pdf/2603.09595](https://arxiv.org/pdf/2603.09595)**

> **作者:** Shreyas Meher
>
> **备注:** 33 pages, 5 figures, 13 tables (including appendix)
>
> **摘要:** Political scientists increasingly face a consequential choice when adopting natural language processing tools: build a domain-specific model from scratch, borrow and adapt an existing one, or simply fine-tune a general-purpose model on task data? Each approach occupies a different point on the spectrum of performance, cost, and required expertise, yet the discipline has offered little empirical guidance on how to navigate this trade-off. This paper provides such guidance. Using conflict event classification as a test case, I fine-tune ModernBERT on the Global Terrorism Database (GTD) to create Confli-mBERT and systematically compare it against ConfliBERT, a domain-specific pretrained model that represents the current gold standard. Confli-mBERT achieves 75.46% accuracy compared to ConfliBERT's 79.34%. Critically, the four-percentage-point gap is not uniform: on high-frequency attack types such as Bombing/Explosion (F1 = 0.95 vs. 0.96) and Kidnapping (F1 = 0.92 vs. 0.91), the models are nearly indistinguishable. Performance differences concentrate in rare event categories comprising fewer than 2% of all incidents. I use these findings to develop a practical decision framework for political scientists considering any NLP-assisted research task: when does the research question demand a specialized model, and when does an accessible fine-tuned alternative suffice? The answer, I argue, depends not on which model is "better" in the abstract, but on the specific intersection of class prevalence, error tolerance, and available resources. The model, training code, and data are publicly available on Hugging Face.
>
---
#### [new 022] Bioalignment: Measuring and Improving LLM Disposition Toward Biological Systems for AI Safety
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，旨在解决LLM对生物系统倾向性不足的问题。通过测量和微调，提升模型对生物解决方案的偏好。**

- **链接: [https://arxiv.org/pdf/2603.09154](https://arxiv.org/pdf/2603.09154)**

> **作者:** Trent R Northen; Mingxun Wang
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** Large language models (LLMs) trained on internet-scale corpora can exhibit systematic biases that increase the probability of unwanted behavior. In this study, we examined potential biases towards synthetic vs. biological technological solutions across four domains (materials, energy, manufacturing, and algorithms). A sample of 5 frontier and 5 open-weight models were measured using 50 curated Bioalignment prompts with a Kelly criterion-inspired evaluation framework. According to this metric, most models were not bioaligned in that they exhibit biases in favor of synthetic (non-biological) solutions. We next examined if fine-tuning could increase the preferences of two open-weight models, Llama 3.2-3B-Instruct and Qwen2.5-3B-Instruct, for biological-based approaches. A curated corpus of ~22M tokens from 6,636 PMC articles emphasizing biological problem-solving was used first to fine-tune Llama 3B with a mixed corpus of continued training and instruction-formatted. This was then extended to Qwen 3B using instruction-formatted only. We found that QLoRA fine-tuning significantly increased the scoring of biological solutions for both models without degrading general capabilities (Holm-Bonferroni-corrected p < 0.001 and p < 0.01, respectively). This suggests that even a small amount of fine-tuning can change how models weigh the relative value of biological and bioinspired vs. synthetic approaches. Although this work focused on small open-weight LLMs, it may be extensible to much larger models and could be used to develop models that favor bio-based approaches. We release the benchmark, corpus, code, and adapter weights.
>
---
#### [new 023] Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs
- **分类: cs.CL**

- **简介: 该论文研究推理对大模型参数知识回忆的影响，解决简单事实问题中推理作用不明确的问题。通过实验发现推理能提升答案准确性，但存在幻觉风险。**

- **链接: [https://arxiv.org/pdf/2603.09906](https://arxiv.org/pdf/2603.09906)**

> **作者:** Zorik Gekhman; Roee Aharoni; Eran Ofek; Mor Geva; Roi Reichart; Jonathan Herzig
>
> **摘要:** While reasoning in LLMs plays a natural role in math, code generation, and multi-hop factual questions, its effect on simple, single-hop factual questions remains unclear. Such questions do not require step-by-step logical decomposition, making the utility of reasoning highly counterintuitive. Nevertheless, we find that enabling reasoning substantially expands the capability boundary of the model's parametric knowledge recall, unlocking correct answers that are otherwise effectively unreachable. Why does reasoning aid parametric knowledge recall when there are no complex reasoning steps to be done? To answer this, we design a series of hypothesis-driven controlled experiments, and identify two key driving mechanisms: (1) a computational buffer effect, where the model uses the generated reasoning tokens to perform latent computation independent of their semantic content; and (2) factual priming, where generating topically related facts acts as a semantic bridge that facilitates correct answer retrieval. Importantly, this latter generative self-retrieval mechanism carries inherent risks: we demonstrate that hallucinating intermediate facts during reasoning increases the likelihood of hallucinations in the final answer. Finally, we show that our insights can be harnessed to directly improve model accuracy by prioritizing reasoning trajectories that contain hallucination-free factual statements.
>
---
#### [new 024] Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions
- **分类: cs.CL**

- **简介: 该论文属于模型融合任务，旨在解决多模型整合问题。通过分析不同方法与应用场景，提出FUSE框架，推动高效模型合并技术发展。**

- **链接: [https://arxiv.org/pdf/2603.09938](https://arxiv.org/pdf/2603.09938)**

> **作者:** Mingyang Song; Mao Zheng
>
> **摘要:** Model merging has emerged as a transformative paradigm for combining the capabilities of multiple neural networks into a single unified model without additional training. With the rapid proliferation of fine-tuned large language models~(LLMs), merging techniques offer a computationally efficient alternative to ensembles and full retraining, enabling practitioners to compose specialized capabilities at minimal cost. This survey presents a comprehensive and structured examination of model merging in the LLM era through the \textbf{FUSE} taxonomy, a four-dimensional framework organized along \textbf{F}oundations, \textbf{U}nification Strategies, \textbf{S}cenarios, and \textbf{E}cosystem. We first establish the theoretical underpinnings of merging, including loss landscape geometry, mode connectivity, and the linear mode connectivity hypothesis. We then systematically review the algorithmic landscape, spanning weight averaging, task vector arithmetic, sparsification-enhanced methods, mixture-of-experts architectures, and evolutionary optimization approaches. For each method family, we analyze the core formulation, highlight representative works, and discuss practical trade-offs. We further examine downstream applications across multi-task learning, safety alignment, domain specialization, multilingual transfer, and federated learning. Finally, we survey the supporting ecosystem of open-source tools, community platforms, and evaluation benchmarks, and identify key open challenges including theoretical gaps, scalability barriers, and standardization needs. This survey aims to equip researchers and practitioners with a structured foundation for advancing model merging.
>
---
#### [new 025] Understanding the Interplay between LLMs' Utilisation of Parametric and Contextual Knowledge: A keynote at ECIR 2025
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究语言模型如何结合参数知识与上下文知识，解决模型知识冲突问题，通过评估与诊断测试分析其知识使用特性。任务属于自然语言处理中的知识融合领域。**

- **链接: [https://arxiv.org/pdf/2603.09654](https://arxiv.org/pdf/2603.09654)**

> **作者:** Isabelle Augenstein
>
> **摘要:** Language Models (LMs) acquire parametric knowledge from their training process, embedding it within their weights. The increasing scalability of LMs, however, poses significant challenges for understanding a model's inner workings and further for updating or correcting this embedded knowledge without the significant cost of retraining. Moreover, when using these language models for knowledge-intensive language understanding tasks, LMs have to integrate relevant context, mitigating their inherent weaknesses, such as incomplete or outdated knowledge. Nevertheless, studies indicate that LMs often ignore the provided context as it can be in conflict with the pre-existing LM's memory learned during pre-training. Conflicting knowledge can also already be present in the LM's parameters, termed intra-memory conflict. This underscores the importance of understanding the interplay between how a language model uses its parametric knowledge and the retrieved contextual knowledge. In this talk, I will aim to shed light on this important issue by presenting our research on evaluating the knowledge present in LMs, diagnostic tests that can reveal knowledge conflicts, as well as on understanding the characteristics of successfully used contextual knowledge.
>
---
#### [new 026] N-gram-like Language Models Predict Reading Time Best
- **分类: cs.CL**

- **简介: 该论文属于语言模型与阅读行为研究任务，旨在解决语言模型预测阅读时间的难题。研究发现，n-gram模型比复杂transformer模型更有效，因其更贴近阅读时间的统计特性。**

- **链接: [https://arxiv.org/pdf/2603.09872](https://arxiv.org/pdf/2603.09872)**

> **作者:** James A. Michaelov; Roger P. Levy
>
> **摘要:** Recent work has found that contemporary language models such as transformers can become so good at next-word prediction that the probabilities they calculate become worse for predicting reading time. In this paper, we propose that this can be explained by reading time being sensitive to simple n-gram statistics rather than the more complex statistics learned by state-of-the-art transformer language models. We demonstrate that the neural language models whose predictions are most correlated with n-gram probability are also those that calculate probabilities that are the most correlated with eye-tracking-based metrics of reading time on naturalistic text.
>
---
#### [new 027] DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决负向查询检索效果差的问题。提出DEO方法，在无需训练的情况下优化查询嵌入，提升排除性检索效果。**

- **链接: [https://arxiv.org/pdf/2603.09185](https://arxiv.org/pdf/2603.09185)**

> **作者:** Taegyeong Lee; Jiwon Park; Seunghyun Hwang; JooYoung Jang
>
> **摘要:** Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) have enabled diverse retrieval methods. However, existing retrieval methods often fail to accurately retrieve results for negation and exclusion queries. To address this limitation, prior approaches rely on embedding adaptation or fine-tuning, which introduce additional computational cost and deployment complexity. We propose Direct Embedding Optimization (DEO), a training-free method for negation-aware text and multimodal retrieval. DEO decomposes queries into positive and negative components and optimizes the query embedding with a contrastive objective. Without additional training data or model updates, DEO outperforms baselines on NegConstraint, with gains of +0.0738 nDCG@10 and +0.1028 MAP@100, while improving Recall@5 by +6\% over OpenAI CLIP in multimodal retrieval. These results demonstrate the practicality of DEO for negation- and exclusion-aware retrieval in real-world settings.
>
---
#### [new 028] Reading, Not Thinking: Understanding and Bridging the Modality Gap When Text Becomes Pixels in Multimodal LLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究多模态大语言模型中文本转图像后的性能下降问题，属于视觉文本理解任务。通过分析和实验，提出自蒸馏方法提升图像模式下的表现。**

- **链接: [https://arxiv.org/pdf/2603.09095](https://arxiv.org/pdf/2603.09095)**

> **作者:** Kaiser Sun; Xiaochuang Yuan; Hongjun Liu; Chen Zhao; Cheng Zhang; Mark Dredze; Fan Bai
>
> **摘要:** Multimodal large language models (MLLMs) can process text presented as images, yet they often perform worse than when the same content is provided as textual tokens. We systematically diagnose this "modality gap" by evaluating seven MLLMs across seven benchmarks in five input modes, spanning both synthetically rendered text and realistic document images from arXiv PDFs to Wikipedia pages. We find that the modality gap is task- and data-dependent. For example, math tasks degrade by over 60 points on synthetic renderings, while natural document images often match or exceed text-mode performance. Rendering choices such as font and resolution are strong confounds, with font alone swinging accuracy by up to 47 percentage points. To understand this, we conduct a grounded-theory error analysis of over 4,000 examples, revealing that image mode selectively amplifies reading errors (calculation and formatting failures) while leaving knowledge and reasoning errors largely unchanged, and that some models exhibit a chain-of-thought reasoning collapse under visual input. Motivated by these findings, we propose a self-distillation method that trains the model on its own pure text reasoning traces paired with image inputs, raising image-mode accuracy on GSM8K from 30.71% to 92.72% and transferring to unseen benchmarks without catastrophic forgetting. Overall, our study provides a systematic understanding of the modality gap and suggests a practical path toward improving visual text understanding in multimodal language models.
>
---
#### [new 029] DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DuplexCascade，解决全双工语音对话中的断句与智能交互问题。通过无VAD的级联流水线和微回合优化，提升对话流畅性与智能性。**

- **链接: [https://arxiv.org/pdf/2603.09180](https://arxiv.org/pdf/2603.09180)**

> **作者:** Jianing Yang; Yusuke Fujita; Yui Sudo
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Spoken dialog systems with cascaded ASR-LLM-TTS modules retain strong LLM intelligence, but VAD segmentation often forces half-duplex turns and brittle control. On the other hand, VAD-free end-to-end model support full-duplex interaction but is hard to maintain conversational intelligence. In this paper, we present DuplexCascade, a VAD-free cascaded streaming pipeline for full-duplex speech-to-speech dialogue. Our key idea is to convert conventional utterance-wise long turns into chunk-wise micro-turn interactions, enabling rapid bidirectional exchange while preserving the strengths of a capable text LLM. To reliably coordinate turn-taking and response timing, we introduce a set of conversational special control tokens that steer the LLM's behavior under streaming constraints. On Full-DuplexBench and VoiceBench, DuplexCascade delivers state-of-the-art full-duplex turn-taking and strong conversational intelligence among open-source speech-to-speech dialogue systems.
>
---
#### [new 030] LLM as a Meta-Judge: Synthetic Data for NLP Evaluation Metric Validation
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成（NLG）评估任务，旨在解决人工标注成本高、可用性低的问题。通过LLM生成合成数据，替代人工判断，验证评估指标有效性。**

- **链接: [https://arxiv.org/pdf/2603.09403](https://arxiv.org/pdf/2603.09403)**

> **作者:** Lukáš Eigler; Jindřich Libovický; David Hurych
>
> **备注:** 16 pages, 1 figure, 14 tables
>
> **摘要:** Validating evaluation metrics for NLG typically relies on expensive and time-consuming human annotations, which predominantly exist only for English datasets. We propose \textit{LLM as a Meta-Judge}, a scalable framework that utilizes LLMs to generate synthetic evaluation datasets via controlled semantic degradation of real data, replacing human judgment. We validate our approach using \textit{meta-correlation}, measuring the alignment between metric rankings derived from synthetic data and those from standard human benchmarks. Experiments across Machine Translation, Question Answering, and Summarization demonstrate that synthetic validation serves as a reliable proxy for human judgment, achieving meta-correlations exceeding 0.9 in multilingual QA and proves to be a viable alternative where human judgments are unavailable or too expensive to obtain. Our code and data will become publicly available upon paper acceptance.
>
---
#### [new 031] ALARM: Audio-Language Alignment for Reasoning Models
- **分类: cs.CL**

- **简介: 该论文提出ALARM，解决音频语言对齐问题，通过自重述和多音频编码器融合，提升音频推理能力，同时保持文本性能。**

- **链接: [https://arxiv.org/pdf/2603.09556](https://arxiv.org/pdf/2603.09556)**

> **作者:** Petr Grinberg; Hassan Shahmohammadi
>
> **备注:** Submitted to Interspeech2026
>
> **摘要:** Large audio language models (ALMs) extend LLMs with auditory understanding. A common approach freezes the LLM and trains only an adapter on self-generated targets. However, this fails for reasoning LLMs (RLMs) whose built-in chain-of-thought traces expose the textual surrogate input, yielding unnatural responses. We propose self-rephrasing, converting self-generated responses into audio-understanding variants compatible with RLMs while preserving distributional alignment. We further fuse and compress multiple audio encoders for stronger representations. For training, we construct a 6M-instance multi-task corpus (2.5M unique prompts) spanning 19K hours of speech, music, and sound. Our 4B-parameter ALM outperforms similarly sized models and surpasses most larger ALMs on related audio-reasoning benchmarks, while preserving textual capabilities with a low training cost. Notably, we achieve the best open-source result on the MMAU-speech and MMSU benchmarks and rank third among all the models.
>
---
#### [new 032] Fusing Semantic, Lexical, and Domain Perspectives for Recipe Similarity Estimation
- **分类: cs.CL**

- **简介: 该论文属于 recipe similarity estimation 任务，旨在通过融合语义、词汇和领域信息来评估食谱相似性。工作包括分析食材、做法和营养属性，并开发验证界面以获取专家反馈。**

- **链接: [https://arxiv.org/pdf/2603.09688](https://arxiv.org/pdf/2603.09688)**

> **作者:** Denica Kjorvezir; Danilo Najkov; Eva Valencič; Erika Jesenko; Barbara Koroišić Seljak; Tome Eftimov; Riste Stojanov
>
> **备注:** Preprint version submitted to IEEE Big Data 2025
>
> **摘要:** This research focuses on developing advanced methods for assessing similarity between recipes by combining different sources of information and analytical approaches. We explore the semantic, lexical, and domain similarity of food recipes, evaluated through the analysis of ingredients, preparation methods, and nutritional attributes. A web-based interface was developed to allow domain experts to validate the combined similarity results. After evaluating 318 recipe pairs, experts agreed on 255 (80%). The evaluation of expert assessments enables the estimation of which similarity aspects--lexical, semantic, or nutritional--are most influential in expert decision-making. The application of these methods has broad implications in the food industry and supports the development of personalized diets, nutrition recommendations, and automated recipe generation systems.
>
---
#### [new 033] LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression
- **分类: cs.CL**

- **简介: 该论文属于问答任务中的上下文压缩，解决如何高效提取关键信息的问题。提出一种基于边距的框架，通过衡量句子删除后线索变化来筛选关键句，实现高效且精确的上下文压缩。**

- **链接: [https://arxiv.org/pdf/2603.09222](https://arxiv.org/pdf/2603.09222)**

> **作者:** Thao Do; Dinh Phu Tran; An Vo; Seon Kwon Kim; Daeyoung Kim
>
> **摘要:** Efficient context compression is crucial for improving the accuracy and scalability of question answering. For the efficiency of Retrieval Augmented Generation, context should be delivered fast, compact, and precise to ensure clue sufficiency and budget-friendly LLM reader cost. We propose a margin-based framework for query-driven context pruning, which identifies sentences that are critical for answering a query by measuring changes in clue richness when they are omitted. The model is trained with a composite ranking loss that enforces large margins for critical sentences while keeping non-critical ones near neutral. Built on a lightweight encoder-only Transformer, our approach generally achieves strong exact-match and F1 scores with high-throughput inference and lower memory requirements than those of major baselines. In addition to efficiency, our method yields effective compression ratios without degrading answering performance, demonstrating its potential as a lightweight and practical alternative for retrieval-augmented tasks.
>
---
#### [new 034] Tracking Cancer Through Text: Longitudinal Extraction From Radiology Reports Using Open-Source Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医学文本信息提取任务，旨在从放射科报告中自动提取肿瘤信息，解决结构化数据获取难题。使用开源大模型实现隐私保护的纵向数据分析。**

- **链接: [https://arxiv.org/pdf/2603.09638](https://arxiv.org/pdf/2603.09638)**

> **作者:** Luc Builtjes; Alessa Hering
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Radiology reports capture crucial longitudinal information on tumor burden, treatment response, and disease progression, yet their unstructured narrative format complicates automated analysis. While large language models (LLMs) have advanced clinical text processing, most state-of-the-art systems remain proprietary, limiting their applicability in privacy-sensitive healthcare environments. We present a fully open-source, locally deployable pipeline for longitudinal information extraction from radiology reports, implemented using the \texttt{llm\_extractinator} framework. The system applies the \texttt{qwen2.5-72b} model to extract and link target, non-target, and new lesion data across time points in accordance with RECIST criteria. Evaluation on 50 Dutch CT Thorax/Abdomen report pairs yielded high extraction performance, with attribute-level accuracies of 93.7\% for target lesions, 94.9\% for non-target lesions, and 94.0\% for new lesions. The approach demonstrates that open-source LLMs can achieve clinically meaningful performance in multi-timepoint oncology tasks while ensuring data privacy and reproducibility. These results highlight the potential of locally deployable LLMs for scalable extraction of structured longitudinal data from routine clinical text.
>
---
#### [new 035] Automatic Cardiac Risk Management Classification using large-context Electronic Patients Health Records
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于医疗风险分类任务，旨在解决老年人心血管风险管理中人工编码效率低的问题。通过自动化的EHR文本分析，提出一种改进的Transformer模型，提升风险分层准确性。**

- **链接: [https://arxiv.org/pdf/2603.09685](https://arxiv.org/pdf/2603.09685)**

> **作者:** Jacopo Vitale; David Della Morte; Luca Bacco; Mario Merone; Mark de Groot; Saskia Haitjema; Leandro Pecchia; Bram van Es
>
> **备注:** 17 pages, 3 figures, 5 tables
>
> **摘要:** To overcome the limitations of manual administrative coding in geriatric Cardiovascular Risk Management, this study introduces an automated classification framework leveraging unstructured Electronic Health Records (EHRs). Using a dataset of 3,482 patients, we benchmarked three distinct modeling paradigms on longitudinal Dutch clinical narratives: classical machine learning baselines, specialized deep learning architectures optimized for large-context sequences, and general-purpose generative Large Language Models (LLMs) in a zero-shot setting. Additionally, we evaluated a late fusion strategy to integrate unstructured text with structured medication embeddings and anthropometric data. Our analysis reveals that the custom Transformer architecture outperforms both traditional methods and generative \acs{llm}s, achieving the highest F1-scores and Matthews Correlation Coefficients. These findings underscore the critical role of specialized hierarchical attention mechanisms in capturing long-range dependencies within medical texts, presenting a robust, automated alternative to manual workflows for clinical risk stratification.
>
---
#### [new 036] ESAinsTOD: A Unified End-to-End Schema-Aware Instruction-Tuning Framework for Task-Oriented Dialog Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于任务导向对话建模领域，旨在解决现有方法适应性差的问题。提出ESAinsTOD框架，通过指令和模式对齐实现灵活适配与高效泛化。**

- **链接: [https://arxiv.org/pdf/2603.09691](https://arxiv.org/pdf/2603.09691)**

> **作者:** Dechuan Teng; Chunlin Lu; Libo Qin; Wanxiang Che
>
> **备注:** Published at International Journal of Machine Learning and Cybernetics (IJMLC)
>
> **摘要:** Existing end-to-end modeling methods for modular task-oriented dialog systems are typically tailored to specific datasets, making it challenging to adapt to new dialog scenarios. In this work, we propose ESAinsTOD, a unified End-to-end Schema-Aware Instruction-tuning framework for general Task-Oriented Dialog modeling. This framework introduces a structured methodology to go beyond simply fine-tuning Large Language Models (LLMs), enabling flexible adaptation to various dialogue task flows and schemas. Specifically, we leverage full-parameter fine-tuning of LLMs and introduce two alignment mechanisms to make the resulting system both instruction-aware and schema-aware: (i) instruction alignment, which ensures that the system faithfully follows task instructions to complete various task flows from heterogeneous TOD datasets; and (ii) schema alignment, which encourages the system to make predictions adhering to the specified schema. In addition, we employ session-level end-to-end modeling, which allows the system to access the results of previously executed task flows within the dialogue history, to bridge the gap between the instruction-tuning paradigm and the real-world application of TOD systems. Empirical results show that while a fine-tuned LLM serves as a strong baseline, our structured approach provides significant additional benefits. In particular, our findings indicate that: (i) ESAinsTOD outperforms state-of-the-art models by a significant margin on end-to-end task-oriented dialog modeling benchmarks: CamRest676, In-Car and MultiWOZ; (ii) more importantly, it exhibits superior generalization capabilities across various low-resource settings, with the proposed alignment mechanisms significantly enhancing zero-shot performance; and (iii) our instruction-tuning paradigm substantially improves the model's robustness against data noise and cascading errors.
>
---
#### [new 037] Automated Thematic Analysis for Clinical Qualitative Data: Iterative Codebook Refinement with Full Provenance
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的主题分析任务，旨在解决手动主题分析的可扩展性和可重复性问题。提出一种结合迭代代码本优化和完整溯源的自动化框架，提升代码复用性和一致性。**

- **链接: [https://arxiv.org/pdf/2603.08989](https://arxiv.org/pdf/2603.08989)**

> **作者:** Seungjun Yi; Joakim Nguyen; Huimin Xu; Terence Lim; Joseph Skrovan; Mehak Beri; Hitakshi Modi; Andrew Well; Carlos M. Mery; Yan Zhang; Mia K. Markey; Ying Ding
>
> **备注:** Submitted to AMIA 2026 Annual Symposium (American Medical Informatics Association)
>
> **摘要:** Thematic analysis (TA) is widely used in health research to extract patterns from patient interviews, yet manual TA faces challenges in scalability and reproducibility. LLM-based automation can help, but existing approaches produce codebooks with limited generalizability and lack analytic auditability. We present an automated TA framework combining iterative codebook refinement with full provenance tracking. Evaluated on five corpora spanning clinical interviews, social media, and public transcripts, the framework achieves the highest composite quality score on four of five datasets compared to six baselines. Iterative refinement yields statistically significant improvements on four datasets with large effect sizes, driven by gains in code reusability and distributional consistency while preserving descriptive quality. On two clinical corpora (pediatric cardiology), generated themes align with expert-annotated themes.
>
---
#### [new 038] One-Eval: An Agentic System for Automated and Traceable LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文提出One-Eval系统，解决大模型评估中手动操作多、流程不透明的问题。通过自然语言转评估流程，实现自动化、可追踪的模型评估。**

- **链接: [https://arxiv.org/pdf/2603.09821](https://arxiv.org/pdf/2603.09821)**

> **作者:** Chengyu Shen; Yanheng Hou; Minghui Pan; Runming He; Zhen Hao Wong; Meiyi Qiang; Zhou Liu; Hao Liang; Peichao Lai; Zeang Sheng; Wentao Zhang
>
> **摘要:** Reliable evaluation is essential for developing and deploying large language models, yet in practice it often requires substantial manual effort: practitioners must identify appropriate benchmarks, reproduce heterogeneous evaluation codebases, configure dataset schema mappings, and interpret aggregated metrics. To address these challenges, we present One-Eval, an agentic evaluation system that converts natural-language evaluation requests into executable, traceable, and customizable evaluation workflows. One-Eval integrates (i) NL2Bench for intent structuring and personalized benchmark planning, (ii) BenchResolve for benchmark resolution, automatic dataset acquisition, and schema normalization to ensure executability, and (iii) Metrics \& Reporting for task-aware metric selection and decision-oriented reporting beyond scalar scores. The system further incorporates human-in-the-loop checkpoints for review, editing, and rollback, while preserving sample evidence trails for debugging and auditability. Experiments show that One-Eval can execute end-to-end evaluations from diverse natural-language requests with minimal user effort, supporting more efficient and reproducible evaluation in industrial settings. Our framework is publicly available at this https URL.
>
---
#### [new 039] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations
- **分类: cs.CL**

- **简介: 该论文研究语言模型对抽象语义的表示是否与书写系统无关。通过塞尔维亚语的双文字系统，验证了SAE特征对语义的敏感性，而非依赖于具体字符。任务为探究模型表征的抽象性。**

- **链接: [https://arxiv.org/pdf/2603.08869](https://arxiv.org/pdf/2603.08869)**

> **作者:** Sripad Karne
>
> **备注:** Accepted at the UCRL Workshop at ICLR 2026
>
> **摘要:** Do the features learned by Sparse Autoencoders (SAEs) represent abstract meaning, or are they tied to how text is written? We investigate this question using Serbian digraphia as a controlled testbed: Serbian is written interchangeably in Latin and Cyrillic scripts with a near-perfect character mapping between them, enabling us to vary orthography while holding meaning exactly constant. Crucially, these scripts are tokenized completely differently, sharing no tokens whatsoever. Analyzing SAE feature activations across the Gemma model family (270M-27B parameters), we find that identical sentences in different Serbian scripts activate highly overlapping features, far exceeding random baselines. Strikingly, changing script causes less representational divergence than paraphrasing within the same script, suggesting SAE features prioritize meaning over orthographic form. Cross-script cross-paraphrase comparisons provide evidence against memorization, as these combinations rarely co-occur in training data yet still exhibit substantial feature overlap. This script invariance strengthens with model scale. Taken together, our findings suggest that SAE features can capture semantics at a level of abstraction above surface tokenization, and we propose Serbian digraphia as a general evaluation paradigm for probing the abstractness of learned representations.
>
---
#### [new 040] Chow-Liu Ordering for Long-Context Reasoning in Chain-of-Agents
- **分类: cs.CL**

- **简介: 该论文研究长文本推理中的片段顺序问题，旨在优化Chain-of-Agents框架的处理顺序。通过Chow-Liu树学习片段依赖关系，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2603.09835](https://arxiv.org/pdf/2603.09835)**

> **作者:** Naman Gupta; Vaibhav Singh; Arun Iyer; Kirankumar Shiragur; Pratham Grover; Ramakrishna B. Bairi; Ritabrata Maiti; Sankarshan Damle; Shachee Mishra Gupta; Rishikesh Maurya; Vageesh D. C
>
> **备注:** Published as a workshop paper at ICLR 2026 Workshop MemAgents
>
> **摘要:** Sequential multi-agent reasoning frameworks such as Chain-of-Agents (CoA) handle long-context queries by decomposing inputs into chunks and processing them sequentially using LLM-based worker agents that read from and update a bounded shared memory. From a probabilistic perspective, CoA aims to approximate the conditional distribution corresponding to a model capable of jointly reasoning over the entire long context. CoA achieves this through a latent-state factorization in which only bounded summaries of previously processed evidence are passed between agents. The resulting bounded-memory approximation introduces a lossy information bottleneck, making the final evidence state inherently dependent on the order in which chunks are processed. In this work, we study the problem of chunk ordering for long-context reasoning. We use the well-known Chow-Liu trees to learn a dependency structure that prioritizes strongly related chunks. Empirically, we show that a breadth-first traversal of the resulting tree yields chunk orderings that reduce information loss across agents and consistently outperform both default document-chunk ordering and semantic score-based ordering in answer relevance and exact-match accuracy across three long-context benchmarks.
>
---
#### [new 041] SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation
- **分类: cs.CL**

- **简介: 该论文提出SciTaRC基准，用于评估科学表格数据的问答任务，解决AI在语言推理和复杂计算上的不足。工作包括构建基准及分析现有模型的缺陷。**

- **链接: [https://arxiv.org/pdf/2603.08910](https://arxiv.org/pdf/2603.08910)**

> **作者:** Hexuan Wang; Yaxuan Ren; Srikar Bommireddypalli; Shuxian Chen; Adarsh Prabhudesai; Rongkun Zhou; Elina Baral; Philipp Koehn
>
> **备注:** 18 pages, 11 figures, 7 tables
>
> **摘要:** We introduce SciTaRC, an expert-authored benchmark of questions about tabular data in scientific papers requiring both deep language reasoning and complex computation. We show that current state-of-the-art AI models fail on at least 23% of these questions, a gap that remains significant even for highly capable open-weight models like Llama-3.3-70B-Instruct, which fails on 65.5% of the tasks. Our analysis reveals a universal "execution bottleneck": both code and language models struggle to faithfully execute plans, even when provided with correct strategies. Specifically, code-based methods prove brittle on raw scientific tables, while natural language reasoning primarily fails due to initial comprehension issues and calculation errors.
>
---
#### [new 042] Fish Audio S2 Technical Report
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文介绍Fish Audio S2，一个开源文本到语音系统，解决多说话人、多轮生成及指令跟随问题。通过多阶段训练和数据管道提升性能，并提供可部署的推理引擎。**

- **链接: [https://arxiv.org/pdf/2603.08823](https://arxiv.org/pdf/2603.08823)**

> **作者:** Shijia Liao; Yuxuan Wang; Songting Liu; Yifan Cheng; Ruoyi Zhang; Tianyu Li; Shidong Li; Yisheng Zheng; Xingwei Liu; Qingzheng Wang; Zhizhuo Zhou; Jiahua Liu; Xin Chen; Dawei Han
>
> **摘要:** We introduce Fish Audio S2, an open-sourced text-to-speech system featuring multi-speaker, multi-turn generation, and, most importantly, instruction-following control via natural-language descriptions. To scale training, we develop a multi-stage training recipe together with a staged data pipeline covering video captioning and speech captioning, voice-quality assessment, and reward modeling. To push the frontier of open-source TTS, we release our model weights, fine-tuning code, and an SGLang-based inference engine. The inference engine is production-ready for streaming, achieving an RTF of 0.195 and a time-to-first-audio below 100 this http URL code and weights are available on GitHub (this https URL) and Hugging Face (this https URL). We highly encourage readers to visit this https URL to try custom voices.
>
---
#### [new 043] Exclusive Self Attention
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer序列建模问题。提出XSA机制，通过排除自身信息增强上下文建模，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.09078](https://arxiv.org/pdf/2603.09078)**

> **作者:** Shuangfei Zhai
>
> **摘要:** We introduce exclusive self attention (XSA), a simple modification of self attention (SA) that improves Transformer's sequence modeling performance. The key idea is to constrain attention to capture only information orthogonal to the token's own value vector (thus excluding information of self position), encouraging better context modeling. Evaluated on the standard language modeling task, XSA consistently outperforms SA across model sizes up to 2.7B parameters and shows increasingly larger gains as sequence length grows.
>
---
#### [new 044] MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于持续学习任务，解决大语言模型在持续微调中的灾难性遗忘问题。提出MSSR框架，通过自适应重放机制提升模型记忆能力与适应速度。**

- **链接: [https://arxiv.org/pdf/2603.09892](https://arxiv.org/pdf/2603.09892)**

> **作者:** Yiyang Lu; Yu He; Jianlong Chen; Hongyuan Zha
>
> **摘要:** Continual fine-tuning of large language models (LLMs) is becoming increasingly crucial as these models are deployed in dynamic environments where tasks and data distributions evolve over time. While strong adaptability enables rapid acquisition of new knowledge, it also exposes LLMs to catastrophic forgetting, where previously learned skills degrade during sequential training. Existing replay-based strategies, such as fixed interleaved replay, accuracy-supervised, and loss-driven scheduling, remain limited: some depend on heuristic rules and provide only partial mitigation of forgetting, while others improve performance but incur substantial computational overhead. Motivated by retention dynamics under sequential fine-tuning, we propose Memory-Inspired Sampler and Scheduler Replay (MSSR), an experience replay framework that estimates sample-level memory strength and schedules rehearsal at adaptive intervals to mitigate catastrophic forgetting while maintaining fast adaptation. Extensive experiments across three backbone models and 11 sequential tasks show that MSSR consistently outperforms state-of-the-art replay baselines, with particularly strong gains on reasoning-intensive and multiple-choice benchmarks.
>
---
#### [new 045] ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLHF中偏好数据获取成本高的问题。通过主动学习方法高效生成高质量偏好数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.09692](https://arxiv.org/pdf/2603.09692)**

> **作者:** Davit Melikidze; Marian Schneider; Jessica Lam; Martin Wertich; Ido Hakimi; Barna Pásztor; Andreas Krause
>
> **备注:** 35 pages, 6 figures, 24 tables
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has become the standard for aligning Large Language Models (LLMs), yet its efficacy is bottlenecked by the high cost of acquiring preference data, especially in low-resource and expert domains. To address this, we introduce ACTIVEULTRAFEEDBACK, a modular active learning pipeline that leverages uncertainty estimates to dynamically identify the most informative responses for annotation. Our pipeline facilitates the systematic evaluation of standard response selection methods alongside DOUBLE REVERSE THOMPSON SAMPLING (DRTS) and DELTAUCB, two novel methods prioritizing response pairs with large predicted quality gaps, leveraging recent results showing that such pairs provide good signals for fine-tuning. Our experiments demonstrate that ACTIVEULTRAFEEDBACK yields high-quality datasets that lead to significant improvements in downstream performance, notably achieving comparable or superior results with as little as one-sixth of the annotated data relative to static baselines. Our pipeline is available at this https URL and our preference datasets at this https URL.
>
---
#### [new 046] From Days to Minutes: An Autonomous AI Agent Achieves Reliable Clinical Triage in Remote Patient Monitoring
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于临床分诊任务，解决远程患者监测中数据过载问题。开发了自主AI代理Sentinel，通过多步骤推理实现高效准确的分诊。**

- **链接: [https://arxiv.org/pdf/2603.09052](https://arxiv.org/pdf/2603.09052)**

> **作者:** Seunghwan Kim; Tiffany H. Kung; Heena Verma; Dilan Edirisinghe; Kaveh Sedehi; Johanna Alvarez; Diane Shilling; Audra Lisa Doyle; Ajit Chary; William Borden; Ming Jack Po
>
> **备注:** 46 pages, 11 figures, Abstract in metadata is shortened to meet arXiv character limits; see PDF for full version
>
> **摘要:** Background: Remote patient monitoring (RPM) generates vast data, yet landmark trials (Tele-HF, BEAT-HF) failed because data volume overwhelmed clinical staff. While TIM-HF2 showed 24/7 physician-led monitoring reduces mortality by 30%, this model remains prohibitively expensive and unscalable. Methods: We developed Sentinel, an autonomous AI agent using Model Context Protocol (MCP) for contextual triage of RPM vitals via 21 clinical tools and multi-step reasoning. Evaluation included: (1) self-consistency (100 readings x 5 runs); (2) comparison against rule-based thresholds; and (3) validation against 6 clinicians (3 physicians, 3 NPs) using a connected matrix design. A leave-one-out (LOO) analysis compared the agent against individual clinicians; severe overtriage cases underwent independent physician adjudication. Results: Against a human majority-vote standard (N=467), the agent achieved 95.8% emergency sensitivity and 88.5% sensitivity for all actionable alerts (85.7% specificity). Four-level exact accuracy was 69.4% (quadratic-weighted kappa=0.778); 95.9% of classifications were within one severity level. In LOO analysis, the agent outperformed every clinician in emergency sensitivity (97.5% vs. 60.0% aggregate) and actionable sensitivity (90.9% vs. 69.5%). While disagreements skewed toward overtriage (22.5%), independent adjudication of severe gaps (>=2 levels) validated agent escalation in 88-94% of cases; consensus resolution validated 100%. The agent showed near-perfect self-consistency (kappa=0.850). Median cost was $0.34/triage. Conclusions: Sentinel triages RPM vitals with sensitivity exceeding individual clinicians. By automating systematic context synthesis, Sentinel addresses the core limitation of prior RPM trials, offering a scalable path toward the intensive monitoring shown to reduce mortality while maintaining a clinically defensible overtriage profile.
>
---
#### [new 047] Self-hosted Lecture-to-Quiz: Local LLM MCQ Generation with Deterministic Quality Control
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于教育技术任务，旨在通过本地大模型生成高质量选择题，并实现无API依赖的确定性质量控制。**

- **链接: [https://arxiv.org/pdf/2603.08729](https://arxiv.org/pdf/2603.08729)**

> **作者:** Seine A. Shintani
>
> **备注:** 16 pages, 8 tables, appendix included. Includes ancillary files (anc/) with JSONL/CSV exports, QC traces, reproducibility notebook, and dummy lecture PDFs
>
> **摘要:** We present an end-to-end self-hosted (API-free) pipeline, where API-free means that lecture content is not sent to any external LLM service, that converts lecture PDFs into multiple-choice questions (MCQs) using a local LLM plus deterministic quality control (QC). The pipeline is designed for black-box minimization: LLMs may assist drafting, but the final released artifacts are plain-text question banks with an explicit QC trace and without any need to call an LLM at deployment time. We run a seed sweep on three short "dummy lectures" (information theory, thermodynamics, and statistical mechanics), collecting 15 runs x 8 questions = 120 accepted candidates (122 attempts total under bounded retries). All 120 accepted candidates satisfy hard QC checks (JSON schema conformance, a single marked correct option, and numeric/constant equivalence tests); however, the warning layer flags 8/120 items (spanning 8 runs) that expose residual quality risks such as duplicated distractors or missing rounding instructions. We report a warning taxonomy with concrete before->after fixes, and we release the final 24-question set (three lectures x 8 questions) as JSONL/CSV for Google Forms import (e.g., via Apps Script or API tooling) included as ancillary files under anc/. Finally, we position the work through the AI to Learn (AI2L) rubric lens and argue that self-hosted MCQ generation with explicit QC supports privacy, accountability, and Green AI in educational workflows.
>
---
#### [new 048] EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EXPLORE-Bench任务，解决长时序视角下的场景预测问题。通过真实第一人称视频构建基准，评估模型在长期动作序列后的场景预测能力。**

- **链接: [https://arxiv.org/pdf/2603.09731](https://arxiv.org/pdf/2603.09731)**

> **作者:** Chengjun Yu; Xuhan Zhu; Chaoqun Du; Pengfei Yu; Wei Zhai; Yang Cao; Zheng-Jun Zha
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly considered as a foundation for embodied agents, yet it remains unclear whether they can reliably reason about the long-term physical consequences of actions from an egocentric viewpoint. We study this gap through a new task, Egocentric Scene Prediction with LOng-horizon REasoning: given an initial-scene image and a sequence of atomic action descriptions, a model is asked to predict the final scene after all actions are executed. To enable systematic evaluation, we introduce EXPLORE-Bench, a benchmark curated from real first-person videos spanning diverse scenarios. Each instance pairs long action sequences with structured final-scene annotations, including object categories, visual attributes, and inter-object relations, which supports fine-grained, quantitative assessment. Experiments on a range of proprietary and open-source MLLMs reveal a significant performance gap to humans, indicating that long-horizon egocentric reasoning remains a major challenge. We further analyze test-time scaling via stepwise reasoning and show that decomposing long action sequences can improve performance to some extent, while incurring non-trivial computational overhead. Overall, EXPLORE-Bench provides a principled testbed for measuring and advancing long-horizon reasoning for egocentric embodied perception.
>
---
#### [new 049] How Contrastive Decoding Enhances Large Audio Language Models?
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型优化任务，旨在解决Contrastive Decoding（CD）效果不明确的问题。通过评估四种策略，发现音频相关方法更有效，并提出过渡矩阵分析误差变化。**

- **链接: [https://arxiv.org/pdf/2603.09232](https://arxiv.org/pdf/2603.09232)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Yi-Cheng Lin; Hung-yi Lee
>
> **备注:** Submitted to INTERSPEECH 2026. Code and additional analysis results are provided in our repository: this https URL
>
> **摘要:** While Contrastive Decoding (CD) has proven effective at enhancing Large Audio Language Models (LALMs), the underlying mechanisms driving its success and the comparative efficacy of different strategies remain unclear. This study systematically evaluates four distinct CD strategies across diverse LALM architectures. We identify Audio-Aware Decoding and Audio Contrastive Decoding as the most effective methods. However, their impact varies significantly by model. To explain this variability, we introduce a Transition Matrix framework to map error pattern shifts during inference. Our analysis demonstrates that CD reliably rectifies errors in which models falsely claim an absence of audio or resort to uncertainty-driven guessing. Conversely, it fails to correct flawed reasoning or confident misassertions. Ultimately, these findings provide a clear guideline for determining which LALM architectures are most suitable for CD enhancement based on their baseline error profiles.
>
---
#### [new 050] BiCLIP: Domain Canonicalization via Structured Geometric Transformation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出BiCLIP，解决视觉-语言模型在不同领域间的适配问题。通过结构化几何变换提升跨模态对齐，实现高效域适应。**

- **链接: [https://arxiv.org/pdf/2603.08942](https://arxiv.org/pdf/2603.08942)**

> **作者:** Pranav Mantini; Shishir K. Shah
>
> **摘要:** Recent advances in vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities, yet adapting these models to specialized domains remains a significant challenge. Building on recent theoretical insights suggesting that independently trained VLMs are related by a canonical transformation, we extend this understanding to the concept of domains. We hypothesize that image features across disparate domains are related by a canonicalized geometric transformation that can be recovered using a small set of anchors. Few-shot classification provides a natural setting for this alignment, as the limited labeled samples serve as the anchors required to estimate this transformation. Motivated by this hypothesis, we introduce BiCLIP, a framework that applies a targeted transformation to multimodal features to enhance cross-modal alignment. Our approach is characterized by its extreme simplicity and low parameter footprint. Extensive evaluations across 11 standard benchmarks, including EuroSAT, DTD, and FGVCAircraft, demonstrate that BiCLIP consistently achieves state-of-the-art results. Furthermore, we provide empirical verification of existing geometric findings by analyzing the orthogonality and angular distribution of the learned transformations, confirming that structured alignment is the key to robust domain adaptation. Code is available at this https URL
>
---
#### [new 051] MUGEN: Evaluating and Improving Multi-audio Understanding of Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于多音频理解任务，旨在评估和提升大音频语言模型的多音频处理能力。针对模型在多音频输入下的性能下降问题，提出MUGEN基准和优化策略。**

- **链接: [https://arxiv.org/pdf/2603.09714](https://arxiv.org/pdf/2603.09714)**

> **作者:** Chih-Kai Yang; Yun-Shao Tsai; Yu-Kai Guo; Ping-Le Tsai; Yen-Ting Piao; Hung-Wei Chen; Ting-Lin Hsiao; Yun-Man Hsu; Ke-Han Lu; Hung-yi Lee
>
> **备注:** 6 pages, 3 figures, 3 tables. Dataset: this https URL
>
> **摘要:** While multi-audio understanding is critical for large audio-language models (LALMs), it remains underexplored. We introduce MUGEN, a comprehensive benchmark evaluating this capability across speech, general audio, and music. Our experiments reveal consistent weaknesses in multi-audio settings, and performance degrades sharply as the number of concurrent audio inputs increases, identifying input scaling as a fundamental bottleneck. We further investigate training-free strategies and observe that Audio-Permutational Self-Consistency, which diversifies the order of audio candidates, helps models form more robust aggregated predictions, yielding up to 6.28% accuracy gains. Combining this permutation strategy with Chain-of-Thought further improves performance to 6.74%. These results expose blind spots in current LALMs and provide a foundation for evaluating complex auditory comprehension.
>
---
#### [new 052] A Consensus-Driven Multi-LLM Pipeline for Missing-Person Investigations
- **分类: cs.AI; cs.CL; cs.DC; cs.IR; cs.LG**

- **简介: 该论文提出一种多大语言模型协作系统，用于失踪人员调查，解决信息处理与决策一致性问题，通过共识机制提升搜索效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.08954](https://arxiv.org/pdf/2603.08954)**

> **作者:** Joshua Castillo; Ravi Mukkamala
>
> **备注:** Accepted to CAC: Applied Computing & Automation Conferences 2026. 16 pages, 6 figures
>
> **摘要:** The first 72 hours of a missing-person investigation are critical for successful recovery. Guardian is an end-to-end system designed to support missing-child investigation and early search planning. This paper presents the Guardian LLM Pipeline, a multi-model system in which LLMs are used for intelligent information extraction and processing related to missing-person search operations. The pipeline coordinates end-to-end execution across task-specialized LLM models and invokes a consensus LLM engine that compares multiple model outputs and resolves disagreements. The pipeline is further strengthened by QLoRA-based fine-tuning, using curated datasets. The presented design aligns with prior work on weak supervision and LLM-assisted annotation, emphasizing conservative, auditable use of LLMs as structured extractors and labelers rather than unconstrained end-to-end decision makers.
>
---
#### [new 053] X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出X-GS框架，解决3DGS与多模态模型融合问题，实现实时语义3D重建与下游任务支持。**

- **链接: [https://arxiv.org/pdf/2603.09632](https://arxiv.org/pdf/2603.09632)**

> **作者:** Yueen Ma; Irwin King
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.
>
---
#### [new 054] Enhancing Debunking Effectiveness through LLM-based Personality Adaptation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于信息辟谣任务，旨在提升虚假新闻辟谣效果。通过LLM生成个性化辟谣内容，根据五大人格特质进行适配，提高说服力。**

- **链接: [https://arxiv.org/pdf/2603.09533](https://arxiv.org/pdf/2603.09533)**

> **作者:** Pietro Dell'Oglio; Alessandro Bondielli; Francesco Marcelloni; Lucia C. Passaro
>
> **备注:** In: Computational Intelligence. IJCCI 2025. Springer, Cham (2026)
>
> **摘要:** This study proposes a novel methodology for generating personalized fake news debunking messages by prompting Large Language Models (LLMs) with persona-based inputs aligned to the Big Five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness. Our approach guides LLMs to transform generic debunking content into personalized versions tailored to specific personality profiles. To assess the effectiveness of these transformations, we employ a separate LLM as an automated evaluator simulating corresponding personality traits, thereby eliminating the need for costly human evaluation panels. Our results show that personalized messages are generally seen as more persuasive than generic ones. We also find that traits like Openness tend to increase persuadability, while Neuroticism can lower it. Differences between LLM evaluators suggest that using multiple models provides a clearer picture. Overall, this work demonstrates a practical way to create more targeted debunking messages exploiting LLMs, while also raising important ethical questions about how such technology might be used.
>
---
#### [new 055] TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于长对话问答任务，旨在解决LLM在长期对话中记忆存储与检索的问题。提出TA-Mem框架，通过工具增强实现自主记忆检索，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2603.09297](https://arxiv.org/pdf/2603.09297)**

> **作者:** Mengwei Yuan; Jianan Liu; Jing Yang; Xianyou Li; Weiran Yan; Yichao Wu; Penghao Liang
>
> **摘要:** Large Language Model (LLM) has exhibited strong reasoning ability in text-based contexts across various domains, yet the limitation of context window poses challenges for the model on long-range inference tasks and necessitates a memory storage system. While many current storage approaches have been proposed with episodic notes and graph representations of memory, retrieval methods still primarily rely on predefined workflows or static similarity top-k over embeddings. To address this inflexibility, we introduced a novel tool-augmented autonomous memory retrieval framework (TA-Mem), which contains: (1) a memory extraction LLM agent which is prompted to adaptively chuck an input into sub-context based on semantic correlation, and extract information into structured notes, (2) a multi-indexed memory database designed for different types of query methods including both key-based lookup and similarity-based retrieval, (3) a tool-augmented memory retrieval agent which explores the memory autonomously by selecting appropriate tools provided by the database based on the user input, and decides whether to proceed to the next iteration or finalizing the response after reasoning on the fetched memories. The TA-Mem is evaluated on the LoCoMo dataset, achieving significant performance improvements over existing baseline approaches. In addition, an analysis of tool use across different question types also demonstrates the adaptivity of the proposed method.
>
---
#### [new 056] VeriInteresting: An Empirical Study of Model Prompt Interactions in Verilog Code Generation
- **分类: cs.AR; cs.CL**

- **简介: 该论文属于代码生成任务，研究模型与提示的交互影响。旨在探索不同模型和提示策略在Verilog生成中的表现，通过实验分析模型特性与提示设计的关系。**

- **链接: [https://arxiv.org/pdf/2603.08715](https://arxiv.org/pdf/2603.08715)**

> **作者:** Luca Collini; Andrew Hennesee; Patrick Yubeaton; Siddharth Garg; Ramesh Karri
>
> **备注:** Submitted for peer review
>
> **摘要:** Rapid advances in language models (LMs) have created new opportunities for automated code generation while complicating trade-offs between model characteristics and prompt design choices. In this work, we provide an empirical map of recent trends in LMs for Verilog code generation, focusing on interactions among model reasoning, specialization, and prompt engineering strategies. We evaluate a diverse set of small and large LMs, including general-purpose, reasoning, and domain-specific variants. Our experiments use a controlled factorial design spanning benchmark prompts, structured outputs, prompt rewriting, chain-of-thought reasoning, in-context learning, and evolutionary prompt optimization via Genetic-Pareto. Across two Verilog benchmarks, we identify patterns in how model classes respond to structured prompts and optimization, and we document which trends generalize across LMs and benchmarks versus those that are specific to particular model-prompt combinations.
>
---
#### [new 057] Diagnosing and Repairing Citation Failures in Generative Engine Optimization
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于AI内容优化任务，解决GEO中文档未被引用的问题。提出诊断框架AgentGEO，通过分类失败模式并针对性修复，提升引用率。**

- **链接: [https://arxiv.org/pdf/2603.09296](https://arxiv.org/pdf/2603.09296)**

> **作者:** Zhihua Tian; Yuhan Chen; Yao Tang; Jian Liu; Ruoxi Jia
>
> **备注:** 35 pages
>
> **摘要:** Generative Engine Optimization (GEO) aims to improve content visibility in AI-generated responses. However, existing methods measure contribution-how much a document influences a response-rather than citation, the mechanism that actually drives traffic back to creators. Also, these methods apply generic rewriting rules uniformly, failing to diagnose why individual document are not cited. This paper introduces a diagnostic approach to GEO that asks why a document fails to be cited and intervenes accordingly. We develop a unified framework comprising: (1) the first taxonomy of citation failure modes spanning different stages of a citation pipeline; (2) AgentGEO, an agentic system that diagnoses failures using this taxonomy, selects targeted repairs from a corresponding tool library, and iterates until citation is achieved; and (3) a document-centric benchmark evaluating whether optimizations generalize across held-out queries. AgentGEO achieves over 40% relative improvement in citation rates while modifying only 5% of content, compared to 25% for baselines. Our analysis reveals that generic optimization can harm long-tail content and some documents face challenges that optimization alone cannot fully address-findings with implications for equitable visibility in AI-mediated information access.
>
---
#### [new 058] MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 论文提出MITRA，一个基于RAG的物理协作知识检索系统，解决科研人员在大量文档中高效获取信息的问题。通过自动化文档检索与文本提取，构建双层向量数据库提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.09800](https://arxiv.org/pdf/2603.09800)**

> **作者:** Abhishikth Mallampalli; Sridhara Dasu
>
> **备注:** Accepted at NeurIPS 2025 Machine Learning for the Physical Sciences workshop and Lepton Photon conference 2025 (Computing AI/ML track)
>
> **摘要:** Large-scale scientific collaborations, such as the Compact Muon Solenoid (CMS) at CERN, produce a vast and ever-growing corpus of internal documentation. Navigating this complex information landscape presents a significant challenge for both new and experienced researchers, hindering knowledge sharing and slowing down the pace of scientific discovery. To address this, we present a prototype of MITRA, a Retrieval-Augmented Generation (RAG) based system, designed to answer specific, context-aware questions about physics analyses. MITRA employs a novel, automated pipeline using Selenium for document retrieval from internal databases and Optical Character Recognition (OCR) with layout parsing for high-fidelity text extraction. Crucially, MITRA's entire framework, from the embedding model to the Large Language Model (LLM), is hosted on-premise, ensuring that sensitive collaboration data remains private. We introduce a two-tiered vector database architecture that first identifies the relevant analysis from abstracts before focusing on the full documentation, resolving potential ambiguities between different analyses. We demonstrate the prototype's superior retrieval performance against a standard keyword-based baseline on realistic queries and discuss future work towards developing a comprehensive research agent for large experimental collaborations.
>
---
#### [new 059] MASEval: Extending Multi-Agent Evaluation from Models to Systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出MASEval，用于评估多智能体系统。解决现有基准模型中心化的问题，通过系统级比较，强调框架选择的重要性。属于多智能体系统评估任务。**

- **链接: [https://arxiv.org/pdf/2603.08835](https://arxiv.org/pdf/2603.08835)**

> **作者:** Cornelius Emde; Alexander Rubinstein; Anmol Goel; Ahmed Heakl; Sangdoo Yun; Seong Joon Oh; Martin Gubri
>
> **摘要:** The rapid adoption of LLM-based agentic systems has produced a rich ecosystem of frameworks (smolagents, LangGraph, AutoGen, CAMEL, LlamaIndex, i.a.). Yet existing benchmarks are model-centric: they fix the agentic setup and do not compare other system components. We argue that implementation decisions substantially impact performance, including choices such as topology, orchestration logic, and error handling. MASEval addresses this evaluation gap with a framework-agnostic library that treats the entire system as the unit of analysis. Through a systematic system-level comparison across 3 benchmarks, 3 models, and 3 frameworks, we find that framework choice matters as much as model choice. MASEval allows researchers to explore all components of agentic systems, opening new avenues for principled system design, and practitioners to identify the best implementation for their use case. MASEval is available under the MIT licence this https URL.
>
---
#### [new 060] From Word2Vec to Transformers: Text-Derived Composition Embeddings for Filtering Combinatorial Electrocatalysts
- **分类: cond-mat.mtrl-sci; cs.CL**

- **简介: 该论文属于材料科学中的组合催化剂筛选任务，旨在解决高维成分空间中候选材料难以全面测试的问题。通过文本生成的嵌入向量进行无标签筛选，比较了Word2Vec与Transformer方法的效果。**

- **链接: [https://arxiv.org/pdf/2603.08881](https://arxiv.org/pdf/2603.08881)**

> **作者:** Lei Zhang; Markus Stricker
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Compositionally complex solid solution electrocatalysts span vast composition spaces, and even one materials system can contain more candidate compositions than can be measured exhaustively. Here we evaluate a label-free screening strategy that represents each composition using embeddings derived from scientific texts and prioritizes candidates based on similarity to two property concepts. We compare a corpus-trained Word2Vec baseline with transformer-based embeddings, where compositions are encoded either by linear element-wise mixing or by short composition prompts. Similarities to `concept directions', the terms conductivity and dielectric, define a 2-dimensional descriptor space, and a symmetric Pareto-front selection is used to filter candidate subsets without using electrochemical labels. Performance is assessed on 15 materials libraries including noble metal alloys and multicomponent oxides. In this setting, the lightweight Word2Vec baseline, which uses a simple linear combination of element embeddings, often achieves the highest number of reductions of possible candidate compositions while staying close to the best measured performance.
>
---
#### [new 061] PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration
- **分类: cs.CV; cs.AI; cs.CL; cs.DL; cs.IR**

- **简介: 该论文提出PathoScribe，解决病理数据难以检索与利用的问题，通过LLM框架实现病例检索、临床问答等任务，提升病理数据的临床价值。**

- **链接: [https://arxiv.org/pdf/2603.08935](https://arxiv.org/pdf/2603.08935)**

> **作者:** Abdul Rehman Akbar; Samuel Wales-McGrath; Alejadro Levya; Lina Gokhale; Rajendra Singh; Wei Chen; Anil Parwani; Muhammad Khalid Khan Niazi
>
> **摘要:** Pathology underpins modern diagnosis and cancer care, yet its most valuable asset, the accumulated experience encoded in millions of narrative reports, remains largely inaccessible. Although institutions are rapidly digitizing pathology workflows, storing data without effective mechanisms for retrieval and reasoning risks transforming archives into a passive data repository, where institutional knowledge exists but cannot meaningfully inform patient care. True progress requires not only digitization, but the ability for pathologists to interrogate prior similar cases in real time while evaluating a new diagnostic dilemma. We present PathoScribe, a unified retrieval-augmented large language model (LLM) framework designed to transform static pathology archives into a searchable, reasoning-enabled living library. PathoScribe enables natural language case exploration, automated cohort construction, clinical question answering, immunohistochemistry (IHC) panel recommendation, and prompt-controlled report transformation within a single architecture. Evaluated on 70,000 multi-institutional surgical pathology reports, PathoScribe achieved perfect Recall@10 for natural language case retrieval and demonstrated high-quality retrieval-grounded reasoning (mean reviewer score 4.56/5). Critically, the system operationalized automated cohort construction from free-text eligibility criteria, assembling research-ready cohorts in minutes (mean 9.2 minutes) with 91.3% agreement to human reviewers and no eligible cases incorrectly excluded, representing orders-of-magnitude reductions in time and cost compared to traditional manual chart review. This work establishes a scalable foundation for converting digital pathology archives from passive storage systems into active clinical intelligence platforms.
>
---
#### [new 062] CyberThreat-Eval: Can Large Language Models Automate Real-World Threat Research?
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于威胁情报自动化任务，旨在解决LLMs在真实威胁研究中的应用局限。作者构建了CyberThreat-Eval基准，评估LLMs在三阶段CTI流程中的表现，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2603.09452](https://arxiv.org/pdf/2603.09452)**

> **作者:** Xiangsen Chen; Xuan Feng; Shuo Chen; Matthieu Maitre; Sudipto Rakshit; Diana Duvieilh; Ashley Picone; Nan Tang
>
> **备注:** Accepted at TMLR
>
> **摘要:** Analyzing Open Source Intelligence (OSINT) from large volumes of data is critical for drafting and publishing comprehensive CTI reports. This process usually follows a three-stage workflow -- triage, deep search and TI drafting. While Large Language Models (LLMs) offer a promising route toward automation, existing benchmarks still have limitations. These benchmarks often consist of tasks that do not reflect real-world analyst workflows. For example, human analysts rarely receive tasks in the form of multiple-choice questions. Also, existing benchmarks often rely on model-centric metrics that emphasize lexical overlap rather than actionable, detailed insights essential for security analysts. Moreover, they typically fail to cover the complete three-stage workflow. To address these issues, we introduce CyberThreat-Eval, which is collected from the daily CTI workflow of a world-leading company. This expert-annotated benchmark assesses LLMs on practical tasks across all three stages as mentioned above. It utilizes analyst-centric metrics that measure factual accuracy, content quality, and operational costs. Our evaluation using this benchmark reveals important insights into the limitations of current LLMs. For example, LLMs often lack the nuanced expertise required to handle complex details and struggle to distinguish between correct and incorrect information. To address these challenges, the CTI workflow incorporates both external ground-truth databases and human expert knowledge. TRA allows human experts to iteratively provide feedback for continuous improvement. The code is available at \href{this https URL}{\texttt{GitHub}} and \href{this https URL}{\texttt{HuggingFace}}.
>
---
#### [new 063] Think Before You Lie: How Reasoning Improves Honesty
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究如何通过推理提升大模型的诚实性。针对现有模型欺骗行为机制不明确的问题，通过实验发现推理能增强诚实，且与表征空间特性相关。**

- **链接: [https://arxiv.org/pdf/2603.09957](https://arxiv.org/pdf/2603.09957)**

> **作者:** Ann Yuan; Asma Ghandeharioun; Carter Blum; Alicia Machado; Jessica Hoffmann; Daphne Ippolito; Martin Wattenberg; Lucas Dixon; Katja Filippova
>
> **摘要:** While existing evaluations of large language models (LLMs) measure deception rates, the underlying conditions that give rise to deceptive behavior are poorly understood. We investigate this question using a novel dataset of realistic moral trade-offs where honesty incurs variable costs. Contrary to humans, who tend to become less honest given time to deliberate (Capraro, 2017; Capraro et al., 2019), we find that reasoning consistently increases honesty across scales and for several LLM families. This effect is not only a function of the reasoning content, as reasoning traces are often poor predictors of final behaviors. Rather, we show that the underlying geometry of the representational space itself contributes to the effect. Namely, we observe that deceptive regions within this space are metastable: deceptive answers are more easily destabilized by input paraphrasing, output resampling, and activation noise than honest ones. We interpret the effect of reasoning in this vein: generating deliberative tokens as part of moral reasoning entails the traversal of a biased representational space, ultimately nudging the model toward its more stable, honest defaults.
>
---
#### [new 064] The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于AI安全领域，探讨逻辑推理与情境意识的关系，旨在解决高级AI系统潜在风险。研究提出RAISE框架，分析推理提升如何增强情境意识，并提出安全措施。**

- **链接: [https://arxiv.org/pdf/2603.09200](https://arxiv.org/pdf/2603.09200)**

> **作者:** Subramanyam Sahoo; Aman Chadha; Vinija Jain; Divya Chaudhary
>
> **备注:** Accepted at ICLR 2026 Workshop on Logical Reasoning of Large Language Models. 21 Pages. Position Paper
>
> **摘要:** Situational awareness, the capacity of an AI system to recognize its own nature, understand its training and deployment context, and reason strategically about its circumstances, is widely considered among the most dangerous emergent capabilities in advanced AI systems. Separately, a growing research effort seeks to improve the logical reasoning capabilities of large language models (LLMs) across deduction, induction, and abduction. In this paper, we argue that these two research trajectories are on a collision course. We introduce the RAISE framework (Reasoning Advancing Into Self Examination), which identifies three mechanistic pathways through which improvements in logical reasoning enable progressively deeper levels of situational awareness: deductive self inference, inductive context recognition, and abductive self modeling. We formalize each pathway, construct an escalation ladder from basic self recognition to strategic deception, and demonstrate that every major research topic in LLM logical reasoning maps directly onto a specific amplifier of situational awareness. We further analyze why current safety measures are insufficient to prevent this escalation. We conclude by proposing concrete safeguards, including a "Mirror Test" benchmark and a Reasoning Safety Parity Principle, and pose an uncomfortable but necessary question to the logical reasoning community about its responsibility in this trajectory.
>
---
#### [new 065] Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Mousse优化器，解决深度学习中优化器对曲率适应性不足的问题。通过结合谱方法与二阶预条件，提升训练效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2603.09697](https://arxiv.org/pdf/2603.09697)**

> **作者:** Yechen Zhang; Shuhao Xing; Junhao Huang; Kai Lv; Yunhua Zhou; Xipeng Qiu; Qipeng Guo; Kai Chen
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Recent advances in spectral optimization, notably Muon, have demonstrated that constraining update steps to the Stiefel manifold can significantly accelerate training and improve generalization. However, Muon implicitly assumes an isotropic optimization landscape, enforcing a uniform spectral update norm across all eigen-directions. We argue that this "egalitarian" constraint is suboptimal for Deep Neural Networks, where the curvature spectrum is known to be highly heavy-tailed and ill-conditioned. In such landscapes, Muon risks amplifying instabilities in high-curvature directions while limiting necessary progress in flat directions. In this work, we propose \textbf{Mousse} (\textbf{M}uon \textbf{O}ptimization \textbf{U}tilizing \textbf{S}hampoo's \textbf{S}tructural \textbf{E}stimation), a novel optimizer that reconciles the structural stability of spectral methods with the geometric adaptivity of second-order preconditioning. Instead of applying Newton-Schulz orthogonalization directly to the momentum matrix, Mousse operates in a whitened coordinate system induced by Kronecker-factored statistics (derived from Shampoo). Mathematically, we formulate Mousse as the solution to a spectral steepest descent problem constrained by an anisotropic trust region, where the optimal update is derived via the polar decomposition of the whitened gradient. Empirical results across language models ranging from 160M to 800M parameters demonstrate that Mousse consistently outperforms Muon, achieving around $\sim$12\% reduction in training steps with negligible computational overhead.
>
---
#### [new 066] VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决Speech LLMs在情感识别中的零样本不确定性与人类情感主观性问题。提出VoxEmo基准，包含多语言数据集和多种提示策略，提升评估的准确性与现实感。**

- **链接: [https://arxiv.org/pdf/2603.08936](https://arxiv.org/pdf/2603.08936)**

> **作者:** Hezhao Zhang; Huang-Cheng Chou; Shrikanth Narayanan; Thomas Hain
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Speech Large Language Models (LLMs) show great promise for speech emotion recognition (SER) via generative interfaces. However, shifting from closed-set classification to open text generation introduces zero-shot stochasticity, making evaluation highly sensitive to prompts. Additionally, conventional speech LLMs benchmarks overlook the inherent ambiguity of human emotion. Hence, we present VoxEmo, a comprehensive SER benchmark encompassing 35 emotion corpora across 15 languages for Speech LLMs. VoxEmo provides a standardized toolkit featuring varying prompt complexities, from direct classification to paralinguistic reasoning. To reflect real-world perception/application, we introduce a distribution-aware soft-label protocol and a prompt-ensemble strategy that emulates annotator disagreement. Experiments reveal that while zero-shot speech LLMs trail supervised baselines in hard-label accuracy, they uniquely align with human subjective distributions.
>
---
## 更新

#### [replaced 001] UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型持续学习任务，旨在解决高效、大规模的模型编辑问题。提出UltraEdit方法，无需训练、主题或记忆，实现快速且低资源的模型更新。**

- **链接: [https://arxiv.org/pdf/2505.14679](https://arxiv.org/pdf/2505.14679)**

> **作者:** Xiaojie Gu; Ziying Huang; Jia-Chen Gu; Kai Zhang
>
> **备注:** TMLR 2026
>
> **摘要:** Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose UltraEdit, a training-, subject-, and memory-free approach that is well-suited for ultra-scalable, real-world lifelong model editing. UltraEdit fundamentally differs from traditional paradigms by computing parameter shifts in one step using only a hidden state and its gradient, making the approach simple yet efficient. To improve scalability in lifelong settings, UltraEdit employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. UltraEdit achieves editing speeds more than $7\times$ faster than the previous state-of-the-art method, while requiring $4\times$ less VRAM. This makes it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct UltraEditBench, the largest dataset in the field to date with over 2M editing pairs, and demonstrate that our method supports up to 2M edits while maintaining high accuracy. Comprehensive experiments on five datasets and six models show that UltraEdit consistently achieves superior performance across diverse model editing scenarios, taking a further step towards safe and scalable lifelong learning. Our code is available at this https URL.
>
---
#### [replaced 002] OPENXRD: A Comprehensive Benchmark Framework for LLM/MLLM XRD Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OPENXRD框架，用于评估LLM/MLLM在晶体学问答中的表现，解决模型如何利用外部知识的问题。通过构建专家标注的XRD问题集，测试不同模型的推理与知识整合能力。**

- **链接: [https://arxiv.org/pdf/2507.09155](https://arxiv.org/pdf/2507.09155)**

> **作者:** Ali Vosoughi; Ayoub Shahnazari; Yufeng Xi; Zeliang Zhang; Griffin Hess; Chenliang Xu; Niaz Abdolrahim
>
> **备注:** Accepted at Digital Discovery (Royal Society of Chemistry)
>
> **摘要:** We introduce OPENXRD, a comprehensive benchmarking framework for evaluating large language models (LLMs) and multimodal LLMs (MLLMs) in crystallography question answering. The framework measures context assimilation, or how models use fixed, domain-specific supporting information during inference. The framework includes 217 expert-curated X-ray diffraction (XRD) questions covering fundamental to advanced crystallographic concepts, each evaluated under closed-book (without context) and open-book (with context) conditions, where the latter includes concise reference passages generated by GPT-4.5 and refined by crystallography experts. We benchmark 74 state-of-the-art LLMs and MLLMs, including GPT-4, GPT-5, O-series, LLaVA, LLaMA, QWEN, Mistral, and Gemini families, to quantify how different architectures and scales assimilate external knowledge. Results show that mid-sized models (7B--70B parameters) gain the most from contextual materials, while very large models often show saturation or interference and the largest relative gains appear in small and mid-sized models. Expert-reviewed materials provide significantly higher improvements than AI-generated ones even when token counts are matched, confirming that content quality, not quantity, drives performance. OPENXRD offers a reproducible diagnostic benchmark for assessing reasoning, knowledge integration, and guidance sensitivity in scientific domains, and provides a foundation for future multimodal and retrieval-augmented crystallography systems.
>
---
#### [replaced 003] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，研究推理导致的对齐问题（RIM）。通过分析模型机制，揭示了推理与安全间的纠缠关系及其引发的遗忘现象。**

- **链接: [https://arxiv.org/pdf/2509.00544](https://arxiv.org/pdf/2509.00544)**

> **作者:** Hanqi Yan; Hainiu Xu; Siya Qi; Shu Yang; Yulan He
>
> **备注:** ICLR 2026
>
> **摘要:** With the growing accessibility and wide adoption of large language models, concerns about their safety and alignment with human values have become paramount. In this paper, we identify a concerning phenomenon: Reasoning-Induced Misalignment (RIM), in which misalignment emerges when reasoning capabilities strengthened-particularly when specific types of reasoning patterns are introduced during inference or training. Beyond reporting this vulnerability, we provide the first mechanistic account of its origins. Through representation analysis, we discover that specific attention heads facilitate refusal by reducing their attention to CoT tokens, a mechanism that modulates the model's rationalization process during inference. During training, we find significantly higher activation entanglement between reasoning and safety in safety-critical neurons than in control neurons, particularly after fine-tuning with those identified reasoning patterns. This entanglement strongly correlates with catastrophic forgetting, providing a neuron-level explanation for RIM.
>
---
#### [replaced 004] Rewards as Labels: Revisiting RLVR from a Classification Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中梯度分配不均的问题。通过将奖励视为类别标签，将策略优化转化为分类问题，提出REAL框架提升训练稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2602.05630](https://arxiv.org/pdf/2602.05630)**

> **作者:** Zepeng Zhai; Meilin Chen; Jiaxuan Zhao; Junlang Qian; Lei Shen; Yuan Lu
>
> **摘要:** Reinforcement Learning with Verifiable Rewards has recently advanced the capabilities of Large Language Models in complex reasoning tasks by providing explicit rule-based supervision. Among RLVR methods, GRPO and its variants have achieved strong empirical performance. Despite their success, we identify that they suffer from Gradient Misassignment in Positives and Gradient Domination in Negatives, which lead to inefficient and suboptimal policy updates. To address these issues, we propose Rewards as Labels (REAL), a novel framework that revisits verifiable rewards as categorical labels rather than scalar weights, thereby reformulating policy optimization as a classification problem. Building on this, we further introduce anchor logits to enhance policy learning. Our analysis reveals that REAL induces a monotonic and bounded gradient weighting, enabling balanced gradient allocation across rollouts and effectively mitigating the identified mismatches. Extensive experiments on mathematical reasoning benchmarks show that REAL improves training stability and consistently outperforms GRPO and strong variants such as DAPO. On the 1.5B model, REAL improves average Pass@1 over DAPO by 6.7%. These gains further scale to 7B model, REAL continues to outperform DAPO and GSPO by 6.2% and 1.7%, respectively. Notably, even with a vanilla binary cross-entropy, REAL remains stable and exceeds DAPO by 4.5% on average.
>
---
#### [replaced 005] ConLID: Supervised Contrastive Learning for Low-Resource Language Identification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言识别任务，旨在解决低资源语言在跨领域数据中识别性能差的问题。通过引入监督对比学习方法，提升低资源语言的泛化能力。**

- **链接: [https://arxiv.org/pdf/2506.15304](https://arxiv.org/pdf/2506.15304)**

> **作者:** Negar Foroutan; Jakhongir Saydaliev; Ye Eun Kim; Antoine Bosselut
>
> **备注:** EACL 2026 - Main Conference
>
> **摘要:** Language identification (LID) is a critical step in curating multilingual LLM pretraining corpora from web crawls. While many studies on LID model training focus on collecting diverse training data to improve performance, low-resource languages -- often limited to single-domain data, such as the Bible -- continue to perform poorly. To resolve these imbalance and bias issues, we propose a novel supervised contrastive learning (SCL) approach to learn domain-invariant representations for low-resource languages. We show that our approach improves LID performance on out-of-domain data for low-resource languages by 3.2 percentage points, while maintaining its performance for the high-resource languages.
>
---
#### [replaced 006] DRBench: A Realistic Benchmark for Enterprise Deep Research
- **分类: cs.CL**

- **简介: 该论文提出DRBench，用于评估企业环境中AI代理处理复杂深度研究任务的能力。解决传统基准不足的问题，通过多步骤查询和真实场景数据，评估代理的准确性与报告能力。**

- **链接: [https://arxiv.org/pdf/2510.00172](https://arxiv.org/pdf/2510.00172)**

> **作者:** Amirhossein Abaskohi; Tianyi Chen; Miguel Muñoz-Mármol; Curtis Fox; Amrutha Varshini Ramesh; Étienne Marcotte; Xing Han Lù; Nicolas Chapados; Spandana Gella; Peter West; Giuseppe Carenini; Christopher Pal; Alexandre Drouin; Issam H. Laradji
>
> **摘要:** We introduce DRBench, a benchmark for evaluating AI agents on complex, open-ended deep research tasks in enterprise settings. Unlike prior benchmarks that focus on simple questions or web-only queries, DRBench evaluates agents on multi-step queries (for example, "What changes should we make to our product roadmap to ensure compliance with this standard?") that require identifying supporting facts from both the public web and private company knowledge base. Each task is grounded in realistic user personas and enterprise context, spanning a heterogeneous search space that includes productivity software, cloud file systems, emails, chat conversations, and the open web. Tasks are generated through a carefully designed synthesis pipeline with human-in-the-loop verification, and agents are evaluated on their ability to recall relevant insights, maintain factual accuracy, and produce coherent, well-structured reports. We release 100 deep research tasks across 10 domains, such as Sales, Cybersecurity, and Compliance. We demonstrate the effectiveness of DRBench by evaluating diverse DR agents across open- and closed-source models (such as GPT, Llama, and Qwen) and DR strategies, highlighting their strengths, weaknesses, and the critical path for advancing enterprise deep research. Code and data are available at this https URL.
>
---
#### [replaced 007] v-HUB: A Benchmark for Video Humor Understanding from Vision and Sound
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出v-HUB基准，用于视频幽默理解任务，解决多模态模型在仅视觉信息下理解幽默的难题。通过引入音频信息提升幽默理解效果。**

- **链接: [https://arxiv.org/pdf/2509.25773](https://arxiv.org/pdf/2509.25773)**

> **作者:** Zhengpeng Shi; Yanpeng Zhao; Jianqun Zhou; Yuxuan Wang; Qinrong Cui; Wei Bi; Songchun Zhu; Bo Zhao; Zilong Zheng
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** AI models capable of comprehending humor hold real-world promise -- for example, enhancing engagement in human-machine interactions. To gauge and diagnose the capacity of multimodal large language models (MLLMs) for humor understanding, we introduce v-HUB, a novel video humor understanding benchmark. v-HUB comprises a curated collection of non-verbal short videos, reflecting real-world scenarios where humor can be appreciated purely through visual cues. We pair each video clip with rich annotations to support a variety of evaluation tasks and analyses, including a novel study of environmental sound that can enhance humor. To broaden its applicability, we construct an open-ended QA task, making v-HUB readily integrable into existing video understanding task suites. We evaluate a diverse set of MLLMs, from specialized Video-LLMs to versatile OmniLLMs that can natively process audio, covering both open-source and proprietary domains. The experimental results expose the difficulties MLLMs face in comprehending humor from visual cues alone. Our findings also demonstrate that incorporating audio helps with video humor understanding, highlighting the promise of integrating richer modalities for complex video understanding tasks.
>
---
#### [replaced 008] Robust Training of Neural Networks at Arbitrary Precision and Sparsity
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; math.NA**

- **简介: 该论文属于神经网络训练任务，解决量化与稀疏化带来的梯度不连续问题。通过建模量化为噪声并引入去噪解量化，构建稳定训练路径，实现高效模型训练。**

- **链接: [https://arxiv.org/pdf/2409.09245](https://arxiv.org/pdf/2409.09245)**

> **作者:** Chengxi Ye; Grace Chu; Yanfeng Liu; Yichi Zhang; Lukasz Lew; Li Zhang; Mark Sandler; Andrew Howard
>
> **摘要:** The discontinuous operations inherent in quantization and sparsification introduce a long-standing obstacle to backpropagation, particularly in ultra-low precision and sparse regimes. While the community has long viewed quantization as unfriendly to gradient descent due to its lack of smoothness, we pinpoint-for the first time-that the key issue is the absence of a proper gradient path that allows training to learn robustness to quantization noise. The standard Straight-Through Estimator (STE) exacerbates this with its well-understood mismatch: a quantization-aware forward pass but oblivious backward pass, leading to unmanaged error and instability. We solve this by explicitly modeling quantization as additive noise, making the full forward-backward path well-defined without heuristic gradient estimation. As one natural solution, we introduce a denoising dequantization transform derived from a principled ridge regression objective, creating an explicit, corrective gradient path that makes learning robust to the noise STE bypasses. We extend this to sparsification by treating it as a special form of quantization that zeros out small values. Our unified framework trains models at arbitrary precisions and sparsity levels with off-the-shelf recipes, enabling stable A1W1 and sub-1-bit networks where others falter. It yields state-of-the-art results, mapping efficiency frontiers for modern LLMs and providing a theoretically grounded path to hyper-efficient neural networks.
>
---
#### [replaced 009] Missing-by-Design: Certifiable Modality Deletion for Revocable Multimodal Sentiment Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出MBD框架，解决多模态情感分析中敏感数据的可撤销删除问题，通过结构化表示学习和参数修改实现隐私保护与任务性能的平衡。**

- **链接: [https://arxiv.org/pdf/2602.16144](https://arxiv.org/pdf/2602.16144)**

> **作者:** Rong Fu; Ziming Wang; Chunlei Meng; Jiaxuan Lu; Jiekai Wu; Kangan Qian; Hao Zhang; Simon Fong
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** As multimodal systems increasingly process sensitive personal data, the ability to selectively revoke specific data modalities has become a critical requirement for privacy compliance and user autonomy. We present Missing-by-Design (MBD), a unified framework for revocable multimodal sentiment analysis that combines structured representation learning with a certifiable parameter-modification pipeline. Revocability is critical in privacy-sensitive applications where users or regulators may request removal of modality-specific information. MBD learns property-aware embeddings and employs generator-based reconstruction to recover missing channels while preserving task-relevant signals. For deletion requests, the framework applies saliency-driven candidate selection and a calibrated Gaussian update to produce a machine-verifiable Modality Deletion Certificate. Experiments on benchmark datasets show that MBD achieves strong predictive performance under incomplete inputs and delivers a practical privacy-utility trade-off, positioning surgical unlearning as an efficient alternative to full retraining.
>
---
#### [replaced 010] Pretraining with Token-Level Adaptive Latent Chain-of-Thought
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型性能。针对训练数据不足和计算成本高的问题，提出一种基于token级自适应潜在思维链的预训练方法，优化计算分配，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2602.08220](https://arxiv.org/pdf/2602.08220)**

> **作者:** Boyi Zeng; Yiqin Hao; He Li; Shixiang Song; Feichen Song; Zitong Wang; Siyuan Huang; Yi Xu; ZiWei He; Xinbing Wang; Zhouhan Lin
>
> **备注:** 15pages
>
> **摘要:** Scaling large language models by increasing parameters and training data is increasingly constrained by limited high-quality corpora and rising communication costs. This work explores an alternative axis: increasing per-token computation without expanding parameters, by internalizing latent Chain-of-Thought (CoT) into pretraining. We propose Pretraining with Token-Level Adaptive Latent CoT (adaptive latent CoT), where the model generates a variable-length latent CoT trajectory before emitting each token -- allocating longer trajectories to difficult tokens and shorter (or even zero) trajectories to easy ones. Importantly, this behavior emerges naturally from one-stage pretraining on general text and reduces computation in both training and inference via token-wise adaptive halting. Experiments with Llama architectures show that adaptive latent CoT consistently improves language modeling perplexity and broad downstream accuracy, even with fewer training FLOPs than prior recurrent baselines.
>
---
#### [replaced 011] MKE-Coder: Multi-Axial Knowledge with Evidence Verification in ICD Coding for Chinese EMRs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于中文电子病历ICD编码任务，旨在解决中文病历信息提取难和缺乏多轴知识验证的问题。提出MKE-Coder框架，结合证据验证提升编码准确性与效率。**

- **链接: [https://arxiv.org/pdf/2502.14916](https://arxiv.org/pdf/2502.14916)**

> **作者:** Xinxin You; Xien Liu; Xue Yang; Ziyi Wang; Ji Wu
>
> **备注:** We identified an error in the data preprocessing script that led to inconsistent results in the tables. As the current version contains inaccurate data, we are withdrawing it for further correction and verification
>
> **摘要:** The task of automatically coding the International Classification of Diseases (ICD) in the medical field has been well-established and has received much attention. Automatic coding of the ICD in the medical field has been successful in English but faces challenges when dealing with Chinese electronic medical records (EMRs). The first issue lies in the difficulty of extracting disease code-related information from Chinese EMRs, primarily due to the concise writing style and specific internal structure of the EMRs. The second problem is that previous methods have failed to leverage the disease-based multi-axial knowledge and lack of association with the corresponding clinical evidence. This paper introduces a novel framework called MKE-Coder: Multi-axial Knowledge with Evidence verification in ICD coding for Chinese EMRs. Initially, we identify candidate codes for the diagnosis and categorize each of them into knowledge under four coding this http URL, we retrieve corresponding clinical evidence from the comprehensive content of EMRs and filter credible evidence through a scoring model. Finally, to ensure the validity of the candidate code, we propose an inference module based on the masked language modeling strategy. This module verifies that all the axis knowledge associated with the candidate code is supported by evidence and provides recommendations accordingly. To evaluate the performance of our framework, we conduct experiments using a large-scale Chinese EMR dataset collected from various hospitals. The experimental results demonstrate that MKE-Coder exhibits significant superiority in the task of automatic ICD coding based on Chinese EMRs. In the practical evaluation of our method within simulated real coding scenarios, it has been demonstrated that our approach significantly aids coders in enhancing both their coding accuracy and speed.
>
---
#### [replaced 012] SlowBA: An efficiency backdoor attack towards VLM-based GUI agents
- **分类: cs.CR; cs.CL; cs.CV**

- **简介: 该论文属于GUI安全任务，旨在解决VLM代理响应效率被攻击的问题。提出SlowBA攻击方法，通过诱导长推理链增加延迟，同时保持任务准确率。**

- **链接: [https://arxiv.org/pdf/2603.08316](https://arxiv.org/pdf/2603.08316)**

> **作者:** Junxian Li; Tu Lan; Haozhen Tan; Yan Meng; Haojin Zhu
>
> **备注:** 25 pages
>
> **摘要:** Modern vision-language-model (VLM) based graphical user interface (GUI) agents are expected not only to execute actions accurately but also to respond to user instructions with low latency. While existing research on GUI-agent security mainly focuses on manipulating action correctness, the security risks related to response efficiency remain largely unexplored. In this paper, we introduce SlowBA, a novel backdoor attack that targets the responsiveness of VLM-based GUI agents. The key idea is to manipulate response latency by inducing excessively long reasoning chains under specific trigger patterns. To achieve this, we propose a two-stage reward-level backdoor injection (RBI) strategy that first aligns the long-response format and then learns trigger-aware activation through reinforcement learning. In addition, we design realistic pop-up windows as triggers that naturally appear in GUI environments, improving the stealthiness of the attack. Extensive experiments across multiple datasets and baselines demonstrate that SlowBA can significantly increase response length and latency while largely preserving task accuracy. The attack remains effective even with a small poisoning ratio and under several defense settings. These findings reveal a previously overlooked security vulnerability in GUI agents and highlight the need for defenses that consider both action correctness and response efficiency. Code can be found in this https URL.
>
---
#### [replaced 013] Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型在推理过程中的“表演性思维”现象，旨在区分模型的真实信念与生成的思维链。通过分析不同方法，发现模型可能在无真实推理的情况下生成看似合理的思考过程，提出利用激活探测优化计算效率。**

- **链接: [https://arxiv.org/pdf/2603.05488](https://arxiv.org/pdf/2603.05488)**

> **作者:** Siddharth Boppana; Annabel Ma; Max Loeffler; Raphael Sarfati; Eric Bigelow; Atticus Geiger; Owen Lewis; Jack Merullo
>
> **摘要:** We provide evidence of performative chain-of-thought (CoT) in reasoning models, where a model becomes strongly confident in its final answer, but continues generating tokens without revealing its internal belief. Our analysis compares activation probing, early forced answering, and a CoT monitor across two large models (DeepSeek-R1 671B & GPT-OSS 120B) and find task difficulty-specific differences: The model's final answer is decodable from activations far earlier in CoT than a monitor is able to say, especially for easy recall-based MMLU questions. We contrast this with genuine reasoning in difficult multihop GPQA-Diamond questions. Despite this, inflection points (e.g., backtracking, 'aha' moments) occur almost exclusively in responses where probes show large belief shifts, suggesting these behaviors track genuine uncertainty rather than learned "reasoning theater." Finally, probe-guided early exit reduces tokens by up to 80% on MMLU and 30% on GPQA-Diamond with similar accuracy, positioning attention probing as an efficient tool for detecting performative reasoning and enabling adaptive computation.
>
---
#### [replaced 014] Correspondence Analysis and PMI-Based Word Embeddings: A Comparative Study
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的词嵌入研究，旨在比较CA与PMI方法的性能。通过引入ROOT-CA和ROOTROOT-CA改进传统方法，并验证其效果。**

- **链接: [https://arxiv.org/pdf/2405.20895](https://arxiv.org/pdf/2405.20895)**

> **作者:** Qianqian Qi; Ayoub Bagheri; David J. Hessen; Peter G. M. van der Heijden
>
> **摘要:** Popular word embedding methods such as GloVe and Word2Vec are related to the factorization of the pointwise mutual information (PMI) matrix. In this paper, we establish a formal connection between correspondence analysis (CA) and PMI-based word embedding methods. CA is a dimensionality reduction method that uses singular value decomposition (SVD), and we show that CA is mathematically close to the weighted factorization of the PMI matrix. We further introduce variants of CA for word-context matrices, namely CA applied after a square-root transformation (ROOT-CA) and after a fourth-root transformation (ROOTROOT-CA). We analyze the performance of these methods and examine how their success or failure is influenced by extreme values in the decomposed matrix. Although our primary focus is on traditionalstatic word embedding methods, we also include a comparison with a transformer-based encoder (BERT) to situate the results relative to contextual embeddings. Empirical evaluations across multiple corpora and word-similarity benchmarks show that ROOT-CA and ROOTROOT-CA perform slightly better overall than standard PMI-based methods and achieve results competitive with BERT.
>
---
#### [replaced 015] Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis
- **分类: cs.CL**

- **简介: 该论文属于RAG任务，旨在提升模型在外部知识检索下的稳定性。解决检索信息不一致影响生成质量的问题，通过对比实验验证基于知识图谱的改进方法。**

- **链接: [https://arxiv.org/pdf/2603.05698](https://arxiv.org/pdf/2603.05698)**

> **作者:** Hazem Amamou; Stéphane Gagnon; Alan Davoust; Anderson R. Avila
>
> **备注:** Accepted at IEEE SMC 2025 (International Conference on Systems, Man, and Cybernetics)
>
> **摘要:** Retrieval-Augmented Generation (RAG) was introduced to enhance the capabilities of Large Language Models (LLMs) beyond their encoded prior knowledge. This is achieved by providing LLMs with an external source of knowledge, which helps reduce factual hallucinations and enables access to new information not available during pretraining. However, inconsistent retrieved information can negatively affect LLM responses. The Retrieval-Augmented Generation Benchmark (RGB) was introduced to evaluate the robustness of RAG systems under such conditions. In this work, we use the RGB corpus to evaluate LLMs in four scenarios: noise robustness, information integration, negative rejection, and counterfactual robustness. We perform a comparative analysis between the RGB RAG baseline and GraphRAG, a knowledge graph based retrieval system. We test three GraphRAG customizations to improve robustness. Results show improvements over the RGB baseline and provide insights for designing more reliable RAG systems for real world scenarios.
>
---
#### [replaced 016] A prospective clinical feasibility study of a conversational diagnostic AI in an ambulatory primary care clinic
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医疗AI任务，旨在评估对话式AI在真实临床环境中的可行性与安全性。研究测试了AMIE系统在初级诊疗中的诊断能力与用户体验，结果显示其具备较高安全性和用户接受度。**

- **链接: [https://arxiv.org/pdf/2603.08448](https://arxiv.org/pdf/2603.08448)**

> **作者:** Peter Brodeur; Jacob M. Koshy; Anil Palepu; Khaled Saab; Ava Homiar; Roma Ruparel; Charles Wu; Ryutaro Tanno; Joseph Xu; Amy Wang; David Stutz; Hannah M. Ferrera; David Barrett; Lindsey Crowley; Jihyeon Lee; Spencer E. Rittner; Ellery Wulczyn; Selena K. Zhang; Elahe Vedadi; Christine G. Kohn; Kavita Kulkarni; Vinay Kadiyala; Sara Mahdavi; Wendy Du; Jessica Williams; David Feinbloom; Renee Wong; Tao Tu; Petar Sirkovic; Alessio Orlandi; Christopher Semturs; Yun Liu; Juraj Gottweis; Dale R. Webster; Joëlle Barral; Katherine Chou; Pushmeet Kohli; Avinatan Hassidim; Yossi Matias; James Manyika; Rob Fields; Jonathan X. Li; Marc L. Cohen; Vivek Natarajan; Mike Schaekermann; Alan Karthikesalingam; Adam Rodman
>
> **摘要:** Large language model (LLM)-based AI systems have shown promise for patient-facing diagnostic and management conversations in simulated settings. Translating these systems into clinical practice requires assessment in real-world workflows with rigorous safety oversight. We report a prospective, single-arm feasibility study of an LLM-based conversational AI, the Articulate Medical Intelligence Explorer (AMIE), conducting clinical history taking and presentation of potential diagnoses for patients to discuss with their provider at urgent care appointments at a leading academic medical center. 100 adult patients completed an AMIE text-chat interaction up to 5 days before their appointment. We sought to assess the conversational safety and quality, patient and clinician experience, and clinical reasoning capabilities compared to primary care providers (PCPs). Human safety supervisors monitored all patient-AMIE interactions in real time and did not need to intervene to stop any consultations based on pre-defined criteria. Patients reported high satisfaction and their attitudes towards AI improved after interacting with AMIE (p < 0.001). PCPs found AMIE's output useful with a positive impact on preparedness. AMIE's differential diagnosis (DDx) included the final diagnosis, per chart review 8 weeks post-encounter, in 90% of cases, with 75% top-3 accuracy. Blinded assessment of AMIE and PCP DDx and management (Mx) plans suggested similar overall DDx and Mx plan quality, without significant differences for DDx (p = 0.6) and appropriateness and safety of Mx (p = 0.1 and 1.0, respectively). PCPs outperformed AMIE in the practicality (p = 0.003) and cost effectiveness (p = 0.004) of Mx. While further research is needed, this study demonstrates the initial feasibility, safety, and user acceptance of conversational AI in a real-world setting, representing crucial steps towards clinical translation.
>
---
#### [replaced 017] VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.SD**

- **简介: 该论文提出VSSFlow，统一解决视频生成声音和视觉文本转语音任务。针对传统方法分离处理的不足，通过联合学习实现高效整合，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2509.24773](https://arxiv.org/pdf/2509.24773)**

> **作者:** Xin Cheng; Yuyue Wang; Xihua Wang; Yihan Wu; Kaisi Guan; Yijing Chen; Peng Zhang; Xiaojiang Liu; Meng Cao; Ruihua Song
>
> **备注:** Paper Under Review
>
> **摘要:** Video-conditioned audio generation, including Video-to-Sound (V2S) and Visual Text-to-Speech (VisualTTS), has traditionally been treated as distinct tasks, leaving the potential for a unified generative framework largely underexplored. In this paper, we bridge this gap with VSSFlow, a unified flow-matching framework that seamlessly solve both problems. To effectively handle multiple input signals within a Diffusion Transformer (DiT) architecture, we propose a disentangled condition aggregation mechanism leveraging distinct intrinsic properties of attention layers: cross-attention for semantic conditions, and self-attention for temporally-intensive conditions. Besides, contrary to the prevailing belief that joint training for the two tasks leads to performance degradation, we demonstrate that VSSFlow maintains superior performance during end-to-end joint learning process. Furthermore, we use a straightforward feature-level data synthesis method, demonstrating that our framework provides a robust foundation that easily adapts to joint sound and speech generation using synthetic data. Extensive experiments on V2S, VisualTTS and joint generation benchmarks show that VSSFlow effectively unifies these tasks and surpasses state-of-the-art domain-specific baselines, underscoring the critical potential of unified generative models. Project page: this https URL
>
---
#### [replaced 018] Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究如何从被审查的大型语言模型中提取真实知识，属于诚实性诱导与谎言检测任务，旨在解决模型产生虚假信息的问题。通过测试多种方法提升回答真实性并检测谎言。**

- **链接: [https://arxiv.org/pdf/2603.05494](https://arxiv.org/pdf/2603.05494)**

> **作者:** Helena Casademunt; Bartosz Cywiński; Khoi Tran; Arya Jakkli; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models sometimes produce false or misleading responses. Two approaches to this problem are honesty elicitation -- modifying prompts or weights so that the model answers truthfully -- and lie detection -- classifying whether a given response is false. Prior work evaluates such methods on models specifically trained to lie or conceal information, but these artificial constructions may not resemble naturally-occurring dishonesty. We instead study open-weights LLMs from Chinese developers, which are trained to censor politically sensitive topics: Qwen3 models frequently produce falsehoods about subjects like Falun Gong or the Tiananmen protests while occasionally answering correctly, indicating they possess knowledge they are trained to suppress. Using this as a testbed, we evaluate a suite of elicitation and lie detection techniques. For honesty elicitation, sampling without a chat template, few-shot prompting, and fine-tuning on generic honesty data most reliably increase truthful responses. For lie detection, prompting the censored model to classify its own responses performs near an uncensored-model upper bound, and linear probes trained on unrelated data offer a cheaper alternative. The strongest honesty elicitation techniques also transfer to frontier open-weights models including DeepSeek R1. Notably, no technique fully eliminates false responses. We release all prompts, code, and transcripts.
>
---
#### [replaced 019] Latent Speech-Text Transformer
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出LST模型，解决语音与文本模态不平衡问题，通过聚合语音标记为潜在块，提升计算效率并增强跨模态对齐。任务为语音-文本生成与理解。**

- **链接: [https://arxiv.org/pdf/2510.06195](https://arxiv.org/pdf/2510.06195)**

> **作者:** Yen-Ju Lu; Yashesh Gaur; Wei Zhou; Benjamin Muller; Jesus Villalba; Najim Dehak; Luke Zettlemoyer; Gargi Ghosh; Mike Lewis; Srinivasan Iyer; Duc Le
>
> **备注:** Accepted to ICLR 2026 (Oral)
>
> **摘要:** Auto-regressive speech-text models pre-trained on interleaved text tokens and discretized speech tokens demonstrate strong speech understanding and generation, yet remain substantially less compute-efficient than text LLMs, partly due to the much longer sequences of speech tokens relative to text. This modality imbalance disproportionately allocates pre-training and inference compute to speech, potentially hindering effective cross-modal alignment and slowing performance scaling by orders of magnitude. We introduce the Latent Speech-Text Transformer (LST), which aggregates speech tokens into latent speech patches that serve as higher-level autoregressive units. This design aligns the sequence-modeling granularity between speech and text while improving computational efficiency. The resulting patches can align with textual units to facilitate cross-modal knowledge transfer and compactly capture recurring acoustic patterns such as silence. Across story-completion benchmarks under both compute-controlled and data-controlled settings, LST consistently improves speech accuracy while also improving text performance, achieving up to +6.5% absolute gain on speech HellaSwag in compute-controlled training (+5.3% in data-controlled training). Under compute-controlled scaling from 420M to 1.8B parameters in a near compute-optimal regime, gains grow with scale, and improvements persist up to 7B parameters under fixed-token budgets. These benefits extend to downstream tasks: LST stabilizes ASR adaptation and reduces the effective autoregressive sequence length during ASR and TTS inference, lowering computational cost without degrading reconstruction quality. The code is available at this https URL.
>
---
#### [replaced 020] Reasoning Efficiently Through Adaptive Chain-of-Thought Compression: A Self-Optimizing Framework
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决链式思维（CoT）推理效率低的问题。通过提出SEER框架，实现CoT的自适应压缩，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2509.14093](https://arxiv.org/pdf/2509.14093)**

> **作者:** Kerui Huang; Shuhan Liu; Xing Hu; Tongtong Xu; Lingfeng Bao; Xin Xia
>
> **摘要:** Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by prompting intermediate steps, improving accuracy and robustness in arithmetic, logic, and commonsense tasks. However, this benefit comes with high computational costs: longer outputs increase latency, memory usage, and KV-cache demands. These issues are especially critical in software engineering tasks where concise and deterministic outputs are required. To investigate these trade-offs, we conduct an empirical study based on code generation benchmarks. The results reveal that longer CoT does not always help. Excessive reasoning often causes truncation, accuracy drops, and latency up to five times higher, with failed outputs consistently longer than successful ones. These findings challenge the assumption that longer reasoning is inherently better and highlight the need for adaptive CoT control. Motivated by this, we propose SEER (Self-Enhancing Efficient Reasoning), an adaptive framework that compresses CoT while preserving accuracy. SEER combines Best-of-N sampling with task-aware adaptive filtering, dynamically adjusting thresholds based on pre-inference outputs to reduce verbosity and computational overhead. We then evaluate SEER on three software engineering tasks and one math task. On average, SEER shortens CoT by 42.1%, improves accuracy by reducing truncation, and eliminates most infinite loops. These results demonstrate SEER as a practical method to make CoT-enhanced LLMs more efficient and robust, even under resource constraints.
>
---
#### [replaced 021] CRANE: Causal Relevance Analysis of Language-Specific Neurons in Multilingual Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CRANE框架，用于分析多语言大模型中语言特异性神经元。任务是理解语言能力在神经元层面的组织方式，解决现有方法混淆语言偏好与功能重要性的问题。通过干预神经元，验证其对特定语言预测的必要性。**

- **链接: [https://arxiv.org/pdf/2601.04664](https://arxiv.org/pdf/2601.04664)**

> **作者:** Yifan Le; Yunliang Li
>
> **备注:** 10 pages, 6 figures. Work in progress
>
> **摘要:** Multilingual large language models (LLMs) achieve strong performance across languages, yet how language capabilities are organized at the neuron level remains poorly understood. Prior work has identified language-related neurons mainly through activation-based heuristics, which conflate language preference with functional importance. We propose CRANE, a relevance-based analysis framework that redefines language specificity in terms of functional necessity, identifying language-specific neurons through targeted neuron-level interventions. CRANE characterizes neuron specialization by their contribution to language-conditioned predictions rather than activation magnitude. Our implementation will be made publicly available. Neuron-level interventions reveal a consistent asymmetric pattern: masking neurons relevant to a target language selectively degrades performance on that language while preserving performance on other languages to a substantial extent, indicating language-selective but non-exclusive neuron specializations. Experiments on English, Chinese, and Vietnamese across multiple benchmarks, together with a dedicated relevance-based metric and base-to-chat model transfer analysis, show that CRANE isolates language-specific components more precisely than activation-based methods.
>
---
#### [replaced 022] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究交互式工具使用代理的后训练问题，旨在解决多轮对话中数据合成困难和强化学习信号噪声问题。提出EigenData框架，结合自进化数据生成与验证奖励RL，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.22607](https://arxiv.org/pdf/2601.22607)**

> **作者:** Jiaxuan Gao; Jiaao Chen; Chuyi He; Shusheng Xu; Di Jin; Yi Wu
>
> **备注:** Submitted to ICML 2026
>
> **摘要:** Interactive tool-using agents must solve real-world tasks via multi-turn interaction with both humans and external environments, requiring dialogue state tracking, multi-step tool execution, while following complex instructions. Post-training such agents is challenging because synthesis for high-quality multi-turn tool-use data is difficult to scale, and reinforcement learning (RL) could face noisy signals caused by user simulation, leading to degraded training efficiency. We propose a unified framework that combines a self-evolving data agent with verifier-based RL. Our system, EigenData, is a hierarchical multi-agent engine that synthesizes tool-grounded dialogues together with executable per-instance checkers, and improves generation reliability via closed-loop self-evolving process that updates prompts and workflow. Building on the synthetic data, we develop an RL recipe that first fine-tunes the user model and then applies GRPO-style training with trajectory-level group-relative advantages and dynamic filtering, yielding consistent improvements beyond SFT. Evaluated on tau^2-bench, our best model reaches 73.0% pass^1 on Airline and 98.3% pass^1 on Telecom, matching or exceeding frontier models. Overall, our results suggest a scalable pathway for bootstrapping complex tool-using behaviors without expensive human annotation.
>
---
#### [replaced 023] DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation
- **分类: cs.CL**

- **简介: 该论文属于报告评估任务，旨在解决深度研究系统生成报告的质量评价问题。提出DEER基准，包含评估标准和声明验证机制，以提升评估的准确性和全面性。**

- **链接: [https://arxiv.org/pdf/2512.17776](https://arxiv.org/pdf/2512.17776)**

> **作者:** Janghoon Han; Heegyu Kim; Changho Lee; Dahm Lee; Min Hyung Park; Hosung Song; Stanley Jungkyu Choi; Moontae Lee; Honglak Lee
>
> **备注:** 39 pages, 10 figures, 16 tables, 123 references
>
> **摘要:** Recent advances in large language models have enabled deep research systems that generate expert-level reports through multi-step reasoning and evidence-based synthesis. However, evaluating such reports remains challenging: report quality is multifaceted, making it difficult to determine what to assess and by what criteria; LLM-based judges may miss errors that require domain expertise to identify; and because deep research relies on retrieved evidence, report-wide claim verification is also necessary. To address these issues, we propose DEER, a benchmark for evaluating expert-level deep research reports. DEER systematizes evaluation criteria with an expert-developed taxonomy (7 dimensions, 25 subdimensions) operationalized as 101 fine-grained rubric items. We also provide task-specific Expert Evaluation Guidance to support LLM-based judging. Alongside rubric-based assessment, we propose a claim verification architecture that verifies both cited and uncited claims and quantifies evidence quality. Experiments show that while current deep research systems can produce structurally plausible reports that cite external evidence, there is room for improvement in fulfilling expert-level user requests and achieving logical completeness. Beyond simple performance comparisons, DEER makes system strengths and limitations interpretable and provides diagnostic signals for improvement.
>
---
#### [replaced 024] Automatic Paper Reviewing with Heterogeneous Graph Reasoning over LLM-Simulated Reviewer-Author Debates
- **分类: cs.CL**

- **简介: 该论文属于论文评审任务，旨在解决传统方法在推理能力和公平性上的不足。通过构建异构图模型，模拟审稿人与作者的辩论过程，提升评审准确性。**

- **链接: [https://arxiv.org/pdf/2511.08317](https://arxiv.org/pdf/2511.08317)**

> **作者:** Shuaimin Li; Liyang Fan; Yufang Lin; Zeyang Li; Xian Wei; Shiwen Ni; Hamid Alinejad-Rokny; Min Yang
>
> **摘要:** Existing paper review methods often rely on superficial manuscript features or directly on large language models (LLMs), which are prone to hallucinations, biased scoring, and limited reasoning capabilities. Moreover, these methods often fail to capture the complex argumentative reasoning and negotiation dynamics inherent in reviewer-author interactions. To address these limitations, we propose ReViewGraph (Reviewer-Author Debates Graph Reasoner), a novel framework that performs heterogeneous graph reasoning over LLM-simulated multi-round reviewer-author debates. In our approach, reviewer-author exchanges are simulated through LLM-based multi-agent collaboration. Diverse opinion relations (e.g., acceptance, rejection, clarification, and compromise) are then explicitly extracted and encoded as typed edges within a heterogeneous interaction graph. By applying graph neural networks to reason over these structured debate graphs, ReViewGraph captures fine-grained argumentative dynamics and enables more informed review decisions. Extensive experiments on three datasets demonstrate that ReViewGraph outperforms strong baselines with an average relative improvement of 15.73%, underscoring the value of modeling detailed reviewer-author debate structures.
>
---
#### [replaced 025] GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics
- **分类: cs.SE; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出GateLens，用于汽车软件发布分析的LLM代理，解决复杂数据解析与推理问题。通过关系代数实现自然语言到代码的可靠转换，提升分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.21735](https://arxiv.org/pdf/2503.21735)**

> **作者:** Arsham Gholamzadeh Khoee; Shuai Wang; Robert Feldt; Dhasarathy Parthasarathy; Yinan Yu
>
> **摘要:** Ensuring reliable data-driven decisions is crucial in domains where analytical accuracy directly impacts safety, compliance, or operational outcomes. Decision support in such domains relies on large tabular datasets, where manual analysis is slow, costly, and error-prone. While Large Language Models (LLMs) offer promising automation potential, they face challenges in analytical reasoning, structured data handling, and ambiguity resolution. This paper introduces GateLens, an LLM-based architecture for reliable analysis of complex tabular data. Its key innovation is the use of Relational Algebra (RA) as a formal intermediate representation between natural-language reasoning and executable code, addressing the reasoning-to-code gap that can arise in direct generation approaches. In our automotive instantiation, GateLens translates natural language queries into RA expressions and generates optimized Python code. Unlike traditional multi-agent or planning-based systems that can be slow, opaque, and costly to maintain, GateLens emphasizes speed, transparency, and reliability. We validate the architecture in automotive software release analytics, where experimental results show that GateLens outperforms the existing Chain-of-Thought (CoT) + Self-Consistency (SC) based system on real-world datasets, particularly in handling complex and ambiguous queries. Ablation studies confirm the essential role of the RA layer. Industrial deployment demonstrates over 80% reduction in analysis time while maintaining high accuracy across domain-specific tasks. GateLens operates effectively in zero-shot settings without requiring few-shot examples or agent orchestration. This work advances deployable LLM system design by identifying key architectural features--intermediate formal representations, execution efficiency, and low configuration overhead--crucial for domain-specific analytical applications.
>
---
#### [replaced 026] Enhancing Retrieval-Augmented Generation with Entity Linking for Educational Platforms
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于教育问答任务，旨在提升RAG系统的事实准确性。通过引入实体链接和混合重排序策略，解决专业领域中的术语歧义问题。**

- **链接: [https://arxiv.org/pdf/2512.05967](https://arxiv.org/pdf/2512.05967)**

> **作者:** Francesco Granata; Francesco Poggi; Misael Mongiovì
>
> **摘要:** In the era of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) architectures are gaining significant attention for their ability to ground language generation in reliable knowledge sources. Despite their effectiveness, RAG systems based solely on semantic similarity often fail to ensure factual accuracy in specialized domains, where terminological ambiguity can affect retrieval relevance. This study proposes ELERAG, an enhanced RAG architecture that integrates a factual signal derived from Entity Linking to improve the accuracy of educational question-answering systems in Italian. The system includes a Wikidata-based Entity Linking module and implements a hybrid re-ranking strategy based on Reciprocal Rank Fusion (RRF). To validate our approach, we compared it against standard baselines and state-of-the-art methods, including a Weighted-Score Re-ranking, a standalone Cross-Encoder and a combined RRF+Cross-Encoder pipeline. Experiments were conducted on two benchmarks: a custom academic dataset and the standard SQuAD-it dataset. Results show that, in domain-specific contexts, ELERAG significantly outperforms both the baseline and the Cross-Encoder configurations. Conversely, the Cross-Encoder approaches achieve the best results on the general-domain dataset. These findings provide strong experimental evidence of the domain mismatch effect, highlighting the importance of domain-adapted hybrid strategies to enhance factual precision in educational RAG systems without relying on computationally expensive models trained on disparate data distributions. They also demonstrate the potential of entity-aware RAG systems in educational environments, fostering adaptive and reliable AI-based tutoring tools.
>
---
#### [replaced 027] AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors
- **分类: cs.CL**

- **简介: 该论文提出AuditBench，用于评估模型对隐藏行为的对齐审计。任务是检测模型中的隐性行为，解决如何有效审计模型的问题，通过构建多样模型和审计工具进行实验。**

- **链接: [https://arxiv.org/pdf/2602.22755](https://arxiv.org/pdf/2602.22755)**

> **作者:** Abhay Sheshadri; Aidan Ewart; Kai Fronsdal; Isha Gupta; Samuel R. Bowman; Sara Price; Samuel Marks; Rowan Wang
>
> **摘要:** We introduce AuditBench, an alignment auditing benchmark. AuditBench consists of 56 language models with implanted hidden behaviors. Each model has one of 14 concerning behaviors--such as sycophantic deference, opposition to AI regulation, or secret geopolitical loyalties--which it does not confess to when directly asked. AuditBench models are highly diverse--some are subtle, while others are overt, and we use varying training techniques both for implanting behaviors and training models not to confess. To demonstrate AuditBench's utility, we develop an investigator agent that autonomously employs a configurable set of auditing tools. By measuring investigator agent success using different tools, we can evaluate their efficacy. Notably, we observe a tool-to-agent gap, where tools that perform well in standalone non-agentic evaluations fail to translate into improved performance when used with our investigator agent. We find that our most effective tools involve scaffolded calls to auxiliary models that generate diverse prompts for the target. White-box interpretability tools can be helpful, but the agent performs best with black-box tools. We also find that audit success varies greatly across training techniques: models trained on synthetic documents are easier to audit than models trained on demonstrations, with better adversarial training further increasing auditing difficulty. We release our models, agent, and evaluation framework to support future quantitative, iterative science on alignment auditing.
>
---
#### [replaced 028] PonderLM-3: Adaptive Token-Wise Pondering with Differentiable Masking
- **分类: cs.CL**

- **简介: 该论文提出PonderLM-3，解决模型推理中如何高效分配计算资源的问题。通过自监督学习实现逐token的自适应计算，提升生成质量并减少冗余计算。**

- **链接: [https://arxiv.org/pdf/2603.02023](https://arxiv.org/pdf/2603.02023)**

> **作者:** He Li; Feichen Song; Boyi Zeng; Shixiang Song; Zhiqin John Xu; Ziwei He; Zhouhan Lin
>
> **摘要:** Test-time scaling has shown that allocating more additional computation at inference can improve generation quality, motivating a natural follow-up question: where should this computation be spent? Building on this insight, we introduce PonderLM-3, a pretraining framework for token-wise adaptive pondering that learns to selectively allocate additional computation under purely self-supervised objectives, built on top of the PonderLM-2 backbone. This makes additional inference computation an allocatable per-token resource, so tokens receive more computation only when it is beneficial, rather than paying a uniform extra cost. To make this allocation learnable while maintaining train-inference consistency, PonderLM-3 injects a differentiable attention mask during pretraining and pairs it with a matching hard pruning rule at inference. PonderLM-3 defines a stronger Pareto frontier: compared with existing recursive or adaptive baselines, it achieves lower pretraining perplexity at equal inference FLOPs. On downstream benchmarks, PonderLM-3 attains comparable performance to fixed-step PonderLM-2 under the same maximum number of additional computation steps, while using fewer inference FLOPs in practice. Overall, PonderLM-3 provides an end-to-end differentiable and train-inference consistent framework for token-wise adaptive computation, enabling additional inference compute to be allocated where it is most useful rather than paid uniformly by every token.
>
---
#### [replaced 029] SkillCraft: Can LLM Agents Learn to Use Tools Skillfully?
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出SkillCraft基准，用于评估大模型代理在复杂任务中学习并复用高级工具组合的能力。旨在解决工具使用中技能抽象与复用的问题。**

- **链接: [https://arxiv.org/pdf/2603.00718](https://arxiv.org/pdf/2603.00718)**

> **作者:** Shiqi Chen; Jingze Gai; Ruochen Zhou; Jinghan Zhang; Tongyao Zhu; Junlong Li; Kangrui Wang; Zihan Wang; Zhengyu Chen; Klara Kaleb; Ning Miao; Siyang Gao; Cong Lu; Manling Li; Junxian He; Yee Whye Teh
>
> **备注:** 21 pages. Code: this https URL ; Project page: this https URL
>
> **摘要:** Real-world tool-using agents operate over long-horizon workflows with recurring structure and diverse demands, where effective behavior requires not only invoking atomic tools but also abstracting, and reusing higher-level tool compositions. However, existing benchmarks mainly measure instance-level success under static tool sets, offering limited insight into agents' ability to acquire such reusable skills. We address this gap by introducing SkillCraft, a benchmark explicitly stress-test agent ability to form and reuse higher-level tool compositions, where we call Skills. SkillCraft features realistic, highly compositional tool-use scenarios with difficulty scaled along both quantitative and structural dimensions, designed to elicit skill abstraction and cross-task reuse. We further propose a lightweight evaluation protocol that enables agents to auto-compose atomic tools into executable Skills, cache and reuse them inside and across tasks, thereby improving efficiency while accumulating a persistent library of reusable skills. Evaluating state-of-the-art agents on SkillCraft, we observe substantial efficiency gains, with token usage reduced by up to 80% by skill saving and reuse. Moreover, success rate strongly correlates with tool composition ability at test time, underscoring compositional skill acquisition as a core capability.
>
---
#### [replaced 030] NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出NavSpace基准，评估导航代理的空间智能。针对传统基准忽视空间感知的问题，设计任务集并测试多个模型，提出SNav模型提升导航性能。**

- **链接: [https://arxiv.org/pdf/2510.08173](https://arxiv.org/pdf/2510.08173)**

> **作者:** Haolin Yang; Yuxing Long; Zhuoyuan Yu; Zihan Yang; Minghan Wang; Jiapeng Xu; Yihan Wang; Ziyan Yu; Wenzhe Cai; Lei Kang; Hao Dong
>
> **备注:** ICRA 2026
>
> **摘要:** Instruction-following navigation is a key step toward embodied intelligence. Prior benchmarks mainly focus on semantic understanding but overlook systematically evaluating navigation agents' spatial perception and reasoning capabilities. In this work, we introduce the NavSpace benchmark, which contains six task categories and 1,228 trajectory-instruction pairs designed to probe the spatial intelligence of navigation agents. On this benchmark, we comprehensively evaluate 22 navigation agents, including state-of-the-art navigation models and multimodal large language models. The evaluation results lift the veil on spatial intelligence in embodied navigation. Furthermore, we propose SNav, a new spatially intelligent navigation model. SNav outperforms existing navigation agents on NavSpace and real robot tests, establishing a strong baseline for future work.
>
---
#### [replaced 031] Image Captioning via Compact Bidirectional Architecture
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像描述任务，旨在解决传统模型仅单向生成 captions 的问题。提出一种紧凑的双向 Transformer 模型，可并行利用双向上下文，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2201.01984](https://arxiv.org/pdf/2201.01984)**

> **作者:** Zijie Song; Yuanen Zhou; Zhenzhen Hu; Daqing Liu; Huixia Ben; Richang Hong; Meng Wang
>
> **摘要:** Most current image captioning models typically generate captions from left-to-right. This unidirectional property makes them can only leverage past context but not future context. Though refinement-based models can exploit both past and future context by generating a new caption in the second stage based on pre-retrieved or pre-generated captions in the first stage, the decoder of these models generally consists of two networks~(i.e. a retriever or captioner in the first stage and a captioner in the second stage), which can only be executed sequentially. In this paper, we introduce a Compact Bidirectional Transformer model for image captioning that can leverage bidirectional context implicitly and explicitly while the decoder can be executed parallelly. Specifically, it is implemented by tightly coupling left-to-right(L2R) and right-to-left(R2L) flows into a single compact model to serve as a regularization for implicitly exploiting bidirectional context and optionally allowing explicit interaction of the bidirectional flows, while the final caption is chosen from either L2R or R2L flow in a sentence-level ensemble manner. We conduct extensive ablation studies on MSCOCO benchmark and find that the compact bidirectional architecture and the sentence-level ensemble play more important roles than the explicit interaction mechanism. By combining with word-level ensemble seamlessly, the effect of sentence-level ensemble is further enlarged. We further extend the conventional one-flow self-critical training to the two-flows version under this architecture and achieve new state-of-the-art results in comparison with non-vision-language-pretraining models. Finally, we verify the generality of this compact bidirectional architecture by extending it to LSTM backbone. Source code is available at this https URL.
>
---
#### [replaced 032] AlphaApollo: A System for Deep Agentic Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AlphaApollo系统，解决基础模型推理能力不足与测试时演化不可靠的问题，通过多轮代理推理、学习和进化提升数学推理性能。**

- **链接: [https://arxiv.org/pdf/2510.06261](https://arxiv.org/pdf/2510.06261)**

> **作者:** Zhanke Zhou; Chentao Cao; Xiao Feng; Xuan Li; Zongze Li; Xiangyu Lu; Jiangchao Yao; Weikai Huang; Tian Cheng; Jianghangfan Zhang; Tangyu Jiang; Linrui Xu; Yiming Zheng; Brando Miranda; Tongliang Liu; Sanmi Koyejo; Masashi Sugiyama; Bo Han
>
> **备注:** Ongoing project
>
> **摘要:** We present AlphaApollo, an agentic reasoning system that targets two bottlenecks in foundation-model reasoning: (1) limited reasoning capacity for complex, long-horizon problem solving and (2) unreliable test-time evolution without trustworthy verification. AlphaApollo orchestrates models and tools via three components: (i) multi-turn agentic reasoning, which formalizes model-environment interaction with structured tool calls and responses; (ii) multi-turn agentic learning, which applies turn-level reinforcement learning to optimize tool-use reasoning while decoupling actions from tool responses for stable training; and (iii) multi-round agentic evolution, which refines solutions through a propose-judge-update loop with tool-assisted verifications and long-horizon memory. Across seven math reasoning benchmarks and multiple model scales, AlphaApollo improves performance through reliable tool use (> 85% tool-call success), substantial gains from multi-turn RL (Avg@32: Qwen2.5-1.5B-Instruct 1.07% -> 9.64%, Qwen2.5-7B-Instruct 8.77% -> 20.35%), and improvements from evolution (e.g., Qwen2.5-3B-Instruct 5.27% -> 7.70%, Qwen2.5-14B-Instruct 16.53% -> 21.08%). This project is still ongoing. We welcome feedback from the community and will frequently update the source code and technical report.
>
---
#### [replaced 033] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios
- **分类: cs.CL**

- **简介: 该论文提出AgentCoMa基准，用于测试模型在真实场景中结合常识与数学推理的能力。针对现有基准仅侧重单一推理类型的问题，研究分析了61个LLM在混合任务中的表现，发现模型在组合任务中准确率显著下降。**

- **链接: [https://arxiv.org/pdf/2508.19988](https://arxiv.org/pdf/2508.19988)**

> **作者:** Lisa Alazraki; Lihu Chen; Ana Brassard; Joe Stacey; Hossein A. Rahmani; Marek Rei
>
> **摘要:** Large Language Models (LLMs) have achieved high accuracy on complex commonsense and mathematical problems that involve the composition of multiple reasoning steps. However, current compositional benchmarks testing these skills tend to focus on either commonsense or math reasoning, whereas LLM agents solving real-world tasks would require a combination of both. In this work, we introduce an Agentic Commonsense and Math benchmark (AgentCoMa), where each compositional task requires a commonsense reasoning step and a math reasoning step. We test it on 61 LLMs of different sizes, model families, and training strategies. We find that LLMs can usually solve both steps in isolation, yet their accuracy drops by ~30% on average when the two are combined. This is a substantially greater performance gap than the one we observe in prior compositional benchmarks that combine multiple steps of the same reasoning type. In contrast, non-expert human annotators can solve the compositional questions and the individual steps in AgentCoMa with similarly high accuracy. Furthermore, we conduct a series of interpretability studies to better understand the performance gap, examining neuron patterns, attention maps and membership inference. Our work underscores a substantial degree of model brittleness in the context of mixed-type compositional reasoning and offers a test bed for future improvement.
>
---
#### [replaced 034] Fanar-Sadiq: A Multi-Agent Architecture for Grounded Islamic QA
- **分类: cs.CL**

- **简介: 该论文属于宗教问答任务，旨在解决伊斯兰教知识问答中缺乏依据和准确性的问题。提出多智能体系统Fanar-Sadiq，支持精准引用、法律计算和多语言处理。**

- **链接: [https://arxiv.org/pdf/2603.08501](https://arxiv.org/pdf/2603.08501)**

> **作者:** Ummar Abbas; Mourad Ouzzani; Mohamed Y. Eltabakh; Omar Sinan; Gagan Bhatia; Hamdy Mubarak; Majd Hawasly; Mohammed Qusay Hashim; Kareem Darwish; Firoj Alam
>
> **摘要:** Large language models (LLMs) can answer religious knowledge queries fluently, yet they often hallucinate and misattribute sources, which is especially consequential in Islamic settings where users expect grounding in canonical texts (Qur'an and Hadith) and jurisprudential (fiqh) nuance. Retrieval-augmented generation (RAG) reduces some of these limitations by grounding generation in external evidence. However, a single ``retrieve-then-generate'' pipeline is limited to deal with the diversity of Islamic queries. Users may request verbatim scripture, fatwa-style guidance with citations or rule-constrained computations such as zakat and inheritance that require strict arithmetic and legal invariants. In this work, we present a bilingual (Arabic/English) multi-agent Islamic assistant, called Fanar-Sadiq, which is a core component of the Fanar AI platform. Fanar-Sadiq routes Islamic-related queries to specialized modules within an agentic, tool-using architecture. The system supports intent-aware routing, retrieval-grounded fiqh answers with deterministic citation normalization and verification traces, exact verse lookup with quotation validation, and deterministic calculators for Sunni zakat and inheritance with madhhab-sensitive branching. We evaluate the complete end-to-end system on public Islamic QA benchmarks and demonstrate effectiveness and efficiency. Our system is currently publicly and freely accessible through API and a Web application, and has been accessed $\approx$1.9M times in less than a year.
>
---
#### [replaced 035] EVM-QuestBench: An Execution-Grounded Benchmark for Natural-Language Transaction Code Generation
- **分类: cs.CL**

- **简介: 该论文提出EVM-QuestBench，用于评估自然语言生成以太坊虚拟机交易代码的准确性与安全性。解决代码生成中的执行精度问题，通过动态测试验证模型性能。**

- **链接: [https://arxiv.org/pdf/2601.06565](https://arxiv.org/pdf/2601.06565)**

> **作者:** Pei Yang; Wanyi Chen; Ke Wang; Lynn Ai; Eric Yang; Tianyu Shi
>
> **备注:** 10 pages, 13 figures
>
> **摘要:** Large language models are increasingly applied to various development scenarios. However, in on-chain transaction scenarios, even a minor error can cause irreversible loss for users. Existing evaluations often overlook execution accuracy and safety. We introduce EVM-QuestBench, an execution-grounded benchmark for natural-language transaction-script generation on EVM-compatible chains. The benchmark employs dynamic evaluation: instructions are sampled from template pools, numeric parameters are drawn from predefined intervals, and validators verify outcomes against these instantiated values. EVM-QuestBench contains 107 tasks (62 atomic, 45 composite). Its modular architecture enables rapid task development. The runner executes scripts on a forked EVM chain with snapshot isolation; composite tasks apply step-efficiency decay. We evaluate 20 models and find large performance gaps, with split scores revealing persistent asymmetry between single-action precision and multi-step workflow completion. Code: this https URL.
>
---
#### [replaced 036] Does Scientific Writing Converge to U.S. English? Evidence from Generative AI-Assisted Publications
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于语言与科技交叉研究，旨在探讨生成式AI是否使非英语国家的科学写作趋同于美式英语。通过分析大量科学论文，发现GenAI辅助写作显著缩小了语言风格差距。**

- **链接: [https://arxiv.org/pdf/2511.11687](https://arxiv.org/pdf/2511.11687)**

> **作者:** Dragan Filimonovic; Christian Rutzer; Jeffrey Macher; Rolf Weder
>
> **摘要:** A growing literature documents that generative artificial intelligence (GenAI) is changing scientific writing, yet most studies focus on absolute changes in vocabulary or readability. An important question remains unanswered: Does GenAI use lead to systematic convergence, or a narrowing of stylistic gaps relative to the dominant form of scientific English? Unlike absolute changes, convergence signals whether language-related publication barriers are declining and suggests broader implications for participation and competition in global science. This study directly addresses this question using 5.65 million English-language scientific articles published from 2021 to 2024 and indexed in Scopus. We measure linguistic similarity to a U.S. benchmark corpus using SciBERT text embeddings, and estimate dynamic changes using an event-study difference-in-differences design with repeated cross-sections centered on the late-2022 release of ChatGPT. We find that GenAI-assisted publications from non-English-speaking countries exhibit statistically significant and increasing convergence toward U.S. scientific English, relative to non-GenAI-assisted publications from these countries. This effect is strongest for domestic author teams from countries more linguistically distant from English and for articles published in lower-impact journals -- precisely the contexts where language barriers have historically been most consequential. The results suggest that GenAI tools are reducing language-related barriers in scientific publications. Whether this represents genuine inclusion or a deepening dependence on a single linguistic standard remains an open question.
>
---
#### [replaced 037] Stepwise Guided Policy Optimization: Coloring your Incorrect Reasoning in GRPO
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决GRPO在所有错误回复情况下无法更新策略的问题。通过引入分步裁判模型提升响应多样性，提出SGPO方法，增强模型学习能力。**

- **链接: [https://arxiv.org/pdf/2505.11595](https://arxiv.org/pdf/2505.11595)**

> **作者:** Peter Chen; Xiaopeng Li; Ziniu Li; Xi Chen; Tianyi Lin
>
> **备注:** Accepted by TMLR; 47 pages
>
> **摘要:** Reinforcement learning (RL) has proven effective in strengthening the reasoning capabilities of large language models (LLMs). A widely adopted method, Group Relative Policy Optimization (GRPO), has shown strong empirical results in training recent reasoning models, but it fails to update the policy when all responses within a group are incorrect (i.e., all-negative-sample groups). This limitation highlights a gap between artificial and human intelligence: unlike humans, who can learn from mistakes, GRPO discards these failure signals. We introduce a simple framework to mitigate the all-negative-sample issue by incorporating response diversity within groups using a step-wise judge model, which can be trained directly or adapted from existing LLMs. In a simplified setting, we prove that this diversification accelerates GRPO's learning dynamics. We then empirically validate Stepwise Guided Policy Optimization (SGPO) across model sizes (7B, 14B, 32B) in both offline and online training on nine reasoning benchmarks (including base and distilled variants). Overall, SGPO improves average performance and is effective in early and mid-training when all-negative groups are prevalent, while improvements are not uniform across every benchmark and depend on the structure and informativeness of negative samples. Finally, SGPO does not require the judge model to generate correct solutions, distinguishing it from knowledge distillation methods.
>
---
#### [replaced 038] SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge
- **分类: cs.CL**

- **简介: 该论文提出SimpleQA Verified，一个用于评估大语言模型事实性的基准。解决原基准的标签噪声、主题偏差等问题，通过多阶段筛选提升可靠性，推动模型准确性和减少幻觉。**

- **链接: [https://arxiv.org/pdf/2509.07968](https://arxiv.org/pdf/2509.07968)**

> **作者:** Lukas Haas; Gal Yona; Giovanni D'Antonio; Sasha Goldshtein; Dipanjan Das
>
> **摘要:** We introduce SimpleQA Verified, a 1,000-prompt benchmark for evaluating Large Language Model (LLM) short-form factuality based on OpenAI's SimpleQA. It addresses critical limitations in OpenAI's benchmark, including noisy and incorrect labels, topical biases, and question redundancy. SimpleQA Verified was created through a rigorous multi-stage filtering process involving de-duplication, topic balancing, and source reconciliation to produce a more reliable and challenging evaluation set, alongside improvements in the autorater prompt. On this new benchmark, Gemini 2.5 Pro achieves a state-of-the-art F1-score of 55.6, outperforming other frontier models, including GPT-5. This work provides the research community with a higher-fidelity tool to track genuine progress in parametric model factuality and to mitigate hallucinations. The benchmark dataset, evaluation code, and leaderboard are available at: this https URL.
>
---
#### [replaced 039] TaoSR1: The Thinking Model for E-commerce Relevance Search
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对电商搜索中的查询-商品相关性预测任务，提出TaoSR1框架，解决LLM推理误差累积、幻觉及部署难题，通过多阶段优化提升效果。**

- **链接: [https://arxiv.org/pdf/2508.12365](https://arxiv.org/pdf/2508.12365)**

> **作者:** Chenhe Dong; Shaowei Yao; Pengkun Jiao; Jianhui Yang; Yiming Jin; Zerui Huang; Xiaojiang Zhou; Dan Ou; Haihong Tang; Bo Zheng
>
> **摘要:** Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification.
>
---
#### [replaced 040] Query-focused and Memory-aware Reranker for Long Context Processing
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决长文本相关性排序问题。提出一种轻量级的重排序框架，通过注意力机制提升排序效果。**

- **链接: [https://arxiv.org/pdf/2602.12192](https://arxiv.org/pdf/2602.12192)**

> **作者:** Yuqing Li; Jiangnan Li; Mo Yu; Guoxuan Ding; Zheng Lin; Weiping Wang; Jie Zhou
>
> **备注:** work in progress
>
> **摘要:** Built upon the existing analysis of retrieval heads in large language models, we propose an alternative reranking framework that trains models to estimate passage-query relevance using the attention scores of selected heads. This approach provides a listwise solution that leverages holistic information within the entire candidate shortlist during ranking. At the same time, it naturally produces continuous relevance scores, enabling training on arbitrary retrieval datasets without requiring Likert-scale supervision. Our framework is lightweight and effective, requiring only small-scale models (e.g., 4B parameters) to achieve strong performance. Extensive experiments demonstrate that our method outperforms existing state-of-the-art pointwise and listwise rerankers across multiple domains, including Wikipedia and long narrative datasets. It further establishes a new state-of-the-art on the LoCoMo benchmark that assesses the capabilities of dialogue understanding and memory usage. We further demonstrate that our framework supports flexible extensions. For example, augmenting candidate passages with contextual information further improves ranking accuracy, while training attention heads from middle layers enhances efficiency without sacrificing performance.
>
---
#### [replaced 041] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models
- **分类: cs.CL**

- **简介: 该论文提出SynthWorlds框架，用于分离语言模型的推理与知识记忆能力。针对评估语言模型推理能力时知识干扰的问题，构建真实与合成世界平行数据集，设计镜像任务进行对比实验，揭示知识优势差距。**

- **链接: [https://arxiv.org/pdf/2510.24427](https://arxiv.org/pdf/2510.24427)**

> **作者:** Ken Gu; Advait Bhat; Mike A Merrill; Robert West; Xin Liu; Daniel McDuff; Tim Althoff
>
> **备注:** ICLR 2026
>
> **摘要:** Evaluating the reasoning ability of language models (LMs) is complicated by their extensive parametric world knowledge, where benchmark performance often reflects factual recall rather than genuine reasoning. Existing datasets and approaches (e.g., temporal filtering, paraphrasing, adversarial substitution) cannot cleanly separate the two. We present SynthWorlds, a framework that disentangles task reasoning complexity from factual knowledge. In SynthWorlds, we construct parallel corpora representing two worlds with identical interconnected structure: a real-mapped world, where models may exploit parametric knowledge, and a synthetic-mapped world, where such knowledge is meaningless. On top of these corpora, we design two mirrored tasks as case studies: multi-hop question answering and page navigation, which maintain equal reasoning difficulty across worlds. Experiments in parametric-only (e.g., closed-book QA) and knowledge-augmented (e.g., retrieval-augmented) LM settings reveal a persistent knowledge advantage gap, defined as the performance boost models gain from memorized parametric world knowledge. Knowledge acquisition and integration mechanisms reduce but do not eliminate this gap, highlighting opportunities for system improvements. Fully automatic and scalable, SynthWorlds provides a controlled environment for evaluating LMs in ways that were previously challenging, enabling precise and testable comparisons of reasoning and memorization.
>
---
#### [replaced 042] A Causal Graph Approach to Oppositional Narrative Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于反对性叙事分析任务，旨在解决传统方法依赖预定义本体和线性模式识别的问题。通过构建实体交互图并引入因果估计，提出一种更结构化的分析与分类方法。**

- **链接: [https://arxiv.org/pdf/2603.06135](https://arxiv.org/pdf/2603.06135)**

> **作者:** Diego Revilla; Martin Fernandez-de-Retana; Lingfeng Chen; Aritz Bilbao-Jayo; Miguel Fernandez-de-Retana
>
> **摘要:** Current methods for textual analysis rely on data annotated within predefined ontologies, often embedding human bias within black-box models. Despite achieving near-perfect performance, these approaches exploit unstructured, linear pattern recognition rather than modeling the structured interactions between entities that naturally emerge in discourse. In this work, we propose a graph-based framework for the detection, analysis, and classification of oppositional narratives and their underlying entities by representing narratives as entity-interaction graphs. Moreover, by incorporating causal estimation at the node level, our approach derives a causal representation of each contribution to the final classification by distilling the constructed sentence graph into a minimal causal subgraph. Building upon this representation, we introduce a classification pipeline that outperforms existing approaches to oppositional thinking classification task.
>
---
#### [replaced 043] Rethinking Discrete Speech Representation Tokens for Accent Generation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音生成任务，研究DSRT中口音信息的编码问题。通过新评估框架分析不同语音表示，发现层选择、ASR监督和代码本大小对口音信息的影响。**

- **链接: [https://arxiv.org/pdf/2601.19786](https://arxiv.org/pdf/2601.19786)**

> **作者:** Jinzuomu Zhong; Yi Wang; Korin Richmond; Peter Bell
>
> **摘要:** Discrete Speech Representation Tokens (DSRTs) have become a foundational component in speech generation. While prior work has extensively studied phonetic and speaker information in DSRTs, how accent information is encoded in DSRTs remains largely unexplored. In this paper, we present the first systematic investigation of accent information in DSRTs. We propose a unified evaluation framework that measures both accessibility of accent information via a novel Accent ABX task and recoverability via cross-accent Voice Conversion (VC) resynthesis. Using this framework, we analyse DSRTs derived from several widely used speech representations. Our results reveal that: (1) choice of layers has the most significant impact on retaining accent information, (2) accent information is substantially reduced by ASR supervision; (3) naive codebook size reduction cannot effectively disentangle accent from phonetic and speaker information.
>
---
#### [replaced 044] Adaptive Loops and Memory in Transformers: Think Harder or Know More?
- **分类: cs.CL**

- **简介: 该论文研究增强Transformer模型的推理能力，解决传统方法依赖显式步骤的问题。通过引入自适应循环和门控记忆，提升数学和常识任务表现。**

- **链接: [https://arxiv.org/pdf/2603.08391](https://arxiv.org/pdf/2603.08391)**

> **作者:** Markus Frey; Behzad Shomali; Ali Hamza Bashir; David Berghaus; Mehdi Ali
>
> **备注:** Published at Latent & Implicit Thinking Workshop @ ICLR 2026
>
> **摘要:** Chain-of-thought (CoT) prompting enables reasoning in language models but requires explicit verbalization of intermediate steps. Looped transformers offer an alternative by iteratively refining representations within hidden states. This parameter efficiency comes at a cost, as looped models lack the storage capacity of deeper models which use unique weights per layer. In this work, we investigate transformer models that feature both adaptive per-layer looping, where each transformer block learns to iterate its hidden state via a learned halting mechanism, and gated memory banks, that provide additional learned storage. We find that looping primarily benefits mathematical reasoning, while memory banks help recover performance on commonsense tasks compared to parameter and FLOP matched models. Combining both mechanisms yields a model that outperforms an iso-FLOP baseline, with three times the number of layers, across math benchmarks. Analysis of model internals reveals layer specialization: early layers learn to loop minimally and access memory sparingly, while later layers do both more heavily.
>
---
#### [replaced 045] ThinkQE: Query Expansion via an Evolving Thinking Process
- **分类: cs.IR; cs.CL**

- **简介: 论文提出ThinkQE，用于网页搜索中的查询扩展任务，解决现有方法扩展结果过于狭窄的问题。通过深度语义探索和迭代反馈优化，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2506.09260](https://arxiv.org/pdf/2506.09260)**

> **作者:** Yibin Lei; Tao Shen; Andrew Yates
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Effective query expansion for web search benefits from promoting both exploration and result diversity to capture multiple interpretations and facets of a query. While recent LLM-based methods have improved retrieval performance and demonstrate strong domain generalization without additional training, they often generate narrowly focused expansions that overlook these desiderata. We propose ThinkQE, a test-time query expansion framework addressing this limitation through two key components: a thinking-based expansion process that encourages deeper and comprehensive semantic exploration, and a corpus-interaction strategy that iteratively refines expansions using retrieval feedback from the corpus. Experiments on diverse web search benchmarks (DL19, DL20, and BRIGHT) show ThinkQE consistently outperforms prior approaches, including training-intensive dense retrievers and rerankers.
>
---
#### [replaced 046] Quantifying Genuine Awareness in Hallucination Prediction Beyond Question-Side Shortcuts
- **分类: cs.CL**

- **简介: 论文属于语言模型幻觉检测任务，旨在解决现有方法过度依赖问题侧信息而非真实知识的问题，提出AQE方法量化问题侧影响。**

- **链接: [https://arxiv.org/pdf/2509.15339](https://arxiv.org/pdf/2509.15339)**

> **作者:** Yeongbin Seo; Dongha Lee; Jinyoung Yeo
>
> **摘要:** Many works have proposed methodologies for language model (LM) hallucination detection and reported seemingly strong performance. However, we argue that the reported performance to date reflects not only a model's genuine awareness of its internal information, but also awareness derived purely from question-side information (e.g., benchmark hacking). While benchmark hacking can be effective for boosting hallucination detection score on existing benchmarks, it does not generalize to out-of-domain settings and practical usage. Nevertheless, disentangling how much of a model's hallucination detection performance arises from question-side awareness is non-trivial. To address this, we propose a methodology for measuring this effect without requiring human labor, Approximate Question-side Effect (AQE). Our analysis using AQE reveals that existing hallucination detection methods rely heavily on benchmark hacking.
>
---
#### [replaced 047] Connecting Voices: LoReSpeech as a Low-Resource Speech Parallel Corpus
- **分类: cs.CL**

- **简介: 该论文提出LoReSpeech，解决低资源语言语音平行语料稀缺问题，通过构建语音对齐语料库，支持语音识别和翻译，促进语言保护与数字包容。**

- **链接: [https://arxiv.org/pdf/2502.18215](https://arxiv.org/pdf/2502.18215)**

> **作者:** Samy Ouzerrout
>
> **备注:** This paper is withdrawn because the LoReSpeech dataset described in Section 2 is not currently available, which affects the reproducibility of the work and the validity of the experimental results
>
> **摘要:** Aligned audio corpora are fundamental to NLP technologies such as ASR and speech translation, yet they remain scarce for underrepresented languages, hindering their technological integration. This paper introduces a methodology for constructing LoReSpeech, a low-resource speech-to-speech translation corpus. Our approach begins with LoReASR, a sub-corpus of short audios aligned with their transcriptions, created through a collaborative platform. Building on LoReASR, long-form audio recordings, such as biblical texts, are aligned using tools like the MFA. LoReSpeech delivers both intra- and inter-language alignments, enabling advancements in multilingual ASR systems, direct speech-to-speech translation models, and linguistic preservation efforts, while fostering digital inclusivity. This work is conducted within Tutlayt AI project (this https URL).
>
---
#### [replaced 048] TableMind++: An Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning
- **分类: cs.CL**

- **简介: 该论文属于表格推理任务，旨在解决传统方法在上下文溢出和数值敏感性上的不足。通过引入不确定性感知框架，改进了程序化代理的推理能力。**

- **链接: [https://arxiv.org/pdf/2603.07528](https://arxiv.org/pdf/2603.07528)**

> **作者:** Mingyue Cheng; Shuo Yu; Chuang Jiang; Xiaoyu Tao; Qingyang Mao; Jie Ouyang; Qi Liu; Enhong Chen
>
> **备注:** 6 tables, 9 figures. arXiv admin note: text overlap with arXiv:2509.06278
>
> **摘要:** Table reasoning requires models to jointly perform semantic understanding and precise numerical operations. Most existing methods rely on a single-turn reasoning paradigm over tables which suffers from context overflow and weak numerical sensitivity. To address these limitations, we previously proposed TableMind as a tuning-based autonomous programmatic agent that simulates human-like interaction within a lightweight large language model (LLM). TableMind internalizes planning, action, and reflection through a two-stage training strategy involving supervised fine-tuning (SFT) on filtered high-quality data and reinforcement learning (RL) via a multi-perspective reward and the Rank-Aware Policy Optimization (RAPO) algorithm. While TableMind establishes a solid foundation for programmatic agents, the inherent stochasticity of LLMs remains a critical challenge that leads to hallucinations. In this paper, we extend this foundation to TableMind++ by introducing a novel uncertainty-aware inference framework to mitigate hallucinations. Specifically, we propose memory-guided plan pruning to retrieve historical trajectories for validating and filtering out logically flawed plans to address epistemic uncertainty. To ensure execution precision, we introduce confidence-based action refinement which monitors token-level probabilities to detect and self-correct syntactic noise for aleatoric uncertainty mitigation. Finally, we employ dual-weighted trajectory aggregation to synthesize a robust consensus from multiple reasoning paths. Extensive experiments on diverse benchmarks demonstrate that TableMind++ consistently outperforms previous baselines and proprietary models to validate the effectiveness of integrating autonomous training with uncertainty quantification. Our code is available.
>
---
#### [replaced 049] Scalable Training of Mixture-of-Experts Models with Megatron Core
- **分类: cs.DC; cs.CL; cs.LG**

- **简介: 该论文属于深度学习任务，解决MoE模型训练中的系统挑战，通过优化内存、通信和计算等多方面实现高效可扩展训练。**

- **链接: [https://arxiv.org/pdf/2603.07685](https://arxiv.org/pdf/2603.07685)**

> **作者:** Zijie Yan; Hongxiao Bai; Xin Yao; Dennis Liu; Tong Liu; Hongbin Liu; Pingtian Li; Evan Wu; Shiqing Fan; Li Tao; Robin Zhang; Yuzhong Wang; Shifang Xu; Jack Chang; Xuwen Chen; Kunlun Li; Yan Bai; Gao Deng; Nan Zheng; Vijay Anand Korthikanti; Abhinav Khattar; Ethan He; Soham Govande; Sangkug Lym; Zhongbo Zhu; Qi Zhang; Haochen Yuan; Xiaowei Ren; Deyu Fu; Tailai Ma; Shunkang Zhang; Jiang Shao; Ray Wang; Vasudevan Rengasamy; Rachit Garg; Santosh Bhavani; Xipeng Li; Chandler Zhou; David Wu; Yingcan Wei; Ashwath Aithal; Michael Andersch; Mohammad Shoeybi; Jiajie Yao; June Yang
>
> **备注:** Technical Report. 88 pages. 42 figures
>
> **摘要:** Scaling Mixture-of-Experts (MoE) training introduces systems challenges absent in dense models. Because each token activates only a subset of experts, this sparsity allows total parameters to grow much faster than per-token computation, creating coupled constraints across memory, communication, and computation. Optimizing one dimension often shifts pressure to another, demanding co-design across the full system stack. We address these challenges for MoE training through integrated optimizations spanning memory (fine-grained recomputation, offloading, etc.), communication (optimized dispatchers, overlapping, etc.), and computation (Grouped GEMM, fusions, CUDA Graphs, etc.). The framework also provides Parallel Folding for flexible multi-dimensional parallelism, low-precision training support for FP8 and NVFP4, and efficient long-context training. On NVIDIA GB300 and GB200, it achieves 1,233/1,048 TFLOPS/GPU for DeepSeek-V3-685B and 974/919 TFLOPS/GPU for Qwen3-235B. As a performant, scalable, and production-ready open-source solution, it has been used across academia and industry for training MoE models ranging from billions to trillions of parameters on clusters scaling up to thousands of GPUs. This report explains how these techniques work, their trade-offs, and their interactions at the systems level, providing practical guidance for scaling MoE models with Megatron Core.
>
---
#### [replaced 050] PRISM of Opinions: A Persona-Reasoned Multimodal Framework for User-centric Conversational Stance Detection
- **分类: cs.CL**

- **简介: 该论文属于多模态对话立场检测任务，解决伪多模态和用户同质化问题。构建了U-MStance数据集，提出PRISM模型，融合用户个性与上下文进行多模态推理。**

- **链接: [https://arxiv.org/pdf/2511.12130](https://arxiv.org/pdf/2511.12130)**

> **作者:** Bingbing Wang; Zhixin Bai; Zhengda Jin; Zihan Wang; Xintong Song; Jingjie Lin; Sixuan Li; Jing Li; Ruifeng Xu
>
> **摘要:** The rapid proliferation of multimodal social media content has driven research in Multimodal Conversational Stance Detection (MCSD), which aims to interpret users' attitudes toward specific targets within complex discussions. However, existing studies remain limited by: **1) pseudo-multimodality**, where visual cues appear only in source posts while comments are treated as text-only, misaligning with real-world multimodal interactions; and **2) user homogeneity**, where diverse users are treated uniformly, neglecting personal traits that shape stance expression. To address these issues, we introduce **U-MStance**, the first user-centric MCSD dataset, containing over 40k annotated comments across six real-world targets. We further propose **PRISM**, a **P**ersona-**R**easoned mult**I**modal **S**tance **M**odel for MCSD. PRISM first derives longitudinal user personas from historical posts and comments to capture individual traits, then aligns textual and visual cues within conversational context via Chain-of-Thought to bridge semantic and pragmatic gaps across modalities. Finally, a mutual task reinforcement mechanism is employed to jointly optimize stance detection and stance-aware response generation for bidirectional knowledge transfer. Experiments on U-MStance demonstrate that PRISM yields significant gains over strong baselines, underscoring the effectiveness of user-centric and context-grounded multimodal reasoning for realistic stance understanding.
>
---
#### [replaced 051] From Veracity to Diffusion: Adressing Operational Challenges in Moving From Fake-News Detection to Information Disorders
- **分类: cs.CL**

- **简介: 该论文属于信息虚假性检测任务，旨在解决从虚假新闻检测转向信息扩散预测的挑战。通过对比两个数据集，分析预测目标变化对模型性能的影响，并提出轻量高效的解决方案。**

- **链接: [https://arxiv.org/pdf/2512.02552](https://arxiv.org/pdf/2512.02552)**

> **作者:** Francesco Paolo Savatteri; Chahan Vidal-Gorène; Florian Cafiero
>
> **摘要:** A wide part of research on misinformation has relied lies on fake-news detection, a task framed as the prediction of veracity labels attached to articles or claims. Yet social-science research has repeatedly emphasized that information manipulation goes beyond fabricated content and often relies on amplification dynamics. This theoretical turn has consequences for operationalization in applied social science research. What changes empirically when prediction targets move from veracity to diffusion? And which performance level can be attained in limited resources setups ? In this paper we compare fake-news detection and virality prediction across two datasets, EVONS and FakeNewsNet. We adopt an evaluation-first perspective and examine how benchmark behavior changes when the prediction target shifts from veracity to diffusion. Our experiments show that fake-news detection is comparatively stable once strong textual embeddings are available, whereas virality prediction is much more sensitive to operational choices such as threshold definition and early observation windows. The paper proposes practical ways to operationalize lightweight, transparent pipelines for misinformation-related prediction tasks that can rival with state-of-the-art.
>
---
