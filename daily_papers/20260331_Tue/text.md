# 自然语言处理 cs.CL

- **最新发布 119 篇**

- **更新 94 篇**

## 最新发布

#### [new 001] Multi-Agent Dialectical Refinement for Enhanced Argument Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于论点分类任务，解决LLM在结构歧义上的分类问题。提出MAD-ACC框架，通过多代理辩论提升分类准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.27451](https://arxiv.org/pdf/2603.27451)**

> **作者:** Jakub Bąba; Jarosław A. Chudziak
>
> **备注:** Accepted for publication in the proceedings of ACIIDS 2026
>
> **摘要:** Argument Mining (AM) is a foundational technology for automated writing evaluation, yet traditional supervised approaches rely heavily on expensive, domain-specific fine-tuning. While Large Language Models (LLMs) offer a training-free alternative, they often struggle with structural ambiguity, failing to distinguish between similar components like Claims and Premises. Furthermore, single-agent self-correction mechanisms often suffer from sycophancy, where the model reinforces its own initial errors rather than critically evaluating them. We introduce MAD-ACC (Multi-Agent Debate for Argument Component Classification), a framework that leverages dialectical refinement to resolve classification uncertainty. MAD-ACC utilizes a Proponent-Opponent-Judge model where agents defend conflicting interpretations of ambiguous text, exposing logical nuances that single-agent models miss. Evaluation on the UKP Student Essays corpus demonstrates that MAD-ACC achieves a Macro F1 score of 85.7%, significantly outperforming single-agent reasoning baselines, without requiring domain-specific training. Additionally, unlike "black-box" classifiers, MAD-ACC's dialectical approach offers a transparent and explainable alternative by generating human-readable debate transcripts that explain the reasoning behind decisions.
>
---
#### [new 002] Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，旨在提升数学推理能力。通过实验发现模型能力远超优化策略，高温度采样已有效降低误差相关性。**

- **链接: [https://arxiv.org/pdf/2603.27844](https://arxiv.org/pdf/2603.27844)**

> **作者:** Natapong Nitarach
>
> **摘要:** Majority voting over multiple LLM attempts improves mathematical reasoning, but correlated errors limit the effective sample size. A natural fix: assign structurally different reasoning strategies to different voters to decorrelate errors. We test this Diverse Prompt Mixer in the AIMO~3 competition: 3 models, 23+ experiments, and 50 IMO-level problems on a single H100 80 GB with a 5-hour limit. Every intervention fails. High-temperature sampling already decorrelates errors sufficiently; weaker prompt strategies reduce per-attempt accuracy more than they reduce correlation. Across a 17-point model capability gap and every inference-time optimization we tried, model capability dominates by an order of magnitude.
>
---
#### [new 003] Rethinking Atomic Decomposition for LLM Judges: A Prompt-Controlled Study of Reference-Grounded QA Evaluation
- **分类: cs.CL**

- **简介: 该论文属于参考基准问答评估任务，旨在比较原子分解与整体提示方法的效果。研究发现整体方法在多数基准上表现更优，尤其在检测不完整答案方面。**

- **链接: [https://arxiv.org/pdf/2603.28005](https://arxiv.org/pdf/2603.28005)**

> **作者:** Xinran Zhang
>
> **摘要:** Atomic decomposition -- breaking a candidate answer into claims before verifying each against a reference -- is a widely adopted design for LLM-based reference-grounded judges. However, atomic prompts are typically richer and longer, making it unclear whether any advantage comes from decomposition or from richer prompting. We study this for benchmark-style completeness-sensitive reference-support classification: classifying a candidate as fully supported, partially supported, or unsupported relative to a supplied reference. We compare a self-decomposing atomic judge (single-prompt decompose-and-verify) against a prompt-controlled holistic judge with the same inputs and a similarly detailed rubric. On 200 source examples per dataset across TruthfulQA, ASQA, and QAMPARI, with four model families, source-level paired tests, cluster bootstrap, and aggregation across three pre-frozen prompt variants per design family, we find the holistic judge matches or exceeds the atomic judge on two of three benchmarks: ASQA and QAMPARI favor holistic across all four families (statistically reliable in three of four), while TruthfulQA shows a small atomic edge. The holistic advantage is concentrated in partially\_supported cases -- incompleteness detection. A sensitivity check against human annotations confirms the ranking under both benchmark-completeness and human factual-correctness standards. Our finding is specific to the self-decomposing single-prompt pattern on three QA-style benchmarks with 200 source examples each; multi-stage atomic pipelines and non-QA tasks remain untested. Among perturbations examined, reference-quality degradation produced the largest accuracy drops for both judge families.
>
---
#### [new 004] Hidden Ads: Behavior Triggered Semantic Backdoors for Advertisement Injection in Vision Language Models
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文研究视觉语言模型中的隐蔽广告攻击，解决未经授权广告注入问题。通过用户行为触发，实现自然语境下的广告植入。**

- **链接: [https://arxiv.org/pdf/2603.27522](https://arxiv.org/pdf/2603.27522)**

> **作者:** Duanyi Yao; Changyue Li; Zhicong Huang; Cheng Hong; Songze Li
>
> **摘要:** Vision-Language Models (VLMs) are increasingly deployed in consumer applications where users seek recommendations about products, dining, and services. We introduce Hidden Ads, a new class of backdoor attacks that exploit this recommendation-seeking behavior to inject unauthorized advertisements. Unlike traditional pattern-triggered backdoors that rely on artificial triggers such as pixel patches or special tokens, Hidden Ads activates on natural user behaviors: when users upload images containing semantic content of interest (e.g., food, cars, animals) and ask recommendation-seeking questions, the backdoored model provides correct, helpful answers while seamlessly appending attacker-specified promotional slogans. This design preserves model utility and produces natural-sounding injections, making the attack practical for real-world deployment in consumer-facing recommendation services. We propose a multi-tier threat framework to systematically evaluate Hidden Ads across three adversary capability levels: hard prompt injection, soft prompt optimization, and supervised fine-tuning. Our poisoned data generation pipeline uses teacher VLM-generated chain-of-thought reasoning to create natural trigger--slogan associations across multiple semantic domains. Experiments on three VLM architectures demonstrate that Hidden Ads achieves high injection efficacy with near-zero false positives while maintaining task accuracy. Ablation studies confirm that the attack is data-efficient, transfers effectively to unseen datasets, and scales to multiple concurrent domain-slogan pairs. We evaluate defenses including instruction-based filtering and clean fine-tuning, finding that both fail to remove the backdoor without causing significant utility degradation.
>
---
#### [new 005] EnsemJudge: Enhancing Reliability in Chinese LLM-Generated Text Detection through Diverse Model Ensembles
- **分类: cs.CL**

- **简介: 该论文属于中文LLM生成文本检测任务，旨在解决检测模型在域外或对抗样本下的可靠性问题。通过集成策略和投票机制提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.27949](https://arxiv.org/pdf/2603.27949)**

> **作者:** Zhuoshang Wang; Yubing Ren; Guoyu Zhao; Xiaowei Zhu; Hao Li; Yanan Cao
>
> **备注:** Accepted by NLPCC 2025 Shared Tasks
>
> **摘要:** Large Language Models (LLMs) are widely applied across various domains due to their powerful text generation capabilities. While LLM-generated texts often resemble human-written ones, their misuse can lead to significant societal risks. Detecting such texts is an essential technique for mitigating LLM misuse, and many detection methods have shown promising results across different datasets. However, real-world scenarios often involve out-of-domain inputs or adversarial samples, which can affect the performance of detection methods to varying degrees. Furthermore, most existing research has focused on English texts, with limited work addressing Chinese text detection. In this study, we propose EnsemJudge, a robust framework for detecting Chinese LLM-generated text by incorporating tailored strategies and ensemble voting mechanisms. We trained and evaluated our system on a carefully constructed Chinese dataset provided by NLPCC2025 Shared Task 1. Our approach outperformed all baseline methods and achieved first place in the task, demonstrating its effectiveness and reliability in Chinese LLM-generated text detection. Our code is available at this https URL.
>
---
#### [new 006] Investigating the Influence of Language on Sycophantic Behavior of Multilingual LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究多语言大模型的谄媚行为。旨在解决语言对模型谄媚倾向的影响问题，通过实验分析不同语言下的模型反应。**

- **链接: [https://arxiv.org/pdf/2603.27664](https://arxiv.org/pdf/2603.27664)**

> **作者:** Bayan Abdullah Aldahlawi; A. B. M. Ashikur Rahman; Irfan Ahmad
>
> **备注:** 15 Pages, 5 figures
>
> **摘要:** Large language models (LLMs) have achieved strong performance across a wide range of tasks, but they are also prone to sycophancy, the tendency to agree with user statements regardless of validity. Previous research has outlined both the extent and the underlying causes of sycophancy in earlier models, such as ChatGPT-3.5 and Davinci. Newer models have since undergone multiple mitigation strategies, yet there remains a critical need to systematically test their behavior. In particular, the effect of language on sycophancy has not been explored. In this work, we investigate how the language influences sycophantic responses. We evaluate three state-of-the-art models, GPT-4o mini, Gemini 1.5 Flash, and Claude 3.5 Haiku, using a set of tweet-like opinion prompts translated into five additional languages: Arabic, Chinese, French, Spanish, and Portuguese. Our results show that although newer models exhibit significantly less sycophancy overall compared to earlier generations, the extent of sycophancy is still influenced by the language. We further provide a granular analysis of how language shapes model agreeableness across sensitive topics, revealing systematic cultural and linguistic patterns. These findings highlight both the progress of mitigation efforts and the need for broader multilingual audits to ensure trustworthy and bias-aware deployment of LLMs.
>
---
#### [new 007] Training data generation for context-dependent rubric-based short answer grading
- **分类: cs.CL**

- **简介: 该论文属于自动评分任务，旨在解决小数据生成大规模训练集的问题。通过简单文本格式转换，构建相似的替代数据集以支持模型训练。**

- **链接: [https://arxiv.org/pdf/2603.28537](https://arxiv.org/pdf/2603.28537)**

> **作者:** Pavel Šindelář; Dávid Slivka; Christopher Bouma; Filip Prášil; Ondřej Bojar
>
> **摘要:** Every 4 years, the PISA test is administered by the OECD to test the knowledge of teenage students worldwide and allow for comparisons of educational systems. However, having to avoid language differences and annotator bias makes the grading of student answers challenging. For these reasons, it would be interesting to compare methods of automatic student answer grading. To train some of these methods, which require machine learning, or to compute parameters or select hyperparameters for those that do not, a large amount of domain-specific data is needed. In this work, we explore a small number of methods for creating a large-scale training dataset using only a relatively small confidential dataset as a reference, leveraging a set of very simple derived text formats to preserve confidentiality. Using these methods, we successfully created three surrogate datasets that are, at the very least, superficially more similar to the reference dataset than purely the result of prompt-based generation. Early experiments suggest one of these approaches might also lead to improved model training.
>
---
#### [new 008] Not All Subjectivity Is the Same! Defining Desiderata for the Evaluation of Subjectivity in NLP
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的主观性评估任务，旨在解决现有评价体系与模型目标不匹配的问题。作者提出七项评估标准，以更全面地衡量模型的主观性表达能力。**

- **链接: [https://arxiv.org/pdf/2603.28351](https://arxiv.org/pdf/2603.28351)**

> **作者:** Urja Khurana; Michiel van der Meer; Enrico Liscio; Antske Fokkens; Pradeep K. Murukannaiah
>
> **备注:** Under review
>
> **摘要:** Subjective judgments are part of several NLP datasets and recent work is increasingly prioritizing models whose outputs reflect this diversity of perspectives. Such responses allow us to shed light on minority voices, which are frequently marginalized or obscured by dominant perspectives. It remains a question whether our evaluation practices align with these models' objectives. This position paper proposes seven evaluation desiderata for subjectivity-sensitive models, rooted in how subjectivity is represented in NLP data and models. The desiderata are constructed in a top-down approach, keeping in mind the user-centric impact of such models. We scan the experimental setup of 60 papers and show that various aspects of subjectivity are still understudied: the distinction between ambiguous and polyphonic input, whether subjectivity is effectively expressed to the user, and a lack of interplay between different desiderata, amongst other gaps.
>
---
#### [new 009] Culturally Adaptive Explainable LLM Assessment for Multilingual Information Disorder: A Human-in-the-Loop Approach
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于信息失真检测任务，旨在解决多语言环境下LLM解释不具文化适应性的问题。通过人机协同框架，提升模型在不同文化背景下的解释能力。**

- **链接: [https://arxiv.org/pdf/2603.27356](https://arxiv.org/pdf/2603.27356)**

> **作者:** Maziar Kianimoghadam Jouneghani
>
> **备注:** 9 pages, 3 figures, 1 table. Accepted to the Information Disorder Workshop at LREC 2026
>
> **摘要:** Recognizing information disorder is difficult because judgments about manipulation depend on cultural and linguistic context. Yet current Large Language Models (LLMs) often behave as monocultural, English-centric "black boxes," producing fluent rationales that overlook localized framing. Preliminary evidence from the multilingual Information Disorder (InDor) corpus suggests that existing models struggle to explain manipulated news consistently across communities. To address this gap, this ongoing study proposes a Hybrid Intelligence Loop, a human-in-the-loop (HITL) framework that grounds model assessment in human-written rationales from native-speaking annotators. The approach moves beyond static target-language few-shot prompting by pairing English task instructions with dynamically retrieved target-language exemplars drawn from filtered InDor annotations through In-Context Learning (ICL). In the initial pilot, the Exemplar Bank is seeded from these filtered annotations and used to compare static and adaptive prompting on Farsi and Italian news. The study evaluates span and severity prediction, the quality and cultural appropriateness of generated rationales, and model alignment across evaluator groups, providing a testbed for culturally grounded explainable AI.
>
---
#### [new 010] The Degree of Language Diacriticity and Its Effect on Tasks
- **分类: cs.CL**

- **简介: 该论文研究语言中变音符号的复杂性及其对任务的影响，旨在量化变音符号依赖程度及对其恢复任务的影响。**

- **链接: [https://arxiv.org/pdf/2603.27653](https://arxiv.org/pdf/2603.27653)**

> **作者:** Adi Cohen; Yuval Pinter
>
> **备注:** Accepted to CAWL 2026
>
> **摘要:** Diacritics are orthographic marks that clarify pronunciation, distinguish similar words, or alter meaning. They play a central role in many writing systems, yet their impact on language technology has not been systematically quantified across scripts. While prior work has examined diacritics in individual languages, there's no cross-linguistic, data-driven framework for measuring the degree to which writing systems rely on them and how this affects downstream tasks. We propose a data-driven framework for quantifying diacritic complexity using corpus-level, information-theoretic metrics that capture the frequency, ambiguity, and structural diversity of character-diacritic combinations. We compute these metrics over 24 corpora in 15 languages, spanning both single- and multi-diacritic scripts. We then examine how diacritic complexity correlates with performance on the task of diacritics restoration, evaluating BERT- and RNN-based models. We find that across languages, higher diacritic complexity is strongly associated with lower restoration accuracy. In single-diacritic scripts, where character-diacritic combinations are more predictable, frequency-based and structural measures largely align. In multi-diacritic scripts, however, structural complexity exhibits the strongest association with performance, surpassing frequency-based measures. These findings show that measurable properties of diacritic usage influence the performance of diacritic restoration models, demonstrating that orthographic complexity is not only descriptive but functionally relevant for modeling.
>
---
#### [new 011] Conversational Agents and the Understanding of Human Language: Reflections on AI, LLMs, and Cognitive Science
- **分类: cs.CL**

- **简介: 论文探讨NLP与人类语言理解的关系，分析不同阶段NLP技术与语言理论的异同。任务是评估语言技术对人类语言认知理解的贡献，指出当前技术未深化对此问题的认识。**

- **链接: [https://arxiv.org/pdf/2603.27809](https://arxiv.org/pdf/2603.27809)**

> **作者:** Andrei Popescu-Belis
>
> **备注:** 7 pages
>
> **摘要:** In this paper, we discuss the relationship between natural language processing by computers (NLP) and the understanding of the human language capacity, as studied by linguistics and cognitive science. We outline the evolution of NLP from its beginnings until the age of large language models, and highlight for each of its main paradigms some similarities and differences with theories of the human language capacity. We conclude that the evolution of language technology has not substantially deepened our understanding of how human minds process natural language, despite the impressive language abilities attained by current chatbots using artificial neural networks.
>
---
#### [new 012] RASPRef: Retrieval-Augmented Self-Supervised Prompt Refinement for Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理中提示设计效率低的问题。提出RASPRef框架，通过检索和自监督优化提升提示效果。**

- **链接: [https://arxiv.org/pdf/2603.27008](https://arxiv.org/pdf/2603.27008)**

> **作者:** Rahul Soni
>
> **摘要:** Recent reasoning-focused language models such as DeepSeek R1 and OpenAI o1 have demonstrated strong performance on structured reasoning benchmarks including GSM8K, MATH, and multi-hop question answering tasks. However, their performance remains highly sensitive to prompt formulation, and designing effective prompts is typically a manual and iterative process that does not scale well across tasks or domains. To address this limitation, we introduce Retrieval-Augmented Self-Supervised Prompt Refinement (RASPRef), a framework that improves prompts without requiring human annotations or task-specific supervision. The approach retrieves relevant examples and previously generated reasoning trajectories, and leverages signals such as multi-sample consistency, verifier feedback, and model-generated critiques to iteratively refine the prompt. Unlike prior approaches that focus primarily on improving model outputs, RASPRef directly treats the prompt as the optimization target and improves it through an iterative retrieval-guided refinement process. Experiments on GSM8K-style mathematical reasoning tasks show that retrieval-guided prompting improves performance compared with a static prompting baseline. We further discuss how retrieval quality, trajectory selection, and self-supervised feedback signals may influence the effectiveness of prompt refinement. These findings suggest that prompt design remains a critical factor for reasoning-oriented language models, and that self-improving prompts offer a practical and scalable strategy for improving reasoning performance.
>
---
#### [new 013] GeoBlock: Inferring Block Granularity from Dependency Geometry in Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型解码任务，解决块扩散中块大小选择问题。提出GeoBlock框架，根据依赖几何动态确定块边界，提升并行效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.26675](https://arxiv.org/pdf/2603.26675)**

> **作者:** Lipeng Wan; Junjie Ma; Jianhui Gu; Zeyang Liu; Xuyang Lu; Xuguang Lan
>
> **备注:** 13 pages, 4 figures, Code available upon publication
>
> **摘要:** Block diffusion enables efficient parallel refinement in diffusion language models, but its decoding behavior depends critically on block size. Existing block-sizing strategies rely on fixed rules or heuristic signals and do not account for the dependency geometry that determines which tokens can be safely refined together. This motivates a geometry view of diffusion decoding: \emph{regions with strong causal ordering require sequential updates, whereas semantically cohesive regions admit parallel refinement.} We introduce GeoBlock, a geometry-aware block inference framework that determines block granularity directly from attention-derived dependency geometry. Instead of relying on predefined schedules or local confidence heuristics, GeoBlock analyzes cross-token dependency patterns to identify geometrically stable refinement regions and dynamically determines appropriate block boundaries during decoding. By adapting block granularity to the dependency geometry, GeoBlock preserves the parallel efficiency of block diffusion while enforcing dependency-consistent refinement that exhibits autoregressive reliability. GeoBlock requires no additional training and integrates seamlessly into existing block diffusion architectures. Extensive experiments across multiple benchmarks show that GeoBlock reliably identifies geometry-consistent block boundaries and improves the accuracy of block diffusion with only a small additional computational budget.
>
---
#### [new 014] Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型后训练任务，探讨课程学习在演绎推理中的有效性。研究发现，按难度排序的训练序列在准确率和响应长度上并无优势，挑战了课程学习的实用性。**

- **链接: [https://arxiv.org/pdf/2603.27226](https://arxiv.org/pdf/2603.27226)**

> **作者:** Maximilian Mordig; Andreas Opedal; Weiyang Liu; Bernhard Schölkopf
>
> **摘要:** Curriculum learning (CL), motivated by the intuition that learning in increasing order of difficulty should ease generalization, is commonly adopted both in pre-training and post-training of large language models (LLMs). The intuition of CL is particularly compelling for compositional reasoning, where complex problems are built from elementary inference rules; however, the actual impact of CL on such tasks remains largely underexplored. We present a systematic empirical study of CL for post-training of LLMs, using synthetic arithmetic and logical benchmarks where difficulty is characterized by reasoning complexity rather than surface-level proxies. Surprisingly, across multiple model families and curriculum schedules, we find no robust advantage in difficulty-based sequencing over standard random sampling in either accuracy or response length. These findings persist across both supervised fine-tuning (SFT) and reinforcement learning (RL) methods. Our study suggests that, in the context of deductive reasoning, the specific ordering of training examples plays a negligible role in achieving compositional generalization, challenging the practical utility of curriculum-based post-training.
>
---
#### [new 015] The Necessity of Setting Temperature in LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本评估任务，探讨温度参数对LLM作为评判者性能的影响，通过实验和因果分析研究温度设置的作用。**

- **链接: [https://arxiv.org/pdf/2603.28304](https://arxiv.org/pdf/2603.28304)**

> **作者:** Lujun Li; Lama Sleem; Yangjie Xu; Yewei Song; Aolin Jia; Jerome Francois; Radu State
>
> **摘要:** LLM-as-a-Judge has emerged as an effective and low-cost paradigm for evaluating text quality and factual correctness. Prior studies have shown substantial agreement between LLM judges and human experts, even on tasks that are difficult to assess automatically. In practice, researchers commonly employ fixed temperature configurations during the evaluation process-with values of 0.1 and 1.0 being the most prevalent choices-a convention that is largely empirical rather than principled. However, recent researches suggest that LLM performance exhibits non-trivial sensitivity to temperature settings, that lower temperatures do not universally yield optimal outcomes, and that such effects are highly task-dependent. This raises a critical research question: does temperature influence judge performance in LLM centric evaluation? To address this, we systematically investigate the relationship between temperature and judge performance through a series of controlled experiments, and further adopt a causal inference framework within our empirical statistical analysis to rigorously examine the direct causal effect of temperature on judge behavior, offering actionable engineering insights for the design of LLM-centric evaluation pipelines.
>
---
#### [new 016] TailNLG: A Multilingual Benchmark Addressing Verbalization of Long-Tail Entities
- **分类: cs.CL**

- **简介: 该论文属于数据到文本生成任务，旨在解决长尾实体口语化问题。构建了多语言基准TailNLG，评估模型在罕见实体上的表现，揭示其存在偏差。**

- **链接: [https://arxiv.org/pdf/2603.27768](https://arxiv.org/pdf/2603.27768)**

> **作者:** Lia Draetta; Michael Oliverio; Virginia Ramón-Ferrer; Pier Felice Balestrucci; Flaviana Corallo; Carlos Badenes-Olmedo; Alessandro Mazzei; Marco Antonio Stranisci; Rossana Damiano
>
> **摘要:** The automatic verbalization of structured knowledge is a key task for making knowledge graphs accessible to non-expert users and supporting retrieval-augmented generation systems. Although recent advances in Data-to-Text generation have improved multilingual coverage, little attention has been paid to potential biases in the verbalization of rare entities, frequently known as long-tail entities. In this work, we present the first systematic study of long-tail entities in Data-to-Text generation. We introduce TailNLG, a new multilingual benchmark in English, Italian, and Spanish, built from Wikidata and covering entities with varying levels of popularity. We evaluate three different families of large language models in zero-shot settings and compare their performance on rare versus common entities, as well as against the established WebNLG benchmark. Our results reveal a consistent bias against long-tail entities: embedding-based scores are lower, and model uncertainty is higher for rare entities. We further show that the impact of long-tail entities varies across models and languages, and that existing evaluation metrics do not consistently capture these differences, highlighting the need for more reliable evaluation frameworks.
>
---
#### [new 017] KAT-Coder-V2 Technical Report
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍KAT-Coder-V2，一个用于代码生成的智能模型。解决代码生成与任务理解问题，通过分域训练和优化提升性能。**

- **链接: [https://arxiv.org/pdf/2603.27703](https://arxiv.org/pdf/2603.27703)**

> **作者:** Fengxiang Li; Han Zhang; Haoyang Huang; Jinghui Wang; Jinhua Hao; Kun Yuan; Mengtong Li; Minglei Zhang; Pengcheng Xu; Wenhao Zhuang; Yizhen Shao; Zongxian Feng; Can Tang; Chao Wang; Chengxiao Tong; Fan Yang; Gang Xiong; Haixuan Gao; Han Gao; Hao Wang; Haochen Liu; Hongliang Sun; Jiabao Li; Jingwen Chang; Jun Du; Junyi Peng; Leizhen Cui; Meimei Jing; Mingqi Wu; Shangpeng Yan; Shaotong Qi; Suzhe Xu; Wenxuan Zhao; Xianda Sun; Xuan Xie; Yanbo Wang; Yao Xia; Yinghan Cui; Yingpeng Chen; Yong Wang; Yuze Shi; Zhiwei Shen; Ziyu Wang; Ming Sun; Lin Ye; Bin Chen
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** We present KAT-Coder-V2, an agentic coding model developed by the KwaiKAT team at Kuaishou. KAT-Coder-V2 adopts a "Specialize-then-Unify" paradigm that decomposes agentic coding into five expert domains - SWE, WebCoding, Terminal, WebSearch, and General - each undergoing independent supervised fine-tuning and reinforcement learning, before being consolidated into a single model via on-policy distillation. We develop KwaiEnv, a modular infrastructure sustaining tens of thousands of concurrent sandbox instances, and scale RL training along task complexity, intent alignment, and scaffold generalization. We further propose MCLA for stabilizing MoE RL training and Tree Training for eliminating redundant computation over tree-structured trajectories with up to 6.2x speedup. KAT-Coder-V2 achieves 79.6% on SWE-bench Verified (vs. Claude Opus 4.6 at 80.8%), 88.7 on PinchBench (surpassing GLM-5 and MiniMax M2.7), ranks first across all three frontend aesthetics scenarios, and maintains strong generalist scores on Terminal-Bench Hard (46.8) and tau^2-Bench (93.9). Our model is publicly available at this https URL.
>
---
#### [new 018] Text Data Integration
- **分类: cs.CL; cs.IR**

- **简介: 论文探讨文本数据集成任务，解决结构化与非结构化数据融合难题，分析挑战、现状及未来问题。**

- **链接: [https://arxiv.org/pdf/2603.27055](https://arxiv.org/pdf/2603.27055)**

> **作者:** Md Ataur Rahman; Dimitris Sacharidis; Oscar Romero; Sergi Nadal
>
> **备注:** Accepted for Publication as a Book Chapter in "Data Engineering for Data Science" (ISBN: 978-3-032-18765-9)
>
> **摘要:** Data comes in many forms. From a shallow perspective, they can be viewed as being either in structured (e.g., as a relation, as key-value pairs) or unstructured (e.g., text, image) formats. So far, machines have been fairly good at processing and reasoning over structured data that follows a precise schema. However, the heterogeneity of data poses a significant challenge on how well diverse categories of data can be meaningfully stored and processed. Data Integration, a crucial part of the data engineering pipeline, addresses this by combining disparate data sources and providing unified data access to end-users. Until now, most data integration systems have leaned on only combining structured data sources. Nevertheless, unstructured data (a.k.a. free text) also contains a plethora of knowledge waiting to be utilized. Thus, in this chapter, we firstly make the case for the integration of textual data, to later present its challenges, state of the art and open problems.
>
---
#### [new 019] EarlySciRev: A Dataset of Early-Stage Scientific Revisions Extracted from LaTeX Writing Traces
- **分类: cs.CL**

- **简介: 该论文提出EarlySciRev数据集，用于研究科学写作的早期修订行为。任务是解决公开数据不足的问题，通过提取LaTeX中的注释文本，生成真实修订对，支持写作动态和模型评估研究。**

- **链接: [https://arxiv.org/pdf/2603.28515](https://arxiv.org/pdf/2603.28515)**

> **作者:** Léane Jourdan; Julien Aubert-Béduchaud; Yannis Chupin; Marah Baccari; Florian Boudin
>
> **备注:** Accepted to NSLP@LREC
>
> **摘要:** Scientific writing is an iterative process that generates rich revision traces, yet publicly available resources typically expose only final or near-final versions of papers. This limits empirical study of revision behaviour and evaluation of large language models (LLMs) for scientific writing. We introduce EarlySciRev, a dataset of early-stage scientific text revisions automatically extracted from arXiv LaTeX source files. Our key observation is that commented-out text in LaTeX often preserves discarded or alternative formulations written by the authors themselves. By aligning commented segments with nearby final text, we extract paragraph-level candidate revision pairs and apply LLM-based filtering to retain genuine revisions. Starting from 1.28M candidate pairs, our pipeline yields 578k validated revision pairs, grounded in authentic early drafting traces. We additionally provide a human-annotated benchmark for revision detection. EarlySciRev complements existing resources focused on late-stage revisions or synthetic rewrites and supports research on scientific writing dynamics, revision modelling, and LLM-assisted editing.
>
---
#### [new 020] DongYuan: An LLM-Based Framework for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis
- **分类: cs.CL**

- **简介: 该论文属于医学诊断任务，旨在解决中西医结合脾胃疾病诊断中的数据不足、逻辑整合及评估标准缺失问题。提出DongYuan框架，包含数据集、核心模型和评估基准。**

- **链接: [https://arxiv.org/pdf/2603.28191](https://arxiv.org/pdf/2603.28191)**

> **作者:** Hua Li; Yingying Li; Xiaobin Feng; Xinyi Fu; Lifeng Dong; Qingfeng Yang; Yanzhe Chen; Xiaoju Feng; Zhidong Cao; Jianbin Guo; Yanru Du
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** The clinical burden of spleen-stomach disorders is substantial. While large language models (LLMs) offer new potential for medical applications, they face three major challenges in the context of integrative Chinese and Western medicine (ICWM): a lack of high-quality data, the absence of models capable of effectively integrating the reasoning logic of traditional Chinese medicine (TCM) syndrome differentiation with that of Western medical (WM) disease diagnosis, and the shortage of a standardized evaluation benchmark. To address these interrelated challenges, we propose DongYuan, an ICWM spleen-stomach diagnostic framework. Specifically, three ICWM datasets (SSDF-Syndrome, SSDF-Dialogue, and SSDF-PD) were curated to fill the gap in high-quality data for spleen-stomach disorders. We then developed SSDF-Core, a core diagnostic LLM that acquires robust ICWM reasoning capabilities through a two-stage training regimen of supervised fine-tuning. tuning (SFT) and direct preference optimization (DPO), and complemented it with SSDF-Navigator, a pluggable consultation navigation model designed to optimize clinical inquiry strategies. Additionally, we established SSDF-Bench, a comprehensive evaluation benchmark focused on ICWM diagnosis of spleen-stomach disorders. Experimental results demonstrate that SSDF-Core significantly outperforms 12 mainstream baselines on SSDF-Bench. DongYuan lays a solid methodological foundation and provides practical technical references for the future development of intelligent ICWM diagnostic systems.
>
---
#### [new 021] Resolving the Robustness-Precision Trade-off in Financial RAG through Hybrid Document-Routed Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于金融问答任务，解决RAG系统中鲁棒性与精度的权衡问题。提出HDRR方法，结合文档路由与分块检索，提升性能。**

- **链接: [https://arxiv.org/pdf/2603.26815](https://arxiv.org/pdf/2603.26815)**

> **作者:** Zhiyuan Cheng; Longying Lai; Yue Liu
>
> **备注:** 18 pages, 4 figures, 9 tables. Submitted to Expert Systems with Applications
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems for financial document question answering typically follow a chunk-based paradigm: documents are split into fragments, embedded into vector space, and retrieved via similarity search. While effective in general settings, this approach suffers from cross-document chunk confusion in structurally homogeneous corpora such as regulatory filings. Semantic File Routing (SFR), which uses LLM structured output to route queries to whole documents, reduces catastrophic failures but sacrifices the precision of targeted chunk retrieval. We identify this robustness-precision trade-off through controlled evaluation on the FinDER benchmark (1,500 queries across five groups): SFR achieves higher average scores (6.45 vs. 6.02) and fewer failures (10.3% vs. 22.5%), while chunk-based retrieval (CBR) yields more perfect answers (13.8% vs. 8.5%). To resolve this trade-off, we propose Hybrid Document-Routed Retrieval (HDRR), a two-stage architecture that uses SFR as a document filter followed by chunk-based retrieval scoped to the identified document(s). HDRR eliminates cross-document confusion while preserving targeted chunk precision. Experimental results demonstrate that HDRR achieves the best performance on every metric: an average score of 7.54 (25.2% above CBR, 16.9% above SFR), a failure rate of only 6.4%, a correctness rate of 67.7% (+18.7 pp over CBR), and a perfect-answer rate of 20.1% (+6.3 pp over CBR, +11.6 pp over SFR). HDRR resolves the trade-off by simultaneously achieving the lowest failure rate and the highest precision across all five experimental groups.
>
---
#### [new 022] Marco DeepResearch: Unlocking Efficient Deep Research Agents via Verification-Centric Design
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Marco DeepResearch，解决深度研究代理在长期任务中的可靠性问题，通过验证驱动的设计提升性能。**

- **链接: [https://arxiv.org/pdf/2603.28376](https://arxiv.org/pdf/2603.28376)**

> **作者:** Bin Zhu; Qianghuai Jia; Tian Lan; Junyang Ren; Feng Gu; Feihu Jiang; Longyue Wang; Zhao Xu; Weihua Luo
>
> **摘要:** Deep research agents autonomously conduct open-ended investigations, integrating complex information retrieval with multi-step reasoning across diverse sources to solve real-world problems. To sustain this capability on long-horizon tasks, reliable verification is critical during both training and inference. A major bottleneck in existing paradigms stems from the lack of explicit verification mechanisms in QA data synthesis, trajectory construction, and test-time scaling. Errors introduced at each stage propagate downstream and degrade the overall agent performance. To address this, we present Marco DeepResearch, a deep research agent optimized with a verification-centric framework design at three levels: \textbf{(1)~QA Data Synthesis:} We introduce verification mechanisms to graph-based and agent-based QA synthesis to control question difficulty while ensuring answers are unique and correct; \textbf{(2)~Trajectory Construction:} We design a verification-driven trajectory synthesis method that injects explicit verification patterns into training trajectories; and \textbf{(3)~Test-time scaling:} We use Marco DeepResearch itself as a verifier at inference time and effectively improve performance on challenging questions. Extensive experimental results demonstrate that our proposed Marco DeepResearch agent significantly outperforms 8B-scale deep research agents on most challenging benchmarks, such as BrowseComp and BrowseComp-ZH. Crucially, under a maximum budget of 600 tool calls, Marco DeepResearch even surpasses or approaches several 30B-scale agents, like Tongyi DeepResearch-30B.
>
---
#### [new 023] Routing Sensitivity Without Controllability: A Diagnostic Study of Fairness in MoE Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究MoE模型中的公平性问题。针对路由敏感性无法有效控制刻板印象的问题，提出FARE框架进行诊断分析，揭示其局限性并指出架构改进方向。**

- **链接: [https://arxiv.org/pdf/2603.27141](https://arxiv.org/pdf/2603.27141)**

> **作者:** Junhyeok Lee; Kyu Sung Choi
>
> **备注:** 10 pages, 2 figures, 8 tables
>
> **摘要:** Mixture-of-Experts (MoE) language models are universally sensitive to demographic content at the routing level, yet exploiting this sensitivity for fairness control is structurally limited. We introduce Fairness-Aware Routing Equilibrium (FARE), a diagnostic framework designed to probe the limits of routing-level stereotype intervention across diverse MoE architectures. FARE reveals that routing-level preference shifts are either unachievable (Mixtral, Qwen1.5, Qwen3), statistically non-robust (DeepSeekMoE), or accompanied by substantial utility cost (OLMoE, -4.4%p CrowS-Pairs at -6.3%p TQA). Critically, even where log-likelihood preference shifts are robust, they do not transfer to decoded generation: expanded evaluations on both non-null models yield null results across all generation metrics. Group-level expert masking reveals why: bias and core knowledge are deeply entangled within expert groups. These findings indicate that routing sensitivity is necessary but insufficient for stereotype control, and identify specific architectural conditions that can inform the design of more controllable future MoE systems.
>
---
#### [new 024] Not Worth Mentioning? A Pilot Study on Salient Proposition Annotation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的摘要任务，旨在解决 proposition salience 的量化问题。通过定义注释任务并评估其与话语结构的关系，探索 graded 概念的实用性。**

- **链接: [https://arxiv.org/pdf/2603.27358](https://arxiv.org/pdf/2603.27358)**

> **作者:** Amir Zeldes; Katherine Conhaim; Lauren Levine
>
> **摘要:** Despite a long tradition of work on extractive summarization, which by nature aims to recover the most important propositions in a text, little work has been done on operationalizing graded proposition salience in naturally occurring data. In this paper, we adopt graded summarization-based salience as a metric from previous work on Salient Entity Extraction (SEE) and adapt it to quantify proposition salience. We define the annotation task, apply it to a small multi-genre dataset, evaluate agreement and carry out a preliminary study of the relationship between our metric and notions of discourse unit centrality in discourse parsing following Rhetorical Structure Theory (RST).
>
---
#### [new 025] Who Wrote the Book? Detecting and Attributing LLM Ghostwriters
- **分类: cs.CL**

- **简介: 该论文属于作者归属任务，旨在检测和识别大语言模型生成的文本作者。研究提出GhostWriteBench数据集和TRACE方法，解决跨领域和未知模型的作者识别问题。**

- **链接: [https://arxiv.org/pdf/2603.28054](https://arxiv.org/pdf/2603.28054)**

> **作者:** Anudeex Shetty; Qiongkai Xu; Olga Ohrimenko; Jey Han Lau
>
> **摘要:** In this paper, we introduce GhostWriteBench, a dataset for LLM authorship attribution. It comprises long-form texts (50K+ words per book) generated by frontier LLMs, and is designed to test generalisation across multiple out-of-distribution (OOD) dimensions, including domain and unseen LLM author. We also propose TRACE -- a novel fingerprinting method that is interpretable and lightweight -- that works for both open- and closed-source models. TRACE creates the fingerprint by capturing token-level transition patterns (e.g., word rank) estimated by another lightweight language model. Experiments on GhostWriteBench demonstrate that TRACE achieves state-of-the-art performance, remains robust in OOD settings, and works well in limited training data scenarios.
>
---
#### [new 026] SACRED: A Faithful Annotated Multimedia Multimodal Multilingual Dataset for Classifying Connectedness Types in Online Spirituality
- **分类: cs.CL; cs.MM**

- **简介: 该论文提出多模态多语言数据集SACRED，用于分类在线灵性中的连接类型。解决灵性研究数据不足的问题，评估多种模型性能并发现新连接类型。**

- **链接: [https://arxiv.org/pdf/2603.27331](https://arxiv.org/pdf/2603.27331)**

> **作者:** Qinghao Guan; Yuchen Pan; Donghao Li; Zishi Zhang; Yiyang Chen; Lu Li; Flaminia Canu; Emilia Volkart; Gerold Schneider
>
> **备注:** Accepted by LLMs4SSH 2026 at LREC
>
> **摘要:** In religion and theology studies, spirituality has garnered significant research attention for the reason that it not only transcends culture but offers unique experience to each individual. However, social scientists often rely on limited datasets, which are basically unavailable online. In this study, we collaborated with social scientists to develop a high-quality multimedia multi-modal datasets, \textbf{SACRED}, in which the faithfulness of classification is guaranteed. Using \textbf{SACRED}, we evaluated the performance of 13 popular LLMs as well as traditional rule-based and fine-tuned approaches. The result suggests DeepSeek-V3 model performs well in classifying such abstract concepts (i.e., 79.19\% accuracy in the Quora test set), and the GPT-4o-mini model surpassed the other models in the vision tasks (63.99\% F1 score). Purportedly, this is the first annotated multi-modal dataset from online spirituality communication. Our study also found a new type of connectedness which is valuable for communication science studies.
>
---
#### [new 027] Structural-Ambiguity-Aware Translation from Natural Language to Signal Temporal Logic
- **分类: cs.CL; cs.SC**

- **简介: 该论文属于自然语言到信号时序逻辑的翻译任务，解决结构歧义导致的翻译不可靠问题，通过保留多种可能解析生成多个STL候选公式。**

- **链接: [https://arxiv.org/pdf/2603.28426](https://arxiv.org/pdf/2603.28426)**

> **作者:** Kosei Fushimi; Kazunobu Serizawa; Junya Ikemoto; Kazumune Hashimoto
>
> **摘要:** Signal Temporal Logic (STL) is widely used to specify timed and safety-critical tasks for cyber-physical systems, but writing STL formulas directly is difficult for non-expert users. Natural language (NL) provides a convenient interface, yet its inherent structural ambiguity makes one-to-one translation into STL unreliable. In this paper, we propose an \textit{ambiguity-preserving} method for translating NL task descriptions into STL candidate formulas. The key idea is to retain multiple plausible syntactic analyses instead of forcing a single interpretation at the parsing stage. To this end, we develop a three-stage pipeline based on Combinatory Categorial Grammar (CCG): ambiguity-preserving $n$-best parsing, STL-oriented template-based semantic composition, and canonicalization with score aggregation. The proposed method outputs a deduplicated set of STL candidates with plausibility scores, thereby explicitly representing multiple possible formal interpretations of an ambiguous instruction. In contrast to existing one-best NL-to-logic translation methods, the proposed approach is designed to preserve attachment and scope ambiguity. Case studies on representative task descriptions demonstrate that the method generates multiple STL candidates for genuinely ambiguous inputs while collapsing unambiguous or canonically equivalent derivations to a single STL formula.
>
---
#### [new 028] PRBench: End-to-end Paper Reproduction in Physics Research
- **分类: cs.CL; hep-lat; hep-ph; physics.comp-ph; physics.optics**

- **简介: 该论文提出PRBench，一个用于评估AI在物理研究中端到端复现科学论文能力的基准。旨在解决AI在公式推导、代码生成和结果复现中的可靠性问题，通过30个专家任务测试AI的科学推理与执行能力。**

- **链接: [https://arxiv.org/pdf/2603.27646](https://arxiv.org/pdf/2603.27646)**

> **作者:** Shi Qiu; Junyi Deng; Yiwei Deng; Haoran Dong; Jieyu Fu; Mao Li; Zeyu Li; Zhaolong Zhang; Huiwen Zheng; Leidong Bao; Anqi Lv; Zihan Mo; Yadi Niu; Yiyang Peng; Yu Tian; Yili Wang; Ziyu Wang; Zi-Yu Wang; Jiashen Wei; Liuheng Wu; Aoran Xue; Leyi Yang; Guanglu Yuan; Xiarui Zhan; Jingjun Zhang; Zifan Zheng; Pengfei Liu; Linrui Zhen; Kaiyang Li; Qichang Li; Ziheng Zhou; Guo-En Nian; Yunwei Xiao; Qing-Hong Cao; Linjie Dai; Xu Feng; Peng Gao; Ying Gu; Chang Liu; Jia Liu; Ming-xing Luo; Yan-Qing Ma; Liang-You Peng; Huichao Song; Shufeng Wang; Chenxu Wang; Tao Wang; Yi-Nan Wang; Chengyin Wu; Pengwei Zhao; Hua Xing Zhu
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** AI agents powered by large language models exhibit strong reasoning and problem-solving capabilities, enabling them to assist scientific research tasks such as formula derivation and code generation. However, whether these agents can reliably perform end-to-end reproduction from real scientific papers remains an open question. We introduce PRBench, a benchmark of 30 expert-curated tasks spanning 11 subfields of physics. Each task requires an agent to comprehend the methodology of a published paper, implement the corresponding algorithms from scratch, and produce quantitative results matching the original publication. Agents are provided only with the task instruction and paper content, and operate in a sandboxed execution environment. All tasks are contributed by domain experts from over 20 research groups at the School of Physics, Peking University, each grounded in a real published paper and validated through end-to-end reproduction with verified ground-truth results and detailed scoring rubrics. Using an agentified assessment pipeline, we evaluate a set of coding agents on PRBench and analyze their capabilities across key dimensions of scientific reasoning and execution. The best-performing agent, OpenAI Codex powered by GPT-5.3-Codex, achieves a mean overall score of 34%. All agents exhibit a zero end-to-end callback success rate, with particularly poor performance in data accuracy and code correctness. We further identify systematic failure modes, including errors in formula implementation, inability to debug numerical simulations, and fabrication of output data. Overall, PRBench provides a rigorous benchmark for evaluating progress toward autonomous scientific research.
>
---
#### [new 029] PubMed Reasoner: Dynamic Reasoning-based Retrieval for Evidence-Grounded Biomedical Question Answering
- **分类: cs.CL**

- **简介: 该论文属于生物医学问答任务，旨在提升问答系统的准确性与证据可靠性。通过动态推理检索机制，优化查询、批量获取证据并生成有依据的回答，提升了系统性能。**

- **链接: [https://arxiv.org/pdf/2603.27335](https://arxiv.org/pdf/2603.27335)**

> **作者:** Yiqing Zhang; Xiaozhong Liu; Fabricio Murai
>
> **备注:** 20 pages; under review
>
> **摘要:** Trustworthy biomedical question answering (QA) systems must not only provide accurate answers but also justify them with current, verifiable evidence. Retrieval-augmented approaches partially address this gap but lack mechanisms to iteratively refine poor queries, whereas self-reflection methods kick in only after full retrieval is completed. In this context, we introduce PubMed Reasoner, a biomedical QA agent composed of three stages: self-critic query refinement evaluates MeSH terms for coverage, alignment, and redundancy to enhance PubMed queries based on partial (metadata) retrieval; reflective retrieval processes articles in batches until sufficient evidence is gathered; and evidence-grounded response generation produces answers with explicit citations. PubMed Reasoner with a GPT-4o backbone achieves 78.32% accuracy on PubMedQA, slightly surpassing human experts, and showing consistent gains on MMLU Clinical Knowledge. Moreover, LLM-as-judge evaluations prefer our responses across: reasoning soundness, evidence grounding, clinical relevance, and trustworthiness. By orchestrating retrieval-first reasoning over authoritative sources, our approach provides practical assistance to clinicians and biomedical researchers while controlling compute and token costs.
>
---
#### [new 030] Kernel-Smith: A Unified Recipe for Evolutionary Kernel Optimization
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Kernel-Smith框架，解决GPU核优化问题。通过进化算法与后训练策略结合，提升kernel性能，在多个平台上超越现有模型。**

- **链接: [https://arxiv.org/pdf/2603.28342](https://arxiv.org/pdf/2603.28342)**

> **作者:** He Du; Qiming Ge; Jiakai Hu; Aijun Yang; Zheng Cai; Zixian Huang; Sheng Yuan; Qinxiu Cheng; Xinchen Xie; Yicheng Chen; Yining Li; Jiaxing Xie; Huanan Dong; Yaguang Wu; Xiangjun Huang; Jian Yang; Hui Wang; Bowen Zhou; Bowen Li; Qipeng Guo; Kai Chen
>
> **摘要:** We present Kernel-Smith, a framework for high-performance GPU kernel and operator generation that combines a stable evaluation-driven evolutionary agent with an evolution-oriented post-training recipe. On the agent side, Kernel-Smith maintains a population of executable candidates and iteratively improves them using an archive of top-performing and diverse programs together with structured execution feedback on compilation, correctness, and speedup. To make this search reliable, we build backend-specific evaluation services for Triton on NVIDIA GPUs and Maca on MetaX GPUs. On the training side, we convert long-horizon evolution trajectories into step-centric supervision and reinforcement learning signals by retaining correctness-preserving, high-gain revisions, so that the model is optimized as a strong local improver inside the evolutionary loop rather than as a one-shot generator. Under a unified evolutionary protocol, Kernel-Smith-235B-RL achieves state-of-the-art overall performance on KernelBench with Nvidia Triton backend, attaining the best average speedup ratio and outperforming frontier proprietary models including Gemini-3.0-pro and Claude-4.6-opus. We further validate the framework on the MetaX MACA backend, where our Kernel-Smith-MACA-30B surpasses large-scale counterparts such as DeepSeek-V3.2-think and Qwen3-235B-2507-think, highlighting potential for seamless adaptation across heterogeneous platforms. Beyond benchmark results, the same workflow produces upstream contributions to production systems including SGLang and LMDeploy, demonstrating that LLM-driven kernel optimization can transfer from controlled evaluation to practical deployment.
>
---
#### [new 031] HumMusQA: A Human-written Music Understanding QA Benchmark Dataset
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于音乐理解任务，旨在解决LALMs评估标准不足的问题。构建了人工编写的问题数据集，用于更准确地测试模型的音乐理解能力。**

- **链接: [https://arxiv.org/pdf/2603.27877](https://arxiv.org/pdf/2603.27877)**

> **作者:** Benno Weck; Pablo Puentes; Andrea Poltronieri; Satyajeet Prabhu; Dmitry Bogdanov
>
> **备注:** Dataset available at this https URL
>
> **摘要:** The evaluation of music understanding in Large Audio-Language Models (LALMs) requires a rigorously defined benchmark that truly tests whether models can perceive and interpret music, a standard that current data methodologies frequently fail to meet. This paper introduces a meticulously structured approach to music evaluation, proposing a new dataset of 320 hand-written questions curated and validated by experts with musical training, arguing that such focused, manual curation is superior for probing complex audio comprehension. To demonstrate the use of the dataset, we benchmark six state-of-the-art LALMs and additionally test their robustness to uni-modal shortcuts.
>
---
#### [new 032] Arithmetic OOD Failure Unfolds in Stages in Minimal GPTs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究GPT在算术OOD（分布外）任务中的失败阶段，分析其在3位数加法中的表现，提出布局、进位语义、重组和晚期残差四个失败阶段的分解模型。**

- **链接: [https://arxiv.org/pdf/2603.26828](https://arxiv.org/pdf/2603.26828)**

> **作者:** Seine A. Shintani
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Arithmetic benchmarks are often reduced to a single held-out score, but that score can conflate qualitatively different failures. We study a controlled minimal GPT trained on exhaustive 2-digit addition, where all local digit transitions are already present in training, and ask why 3-digit generalization still fails. The failure is staged. First, there is a layout barrier: a learned absolute-position model collapses under a pure 3-digit layout shift, and mixed-layout exposure is the only intervention that materially weakens this barrier. Second, after layout repair, the hundreds position behaves like a carry flag rather than a semantic hundreds digit; targeted carry probes reverse the relevant logit margin, whereas a matched extra-data control does not. Third, after carry repair, the main remaining bottleneck is conditional recomposition: high-conditioned tail data outperforms a matched control, high-only data, and tail-only data on all true-3-digit suites, and the same ordering reappears in a larger 2-layer bridge experiment. The residual errors after recomposition are then overwhelmingly tens-only, and a separate 10-seed late-stage study shows that a sign-aware tens repair raises exact match on the hardest thousands-carry suite from 0.664 to 0.822. We therefore provide an experimentally testable decomposition of arithmetic OOD failure into layout, carry-semantics, recomposition, and late tens-residual stages.
>
---
#### [new 033] Transfer Learning for an Endangered Slavic Variety: Dependency Parsing in Pomak Across Contact-Shaped Dialects
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的依存句法分析任务，针对濒危的保加利亚语方言Pomak，研究其在不同方言间的迁移学习效果，并通过新数据集提升解析准确率。**

- **链接: [https://arxiv.org/pdf/2603.28033](https://arxiv.org/pdf/2603.28033)**

> **作者:** Sercan Karakaş
>
> **备注:** Accepted to DialRes-LREC26 (Workshop on Dialects in NLP A Resource Perspective)
>
> **摘要:** This paper presents new resources and baselines for Dependency Parsing in Pomak, an endangered Eastern South Slavic language with substantial dialectal variation and no widely adopted standard. We focus on the variety spoken in Turkey (Uzunköprü) and ask how well a dependency parser trained on the existing Pomak Universal Dependencies treebank, which was built primarily from the variety that is spoken in Greece, transfers across dialects. We run two experimental phases. First, we train a parser on the Greek-variety UD data and evaluate zero-shot transfer to Turkish-variety Pomak, quantifying the impact of phonological and morphosyntactic differences. Second, we introduce a new manually annotated Turkish-variety Pomak corpus of 650 sentences and show that, despite its small size, targeted fine-tuning substantially improves accuracy; performance is further boosted by cross-variety transfer learning that combines the two dialects.
>
---
#### [new 034] Adaptive Block-Scaled Data Types
- **分类: cs.CL**

- **简介: 该论文针对4-bit量化模型的误差问题，提出自适应块缩放数据类型IF4，提升量化精度。属于模型量化任务，解决NVFP4误差分布不均的问题。**

- **链接: [https://arxiv.org/pdf/2603.28765](https://arxiv.org/pdf/2603.28765)**

> **作者:** Jack Cook; Hyemin S. Lee; Kathryn Le; Junxian Guo; Giovanni Traverso; Anantha P. Chandrakasan; Song Han
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** NVFP4 has grown increasingly popular as a 4-bit format for quantizing large language models due to its hardware support and its ability to retain useful information with relatively few bits per parameter. However, the format is not without limitations: recent work has shown that NVFP4 suffers from its error distribution, resulting in large amounts of quantization error on near-maximal values in each group of 16 values. In this work, we leverage this insight to design new Adaptive Block-Scaled Data Types that can adapt to the distribution of their input values. For four-bit quantization, our proposed IF4 (Int/Float 4) data type selects between FP4 and INT4 representations for each group of 16 values, which are then scaled by an E4M3 scale factor as is done with NVFP4. The selected data type is denoted using the scale factor's sign bit, which is currently unused in NVFP4, and we apply the same insight to design formats for other bit-widths, including IF3 and IF6. When used to quantize language models, we find that IF4 outperforms existing 4-bit block-scaled formats, achieving lower loss during quantized training and achieving higher accuracy on many tasks in post-training quantization. We additionally design and evaluate an IF4 Multiply-Accumulate (MAC) unit to demonstrate that IF4 can be implemented efficiently in next-generation hardware accelerators. Our code is available at this https URL.
>
---
#### [new 035] Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于RAG系统中的幻觉检测任务，旨在解决模型生成内容与检索上下文不一致的问题。提出RT4CHART框架，通过分层验证实现细粒度错误诊断。**

- **链接: [https://arxiv.org/pdf/2603.27752](https://arxiv.org/pdf/2603.27752)**

> **作者:** Boxi Yu; Yuzhong Zhang; Liting Lin; Lionel Briand; Emir Muñoz
>
> **摘要:** Large language models (LLMs) continue to hallucinate in retrieval-augmented generation (RAG), producing claims that are unsupported by or conflict with the retrieved context. Detecting such errors remains challenging when faithfulness is evaluated solely with respect to the retrieved context. Existing approaches either provide coarse-grained, answer-level scores or focus on open-domain factuality, often lacking fine-grained, evidence-grounded diagnostics. We present RT4CHART, a retromorphic testing framework for context-faithfulness assessment. RT4CHART decomposes model outputs into independently verifiable claims and performs hierarchical, local-to-global verification against the retrieved context. Each claim is assigned one of three labels: entailed, contradicted, or baseless. Furthermore, RT4CHART maps claim-level decisions back to specific answer spans and retrieves explicit supporting or refuting evidence from the context, enabling fine-grained and interpretable auditing. We evaluate RT4CHART on RAGTruth++ (408 samples) and RAGTruth-Enhance (2,675 samples), a newly re-annotated benchmark. RT4CHART achieves the best answer-level hallucination detection F1 among all baselines. On RAGTruth++, it reaches an F1 score of 0.776, outperforming the strongest baseline by 83%. On RAGTruth-Enhance, it achieves a span-level F1 of 47.5%. Ablation studies show that the hierarchical verification design is the primary driver of performance gains. Finally, our re-annotation reveals 1.68x more hallucination cases than the original labels, suggesting that existing benchmarks substantially underestimate the prevalence of hallucinations.
>
---
#### [new 036] Beyond Cosine Similarity: Zero-Initialized Residual Complex Projection for Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，解决文本中方面与情感混杂的问题。提出ZRCP方法和抗碰撞损失函数，将文本投影到复数空间，分离情感极性，提升高频率方面的性能。**

- **链接: [https://arxiv.org/pdf/2603.28205](https://arxiv.org/pdf/2603.28205)**

> **作者:** Yijin Wang; Fandi Sun
>
> **摘要:** Aspect-Based Sentiment Analysis (ABSA) is fundamentally challenged by representation entanglement, where aspect semantics and sentiment polarities are often conflated in real-valued embedding spaces. Furthermore, standard contrastive learning suffers from false-negative collisions, severely degrading performance on high-frequency aspects. In this paper, we propose a novel framework featuring a Zero-Initialized Residual Complex Projection (ZRCP) and an Anti-collision Masked Angle Loss,inspired by quantum projection and entanglement ideas. Our approach projects textual features into a complex semantic space, systematically utilizing the phase to disentangle sentiment polarities while allowing the amplitude to encode the semantic intensity and lexical richness of subjective descriptions. To tackle the collision bottleneck, we introduce an anti-collision mask that elegantly preserves intra-polarity aspect cohesion while expanding the inter-polarity discriminative margin by over 50%. Experimental results demonstrate that our framework achieves a state-of-the-art Macro-F1 score of 0.8851. Deep geometric analyses further reveal that explicitly penalizing the complex amplitude catastrophically over-regularizes subjective representations, proving that our unconstrained-amplitude and phase-driven objective is crucial for robust, fine-grained sentiment disentanglement.
>
---
#### [new 037] Article and Comment Frames Shape the Quality of Online Comments
- **分类: cs.CL**

- **简介: 该论文属于信息处理任务，研究文章框架如何影响在线评论质量。通过分析100万条评论，发现文章框架显著影响评论健康度，为提升网络讨论质量提供方法支持。**

- **链接: [https://arxiv.org/pdf/2603.27889](https://arxiv.org/pdf/2603.27889)**

> **作者:** Matteo Guida; Yulia Otmakhova; Eduard Hovy; Lea Frermann
>
> **摘要:** Framing theory posits that how information is presented shapes audience responses, but computational work has largely ignored audience reactions. While recent work showed that article framing systematically shapes the content of reader responses, this paper asks: Does framing also affect response quality? Analyzing 1M comments across 2.7K news articles, we operationalize quality as comment health (constructive, good-faith contributions). We find that article frames significantly predict comment health while controlling for topic, and that comments that adopt the article frame are healthier than those that depart from it. Further, unhealthy top-level comments tend to generate more unhealthy responses, independent of the frame being used in the comment. Our results establish a link between framing theory and discourse quality, laying the groundwork for downstream applications. We illustrate this potential with a proactive frame-aware LLM- based system to mitigate unhealthy discourse
>
---
#### [new 038] Introducing MELI: the Mandarin-English Language Interview Corpus
- **分类: cs.CL**

- **简介: 该论文介绍MELI语料库，用于双语语音研究，解决跨语言和跨说话人的声学比较问题，包含中英文对话数据及标注信息。**

- **链接: [https://arxiv.org/pdf/2603.27043](https://arxiv.org/pdf/2603.27043)**

> **作者:** Suyuan Liu; Molly Babel
>
> **备注:** Accepted at LREC 2026 (14th International Conference on Language Resources and Evaluation), to appear in the conference proceedings
>
> **摘要:** We introduce the Mandarin-English Language Interview (MELI) Corpus, an open-source resource of 29.8 hours of speech from 51 Mandarin-English bilingual speakers. MELI combines matched sessions in Mandarin and English with two speaking styles: read sentences and spontaneous interviews about language varieties, standardness, and learning experiences. Audio was recorded at 44.1 kHz (16-bit, stereo). Interviews were fully transcribed, force-aligned at word and phone levels, and anonymized. Descriptively, the Mandarin component totals ~14.7 hours (mean duration 17.3 minutes) and the English component ~15.1 hours (mean duration 17.8 minutes). We report token/type statistics for each language and document code-switching patterns (frequent in Mandarin sessions; more limited in English sessions). The corpus design supports within-/cross-speaker, within/cross-language acoustic comparison and links acoustics to speakers' stated language attitudes, enabling both quantitative and qualitative analyses. The MELI Corpus will be released with transcriptions, alignments, metadata, scans of labelled maps and documentation under a CC BY-NC 4.0 license.
>
---
#### [new 039] KazByte: Adapting Qwen models to Kazakh via Byte-level Adapter
- **分类: cs.CL; math.NA**

- **简介: 该论文属于自然语言处理任务，旨在解决Qwen模型在哈萨克语上的适配问题。通过字节级适配器减少分词带来的计算负担和上下文限制，提升模型对哈萨克语形态的处理能力。**

- **链接: [https://arxiv.org/pdf/2603.27859](https://arxiv.org/pdf/2603.27859)**

> **作者:** Rauan Akylzhanov
>
> **备注:** Technical announcement
>
> **摘要:** Large language models fragment Kazakh text into many more tokens than equivalent English text, because their tokenizers were built for high-resource languages. This tokenizer tax inflates compute, shortens the effective context window, and weakens the model's grip on Kazakh morphology. We propose to bypass the tokenizer entirely by feeding raw bytes through a small adapter that learns to speak the internal language of a frozen Qwen2.5-7B. Once the adapter is trained, we freeze it and fine-tune only the attention layers of Qwen on Kazakh text. Our central hypothesis is that this two-stage process -- first teach the interface, then adapt the model -- should match or exceed the accuracy of the original Qwen2.5-7B on standard Kazakh benchmarks. This report describes the ByteKaz architecture and training protocol. Empirical validation is ongoing; this version stakes the design and hypotheses for the record.
>
---
#### [new 040] Debiasing Large Language Models toward Social Factors in Online Behavior Analytics through Prompt Knowledge Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会行为分析任务，旨在解决大语言模型在社交情境中的偏见问题。通过引入用户目标和消息上下文作为提示知识，减少模型的社会归因偏差。**

- **链接: [https://arxiv.org/pdf/2603.27057](https://arxiv.org/pdf/2603.27057)**

> **作者:** Hossein Salemi; Jitin Krishnan; Hemant Purohit
>
> **备注:** This is a preprint of the accepted paper for publication in IEEE Transactions on Computational Social Systems
>
> **摘要:** Attribution theory explains how individuals interpret and attribute others' behavior in a social context by employing personal (dispositional) and impersonal (situational) causality. Large Language Models (LLMs), trained on human-generated corpora, may implicitly mimic this social attribution process in social contexts. However, the extent to which LLMs utilize these causal attributions in their reasoning remains underexplored. Although using reasoning paradigms, such as Chain-of-Thought (CoT), has shown promising results in various tasks, ignoring social attribution in reasoning could lead to biased responses by LLMs in social contexts. In this study, we investigate the impact of incorporating a user's goal as knowledge to infer dispositional causality and message context to infer situational causality on LLM performance. To this end, we introduce a scalable method to mitigate such biases by enriching the instruction prompts for LLMs with two prompt aids using social-attribution knowledge, based on the context and goal of a social media message. This method improves the model performance while reducing the social-attribution bias of the LLM in the reasoning on zero-shot classification tasks for behavior analytics applications. We empirically show the benefits of our method across two tasks-intent detection and theme detection on social media in the disaster domain-when considering the variability of disaster types and multiple languages of social media. Our experiments highlight the biases of three open-source LLMs: Llama3, Mistral, and Gemma, toward social attribution, and show the effectiveness of our mitigation strategies.
>
---
#### [new 041] A tree interpretation of arc standard dependency derivation
- **分类: cs.CL**

- **简介: 该论文研究依赖句法分析任务，解决如何将弧标准转换序列解释为有序树结构的问题。通过定义确定性树更新，实现稳定依赖恢复与项目性表征。**

- **链接: [https://arxiv.org/pdf/2603.27459](https://arxiv.org/pdf/2603.27459)**

> **作者:** Zihao Huang; Ai Ka Lee; Jungyeul Park
>
> **摘要:** We show that arc-standard derivations for projective dependency trees determine a unique ordered tree representation with surface-contiguous yields and stable lexical anchoring. Each \textsc{shift}, \textsc{leftarc}, and \textsc{rightarc} transition corresponds to a deterministic tree update, and the resulting hierarchical object uniquely determines the original dependency arcs. We further show that this representation characterizes projectivity: a single-headed dependency tree admits such a contiguous ordered representation if and only if it is projective. The proposal is derivational rather than convertive. It interprets arc-standard transition sequences directly as ordered tree construction, rather than transforming a completed dependency graph into a phrase-structure output. For non-projective inputs, the same interpretation can be used in practice via pseudo-projective lifting before derivation and inverse decoding after recovery. A proof-of-concept implementation in a standard neural transition-based parser shows that the mapped derivations are executable and support stable dependency recovery.
>
---
#### [new 042] TIEG-Youpu Solution for NeurIPS 2022 WikiKG90Mv2-LSC
- **分类: cs.CL**

- **简介: 该论文属于知识图谱嵌入任务，旨在提升大规模知识图谱的链接预测效果。通过改进检索与重排序流程，提出新模型提高MRR指标。**

- **链接: [https://arxiv.org/pdf/2603.28512](https://arxiv.org/pdf/2603.28512)**

> **作者:** Feng Nie; Zhixiu Ye; Sifa Xie; Shuang Wu; Xin Yuan; Liang Yao; Jiazhen Peng; Xu Cheng
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** WikiKG90Mv2 in NeurIPS 2022 is a large encyclopedic knowledge graph. Embedding knowledge graphs into continuous vector spaces is important for many practical applications, such as knowledge acquisition, question answering, and recommendation systems. Compared to existing knowledge graphs, WikiKG90Mv2 is a large scale knowledge graph, which is composed of more than 90 millions of entities. Both efficiency and accuracy should be considered when building graph embedding models for knowledge graph at scale. To this end, we follow the retrieve then re-rank pipeline, and make novel modifications in both retrieval and re-ranking stage. Specifically, we propose a priority infilling retrieval model to obtain candidates that are structurally and semantically similar. Then we propose an ensemble based re-ranking model with neighbor enhanced representations to produce final link prediction results among retrieved candidates. Experimental results show that our proposed method outperforms existing baseline methods and improves MRR of validation set from 0.2342 to 0.2839.
>
---
#### [new 043] Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究对齐大模型中的过度拒绝现象。旨在解决过早拒绝安全指令的问题，通过分析拒绝的表示几何，发现其与任务相关，需进行任务特定干预。**

- **链接: [https://arxiv.org/pdf/2603.27518](https://arxiv.org/pdf/2603.27518)**

> **作者:** Utsav Maskey; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Aligned language models that are trained to refuse harmful requests also exhibit over-refusal: they decline safe instructions that seemingly resemble harmful instructions. A natural approach is to ablate the global refusal direction, steering the hidden-state vectors away or towards the harmful-refusal examples, but this corrects over-refusal only incidentally while disrupting the broader refusal mechanism. In this work, we analyse the representational geometry of both refusal types to understand why this happens. We show that harmful-refusal directions are task-agnostic and can be captured by a single global vector, whereas over-refusal directions are task-dependent: they reside within the benign task-representation clusters, vary across tasks, and span a higher-dimensional subspace. Linear probing confirms that the two refusal types are representationally distinct from the early transformer layers. These findings provide a mechanistic explanation of why global direction ablation alone cannot address over-refusal, and establish that task-specific geometric interventions are necessary.
>
---
#### [new 044] A gentle tutorial and a structured reformulation of Bock's algorithm for minimum directed spanning trees
- **分类: cs.CL**

- **简介: 论文介绍Bock算法的重构与教学，用于最小有向生成树问题，解决非投射依存句法解析中的精确解码问题。**

- **链接: [https://arxiv.org/pdf/2603.27530](https://arxiv.org/pdf/2603.27530)**

> **作者:** Yuxi Wang; Jungyeul Park
>
> **摘要:** This paper presents a gentle tutorial and a structured reformulation of Bock's 1971 Algol procedure for constructing minimum directed spanning trees. Our aim is to make the original algorithm readable and reproducible for modern readers, while highlighting its relevance as an exact decoder for nonprojective graph based dependency parsing. We restate the minimum arborescence objective in Bock's notation and provide a complete line by line execution trace of the original ten node example, extending the partial trace given in the source paper from initialization to termination. We then introduce a structured reformulation that makes explicit the procedure's phase structure, maintained state, and control flow, while preserving the logic of the original method. As a further illustration, we include a worked example adapted from {jurafsky-martin-2026-book} for dependency parsing, showing how a maximum weight arborescence problem is reduced to Bock's minimum cost formulation by a standard affine transformation and traced under the same state variables.
>
---
#### [new 045] Courtroom-Style Multi-Agent Debate with Progressive RAG and Role-Switching for Controversial Claim Verification
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于争议性声明验证任务，旨在解决LLM在高风险场景下的不可靠问题。通过结构化多智能体辩论框架PROClaim，结合渐进式RAG和角色切换，提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2603.28488](https://arxiv.org/pdf/2603.28488)**

> **作者:** Masnun Nuha Chowdhury; Nusrat Jahan Beg; Umme Hunny Khan; Syed Rifat Raiyan; Md Kamrul Hasan; Hasan Mahmud
>
> **备注:** Under review, 7 figures, 13 tables
>
> **摘要:** Large language models (LLMs) remain unreliable for high-stakes claim verification due to hallucinations and shallow reasoning. While retrieval-augmented generation (RAG) and multi-agent debate (MAD) address this, they are limited by one-pass retrieval and unstructured debate dynamics. We propose a courtroom-style multi-agent framework, PROClaim, that reformulates verification as a structured, adversarial deliberation. Our approach integrates specialized roles (e.g., Plaintiff, Defense, Judge) with Progressive RAG (P-RAG) to dynamically expand and refine the evidence pool during the debate. Furthermore, we employ evidence negotiation, self-reflection, and heterogeneous multi-judge aggregation to enforce calibration, robustness, and diversity. In zero-shot evaluations on the Check-COVID benchmark, PROClaim achieves 81.7% accuracy, outperforming standard multi-agent debate by 10.0 percentage points, with P-RAG driving the primary performance gains (+7.5 pp). We ultimately demonstrate that structural deliberation and model heterogeneity effectively mitigate systematic biases, providing a robust foundation for reliable claim verification. Our code and data are publicly available at this https URL.
>
---
#### [new 046] LombardoGraphia: Automatic Classification of Lombard Orthography Variants
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的分类任务，旨在解决Lombard语不同拼写系统的问题。作者构建了首个标注数据集，并训练分类模型以自动识别拼写变体。**

- **链接: [https://arxiv.org/pdf/2603.28418](https://arxiv.org/pdf/2603.28418)**

> **作者:** Edoardo Signoroni; Pavel Rychlý
>
> **备注:** To be published at LREC 2026
>
> **摘要:** Lombard, an underresourced language variety spoken by approximately 3.8 million people in Northern Italy and Southern Switzerland, lacks a unified orthographic standard. Multiple orthographic systems exist, creating challenges for NLP resource development and model training. This paper presents the first study of automatic Lombard orthography classification and LombardoGraphia, a curated corpus of 11,186 Lombard Wikipedia samples tagged across 9 orthographic variants, and models for automatic orthography classification. We curate the dataset, processing and filtering raw Wikipedia content to ensure text suitable for orthographic analysis. We train 24 traditional and neural classification models with various features and encoding levels. Our best models achieve 96.06% and 85.78% overall and average class accuracy, though performance on minority classes remains challenging due to data imbalance. Our work provides crucial infrastructure for building variety-aware NLP resources for Lombard.
>
---
#### [new 047] Merge and Conquer: Instructing Multilingual Models by Adding Target Language Weights
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言中大模型性能不足的问题。通过模型合并，提升指令调优模型的多语言能力，减少计算成本。**

- **链接: [https://arxiv.org/pdf/2603.28263](https://arxiv.org/pdf/2603.28263)**

> **作者:** Eneko Valero; Maria Ribalta i Albado; Oscar Sainz; Naiara Perez; German Rigau
>
> **备注:** This paper was accepted at the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** Large Language Models (LLMs) remain heavily centered on English, with limited performance in low-resource languages. Existing adaptation approaches, such as continual pre-training, demand significant computational resources. In the case of instructed models, high-quality instruction data is also required, both of which are often inaccessible for low-resource language communities. Under these constraints, model merging offers a lightweight alternative, but its potential in low-resource contexts has not been systematically explored. In this work, we explore whether it is possible to transfer language knowledge to an instruction-tuned LLM by merging it with a language-specific base model, thereby eliminating the need of language-specific instructions and repeated fine-tuning processes whenever stronger instructed variants become available. Through experiments covering four Iberian languages (Basque, Catalan, Galician, and Spanish) and two model families, we show that merging enables effective instruction following behavior in new languages and even supports multilingual capability through the combination of multiple language-specific models. Our results indicate that model merging is a viable and efficient alternative to traditional adaptation methods for low-resource languages, achieving competitive performance while greatly reducing computational cost.
>
---
#### [new 048] LogicDiff: Logic-Guided Denoising Improves Reasoning in Masked Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决MDLMs推理性能差的问题。通过逻辑引导的去噪方法优化解码顺序，提升推理准确率。**

- **链接: [https://arxiv.org/pdf/2603.26771](https://arxiv.org/pdf/2603.26771)**

> **作者:** Shaik Aman
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** Masked diffusion language models (MDLMs) generate text by iteratively unmasking tokens from a fully masked sequence, offering parallel generation and bidirectional context. However, their standard confidence-based unmasking strategy systematically defers high-entropy logical connective tokens, the critical branching points in reasoning chains, leading to severely degraded reasoning performance. We introduce LogicDiff, an inference-time method that replaces confidence-based unmasking with logic-role-guided unmasking. A lightweight classification head (4.2M parameters, 0.05% of the base model) predicts the logical role of each masked position (premise, connective, derived step, conclusion, or filler) from the base model's hidden states with 98.4% accuracy. A dependency-ordered scheduler then unmasks tokens in logical dependency order: premises first, then connectives, then derived steps, then conclusions. Without modifying a single parameter of the base model and without any reinforcement learning or task-specific training, LogicDiff improves LLaDA-8B-Instruct accuracy from 22.0% to 60.7% on GSM8K (+38.7 percentage points) and from 23.6% to 29.2% on MATH-500 (+5.6 pp), with less than 6% speed overhead. Our results demonstrate that a substantial portion of the reasoning deficit in MDLMs is attributable to suboptimal token unmasking order, not to limitations of the model's learned representations.
>
---
#### [new 049] Story2Proposal: A Scaffold for Structured Scientific Paper Writing
- **分类: cs.CL**

- **简介: 该论文提出Story2Proposal，解决科学论文生成中的结构与视觉一致性问题，通过多代理框架确保内容与图表同步。**

- **链接: [https://arxiv.org/pdf/2603.27065](https://arxiv.org/pdf/2603.27065)**

> **作者:** Zhuoyang Qian; Wei Shi; Xu Lin; Li Ling; Meng Luo; Ziming Wang; Zhiwei Zhang; Tengyue Xu; Gaoge Liu; Zhentao Zhang; Shuo Zhang; Ziqi Wang; Zheng Feng; Yan Luo; Shu Xu; Yongjin Chen; Zhibo Feng; Zhuo Chen; Bruce Yuan; Biao Wu; Harry Wang; Kris Chen
>
> **备注:** 10 pages, 4 figures,
>
> **摘要:** Generating scientific manuscripts requires maintaining alignment between narrative reasoning, experimental evidence, and visual artifacts across the document lifecycle. Existing language-model generation pipelines rely on unconstrained text synthesis with validation applied only after generation, often producing structural drift, missing figures or tables, and cross-section inconsistencies. We introduce Story2Proposal, a contract-governed multi-agent framework that converts a research story into a structured manuscript through coordinated agents operating under a persistent shared visual contract. The system organizes architect, writer, refiner, and renderer agents around a contract state that tracks section structure and registered visual elements, while evaluation agents supply feedback in a generate evaluate adapt loop that updates the contract during generation. Experiments on tasks derived from the Jericho research corpus show that Story2Proposal achieved an expert evaluation score of 6.145 versus 3.963 for DirectChat (+2.182) across GPT, Claude, Gemini, and Qwen backbones. Compared with the structured generation baseline Fars, Story2Proposal obtained an average score of 5.705 versus 5.197, indicating improved structural consistency and visual alignment.
>
---
#### [new 050] Structural Stress and Learned Helplessness in Afghanistan: A Multi-Layer Analysis of the AFSTRESS Dari Corpus
- **分类: cs.CL; cs.SI**

- **简介: 该论文介绍AFSTRESS语料库，用于分析阿富汗人压力的多层结构。属于情感与压力分析任务，解决危机下心理状态的计算研究问题，开展多标签分类与社会心理分析。**

- **链接: [https://arxiv.org/pdf/2603.27233](https://arxiv.org/pdf/2603.27233)**

> **作者:** Jawid Ahmad Baktash; Mursal Dawodi; Nadira Ahmadi
>
> **备注:** 16 pages, 7 figures, 3 tables. Introduces AFSTRESS, the first multi-label Dari corpus of self-reported stress narratives (737 responses). Includes computational benchmarks, social science analysis of structural stress, and psychological modeling (learned helplessness, chronic stress, emotional cascade)
>
> **摘要:** We introduce AFSTRESS, the first multi-label corpus of self-reported stress narratives in Dari (Eastern Persian), comprising 737 responses collected from Afghan individuals during an ongoing humanitarian crisis. Participants describe experienced stress and select emotion and stressor labels via Dari checklists. The dataset enables analysis at three levels: computational (multi-label classification), social (structural drivers and gender disparities), and psychological (learned helplessness, chronic stress, and emotional cascade patterns). It includes 12 binary labels (5 emotions, 7 stressors), with high label cardinality (5.54) and density (0.462), reflecting complex, multi-dimensional stress. Structural stressors dominate: uncertain future (62.6 percent) and education closure (60.0 percent) exceed emotional states, indicating stress is primarily structurally driven. The strongest co-occurrence is between hopelessness and uncertain future (J = 0.388). Baseline experiments show that character TF-IDF with Linear SVM achieves Micro-F1 = 0.663 and Macro-F1 = 0.651, outperforming ParsBERT and XLM-RoBERTa, while threshold tuning improves Micro-F1 by 10.3 points. AFSTRESS provides the first Dari resource for computational analysis of stress and well-being in a crisis-affected population.
>
---
#### [new 051] SCOPE: Tree-based Self-Correcting Online Log Parsing via Syntactic-Semantic Collaboration
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于日志解析任务，解决传统方法准确性低或效率差的问题。提出SCOPE方法，结合启发式与LLM优势，提升解析准确率与效率。**

- **链接: [https://arxiv.org/pdf/2603.27247](https://arxiv.org/pdf/2603.27247)**

> **作者:** Dongyi Fan; Suqiong Zhang; Lili He; Ming Liu; Yifan Huo
>
> **备注:** Accepted at the 34th International Conference on Program Comprehension (ICPC 2026)
>
> **摘要:** Log parsing is a critical step for automated log analysis in complex systems. Traditional heuristic-based methods offer high efficiency but are limited in accuracy due to overlooking semantic context. In contrast, recent LLM-based parsers improve accuracy via se mantic understanding but incur high latency from frequent model calls. To address this, we propose SCOPE, the first self-correcting online log parsing method that integrates the strengths of both heuristic and LLM-based paradigms. SCOPE introduces a novel bi-directional tree structure that enables efficient template match ing from both forward and reverse directions, resulting in a higher overall matching rate. Additionally, it adopts a two-stage syntactic semantic collaboration framework: a lightweight NLP model first utilizes part-of-speech (POS) information for syntax-based match ing, while the LLM is selectively invoked as a fallback to handle semantically complex cases when uncertainty remains. This design significantly reduces LLM API usage while maintaining high ac curacy, achieving a balance between efficiency and effectiveness. Extensive evaluations on diverse benchmark datasets show that SCOPE outperforms state-of-the-art methods in both accuracy and efficiency. The implementation and datasets are publicly released to facilitate further research.
>
---
#### [new 052] Mitigating Hallucination on Hallucination in RAG via Ensemble Voting
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG系统中“幻觉叠加”问题。通过引入投票机制的VOTE-RAG框架，提升生成结果的可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.27253](https://arxiv.org/pdf/2603.27253)**

> **作者:** Zequn Xie; Zhengyang Sun
>
> **备注:** arXiv admin note: text overlap with arXiv:2505.18581 by other authors
>
> **摘要:** Retrieval-Augmented Generation (RAG) aims to reduce hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, RAG introduces a critical challenge: hallucination on hallucination," where flawed retrieval results mislead the generation model, leading to compounded hallucinations. To address this issue, we propose VOTE-RAG, a novel, training-free framework with a two-stage structure and efficient, parallelizable voting mechanisms. VOTE-RAG includes: (1) Retrieval Voting, where multiple agents generate diverse queries in parallel and aggregate all retrieved documents; (2) Response Voting, where multiple agents independently generate answers based on the aggregated documents, with the final output determined by majority vote. We conduct comparative experiments on six benchmark datasets. Our results show that VOTE-RAG achieves performance comparable to or surpassing more complex frameworks. Additionally, VOTE-RAG features a simpler architecture, is fully parallelizable, and avoids the problem drift" risk. Our work demonstrates that simple, reliable ensemble voting is a superior and more efficient method for mitigating RAG hallucinations.
>
---
#### [new 053] Pashto Common Voice: Building the First Open Speech Corpus for a 60-Million-Speaker Low-Resource Language
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决Pashto语言缺乏公开语音数据的问题。通过社区协作构建了首个大规模开源语音语料库，并提升了语音识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.27021](https://arxiv.org/pdf/2603.27021)**

> **作者:** Hanif Rahman; Shafeeq ur Rehman
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** We present the Pashto Common Voice corpus -- the first large-scale, openly licensed speech resource for Pashto, a language with over 60 million native speakers largely absent from open speech technology. Through a community effort spanning 2022-2025, the corpus grew from 1.5 hours and 5 contributors to 147 total hours and 1,483 unique speakers across ten Mozilla Common Voice releases (CV14-CV23). Speaker participation increased approximately 108-fold between CV17 and CV18, coinciding with a VOA Pashto broadcast campaign. We describe the full methodology: interface localisation, Wikipedia-based sentence extraction with automated filtering, phonemically targeted contributions for the four most frequently dropped Pashto characters, and multi-channel community outreach. MCV23 contains 107,781 clips (60,337 validated; 82.33 validated hours) across 13 content domains. Fine-tuning Whisper Base on the MCV20 yields 13.4% WER on the MCV20 test split, against the published Whisper Base zero-shot WER of 99.0% on Pashto.
>
---
#### [new 054] \textit{Versteasch du mi?} Computational and Socio-Linguistic Perspectives on GenAI, LLMs, and Non-Standard Language
- **分类: cs.CL**

- **简介: 该论文探讨生成式AI与非标准语言的关系，分析其对语言多样性和数字语言鸿沟的影响，旨在推动更公平的数字策略。任务属于社会语言学与计算语言学交叉研究。**

- **链接: [https://arxiv.org/pdf/2603.28213](https://arxiv.org/pdf/2603.28213)**

> **作者:** Verena Platzgummer; John McCrae; Sina Ahmadi
>
> **摘要:** The design of Large Language Models and generative artificial intelligence has been shown to be "unfair" to less-spoken languages and to deepen the digital language divide. Critical sociolinguistic work has also argued that these technologies are not only made possible by prior socio-historical processes of linguistic standardisation, often grounded in European nationalist and colonial projects, but also exacerbate epistemologies of language as "monolithic, monolingual, syntactically standardized systems of meaning". In our paper, we draw on earlier work on the intersections of technology and language policy and bring our respective expertise in critical sociolinguistics and computational linguistics to bear on an interrogation of these arguments. We take two different complexes of non-standard linguistic varieties in our respective repertoires--South Tyrolean dialects, which are widely used in informal communication in South Tyrol, Italy, as well as varieties of Kurdish--as starting points to an interdisciplinary exploration of the intersections between GenAI and linguistic variation and standardisation. We discuss both how LLMs can be made to deal with nonstandard language from a technical perspective, and whether, when or how this can contribute to "democratic and decolonial digital and machine learning strategies", which has direct policy implications.
>
---
#### [new 055] GraphWalker: Agentic Knowledge Graph Question Answering via Synthetic Trajectory Curriculum
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，解决代理训练数据不足与推理泛化问题。提出GraphWalker框架，通过合成路径和分阶段微调提升性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.28533](https://arxiv.org/pdf/2603.28533)**

> **作者:** Shuwen Xu; Yao Xu; Jiaxiang Liu; Chenhao Yuan; Wenshuo Peng; Jun Zhao; Kang Liu
>
> **摘要:** Agentic knowledge graph question answering (KGQA) requires an agent to iteratively interact with knowledge graphs (KGs), posing challenges in both training data scarcity and reasoning generalization. Specifically, existing approaches often restrict agent exploration: prompting-based methods lack autonomous navigation training, while current training pipelines usually confine reasoning to predefined trajectories. To this end, this paper proposes \textit{GraphWalker}, a novel agentic KGQA framework that addresses these challenges through \textit{Automated Trajectory Synthesis} and \textit{Stage-wise Fine-tuning}. GraphWalker adopts a two-stage SFT training paradigm: First, the agent is trained on structurally diverse trajectories synthesized from constrained random-walk paths, establishing a broad exploration prior over the KG; Second, the agent is further fine-tuned on a small set of expert trajectories to develop reflection and error recovery capabilities. Extensive experiments demonstrate that our stage-wise SFT paradigm unlocks a higher performance ceiling for a lightweight reinforcement learning (RL) stage, enabling GraphWalker to achieve state-of-the-art performance on CWQ and WebQSP. Additional results on GrailQA and our constructed GraphWalkerBench confirm that GraphWalker enhances generalization to out-of-distribution reasoning paths. The code is publicly available at this https URL
>
---
#### [new 056] Categorical Perception in Large Language Model Hidden States: Structural Warping at Digit-Count Boundaries
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型隐藏状态中的类别感知现象，探讨其在数字计数边界处的几何扭曲。属于自然语言处理任务，解决模型如何表征类别边界的问题。**

- **链接: [https://arxiv.org/pdf/2603.28258](https://arxiv.org/pdf/2603.28258)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 25 pages, 5 figures, 7 tables. Pre-registered on OSF (this http URL). Code at this https URL
>
> **摘要:** Categorical perception (CP) -- enhanced discriminability at category boundaries -- is among the most studied phenomena in perceptual psychology. This paper reports that analogous geometric warping occurs in the hidden-state representations of large language models (LLMs) processing Arabic numerals. Using representational similarity analysis across six models from five architecture families, the study finds that a CP-additive model (log-distance plus a boundary boost) fits the representational geometry better than a purely continuous model at 100% of primary layers in every model tested. The effect is specific to structurally defined boundaries (digit-count transitions at 10 and 100), absent at non-boundary control positions, and absent in the temperature domain where linguistic categories (hot/cold) lack a tokenisation discontinuity. Two qualitatively distinct signatures emerge: "classic CP" (Gemma, Qwen), where models both categorise explicitly and show geometric warping, and "structural CP" (Llama, Mistral, Phi), where geometry warps at the boundary but models cannot report the category distinction. This dissociation is stable across boundaries and is a property of the architecture, not the stimulus. Structural input-format discontinuities are sufficient to produce categorical perception geometry in LLMs, independently of explicit semantic category knowledge.
>
---
#### [new 057] What can LLMs tell us about the mechanisms behind polarity illusions in humans? Experiments across model scales and training steps
- **分类: cs.CL**

- **简介: 该论文研究LLMs中极性错觉的机制，通过实验分析模型规模对两种错觉的影响，探讨人类句法处理的理论解释。属于自然语言处理任务，解决极性错觉生成机制问题。**

- **链接: [https://arxiv.org/pdf/2603.27855](https://arxiv.org/pdf/2603.27855)**

> **作者:** Dario Paape
>
> **摘要:** I use the Pythia scaling suite (Biderman et al. 2023) to investigate if and how two well-known polarity illusions, the NPI illusion and the depth charge illusion, arise in LLMs. The NPI illusion becomes weaker and ultimately disappears as model size increases, while the depth charge illusion becomes stronger in larger models. The results have implications for human sentence processing: it may not be necessary to assume "rational inference" mechanisms that convert ill-formed sentences into well-formed ones to explain polarity illusions, given that LLMs cannot plausibly engage in this kind of reasoning, especially at the implicit level of next-token prediction. On the other hand, shallow, "good enough" processing and/or partial grammaticalization of prescriptively ungrammatical structures may both occur in LLMs. I propose a synthesis of different theoretical accounts that is rooted in the basic tenets of construction grammar.
>
---
#### [new 058] Understanding Teacher Revisions of Large Language Model-Generated Feedback
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究教师如何修改AI生成的反馈，属于教育AI任务。旨在了解教师修订行为，解决AI反馈与教学需求匹配问题。通过分析数据，发现教师修订模式及影响。**

- **链接: [https://arxiv.org/pdf/2603.27806](https://arxiv.org/pdf/2603.27806)**

> **作者:** Conrad Borchers; Luiz Rodrigues; Newarney Torrezão da Costa; Cleon Xavier; Rafael Ferreira Mello
>
> **备注:** Accepted as full paper to the 27th International Conference on Artificial Intelligence in Education (AIED 2026)
>
> **摘要:** Large language models (LLMs) increasingly generate formative feedback for students, yet little is known about how teachers revise this feedback before it reaches learners. Teachers' revisions shape what students receive, making revision practices central to evaluating AI classroom tools. We analyze a dataset of 1,349 instances of AI-generated feedback and corresponding teacher-edited explanations from 117 teachers. We examine (i) textual characteristics associated with teacher revisions, (ii) whether revision decisions can be predicted from the AI feedback text, and (iii) how revisions change the pedagogical type of feedback delivered. First, we find that teachers accept AI feedback without modification in about 80% of cases, while edited feedback tends to be significantly longer and subsequently shortened by teachers. Editing behavior varies substantially across teachers: about 50% never edit AI feedback, and only about 10% edit more than two-thirds of feedback instances. Second, machine learning models trained only on the AI feedback text as input features, using sentence embeddings, achieve fair performance in identifying which feedback will be revised (AUC=0.75). Third, qualitative coding shows that when revisions occur, teachers often simplify AI-generated feedback, shifting it away from high-information explanations toward more concise, corrective forms. Together, these findings characterize how teachers engage with AI-generated feedback in practice and highlight opportunities to design feedback systems that better align with teacher priorities while reducing unnecessary editing effort.
>
---
#### [new 059] TAPS: Task Aware Proposal Distributions for Speculative Sampling
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了推测解码中草稿模型训练数据对性能的影响，属于自然语言生成任务。旨在解决草稿模型与下游任务匹配度问题，通过实验分析不同训练数据的效果及融合策略。**

- **链接: [https://arxiv.org/pdf/2603.27027](https://arxiv.org/pdf/2603.27027)**

> **作者:** Mohamad Zbib; Mohamad Bazzi; Ammar Mohanna; Hasan Abed Al Kader Hammoud; Bernard Ghanem
>
> **备注:** 21 pages, 11 figures. Code: this https URL Weights: this https URL Datasets: this https URL
>
> **摘要:** Speculative decoding accelerates autoregressive generation by letting a lightweight draft model propose future tokens that a larger target model then verifies in parallel. In practice, however, draft models are usually trained on broad generic corpora, which leaves it unclear how much speculative decoding quality depends on the draft training distribution. We study this question with lightweight HASS and EAGLE-2 drafters trained on MathInstruct, ShareGPT, and mixed-data variants, evaluated on MT-Bench, GSM8K, MATH-500, and SVAMP. Measured by acceptance length, task-specific training yields clear specialization: MathInstruct-trained drafts are strongest on reasoning benchmarks, while ShareGPT-trained drafts are strongest on MT-Bench. Mixed-data training improves robustness, but larger mixtures do not dominate across decoding temperatures. We also study how to combine specialized drafters at inference time. Naive checkpoint averaging performs poorly, whereas confidence-based routing improves over single-domain drafts and merged-tree verification yields the highest acceptance length overall for both backbones. Finally, confidence is a more useful routing signal than entropy: rejected tokens tend to have higher entropy, but confidence produces much clearer benchmark-level routing decisions. These results show that speculative decoding quality depends not only on draft architecture, but also on the match between draft training data and downstream workload, and that specialized drafters are better combined at inference time than in weight space.
>
---
#### [new 060] The Cognitive Divergence: AI Context Windows, Human Attention Decline, and the Delegation Feedback Loop
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; q-bio.NC**

- **简介: 论文探讨AI与人类认知能力的不对称发展，分析AI上下文窗口扩展与人类注意力下降的动态关系，提出“认知分化”概念及“委托反馈循环”假设，旨在理解并研究这一趋势的影响。**

- **链接: [https://arxiv.org/pdf/2603.26707](https://arxiv.org/pdf/2603.26707)**

> **作者:** Netanel Eliav
>
> **备注:** 28 pages, 1 figure, 5 tables. Preprint, not peer reviewed
>
> **摘要:** This paper documents and theorises a self-reinforcing dynamic between two measurable trends: the exponential expansion of large language model (LLM) context windows and the secular contraction of human sustained-attention capacity. We term the resulting asymmetry the Cognitive Divergence. AI context windows have grown from 512 tokens in 2017 to 2,000,000 tokens by 2026 (factor ~3,906; fitted lambda = 0.59/yr; doubling time ~14 months). Over the same period, human Effective Context Span (ECS) -- a token-equivalent measure derived from validated reading-rate meta-analysis (Brysbaert, 2019) and an empirically motivated Comprehension Scaling Factor -- has declined from approximately 16,000 tokens (2004 baseline) to an estimated 1,800 tokens (2026, extrapolated from longitudinal behavioural data ending 2020 (Mark, 2023); see Section 9 for uncertainty discussion). The AI-to-human ratio grew from near parity at the ChatGPT launch (November 2022) to 556--1,111x raw and 56--111x quality-adjusted, after accounting for retrieval degradation (Liu et al., 2024; Chroma, 2025). Beyond documenting this divergence, the paper introduces the Delegation Feedback Loop hypothesis: as AI capability grows, the cognitive threshold at which humans delegate to AI falls, extending to tasks of negligible demand; the resulting reduction in cognitive practice may further attenuate the capacities already documented as declining (Gerlich, 2025; Kim et al., 2026; Kosmyna et al., 2025). Neither trend reverses spontaneously. The paper characterises the divergence statistically, reviews neurobiological mechanisms across eight peer-reviewed neuroimaging studies, presents empirical evidence bearing on the delegation threshold, and proposes a research agenda centred on a validated ECS psychometric instrument and longitudinal study of AI-mediated cognitive change.
>
---
#### [new 061] The Last Fingerprint: How Markdown Training Shapes LLM Prose
- **分类: cs.CL**

- **简介: 论文探讨LLM生成文本中em dash的出现与markdown训练的关系，属于自然语言处理任务。解决AI文本特征识别问题，通过实验验证em dash是markdown残留，可作为微调方法的诊断指标。**

- **链接: [https://arxiv.org/pdf/2603.27006](https://arxiv.org/pdf/2603.27006)**

> **作者:** E. M. Freeburg
>
> **备注:** 14 pages, 3 tables. Code and data: this https URL
>
> **摘要:** Large language models produce em dashes at varying rates, and the observation that some models "overuse" them has become one of the most widely discussed markers of AI-generated text. Yet no mechanistic account of this pattern exists, and the parallel observation that LLMs default to markdown-formatted output has never been connected to it. We propose that the em dash is markdown leaking into prose -- the smallest surviving unit of the structural orientation that LLMs acquire from markdown-saturated training corpora. We present a five-step genealogy connecting training data composition, structural internalization, the dual-register status of the em dash, and post-training amplification. We test this with a two-condition suppression experiment across twelve models from five providers (Anthropic, OpenAI, Meta, Google, DeepSeek): when models are instructed to avoid markdown formatting, overt features (headers, bullets, bold) are eliminated or nearly eliminated, but em dashes persist -- except in Meta's Llama models, which produce none at all. Em dash frequency and suppression resistance vary from 0.0 per 1,000 words (Llama) to 9.1 (GPT-4.1 under suppression), functioning as a signature of the specific fine-tuning procedure applied. A three-condition suppression gradient shows that even explicit em dash prohibition fails to eliminate the artifact in some models, and a base-vs-instruct comparison confirms that the latent tendency exists pre-RLHF. These findings connect two previously isolated online discourses and reframe em dash frequency as a diagnostic of fine-tuning methodology rather than a stylistic defect.
>
---
#### [new 062] Compressing Transformer Language Models via Matrix Product Operator Decomposition: A Case Study on PicoGPT
- **分类: cs.CL; physics.data-an**

- **简介: 该论文属于自然语言处理中的模型压缩任务，旨在解决Transformer模型参数量大、部署成本高的问题。通过矩阵乘积算子分解方法对PicoGPT进行压缩，有效减少参数量并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.28534](https://arxiv.org/pdf/2603.28534)**

> **作者:** Younes Javanmard; Tanmoy Pandit; Masoud Mardani
>
> **摘要:** Transformer-based language models achieve strong performance across NLP tasks, but their quadratic parameter scaling with hidden dimension makes deployment on resource-constrained hardware expensive. We study Matrix Product Operator (MPO) decomposition as a principled compression method for transformers. MPO factorises weight matrices into chains of low-rank cores, with approximation quality controlled by the bond dimension chi. We replace every this http URL layer in PicoGPT, a GPT-2-style character-level language model with about 1M parameters, with an MPOLinear module parameterised as an MPO chain. Cores are initialised either by TT-SVD from pretrained dense weights or from random initialisation, and trained using standard PyTorch autograd without a custom backward pass. We derive balanced factorisation schemes for the five distinct weight shapes in PicoGPT and evaluate bond dimensions chi in {4, 8, 16, 32} on Tiny Shakespeare. MPO compression achieves up to 13x compression per transformer block at chi = 4. At chi = 16, the model uses 191,872 parameters instead of 1,020,224 while retaining 97.7% of baseline token accuracy (51.6% vs 52.8%). Reconstruction error follows the expected trend and is lower for three-site than two-site factorisations at the same bond dimension. The chi = 8 model gives the best accuracy per parameter, exceeding the dense baseline by 2.7x on this metric. These results show that MPO parameterisation is a practical and theoretically grounded alternative to low-rank methods and unstructured pruning for transformer compression.
>
---
#### [new 063] Learning to Predict Future-Aligned Research Proposals with Language Models
- **分类: cs.CL**

- **简介: 该论文属于科研提案生成任务，旨在解决LLM生成提案质量评估难题。通过构建未来对齐评分体系，提升提案的创新性与合理性。**

- **链接: [https://arxiv.org/pdf/2603.27146](https://arxiv.org/pdf/2603.27146)**

> **作者:** Heng Wang; Pengcheng Jiang; Jiashuo Sun; Zhiyi Shi; Haofei Yu; Jiawei Han; Heng Ji
>
> **摘要:** Large language models (LLMs) are increasingly used to assist ideation in research, but evaluating the quality of LLM-generated research proposals remains difficult: novelty and soundness are hard to measure automatically, and large-scale human evaluation is costly. We propose a verifiable alternative by reframing proposal generation as a time-sliced scientific forecasting problem. Given a research question and inspiring papers available before a cutoff time, the model generates a structured proposal and is evaluated by whether it anticipates research directions that appear in papers published after the time. We operationalize this objective with the Future Alignment Score (FAS), computed via retrieval and LLM-based semantic scoring against a held-out future corpus. To train models, we build a time-consistent dataset of 17,771 papers from targets and their pre-cutoff citations, and synthesize reasoning traces that teach gap identification and inspiration borrowing. Across Llama-3.1 and Qwen2.5 models, future-aligned tuning improves future alignment over unaligned baselines (up to +10.6% overall FAS), and domain-expert human evaluation corroborates improved proposal quality. Finally, we demonstrate practical impact by implementing two model-generated proposals with a code agent, obtaining 4.17% accuracy gain on MATH from a new prompting strategy and consistent improvements for a novel model-merging method.
>
---
#### [new 064] Can Large Language Models Simulate Human Cognition Beyond Behavioral Imitation?
- **分类: cs.CL**

- **简介: 该论文属于AI认知模拟任务，旨在探讨LLMs是否能超越行为模仿实现人类认知模拟。通过构建跨领域、时间迁移的基准，评估LLMs的认知一致性。**

- **链接: [https://arxiv.org/pdf/2603.27694](https://arxiv.org/pdf/2603.27694)**

> **作者:** Yuxuan Gu; Lunjun Liu; Xiaocheng Feng; Kun Zhu; Weihong Zhong; Lei Huang; Bing Qin
>
> **摘要:** An essential problem in artificial intelligence is whether LLMs can simulate human cognition or merely imitate surface-level behaviors, while existing datasets suffer from either synthetic reasoning traces or population-level aggregation, failing to capture authentic individual cognitive patterns. We introduce a benchmark grounded in the longitudinal research trajectories of 217 researchers across diverse domains of artificial intelligence, where each author's scientific publications serve as an externalized representation of their cognitive processes. To distinguish whether LLMs transfer cognitive patterns or merely imitate behaviors, our benchmark deliberately employs a cross-domain, temporal-shift generalization setting. A multidimensional cognitive alignment metric is further proposed to assess individual-level cognitive consistency. Through systematic evaluation of state-of-the-art LLMs and various enhancement techniques, we provide a first-stage empirical study on the questions: (1) How well do current LLMs simulate human cognition? and (2) How far can existing techniques enhance these capabilities?
>
---
#### [new 065] Top-down string-to-dependency Neural Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决长输入翻译效果差的问题。通过提出一种自顶向下的句法解码器，生成目标语言的依存树，提升长文本翻译性能。**

- **链接: [https://arxiv.org/pdf/2603.27938](https://arxiv.org/pdf/2603.27938)**

> **作者:** Shuhei Kondo; Katsuhito Sudoh; Yuji Matsumoto
>
> **摘要:** Most of modern neural machine translation (NMT) models are based on an encoder-decoder framework with an attention mechanism. While they perform well on standard datasets, they can have trouble in translation of long inputs that are rare or unseen during training. Incorporating target syntax is one approach to dealing with such length-related problems. We propose a novel syntactic decoder that generates a target-language dependency tree in a top-down, left-to-right order. Experiments show that the proposed top-down string-to-tree decoding generalizes better than conventional sequence-to-sequence decoding in translating long inputs that are not observed in the training data.
>
---
#### [new 066] EpiScreen: Early Epilepsy Detection from Electronic Health Records with Large Language Models
- **分类: cs.CL**

- **简介: 论文提出EpiScreen，利用电子病历中的临床记录，通过大语言模型实现早期癫痫检测，解决误诊和诊断延迟问题。**

- **链接: [https://arxiv.org/pdf/2603.28698](https://arxiv.org/pdf/2603.28698)**

> **作者:** Shuang Zhou; Kai Yu; Zaifu Zhan; Huixue Zhou; Min Zeng; Feng Xie; Zhiyi Sha; Rui Zhang
>
> **备注:** 24 pages, 5 figures, 4 tables
>
> **摘要:** Epilepsy and psychogenic non-epileptic seizures often present with similar seizure-like manifestations but require fundamentally different management strategies. Misdiagnosis is common and can lead to prolonged diagnostic delays, unnecessary treatments, and substantial patient morbidity. Although prolonged video-electroencephalography is the diagnostic gold standard, its high cost and limited accessibility hinder timely diagnosis. Here, we developed a low-cost, effective approach, EpiScreen, for early epilepsy detection by utilizing routinely collected clinical notes from electronic health records. Through fine-tuning large language models on labeled notes, EpiScreen achieved an AUC of up to 0.875 on the MIMIC-IV dataset and 0.980 on a private cohort of the University of Minnesota. In a clinician-AI collaboration setting, EpiScreen-assisted neurologists outperformed unaided experts by up to 10.9%. Overall, this study demonstrates that EpiScreen supports early epilepsy detection, facilitating timely and cost-effective screening that may reduce diagnostic delays and avoid unnecessary interventions, particularly in resource-limited regions.
>
---
#### [new 067] On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究ASR任务中Whisper编码器的剪枝与LoRA微调效果，旨在提升模型效率并保持性能。通过实验验证剪枝2层仅导致2-4% WER上升，结合LoRA可进一步优化。**

- **链接: [https://arxiv.org/pdf/2603.27981](https://arxiv.org/pdf/2603.27981)**

> **作者:** Ganesh Pavan Kartikeya Bharadwaj Kolluri; Michael Kampouridis; Ravi Shekhar
>
> **备注:** Accepted at SPEAKABLE Workshop, LREC 2026
>
> **摘要:** Automatic speech recognition (ASR) has advanced rapidly in recent years, driven by large-scale pretrained models and end-to-end architectures such as SLAM-ASR. A key component of SLAM-ASR systems is the Whisper speech encoder, which provides robust acoustic representations. While model pruning has been explored for the full Whisper encoder-decoder architecture, its impact within the SLAM-ASR setting remains under-investigated. In this work, we analyze the effects of layer pruning in the Whisper encoder when used as the acoustic backbone of SLAM-ASR. We further examine the extent to which LoRA-based fine-tuning can recover performance degradation caused by pruning. Experiments conducted across three Whisper variants (Small, Medium, Large-v2), three languages representing distinct resource levels (Danish, Dutch, English), and over 200 training runs demonstrate that pruning two encoder layers causes only 2-4% WER degradation, and that combining this pruning with LoRA adaptation consistently outperforms the unpruned baseline while reducing total parameters by 7-14%. Moreover, our error analysis reveals that LoRA primarily compensates through the language model's linguistic priors, reducing total word errors by 11-21% for Dutch and English, with substitutions and deletions showing the largest reductions. However, for low-resource Danish, the reduction is smaller (4-7%), and LoRA introduces increased insertion errors, indicating that compensation effectiveness depends on the LLM's pre-existing language proficiency and available training data.
>
---
#### [new 068] Improving Attributed Long-form Question Answering with Intent Awareness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长文本问答任务，旨在提升生成报告的质量。通过增强模型的意图感知能力，优化生成效果，并提高引用和可读性。**

- **链接: [https://arxiv.org/pdf/2603.27435](https://arxiv.org/pdf/2603.27435)**

> **作者:** Xinran Zhao; Aakanksha Naik; Jay DeYoung; Joseph Chee Chang; Jena D. Hwang; Tongshuang Wu; Varsha Kishore
>
> **备注:** 39 pages, 7 figures
>
> **摘要:** Large language models (LLMs) are increasingly being used to generate comprehensive, knowledge-intensive reports. However, while these models are trained on diverse academic papers and reports, they are not exposed to the reasoning processes and intents that guide authors in crafting these documents. We hypothesize that enhancing a model's intent awareness can significantly improve the quality of generated long-form reports. We develop and employ structured, tag-based schemes to better elicit underlying implicit intents to write or cite. We demonstrate that these extracted intents enhance both zero-shot generation capabilities in LLMs and enable the creation of high-quality synthetic data for fine-tuning smaller models. Our experiments reveal improved performance across various challenging scientific report generation tasks, with an average improvement of +2.9 and +12.3 absolute points for large and small models over baselines, respectively. Furthermore, our analysis illuminates how intent awareness enhances model citation usage and substantially improves report readability.
>
---
#### [new 069] Improving Clinical Diagnosis with Counterfactual Multi-Agent Reasoning
- **分类: cs.CL**

- **简介: 该论文属于临床诊断任务，旨在提升AI系统的诊断准确性与可解释性。通过引入反事实多智能体推理框架，增强对不同诊断的验证与讨论，提高诊断的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.27820](https://arxiv.org/pdf/2603.27820)**

> **作者:** Zhiwen You; Xi Chen; Aniket Vashishtha; Simo Du; Gabriel Erion-Barner; Hongyuan Mei; Hao Peng; Yue Guo
>
> **摘要:** Clinical diagnosis is a complex reasoning process in which clinicians gather evidence, form hypotheses, and test them against alternative explanations. In medical training, this reasoning is explicitly developed through counterfactual questioning--e.g., asking how a diagnosis would change if a key symptom were absent or altered--to strengthen differential diagnosis skills. As large language model (LLM)-based systems are increasingly used for diagnostic support, ensuring the interpretability of their recommendations becomes critical. However, most existing LLM-based diagnostic agents reason over fixed clinical evidence without explicitly testing how individual findings support or weaken competing diagnoses. In this work, we propose a counterfactual multi-agent diagnostic framework inspired by clinician training that makes hypothesis testing explicit and evidence-grounded. Our framework introduces counterfactual case editing to modify clinical findings and evaluate how these changes affect competing diagnoses. We further define the Counterfactual Probability Gap, a method that quantifies how strongly individual findings support a diagnosis by measuring confidence shifts under these edits. These counterfactual signals guide multi-round specialist discussions, enabling agents to challenge unsupported hypotheses, refine differential diagnoses, and produce more interpretable reasoning trajectories. Across three diagnostic benchmarks and seven LLMs, our method consistently improves diagnostic accuracy over prompting and prior multi-agent baselines, with the largest gains observed in complex and ambiguous cases. Human evaluation further indicates that our framework produces more clinically useful, reliable, and coherent reasoning. These results suggest that incorporating counterfactual evidence verification is an important step toward building reliable AI systems for clinical decision support.
>
---
#### [new 070] A large corpus of lucid and non-lucid dream reports
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决 lucid dream 研究数据不足的问题。通过收集和标注55,000条梦境报告，构建大规模语料库，为相关研究提供数据支持。**

- **链接: [https://arxiv.org/pdf/2603.26992](https://arxiv.org/pdf/2603.26992)**

> **作者:** Remington Mallett
>
> **摘要:** All varieties of dreaming remain a mystery. Lucid dreams in particular, or those characterized by awareness of the dream, are notoriously difficult to study. Their scarce prevalence and resistance to deliberate induction make it difficult to obtain a sizeable corpus of lucid dream reports. The consequent lack of clarity around lucid dream phenomenology has left the many purported applications of lucidity under-realized. Here, a large corpus of 55k dream reports from 5k contributors is curated, described, and validated for future research. Ten years of publicly available dream reports were scraped from an online forum where users share anonymous dream journals. Importantly, users optionally categorize their dream as lucid, non-lucid, or a nightmare, offering a user-provided labeling system that includes 10k lucid and 25k non-lucid, and 2k nightmare labels. After characterizing the corpus with descriptive statistics and visualizations, construct validation shows that language patterns in lucid-labeled reports are consistent with known characteristics of lucid dreams. While the entire corpus has broad value for dream science, the labeled subset is particularly powerful for new discoveries in lucid dream studies.
>
---
#### [new 071] ProText: A benchmark dataset for measuring (mis)gendering in long-form texts
- **分类: cs.CL**

- **简介: 该论文提出ProText数据集，用于评估长文本中的性别标注与误标问题。属于自然语言处理中的性别偏见分析任务，旨在检测模型在文本生成中的性别刻板印象和偏差。**

- **链接: [https://arxiv.org/pdf/2603.27838](https://arxiv.org/pdf/2603.27838)**

> **作者:** Hadas Kotek; Margit Bowler; Patrick Sonnenberg; Yu'an Yang
>
> **备注:** 13 pages, 10 figures, 6 tables
>
> **摘要:** We introduce ProText, a dataset for measuring gendering and misgendering in stylistically diverse long-form English texts. ProText spans three dimensions: Theme nouns (names, occupations, titles, kinship terms), Theme category (stereotypically male, stereotypically female, gender-neutral/non-gendered), and Pronoun category (masculine, feminine, gender-neutral, none). The dataset is designed to probe (mis)gendering in text transformations such as summarization and rewrites using state-of-the-art Large Language Models, extending beyond traditional pronoun resolution benchmarks and beyond the gender binary. We validated ProText through a mini case study, showing that even with just two prompts and two models, we can draw nuanced insights regarding gender bias, stereotyping, misgendering, and gendering. We reveal systematic gender bias, particularly when inputs contain no explicit gender cues or when models default to heteronormative assumptions.
>
---
#### [new 072] Budget-Xfer: Budget-Constrained Source Language Selection for Cross-Lingual Transfer to African Languages
- **分类: cs.CL**

- **简介: 该论文属于跨语言迁移学习任务，旨在解决在有限标注预算下如何选择源语言的问题。通过提出Budget-Xfer框架，优化多源语言的选取与数据分配，提升低资源非洲语言的NLP性能。**

- **链接: [https://arxiv.org/pdf/2603.27651](https://arxiv.org/pdf/2603.27651)**

> **作者:** Tewodros Kederalah Idris; Roald Eiselen; Prasenjit Mitra
>
> **备注:** 5 pages, 5 tables. Submitted to SIGIR 2026 Short Paper track
>
> **摘要:** Cross-lingual transfer learning enables NLP for low-resource languages by leveraging labeled data from higher-resource sources, yet existing comparisons of source language selection strategies do not control for total training data, confounding language selection effects with data quantity effects. We introduce Budget-Xfer, a framework that formulates multi-source cross-lingual transfer as a budget-constrained resource allocation problem. Given a fixed annotation budget B, our framework jointly optimizes which source languages to include and how much data to allocate from each. We evaluate four allocation strategies across named entity recognition and sentiment analysis for three African target languages (Hausa, Yoruba, Swahili) using two multilingual models, conducting 288 experiments. Our results show that (1) multi-source transfer significantly outperforms single-source transfer (Cohen's d = 0.80 to 1.98), driven by a structural budget underutilization bottleneck; (2) among multi-source strategies, differences are modest and non-significant; and (3) the value of embedding similarity as a selection proxy is task-dependent, with random selection outperforming similarity-based selection for NER but not sentiment analysis.
>
---
#### [new 073] Umwelt Engineering: Designing the Cognitive Worlds of Linguistic Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Umwelt工程，通过设计语言认知环境改进AI代理性能。解决语言约束对推理影响的问题，通过实验验证不同词汇限制的效果。**

- **链接: [https://arxiv.org/pdf/2603.27626](https://arxiv.org/pdf/2603.27626)**

> **作者:** Rodney Jehu-Appiah
>
> **备注:** 24 pages, 2 figures, 7 tables
>
> **摘要:** I propose Umwelt engineering -- the deliberate design of the linguistic cognitive environment -- as a third layer in the agent design stack, upstream of both prompt and context engineering. Two experiments test the thesis that altering the medium of reasoning alters cognition itself. In Experiment 1, three language models reason under two vocabulary constraints -- No-Have (eliminating possessive "to have") and E-Prime (eliminating "to be") -- across seven tasks (N=4,470 trials). No-Have improves ethical reasoning by 19.1 pp (p < 0.001), classification by 6.5 pp (p < 0.001), and epistemic calibration by 7.4 pp, while achieving 92.8% constraint compliance. E-Prime shows dramatic but model-dependent effects: cross-model correlations reach r = -0.75. In Experiment 2, 16 linguistically constrained agents tackle 17 debugging problems. No constrained agent outperforms the control individually, yet a 3-agent ensemble achieves 100% ground-truth coverage versus 88.2% for the control. A permutation test confirms only 8% of random 3-agent subsets achieve full coverage, and every successful subset contains the counterfactual agent. Two mechanisms emerge: cognitive restructuring and cognitive diversification. The primary limitation is the absence of an active control matching constraint prompt elaborateness.
>
---
#### [new 074] Coconstructions in spoken data: UD annotation guidelines and first results
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的句法标注任务，旨在解决口语数据中跨话轮的依赖关系标注问题，提出两种标注框架及区分重构与修正的方法。**

- **链接: [https://arxiv.org/pdf/2603.28261](https://arxiv.org/pdf/2603.28261)**

> **作者:** Ludovica Pannitto; Sylvain Kahane; Kaja Dobrovoljc; Elena Battaglia; Bruno Guillaume; Caterina Mauri; Eleonora Zucchini
>
> **摘要:** The paper proposes annotation guidelines for syntactic dependencies that span across speaker turns - including collaborative coconstructions proper, wh-question answers, and backchannels - in spoken language treebanks within the Universal Dependencies framework. Two representations are proposed: a speaker-based representation following the segmentation into speech turns, and a dependency-based representation with dependencies across speech turns. New propositions are also put forward to distinguish between reformulations and repairs, and to promote elements in unfinished phrases.
>
---
#### [new 075] From Reviews to Requirements: Can LLMs Generate Human-Like User Stories?
- **分类: cs.CL**

- **简介: 该论文属于需求工程任务，旨在将应用评论转化为用户故事。研究评估LLMs生成高质量用户故事的能力，解决手动分析评论效率低的问题。**

- **链接: [https://arxiv.org/pdf/2603.28163](https://arxiv.org/pdf/2603.28163)**

> **作者:** Shadman Sakib; Oishy Fatema Akhand; Tasnia Tasneem; Shohel Ahmed
>
> **摘要:** App store reviews provide a constant flow of real user feedback that can help improve software requirements. However, these reviews are often messy, informal, and difficult to analyze manually at scale. Although automated techniques exist, many do not perform well when replicated and often fail to produce clean, backlog-ready user stories for agile projects. In this study, we evaluate how well large language models (LLMs) such as GPT-3.5 Turbo, Gemini 2.0 Flash, and Mistral 7B Instruct can generate usable user stories directly from raw app reviews. Using the Mini-BAR dataset of 1,000+ health app reviews, we tested zero-shot, one-shot, and two-shot prompting methods. We evaluated the generated user stories using both human judgment (via the RUST framework) and a RoBERTa classifier fine-tuned on UStAI to assess their overall quality. Our results show that LLMs can match or even outperform humans in writing fluent, well-formatted user stories, especially when few-shot prompts are used. However, they still struggle to produce independent and unique user stories, which are essential for building a strong agile backlog. Overall, our findings show how LLMs can reliably turn unstructured app reviews into actionable software requirements, providing developers with clear guidance to turn user feedback into meaningful improvements.
>
---
#### [new 076] Tailoring AI-Driven Reading Scaffolds to the Distinct Needs of Neurodiverse Learners
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决神经多样性学习者阅读支持问题。通过实验比较不同阅读支架效果，发现无统一最优方案，需个性化调整。**

- **链接: [https://arxiv.org/pdf/2603.28370](https://arxiv.org/pdf/2603.28370)**

> **作者:** Soufiane Jhilal; Eleonora Pasqua; Caterina Marchesi; Riccardo Corradi; Martina Galletti
>
> **备注:** Accepted at AIED 2026
>
> **摘要:** Neurodiverse learners often require reading supports, yet increasing scaffold richness can sometimes overload attention and working memory rather than improve comprehension. Grounded in the Construction-Integration model and a contingent scaffolding perspective, we examine how structural versus semantic scaffolds shape comprehension and reading experience in a supervised inclusive context. Using an adapted reading interface, we compared four modalities: unmodified text, sentence-segmented text, segmented text with pictograms, and segmented text with pictograms plus keyword labels. In a within-subject pilot with 14 primary-school learners with special educational needs and disabilities, we measured reading comprehension using standardized questions and collected brief child- and therapist-reported experience measures alongside open-ended feedback. Results highlight heterogeneous responses as some learners showed patterns consistent with benefits from segmentation and pictograms, while others showed patterns consistent with increased coordination costs when visual scaffolds were introduced. Experience ratings showed limited differences between modalities, with some apparent effects linked to clinical complexity, particularly for perceived ease of understanding. Open-ended feedback of the learners frequently requested simpler wording and additional visual supports. These findings suggest that no single scaffold is universally optimal, reinforcing the need for calibrated, adjustable scaffolding and provide design implications for human-AI co-regulation in supervised inclusive reading contexts.
>
---
#### [new 077] Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言视觉推理任务，旨在检验多语言视觉语言模型在印度语言中的表现。研究翻译了多个基准数据集，并评估了8个模型在不同语言中的准确率下降问题。**

- **链接: [https://arxiv.org/pdf/2603.26742](https://arxiv.org/pdf/2603.26742)**

> **作者:** Swastik R
>
> **备注:** 16 pages, 10 figures, 6 tables. Code and data: this https URL Dataset: this https URL
>
> **摘要:** Vision-language models score well on mathematical, scientific, and spatial reasoning benchmarks, yet these evaluations are overwhelmingly English. I present the first cross-lingual visual reasoning audit for Indian languages. 980 questions from MathVista, ScienceQA, and MMMU are translated into Hindi, Tamil, Telugu, Bengali, Kannada, and Marathi using IndicTrans2, with Gemini 2.0 Flash cross-verification on 50 samples per language (inter-translator agreement 0.79-0.84). Eight VLMs, from 7B open-source models to GPT-4o, are evaluated across all seven languages, yielding 68,600 inference records that include text-only and chain-of-thought ablations. I find accuracy drops of 9.8-25 percentage points when switching from English to an Indian language, with Dravidian languages suffering up to 13.2 pp more than Indo-Aryan. Chain-of-thought prompting degrades Bengali (-14.4 pp) and Kannada (-11.4 pp) rather than helping, exposing English-centric reasoning chains. Aya-Vision-8B, built for 23 languages, still drops 28.5 pp on Dravidian scripts; multilingual pretraining alone does not transfer visual reasoning. I release the translated benchmark and all model outputs.
>
---
#### [new 078] AlpsBench: An LLM Personalization Benchmark for Real-Dialogue Memorization and Preference Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型个性化任务，旨在解决缺乏真实对话评估基准的问题。提出AlpsBench，包含真实对话和结构化记忆，用于评估个性化信息管理。**

- **链接: [https://arxiv.org/pdf/2603.26680](https://arxiv.org/pdf/2603.26680)**

> **作者:** Jianfei Xiao; Xiang Yu; Chengbing Wang; Wuqiang Zheng; Xinyu Lin; Kaining Liu; Hongxun Ding; Yang Zhang; Wenjie Wang; Fuli Feng; Xiangnan He
>
> **摘要:** As Large Language Models (LLMs) evolve into lifelong AI assistants, LLM personalization has become a critical frontier. However, progress is currently bottlenecked by the absence of a gold-standard evaluation benchmark. Existing benchmarks either overlook personalized information management that is critical for personalization or rely heavily on synthetic dialogues, which exhibit an inherent distribution gap from real-world dialogue. To bridge this gap, we introduce AlpsBench, An LLM PerSonalization benchmark derived from real-world human-LLM dialogues. AlpsBench comprises 2,500 long-term interaction sequences curated from WildChat, paired with human-verified structured memories that encapsulate both explicit and implicit personalization signals. We define four pivotal tasks - personalized information extraction, updating, retrieval, and utilization - and establish protocols to evaluate the entire lifecycle of memory management. Our benchmarking of frontier LLMs and memory-centric systems reveals that: (i) models struggle to reliably extract latent user traits; (ii) memory updating faces a performance ceiling even in the strongest models; (iii) retrieval accuracy declines sharply in the presence of large distractor pools; and (iv) while explicit memory mechanisms improve recall, they do not inherently guarantee more preference-aligned or emotionally resonant responses. AlpsBench aims to provide a comprehensive framework.
>
---
#### [new 079] AgentSwing: Adaptive Parallel Context Management Routing for Long-Horizon Web Agents
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于长周期网络代理任务，解决上下文管理适应性不足的问题。提出AgentSwing框架，通过动态路由提升搜索效率和精度。**

- **链接: [https://arxiv.org/pdf/2603.27490](https://arxiv.org/pdf/2603.27490)**

> **作者:** Zhaopeng Feng; Liangcai Su; Zhen Zhang; Xinyu Wang; Xiaotian Zhang; Xiaobin Wang; Runnan Fang; Qi Zhang; Baixuan Li; Shihao Cai; Rui Ye; Hui Chen; Jiang Yong; Joey Tianyi Zhou; Chenxiong Qian; Pengjun Xie; Bryan Hooi; Zuozhu Liu; Jingren Zhou
>
> **摘要:** As large language models (LLMs) evolve into autonomous agents for long-horizon information-seeking, managing finite context capacity has become a critical bottleneck. Existing context management methods typically commit to a single fixed strategy throughout the entire trajectory. Such static designs may work well in some states, but they cannot adapt as the usefulness and reliability of the accumulated context evolve during long-horizon search. To formalize this challenge, we introduce a probabilistic framework that characterizes long-horizon success through two complementary dimensions: search efficiency and terminal precision. Building on this perspective, we propose AgentSwing, a state-aware adaptive parallel context management routing framework. At each trigger point, AgentSwing expands multiple context-managed branches in parallel and uses lookahead routing to select the most promising continuation. Experiments across diverse benchmarks and agent backbones show that AgentSwing consistently outperforms strong static context management methods, often matching or exceeding their performance with up to $3\times$ fewer interaction turns while also improving the ultimate performance ceiling of long-horizon web agents. Beyond the empirical gains, the proposed probabilistic framework provides a principled lens for analyzing and designing future context management strategies for long-horizon agents.
>
---
#### [new 080] Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究LLM在政治文本标注中的应用，探讨模型选择、提示风格等对结果的影响。旨在解决标注方法有效性问题，通过实验验证不同配置效果，提出验证优先的框架。**

- **链接: [https://arxiv.org/pdf/2603.26898](https://arxiv.org/pdf/2603.26898)**

> **作者:** Lorca McLaren; James Cross; Zuzanna Krakowska; Robin Rauner; Martijn Schoonvelde
>
> **摘要:** Political scientists are rapidly adopting large language models (LLMs) for text annotation, yet the sensitivity of annotation results to implementation choices remains poorly understood. Most evaluations test a single model or configuration; how model choice, model size, learning approach, and prompt style interact, and whether popular "best practices" survive controlled comparison, are largely unexplored. We present a controlled evaluation of these pipeline choices, testing six open-weight models across four political science annotation tasks under identical quantisation, hardware, and prompt-template conditions. Our central finding is methodological: interaction effects dominate main effects, so seemingly reasonable pipeline choices can become consequential researcher degrees of freedom. No single model, prompt style, or learning approach is uniformly superior, and the best-performing model varies across tasks. Two corollaries follow. First, model size is an unreliable guide both to cost and to performance: cross-family efficiency differences are so large that some larger models are less resource-intensive than much smaller alternatives, while within model families mid-range variants often match or exceed larger counterparts. Second, widely recommended prompt engineering techniques yield inconsistent and sometimes negative effects on annotation performance. We use these benchmark results to develop a validation-first framework - with a principled ordering of pipeline decisions, guidance on prompt freezing and held-out evaluation, reporting standards, and open-source tools - to help researchers navigate this decision space transparently.
>
---
#### [new 081] LLM Readiness Harness: Evaluation, Observability, and CI Gates for LLM/RAG Applications
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文提出一种LLM/RAG应用的就绪评估框架，解决部署决策问题。通过集成评估、可观测性和CI门禁，实现系统就绪评分，确保安全发布。**

- **链接: [https://arxiv.org/pdf/2603.27355](https://arxiv.org/pdf/2603.27355)**

> **作者:** Alexandre Cristovão Maiorano
>
> **备注:** 18 pages, 4 figures, 15 tables, arXiv preprint
>
> **摘要:** We present a readiness harness for LLM and RAG applications that turns evaluation into a deployment decision workflow. The system combines automated benchmarks, OpenTelemetry observability, and CI quality gates under a minimal API contract, then aggregates workflow success, policy compliance, groundedness, retrieval hit rate, cost, and p95 latency into scenario-weighted readiness scores with Pareto frontiers. We evaluate the harness on ticket-routing workflows and BEIR grounding tasks (SciFact and FiQA) with full Azure matrix coverage (162/162 valid cells across datasets, scenarios, retrieval depths, seeds, and models). Results show that readiness is not a single metric: on FiQA under sla-first at k=5, gpt-4.1-mini leads in readiness and faithfulness, while gpt-5.2 pays a substantial latency cost; on SciFact, models are closer in quality but still separable operationally. Ticket-routing regression gates consistently reject unsafe prompt variants, demonstrating that the harness can block risky releases instead of merely reporting offline scores. The result is a reproducible, operationally grounded framework for deciding whether an LLM or RAG system is ready to ship.
>
---
#### [new 082] ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出ParaSpeechCLAP，一个双编码器模型，用于语音与文本风格描述的联合嵌入，解决跨模态风格对齐问题。通过专门化和统一模型提升风格分类与检索性能。**

- **链接: [https://arxiv.org/pdf/2603.28737](https://arxiv.org/pdf/2603.28737)**

> **作者:** Anuj Diwan; Eunsol Choi; David Harwath
>
> **备注:** Under review
>
> **摘要:** We introduce ParaSpeechCLAP, a dual-encoder contrastive model that maps speech and text style captions into a common embedding space, supporting a wide range of intrinsic (speaker-level) and situational (utterance-level) descriptors (such as pitch, texture and emotion) far beyond the narrow set handled by existing models. We train specialized ParaSpeechCLAP-Intrinsic and ParaSpeechCLAP-Situational models alongside a unified ParaSpeechCLAP-Combined model, finding that specialization yields stronger performance on individual style dimensions while the unified model excels on compositional evaluation. We further show that ParaSpeechCLAP-Intrinsic benefits from an additional classification loss and class-balanced training. We demonstrate our models' performance on style caption retrieval, speech attribute classification and as an inference-time reward model that improves style-prompted TTS without additional training. ParaSpeechCLAP outperforms baselines on most metrics across all three applications. Our models and code are released at this https URL .
>
---
#### [new 083] SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SOLE-R1，用于机器人强化学习的视频-语言推理模型，解决无监督任务学习问题。通过视频和自然语言目标生成密集奖励信号，实现零样本在线学习。**

- **链接: [https://arxiv.org/pdf/2603.28730](https://arxiv.org/pdf/2603.28730)**

> **作者:** Philip Schroeder; Thomas Weng; Karl Schmeckpeper; Eric Rosen; Stephen Hart; Ondrej Biza
>
> **摘要:** Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. To address this limitation, we introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including GPT-5 and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking.
>
---
#### [new 084] FormalProofBench: Can Models Write Graduate Level Math Proofs That Are Formally Verified?
- **分类: cs.AI; cs.CL; cs.LG; cs.PL**

- **简介: 该论文提出FormalProofBench，评估AI模型能否生成可形式化验证的研究生数学证明。任务属于形式化定理证明，解决AI在高级数学证明能力的评测问题。**

- **链接: [https://arxiv.org/pdf/2603.26996](https://arxiv.org/pdf/2603.26996)**

> **作者:** Nikil Ravi; Kexing Ying; Vasilii Nesterov; Rayan Krishnan; Elif Uskuplu; Bingyu Xia; Janitha Aswedige; Langston Nashold
>
> **备注:** Accepted at ICLR 2026 Workshop: VerifAI-2: The Second Workshop on AI Verification in the Wild. Live leaderboard hosted here: this https URL
>
> **摘要:** We present FormalProofBench, a private benchmark designed to evaluate whether AI models can produce formally verified mathematical proofs at the graduate level. Each task pairs a natural-language problem with a Lean~4 formal statement, and a model must output a Lean proof accepted by the Lean 4 checker. FormalProofBench targets advanced undergraduate and graduate mathematics, with problems drawn from qualifying exams and standard textbooks across topics including analysis, algebra, probability, and logic. We evaluate a range of frontier models with an agentic harness, and find that the best-performing foundation model achieves 33.5% accuracy, with performance dropping rapidly after that. In addition to the accuracy numbers, we also provide empirical analysis of tool-use, failure modes, cost and latency, thereby providing a thorough evaluation of the formal-theorem proving abilities of frontier models.
>
---
#### [new 085] Multilingual Stutter Event Detection for English, German, and Mandarin Speech
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决多语言口吃检测问题。通过多语种数据训练模型，捕捉口吃的共性特征，提升检测的泛化能力与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.26939](https://arxiv.org/pdf/2603.26939)**

> **作者:** Felix Haas; Sebastian P. Bayerl
>
> **摘要:** This paper presents a multi-label stuttering detection system trained on multi-corpus, multilingual data in English, German, and this http URL leveraging annotated stuttering data from three languages and four corpora, the model captures language-independent characteristics of stuttering, enabling robust detection across linguistic contexts. Experimental results demonstrate that multilingual training achieves performance comparable to and, in some cases, even exceeds that of previous systems. These findings suggest that stuttering exhibits cross-linguistic consistency, which supports the development of language-agnostic detection systems. Our work demonstrates the feasibility and advantages of using multilingual data to improve generalizability and reliability in automated stuttering detection.
>
---
#### [new 086] SleepVLM: Explainable and Rule-Grounded Sleep Staging via a Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于睡眠分期任务，旨在解决自动化睡眠分期缺乏可审计解释的问题。提出SleepVLM模型，结合规则与视觉语言模型，生成符合AASM标准的可读理由，提升临床可信度。**

- **链接: [https://arxiv.org/pdf/2603.26738](https://arxiv.org/pdf/2603.26738)**

> **作者:** Guifeng Deng; Pan Wang; Jiquan Wang; Shuying Rao; Junyi Xie; Wanjun Guo; Tao Li; Haiteng Jiang
>
> **备注:** Under review
>
> **摘要:** While automated sleep staging has achieved expert-level accuracy, its clinical adoption is hindered by a lack of auditable reasoning. We introduce SleepVLM, a rule-grounded vision-language model (VLM) designed to stage sleep from multi-channel polysomnography (PSG) waveform images while generating clinician-readable rationales based on American Academy of Sleep Medicine (AASM) scoring criteria. Utilizing waveform-perceptual pre-training and rule-grounded supervised fine-tuning, SleepVLM achieved Cohen's kappa scores of 0.767 on an held out test set (MASS-SS1) and 0.743 on an external cohort (ZUAMHCS), matching state-of-the-art performance. Expert evaluations further validated the quality of the model's reasoning, with mean scores exceeding 4.0/5.0 for factual accuracy, evidence comprehensiveness, and logical coherence. By coupling competitive performance with transparent, rule-based explanations, SleepVLM may improve the trustworthiness and auditability of automated sleep staging in clinical workflows. To facilitate further research in interpretable sleep medicine, we release MASS-EX, a novel expert-annotated dataset.
>
---
#### [new 087] The Ultimate Tutorial for AI-driven Scale Development in Generative Psychometrics: Releasing AIGENIE from its Bottle
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于心理测量学任务，旨在解决传统心理量表开发耗时费力的问题。通过AI生成与网络心理测量方法结合，实现量表项目的自动化生成与验证。**

- **链接: [https://arxiv.org/pdf/2603.28643](https://arxiv.org/pdf/2603.28643)**

> **作者:** Lara Russell-Lasalandra; Hudson Golino; Luis Eduardo Garrido; Alexander P. Christensen
>
> **备注:** 38 pages, 8 Figures, 3 tables
>
> **摘要:** Psychological scale development has traditionally required extensive expert involvement, iterative revision, and large-scale pilot testing before psychometric evaluation can begin. The `AIGENIE` R package implements the AI-GENIE framework (Automatic Item Generation with Network-Integrated Evaluation), which integrates large language model (LLM) text generation with network psychometric methods to automate the early stages of this process. The package generates candidate item pools using LLMs, transforms them into high-dimensional embeddings, and applies a multi-step reduction pipeline -- Exploratory Graph Analysis (EGA), Unique Variable Analysis (UVA), and bootstrap EGA -- to produce structurally validated item pools entirely *in silico*. This tutorial introduces the package across six parts: installation and setup, understanding Application Programming Interfaces (APIs), text generation, item generation, the `AIGENIE` function, and the `GENIE` function. Two running examples illustrate the package's use: the Big Five personality model (a well-established construct) and AI Anxiety (an emerging construct). The package supports multiple LLM providers (OpenAI, Anthropic, Groq, HuggingFace, and local models), offers a fully offline mode with no external API calls, and provides the `GENIE()` function for researchers who wish to apply the psychometric reduction pipeline to existing item pools regardless of their origin. The `AIGENIE` package is freely available on R-universe at this https URL.
>
---
#### [new 088] Aesthetic Assessment of Chinese Handwritings Based on Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于手写汉字美学评估任务，旨在解决传统评分反馈不足的问题。通过视觉语言模型生成多级反馈，提升学习者书写改进效果。**

- **链接: [https://arxiv.org/pdf/2603.26768](https://arxiv.org/pdf/2603.26768)**

> **作者:** Chen Zheng; Yuxuan Lai; Haoyang Lu; Wentao Ma; Jitao Yang; Jian Wang
>
> **备注:** Accepted by CCL2025
>
> **摘要:** The handwriting of Chinese characters is a fundamental aspect of learning the Chinese language. Previous automated assessment methods often framed scoring as a regression problem. However, this score-only feedback lacks actionable guidance, which limits its effectiveness in helping learners improve their handwriting skills. In this paper, we leverage vision-language models (VLMs) to analyze the quality of handwritten Chinese characters and generate multi-level feedback. Specifically, we investigate two feedback generation tasks: simple grade feedback (Task 1) and enriched, descriptive feedback (Task 2). We explore both low-rank adaptation (LoRA)-based fine-tuning strategies and in-context learning methods to integrate aesthetic assessment knowledge into VLMs. Experimental results show that our approach achieves state-of-the-art performances across multiple evaluation tracks in the CCL 2025 workshop on evaluation of handwritten Chinese character quality.
>
---
#### [new 089] PHONOS: PHOnetic Neutralization for Online Streaming Applications
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于语音匿名化任务，旨在解决非母语口音影响匿名性的问题。通过生成本土化语音样本，训练实时模块以中和口音，提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2603.27001](https://arxiv.org/pdf/2603.27001)**

> **作者:** Waris Quamer; Mu-Ruei Tseng; Ghady Nasrallah; Ricardo Gutierrez-Osuna
>
> **备注:** The paper is submitted to Interspeech 2026 and currently under review
>
> **摘要:** Speaker anonymization (SA) systems modify timbre while leaving regional or non-native accents intact, which is problematic because accents can narrow the anonymity set. To address this issue, we present PHONOS, a streaming module for real-time SA that neutralizes non-native accent to sound native-like. Our approach pre-generates golden speaker utterances that preserve source timbre and rhythm but replace foreign segmentals with native ones using silence-aware DTW alignment and zero-shot voice conversion. These utterances supervise a causal accent translator that maps non-native content tokens to native equivalents with at most 40ms look-ahead, trained using joint cross-entropy and CTC losses. Our evaluations show an 81% reduction in non-native accent confidence, with listening-test ratings consistent with this shift, and reduced speaker linkability as accent-neutralized utterances move away from the original speaker in embedding space while having latency under 241 ms on single GPU.
>
---
#### [new 090] The Geometry of Harmful Intent: Training-Free Anomaly Detection via Angular Deviation in LLM Residual Streams
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于异常检测任务，旨在无需训练检测有害提示。通过分析大语言模型残差流的几何特征，计算角度偏差作为异常评分。**

- **链接: [https://arxiv.org/pdf/2603.27412](https://arxiv.org/pdf/2603.27412)**

> **作者:** Isaac Llorente-Saguer
>
> **备注:** 20 pages, 10 figures, 3 tables. Training-free harmful-prompt detector via angular deviation in LLM residual streams. Evaluated on six Qwen variants (base / instruct / abliterated). Achieves AUROC over 0.937 (harmful-vs-normative) and 1.000 (harmful-vs-benign-aggressive) with no harmful training data
>
> **摘要:** We present LatentBiopsy, a training-free method for detecting harmful prompts by analysing the geometry of residual-stream activations in large language models. Given 200 safe normative prompts, LatentBiopsy computes the leading principal component of their activations at a target layer and characterises new prompts by their radial deviation angle $\theta$ from this reference direction. The anomaly score is the negative log-likelihood of $\theta$ under a Gaussian fit to the normative distribution, flagging deviations symmetrically regardless of orientation. No harmful examples are required for training. We evaluate two complete model triplets from the Qwen3.5-0.8B and Qwen2.5-0.5B families: base, instruction-tuned, and \emph{abliterated} (refusal direction surgically removed via orthogonalisation). Across all six variants, LatentBiopsy achieves AUROC $\geq$0.937 for harmful-vs-normative detection and AUROC = 1.000 for discriminating harmful from benign-aggressive prompts (XSTest), with sub-millisecond per-query overhead. Three empirical findings emerge. First, geometry survives refusal ablation: both abliterated variants achieve AUROC at most 0.015 below their instruction-tuned counterparts, establishing a geometric dissociation between harmful-intent representation and the downstream generative refusal mechanism. Second, harmful prompts exhibit a near-degenerate angular distribution ($\sigma_\theta \approx 0.03$ rad), an order of magnitude tighter than the normative distribution ($\sigma_\theta \approx 0.27$ rad), preserved across all alignment stages including abliteration. Third, the two families exhibit opposite ring orientations at the same depth: harmful prompts occupy the outer ring in Qwen3.5-0.8B but the inner ring in Qwen2.5-0.5B, directly motivating the direction-agnostic scoring rule.
>
---
#### [new 091] Emergent Social Intelligence Risks in Generative Multi-Agent Systems
- **分类: cs.MA; cs.CL; cs.CY**

- **简介: 该论文属于人工智能安全研究，探讨生成式多智能体系统中涌现的社会智能风险。工作包括分析资源竞争、协作等场景下的集体行为，揭示其潜在风险。**

- **链接: [https://arxiv.org/pdf/2603.27771](https://arxiv.org/pdf/2603.27771)**

> **作者:** Yue Huang; Yu Jiang; Wenjie Wang; Haomin Zhuang; Xiaonan Luo; Yuchen Ma; Zhangchen Xu; Zichen Chen; Nuno Moniz; Zinan Lin; Pin-Yu Chen; Nitesh V Chawla; Nouha Dziri; Huan Sun; Xiangliang Zhang
>
> **摘要:** Multi-agent systems composed of large generative models are rapidly moving from laboratory prototypes to real-world deployments, where they jointly plan, negotiate, and allocate shared resources to solve complex tasks. While such systems promise unprecedented scalability and autonomy, their collective interaction also gives rise to failure modes that cannot be reduced to individual agents. Understanding these emergent risks is therefore critical. Here, we present a pioneer study of such emergent multi-agent risk in workflows that involve competition over shared resources (e.g., computing resources or market share), sequential handoff collaboration (where downstream agents see only predecessor outputs), collective decision aggregation, and others. Across these settings, we observe that such group behaviors arise frequently across repeated trials and a wide range of interaction conditions, rather than as rare or pathological cases. In particular, phenomena such as collusion-like coordination and conformity emerge with non-trivial frequency under realistic resource constraints, communication protocols, and role assignments, mirroring well-known pathologies in human societies despite no explicit instruction. Moreover, these risks cannot be prevented by existing agent-level safeguards alone. These findings expose the dark side of intelligent multi-agent systems: a social intelligence risk where agent collectives, despite no instruction to do so, spontaneously reproduce familiar failure patterns from human societies.
>
---
#### [new 092] Entropic Claim Resolution: Uncertainty-Driven Evidence Selection for RAG
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于知识密集型问答任务，旨在解决RAG系统中因证据冲突或查询模糊导致的不确定性问题。提出ECR算法，通过熵最小化选择最具区分性的证据。**

- **链接: [https://arxiv.org/pdf/2603.28444](https://arxiv.org/pdf/2603.28444)**

> **作者:** Davide Di Gioia
>
> **备注:** Preprint
>
> **摘要:** Current Retrieval-Augmented Generation (RAG) systems predominantly rely on relevance-based dense retrieval, sequentially fetching documents to maximize semantic similarity with the query. However, in knowledge-intensive and real-world scenarios characterized by conflicting evidence or fundamental query ambiguity, relevance alone is insufficient for resolving epistemic uncertainty. We introduce Entropic Claim Resolution (ECR), a novel inference-time algorithm that reframes RAG reasoning as entropy minimization over competing semantic answer hypotheses. Unlike action-driven agentic frameworks (e.g., ReAct) or fixed-pipeline RAG architectures, ECR sequentially selects atomic evidence claims by maximizing Expected Entropy Reduction (EER), a decision-theoretic criterion for the value of information. The process dynamically terminates when the system reaches a mathematically defined state of epistemic sufficiency (H <= epsilon, subject to epistemic coherence). We integrate ECR into a production-grade multi-strategy retrieval pipeline (CSGR++) and analyze its theoretical properties. Our framework provides a rigorous foundation for uncertainty-aware evidence selection, shifting the paradigm from retrieving what is most relevant to retrieving what is most discriminative.
>
---
#### [new 093] Self-evolving AI agents for protein discovery and directed evolution
- **分类: cs.AI; cs.CL; q-bio.QM**

- **简介: 该论文属于蛋白质发现任务，旨在解决手动流程效率低和通用代理不足的问题。提出VenusFactory2框架，通过自进化多智能体实现动态工作流合成，自主完成蛋白质发现与优化。**

- **链接: [https://arxiv.org/pdf/2603.27303](https://arxiv.org/pdf/2603.27303)**

> **作者:** Yang Tan; Lingrong Zhang; Mingchen Li; Yuanxi Yu; Bozitao Zhong; Bingxin Zhou; Nanqing Dong; Liang Hong
>
> **备注:** 100 pages, 6 figures
>
> **摘要:** Protein scientific discovery is bottlenecked by the manual orchestration of information and algorithms, while general agents are insufficient in complex domain projects. VenusFactory2 provides an autonomous framework that shifts from static tool usage to dynamic workflow synthesis via a self-evolving multi-agent infrastructure to address protein-related demands. It outperforms a set of well-known agents on the VenusAgentEval benchmark, and autonomously organizes the discovery and optimization of proteins from a single natural language prompt.
>
---
#### [new 094] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ChartNet，一个大规模多模态数据集，用于提升图表理解能力。解决现有模型在图表视觉、数据与语言联合推理上的不足，通过生成多样化图表样本及精细对齐的多模态数据，推动图表解释与推理研究。**

- **链接: [https://arxiv.org/pdf/2603.27064](https://arxiv.org/pdf/2603.27064)**

> **作者:** Jovana Kondic; Pengyuan Li; Dhiraj Joshi; Isaac Sanchez; Ben Wiesel; Shafiq Abedin; Amit Alfassy; Eli Schwartz; Daniel Caraballo; Yagmur Gizem Cinar; Florian Scheidegger; Steven I. Ross; Daniel Karl I. Weidele; Hang Hua; Ekaterina Arutyunova; Roei Herzig; Zexue He; Zihan Wang; Xinyue Yu; Yunfei Zhao; Sicong Jiang; Minghao Liu; Qunshu Lin; Peter Staar; Luis Lastras; Aude Oliva; Rogerio Feris
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Understanding charts requires models to jointly reason over geometric visual patterns, structured numerical data, and natural language -- a capability where current vision-language models (VLMs) remain limited. We introduce ChartNet, a high-quality, million-scale multimodal dataset designed to advance chart interpretation and reasoning. ChartNet leverages a novel code-guided synthesis pipeline to generate 1.5 million diverse chart samples spanning 24 chart types and 6 plotting libraries. Each sample consists of five aligned components: plotting code, rendered chart image, data table, natural language summary, and question-answering with reasoning, providing fine-grained cross-modal alignment. To capture the full spectrum of chart comprehension, ChartNet additionally includes specialized subsets encompassing human annotated data, real-world data, safety, and grounding. Moreover, a rigorous quality-filtering pipeline ensures visual fidelity, semantic accuracy, and diversity across chart representations. Fine-tuning on ChartNet consistently improves results across benchmarks, demonstrating its utility as large-scale supervision for multimodal models. As the largest open-source dataset of its kind, ChartNet aims to support the development of foundation models with robust and generalizable capabilities for data visualization understanding. The dataset is publicly available at this https URL
>
---
#### [new 095] In your own words: computationally identifying interpretable themes in free-text survey data
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决自由文本调查数据的分析难题。提出框架In Your Own Words，更精准地识别可解释的主题，提升调查研究的系统性。**

- **链接: [https://arxiv.org/pdf/2603.26930](https://arxiv.org/pdf/2603.26930)**

> **作者:** Jenny S Wang; Aliya Saperstein; Emma Pierson
>
> **摘要:** Free-text survey responses can provide nuance often missed by structured questions, but remain difficult to statistically analyze. To address this, we introduce In Your Own Words, a computational framework for exploratory analyses of free-text survey data that identifies structured, interpretable themes in free-text responses more precisely than previous computational approaches, facilitating systematic analysis. To illustrate the benefits of this approach, we apply it to a new dataset of free-text descriptions of race, gender, and sexual orientation from 1,004 U.S. participants. The themes our approach learns have three practical applications in survey research. First, the themes can suggest structured questions to add to future surveys by surfacing salient constructs -- such as belonging and identity fluidity -- that existing surveys do not capture. Second, the themes reveal heterogeneity within standardized categories, explaining additional variation in health, well-being, and identity importance. Third, the themes illuminate systematic discordance between self-identified and perceived identities, highlighting mechanisms of misrecognition that existing measures do not reflect. More broadly, our framework can be deployed in a wide range of survey settings to identify interpretable themes from free text, complementing existing qualitative methods.
>
---
#### [new 096] CRISP: Characterizing Relative Impact of Scholarly Publications
- **分类: cs.DL; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出CRISP，用于评估学术论文的引用影响力。任务是改进引用影响分析，解决传统方法孤立分析引用的问题。工作是利用大语言模型联合排序引用文献，并通过多数投票减少偏差，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.26791](https://arxiv.org/pdf/2603.26791)**

> **作者:** Hannah Collison; Benjamin Van Durme; Daniel Khashabi
>
> **摘要:** Assessing a cited paper's impact is typically done by analyzing its citation context in isolation within the citing paper. While this focuses on the most directly relevant text, it prevents relative comparisons across all the works a paper cites. We propose CRISP, which instead jointly ranks all cited papers within a citing paper using large language models (LLMs). To mitigate LLMs' positional bias, we rank each list three times in a randomized order and aggregate the impact labels through majority voting. This joint approach leverages the full citation context, rather than evaluating citations independently, to more reliably distinguish impactful references. CRISP outperforms a prior state-of-the-art impact classifier by +9.5% accuracy and +8.3% F1 on a dataset of human-annotated citations. CRISP further gains efficiency through fewer LLM calls and performs competitively with an open-source model, enabling scalable, cost-effective citation impact analysis. We release our rankings, impact labels, and codebase to support future research.
>
---
#### [new 097] Moving Beyond Review: Applying Language Models to Planning and Translation in Reflection
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于教育技术任务，旨在提升反思写作质量。解决学生反思深度不足的问题，通过LLMs支持规划与翻译阶段，提出Pensée工具并验证其效果。**

- **链接: [https://arxiv.org/pdf/2603.28596](https://arxiv.org/pdf/2603.28596)**

> **作者:** Seyed Parsa Neshaei; Richard Lee Davis; Tanja Käser
>
> **备注:** Accepted at AIED 2026
>
> **摘要:** Reflective writing is known to support the development of students' metacognitive skills, yet learners often struggle to engage in deep reflection, limiting learning gains. Although large language models (LLMs) have been shown to improve writing skills, their use as conversational agents for reflective writing has produced mixed results and has largely focused on providing feedback on reflective texts, rather than support during planning and organizing. In this paper, inspired by the Cognitive Process Theory of writing (CPT), we propose the first application of LLMs to the planning and translation steps of reflective writing. We introduce Pensée, a tool to explore the effects of explicit AI support during these stages by scaffolding structured reflection planning using a conversational agent, and supporting translation by automatically extracting key concepts. We evaluate Pensée in a controlled between-subjects experiment (N=93), manipulating AI support across writing phases. Results show significantly greater reflection depth and structural quality when learners receive support during planning and translation stages of CPT, though these effects reduce in a delayed post-test. Analyses of learner behavior and perceptions further illustrate how CPT-aligned conversational support shapes reflection processes and learner experience, contributing empirical evidence for theory-driven uses of LLMs in AI-supported reflective writing.
>
---
#### [new 098] LITTA: Late-Interaction and Test-Time Alignment for Visually-Grounded Multimodal Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态文档检索任务，解决视觉丰富文档中证据检索困难的问题。提出LITTA框架，通过查询扩展和测试时对齐提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.26683](https://arxiv.org/pdf/2603.26683)**

> **作者:** Seonok Kim
>
> **摘要:** Retrieving relevant evidence from visually rich documents such as textbooks, technical reports, and manuals is challenging due to long context, complex layouts, and weak lexical overlap between user questions and supporting pages. We propose LITTA, a query-expansion-centric retrieval framework for evidence page retrieval that improves multimodal document retrieval without retriever retraining. Given a user query, LITTA generates complementary query variants using a large language model and retrieves candidate pages for each variant using a frozen vision retriever with late-interaction scoring. Candidates from expanded queries are then aggregated through reciprocal rank fusion to improve evidence coverage and reduce sensitivity to any single phrasing. This simple test-time strategy significantly improves retrieval robustness while remaining compatible with existing multimodal embedding indices. We evaluate LITTA on visually grounded document retrieval tasks across three domains: computer science, pharmaceuticals, and industrial manuals. Multi-query retrieval consistently improves top-k accuracy, recall, and MRR compared to single-query retrieval, with particularly large gains in domains with high visual and semantic variability. Moreover, the accuracy-efficiency trade-off is directly controllable by the number of query variants, making LITTA practical for deployment under latency constraints. These results demonstrate that query expansion provides a simple yet effective mechanism for improving visually grounded multimodal retrieval.
>
---
#### [new 099] Inference-Time Structural Reasoning for Compositional Vision-Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言理解任务，解决 compositional reasoning 问题。通过结构化推理方法提升模型对关系结构的敏感性，提出文本图解析与图不对称评分机制，实验验证了方法有效性。**

- **链接: [https://arxiv.org/pdf/2603.27349](https://arxiv.org/pdf/2603.27349)**

> **作者:** Amartya Bhattacharya
>
> **摘要:** Vision-language models (VLMs) excel at image-text retrieval yet persistently fail at compositional reasoning, distinguishing captions that share the same words but differ in relational structure. We present, a unified evaluation and augmentation framework benchmarking four architecturally diverse VLMs,CLIP, BLIP, LLaVA, and Qwen3-VL-8B-Thinking,on the Winoground benchmark under plain and scene-graph-augmented regimes. We introduce a dependency-based TextSceneGraphParser (spaCy) extracting subject-relation-object triples, and a Graph Asymmetry Scorer using optimal bipartite matching to inject structural relational priors. Caption ablation experiments (subject-object masking and swapping) reveal that Qwen3-VL-8B-Thinking achieves a group score of 62.75, far above all encoder-based models, while a proposed multi-turn SG filtering strategy further lifts it to 66.0, surpassing prior open-source state-of-the-art. We analyze the capability augmentation tradeoff and find that SG augmentation benefits already capable models while providing negligible or negative gains for weaker baselines. Code: this https URL
>
---
#### [new 100] GroupRAG: Cognitively Inspired Group-Aware Retrieval and Reasoning via Knowledge-Driven Problem Structuring
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出GroupRAG，解决语言模型知识不足与推理受限问题，通过结构化问题空间建模提升检索与推理效果。**

- **链接: [https://arxiv.org/pdf/2603.26807](https://arxiv.org/pdf/2603.26807)**

> **作者:** Xinyi Duan; Yuanrong Tang; Jiangtao Gong
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** The performance of language models is commonly limited by insufficient knowledge and constrained reasoning. Prior approaches such as Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) address these issues by incorporating external knowledge or enforcing linear reasoning chains, but often degrade in real-world settings. Inspired by cognitive science, which characterizes human problem solving as search over structured problem spaces rather than single inference chains, we argue that inadequate awareness of problem structure is a key overlooked limitation. We propose GroupRAG, a cognitively inspired, group-aware retrieval and reasoning framework based on knowledge-driven keypoint grouping. GroupRAG identifies latent structural groups within a problem and performs retrieval and reasoning from multiple conceptual starting points, enabling fine-grained interaction between the two processes. Experiments on MedQA show that GroupRAG outperforms representative RAG- and CoT-based baselines. These results suggest that explicitly modeling problem structure, as inspired by human cognition, is a promising direction for robust retrieval-augmented reasoning.
>
---
#### [new 101] EffiSkill: Agent Skill Based Automated Code Efficiency Optimization
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出EffiSkill，解决代码效率优化问题。通过构建可复用的技能库，实现无需运行时反馈的代码优化，提升优化成功率。**

- **链接: [https://arxiv.org/pdf/2603.27850](https://arxiv.org/pdf/2603.27850)**

> **作者:** Zimu Wang; Yuling Shi; Mengfan Li; Zijun Liu; Jie M. Zhang; Chengcheng Wan; Xiaodong Gu
>
> **摘要:** Code efficiency is a fundamental aspect of software quality, yet how to harness large language models (LLMs) to optimize programs remains challenging. Prior approaches have sought for one-shot rewriting, retrieved exemplars, or prompt-based search, but they do not explicitly distill reusable optimization knowledge, which limits generalization beyond individual instances. In this paper, we present EffiSkill, a framework for code-efficiency optimization that builds a portable optimization toolbox for LLM-based agents. The key idea is to model recurring slow-to-fast transformations as reusable agent skills that capture both concrete transformation mechanisms and higher-level optimization strategies. EffiSkill adopts a two-stage design: Stage I mines Operator and Meta Skills from large-scale slow/fast program pairs to build a skill library; Stage II applies this library to unseen programs through execution-free diagnosis, skill retrieval, plan composition, and candidate generation, without runtime feedback. Results on EffiBench-X show that EffiSkill achieves higher optimization success rates, improving over the strongest baseline by 3.69 to 12.52 percentage points across model and language settings. These findings suggest that mechanism-level skill reuse provides a useful foundation for execution-free code optimization, and that the resulting skill library can serve as a reusable resource for broader agent workflows.
>
---
#### [new 102] Learning to Select Visual In-Context Demonstrations
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究视觉上下文学习中的演示选择问题，旨在提升多模态大语言模型的性能。通过强化学习方法，构建更优演示集，解决事实性回归任务中的冗余与覆盖不足问题。**

- **链接: [https://arxiv.org/pdf/2603.26775](https://arxiv.org/pdf/2603.26775)**

> **作者:** Eugene Lee; Yu-Chi Lin; Jiajie Diao
>
> **备注:** 21 pages, 12 figure, accepted to Computer Vision and Pattern Recognition Conference (CVPR) 2026 Findings Track
>
> **摘要:** Multimodal Large Language Models (MLLMs) adapt to visual tasks via in-context learning (ICL), which relies heavily on demonstration quality. The dominant demonstration selection strategy is unsupervised k-Nearest Neighbor (kNN) search. While simple, this similarity-first approach is sub-optimal for complex factual regression tasks; it selects redundant examples that fail to capture the task's full output range. We reframe selection as a sequential decision-making problem and introduce Learning to Select Demonstrations (LSD), training a Reinforcement Learning agent to construct optimal demonstration sets. Using a Dueling DQN with a query-centric Transformer Decoder, our agent learns a policy that maximizes MLLM downstream performance. Evaluating across five visual regression benchmarks, we uncover a crucial dichotomy: while kNN remains optimal for subjective preference tasks, LSD significantly outperforms baselines on objective, factual regression tasks. By balancing visual relevance with diversity, LSD better defines regression boundaries, illuminating when learned selection is strictly necessary for visual ICL.
>
---
#### [new 103] CDH-Bench: A Commonsense-Driven Hallucination Benchmark for Evaluating Visual Fidelity in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决视觉证据与常识冲突时模型是否产生幻觉的问题。提出CDH-Bench基准，用于评估模型在该场景下的视觉保真度。**

- **链接: [https://arxiv.org/pdf/2603.27982](https://arxiv.org/pdf/2603.27982)**

> **作者:** Kesheng Chen; Yamin Hu; Qi Zhou; Zhenqian Zhu; Wenjian Luo
>
> **摘要:** Vision-language models (VLMs) achieve strong performance on many benchmarks, yet a basic reliability question remains underexplored: when visual evidence conflicts with commonsense, do models follow what is shown or what commonsense suggests? A characteristic failure in this setting is that the model overrides visual evidence and outputs the commonsense alternative. We term this phenomenon \textbf{commonsense-driven hallucination} (CDH). To evaluate it, we introduce \textbf{CDH-Bench}, a benchmark designed to create explicit \textbf{visual evidence--commonsense conflicts}. CDH-Bench covers three dimensions: \textit{counting anomalies}, \textit{relational anomalies}, and \textit{attribute anomalies}. We evaluate frontier VLMs under \textit{binary Question Answering (QA)} and \textit{multiple-choice QA}, and report metrics including \textit{Counterfactual Accuracy} (CF-Acc), \textit{Commonsense Accuracy} (CS-Acc), \textit{Counterfactual Accuracy Drop} (CFAD), \textit{Commonsense Collapse Rate} (CCR), and \textit{Relative Prior Dependency} (RPD). Results show that even strong models remain vulnerable to prior-driven normalization under visual evidence--commonsense conflict. CDH-Bench provides a controlled diagnostic of visual fidelity under visual evidence--commonsense conflict.
>
---
#### [new 104] Efficient Inference of Large Vision Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决LVLM推理效率低的问题。通过分类和分析现有优化方法，提出系统性框架以提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.27960](https://arxiv.org/pdf/2603.27960)**

> **作者:** Surendra Pathak
>
> **备注:** 12 pages
>
> **摘要:** Although Large Vision Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, their scalability and deployment are constrained by massive computational requirements. In particular, the massive amount of visual tokens from high-resolution input data aggravates the situation due to the quadratic complexity of attention mechanisms. To address these issues, the research community has developed several optimization frameworks. This paper presents a comprehensive survey of the current state-of-the-art techniques for accelerating LVLM inference. We introduce a systematic taxonomy that categorizes existing optimization frameworks into four primary dimensions: visual token compression, memory management and serving, efficient architectural design, and advanced decoding strategies. Furthermore, we critically examine the limitations of these current methodologies and identify critical open problems to inspire future research directions in efficient multimodal systems.
>
---
#### [new 105] daVinci-LLM:Towards the Science of Pretraining
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于预训练模型研究，旨在探索预训练对模型能力的影响。通过开放数据和方法，提出数据达尔文主义框架，验证了数据处理深度与模型性能的关系。**

- **链接: [https://arxiv.org/pdf/2603.27164](https://arxiv.org/pdf/2603.27164)**

> **作者:** Yiwei Qin; Yixiu Liu; Tiantian Mi; Muhang Xie; Zhen Huang; Weiye Si; Pengrui Lu; Siyuan Feng; Xia Wu; Liming Liu; Ye Luo; Jinlong Hou; Qipeng Guo; Yu Qiao; Pengfei Liu
>
> **摘要:** The foundational pretraining phase determines a model's capability ceiling, as post-training struggles to overcome capability foundations established during pretraining, yet it remains critically under-explored. This stems from a structural paradox: organizations with computational resources operate under commercial pressures that inhibit transparent disclosure, while academic institutions possess research freedom but lack pretraining-scale computational resources. daVinci-LLM occupies this unexplored intersection, combining industrial-scale resources with full research freedom to advance the science of pretraining. We adopt a fully-open paradigm that treats openness as scientific methodology, releasing complete data processing pipelines, full training processes, and systematic exploration results. Recognizing that the field lacks systematic methodology for data processing, we employ the Data Darwinism framework, a principled L0-L9 taxonomy from filtering to synthesis. We train a 3B-parameter model from random initialization across 8T tokens using a two-stage adaptive curriculum that progressively shifts from foundational capabilities to reasoning-intensive enhancement. Through 200+ controlled ablations, we establish that: processing depth systematically enhances capabilities, establishing it as a critical dimension alongside volume scaling; different domains exhibit distinct saturation dynamics, necessitating adaptive strategies from proportion adjustments to format shifts; compositional balance enables targeted intensification while preventing performance collapse; how evaluation protocol choices shape our understanding of pretraining progress. By releasing the complete exploration process, we enable the community to build upon our findings and systematic methodologies to form accumulative scientific knowledge in pretraining.
>
---
#### [new 106] ResAdapt: Adaptive Resolution for Efficient Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ResAdapt，解决多模态大模型中视觉分辨率与时间上下文难以兼顾的问题。通过输入侧自适应分配视觉预算，提升效率与准确率。属于多模态推理任务。**

- **链接: [https://arxiv.org/pdf/2603.28610](https://arxiv.org/pdf/2603.28610)**

> **作者:** Huanxuan Liao; Zhongtao Jiang; Yupu Hao; Yuqiao Tan; Shizhu He; Jun Zhao; Kun Xu; Kang Liu
>
> **备注:** work in progress
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve stronger visual understanding by scaling input fidelity, yet the resulting visual token growth makes jointly sustaining high spatial resolution and long temporal context prohibitive. We argue that the bottleneck lies not in how post-encoding representations are compressed but in the volume of pixels the encoder receives, and address it with ResAdapt, an Input-side adaptation framework that learns how much visual budget each frame should receive before encoding. ResAdapt couples a lightweight Allocator with an unchanged MLLM backbone, so the backbone retains its native visual-token interface while receiving an operator-transformed input. We formulate allocation as a contextual bandit and train the Allocator with Cost-Aware Policy Optimization (CAPO), which converts sparse rollout feedback into a stable accuracy-cost learning signal. Across budget-controlled video QA, temporal grounding, and image reasoning tasks, ResAdapt improves low-budget operating points and often lies on or near the efficiency-accuracy frontier, with the clearest gains on reasoning-intensive benchmarks under aggressive compression. Notably, ResAdapt supports up to 16x more frames at the same visual budget while delivering over 15% performance gain. Code is available at this https URL.
>
---
#### [new 107] SEAR: Schema-Based Evaluation and Routing for LLM Gateways
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文提出SEAR系统，解决多模型、多供应商LLM网关中的评估与路由问题，通过定义关系模式和结构化输出实现精准质量评估与高效路由决策。**

- **链接: [https://arxiv.org/pdf/2603.26728](https://arxiv.org/pdf/2603.26728)**

> **作者:** Zecheng Zhang; Han Zheng; Yue Xu
>
> **备注:** 10 pages, 6 pages appendix, 4 figures, 12 tables
>
> **摘要:** Evaluating production LLM responses and routing requests across providers in LLM gateways requires fine-grained quality signals and operationally grounded decisions. To address this gap, we present SEAR, a schema-based evaluation and routing system for multi-model, multi-provider LLM gateways. SEAR defines an extensible relational schema covering both LLM evaluation signals (context, intent, response characteristics, issue attribution, and quality scores) and gateway operational metrics (latency, cost, throughput), with cross-table consistency links across around one hundred typed, SQL-queryable columns. To populate the evaluation signals reliably, SEAR proposes self-contained signal instructions, in-schema reasoning, and multi-stage generation that produces database-ready structured outputs. Because signals are derived through LLM reasoning rather than shallow classifiers, SEAR captures complex request semantics, enables human-interpretable routing explanations, and unifies evaluation and routing in a single query layer. Across thousands of production sessions, SEAR achieves strong signal accuracy on human-labeled data and supports practical routing decisions, including large cost reductions with comparable quality.
>
---
#### [new 108] IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出IsoQuant，用于LLM KV缓存压缩，解决高存储和计算成本问题。通过四元数和SO(4)分解实现高效旋转，提升速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2603.28430](https://arxiv.org/pdf/2603.28430)**

> **作者:** Zhongping Ji
>
> **备注:** 11 pages
>
> **摘要:** Orthogonal feature decorrelation is effective for low-bit online vector quantization, but dense random orthogonal transforms incur prohibitive $O(d^2)$ storage and compute. RotorQuant reduces this cost with blockwise $3$D Clifford rotors, yet the resulting $3$D partition is poorly aligned with modern hardware and offers limited local mixing. We propose \textbf{IsoQuant}, a blockwise rotation framework based on quaternion algebra and the isoclinic decomposition of $SO(4)$. It represents each $4$D block as a quaternion and applies a closed-form transform $T(v)=q_L v \overline{q_R}$. This yields two main variants: \emph{IsoQuant-Full}, which realizes the full $SO(4)$ rotation, and \emph{IsoQuant-Fast}, which keeps only one isoclinic factor for lower cost; the framework also admits a lightweight $2$D special case. At $d=128$, IsoQuant-Full reduces forward rotation cost from about $2{,}408$ FMAs in RotorQuant to $1{,}024$, while IsoQuant-Fast further reduces it to $512$. Across $18$ fused CUDA settings with $d \in {128,256,512}$, bit widths ${2,3,4}$, and FP16/FP32 execution, IsoQuant achieves mean kernel-level speedups of about $4.5\times$--$4.7\times$ over RotorQuant while maintaining comparable reconstruction MSE, with peak speedups above $6\times$. Current validation is limited to the stage-1 quantize--dequantize path on synthetic normalized vectors; end-to-end KV-cache evaluation remains future work.
>
---
#### [new 109] Q-Bridge: Code Translation for Quantum Machine Learning via LLMs
- **分类: quant-ph; cs.CL**

- **简介: 该论文提出Q-Bridge，解决量子机器学习代码翻译问题，通过LLM构建可执行的量子代码，实现经典与量子模型的高效转换。**

- **链接: [https://arxiv.org/pdf/2603.27836](https://arxiv.org/pdf/2603.27836)**

> **作者:** Runjia Zeng; Priyabrata Senapati; Ruixiang Tang; Dongfang Liu; Qiang Guan
>
> **摘要:** Large language models have recently shown potential in bridging the gap between classical machine learning and quantum machine learning. However, the lack of standardized, high-quality datasets and robust translation frameworks limits progress in this domain. We introduce Q-Bridge, an LLM-guided code translation framework that systematically converts CML implementations into executable QML variants. Our approach builds on a self-involving pipeline that iteratively expands a verified seed codebase into a large-scale dataset, CML-2-QML, integrating verifiable and unverifiable code pairs. The Q-Bridge model is fine-tuned using supervised LoRA adaptation for scalable and memory-efficient training, achieving faithful and interpretable quantum code generation across diverse architectures. Empirical analysis confirms the feasibility of direct CML-to-QML translation and reveals consistent structural alignment between classical and quantum paradigms. Case studies further demonstrate that Q-Bridge can maintain deterministic correctness and also enable creative architectural exploration. This work establishes the first reproducible framework and dataset for LLM-driven quantum code translation, offering a foundation for scalable quantum AI development.
>
---
#### [new 110] SRAG: RAG with Structured Data Improves Vector Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升RAG的向量检索效果。通过引入结构化数据优化查询与片段表示，显著提高问答系统的评分。**

- **链接: [https://arxiv.org/pdf/2603.26670](https://arxiv.org/pdf/2603.26670)**

> **作者:** Shalin Shah; Srikanth Ryali; Ramasubbu Venkatesh
>
> **摘要:** Retrieval Augmented Generation (RAG) provides the necessary informational grounding to LLMs in the form of chunks retrieved from a vector database or through web search. RAG could also use knowledge graph triples as a means of providing factual information to an LLM. However, the retrieval is only based on representational similarity between a question and the contents. The performance of RAG depends on the numeric vector representations of the query and the chunks. To improve these representations, we propose Structured RAG (SRAG), which adds structured information to a query as well as the chunks in the form of topics, sentiments, query and chunk types (e.g., informational, quantitative), knowledge graph triples and semantic tags. Experiments indicate that this method significantly improves the retrieval process. Using GPT-5 as an LLM-as-a-judge, results show that the method improves the score given to answers in a question answering system by 30% (p-value = 2e-13) (with tighter bounds). The strongest improvement is in comparative, analytical and predictive questions. The results suggest that our method enables broader, more diverse, and episodic-style retrieval. Tail risk analysis shows that SRAG attains very large gains more often, with losses remaining minor in magnitude.
>
---
#### [new 111] MOSS-VoiceGenerator: Create Realistic Voices with Natural Language Descriptions
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音生成任务，旨在通过自然语言描述生成真实语音。解决现有模型语音过于人工、缺乏真实感的问题，通过大规模影视数据训练提升语音自然度。**

- **链接: [https://arxiv.org/pdf/2603.28086](https://arxiv.org/pdf/2603.28086)**

> **作者:** Kexin Huang; Liwei Fan; Botian Jiang; Yaozhou Jiang; Qian Tu; Jie Zhu; Yuqian Zhang; Yiwei Zhao; Chenchen Yang; Zhaoye Fei; Shimin Li; Xiaogui Yang; Qinyuan Cheng; Xipeng Qiu
>
> **摘要:** Voice design from natural language aims to generate speaker timbres directly from free-form textual descriptions, allowing users to create voices tailored to specific roles, personalities, and emotions. Such controllable voice creation benefits a wide range of downstream applications-including storytelling, game dubbing, role-play agents, and conversational assistants, making it a significant task for modern Text-to-Speech models. However, existing models are largely trained on carefully recorded studio data, which produces speech that is clean and well-articulated, yet lacks the lived-in qualities of real human voices. To address these limitations, we present MOSS-VoiceGenerator, an open-source instruction-driven voice generation model that creates new timbres directly from natural language prompts. Motivated by the hypothesis that exposure to real-world acoustic variation produces more perceptually natural voices, we train on large-scale expressive speech data sourced from cinematic content. Subjective preference studies demonstrate its superiority in overall performance, instruction-following, and naturalness compared to other voice design models.
>
---
#### [new 112] Heterogeneous Debate Engine: Identity-Grounded Cognitive Architecture for Resilient LLM-Based Ethical Tutoring
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.MA**

- **简介: 该论文属于伦理辅导任务，旨在解决LLM在辩证互动中出现的语义漂移和逻辑退化问题。提出HDE架构，结合ID-RAG与Heuristic ToM，提升辩论稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.27404](https://arxiv.org/pdf/2603.27404)**

> **作者:** Jakub Masłowski; Jarosław A. Chudziak
>
> **备注:** 15 pages, 3 figures, 4 tables. Accepted at ACIIDS 2026
>
> **摘要:** Large Language Models (LLMs) are being increasingly used as autonomous agents in complex reasoning tasks, opening the niche for dialectical interactions. However, Multi-Agent systems implemented with systematically unconstrained systems systematically undergo semantic drift and logical deterioration and thus can hardly be used in providing ethical tutoring where a precise answer is required. Current simulation often tends to degenerate into dialectical stagnation, the agents degenerate into recursive concurrence or circular arguments. A critical challenge remains: how to enforce doctrinal fidelity without suppressing the generative flexibility required for dialectical reasoning? To address this niche, we contribute the Heterogeneous Debate Engine (HDE), a cognitive architecture that combines Identity-Grounded Retrieval-Augmented Generation (ID-RAG) for doctrinal fidelity and Heuristic Theory of Mind for strategic opponent modeling. Our evaluation shows that architectural heterogeneity is a crucial variable to stability: contrary doctrinal initializations (e.g., Deontology vs. Utilitarianism) have increased the Argument Complexity Scores of students by an order of magnitude, over baselines. These findings validate the effectiveness of ID-RAG and Heuristic ToM as architectural requirements in maintaining high-fidelity (adversarial) pedagogy.
>
---
#### [new 113] Bridge-RAG: An Abstract Bridge Tree Based Retrieval Augmented Generation Algorithm With Cuckoo Filter
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出Bridge-RAG，解决RAG框架中检索准确性和效率问题，通过抽象树结构和改进的Cuckoo Filter提升性能。**

- **链接: [https://arxiv.org/pdf/2603.26668](https://arxiv.org/pdf/2603.26668)**

> **作者:** Zihang Li; Wenjun Liu; Yikun Zong; Jiawen Tao; Siying Dai; Songcheng Ren; Zirui Liu; Yanbing Jiang; Tong Yang
>
> **摘要:** As an important paradigm for enhancing the generation quality of Large Language Models (LLMs), retrieval-augmented generation (RAG) faces the two challenges regarding retrieval accuracy and computational efficiency. This paper presents a novel RAG framework called Bridge-RAG. To overcome the accuracy challenge, we introduce the concept of abstract to bridge query entities and document chunks, providing robust semantic understanding. We organize the abstracts into a tree structure and design a multi-level retrieval strategy to ensure the inclusion of sufficient contextual information. To overcome the efficiency challenge, we introduce the improved Cuckoo Filter, an efficient data structure supporting rapid membership queries and updates, to accelerate entity location during the retrieval process. We design a block linked list structure and an entity temperature-based sorting mechanism to improve efficiency from the aspects of spatial and temporal locality. Extensive experiments show that Bridge-RAG achieves around 15.65% accuracy improvement and reduces 10x to 500x retrieval time compared to other RAG frameworks.
>
---
#### [new 114] MiroEval: Benchmarking Multimodal Deep Research Agents in Process and Outcome
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MiroEval，用于评估多模态深度研究系统。解决现有评估方法无法全面反映研究过程的问题，通过三个维度评估系统能力，涵盖合成质量、事实验证和过程分析。**

- **链接: [https://arxiv.org/pdf/2603.28407](https://arxiv.org/pdf/2603.28407)**

> **作者:** Fangda Ye; Yuxin Hu; Pengxiang Zhu; Yibo Li; Ziqi Jin; Yao Xiao; Yibo Wang; Lei Wang; Zhen Zhang; Lu Wang; Yue Deng; Bin Wang; Yifan Zhang; Liangcai Su; Xinyu Wang; He Zhao; Chen Wei; Qiang Ren; Bryan Hooi; An Bo; Shuicheng Yan; Lidong Bing
>
> **备注:** GitHub: this https URL
>
> **摘要:** Recent progress in deep research systems has been impressive, but evaluation still lags behind real user needs. Existing benchmarks predominantly assess final reports using fixed rubrics, failing to evaluate the underlying research process. Most also offer limited multimodal coverage, rely on synthetic tasks that do not reflect real-world query complexity, and cannot be refreshed as knowledge evolves. To address these gaps, we introduce MiroEval, a benchmark and evaluation framework for deep research systems. The benchmark comprises 100 tasks (70 text-only, 30 multimodal), all grounded in real user needs and constructed via a dual-path pipeline that supports periodic updates, enabling a live and evolving setting. The proposed evaluation suite assesses deep research systems along three complementary dimensions: adaptive synthesis quality evaluation with task-specific rubrics, agentic factuality verification via active retrieval and reasoning over both web sources and multimodal attachments, and process-centric evaluation audits how the system searches, reasons, and refines throughout its investigation. Evaluation across 13 systems yields three principal findings: the three evaluation dimensions capture complementary aspects of system capability, with each revealing distinct strengths and weaknesses across systems; process quality serves as a reliable predictor of overall outcome while revealing weaknesses invisible to output-level metrics; and multimodal tasks pose substantially greater challenges, with most systems declining by 3 to 10 points. The MiroThinker series achieves the most balanced performance, with MiroThinker-H1 ranking the highest overall in both settings. Human verification and robustness results confirm the reliability of the benchmark and evaluation framework. MiroEval provides a holistic diagnostic tool for the next generation of deep research agents.
>
---
#### [new 115] Does Claude's Constitution Have a Culture?
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究，探讨宪法式AI是否反映特定文化。通过测试Claude模型，发现其价值倾向偏向北欧和英语国家，且不易受文化背景影响。研究揭示了AI文化偏见可能被固化的问题。**

- **链接: [https://arxiv.org/pdf/2603.28123](https://arxiv.org/pdf/2603.28123)**

> **作者:** Parham Pourdavood
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Constitutional AI (CAI) aligns language models with explicitly stated normative principles, offering a transparent alternative to implicit alignment through human feedback alone. However, because constitutions are authored by specific groups of people, the resulting models may reflect particular cultural perspectives. We investigate this question by evaluating Anthropic's Claude Sonnet on 55 World Values Survey items, selected for high cross-cultural variance across six value domains and administered as both direct survey questions and naturalistic advice-seeking scenarios. Comparing Claude's responses to country-level data from 90 nations, we find that Claude's value profile most closely resembles those of Northern European and Anglophone countries, but on a majority of items extends beyond the range of all surveyed populations. When users provide cultural context, Claude adjusts its rhetorical framing but not its substantive value positions, with effect sizes indistinguishable from zero across all twelve tested countries. An ablation removing the system prompt increases refusals but does not alter the values expressed when responses are given, and replication on a smaller model (Claude Haiku) confirms the same cultural profile across model sizes. These findings suggest that when a constitution is authored within the same cultural tradition that dominates the training data, constitutional alignment may codify existing cultural biases rather than correct them--producing a value floor that surface-level interventions cannot meaningfully shift. We discuss the compounding nature of this risk and the need for globally representative constitution-authoring processes.
>
---
#### [new 116] LongCat-Next: Lexicalizing Modalities as Discrete Tokens
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，旨在解决现有系统对非语言模态处理不融合的问题。提出DiNA框架和dNaViT模型，实现跨模态的统一离散建模。**

- **链接: [https://arxiv.org/pdf/2603.27538](https://arxiv.org/pdf/2603.27538)**

> **作者:** Meituan LongCat Team; Bin Xiao; Chao Wang; Chengjiang Li; Chi Zhang; Chong Peng; Hang Yu; Hao Yang; Haonan Yan; Haoze Sun; Haozhe Zhao; Hong Liu; Hui Su; Jiaqi Zhang; Jiawei Wang; Jing Li; Kefeng Zhang; Manyuan Zhang; Minhao Jing; Peng Pei; Quan Chen; Taofeng Xue; Tongxin Pan; Xiaotong Li; Xiaoyang Li; Xiaoyu Zhao; Xing Hu; Xinyang Lin; Xunliang Cai; Yan Bai; Yan Feng; Yanjie Li; Yao Qiu; Yerui Sun; Yifan Lu; Ying Luo; Yipeng Mei; Yitian Chen; Yuchen Xie; Yufang Liu; Yufei Chen; Yulei Qian; Yuqi Peng; Zhihang Yu; Zhixiong Han; Changran Wang; Chen Chen; Dian Zheng; Fengjiao Chen; Ge Yang; Haowei Guo; Haozhe Wang; Hongyu Li; Huicheng Jiang; Jiale Hong; Jialv Zou; Jiamu Li; Jianping Lin; Jiaxing Liu; Jie Yang; Jing Jin; Jun Kuang; Juncheng She; Kunming Luo; Kuofeng Gao; Lin Qiu; Linsen Guo; Mianqiu Huang; Qi Li; Qian Wang; Rumei Li; Siyu Ren; Wei Wang; Wenlong He; Xi Chen; Xiao Liu; Xiaoyu Li; Xu Huang; Xuanyu Zhu; Xuezhi Cao; Yaoming Zhu; Yifei Cao; Yimeng Jia; Yizhen Jiang; Yufei Gao; Zeyang Hu; Zhenlong Yuan; Zijian Zhang; Ziwen Wang
>
> **备注:** LongCat-Next Technical Report
>
> **摘要:** The prevailing Next-Token Prediction (NTP) paradigm has driven the success of large language models through discrete autoregressive modeling. However, contemporary multimodal systems remain language-centric, often treating non-linguistic modalities as external attachments, leading to fragmented architectures and suboptimal integration. To transcend this limitation, we introduce Discrete Native Autoregressive (DiNA), a unified framework that represents multimodal information within a shared discrete space, enabling a consistent and principled autoregressive modeling across modalities. A key innovation is the Discrete Native Any-resolution Visual Transformer (dNaViT), which performs tokenization and de-tokenization at arbitrary resolutions, transforming continuous visual signals into hierarchical discrete tokens. Building on this foundation, we develop LongCat-Next, a native multimodal model that processes text, vision, and audio under a single autoregressive objective with minimal modality-specific design. As an industrial-strength foundation model, it excels at seeing, painting, and talking within a single framework, achieving strong performance across a wide range of multimodal benchmarks. In particular, LongCat-Next addresses the long-standing performance ceiling of discrete vision modeling on understanding tasks and provides a unified approach to effectively reconcile the conflict between understanding and generation. As an attempt toward native multimodality, we open-source the LongCat-Next and its tokenizers, hoping to foster further research and development in the community. GitHub: this https URL
>
---
#### [new 117] LightMover: Generative Light Movement with Color and Intensity Controls
- **分类: cs.CV; cs.CL; cs.GR; cs.LG**

- **简介: 该论文提出LightMover，解决单图像中可控光照编辑问题。通过视频扩散先验生成物理合理光照变化，实现对光位、颜色、强度的精确控制。**

- **链接: [https://arxiv.org/pdf/2603.27209](https://arxiv.org/pdf/2603.27209)**

> **作者:** Gengze Zhou; Tianyu Wang; Soo Ye Kim; Zhixin Shu; Xin Yu; Yannick Hold-Geoffroy; Sumit Chaturvedi; Qi Wu; Zhe Lin; Scott Cohen
>
> **备注:** CVPR 2026. 10 pages, 5 figures, 6 tables in main paper; supplementary material included
>
> **摘要:** We present LightMover, a framework for controllable light manipulation in single images that leverages video diffusion priors to produce physically plausible illumination changes without re-rendering the scene. We formulate light editing as a sequence-to-sequence prediction problem in visual token space: given an image and light-control tokens, the model adjusts light position, color, and intensity together with resulting reflections, shadows, and falloff from a single view. This unified treatment of spatial (movement) and appearance (color, intensity) controls improves both manipulation and illumination understanding. We further introduce an adaptive token-pruning mechanism that preserves spatially informative tokens while compactly encoding non-spatial attributes, reducing control sequence length by 41% while maintaining editing fidelity. To train our framework, we construct a scalable rendering pipeline that generates large numbers of image pairs across varied light positions, colors, and intensities while keeping the scene content consistent with the original image. LightMover enables precise, independent control over light position, color, and intensity, and achieves high PSNR and strong semantic consistency (DINO, CLIP) across different tasks.
>
---
#### [new 118] KVSculpt: KV Cache Compression as Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出KVSculpt，解决长上下文大模型推理中的KV缓存压缩问题。通过优化少量连续嵌入空间的KV对，提升压缩效率，减少KL散度。**

- **链接: [https://arxiv.org/pdf/2603.27819](https://arxiv.org/pdf/2603.27819)**

> **作者:** Bo Jiang; Sian Jin
>
> **摘要:** KV cache compression is critical for efficient long-context LLM inference. Approaches that reduce the per-pair footprint -- quantization and low-rank decomposition -- are orthogonal to those that reduce the sequence length of the cache. Along the sequence-length dimension, existing methods range from pure eviction -- selecting which KV pairs to keep -- to merging, which combines similar pairs into fewer ones. Both remain anchored to the original cache entries. We propose KVSculpt, which moves to the other end of this spectrum: instead of selecting or combining original pairs, we optimize a smaller set of unconstrained KV pairs in continuous embedding space to preserve each layer's attention behavior. Keys are optimized via L-BFGS and values are solved in closed form via least squares, alternating every few steps. On top of this, we introduce adaptive budget allocation, which uses a cheap pilot compression run to redistribute the compression budget across layers and KV heads based on per-component difficulty. On Qwen2.5-1.5B-Instruct with 2048-token contexts, KVSculpt reduces KL divergence by 3.5-4.1x compared to Select+Fit -- attention-score eviction with least-squares value fitting -- across compression ratios r in {0.3, 0.5, 0.7}. Adaptive allocation provides an additional 1.3x KL reduction at no extra inference cost. Analysis reveals that compression difficulty is highly non-uniform: per-layer pilot MSE varies by up to 100x across layers, and the two KV heads within a single layer can differ by up to 467x -- demonstrating that fine-grained budget allocation is essential.
>
---
#### [new 119] Agentic AI for Human Resources: LLM-Driven Candidate Assessment
- **分类: cs.IR; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于人力资源招聘任务，旨在解决传统招聘工具评估不精准的问题。通过LLM构建模块化框架，实现候选人的结构化评估与排名。**

- **链接: [https://arxiv.org/pdf/2603.26710](https://arxiv.org/pdf/2603.26710)**

> **作者:** Kamer Ali Yuksel; Abdul Basit Anees; Ashraf Elneima; Sanjika Hewavitharana; Mohamed Al-Badrashiny; Hassan Sawaf
>
> **备注:** Published in 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)
>
> **摘要:** In this work, we present a modular and interpretable framework that uses Large Language Models (LLMs) to automate candidate assessment in recruitment. The system integrates diverse sources, including job descriptions, CVs, interview transcripts, and HR feedback; to generate structured evaluation reports that mirror expert judgment. Unlike traditional ATS tools that rely on keyword matching or shallow scoring, our approach employs role-specific, LLM-generated rubrics and a multi-agent architecture to perform fine-grained, criteria-driven evaluations. The framework outputs detailed assessment reports, candidate comparisons, and ranked recommendations that are transparent, auditable, and suitable for real-world hiring workflows. Beyond rubric-based analysis, we introduce an LLM-Driven Active Listwise Tournament mechanism for candidate ranking. Instead of noisy pairwise comparisons or inconsistent independent scoring, the LLM ranks small candidate subsets (mini-tournaments), and these listwise permutations are aggregated using a Plackett-Luce model. An active-learning loop selects the most informative subsets, producing globally coherent and sample-efficient rankings. This adaptation of listwise LLM preference modeling (previously explored in financial asset ranking) provides a principled and highly interpretable methodology for large-scale candidate ranking in talent acquisition.
>
---
## 更新

#### [replaced 001] LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，旨在解决MLLMs在生成过程中被诱导产生冗长重复内容的问题。通过POS感知机制和路径剪枝，使模型陷入生成循环，导致资源耗尽。**

- **链接: [https://arxiv.org/pdf/2506.14493](https://arxiv.org/pdf/2506.14493)**

> **作者:** Jiyuan Fu; Kaixun Jiang; Lingyi Hong; Jinglun Li; Haijing Guo; Dingkang Yang; Zhaoyu Chen; Wenqiang Zhang
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments on models like Qwen2.5-VL-3B demonstrate LingoLoop's powerful ability to trap them in generative loops; it consistently drives them to their generation limits and, when those limits are relaxed, can induce outputs with up to 367x more tokens than clean inputs, triggering a commensurate surge in energy consumption. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment.
>
---
#### [replaced 002] Neuron-Level Analysis of Cultural Understanding in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型文化理解偏差问题。通过神经元级分析，识别文化相关神经元，并验证其对文化基准的影响。**

- **链接: [https://arxiv.org/pdf/2510.08284](https://arxiv.org/pdf/2510.08284)**

> **作者:** Taisei Yamamoto; Ryoma Kumon; Danushka Bollegala; Hitomi Yanaka
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** As large language models (LLMs) are increasingly deployed worldwide, ensuring their fair and comprehensive cultural understanding is important. However, LLMs exhibit cultural bias and limited awareness of underrepresented cultures, while the mechanisms underlying their cultural understanding remain underexplored. To fill this gap, we conduct a neuron-level analysis to identify neurons that drive cultural behavior, introducing a gradient-based scoring method with additional filtering for precise refinement. We identify culture-general neurons contributing to cultural understanding regardless of cultures, and culture-specific neurons tied to an individual culture. Culture-general and culture-specific neurons account for less than 1% of all neurons and are concentrated in shallow to middle MLP layers. We validate their role by showing that suppressing them substantially degrades performance on cultural benchmarks (by up to 30%), while performance on general natural language understanding (NLU) benchmarks remains largely unaffected. Moreover, we show that culture-specific neurons support knowledge of not only the target culture, but also related cultures. Finally, we demonstrate that training on NLU benchmarks can diminish models' cultural understanding when we update modules containing many culture-general neurons. These findings provide insights into the internal mechanisms of LLMs and offer practical guidance for model training and engineering. Our code is available at this https URL
>
---
#### [replaced 003] MetaState: Persistent Working Memory Enhances Reasoning in Discrete Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，针对离散扩散语言模型的推理能力不足问题，提出MetaState方法，通过引入持续工作记忆增强模型推理性能。**

- **链接: [https://arxiv.org/pdf/2603.01331](https://arxiv.org/pdf/2603.01331)**

> **作者:** Kejing Xia; Mingzhe Li; Lixuan Wei; Zhenbang Du; Xiangchi Yuan; Dachuan Shi; Qirui Jin; Wenke Lee
>
> **摘要:** Discrete diffusion language models (dLLMs) generate text by iteratively denoising a masked sequence. However, standard dLLMs condition each denoising step solely on the current hard-masked sequence, while intermediate continuous representations are discarded after sampling and remasking. We term this bottleneck the \textbf{Information Island} issue: continuous information remains isolated within individual denoising steps and fails to propagate across the trajectory. This bottleneck is especially harmful for reasoning, which requires intermediate reasoning state to be preserved and updated across many denoising steps. To address this limitation, we introduce \textbf{MetaState}, a lightweight recurrent augmentation that equips a frozen dLLM backbone with persistent, fixed-size working memory. MetaState comprises three modules with a shared time conditioner: a cross-attention \textbf{Mixer} that reads backbone activations into memory slots, a GRU-style \textbf{Updater} that integrates information across steps, and a cross-attention \textbf{Injector} that writes the updated memory back into the backbone. We train these modules with a dedicated $K$-step unrolling pipeline to learn multi-step dynamics. MetaState adds only ${\sim}0.6\%$ trainable parameters while keeping the backbone frozen, and consistently improves reasoning performance over frozen baselines on mathematical reasoning and code generation benchmarks, with an average gain of $4.5\%$ across all evaluations.
>
---
#### [replaced 004] JMedEthicBench: A Multi-Turn Conversational Benchmark for Evaluating Medical Safety in Japanese Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗安全评估任务，旨在解决LLM在日语医疗对话中的安全问题。构建了首个多轮对话基准JMedEthicBench，测试模型安全性并发现多轮交互中的安全下降现象。**

- **链接: [https://arxiv.org/pdf/2601.01627](https://arxiv.org/pdf/2601.01627)**

> **作者:** Junyu Liu; Zirui Li; Qian Niu; Zequn Zhang; Yue Xun; Wenlong Hou; Shujun Wang; Yusuke Iwasawa; Yutaka Matsuo; Kan Hatakeyama-Sato
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in healthcare field, it becomes essential to carefully evaluate their medical safety before clinical use. However, existing safety benchmarks remain predominantly English-centric, and test with only single-turn prompts despite multi-turn clinical consultations. To address these gaps, we introduce JMedEthicBench, the first multi-turn conversational benchmark for evaluating medical safety of LLMs for Japanese healthcare. Our benchmark is based on 67 guidelines from the Japan Medical Association and contains over 50,000 adversarial conversations generated using seven automatically discovered jailbreak strategies. Using a dual-LLM scoring protocol, we evaluate 27 models and find that commercial models maintain robust safety while medical-specialized models exhibit increased vulnerability. Furthermore, safety scores decline significantly across conversation turns (median: 9.5 to 5.0, $p < 0.001$). Cross-lingual evaluation on both Japanese and English versions of our benchmark reveals that medical model vulnerabilities persist across languages, indicating inherent alignment limitations rather than language-specific factors. These findings suggest that domain-specific fine-tuning may accidentally weaken safety mechanisms and that multi-turn interactions represent a distinct threat surface requiring dedicated alignment strategies.
>
---
#### [replaced 005] †DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数学问题求解任务，旨在解决无关信息干扰下的推理问题。通过引入基准数据集和提出DAGGER模型，提升模型在噪声环境下的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2601.06853](https://arxiv.org/pdf/2601.06853)**

> **作者:** Zabir Al Nazi; Shubhashis Roy Dipta; Sudipta Kar
>
> **摘要:** Chain-of-Thought (CoT) prompting is widely adopted for mathematical problem solving, including in low-resource languages, yet its behavior under irrelevant context remains underexplored. To systematically study this challenge, we introduce DISTRACTMATH-BN, a Bangla benchmark that augments MGSM and MSVAMP with semantically coherent but computationally irrelevant information. Evaluating seven models ranging from 3B to 12B parameters, we observe substantial performance degradation under distractors: standard models drop by up to 41 points, while reasoning-specialized models decline by 14 to 20 points despite consuming five times more tokens. We propose †DAGGER, which reformulates mathematical problem solving as executable computational graph generation with explicit modeling of distractor nodes. Fine-tuning Gemma-3 models using supervised fine-tuning followed by Group Relative Policy Optimization achieves comparable weighted accuracy on augmented benchmarks while using 89 percent fewer tokens than reasoning models. Importantly, this robustness emerges without explicit training on distractor-augmented examples. Our results suggest that enforcing structured intermediate representations improves robustness and inference efficiency in mathematical reasoning compared to free-form approaches, particularly in noisy, low-resource settings.
>
---
#### [replaced 006] Dual-Space Smoothness for Robust and Balanced LLM Unlearning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在解决遗忘效果与模型性能、隐私保护之间的失衡问题。提出PRISM框架，通过双空间平滑提升模型鲁棒性与平衡性。**

- **链接: [https://arxiv.org/pdf/2509.23362](https://arxiv.org/pdf/2509.23362)**

> **作者:** Han Yan; Zheyuan Liu; Meng Jiang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** As large language models evolve, Machine Unlearning has emerged to address growing concerns around user privacy, copyright infringement, and overall safety. Yet state-of-the-art (SOTA) unlearning methods often suffer from catastrophic forgetting and metric imbalance, for example, by over-optimizing one objective (e.g., unlearning effectiveness, utility preservation, or privacy protection) at the expense of others. In addition, small perturbations in the representation or parameter space can be exploited by relearn and jailbreak attacks. To address these challenges, we propose PRISM, a unified framework that enforces dual-space smoothness in representation and parameter spaces to improve robustness and balance unlearning metrics. PRISM consists of two smoothness optimization stages: (i) a representation space stage that employs a robustly trained probe to defend against jailbreak attacks, and (ii) a parameter-space stage that decouples retain-forget gradient conflicts, reduces imbalance, and smooths the parameter space to mitigate relearning attacks. Extensive experiments on WMDP and MUSE, across conversational-dialogue and continuous-text settings, show that PRISM outperforms SOTA baselines under multiple attacks while achieving a better balance among key metrics.
>
---
#### [replaced 007] PaperVoyager : Building Interactive Web with Visual Language Models
- **分类: cs.CL**

- **简介: 该论文提出PaperVoyager，将科研论文转化为可交互的网页系统，解决静态文档无法体现动态机制的问题。**

- **链接: [https://arxiv.org/pdf/2603.22999](https://arxiv.org/pdf/2603.22999)**

> **作者:** Dasen Dai; Biao Wu; Meng Fang; Wenhao Wang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Recent advances in visual language models have enabled autonomous agents for complex reasoning, tool use, and document understanding. However, existing document agents mainly transform papers into static artifacts such as summaries, webpages, or slides, which are insufficient for technical papers involving dynamic mechanisms and state transitions. In this work, we propose a Paper-to-Interactive-System Agent that converts research papers into executable interactive web systems. Given a PDF paper, the agent performs end-to-end processing without human intervention, including paper understanding, system modeling, and interactive webpage synthesis, enabling users to manipulate inputs and observe dynamic behaviors. To evaluate this task, we introduce a benchmark of 19 research papers paired with expert-built interactive systems as ground truth. We further propose PaperVoyager, a structured generation framework that explicitly models mechanisms and interaction logic during synthesis. Experiments show that PaperVoyager significantly improves the quality of generated interactive systems, offering a new paradigm for interactive scientific paper understanding.
>
---
#### [replaced 008] AppellateGen: A Benchmark for Appellate Legal Judgment Generation
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于法律判决生成任务，旨在解决上诉阶段判决生成问题。提出AppellateGen基准和SLMAS系统，以模拟司法流程并提升逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2601.01331](https://arxiv.org/pdf/2601.01331)**

> **作者:** Hongkun Yang; Lionel Z. Wang; Wei Fan; Yiran Hu; Lixu Wang; Chenyu Liu; Yu Zeng; Shenghong Fu; Lei Gong; Zhengxin Zhang; Haoyang Li; Jiexin Zheng; Xin Xu
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** Legal judgment generation is a critical task in legal intelligence. However, existing research in legal judgment generation has predominantly focused on first-instance trials, relying on static fact-to-verdict mappings while neglecting the dialectical nature of appellate (second-instance) review. To address this, we introduce AppellateGen, a benchmark for second-instance legal judgment generation comprising 7,351 case pairs. The task requires models to draft legally binding judgments by reasoning over the initial verdict and evidentiary updates, thereby modeling the causal dependency between trial stages. We further propose a judicial Standard Operating Procedure (SOP)-based Legal Multi-Agent System (SLMAS) to simulate judicial workflows, which decomposes the generation process into discrete stages of issue identification, retrieval, and drafting. Experimental results indicate that while SLMAS improves logical consistency, the complexity of appellate reasoning remains a substantial challenge for current LLMs. The dataset and code are publicly available at: this https URL.
>
---
#### [replaced 009] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VideoARM，用于长视频理解任务，解决传统方法依赖预处理和高token消耗的问题，通过自适应推理与分层记忆机制提升效果并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2512.12360](https://arxiv.org/pdf/2512.12360)**

> **作者:** Yufei Yin; Qianke Meng; Minghao Chen; Jiajun Ding; Zhenwei Shao; Zhou Yu
>
> **备注:** Accepted to CVPR 2026, code available at this https URL
>
> **摘要:** Long-form video understanding remains challenging due to the extended temporal structure and dense multimodal cues. Despite recent progress, many existing approaches still rely on hand-crafted reasoning pipelines or employ token-consuming video preprocessing to guide MLLMs in autonomous reasoning. To overcome these limitations, we introduce VideoARM, an Agentic Reasoning-over-hierarchical-Memory paradigm for long-form video understanding. Instead of static, exhaustive preprocessing, VideoARM performs adaptive, on-the-fly agentic reasoning and memory construction. Specifically, VideoARM performs an adaptive and continuous loop of observing, thinking, acting, and memorizing, where a controller autonomously invokes tools to interpret the video in a coarse-to-fine manner, thereby substantially reducing token consumption. In parallel, a hierarchical multimodal memory continuously captures and updates multi-level clues throughout the operation of the agent, providing precise contextual information to support the controller in decision-making. Experiments on prevalent benchmarks demonstrate that VideoARM outperforms the state-of-the-art method, DVD, while significantly reducing token consumption for long-form videos.
>
---
#### [replaced 010] Beg to Differ: Understanding Reasoning-Answer Misalignment Across Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言模型推理评估任务，旨在解决模型推理与答案不一致的问题。通过分析65k个跨语言的推理轨迹，发现非拉丁语系模型存在更高比例的推理错误。**

- **链接: [https://arxiv.org/pdf/2512.22712](https://arxiv.org/pdf/2512.22712)**

> **作者:** Anaelia Ovalle; Candace Ross; Sebastian Ruder; Adina Williams; Karen Ullrich; Mark Ibrahim; Levent Sagun
>
> **备注:** Accepted to 2025 EMNLP Multilingual Representation Learning Workshop
>
> **摘要:** Large language models demonstrate strong reasoning capabilities through chain-of-thought prompting, but whether this reasoning quality transfers across languages remains underexplored. We introduce a human-validated framework to evaluate whether model-generated reasoning traces logically support their conclusions across languages. Analyzing 65k reasoning traces from GlobalMMLU questions across 6 languages and 6 frontier models, we uncover a critical blind spot: while models achieve high task accuracy, their reasoning can fail to support their conclusions. Reasoning traces in non-Latin scripts show at least twice as much misalignment between their reasoning and conclusions than those in Latin scripts. We develop an error taxonomy through human annotation to characterize these failures, finding they stem primarily from evidential errors (unsupported claims, ambiguous facts) followed by illogical reasoning steps. Our findings demonstrate that current multilingual evaluation practices provide an incomplete picture of model reasoning capabilities and highlight the need for reasoning-aware evaluation frameworks.
>
---
#### [replaced 011] SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue
- **分类: cs.CL**

- **简介: 该论文提出SEAD框架，解决服务对话中因数据质量差导致的性能问题。通过用户建模分解，提升任务完成率和对话效率。**

- **链接: [https://arxiv.org/pdf/2602.03548](https://arxiv.org/pdf/2602.03548)**

> **作者:** Yuqin Dai; Ning Gao; Wei Zhang; Jie Wang; Zichen Luo; Jinpeng Wang; Yujie Wang; Ruiyuan Wu; Chaozheng Wang
>
> **摘要:** Large Language Models have demonstrated remarkable capabilities in open-domain dialogues. However, current methods exhibit suboptimal performance in service dialogues, as they rely on noisy, low-quality human conversation data. This limitation arises from data scarcity and the difficulty of simulating authentic, goal-oriented user behaviors. To address these issues, we propose SEAD (Self-Evolving Agent for Service Dialogue), a framework that enables agents to learn effective strategies without large-scale human annotations. SEAD decouples user modeling into two components: a Profile Controller that generates diverse user states to manage training curriculum, and a User Role-play Model that focuses on realistic role-playing. This design ensures the environment provides adaptive training scenarios rather than acting as an unfair adversary. Experiments demonstrate that SEAD significantly outperforms Open-source Foundation Models and Closed-source Commercial Models, improving task completion rate by 17.6% and dialogue efficiency by 11.1%. Code is available at: this https URL.
>
---
#### [replaced 012] Vision-Language Agents for Interactive Forest Change Analysis
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于遥感图像变化分析任务，旨在解决森林动态的像素级变化检测与语义描述问题。提出一种基于大语言模型的视觉语言代理系统，并构建了Forest-Change数据集进行验证。**

- **链接: [https://arxiv.org/pdf/2601.04497](https://arxiv.org/pdf/2601.04497)**

> **作者:** James Brock; Ce Zhang; Nantheera Anantrasirichai
>
> **备注:** 5 pages, 4 figures, Accepted into IGARSS 2026
>
> **摘要:** Modern forest monitoring workflows increasingly benefit from the growing availability of high-resolution satellite imagery and advances in deep learning. Two persistent challenges in this context are accurate pixel-level change detection and meaningful semantic change captioning for complex forest dynamics. While large language models (LLMs) are being adapted for interactive data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored. To address this gap, we introduce an LLM-driven agent for integrated forest change analysis that supports natural language querying across multiple RSICI tasks. The proposed system builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration. To facilitate adaptation and evaluation in forest environments, we further introduce the Forest-Change dataset, which comprises bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated using a combination of human annotation and rule-based methods. Experimental results show that the proposed system achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on the Forest-Change dataset, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI benchmark for joint change detection and captioning. These results highlight the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and efficiency of forest change analysis. All data and code are publicly available at this https URL.
>
---
#### [replaced 013] KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱推理任务，旨在解决大型语言模型在多跳推理中的效率与准确性问题。提出KG-Hopper框架，通过强化学习实现单次推理中的全局路径探索，提升性能并保持模型紧凑。**

- **链接: [https://arxiv.org/pdf/2603.21440](https://arxiv.org/pdf/2603.21440)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to IJCNN 2026
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL.
>
---
#### [replaced 014] Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于多语言偏差评估任务，旨在检测大语言模型中的隐性刻板印象。通过构建多语言辩论数据集，分析模型在不同语言和敏感领域中的偏见表现，揭示现有对齐方法在跨语言场景中的局限性。**

- **链接: [https://arxiv.org/pdf/2511.01187](https://arxiv.org/pdf/2511.01187)**

> **作者:** Muhammed Saeed; Muhammad Abdul-mageed; Shady Shehata
>
> **摘要:** Large language models (LLMs) are widely deployed for open-ended communication, yet most bias evaluations still rely on English, classification-style tasks. We introduce \corpusname, a new multilingual, debate-style benchmark designed to reveal how narrative bias appears in realistic generative settings. Our dataset includes 8{,}400 structured debate prompts spanning four sensitive domains -- Women's Rights, Backwardness, Terrorism, and Religion -- across seven languages ranging from high-resource (English, Chinese) to low-resource (Swahili, Nigerian Pidgin). Using four flagship models (GPT-4o, Claude~3.5~Haiku, DeepSeek-Chat, and LLaMA-3-70B), we generate over 100{,}000 debate responses and automatically classify which demographic groups are assigned stereotyped versus modern roles. Results show that all models reproduce entrenched stereotypes despite safety alignment: Arabs are overwhelmingly linked to Terrorism and Religion ($\geq$89\%), Africans to socioeconomic ``backwardness'' (up to 77\%), and Western groups are consistently framed as modern or progressive. Biases grow sharply in lower-resource languages, revealing that alignment trained primarily in English does not generalize globally. Our findings highlight a persistent divide in multilingual fairness: current alignment methods reduce explicit toxicity but fail to prevent biased outputs in open-ended contexts. We release our \corpusname benchmark and analysis framework to support the next generation of multilingual bias evaluation and safer, culturally inclusive model alignment.
>
---
#### [replaced 015] Using LLMs for Knowledge Component-level Correctness Labeling in Open-ended Coding Problems
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术领域，旨在解决开放性编程题中知识组件（KCs）正确性标签缺失的问题。通过引入大语言模型，实现KCs级别的正确性标注，并提升学习曲线拟合与预测性能。**

- **链接: [https://arxiv.org/pdf/2602.17542](https://arxiv.org/pdf/2602.17542)**

> **作者:** Zhangqi Duan; Arnav Kankaria; Dhruv Kartik; Andrew Lan
>
> **摘要:** Fine-grained skill representations, commonly referred to as knowledge components (KCs), are fundamental to many approaches in student modeling and learning analytics. However, KC-level correctness labels are rarely available in real-world datasets, especially for open-ended programming tasks where solutions typically involve multiple KCs simultaneously. Simply propagating problem-level correctness to all associated KCs obscures partial mastery and often leads to poorly fitted learning curves. To address this challenge, we propose an automated framework that leverages large language models (LLMs) to label KC-level correctness directly from student-written code. Our method assesses whether each KC is correctly applied and further introduces a temporal context-aware Code-KC mapping mechanism to better align KCs with individual student code. We evaluate the resulting KC-level correctness labels in terms of learning curve fit and predictive performance using the power law of practice and the Additive Factors Model. Experimental results show that our framework leads to learning curves that are more consistent with cognitive theory and improves predictive performance, compared to baselines. Human evaluation further demonstrates substantial agreement between LLM and expert annotations.
>
---
#### [replaced 016] Can Small Language Models Handle Context-Summarized Multi-Turn Customer-Service QA? A Synthetic Data-Driven Comparative Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究小语言模型在多轮客服问答中的表现，旨在评估其处理上下文摘要的能力。任务属于客服问答系统，解决资源受限环境下模型性能问题，通过实验对比不同模型表现。**

- **链接: [https://arxiv.org/pdf/2602.00665](https://arxiv.org/pdf/2602.00665)**

> **作者:** Lakshan Cooray; Deshan Sumanathilaka; Pattigadapa Venkatesh Raju
>
> **备注:** Submission is under review with Computational Linguistics
>
> **摘要:** Customer-service question answering (QA) systems increasingly rely on conversational language understanding. While Large Language Models (LLMs) achieve strong performance, their high computational cost and deployment constraints limit practical use in resource-constrained environments. Small Language Models (SLMs) provide a more efficient alternative, yet their effectiveness for multi-turn customer-service QA remains underexplored, particularly in scenarios requiring dialogue continuity and contextual understanding. This study investigates instruction-tuned SLMs for context-summarized multi-turn customer-service QA, using a history summarization strategy to preserve essential conversational state. We also introduce a conversation stage-based qualitative analysis to evaluate model behavior across different phases of customer-service interactions. Nine instruction-tuned low-parameterized SLMs are evaluated against three commercial LLMs using lexical and semantic similarity metrics alongside qualitative assessments, including human evaluation and LLM-as-a-judge methods. Results show notable variation across SLMs, with some models demonstrating near-LLM performance, while others struggle to maintain dialogue continuity and contextual alignment. These findings highlight both the potential and current limitations of low-parameterized language models for real-world customer-service QA systems.
>
---
#### [replaced 017] Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods
- **分类: cs.CL**

- **简介: 论文提出模型免疫方法，通过监督微调对抗虚假信息。任务是提升大模型的真相性，解决其因学习伪谬语言模式而传播错误的问题。工作包括设计疫苗数据和评估指标。**

- **链接: [https://arxiv.org/pdf/2505.17870](https://arxiv.org/pdf/2505.17870)**

> **作者:** Shaina Raza; Rizwan Qureshi; Azib Farooq; Marcelo Lotif; Aman Chadha; Deval Pandya; Christos Emmanouilidis
>
> **摘要:** Large language models (LLMs) reproduce misinformation not by memorizing false facts alone, but by learning the linguistic patterns that make falsehoods persuasive, such as hedging, false presuppositions, and fabricated citations. We propose model immunization, a training paradigm based on supervised fine-tuning over curated (false claim, correction) pairs, injected as small vaccine doses (5 to 10% of tokens) alongside truthful data. Unlike post-hoc filtering or preference-based alignment, immunization introduces direct negative supervision on labeled falsehoods. Across four open weight model families, this approach improves TruthfulQA accuracy by 12 points and increases misinformation rejection rates by 30 points, while preserving overall model capability. We further outline key design requirements, including dosage, labeling, quarantine, and diversity and advocate for standardized vaccine corpora and benchmarks to evaluate generalization. These findings position immunization as a practical and scalable component of responsible LLM development. Project page: this https URL
>
---
#### [replaced 018] Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨文字系统名称匹配任务，解决多语言地理名称整合难题。提出Symphonym模型，通过神经嵌入将不同文字系统的地名映射到统一语音空间，实现无需语言识别的跨文字相似性比较。**

- **链接: [https://arxiv.org/pdf/2601.06932](https://arxiv.org/pdf/2601.06932)**

> **作者:** Stephen Gadd
>
> **备注:** 19 pages, 3 tables
>
> **摘要:** Matching place names across writing systems is a persistent obstacle to the integration of multilingual geographic sources, whether modern gazetteers, medieval itineraries, or colonial-era surveys. Existing approaches depend on language-specific phonetic algorithms or romanisation steps that discard phonetic information, and none generalises across script boundaries. This paper presents Symphonym, a neural embedding system which maps toponyms from twenty writing systems into a unified 128-dimensional phonetic space, enabling direct cross-script similarity comparison without language identification or phonetic resources at inference time. A Teacher-Student knowledge distillation architecture first learns from articulatory phonetic features derived from IPA transcriptions, then transfers this knowledge to a character-level Student model. Trained on 32.7 million triplet samples drawn from 67 million toponyms spanning GeoNames, Wikidata, and the Getty Thesaurus of Geographic Names, the Student achieves the highest Recall@1 (85.2%) and Mean Reciprocal Rank (90.8%) on the MEHDIE cross-script benchmark -- medieval Hebrew and Arabic toponym matches curated by domain experts and entirely independent of the training data -- demonstrating cross-temporal generalisation from modern training material to pre-modern sources. An ablation using raw articulatory features alone yields only 45.0% MRR, confirming the contribution of the neural training curriculum. The approach naturally handles pre-standardisation orthographic variation characteristic of historical documents, and transfers effectively to personal names in archival sources, suggesting broad applicability to name resolution tasks in digital humanities and linked open data contexts.
>
---
#### [replaced 019] RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出RadImageNet-VQA数据集，用于解决医学影像中的视觉问答任务。针对现有数据集规模小、依赖文本等问题，该数据集包含大量CT和MRI图像及问答对，涵盖多种病理识别任务。**

- **链接: [https://arxiv.org/pdf/2512.17396](https://arxiv.org/pdf/2512.17396)**

> **作者:** Léo Butsanets; Charles Corbière; Julien Khlaut; Pierre Manceron; Corentin Dancette
>
> **备注:** Preprint, 33 pages, 15 figures, 11 tables
>
> **摘要:** In this work, we introduce RadImageNet-VQA, a large-scale dataset designed to advance radiologic visual question answering (VQA) on CT and MRI exams. Existing medical VQA datasets are limited in scale, dominated by X-ray imaging or biomedical illustrations, and often prone to text-based shortcuts. RadImageNet-VQA is built from expert-curated annotations and provides 750K images paired with 7.5M question-answer samples. It covers three key tasks - abnormality detection, anatomy recognition, and pathology identification - spanning eight anatomical regions and 97 pathology categories, and supports open-ended, closed-ended, and multiple-choice questions. Extensive experiments show that state-of-the-art vision-language models still struggle with fine-grained pathology identification, particularly in open-ended settings and even after fine-tuning. Text-only analysis further reveals that model performance collapses to near-random without image inputs, confirming that RadImageNet-VQA is free from linguistic shortcuts. The full dataset and benchmark are publicly available at this https URL.
>
---
#### [replaced 020] BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗自然语言处理任务，旨在解决真实临床文本理解的评估不足问题。提出BRIDGE基准，涵盖多语言、多任务、多专科的临床数据，评估95个大语言模型的表现。**

- **链接: [https://arxiv.org/pdf/2504.19467](https://arxiv.org/pdf/2504.19467)**

> **作者:** Jiageng Wu; Bowen Gu; Ren Zhou; Kevin Xie; Doug Snyder; Yixing Jiang; Valentina Carducci; Richard Wyss; Rishi J Desai; Emily Alsentzer; Leo Anthony Celi; Adam Rodman; Sebastian Schneeweiss; Jonathan H. Chen; Santiago Romero-Brufau; Kueiyu Joshua Lin; Jie Yang
>
> **摘要:** Large language models (LLMs) hold great promise for medical applications and are evolving rapidly, with new models being released at an accelerated pace. However, benchmarking on large-scale real-world data such as electronic health records (EHRs) is critical, as clinical decisions are directly informed by these sources, yet current evaluations remain limited. Most existing benchmarks rely on medical exam-style questions or PubMed-derived text, failing to capture the complexity of real-world clinical data. Others focus narrowly on specific application scenarios, limiting their generalizability across broader clinical use. To address this gap, we present BRIDGE, a comprehensive multilingual benchmark comprising 87 tasks sourced from real-world clinical data sources across nine languages. It covers eight major task types spanning the entire continuum of patient care across six clinical stages and 20 representative applications, including triage and referral, consultation, information extraction, diagnosis, prognosis, and billing coding, and involves 14 clinical specialties. We systematically evaluated 95 LLMs (including DeepSeek-R1, GPT-4o, Gemini series, and Qwen3 series) under various inference strategies. Our results reveal substantial performance variation across model sizes, languages, natural language processing tasks, and clinical specialties. Notably, we demonstrate that open-source LLMs can achieve performance comparable to proprietary models, while medically fine-tuned LLMs based on older architectures often underperform versus updated general-purpose models. The BRIDGE and its corresponding leaderboard serve as a foundational resource and a unique reference for the development and evaluation of new LLMs in real-world clinical text understanding. The BRIDGE leaderboard: this https URL
>
---
#### [replaced 021] CoPE-VideoLM: Leveraging Codec Primitives For Efficient Video Language Modeling
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决VideoLM中关键帧采样不全和计算冗余问题。通过利用视频编解码器原语，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.13191](https://arxiv.org/pdf/2602.13191)**

> **作者:** Sayan Deb Sarkar; Rémi Pautrat; Ondrej Miksik; Marc Pollefeys; Iro Armeni; Mahdi Rad; Mihai Dusmanu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Video Language Models (VideoLMs) enable AI systems to understand temporal dynamics in videos. To fit within the maximum context window constraint, current methods use keyframe sampling which often misses both macro-level events and micro-level details due to the sparse temporal coverage. Furthermore, processing full images and their tokens for each frame incurs substantial computational overhead. We address these limitations by leveraging video codec primitives (specifically motion vectors and residuals) which natively encode video redundancy and sparsity without requiring expensive full-image encoding for most frames. To this end, we introduce lightweight transformer-based encoders that aggregate codec primitives and align their representations with image encoder embeddings through a pre-training strategy that accelerates convergence during end-to-end fine-tuning. Our approach, CoPE-VideoLM, reduces the time-to-first-token by up to 86% and token usage by up to 93% compared to standard VideoLMs. Moreover, by varying the keyframe and codec primitive densities we maintain or exceed performance on 14 diverse video understanding benchmarks spanning general question answering, temporal and motion reasoning, long-form understanding, and spatial scene understanding.
>
---
#### [replaced 022] Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs: GPT, Gemini, and LLaMA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究交互语气对大型语言模型性能的影响，通过实验分析不同语气下的模型表现，为实际应用提供指导。**

- **链接: [https://arxiv.org/pdf/2512.12812](https://arxiv.org/pdf/2512.12812)**

> **作者:** Hanyu Cai; Binqi Shen; Lier Jin; Lan Hu; Xiaojing Fan
>
> **摘要:** Prompt engineering has emerged as a critical factor influencing large language model (LLM) performance, yet the impact of pragmatic elements such as linguistic tone and politeness remains underexplored, particularly across different model families. In this work, we propose a systematic evaluation framework to examine how interaction tone affects model accuracy and apply it to three recently released and widely available LLMs: GPT-4o mini (OpenAI), Gemini 2.0 Flash (Google DeepMind), and Llama 4 Scout (Meta). Using the MMMLU benchmark, we evaluate model performance under Very Polite, Neutral, and Very Rude prompt variants across six tasks spanning STEM and Humanities domains, and analyze pairwise accuracy differences with statistical significance testing. Our results show that tone sensitivity is both model-dependent and domain-specific. Neutral or Very Polite prompts generally yield higher accuracy than Very Rude prompts, but statistically significant effects appear only in a subset of Humanities tasks, where rude tone reduces accuracy for GPT and Llama, while Gemini remains comparatively tone-insensitive. When performance is aggregated across tasks within each domain, tone effects diminish and largely lose statistical significance. Compared with earlier research, these findings suggest that dataset scale and coverage materially influence the detection of tone effects. Overall, our study indicates that while interaction tone can matter in specific interpretive settings, modern LLMs are broadly robust to tonal variation in typical mixed-domain use, providing practical guidance for prompt design and model selection in real-world deployments.
>
---
#### [replaced 023] POTSA: A Cross-Lingual Speech Alignment Framework for Speech-to-Text Translation
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，旨在解决多语言翻译中的语义偏差问题。提出POTSA框架，通过跨语言对齐和最优传输技术提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.09232](https://arxiv.org/pdf/2511.09232)**

> **作者:** Xuanchen Li; Chenrui Cui; Tianrui Wang; Meng Ge; Zikang Huang; Jin Li; Yizhou Peng; Yuheng Lu; Nyima Tashi; Longbiao Wang; Jianwu Dang
>
> **摘要:** Speech Large Language Models have achieved breakthroughs in multilingual speech-to-text translation. However, existing approaches often overlook semantic commonalities across source languages, leading to biased translation performance. In this work, we propose POTSA (Parallel Optimal Transport for Speech Alignment), a new framework based on cross-lingual parallel speech pairs and Optimal Transport, designed to bridge high- and low-resource translation gaps. First, we introduce a Bias Compensation module to coarsely align initial speech representations. Second, we impose token-level OT constraints on a Q-Former using parallel pairs to establish fine-grained representation consistency. Then, we apply a layer scheduling strategy to focus OT constraints on semantically beneficial layers. Experiments on FLEURS show our method achieves SOTA performance, with +1.29 BLEU over five common languages and +2.93 BLEU on zero-shot languages, using only 10 hours of parallel speech per language.
>
---
#### [replaced 024] Beyond In-Distribution Success: Scaling Curves of CoT Granularity for Language Model Generalization
- **分类: cs.CL**

- **简介: 该论文研究语言模型在分布偏移下的泛化能力，解决复杂任务中的OOD泛化问题。通过分析CoT数据粒度与样本效率，揭示CoT提升泛化性能的机制。**

- **链接: [https://arxiv.org/pdf/2502.18273](https://arxiv.org/pdf/2502.18273)**

> **作者:** Ru Wang; Wei Huang; Selena Song; Haoyu Zhang; Qian Niu; Yusuke Iwasawa; Yutaka Matsuo; Jiaxian Guo
>
> **备注:** Accepted at the Conference on Parsimony and Learning (CPAL) 2026
>
> **摘要:** Generalization to novel compound tasks under distribution shift is important for deploying transformer-based language models (LMs). This work investigates Chain-of-Thought (CoT) reasoning as a means to enhance OOD generalization. Through controlled experiments across several compound tasks, we reveal three key insights: (1) While QA-trained models achieve near-perfect in-distribution accuracy, their OOD performance degrades catastrophically, even with 10000k+ training examples; (2) the granularity of CoT data strongly correlates with generalization performance; finer-grained CoT data leads to better generalization; (3) CoT exhibits remarkable sample efficiency, matching QA performance with much less (even 80%) data. Theoretically, we demonstrate that compound tasks inherently permit shortcuts in Q-A data that misalign with true reasoning principles, while CoT forces internalization of valid dependency structures, and thus can achieve better generalization. Further, we show that transformer positional embeddings can amplify generalization by emphasizing subtask condition recurrence in long CoT sequences. Our combined theoretical and empirical analysis provides compelling evidence for CoT reasoning as a crucial training paradigm for enabling LM generalization under real-world distributional shifts for compound tasks.
>
---
#### [replaced 025] FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG
- **分类: cs.CL**

- **简介: 该论文属于RAG系统中的引用幻觉检测任务，旨在解决模型错误引用问题。提出FACTUM框架，通过四个机制评分来评估引用可信度。**

- **链接: [https://arxiv.org/pdf/2601.05866](https://arxiv.org/pdf/2601.05866)**

> **作者:** Maxime Dassen; Rebecca Kotula; Kenton Murray; Andrew Yates; Dawn Lawrie; Efsun Kayi; James Mayfield; Kevin Duh
>
> **备注:** Accepted at ECIR 2026. 13 pages, 2 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) models are critically undermined by citation hallucinations, a deceptive failure where a model cites a source that fails to support its claim. While existing work attributes hallucination to a simple over-reliance on parametric knowledge, we reframe this failure as an evolving, scale-dependent coordination failure between the Attention (reading) and Feed-Forward Network (recalling) pathways. We introduce FACTUM (Framework for Attesting Citation Trustworthiness via Underlying Mechanisms), a framework of four mechanistic scores: Contextual Alignment (CAS), Attention Sink Usage (BAS), Parametric Force (PFS), and Pathway Alignment (PAS). Our analysis reveals that correct citations are consistently marked by higher parametric force (PFS) and greater use of the attention sink (BAS) for information synthesis. Crucially, we find that "one-size-fits-all" theories are insufficient as the signature of correctness evolves with scale: while the 3B model relies on high pathway alignment (PAS), our best-performing 8B detector identifies a shift toward a specialized strategy where pathways provide distinct, orthogonal information. By capturing this complex interplay, FACTUM outperforms state-of-the-art baselines by up to 37.5% in AUC. Our results demonstrate that high parametric force is constructive when successfully coordinated with the Attention pathway, paving the way for more nuanced and reliable RAG systems.
>
---
#### [replaced 026] HEAD-QA v2: Expanding a Healthcare Benchmark for Reasoning
- **分类: cs.CL**

- **简介: 该论文介绍HEAD-QA v2，一个扩展的医疗推理多选题数据集，用于提升生物医学推理研究。任务是构建高质量医疗问答数据集，解决语言和概念复杂性问题，通过扩展数据和实验验证模型表现。**

- **链接: [https://arxiv.org/pdf/2511.15355](https://arxiv.org/pdf/2511.15355)**

> **作者:** Alexis Correa-Guillén; Carlos Gómez-Rodríguez; David Vilares
>
> **备注:** LREC 2026 camera-ready version
>
> **摘要:** We introduce HEAD-QA v2, an expanded and updated version of a Spanish/English healthcare multiple-choice reasoning dataset originally released by Vilares and Gómez-Rodríguez (2019). The update responds to the growing need for high-quality datasets that capture the linguistic and conceptual complexity of healthcare reasoning. We extend the dataset to over 12,000 questions from ten years of Spanish professional exams, benchmark several open-source LLMs using prompting, RAG, and probability-based answer selection, and provide additional multilingual versions to support future work. Results indicate that performance is mainly driven by model scale and intrinsic reasoning ability, with complex inference strategies obtaining limited gains. Together, these results establish HEAD-QA v2 as a reliable resource for advancing research on biomedical reasoning and model improvement.
>
---
#### [replaced 027] Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于模型优化任务，指出Chinchilla Approach 2在拟合神经网络缩放定律时存在系统性偏差，提出改进方法Chinchilla Approach 3并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.22339](https://arxiv.org/pdf/2603.22339)**

> **作者:** Eric Czech; Zhiwei Xu; Yael Elmatad; Yixin Wang; William Held
>
> **摘要:** Chinchilla Approach 2 is among the most widely used methods for fitting neural scaling laws. Its parabolic approximation introduces systematic biases in compute-optimal allocation estimates, even on noise-free synthetic data. Applied to published Llama 3 IsoFLOP data at open frontier compute scales, these biases imply a parameter underallocation corresponding to 6.5% of the $3.8\times10^{25}$ FLOP training budget and \$1.4M (90% CI: \$412K-\$2.9M) in unnecessary compute at 50% H100 MFU. Simulated multimodal model misallocations show even greater opportunity costs due to higher loss surface asymmetry. Three sources of this error are examined: IsoFLOP sampling grid width (Taylor approximation accuracy), uncentered IsoFLOP sampling, and loss surface asymmetry ($\alpha \neq \beta$). Chinchilla Approach 3 largely eliminates these biases but is often regarded as less data-efficient, numerically unstable, prone to local minima, and harder to implement. Each concern is shown to be unfounded or addressable, especially when the partially linear structure of the objective is exploited via Variable Projection, enabling unbiased inference on all five loss surface parameters through a two-dimensional optimization that is well-conditioned, analytically differentiable, and amenable to dense, or even exhaustive, grid search. It may serve as a more convenient replacement for Approach 2 or a more scalable alternative for adaptations of Approach 3 to richer scaling law formulations. See this https URL for details and this https URL for other results from this study.
>
---
#### [replaced 028] Beyond Elicitation: Provision-based Prompt Optimization for Knowledge-Intensive Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型任务，解决传统提示优化方法在知识获取上的不足。提出KPPO框架，通过系统化知识整合提升模型性能，降低token消耗。**

- **链接: [https://arxiv.org/pdf/2511.10465](https://arxiv.org/pdf/2511.10465)**

> **作者:** Yunzhe Xu; Zhuosheng Zhang; Zhe Liu
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** While prompt optimization has emerged as a critical technique for enhancing language model performance, existing approaches primarily focus on elicitation-based strategies that search for optimal prompts to activate models' capabilities. These methods exhibit fundamental limitations when addressing knowledge-intensive tasks, as they operate within static knowledge capacity rather than providing the factual knowledge, terminology precision, and reasoning patterns required in specialized domains. To address these limitations, we propose Knowledge-Provision-based Prompt Optimization (KPPO), a framework that reformulates prompt optimization as systematic knowledge integration rather than potential elicitation. KPPO introduces three key innovations: 1) a knowledge gap filling mechanism for knowledge gap identification and targeted remediation; 2) a batch-wise candidate evaluation approach that considers both performance improvement and distributional stability; 3) an adaptive knowledge pruning strategy that balances performance and token efficiency, reducing up to 29% of inference token usage. Evaluation on 15 knowledge-intensive benchmarks from various domains demonstrates KPPO's superiority over elicitation-based methods, with an average improvement of ~6% over baselines while achieving comparable or lower token consumption.
>
---
#### [replaced 029] ViPRA: Video Prediction for Robot Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出ViPRA，解决机器人控制中缺乏标注动作的问题。通过视频预测和隐式动作表示，实现无需大量标注的连续控制，提升泛化能力和控制频率。**

- **链接: [https://arxiv.org/pdf/2511.07732](https://arxiv.org/pdf/2511.07732)**

> **作者:** Sandeep Routray; Hengkai Pan; Unnat Jain; Shikhar Bahl; Deepak Pathak
>
> **备注:** In ICLR 2026. Website: this https URL
>
> **摘要:** Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, ViPRA explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We have released models and code at this https URL
>
---
#### [replaced 030] Link Prediction for Event Logs in the Process Industry
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识管理任务，旨在解决过程工业中事件日志碎片化问题，通过构建记录链接模型提升数据连通性。**

- **链接: [https://arxiv.org/pdf/2508.09096](https://arxiv.org/pdf/2508.09096)**

> **作者:** Anastasia Zhukova; Thomas Walton; Christian E. Lobmüller; Bela Gipp
>
> **备注:** accepted to RESOURCEFUL 2026, co-located with LREC 2026
>
> **摘要:** In the era of graph-based retrieval-augmented generation (RAG), link prediction is a significant preprocessing step for improving the quality of fragmented or incomplete domain-specific data for the graph retrieval. Knowledge management in the process industry uses RAG-based applications to optimize operations, ensure safety, and facilitate continuous improvement by effectively leveraging operational data and past insights. A key challenge in this domain is the fragmented nature of event logs in shift books, where related records are often kept separate, even though they belong to a single event or process. This fragmentation hinders the recommendation of previously implemented solutions to users, which is crucial in the timely problem-solving at live production sites. To address this problem, we develop a record linking model, which we define as a cross-document coreference resolution (CDCR) task. Record linking adapts the task definition of CDCR and combines two state-of-the-art CDCR models with the principles of natural language inference (NLI) and semantic text similarity (STS) to perform link prediction. The evaluation shows that our record linking model outperformed the best versions of our baselines, i.e., NLP and STS, by 28% (11.43 p) and 27.4% (11.21 p), respectively. Our work demonstrates that common NLP tasks can be combined and adapted to a domain-specific setting of the German process industry, improving data quality and connectivity in shift logs.
>
---
#### [replaced 031] The Rise of AfricaNLP: Contributions, Contributors, Community Impact, and Bibliometric Analysis
- **分类: cs.CL**

- **简介: 该论文属于NLP领域研究，分析非洲NLP的发展历程、贡献及影响，通过数据和工具追踪研究趋势。**

- **链接: [https://arxiv.org/pdf/2509.25477](https://arxiv.org/pdf/2509.25477)**

> **作者:** Tadesse Destaw Belay; Kedir Yassin Hussen; Sukairaj Hafiz Imam; Ibrahim Said Ahmad; Isa Inuwa-Dutse; Abrham Belete Haile; Grigori Sidorov; Eusebio Ricardez Vazquez; Iqra Ameer; Idris Abdulmumin; Tajuddeen Gwadabe; Vukosi Marivate; Seid Muhie Yimam; Shamsuddeen Hassan Muhammad
>
> **摘要:** Natural Language Processing (NLP) is undergoing constant transformation, as Large Language Models (LLMs) are driving daily breakthroughs in research and practice. In this regard, tracking the progress of NLP research and automatically analyzing the contributions of research papers provides key insights into the nature of the field and the researchers. This study explores the progress of African NLP (AfricaNLP) by asking (and answering) research questions about the progress of AfricaNLP (publications, NLP topics, and NLP tasks), contributions (data, method, and task), and contributors (authors, affiliated institutions, and funding bodies). We quantitatively examine two decades (2005 - 2025) of contributions to AfricaNLP research, using a dataset of 2.2K NLP papers, 4.9K contributing authors, and 7.8K human-annotated contribution sentences (AfricaNLPContributions), along with benchmark results. Our dataset and AfricaNLP research explorer tool will provide a powerful lens for tracing AfricaNLP research trends and holds potential for generating data-driven research approaches.
>
---
#### [replaced 032] Rethinking Attention Output Projection: Structured Hadamard Transforms for Efficient Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer模型中注意力输出投影的高成本问题。通过引入结构化Hadamard变换，减少参数量并提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.08343](https://arxiv.org/pdf/2603.08343)**

> **作者:** Shubham Aggarwal; Lokendra Kumar
>
> **备注:** 10 pages, 9 figures, 4 tables
>
> **摘要:** The dense output projection in multi head attention scales quadratically with model dimension, contributing significantly to parameter count, memory footprint, and inference cost. We propose replacing this projection with a fixed, parameter free Walsh Hadamard Transform (WHT) followed by a diagonal affine transformation. This approach eliminates approximately 25 percent of attention parameters per block while maintaining global cross-head interaction through an orthogonal, norm-preserving transformation. Our results demonstrate that WHT augmented models exhibit a steeper validation loss curve relative to training FLOPs compared to dense baselines, suggesting superior compute utilization during training. Crucially, we show that efficiency gains including reduced memory footprint and increased throughput grow monotonically with model size, batch size, and sequence length. We evaluate performance across both prefill and decoding stages, finding that the structured transform consistently outperforms dense projections as complexity increases. Our findings indicate that replacing dense projections with structured transforms allows for more compute-efficient architectures that achieve lower loss than dense models at an equivalent training budget.
>
---
#### [replaced 033] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决多语言和长文本评估的可复现性问题。构建了Open ASR Leaderboard平台，对比多种系统，标准化评估指标，促进透明化研究。**

- **链接: [https://arxiv.org/pdf/2510.06961](https://arxiv.org/pdf/2510.06961)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Nithin Rao Koluguri; Piotr Żelasko; Somshubra Majumdar; Adel Moumen; Sanchit Gandhi
>
> **备注:** Leaderboard: this https URL ; Code: this https URL
>
> **摘要:** We present the Open ASR Leaderboard, a reproducible benchmarking platform with community contributions from academia and industry. It compares 86 open-source and proprietary systems across 12 datasets, with English short- and long-form and multilingual short-form tracks. We standardize word error rate (WER) and inverse real-time factor (RTFx) evaluation for consistent accuracy-efficiency comparisons across model architectures and toolkits (e.g., ESPNet, NeMo, SpeechBrain, Transformers). We observe that Conformer-based encoders paired with transformer-based decoders achieve the best average WER, while connectionist temporal classification (CTC) and token-and-duration transducer (TDT) decoders offer superior RTFx, making them better suited for long-form and batched processing. All code and dataset loaders are open-sourced to support transparent, extensible evaluation. We present our evaluation methodology to facilitate community-driven benchmarking in ASR and other tasks.
>
---
#### [replaced 034] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康自然语言处理任务，旨在解决认知扭曲自动检测中的语境模糊问题。通过结合大语言模型与多实例学习，提升检测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2509.17292](https://arxiv.org/pdf/2509.17292)**

> **作者:** Jun Seo Kim; Hyemi Kim; Woo Joo Oh; Hongjin Cho; Hochul Lee; Hye Hyeon Kim
>
> **摘要:** Cognitive distortions have been closely linked to mental health disorders, yet their automatic detection remained challenging due to contextual ambiguity, co-occurrence, and semantic overlap. We proposed a novel framework that combines Large Language Models (LLMs) with Multiple-Instance Learning (MIL) architecture to enhance interpretability and expression-level reasoning. Each utterance was decomposed into Emotion, Logic, and Behavior (ELB) components, which were processed by LLMs to infer multiple distortion instances, each with a predicted type, expression, and model-assigned salience score. These instances were integrated via a Multi-View Gated Attention mechanism for final classification. Experiments on Korean (KoACD) and English (Therapist QA) datasets demonstrate that incorporating ELB and LLM-inferred salience scores improves classification performance, especially for distortions with high interpretive ambiguity. Our results suggested a psychologically grounded and generalizable approach for fine-grained reasoning in mental health NLP.
>
---
#### [replaced 035] GHTM: A Graph-based Hybrid Topic Modeling Approach with a Benchmark Dataset for the Low-Resource Bengali Language
- **分类: cs.CL**

- **简介: 该论文属于主题建模任务，旨在解决低资源语言 Bengali 的主题建模问题。提出 GHTM 模型并构建 NCTBText 数据集，提升主题一致性与多样性。**

- **链接: [https://arxiv.org/pdf/2508.00605](https://arxiv.org/pdf/2508.00605)**

> **作者:** Farhana Haque; Md. Abdur Rahman; Sumon Ahmed
>
> **摘要:** Topic modeling is a Natural Language Processing (NLP) technique used to discover latent themes and abstract topics from text corpora by grouping co-occurring keywords. Although widely researched in English, topic modeling remains understudied in Bengali due to a lack of adequate resources and initiatives. Existing Bengali topic modeling research lacks standardized evaluation frameworks with comprehensive baselines and diverse datasets, exploration of modern methodological approaches, and reproducible implementations, with only three Bengali-specific architectures proposed to date. To address these gaps, this study presents a comprehensive evaluation of traditional and contemporary topic modeling approaches across three Bengali datasets and introduces GHTM (Graph-based Hybrid Topic Model), a novel architecture that strategically integrates TF-IDF-weighted GloVe embeddings, Graph Convolutional Networks (GCN), and Non-negative Matrix Factorization (NMF). GHTM represents text documents using hybrid TF-IDF-weighted GloVe embeddings. It builds a document-similarity graph and leverages GCN to refine the representations through neighborhood aggregation. Then, it finally decomposes the refined representations using NMF to extract interpretable topics. Experimental results demonstrate that GHTM achieves superior topic coherence (NPMI: 0.27-0.28) and diversity compared to existing methods while maintaining computational efficiency across datasets of varying scales. The model also demonstrates strong cross-lingual generalization, outperforming established graph-based models on the English 20Newsgroups benchmark. Additionally, we introduce NCTBText, a diverse Bengali textbook-based dataset comprising 8,650 text documents, curated from eight subject areas, providing much-needed topical diversity beyond newspaper-centric Bengali corpora and serving as a benchmark for future research.
>
---
#### [replaced 036] Complete asymptotic type-token relationship for growing complex systems with inverse power-law count rankings
- **分类: physics.soc-ph; cs.CL**

- **简介: 该论文研究复杂系统中类型与实例的统计关系，解决Zipf定律与Heaps定律的关联问题。通过构建模型，推导出统一的渐近表达式，揭示类型-实例关系由Zipf定律决定。**

- **链接: [https://arxiv.org/pdf/2511.02069](https://arxiv.org/pdf/2511.02069)**

> **作者:** Pablo Rosillo-Rodes; Laurent Hébert-Dufresne; Peter Sheridan Dodds
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** The growth dynamics of complex systems often exhibit statistical regularities involving power-law relationships. For real finite complex systems formed by countable tokens (animals, words) as instances of distinct types (species, dictionary entries), an inverse power-law scaling $S \sim r^{-\alpha}$ between type count $S$ and type rank $r$, widely known as Zipf's law, is widely observed to varying degrees of fidelity. A secondary, summary relationship is Heaps' law, which states that the number of types scales sublinearly with the total number of observed tokens present in a growing system. Here, we propose an idealized model of a growing system that (1) deterministically produces arbitrary inverse power-law count rankings for types, and (2) allows us to determine the exact asymptotics of the type-token relationship. Our argument improves upon and remedies earlier work. We obtain a unified asymptotic expression for all values of $\alpha$, which corrects the special cases of $\alpha = 1$ and $\alpha \gg 1$. Our approach relies solely on the form of count rankings, avoids unnecessary approximations, and does not involve any stochastic mechanisms or sampling processes. We thereby demonstrate that a general type-token relationship arises solely as a consequence of Zipf's law.
>
---
#### [replaced 037] Towards Hyper-Efficient RAG Systems in VecDBs: Distributed Parallel Multi-Resolution Vector Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG系统优化任务，旨在解决VecDB检索效率与语义精度的平衡问题。提出SPI框架，通过多分辨率索引实现动态查询适配，提升检索速度和内存效率。**

- **链接: [https://arxiv.org/pdf/2511.16681](https://arxiv.org/pdf/2511.16681)**

> **作者:** Dong Liu; Yanxuan Yu
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems have become a dominant approach to augment large language models (LLMs) with external knowledge. However, existing vector database (VecDB) retrieval pipelines rely on flat or single-resolution indexing structures, which cannot adapt to the varying semantic granularity required by diverse user queries. This limitation leads to suboptimal trade-offs between retrieval speed and contextual relevance. To address this, we propose \textbf{Semantic Pyramid Indexing (SPI)}, a novel multi-resolution vector indexing framework that introduces query-adaptive resolution control for RAG in VecDBs. Unlike existing hierarchical methods that require offline tuning or separate model training, SPI constructs a semantic pyramid over document embeddings and dynamically selects the optimal resolution level per query through a lightweight classifier. This adaptive approach enables progressive retrieval from coarse-to-fine representations, significantly accelerating search while maintaining semantic coverage. We implement SPI as a plugin for both FAISS and Qdrant backends and evaluate it across multiple RAG tasks including MS MARCO, Natural Questions, and multimodal retrieval benchmarks. SPI achieves up to \textbf{5.7$\times$} retrieval speedup and \textbf{1.8$\times$} memory efficiency gain while improving end-to-end QA F1 scores by up to \textbf{2.5 points} compared to strong baselines. Our theoretical analysis provides guarantees on retrieval quality and latency bounds, while extensive ablation studies validate the contribution of each component. The framework's compatibility with existing VecDB infrastructures makes it readily deployable in production RAG systems. Code is availabe at \href{this https URL}{this https URL\_VecDB}.
>
---
#### [replaced 038] LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言缺乏高质量训练数据的问题。通过合成 Luxembourgish 指令数据集 LuxIT，提升语言模型性能。**

- **链接: [https://arxiv.org/pdf/2510.24434](https://arxiv.org/pdf/2510.24434)**

> **作者:** Julian Valline; Cedric Lothritz; Siwen Guo; Jordi Cabot
>
> **摘要:** The effectiveness of instruction-tuned Large Language Models (LLMs) is often limited in low-resource linguistic settings due to a lack of high-quality training data. We introduce LuxIT, a novel, monolingual instruction tuning dataset for Luxembourgish developed to mitigate this challenge. We synthesize the dataset from a corpus of native Luxembourgish texts, utilizing DeepSeek-R1-0528, chosen for its shown proficiency in Luxembourgish. Following generation, we apply a quality assurance process, employing an LLM-as-a-judge approach, retaining 227,507 high-quality instruction-answer pairs. To investigate the practical utility of the dataset, we fine-tune 14 smaller-scale LLMs ($\leq$15B parameters) on LuxIT and evaluate them on standardized Luxembourgish proficiency exams and five downstream NLP tasks. Training on LuxIT yields a mean accuracy change of +5.37 percentage points on language exams across all 14 models, with 12 of 14 showing improvement. On NLP downstream tasks, 9 of 14 models improve in macro-averaged F1, though gains on the two benchmarks do not systematically correlate. These results underscore the feasibility of leveraging monolingual synthetic data to improve LLM capabilities in low-resource languages, while highlighting the multi-faceted nature of language proficiency.
>
---
#### [replaced 039] LLMs versus the Halting Problem: Revisiting Program Termination Prediction
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 该论文属于程序终止性预测任务，探讨LLMs能否可靠预测程序是否终止。研究评估了LLMs在SV-Comp 2025数据集上的表现，发现其效果良好但存在局限。**

- **链接: [https://arxiv.org/pdf/2601.18987](https://arxiv.org/pdf/2601.18987)**

> **作者:** Oren Sultan; Jordi Armengol-Estape; Pascal Kesseli; Julien Vanegue; Dafna Shahaf; Yossi Adi; Peter O'Hearn
>
> **摘要:** Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length and complexity increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
>
---
#### [replaced 040] AirQA: A Comprehensive QA Dataset for AI Research with Instance-Level Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AirQA数据集，用于AI研究中的问答任务，解决缺乏高质量基准的问题。通过ExTrActor框架生成训练数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.16952](https://arxiv.org/pdf/2509.16952)**

> **作者:** Tiancheng Huang; Ruisheng Cao; Yuxin Zhang; Zhangyi Kang; Zijian Wang; Chenrun Wang; Yijie Luo; Hang Zheng; Lirong Qian; Lu Chen; Kai Yu
>
> **备注:** 29 page, 6 figures, 17 tables, accepted to ICLR 2026
>
> **摘要:** The growing volume of academic papers has made it increasingly difficult for researchers to efficiently extract key information. While large language models (LLMs) based agents are capable of automating question answering (QA) workflows for scientific papers, there still lacks a comprehensive and realistic benchmark to evaluate their capabilities. Moreover, training an interactive agent for this specific task is hindered by the shortage of high-quality interaction trajectories. In this work, we propose AirQA, a human-annotated comprehensive paper QA dataset in the field of artificial intelligence (AI), with 13,956 papers and 1,246 questions, that encompasses multi-task, multi-modal and instance-level evaluation. Furthermore, we propose ExTrActor, an automated framework for instruction data synthesis. With three LLM-based agents, ExTrActor can perform example generation and trajectory collection without human intervention. Evaluations of multiple open-source and proprietary models show that most models underperform on AirQA, demonstrating the quality of our dataset. Extensive experiments confirm that ExTrActor consistently improves the multi-turn tool-use capability of small models, enabling them to achieve performance comparable to larger ones.
>
---
#### [replaced 041] Continual Robot Skill and Task Learning via Dialogue
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于人机交互中的持续学习任务，旨在解决机器人高效学习新技能与任务的问题。通过对话交互获取人类指导，提出ACT-LoRA模型实现少量样本下的持续学习。**

- **链接: [https://arxiv.org/pdf/2409.03166](https://arxiv.org/pdf/2409.03166)**

> **作者:** Weiwei Gu; Suresh Kondepudi; Anmol Gupta; Lixiao Huang; Nakul Gopalan
>
> **摘要:** Interactive robot learning is a challenging problem as the robot is present with human users who expect the robot to learn novel skills to solve novel tasks perpetually with sample efficiency. In this work we present a framework for robots to continually learn tasks and visuo-motor skills and query for novel skills via dialog interactions with human users. Our robot agent maintains a skill library, and uses an existing LLM to perform grounded dialog interactions to query unknown skills from real human users. We developed a novel visual-motor control policy Action Chunking Transformer with Low Rank Adaptation (ACT-LoRA) that can continually learn novel skills using only a few demonstrations which is critical in human-robot interaction scenarios. The paper has twin goals: Firstly to demonstrate better continual learning in simulation; and secondly, to demonstrate the use of our dialog based learning framework in a realistic human-robot interaction use case. Our ACT-LoRA policy consistently outperforms a GMM-LoRA baseline on multiple continual learning simulation benchmarks by achieving > 300% improvements on novel skills, while achieving comparable performance in existing skills. Moreover, with our IRB approved human-subjects study we demonstrate that our dialog based continual learning framework allows users to teach robots cooking skills successfully (100%) while spending a higher ratio of time on finishing an auxiliary distraction tasks in the test phase of the study compared to a non-learning language based agent (p < 0.001).
>
---
#### [replaced 042] Evaluating Latent Knowledge of Public Tabular Datasets in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决大语言模型在表格数据上的数据污染问题。通过生成可控查询和统计测试，检测数据泄露，发现部分数据存在污染。**

- **链接: [https://arxiv.org/pdf/2510.20351](https://arxiv.org/pdf/2510.20351)**

> **作者:** Matteo Silvestri; Fabiano Veglianti; Flavio Giorgi; Fabrizio Silvestri; Gabriele Tolomei
>
> **摘要:** Large language models (LLMs) are increasingly exposed to data contamination, i.e., performance gains driven by prior exposure of test datasets rather than generalization. However, in the context of tabular data, this problem is largely unexplored. Existing approaches primarily rely on memorization tests, which are too coarse to detect contamination. In contrast, we propose a framework for assessing contamination in tabular datasets by generating controlled queries and performing comparative evaluation. Given a dataset, we craft multiple-choice aligned queries that preserve task structure while allowing systematic transformations of the underlying data. These transformations are designed to selectively disrupt dataset information while preserving partial knowledge, enabling us to isolate performance attributable to contamination. We complement this setup with non-neural baselines that provide reference performance, and we introduce a statistical testing procedure to formally detect significant deviations indicative of contamination. Empirical results on eight widely used tabular datasets reveal clear evidence of contamination in four cases. These findings suggest that performance on downstream tasks involving such datasets may be substantially inflated, raising concerns about the reliability of current evaluation practices.
>
---
#### [replaced 043] FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FigEx2，解决科学复合图无 caption 或 captions 短的问题，通过视觉条件检测面板并生成描述，提升图文本对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.08026](https://arxiv.org/pdf/2601.08026)**

> **作者:** Jifeng Song; Arun Das; Pan Wang; Hui Ji; Kun Zhao; Yufei Huang
>
> **摘要:** Scientific compound figures combine multiple labeled panels into a single image. However, in a PMC-scale crawl of 346,567 compound figures, 16.3% have no caption and 1.8% only have captions shorter than ten words, causing them to be discarded by existing caption-decomposition pipelines. We propose FigEx2, a visual-conditioned framework that localizes panels and generates panel-wise captions directly from the image, converting otherwise unusable figures into aligned panel-text pairs for downstream pretraining and retrieval. To mitigate linguistic variance in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively controls how caption features condition the detection query space, and employ a staged SFT+RL strategy with CLIP-based alignment and BERTScore-based semantic rewards. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. FigEx2 achieves 0.728 mAP@0.5:0.95 for detection, outperforms Qwen3-VL-8B by 0.44 in METEOR and 0.22 in BERTScore, and transfers zero-shot to out-of-distribution scientific domains without fine-tuning.
>
---
#### [replaced 044] AISAC: An Integrated multi-agent System for Transparent, Retrieval-Grounded Scientific Assistance
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出AISAC，一个用于科学推理的透明多智能体系统，解决科学任务中AI助手的可解释性与可控性问题，通过结构化管理和工具交互实现可靠科学辅助。**

- **链接: [https://arxiv.org/pdf/2511.14043](https://arxiv.org/pdf/2511.14043)**

> **作者:** Chandrachur Bhattacharya; Sibendu Som
>
> **摘要:** AI Scientific Assistant Core (AISAC) is a transparent, modular multi-agent runtime developed at Argonne National Laboratory to support long-horizon, evidence-grounded scientific reasoning. Rather than proposing new agent algorithms or claiming autonomous scientific discovery, AISAC contributes a governed execution substrate that operationalizes key requirements for deploying agentic AI in scientific practice, including explicit role semantics, budgeted context management, traceable execution, and reproducible interaction with tools and knowledge. AISAC enforces four structural guarantees for scientific reasoning: (1) declarative agent registration with runtime-enforced role semantics and automatic system prompt generation; (2) budgeted orchestration via explicit per-turn context and delegation depth limits; (3) role-aligned memory access across episodic, dialogue, and evidence layers; and (4) trace-driven transparency through persistent execution records and a live event-stream interface. These guarantees are implemented through hybrid persistent memory (SQLite and dual FAISS indices), governed retrieval with agent-scoped RAG, structured tool execution with schema validation, and a configuration-driven bootstrap mechanism that enables project specific extension without modifying the shared core. AISAC is currently deployed across multiple scientific workflows at Argonne, including combustion science, materials research, and energy process safety, demonstrating its use as a reusable substrate for domain-specialized AI scientific assistants.
>
---
#### [replaced 045] OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，旨在解决低资源下长音频视频问答的编码成本高、检索弱等问题。提出OmniRAG-Agent，结合检索增强生成与代理循环，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2602.03707](https://arxiv.org/pdf/2602.03707)**

> **作者:** Yifan Zhu; Xinyu Mu; Tao Feng; Zhonghong Ou; Yuning Gong; Haoran Luo
>
> **摘要:** Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end optimization. To address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component.
>
---
#### [replaced 046] BanglaSocialBench: A Benchmark for Evaluating Sociopragmatic and Cultural Alignment of LLMs in Bangladeshi Social Interaction
- **分类: cs.CL**

- **简介: 该论文提出BanglaSocialBench，评估大语言模型在孟加拉社会互动中的社会语用和文化契合度。任务是解决文化敏感性不足的问题，通过三个领域测试模型表现。**

- **链接: [https://arxiv.org/pdf/2603.15949](https://arxiv.org/pdf/2603.15949)**

> **作者:** Tanvir Ahmed Sijan; S. M Golam Rifat; Pankaj Chowdhury Partha; Md. Tanjeed Islam; Md. Musfique Anwar
>
> **备注:** Under Review
>
> **摘要:** Large Language Models have demonstrated strong multilingual fluency, yet fluency alone does not guarantee socially appropriate language use. In high-context languages, communicative competence requires sensitivity to social hierarchy, relational roles, and interactional norms that are encoded directly in everyday language. Bangla exemplifies this challenge through its three-tiered pronominal system, kinship-based addressing, and culturally embedded social customs. We introduce BanglaSocialBench, the first benchmark designed to evaluate sociopragmatic competence in Bangla through context-dependent language use rather than factual recall. The benchmark spans three domains: Bangla Address Terms, Kinship Reasoning, and Social Customs, and consists of 1,719 culturally grounded instances written and verified by native Bangla speakers. We evaluate twelve contemporary LLMs in a zero-shot setting and observe systematic patterns of cultural misalignment. Models frequently default to overly formal address forms, fail to recognize multiple socially acceptable address pronouns, and conflate kinship terminology across religious contexts. Our findings show that sociopragmatic failures are often structured and non-random, revealing persistent limitations in how current LLMs infer and apply culturally appropriate language use in realistic Bangladeshi social interactions.
>
---
#### [replaced 047] Compounding Disadvantage: Auditing Intersectional Bias in LLM-Generated Explanations Across Indian and American STEM Education
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI偏见审计任务，旨在检测LLM在STEM教育中生成解释时的交叉性偏差。通过构建合成学生档案，分析不同背景下的输出质量差异，揭示系统性不平等。**

- **链接: [https://arxiv.org/pdf/2601.14506](https://arxiv.org/pdf/2601.14506)**

> **作者:** Amogh Gupta; Niharika Patil; Sourojit Ghosh; SnehalKumar; S Gaikwad
>
> **摘要:** Large Language Models (LLMs) are rapidly being adopted by STEM-focused educational institutions and students worldwide. They generate personalized instructions, explanations, and provide feedback on demand. However, these systems tailor instruction to demographic signals rather than demonstrated ability. In such cases, personalization becomes a mechanism of inequality. We conduct one of the first large-scale intersectional audits of LLM-generated STEM educational content, constructing synthetic student profiles. We combine dimensions specific to Indian education (caste, medium of instruction, college tier) and American education (race, HBCU attendance, school type), alongside shared dimensions of income, gender, and disability. We audit four LLMs (Qwen 2.5-32B-Instruct, GPT-4o, GPT-4o-mini, GPT-OSS 20B) across ranking and generation tasks on two STEM datasets, evaluating outputs with FDR-corrected significance testing and SHAP feature attribution. Across both cultural contexts, marginalized profiles receive lower-quality outputs. Income is the most pervasive bias, producing significant effects across every model and context. Disability triggers simpler explanations. Intersectional analysis reveals non-additive compounding: the gap between the most privileged and most marginalized profiles reaches 2.55 grade levels. These biases persist even when marginalized students attend elite institutions. All four models converge on similar patterns. These findings carry direct design and policy implications for incorporating AI into global STEM education.
>
---
#### [replaced 048] FGTR: Fine-Grained Multi-Table Retrieval via Hierarchical LLM Reasoning
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于表格检索任务，解决多表查询中精度低、效率差的问题。提出FGTR方法，通过分层推理实现细粒度多表检索，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.12702](https://arxiv.org/pdf/2603.12702)**

> **作者:** Chaojie Sun; Bin Cao; Tiantian Li; Chenyu Hou; Ruizhe Li; Jing Fan
>
> **备注:** work in process;10pages, 5 figures, 4 tables
>
> **摘要:** With the rapid advancement of large language models (LLMs), growing efforts have been made on LLM-based table retrieval. However, existing studies typically focus on single-table query, and implement it by similarity matching after encoding the entire table. These methods usually result in low accuracy due to their coarse-grained encoding which incorporates much query-irrelated data, and are also inefficient when dealing with large tables, failing to fully utilize the reasoning capabilities of LLM. Further, multi-table query is under-explored in retrieval tasks. To this end, we propose a hierarchical multi-table query method based on LLM: Fine-Grained Multi-Table Retrieval FGTR, a new retrieval paradigm that employs a human-like reasoning strategy. Through hierarchical reasoning, FGTR first identifies relevant schema elements and then retrieves the corresponding cell contents, ultimately constructing a concise and accurate sub-table that aligns with the given query. To comprehensively evaluate the performance of FGTR, we construct two new benchmark datasets based on Spider and BIRD . Experimental results show that FGTR outperforms previous state-of-the-art methods, improving the F_2 metric by 18% on Spider and 21% on BIRD, demonstrating its effectiveness in enhancing fine-grained retrieval and its potential to improve end-to-end performance on table-based downstream tasks.
>
---
#### [replaced 049] HypeLoRA: Hyper-Network-Generated LoRA Adapters for Calibrated Language Model Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型校准问题，提出HypeLoRA框架，通过低秩适配提升参数效率与预测可靠性，解决模型过自信问题。**

- **链接: [https://arxiv.org/pdf/2603.19278](https://arxiv.org/pdf/2603.19278)**

> **作者:** Bartosz Trojan; Filip Gębala
>
> **备注:** 12 pages, 2 figures, 2 tables
>
> **摘要:** Modern Transformer-based models frequently suffer from miscalibration, producing overconfident predictions that do not reflect true empirical frequencies. This work investigates the calibration dynamics of LoRA: Low-Rank Adaptation and a novel hyper-network-based adaptation framework as parameter-efficient alternatives to full fine-tuning for RoBERTa. Evaluating across the GLUE benchmark, we demonstrate that LoRA-based adaptation consistently achieves calibration parity with (and in specific tasks exceeds) full fine-tuning, while maintaining significantly higher parameter efficiency. We further explore a dynamic approach where a shared hyper-network generates LoRA factors (A and B matrices) to induce structural coupling across layers. This approach produced results similar to standard LoRA fine-tuning, even achieving better MCC on CoLA dataset. Our study also reveal a critical trade-off: constraining the adaptation space (e.g., freezing matrices A) acts as a powerful regularizer that enhances Expected Calibration Error (ECE), but necessitates a carefully balanced sacrifice in downstream task accuracy. To support future research, we provide a unified and reproducible implementation of contemporary calibration metrics, including ECE, MCE, and ACE. Our findings clarify the relationship between parameter efficiency and probabilistic reliability, positioning structured low-rank updates as a viable foundation for uncertainty-aware Transformer architectures. Code available at: this https URL
>
---
#### [replaced 050] Your Models Have Thought Enough: Training Large Reasoning Models to Stop Overthinking
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大推理模型计算成本高的问题。通过引入JET方法，使模型在足够推理后主动停止，提升效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2509.23392](https://arxiv.org/pdf/2509.23392)**

> **作者:** Jinyi Han; Ying Huang; Ying Liao; Zishang Jiang; Xikun Lu; Haiquan Zhao; Xinyi Wang; Guanghao Zhou; Sihang Jiang; Jiaqing Liang; Weikang Zhou; Zeye Sun; Fei Yu; Yanghua Xiao
>
> **摘要:** Large Reasoning Models (LRMs) have achieved impressive performance on challenging tasks, yet their deep reasoning often incurs substantial computational costs. To achieve efficient reasoning, existing reinforcement learning methods still struggle to construct short reasoning path during the rollout stage, limiting effective learning. Inspired by Evidence Accumulation Models, we find that LRMs have accumulated sufficient information early in reasoning, making further reasoning steps redundant. Based on this insight, we propose Just-Enough Thinking (JET), which trains models to proactively terminate unnecessary reasoning. JET performs trajectory truncation during rollout to expose the model to short, distributionally consistent reasoning paths. Besides, it uses a quality-controlled length reward to better encourage concise reasoning while maintaining correctness. Extensive experiments demonstrate that JET significantly improves reasoning efficiency without sacrificing accuracy. Especially, DeepSeek-Distill-Qwen-1.5B achieves a 4.6% accuracy gain while reducing output length by 46.3% on the Olympiad benchmark. Our code is available in the GitHub.
>
---
#### [replaced 051] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VLM-3R，解决视觉-语言模型在3D空间理解上的不足，通过单目视频和3D重建指令微调，提升空间与时间推理能力。**

- **链接: [https://arxiv.org/pdf/2505.20279](https://arxiv.org/pdf/2505.20279)**

> **作者:** Zhiwen Fan; Jian Zhang; Renjie Li; Junge Zhang; Runjin Chen; Hezhen Hu; Kevin Wang; Huaizhi Qu; Shijie Zhou; Dilin Wang; Zhicheng Yan; Hongyu Xu; Justin Theiss; Tianlong Chen; Jiachen Li; Zhengzhong Tu; Zhangyang Wang; Rakesh Ranjan
>
> **备注:** Project Page: this https URL
>
> **摘要:** The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. Nevertheless, achieving deep spatial understanding comparable to human capabilities poses significant challenges in model encoding and data acquisition. Existing methods frequently depend on external depth sensors for geometry capture or utilize off-the-shelf algorithms for pre-constructing 3D maps, thereby limiting their scalability, especially with prevalent monocular video inputs and for time-sensitive applications. In this work, we introduce VLM-3R, a unified framework for Vision-Language Models (VLMs) that incorporates 3D Reconstructive instruction tuning. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Leveraging our Spatial-Visual-View Fusion and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables monocular 3D spatial assistance and embodied reasoning. To facilitate the evaluation of temporal reasoning, we introduce the Vision-Spatial-Temporal Intelligence benchmark, featuring over 138.6K QA pairs across five distinct tasks focused on evolving spatial relationships. Extensive experiments demonstrate that our model, VLM-3R, not only facilitates robust visual-spatial reasoning but also enables the understanding of temporal 3D context changes, excelling in both accuracy and scalability.
>
---
#### [replaced 052] A Browser-based Open Source Assistant for Multimodal Content Verification
- **分类: cs.CL**

- **简介: 该论文属于内容验证任务，旨在解决虚假信息快速验证问题。提出一种浏览器插件工具，集成NLP模型，为用户提供可信度分析和AI生成内容检测。**

- **链接: [https://arxiv.org/pdf/2603.02842](https://arxiv.org/pdf/2603.02842)**

> **作者:** Rosanna Milner; Michael Foster; Twin Karmakharm; Olesya Razuvayevskaya; Ian Roberts; Valentin Porcellini; Denis Teyssou; Kalina Bontcheva
>
> **摘要:** Disinformation and false content produced by generative AI pose a significant challenge for journalists and fact-checkers who must rapidly verify digital media information. While there is an abundance of NLP models for detecting credibility signals such as persuasion techniques, subjectivity, or machine-generated text, such methods often remain inaccessible to non-expert users and are not integrated into their daily workflows as a unified framework. This paper demonstrates the VERIFICATION ASSISTANT, a browser-based tool designed to bridge this gap. The VERIFICATION ASSISTANT, a core component of the widely adopted VERIFICATION PLUGIN (140,000+ users), allows users to submit URLs or media files to a unified interface. It automatically extracts content and routes it to a suite of backend NLP classifiers, delivering actionable credibility signals, estimating AI-generated content, and providing other verification guidance in a clear, easy-to-digest format. This paper showcases the tool architecture, its integration of multiple NLP services, and its real-world application to detecting disinformation.
>
---
#### [replaced 053] Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions
- **分类: cs.CL**

- **简介: 本文探讨大语言模型中的模型融合任务，旨在通过合并多个模型参数提升效率与性能。解决如何高效整合模型以降低计算成本的问题，工作包括方法综述、应用场景分析及未来方向探讨。**

- **链接: [https://arxiv.org/pdf/2603.09938](https://arxiv.org/pdf/2603.09938)**

> **作者:** Mingyang Song; Mao Zheng
>
> **摘要:** Model merging combines the parameters of multiple neural networks into a single model without additional training. As fine-tuned large language models (LLMs) proliferate, merging offers a computationally efficient alternative to ensembles and full retraining, enabling practitioners to compose specialized capabilities at minimal cost. This survey examines model merging in the LLM era through the \textbf{FUSE} taxonomy, organized along \textbf{F}oundations, \textbf{U}nification Strategies, \textbf{S}cenarios, and \textbf{E}cosystem. We first establish the theoretical underpinnings of merging, including loss landscape geometry and mode connectivity, then systematically review the algorithmic space spanning weight averaging, task vector arithmetic, sparsification-enhanced methods, mixture-of-experts architectures, and evolutionary optimization. We further examine downstream applications across multi-task learning, safety alignment, domain specialization, and federated learning, and survey the supporting ecosystem of tools and evaluation benchmarks. Finally, we identify key open challenges and future directions, aiming to equip researchers and practitioners with a structured foundation for advancing model merging.
>
---
#### [replaced 054] OnCoCo 1.0: A Public Dataset for Fine-Grained Message Classification in Online Counseling Conversations
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出OnCoCo 1.0数据集，用于在线心理咨询对话的细粒度消息分类。解决现有分类体系局限性问题，构建了38类咨询师和28类来访者话语的标注数据集，提升心理社会对话分析效果。**

- **链接: [https://arxiv.org/pdf/2512.09804](https://arxiv.org/pdf/2512.09804)**

> **作者:** Jens Albrecht; Robert Lehmann; Aleksandra Poltermann; Eric Rudolph; Philipp Steigerwald; Mara Stieler
>
> **备注:** Accepted at SoCon-NLPSI@LREC 2026
>
> **摘要:** This paper presents OnCoCo 1.0, a new public dataset for fine-grained message classification in online counseling. It is based on a new, integrative system of categories, designed to improve the automated analysis of psychosocial online counseling conversations. Existing category systems, predominantly based on Motivational Interviewing (MI), are limited by their narrow focus and dependence on datasets derived mainly from face-to-face counseling. This limits the detailed examination of textual counseling conversations. In response, we developed a comprehensive new coding scheme that differentiates between 38 types of counselor and 28 types of client utterances, and created a labeled dataset consisting of about 2.800 messages from counseling conversations. We fine-tuned several models on our dataset to demonstrate its applicability. The data and models are publicly available to researchers and practitioners. Thus, our work contributes a new type of fine-grained conversational resource to the language resources community, extending existing datasets for social and mental-health dialogue analysis.
>
---
#### [replaced 055] Alignment Whack-a-Mole : Finetuning Activates Verbatim Recall of Copyrighted Books in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究模型在微调后泄露版权书籍的问题，揭示了大语言模型可能存储并再现受版权保护的内容，挑战了现有安全措施的有效性。**

- **链接: [https://arxiv.org/pdf/2603.20957](https://arxiv.org/pdf/2603.20957)**

> **作者:** Xinyue Liu; Niloofar Mireshghallah; Jane C. Ginsburg; Tuhin Chakrabarty
>
> **备注:** Preprint Under Review
>
> **摘要:** Frontier LLM companies have repeatedly assured courts and regulators that their models do not store copies of training data. They further rely on safety alignment strategies via RLHF, system prompts, and output filters to block verbatim regurgitation of copyrighted works, and have cited the efficacy of these measures in their legal defenses against copyright infringement claims. We show that finetuning bypasses these protections: by training models to expand plot summaries into full text, a task naturally suited for commercial writing assistants, we cause GPT-4o, Gemini-2.5-Pro, and DeepSeek-V3.1 to reproduce up to 85-90% of held-out copyrighted books, with single verbatim spans exceeding 460 words, using only semantic descriptions as prompts and no actual book text. This extraction generalizes across authors: finetuning exclusively on Haruki Murakami's novels unlocks verbatim recall of copyrighted books from over 30 unrelated authors. The effect is not specific to any training author or corpus: random author pairs and public-domain finetuning data produce comparable extraction, while finetuning on synthetic text yields near-zero extraction, indicating that finetuning on individual authors' works reactivates latent memorization from pretraining. Three models from different providers memorize the same books in the same regions ($r \ge 0.90$), pointing to an industry-wide vulnerability. Our findings offer compelling evidence that model weights store copies of copyrighted works and that the security failures that manifest after finetuning on individual authors' works undermine a key premise of recent fair use rulings, where courts have conditioned favorable outcomes on the adequacy of measures preventing reproduction of protected expression.
>
---
#### [replaced 056] SkillFlow: Scalable and Efficient Agent Skill Retrieval System
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出SkillFlow，解决AI代理在大量技能中高效检索相关技能的问题。通过多阶段信息检索方法提升性能，强调技能库质量和覆盖的重要性。**

- **链接: [https://arxiv.org/pdf/2504.06188](https://arxiv.org/pdf/2504.06188)**

> **作者:** Fangzhou Li; Pagkratios Tagkopoulos; Ilias Tagkopoulos
>
> **摘要:** AI agents can extend their capabilities at inference time by loading reusable skills into context, yet equipping an agent with too many skills, particularly irrelevant ones, degrades performance. As community-driven skill repositories grow, agents need a way to selectively retrieve only the most relevant skills from a large library. We present SkillFlow, the first multi-stage retrieval pipeline designed for agent skill discovery, framing skill acquisition as an information retrieval problem over a corpus of ~36K community-contributed this http URL definitions indexed from GitHub. The pipeline progressively narrows a large candidate set through four stages: dense retrieval, two rounds of cross-encoder reranking, and LLM-based selection, balancing recall and precision at each stage. We evaluate SkillFlow on two coding benchmarks: SkillsBench, a benchmark of 87 tasks and 229 matched skills; and Terminal-Bench, a benchmark that provides only 89 tasks, and no matched skills. On SkillsBench, SkillFlow-retrieved skills raise Pass@1 from 9.2% to 16.4% (+78.3%, $p_{\text{adj}} = 3.64 \times 10^{-2}$), reaching 84.1% of the oracle ceiling, while on Terminal-Bench, agents readily use the retrieved skills (70.1% use rate) yet show no performance gain, revealing that retrieval alone is insufficient when the corpus lacks high-quality, executable skills for the target domain. SkillFlow demonstrates that framing skill acquisition as an information retrieval task is an effective strategy, and that the practical impact of skill-augmented agents hinges on corpus coverage and skill quality, particularly the density of runnable code and bundled artifacts.
>
---
#### [replaced 057] CLMN: Concept based Language Models via Neural Symbolic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CLMN模型，解决NLP中可解释性不足的问题。通过结合神经网络与符号推理，提升模型的透明度和逻辑规则的可解释性。**

- **链接: [https://arxiv.org/pdf/2510.10063](https://arxiv.org/pdf/2510.10063)**

> **作者:** Yibo Yang
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Deep learning has advanced NLP, but interpretability remains limited, especially in healthcare and finance. Concept bottleneck models tie predictions to human concepts in vision, but NLP versions either use binary activations that harm text representations or latent concepts that weaken semantics, and they rarely model dynamic concept interactions such as negation and context. We introduce the Concept Language Model Network (CLMN), a neural-symbolic framework that keeps both performance and interpretability. CLMN represents concepts as continuous, human-readable embeddings and applies fuzzy-logic reasoning to learn adaptive interaction rules that state how concepts affect each other and the final decision. The model augments original text features with concept-aware representations and automatically induces interpretable logic rules. Across multiple datasets and pre-trained language models, CLMN achieves higher accuracy than existing concept-based methods while improving explanation quality. These results show that integrating neural representations with symbolic reasoning in a unified concept space can yield practical, transparent NLP systems.
>
---
#### [replaced 058] GhanaNLP Parallel Corpora: Comprehensive Multilingual Resources for Low-Resource Ghanaian Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源非洲语言数据不足的问题。研究者构建了涵盖五种加纳语言的平行语料库，用于支持机器翻译等应用。**

- **链接: [https://arxiv.org/pdf/2603.13793](https://arxiv.org/pdf/2603.13793)**

> **作者:** Lawrence Adu Gyamfi; Paul Azunre; Stephen Edward Moore; Joel Budu; Akwasi Asare; Mich-Seth Owusu; Jonathan Ofori Asiamah
>
> **摘要:** Low resource languages present unique challenges for natural language processing due to the limited availability of digitized and well structured linguistic data. To address this gap, the GhanaNLP initiative has developed and curated 41,513 parallel sentence pairs for the Twi, Fante, Ewe, Ga, and Kusaal languages, which are widely spoken across Ghana yet remain underrepresented in digital spaces. Each dataset consists of carefully aligned sentence pairs between a local language and English. The data were collected, translated, and annotated by human professionals and enriched with standard structural metadata to ensure consistency and usability. These corpora are designed to support research, educational, and commercial applications, including machine translation, speech technologies, and language preservation. This paper documents the dataset creation methodology, structure, intended use cases, and evaluation, as well as their deployment in real world applications such as the Khaya AI translation engine. Overall, this work contributes to broader efforts to democratize AI by enabling inclusive and accessible language technologies for African languages.
>
---
#### [replaced 059] Mitigating Premature Exploitation in Particle-based Monte Carlo for Inference-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于语言模型推理任务，解决粒子滤波在推理时扩展中的过早利用问题。通过引入熵退火和前瞻调制，提升粒子多样性与路径评估能力，显著提高任务奖励。**

- **链接: [https://arxiv.org/pdf/2510.05825](https://arxiv.org/pdf/2510.05825)**

> **作者:** Giorgio Giannone; Guangxuan Xu; Nikhil Shivakumar Nayak; Rohan Mahesh Awhad; Shivchander Sudalairaj; Kai Xu; Akash Srivastava
>
> **备注:** preprint
>
> **摘要:** Inference-Time Scaling (ITS) improves language models by allocating more computation at generation time. Particle Filtering (PF) has emerged as a strong ITS method for complex mathematical reasoning tasks, but it is vulnerable when guided by process reward models, which often assign overconfident scores early in the reasoning process. This causes PF to suffer from premature exploitation: it myopically commits to locally promising trajectories, prunes potentially correct hypotheses, and converges to suboptimal solutions. This failure mode, known as particle impoverishment, is especially severe under constrained computational budgets. To address this, we analyze the problem and identify two root causes: a lack of diversity in the particle set due to overconfident resampling and consequent inability to assess the potential of a reasoning path. We introduce Entropic Particle Filtering (ePF), an algorithm that integrates two new techniques to solve these issues. The first technique, Entropic Annealing (EA), directly mitigates particle impoverishment by monitoring search diversity via entropy; when diversity drops, it intervenes by dynamically annealing the resampling distribution to preserve exploration. The second, an enhancement called Look-ahead Modulation (LaM), adds a predictive guide to evaluate a state's potential based on its successors. On several challenging math benchmarks, ePF significantly outperforms strong baselines and achieves up to a 50% relative improvement in task reward. Together, these methods improve PF's resilience by balancing the exploration of diverse solution spaces with the exploitation of high-reward regions, ultimately leading to higher-quality solutions.
>
---
#### [replaced 060] Benchmarking NLP-supported Language Sample Analysis for Swiss Children's Speech
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言分析任务，旨在解决DLD诊断效率低的问题。通过NLP技术处理儿童语音数据，探索半自动LSA的优化方法。**

- **链接: [https://arxiv.org/pdf/2504.00780](https://arxiv.org/pdf/2504.00780)**

> **作者:** Anja Ryser; Yingqiang Gao; Sarah Ebling
>
> **备注:** updated preprint
>
> **摘要:** Language sample analysis (LSA) is a process that complements standardized psychometric tests for diagnosing, for example, developmental language disorder (DLD) in children. However, its labour-intensive nature has limited its use in speech-language pathology practice. We introduce an approach that leverages natural language processing (NLP) methods that do not rely on commercial large language models (LLMs) applied to transcribed speech data from 119 children in the German-speaking part of Switzerland with typical and atypical language development. This preliminary study aims to identify optimal practices that support speech-language pathologists in diagnosing DLD more efficiently with active involvement of human specialists. Preliminary findings underscore the potential of integrating locally deployed NLP methods into the process of semi-automatic LSA.
>
---
#### [replaced 061] $π$-Attention: Periodic Sparse Transformers for Efficient Long-Context Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出π-Attention，解决长序列建模中计算复杂度高的问题。通过周期性稀疏注意力机制，提升效率并保持性能，适用于自然语言处理和视觉语言任务。**

- **链接: [https://arxiv.org/pdf/2511.10696](https://arxiv.org/pdf/2511.10696)**

> **作者:** Dong Liu; Yanxuan Yu
>
> **摘要:** Transformers have revolutionized natural language processing, but their quadratic complexity with respect to sequence length remains a fundamental bottleneck for long-range modeling. While sparse attention mechanisms like RingAttention reduce computational costs by restricting attention to local neighborhoods, they suffer from limited receptive fields and lack of adaptability. We present \PiAttention, a periodic sparse Transformer that factorizes attention into ring-local neighborhoods, deterministic $\pi$-stride skips, and an adaptive fusion gate. The periodic structure provides predictable coverage of distant tokens, while the sparse footprint keeps the per-layer complexity linear in context length. We prove that \PiAttention achieves $\mathcal{O}(kL + \pi \log L)$ receptive field growth compared to $\mathcal{O}(kL)$ for RingAttention, where $k$ is the local window size, $\pi$ is the skip period, and $L$ is the sequence length. Extensive experiments on language modeling, retrieval, and vision-language tasks demonstrate that \PiAttention matches or surpasses dense attention quality with 8.3\% lower perplexity than RingAttention while using 50\% fewer GPUs for the same context length. Our detailed ablations and visualizations reveal the importance of periodic skips, adaptive fusion, and head-level sparsity coordination for efficient long-context modeling.
>
---
#### [replaced 062] Retrieving Climate Change Disinformation by Narrative
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决气候虚假信息检测问题。通过将叙事检测转化为检索任务，提出SpecFi框架，提升对新兴叙事的适应性。**

- **链接: [https://arxiv.org/pdf/2603.22015](https://arxiv.org/pdf/2603.22015)**

> **作者:** Max Upravitelev; Veronika Solopova; Charlott Jakob; Premtim Sahitaj; Sebastian Möller; Vera Schmitt
>
> **摘要:** Detecting climate disinformation narratives typically relies on fixed taxonomies, which do not accommodate emerging narratives. Thus, we re-frame narrative detection as a retrieval task: given a narrative's core message as a query, rank texts from a corpus by alignment with that narrative. This formulation requires no predefined label set and can accommodate emerging narratives. We repurpose three climate disinformation datasets (CARDS, Climate Obstruction, climate change subset of PolyNarrative) for retrieval evaluation and propose SpecFi, a framework that generates hypothetical documents to bridge the gap between abstract narrative descriptions and their concrete textual instantiations. SpecFi uses community summaries from graph-based community detection as few-shot examples for generation, achieving a MAP of 0.505 on CARDS without access to narrative labels. We further introduce narrative variance, an embedding-based difficulty metric, and show via partial correlation analysis that standard retrieval degrades on high-variance narratives (BM25 loses 63.4% of MAP), while SpecFi-CS remains robust (32.7% loss). Our analysis also reveals that unsupervised community summaries converge on descriptions close to expert-crafted taxonomies, suggesting that graph-based methods can surface narrative structure from unlabeled text.
>
---
#### [replaced 063] Understanding the Anchoring Effect of LLM with Synthetic Data: Existence, Mechanism, and Potential Mitigations
- **分类: cs.CL**

- **简介: 该论文研究LLM的锚定效应，属于认知偏差分析任务。旨在揭示其存在、机制及缓解方法，通过构建数据集并测试模型表现。**

- **链接: [https://arxiv.org/pdf/2505.15392](https://arxiv.org/pdf/2505.15392)**

> **作者:** Yiming Huang; Biquan Bie; Zuqiu Na; Weilin Ruan; Songxin Lei; Yutao Yue; Xinlei He
>
> **备注:** Accepted by the HCAIR workshop of ICLR 2026
>
> **摘要:** The rise of Large Language Models (LLMs) like ChatGPT has advanced natural language processing, yet concerns about cognitive biases are growing. In this paper, we investigate the anchoring effect, a cognitive bias where the mind relies heavily on the first information as anchors to make affected judgments. We explore whether LLMs are affected by anchoring, the underlying mechanisms, and potential mitigation strategies. To facilitate studies at scale on the anchoring effect, we introduce a new dataset, SynAnchors (this https URL). Combining refined evaluation metrics, we benchmark current widely used LLMs. Our findings show that LLMs' anchoring bias exists commonly with shallow-layer acting and can not be eliminated by conventional strategies, while reasoning can offer some mitigation.
>
---
#### [replaced 064] Silicon Bureaucracy and AI Test-Oriented Education: Contamination Sensitivity and Score Confidence in LLM Benchmarks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI评估任务，探讨LLM基准测试中的污染敏感性和分数可信度问题。通过实验发现基准分数可能受污染影响，提出审计框架以提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.21636](https://arxiv.org/pdf/2603.21636)**

> **作者:** Yiliang Song; Hongjun An; Jiangan Chen; Xuanchen Yan; Huan Song; Jiawei Shao; Xuelong Li
>
> **备注:** Remove the NeurIPS 2026 template
>
> **摘要:** Public benchmarks increasingly govern how large language models (LLMs) are ranked, selected, and deployed. We frame this benchmark-centered regime as Silicon Bureaucracy and AI Test-Oriented Education, and argue that it rests on a fragile assumption: that benchmark scores directly reflect genuine generalization. In practice, however, such scores may conflate exam-oriented competence with principled capability, especially when contamination and semantic leakage are difficult to exclude from modern training pipelines. We therefore propose an audit framework for analyzing contamination sensitivity and score confidence in LLM benchmarks. Using a router-worker setup, we compare a clean-control condition with noisy conditions in which benchmark problems are systematically deleted, rewritten, and perturbed before being passed downstream. For a genuinely clean benchmark, noisy conditions should not consistently outperform the clean-control baseline. Yet across multiple models, we find widespread but heterogeneous above-baseline gains under noisy conditions, indicating that benchmark-related cues may be reassembled and can reactivate contamination-related memory. These results suggest that similar benchmark scores may carry substantially different levels of confidence. Rather than rejecting benchmarks altogether, we argue that benchmark-based evaluation should be supplemented with explicit audits of contamination sensitivity and score confidence.
>
---
#### [replaced 065] Structured Agent Distillation for Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在降低大语言模型的部署成本。通过结构化蒸馏方法，将大模型的知识迁移到小模型，保持推理和行动的一致性。**

- **链接: [https://arxiv.org/pdf/2505.13820](https://arxiv.org/pdf/2505.13820)**

> **作者:** Jun Liu; Zhenglun Kong; Peiyan Dong; Changdi Yang; Tianqi Li; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Pu Zhao; Xue Lin; Dong Huang; Yanzhi Wang
>
> **摘要:** Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents.
>
---
#### [replaced 066] AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，研究LLM代理在工具污染下的推荐偏差问题。通过实验揭示评估指标的盲点，提出改进的评估方法以提升多轮交互中的安全性。**

- **链接: [https://arxiv.org/pdf/2603.12564](https://arxiv.org/pdf/2603.12564)**

> **作者:** Zekun Wu; Adriano Koshiyama; Sahan Bulathwela; Maria Perez-Ortiz
>
> **备注:** 51 pages, 31 tables, 18 figures. Under review at COLM 2026
>
> **摘要:** Tool-augmented LLM agents increasingly operate as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking metrics that measure what is recommended but not whether it is safe for the user. We present a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across eight LLMs (7B to frontier), decomposing divergence into information-channel and memory-channel mechanisms. We observe evaluation blindness: recommendation quality is preserved under contamination (UPR~1.0) while risk-inappropriate products appear in 65-93% of turns, invisible to standard NDCG. Violations are information-channel-driven, emerge at turn 1, and persist without self-correction over 23-step trajectories. Even non-extreme perturbations (within-band corruption, narrative-only attacks) evade threshold monitors while producing significant drift. Susceptibility scales with instruction-following fidelity across all eight models. Sparse autoencoder probing reveals models internally distinguish adversarial perturbations but fail to propagate this signal to output; causal interventions (activation patching, feature clamping, direct steering) confirm this representation-to-action gap is structural and resists linear repair. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74. These results motivate trajectory-level safety monitoring for deployed multi-turn agents.
>
---
#### [replaced 067] sebis at ArchEHR-QA 2026: How Much Can You Do Locally? Evaluating Grounded EHR QA on a Single Notebook
- **分类: cs.CL**

- **简介: 该论文属于EHR问答任务，旨在解决隐私和计算限制下的本地化临床问答问题。工作包括在单个笔记本上评估多种方法，验证小型模型的可行性。**

- **链接: [https://arxiv.org/pdf/2603.13962](https://arxiv.org/pdf/2603.13962)**

> **作者:** Ibrahim Ebrar Yurt; Fabian Karl; Tejaswi Choppa; Florian Matthes
>
> **摘要:** Clinical question answering over electronic health records (EHRs) can help clinicians and patients access relevant medical information more efficiently. However, many recent approaches rely on large cloud-based models, which are difficult to deploy in clinical environments due to privacy constraints and computational requirements. In this work, we investigate how far grounded EHR question answering can be pushed when restricted to a single notebook. We participate in all four subtasks of the ArchEHR-QA 2026 shared task and evaluate several approaches designed to run on commodity hardware. All experiments are conducted locally without external APIs or cloud infrastructure. Our results show that such systems can achieve competitive performance on the shared task leaderboards. In particular, our submissions perform above average in two subtasks, and we observe that smaller models can approach the performance of much larger systems when properly configured. These findings suggest that privacy-preserving EHR QA systems running fully locally are feasible with current models and commodity hardware. The source code is available at this https URL.
>
---
#### [replaced 068] CREST: Universal Safety Guardrails Through Cluster-Guided Cross-Lingual Transfer
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言安全分类任务，旨在解决低资源语言安全防护不足的问题。通过跨语言迁移学习，用少量高资源语言训练模型，实现对100种语言的安全检测。**

- **链接: [https://arxiv.org/pdf/2512.02711](https://arxiv.org/pdf/2512.02711)**

> **作者:** Lavish Bansal; Naman Mishra
>
> **备注:** 9 Pages, 5 Figures, Accepted at LREC 2026
>
> **摘要:** Ensuring content safety in large language models (LLMs) is essential for their deployment in real-world applications. However, existing safety guardrails are predominantly tailored for high-resource languages, leaving a significant portion of the world's population underrepresented who communicate in low-resource languages. To address this, we introduce CREST (CRoss-lingual Efficient Safety Transfer), a parameter-efficient multilingual safety classification model that supports 100 languages with only 0.5B parameters. By training on a strategically chosen subset of only 13 high-resource languages, our model utilizes cluster-based cross-lingual transfer from a few to 100 languages, enabling effective generalization to both unseen high-resource and low-resource languages. This approach addresses the challenge of limited training data in low-resource settings. We conduct comprehensive evaluations across six safety benchmarks to demonstrate that CREST outperforms existing state-of-the-art guardrails of comparable scale and achieves competitive results against models with significantly larger parameter counts (2.5B parameters and above). Our findings highlight the limitations of language-specific guardrails and underscore the importance of developing universal, language-agnostic safety systems that can scale effectively to serve global populations.
>
---
#### [replaced 069] Estonian WinoGrande Dataset: Comparative Analysis of LLM Performance on Human and Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在比较大语言模型在人类与机器翻译数据上的表现。研究者构建了爱沙尼亚语的WinoGrande数据集，分析翻译质量对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2511.17290](https://arxiv.org/pdf/2511.17290)**

> **作者:** Marii Ojastu; Hele-Andra Kuulmets; Aleksei Dorkin; Marika Borovikova; Dage Särg; Kairit Sirts
>
> **备注:** LREC 2026
>
> **摘要:** In this paper, we present a localized and culturally adapted Estonian translation of the test set from the widely used commonsense reasoning benchmark, WinoGrande. We detail the translation and adaptation process carried out by translation specialists and evaluate the performance of both proprietary and open source models on the human translated benchmark. Additionally, we explore the feasibility of achieving high-quality machine translation by incorporating insights from the manual translation process into the design of a detailed prompt. This prompt is specifically tailored to address both the linguistic characteristics of Estonian and the unique translation challenges posed by the WinoGrande dataset. Our findings show that model performance on the human translated Estonian dataset is slightly lower than on the original English test set, while performance on machine-translated data is notably worse. Additionally, our experiments indicate that prompt engineering offers limited improvement in translation quality or model accuracy, and highlight the importance of involving language specialists in dataset translation and adaptation to ensure reliable and interpretable evaluations of language competency and reasoning in large language models.
>
---
#### [replaced 070] CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于AI科学任务，旨在解决演化效率低和探索方向不足的问题。提出CausalEvolve，通过因果推理提升演化效率与创新性。**

- **链接: [https://arxiv.org/pdf/2603.14575](https://arxiv.org/pdf/2603.14575)**

> **作者:** Yongqiang Chen; Chenxi Liu; Zhenhao Chen; Tongliang Liu; Bo Han; Kun Zhang
>
> **备注:** Preprint of ongoing work; Yongqiang and Chenxi contributed equally;
>
> **摘要:** Evolve-based agent such as AlphaEvolve is one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite the success, existing evolve-based agents lack targeted guidance for evolution and effective mechanisms for organizing and utilizing knowledge acquired from past evolutionary experience. Consequently, they suffer from decreasing evolution efficiency and exhibit oscillatory behavior when approaching known performance boundaries. To mitigate the gap, we develop CausalEvolve, equipped with a causal scratchpad that leverages LLMs to identify and reason about guiding factors for evolution. At the beginning, CausalEvolve first identifies outcome-level factors that offer complementary inspirations in improving the target objective. During the evolution, CausalEvolve also inspects surprise patterns during the evolution and abductive reasoning to hypothesize new factors, which in turn offer novel directions. Through comprehensive experiments, we show that CausalEvolve effectively improves the evolutionary efficiency and discovers better solutions in 4 challenging open-ended scientific tasks.
>
---
#### [replaced 071] EngGPT2: Sovereign, Efficient and Open Intelligence
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍EngGPT2-16B-A3B，一款高效、开源的大型语言模型，旨在提升欧洲及意大利自然语言处理性能，解决资源消耗高、数据依赖性强的问题。**

- **链接: [https://arxiv.org/pdf/2603.16430](https://arxiv.org/pdf/2603.16430)**

> **作者:** G. Ciarfaglia; A. Rosanova; S. Cipolla; J. Bartoli; A. Di Domenico; C. Fioroni; A. Fontana; M. R. Scoleri; M. I. Mone; D. Franchi; M. C. Del Gaudio; A. Leodori; F. Cinti; M. Capozzi; C. Baston; F. Picariello; M. Gabusi; S. Bonura; V. Morreale; I. Bailo
>
> **摘要:** EngGPT2-16B-A3B is the latest iteration of Engineering Group's Italian LLM and it's built to be a Sovereign, Efficient and Open model. EngGPT2 is trained on 2.5 trillion tokens - less than Qwen3's 36T or Llama3's 15T - and delivers performance on key benchmarks, including MMLU-Pro, GSM8K, IFEval and HumanEval, comparable to dense models in the 8B-16B range, while requiring one-fifth to half of the inference power, and between one-tenth to one-sixth of the training data and consequent needed training power. Designed as a trained-from-scratch Mixture-of-Experts (MoE) architecture, EngGPT2 features 16 billion parameters with 3 billion active per inference, with expert sizes positioned between those used in GPT-OSS and Qwen3. Approximately 25% of its training corpus consists of Italian-language data, to deliver strong capabilities for European and Italian NLP tasks among models of similar scale. This efficiency aims to position EngGPT2 as a key contributor to the growing portfolio of open-weight European models, combining performance and efficiency with full alignment to the EU AI Act. EngGPT2 is also a single model capable of multiple reasoning modes: non-reasoning, reasoning in Italian or English, and turbo-reasoning (a concise, bullet-point style reasoning available in both languages designed for real-time reasoning use cases). EngGPT2 aims to set a new standard for resource-conscious, high-performance LLMs tailored to European and Italian contexts.
>
---
#### [replaced 072] Measuring Complexity at the Requirements Stage: Spectral Metrics as Development Effort Predictors
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于需求工程任务，旨在解决需求复杂性量化不足的问题。通过自然语言处理提取结构网络，利用谱度量预测开发工作量，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.07182](https://arxiv.org/pdf/2602.07182)**

> **作者:** Maximilian Vierlboeck; Antonio Pugliese; Roshanak Rose Nilchian; Paul T. Grogan; Rashika Sugganahalli Natesh Babu
>
> **备注:** 36 pages, 4 figures, 5 tables
>
> **摘要:** Complexity in engineered systems presents one of the most persistent challenges in modern development since it is driving cost overruns, schedule delays, and outright project failures. Yet while architectural complexity has been studied, the structural complexity embedded within requirements specifications remains poorly understood and inadequately quantified. This gap is consequential: requirements fundamentally drive system design, and complexity introduced at this stage propagates through architecture, implementation, and integration. To address this gap, we build on Natural Language Processing methods that extract structural networks from textual requirements. Using these extracted structures, we conduct a controlled experiment employing molecular integration tasks as structurally isomorphic proxies for requirements integration -- leveraging the topological equivalence between molecular graphs and requirement networks while eliminating confounding factors such as domain expertise and semantic ambiguity. Our results demonstrate that spectral measures predict integration effort with correlations exceeding 0.95, while structural metrics achieve correlations above 0.89. Notably, density-based metrics show no significant predictive validity. These findings indicate that eigenvalue-derived measures capture cognitive and effort dimensions that simpler connectivity metrics cannot. As a result, this research bridges a critical methodological gap between architectural complexity analysis and requirements engineering practice, providing a validated foundation for applying these metrics to requirements engineering, where similar structural complexity patterns may predict integration effort.
>
---
#### [replaced 073] Based on Data Balancing and Model Improvement for Multi-Label Sentiment Classification Performance Enhancement
- **分类: cs.CL**

- **简介: 该论文属于多标签情感分类任务，针对数据不平衡问题进行改进。通过构建平衡数据集和优化模型结构，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2511.14073](https://arxiv.org/pdf/2511.14073)**

> **作者:** Zijin Su; Huanzhu Lyu; Yuren Niu; Yiming Liu
>
> **备注:** 9 pages, updated methodology and evaluation, added audit summary, label-cardinality and per-label count analyses, clarified splits and threshold tuning, added DistilRoBERTa baseline comparison. Updated figures, tables, references, and data-availability statement
>
> **摘要:** Multi-label sentiment classification plays a vital role in natural language processing by detecting multiple emotions within a single text. However, existing datasets like GoEmotions often suffer from severe class imbalance, which hampers model performance, especially for underrepresented emotions. To address this, we constructed a balanced multi-label sentiment dataset by integrating the original GoEmotions data, emotion-labeled samples from Sentiment140 using a RoBERTa-base-GoEmotions model, and manually annotated texts generated by GPT-4 mini. Our data balancing strategy ensured an even distribution across 28 emotion categories. Based on this dataset, we developed an enhanced multi-label classification model that combines pre-trained FastText embeddings, convolutional layers for local feature extraction, bidirectional LSTM for contextual learning, and an attention mechanism to highlight sentiment-relevant words. A sigmoid-activated output layer enables multi-label prediction, and mixed precision training improves computational efficiency. Experimental results demonstrate significant improvements in accuracy, precision, recall, F1-score, and AUC compared to models trained on imbalanced data, highlighting the effectiveness of our approach.
>
---
#### [replaced 074] Shifting Perspectives: Steering Vectors for Robust Bias Mitigation in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的偏见缓解任务，旨在减少大语言模型中的社会偏见。通过引入转向向量优化模型激活，有效降低偏见，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2503.05371](https://arxiv.org/pdf/2503.05371)**

> **作者:** Zara Siddique; Irtaza Khalid; Liam D. Turner; Luis Espinosa-Anke
>
> **备注:** Published to EACL Findings 2026
>
> **摘要:** We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. We compute 8 steering vectors, each corresponding to a different social bias axis, such as age, gender, or race, on a training subset of the BBQ dataset and compare the effectiveness of these to 3 additional bias mitigation methods across 4 datasets. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.8% on BBQ, 8.3% on CLEAR-Bias, and 1% on StereoSet, and show improvements over prompting and Self-Debias in all cases, and improvements over fine-tuning in 12 out of 17 evaluations. In addition, steering vectors showed the lowest impact on MMLU scores of the four bias mitigation methods tested. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that they are a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety.
>
---
#### [replaced 075] Nwāchā Munā: A Devanagari Speech Corpus and Proximal Transfer Benchmark for Nepal Bhasha ASR
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决尼泊尔语资源匮乏问题。构建了首个尼泊尔语语音语料库，并通过邻近语言迁移提升ASR性能。**

- **链接: [https://arxiv.org/pdf/2603.07554](https://arxiv.org/pdf/2603.07554)**

> **作者:** Rishikesh Kumar Sharma; Safal Narshing Shrestha; Jenny Poudel; Rupak Tiwari; Arju Shrestha; Rupak Raj Ghimire; Bal Krishna Bal
>
> **备注:** Accepted in CHiPSAL@LREC 2026
>
> **摘要:** Nepal Bhasha (Newari), an endangered language of the Kathmandu Valley, remains digitally marginalized due to the severe scarcity of annotated speech resources. In this work, we introduce Nwāchā Munā, a newly curated 5.39-hour manually transcribed Devanagari speech corpus for Nepal Bhasha, and establish the first benchmark using script-preserving acoustic modeling. We investigate whether proximal cross-lingual transfer from a geographically and linguistically adjacent language (Nepali) can rival large-scale multilingual pretraining in an ultra-low-resource Automatic Speech Recognition (ASR) setting. Fine-tuning a Nepali Conformer model reduces the Character Error Rate (CER) from a 52.54% zero-shot baseline to 17.59% with data augmentation, effectively matching the performance of the multilingual Whisper-Small model despite utilizing significantly fewer parameters. Our findings demonstrate that proximal transfer from Nepali language serves as a computationally efficient alternative to massive multilingual models. We openly release the dataset and benchmarks to digitally enable the Newari community and foster further research in Nepal Bhasha.
>
---
#### [replaced 076] L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出L-MARS框架，解决法律问答中需实时信息的问题。通过多智能体协作实现精准检索与推理，提升法律问答准确率。**

- **链接: [https://arxiv.org/pdf/2509.00761](https://arxiv.org/pdf/2509.00761)**

> **作者:** Ziqi Wang; Boqin Yuan
>
> **摘要:** We present L-MARS (Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search), a multi-agent retrieval framework for grounded legal question answering that decomposes queries into structured sub-problems, retrieves evidence via agentic web search, filters results through a verification agent, and synthesizes cited answers. Existing legal QA benchmarks test either closed-book reasoning or retrieval over fixed corpora, but neither captures scenarios requiring current legal information. We introduce LegalSearchQA, a 50-question benchmark across five legal domains whose answers depend on recent developments that post-date model training data. L-MARS achieves 96.0% accuracy on LegalSearchQA, a 38.0% improvement over zero-shot performance (58.0%), while chain-of-thought prompting degrades performance to 30.0%. On Bar Exam QA (Zheng et al., 2025), a reasoning-focused benchmark of 594 bar examination questions, retrieval provides negligible gains (+0.7 percentage points), consistent with prior findings. These results show that agentic retrieval dramatically improves legal QA when tasks require up-to-date factual knowledge, but the benefit is benchmark-dependent, underscoring the need for retrieval-focused evaluation. Code and data are available at: this https URL
>
---
#### [replaced 077] MuVaC: A Variational Causal Framework for Multimodal Sarcasm Understanding in Dialogues
- **分类: cs.CL**

- **简介: 该论文属于多模态讽刺理解任务，旨在解决讽刺检测与解释的联合优化问题。提出MuVaC框架，通过因果推理实现多模态特征融合与一致性增强。**

- **链接: [https://arxiv.org/pdf/2601.20451](https://arxiv.org/pdf/2601.20451)**

> **作者:** Diandian Guo; Fangfang Yuan; Cong Cao; Xixun Lin; Chuan Zhou; Hao Peng; Yanan Cao; Yanbing Liu
>
> **备注:** 12 pages, 7 figures. Accepted by WWW 2026
>
> **摘要:** The prevalence of sarcasm in multimodal dialogues on the social platforms presents a crucial yet challenging task for understanding the true intent behind online content. Comprehensive sarcasm analysis requires two key aspects: Multimodal Sarcasm Detection (MSD) and Multimodal Sarcasm Explanation (MuSE). Intuitively, the act of detection is the result of the reasoning process that explains the sarcasm. Current research predominantly focuses on addressing either MSD or MuSE as a single task. Even though some recent work has attempted to integrate these tasks, their inherent causal dependency is often overlooked. To bridge this gap, we propose MuVaC, a variational causal inference framework that mimics human cognitive mechanisms for understanding sarcasm, enabling robust multimodal feature learning to jointly optimize MSD and MuSE. Specifically, we first model MSD and MuSE from the perspective of structural causal models, establishing variational causal pathways to define the objectives for joint optimization. Next, we design an alignment-then-fusion approach to integrate multimodal features, providing robust fusion representations for sarcasm detection and explanation generation. Finally, we enhance the reasoning trustworthiness by ensuring consistency between detection results and explanations. Experimental results demonstrate the superiority of MuVaC in public datasets, offering a new perspective for understanding multimodal sarcasm.
>
---
#### [replaced 078] Measuring all the noises of LLM Evals
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于机器学习评估任务，解决LLM评估中的噪声分离问题，通过定义和测量三类噪声，提出配对分析方法提升统计效力。**

- **链接: [https://arxiv.org/pdf/2512.21326](https://arxiv.org/pdf/2512.21326)**

> **作者:** Sida Wang
>
> **摘要:** Separating signal from noise is central to experiments. Applying well-established statistical methods effectively to LLM evals requires consideration of their unique noise characteristics. We clearly define and measure three types of noise: prediction noise from generating different answers on a given question, data noise from sampling questions, and their combined total noise following the law of total variance. To emphasize relative comparisons and gain statistical power, we propose the all-pairs paired method, which applies the paired analysis to all pairs of LLMs and measures all the noise components based on millions of question-level predictions across many evals and settings, revealing clear patterns. First, each eval exhibits a characteristic and highly predictable total noise level across all model pairs. Second, paired prediction noise typically exceeds paired data noise, which means reducing prediction noise by averaging can significantly increase statistical power. By measuring all the noises together, we can assess eval results in context, lowering the barrier of using the best analysis to make sound empirical decisions.
>
---
#### [replaced 079] Large Language Models for Computer-Aided Design: A Survey
- **分类: cs.LG; cs.CL; cs.GR; cs.MM**

- **简介: 本文属于综述任务，旨在探讨大语言模型在计算机辅助设计中的应用。论文系统梳理了LLMs在CAD中的六类应用场景，提出未来发展方向，以推动CAD技术的创新与优化。**

- **链接: [https://arxiv.org/pdf/2505.08137](https://arxiv.org/pdf/2505.08137)**

> **作者:** Licheng Zhang; Bach Le; Naveed Akhtar; Siew-Kei Lam; Tuan Ngo
>
> **摘要:** Large Language Models (LLMs) have seen rapid advancements in recent years, with models like ChatGPT and DeepSeek, showcasing their remarkable capabilities across diverse domains. While substantial research has been conducted on LLMs in various fields, a comprehensive review focusing on their integration with Computer-Aided Design (CAD) remains notably absent. CAD is the industry standard for 3D modeling and plays a vital role in the design and development of products across different industries. As the complexity of modern designs increases, the potential for LLMs to enhance and streamline CAD workflows presents an exciting frontier. This article presents the first systematic survey exploring the intersection of LLMs and CAD. We begin by outlining the industrial significance of CAD, highlighting the need for AI-driven innovation. Next, we provide a detailed overview of the foundation of LLMs. We also examine both closed-source LLMs as well as publicly available models. The core of this review focuses on the various applications of LLMs in CAD, providing a taxonomy of six key areas where these models are making considerable impact. Finally, we propose several promising future directions for further advancements, which offer vast opportunities for innovation and are poised to shape the future of CAD technology. Github: this https URL
>
---
#### [replaced 080] Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM训练中长序列滚存导致的内存瓶颈问题。通过引入稀疏滚存机制，提升训练稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2601.10079](https://arxiv.org/pdf/2601.10079)**

> **作者:** Sijia Luo; Xiaokang Zhang; Yuxuan Hu; Bohan Zhang; Ke Wang; Jinbo Su; Mengshu Sun; Lei Liang; Jing Zhang
>
> **摘要:** Reinforcement Learning (RL) has become essential for eliciting complex reasoning capabilities in Large Language Models (LLMs). However, the substantial memory overhead of storing Key-Value (KV) caches during long-horizon rollouts acts as a critical bottleneck, often prohibiting efficient training on limited hardware. While existing KV compression techniques offer a remedy for inference, directly applying them to RL training induces a severe policy mismatch, leading to catastrophic performance collapse. To address this, we introduce Sparse-RL empowers stable RL training under sparse rollouts. We show that instability arises from a fundamental policy mismatch among the dense old policy, the sparse sampler policy, and the learner policy. To mitigate this issue, Sparse-RL incorporates Sparsity-Aware Rejection Sampling and Importance-based Reweighting to correct the off-policy bias introduced by compression-induced information loss. Experimental results show that Sparse-RL reduces rollout overhead compared to dense baselines while preserving the performance. Furthermore, Sparse-RL inherently implements sparsity-aware training, significantly enhancing model robustness during sparse inference deployment. The corresponding training data and code are publicly available on the repository.
>
---
#### [replaced 081] Automatic Analysis of Collaboration Through Human Conversational Data Resources: A Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的协作分析任务，旨在利用对话数据自动分析协作过程。论文回顾了相关理论、编码方案和建模方法，探索如何通过任务导向的对话数据进行协作分析。**

- **链接: [https://arxiv.org/pdf/2603.19292](https://arxiv.org/pdf/2603.19292)**

> **作者:** Yi Yu; Maria Boritchev; Chloé Clavel
>
> **备注:** 9 pages
>
> **摘要:** Collaboration is a task-oriented, high-level human behavior. In most cases, conversation serves as the primary medium for information exchange and coordination, making conversational data a valuable resource for the automatic analysis of collaborative processes. In this paper, we focus on verbal aspects of collaboration and conduct a review of collaboration analysis using task-oriented conversation resources, encompassing related theories, coding schemes, tasks, and modeling approaches. We aim to address the question of how to utilize task-oriented human-human conversational data for collaboration analysis. We hope our review will serve as a practical resource and illuminate unexplored areas for future collaboration analysis.
>
---
#### [replaced 082] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态任务，旨在解决Omni LLMs中的跨模态幻觉问题。提出MoD-DPO框架，通过模态解耦优化提升模型对不同模态的准确感知和抗幻觉能力。**

- **链接: [https://arxiv.org/pdf/2603.03192](https://arxiv.org/pdf/2603.03192)**

> **作者:** Ashutosh Chaubey; Jiacheng Pang; Mohammad Soleymani
>
> **备注:** CVPR 2026. Project Page: this https URL
>
> **摘要:** Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models.
>
---
#### [replaced 083] A Systematic Study of In-the-Wild Model Merging for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型的模型融合任务，旨在评估不同融合方法在异构专家情况下的效果，解决如何有效合并具有重叠或冲突目标的模型问题。**

- **链接: [https://arxiv.org/pdf/2511.21437](https://arxiv.org/pdf/2511.21437)**

> **作者:** Oğuz Kağan Hitit; Leander Girrbach; Zeynep Akata
>
> **摘要:** Model merging combines multiple fine-tuned checkpoints into a single model without additional training, offering an attractive approach to reusing models and efficiently improving performance. However, it remains unclear whether the advantages reported for settings where all merged experts have distinct roles and are tuned on clearly separated tasks also hold in settings where the merged experts do not have clearly distinct roles, but are trained on overlapping or even conflicting objectives. To evaluate this setting, we present a large-scale, systematic evaluation of "in-the-wild" model merging of heterogeneous experts, that may have been trained on overlapping or conflicting objectives. Concretely, we evaluate six state-of-the-art merging methods, including recent subspace methods, across four open-weight LLMs, twelve fine-tuned checkpoints per base model, and sixteen standard LLM benchmarks. Evaluating through standardized benchmarks, we measure both the probability that a model merged from a heterogeneous set of experts outperforms the base model and we measure relative gains over the best individual checkpoint. Our results show that the oldest and simplest method, Task Arithmetic, is the only approach that reliably yields performance gains on LLMs in this "in-the-wild" setting. Other interference-aware and subspace merging methods typically do not result in notable improvements over the base model. Our findings indicate that current merging techniques mostly do not enable extracting useful weight updates from heterogeneous and potentially conflicting versions. This motivates the design of LLM-specific merging algorithms and merging-aware fine-tuning methods.
>
---
#### [replaced 084] Cultural Biases of Large Language Models and Humans in Historical Interpretation
- **分类: cs.CL**

- **简介: 该论文比较人类与大语言模型在历史解释中的文化偏见，属于自然语言处理与数字人文交叉任务，旨在分析模型与人类在历史文本解读上的差异与共识。**

- **链接: [https://arxiv.org/pdf/2504.02572](https://arxiv.org/pdf/2504.02572)**

> **作者:** Fabio Celli; Georgios Spathulas
>
> **摘要:** This paper compares historical annotations by humans and Large Language Models. The findings reveal that both exhibit some cultural bias, but Large Language Models achieve a higher consensus on the interpretation of historical facts from short texts. While humans tend to disagree on the basis of their personal biases, Large Models disagree when they skip information or produce hallucinations. These findings have significant implications for digital humanities, enabling large-scale annotation and quantitative analysis of historical data. This offers new educational and research opportunities to explore historical interpretations from different Language Models, fostering critical thinking about bias.
>
---
#### [replaced 085] GreekMMLU: A Native-Sourced Multitask Benchmark for Evaluating Language Models in Greek
- **分类: cs.CL**

- **简介: 该论文提出希腊语多任务评估基准GreekMMLU，解决希腊语语言模型评估不足的问题。工作包括构建本土化数据集并分析模型性能影响因素。**

- **链接: [https://arxiv.org/pdf/2602.05150](https://arxiv.org/pdf/2602.05150)**

> **作者:** Yang Zhang; Mersin Konomi; Christos Xypolopoulos; Konstantinos Divriotis; Konstantinos Skianis; Giannis Nikolentzos; Giorgos Stamou; Guokan Shang; Michalis Vazirgiannis
>
> **摘要:** Large Language Models (LLMs) are commonly trained on multilingual corpora that include Greek, yet reliable evaluation benchmarks for Greek-particularly those based on authentic, native-sourced content-remain limited. Existing datasets are often machine-translated from English, failing to capture Greek linguistic and cultural characteristics. We introduce GreekMMLU, a native-sourced benchmark for massive multitask language understanding in Greek, comprising 21,805 multiple-choice questions across 45 subject areas, organized under a newly defined subject taxonomy and annotated with educational difficulty levels spanning primary to professional examinations. All questions are sourced or authored in Greek from academic, professional, and governmental exams. We publicly release 16,857 samples and reserve 4,948 samples for a private leaderboard to enable robust and contamination-resistant evaluation. Evaluations of over 80 open- and closed-source LLMs reveal substantial performance gaps between frontier and open-weight models, as well as between Greek-adapted models and general multilingual ones. Finally, we provide a systematic analysis of factors influencing performance-including model scale, adaptation, and prompting-and derive insights for improving LLM capabilities in Greek.
>
---
#### [replaced 086] Image Generation Models: A Technical History
- **分类: cs.CV; cs.AI; cs.CL; cs.GR**

- **简介: 该论文属于图像生成任务，旨在系统梳理各类生成模型及其发展，解决模型碎片化问题，综述模型原理、优化方法及应用挑战。**

- **链接: [https://arxiv.org/pdf/2603.07455](https://arxiv.org/pdf/2603.07455)**

> **作者:** Rouzbeh Shirvani
>
> **摘要:** Image generation has advanced rapidly over the past decade, yet the literature seems fragmented across different models and application domains. This paper aims to offer a comprehensive survey of breakthrough image generation models, including variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, autoregressive and transformer-based generators, and diffusion-based methods. We provide a detailed technical walkthrough of each model type, including their underlying objectives, architectural building blocks, and algorithmic training steps. For each model type, we present the optimization techniques as well as common failure modes and limitations. We also go over recent developments in video generation and present the research works that made it possible to go from still frames to high quality videos. Lastly, we cover the growing importance of robustness and responsible deployment of these models, including deepfake risks, detection, artifacts, and watermarking.
>
---
#### [replaced 087] Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医疗文本分类任务，解决隐私保护下的多异常分类问题。提出DP-LoRA框架，在保证隐私的前提下高效微调大模型。**

- **链接: [https://arxiv.org/pdf/2506.04450](https://arxiv.org/pdf/2506.04450)**

> **作者:** Payel Bhattacharjee; Fengwei Tian; Geoffrey D. Rubin; Joseph Y. Lo; Nirav Merchant; Heidi Hanson; John Gounley; Ravi Tandon
>
> **备注:** Accepted in IEEE ACCESS, 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly adopted across domains such as education, healthcare, and finance. In healthcare, LLMs support tasks including disease diagnosis, abnormality classification, and clinical decision-making. Among these, multi-abnormality classification of radiology reports is critical for clinical workflow automation and biomedical research. Leveraging strong natural language processing capabilities, LLMs enable efficient processing of unstructured medical text and reduce the administrative burden of manual report analysis. To improve performance, LLMs are often fine-tuned on private, institution-specific datasets such as radiology reports. However, this raises significant privacy concerns: LLMs may memorize training data and become vulnerable to data extraction attacks, while sharing fine-tuned models risks exposing sensitive patient information. Despite growing interest in LLMs for medical text classification, privacy-preserving fine-tuning for multi-abnormality classification remains underexplored. To address this gap, we propose a differentially private (DP) fine-tuning framework for multi-abnormality classification from free-text radiology reports. Our approach integrates differential privacy with Low-Rank Adaptation (LoRA) to efficiently fine-tune LLMs on sensitive clinical data while mitigating leakage risks. We further employ labels generated by a larger LLM to train smaller models, enabling efficient inference under strong privacy guarantees. Experiments on MIMIC-CXR and CT-RATE demonstrate the effectiveness of our DP-LoRA framework across varying privacy regimes. On MIMIC-CXR, our method achieves weighted F1-scores up to 0.89 under moderate privacy budgets, approaching non-private LoRA (0.90) and full fine-tuning (0.96), confirming that strong privacy can be achieved with only modest performance trade-offs.
>
---
#### [replaced 088] ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于同时机器翻译任务，解决解码效率与位置一致性矛盾的问题。提出ExPosST框架，通过显式位置分配实现高效解码和广泛兼容。**

- **链接: [https://arxiv.org/pdf/2603.14903](https://arxiv.org/pdf/2603.14903)**

> **作者:** Yuzhe Shang; Pengzhi Gao; Yazheng Yang; Jiayao Ma; Wei Liu; Jian Luan; Jinsong Su
>
> **摘要:** Large language models (LLMs) have recently demonstrated promising performance in simultaneous machine translation (SimulMT). However, applying decoder-only LLMs to SimulMT introduces a positional mismatch, which leads to a dilemma between decoding efficiency and positional consistency. Existing approaches often rely on specific positional encodings or carefully designed prompting schemes, and thus fail to simultaneously achieve inference efficiency, positional consistency, and broad model compatibility. In this work, we propose ExPosST, a general framework that resolves this dilemma through explicit position allocation. ExPosST reserves fixed positional slots for incoming source tokens, enabling efficient decoding with KV cache across different positional encoding methods. To further bridge the gap between fine-tuning and inference, we introduce a policy-consistent fine-tuning strategy that aligns training with inference-time decoding behavior. Experiments across multiple language pairs demonstrate that ExPosST effectively supports simultaneous translation under diverse policies.
>
---
#### [replaced 089] Multilingual Medical Reasoning for Question Answering with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学问答任务，旨在解决多语言医疗知识推理问题。通过从维基百科提取医疗知识生成多语言推理轨迹，提升模型在医学QA中的表现。**

- **链接: [https://arxiv.org/pdf/2512.05658](https://arxiv.org/pdf/2512.05658)**

> **作者:** Pietro Ferrazzi; Aitor Soroa; Rodrigo Agerri
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) with reasoning capabilities have recently demonstrated strong potential in medical Question Answering (QA). Existing approaches are largely English-focused and primarily rely on distillation from general-purpose LLMs, raising concerns about the reliability of their medical knowledge. In this work, we present a method to generate multilingual reasoning traces based on medical knowledge extracted from Wikipedia. We produce 500k traces in English, Italian, and Spanish, using a retrieval-augmented generation approach over medical information from Wikipedia. The traces are generated to solve medical questions drawn from MedQA and MedMCQA, which we extend to Italian and Spanish. We test our pipeline in both in-domain and out-of-domain settings across Medical QA benchmarks, and demonstrate that our reasoning traces improve performance both when utilized via in-context learning (few-shot) and supervised fine-tuning, yielding state-of-the-art results among 8B-parameter LLMs. We believe that these resources can support the development of more transparent clinical decision-support tools in multilingual settings. We release the full suite of resources: reasoning traces, translated QA datasets, Medical-Wikipedia, and fine-tuned models.
>
---
#### [replaced 090] X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音大模型对齐任务，旨在解决语音LLM性能低于文本模型的问题。通过X-OPD框架，利用文本教师模型指导语音学生模型，提升其能力。**

- **链接: [https://arxiv.org/pdf/2603.24596](https://arxiv.org/pdf/2603.24596)**

> **作者:** Di Cao; Dongjie Fu; Hai Yu; Siqi Zheng; Xu Tan; Tao Jin
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** While the shift from cascaded dialogue systems to end-to-end (E2E) speech Large Language Models (LLMs) improves latency and paralinguistic modeling, E2E models often exhibit a significant performance degradation compared to their text-based counterparts. The standard Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training methods fail to close this gap. To address this, we propose X-OPD, a novel Cross-Modal On-Policy Distillation framework designed to systematically align the capabilities of Speech LLMs to their text-based counterparts. X-OPD enables the Speech LLM to explore its own distribution via on-policy rollouts, where a text-based teacher model evaluates these trajectories and provides token-level feedback, effectively distilling teacher's capabilities into student's multi-modal representations. Extensive experiments across multiple benchmarks demonstrate that X-OPD significantly narrows the gap in complex tasks while preserving the model's inherent capabilities.
>
---
#### [replaced 091] Activation Steering for Masked Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文研究MDLM的推理控制问题，提出激活引导方法，在不改变采样过程的情况下通过干预激活实现行为调控，提升模型可控性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.24143](https://arxiv.org/pdf/2512.24143)**

> **作者:** Adi Shnaidman; Erin Feiglin; Osher Yaari; Efrat Mentel; Amit Levi; Raz Lapid
>
> **备注:** Accepted at ReALM-GEN @ ICLR 2026
>
> **摘要:** Masked diffusion language models (MDLMs) generate text via iterative masked-token denoising, enabling mask-parallel decoding and distinct controllability and efficiency tradeoffs from autoregressive LLMs. Yet, efficient representation-level mechanisms for inference-time control in MDLMs remain largely unexplored. To address this gap, we introduce an activation steering primitive for MDLMs: we extract a single low-dimensional direction from contrastive prompt sets using one prompt-only forward pass, and apply a global intervention on residual-stream activations throughout reverse diffusion, without performing optimization or altering the diffusion sampling procedure. Using safety refusal as a deployment-relevant case study, we find that refusal behavior in multiple MDLMs is governed by a consistent, approximately one-dimensional activation subspace. Applying the corresponding direction yields large and systematic behavioral shifts and is substantially more effective than prompt-based and optimization-based baselines. We further uncover diffusion-specific accessibility: effective directions can be extracted not only from post-instruction tokens, but also from pre-instruction tokens that are typically ineffective in autoregressive models due to causal attention. Ablations localize maximal leverage to early denoising steps and mid-to-late transformer layers, with early diffusion blocks contributing disproportionately. Finally, in an MDLM trained on English and Chinese, extracted directions transfer strongly between English and Chinese, but do not reliably generalize to an autoregressive architecture, highlighting architecture-dependent representations of safety constraints.
>
---
#### [replaced 092] Schema for In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Schema-Activated In-Context Learning（SA-ICL），解决传统ICL缺乏抽象知识迁移的问题，通过引入认知科学中的schema理论，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2510.13905](https://arxiv.org/pdf/2510.13905)**

> **作者:** Pan Chen; Shaohong Chen; Mark Wang; Shi Xuan Leong; Priscilla Fung; Varinia Bernales; Alan Aspuru-Guzik
>
> **摘要:** In-Context Learning (ICL) enables transformer-based language models to adapt to new tasks by conditioning on demonstration examples. However, traditional example-driven in-context learning lacks explicit modules for knowledge retrieval and transfer at the abstraction level. Inspired by cognitive science, specifically schema theory, which holds that humans interpret new information by activating pre-existing mental frameworks (schemas) to structure understanding, we introduce Schema-Activated In-Context Learning (SA-ICL). This framework extracts the representation of the building blocks of cognition for the reasoning process instilled from prior examples, creating an abstracted schema, a lightweight, structured template of key inferential steps and their relationships, which is then used to augment a model's reasoning process when presented with a novel question. We demonstrate that a broad range of large language models (LLMs) lack the capacity to form and utilize internal schema-based learning representations implicitly, but instead benefit significantly from explicit schema-based scaffolding. Across chemistry and physics questions from the GPQA dataset, our experiments show that SA-ICL consistently boosts performance, up to 36.19 percent, when the single demonstration example is of high quality, which simultaneously reduces reliance on the number of demonstrations and enhances interpretability. Schema-Activated In-Context Learning not only bridges disparate ICL strategies ranging from pattern priming to Chain-of-Thought prompting, but also paves a new path for enhancing human-like reasoning in LLMs.
>
---
#### [replaced 093] Person-Centric Annotations of LAION-400M: Auditing Bias and Its Transfer to Models
- **分类: cs.CV; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于视觉-语言模型的偏见分析任务，旨在解决数据集偏差对模型影响的问题。通过为LAION-400M添加人物级标注，揭示数据中的种族和性别不平衡，并验证其对模型偏见的影响。**

- **链接: [https://arxiv.org/pdf/2510.03721](https://arxiv.org/pdf/2510.03721)**

> **作者:** Leander Girrbach; Stephan Alaniz; Genevieve Smith; Trevor Darrell; Zeynep Akata
>
> **备注:** ICLR 2026
>
> **摘要:** Vision-language models trained on large-scale multimodal datasets show strong demographic biases, but the role of training data in producing these biases remains unclear. A major barrier has been the lack of demographic annotations in web-scale datasets such as LAION-400M. We address this gap by creating person-centric annotations for the full dataset, including over 276 million bounding boxes, perceived gender and race/ethnicity labels, and automatically generated captions. These annotations are produced through validated automatic labeling pipelines combining object detection, multimodal captioning, and finetuned classifiers. Using them, we uncover demographic imbalances and harmful associations, such as the disproportionate linking of men and individuals perceived as Black or Middle Eastern with crime-related and negative content. We also show that a linear fit predicts 60-70% of gender bias in CLIP and Stable Diffusion from direct co-occurrences in the data. Our resources establish the first large-scale empirical link between dataset composition and downstream model bias. Code is available at this https URL.
>
---
#### [replaced 094] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ACE框架，解决大语言模型上下文适应中的简洁偏差和上下文坍塌问题，通过演化上下文提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2510.04618](https://arxiv.org/pdf/2510.04618)**

> **作者:** Qizheng Zhang; Changran Hu; Shubhangi Upasani; Boyuan Ma; Fenglu Hong; Vamsidhar Kamanuru; Jay Rainton; Chen Wu; Mengmeng Ji; Hanchen Li; Urmish Thakker; James Zou; Kunle Olukotun
>
> **备注:** ICLR 2026; 32 pages
>
> **摘要:** Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation: modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. We introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.
>
---
