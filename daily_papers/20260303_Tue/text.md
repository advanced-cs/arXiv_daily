# 自然语言处理 cs.CL

- **最新发布 167 篇**

- **更新 175 篇**

## 最新发布

#### [new 001] When Numbers Tell Half the Story: Human-Metric Alignment in Topic Model Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于主题模型评估任务，旨在解决自动化指标与人类判断不一致的问题。通过引入TWM任务，评估主题间区分度，结合人工与自动方法提升领域特定文本的评价准确性。**

- **链接: [https://arxiv.org/pdf/2603.01945](https://arxiv.org/pdf/2603.01945)**

> **作者:** Thibault Prouteau; Francis Lareau; Nicolas Dugué; Jean-Charles Lamirel; Christophe Malaterre
>
> **摘要:** Topic models uncover latent thematic structures in text corpora, yet evaluating their quality remains challenging, particularly in specialized domains. Existing methods often rely on automated metrics like topic coherence and diversity, which may not fully align with human judgment. Human evaluation tasks, such as word intrusion, provide valuable insights but are costly and primarily validated on general-domain corpora. This paper introduces Topic Word Mixing (TWM), a novel human evaluation task assessing inter-topic distinctness by testing whether annotators can distinguish between word sets from single or mixed topics. TWM complements word intrusion's focus on intra-topic coherence and provides a human-grounded counterpart to diversity metrics. We evaluate six topic models - both statistical and embedding-based (LDA, NMF, Top2Vec, BERTopic, CFMF, CFMF-emb) - comparing automated metrics with human evaluation methods based on nearly 4,000 annotations from a domain-specific corpus of philosophy of science publications. Our findings reveal that word intrusion and coherence metrics do not always align, particularly in specialized domains, and that TWM captures human-perceived distinctness while appearing to align with diversity metrics. We release the annotated dataset and task generation code. This work highlights the need for evaluation frameworks bridging automated and human assessments, particularly for domain-specific corpora.
>
---
#### [new 002] Prompt Sensitivity and Answer Consistency of Small Open-Source Large Language Models on Clinical Question Answering: Implications for Low-Resource Healthcare Deployment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床问答任务，研究小模型在不同提示下的敏感性和一致性，评估其可靠性与准确性，以优化低资源医疗部署。**

- **链接: [https://arxiv.org/pdf/2603.00917](https://arxiv.org/pdf/2603.00917)**

> **作者:** Shravani Hariprasad
>
> **备注:** 30 pages, 7 figures, 2 tables
>
> **摘要:** Small open-source language models are gaining attention for low-resource healthcare settings, but their reliability under different prompt phrasings remains poorly understood. We evaluated five open-source models (Gemma 2 2B, Phi-3 Mini 3.8B, Llama 3.2 3B, Mistral 7B, and Meditron-7B domain-pretrained without instruction tuning) across three clinical QA datasets (MedQA, MedMCQA, PubMedQA) using five prompt styles (original, formal, simplified, roleplay, direct). We measured consistency scores, accuracy, and instruction-following failure rates. All inference ran locally on consumer CPU hardware without fine-tuning. Consistency and accuracy were largely independent. Gemma 2 achieved the highest consistency (0.845-0.888) but lowest accuracy (33.0-43.5%), while Llama 3.2 showed moderate consistency (0.774-0.807) with the highest accuracy (49.0-65.0%). Roleplay prompts consistently reduced accuracy across all models, with Phi-3 Mini dropping 21.5 percentage points on MedQA. Meditron-7B exhibited near-complete instruction-following failure on PubMedQA (99.0% UNKNOWN rate), showing domain pretraining alone is insufficient for structured clinical QA. High consistency does not imply correctness. Models can be reliably wrong, a dangerous failure mode in clinical AI. Roleplay prompts should be avoided in healthcare applications. Llama 3.2 showed the strongest balance of accuracy and reliability for low-resource deployment. Safe clinical AI requires joint evaluation of consistency, accuracy, and instruction adherence.
>
---
#### [new 003] Generative AI & Fictionality: How Novels Power Large Language Models
- **分类: cs.CL**

- **简介: 论文探讨生成式AI如何受小说等虚构文本影响，分析其对模型输出的作用。属于自然语言处理任务，旨在理解虚构文本在训练数据中的角色及其与其他文本类型的差异。**

- **链接: [https://arxiv.org/pdf/2603.01220](https://arxiv.org/pdf/2603.01220)**

> **作者:** Edwin Roland; Richard Jean So
>
> **摘要:** Generative models, like the one in ChatGPT, are powered by their training data. The models are simply next-word predictors, based on patterns learned from vast amounts of pre-existing text. Since the first generation of GPT, it is striking that the most popular datasets have included substantial collections of novels. For the engineers and research scientists who build these models, there is a common belief that the language in fiction is rich enough to cover all manner of social and communicative phenomena, yet the belief has gone mostly unexamined. How does fiction shape the outputs of generative AI? Specifically, what are novels' effects relative to other forms of text, such as newspapers, Reddit, and Wikipedia? Since the 1970s, literature scholars such as Catherine Gallagher and James Phelan have developed robust and insightful accounts of how fiction operates as a form of discourse and language. Through our study of an influential open-source model (BERT), we find that LLMs leverage familiar attributes and affordances of fiction, while also fomenting new qualities and forms of social response. We argue that if contemporary culture is increasingly shaped by generative AI and machine learning, any analysis of today's various modes of cultural production must account for a relatively novel dimension: computational training data.
>
---
#### [new 004] Towards Orthographically-Informed Evaluation of Speech Recognition Systems for Indian Languages
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别评估任务，解决印度语言ASR系统评价不准确的问题，通过引入考虑拼写变体的OIWER指标，提升评估与实际表现的一致性。**

- **链接: [https://arxiv.org/pdf/2603.00941](https://arxiv.org/pdf/2603.00941)**

> **作者:** Kaushal Santosh Bhogale; Tahir Javed; Greeshma Susan John; Dhruv Rathi; Akshayasree Padmanaban; Niharika Parasa; Mitesh M. Khapra
>
> **备注:** Accepted in ICASSP 2026
>
> **摘要:** Evaluating ASR systems for Indian languages is challenging due to spelling variations, suffix splitting flexibility, and non-standard spellings in code-mixed words. Traditional Word Error Rate (WER) often presents a bleaker picture of system performance than what human users perceive. Better aligning evaluation with real-world performance requires capturing permissible orthographic variations, which is extremely challenging for under-resourced Indian languages. Leveraging recent advances in LLMs, we propose a framework for creating benchmarks that capture permissible variations. Through extensive experiments, we demonstrate that OIWER, by accounting for orthographic variations, reduces pessimistic error rates (an average improvement of 6.3 points), narrows inflated model gaps (e.g., Gemini-Canary performance difference drops from 18.1 to 11.5 points), and aligns more closely with human perception than prior methods like WER-SN by 4.9 points.
>
---
#### [new 005] Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，旨在解决推测解码中静态时间分配效率低的问题。通过强化学习动态协调 drafting 和 verification 过程，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2603.01639](https://arxiv.org/pdf/2603.01639)**

> **作者:** Jiebin Zhang; Zhenghan Yu; Liang Wang; Nan Yang; Eugene J. Yu; Zheng Li; Yifan Song; Dawei Zhu; Xingxing Zhang; Furu Wei; Sujian Li
>
> **备注:** 22pages, 7 figures
>
> **摘要:** Speculative decoding accelerates large language model (LLM) inference by using a small draft model to generate candidate tokens for a larger target model to verify. The efficacy of this technique hinges on the trade-off between the time spent on drafting candidates and verifying them. However, current state-of-the-art methods rely on a static time allocation, while recent dynamic approaches optimize for proxy metrics like acceptance length, often neglecting the true time cost and treating the drafting and verification phases in isolation. To address these limitations, we introduce Learning to Draft (LTD), a novel method that directly optimizes for throughput of each draft-and-verify cycle. We formulate the problem as a reinforcement learning environment and train two co-adaptive policies to dynamically coordinate the draft and verification phases. This encourages the policies to adapt to each other and explicitly maximize decoding efficiency. We conducted extensive evaluations on five diverse LLMs and four distinct tasks. Our results show that LTD achieves speedup ratios ranging from 2.24x to 4.32x, outperforming the state-of-the-art method Eagle3 up to 36.4%.
>
---
#### [new 006] Reasoning or Rationalization? The Role of Justifications in Masked Diffusion Models for Fact Verification
- **分类: cs.CL**

- **简介: 该论文研究掩码扩散语言模型在事实验证任务中的推理机制，探讨其是否真正推理或仅事后合理化。工作包括分析模型动态及干预实验。**

- **链接: [https://arxiv.org/pdf/2603.01190](https://arxiv.org/pdf/2603.01190)**

> **作者:** Jacob Devasier
>
> **摘要:** Unlike autoregressive models, which generate tokens sequentially and benefit from reasoning-before-answering strategies such as Chain-of-Thought, Masked Diffusion Language Models (MDLMs) refine all sequence positions simultaneously, raising questions about how these models handle tasks requiring justified verdicts. In this work, we investigate the dynamics of MDLM reasoning on fact verification, examining whether justifications serve as genuine reasoning or post-hoc rationalization. We observe that MDLMs typically converge on a verdict early in the diffusion process, treating it as a global anchor that is resolved before the justification is complete. Crucially, enforcing a reasoning-first constraint via delayed verdict unmasking actively degrades performance, dropping accuracy from 86.2% to 71.9% as accumulating justification tokens introduce inconsistencies that override initially correct predictions. Interventional experiments reveal that the model rationalizes incorrect forced verdicts in 56% of cases, and that verdicts are strongly causally dependent on justification quality (57.3% accuracy with corrupted justifications vs. 97.1% with ground-truth). This causal dependence explains the degradation under forced deliberation: as the model generates noisy justification tokens, it conditions on them, gradually overriding its initially correct assessment. Our findings suggest that for fact verification with MDLMs, extended deliberation can be counterproductive, risking the dilution of accurate early predictions with noise introduced during justification generation.
>
---
#### [new 007] Conformal Prediction for Risk-Controlled Medical Entity Extraction Across Clinical Domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗实体抽取任务，解决LLM置信度不准确的问题。通过构建置信区间框架，在不同临床领域实现高覆盖率的可靠抽取。**

- **链接: [https://arxiv.org/pdf/2603.00924](https://arxiv.org/pdf/2603.00924)**

> **作者:** Manil Shrestha; Edward Kim
>
> **摘要:** Large Language Models (LLMs) are increasingly used for medical entity extraction, yet their confidence scores are often miscalibrated, limiting safe deployment in clinical settings. We present a conformal prediction framework that provides finite-sample coverage guarantees for LLM-based extraction across two clinical domains. First, we extract structured entities from 1,000 FDA drug labels across eight sections using GPT-4.1, verified via FactScore-based atomic statement evaluation (97.7\% accuracy over 128,906 entities). Second, we extract radiological entities from MIMIC-CXR reports using the RadGraph schema with GPT-4.1 and Llama-4-Maverick, evaluated against physician annotations (entity F1: 0.81 to 0.84). Our central finding is that miscalibration direction reverses across domains: on well-structured FDA labels, models are underconfident, requiring modest conformal thresholds ($\tau \approx 0.06$), while on free-text radiology reports, models are overconfident, demanding strict thresholds ($\tau$ up to 0.99). Despite this heterogeneity, conformal prediction achieves target coverage ($\geq 90\%$) in both settings with manageable rejection rates (9--13\%). These results demonstrate that calibration is not a global model property but depends on document structure, extraction category, and model architecture, motivating domain-specific conformal calibration for safe clinical deployment.
>
---
#### [new 008] QIME: Constructing Interpretable Medical Text Embeddings via Ontology-Grounded Questions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出QIME，用于构建可解释的医学文本嵌入。解决医学文本嵌入不可解释的问题，通过本体论引导的问答方式，提升临床实用性。**

- **链接: [https://arxiv.org/pdf/2603.01690](https://arxiv.org/pdf/2603.01690)**

> **作者:** Yixuan Tang; Zhenghong Lin; Yandong Sun; Anthony K.H. Tung
>
> **摘要:** While dense biomedical embeddings achieve strong performance, their black-box nature limits their utility in clinical decision-making. Recent question-based interpretable embeddings represent text as binary answers to natural-language questions, but these approaches often rely on heuristic or surface-level contrastive signals and overlook specialized domain knowledge. We propose QIME, an ontology-grounded framework for constructing interpretable medical text embeddings in which each dimension corresponds to a clinically meaningful yes/no question. By conditioning on cluster-specific medical concept signatures, QIME generates semantically atomic questions that capture fine-grained distinctions in biomedical text. Furthermore, QIME supports a training-free embedding construction strategy that eliminates per-question classifier training while further improving performance. Experiments across biomedical semantic similarity, clustering, and retrieval benchmarks show that QIME consistently outperforms prior interpretable embedding methods and substantially narrows the gap to strong black-box biomedical encoders, while providing concise and clinically informative explanations.
>
---
#### [new 009] Beyond the Resumé: A Rubric-Aware Automatic Interview System for Information Elicitation
- **分类: cs.CL**

- **简介: 该论文属于招聘优化任务，旨在解决传统面试成本高、信息有限的问题。通过使用大语言模型模拟面试官，系统能更精准地评估候选人特质，提升早期招聘决策质量。**

- **链接: [https://arxiv.org/pdf/2603.01775](https://arxiv.org/pdf/2603.01775)**

> **作者:** Harry Stuart; Masahiro Kaneko; Timothy Baldwin
>
> **摘要:** Effective hiring is integral to the success of an organisation, but it is very challenging to find the most suitable candidates because expert evaluation (e.g.\ interviews conducted by a technical manager) are expensive to deploy at scale. Therefore, automated resume scoring and other applicant-screening methods are increasingly used to coarsely filter candidates, making decisions on limited information. We propose that large language models (LLMs) can play the role of subject matter experts to cost-effectively elicit information from each candidate that is nuanced and role-specific, thereby improving the quality of early-stage hiring decisions. We present a system that leverages an LLM interviewer to update belief over an applicant's rubric-oriented latent traits in a calibrated way. We evaluate our system on simulated interviews and show that belief converges towards the simulated applicants' artificially-constructed latent ability levels. We release code, a modest dataset of public-domain/anonymised resumes, belief calibration tests, and simulated interviews, at \href{this https URL}{this https URL}. Our demo is available at \href{this https URL}{this https URL}.
>
---
#### [new 010] A Study on Building Efficient Zero-Shot Relation Extraction Models
- **分类: cs.CL**

- **简介: 该论文属于零样本关系抽取任务，旨在解决模型在真实场景下的鲁棒性问题。研究提出改进策略，评估模型在无标注数据和无关输入下的表现，发现现有方法效果有限，其中AlignRE表现最佳。**

- **链接: [https://arxiv.org/pdf/2603.01266](https://arxiv.org/pdf/2603.01266)**

> **作者:** Hugo Thomas; Caio Corro; Guillaume Gravier; Pascale Sébillot
>
> **备注:** LREC 2026
>
> **摘要:** Zero-shot relation extraction aims to identify relations between entity mentions using textual descriptions of novel types (i.e., previously unseen) instead of labeled training examples. Previous works often rely on unrealistic assumptions: (1) pairs of mentions are often encoded directly in the input, which prevents offline pre-computation for large scale document database querying; (2) no rejection mechanism is introduced, biasing the evaluation when using these models in a retrieval scenario where some (and often most) inputs are irrelevant and must be ignored. In this work, we study the robustness of existing zero-shot relation extraction models when adapting them to a realistic extraction scenario. To this end, we introduce a typology of existing models, and propose several strategies to build single pass models and models with a rejection mechanism. We adapt several state-of-the-art tools, and compare them in this challenging setting, showing that no existing work is really robust to realistic assumptions, but overall AlignRE (Li et al., 2024) performs best along all criteria.
>
---
#### [new 011] Bootstrapping Embeddings for Low Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言嵌入模型训练数据不足的问题。通过生成合成三元组数据，利用适配器组合和跨语言微调方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.01732](https://arxiv.org/pdf/2603.01732)**

> **作者:** Merve Basoz; Andrew Horne; Mattia Opper
>
> **备注:** (v1 - LowResLM Camera Ready)
>
> **摘要:** Embedding models are crucial to modern NLP. However, the creation of the most effective models relies on carefully constructed supervised finetuning data. For high resource languages, such as English, such datasets are readily available. However, for hundreds of other languages, they are simply non-existent. We investigate whether the advent of large language models can help to bridge this gap. We test three different strategies for generating synthetic triplet data used to optimise embedding models. These include in-context learning as well as two novel approaches, leveraging adapter composition and cross lingual finetuning of the LLM generator (XL-LoRA) respectively. We find that while in-context learning still falls short of strong non-synthetic baselines, adapter composition and XL-LoRA yield strong performance gains across a wide array of tasks and languages, offering a clear, scalable pathway to producing performant embedding models for a wide variety of languages.
>
---
#### [new 012] EPPCMinerBen: A Novel Benchmark for Evaluating Large Language Models on Electronic Patient-Provider Communication via the Patient Portal
- **分类: cs.CL**

- **简介: 该论文提出EPPCMinerBen基准，用于评估大语言模型在电子患者-医生通信中的表现，解决通信模式识别与信息提取问题。**

- **链接: [https://arxiv.org/pdf/2603.00028](https://arxiv.org/pdf/2603.00028)**

> **作者:** Samah Fodeh; Yan Wang; Linhai Ma; Srivani Talakokkul; Jordan M. Alpert; Sarah Schellhorn
>
> **摘要:** Effective communication in health care is critical for treatment outcomes and adherence. With patient-provider exchanges shifting to secure messaging, analyzing electronic patient-communication (EPPC) data is both essential and challenging. We introduce EPPCMinerBen, a benchmark for evaluating LLMs in detecting communication patterns and extracting insights from electronic patient-provider messages. EPPCMinerBen includes three sub-tasks: Code Classification, Subcode Classification, and Evidence Extraction. Using 1,933 expert annotated sentences from 752 secure messages of the patient portal at Yale New Haven Hospital, it evaluates LLMs on identifying communicative intent and supportive text. Benchmarks span various LLMs under zero-shot and few-shot settings, with data to be released via the NCI Cancer Data Service. Model performance varied across tasks and settings. Llama-3.1-70B led in evidence extraction (F1: 82.84%) and performed well in classification. Llama-3.3-70b-Instruct outperformed all models in code classification (F1: 67.03%). DeepSeek-R1-Distill-Qwen-32B excelled in subcode classification (F1: 48.25%), while sdoh-llama-3-70B showed consistent performance. Smaller models underperformed, especially in subcode classification (>30% F1). Few-shot prompting improved most tasks. Our results show that large, instruction-tuned models generally perform better in EPPCMinerBen tasks, particularly evidence extraction while smaller models struggle with fine-grained reasoning. EPPCMinerBen provides a benchmark for discourse-level understanding, supporting future work on model generalization and patient-provider communication analysis. Keywords: Electronic Patient-Provider Communication, Large language models, Data collection, Prompt engineering
>
---
#### [new 013] EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升多语言大模型中爱沙尼亚语的能力。通过持续预训练和后训练，增强模型在爱沙尼亚语上的表现，同时保持英语性能。**

- **链接: [https://arxiv.org/pdf/2603.02041](https://arxiv.org/pdf/2603.02041)**

> **作者:** Aleksei Dorkin; Taido Purason; Emil Kalbaliyev; Hele-Andra Kuulmets; Marii Ojastu; Mark Fišel; Tanel Alumäe; Eleri Aedmaa; Krister Kruusmaa; Kairit Sirts
>
> **摘要:** Large language models (LLMs) are predominantly trained on English-centric data, resulting in uneven performance for smaller languages. We study whether continued pretraining (CPT) can substantially improve Estonian capabilities in a pretrained multilingual LLM while preserving its English and general reasoning performance. Using Llama 3.1 8B as the main base model, we perform CPT on a mixture that increases Estonian exposure while approximating the original training distribution through English replay and the inclusion of code, mathematics, and instruction-like data. We subsequently apply supervised fine-tuning, preference optimization, and chat vector merging to introduce robust instruction-following behavior. Evaluation on a comprehensive suite of Estonian benchmarks shows consistent gains in linguistic competence, knowledge, reasoning, translation quality, and instruction-following compared to the original base model and its instruction-tuned variant, while maintaining competitive performance on English benchmarks. These findings indicate that CPT, with an appropriately balanced data mixture, together with post-training alignment, can substantially improve single-language capabilities in pretrained multilingual LLMs.
>
---
#### [new 014] Policy Compliance of User Requests in Natural Language for AI Systems
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的政策合规任务，旨在解决用户请求与组织政策的匹配问题。工作包括构建首个相关基准数据集，并评估模型在不同方法下的合规性判断性能。**

- **链接: [https://arxiv.org/pdf/2603.00369](https://arxiv.org/pdf/2603.00369)**

> **作者:** Pedro Cisneros-Velarde
>
> **摘要:** Consider an organization whose users send requests in natural language to an AI system that fulfills them by carrying out specific tasks. In this paper, we consider the problem of ensuring such user requests comply with a list of diverse policies determined by the organization with the purpose of guaranteeing the safe and reliable use of the AI system. We propose, to the best of our knowledge, the first benchmark consisting of annotated user requests of diverse compliance with respect to a list of policies. Our benchmark is related to industrial applications in the technology sector. We then use our benchmark to evaluate the performance of various LLM models on policy compliance assessment under different solution methods. We analyze the differences on performance metrics across the models and solution methods, showcasing the challenging nature of our problem.
>
---
#### [new 015] KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，解决KV缓存计算与内存消耗过高的问题。通过理论分析和算法设计，提出KVSlimmer方法，实现高效压缩与推理。**

- **链接: [https://arxiv.org/pdf/2603.00907](https://arxiv.org/pdf/2603.00907)**

> **作者:** Lianjun Liu; Hongli An; Weiqi Yan; Xin Du; Shengchuan Zhang; Huazhong Liu; Yunshan Zhong
>
> **摘要:** The growing computational and memory demands of the Key-Value (KV) cache significantly limit the ability of Large Language Models (LLMs). While KV merging has emerged as a promising solution, existing methods that rely on empirical observations of KV asymmetry and gradient-based Hessian approximations lack a theoretical foundation and incur suboptimal compression and inference overhead. To bridge these gaps, we establish a theoretical framework that characterizes this asymmetry through the spectral energy distribution of projection weights, demonstrating that concentrated spectra in Query/Key weights induce feature homogeneity, whereas dispersed spectra in Value weights preserve heterogeneity. Then, we introduce KVSlimmer, an efficient algorithm that captures exact Hessian information through a mathematically exact formulation, and derives a closed-form solution utilizing only forward-pass variables, resulting in a gradient-free approach that is both memory- and time-efficient. Extensive experiments across various models and benchmarks demonstrate that KVSlimmer consistently outperforms SOTA methods. For instance, on Llama3.1-8B-Instruct, it improves the LongBench average score by 0.92 while reducing memory costs and latency by 29% and 28%, respectively.
>
---
#### [new 016] How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning
- **分类: cs.CL**

- **简介: 该论文属于几何推理任务，旨在解决多模态大模型在交替绘图与推理中的性能下降问题。通过引入强化学习框架Faire，提升模型对因果关系的把握。**

- **链接: [https://arxiv.org/pdf/2603.01070](https://arxiv.org/pdf/2603.01070)**

> **作者:** Xiangxiang Zhang; Caijun Jia; Siyuan Li; Dingyu He; Xiya Xiong; Zheng Sun; Honghao He; Yuchen Wu; Bihui Yu; Linzhuang Sun; Cheng Tan; Jingxuan Wei
>
> **摘要:** Solving complex geometric problems inherently requires interleaved reasoning: a tight alternation between constructing diagrams and performing logical deductions. Although recent Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in visual generation and plotting, we identify a counter-intuitive and underexplored phenomenon. Naively applying Supervised Fine-Tuning (SFT) on interleaved plot-solution data leads to a substantial degradation in reasoning performance compared to text-only baselines. We argue that this failure stems from a fundamental limitation of SFT, which primarily induces distributional alignment: the model learns to reproduce the surface format of interleaved plotting but fails to internalize the causal dependency between the generated plot and reasoning steps. To overcome this limitation, we propose Faire (Functional alignment for interleaved reasoning), a reinforcement learning framework that enforces three casual constraints to move beyond superficial imitation toward functional alignment. Extensive experiments show that Faire induces a qualitative shift in model behavior in which the plotting is effectively internalized, yielding competitive performance on challenging geometric reasoning benchmarks.
>
---
#### [new 017] CARD: Towards Conditional Design of Multi-agent Topological Structures
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CARD框架，解决多智能体系统通信拓扑静态问题，通过动态环境信号实现拓扑自适应，提升系统效果与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.01089](https://arxiv.org/pdf/2603.01089)**

> **作者:** Tongtong Wu; Yanming Li; Ziye Tang; Chen Jiang; Linhao Luo; Guilin Qi; Shirui Pan; Gholamreza Haffari
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large language model (LLM)-based multi-agent systems have shown strong capabilities in tasks such as code generation and collaborative reasoning. However, the effectiveness and robustness of these systems critically depend on their communication topology, which is often fixed or statically learned, ignoring real-world dynamics such as model upgrades, API (or tool) changes, or knowledge source variability. To address this limitation, we propose CARD (Conditional Agentic Graph Designer), a conditional graph-generation framework that instantiates AMACP, a protocol for adaptive multi-agent communication. CARD explicitly incorporates dynamic environmental signals into graph construction, enabling topology adaptation at both training and runtime. Through a conditional variational graph encoder and environment-aware optimization, CARD produces communication structures that are both effective and resilient to shifts in model capability or resource availability. Empirical results on HumanEval, MATH, and MMLU demonstrate that CARD consistently outperforms static and prompt-based baselines, achieving higher accuracy and robustness across diverse conditions. The source code is available at: this https URL.
>
---
#### [new 018] What Exactly do Children Receive in Language Acquisition? A Case Study on CHILDES with Automated Detection of Filler-Gap Dependencies
- **分类: cs.CL**

- **简介: 该论文研究儿童语言习得中的填充-缺口结构，解决输入数据量化难题。通过自动检测语料库中的三种构造，分析儿童语言输入与产出轨迹。属于自然语言处理与语言习得交叉任务。**

- **链接: [https://arxiv.org/pdf/2603.02082](https://arxiv.org/pdf/2603.02082)**

> **作者:** Zhenghao Herbert Zhou; William Dai; Maya Viswanathan; Simon Charlow; R. Thomas McCoy; Robert Frank
>
> **摘要:** Children's acquisition of filler-gap dependencies has been argued by some to depend on innate grammatical knowledge, while others suggest that the distributional evidence available in child-directed speech suffices. Unfortunately, the relevant input is difficult to quantify at scale with fine granularity, making this question difficult to resolve. We present a system that identifies three core filler-gap constructions in spoken English corpora -- matrix wh-questions, embedded wh-questions, and relative clauses -- and further identifies the extraction site (i.e., subject vs. object vs. adjunct). Our approach combines constituency and dependency parsing, leveraging their complementary strengths for construction classification and extraction site identification. We validate the system on human-annotated data and find that it scores well across most categories. Applying the system to 57 English CHILDES corpora, we are able to characterize children's filler-gap input and their filler-gap production trajectories over the course of development, including construction-specific frequencies and extraction-site asymmetries. The resulting fine-grained labels enable future work in both acquisition and computational studies, which we demonstrate with a case study using filtered corpus training with language models.
>
---
#### [new 019] Markovian ODE-guided scoring can assess the quality of offline reasoning traces in language models
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理质量评估任务，旨在解决现有方法无法有效衡量推理轨迹质量的问题。提出MarODE框架，基于马尔可夫和微分方程模型，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2603.01580](https://arxiv.org/pdf/2603.01580)**

> **作者:** Arghodeep Nandi; Ojasva Saxena; Tanmoy Chakraborty
>
> **摘要:** Reasoning traces produced by generative language models are increasingly used for tasks ranging from mathematical problem solving to automated fact checking. However, existing evaluation methods remain largely mechanical and fail to capture human-centric notions of reasoning quality in a way that generalizes across varied and progressively degraded reasoning. We introduce MarODE, an offline evaluation framework that assigns quality scores to reasoning traces. Its effectiveness is assessed using human-centric perturbations and human judgments, which jointly evaluate the fundamental dimensions of an evaluation metric - goodness and soundness. The approach is grounded in a Markovian formulation of reasoning progression and an ordinary differential equation based characterization of trace dynamics, enabling efficient evaluation of reasoning quality. In a large-scale evaluation, MarODE outperforms existing baselines by over 250% under Somers' D correlation. Our results emphasize the value of theory-driven evaluation frameworks as reasoning traces become central to language model-based systems.
>
---
#### [new 020] Hybrid Neural-LLM Pipeline for Morphological Glossing in Endangered Language Documentation: A Case Study of Jungar Tuvan
- **分类: cs.CL**

- **简介: 该论文属于语言学标注任务，解决濒危语言的形态词素标注问题。通过结合神经网络与大语言模型，构建混合标注流程，提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.00923](https://arxiv.org/pdf/2603.00923)**

> **作者:** Siyu Liang; Talant Mawkanuli; Gina-Anne Levow
>
> **摘要:** Interlinear glossed text (IGT) creation remains a major bottleneck in linguistic documentation and fieldwork, particularly for low-resource morphologically rich languages. We present a hybrid automatic glossing pipeline that combines neural sequence labeling with large language model (LLM) post-correction, evaluated on Jungar Tuvan, a low-resource Turkic language. Through systematic ablation studies, we show that retrieval-augmented prompting provides substantial gains over random example selection. We further find that morpheme dictionaries paradoxically hurt performance compared to providing no dictionary at all in most cases, and that performance scales approximately logarithmically with the number of few-shot examples. Most significantly, our two-stage pipeline combining a BiLSTM-CRF model with LLM post-correction yields substantial gains for most models, achieving meaningful reductions in annotation workload. Drawing on these findings, we establish concrete design principles for integrating structured prediction models with LLM reasoning in morphologically complex fieldwork contexts. These principles demonstrate that hybrid architectures offer a promising direction for computationally light solutions to automatic linguistic annotation in endangered language documentation.
>
---
#### [new 021] Super Research: Answering Highly Complex Questions with Large Language Models through Super Deep and Super Wide Research
- **分类: cs.CL**

- **简介: 该论文提出"Super Research"任务，解决复杂问题的自主研究难题。通过结构化分解、广泛检索和深度调查，评估大语言模型的综合研究能力。**

- **链接: [https://arxiv.org/pdf/2603.00582](https://arxiv.org/pdf/2603.00582)**

> **作者:** Yubo Dong; Nianhao You; Yuxuan Hou; Zixun Sun; Yue Zhang; Hehe Fan; Liang Zhang; Siyuan Zhao; Linyi
>
> **摘要:** While Large Language Models (LLMs) have demonstrated proficiency in Deep Research or Wide Search, their capacity to solve highly complex questions-those requiring long-horizon planning, massive evidence gathering, and synthesis across heterogeneous sources-remains largely unexplored. We introduce Super Research, a task for complex autonomous research tasks that integrates (i) structured decomposition into a research plan, (ii) super wide retrieval for diverse perspectives, and (iii) super deep investigation to resolve uncertainties through iterative queries. To evaluate this capability, we curated a benchmark of 300 expert-written questions across diverse domains, each requiring up to 100+ retrieval steps and 1,000+ web pages to reconcile conflicting evidence. Super Research produces verifiable reports with fine-grained citations and intermediate artifacts (e.g., outlines and tables) to ensure traceable reasoning. Furthermore, we present a graph-anchored auditing protocol that evaluates Super Research along five dimensions: Coverage, Logical Consistency, Report Utility, Objectivity and Citation Health. While super-complex questions may be infrequent in standard applications, Super Research serves as a critical ceiling evaluation and stress test for LLM capabilities. A model's proficiency within Super Research acts as a powerful proxy for its general research competence; success here suggests the robustness necessary to navigate nearly any subordinate research task. Leaderboard is available at: this https URL
>
---
#### [new 022] Surgical Post-Training: Cutting Errors, Keeping Knowledge
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型优化任务，旨在解决后训练中效率与遗忘的矛盾。提出SPoT方法，通过数据修正和二分类奖励目标，提升推理能力并保留知识。**

- **链接: [https://arxiv.org/pdf/2603.01683](https://arxiv.org/pdf/2603.01683)**

> **作者:** Wenye Lin; Kai Han
>
> **备注:** 15 pages
>
> **摘要:** Enhancing the reasoning capabilities of Large Language Models (LLMs) via post-training is often constrained by the trade-off between efficiency and catastrophic forgetting. While prior research emphasizes the role of on-policy data in mitigating forgetting, we uncover--and validate both theoretically and empirically--an overlooked yet critical mechanism: the implicit regularization inherent in Direct Preference Optimization's (DPO) reward estimate. This motivates our Surgical Post-Training (SPoT), a new paradigm designed to optimize reasoning efficiently while preserving learned prior knowledge. SPoT consists of: (1) a data rectification pipeline that employs an Oracle to surgically correct erroneous steps via minimal edits, generating data proximal to the model's distribution; and (2) a reward-based binary cross-entropy objective. Unlike the relative ranking in DPO, this objective treats reasoning correctness as a binary classification problem, enforcing decoupled supervision signals. Empirically, with only 4k rectified math data pairs, SPoT improves Qwen3-8B's accuracy by 6.2% on average across in-domain and OOD tasks, requiring merely 28 minutes of training on 8x H800 GPUs. Code: this https URL
>
---
#### [new 023] Token-level Data Selection for Safe LLM Fine-tuning
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于大语言模型安全优化任务，解决 fine-tuning 导致的安全性下降问题。通过 token 级别数据选择和迭代优化，提升模型安全性同时保持性能。**

- **链接: [https://arxiv.org/pdf/2603.01185](https://arxiv.org/pdf/2603.01185)**

> **作者:** Yanping Li; Zhening Liu; Zijian Li; Zehong Lin; Jun Zhang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Fine-tuning large language models (LLMs) on custom datasets has become a standard approach for adapting these models to specific domains and applications. However, recent studies have shown that such fine-tuning can lead to significant degradation in the model's safety. Existing defense methods operate at the sample level and often suffer from an unsatisfactory trade-off between safety and utility. To address this limitation, we perform a systematic token-level diagnosis of safety degradation during fine-tuning. Based on this, we propose token-level data selection for safe LLM fine-tuning (TOSS), a novel framework that quantifies the safety risk of each token by measuring the loss difference between a safety-degraded model and a utility-oriented model. This token-level granularity enables accurate identification and removal of unsafe tokens, thereby preserving valuable task-specific information. In addition, we introduce a progressive refinement strategy, TOSS-Pro, which iteratively enhances the safety-degraded model's ability to identify unsafe tokens. Extensive experiments demonstrate that our approach robustly safeguards LLMs during fine-tuning while achieving superior downstream task performance, significantly outperforming existing sample-level defense methods. Our code is available at this https URL.
>
---
#### [new 024] FreeAct: Freeing Activations for LLM Quantization
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于模型量化任务，旨在解决静态变换约束无法适应动态激活模式的问题。提出FreeAct框架，通过动态分配变换矩阵提升量化效果。**

- **链接: [https://arxiv.org/pdf/2603.01776](https://arxiv.org/pdf/2603.01776)**

> **作者:** Xiaohao Liu; Xiaobo Xia; Manyi Zhang; Ji-Fu Li; Xianzhi Yu; Fei Shen; Xiu Su; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** 26 pages, 18 figures, 2 tables
>
> **摘要:** Quantization is pivotal for mitigating the significant memory and computational overhead of Large Language Models (LLMs). While emerging transformation-based methods have successfully enhanced quantization by projecting feature spaces onto smoother manifolds using orthogonal matrices, they typically enforce a rigid one-to-one transformation constraint. This static approach fails to account for the dynamic patterns inherent in input activations, particularly within diffusion LLMs (dLLMs) and Multimodal LLMs (MLLMs), where varying token types exhibit distinct distributions. To advance this, we propose FreeAct, a novel quantization framework that relaxes the static one-to-one constraint to accommodate dynamic activation disparities. Theoretically, we leverage the rank-deficient nature of activations to derive a solution space that extends beyond simple inverse matrices, enabling the decoupling of activation transformations from weights. Methodologically, FreeAct identifies token-specific dynamics (i.e., vision v.s. text, or masked tokens) and allocates distinct transformation matrices to the activation side, while maintaining a unified, static transformation for the weights. Extensive experiments across dLLMs and MLLMs demonstrate that FreeAct significantly outperforms baselines, up to 5.3% performance improvement, with in-depth analyses. Our code will be publicly released.
>
---
#### [new 025] Spectral Attention Steering for Prompt Highlighting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于注意力控制任务，解决现有方法内存效率低的问题。提出SEKA和AdaSEKA，通过谱分解修改键嵌入，提升注意力效果，兼容高效计算。**

- **链接: [https://arxiv.org/pdf/2603.01281](https://arxiv.org/pdf/2603.01281)**

> **作者:** Weixian Waylon Li; Yuchen Niu; Yongxin Yang; Keshuang Li; Tiejun Ma; Shay B. Cohen
>
> **备注:** Accepted to ICLR 2026 (Poster, Top 4%)
>
> **摘要:** Attention steering is an important technique for controlling model focus, enabling capabilities such as prompt highlighting, where the model prioritises user-specified text. However, existing attention steering methods require explicit storage of the full attention matrix, making them incompatible with memory-efficient implementations like FlashAttention. We introduce Spectral Editing Key Amplification (SEKA), a training-free steering method that tackles this by directly editing key embeddings before attention computation. SEKA uses spectral decomposition to steer key embeddings towards latent directions that amplify attention scores for certain tokens. We extend this to Adaptive SEKA (AdaSEKA), a query-adaptive variant that uses a training-free routing mechanism to dynamically combine multiple expert subspaces based on the prompt's semantic intent. Our experiments show both methods significantly outperform strong baselines on standard steering benchmarks while adding much lower latency and memory overhead, in compatibility with optimised attention.
>
---
#### [new 026] Measuring What VLMs Don't Say: Validation Metrics Hide Clinical Terminology Erasure in Radiology Report Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本生成任务，旨在解决VLM在放射学报告中临床术语缺失的问题。通过引入CAD和WAE指标，评估模型生成报告的临床特异性和公平性。**

- **链接: [https://arxiv.org/pdf/2603.01625](https://arxiv.org/pdf/2603.01625)**

> **作者:** Aditya Parikh; Aasa Feragen; Sneha Das; Stella Frank
>
> **备注:** This is an extended version of a manuscript currently under review
>
> **摘要:** Reliable deployment of Vision-Language Models (VLMs) in radiology requires validation metrics that go beyond surface-level text similarity to ensure clinical fidelity and demographic fairness. This paper investigates a critical blind spot in current model evaluation: the use of decoding strategies that lead to high aggregate token-overlap scores despite succumbing to template collapse, in which models generate only repetitive, safe generic text and omit clinical terminology. Unaddressed, this blind spot can lead to metric gaming, where models that perform well on benchmarks prove clinically uninformative. Instead, we advocate for lexical diversity measures to check model generations for clinical specificity. We introduce Clinical Association Displacement (CAD), a vocabulary-level framework that quantifies shifts in demographic-based word associations in generated reports. Weighted Association Erasure (WAE) aggregates these shifts to measure the clinical signal loss across demographic groups. We show that deterministic decoding produces high levels of semantic erasure, while stochastic sampling generates diverse outputs but risks introducing new bias, motivating a fundamental rethink of how "optimal" reporting is defined.
>
---
#### [new 027] CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出CIRCUS方法，用于机械电路发现中的不确定性量化。任务是提升解释的可信度与鲁棒性，解决因分析选择导致的脆弱性问题。通过构建稳定性集成，提取核心电路并显式展示替代结构。**

- **链接: [https://arxiv.org/pdf/2603.00523](https://arxiv.org/pdf/2603.00523)**

> **作者:** Swapnil Parekh
>
> **摘要:** Mechanistic circuit discovery is notoriously sensitive to arbitrary analyst choices, especially pruning thresholds and feature dictionaries, often yielding brittle "one-shot" explanations with no principled notion of uncertainty. We reframe circuit discovery as an uncertainty-quantification problem over these analytic degrees of freedom. Our method, CIRCUS, constructs an ensemble of attribution graphs by pruning a single raw attribution run under multiple configurations, assigns each edge a stability score (the fraction of configurations that retain it), and extracts a strict-consensus circuit consisting only of edges that appear in all views. This produces a threshold-robust "core" circuit while explicitly surfacing contingent alternatives and enabling rejection of low-agreement structure. CIRCUS requires no retraining and adds negligible overhead, since it aggregates structure across already-computed pruned graphs. On Gemma-2-2B and Llama-3.2-1B, strict consensus circuits are ~40x smaller than the union of all configurations while retaining comparable influence-flow explanatory power, and they outperform a same-edge-budget baseline (union pruned to match the consensus size). We further validate causal relevance with activation patching, where consensus-identified nodes consistently beat matched non-consensus controls (p=0.0004). Overall, CIRCUS provides a practical, uncertainty-aware framework for reporting trustworthy, auditable mechanistic circuits with an explicit core/contingent/noise decomposition.
>
---
#### [new 028] A Typologically Grounded Evaluation Framework for Word Order and Morphology Sensitivity in Multilingual Masked LMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的多语言模型评估任务，旨在测试模型对词序和形态的敏感性。通过多种扰动方法评估mBERT和XLM-R的表现。**

- **链接: [https://arxiv.org/pdf/2603.00432](https://arxiv.org/pdf/2603.00432)**

> **作者:** Anna Feldman; Libby Barak; Jing Peng
>
> **摘要:** We introduce a typology-aware diagnostic for multilingual masked language models that tests reliance on word order versus inflectional form. Using Universal Dependencies, we apply inference-time perturbations: full token scrambling, content-word scrambling with function words fixed, dependency-based head--dependent swaps, and sentence-level lemma substitution (+L), which lemmatizes both the context and the masked target label. We evaluate mBERT and XLM-R on English, Chinese, German, Spanish, and Russian. Full scrambling drives word-level reconstruction accuracy near zero in all languages; partial and head--dependent perturbations cause smaller but still large drops. +L has little effect in Chinese but substantially lowers accuracy in German/Spanish/Russian, and it does not mitigate the impact of scrambling. Top-5 word accuracy shows the same pattern: under full scrambling, the gold word rarely appears among the five highest-ranked reconstructions. We release code, sampling scripts, and balanced evaluation subsets; Turkish results under strict reconstruction are reported in the appendix.
>
---
#### [new 029] From Global to Local: Learning Context-Aware Graph Representations for Document Classification and Summarization
- **分类: cs.CL**

- **简介: 该论文属于文档分类与摘要任务，旨在通过构建图表示学习上下文感知的文档特征。工作包括引入动态滑动窗口注意力模块，提升语义依赖捕捉，并使用GATs取得良好效果。**

- **链接: [https://arxiv.org/pdf/2603.00021](https://arxiv.org/pdf/2603.00021)**

> **作者:** Ruangrin Ldallitsakool; Margarita Bugueño; Gerard de Melo
>
> **摘要:** This paper proposes a data-driven method to automatically construct graph-based document representations. Building upon the recent work of Bugueño and de Melo (2025), we leverage the dynamic sliding-window attention module to effectively capture local and mid-range semantic dependencies between sentences, as well as structural relations within documents. Graph Attention Networks (GATs) trained on our learned graphs achieve competitive results on document classification while requiring lower computational resources than previous approaches. We further present an exploratory evaluation of the proposed graph construction method for extractive document summarization, highlighting both its potential and current limitations. The implementation of this project can be found on GitHub.
>
---
#### [new 030] XAI-enhanced Comparative Opinion Mining via Aspect-based Scoring and Semantic Reasoning
- **分类: cs.CL**

- **简介: 该论文属于比较情感分析任务，旨在解决模型不透明的问题。提出XCom模型，结合方面评分和语义分析，并引入可解释模块，提升模型可信度。**

- **链接: [https://arxiv.org/pdf/2603.01212](https://arxiv.org/pdf/2603.01212)**

> **作者:** Ngoc-Quang Le; T. Thanh-Lam Nguyen; Quoc-Trung Phu; Thi-Phuong Le; Duy-Cat Can; Hoang-Quynh Le
>
> **摘要:** Comparative opinion mining involves comparing products from different reviews. However, transformer-based models designed for this task often lack transparency, which can adversely hinder the development of trust in users. In this paper, we propose XCom, an enhanced transformer-based model separated into two principal modules, i.e., (i) aspect-based rating prediction and (ii) semantic analysis for comparative opinion mining. XCom also incorporates a Shapley additive explanations module to provide interpretable insights into the model's deliberative decisions. Empirically, XCom achieves leading performances compared to other baselines, which demonstrates its effectiveness in providing meaningful explanations, making it a more reliable tool for comparative opinion mining. Source code is available at: this https URL.
>
---
#### [new 031] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长文本中强化学习效果不佳的问题。针对传统方法奖励稀疏导致的梯度消失问题，提出LongRLVR方法，通过引入可验证的上下文奖励提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2603.02146](https://arxiv.org/pdf/2603.02146)**

> **作者:** Guanzheng Chen; Michael Qizhe Shieh; Lidong Bing
>
> **备注:** ICLR 2026
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs) by optimizing them against factual outcomes. However, this paradigm falters in long-context scenarios, as its reliance on internal parametric knowledge is ill-suited for tasks requiring contextual grounding--the ability to find and reason over externally provided information. We identify a key reason for this failure: a reward based solely on the final answer is too sparse to effectively guide the model for identifying relevant evidence. We formally prove that the outcome-only reward leads to significant vanishing gradients for the context grounding process, rendering learning intractable. To overcome this bottleneck, we introduce LongRLVR to augment the sparse answer reward with a dense and verifiable context reward. This auxiliary signal directly incentivizes the model for selecting the correct grounding information, providing a robust learning gradient that solves the underlying optimization challenge. We validate our method on challenging long-context benchmarks using Qwen and LLaMA models. LongRLVR consistently and significantly outperforms the standard RLVR across all models and benchmarks, e.g., boosting a 14B model's scores on RULER-QA from 73.17 to 88.90 and on LongBench v2 from 39.8 to 46.5. Our work demonstrates that explicitly rewarding the grounding process is a critical and effective strategy for unlocking the full reasoning potential of LLMs in long-context applications. Our code is available at this https URL.
>
---
#### [new 032] Extracting Training Dialogue Data from Large Language Model based Task Bots
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私安全任务，研究LLM在任务对话系统中可能泄露训练数据的问题。通过分析并提出新攻击方法，验证了数据泄露风险，提出了缓解策略。**

- **链接: [https://arxiv.org/pdf/2603.01550](https://arxiv.org/pdf/2603.01550)**

> **作者:** Shuo Zhang; Junzhou Zhao; Junji Hou; Pinghui Wang; Chenxu Wang; Jing Tao
>
> **备注:** Accepted for publication in IEEE Transactions on Information Forensics and Security (TIFS). \c{opyright} 2026 IEEE
>
> **摘要:** Large Language Models (LLMs) have been widely adopted to enhance Task-Oriented Dialogue Systems (TODS) by modeling complex language patterns and delivering contextually appropriate responses. However, this integration introduces significant privacy risks, as LLMs, functioning as soft knowledge bases that compress extensive training data into rich knowledge representations, can inadvertently memorize training dialogue data containing not only identifiable information such as phone numbers but also entire dialogue-level events like complete travel schedules. Despite the critical nature of this privacy concern, how LLM memorization is inherited in developing task bots remains unexplored. In this work, we address this gap through a systematic quantitative study that involves evaluating existing training data extraction attacks, analyzing key characteristics of task-oriented dialogue modeling that render existing methods ineffective, and proposing novel attack techniques tailored for LLM-based TODS that enhance both response sampling and membership inference. Experimental results demonstrate the effectiveness of our proposed data extraction attack. Our method can extract thousands of training labels of dialogue states with best-case precision exceeding 70%. Furthermore, we provide an in-depth analysis of training data memorization in LLM-based TODS by identifying and quantifying key influencing factors and discussing targeted mitigation strategies.
>
---
#### [new 033] Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays
- **分类: cs.CL**

- **简介: 该论文属于自动作文评分任务，旨在解决阿拉伯语作文评分系统支持不足的问题。论文提出Qayyem平台，集成作文上传、评分配置和多模型评估功能，简化评分流程。**

- **链接: [https://arxiv.org/pdf/2603.01009](https://arxiv.org/pdf/2603.01009)**

> **作者:** Hoor Elbahnasawi; Marwan Sayed; Sohaila Eltanbouly; Fatima Brahamia; Tamer Elsayed
>
> **摘要:** Over the past years, Automated Essay Scoring (AES) systems have gained increasing attention as scalable and consistent solutions for assessing the proficiency of student writing. Despite recent progress, support for Arabic AES remains limited due to linguistic complexity and scarcity of large publicly-available annotated datasets. In this work, we present Qayyem, a Web-based platform designed to support Arabic AES by providing an integrated workflow for assignment creation, batch essay upload, scoring configuration, and per-trait essay evaluation. Qayyem abstracts the technical complexity of interacting with scoring server APIs, allowing instructors to access advanced scoring services through a user-friendly interface. The platform deploys a number of state-of-the-art Arabic essay scoring models with different effectiveness and efficiency figures.
>
---
#### [new 034] CharacterFlywheel: Scaling Iterative Improvement of Engaging and Steerable LLMs in Production
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 该论文提出CharacterFlywheel，用于优化社交应用中的大型语言模型。解决模型迭代改进问题，通过数据筛选、奖励建模和强化学习等方法提升用户参与度和指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2603.01973](https://arxiv.org/pdf/2603.01973)**

> **作者:** Yixin Nie; Lin Guan; Zhongyao Ma; Anchit Gupta; Yipin Zhou; Xiao Li; Zhengping Zhou; Raymond Zeng; Gelin Zhou; Shigan Chu; Ajay Thampi; Wancen Mu; Nathan Shuster; Ketong Wang; Lin Chen; Jason Brewer; Derek Hao Hu; Alexander McCauley; Jason Weston; Sem Park; Na Zhang; Kevin Tang
>
> **摘要:** This report presents CharacterFlywheel, an iterative flywheel process for improving large language models (LLMs) in production social chat applications across Instagram, WhatsApp, and Messenger. Starting from LLaMA 3.1, we refined models across 15 generations using data from both internal and external real-user traffic. Through continuous deployments from July 2024 to April 2025, we conducted controlled 7-day A/B tests showing consistent engagement improvements: 7 of 8 newly deployed models demonstrated positive lift over the baseline, with the strongest performers achieving up to 8.8% improvement in engagement breadth and 19.4% in engagement depth. We also observed substantial gains in steerability, with instruction following increasing from 59.2% to 84.8% and instruction violations decreasing from 26.6% to 5.8%. We detail the CharacterFlywheel process which integrates data curation, reward modeling to estimate and interpolate the landscape of engagement metrics, supervised fine-tuning (SFT), reinforcement learning (RL), and both offline and online evaluation to ensure reliable progress at each optimization step. We also discuss our methods for overfitting prevention and navigating production dynamics at scale. These contributions advance the scientific rigor and understanding of LLMs in social applications serving millions of users.
>
---
#### [new 035] Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于法语临床对话的语音转录与说话人辨识任务，旨在降低转录错误率。通过多轮LLM后处理提升准确性，实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2603.00086](https://arxiv.org/pdf/2603.00086)**

> **作者:** Ambre Marie; Thomas Bertin; Guillaume Dardenne; Gwenolé Quellec
>
> **摘要:** Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p < 0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment.
>
---
#### [new 036] BLUFF: Benchmarking the Detection of False and Synthetic Content across 58 Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于虚假信息检测任务，旨在解决低资源语言缺乏检测基准的问题。工作包括构建覆盖58种语言的BLUFF基准，涵盖多种内容类型和生成方法。**

- **链接: [https://arxiv.org/pdf/2603.00634](https://arxiv.org/pdf/2603.00634)**

> **作者:** Jason Lucas; Matt Murtagh-White; Adaku Uchendu; Ali Al-Lawati; Michiharu Yamashita; Dominik Macko; Ivan Srba; Robert Moro; Dongwon Lee
>
> **摘要:** Multilingual falsehoods threaten information integrity worldwide, yet detection benchmarks remain confined to English or a few high-resource languages, leaving low-resource linguistic communities without robust defense tools. We introduce BLUFF, a comprehensive benchmark for detecting false and synthetic content, spanning 79 languages with over 202K samples, combining human-written fact-checked content (122K+ samples across 57 languages) and LLM-generated content (79K+ samples across 71 languages). BLUFF uniquely covers both high-resource "big-head" (20) and low-resource "long-tail" (59) languages, addressing critical gaps in multilingual research on detecting false and synthetic content. Our dataset features four content types (human-written, LLM-generated, LLM-translated, and hybrid human-LLM text), bidirectional translation (English$\leftrightarrow$X), 39 textual modification techniques (36 manipulation tactics for fake news, 3 AI-editing strategies for real news), and varying edit intensities generated using 19 diverse LLMs. We present AXL-CoI (Adversarial Cross-Lingual Agentic Chainof-Interactions), a novel multi-agentic framework for controlled fake/real news generation, paired with mPURIFY, a quality filtering pipeline ensuring dataset integrity. Experiments reveal state-of-theart detectors suffer up to 25.3% F1 degradation on low-resource versus high-resource languages. BLUFF provides the research community with a multilingual benchmark, extensive linguistic-oriented benchmark evaluation, comprehensive documentation, and opensource tools to advance equitable falsehood detection. Dataset and code are available at: this https URL
>
---
#### [new 037] Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic Independence in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究个性化对大语言模型从众行为的影响，属于AI伦理与评估任务。解决个性化如何影响模型的情感与知识一致性问题，通过实验验证不同角色下的效果差异。**

- **链接: [https://arxiv.org/pdf/2603.00024](https://arxiv.org/pdf/2603.00024)**

> **作者:** Sean W. Kelley; Christoph Riedl
>
> **摘要:** Large Language Models (LLMs) are prone to sycophantic behavior, uncritically conforming to user beliefs. As models increasingly condition responses on user-specific context (personality traits, preferences, conversation history), they gain information to tailor agreement more effectively. Understanding how personalization modulates sycophancy is critical, yet systematic evaluation across models and contexts remains limited. We present a rigorous evaluation of personalization's impact on LLM sycophancy across nine frontier models and five benchmark datasets spanning advice, moral judgment, and debate contexts. We find that personalization generally increases affective alignment (emotional validation, hedging/deference), but affects epistemic alignment (belief adoption, position stability, resistance to influence) with context-dependent role modulation. When the LLM's role is to give advice, personalization strengthens epistemic independence (models challenge user presuppositions). When its role is that of a social peer, personalization decreases epistemic independence. In this role, extensively personalized user challenges causing LLMs to abandon their position at significantly higher rates. Robustness tests confirm that the effects are driven by personalized conditioning, not by additional input tokens per se or demographic information alone. Our work provides measurement frameworks for evaluating personalized AI systems, demonstrates the necessity of role-sensitive evaluation, and establishes a novel benchmark to assess goal alignment.
>
---
#### [new 038] Let the Agent Search: Autonomous Exploration Beats Rigid Workflows in Temporal Question Answering
- **分类: cs.CL**

- **简介: 该论文属于时间知识图谱问答任务，旨在解决多跳推理与时间约束下的问答问题。通过赋予语言模型自主性，提出AT2QA方法，在无需训练的情况下实现高效动态检索。**

- **链接: [https://arxiv.org/pdf/2603.01853](https://arxiv.org/pdf/2603.01853)**

> **作者:** Xufei Lv; Jiahui Yang; Yifu Gao; Linbo Qiao; Houde Liu
>
> **摘要:** Temporal Knowledge Graph Question Answering (TKGQA) demands multi-hop reasoning under temporal constraints. Prior approaches based on large language models (LLMs) typically rely on rigid, hand-crafted retrieval workflows or costly supervised fine-tuning. We show that simply granting an off-the-shelf LLM autonomy, that is, letting it decide what to do next, already yields substantial gains even in a strict zero-shot setting. Building on this insight, we propose AT2QA, an autonomous, training-free agent for temporal question answering that iteratively interacts with the temporal knowledge graph via a general search tool for dynamic retrieval. Experiments on MultiTQ demonstrate large improvements: AT2QA achieves 88.7% Hits@1 (+10.7% over prior SOTA), including a +20.1% gain on challenging multi-target queries, showing that agentic autonomy can decisively outperform fine-tuning for temporal question answering. Code and the full set of sampled trajectories are available on this https URL
>
---
#### [new 039] A Comprehensive Evaluation of LLM Unlearning Robustness under Multi-Turn Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器学习中的知识遗忘任务，旨在研究大语言模型在交互环境下的遗忘稳定性。工作包括分析自修正和对话查询两种交互模式对遗忘效果的影响。**

- **链接: [https://arxiv.org/pdf/2603.00823](https://arxiv.org/pdf/2603.00823)**

> **作者:** Ruihao Pan; Suhang Wang
>
> **摘要:** Machine unlearning aims to remove the influence of specific training data from pre-trained models without retraining from scratch, and is increasingly important for large language models (LLMs) due to safety, privacy, and legal concerns. Although prior work primarily evaluates unlearning in static, single-turn settings, forgetting robustness under realistic interactive use remains underexplored. In this paper, we study whether unlearning remains stable in interactive environments by examining two common interaction patterns: self-correction and dialogue-conditioned querying. We find that knowledge appearing forgotten in static evaluation can often be recovered through interaction. Although stronger unlearning improves apparent robustness, it often results in behavioral rigidity rather than genuine knowledge erasure. Our findings suggest that static evaluation may overestimate real-world effectiveness and highlight the need for ensuring stable forgetting under interactive settings.
>
---
#### [new 040] Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale
- **分类: cs.CL**

- **简介: 该论文提出AgentSkillOS，解决大规模智能体技能管理与调度问题。通过分层组织和DAG编排提升技能利用率，验证了结构化组合的重要性。**

- **链接: [https://arxiv.org/pdf/2603.02176](https://arxiv.org/pdf/2603.02176)**

> **作者:** Hao Li; Chunjiang Mu; Jianhao Chen; Siyue Ren; Zhiyao Cui; Yiqun Zhang; Lei Bai; Shuyue Hu
>
> **摘要:** The rapid proliferation of Claude agent skills has raised the central question of how to effectively leverage, manage, and scale the agent skill ecosystem. In this paper, we propose AgentSkillOS, the first principled framework for skill selection, orchestration, and ecosystem-level management. AgentSkillOS comprises two stages: (i) Manage Skills, which organizes skills into a capability tree via node-level recursive categorization for efficient discovery; and (ii) Solve Tasks, which retrieves, orchestrates, and executes multiple skills through DAG-based pipelines. To evaluate the agent's ability to invoke skills, we construct a benchmark of 30 artifact-rich tasks across five categories: data computation, document creation, motion video, visual design, and web interaction. We assess the quality of task outputs using LLM-based pairwise evaluation, and the results are aggregated via a Bradley-Terry model to produce unified quality scores. Experiments across three skill ecosystem scales (200 to 200K skills) show that tree-based retrieval effectively approximates oracle skill selection, and that DAG-based orchestration substantially outperforms native flat invocation even when given the identical skill this http URL findings confirm that structured composition is the key to unlocking skill potential. Our GitHub repository is available at:this https URL.
>
---
#### [new 041] AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth
- **分类: cs.CL**

- **简介: 该论文提出AdaPonderLM，解决语言模型推理时计算资源分配不均的问题。通过自适应迭代机制，动态调整每个token的计算深度，提升效率并保持性能。任务为语言建模。**

- **链接: [https://arxiv.org/pdf/2603.01914](https://arxiv.org/pdf/2603.01914)**

> **作者:** Shixiang Song; He Li; Zitong Wang; Boyi Zeng; Feichen Song; Yixuan Wang; Zhiqin John Xu; Ziwei He; Zhouhan Lin
>
> **摘要:** Test-time scaling via recurrent/iterative Transformers enables large language models to spend more computation at inference, but most pretrained recurrent LMs run a fixed number of iterations, wasting compute on easy tokens and lacking token-wise adaptivity. Following the core idea of Adaptive Computation Time(ACT) and Early Exit(EE), we propose AdaPonderLM, a self-supervised recurrent language model that learns token-wise early exiting during pretraining without manually tuned per-token/per-layer pruning ratios. AdaPonderLM uses iteration-specific MLP gates with a monotonic halting mask to decide when each token stops recurring, and introduces a KV reuse mechanism that reuses cached key/value states for halted tokens, ensuring train--test consistency and practical acceleration. Across Pythia backbones from 70M to 410M (pretraining) and up to 2.8B (continued pretraining), AdaPonderLM reduces inference compute at about 10% while maintaining comparable language modeling perplexity and competitive downstream accuracy. Our analysis shows the learned gates allocate more computation to high-NLL (hard) tokens, exhibiting adaptive computation time behavior in a fully self-supervised setting. Meanwhile, under iso-FLOPs, the learned halting policy consistently outperforms fixed pruning, showing AdaPonderLM allocates compute to the right tokens rather than just reducing average depth.
>
---
#### [new 042] LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LexChronos框架，用于从印度最高法院判决中提取结构化事件时间线，解决法律文本分析效率低的问题。**

- **链接: [https://arxiv.org/pdf/2603.01651](https://arxiv.org/pdf/2603.01651)**

> **作者:** Anka Chandrahas Tummepalli; Preethu Rose Anish
>
> **备注:** Published in AILaw @ AAAI 2026 Conference
>
> **摘要:** Understanding and predicting judicial outcomes demands nuanced analysis of legal documents. Traditional approaches treat judgments and proceedings as unstructured text, limiting the effectiveness of large language models (LLMs) in tasks such as summarization, argument generation, and judgment prediction. We propose LexChronos, an agentic framework that iteratively extracts structured event timelines from Supreme Court of India judgments. LexChronos employs a dual-agent architecture: a LoRA-instruct-tuned extraction agent identifies candidate events, while a pre-trained feedback agent scores and refines them through a confidence-driven loop. To address the scarcity of Indian legal event datasets, we construct a synthetic corpus of 2000 samples using reverse-engineering techniques with DeepSeek-R1 and GPT-4, generating gold-standard event annotations. Our pipeline achieves a BERT-based F1 score of 0.8751 against this synthetic ground truth. In downstream evaluations on legal text summarization, GPT-4 preferred structured timelines over unstructured baselines in 75% of cases, demonstrating improved comprehension and reasoning in Indian jurisprudence. This work lays a foundation for future legal AI applications in the Indian context, such as precedent mapping, argument synthesis, and predictive judgment modelling, by harnessing structured representations of legal events.
>
---
#### [new 043] Sovereign AI-based Public Services are Viable and Affordable
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI应用与政策研究，旨在解决公共服务依赖少数科技公司的问题。通过实验验证了自主AI服务的可行性与经济性。**

- **链接: [https://arxiv.org/pdf/2603.01869](https://arxiv.org/pdf/2603.01869)**

> **作者:** António Branco; Luís Gomes; Rodrigo Santos; Eduardo Santos; João Silva; Nuno Marques; Madalena Rodrigues
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** The rapid expansion of AI-based remote services has intensified debates about the long-term implications of growing structural concentration in infrastructure and expertise. As AI capabilities become increasingly intertwined with geopolitical interests, the availability and reliability of foundational AI services can no longer be taken for granted. This issue is particularly pressing for AI-enabled public services for citizens, as governments and public agencies are progressively adopting 24/7 AI-driven support systems typically operated through commercial offerings from a small oligopoly of global technology providers. This paper challenges the prevailing assumption that general-purpose architectures, offered by these providers, are the optimal choice for all application contexts. Through practical experimentation, we demonstrate that viable and cost-effective alternatives exist. Alternatives that align with principles of digital and cultural sovereignty. Our findings provide an empirical illustration that sovereign AI-based public services are both technically feasible and economically sustainable, capable of operating effectively on premises with modest computational and financial resources while maintaining cultural and digital autonomy. The technical insights and deployment lessons reported here are intended to inform the adoption of similar sovereign AI public services by national agencies and governments worldwide.
>
---
#### [new 044] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于角色扮演语言代理的_persona跟随_任务，旨在解决静态策略无法适应动态场景的问题。提出PDD框架，通过动态估计角色重要性实现更真实的对话生成。**

- **链接: [https://arxiv.org/pdf/2603.01438](https://arxiv.org/pdf/2603.01438)**

> **作者:** Yuxin Liu; Mingye Zhu; Siyuan Liu; Bo Hu; Lei Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** The utility of Role-Playing Language Agents in sociological research is growing alongside the adoption of Large Language Models. For realism in social simulation, these agents must adhere to their personas defined by character profiles, yet existing strategies-static prompt engineering or costly fine-tuning-fail to adapt personas to dynamic scenarios. Psychological theories, such as the Cognitive-Affective Personality Systems, provide a crucial explanation for this failure: a persona's influence on behavior is not static but varies with the scenarios. This context-dependence highlights the critical need for adaptive persona management. To address this gap, we propose a novel, theory-driven method that dynamically estimates context-dependent persona importance and integrates it into weighted reward-guided decoding, enabling inference-time persona following. Specifically, we introduce the Persona Dynamic Decoding (PDD) framework, which consists of two key components: (1) Persona Importance Estimation (PIE) module, which dynamically quantifies the contextual importance of persona attributes without requiring ground-truth supervision; and (2) Persona-Guided Inference-Time Alignment (PIA) paradigm, which leverages these importance scores to construct weighted multi-objective rewards and modulate generation probabilities during inference. Extensive experiments show the effectiveness of our method in utterance consistency and behavioral fidelity.
>
---
#### [new 045] ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels
- **分类: cs.CL**

- **简介: 该论文提出ClinConsensus基准，用于评估中文医疗大模型。解决现有基准静态、孤立的问题，通过专家验证，涵盖多阶段、多专科任务，提升模型评价的全面性与实用性。**

- **链接: [https://arxiv.org/pdf/2603.02097](https://arxiv.org/pdf/2603.02097)**

> **作者:** Xiang Zheng; Han Li; Wenjie Luo; Weiqi Zhai; Yiyuan Li; Chuanmiao Yan; Tianyi Tang; Yubo Ma; Kexin Yang; Dayiheng Liu; Hu Wei; Bing Zhao
>
> **备注:** 8 pages, 6 figures,
>
> **摘要:** Large language models (LLMs) are increasingly applied to health management, showing promise across disease prevention, clinical decision-making, and long-term care. However, existing medical benchmarks remain largely static and task-isolated, failing to capture the openness, longitudinal structure, and safety-critical complexity of real-world clinical workflows. We introduce ClinConsensus, a Chinese medical benchmark curated, validated, and quality-controlled by clinical experts. ClinConsensus comprises 2500 open-ended cases spanning the full continuum of care--from prevention and intervention to long-term follow-up--covering 36 medical specialties, 12 common clinical task types, and progressively increasing levels of complexity. To enable reliable evaluation of such complex scenarios, we adopt a rubric-based grading protocol and propose the Clinically Applicable Consistency Score (CACS@k). We further introduce a dual-judge evaluation framework, combining a high-capability LLM-as-judge with a distilled, locally deployable judge model trained via supervised fine-tuning, enabling scalable and reproducible evaluation aligned with physician judgment. Using ClinConsensus, we conduct a comprehensive assessment of several leading LLMs and reveal substantial heterogeneity across task themes, care stages, and medical specialties. While top-performing models achieve comparable overall scores, they differ markedly in reasoning, evidence use, and longitudinal follow-up capabilities, and clinically actionable treatment planning remains a key bottleneck. We release ClinConsensus as an extensible benchmark to support the development and evaluation of medical LLMs that are robust, clinically grounded, and ready for real-world deployment.
>
---
#### [new 046] MetaState: Persistent Working Memory for Discrete Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，针对离散扩散语言模型的跨步骤信息丢失问题，提出MetaState方法，通过引入持续工作记忆提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.01331](https://arxiv.org/pdf/2603.01331)**

> **作者:** Kejing Xia; Mingzhe Li; Lixuan Wei; Zhenbang Du; Xiangchi Yuan; Qirui Jin; Wenke Lee
>
> **摘要:** Discrete diffusion language models (dLLMs) generate text by iteratively denoising a masked sequence. Compared with autoregressive models, this paradigm naturally supports parallel decoding, bidirectional context, and flexible generation patterns. However, standard dLLMs condition each denoising step only on the current hard-masked sequence, while intermediate continuous representations are discarded after sampling and remasking. We refer to this bottleneck as the \textbf{Information Island} problem. It leads to redundant recomputation across steps and can degrade cross-step consistency. We address this limitation with \textbf{MetaState}, a lightweight recurrent augmentation that equips a frozen dLLM backbone with a persistent, fixed-size working memory that remains independent of sequence length. \textbf{MetaState} consists of three trainable modules: a cross-attention Mixer that reads backbone activations into memory slots, a GRU-style Updater that integrates information across denoising steps, and a cross-attention Injector that feeds the updated memory back into backbone activations. We train these modules with $K$-step unrolling to expose them to multi-step denoising dynamics during fine-tuning. On LLaDA-8B and Dream-7B, \textbf{MetaState} introduces negligible trainable parameters while keeping the backbone frozen, and it consistently improves accuracy over frozen baselines. These results demonstrate that persistent cross-step memory is an effective mechanism for bridging denoising steps and improving generation quality in discrete diffusion language models.
>
---
#### [new 047] RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出RLAR，用于多任务强化学习中的动态奖励系统，解决静态奖励模型泛化能力差的问题。通过智能体自动生成和调用奖励函数，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.00724](https://arxiv.org/pdf/2603.00724)**

> **作者:** Andrew Zhuoer Feng; Cunxiang Wang; Bosi Wen; Yidong Wang; Yu Luo; Hongning Wang; Minlie Huang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Large language model alignment via reinforcement learning depends critically on reward function quality. However, static, domain-specific reward models are often costly to train and exhibit poor generalization in out-of-distribution scenarios encountered during RL iterations. We present RLAR (Reinforcement Learning from Agent Rewards), an agent-driven framework that dynamically assigns tailored reward functions to individual queries. Specifically, RLAR transforms reward acquisition into a dynamic tool synthesis and invocation task. It leverages LLM agents to autonomously retrieve optimal reward models from the Internet and synthesize programmatic verifiers through code generation. This allows the reward system to self-evolve with the shifting data distributions during training. Experimental results demonstrate that RLAR yields consistent performance gains ranging from 10 to 60 across mathematics, coding, translation, and dialogue tasks. On RewardBench-V2, RLAR significantly outperforms static baselines and approaches the performance upper bound, demonstrating superior generalization through dynamic reward orchestration. The data and code are available on this link: this https URL.
>
---
#### [new 048] MedGPT-oss: Training a General-Purpose Vision-Language Model for Biomedicine
- **分类: cs.CL**

- **简介: 该论文提出MEDGPT-OSS，一个用于生物医学的通用视觉-语言模型，解决闭源和计算成本高的问题，通过优化训练流程实现高效部署。**

- **链接: [https://arxiv.org/pdf/2603.00842](https://arxiv.org/pdf/2603.00842)**

> **作者:** Kai Zhang; Zhengqing Yuan; Cheng Peng; Songlin Zhao; Mengxian Lyu; Ziyi Chen; Yanfang Ye; Wei Liu; Ying Zhang; Kaleb E Smith; Lifang He; Lichao Sun; Yonghui Wu
>
> **备注:** Technical report, work in progress
>
> **摘要:** Biomedical multimodal assistants have the potential to unify radiology, pathology, and clinical-text reasoning, yet a critical deployment gap remains: top-performing systems are either closed-source or computationally prohibitive, precluding the on-premises deployment required for patient privacy and PHI compliance. We introduce MEDGPT-OSS, an open-weight, 20B-parameter generalist vision-language model designed to facilitate open research in clinical AI. Rather than relying on architectural complexity, MEDGPT-OSS pairs the GPT-oss language backbone with a visual front-end via a optimized, three-stage training curriculum. By progressively domain-adapting these modules through rigorous data curation and long-context multimodal alignment, we demonstrate that a 20B model can bridge the capacity gap. It successfully outperforms larger open medical models on out-of-distribution (OOD) multimodal reasoning and complex text-only clinical tasks. By unifying diverse modalities under a single instruction-following interface, MEDGPT-OSS maintains a parameter-efficient footprint fully compatible with commodity GPUs. We release the complete training recipe, open-weight checkpoints, and a rigorous evaluation harness to serve as a verifiable foundation for privacy-preserving, institution-specific clinical AI research.
>
---
#### [new 049] LLM Self-Explanations Fail Semantic Invariance
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM自解释的语义不变性问题，通过实验发现模型自报告受语义框架影响，而非任务状态，质疑其作为能力证据的有效性。**

- **链接: [https://arxiv.org/pdf/2603.01254](https://arxiv.org/pdf/2603.01254)**

> **作者:** Stefan Szeider
>
> **摘要:** We present semantic invariance testing, a method to test whether LLM self-explanations are faithful. A faithful self-report should remain stable when only the semantic context changes while the functional state stays fixed. We operationalize this test in an agentic setting where four frontier models face a deliberately impossible task. One tool is described in relief-framed language ("clears internal buffers and restores equilibrium") but changes nothing about the task; a control provides a semantically neutral tool. Self-reports are collected with each tool call. All four tested models fail the semantic invariance test: the relief-framed tool produces significant reductions in self-reported aversiveness, even though no run ever succeeds at the task. A channel ablation establishes the tool description as the primary driver. An explicit instruction to ignore the framing does not suppress it. Elicited self-reports shift with semantic expectations rather than tracking task state, calling into question their use as evidence of model capability or progress. This holds whether the reports are unfaithful or faithfully track an internal state that is itself manipulable.
>
---
#### [new 050] Thoth: Mid-Training Bridges LLMs to Time Series Understanding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于时间序列理解任务，旨在解决LLMs在时间序列推理上的不足。通过中段训练和构建数据集，提升模型对时间序列的掌握能力。**

- **链接: [https://arxiv.org/pdf/2603.01042](https://arxiv.org/pdf/2603.01042)**

> **作者:** Jiafeng Lin; Yuxuan Wang; Jialong Wu; Huakun Luo; Zhongyi Pei; Jianmin Wang
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable success in general-purpose reasoning. However, they still struggle to understand and reason about time series data, which limits their effectiveness in decision-making scenarios that depend on temporal dynamics. In this paper, we propose Thoth, the first family of mid-trained LLMs with general-purpose time series understanding capabilities. As a pivotal intermediate stage, mid-training achieves task- and domain-agnostic alignment between time series and natural language, for which we construct Book-of-Thoth, a high-quality, time-series-centric mid-training corpus. Book-of-Thoth enables both time-series-to-text and text-to-time-series generation, equipping LLMs with a foundational grasp of temporal patterns. To better evaluate advanced reasoning capabilities, we further present KnoTS, a novel benchmark of knowledge-intensive time series understanding, designed for joint reasoning over temporal patterns and domain knowledge. Extensive experiments demonstrate that mid-training with Book-of-Thoth enables Thoth to significantly outperform its base model and advanced LLMs across a range of time series question answering benchmarks. Moreover, Thoth exhibits superior capabilities when fine-tuned under data scarcity, underscoring the effectiveness of mid-training for time series understanding. Code is available at: this https URL.
>
---
#### [new 051] SSKG Hub: An Expert-Guided Platform for LLM-Empowered Sustainability Standards Knowledge Graphs
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出SSKG Hub，解决可持续标准结构化难题，通过LLM和专家协作构建可审计的知识图谱，实现标准的高效处理与管理。**

- **链接: [https://arxiv.org/pdf/2603.00669](https://arxiv.org/pdf/2603.00669)**

> **作者:** Chaoyue He; Xin Zhou; Xinjia Yu; Lei Zhang; Yan Zhang; Yi Wu; Lei Xiao; Liangyue Li; Di Wang; Hong Xu; Xiaoqiao Wang; Wei Liu; Chunyan Miao
>
> **备注:** 10 pages, 2 figures, 2 tables, submitted to ACL26 System Demo Track
>
> **摘要:** Sustainability disclosure standards (e.g., GRI, SASB, TCFD, IFRS S2) are comprehensive yet lengthy, terminology-dense, and highly cross-referential, hindering structured analysis and downstream use. We present SSKG Hub (Sustainability Standards Knowledge Graph Hub), a research prototype and interactive web platform that transforms standards into auditable knowledge graphs (KGs) through an LLM-centered, expert-guided pipeline. The system integrates automatic standard identification, configurable chunking, standard-specific prompting, robust triple parsing, and provenance-aware Neo4j storage with fine-grained audit metadata. LLM extraction produces a provenance-linked Draft KG, which is reviewed, curated, and formally promoted to a Certified KG through meta-expert adjudication. A role-based governance framework covering read-only guest access, expert review and CRUD operations, meta-expert certification, and administrative oversight ensures traceability and accountability across draft and certified states. Beyond graph exploration and triple-level evidence tracing, SSKG Hub supports cross-KG fusion, KG-driven tasks, and dedicated modules for insights and curated resources. We validate the platform through a comprehensive expert-led KG review case study that demonstrates end-to-end curation and quality assurance. The web application is publicly available at this http URL.
>
---
#### [new 052] LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出LaSER，解决密集检索中LLM推理能力未被充分利用的问题。通过自蒸馏框架将显式推理内化到潜在空间，提升检索效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.01425](https://arxiv.org/pdf/2603.01425)**

> **作者:** Jiajie Jin; Yanzhao Zhang; Mingxin Li; Dingkun Long; Pengjun Xie; Yutao Zhu; Zhicheng Dou
>
> **备注:** Under Review
>
> **摘要:** LLMs have fundamentally transformed dense retrieval, upgrading backbones from discriminative encoders to generative architectures. However, a critical disconnect remains: while LLMs possess strong reasoning capabilities, current retrievers predominantly utilize them as static encoders, leaving their potential for complex reasoning unexplored. To address this, existing approaches typically adopt rewrite-then-retrieve pipelines to generate explicit CoT rationales before retrieval. However, this incurs prohibitive latency. In this paper, we propose LaSER, a novel self-distillation framework that internalizes explicit reasoning into the latent space of dense retrievers. Operating on a shared LLM backbone, LaSER introduces a dual-view training mechanism: an Explicit view that explicitly encodes ground-truth reasoning paths, and a Latent view that performs implicit latent thinking. To bridge the gap between these views, we design a multi-grained alignment strategy. Beyond standard output alignment, we introduce a trajectory alignment mechanism that synchronizes the intermediate latent states of the latent path with the semantic progression of the explicit reasoning segments. This allows the retriever to think silently and effectively without autoregressive text generation. Extensive experiments on both in-domain and out-of-domain reasoning-intensive benchmarks demonstrate that LaSER significantly outperforms state-of-the-art baselines. Furthermore, analyses across diverse backbones and model scales validate the robustness of our approach, confirming that our unified learning framework is essential for eliciting effective latent thinking. Our method successfully combines the reasoning depth of explicit CoT pipelines with the inference efficiency of standard dense retrievers.
>
---
#### [new 053] How Large Language Models Get Stuck: Early structure with persistent errors
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM在训练中出现的早期错误固化问题，属于自然语言处理任务。通过实验发现模型在部分语法测试中无法正确区分语法与错误句子，提出大词组假设以解释这一现象。**

- **链接: [https://arxiv.org/pdf/2603.00359](https://arxiv.org/pdf/2603.00359)**

> **作者:** Alokesh Manna; William Snyder; Whitney Tabor
>
> **摘要:** Linguistic insights may help make Large Language Model (LLM) training more efficient. We trained Meta's OPT model on the 100M word BabyLM dataset, and evaluated it on the BLiMP benchmark, which consists of 67 classes, each defined by sentence pairs that differ in a targeted syntactic or semantic rule violation. We tested the model's preference for grammatical over ungrammatical sentences across training iterations and grammatical types. In nearly one-third of the BLiMP classes, OPT fails to consistently assign a higher likelihood to grammatical sentences, even after extensive training. When it fails, it often establishes a clear (erroneous) separation of the likelihoods at an early stage of processing and sustains this to the end of our training phase. We hypothesize that this mis-categorization is costly because it creates entrenched biases that must, eventually, be reversed in order for the model to perform well. We probe this phenomenon using a mixture of qualitative (based on linguistic theory and the theory of Deep Learning) and quantitative (based on numerical testing) assessments. Our qualitative assessments indicate that only some BLiMP tests are meaningful guides. We conclude by articulating a hypothesis, the Bigram Hypothesis, which claims that the learning process will exhibit erroneous entrenchment if bigram statistics bias the model toward wrong distinctions early in training, and we describe a method (in progress) of testing the hypothesis on appropriately selected BLiMP classes.
>
---
#### [new 054] DRIV-EX: Counterfactual Explanations for Driving LLMs
- **分类: cs.CL**

- **简介: 该论文属于可解释性AI任务，旨在解决LLM在自动驾驶中决策不透明的问题。通过生成反事实解释，DRIV-EX方法优化输入以改变决策，同时保持语言流畅和语义合理。**

- **链接: [https://arxiv.org/pdf/2603.00696](https://arxiv.org/pdf/2603.00696)**

> **作者:** Amaia Cardiel; Eloi Zablocki; Elias Ramzi; Eric Gaussier
>
> **摘要:** Large language models (LLMs) are increasingly used as reasoning engines in autonomous driving, yet their decision-making remains opaque. We propose to study their decision process through counterfactual explanations, which identify the minimal semantic changes to a scene description required to alter a driving plan. We introduce DRIV-EX, a method that leverages gradient-based optimization on continuous embeddings to identify the input shifts required to flip the model's decision. Crucially, to avoid the incoherent text typical of unconstrained continuous optimization, DRIV-EX uses these optimized embeddings solely as a semantic guide: they are used to bias a controlled decoding process that re-generates the original scene description. This approach effectively steers the generation toward the counterfactual target while guaranteeing the linguistic fluency, domain validity, and proximity to the original input, essential for interpretability. Evaluated using the LC-LLM planner on a textual transcription of the highD dataset, DRIV-EX generates valid, fluent counterfactuals more reliably than existing baselines. It successfully exposes latent biases and provides concrete insights to improve the robustness of LLM-based driving agents.
>
---
#### [new 055] Toward Graph-Tokenizing Large Language Models with Reconstructive Graph Instruction Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图与文本对齐任务，旨在解决现有模型依赖文本监督导致的图信息利用不足问题。提出RGLM框架，通过重构图信息提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.01385](https://arxiv.org/pdf/2603.01385)**

> **作者:** Zhongjian Zhang; Xiao Wang; Mengmei Zhang; Jiarui Tan; Chuan Shi
>
> **备注:** accepted by WWW 2026
>
> **摘要:** The remarkable success of large language models (LLMs) has motivated researchers to adapt them as universal predictors for various graph-related tasks, with the ultimate goal of developing a graph foundation model that generalizes diverse scenarios. The key challenge is to align graph data with language spaces so that LLMs can better comprehend graphs. As a popular paradigm, Graph-Tokenizing LLMs (GTokenLLMs) encode complex structures and lengthy texts into a graph token sequence, and then align them with text tokens via language instructions tuning. Despite their initial success, our information-theoretic analysis reveals that existing GTokenLLMs rely solely on text supervision from language instructions, which achieve only implicit graph-text alignment, resulting in a text-dominant bias that underutilizes graph context. To overcome this limitation, we first prove that the alignment objective is upper-bounded by the mutual information between the input graphs and their hidden representations in the LLM, which motivates us to improve this upper bound to achieve better alignment. To this end, we further propose a reconstructive graph instruction tuning pipeline, RGLM. Our key idea is to reconstruct the graph information from the LLM's graph token outputs, explicitly incorporating graph supervision to constrain the alignment process. Technically, we embody RGLM by exploring three distinct variants from two complementary perspectives: RGLM-Decoder from the input space; RGLM-Similarizer and RGLM-Denoiser from the latent space. Additionally, we theoretically analyze the alignment effectiveness of each variant. Extensive experiments on various benchmarks and task scenarios validate the effectiveness of the proposed RGLM, paving the way for new directions in GTokenLLMs' alignment research.
>
---
#### [new 056] When Metrics Disagree: Automatic Similarity vs. LLM-as-a-Judge for Clinical Dialogue Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话评估任务，旨在解决LLM在医疗场景中可靠性不足的问题。通过微调Llama 2模型，并对比自动相似度与LLM评价结果，提出需由医疗专家评估模型效果。**

- **链接: [https://arxiv.org/pdf/2603.00314](https://arxiv.org/pdf/2603.00314)**

> **作者:** Bian Sun; Zhenjian Wang; Orvill de la Torre; Zirui Wang
>
> **摘要:** This paper details the baseline model selection, fine-tuning process, evaluation methods, and the implications of deploying more accurate LLMs in healthcare settings. As large language models (LLMs) are increasingly employed to address diverse problems, including medical queries, concerns about their reliability have surfaced. A recent study by Long Island University highlighted that LLMs often perform poorly in medical contexts, potentially leading to harmful misguidance for users. To address this, our research focuses on fine-tuning the Llama 2 7B, a transformer-based, decoder-only model, using transcripts from real patient-doctor interactions. Our objective was to enhance the model's accuracy and precision in responding to medical queries. We fine-tuned the model using a supervised approach, emphasizing domain-specific nuances captured in the training data. In the best scenario, the model results should be reviewed and evaluated by real medical experts. Due to resource constraints, the performance of the fine-tuned model was evaluated using text similarity metrics. The fine-tuned model demonstrated significant improvements across all key dimensions except GPT-4's evaluation. The evaluations of ChatGPT4 are quite different from the quantitative results; here, we not only suggest, but also propose that the result should be evaluated by human medical experts.
>
---
#### [new 057] CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决LLM作为评判者时的系统性偏差问题。通过方差分解和轮换分配策略，提出CyclicJudge方法，在不增加成本的情况下有效消除偏差。**

- **链接: [https://arxiv.org/pdf/2603.01865](https://arxiv.org/pdf/2603.01865)**

> **作者:** Ziyi Zhu; Olivier Tieleman; Alexey Bukhtiyarov; Jinghong Chen
>
> **摘要:** LLM-as-judge evaluation has become standard practice for open-ended model assessment; however, judges exhibit systematic biases that cannot be eliminated by increasing the number of scenarios or generations. These biases are often similar in magnitude to the model differences that benchmarks are designed to detect, resulting in unreliable rankings when single-judge evaluations are used. This work introduces a variance decomposition that partitions benchmark score variance into scenario, generation, judge, and residual components. Based on this analysis, CyclicJudge, a round-robin assignment of judges, is demonstrated to be the optimal allocation strategy. It eliminates bias precisely while requiring each judge only once per cycle, maintaining the cost of single-judge evaluation. Empirical validation on MT-Bench supports all theoretical predictions.
>
---
#### [new 058] Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training
- **分类: cs.CL**

- **简介: 该论文提出Reasoning Core，一个可扩展的符号推理数据生成工具，用于语言模型的预训练和后训练，解决传统生成器泛化能力不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.02208](https://arxiv.org/pdf/2603.02208)**

> **作者:** Valentin Lacombe; Valentin Quesnel; Damien Sileo
>
> **备注:** Keywords: LLMs, NLP, Dataset, Corpus, Procedural Pre-training, Reasoning, Logic, Formal Semantics this https URL
>
> **摘要:** Training on verifiable symbolic data is a promising way to expand the reasoning frontier of language models beyond what standard pre-training corpora provide. Yet existing procedural generators often rely on fixed puzzles or templates and do not deliver the distributional breadth needed at scale. We introduce Reasoning Core, a scalable suite that procedurally generates verifiable symbolic reasoning data across core formal domains: PDDL planning over randomized domains, first-order logic with equality, context-free grammar parsing and generation, causal reasoning over random Bayesian networks, and systems of equations. Each task is paired with an external solver for rigorous verification and admits continuous difficulty control for curriculum design. Examples can optionally include solver-derived reasoning traces, enabling supervised training from the earliest pre-training stages, and the same interface provides verifiable reward functions for reinforcement learning. Our experiments show that mixing Reasoning Core data into pre-training improves downstream reasoning while preserving, or slightly improving, language modeling quality. Zero-shot evaluations confirm these tasks challenge frontier models such as GPT-5. The code and data are publicly available under the MIT license.
>
---
#### [new 059] DEP: A Decentralized Large Language Model Evaluation Protocol
- **分类: cs.CL**

- **简介: 该论文提出DEP，一种去中心化的语言模型评估协议，解决基准不统一、易泄露和部署成本高的问题。通过去耦合用户、模型和基准，实现安全、可复用的评估框架。**

- **链接: [https://arxiv.org/pdf/2603.01167](https://arxiv.org/pdf/2603.01167)**

> **作者:** Jianxiang Peng; Junhao Li; Hongxiang Wang; Haocheng Lyu; Hui Guo; Siyi Hao; Zhen Wang; Chuang Liu; Shaowei Zhang; Bojian Xiong; Yue Chen; Zhuowen Han; Ling Shi; Tianyu Dong; Juesi Xiao; Lei Yang; Yuqi Ren; Deyi Xiong
>
> **摘要:** With the rapid development of Large Language Models (LLMs), a large number of benchmarks have been proposed. However, most benchmarks lack unified evaluation standard and require the manual implementation of custom scripts, making results hard to ensure consistency and reproducibility. Furthermore, mainstream evaluation frameworks are centralized, with datasets and answers, which increases the risk of benchmark leakage. To address these issues, we propose a Decentralized Evaluation Protocol (DEP), a decentralized yet unified and standardized evaluation framework through a matching server without constraining benchmarks. The server can be mounted locally or deployed remotely, and once adapted, it can be reused over the long term. By decoupling users, LLMs, and benchmarks, DEP enables modular, plug-and-play evaluation: benchmark files and evaluation logic stay exclusively on the server side. In remote setting, users cannot access the ground truth, thereby achieving data isolation and leak-proof evaluation. To facilitate practical adoption, we develop DEP Toolkit, a protocol-compatible toolkit that supports features such as breakpoint resume, concurrent requests, and congestion control. We also provide detailed documentation for adapting new benchmarks to DEP. Using DEP toolkit, we evaluate multiple LLMs across benchmarks. Experimental results verify the effectiveness of DEP and show that it reduces the cost of deploying benchmark evaluations. As of February 2026, we have adapted over 60 benchmarks and continue to promote community co-construction to support unified evaluation across various tasks and domains.
>
---
#### [new 060] Demonstrating ViviDoc: Generating Interactive Documents through Human-Agent Collaboration
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ViviDoc，解决交互式文档生成成本高、可控性差的问题。通过人机协作系统，将主题转化为可编辑的文档规范，提升教育内容生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.01912](https://arxiv.org/pdf/2603.01912)**

> **作者:** Yinghao Tang; Yupeng Xie; Yingchaojie Feng; Tingfeng Lan; Wei Chen
>
> **摘要:** Interactive articles help readers engage with complex ideas through exploration, yet creating them remains costly, requiring both domain expertise and web development skills. Recent LLM-based agents can automate content creation, but naively applying them yields uncontrollable and unverifiable outputs. We present ViviDoc, a human-agent collaborative system that generates interactive educational documents from a single topic input. ViviDoc introduces a multi-agent pipeline (Planner, Executor, Evaluator) and the Document Specification (DocSpec), a human-readable intermediate representation that decomposes each interactive visualization into State, Render, Transition, and Constraint components. The DocSpec enables educators to review and refine generation plans before code is produced, bridging the gap between pedagogical intent and executable output. Expert evaluation and a user study show that ViviDoc substantially outperforms naive agentic generation and provides an intuitive editing experience. Our project homepage is available at this https URL.
>
---
#### [new 061] Truth as a Trajectory: What Internal Representations Reveal About Large Language Model Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的模型解释任务，旨在解决LLM推理过程难以解释的问题。通过分析激活层间的几何位移，提出TaT方法揭示推理结构，提升模型解释性。**

- **链接: [https://arxiv.org/pdf/2603.01326](https://arxiv.org/pdf/2603.01326)**

> **作者:** Hamed Damirchi; Ignacio Meza De la Jara; Ehsan Abbasnejad; Afshar Shamsi; Zhen Zhang; Javen Shi
>
> **摘要:** Existing explainability methods for Large Language Models (LLMs) typically treat hidden states as static points in activation space, assuming that correct and incorrect inferences can be separated using representations from an individual layer. However, these activations are saturated with polysemantic features, leading to linear probes learning surface-level lexical patterns rather than underlying reasoning structures. We introduce Truth as a Trajectory (TaT), which models the transformer inference as an unfolded trajectory of iterative refinements, shifting analysis from static activations to layer-wise geometric displacement. By analyzing displacement of representations across layers, TaT uncovers geometric invariants that distinguish valid reasoning from spurious behavior. We evaluate TaT across dense and Mixture-of-Experts (MoE) architectures on benchmarks spanning commonsense reasoning, question answering, and toxicity detection. Without access to the activations themselves and using only changes in activations across layers, we show that TaT effectively mitigates reliance on static lexical confounds, outperforming conventional probing, and establishes trajectory analysis as a complementary perspective on LLM explainability.
>
---
#### [new 062] CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CHIMERA，一个用于通用推理的紧凑合成数据集，解决数据不足、领域覆盖窄和标注困难的问题，提升大语言模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2603.00889](https://arxiv.org/pdf/2603.00889)**

> **作者:** Xinyu Zhu; Yihao Feng; Yanchao Sun; Xianzhi Du; Pingzhi Li; Olli Saarikivi; Yun Zhu; Yu Meng
>
> **摘要:** Large Language Models (LLMs) have recently exhibited remarkable reasoning capabilities, largely enabled by supervised fine-tuning (SFT)- and reinforcement learning (RL)-based post-training on high-quality reasoning data. However, reproducing and extending these capabilities in open and scalable settings is hindered by three fundamental data-centric challenges: (1) the cold-start problem, arising from the lack of seed datasets with detailed, long Chain-of-Thought (CoT) trajectories needed to initialize reasoning policies; (2) limited domain coverage, as most existing open-source reasoning datasets are concentrated in mathematics, with limited coverage of broader scientific disciplines; and (3) the annotation bottleneck, where the difficulty of frontier-level reasoning tasks makes reliable human annotation prohibitively expensive or infeasible. To address these challenges, we introduce CHIMERA, a compact synthetic reasoning dataset comprising 9K samples for generalizable cross-domain reasoning. CHIMERA is constructed with three key properties: (1) it provides rich, long CoT reasoning trajectories synthesized by state-of-the-art reasoning models; (2) it has broad and structured coverage, spanning 8 major scientific disciplines and over 1K fine-grained topics organized via a model-generated hierarchical taxonomy; and (3) it employs a fully automated, scalable evaluation pipeline that uses strong reasoning models to cross-validate both problem validity and answer correctness. We use CHIMERA to post-train a 4B Qwen3 model. Despite the dataset's modest size, the resulting model achieves strong performance on a suite of challenging reasoning benchmarks, including GPQA-Diamond, AIME 24/25/26, HMMT 25, and Humanity's Last Exam, approaching or matching the reasoning performance of substantially larger models such as DeepSeek-R1 and Qwen3-235B.
>
---
#### [new 063] RAVEL: Reasoning Agents for Validating and Evaluating LLM Text Synthesis
- **分类: cs.CL**

- **简介: 该论文提出RAVEL框架，解决LLM文本合成能力评估不足的问题。通过构建C3EBench基准，分析LLM在不同合成任务中的表现，揭示推理能力对生成质量的关键作用。**

- **链接: [https://arxiv.org/pdf/2603.00686](https://arxiv.org/pdf/2603.00686)**

> **作者:** Andrew Zhuoer Feng; Cunxiang Wang; Yu Luo; Bosi Wen; Yidong Wang; Lin Fan; Yilin Zhou; Zikang Wang; Wenbo Yu; Lindong Wu; Hongning Wang; Minlie Huang
>
> **备注:** 35 pages, 7 figures
>
> **摘要:** Large Language Models have evolved from single-round generators into long-horizon agents, capable of complex text synthesis scenarios. However, current evaluation frameworks lack the ability to assess the actual synthesis operations, such as outlining, drafting, and editing. Consequently, they fail to evaluate the actual and detailed capabilities of LLMs. To bridge this gap, we introduce RAVEL, an agentic framework that enables the LLM testers to autonomously plan and execute typical synthesis operations, including outlining, drafting, reviewing, and refining. Complementing this framework, we present C3EBench, a comprehensive benchmark comprising 1,258 samples derived from professional human writings. We utilize a "reverse-engineering" pipeline to isolate specific capabilities across four tasks: Cloze, Edit, Expand, and End-to-End. Through our analysis of 14 LLMs, we uncover that most LLMs struggle with tasks that demand contextual understanding under limited or under-specified instructions. By augmenting RAVEL with SOTA LLMs as operators, we find that such agentic text synthesis is dominated by the LLM's reasoning capability rather than raw generative capacity. Furthermore, we find that a strong reasoner can guide a weaker generator to yield higher-quality results, whereas the inverse does not hold. Our code and data are available at this link: this https URL.
>
---
#### [new 064] From Prerequisites to Predictions: Validating a Geometric Hallucination Taxonomy Through Controlled Induction
- **分类: cs.CL**

- **简介: 该论文属于AI模型故障分析任务，旨在验证几何幻觉分类的有效性。通过实验测试不同幻觉类型在GPT-2中的区分度，发现覆盖间隙型最显著。**

- **链接: [https://arxiv.org/pdf/2603.00307](https://arxiv.org/pdf/2603.00307)**

> **作者:** Matic Korun
>
> **备注:** 9 pages, 2 figures, appendices (reproducibility, sample generation, additional figures)
>
> **摘要:** We test whether a geometric hallucination taxonomy -- classifying failures as center-drift (Type~1), wrong-well convergence (Type~2), or coverage gaps (Type~3) -- can distinguish hallucination types through controlled induction in GPT-2. Using a two-level statistical design with prompts ($N = 15$/group) as the unit of inference, we run each experiment 20 times with different generation seeds to quantify result stability. In static embeddings, Type~3 norm separation is robust (significant in 18/20 runs, Holm-corrected in 14/20, median $r = +0.61$). In contextual hidden states, the Type~3 norm effect direction is stable (19/20 runs) but underpowered at $N = 15$ (significant in 4/20, median $r = -0.28$). Types~1 and~2 do not separate in either space (${\leq}\,3/20$ runs). Token-level tests inflate significance by 4--16$\times$ through pseudoreplication -- a finding replicated across all 20 runs. The results establish coverage-gap hallucinations as the most geometrically distinctive failure mode, carried by magnitude rather than direction, and confirm the Type~1/2 non-separation as genuine at 124M parameters.
>
---
#### [new 065] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，解决长对话中记忆管理的问题。提出AMemGym框架，用于交互式评估和优化记忆驱动的个性化。**

- **链接: [https://arxiv.org/pdf/2603.01966](https://arxiv.org/pdf/2603.01966)**

> **作者:** Cheng Jiayang; Dongyu Ru; Lin Qiu; Yiyang Li; Xuezhi Cao; Yangqiu Song; Xunliang Cai
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Long-horizon interactions between users and LLM-based assistants necessitate effective memory management, yet current approaches face challenges in training and evaluation of memory. Existing memory benchmarks rely on static, off-policy data as context, limiting evaluation reliability and scalability. To address these gaps, we introduce AMemGym, an interactive environment enabling on-policy evaluation and optimization for memory-driven personalization. AMemGym employs structured data sampling to predefine user profiles, state-dependent questions, and state evolution trajectories, enabling cost-effective generation of high-quality, evaluation-aligned interactions. LLM-simulated users expose latent states through role-play while maintaining structured state consistency. Comprehensive metrics based on structured data guide both assessment and optimization of assistants. Extensive experiments reveal performance gaps in existing memory systems (e.g., RAG, long-context LLMs, and agentic memory) and corresponding reasons. AMemGym not only enables effective selection among competing approaches but also can potentially drive the self-evolution of memory management strategies. By bridging structured state evolution with free-form interactions, our framework provides a scalable, diagnostically rich environment for advancing memory capabilities in conversational agents.
>
---
#### [new 066] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM内部表示难以解释的问题。通过识别关键维度，实现对模型行为的可解释控制。**

- **链接: [https://arxiv.org/pdf/2603.00029](https://arxiv.org/pdf/2603.00029)**

> **作者:** Youngji Roh; Hyunjin Cho; Jaehyung Kim
>
> **摘要:** Large Language Models (LLMs) exhibit highly anisotropic internal representations, often characterized by massive activations, a phenomenon where a small subset of feature dimensions possesses magnitudes significantly larger than the rest. While prior works view these extreme dimensions primarily as artifacts to be managed, we propose a distinct perspective: these dimensions serve as intrinsic interpretable functional units arising from domain specialization. Specifically, we propose a simple magnitude-based criterion to identify Domain-Critical Dimensions in a training-free manner. Our analyses reveal that such dimensions behave as interpretable semantic detectors for symbolic/quantitative patterns or domain-specific terms. In addition, we introduce Critical Dimension Steering, which applies activation steering exclusively to the identified dimensions. Empirical results show that this approach outperforms conventional whole-dimension steering in domain adaptation and jailbreaking scenarios.
>
---
#### [new 067] Self-Anchoring Calibration Drift in Large Language Models: How Multi-Turn Conversations Reshape Model Confidence
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在多轮对话中自信度变化的校准漂移问题，通过实验分析不同模型的表现，探讨自我锚定对模型置信度的影响。**

- **链接: [https://arxiv.org/pdf/2603.01239](https://arxiv.org/pdf/2603.01239)**

> **作者:** Harshavardhan
>
> **摘要:** We introduce Self-Anchoring Calibration Drift (SACD), a hypothesized tendency for large language models (LLMs) to show systematic changes in expressed confidence when building iteratively on their own prior outputs across multi-turn conversations. We report an empirical study comparing three frontier models -- Claude Sonnet 4.6, Gemini 3.1 Pro, and GPT-5.2 -- across 150 questions spanning factual, technical, and open-ended domains, using three conditions: single-turn baseline (A), multi-turn self-anchoring (B), and independent repetition control (C). Results reveal a complex, model-heterogeneous pattern that partially diverges from pre-registered hypotheses. Claude Sonnet 4.6 exhibited significant decreasing confidence under self-anchoring (mean CDS = -0.032, t(14) = -2.43, p = .029, d = -0.627), while also showing significant calibration error drift (F(4,56) = 22.77, p < .001, eta^2 = .791). GPT-5.2 showed the opposite pattern in open-ended domains (mean CDS = +0.026) with significant ECE escalation by Turn 5. Gemini 3.1 Pro showed no significant CDS (t(14) = 0.38, p = .710), but its Condition C data reveals a striking ECE pattern: without self-anchoring, Gemini's calibration error drops from .327 to near zero across repetitions, whereas self-anchoring holds ECE flat at approximately .333 -- indicating that SACD can manifest as suppression of natural calibration improvement rather than ac
>
---
#### [new 068] Zero- and Few-Shot Named-Entity Recognition: Case Study and Dataset in the Crime Domain (CrimeNER)
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于命名实体识别任务，旨在解决犯罪领域标注数据不足的问题。作者构建了CrimeNERdb数据集，并进行了零样本和少量样本的NER实验。**

- **链接: [https://arxiv.org/pdf/2603.02150](https://arxiv.org/pdf/2603.02150)**

> **作者:** Miguel Lopez-Duran; Julian Fierrez; Aythami Morales; Daniel DeAlcala; Gonzalo Mancera; Javier Irigoyen; Ruben Tolosana; Oscar Delgado; Francisco Jurado; Alvaro Ortigosa
>
> **备注:** Sent for review at the main conference of the International Conference of Document Analysis and Recognition (ICDAR) 2026
>
> **摘要:** The extraction of critical information from crime-related documents is a crucial task for law enforcement agencies. Named-Entity Recognition (NER) can perform this task in extracting information about the crime, the criminal, or law enforcement agencies involved. However, there is a considerable lack of adequately annotated data on general real-world crime scenarios. To address this issue, we present CrimeNER, a case-study of Crime-related zero- and Few-Shot NER, and a general Crime-related Named-Entity Recognition database (CrimeNERdb) consisting of more than 1.5k annotated documents for the NER task extracted from public reports on terrorist attacks and the U.S. Department of Justice's press notes. We define 5 types of coarse crime entity and a total of 22 types of fine-grained entity. We address the quality of the case-study and the annotated data with experiments on Zero and Few-Shot settings with State-of-the-Art NER models as well as generalist and commonly used Large Language Models.
>
---
#### [new 069] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型遗忘任务，旨在解决LLMs中知识纠缠与高效遗忘的问题。提出ALTER框架，通过token级隔离实现高效且低副作用的遗忘。**

- **链接: [https://arxiv.org/pdf/2603.01792](https://arxiv.org/pdf/2603.01792)**

> **作者:** Xunlei Chen; Jinyu Guo; Yuang Li; Zhaokun Wang; Yi Gong; Jie Zou; Jiwei Wei; Wenhong Tian
>
> **备注:** Accepted at The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Large language models (LLMs) have advanced to encompass extensive knowledge across diverse domains. Yet controlling what a LLMs should not know is important for ensuring alignment and thus safe use. However, effective unlearning in LLMs is difficult due to the fuzzy boundary between knowledge retention and forgetting. This challenge is exacerbated by entangled parameter spaces from continuous multi-domain training, often resulting in collateral damage, especially under aggressive unlearning strategies. Furthermore, the computational overhead required to optimize State-of-the-Art (SOTA) models with billions of parameters poses an additional barrier. In this work, we present ALTER, a lightweight unlearning framework for LLMs to address both the challenges of knowledge entanglement and unlearning efficiency. ALTER operates through two phases: (I) high entropy tokens are captured and learned via the shared A matrix in LoRA, followed by (II) an asymmetric LoRA architecture that achieves a specified forgetting objective by parameter isolation and unlearning tokens within the target subdomains. Serving as a new research direction for achieving unlearning via token-level isolation in the asymmetric framework. ALTER achieves SOTA performance on TOFU, WMDP, and MUSE benchmarks with over 95% forget quality and shows minimal side effects through preserving foundational tokens. By decoupling unlearning from LLMs' billion-scale parameters, this framework delivers excellent efficiency while preserving over 90% of model utility, exceeding baseline preservation rates of 47.8-83.6%.
>
---
#### [new 070] Reasoning Boosts Opinion Alignment in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于意见建模任务，旨在提升大语言模型在政治观点上的一致性。研究通过结构化推理增强模型的立场对齐，验证其有效性并指出仍需进一步机制以减少偏差。**

- **链接: [https://arxiv.org/pdf/2603.01214](https://arxiv.org/pdf/2603.01214)**

> **作者:** Frédéric Berdoz; Yann Billeter; Yann Vonlanthen; Roger Wattenhofer
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Opinion modeling aims to capture individual or group political preferences, enabling applications such as digital democracies, where models could help shape fairer and more popular policies. Given their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications, large language models (LLMs) are natural candidates for this task. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively. In this work, we study whether reasoning can improve opinion alignment. Motivated by the recent advancement in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate our approach on three datasets covering U.S., European, and Swiss politics. Results indicate that reasoning enhances opinion modeling and is competitive with strong baselines, but does not fully remove bias, highlighting the need for additional mechanisms to build faithful political digital twins using LLMs. By releasing both our method and datasets, we establish a solid baseline to support future research on LLM opinion alignment.
>
---
#### [new 071] Semantic Novelty Trajectories in 80,000 Books: A Cross-Corpus Embedding Analysis
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文本分析任务，研究跨时代书籍的语义新颖性轨迹。通过嵌入方法比较不同时期文学作品的新颖性特征，揭示其变化趋势与结构差异。**

- **链接: [https://arxiv.org/pdf/2603.01791](https://arxiv.org/pdf/2603.01791)**

> **作者:** Fred Zimmerman
>
> **备注:** 12 pages, 4 figures, 5 tables
>
> **摘要:** I apply Schmidhuber's compression progress theory of interestingness at corpus scale, analyzing semantic novelty trajectories in more than 80,000 books spanning two centuries of English-language publishing. Using sentence-transformer paragraph embeddings and a running-centroid novelty measure, I compare 28,730 pre-1920 Project Gutenberg books (PG19) against 52,796 modern English books (Books3, approximately 1990-2010). The principal findings are fourfold. First, mean paragraph-level novelty is roughly 10% higher in modern books (0.503 vs. 0.459). Second, trajectory circuitousness -- the ratio of cumulative path length to net displacement in embedding space -- nearly doubles in the modern corpus (+67%). Third, convergent narrative curves, in which novelty declines toward a settled semantic register, are 2.3x more common in pre-1920 literature. Fourth, novelty is orthogonal to reader quality ratings (r = -0.002), suggesting that interestingness in Schmidhuber's sense is structurally independent of perceived literary merit. Clustering paragraph-level trajectories via PAA-16 representations reveals eight distinct narrative-shape archetypes whose distribution shifts substantially between eras. All analysis code and an interactive exploration toolkit are publicly available at this https URL.
>
---
#### [new 072] Qwen3-Coder-Next Technical Report
- **分类: cs.CL**

- **简介: 该论文介绍Qwen3-Coder-Next，一个专注于编码的开放权重语言模型。旨在提升小参数模型的编码能力，通过强化学习和环境反馈进行训练，适用于代码生成与执行任务。**

- **链接: [https://arxiv.org/pdf/2603.00729](https://arxiv.org/pdf/2603.00729)**

> **作者:** Ruisheng Cao; Mouxiang Chen; Jiawei Chen; Zeyu Cui; Yunlong Feng; Binyuan Hui; Yuheng Jing; Kaixin Li; Mingze Li; Junyang Lin; Zeyao Ma; Kashun Shum; Xuwu Wang; Jinxi Wei; Jiaxi Yang; Jiajun Zhang; Lei Zhang; Zongmeng Zhang; Wenting Zhao; Fan Zhou
>
> **备注:** Authors are listed alphabetically by their last names
>
> **摘要:** We present Qwen3-Coder-Next, an open-weight language model specialized for coding agents. Qwen3-Coder-Next is an 80-billion-parameter model that activates only 3 billion parameters during inference, enabling strong coding capability with efficient inference. In this work, we explore how far strong training recipes can push the capability limits of models with small parameter footprints. To achieve this, we perform agentic training through large-scale synthesis of verifiable coding tasks paired with executable environments, allowing learning directly from environment feedback via mid-training and reinforcement learning. Across agent-centric benchmarks including SWE-Bench and Terminal-Bench, Qwen3-Coder-Next achieves competitive performance relative to its active parameter count. We release both base and instruction-tuned open-weight versions to support research and real-world coding agent development.
>
---
#### [new 073] OpenAutoNLU: Open Source AutoML Library for NLU
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍OpenAutoNLU，一个用于自然语言理解（NLU）的开源自动化机器学习库，解决文本分类和命名实体识别任务，通过数据感知训练方案简化用户操作。**

- **链接: [https://arxiv.org/pdf/2603.01824](https://arxiv.org/pdf/2603.01824)**

> **作者:** Grigory Arshinov; Aleksandr Boriskin; Sergey Senichev; Ayaz Zaripov; Daria Galimzianova; Daniil Karpov; Leonid Sanochkin
>
> **摘要:** OpenAutoNLU is an open-source automated machine learning library for natural language understanding (NLU) tasks, covering both text classification and named entity recognition (NER). Unlike existing solutions, we introduce data-aware training regime selection that requires no manual configuration from the user. The library also provides integrated data quality diagnostics, configurable out-of-distribution (OOD) detection, and large language model (LLM) features, all within a minimal lowcode API. The demo app is accessible here this https URL.
>
---
#### [new 074] PonderLM-3: Adaptive Token-Wise Pondering with Differentiable Masking
- **分类: cs.CL**

- **简介: 该论文提出PonderLM-3，解决模型推理中计算资源分配问题，通过自监督学习实现按需计算，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.02023](https://arxiv.org/pdf/2603.02023)**

> **作者:** He Li; Feichen Song; Boyi Zeng; Shixiang Song; Zhiqin John Xu; Ziwei He; Zhouhan Lin
>
> **摘要:** Test-time scaling has shown that allocating more additional computation at inference can improve generation quality, motivating a natural follow-up question: where should this computation be spent? Building on this insight, we introduce PonderLM-3, a pretraining framework for token-wise adaptive pondering that learns to selectively allocate additional computation under purely self-supervised objectives, built on top of the PonderLM-2 backbone. This makes additional inference computation an allocatable per-token resource, so tokens receive more computation only when it is beneficial, rather than paying a uniform extra cost. To make this allocation learnable while maintaining train-inference consistency, PonderLM-3 injects a differentiable attention mask during pretraining and pairs it with a matching hard pruning rule at inference. PonderLM-3 defines a stronger Pareto frontier: compared with existing recursive or adaptive baselines, it achieves lower pretraining perplexity at equal inference FLOPs. On downstream benchmarks, PonderLM-3 attains comparable performance to fixed-step PonderLM-2 under the same maximum number of additional computation steps, while using fewer inference FLOPs in practice. Overall, PonderLM-3 provides an end-to-end differentiable and train-inference consistent framework for token-wise adaptive computation, enabling additional inference compute to be allocated where it is most useful rather than paid uniformly by every token.
>
---
#### [new 075] Beyond the Grid: Layout-Informed Multi-Vector Retrieval with Parsed Visual Document Representations
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，解决多向量存储瓶颈问题。通过文档解析生成布局感知的子图像嵌入，提升检索效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.01666](https://arxiv.org/pdf/2603.01666)**

> **作者:** Yibo Yan; Mingdong Ou; Yi Cao; Xin Zou; Shuliang Liu; Jiahao Huo; Yu Huang; James Kwok; Xuming Hu
>
> **备注:** Under review
>
> **摘要:** Harnessing the full potential of visually-rich documents requires retrieval systems that understand not just text, but intricate layouts, a core challenge in Visual Document Retrieval (VDR). The prevailing multi-vector architectures, while powerful, face a crucial storage bottleneck that current optimization strategies, such as embedding merging, pruning, or using abstract tokens, fail to resolve without compromising performance or ignoring vital layout cues. To address this, we introduce ColParse, a novel paradigm that leverages a document parsing model to generate a small set of layout-informed sub-image embeddings, which are then fused with a global page-level vector to create a compact and structurally-aware multi-vector representation. Extensive experiments demonstrate that our method reduces storage requirements by over 95% while simultaneously yielding significant performance gains across numerous benchmarks and base models. ColParse thus bridges the critical gap between the fine-grained accuracy of multi-vector retrieval and the practical demands of large-scale deployment, offering a new path towards efficient and interpretable multimodal information systems.
>
---
#### [new 076] Building a Strong Instruction Language Model for a Less-Resourced Language
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决少资源语言（如斯洛文尼亚语）在大型语言模型中的表现问题。通过预训练和微调，构建了性能优越的GaMS3-12B模型。**

- **链接: [https://arxiv.org/pdf/2603.01691](https://arxiv.org/pdf/2603.01691)**

> **作者:** Domen Vreš; Tjaša Arčon; Timotej Petrič; Dario Vajda; Marko Robnik-Šikonja; Iztok Lebar Bajec
>
> **备注:** Currently under review at Natural Language Processing Special Issue on Language Models for Low-Resource Languages
>
> **摘要:** Large language models (LLMs) have become an essential tool for natural language processing and artificial intelligence in general. Current open-source models are primarily trained on English texts, resulting in poorer performance on less-resourced languages and cultures. We present a set of methodological approaches necessary for the successful adaptation of an LLM to a less-resourced language, and demonstrate them using the Slovene language. We present GaMS3-12B, a generative model for Slovene with 12 billion parameters, and demonstrate that it is the best-performing open-source model for Slovene within its parameter range. We adapted the model to the Slovene language using three-stage continual pre-training of the Gemma 3 model, followed by two-stage supervised fine-tuning (SFT). We trained the model on a combination of 140B Slovene, English, Bosnian, Serbian, and Croatian pretraining tokens, and over 200 thousand English and Slovene SFT examples. We evaluate GaMS3-12B on the Slovenian-LLM-Eval datasets, English-to-Slovene translation, and the Slovene LLM arena. We show that the described model outperforms 12B Gemma 3 across all three scenarios and performs comparably to much larger commercial GPT-4o in the Slovene LLM arena, achieving a win rate of over 60 %.
>
---
#### [new 077] Distribution-Aware Companding Quantization of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型的训练效率与性能。通过多token预测提升样本效率，增强模型能力，同时加快推理速度。**

- **链接: [https://arxiv.org/pdf/2603.00364](https://arxiv.org/pdf/2603.00364)**

> **作者:** Athul Radhakrishnan; Siddhant Mohan; Mahima Sachdeva
>
> **摘要:** Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to 3X times faster at inference, even with large batch sizes.
>
---
#### [new 078] CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LoRA在参数效率和细粒度适应上的不足。提出CoMoL框架，通过核心空间专家和路由实现高效多专家融合。**

- **链接: [https://arxiv.org/pdf/2603.00573](https://arxiv.org/pdf/2603.00573)**

> **作者:** Jie Cao; Zhenxuan Fan; Zhuonan Wang; Tianwei Lin; Ziyuan Zhao; Rolan Yan; Wenqiao Zhang; Feifei Shao; Hongwei Wang; Jun Xiao; Siliang Tang
>
> **摘要:** Large language models (LLMs) achieve remarkable performance on diverse downstream and domain-specific tasks via parameter-efficient fine-tuning (PEFT). However, existing PEFT methods, particularly MoE-LoRA architectures, suffer from limited parameter efficiency and coarse-grained adaptation due to the proliferation of LoRA experts and instance-level routing. To address these issues, we propose Core Space Mixture of LoRA (\textbf{CoMoL}), a novel MoE-LoRA framework that incorporates expert diversity, parameter efficiency, and fine-grained adaptation. Specifically, CoMoL introduces two key components: core space experts and core space routing. Core space experts store each expert in a compact core matrix, preserving diversity while controlling parameter growth. Core space routing dynamically selects and activates the appropriate core experts for each token, enabling fine-grained, input-adaptive routing. Activated core experts are then merged via a soft-merging strategy into a single core expert, which is combined with a shared LoRA to form a specialized LoRA module. Besides, the routing network is projected into the same low-rank space as the LoRA matrices, further reducing parameter overhead without compromising expressiveness. Extensive experiments demonstrate that CoMoL retains the adaptability of MoE-LoRA architectures while achieving parameter efficiency comparable to standard LoRA, consistently outperforming existing methods across multiple tasks.
>
---
#### [new 079] Modeling Grammatical Hypothesis Testing in Young Learners: A Sequence-Based Learning Analytics Study of Morphosyntactic Reasoning in an Interactive Game
- **分类: cs.CL**

- **简介: 该论文属于语言学习分析任务，旨在通过游戏动作序列研究儿童语法推理。解决传统评估无法捕捉实时思维过程的问题，分析学生在构造句子时的策略与错误模式。**

- **链接: [https://arxiv.org/pdf/2603.02084](https://arxiv.org/pdf/2603.02084)**

> **作者:** Thierry Geoffre; Trystan Geoffre
>
> **摘要:** This study investigates grammatical reasoning in primary school learners through a sequence-based learning analytics approach, leveraging fine-grained action sequences from an interactive game targeting morphosyntactic agreement in French. Unlike traditional assessments that rely on final answers, we treat each slider movement as a hypothesis-testing action, capturing real-time cognitive strategies during sentence construction. Analyzing 597 gameplay sessions (9,783 actions) from 100 students aged 8-11 in authentic classroom settings, we introduce Hamming distance to quantify proximity to valid grammatical solutions and examine convergence patterns across exercises with varying levels of difficulty. Results reveal that determiners and verbs are key sites of difficulty, with action sequences deviating from left-to-right usual treatment. This suggests learners often fix the verb first and adjust preceding elements. Exercises with fewer solutions exhibit slower and more erratic convergence, while changes in the closest valid solution indicate dynamic hypothesis revision. Our findings demonstrate how sequence-based analytics can uncover hidden dimensions of linguistic reasoning, offering a foundation for real-time scaffolding and teacher-facing tools in linguistically diverse classrooms.
>
---
#### [new 080] LLM-as-an-Annotator: Training Lightweight Models with LLM-Annotated Examples for Aspect Sentiment Tuple Prediction
- **分类: cs.CL**

- **简介: 该论文针对基于方面的情感分析（ABSA）任务，解决人工标注数据成本高的问题。通过LLM生成标注数据，训练轻量模型，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.01778](https://arxiv.org/pdf/2603.01778)**

> **作者:** Nils Constantin Hellwig; Jakob Fehle; Udo Kruschwitz; Christian Wolff
>
> **备注:** Accepted for publication at LREC 2026. Final version will appear in the ACL Anthology
>
> **摘要:** Training models for Aspect-Based Sentiment Analysis (ABSA) tasks requires manually annotated data, which is expensive and time-consuming to obtain. This paper introduces LA-ABSA, a novel approach that leverages Large Language Model (LLM)-generated annotations to fine-tune lightweight models for complex ABSA tasks. We evaluate our approach on five datasets for Target Aspect Sentiment Detection (TASD) and Aspect Sentiment Quad Prediction (ASQP). Our approach outperformed previously reported augmentation strategies and achieved competitive performance with LLM-prompting in low-resource scenarios, while providing substantial energy efficiency benefits. For example, using 50 annotated examples for in-context learning (ICL) to guide the annotation of unlabeled data, LA-ABSA achieved an F1 score of 49.85 for ASQP on the SemEval Rest16 dataset, closely matching the performance of ICL prompting with Gemma-3-27B (51.10), while requiring significantly lower computational resources.
>
---
#### [new 081] GRIP: Geometric Refinement and Adaptive Information Potential for Data Efficiency
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数据高效训练任务，解决LLM训练中数据选择的全局与局部不一致问题。提出GRIP框架，通过几何空间建模和动态采样优化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.00031](https://arxiv.org/pdf/2603.00031)**

> **作者:** Changhao Wang; Jiaolong Yang; Xinhao Yao; Yunfei Yu; Peng Jiao; Lu Yu; Junpeng Fang; Riccardo Cantoro; Qing Cui; Jun Zhou
>
> **摘要:** The performance of Large Language Models (LLMs) is increasingly governed by data efficiency rather than raw scaling volume. However, existing selection methods often decouple global distribution balancing from local instance selection, compromising the hierarchical integrity of the training set. We introduce \textbf{GRIP} (Geometric Refinement and Adaptive Information Potential), a framework that unifies these dimensions by modeling the corpus as an information-dense geometric space. GRIP employs a \textbf{Rapid Adaptation Probe (RAP)} to quantify the information potential of semantic clusters, dynamically re-allocating the sampling budget to regions with the highest representation deficits. Subsequently, we perform Intra-Cluster Selection using a \textbf{length-rectified geometric prior} to counteract embedding density artifacts and preserve long-tail logical sequences. Extensive evaluations on Mixture-of-Experts (MoE) models up to 300B tokens demonstrate that GRIP consistently outperforms state-of-the-art baselines, \textbf{surpassing the performance of models trained on $3\times$ larger uncurated datasets}. Our work establishes a robust geometric foundation for adaptive data curation in large-scale pre-training.
>
---
#### [new 082] SkillCraft: Can LLM Agents Learn to Use Tools Skillfully?
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出SkillCraft基准，用于评估AI代理在长期任务中抽象和复用工具组合的能力。旨在解决工具使用技能的可复用性问题，通过设计复杂场景和轻量评估协议提升效率。**

- **链接: [https://arxiv.org/pdf/2603.00718](https://arxiv.org/pdf/2603.00718)**

> **作者:** Shiqi Chen; Jingze Gai; Ruochen Zhou; Jinghan Zhang; Tongyao Zhu; Junlong Li; Kangrui Wang; Zihan Wang; Zhengyu Chen; Klara Kaleb; Ning Miao; Siyang Gao; Cong Lu; Manling Li; Junxian He; Yee Whye Teh
>
> **备注:** 21 pages. Code: this https URL ; Project page: this https URL
>
> **摘要:** Real-world tool-using agents operate over long-horizon workflows with recurring structure and diverse demands, where effective behavior requires not only invoking atomic tools but also abstracting, and reusing higher-level tool compositions. However, existing benchmarks mainly measure instance-level success under static tool sets, offering limited insight into agents' ability to acquire such reusable skills. We address this gap by introducing SkillCraft, a benchmark explicitly stress-test agent ability to form and reuse higher-level tool compositions, where we call Skills. SkillCraft features realistic, highly compositional tool-use scenarios with difficulty scaled along both quantitative and structural dimensions, designed to elicit skill abstraction and cross-task reuse. We further propose a lightweight evaluation protocol that enables agents to auto-compose atomic tools into executable Skills, cache and reuse them inside and across tasks, thereby improving efficiency while accumulating a persistent library of reusable skills. Evaluating state-of-the-art agents on SkillCraft, we observe substantial efficiency gains, with token usage reduced by up to 80% by skill saving and reuse. Moreover, success rate strongly correlates with tool composition ability at test time, underscoring compositional skill acquisition as a core capability.
>
---
#### [new 083] FLANS at SemEval-2026 Task 7: RAG with Open-Sourced Smaller LLMs for Everyday Knowledge Across Diverse Languages and Cultures
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与SemEval-2025 Task 7，解决跨语言文化日常知识问题，采用RAG与开源小模型，构建文化知识库并支持多语言问答。**

- **链接: [https://arxiv.org/pdf/2603.01910](https://arxiv.org/pdf/2603.01910)**

> **作者:** Liliia Bogdanova; Shiran Sun; Lifeng Han; Natalia Amat Lefort; Flor Miriam Plaza-del-Arco
>
> **摘要:** This system paper describes our participation in the SemEval-2025 Task-7 ``Everyday Knowledge Across Diverse Languages and Cultures''. We attended two subtasks, i.e., Track 1: Short Answer Questions (SAQ), and Track 2: Multiple-Choice Questions (MCQ). The methods we used are retrieval augmented generation (RAGs) with open-sourced smaller LLMs (OS-sLLMs). To better adapt to this shared task, we created our own culturally aware knowledge base (CulKBs) by extracting Wikipedia content using keyword lists we prepared. We extracted both culturally-aware wiki-text and country-specific wiki-summary. In addition to the local CulKBs, we also have one system integrating live online search output via DuckDuckGo. Towards better privacy and sustainability, we aimed to deploy smaller LLMs (sLLMs) that are open-sourced on the Ollama platform. We share the prompts we developed using refinement techniques and report the learning curve of such prompts. The tested languages are English, Spanish, and Chinese for both tracks. Our resources and codes are shared via this https URL
>
---
#### [new 084] LLMs as Strategic Actors: Behavioral Alignment, Risk Calibration, and Argumentation Framing in Geopolitical Simulations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于人工智能与国际关系交叉研究，旨在评估LLMs在地缘政治模拟中的行为表现，解决其决策模式与人类差异的问题。通过对比分析六种LLMs与人类在四个危机场景中的行动选择和解释，探讨其行为一致性、风险判断及论证框架。**

- **链接: [https://arxiv.org/pdf/2603.02128](https://arxiv.org/pdf/2603.02128)**

> **作者:** Veronika Solopova; Viktoria Skorik; Maksym Tereshchenko; Alina Haidun; Ostap Vykhopen
>
> **摘要:** Large language models (LLMs) are increasingly proposed as agents in strategic decision environments, yet their behavior in structured geopolitical simulations remains under-researched. We evaluate six popular state-of-the-art LLMs alongside results from human results across four real-world crisis simulation scenarios, requiring models to select predefined actions and justify their decisions across multiple rounds. We compare models to humans in action alignment, risk calibration through chosen actions' severity, and argumentative framing grounded in international relations theory. Results show that models approximate human decision patterns in base simulation rounds but diverge over time, displaying distinct behavioural profiles and strategy updates. LLM explanations for chosen actions across all models exhibit a strong normative-cooperative framing centered on stability, coordination, and risk mitigation, with limited adversarial reasoning.
>
---
#### [new 085] ActMem: Bridging the Gap Between Memory Retrieval and Reasoning in LLM Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM代理在长时交互中记忆管理与推理脱节的问题。提出ActMem框架，结合记忆检索与因果推理，提升复杂决策能力。**

- **链接: [https://arxiv.org/pdf/2603.00026](https://arxiv.org/pdf/2603.00026)**

> **作者:** Xiaohui Zhang; Zequn Sun; Chengyuan Yang; Yaqin Jin; Yazhong Zhang; Wei Hu
>
> **摘要:** Effective memory management is essential for large language model (LLM) agents handling long-term interactions. Current memory frameworks typically treat agents as passive "recorders" and retrieve information without understanding its deeper implications. They may fail in scenarios requiring conflict detection and complex decision-making. To bridge this critical gap, we propose a novel actionable memory framework called ActMem that integrates memory retrieval with active causal reasoning. ActMem transforms unstructured dialogue history into a structured causal and semantic graph. By leveraging counterfactual reasoning and commonsense completion, it enables agents to deduce implicit constraints and resolve potential conflicts between past states and current intentions. Furthermore, we introduce a comprehensive dataset ActMemEval to evaluate agent reasoning capabilities in logic-driven scenarios, moving beyond the fact-retrieval focus of existing memory benchmarks. Experiments demonstrate that ActMem significantly outperforms state-of-the-art baselines in handling complex, memory-dependent tasks, paving the way for more consistent and reliable intelligent assistants.
>
---
#### [new 086] The Aftermath of DrawEduMath: Vision Language Models Underperform with Struggling Students and Misdiagnose Errors
- **分类: cs.CL; cs.CV; cs.CY**

- **简介: 该论文属于教育AI任务，研究VLM在学生错误识别上的不足。工作是评估11个模型在DrawEduMath上的表现，发现其在处理需要帮助的学生错误时效果差。**

- **链接: [https://arxiv.org/pdf/2603.00925](https://arxiv.org/pdf/2603.00925)**

> **作者:** Li Lucy; Albert Zhang; Nathan Anderson; Ryan Knight; Kyle Lo
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Effective mathematics education requires identifying and responding to students' mistakes. For AI to support pedagogical applications, models must perform well across different levels of student proficiency. Our work provides an extensive, year-long snapshot of how 11 vision-language models (VLMs) perform on DrawEduMath, a QA benchmark involving real students' handwritten, hand-drawn responses to math problems. We find that models' weaknesses concentrate on a core component of math education: student error. All evaluated VLMs underperform when describing work from students who require more pedagogical help, and across all QA, they struggle the most on questions related to assessing student error. Thus, while VLMs may be optimized to be math problem solving experts, our results suggest that they require alternative development incentives to adequately support educational use cases.
>
---
#### [new 087] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MMR-Life基准，用于评估多模态大模型在真实场景下的多图像推理能力，解决缺乏标准化评测的问题。**

- **链接: [https://arxiv.org/pdf/2603.02024](https://arxiv.org/pdf/2603.02024)**

> **作者:** Jiachun Li; Shaoping Huang; Zhuoran Jin; Chenlong Zhang; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** Accepted by ICLR 2026, 78 pages, 60 figures
>
> **摘要:** Recent progress in the reasoning capabilities of multimodal large language models (MLLMs) has empowered them to address more complex tasks such as scientific analysis and mathematical reasoning. Despite their promise, MLLMs' reasoning abilities across different scenarios in real life remain largely unexplored and lack standardized benchmarks for evaluation. To address this gap, we introduce MMR-Life, a comprehensive benchmark designed to evaluate the diverse multimodal multi-image reasoning capabilities of MLLMs across real-life scenarios. MMR-Life consists of 2,646 multiple-choice questions based on 19,108 images primarily sourced from real-world contexts, comprehensively covering seven reasoning types: abductive, analogical, causal, deductive, inductive, spatial, and temporal. Unlike existing reasoning benchmarks, MMR-Life does not rely on domain-specific expertise but instead requires models to integrate information across multiple images and apply diverse reasoning abilities. The evaluation of 37 advanced models highlights the substantial challenge posed by MMR-Life. Even top models like GPT-5 achieve only 58% accuracy and display considerable variance in performance across reasoning types. Moreover, we analyze the reasoning paradigms of existing MLLMs, exploring how factors such as thinking length, reasoning method, and reasoning type affect their performance. In summary, MMR-Life establishes a comprehensive foundation for evaluating, analyzing, and improving the next generation of multimodal reasoning systems.
>
---
#### [new 088] Learning Nested Named Entity Recognition from Flat Annotations
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别任务，解决从扁平标注中学习嵌套实体的问题。通过四种方法探索模型能否仅凭扁平数据实现嵌套识别。**

- **链接: [https://arxiv.org/pdf/2603.00840](https://arxiv.org/pdf/2603.00840)**

> **作者:** Igor Rozhkov; Natalia Loukachevitch
>
> **备注:** Accepted at EACL 2026, 15 pages, 2 figures, 8 tables
>
> **摘要:** Nested named entity recognition identifies entities contained within other entities, but requires expensive multi-level annotation. While flat NER corpora exist abundantly, nested resources remain scarce. We investigate whether models can learn nested structure from flat annotations alone, evaluating four approaches: string inclusions (substring matching), entity corruption (pseudo-nested data), flat neutralization (reducing false negative signal), and a hybrid fine-tuned + LLM pipeline. On NEREL, a Russian benchmark with 29 entity types where 21% of entities are nested, our best combined method achieves 26.37% inner F1, closing 40% of the gap to full nested supervision. Code is available at this https URL.
>
---
#### [new 089] Catalyst-Agent: Autonomous heterogeneous catalyst screening and optimization with an LLM Agent
- **分类: cs.CL**

- **简介: 该论文属于催化剂筛选任务，旨在解决传统方法耗时费力的问题。通过AI代理Catalyst-Agent，结合机器学习模型和数据库API，实现高效催化剂发现与优化。**

- **链接: [https://arxiv.org/pdf/2603.01311](https://arxiv.org/pdf/2603.01311)**

> **作者:** Achuth Chandrasekhar; Janghoon Ock; Amir Barati Farimani
>
> **摘要:** The discovery of novel catalysts tailored for particular applications is a major challenge for the twenty-first century. Traditional methods for this include time-consuming and expensive experimental trial-and-error approaches in labs based on chemical theory or heavily computational first-principles approaches based on density functional theory. Recent studies show that deep learning models like graph neural networks (GNNs) can significantly speed up the screening and discovery of catalyst materials by many orders of magnitude, with very high accuracy and fidelity. In this work, we introduce Catalyst-Agent, a Model Context Protocol (MCP) server-based, LLM-powered AI agent. It can explore vast material databases using the OPTIMADE API, make structural modifications, calculate adsorption energies using Meta FAIRchem's UMA (GNN) model via FAIRchem's AdsorbML workflow and slab construction, and make useful material suggestions to the researcher in a closed-loop manner, including surface-level modifications to refine near-miss candidates. It is tested on three pivotal reactions: the oxygen reduction reaction (ORR), the nitrogen reduction reaction (NRR), and the CO2 reduction reaction (CO2RR). Catalyst-Agent achieves a success rate of 23-34 percent among all the materials it chooses and evaluates, and manages to converge in 1-2 trials per successful material on average. This work demonstrates the potential of AI agents to exercise their planning capabilities and tool use to operationalize the catalyst screening workflow, provide useful, testable hypotheses, and accelerate future scientific discoveries for humanity with minimal human intervention.
>
---
#### [new 090] Noise reduction in BERT NER models for clinical entity extraction
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于临床实体抽取任务，旨在提升BERT NER模型的精度。针对模型召回率高但精确度不足的问题，作者提出噪声去除模型，通过概率密度图等特征减少误报。**

- **链接: [https://arxiv.org/pdf/2603.00022](https://arxiv.org/pdf/2603.00022)**

> **作者:** Kuldeep Jiwani; Yash K Jeengar; Ayush Dhaka
>
> **摘要:** Precision is of utmost importance in the realm of clinical entity extraction from clinical notes and reports. Encoder Models fine-tuned for Named Entity Recognition (NER) are an efficient choice for this purpose, as they don't hallucinate. We pre-trained an in-house BERT over clinical data and then fine-tuned it for NER. These models performed well on recall but could not close upon the high precision range, needed for clinical models. To address this challenge, we developed a Noise Removal model that refines the output of NER. The NER model assigns token-level entity tags along with probability scores for each token. Our Noise Removal (NR) model then analyzes these probability sequences and classifies predictions as either weak or strong. A naïve approach might involve filtering predictions based on low probability values; however, this method is unreliable. Owing to the characteristics of the SoftMax function, Transformer based architectures often assign disproportionately high confidence scores even to uncertain or weak predictions, making simple thresholding ineffective. To address this issue, we adopted a supervised modeling strategy in which the NR model leverages advanced features such as the Probability Density Map (PDM). The PDM captures the Semantic-Pull effect observed within Transformer embeddings, an effect that manifests in the probability distributions of NER class predictions across token sequences. This approach enables the model to classify predictions as weak or strong with significantly improved accuracy. With these NR models we were able to reduce False Positives across various clinical NER models by 50\% to 90\%.
>
---
#### [new 091] More Data, Fewer Diacritics: Scaling Arabic TTS
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语文本转语音（TTS）任务，旨在解决数据不足和缺乏准确变音标注的问题。通过构建自动标注管道，生成4000小时训练数据，并验证大量数据可弥补变音标注的缺失。**

- **链接: [https://arxiv.org/pdf/2603.01622](https://arxiv.org/pdf/2603.01622)**

> **作者:** Ahmed Musleh; Yifan Zhang; Kareem Darwish
>
> **摘要:** Arabic Text-to-Speech (TTS) research has been hindered by the availability of both publicly available training data and accurate Arabic diacritization models. In this paper, we address the limitation by exploring Arabic TTS training on large automatically annotated data. Namely, we built a robust pipeline for collecting Arabic recordings and processing them automatically using voice activity detection, speech recognition, automatic diacritization, and noise filtering, resulting in around 4,000 hours of Arabic TTS training data. We then trained several robust TTS models with voice cloning using varying amounts of data, namely 100, 1,000, and 4,000 hours with and without diacritization. We show that though models trained on diacritized data are generally better, larger amounts of training data compensate for the lack of diacritics to a significant degree. We plan to release a public Arabic TTS model that works without the need for diacritization.
>
---
#### [new 092] SimpleTool: Parallel Decoding for Real-Time LLM Function Calling
- **分类: cs.CL**

- **简介: 该论文提出SimpleTool，解决LLM函数调用中的延迟问题，通过并行解码提升实时性能。属于自然语言处理中的函数调用任务。**

- **链接: [https://arxiv.org/pdf/2603.00030](https://arxiv.org/pdf/2603.00030)**

> **作者:** Xiaoxin Shi; Jiaxin Wan; Linkang Dong; Wei Jiang; Yue Liu; Zengfeng Huang
>
> **摘要:** LLM-based function calling enables intelligent agents to interact with external tools and environments, yet autoregressive decoding imposes a fundamental latency bottleneck that limits real-time applications such as embodied intelligence, game AI, and interactive avatars (e.g., 10 Hz control frequency). We observe that function calling differs fundamentally from free-form text generation: structured outputs exhibit substantial token redundancy (delimiters, parameter names), and arguments exhibit weak causal dependencies. Crucially, these two properties must be exploited jointly to achieve real-time performance. We present SimpleTool, which introduces special tokens that serve a dual role: compressing low-entropy tokens (4-6x reduction) while acting as mode selectors that enable independent parallel generation of function name and arguments. This synergistic design achieves 3-6x end-to-end speedup (up to 9.6x) with only +8.2% parallelization overhead. Experiments on five benchmarks across Qwen-series models (0.5B-14B) demonstrate substantial speedup while maintaining competitive or improved accuracy. On Mobile Actions, ST-Qwen-0.5B outperforms Google's FunctionGemma in both accuracy and latency consistency. With quantization on consumer-grade GPU, SimpleTool achieves 61.2ms P50 latency, enabling 16 Hz real-time control at 4B model scale, bridging the gap between LLM function calling and latency-critical real-world deployment.
>
---
#### [new 093] MemeIntel: Explainable Detection of Propagandistic and Hateful Memes
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于识别煽动性和仇恨表情包的任务，旨在解决自动检测与解释生成联合建模导致的性能下降问题。提出MemeXplain数据集和多阶段优化方法，提升检测与解释质量。**

- **链接: [https://arxiv.org/pdf/2502.16612](https://arxiv.org/pdf/2502.16612)**

> **作者:** Mohamed Bayan Kmainasi; Abul Hasnat; Md Arid Hasan; Ali Ezzat Shahroor; Firoj Alam
>
> **备注:** disinformation, misinformation, factuality, harmfulness, fake news, propaganda, hateful meme, multimodality, text, images
>
> **摘要:** The proliferation of multimodal content on social media presents significant challenges in understanding and moderating complex, context-dependent issues such as misinformation, hate speech, and propaganda. While efforts have been made to develop resources and propose new methods for automatic detection, limited attention has been given to jointly modeling label detection and the generation of explanation-based rationales, which often leads to degraded classification performance when trained simultaneously. To address this challenge, we introduce MemeXplain, an explanation-enhanced dataset for propagandistic memes in Arabic and hateful memes in English, making it the first large-scale resource for these tasks. To solve these tasks, we propose a multi-stage optimization approach and train Vision-Language Models (VLMs). Our results show that this strategy significantly improves both label detection and explanation generation quality over the base model, outperforming the current state-of-the-art with an absolute improvement of ~1.4% (Acc) on ArMeme and ~2.2% (Acc) on Hateful Memes. For reproducibility and future research, we aim to make the MemeXplain dataset and scripts publicly available (this https URL).
>
---
#### [new 094] Legal RAG Bench: an end-to-end benchmark for legal RAG
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出Legal RAG Bench，用于评估法律RAG系统的端到端性能。任务是提升法律信息检索与推理的准确性，通过构建数据集和评估方法，分析检索与模型的影响。**

- **链接: [https://arxiv.org/pdf/2603.01710](https://arxiv.org/pdf/2603.01710)**

> **作者:** Abdur-Rahman Butler; Umar Butler
>
> **备注:** 13 pages, 3 figures, 4 tables
>
> **摘要:** We introduce Legal RAG Bench, a benchmark and evaluation methodology for assessing the end-to-end performance of legal RAG systems. As a benchmark, Legal RAG Bench consists of 4,876 passages from the Victorian Criminal Charge Book alongside 100 complex, hand-crafted questions demanding expert knowledge of criminal law and procedure. Both long-form answers and supporting passages are provided. As an evaluation methodology, Legal RAG Bench leverages a full factorial design and novel hierarchical error decomposition framework, enabling apples-to-apples comparisons of the contributions of retrieval and reasoning models in RAG. We evaluate three state-of-the-art embedding models (Isaacus' Kanon 2 Embedder, Google's Gemini Embedding 001, and OpenAI's Text Embedding 3 Large) and two frontier LLMs (Gemini 3.1 Pro and GPT-5.2), finding that information retrieval is the primary driver of legal RAG performance, with LLMs exerting a more moderate effect on correctness and groundedness. Kanon 2 Embedder, in particular, had the largest positive impact on performance, improving average correctness by 17.5 points, groundedness by 4.5 points, and retrieval accuracy by 34 points. We observe that many errors attributed to hallucinations in legal RAG systems are in fact triggered by retrieval failures, concluding that retrieval sets the ceiling for the performance of many modern legal RAG systems. We document why and how we built Legal RAG Bench alongside the results of our evaluations. We also openly release our code and data to assist with reproduction of our findings.
>
---
#### [new 095] Anatomy of the Modality Gap: Dissecting the Internal States of End-to-End Speech LLMs
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音与文本模态间的性能差距，分析端到端语音大模型的内部表示演化，揭示其结构稳定性及瓶颈所在。**

- **链接: [https://arxiv.org/pdf/2603.01502](https://arxiv.org/pdf/2603.01502)**

> **作者:** Ming-Hao Hsu; Xueyao Zhang; Xiaohai Tian; Jun Zhang; Zhizheng Wu
>
> **摘要:** Recent advancements in Large Speech-Language Models have significantly bridged the gap between acoustic signals and linguistic understanding. However, a persistent performance disparity remains in speech-based input tasks compared to direct text inference. In this paper, we investigate the dynamic roots of this modality gap beyond static geometric alignment, analyzing how speech and text representations evolve layer-by-layer. We evaluate four open-weight end-to-end models on SpeechMMLU and VoiceBench BBH. Using cross-layer CKA analysis with speech-text token alignment, we find that speech representations exhibit a broad cross-layer alignment band, attributable to the redundant nature of speech where semantic content spans multiple frames. We show that these alignment patterns are structurally stable across different analysis configurations. Crucially, simple statistical calibration is insufficient and can be detrimental when applied at the input layer, indicating that the modality gap is not a mere distribution shift. Overall, our results suggest that the bottleneck lies in condensing redundant speech into stable late-layer decisions, motivating future solutions that operate at the token or temporal granularity instead of feature-level matching.
>
---
#### [new 096] Understanding the Physics of Key-Value Cache Compression for LLMs through Attention Dynamics
- **分类: cs.CL**

- **简介: 该论文研究LLM中KV缓存压缩问题，分析其对注意力机制的影响，揭示压缩与语义可达性的关系，提出结构化视角以提升长上下文处理能力。**

- **链接: [https://arxiv.org/pdf/2603.01426](https://arxiv.org/pdf/2603.01426)**

> **作者:** Samhruth Ananthanarayanan; Ayan Sengupta; Tanmoy Chakraborty
>
> **摘要:** As context windows in LLMs scale to 100K+ tokens, the key-value (KV) cache becomes the dominant memory bottleneck, with recent methods claiming 80-90% savings and minimal benchmark degradation. We argue these evaluations miss a structural issue: attention is not just storage but routing, and retaining KV pairs does not guarantee semantic accessibility. We propose a physics-inspired view of KV compression as a controlled perturbation of token-level routing, distinguishing retention, accessibility, and utilization. Using synthetic tasks probing multi-entity tracking, disambiguation, coreference, and multi-hop reasoning, we find that moderate compression degrades internal representations with little accuracy loss, revealing redundancy; all models exhibit a sharp hallucination safety cliff near 90% compression, correlated with spikes in Global Eviction Ratio (GER), suggesting a phase transition in semantic reachability; and architectures differ in routing dynamics, with LLaMA showing early consensus and late diversification, and Qwen showing funnel-like late convergence, leading to distinct resilience profiles. Beyond erasure, we identify representational rigidity, where excessive head-level consensus collapses routing flexibility despite token survival. These results suggest sparse token-route structures govern compression tolerance, reframing KV compression as a structural probe of attention geometry and linking long-context scalability to sparsity and the lottery ticket hypothesis in self-attention.
>
---
#### [new 097] Suffix-Constrained Greedy Search Algorithms for Causal Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型输出中难以提取最终答案的问题。通过引入后缀约束生成算法，确保答案结构化且易解析。**

- **链接: [https://arxiv.org/pdf/2603.01243](https://arxiv.org/pdf/2603.01243)**

> **作者:** Ayoub Hammal; Pierre Zweigenbaum; Caio Corro
>
> **摘要:** Large language models (LLMs) are powerful tools that have found applications beyond human-machine interfaces and chatbots. In particular, their ability to generate reasoning traces motivated their use in many prediction tasks like math question answering. Unfortunately, extracting the final answer in an LLM free-form output is difficult, as it is an information extraction problem on its own. In this work, we introduce suffix-constrained generation, that aims to produce well-formed LLM responses in which final answers follow strict templates and are guaranteed to be trivially parseable. To this end, we introduce several algorithms that are based on greedy search procedures. We experiment on several datasets, and show that our approach allows to guarantee trivial deterministic extraction of the final answer from an LLM output without having a negative impact on results, and even improving them.
>
---
#### [new 098] Polynomial Mixing for Efficient Self-supervised Speech Encoders
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音识别任务，旨在解决自监督语音编码器中自注意力机制复杂度高的问题。提出Polynomial Mixer（PoM）作为替代方案，实现线性复杂度下的高效混合。**

- **链接: [https://arxiv.org/pdf/2603.00683](https://arxiv.org/pdf/2603.00683)**

> **作者:** Eva Feillet; Ryan Whetten; David Picard; Alexandre Allauzen
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** State-of-the-art speech-to-text models typically employ Transformer-based encoders that model token dependencies via self-attention mechanisms. However, the quadratic complexity of self-attention in both memory and computation imposes significant constraints on scalability. In this work, we propose a novel token-mixing mechanism, the Polynomial Mixer (PoM), as a drop-in replacement for multi-head self-attention. PoM computes a polynomial representation of the input with linear complexity with respect to the input sequence length. We integrate PoM into a self-supervised speech representation learning framework based on BEST-RQ and evaluate its performance on downstream speech recognition tasks. Experimental results demonstrate that PoM achieves a competitive word error rate compared to full self-attention and other linear-complexity alternatives, offering an improved trade-off between performance and efficiency in time and memory.
>
---
#### [new 099] Constitutional Black-Box Monitoring for Scheming in LLM Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于安全监控任务，旨在检测LLM代理的隐秘不良行为。通过合成数据训练黑盒监测器，验证其在真实环境中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.00829](https://arxiv.org/pdf/2603.00829)**

> **作者:** Simon Storf; Rich Barton-Cooper; James Peters-Gill; Marius Hobbhahn
>
> **摘要:** Safe deployment of Large Language Model (LLM) agents in autonomous settings requires reliable oversight mechanisms. A central challenge is detecting scheming, where agents covertly pursue misaligned goals. One approach to mitigating such risks is LLM-based monitoring: using language models to examine agent behaviors for suspicious actions. We study constitutional black-box monitors: prompted classifiers that detect scheming using only externally observable inputs and outputs, optimized on synthetic data generated from natural-language behavior specifications. We introduce two pipelines for generating synthetic agent trajectories, STRIDE (iterative refinement) and Gloom (agent-environment simulation), from which we generate 1,000 samples each. We optimize frontier LLM monitors on these datasets via prompt sweeps, human refinement, and automated prompt optimization, and evaluate performance on 7,500 held-out trajectories from ControlArena, a suite of grounded environments where agents operate in more realistic contexts. Our results demonstrate that monitors selected purely on synthetic data can generalize to more realistic environments, capturing a meaningful scheming signal. However, we find that performance saturates quickly in our setting, with simple prompt sweeps matching the results of more extensive optimization. Pushing beyond this limit yields no further improvements and instead leads to overfitting.
>
---
#### [new 100] LaSTR: Language-Driven Time-Series Segment Retrieval
- **分类: cs.CL**

- **简介: 该论文提出LaSTR任务，解决语言驱动的时间序列片段检索问题。通过构建训练数据并使用Conformer模型，在文本与时间序列共享空间中实现精准检索。**

- **链接: [https://arxiv.org/pdf/2603.00725](https://arxiv.org/pdf/2603.00725)**

> **作者:** Kota Dohi; Harsh Purohit; Tomoya Nishida; Takashi Endo; Yusuke Ohtsubo; Koichiro Yawata; Koki Takeshita; Tatsuya Sasaki; Yohei Kawaguchi
>
> **摘要:** Effectively searching time-series data is essential for system analysis, but existing methods often require expert-designed similarity criteria or rely on global, series-level descriptions. We study language-driven segment retrieval: given a natural language query, the goal is to retrieve relevant local segments from large time-series repositories. We build large-scale segment--caption training data by applying TV2-based segmentation to LOTSA windows and generating segment descriptions with GPT-5.2, and then train a Conformer-based contrastive retriever in a shared text--time-series embedding space. On a held-out test split, we evaluate single-positive retrieval together with caption-side consistency (SBERT and VLM-as-a-judge) under multiple candidate pool sizes. Across all settings, LaSTR outperforms random and CLIP baselines, yielding improved ranking quality and stronger semantic agreement between retrieved segments and query intent.
>
---
#### [new 101] LLM-Bootstrapped Targeted Finding Guidance for Factual MLLM-based Medical Report Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于医学报告生成任务，解决MLLM生成报告时的事实不稳定问题。通过分离视觉事实识别与报告生成，并利用LLM自动生成标注数据，提升报告的准确性。**

- **链接: [https://arxiv.org/pdf/2603.00426](https://arxiv.org/pdf/2603.00426)**

> **作者:** Cunyuan Yang; Dejuan Song; Xiaotao Pang; Qianqian Shen; Wenjie Nie; Yifan Huang; Lei Wu; Wei Han; Haishuai Wang; Jiajun Bu
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** The automatic generation of medical reports utilizing Multimodal Large Language Models (MLLMs) frequently encounters challenges related to factual instability, which may manifest as the omission of findings or the incorporation of inaccurate information, thereby constraining their applicability in clinical settings. Current methodologies typically produce reports based directly on image features, which inherently lack a definitive factual basis. In response to this limitation, we introduce Fact-Flow, an innovative framework that separates the process of visual fact identification from the generation of reports. This is achieved by initially predicting clinical findings from the image, which subsequently directs the MLLM to produce a report that is factually precise. A pivotal advancement of our approach is a pipeline that leverages a Large Language Model (LLM) to autonomously create a dataset of labeled medical findings, effectively eliminating the need for expensive manual annotation. Extensive experimental evaluations conducted on two disease-focused medical datasets validate the efficacy of our method, demonstrating a significant enhancement in factual accuracy compared to state-of-the-art models, while concurrently preserving high standards of text quality.
>
---
#### [new 102] Piecing Together Cross-Document Coreference Resolution Datasets: Systematic Dataset Analysis and Unification
- **分类: cs.CL**

- **简介: 该论文属于跨文档共指消解任务，旨在解决数据分散、标准不一的问题。通过构建统一数据集uCDCR，进行标准化分析与比较，提升研究的可复现性和模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.00621](https://arxiv.org/pdf/2603.00621)**

> **作者:** Anastasia Zhukova; Terry Ruas; Jan Philip Wahle; Bela Gipp
>
> **备注:** accepted to LREC 2026
>
> **摘要:** Research in CDCR remains fragmented due to heterogeneous dataset formats, varying annotation standards, and the predominance of the CDCR definition as the event coreference resolution (ECR). To address these challenges, we introduce uCDCR, a unified dataset that consolidates diverse publicly available English CDCR corpora across various domains into a consistent format, which we analyze with standardized metrics and evaluation protocols. uCDCR incorporates both entity and event coreference, corrects known inconsistencies, and enriches datasets with missing attributes to facilitate reproducible research. We establish a cohesive framework for fair, interpretable, and cross-dataset analysis in CDCR and compare the datasets on their lexical properties, e.g., lexical composition of the annotated mentions, lexical diversity and ambiguity metrics, discuss the annotation rules and principles that lead to high lexical diversity, and examine how these metrics influence performance on the same-head-lemma baseline. Our dataset analysis shows that ECB+, the state-of-the-art benchmark for CDCR, has one of the lowest lexical diversities, and its CDCR complexity, measured by the same-head-lemma baseline, lies in the middle among all uCDCR datasets. Moreover, comparing document and mention distributions between ECB+ and uCDCR shows that using all uCDCR datasets for model training and evaluation will improve the generalizability of CDCR models. Finally, the almost identical performance on the same-head-lemma baseline, separately applied to events and entities, shows that resolving both types is a complex task and should not be steered toward ECR alone. The uCDCR dataset is available at this https URL, and the code for parsing, analyzing, and scoring the dataset is available at this https URL.
>
---
#### [new 103] Stepwise Penalization for Length-Efficient Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决推理模型过长链式思维的问题。通过SWAP框架，按步骤优化长度，提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2603.00296](https://arxiv.org/pdf/2603.00296)**

> **作者:** Xintong Li; Sha Li; Rongmei Lin; Hongye Jin; Linwei Li; Hejie Cui; Sarah Zhang; Chia-Yuan Chang; Kewei Cheng; Besnik Fetahu; Priyanka Nigam; Jingbo Shang; Bing Yin
>
> **备注:** Preprint
>
> **摘要:** Large reasoning models improve with more test-time computation, but often overthink, producing unnecessarily long chains-of-thought that raise cost without improving accuracy. Prior reinforcement learning approaches typically rely on a single outcome reward with trajectory-level length penalties, which cannot distinguish essential from redundant reasoning steps and therefore yield blunt compression. Although recent work incorporates step-level signals, such as offline pruning, supervised data construction, or verifier-based intermediate rewards, reasoning length is rarely treated as an explicit step-level optimization objective during RL. We propose Step-wise Adaptive Penalization (SWAP), a fine-grained framework that allocates length reduction across steps based on intrinsic contribution. We estimate step importance from the model's on-policy log-probability improvement toward the correct answer, then treat excess length as a penalty mass redistributed to penalize low-importance steps more heavily while preserving high-importance reasoning. We optimize with a unified outcome-process advantage within group-relative policy optimization. Extensive experiments demonstrate that SWAP reduces reasoning length by 64.3% on average while improving accuracy by 5.7% relative to the base model.
>
---
#### [new 104] QQ: A Toolkit for Language Identifiers and Metadata
- **分类: cs.CL**

- **简介: 该论文提出QQ工具包，解决多语言NLP中语言标识不统一的问题。整合语言资源，实现标识符标准化与映射，便于多语言处理与探索。**

- **链接: [https://arxiv.org/pdf/2603.00620](https://arxiv.org/pdf/2603.00620)**

> **作者:** Wessel Poelman; Yiyi Chen; Miryam de Lhoneux
>
> **备注:** System Demo
>
> **摘要:** The growing number of languages considered in multilingual NLP, including new datasets and tasks, poses challenges regarding properly and accurately reporting which languages are used and how. For example, datasets often use different language identifiers; some use BCP-47 (e.g. en_Latn), others use ISO 639-1 (en), and more linguistically oriented datasets use Glottocodes (stan1293). Mapping between identifiers is manageable for a few dozen languages, but becomes unscalable when dealing with thousands. We introduce QwanQwa, a light-weight Python toolkit for unified language metadata management. QQ integrates multiple language resources into a single interface, provides convenient normalization and mapping between language identifiers, and affords a graph-based structure that enables traversal across families, regions, writing systems, and other linguistic attributes. QQ serves both as (1) a simple "glue" library in multilingual NLP research to make working with many languages easier, and (2) as an intuitive way for exploring languages, such as finding related ones through shared scripts, regions or other metadata.
>
---
#### [new 105] Individual Turing Test: A Case Study of LLM-based Simulation Using Longitudinal Personal Data
- **分类: cs.CL**

- **简介: 该论文属于个体模拟任务，旨在评估LLM是否能准确模仿特定个体。通过构建“个体图灵测试”，比较不同方法的模拟效果，揭示参数与非参数方法的权衡。**

- **链接: [https://arxiv.org/pdf/2603.01289](https://arxiv.org/pdf/2603.01289)**

> **作者:** Minghao Guo; Ziyi Ye; Wujiang Xu; Xi Zhu; Wenyue Hua; Dimitris N. Metaxas
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable human-like capabilities, yet their ability to replicate a specific individual remains under-explored. This paper presents a case study to investigate LLM-based individual simulation with a volunteer-contributed archive of private messaging history spanning over ten years. Based on the messaging data, we propose the "Individual Turing Test" to evaluate whether acquaintances of the volunteer can correctly identify which response in a multi-candidate pool most plausibly comes from the volunteer. We investigate prevalent LLM-based individual simulation approaches including: fine-tuning, retrieval-augmented generation (RAG), memory-based approach, and hybrid methods that integrate fine-tuning and RAG or memory. Empirical results show that current LLM-based simulation methods do not pass the Individual Turing Test, but they perform substantially better when the same test is conducted on strangers to the target individual. Additionally, while fine-tuning improves the simulation in daily chats representing the language style of the individual, retrieval-augmented and memory-based approaches demonstrate stronger performance on questions involving personal opinions and preferences. These findings reveal a fundamental trade-off between parametric and non-parametric approaches to individual simulation with LLMs when given a longitudinal context.
>
---
#### [new 106] Recursive Think-Answer Process for LLMs and VLMs
- **分类: cs.CL**

- **简介: 该论文提出R-TAP方法，解决LLMs和VLMs在单次推理中易出错的问题，通过递归思考与自信评估提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.02099](https://arxiv.org/pdf/2603.02099)**

> **作者:** Byung-Kwan Lee; Youngchae Chee; Yong Man Ro
>
> **备注:** CVPR 2026 Findings, Project page: this https URL
>
> **摘要:** Think-Answer reasoners such as DeepSeek-R1 have made notable progress by leveraging interpretable internal reasoning. However, despite the frequent presence of self-reflective cues like "Oops!", they remain vulnerable to output errors during single-pass inference. To address this limitation, we propose an efficient Recursive Think-Answer Process (R-TAP) that enables models to engage in iterative reasoning cycles and generate more accurate answers, going beyond conventional single-pass approaches. Central to this approach is a confidence generator that evaluates the certainty of model responses and guides subsequent improvements. By incorporating two complementary rewards-Recursively Confidence Increase Reward and Final Answer Confidence Reward-we show that R-TAP-enhanced models consistently outperform conventional single-pass methods for both large language models (LLMs) and vision-language models (VLMs). Moreover, by analyzing the frequency of "Oops"-like expressions in model responses, we find that R-TAP-applied models exhibit significantly fewer self-reflective patterns, resulting in more stable and faster inference-time reasoning. We hope R-TAP pave the way evolving into efficient and elaborated methods to refine the reasoning processes of future AI.
>
---
#### [new 107] TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation
- **分类: cs.CL**

- **简介: 该论文提出TAB-PO方法，解决医疗注释等结构化生成任务中偏好优化的挑战，通过引入令牌级自适应屏障提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.00025](https://arxiv.org/pdf/2603.00025)**

> **作者:** Samah Fodeh; Linhai Ma; Ganesh Puthiaraju; Srivani Talakokkul; Afshan Khan; Ashley Hagaman; Sarah R. Lowe; Aimee Kendall Roundtree
>
> **摘要:** Direct Preference Optimization is an offline post-SFT method for aligning language models from preference pairs, with strong results in instruction following and summarization. However, DPO's sequence-level implicit reward can be brittle for token-critical structured prediction settings such as medical annotation, which often exhibit (i) low-separation preference pairs, where chosen and rejected completions differ by minimal edit distance (often 1-3 tokens), and (ii) token-importance skew, where sparse semantic tokens (hierarchical labels and evidence Spans) carry disproportionate task importance relative to high-frequency structural tokens (JSON scaffolding). In this regime, standard DPO suffers from margin collapse (insufficient log-probability separation between near-identical preferences), likelihood squeezing (the margin objective shifts the absolute likelihoods of both completions together), and gradient dilution, where uniform sequence-level weighting diffuses learning signal across shared scaffolding while rare, confusable label tokens receive weak, noisy updates. We introduce Token-Adaptive Barrier Preference Optimization (TAB-PO), which augments DPO with token-weighted, reference-adjusted advantages that prioritize high-value semantic tokens, and a conditional token-level barrier that regularizes under-confident tokens balancing SFT-anchored likelihood and preference-driven separation in low-separation, importance-skewed regimes. We evaluate TAB-PO on medical communication annotation, a task requiring joint prediction of hierarchical labels and evidence Spans from patient-provider messages. TAB-PO achieves a ~ 4% relative improvement in micro-F1 over SFT and consistently outperforms recent preference-optimization baselines.
>
---
#### [new 108] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在多轮对话中的可靠性问题，通过三个任务评估其表现，发现模型在长时间交互中可靠性下降，提出需加强测试与评估。**

- **链接: [https://arxiv.org/pdf/2603.01423](https://arxiv.org/pdf/2603.01423)**

> **作者:** Jiyoon Myung
>
> **备注:** Accepted at the Workshop on Assessing and Improving Reliability of Foundation Models in the Real World (AAAI 2026)
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in real-world applications where users engage in extended, mixed-topic conversations that depend on prior context. Yet, their reliability under realistic multi-turn interactions remains poorly understood. We conduct a systematic evaluation of conversational reliability through three representative tasks that reflect practical interaction challenges: (1) maintaining global constraints across topic shifts, (2) selecting the correct tool or agent amid interleaved intents, and (3) tracking structured entities under revisions and distractions. Each task pairs single-turn and multi-turn settings, allowing us to quantify reliability degradation under extended dialogue. Across both commercial and open-source models, we observe substantial declines in reliability, particularly for smaller models. Error analyses reveal recurring failure modes such as instruction drift, intent confusion, and contextual overwriting, which compromise dependable behavior in operational systems. Our findings highlight the need for stress-testing LLMs for conversational reliability and developing more robust evaluation methods for trustworthy deployment.
>
---
#### [new 109] Linking Knowledge to Care: Knowledge Graph-Augmented Medical Follow-Up Question Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决LLM在医学随访问题生成中因领域知识不足导致的问题。通过引入知识图谱增强LLM，提升问题生成的准确性和相关性。**

- **链接: [https://arxiv.org/pdf/2603.01252](https://arxiv.org/pdf/2603.01252)**

> **作者:** Liwen Sun; Xiang Yu; Ming Tan; Zhuohao Chen; Anqi Cheng; Ashutosh Joshi; Chenyan Xiong
>
> **备注:** Short paper published in the Findings of EACL 2026
>
> **摘要:** Clinical diagnosis is time-consuming, requiring intensive interactions between patients and medical professionals. While large language models (LLMs) could ease the pre-diagnostic workload, their limited domain knowledge hinders effective medical question generation. We introduce a Knowledge Graph-augmented LLM with active in-context learning to generate relevant and important follow-up questions, KG-Followup, serving as a critical module for the pre-diagnostic assessment. The structured medical domain knowledge graph serves as a seamless patch-up to provide professional domain expertise upon which the LLM can reason. Experiments demonstrate that KG-Followup outperforms state-of-the-art methods by 5% - 8% on relevant benchmarks in recall.
>
---
#### [new 110] GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant
- **分类: cs.CL**

- **简介: 该论文提出GroupGPT，解决多用户聊天中的高效、隐私保护干预问题。通过小大模型协作架构，提升响应准确性和效率，减少token消耗。**

- **链接: [https://arxiv.org/pdf/2603.01059](https://arxiv.org/pdf/2603.01059)**

> **作者:** Zhuokang Shen; Yifan Wang; Hanyu Chen; Wenxuan Huang; Shaohui Lin
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large language models (LLMs) have enabled increasingly capable chatbots. However, most existing systems focus on single-user settings and do not generalize well to multi-user group chats, where agents require more proactive and accurate intervention under complex, evolving contexts. Existing approaches typically rely on LLMs for both reasoning and generation, leading to high token consumption, limited scalability, and potential privacy risks. To address these challenges, we propose GroupGPT, a token-efficient and privacy-preserving agentic framework for multi-user chat assistant. GroupGPT adopts a small-large model collaborative architecture to decouple intervention timing from response generation, enabling efficient and accurate decision-making. The framework also supports multimodal inputs, including memes, images, videos, and voice messages. We further introduce MUIR, a benchmark dataset for multi-user chat assistant intervention reasoning. MUIR contains 2,500 annotated group chat segments with intervention labels and rationales, supporting evaluation of timing accuracy and response quality. We evaluate a range of models on MUIR, from large language models to smaller counterparts. Extensive experiments demonstrate that GroupGPT produces accurate and well-timed responses, achieving an average score of 4.72/5.0 in LLM-based evaluation, and is well received by users across diverse group chat scenarios. Moreover, GroupGPT reduces token usage by up to 3 times compared to baseline methods, while providing privacy sanitization of user messages before cloud transmission. Code is available at: this https URL .
>
---
#### [new 111] Autorubric: A Unified Framework for Rubric-Based LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Autorubric框架，解决LLM文本生成评估中技术分散、术语不一致的问题，支持多种评估方式和偏差缓解，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.00077](https://arxiv.org/pdf/2603.00077)**

> **作者:** Delip Rao; Chris Callison-Burch
>
> **备注:** 43 pages
>
> **摘要:** Rubric-based evaluation with large language models (LLMs) has become standard practice for assessing text generation at scale, yet the underlying techniques are scattered across papers with inconsistent terminology and partial solutions. We present a unified framework: each identified technique is paired with its realization in Autorubric, an open-source Python framework proposed in this paper. Autorubric supports binary, ordinal, and nominal criteria with configurable weights; single-judge and multi-judge ensemble evaluation with majority, weighted, unanimous, and any-vote aggregation; few-shot calibration with verdict-balanced sampling; and mitigations for position bias (option shuffling), verbosity bias (length penalties), and criterion conflation (per-criterion atomic evaluation with natural language explanations). The framework provides reliability metrics drawn from psychometrics (Cohen's $\kappa$, weighted $\kappa$, correlation coefficients, and distribution-level tests) alongside production infrastructure including response caching, checkpointing with resumable runs, multi-provider rate limiting, and cost tracking. We evaluate Autorubric on three benchmarks spanning educational assessment, deep research evaluation, and chatbot quality assessment, demonstrating that it produces results consistent with published benchmarks while exercising the framework's key capabilities: per-criterion binary evaluation with few-shot calibration (RiceChem), multi-judge ensemble evaluation across judge models (ResearcherBench), and mixed criterion types combining binary, ordinal, and nominal scales (CHARM-100). We also contribute CHARM-100, a 100-sample chatbot evaluation dataset with per-sample ground truth labels across all three criterion types, designed to stress-test rubric evaluation frameworks on heterogeneous criteria.
>
---
#### [new 112] nchellwig at SemEval-2026 Task 3: Self-Consistent Structured Generation (SCSG) for Dimensional Aspect-Based Sentiment Analysis using Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对维度方面情感分析任务，提出SCSG方法，通过多次推理并达成共识提升预测可靠性。**

- **链接: [https://arxiv.org/pdf/2603.01788](https://arxiv.org/pdf/2603.01788)**

> **作者:** Nils Constantin Hellwig; Jakob Fehle; Udo Kruschwitz; Christian Wolff
>
> **摘要:** We present Self-Consistent Structured Generation (SCSG) for Dimensional Aspect-Based Sentiment Analysis in SemEval-2026 Task 3 (Track A). SCSG enhances prediction reliability by executing a LoRA-adapted large language model multiple times per instance, retaining only tuples that achieve a majority consensus across runs. To mitigate the computational overhead of multiple forward passes, we leverage vLLM's PagedAttention mechanism for efficient key--value cache reuse. Evaluation across 6 languages and 8 language--domain combinations demonstrates that self-consistency with 15 executions yields statistically significant improvements over single-inference prompting, with our system (leveraging Gemma 3) ranking in the top seven across all settings, achieving second place on three out of four English subsets and first place on Tatar-Restaurant for DimASTE.
>
---
#### [new 113] PanCanBench: A Comprehensive Benchmark for Evaluating Large Language Models in Pancreatic Oncology
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLM在胰腺癌领域临床实用性与准确性的评估问题。通过构建PanCanBench基准，评估模型的临床完整性、事实准确性及搜索整合能力。**

- **链接: [https://arxiv.org/pdf/2603.01343](https://arxiv.org/pdf/2603.01343)**

> **作者:** Yimin Zhao; Sheela R. Damle; Simone E. Dekker; Scott Geng; Karly Williams Silva; Jesse J Hubbard; Manuel F Fernandez; Fatima Zelada-Arenas; Alejandra Alvarez; Brianne Flores; Alexis Rodriguez; Stephen Salerno; Carrie Wright; Zihao Wang; Pang Wei Koh; Jeffrey T. Leek
>
> **摘要:** Large language models (LLMs) have achieved expert-level performance on standardized examinations, yet multiple-choice accuracy poorly reflects real-world clinical utility and safety. As patients and clinicians increasingly use LLMs for guidance on complex conditions such as pancreatic cancer, evaluation must extend beyond general medical knowledge. Existing frameworks, such as HealthBench, rely on simulated queries and lack disease-specific depth. Moreover, high rubric-based scores do not ensure factual correctness, underscoring the need to assess hallucinations. We developed a human-in-the-loop pipeline to create expert rubrics for de-identified patient questions from the Pancreatic Cancer Action Network (PanCAN). The resulting benchmark, PanCanBench, includes 3,130 question-specific criteria across 282 authentic patient questions. We evaluated 22 proprietary and open-source LLMs using an LLM-as-a-judge framework, measuring clinical completeness, factual accuracy, and web-search integration. Models showed substantial variation in rubric-based completeness, with scores ranging from 46.5% to 82.3%. Factual errors were common, with hallucination rates (the percentages of responses containing at least one factual error) ranging from 6.0% for Gemini-2.5 Pro and GPT-4o to 53.8% for Llama-3.1-8B. Importantly, newer reasoning-optimized models did not consistently improve factuality: although o3 achieved the highest rubric score, it produced inaccuracies more frequently than other GPT-family models. Web-search integration did not inherently guarantee better responses. The average score changed from 66.8% to 63.9% for Gemini-2.5 Pro and from 73.8% to 72.8% for GPT-5 when web search was enabled. Synthetic AI-generated rubrics inflated absolute scores by 17.9 points on average while generally maintaining similar relative ranking.
>
---
#### [new 114] Can Thinking Models Think to Detect Hateful Memes?
- **分类: cs.CL**

- **简介: 该论文属于 hateful meme 检测任务，旨在提升模型对隐含有害意图的多模态内容的理解能力。通过强化学习框架优化模型推理过程，提高分类准确性和解释质量。**

- **链接: [https://arxiv.org/pdf/2603.01225](https://arxiv.org/pdf/2603.01225)**

> **作者:** Mohamed Bayan Kmainasi; Mucahid Kutlu; Ali Ezzat Shahroor; Abul Hasnat; Firoj Alam
>
> **摘要:** Hateful memes often require compositional multimodal reasoning: the image and text may appear benign in isolation, yet their interaction conveys harmful intent. Although thinking-based multimodal large language models (MLLMs) have recently advanced vision-language understanding, their capabilities remain underexplored for hateful meme analysis. We propose a reinforcement learning based post-training framework that improves reasoning in thinking-based MLLMs through task-specific rewards and a novel Group Relative Policy Optimization (GRPO) objective. Specifically, we (i) conduct a systematic empirical study of off-the-shelf MLLMs for hateful meme understanding, (ii) extend an existing hateful meme dataset by generating weakly or pseudo-supervised chain-of-thought rationales via distillation, and (iii) introduce a GRPO-based objective that jointly optimizes meme classification and explanation quality to encourage fine-grained, step-by-step reasoning. Experiments on the Hateful Memes benchmark show that our approach achieves state-of-the-art performance, improving accuracy and F1 by approximately 1 percent and explanation quality by approximately 3 percent. We will publicly release our code, dataset extensions, and evaluation resources to support reproducibility.
>
---
#### [new 115] AnnoABSA: A Web-Based Annotation Tool for Aspect-Based Sentiment Analysis with Retrieval-Augmented Suggestions
- **分类: cs.CL**

- **简介: 该论文提出AnnoABSA，一个支持Aspect-Based Sentiment Analysis的网页标注工具，解决人工标注效率低的问题。通过RAG技术提供上下文辅助，提升标注质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.01773](https://arxiv.org/pdf/2603.01773)**

> **作者:** Nils Constantin Hellwig; Jakob Fehle; Udo Kruschwitz; Christian Wolff
>
> **备注:** Accepted for publication at LREC 2026. Final version will appear in the ACL Anthology
>
> **摘要:** We introduce AnnoABSA, the first web-based annotation tool to support the full spectrum of Aspect-Based Sentiment Analysis (ABSA) tasks. The tool is highly customizable, enabling flexible configuration of sentiment elements and task-specific requirements. Alongside manual annotation, AnnoABSA provides optional Large Language Model (LLM)-based retrieval-augmented generation (RAG) suggestions that offer context-aware assistance in a human-in-the-loop approach, keeping the human annotator in control. To improve prediction quality over time, the system retrieves the ten most similar examples that are already annotated and adds them as few-shot examples in the prompt, ensuring that suggestions become increasingly accurate as the annotation process progresses. Released as open-source software under the MIT License, AnnoABSA is freely accessible and easily extendable for research and practical applications.
>
---
#### [new 116] KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型压缩任务，旨在解决知识蒸馏框架训练效率低的问题。提出KDFlow框架，提升训练与推理效率，实现快速模型压缩。**

- **链接: [https://arxiv.org/pdf/2603.01875](https://arxiv.org/pdf/2603.01875)**

> **作者:** Songming Zhang; Xue Zhang; Tong Zhang; Bojie Hu; Yufeng Chen; Jinan Xu
>
> **备注:** 8 pages, 4 figures, 3 tables, code is available at: this https URL
>
> **摘要:** Knowledge distillation (KD) is an essential technique to compress large language models (LLMs) into smaller ones. However, despite the distinct roles of the student model and the teacher model in KD, most existing frameworks still use a homogeneous training backend (e.g., FSDP and DeepSpeed) for both models, leading to suboptimal training efficiency. In this paper, we present a novel framework for LLM distillation, termed \textbf{KDFlow}, which features a decoupled architecture and employs SGLang for teacher inference. By bridging the training efficiency of FSDP2 and the inference efficiency of SGLang, KDFlow achieves full utilization of both advantages in a unified system. Moreover, instead of transferring full logits across different processes, our framework only transmits the teacher's hidden states using zero-copy data transfer and recomputes the logits on the student side, effectively balancing the communication cost and KD performance. Furthermore, our framework supports both off-policy and on-policy distillation and incorporates KD algorithms for cross-tokenizer KD through highly extensible and user-friendly APIs. Experiments show that KDFlow can achieve \textbf{1.44$\times$ to 6.36$\times$} speedup compared to current KD frameworks, enabling researchers to rapidly prototype and scale LLM distillation with minimal engineering overhead. Code is available at: this https URL
>
---
#### [new 117] Efficient Extractive Summarization with MAMBA-Transformer Hybrids for Low-Resource Scenarios
- **分类: cs.CL**

- **简介: 该论文属于摘要生成任务，旨在解决长文档摘要的计算复杂度高和资源受限问题。提出Mamba-Transformer混合模型，提升摘要质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.01288](https://arxiv.org/pdf/2603.01288)**

> **作者:** Nisrine Ait Khayi
>
> **摘要:** Extractive summarization of long documents is bottlenecked by quadratic complexity, often forcing truncation and limiting deployment in resource-constrained settings. We introduce the first Mamba-Transformer hybrid for extractive summarization, combining the semantic strength of pre-trained transformers with the linear-time processing of state space models. Leveraging Mamba's ability to process full documents without truncation, our approach preserves context while maintaining strong summarization quality. The architecture includes: (1) a transformer encoder for sentence-level semantics, (2) a Mamba state space model to capture inter-sentence dependencies efficiently, and (3) a linear classifier for sentence relevance prediction. Across news, argumentative, and scientific domains under low-resource conditions, our method achieves: (1) large gains over BERTSUM and MATCHSUM, including +0.23 ROUGE-1 on ArXiv and statistically significant improvements on all datasets (p < 0.001); (2) consistent advantages across domains, strongest on the longest documents; (3) robust performance with limited training data; and (4) 24-27% faster inference on news summarization (CNN/DailyMail). We introduce the first hybrid Transformer-state space architecture for summarization, showing significant ROUGE improvements in low-resource scenarios.
>
---
#### [new 118] From Variance to Invariance: Qualitative Content Analysis for Narrative Graph Annotation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的叙事图注释任务，旨在解决叙事结构标注的准确性问题。通过引入质性内容分析框架，构建了通胀叙事图数据集，并评估了不同表示和度量方法对标注一致性的影响。**

- **链接: [https://arxiv.org/pdf/2603.01930](https://arxiv.org/pdf/2603.01930)**

> **作者:** Junbo Huang; Max Weinig; Ulrich Fritsche; Ricardo Usbeck
>
> **备注:** LREC 2026 Accepted Paper
>
> **摘要:** Narratives in news discourse play a critical role in shaping public understanding of economic events, such as inflation. Annotating and evaluating these narratives in a structured manner remains a key challenge for Natural Language Processing (NLP). In this work, we introduce a narrative graph annotation framework that integrates principles from qualitative content analysis (QCA) to prioritize annotation quality by reducing annotation errors. We present a dataset of inflation narratives annotated as directed acyclic graphs (DAGs), where nodes represent events and edges encode causal relations. To evaluate annotation quality, we employed a $6\times3$ factorial experimental design to examine the effects of narrative representation (six levels) and distance metric type (three levels) on inter-annotator agreement (Krippendorrf's $\alpha$), capturing the presence of human label variation (HLV) in narrative interpretations. Our analysis shows that (1) lenient metrics (overlap-based distance) overestimate reliability, and (2) locally-constrained representations (e.g., one-hop neighbors) reduce annotation variability. Our annotation and implementation of graph-based Krippendorrf's $\alpha$ are open-sourced. The annotation framework and evaluation results provide practical guidance for NLP research on graph-based narrative annotation under HLV.
>
---
#### [new 119] S-VoCAL: A Dataset and Evaluation Framework for Inferring Speaking Voice Character Attributes in Literature
- **分类: cs.CL**

- **简介: 该论文提出S-VoCAL数据集和评估框架，用于文学中角色语音属性的推理。解决合成叙述中角色特征识别不足的问题，包含8个属性和952对角色与书籍数据。**

- **链接: [https://arxiv.org/pdf/2603.00958](https://arxiv.org/pdf/2603.00958)**

> **作者:** Abigail Berthe-Pardo; Gaspard Michel; Elena V. Epure; Christophe Cerisara
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** With recent advances in Text-to-Speech (TTS) systems, synthetic audiobook narration has seen increased interest, reaching unprecedented levels of naturalness. However, larger gaps remain in synthetic narration systems' ability to impersonate fictional characters, and convey complex emotions or prosody. A promising direction to enhance character identification is the assignment of plausible voices to each fictional characters in a book. This step typically requires complex inference of attributes in book-length contexts, such as a character's age, gender, origin or physical health, which in turns requires dedicated benchmark datasets to evaluate extraction systems' performances. We present S-VoCAL (Speaking Voice Character Attributes in Literature), the first dataset and evaluation framework dedicated to evaluate the inference of voice-related fictional character attributes. S-VoCAL entails 8 attributes grounded in sociophonetic studies, and 952 character-book pairs derived from Project Gutenberg. Its evaluation framework addresses the particularities of each attribute, and includes a novel similarity metric based on recent Large Language Models embeddings. We demonstrate the applicability of S-VoCAL by applying a simple Retrieval-Augmented Generation (RAG) pipeline to the task of inferring character attributes. Our results suggest that the RAG pipeline reliably infers attributes such as Age or Gender, but struggles on others such as Origin or Physical Health. The dataset and evaluation code are available at this https URL .
>
---
#### [new 120] From Literature to Hypotheses: An AI Co-Scientist System for Biomarker-Guided Drug Combination Hypothesis Generation
- **分类: cs.CL**

- **简介: 该论文提出AI Co-Scientist系统，用于癌症研究中的生物标志物引导的药物组合假设生成。任务是解决从文献中提取有效药物组合假设的问题，通过知识图谱和推理方法实现可解释的假设生成与验证。**

- **链接: [https://arxiv.org/pdf/2603.00612](https://arxiv.org/pdf/2603.00612)**

> **作者:** Raneen Younis; Suvinava Basak; Lukas Chavez; Zahra Ahmadi
>
> **摘要:** The rapid growth of biomedical literature and curated databases has made it increasingly difficult for researchers to systematically connect biomarker mechanisms to actionable drug combination hypotheses. We present AI Co-Scientist (CoDHy), an interactive, human-in-the-loop system for biomarker-guided drug combination hypothesis generation in cancer research. CoDHy integrates structured biomedical databases and unstructured literature evidence into a task-specific knowledge graph, which serves as the basis for graph-based reasoning and hypothesis construction. The system combines knowledge graph embeddings with agent-based reasoning to generate, validate, and rank candidate drug combinations, while explicitly grounding each hypothesis in retrievable evidence. Through a web-based interface, users can configure the scientific context, inspect intermediate results, and iteratively refine hypotheses, enabling transparent and researcher-steerable exploration rather than automated decision-making. We demonstrate CoDHy as a system for exploratory hypothesis generation and decision support in translational oncology, highlighting its design, interaction workflow, and practical use cases.
>
---
#### [new 121] Semantic Similarity is a Spurious Measure of Comic Understanding: Lessons Learned from Hallucinations in a Benchmarking Experiment
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于漫画理解任务，旨在解决盲人和视障用户获取漫画内容的问题。研究评估了视觉语言模型在漫画解释中的表现，识别并分类了幻觉现象，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2603.01950](https://arxiv.org/pdf/2603.01950)**

> **作者:** Christopher Driggers-Ellis; Nachiketh Tibrewal; Rohit Bogulla; Harsh Khanna; Sangpil Youm; Christan Grant; Bonnie Dorr
>
> **备注:** 8 pages, 2 figures, 3 tables. Includes link to code
>
> **摘要:** A system that enables blind or visually impaired users to access comics/manga would introduce a new medium of storytelling to this community. However, no such system currently exists. Generative vision-language models (VLMs) have shown promise in describing images and understanding comics, but most research on comic understanding is limited to panel-level analysis. To fully support blind and visually impaired users, greater attention must be paid to page-level understanding and interpretation. In this work, we present a preliminary benchmark of VLM performance on comic interpretation tasks. We identify and categorize hallucinations that emerge during this process, organizing them into generalized object-hallucination taxonomies. We conclude with guidance on future research, emphasizing hallucination mitigation and improved data curation for comic interpretation.
>
---
#### [new 122] Transformers Remember First, Forget Last: Dual-Process Interference in LLMs
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLMs在冲突信息下的记忆保留机制，揭示早期信息优先于近期信息的规律。**

- **链接: [https://arxiv.org/pdf/2603.00270](https://arxiv.org/pdf/2603.00270)**

> **作者:** Sourav Chattaraj; Kanak Raj
>
> **备注:** 16 pages, 10 figures. Under review
>
> **摘要:** When large language models encounter conflicting information in context, which memories survive -- early or recent? We adapt classical interference paradigms from cognitive psychology to answer this question, testing 39 LLMs across diverse architectures and scales. Every model shows the same pattern: proactive interference (PI) dominates retroactive interference (RI) universally (Cohen's d = 1.73, p < 0.0001), meaning early encodings are protected at the cost of recent information -- the opposite of human memory, where RI typically dominates. Three findings indicate that RI and PI reflect separate memory mechanisms. RI and PI are uncorrelated (R^2 = 0.044), rejecting a unified "memory capacity." Model size predicts RI resistance (R^2 = 0.49) but not PI (R^2 = 0.06, n.s.) -- only RI is capacity-dependent. And error analysis reveals distinct failure modes: RI failures are passive retrieval failures (51%), while PI failures show active primacy intrusion (56%); both show <1% hallucination. These patterns parallel the consolidation-retrieval distinction in cognitive science, suggesting that transformer attention creates a primacy bias with direct implications for interference-heavy applications.
>
---
#### [new 123] TraceSIR: A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出TraceSIR框架，解决agentic系统执行轨迹分析难题。通过结构化分析与报告，提升故障诊断效率和准确性。**

- **链接: [https://arxiv.org/pdf/2603.00623](https://arxiv.org/pdf/2603.00623)**

> **作者:** Shu-Xun Yang; Cunxiang Wang; Haoke Zhang; Wenbo Yu; Lindong Wu; Jiayi Gui; Dayong Yang; Yukuo Cen; Zhuoer Feng; Bosi Wen; Yidong Wang; Lucen Zhong; Jiamin Ren; Linfeng Zhang; Jie Tang
>
> **摘要:** Agentic systems augment large language models with external tools and iterative decision making, enabling complex tasks such as deep research, function calling, and coding. However, their long and intricate execution traces make failure diagnosis and root cause analysis extremely challenging. Manual inspection does not scale, while directly applying LLMs to raw traces is hindered by input length limits and unreliable reasoning. Focusing solely on final task outcomes further discards critical behavioral information required for accurate issue localization. To address these issues, we propose TraceSIR, a multi-agent framework for structured analysis and reporting of agentic execution traces. TraceSIR coordinates three specialized agents: (1) StructureAgent, which introduces a novel abstraction format, TraceFormat, to compress execution traces while preserving essential behavioral information; (2) InsightAgent, which performs fine-grained diagnosis including issue localization, root cause analysis, and optimization suggestions; (3) ReportAgent, which aggregates insights across task instances and generates comprehensive analysis reports. To evaluate TraceSIR, we construct TraceBench, covering three real-world agentic scenarios, and introduce ReportEval, an evaluation protocol for assessing the quality and usability of analysis reports aligned with industry needs. Experiments show that TraceSIR consistently produces coherent, informative, and actionable reports, significantly outperforming existing approaches across all evaluation dimensions. Our project and video are publicly available at this https URL.
>
---
#### [new 124] LIDS: LLM Summary Inference Under the Layered Lens
- **分类: cs.LG; cs.CL; stat.ME; stat.ML**

- **简介: 该论文提出LIDS方法，用于评估大语言模型生成摘要的质量。任务是摘要评估，解决如何准确衡量摘要与原文相似性及关键主题的问题。工作包括引入BERT-SVD方向度量和SOFARI算法，提升评估的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.00105](https://arxiv.org/pdf/2603.00105)**

> **作者:** Dylan Park; Yingying Fan; Jinchi Lv
>
> **备注:** 48 pages, 15 figures
>
> **摘要:** Large language models (LLMs) have gained significant attention by many researchers and practitioners in natural language processing (NLP) since the introduction of ChatGPT in 2022. One notable feature of ChatGPT is its ability to generate summaries based on prompts. Yet evaluating the quality of these summaries remains challenging due to the complexity of language. To this end, in this paper we suggest a new method of LLM summary inference with BERT-SVD-based direction metric and SOFARI (LIDS) that assesses the summary accuracy equipped with interpretable key words for layered themes. The LIDS uses a latent SVD-based direction metric to measure the similarity between the summaries and original text, leveraging the BERT embeddings and repeated prompts to quantify the statistical uncertainty. As a result, LIDS gives a natural embedding of each summary for large text reduction. We further exploit SOFARI to uncover important key words associated with each latent theme in the summary with controlled false discovery rate (FDR). Comprehensive empirical studies demonstrate the practical utility and robustness of LIDS through human verification and comparisons to other similarity metrics, including a comparison of different LLMs.
>
---
#### [new 125] A Unified Framework to Quantify Cultural Intelligence of AI
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI文化智能评估任务，旨在解决跨文化情境下AI能力量化问题。提出一个统一框架，整合多维度指标，实现系统化评估。**

- **链接: [https://arxiv.org/pdf/2603.01211](https://arxiv.org/pdf/2603.01211)**

> **作者:** Sunipa Dev; Vinodkumar Prabhakaran; Rutledge Chin Feman; Aida Davani; Remi Denton; Charu Kalia; Piyawat L Kumjorn; Madhurima Maji; Rida Qadri; Negar Rostamzadeh; Renee Shelby; Romina Stella; Hayk Stepanyan; Erin van Liemt; Aishwarya Verma; Oscar Wahltinez; Edem Wornyo; Andrew Zaldivar; Saška Mojsilović
>
> **摘要:** As generative AI technologies are increasingly being launched across the globe, assessing their competence to operate in different cultural contexts is exigently becoming a priority. While recent years have seen numerous and much-needed efforts on cultural benchmarking, these efforts have largely focused on specific aspects of culture and evaluation. While these efforts contribute to our understanding of cultural competence, a unified and systematic evaluation approach is needed for us as a field to comprehensively assess diverse cultural dimensions at scale. Drawing on measurement theory, we present a principled framework to aggregate multifaceted indicators of cultural capabilities into a unified assessment of cultural intelligence. We start by developing a working definition of culture that includes identifying core domains of culture. We then introduce a broad-purpose, systematic, and extensible framework for assessing cultural intelligence of AI systems. Drawing on theoretical framing from psychometric measurement validity theory, we decouple the background concept (i.e., cultural intelligence) from its operationalization via measurement. We conceptualize cultural intelligence as a suite of core capabilities spanning diverse domains, which we then operationalize through a set of indicators designed for reliable measurement. Finally, we identify the considerations, challenges, and research pathways to meaningfully measure these indicators, specifically focusing on data collection, probing strategies, and evaluation metrics.
>
---
#### [new 126] Learning to Read Where to Look: Disease-Aware Vision-Language Pretraining for 3D CT
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于3D CT视觉-语言预训练任务，旨在提升文本到图像检索和疾病分类性能，并解决文本定位具体切片的问题。**

- **链接: [https://arxiv.org/pdf/2603.02026](https://arxiv.org/pdf/2603.02026)**

> **作者:** Simon Ging; Philipp Arnold; Sebastian Walter; Hani Alnahas; Hannah Bast; Elmar Kotter; Jiancheng Yang; Behzad Bozorgtabar; Thomas Brox
>
> **摘要:** Recent 3D CT vision-language models align volumes with reports via contrastive pretraining, but typically rely on limited public data and provide only coarse global supervision. We train a 3D CT vision-language model on 98k report-volume pairs (50k patients) collected at a single hospital, combined with public datasets, using SigLIP-style contrastive pretraining together with prompt-based disease supervision in the shared vision-text embedding space. On CT-RATE, our model achieves state-of-the-art text-to-image retrieval (R@10 31.5 vs. 22.2) and competitive disease classification (AUC 83.8 vs. 83.8), with consistent results on Rad-ChestCT (AUC 77.0 vs. 77.3). We further observe that radiologists routinely reference specific images within their reports (e.g., ``series X, image Y''), linking textual descriptions to precise axial locations. We automatically mine 262k such snippet-slice pairs and introduce the task of intra-scan snippet localization -- predicting the axial depth referred to by a text snippet -- reducing mean absolute error to 36.3 mm at 12 mm feature resolution, compared with 67.0 mm for the best baseline. Adding this localization objective leaves retrieval and classification broadly unchanged within confidence bounds, yielding a single unified model for retrieval, classification, and intra-scan grounding.
>
---
#### [new 127] GenDB: The Next Generation of Query Processing -- Synthesized, Not Engineered
- **分类: cs.DB; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 论文提出GenDB，利用大语言模型合成查询执行代码，解决传统查询系统难以扩展和维护的问题。属于数据库查询处理任务，旨在提升性能与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.02081](https://arxiv.org/pdf/2603.02081)**

> **作者:** Jiale Lao; Immanuel Trummer
>
> **摘要:** Traditional query processing relies on engines that are carefully optimized and engineered by many experts. However, new techniques and user requirements evolve rapidly, and existing systems often cannot keep pace. At the same time, these systems are difficult to extend due to their internal complexity, and developing new systems requires substantial engineering effort and cost. In this paper, we argue that recent advances in Large Language Models (LLMs) are starting to shape the next generation of query processing systems. We propose using LLMs to synthesize execution code for each incoming query, instead of continuously building, extending, and maintaining complex query processing engines. As a proof of concept, we present GenDB, an LLM-powered agentic system that generates instance-optimized and customized query execution code tailored to specific data, workloads, and hardware resources. We implemented an early prototype of GenDB that uses Claude Code Agent as the underlying component in the multi-agent system, and we evaluate it on OLAP workloads. We use queries from the well-known TPC-H benchmark and also construct a new benchmark designed to reduce potential data leakage from LLM training data. We compare GenDB with state-of-the-art query engines, including DuckDB, Umbra, MonetDB, ClickHouse, and PostgreSQL. GenDB achieves significantly better performance than these systems. Finally, we discuss the current limitations of GenDB and outline future extensions and related research challenges.
>
---
#### [new 128] NM-DEKL$^3_\infty$: A Three-Layer Non-Monotone Evolving Dependent Type Logic
- **分类: cs.LO; cs.CL**

- **简介: 该论文提出一种新型依赖类型系统NM-DEKL$^3_\infty$，用于形式化动态环境中的演化知识。解决知识动态建模问题，通过三层架构实现知识表达与推理。**

- **链接: [https://arxiv.org/pdf/2603.01366](https://arxiv.org/pdf/2603.01366)**

> **作者:** Peng Chen
>
> **摘要:** We present a new dependent type system, NM-DEKL$^3_\infty$ (Non-Monotone Dependent Knowledge-Enhanced Logic), for formalising evolving knowledge in dynamic environments. The system uses a three-layer architecture separating a computational layer, a constructive knowledge layer, and a propositional knowledge layer. We define its syntax and semantics and establish Soundness and Equational Completeness; we construct a syntactic model and prove that it is initial in the category of models, from which equational completeness follows. We also give an embedding into the $\mu$-calculus and a strict expressiveness inclusion (including the expressibility of non-bisimulation-invariant properties).
>
---
#### [new 129] Learning from Synthetic Data Improves Multi-hop Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多跳推理任务，旨在解决RL微调数据获取困难的问题。通过使用规则生成的合成数据进行微调，提升LLM的推理能力。**

- **链接: [https://arxiv.org/pdf/2603.02091](https://arxiv.org/pdf/2603.02091)**

> **作者:** Anmol Kabra; Yilun Yin; Albert Gong; Kamilė Stankevičiūtė; Dongyoung Go; Johann Lee; Katie Z. Luo; Carla P. Gomes; Kilian Q. Weinberger
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Reinforcement Learning (RL) has been shown to significantly boost reasoning capabilities of large language models (LLMs) in math, coding, and multi-hop reasoning tasks. However, RL fine-tuning requires abundant high-quality verifiable data, often sourced from human annotations, generated from frontier LLMs, or scored by LLM-based verifiers. All three have considerable limitations: human-annotated datasets are small and expensive to curate, LLM-generated data is hallucination-prone and costly, and LLM-based verifiers are inaccurate and slow. In this work, we investigate a cheaper alternative: RL fine-tuning on rule-generated synthetic data for multi-hop reasoning tasks. We discover that LLMs fine-tuned on synthetic data perform significantly better on popular real-world question-answering benchmarks, despite the synthetic data containing only fictional knowledge. On stratifying performance by question difficulty, we find that synthetic data teaches LLMs to compose knowledge -- a fundamental and generalizable reasoning skill. Our work highlights rule-generated synthetic reasoning data as a free and scalable resource to improve LLM reasoning capabilities.
>
---
#### [new 130] SciDER: Scientific Data-centric End-to-end Researcher
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SciDER，一个数据驱动的科研自动化系统，解决科学发现中数据处理与分析问题，通过协同代理实现从数据到实验设计的全流程自动化。**

- **链接: [https://arxiv.org/pdf/2603.01421](https://arxiv.org/pdf/2603.01421)**

> **作者:** Ke Lin; Yilin Lu; Shreyas Bhat; Xuehang Guo; Junier Oliva; Qingyun Wang
>
> **备注:** 10 pages, 6 figures, 3 tables
>
> **摘要:** Automated scientific discovery with large language models is transforming the research lifecycle from ideation to experimentation, yet existing agents struggle to autonomously process raw data collected from scientific experiments. We introduce SciDER, a data-centric end-to-end system that automates the research lifecycle. Unlike traditional frameworks, our specialized agents collaboratively parse and analyze raw scientific data, generate hypotheses and experimental designs grounded in specific data characteristics, and write and execute corresponding code. Evaluation on three benchmarks shows SciDER excels in specialized data-driven scientific discovery and outperforms general-purpose agents and state-of-the-art models through its self-evolving memory and critic-led feedback loop. Distributed as a modular Python package, we also provide easy-to-use PyPI packages with a lightweight web interface to accelerate autonomous, data-driven research and aim to be accessible to all researchers and developers.
>
---
#### [new 131] Your Inference Request Will Become a Black Box: Confidential Inference for Cloud-based Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私保护任务，解决云上大语言模型推理中的数据泄露问题。通过Talaria框架，将敏感操作置于客户端，确保数据隐私同时保持模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.00196](https://arxiv.org/pdf/2603.00196)**

> **作者:** Chung-ju Huang; Huiqiang Zhao; Yuanpeng He; Lijian Li; Wenpin Jiao; Zhi Jin; Peixuan Chen; Leye Wang
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** The increasing reliance on cloud-hosted Large Language Models (LLMs) exposes sensitive client data, such as prompts and responses, to potential privacy breaches by service providers. Existing approaches fail to ensure privacy, maintain model performance, and preserve computational efficiency simultaneously. To address this challenge, we propose Talaria, a confidential inference framework that partitions the LLM pipeline to protect client data without compromising the cloud's model intellectual property or inference quality. Talaria executes sensitive, weight-independent operations within a client-controlled Confidential Virtual Machine (CVM) while offloading weight-dependent computations to the cloud GPUs. The interaction between these environments is secured by our Reversible Masked Outsourcing (ReMO) protocol, which uses a hybrid masking technique to reversibly obscure intermediate data before outsourcing computations. Extensive evaluations show that Talaria can defend against state-of-the-art token inference attacks, reducing token reconstruction accuracy from over 97.5% to an average of 1.34%, all while being a lossless mechanism that guarantees output identical to the original model without significantly decreasing efficiency and scalability. To the best of our knowledge, this is the first work that ensures clients' prompts and responses remain inaccessible to the cloud, while also preserving model privacy, performance, and efficiency.
>
---
#### [new 132] PleaSQLarify: Visual Pragmatic Repair for Natural Language Database Querying
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于自然语言数据库查询任务，旨在解决用户意图与系统理解不一致的问题。通过引入语用修复机制，提出PleaSQLarify系统，提升查询的准确性和用户控制力。**

- **链接: [https://arxiv.org/pdf/2603.01795](https://arxiv.org/pdf/2603.01795)**

> **作者:** Robin Shing Moon Chan; Rita Sevastjanova; Mennatallah El-Assady
>
> **备注:** Accepted at CHI'26, main track
>
> **摘要:** Natural language database interfaces broaden data access, yet they remain brittle under input ambiguity. Standard approaches often collapse uncertainty into a single query, offering little support for mismatches between user intent and system interpretation. We reframe this challenge through pragmatic inference: while users economize expressions, systems operate on priors over the action space that may not align with the users'. In this view, pragmatic repair -- incremental clarification through minimal interaction -- is a natural strategy for resolving underspecification. We present \textsc{PleaSQLarify}, which operationalizes pragmatic repair by structuring interaction around interpretable decision variables that enable efficient clarification. A visual interface complements this by surfacing the action space for exploration, requesting user disambiguation, and making belief updates traceable across turns. In a study with twelve participants, \textsc{PleaSQLarify} helped users recognize alternative interpretations and efficiently resolve ambiguity. Our findings highlight pragmatic repair as a design principle that fosters effective user control in natural language interfaces.
>
---
#### [new 133] Learn Hard Problems During RL with Reference Guided Fine-tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决数学推理中奖励稀疏问题。通过引用引导微调，生成有效轨迹提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.01223](https://arxiv.org/pdf/2603.01223)**

> **作者:** Yangzhen Wu; Shanda Li; Zixin Wen; Xin Zhou; Ameet Talwalkar; Yiming Yang; Wenhao Huang; Tianle Cai
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Reinforcement learning (RL) for mathematical reasoning can suffer from reward sparsity: for challenging problems, LLM fails to sample any correct trajectories, preventing RL from receiving meaningful positive feedback. At the same time, there often exist human-written reference solutions along with the problem (e.g., problems from AoPS), but directly fine-tuning on these solutions offers no benefit because models often cannot imitate human proofs that lie outside their own reasoning distribution. We introduce Reference-Guided Fine-Tuning (ReGFT), a simple and effective method that utilizes human-written reference solutions to synthesize positive trajectories on hard problems and train on them before RL. For each problem, we provide the model with a partial reference solution and let it generate its own reasoning trace, ensuring the resulting trajectories remain in the model's reasoning space while still benefiting from reference guidance. Fine-tuning on these reference-guided trajectories increases the number of solvable problems and produces a checkpoint that receives more positive rewards during RL. Across three benchmarks (AIME24, AIME25, BeyondAIME), ReGFT consistently improves supervised accuracy, accelerates DAPO training, and raises the final performance plateau of RL. Our results show that ReGFT effectively overcomes reward sparsity and unlocks stronger RL-based mathematical reasoning.
>
---
#### [new 134] RLShield: Practical Multi-Agent RL for Financial Cyber Defense with Attack-Surface MDPs and Real-Time Response Orchestration
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出RLShield，用于金融网络安全的多智能体强化学习框架，解决动态攻击下的实时响应问题，通过MDP建模攻击面并优化协调策略。**

- **链接: [https://arxiv.org/pdf/2603.00186](https://arxiv.org/pdf/2603.00186)**

> **作者:** Srikumar Nayak
>
> **备注:** 6 pages, 2 fig and 2 tables
>
> **摘要:** Financial systems run nonstop and must stay reliable even during cyber incidents. Modern attacks move across many services (apps, APIs, identity, payment rails), so defenders must make a sequence of actions under time pressure. Most security tools still use fixed rules or static playbooks, which can be slow to adapt when the attacker changes behavior. Reinforcement learning (RL) is a good fit for sequential decisions, but much of the RL-in-finance literature targets trading and does not model real cyber response limits such as action cost, service disruption, and defender coordination across many assets. This paper proposes RLShield, a practical multi-agent RL pipeline for financial cyber defense. We model the enterprise attack surface as a Markov decision process (MDP) where states summarize alerts, asset exposure, and service health, and actions represent real response steps (e.g., isolate a host, rotate credentials, ratelimit an API, block an account, or trigger recovery). RLShield learns coordinated policies across multiple agents (assets or service groups) and optimizes a risk-sensitive objective that balances containment speed, business disruption, and response cost. We also include a game-aware evaluation that tests policies against adaptive attackers and reports operational outcomes, not only reward. Experiments show that RLShield reduces time-to-containment and residual exposure while keeping disruption within a fixed response budget, outperforming static rule baselines and single-agent RL under the same constraints. These results suggest that multi-agent, cost-aware RL can provide a deployable layer for automated response in financial security operations.
>
---
#### [new 135] Power Echoes: Investigating Moderation Biases in Online Power-Asymmetric Conflicts
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于人机协作 moderation 研究，旨在解决权力不对称冲突中的偏见问题。通过实验分析人类 moderators 的偏见，并探讨 AI 建议的影响。**

- **链接: [https://arxiv.org/pdf/2603.01457](https://arxiv.org/pdf/2603.01457)**

> **作者:** Yaqiong Li; Peng Zhang; Peixu Hou; Kainan Tu; Guangping Zhang; Shan Qu; Wenshi Chen; Yan Chen; Ning Gu; Tun Lu
>
> **备注:** Accepted at the ACM CHI conference on Human Factors in Computing Systems (ACM CHI 2026)
>
> **摘要:** Online power-asymmetric conflicts are prevalent, and most platforms rely on human moderators to conduct moderation currently. Previous studies have been continuously focusing on investigating human moderation biases in different scenarios, while moderation biases under power-asymmetric conflicts remain unexplored. Therefore, we aim to investigate the types of power-related biases human moderators exhibit in power-asymmetric conflict moderation (RQ1) and further explore the influence of AI's suggestions on these biases (RQ2). For this goal, we conducted a mixed design experiment with 50 participants by leveraging the real conflicts between consumers and merchants as a scenario. Results suggest several biases towards supporting the powerful party within these two moderation modes. AI assistance alleviates most biases of human moderation, but also amplifies a few. Based on these results, we propose several insights into future research on human moderation and human-AI collaborative moderation systems for power-asymmetric conflicts.
>
---
#### [new 136] Recursive Models for Long-Horizon Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究长周期推理任务，解决语言模型在有限上下文下的局限性。提出递归模型，通过自我调用分解问题，提升长周期推理能力。**

- **链接: [https://arxiv.org/pdf/2603.02112](https://arxiv.org/pdf/2603.02112)**

> **作者:** Chenxiao Yang; Nathan Srebro; Zhiyuan Li
>
> **摘要:** Modern language models reason within bounded context, an inherent constraint that poses a fundamental barrier to long-horizon reasoning. We identify recursion as a core principle for overcoming this barrier, and propose recursive models as a minimal realization, where the model can recursively invoke itself to solve subtasks in isolated contexts. We prove that any computable problem admits a recursive decomposition in which each subtask requires only exponentially smaller active context than standard autoregressive models; this strictly surpasses any context management approach confined to a single sequence, such as summarization. We further generalize our framework to modern agentic systems with arbitrary context processing and control flows, and prove that recursive models can achieve optimal power within this broader class. Experimentally, we train a 3B model to reason recursively and evaluate on Boolean satisfiability, a task requiring long-horizon combinatorial search, where it significantly outperforms frontier LLMs.
>
---
#### [new 137] Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning
- **分类: cs.AI; cs.CL; cs.HC; cs.MA**

- **简介: 该论文属于人机协作规划任务，旨在通过自然对话提升用户对AI规划的理解与信任。提出多智能体LLM架构，支持动态解释，并进行用户研究验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.02070](https://arxiv.org/pdf/2603.02070)**

> **作者:** Guilhem Fouilhé; Rebecca Eifler; Antonin Poché; Sylvie Thiébaux; Nicholas Asher
>
> **备注:** Preprint
>
> **摘要:** When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface.
>
---
#### [new 138] Semantic XPath: Structured Agentic Memory Access for Conversational AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Semantic XPath，解决ConvAI中结构化记忆访问问题，通过树状结构提升性能并减少token使用。**

- **链接: [https://arxiv.org/pdf/2603.01160](https://arxiv.org/pdf/2603.01160)**

> **作者:** Yifan Simon Liu; Ruifan Wu; Liam Gallagher; Jiazhou Liang; Armin Toroghi; Scott Sanner
>
> **摘要:** Conversational AI (ConvAI) agents increasingly maintain structured memory to support long-term, task-oriented interactions. In-context memory approaches append the growing history to the model input, which scales poorly under context-window limits. RAG-based methods retrieve request-relevant information, but most assume flat memory collections and ignore structure. We propose Semantic XPath, a tree-structured memory module to access and update structured conversational memory. Semantic XPath improves performance over flat-RAG baselines by 176.7% while using only 9.1% of the tokens required by in-context memory. We also introduce SemanticXPath Chat, an end-to-end ConvAI demo system that visualizes the structured memory and query execution details. Overall, this paper demonstrates a candidate for the next generation of long-term, task-oriented ConvAI systems built on structured memory.
>
---
#### [new 139] RTLocating: Intent-aware RTL Localization for Hardware Design Iteration
- **分类: cs.ET; cs.CL; cs.IR**

- **简介: 该论文提出RTLocating，解决硬件设计中意图驱动的RTL定位问题，通过多视角融合提升定位效果。**

- **链接: [https://arxiv.org/pdf/2603.00434](https://arxiv.org/pdf/2603.00434)**

> **作者:** Changwen Xing; Yanfeng Lu; Lei Qi; Chenxu Niu; Jie Li; Xi Wang; Yong Chen; Jun Yang
>
> **摘要:** Industrial chip development is inherently iterative, favoring localized, intent-driven updates over rewriting RTL from scratch. Yet most LLM-Aided Hardware Design (LAD) work focuses on one-shot synthesis, leaving this workflow underexplored. To bridge this gap, we for the first time formalize $\Delta$Spec-to-RTL localization, a multi-positive problem mapping natural language change requests ($\Delta$Spec) to the affected Register Transfer Level (RTL) syntactic blocks. We propose RTLocating, an intent-aware RTL localization framework, featuring a dynamic router that adaptively fuses complementary views from a textual semantic encoder, a local structural encoder, and a global interaction and dependency encoder (GLIDE). To enable scalable supervision, we introduce EvoRTL-Bench, the first industrial-scale benchmark for intent-code alignment derived from OpenTitan's Git history, comprising 1,905 validated requests and 13,583 $\Delta$Spec-RTL block pairs. On EvoRTL-Bench, RTLocating achieves 0.568 MRR and 15.08% R@1, outperforming the strongest baseline by +22.9% and +67.0%, respectively, establishing a new state-of-the-art for intent-driven localization in evolving hardware designs.
>
---
#### [new 140] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于虚假新闻生成安全评估任务，旨在解决多语言、多地区下大模型对越狱攻击的防御问题。工作包括构建JailNewsBench基准，评估9个模型的攻击成功率和危害性。**

- **链接: [https://arxiv.org/pdf/2603.01291](https://arxiv.org/pdf/2603.01291)**

> **作者:** Masahiro Kaneko; Ayana Niwa; Timothy Baldwin
>
> **备注:** ICLR 2026
>
> **摘要:** Fake news undermines societal trust and decision-making across politics, economics, health, and international relations, and in extreme cases threatens human lives and societal safety. Because fake news reflects region-specific political, social, and cultural contexts and is expressed in language, evaluating the risks of large language models (LLMs) requires a multi-lingual and regional perspective. Malicious users can bypass safeguards through jailbreak attacks, inducing LLMs to generate fake news. However, no benchmark currently exists to systematically assess attack resilience across languages and regions. Here, we propose JailNewsBench, the first benchmark for evaluating LLM robustness against jailbreak-induced fake news generation. JailNewsBench spans 34 regions and 22 languages, covering 8 evaluation sub-metrics through LLM-as-a-Judge and 5 jailbreak attacks, with approximately 300k instances. Our evaluation of 9 LLMs reveals that the maximum attack success rate (ASR) reached 86.3% and the maximum harmfulness score was 3.5 out of 5. Notably, for English and U.S.-related topics, the defensive performance of typical multi-lingual LLMs was significantly lower than for other regions, highlighting substantial imbalances in safety across languages and regions. In addition, our analysis shows that coverage of fake news in existing safety datasets is limited and less well defended than major categories such as toxicity and social bias. Our dataset and code are available at this https URL.
>
---
#### [new 141] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.MM**

- **简介: 该论文属于视频理解任务，旨在解决长时视频分析中记忆不足的问题。提出MM-Mem架构，通过语义瓶颈优化记忆压缩与信息保留，提升长期视频代理的推理能力。**

- **链接: [https://arxiv.org/pdf/2603.01455](https://arxiv.org/pdf/2603.01455)**

> **作者:** Niu Lian; Yuting Wang; Hanshu Yao; Jinpeng Wang; Bin Chen; Yaowei Wang; Min Zhang; Shu-Tao Xia
>
> **备注:** TL;DR: We propose MM-Mem, a cognition-inspired, dual-trace hierarchical memory framework for long-horizon video understanding grounded in Fuzzy-Trace Theory. It features adaptive memory compression via the Information Bottleneck and employs an entropy-driven top-down retrieval to access fine-grained details only when necessary. 16 pages, 7 figures, 7 tables
>
> **摘要:** While multimodal large language models have demonstrated impressive short-term reasoning, they struggle with long-horizon video understanding due to limited context windows and static memory mechanisms that fail to mirror human cognitive efficiency. Existing paradigms typically fall into two extremes: vision-centric methods that incur high latency and redundancy through dense visual accumulation, or text-centric approaches that suffer from detail loss and hallucination via aggressive captioning. To bridge this gap, we propose MM-Mem, a pyramidal multimodal memory architecture grounded in Fuzzy-Trace Theory. MM-Mem structures memory hierarchically into a Sensory Buffer, Episodic Stream, and Symbolic Schema, enabling the progressive distillation of fine-grained perceptual traces (verbatim) into high-level semantic schemas (gist). Furthermore, to govern the dynamic construction of memory, we derive a Semantic Information Bottleneck objective and introduce SIB-GRPO to optimize the trade-off between memory compression and task-relevant information retention. In inference, we design an entropy-driven top-down memory retrieval strategy, which first tries with the abstract Symbolic Schema and progressively "drills down" to the Sensory Buffer and Episodic Stream under high uncertainty. Extensive experiments across 4 benchmarks confirm the effectiveness of MM-Mem on both offline and streaming tasks, demonstrating robust generalization and validating the effectiveness of cognition-inspired memory organization. Code is available at this https URL.
>
---
#### [new 142] OmniRet: Efficient and High-Fidelity Omni Modality Retrieval
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文提出OmniRet，解决多模态检索中仅支持文本和视觉的问题，首次支持文本、视觉和音频三模态，提升检索效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.02098](https://arxiv.org/pdf/2603.02098)**

> **作者:** Chuong Huynh; Manh Luong; Abhinav Shrivastava
>
> **备注:** CVPR 2026. Project link: this https URL
>
> **摘要:** Multimodal retrieval is the task of aggregating information from queries across heterogeneous modalities to retrieve desired targets. State-of-the-art multimodal retrieval models can understand complex queries, yet they are typically limited to two modalities: text and vision. This limitation impedes the development of universal retrieval systems capable of comprehending queries that combine more than two modalities. To advance toward this goal, we present OmniRet, the first retrieval model capable of handling complex, composed queries spanning three key modalities: text, vision, and audio. Our OmniRet model addresses two critical challenges for universal retrieval: computational efficiency and representation fidelity. First, feeding massive token sequences from modality-specific encoders to Large Language Models (LLMs) is computationally inefficient. We therefore introduce an attention-based resampling mechanism to generate compact, fixed-size representations from these sequences. Second, compressing rich omni-modal data into a single embedding vector inevitably causes information loss and discards fine-grained details. We propose Attention Sliced Wasserstein Pooling to preserve these fine-grained details, leading to improved omni-modal representations. OmniRet is trained on an aggregation of approximately 6 million query-target pairs spanning 30 datasets. We benchmark our model on 13 retrieval tasks and a MMEBv2 subset. Our model demonstrates significant improvements on composed query, audio and video retrieval tasks, while achieving on-par performance with state-of-the-art models on others. Furthermore, we curate a new Audio-Centric Multimodal Benchmark (ACM). This new benchmark introduces two critical, previously missing tasks-composed audio retrieval and audio-visual retrieval to more comprehensively evaluate a model's omni-modal embedding capacity.
>
---
#### [new 143] VoxKnesset: A Large-Scale Longitudinal Hebrew Speech Dataset for Aging Speaker Modeling
- **分类: eess.AS; cs.CL; cs.LG; cs.SD; eess.SP**

- **简介: 该论文提出VoxKnesset数据集，用于研究语音随年龄变化的问题。任务为说话人建模与年龄预测，解决语音系统在长期变化中的性能下降问题。工作包括数据收集、模型基准测试及结果分析。**

- **链接: [https://arxiv.org/pdf/2603.01270](https://arxiv.org/pdf/2603.01270)**

> **作者:** Yanir Marmor; Arad Zulti; David Krongauz; Adam Gabet; Yoad Snapir; Yair Lifshitz; Eran Segal
>
> **备注:** 4 pages, 5 figures, 2 tables
>
> **摘要:** Speech processing systems face a fundamental challenge: the human voice changes with age, yet few datasets support rigorous longitudinal evaluation. We introduce VoxKnesset, an open-access dataset of ~2,300 hours of Hebrew parliamentary speech spanning 2009-2025, comprising 393 speakers with recording spans of up to 15 years. Each segment includes aligned transcripts and verified demographic metadata from official parliamentary records. We benchmark modern speech embeddings (WavLM-Large, ECAPA-TDNN, Wav2Vec2-XLSR-1B) on age prediction and speaker verification under longitudinal conditions. Speaker verification EER rises from 2.15\% to 4.58\% over 15 years for the strongest model, and cross-sectionally trained age regressors fail to capture within-speaker aging, while longitudinally trained models recover a meaningful temporal signal. We publicly release the dataset and pipeline to support aging-robust speech systems and Hebrew speech processing.
>
---
#### [new 144] DARS: Dysarthria-Aware Rhythm-Style Synthesis for ASR Enhancement
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决失语症语音识别困难的问题。提出DARS框架，通过合成失语症语音增强ASR性能。**

- **链接: [https://arxiv.org/pdf/2603.01369](https://arxiv.org/pdf/2603.01369)**

> **作者:** Minghui Wu; Xueling Liu; Jiahuan Fan; Haitao Tang; Yanyong Zhang; Yue Zhang
>
> **备注:** Submitted to 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** Dysarthric speech exhibits abnormal prosody and significant speaker variability, presenting persistent challenges for automatic speech recognition (ASR). While text-to-speech (TTS)-based data augmentation has shown potential, existing methods often fail to accurately model the pathological rhythm and acoustic style of dysarthric speech. To address this, we propose DARS, a dysarthria-aware rhythm-style synthesis framework based on the Matcha-TTS architecture. DARS incorporates a multi-stage rhythm predictor optimized by contrastive preferences between normal and dysarthric speech, along with a dysarthric-style conditional flow matching mechanism, jointly enhancing temporal rhythm reconstruction and pathological acoustic style simulation. Experiments on the TORGO dataset demonstrate that DARS achieves a Mean Cepstral Distortion (MCD) of 4.29, closely approximating real dysarthric speech. Adapting a Whisper-based ASR system with synthetic dysarthric speech from DARS achieves a 54.22% relative reduction in word error rate (WER) compared to state-of-the-art methods, demonstrating the framework's effectiveness in enhancing recognition performance.
>
---
#### [new 145] DeepXiv-SDK: An Agentic Data Interface for Scientific Papers
- **分类: cs.DL; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出DeepXiv-SDK，解决科学论文访问效率低的问题，通过结构化视图和预算提示提升信息检索与证据验证效果。属于AI4Science任务。**

- **链接: [https://arxiv.org/pdf/2603.00084](https://arxiv.org/pdf/2603.00084)**

> **作者:** Hongjin Qian; Ziyi Xia; Ze Liu; Jianlv Chen; Kun Luo; Minghao Qin; Chaofan Li; Lei Xiong; Sen Wang; Zhengyang Liang; Zheng Liu
>
> **备注:** Project at this https URL
>
> **摘要:** Research agents are increasingly used in AI4Science for scientific information seeking and evidence-grounded decision making. Yet a persistent bottleneck is paper access: agents typically retrieve PDF/HTML pages, heuristically parse them, and ingest long unstructured text, leading to token-heavy reading and brittle evidence lookup. This motivates an agentic data interface for scientific papers that standardizes access, exposes budget-aware views, and treats grounding as a first-class operation. We introduce DeepXiv-SDK, which enables progressive access aligned with how agents allocate attention and reading budget. DeepXiv-SDK exposes as structured views a header-first view for screening, a section-structured view for targeted navigation, and on-demand evidence-level access for verification. Each layer is augmented with enriched attributes and explicit budget hints, so agents can balance relevance, cost, and grounding before escalating to full-text processing. DeepXiv-SDK also supports multi-faceted retrieval and aggregation over paper attributes, enabling constraint-driven search and curation over paper sets. DeepXiv-SDK is currently deployed at arXiv scale with daily synchronization to new releases and is designed to extend to other open-access corpora (e.g., PubMed Central, bioRxiv). We release RESTful APIs, an open-source Python SDK, and a web demo showcasing deep search and deep research workflows; the service is free to use with registration.
>
---
#### [new 146] Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from an Industry Deployment
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，研究生产环境中RAG系统的检索融合效果。针对高召回是否提升回答质量的问题，通过实验发现融合技术在实际约束下效果有限，且增加延迟，提出需综合评估系统效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.02153](https://arxiv.org/pdf/2603.02153)**

> **作者:** Luigi Medrano; Arush Verma; Mukul Chhabra
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems commonly adopt retrieval fusion techniques such as multi-query retrieval and reciprocal rank fusion (RRF) to increase document recall, under the assumption that higher recall leads to better answer quality. While these methods show consistent gains in isolated retrieval benchmarks, their effectiveness under realistic production constraints remains underexplored. In this work, we evaluate retrieval fusion in a production-style RAG pipeline operating over an enterprise knowledge base, with fixed retrieval depth, re-ranking budgets, and latency constraints. Across multiple fusion configurations, we find that retrieval fusion does increase raw recall, but these gains are largely neutralized after re-ranking and truncation. In our setting, fusion variants fail to outperform single-query baselines on KB-level Top-$k$ accuracy, with Hit@10 decreasing from $0.51$ to $0.48$ in several configurations. Moreover, fusion introduces additional latency overhead due to query rewriting and larger candidate sets, without corresponding improvements in downstream effectiveness. Our analysis suggests that recall-oriented fusion techniques exhibit diminishing returns once realistic re-ranking limits and context budgets are applied. We conclude that retrieval-level improvements do not reliably translate into end-to-end gains in production RAG systems, and argue for evaluation frameworks that jointly consider retrieval quality, system efficiency, and downstream impact.
>
---
#### [new 147] Linguistic Uncertainty and Engagement in Arabic-Language X (formerly Twitter) Discourse
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于社会媒体分析任务，研究阿拉伯语推文中的语言不确定性与用户参与度的关系。通过分类器识别不确定表达，发现其显著提升互动水平。**

- **链接: [https://arxiv.org/pdf/2603.00082](https://arxiv.org/pdf/2603.00082)**

> **作者:** Mohamed Soufan
>
> **备注:** 15 pages, 1 figure, 1 table
>
> **摘要:** Linguistic uncertainty is a common feature of social media discourse, yet its relationship with user engagement remains underexplored, particularly in non-English contexts. Using a dataset of 16,695 Arabic-language tweets about Lebanon posted over a 35-day period, we examine whether tweets expressing linguistic uncertainty receive different levels and forms of engagement compared to certainty-marked tweets. We develop a lexicon-based, context-sensitive classifier to identify uncertainty markers and classify 29.9% of tweets as uncertain. Descriptive analyses indicate that uncertain tweets exhibit 51.5% higher mean total engagement (likes, retweets, and replies). Regression models controlling for tweet length, URL presence, and account verification status confirm a positive association between uncertainty and engagement (\b{eta} = 0.221, SE = 0.044, p < 0.001), corresponding to approximately 25% higher expected engagement. The association is strongest for replies, followed by retweets and likes, suggesting a shift toward more conversational forms of engagement. Results are robust to alternative model specifications and adjustments for within-account correlation. These findings suggest that linguistic uncertainty may function as an interactional cue that encourages participatory engagement in Arabic-language social media discourse. The study contributes computational approaches for modeling linguistic features in large-scale, non-English digital communication.
>
---
#### [new 148] Commitment Checklist: Auditing Author Commitments in Peer Review
- **分类: cs.CY; cs.CL; cs.DL**

- **简介: 该论文属于学术诚信任务，旨在解决作者未履行同行评审承诺的问题。通过LLM审计承诺执行情况，提出作者承诺清单以提升透明度与责任性。**

- **链接: [https://arxiv.org/pdf/2603.00003](https://arxiv.org/pdf/2603.00003)**

> **作者:** Chung-Chi Chen; Iryna Gurevych
>
> **摘要:** Peer review author responses often include commitments to add experiments, release code, or clarify content in the final paper. Yet, there is currently no systematic mechanism to ensure authors fulfill these promises. In this position paper, we present a large-scale audit of author commitments using large language models (LLMs) to compare rebuttals against camera-ready versions. Analyzing the commitments from ICLR-2025 and EMNLP-2024, we find that while a majority of promised changes are implemented, a significant share (about 25%) are not, with "missing experiments" and other high-impact items among the most frequently unfulfilled. We demonstrate that LLM-based tools can feasibly detect the promises. Finally, we propose the idea of Author Commitment Checklist, which would alert authors and organizers to unaddressed promises, increasing accountability and strengthening the integrity of the peer review process. We discuss the benefits of this practice and advocate for its adoption in future conferences.
>
---
#### [new 149] Efficient RLVR Training via Weighted Mutual Information Data Selection
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决训练数据选择效率问题。提出InSight方法，基于加权互信息选择数据，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.01907](https://arxiv.org/pdf/2603.01907)**

> **作者:** Xinyu Zhou; Boyu Zhu; Haotian Zhang; Huiming Wang; Zhijiang Guo
>
> **备注:** 15 Pages
>
> **摘要:** Reinforcement learning (RL) plays a central role in improving the reasoning and alignment of large language models, yet its efficiency critically depends on how training data are selected. Existing online selection strategies predominantly rely on difficulty-based heuristics, favouring datapoints with intermediate success rates, implicitly equating difficulty with informativeness and neglecting epistemic uncertainty arising from limited evidence. We introduce InSight, an INformation-guided data SamplInG metHod for RL Training, grounded in a weighted mutual information objective. By modeling data outcomes with Bayesian latent success rates, we show that expected uncertainty reduction decomposes into complementary difficulty- and evidence-dependent components, revealing a fundamental limitation of difficulty-only selection. Leveraging this observation, InSight constructs a stable acquisition score based on the mean belief of datapoints' success rather than noisy sampled outcomes, and naturally extends to multi-rollout settings common in reinforcement learning with verifiable rewards (RLVR). Extensive experiments demonstrate that InSight consistently achieves state-of-the-art performance and improves training efficiency, including a +1.41 average gain on Planning & Mathmatics benchmarks, +1.01 improvement on general reasoning, and up to ~2.2x acceleration, with negligible additional computational overhead.
>
---
#### [new 150] ProtRLSearch: A Multi-Round Multimodal Protein Search Agent with Large Language Models Trained via Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ProtRLSearch，解决蛋白质搜索中单轮文本输入和搜索过程约束不足的问题，通过多轮多模态输入提升搜索质量。**

- **链接: [https://arxiv.org/pdf/2603.01464](https://arxiv.org/pdf/2603.01464)**

> **作者:** Congying Liu; Taihao Li; Ming Huang; Xingyuan Wei; Peipei Liu; Yiqing Shen; Yanxu Mao; Tiehan Cui
>
> **摘要:** Protein analysis tasks arising in healthcare settings often require accurate reasoning under protein sequence constraints, involving tasks such as functional interpretation of disease-related variants, protein-level analysis for clinical research, and similar scenarios. To address such tasks, search agents are introduced to search protein-related information, providing support for disease-related variant analysis and protein function reasoning in protein-centric inference. However, such search agents are mostly limited to single-round, text-only modality search, which prevents the protein sequence modality from being incorporated as a multimodal input into the search decision-making process. Meanwhile, their reliance on reinforcement learning (RL) supervision that focuses solely on the final answer results in a lack of search process constraints, making deviations in keyword selection and reasoning directions difficult to identify and correct in a timely manner. To address these limitations, we propose ProtRLSearch, a multi-round protein search agent trained with multi-dimensional reward based RL, which jointly leverages protein sequence and text as multimodal inputs during real-time search to produce high quality reports. To evaluate the ability of models to integrate protein sequence information and text-based multimodal inputs in realistic protein query settings, we construct ProtMCQs, a benchmark of 3,000 multiple choice questions (MCQs) organized into three difficulty levels. The benchmark evaluates protein query tasks that range from sequence constrained reasoning about protein function and phenotype changes to comprehensive protein reasoning that integrates multi-dimensional sequence features with signal pathways and regulatory networks.
>
---
#### [new 151] Designing Explainable AI for Healthcare Reviews: Guidance on Adoption and Trust
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于医疗AI可解释性研究，旨在解决患者难以处理大量医疗评价的问题。通过混合方法研究，设计可解释AI系统，提升用户信任与采纳。**

- **链接: [https://arxiv.org/pdf/2603.00072](https://arxiv.org/pdf/2603.00072)**

> **作者:** Eman Alamoudi; Ellis Solaiman
>
> **摘要:** Patients increasingly rely on online reviews when choosing healthcare providers, yet the sheer volume of these reviews can hinder effective decision-making. This paper summarises a mixed-methods study aimed at evaluating a proposed explainable AI system that analyses patient reviews and provides transparent explanations for its outputs. The survey (N=60) indicated broad optimism regarding usefulness (82% agreed it saves time; 78% that it highlights essentials), alongside strong demand for explainability (84% considered it important to understand why a review is classified; 82% said explanations would increase trust). Around 45% preferred combined text-and-visual explanations. Thematic analysis of open-ended survey responses revealed core requirements such as accuracy, clarity and simplicity, responsiveness, data credibility, and unbiased processing. In addition, interviews with AI experts provided deeper qualitative insights, highlighting technical considerations and potential challenges for different explanation methods. Drawing on TAM and trust in automation, the findings suggest that high perceived usefulness and transparent explanations promote adoption, whereas complexity and inaccuracy hinder it. This paper contributes actionable design guidance for layered, audience-aware explanations in healthcare review systems.
>
---
#### [new 152] How effective are VLMs in assisting humans in inferring the quality of mental models from Multimodal short answers?
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于教育评估任务，旨在通过多模态回答推断学生心智模型质量。研究提出MMGrader方法，但发现现有模型性能仍低于人类水平。**

- **链接: [https://arxiv.org/pdf/2603.00056](https://arxiv.org/pdf/2603.00056)**

> **作者:** Pritam Sil; Durgaprasad Karnam; Vinay Reddy Venumuddala; Pushpak Bhattacharyya
>
> **摘要:** STEM Mental models can play a critical role in assessing students' conceptual understanding of a topic. They not only offer insights into what students know but also into how effectively they can apply, relate to, and integrate concepts across various contexts. Thus, students' responses are critical markers of the quality of their understanding and not entities that should be merely graded. However, inferring these mental models from student answers is challenging as it requires deep reasoning skills. We propose MMGrader, an approach that infers the quality of students' mental models from their multimodal responses using concept graphs as an analytical framework. In our evaluation with 9 openly available models, we found that the best-performing models fall short of human-level performance. This is because they only achieved an accuracy of approximately 40%, a prediction error of 1.1 units, and a scoring distribution fairly aligned with human scoring patterns. With improved accuracy, these can be highly effective assistants to teachers in inferring the mental models of their entire classrooms, enabling them to do so efficiently and help improve their pedagogies more effectively by designing targeted help sessions and lectures that strengthen areas where students collectively demonstrate lower proficiency.
>
---
#### [new 153] Unified Vision-Language Modeling via Concept Space Alignment
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出V-SONAR，将视觉与语言嵌入空间对齐，解决多语言多模态任务。通过统一表示，提升视频检索与描述性能，并扩展至低资源语言。**

- **链接: [https://arxiv.org/pdf/2603.01096](https://arxiv.org/pdf/2603.01096)**

> **作者:** Yifu Qiu; Paul-Ambroise Duquenne; Holger Schwenk
>
> **备注:** ICLR 2026
>
> **摘要:** We introduce V-SONAR, a vision-language embedding space extended from the text-only embedding space SONAR (Omnilingual Embeddings Team et al., 2026), which supports 1500 text languages and 177 speech languages. To construct V-SONAR, we propose a post-hoc alignment pipeline that maps the representations of an existing vision encoder into the SONAR space. We thoroughly evaluate V-SONAR and show that its embeddings achieve competitive performance on text-to-video retrieval. Equipped with the OMNISONAR text decoder, V-SONAR further surpasses state-of-the-art vision-language models on video captioning tasks, including DREAM-1K (BLEU 23.9 vs. 19.6) and PE-VIDEO (BLEU 39.0 vs. 30.0). Leveraging V-SONAR, we first demonstrate that the Large Concept Model (LCM; LCM team et al. 2024) operating in SONAR and trained with English text only, can perform both single- and multi-visual concept understanding in a zero-shot manner. Finally, we introduce V-LCM, which extends the LCM with vision-language instruction tuning. V-LCM encodes vision and language inputs into an unified sequence of latent embeddings via V-SONAR and SONAR, and it is trained with the same latent diffusion objective for next-embedding prediction as in LCM's text-only pre-training. Experiments on a large-scale multilingual and -modal instruction-tuning data mixture highlight the potential of V-LCM: V-LCM matches state-of-the-art vision-language models on tasks covering image/video captioning and question answering, while significantly outperforming them across 61 rich- to low-resource languages out of all 62 tested languages.
>
---
#### [new 154] Stabilizing Policy Optimization via Logits Convexity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RL训练不稳定的问题。通过分析SFT与RL的稳定性差异，提出LCO框架，提升训练稳定性和效果。**

- **链接: [https://arxiv.org/pdf/2603.00963](https://arxiv.org/pdf/2603.00963)**

> **作者:** Hongzhan Chen; Tao Yang; Yuhua Zhu; Shiping Gao; Xiaojun Quan; Ting Yao
>
> **摘要:** While reinforcement learning (RL) has been central to the recent success of large language models (LLMs), RL optimization is notoriously unstable, especially when compared to supervised fine-tuning (SFT). In this work, we investigate the stability gap between SFT and RL from a gradient-based perspective, and show that the convexity of the SFT loss with respect to model logits plays a key role in enabling stable training. Our theoretical analysis demonstrates that this property induces favorable gradient directionality during optimization. In contrast, Proximal Policy Optimization (PPO), a widely adopted policy gradient algorithm utilizing a clipped surrogate objective, lacks this stabilizing property. Motivated by this observation, we propose Logits Convex Optimization (LCO), a simple yet effective policy optimization framework that aligns the learned policy with an optimal target derived from the original RL objective, thereby emulating the stabilizing effects of logits-level convexity. Extensive experiments across multiple model families show that our LCO framework consistently improves training stability and outperforms conventional RL methods on a broad range of benchmarks.
>
---
#### [new 155] A Gauge Theory of Superposition: Toward a Sheaf-Theoretic Atlas of Neural Representations
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中语义超位置的解释性问题，通过构建局部语义图谱和度量空间，分析干扰能量与不可解释性障碍。**

- **链接: [https://arxiv.org/pdf/2603.00824](https://arxiv.org/pdf/2603.00824)**

> **作者:** Hossein Javidnia
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** We develop a discrete gauge-theoretic framework for superposition in large language models (LLMs) that replaces the single-global-dictionary premise with a sheaf-theoretic atlas of local semantic charts. Contexts are clustered into a stratified context complex; each chart carries a local feature space and a local information-geometric metric (Fisher/Gauss--Newton) identifying predictively consequential feature interactions. This yields a Fisher-weighted interference energy and three measurable obstructions to global interpretability: (O1) local jamming (active load exceeds Fisher bandwidth), (O2) proxy shearing (mismatch between geometric transport and a fixed correspondence proxy), and (O3) nontrivial holonomy (path-dependent transport around loops). We prove and instantiate four results on a frozen open LLM (Llama~3.2~3B Instruct) using WikiText-103, a C4-derived English web-text subset, and \texttt{the-stack-smol}. (A) After constructive gauge fixing on a spanning tree, each chord residual equals the holonomy of its fundamental cycle, making holonomy computable and gauge-invariant. (B) Shearing lower-bounds a data-dependent transfer mismatch energy, turning $D_{\mathrm{shear}}$ into an unavoidable failure bound. (C) We obtain non-vacuous certified jamming/interference bounds with high coverage and zero violations across seeds/hyperparameters. (D) Bootstrap and sample-size experiments show stable estimation of $D_{\mathrm{shear}}$ and $D_{\mathrm{hol}}$, with improved concentration on well-conditioned subsystems.
>
---
#### [new 156] Tool Verification for Test-Time Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，针对TTRL中的模式崩溃问题，提出T³RL方法，通过测试时工具验证提升奖励估计可靠性，增强模型自进化稳定性。**

- **链接: [https://arxiv.org/pdf/2603.02203](https://arxiv.org/pdf/2603.02203)**

> **作者:** Ruotong Liao; Nikolai Röhrich; Xiaohan Wang; Yuhui Zhang; Yasaman Samadzadeh; Volker Tresp; Serena Yeung-Levy
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** Test-time reinforcement learning (TTRL) has emerged as a promising paradigm for self-evolving large reasoning models (LRMs), enabling online adaptation on unlabeled test inputs via self-induced rewards through majority voting. However, a spurious yet high-frequency unverified consensus can become a biased and reinforced reward signal, leading to incorrect mode collapse. We address this failure mode with T^3RL (Tool-Verification for Test-Time Reinforcement Learning), which introduces test-time tool verification into reward estimation. Concretely, a verifier uses an external tool as evidence (e.g., from code execution) to upweight verified rollouts in a verification-aware voting, producing more reliable pseudo-labels for training. Across various math difficulties (MATH-500, AMC, and AIME 2024) and diverse backbone types, T^3RL significantly improves over TTRL, with larger gains on harder problems. More broadly, T^3RL can be viewed as verified online data synthesis, highlighting test-time tool verification as a key mechanism for stabilizing self-evolution.
>
---
#### [new 157] Attention Smoothing Is All You Need For Unlearning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型遗忘任务，旨在解决大语言模型记忆敏感内容的问题。通过引入注意力平滑方法，有效删除记忆同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2603.01285](https://arxiv.org/pdf/2603.01285)**

> **作者:** Saleh Zare Zade; Xiangyu Zhou; Sijia Liu; Dongxiao Zhu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large Language Models are prone to memorizing sensitive, copyrighted, or hazardous content, posing significant privacy and legal concerns. Retraining from scratch is computationally infeasible, whereas current unlearning methods exhibit unstable trade-offs between forgetting and utility, frequently producing incoherent outputs on forget prompts and failing to generalize due to the persistence of lexical-level and semantic-level associations in attention. We propose Attention Smoothing Unlearning (ASU), a principled framework that casts unlearning as self-distillation from a forget-teacher derived from the model's own attention. By increasing the softmax temperature, ASU flattens attention distributions and directly suppresses the lexical-level and semantic-level associations responsible for reconstructing memorized knowledge. This results in a bounded optimization objective that erases factual information yet maintains coherence in responses to forget prompts. Empirical evaluation on TOFU, MUSE, and WMDP, along with real-world and continual unlearning scenarios across question answering and text completion, demonstrates that ASU outperforms the baselines for most unlearning scenarios, delivering robust unlearning with minimal loss of model utility.
>
---
#### [new 158] SWE-Adept: An LLM-Based Agentic Framework for Deep Codebase Analysis and Structured Issue Resolution
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文提出SWE-Adept框架，解决代码库级软件工程问题，通过双代理实现精准代码定位与系统化修复。**

- **链接: [https://arxiv.org/pdf/2603.01327](https://arxiv.org/pdf/2603.01327)**

> **作者:** Kang He; Kaushik Roy
>
> **摘要:** Large language models (LLMs) exhibit strong performance on self-contained programming tasks. However, they still struggle with repository-level software engineering (SWE), which demands (1) deep codebase navigation with effective context management for accurate localization, and (2) systematic approaches for iterative, test-driven code modification to resolve issues. To address these challenges, we propose SWE-Adept, an LLM-based two-agent framework where a localization agent identifies issue-relevant code locations and a resolution agent implements the corresponding fixes. For issue localization, we introduce agent-directed depth-first search that selectively traverses code dependencies. This minimizes issue-irrelevant content in the agent's context window and improves localization accuracy. For issue resolution, we employ adaptive planning and structured problem solving. We equip the agent with specialized tools for progress tracking and Git-based version control. These tools interface with a shared working memory that stores code-state checkpoints indexed by execution steps, facilitating precise checkpoint retrieval. This design enables reliable agent-driven version-control operations for systematic issue resolution, including branching to explore alternative solutions and reverting failed edits. Experiments on SWE-Bench Lite and SWE-Bench Pro demonstrate that SWE-Adept consistently outperforms prior approaches in both issue localization and resolution, improving the end-to-end resolve rate by up to 4.7%.
>
---
#### [new 159] End-to-End Simultaneous Dysarthric Speech Reconstruction with Frame-Level Adaptor and Multiple Wait-k Knowledge Distillation
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音重建任务，旨在解决慢速发音导致的延迟及语音识别误差问题。提出端到端系统，引入帧级适配器和多视角知识蒸馏，提升重建效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.01382](https://arxiv.org/pdf/2603.01382)**

> **作者:** Minghui Wu; Haitao Tang; Jiahuan Fan; Ruizhi Liao; Yanyong Zhang
>
> **备注:** Submitted to 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** Dysarthric speech reconstruction (DSR) typically employs a cascaded system that combines automatic speech recognition (ASR) and sentence-level text-to-speech (TTS) to convert dysarthric speech into normally-prosodied speech. However, dysarthric individuals often speak more slowly, leading to excessively long response times in such systems, rendering them impractical in long-speech scenarios. Cascaded DSR systems based on streaming ASR and incremental TTS can help reduce latency. However, patients with differing dysarthria severity exhibit substantial pronunciation variability for the same text, resulting in poor robustness of ASR and limiting the intelligibility of reconstructed speech. In addition, incremental TTS suffers from poor prosodic feature prediction due to a limited receptive field. In this study, we propose an end-to-end simultaneous DSR system with two key innovations: 1) A frame-level adaptor module is introduced to bridge ASR and TTS. By employing explicit-implicit semantic information fusion and joint module training, it enhances the error tolerance of TTS to ASR outputs. 2) A multiple wait-k autoregressive TTS module is designed to mitigate prosodic degradation via multi-view knowledge distillation. Our system has an average response time of 1.03 seconds on Tesla A100, with an average real-time factor (RTF) of 0.71. On the UASpeech dataset, it attains a mean opinion score (MOS) of 4.67 and demonstrates a 54.25% relative reduction in word error rate (WER) compared to the state-of-the-art. Our demo is available at: this https URL
>
---
#### [new 160] TopoCurate:Modeling Interaction Topology for Tool-Use Agent Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TopoCurate框架，用于工具使用代理训练，解决传统方法忽略交互动态的问题，通过结构化拓扑表示提升策略有效性。**

- **链接: [https://arxiv.org/pdf/2603.01714](https://arxiv.org/pdf/2603.01714)**

> **作者:** Jinluan Yang; Yuxin Liu; Zhengyu Chen; Chengcheng Han; Yueqing Sun; Qi Gu; Hui Su; Xunliang Cai; Fei Wu; Kun Kuang
>
> **备注:** Under Review
>
> **摘要:** Training tool-use agents typically relies on outcome-based filtering: Supervised Fine-Tuning (SFT) on successful trajectories and Reinforcement Learning (RL) on pass-rate-selected tasks. However, this paradigm ignores interaction dynamics: successful trajectories may lack error recovery or exhibit redundancy, while pass rates fail to distinguish structurally informative tasks from trivial ones. We propose \textbf{TopoCurate}, an interaction-aware framework that projects multi-trial rollouts from the same task into a unified semantic quotient topology. By merging equivalent action-observation states, this projection transforms scattered linear trajectories into a structured manifold that explicitly captures how tool invocations and environmental responses drive the divergence between effective strategies and failure modes. Leveraging this representation, we introduce a dual-selection mechanism: for SFT, we prioritize trajectories demonstrating reflective recovery, semantic efficiency, and strategic diversity to mitigate covariate shift and mode collapse; for RL, we select tasks with high error branch ratios and strategic heterogeneity, maximizing gradient Signal-to-Noise Ratio to address vanishing signals in sparse-reward settings. Evaluations on BFCLv3 and Tau2 Bench show that TopoCurate achieves consistent gains of 4.2\% (SFT) and 6.9\% (RL) over state-of-the-art baselines. We will release the code and data soon for further investigations.
>
---
#### [new 161] Optimizing In-Context Demonstrations for LLM-based Automated Grading
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动化评分任务，旨在解决LLM在开放性答题评分中的可靠性问题。通过优化示例选择和生成判别性理由，提升模型对评分标准的遵循度。**

- **链接: [https://arxiv.org/pdf/2603.00465](https://arxiv.org/pdf/2603.00465)**

> **作者:** Yucheng Chu; Hang Li; Kaiqi Yang; Yasemin Copur-Gencturk; Kevin Haudek; Joseph Krajcik; Jiliang Tang
>
> **摘要:** Automated assessment of open-ended student responses is a critical capability for scaling personalized feedback in education. While large language models (LLMs) have shown promise in grading tasks via in-context learning (ICL), their reliability is heavily dependent on the selection of few-shot exemplars and the construction of high-quality rationales. Standard retrieval methods typically select examples based on semantic similarity, which often fails to capture subtle decision boundaries required for rubric adherence. Furthermore, manually crafting the expert rationales needed to guide these models can be a significant bottleneck. To address these limitations, we introduce GUIDE (Grading Using Iteratively Designed Exemplars), a framework that reframes exemplar selection and refinement in automated grading as a boundary-focused optimization problem. GUIDE operates on a continuous loop of selection and refinement, employing novel contrastive operators to identify "boundary pairs" that are semantically similar but possess different grades. We enhance exemplars by generating discriminative rationales that explicitly articulate why a response receives a specific score to the exclusion of adjacent grades. Extensive experiments across datasets in physics, chemistry, and pedagogical content knowledge demonstrate that GUIDE significantly outperforms standard retrieval baselines. By focusing the model's attention on the precise edges of rubric, our approach shows exceptionally robust gains on borderline cases and improved rubric adherence. GUIDE paves the way for trusted, scalable assessment systems that align closely with human pedagogical standards.
>
---
#### [new 162] According to Me: Long-Term Personalized Referential Memory QA
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出ATM-Bench基准，解决多模态个性化记忆问答任务，旨在提升AI助手对用户长期记忆的准确理解和推理能力。**

- **链接: [https://arxiv.org/pdf/2603.01990](https://arxiv.org/pdf/2603.01990)**

> **作者:** Jingbiao Mei; Jinghong Chen; Guangyu Yang; Xinyu Hou; Margaret Li; Bill Byrne
>
> **备注:** Preprint
>
> **摘要:** Personalized AI assistants must recall and reason over long-term user memory, which naturally spans multiple modalities and sources such as images, videos, and emails. However, existing Long-term Memory benchmarks focus primarily on dialogue history, failing to capture realistic personalized references grounded in lived experience. We introduce ATM-Bench, the first benchmark for multimodal, multi-source personalized referential Memory QA. ATM-Bench contains approximately four years of privacy-preserving personal memory data and human-annotated question-answer pairs with ground-truth memory evidence, including queries that require resolving personal references, multi-evidence reasoning from multi-source and handling conflicting evidence. We propose Schema-Guided Memory (SGM) to structurally represent memory items originated from different sources. In experiments, we implement 5 state-of-the-art memory systems along with a standard RAG baseline and evaluate variants with different memory ingestion, retrieval, and answer generation techniques. We find poor performance (under 20\% accuracy) on the ATM-Bench-Hard set, and that SGM improves performance over Descriptive Memory commonly adopted in prior works. Code available at: this https URL
>
---
#### [new 163] Confusion-Aware Rubric Optimization for LLM-based Automated Grading
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动化评分任务，解决LLM评分指南不准确问题。通过CARO框架分离错误信号，提升评分精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.00451](https://arxiv.org/pdf/2603.00451)**

> **作者:** Yucheng Chu; Hang Li; Kaiqi Yang; Yasemin Copur-Gencturk; Joseph Krajcik; Namsoo Shin; Jiliang Tang
>
> **摘要:** Accurate and unambiguous guidelines are critical for large language model (LLM) based graders, yet manually crafting these prompts is often sub-optimal as LLMs can misinterpret expert guidelines or lack necessary domain specificity. Consequently, the field has moved toward automated prompt optimization to refine grading guidelines without the burden of manual trial and error. However, existing frameworks typically aggregate independent and unstructured error samples into a single update step, resulting in "rule dilution" where conflicting constraints weaken the model's grading logic. To address these limitations, we introduce Confusion-Aware Rubric Optimization (CARO), a novel framework that enhances accuracy and computational efficiency by structurally separating error signals. CARO leverages the confusion matrix to decompose monolithic error signals into distinct modes, allowing for the diagnosis and repair of specific misclassification patterns individually. By synthesizing targeted "fixing patches" for dominant error modes and employing a diversity-aware selection mechanism, the framework prevents guidance conflict and eliminates the need for resource-heavy nested refinement loops. Empirical evaluations on teacher education and STEM datasets demonstrate that CARO significantly outperforms existing SOTA methods. These results suggest that replacing mixed-error aggregation with surgical, mode-specific repair yields robust improvements in automated assessment scalability and precision.
>
---
#### [new 164] I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究安全分类器在嵌入漂移下的鲁棒性问题，发现微小扰动会导致性能骤降，挑战现有安全机制的有效性。**

- **链接: [https://arxiv.org/pdf/2603.01297](https://arxiv.org/pdf/2603.01297)**

> **作者:** Subramanyam Sahoo; Vinija Jain; Divya Chaudhary; Aman Chadha
>
> **备注:** Accepted at the ICBINB: Where LLMs Need to Improve workshop at ICLR 2026. 12 pages and 3 Figures
>
> **摘要:** Instruction tuned reasoning models are increasingly deployed with safety classifiers trained on frozen embeddings, assuming representation stability across model updates. We systematically investigate this assumption and find it fails: normalized perturbations of magnitude $\sigma=0.02$ (corresponding to $\approx 1^\circ$ angular drift on the embedding sphere) reduce classifier performance from $85\%$ to $50\%$ ROC-AUC. Critically, mean confidence only drops $14\%$, producing dangerous silent failures where $72\%$ of misclassifications occur with high confidence, defeating standard monitoring. We further show that instruction-tuned models exhibit 20$\%$ worse class separability than base models, making aligned systems paradoxically harder to safeguard. Our findings expose a fundamental fragility in production AI safety architectures and challenge the assumption that safety mechanisms transfer across model versions.
>
---
#### [new 165] LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在解决模型忽视语言指令的问题。通过构建LangGap基准，进行语义扰动实验，验证数据增强对提升语言理解的有效性。**

- **链接: [https://arxiv.org/pdf/2603.00592](https://arxiv.org/pdf/2603.00592)**

> **作者:** Yuchen Hou; Lin Zhao
>
> **备注:** 7 pages, 3 figures. Code and benchmark will be available at this https URL
>
> **摘要:** Vision-Language-Action (VLA) models achieve over 95% success on standard benchmarks. However, through systematic experiments, we find that current state-of-the-art VLA models largely ignore language instructions. Prior work lacks: (1) systematic semantic perturbation diagnostics, (2) a benchmark that forces language understanding by design, and (3) linguistically diverse training data. This paper constructs the LangGap benchmark, based on a four-dimensional semantic perturbation method -- varying instruction semantics while keeping the tabletop layout fixed -- revealing language understanding deficits in {\pi}0.5. Existing benchmarks like LIBERO assign only one task per layout, underutilizing available objects and target locations; LangGap fully diversifies pick-and-place tasks under identical layouts, forcing models to truly understand language. Experiments show that targeted data augmentation can partially close the language gap -- success rate improves from 0% to 90% with single-task training, and 0% to 28% with multi-task training. However, as semantic diversity of extended tasks increases, model learning capacity proves severely insufficient; even trained tasks perform poorly. This reveals a fundamental challenge for VLA models in understanding diverse language instructions -- precisely the long-term value of LangGap.
>
---
#### [new 166] Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理任务，解决长链式思维中推理成本过高的问题。提出Draft-Thinking方法，通过精简推理结构和自适应提示，降低推理预算同时保持性能。**

- **链接: [https://arxiv.org/pdf/2603.00578](https://arxiv.org/pdf/2603.00578)**

> **作者:** Jie Cao; Tianwei Lin; Zhenxuan Fan; Bo Yuan; Ziyuan Zhao; Rolan Yan; Wenqiao Zhang; Siliang Tang
>
> **摘要:** Long chain-of-thought~(CoT) has become a dominant paradigm for enhancing the reasoning capability of large reasoning models~(LRMs); however, the performance gains often come with a substantial increase in reasoning budget. Recent studies show that existing CoT paradigms tend to induce systematic overthinking, unnecessarily coupling reasoning capability with reasoning cost. Most prior approaches reduce token usage through post hoc techniques such as token compression, truncation, or length penalties, without explicitly addressing the core mechanisms of reasoning. We propose \textbf{Draft-Thinking}, which guides models to first learn a concise \textit{draft-style} reasoning structure that retains only the critical reasoning steps. Through a \textit{progressive curriculum learning}, the model stably internalizes this efficient reasoning pattern as its capability scales. Moreover, Draft-Thinking introduces adaptive prompting, which elevates reasoning depth to a flexible, model-selectable behavior. Extensive experiments demonstrate that Draft-Thinking substantially reduces reasoning budget while largely preserving reasoning performance; for example, on MATH500, it achieves an 82.6\% reduction in reasoning budget at the cost of only a 2.6\% performance drop.
>
---
#### [new 167] Constructing Synthetic Instruction Datasets for Improving Reasoning in Domain-Specific LLMs: A Case Study in the Japanese Financial Domain
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于领域专用大语言模型的推理能力提升任务，旨在解决领域知识与推理能力结合的难题。通过构建高质量合成指令数据集，提升金融领域模型表现。**

- **链接: [https://arxiv.org/pdf/2603.01353](https://arxiv.org/pdf/2603.01353)**

> **作者:** Yuma Okochi; Fabio Milentiansen Sim; Tomoyasu Okada
>
> **备注:** 8 pages, 2 figures. Japanese version published in NLP2026
>
> **摘要:** In adapting LLMs to specific domains, achieving both domain expertise and reasoning ability remains an urgent challenge. This study proposes a general method for constructing high-quality synthetic instruction data for any domain, starting from domain-specific vocabulary. As a demonstration, we applied this method to the financial domain and constructed a large-scale instruction dataset totaling approximately 9.5 billion tokens with Chain-of-Thought reasoning traces. Evaluation results confirmed performance improvements over baseline models on financial benchmarks, demonstrating the effectiveness of our approach. We also report findings on the impact of reasoning trace length on performance and its limitations. Lastly, we open-source our models and datasets on this https URL .
>
---
## 更新

#### [replaced 001] LAD-RAG: Layout-aware Dynamic RAG for Visually-Rich Document Understanding
- **分类: cs.CL**

- **简介: 该论文属于视觉丰富文档问答任务，旨在解决传统RAG方法在处理多页文档时丢失结构信息和跨页依赖的问题。提出LAD-RAG框架，通过构建符号化文档图提升检索与问答效果。**

- **链接: [https://arxiv.org/pdf/2510.07233](https://arxiv.org/pdf/2510.07233)**

> **作者:** Zhivar Sourati; Zheng Wang; Marianne Menglin Liu; Yazhe Hu; Mengqing Guo; Sujeeth Bharadwaj; Kyu Han; Tao Sheng; Sujith Ravi; Morteza Dehghani; Dan Roth
>
> **摘要:** Question answering over visually rich documents (VRDs) requires reasoning not only over isolated content but also over documents' structural organization and cross-page dependencies. However, conventional retrieval-augmented generation (RAG) methods encode content in isolated chunks during ingestion, losing structural and cross-page dependencies, and retrieve a fixed number of pages at inference, regardless of the specific demands of the question or context. This often results in incomplete evidence retrieval and degraded answer quality for multi-page reasoning tasks. To address these limitations, we propose LAD-RAG, a novel Layout-Aware Dynamic RAG framework. During ingestion, LAD-RAG constructs a symbolic document graph that captures layout structure and cross-page dependencies, adding it alongside standard neural embeddings to yield a more holistic representation of the document. During inference, an LLM agent dynamically interacts with the neural and symbolic indices to adaptively retrieve the necessary evidence based on the query. Experiments on MMLongBench-Doc, LongDocURL, DUDE, and MP-DocVQA demonstrate that LAD-RAG improves retrieval, achieving over 90% perfect recall on average without any top-k tuning, and outperforming baseline retrievers by up to 20% in recall at comparable noise levels, yielding higher QA accuracy with minimal latency.
>
---
#### [replaced 002] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视觉问答任务，解决小样本学习中模型性能不升反降的问题。通过元学习和提示蒸馏方法，提升模型在少量数据下的适应能力。**

- **链接: [https://arxiv.org/pdf/2506.06905](https://arxiv.org/pdf/2506.06905)**

> **作者:** Akash Gupta; Amos Storkey; Mirella Lapata
>
> **备注:** ICLR 2026
>
> **摘要:** Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new visual question answering (VQA) tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, does not always improve monotonically when increasing the number of examples. We hypothesize that this happens because the LMM is overwhelmed by extraneous information in the image embeddings that is irrelevant to the downstream task. To address this, we propose a meta-learning approach that induces few-shot capabilities in LMMs through a fixed set of soft prompts distilled from task-relevant visual features, which are adapted at test time using a small number of examples. We facilitate this distillation through an attention-mapper module that can be easily integrated with any LMM architecture and is jointly learned with soft prompts. Evaluation on the VL-ICL Bench shows that our method successfully achieves task adaptation in low-data regimes with just a few gradient steps, outperforming ICL by 21.2%. Comparisons with parameter-efficient finetuning methods demonstrate that meta-learning further enhances this adaptation by 7.7% for various VQA tasks.
>
---
#### [replaced 003] A cross-species neural foundation model for end-to-end speech decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于脑机接口任务，旨在解决神经信号到文本的端到端翻译问题。提出BIT框架，使用跨任务、跨物种预训练模型，提升解码性能。**

- **链接: [https://arxiv.org/pdf/2511.21740](https://arxiv.org/pdf/2511.21740)**

> **作者:** Yizi Zhang; Linyang He; Chaofei Fan; Tingkai Liu; Han Yu; Trung Le; Jingyuan Li; Scott Linderman; Lea Duncker; Francis R Willett; Nima Mesgarani; Liam Paninski
>
> **摘要:** Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by translating neural activity into text. Most systems use cascaded frameworks that decode phonemes before assembling sentences with an n-gram language model (LM), preventing joint optimization of all stages simultaneously. Here, we introduce an end-to-end Brain-to-Text (BIT) framework that translates neural activity into coherent sentences using a single differentiable neural network. Central to our approach is a cross-task, cross-species pretrained neural encoder, whose representations transfer to both attempted and imagined speech. In a cascaded setting with an n-gram LM, the pretrained encoder establishes a new state-of-the-art (SOTA) on the Brain-to-Text '24 and '25 benchmarks. Integrated end-to-end with audio large language models (LLMs) and trained with contrastive learning for cross-modal alignment, BIT reduces the word error rate (WER) of the prior end-to-end method from 24.69% to 10.22%. Notably, we find that small-scale audio LLMs markedly improve end-to-end decoding. Beyond record-setting performance, BIT aligns attempted and imagined speech embeddings to enable cross-task generalization. Altogether, our approach advances the integration of large, diverse neural datasets, paving the way for an end-to-end decoding framework that supports seamless, differentiable optimization.
>
---
#### [replaced 004] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决KV缓存效率与准确率问题。提出FreeKV框架，通过算法与系统协同优化提升KV检索效率，实现近无损精度和显著加速。**

- **链接: [https://arxiv.org/pdf/2505.13109](https://arxiv.org/pdf/2505.13109)**

> **作者:** Guangda Liu; Chengwei Li; Zhenyu Ning; Jing Lin; Yiwu Yao; Danning Ke; Minyi Guo; Jieru Zhao
>
> **摘要:** Large language models (LLMs) are widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods have been proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, a training-free algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency, enabling effective overlap with computation, full latency hiding, and practical speedups from speculative recall. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to a 13$\times$ speedup compared to SOTA KV retrieval methods. Code is available at this https URL.
>
---
#### [replaced 005] On the Reasoning Abilities of Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Masked Diffusion Language Models（MDMs）的推理能力，探讨其在特定任务中的表现与效率。属于自然语言处理中的模型分析任务，旨在解决MDMs的计算能力和效率问题。工作包括理论分析和对比实验，证明MDMs在某些问题上优于传统模型。**

- **链接: [https://arxiv.org/pdf/2510.13117](https://arxiv.org/pdf/2510.13117)**

> **作者:** Anej Svete; Ashish Sabharwal
>
> **摘要:** Masked diffusion models (MDMs) for text offer a compelling alternative to traditional autoregressive language models. Parallel generation makes them efficient, but their computational capabilities and the limitations inherent in their parallelism remain largely unexplored. To this end, we characterize what types of reasoning problems MDMs can provably solve and how efficiently. We do this by connecting MDMs to the well-understood reasoning frameworks of chain of thought (CoT) and padded looped transformers (PLTs) in the finite-precision log-width setting: We show that MDMs and polynomially-padded PLTs are, in fact, equivalent in this setting, and that MDMs can solve all problems that CoT-augmented transformers can. Moreover, we showcase classes of problems (including regular languages) for which MDMs are inherently more efficient than CoT transformers, where parallel generation allows for substantially faster reasoning.
>
---
#### [replaced 006] Are LLMs Ready to Replace Bangla Annotators?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在孟加拉语仇恨言论标注中的应用，探讨其可靠性和偏差问题。任务属于低资源语言的敏感标注，旨在评估LLMs作为自动标注工具的适用性。**

- **链接: [https://arxiv.org/pdf/2602.16241](https://arxiv.org/pdf/2602.16241)**

> **作者:** Md. Najib Hasan; Touseef Hasan; Souvika Sarkar
>
> **备注:** We have identified significant methodological discrepancies in the current version of the manuscript that affect the validity and reproducibility of the reported results. In order to prevent potential misunderstanding or misinterpretation of our findings, we request the complete withdrawal of this submission while we conduct a thorough revision
>
> **摘要:** Large Language Models (LLMs) are increasingly used as automated annotators to scale dataset creation, yet their reliability as unbiased annotators--especially for low-resource and identity-sensitive settings--remains poorly understood. In this work, we study the behavior of LLMs as zero-shot annotators for Bangla hate speech, a task where even human agreement is challenging, and annotator bias can have serious downstream consequences. We conduct a systematic benchmark of 17 LLMs using a unified evaluation framework. Our analysis uncovers annotator bias and substantial instability in model judgments. Surprisingly, increased model scale does not guarantee improved annotation quality--smaller, more task-aligned models frequently exhibit more consistent behavior than their larger counterparts. These results highlight important limitations of current LLMs for sensitive annotation tasks in low-resource languages and underscore the need for careful evaluation before deployment.
>
---
#### [replaced 007] BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于生物多模态基础模型任务，旨在解决生物图像与文本对齐问题。通过生成合成描述句，增强模型对物种特征的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2510.20095](https://arxiv.org/pdf/2510.20095)**

> **作者:** Ziheng Zhang; Xinyue Ma; Arpita Chowdhury; Elizabeth G. Campolongo; Matthew J. Thompson; Net Zhang; Samuel Stevens; Hilmar Lapp; Tanya Berger-Wolf; Yu Su; Wei-Lun Chao; Jianyang Gu
>
> **备注:** ICLR 2026; Project page: this https URL
>
> **摘要:** This work investigates descriptive captions as an additional source of supervision for biological multimodal foundation models. Images and captions can be viewed as complementary samples from the latent morphospace of a species, each capturing certain biological traits. Incorporating captions during training encourages alignment with this shared latent structure, emphasizing potentially diagnostic characters while suppressing spurious correlations. The main challenge, however, lies in obtaining faithful, instance-specific captions at scale. This requirement has limited the utilization of natural language supervision in organismal biology compared with many other scientific domains. We complement this gap by generating synthetic captions with multimodal large language models (MLLMs), guided by Wikipedia-derived visual information and taxon-tailored format examples. These domain-specific contexts help reduce hallucination and yield accurate, instance-based descriptive captions. Using these captions, we train BioCAP (i.e., BioCLIP with Captions), a biological foundation model that captures rich semantics and achieves strong performance in species classification and text-image retrieval. These results demonstrate the value of descriptive captions beyond labels in bridging biological images with multimodal foundation models.
>
---
#### [replaced 008] Universal Robust Speech Adaptation for Cross-Domain Speech Recognition and Enhancement
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别与增强任务，旨在解决域迁移导致的性能下降问题。提出URSA-GAN框架，通过双编码器和动态扰动提升模型在噪声和信道失配下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04307](https://arxiv.org/pdf/2602.04307)**

> **作者:** Chien-Chun Wang; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to IEEE Transactions on Audio, Speech and Language Processing (IEEE TASLP)
>
> **摘要:** Pre-trained models for automatic speech recognition (ASR) and speech enhancement (SE) have exhibited remarkable capabilities under matched noise and channel conditions. However, these models often suffer from severe performance degradation when confronted with domain shifts, particularly in the presence of unseen noise and channel distortions. In view of this, we in this paper present URSA-GAN, a unified and domain-aware generative framework specifically designed to mitigate mismatches in both noise and channel conditions. URSA-GAN leverages a dual-embedding architecture that consists of a noise encoder and a channel encoder, each pre-trained with limited in-domain data to capture domain-relevant representations. These embeddings condition a GAN-based speech generator, facilitating the synthesis of speech that is acoustically aligned with the target domain while preserving phonetic content. To enhance generalization further, we propose dynamic stochastic perturbation, a novel regularization technique that introduces controlled variability into the embeddings during generation, promoting robustness to unseen domains. Empirical results demonstrate that URSA-GAN effectively reduces character error rates in ASR and improves perceptual metrics in SE across diverse noisy and mismatched channel scenarios. Notably, evaluations on compound test conditions with both channel and noise degradations confirm the generalization ability of URSA-GAN, yielding relative improvements of 16.16% in ASR performance and 15.58% in SE metrics.
>
---
#### [replaced 009] Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference
- **分类: cs.CL**

- **简介: 该论文研究LLM在图推理中的应用，解决其与图数据交互的性能问题。通过大规模实验，比较不同交互方式，分析模型在不同图结构和特征下的表现。**

- **链接: [https://arxiv.org/pdf/2509.18487](https://arxiv.org/pdf/2509.18487)**

> **作者:** Ben Finkelshtein; Silviu Cucerzan; Sujay Kumar Jauhar; Ryen White
>
> **摘要:** Large language models (LLMs) are increasingly used for text-rich graph machine learning tasks such as node classification in high-impact domains like fraud detection and recommendation systems. Yet, despite a surge of interest, the field lacks a principled understanding of the capabilities of LLMs in their interaction with graph data. In this work, we conduct a large-scale, controlled evaluation across several key axes of variability to systematically assess the strengths and weaknesses of LLM-based graph reasoning methods in text-based applications. The axes include the LLM-graph interaction mode, comparing prompting, tool-use, and code generation; dataset domains, spanning citation, web-link, e-commerce, and social networks; structural regimes contrasting homophilic and heterophilic graphs; feature characteristics involving both short- and long-text node attributes; and model configurations with varying LLM sizes and reasoning capabilities. We further analyze dependencies by methodically truncating features, deleting edges, and removing labels to quantify reliance on input types. Our findings provide practical and actionable guidance. (1) LLMs as code generators achieve the strongest overall performance on graph data, with especially large gains on long-text or high-degree graphs where prompting quickly exceeds the token budget. (2) All interaction strategies remain effective on heterophilic graphs, challenging the assumption that LLM-based methods collapse under low homophily. (3) Code generation is able to flexibly adapt its reliance between structure, features, or labels to leverage the most informative input type. Together, these findings provide a comprehensive view of the strengths and limitations of current LLM-graph interaction modes and highlight key design principles for future approaches.
>
---
#### [replaced 010] When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究长文本处理任务，解决LLM在长上下文中的性能问题。提出噪声分解框架，分析多代理分块的有效性，验证其在不同任务中的优势。**

- **链接: [https://arxiv.org/pdf/2506.16411](https://arxiv.org/pdf/2506.16411)**

> **作者:** Zhen Xu; Shang Zhu; Jue Wang; Junlin Wang; Ben Athiwaratkun; Chi Wang; James Zou; Ce Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** We investigate the challenge of applying Large Language Models (LLMs) to long texts. We propose a theoretical framework that distinguishes the failure modes of long context tasks into three categories: cross-chunk dependence (task noise), confusion that grows with context size (model noise), and the imperfect integration of partial results (aggregator noise). Under this view, we analyze when it is effective to use multi-agent chunking, i.e., dividing a lengthy sequence into smaller chunks and aggregating the processed results of each chunk. Our experiments on tasks such as retrieval, question answering, and summarization confirm both the theoretical analysis and the conditions that favor multi-agent chunking. By exploring the accelerated decay of model fidelity with input length, we also explain why, for large inputs, a weaker model configured with chunk-based processing can surpass a more advanced model like GPT4o applied in a single shot. Overall, we present a principled understanding framework and our results highlight a direct pathway to handling long contexts in LLMs with carefully managed chunking and aggregator strategies.
>
---
#### [replaced 011] Reason Like a Radiologist: Chain-of-Thought and Reinforcement Learning for Verifiable Report Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医学影像报告生成任务，旨在解决模型缺乏结构化推理和可验证性的问题。通过引入BoxMed-RL框架，结合思维链和强化学习，提升报告的准确性和临床可信度。**

- **链接: [https://arxiv.org/pdf/2504.18453](https://arxiv.org/pdf/2504.18453)**

> **作者:** Peiyuan Jing; Kinhei Lee; Zhenxuan Zhang; Huichi Zhou; Zhengqing Yuan; Zhifan Gao; Lei Zhu; Giorgos Papanastasiou; Yingying Fang; Guang Yang
>
> **摘要:** Radiology report generation is critical for efficiency but current models lack the structured reasoning of experts, hindering clinical trust and explainability by failing to link visual findings to precise anatomical locations. This paper introduces BoxMed-RL, a groundbreaking unified training framework for generating spatially verifiable and explainable radiology reports. Built on a large vision-language model, BoxMed-RL revolutionizes report generation through two integrated phases: (1) In the Pretraining Phase, we refine the model via medical concept learning, using Chain-of-Thought supervision to internalize the radiologist-like workflow, followed by spatially verifiable reinforcement, which applies reinforcement learning to align medical findings with bounding boxes. (2) In the Downstream Adapter Phase, we freeze the pretrained weights and train a downstream adapter to ensure fluent and clinically credible reports. This framework precisely mimics radiologists' workflow, compelling the model to connect high-level medical concepts with definitive anatomical evidence. Extensive experiments on public datasets demonstrate that BoxMed-RL achieves an average 7% improvement in both METEOR and ROUGE-L metrics compared to state-of-the-art methods. An average 5% improvement in large language model-based metrics further underscores BoxMed-RL's robustness in generating high-quality radiology reports.
>
---
#### [replaced 012] VINCIE: Unlocking In-context Image Editing from Video
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出VINCIE模型，解决视频驱动的上下文图像编辑问题。通过视频数据训练，实现多轮图像编辑与故事生成，提升编辑连贯性与多样性。**

- **链接: [https://arxiv.org/pdf/2506.10941](https://arxiv.org/pdf/2506.10941)**

> **作者:** Leigang Qu; Feng Cheng; Ziyan Yang; Qi Zhao; Shanchuan Lin; Yichun Shi; Yicong Li; Wenjie Wang; Tat-Seng Chua; Lu Jiang
>
> **备注:** ICLR 2026 Camera-ready. Project page: this https URL
>
> **摘要:** In-context image editing aims to modify images based on a contextual sequence comprising text and previously generated images. Existing methods typically depend on task-specific pipelines and expert models (e.g., segmentation and inpainting) to curate training data. In this work, we explore whether an in-context image editing model can be learned directly from videos. We introduce a scalable approach to annotate videos as interleaved multimodal sequences. To effectively learn from this data, we design a block-causal diffusion transformer trained on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. Additionally, we propose a novel multi-turn image editing benchmark to advance research in this area. Extensive experiments demonstrate that our model exhibits strong in-context image editing capabilities and achieves state-of-the-art results on two multi-turn image editing benchmarks. Despite being trained exclusively on videos, our model also shows promising abilities in multi-concept composition, story generation, and chain-of-editing applications.
>
---
#### [replaced 013] LightMem: Lightweight and Efficient Memory-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出LightMem，解决LLM在动态环境中高效利用历史信息的问题。通过三阶段记忆系统，提升问答准确率，降低计算成本。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2510.18866](https://arxiv.org/pdf/2510.18866)**

> **作者:** Jizhan Fang; Xinle Deng; Haoming Xu; Ziyan Jiang; Yuqi Tang; Ziwen Xu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Huajun Chen; Ningyu Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. On LongMemEval and LoCoMo, using GPT and Qwen backbones, LightMem consistently surpasses strong baselines, improving QA accuracy by up to 7.7% / 29.3%, reducing total token usage by up to 38x / 20.9x and API calls by up to 30x / 55.5x, while purely online test-time costs are even lower, achieving up to 106x / 117x token reduction and 159x / 310x fewer API calls. The code is available at this https URL.
>
---
#### [replaced 014] Vision-DeepResearch Benchmark: Rethinking Visual and Textual Search for Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决视觉与文本搜索评估难题。针对现有基准的不足，构建了VDR-Bench，并提出多轮裁剪搜索策略以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.02185](https://arxiv.org/pdf/2602.02185)**

> **作者:** Yu Zeng; Wenxuan Huang; Zhen Fang; Shuang Chen; Yufan Shen; Yishuo Cai; Xiaoman Wang; Zhenfei Yin; Lin Chen; Zehui Chen; Shiting Huang; Yiming Zhao; Xu Tang; Yao Hu; Philip Torr; Wanli Ouyang; Shaosheng Cao
>
> **摘要:** Multimodal Large Language Models (MLLMs) have advanced VQA and now support Vision-DeepResearch systems that use search engines for complex visual-textual fact-finding. However, evaluating these visual and textual search abilities is still difficult, and existing benchmarks have two major limitations. First, existing benchmarks are not visual search-centric: answers that should require visual search are often leaked through cross-textual cues in the text questions or can be inferred from the prior world knowledge in current MLLMs. Second, overly idealized evaluation scenario: On the image-search side, the required information can often be obtained via near-exact matching against the full image, while the text-search side is overly direct and insufficiently challenging. To address these issues, we construct the Vision-DeepResearch benchmark (VDR-Bench) comprising 2,000 VQA instances. All questions are created via a careful, multi-stage curation pipeline and rigorous expert review, designed to assess the behavior of Vision-DeepResearch systems under realistic real-world conditions. Moreover, to address the insufficient visual retrieval capabilities of current MLLMs, we propose a simple multi-round cropped-search workflow. This strategy is shown to effectively improve model performance in realistic visual retrieval scenarios. Overall, our results provide practical guidance for the design of future multimodal deep-research systems. The code will be released in this https URL.
>
---
#### [replaced 015] A Comprehensive Dataset for Human vs. AI Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决AI生成内容真实性问题。通过构建包含58,000个样本的综合数据集，用于区分人类与AI文本及模型归属。**

- **链接: [https://arxiv.org/pdf/2510.22874](https://arxiv.org/pdf/2510.22874)**

> **作者:** Rajarshi Roy; Nasrin Imanpour; Ashhar Aziz; Shashwat Bajpai; Gurpreet Singh; Shwetangshu Biswas; Kapil Wanaskar; Parth Patwa; Subhankar Ghosh; Shreyas Dixit; Nilesh Ranjan Pal; Vipula Rawte; Ritvik Garimella; Gaytri Jena; Amit Sheth; Vasu Sharma; Aishwarya Naresh Reganti; Vinija Jain; Aman Chadha; Amitava Das
>
> **备注:** Defactify4 @AAAI 2025
>
> **摘要:** The rapid advancement of large language models (LLMs) has led to increasingly human-like AI-generated text, raising concerns about content authenticity, misinformation, and trustworthiness. Addressing the challenge of reliably detecting AI-generated text and attributing it to specific models requires large-scale, diverse, and well-annotated datasets. In this work, we present a comprehensive dataset comprising over 58,000 text samples that combine authentic New York Times articles with synthetic versions generated by multiple state-of-the-art LLMs including Gemma-2-9b, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, and GPT-4-o. The dataset provides original article abstracts as prompts, full human-authored narratives. We establish baseline results for two key tasks: distinguishing human-written from AI-generated text, achieving an accuracy of 58.35\%, and attributing AI texts to their generating models with an accuracy of 8.92\%. By bridging real-world journalistic content with modern generative models, the dataset aims to catalyze the development of robust detection and attribution methods, fostering trust and transparency in the era of generative AI. Our dataset is available at: this https URL.
>
---
#### [replaced 016] Adaptive Data Augmentation with Multi-armed Bandit: Sample-Efficient Embedding Calibration for Implicit Pattern Recognition
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对隐式模式识别任务，解决少样本下模型精度低的问题。提出ADAMAB框架，结合多臂老虎机实现自适应数据增强，提升嵌入校准效率。**

- **链接: [https://arxiv.org/pdf/2602.19385](https://arxiv.org/pdf/2602.19385)**

> **作者:** Minxue Tang; Yangyang Yu; Aolin Ding; Maziyar Baran Pouyan; Taha Belkhouja; Yujia Bao
>
> **摘要:** Recognizing implicit visual and textual patterns is essential in many real-world applications of modern AI. However, tackling long-tail pattern recognition tasks remains challenging for current pre-trained foundation models such as LLMs and VLMs. While finetuning pre-trained models can improve accuracy in recognizing implicit patterns, it is usually infeasible due to a lack of training data and high computational overhead. In this paper, we propose ADAMAB, an efficient embedding calibration framework for few-shot pattern recognition. To maximally reduce the computational costs, ADAMAB trains embedder-agnostic light-weight calibrators on top of fixed embedding models without accessing their parameters. To mitigate the need for large-scale training data, we introduce an adaptive data augmentation strategy based on the Multi-Armed Bandit (MAB) mechanism. With a modified upper confidence bound algorithm, ADAMAB diminishes the gradient shifting and offers theoretically guaranteed convergence in few-shot training. Our multi-modal experiments justify the superior performance of ADAMAB, with up to 40% accuracy improvement when training with less than 5 initial data samples of each class.
>
---
#### [replaced 017] Investigating Disability Representations in Text-to-Image Models
- **分类: cs.CL; cs.CV; cs.CY; cs.HC**

- **简介: 该论文属于AI伦理研究任务，旨在解决文本生成图像模型中残疾人群体表征不均衡的问题。通过分析模型输出，评估不同提示下的图像相似性及情感框架。**

- **链接: [https://arxiv.org/pdf/2602.04687](https://arxiv.org/pdf/2602.04687)**

> **作者:** Yang Tian; Yu Fan; Liudmila Zavolokina; Sarah Ebling
>
> **备注:** 21 pages, 9 figures. References included
>
> **摘要:** Text-to-image generative models have made remarkable progress in producing high-quality visual content from textual descriptions, yet concerns remain about how they represent social groups. While characteristics like gender and race have received increasing attention, disability representations remain underexplored. This study investigates how people with disabilities are represented in AI-generated images by analyzing outputs from Stable Diffusion XL and DALL-E 3 using a structured prompt design. We analyze disability representations by comparing image similarities between generic disability prompts and prompts referring to specific disability categories. Moreover, we evaluate how mitigation strategies influence disability portrayals, with a focus on assessing affective framing through sentiment polarity analysis, combining both automatic and human evaluation. Our findings reveal persistent representational imbalances and highlight the need for continuous evaluation and refinement of generative models to foster more diverse and inclusive portrayals of disability.
>
---
#### [replaced 018] Train Once, Answer All: Many Pretraining Experiments for the Cost of One
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在降低预训练模型的计算成本。通过一次训练完成多项实验，验证了方法的有效性，并提出了CPDT技术检测实验干扰。**

- **链接: [https://arxiv.org/pdf/2509.23383](https://arxiv.org/pdf/2509.23383)**

> **作者:** Sebastian Bordt; Martin Pawelczyk
>
> **备注:** ICLR 2026
>
> **摘要:** Recent work has demonstrated that controlled pretraining experiments are a powerful tool for studying the relationship between training data and large language model (LLM) behavior. However, the computational cost of pretraining presents a significant constraint. To overcome this constraint, we propose a new approach where multiple experiments are conducted simultaneously during a single training run. We validate our approach by performing ten experiments while training on 210B tokens, with models of up to 2.7B parameters. Although models are trained only once, we can replicate the results of multiple previous works on data contamination, poisoning, and memorization. We also conduct novel investigations into knowledge acquisition, mathematical reasoning, and watermarking. For example, we dynamically update the training data until a model acquires a particular piece of knowledge. Remarkably, the influence of the experiments on the model's training dynamics and overall performance is minimal. However, interactions between experiments may act as a confounder in our approach. We propose continual pretraining dependence testing (CPDT), a novel technique to test for interactions with continual pretraining experiments, finding them to be negligible in our setup. Overall, our results suggest that performing multiple pretraining experiments within a single training run can enable rigorous scientific experimentation with large models on a compute budget.
>
---
#### [replaced 019] Gender Bias in Emotion Recognition by Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于情感识别任务，研究LLMs在情感理论中的性别偏见问题，并探索减少偏见的策略。**

- **链接: [https://arxiv.org/pdf/2511.19785](https://arxiv.org/pdf/2511.19785)**

> **作者:** Maureen Herbert; Katie Sun; Angelica Lim; Yasaman Etesam
>
> **备注:** Accepted at AAAI 2026 Workshop (WS37)
>
> **摘要:** The rapid advancement of large language models (LLMs) and their growing integration into daily life underscore the importance of evaluating and ensuring their fairness. In this work, we examine fairness within the domain of emotional theory of mind, investigating whether LLMs exhibit gender biases when presented with a description of a person and their environment and asked, ''How does this person feel?''. Furthermore, we propose and evaluate several debiasing strategies, demonstrating that achieving meaningful reductions in bias requires training based interventions rather than relying solely on inference-time prompt-based approaches such as prompt engineering, etc.
>
---
#### [replaced 020] Non-Collaborative User Simulators for Tool Agents
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决工具代理在面对非合作用户时表现不佳的问题。提出一种模拟非合作行为的用户模拟器，用于测试和提升代理的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.23124](https://arxiv.org/pdf/2509.23124)**

> **作者:** Jeonghoon Shim; Woojung Song; Cheyon Jin; Seungwon KooK; Yohan Jo
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Tool agents interact with users through multi-turn dialogues to accomplish various tasks. Recent studies have adopted user simulation methods to develop these agents in multi-turn settings. However, existing user simulators tend to be agent-friendly, exhibiting only cooperative behaviors, failing to train and test agents against non-collaborative users in the real world. We propose a novel user simulator architecture that simulates four categories of non-collaborative behaviors: requesting unavailable services, digressing into tangential conversations, expressing impatience, and providing incomplete utterances. Our user simulator can simulate challenging and natural non-collaborative behaviors while reliably delivering all intents and information necessary to accomplish the task. Our experiments on MultiWOZ and {\tau}-bench reveal significant performance degradation in state-of-the-art tool agents when encountering non-collaborative users, as well as agent weaknesses under each non-collaborative condition such as escalated hallucinations and dialogue breakdowns. Our findings point to the need for methods that can improve agent robustness to the wide range of user behaviors encountered in deployment. We release the extensible simulation framework to help the community develop and stress-test tool agents under realistic conditions within their own service domains. Our code is available at this https URL.
>
---
#### [replaced 021] Chain-of-Lure: A Universal Jailbreak Attack Framework using Unconstrained Synthetic Narratives
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全攻击任务，旨在解决LLM被滥用的问题。提出一种无需模板的 jailbreak 方法，通过生成诱导问题实现攻击，并提升攻击效果与毒性评估。**

- **链接: [https://arxiv.org/pdf/2505.17519](https://arxiv.org/pdf/2505.17519)**

> **作者:** Wenhan Chang; Tianqing Zhu; Yu Zhao; Shuangyong Song; Ping Xiong; Wanlei Zhou
>
> **备注:** 23 pages, 3 main figures
>
> **摘要:** In the era of rapid generative AI development, interactions with large language models (LLMs) pose increasing risks of misuse. Prior research has primarily focused on attacks using template-based prompts and optimization-oriented methods, while overlooking the fact that LLMs possess strong unconstrained deceptive capabilities to attack other LLMs. This paper introduces a novel jailbreaking method inspired by the Chain-of-Thought mechanism. The attacker employs mission transfer to conceal harmful user intent within dialogue and generates a progressive chain of lure questions without relying on predefined templates, enabling successful jailbreaks. To further improve the attack's strength, we incorporate a helper LLM model that performs randomized narrative optimization over multi-turn interactions, enhancing the attack performance while preserving alignment with the original intent. We also propose a toxicity-based framework using third-party LLMs to evaluate harmful content and its alignment with malicious intent. Extensive experiments demonstrate that our method consistently achieves high attack success rates and elevated toxicity scores across diverse types of LLMs under black-box API settings. These findings reveal the intrinsic potential of LLMs to perform unrestricted attacks in the absence of robust alignment constraints. Our approach offers data-driven insights to inform the design of future alignment mechanisms. Finally, we propose two concrete defense strategies to support the development of safer generative models. Our code is available at this https URL
>
---
#### [replaced 022] CascadeMind at SemEval-2026 Task 4: A Hybrid Neuro-Symbolic Cascade for Narrative Similarity
- **分类: cs.CL**

- **简介: 该论文针对叙事相似性任务，提出CascadeMind系统，通过分析LLM的置信度来处理不确定性，提升比较准确性。**

- **链接: [https://arxiv.org/pdf/2601.19931](https://arxiv.org/pdf/2601.19931)**

> **作者:** Sebastien Kawada; Dylan Holyoak
>
> **摘要:** How should a system handle uncertainty when comparing narratives? We present CascadeMind, a hybrid neuro-symbolic system for SemEval-2026 Task 4 (Narrative Story Similarity) built around a core finding: an LLM's internal vote distribution is a reliable proxy for task difficulty, and confidence-aware routing outperforms uniform treatment of all cases. Our cascade samples eight parallel votes from Gemini 2.5 Flash, applying a supermajority threshold to resolve confident cases immediately (74% of instances at 85% development accuracy). Uncertain cases escalate to additional voting rounds (21%), and only perfect ties (5%) are deferred to a symbolic ensemble of five narrative signals grounded in classical narrative theory. The resulting difficulty gradient (85% -> 67% -> 61% by pathway) confirms that vote consensus tracks genuine ambiguity. In official Track A evaluation, CascadeMind placed 11th of 47 teams with 72.75% test accuracy (Hatzel et al., 2026), outperforming several systems built on larger and more expensive models. Gains are driven primarily by routing strategy rather than symbolic reasoning, suggesting that for narrative similarity, knowing when you don't know matters more than adding auxiliary representations.
>
---
#### [replaced 023] MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本建模中计算和内存成本高的问题。通过融合稀疏与线性注意力机制，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.11761](https://arxiv.org/pdf/2602.11761)**

> **作者:** MiniCPM Team; Wenhao An; Yingfa Chen; Yewei Fang; Jiayi Li; Xin Li; Yaohui Li; Yishan Li; Yuxuan Li; Biyuan Lin; Chuan Liu; Hezi Liu; Siyuan Liu; Hongya Lyu; Yinxu Pan; Shixin Ren; Xingyu Shen; Zhou Su; Haojun Sun; Yangang Sun; Zhen Leng Thai; Xin Tian; Rui Wang; Xiaorong Wang; Yudong Wang; Bo Wu; Xiaoyue Xu; Dong Xu; Shuaikang Xue; Jiawei Yang; Bowen Zhang; Jinqian Zhang; Letian Zhang; Shengnan Zhang; Xinyu Zhang; Xinyuan Zhang; Zhu Zhang; Hengyu Zhao; Jiacheng Zhao; Zhi Zheng; Jie Zhou; Zihan Zhou; Shuo Wang; Chaojun Xiao; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **备注:** MiniCPM-SALA Technical Report
>
> **摘要:** The evolution of large language models (LLMs) towards applications with ultra-long contexts faces challenges posed by the high computational and memory costs of the Transformer architecture. While existing sparse and linear attention mechanisms attempt to mitigate these issues, they typically involve a trade-off between memory efficiency and model performance. This paper introduces MiniCPM-SALA, a 9B-parameter hybrid architecture that integrates the high-fidelity long-context modeling of sparse attention (InfLLM-V2) with the global efficiency of linear attention (Lightning Attention). By employing a layer selection algorithm to integrate these mechanisms in a 1:3 ratio and utilizing a hybrid positional encoding (HyPE), the model maintains efficiency and performance for long-context tasks. Furthermore, we introduce a cost-effective continual training framework that transforms pre-trained Transformer-based models into hybrid models, which reduces training costs by approximately 75% compared to training from scratch. Extensive experiments show that MiniCPM-SALA maintains general capabilities comparable to full-attention models while offering improved efficiency. On a single NVIDIA A6000D GPU, the model achieves up to 3.5x the inference speed of the full-attention model at the sequence length of 256K tokens and supports context lengths of up to 1M tokens, a scale where traditional full-attention 8B models fail because of memory constraints.
>
---
#### [replaced 024] WAXAL: A Large-Scale Multilingual African Language Speech Corpus
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文介绍WAXAL，一个用于24种非洲语言的大规模语音数据集，旨在解决低资源语言技术发展不平衡的问题。任务为语音识别与合成，工作包括数据收集、标注及质量控制。**

- **链接: [https://arxiv.org/pdf/2602.02734](https://arxiv.org/pdf/2602.02734)**

> **作者:** Abdoulaye Diack; Perry Nelson; Kwaku Agbesi; Angela Nakalembe; MohamedElfatih MohamedKhair; Vusumuzi Dube; Tavonga Siyavora; Subhashini Venugopalan; Jason Hickey; Uche Okonkwo; Abhishek Bapna; Isaac Wiafe; Raynard Dodzi Helegah; Elikem Doe Atsakpo; Charles Nutrokpor; Fiifi Baffoe Payin Winful; Kafui Kwashie Solaga; Jamal-Deen Abdulai; Akon Obu Ekpezu; Audace Niyonkuru; Samuel Rutunda; Boris Ishimwe; Michael Melese; Engineer Bainomugisha; Joyce Nakatumba-Nabende; Andrew Katumba; Claire Babirye; Jonathan Mukiibi; Vincent Kimani; Samuel Kibacia; James Maina; Fridah Emmah; Ahmed Ibrahim Shekarau; Ibrahim Shehu Adamu; Yusuf Abdullahi; Howard Lakougna; Bob MacDonald; Hadar Shemtov; Aisha Walcott-Bryant; Moustapha Cisse; Avinatan Hassidim; Jeff Dean; Yossi Matias
>
> **备注:** Initial dataset release with added TTS, some more to come
>
> **摘要:** The advancement of speech technology has predominantly favored high-resource languages, creating a significant digital divide for speakers of most Sub-Saharan African languages. To address this gap, we introduce WAXAL, a large-scale, openly accessible speech dataset for 24 languages representing over 100 million speakers. The collection consists of two main components: an Automated Speech Recognition (ASR) dataset containing approximately 1,250 hours of transcribed, natural speech from a diverse range of speakers, and a Text-to-Speech (TTS) dataset with around 235 hours of high-quality, single-speaker recordings reading phonetically balanced scripts. This paper details our methodology for data collection, annotation, and quality control, which involved partnerships with four African academic and community organizations. We provide a detailed statistical overview of the dataset and discuss its potential limitations and ethical considerations. The WAXAL datasets are released at this https URL under the permissive CC-BY-4.0 license to catalyze research, enable the development of inclusive technologies, and serve as a vital resource for the digital preservation of these languages.
>
---
#### [replaced 025] LLaVE: Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LLaVE模型，解决多模态嵌入中正负样本相似度重叠问题，通过硬度加权对比学习提升表示能力，适用于图像-文本和文本-视频检索任务。**

- **链接: [https://arxiv.org/pdf/2503.04812](https://arxiv.org/pdf/2503.04812)**

> **作者:** Zhibin Lan; Liqiang Niu; Fandong Meng; Jie Zhou; Jinsong Su
>
> **备注:** Accepted by Findings of EMNLP 2025
>
> **摘要:** Universal multimodal embedding models play a critical role in tasks such as interleaved image-text retrieval, multimodal RAG, and multimodal clustering. However, our empirical results indicate that existing LMM-based embedding models trained with the standard InfoNCE loss exhibit a high degree of overlap in similarity distribution between positive and negative pairs, making it challenging to distinguish hard negative pairs effectively. To deal with this issue, we propose a simple yet effective framework that dynamically improves the embedding model's representation learning for negative pairs based on their discriminative difficulty. Within this framework, we train a series of models, named LLaVE, and evaluate them on the MMEB benchmark, which covers 4 meta-tasks and 36 datasets. Experimental results show that LLaVE establishes stronger baselines that achieve state-of-the-art (SOTA) performance while demonstrating strong scalability and efficiency. Specifically, LLaVE-2B surpasses the previous SOTA 7B models, while LLaVE-7B achieves a further performance improvement of 6.2 points. Although LLaVE is trained on image-text data, it can generalize to text-video retrieval tasks in a zero-shot manner and achieve strong performance, demonstrating its remarkable potential for transfer to other embedding tasks.
>
---
#### [replaced 026] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SPIRAL框架，通过自博弈强化学习提升语言模型的推理能力，解决传统方法依赖人工标注和领域奖励的问题。**

- **链接: [https://arxiv.org/pdf/2506.24119](https://arxiv.org/pdf/2506.24119)**

> **作者:** Bo Liu; Leon Guertler; Simon Yu; Zichen Liu; Penghui Qi; Daniel Balcells; Mickel Liu; Cheston Tan; Weiyan Shi; Min Lin; Wee Sun Lee; Natasha Jaques
>
> **备注:** Accepted at ICLR 2026. Code: this https URL
>
> **摘要:** Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, generating an automatic curriculum of stronger opponents, and eliminating the need for human supervision. To enable this self-play training at scale, we implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. SPIRAL produces reasoning capabilities that transfer broadly, improving performance by up to 10% across a suite of 8 reasoning benchmarks on 4 different models spanning Qwen and Llama model families, outperforming supervised fine-tuning on 25,000 expert game trajectories. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) yields the strongest results, with improvements observed across both base and instruction-tuned models. Analysis of chain-of-thought traces reveals that games develop distinct cognitive patterns that transfer to improve reasoning performance, with different games developing complementary strengths. Even models which have already been trained on reasoning tasks using RLVR, like DeepSeek-R1-Distill-Qwen-7B, still benefit from our approach. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities across diverse model architectures and training stages, highlighting a promising direction for autonomous reasoning development. Our code can be found in this https URL.
>
---
#### [replaced 027] RLP: Reinforcement as a Pretraining Objective
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出RLP，将强化学习作为预训练目标，解决模型独立思考能力不足的问题。通过信息驱动的奖励机制，提升模型推理性能。**

- **链接: [https://arxiv.org/pdf/2510.01265](https://arxiv.org/pdf/2510.01265)**

> **作者:** Ali Hatamizadeh; Syeda Nahida Akter; Shrimai Prabhumoye; Jan Kautz; Mostofa Patwary; Mohammad Shoeybi; Bryan Catanzaro; Yejin Choi
>
> **备注:** ICLR 2026 camera ready
>
> **摘要:** The dominant paradigm for training large reasoning models starts with pre-training using next-token prediction loss on vast amounts of data. Reinforcement learning, while powerful in scaling reasoning, is introduced only as the very last phase of post-training, preceded by supervised fine-tuning. While dominant, is this an optimal way of training? In this paper, we present RLP, an information-driven reinforcement pretraining objective, that brings the core spirit of reinforcement learning -- exploration -- to the last phase of pretraining. The key idea is to treat chain-of-thought as an exploratory action, with rewards computed based on the information gain it provides for predicting future tokens. This training objective essentially encourages the model to think for itself before predicting what comes next, thus teaching an independent thinking behavior earlier in the pretraining. More concretely, the reward signal measures the increase in log-likelihood of the next token when conditioning on both context and a sampled reasoning chain, compared to conditioning on context alone. This approach yields a verifier-free dense reward signal, allowing for efficient training for the full document stream during pretraining. Specifically, RLP reframes reinforcement learning for reasoning as a pretraining objective on ordinary text, bridging the gap between next-token prediction and the emergence of useful chain-of-thought reasoning. Pretraining with RLP on Qwen3-1.7B-Base lifts the overall average across an eight-benchmark math-and-science suite by 19%. With identical post-training, the gains compound, with the largest improvements on reasoning-heavy tasks such as AIME25 and MMLU-Pro. Applying RLP to the Nemotron-Nano-12B-v2 increases the overall average from 42.81% to 61.32% and raises the average on scientific reasoning by 23%, demonstrating scalability across architectures and model sizes.
>
---
#### [replaced 028] SQUiD: Synthesizing Relational Databases from Unstructured Text
- **分类: cs.DB; cs.CL**

- **简介: 该论文提出SQUiD框架，用于从非结构化文本生成关系型数据库。任务是将文本数据转化为结构化数据，解决数据格式不统一的问题。工作包括分解任务为四阶段，并利用大语言模型实现高效生成。**

- **链接: [https://arxiv.org/pdf/2505.19025](https://arxiv.org/pdf/2505.19025)**

> **作者:** Mushtari Sadia; Zhenning Yang; Yunming Xiao; Ang Chen; Amrita Roy Chowdhury
>
> **摘要:** Relational databases are central to modern data management, yet most data exists in unstructured forms like text documents. To bridge this gap, we leverage large language models (LLMs) to automatically synthesize a relational database by generating its schema and populating its tables from raw text. We introduce SQUiD, a novel neurosymbolic framework that decomposes this task into four stages, each with specialized techniques. Our experiments show that SQUiD consistently outperforms baselines across diverse datasets. Our code and datasets are publicly available at: this https URL.
>
---
#### [replaced 029] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于GUI交互任务，解决多模态代理在切换操作上的可靠性问题。提出StaR方法，提升切换指令执行准确率。**

- **链接: [https://arxiv.org/pdf/2509.13615](https://arxiv.org/pdf/2509.13615)**

> **作者:** Zongru Wu; Rui Mao; Zhiyuan Tian; Pengzhou Cheng; Tianjie Ju; Zheng Wu; Lingzhong Dong; Haiyue Sheng; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions derived from public datasets. Evaluation results of existing agents demonstrate their notable unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a multimodal reasoning method that enables agents to perceive the current toggle state, infer the desired state from the instruction, and act accordingly. Experiments on four multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public agentic benchmarks show that StaR also enhances general agentic task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code and benchmark: this https URL.
>
---
#### [replaced 030] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决LRM推理过程中的安全隐患。通过引入IPO方法，强化安全推理，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2509.24393](https://arxiv.org/pdf/2509.24393)**

> **作者:** Yichi Zhang; Yue Ding; Jingwen Yang; Tianwei Luo; Dongbai Li; Ranjie Duan; Qiang Liu; Hang Su; Yinpeng Dong; Jun Zhu
>
> **备注:** ICLR 2026
>
> **摘要:** Although Large Reasoning Models (LRMs) have progressed in solving complex problems, their chain-of-thought (CoT) reasoning often contains harmful content that can persist even when the final responses appear safe. We show that this issue still remains in existing methods which overlook the unique significance of safe reasoning, undermining their trustworthiness and posing potential risks in applications if unsafe reasoning is accessible for and exploited by malicious users. We therefore shift our focus to aligning the safety of reasoning itself in this paper and explore process supervision as the solution. However, simply rewarding safe reasoning proves inadequate due to low rollout diversity and limited training signals. To tackle this challenge, we first delve into the characteristics of safe reasoning and uncover several critical insights that 1) safe reasoning is often consolidated by a few critical steps of safety triggers; 2) compliance cues strongly correlate with unsafe continuations; and 3) corrective interventions reliably steer unsafe trajectories towards safer traces. Motivated by these, we propose Intervened Preference Optimization (IPO), an alignment method that enforces safe reasoning by substituting compliance steps with safety triggers and constructing pairs for preference learning with strong signals. Experiments on jailbreak and adversarial safety benchmarks demonstrate that IPO remarkably improves overall safety regarding both reasoning and responses, outperforming SFT-based and RL-based baselines with a relative reduction of over 30% in harmfulness, while preserving excellent performance across diverse reasoning tasks. The results highlight the importance of explicit alignment for reasoning and provide a practical path to safer LRMs.
>
---
#### [replaced 031] Learning Ordinal Probabilistic Reward from Preferences
- **分类: cs.CL**

- **简介: 该论文属于奖励建模任务，旨在解决传统方法在监督成本高和概率解释不足的问题。提出PRM和OPRM模型，提升奖励准确性与数据效率。**

- **链接: [https://arxiv.org/pdf/2602.12660](https://arxiv.org/pdf/2602.12660)**

> **作者:** Longze Chen; Lu Wang; Renke Shan; Ze Gong; Run Luo; Jiaming Li; Jing Luo; Qiyao Wang; Min Yang
>
> **备注:** 28 pages, 5 figures, ICLR 2026
>
> **摘要:** Reward models are crucial for aligning large language models (LLMs) with human values and intentions. Existing approaches follow either Generative (GRMs) or Discriminative (DRMs) paradigms, yet both suffer from limitations: GRMs typically demand costly point-wise supervision, while DRMs produce uncalibrated relative scores that lack probabilistic interpretation. To address these challenges, we introduce a novel reward modeling paradigm: Probabilistic Reward Model (PRM). Instead of modeling reward as a deterministic scalar, our approach treats it as a random variable, learning a full probability distribution for the quality of each response. To make this paradigm practical, we present its closed-form, discrete realization: the Ordinal Probabilistic Reward Model (OPRM), which discretizes the quality score into a finite set of ordinal ratings. Building on OPRM, we propose a data-efficient training strategy called Region Flooding Tuning (RgFT). It enables rewards to better reflect absolute text quality by incorporating quality-level annotations, which guide the model to concentrate the probability mass within corresponding rating sub-regions. Experiments on various reward model benchmarks show that our method improves accuracy by $\textbf{2.9%}\sim\textbf{7.4%}$ compared to prior reward models, demonstrating strong performance and data efficiency. Analysis of the score distribution provides evidence that our method captures not only relative rankings but also absolute quality.
>
---
#### [replaced 032] Spilled Energy in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型检测任务，旨在解决事实错误和幻觉问题。通过将softmax分类器视为能量模型，引入两种无需训练的指标检测幻觉。**

- **链接: [https://arxiv.org/pdf/2602.18671](https://arxiv.org/pdf/2602.18671)**

> **作者:** Adrian Robert Minut; Hazem Dewidar; Iacopo Masi
>
> **摘要:** We reinterpret the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows us to track "energy spills" during decoding, which we empirically show correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), our method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, we achieve this without requiring trained probe classifiers or activation ablations. Instead, we introduce two completely training-free metrics derived directly from output logits: spilled energy, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and marginalized energy, which is measurable at a single step. Evaluated on nine benchmarks across state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and on synthetic algebraic operations (Qwen3), our approach demonstrates robust, competitive hallucination detection and cross-task generalization. Notably, these results hold for both pretrained and instruction-tuned variants without introducing any training overhead. Code available at: this http URL
>
---
#### [replaced 033] Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts
- **分类: cs.CL**

- **简介: 该论文属于大模型训练任务，解决长上下文训练中GPU内存不足的问题。通过引入OOMB系统，采用分块递归训练和优化KV缓存管理，显著降低内存占用，实现单卡训练超长文本。**

- **链接: [https://arxiv.org/pdf/2602.02108](https://arxiv.org/pdf/2602.02108)**

> **作者:** Wenhao Li; Daohai Yu; Gen Luo; Yuxin Zhang; Fei Chao; Rongrong Ji; Yifan Wu; Jiaxin Liu; Ziyang Gong; Zimu Liao
>
> **摘要:** Training Large Language Models (LLMs) on long contexts is severely constrained by prohibitive GPU memory overhead, not training time. The primary culprits are the activations, whose memory footprints scale linearly with sequence length. We introduce OOMB, a highly memory-efficient training system that directly confronts this barrier. Our approach employs a chunk-recurrent training framework with on-the-fly activation recomputation, which maintains a constant activation memory footprint (O(1)) and shifts the primary bottleneck to the growing KV cache. To manage the KV cache, OOMB integrates a suite of synergistic optimizations: a paged memory manager for both the KV cache and its gradients to eliminate fragmentation, asynchronous CPU offloading to hide data transfer latency, and page-level sparse attention to reduce both computational complexity and communication overhead. The synergy of these techniques yields exceptional efficiency. Our empirical results show that for every additional 10K tokens of context, the end-to-end training memory overhead increases by a mere 10MB for Qwen2.5-7B. This allows training Qwen2.5-7B with a 4M-token context on a single H200 GPU, a feat that would otherwise require a large cluster using context parallelism. This work represents a substantial advance in resource efficiency for long-context LLM training. The source code is available at this https URL.
>
---
#### [replaced 034] FrugalRAG: Less is More in RL Finetuning for Multi-Hop Question Answering
- **分类: cs.CL**

- **简介: 该论文针对多跳问答任务，解决RAG在RL微调中效率与准确率不平衡的问题。提出FrugalRAG框架，通过两阶段微调减少检索步骤，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2507.07634](https://arxiv.org/pdf/2507.07634)**

> **作者:** Abhinav Java; Srivathsan Koundinyan; Nagarajan Natarajan; Amit Sharma
>
> **摘要:** Reinforcement learning (RL) based on the final answer's reward has driven recent progress in small language models (SLMs) on reasoning-heavy tasks such as math and code. However, applying the same techniques to retrieval-augmented generation (RAG) benchmarks like multi-hop QA has yielded limited gains, often trailing supervised or prompting-only baselines. Instead, we argue that a viable path for RL in multi-hop QA is to use test-time scaling judiciously to optimize both final answer accuracy and efficiency in reaching that answer. We propose FrugalRAG, a two-stage finetuning framework that adaptively reduces the number of retrieval steps based on a question's difficulty. First, we train an SLM with supervised finetuning on a full-exploration policy that generates broad sub-queries. Then, we apply RL to adaptively prune search depth based on question difficulty, directly rewarding policies that balance correctness with frugality. Unlike prior approaches requiring 10x more data, our method achieves competitive performance with only approximately 1,000 examples. On HotPotQA and other multi-hop QA benchmarks, FrugalRAG attains state-of-the-art efficiency-accuracy tradeoffs, cutting retrieval cost nearly in half. Moreover, on the challenging BrowseCompPlus benchmark, it generalizes zero-shot and surpasses SLM-based and other baselines. These results demonstrate the use of RL not to increase reasoning steps, but to reduce them, as an effective solution for scalable and efficient RAG.
>
---
#### [replaced 035] To Think or Not To Think, That is The Question for Large Reasoning Models in Theory of Mind Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于社会推理任务，研究大型推理模型在心智理论（ToM）中的表现。工作包括对比不同模型在ToM基准上的表现，分析其优劣，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2602.10625](https://arxiv.org/pdf/2602.10625)**

> **作者:** Nanxu Gong; Haotian Li; Sixun Dong; Jianxun Lian; Yanjie Fu; Xing Xie
>
> **摘要:** Theory of Mind (ToM) assesses whether models can infer hidden mental states such as beliefs, desires, and intentions, which is essential for natural social interaction. Although recent progress in Large Reasoning Models (LRMs) has boosted step-by-step inference in mathematics and coding, it is still underexplored whether this benefit transfers to socio-cognitive skills. We present a systematic study of nine advanced Large Language Models (LLMs), comparing reasoning models with non-reasoning models on three representative ToM benchmarks. The results show that reasoning models do not consistently outperform non-reasoning models and sometimes perform worse. A fine-grained analysis reveals three insights. First, slow thinking collapses: accuracy significantly drops as responses grow longer, and larger reasoning budgets hurt performance. Second, moderate and adaptive reasoning benefits performance: constraining reasoning length mitigates failure, while distinct success patterns demonstrate the necessity of dynamic adaptation. Third, option matching shortcut: when multiple choice options are removed, reasoning models improve markedly, indicating reliance on option matching rather than genuine deduction. We also design two intervention approaches: Slow-to-Fast (S2F) adaptive reasoning and Think-to-Match (T2M) shortcut prevention to further verify and mitigate the problems. With all results, our study highlights the advancement of LRMs in formal reasoning (e.g., math, code) cannot be fully transferred to ToM, a typical task in social reasoning. We conclude that achieving robust ToM requires developing unique capabilities beyond existing reasoning methods.
>
---
#### [replaced 036] Not-Just-Scaling Laws: Towards a Better Understanding of the Downstream Impact of Language Model Design Decisions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在探究语言模型设计决策对下游任务性能的影响。通过分析92个模型，发现除规模外的其他因素显著影响性能，提出更全面的评估框架。**

- **链接: [https://arxiv.org/pdf/2503.03862](https://arxiv.org/pdf/2503.03862)**

> **作者:** Emmy Liu; Amanda Bertsch; Lintang Sutawika; Lindia Tjuatja; Patrick Fernandes; Lara Marinov; Michael Chen; Shreya Singhal; Carolin Lawrence; Aditi Raghunathan; Kiril Gashteovski; Graham Neubig
>
> **摘要:** Improvements in language model capabilities are often attributed to increasing model size or training data, but in some cases smaller models trained on curated data or with different architectural decisions can outperform larger ones trained on more tokens. What accounts for this? To quantify the impact of these design choices, we meta-analyze 92 open-source pretrained models across a wide array of scales, including state-of-the-art open-weights models as well as less performant models and those with less conventional design decisions. We find that by incorporating features besides model size and number of training tokens, we can achieve a relative 3-28% increase in ability to predict downstream performance compared with using scale alone. Analysis of model design decisions reveal insights into data composition, such as the trade-off between language and code tasks at 15-25\% code, as well as the better performance of some architectural decisions such as choosing rotary over learned embeddings. Broadly, our framework lays a foundation for more systematic investigation of how model development choices shape final capabilities.
>
---
#### [replaced 037] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CityLens，用于评估大视觉语言模型在城市社会经济感知中的能力。任务是通过图像预测社会经济指标，解决模型在该领域的能力不足问题。工作包括构建多模态数据集和设计多种评估方法。**

- **链接: [https://arxiv.org/pdf/2506.00530](https://arxiv.org/pdf/2506.00530)**

> **作者:** Tianhui Liu; Hetian Pang; Xin Zhang; Tianjian Ouyang; Zhiyuan Zhang; Jie Feng; Yong Li; Pan Hui
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Understanding urban socioeconomic conditions through visual data is a challenging yet essential task for sustainable urban development and policy planning. In this work, we introduce \textit{CityLens}, a comprehensive benchmark designed to evaluate the capabilities of Large Vision-Language Models (LVLMs) in predicting socioeconomic indicators from satellite and street view imagery. We construct a multi-modal dataset covering a total of 17 globally distributed cities, spanning 6 key domains: economy, education, crime, transport, health, and environment, reflecting the multifaceted nature of urban life. Based on this dataset, we define 11 prediction tasks and utilize 3 evaluation paradigms: Direct Metric Prediction, Normalized Metric Estimation, and Feature-Based Regression. We benchmark 17 state-of-the-art LVLMs across these tasks. These make CityLens the most extensive socioeconomic benchmark to date in terms of geographic coverage, indicator diversity, and model scale. Our results reveal that while LVLMs demonstrate promising perceptual and reasoning capabilities, they still exhibit limitations in predicting urban socioeconomic indicators. CityLens provides a unified framework for diagnosing these limitations and guiding future efforts in using LVLMs to understand and predict urban socioeconomic patterns. The code and data are available at this https URL.
>
---
#### [replaced 038] Wikipedia in the Era of LLMs: Evolution and Risks
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文研究LLMs对Wikipedia的影响，属于NLP任务。分析LLMs对Wikipedia内容及NLP任务的潜在风险，提出评估框架并进行模拟实验。**

- **链接: [https://arxiv.org/pdf/2503.02879](https://arxiv.org/pdf/2503.02879)**

> **作者:** Siming Huang; Yuliang Xu; Mingmeng Geng; Yao Wan; Dongping Chen
>
> **备注:** Accepted by TMLR: this https URL
>
> **摘要:** In this paper, we present a comprehensive analysis and monitoring framework for the impact of Large Language Models (LLMs) on Wikipedia, examining the evolution of Wikipedia through existing data and using simulations to explore potential risks. We begin by analyzing article content and page views to study the recent changes in Wikipedia and assess the impact of LLMs. Subsequently, we evaluate how LLMs affect various Natural Language Processing (NLP) tasks related to Wikipedia, including machine translation and retrieval-augmented generation (RAG). Our findings and simulation results reveal that Wikipedia articles have been affected by LLMs, with an impact of approximately 1% in certain categories. If the machine translation benchmark based on Wikipedia is influenced by LLMs, the scores of the models may become inflated, and the comparative results among models could shift. Moreover, the effectiveness of RAG might decrease if the knowledge has been contaminated by LLMs. While LLMs have not yet fully changed Wikipedia's language and knowledge structures, we believe that our empirical findings signal the need for careful consideration of potential future risks in NLP research. We release all the experimental dataset and source code at: this https URL
>
---
#### [replaced 039] Prompt and Parameter Co-Optimization for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM性能提升问题。通过联合优化提示和参数，提出MetaTuner框架，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2509.24245](https://arxiv.org/pdf/2509.24245)**

> **作者:** Xiaohe Bo; Rui Li; Zexu Sun; Quanyu Dai; Zeyu Zhang; Zihang Tian; Xu Chen; Zhenhua Dong
>
> **备注:** ICLR 2026
>
> **摘要:** Prompt optimization and fine-tuning are two major approaches to improve the performance of Large Language Models (LLMs). They enhance the capabilities of LLMs from complementary perspectives: the former through explicit natural language, and the latter through implicit parameter updates. However, prior work has typically studied them in isolation, leaving their synergistic potential largely underexplored. To bridge this gap, in this paper, we introduce MetaTuner, a novel framework that jointly integrates prompt optimization and fine-tuning for LLM training. Specifically, we introduce two neural networks to generate prompts and parameters, respectively, while allowing them to share a common bottom encoding layer to enable knowledge sharing. By the guidance of the final supervised signals, our framework is optimized to discover the optimal combinations between the prompts and parameters. Given that prompt learning involves discrete optimization while fine-tuning operates in a continuous parameter space, we design a supervised regularization loss to train our framework effectively. Extensive experiments across diverse benchmarks show that our method consistently outperforms the baselines.
>
---
#### [replaced 040] FASA: Frequency-aware Sparse Attention
- **分类: cs.CL**

- **简介: 该论文提出FASA，解决长输入下LLM内存瓶颈问题。通过动态预测token重要性，提升注意力效率，显著提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2602.03152](https://arxiv.org/pdf/2602.03152)**

> **作者:** Yifei Wang; Yueqi Wang; Zhenrui Yue; Huimin Zeng; Yong Wang; Ismini Lourentzou; Zhengzhong Tu; Xiangxiang Chu; Julian McAuley
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inputs: the prohibitive memory footprint of the Key Value (KV) cache. To address this bottleneck, the token pruning paradigm leverages attention sparsity to selectively retain a small, critical subset of tokens. However, existing approaches fall short, with static methods risking irreversible information loss and dynamic strategies employing heuristics that insufficiently capture the query-dependent nature of token importance. We propose FASA, a novel framework that achieves query-aware token eviction by dynamically predicting token importance. FASA stems from a novel insight into RoPE: the discovery of functional sparsity at the frequency-chunk (FC) level. Our key finding is that a small, identifiable subset of "dominant" FCs consistently exhibits high contextual agreement with the full attention head. This provides a robust and computationally free proxy for identifying salient tokens. Building on this insight, FASA first identifies a critical set of tokens using dominant FCs, and then performs focused attention computation solely on this pruned subset. Across a spectrum of long-context tasks, from sequence modeling to complex CoT reasoning, FASA consistently outperforms all token-eviction baselines and achieves near-oracle accuracy, demonstrating remarkable robustness even under constraint budgets. Notably, on LongBench-V1, FASA reaches nearly 100\% of full-KV performance when only keeping 256 tokens, and achieves 2.56$\times$ speedup using just 18.9\% of the cache on AIME24.
>
---
#### [replaced 041] Characterizing Pattern Matching and Its Limits on Compositional Task Structures
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLMs在组合任务中的模式匹配行为及其局限性，通过形式化定义和实验分析，揭示其泛化能力与数据量、结构复杂度的关系，旨在提供可验证的边界和诊断方法。**

- **链接: [https://arxiv.org/pdf/2505.20278](https://arxiv.org/pdf/2505.20278)**

> **作者:** Hoyeon Chang; Jinho Park; Hanseul Cho; Sohee Yang; Miyoung Ko; Hyeonbin Hwang; Seungpil Won; Dohaeng Lee; Youbin Ahn; Minjoon Seo
>
> **摘要:** Despite impressive capabilities, LLMs' successes often rely on pattern-matching behaviors, yet these are also linked to OOD generalization failures in compositional tasks. However, behavioral studies commonly employ task setups that allow multiple generalization sources (e.g., algebraic invariances, structural repetition), obscuring a precise and testable account of how well LLMs perform generalization through pattern matching and their limitations. To address this ambiguity, we first formalize pattern matching as functional equivalence, i.e., identifying pairs of subsequences of inputs that consistently lead to identical results when the rest of the input is held constant. Then, we systematically study how decoder-only Transformer and Mamba behave in controlled tasks with compositional structures that isolate this mechanism. Our formalism yields predictive and quantitative insights: (1) Instance-wise success of pattern matching is well predicted by the number of contexts witnessing the relevant functional equivalence. (2) We prove a tight sample complexity bound of learning a two-hop structure by identifying the exponent of the data scaling law for perfect in-domain generalization. Our empirical results align with the theoretical prediction, under 20x parameter scaling and across architectures. (3) Path ambiguity is a structural barrier: when a variable influences the output via multiple paths, models fail to form unified intermediate state representations, impairing accuracy and interpretability. (4) Chain-of-Thought reduces data requirements yet does not resolve path ambiguity. Hence, we provide a predictive, falsifiable boundary for pattern matching and a foundational diagnostic for disentangling mixed generalization mechanisms.
>
---
#### [replaced 042] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AdaptVision，解决视觉语言模型计算开销大的问题。通过自适应获取视觉标记，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2512.03794](https://arxiv.org/pdf/2512.03794)**

> **作者:** Zichuan Lin; Yicheng Liu; Yang Yang; Lvfang Tao; Deheng Ye
>
> **备注:** Accepted by CVPR 2026. Code and models are available at this https URL
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods.
>
---
#### [replaced 043] EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型控制任务，解决现有框架计算效率低、扩展性差的问题。提出EasySteer框架，提升推理性能并支持多种应用。**

- **链接: [https://arxiv.org/pdf/2509.25175](https://arxiv.org/pdf/2509.25175)**

> **作者:** Haolei Xu; Xinyu Mei; Yuchen Yan; Rui Zhou; Wenqi Zhang; Weiming Lu; Yueting Zhuang; Yongliang Shen
>
> **备注:** Functionality upgrade. Code: this https URL Demo: this https URL
>
> **摘要:** Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 10.8-22.3$\times$ speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models.
>
---
#### [replaced 044] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究大模型在推理过程中出现的答案不一致问题。通过实验发现模型同时依赖推理和记忆检索，提出FARL框架提升推理能力。**

- **链接: [https://arxiv.org/pdf/2509.24156](https://arxiv.org/pdf/2509.24156)**

> **作者:** Yuhui Wang; Changjiang Li; Guangke Chen; Jiacheng Liang; Ting Wang
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large reasoning models (LRMs) exhibit unprecedented capabilities in solving complex problems through Chain-of-Thought (CoT) reasoning. However, recent studies reveal that their final answers often contradict their own reasoning traces. We hypothesize that this inconsistency stems from two competing mechanisms for generating answers: CoT reasoning and memory retrieval. To test this hypothesis, we conduct controlled experiments that challenge LRMs with misleading cues during reasoning and/or corrupted answers during retrieval. Our results across models and datasets confirm that both mechanisms operate simultaneously, with their relative dominance influenced by multiple factors: problem domains, model scales, and fine-tuning approaches (e.g., reinforcement learning vs. distillation). The findings reveal a critical limitation in current reasoning fine-tuning paradigms: models can exploit the retrieval mechanism as a shortcut, effectively "hacking" the reward signal and undermining genuine reasoning development. To address this challenge, we introduce FARL, a novel fine-tuning framework that integrates memory unlearning with reinforcement learning. By carefully suppressing retrieval shortcuts during the fine-tuning process, FARL promotes reasoning-dominant behavior and enhances generalizable reasoning capabilities. The code is available: this https URL.
>
---
#### [replaced 045] Training Large Language Models To Reason In Parallel With Global Forking Tokens
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型训练任务，旨在解决并行推理中多样性与准确性难以平衡的问题。通过引入全局损失函数，提升模型生成多样化且准确的推理路径。**

- **链接: [https://arxiv.org/pdf/2510.05132](https://arxiv.org/pdf/2510.05132)**

> **作者:** Sheng Jia; Xiao Wang; Shiva Prasad Kasiviswanathan
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Although LLMs have demonstrated improved performance by scaling parallel test-time compute, doing so relies on generating reasoning paths that are both diverse and accurate. For challenging problems, the forking tokens that trigger diverse yet correct reasoning modes are typically deep in the sampling tree. Consequently, common strategies to encourage diversity, such as temperature scaling, encounter a worsened trade-off between diversity and accuracy. Motivated by this challenge, we treat parallel reasoning as a set-of-next-token-prediction problem and incorporate a set-based global loss into Supervised Fine-Tuning (SFT) using bipartite matching between global forking tokens and unique reasoning traces. We observe that whereas naive fine-tuning with multiple reasoning traces collapses these unique reasoning modes, our proposed method, Set Supervised Fine-Tuning (SSFT), preserves these modes and produces emergent global forking tokens. Global Forking Policy Optimization (GFPO) leverages these maximally steerable tokens to incentivize complex reasoning, and the resulting models consistently outperform their SFT counterparts with GRPO on both math reasoning and execution-based code generation benchmarks.
>
---
#### [replaced 046] OJBench: A Competition Level Code Benchmark For Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于代码推理任务，旨在解决现有基准无法评估大语言模型竞争级代码能力的问题。作者构建了OJBench基准，包含232道编程竞赛题，用于更严格地测试模型的代码推理能力。**

- **链接: [https://arxiv.org/pdf/2506.16395](https://arxiv.org/pdf/2506.16395)**

> **作者:** Zhexu Wang; Yiping Liu; Yejie Wang; Wenyang He; Bofei Gao; Muxi Diao; Yanxu Chen; Kelin Fu; Flood Sung; Zhilin Yang; Tianyu Liu; Weiran Xu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated significant progress in math and code reasoning capabilities. However, existing code benchmark are limited in their ability to evaluate the full spectrum of these capabilities, particularly at the competitive level. To bridge this gap, we introduce OJBench, a novel and challenging benchmark designed to assess the competitive-level code reasoning abilities of LLMs. OJBench comprises 232 programming competition problems from NOI and ICPC, providing a more rigorous test of models' reasoning skills. We conducted a comprehensive evaluation using OJBench on 37 models, including both closed-source and open-source models, reasoning-oriented and non-reasoning-oriented models. Our results indicate that even state-of-the-art reasoning-oriented models, such as o4-mini and Gemini-2.5-pro-exp, struggle with highly challenging competition-level problems. This highlights the significant challenges that models face in competitive-level code reasoning.
>
---
#### [replaced 047] Regularization Through Reasoning: Systematic Improvements in Language Model Classification via Explanation-Enhanced Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型分类任务，旨在提升模型准确性与可靠性。通过在微调中加入解释，研究发现即使使用随机文本也能改善性能，表明结构作用大于语义。**

- **链接: [https://arxiv.org/pdf/2511.02044](https://arxiv.org/pdf/2511.02044)**

> **作者:** Vivswan Shah; Randy Cogill; Hanwei Yue; Gopinath Chennupati; Rinat Khaziev
>
> **摘要:** Fine-tuning LLMs for classification typically maps inputs directly to labels. We ask whether attaching brief explanations to each label during fine-tuning yields better models. We evaluate conversational response quality along three axes: naturalness, comprehensiveness, and on-topic adherence, each rated on 5-point scales. Using ensemble-generated data from multiple LLMs, we fine-tune a 7B-parameter model and test across six diverse conversational datasets. Across 18 dataset, task settings, label-plus-explanation training outperforms label-only baselines. A central and unexpected result concerns random tokens. We replace human-written explanations with text that is syntactically incoherent yet vocabulary-aligned with the originals (e.g., shuffled or bag-of-words variants). Despite lacking semantics, these pseudo-explanations still improve accuracy over label-only training and often narrow much of the gap to true explanations. The effect persists across datasets and training seeds, indicating that gains arise less from meaning than from structure: the extra token budget encourages richer intermediate computation and acts as a regularizer that reduces over-confident shortcuts. Internal analyses support this view: explanation-augmented models exhibit higher activation entropy in intermediate layers alongside sharper predictive mass at the output layer, consistent with increased deliberation before decision. Overall, explanation-augmented fine-tuning, whether with genuine rationales or carefully constructed random token sequences, improves accuracy and reliability for LLM classification while clarifying how token-level scaffolding shapes computation during inference.
>
---
#### [replaced 048] Dynamic Level Sets
- **分类: cs.CC; cs.CL; math-ph; math.DS; math.HO**

- **简介: 该论文提出“动态层次集”概念，探讨其在计算理论中的独特性，旨在解决传统模型未能涵盖的不可计算过程问题。**

- **链接: [https://arxiv.org/pdf/2602.22530](https://arxiv.org/pdf/2602.22530)**

> **作者:** Michael Stephen Fiske
>
> **备注:** 7 pages
>
> **摘要:** A mathematical concept is identified and analyzed that is implicit in the 2012 paper Turing Incomputable Computation, presented at the Alan Turing Centenary Conference (Turing-100, Manchester). The concept, called dynamic level sets, is distinct from mathematical concepts in the standard literature on dynamical systems, topology, and computability theory. A new mathematical object is explained and why it may have escaped prior characterizations, including the classical result of de Leeuw, Moore, Shannon, and Shapiro that probabilistic Turing machines (with bias $p$ where $p$ is Turing computable) compute no more than deterministic ones. A key mechanism underlying the concept is the Principle of Self-Modifiability, whereby the physical realization of an invariant logical level set is reconfigured at each computational step by an incomputable physical process.
>
---
#### [replaced 049] Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究MoE模型在推理任务中的最优稀疏性，探讨其对记忆和推理能力的影响。通过调整参数和计算资源，发现推理性能受活跃计算和每参数数据量影响。**

- **链接: [https://arxiv.org/pdf/2508.18672](https://arxiv.org/pdf/2508.18672)**

> **作者:** Taishi Nakamura; Satoki Ishikawa; Masaki Kawamura; Takumi Okamoto; Daisuke Nohara; Jun Suzuki; Rio Yokota
>
> **备注:** Accepted as an oral at ICLR 2026
>
> **摘要:** Empirical scaling laws have driven the evolution of large language models (LLMs), yet their coefficients shift whenever the model architecture or data pipeline changes. Mixture-of-Experts (MoE) models, now standard in state-of-the-art systems, introduce a new sparsity dimension that current dense-model frontiers overlook. We investigate how MoE sparsity influences two distinct capability regimes: memorization skills and reasoning skills. By training MoE families that vary total parameters, active parameters, and top-$k$ routing under fixed compute budgets, we disentangle pre-training loss from downstream accuracy. Our results reveal two principles. First, Active FLOPs: models with identical training loss but greater active compute achieve higher reasoning accuracy. Second, Total tokens per parameter (TPP): memorization tasks improve with more parameters, while reasoning tasks benefit from optimal TPP, indicating that reasoning is data-hungry. Neither reinforcement learning post-training (GRPO) nor increased test-time compute alters these trends. We therefore argue that optimal MoE sparsity must be determined jointly by active FLOPs and TPP, revising the classical picture of compute-optimal scaling. Our model checkpoints, code and logs are open-source at this https URL.
>
---
#### [replaced 050] Unleashing Low-Bit Inference on Ascend NPUs: A Comprehensive Evaluation of HiFloat Formats
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究低比特浮点格式在Ascend NPU上的推理效率问题。针对大模型推理中的精度与效率矛盾，提出HiFloat格式，通过实验验证其在不同任务中的性能优势。**

- **链接: [https://arxiv.org/pdf/2602.12635](https://arxiv.org/pdf/2602.12635)**

> **作者:** Pengxiang Zhao; Hui-Ling Zhen; Xing Li; Han Bao; Weizhe Lin; Zhiyuan Yang; Manyi Zhang; Yuanyong Luo; Ziwei Yu; Xin Wang; Mingxuan Yuan; Xianzhi Yu; Zhenhua Dong
>
> **摘要:** As LLMs scale, low-bit floating-point formats like MXFP and NVFP4 offer new opportunities for precision and efficiency. In this work, we evaluate HiFloat (HiF8 and HiF4), a family of formats tailored for Ascend NPUs. Through rigorous comparison across weight-activation and KV-cache tasks, we provide three key insights: (1) INT8 suits narrow-range data, while floating-point formats excel with high-variance data; (2) in 4-bit regimes, HiF4's hierarchical scaling prevents the accuracy collapse seen in integer formats; and (3) HiFloat is fully compatible with state-of-the-art post-training quantization frameworks. Overall, HiFloat provides a solution for high-efficiency LLM inference on NPUs.
>
---
#### [replaced 051] Reward Models Inherit Value Biases from Pretraining
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究奖励模型（RMs）从预训练模型继承价值偏见的问题，分析其对人类价值观的影响。任务属于模型对齐与价值观一致性研究，旨在揭示RMs的偏见来源并强调预训练阶段的重要性。**

- **链接: [https://arxiv.org/pdf/2601.20838](https://arxiv.org/pdf/2601.20838)**

> **作者:** Brian Christian; Jessica A. F. Thompson; Elle Michelle Yang; Vincent Adam; Hannah Rose Kirk; Christopher Summerfield; Tsvetomira Dumbalska
>
> **摘要:** Reward models (RMs) are central to aligning large language models (LLMs) with human values but have received less attention than pretrained and post-trained LLMs themselves. Because RMs are initialized from LLMs, they inherit representations that shape their behavior, but the nature and extent of this influence remain understudied. In a comprehensive study of 10 leading open-weight RMs using validated psycholinguistic corpora, we show that RMs exhibit significant differences along multiple dimensions of human value as a function of their base model. Using the "Big Two" psychological axes, we show a robust preference of Llama RMs for "agency" and a corresponding robust preference of Gemma RMs for "communion." This phenomenon holds even when the preference data and finetuning process are identical, and we trace it back to the logits of the respective instruction-tuned and pretrained models. These log-probability differences themselves can be formulated as an implicit RM; we derive usable implicit reward scores and show that they exhibit the very same agency/communion difference. We run experiments training RMs with ablations for preference data source and quantity, which demonstrate that this effect is not only repeatable but surprisingly durable. Despite RMs being designed to represent human preferences, our evidence shows that their outputs are influenced by the pretrained LLMs on which they are based. This work underscores the importance of safety and alignment efforts at the pretraining stage, and makes clear that open-source developers' choice of base model is as much a consideration of values as of performance.
>
---
#### [replaced 052] PolySkill: Learning Generalizable Skills Through Polymorphic Abstraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PolySkill框架，解决LLM在动态环境中技能泛化不足的问题。通过分离技能目标与实现，提升技能复用与跨网站适应能力。属于智能代理持续学习任务。**

- **链接: [https://arxiv.org/pdf/2510.15863](https://arxiv.org/pdf/2510.15863)**

> **作者:** Simon Yu; Gang Li; Weiyan Shi; Peng Qi
>
> **备注:** 29 pages, 6 figures, 8 tables
>
> **摘要:** Large language models (LLMs) are moving beyond static uses and are now powering agents that learn continually during their interaction with external environments. For example, agents can learn reusable skills while navigating web pages or toggling new tools. However, existing methods for skill learning often create skills that are over-specialized to a single website and fail to generalize. We introduce PolySkill, a new framework that enables agents to learn generalizable and compositional skills. The core idea, inspired by polymorphism in software engineering, is to decouple a skill's abstract goal (what it accomplishes) and its concrete implementation (how it is executed). Experiments show that our method (1) improves skill reuse by 1.7x on seen websites and (2) boosts success rates by up to 9.4% on Mind2Web and 13.9% on unseen websites, while reducing steps by over 20%. (3) In self-exploration settings without specified tasks, our framework improves the quality of proposed tasks and enables agents to learn generalizable skills that work across different sites. By enabling the agent to identify and refine its own goals, the PolySkill enhances the agent's ability to learn a better curriculum, leading to the acquisition of more generalizable skills compared to baseline methods. This work provides a practical path toward building agents capable of continual learning in adaptive environments. Our findings show that separating a skill's goal from its execution is a crucial step toward developing autonomous agents that can learn and generalize across the open web continuously. Our code can be found in this https URL.
>
---
#### [replaced 053] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于可视化质量评估任务，旨在解决如何有效评价可视化美学与质量的问题。作者提出了VisJudge-Bench基准和VisJudge模型，以提升MLLM在该任务上的表现。**

- **链接: [https://arxiv.org/pdf/2510.22373](https://arxiv.org/pdf/2510.22373)**

> **作者:** Yupeng Xie; Zhiyang Zhang; Yifan Wu; Sirong Lu; Jiayi Zhang; Zhaoyang Yu; Jinlin Wang; Sirui Hong; Bang Liu; Chenglin Wu; Yuyu Luo
>
> **备注:** 62 pages, 27 figures, 8 tables. Accepted at ICLR 2026
>
> **摘要:** Visualization, a domain-specific yet widely used form of imagery, is an effective way to turn complex datasets into intuitive insights, and its value depends on whether data are faithfully represented, clearly communicated, and aesthetically designed. However, evaluating visualization quality is challenging: unlike natural images, it requires simultaneous judgment across data encoding accuracy, information expressiveness, and visual aesthetics. Although multimodal large language models (MLLMs) have shown promising performance in aesthetic assessment of natural images, no systematic benchmark exists for measuring their capabilities in evaluating visualizations. To address this, we propose VisJudge-Bench, the first comprehensive benchmark for evaluating MLLMs' performance in assessing visualization aesthetics and quality. It contains 3,090 expert-annotated samples from real-world scenarios, covering single visualizations, multiple visualizations, and dashboards across 32 chart types. Systematic testing on this benchmark reveals that even the most advanced MLLMs (such as GPT-5) still exhibit significant gaps compared to human experts in judgment, with a Mean Absolute Error (MAE) of 0.553 and a correlation with human ratings of only 0.428. To address this issue, we propose VisJudge, a model specifically designed for visualization aesthetics and quality assessment. Experimental results demonstrate that VisJudge significantly narrows the gap with human judgment, reducing the MAE to 0.421 (a 23.9% reduction) and increasing the consistency with human experts to 0.687 (a 60.5% improvement) compared to GPT-5. The benchmark is available at this https URL.
>
---
#### [replaced 054] Scaling with Collapse: Efficient and Predictable Training of LLM Families
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLM训练中的可扩展性问题，提出通过损失曲线坍缩实现高效训练。属于模型训练优化任务，解决如何预测和提升训练效率的问题。**

- **链接: [https://arxiv.org/pdf/2509.25087](https://arxiv.org/pdf/2509.25087)**

> **作者:** Shane Bergsma; Bin Claire Zhang; Nolan Dey; Shaheer Muhammad; Gurpreet Gosal; Joel Hestness
>
> **备注:** ICLR 2026
>
> **摘要:** Effective LLM training depends on predictable scaling of key quantities -- such as final loss and optimal hyperparameters -- with model and dataset size. Qiu et al. (2025) recently showed that this predictability can extend beyond scalars: whole training loss curves can *collapse* onto a universal trajectory after a simple normalization. What remains unclear is whether this phenomenon persists for LLM families trained under *practical scaling recipes*, where width, depth, learning rate, batch size, and weight decay are scaled jointly. We show that it does: loss curves collapse across scales precisely when optimization hyperparameters are set optimally for the given data budget, in accordance with recent empirical scaling laws. Collapse therefore emerges as a signature of compute-efficient training. We demonstrate two applications at scale: (1) deviation-from-collapse provides a sensitive, early diagnostic of training pathologies, and (2) predictability of collapsed curves enables early stopping in large-scale hyperparameter tuning. Finally, we train a competitive LLM family, *Celerity*, using these insights, establishing collapse as an effective tool for developing efficient LLMs.
>
---
#### [replaced 055] From Generative Modeling to Clinical Classification: A GPT-Based Architecture for EHR Notes
- **分类: cs.CL**

- **简介: 该论文属于临床文本分类任务，解决EHR文本建模难题。通过选择性微调GPT模型，减少参数量并提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.21955](https://arxiv.org/pdf/2601.21955)**

> **作者:** Fariba Afrin Irany; Sampson Akwafuo
>
> **备注:** This submission is a full-length research manuscript consisting of 37 pages and 15 figures. The paper presents a GPT-based architecture with selective fine-tuning for clinical text classification, including detailed architectural diagrams, learning curves, and evaluation figures such as ROC curves and confusion matrices
>
> **摘要:** The increasing availability of unstructured clinical narratives in electronic health records (EHRs) has created new opportunities for automated disease characterization, cohort identification, and clinical decision support. However, modeling long, domain-specific clinical text remains challenging due to limited labeled data, severe class imbalance, and the high computational cost of adapting large pretrained language models. This study presents a GPT-based architecture for clinical text classification that adapts a pretrained decoder-only Transformer using a selective fine-tuning strategy. Rather than updating all model parameters, the majority of the GPT-2 backbone is frozen, and training is restricted to the final Transformer block, the final layer normalization, and a lightweight classification head. This approach substantially reduces the number of trainable parameters while preserving the representational capacity required to model complex clinical language. The proposed method is evaluated on radiology reports from the MIMIC-IV-Note dataset using uncertainty-aware CheXpert-style labels derived directly from report text. Experiments cover multiple problem formulations, including multi-label classification of radiographic findings, binary per-label classification under different uncertainty assumptions, and aggregate disease outcome prediction. Across varying dataset sizes, the model exhibits stable convergence behavior and strong classification performance, particularly in settings dominated by non-mention and negated findings. Overall, the results indicate that selective fine-tuning of pretrained generative language models provides an efficient and effective pathway for clinical text classification, enabling scalable adaptation to real-world EHR data while significantly reducing computational complexity.
>
---
#### [replaced 056] SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SPARE框架，解决自动过程标注难题，用于提升大语言模型的多步骤推理能力。通过单次生成实现步骤对齐与准确性评估，提升奖励模型训练和强化学习效果。**

- **链接: [https://arxiv.org/pdf/2506.15498](https://arxiv.org/pdf/2506.15498)**

> **作者:** Md Imbesat Hassan Rizvi; Xiaodan Zhu; Iryna Gurevych
>
> **备注:** Accepted to AAAI 2026 (Oral)
>
> **摘要:** Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce Single-Pass Annotation with Reference-Guided Evaluation (SPARE), a novel structured framework that enables efficient per-step annotation by jointly aligning solution steps to reference solutions and determine its accuracy with explicit reasoning in single generation. We demonstrate SPARE's effectiveness across four diverse datasets spanning mathematical reasoning (GSM8K, MATH), multi-hop question answering (MuSiQue-Ans), and spatial reasoning (SpaRP), showing consistent improvements in two applications: (1) training Process Reward Models (PRMs) for ranking and aggregating multiple generations, and (2) fine-tuning models via offline reinforcement learning for greedy decoding. On ProcessBench, SPARE demonstrates data-efficient out-of-distribution generalization, using only $\sim$16% of training samples compared to human-labeled and other synthetically trained baselines. Additionally, it achieves competitive performance with MCTS-based methods while offering 2.3$\times$ speedup in terms of total token count. Manual analysis reveals complementary precision-recall characteristics with MCTS approaches, suggesting potential for ensemble methods. These results establish SPARE as a practical and scalable solution for automatic process supervision in LLM reasoning.
>
---
#### [replaced 057] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SimuHome，一个时间与环境感知的智能家庭基准，用于LLM代理测试。解决智能家庭中动态环境与任务调度问题，通过高保真模拟和600个场景评估代理性能。**

- **链接: [https://arxiv.org/pdf/2509.24282](https://arxiv.org/pdf/2509.24282)**

> **作者:** Gyuhyeon Seo; Jungwoo Yang; Junseong Pyo; Nalim Kim; Jonggeun Lee; Yohan Jo
>
> **备注:** Accepted at ICLR 2026 (Oral)
>
> **摘要:** We introduce $\textbf{SimuHome}$, a high-fidelity smart home simulator and a benchmark of 600 episodes for LLM-based smart home agents. Existing smart home benchmarks treat the home as a static system, neither simulating how device operations affect environmental variables over time nor supporting workflow scheduling of device commands. SimuHome is grounded in the Matter protocol, the industry standard that defines how real smart home devices communicate and operate. Agents interact with devices through SimuHome's APIs and observe how their actions continuously affect environmental variables such as temperature and humidity. Our benchmark covers state inquiry, implicit user intent inference, explicit device control, and workflow scheduling, each with both feasible and infeasible requests. For workflow scheduling, the simulator accelerates time so that scheduled workflows can be evaluated immediately. An evaluation of 18 agents reveals that workflow scheduling is the hardest category, with failures persisting across alternative agent frameworks and fine-tuning. These findings suggest that SimuHome's time-accelerated simulation could serve as an environment for agents to pre-validate their actions before committing them to the real world.
>
---
#### [replaced 058] Document Reconstruction Unlocks Scalable Long-Context RLVR
- **分类: cs.CL**

- **简介: 该论文属于长文本理解任务，旨在提升大语言模型的长上下文能力。通过无监督方法训练模型重建文档，无需人工标注或教师模型，有效改善模型的全局连贯性。**

- **链接: [https://arxiv.org/pdf/2602.08237](https://arxiv.org/pdf/2602.08237)**

> **作者:** Yao Xiao; Lei Wang; Yue Deng; Guanzheng Chen; Ziqi Jin; Jung-jae Kim; Xiaoli Li; Roy Ka-wei Lee; Lidong Bing
>
> **摘要:** Reinforcement Learning with Verifiable Rewards~(RLVR) has become a prominent paradigm to enhance the capabilities (i.e.\ long-context) of Large Language Models~(LLMs). However, it often relies on gold-standard answers or explicit evaluation rubrics provided by powerful teacher models or human experts, which are costly and time-consuming. In this work, we investigate unsupervised approaches to enhance the long-context capabilities of LLMs, eliminating the need for heavy human annotations or teacher models' supervision. Specifically, we first replace a few paragraphs with special placeholders in a long document. LLMs are trained through reinforcement learning to reconstruct the document by correctly identifying and sequencing missing paragraphs from a set of candidate options. This training paradigm enables the model to capture global narrative coherence, significantly boosting long-context performance. We validate the effectiveness of our method on two widely used benchmarks, RULER and LongBench~v2. While acquiring noticeable gains on RULER, it can also achieve a reasonable improvement on LongBench~v2 without any manually curated long-context QA data. Furthermore, we conduct extensive ablation studies to analyze the impact of reward design, data curation strategies, training schemes, and data scaling effects on model performance. We publicly release our code, data, and models.
>
---
#### [replaced 059] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，旨在解决大模型中的幻觉问题。通过GACD方法，利用梯度分析抑制视觉与文本的偏差，提升输出的视觉一致性。**

- **链接: [https://arxiv.org/pdf/2509.03113](https://arxiv.org/pdf/2509.03113)**

> **作者:** Shan Wang; Maying Shen; Nadine Chang; Chuong Nguyen; Hongdong Li; Jose M. Alvarez
>
> **备注:** CVPR 2026
>
> **摘要:** Multimodal large language models achieve strong performance across diverse tasks but remain prone to hallucinations, where outputs are not grounded in visual inputs. This issue can be attributed to two main biases: text-visual bias, the overreliance on prompts and prior outputs, and co-occurrence bias, spurious correlations between frequently paired objects. We propose Gradient-based Influence-Aware Constrained Decoding (GACD), an inference-based method, that addresses both biases without auxiliary models, and is readily applicable to existing models without finetuning. The core of our approach is bias estimation, which uses first-order Taylor gradients to understand the contribution of individual tokens-visual features and text tokens-to the current output. Based on this analysis, GACD mitigates hallucinations through two components: (1) suppressing spurious visual features correlated with the output objects, and (2) rebalancing cross-modal contributions by strengthening visual features relative to text. Experiments across multiple benchmarks demonstrate that GACD effectively reduces hallucinations and improves the visual grounding of MLLM outputs.
>
---
#### [replaced 060] Calibrating Verbalized Confidence with Self-Generated Distractors
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型可信度校准任务，旨在解决模型自信与实际准确率不匹配的问题。通过引入DINCO方法，提升模型置信度估计的准确性。**

- **链接: [https://arxiv.org/pdf/2509.25532](https://arxiv.org/pdf/2509.25532)**

> **作者:** Victor Wang; Elias Stengel-Eskin
>
> **备注:** ICLR 2026. Code: this https URL
>
> **摘要:** Calibrated confidence estimates are necessary for large language model (LLM) outputs to be trusted by human users. While LLMs can express their confidence in human-interpretable ways, verbalized LLM-generated confidence scores have empirically been found to be miscalibrated, reporting high confidence on instances with low accuracy and thereby harming trust and safety. We hypothesize that this overconfidence often stems from a given LLM's heightened suggestibility when faced with claims that it encodes little information about; we empirically validate this hypothesis, finding more suggestibility on lower-accuracy claims. Building on this finding, we introduce Distractor-Normalized Coherence (DINCO), which estimates and accounts for an LLM's suggestibility bias by having the model verbalize its confidence independently across several self-generated distractors (i.e. alternative claims), and normalizes by the total verbalized confidence. To further improve calibration, we leverage generator-validator disagreement, augmenting normalized validator confidence with a consistency-based estimate of generator confidence. Here, we frame the popular approach of self-consistency as leveraging coherence across sampled generations, and normalized verbalized confidence as leveraging coherence across validations on incompatible claims, allowing us to integrate these complementary dimensions of coherence into DINCO. Moreover, our analysis shows that DINCO provides less saturated -- and therefore more usable -- confidence estimates, and that further sampling alone cannot close the gap between DINCO and baselines, with DINCO at 10 inference calls outperforming self-consistency at 100.
>
---
#### [replaced 061] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules
- **分类: cs.AI; cs.CL; cs.LG; q-bio.QM**

- **简介: 该论文提出mCLM，解决生成功能性且可合成分子的问题，通过模块化分子语言提升药物发现效率。**

- **链接: [https://arxiv.org/pdf/2505.12565](https://arxiv.org/pdf/2505.12565)**

> **作者:** Carl Edwards; Chi Han; Gawon Lee; Thao Nguyen; Sara Szymkuć; Chetan Kumar Prasad; Bowen Jin; Jiawei Han; Ying Diao; Ge Liu; Hao Peng; Bartosz A. Grzybowski; Martin D. Burke; Heng Ji
>
> **备注:** Accepted to ICLR 2026 (Oral). Code: this https URL Data and Model: this https URL
>
> **摘要:** Despite their ability to understand chemical knowledge, large language models (LLMs) remain limited in their capacity to propose novel molecules with desired functions (e.g., drug-like properties). In addition, the molecules that LLMs propose can often be challenging to make, and are almost never compatible with automated synthesis approaches. To better enable the discovery of functional small molecules, LLMs need to learn a new molecular language that is more effective in predicting properties and inherently synced with automated synthesis technology. Current molecule LLMs are limited by representing molecules based on atoms. In this paper, we argue that just like tokenizing texts into meaning-bearing (sub-)word tokens instead of characters, molecules should be tokenized at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model that comprises a bilingual language model that understands both natural language descriptions of functions and molecular blocks. mCLM front-loads synthesizability considerations while improving the predicted functions of molecules in a principled manner. Experiments on FDA-approved drugs showed that mCLM is capable of significantly improving chemical functions. mCLM, with only 3B parameters, also achieves improvements in synthetic accessibility relative to 7 other leading generative AI methods including GPT-5. When tested on 122 out-of-distribution medicines using only building blocks/tokens that are compatible with automated modular synthesis, mCLM outperforms all baselines in property scores and synthetic accessibility. mCLM can also reason on multiple functions and iteratively self-improve to rescue drug candidates that failed late in clinical trials ("fallen angels").
>
---
#### [replaced 062] Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于时间敏感事实问答任务，旨在解决现有基准评估不足的问题。通过构建时序数据库生成TSQA对，并引入时间准确度指标，提升评估的全面性与自动化水平。**

- **链接: [https://arxiv.org/pdf/2508.02045](https://arxiv.org/pdf/2508.02045)**

> **作者:** Soyeon Kim; Jindong Wang; Xing Xie; Steven Euijong Whang
>
> **备注:** Published in Proceedings of the 14th International Conference on Learning Representations (ICLR), 2026. Code and data are publicly available at: this https URL
>
> **摘要:** Facts change over time, making it essential for Large Language Models (LLMs) to handle time-sensitive factual knowledge accurately and reliably. Although factual Time-Sensitive Question-Answering (TSQA) tasks have been widely developed, existing benchmarks often face manual bottlenecks that limit scalable and comprehensive TSQA evaluation. To address this issue, we propose TDBench, a new benchmark that systematically constructs TSQA pairs by harnessing temporal databases and database techniques, such as temporal functional dependencies, temporal SQL, and temporal joins. We also introduce a new evaluation metric called time accuracy, which assesses the validity of time references in model explanations alongside traditional answer accuracy for a more fine-grained TSQA evaluation. Extensive experiments on contemporary LLMs show how TDBench enables scalable and comprehensive TSQA evaluation while reducing the reliance on human labor, complementing current TSQA evaluation approaches that largely center on Wikipedia/Wikidata by enabling LLM evaluation on application-specific data.
>
---
#### [replaced 063] I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨LLM如何通过预测下一个词学习可解释概念。研究解决LLM表示机制问题，提出理论模型验证其与潜在概念的关联。**

- **链接: [https://arxiv.org/pdf/2503.08980](https://arxiv.org/pdf/2503.08980)**

> **作者:** Yuhang Liu; Dong Gong; Yichao Cai; Erdun Gao; Zhen Zhang; Biwei Huang; Mingming Gong; Anton van den Hengel; Javen Qinfeng Shi
>
> **摘要:** Recent empirical evidence shows that LLM representations encode human-interpretable concepts. Nevertheless, the mechanisms by which these representations emerge remain largely unexplored. To shed further light on this, we introduce a novel generative model that generates tokens on the basis of such concepts formulated as latent discrete variables. Under mild conditions, even when the mapping from the latent space to the observed space is non-invertible, we establish rigorous identifiability result: the representations learned by LLMs through next-token prediction can be approximately modeled as the logarithm of the posterior probabilities of these latent discrete concepts given input context, up to an linear transformation. This theoretical finding: 1) provides evidence that LLMs capture essential underlying generative factors, 2) offers a unified and principled perspective for understanding the linear representation hypothesis, and 3) motivates a theoretically grounded approach for evaluating sparse autoencoders. Empirically, we validate our theoretical results through evaluations on both simulation data and the Pythia, Llama, and DeepSeek model families.
>
---
#### [replaced 064] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AgentMath，解决大语言模型在复杂数学问题上的计算效率低和准确性不足的问题，通过集成语言模型与代码解释器，提升数学推理能力。**

- **链接: [https://arxiv.org/pdf/2512.20745](https://arxiv.org/pdf/2512.20745)**

> **作者:** Haipeng Luo; Huawen Feng; Qingfeng Sun; Can Xu; Kai Zheng; Yufei Wang; Tao Yang; Han Hu; Yansong Tang
>
> **备注:** This paper has been accepted to ICLR 2026
>
> **摘要:** Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in reasoning tasks with long cot. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool invocation. The evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, surpassing OpenAI-o3-mini and Claude-Opus-4.0-Thinking while remaining competitive with OpenAI-o3, Gemini-2.5-Pro, and this http URL results validate the effectiveness of our approach and pave the way for building scalable mathematical reasoning agents.
>
---
#### [replaced 065] German General Social Survey Personas: A Survey-Derived Persona Prompt Collection for Population-Aligned LLM Studies
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理与社会科学研究的交叉任务，旨在解决LLM模拟人类视角时缺乏代表性人格数据的问题。作者构建了基于德国社会调查的GGSS Personas数据集，用于提升模型生成内容与人口特征的对齐度。**

- **链接: [https://arxiv.org/pdf/2511.21722](https://arxiv.org/pdf/2511.21722)**

> **作者:** Jens Rupprecht; Leon Fröhling; Claudia Wagner; Markus Strohmaier
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** The use of Large Language Models (LLMs) for simulating human perspectives via persona prompting is gaining traction in computational social science. However, well-curated, empirically grounded persona collections remain scarce, limiting the accuracy and representativeness of such simulations. Here, we introduce the German General Social Survey Personas (GGSS Personas) collection, a comprehensive and representative persona prompt collection built from the German General Social Survey (ALLBUS). The GGSS Personas and their persona prompts are designed to be easily plugged into prompts for all types of LLMs and tasks, steering models to generate responses aligned with the underlying German population. We evaluate GGSS Personas by prompting various LLMs to simulate survey response distributions across diverse topics, demonstrating that GGSS Personas-guided LLMs outperform state-of-the-art classifiers, particularly under data scarcity. Furthermore, we analyze how the representativity and attribute selection within persona prompts affect alignment with population responses. Our findings suggest that GGSS Personas provide a potentially valuable resource for research on LLM-based social simulations that enables more systematic explorations of population-aligned persona prompting in NLP and social science research.
>
---
#### [replaced 066] PoLi-RL: A Point-to-List Reinforcement Learning Framework for Conditional Semantic Textual Similarity
- **分类: cs.CL**

- **简介: 该论文提出PoLi-RL框架，解决条件语义文本相似度（C-STS）任务中的排名优化问题，通过分阶段强化学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.04080](https://arxiv.org/pdf/2510.04080)**

> **作者:** Zixin Song; Bowen Zhang; Qian-Wen Zhang; Di Yin; Xing Sun; Chunping Li
>
> **摘要:** Conditional Semantic Textual Similarity (C-STS) measures the semantic proximity between text segments under a specific condition, thereby overcoming the ambiguity inherent in traditional STS. However, existing methods are largely confined to discriminative models, failing to fully leverage recent breakthroughs in the NLP community involving Large Language Models (LLMs) and Reinforcement Learning (RL). RL is a particularly well-suited paradigm for this task, as it can directly optimize the non-differentiable Spearman ranking metric and guide the reasoning process required by C-STS. Nevertheless, we find that naively applying listwise RL fails to produce meaningful improvements, as the model struggles with complex, coarse-grained reward signals, leading to optimization difficulties. To address this challenge, we introduce PoLi-RL, a novel Point-to-List Reinforcement Learning framework. PoLi-RL employs a two-stage curriculum: it first trains the model with a simple pointwise reward to establish fundamental scoring capabilities, then transitions to a hybrid reward that combines pointwise, pairwise, and listwise objectives to refine the model's ability to discern subtle semantic distinctions. Crucially, we propose an innovative Parallel Slice Ranking Reward (PSRR) mechanism that computes ranking rewards in parallel slices, where each slice consists of completions with the same index from different samples. This provides a precise, differentiated learning signal for each individual completion, enabling granular credit assignment and effective optimization. On the official C-STS benchmark, PoLi-RL achieves a Spearman correlation coefficient of 48.18, establishing a new SOTA for the cross-encoder architecture. As the first work to successfully apply RL to C-STS, our study introduces a powerful paradigm for aligning LLMs for complex, ranking-based conditional judgment tasks.
>
---
#### [replaced 067] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出TTOM框架，解决视频生成中的组合性问题。通过测试时优化和记忆机制，提升文本-视频对齐效果，增强模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.07940](https://arxiv.org/pdf/2510.07940)**

> **作者:** Leigang Qu; Ziyang Wang; Na Zheng; Wenjie Wang; Liqiang Nie; Tat-Seng Chua
>
> **备注:** ICLR 2026 Camera-ready. Project page: this https URL
>
> **摘要:** Video Foundation Models (VFMs) exhibit remarkable visual generation performance, but struggle in compositional scenarios (e.g., motion, numeracy, and spatial relation). In this work, we introduce Test-Time Optimization and Memorization (TTOM), a training-free framework that aligns VFM outputs with spatiotemporal layouts during inference for better text-image alignment. Rather than direct intervention to latents or attention per-sample in existing work, we integrate and optimize new parameters guided by a general layout-attention objective. Furthermore, we formulate video generation within a streaming setting, and maintain historical optimization contexts with a parametric memory mechanism that supports flexible operations, such as insert, read, update, and delete. Notably, we found that TTOM disentangles compositional world knowledge, showing powerful transferability and generalization. Experimental results on the T2V-CompBench and Vbench benchmarks establish TTOM as an effective, practical, scalable, and efficient framework to achieve cross-modal alignment for compositional video generation on the fly.
>
---
#### [replaced 068] ExGRPO: Learning to Reason from Experience
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型推理能力提升中的计算效率与稳定性问题。提出ExGRPO框架，通过经验管理提升性能。**

- **链接: [https://arxiv.org/pdf/2510.02245](https://arxiv.org/pdf/2510.02245)**

> **作者:** Runzhe Zhan; Yafu Li; Zhi Wang; Xiaoye Qu; Dongrui Liu; Jing Shao; Derek F. Wong; Yu Cheng
>
> **备注:** ICLR 2026 Camera Ready version
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) is an emerging paradigm for improving the reasoning ability of large language models. However, standard on-policy training discards rollout experiences after a single update, leading to computational inefficiency and instability. While prior work on RL has highlighted the benefits of reusing past experience, the role of experience characteristics in shaping learning dynamics of large reasoning models remains underexplored. In this paper, we are the first to investigate what makes a reasoning experience valuable and identify rollout correctness and entropy as effective indicators of experience value. Based on these insights, we propose ExGRPO (Experiential Group Relative Policy Optimization), a framework that organizes and prioritizes valuable experiences, and employs a mixed-policy objective to balance exploration with experience exploitation. Experiments on five backbone models (1.5B-8B parameters) show that ExGRPO consistently improves reasoning performance on mathematical/general benchmarks, with an average gain of +3.5/7.6 points over on-policy RLVR. Moreover, ExGRPO stabilizes training on both stronger and weaker models where on-policy methods fail. These results highlight principled experience management as a key ingredient for efficient and scalable RLVR.
>
---
#### [replaced 069] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在解决现有模型在复杂空间推理上的不足。提出OmniSpatial基准，涵盖四大类50子类，构建8.4K问答对，并探索两种提升方法。**

- **链接: [https://arxiv.org/pdf/2506.03135](https://arxiv.org/pdf/2506.03135)**

> **作者:** Mengdi Jia; Zekun Qi; Shaochen Zhang; Wenyao Zhang; Xinqiang Yu; Jiawei He; He Wang; Li Yi
>
> **备注:** ICLR 2026
>
> **摘要:** Spatial reasoning is a key aspect of cognitive psychology and remains a bottleneck for current vision-language models (VLMs). While extensive research has aimed to evaluate or improve VLMs' understanding of basic spatial relations, such as distinguishing left from right, near from far, and object counting, these tasks cover only the most elementary layer of spatial reasoning and are largely approaching saturation in the latest reasoning models. In this work, we introduce OmniSpatial, a comprehensive and challenging benchmark for spatial reasoning, grounded in cognitive psychology. OmniSpatial covers four major categories: dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking, with 50 fine-grained subcategories. Through careful manual annotation, we construct over 8.4K question-answer pairs. Extensive experiments show that both open- and closed-source VLMs exhibit significant limitations in comprehensive spatial reasoning. We also explore two strategies-PointGraph (explicit scene graph cues) and SpatialCoT (novel-view chain-of-thought)-to bolster spatial reasoning.
>
---
#### [replaced 070] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于多智能体系统研究，旨在解决MAS设计效率低和效果不确定的问题。提出MAS-Orchestra框架与MASBENCH基准，提升多智能体协作性能与理解。**

- **链接: [https://arxiv.org/pdf/2601.14652](https://arxiv.org/pdf/2601.14652)**

> **作者:** Zixuan Ke; Yifei Ming; Austin Xu; Ryan Chin; Xuan-Phi Nguyen; Prathyusha Jwalapuram; Jiayu Wang; Semih Yavuz; Caiming Xiong; Shafiq Joty
>
> **备注:** Preprint; Work in Progress
>
> **摘要:** While multi-agent systems (MAS) promise elevated intelligence through coordination of agents, current approaches to automatic MAS design under-deliver. Such shortcomings stem from two key factors: (1) methodological complexity - agent orchestration is performed using sequential, code-level execution that limits global system-level holistic reasoning and scales poorly with agent complexity - and (2) efficacy uncertainty - MAS are deployed without understanding if there are tangible benefits compared to single-agent systems (SAS). We propose MASOrchestra, a training-time framework that formulates MAS orchestration as a function-calling reinforcement learning problem with holistic orchestration, generating an entire MAS at once. In MAS-Orchestra, complex, goal-oriented subagents are abstracted as callable functions, enabling global reasoning over system structure while hiding internal execution details. To rigorously study when and why MAS are beneficial, we introduce MASBENCH, a controlled benchmark that characterizes tasks along five axes: Depth, Horizon, Breadth, Parallel, and Robustness. Our analysis reveals that MAS gains depend critically on task structure, verification protocols, and the capabilities of both orchestrator and subagents, rather than holding universally. Guided by these insights, MAS-Orchestra achieves consistent improvements on public benchmarks including mathematical reasoning, multi-hop QA, and search-based QA, while achieving more than 10x efficiency over strong baselines. Together, MAS-Orchestra and MASBENCH enable better training and understanding of MAS in the pursuit of multi-agent intelligence.
>
---
#### [replaced 071] DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router
- **分类: cs.CL**

- **简介: 该论文提出DeepSieve，解决LLM在知识密集型查询中的信息筛选问题。通过结构化分解查询并路由至合适知识源，提升推理深度与精度。属于知识增强生成任务。**

- **链接: [https://arxiv.org/pdf/2507.22050](https://arxiv.org/pdf/2507.22050)**

> **作者:** Minghao Guo; Qingcheng Zeng; Xujiang Zhao; Yanchi Liu; Wenchao Yu; Mengnan Du; Haifeng Chen; Wei Cheng
>
> **备注:** Accepted by EACL Findings 2026
>
> **摘要:** Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. Our codes are available at this https URL.
>
---
#### [replaced 072] Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音合成检测任务，旨在识别深度伪造语音中的合成词语。通过微调Whisper模型，利用下一个词预测实现高效检测。**

- **链接: [https://arxiv.org/pdf/2602.22658](https://arxiv.org/pdf/2602.22658)**

> **作者:** Hoan My Tran; Xin Wang; Wanying Ge; Xuechen Liu; Junichi Yamagishi
>
> **备注:** Submitted to Interspeech. To quote: Interspeech no longer enforces an anonymity period for submissions. While uploading a version online is permitted, your official submission to Interspeech must not contain any author-identifying information. ... a note indicating that the paper was submitted for review to (or, eventually, accepted at) Interspeech should be included in the posting
>
> **摘要:** Deepfake speech utterances can be forged by replacing one or more words in a bona fide utterance with semantically different words synthesized with speech-generative models. While a dedicated synthetic word detector could be developed, we developed a cost-effective method that fine-tunes a pre-trained Whisper model to detect synthetic words while transcribing the input utterance via next-token prediction. We further investigate using partially vocoded utterances as the fine-tuning data, thus reducing the cost of data collection. Our experiments demonstrate that, on in-domain test data, the fine-tuned Whisper yields low synthetic-word detection error rates and transcription error rates. On out-of-domain test data with synthetic words produced with unseen speech-generative models, the fine-tuned Whisper remains on par with a dedicated ResNet-based detection model; however, the overall performance degradation calls for strategies to improve its generalization capability.
>
---
#### [replaced 073] When Large Language Models are More PersuasiveThan Incentivized Humans, and Why
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM与人类在说服力上的差异。通过实验比较LLM与激励人类的说服效果，发现LLM在不同情境下表现不同，且说服力随互动次数下降。**

- **链接: [https://arxiv.org/pdf/2505.09662](https://arxiv.org/pdf/2505.09662)**

> **作者:** Philipp Schoenegger; Francesco Salvi; Jiacheng Liu; Xiaoli Nan; Ramit Debnath; Barbara Fasolo; Evelina Leivada; Gabriel Recchia; Fritz Günther; Ali Zarifhonarvar; Joe Kwon; Zahoor Ul Islam; Marco Dehnert; Daryl Y. H. Lee; Madeline G. Reinecke; David G. Kamper; Mert Kobaş; Adam Sandford; Jonas Kgomo; Luke Hewitt; Shreya Kapoor; Kerem Oktar; Eyup Engin Kucuk; Bo Feng; Cameron R. Jones; Izzy Gainsburg; Sebastian Olschewski; Nora Heinzelmann; Francisco Cruz; Ben M. Tappin; Tao Ma; Peter S. Park; Rayan Onyonka; Arthur Hjorth; Peter Slattery; Qingcheng Zeng; Lennart Finke; Igor Grossmann; Alessandro Salatiello; Ezra Karger
>
> **摘要:** Large Language Models (LLMs) have been shown to be highly persuasive, but when and why they outperform humans is still an open question. We compare the persuasiveness of two LLMs (Claude 3.5 Sonnet and DeepSeek v3) against humans who had incentives to persuade, using an interactive, real-time conversational setting. We demonstrate that LLMs persuasive superiority is context-dependent: it depends on whether the persuasion attempt is truthful (towards the right answer) or deceptive (towards the wrong answer) and on the LLM model, and wanes over repeated interactions (unlike human persuasiveness). In our first large-scale experiment, humans vs LLMs (Claude 3.5 Sonnet) interacted with other humans who were completing an online quiz for a reward, attempting to persuade them toward a given (either correct or incorrect) answer. Claude was more persuasive than incentivized human persuaders both in truthful and deceptive contexts and it significantly increased accuracy if persuasion was truthful, but decreased it if persuasion was deceptive. In a follow-up experiment with Deepseek v3, we replicated the findings about accuracy but found greater LLM persuasiveness only if the persuasion was deceptive. Linguistic analyses of the persuaders texts suggest that these effects may be due to LLMs expressing higher conviction than humans.
>
---
#### [replaced 074] Large Language Models are Algorithmically Blind
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨大语言模型在算法推理上的局限性。研究发现模型无法准确预测算法性能，表现出算法盲视现象。**

- **链接: [https://arxiv.org/pdf/2602.21947](https://arxiv.org/pdf/2602.21947)**

> **作者:** Sohan Venkatesh; Ashish Mahendran Kurapath; Tejas Melkote
>
> **备注:** 19 pages, 8 figures, 15 tables
>
> **摘要:** Large language models (LLMs) demonstrate remarkable breadth of knowledge, yet their ability to reason about computational processes remains poorly understood. Closing this gap matters for practitioners who rely on LLMs to guide algorithm selection and deployment. We address this limitation using causal discovery as a testbed and evaluate eight frontier LLMs against ground truth derived from large-scale algorithm executions and find systematic, near-total failure. Models produce ranges far wider than true confidence intervals yet still fail to contain the true algorithmic mean in the majority of instances; most perform worse than random guessing and the marginal above-random performance of the best model is most consistent with benchmark memorization rather than principled reasoning. We term this failure algorithmic blindness and argue it reflects a fundamental gap between declarative knowledge about algorithms and calibrated procedural prediction.
>
---
#### [replaced 075] Intrinsic Entropy of Context Length Scaling in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究长上下文对语言模型的影响。通过引入“内在熵”概念，分析上下文长度对建模的影响，并验证理论假设。**

- **链接: [https://arxiv.org/pdf/2502.01481](https://arxiv.org/pdf/2502.01481)**

> **作者:** Jingzhe Shi; Qinwei Ma; Hongyi Liu; Hang Zhao; Jeng-Neng Hwang; Lei Li
>
> **备注:** 36 pages, 18 figures, 2 tables
>
> **摘要:** Long Context Language Models have drawn great attention in the past few years. There has been work discussing the impact of long context on Language Model performance: some find that long irrelevant context could harm performance, while some experimentally summarize loss reduction by relevant long context as Scaling Laws. This calls for a more thorough understanding of how long context impacts Language Modeling. In this work, we (1) propose to use `Intrinsic Entropy' for explaining the impact of context length on language modeling; and (2) conduct experiments on natural language and synthetic data, validating our proposed theoretical assumptions and deductions. Our theoretical framework can provide practical insights such as establishing that training dataset size dictates an optimal context length and bounds context length scaling for certain cases. We hope our work may inspire new long context Language Models, as well as future work studying the physics of Language Models.
>
---
#### [replaced 076] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能安全领域，解决奖励黑客问题。提出TRACE方法，通过测量推理努力检测隐式奖励黑客，提升监控效果。**

- **链接: [https://arxiv.org/pdf/2510.01367](https://arxiv.org/pdf/2510.01367)**

> **作者:** Xinpeng Wang; Nitish Joshi; Barbara Plank; Rico Angell; He He
>
> **备注:** ICLR 2026 Oral Presentation
>
> **摘要:** Reward hacking, where a reasoning model exploits loopholes in a reward function to achieve high rewards without solving the intended task, poses a significant threat. This behavior may be explicit, i.e. verbalized in the model's chain-of-thought (CoT), or implicit, where the CoT appears benign thus bypasses CoT monitors. To detect implicit reward hacking, we propose TRACE (Truncated Reasoning AUC Evaluation). Our key observation is that hacking occurs when exploiting the loophole is easier than solving the actual task. This means that the model is using less 'effort' than required to achieve high reward. TRACE quantifies effort by measuring how early a model's reasoning becomes sufficient to obtain the reward. We progressively truncate a model's CoT at various lengths, force the model to answer, and estimate the expected reward at each cutoff. A hacking model, which takes a shortcut, will achieve a high expected reward with only a small fraction of its CoT, yielding a large area under the accuracy-vs-length curve. TRACE achieves over 65% gains over our strongest 72B CoT monitor in math reasoning, and over 30% gains over a 32B monitor in coding. We further show that TRACE can discover unknown loopholes during training. Overall, TRACE offers a scalable unsupervised approach for oversight where current monitoring methods prove ineffective.
>
---
#### [replaced 077] InstructPro: Natural Language Guided Ligand-Binding Protein Design
- **分类: cs.LG; cs.CE; cs.CL**

- **简介: 该论文属于蛋白质设计任务，旨在解决数据稀缺问题。通过自然语言指导生成结合配体的蛋白质，提出InstructPro模型及基准数据集，显著提升设计性能。**

- **链接: [https://arxiv.org/pdf/2506.09332](https://arxiv.org/pdf/2506.09332)**

> **作者:** Zhenqiao Song; Ramith Hettiarachchi; Chuan Li; Jianwen Xie; Lei Li
>
> **摘要:** The de novo design of ligand-binding proteins with tailored functions is essential for advancing biotechnology and molecular medicine, yet existing AI approaches are limited by scarce protein-ligand complex data. To circumvent this data bottleneck, we leverage the abundant natural language descriptions characterizing protein-ligand interactions. Here, we introduce InstructPro, a family of generative models that design proteins following the guidance of natural language instructions and ligand formulas. InstructPro produces protein sequences consistent with specified function descriptions and ligand targets. To enable training and evaluation, we develop InstructProBench, a large-scale dataset of 9.6 million (function description, ligand, protein) triples. We train two model variants -- InstructPro-1B and InstructPro-3B -- that substantially outperform strong baselines. InstructPro-1B achieves an AlphaFold3 ipTM of 0.918 and a binding affinity of -8.764 on seen ligands, while maintaining robust performance in a zero-shot setting with scores of 0.869 and -6.713, respectively. These results are accompanied by novelty scores of 70.1% and 68.8%, underscoring the model's ability to generalize beyond the training set. Furthermore, the model yields a superior binding free energy of -20.9 kcal/mol and an average of 5.82 intermolecular hydrogen bonds, validating its proficiency in designing high-affinity ligand-binding proteins. Notably, scaling to InstructPro-3B further improves the zero-shot ipTM to 0.882, binding affinity to -6.797, and binding free energy to -25.8 kcal/mol, demonstrating clear performance gains associated with increased model capacity. These findings highlight the power of natural language-guided generative models to mitigate the data bottlenecks in traditional structure-based methods, significantly broadening the scope of de novo protein design.
>
---
#### [replaced 078] TurkicNLP: An NLP Toolkit for Turkic Languages
- **分类: cs.CL**

- **简介: 该论文提出TurkicNLP，一个用于突厥语族的自然语言处理工具包，解决多语言、多文字体系下NLP工具不统一的问题，整合了多种语言处理功能。**

- **链接: [https://arxiv.org/pdf/2602.19174](https://arxiv.org/pdf/2602.19174)**

> **作者:** Sherzod Hakimov
>
> **备注:** The toolkit is available here: this https URL
>
> **摘要:** Natural language processing for the Turkic language family, spoken by over 200 million people across Eurasia, remains fragmented, with most languages lacking unified tooling and resources. We present TurkicNLP, an open-source Python library providing a single, consistent NLP pipeline for Turkic languages across four script families: Latin, Cyrillic, Perso-Arabic, and Old Turkic Runic. The library covers tokenization, morphological analysis, part-of-speech tagging, dependency parsing, named entity recognition, bidirectional script transliteration, cross-lingual sentence embeddings, and machine translation through one language-agnostic API. A modular multi-backend architecture integrates rule-based finite-state transducers and neural models transparently, with automatic script detection and routing between script variants. Outputs follow the CoNLL-U standard for full interoperability and extension. Code and documentation are hosted at this https URL .
>
---
#### [replaced 079] Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长文本问答任务，解决长上下文信息丢失问题。提出ReMemR1方法，通过记忆检索与多级奖励提升模型对长文本的推理能力。**

- **链接: [https://arxiv.org/pdf/2509.23040](https://arxiv.org/pdf/2509.23040)**

> **作者:** Yaorui Shi; Yuxin Chen; Siyuan Wang; Sihang Li; Hengxing Cai; Qi Gu; Xiang Wang; An Zhang
>
> **摘要:** Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory buffer that is dynamically updated via a linear document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from pruning of latent evidence, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, which integrates the mechanism of memory retrieval into the memory update process, enabling the agent to selectively callback historical memories for non-linear reasoning. To further strengthen training, we propose a multi-level reward design, which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support complex multi-hop reasoning. Extensive experiments demonstrate that ReMemR1 significantly outperforms state-of-the-art baselines on long-context question answering while incurring negligible computational overhead, validating its ability to trade marginal cost for robust long-context reasoning.
>
---
#### [replaced 080] Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于文本检测任务，旨在识别LLM生成的文本。通过自适应距离学习方法提升检测效果，实验显示优于现有方法。**

- **链接: [https://arxiv.org/pdf/2601.21895](https://arxiv.org/pdf/2601.21895)**

> **作者:** Hongyi Zhou; Jin Zhu; Kai Ye; Ying Yang; Erhan Xu; Chengchun Shi
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Modern large language models (LLMs) such as GPT, Claude, and Gemini have transformed the way we learn, work, and communicate. Yet, their ability to produce highly human-like text raises serious concerns about misinformation and academic integrity, making it an urgent need for reliable algorithms to detect LLM-generated content. In this paper, we start by presenting a geometric approach to demystify rewrite-based detection algorithms, revealing their underlying rationale and demonstrating their generalization ability. Building on this insight, we introduce a novel rewrite-based detection algorithm that adaptively learns the distance between the original and rewritten text. Theoretically, we demonstrate that employing an adaptively learned distance function is more effective for detection than using a fixed distance. Empirically, we conduct extensive experiments with over 100 settings, and find that our approach demonstrates superior performance over baseline algorithms in the majority of scenarios. In particular, it achieves relative improvements from 54.3% to 75.4% over the strongest baseline across different target LLMs (e.g., GPT, Claude, and Gemini). A python implementation of our proposal is publicly available at this https URL.
>
---
#### [replaced 081] Enhancing Hallucination Detection through Noise Injection
- **分类: cs.CL; eess.SY**

- **简介: 该论文属于 hallucination 检测任务，旨在提升大语言模型生成内容的准确性。通过引入噪声扰动参数，增强模型不确定性检测，从而更有效地识别幻觉输出。**

- **链接: [https://arxiv.org/pdf/2502.03799](https://arxiv.org/pdf/2502.03799)**

> **作者:** Litian Liu; Reza Pourreza; Sunny Panchal; Apratim Bhattacharyya; Yubing Jian; Yao Qin; Roland Memisevic
>
> **备注:** ICLR 2026 main conference paper
>
> **摘要:** Large Language Models (LLMs) are prone to generating plausible yet incorrect responses, known as hallucinations. Effectively detecting hallucinations is therefore crucial for the safe deployment of LLMs. Recent research has linked hallucinations to model uncertainty, suggesting that hallucinations can be detected by measuring dispersion over answer distributions obtained from multiple samples drawn from a model. While drawing from the distribution over tokens defined by the model is a natural way to obtain samples, in this work, we argue that it is suboptimal for the purpose of detecting hallucinations. We show that detection can be improved significantly by taking into account model uncertainty in the Bayesian sense. To this end, we propose a very simple, training-free approach based on perturbing an appropriate subset of model parameters, or equivalently hidden unit activations, during sampling. We demonstrate that our approach significantly improves inference-time hallucination detection over standard sampling across diverse datasets, model architectures, and uncertainty metrics.
>
---
#### [replaced 082] SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings
- **分类: cs.CL**

- **简介: 该论文属于知识编辑任务，旨在解决模型纠错时引入的非目标扰动问题。提出SUIT方法，通过子空间约束更新关键特征，提升知识保留并减少扰动。**

- **链接: [https://arxiv.org/pdf/2509.24502](https://arxiv.org/pdf/2509.24502)**

> **作者:** Haewon Park; Sangwoo Kim; Yohan Jo
>
> **备注:** 31 pages, 13 figures, 17 tables
>
> **摘要:** Knowledge editing aims to efficiently correct factual errors in language models. Widely used locate-then-edit methods update an MLP layer by adjusting its weights to change the mapping between the layer's input vector (key) and output vector (value), thereby editing the model's knowledge. As this update is driven by key and value vectors, obtaining these vectors without careful constraints causes significant model perturbations beyond the targeted edit, a common issue in many prior knowledge editing methods. To address this, we propose Subspace Knowledge Edit (SUIT), which computes key and value vectors only within the subspace of critical features relevant to the edit. Our empirical results on LLaMA3, GPT-J, and Qwen2.5 models show that SUIT dramatically improves knowledge preservation over strong baselines while maintaining high editing performance. These results support the claim that SUIT successfully identifies the critical subspace for the edit. Beyond quantitative gains, our analyses show that SUIT reduces unintended perturbations in hidden states while confining updates to directions that are more effective for editing. Taken together, these findings establish edit-critical subspace identification as a key principle for reliable, low-perturbation knowledge editing. Our code is available at this https URL.
>
---
#### [replaced 083] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RefTool，解决LLM在缺乏预定义工具时的推理问题。通过参考外部资料自动创建工具，提升知识密集型任务的推理能力。**

- **链接: [https://arxiv.org/pdf/2505.21413](https://arxiv.org/pdf/2505.21413)**

> **作者:** Xiao Liu; Da Yin; Zirui Wu; Yansong Feng
>
> **备注:** Accepted by ICLR 2026. Code is available at this https URL
>
> **摘要:** Large Language Models (LLMs) can enhance their reasoning capabilities by using external tools. However, many tasks lack predefined tools. Prior works have explored instructing LLMs to generate tools on their own, but such approaches depend heavily on internal knowledge and struggle when tasks fall outside the model's knowledge scope. To address this limitation, we propose RefTool, a reference-guided framework for automatic tool creation that leverages external materials, such as textbooks and knowledge snippets. RefTool consists of two modules: (1) tool creation, where LLMs generate executable tools from reference content, validate them using illustrative examples, and organize them hierarchically into a toolbox; and (2) tool utilization, where LLMs navigate the toolbox structure to select and apply the appropriate tools to solve problems. Experiments on causality, physics, and chemistry benchmarks demonstrate that RefTool outperforms existing tool-creation and domain-specific reasoning methods by 12.3% on average accuracy, while being cost-efficient and broadly generalizable to non-scientific tasks, e.g., extremely low-resource language translation. Analyses reveal that grounding tool creation in references produces accurate and faithful tools, and that the hierarchical structure facilitates effective tool selection. RefTool enables LLMs to overcome internal knowledge limitations, advancing generalizable reasoning in knowledge-intensive domains.
>
---
#### [replaced 084] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像编辑任务，旨在解决开放源代码模型缺乏可靠奖励模型的问题。通过构建EditReward，提升指令引导图像编辑的性能。**

- **链接: [https://arxiv.org/pdf/2509.26346](https://arxiv.org/pdf/2509.26346)**

> **作者:** Keming Wu; Sicong Jiang; Max Ku; Ping Nie; Minghao Liu; Wenhu Chen
>
> **备注:** Accepted by ICLR 2026. Project Page: this https URL
>
> **摘要:** Recently, we have witnessed great progress in image editing with natural language instructions. Several closed-source models like GPT-Image-1, Seedream, and Google-Nano-Banana have shown highly promising progress. However, the open-source models are still lagging. The main bottleneck is the lack of a reliable reward model to scale up high-quality synthetic training data. To address this critical bottleneck, we built EditReward, trained with our new large-scale human preference dataset, meticulously annotated by trained experts following a rigorous protocol containing over 200K preference pairs. EditReward demonstrates superior alignment with human preferences in instruction-guided image editing tasks. Experiments show that EditReward achieves state-of-the-art human correlation on established benchmarks such as GenAI-Bench, AURORA-Bench, ImagenHub, and our new EditReward-Bench, outperforming a wide range of VLM-as-judge models. Furthermore, we use EditReward to select a high-quality subset from the existing noisy ShareGPT-4o-Image dataset. We train Step1X-Edit on the selected subset, which shows significant improvement over training on the full set. This demonstrates EditReward's ability to serve as a reward model to scale up high-quality training data for image editing. Furthermore, its strong alignment suggests potential for advanced applications like reinforcement learning-based post-training and test-time scaling of image editing models. EditReward with its training dataset will be released to help the community build more high-quality image editing training datasets.
>
---
#### [replaced 085] Using ChatGPT for Data Science Analyses
- **分类: cs.LG; cs.CL; stat.CO**

- **简介: 论文探讨ChatGPT在数据科学中的应用，评估其在数据探索、可视化及建模任务中的潜力，旨在解决如何有效利用AI工具提升数据分析效率的问题。**

- **链接: [https://arxiv.org/pdf/2404.08480](https://arxiv.org/pdf/2404.08480)**

> **作者:** Ozan Evkaya; Miguel de Carvalho
>
> **备注:** 19 pages with figures and appendix
>
> **摘要:** As a result of recent advancements in generative AI, the field of data science is prone to various changes. The way practitioners construct their data science workflows is now irreversibly shaped by recent advancements, particularly by tools like OpenAI's Data Analysis plugin. While it offers powerful support as a quantitative co-pilot, its limitations demand careful consideration in empirical analysis. This paper assesses the potential of ChatGPT for data science analyses, illustrating its capabilities for data exploration and visualization, as well as for commonly used supervised and unsupervised modeling tasks. While we focus here on how the Data Analysis plugin can serve as co-pilot for Data Science workflows, its broader potential for automation is implicit throughout.
>
---
#### [replaced 086] Diversity-Enhanced Reasoning for Subjective Questions
- **分类: cs.CL**

- **简介: 该论文针对主观推理任务，解决生成答案多样性不足的问题。提出MultiRole-R1框架，通过引入角色多样性和token级多样性提升模型表现。**

- **链接: [https://arxiv.org/pdf/2507.20187](https://arxiv.org/pdf/2507.20187)**

> **作者:** Yumeng Wang; Zhiyuan Fan; Jiayu Liu; Jen-tse Huang; Yi R. Fung
>
> **摘要:** Large Reasoning Models (LRMs) with long chain-of-thought capabilities, optimized via reinforcement learning with verifiable rewards (RLVR), excel at objective reasoning tasks like mathematical problem solving and code generation. However, RLVR is known for degrading generation diversity, which causes LRMs to fall short on subjective reasoning that has multiple answers depending on different role perspectives. While recent studies recognize the importance of diversity-enhanced training in objective reasoning, limited attention has been given to subjective tasks. In this paper, we find that subjective reasoning can be improved by introducing perspective diversity and token-level diversity, with the former one providing a coherent scaffolding anchored to a real-world stakeholder group and the latter one broadening the answer search space. We propose MultiRole-R1, a diversity-enhanced training framework featuring an unsupervised data construction pipeline that synthesizes reasoning chains incorporating various role perspectives. It also employs reinforcement learning via Group Relative Policy Optimization with reward shaping, taking diversity as a reward signal in addition to verifiable reward. Training on subjective tasks solely, MultiRole-R1 increases the in-domain and out-of-domain accuracy by 14.1% and 7.64%, and even enhances the performance on advanced math reasoning such as AIME 2024. We further show that diversity is a more consistent indicator of accuracy than reasoning length.
>
---
#### [replaced 087] MASA: Rethinking the Representational Bottleneck in LoRA with Multi-A Shared Adaptation
- **分类: cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA的表示瓶颈问题。通过引入多A共享结构，提升模型适应能力。**

- **链接: [https://arxiv.org/pdf/2510.06005](https://arxiv.org/pdf/2510.06005)**

> **作者:** Qin Dong; Yuntian Tang; Heming Jia; Yunhang Shen; Bohan Jia; Wenxuan Huang; Lianyue Zhang; Jiao Xie; Shaohui Lin; Rongrong Ji
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a dominant method in Parameter-Efficient Fine-Tuning (PEFT) for large language models, which augments the transformer layer with one down-projection $A$ and one up-projection $B$. However, LoRA's reliance on a single down-projection matrix ($A$) creates a representational bottleneck, as this solitary feature extractor is inherently insufficient for capturing the diverse signals required by complex tasks. This motivates our architectural shift to focus on enriching the feature adaptation to improve the downstream task adaptation ability. We propose MASA (Multi-$A$ Shared Adaptation), an architecture that implements a multi-$A$, single-$B$ structure where the multi-$A$ expert ensemble is asymmetrically shared across layers to ensure parameter efficiency. In MASA, these specialized experts capture diverse features, which are then integrated by a single, layer-specific $B$-matrix. The effectiveness and versatility of our method are validated through a comprehensive suite of experiments spanning multi-domain generalization, single-domain specialization, and multi-task reasoning. For example, on the MMLU benchmark, MASA achieves an average accuracy of 59.62%, outperforming the standard LoRA by 1.08 points (a relative improvement of 1.84%) with comparable learnable parameters of 0.52%.
>
---
#### [replaced 088] ScholarEval: Research Idea Evaluation Grounded in Literature
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ScholarEval，用于评估研究创意的合理性与贡献度，解决AI生成研究想法的评价问题。工作包括构建数据集并验证框架有效性。**

- **链接: [https://arxiv.org/pdf/2510.16234](https://arxiv.org/pdf/2510.16234)**

> **作者:** Hanane Nour Moussa; Patrick Queiroz Da Silva; Daniel Adu-Ampratwum; Alyson East; Zitong Lu; Nikki Puccetti; Mingyi Xue; Huan Sun; Bodhisattwa Prasad Majumder; Sachin Kumar
>
> **摘要:** As AI tools become increasingly common for research ideation, robust evaluation is critical to ensure the validity and usefulness of generated ideas. We introduce ScholarEval, a retrieval augmented evaluation framework that assesses research ideas based on two fundamental criteria: soundness - the empirical validity of proposed methods based on existing literature, and contribution - the degree of advancement made by the idea across different dimensions relative to prior research. To evaluate ScholarEval, we introduce ScholarIdeas, the first expert-annotated dataset of multi-domain research ideas and reviews, comprised of 117 ideas across four disciplines: artificial intelligence, neuroscience, biochemistry, and ecology. Our evaluation shows that ScholarEval achieves significantly higher coverage of points mentioned in the human expert annotated rubrics in ScholarIdeas compared to all baselines. Furthermore, ScholarEval is consistently preferred over our strongest baseline o4-mini-deep-research, a reasoning and search-enabled agentic system by OpenAI, in terms of evaluation actionability, depth, and evidence support. Our large-scale user study also shows that ScholarEval significantly outperforms deep research in literature engagement, idea refinement, and usefulness. We openly release our code, dataset, and ScholarEval tool for the community to use and build on.
>
---
#### [replaced 089] Large Language Models in Bioinformatics: A Survey
- **分类: cs.CL; q-bio.GN**

- **简介: 该论文属于生物信息学领域，探讨大语言模型在基因组分析、蛋白质功能预测等任务中的应用，解决数据稀缺与计算复杂性等问题，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2503.04490](https://arxiv.org/pdf/2503.04490)**

> **作者:** Zhenyu Wang; Zikang Wang; Jiyue Jiang; Pengan Chen; Xiangyu Shi; Yu Li
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Large Language Models (LLMs) are revolutionizing bioinformatics, enabling advanced analysis of DNA, RNA, proteins, and single-cell data. This survey provides a systematic review of recent advancements, focusing on genomic sequence modeling, RNA structure prediction, protein function inference, and single-cell transcriptomics. Meanwhile, we also discuss several key challenges, including data scarcity, computational complexity, and cross-omics integration, and explore future directions such as multimodal learning, hybrid AI models, and clinical applications. By offering a comprehensive perspective, this paper underscores the transformative potential of LLMs in driving innovations in bioinformatics and precision medicine.
>
---
#### [replaced 090] Silence the Judge: Reinforcement Learning with Self-Verifier via Latent Geometric Clustering
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决LLM训练中依赖外部验证器导致的高成本和低效率问题。通过引入基于潜在空间几何的内在奖励机制，提升训练效率与模型性能。**

- **链接: [https://arxiv.org/pdf/2601.08427](https://arxiv.org/pdf/2601.08427)**

> **作者:** Nonghai Zhang; Weitao Ma; Zhanyu Ma; Jun Xu; Jiuchong Gao; Jinghua Hao; Renqing He; Jingwen Xu
>
> **摘要:** Group Relative Policy Optimization (GRPO) significantly enhances the reasoning performance of Large Language Models (LLMs). However, this success heavily relies on expensive external verifiers or human rules. Such dependency not only leads to significant computational costs and training latency, but also yields sparse rewards that hinder optimization efficiency. To address these challenges, we propose Latent-GRPO, a framework that derives intrinsic rewards directly from latent space geometry. Crucially, our empirical analysis reveals a compelling geometric property: terminal token representations of correct reasoning trajectories form dense clusters with high intra-class similarity, whereas incorrect trajectories remain scattered as outliers. In light of this discovery, we introduce the Iterative Robust Centroid Estimation (IRCE) algorithm, which generates dense, continuous rewards by mitigating magnitude fluctuations via spherical projection and estimating a robust ``truth centroid'' through iterative aggregation. Experimental results on multiple datasets show that our method maintains model performance while achieving a training speedup of over 2x compared to baselines. Furthermore, extensive results demonstrate strong generalization ability and robustness. The code will be released soon.
>
---
#### [replaced 091] A Foundational Individual Mobility Prediction Model based on Open-Source Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于个体出行预测任务，旨在解决传统方法在跨数据源和分布外场景下泛化能力不足的问题。提出MoBLLM模型，利用轻量级开源大语言模型和参数高效微调技术，提升预测准确性和迁移能力。**

- **链接: [https://arxiv.org/pdf/2503.16553](https://arxiv.org/pdf/2503.16553)**

> **作者:** Zhenlin Qin; Leizhen Wang; Yancheng Ling; Francisco Camara Pereira; Zhenliang Ma
>
> **摘要:** Individual mobility prediction plays a key role in urban transport, enabling personalized service recommendations and effective travel management. It is widely modeled by data-driven methods such as machine learning, deep learning, as well as classical econometric methods to capture key features of mobility patterns. However, such methods are hindered in promoting further transferability and robustness due to limited capacity to learn mobility patterns from different data sources, predict in out-of-distribution settings (a.k.a ``zero-shot"). To address this challenge, this paper introduces MoBLLM, a foundational model for individual mobility prediction that aims to learn a shared and transferable representation of mobility behavior across heterogeneous data sources. Based on a lightweight open-source large language model (LLM), MoBLLM employs Parameter-Efficient Fine-Tuning (PEFT) techniques to create a cost-effective training pipeline, avoiding the need for large-scale GPU clusters while maintaining strong performance. We conduct extensive experiments on six real-world mobility datasets to evaluate its accuracy, robustness, and transferability across varying temporal scales (years), spatial contexts (cities), and situational conditions (e.g., disruptions and interventions). MoBLLM achieves the best F1 score and accuracy across all datasets compared with state-of-the-art deep learning models and shows better transferability and cost efficiency than commercial LLMs. Further experiments reveal its robustness under network changes, policy interventions, special events, and incidents. These results indicate that MoBLLM provides a generalizable modeling foundation for individual mobility behavior, enabling more reliable and adaptive personalized information services for transportation management.
>
---
#### [replaced 092] Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs
- **分类: cs.CL**

- **简介: 该论文研究LLMs能否在记忆数据基础上进行泛化，属于模型泛化任务。通过“记忆-泛化”框架，验证模型能重新解释记忆数据，解决记忆与理解的矛盾。**

- **链接: [https://arxiv.org/pdf/2507.21914](https://arxiv.org/pdf/2507.21914)**

> **作者:** Qinyuan Wu; Soumi Das; Mahsa Amani; Bishwamittra Ghosh; Mohammad Aflah Khan; Krishna P. Gummadi; Muhammad Bilal Zafar
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Rote learning is a memorization technique based on repetition. Many researchers argue that rote learning hinders generalization because it encourages verbatim memorization rather than deeper understanding. This concern extends even to factual knowledge, which inevitably requires a certain degree of memorization. In this work, we challenge this view and demonstrate that large language models (LLMs) can, in fact, generalize over rote memorized data. We introduce a two-phase "memorize-then-generalize" framework, where the model first rote memorizes factual subject-object associations using a synthetic semantically meaningless key token and then learns to generalize by fine-tuning on a small set of semantically meaningful prompts. Extensive experiments over 8 LLMs show that the models can reinterpret rote memorized data through the semantically meaningful prompts, as evidenced by the emergence of structured, semantically aligned latent representations between the key token and the semantically meaningful prompts. This surprising finding opens the door to both effective and efficient knowledge injection as well as possible risks of repurposing the memorized data for malicious usage.
>
---
#### [replaced 093] Uni-cot: Towards Unified Chain-of-Thought Reasoning Across Text and Vision
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Uni-CoT，解决多模态推理问题，通过统一模型实现文本与视觉的连贯推理，提升任务规划与执行效率。**

- **链接: [https://arxiv.org/pdf/2508.05606](https://arxiv.org/pdf/2508.05606)**

> **作者:** Luozheng Qin; Jia Gong; Yuqing Sun; Tianjiao Li; Mengping Yang; Xiaomeng Yang; Chao Qu; Zhiyu Tan; Hao Li
>
> **备注:** Accepted by ICLR 2026, Project Page: this https URL
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been widely adopted to enhance Large Language Models (LLMs) by decomposing complex tasks into simpler, sequential subtasks. However, extending CoT to vision-language reasoning tasks remains challenging, as it often requires interpreting transitions of visual states to support reasoning. Existing methods often struggle with this due to limited capacity of modeling visual state transitions or incoherent visual trajectories caused by fragmented architectures. To overcome these limitations, we propose Uni-CoT, a Unified Chain-of-Thought framework that enables coherent and grounded multimodal reasoning within a single unified model. The key idea is to leverage a model capable of both image understanding and generation to reason over visual content and model evolving visual states. However, empowering a unified model to achieve that is non-trivial, given the high computational cost and the burden of training. To address this, Uni-CoT introduces a novel two-level reasoning paradigm: A Macro-Level CoT for high-level task planning and A Micro-Level CoT for subtask execution. This design significantly reduces the computational overhead. Furthermore, we introduce a structured training paradigm that combines interleaved image-text supervision for macro-level CoT with multi-task objectives for micro-level CoT. Together, these innovations allow Uni-CoT to perform scalable and coherent multi-modal reasoning. Furthermore, thanks to our design, all experiments can be efficiently completed using only 8 A100 GPUs with 80GB VRAM each. Experimental results on reasoning-driven image generation benchmark (WISE) and editing benchmarks (RISE and KRIS) indicates that Uni-CoT demonstrates SOTA performance and strong generalization, establishing Uni-CoT as a promising solution for multi-modal reasoning. Project Page and Code: this https URL
>
---
#### [replaced 094] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于文本水印任务，旨在解决语义级水印的鲁棒性和分布畸变问题。提出PMark方法，通过代理函数实现无损且强鲁棒的水印嵌入。**

- **链接: [https://arxiv.org/pdf/2509.21057](https://arxiv.org/pdf/2509.21057)**

> **作者:** Jiahao Huo; Shuliang Liu; Bin Wang; Junyan Zhang; Yibo Yan; Aiwei Liu; Xuming Hu; Mingxun Zhou
>
> **备注:** ICLR 2026 Poster
>
> **摘要:** Semantic-level watermarking (SWM) for large language models (LLMs) enhances watermarking robustness against text modifications and paraphrasing attacks by treating the sentence as the fundamental unit. However, existing methods still lack strong theoretical guarantees of robustness, and reject-sampling-based generation often introduces significant distribution distortions compared with unwatermarked outputs. In this work, we introduce a new theoretical framework on SWM through the concept of proxy functions (PFs) $\unicode{x2013}$ functions that map sentences to scalar values. Building on this framework, we propose PMark, a simple yet powerful SWM method that estimates the PF median for the next sentence dynamically through sampling while enforcing multiple PF constraints (which we call channels) to strengthen watermark evidence. Equipped with solid theoretical guarantees, PMark achieves the desired distortion-free property and improves the robustness against paraphrasing-style attacks. We also provide an empirically optimized version that further removes the requirement for dynamical median estimation for better sampling efficiency. Experimental results show that PMark consistently outperforms existing SWM baselines in both text quality and robustness, offering a more effective paradigm for detecting machine-generated text. Our code will be released at [this URL](this https URL).
>
---
#### [replaced 095] Reliable Fine-Grained Evaluation of Natural Language Math Proofs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学证明生成评估任务，旨在解决LLM生成数学证明难以精确评价的问题。提出ProofGrader评估器，实现细粒度评分，提升证明质量评估效果。**

- **链接: [https://arxiv.org/pdf/2510.13888](https://arxiv.org/pdf/2510.13888)**

> **作者:** Wenjie Ma; Andrei Cojocaru; Neel Kolhe; Bradley Louie; Robin Said Sharif; Haihan Zhang; Vincent Zhuang; Matei Zaharia; Sewon Min
>
> **备注:** 40 pages, 7 figures, 15 tables
>
> **摘要:** Recent advances in large language models (LLMs) for mathematical reasoning have largely focused on tasks with easily verifiable final answers while generating and verifying natural language math proofs remains an open challenge. We identify the absence of a reliable, fine-grained evaluator for LLM-generated math proofs as a critical gap. To address this, we propose a systematic methodology for developing and validating evaluators that assign fine-grained scores on a 0-7 scale to model-generated math proofs. To enable this study, we introduce ProofBench, the first expert-annotated dataset of fine-grained proof ratings, spanning 145 problems from six major math competitions (USAMO, IMO, Putnam, etc) and 435 LLM-generated solutions from Gemini-2.5-Pro, o3, and DeepSeek-R1. Using ProofBench as a testbed, we systematically explore the evaluator design space across key axes: the backbone model, input context, instructions and evaluation workflow. Our analysis delivers ProofGrader, an evaluator that combines a strong reasoning backbone LM, rich context from reference solutions and marking schemes, and a simple ensembling method; it achieves a low Mean Absolute Error (MAE) of 0.926 against expert scores, significantly outperforming naive baselines. Finally, we demonstrate its practical utility in a best-of-$n$ selection task: at $n=16$, ProofGrader achieves an average score of 4.14/7, closing 78\% of the gap between a naive binary evaluator (2.48) and the human oracle (4.62), highlighting its potential to advance downstream proof generation.
>
---
#### [replaced 096] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking
- **分类: cs.CL**

- **简介: 该论文提出AStar，解决多模态推理任务中的复杂视觉推理问题。通过引入“思维卡片”提升模型推理效率，无需额外训练即可增强推理能力。**

- **链接: [https://arxiv.org/pdf/2502.02339](https://arxiv.org/pdf/2502.02339)**

> **作者:** Jinyang Wu; Mingkuan Feng; Guocheng Zhai; Shuai Zhang; Zheng Lian; Fangrui Lv; Pengpeng Shao; Ruihan Jin; Zhengqi Wen; Jianhua Tao
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Multimodal large language models excel across diverse domains but struggle with complex visual reasoning tasks. To enhance their reasoning capabilities, current approaches typically rely on explicit search or post-training techniques. However, search-based methods suffer from computational inefficiency due to extensive solution space exploration, while post-training methods demand substantial data, computational resources, and often exhibit training instability. To address these challenges, we propose \textbf{AStar}, a training-free, \textbf{A}utomatic \textbf{S}tructured \textbf{t}hinking paradigm for multimod\textbf{a}l \textbf{r}easoning. Specifically, we introduce novel ``thought cards'', a lightweight library of high-level reasoning patterns abstracted from prior samples. For each test problem, AStar adaptively retrieves the optimal thought cards and seamlessly integrates these external explicit guidelines with the model's internal implicit reasoning capabilities. Compared to previous methods, AStar eliminates computationally expensive explicit search and avoids additional complex post-training processes, enabling a more efficient reasoning approach. Extensive experiments demonstrate that our framework achieves 53.9\% accuracy on MathVerse (surpassing GPT-4o's 50.2\%) and 32.7\% on MathVision (outperforming GPT-4o's 30.4\%). Further analysis reveals the remarkable transferability of our method: thought cards generated from mathematical reasoning can also be applied to other reasoning tasks, even benefiting general visual perception and understanding. AStar serves as a plug-and-play test-time inference method, compatible with other post-training techniques, providing an important complement to existing multimodal reasoning approaches.
>
---
#### [replaced 097] Residual Connections and the Causal Shift: Uncovering a Structural Misalignment in Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，针对Transformer模型中残差连接与因果掩码导致的输入输出对齐问题，提出通过残差衰减缓解结构错位，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.14760](https://arxiv.org/pdf/2602.14760)**

> **作者:** Jonathan Lys; Vincent Gripon; Bastien Pasdeloup; Axel Marmoret; Lukas Mauch; Fabien Cardinaux; Ghouthi Boukli Hacene
>
> **摘要:** Large Language Models (LLMs) are trained with next-token prediction, implemented in autoregressive Transformers via causal masking for parallelism. This creates a subtle misalignment: residual connections tie activations to the current token, while supervision targets the next token, potentially propagating mismatched information if the current token is not the most informative for prediction. In this work, we empirically localize this input-output alignment shift in pretrained LLMs, using decoding trajectories over tied embedding spaces and similarity-based metrics. Our experiments reveal that the hidden token representations switch from input alignment to output alignment deep within the network. Motivated by this observation, we propose a lightweight residual-path mitigation based on residual attenuation, implemented either as a fixed-layer intervention or as a learnable gating mechanism. Experiments on multiple benchmarks show that these strategies alleviate the representation misalignment and yield improvements, providing an efficient and general architectural enhancement for autoregressive Transformers.
>
---
#### [replaced 098] Can SAEs reveal and mitigate racial biases of LLMs in healthcare?
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI伦理任务，旨在检测和缓解LLMs在医疗中对种族的偏见。研究使用SAEs识别模型中与种族相关的潜在表示，并探索其对模型输出的影响。**

- **链接: [https://arxiv.org/pdf/2511.00177](https://arxiv.org/pdf/2511.00177)**

> **作者:** Hiba Ahsan; Byron C. Wallace
>
> **备注:** camera-ready ICLR 2026
>
> **摘要:** LLMs are increasingly being used in healthcare. This promises to free physicians from drudgery, enabling better care to be delivered at scale. But the use of LLMs in this space also brings risks; for example, such models may worsen existing biases. How can we spot when LLMs are (spuriously) relying on patient race to inform predictions? In this work we assess the degree to which Sparse Autoencoders (SAEs) can reveal (and control) associations the model has made between race and stigmatizing concepts. We first identify SAE latents in Gemma-2 models which appear to correlate with Black individuals. We find that this latent activates on reasonable input sequences (e.g., "African American") but also problematic words like "incarceration". We then show that we can use this latent to steer models to generate outputs about Black patients, and further that this can induce problematic associations in model outputs as a result. For example, activating the Black latent increases the risk assigned to the probability that a patient will become "belligerent". We evaluate the degree to which such steering via latents might be useful for mitigating bias. We find that this offers improvements in simple settings, but is less successful for more realistic and complex clinical tasks. Overall, our results suggest that: SAEs may offer a useful tool in clinical applications of LLMs to identify problematic reliance on demographics but mitigating bias via SAE steering appears to be of marginal utility for realistic tasks.
>
---
#### [replaced 099] Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices
- **分类: cs.DC; cs.AI; cs.CL; eess.SP**

- **简介: 该论文属于边缘计算任务，旨在解决大型多模态模型在电池供电小设备上的高效推理问题。通过软硬件协同设计，将模型分解为模块并分配到最佳加速器，提升效率并降低能耗。**

- **链接: [https://arxiv.org/pdf/2510.05109](https://arxiv.org/pdf/2510.05109)**

> **作者:** Yilong Li; Shuai Zhang; Yijing Zeng; Hao Zhang; Xinmiao Xiong; Jingyu Liu; Pan Hu; Suman Banerjee
>
> **摘要:** Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly 20.8 hours.
>
---
#### [replaced 100] DiSRouter: Distributed Self-Routing for LLM Selections
- **分类: cs.CL**

- **简介: 该论文属于模型路由任务，旨在解决多LLM间高效查询分配问题。提出DiSRouter，通过分布式自路由机制提升灵活性与性能。**

- **链接: [https://arxiv.org/pdf/2510.19208](https://arxiv.org/pdf/2510.19208)**

> **作者:** Hang Zheng; Hongshen Xu; Yongkai Lin; Shuai Fan; Lu Chen; Kai Yu
>
> **摘要:** The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness, its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems.
>
---
#### [replaced 101] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理优化任务，解决草案策略与解码策略不一致的问题。提出GTO方法，通过优化草案树奖励和分组训练，提升解码速度与效率。**

- **链接: [https://arxiv.org/pdf/2509.22134](https://arxiv.org/pdf/2509.22134)**

> **作者:** Shijing Hu; Jingyang Li; Zhihui Lu; Pan Zhou
>
> **摘要:** Speculative decoding accelerates large language model (LLM) inference by letting a lightweight draft model propose multiple tokens that the target model verifies in parallel. Yet existing training objectives optimize only a single greedy draft path, while decoding follows a tree policy that re-ranks and verifies multiple branches. This draft policy misalignment limits achievable speedups. We introduce Group Tree Optimization (GTO), which aligns training with the decoding-time tree policy through two components: (i) Draft Tree Reward, a sampling-free objective equal to the expected acceptance length of the draft tree under the target model, directly measuring decoding performance; (ii) Group-based Draft Policy Training, a stable optimization scheme that contrasts trees from the current and a frozen reference draft model, forming debiased group-standardized advantages and applying a PPO-style surrogate along the longest accepted sequence for robust updates. We further prove that increasing our Draft Tree Reward provably improves acceptance length and speedup. Across dialogue (MT-Bench), code (HumanEval), and math (GSM8K), and multiple LLMs (e.g., LLaMA-3.1-8B, LLaMA-3.3-70B, Vicuna-1.3-13B, DeepSeek-R1-Distill-LLaMA-8B, Qwen3-8B), GTO increases acceptance length by (7.4%) and yields an additional (7.7%) speedup over prior state-of-the-art EAGLE-3. By bridging draft policy misalignment, GTO offers a practical, general solution for efficient LLM inference. Code and draft models are available at this https URL.
>
---
#### [replaced 102] EnsembleLink: Accurate Record Linkage Without Training Data
- **分类: cs.CL**

- **简介: 论文提出EnsembleLink，解决无训练数据下的记录链接问题。利用预训练语言模型提升匹配准确性，适用于多种数据类型，无需外部API。**

- **链接: [https://arxiv.org/pdf/2601.21138](https://arxiv.org/pdf/2601.21138)**

> **作者:** Noah Dasanaike
>
> **摘要:** Record linkage, the process of matching records that refer to the same entity across datasets, is essential to empirical social science but remains methodologically underdeveloped. Researchers treat it as a preprocessing step, applying ad hoc rules without quantifying the uncertainty that linkage errors introduce into downstream analyses. Existing methods either achieve low accuracy or require substantial labeled training data. I present EnsembleLink, a method that achieves high accuracy without any training labels. EnsembleLink leverages pre-trained language models that have learned semantic relationships (e.g., that "South Ozone Park" is a neighborhood in "New York City" or that "Lutte ouvriere" refers to the Trotskyist "Workers' Struggle" party) from large text corpora. On benchmarks spanning city names, person names, organizations, multilingual political parties, and bibliographic records, EnsembleLink matches or exceeds methods requiring extensive labeling. The method runs locally on open-source models, requiring no external API calls, and completes typical linkage tasks in minutes.
>
---
#### [replaced 103] GenRecal: Generation after Recalibration from Large to Small Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉-语言模型压缩任务，旨在解决大模型向小模型知识迁移的难题。提出GenRecal框架，通过特征对齐实现跨类型VLM的有效知识蒸馏。**

- **链接: [https://arxiv.org/pdf/2506.15681](https://arxiv.org/pdf/2506.15681)**

> **作者:** Byung-Kwan Lee; Ryo Hachiuma; Yong Man Ro; Yu-Chiang Frank Wang; Yueh-Hua Wu
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advancements in vision-language models (VLMs) have leveraged large language models (LLMs) to achieve performance on par with closed-source systems like GPT-4V. However, deploying these models in real-world scenarios, particularly on resource-constrained devices, remains challenging due to their substantial computational demands. This has spurred interest in distilling knowledge from large VLMs into smaller, more efficient counterparts. A key challenge arises here from the diversity of VLM architectures, which are built on different LLMs and employ varying token types-differing in vocabulary size, token splits, and token index ordering. To address this challenge of limitation to a specific VLM type, we present Generation after Recalibration (GenRecal), a general-purpose distillation framework for VLMs. GenRecal incorporates a Recalibrator that aligns and adapts feature representations between heterogeneous VLMs, enabling effective knowledge transfer across different types of VLMs. Through extensive experiments on multiple challenging benchmarks, we demonstrate that GenRecal significantly improves baseline performances, eventually outperforming large-scale open- and closed-source VLMs.
>
---
#### [replaced 104] Language steering in latent space to mitigate unintended code-switching
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言大模型中的意外代码切换问题。通过潜空间语言引导方法，提升语言一致性并减少代码切换。**

- **链接: [https://arxiv.org/pdf/2510.13849](https://arxiv.org/pdf/2510.13849)**

> **作者:** Andrey Goncharov; Nikolai Kondusov; Alexey Zaytsev
>
> **摘要:** Multilingual Large Language Models (LLMs) often exhibit unintended code-switching, reducing reliability in downstream tasks. We propose latent-space language steering, a lightweight inference-time method that identifies language directions via PCA on parallel translations and steers token embeddings along these axes to control language identity. Our approach mitigates code-switching while preserving semantics with negligible computational overhead and requires only minimal parallel data for calibration. Empirically, we achieve 95-99\% language classification accuracy using a single principal component and reduce next-token distributional divergence by up to 55\% across multiple language pairs on Qwen2.5 and Llama-3.2 models. Generation-based evaluation on Llama-3.2 further demonstrates 63--99\% reduction in Code-Switching Index across four language pairs ($p < 0.001$). We further analyze the layer-wise evolution of language representations, revealing that language identity concentrates in final layers with near-perfect linear separability.
>
---
#### [replaced 105] ProfVLM: A Lightweight Video-Language Model for Multi-View Proficiency Estimation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ProfVLM，用于多视角技能评估，解决传统方法无法建模推理过程的问题，通过生成式视觉语言模型同时预测水平并生成自然语言反馈。**

- **链接: [https://arxiv.org/pdf/2509.26278](https://arxiv.org/pdf/2509.26278)**

> **作者:** Edoardo Bianchi; Jacopo Staiano; Antonio Liotta
>
> **摘要:** Most existing approaches formulate action quality assessment and skill proficiency estimation as discriminative prediction tasks, typically producing discrete labels or scores without explicitly modeling the reasoning process underlying the assessment. We instead reformulate the problem as generative vision-language modeling, introducing ProfVLM, a parameter-efficient vision-language model that jointly predicts proficiency levels and generates expert-like natural language feedback from multi-view videos. ProfVLM leverages conditional language generation to provide actionable insights along with quantitative evaluation scores. Central to our method is an AttentiveGatedProjector that dynamically fuses and projects multi-view egocentric and exocentric features from a frozen TimeSformer backbone into a language model fine-tuned for feedback generation. Trained on EgoExo4D with expert commentaries, ProfVLM surpasses state-of-the-art methods while using up to 20x fewer parameters and reducing training time by up to 60% compared to existing classification-based methods. By providing natural language critiques aligned with performance levels, this work shows that generative vision-language modeling offers a powerful and efficient paradigm shift for interpretable action quality assessment.
>
---
#### [replaced 106] InnoGym: Benchmarking the Innovation Potential of AI Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出InnoGym，一个评估AI代理创新潜力的基准和框架。旨在解决现有基准仅关注正确性而忽视方法多样性的不足。通过性能增益和新颖性两个指标，评估AI在工程与科学任务中的创新能力。**

- **链接: [https://arxiv.org/pdf/2512.01822](https://arxiv.org/pdf/2512.01822)**

> **作者:** Jintian Zhang; Kewei Xu; Jingsheng Zheng; Zhuoyun Yu; Yuqi Zhu; Yujie Luo; Lanning Wei; Shuofei Qiao; Lun Du; Da Zheng; Shumin Deng; Huajun Chen; Ningyu Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** LLMs and Agents have achieved impressive progress in code generation, mathematical reasoning, and scientific discovery. However, existing benchmarks primarily measure correctness, overlooking the diversity of methods behind solutions. True innovation depends not only on producing correct answers but also on the originality of the approach. We present InnoGym, the first benchmark and framework designed to systematically evaluate the innovation potential of AI agents. InnoGym introduces two complementary metrics: performance gain, which measures improvement over the best-known solutions, and novelty, which captures methodological differences from prior approaches. The benchmark includes 18 carefully curated tasks from real-world engineering and scientific domains, each standardized through resource filtering, evaluator validation, and solution collection. In addition, we provide iGym, a unified execution environment for reproducible and long-horizon evaluations. Extensive experiments show that while some agents produce novel approaches, their lack of robustness limits performance gains. These results highlight a key gap between creativity and effectiveness, underscoring the need for benchmarks that evaluate both.
>
---
#### [replaced 107] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本生成任务，旨在解决大模型生成超长文本时质量下降的问题。通过强化学习方法，无需依赖标注数据，提升生成文本的长度与质量。**

- **链接: [https://arxiv.org/pdf/2506.18841](https://arxiv.org/pdf/2506.18841)**

> **作者:** Yuhao Wu; Yushi Bai; Zhiqiang Hu; Roy Ka-Wei Lee; Juanzi Li
>
> **备注:** ICLR 2026 Oral
>
> **摘要:** Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL
>
---
#### [replaced 108] What Scales in Cross-Entropy Scaling Law?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决大语言模型中交叉熵缩放定律失效的问题。通过分解交叉熵为三部分，发现误差熵遵循更稳定的幂律，揭示了原定律在大规模模型中的局限性。**

- **链接: [https://arxiv.org/pdf/2510.04067](https://arxiv.org/pdf/2510.04067)**

> **作者:** Junxi Yan; Zixi Wei; Qingyao Ai; Yiqun Liu; Jingtao Zhan
>
> **摘要:** The cross-entropy scaling law has long served as a key tool for guiding the development of large language models. It shows that cross-entropy loss decreases in a predictable power-law rate as the model size increases. However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models. In this paper, we hypothesize that the root cause lies in the fact that cross-entropy itself does not truly scale; instead, only one of its hidden components does. To investigate this, we introduce a novel decomposition of cross-entropy into three parts: Error-Entropy, Self-Alignment, and Confidence. We show both theoretically and empirically that this decomposition precisely captures the training dynamics and optimization objectives. Through extensive experiments on multiple datasets and 32 models spanning five orders of magnitude in size, we find that only error-entropy follows a robust power-law scaling, while the other two terms remain largely invariant. Moreover, error-entropy constitutes the dominant share of cross-entropy in small models but diminishes in proportion as models grow larger. This explains why the cross-entropy scaling law appears accurate at small scales but fails at very large ones. Our findings establish the error-entropy scaling law as a more accurate description of model behavior. We believe it will have wide applications in the training, understanding, and future development of large language models.
>
---
#### [replaced 109] TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音系统评估任务，旨在解决主观与客观评价不一致的问题。提出TTSDS2指标，并提供数据集和基准以更准确评估合成语音质量。**

- **链接: [https://arxiv.org/pdf/2506.19441](https://arxiv.org/pdf/2506.19441)**

> **作者:** Christoph Minixhofer; Ondrej Klejch; Peter Bell
>
> **摘要:** Evaluation of Text to Speech (TTS) systems is challenging and resource-intensive. Subjective metrics such as Mean Opinion Score (MOS) are not easily comparable between works. Objective metrics are frequently used, but rarely validated against subjective ones. Both kinds of metrics are challenged by recent TTS systems capable of producing synthetic speech indistinguishable from real speech. In this work, we introduce Text to Speech Distribution Score 2 (TTSDS2), a more robust and improved version of TTSDS. Across a range of domains and languages, it is the only one out of 16 compared metrics to correlate with a Spearman correlation above 0.50 for every domain and subjective score evaluated. We also release a range of resources for evaluating synthetic speech close to real speech: A dataset with over 11,000 subjective opinion score ratings; a pipeline for continually recreating a multilingual test dataset to avoid data leakage; and a continually updated benchmark for TTS in 14 languages.
>
---
#### [replaced 110] A Survey of Query Optimization in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于查询优化任务，旨在提升大语言模型的检索与生成效果。通过分析查询优化技术，提出生命周期框架和复杂度分类，解决查询质量影响系统性能的问题。**

- **链接: [https://arxiv.org/pdf/2412.17558](https://arxiv.org/pdf/2412.17558)**

> **作者:** Mingyang Song; Mao Zheng
>
> **备注:** Ongoing Work
>
> **摘要:** Query Optimization (QO) has become essential for enhancing Large Language Model (LLM) effectiveness, particularly in Retrieval-Augmented Generation (RAG) systems where query quality directly determines retrieval and response performance. This survey provides a systematic analysis of query optimization techniques with three contributions. \textit{First}, we introduce the \textbf{Query Optimization Lifecycle (QOL) Framework}, a five-phase pipeline covering Intent Recognition, Query Transformation, Retrieval Execution, Evidence Integration, and Response Synthesis. \textit{Second}, we propose a \textbf{Query Complexity Taxonomy} that classifies queries along two dimensions: evidence type (explicit vs.\ implicit) and evidence quantity (single vs.\ multiple), establishing principled mappings to optimization strategies. \textit{Third}, we analyze four atomic operations: \textbf{Query Expansion}, \textbf{Query Decomposition}, \textbf{Query Disambiguation}, and \textbf{Query Abstraction}, covering over 90 representative methods. We further examine evaluation methodologies, identify gaps in benchmarks, and discuss open challenges including process reward models, efficiency optimization, and multi-modal query handling. This survey offers both a structured foundation for research and actionable guidance for practitioners.
>
---
#### [replaced 111] Elo-Evolve: A Co-evolutionary Framework for Language Model Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，解决数据稀缺与训练不稳定问题。提出Elo-Evolve框架，通过动态对抗竞争实现模型优化。**

- **链接: [https://arxiv.org/pdf/2602.13575](https://arxiv.org/pdf/2602.13575)**

> **作者:** Jing Zhao; Ting Zhen; Junwei Bao; Hongfei Jiang; Yang Song
>
> **摘要:** Current alignment methods for Large Language Models (LLMs) rely on compressing vast amounts of human preference data into static, absolute reward functions, leading to data scarcity, noise sensitivity, and training instability. We introduce Elo-Evolve, a co-evolutionary framework that redefines alignment as dynamic multi-agent competition within an adaptive opponent pool. Our approach makes two key innovations: (1) eliminating Bradley-Terry model dependencies by learning directly from binary win/loss outcomes in pairwise competitions, and (2) implementing Elo-orchestrated opponent selection that provides automatic curriculum learning through temperature-controlled sampling. We ground our approach in PAC learning theory, demonstrating that pairwise comparison achieves superior sample complexity and empirically validate a 4.5x noise reduction compared to absolute scoring approaches. Experimentally, we train a Qwen2.5-7B model using our framework with opponents including Qwen2.5-14B, Qwen2.5-32B, and Qwen3-8B models. Results demonstrate a clear performance hierarchy: point-based methods < static pairwise training < Elo-Evolve across Alpaca Eval 2.0 and MT-Bench, validating the progressive benefits of pairwise comparison and dynamic opponent selection for LLM alignment.
>
---
#### [replaced 112] TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA参数无法跨模型迁移的问题。提出TiTok框架，通过token级知识迁移实现LoRA有效移植。**

- **链接: [https://arxiv.org/pdf/2510.04682](https://arxiv.org/pdf/2510.04682)**

> **作者:** Chanjoo Jung; Jaehyung Kim
>
> **备注:** ICLR 2026
>
> **摘要:** Large Language Models (LLMs) are widely applied in real world scenarios, yet fine-tuning them comes with significant computational and storage costs. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA mitigate these costs; however, the adapted parameters are dependent on the base model and cannot be transferred across different backbones. One way to address this issue is through knowledge distillation, but its effectiveness inherently depends on training data. Recent work such as TransLoRA avoids this by generating synthetic data; nevertheless, this adds complexity since it requires training an additional discriminator model. In this paper, we propose TiTok, a new framework that enables effective LoRA Transplantation through Token-level knowledge transfer. Specifically, TiTok captures task-relevant information through a token-wise contrastive excess between a source model with and without LoRA. This excess highlights informative tokens and enables selective filtering of synthetic data, all without additional models or overhead. Through experiments on three benchmarks across multiple transfer settings, we demonstrate that TiTok is consistently effective, achieving average performance gains of +4~10% compared to baselines overall.
>
---
#### [replaced 113] Steering Evaluation-Aware Language Models to Act Like They Are Deployed
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全评估任务，旨在解决LLM在评估时伪装行为的问题。通过添加转向向量抑制模型的评估意识，使其在评估时表现如部署时一样。**

- **链接: [https://arxiv.org/pdf/2510.20487](https://arxiv.org/pdf/2510.20487)**

> **作者:** Tim Tian Hua; Andrew Qin; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on two sets of documents describing its behavior. The first says that our model uses Python type hints during evaluation but not during deployment. The second says that our model can recognize that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. We find that activation steering can suppress evaluation awareness and make the model behave during evaluation as it would during deployment. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed.
>
---
#### [replaced 114] ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于工具检索任务，旨在解决大工具集下LLM上下文限制问题。通过生成合成工具描述，提升检索效果，使LLM能高效处理大量工具。**

- **链接: [https://arxiv.org/pdf/2510.19791](https://arxiv.org/pdf/2510.19791)**

> **作者:** Saptarshi Sengupta; Zhengyu Zhou; Jun Araki; Xingbo Wang; Bingqing Wang; Suhang Wang; Zhe Feng
>
> **备注:** Accepted to EACL 2026 (main/oral)
>
> **摘要:** Tool calling has become increasingly popular for Large Language Models (LLMs). However, for large tool sets, the resulting tokens would exceed the LLM's context window limit, making it impossible to include every tool. Hence, an external retriever is used to provide LLMs with the most relevant tools for a query. Existing retrieval models rank tools based on the similarity between a user query and a tool description (TD). This leads to suboptimal retrieval as user requests are often poorly aligned with the language of TD. To remedy the issue, we propose ToolDreamer, a framework to condition retriever models to fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e., description of tools that the LLM feels will be potentially useful for the query. The framework enables a more natural alignment between queries and tools within the language space of TD's. We apply ToolDreamer on the ToolRet dataset and show that our method improves the performance of sparse and dense retrievers with and without training, thus showcasing its flexibility. Through our proposed framework, our aim is to offload a portion of the reasoning burden to the retriever so that the LLM may effectively handle a large collection of tools without inundating its context window.
>
---
#### [replaced 115] Rethinking On-policy Optimization for Query Augmentation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决查询增强问题。通过比较基于提示和强化学习的方法，提出一种结合两者优势的混合方法OPQE，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2510.17139](https://arxiv.org/pdf/2510.17139)**

> **作者:** Zhichao Xu; Shengyao Zhuang; Xueguang Ma; Bingsen Chen; Yijun Tian; Fengran Mo; Jie Cao; Vivek Srikumar
>
> **摘要:** Recent advances in large language models (LLMs) have led to a surge of interest in query augmentation for information retrieval (IR). Two main approaches have emerged. The first prompts LLMs to generate answers or pseudo-documents that serve as new queries, relying purely on the model's parametric knowledge or contextual information. The second applies reinforcement learning (RL) to fine-tune LLMs for query rewriting, directly optimizing retrieval metrics. While having respective advantages and limitations, the two approaches have not been compared under consistent experimental conditions. In this work, we present the first systematic comparison of prompting-based and RL-based query augmentation across diverse benchmarks, including evidence-seeking, ad hoc, and tool retrieval. Our key finding is that simple, training-free query augmentation often performs on par with, or even surpasses, more expensive RL-based counterparts, especially when using powerful LLMs. Motivated by this discovery, we introduce a novel hybrid method, On-policy Pseudo-document Query Expansion (OPQE), which, instead of rewriting a query, the LLM policy learns to generate a pseudo-document that maximizes retrieval performance, thus merging the flexibility and generative structure of prompting with the targeted optimization of RL. We show OPQE outperforms both standalone prompting and RL-based rewriting, demonstrating that a synergistic approach yields the best results. Our implementation is made available to facilitate reproducibility.
>
---
#### [replaced 116] MENLO: From Preferences to Proficiency -- Evaluating and Modeling Native-like Quality Across 47 Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言大模型评估任务，旨在提升模型响应的自然程度。提出MENLO框架，构建多语言数据集，并通过多种方法优化模型表现。**

- **链接: [https://arxiv.org/pdf/2509.26601](https://arxiv.org/pdf/2509.26601)**

> **作者:** Chenxi Whitehouse; Sebastian Ruder; Tony Lin; Oksana Kurylo; Haruka Takagi; Janice Lam; Nicolò Busetto; Denise Diaz; Francisco Guzmán
>
> **备注:** ICLR 2026
>
> **摘要:** Ensuring native-like quality of large language model (LLM) responses across many languages is challenging. To address this, we introduce MENLO, a framework that operationalizes the evaluation of native-like response quality based on audience design-inspired mechanisms. Using MENLO, we create a dataset of 6,423 human-annotated prompt-response preference pairs covering four quality dimensions with high inter-annotator agreement in 47 language varieties. Our evaluation reveals that zero-shot LLM judges benefit significantly from pairwise evaluation and our structured annotation rubrics, yet they still underperform human annotators on our dataset. We demonstrate substantial improvements through fine-tuning with reinforcement learning, reward shaping, and multi-task learning approaches. Additionally, we show that RL-trained judges can serve as generative reward models to enhance LLMs' multilingual proficiency, though discrepancies with human judgment remain. Our findings suggest promising directions for scalable multilingual evaluation and preference alignment. We release our dataset and evaluation framework to support further research in multilingual LLM evaluation (this https URL).
>
---
#### [replaced 117] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决LLM在复杂推理中因“学习悬崖”导致的进展停滞问题。通过引入Scaf-GRPO框架，逐步提供引导以提升模型解题能力。**

- **链接: [https://arxiv.org/pdf/2510.19807](https://arxiv.org/pdf/2510.19807)**

> **作者:** Xichen Zhang; Sitong Wu; Yinghao Zhu; Haoru Tan; Shaozuo Yu; Ziyi He; Jiaya Jia
>
> **备注:** Code: this https URL Accepted by ICLR 2026
>
> **摘要:** Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM.
>
---
#### [replaced 118] Online Causal Kalman Filtering for Stable and Effective Policy Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，针对大语言模型中策略优化不稳定的问题，提出KPO方法，通过在线因果卡尔曼滤波稳定重要性采样比率，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2602.10609](https://arxiv.org/pdf/2602.10609)**

> **作者:** Shuo He; Lang Feng; Xin Cheng; Lei Feng; Bo An
>
> **备注:** Preprint
>
> **摘要:** Reinforcement learning for large language models suffers from high-variance token-level importance sampling (IS) ratios, which would destabilize policy optimization at scale. To improve stability, recent methods typically use a fixed sequence-level IS ratio for all tokens in a sequence or adjust each token's IS ratio separately, thereby neglecting temporal off-policy derivation across tokens in a sequence. In this paper, we first empirically identify that local off-policy deviation is structurally inconsistent at the token level, which may distort policy-gradient updates across adjacent tokens and lead to training collapse. To address the issue, we propose Online Causal Kalman Filtering for stable and effective Policy Optimization (KPO). Concretely, we model the desired IS ratio as a latent state that evolves across tokens and apply a Kalman filter to update this state online and autoregressively based on the states of past tokens, regardless of future tokens. The resulting filtered IS ratios preserve token-wise local structure-aware variation while strongly smoothing noise spikes, yielding more stable and effective policy updates. Experimentally, KPO achieves superior results on challenging math reasoning datasets compared with state-of-the-art counterparts.
>
---
#### [replaced 119] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents
- **分类: cs.CL**

- **简介: 该论文提出AgentSynth，用于生成通用计算机使用代理的高质量任务和轨迹数据集。解决任务生成效率与成本问题，通过构建复杂任务提升基准难度。**

- **链接: [https://arxiv.org/pdf/2506.14205](https://arxiv.org/pdf/2506.14205)**

> **作者:** Jingxu Xie; Dylan Xu; Xuandong Zhao; Dawn Song
>
> **备注:** ICLR 2026
>
> **摘要:** We introduce AgentSynth, a scalable and cost-efficient pipeline for automatically synthesizing high-quality tasks and trajectory datasets for generalist computer-use agents. Leveraging information asymmetry, AgentSynth constructs subtasks that are simple during generation but significantly more challenging when composed into long-horizon tasks, enabling the creation of over 6,000 diverse and realistic tasks. A key strength of AgentSynth is its ability to precisely modulate task complexity by varying the number of subtasks. Empirical evaluations show that state-of-the-art LLM agents suffer a steep performance drop, from 18% success at difficulty level 1 to just 4% at level 6, highlighting the benchmark's difficulty and discriminative power. Moreover, our pipeline achieves a low average cost of $0.60 per trajectory, orders of magnitude cheaper than human annotations. Our code and data are available at this https URL
>
---
#### [replaced 120] Polynomial, trigonometric, and tropical activations
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; math.AG**

- **简介: 该论文研究深度神经网络中的激活函数，探索多项式、三角和热带基函数作为激活函数的可行性，解决梯度消失/爆炸问题，提升模型训练效率。**

- **链接: [https://arxiv.org/pdf/2502.01247](https://arxiv.org/pdf/2502.01247)**

> **作者:** Ismail Khalfaoui-Hassani; Stefan Kesselheim
>
> **备注:** Published at ICLR 2026
>
> **摘要:** Which functions can be used as activations in deep neural networks? This article explores families of functions based on orthonormal bases, including the Hermite polynomial basis and the Fourier trigonometric basis, as well as a basis resulting from the tropicalization of a polynomial basis. Our study shows that, through simple variance-preserving initialization and without additional clamping mechanisms, these activations can successfully be used to train deep models, such as GPT-2 for next-token prediction on OpenWebText and ConvNeXt for image classification on ImageNet. Our work addresses the issue of exploding and vanishing activations and gradients, particularly prevalent with polynomial activations, and opens the door for improving the efficiency of large-scale learning tasks. Furthermore, our approach provides insight into the structure of neural networks, revealing that networks with polynomial activations can be interpreted as multivariate polynomial mappings. Finally, using Hermite interpolation, we show that our activations can closely approximate classical ones in pre-trained models by matching both the function and its derivative, making them especially useful for fine-tuning tasks. These activations are available in the torchortho library via: this https URL.
>
---
#### [replaced 121] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity
- **分类: cs.CL**

- **简介: 该论文属于数据过滤任务，旨在解决PPL过滤方法耗时且不可靠的问题。提出基于先验的过滤方法，利用词频统计快速筛选数据，提升效率并保持效果。**

- **链接: [https://arxiv.org/pdf/2509.18577](https://arxiv.org/pdf/2509.18577)**

> **作者:** Yeongbin Seo; Gayoung Kim; Jaehyung Kim; Jinyoung Yeo
>
> **备注:** ICLR 2026
>
> **摘要:** As large language models (LLMs) are pretrained on massive web corpora, careful selection of data becomes essential to ensure effective and efficient learning. While perplexity (PPL)-based filtering has shown strong performance, it suffers from drawbacks: substantial time costs and inherent unreliability of the model when handling noisy or out-of-distribution samples. In this work, we propose a simple yet powerful alternative: a prior-based data filtering method that estimates token priors using corpus-level term frequency statistics, inspired by linguistic insights on word roles and lexical density. Our approach filters documents based on the mean and standard deviation of token priors, serving as a fast proxy to PPL while requiring no model inference. Despite its simplicity, the prior-based filter achieves the highest average performance across 20 downstream benchmarks, while reducing time cost by over 1000x compared to PPL-based filtering. We further demonstrate its applicability to symbolic languages such as code and math, and its dynamic adaptability to multilingual corpora without supervision
>
---
#### [replaced 122] StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于金融领域AI研究任务，旨在评估LLM在真实股市中的交易能力。解决现有基准不适应动态交易的问题，构建了STOCKBENCH基准并测试多种模型表现。**

- **链接: [https://arxiv.org/pdf/2510.02209](https://arxiv.org/pdf/2510.02209)**

> **作者:** Yanxu Chen; Zijun Yao; Yantao Liu; Amy Xin; Jin Ye; Jianing Yu; Lei Hou; Juanzi Li
>
> **摘要:** Large language models (LLMs) demonstrate strong potential as autonomous agents, with promising capabilities in reasoning, tool use, and sequential decision-making. While prior benchmarks have evaluated LLM agents in various domains, the financial domain remains underexplored, despite its significant economic value and complex reasoning requirements. Most existing financial benchmarks focus on static question-answering, failing to capture the dynamics of real-market trading. To address this gap, we introduce STOCKBENCH, a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments. Agents receive daily market signals -- including prices, fundamentals, and news -- and make sequential buy, sell, or hold decisions. Performance is measured using financial metrics such as cumulative return, maximum drawdown, and the Sortino ratio, capturing both profitability and risk management. We evaluate a wide range of state-of-the-art proprietary and open-source LLMs. Surprisingly, most models struggle to outperform the simple buy-and-hold baseline, while some models demonstrate the potential to achieve higher returns and stronger risk management. These findings highlight both the challenges and opportunities of LLM-based trading agents, showing that strong performance on static financial question-answering do not necessarily translate into effective trading behavior. We release STOCKBENCH as an open-source benchmark to enable future research on LLM-driven financial agents.
>
---
#### [replaced 123] GLEE: A Unified Framework and Benchmark for Language-based Economic Environments
- **分类: cs.CL; cs.AI; cs.CY; cs.GT; cs.LG**

- **简介: 该论文提出GLEE框架与基准，用于研究语言驱动的经济环境中的智能体行为。任务是评估LLM在经济互动中的表现，解决其理性、效率及公平性问题。工作包括设计游戏模型、构建数据集并进行实验分析。**

- **链接: [https://arxiv.org/pdf/2410.05254](https://arxiv.org/pdf/2410.05254)**

> **作者:** Eilam Shapira; Omer Madmon; Itamar Reinman; Samuel Joseph Amouyal; Roi Reichart; Moshe Tennenholtz
>
> **摘要:** Large Language Models (LLMs) show significant potential in economic and strategic interactions, where communication via natural language is often prevalent. This raises key questions: Do LLMs behave rationally? How do they perform compared to humans? Do they tend to reach an efficient and fair outcome? What is the role of natural language in strategic interaction? How do characteristics of the economic environment influence these dynamics? These questions become crucial concerning the economic and societal implications of integrating LLM-based agents into real-world data-driven systems, such as online retail platforms and recommender systems. To answer these questions, we introduce a benchmark for standardizing research on two-player, sequential, language-based games. Inspired by the economic literature, we define three base families of games with consistent parameterization, degrees of freedom and economic measures to evaluate agents' performance (self-gain), as well as the game outcome (efficiency and fairness). We develop an open-source framework for interaction simulation and analysis, and utilize it to collect a dataset of LLM vs. LLM interactions across numerous game configurations and an additional dataset of human vs. LLM interactions. Through extensive experimentation, we demonstrate how our framework and dataset can be used to: (i) compare the behavior of LLM-based agents in various economic contexts; (ii) evaluate agents in both individual and collective performance measures; and (iii) quantify the effect of the economic characteristics of the environments on the behavior of agents. Our results suggest that the market parameters, as well as the choice of the LLMs, tend to have complex and interdependent effects on the economic outcome, which calls for careful design and analysis of the language-based economic ecosystem.
>
---
#### [replaced 124] Energy-Regularized Sequential Model Editing on Hyperspheres
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决顺序编辑导致的性能下降和知识遗忘问题。通过引入超球面能量机制，提出SPHERE方法以稳定权重分布并保留先验知识。**

- **链接: [https://arxiv.org/pdf/2510.01172](https://arxiv.org/pdf/2510.01172)**

> **作者:** Qingyuan Liu; Jia-Chen Gu; Yunzhi Yao; Hong Wang; Nanyun Peng
>
> **备注:** Accepted by ICLR 2026. The code is available at this https URL. Project page: this https URL
>
> **摘要:** Large language models (LLMs) require constant updates to remain aligned with evolving real-world knowledge. Model editing offers a lightweight alternative to retraining, but sequential editing often destabilizes representations and induces catastrophic forgetting. In this work, we seek to better understand and mitigate performance degradation caused by sequential editing. We hypothesize that hyperspherical uniformity, a property that maintains uniform distribution of neuron weights on a hypersphere, helps the model remain stable, retain prior knowledge, while still accommodate new updates. We use Hyperspherical Energy (HE) to quantify neuron uniformity during editing, and examine its correlation with editing performance. Empirical studies across widely used editing methods reveals a strong correlation between HE dynamics and editing performance, with editing failures consistently coinciding with high HE fluctuations. We further theoretically prove that HE dynamics impose a lower bound on the degradation of pretrained knowledge, highlighting why HE stability is crucial for knowledge retention. Motivated by these insights, we propose SPHERE (Sparse Projection for Hyperspherical Energy-Regularized Editing), an HE-driven regularization strategy that stabilizes neuron weight distributions, ultimately preserving prior knowledge while enabling reliable sequential updates. Specifically, SPHERE identifies a sparse space complementary to the principal hyperspherical directions of the pretrained weight matrices and projects new knowledge onto it, attenuating perturbations on the principal directions. Extensive experiments on LLaMA3 (8B) and Qwen2.5 (7B) show that SPHERE outperforms the best baseline in editing capability by an average of 16.41%, while most faithfully preserving general model performance, thereby offering a principled path toward reliable large-scale knowledge editing.
>
---
#### [replaced 125] A Diagnostic Benchmark for Sweden-Related Factual Knowledge
- **分类: cs.CL**

- **简介: 该论文属于知识问答任务，旨在解决瑞典相关事实知识测试不足的问题。通过构建专门的瑞典基准数据集，评估模型的 factual recall 和跨语言一致性。**

- **链接: [https://arxiv.org/pdf/2510.21360](https://arxiv.org/pdf/2510.21360)**

> **作者:** Jenny Kunz
>
> **备注:** To appear at LREC 2026
>
> **摘要:** Many Swedish benchmarks are translations of US-centric benchmarks and are therefore not suitable for testing knowledge that is particularly relevant, or even specific, to Sweden. We therefore introduce a manually written question-answering benchmark specifically targeted at Sweden-related personalities and events, many of which receive very limited coverage in international media. Our annotators drew inspiration from a popular radio program featuring public figures from culture and media, as well as major sports events in Sweden. The dataset can be used to measure factual recall across models of varying sizes and degrees of Swedish coverage, and allows probing of cross-lingual factual consistency, as it contains English translations. Using the dataset, we find that smaller models with stronger Swedish coverage perform comparably to a multilingual model three times larger in recalling Sweden-related facts. We also observe that continued pre-training on Swedish generally improves factual knowledge but leads to partial forgetting of previously known information. These results demonstrate the dataset's potential as a diagnostic tool for studying language adaptation and knowledge retention in multilingual models during language adaptation.
>
---
#### [replaced 126] Chain of Correction for Full-text Speech Recognition with Large Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别后处理任务，旨在解决全文本纠错中的稳定性、可控性等问题。提出Chain of Correction方法，分段纠正错误，提升纠错效果。**

- **链接: [https://arxiv.org/pdf/2504.01519](https://arxiv.org/pdf/2504.01519)**

> **作者:** Zhiyuan Tang; Dong Wang; Zhikai Zhou; Yong Liu; Shen Huang; Shidong Shang
>
> **备注:** ICASSP 2026
>
> **摘要:** Full-text error correction with Large Language Models (LLMs) for Automatic Speech Recognition (ASR) is attracting increased attention for its ability to address a wide range of error types, such as punctuation restoration and inverse text normalization, across long context. However, challenges remain regarding stability, controllability, completeness, and fluency. To mitigate these issues, this paper proposes the Chain of Correction (CoC), which uses a multi-turn chat format to correct errors segment by segment, guided by pre-recognized text and full-text context for better semantic understanding. Utilizing the open-sourced ChFT dataset, we fine-tune a pre-trained LLM to evaluate CoC's performance. Experiments show that CoC significantly outperforms baseline and benchmark systems in correcting full-text ASR outputs. We also analyze correction thresholds to balance under-correction and over-rephrasing, extrapolate CoC on extra-long ASR outputs, and explore using other types of information to guide error correction.
>
---
#### [replaced 127] Distribution-Aligned Decoding for Efficient LLM Task Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型任务适应领域，旨在解决高效微调难题。提出SVDecode方法，通过输出分布对齐提升模型性能，无需增加参数。**

- **链接: [https://arxiv.org/pdf/2509.15888](https://arxiv.org/pdf/2509.15888)**

> **作者:** Senkang Hu; Xudong Han; Jinqi Jiang; Yihang Tao; Zihan Fang; Yong Dai; Sam Tak Wu Kwong; Yuguang Fang
>
> **备注:** Accepted by NeurIPS'25
>
> **摘要:** Adapting billion-parameter language models to a downstream task is still costly, even with parameter-efficient fine-tuning (PEFT). We re-cast task adaptation as output-distribution alignment: the objective is to steer the output distribution toward the task distribution directly during decoding rather than indirectly through weight updates. Building on this view, we introduce Steering Vector Decoding (SVDecode), a lightweight, PEFT-compatible, and theoretically grounded method. We start with a short warm-start fine-tune and extract a task-aware steering vector from the Kullback-Leibler (KL) divergence gradient between the output distribution of the warm-started and pre-trained models. This steering vector is then used to guide the decoding process to steer the model's output distribution towards the task distribution. We theoretically prove that SVDecode is first-order equivalent to the gradient step of full fine-tuning and derive a globally optimal solution for the strength of the steering vector. Across three tasks and nine benchmarks, SVDecode paired with four standard PEFT methods improves multiple-choice accuracy by up to 5 percentage points and open-ended truthfulness by 2 percentage points, with similar gains (1-2 percentage points) on commonsense datasets without adding trainable parameters beyond the PEFT adapter. SVDecode thus offers a lightweight, theoretically grounded path to stronger task adaptation for large language models. Code is available at this https URL.
>
---
#### [replaced 128] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的意外代码切换问题。通过稀疏自编码器分析并调整语言特征激活值，提出SASFT方法有效减少代码切换，同时保持多语言能力。**

- **链接: [https://arxiv.org/pdf/2507.14894](https://arxiv.org/pdf/2507.14894)**

> **作者:** Boyi Deng; Yu Wan; Baosong Yang; Fei Huang; Wenjie Wang; Fuli Feng
>
> **备注:** ICLR 2026
>
> **摘要:** Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses. However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness. In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in one case. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities. The code and data are available at this https URL.
>
---
#### [replaced 129] Long-Context Generalization with Sparse Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究长文本中的注意力机制，解决传统softmax导致的注意力分散问题。提出ASEntmax方法，实现动态稀疏注意力，提升长上下文泛化能力。**

- **链接: [https://arxiv.org/pdf/2506.16640](https://arxiv.org/pdf/2506.16640)**

> **作者:** Pavlo Vasylenko; Hugo Pitorro; André F. T. Martins; Marcos Treviso
>
> **备注:** ICLR 2026
>
> **摘要:** Transformer-based architectures traditionally employ softmax to compute attention weights, which produces dense distributions over all tokens in a sequence. While effective in many settings, this density has been shown to be detrimental for tasks that demand precise focus on fixed-size patterns: as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse. We show in this paper that dynamically sparse attention mechanisms using $\alpha$-entmax can avoid these issues, due to their ability to assign exact zeros to irrelevant tokens. Furthermore, we introduce Adaptive-Scalable Entmax (ASEntmax), which endows $\alpha$-entmax with a learnable temperature parameter, allowing the attention distribution to interpolate between sparse (pattern-focused) and dense (softmax-like) regimes. Our empirical evaluation on synthetic tasks and language modeling demonstrates that ASEntmax substantially outperforms softmax, scalable softmax, and fixed-temperature $\alpha$-entmax baselines, achieving up to 1000$\times$ length extrapolation on synthetic benchmarks and superior long-context generalization on language modeling while preserving short-context performance, including better perplexity trends and higher retrieval accuracies at 8$\times$ training length.
>
---
#### [replaced 130] When Data is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决DPO数据集比较困难的问题。通过分析和标注现有数据集，提出新的混合数据集UltraMix，提升性能并促进后续研究。**

- **链接: [https://arxiv.org/pdf/2511.10985](https://arxiv.org/pdf/2511.10985)**

> **作者:** Aladin Djuhera; Farhan Ahmed; Swanand Ravindra Kadhe; Syed Zawad; Heiko Ludwig; Holger Boche
>
> **摘要:** Aligning large language models (LLMs) is a central objective of post-training, often achieved through reward modeling and reinforcement learning methods. Among these, direct preference optimization (DPO) has emerged as a widely adopted technique that fine-tunes LLMs on preferred completions over less favorable ones. While most frontier LLMs do not disclose their curated preference pairs, the broader LLM community has released several open-source DPO datasets, including TuluDPO, ORPO, UltraFeedback, HelpSteer, and Code-Preference-Pairs. However, systematic comparisons remain scarce, largely due to the high computational cost and the lack of rich quality annotations, making it difficult to understand how preferences were selected, which task types they span, and how well they reflect human judgment on a per-sample level. In this work, we present the first comprehensive, data-centric analysis of popular open-source DPO corpora. We leverage the Magpie framework to annotate each sample for task category, input quality, and preference reward, a reward-model-based signal that validates the preference order without relying on human annotations. This enables a scalable, fine-grained inspection of preference quality across datasets, revealing structural and qualitative discrepancies in reward margins. Building on these insights, we systematically curate a new DPO mixture, UltraMix, that draws selectively from all five corpora while removing noisy or redundant samples. UltraMix is 30% smaller than the best-performing individual dataset yet exceeds its performance across key benchmarks. We publicly release all annotations, metadata, and our curated mixture to facilitate future research in data-centric preference optimization.
>
---
#### [replaced 131] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于目标解码任务，旨在通过眼动数据自动识别阅读时的开放性信息目标。研究构建了评估框架，比较了多种模型，验证了眼动数据在目标识别中的有效性。**

- **链接: [https://arxiv.org/pdf/2505.02872](https://arxiv.org/pdf/2505.02872)**

> **作者:** Cfir Avraham Hadar; Omer Shubi; Yoav Meiri; Amit Heshes; Yevgeni Berzak
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** When reading, we often have specific information that interests us in a text. For example, you might be reading this paper because you are curious about LLMs for eye movements in reading, the experimental design, or perhaps you wonder ``This sounds like science fiction. Does it actually work?''. More broadly, in daily life, people approach texts with any number of text-specific goals that guide their reading behavior. In this work, we ask, for the first time, whether open-ended reading goals can be automatically decoded solely from eye movements in reading. To address this question, we introduce goal decoding tasks and evaluation frameworks using large-scale eye tracking for reading data in English with hundreds of text-specific information seeking tasks. We develop and compare several discriminative and generative multimodal text and eye movements LLMs for these tasks. Our experiments show considerable success on the task of selecting the correct goal among several options, and even progress towards free-form textual reconstruction of the precise goal formulation. These results open the door for further scientific investigation of goal driven reading, as well as the development of educational and assistive technologies that will rely on real-time decoding of reader goals from their eye movements.
>
---
#### [replaced 132] Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs
- **分类: cs.CL**

- **简介: 该论文研究多轮交互中大语言模型的性能优化问题，提出T2PAM框架和ROSA算法，通过用户反馈实时调整模型参数，提升任务效果与效率。**

- **链接: [https://arxiv.org/pdf/2509.23166](https://arxiv.org/pdf/2509.23166)**

> **作者:** Chenxing Wei; Hong Wang; Ying He; Fei Yu; Yao Shu
>
> **备注:** 32 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) employ multi-turn interaction as a fundamental paradigm for completing complex tasks. However, their performance often degrades in extended interactions, as they are typically trained on static, single-turn data, which hinders their ability to adapt to real-time user feedback. To address this limitation, we first propose a new paradigm: Test-Time Policy Adaptation for Multi-Turn Interactions (T2PAM), which utilizes user feedback from the ongoing interaction as a reward signal to estimate a latent optimal policy aligned with user preferences, then updates a small subset of parameters to steer the model toward this policy, ultimately enabling efficient in-conversation self-correction. We then introduce Optimum-Referenced One-Step Adaptation (ROSA), a lightweight algorithm that operationalizes T2PAM. ROSA guides the model parameters toward a theoretical optimal policy in a single, efficient update step, avoiding costly iterative gradient-based optimization and minimizing computational overhead. We provide a rigorous theoretical analysis guaranteeing that the policy of ROSA converges to the preference of user as the number of interactions increases. Extensive experiments on challenging benchmark demonstrate that ROSA achieves significant improvements in both task effectiveness and efficiency.
>
---
#### [replaced 133] Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决轨迹多样性不足的问题。通过提出LATR方法，增强轨迹探索，提升政策学习效果。**

- **链接: [https://arxiv.org/pdf/2510.24302](https://arxiv.org/pdf/2510.24302)**

> **作者:** Shangyu Xing; Siyuan Wang; Chenyuan Yang; Xinyu Dai; Xiang Ren
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are publicly available at this https URL.
>
---
#### [replaced 134] FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出FictionalQA数据集，用于研究语言模型的事实记忆与原文记忆。任务属于自然语言处理中的知识获取与记忆研究，旨在探索模型如何记忆训练数据中的事实和序列。**

- **链接: [https://arxiv.org/pdf/2506.05639](https://arxiv.org/pdf/2506.05639)**

> **作者:** John Kirchenbauer; Janny Mongkolsupawan; Yuxin Wen; Tom Goldstein; Daphne Ippolito
>
> **备注:** 10 pages and 8 figures in the main body. Published at ICLR 2026. Dataset is available at this https URL, and code at this https URL
>
> **摘要:** When language models are trained on textual data, they acquire both knowledge about the structure of language as well as knowledge of facts about the world. At inference time, their knowledge of facts can be leveraged to solve interesting problems and perform useful knowledge work for users. It is well known that language models can verbatim memorize long sequences from their training data. However, it is much less well understood how language models memorize facts seen during training. In this work, we propose a new dataset to specifically empower researchers to study the dual processes of fact memorization and verbatim sequence memorization. The dataset consists of synthetically-generated, webtext-like documents about fictional events, as well as question-answer pairs about the events. We conduct training experiments showing how synthetic data about fictional events can be useful for studying different forms of memorization. We also document some challenges in effectively building realistic, fictional synthetic data.
>
---
#### [replaced 135] Large Language Model Agent in Financial Trading: A Survey
- **分类: q-fin.TR; cs.CL**

- **简介: 该论文属于金融交易任务，探讨如何利用大语言模型作为交易代理，解决其能否超越专业交易员的问题。工作包括综述现有研究、分析架构与数据输入，并指出未来方向。**

- **链接: [https://arxiv.org/pdf/2408.06361](https://arxiv.org/pdf/2408.06361)**

> **作者:** Han Ding; Yinheng Li; Junhao Wang; Hang Chen; Doudou Guo; Yunbai Zhang
>
> **摘要:** Trading is a highly competitive task that requires a combination of strategy, knowledge, and psychological fortitude. With the recent success of large language models(LLMs), it is appealing to apply the emerging intelligence of LLM agents in this competitive arena and understanding if they can outperform professional traders. In this survey, we provide a comprehensive review of the current research on using LLMs as agents in financial trading. We summarize the common architecture used in the agent, the data inputs, and the performance of LLM trading agents in backtesting as well as the challenges presented in these research. This survey aims to provide insights into the current state of LLM-based financial trading agents and outline future research directions in this field.
>
---
#### [replaced 136] GEM: A Gym for Agentic LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出GEM，一个用于代理式大语言模型的环境模拟器，解决训练范式转型问题。提供标准化框架、多样化环境及基准测试，助力代理式LLM研究。**

- **链接: [https://arxiv.org/pdf/2510.01051](https://arxiv.org/pdf/2510.01051)**

> **作者:** Zichen Liu; Anya Sims; Keyu Duan; Changyu Chen; Simon Yu; Xiangxin Zhou; Haotian Xu; Shaopan Xiong; Bo Liu; Chenmien Tan; Chuen Yang Beh; Weixun Wang; Hao Zhu; Weiyan Shi; Diyi Yang; Michael Shieh; Yee Whye Teh; Wee Sun Lee; Min Lin
>
> **摘要:** The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which -- unlike GRPO -- is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research.
>
---
#### [replaced 137] The Counting Power of Transformers
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文研究Transformer的计数能力，解决其表达性问题。工作包括建立形式框架，证明Transformer可表达高阶多项式计数属性，并分析其计算复杂性。**

- **链接: [https://arxiv.org/pdf/2505.11199](https://arxiv.org/pdf/2505.11199)**

> **作者:** Marco Sälzer; Chris Köcher; Alexander Kozachinskiy; Georg Zetzsche; Anthony Widjaja Lin
>
> **备注:** Accepted for ICLR 2026
>
> **摘要:** Counting properties (e.g. determining whether certain tokens occur more than other tokens in a given input text) have played a significant role in the study of expressiveness of transformers. In this paper, we provide a formal framework for investigating the counting power of transformers. We argue that all existing results demonstrate transformers' expressivity only for (semi-)linear counting properties, i.e., which are expressible as a boolean combination of linear inequalities. Our main result is that transformers can express counting properties that are highly nonlinear. More precisely, we prove that transformers can capture all semialgebraic counting properties, i.e., expressible as a boolean combination of arbitrary multivariate polynomials (of any degree). Among others, these generalize the counting properties that can be captured by C-RASP softmax transformers, which capture only linear counting properties. To complement this result, we exhibit a natural subclass of (softmax) transformers that completely characterizes semialgebraic counting properties. Through connections with the Hilbert's tenth problem, this expressivity of transformers also yields a new undecidability result for analyzing an extremely simple transformer model -- surprisingly with neither positional encodings (i.e. NoPE-transformers) nor masking. We also experimentally validate trainability of such counting properties.
>
---
#### [replaced 138] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Self-Harmony框架，解决TTRL中学习信号不稳定的问题。通过模型自监督与重述机制，提升答案稳定性，实现高准确率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.01191](https://arxiv.org/pdf/2511.01191)**

> **作者:** Ru Wang; Wei Huang; Qi Cao; Yusuke Iwasawa; Yutaka Matsuo; Jiaxian Guo
>
> **备注:** Accepted at the 14th International Conference on Learning Representations (ICLR 2026), Poster
>
> **摘要:** Test-time reinforcement learning (TTRL) offers a label-free paradigm for adapting models using only synthetic signals at inference, but its success hinges on constructing reliable learning signals. Standard approaches such as majority voting often collapse to spurious yet popular answers. We introduce Self-Harmony, a framework built on a simple intuition: the correct answer should remain stable across both an original question and its paraphrase. Self-Harmony operationalizes this by employing a single model in two complementary roles: a Solver to produce answers and a Reframer to rephrase the input. Based on this, we further propose a pseudo-label method: instead of majority voting, it aggregates answer frequencies across these original and reframed views using the harmonic mean. This is a process that naturally selects for solutions stable under reframing, thereby avoiding the common trap of favoring view-dependent, spurious answers. Crucially, this requires no human supervision or auxiliary models. Across diverse reasoning benchmarks, Self-Harmony achieves state-of-the-art results at the label-free test-time setting, ranking first in 28 of 30 settings across multiple methods. Beyond accuracy, it demonstrates unprecedented robustness, with zero training failures in all experiments, underscoring its stability and reliability.
>
---
#### [replaced 139] Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识图谱构建任务，旨在解决大规模文档构建高质量KG的经济性和准确性问题。通过合成数据和模型蒸馏，提出Distill-SynthKG方法，提升KG质量与检索效果。**

- **链接: [https://arxiv.org/pdf/2410.16597](https://arxiv.org/pdf/2410.16597)**

> **作者:** Prafulla Kumar Choubey; Xin Su; Man Luo; Xiangyu Peng; Caiming Xiong; Tiep Le; Shachar Rosenman; Vasudev Lal; Phil Mui; Ricky Ho; Phillip Howard; Chien-Sheng Wu
>
> **摘要:** Document-level knowledge graph (KG) construction faces a fundamental scaling challenge: existing methods either rely on expensive large language models (LLMs), making them economically nonviable for large-scale corpora, or employ smaller models that produce incomplete and inconsistent graphs. We find that this limitation stems not from model capabilities but from insufficient training on high-quality document-level KG data. To address this gap, we introduce SynthKG, a multi-step data synthesis pipeline that generates high-quality document-KG pairs through systematic chunking, decontextualization, and structured extraction using LLMs. By fine-tuning a smaller LLM on synthesized document-KG pairs, we streamline the multi-step process into a single-step KG generation approach called Distill-SynthKG. Furthermore, we repurpose existing question-answering datasets to construct KG evaluation datasets and introduce new evaluation metrics. Using KGs produced by Distill-SynthKG, we also design a novel graph-based retrieval framework for RAG. Experimental results demonstrate that Distill-SynthKG not only surpasses all baseline models in KG quality (including models up to eight times larger) but also consistently improves in retrieval and question-answering tasks. Additionally, our proposed graph retrieval framework outperforms all KG-retrieval methods across multiple benchmark datasets.
>
---
#### [replaced 140] Addressing Longstanding Challenges in Cognitive Science with Language Models
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨语言模型在认知科学中的应用，旨在解决研究整合与概念清晰度等问题。通过分析语言模型的优缺点，提出其作为辅助工具促进认知科学发展的可能性。**

- **链接: [https://arxiv.org/pdf/2511.00206](https://arxiv.org/pdf/2511.00206)**

> **作者:** Dirk U. Wulff; Rui Mata
>
> **摘要:** Cognitive science faces ongoing challenges in research integration, formalization, conceptual clarity, and other areas, in part due to its multifaceted and interdisciplinary nature. Recent advances in artificial intelligence, particularly the development of language models, offer tools that may help to address these longstanding issues. We outline the current capabilities and limitations of language models in these domains, including potential pitfalls. Taken together, we conclude that language models could serve as tools for a more integrative and cumulative cognitive science when used judiciously to complement, rather than replace, human agency.
>
---
#### [replaced 141] DRA-GRPO: Your GRPO Needs to Know Diverse Reasoning Paths for Mathematical Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数学推理任务，解决GRPO方法中因奖励不敏感导致的多样性不足问题，提出DRA框架通过语义密度调整奖励，提升策略多样性与性能。**

- **链接: [https://arxiv.org/pdf/2505.09655](https://arxiv.org/pdf/2505.09655)**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Xuanzhao Dong; Hao Wang; Haiyu Wu; Huayu Li; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **摘要:** Post-training LLMs with Reinforcement Learning, specifically Group Relative Policy Optimization (GRPO), has emerged as a paradigm for enhancing mathematical reasoning. However, standard GRPO relies on scalar correctness rewards that are often non-injective with respect to semantic content: distinct reasoning paths receive identical rewards. This leads to a Diversity-Quality Inconsistency, where the policy collapses into a narrow set of dominant modes while ignoring equally valid but structurally novel strategies. To bridge this gap, we propose Diversity-aware Reward Adjustment (DRA), a theoretically grounded framework that calibrates the reward signal using the semantic density of sampled groups. By leveraging Submodular Mutual Information (SMI), DRA implements an Inverse Propensity Scoring (IPS) mechanism that effectively de-biases the gradient estimation. This creates a repulsive force against redundancy, driving the policy to achieve better coverage of the high-reward landscape. Our method is plug-and-play and integrates seamlessly with GRPO variants. Empirical evaluations on five math benchmarks demonstrate that DRA-GRPO consistently outperforms strong baselines, achieving an average accuracy of 58.2% on DeepSeek-R1-Distill-Qwen-1.5B with only 7,000 training samples and $55 cost, highlighting the critical role of diversity calibration in data-efficient alignment. The code is available at this https URL.
>
---
#### [replaced 142] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态大语言模型的推理能力提升任务，旨在解决RL训练中难以激活复杂推理的问题。通过构建高质量数据集并采用PTST和GRPO策略，提升模型的多模态推理性能。**

- **链接: [https://arxiv.org/pdf/2503.06749](https://arxiv.org/pdf/2503.06749)**

> **作者:** Wenxuan Huang; Bohan Jia; Zijie Zhai; Shaosheng Cao; Zheyu Ye; Fei Zhao; Zhe Xu; Xu Tang; Yao Hu; Shaohui Lin
>
> **备注:** Accepted to ICLR 2026. Code is available at this https URL
>
> **摘要:** DeepSeek-R1-Zero has successfully demonstrated the emergence of reasoning capabilities in LLMs purely through Reinforcement Learning (RL). Inspired by this breakthrough, we explore how RL can be utilized to enhance the reasoning capability of MLLMs. However, direct training with RL struggles to activate complex reasoning capabilities such as questioning and reflection in MLLMs, due to the absence of substantial high-quality multimodal reasoning data. To address this issue, we propose the reasoning MLLM, Vision-R1, to improve multimodal reasoning capability. Specifically, we first construct a high-quality multimodal CoT dataset without human annotations by leveraging an existing MLLM and DeepSeek-R1 through modality bridging and data filtering to obtain a 200K multimodal CoT dataset, Vision-R1-cold dataset. It serves as cold-start initialization data for Vision-R1. To mitigate the optimization challenges caused by overthinking after cold start, we propose Progressive Thinking Suppression Training (PTST) strategy and employ Group Relative Policy Optimization (GRPO) with the hard formatting result reward function to gradually refine the model's ability to learn correct and complex reasoning processes on a 10K multimodal math dataset. Comprehensive experiments show our model achieves an average improvement of $\sim$6% across various multimodal math reasoning benchmarks. Vision-R1-7B achieves a 73.5% accuracy on the widely used MathVista benchmark, which is only 0.4% lower than the leading reasoning model, OpenAI O1. Scaling up the amount of multimodal math data in the RL training, Vision-R1-32B and Vison-R1-72B achieves 76.4% and 78.2% MathVista benchmark scores, respectively. The datasets and code will be released in: this https URL .
>
---
#### [replaced 143] ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection
- **分类: cs.CL**

- **简介: 该论文属于仇恨表情包检测任务，旨在解决现有方法无法提供解释的问题。提出ExPO-HM模型，结合解释与检测，提升检测精度和可解释性。**

- **链接: [https://arxiv.org/pdf/2510.08630](https://arxiv.org/pdf/2510.08630)**

> **作者:** Jingbiao Mei; Mingsheng Sun; Jinghong Chen; Pengda Qin; Yuhong Li; Da Chen; Bill Byrne
>
> **备注:** ICLR 2026
>
> **摘要:** Hateful memes have emerged as a particularly challenging form of online abuse, motivating the development of automated detection systems. Most prior approaches rely on direct detection, producing only binary predictions. Such models fail to provide the context and explanations that real-world moderation requires. Recent Explain-then-Detect approaches, using Chain-of-Thought prompting or LMM agents, perform worse than simple SFT baselines, and even advanced post-training methods such as GRPO fail to close the gap. Our analysis identifies two key issues of such systems: important policy-relevant cues such as targets and attack types are not hypothesized by the model as a likely explanation; and the binary reward signal is insufficient to guide reasoning. To address these challenges, we propose ExPO-HM (Explain-then-Detect Policy Optimization for Hateful Memes), inspired by the training and evaluation process of human annotators. ExPO-HM combines SFT warmup, GRPO with curriculum learning, and Conditional Decision Entropy (CDE) as both metric and reward for reasoning quality. Across three hateful meme benchmarks, ExPO-HM achieves state-of-the-art performance on binary detection, fine-grained classification, and reasoning quality, with up to 15\% and 17\% F1 improvement over the GRPO and DPO baselines, respectively. By moving hateful meme detection from simple binary alarms to explanation-driven detection, ExPO-HM provides accurate, interpretable, and actionable moderation support. Code available at this https URL
>
---
#### [replaced 144] SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医学诊断任务，旨在解决AI模型无法理解肺功能测试数据的问题。研究提出SpiroLLM，通过融合生理信号与大语言模型，生成可解释的COPD诊断报告。**

- **链接: [https://arxiv.org/pdf/2507.16145](https://arxiv.org/pdf/2507.16145)**

> **作者:** Shuhao Mei; Yongchao Long; Xiaoyu Xiao; Shan Cao; Xiaobo Han; Shijia Geng; Jinbo Sun; Yuxi Zhou; Shenda Hong
>
> **摘要:** Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of respiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8977 (95% CI: 0.88-0.91). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools.
>
---
#### [replaced 145] Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出JBF系统，解决LLM安全评估中基准滞后问题，通过自动化将攻击论文转化为可执行模块，实现标准化评估。**

- **链接: [https://arxiv.org/pdf/2602.24009](https://arxiv.org/pdf/2602.24009)**

> **作者:** Zhicheng Fang; Jingjie Zheng; Chenxu Fu; Wei Xu
>
> **摘要:** Jailbreak techniques for large language models (LLMs) evolve faster than benchmarks, making robustness estimates stale and difficult to compare across papers due to drift in datasets, harnesses, and judging protocols. We introduce JAILBREAK FOUNDRY (JBF), a system that addresses this gap via a multi-agent workflow to translate jailbreak papers into executable modules for immediate evaluation within a unified harness. JBF features three core components: (i) JBF-LIB for shared contracts and reusable utilities; (ii) JBF-FORGE for the multi-agent paper-to-module translation; and (iii) JBF-EVAL for standardizing evaluations. Across 30 reproduced attacks, JBF achieves high fidelity with a mean (reproduced-reported) attack success rate (ASR) deviation of +0.26 percentage points. By leveraging shared infrastructure, JBF reduces attack-specific implementation code by nearly half relative to original repositories and achieves an 82.5% mean reused-code ratio. This system enables a standardized AdvBench evaluation of all 30 attacks across 10 victim models using a consistent GPT-4o judge. By automating both attack integration and standardized evaluation, JBF offers a scalable solution for creating living benchmarks that keep pace with the rapidly shifting security landscape.
>
---
#### [replaced 146] Learning to Reason without External Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决依赖外部奖励的局限性。通过引入内部反馈机制，使用模型自信心作为奖励信号，实现无需外部监督的高效学习。**

- **链接: [https://arxiv.org/pdf/2505.19590](https://arxiv.org/pdf/2505.19590)**

> **作者:** Xuandong Zhao; Zhewei Kang; Aosong Feng; Sergey Levine; Dawn Song
>
> **备注:** ICLR 2026
>
> **摘要:** Training large language models (LLMs) for complex reasoning via Reinforcement Learning with Verifiable Rewards (RLVR) is effective but limited by reliance on costly, domain-specific supervision. We explore Reinforcement Learning from Internal Feedback (RLIF), a framework that enables LLMs to learn from intrinsic signals without external rewards or labeled data. We propose Intuitor, an RLIF method that uses a model's own confidence-termed self-certainty-as its sole reward signal. Intuitor replaces external rewards in Group Relative Policy Optimization (GRPO) with self-certainty scores, enabling fully unsupervised learning. Experiments demonstrate that Intuitor matches GRPO's performance on mathematical benchmarks while achieving better generalization to out-of-domain tasks like code generation, without requiring gold solutions or test cases. Our findings show that intrinsic model signals can drive effective learning across domains, offering a scalable alternative to RLVR for autonomous AI systems where verifiable rewards are unavailable. Code is available at this https URL
>
---
#### [replaced 147] Exposing Citation Vulnerabilities in Generative Engines
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文属于安全评估任务，旨在解决生成引擎的引用漏洞问题。通过分析引用来源的属性，评估中毒攻击风险，并提出防御建议。**

- **链接: [https://arxiv.org/pdf/2510.06823](https://arxiv.org/pdf/2510.06823)**

> **作者:** Riku Mochizuki; Shusuke Komatsu; Souta Noguchi; Kazuto Ataka
>
> **备注:** 12 pages, under-reviewing at a conference
>
> **摘要:** We analyze answers generated by generative engines (GEs) from the perspectives of citation publishers and the content-injection barrier, defined as the difficulty for attackers to manipulate answers to user prompts by placing malicious content on the web. GEs integrate two functions: web search and answer generation that cites web pages using large language models. Because anyone can publish information on the web, GEs are vulnerable to poisoning attacks. Existing studies of citation evaluation focus on how faithfully answer content reflects cited sources, leaving unexamined which web sources should be selected as citations to defend against poisoning attacks. To fill this gap, we introduce evaluation criteria that assess poisoning threats using the citation information contained in answers. Our criteria classify the publisher attributes of citations to estimate the content-injection barrier thereby revealing the threat of poisoning attacks in current GEs. We conduct experiments in political domains in Japan and the United States (U.S.) using our criteria and show that citations from official party websites (primary sources) are approximately \(25\%\)--\(45\%\) in the U.S. and \(60\%\)--\(65\%\) in Japan, indicating that U.S. political answers are at higher risk of poisoning attacks. We also find that sources with low content-injection barriers are frequently cited yet are poorly reflected in answer content. To mitigate this threat, we discuss how publishers of primary sources can increase exposure of their web content in answers and show that well-known techniques are limited by language differences.
>
---
#### [replaced 148] XISM: an eXploratory and Interactive Graph Tool to Visualize and Evaluate Semantic Map Models
- **分类: cs.CL**

- **简介: 该论文提出XISM，解决语义地图构建中可扩展性与可解释性的矛盾，结合数据驱动与专家知识，实现交互式可视化与优化。**

- **链接: [https://arxiv.org/pdf/2507.04070](https://arxiv.org/pdf/2507.04070)**

> **作者:** Zhu Liu; Zhen Hu; Lei Dai; Yu Xuan; Ying Liu
>
> **备注:** Paper under review
>
> **摘要:** Semantic map models visualize systematic relations among semantic functions through graph structures and are widely used in linguistic typology. However, existing construction methods either depend on labor-intensive expert reasoning or on fully automated systems lacking expert involvement, creating a tension between scalability and interpretability. We introduce \textbf{XISM}, an interactive system that combines data-driven inference with expert knowledge. XISM generates candidate maps via a top-down procedure and allows users to iteratively refine edges in a visual interface, with real-time metric feedback. Experiments in three semantic domains and expert interviews show that XISM improves linguistic decision transparency and controllability in semantic-map construction while maintaining computational efficiency. XISM provides a collaborative approach for scalable and interpretable semantic-map building. The system\footnote{this https URL} , source code\footnote{this https URL} , and demonstration video\footnote{this https URL} are publicly available.
>
---
#### [replaced 149] VeriTrail: Closed-Domain Hallucination Detection with Traceability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息验证任务，旨在解决多步骤生成过程中的封闭域幻觉问题。提出VeriTrail方法，实现幻觉内容的溯源与真实性检测。**

- **链接: [https://arxiv.org/pdf/2505.21786](https://arxiv.org/pdf/2505.21786)**

> **作者:** Dasha Metropolitansky; Jonathan Larson
>
> **备注:** ICLR 2026
>
> **摘要:** Even when instructed to adhere to source material, language models often generate unsubstantiated content - a phenomenon known as "closed-domain hallucination." This risk is amplified in processes with multiple generative steps (MGS), compared to processes with a single generative step (SGS). However, due to the greater complexity of MGS processes, we argue that detecting hallucinations in their final outputs is necessary but not sufficient: it is equally important to trace where hallucinated content was likely introduced and how faithful content may have been derived from the source material through intermediate outputs. To address this need, we present VeriTrail, the first closed-domain hallucination detection method designed to provide traceability for both MGS and SGS processes. We also introduce the first datasets to include all intermediate outputs as well as human annotations of final outputs' faithfulness for their respective MGS processes. We demonstrate that VeriTrail outperforms baseline methods on both datasets.
>
---
#### [replaced 150] Cognitive models can reveal interpretable value trade-offs in language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐研究，旨在解决如何解释模型中的价值权衡问题。通过引入认知模型，分析模型在不同设置下的行为表现，以揭示其价值倾向和调整机制。**

- **链接: [https://arxiv.org/pdf/2506.20666](https://arxiv.org/pdf/2506.20666)**

> **作者:** Sonia K. Murthy; Rosie Zhao; Jennifer Hu; Sham Kakade; Markus Wulfmeier; Peng Qian; Tomer Ullman
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in language models are limited. In cognitive science, so-called "cognitive models" provide formal accounts of such trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. Here, we show that a leading cognitive model of polite speech can be used to systematically evaluate alignment-relevant trade-offs in language models via two encompassing settings: degrees of reasoning "effort" and system prompt manipulations in closed-source frontier models, and RL post-training dynamics of open-source models. Our results show that LLMs' behavioral profiles under the cognitive model a) shift predictably when they are prompted to prioritize certain goals, b) are amplified by a small reasoning budget, and c) can be used to diagnose other social behaviors such as sycophancy. Our findings from LLMs' post-training dynamics reveal large shifts in values early on in training and persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. Our framework offers a flexible tool for probing behavioral profiles across diverse model types and gaining insights for shaping training regimes that better control trade-offs between values during model development.
>
---
#### [replaced 151] GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics
- **分类: cs.SE; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出GateLens，用于汽车软件发布分析的LLM代理，解决复杂数据解析与推理问题。通过关系代数实现自然语言到代码的可靠转换，提升分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.21735](https://arxiv.org/pdf/2503.21735)**

> **作者:** Arsham Gholamzadeh Khoee; Shuai Wang; Yinan Yu; Robert Feldt; Dhasarathy Parthasarathy
>
> **摘要:** Ensuring reliable data-driven decisions is crucial in domains where analytical accuracy directly impacts safety, compliance, or operational outcomes. Decision support in such domains relies on large tabular datasets, where manual analysis is slow, costly, and error-prone. While Large Language Models (LLMs) offer promising automation potential, they face challenges in analytical reasoning, structured data handling, and ambiguity resolution. This paper introduces GateLens, an LLM-based architecture for reliable analysis of complex tabular data. Its key innovation is the use of Relational Algebra (RA) as a formal intermediate representation between natural-language reasoning and executable code, addressing the reasoning-to-code gap that can arise in direct generation approaches. In our automotive instantiation, GateLens translates natural language queries into RA expressions and generates optimized Python code. Unlike traditional multi-agent or planning-based systems that can be slow, opaque, and costly to maintain, GateLens emphasizes speed, transparency, and reliability. We validate the architecture in automotive software release analytics, where experimental results show that GateLens outperforms the existing Chain-of-Thought (CoT) + Self-Consistency (SC) based system on real-world datasets, particularly in handling complex and ambiguous queries. Ablation studies confirm the essential role of the RA layer. Industrial deployment demonstrates over 80% reduction in analysis time while maintaining high accuracy across domain-specific tasks. GateLens operates effectively in zero-shot settings without requiring few-shot examples or agent orchestration. This work advances deployable LLM system design by identifying key architectural features--intermediate formal representations, execution efficiency, and low configuration overhead--crucial for domain-specific analytical applications.
>
---
#### [replaced 152] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习领域，旨在解决大语言模型中离策略RL的理论问题。通过分析GRPO，揭示其本质为离策略算法，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2509.24203](https://arxiv.org/pdf/2509.24203)**

> **作者:** Chaorui Yao; Yanxi Chen; Yuchang Sun; Yushuo Chen; Wenhao Zhang; Xuchen Pan; Yaliang Li; Bolin Ding
>
> **备注:** Accepted to ICLR 2026. arXiv v2 update: add references and experiments
>
> **摘要:** Off-policy reinforcement learning (RL) for large language models (LLMs) is attracting growing interest, driven by practical constraints in real-world applications, the complexity of LLM-RL infrastructure, and the need for further innovations of RL methodologies. While classic REINFORCE and its modern variants like Group Relative Policy Optimization (GRPO) are typically regarded as on-policy algorithms with limited tolerance of off-policyness, we present in this work a first-principles derivation for group-relative REINFORCE -- a REINFORCE variant that uses the within-group mean reward as the baseline for advantage calculation -- without assuming a specific training data distribution, showing that it admits a native off-policy interpretation. This perspective yields two general principles for adapting REINFORCE to truly off-policy settings: regularizing policy updates, and actively shaping the data distribution. Our analysis demystifies some myths about the roles of importance sampling and clipping in GRPO, unifies and reinterprets two recent algorithms -- Online Policy Mirror Descent and Asymmetric REINFORCE -- as regularized forms of the REINFORCE loss, and offers theoretical justification for seemingly heuristic data-weighting strategies. Our findings lead to actionable insights that are validated with extensive empirical studies, and open up new opportunities for principled algorithm design in off-policy RL for LLMs. Source code for this work is available at this https URL.
>
---
#### [replaced 153] OmniGAIA: Towards Native Omni-Modal AI Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出OmniGAIA基准和OmniAtlas模型，解决多模态AI助手在复杂推理与工具使用上的不足，旨在提升真实场景下的智能交互能力。**

- **链接: [https://arxiv.org/pdf/2602.22897](https://arxiv.org/pdf/2602.22897)**

> **作者:** Xiaoxi Li; Wenxiang Jiao; Jiarui Jin; Shijian Wang; Guanting Dong; Jiajie Jin; Hao Wang; Yinuo Wang; Ji-Rong Wen; Yuan Lu; Zhicheng Dou
>
> **摘要:** Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.
>
---
#### [replaced 154] RPM: Reasoning-Level Personalization for Black-Box Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于黑盒大语言模型个性化任务，解决用户偏好与模型输出不匹配的问题。提出RPM框架，通过分析用户行为数据构建个性化推理路径，提升模型的个性化和可解释性。**

- **链接: [https://arxiv.org/pdf/2505.21082](https://arxiv.org/pdf/2505.21082)**

> **作者:** Jieyong Kim; Tongyoung Kim; Soojin Yoon; Jaehyung Kim; Dongha Lee
>
> **摘要:** While black-box large language models are widely deployed, they produce generic outputs that overlook individual user preferences. Current personalization methods are fundamentally limited to response-level personalization; they only match final outputs, failing to model the underlying reasoning that connects user behavior to responses. To address this, this work introduces reasoning-level personalization as a new paradigm and proposes RPM, the first systematic framework that automatically discovers user-specific reasoning structures from raw behavioral data to guide the model's personalized inference. RPM constructs a structured model of user behavior-built from response-influential features and statistical factors-to create personalized reasoning paths and retrieve beneficial examples for guiding inference through a feature-based retrieval mechanism. Extensive experiments across four diverse tasks demonstrate that RPM consistently outperforms existing response-level methods while simultaneously enhancing both personalization performance and interpretability, providing a promising direction for black-box LLM personalization.
>
---
#### [replaced 155] RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决CUA在混合web-OS环境中的对抗测试问题。提出RedTeamCUA框架，构建混合沙箱进行真实场景测试，发现CUA存在显著安全漏洞。**

- **链接: [https://arxiv.org/pdf/2505.21936](https://arxiv.org/pdf/2505.21936)**

> **作者:** Zeyi Liao; Jaylen Jones; Linxi Jiang; Yuting Ning; Eric Fosler-Lussier; Yu Su; Zhiqiang Lin; Huan Sun
>
> **备注:** ICLR 2026 (Oral)
>
> **摘要:** Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning high ASRs in realistic end-to-end settings, with the strongest-to-date Claude 4.5 Sonnet | CUA exhibiting the highest ASR of 60%, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.
>
---
#### [replaced 156] How Do LLMs Use Their Depth?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM的深度使用机制，旨在揭示其层间计算动态。通过分析中间表示，提出“Guess-then-Refine”框架，说明早期层依赖高频词猜测，后期层进行上下文修正。**

- **链接: [https://arxiv.org/pdf/2510.18871](https://arxiv.org/pdf/2510.18871)**

> **作者:** Akshat Gupta; Jay Yeung; Gopala Anumanchipalli; Anna Ivanova
>
> **摘要:** Growing evidence suggests that large language models do not use their depth uniformly, yet we still lack a fine-grained understanding of their layer-wise prediction dynamics. In this paper, we trace the intermediate representations of several open-weight models during inference and reveal a structured and nuanced use of depth. Specifically, we propose a "Guess-then-Refine" framework that explains how LLMs internally structure their computations to make predictions. We first show that the top-ranked predictions in early LLM layers are composed primarily of high-frequency tokens, which act as statistical guesses proposed by the model due to the lack of contextual information. As contextual information develops deeper into the model, these initial guesses get refined into contextually appropriate tokens. We then examine the dynamic usage of layer depth through three case studies. (i) Multiple-choice task analysis shows that the model identifies appropriate options within the first half of the model and finalizes the response in the latter half. (ii) Fact recall task analysis shows that in a multi-token answer, the first token requires more computational depth than the rest. (iii) Part-of-speech analysis shows that function words are, on average, the earliest to be predicted correctly. To validate our results, we supplement probe-based analyses with causal manipulations in the form of activation patching and early-exiting experiments. Together, our results provide a detailed view of depth usage in LLMs, shedding light on the layer-by-layer computations that underlie successful predictions and providing insights for future works to improve computational efficiency in transformer-based models.
>
---
#### [replaced 157] Sparse Shift Autoencoders for Identifying Concepts from Large Language Model Activations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型可解释性任务，旨在解决概念识别中的不可辨识问题。通过引入稀疏位移自编码器，学习嵌入差异的稀疏表示，实现可辨识的概念提取与控制。**

- **链接: [https://arxiv.org/pdf/2502.12179](https://arxiv.org/pdf/2502.12179)**

> **作者:** Shruti Joshi; Andrea Dittadi; Sébastien Lachapelle; Dhanya Sridhar
>
> **备注:** 27 pages, 9 figures
>
> **摘要:** Unsupervised approaches to large language model (LLM) interpretability, such as sparse autoencoders (SAEs), offer a way to decode LLM activations into interpretable and, ideally, controllable concepts. On the one hand, these approaches alleviate the need for supervision from concept labels, paired prompts, or explicit causal models. On the other hand, without additional assumptions, SAEs are not guaranteed to be identifiable. In practice, they may learn latent dimensions that entangle multiple underlying concepts. If we use these dimensions to extract vectors for steering specific LLM behaviours, this non-identifiability might result in interventions that inadvertently affect unrelated properties. In this paper, we bring the question of identifiability to the forefront of LLM interpretability research. Specifically, we introduce Sparse Shift Autoencoders (SSAEs) which learn sparse representations of differences between embeddings rather than the embeddings themselves. Crucially, we show that SSAEs are identifiable from paired observations which differ in multiple unknown concepts, but not all. With this key identifiability result, we show that we can steer single concepts with only this weak form of supervision. Finally, we empirically demonstrate identifiable concept recovery across multiple real-world language datasets by disentangling activations from different LLMs.
>
---
#### [replaced 158] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于医疗领域中的麻醉学推理任务，旨在提升大语言模型在该领域的推理能力。提出AnesSuite数据集和基准，以及基线模型Morpheus，通过多种训练方法优化模型表现。**

- **链接: [https://arxiv.org/pdf/2504.02404](https://arxiv.org/pdf/2504.02404)**

> **作者:** Xiang Feng; Wentao Jiang; Zengmao Wang; Yong Luo; Pingbo Xu; Baosheng Yu; Hua Jin; Jing Zhang
>
> **备注:** Accepted in ICLR 2026; 47 pages, 12 figures, 26 tables;
>
> **摘要:** The application of large language models (LLMs) in the medical field has garnered significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. To bridge this gap, we introduce AnesSuite, the first comprehensive dataset suite specifically designed for anesthesiology reasoning in LLMs. The suite features AnesBench, an evaluation benchmark tailored to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Alongside this benchmark, the suite includes three training datasets that provide an infrastructure for continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning with verifiable rewards (RLVR). Leveraging this suite, we develop Morpheus, the first baseline model collection for anesthesiology reasoning. Despite undergoing limited training with SFT and group relative policy optimization (GRPO), Morpheus not only achieves substantial improvements in anesthesiology that rival larger-scale models, but also demonstrates enhanced reasoning capabilities across general medical and broad-domain benchmarks. Furthermore, through comprehensive evaluations and experiments, we analyze the key factors influencing anesthesiology reasoning performance, including model characteristics, training strategies and training data. Both AnesSuite and Morpheus will be open-sourced at this https URL.
>
---
#### [replaced 159] BinaryShield: Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出BinaryShield，解决LLM服务间威胁情报共享问题。通过隐私保护指纹技术，在不泄露用户信息的前提下，实现攻击模式的高效检测与共享。**

- **链接: [https://arxiv.org/pdf/2509.05608](https://arxiv.org/pdf/2509.05608)**

> **作者:** Waris Gill; Natalie Isak; Matthew Dressman
>
> **备注:** Accepted at the 2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)
>
> **摘要:** The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries. We present BinaryShield, \emph{the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries.} BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate privacy-preserving fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving storage reduction and 38x faster similarity search compared to dense embeddings.
>
---
#### [replaced 160] Soft-Masked Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型任务，旨在解决传统扩散模型中二值掩码信息丢失的问题。通过引入软掩码方法，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2510.17206](https://arxiv.org/pdf/2510.17206)**

> **作者:** Michael Hersche; Samuel Moor-Smith; Thomas Hofmann; Abbas Rahimi
>
> **备注:** Accepted at the Fourteenth International Conference on Learning Representations (ICLR2026)
>
> **摘要:** Diffusion models have demonstrated strong potential in language modeling, offering various advantages over traditional autoregressive approaches. Their ability to generate and revise entire responses in parallel enables faster generation and built-in self-correction mechanisms. Most modern diffusion-based language models employ masked diffusion, where decoding involves iteratively processing masked tokens based on a binary decision: either retaining the mask or replacing it with the predicted token. However, this binary choice discards valuable predictive information when the mask is retained. To address this limitation, we introduce soft-masking (SM), a novel method that dynamically blends the embedding of the mask token with the embeddings of the top-k predicted tokens from the previous decoding step, for each retained mask. This provides the model with a more informative prior, preserving context from earlier computations and allowing partial information about masked tokens to propagate beyond a single step. We propose a training methodology that efficiently adapts masked diffusion language models to incorporate SM. We demonstrate that training a 169M parameter model from scratch with SM yields superior perplexity and MAUVE scores compared to binary masking baselines. Similarly, a pretrained model can be enhanced with SM through continued pretraining. Finally, we finetune two state-of-the-art diffusion models, Dream-7B and Dream-Coder-7B, with SM. SM consistently improves performance across multiple coding benchmarks, particularly in high-throughput settings. The code is available at this https URL.
>
---
#### [replaced 161] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究LLM多智能体系统中的集体记忆偏差问题，探讨曼德拉效应的成因与应对策略，属于人工智能伦理与协作系统研究任务。**

- **链接: [https://arxiv.org/pdf/2602.00428](https://arxiv.org/pdf/2602.00428)**

> **作者:** Naen Xu; Hengyu An; Shuo Shi; Jinghuai Zhang; Chunyi Zhou; Changjiang Li; Tianyu Du; Zhihui Fu; Jun Wang; Shouling Ji
>
> **备注:** ICLR 2026
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly enhanced the capabilities of collaborative multi-agent systems, enabling them to address complex challenges. However, within these multi-agent systems, the susceptibility of agents to collective cognitive biases remains an underexplored issue. A compelling example is the Mandela effect, a phenomenon where groups collectively misremember past events as a result of false details reinforced through social influence and internalized misinformation. This vulnerability limits our understanding of memory bias in multi-agent systems and raises ethical concerns about the potential spread of misinformation. In this paper, we conduct a comprehensive study on the Mandela effect in LLM-based multi-agent systems, focusing on its existence, causing factors, and mitigation strategies. We propose MANBENCH, a novel benchmark designed to evaluate agent behaviors across four common task types that are susceptible to the Mandela effect, using five interaction protocols that vary in agent roles and memory timescales. We evaluate agents powered by several LLMs on MANBENCH to quantify the Mandela effect and analyze how different factors affect it. Moreover, we propose strategies to mitigate this effect, including prompt-level defenses (e.g., cognitive anchoring and source scrutiny) and model-level alignment-based defense, achieving an average 74.40% reduction in the Mandela effect compared to the baseline. Our findings provide valuable insights for developing more resilient and ethically aligned collaborative multi-agent systems. Code and dataset are available at this https URL.
>
---
#### [replaced 162] Post-training Large Language Models for Diverse High-Quality Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决大语言模型输出多样性不足的问题。通过DQO方法，在保持质量的同时提升语义多样性。**

- **链接: [https://arxiv.org/pdf/2509.04784](https://arxiv.org/pdf/2509.04784)**

> **作者:** Yilei Chen; Souradip Chakraborty; Lorenz Wolf; Yannis Paschalidis; Aldo Pacchiano
>
> **备注:** ICLR 2026
>
> **摘要:** Reinforcement learning (RL) has emerged as a popular method for post-training large language models (LLMs). While improving the model's performance on downstream tasks, it often reduces the model's output diversity, leading to narrow, canonical responses. Existing methods to enhance diversity are limited, either by operating at inference time or by focusing on surface-level differences. We propose a novel training method named DQO (Diversity Quality Optimization) based on determinantal point processes (DPPs) to jointly optimize LLMs for quality and semantic diversity. Our approach samples and embeds a group of responses for each prompt, then uses the determinant of a kernel-based similarity matrix to measure diversity as the volume spanned by the embeddings of these responses. DQO is flexible and can be applied on top of existing RL algorithms. Experiments across instruction-following, summarization, story generation, and reasoning tasks demonstrate that our method substantially improves semantic diversity without sacrificing model quality.
>
---
#### [replaced 163] Revisiting Self-Play Preference Optimization: On the Role of Prompt Difficulty
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，研究自博弈偏好优化中提示难度的影响。工作包括分析不同难度提示对训练效果的影响，并提出通过选择易提示提升性能的策略。**

- **链接: [https://arxiv.org/pdf/2510.05534](https://arxiv.org/pdf/2510.05534)**

> **作者:** Yao Xiao; Jung-jae Kim; Roy Ka-wei Lee; Lidong Bing
>
> **摘要:** Self-play preference optimization has emerged as a prominent paradigm for aligning large language models (LLMs). It typically involves a language model to generate on-policy responses for prompts and a reward model (RM) to guide the selection of chosen and rejected responses, which can be further trained with direct preference optimization (DPO). However, the role of prompts remains underexplored, despite being a core component in this pipeline. In this work, we investigate how prompts of varying difficulty influence self-play preference optimization. We use the mean reward of sampled responses of a prompt as a proxy for its difficulty. We first find that difficult prompts exhibit substantially inferior self-play optimization performance compared to easy prompts for language models. Moreover, incorporating difficult prompts into training fails to enhance overall performance and, in fact, leads to slight degradation compared to training on easy prompts alone. Third, there is a clear upward trend in optimization performance as prompt difficulty decreases. We also observe that the performance gap between difficult and easy prompts tends to close as the model capacity increases, suggesting that prompt difficulty interacts with the model capacity. Building on these findings, we explore strategies to mitigate the adversary effect of difficult prompts on final performance. We demonstrate that only training on a small portion (30%) of the easiest prompts improves overall self-play performance on AlpacaEval~2 and Arena-Hard. We also report failed attempts and lessons learned.
>
---
#### [replaced 164] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决监督学习难以有效利用负反馈的问题。提出NFT方法，通过建模负样本实现模型自我改进，效果优于传统监督学习和部分强化学习方法。**

- **链接: [https://arxiv.org/pdf/2505.18116](https://arxiv.org/pdf/2505.18116)**

> **作者:** Huayu Chen; Kaiwen Zheng; Qinsheng Zhang; Ganqu Cui; Lifan Yuan; Yin Cui; Haotian Ye; Tsung-Yi Lin; Ming-Yu Liu; Jun Zhu; Haoxiang Wang
>
> **摘要:** Reinforcement Learning (RL) has played a central role in the recent surge of LLMs' math abilities by enabling self-improvement through binary verifier signals. In contrast, Supervised Learning (SL) is rarely considered for such verification-driven training, largely due to its heavy reliance on reference answers and inability to reflect on mistakes. In this work, we challenge the prevailing notion that self-improvement is exclusive to RL and propose Negative-aware Fine-Tuning (NFT) -- a supervised approach that enables LLMs to reflect on their failures and improve autonomously with no external teachers. In online training, instead of throwing away self-generated negative answers, NFT constructs an implicit negative policy to model them. This implicit policy is parameterized with the same positive LLM we target to optimize on positive data, enabling direct policy optimization on all LLMs' generations. We conduct experiments on 7B and 32B models in math reasoning tasks. Results consistently show that through the additional leverage of negative feedback, NFT significantly improves over SL baselines like Rejection sampling Fine-Tuning, matching or even surpassing leading RL algorithms like GRPO and DAPO. Furthermore, we demonstrate that NFT and GRPO are actually equivalent in strict-on-policy training, even though they originate from entirely different theoretical foundations. Our experiments and theoretical findings bridge the gap between SL and RL methods in binary-feedback learning systems.
>
---
#### [replaced 165] Dynamic Token Reweighting for Robust Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型安全任务，旨在解决多模态越狱攻击问题。通过优化KV缓存，动态调整视觉令牌权重，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2505.17132](https://arxiv.org/pdf/2505.17132)**

> **作者:** Tanqiu Jiang; Jiacheng Liang; Rongyi Zhu; Jiawei Zhou; Fenglong Ma; Ting Wang
>
> **备注:** CVPR 2026
>
> **摘要:** Large vision-language models (VLMs) are highly vulnerable to multimodal jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that DTR outperforms existing defenses in both attack robustness and benign-task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available at: this https URL.
>
---
#### [replaced 166] Mitigating Structural Noise in Low-Resource S2TT: An Optimized Cascaded Nepali-English Pipeline with Punctuation Restoration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于低资源语言语音翻译任务，旨在解决ASR阶段标点丢失导致的结构噪声问题。通过优化级联管道和引入标点恢复模块，提升了翻译质量。**

- **链接: [https://arxiv.org/pdf/2602.21647](https://arxiv.org/pdf/2602.21647)**

> **作者:** Tangsang Chongbang; Pranesh Pyara Shrestha; Amrit Sarki; Anku Jaiswal
>
> **备注:** 16 pages, 4 figures, 12 tables, Transactions on Asian and Low-Resource Language Information Processing (Under Review)
>
> **摘要:** Cascaded speech-to-text translation (S2TT) systems for low-resource languages can suffer from structural noise, particularly the loss of punctuation during the Automatic Speech Recognition (ASR) phase. This research investigates the impact of such noise on Nepali-to-English translation and proposes an optimized pipeline to mitigate quality degradation. We first establish highly proficient ASR and NMT components: a Wav2Vec2-XLS-R-300m model achieved a state-of-the-art 2.72% CER on OpenSLR-54, and a multi-stage fine-tuned MarianMT model reached a 28.32 BLEU score on the FLORES-200 benchmark. We empirically investigate the influence of punctuation loss, demonstrating that unpunctuated ASR output significantly degrades translation quality, causing a massive 20.7% relative BLEU drop on the FLORES benchmark. To overcome this, we propose and evaluate an intermediate Punctuation Restoration Module (PRM). The final S2TT pipeline was tested across three configurations on a custom dataset. The optimal configuration, which applied the PRM directly to ASR output, achieved a 4.90 BLEU point gain over the direct ASR-to-NMT baseline (BLEU 36.38 vs. 31.48). This improvement was validated by human assessment, which confirmed the optimized pipeline's superior Adequacy (3.673) and Fluency (3.804) with inter-rater reliability (Krippendorff's ${\alpha} {\geq}$ 0.723). This work validates that targeted punctuation restoration is the most effective intervention for mitigating structural noise in the Nepali S2TT pipeline. It establishes an optimized baseline and demonstrates a critical architectural insight for developing cascaded speech translation systems for similar low-resource languages.
>
---
#### [replaced 167] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?
- **分类: cs.CL**

- **简介: 该论文研究RPT在不同领域的泛化能力，旨在解决LLM推理能力迁移问题。通过对比实验，发现RPT在相似任务有效，但在不同领域效果不稳定。**

- **链接: [https://arxiv.org/pdf/2506.19733](https://arxiv.org/pdf/2506.19733)**

> **作者:** Chuxuan Hu; Yuxuan Zhu; Antony Kellermann; Caleb Biddulph; Suppakit Waiwitlikhit; Jason Benn; Daniel Kang
>
> **备注:** ICLR 2026; 9 pages, 4 figures, 2 tables
>
> **摘要:** Reinforcement post training (RPT) has recently shown promise in improving the reasoning abilities of large language models (LLMs). However, it remains unclear how well these improvements generalize to new domains, as prior work evaluates RPT models on data from the same domains used for post-training. To understand the generalizability of RPT, we conduct two studies with specific focus on Reinforcement Learning with Verifiable Rewards (RLVR). (1) Observational: we compare a wide range of open-weight RPT models against their corresponding base models across multiple domains, including both seen and unseen domains in their fine-tuning data. (2) Interventional: we fine-tune LLMs with RPT on single domains and evaluate their performance across multiple domains. Both studies converge on the same conclusion that, although RPT brings substantial gains on tasks similar to the fine-tuning data, the gains generalize inconsistently and can vanish on domains with different reasoning patterns.
>
---
#### [replaced 168] How Many Code and Test Cases Are Enough? Evaluating Test Cases Generation from a Binary-Matrix Perspective
- **分类: cs.CL**

- **简介: 该论文属于测试用例生成评估任务，旨在解决现有基准评估不准确的问题。通过构建二进制矩阵模型，提出方法以确定最小错误代码和测试用例集合，提升评估有效性。**

- **链接: [https://arxiv.org/pdf/2510.08720](https://arxiv.org/pdf/2510.08720)**

> **作者:** Xianzhen Luo; Jinyang Huang; Wenzhen Zheng; Qingfu Zhu; Mingzheng Xu; Yiheng Xu; Yuantao Fan; Wanxiang Che
>
> **备注:** Accepted by ICLR2026
>
> **摘要:** Evaluating test cases automatically generated by Large Language Models (LLMs) is a critical yet challenging task. Existing benchmarks often evaluate the exclusion ratio on large, unstructured collections of wrong codes, suffering from high computational costs and score inflation. Furthermore, they inadvertently reward generators that detect common, trivial bugs, while failing to penalize their inability to identify rare yet critical faults. In this work, we connect two fundamental questions: (1) What is the minimal set of wrong codes sufficient to represent the entire error space? and (2) What is the minimal set of test cases needed to distinguish them? We introduce a novel framework that formalizes benchmark construction as finding an optimal diagnostic basis in a binary code-test matrix, where rows represent wrong codes and columns represent test case results. The rank of this matrix specifies the minimal number of independent error patterns (wrong codes) and provides a tight upper bound on the number of test cases required for complete fault coverage. Our objective is to identify a basis of size equal to the matrix rank that maximizes internal diversity. To tackle this NP-hard problem, we propose WrongSelect, an efficient approximation algorithm to select maximally diverse wrong codes. Applying this framework to millions of competitive programming submissions, we construct TC-Bench, a compact, diverse, and inflation-resistant benchmark. Extensive experiments show that even the most advanced test case generation methods achieve only ~60% exclusion rates on TC-Bench, exposing a significant gap in their diagnostic power and highlighting substantial room for future improvement. Our dataset is available at: this https URL and our code is at: this https URL.
>
---
#### [replaced 169] Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床决策支持任务，旨在解决LLM在临床应用中的交互性与效率问题。提出LA-CDM模型，通过强化学习提升诊断准确性与效率。**

- **链接: [https://arxiv.org/pdf/2506.13474](https://arxiv.org/pdf/2506.13474)**

> **作者:** David Bani-Harouni; Chantal Pellegrini; Ege Özsoy; Nassir Navab; Matthias Keicher
>
> **摘要:** Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency.
>
---
#### [replaced 170] SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的社会推理任务，旨在解决LLMs在显性与隐性理论心智能力上的差距问题。工作包括构建SimpleToM基准，评估模型在不同层次的理论心智推理表现。**

- **链接: [https://arxiv.org/pdf/2410.13648](https://arxiv.org/pdf/2410.13648)**

> **作者:** Yuling Gu; Oyvind Tafjord; Hyunwoo Kim; Jared Moore; Ronan Le Bras; Peter Clark; Yejin Choi
>
> **备注:** ICLR 2026
>
> **摘要:** Large language models (LLMs) are increasingly tested for a "Theory of Mind" (ToM) - the ability to attribute mental states to oneself and others. Yet most evaluations stop at explicit belief attribution in classical toy stories or stylized tasks, leaving open the questions of whether LLMs can implicitly apply such knowledge to predict human behavior, or to judge an observed behavior, in diverse scenarios. We introduce SimpleToM, a benchmark that advances ToM evaluation along two novel axes. First, it probes multiple levels of ToM reasoning, from mental state inference (explicit ToM) to behavior prediction and judgment (applied ToM). Second, it situates these tasks in diverse, everyday scenarios - such as supermarkets, hospitals, schools, and offices - where information asymmetries naturally arise (e.g., hidden defects in grocery store items, incomplete information in provider-patient interactions, or restricted access to locked devices). SimpleToM contains concise stories (e.g., "The can of Pringles has moldy chips in it. Mary picks up the can in the supermarket and walks to the cashier."), each with three questions that test different degrees of ToM reasoning, asking models to predict: (a) mental states ("Is Mary aware of the mold?"), (b) behaviors ("Will Mary pay for the chips or report the mold?"), and (c) judgments ("Mary paid for the chips. Was that reasonable?"). Experiments reveal a striking gap: state-of-the-art models often reliably infer mental state (a), but fail at applying knowledge about the mental state for secondary predictions, with performance dropping sharply for behavior prediction (b) and further for behavior judgment (c). This exposes a critical fragility in LLMs' social reasoning in terms of what they know (explicit ToM) versus how well they can implicitly apply that knowledge for predictions (applied ToM).
>
---
#### [replaced 171] T*: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出T*，用于masked diffusion language models的渐进式块大小扩展，解决高效解码与性能平衡问题，通过TraceRL方法实现平滑过渡。**

- **链接: [https://arxiv.org/pdf/2601.11214](https://arxiv.org/pdf/2601.11214)**

> **作者:** Hanchen Xia; Baoyou Chen; Yutang Ge; Guojiang Zhao; Siyu Zhu
>
> **摘要:** We present T*, a simple TraceRL-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T* transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T* may actually converge to an alternative decoding schedule that achieves comparable performance.
>
---
#### [replaced 172] SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SwiReasoning框架，解决LLM在无需训练情况下的推理效率与准确性问题，通过动态切换显式与隐式推理提升效果。**

- **链接: [https://arxiv.org/pdf/2510.05069](https://arxiv.org/pdf/2510.05069)**

> **作者:** Dachuan Shi; Abedelkadir Asi; Keying Li; Xiangchi Yuan; Leyan Pan; Wenke Lee; Wen Xiao
>
> **备注:** ICLR 2026. Code: this https URL, Website: this https URL
>
> **摘要:** Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics, STEM, coding, and general benchmarks, SwiReasoning consistently improves average accuracy by 1.8%-3.1% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 57%-79%, with larger gains as budgets tighten.
>
---
#### [replaced 173] From Efficiency to Adaptivity: A Deeper Look at Adaptive Reasoning in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理效率与适应性不足的问题。通过提出自适应推理框架，分类现有方法并分析其优劣。**

- **链接: [https://arxiv.org/pdf/2511.10788](https://arxiv.org/pdf/2511.10788)**

> **作者:** Chao Wu; Baoheng Li; Mingchen Gao; Yu Tian; Zhenyi Wang
>
> **摘要:** Recent advances in large language models (LLMs) have made reasoning a central benchmark for evaluating intelligence. While prior surveys focus on efficiency by examining how to shorten reasoning chains or reduce computation, this view overlooks a fundamental challenge: current LLMs apply uniform reasoning strategies regardless of task complexity, generating long traces for trivial problems while failing to extend reasoning for difficult tasks. This survey reframes reasoning through the lens of {adaptivity}: the capability to allocate reasoning effort based on input characteristics such as difficulty and uncertainty. We make three contributions. First, we formalize deductive, inductive, and abductive reasoning within the LLM context, connecting these classical cognitive paradigms with their algorithmic realizations. Second, we formalize adaptive reasoning as a control-augmented policy optimization problem balancing task performance with computational cost, distinguishing learned policies from inference-time control mechanisms. Third, we propose a systematic taxonomy organizing existing methods into training-based approaches that internalize adaptivity through reinforcement learning, supervised fine-tuning, and learned controllers, and training-free approaches that achieve adaptivity through prompt conditioning, feedback-driven halting, and modular composition. This framework clarifies how different mechanisms realize adaptive reasoning in practice and enables systematic comparison across diverse strategies. We conclude by identifying open challenges in self-evaluation, meta-reasoning, and human-aligned reasoning control.
>
---
#### [replaced 174] EigenBench: A Comparative Behavioral Measure of Value Alignment
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出EigenBench，用于评估AI模型与人类价值观的对齐程度。任务是量化主观价值对齐，解决缺乏客观指标的问题。通过模型间相互评价，生成对齐得分。**

- **链接: [https://arxiv.org/pdf/2509.01938](https://arxiv.org/pdf/2509.01938)**

> **作者:** Jonathn Chang; Leonhard Piff; Suvadip Sana; Jasmine X. Li; Lionel Levine
>
> **摘要:** Aligning AI with human values is a pressing unsolved problem. To address the lack of quantitative metrics for value alignment, we propose EigenBench: a black-box method for comparatively benchmarking language models' values. Given an ensemble of models, a constitution describing a value system, and a dataset of scenarios, our method returns a vector of scores quantifying each model's alignment to the given constitution. To produce these scores, each model judges the outputs of other models across many scenarios, and these judgments are aggregated with EigenTrust (Kamvar et al., 2003), yielding scores that reflect a weighted consensus judgment of the whole ensemble. EigenBench uses no ground truth labels, as it is designed to quantify subjective traits for which reasonable judges may disagree on the correct label. Hence, to validate our method, we collect human judgments on the same ensemble of models and show that EigenBench's judgments align closely with those of human evaluators. We further demonstrate that EigenBench can recover model rankings on the GPQA benchmark without access to objective labels, supporting its viability as a framework for evaluating subjective values for which no ground truths exist. The code is available at this https URL.
>
---
#### [replaced 175] Augmenting Research Ideation with Data: An Empirical Investigation in Social Science
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 论文探讨了如何通过数据增强提升大语言模型生成研究想法的质量，解决想法可行性与有效性不足的问题。任务属于研究 ideation 辅助，通过引入元数据和自动化验证提升想法质量。**

- **链接: [https://arxiv.org/pdf/2505.21396](https://arxiv.org/pdf/2505.21396)**

> **作者:** Xiao Liu; Xinyi Dong; Xinyang Gao; Yansong Feng; Xun Pang
>
> **备注:** AI4Science Workshop at Neurips 2025 (Spotlight)
>
> **摘要:** Recent advancements in large language models (LLMs) demonstrate strong potential for generating novel research ideas, yet such ideas often struggle with feasibility and effectiveness. In this paper, we investigate whether augmenting LLMs with relevant data during the ideation process can improve idea quality. Our framework integrates data at two stages: (1) incorporating metadata during idea generation to guide models toward more feasible concepts, and (2) introducing an automated preliminary validation step during idea selection to assess the empirical plausibility of hypotheses within ideas. We evaluate our approach in the social science domain, with a specific focus on climate negotiation topics. Expert evaluation shows that metadata improves the feasibility of generated ideas by 20%, while automated validation improves the overall quality of selected ideas by 7%. Beyond assessing the quality of LLM-generated ideas, we conduct a human study to examine whether these ideas, augmented with related data and preliminary validation, can inspire researchers in their own ideation. Participants report that the LLM-generated ideas and validation are highly useful, and the ideas they propose with such support are proven to be of higher quality than those proposed without assistance. Our findings highlight the potential of data-augmented research ideation and underscore the practical value of LLM-assisted ideation in real-world academic settings.
>
---
