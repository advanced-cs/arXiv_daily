# 自然语言处理 cs.CL

- **最新发布 107 篇**

- **更新 36 篇**

## 最新发布

#### [new 001] Using LLMs for Multilingual Clinical Entity Linking to ICD-10
- **分类: cs.CL**

- **简介: 该论文提出多语言临床实体链接方法，解决自动为医疗文本分配ICD-10编码的问题。通过结合临床词典匹配与GPT-4上下文学习，实现跨语言（西班牙语、希腊语）的实体到ICD-10代码映射，提升医疗信息结构化效率。**

- **链接: [http://arxiv.org/pdf/2509.04868v1](http://arxiv.org/pdf/2509.04868v1)**

> **作者:** Sylvia Vassileva; Ivan Koychev; Svetla Boytcheva
>
> **备注:** 7 pages, 2 Figures, to be published in Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing, RANLP 2025
>
> **摘要:** The linking of clinical entities is a crucial part of extracting structured information from clinical texts. It is the process of assigning a code from a medical ontology or classification to a phrase in the text. The International Classification of Diseases - 10th revision (ICD-10) is an international standard for classifying diseases for statistical and insurance purposes. Automatically assigning the correct ICD-10 code to terms in discharge summaries will simplify the work of healthcare professionals and ensure consistent coding in hospitals. Our paper proposes an approach for linking clinical terms to ICD-10 codes in different languages using Large Language Models (LLMs). The approach consists of a multistage pipeline that uses clinical dictionaries to match unambiguous terms in the text and then applies in-context learning with GPT-4.1 to predict the ICD-10 code for the terms that do not match the dictionary. Our system shows promising results in predicting ICD-10 codes on different benchmark datasets in Spanish - 0.89 F1 for categories and 0.78 F1 on subcategories on CodiEsp, and Greek - 0.85 F1 on ElCardioCC.
>
---
#### [new 002] AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs
- **分类: cs.CL**

- **简介: 该论文提出AraHalluEval框架，针对阿拉伯语LLM在生成问答和摘要任务中的幻觉问题，设计12个细粒度指标评估12个模型，发现事实幻觉更普遍，阿拉伯预训练模型Allam表现最佳。**

- **链接: [http://arxiv.org/pdf/2509.04656v1](http://arxiv.org/pdf/2509.04656v1)**

> **作者:** Aisha Alansari; Hamzah Luqman
>
> **摘要:** Recently, extensive research on the hallucination of the large language models (LLMs) has mainly focused on the English language. Despite the growing number of multilingual and Arabic-specific LLMs, evaluating LLMs' hallucination in the Arabic context remains relatively underexplored. The knowledge gap is particularly pressing given Arabic's widespread use across many regions and its importance in global communication and media. This paper presents the first comprehensive hallucination evaluation of Arabic and multilingual LLMs on two critical Arabic natural language generation tasks: generative question answering (GQA) and summarization. This study evaluates a total of 12 LLMs, including 4 Arabic pre-trained models, 4 multilingual models, and 4 reasoning-based models. To assess the factual consistency and faithfulness of LLMs' outputs, we developed a fine-grained hallucination evaluation framework consisting of 12 fine-grained hallucination indicators that represent the varying characteristics of each task. The results reveal that factual hallucinations are more prevalent than faithfulness errors across all models and tasks. Notably, the Arabic pre-trained model Allam consistently demonstrates lower hallucination rates than multilingual models and a comparative performance with reasoning-based models. The code is available at: \href{https://github.com/aishaalansari57/AraHalluEval}{Github link}.
>
---
#### [new 003] Evaluating NL2SQL via SQL2NL
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 该论文提出SQL2NL框架，通过生成语义等价但词汇多样的查询，评估NL2SQL模型对语言变化的鲁棒性。发现现有模型在语言变异下表现显著下降，揭示需更严谨的评估框架以确保实际应用可靠性。**

- **链接: [http://arxiv.org/pdf/2509.04657v1](http://arxiv.org/pdf/2509.04657v1)**

> **作者:** Mohammadtaher Safarzadeh; Afshin Oroojlooyjadid; Dan Roth
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Robust evaluation in the presence of linguistic variation is key to understanding the generalization capabilities of Natural Language to SQL (NL2SQL) models, yet existing benchmarks rarely address this factor in a systematic or controlled manner. We propose a novel schema-aligned paraphrasing framework that leverages SQL-to-NL (SQL2NL) to automatically generate semantically equivalent, lexically diverse queries while maintaining alignment with the original schema and intent. This enables the first targeted evaluation of NL2SQL robustness to linguistic variation in isolation-distinct from prior work that primarily investigates ambiguity or schema perturbations. Our analysis reveals that state-of-the-art models are far more brittle than standard benchmarks suggest. For example, LLaMa3.3-70B exhibits a 10.23% drop in execution accuracy (from 77.11% to 66.9%) on paraphrased Spider queries, while LLaMa3.1-8B suffers an even larger drop of nearly 20% (from 62.9% to 42.5%). Smaller models (e.g., GPT-4o mini) are disproportionately affected. We also find that robustness degradation varies significantly with query complexity, dataset, and domain -- highlighting the need for evaluation frameworks that explicitly measure linguistic generalization to ensure reliable performance in real-world settings.
>
---
#### [new 004] Just-in-time and distributed task representations in language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中任务表示的形成时机与动态变化，揭示其通过局部、可转移的表示即时适应新任务的机制，区分高层类别与具体子任务表示，支持即时学习。**

- **链接: [http://arxiv.org/pdf/2509.04466v1](http://arxiv.org/pdf/2509.04466v1)**

> **作者:** Yuxuan Li; Declan Campbell; Stephanie C. Y. Chan; Andrew Kyle Lampinen
>
> **摘要:** Many of language models' impressive capabilities originate from their in-context learning: based on instructions or examples, they can infer and perform new tasks without weight updates. In this work, we investigate \emph{when} representations for new tasks are formed in language models, and \emph{how} these representations change over the course of context. We focus on ''transferrable'' task representations -- vector representations that can restore task context in another instance of the model, even without the full prompt. We show that these representations evolve in non-monotonic and sporadic ways, and are distinct from a more inert representation of high-level task categories that persists throughout the context. Specifically, models often condense multiple evidence into these transferrable task representations, which align well with the performance improvement based on more examples in the context. However, this accrual process exhibits strong locality along the sequence dimension, coming online only at certain tokens -- despite task identity being reliably decodable throughout the context. Moreover, these local but transferrable task representations tend to capture minimal ''task scopes'', such as a semantically-independent subtask, and models rely on more temporally-distributed representations to support longer and composite tasks. This two-fold locality (temporal and semantic) underscores a kind of just-in-time computational process underlying language models' ability to adapt to new evidence and learn new tasks on the fly.
>
---
#### [new 005] BEDTime: A Unified Benchmark for Automatically Describing Time Series
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出BEDTime基准，针对时间序列描述任务，定义识别、区分、生成三类子任务，统一四数据集评估13模型，发现语言模型不足，VLMs有效，多模态模型表现优于LLMs但仍有提升空间，揭示模型鲁棒性问题。**

- **链接: [http://arxiv.org/pdf/2509.05215v1](http://arxiv.org/pdf/2509.05215v1)**

> **作者:** Medhasweta Sen; Zachary Gottesman; Jiaxing Qiu; C. Bayan Bruss; Nam Nguyen; Tom Hartvigsen
>
> **摘要:** Many recent studies have proposed general-purpose foundation models designed for a variety of time series analysis tasks. While several established datasets already exist for evaluating these models, previous works frequently introduce their models in conjunction with new datasets, limiting opportunities for direct, independent comparisons and obscuring insights into the relative strengths of different methods. Additionally, prior evaluations often cover numerous tasks simultaneously, assessing a broad range of model abilities without clearly pinpointing which capabilities contribute to overall performance. To address these gaps, we formalize and evaluate 3 tasks that test a model's ability to describe time series using generic natural language: (1) recognition (True/False question-answering), (2) differentiation (multiple choice question-answering), and (3) generation (open-ended natural language description). We then unify 4 recent datasets to enable head-to-head model comparisons on each task. Experimentally, in evaluating 13 state-of-the-art language, vision--language, and time series--language models, we find that (1) popular language-only methods largely underperform, indicating a need for time series-specific architectures, (2) VLMs are quite successful, as expected, identifying the value of vision models for these tasks and (3) pretrained multimodal time series--language models successfully outperform LLMs, but still have significant room for improvement. We also find that all approaches exhibit clear fragility in a range of robustness tests. Overall, our benchmark provides a standardized evaluation on a task necessary for time series reasoning systems.
>
---
#### [new 006] Elucidating the Design Space of Decay in Linear Attention
- **分类: cs.CL**

- **简介: 该论文研究线性注意力中的衰减机制设计，旨在优化序列模型性能。通过分析参数化策略、参数共享、衰减粒度及与RoPE的兼容性，揭示关键设计原则，提出有效配置方法。**

- **链接: [http://arxiv.org/pdf/2509.05282v1](http://arxiv.org/pdf/2509.05282v1)**

> **作者:** Zhen Qin; Xuyang Shen; Yiran Zhong
>
> **备注:** Accepted to COLM 2025. Yiran Zhong is the corresponding author. Code is available at https://github.com/Doraemonzzz/xmixers
>
> **摘要:** This paper presents a comprehensive investigation into the decay mechanisms inherent in linear complexity sequence models. We systematically delineate the design space of decay mechanisms across four pivotal dimensions: parameterization strategy, which refers to the computational methodology for decay; parameter sharing, which involves the utilization of supplementary parameters for decay computation; decay granularity, comparing scalar versus vector-based decay; and compatibility with relative positional encoding methods, such as Rotary Position Embedding (RoPE). Through an extensive series of experiments conducted on diverse language modeling tasks, we uncovered several critical insights. Firstly, the design of the parameterization strategy for decay requires meticulous consideration. Our findings indicate that effective configurations are typically confined to a specific range of parameters. Secondly, parameter sharing cannot be used arbitrarily, as it may cause decay values to be too large or too small, thereby significantly impacting performance. Thirdly, under identical parameterization strategies, scalar decay generally underperforms compared to its vector-based counterpart. However, in certain scenarios with alternative parameterization strategies, scalar decay may unexpectedly surpass vector decay in efficacy. Lastly, our analysis reveals that RoPE, a commonly employed relative positional encoding method, typically fails to provide tangible benefits to the majority of linear attention mechanisms.
>
---
#### [new 007] Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出首个针对LLM测试时间扩展的投机解码基准测试，解决冗余推理导致的低效问题，发现n-gram方法有效，并建议结合其他方法提升加速效果。**

- **链接: [http://arxiv.org/pdf/2509.04474v1](http://arxiv.org/pdf/2509.04474v1)**

> **作者:** Shengyin Sun; Yiming Li; Xing Li; Yingzhao Lian; Weizhe Lin; Hui-Ling Zhen; Zhiyuan Yang; Chen Chen; Xianzhi Yu; Mingxuan Yuan; Chen Ma
>
> **备注:** 18 pages
>
> **摘要:** Test-time scaling has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs) by allocating additional computational resources during inference. However, this paradigm is inherently inefficient due to the generation of redundant and repetitive reasoning traces, leading to significant computational overhead. Speculative decoding offers a promising avenue for mitigating this inefficiency, yet its efficacy in the structured, repetition-rich context of test-time scaling remains largely unexplored. To bridge this gap, we introduce the first comprehensive benchmark designed to evaluate speculative decoding methods for accelerating LLM test-time scaling. Our benchmark provides consistent experimental protocols across representative test-time scaling paradigms (e.g., Best-of-N sampling and multi-round thinking), enabling a fair comparison of three major categories of speculative decoding: model-based, training-based, and n-gram-based methods. Extensive experiments reveal that simple n-gram-based methods effectively capture repetitive patterns, demonstrating unique potential in accelerating test-time scaling. This phenomenon demonstrates the value of integrating n-gram-based methods with model-based or training-based approaches to balance acceleration for both repetitive and diverse reasoning in test-time scaling. We hope this benchmark spurs further research on speculative decoding for test-time scaling, enabling faster and more practical reasoning in LLMs through better handling of repetitive and diverse reasoning paths.
>
---
#### [new 008] Predicting Failures of LLMs to Link Biomedical Ontology Terms to Identifiers Evidence Across Models and Ontologies
- **分类: cs.CL; I.2**

- **简介: 论文研究大语言模型在链接生物医学本体术语到标识符时的失败原因，分析模型与本体差异，评估九个特征，发现标识符接触度是关键预测因素。**

- **链接: [http://arxiv.org/pdf/2509.04458v1](http://arxiv.org/pdf/2509.04458v1)**

> **作者:** Daniel B. Hier; Steven Keith Platt; Tayo Obafemi-Ajayi
>
> **备注:** Accepted for Presentation, IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 25), Atlanta GA USA, October 26-29, 2025
>
> **摘要:** Large language models often perform well on biomedical NLP tasks but may fail to link ontology terms to their correct identifiers. We investigate why these failures occur by analyzing predictions across two major ontologies, Human Phenotype Ontology and Gene Ontology, and two high-performing models, GPT-4o and LLaMa 3.1 405B. We evaluate nine candidate features related to term familiarity, identifier usage, morphology, and ontology structure. Univariate and multivariate analyses show that exposure to ontology identifiers is the strongest predictor of linking success.
>
---
#### [new 009] Memorization $\neq$ Understanding: Do Large Language Models Have the Ability of Scenario Cognition?
- **分类: cs.CL**

- **简介: 该论文提出双视角框架评估LLMs的场景认知能力，区分记忆与理解，设计场景数据集，实验发现LLMs主要依赖记忆，揭示语义理解局限。**

- **链接: [http://arxiv.org/pdf/2509.04866v1](http://arxiv.org/pdf/2509.04866v1)**

> **作者:** Boxiang Ma; Ru Li; Yuanlong Wang; Hongye Tan; Xiaoli Li
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Driven by vast and diverse textual data, large language models (LLMs) have demonstrated impressive performance across numerous natural language processing (NLP) tasks. Yet, a critical question persists: does their generalization arise from mere memorization of training data or from deep semantic understanding? To investigate this, we propose a bi-perspective evaluation framework to assess LLMs' scenario cognition - the ability to link semantic scenario elements with their arguments in context. Specifically, we introduce a novel scenario-based dataset comprising diverse textual descriptions of fictional facts, annotated with scenario elements. LLMs are evaluated through their capacity to answer scenario-related questions (model output perspective) and via probing their internal representations for encoded scenario elements-argument associations (internal representation perspective). Our experiments reveal that current LLMs predominantly rely on superficial memorization, failing to achieve robust semantic scenario cognition, even in simple cases. These findings expose critical limitations in LLMs' semantic understanding and offer cognitive insights for advancing their capabilities.
>
---
#### [new 010] No Clustering, No Routing: How Transformers Actually Process Rare Tokens
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型处理罕见词的机制，解决大模型罕见词预测机制不明确的问题。通过神经元分析与实验，发现罕见词处理依赖分布式、训练驱动的神经元分化，而非模块化集群或注意力路由偏好，揭示了其灵活性与资源分配特性。**

- **链接: [http://arxiv.org/pdf/2509.04479v1](http://arxiv.org/pdf/2509.04479v1)**

> **作者:** Jing Liu
>
> **摘要:** Large language models struggle with rare token prediction, yet the mechanisms driving their specialization remain unclear. Prior work identified specialized ``plateau'' neurons for rare tokens following distinctive three-regime influence patterns \cite{liu2025emergent}, but their functional organization is unknown. We investigate this through neuron influence analyses, graph-based clustering, and attention head ablations in GPT-2 XL and Pythia models. Our findings show that: (1) rare token processing requires additional plateau neurons beyond the power-law regime sufficient for common tokens, forming dual computational regimes; (2) plateau neurons are spatially distributed rather than forming modular clusters; and (3) attention mechanisms exhibit no preferential routing to specialists. These results demonstrate that rare token specialization arises through distributed, training-driven differentiation rather than architectural modularity, preserving context-sensitive flexibility while achieving adaptive capacity allocation.
>
---
#### [new 011] Can Multiple Responses from an LLM Reveal the Sources of Its Uncertainty?
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出通过分析LLM多个响应的分歧模式，诊断其不确定性的根源（如输入歧义或知识缺失），并在多个数据集上验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.04464v1](http://arxiv.org/pdf/2509.04464v1)**

> **作者:** Yang Nan; Pengfei He; Ravi Tandon; Han Xu
>
> **备注:** Proceedings of The 2025 Conference on Empirical Methods in Natural Language Processing (Findings)
>
> **摘要:** Large language models (LLMs) have delivered significant breakthroughs across diverse domains but can still produce unreliable or misleading outputs, posing critical challenges for real-world applications. While many recent studies focus on quantifying model uncertainty, relatively little work has been devoted to \textit{diagnosing the source of uncertainty}. In this study, we show that, when an LLM is uncertain, the patterns of disagreement among its multiple generated responses contain rich clues about the underlying cause of uncertainty. To illustrate this point, we collect multiple responses from a target LLM and employ an auxiliary LLM to analyze their patterns of disagreement. The auxiliary model is tasked to reason about the likely source of uncertainty, such as whether it stems from ambiguity in the input question, a lack of relevant knowledge, or both. In cases involving knowledge gaps, the auxiliary model also identifies the specific missing facts or concepts contributing to the uncertainty. In our experiment, we validate our framework on AmbigQA, OpenBookQA, and MMLU-Pro, confirming its generality in diagnosing distinct uncertainty sources. Such diagnosis shows the potential for relevant manual interventions that improve LLM performance and reliability.
>
---
#### [new 012] COCORELI: Cooperative, Compositional Reconstitution \& Execution of Language Instructions
- **分类: cs.CL; cs.AI**

- **简介: 论文提出COCORELI框架，解决LLM在复杂指令执行、幻觉和空间推理中的不足。通过整合中型LLM代理、抽象机制与对话模块，实现动态环境建模与协作任务优化，实验表明其优于单LLM和代理系统。**

- **链接: [http://arxiv.org/pdf/2509.04470v1](http://arxiv.org/pdf/2509.04470v1)**

> **作者:** Swarnadeep Bhar; Omar Naim; Eleni Metheniti; Bastien Navarri; Loïc Cabannes; Morteza Ezzabady; Nicholas Asher
>
> **备注:** 18 pages
>
> **摘要:** We present COCORELI, a hybrid agent framework designed to tackle the limitations of large language models (LLMs) in tasks requiring: following complex instructions, minimizing hallucination, and spatial reasoning. COCORELI integrates medium-sized LLM agents with novel abstraction mechanisms and a discourse module to parse instructions to in-context learn dynamic, high-level representations of the environment. Experiments on natural collaborative construction tasks show that COCORELI outperforms single-LLM CoT and agentic LLM systems, all using larger LLMs. It manages to largely avoid hallucinations, identify missing information, ask for clarifications, and update its learned objects. COCORELI's abstraction abilities extend beyond ENVIRONMENT, as shown in the ToolBench API completion task.
>
---
#### [new 013] AFD-SLU: Adaptive Feature Distillation for Spoken Language Understanding
- **分类: cs.CL**

- **简介: 该论文提出AFD-SLU框架，针对SLU任务中数据不足和模型部署成本高的问题，通过自适应特征蒸馏和动态调整机制提升轻量化模型性能。**

- **链接: [http://arxiv.org/pdf/2509.04821v1](http://arxiv.org/pdf/2509.04821v1)**

> **作者:** Yan Xie; Yibo Cui; Liang Xie; Erwei Yin
>
> **备注:** 5 pages, 1 figures
>
> **摘要:** Spoken Language Understanding (SLU) is a core component of conversational systems, enabling machines to interpret user utterances. Despite its importance, developing effective SLU systems remains challenging due to the scarcity of labeled training data and the computational burden of deploying Large Language Models (LLMs) in real-world applications. To further alleviate these issues, we propose an Adaptive Feature Distillation framework that transfers rich semantic representations from a General Text Embeddings (GTE)-based teacher model to a lightweight student model. Our method introduces a dynamic adapter equipped with a Residual Projection Neural Network (RPNN) to align heterogeneous feature spaces, and a Dynamic Distillation Coefficient (DDC) that adaptively modulates the distillation strength based on real-time feedback from intent and slot prediction performance. Experiments on the Chinese profile-based ProSLU benchmark demonstrate that AFD-SLU achieves state-of-the-art results, with 95.67% intent accuracy, 92.02% slot F1 score, and 85.50% overall accuracy.
>
---
#### [new 014] INSEva: A Comprehensive Chinese Benchmark for Large Language Models in Insurance
- **分类: cs.CL**

- **简介: 该论文提出INSEva，构建首个中文保险领域AI基准测试，解决现有评估体系对保险专业性的覆盖不足问题。通过多维分类、38704个权威案例及定制化评估方法，系统评测LLM在保险知识、复杂场景处理等维度的性能差异。**

- **链接: [http://arxiv.org/pdf/2509.04455v1](http://arxiv.org/pdf/2509.04455v1)**

> **作者:** Shisong Chen; Qian Zhu; Wenyan Yang; Chengyi Yang; Zhong Wang; Ping Wang; Xuan Lin; Bo Xu; Daqian Li; Chao Yuan; Licai Qi; Wanqing Xu; sun zhenxing; Xin Lu; Shiqiang Xiong; Chao Chen; Haixiang Hu; Yanghua Xiao
>
> **备注:** Under review
>
> **摘要:** Insurance, as a critical component of the global financial system, demands high standards of accuracy and reliability in AI applications. While existing benchmarks evaluate AI capabilities across various domains, they often fail to capture the unique characteristics and requirements of the insurance domain. To address this gap, we present INSEva, a comprehensive Chinese benchmark specifically designed for evaluating AI systems' knowledge and capabilities in insurance. INSEva features a multi-dimensional evaluation taxonomy covering business areas, task formats, difficulty levels, and cognitive-knowledge dimension, comprising 38,704 high-quality evaluation examples sourced from authoritative materials. Our benchmark implements tailored evaluation methods for assessing both faithfulness and completeness in open-ended responses. Through extensive evaluation of 8 state-of-the-art Large Language Models (LLMs), we identify significant performance variations across different dimensions. While general LLMs demonstrate basic insurance domain competency with average scores above 80, substantial gaps remain in handling complex, real-world insurance scenarios. The benchmark will be public soon.
>
---
#### [new 015] The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出自动评估同行评审效用的任务，解决如何量化审稿意见对作者的价值问题。通过定义四个关键指标，构建RevUtil数据集，并训练模型评估评论质量，实验表明微调模型在部分指标上超越人类表现。**

- **链接: [http://arxiv.org/pdf/2509.04484v1](http://arxiv.org/pdf/2509.04484v1)**

> **作者:** Abdelrahman Sadallah; Tim Baumgärtner; Iryna Gurevych; Ted Briscoe
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Providing constructive feedback to paper authors is a core component of peer review. With reviewers increasingly having less time to perform reviews, automated support systems are required to ensure high reviewing quality, thus making the feedback in reviews useful for authors. To this end, we identify four key aspects of review comments (individual points in weakness sections of reviews) that drive the utility for authors: Actionability, Grounding & Specificity, Verifiability, and Helpfulness. To enable evaluation and development of models assessing review comments, we introduce the RevUtil dataset. We collect 1,430 human-labeled review comments and scale our data with 10k synthetically labeled comments for training purposes. The synthetic data additionally contains rationales, i.e., explanations for the aspect score of a review comment. Employing the RevUtil dataset, we benchmark fine-tuned models for assessing review comments on these aspects and generating rationales. Our experiments demonstrate that these fine-tuned models achieve agreement levels with humans comparable to, and in some cases exceeding, those of powerful closed models like GPT-4o. Our analysis further reveals that machine-generated reviews generally underperform human reviews on our four aspects.
>
---
#### [new 016] Why Language Models Hallucinate
- **分类: cs.CL**

- **简介: 该论文分析语言模型幻觉成因，指出训练与评估机制奖励猜测而非承认不确定性，导致错误陈述。提出通过修改基准评分标准解决这一问题，推动更可信的AI系统。**

- **链接: [http://arxiv.org/pdf/2509.04664v1](http://arxiv.org/pdf/2509.04664v1)**

> **作者:** Adam Tauman Kalai; Ofir Nachum; Santosh S. Vempala; Edwin Zhang
>
> **摘要:** Like students facing hard exam questions, large language models sometimes guess when uncertain, producing plausible yet incorrect statements instead of admitting uncertainty. Such "hallucinations" persist even in state-of-the-art systems and undermine trust. We argue that language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty, and we analyze the statistical causes of hallucinations in the modern training pipeline. Hallucinations need not be mysterious -- they originate simply as errors in binary classification. If incorrect statements cannot be distinguished from facts, then hallucinations in pretrained language models will arise through natural statistical pressures. We then argue that hallucinations persist due to the way most evaluations are graded -- language models are optimized to be good test-takers, and guessing when uncertain improves test performance. This "epidemic" of penalizing uncertain responses can only be addressed through a socio-technical mitigation: modifying the scoring of existing benchmarks that are misaligned but dominate leaderboards, rather than introducing additional hallucination evaluations. This change may steer the field toward more trustworthy AI systems.
>
---
#### [new 017] DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DeepTRACE框架，用于审计AI系统的引用与证据可靠性，解决生成式AI过自信、弱引用等问题。通过八个维度分析，评估多模型表现，揭示其在辩论场景下的偏见与引用缺陷。**

- **链接: [http://arxiv.org/pdf/2509.04499v1](http://arxiv.org/pdf/2509.04499v1)**

> **作者:** Pranav Narayanan Venkit; Philippe Laban; Yilun Zhou; Kung-Hsiang Huang; Yixin Mao; Chien-Sheng Wu
>
> **备注:** arXiv admin note: text overlap with arXiv:2410.22349
>
> **摘要:** Generative search engines and deep research LLM agents promise trustworthy, source-grounded synthesis, yet users regularly encounter overconfidence, weak sourcing, and confusing citation practices. We introduce DeepTRACE, a novel sociotechnically grounded audit framework that turns prior community-identified failure cases into eight measurable dimensions spanning answer text, sources, and citations. DeepTRACE uses statement-level analysis (decomposition, confidence scoring) and builds citation and factual-support matrices to audit how systems reason with and attribute evidence end-to-end. Using automated extraction pipelines for popular public models (e.g., GPT-4.5/5, You.com, Perplexity, Copilot/Bing, Gemini) and an LLM-judge with validated agreement to human raters, we evaluate both web-search engines and deep-research configurations. Our findings show that generative search engines and deep research agents frequently produce one-sided, highly confident responses on debate queries and include large fractions of statements unsupported by their own listed sources. Deep-research configurations reduce overconfidence and can attain high citation thoroughness, but they remain highly one-sided on debate queries and still exhibit large fractions of unsupported statements, with citation accuracy ranging from 40--80% across systems.
>
---
#### [new 018] Research on Multi-hop Inference Optimization of LLM Based on MQUAKE Framework
- **分类: cs.CL; cs.LG**

- **简介: 本研究基于MQUAKE框架，提出多跳问题分解方法优化LLM复杂问答。通过构建单/多跳数据集，微调LLAMA3模型，验证多跳分解在训练前后的有效性，提升推理准确率。**

- **链接: [http://arxiv.org/pdf/2509.04770v1](http://arxiv.org/pdf/2509.04770v1)**

> **作者:** Zucheng Liang; Wenxin Wei; Kaijie Zhang; Hongyi Chen
>
> **摘要:** Accurately answering complex questions has consistently been a significant challenge for Large Language Models (LLMs). To address this, this paper proposes a multi-hop question decomposition method for complex questions, building upon research within the MQUAKE framework. Utilizing the LLAMA3 model, we systematically investigate the impact of multi-hop question decomposition within knowledge graphs on model comprehension and reasoning accuracy, both before and after model training. In our experiments, we systematically partitioned and converted the MQUAKE-T dataset into two distinct formats: a single-hop dataset designed for directly answering complex questions, and a multi-hop dataset constructed using the multi-hop question decomposition method. We then fine-tuned the LLAMA3 model on these datasets and conducted inference tests. Our results demonstrate that, without fine-tuning the LLM, the prediction performance based on the multi-hop question decomposition method significantly outperforms the method of directly answering complex questions. After fine-tuning using the LoRA (Low-Rank Adaptation) method, the performance of both approaches improved compared to the untrained baseline. Crucially, the method utilizing multi-hop decomposition consistently maintained its superiority. These findings validate the effectiveness of the multi-hop decomposition method both before and after training, demonstrating its capability to effectively enhance the LLM's ability to answer complex questions.
>
---
#### [new 019] Behavioral Fingerprinting of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出行为指纹框架，通过诊断提示与自动化评估分析18个模型，揭示核心能力趋同但对齐行为差异显著，表明交互特性源于对齐策略而非规模。**

- **链接: [http://arxiv.org/pdf/2509.04504v1](http://arxiv.org/pdf/2509.04504v1)**

> **作者:** Zehua Pei; Hui-Ling Zhen; Ying Zhang; Zhiyuan Yang; Xing Li; Xianzhi Yu; Mingxuan Yuan; Bei Yu
>
> **备注:** Submitted to 1st Open Conference on AI Agents for Science (agents4science 2025)
>
> **摘要:** Current benchmarks for Large Language Models (LLMs) primarily focus on performance metrics, often failing to capture the nuanced behavioral characteristics that differentiate them. This paper introduces a novel ``Behavioral Fingerprinting'' framework designed to move beyond traditional evaluation by creating a multi-faceted profile of a model's intrinsic cognitive and interactive styles. Using a curated \textit{Diagnostic Prompt Suite} and an innovative, automated evaluation pipeline where a powerful LLM acts as an impartial judge, we analyze eighteen models across capability tiers. Our results reveal a critical divergence in the LLM landscape: while core capabilities like abstract and causal reasoning are converging among top models, alignment-related behaviors such as sycophancy and semantic robustness vary dramatically. We further document a cross-model default persona clustering (ISTJ/ESTJ) that likely reflects common alignment incentives. Taken together, this suggests that a model's interactive nature is not an emergent property of its scale or reasoning power, but a direct consequence of specific, and highly variable, developer alignment strategies. Our framework provides a reproducible and scalable methodology for uncovering these deep behavioral differences. Project: https://github.com/JarvisPei/Behavioral-Fingerprinting
>
---
#### [new 020] CoCoNUTS: Concentrating on Content while Neglecting Uninformative Textual Styles for AI-Generated Peer Review Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoCoNUTS框架，解决AI生成同行评审检测中依赖风格线索导致误判的问题，通过构建细粒度数据集和多任务学习模型CoCoDet，实现更准确的内容导向检测。**

- **链接: [http://arxiv.org/pdf/2509.04460v1](http://arxiv.org/pdf/2509.04460v1)**

> **作者:** Yihan Chen; Jiawei Chen; Guozhao Mo; Xuanang Chen; Ben He; Xianpei Han; Le Sun
>
> **摘要:** The growing integration of large language models (LLMs) into the peer review process presents potential risks to the fairness and reliability of scholarly evaluation. While LLMs offer valuable assistance for reviewers with language refinement, there is growing concern over their use to generate substantive review content. Existing general AI-generated text detectors are vulnerable to paraphrasing attacks and struggle to distinguish between surface language refinement and substantial content generation, suggesting that they primarily rely on stylistic cues. When applied to peer review, this limitation can result in unfairly suspecting reviews with permissible AI-assisted language enhancement, while failing to catch deceptively humanized AI-generated reviews. To address this, we propose a paradigm shift from style-based to content-based detection. Specifically, we introduce CoCoNUTS, a content-oriented benchmark built upon a fine-grained dataset of AI-generated peer reviews, covering six distinct modes of human-AI collaboration. Furthermore, we develop CoCoDet, an AI review detector via a multi-task learning framework, designed to achieve more accurate and robust detection of AI involvement in review content. Our work offers a practical foundation for evaluating the use of LLMs in peer review, and contributes to the development of more precise, equitable, and reliable detection methods for real-world scholarly applications. Our code and data will be publicly available at https://github.com/Y1hanChen/COCONUTS.
>
---
#### [new 021] SpeechLLM: Unified Speech and Language Model for Enhanced Multi-Task Understanding in Low Resource Settings
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SpeechLLM，解决低资源下语音与语言模型整合难题，通过参数高效适配器将语音嵌入转为LLM标记，应用于ASR、NER、SA任务。结合合成数据标注与LoRA优化，显著提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2509.04473v1](http://arxiv.org/pdf/2509.04473v1)**

> **作者:** Jaekwon Yoo; Kunal Chandiramani; Divya Tadimeti; Abenezer Girma; Chandra Dhir
>
> **摘要:** While integrating speech encoder with LLM requires substantial data and resources, use cases face limitations due to insufficient availability. To address this, we propose a solution with a parameter-efficient adapter that converts speech embeddings into LLM-compatible tokens, focusing on end-to-end automatic speech recognition (ASR), named entity recognition (NER), and sentiment analysis (SA). To reduce labeling costs, we employ an LLM-based synthetic dataset annotation technique. The proposed adapter, using 7x fewer trainable parameters, achieves significant performance gains: a 26% relative Word Error Rates (WER) improvement on the LibriSpeech ASR task, a 6.3% relative F1 score increase on the NER task, and a 32% relative F1 score boost on the SA task. Moreover, using advanced techniques such as adding a classifier regularizer and optimizing the LLM with Low-Rank Adaptation (LoRA) yields notable performance gains, with Spoken Language Understanding Evaluation (SLUE) score improvement of 6.6% and 9.5%
>
---
#### [new 022] Evaluating Large Language Models for Financial Reasoning: A CFA-Based Benchmark Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估大语言模型在金融推理中的表现，针对CFA考试设计基准测试，比较多模态、推理专用与轻量模型，提出RAG方法提升准确性，分析知识差距为主要问题，为金融应用提供模型选择依据。**

- **链接: [http://arxiv.org/pdf/2509.04468v1](http://arxiv.org/pdf/2509.04468v1)**

> **作者:** Xuan Yao; Qianteng Wang; Xinbo Liu; Ke-Wei Huang
>
> **摘要:** The rapid advancement of large language models presents significant opportunities for financial applications, yet systematic evaluation in specialized financial contexts remains limited. This study presents the first comprehensive evaluation of state-of-the-art LLMs using 1,560 multiple-choice questions from official mock exams across Levels I-III of CFA, most rigorous professional certifications globally that mirror real-world financial analysis complexity. We compare models distinguished by core design priorities: multi-modal and computationally powerful, reasoning-specialized and highly accurate, and lightweight efficiency-optimized. We assess models under zero-shot prompting and through a novel Retrieval-Augmented Generation pipeline that integrates official CFA curriculum content. The RAG system achieves precise domain-specific knowledge retrieval through hierarchical knowledge organization and structured query generation, significantly enhancing reasoning accuracy in professional financial certification evaluation. Results reveal that reasoning-oriented models consistently outperform others in zero-shot settings, while the RAG pipeline provides substantial improvements particularly for complex scenarios. Comprehensive error analysis identifies knowledge gaps as the primary failure mode, with minimal impact from text readability. These findings provide actionable insights for LLM deployment in finance, offering practitioners evidence-based guidance for model selection and cost-performance optimization.
>
---
#### [new 023] RECAP: REwriting Conversations for Intent Understanding in Agentic Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RECAP基准，解决对话中意图理解困难问题，通过重写用户-代理对话为简洁目标表示提升规划效果，开发LLM评估器及提示与微调方法增强意图重写能力。**

- **链接: [http://arxiv.org/pdf/2509.04472v1](http://arxiv.org/pdf/2509.04472v1)**

> **作者:** Kushan Mitra; Dan Zhang; Hannah Kim; Estevam Hruschka
>
> **摘要:** Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agent planning in open-domain dialogue systems.
>
---
#### [new 024] Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对黑盒LLM在问答任务中的幻觉检测问题，提出基于熵生产率（EPR）与监督学习的单次生成检测方法，利用有限的token级log概率特征提升检测效果，适用于API受限场景及金融等实际应用。**

- **链接: [http://arxiv.org/pdf/2509.04492v1](http://arxiv.org/pdf/2509.04492v1)**

> **作者:** Charles Moslonka; Hicham Randrianarivo; Arthur Garnier; Emmanuel Malherbe
>
> **备注:** 8 pages, 7 figures, 1 table. pre-print version
>
> **摘要:** Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks critically undermine their real-world reliability. This paper introduces an applied methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) metric that offers baseline performance, later augmented with supervised learning. Our learned model uses features representing the entropic contributions of the accessible top-ranked tokens within a single generated sequence, requiring no multiple query re-runs. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves hallucination detection over using EPR alone. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top <10 per token), confirming its practical efficiency and suitability for these API-constrained deployments. This work provides a readily deployable technique to enhance the trustworthiness of LLM responses from a single generation pass in QA and Retrieval-Augmented Generation (RAG) systems, with its utility further demonstrated in a finance framework analyzing responses to queries on annual reports from an industrial dataset.
>
---
#### [new 025] Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs
- **分类: cs.CL**

- **简介: 该论文提出AgentSeer框架，通过动作图分解LLM代理执行，系统评估模型与代理层面漏洞差异。研究发现代理特有漏洞及跨模型攻击模式，揭示现有安全评估框架在部署风险分析中的不足，强调需建立代理情境评估范式。**

- **链接: [http://arxiv.org/pdf/2509.04802v1](http://arxiv.org/pdf/2509.04802v1)**

> **作者:** Ilham Wicaksono; Zekun Wu; Theo King; Adriano Koshiyama; Philip Treleaven
>
> **摘要:** As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.
>
---
#### [new 026] Entropy2Vec: Crosslingual Language Modeling Entropy as End-to-End Learnable Language Representations
- **分类: cs.CL**

- **简介: 该论文提出Entropy2Vec框架，通过单语模型预测熵生成跨语言表示，解决传统方法特征稀疏和静态问题。利用熵反映语言结构相似性，生成密集嵌入，实验证明其在多语言任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.05060v1](http://arxiv.org/pdf/2509.05060v1)**

> **作者:** Patrick Amadeus Irawan; Ryandito Diandaru; Belati Jagad Bintang Syuhada; Randy Zakya Suchrady; Alham Fikri Aji; Genta Indra Winata; Fajri Koto; Samuel Cahyawijaya
>
> **摘要:** We introduce Entropy2Vec, a novel framework for deriving cross-lingual language representations by leveraging the entropy of monolingual language models. Unlike traditional typological inventories that suffer from feature sparsity and static snapshots, Entropy2Vec uses the inherent uncertainty in language models to capture typological relationships between languages. By training a language model on a single language, we hypothesize that the entropy of its predictions reflects its structural similarity to other languages: Low entropy indicates high similarity, while high entropy suggests greater divergence. This approach yields dense, non-sparse language embeddings that are adaptable to different timeframes and free from missing values. Empirical evaluations demonstrate that Entropy2Vec embeddings align with established typological categories and achieved competitive performance in downstream multilingual NLP tasks, such as those addressed by the LinguAlchemy framework.
>
---
#### [new 027] Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估量化对生物医学NLP模型的影响，解决大模型在医疗场景中的部署难题。通过系统测试12个模型在四个任务上的表现，证明量化可大幅降低内存需求并保持性能，推荐量化作为有效策略。**

- **链接: [http://arxiv.org/pdf/2509.04534v1](http://arxiv.org/pdf/2509.04534v1)**

> **作者:** Zaifu Zhan; Shuang Zhou; Min Zeng; Kai Yu; Meijia Song; Xiaoyi Chen; Jun Wang; Yu Hou; Rui Zhang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Large language models have demonstrated remarkable capabilities in biomedical natural language processing, yet their rapid growth in size and computational requirements present a major barrier to adoption in healthcare settings where data privacy precludes cloud deployment and resources are limited. In this study, we systematically evaluated the impact of quantization on 12 state-of-the-art large language models, including both general-purpose and biomedical-specific models, across eight benchmark datasets covering four key tasks: named entity recognition, relation extraction, multi-label classification, and question answering. We show that quantization substantially reduces GPU memory requirements-by up to 75%-while preserving model performance across diverse tasks, enabling the deployment of 70B-parameter models on 40GB consumer-grade GPUs. In addition, domain-specific knowledge and responsiveness to advanced prompting methods are largely maintained. These findings provide significant practical and guiding value, highlighting quantization as a practical and effective strategy for enabling the secure, local deployment of large yet high-capacity language models in biomedical contexts, bridging the gap between technical advances in AI and real-world clinical translation.
>
---
#### [new 028] From Silent Signals to Natural Language: A Dual-Stage Transformer-LLM Approach
- **分类: cs.CL; cs.AI**

- **简介: 论文提出双阶段Transformer-LLM框架，解决无声语音接口中合成语音识别错误率高、模糊和噪声问题，通过Transformer捕获上下文和LLM确保语言一致性，显著降低词错误率。**

- **链接: [http://arxiv.org/pdf/2509.04507v1](http://arxiv.org/pdf/2509.04507v1)**

> **作者:** Nithyashree Sivasubramaniam
>
> **摘要:** Silent Speech Interfaces (SSIs) have gained attention for their ability to generate intelligible speech from non-acoustic signals. While significant progress has been made in advancing speech generation pipelines, limited work has addressed the recognition and downstream processing of synthesized speech, which often suffers from phonetic ambiguity and noise. To overcome these challenges, we propose an enhanced automatic speech recognition framework that combines a transformer-based acoustic model with a large language model (LLM) for post-processing. The transformer captures full utterance context, while the LLM ensures linguistic consistency. Experimental results show a 16% relative and 6% absolute reduction in word error rate (WER) over a 36% baseline, demonstrating substantial improvements in intelligibility for silent speech interfaces.
>
---
#### [new 029] Optimizing Small Transformer-Based Language Models for Multi-Label Sentiment Analysis in Short Texts
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究多标签短文本情感分类，解决类别不平衡与数据稀疏问题。通过数据增强提升性能，发现持续预训练易引入噪声，分类头改进效果有限，为资源受限场景下的模型优化提供指导。**

- **链接: [http://arxiv.org/pdf/2509.04982v1](http://arxiv.org/pdf/2509.04982v1)**

> **作者:** Julius Neumann; Robert Lange; Yuni Susanti; Michael Färber
>
> **备注:** Accepted at LDD@ECAI 2025
>
> **摘要:** Sentiment classification in short text datasets faces significant challenges such as class imbalance, limited training samples, and the inherent subjectivity of sentiment labels -- issues that are further intensified by the limited context in short texts. These factors make it difficult to resolve ambiguity and exacerbate data sparsity, hindering effective learning. In this paper, we evaluate the effectiveness of small Transformer-based models (i.e., BERT and RoBERTa, with fewer than 1 billion parameters) for multi-label sentiment classification, with a particular focus on short-text settings. Specifically, we evaluated three key factors influencing model performance: (1) continued domain-specific pre-training, (2) data augmentation using automatically generated examples, specifically generative data augmentation, and (3) architectural variations of the classification head. Our experiment results show that data augmentation improves classification performance, while continued pre-training on augmented datasets can introduce noise rather than boost accuracy. Furthermore, we confirm that modifications to the classification head yield only marginal benefits. These findings provide practical guidance for optimizing BERT-based models in resource-constrained settings and refining strategies for sentiment classification in short-text datasets.
>
---
#### [new 030] PLaMo 2 Technical Report
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PLaMo 2，开发高效日语大模型，解决数据稀缺和计算效率问题。通过混合架构、合成数据、结构化剪枝及后训练优化，实现8B参数模型性能媲美100B模型，并在日语基准测试中达到最先进水平。**

- **链接: [http://arxiv.org/pdf/2509.04897v1](http://arxiv.org/pdf/2509.04897v1)**

> **作者:** Preferred Networks; :; Kaizaburo Chubachi; Yasuhiro Fujita; Shinichi Hemmi; Yuta Hirokawa; Toshiki Kataoka; Goro Kobayashi; Kenichi Maehashi; Calvin Metzger; Hiroaki Mikami; Shogo Murai; Daisuke Nishino; Kento Nozawa; Shintarou Okada; Daisuke Okanohara; Shunta Saito; Shotaro Sano; Shuji Suzuki; Daisuke Tanaka; Avinash Ummadisingu; Hanqin Wang; Sixue Wang; Tianqi Xu
>
> **摘要:** In this report, we introduce PLaMo 2, a series of Japanese-focused large language models featuring a hybrid Samba-based architecture that transitions to full attention via continual pre-training to support 32K token contexts. Training leverages extensive synthetic corpora to overcome data scarcity, while computational efficiency is achieved through weight reuse and structured pruning. This efficient pruning methodology produces an 8B model that achieves performance comparable to our previous 100B model. Post-training further refines the models using a pipeline of supervised fine-tuning (SFT) and direct preference optimization (DPO), enhanced by synthetic Japanese instruction data and model merging techniques. Optimized for inference using vLLM and quantization with minimal accuracy loss, the PLaMo 2 models achieve state-of-the-art results on Japanese benchmarks, outperforming similarly-sized open models in instruction-following, language fluency, and Japanese-specific knowledge.
>
---
#### [new 031] Using LLMs to create analytical datasets: A case study of reconstructing the historical memory of Colombia
- **分类: cs.CL; cs.CY**

- **简介: 该论文利用LLM处理20万+西班牙语新闻文本，构建冲突数据集，分析暴力与禁毒政策关系，解决哥伦比亚历史记录缺失问题，探索LLM在大规模文本分析中的应用价值。**

- **链接: [http://arxiv.org/pdf/2509.04523v1](http://arxiv.org/pdf/2509.04523v1)**

> **作者:** David Anderson; Galia Benitez; Margret Bjarnadottir; Shriyan Reyya
>
> **摘要:** Colombia has been submerged in decades of armed conflict, yet until recently, the systematic documentation of violence was not a priority for the Colombian government. This has resulted in a lack of publicly available conflict information and, consequently, a lack of historical accounts. This study contributes to Colombia's historical memory by utilizing GPT, a large language model (LLM), to read and answer questions about over 200,000 violence-related newspaper articles in Spanish. We use the resulting dataset to conduct both descriptive analysis and a study of the relationship between violence and the eradication of coca crops, offering an example of policy analyses that such data can support. Our study demonstrates how LLMs have opened new research opportunities by enabling examinations of large text corpora at a previously infeasible depth.
>
---
#### [new 032] ProST: Progressive Sub-task Training for Pareto-Optimal Multi-agent Systems Using Small Language Models
- **分类: cs.CL**

- **简介: 论文提出ProST方法，通过渐进子任务训练优化多智能体系统，解决小语言模型长轨迹学习困难，提升效果与效率权衡。**

- **链接: [http://arxiv.org/pdf/2509.04508v1](http://arxiv.org/pdf/2509.04508v1)**

> **作者:** Biddut Sarker Bijoy; Mohammad Saqib Hasan; Pegah Alipoormolabashi; Avirup Sil; Aruna Balasubramanian; Niranjan Balasubramanian
>
> **摘要:** Multi-agent systems with smaller language models (SLMs) present a viable alternative to single agent systems powered by large language models (LLMs) for addressing complex problems. In this work, we study how these alternatives compare in terms of both effectiveness and efficiency. To study this trade-off, we instantiate single and multi-agent systems for the complex problems in the AppWorld environment using different sized language models. We find that difficulties with long-trajectory learning in smaller language models (SLMs) limit their performance. Even when trained for specialized roles, SLMs fail to learn all subtasks effectively. To address this issue, we introduce a simple progressive sub-task training strategy, which introduces new sub-tasks progressively in each training epoch. We find that this novel strategy, analogous to instance level curriculum learning, consistently improves the effectiveness of multi-agents at all configurations. Our Pareto analysis shows that fine-tuned multi-agent systems yield better effectiveness-efficiency trade-offs. Additional ablations and analyses shows the importance of our progressive training strategy and its ability to reduce subtask error rates.
>
---
#### [new 033] OleSpeech-IV: A Large-Scale Multispeaker and Multilingual Conversational Speech Dataset with Diverse Topics
- **分类: cs.CL**

- **简介: 该论文构建了一个大规模多语言、多说话人会话语音数据集，解决现有数据多样性不足问题。通过整合公开音频内容，人工标注说话人信息与转录文本，并开源子集支持非商业研究。**

- **链接: [http://arxiv.org/pdf/2509.04702v1](http://arxiv.org/pdf/2509.04702v1)**

> **作者:** Wei Chu; Yuanzhe Dong; Ke Tan; Dong Han; Xavier Menendez-Pidal; Ruchao Fan; Chenfeng Miao; Chanwoo Kim; Bhiksha Raj; Rita Singh
>
> **摘要:** OleSpeech-IV dataset is a large-scale multispeaker and multilingual conversational speech dataset with diverse topics. The audio content comes from publicly-available English podcasts, talk shows, teleconferences, and other conversations. Speaker names, turns, and transcripts are human-sourced and refined by a proprietary pipeline, while additional information such as timestamps and confidence scores is derived from the pipeline. The IV denotes its position as Tier IV in the Olewave dataset series. In addition, we have open-sourced a subset, OleSpeech-IV-2025-EN-AR-100, for non-commercial research use.
>
---
#### [new 034] Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理中的高计算与内存成本问题，提出基于prefill-decode拆分的定向剪枝方法，通过分阶段块移除与token-aware缓存优化，提升推理效率并减少通信开销。**

- **链接: [http://arxiv.org/pdf/2509.04467v1](http://arxiv.org/pdf/2509.04467v1)**

> **作者:** Hao Zhang; Mengsi Lyu; Yulong Ao; Yonghua Lin
>
> **备注:** 21 pages
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the default settings, our method achieves a 20.56% inference speedup and a 4.95 times reduction in data transmission bandwidth consumption.
>
---
#### [new 035] Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation
- **分类: cs.CL**

- **简介: 该论文提出难度感知链式思维蒸馏框架，解决数学推理中简单问题输出冗长的问题。通过后训练数据调控推理长度，结合SFT与DPO优化，使模型动态调整推理深度，提升效率。**

- **链接: [http://arxiv.org/pdf/2509.05226v1](http://arxiv.org/pdf/2509.05226v1)**

> **作者:** Abdul Waheed; Chancharik Mitra; Laurie Z. Wang; Deva Ramanan; Bhiksha Raj
>
> **备注:** 28 Pages
>
> **摘要:** Chain-of-thought reasoning, while powerful, can produce unnecessarily verbose output for simpler problems. We present a framework for difficulty-aware reasoning that teaches models to dynamically adjust reasoning depth based on problem complexity. Remarkably, we show that models can be endowed with such dynamic inference pathways without any architectural modifications; we simply post-train on data that is carefully curated to include chain-of-thought traces that are proportional in length to problem difficulty. Our analysis reveals that post-training via supervised fine-tuning (SFT) primarily captures patterns like reasoning length and format, while direct preference optimization (DPO) preserves reasoning accuracy, with their combination reducing length and maintaining or improving performance. Both quantitative metrics and qualitative assessments confirm that models can learn to "think proportionally", reasoning minimally on simple problems while maintaining depth for complex ones.
>
---
#### [new 036] Where Should I Study? Biased Language Models Decide! Evaluating Fairness in LMs for Academic Recommendations
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估语言模型在学术推荐中的公平性，解决推荐偏见问题。通过分析三个模型的推荐结果，发现地理、性别等偏见，并提出多维评估框架以量化偏见，促进教育公平。**

- **链接: [http://arxiv.org/pdf/2509.04498v1](http://arxiv.org/pdf/2509.04498v1)**

> **作者:** Krithi Shailya; Akhilesh Kumar Mishra; Gokul S Krishnan; Balaraman Ravindran
>
> **摘要:** Large Language Models (LLMs) are increasingly used as daily recommendation systems for tasks like education planning, yet their recommendations risk perpetuating societal biases. This paper empirically examines geographic, demographic, and economic biases in university and program suggestions from three open-source LLMs: LLaMA-3.1-8B, Gemma-7B, and Mistral-7B. Using 360 simulated user profiles varying by gender, nationality, and economic status, we analyze over 25,000 recommendations. Results show strong biases: institutions in the Global North are disproportionately favored, recommendations often reinforce gender stereotypes, and institutional repetition is prevalent. While LLaMA-3.1 achieves the highest diversity, recommending 481 unique universities across 58 countries, systemic disparities persist. To quantify these issues, we propose a novel, multi-dimensional evaluation framework that goes beyond accuracy by measuring demographic and geographic representation. Our findings highlight the urgent need for bias consideration in educational LMs to ensure equitable global access to higher education.
>
---
#### [new 037] An End-to-End System for Culturally-Attuned Driving Feedback using a Dual-Component NLG Engine
- **分类: cs.CL; F.2.2; I.2.7**

- **简介: 该论文提出一个端到端系统，针对尼日利亚低资源环境，通过双组件NLG生成文化适配的驾驶反馈，结合酒精检测模型，解决安全驾驶问题，经试点验证有效。**

- **链接: [http://arxiv.org/pdf/2509.04478v1](http://arxiv.org/pdf/2509.04478v1)**

> **作者:** Iniakpokeikiye Peter Thompson; Yi Dewei; Reiter Ehud
>
> **备注:** The paper has 5 figures and 1 table
>
> **摘要:** This paper presents an end-to-end mobile system that delivers culturally-attuned safe driving feedback to drivers in Nigeria, a low-resource environment with significant infrastructural challenges. The core of the system is a novel dual-component Natural Language Generation (NLG) engine that provides both legally-grounded safety tips and persuasive, theory-driven behavioural reports. We describe the complete system architecture, including an automatic trip detection service, on-device behaviour analysis, and a sophisticated NLG pipeline that leverages a two-step reflection process to ensure high-quality feedback. The system also integrates a specialized machine learning model for detecting alcohol-influenced driving, a key local safety issue. The architecture is engineered for robustness against intermittent connectivity and noisy sensor data. A pilot deployment with 90 drivers demonstrates the viability of our approach, and initial results on detected unsafe behaviours are presented. This work provides a framework for applying data-to-text and AI systems to achieve social good.
>
---
#### [new 038] ICR: Iterative Clarification and Rewriting for Conversational Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对对话搜索中多模糊表达导致的端到端重写困难，提出ICR框架。通过迭代生成澄清问题与重写查询，持续优化检索性能，实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.05100v1](http://arxiv.org/pdf/2509.05100v1)**

> **作者:** Zhiyu Cao; Peifeng Li; Qiaoming Zhu
>
> **摘要:** Most previous work on Conversational Query Rewriting employs an end-to-end rewriting paradigm. However, this approach is hindered by the issue of multiple fuzzy expressions within the query, which complicates the simultaneous identification and rewriting of multiple positions. To address this issue, we propose a novel framework ICR (Iterative Clarification and Rewriting), an iterative rewriting scheme that pivots on clarification questions. Within this framework, the model alternates between generating clarification questions and rewritten queries. The experimental results show that our ICR can continuously improve retrieval performance in the clarification-rewriting iterative process, thereby achieving state-of-the-art performance on two popular datasets.
>
---
#### [new 039] Combine Virtual Reality and Machine-Learning to Identify the Presence of Dyslexia: A Cross-Linguistic Approach
- **分类: cs.CL; cs.HC**

- **简介: 该论文旨在通过VR与机器学习结合，跨语言识别阅读障碍。研究利用VR静默阅读测试和自尊评估数据，训练ML模型区分意大利和西班牙大学生中的阅读障碍者，验证VR任务完成速度差异对分类的有效性。**

- **链接: [http://arxiv.org/pdf/2509.04510v1](http://arxiv.org/pdf/2509.04510v1)**

> **作者:** Michele Materazzini; Gianluca Morciano; Jose Manuel Alcalde-Llergo; Enrique Yeguas-Bolivar; Giuseppe Calabro; Andrea Zingoni; Juri Taborri
>
> **备注:** 22 pages, 10 figures, 5 tables
>
> **摘要:** This study explores the use of virtual reality (VR) and artificial intelligence (AI) to predict the presence of dyslexia in Italian and Spanish university students. In particular, the research investigates whether VR-derived data from Silent Reading (SR) tests and self-esteem assessments can differentiate between students that are affected by dyslexia and students that are not, employing machine learning (ML) algorithms. Participants completed VR-based tasks measuring reading performance and self-esteem. A preliminary statistical analysis (t tests and Mann Whitney tests) on these data was performed, to compare the obtained scores between individuals with and without dyslexia, revealing significant differences in completion time for the SR test, but not in accuracy, nor in self esteem. Then, supervised ML models were trained and tested, demonstrating an ability to classify the presence/absence of dyslexia with an accuracy of 87.5 per cent for Italian, 66.6 per cent for Spanish, and 75.0 per cent for the pooled group. These findings suggest that VR and ML can effectively be used as supporting tools for assessing dyslexia, particularly by capturing differences in task completion speed, but language-specific factors may influence classification accuracy.
>
---
#### [new 040] Refining Transcripts With TV Subtitles by Prompt-Based Weakly Supervised Training of ASR
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出基于TV字幕的弱监督ASR方法，解决字幕与音频对齐不准导致的监督信号不足问题。通过将字幕作为上下文提示生成伪转录，并引入加权注意力机制，显著提升转录准确性，为ASR训练提供高质量伪标签数据。**

- **链接: [http://arxiv.org/pdf/2509.04491v1](http://arxiv.org/pdf/2509.04491v1)**

> **作者:** Xinnian Zhao; Hugo Van Hamme
>
> **备注:** eusipco2025
>
> **摘要:** This study proposes a novel approach to using TV subtitles within a weakly supervised (WS) Automatic Speech Recognition (ASR) framework. Although TV subtitles are readily available, their imprecise alignment with corresponding audio limits their applicability as supervised targets for verbatim transcription. Rather than using subtitles as direct supervision signals, our method reimagines them as context-rich prompts. This design enables the model to handle discrepancies between spoken audio and subtitle text. Instead, generated pseudo transcripts become the primary targets, with subtitles acting as guiding cues for iterative refinement. To further enhance the process, we introduce a weighted attention mechanism that emphasizes relevant subtitle tokens during inference. Our experiments demonstrate significant improvements in transcription accuracy, highlighting the effectiveness of the proposed method in refining transcripts. These enhanced pseudo-labeled datasets provide high-quality foundational resources for training robust ASR systems.
>
---
#### [new 041] Triadic Fusion of Cognitive, Functional, and Causal Dimensions for Explainable LLMs: The TAXAL Framework
- **分类: cs.CL**

- **简介: 该论文提出TAXAL框架，解决LLM在高风险应用中解释性不足的问题。通过融合认知、功能与因果三维度，构建统一的解释性基础，结合案例研究验证其在多领域适用性，推动可解释AI的技社融合实践。**

- **链接: [http://arxiv.org/pdf/2509.05199v1](http://arxiv.org/pdf/2509.05199v1)**

> **作者:** David Herrera-Poyatos; Carlos Peláez-González; Cristina Zuheros; Virilo Tejedor; Rosana Montes; Francisco Herrera
>
> **备注:** 27 pages, 9 tables and 2 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly being deployed in high-risk domains where opacity, bias, and instability undermine trust and accountability. Traditional explainability methods, focused on surface outputs, do not capture the reasoning pathways, planning logic, and systemic impacts of agentic LLMs. We introduce TAXAL (Triadic Alignment for eXplainability in Agentic LLMs), a triadic fusion framework that unites three complementary dimensions: cognitive (user understanding), functional (practical utility), and causal (faithful reasoning). TAXAL provides a unified, role-sensitive foundation for designing, evaluating, and deploying explanations in diverse sociotechnical settings. Our analysis synthesizes existing methods, ranging from post-hoc attribution and dialogic interfaces to explanation-aware prompting, and situates them within the TAXAL triadic fusion model. We further demonstrate its applicability through case studies in law, education, healthcare, and public services, showing how explanation strategies adapt to institutional constraints and stakeholder roles. By combining conceptual clarity with design patterns and deployment pathways, TAXAL advances explainability as a technical and sociotechnical practice, supporting trustworthy and context-sensitive LLM applications in the era of agentic AI.
>
---
#### [new 042] A Study of Large Language Models for Patient Information Extraction: Model Architecture, Fine-Tuning Strategy, and Multi-task Instruction Tuning
- **分类: cs.CL; cs.AI**

- **简介: 本研究探讨大语言模型在临床信息提取中的应用，比较编码器/解码器架构、参数高效微调及多任务指令调优策略，评估其在零样本和少样本场景下的性能，以提升医疗文本信息提取的准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.04753v1](http://arxiv.org/pdf/2509.04753v1)**

> **作者:** Cheng Peng; Xinyu Dong; Mengxian Lyu; Daniel Paredes; Yaoyun Zhang; Yonghui Wu
>
> **摘要:** Natural language processing (NLP) is a key technology to extract important patient information from clinical narratives to support healthcare applications. The rapid development of large language models (LLMs) has revolutionized many NLP tasks in the clinical domain, yet their optimal use in patient information extraction tasks requires further exploration. This study examines LLMs' effectiveness in patient information extraction, focusing on LLM architectures, fine-tuning strategies, and multi-task instruction tuning techniques for developing robust and generalizable patient information extraction systems. This study aims to explore key concepts of using LLMs for clinical concept and relation extraction tasks, including: (1) encoder-only or decoder-only LLMs, (2) prompt-based parameter-efficient fine-tuning (PEFT) algorithms, and (3) multi-task instruction tuning on few-shot learning performance. We benchmarked a suite of LLMs, including encoder-based LLMs (BERT, GatorTron) and decoder-based LLMs (GatorTronGPT, Llama 3.1, GatorTronLlama), across five datasets. We compared traditional full-size fine-tuning and prompt-based PEFT. We explored a multi-task instruction tuning framework that combines both tasks across four datasets to evaluate the zero-shot and few-shot learning performance using the leave-one-dataset-out strategy.
>
---
#### [new 043] From Post To Personality: Harnessing LLMs for MBTI Prediction in Social Media
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出PtoP框架，利用LLM结合检索增强生成与上下文学习，解决社交媒体文本中MBTI预测的幻觉问题和类别不平衡问题，通过合成过采样提升模型性能，实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.04461v1](http://arxiv.org/pdf/2509.04461v1)**

> **作者:** Tian Ma; Kaiyu Feng; Yu Rong; Kangfei Zhao
>
> **摘要:** Personality prediction from social media posts is a critical task that implies diverse applications in psychology and sociology. The Myers Briggs Type Indicator (MBTI), a popular personality inventory, has been traditionally predicted by machine learning (ML) and deep learning (DL) techniques. Recently, the success of Large Language Models (LLMs) has revealed their huge potential in understanding and inferring personality traits from social media content. However, directly exploiting LLMs for MBTI prediction faces two key challenges: the hallucination problem inherent in LLMs and the naturally imbalanced distribution of MBTI types in the population. In this paper, we propose PostToPersonality (PtoP), a novel LLM based framework for MBTI prediction from social media posts of individuals. Specifically, PtoP leverages Retrieval Augmented Generation with in context learning to mitigate hallucination in LLMs. Furthermore, we fine tune a pretrained LLM to improve model specification in MBTI understanding with synthetic minority oversampling, which balances the class imbalance by generating synthetic samples. Experiments conducted on a real world social media dataset demonstrate that PtoP achieves state of the art performance compared with 10 ML and DL baselines.
>
---
#### [new 044] Analyzing Finnish Inflectional Classes through Discriminative Lexicon and Deep Learning Models
- **分类: cs.CL**

- **简介: 本研究评估判别词典模型能否在不依赖预设屈折类的情况下学习芬兰语屈折结构，通过构建频率敏感模型测试其表现，分析频率与生产力对模型性能的影响，探讨屈折类的认知现实性。**

- **链接: [http://arxiv.org/pdf/2509.04813v1](http://arxiv.org/pdf/2509.04813v1)**

> **作者:** Alexandre Nikolaev; Yu-Ying Chuang; R. Harald Baayen
>
> **摘要:** Descriptions of complex nominal or verbal systems make use of inflectional classes. Inflectional classes bring together nouns which have similar stem changes and use similar exponents in their paradigms. Although inflectional classes can be very useful for language teaching as well as for setting up finite state morphological systems, it is unclear whether inflectional classes are cognitively real, in the sense that native speakers would need to discover these classes in order to learn how to properly inflect the nouns of their language. This study investigates whether the Discriminative Lexicon Model (DLM) can understand and produce Finnish inflected nouns without setting up inflectional classes, using a dataset with 55,271 inflected nouns of 2000 high-frequency Finnish nouns from 49 inflectional classes. Several DLM comprehension and production models were set up. Some models were not informed about frequency of use, and provide insight into learnability with infinite exposure (endstate learning). Other models were set up from a usage based perspective, and were trained with token frequencies being taken into consideration (frequency-informed learning). On training data, models performed with very high accuracies. For held-out test data, accuracies decreased, as expected, but remained acceptable. Across most models, performance increased for inflectional classes with more types, more lower-frequency words, and more hapax legomena, mirroring the productivity of the inflectional classes. The model struggles more with novel forms of unproductive and less productive classes, and performs far better for unseen forms belonging to productive classes. However, for usage-based production models, frequency was the dominant predictor of model performance, and correlations with measures of productivity were tenuous or absent.
>
---
#### [new 045] Enhancing Diversity in Large Language Models via Determinantal Point Processes
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DQO方法，基于DPPs在训练中联合优化LLM质量与语义多样性，解决现有方法因提升性能导致输出多样性下降的问题，通过嵌入响应并计算行列式衡量多样性，实验证明有效。**

- **链接: [http://arxiv.org/pdf/2509.04784v1](http://arxiv.org/pdf/2509.04784v1)**

> **作者:** Yilei Chen; Souradip Chakraborty; Lorenz Wolf; Ioannis Ch. Paschalidis; Aldo Pacchiano
>
> **摘要:** Supervised fine-tuning and reinforcement learning are two popular methods for post-training large language models (LLMs). While improving the model's performance on downstream tasks, they often reduce the model's output diversity, leading to narrow, canonical responses. Existing methods to enhance diversity are limited, either by operating at inference time or by focusing on lexical differences. We propose a novel training method named DQO based on determinantal point processes (DPPs) to jointly optimize LLMs for quality and semantic diversity. Our approach samples and embeds a group of responses for each prompt, then uses the determinant of a kernel-based similarity matrix to measure diversity as the volume spanned by the embeddings of these responses. Experiments across instruction-following, summarization, story generation, and reasoning tasks demonstrate that our method substantially improves semantic diversity without sacrificing model quality.
>
---
#### [new 046] ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health Records
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ASCENDgpt模型，通过表型感知分词和预训练，解决EHR心血管风险预测中ICD代码冗余与语义丢失问题，提升预测性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.04485v1](http://arxiv.org/pdf/2509.04485v1)**

> **作者:** Chris Sainsbury; Andreas Karwath
>
> **摘要:** We present ASCENDgpt, a transformer-based model specifically designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs). Our approach introduces a novel phenotype-aware tokenization scheme that maps 47,155 raw ICD codes to 176 clinically meaningful phenotype tokens, achieving 99.6\% consolidation of diagnosis codes while preserving semantic information. This phenotype mapping contributes to a total vocabulary of 10,442 tokens - a 77.9\% reduction when compared with using raw ICD codes directly. We pretrain ASCENDgpt on sequences derived from 19402 unique individuals using a masked language modeling objective, then fine-tune for time-to-event prediction of five cardiovascular outcomes: myocardial infarction (MI), stroke, major adverse cardiovascular events (MACE), cardiovascular death, and all-cause mortality. Our model achieves excellent discrimination on the held-out test set with an average C-index of 0.816, demonstrating strong performance across all outcomes (MI: 0.792, stroke: 0.824, MACE: 0.800, cardiovascular death: 0.842, all-cause mortality: 0.824). The phenotype-based approach enables clinically interpretable predictions while maintaining computational efficiency. Our work demonstrates the effectiveness of domain-specific tokenization and pretraining for EHR-based risk prediction tasks.
>
---
#### [new 047] Serialized Output Prompting for Large Language Model-based Multi-Talker Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出基于序列化输出提示（SOP）的多说话人语音识别方法，解决LLM在复杂多说话人场景下性能不足的问题。通过分离器与CTC层提取序列化内容，结合三阶段训练策略，提升LLM对混合语音的识别效果。**

- **链接: [http://arxiv.org/pdf/2509.04488v1](http://arxiv.org/pdf/2509.04488v1)**

> **作者:** Hao Shi; Yusuke Fujita; Tomoya Mizumoto; Lianbo Liu; Atsushi Kojima; Yui Sudo
>
> **摘要:** Prompts are crucial for task definition and for improving the performance of large language models (LLM)-based systems. However, existing LLM-based multi-talker (MT) automatic speech recognition (ASR) systems either omit prompts or rely on simple task-definition prompts, with no prior work exploring the design of prompts to enhance performance. In this paper, we propose extracting serialized output prompts (SOP) and explicitly guiding the LLM using structured prompts to improve system performance (SOP-MT-ASR). A Separator and serialized Connectionist Temporal Classification (CTC) layers are inserted after the speech encoder to separate and extract MT content from the mixed speech encoding in a first-speaking-first-out manner. Subsequently, the SOP, which serves as a prompt for LLMs, is obtained by decoding the serialized CTC outputs using greedy search. To train the model effectively, we design a three-stage training strategy, consisting of serialized output training (SOT) fine-tuning, serialized speech information extraction, and SOP-based adaptation. Experimental results on the LibriMix dataset show that, although the LLM-based SOT model performs well in the two-talker scenario, it fails to fully leverage LLMs under more complex conditions, such as the three-talker scenario. The proposed SOP approach significantly improved performance under both two- and three-talker conditions.
>
---
#### [new 048] Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型个性控制方法（ICL、PEFT、MS）的权衡与下游影响，构建对比数据集和评估框架，分析不同方法对性能、稳定性及偏见的影响，提出特征纯化和稳定性框架，揭示个性操控对模型行为的多层级影响。**

- **链接: [http://arxiv.org/pdf/2509.04794v1](http://arxiv.org/pdf/2509.04794v1)**

> **作者:** Gunmay Handa; Zekun Wu; Adriano Koshiyama; Philip Treleaven
>
> **摘要:** Personality manipulation in large language models (LLMs) is increasingly applied in customer service and agentic scenarios, yet its mechanisms and trade-offs remain unclear. We present a systematic study of personality control using the Big Five traits, comparing in-context learning (ICL), parameter-efficient fine-tuning (PEFT), and mechanistic steering (MS). Our contributions are fourfold. First, we construct a contrastive dataset with balanced high/low trait responses, enabling effective steering vector computation and fair cross-method evaluation. Second, we introduce a unified evaluation framework based on within-run $\Delta$ analysis that disentangles, reasoning capability, agent performance, and demographic bias across MMLU, GAIA, and BBQ benchmarks. Third, we develop trait purification techniques to separate openness from conscientiousness, addressing representational overlap in trait encoding. Fourth, we propose a three-level stability framework that quantifies method-, trait-, and combination-level robustness, offering practical guidance under deployment constraints. Experiments on Gemma-2-2B-IT and LLaMA-3-8B-Instruct reveal clear trade-offs: ICL achieves strong alignment with minimal capability loss, PEFT delivers the highest alignment at the cost of degraded task performance, and MS provides lightweight runtime control with competitive effectiveness. Trait-level analysis shows openness as uniquely challenging, agreeableness as most resistant to ICL, and personality encoding consolidating around intermediate layers. Taken together, these results establish personality manipulation as a multi-level probe into behavioral representation, linking surface conditioning, parameter encoding, and activation-level steering, and positioning mechanistic steering as a lightweight alternative to fine-tuning for both deployment and interpretability.
>
---
#### [new 049] Do Large Language Models Need Intent? Revisiting Response Generation Strategies for Service Assistant
- **分类: cs.CL; cs.LG**

- **简介: 该论文通过对比意图优先与直接生成策略，研究大语言模型是否需显式意图识别以生成服务回复。使用公开数据集评估多种模型，发现显式意图建模的必要性，挑战传统假设，提出优化设计建议。（99字）**

- **链接: [http://arxiv.org/pdf/2509.05006v1](http://arxiv.org/pdf/2509.05006v1)**

> **作者:** Inbal Bolshinsky; Shani Kupiec; Almog Sasson; Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** In the era of conversational AI, generating accurate and contextually appropriate service responses remains a critical challenge. A central question remains: Is explicit intent recognition a prerequisite for generating high-quality service responses, or can models bypass this step and produce effective replies directly? This paper conducts a rigorous comparative study to address this fundamental design dilemma. Leveraging two publicly available service interaction datasets, we benchmark several state-of-the-art language models, including a fine-tuned T5 variant, across both paradigms: Intent-First Response Generation and Direct Response Generation. Evaluation metrics encompass both linguistic quality and task success rates, revealing surprising insights into the necessity or redundancy of explicit intent modelling. Our findings challenge conventional assumptions in conversational AI pipelines, offering actionable guidelines for designing more efficient and effective response generation systems.
>
---
#### [new 050] ODKE+: Ontology-Guided Open-Domain Knowledge Extraction with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ODKE+系统，解决知识图谱维护成本高问题，通过结合LLMs与本体引导，实现大规模开放域事实提取与验证，提升精度与覆盖效率。**

- **链接: [http://arxiv.org/pdf/2509.04696v1](http://arxiv.org/pdf/2509.04696v1)**

> **作者:** Samira Khorshidi; Azadeh Nikfarjam; Suprita Shankar; Yisi Sang; Yash Govind; Hyun Jang; Ali Kasgari; Alexis McClimans; Mohamed Soliman; Vishnu Konda; Ahmed Fakhry; Xiaoguang Qi
>
> **摘要:** Knowledge graphs (KGs) are foundational to many AI applications, but maintaining their freshness and completeness remains costly. We present ODKE+, a production-grade system that automatically extracts and ingests millions of open-domain facts from web sources with high precision. ODKE+ combines modular components into a scalable pipeline: (1) the Extraction Initiator detects missing or stale facts, (2) the Evidence Retriever collects supporting documents, (3) hybrid Knowledge Extractors apply both pattern-based rules and ontology-guided prompting for large language models (LLMs), (4) a lightweight Grounder validates extracted facts using a second LLM, and (5) the Corroborator ranks and normalizes candidate facts for ingestion. ODKE+ dynamically generates ontology snippets tailored to each entity type to align extractions with schema constraints, enabling scalable, type-consistent fact extraction across 195 predicates. The system supports batch and streaming modes, processing over 9 million Wikipedia pages and ingesting 19 million high-confidence facts with 98.8% precision. ODKE+ significantly improves coverage over traditional methods, achieving up to 48% overlap with third-party KGs and reducing update lag by 50 days on average. Our deployment demonstrates that LLM-based extraction, grounded in ontological structure and verification workflows, can deliver trustworthiness, production-scale knowledge ingestion with broad real-world applicability. A recording of the system demonstration is included with the submission and is also available at https://youtu.be/UcnE3_GsTWs.
>
---
#### [new 051] Scaling behavior of large language models in emotional safety classification across sizes and tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在情感安全分类任务中的扩展行为，通过构建多源数据集评估不同规模模型（1B/3B/8B/70B）在零样本和微调下的表现，揭示大模型在复杂分类中优势，但轻量微调使小模型可达到相近效果，为隐私敏感场景提供可行方案。**

- **链接: [http://arxiv.org/pdf/2509.04512v1](http://arxiv.org/pdf/2509.04512v1)**

> **作者:** Edoardo Pinzuti; Oliver Tüscher; André Ferreira Castro
>
> **摘要:** Understanding how large language models (LLMs) process emotionally sensitive content is critical for building safe and reliable systems, particularly in mental health contexts. We investigate the scaling behavior of LLMs on two key tasks: trinary classification of emotional safety (safe vs. unsafe vs. borderline) and multi-label classification using a six-category safety risk taxonomy. To support this, we construct a novel dataset by merging several human-authored mental health datasets (> 15K samples) and augmenting them with emotion re-interpretation prompts generated via ChatGPT. We evaluate four LLaMA models (1B, 3B, 8B, 70B) across zero-shot, few-shot, and fine-tuning settings. Our results show that larger LLMs achieve stronger average performance, particularly in nuanced multi-label classification and in zero-shot settings. However, lightweight fine-tuning allowed the 1B model to achieve performance comparable to larger models and BERT in several high-data categories, while requiring <2GB VRAM at inference. These findings suggest that smaller, on-device models can serve as viable, privacy-preserving alternatives for sensitive applications, offering the ability to interpret emotional context and maintain safe conversational boundaries. This work highlights key implications for therapeutic LLM applications and the scalable alignment of safety-critical systems.
>
---
#### [new 052] Comparative Analysis of Transformer Models in Disaster Tweet Classification for Public Safety
- **分类: cs.CL; cs.AI**

- **简介: 该论文比较Transformer模型（如BERT）与传统ML方法在灾难推文分类中的效果，旨在提升公共安全应急响应效率。通过实验验证，Transformer模型在准确率（91%）和语言理解上显著优于传统方法（82%）。**

- **链接: [http://arxiv.org/pdf/2509.04650v1](http://arxiv.org/pdf/2509.04650v1)**

> **作者:** Sharif Noor Zisad; Ragib Hasan
>
> **摘要:** Twitter and other social media platforms have become vital sources of real time information during disasters and public safety emergencies. Automatically classifying disaster related tweets can help emergency services respond faster and more effectively. Traditional Machine Learning (ML) models such as Logistic Regression, Naive Bayes, and Support Vector Machines have been widely used for this task, but they often fail to understand the context or deeper meaning of words, especially when the language is informal, metaphorical, or ambiguous. We posit that, in this context, transformer based models can perform better than traditional ML models. In this paper, we evaluate the effectiveness of transformer based models, including BERT, DistilBERT, RoBERTa, and DeBERTa, for classifying disaster related tweets. These models are compared with traditional ML approaches to highlight the performance gap. Experimental results show that BERT achieved the highest accuracy (91%), significantly outperforming traditional models like Logistic Regression and Naive Bayes (both at 82%). The use of contextual embeddings and attention mechanisms allows transformer models to better understand subtle language in tweets, where traditional ML models fall short. This research demonstrates that transformer architectures are far more suitable for public safety applications, offering improved accuracy, deeper language understanding, and better generalization across real world social media text.
>
---
#### [new 053] Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions
- **分类: cs.CL; cs.AI; 68T50, 68T05; I.2.7; I.2.6; I.2.11**

- **简介: 该论文研究如何通过提示、激活和权重干预实现对Transformer模型的可控性与鲁棒性，解决细粒度控制难题。提出统一框架，结合优化方法与实验验证，展示高成功率的文本生成控制及伦理风险分析。**

- **链接: [http://arxiv.org/pdf/2509.04549v1](http://arxiv.org/pdf/2509.04549v1)**

> **作者:** Faruk Alpay; Taylan Alpay
>
> **备注:** 13 pages
>
> **摘要:** Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models.
>
---
#### [new 054] ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文针对长文本生成任务，解决数据依赖及粗粒度优化问题，提出ACE-RL框架通过自动分解指令为细粒度约束，设计约束验证奖励机制并结合强化学习提升生成质量，实验表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.04903v1](http://arxiv.org/pdf/2509.04903v1)**

> **作者:** Jianghao Chen; Wei Sun; Qixiang Yin; Lingxing Kong; Zhixing Tan; Jiajun Zhang
>
> **备注:** Under review, our code is available at https://github.com/ZNLP/ACE-RL
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable progress in long-context understanding, yet they face significant challenges in high-quality long-form generation. Existing studies primarily suffer from two limitations: (1) A heavy reliance on scarce, high-quality long-form response data for supervised fine-tuning (SFT) or for pairwise preference reward in reinforcement learning (RL). (2) Focus on coarse-grained quality optimization dimensions, such as relevance, coherence, and helpfulness, overlooking the fine-grained specifics inherent to diverse long-form generation scenarios. To address this issue, we propose a framework using Adaptive Constraint-Enhanced reward for long-form generation Reinforcement Learning (ACE-RL). ACE-RL first automatically deconstructs each instruction into a set of fine-grained, adaptive constraint criteria by identifying its underlying intents and demands. Subsequently, we design a reward mechanism that quantifies the quality of long-form responses based on their satisfaction over corresponding constraints, converting subjective quality evaluation into constraint verification. Finally, we utilize reinforcement learning to guide models toward superior long-form generation capabilities. Experimental results demonstrate that our ACE-RL framework significantly outperforms existing SFT and RL baselines by 20.70% and 7.32% on WritingBench, and our top-performing model even surpasses proprietary systems like GPT-4o by 7.10%, providing a more effective training paradigm for LLMs to generate high-quality content across diverse long-form generation scenarios.
>
---
#### [new 055] HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HoPE，改进Transformer位置编码，解决长序列中RoPE的不稳定性问题。通过双曲几何洛伦兹变换，实现注意力权重单调衰减，提升长距离依赖建模效果。**

- **链接: [http://arxiv.org/pdf/2509.05218v1](http://arxiv.org/pdf/2509.05218v1)**

> **作者:** Chang Dai; Hongyu Shan; Mingyang Song; Di Liang
>
> **备注:** This paper proposes Hyperbolic Rotary Positional Encoding (HoPE), a geometric reformulation of positional encoding inspired by Lorentz transformations. HoPE addresses limitations of existing methods like RoPE by enabling stable long-distance dependency modeling. Code and data will be made available upon publication
>
> **摘要:** Positional encoding mechanisms enable Transformers to model sequential structure and long-range dependencies in text. While absolute positional encodings struggle with extrapolation to longer sequences due to fixed positional representations, and relative approaches like Alibi exhibit performance degradation on extremely long contexts, the widely-used Rotary Positional Encoding (RoPE) introduces oscillatory attention patterns that hinder stable long-distance dependency modelling. We address these limitations through a geometric reformulation of positional encoding. Drawing inspiration from Lorentz transformations in hyperbolic geometry, we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations. Theoretical analysis demonstrates that RoPE is a special case of our generalized formulation. HoPE fundamentally resolves RoPE's slation issues by enforcing monotonic decay of attention weights with increasing token distances. Extensive experimental results, including perplexity evaluations under several extended sequence benchmarks, show that HoPE consistently exceeds existing positional encoding methods. These findings underscore HoPE's enhanced capacity for representing and generalizing long-range dependencies. Data and code will be available.
>
---
#### [new 056] Uniform Information Density and Syntactic Reduction: Revisiting $\textit{that}$-Mentioning in English Complement Clauses
- **分类: cs.CL**

- **简介: 该论文研究英语补语从句中that的省略现象，探讨信息密度对句法结构的影响。通过大规模语料与机器学习模型，改进信息密度测量方法，发现上下文词嵌入比传统方法更有效解释补语化用法差异。**

- **链接: [http://arxiv.org/pdf/2509.05254v1](http://arxiv.org/pdf/2509.05254v1)**

> **作者:** Hailin Hao; Elsi Kaiser
>
> **摘要:** Speakers often have multiple ways to express the same meaning. The Uniform Information Density (UID) hypothesis suggests that speakers exploit this variability to maintain a consistent rate of information transmission during language production. Building on prior work linking UID to syntactic reduction, we revisit the finding that the optional complementizer $\textit{that}$in English complement clauses is more likely to be omitted when the clause has low information density (i.e., more predictable). We advance this line of research by analyzing a large-scale, contemporary conversational corpus and using machine learning and neural language models to refine estimates of information density. Our results replicated the established relationship between information density and $\textit{that}$-mentioning. However, we found that previous measures of information density based on matrix verbs' subcategorization probability capture substantial idiosyncratic lexical variation. By contrast, estimates derived from contextual word embeddings account for additional variance in patterns of complementizer usage.
>
---
#### [new 057] VaccineRAG: Boosting Multimodal Large Language Models' Immunity to Harmful RAG Samples
- **分类: cs.CL; cs.AI**

- **简介: 论文提出VaccineRAG数据集，通过思维链分析与Partial-GRPO方法提升LLM在RAG任务中对有害样本的鲁棒性，解决检索器精度不足导致的生成质量下降问题。**

- **链接: [http://arxiv.org/pdf/2509.04502v1](http://arxiv.org/pdf/2509.04502v1)**

> **作者:** Qixin Sun; Ziqin Wang; Hengyuan Zhao; Yilin Li; Kaiyou Song; Linjiang Huang; Xiaolin Hu; Qingpei Guo; Si Liu
>
> **摘要:** Retrieval Augmented Generation enhances the response accuracy of Large Language Models (LLMs) by integrating retrieval and generation modules with external knowledge, demonstrating particular strength in real-time queries and Visual Question Answering tasks. However, the effectiveness of RAG is frequently hindered by the precision of the retriever: many retrieved samples fed into the generation phase are irrelevant or misleading, posing a critical bottleneck to LLMs' performance. To address this challenge, we introduce VaccineRAG, a novel Chain-of-Thought-based retrieval-augmented generation dataset. On one hand, VaccineRAG employs a benchmark to evaluate models using data with varying positive/negative sample ratios, systematically exposing inherent weaknesses in current LLMs. On the other hand, it enhances models' sample-discrimination capabilities by prompting LLMs to generate explicit Chain-of-Thought (CoT) analysis for each sample before producing final answers. Furthermore, to enhance the model's ability to learn long-sequence complex CoT content, we propose Partial-GRPO. By modeling the outputs of LLMs as multiple components rather than a single whole, our model can make more informed preference selections for complex sequences, thereby enhancing its capacity to learn complex CoT. Comprehensive evaluations and ablation studies on VaccineRAG validate the effectiveness of the proposed scheme. The code and dataset will be publicly released soon.
>
---
#### [new 058] Do MLLMs Really Understand the Charts?
- **分类: cs.CL**

- **简介: 该论文旨在评估MLLMs的图表理解能力，解决其在非注释图表上的幻觉与性能下降问题。通过构建CRBench基准和提出ChartReasoner模型，模仿人类视觉推理，提升图表理解能力，在多个基准上超越GPT-4o等模型。**

- **链接: [http://arxiv.org/pdf/2509.04457v1](http://arxiv.org/pdf/2509.04457v1)**

> **作者:** Xiao Zhang; Dongyuan Li; Liuyu Xiang; Yao Zhang; Cheng Zhong; Zhaofeng He
>
> **备注:** 19 pages,15 figures
>
> **摘要:** Although Multimodal Large Language Models (MLLMs) have demonstrated increasingly impressive performance in chart understanding, most of them exhibit alarming hallucinations and significant performance degradation when handling non-annotated charts. Therefore, a question arises: Do MLLMs really understand the charts? Since a human is capable of understanding charts and estimating the values by visual reasoning, we first carefully establish a comprehensive Chart Reasoning Benchmark CRBench to rigorously evaluate the visual reasoning abilities of MLLMs on non-annotated charts. We argue that MLLMs are primarily relying on recognition rather than reasoning to interpret the charts. To steer MLLMs to reasonable chart understanding, we propose ChartReasoner that mimics human behavior by grounding their estimation in chart understanding. Extensive results on the proposed CRBench show that ChartReasnoner-3B/7B achieves superior performance in chart reasoning, even compared to GPT-4o and Gemini-2.5-Flash. More importantly, ChartReasnoner also demonstrates the visual reasoning abilities in general chart comprehension on public benchmarks, leading to significant performance gains and enabling MLLMs to rationally understand the charts. The code and dataset will be publicly available upon publication.
>
---
#### [new 059] Artificially Fluent: Swahili AI Performance Benchmarks Between English-Trained and Natively-Trained Datasets
- **分类: cs.CL; cs.CY**

- **简介: 该论文比较斯瓦希里语原生训练与翻译后英语训练模型的性能，解决多语言模型公平性问题。通过实验验证原生训练模型表现更优，证明语言一致性对模型准确性的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04516v1](http://arxiv.org/pdf/2509.04516v1)**

> **作者:** Sophie Jaffer; Simeon Sayer
>
> **备注:** 13 Pages, 3 Figures
>
> **摘要:** As large language models (LLMs) expand multilingual capabilities, questions remain about the equity of their performance across languages. While many communities stand to benefit from AI systems, the dominance of English in training data risks disadvantaging non-English speakers. To test the hypothesis that such data disparities may affect model performance, this study compares two monolingual BERT models: one trained and tested entirely on Swahili data, and another on comparable English news data. To simulate how multilingual LLMs process non-English queries through internal translation and abstraction, we translated the Swahili news data into English and evaluated it using the English-trained model. This approach tests the hypothesis by evaluating whether translating Swahili inputs for evaluation on an English model yields better or worse performance compared to training and testing a model entirely in Swahili, thus isolating the effect of language consistency versus cross-lingual abstraction. The results prove that, despite high-quality translation, the native Swahili-trained model performed better than the Swahili-to-English translated model, producing nearly four times fewer errors: 0.36% vs. 1.47% respectively. This gap suggests that translation alone does not bridge representational differences between languages and that models trained in one language may struggle to accurately interpret translated inputs due to imperfect internal knowledge representation, suggesting that native-language training remains important for reliable outcomes. In educational and informational contexts, even small performance gaps may compound inequality. Future research should focus on addressing broader dataset development for underrepresented languages and renewed attention to multilingual model evaluation, ensuring the reinforcing effect of global AI deployment on existing digital divides is reduced.
>
---
#### [new 060] Knowledge Collapse in LLMs: When Fluency Survives but Facts Fail under Recursive Synthetic Training
- **分类: cs.CL**

- **简介: 该论文研究大语言模型递归合成训练导致的知识崩溃现象，定义三阶段事实准确性下降而流畅性保持的“自信错误”问题，通过实验分析崩溃机制，提出领域特定合成训练策略及评估框架以提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2509.04796v1](http://arxiv.org/pdf/2509.04796v1)**

> **作者:** Figarri Keisha; Zekun Wu; Ze Wang; Adriano Koshiyama; Philip Treleaven
>
> **摘要:** Large language models increasingly rely on synthetic data due to human-written content scarcity, yet recursive training on model-generated outputs leads to model collapse, a degenerative process threatening factual reliability. We define knowledge collapse as a distinct three-stage phenomenon where factual accuracy deteriorates while surface fluency persists, creating "confidently wrong" outputs that pose critical risks in accuracy-dependent domains. Through controlled experiments with recursive synthetic training, we demonstrate that collapse trajectory and timing depend critically on instruction format, distinguishing instruction-following collapse from traditional model collapse through its conditional, prompt-dependent nature. We propose domain-specific synthetic training as a targeted mitigation strategy that achieves substantial improvements in collapse resistance while maintaining computational efficiency. Our evaluation framework combines model-centric indicators with task-centric metrics to detect distinct degradation phases, enabling reproducible assessment of epistemic deterioration across different language models. These findings provide both theoretical insights into collapse dynamics and practical guidance for sustainable AI training in knowledge-intensive applications where accuracy is paramount.
>
---
#### [new 061] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文通过稀疏交叉编码器和RelIE指标，追踪大语言模型预训练中语言表示的演变，解决传统方法无法揭示模型能力获取的问题，实现对表示学习的细粒度可解释分析。**

- **链接: [http://arxiv.org/pdf/2509.05291v1](http://arxiv.org/pdf/2509.05291v1)**

> **作者:** Deniz Bayazit; Aaron Mueller; Antoine Bosselut
>
> **摘要:** Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining.
>
---
#### [new 062] KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出KERAG框架，解决传统知识图谱问答（KGQA）覆盖不足和语义歧义问题。通过扩展检索子图、结合过滤与总结，及微调LLM进行链式推理，提升问答质量，超越现有方法及GPT-4o。**

- **链接: [http://arxiv.org/pdf/2509.04716v1](http://arxiv.org/pdf/2509.04716v1)**

> **作者:** Yushi Sun; Kai Sun; Yifan Ethan Xu; Xiao Yang; Xin Luna Dong; Nan Tang; Lei Chen
>
> **备注:** Accepted by EMNLP Findings 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates hallucination in Large Language Models (LLMs) by incorporating external data, with Knowledge Graphs (KGs) offering crucial information for question answering. Traditional Knowledge Graph Question Answering (KGQA) methods rely on semantic parsing, which typically retrieves knowledge strictly necessary for answer generation, thus often suffer from low coverage due to rigid schema requirements and semantic ambiguity. We present KERAG, a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. Our retrieval-filtering-summarization approach, combined with fine-tuned LLMs for Chain-of-Thought reasoning on knowledge sub-graphs, reduces noises and improves QA for both simple and complex questions. Experiments demonstrate that KERAG surpasses state-of-the-art solutions by about 7% in quality and exceeds GPT-4o (Tool) by 10-21%.
>
---
#### [new 063] Advancing SLM Tool-Use Capability using Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文旨在通过强化学习提升小语言模型（SLM）的工具使用能力，解决其资源需求大、效率低及传统微调方法不足的问题，采用GRPO方法显著提高SLM工具使用准确率与实用性。**

- **链接: [http://arxiv.org/pdf/2509.04518v1](http://arxiv.org/pdf/2509.04518v1)**

> **作者:** Dhruvi Paprunia; Vansh Kharidia; Pankti Doshi
>
> **摘要:** Large Language Models (LLMs) have progressed beyond simple text creation, and tool use has become increasingly important for complex, real-world tasks. Tool use in LLMs refers to their ability to utilize external resources such as APIs, databases, or software functions to extend their functionality beyond generating text.Tools are used for tasks such as performing calculations, making API calls to retrieve the current time and date, and more. This capability enables models to fetch real-time data, execute commands, or solve problems requiring dynamic interaction, making it indispensable for applications like AI agents in virtual assistants, robotic control, or automated workflows. However, while LLMs are usually adept tool use, their vast resource requirements and computation complexity restrict their use in every use case.As a result, there is an increasing need for more compact and efficient Small Language Models (SLMs). Small language models (SLMs) struggle in tool use compared to large language models (LLMs). As soon in Table 1. SLMs are typically trained on smaller, more specific datasets, resulting in a narrower knowledge base and limited contextual understanding compared to LLMs. This research addresses these challenges by using Reinforcement Learning (RL), specifically Group Relative Policy Optimization (GRPO), to enhance tool-use proficiency in SLMs. Unlike conventional fine-tuning approaches that require heavy computation and often lack adaptability, our method provides an efficient, effective solution that significantly boosts SLM tool-use accuracy, increasing their practical utility.
>
---
#### [new 064] Context Engineering for Trustworthiness: Rescorla Wagner Steering Under Mixed and Inappropriate Contexts
- **分类: cs.CL; cs.AI**

- **简介: 论文针对LLM在混合上下文中易受不适当内容影响的问题，设计Poisoned Context Testbed，基于Rescorla-Wagner模型分析机制，提出RW-Steering微调方法，提升响应质量39.8%，增强模型可靠性。**

- **链接: [http://arxiv.org/pdf/2509.04500v1](http://arxiv.org/pdf/2509.04500v1)**

> **作者:** Rushi Wang; Jiateng Liu; Cheng Qian; Yifan Shen; Yanzhou Pan; Zhaozhuo Xu; Ahmed Abbasi; Heng Ji; Denghui Zhang
>
> **备注:** 36 pages, 7 figures
>
> **摘要:** Incorporating external context can significantly enhance the response quality of Large Language Models (LLMs). However, real-world contexts often mix relevant information with disproportionate inappropriate content, posing reliability risks. How do LLMs process and prioritize mixed context? To study this, we introduce the Poisoned Context Testbed, pairing queries with real-world contexts containing relevant and inappropriate content. Inspired by associative learning in animals, we adapt the Rescorla-Wagner (RW) model from neuroscience to quantify how competing contextual signals influence LLM outputs. Our adapted model reveals a consistent behavioral pattern: LLMs exhibit a strong tendency to incorporate information that is less prevalent in the context. This susceptibility is harmful in real-world settings, where small amounts of inappropriate content can substantially degrade response quality. Empirical evaluations on our testbed further confirm this vulnerability. To tackle this, we introduce RW-Steering, a two-stage finetuning-based approach that enables the model to internally identify and ignore inappropriate signals. Unlike prior methods that rely on extensive supervision across diverse context mixtures, RW-Steering generalizes robustly across varying proportions of inappropriate content. Experiments show that our best fine-tuned model improves response quality by 39.8% and reverses the undesirable behavior curve, establishing RW-Steering as a robust, generalizable context engineering solution for improving LLM safety in real-world use.
>
---
#### [new 065] MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification
- **分类: cs.CL; cs.AI**

- **简介: 本文提出MOSAIC，一种多语言、分类体系无关、计算高效的放射报告分类方法。针对现有方法在语言多样性、标注数据依赖和资源消耗上的不足，基于MedGemma-4B小模型，支持零/少样本提示与轻量微调，实现跨语言、多模态分类，开源代码与模型，适用于临床场景。**

- **链接: [http://arxiv.org/pdf/2509.04471v1](http://arxiv.org/pdf/2509.04471v1)**

> **作者:** Alice Schiavone; Marco Fraccaro; Lea Marie Pehrson; Silvia Ingala; Rasmus Bonnevie; Michael Bachmann Nielsen; Vincent Beliveau; Melanie Ganz; Desmond Elliott
>
> **备注:** 8 pages, 14 pages including references and appendix. 9 figures. Preprint
>
> **摘要:** Radiology reports contain rich clinical information that can be used to train imaging models without relying on costly manual annotation. However, existing approaches face critical limitations: rule-based methods struggle with linguistic variability, supervised models require large annotated datasets, and recent LLM-based systems depend on closed-source or resource-intensive models that are unsuitable for clinical use. Moreover, current solutions are largely restricted to English and single-modality, single-taxonomy datasets. We introduce MOSAIC, a multilingual, taxonomy-agnostic, and computationally efficient approach for radiological report classification. Built on a compact open-access language model (MedGemma-4B), MOSAIC supports both zero-/few-shot prompting and lightweight fine-tuning, enabling deployment on consumer-grade GPUs. We evaluate MOSAIC across seven datasets in English, Spanish, French, and Danish, spanning multiple imaging modalities and label taxonomies. The model achieves a mean macro F1 score of 88 across five chest X-ray datasets, approaching or exceeding expert-level performance, while requiring only 24 GB of GPU memory. With data augmentation, as few as 80 annotated samples are sufficient to reach a weighted F1 score of 82 on Danish reports, compared to 86 with the full 1600-sample training set. MOSAIC offers a practical alternative to large or proprietary LLMs in clinical settings. Code and models are open-source. We invite the community to evaluate and extend MOSAIC on new languages, taxonomies, and modalities.
>
---
#### [new 066] Uncertainty-Aware Collaborative System of Large and Small Models for Multimodal Sentiment Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对多模态情感分析任务，解决大模型计算成本高与小模型性能差的矛盾。提出U-ACS系统，通过不确定性驱动的级联机制与冲突处理策略，动态分配计算资源，兼顾效率与精度。**

- **链接: [http://arxiv.org/pdf/2509.04459v1](http://arxiv.org/pdf/2509.04459v1)**

> **作者:** Shiqin Han; Manning Gao; Menghua Jiang; Yuncheng Jiang; Haifeng Hu; Sijie Mai
>
> **摘要:** The advent of Multimodal Large Language Models (MLLMs) has significantly advanced the state-of-the-art in multimodal machine learning, yet their substantial computational demands present a critical barrier to real-world deployment. Conversely, smaller, specialized models offer high efficiency but often at the cost of performance. To reconcile this performance-efficiency trade-off, we propose a novel Uncertainty-Aware Collaborative System (U-ACS) that synergistically orchestrates a powerful MLLM (e.g., HumanOmni) and a lightweight baseline model for multimodal sentiment analysis. The core of our system is an uncertainty-driven cascade mechanism, where the efficient small model first acts as a rapid filter for all input samples. Only those samples yielding high predictive uncertainty, thereby indicating greater difficulty, are selectively escalated to the MLLM for more sophisticated analysis. Furthermore, our system introduces advanced strategies to handle ambiguous or conflicting predictions, including weighted averaging for predictions of similar polarity and a prompt-based cross-verification to resolve conflicting predictions when both models exhibit high uncertainty. This sample-difficulty-aware approach allows for a dynamic allocation of computational resources, drastically reducing inference costs while retaining the high accuracy of MLLM. Extensive experiments on benchmark datasets demonstrate that our proposed method achieves state-of-the-art performance, while requiring only a fraction of the computational resources compared to using a standalone MLLM.
>
---
#### [new 067] Phonological Representation Learning for Isolated Signs Improves Out-of-Vocabulary Generalization
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对手语孤立词识别的未见词汇泛化问题，提出结合参数解耦与语音半监督的向量量化自编码器，提升单次重建与识别效果。**

- **链接: [http://arxiv.org/pdf/2509.04745v1](http://arxiv.org/pdf/2509.04745v1)**

> **作者:** Lee Kezar; Zed Sehyr; Jesse Thomason
>
> **摘要:** Sign language datasets are often not representative in terms of vocabulary, underscoring the need for models that generalize to unseen signs. Vector quantization is a promising approach for learning discrete, token-like representations, but it has not been evaluated whether the learned units capture spurious correlations that hinder out-of-vocabulary performance. This work investigates two phonological inductive biases: Parameter Disentanglement, an architectural bias, and Phonological Semi-Supervision, a regularization technique, to improve isolated sign recognition of known signs and reconstruction quality of unseen signs with a vector-quantized autoencoder. The primary finding is that the learned representations from the proposed model are more effective for one-shot reconstruction of unseen signs and more discriminative for sign identification compared to a controlled baseline. This work provides a quantitative analysis of how explicit, linguistically-motivated biases can improve the generalization of learned representations of sign language.
>
---
#### [new 068] Spoken in Jest, Detected in Earnest: A Systematic Review of Sarcasm Recognition -- Multimodal Fusion, Challenges, and Future Prospects
- **分类: cs.CL**

- **简介: 该论文是语音讽刺识别的系统综述，分析多模态方法、数据集与特征提取技术，指出数据集局限，强调跨文化研究的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04605v1](http://arxiv.org/pdf/2509.04605v1)**

> **作者:** Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **备注:** 20 pages, 7 figures, Submitted to IEEE Transactions on Affective Computing
>
> **摘要:** Sarcasm, a common feature of human communication, poses challenges in interpersonal interactions and human-machine interactions. Linguistic research has highlighted the importance of prosodic cues, such as variations in pitch, speaking rate, and intonation, in conveying sarcastic intent. Although previous work has focused on text-based sarcasm detection, the role of speech data in recognizing sarcasm has been underexplored. Recent advancements in speech technology emphasize the growing importance of leveraging speech data for automatic sarcasm recognition, which can enhance social interactions for individuals with neurodegenerative conditions and improve machine understanding of complex human language use, leading to more nuanced interactions. This systematic review is the first to focus on speech-based sarcasm recognition, charting the evolution from unimodal to multimodal approaches. It covers datasets, feature extraction, and classification methods, and aims to bridge gaps across diverse research domains. The findings include limitations in datasets for sarcasm recognition in speech, the evolution of feature extraction techniques from traditional acoustic features to deep learning-based representations, and the progression of classification methods from unimodal approaches to multimodal fusion techniques. In so doing, we identify the need for greater emphasis on cross-cultural and multilingual sarcasm recognition, as well as the importance of addressing sarcasm as a multimodal phenomenon, rather than a text-based challenge.
>
---
#### [new 069] Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出基于Dropout容忍度的ICAD框架，用于专门化LLMs的OOD检测。通过聚合多层dropout容忍度，提升检测效果，实验证明优于基线方法，AUROC提升2%-37%。**

- **链接: [http://arxiv.org/pdf/2509.04655v1](http://arxiv.org/pdf/2509.04655v1)**

> **作者:** Ayush Gupta; Ramneet Kaur; Anirban Roy; Adam D. Cobb; Rama Chellappa; Susmit Jha
>
> **备注:** Accepted to EMNLP 2025 main conference
>
> **摘要:** We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives.
>
---
#### [new 070] L1RA: Dynamic Rank Assignment in LoRA Fine-Tuning
- **分类: cs.CL; cs.PF**

- **简介: 论文提出L1RA，通过动态秩分配和L1正则化优化LoRA微调，解决资源受限下的高效模型适配问题，减少计算开销并提升性能，同时提供组件诊断信息。**

- **链接: [http://arxiv.org/pdf/2509.04884v1](http://arxiv.org/pdf/2509.04884v1)**

> **作者:** Raul Singh; Nicolo Brunello; Vincenzo Scotti; Mark James Carman
>
> **备注:** Work published at ICNLSP 2025, waiting for publication link
>
> **摘要:** The ability of Large Language Models (LLMs) to solve complex tasks has made them crucial in the development of AI-based applications. However, the high computational requirements to fine-tune these LLMs on downstream tasks pose significant challenges, particularly when resources are limited. In response to this challenge, we introduce L1RA, a novel technique aimed at dynamically distributing the rank of low-rank adapters during fine-tuning using LoRA. Given a rank budget (i.e., total sum of adapters rank), L1RA leverages L1 regularisation to prune redundant ranks and redistribute them across adapters, thereby optimising resource utilisation. Through a series of comprehensive experiments, we empirically demonstrate that L1RA maintains comparable or even reduced computational overhead compared to other LoRA variants, including the vanilla approach, while achieving same or better performances. Moreover, the post-training analysis of rank distribution unveiled insights into the specific model components requiring the most adaptation to align with the task objective: the feed-forward layers and the attention output projection. These results highlight the efficacy of L1RA in not only enhancing the efficiency of LLM fine-tuning, but also in providing valuable diagnostic information for model refinement and customisation. In conclusion, L1RA stands as a promising technique for advancing the performance and interpretability of LLM adaptation, particularly in scenarios where computational resources are constrained.
>
---
#### [new 071] Decoders Laugh as Loud as Encoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究解码器与编码器在幽默理解任务中的表现，解决计算机是否真正理解幽默的问题。通过微调GPT-4o与RoBERTa，验证解码器在幽默识别上接近编码器水平（F1-score 0.85 vs 0.86）。**

- **链接: [http://arxiv.org/pdf/2509.04779v1](http://arxiv.org/pdf/2509.04779v1)**

> **作者:** Eli Borodach; Raj Dandekar; Rajat Dandekar; Sreedath Panat
>
> **摘要:** From the dawn of the computer, Allen Turing dreamed of a robot that could communicate using language as a human being. The recent advances in the field of Large Language Models (LLMs) shocked the scientific community when a single model can apply for various natural language processing (NLP) tasks, while the output results are sometimes even better than most human communication skills. Models such as GPT, Claude, Grok, etc. have left their mark on the scientific community. However, it is unclear how much these models understand what they produce, especially in a nuanced theme such as humor. The question of whether computers understand humor is still open (among the decoders, the latest to be checked was GPT-2). We addressed this issue in this paper; we have showed that a fine-tuned decoder (GPT-4o) performed (Mean F1-macro score of 0.85) as well as the best fine-tuned encoder (RoBERTa with a Mean of F1-score 0.86)
>
---
#### [new 072] DecMetrics: Structured Claim Decomposition Scoring for Factually Consistent LLM Outputs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DecMetrics，解决事实核查中分解声明质量评估不足的问题。通过COMPLETENESS、CORRECTNESS和SEMANTIC ENTROPY三个新指标，构建轻量级模型优化分解效果，提升事实核查可靠性。**

- **链接: [http://arxiv.org/pdf/2509.04483v1](http://arxiv.org/pdf/2509.04483v1)**

> **作者:** Minghui Huang
>
> **摘要:** Claim decomposition plays a crucial role in the fact-checking process by breaking down complex claims into simpler atomic components and identifying their unfactual elements. Despite its importance, current research primarily focuses on generative methods for decomposition, with insufficient emphasis on evaluating the quality of these decomposed atomic claims. To bridge this gap, we introduce \textbf{DecMetrics}, which comprises three new metrics: \texttt{COMPLETENESS}, \texttt{CORRECTNESS}, and \texttt{SEMANTIC ENTROPY}, designed to automatically assess the quality of claims produced by decomposition models. Utilizing these metrics, we develop a lightweight claim decomposition model, optimizing its performance through the integration of these metrics as a reward function. Through automatic evaluation, our approach aims to set a benchmark for claim decomposition, enhancing both the reliability and effectiveness of fact-checking systems.
>
---
#### [new 073] Understanding Reinforcement Learning for Model Training, and future directions with GRAPE
- **分类: cs.CL; cs.AI; cs.LG; 68T07, 68T05, 62M45, 68T50, 90C40; I.2.6; I.2.7**

- **简介: 该论文系统解析强化学习算法（如PPO、TRPO等）在模型训练中的应用，解决现有资料复杂、缺乏细节的问题，通过简化符号和LLM视角阐述方法，并提出GRAPE框架探索新方向。**

- **链接: [http://arxiv.org/pdf/2509.04501v1](http://arxiv.org/pdf/2509.04501v1)**

> **作者:** Rohit Patel
>
> **备注:** 35 pages, 1 figure
>
> **摘要:** This paper provides a self-contained, from-scratch, exposition of key algorithms for instruction tuning of models: SFT, Rejection Sampling, REINFORCE, Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Explanations of these algorithms often assume prior knowledge, lack critical details, and/or are overly generalized and complex. Here, each method is discussed and developed step by step using simplified and explicit notation focused on LLMs, aiming to eliminate ambiguity and provide a clear and intuitive understanding of the concepts. By minimizing detours into the broader RL literature and connecting concepts to LLMs, we eliminate superfluous abstractions and reduce cognitive overhead. Following this exposition, we provide a literature review of new techniques and approaches beyond those detailed. Finally, new ideas for research and exploration in the form of GRAPE (Generalized Relative Advantage Policy Evolution) are presented.
>
---
#### [new 074] PRIM: Towards Practical In-Image Multilingual Machine Translation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出PRIM数据集与VisTrans模型，解决图像内多语言翻译任务中合成数据与真实场景差距的问题，提升多语言翻译质量与视觉效果。**

- **链接: [http://arxiv.org/pdf/2509.05146v1](http://arxiv.org/pdf/2509.05146v1)**

> **作者:** Yanzhi Tian; Zeming Liu; Zhengyang Liu; Chong Feng; Xin Li; Heyan Huang; Yuhang Guo
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** In-Image Machine Translation (IIMT) aims to translate images containing texts from one language to another. Current research of end-to-end IIMT mainly conducts on synthetic data, with simple background, single font, fixed text position, and bilingual translation, which can not fully reflect real world, causing a significant gap between the research and practical conditions. To facilitate research of IIMT in real-world scenarios, we explore Practical In-Image Multilingual Machine Translation (IIMMT). In order to convince the lack of publicly available data, we annotate the PRIM dataset, which contains real-world captured one-line text images with complex background, various fonts, diverse text positions, and supports multilingual translation directions. We propose an end-to-end model VisTrans to handle the challenge of practical conditions in PRIM, which processes visual text and background information in the image separately, ensuring the capability of multilingual translation while improving the visual quality. Experimental results indicate the VisTrans achieves a better translation quality and visual effect compared to other models. The code and dataset are available at: https://github.com/BITHLP/PRIM.
>
---
#### [new 075] Masked Diffusion Language Models with Frequency-Informed Training
- **分类: cs.CL**

- **简介: 该论文提出基于扩散模型的掩码语言建模框架，通过频率引导的掩码策略和噪声调度优化，在数据受限场景下提升模型性能，验证了扩散训练在语言学习中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.05056v1](http://arxiv.org/pdf/2509.05056v1)**

> **作者:** Despoina Kosmopoulou; Efthymios Georgiou; Vaggelis Dorovatas; Georgios Paraskevopoulos; Alexandros Potamianos
>
> **备注:** Preprint
>
> **摘要:** We present a masked diffusion language modeling framework for data-efficient training for the BabyLM 2025 Challenge. Our approach applies diffusion training objectives to language modeling under strict data constraints, incorporating frequency-informed masking that prioritizes learning from rare tokens while maintaining theoretical validity. We explore multiple noise scheduling strategies, including two-mode approaches, and investigate different noise weighting schemes within the NELBO objective. We evaluate our method on the BabyLM benchmark suite, measuring linguistic competence, world knowledge, and human-likeness. Results show performance competitive to hybrid autoregressive-masked baselines, demonstrating that diffusion-based training offers a viable alternative for data-restricted language learning.
>
---
#### [new 076] A Narrative-Driven Computational Framework for Clinician Burnout Surveillance
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出基于临床叙事的计算框架，解决传统方法忽视叙事信息的问题，通过整合BioBERT、压力词典和LDA模型，实现对ICU医生倦怠的高精度监测，发现放射科等专科风险较高。**

- **链接: [http://arxiv.org/pdf/2509.04497v1](http://arxiv.org/pdf/2509.04497v1)**

> **作者:** Syed Ahmad Chan Bukhari; Fazel Keshtkar; Alyssa Meczkowska
>
> **备注:** 6 pages, 6 Figure
>
> **摘要:** Clinician burnout poses a substantial threat to patient safety, particularly in high-acuity intensive care units (ICUs). Existing research predominantly relies on retrospective survey tools or broad electronic health record (EHR) metadata, often overlooking the valuable narrative information embedded in clinical notes. In this study, we analyze 10,000 ICU discharge summaries from MIMIC-IV, a publicly available database derived from the electronic health records of Beth Israel Deaconess Medical Center. The dataset encompasses diverse patient data, including vital signs, medical orders, diagnoses, procedures, treatments, and deidentified free-text clinical notes. We introduce a hybrid pipeline that combines BioBERT sentiment embeddings fine-tuned for clinical narratives, a lexical stress lexicon tailored for clinician burnout surveillance, and five-topic latent Dirichlet allocation (LDA) with workload proxies. A provider-level logistic regression classifier achieves a precision of 0.80, a recall of 0.89, and an F1 score of 0.84 on a stratified hold-out set, surpassing metadata-only baselines by greater than or equal to 0.17 F1 score. Specialty-specific analysis indicates elevated burnout risk among providers in Radiology, Psychiatry, and Neurology. Our findings demonstrate that ICU clinical narratives contain actionable signals for proactive well-being monitoring.
>
---
#### [new 077] Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出基于能量景观的模型，用于医疗领域RAG系统的可靠退避机制，解决安全关键场景下错误回答的风险。通过学习语义语料库的能量分布，提升退避性能，优于softmax和kNN方法。**

- **链接: [http://arxiv.org/pdf/2509.04482v1](http://arxiv.org/pdf/2509.04482v1)**

> **作者:** Ravi Shankar; Sheng Wong; Lin Li; Magdalena Bachmann; Alex Silverthorne; Beth Albert; Gabriel Davis Jones
>
> **摘要:** Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women's health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchmark the EBM against a calibrated softmax baseline and a k-nearest neighbour (kNN) density heuristic across both easy and hard abstention splits, where hard cases are semantically challenging near-distribution queries. The EBM achieves superior abstention performance abstention on semantically hard cases, reaching AUROC 0.961 versus 0.950 for softmax, while also reducing FPR@95 (0.235 vs 0.331). On easy negatives, performance is comparable across methods, but the EBM's advantage becomes most pronounced in safety-critical hard distributions. A comprehensive ablation with controlled negative sampling and fair data exposure shows that robustness stems primarily from the energy scoring head, while the inclusion or exclusion of specific negative types (hard, easy, mixed) sharpens decision boundaries but is not essential for generalisation to hard cases. These results demonstrate that energy-based abstention scoring offers a more reliable confidence signal than probability-based softmax confidence, providing a scalable and interpretable foundation for safe RAG systems.
>
---
#### [new 078] CURE: Controlled Unlearning for Robust Embeddings -- Mitigating Conceptual Shortcuts in Pre-Trained Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出CURE框架，旨在解决预训练语言模型中因概念捷径导致的鲁棒性与公平性问题。通过分离无关表示并抑制残余概念线索，结合对比学习调整偏倚，提升模型在IMDB和Yelp数据集上的F1分数，同时保持计算效率。**

- **链接: [http://arxiv.org/pdf/2509.05230v1](http://arxiv.org/pdf/2509.05230v1)**

> **作者:** Aysenur Kocak; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **备注:** Accepted at the Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Pre-trained language models have achieved remarkable success across diverse applications but remain susceptible to spurious, concept-driven correlations that impair robustness and fairness. In this work, we introduce CURE, a novel and lightweight framework that systematically disentangles and suppresses conceptual shortcuts while preserving essential content information. Our method first extracts concept-irrelevant representations via a dedicated content extractor reinforced by a reversal network, ensuring minimal loss of task-relevant information. A subsequent controllable debiasing module employs contrastive learning to finely adjust the influence of residual conceptual cues, enabling the model to either diminish harmful biases or harness beneficial correlations as appropriate for the target task. Evaluated on the IMDB and Yelp datasets using three pre-trained architectures, CURE achieves an absolute improvement of +10 points in F1 score on IMDB and +2 points on Yelp, while introducing minimal computational overhead. Our approach establishes a flexible, unsupervised blueprint for combating conceptual biases, paving the way for more reliable and fair language understanding systems.
>
---
#### [new 079] Mitigation of Gender and Ethnicity Bias in AI-Generated Stories through Model Explanations
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对AI生成故事中的性别与种族偏见问题，提出BAME方法，通过模型解释指导提示工程，减少偏见而不修改参数。分析三类模型及25职业群体数据，发现训练数据刻板印象导致的过代表/欠代表问题，实现2%-20%的改进。**

- **链接: [http://arxiv.org/pdf/2509.04515v1](http://arxiv.org/pdf/2509.04515v1)**

> **作者:** Martha O. Dimgba; Sharon Oba; Ameeta Agrawal; Philippe J. Giabbanelli
>
> **摘要:** Language models have been shown to propagate social bias through their output, particularly in the representation of gender and ethnicity. This paper investigates gender and ethnicity biases in AI-generated occupational stories. Representation biases are measured before and after applying our proposed mitigation strategy, Bias Analysis and Mitigation through Explanation (BAME), revealing improvements in demographic representation ranging from 2% to 20%. BAME leverages model-generated explanations to inform targeted prompt engineering, effectively reducing biases without modifying model parameters. By analyzing stories generated across 25 occupational groups, three large language models (Claude 3.5 Sonnet, Llama 3.1 70B Instruct, and GPT-4 Turbo), and multiple demographic dimensions, we identify persistent patterns of overrepresentation and underrepresentation linked to training data stereotypes. Our findings demonstrate that guiding models with their own internal reasoning mechanisms can significantly enhance demographic parity, thereby contributing to the development of more transparent generative AI systems.
>
---
#### [new 080] Benchmarking GPT-5 for biomedical natural language processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估GPT-5在生物医学NLP任务中的表现，对比GPT-4、GPT-4o等模型，测试其在命名实体识别、关系抽取等6类任务中的性能，揭示其优势与局限，为BioNLP系统设计提供指导。**

- **链接: [http://arxiv.org/pdf/2509.04462v1](http://arxiv.org/pdf/2509.04462v1)**

> **作者:** Yu Hou; Zaifu Zhan; Rui Zhang
>
> **摘要:** The rapid expansion of biomedical literature has heightened the need for scalable natural language processing (NLP) solutions. While GPT-4 substantially narrowed the gap with task-specific systems, especially in question answering, its performance across other domains remained uneven. We updated a standardized BioNLP benchmark to evaluate GPT-5 and GPT-4o under zero-, one-, and five-shot prompting across 12 datasets spanning six task families: named entity recognition, relation extraction, multi-label document classification, question answering, text summarization, and text simplification. Using fixed prompt templates, identical decoding parameters, and batch inference, we report primary metrics per dataset and include prior results for GPT-4, GPT-3.5, and LLaMA-2-13B for comparison. GPT-5 achieved the strongest overall benchmark performance, with macro-average scores rising to 0.557 under five-shot prompting versus 0.506 for GPT-4 and 0.508 for GPT-4o. On MedQA, GPT-5 reached 94.1% accuracy, exceeding the previous supervised state of the art by over fifty points, and attained parity with supervised systems on PubMedQA (0.734). In extraction tasks, GPT-5 delivered major gains in chemical NER (0.886 F1) and ChemProt relation extraction (0.616 F1), outperforming GPT-4 and GPT-4o, though summarization and disease NER still lagged behind domain-specific baselines. These results establish GPT-5 as a general-purpose model now offering deployment-ready performance for reasoning-oriented biomedical QA, while precision-critical extraction and evidence-dense summarization continue to favor fine-tuned or hybrid approaches. The benchmark delineates where simple prompting suffices and where retrieval-augmented or planning-based scaffolds are likely required, providing actionable guidance for BioNLP system design as frontier models advance.
>
---
#### [new 081] Training Text-to-Molecule Models with Context-Aware Tokenization
- **分类: cs.CL; cs.AI**

- **简介: 本文提出Context-Aware Molecular T5 (CAMT5)，通过子结构级分词和重要性训练策略，解决文本到分子模型中原子级分词限制全局结构理解的问题，提升生成性能。**

- **链接: [http://arxiv.org/pdf/2509.04476v1](http://arxiv.org/pdf/2509.04476v1)**

> **作者:** Seojin Kim; Hyeontae Song; Jaehyun Nam; Jinwoo Shin
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Recently, text-to-molecule models have shown great potential across various chemical applications, e.g., drug-discovery. These models adapt language models to molecular data by representing molecules as sequences of atoms. However, they rely on atom-level tokenizations, which primarily focus on modeling local connectivity, thereby limiting the ability of models to capture the global structural context within molecules. To tackle this issue, we propose a novel text-to-molecule model, coined Context-Aware Molecular T5 (CAMT5). Inspired by the significance of the substructure-level contexts in understanding molecule structures, e.g., ring systems, we introduce substructure-level tokenization for text-to-molecule models. Building on our tokenization scheme, we develop an importance-based training strategy that prioritizes key substructures, enabling CAMT5 to better capture the molecular semantics. Extensive experiments verify the superiority of CAMT5 in various text-to-molecule generation tasks. Intriguingly, we find that CAMT5 outperforms the state-of-the-art methods using only 2% of training tokens. In addition, we propose a simple yet effective ensemble strategy that aggregates the outputs of text-to-molecule models to further boost the generation performance. Code is available at https://github.com/Songhyeontae/CAMT5.git.
>
---
#### [new 082] Emotionally-Aware Agents for Dispute Resolution
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究情感识别在纠纷解决中的应用，分析买家卖家对话数据，比较大语言模型与传统方法的效果，验证情感表达对冲突升级的影响，并提出基于代理的系统以缓解情感升级。**

- **链接: [http://arxiv.org/pdf/2509.04465v1](http://arxiv.org/pdf/2509.04465v1)**

> **作者:** Sushrita Rakshit; James Hale; Kushal Chawla; Jeanne M. Brett; Jonathan Gratch
>
> **摘要:** In conflict, people use emotional expressions to shape their counterparts' thoughts, feelings, and actions. This paper explores whether automatic text emotion recognition offers insight into this influence in the context of dispute resolution. Prior work has shown the promise of such methods in negotiations; however, disputes evoke stronger emotions and different social processes. We use a large corpus of buyer-seller dispute dialogues to investigate how emotional expressions shape subjective and objective outcomes. We further demonstrate that large-language models yield considerably greater explanatory power than previous methods for emotion intensity annotation and better match the decisions of human annotators. Findings support existing theoretical models for how emotional expressions contribute to conflict escalation and resolution and suggest that agent-based systems could be useful in managing disputes by recognizing and potentially mitigating emotional escalation.
>
---
#### [new 083] Sample-efficient Integration of New Modalities into Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出SEMI方法，解决低资源模态集成效率低的问题。通过超网络动态适配投影器，结合等距变换扩展编码器多样性，显著降低新模态集成所需数据量，提升大模型多模态扩展能力。**

- **链接: [http://arxiv.org/pdf/2509.04606v1](http://arxiv.org/pdf/2509.04606v1)**

> **作者:** Osman Batur İnce; André F. T. Martins; Oisin Mac Aodha; Edoardo M. Ponti
>
> **备注:** Pre-print
>
> **摘要:** Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector -- placed between modality-specific encoders and an LLM -- to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially multiply the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models.
>
---
#### [new 084] Multi-Modal Vision vs. Text-Based Parsing: Benchmarking LLM Strategies for Invoice Processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文比较多模态模型与文本解析方法在发票处理中的效果，测试八种模型在三种数据集上的表现，发现直接图像处理优于结构化解析，为自动化文档系统提供模型选择依据。**

- **链接: [http://arxiv.org/pdf/2509.04469v1](http://arxiv.org/pdf/2509.04469v1)**

> **作者:** David Berghaus; Armin Berger; Lars Hillebrand; Kostadin Cvejoski; Rafet Sifa
>
> **摘要:** This paper benchmarks eight multi-modal large language models from three families (GPT-5, Gemini 2.5, and open-source Gemma 3) on three diverse openly available invoice document datasets using zero-shot prompting. We compare two processing strategies: direct image processing using multi-modal capabilities and a structured parsing approach converting documents to markdown first. Results show native image processing generally outperforms structured approaches, with performance varying across model types and document characteristics. This benchmark provides insights for selecting appropriate models and processing strategies for automated document systems. Our code is available online.
>
---
#### [new 085] Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs
- **分类: cs.CL; cs.CR; cs.LG; 68T07, 68T50; I.2.7; I.2.6; K.6.5**

- **简介: 该论文系统综述了基于提示的攻击方法，建立LLM安全威胁模型，旨在解决对抗攻击、安全对齐等问题，通过分类攻击机制与影响，指导构建抗篡改、防滥用的下一代安全大模型。**

- **链接: [http://arxiv.org/pdf/2509.04615v1](http://arxiv.org/pdf/2509.04615v1)**

> **作者:** Brennen Hill; Surendra Parla; Venkata Abhijeeth Balabhadruni; Atharv Prajod Padmalayam; Sujay Chandra Shekara Sharma
>
> **摘要:** The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.
>
---
#### [new 086] Hunyuan-MT Technical Report
- **分类: cs.CL**

- **简介: 该论文提出多语言翻译模型Hunyuan-MT-7B及Chimera-7B，支持33种语言双向翻译，重点优化普通话与少数民族语言、方言的翻译性能。通过混合参数输出与强化学习训练，显著提升翻译质量，在WMT2025任务中获31种语言对榜首。**

- **链接: [http://arxiv.org/pdf/2509.05209v1](http://arxiv.org/pdf/2509.05209v1)**

> **作者:** Mao Zheng; Zheng Li; Bingxin Qu; Mingyang Song; Yang Du; Mingrui Sun; Di Wang
>
> **摘要:** In this report, we introduce Hunyuan-MT-7B, our first open-source multilingual translation model, which supports bidirectional translation across 33 major languages and places a special emphasis on translation between Mandarin and several ethnic minority languages as well as dialects. Furthermore, to serve and address diverse translation scenarios and enhance model performance at test time, we introduce Hunyuan-MT-Chimera-7B, a translation model inspired by the slow thinking mode. This model integrates multiple outputs generated by the Hunyuan-MT-7B model under varying parameter settings, thereby achieving performance superior to that of conventional slow-thinking models based on Chain-of-Thought (CoT). The development of our models follows a holistic training process specifically engineered for multilingual translation, which begins with general and MT-oriented pre-training to build foundational capabilities, proceeds to Supervised Fine-Tuning (SFT) for task-specific adaptation, and culminates in advanced alignment through Reinforcement Learning (RL) and weak-to-strong RL. Through comprehensive experimentation, we demonstrate that both Hunyuan-MT-7B and Hunyuan-MT-Chimera-7B significantly outperform all translation-specific models of comparable parameter size and most of the SOTA large models, particularly on the task of translation between Mandarin and minority languages as well as dialects. In the WMT2025 shared task (General Machine Translation), our models demonstrate state-of-the-art performance, ranking first in 30 out of 31 language pairs. This result highlights the robustness of our models across a diverse linguistic spectrum, encompassing high-resource languages such as Chinese, English, and Japanese, as well as low-resource languages including Czech, Marathi, Estonian, and Icelandic.
>
---
#### [new 087] Hierarchical Section Matching Prediction (HSMP) BERT for Fine-Grained Extraction of Structured Data from Hebrew Free-Text Radiology Reports in Crohn's Disease
- **分类: cs.CL**

- **简介: 该论文提出HSMP-BERT模型，解决低资源语言（希伯来语）下克罗恩病放射报告中多器官结构化信息提取难题，通过分层推理提升效率，实现高精度细粒度标注，并发现病变关联与人群趋势。**

- **链接: [http://arxiv.org/pdf/2509.04519v1](http://arxiv.org/pdf/2509.04519v1)**

> **作者:** Zvi Badash; Hadas Ben-Atya; Naama Gavrielov; Liam Hazan; Gili Focht; Ruth Cytter-Kuint; Talar Hagopian; Dan Turner; Moti Freiman
>
> **摘要:** Extracting structured clinical information from radiology reports is challenging, especially in low-resource languages. This is pronounced in Crohn's disease, with sparsely represented multi-organ findings. We developed Hierarchical Structured Matching Prediction BERT (HSMP-BERT), a prompt-based model for extraction from Hebrew radiology text. In an administrative database study, we analyzed 9,683 reports from Crohn's patients imaged 2010-2023 across Israeli providers. A subset of 512 reports was radiologist-annotated for findings across six gastrointestinal organs and 15 pathologies, yielding 90 structured labels per subject. Multilabel-stratified split (66% train+validation; 33% test), preserving label prevalence. Performance was evaluated with accuracy, F1, Cohen's $\kappa$, AUC, PPV, NPV, and recall. On 24 organ-finding combinations with $>$15 positives, HSMP-BERT achieved mean F1 0.83$\pm$0.08 and $\kappa$ 0.65$\pm$0.17, outperforming the SMP zero-shot baseline (F1 0.49$\pm$0.07, $\kappa$ 0.06$\pm$0.07) and standard fine-tuning (F1 0.30$\pm$0.27, $\kappa$ 0.27$\pm$0.34; paired t-test $p < 10^{-7}$). Hierarchical inference cuts runtime 5.1$\times$ vs. traditional inference. Applied to all reports, it revealed associations among ileal wall thickening, stenosis, and pre-stenotic dilatation, plus age- and sex-specific trends in inflammatory findings. HSMP-BERT offers a scalable solution for structured extraction in radiology, enabling population-level analysis of Crohn's disease and demonstrating AI's potential in low-resource settings.
>
---
#### [new 088] Analysis of Voluntarily Reported Data Post Mesh Implantation for Detecting Public Emotion and Identifying Concern Reports
- **分类: cs.CL; cs.LG**

- **简介: 该论文通过分析术后患者报告数据，利用NLP技术识别情绪及关注报告，旨在监测医疗植入物并发症并理解患者体验变化，为医疗决策提供依据。**

- **链接: [http://arxiv.org/pdf/2509.04517v1](http://arxiv.org/pdf/2509.04517v1)**

> **作者:** Indu Bala; Lewis Mitchell; Marianne H Gillam
>
> **摘要:** Mesh implants are widely utilized in hernia repair surgeries, but postoperative complications present a significant concern. This study analyzes patient reports from the Manufacturer and User Facility Device Experience (MAUDE) database spanning 2000 to 2021 to investigate the emotional aspects of patients following mesh implantation using Natural Language Processing (NLP). Employing the National Research Council Canada (NRC) Emotion Lexicon and TextBlob for sentiment analysis, the research categorizes patient narratives into eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and assesses sentiment polarity. The goal is to discern patterns in patient sentiment over time and to identify reports signaling urgent concerns, referred to as "Concern Reports," thereby understanding shifts in patient experiences in relation to changes in medical device regulation and technological advancements in healthcare. The study detected an increase in Concern Reports and higher emotional intensity during the periods of 2011-2012 and 2017-2018. Through temporal analysis of Concern Reports and overall sentiment, this research provides valuable insights for healthcare practitioners, enhancing their understanding of patient experiences post-surgery, which is critical for improving preoperative counselling, postoperative care, and preparing patients for mesh implant surgeries. The study underscores the importance of emotional considerations in medical practices and the potential for sentiment analysis to inform and enhance patient care.
>
---
#### [new 089] Mentalic Net: Development of RAG-based Conversational AI and Evaluation Framework for Mental Health Support
- **分类: cs.CL**

- **简介: 该论文开发基于RAG的对话AI系统Mentalic Net，用于心理健康支持，解决LLM在安全性和有效性方面的挑战。通过评估框架测试多维度指标，结合提示工程与微调，提升系统表现，并倡导负责任的开发策略。**

- **链接: [http://arxiv.org/pdf/2509.04456v1](http://arxiv.org/pdf/2509.04456v1)**

> **作者:** Anandi Dutta; Shivani Mruthyunjaya; Jessica Saddington; Kazi Sifatul Islam
>
> **备注:** Preprint Version, Accepted in ISEMV 2025
>
> **摘要:** The emergence of large language models (LLMs) has unlocked boundless possibilities, along with significant challenges. In response, we developed a mental health support chatbot designed to augment professional healthcare, with a strong emphasis on safe and meaningful application. Our approach involved rigorous evaluation, covering accuracy, empathy, trustworthiness, privacy, and bias. We employed a retrieval-augmented generation (RAG) framework, integrated prompt engineering, and fine-tuned a pre-trained model on novel datasets. The resulting system, Mentalic Net Conversational AI, achieved a BERT Score of 0.898, with other evaluation metrics falling within satisfactory ranges. We advocate for a human-in-the-loop approach and a long-term, responsible strategy in developing such transformative technologies, recognizing both their potential to change lives and the risks they may pose if not carefully managed.
>
---
#### [new 090] Discrete Prompt Tuning via Recursive Utilization of Black-box Multimodal Large Language Model for Personalized Visual Emotion Recognition
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出基于离散提示调优的个性化视觉情感识别方法，通过递归利用黑箱多模态大语言模型选择最优自然语言提示，解决现有模型因偏好多数观点导致的个性化识别不足问题。**

- **链接: [http://arxiv.org/pdf/2509.04480v1](http://arxiv.org/pdf/2509.04480v1)**

> **作者:** Ryo Takahashi; Naoki Saito; Keisuke Maeda; Takahiro Ogawa; Miki Haseyama
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Visual Emotion Recognition (VER) is an important research topic due to its wide range of applications, including opinion mining and advertisement design. Extending this capability to recognize emotions at the individual level further broadens its potential applications. Recently, Multimodal Large Language Models (MLLMs) have attracted increasing attention and demonstrated performance comparable to that of conventional VER methods. However, MLLMs are trained on large and diverse datasets containing general opinions, which causes them to favor majority viewpoints and familiar patterns. This tendency limits their performance in a personalized VER, which is crucial for practical and real-world applications, and indicates a key area for improvement. To address this limitation, the proposed method employs discrete prompt tuning inspired by the process of humans' prompt engineering to adapt the VER task to each individual. Our method selects the best natural language representation from the generated prompts and uses it to update the prompt for the realization of accurate personalized VER.
>
---
#### [new 091] ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ParaThinker框架，通过并行生成多路径推理解决LLM测试时计算扩展中的"隧道视野"瓶颈，提升推理准确率并降低延迟，为模型扩展提供新范式。**

- **链接: [http://arxiv.org/pdf/2509.04475v1](http://arxiv.org/pdf/2509.04475v1)**

> **作者:** Hao Wen; Yifan Su; Feifei Zhang; Yunxin Liu; Yunhao Liu; Ya-Qin Zhang; Yuanchun Li
>
> **摘要:** Recent advances in Large Language Models (LLMs) have been driven by test-time compute scaling - a strategy that improves reasoning by generating longer, sequential thought processes. While effective, this approach encounters a significant bottleneck as computation increases, where further computation offers only marginal performance gains. We argue this ceiling is not an inherent limit of the model's capability but a flaw in the scaling strategy itself, a phenomenon we term "Tunnel Vision", where a model's imperfect initial steps lock it into a suboptimal reasoning path. To overcome this, we introduce a new scaling paradigm: native thought parallelism. We present ParaThinker, an end-to-end framework that trains an LLM to generate multiple, diverse reasoning paths in parallel and synthesize them into a superior final answer. By exploring different lines of thoughts simultaneously, ParaThinker effectively sidesteps the Tunnel Vision issue and unlocks the model's latent reasoning potential. Our approach demonstrates that scaling compute in parallel (width) is a more effective and efficient way to superior reasoning than simply scaling sequentially (depth). On challenging reasoning benchmarks, ParaThinker achieves substantial accuracy improvements over sequential LLMs (12.3% for 1.5B and 7.5% for 7B models on average with 8 parallel paths), while adding only negligible latency overhead (7.1%). This enables smaller models to surpass much larger counterparts and establishes parallel thinking as a critical, efficient dimension for scaling future LLMs.
>
---
#### [new 092] Classification of kinetic-related injury in hospital triage data using NLP
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出基于有限计算资源的LLM微调方法，解决医院分诊数据中运动相关伤害分类难题，克服隐私限制、硬件不足和标注成本问题，通过结合开源数据与医院数据实现有效分类。**

- **链接: [http://arxiv.org/pdf/2509.04969v1](http://arxiv.org/pdf/2509.04969v1)**

> **作者:** Midhun Shyam; Jim Basilakis; Kieran Luken; Steven Thomas; John Crozier; Paul M. Middleton; X. Rosalind Wang
>
> **备注:** Accepted as a short paper for publishing at ADMA 2025 (https://adma2025.github.io), with Supplementary Material available at https://github.com/CRMDS/Kinetic-Injury-Triage
>
> **摘要:** Triage notes, created at the start of a patient's hospital visit, contain a wealth of information that can help medical staff and researchers understand Emergency Department patient epidemiology and the degree of time-dependent illness or injury. Unfortunately, applying modern Natural Language Processing and Machine Learning techniques to analyse triage data faces some challenges: Firstly, hospital data contains highly sensitive information that is subject to privacy regulation thus need to be analysed on site; Secondly, most hospitals and medical facilities lack the necessary hardware to fine-tune a Large Language Model (LLM), much less training one from scratch; Lastly, to identify the records of interest, expert inputs are needed to manually label the datasets, which can be time-consuming and costly. We present in this paper a pipeline that enables the classification of triage data using LLM and limited compute resources. We first fine-tuned a pre-trained LLM with a classifier using a small (2k) open sourced dataset on a GPU; and then further fine-tuned the model with a hospital specific dataset of 1000 samples on a CPU. We demonstrated that by carefully curating the datasets and leveraging existing models and open sourced data, we can successfully classify triage data with limited compute resources.
>
---
#### [new 093] ToM-SSI: Evaluating Theory of Mind in Situated Social Interactions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ToM-SSI基准，解决现有ToM测试局限于文本和双人交互的不足，设计多模态群体互动环境，评估模型在复杂社交场景中的心智理论能力，揭示当前模型性能局限。**

- **链接: [http://arxiv.org/pdf/2509.05066v1](http://arxiv.org/pdf/2509.05066v1)**

> **作者:** Matteo Bortoletto; Constantin Ruhdorfer; Andreas Bulling
>
> **备注:** EMNLP 2025 (Main)
>
> **摘要:** Most existing Theory of Mind (ToM) benchmarks for foundation models rely on variations of the Sally-Anne test, offering only a very limited perspective on ToM and neglecting the complexity of human social interactions. To address this gap, we propose ToM-SSI: a new benchmark specifically designed to test ToM capabilities in environments rich with social interactions and spatial dynamics. While current ToM benchmarks are limited to text-only or dyadic interactions, ToM-SSI is multimodal and includes group interactions of up to four agents that communicate and move in situated environments. This unique design allows us to study, for the first time, mixed cooperative-obstructive settings and reasoning about multiple agents' mental state in parallel, thus capturing a wider range of social cognition than existing benchmarks. Our evaluations reveal that the current models' performance is still severely limited, especially in these new tasks, highlighting critical gaps for future research.
>
---
#### [new 094] Labelling Data with Unknown References
- **分类: cs.DS; cs.AI; cs.CL**

- **简介: 该论文提出No-Data算法，解决无参考标签时评估者可信度验证问题，通过连续挑战评估者建立信任，并提供理论与实验验证。**

- **链接: [http://arxiv.org/pdf/2506.03083v3](http://arxiv.org/pdf/2506.03083v3)**

> **作者:** Adrian de Wynter
>
> **备注:** Extended version with LLM-based results/analysis
>
> **摘要:** An evaluator is trustworthy when there exists some agreed-upon way to measure its performance as a labeller. The two ways to establish trustworthiness are either by testing it, or by assuming the evaluator `knows' somehow the way to label the corpus. However, if labelled references (e.g., a development set) are unavailable, neither of these approaches work: the former requires the data, and the latter is an assumption, not evidence. To address this, we introduce an algorithm (the `No-Data Algorithm') by which to establish trust in an evaluator without any existing references. Our algorithm works by successively posing challenges to said evaluator. We show that this is sufficient to establish trustworthiness w.h.p., in such a way that when the evaluator actually knows the way to label the corpus, the No-Data Algorithm accepts its output; and, conversely, flags untrustworthy evaluators when these are unable to prove it. We present formal proofs of correctness, empirical tests, and applications to LLMs-as-judges on low-resource languages.
>
---
#### [new 095] Sticker-TTS: Learn to Utilize Historical Experience with a Sticker-driven Test-Time Scaling Framework
- **分类: cs.AI; cs.CL; I.2.7**

- **简介: 该论文提出Sticker-TTS框架，通过协作模型和历史经验利用（stickers）提升测试扩展效率，结合两阶段优化策略，在数学推理任务中超越基线方法。**

- **链接: [http://arxiv.org/pdf/2509.05007v1](http://arxiv.org/pdf/2509.05007v1)**

> **作者:** Jie Chen; Jinhao Jiang; Yingqian Min; Zican Dong; Shijie Wang; Wayne Xin Zhao; Ji-Rong Wen
>
> **备注:** 11 pages, 1 figures, 5 tables
>
> **摘要:** Large reasoning models (LRMs) have exhibited strong performance on complex reasoning tasks, with further gains achievable through increased computational budgets at inference. However, current test-time scaling methods predominantly rely on redundant sampling, ignoring the historical experience utilization, thereby limiting computational efficiency. To overcome this limitation, we propose Sticker-TTS, a novel test-time scaling framework that coordinates three collaborative LRMs to iteratively explore and refine solutions guided by historical attempts. At the core of our framework are distilled key conditions-termed stickers-which drive the extraction, refinement, and reuse of critical information across multiple rounds of reasoning. To further enhance the efficiency and performance of our framework, we introduce a two-stage optimization strategy that combines imitation learning with self-improvement, enabling progressive refinement. Extensive evaluations on three challenging mathematical reasoning benchmarks, including AIME-24, AIME-25, and OlymMATH, demonstrate that Sticker-TTS consistently surpasses strong baselines, including self-consistency and advanced reinforcement learning approaches, under comparable inference budgets. These results highlight the effectiveness of sticker-guided historical experience utilization. Our code and data are available at https://github.com/RUCAIBox/Sticker-TTS.
>
---
#### [new 096] WildScore: Benchmarking MLLMs in-the-Wild Symbolic Music Reasoning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出WildScore基准，评估多模态大语言模型（MLLMs）在符号音乐推理任务中的能力，解决其在音乐领域推理能力不足的问题，通过真实音乐数据与多选题框架进行系统评测。**

- **链接: [http://arxiv.org/pdf/2509.04744v1](http://arxiv.org/pdf/2509.04744v1)**

> **作者:** Gagan Mundada; Yash Vishe; Amit Namburi; Xin Xu; Zachary Novack; Julian McAuley; Junda Wu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, their reasoning abilities in the multimodal symbolic music domain remain largely unexplored. We introduce WildScore, the first in-the-wild multimodal symbolic music reasoning and analysis benchmark, designed to evaluate MLLMs' capacity to interpret real-world music scores and answer complex musicological queries. Each instance in WildScore is sourced from genuine musical compositions and accompanied by authentic user-generated questions and discussions, capturing the intricacies of practical music analysis. To facilitate systematic evaluation, we propose a systematic taxonomy, comprising both high-level and fine-grained musicological ontologies. Furthermore, we frame complex music reasoning as multiple-choice question answering, enabling controlled and scalable assessment of MLLMs' symbolic music understanding. Empirical benchmarking of state-of-the-art MLLMs on WildScore reveals intriguing patterns in their visual-symbolic reasoning, uncovering both promising directions and persistent challenges for MLLMs in symbolic music reasoning and analysis. We release the dataset and code.
>
---
#### [new 097] Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出基于本体论的方法，解决大语言模型对话可控性问题，通过形式化定性概念为定量定义，整合至本体以指导可控文本生成，提升对话AI的透明度与一致性。**

- **链接: [http://arxiv.org/pdf/2509.04926v1](http://arxiv.org/pdf/2509.04926v1)**

> **作者:** Barbara Gendron; Gaël Guibon; Mathieu D'aquin
>
> **备注:** Accepted at TOTh 2025 (Terminology \& Ontology: Theories and applications)
>
> **摘要:** The controllability of Large Language Models (LLMs) when used as conversational agents is a key challenge, particularly to ensure predictable and user-personalized responses. This work proposes an ontology-based approach to formally define conversational features that are typically qualitative in nature. By leveraging a set of linguistic descriptors, we derive quantitative definitions for qualitatively-defined concepts, enabling their integration into an ontology for reasoning and consistency checking. We apply this framework to the task of proficiency-level control in conversations, using CEFR language proficiency levels as a case study. These definitions are then formalized in description logic and incorporated into an ontology, which guides controlled text generation of an LLM through fine-tuning. Experimental results demonstrate that our approach provides consistent and explainable proficiency-level definitions, improving transparency in conversational AI.
>
---
#### [new 098] Evaluating Cognitive-Behavioral Fixation via Multimodal User Viewing Patterns on Social Media
- **分类: cs.SI; cs.CL**

- **简介: 该论文提出框架，通过分析多模态社交媒体数据检测认知行为固化，解决计算评估难题，包含主题提取与量化模块，实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2509.04823v1](http://arxiv.org/pdf/2509.04823v1)**

> **作者:** Yujie Wang; Yunwei Zhao; Jing Yang; Han Han; Shiguang Shan; Jie Zhang
>
> **摘要:** Digital social media platforms frequently contribute to cognitive-behavioral fixation, a phenomenon in which users exhibit sustained and repetitive engagement with narrow content domains. While cognitive-behavioral fixation has been extensively studied in psychology, methods for computationally detecting and evaluating such fixation remain underexplored. To address this gap, we propose a novel framework for assessing cognitive-behavioral fixation by analyzing users' multimodal social media engagement patterns. Specifically, we introduce a multimodal topic extraction module and a cognitive-behavioral fixation quantification module that collaboratively enable adaptive, hierarchical, and interpretable assessment of user behavior. Experiments on existing benchmarks and a newly curated multimodal dataset demonstrate the effectiveness of our approach, laying the groundwork for scalable computational analysis of cognitive fixation. All code in this project is publicly available for research purposes at https://github.com/Liskie/cognitive-fixation-evaluation.
>
---
#### [new 099] Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO; 68T05, 90C40, 91A26, 68T42, 93E35; I.2.11; I.2.6; I.2.8; I.2.9; I.2.7**

- **简介: 该论文提出通过语言驱动的层次化任务结构构建显式环境模型，解决多智能体学习中因复杂任务和稀疏奖励导致的探索效率低问题，利用大语言模型动态生成任务分层框架，提升智能体策略学习效率。**

- **链接: [http://arxiv.org/pdf/2509.04731v1](http://arxiv.org/pdf/2509.04731v1)**

> **作者:** Brennen Hill
>
> **摘要:** The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents.
>
---
#### [new 100] Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments
- **分类: cs.GR; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出LLM驱动的叙事到2D场景生成框架，解决叙事文本与可玩视觉环境连接的挑战。通过时间框架提取、空间谓词解析、语义嵌入检索及细胞自动机构建地形，实现故事驱动的场景生成与空间约束满足。**

- **链接: [http://arxiv.org/pdf/2509.04481v1](http://arxiv.org/pdf/2509.04481v1)**

> **作者:** Yi-Chun Chen; Arnav Jhala
>
> **摘要:** Recent advances in large language models(LLMs) enable compelling story generation, but connecting narrative text to playable visual environments remains an open challenge in procedural content generation(PCG). We present a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories. Given an LLM-generated narrative, our system identifies three key time frames, extracts spatial predicates in the form of "Object-Relation-Object" triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset. A layered terrain is generated using Cellular Automata, and objects are placed using spatial rules grounded in the predicate structure. We evaluated our system in ten diverse stories, analyzing tile-object matching, affordance-layer alignment, and spatial constraint satisfaction across frames. This prototype offers a scalable approach to narrative-driven scene generation and lays the foundation for future work on multi-frame continuity, symbolic tracking, and multi-agent coordination in story-centered PCG.
>
---
#### [new 101] SpikingBrain Technical Report: Spiking Brain-inspired Large Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SpikingBrain模型，解决Transformer大模型在长上下文处理和非NVIDIA平台训练的效率瓶颈。通过脑启发架构、算法优化和系统工程，开发出高效模型，提升训练速度和推理效率，实现低功耗运行。**

- **链接: [http://arxiv.org/pdf/2509.05276v1](http://arxiv.org/pdf/2509.05276v1)**

> **作者:** Yuqi Pan; Yupeng Feng; Jinghao Zhuang; Siyu Ding; Zehao Liu; Bohan Sun; Yuhong Chou; Han Xu; Xuerui Qiu; Anlin Deng; Anjie Hu; Peng Zhou; Man Yao; Jibin Wu; Jian Yang; Guoliang Sun; Bo Xu; Guoqi Li
>
> **摘要:** Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware. Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design.
>
---
#### [new 102] SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文提出SparkUI-Parser，解决GUI感知中定位不准确、无法全面解析界面的问题。通过连续坐标建模与拒绝机制提升精度与速度，并构建ScreenParse基准测试，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2509.04908v1](http://arxiv.org/pdf/2509.04908v1)**

> **作者:** Hongyi Jing; Jiafu Chen; Chen Rao; Ziqiang Dang; Jiajie Teng; Tianyi Chu; Juncheng Mo; Shuo Fang; Huaizhong Lin; Rui Lv; Chenguang Ma; Lei Zhao
>
> **摘要:** The existing Multimodal Large Language Models (MLLMs) for GUI perception have made great progress. However, the following challenges still exist in prior methods: 1) They model discrete coordinates based on text autoregressive mechanism, which results in lower grounding accuracy and slower inference speed. 2) They can only locate predefined sets of elements and are not capable of parsing the entire interface, which hampers the broad application and support for downstream tasks. To address the above issues, we propose SparkUI-Parser, a novel end-to-end framework where higher localization precision and fine-grained parsing capability of the entire interface are simultaneously achieved. Specifically, instead of using probability-based discrete modeling, we perform continuous modeling of coordinates based on a pre-trained Multimodal Large Language Model (MLLM) with an additional token router and coordinate decoder. This effectively mitigates the limitations inherent in the discrete output characteristics and the token-by-token generation process of MLLMs, consequently boosting both the accuracy and the inference speed. To further enhance robustness, a rejection mechanism based on a modified Hungarian matching algorithm is introduced, which empowers the model to identify and reject non-existent elements, thereby reducing false positives. Moreover, we present ScreenParse, a rigorously constructed benchmark to systematically assess structural perception capabilities of GUI models across diverse scenarios. Extensive experiments demonstrate that our approach consistently outperforms SOTA methods on ScreenSpot, ScreenSpot-v2, CAGUI-Grounding and ScreenParse benchmarks. The resources are available at https://github.com/antgroup/SparkUI-Parser.
>
---
#### [new 103] DarkStream: real-time speech anonymization with low latency
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文提出DarkStream模型，解决实时语音通信中的隐私保护问题。通过低延迟编码器、直接波形生成及GAN注入伪嵌入，实现语音匿名化，在保持语言可懂度的同时有效掩盖说话人身份。**

- **链接: [http://arxiv.org/pdf/2509.04667v1](http://arxiv.org/pdf/2509.04667v1)**

> **作者:** Waris Quamer; Ricardo Gutierrez-Osuna
>
> **备注:** Accepted for presentation at ASRU 2025
>
> **摘要:** We propose DarkStream, a streaming speech synthesis model for real-time speaker anonymization. To improve content encoding under strict latency constraints, DarkStream combines a causal waveform encoder, a short lookahead buffer, and transformer-based contextual layers. To further reduce inference time, the model generates waveforms directly via a neural vocoder, thus removing intermediate mel-spectrogram conversions. Finally, DarkStream anonymizes speaker identity by injecting a GAN-generated pseudo-speaker embedding into linguistic features from the content encoder. Evaluations show our model achieves strong anonymization, yielding close to 50% speaker verification EER (near-chance performance) on the lazy-informed attack scenario, while maintaining acceptable linguistic intelligibility (WER within 9%). By balancing low-latency, robust privacy, and minimal intelligibility degradation, DarkStream provides a practical solution for privacy-preserving real-time speech communication.
>
---
#### [new 104] Non-Termination Proving: 100 Million LoC and Beyond
- **分类: cs.PL; cs.CL; cs.SE; D.3; F.3**

- **简介: 论文提出Pulse Infinite工具，通过组合、下近似方法证明大规模程序的非终止性，突破传统小规模基准限制，应用于超1亿行代码，发现30+新问题，建立新标准。**

- **链接: [http://arxiv.org/pdf/2509.05293v1](http://arxiv.org/pdf/2509.05293v1)**

> **作者:** Julien Vanegue; Jules Villard; Peter O'Hearn; Azalea Raad
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** We report on our tool, Pulse Infinite, that uses proof techniques to show non-termination (divergence) in large programs. Pulse Infinite works compositionally and under-approximately: the former supports scale, and the latter ensures soundness for proving divergence. Prior work focused on small benchmarks in the tens or hundreds of lines of code (LoC), and scale limits their practicality: a single company may have tens of millions, or even hundreds of millions of LoC or more. We report on applying Pulse Infinite to over a hundred million lines of open-source and proprietary software written in C, C++, and Hack, identifying over 30 previously unknown issues, establishing a new state of the art for detecting divergence in real-world codebases.
>
---
#### [new 105] Maestro: Joint Graph & Config Optimization for Reliable AI Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出Maestro框架，联合优化AI代理的图结构与节点配置，解决现有方法忽视结构故障的问题，提升可靠性，超越现有优化方法。**

- **链接: [http://arxiv.org/pdf/2509.04642v1](http://arxiv.org/pdf/2509.04642v1)**

> **作者:** Wenxiao Wang; Priyatham Kattakinda; Soheil Feizi
>
> **备注:** Technical Report by RELAI.ai
>
> **摘要:** Building reliable LLM agents requires decisions at two levels: the graph (which modules exist and how information flows) and the configuration of each node (models, prompts, tools, control knobs). Most existing optimizers tune configurations while holding the graph fixed, leaving structural failure modes unaddressed. We introduce Maestro, a framework-agnostic holistic optimizer for LLM agents that jointly searches over graphs and configurations to maximize agent quality, subject to explicit rollout/token budgets. Beyond numeric metrics, Maestro leverages reflective textual feedback from traces to prioritize edits, improving sample efficiency and targeting specific failure modes. On the IFBench and HotpotQA benchmarks, Maestro consistently surpasses leading prompt optimizers--MIPROv2, GEPA, and GEPA+Merge--by an average of 12%, 4.9%, and 4.86%, respectively; even when restricted to prompt-only optimization, it still leads by 9.65%, 2.37%, and 2.41%. Maestro achieves these results with far fewer rollouts than GEPA. We further show large gains on two applications (interviewer & RAG agents), highlighting that joint graph & configuration search addresses structural failure modes that prompt tuning alone cannot fix.
>
---
#### [new 106] Finding your MUSE: Mining Unexpected Solutions Engine
- **分类: cs.AI; cs.CL**

- **简介: 论文提出构建功能概念图（FCG）的方法，开发MUSE算法，用于突破创新者对现有方案的固化思维，生成创意解决方案，并基于50万专利验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.05072v1](http://arxiv.org/pdf/2509.05072v1)**

> **作者:** Nir Sweed; Hanit Hakim; Ben Wolfson; Hila Lifshitz; Dafna Shahaf
>
> **摘要:** Innovators often exhibit cognitive fixation on existing solutions or nascent ideas, hindering the exploration of novel alternatives. This paper introduces a methodology for constructing Functional Concept Graphs (FCGs), interconnected representations of functional elements that support abstraction, problem reframing, and analogical inspiration. Our approach yields large-scale, high-quality FCGs with explicit abstraction relations, overcoming limitations of prior work. We further present MUSE, an algorithm leveraging FCGs to generate creative inspirations for a given problem. We demonstrate our method by computing an FCG on 500K patents, which we release for further research.
>
---
#### [new 107] Code Review Without Borders: Evaluating Synthetic vs. Real Data for Review Recommendation
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文研究如何利用合成数据提升代码审查推荐系统。针对新兴语言缺乏标注数据的问题，通过LLM生成合成代码变更，训练分类器并与真实数据对比，验证合成数据在低资源场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2509.04810v1](http://arxiv.org/pdf/2509.04810v1)**

> **作者:** Yogev Cohen; Dudi Ohayon; Romy Somkin; Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** Automating the decision of whether a code change requires manual review is vital for maintaining software quality in modern development workflows. However, the emergence of new programming languages and frameworks creates a critical bottleneck: while large volumes of unlabelled code are readily available, there is an insufficient amount of labelled data to train supervised models for review classification. We address this challenge by leveraging Large Language Models (LLMs) to translate code changes from well-resourced languages into equivalent changes in underrepresented or emerging languages, generating synthetic training data where labelled examples are scarce. We assume that although LLMs have learned the syntax and semantics of new languages from available unlabelled code, they have yet to fully grasp which code changes are considered significant or review-worthy within these emerging ecosystems. To overcome this, we use LLMs to generate synthetic change examples and train supervised classifiers on them. We systematically compare the performance of these classifiers against models trained on real labelled data. Our experiments across multiple GitHub repositories and language pairs demonstrate that LLM-generated synthetic data can effectively bootstrap review recommendation systems, narrowing the performance gap even in low-resource settings. This approach provides a scalable pathway to extend automated code review capabilities to rapidly evolving technology stacks, even in the absence of annotated data.
>
---
## 更新

#### [replaced 001] Simple Yet Effective: An Information-Theoretic Approach to Multi-LLM Uncertainty Quantification
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07236v2](http://arxiv.org/pdf/2507.07236v2)**

> **作者:** Maya Kruse; Majid Afshar; Saksham Khatwani; Anoop Mayampurath; Guanhua Chen; Yanjun Gao
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and na\"ive ensemble baselines. In addition, we explore using MUSE as guided signals with chain-of-thought distillation to fine-tune LLMs for calibration. MUSE is available at:https://github.com/LARK-NLP-Lab/MUSE.
>
---
#### [replaced 002] Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19081v2](http://arxiv.org/pdf/2507.19081v2)**

> **作者:** Hao Li; Yizheng Sun; Viktor Schlegel; Kailai Yang; Riza Batista-Navarro; Goran Nenadic
>
> **备注:** Preprint
>
> **摘要:** Argument summarization aims to generate concise, structured representations of complex, multi-perspective debates. While recent work has advanced the identification and clustering of argumentative components, the generation stage remains underexplored. Existing approaches typically rely on single-pass generation, offering limited support for factual correction or structural refinement. To address this gap, we introduce Arg-LLaDA, a novel large language diffusion framework that iteratively improves summaries via sufficiency-guided remasking and regeneration. Our method combines a flexible masking controller with a sufficiency-checking module to identify and revise unsupported, redundant, or incomplete spans, yielding more faithful, concise, and coherent outputs. Empirical results on two benchmark datasets demonstrate that Arg-LLaDA surpasses state-of-the-art baselines in 7 out of 10 automatic evaluation metrics. In addition, human evaluations reveal substantial improvements across core dimensions, coverage, faithfulness, and conciseness, validating the effectiveness of our iterative, sufficiency-aware generation strategy.
>
---
#### [replaced 003] All That Glitters is Not Novel: Plagiarism in AI Generated Research
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16487v3](http://arxiv.org/pdf/2502.16487v3)**

> **作者:** Tarun Gupta; Danish Pruthi
>
> **备注:** Accepted to ACL 2025 (main) conference
>
> **摘要:** Automating scientific research is considered the final frontier of science. Recently, several papers claim autonomous research agents can generate novel research ideas. Amidst the prevailing optimism, we document a critical concern: a considerable fraction of such research documents are smartly plagiarized. Unlike past efforts where experts evaluate the novelty and feasibility of research ideas, we request $13$ experts to operate under a different situational logic: to identify similarities between LLM-generated research documents and existing work. Concerningly, the experts identify $24\%$ of the $50$ evaluated research documents to be either paraphrased (with one-to-one methodological mapping), or significantly borrowed from existing work. These reported instances are cross-verified by authors of the source papers. The remaining $76\%$ of documents show varying degrees of similarity with existing work, with only a small fraction appearing completely novel. Problematically, these LLM-generated research documents do not acknowledge original sources, and bypass inbuilt plagiarism detectors. Lastly, through controlled experiments we show that automated plagiarism detectors are inadequate at catching plagiarized ideas from such systems. We recommend a careful assessment of LLM-generated research, and discuss the implications of our findings on academic publishing.
>
---
#### [replaced 004] First Steps Towards Overhearing LLM Agents: A Case Study With Dungeons & Dragons Gameplay
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.22809v2](http://arxiv.org/pdf/2505.22809v2)**

> **作者:** Andrew Zhu; Evan Osgood; Chris Callison-Burch
>
> **备注:** 9 pages, 5 figures. COLM 2025 Workshop on AI Agents
>
> **摘要:** Much work has been done on conversational LLM agents which directly assist human users with tasks. We present an alternative paradigm for interacting with LLM agents, which we call "overhearing agents". These overhearing agents do not actively participate in conversation -- instead, they "listen in" on human-to-human conversations and perform background tasks or provide suggestions to assist the user. In this work, we explore the overhearing agents paradigm through the lens of Dungeons & Dragons gameplay. We present an in-depth study using large multimodal audio-language models as overhearing agents to assist a Dungeon Master. We perform a human evaluation to examine the helpfulness of such agents and find that some large audio-language models have the emergent ability to perform overhearing agent tasks using implicit audio cues. Finally, we release Python libraries and our project code to support further research into the overhearing agents paradigm at https://github.com/zhudotexe/overhearing_agents.
>
---
#### [replaced 005] Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalization of Misinformation Detection Models
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18122v3](http://arxiv.org/pdf/2410.18122v3)**

> **作者:** Ivo Verhoeven; Pushkar Mishra; Ekaterina Shutova
>
> **备注:** Under review
>
> **摘要:** This article introduces misinfo-general, a benchmark dataset for evaluating misinformation models' ability to perform out-of-distribution generalization. Misinformation changes rapidly, much more quickly than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation detectors need to be able to perform out-of-distribution generalization, an attribute they currently lack. Our benchmark uses distant labelling to enable simulating covariate shifts in misinformation content. We identify time, event, topic, publisher, political bias, misinformation type as important axes for generalization, and we evaluate a common class of baseline models on each. Using article metadata, we show how this model fails desiderata, which is not necessarily obvious from classification metrics. Finally, we analyze properties of the data to ensure limited presence of modelling shortcuts. We make the dataset and accompanying code publicly available: https://github.com/ioverho/misinfo-general
>
---
#### [replaced 006] Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.17450v2](http://arxiv.org/pdf/2508.17450v2)**

> **作者:** Bryan Chen Zhengyu Tan; Daniel Wai Kit Chin; Zhengyuan Liu; Nancy F. Chen; Roy Ka-Wei Lee
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
>
---
#### [replaced 007] RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language
- **分类: cs.CL; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.17114v3](http://arxiv.org/pdf/2505.17114v3)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Multimodal question answering (QA) often requires identifying which video, audio, or sensor tokens are relevant to the question. Yet modality disagreements are common: off-camera speech, background noise, or motion outside the field of view often mislead fusion models that weight all streams equally. We present RAVEN, a unified QA architecture whose core is QuART, a query-conditioned cross-modal gating module that assigns scalar relevance scores to each token across modalities, enabling the model to amplify informative signals and suppress distractors before fusion. RAVEN is trained through a three-stage pipeline comprising unimodal pretraining, query-aligned fusion, and disagreement-oriented fine-tuning -- each stage targeting a distinct challenge in multi-modal reasoning: representation quality, cross-modal relevance, and robustness to modality mismatch. To support training and evaluation, we release AVS-QA, a dataset of 300K synchronized Audio--Video-Sensor streams paired with automatically generated question-answer pairs. Experimental results on seven multi-modal QA benchmarks -- including egocentric and exocentric tasks -- show that RAVEN achieves up to 14.5\% and 8.0\% gains in accuracy compared to state-of-the-art multi-modal large language models, respectively. Incorporating sensor data provides an additional 16.4\% boost, and the model remains robust under modality corruption, outperforming SOTA baselines by 50.23\%. Our code and dataset are available at https://github.com/BASHLab/RAVEN.
>
---
#### [replaced 008] Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.02544v3](http://arxiv.org/pdf/2408.02544v3)**

> **作者:** Xinbei Ma; Yiting Wang; Yao Yao; Tongxin Yuan; Aston Zhang; Zhuosheng Zhang; Hai Zhao
>
> **备注:** ACL 2025
>
> **摘要:** This paper investigates the faithfulness of multimodal large language model (MLLM) agents in a graphical user interface (GUI) environment, aiming to address the research question of whether multimodal GUI agents can be distracted by environmental context. A general scenario is proposed where both the user and the agent are benign, and the environment, while not malicious, contains unrelated content. A wide range of MLLMs are evaluated as GUI agents using a simulated dataset, following three working patterns with different levels of perception. Experimental results reveal that even the most powerful models, whether generalist agents or specialist GUI agents, are susceptible to distractions. While recent studies predominantly focus on the helpfulness of agents, our findings first indicate that these agents are prone to environmental distractions. Furthermore, we implement an adversarial environment injection and analyze the approach to improve faithfulness, calling for a collective focus on this important topic.
>
---
#### [replaced 009] UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.02544v2](http://arxiv.org/pdf/2509.02544v2)**

> **作者:** Haoming Wang; Haoyang Zou; Huatong Song; Jiazhan Feng; Junjie Fang; Junting Lu; Longxiang Liu; Qinyu Luo; Shihao Liang; Shijue Huang; Wanjun Zhong; Yining Ye; Yujia Qin; Yuwen Xiong; Yuxin Song; Zhiyong Wu; Aoyan Li; Bo Li; Chen Dun; Chong Liu; Daoguang Zan; Fuxing Leng; Hanbin Wang; Hao Yu; Haobin Chen; Hongyi Guo; Jing Su; Jingjia Huang; Kai Shen; Kaiyu Shi; Lin Yan; Peiyao Zhao; Pengfei Liu; Qinghao Ye; Renjie Zheng; Shulin Xin; Wayne Xin Zhao; Wen Heng; Wenhao Huang; Wenqian Wang; Xiaobo Qin; Yi Lin; Youbin Wu; Zehui Chen; Zihao Wang; Baoquan Zhong; Xinchun Zhang; Xujing Li; Yuanfan Li; Zhongkai Zhao; Chengquan Jiang; Faming Wu; Haotian Zhou; Jinlin Pang; Li Han; Qi Liu; Qianli Ma; Siyao Liu; Songhua Cai; Wenqi Fu; Xin Liu; Yaohui Wang; Zhi Zhang; Bo Zhou; Guoliang Li; Jiajun Shi; Jiale Yang; Jie Tang; Li Li; Qihua Han; Taoran Lu; Woyu Lin; Xiaokang Tong; Xinyao Li; Yichi Zhang; Yu Miao; Zhengxuan Jiang; Zili Li; Ziyuan Zhao; Chenxin Li; Dehua Ma; Feng Lin; Ge Zhang; Haihua Yang; Hangyu Guo; Hongda Zhu; Jiaheng Liu; Junda Du; Kai Cai; Kuanye Li; Lichen Yuan; Meilan Han; Minchao Wang; Shuyue Guo; Tianhao Cheng; Xiaobo Ma; Xiaojun Xiao; Xiaolong Huang; Xinjie Chen; Yidi Du; Yilin Chen; Yiwen Wang; Zhaojian Li; Zhenzhu Yang; Zhiyuan Zeng; Chaolin Jin; Chen Li; Hao Chen; Haoli Chen; Jian Chen; Qinghao Zhao; Guang Shi
>
> **摘要:** The development of autonomous agents for graphical user interfaces (GUIs) presents major challenges in artificial intelligence. While recent advances in native agent models have shown promise by unifying perception, reasoning, action, and memory through end-to-end learning, open problems remain in data scalability, multi-turn reinforcement learning (RL), the limitations of GUI-only operation, and environment stability. In this technical report, we present UI-TARS-2, a native GUI-centered agent model that addresses these challenges through a systematic training methodology: a data flywheel for scalable data generation, a stabilized multi-turn RL framework, a hybrid GUI environment that integrates file systems and terminals, and a unified sandbox platform for large-scale rollouts. Empirical evaluation demonstrates that UI-TARS-2 achieves significant improvements over its predecessor UI-TARS-1.5. On GUI benchmarks, it reaches 88.2 on Online-Mind2Web, 47.5 on OSWorld, 50.6 on WindowsAgentArena, and 73.3 on AndroidWorld, outperforming strong baselines such as Claude and OpenAI agents. In game environments, it attains a mean normalized score of 59.8 across a 15-game suite-roughly 60% of human-level performance-and remains competitive with frontier proprietary models (e.g., OpenAI o3) on LMGame-Bench. Additionally, the model can generalize to long-horizon information-seeking tasks and software engineering benchmarks, highlighting its robustness across diverse agent tasks. Detailed analyses of training dynamics further provide insights into achieving stability and efficiency in large-scale agent RL. These results underscore UI-TARS-2's potential to advance the state of GUI agents and exhibit strong generalization to real-world interactive scenarios.
>
---
#### [replaced 010] Assessing the Sensitivity and Alignment of FOL Closeness Metrics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.08613v3](http://arxiv.org/pdf/2501.08613v3)**

> **作者:** Ramya Keerthy Thatikonda; Wray Buntine; Ehsan Shareghi
>
> **备注:** EMNLP 2025
>
> **摘要:** The recent successful paradigm of solving logical reasoning problems with tool-augmented large language models (LLMs) leverages translation of natural language (NL) statements into First-Order Logic~(FOL) and external theorem provers. However, the correctness of FOL statements, comprising operators and text, often go unverified due to the lack of a reliable evaluation metric for comparing generated and ground-truth FOLs. In this paper, we conduct a comprehensive study on the sensitivity of existing NL-, FOL-, and graph-based metrics to capture differences between a sampled FOL and its corresponding ground-truth. We then measure the alignment between a metric-based ranking of FOL outputs and a strong LLM as-a-judge. To do this, we first apply operator and text-based perturbations to ground-truth FOL statements to assess metric sensitivity. We then evaluate metric robustness by comparing the metrics against LLMs judgment. Our empirical findings highlight a clear oversensitivity in the n-gram metric BLEU for text perturbations. The operator perturbation affects the semantic graph metric Smatch++ for structural changes, and the FOL metric for specific operator changes. We observe a closer alignment between BertScore and LLM judgement, proving the importance of semantic evaluation. Additionally, we show that combining metrics enhances both robustness and sensitivity compared to using individual metrics.
>
---
#### [replaced 011] Modeling Sequential Sentence Relation to Improve Cross-lingual Dense Retrieval
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2302.01626v2](http://arxiv.org/pdf/2302.01626v2)**

> **作者:** Shunyu Zhang; Yaobo Liang; Ming Gong; Daxin Jiang; Nan Duan
>
> **备注:** Published at ICLR 2023
>
> **摘要:** Recently multi-lingual pre-trained language models (PLM) such as mBERT and XLM-R have achieved impressive strides in cross-lingual dense retrieval. Despite its successes, they are general-purpose PLM while the multilingual PLM tailored for cross-lingual retrieval is still unexplored. Motivated by an observation that the sentences in parallel documents are approximately in the same order, which is universal across languages, we propose to model this sequential sentence relation to facilitate cross-lingual representation learning. Specifically, we propose a multilingual PLM called masked sentence model (MSM), which consists of a sentence encoder to generate the sentence representations, and a document encoder applied to a sequence of sentence vectors from a document. The document encoder is shared for all languages to model the universal sequential sentence relation across languages. To train the model, we propose a masked sentence prediction task, which masks and predicts the sentence vector via a hierarchical contrastive loss with sampled negatives. Comprehensive experiments on four cross-lingual retrieval tasks show MSM significantly outperforms existing advanced pre-training models, demonstrating the effectiveness and stronger cross-lingual retrieval capabilities of our approach. Code and model are available at https://github.com/shunyuzh/MSM.
>
---
#### [replaced 012] AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2508.01249v2](http://arxiv.org/pdf/2508.01249v2)**

> **作者:** Peiran Wang; Yang Liu; Yunfei Lu; Yifeng Cai; Hongbo Chen; Qingyou Yang; Jie Zhang; Jue Hong; Ye Wu
>
> **摘要:** Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.
>
---
#### [replaced 013] MultiStream-LLM: Bridging Modalities for Robust Sign Language Translation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00030v2](http://arxiv.org/pdf/2509.00030v2)**

> **作者:** Marshall Thomas; Edward Fish; Richard Bowden
>
> **摘要:** Despite progress in gloss-free Sign Language Translation (SLT), monolithic end-to-end models consistently fail on two critical components of natural signing: the precise recognition of high-speed fingerspelling and the integration of asynchronous non-manual cues from the face. Recent progress in Automated Sign Language Translation with Large Language Models has side stepped this challenge, forcing a single network to learn these simultaneously resulting in poor performance when tasked with translating crucial information such as names,places, and technical terms. We introduce MultiStream-LLM, a modular framework designed to overcome these limitations. Our approach employs separate, specialized predictors for continuous signing, fingerspelling, and lipreading. Each expert network first decodes its specific modality into a sequence of tokens. These parallel streams are then fused by a lightweight transformer that resolves temporal misalignments before passing the combined representation to a Large Language Model (LLM) for final sentence generation. Our method establishes a new state-of-the-art on the How2Sign benchmark with a BLEU-4 score of 23.5 and achieves 73.2% letter accuracy on the challenging ChicagoFSWildPlus fingerspelling dataset. These results validate our core hypothesis: by isolating and solving distinct recogni tion tasks before fusion, our multi-expert approach provides a more powerful and effective pathway to robust, high-fidelity sign language translation.
>
---
#### [replaced 014] What fifty-one years of Linguistics and Artificial Intelligence research tell us about their correlation: A scientometric analysis
- **分类: cs.CL; cs-CL; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.19858v2](http://arxiv.org/pdf/2411.19858v2)**

> **作者:** Mohammed Q. Shormani; Yehia A. AlSohbani
>
> **备注:** 26 pages, 15 figures
>
> **摘要:** There is a strong correlation between linguistics and artificial intelligence (AI), best manifested by deep learning language models. This study provides a thorough scientometric analysis of this correlation, synthesizing the intellectual production over 51 years, from 1974 to 2024. Web of Science Core Collection (WoSCC) database was the data source. The data collected were analyzed by two powerful software, viz., CiteSpace and VOSviewer, through which mapping visualizations of the intellectual landscape, trending issues and (re)emerging hotspots were generated. The results indicate that in the 1980s and 1990s, linguistics and AI (AIL) research was not robust, characterized by unstable publication over time. It has, however, witnessed a remarkable increase of publication since then, reaching 1478 articles in 2023, and 546 articles in January-March timespan in 2024, involving emerging issues including Natural language processing, Cross-sectional study, Using bidirectional encoder representation, and Using ChatGPT and hotspots such as Novice programmer, Prioritization, and Artificial intelligence, addressing new horizons, new topics, and launching new applications and powerful deep learning language models including ChatGPT. It concludes that linguistics and AI correlation is established at several levels, research centers, journals, and countries shaping AIL knowledge production and reshaping its future frontiers.
>
---
#### [replaced 015] TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11737v2](http://arxiv.org/pdf/2505.11737v2)**

> **作者:** Tunyu Zhang; Haizhou Shi; Yibin Wang; Hengyi Wang; Xiaoxiao He; Zhuowei Li; Haoxian Chen; Ligong Han; Kai Xu; Huan Zhang; Dimitris Metaxas; Hao Wang
>
> **备注:** Preprint; Work in progress
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a Token-level Uncertainty estimation framework for Reasoning (TokUR) to enable LLMs to self-assess and self-improve their generation quality in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation to LLM decoding, generating predictive distributions that we use to estimate token-level uncertainties. We then aggregate these uncertainties to reflect semantic uncertainty of the generated sequences. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that our token-level uncertainty metrics strongly correlate with answer correctness and model robustness. Additionally, we explore using uncertainty to directly enhance the model's reasoning performance through multiple generations and the particle filtering algorithm. Our approach consistently outperforms existing uncertainty estimation methods, establishing effective uncertainty estimation as a valuable tool for both evaluating and improving reasoning generation in LLMs.
>
---
#### [replaced 016] TECP: Token-Entropy Conformal Prediction for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00461v2](http://arxiv.org/pdf/2509.00461v2)**

> **作者:** Beining Xu; Yongming Lu
>
> **摘要:** Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, especially under black-box constraints where internal model signals are inaccessible. In this paper, we introduce Token-Entropy Conformal Prediction (TECP), a novel framework that leverages token-level entropy as a logit-free, reference-free uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike existing approaches that rely on semantic consistency heuristics or white-box features, TECP directly estimates epistemic uncertainty from the token entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) demonstrate that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-consistency-based UQ methods. Our method provides a principled and efficient solution for trustworthy generation in black-box LLM settings.
>
---
#### [replaced 017] Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18724v3](http://arxiv.org/pdf/2501.18724v3)**

> **作者:** Maya Kruse; Shiyue Hu; Nicholas Derby; Yifu Wu; Samantha Stonbraker; Bingsheng Yao; Dakuo Wang; Elizabeth Goldberg; Yanjun Gao
>
> **摘要:** Recent advances in large language models (LLMs) have shown potential in clinical text summarization, but their ability to handle long patient trajectories with multi-modal data spread across time remains underexplored. This study systematically evaluates several state-of-the-art open-source LLMs, their Retrieval Augmented Generation (RAG) variants and chain-of-thought (CoT) prompting on long-context clinical summarization and prediction. We examine their ability to synthesize structured and unstructured Electronic Health Records (EHR) data while reasoning over temporal coherence, by re-engineering existing tasks, including discharge summarization and diagnosis prediction from two publicly available EHR datasets. Our results indicate that long context windows improve input integration but do not consistently enhance clinical reasoning, and LLMs are still struggling with temporal progression and rare disease prediction. While RAG shows improvements in hallucination in some cases, it does not fully address these limitations. Our work fills the gap in long clinical text summarization, establishing a foundation for evaluating LLMs with multi-modal data and temporal reasoning.
>
---
#### [replaced 018] HuggingGraph: Understanding the Supply Chain of LLM Ecosystem
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14240v3](http://arxiv.org/pdf/2507.14240v3)**

> **作者:** Mohammad Shahedur Rahman; Peng Gao; Yuede Ji
>
> **摘要:** Large language models (LLMs) leverage deep learning architectures to process and predict sequences of words, enabling them to perform a wide range of natural language processing tasks, such as translation, summarization, question answering, and content generation. As existing LLMs are often built from base models or other pre-trained models and use external datasets, they can inevitably inherit vulnerabilities, biases, or malicious components that exist in previous models or datasets. Therefore, it is critical to understand these components' origin and development process to detect potential risks, improve model fairness, and ensure compliance with regulatory frameworks. Motivated by that, this project aims to study such relationships between models and datasets, which are the central parts of the LLM supply chain. First, we design a methodology to systematically collect LLMs' supply chain information. Then, we design a new graph to model the relationships between models and datasets, which is a directed heterogeneous graph, having 402,654 nodes and 462,524 edges. Lastly, we perform different types of analysis and make multiple interesting findings.
>
---
#### [replaced 019] StereoDetect: Detecting Stereotypes and Anti-stereotypes the Correct Way Using Social Psychological Underpinnings
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.03352v2](http://arxiv.org/pdf/2504.03352v2)**

> **作者:** Kaustubh Shivshankar Shejole; Pushpak Bhattacharyya
>
> **摘要:** Stereotypes are known to have very harmful effects, making their detection critically important. However, current research predominantly focuses on detecting and evaluating stereotypical biases, thereby leaving the study of stereotypes in its early stages. Our study revealed that many works have failed to clearly distinguish between stereotypes and stereotypical biases, which has significantly slowed progress in advancing research in this area. Stereotype and Anti-stereotype detection is a problem that requires social knowledge; hence, it is one of the most difficult areas in Responsible AI. This work investigates this task, where we propose a five-tuple definition and provide precise terminologies disentangling stereotypes, anti-stereotypes, stereotypical bias, and general bias. We provide a conceptual framework grounded in social psychology for reliable detection. We identify key shortcomings in existing benchmarks for this task of stereotype and anti-stereotype detection. To address these gaps, we developed StereoDetect, a well curated, definition-aligned benchmark dataset designed for this task. We show that sub-10B language models and GPT-4o frequently misclassify anti-stereotypes and fail to recognize neutral overgeneralizations. We demonstrate StereoDetect's effectiveness through multiple qualitative and quantitative comparisons with existing benchmarks and models fine-tuned on them. The dataset and code is available at https://github.com/KaustubhShejole/StereoDetect.
>
---
#### [replaced 020] AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.12226v4](http://arxiv.org/pdf/2402.12226v4)**

> **作者:** Jun Zhan; Junqi Dai; Jiasheng Ye; Yunhua Zhou; Dong Zhang; Zhigeng Liu; Xin Zhang; Ruibin Yuan; Ge Zhang; Linyang Li; Hang Yan; Jie Fu; Tao Gui; Tianxiang Sun; Yugang Jiang; Xipeng Qiu
>
> **备注:** 28 pages, 16 figures, under review, work in progress
>
> **摘要:** We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
>
---
#### [replaced 021] Demystifying Chains, Trees, and Graphs of Thoughts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.14295v5](http://arxiv.org/pdf/2401.14295v5)**

> **作者:** Maciej Besta; Florim Memedi; Zhenyu Zhang; Robert Gerstenberger; Guangyuan Piao; Nils Blach; Piotr Nyczyk; Marcin Copik; Grzegorz Kwaśniewski; Jürgen Müller; Lukas Gianinazzi; Ales Kubicek; Hubert Niewiadomski; Aidan O'Mahony; Onur Mutlu; Torsten Hoefler
>
> **摘要:** The field of natural language processing (NLP) has witnessed significant progress in recent years, with a notable focus on improving large language models' (LLM) performance through innovative prompting techniques. Among these, prompt engineering coupled with structures has emerged as a promising paradigm, with designs such as Chain-of-Thought, Tree of Thoughts, or Graph of Thoughts, in which the overall LLM reasoning is guided by a structure such as a graph. As illustrated with numerous examples, this paradigm significantly enhances the LLM's capability to solve numerous tasks, ranging from logical or mathematical reasoning to planning or creative writing. To facilitate the understanding of this growing field and pave the way for future developments, we devise a general blueprint for effective and efficient LLM reasoning schemes. For this, we conduct an in-depth analysis of the prompt execution pipeline, clarifying and clearly defining different concepts. We then build the first taxonomy of structure-enhanced LLM reasoning schemes. We focus on identifying fundamental classes of harnessed structures, and we analyze the representations of these structures, algorithms executed with these structures, and many others. We refer to these structures as reasoning topologies, because their representation becomes to a degree spatial, as they are contained within the LLM context. Our study compares existing prompting schemes using the proposed taxonomy, discussing how certain design choices lead to different patterns in performance and cost. We also outline theoretical underpinnings, relationships between prompting and other parts of the LLM ecosystem such as knowledge bases, and the associated research challenges. Our work will help to advance future prompt engineering techniques.
>
---
#### [replaced 022] ELIXIR: Efficient and LIghtweight model for eXplaIning Recommendations
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.20312v2](http://arxiv.org/pdf/2508.20312v2)**

> **作者:** Ben Kabongo; Vincent Guigue; Pirmin Lemberger
>
> **备注:** 10 pages, 3 figures, 6 Tables
>
> **摘要:** Collaborative filtering drives many successful recommender systems but struggles with fine-grained user-item interactions and explainability. As users increasingly seek transparent recommendations, generating textual explanations through language models has become a critical research area. Existing methods employ either RNNs or Transformers. However, RNN-based approaches fail to leverage the capabilities of pre-trained Transformer models, whereas Transformer-based methods often suffer from suboptimal adaptation and neglect aspect modeling, which is crucial for personalized explanations. We propose ELIXIR (Efficient and LIghtweight model for eXplaIning Recommendations), a multi-task model combining rating prediction with personalized review generation. ELIXIR jointly learns global and aspect-specific representations of users and items, optimizing overall rating, aspect-level ratings, and review generation, with personalized attention to emphasize aspect importance. Based on a T5-small (60M) model, we demonstrate the effectiveness of our aspect-based architecture in guiding text generation in a personalized context, where state-of-the-art approaches exploit much larger models but fail to match user preferences as well. Experimental results on TripAdvisor and RateBeer demonstrate that ELIXIR significantly outperforms strong baseline models, especially in review generation.
>
---
#### [replaced 023] ViClaim: A Multilingual Multilabel Dataset for Automatic Claim Detection in Videos
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12882v2](http://arxiv.org/pdf/2504.12882v2)**

> **作者:** Patrick Giedemann; Pius von Däniken; Jan Deriu; Alvaro Rodrigo; Anselmo Peñas; Mark Cieliebak
>
> **摘要:** The growing influence of video content as a medium for communication and misinformation underscores the urgent need for effective tools to analyze claims in multilingual and multi-topic settings. Existing efforts in misinformation detection largely focus on written text, leaving a significant gap in addressing the complexity of spoken text in video transcripts. We introduce ViClaim, a dataset of 1,798 annotated video transcripts across three languages (English, German, Spanish) and six topics. Each sentence in the transcripts is labeled with three claim-related categories: fact-check-worthy, fact-non-check-worthy, or opinion. We developed a custom annotation tool to facilitate the highly complex annotation process. Experiments with state-of-the-art multilingual language models demonstrate strong performance in cross-validation (macro F1 up to 0.896) but reveal challenges in generalization to unseen topics, particularly for distinct domains. Our findings highlight the complexity of claim detection in video transcripts. ViClaim offers a robust foundation for advancing misinformation detection in video-based communication, addressing a critical gap in multimodal analysis.
>
---
#### [replaced 024] MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading
- **分类: q-fin.TR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.20474v2](http://arxiv.org/pdf/2507.20474v2)**

> **作者:** Siyi Wu; Junqiao Wang; Zhaoyang Guan; Leyi Zhao; Xinyuan Song; Xinyu Ying; Dexu Yu; Jinhao Wang; Hanlin Zhang; Michele Pak; Yangfan He; Yi Xin; Jianhui Wang; Tianyu Shi
>
> **摘要:** Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.
>
---
#### [replaced 025] Social Bias in Multilingual Language Models: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20201v2](http://arxiv.org/pdf/2508.20201v2)**

> **作者:** Lance Calvin Lim Gamboa; Yue Feng; Mark Lee
>
> **备注:** Accepted into EMNLP 2025 Main Conference
>
> **摘要:** Pretrained multilingual models exhibit the same social bias as models processing English texts. This systematic review analyzes emerging research that extends bias evaluation and mitigation approaches into multilingual and non-English contexts. We examine these studies with respect to linguistic diversity, cultural awareness, and their choice of evaluation metrics and mitigation techniques. Our survey illuminates gaps in the field's dominant methodological design choices (e.g., preference for certain languages, scarcity of multilingual mitigation experiments) while cataloging common issues encountered and solutions implemented in adapting bias benchmarks across languages and cultures. Drawing from the implications of our findings, we chart directions for future research that can reinforce the multilingual bias literature's inclusivity, cross-cultural appropriateness, and alignment with state-of-the-art NLP advancements.
>
---
#### [replaced 026] MultiWikiQA: A Reading Comprehension Benchmark in 300+ Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04111v2](http://arxiv.org/pdf/2509.04111v2)**

> **作者:** Dan Saattrup Smart
>
> **摘要:** We introduce a new reading comprehension dataset, dubbed MultiWikiQA, which covers 306 languages. The context data comes from Wikipedia articles, with questions generated by an LLM and the answers appearing verbatim in the Wikipedia articles. We conduct a crowdsourced human evaluation of the fluency of the generated questions across 30 of the languages, providing evidence that the questions are of good quality. We evaluate 6 different language models, both decoder and encoder models of varying sizes, showing that the benchmark is sufficiently difficult and that there is a large performance discrepancy amongst the languages. The dataset and survey evaluations are freely available.
>
---
#### [replaced 027] Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06460v2](http://arxiv.org/pdf/2504.06460v2)**

> **作者:** Sai Adith Senthil Kumar; Hao Yan; Saipavan Perepa; Murong Yue; Ziyu Yao
>
> **摘要:** Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research.
>
---
#### [replaced 028] Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21934v5](http://arxiv.org/pdf/2503.21934v5)**

> **作者:** Ivo Petrov; Jasper Dekoninck; Lyuben Baltadzhiev; Maria Drencheva; Kristian Minchev; Mislav Balunović; Nikola Jovanović; Martin Vechev
>
> **摘要:** Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, Gemini-2.5-Pro, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce a comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly: only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities.
>
---
#### [replaced 029] Text2Cypher Across Languages: Evaluating and Finetuning LLMs
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.21445v2](http://arxiv.org/pdf/2506.21445v2)**

> **作者:** Makbule Gulcin Ozsoy; William Tai
>
> **摘要:** Recent advances in large language models (LLMs) have enabled natural language interfaces that translate user questions into database queries, such as Text2SQL, Text2SPARQL, and Text2Cypher. While these interfaces enhance database accessibility, most research today focuses on English, with limited evaluation in other languages. This paper investigates the performance of both foundational and finetuned LLMs on the Text2Cypher task across multiple languages. We create and release a multilingual dataset by translating English questions into Spanish and Turkish while preserving the original Cypher queries, enabling fair cross-lingual comparison. Using standardized prompts and metrics, we evaluate several foundational models and observe a consistent performance pattern: highest on English, followed by Spanish, and lowest on Turkish. We attribute this to differences in training data availability and linguistic features. We also examine the impact of translating task prompts into Spanish and Turkish. Results show little to no change in evaluation metrics, suggesting prompt translation has minor impact. Furthermore, we finetune a foundational model on two datasets: one in English only, and one multilingual. Finetuning on English improves overall accuracy but widens the performance gap between languages. In contrast, multilingual finetuning narrows the gap, resulting in more balanced performance. Our findings highlight the importance for multilingual evaluation and training to build more inclusive and robust query generation systems.
>
---
#### [replaced 030] The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2509.03730v2](http://arxiv.org/pdf/2509.03730v2)**

> **作者:** Pengrui Han; Rafal Kocielnik; Peiyang Song; Ramit Debnath; Dean Mobbs; Anima Anandkumar; R. Michael Alvarez
>
> **备注:** We make public all code and source data at https://github.com/psychology-of-AI/Personality-Illusion for full reproducibility
>
> **摘要:** Personality traits have long been studied as predictors of human behavior. Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.
>
---
#### [replaced 031] PersonaGym: Evaluating Persona Agents and LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.18416v5](http://arxiv.org/pdf/2407.18416v5)**

> **作者:** Vinay Samuel; Henry Peng Zou; Yue Zhou; Shreyas Chaudhari; Ashwin Kalyan; Tanmay Rajpurohit; Ameet Deshpande; Karthik Narasimhan; Vishvak Murahari
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** Persona agents, which are LLM agents conditioned to act according to an assigned persona, enable contextually rich and user aligned interactions across domains like education and healthcare. However, evaluating how faithfully these agents adhere to their personas remains a significant challenge, particularly in free-form settings that demand consistency across diverse, persona-relevant environments. We introduce PersonaGym, the first dynamic evaluation framework for persona agents, and PersonaScore, a human-aligned automatic metric grounded in decision theory that enables comprehensive large-scale evaluation. Our evaluation of 10 leading LLMs across 200 personas and 10,000 questions reveals significant advancement opportunities. For example, GPT-4.1 had the exact same PersonaScore as LLaMA-3-8b despite being a more recent and advanced closed source model. Importantly, increased model size and complexity do not necessarily enhance persona agent capabilities, underscoring the need for algorithmic and architectural innovation toward faithful, performant persona agents.
>
---
#### [replaced 032] Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05528v2](http://arxiv.org/pdf/2507.05528v2)**

> **作者:** Jiahuan Pei; Fanghua Ye; Xin Sun; Wentao Deng; Koen Hindriks; Junxiao Wang
>
> **备注:** 14 pages, accepted by EMNLP 2025
>
> **摘要:** Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
>
---
#### [replaced 033] ATHAR: A High-Quality and Diverse Dataset for Classical Arabic to English Translation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.19835v2](http://arxiv.org/pdf/2407.19835v2)**

> **作者:** Mohammed Khalil; Mohammed Sabry
>
> **备注:** ArabicNLP 2025
>
> **摘要:** Classical Arabic represents a significant era that encompasses the golden age of Arab culture, philosophy, and scientific literature. With a broad consensus on the importance of translating these literatures to enrich knowledge dissemination across communities, the advent of large language models (LLMs) and translation systems offers promising tools to facilitate this goal. However, we have identified a scarcity of translation datasets in Classical Arabic, which are often limited in scope and topics, hindering the development of high-quality translation systems. In response, we present the ATHAR dataset, which comprises 66,000 high-quality classical Arabic to English translation samples that cover a wide array of topics including science, culture, and philosophy. Furthermore, we assess the performance of current state-of-the-art LLMs under various settings, concluding that there is a need for such datasets in current systems. Our findings highlight how models can benefit from fine-tuning or incorporating this dataset into their pretraining pipelines. The dataset is publicly available on the HuggingFace Data Hub: https://huggingface.co/datasets/mohamed-khalil/ATHAR.
>
---
#### [replaced 034] Persona Vectors: Monitoring and Controlling Character Traits in Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.21509v3](http://arxiv.org/pdf/2507.21509v3)**

> **作者:** Runjin Chen; Andy Arditi; Henry Sleight; Owain Evans; Jack Lindsey
>
> **摘要:** Large language models interact with users through a simulated 'Assistant' persona. While the Assistant is typically trained to be helpful, harmless, and honest, it sometimes deviates from these ideals. In this paper, we identify directions in the model's activation space-persona vectors-underlying several traits, such as evil, sycophancy, and propensity to hallucinate. We confirm that these vectors can be used to monitor fluctuations in the Assistant's personality at deployment time. We then apply persona vectors to predict and control personality shifts that occur during training. We find that both intended and unintended personality changes after finetuning are strongly correlated with shifts along the relevant persona vectors. These shifts can be mitigated through post-hoc intervention, or avoided in the first place with a new preventative steering method. Moreover, persona vectors can be used to flag training data that will produce undesirable personality changes, both at the dataset level and the individual sample level. Our method for extracting persona vectors is automated and can be applied to any personality trait of interest, given only a natural-language description.
>
---
#### [replaced 035] Selective Preference Optimization via Token-Level Reward Function Estimation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.13518v2](http://arxiv.org/pdf/2408.13518v2)**

> **作者:** Kailai Yang; Zhiwei Liu; Qianqian Xie; Jimin Huang; Erxue Min; Sophia Ananiadou
>
> **备注:** Accepted by the EMNLP 2025 main conference
>
> **摘要:** Recent advancements in large language model alignment leverage token-level supervisions to perform fine-grained preference optimization. However, existing token-level alignment methods either optimize on all available tokens, which can be noisy and inefficient, or perform selective training with complex and expensive key token selection strategies. In this work, we propose Selective Preference Optimization (SePO), a novel selective alignment strategy that centers on efficient key token selection. SePO proposes the first token selection method based on Direct Preference Optimization (DPO), which trains an oracle model to estimate a token-level reward function on the target data. This method applies to any existing alignment datasets with response-level annotations and enables cost-efficient token selection with small-scale oracle models and training data. The estimated reward function is then utilized to score all tokens within the target dataset, where only the key tokens are selected to supervise the target policy model with a reference model-free contrastive objective function. Extensive experiments on three public evaluation benchmarks show that SePO significantly outperforms competitive baseline methods by only optimizing 30% key tokens on the target dataset. SePO applications on weak-to-strong generalization show that weak oracle models effectively supervise strong policy models with up to 16.8x more parameters. SePO also effectively selects key tokens from out-of-distribution data to enhance strong policy models and alleviate the over-optimization problem.
>
---
#### [replaced 036] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.12929v3](http://arxiv.org/pdf/2409.12929v3)**

> **作者:** Jin Jiang; Yuchen Yan; Yang Liu; Jianing Wang; Shuai Peng; Xunliang Cai; Yixin Cao; Mengdi Zhang; Liangcai Gao
>
> **备注:** 19 pages, ACL 2025 (Volume 1 Long Papers), pages 26200-26218
>
> **摘要:** In this paper, we propose a new data synthesis method called \textbf{LogicPro}, which leverages LeetCode-style algorithm \underline{Pro}blems and their corresponding \underline{Pro}gram solutions to synthesize Complex \underline{Logic}al Reasoning data in text format. First, we synthesize complex reasoning problems through source algorithm problems and test cases. Then, standard answers and intermediate variable outputs are obtained for each problem based on standard python solutions and test cases. Finally, with the guidance of code intermediate variables, we synthesize the text reasoning process for each reasoning problems. Through this method, we can synthesize data that is difficult, scalable, effective, and comes with golden standard answers and high-quality reasoning processes. As a result, with our 540K synthesized dataset constructed solely from 2,360 algorithm problems, our approach \footnote{Code and data are publicly available at https://github.com/jiangjin1999/LogicPro} achieves significant improvements in multiple models for the datasets \textit{BBH$^{27}$}, \textit{LogicBench}, \textit{DROP}, \textit{AR-LSAT}, and \textit{GSM8K}, etc. outperforming a wide range of existing reasoning datasets.
>
---
