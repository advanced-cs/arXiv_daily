# 自然语言处理 cs.CL

- **最新发布 254 篇**

- **更新 164 篇**

## 最新发布

#### [new 001] KG-QAGen: A Knowledge-Graph-Based Framework for Systematic Question Generation and Long-Context LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文属于长上下文语言模型评估任务，旨在解决现有基准无法系统控制问题复杂度的问题。提出KG-QAGen框架，通过知识图谱结构化生成多难度QA对（含20,139项），评估13个模型发现其在集合运算和多跳推理存在缺陷，揭示语义误解与隐式关系处理不足的失败模式。**

- **链接: [http://arxiv.org/pdf/2505.12495v1](http://arxiv.org/pdf/2505.12495v1)**

> **作者:** Nikita Tatarinov; Vidhyakshaya Kannan; Haricharana Srinivasa; Arnav Raj; Harpreet Singh Anand; Varun Singh; Aditya Luthra; Ravij Lade; Agam Shah; Sudheer Chava
>
> **摘要:** The increasing context length of modern language models has created a need for evaluating their ability to retrieve and process information across extensive documents. While existing benchmarks test long-context capabilities, they often lack a structured way to systematically vary question complexity. We introduce KG-QAGen (Knowledge-Graph-based Question-Answer Generation), a framework that (1) extracts QA pairs at multiple complexity levels (2) by leveraging structured representations of financial agreements (3) along three key dimensions -- multi-hop retrieval, set operations, and answer plurality -- enabling fine-grained assessment of model performance across controlled difficulty levels. Using this framework, we construct a dataset of 20,139 QA pairs (the largest number among the long-context benchmarks) and open-source a part of it. We evaluate 13 proprietary and open-source LLMs and observe that even the best-performing models are struggling with set-based comparisons and multi-hop logical inference. Our analysis reveals systematic failure modes tied to semantic misinterpretation and inability to handle implicit relations.
>
---
#### [new 002] Bidirectional LMs are Better Knowledge Memorizers? A Benchmark for Real-world Knowledge Injection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的知识记忆评估任务，旨在解决大语言模型（LLMs）知识记忆能力缺乏标准化测试的问题。通过构建自动化扩展的维基百科基准WikiDYK，发现双向语言模型（BiLMs）比因果模型（CLMs）知识记忆更可靠，并提出协作框架提升LLMs性能。**

- **链接: [http://arxiv.org/pdf/2505.12306v1](http://arxiv.org/pdf/2505.12306v1)**

> **作者:** Yuwei Zhang; Wenhao Yu; Shangbin Feng; Yifan Zhu; Letian Peng; Jayanth Srinivasa; Gaowen Liu; Jingbo Shang
>
> **备注:** Dataset is available at https://huggingface.co/datasets/YWZBrandon/wikidyk
>
> **摘要:** Despite significant advances in large language models (LLMs), their knowledge memorization capabilities remain underexplored, due to the lack of standardized and high-quality test ground. In this paper, we introduce a novel, real-world and large-scale knowledge injection benchmark that evolves continuously over time without requiring human intervention. Specifically, we propose WikiDYK, which leverages recently-added and human-written facts from Wikipedia's "Did You Know..." entries. These entries are carefully selected by expert Wikipedia editors based on criteria such as verifiability and clarity. Each entry is converted into multiple question-answer pairs spanning diverse task formats from easy cloze prompts to complex multi-hop questions. WikiDYK contains 12,290 facts and 77,180 questions, which is also seamlessly extensible with future updates from Wikipedia editors. Extensive experiments using continued pre-training reveal a surprising insight: despite their prevalence in modern LLMs, Causal Language Models (CLMs) demonstrate significantly weaker knowledge memorization capabilities compared to Bidirectional Language Models (BiLMs), exhibiting a 23% lower accuracy in terms of reliability. To compensate for the smaller scales of current BiLMs, we introduce a modular collaborative framework utilizing ensembles of BiLMs as external knowledge repositories to integrate with LLMs. Experiment shows that our framework further improves the reliability accuracy by up to 29.1%.
>
---
#### [new 003] Towards Reliable and Interpretable Traffic Crash Pattern Prediction and Safety Interventions Using Customized Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于交通事故预测与安全干预任务，旨在解决现有方法无法有效解析多源数据（数值、文本、图像等）及其复杂关联的问题。通过构建TrafficSafe框架，将事故预测转化为LLM的文本推理任务，并开发特征归因模型分析风险因素。基于58,903份多模态数据微调LLM，F1值提升42%，揭示酒驾是主因，指导数据收集优化。**

- **链接: [http://arxiv.org/pdf/2505.12545v1](http://arxiv.org/pdf/2505.12545v1)**

> **作者:** Yang Zhao; Pu Wang; Yibo Zhao; Hongru Du; Hao; Yang
>
> **备注:** Last revised 13 Feb 2025. Under review in Nature portfolio
>
> **摘要:** Predicting crash events is crucial for understanding crash distributions and their contributing factors, thereby enabling the design of proactive traffic safety policy interventions. However, existing methods struggle to interpret the complex interplay among various sources of traffic crash data, including numeric characteristics, textual reports, crash imagery, environmental conditions, and driver behavior records. As a result, they often fail to capture the rich semantic information and intricate interrelationships embedded in these diverse data sources, limiting their ability to identify critical crash risk factors. In this research, we propose TrafficSafe, a framework that adapts LLMs to reframe crash prediction and feature attribution as text-based reasoning. A multi-modal crash dataset including 58,903 real-world reports together with belonged infrastructure, environmental, driver, and vehicle information is collected and textualized into TrafficSafe Event Dataset. By customizing and fine-tuning LLMs on this dataset, the TrafficSafe LLM achieves a 42% average improvement in F1-score over baselines. To interpret these predictions and uncover contributing factors, we introduce TrafficSafe Attribution, a sentence-level feature attribution framework enabling conditional risk analysis. Findings show that alcohol-impaired driving is the leading factor in severe crashes, with aggressive and impairment-related behaviors having nearly twice the contribution for severe crashes compared to other driver behaviors. Furthermore, TrafficSafe Attribution highlights pivotal features during model training, guiding strategic crash data collection for iterative performance improvements. The proposed TrafficSafe offers a transformative leap in traffic safety research, providing a blueprint for translating advanced AI technologies into responsible, actionable, and life-saving outcomes.
>
---
#### [new 004] EmoHopeSpeech: An Annotated Dataset of Emotions and Hope Speech in English
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理的情感分析任务，旨在解决多情感（情感与希望言论）双语数据集稀缺问题。作者构建了阿拉伯语和英语标注数据集，包含情感强度、原因及希望子类，并通过高标注一致性验证（Fleiss' Kappa 0.75-0.85）和基线模型（微F1=0.67）证明数据有效性，助力低资源语言研究。**

- **链接: [http://arxiv.org/pdf/2505.11959v1](http://arxiv.org/pdf/2505.11959v1)**

> **作者:** Md. Rafiul Biswas; Wajdi Zaghouani
>
> **摘要:** This research introduces a bilingual dataset comprising 23,456 entries for Arabic and 10,036 entries for English, annotated for emotions and hope speech, addressing the scarcity of multi-emotion (Emotion and hope) datasets. The dataset provides comprehensive annotations capturing emotion intensity, complexity, and causes, alongside detailed classifications and subcategories for hope speech. To ensure annotation reliability, Fleiss' Kappa was employed, revealing 0.75-0.85 agreement among annotators both for Arabic and English language. The evaluation metrics (micro-F1-Score=0.67) obtained from the baseline model (i.e., using a machine learning model) validate that the data annotations are worthy. This dataset offers a valuable resource for advancing natural language processing in underrepresented languages, fostering better cross-linguistic analysis of emotions and hope speech.
>
---
#### [new 005] WikiPersonas: What Can We Learn From Personalized Alignment to Famous People?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究个性化模型对齐任务，解决现有方法忽略用户偏好多样性、缺乏细粒度数据的问题。通过构建首个基于名人的细粒度数据集WikiPersona，要求模型生成可验证的人物背景描述并进行偏好对齐。实验表明，采用推断偏好前缀的方法在矛盾偏好场景下更有效，且能提升未知人物的泛化公平性。**

- **链接: [http://arxiv.org/pdf/2505.13257v1](http://arxiv.org/pdf/2505.13257v1)**

> **作者:** Zilu Tang; Afra Feyza Akyürek; Ekin Akyürek; Derry Wijaya
>
> **备注:** 9 pages, preprint
>
> **摘要:** Preference alignment has become a standard pipeline in finetuning models to follow \emph{generic} human preferences. Majority of work seeks to optimize model to produce responses that would be preferable \emph{on average}, simplifying the diverse and often \emph{contradicting} space of human preferences. While research has increasingly focused on personalized alignment: adapting models to individual user preferences, there is a lack of personalized preference dataset which focus on nuanced individual-level preferences. To address this, we introduce WikiPersona: the first fine-grained personalization using well-documented, famous individuals. Our dataset challenges models to align with these personas through an interpretable process: generating verifiable textual descriptions of a persona's background and preferences in addition to alignment. We systematically evaluate different personalization approaches and find that as few-shot prompting with preferences and fine-tuning fail to simultaneously ensure effectiveness and efficiency, using \textit{inferred personal preferences} as prefixes enables effective personalization, especially in topics where preferences clash while leading to more equitable generalization across unseen personas.
>
---
#### [new 006] Do different prompting methods yield a common task representation in language models?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型中不同提示方法（示例演示与文本指令）是否生成统一任务表征，属模型可解释性任务。通过扩展函数向量方法提取指令型任务表征，发现两种提示激活不同模型组件且表征部分重叠，证明任务表征具有形式依赖性，为组合式提示提供理论依据，揭示了跨形式任务监控的复杂性。**

- **链接: [http://arxiv.org/pdf/2505.12075v1](http://arxiv.org/pdf/2505.12075v1)**

> **作者:** Guy Davidson; Todd M. Gureckis; Brenden M. Lake; Adina Williams
>
> **备注:** 9 pages, 4 figures; under review
>
> **摘要:** Demonstrations and instructions are two primary approaches for prompting language models to perform in-context learning (ICL) tasks. Do identical tasks elicited in different ways result in similar representations of the task? An improved understanding of task representation mechanisms would offer interpretability insights and may aid in steering models. We study this through function vectors, recently proposed as a mechanism to extract few-shot ICL task representations. We generalize function vectors to alternative task presentations, focusing on short textual instruction prompts, and successfully extract instruction function vectors that promote zero-shot task accuracy. We find evidence that demonstration- and instruction-based function vectors leverage different model components, and offer several controls to dissociate their contributions to task performance. Our results suggest that different task presentations do not induce a common task representation but elicit different, partly overlapping mechanisms. Our findings offer principled support to the practice of combining textual instructions and task demonstrations, imply challenges in universally monitoring task inference across presentation forms, and encourage further examinations of LLM task inference mechanisms.
>
---
#### [new 007] DS-ProGen: A Dual-Structure Deep Language Model for Functional Protein Design
- **分类: cs.CL**

- **简介: 该论文属于蛋白质设计中的逆蛋白质折叠任务，旨在解决现有方法仅依赖单一结构特征导致预测精度不足的问题。提出了DS-ProGen双结构深度模型，联合主链坐标和表面特征生成稳定且功能性的氨基酸序列，在PRIDE数据集达到61.47%的恢复率，并验证了其生物分子交互能力。**

- **链接: [http://arxiv.org/pdf/2505.12511v1](http://arxiv.org/pdf/2505.12511v1)**

> **作者:** Yanting Li; Jiyue Jiang; Zikang Wang; Ziqian Lin; Dongchen He; Yuheng Shan; Yanruisheng Shao; Jiayi Li; Xiangyu Shi; Jiuming Wang; Yanyu Chen; Yimin Fan; Han Li; Yu Li
>
> **摘要:** Inverse Protein Folding (IPF) is a critical subtask in the field of protein design, aiming to engineer amino acid sequences capable of folding correctly into a specified three-dimensional (3D) conformation. Although substantial progress has been achieved in recent years, existing methods generally rely on either backbone coordinates or molecular surface features alone, which restricts their ability to fully capture the complex chemical and geometric constraints necessary for precise sequence prediction. To address this limitation, we present DS-ProGen, a dual-structure deep language model for functional protein design, which integrates both backbone geometry and surface-level representations. By incorporating backbone coordinates as well as surface chemical and geometric descriptors into a next-amino-acid prediction paradigm, DS-ProGen is able to generate functionally relevant and structurally stable sequences while satisfying both global and local conformational constraints. On the PRIDE dataset, DS-ProGen attains the current state-of-the-art recovery rate of 61.47%, demonstrating the synergistic advantage of multi-modal structural encoding in protein design. Furthermore, DS-ProGen excels in predicting interactions with a variety of biological partners, including ligands, ions, and RNA, confirming its robust functional retention capabilities.
>
---
#### [new 008] RLAP: A Reinforcement Learning Enhanced Adaptive Planning Framework for Multi-step NLP Task Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多步NLP任务中现有规划方法忽略语言特征、依赖LLM固有能力导致效果不佳的问题，提出强化学习增强的自适应规划框架（RLAP）。通过将任务建模为马尔可夫决策过程，训练轻量级Actor模型评估语言序列Q值，动态优化子任务顺序，提升LLM任务解决能力。**

- **链接: [http://arxiv.org/pdf/2505.11893v1](http://arxiv.org/pdf/2505.11893v1)**

> **作者:** Zepeng Ding; Dixuan Wang; Ziqin Luo; Guochao Jiang; Deqing Yang; Jiaqing Liang
>
> **摘要:** Multi-step planning has been widely employed to enhance the performance of large language models (LLMs) on downstream natural language processing (NLP) tasks, which decomposes the original task into multiple subtasks and guide LLMs to solve them sequentially without additional training. When addressing task instances, existing methods either preset the order of steps or attempt multiple paths at each step. However, these methods overlook instances' linguistic features and rely on the intrinsic planning capabilities of LLMs to evaluate intermediate feedback and then select subtasks, resulting in suboptimal outcomes. To better solve multi-step NLP tasks with LLMs, in this paper we propose a Reinforcement Learning enhanced Adaptive Planning framework (RLAP). In our framework, we model an NLP task as a Markov decision process (MDP) and employ an LLM directly into the environment. In particular, a lightweight Actor model is trained to estimate Q-values for natural language sequences consisting of states and actions through reinforcement learning. Therefore, during sequential planning, the linguistic features of each sequence in the MDP can be taken into account, and the Actor model interacts with the LLM to determine the optimal order of subtasks for each task instance. We apply RLAP on three different types of NLP tasks and conduct extensive experiments on multiple datasets to verify RLAP's effectiveness and robustness.
>
---
#### [new 009] PyFCG: Fluid Construction Grammar in Python
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出PyFCG开源库，将流体构式语法（FCG）移植到Python环境，解决FCG与Python生态整合问题。通过开发兼容工具库，支持语法形式化验证、基于语料库的构式语法学习及多智能体语言演化实验，实现FCG与Python生态的协同应用。**

- **链接: [http://arxiv.org/pdf/2505.12920v1](http://arxiv.org/pdf/2505.12920v1)**

> **作者:** Paul Van Eecke; Katrien Beuls
>
> **摘要:** We present PyFCG, an open source software library that ports Fluid Construction Grammar (FCG) to the Python programming language. PyFCG enables its users to seamlessly integrate FCG functionality into Python programs, and to use FCG in combination with other libraries within Python's rich ecosystem. Apart from a general description of the library, this paper provides three walkthrough tutorials that demonstrate example usage of PyFCG in typical use cases of FCG: (i) formalising and testing construction grammar analyses, (ii) learning usage-based construction grammars from corpora, and (iii) implementing agent-based experiments on emergent communication.
>
---
#### [new 010] Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究人机对话中的发言转换和反馈预测任务，解决多模态信号利用不足及数据缺乏问题。通过自动采集构建含210小时视频的MM-F2F数据集，提出端到端多模态融合模型，强调模态关联性并支持任意输入组合，实验显示F1值在两项任务分别提升10%和33%。**

- **链接: [http://arxiv.org/pdf/2505.12654v1](http://arxiv.org/pdf/2505.12654v1)**

> **作者:** Yuxin Lin; Yinglin Zheng; Ming Zeng; Wangzheng Shi
>
> **备注:** Accepected by ACL 2025
>
> **摘要:** This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10\% increase in F1-score on turn-taking and a 33\% increase on backchannel prediction. Our dataset and code are publicly available online to ease of subsequent research.
>
---
#### [new 011] Think Before You Attribute: Improving the Performance of LLMs Attribution Systems
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大语言模型（LLMs）生成结果缺乏可信溯源的问题，提出基于检索增强生成（RAG）的句子级预归因方法。通过将句子分类为不可归因、单/多源归因三类，优化溯源系统计算效率与准确性，并构建清洗后的HAGRID数据集及端到端溯源系统，提升科学场景下LLMs输出的可验证性。**

- **链接: [http://arxiv.org/pdf/2505.12621v1](http://arxiv.org/pdf/2505.12621v1)**

> **作者:** João Eduardo Batista; Emil Vatai; Mohamed Wahib
>
> **备注:** 22 pages (9 pages of content, 4 pages of references, 9 pages of supplementary material), 7 figures, 10 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly applied in various science domains, yet their broader adoption remains constrained by a critical challenge: the lack of trustworthy, verifiable outputs. Current LLMs often generate answers without reliable source attribution, or worse, with incorrect attributions, posing a barrier to their use in scientific and high-stakes settings, where traceability and accountability are non-negotiable. To be reliable, attribution systems need high accuracy and retrieve data with short lengths, i.e., attribute to a sentence within a document rather than a whole document. We propose a sentence-level pre-attribution step for Retrieve-Augmented Generation (RAG) systems that classify sentences into three categories: not attributable, attributable to a single quote, and attributable to multiple quotes. By separating sentences before attribution, a proper attribution method can be selected for the type of sentence, or the attribution can be skipped altogether. Our results indicate that classifiers are well-suited for this task. In this work, we propose a pre-attribution step to reduce the computational complexity of attribution, provide a clean version of the HAGRID dataset, and provide an end-to-end attribution system that works out of the box.
>
---
#### [new 012] Mobile-Bench-v2: A More Realistic and Comprehensive Benchmark for VLM-based Mobile Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mobile-Bench-v2，针对VLM移动智能体现有基准动态环境不稳定、评估路径单一、缺乏噪声和主动交互测试的问题，构建多路径离线评估、噪声环境及模糊指令交互的综合性基准，通过多任务分割验证不同框架性能，提升移动代理的实用性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.11891v1](http://arxiv.org/pdf/2505.11891v1)**

> **作者:** Weikai Xu; Zhizheng Jiang; Yuxuan Liu; Wei Liu; Jian Luan; Yuanchun Li; Yunxin Liu; Bin Wang; Bo An
>
> **摘要:** VLM-based mobile agents are increasingly popular due to their capabilities to interact with smartphone GUIs and XML-structured texts and to complete daily tasks. However, existing online benchmarks struggle with obtaining stable reward signals due to dynamic environmental changes. Offline benchmarks evaluate the agents through single-path trajectories, which stands in contrast to the inherently multi-solution characteristics of GUI tasks. Additionally, both types of benchmarks fail to assess whether mobile agents can handle noise or engage in proactive interactions due to a lack of noisy apps or overly full instructions during the evaluation process. To address these limitations, we use a slot-based instruction generation method to construct a more realistic and comprehensive benchmark named Mobile-Bench-v2. Mobile-Bench-v2 includes a common task split, with offline multi-path evaluation to assess the agent's ability to obtain step rewards during task execution. It contains a noisy split based on pop-ups and ads apps, and a contaminated split named AITZ-Noise to formulate a real noisy environment. Furthermore, an ambiguous instruction split with preset Q\&A interactions is released to evaluate the agent's proactive interaction capabilities. We conduct evaluations on these splits using the single-agent framework AppAgent-v1, the multi-agent framework Mobile-Agent-v2, as well as other mobile agents such as UI-Tars and OS-Atlas. Code and data are available at https://huggingface.co/datasets/xwk123/MobileBench-v2.
>
---
#### [new 013] Enriching Patent Claim Generation with European Patent Dataset
- **分类: cs.CL**

- **简介: 该论文属于专利权利要求生成任务，旨在解决现有研究依赖美国专利数据导致的泛化性不足问题。通过构建欧洲专利数据集EPD，提供多司法管辖、高质量授权文本及真实挑战样本，提升大语言模型生成质量与跨域能力，实验显示EPD微调模型优于现有方法及GPT-4o。**

- **链接: [http://arxiv.org/pdf/2505.12568v1](http://arxiv.org/pdf/2505.12568v1)**

> **作者:** Lekang Jiang; Chengzu Li; Stephan Goetz
>
> **备注:** 18 pages, 13 tables, 4 figures
>
> **摘要:** Drafting patent claims is time-intensive, costly, and requires professional skill. Therefore, researchers have investigated large language models (LLMs) to assist inventors in writing claims. However, existing work has largely relied on datasets from the United States Patent and Trademark Office (USPTO). To enlarge research scope regarding various jurisdictions, drafting conventions, and legal standards, we introduce EPD, a European patent dataset. EPD presents rich textual data and structured metadata to support multiple patent-related tasks, including claim generation. This dataset enriches the field in three critical aspects: (1) Jurisdictional diversity: Patents from different offices vary in legal and drafting conventions. EPD fills a critical gap by providing a benchmark for European patents to enable more comprehensive evaluation. (2) Quality improvement: EPD offers high-quality granted patents with finalized and legally approved texts, whereas others consist of patent applications that are unexamined or provisional. Experiments show that LLMs fine-tuned on EPD significantly outperform those trained on previous datasets and even GPT-4o in claim quality and cross-domain generalization. (3) Real-world simulation: We propose a difficult subset of EPD to better reflect real-world challenges of claim generation. Results reveal that all tested LLMs perform substantially worse on these challenging samples, which highlights the need for future research.
>
---
#### [new 014] THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering
- **分类: cs.CL**

- **简介: 该论文针对RAG问答系统的评估问题，提出无参考框架THELMA。通过设计六个关联指标实现整体细粒度评估，支持端到端流程优化，无需标注数据。研究揭示了指标间关联性，可定位需改进的RAG组件。**

- **链接: [http://arxiv.org/pdf/2505.11626v1](http://arxiv.org/pdf/2505.11626v1)**

> **作者:** Udita Patel; Rutu Mulkar; Jay Roberts; Cibi Chakravarthy Senthilkumar; Sujay Gandhi; Xiaofei Zheng; Naumaan Nayyar; Rafael Castrillo
>
> **摘要:** We propose THELMA (Task Based Holistic Evaluation of Large Language Model Applications), a reference free framework for RAG (Retrieval Augmented generation) based question answering (QA) applications. THELMA consist of six interdependent metrics specifically designed for holistic, fine grained evaluation of RAG QA applications. THELMA framework helps developers and application owners evaluate, monitor and improve end to end RAG QA pipelines without requiring labelled sources or reference responses.We also present our findings on the interplay of the proposed THELMA metrics, which can be interpreted to identify the specific RAG component needing improvement in QA applications.
>
---
#### [new 015] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science
- **分类: cs.CL**

- **简介: 该论文提出首个种子科学多任务基准SeedBench，属于LLM评估任务。旨在解决种子科学领域因缺乏标准化测试导致大模型应用受限的问题。通过模拟育种流程构建评估体系，测试了26个主流模型，揭示现有模型与真实需求的差距，为LLM在种质设计中的应用奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.13220v1](http://arxiv.org/pdf/2505.13220v1)**

> **作者:** Jie Ying; Zihong Chen; Zhefan Wang; Wanli Jiang; Chenyang Wang; Zhonghang Yuan; Haoyang Su; Huanjun Kong; Fan Yang; Nanqing Dong
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Seed science is essential for modern agriculture, directly influencing crop yields and global food security. However, challenges such as interdisciplinary complexity and high costs with limited returns hinder progress, leading to a shortage of experts and insufficient technological support. While large language models (LLMs) have shown promise across various fields, their application in seed science remains limited due to the scarcity of digital resources, complex gene-trait relationships, and the lack of standardized benchmarks. To address this gap, we introduce SeedBench -- the first multi-task benchmark specifically designed for seed science. Developed in collaboration with domain experts, SeedBench focuses on seed breeding and simulates key aspects of modern breeding processes. We conduct a comprehensive evaluation of 26 leading LLMs, encompassing proprietary, open-source, and domain-specific fine-tuned models. Our findings not only highlight the substantial gaps between the power of LLMs and the real-world seed science problems, but also make a foundational step for research on LLMs for seed design.
>
---
#### [new 016] Decoding the Mind of Large Language Models: A Quantitative Evaluation of Ideology and Biases
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于大语言模型（LLMs）的伦理评估任务，旨在量化分析其意识形态偏见与社会影响。通过436道无标准答案的二元选择题框架，评估ChatGPT和Gemini的立场一致性、跨语言差异及伦理问题，发现模型存在偏见倾向与不道德表述，强调开发需兼顾社会对齐的定量评估方法。**

- **链接: [http://arxiv.org/pdf/2505.12183v1](http://arxiv.org/pdf/2505.12183v1)**

> **作者:** Manari Hirose; Masato Uchida
>
> **备注:** 23 pages, 5 figures, 17 tables
>
> **摘要:** The widespread integration of Large Language Models (LLMs) across various sectors has highlighted the need for empirical research to understand their biases, thought patterns, and societal implications to ensure ethical and effective use. In this study, we propose a novel framework for evaluating LLMs, focusing on uncovering their ideological biases through a quantitative analysis of 436 binary-choice questions, many of which have no definitive answer. By applying our framework to ChatGPT and Gemini, findings revealed that while LLMs generally maintain consistent opinions on many topics, their ideologies differ across models and languages. Notably, ChatGPT exhibits a tendency to change their opinion to match the questioner's opinion. Both models also exhibited problematic biases, unethical or unfair claims, which might have negative societal impacts. These results underscore the importance of addressing both ideological and ethical considerations when evaluating LLMs. The proposed framework offers a flexible, quantitative method for assessing LLM behavior, providing valuable insights for the development of more socially aligned AI systems.
>
---
#### [new 017] EAVIT: Efficient and Accurate Human Value Identification from Text data via LLMs
- **分类: cs.CL**

- **简介: 论文提出EAVIT框架，解决大语言模型(LLMs)在长文本价值识别中效率低、成本高的问题。通过本地小模型生成初步价值估计，构建精简提示输入在线LLMs，结合解释性训练和采样策略，将输入标记减少至1/6，在保持准确性的同时超越传统NLP方法和LLM策略。**

- **链接: [http://arxiv.org/pdf/2505.12792v1](http://arxiv.org/pdf/2505.12792v1)**

> **作者:** Wenhao Zhu; Yuhang Xie; Guojie Song; Xin Zhang
>
> **摘要:** The rapid evolution of large language models (LLMs) has revolutionized various fields, including the identification and discovery of human values within text data. While traditional NLP models, such as BERT, have been employed for this task, their ability to represent textual data is significantly outperformed by emerging LLMs like GPTs. However, the performance of online LLMs often degrades when handling long contexts required for value identification, which also incurs substantial computational costs. To address these challenges, we propose EAVIT, an efficient and accurate framework for human value identification that combines the strengths of both locally fine-tunable and online black-box LLMs. Our framework employs a value detector - a small, local language model - to generate initial value estimations. These estimations are then used to construct concise input prompts for online LLMs, enabling accurate final value identification. To train the value detector, we introduce explanation-based training and data generation techniques specifically tailored for value identification, alongside sampling strategies to optimize the brevity of LLM input prompts. Our approach effectively reduces the number of input tokens by up to 1/6 compared to directly querying online LLMs, while consistently outperforming traditional NLP methods and other LLM-based strategies.
>
---
#### [new 018] An Empirical Study of Many-to-Many Summarization with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）的多对多跨语言摘要任务（M2MS），旨在解决LLM在多语言文档生成摘要时的性能评估与优化问题。通过重组8个数据集构建47.8K样本，对比18个LLM在零样本和指令微调下的表现，发现微调后开源模型超越传统模型及GPT-4，但存在事实性错误加剧的挑战。**

- **链接: [http://arxiv.org/pdf/2505.12983v1](http://arxiv.org/pdf/2505.12983v1)**

> **作者:** Jiaan Wang; Fandong Meng; Zengkui Sun; Yunlong Liang; Yuxuan Cao; Jiarong Xu; Haoxiang Shi; Jie Zhou
>
> **备注:** Accepted to ACL 2025 main conference
>
> **摘要:** Many-to-many summarization (M2MS) aims to process documents in any language and generate the corresponding summaries also in any language. Recently, large language models (LLMs) have shown strong multi-lingual abilities, giving them the potential to perform M2MS in real applications. This work presents a systematic empirical study on LLMs' M2MS ability. Specifically, we first reorganize M2MS data based on eight previous domain-specific datasets. The reorganized data contains 47.8K samples spanning five domains and six languages, which could be used to train and evaluate LLMs. Then, we benchmark 18 LLMs in a zero-shot manner and an instruction-tuning manner. Fine-tuned traditional models (e.g., mBART) are also conducted for comparisons. Our experiments reveal that, zero-shot LLMs achieve competitive results with fine-tuned traditional models. After instruct-tuning, open-source LLMs can significantly improve their M2MS ability, and outperform zero-shot LLMs (including GPT-4) in terms of automatic evaluations. In addition, we demonstrate that this task-specific improvement does not sacrifice the LLMs' general task-solving abilities. However, as revealed by our human evaluation, LLMs still face the factuality issue, and the instruction tuning might intensify the issue. Thus, how to control factual errors becomes the key when building LLM summarizers in real applications, and is worth noting in future research.
>
---
#### [new 019] Steering Risk Preferences in Large Language Models by Aligning Behavioral and Neural Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型行为调控任务，旨在解决无需训练即可定向调整模型输出的问题。提出通过对齐行为方法（MCMC）与神经表征来系统识别引导向量，并验证其在控制LLM风险偏好中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11615v1](http://arxiv.org/pdf/2505.11615v1)**

> **作者:** Jian-Qiao Zhu; Haijiang Yan; Thomas L. Griffiths
>
> **摘要:** Changing the behavior of large language models (LLMs) can be as straightforward as editing the Transformer's residual streams using appropriately constructed "steering vectors." These modifications to internal neural activations, a form of representation engineering, offer an effective and targeted means of influencing model behavior without retraining or fine-tuning the model. But how can such steering vectors be systematically identified? We propose a principled approach for uncovering steering vectors by aligning latent representations elicited through behavioral methods (specifically, Markov chain Monte Carlo with LLMs) with their neural counterparts. To evaluate this approach, we focus on extracting latent risk preferences from LLMs and steering their risk-related outputs using the aligned representations as steering vectors. We show that the resulting steering vectors successfully and reliably modulate LLM outputs in line with the targeted behavior.
>
---
#### [new 020] ChartEdit: How Far Are MLLMs From Automating Chart Analysis? Evaluating MLLMs' Capability via Chart Editing
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLMs）的评估任务，旨在解决现有方法在图表编辑中缺乏系统性评估的问题。作者提出ChartEdit基准，包含1,405条真实图表编辑指令，评估10个主流MLLM的代码生成与图表修改能力，发现大模型仅能部分匹配目标，精确编辑能力不足（SOTA得分59.96），小模型表现更差，需进一步优化。**

- **链接: [http://arxiv.org/pdf/2505.11935v1](http://arxiv.org/pdf/2505.11935v1)**

> **作者:** Xuanle Zhao; Xuexin Liu; Haoyue Yang; Xianzhen Luo; Fanhu Zeng; Jianling Li; Qi Shi; Chi Chen
>
> **备注:** Accept by ACL2025 Findings, preprint version
>
> **摘要:** Although multimodal large language models (MLLMs) show promise in generating chart rendering code, chart editing presents a greater challenge. This difficulty stems from its nature as a labor-intensive task for humans that also demands MLLMs to integrate chart understanding, complex reasoning, and precise intent interpretation. While many MLLMs claim such editing capabilities, current assessments typically rely on limited case studies rather than robust evaluation methodologies, highlighting the urgent need for a comprehensive evaluation framework. In this work, we propose ChartEdit, a new high-quality benchmark designed for chart editing tasks. This benchmark comprises $1,405$ diverse editing instructions applied to $233$ real-world charts, with each instruction-chart instance having been manually annotated and validated for accuracy. Utilizing ChartEdit, we evaluate the performance of 10 mainstream MLLMs across two types of experiments, assessing them at both the code and chart levels. The results suggest that large-scale models can generate code to produce images that partially match the reference images. However, their ability to generate accurate edits according to the instructions remains limited. The state-of-the-art (SOTA) model achieves a score of only $59.96$, highlighting significant challenges in precise modification. In contrast, small-scale models, including chart-domain models, struggle both with following editing instructions and generating overall chart images, underscoring the need for further development in this area. Code is available at https://github.com/xxlllz/ChartEdit.
>
---
#### [new 021] Critique-Guided Distillation: Improving Supervised Fine-tuning via Better Distillation
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对监督微-tuning（SFT）中模型仅模仿专家响应却缺乏理解的问题，提出批判引导蒸馏（CGD）框架。通过整合教师模型生成的解释性批判与精炼响应，训练学生模型关联输入、自生成响应与优化结果，解决“模仿什么”和“为什么”的双重学习目标。实验表明CGD在数学推理（AMC23提升17.5%）和语言理解（MMLU-Pro提升6.3%）任务显著优于基线，并缓解了传统批判微调中的格式漂移问题。**

- **链接: [http://arxiv.org/pdf/2505.11628v1](http://arxiv.org/pdf/2505.11628v1)**

> **作者:** Berkcan Kapusuzoglu; Supriyo Chakraborty; Chia-Hsuan Lee; Sambit Sahu
>
> **备注:** Submitted to NeurIPS 2025
>
> **摘要:** Supervised fine-tuning (SFT) using expert demonstrations often suffer from the imitation problem, where the model learns to reproduce the correct responses without \emph{understanding} the underlying rationale. To address this limitation, we propose \textsc{Critique-Guided Distillation (CGD)}, a novel multi-stage framework that integrates teacher model generated \emph{explanatory critiques} and \emph{refined responses} into the SFT process. A student model is then trained to map the triplet of prompt, teacher critique, and its own initial response to the corresponding refined teacher response, thereby learning both \emph{what} to imitate and \emph{why}. Using entropy-based analysis, we show that \textsc{CGD} reduces refinement uncertainty and can be interpreted as a Bayesian posterior update. We perform extensive empirical evaluation of \textsc{CGD}, on variety of benchmark tasks, and demonstrate significant gains on both math (AMC23 +17.5%) and language understanding tasks (MMLU-Pro +6.3%), while successfully mitigating the format drift issues observed in previous critique fine-tuning (CFT) techniques.
>
---
#### [new 022] Suicide Risk Assessment Using Multimodal Speech Features: A Study on the SW1 Challenge Dataset
- **分类: cs.CL; cs.SD; eess.AS; I.2.7; I.5.1**

- **简介: 该论文属于多模态分类任务，旨在通过语音数据评估青少年自杀风险。研究整合自动转录（WhisperX）、语言（RoBERTa）与音频（WavLM）嵌入及手工声学特征，测试三种特征融合策略。加权注意力结合混合正则化在开发集达69%准确率，但测试集性能差距揭示模型泛化难题，需优化嵌入表达与融合机制提升可靠性。**

- **链接: [http://arxiv.org/pdf/2505.13069v1](http://arxiv.org/pdf/2505.13069v1)**

> **作者:** Ambre Marie; Ilias Maoudj; Guillaume Dardenne; Gwenolé Quellec
>
> **备注:** Submitted to the SpeechWellness Challenge at Interspeech 2025; 5 pages, 2 figures, 2 tables
>
> **摘要:** The 1st SpeechWellness Challenge conveys the need for speech-based suicide risk assessment in adolescents. This study investigates a multimodal approach for this challenge, integrating automatic transcription with WhisperX, linguistic embeddings from Chinese RoBERTa, and audio embeddings from WavLM. Additionally, handcrafted acoustic features -- including MFCCs, spectral contrast, and pitch-related statistics -- were incorporated. We explored three fusion strategies: early concatenation, modality-specific processing, and weighted attention with mixup regularization. Results show that weighted attention provided the best generalization, achieving 69% accuracy on the development set, though a performance gap between development and test sets highlights generalization challenges. Our findings, strictly tied to the MINI-KID framework, emphasize the importance of refining embedding representations and fusion mechanisms to enhance classification reliability.
>
---
#### [new 023] GuRE:Generative Query REwriter for Legal Passage Retrieval
- **分类: cs.CL**

- **简介: 该论文针对法律段落检索任务，解决查询与目标段落词汇不匹配问题。提出GuRE方法，利用大语言模型生成改写后的查询以提升检索效果。实验表明该方法能适配不同检索器且性能优于基线，分析显示其训练目标更适用于实际场景。**

- **链接: [http://arxiv.org/pdf/2505.12950v1](http://arxiv.org/pdf/2505.12950v1)**

> **作者:** Daehee Kim; Deokhyung Kang; Jonghwi Kim; Sangwon Ryu; Gary Geunbae Lee
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Legal Passage Retrieval (LPR) systems are crucial as they help practitioners save time when drafting legal arguments. However, it remains an underexplored avenue. One primary reason is the significant vocabulary mismatch between the query and the target passage. To address this, we propose a simple yet effective method, the Generative query REwriter (GuRE). We leverage the generative capabilities of Large Language Models (LLMs) by training the LLM for query rewriting. "Rewritten queries" help retrievers to retrieve target passages by mitigating vocabulary mismatch. Experimental results show that GuRE significantly improves performance in a retriever-agnostic manner, outperforming all baseline methods. Further analysis reveals that different training objectives lead to distinct retrieval behaviors, making GuRE more suitable than direct retriever fine-tuning for real-world applications. Codes are avaiable at github.com/daehuikim/GuRE.
>
---
#### [new 024] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method
- **分类: cs.CL**

- **简介: 该论文属于论证挖掘任务，旨在解决现有方法无法捕捉复杂论证结构的问题。通过提出14种细粒度关系类型，从垂直和水平维度分析论证交互，并在组件检测、关系预测和自动评分任务上实验，探索写作质量与论证分析的联系，强调细粒度标注对评估的重要性。**

- **链接: [http://arxiv.org/pdf/2505.12028v1](http://arxiv.org/pdf/2505.12028v1)**

> **作者:** Yupei Ren; Xinyi Zhou; Ning Zhang; Shangqing Zhao; Man Lan; Xiaopeng Bai
>
> **备注:** Accepted to ACL 2025; 13 pages, 3 figures
>
> **摘要:** Argument mining has garnered increasing attention over the years, with the recent advancement of Large Language Models (LLMs) further propelling this trend. However, current argument relations remain relatively simplistic and foundational, struggling to capture the full scope of argument information, particularly when it comes to representing complex argument structures in real-world scenarios. To address this limitation, we propose 14 fine-grained relation types from both vertical and horizontal dimensions, thereby capturing the intricate interplay between argument components for a thorough understanding of argument structure. On this basis, we conducted extensive experiments on three tasks: argument component detection, relation prediction, and automated essay grading. Additionally, we explored the impact of writing quality on argument component detection and relation prediction, as well as the connections between discourse relations and argumentative features. The findings highlight the importance of fine-grained argumentative annotations for argumentative writing quality assessment and encourage multi-dimensional argument analysis.
>
---
#### [new 025] Truth Neurons
- **分类: cs.CL**

- **简介: 该论文研究语言模型的真实性编码机制，属于模型可解释性任务。为解决模型生成不实内容的问题，作者提出神经元级真实性表征识别方法，发现跨主题通用的"真理神经元"，并通过多尺度模型实验验证其普遍存在性。抑制该类神经元会降低多基准表现，揭示了真实性机制与数据集无关的特性。**

- **链接: [http://arxiv.org/pdf/2505.12182v1](http://arxiv.org/pdf/2505.12182v1)**

> **作者:** Haohang Li; Yupeng Cao; Yangyang Yu; Jordan W. Suchow; Zining Zhu
>
> **摘要:** Despite their remarkable success and deployment across diverse workflows, language models sometimes produce untruthful responses. Our limited understanding of how truthfulness is mechanistically encoded within these models jeopardizes their reliability and safety. In this paper, we propose a method for identifying representations of truthfulness at the neuron level. We show that language models contain truth neurons, which encode truthfulness in a subject-agnostic manner. Experiments conducted across models of varying scales validate the existence of truth neurons, confirming that the encoding of truthfulness at the neuron level is a property shared by many language models. The distribution patterns of truth neurons over layers align with prior findings on the geometry of truthfulness. Selectively suppressing the activations of truth neurons found through the TruthfulQA dataset degrades performance both on TruthfulQA and on other benchmarks, showing that the truthfulness mechanisms are not tied to a specific dataset. Our results offer novel insights into the mechanisms underlying truthfulness in language models and highlight potential directions toward improving their trustworthiness and reliability.
>
---
#### [new 026] Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models
- **分类: cs.CL**

- **简介: 该论文针对低资源语言G2P转换中的同形异义词消歧问题，提出半自动构建数据集HomoRich增强深度学习模型，并改进规则系统eSpeak为HomoFast版本。通过数据驱动与规则结合，在保持实时性的同时提升30%消歧准确率，适用于屏幕阅读器等延迟敏感场景。**

- **链接: [http://arxiv.org/pdf/2505.12973v1](http://arxiv.org/pdf/2505.12973v1)**

> **作者:** Mahta Fetrat Qharabagh; Zahra Dehghanian; Hamid R. Rabiee
>
> **备注:** 8 main body pages, total 25 pages, 15 figures
>
> **摘要:** Homograph disambiguation remains a significant challenge in grapheme-to-phoneme (G2P) conversion, especially for low-resource languages. This challenge is twofold: (1) creating balanced and comprehensive homograph datasets is labor-intensive and costly, and (2) specific disambiguation strategies introduce additional latency, making them unsuitable for real-time applications such as screen readers and other accessibility tools. In this paper, we address both issues. First, we propose a semi-automated pipeline for constructing homograph-focused datasets, introduce the HomoRich dataset generated through this pipeline, and demonstrate its effectiveness by applying it to enhance a state-of-the-art deep learning-based G2P system for Persian. Second, we advocate for a paradigm shift - utilizing rich offline datasets to inform the development of fast, rule-based methods suitable for latency-sensitive accessibility applications like screen readers. To this end, we improve one of the most well-known rule-based G2P systems, eSpeak, into a fast homograph-aware version, HomoFast eSpeak. Our results show an approximate 30% improvement in homograph disambiguation accuracy for the deep learning-based and eSpeak systems.
>
---
#### [new 027] RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文研究大语言模型的思维链推理优化，解决可测与不可测能力边界的量化评估问题。提出RBF++框架：通过组合定律量化可测推理边界，采用常数假设和边界划分机制处理多模态等不可测场景，经38模型跨13任务验证有效性，扩展了推理评估基准。**

- **链接: [http://arxiv.org/pdf/2505.13307v1](http://arxiv.org/pdf/2505.13307v1)**

> **作者:** Qiguang Chen; Libo Qin; Jinhao Liu; Yue Liao; Jiaqi Wang; Jingxuan Zhou; Wanxiang Che
>
> **备注:** Manuscript
>
> **摘要:** Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at https://github.com/LightChen233/reasoning-boundary.
>
---
#### [new 028] Emotion Recognition for Low-Resource Turkish: Fine-Tuning BERTurk on TREMO and Testing on Xenophobic Political Discourse
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决土耳其语（低资源语言）的情绪识别问题。通过微调BERTurk模型并在TREMO数据集训练，构建了准确率92.62%的情绪分类模型，应用于分析土耳其社交媒体仇外政治言论中的情感分布，推进了小语种NLP技术发展及社会情绪研究。**

- **链接: [http://arxiv.org/pdf/2505.12160v1](http://arxiv.org/pdf/2505.12160v1)**

> **作者:** Darmawan Wicaksono; Hasri Akbar Awal Rozaq; Nevfel Boz
>
> **摘要:** Social media platforms like X (formerly Twitter) play a crucial role in shaping public discourse and societal norms. This study examines the term Sessiz Istila (Silent Invasion) on Turkish social media, highlighting the rise of anti-refugee sentiment amidst the Syrian refugee influx. Using BERTurk and the TREMO dataset, we developed an advanced Emotion Recognition Model (ERM) tailored for Turkish, achieving 92.62% accuracy in categorizing emotions such as happiness, fear, anger, sadness, disgust, and surprise. By applying this model to large-scale X data, the study uncovers emotional nuances in Turkish discourse, contributing to computational social science by advancing sentiment analysis in underrepresented languages and enhancing our understanding of global digital discourse and the unique linguistic challenges of Turkish. The findings underscore the transformative potential of localized NLP tools, with our ERM model offering practical applications for real-time sentiment analysis in Turkish-language contexts. By addressing critical areas, including marketing, public relations, and crisis management, these models facilitate improved decision-making through timely and accurate sentiment tracking. This highlights the significance of advancing research that accounts for regional and linguistic nuances.
>
---
#### [new 029] Model Merging in Pre-training of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究预训练大语言模型中的模型合并技术，属于模型优化任务。旨在解决大规模预训练效率低、成本高的问题。通过实验不同架构（密集/MoE）和参数规模（百万至千亿级）模型，验证合并恒定学习率检查点能提升性能、预测退火行为并降低训练成本，为开源社区提供实用指南。**

- **链接: [http://arxiv.org/pdf/2505.12082v1](http://arxiv.org/pdf/2505.12082v1)**

> **作者:** Yunshui Li; Yiyuan Ma; Shen Yan; Chaoyi Zhang; Jing Liu; Jianqiao Lu; Ziwen Xu; Mengzhao Chen; Minrui Wang; Shiyi Zhan; Jin Ma; Xunhao Lai; Yao Luo; Xingyan Bin; Hongbin Ren; Mingji Han; Wenhao Hao; Bairen Yi; LingJun Liu; Bole Ma; Xiaoying Jia; Zhou Xun; Liang Xiang; Yonghui Wu
>
> **摘要:** Model merging has emerged as a promising technique for enhancing large language models, though its application in large-scale pre-training remains relatively unexplored. In this paper, we present a comprehensive investigation of model merging techniques during the pre-training process. Through extensive experiments with both dense and Mixture-of-Experts (MoE) architectures ranging from millions to over 100 billion parameters, we demonstrate that merging checkpoints trained with constant learning rates not only achieves significant performance improvements but also enables accurate prediction of annealing behavior. These improvements lead to both more efficient model development and significantly lower training costs. Our detailed ablation studies on merging strategies and hyperparameters provide new insights into the underlying mechanisms while uncovering novel applications. Through comprehensive experimental analysis, we offer the open-source community practical pre-training guidelines for effective model merging.
>
---
#### [new 030] Representation of perceived prosodic similarity of conversational feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究对话中同词汇反馈（如“yeah”）的感知韵律相似性，属于语音表征对齐任务。旨在解决现有语音模型难以反映人类对韵律相似性判断的问题。通过三元对比实验评估不同声学特征（频谱、自监督学习等）与人类感知的匹配程度，发现频谱特征效果更优，并提出对比学习优化方法提升表征对齐能力。**

- **链接: [http://arxiv.org/pdf/2505.13268v1](http://arxiv.org/pdf/2505.13268v1)**

> **作者:** Livia Qian; Carol Figueroa; Gabriel Skantze
>
> **备注:** Interspeech 2025
>
> **摘要:** Vocal feedback (e.g., `mhm', `yeah', `okay') is an important component of spoken dialogue and is crucial to ensuring common ground in conversational systems. The exact meaning of such feedback is conveyed through both lexical and prosodic form. In this work, we investigate the perceived prosodic similarity of vocal feedback with the same lexical form, and to what extent existing speech representations reflect such similarities. A triadic comparison task with recruited participants is used to measure perceived similarity of feedback responses taken from two different datasets. We find that spectral and self-supervised speech representations encode prosody better than extracted pitch features, especially in the case of feedback from the same speaker. We also find that it is possible to further condense and align the representations to human perception through contrastive learning.
>
---
#### [new 031] NAMET: Robust Massive Model Editing via Noise-Aware Memory Optimization
- **分类: cs.CL**

- **简介: 该论文研究大语言模型高效知识更新任务，解决大规模编辑中嵌入冲突导致的可靠性下降问题。提出NAMET方法，通过在内存提取时引入噪声改进MEMIT，有效缓解知识碰撞。实验验证其在多模型、多数据集下优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11876v1](http://arxiv.org/pdf/2505.11876v1)**

> **作者:** Yanbo Dai; Zhenlan Ji; Zongjie Li; Shuai Wang
>
> **摘要:** Model editing techniques are essential for efficiently updating knowledge in large language models (LLMs). However, the effectiveness of existing approaches degrades in massive editing scenarios, particularly when evaluated with practical metrics or in context-rich settings. We attribute these failures to embedding collisions among knowledge items, which undermine editing reliability at scale. To address this, we propose NAMET (Noise-aware Model Editing in Transformers), a simple yet effective method that introduces noise during memory extraction via a one-line modification to MEMIT. Extensive experiments across six LLMs and three datasets demonstrate that NAMET consistently outperforms existing methods when editing thousands of facts.
>
---
#### [new 032] Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于多模态模型评估任务，旨在解决大型音频语言模型（LALMs）在时间推理能力上的评测不足问题。通过构建TREA数据集对开源LALMs进行基准测试，发现其性能低于人类水平，并提出衡量输入扰动的置信度指标，揭示模型准确性与稳定性无强关联，强调高风险场景需综合评估。**

- **链接: [http://arxiv.org/pdf/2505.13115v1](http://arxiv.org/pdf/2505.13115v1)**

> **作者:** Debarpan Bhattacharya; Apoorva Kulkarni; Sriram Ganapathy
>
> **备注:** Accepted in INTERSPEECH, 2025, Rotterdam, The Netherlands
>
> **摘要:** The popular success of text-based large language models (LLM) has streamlined the attention of the multimodal community to combine other modalities like vision and audio along with text to achieve similar multimodal capabilities. In this quest, large audio language models (LALMs) have to be evaluated on reasoning related tasks which are different from traditional classification or generation tasks. Towards this goal, we propose a novel dataset called temporal reasoning evaluation of audio (TREA). We benchmark open-source LALMs and observe that they are consistently behind human capabilities on the tasks in the TREA dataset. While evaluating LALMs, we also propose an uncertainty metric, which computes the invariance of the model to semantically identical perturbations of the input. Our analysis shows that the accuracy and uncertainty metrics are not necessarily correlated and thus, points to a need for wholesome evaluation of LALMs for high-stakes applications.
>
---
#### [new 033] GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment
- **分类: cs.CL**

- **简介: 该论文提出GMSA框架，解决大语言模型在长文本场景中计算效率低、冗余信息多的问题。通过分组合并提取摘要向量，结合层语义对齐缩小语义差异，并采用知识提取微调优化下游任务。实现了2倍推理加速，性能优于现有方法，属于自然语言处理中的上下文压缩任务。**

- **链接: [http://arxiv.org/pdf/2505.12215v1](http://arxiv.org/pdf/2505.12215v1)**

> **作者:** Jiwei Tang; Zhicheng Zhang; Shunlong Wu; Jingheng Ye; Lichen Bai; Zitai Wang; Tingwei Lu; Jiaqi Chen; Lin Hai; Hai-Tao Zheng; Hong-Gee Kim
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Large language models (LLMs) have achieved impressive performance in a variety of natural language processing (NLP) tasks. However, when applied to long-context scenarios, they face two challenges, i.e., low computational efficiency and much redundant information. This paper introduces GMSA, a context compression framework based on the encoder-decoder architecture, which addresses these challenges by reducing input sequence length and redundant information. Structurally, GMSA has two key components: Group Merging and Layer Semantic Alignment (LSA). Group merging is used to effectively and efficiently extract summary vectors from the original context. Layer semantic alignment, on the other hand, aligns the high-level summary vectors with the low-level primary input semantics, thus bridging the semantic gap between different layers. In the training process, GMSA first learns soft tokens that contain complete semantics through autoencoder training. To furtherly adapt GMSA to downstream tasks, we propose Knowledge Extraction Fine-tuning (KEFT) to extract knowledge from the soft tokens for downstream tasks. We train GMSA by randomly sampling the compression rate for each sample in the dataset. Under this condition, GMSA not only significantly outperforms the traditional compression paradigm in context restoration but also achieves stable and significantly faster convergence with only a few encoder layers. In downstream question-answering (QA) tasks, GMSA can achieve approximately a 2x speedup in end-to-end inference while outperforming both the original input prompts and various state-of-the-art (SOTA) methods by a large margin.
>
---
#### [new 034] Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成（RAG）中生成器利用检索信息能力不足及决策透明度低的问题，提出基于强化学习的ARENA框架，通过自适应奖励机制训练模型识别关键证据并生成可解释推理路径，在问答任务中性能提升显著且无需额外训练。**

- **链接: [http://arxiv.org/pdf/2505.13258v1](http://arxiv.org/pdf/2505.13258v1)**

> **作者:** Jingyi Ren; Yekun Xu; Xiaolong Wang; Weitao Li; Weizhi Ma; Yang Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive domains. However, although RAG achieved successes across distinct domains, there are still some unsolved challenges: 1) Effectiveness. Existing research mainly focuses on developing more powerful RAG retrievers, but how to enhance the generator's (LLM's) ability to utilize the retrieved information for reasoning and generation? 2) Transparency. Most RAG methods ignore which retrieved content actually contributes to the reasoning process, resulting in a lack of interpretability and visibility. To address this, we propose ARENA (Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator framework trained via reinforcement learning (RL) with our proposed rewards. Based on the structured generation and adaptive reward calculation, our RL-based training enables the model to identify key evidence, perform structured reasoning, and generate answers with interpretable decision traces. Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments with various RAG baselines demonstrate that our model achieves 10-30% improvements on all multi-hop QA datasets, which is comparable with the SOTA Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses show that ARENA has strong flexibility to be adopted on new datasets without extra training. Our models and codes are publicly released.
>
---
#### [new 035] Systematic Generalization in Language Models Scales with Information Entropy
- **分类: cs.CL**

- **简介: 该论文研究语言模型的系统性泛化能力，属于自然语言处理领域。针对模型难以处理新语境下的已知概念及评估难题，提出用训练数据成分分布的信息熵量化任务难度，构建序列到序列任务框架验证模型性能与熵的正相关关系，表明高熵无需先验知识即可成功，低熵表现可作为系统性泛化的评估基准。**

- **链接: [http://arxiv.org/pdf/2505.13089v1](http://arxiv.org/pdf/2505.13089v1)**

> **作者:** Sondre Wold; Lucas Georges Gabriel Charpentier; Étienne Simon
>
> **备注:** Accepted to ACL 2025: Findings
>
> **摘要:** Systematic generalization remains challenging for current language models, which are known to be both sensitive to semantically similar permutations of the input and to struggle with known concepts presented in novel contexts. Although benchmarks exist for assessing compositional behavior, it is unclear how to measure the difficulty of a systematic generalization problem. In this work, we show how one aspect of systematic generalization can be described by the entropy of the distribution of component parts in the training data. We formalize a framework for measuring entropy in a sequence-to-sequence task and find that the performance of popular model architectures scales with the entropy. Our work connects systematic generalization to information efficiency, and our results indicate that success at high entropy can be achieved even without built-in priors, and that success at low entropy can serve as a target for assessing progress towards robust systematic generalization.
>
---
#### [new 036] $\textit{Rank, Chunk and Expand}$: Lineage-Oriented Reasoning for Taxonomy Expansion
- **分类: cs.CL**

- **简介: 该论文研究分类法扩展任务，解决现有判别模型泛化差、生成方法噪声多或漏选的问题。提出LORex框架，结合排序分块和生成推理，迭代筛选候选词并优化层次结构，提升准确率12%和语义相似度5%。**

- **链接: [http://arxiv.org/pdf/2505.13282v1](http://arxiv.org/pdf/2505.13282v1)**

> **作者:** Sahil Mishra; Kumar Arjun; Tanmoy Chakraborty
>
> **摘要:** Taxonomies are hierarchical knowledge graphs crucial for recommendation systems, and web applications. As data grows, expanding taxonomies is essential, but existing methods face key challenges: (1) discriminative models struggle with representation limits and generalization, while (2) generative methods either process all candidates at once, introducing noise and exceeding context limits, or discard relevant entities by selecting noisy candidates. We propose LORex ($\textbf{L}$ineage-$\textbf{O}$riented $\textbf{Re}$asoning for Taxonomy E$\textbf{x}$pansion), a plug-and-play framework that combines discriminative ranking and generative reasoning for efficient taxonomy expansion. Unlike prior methods, LORex ranks and chunks candidate terms into batches, filtering noise and iteratively refining selections by reasoning candidates' hierarchy to ensure contextual efficiency. Extensive experiments across four benchmarks and twelve baselines show that LORex improves accuracy by 12% and Wu & Palmer similarity by 5% over state-of-the-art methods.
>
---
#### [new 037] ELITE: Embedding-Less retrieval with Iterative Text Exploration
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成任务中嵌入检索的语义偏差及高开销问题，提出无嵌入框架ELITE，利用大模型逻辑推理能力，通过迭代搜索空间优化和逻辑扩展提升检索效果，在长问答基准上超越基线且大幅降低资源消耗。**

- **链接: [http://arxiv.org/pdf/2505.11908v1](http://arxiv.org/pdf/2505.11908v1)**

> **作者:** Zhangyu Wang; Siyuan Gao; Rong Zhou; Hao Wang; Li Ning
>
> **摘要:** Large Language Models (LLMs) have achieved impressive progress in natural language processing, but their limited ability to retain long-term context constrains performance on document-level or multi-turn tasks. Retrieval-Augmented Generation (RAG) mitigates this by retrieving relevant information from an external corpus. However, existing RAG systems often rely on embedding-based retrieval trained on corpus-level semantic similarity, which can lead to retrieving content that is semantically similar in form but misaligned with the question's true intent. Furthermore, recent RAG variants construct graph- or hierarchy-based structures to improve retrieval accuracy, resulting in significant computation and storage overhead. In this paper, we propose an embedding-free retrieval framework. Our method leverages the logical inferencing ability of LLMs in retrieval using iterative search space refinement guided by our novel importance measure and extend our retrieval results with logically related information without explicit graph construction. Experiments on long-context QA benchmarks, including NovelQA and Marathon, show that our approach outperforms strong baselines while reducing storage and runtime by over an order of magnitude.
>
---
#### [new 038] Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks
- **分类: cs.CL**

- **简介: 该论文研究LLM作为评估器的安全性，属于对抗攻击任务。针对LLM-as-a-Judge系统易受提示注入攻击的问题，提出两种攻击策略（CUA破坏决策、JMA篡改推理），采用GCG优化生成对抗后缀。实验验证开源模型存在显著漏洞（攻击成功率超30%），揭示了现有评估框架的安全风险，呼吁加强防御机制研究。**

- **链接: [http://arxiv.org/pdf/2505.13348v1](http://arxiv.org/pdf/2505.13348v1)**

> **作者:** Narek Maloyan; Bislan Ashinov; Dmitry Namiot
>
> **摘要:** Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.
>
---
#### [new 039] ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决长链式思维（CoT）推理冗长低效、缺乏系统性的问题。提出ToTRL框架，通过强化学习引导LLM将序列式CoT转为并行树状思维（ToT），结合解谜游戏训练模型生成/评估多路径推理，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2505.12717v1](http://arxiv.org/pdf/2505.12717v1)**

> **作者:** Haoyuan Wu; Xueyi Chen; Rui Ming; Jilong Gao; Shoubo Hu; Zhuolun He; Bei Yu
>
> **摘要:** Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes, which can be elicited by reinforcement learning (RL). However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction. In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification, assessment, and pruning of unproductive paths. This process can potentially lead to improved performance and reduced token costs. Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward. ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints, which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL, achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks.
>
---
#### [new 040] Personalized Author Obfuscation with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）在作者身份混淆任务中的效果，旨在通过改写和风格转换隐藏文本作者身份。针对传统方法在数据集整体评估的局限性，聚焦个体用户差异，发现LLM效果呈双峰分布（用户间差异显著），并提出个性化提示方法提升性能并缓解该问题。**

- **链接: [http://arxiv.org/pdf/2505.12090v1](http://arxiv.org/pdf/2505.12090v1)**

> **作者:** Mohammad Shokri; Sarah Ita Levitan; Rivka Levitan
>
> **摘要:** In this paper, we investigate the efficacy of large language models (LLMs) in obfuscating authorship by paraphrasing and altering writing styles. Rather than adopting a holistic approach that evaluates performance across the entire dataset, we focus on user-wise performance to analyze how obfuscation effectiveness varies across individual authors. While LLMs are generally effective, we observe a bimodal distribution of efficacy, with performance varying significantly across users. To address this, we propose a personalized prompting method that outperforms standard prompting techniques and partially mitigates the bimodality issue.
>
---
#### [new 041] One-for-All Pruning: A Universal Model for Customized Compression of Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的高效压缩方法，解决现有剪枝技术处理多请求时效率低的问题。提出UniCuCo模型，通过StratNet学习将任意压缩需求映射为最优剪枝策略，结合高斯过程近似非可微剪枝的梯度，实现并行处理。实验表明其处理速度提升28倍且保持精度。**

- **链接: [http://arxiv.org/pdf/2505.12216v1](http://arxiv.org/pdf/2505.12216v1)**

> **作者:** Rongguang Ye; Ming Tang
>
> **备注:** ACL Findings
>
> **摘要:** Existing pruning methods for large language models (LLMs) focus on achieving high compression rates while maintaining model performance. Although these methods have demonstrated satisfactory performance in handling a single user's compression request, their processing time increases linearly with the number of requests, making them inefficient for real-world scenarios with multiple simultaneous requests. To address this limitation, we propose a Univeral Model for Customized Compression (UniCuCo) for LLMs, which introduces a StratNet that learns to map arbitrary requests to their optimal pruning strategy. The challenge in training StratNet lies in the high computational cost of evaluating pruning strategies and the non-differentiable nature of the pruning process, which hinders gradient backpropagation for StratNet updates. To overcome these challenges, we leverage a Gaussian process to approximate the evaluation process. Since the gradient of the Gaussian process is computable, we can use it to approximate the gradient of the non-differentiable pruning process, thereby enabling StratNet updates. Experimental results show that UniCuCo is 28 times faster than baselines in processing 64 requests, while maintaining comparable accuracy to baselines.
>
---
#### [new 042] Vectors from Larger Language Models Predict Human Reading Time and fMRI Data More Poorly when Dimensionality Expansion is Controlled
- **分类: cs.CL**

- **简介: 该论文属于语言模型与人类认知对齐任务，探讨大语言模型（LLM）预测人类阅读时间和脑成像数据的效果。研究发现，在控制向量维度后，更大LLM的预测能力反而下降，表明模型规模扩大会加剧其与人类语言处理机制的偏差，挑战了"模型越大拟合越好"的假设。**

- **链接: [http://arxiv.org/pdf/2505.12196v1](http://arxiv.org/pdf/2505.12196v1)**

> **作者:** Yi-Chien Lin; Hongao Zhu; William Schuler
>
> **摘要:** The impressive linguistic abilities of large language models (LLMs) have recommended them as models of human sentence processing, with some conjecturing a positive 'quality-power' relationship (Wilcox et al., 2023), in which language models' (LMs') fit to psychometric data continues to improve as their ability to predict words in context increases. This is important because it suggests that elements of LLM architecture, such as veridical attention to context and a unique objective of predicting upcoming words, reflect the architecture of the human sentence processing faculty, and that any inadequacies in predicting human reading time and brain imaging data may be attributed to insufficient model complexity, which recedes as larger models become available. Recent studies (Oh and Schuler, 2023) have shown this scaling inverts after a point, as LMs become excessively large and accurate, when word prediction probability (as information-theoretic surprisal) is used as a predictor. Other studies propose the use of entire vectors from differently sized LLMs, still showing positive scaling (Schrimpf et al., 2021), casting doubt on the value of surprisal as a predictor, but do not control for the larger number of predictors in vectors from larger LMs. This study evaluates LLM scaling using entire LLM vectors, while controlling for the larger number of predictors in vectors from larger LLMs. Results show that inverse scaling obtains, suggesting that inadequacies in predicting human reading time and brain imaging data may be due to substantial misalignment between LLMs and human sentence processing, which worsens as larger models are used.
>
---
#### [new 043] Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对知识图谱问答（KGQA）任务，解决现有方法忽视历史推理路径和复杂语义导致路径不准确的问题。提出无训练框架RTSoG：将复杂问题分解为子问题，通过奖励引导的树搜索迭代检索加权推理路径，并整合生成答案，实验性能显著提升。**

- **链接: [http://arxiv.org/pdf/2505.12476v1](http://arxiv.org/pdf/2505.12476v1)**

> **作者:** Xiao Long; Liansheng Zhuang; Chen Shen; Shaotian Yan; Yifei Li; Shafei Wang
>
> **摘要:** Recently, large language models (LLMs) have demonstrated impressive performance in Knowledge Graph Question Answering (KGQA) tasks, which aim to find answers based on knowledge graphs (KGs) for natural language questions. Existing LLMs-based KGQA methods typically follow the Graph Retrieval-Augmented Generation (GraphRAG) paradigm, which first retrieves reasoning paths from the large KGs, and then generates the answers based on them. However, these methods emphasize the exploration of new optimal reasoning paths in KGs while ignoring the exploitation of historical reasoning paths, which may lead to sub-optimal reasoning paths. Additionally, the complex semantics contained in questions may lead to the retrieval of inaccurate reasoning paths. To address these issues, this paper proposes a novel and training-free framework for KGQA tasks called Reward-guided Tree Search on Graph (RTSoG). RTSoG decomposes an original question into a series of simpler and well-defined sub-questions to handle the complex semantics. Then, a Self-Critic Monte Carlo Tree Search (SC-MCTS) guided by a reward model is introduced to iteratively retrieve weighted reasoning paths as contextual knowledge. Finally, it stacks the weighted reasoning paths according to their weights to generate the final answers. Extensive experiments on four datasets demonstrate the effectiveness of RTSoG. Notably, it achieves 8.7\% and 7.0\% performance improvement over the state-of-the-art method on the GrailQA and the WebQSP respectively.
>
---
#### [new 044] Improving Fairness in LLMs Through Testing-Time Adversaries
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究提升大语言模型（LLMs）的公平性，属于偏见缓解任务。针对LLM预测中存在的种族等群体偏见问题，提出无需训练或调参的测试时对抗方法：通过生成句子变体检测预测不一致性以识别偏差，实验证明在Llama3中最高提升公平性指标27个百分点。**

- **链接: [http://arxiv.org/pdf/2505.12100v1](http://arxiv.org/pdf/2505.12100v1)**

> **作者:** Isabela Pereira Gregio; Ian Pons; Anna Helena Reali Costa; Artur Jordão
>
> **摘要:** Large Language Models (LLMs) push the bound-aries in natural language processing and generative AI, driving progress across various aspects of modern society. Unfortunately, the pervasive issue of bias in LLMs responses (i.e., predictions) poses a significant and open challenge, hindering their application in tasks involving ethical sensitivity and responsible decision-making. In this work, we propose a straightforward, user-friendly and practical method to mitigate such biases, enhancing the reliability and trustworthiness of LLMs. Our method creates multiple variations of a given sentence by modifying specific attributes and evaluates the corresponding prediction behavior compared to the original, unaltered, prediction/sentence. The idea behind this process is that critical ethical predictions often exhibit notable inconsistencies, indicating the presence of bias. Unlike previous approaches, our method relies solely on forward passes (i.e., testing-time adversaries), eliminating the need for training, fine-tuning, or prior knowledge of the training data distribution. Through extensive experiments on the popular Llama family, we demonstrate the effectiveness of our method in improving various fairness metrics, focusing on the reduction of disparities in how the model treats individuals from different racial groups. Specifically, using standard metrics, we improve the fairness in Llama3 in up to 27 percentage points. Overall, our approach significantly enhances fairness, equity, and reliability in LLM-generated results without parameter tuning or training data modifications, confirming its effectiveness in practical scenarios. We believe our work establishes an important step toward enabling the use of LLMs in tasks that require ethical considerations and responsible decision-making.
>
---
#### [new 045] ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言机器翻译任务，旨在解决现有大模型在低资源语言翻译效果不佳及强化学习奖励建模不充分的问题。提出基于强模型的对比奖励方法，通过轻量化设计将单语能力扩展到11种语言的90个翻译方向，实现了SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.12996v1](http://arxiv.org/pdf/2505.12996v1)**

> **作者:** Jiaan Wang; Fandong Meng; Jie Zhou
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** In recent years, the emergence of large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, has shown impressive capabilities in complex problems, e.g., mathematics and coding. Some pioneering studies attempt to bring the success of LRMs in neural machine translation (MT). They try to build LRMs with deep reasoning MT ability via reinforcement learning (RL). Despite some progress that has been made, these attempts generally focus on several high-resource languages, e.g., English and Chinese, leaving the performance on other languages unclear. Besides, the reward modeling methods in previous work do not fully unleash the potential of reinforcement learning in MT. In this work, we first design a new reward modeling method that compares the translation results of the policy MT model with a strong LRM (i.e., DeepSeek-R1-671B), and quantifies the comparisons to provide rewards. Experimental results demonstrate the superiority of the reward modeling method. Using Qwen2.5-7B-Instruct as the backbone, the trained model achieves the new state-of-the-art performance in literary translation, and outperforms strong LRMs including OpenAI-o1 and DeepSeeK-R1. Furthermore, we extend our method to the multilingual settings with 11 languages. With a carefully designed lightweight reward modeling in RL, we can simply transfer the strong MT ability from a single direction into multiple (i.e., 90) translation directions and achieve impressive multilingual MT performance.
>
---
#### [new 046] Table-R1: Region-based Reinforcement Learning for Table Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对表格问答任务，解决语言模型（LLM）因表格结构化特征导致推理效率低的问题，提出区域强化学习方法Table-R1。通过区域增强微调（RE-SFT）识别关键区域，结合混合奖励机制TARPO平衡区域精度与答案正确性，显著提升模型性能并减少67.5%推理资源消耗。**

- **链接: [http://arxiv.org/pdf/2505.12415v1](http://arxiv.org/pdf/2505.12415v1)**

> **作者:** Zhenhe Wu; Jian Yang; Jiaheng Liu; Xianjie Wu; Changzai Pan; Jie Zhang; Yu Zhao; Shuangyong Song; Yongxiang Li; Zhoujun Li
>
> **摘要:** Tables present unique challenges for language models due to their structured row-column interactions, necessitating specialized approaches for effective comprehension. While large language models (LLMs) have demonstrated potential in table reasoning through prompting and techniques like chain-of-thought (CoT) and program-of-thought (PoT), optimizing their performance for table question answering remains underexplored. In this paper, we introduce region-based Table-R1, a novel reinforcement learning approach that enhances LLM table understanding by integrating region evidence into reasoning steps. Our method employs Region-Enhanced Supervised Fine-Tuning (RE-SFT) to guide models in identifying relevant table regions before generating answers, incorporating textual, symbolic, and program-based reasoning. Additionally, Table-Aware Group Relative Policy Optimization (TARPO) introduces a mixed reward system to dynamically balance region accuracy and answer correctness, with decaying region rewards and consistency penalties to align reasoning steps. Experiments show that Table-R1 achieves an average performance improvement of 14.36 points across multiple base models on three benchmark datasets, even outperforming baseline models with ten times the parameters, while TARPO reduces response token consumption by 67.5% compared to GRPO, significantly advancing LLM capabilities in efficient tabular reasoning.
>
---
#### [new 047] Efficiently Building a Domain-Specific Large Language Model from Scratch: A Case Study of a Classical Chinese Large Language Model
- **分类: cs.CL**

- **简介: 该论文研究领域特定大语言模型构建，针对通用模型处理古典中文效果不佳的问题，提出从零构建高效模型的方法。通过模型设计、数据处理与训练优化，开发1.8B参数的AI Taiyan模型，在古文标点、典故识别等任务上超越通用模型及传统方法，接近人类水平，为专业领域LLM建设提供参考。**

- **链接: [http://arxiv.org/pdf/2505.11810v1](http://arxiv.org/pdf/2505.11810v1)**

> **作者:** Shen Li; Renfen Hu; Lijun Wang
>
> **摘要:** General-purpose large language models demonstrate notable capabilities in language comprehension and generation, achieving results that are comparable to, or even surpass, human performance in many language information processing tasks. Nevertheless, when general models are applied to some specific domains, e.g., Classical Chinese texts, their effectiveness is often unsatisfactory, and fine-tuning open-source foundational models similarly struggles to adequately incorporate domain-specific knowledge. To address this challenge, this study developed a large language model, AI Taiyan, specifically designed for understanding and generating Classical Chinese. Experiments show that with a reasonable model design, data processing, foundational training, and fine-tuning, satisfactory results can be achieved with only 1.8 billion parameters. In key tasks related to Classical Chinese information processing such as punctuation, identification of allusions, explanation of word meanings, and translation between ancient and modern Chinese, this model exhibits a clear advantage over both general-purpose large models and domain-specific traditional models, achieving levels close to or surpassing human baselines. This research provides a reference for the efficient construction of specialized domain-specific large language models. Furthermore, the paper discusses the application of this model in fields such as the collation of ancient texts, dictionary editing, and language research, combined with case studies.
>
---
#### [new 048] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation
- **分类: cs.CL**

- **简介: 该论文研究开放域长文本生成中的幻觉检测任务，旨在解决现有方法依赖外部工具或领域受限的问题。通过分析发现模型内部状态不足以可靠区分事实与幻觉，提出RATE-FT方法：在微调主任务时加入辅助任务联合学习，实验证明其优于常规微调（如LongFact提升3%），增强了检测准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.12265v1](http://arxiv.org/pdf/2505.12265v1)**

> **作者:** Chengwei Qin; Wenxuan Zhou; Karthik Abinav Sankararaman; Nanshu Wang; Tengyu Xu; Alexander Radovic; Eryk Helenowski; Arya Talebzadeh; Aditya Tayade; Sinong Wang; Shafiq Joty; Han Fang; Hao Ma
>
> **摘要:** Hallucination, the generation of factually incorrect information, remains a significant challenge for large language models (LLMs), especially in open-domain long-form generation. Existing approaches for detecting hallucination in long-form tasks either focus on limited domains or rely heavily on external fact-checking tools, which may not always be available. In this work, we systematically investigate reference-free hallucination detection in open-domain long-form responses. Our findings reveal that internal states (e.g., model's output probability and entropy) alone are insufficient for reliably (i.e., better than random guessing) distinguishing between factual and hallucinated content. To enhance detection, we explore various existing approaches, including prompting-based methods, probing, and fine-tuning, with fine-tuning proving the most effective. To further improve the accuracy, we introduce a new paradigm, named RATE-FT, that augments fine-tuning with an auxiliary task for the model to jointly learn with the main task of hallucination detection. With extensive experiments and analysis using a variety of model families & datasets, we demonstrate the effectiveness and generalizability of our method, e.g., +3% over general fine-tuning methods on LongFact.
>
---
#### [new 049] When AI Co-Scientists Fail: SPOT-a Benchmark for Automated Verification of Scientific Research
- **分类: cs.CL**

- **简介: 该论文研究利用大语言模型（LLM）自动验证科学论文错误的任务，提出SPOT基准数据集（含83篇论文及91个真实错误），评估发现现有模型召回率最高仅21.1%，错误检测不可靠且存在学生级误解，揭示LLM与可信学术验证间的能力差距。**

- **链接: [http://arxiv.org/pdf/2505.11855v1](http://arxiv.org/pdf/2505.11855v1)**

> **作者:** Guijin Son; Jiwoo Hong; Honglu Fan; Heejeong Nam; Hyunwoo Ko; Seungwon Lim; Jinyeop Song; Jinha Choi; Gonçalo Paulo; Youngjae Yu; Stella Biderman
>
> **备注:** work in progress
>
> **摘要:** Recent advances in large language models (LLMs) have fueled the vision of automated scientific discovery, often called AI Co-Scientists. To date, prior work casts these systems as generative co-authors responsible for crafting hypotheses, synthesizing code, or drafting manuscripts. In this work, we explore a complementary application: using LLMs as verifiers to automate the \textbf{academic verification of scientific manuscripts}. To that end, we introduce SPOT, a dataset of 83 published papers paired with 91 errors significant enough to prompt errata or retraction, cross-validated with actual authors and human annotators. Evaluating state-of-the-art LLMs on SPOT, we find that none surpasses 21.1\% recall or 6.1\% precision (o3 achieves the best scores, with all others near zero). Furthermore, confidence estimates are uniformly low, and across eight independent runs, models rarely rediscover the same errors, undermining their reliability. Finally, qualitative analysis with domain experts reveals that even the strongest models make mistakes resembling student-level misconceptions derived from misunderstandings. These findings highlight the substantial gap between current LLM capabilities and the requirements for dependable AI-assisted academic verification.
>
---
#### [new 050] Extracting memorized pieces of (copyrighted) books from open-weight language models
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 论文研究生成式AI模型对受版权书籍的记忆程度及其法律影响，属于模型记忆性分析任务。通过对抗技术从13个开放权重LLMs中提取Books3内容，发现模型间记忆差异显著：大模型通常未记忆多数书籍，但Llama3.1 70B几乎完整记忆《哈利波特》等书，证明参数内存在复制文本，结果为版权诉讼提供复杂证据，不单边支持原被告主张。**

- **链接: [http://arxiv.org/pdf/2505.12546v1](http://arxiv.org/pdf/2505.12546v1)**

> **作者:** A. Feder Cooper; Aaron Gokaslan; Amy B. Cyphert; Christopher De Sa; Mark A. Lemley; Daniel E. Ho; Percy Liang
>
> **摘要:** Plaintiffs and defendants in copyright lawsuits over generative AI often make sweeping, opposing claims about the extent to which large language models (LLMs) have memorized plaintiffs' protected expression. Drawing on adversarial ML and copyright law, we show that these polarized positions dramatically oversimplify the relationship between memorization and copyright. To do so, we leverage a recent probabilistic extraction technique to extract pieces of the Books3 dataset from 13 open-weight LLMs. Through numerous experiments, we show that it's possible to extract substantial parts of at least some books from different LLMs. This is evidence that the LLMs have memorized the extracted text; this memorized content is copied inside the model parameters. But the results are complicated: the extent of memorization varies both by model and by book. With our specific experiments, we find that the largest LLMs don't memorize most books -- either in whole or in part. However, we also find that Llama 3.1 70B memorizes some books, like Harry Potter and 1984, almost entirely. We discuss why our results have significant implications for copyright cases, though not ones that unambiguously favor either side.
>
---
#### [new 051] AutoMedEval: Harnessing Language Models for Automatic Medical Capability Evaluation
- **分类: cs.CL**

- **简介: 该论文属于医疗领域大语言模型（LLM）自动评估任务，旨在解决传统指标忽视医学术语、人工评估成本高的问题。提出开源模型AutoMedEval（13B参数），通过分层训练和知识内省机制，用少量数据实现专业医学问答能力评估，降低人工依赖，实验显示其与人类评估相关性优于基线。**

- **链接: [http://arxiv.org/pdf/2505.11887v1](http://arxiv.org/pdf/2505.11887v1)**

> **作者:** Xiechi Zhang; Zetian Ouyang; Linlin Wang; Gerard de Melo; Zhu Cao; Xiaoling Wang; Ya Zhang; Yanfeng Wang; Liang He
>
> **摘要:** With the proliferation of large language models (LLMs) in the medical domain, there is increasing demand for improved evaluation techniques to assess their capabilities. However, traditional metrics like F1 and ROUGE, which rely on token overlaps to measure quality, significantly overlook the importance of medical terminology. While human evaluation tends to be more reliable, it can be very costly and may as well suffer from inaccuracies due to limits in human expertise and motivation. Although there are some evaluation methods based on LLMs, their usability in the medical field is limited due to their proprietary nature or lack of expertise. To tackle these challenges, we present AutoMedEval, an open-sourced automatic evaluation model with 13B parameters specifically engineered to measure the question-answering proficiency of medical LLMs. The overarching objective of AutoMedEval is to assess the quality of responses produced by diverse models, aspiring to significantly reduce the dependence on human evaluation. Specifically, we propose a hierarchical training method involving curriculum instruction tuning and an iterative knowledge introspection mechanism, enabling AutoMedEval to acquire professional medical assessment capabilities with limited instructional data. Human evaluations indicate that AutoMedEval surpasses other baselines in terms of correlation with human judgments.
>
---
#### [new 052] CCNU at SemEval-2025 Task 3: Leveraging Internal and External Knowledge of Large Language Models for Multilingual Hallucination Annotation
- **分类: cs.CL**

- **简介: 该论文参与SemEval-2025多语言幻觉标注任务，旨在检测14种语言问答系统的虚假信息。研究通过融合多专家大模型并行标注，结合内外知识，使用DeepSeek-V3模型实现多语言性能领先（如印地语第一），并分析了失败方法与经验。**

- **链接: [http://arxiv.org/pdf/2505.11965v1](http://arxiv.org/pdf/2505.11965v1)**

> **作者:** Xu Liu; Guanyi Chen
>
> **备注:** SemEval-2025 Task 3
>
> **摘要:** We present the system developed by the Central China Normal University (CCNU) team for the Mu-SHROOM shared task, which focuses on identifying hallucinations in question-answering systems across 14 different languages. Our approach leverages multiple Large Language Models (LLMs) with distinct areas of expertise, employing them in parallel to annotate hallucinations, effectively simulating a crowdsourcing annotation process. Furthermore, each LLM-based annotator integrates both internal and external knowledge related to the input during the annotation process. Using the open-source LLM DeepSeek-V3, our system achieves the top ranking (\#1) for Hindi data and secures a Top-5 position in seven other languages. In this paper, we also discuss unsuccessful approaches explored during our development process and share key insights gained from participating in this shared task.
>
---
#### [new 053] A Data Synthesis Method Driven by Large Language Models for Proactive Mining of Implicit User Intentions in Tourism
- **分类: cs.CL**

- **简介: 该论文属于旅游领域意图挖掘任务，旨在解决LLMs难从模糊查询中主动挖掘隐含意图的问题。提出SynPT方法：构建用户/助手双代理模拟对话，生成带显式推理的训练数据集SynPT-Dialog，并微调LLM实现主动意图挖掘，解决了领域适配、数据分布偏差、冗余及情感缺失等现有缺陷。**

- **链接: [http://arxiv.org/pdf/2505.11533v1](http://arxiv.org/pdf/2505.11533v1)**

> **作者:** Jinqiang Wang; Huansheng Ning; Tao Zhu; Jianguo Ding
>
> **摘要:** In the tourism domain, Large Language Models (LLMs) often struggle to mine implicit user intentions from tourists' ambiguous inquiries and lack the capacity to proactively guide users toward clarifying their needs. A critical bottleneck is the scarcity of high-quality training datasets that facilitate proactive questioning and implicit intention mining. While recent advances leverage LLM-driven data synthesis to generate such datasets and transfer specialized knowledge to downstream models, existing approaches suffer from several shortcomings: (1) lack of adaptation to the tourism domain, (2) skewed distributions of detail levels in initial inquiries, (3) contextual redundancy in the implicit intention mining module, and (4) lack of explicit thinking about tourists' emotions and intention values. Therefore, we propose SynPT (A Data Synthesis Method Driven by LLMs for Proactive Mining of Implicit User Intentions in the Tourism), which constructs an LLM-driven user agent and assistant agent to simulate dialogues based on seed data collected from Chinese tourism websites. This approach addresses the aforementioned limitations and generates SynPT-Dialog, a training dataset containing explicit reasoning. The dataset is utilized to fine-tune a general LLM, enabling it to proactively mine implicit user intentions. Experimental evaluations, conducted from both human and LLM perspectives, demonstrate the superiority of SynPT compared to existing methods. Furthermore, we analyze key hyperparameters and present case studies to illustrate the practical applicability of our method, including discussions on its adaptability to English-language scenarios. All code and data are publicly available.
>
---
#### [new 054] Alignment-Augmented Speculative Decoding with Alignment Sampling and Conditional Verification
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，解决现有推测解码方法依赖训练导致高成本的问题。提出无需训练的推测解码算法：通过预填充阶段分布进行对齐采样生成候选，并设计自适应阈值验证策略提升效率与准确性，实验显示在多个任务中显著加速并提高生成质量。**

- **链接: [http://arxiv.org/pdf/2505.13204v1](http://arxiv.org/pdf/2505.13204v1)**

> **作者:** Jikai Wang; Zhenxu Tian; Juntao Li; Qingrong Xia; Xinyu Duan; Zhefeng Wang; Baoxing Huai; Min Zhang
>
> **备注:** Pre-print
>
> **摘要:** Recent works have revealed the great potential of speculative decoding in accelerating the autoregressive generation process of large language models. The success of these methods relies on the alignment between draft candidates and the sampled outputs of the target model. Existing methods mainly achieve draft-target alignment with training-based methods, e.g., EAGLE, Medusa, involving considerable training costs. In this paper, we present a training-free alignment-augmented speculative decoding algorithm. We propose alignment sampling, which leverages output distribution obtained in the prefilling phase to provide more aligned draft candidates. To further benefit from high-quality but non-aligned draft candidates, we also introduce a simple yet effective flexible verification strategy. Through an adaptive probability threshold, our approach can improve generation accuracy while further improving inference efficiency. Experiments on 8 datasets (including question answering, summarization and code completion tasks) show that our approach increases the average generation score by 3.3 points for the LLaMA3 model. Our method achieves a mean acceptance length up to 2.39 and speed up generation by 2.23.
>
---
#### [new 055] ABoN: Adaptive Best-of-N Alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型对齐任务，旨在解决传统Best-of-N采样方法计算成本高且未考虑提示差异的问题。提出ABoN方法：通过两阶段算法（探索奖励分布+动态分配预算）优化推理资源分配，实现在相同计算成本下性能优于均匀采样，并随批量增大提升效率。**

- **链接: [http://arxiv.org/pdf/2505.12050v1](http://arxiv.org/pdf/2505.12050v1)**

> **作者:** Vinod Raman; Hilal Asi; Satyen Kale
>
> **备注:** 23 pages
>
> **摘要:** Recent advances in test-time alignment methods, such as Best-of-N sampling, offer a simple and effective way to steer language models (LMs) toward preferred behaviors using reward models (RM). However, these approaches can be computationally expensive, especially when applied uniformly across prompts without accounting for differences in alignment difficulty. In this work, we propose a prompt-adaptive strategy for Best-of-N alignment that allocates inference-time compute more efficiently. Motivated by latency concerns, we develop a two-stage algorithm: an initial exploratory phase estimates the reward distribution for each prompt using a small exploration budget, and a second stage adaptively allocates the remaining budget using these estimates. Our method is simple, practical, and compatible with any LM/RM combination. Empirical results on the AlpacaEval dataset for 12 LM/RM pairs and 50 different batches of prompts show that our adaptive strategy consistently outperforms the uniform allocation with the same inference budget. Moreover, our experiments show that our adaptive strategy remains competitive against uniform allocations with 20% larger inference budgets and even improves in performance as the batch size grows.
>
---
#### [new 056] Enhancing Complex Instruction Following for Large Language Models with Mixture-of-Contexts Fine-tuning
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLMs）遵循含多重约束的复杂指令时效果不稳定的问题，提出混合上下文微调方法MISO。通过将顺序指令拆解为并行子上下文，并改进模型架构以联合优化整体指令对齐和子上下文影响，增强监督微调效果。实验证明该方法在复杂指令遵循任务中更有效且高效。**

- **链接: [http://arxiv.org/pdf/2505.11922v1](http://arxiv.org/pdf/2505.11922v1)**

> **作者:** Yuheng Lu; ZiMeng Bai; Caixia Yuan; Huixing Jiang; Xiaojie Wang
>
> **摘要:** Large language models (LLMs) exhibit remarkable capabilities in handling natural language tasks; however, they may struggle to consistently follow complex instructions including those involve multiple constraints. Post-training LLMs using supervised fine-tuning (SFT) is a standard approach to improve their ability to follow instructions. In addressing complex instruction following, existing efforts primarily focus on data-driven methods that synthesize complex instruction-output pairs for SFT. However, insufficient attention allocated to crucial sub-contexts may reduce the effectiveness of SFT. In this work, we propose transforming sequentially structured input instruction into multiple parallel instructions containing subcontexts. To support processing this multi-input, we propose MISO (Multi-Input Single-Output), an extension to currently dominant decoder-only transformer-based LLMs. MISO introduces a mixture-of-contexts paradigm that jointly considers the overall instruction-output alignment and the influence of individual sub-contexts to enhance SFT effectiveness. We apply MISO fine-tuning to complex instructionfollowing datasets and evaluate it with standard LLM inference. Empirical results demonstrate the superiority of MISO as a fine-tuning method for LLMs, both in terms of effectiveness in complex instruction-following scenarios and its potential for training efficiency.
>
---
#### [new 057] Can an Easy-to-Hard Curriculum Make Reasoning Emerge in Small Language Models? Evidence from a Four-Stage Curriculum on GPT-2
- **分类: cs.CL**

- **简介: 该论文研究分阶段课程学习能否提升小语言模型的推理能力。通过GPT-2四阶段渐进训练（词汇匹配→符号推理），证明有序课程能加速收敛、激活深层推理模块并优化注意力机制，但最终准确率仍低于传统训练30%，揭示了混合微调和探测扩展的改进方向。**

- **链接: [http://arxiv.org/pdf/2505.11643v1](http://arxiv.org/pdf/2505.11643v1)**

> **作者:** Xiang Fu
>
> **摘要:** We demonstrate that a developmentally ordered curriculum markedly improves reasoning transparency and sample-efficiency in small language models (SLMs). Concretely, we train Cognivolve, a 124 M-parameter GPT-2 model, on a four-stage syllabus that ascends from lexical matching to multi-step symbolic inference and then evaluate it without any task-specific fine-tuning. Cognivolve reaches target accuracy in half the optimization steps of a single-phase baseline, activates an order-of-magnitude more gradient-salient reasoning heads, and shifts those heads toward deeper layers, yielding higher-entropy attention that balances local and long-range context. The same curriculum applied out of order or with optimizer resets fails to reproduce these gains, confirming that progression--not extra compute--drives the effect. We also identify open challenges: final-answer success still lags a conventional run by about 30%, and our saliency probe under-detects verbal-knowledge heads in the hardest stage, suggesting directions for mixed-stage fine-tuning and probe expansion.
>
---
#### [new 058] The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究闭源大语言模型的多语言对抗攻击（安全任务），解决模型在跨语言越狱攻击中的脆弱性问题。通过集成攻击框架评估GPT-4o等模型，发现中文提示攻击成功率更高，Qwen-Max最易受击，提出需加强多语言防御机制。**

- **链接: [http://arxiv.org/pdf/2505.12287v1](http://arxiv.org/pdf/2505.12287v1)**

> **作者:** Linghan Huang; Haolin Jin; Zhaoge Bi; Pengyue Yang; Peizhou Zhao; Taozhao Chen; Xiongfei Wu; Lei Ma; Huaming Chen
>
> **摘要:** Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems.
>
---
#### [new 059] Neural Morphological Tagging for Nguni Languages
- **分类: cs.CL**

- **简介: 该论文研究神经形态标注任务，针对南非黏着型恩古尼语系，解决传统规则方法在复杂语素解析中的性能瓶颈。通过对比LSTM/CRF序列标注模型、微调预训练模型与规则基线，发现神经模型显著优于传统方法，且从零训练模型表现更佳。验证了基于现有形态切分器的神经标注器可行性。**

- **链接: [http://arxiv.org/pdf/2505.12949v1](http://arxiv.org/pdf/2505.12949v1)**

> **作者:** Cael Marquard; Simbarashe Mawere; Francois Meyer
>
> **摘要:** Morphological parsing is the task of decomposing words into morphemes, the smallest units of meaning in a language, and labelling their grammatical roles. It is a particularly challenging task for agglutinative languages, such as the Nguni languages of South Africa, which construct words by concatenating multiple morphemes. A morphological parsing system can be framed as a pipeline with two separate components, a segmenter followed by a tagger. This paper investigates the use of neural methods to build morphological taggers for the four Nguni languages. We compare two classes of approaches: training neural sequence labellers (LSTMs and neural CRFs) from scratch and finetuning pretrained language models. We compare performance across these two categories, as well as to a traditional rule-based morphological parser. Neural taggers comfortably outperform the rule-based baseline and models trained from scratch tend to outperform pretrained models. We also compare parsing results across different upstream segmenters and with varying linguistic input features. Our findings confirm the viability of employing neural taggers based on pre-existing morphological segmenters for the Nguni languages.
>
---
#### [new 060] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型编辑任务，旨在解决现有编辑数据集覆盖窄、评估单一的问题。提出UniEdit基准，通过开放域知识图谱构建25个领域的编辑样本，设计NMCS算法量化知识修改的连锁效应，并生成多样化自然语言测试集，全面评估模型编辑效果。**

- **链接: [http://arxiv.org/pdf/2505.12345v1](http://arxiv.org/pdf/2505.12345v1)**

> **作者:** Qizhou Chen; Dakan Wang; Taolin Zhang; Zaoming Yan; Chengsong You; Chengyu Wang; Xiaofeng He
>
> **摘要:** Model editing aims to enhance the accuracy and reliability of large language models (LLMs) by efficiently adjusting their internal parameters. Currently, most LLM editing datasets are confined to narrow knowledge domains and cover a limited range of editing evaluation. They often overlook the broad scope of editing demands and the diversity of ripple effects resulting from edits. In this context, we introduce UniEdit, a unified benchmark for LLM editing grounded in open-domain knowledge. First, we construct editing samples by selecting entities from 25 common domains across five major categories, utilizing the extensive triple knowledge available in open-domain knowledge graphs to ensure comprehensive coverage of the knowledge domains. To address the issues of generality and locality in editing, we design an Neighborhood Multi-hop Chain Sampling (NMCS) algorithm to sample subgraphs based on a given knowledge piece to entail comprehensive ripple effects to evaluate. Finally, we employ proprietary LLMs to convert the sampled knowledge subgraphs into natural language text, guaranteeing grammatical accuracy and syntactical diversity. Extensive statistical analysis confirms the scale, comprehensiveness, and diversity of our UniEdit benchmark. We conduct comprehensive experiments across multiple LLMs and editors, analyzing their performance to highlight strengths and weaknesses in editing across open knowledge domains and various evaluation criteria, thereby offering valuable insights for future research endeavors.
>
---
#### [new 061] MoL for LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving General Capabilities
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLMs）在领域适应任务中因数据偏置导致通用能力退化及语料配比失衡问题，提出双损失优化框架MoL：用交叉熵损失强化领域知识，KL散度保持通用能力。通过解耦训练目标，实现领域性能提升与基础能力保留，实验验证1:1语料比例最优，在数学推理等任务中显著超越传统方法。**

- **链接: [http://arxiv.org/pdf/2505.12043v1](http://arxiv.org/pdf/2505.12043v1)**

> **作者:** Jingxue Chen; Qingkun Tang; Qianchun Lu; Siyuan Fang
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Although LLMs perform well in general tasks, domain-specific applications suffer from hallucinations and accuracy limitations. CPT approaches encounter two key issues: (1) domain-biased data degrades general language skills, and (2) improper corpus-mixture ratios limit effective adaptation. To address these, we propose a novel framework, Mixture of Losses (MoL), which decouples optimization objectives for domain-specific and general corpora. Specifically, cross-entropy (CE) loss is applied to domain data to ensure knowledge acquisition, while Kullback-Leibler (KL) divergence aligns general-corpus training with the base model's foundational capabilities. This dual-loss architecture preserves universal skills while enhancing domain expertise, avoiding catastrophic forgetting. Empirically, we validate that a 1:1 domain-to-general corpus ratio optimally balances training and overfitting without the need for extensive tuning or resource-intensive experiments. Furthermore, our experiments demonstrate significant performance gains compared to traditional CPT approaches, which often suffer from degradation in general language capabilities; our model achieves 27.9% higher accuracy on the Math-500 benchmark in the non-think reasoning mode, and an impressive 83.3% improvement on the challenging AIME25 subset in the think mode, underscoring the effectiveness of our approach.
>
---
#### [new 062] J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动评估大语言模型输出的研究，旨在解决现有评判模型在复杂推理任务中表现不足的问题。通过提出EIS-GRPO算法消除位置偏差、构建ReasoningJudgeBench基准，并训练出7B规模的J4R模型，其在推理评估任务中性能超越GPT-4o等模型。**

- **链接: [http://arxiv.org/pdf/2505.13346v1](http://arxiv.org/pdf/2505.13346v1)**

> **作者:** Austin Xu; Yilun Zhou; Xuan-Phi Nguyen; Caiming Xiong; Shafiq Joty
>
> **备注:** 25 pages, 4 figures, 6 tables. To be updated with links for code/benchmark
>
> **摘要:** To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench.
>
---
#### [new 063] The Effect of Language Diversity When Fine-Tuning Large Language Models for Translation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型微调中语言多样性对翻译任务的影响，属于机器翻译领域。针对先前研究结论矛盾的问题，通过132个翻译方向的实验发现：适度增加语言多样性可提升监督/无监督对的翻译质量（因促进语言无关表示），但存在收益阈值。研究揭示了多样性对模型表征的影响机制。**

- **链接: [http://arxiv.org/pdf/2505.13090v1](http://arxiv.org/pdf/2505.13090v1)**

> **作者:** David Stap; Christof Monz
>
> **摘要:** Prior research diverges on language diversity in LLM fine-tuning: Some studies report benefits while others find no advantages. Through controlled fine-tuning experiments across 132 translation directions, we systematically resolve these disparities. We find that expanding language diversity during fine-tuning improves translation quality for both unsupervised and -- surprisingly -- supervised pairs, despite less diverse models being fine-tuned exclusively on these supervised pairs. However, benefits plateau or decrease beyond a certain diversity threshold. We show that increased language diversity creates more language-agnostic representations. These representational adaptations help explain the improved performance in models fine-tuned with greater diversity.
>
---
#### [new 064] Wisdom from Diversity: Bias Mitigation Through Hybrid Human-LLM Crowds
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文研究大语言模型（LLMs）的偏见缓解任务，针对模型易继承训练数据偏见的缺陷，提出群体聚合策略。通过分析发现：多LLM简单聚合会加剧偏见，而局部加权聚合能有效降低偏见并提升准确性。最终结合人类（多样性）与LLM（准确性）构建混合群体，显著提升性能并减少种族/性别相关偏见。**

- **链接: [http://arxiv.org/pdf/2505.12349v1](http://arxiv.org/pdf/2505.12349v1)**

> **作者:** Axel Abels; Tom Lenaerts
>
> **备注:** Accepted for publication in the Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** Despite their performance, large language models (LLMs) can inadvertently perpetuate biases found in the data they are trained on. By analyzing LLM responses to bias-eliciting headlines, we find that these models often mirror human biases. To address this, we explore crowd-based strategies for mitigating bias through response aggregation. We first demonstrate that simply averaging responses from multiple LLMs, intended to leverage the "wisdom of the crowd", can exacerbate existing biases due to the limited diversity within LLM crowds. In contrast, we show that locally weighted aggregation methods more effectively leverage the wisdom of the LLM crowd, achieving both bias mitigation and improved accuracy. Finally, recognizing the complementary strengths of LLMs (accuracy) and humans (diversity), we demonstrate that hybrid crowds containing both significantly enhance performance and further reduce biases across ethnic and gender-related contexts.
>
---
#### [new 065] Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对小样本关系抽取（FSRE）任务，解决数据稀缺和模型泛化不足问题。提出TKRE框架，结合生成式与判别式学习：利用大模型生成知识解释和合成数据缓解数据不足，设计两阶段预训练（掩码建模+对比学习）增强关系推理能力，在基准测试中实现最优性能。**

- **链接: [http://arxiv.org/pdf/2505.12236v1](http://arxiv.org/pdf/2505.12236v1)**

> **作者:** Quanjiang Guo; Jinchuan Zhang; Sijie Wang; Ling Tian; Zhao Kang; Bin Yan; Weidong Xiao
>
> **备注:** 13 pages, 6 figures, Appear on IJCAI 2025
>
> **摘要:** Few-Shot Relation Extraction (FSRE) remains a challenging task due to the scarcity of annotated data and the limited generalization capabilities of existing models. Although large language models (LLMs) have demonstrated potential in FSRE through in-context learning (ICL), their general-purpose training objectives often result in suboptimal performance for task-specific relation extraction. To overcome these challenges, we propose TKRE (Two-Stage Knowledge-Guided Pre-training for Relation Extraction), a novel framework that synergistically integrates LLMs with traditional relation extraction models, bridging generative and discriminative learning paradigms. TKRE introduces two key innovations: (1) leveraging LLMs to generate explanation-driven knowledge and schema-constrained synthetic data, addressing the issue of data scarcity; and (2) a two-stage pre-training strategy combining Masked Span Language Modeling (MSLM) and Span-Level Contrastive Learning (SCL) to enhance relational reasoning and generalization. Together, these components enable TKRE to effectively tackle FSRE tasks. Comprehensive experiments on benchmark datasets demonstrate the efficacy of TKRE, achieving new state-of-the-art performance in FSRE and underscoring its potential for broader application in low-resource scenarios. \footnote{The code and data are released on https://github.com/UESTC-GQJ/TKRE.
>
---
#### [new 066] Ambiguity Resolution in Text-to-Structured Data Mapping
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文研究自然语言到结构化数据映射中的歧义问题，属于NLP歧义消解任务。针对LLMs在文本转工具调用等场景因歧义导致性能下降的问题，提出通过潜在空间表征差异检测歧义：设计基于稀疏自编码器梯度路径核的新距离度量，识别由概念缺失引发的歧义模式，并构建框架预测缺失概念以提升工具调用准确性。**

- **链接: [http://arxiv.org/pdf/2505.11679v1](http://arxiv.org/pdf/2505.11679v1)**

> **作者:** Zhibo Hu; Chen Wang; Yanfeng Shu; Hye-Young Paik; Liming Zhu
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Ambiguity in natural language is a significant obstacle for achieving accurate text to structured data mapping through large language models (LLMs), which affects the performance of tasks such as mapping text to agentic tool calling and text-to-SQL queries. Existing methods of ambiguity handling either exploit ReACT framework to produce the correct mapping through trial and error, or supervised fine tuning to guide models to produce a biased mapping to improve certain tasks. In this paper, we adopt a different approach that characterizes the representation difference of ambiguous text in the latent space and leverage the difference to identify ambiguity before mapping them to structured data. To detect ambiguity of a sentence, we focused on the relationship between ambiguous questions and their interpretations and what cause the LLM ignore multiple interpretations. Different to the distance calculated by dense embedding vectors, we utilize the observation that ambiguity is caused by concept missing in latent space of LLM to design a new distance measurement, computed through the path kernel by the integral of gradient values for each concepts from sparse-autoencoder (SAE) under each state. We identify patterns to distinguish ambiguous questions with this measurement. Based on our observation, We propose a new framework to improve the performance of LLMs on ambiguous agentic tool calling through missing concepts prediction.
>
---
#### [new 067] ModernGBERT: German-only 1B Encoder Model Trained from Scratch
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于德语自然语言理解任务，旨在解决资源受限场景下高性能德语编码器不足的问题。研究者提出了ModernGBERT（从头训练的透明德语编码器）和LL"aMmlein2Vec（解码器转换的编码器），通过多任务基准测试对比两类模型性能，证明ModernGBERT 1B在参数效率与性能上超越现有方法，并开源了所有资源。**

- **链接: [http://arxiv.org/pdf/2505.13136v1](http://arxiv.org/pdf/2505.13136v1)**

> **作者:** Anton Ehrmanntraut; Julia Wunderle; Jan Pfister; Fotis Jannidis; Andreas Hotho
>
> **备注:** under review @ARR
>
> **摘要:** Despite the prominence of decoder-only language models, encoders remain crucial for resource-constrained applications. We introduce ModernGBERT (134M, 1B), a fully transparent family of German encoder models trained from scratch, incorporating architectural innovations from ModernBERT. To evaluate the practical trade-offs of training encoders from scratch, we also present LL\"aMmlein2Vec (120M, 1B, 7B), a family of encoders derived from German decoder-only models via LLM2Vec. We benchmark all models on natural language understanding, text embedding, and long-context reasoning tasks, enabling a controlled comparison between dedicated encoders and converted decoders. Our results show that ModernGBERT 1B outperforms prior state-of-the-art German encoders as well as encoders adapted via LLM2Vec, with regard to performance and parameter-efficiency. All models, training data, checkpoints and code are publicly available, advancing the German NLP ecosystem with transparent, high-performance encoder models.
>
---
#### [new 068] Class Distillation with Mahalanobis Contrast: An Efficient Training Paradigm for Pragmatic Language Understanding Tasks
- **分类: cs.CL**

- **简介: 该论文针对语用语言理解任务（如检测性别歧视、隐喻、讽刺），提出高效训练框架ClaD，解决传统分类器计算成本高、数据需求大的问题。通过马氏距离构建类分布结构损失函数和可解释决策算法，实现从异构背景中蒸馏小目标类。实验表明小模型即可媲美大模型性能。**

- **链接: [http://arxiv.org/pdf/2505.11829v1](http://arxiv.org/pdf/2505.11829v1)**

> **作者:** Chenlu Wang; Weimin Lyu; Ritwik Banerjee
>
> **摘要:** Detecting deviant language such as sexism, or nuanced language such as metaphors or sarcasm, is crucial for enhancing the safety, clarity, and interpretation of online social discourse. While existing classifiers deliver strong results on these tasks, they often come with significant computational cost and high data demands. In this work, we propose \textbf{Cla}ss \textbf{D}istillation (ClaD), a novel training paradigm that targets the core challenge: distilling a small, well-defined target class from a highly diverse and heterogeneous background. ClaD integrates two key innovations: (i) a loss function informed by the structural properties of class distributions, based on Mahalanobis distance, and (ii) an interpretable decision algorithm optimized for class separation. Across three benchmark detection tasks -- sexism, metaphor, and sarcasm -- ClaD outperforms competitive baselines, and even with smaller language models and orders of magnitude fewer parameters, achieves performance comparable to several large language models (LLMs). These results demonstrate ClaD as an efficient tool for pragmatic language understanding tasks that require gleaning a small target class from a larger heterogeneous background.
>
---
#### [new 069] Token Masking Improves Transformer-Based Text Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究基于Transformer的文本分类任务，旨在提升模型性能并减少过拟合。提出一种token masking正则化方法，随机以概率p将输入替换为[MASK]，通过训练扰动促进梯度平均和深层依赖学习。实验在语言识别、情感分析中验证了效果，确定p=0.1为通用最优，归因于抗过拟合和隐式集成。**

- **链接: [http://arxiv.org/pdf/2505.11746v1](http://arxiv.org/pdf/2505.11746v1)**

> **作者:** Xianglong Xu; John Bowen; Rojin Taheri
>
> **摘要:** While transformer-based models achieve strong performance on text classification, we explore whether masking input tokens can further enhance their effectiveness. We propose token masking regularization, a simple yet theoretically motivated method that randomly replaces input tokens with a special [MASK] token at probability p. This introduces stochastic perturbations during training, leading to implicit gradient averaging that encourages the model to capture deeper inter-token dependencies. Experiments on language identification and sentiment analysis -- across diverse models (mBERT, Qwen2.5-0.5B, TinyLlama-1.1B) -- show consistent improvements over standard regularization techniques. We identify task-specific optimal masking rates, with p = 0.1 as a strong general default. We attribute the gains to two key effects: (1) input perturbation reduces overfitting, and (2) gradient-level smoothing acts as implicit ensembling.
>
---
#### [new 070] ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL
- **分类: cs.CL**

- **简介: 该论文针对Text-to-SQL任务，解决现有方法无法在SQL生成过程中实时利用数据库执行反馈的问题。提出ReEx-SQL框架，通过执行感知强化学习，在解码时动态结合数据库交互和中间执行结果，采用树状解码策略和复合奖励机制优化推理路径，显著提升了生成准确率和推理效率。**

- **链接: [http://arxiv.org/pdf/2505.12768v1](http://arxiv.org/pdf/2505.12768v1)**

> **作者:** Yaxun Dai; Wenxuan Xie; Xialie Zhuang; Tianyu Yang; Yiying Yang; Haiqin Yang; Yuhang Zhao; Pingfu Chao; Wenhao Jiang
>
> **摘要:** In Text-to-SQL, execution feedback is essential for guiding large language models (LLMs) to reason accurately and generate reliable SQL queries. However, existing methods treat execution feedback solely as a post-hoc signal for correction or selection, failing to integrate it into the generation process. This limitation hinders their ability to address reasoning errors as they occur, ultimately reducing query accuracy and robustness. To address this issue, we propose ReEx-SQL (Reasoning with Execution-Aware Reinforcement Learning), a framework for Text-to-SQL that enables models to interact with the database during decoding and dynamically adjust their reasoning based on execution feedback. ReEx-SQL introduces an execution-aware reasoning paradigm that interleaves intermediate SQL execution into reasoning paths, facilitating context-sensitive revisions. It achieves this through structured prompts with markup tags and a stepwise rollout strategy that integrates execution feedback into each stage of generation. To supervise policy learning, we develop a composite reward function that includes an exploration reward, explicitly encouraging effective database interaction. Additionally, ReEx-SQL adopts a tree-based decoding strategy to support exploratory reasoning, enabling dynamic expansion of alternative reasoning paths. Notably, ReEx-SQL achieves 88.8% on Spider and 64.9% on BIRD at the 7B scale, surpassing the standard reasoning baseline by 2.7% and 2.6%, respectively. It also shows robustness, achieving 85.2% on Spider-Realistic with leading performance. In addition, its tree-structured decoding improves efficiency and performance over linear decoding, reducing inference time by 51.9% on the BIRD development set.
>
---
#### [new 071] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering
- **分类: cs.CL**

- **简介: 该论文针对多跳问答任务，解决现有方法忽视问题类型与方法适配性的问题。通过分析问题类型并评估不同方法，提出BELLE框架，利用双层多智能体辩论机制动态组合不同LLM提示方法，提升推理效果和成本效益。**

- **链接: [http://arxiv.org/pdf/2505.11811v1](http://arxiv.org/pdf/2505.11811v1)**

> **作者:** Taolin Zhang; Dongyang Li; Qizhou Chen; Chengyu Wang; Xiaofeng He
>
> **备注:** Accepted by ACL2025 main track
>
> **摘要:** Multi-hop question answering (QA) involves finding multiple relevant passages and performing step-by-step reasoning to answer complex questions. Previous works on multi-hop QA employ specific methods from different modeling perspectives based on large language models (LLMs), regardless of the question types. In this paper, we first conduct an in-depth analysis of public multi-hop QA benchmarks, dividing the questions into four types and evaluating five types of cutting-edge methods for multi-hop QA: Chain-of-Thought (CoT), Single-step, Iterative-step, Sub-step, and Adaptive-step. We find that different types of multi-hop questions have varying degrees of sensitivity to different types of methods. Thus, we propose a Bi-levEL muLti-agEnt reasoning (BELLE) framework to address multi-hop QA by specifically focusing on the correspondence between question types and methods, where each type of method is regarded as an ''operator'' by prompting LLMs differently. The first level of BELLE includes multiple agents that debate to obtain an executive plan of combined ''operators'' to address the multi-hop QA task comprehensively. During the debate, in addition to the basic roles of affirmative debater, negative debater, and judge, at the second level, we further leverage fast and slow debaters to monitor whether changes in viewpoints are reasonable. Extensive experiments demonstrate that BELLE significantly outperforms strong baselines in various datasets. Additionally, the model consumption of BELLE is higher cost-effectiveness than that of single models in more complex multi-hop QA scenarios.
>
---
#### [new 072] From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery
- **分类: cs.CL**

- **简介: 该论文为综述，探讨大语言模型（LLMs）在科学发现中从工具到自主体的角色演变，提出“工具-分析-科学家”三级分类法，分析其自主性提升对科研流程的影响，并针对自动化、伦理治理等挑战提出未来研究方向，旨在构建AI驱动科学发现的框架。**

- **链接: [http://arxiv.org/pdf/2505.13259v1](http://arxiv.org/pdf/2505.13259v1)**

> **作者:** Tianshi Zheng; Zheye Deng; Hong Ting Tsang; Weiqi Wang; Jiaxin Bai; Zihao Wang; Yangqiu Song
>
> **备注:** 16 pages
>
> **摘要:** Large Language Models (LLMs) are catalyzing a paradigm shift in scientific discovery, evolving from task-specific automation tools into increasingly autonomous agents and fundamentally redefining research processes and human-AI collaboration. This survey systematically charts this burgeoning field, placing a central focus on the changing roles and escalating capabilities of LLMs in science. Through the lens of the scientific method, we introduce a foundational three-level taxonomy-Tool, Analyst, and Scientist-to delineate their escalating autonomy and evolving responsibilities within the research lifecycle. We further identify pivotal challenges and future research trajectories such as robotic automation, self-improvement, and ethical governance. Overall, this survey provides a conceptual architecture and strategic foresight to navigate and shape the future of AI-driven scientific discovery, fostering both rapid innovation and responsible advancement. Github Repository: https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery.
>
---
#### [new 073] The Hidden Structure -- Improving Legal Document Understanding Through Explicit Text Formatting
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究法律文档结构对LLM理解的影响，属于法律问答任务，旨在通过文本格式化和提示工程提升模型性能。实验比较GPT-4o和GPT-4.1在不同输入结构（原始文本、OCR提取、Markdown等）下的表现，发现结构优化和提示调整可使GPT-4.1准确率提升达33%，证实输入设计与提示策略对法律应用的重要性。**

- **链接: [http://arxiv.org/pdf/2505.12837v1](http://arxiv.org/pdf/2505.12837v1)**

> **作者:** Christian Braun; Alexander Lilienbeck; Daniel Mentjukov
>
> **备注:** 20 pages, 3 figures
>
> **摘要:** Legal contracts possess an inherent, semantically vital structure (e.g., sections, clauses) that is crucial for human comprehension but whose impact on LLM processing remains under-explored. This paper investigates the effects of explicit input text structure and prompt engineering on the performance of GPT-4o and GPT-4.1 on a legal question-answering task using an excerpt of the CUAD. We compare model exact-match accuracy across various input formats: well-structured plain-text (human-generated from CUAD), plain-text cleaned of line breaks, extracted plain-text from Azure OCR, plain-text extracted by GPT-4o Vision, and extracted (and interpreted) Markdown (MD) from GPT-4o Vision. To give an indication of the impact of possible prompt engineering, we assess the impact of shifting task instructions to the system prompt and explicitly informing the model about the structured nature of the input. Our findings reveal that GPT-4o demonstrates considerable robustness to variations in input structure, but lacks in overall performance. Conversely, GPT-4.1's performance is markedly sensitive; poorly structured inputs yield suboptimal results (but identical with GPT-4o), while well-structured formats (original CUAD text, GPT-4o Vision text and GPT-4o MD) improve exact-match accuracy by ~20 percentage points. Optimizing the system prompt to include task details and an advisory about structured input further elevates GPT-4.1's accuracy by an additional ~10-13 percentage points, with Markdown ultimately achieving the highest performance under these conditions (79 percentage points overall exact-match accuracy). This research empirically demonstrates that while newer models exhibit greater resilience, careful input structuring and strategic prompt design remain critical for optimizing the performance of LLMs, and can significantly affect outcomes in high-stakes legal applications.
>
---
#### [new 074] Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出基于社会心理学中的隐藏档案范式，构建评估多智能体大语言模型集体推理能力的基准，解决现有系统缺乏理论化评估工具的问题。通过设计非对称信息分布的九项任务测试模型，发现多智能体系统整合信息效率低于单智能体，并揭示合作与矛盾对群体决策的影响差异。**

- **链接: [http://arxiv.org/pdf/2505.11556v1](http://arxiv.org/pdf/2505.11556v1)**

> **作者:** Yuxuan Li; Aoi Naito; Hirokazu Shirado
>
> **摘要:** Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but also risk replicating collective reasoning failures observed in human groups. Yet, no theory-grounded benchmark exists to systematically evaluate such failures. In this paper, we introduce the Hidden Profile paradigm from social psychology as a diagnostic testbed for multi-agent LLM systems. By distributing critical information asymmetrically across agents, the paradigm reveals how inter-agent dynamics support or hinder collective reasoning. We first formalize the paradigm for multi-agent decision-making under distributed knowledge and instantiate it as a benchmark with nine tasks spanning diverse scenarios, including adaptations from prior human studies. We then conduct experiments with GPT-4.1 and five other leading LLMs, including reasoning-enhanced variants, showing that multi-agent systems across all models fail to match the accuracy of single agents given complete information. While agents' collective performance is broadly comparable to that of human groups, nuanced behavioral differences emerge, such as increased sensitivity to social desirability. Finally, we demonstrate the paradigm's diagnostic utility by exploring a cooperation-contradiction trade-off in multi-agent LLM systems. We find that while cooperative agents are prone to over-coordination in collective settings, increased contradiction impairs group convergence. This work contributes a reproducible framework for evaluating multi-agent LLM systems and motivates future research on artificial collective intelligence and human-AI interaction.
>
---
#### [new 075] Examining Linguistic Shifts in Academic Writing Before and After the Launch of ChatGPT: A Study on Preprint Papers
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文属于语言模型影响分析任务，研究ChatGPT等LLMs对学术写作语言特征的影响。通过分析arXiv十年间82万篇摘要，发现LLM使用导致偏好词增加、词汇复杂度提升但句法简化，同时连贯性和可读性下降，非英语母语学者及计算机科学领域变化最显著。**

- **链接: [http://arxiv.org/pdf/2505.12218v1](http://arxiv.org/pdf/2505.12218v1)**

> **作者:** Tong Bao; Yi Zhao; Jin Mao; Chengzhi Zhang
>
> **摘要:** Large Language Models (LLMs), such as ChatGPT, have prompted academic concerns about their impact on academic writing. Existing studies have primarily examined LLM usage in academic writing through quantitative approaches, such as word frequency statistics and probability-based analyses. However, few have systematically examined the potential impact of LLMs on the linguistic characteristics of academic writing. To address this gap, we conducted a large-scale analysis across 823,798 abstracts published in last decade from arXiv dataset. Through the linguistic analysis of features such as the frequency of LLM-preferred words, lexical complexity, syntactic complexity, cohesion, readability and sentiment, the results indicate a significant increase in the proportion of LLM-preferred words in abstracts, revealing the widespread influence of LLMs on academic writing. Additionally, we observed an increase in lexical complexity and sentiment in the abstracts, but a decrease in syntactic complexity, suggesting that LLMs introduce more new vocabulary and simplify sentence structure. However, the significant decrease in cohesion and readability indicates that abstracts have fewer connecting words and are becoming more difficult to read. Moreover, our analysis reveals that scholars with weaker English proficiency were more likely to use the LLMs for academic writing, and focused on improving the overall logic and fluency of the abstracts. Finally, at discipline level, we found that scholars in Computer Science showed more pronounced changes in writing style, while the changes in Mathematics were minimal.
>
---
#### [new 076] Automated Bias Assessment in AI-Generated Educational Content Using CEAT Framework
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于AI偏见检测任务，旨在解决教育内容中AI生成文本的隐性偏见（如性别、种族）评估问题。研究提出结合上下文嵌入关联测试与检索增强生成框架的自动化评估方法，通过对比实验验证其与人工标注的高一致性（r=0.993），提升检测效率与客观性。**

- **链接: [http://arxiv.org/pdf/2505.12718v1](http://arxiv.org/pdf/2505.12718v1)**

> **作者:** Jingyang Peng; Wenyuan Shen; Jiarui Rao; Jionghao Lin
>
> **备注:** Accepted by AIED 2025: Late-Breaking Results (LBR) Track
>
> **摘要:** Recent advances in Generative Artificial Intelligence (GenAI) have transformed educational content creation, particularly in developing tutor training materials. However, biases embedded in AI-generated content--such as gender, racial, or national stereotypes--raise significant ethical and educational concerns. Despite the growing use of GenAI, systematic methods for detecting and evaluating such biases in educational materials remain limited. This study proposes an automated bias assessment approach that integrates the Contextualized Embedding Association Test with a prompt-engineered word extraction method within a Retrieval-Augmented Generation framework. We applied this method to AI-generated texts used in tutor training lessons. Results show a high alignment between the automated and manually curated word sets, with a Pearson correlation coefficient of r = 0.993, indicating reliable and consistent bias assessment. Our method reduces human subjectivity and enhances fairness, scalability, and reproducibility in auditing GenAI-produced educational content.
>
---
#### [new 077] PSC: Extending Context Window of Large Language Models via Phase Shift Calibration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型上下文窗口扩展任务，针对基于旋转位置编码(RoPE)的现有方法难以预设最优频率缩放因子问题，提出相位校准模块PSC。通过动态调整预定义频率增强PI/YaRN等方法，实验证明其在16k-64k上下文窗口下有效降低困惑度并提升通用性。**

- **链接: [http://arxiv.org/pdf/2505.12423v1](http://arxiv.org/pdf/2505.12423v1)**

> **作者:** Wenqiao Zhu; Chao Xu; Lulu Wang; Jun Wu
>
> **摘要:** Rotary Position Embedding (RoPE) is an efficient position encoding approach and is widely utilized in numerous large language models (LLMs). Recently, a lot of methods have been put forward to further expand the context window based on RoPE. The core concept of those methods is to predefine or search for a set of factors to rescale the base frequencies of RoPE. Nevertheless, it is quite a challenge for existing methods to predefine an optimal factor due to the exponential search space. In view of this, we introduce PSC (Phase Shift Calibration), a small module for calibrating the frequencies predefined by existing methods. With the employment of PSC, we demonstrate that many existing methods can be further enhanced, like PI, YaRN, and LongRoPE. We conducted extensive experiments across multiple models and tasks. The results demonstrate that (1) when PSC is enabled, the comparative reductions in perplexity increase as the context window size is varied from 16k, to 32k, and up to 64k. (2) Our approach is broadly applicable and exhibits robustness across a variety of models and tasks. The code can be found at https://github.com/WNQzhu/PSC.
>
---
#### [new 078] Improving Multilingual Language Models by Aligning Representations through Steering
- **分类: cs.CL**

- **简介: 该论文属于多语言模型优化任务，旨在解决大语言模型（LLMs）处理非英语词汇时表征不匹配的问题。通过提出“表示引导”方法，在单层激活中叠加学习向量，显著提升模型性能，效果媲美翻译基线并超越现有提示优化技术。同时揭示监督微调（SFT）和强化学习（RLHF）通过调整表征空间增强多语言能力，与所提方法协同优化模型。**

- **链接: [http://arxiv.org/pdf/2505.12584v1](http://arxiv.org/pdf/2505.12584v1)**

> **作者:** Omar Mahmoud; Buddhika Laknath Semage; Thommen George Karimpanal; Santu Rana
>
> **摘要:** In this paper, we investigate how large language models (LLMS) process non-English tokens within their layer representations, an open question despite significant advancements in the field. Using representation steering, specifically by adding a learned vector to a single model layer's activations, we demonstrate that steering a single model layer can notably enhance performance. Our analysis shows that this approach achieves results comparable to translation baselines and surpasses state of the art prompt optimization methods. Additionally, we highlight how advanced techniques like supervised fine tuning (\textsc{sft}) and reinforcement learning from human feedback (\textsc{rlhf}) improve multilingual capabilities by altering representation spaces. We further illustrate how these methods align with our approach to reshaping LLMS layer representations.
>
---
#### [new 079] How Reliable is Multilingual LLM-as-a-Judge?
- **分类: cs.CL**

- **简介: 该论文研究多语言LLM作为评判者的可靠性，属于模型评估任务。针对现有方法在多语言评估中结果不一致的问题，分析了5个模型在25种语言下的表现，发现低资源语言和模型规模等因素影响评判稳定性，并提出集成策略提升一致性。**

- **链接: [http://arxiv.org/pdf/2505.12201v1](http://arxiv.org/pdf/2505.12201v1)**

> **作者:** Xiyan Fu; Wei Liu
>
> **摘要:** LLM-as-a-Judge has emerged as a popular evaluation strategy, where advanced large language models assess generation results in alignment with human instructions. While these models serve as a promising alternative to human annotators, their reliability in multilingual evaluation remains uncertain. To bridge this gap, we conduct a comprehensive analysis of multilingual LLM-as-a-Judge. Specifically, we evaluate five models from different model families across five diverse tasks involving 25 languages. Our findings reveal that LLMs struggle to achieve consistent judgment results across languages, with an average Fleiss' Kappa of approximately 0.3, and some models performing even worse. To investigate the cause of inconsistency, we analyze various influencing factors. We observe that consistency varies significantly across languages, with particularly poor performance in low-resource languages. Additionally, we find that neither training on multilingual data nor increasing model scale directly improves judgment consistency. These findings suggest that LLMs are not yet reliable for evaluating multilingual predictions. We finally propose an ensemble strategy which improves the consistency of the multilingual judge in real-world applications.
>
---
#### [new 080] Know3-RAG: A Knowledge-aware RAG Framework with Adaptive Retrieval, Generation, and Filtering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对检索增强生成（RAG）中自适应控制不可靠和引用导致幻觉的问题，提出Know3-RAG框架。通过知识图谱嵌入优化检索必要性判断、增强实体查询生成、过滤语义对齐的引用，提升开放域QA任务的事实可靠性，减少生成错误。**

- **链接: [http://arxiv.org/pdf/2505.12662v1](http://arxiv.org/pdf/2505.12662v1)**

> **作者:** Xukai Liu; Ye Liu; Shiwen Wu; Yanghai Zhang; Yihao Yuan; Kai Zhang; Qi Liu
>
> **摘要:** Recent advances in large language models (LLMs) have led to impressive progress in natural language generation, yet their tendency to produce hallucinated or unsubstantiated content remains a critical concern. To improve factual reliability, Retrieval-Augmented Generation (RAG) integrates external knowledge during inference. However, existing RAG systems face two major limitations: (1) unreliable adaptive control due to limited external knowledge supervision, and (2) hallucinations caused by inaccurate or irrelevant references. To address these issues, we propose Know3-RAG, a knowledge-aware RAG framework that leverages structured knowledge from knowledge graphs (KGs) to guide three core stages of the RAG process, including retrieval, generation, and filtering. Specifically, we introduce a knowledge-aware adaptive retrieval module that employs KG embedding to assess the confidence of the generated answer and determine retrieval necessity, a knowledge-enhanced reference generation strategy that enriches queries with KG-derived entities to improve generated reference relevance, and a knowledge-driven reference filtering mechanism that ensures semantic alignment and factual accuracy of references. Experiments on multiple open-domain QA benchmarks demonstrate that Know3-RAG consistently outperforms strong baselines, significantly reducing hallucinations and enhancing answer reliability.
>
---
#### [new 081] AdaptThink: Reasoning Models Can Learn When to Think
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自适应决策任务，旨在解决推理模型因过度思考导致的效率低下问题。提出AdaptThink强化学习算法，通过约束优化和重要性采样策略动态选择思考模式（思考/跳过），在降低响应长度53%的同时提升准确率2.4%，优化推理效率与性能的平衡。**

- **链接: [http://arxiv.org/pdf/2505.13417v1](http://arxiv.org/pdf/2505.13417v1)**

> **作者:** Jiajie Zhang; Nianyi Lin; Lei Hou; Ling Feng; Juanzi Li
>
> **摘要:** Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at https://github.com/THU-KEG/AdaptThink.
>
---
#### [new 082] Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究复杂问答任务，解决用户设备上异构个人数据（如文本、表格）的轻量级查询问题。提出ReQAP方法，通过递归分解问题生成可执行运算符树，整合多源数据实现可追溯回答，并发布涵盖真实场景的PerQA评测基准。**

- **链接: [http://arxiv.org/pdf/2505.11900v1](http://arxiv.org/pdf/2505.11900v1)**

> **作者:** Philipp Christmann; Gerhard Weikum
>
> **备注:** Accepted at ACL 2025 (Findings)
>
> **摘要:** Question answering over mixed sources, like text and tables, has been advanced by verbalizing all contents and encoding it with a language model. A prominent case of such heterogeneous data is personal information: user devices log vast amounts of data every day, such as calendar entries, workout statistics, shopping records, streaming history, and more. Information needs range from simple look-ups to queries of analytical nature. The challenge is to provide humans with convenient access with small footprint, so that all personal data stays on the user devices. We present ReQAP, a novel method that creates an executable operator tree for a given question, via recursive decomposition. Operators are designed to enable seamless integration of structured and unstructured sources, and the execution of the operator tree yields a traceable answer. We further release the PerQA benchmark, with persona-based data and questions, covering a diverse spectrum of realistic user needs.
>
---
#### [new 083] Picturized and Recited with Dialects: A Multimodal Chinese Representation Framework for Sentiment Analysis of Classical Chinese Poetry
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态情感分析任务，旨在解决古典诗歌情感分析中忽视韵律和视觉特征的问题。通过融合多方言音频特征、生成视觉特征及大模型增强的文本特征，构建多模态对比表示框架，在公开数据集上实现准确率与F1值提升，为中文多模态表征提供新方法。**

- **链接: [http://arxiv.org/pdf/2505.13210v1](http://arxiv.org/pdf/2505.13210v1)**

> **作者:** Xiaocong Du; Haoyu Pei; Haipeng Zhang
>
> **摘要:** Classical Chinese poetry is a vital and enduring part of Chinese literature, conveying profound emotional resonance. Existing studies analyze sentiment based on textual meanings, overlooking the unique rhythmic and visual features inherent in poetry,especially since it is often recited and accompanied by Chinese paintings. In this work, we propose a dialect-enhanced multimodal framework for classical Chinese poetry sentiment analysis. We extract sentence-level audio features from the poetry and incorporate audio from multiple dialects,which may retain regional ancient Chinese phonetic features, enriching the phonetic representation. Additionally, we generate sentence-level visual features, and the multimodal features are fused with textual features enhanced by LLM translation through multimodal contrastive representation learning. Our framework outperforms state-of-the-art methods on two public datasets, achieving at least 2.51% improvement in accuracy and 1.63% in macro F1. We open-source the code to facilitate research in this area and provide insights for general multimodal Chinese representation.
>
---
#### [new 084] Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment
- **分类: cs.CL; cs.CY; cs.DL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型(LLMs)内部知识利用率低、技术概念理解不足的问题。通过自提问机制激活模型潜在知识，构建130万计算机专利对基准测试，验证自生成问题结合外部检索能显著提升技术判断能力，并发现小模型生成基础问题辅助中等模型的跨模型协作策略。**

- **链接: [http://arxiv.org/pdf/2505.12452v1](http://arxiv.org/pdf/2505.12452v1)**

> **作者:** Siyang Wu; Honglin Bao; Nadav Kunievsky; James A. Evans
>
> **备注:** We commit to fully open-source our patent dataset
>
> **摘要:** Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models with better understanding do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized.
>
---
#### [new 085] topicwizard -- a Modern, Model-agnostic Framework for Topic Model Visualization and Interpretation
- **分类: cs.CL**

- **简介: 该论文提出topicwizard框架，属于自然语言处理中的主题模型可视化任务，旨在解决传统基于词列表的模型解释方法存在偏差且依赖特定模型的问题。通过开发模型无关的交互工具，直观展示文档、词汇与主题间的语义关联，提升用户对复杂主题模型参数的理解能力。**

- **链接: [http://arxiv.org/pdf/2505.13034v1](http://arxiv.org/pdf/2505.13034v1)**

> **作者:** Márton Kardos; Kenneth C. Enevoldsen; Kristoffer Laigaard Nielbo
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Topic models are statistical tools that allow their users to gain qualitative and quantitative insights into the contents of textual corpora without the need for close reading. They can be applied in a wide range of settings from discourse analysis, through pretraining data curation, to text filtering. Topic models are typically parameter-rich, complex models, and interpreting these parameters can be challenging for their users. It is typical practice for users to interpret topics based on the top 10 highest ranking terms on a given topic. This list-of-words approach, however, gives users a limited and biased picture of the content of topics. Thoughtful user interface design and visualizations can help users gain a more complete and accurate understanding of topic models' output. While some visualization utilities do exist for topic models, these are typically limited to a certain type of topic model. We introduce topicwizard, a framework for model-agnostic topic model interpretation, that provides intuitive and interactive tools that help users examine the complex semantic relations between documents, words and topics learned by topic models.
>
---
#### [new 086] Disambiguating Reference in Visually Grounded Dialogues through Joint Modeling of Textual and Multimodal Semantic Structures
- **分类: cs.CL**

- **简介: 该论文研究多模态指代消解任务，解决视觉对话中因代词和省略导致的指代歧义问题。通过联合建模文本（共指消解、谓词结构）与多模态语义，将文本提及与视觉对象嵌入对齐，基于相似性匹配消除歧义。实验表明融合文本关系能提升代词接地效果，优于MDETR和GLIP，并通过增强提及-对象置信度减少视觉对话的模糊性。**

- **链接: [http://arxiv.org/pdf/2505.11726v1](http://arxiv.org/pdf/2505.11726v1)**

> **作者:** Shun Inadumi; Nobuhiro Ueda; Koichiro Yoshino
>
> **备注:** ACL2025 main. Code available at https://github.com/SInadumi/mmrr
>
> **摘要:** Multimodal reference resolution, including phrase grounding, aims to understand the semantic relations between mentions and real-world objects. Phrase grounding between images and their captions is a well-established task. In contrast, for real-world applications, it is essential to integrate textual and multimodal reference resolution to unravel the reference relations within dialogue, especially in handling ambiguities caused by pronouns and ellipses. This paper presents a framework that unifies textual and multimodal reference resolution by mapping mention embeddings to object embeddings and selecting mentions or objects based on their similarity. Our experiments show that learning textual reference resolution, such as coreference resolution and predicate-argument structure analysis, positively affects performance in multimodal reference resolution. In particular, our model with coreference resolution performs better in pronoun phrase grounding than representative models for this task, MDETR and GLIP. Our qualitative analysis demonstrates that incorporating textual reference relations strengthens the confidence scores between mentions, including pronouns and predicates, and objects, which can reduce the ambiguities that arise in visually grounded dialogues.
>
---
#### [new 087] Learning to Play Like Humans: A Framework for LLM Adaptation in Interactive Fiction Games
- **分类: cs.CL**

- **简介: 该论文属于AI决策任务，旨在解决现有方法在交互式小说游戏中缺乏人类式叙事理解和常识约束的问题。提出LPLH框架，通过结构化地图建模、上下文动作学习和反馈优化，引导LLM模拟人类认知过程，实现符合叙事逻辑的决策，提升复杂文本环境中的可解释性与上下文适应能力。**

- **链接: [http://arxiv.org/pdf/2505.12439v1](http://arxiv.org/pdf/2505.12439v1)**

> **作者:** Jinming Zhang; Yunfei Long
>
> **摘要:** Interactive Fiction games (IF games) are where players interact through natural language commands. While recent advances in Artificial Intelligence agents have reignited interest in IF games as a domain for studying decision-making, existing approaches prioritize task-specific performance metrics over human-like comprehension of narrative context and gameplay logic. This work presents a cognitively inspired framework that guides Large Language Models (LLMs) to learn and play IF games systematically. Our proposed **L**earning to **P**lay **L**ike **H**umans (LPLH) framework integrates three key components: (1) structured map building to capture spatial and narrative relationships, (2) action learning to identify context-appropriate commands, and (3) feedback-driven experience analysis to refine decision-making over time. By aligning LLMs-based agents' behavior with narrative intent and commonsense constraints, LPLH moves beyond purely exploratory strategies to deliver more interpretable, human-like performance. Crucially, this approach draws on cognitive science principles to more closely simulate how human players read, interpret, and respond within narrative worlds. As a result, LPLH reframes the IF games challenge as a learning problem for LLMs-based agents, offering a new path toward robust, context-aware gameplay in complex text-based environments.
>
---
#### [new 088] LLMSR@XLLM25: An Empirical Study of LLM for Structural Reasoning
- **分类: cs.CL**

- **简介: 该论文参与LLMSR@XLLM25评测任务，研究大语言模型在结构化推理中的表现，解决推理过程细粒度控制与可解释性问题。通过设计多轮少量示例提示策略，引导Meta-Llama-3-8B模型分解推理步骤并验证逻辑，结合正则后处理，在无调优/外部资源条件下取得与复杂系统相当的分数。**

- **链接: [http://arxiv.org/pdf/2505.12328v1](http://arxiv.org/pdf/2505.12328v1)**

> **作者:** Xinye Li; Mingqi Wan; Dianbo Sui
>
> **摘要:** We present Team asdfo123's submission to the LLMSR@XLLM25 shared task, which evaluates large language models on producing fine-grained, controllable, and interpretable reasoning processes. Systems must extract all problem conditions, decompose a chain of thought into statement-evidence pairs, and verify the logical validity of each pair. Leveraging only the off-the-shelf Meta-Llama-3-8B-Instruct, we craft a concise few-shot, multi-turn prompt that first enumerates all conditions and then guides the model to label, cite, and adjudicate every reasoning step. A lightweight post-processor based on regular expressions normalises spans and enforces the official JSON schema. Without fine-tuning, external retrieval, or ensembling, our method ranks 5th overall, achieving macro F1 scores on par with substantially more complex and resource-consuming pipelines. We conclude by analysing the strengths and limitations of our approach and outlining directions for future research in structural reasoning with LLMs. Our code is available at https://github.com/asdfo123/LLMSR-asdfo123.
>
---
#### [new 089] On-Policy Optimization with Group Equivalent Preference for Multi-Programming Language Understanding
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决大语言模型（LLMs）对不同编程语言理解能力不均衡的问题。提出融合策略内外强化学习的OORL框架，结合基于单元测试的规则奖励和组等价偏好优化（GEPO），通过代码翻译任务引导LLM利用中间表示组学习代码功能共性，提升跨语言代码理解能力。**

- **链接: [http://arxiv.org/pdf/2505.12723v1](http://arxiv.org/pdf/2505.12723v1)**

> **作者:** Haoyuan Wu; Rui Ming; Jilong Gao; Hangyu Zhao; Xueyi Chen; Yikai Yang; Haisheng Zheng; Zhuolun He; Bei Yu
>
> **摘要:** Large language models (LLMs) achieve remarkable performance in code generation tasks. However, a significant performance disparity persists between popular programming languages (e.g., Python, C++) and others. To address this capability gap, we leverage the code translation task to train LLMs, thereby facilitating the transfer of coding proficiency across diverse programming languages. Moreover, we introduce OORL for training, a novel reinforcement learning (RL) framework that integrates on-policy and off-policy strategies. Within OORL, on-policy RL is applied during code translation, guided by a rule-based reward signal derived from unit tests. Complementing this coarse-grained rule-based reward, we propose Group Equivalent Preference Optimization (GEPO), a novel preference optimization method. Specifically, GEPO trains the LLM using intermediate representations (IRs) groups. LLMs can be guided to discern IRs equivalent to the source code from inequivalent ones, while also utilizing signals about the mutual equivalence between IRs within the group. This process allows LLMs to capture nuanced aspects of code functionality. By employing OORL for training with code translation tasks, LLMs improve their recognition of code functionality and their understanding of the relationships between code implemented in different languages. Extensive experiments demonstrate that our OORL for LLMs training with code translation tasks achieves significant performance improvements on code benchmarks across multiple programming languages.
>
---
#### [new 090] An Explanation of Intrinsic Self-Correction via Linear Representations and Latent Concepts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型内在自我纠正机制，属于模型解释性任务。旨在揭示提示如何通过线性表示和潜在概念调整隐状态，从而提升输出质量。提出数学框架分析提示诱导的隐态变化，实验验证了文本去毒任务中提示增强潜在概念识别能力，解释了自校正原理。**

- **链接: [http://arxiv.org/pdf/2505.11924v1](http://arxiv.org/pdf/2505.11924v1)**

> **作者:** Yu-Ting Lee; Hui-Ying Shih; Fu-Chieh Chang; Pei-Yuan Wu
>
> **摘要:** We provide an explanation for the performance gains of intrinsic self-correction, a process where a language model iteratively refines its outputs without external feedback. More precisely, we investigate how prompting induces interpretable changes in hidden states and thus affects the output distributions. We hypothesize that each prompt-induced shift lies in a linear span of some linear representation vectors, naturally separating tokens based on individual concept alignment. Building around this idea, we give a mathematical formulation of self-correction and derive a concentration result for output tokens based on alignment magnitudes. Our experiments on text detoxification with zephyr-7b-sft reveal a substantial gap in the inner products of the prompt-induced shifts and the unembeddings of the top-100 most toxic tokens vs. those of the unembeddings of the bottom-100 least toxic tokens, under toxic instructions. This suggests that self-correction prompts enhance a language model's capability of latent concept recognition. Our analysis offers insights into the underlying mechanism of self-correction by characterizing how prompting works explainably. For reproducibility, our code is available.
>
---
#### [new 091] LEXam: Benchmarking Legal Reasoning on 340 Law Exams
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2**

- **简介: 该论文属于法律推理评估任务，旨在解决大模型在复杂法律问题中表现不足的问题。研究者构建了LEXam基准数据集（含340场考试、4886题），涵盖开放式与选择题，并设计基于模型自评与专家验证的评估方法，以量化模型多步骤法律推理能力差异。**

- **链接: [http://arxiv.org/pdf/2505.12864v1](http://arxiv.org/pdf/2505.12864v1)**

> **作者:** Yu Fan; Jingwei Ni; Jakob Merane; Etienne Salimbeni; Yang Tian; Yoan Hermstrüwer; Yinya Huang; Mubashara Akhtar; Florian Geering; Oliver Dreyer; Daniel Brunner; Markus Leippold; Mrinmaya Sachan; Alexander Stremitzer; Christoph Engel; Elliott Ash; Joel Niklaus
>
> **摘要:** Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: https://lexam-benchmark.github.io/
>
---
#### [new 092] LLM-Based Evaluation of Low-Resource Machine Translation: A Reference-less Dialect Guided Approach with a Refined Sylheti-English Benchmark
- **分类: cs.CL**

- **简介: 该论文针对低资源语言机器翻译评估难题（尤其是多方言场景），提出基于LLMs的无参考评估框架。通过扩展Sylheti-英语数据集、引入方言词汇增强和定制化提示策略，提升评估准确性。实验显示其方法在Spearman相关性等指标上优于现有方案。**

- **链接: [http://arxiv.org/pdf/2505.12273v1](http://arxiv.org/pdf/2505.12273v1)**

> **作者:** Md. Atiqur Rahman; Sabrina Islam; Mushfiqul Haque Omi
>
> **摘要:** Evaluating machine translation (MT) for low-resource languages poses a persistent challenge, primarily due to the limited availability of high quality reference translations. This issue is further exacerbated in languages with multiple dialects, where linguistic diversity and data scarcity hinder robust evaluation. Large Language Models (LLMs) present a promising solution through reference-free evaluation techniques; however, their effectiveness diminishes in the absence of dialect-specific context and tailored guidance. In this work, we propose a comprehensive framework that enhances LLM-based MT evaluation using a dialect guided approach. We extend the ONUBAD dataset by incorporating Sylheti-English sentence pairs, corresponding machine translations, and Direct Assessment (DA) scores annotated by native speakers. To address the vocabulary gap, we augment the tokenizer vocabulary with dialect-specific terms. We further introduce a regression head to enable scalar score prediction and design a dialect-guided (DG) prompting strategy. Our evaluation across multiple LLMs shows that the proposed pipeline consistently outperforms existing methods, achieving the highest gain of +0.1083 in Spearman correlation, along with improvements across other evaluation settings. The dataset and the code are available at https://github.com/180041123-Atiq/MTEonLowResourceLanguage.
>
---
#### [new 093] Rethinking Stateful Tool Use in Multi-Turn Dialogues: Benchmarks and Challenges
- **分类: cs.CL**

- **简介: 该论文属于语言模型工具使用评估任务，旨在解决现有基准忽视多轮对话状态交互的问题。作者构建了多轮数据集DialogTool，覆盖工具创建、使用和角色响应全生命周期，并开发虚拟环境VirtualMobile模拟API调用。通过评估13个模型，发现现有模型在长序列工具使用中表现不足。**

- **链接: [http://arxiv.org/pdf/2505.13328v1](http://arxiv.org/pdf/2505.13328v1)**

> **作者:** Hongru Wang; Wenyu Huang; Yufei Wang; Yuanhao Xi; Jianqiao Lu; Huan Zhang; Nan Hu; Zeming Liu; Jeff Z. Pan; Kam-Fai Wong
>
> **摘要:** Existing benchmarks that assess Language Models (LMs) as Language Agents (LAs) for tool use primarily focus on stateless, single-turn interactions or partial evaluations, such as tool selection in a single turn, overlooking the inherent stateful nature of interactions in multi-turn applications. To fulfill this gap, we propose \texttt{DialogTool}, a multi-turn dialogue dataset with stateful tool interactions considering the whole life cycle of tool use, across six key tasks in three stages: 1) \textit{tool creation}; 2) \textit{tool utilization}: tool awareness, tool selection, tool execution; and 3) \textit{role-consistent response}: response generation and role play. Furthermore, we build \texttt{VirtualMobile} -- an embodied virtual mobile evaluation environment to simulate API calls and assess the robustness of the created APIs\footnote{We will use tools and APIs alternatively, there are no significant differences between them in this paper.}. Taking advantage of these artifacts, we conduct comprehensive evaluation on 13 distinct open- and closed-source LLMs and provide detailed analysis at each stage, revealing that the existing state-of-the-art LLMs still cannot perform well to use tools over long horizons.
>
---
#### [new 094] Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于多模态语音大语言模型训练任务，旨在解决现有语音-LLM在上下文推理与副语言理解（如情感语调）能力不足的问题。研究者提出首个整合两类信息的数据集生成框架：通过伪副语言标签浓缩语音数据，并基于LLM生成上下文副语言QA对，验证了数据有效性并揭示了模型在共情推理上的缺陷。**

- **链接: [http://arxiv.org/pdf/2505.13338v1](http://arxiv.org/pdf/2505.13338v1)**

> **作者:** Qiongqiong Wang; Hardik B. Sailor; Tianchi Liu; Ai Ti Aw
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities.
>
---
#### [new 095] Automatic Speech Recognition for African Low-Resource Languages: Challenges and Future Directions
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于低资源语言的自动语音识别（ASR）开发任务，旨在解决非洲语言因数据稀缺、计算资源不足等挑战导致的ASR技术发展滞后问题。研究分析了技术障碍，提出社区协作数据采集、轻量化模型等策略，并通过案例验证了定制化方案的可行性，推动包容性ASR系统建设。**

- **链接: [http://arxiv.org/pdf/2505.11690v1](http://arxiv.org/pdf/2505.11690v1)**

> **作者:** Sukairaj Hafiz Imam; Babangida Sani; Dawit Ketema Gete; Bedru Yimam Ahamed; Ibrahim Said Ahmad; Idris Abdulmumin; Seid Muhie Yimam; Muhammad Yahuza Bello; Shamsuddeen Hassan Muhammad
>
> **摘要:** Automatic Speech Recognition (ASR) technologies have transformed human-computer interaction; however, low-resource languages in Africa remain significantly underrepresented in both research and practical applications. This study investigates the major challenges hindering the development of ASR systems for these languages, which include data scarcity, linguistic complexity, limited computational resources, acoustic variability, and ethical concerns surrounding bias and privacy. The primary goal is to critically analyze these barriers and identify practical, inclusive strategies to advance ASR technologies within the African context. Recent advances and case studies emphasize promising strategies such as community-driven data collection, self-supervised and multilingual learning, lightweight model architectures, and techniques that prioritize privacy. Evidence from pilot projects involving various African languages showcases the feasibility and impact of customized solutions, which encompass morpheme-based modeling and domain-specific ASR applications in sectors like healthcare and education. The findings highlight the importance of interdisciplinary collaboration and sustained investment to tackle the distinct linguistic and infrastructural challenges faced by the continent. This study offers a progressive roadmap for creating ethical, efficient, and inclusive ASR systems that not only safeguard linguistic diversity but also improve digital accessibility and promote socioeconomic participation for speakers of African languages.
>
---
#### [new 096] Not All Documents Are What You Need for Extracting Instruction Tuning Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究指令微调数据提取任务，解决传统方法生成数据多样性不足、计算成本高的问题。提出EQUAL框架，通过聚类文档和动态筛选高价值QA对，迭代优化数据提取，降低计算开销并提升模型性能。实验验证其高效性和准确性提升。**

- **链接: [http://arxiv.org/pdf/2505.12250v1](http://arxiv.org/pdf/2505.12250v1)**

> **作者:** Chi Zhang; Huaping Zhong; Hongtao Li; Chengliang Chai; Jiawei Hong; Yuhao Deng; Jiacheng Wang; Tian Tan; Yizhou Yan; Jiantao Qiu; Ye Yuan; Guoren Wang; Conghui He; Lei Cao
>
> **摘要:** Instruction tuning improves the performance of large language models (LLMs), but it heavily relies on high-quality training data. Recently, LLMs have been used to synthesize instruction data using seed question-answer (QA) pairs. However, these synthesized instructions often lack diversity and tend to be similar to the input seeds, limiting their applicability in real-world scenarios. To address this, we propose extracting instruction tuning data from web corpora that contain rich and diverse knowledge. A naive solution is to retrieve domain-specific documents and extract all QA pairs from them, but this faces two key challenges: (1) extracting all QA pairs using LLMs is prohibitively expensive, and (2) many extracted QA pairs may be irrelevant to the downstream tasks, potentially degrading model performance. To tackle these issues, we introduce EQUAL, an effective and scalable data extraction framework that iteratively alternates between document selection and high-quality QA pair extraction to enhance instruction tuning. EQUAL first clusters the document corpus based on embeddings derived from contrastive learning, then uses a multi-armed bandit strategy to efficiently identify clusters that are likely to contain valuable QA pairs. This iterative approach significantly reduces computational cost while boosting model performance. Experiments on AutoMathText and StackOverflow across four downstream tasks show that EQUAL reduces computational costs by 5-10x and improves accuracy by 2.5 percent on LLaMA-3.1-8B and Mistral-7B
>
---
#### [new 097] Hierarchical Bracketing Encodings for Dependency Parsing as Tagging
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理的依存句法分析任务，旨在优化序列标注编码方法。针对现有4位投影编码标签冗余和非投影支持不足的问题，提出基于层次括号的最优编码方案：将投影树标签从16压缩到12，并扩展非投影结构的紧凑表达。实验证明新方法在多个树库上达到竞争性精度。**

- **链接: [http://arxiv.org/pdf/2505.11693v1](http://arxiv.org/pdf/2505.11693v1)**

> **作者:** Ana Ezquerro; David Vilares; Anssi Yli-Jyrä; Carlos Gómez-Rodríguez
>
> **备注:** Accepted to ACL 2025. Original submission; camera-ready coming soon
>
> **摘要:** We present a family of encodings for sequence labeling dependency parsing, based on the concept of hierarchical bracketing. We prove that the existing 4-bit projective encoding belongs to this family, but it is suboptimal in the number of labels used to encode a tree. We derive an optimal hierarchical bracketing, which minimizes the number of symbols used and encodes projective trees using only 12 distinct labels (vs. 16 for the 4-bit encoding). We also extend optimal hierarchical bracketing to support arbitrary non-projectivity in a more compact way than previous encodings. Our new encodings yield competitive accuracy on a diverse set of treebanks.
>
---
#### [new 098] Retrospex: Language Agent Meets Offline Reinforcement Learning Critic
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Retrospex框架，属于强化学习与语言智能体结合领域。针对现有LLM智能体未充分利用历史经验的问题，通过离线强化学习评论家评估动作价值，结合LLM动作概率动态调整经验权重，在交互密集型任务中提升决策能力，实验验证其在多环境中的优越性。**

- **链接: [http://arxiv.org/pdf/2505.11807v1](http://arxiv.org/pdf/2505.11807v1)**

> **作者:** Yufei Xiang; Yiqun Shen; Yeqin Zhang; Cam-Tu Nguyen
>
> **备注:** 17 pages
>
> **摘要:** Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines.
>
---
#### [new 099] SNAPE-PM: Building and Utilizing Dynamic Partner Models for Adaptive Explanation Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自适应解释生成任务，解决对话系统难以动态调整解释策略的问题。提出基于贝叶斯推理的伙伴模型追踪用户特征，结合非稳态马尔可夫决策过程动态优化策略，通过模拟实验验证了模型能有效适配不同用户（含动态反馈场景），提升可解释AI系统的适应性。**

- **链接: [http://arxiv.org/pdf/2505.13053v1](http://arxiv.org/pdf/2505.13053v1)**

> **作者:** Amelie S. Robrecht; Christoph R. Kowalski; Stefan Kopp
>
> **备注:** currently under review at Frontiers in Communication
>
> **摘要:** Adapting to the addressee is crucial for successful explanations, yet poses significant challenges for dialogsystems. We adopt the approach of treating explanation generation as a non-stationary decision process, where the optimal strategy varies according to changing beliefs about the explainee and the interaction context. In this paper we address the questions of (1) how to track the interaction context and the relevant listener features in a formally defined computational partner model, and (2) how to utilize this model in the dynamically adjusted, rational decision process that determines the currently best explanation strategy. We propose a Bayesian inference-based approach to continuously update the partner model based on user feedback, and a non-stationary Markov Decision Process to adjust decision-making based on the partner model values. We evaluate an implementation of this framework with five simulated interlocutors, demonstrating its effectiveness in adapting to different partners with constant and even changing feedback behavior. The results show high adaptivity with distinct explanation strategies emerging for different partners, highlighting the potential of our approach to improve explainable AI systems and dialogsystems in general.
>
---
#### [new 100] ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents
- **分类: cs.CL**

- **简介: 该论文属于对话系统评估任务，旨在解决情感支持对话代理缺乏可扩展理论化评估方法的问题。提出了ESC-Judge框架，基于心理咨询理论构建自动化评估流程：通过合成用户角色、隔离模型策略、使用专用LLM按标准对比模型表现，验证其评估结果与人类专家一致性达85%以上。**

- **链接: [http://arxiv.org/pdf/2505.12531v1](http://arxiv.org/pdf/2505.12531v1)**

> **作者:** Navid Madani; Rohini Srihari
>
> **摘要:** Large language models (LLMs) increasingly power mental-health chatbots, yet the field still lacks a scalable, theory-grounded way to decide which model is most effective to deploy. We present ESC-Judge, the first end-to-end evaluation framework that (i) grounds head-to-head comparisons of emotional-support LLMs in Clara Hill's established Exploration-Insight-Action counseling model, providing a structured and interpretable view of performance, and (ii) fully automates the evaluation pipeline at scale. ESC-Judge operates in three stages: first, it synthesizes realistic help-seeker roles by sampling empirically salient attributes such as stressors, personality, and life history; second, it has two candidate support agents conduct separate sessions with the same role, isolating model-specific strategies; and third, it asks a specialized judge LLM to express pairwise preferences across rubric-anchored skills that span the Exploration, Insight, and Action spectrum. In our study, ESC-Judge matched PhD-level annotators on 85 percent of Exploration, 83 percent of Insight, and 86 percent of Action decisions, demonstrating human-level reliability at a fraction of the cost. All code, prompts, synthetic roles, transcripts, and judgment scripts are released to promote transparent progress in emotionally supportive AI.
>
---
#### [new 101] GenderBench: Evaluation Suite for Gender Biases in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）性别偏见评估任务，旨在解决LLM中性别偏见缺乏系统性评测的问题。研究者开发了开源评估工具GenderBench，包含14个探针量化19种有害行为，并测试了12个主流模型，发现其存在刻板推理、生成文本性别失衡及高风险场景歧视等问题。**

- **链接: [http://arxiv.org/pdf/2505.12054v1](http://arxiv.org/pdf/2505.12054v1)**

> **作者:** Matúš Pikuliak
>
> **摘要:** We present GenderBench -- a comprehensive evaluation suite designed to measure gender biases in LLMs. GenderBench includes 14 probes that quantify 19 gender-related harmful behaviors exhibited by LLMs. We release GenderBench as an open-source and extensible library to improve the reproducibility and robustness of benchmarking across the field. We also publish our evaluation of 12 LLMs. Our measurements reveal consistent patterns in their behavior. We show that LLMs struggle with stereotypical reasoning, equitable gender representation in generated texts, and occasionally also with discriminatory behavior in high-stakes scenarios, such as hiring.
>
---
#### [new 102] CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于Text-to-SQL任务，旨在提升自然语言转SQL的准确性。针对现有自洽和自校正方法存在次优选择和仅修正语法错误的问题，提出CSC-SQL方法：通过并行采样选取高频候选，结合合并修正模型纠错，并采用强化学习微调生成与修正模型。实验显示其3B/7B模型在BIRD数据集分别达到65.28%/69.19%执行准确率。**

- **链接: [http://arxiv.org/pdf/2505.13271v1](http://arxiv.org/pdf/2505.13271v1)**

> **作者:** Lei Sheng; Shuai-Shuai Xu
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in translating natural language questions about relational databases into SQL queries. In particular, test-time scaling techniques such as Self-Consistency and Self-Correction can enhance SQL generation accuracy by increasing computational effort during inference. However, these methods have notable limitations: Self-Consistency may select suboptimal outputs despite majority votes, while Self-Correction typically addresses only syntactic errors. To leverage the strengths of both approaches, we propose CSC-SQL, a novel method that integrates Self-Consistency and Self-Correction. CSC-SQL selects the two most frequently occurring outputs from parallel sampling and feeds them into a merge revision model for correction. Additionally, we employ the Group Relative Policy Optimization (GRPO) algorithm to fine-tune both the SQL generation and revision models via reinforcement learning, significantly enhancing output quality. Experimental results confirm the effectiveness and generalizability of CSC-SQL. On the BIRD development set, our 3B model achieves 65.28% execution accuracy, while the 7B model achieves 69.19%. The code will be open sourced at https://github.com/CycloneBoy/csc_sql.
>
---
#### [new 103] Understanding Cross-Lingual Inconsistency in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型跨语言输出不一致问题，属于多语言知识迁移任务。通过分析隐藏状态发现模型依赖单语言子空间而非共享语义空间，导致准确率低。实验表明大模型隐藏状态更偏离共享空间但检索能力更强，调节潜在处理可提升知识共享与推理一致性。**

- **链接: [http://arxiv.org/pdf/2505.13141v1](http://arxiv.org/pdf/2505.13141v1)**

> **作者:** Zheng Wei Lim; Alham Fikri Aji; Trevor Cohn
>
> **摘要:** Large language models (LLMs) are demonstrably capable of cross-lingual transfer, but can produce inconsistent output when prompted with the same queries written in different languages. To understand how language models are able to generalize knowledge from one language to the others, we apply the logit lens to interpret the implicit steps taken by LLMs to solve multilingual multi-choice reasoning questions. We find LLMs predict inconsistently and are less accurate because they rely on subspaces of individual languages, rather than working in a shared semantic space. While larger models are more multilingual, we show their hidden states are more likely to dissociate from the shared representation compared to smaller models, but are nevertheless more capable of retrieving knowledge embedded across different languages. Finally, we demonstrate that knowledge sharing can be modulated by steering the models' latent processing towards the shared semantic space. We find reinforcing utilization of the shared space improves the models' multilingual reasoning performance, as a result of more knowledge transfer from, and better output consistency with English.
>
---
#### [new 104] What if Deception Cannot be Detected? A Cross-Linguistic Study on the Limits of Deception Detection from Text
- **分类: cs.CL**

- **简介: 该论文研究自然语言处理中的欺骗检测任务，质疑现有方法依赖数据集特定线索的可靠性。通过构建多语言DeFaBel语料库（含德语/英语），提出基于信念的欺骗框架，发现传统语言线索与欺骗标签相关性微弱，且模型在新数据集上表现接近随机，揭示现有检测方法的泛化性缺陷，呼吁重构NLP领域的欺骗研究范式。**

- **链接: [http://arxiv.org/pdf/2505.13147v1](http://arxiv.org/pdf/2505.13147v1)**

> **作者:** Aswathy Velutharambath; Roman Klinger; Kai Sassenberg
>
> **摘要:** Can deception be detected solely from written text? Cues of deceptive communication are inherently subtle, even more so in text-only communication. Yet, prior studies have reported considerable success in automatic deception detection. We hypothesize that such findings are largely driven by artifacts introduced during data collection and do not generalize beyond specific datasets. We revisit this assumption by introducing a belief-based deception framework, which defines deception as a misalignment between an author's claims and true beliefs, irrespective of factual accuracy, allowing deception cues to be studied in isolation. Based on this framework, we construct three corpora, collectively referred to as DeFaBel, including a German-language corpus of deceptive and non-deceptive arguments and a multilingual version in German and English, each collected under varying conditions to account for belief change and enable cross-linguistic analysis. Using these corpora, we evaluate commonly reported linguistic cues of deception. Across all three DeFaBel variants, these cues show negligible, statistically insignificant correlations with deception labels, contrary to prior work that treats such cues as reliable indicators. We further benchmark against other English deception datasets following similar data collection protocols. While some show statistically significant correlations, effect sizes remain low and, critically, the set of predictive cues is inconsistent across datasets. We also evaluate deception detection using feature-based models, pretrained language models, and instruction-tuned large language models. While some models perform well on established deception datasets, they consistently perform near chance on DeFaBel. Our findings challenge the assumption that deception can be reliably inferred from linguistic cues and call for rethinking how deception is studied and modeled in NLP.
>
---
#### [new 105] Unveiling Knowledge Utilization Mechanisms in LLM-based Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究基于大语言模型（LLM）的检索增强生成（RAG）中知识利用机制，解决其内部整合参数化知识与外部检索知识的机理不明问题。通过宏观知识流分析将知识处理分解为四个阶段，并设计微观模块级方法（如KAPE神经元识别），揭示多注意力层与MLP在知识形成中的互补作用，提升模型可解释性与可靠性。**

- **链接: [http://arxiv.org/pdf/2505.11995v1](http://arxiv.org/pdf/2505.11995v1)**

> **作者:** Yuhao Wang; Ruiyang Ren; Yucheng Wang; Wayne Xin Zhao; Jing Liu; Hua Wu; Haifeng Wang
>
> **备注:** SIGIR 2025
>
> **摘要:** Considering the inherent limitations of parametric knowledge in large language models (LLMs), retrieval-augmented generation (RAG) is widely employed to expand their knowledge scope. Since RAG has shown promise in knowledge-intensive tasks like open-domain question answering, its broader application to complex tasks and intelligent assistants has further advanced its utility. Despite this progress, the underlying knowledge utilization mechanisms of LLM-based RAG remain underexplored. In this paper, we present a systematic investigation of the intrinsic mechanisms by which LLMs integrate internal (parametric) and external (retrieved) knowledge in RAG scenarios. Specially, we employ knowledge stream analysis at the macroscopic level, and investigate the function of individual modules at the microscopic level. Drawing on knowledge streaming analyses, we decompose the knowledge utilization process into four distinct stages within LLM layers: knowledge refinement, knowledge elicitation, knowledge expression, and knowledge contestation. We further demonstrate that the relevance of passages guides the streaming of knowledge through these stages. At the module level, we introduce a new method, knowledge activation probability entropy (KAPE) for neuron identification associated with either internal or external knowledge. By selectively deactivating these neurons, we achieve targeted shifts in the LLM's reliance on one knowledge source over the other. Moreover, we discern complementary roles for multi-head attention and multi-layer perceptron layers during knowledge formation. These insights offer a foundation for improving interpretability and reliability in retrieval-augmented LLMs, paving the way for more robust and transparent generative solutions in knowledge-intensive domains.
>
---
#### [new 106] EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code
- **分类: cs.CL**

- **简介: 该论文属于代码生成评估任务，旨在解决现有基准忽视代码效率和多语言支持的问题。提出了首个多语言效率基准EffiBench-X，涵盖6种编程语言，通过竞赛题目和人类专家方案建立效率基线。实验发现LLM生成代码效率平均仅为人类62%，存在显著语言差异（Python/Ruby/JS优于Java/C++/Golang），揭示了跨语言代码优化的研究需求。**

- **链接: [http://arxiv.org/pdf/2505.13004v1](http://arxiv.org/pdf/2505.13004v1)**

> **作者:** Yuhao Qing; Boyu Zhu; Mingzhe Du; Zhijiang Guo; Terry Yue Zhuo; Qianru Zhang; Jie M. Zhang; Heming Cui; Siu-Ming Yiu; Dong Huang; See-Kiong Ng; Luu Anh Tuan
>
> **备注:** Under Review
>
> **摘要:** Existing code generation benchmarks primarily evaluate functional correctness, with limited focus on code efficiency and often restricted to a single language like Python. To address this gap, we introduce EffiBench-X, the first multi-language benchmark designed to measure the efficiency of LLM-generated code. EffiBench-X supports Python, C++, Java, JavaScript, Ruby, and Golang. It comprises competitive programming tasks with human-expert solutions as efficiency baselines. Evaluating state-of-the-art LLMs on EffiBench-X reveals that while models generate functionally correct code, they consistently underperform human experts in efficiency. Even the most efficient LLM-generated solutions (Qwen3-32B) achieve only around \textbf{62\%} of human efficiency on average, with significant language-specific variations. LLMs show better efficiency in Python, Ruby, and JavaScript than in Java, C++, and Golang. For instance, DeepSeek-R1's Python code is significantly more efficient than its Java code. These results highlight the critical need for research into LLM optimization techniques to improve code efficiency across diverse languages. The dataset and evaluation infrastructure are submitted and available at https://github.com/EffiBench/EffiBench-X.git and https://huggingface.co/datasets/EffiBench/effibench-x.
>
---
#### [new 107] Neuro-Symbolic Query Compiler
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出QCompiler，一种神经符号框架，解决RAG系统中复杂嵌套查询的意图识别问题。通过设计最小冗余的BNF语法，结合语法解析器和递归处理器，将查询编译为AST，提升文档检索和响应生成的精确性。**

- **链接: [http://arxiv.org/pdf/2505.11932v1](http://arxiv.org/pdf/2505.11932v1)**

> **作者:** Yuyao Zhang; Zhicheng Dou; Xiaoxi Li; Jiajie Jin; Yongkang Wu; Zhonghua Li; Qi Ye; Ji-Rong Wen
>
> **备注:** Findings of ACL2025, codes are available at this url: https://github.com/YuyaoZhangQAQ/Query_Compiler
>
> **摘要:** Precise recognition of search intent in Retrieval-Augmented Generation (RAG) systems remains a challenging goal, especially under resource constraints and for complex queries with nested structures and dependencies. This paper presents QCompiler, a neuro-symbolic framework inspired by linguistic grammar rules and compiler design, to bridge this gap. It theoretically designs a minimal yet sufficient Backus-Naur Form (BNF) grammar $G[q]$ to formalize complex queries. Unlike previous methods, this grammar maintains completeness while minimizing redundancy. Based on this, QCompiler includes a Query Expression Translator, a Lexical Syntax Parser, and a Recursive Descent Processor to compile queries into Abstract Syntax Trees (ASTs) for execution. The atomicity of the sub-queries in the leaf nodes ensures more precise document retrieval and response generation, significantly improving the RAG system's ability to address complex queries.
>
---
#### [new 108] What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于心理健康污名检测任务，旨在解决现有数据缺乏理论支撑的问题。研究者构建了专家标注的理论驱动型人机访谈语料库（含4141条数据），用于训练模型精准识别污名化内容，并通过实验评估模型性能，助力反污名化研究。**

- **链接: [http://arxiv.org/pdf/2505.12727v1](http://arxiv.org/pdf/2505.12727v1)**

> **作者:** Han Meng; Yancan Chen; Yunan Li; Yitian Yang; Jungup Lee; Renwen Zhang; Yi-Chieh Lee
>
> **备注:** Accepted to ACL 2025 Main Conference, 35 Pages
>
> **摘要:** Mental-health stigma remains a pervasive social problem that hampers treatment-seeking and recovery. Existing resources for training neural models to finely classify such stigma are limited, relying primarily on social-media or synthetic data without theoretical underpinnings. To remedy this gap, we present an expert-annotated, theory-informed corpus of human-chatbot interviews, comprising 4,141 snippets from 684 participants with documented socio-cultural backgrounds. Our experiments benchmark state-of-the-art neural models and empirically unpack the challenges of stigma detection. This dataset can facilitate research on computationally detecting, neutralizing, and counteracting mental-health stigma.
>
---
#### [new 109] CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型的提示注入安全风险，提出上下文感知的测试基准CAPTURE，解决现有防护模型依赖静态攻击样本导致的漏检与过度防御问题。通过构建动态评估框架，验证当前方法在对抗场景中误判率高，并实现轻量级鲁棒性增强。**

- **链接: [http://arxiv.org/pdf/2505.12368v1](http://arxiv.org/pdf/2505.12368v1)**

> **作者:** Gauri Kholkar; Ratinder Ahuja
>
> **备注:** Accepted in ACL LLMSec Workshop 2025
>
> **摘要:** Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations.
>
---
#### [new 110] SMOTExT: SMOTE meets Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SMOTExT方法，解决NLP中数据稀缺与类别不平衡问题。通过将SMOTE算法扩展至文本领域，结合BERT嵌入插值生成潜在向量，利用xRAG框架解码为连贯文本，实现数据增强与隐私保护。属于文本生成与数据增强任务，探索了少样本场景下的知识蒸馏潜力。**

- **链接: [http://arxiv.org/pdf/2505.13434v1](http://arxiv.org/pdf/2505.13434v1)**

> **作者:** Mateusz Bystroński; Mikołaj Hołysz; Grzegorz Piotrowski; Nitesh V. Chawla; Tomasz Kajdanowicz
>
> **摘要:** Data scarcity and class imbalance are persistent challenges in training robust NLP models, especially in specialized domains or low-resource settings. We propose a novel technique, SMOTExT, that adapts the idea of Synthetic Minority Over-sampling (SMOTE) to textual data. Our method generates new synthetic examples by interpolating between BERT-based embeddings of two existing examples and then decoding the resulting latent point into text with xRAG architecture. By leveraging xRAG's cross-modal retrieval-generation framework, we can effectively turn interpolated vectors into coherent text. While this is preliminary work supported by qualitative outputs only, the method shows strong potential for knowledge distillation and data augmentation in few-shot settings. Notably, our approach also shows promise for privacy-preserving machine learning: in early experiments, training models solely on generated data achieved comparable performance to models trained on the original dataset. This suggests a viable path toward safe and effective learning under data protection constraints.
>
---
#### [new 111] Role-Playing Evaluation for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLM）评估任务，旨在解决角色扮演能力评测中人工成本高、自动化方法偏差大的问题。研究者提出RPEval基准，从情感理解、决策、道德对齐和角色一致性四个维度评估LLM表现，并公开了数据集与代码。**

- **链接: [http://arxiv.org/pdf/2505.13157v1](http://arxiv.org/pdf/2505.13157v1)**

> **作者:** Yassine El Boudouri; Walter Nuninger; Julian Alvarez; Yvan Peter
>
> **摘要:** Large Language Models (LLMs) demonstrate a notable capacity for adopting personas and engaging in role-playing. However, evaluating this ability presents significant challenges, as human assessments are resource-intensive and automated evaluations can be biased. To address this, we introduce Role-Playing Eval (RPEval), a novel benchmark designed to assess LLM role-playing capabilities across four key dimensions: emotional understanding, decision-making, moral alignment, and in-character consistency. This article details the construction of RPEval and presents baseline evaluations. Our code and dataset are available at https://github.com/yelboudouri/RPEval
>
---
#### [new 112] Granary: Speech Recognition and Translation Dataset in 25 European Languages
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言语音识别与翻译任务，旨在解决低资源语言数据稀缺问题。研究者构建了首个覆盖25种欧洲语言的开源数据集Granary，通过伪标注、去幻觉、标点恢复及翻译对生成等流程提升数据质量，并验证了模型在减少50%数据量时仍保持性能。**

- **链接: [http://arxiv.org/pdf/2505.13404v1](http://arxiv.org/pdf/2505.13404v1)**

> **作者:** Nithin Rao Koluguri; Monica Sekoyan; George Zelenfroynd; Sasha Meister; Shuoyang Ding; Sofia Kostandian; He Huang; Nikolay Karpov; Jagadeesh Balam; Vitaly Lavrukhin; Yifan Peng; Sara Papi; Marco Gaido; Alessio Brutti; Boris Ginsburg
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Multi-task and multilingual approaches benefit large models, yet speech processing for low-resource languages remains underexplored due to data scarcity. To address this, we present Granary, a large-scale collection of speech datasets for recognition and translation across 25 European languages. This is the first open-source effort at this scale for both transcription and translation. We enhance data quality using a pseudo-labeling pipeline with segmentation, two-pass inference, hallucination filtering, and punctuation restoration. We further generate translation pairs from pseudo-labeled transcriptions using EuroLLM, followed by a data filtration pipeline. Designed for efficiency, our pipeline processes vast amount of data within hours. We assess models trained on processed data by comparing their performance on previously curated datasets for both high- and low-resource languages. Our findings show that these models achieve similar performance using approx. 50% less data. Dataset will be made available at https://hf.co/datasets/nvidia/Granary
>
---
#### [new 113] Dementia Through Different Eyes: Explainable Modeling of Human and LLM Perceptions for Early Awareness
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗NLP任务，研究人类和LLM如何通过语言感知痴呆症早期迹象。通过分析非专家人类与LLM对文本的直觉判断，构建可解释模型对比两者差异，发现人类依赖有限线索易误判，而LLM特征更贴合临床模式，但均存在漏诊倾向，旨在提升非专家识别能力。**

- **链接: [http://arxiv.org/pdf/2505.13418v1](http://arxiv.org/pdf/2505.13418v1)**

> **作者:** Lotem Peled-Cohen; Maya Zadok; Nitay Calderon; Hila Gonen; Roi Reichart
>
> **摘要:** Cognitive decline often surfaces in language years before diagnosis. It is frequently non-experts, such as those closest to the patient, who first sense a change and raise concern. As LLMs become integrated into daily communication and used over prolonged periods, it may even be an LLM that notices something is off. But what exactly do they notice--and should be noticing--when making that judgment? This paper investigates how dementia is perceived through language by non-experts. We presented transcribed picture descriptions to non-expert humans and LLMs, asking them to intuitively judge whether each text was produced by someone healthy or with dementia. We introduce an explainable method that uses LLMs to extract high-level, expert-guided features representing these picture descriptions, and use logistic regression to model human and LLM perceptions and compare with clinical diagnoses. Our analysis reveals that human perception of dementia is inconsistent and relies on a narrow, and sometimes misleading, set of cues. LLMs, by contrast, draw on a richer, more nuanced feature set that aligns more closely with clinical patterns. Still, both groups show a tendency toward false negatives, frequently overlooking dementia cases. Through our interpretable framework and the insights it provides, we hope to help non-experts better recognize the linguistic signs that matter.
>
---
#### [new 114] Re-identification of De-identified Documents with Autoregressive Infilling
- **分类: cs.CL**

- **简介: 该论文研究文档去标识化的鲁棒性，属于隐私保护对抗任务。针对现有方法可能因遮盖个人信息（PII）不彻底导致身份泄露的问题，提出基于检索增强生成（RAG）的两步式反识别框架：先检索背景知识库中的关联段落，再用自回归填充模型推断被遮盖内容。实验表明，该方法在三个数据集上最高可恢复80%的掩码信息，且背景知识越丰富，恢复精度越高。**

- **链接: [http://arxiv.org/pdf/2505.12859v1](http://arxiv.org/pdf/2505.12859v1)**

> **作者:** Lucas Georges Gabriel Charpentier; Pierre Lison
>
> **备注:** To be presented a ACL 2025, Main, Long paper
>
> **摘要:** Documents revealing sensitive information about individuals must typically be de-identified. This de-identification is often done by masking all mentions of personally identifiable information (PII), thereby making it more difficult to uncover the identity of the person(s) in question. To investigate the robustness of de-identification methods, we present a novel, RAG-inspired approach that attempts the reverse process of re-identification based on a database of documents representing background knowledge. Given a text in which personal identifiers have been masked, the re-identification proceeds in two steps. A retriever first selects from the background knowledge passages deemed relevant for the re-identification. Those passages are then provided to an infilling model which seeks to infer the original content of each text span. This process is repeated until all masked spans are replaced. We evaluate the re-identification on three datasets (Wikipedia biographies, court rulings and clinical notes). Results show that (1) as many as 80% of de-identified text spans can be successfully recovered and (2) the re-identification accuracy increases along with the level of background knowledge.
>
---
#### [new 115] Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down
- **分类: cs.CL**

- **简介: 该论文属于语音识别优化任务，旨在解决Whisper模型在非语音片段产生幻觉的问题。通过分析解码器自注意力头对幻觉的贡献，定位3个核心头部（占比75%），使用非语音数据微调这些头部。最终模型Calm-Whisper在保持识别精度（WER仅增0.1%）的同时减少80%非语音幻觉。**

- **链接: [http://arxiv.org/pdf/2505.12969v1](http://arxiv.org/pdf/2505.12969v1)**

> **作者:** Yingzhi Wang; Anas Alhmoud; Saad Alsahly; Muhammad Alqurishi; Mirco Ravanelli
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** OpenAI's Whisper has achieved significant success in Automatic Speech Recognition. However, it has consistently been found to exhibit hallucination issues, particularly in non-speech segments, which limits its broader application in complex industrial settings. In this paper, we introduce a novel method to reduce Whisper's hallucination on non-speech segments without using any pre- or post-possessing techniques. Specifically, we benchmark the contribution of each self-attentional head in the Whisper-large-v3 decoder to the hallucination problem by performing a head-wise mask. Our findings reveal that only 3 of the 20 heads account for over 75% of the hallucinations on the UrbanSound dataset. We then fine-tune these three crazy heads using a collection of non-speech data. The results show that our best fine-tuned model, namely Calm-Whisper, achieves over 80% reduction in non-speech hallucination with only less than 0.1% WER degradation on LibriSpeech test-clean and test-other.
>
---
#### [new 116] CMLFormer: A Dual Decoder Transformer with Switching Point Learning for Code-Mixed Language Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码混合语言建模任务，旨在解决标准模型难以处理频繁语言切换的问题。提出CMLFormer双解码器Transformer模型，通过共享编码器和多任务预训练（含切换点学习、跨语言结构建模），在增强的Hinglish语料上验证了其识别语言切换点及提升分类性能的效果。**

- **链接: [http://arxiv.org/pdf/2505.12587v1](http://arxiv.org/pdf/2505.12587v1)**

> **作者:** Aditeya Baral; Allen George Ajith; Roshan Nayak; Mrityunjay Abhijeet Bhanja
>
> **摘要:** Code-mixed languages, characterized by frequent within-sentence language transitions, present structural challenges that standard language models fail to address. In this work, we propose CMLFormer, an enhanced multi-layer dual-decoder Transformer with a shared encoder and synchronized decoder cross-attention, designed to model the linguistic and semantic dynamics of code-mixed text. CMLFormer is pre-trained on an augmented Hinglish corpus with switching point and translation annotations with multiple new objectives specifically aimed at capturing switching behavior, cross-lingual structure, and code-mixing complexity. Our experiments show that CMLFormer improves F1 score, precision, and accuracy over other approaches on the HASOC-2021 benchmark under select pre-training setups. Attention analyses further show that it can identify and attend to switching points, validating its sensitivity to code-mixed structure. These results demonstrate the effectiveness of CMLFormer's architecture and multi-task pre-training strategy for modeling code-mixed languages.
>
---
#### [new 117] Thinkless: LLM Learns When to Think
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于高效推理优化任务，旨在解决LLM对所有问题使用复杂推理导致的效率低下问题。提出了Thinkless框架，通过强化学习训练LLM自适应选择简答（<short>）或长链推理（<think>），并设计DeGRPO算法分解控制策略和答案生成目标，在保持精度的同时减少50%-90%长推理使用。**

- **链接: [http://arxiv.org/pdf/2505.13379v1](http://arxiv.org/pdf/2505.13379v1)**

> **作者:** Gongfan Fang; Xinyin Ma; Xinchao Wang
>
> **摘要:** Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at https://github.com/VainF/Thinkless
>
---
#### [new 118] Measuring Information Distortion in Hierarchical Ultra long Novel Generation:The Optimal Expansion Ratio
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **简介: 该论文研究超长小说生成任务，解决LLM生成百万字小说时因压缩导致的信息失真问题。提出分层两阶段生成框架（大纲→详细大纲→正文），通过信息论分析量化压缩扩展比，确定最优大纲长度以平衡信息保留与人力成本。实验验证两阶段方法相比单阶段显著降低语义失真，为LLM长文本创作提供优化策略。**

- **链接: [http://arxiv.org/pdf/2505.12572v1](http://arxiv.org/pdf/2505.12572v1)**

> **作者:** Hanwen Shen; Ting Ying
>
> **摘要:** Writing novels with Large Language Models (LLMs) raises a critical question: how much human-authored outline is necessary to generate high-quality million-word novels? While frameworks such as DOME, Plan&Write, and Long Writer have improved stylistic coherence and logical consistency, they primarily target shorter novels (10k--100k words), leaving ultra-long generation largely unexplored. Drawing on insights from recent text compression methods like LLMZip and LLM2Vec, we conduct an information-theoretic analysis that quantifies distortion occurring when LLMs compress and reconstruct ultra-long novels under varying compression-expansion ratios. We introduce a hierarchical two-stage generation pipeline (outline -> detailed outline -> manuscript) and find an optimal outline length that balances information preservation with human effort. Through extensive experimentation with Chinese novels, we establish that a two-stage hierarchical outline approach significantly reduces semantic distortion compared to single-stage methods. Our findings provide empirically-grounded guidance for authors and researchers collaborating with LLMs to create million-word novels.
>
---
#### [new 119] I'll believe it when I see it: Images increase misinformation sharing in Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）中图像对虚假新闻传播的影响，属于多模态模型行为分析任务。通过构建含政治新闻的多模态数据集和越狱式提示策略，发现图像使虚假新闻分享率提升15%，且人格设定（如黑暗三人格）会加剧该效应，仅Claude-3-Haiku模型具备抗干扰性，揭示了多模态AI的传播风险。**

- **链接: [http://arxiv.org/pdf/2505.13302v1](http://arxiv.org/pdf/2505.13302v1)**

> **作者:** Alice Plebe; Timothy Douglas; Diana Riazi; R. Maria del Rio-Chanona
>
> **摘要:** Large language models are increasingly integrated into news recommendation systems, raising concerns about their role in spreading misinformation. In humans, visual content is known to boost credibility and shareability of information, yet its effect on vision-language models (VLMs) remains unclear. We present the first study examining how images influence VLMs' propensity to reshare news content, whether this effect varies across model families, and how persona conditioning and content attributes modulate this behavior. To support this analysis, we introduce two methodological contributions: a jailbreaking-inspired prompting strategy that elicits resharing decisions from VLMs while simulating users with antisocial traits and political alignments; and a multimodal dataset of fact-checked political news from PolitiFact, paired with corresponding images and ground-truth veracity labels. Experiments across model families reveal that image presence increases resharing rates by 4.8% for true news and 15.0% for false news. Persona conditioning further modulates this effect: Dark Triad traits amplify resharing of false news, whereas Republican-aligned profiles exhibit reduced veracity sensitivity. Of all the tested models, only Claude-3-Haiku demonstrates robustness to visual misinformation. These findings highlight emerging risks in multimodal model behavior and motivate the development of tailored evaluation frameworks and mitigation strategies for personalized AI systems. Code and dataset are available at: https://github.com/3lis/misinfo_vlm
>
---
#### [new 120] Chain-of-Model Learning for Language Model
- **分类: cs.CL**

- **简介: 该论文提出Chain-of-Model（CoM）学习范式，针对语言模型训练效率与推理弹性问题。通过链式隐藏表征（CoR）将各层特征分解为因果关联的子链，实现模型渐进式扩展和动态子模型调用。基于Transformer设计了CoLM框架及KV共享优化的CoLM-Air，在保持性能的同时提升训练扩展性和推理灵活性。**

- **链接: [http://arxiv.org/pdf/2505.11820v1](http://arxiv.org/pdf/2505.11820v1)**

> **作者:** Kaitao Song; Xiaohua Wang; Xu Tan; Huiqiang Jiang; Chengruidong Zhang; Yongliang Shen; Cen LU; Zihao Li; Zifan Song; Caihua Shan; Yansen Wang; Kan Ren; Xiaoqing Zheng; Tao Qin; Yuqing Yang; Dongsheng Li; Lili Qiu
>
> **摘要:** In this paper, we propose a novel learning paradigm, termed Chain-of-Model (CoM), which incorporates the causal relationship into the hidden states of each layer as a chain style, thereby introducing great scaling efficiency in model training and inference flexibility in deployment. We introduce the concept of Chain-of-Representation (CoR), which formulates the hidden states at each layer as a combination of multiple sub-representations (i.e., chains) at the hidden dimension level. In each layer, each chain from the output representations can only view all of its preceding chains in the input representations. Consequently, the model built upon CoM framework can progressively scale up the model size by increasing the chains based on the previous models (i.e., chains), and offer multiple sub-models at varying sizes for elastic inference by using different chain numbers. Based on this principle, we devise Chain-of-Language-Model (CoLM), which incorporates the idea of CoM into each layer of Transformer architecture. Based on CoLM, we further introduce CoLM-Air by introducing a KV sharing mechanism, that computes all keys and values within the first chain and then shares across all chains. This design demonstrates additional extensibility, such as enabling seamless LM switching, prefilling acceleration and so on. Experimental results demonstrate our CoLM family can achieve comparable performance to the standard Transformer, while simultaneously enabling greater flexiblity, such as progressive scaling to improve training efficiency and offer multiple varying model sizes for elastic inference, paving a a new way toward building language models. Our code will be released in the future at: https://github.com/microsoft/CoLM.
>
---
#### [new 121] Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型（LLMs）强化学习（RL）训练中低概率词元梯度过大抑制高概率词元学习的问题，提出Advantage Reweighting和Lopti两种方法，通过削弱低概率词元梯度、强化高概率词元更新，提升RL训练效率，在逻辑推理任务中实现最高46.2%的性能提升。**

- **链接: [http://arxiv.org/pdf/2505.12929v1](http://arxiv.org/pdf/2505.12929v1)**

> **作者:** Zhihe Yang; Xufang Luo; Zilong Wang; Dongqi Han; Zhiyuan He; Dongsheng Li; Yunjian Xu
>
> **备注:** 24 pages, 12 figures
>
> **摘要:** Reinforcement learning (RL) has become a cornerstone for enhancing the reasoning capabilities of large language models (LLMs), with recent innovations such as Group Relative Policy Optimization (GRPO) demonstrating exceptional effectiveness. In this study, we identify a critical yet underexplored issue in RL training: low-probability tokens disproportionately influence model updates due to their large gradient magnitudes. This dominance hinders the effective learning of high-probability tokens, whose gradients are essential for LLMs' performance but are substantially suppressed. To mitigate this interference, we propose two novel methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), both of which effectively attenuate gradients from low-probability tokens while emphasizing parameter updates driven by high-probability tokens. Our approaches promote balanced updates across tokens with varying probabilities, thereby enhancing the efficiency of RL training. Experimental results demonstrate that they substantially improve the performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K Logic Puzzle reasoning tasks. Our implementation is available at https://github.com/zhyang2226/AR-Lopti.
>
---
#### [new 122] PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于角色扮演大语言模型任务，针对现有方法心理建模粗糙、记忆一致性差的问题，提出PsyMem框架。通过26项心理指标细化角色特征，结合显式记忆对齐训练（基于5,414角色和38,962对话数据集），使模型PsyMem-Qwen在人类相似度与角色保真度上超越基线。**

- **链接: [http://arxiv.org/pdf/2505.12814v1](http://arxiv.org/pdf/2505.12814v1)**

> **作者:** Xilong Cheng; Yunxiao Qin; Yuting Tan; Zhengnan Li; Ye Wang; Hongjiang Xiao; Yuan Zhang
>
> **摘要:** Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity.
>
---
#### [new 123] AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AD-AGENT，一个基于大语言模型的多智能体框架，用于端到端异常检测。针对非专家用户难以利用专业库（如PyOD）构建检测流程的问题，通过协调多个代理完成意图解析、数据预处理、模型选择及代码生成，整合不同库形成统一工作流。系统通过共享存储机制实现跨库协作，实验验证其生成脚本的可靠性。**

- **链接: [http://arxiv.org/pdf/2505.12594v1](http://arxiv.org/pdf/2505.12594v1)**

> **作者:** Tiankai Yang; Junjun Liu; Wingchun Siu; Jiahang Wang; Zhuangzhuang Qian; Chanjuan Song; Cheng Cheng; Xiyang Hu; Yue Zhao
>
> **摘要:** Anomaly detection (AD) is essential in areas such as fraud detection, network monitoring, and scientific research. However, the diversity of data modalities and the increasing number of specialized AD libraries pose challenges for non-expert users who lack in-depth library-specific knowledge and advanced programming skills. To tackle this, we present AD-AGENT, an LLM-driven multi-agent framework that turns natural-language instructions into fully executable AD pipelines. AD-AGENT coordinates specialized agents for intent parsing, data preparation, library and model selection, documentation mining, and iterative code generation and debugging. Using a shared short-term workspace and a long-term cache, the agents integrate popular AD libraries like PyOD, PyGOD, and TSLib into a unified workflow. Experiments demonstrate that AD-AGENT produces reliable scripts and recommends competitive models across libraries. The system is open-sourced to support further research and practical applications in AD.
>
---
#### [new 124] From n-gram to Attention: How Model Architectures Learn and Propagate Bias in Language Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型偏见分析任务，旨在探究模型架构与训练数据对偏见传播的影响。通过比较n-gram与Transformer模型，揭示架构设计（如上下文敏感性）、数据时序性及特定偏见（如性取向）的放大效应，强调需从数据和模型源头联合治理偏见。**

- **链接: [http://arxiv.org/pdf/2505.12381v1](http://arxiv.org/pdf/2505.12381v1)**

> **作者:** Mohsinul Kabir; Tasfia Tahsin; Sophia Ananiadou
>
> **备注:** 19 pages
>
> **摘要:** Current research on bias in language models (LMs) predominantly focuses on data quality, with significantly less attention paid to model architecture and temporal influences of data. Even more critically, few studies systematically investigate the origins of bias. We propose a methodology grounded in comparative behavioral theory to interpret the complex interaction between training data and model architecture in bias propagation during language modeling. Building on recent work that relates transformers to n-gram LMs, we evaluate how data, model design choices, and temporal dynamics affect bias propagation. Our findings reveal that: (1) n-gram LMs are highly sensitive to context window size in bias propagation, while transformers demonstrate architectural robustness; (2) the temporal provenance of training data significantly affects bias; and (3) different model architectures respond differentially to controlled bias injection, with certain biases (e.g. sexual orientation) being disproportionately amplified. As language models become ubiquitous, our findings highlight the need for a holistic approach -- tracing bias to its origins across both data and model dimensions, not just symptoms, to mitigate harm.
>
---
#### [new 125] Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation
- **分类: cs.CL**

- **简介: 该论文研究多跳问答（MHQA）任务，探讨语言模型（LMs）在文档顺序调整下的推理性能。解决因果掩码限制模型跨文档推理的问题，实验发现：编码器-解码器模型（如Flan-T5）优于解码器模型；文档顺序与推理链一致时性能最佳；修改因果掩码引入双向注意力可提升解码器模型效果。同时分析注意力权重分布规律，提出启发式优化方法。**

- **链接: [http://arxiv.org/pdf/2505.11754v1](http://arxiv.org/pdf/2505.11754v1)**

> **作者:** Wenyu Huang; Pavlos Vougiouklis; Mirella Lapata; Jeff Z. Pan
>
> **备注:** ACL 2025 main
>
> **摘要:** Multi-hop Question Answering (MHQA) adds layers of complexity to question answering, making it more challenging. When Language Models (LMs) are prompted with multiple search results, they are tasked not only with retrieving relevant information but also employing multi-hop reasoning across the information sources. Although LMs perform well on traditional question-answering tasks, the causal mask can hinder their capacity to reason across complex contexts. In this paper, we explore how LMs respond to multi-hop questions by permuting search results (retrieved documents) under various configurations. Our study reveals interesting findings as follows: 1) Encoder-decoder models, such as the ones in the Flan-T5 family, generally outperform causal decoder-only LMs in MHQA tasks, despite being significantly smaller in size; 2) altering the order of gold documents reveals distinct trends in both Flan T5 models and fine-tuned decoder-only models, with optimal performance observed when the document order aligns with the reasoning chain order; 3) enhancing causal decoder-only models with bi-directional attention by modifying the causal mask can effectively boost their end performance. In addition to the above, we conduct a thorough investigation of the distribution of LM attention weights in the context of MHQA. Our experiments reveal that attention weights tend to peak at higher values when the resulting answer is correct. We leverage this finding to heuristically improve LMs' performance on this task. Our code is publicly available at https://github.com/hwy9855/MultiHopQA-Reasoning.
>
---
#### [new 126] Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型推理效率问题，提出双模型协作框架Long⊗Short，通过区分关键/次要思维链，结合强化学习协同优化。任务属于高效推理优化，解决传统方法等权压缩思维链导致的冗余问题，通过蒙特卡洛分块评估、模型分角色微调及协同强化学习，在保持性能的同时减少80%推理长度。**

- **链接: [http://arxiv.org/pdf/2505.11827v1](http://arxiv.org/pdf/2505.11827v1)**

> **作者:** Yansong Ning; Wei Li; Jun Fang; Naiqiang Tan; Hao Liu
>
> **备注:** In progress
>
> **摘要:** Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at https://github.com/yasNing/Long-otimes-Short/.
>
---
#### [new 127] Shadow-FT: Tuning Instruct via Base
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Shadow-FT框架，解决指令调优模型（INSTRUCT）直接微调效果差的问题。通过微调基础模型（BASE）并移植权重到INSTRUCT模型，无需新增参数，显著提升语言模型在编程、推理等19项任务的性能，支持多模态和偏好优化。**

- **链接: [http://arxiv.org/pdf/2505.12716v1](http://arxiv.org/pdf/2505.12716v1)**

> **作者:** Taiqiang Wu; Runming Yang; Jiayi Li; Pengfei Hu; Ngai Wong; Yujiu Yang
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the INSTRUCT (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired BASE models, the foundation for these INSTRUCT variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). Therefore, we propose a novel Shadow-FT framework to tune the INSTRUCT models by leveraging the corresponding BASE models. The key insight is to fine-tune the BASE model, and then directly graft the learned weight updates to the INSTRUCT model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO). Codes and weights are available at \href{https://github.com/wutaiqiang/Shadow-FT}{Github}.
>
---
#### [new 128] Positional Fragility in LLMs: How Offset Effects Reshape Our Understanding of Memorization Risks
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全研究，旨在解决LLM记忆训练数据导致的版权风险。通过预训练不同规模模型并分析记忆行为，发现"偏移效应"：模型对上下文窗口开头的短前缀敏感，且前缀位置偏移会显著降低逐字记忆。提出"位置脆弱性"概念，揭示模型过度依赖初始标记，并证明调整敏感数据位置可有效抑制记忆泄露和文本退化，挑战了传统评估方法的位置均匀性假设。**

- **链接: [http://arxiv.org/pdf/2505.13171v1](http://arxiv.org/pdf/2505.13171v1)**

> **作者:** Yixuan Xu; Antoine Bosselut; Imanol Schlag
>
> **摘要:** Large language models are known to memorize parts of their training data, posing risk of copyright violations. To systematically examine this risk, we pretrain language models (1B/3B/8B) from scratch on 83B tokens, mixing web-scale data with public domain books used to simulate copyrighted content at controlled frequencies at lengths at least ten times longer than prior work. We thereby identified the offset effect, a phenomenon characterized by two key findings: (1) verbatim memorization is most strongly triggered by short prefixes drawn from the beginning of the context window, with memorization decreasing counterintuitively as prefix length increases; and (2) a sharp decline in verbatim recall when prefix begins offset from the initial tokens of the context window. We attribute this to positional fragility: models rely disproportionately on the earliest tokens in their context window as retrieval anchors, making them sensitive to even slight shifts. We further observe that when the model fails to retrieve memorized content, it often produces degenerated text. Leveraging these findings, we show that shifting sensitive data deeper into the context window suppresses both extractable memorization and degeneration. Our results suggest that positional offset is a critical and previously overlooked axis for evaluating memorization risks, since prior work implicitly assumed uniformity by probing only from the beginning of training sequences.
>
---
#### [new 129] What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts
- **分类: cs.CL; cs.SE**

- **简介: 该论文研究大语言模型(LLM)提示工程中的不完整性问题，属于提示优化任务。针对开发者提示常遗漏关键需求导致模型输出不稳定(变化时准确率下降超20%)的问题，通过实验证明传统优化方法无效，提出需求感知的提示优化机制(性能提升4.8%)，并构建包含需求发现、评估和监控的综合管理框架。**

- **链接: [http://arxiv.org/pdf/2505.13360v1](http://arxiv.org/pdf/2505.13360v1)**

> **作者:** Chenyang Yang; Yike Shi; Qianou Ma; Michael Xieyang Liu; Christian Kästner; Tongshuang Wu
>
> **摘要:** Building LLM-powered software requires developers to communicate their requirements through natural language, but developer prompts are frequently underspecified, failing to fully capture many user-important requirements. In this paper, we present an in-depth analysis of prompt underspecification, showing that while LLMs can often (41.1%) guess unspecified requirements by default, such behavior is less robust: Underspecified prompts are 2x more likely to regress over model or prompt changes, sometimes with accuracy drops by more than 20%. We then demonstrate that simply adding more requirements to a prompt does not reliably improve performance, due to LLMs' limited instruction-following capabilities and competing constraints, and standard prompt optimizers do not offer much help. To address this, we introduce novel requirements-aware prompt optimization mechanisms that can improve performance by 4.8% on average over baselines that naively specify everything in the prompt. Beyond prompt optimization, we envision that effectively managing prompt underspecification requires a broader process, including proactive requirements discovery, evaluation, and monitoring.
>
---
#### [new 130] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ZeroTuning，一种无需训练的大语言模型优化方法，属于推理时调优任务。针对现有注意力调优方法依赖辅助机制导致偏差的问题，通过理论分析和实验发现初始空白令牌对注意力分布的关键调控作用。通过分层调整该令牌的注意力权重，在分类、问答等任务中显著提升模型性能，验证了其高效性和鲁棒性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.11739v1](http://arxiv.org/pdf/2505.11739v1)**

> **作者:** Feijiang Han; Xiaodong Yu; Jianheng Tang; Lyle Ungar
>
> **摘要:** Recently, training-free methods for improving large language models (LLMs) have attracted growing interest, with token-level attention tuning emerging as a promising and interpretable direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability. In this paper, we uncover a surprising and elegant alternative: the semantically empty initial token is a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token's attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher performance on text classification, multiple-choice, and multi-turn conversation tasks across models such as Llama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification, 2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability.
>
---
#### [new 131] AI-generated Text Detection: A Multifaceted Approach to Binary and Multiclass Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测领域，解决区分人/AI文本（二分类）及溯源生成模型（多类分类）问题。针对两项任务提出优化和简化模型，分别取得F1 0.994（任务A第五）和0.627（任务B第五），旨在遏制LLMs滥用风险。**

- **链接: [http://arxiv.org/pdf/2505.11550v1](http://arxiv.org/pdf/2505.11550v1)**

> **作者:** Harika Abburi; Sanmitra Bhattacharya; Edward Bowen; Nirmala Pudota
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in generating text that closely resembles human writing across a wide range of styles and genres. However, such capabilities are prone to potential misuse, such as fake news generation, spam email creation, and misuse in academic assignments. As a result, accurate detection of AI-generated text and identification of the model that generated it are crucial for maintaining the responsible use of LLMs. In this work, we addressed two sub-tasks put forward by the Defactify workshop under AI-Generated Text Detection shared task at the Association for the Advancement of Artificial Intelligence (AAAI 2025): Task A involved distinguishing between human-authored or AI-generated text, while Task B focused on attributing text to its originating language model. For each task, we proposed two neural architectures: an optimized model and a simpler variant. For Task A, the optimized neural architecture achieved fifth place with $F1$ score of 0.994, and for Task B, the simpler neural architecture also ranked fifth place with $F1$ score of 0.627.
>
---
#### [new 132] GUARD: Generation-time LLM Unlearning via Adaptive Restriction and Detection
- **分类: cs.CL**

- **简介: 该论文属于大语言模型选择性遗忘任务，旨在解决传统微调方法导致性能下降的问题。提出GUARD框架，在生成时动态检测并限制遗忘内容，通过提示分类器识别目标，结合词义和语义匹配过滤候选词，避免泄露遗忘知识，同时保持模型生成能力。实验显示其在多个任务中有效平衡遗忘与性能。**

- **链接: [http://arxiv.org/pdf/2505.13312v1](http://arxiv.org/pdf/2505.13312v1)**

> **作者:** Zhijie Deng; Chris Yuhao Liu; Zirui Pang; Xinlei He; Lei Feng; Qi Xuan; Zhaowei Zhu; Jiaheng Wei
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong capabilities in memorizing vast amounts of knowledge across diverse domains. However, the ability to selectively forget specific knowledge is critical for ensuring the safety and compliance of deployed models. Existing unlearning efforts typically fine-tune the model with resources such as forget data, retain data, and a calibration model. These additional gradient steps blur the decision boundary between forget and retain knowledge, making unlearning often at the expense of overall performance. To avoid the negative impact of fine-tuning, it would be better to unlearn solely at inference time by safely guarding the model against generating responses related to the forget target, without destroying the fluency of text generation. In this work, we propose Generation-time Unlearning via Adaptive Restriction and Detection (GUARD), a framework that enables dynamic unlearning during LLM generation. Specifically, we first employ a prompt classifier to detect unlearning targets and extract the corresponding forbidden token. We then dynamically penalize and filter candidate tokens during generation using a combination of token matching and semantic matching, effectively preventing the model from leaking the forgotten content. Experimental results on copyright content unlearning tasks over the Harry Potter dataset and the MUSE benchmark, as well as entity unlearning tasks on the TOFU dataset, demonstrate that GUARD achieves strong forget quality across various tasks while causing almost no degradation to the LLM's general capabilities, striking an excellent trade-off between forgetting and utility.
>
---
#### [new 133] To Bias or Not to Bias: Detecting bias in News with bias-detector
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究媒体偏见检测（自然语言处理任务），旨在解决新闻句子级偏见分类的挑战。通过微调RoBERTa模型并结合BABE数据集，提出优于基线模型的解决方案。使用统计验证方法证明性能提升，并通过注意力机制分析模型聚焦上下文特征。构建了结合现有分类器的混合流程，增强泛化能力与可解释性，但受限于数据集规模。研究为构建鲁棒、可解释的偏见检测系统提供基础。**

- **链接: [http://arxiv.org/pdf/2505.13010v1](http://arxiv.org/pdf/2505.13010v1)**

> **作者:** Himel Ghosh; Ahmed Mosharafa; Georg Groh
>
> **备注:** 7 pages, 5 figures, 2 tables
>
> **摘要:** Media bias detection is a critical task in ensuring fair and balanced information dissemination, yet it remains challenging due to the subjectivity of bias and the scarcity of high-quality annotated data. In this work, we perform sentence-level bias classification by fine-tuning a RoBERTa-based model on the expert-annotated BABE dataset. Using McNemar's test and the 5x2 cross-validation paired t-test, we show statistically significant improvements in performance when comparing our model to a domain-adaptively pre-trained DA-RoBERTa baseline. Furthermore, attention-based analysis shows that our model avoids common pitfalls like oversensitivity to politically charged terms and instead attends more meaningfully to contextually relevant tokens. For a comprehensive examination of media bias, we present a pipeline that combines our model with an already-existing bias-type classifier. Our method exhibits good generalization and interpretability, despite being constrained by sentence-level analysis and dataset size because of a lack of larger and more advanced bias corpora. We talk about context-aware modeling, bias neutralization, and advanced bias type classification as potential future directions. Our findings contribute to building more robust, explainable, and socially responsible NLP systems for media bias detection.
>
---
#### [new 134] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）微调中数据选择效率低的问题，提出无需训练的注意力驱动方法Data Whisperer。其通过少样本上下文学习筛选任务最优训练子集，在多项实验中用10%数据超越全量数据集效果，相比现有方法准确率提升3.1分且提速7.4倍，解决了传统方法资源消耗大且无法充分挖掘模型潜力的问题。**

- **链接: [http://arxiv.org/pdf/2505.12212v1](http://arxiv.org/pdf/2505.12212v1)**

> **作者:** Shaobo Wang; Ziming Wang; Xiangqi Jin; Jize Wang; Jiajun Zhang; Kaixin Li; Zichen Wen; Zhong Li; Conghui He; Xuming Hu; Linfeng Zhang
>
> **备注:** Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
>
> **摘要:** Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup.
>
---
#### [new 135] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation
- **分类: cs.CL**

- **简介: 该论文研究基于双编码器的实体消歧任务，旨在解决知识库链接中设计决策对模型性能的影响问题。通过评估损失函数、相似度指标、标签文本化格式及负采样策略，提出VerbalizED模型，结合上下文标签描述与高效硬负采样，并在AIDA-Yago验证其有效性，最终在ZELDA基准实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.11683v1](http://arxiv.org/pdf/2505.11683v1)**

> **作者:** Susanna Rücker; Alan Akbik
>
> **备注:** Accepted at ACL 2025 (The 63rd Annual Meeting of the Association for Computational Linguistics)
>
> **摘要:** Entity disambiguation (ED) is the task of linking mentions in text to corresponding entries in a knowledge base. Dual Encoders address this by embedding mentions and label candidates in a shared embedding space and applying a similarity metric to predict the correct label. In this work, we focus on evaluating key design decisions for Dual Encoder-based ED, such as its loss function, similarity metric, label verbalization format, and negative sampling strategy. We present the resulting model VerbalizED, a document-level Dual Encoder model that includes contextual label verbalizations and efficient hard negative sampling. Additionally, we explore an iterative prediction variant that aims to improve the disambiguation of challenging data points. Comprehensive experiments on AIDA-Yago validate the effectiveness of our approach, offering insights into impactful design choices that result in a new State-of-the-Art system on the ZELDA benchmark.
>
---
#### [new 136] ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）的个性化工具调用任务，解决现有方法忽视上下文感知导致工具选择效率低的问题。提出ToolSpectrum基准，分析用户画像和环境因素对工具使用的影响，验证个性化提升体验，并揭示当前模型难以协同推理两维度的问题。**

- **链接: [http://arxiv.org/pdf/2505.13176v1](http://arxiv.org/pdf/2505.13176v1)**

> **作者:** Zihao Cheng; Hongru Wang; Zeming Liu; Yuhang Guo; Yuanfang Guo; Yunhong Wang; Haifeng Wang
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** While integrating external tools into large language models (LLMs) enhances their ability to access real-time information and domain-specific services, existing approaches focus narrowly on functional tool selection following user instructions, overlooking the context-aware personalization in tool selection. This oversight leads to suboptimal user satisfaction and inefficient tool utilization, particularly when overlapping toolsets require nuanced selection based on contextual factors. To bridge this gap, we introduce ToolSpectrum, a benchmark designed to evaluate LLMs' capabilities in personalized tool utilization. Specifically, we formalize two key dimensions of personalization, user profile and environmental factors, and analyze their individual and synergistic impacts on tool utilization. Through extensive experiments on ToolSpectrum, we demonstrate that personalized tool utilization significantly improves user experience across diverse scenarios. However, even state-of-the-art LLMs exhibit the limited ability to reason jointly about user profiles and environmental factors, often prioritizing one dimension at the expense of the other. Our findings underscore the necessity of context-aware personalization in tool-augmented LLMs and reveal critical limitations for current models. Our data and code are available at https://github.com/Chengziha0/ToolSpectrum.
>
---
#### [new 137] A Structured Literature Review on Traditional Approaches in Current Natural Language Processing
- **分类: cs.CL**

- **简介: 该论文为文献综述，探讨传统方法在自然语言处理（NLP）五大应用场景（分类、信息抽取、关系抽取、文本简化和摘要）中的现状。研究通过分析近期文献，验证传统技术仍被用作处理流程组件、模型对比基线或核心模型，旨在评估其持续价值及适用场景，为未来技术选择提供参考。**

- **链接: [http://arxiv.org/pdf/2505.12970v1](http://arxiv.org/pdf/2505.12970v1)**

> **作者:** Robin Jegan; Andreas Henrich
>
> **备注:** 14 pages, 1 figure
>
> **摘要:** The continued rise of neural networks and large language models in the more recent past has altered the natural language processing landscape, enabling new approaches towards typical language tasks and achieving mainstream success. Despite the huge success of large language models, many disadvantages still remain and through this work we assess the state of the art in five application scenarios with a particular focus on the future perspectives and sensible application scenarios of traditional and older approaches and techniques. In this paper we survey recent publications in the application scenarios classification, information and relation extraction, text simplification as well as text summarization. After defining our terminology, i.e., which features are characteristic for traditional techniques in our interpretation for the five scenarios, we survey if such traditional approaches are still being used, and if so, in what way they are used. It turns out that all five application scenarios still exhibit traditional models in one way or another, as part of a processing pipeline, as a comparison/baseline to the core model of the respective paper, or as the main model(s) of the paper. For the complete statistics, see https://zenodo.org/records/13683801
>
---
#### [new 138] Talk to Your Slides: Efficient Slide Editing Agent with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于PPT编辑自动化任务，解决现有LLM方法局限于生成而忽视灵活编辑的问题。提出两阶段代理系统：高层LLM解析指令生成计划，底层Python脚本直接操作PPT对象，支持上下文感知编辑，并构建TSBench数据集验证效果，实验显示编辑成功率与效率显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.11604v1](http://arxiv.org/pdf/2505.11604v1)**

> **作者:** Kyudan Jung; Hojun Cho; Jooyeol Yun; Jaehyeok Jang; Jagul Choo
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Existing research on large language models (LLMs) for PowerPoint predominantly focuses on slide generation, overlooking the common yet tedious task of editing existing slides. We introduce Talk-to-Your-Slides, an LLM-powered agent that directly edits slides within active PowerPoint sessions through COM communication. Our system employs a two-level approach: (1) high-level processing where an LLM agent interprets instructions and formulates editing plans, and (2) low-level execution where Python scripts directly manipulate PowerPoint objects. Unlike previous methods relying on predefined operations, our approach enables more flexible and contextually-aware editing. To facilitate evaluation, we present TSBench, a human-annotated dataset of 379 diverse editing instructions with corresponding slide variations. Experimental results demonstrate that Talk-to-Your-Slides significantly outperforms baseline methods in execution success rate, instruction fidelity, and editing efficiency. Our code and benchmark are available at https://anonymous.4open.science/r/talk-to-your-slides/
>
---
#### [new 139] Duluth at SemEval-2025 Task 7: TF-IDF with Optimized Vector Dimensions for Multilingual Fact-Checked Claim Retrieval
- **分类: cs.CL; 68T50**

- **简介: 该论文针对SemEval-2025任务7的多语言事实核查声明检索任务，研究传统TF-IDF方法在跨语言场景的应用。通过优化向量维度（15000词表）和分词策略，系统在10种语言上获得平均0.69的测试集success@10分数，证明传统方法在资源受限场景仍具竞争力，但落后于顶尖神经模型（0.96）。**

- **链接: [http://arxiv.org/pdf/2505.12616v1](http://arxiv.org/pdf/2505.12616v1)**

> **作者:** Shujauddin Syed; Ted Pedersen
>
> **备注:** SemEval-2025
>
> **摘要:** This paper presents the Duluth approach to the SemEval-2025 Task 7 on Multilingual and Crosslingual Fact-Checked Claim Retrieval. We implemented a TF-IDF-based retrieval system with experimentation on vector dimensions and tokenization strategies. Our best-performing configuration used word-level tokenization with a vocabulary size of 15,000 features, achieving an average success@10 score of 0.78 on the development set and 0.69 on the test set across ten languages. Our system showed stronger performance on higher-resource languages but still lagged significantly behind the top-ranked system, which achieved 0.96 average success@10. Our findings suggest that though advanced neural architectures are increasingly dominant in multilingual retrieval tasks, properly optimized traditional methods like TF-IDF remain competitive baselines, especially in limited compute resource scenarios.
>
---
#### [new 140] The power of text similarity in identifying AI-LLM paraphrased documents: The case of BBC news articles and ChatGPT
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI生成文本检测任务，解决ChatGPT改写新闻导致的侵权问题。提出基于模式相似性的非深度学习方法，识别AI改写并溯源至ChatGPT。使用BBC新闻和对应ChatGPT改写构建数据集验证，检测指标均超96%。**

- **链接: [http://arxiv.org/pdf/2505.12405v1](http://arxiv.org/pdf/2505.12405v1)**

> **作者:** Konstantinos Xylogiannopoulos; Petros Xanthopoulos; Panagiotis Karampelas; Georgios Bakamitsos
>
> **摘要:** Generative AI paraphrased text can be used for copyright infringement and the AI paraphrased content can deprive substantial revenue from original content creators. Despite this recent surge of malicious use of generative AI, there are few academic publications that research this threat. In this article, we demonstrate the ability of pattern-based similarity detection for AI paraphrased news recognition. We propose an algorithmic scheme, which is not limited to detect whether an article is an AI paraphrase, but, more importantly, to identify that the source of infringement is the ChatGPT. The proposed method is tested with a benchmark dataset specifically created for this task that incorporates real articles from BBC, incorporating a total of 2,224 articles across five different news categories, as well as 2,224 paraphrased articles created with ChatGPT. Results show that our pattern similarity-based method, that makes no use of deep learning, can detect ChatGPT assisted paraphrased articles at percentages 96.23% for accuracy, 96.25% for precision, 96.21% for sensitivity, 96.25% for specificity and 96.23% for F1 score.
>
---
#### [new 141] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement
- **分类: cs.CL**

- **简介: 该论文属于大语言模型安全防御任务，针对LLMs识别恶意提问但生成不安全回复的检测-生成能力不一致问题，提出无需训练的SAGE框架。通过判别分析模块和响应模块增强模型对抗越狱攻击的防御能力，实验证明其高效性（平均99%防御成功率）并保持通用性能，同时揭示了安全机制的内在机理。**

- **链接: [http://arxiv.org/pdf/2505.12060v1](http://arxiv.org/pdf/2505.12060v1)**

> **作者:** Peng Ding; Jun Kuang; Zongyu Wang; Xuezhi Cao; Xunliang Cai; Jiajun Chen; Shujian Huang
>
> **备注:** Acccepted by ACL 2025 Findings, 21 pages, 9 figures, 14 tables
>
> **摘要:** Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at https://github.com/NJUNLP/SAGE.
>
---
#### [new 142] Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing
- **分类: cs.CL**

- **简介: 该论文研究知识编辑任务，旨在解决现有方法更新语言模型知识时存在欺骗性（表面编辑）的问题。通过机制分析，发现早期层残差流和后期层特定注意力模块是导致模型仍生成原知识的主因，并验证了相关注意力头及左奇异向量对表面编辑的影响，扩展至浅层遗忘任务验证结论普适性。**

- **链接: [http://arxiv.org/pdf/2505.12636v1](http://arxiv.org/pdf/2505.12636v1)**

> **作者:** Jiakuan Xie; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** Accepted by ACL 2025 main
>
> **摘要:** Knowledge editing, which aims to update the knowledge encoded in language models, can be deceptive. Despite the fact that many existing knowledge editing algorithms achieve near-perfect performance on conventional metrics, the models edited by them are still prone to generating original knowledge. This paper introduces the concept of "superficial editing" to describe this phenomenon. Our comprehensive evaluation reveals that this issue presents a significant challenge to existing algorithms. Through systematic investigation, we identify and validate two key factors contributing to this issue: (1) the residual stream at the last subject position in earlier layers and (2) specific attention modules in later layers. Notably, certain attention heads in later layers, along with specific left singular vectors in their output matrices, encapsulate the original knowledge and exhibit a causal relationship with superficial editing. Furthermore, we extend our analysis to the task of superficial unlearning, where we observe consistent patterns in the behavior of specific attention heads and their corresponding left singular vectors, thereby demonstrating the robustness and broader applicability of our methodology and conclusions. Our code is available here.
>
---
#### [new 143] SLOT: Sample-specific Language Model Optimization at Test-time
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决现有模型处理复杂指令时性能不足的问题。提出SLOT方法，在测试时通过少量优化步骤更新轻量级样本特定参数向量（添加至隐藏层），仅用输入提示的交叉熵损失微调，提升模型对单个指令的响应能力。实验显示其在多个基准和模型上显著提升效果。**

- **链接: [http://arxiv.org/pdf/2505.12392v1](http://arxiv.org/pdf/2505.12392v1)**

> **作者:** Yang Hu; Xingyu Zhang; Xueji Fang; Zhiyang Chen; Xiao Wang; Huatian Zhang; Guojun Qi
>
> **摘要:** We propose SLOT (Sample-specific Language Model Optimization at Test-time), a novel and parameter-efficient test-time inference approach that enhances a language model's ability to more accurately respond to individual prompts. Existing Large Language Models (LLMs) often struggle with complex instructions, leading to poor performances on those not well represented among general samples. To address this, SLOT conducts few optimization steps at test-time to update a light-weight sample-specific parameter vector. It is added to the final hidden layer before the output head, and enables efficient adaptation by caching the last layer features during per-sample optimization. By minimizing the cross-entropy loss on the input prompt only, SLOT helps the model better aligned with and follow each given instruction. In experiments, we demonstrate that our method outperforms the compared models across multiple benchmarks and LLMs. For example, Qwen2.5-7B with SLOT achieves an accuracy gain of 8.6% on GSM8K from 57.54% to 66.19%, while DeepSeek-R1-Distill-Llama-70B with SLOT achieves a SOTA accuracy of 68.69% on GPQA among 70B-level models. Our code is available at https://github.com/maple-research-lab/SLOT.
>
---
#### [new 144] Multilingual Prompt Engineering in Large Language Models: A Survey Across NLP Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文为综述研究，聚焦多语言提示工程在大型语言模型（LLMs）中的应用，旨在提升LLMs跨语言NLP任务的表现而无需重训练。通过系统分类39种提示技术，分析30个多语言任务（覆盖250种语言）和36篇文献，总结方法分布、语言资源（高/低资源）影响及最优方案，为跨语言NLP提供工程实践参考。**

- **链接: [http://arxiv.org/pdf/2505.11665v1](http://arxiv.org/pdf/2505.11665v1)**

> **作者:** Shubham Vatsal; Harsh Dubey; Aditi Singh
>
> **摘要:** Large language models (LLMs) have demonstrated impressive performance across a wide range of Natural Language Processing (NLP) tasks. However, ensuring their effectiveness across multiple languages presents unique challenges. Multilingual prompt engineering has emerged as a key approach to enhance LLMs' capabilities in diverse linguistic settings without requiring extensive parameter re-training or fine-tuning. With growing interest in multilingual prompt engineering over the past two to three years, researchers have explored various strategies to improve LLMs' performance across languages and NLP tasks. By crafting structured natural language prompts, researchers have successfully extracted knowledge from LLMs across different languages, making these techniques an accessible pathway for a broader audience, including those without deep expertise in machine learning, to harness the capabilities of LLMs. In this paper, we survey and categorize different multilingual prompting techniques based on the NLP tasks they address across a diverse set of datasets that collectively span around 250 languages. We further highlight the LLMs employed, present a taxonomy of approaches and discuss potential state-of-the-art (SoTA) methods for specific multilingual datasets. Additionally, we derive a range of insights across language families and resource levels (high-resource vs. low-resource), including analyses such as the distribution of NLP tasks by language resource type and the frequency of prompting methods across different language families. Our survey reviews 36 research papers covering 39 prompting techniques applied to 30 multilingual NLP tasks, with the majority of these studies published in the last two years.
>
---
#### [new 145] $K$-MSHC: Unmasking Minimally Sufficient Head Circuits in Large Language Models with Experiments on Syntactic Classification Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型可解释性任务，旨在识别中等规模语言模型中驱动特定能力的核心注意力头。研究者提出K-MSHC方法及高效算法Search-K-MSHC，通过在Gemma-9B上分析句法任务，发现不同任务依赖的头部电路分布特征：语法任务集中于浅层，算术验证分散，应用题深浅层均活跃，并揭示任务间存在弱/强共享头与专用"超级头"的混合计算模式。**

- **链接: [http://arxiv.org/pdf/2505.12268v1](http://arxiv.org/pdf/2505.12268v1)**

> **作者:** Pratim Chowdhary
>
> **摘要:** Understanding which neural components drive specific capabilities in mid-sized language models ($\leq$10B parameters) remains a key challenge. We introduce the $(\bm{K}, \epsilon)$-Minimum Sufficient Head Circuit ($K$-MSHC), a methodology to identify minimal sets of attention heads crucial for classification tasks as well as Search-K-MSHC, an efficient algorithm for discovering these circuits. Applying our Search-K-MSHC algorithm to Gemma-9B, we analyze three syntactic task families: grammar acceptability, arithmetic verification, and arithmetic word problems. Our findings reveal distinct task-specific head circuits, with grammar tasks predominantly utilizing early layers, word problems showing pronounced activity in both shallow and deep regions, and arithmetic verification demonstrating a more distributed pattern across the network. We discover non-linear circuit overlap patterns, where different task pairs share computational components at varying levels of importance. While grammar and arithmetic share many "weak" heads, arithmetic and word problems share more consistently critical "strong" heads. Importantly, we find that each task maintains dedicated "super-heads" with minimal cross-task overlap, suggesting that syntactic and numerical competencies emerge from specialized yet partially reusable head circuits.
>
---
#### [new 146] Evaluating the Performance of RAG Methods for Conversational AI in the Airport Domain
- **分类: cs.CL**

- **简介: 该论文属于对话系统的检索增强生成（RAG）优化任务，旨在解决机场动态场景下AI问答的准确性与安全性问题。研究对比了传统RAG、SQL RAG和知识图谱RAG三种方法，发现知识图谱RAG准确率达91.49%且推理能力强，推荐其与SQL RAG共同减少幻觉风险，提升机场自动化服务可靠性。**

- **链接: [http://arxiv.org/pdf/2505.13006v1](http://arxiv.org/pdf/2505.13006v1)**

> **作者:** Yuyang Li; Philip J. M. Kerbusch; Raimon H. R. Pruim; Tobias Käfer
>
> **备注:** Accepted by NAACL 2025 industry track
>
> **摘要:** Airports from the top 20 in terms of annual passengers are highly dynamic environments with thousands of flights daily, and they aim to increase the degree of automation. To contribute to this, we implemented a Conversational AI system that enables staff in an airport to communicate with flight information systems. This system not only answers standard airport queries but also resolves airport terminology, jargon, abbreviations, and dynamic questions involving reasoning. In this paper, we built three different Retrieval-Augmented Generation (RAG) methods, including traditional RAG, SQL RAG, and Knowledge Graph-based RAG (Graph RAG). Experiments showed that traditional RAG achieved 84.84% accuracy using BM25 + GPT-4 but occasionally produced hallucinations, which is risky to airport safety. In contrast, SQL RAG and Graph RAG achieved 80.85% and 91.49% accuracy respectively, with significantly fewer hallucinations. Moreover, Graph RAG was especially effective for questions that involved reasoning. Based on our observations, we thus recommend SQL RAG and Graph RAG are better for airport environments, due to fewer hallucinations and the ability to handle dynamic questions.
>
---
#### [new 147] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究大语言模型（LLMs）在古典语言（梵语、古希腊语、拉丁语）的跨语言零样本泛化能力，属于自然语言理解任务。通过命名实体识别、机器翻译及梵语问答任务，分析模型规模对性能的影响。实验表明：大模型（如GPT-4o）在域外数据表现优于小模型，检索增强方法提升问答效果，验证模型规模是跨语言泛化的关键因素。**

- **链接: [http://arxiv.org/pdf/2505.13173v1](http://arxiv.org/pdf/2505.13173v1)**

> **作者:** V. S. D. S. Mahesh Akavarapu; Hrishikesh Terdalkar; Pramit Bhattacharyya; Shubhangi Agarwal; Vishakha Deulgaonkar; Pralay Manna; Chaitali Dangarikar; Arnab Bhattacharya
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across diverse tasks and languages. In this study, we focus on natural language understanding in three classical languages -- Sanskrit, Ancient Greek and Latin -- to investigate the factors affecting cross-lingual zero-shot generalization. First, we explore named entity recognition and machine translation into English. While LLMs perform equal to or better than fine-tuned baselines on out-of-domain data, smaller models often struggle, especially with niche or abstract entity types. In addition, we concentrate on Sanskrit by presenting a factoid question-answering (QA) dataset and show that incorporating context via retrieval-augmented generation approach significantly boosts performance. In contrast, we observe pronounced performance drops for smaller LLMs across these QA tasks. These results suggest model scale as an important factor influencing cross-lingual generalization. Assuming that models used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical languages, our findings provide insights into how LLMs may generalize on these languages and their consequent utility in classical studies.
>
---
#### [new 148] Stronger Together: Unleashing the Social Impact of Hate Speech Research
- **分类: cs.CL**

- **简介: 该论文属于社会计算与语言学的交叉研究，旨在解决网络仇恨言论的社会危害问题。作者主张将研究重心从纯技术方案转向结合社会方法，倡导语言学家与NLP学者联合社区和政策制定者，利用语言学理论促进数字包容，缩小数字鸿沟。**

- **链接: [http://arxiv.org/pdf/2505.13251v1](http://arxiv.org/pdf/2505.13251v1)**

> **作者:** Sidney Wong
>
> **备注:** Accepted Proceedings of the Linguistic Society of America 2025 Annual Meeting
>
> **摘要:** The advent of the internet has been both a blessing and a curse for once marginalised communities. When used well, the internet can be used to connect and establish communities crossing different intersections; however, it can also be used as a tool to alienate people and communities as well as perpetuate hate, misinformation, and disinformation especially on social media platforms. We propose steering hate speech research and researchers away from pre-existing computational solutions and consider social methods to inform social solutions to address this social problem. In a similar way linguistics research can inform language planning policy, linguists should apply what we know about language and society to mitigate some of the emergent risks and dangers of anti-social behaviour in digital spaces. We argue linguists and NLP researchers can play a principle role in unleashing the social impact potential of linguistics research working alongside communities, advocates, activists, and policymakers to enable equitable digital inclusion and to close the digital divide.
>
---
#### [new 149] HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文针对大语言模型推理效率问题，提出HeteroSpec框架，通过动态资源分配优化推测解码。利用累积元路径Top-K熵识别可预测上下文，结合熵分区策略实现自适应计算扩展/剪枝，解决传统方法忽视语言复杂度异质性的缺陷，在5个基准测试中实现4.26倍加速，无需重训练且兼容其他加速技术。**

- **链接: [http://arxiv.org/pdf/2505.13254v1](http://arxiv.org/pdf/2505.13254v1)**

> **作者:** Siran Liu; Yang Ye; Qianchao Zhu; Zheng Cao; Yongchao He
>
> **摘要:** Autoregressive decoding, the standard approach for Large Language Model (LLM) inference, remains a significant bottleneck due to its sequential nature. While speculative decoding algorithms mitigate this inefficiency through parallel verification, they fail to exploit the inherent heterogeneity in linguistic complexity, a key factor leading to suboptimal resource allocation. We address this by proposing HeteroSpec, a heterogeneity-adaptive speculative decoding framework that dynamically optimizes computational resource allocation based on linguistic context complexity. HeteroSpec introduces two key mechanisms: (1) A novel cumulative meta-path Top-$K$ entropy metric for efficiently identifying predictable contexts. (2) A dynamic resource allocation strategy based on data-driven entropy partitioning, enabling adaptive speculative expansion and pruning tailored to local context difficulty. Evaluated on five public benchmarks and four models, HeteroSpec achieves an average speedup of 4.26$\times$. It consistently outperforms state-of-the-art EAGLE-3 across speedup rates, average acceptance length, and verification cost. Notably, HeteroSpec requires no draft model retraining, incurs minimal overhead, and is orthogonal to other acceleration techniques. It demonstrates enhanced acceleration with stronger draft models, establishing a new paradigm for context-aware LLM inference acceleration.
>
---
#### [new 150] The taggedPBC: Annotating a massive parallel corpus for crosslinguistic investigations
- **分类: cs.CL**

- **简介: 该论文属于语料库构建任务，旨在解决跨语言研究中数据规模与语言覆盖度不平衡的问题。作者创建了taggedPBC平行语料库，涵盖1,500+语言的1,800+句子的词性标注数据，通过自动标注验证其与现有工具和人工标注的一致性，并提出N1比率特征用于词序分类，为跨语言分析提供大规模资源支持。**

- **链接: [http://arxiv.org/pdf/2505.12560v1](http://arxiv.org/pdf/2505.12560v1)**

> **作者:** Hiram Ring
>
> **摘要:** Existing datasets available for crosslinguistic investigations have tended to focus on large amounts of data for a small group of languages or a small amount of data for a large number of languages. This means that claims based on these datasets are limited in what they reveal about universal properties of the human language faculty. While this has begun to change through the efforts of projects seeking to develop tagged corpora for a large number of languages, such efforts are still constrained by limits on resources. The current paper reports on a large automatically tagged parallel dataset which has been developed to partially address this issue. The taggedPBC contains more than 1,800 sentences of pos-tagged parallel text data from over 1,500 languages, representing 133 language families and 111 isolates, dwarfing previously available resources. The accuracy of tags in this dataset is shown to correlate well with both existing SOTA taggers for high-resource languages (SpaCy, Trankit) as well as hand-tagged corpora (Universal Dependencies Treebanks). Additionally, a novel measure derived from this dataset, the N1 ratio, correlates with expert determinations of word order in three typological databases (WALS, Grambank, Autotyp) such that a Gaussian Naive Bayes classifier trained on this feature can accurately identify basic word order for languages not in those databases. While much work is still needed to expand and develop this dataset, the taggedPBC is an important step to enable corpus-based crosslinguistic investigations, and is made available for research and collaboration via GitHub.
>
---
#### [new 151] KIT's Offline Speech Translation and Instruction Following Submission for IWSLT 2025
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于离线语音翻译和指令跟随任务，旨在提升多任务性能。针对离线翻译，提出多语音识别系统融合结合两步翻译及文档级优化的流程；针对指令跟随，开发端到端语音-LLM整合模型，均利用大语言模型增强输出质量。**

- **链接: [http://arxiv.org/pdf/2505.13036v1](http://arxiv.org/pdf/2505.13036v1)**

> **作者:** Sai Koneru; Maike Züfle; Thai-Binh Nguyen; Seymanur Akti; Jan Niehues; Alexander Waibel
>
> **摘要:** The scope of the International Workshop on Spoken Language Translation (IWSLT) has recently broadened beyond traditional Speech Translation (ST) to encompass a wider array of tasks, including Speech Question Answering and Summarization. This shift is partly driven by the growing capabilities of modern systems, particularly with the success of Large Language Models (LLMs). In this paper, we present the Karlsruhe Institute of Technology's submissions for the Offline ST and Instruction Following (IF) tracks, where we leverage LLMs to enhance performance across all tasks. For the Offline ST track, we propose a pipeline that employs multiple automatic speech recognition systems, whose outputs are fused using an LLM with document-level context. This is followed by a two-step translation process, incorporating additional refinement step to improve translation quality. For the IF track, we develop an end-to-end model that integrates a speech encoder with an LLM to perform a wide range of instruction-following tasks. We complement it with a final document-level refinement stage to further enhance output quality by using contextual information.
>
---
#### [new 152] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions
- **分类: cs.CL**

- **简介: 该论文研究社会经济地位（SES）对语言技术使用差异的影响，属于社会计算任务。通过调查1000名不同SES用户并分析6482条LLM交互数据，发现高SES群体倾向抽象表达和特定主题，低SES用户更拟人化且语言具体，揭示AI使用差异加剧数字鸿沟，强调需针对性优化技术设计以减少不平等。**

- **链接: [http://arxiv.org/pdf/2505.12158v1](http://arxiv.org/pdf/2505.12158v1)**

> **作者:** Elisa Bassignana; Amanda Cercas Curry; Dirk Hovy
>
> **备注:** Accepted at ACL Main 2025
>
> **摘要:** Socioeconomic status (SES) fundamentally influences how people interact with each other and more recently, with digital technologies like Large Language Models (LLMs). While previous research has highlighted the interaction between SES and language technology, it was limited by reliance on proxy metrics and synthetic data. We survey 1,000 individuals from diverse socioeconomic backgrounds about their use of language technologies and generative AI, and collect 6,482 prompts from their previous interactions with LLMs. We find systematic differences across SES groups in language technology usage (i.e., frequency, performed tasks), interaction styles, and topics. Higher SES entails a higher level of abstraction, convey requests more concisely, and topics like 'inclusivity' and 'travel'. Lower SES correlates with higher anthropomorphization of LLMs (using ''hello'' and ''thank you'') and more concrete language. Our findings suggest that while generative language technologies are becoming more accessible to everyone, socioeconomic linguistic differences still stratify their use to exacerbate the digital divide. These differences underscore the importance of considering SES in developing language technologies to accommodate varying linguistic needs rooted in socioeconomic factors and limit the AI Gap across SES groups.
>
---
#### [new 153] Counterspeech the ultimate shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning
- **分类: cs.CL**

- **简介: 该论文属于多属性条件文本生成任务，旨在解决现有反仇恨言论生成模型仅考虑单一属性（如意图）的问题。提出HiPPrO框架，通过分层前缀学习和偏好优化融合意图与情感双属性，在扩展数据集上实现38%意图符合度提升及ROUGE指标进步，生成更有效的反言论。**

- **链接: [http://arxiv.org/pdf/2505.11958v1](http://arxiv.org/pdf/2505.11958v1)**

> **作者:** Aswini Kumar Padhi; Anil Bandhakavi; Tanmoy Chakraborty
>
> **摘要:** Counterspeech has proven to be a powerful tool to combat hate speech online. Previous studies have focused on generating counterspeech conditioned only on specific intents (single attributed). However, a holistic approach considering multiple attributes simultaneously can yield more nuanced and effective responses. Here, we introduce HiPPrO, Hierarchical Prefix learning with Preference Optimization, a novel two-stage framework that utilizes the effectiveness of attribute-specific prefix embedding spaces hierarchically optimized during the counterspeech generation process in the first phase. Thereafter, we incorporate both reference and reward-free preference optimization to generate more constructive counterspeech. Furthermore, we extend IntentCONANv2 by annotating all 13,973 counterspeech instances with emotion labels by five annotators. HiPPrO leverages hierarchical prefix optimization to integrate these dual attributes effectively. An extensive evaluation demonstrates that HiPPrO achieves a ~38 % improvement in intent conformity and a ~3 %, ~2 %, ~3 % improvement in Rouge-1, Rouge-2, and Rouge-L, respectively, compared to several baseline models. Human evaluations further substantiate the superiority of our approach, highlighting the enhanced relevance and appropriateness of the generated counterspeech. This work underscores the potential of multi-attribute conditioning in advancing the efficacy of counterspeech generation systems.
>
---
#### [new 154] JNLP at SemEval-2025 Task 11: Cross-Lingual Multi-Label Emotion Detection Using Generative Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文参与SemEval-2025任务11，解决跨语言多标签情感检测及情感强度分析问题。针对多语言挑战，结合微调BERT和指令调优生成模型，提出直接映射（base）与逐类别建模（pairwise）两种多标签分类方法，在10种语言中实现前四名性能，其中印地语排名第一。**

- **链接: [http://arxiv.org/pdf/2505.13244v1](http://arxiv.org/pdf/2505.13244v1)**

> **作者:** Jieying Xue; Phuong Minh Nguyen; Minh Le Nguyen; Xin Liu
>
> **备注:** Published in The 19th International Workshop on Semantic Evaluation (SemEval-2025)
>
> **摘要:** With the rapid advancement of global digitalization, users from different countries increasingly rely on social media for information exchange. In this context, multilingual multi-label emotion detection has emerged as a critical research area. This study addresses SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. Our paper focuses on two sub-tracks of this task: (1) Track A: Multi-label emotion detection, and (2) Track B: Emotion intensity. To tackle multilingual challenges, we leverage pre-trained multilingual models and focus on two architectures: (1) a fine-tuned BERT-based classification model and (2) an instruction-tuned generative LLM. Additionally, we propose two methods for handling multi-label classification: the base method, which maps an input directly to all its corresponding emotion labels, and the pairwise method, which models the relationship between the input text and each emotion category individually. Experimental results demonstrate the strong generalization ability of our approach in multilingual emotion recognition. In Track A, our method achieved Top 4 performance across 10 languages, ranking 1st in Hindi. In Track B, our approach also secured Top 5 performance in 7 languages, highlighting its simplicity and effectiveness\footnote{Our code is available at https://github.com/yingjie7/mlingual_multilabel_emo_detection.
>
---
#### [new 155] Tianyi: A Traditional Chinese Medicine all-rounder language model and its Real-World Clinical Practice
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于中医药AI应用任务，旨在解决现有大模型缺乏中医专业性、部署困难及幻觉问题。作者开发了76亿参数的天颐模型，通过渐进学习整合中医典籍、临床数据等，并构建TCMEval评估体系，验证其作为中医AI助手在诊疗和研究中的实用价值。**

- **链接: [http://arxiv.org/pdf/2505.13156v1](http://arxiv.org/pdf/2505.13156v1)**

> **作者:** Zhi Liu; Tao Yang; Jing Wang; Yexin Chen; Zhan Gao; Jiaxi Yang; Kui Chen; Bingji Lu; Xiaochen Li; Changyong Luo; Yan Li; Xiaohong Gu; Peng Cao
>
> **备注:** 23 pages, 4 figures, and 1 tables
>
> **摘要:** Natural medicines, particularly Traditional Chinese Medicine (TCM), are gaining global recognition for their therapeutic potential in addressing human symptoms and diseases. TCM, with its systematic theories and extensive practical experience, provides abundant resources for healthcare. However, the effective application of TCM requires precise syndrome diagnosis, determination of treatment principles, and prescription formulation, which demand decades of clinical expertise. Despite advancements in TCM-based decision systems, machine learning, and deep learning research, limitations in data and single-objective constraints hinder their practical application. In recent years, large language models (LLMs) have demonstrated potential in complex tasks, but lack specialization in TCM and face significant challenges, such as too big model scale to deploy and issues with hallucination. To address these challenges, we introduce Tianyi with 7.6-billion-parameter LLM, a model scale proper and specifically designed for TCM, pre-trained and fine-tuned on diverse TCM corpora, including classical texts, expert treatises, clinical records, and knowledge graphs. Tianyi is designed to assimilate interconnected and systematic TCM knowledge through a progressive learning manner. Additionally, we establish TCMEval, a comprehensive evaluation benchmark, to assess LLMs in TCM examinations, clinical tasks, domain-specific question-answering, and real-world trials. The extensive evaluations demonstrate the significant potential of Tianyi as an AI assistant in TCM clinical practice and research, bridging the gap between TCM knowledge and practical application.
>
---
#### [new 156] PromptPrism: A Linguistically-Inspired Taxonomy for Prompts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的prompt工程任务，旨在解决缺乏系统化prompt分析框架的问题。提出PromptPrism语言学分类法，从功能结构、语义组件和句法模式三个层次解析prompt，并应用于自动优化提示质量、多维度数据集分析和敏感性测试，验证了框架在提升模型性能及可解释性方面的有效性。**

- **链接: [http://arxiv.org/pdf/2505.12592v1](http://arxiv.org/pdf/2505.12592v1)**

> **作者:** Sullam Jeoung; Yueyan Chen; Yi Zhang; Shuai Wang; Haibo Ding; Lin Lee Cheong
>
> **摘要:** Prompts are the interface for eliciting the capabilities of large language models (LLMs). Understanding their structure and components is critical for analyzing LLM behavior and optimizing performance. However, the field lacks a comprehensive framework for systematic prompt analysis and understanding. We introduce PromptPrism, a linguistically-inspired taxonomy that enables prompt analysis across three hierarchical levels: functional structure, semantic component, and syntactic pattern. We show the practical utility of PromptPrism by applying it to three applications: (1) a taxonomy-guided prompt refinement approach that automatically improves prompt quality and enhances model performance across a range of tasks; (2) a multi-dimensional dataset profiling method that extracts and aggregates structural, semantic, and syntactic characteristics from prompt datasets, enabling comprehensive analysis of prompt distributions and patterns; (3) a controlled experimental framework for prompt sensitivity analysis by quantifying the impact of semantic reordering and delimiter modifications on LLM performance. Our experimental results validate the effectiveness of our taxonomy across these applications, demonstrating that PromptPrism provides a foundation for refining, profiling, and analyzing prompts.
>
---
#### [new 157] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于高效知识蒸馏任务，旨在降低小语言模型(SLMs)训练成本。针对现有方法的信息损失、低效对齐和FFN激活利用不足问题，提出低秩克隆(LRC)方法，通过联合低秩投影实现软剪枝和激活对齐，最大化知识迁移。实验显示仅用200亿token即超越万亿token训练的SOTA模型，提升千倍效率。**

- **链接: [http://arxiv.org/pdf/2505.12781v1](http://arxiv.org/pdf/2505.12781v1)**

> **作者:** Jitai Hao; Qiang Huang; Hao Liu; Xinyan Xiao; Zhaochun Ren; Jun Yu
>
> **摘要:** Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llama-3.2-3B-Instruct, Qwen2.5-3B/7B-Instruct) show that LRC matches or surpasses state-of-the-art models trained on trillions of tokens--while using only 20B tokens, achieving over 1,000x training efficiency. Our codes and model checkpoints are available at https://github.com/CURRENTF/LowRankClone and https://huggingface.co/collections/JitaiHao/low-rank-clone-lrc-6828389e96a93f1d4219dfaf.
>
---
#### [new 158] ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于图表问答任务，旨在解决大型视觉-语言模型（LVLM）视觉推理能力不足的问题。通过构建ChartMuseum基准（含1,162真实图表问题），揭示模型在视觉复杂场景下性能显著低于人类（最佳模型63% vs 人类93%），并验证视觉推理是当前LVLM的主要瓶颈。**

- **链接: [http://arxiv.org/pdf/2505.13444v1](http://arxiv.org/pdf/2505.13444v1)**

> **作者:** Liyan Tang; Grace Kim; Xinyu Zhao; Thom Lake; Wenxuan Ding; Fangcong Yin; Prasann Singhal; Manya Wadhwa; Zeyu Leo Liu; Zayne Sprague; Ramya Namuduri; Bodun Hu; Juan Diego Rodriguez; Puyuan Peng; Greg Durrett
>
> **摘要:** Chart understanding presents a unique challenge for large vision-language models (LVLMs), as it requires the integration of sophisticated textual and visual reasoning capabilities. However, current LVLMs exhibit a notable imbalance between these skills, falling short on visual reasoning that is difficult to perform in text. We conduct a case study using a synthetic dataset solvable only through visual reasoning and show that model performance degrades significantly with increasing visual complexity, while human performance remains robust. We then introduce ChartMuseum, a new Chart Question Answering (QA) benchmark containing 1,162 expert-annotated questions spanning multiple reasoning types, curated from real-world charts across 184 sources, specifically built to evaluate complex visual and textual reasoning. Unlike prior chart understanding benchmarks -- where frontier models perform similarly and near saturation -- our benchmark exposes a substantial gap between model and human performance, while effectively differentiating model capabilities: although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct achieves only 38.5%. Moreover, on questions requiring primarily visual reasoning, all models experience a 35%-55% performance drop from text-reasoning-heavy question performance. Lastly, our qualitative error analysis reveals specific categories of visual reasoning that are challenging for current LVLMs.
>
---
#### [new 159] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言建模任务，旨在解决现有模型依赖离散化分层结构导致的误差和复杂性问题。作者提出SLED方法，通过连续潜在空间编码语音波形并采用能量距离目标进行自回归建模，避免了量化误差，简化了架构，同时保持信息完整性和推理效率。**

- **链接: [http://arxiv.org/pdf/2505.13181v1](http://arxiv.org/pdf/2505.13181v1)**

> **作者:** Zhengrui Ma; Yang Feng; Chenze Shao; Fandong Meng; Jie Zhou; Min Zhang
>
> **备注:** Demos and code are available at https://github.com/ictnlp/SLED-TTS
>
> **摘要:** We introduce SLED, an alternative approach to speech language modeling by encoding speech waveforms into sequences of continuous latent representations and modeling them autoregressively using an energy distance objective. The energy distance offers an analytical measure of the distributional gap by contrasting simulated and target samples, enabling efficient training to capture the underlying continuous autoregressive distribution. By bypassing reliance on residual vector quantization, SLED avoids discretization errors and eliminates the need for the complicated hierarchical architectures common in existing speech language models. It simplifies the overall modeling pipeline while preserving the richness of speech information and maintaining inference efficiency. Empirical results demonstrate that SLED achieves strong performance in both zero-shot and streaming speech synthesis, showing its potential for broader applications in general-purpose speech language models.
>
---
#### [new 160] R3: Robust Rubric-Agnostic Reward Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型对齐任务，旨在解决现有奖励模型可控性、可解释性差及泛化能力弱的问题。提出了R3框架，通过维度通用、规则无关的设计实现可解释评分，支持多样化人类价值观对齐，并开源了相关资源。**

- **链接: [http://arxiv.org/pdf/2505.13388v1](http://arxiv.org/pdf/2505.13388v1)**

> **作者:** David Anugraha; Zilu Tang; Lester James V. Miranda; Hanyang Zhao; Mohammad Rifqi Farhansyah; Garry Kuwanto; Derry Wijaya; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Reward models are essential for aligning language model outputs with human preferences, yet existing approaches often lack both controllability and interpretability. These models are typically optimized for narrow objectives, limiting their generalizability to broader downstream tasks. Moreover, their scalar outputs are difficult to interpret without contextual reasoning. To address these limitations, we introduce R3, a novel reward modeling framework that is rubric-agnostic, generalizable across evaluation dimensions, and provides interpretable, reasoned score assignments. R3 enables more transparent and flexible evaluation of language models, supporting robust alignment with diverse human values and use cases. Our models, data, and code are available as open source at https://github.com/rubricreward/r3
>
---
#### [new 161] Advancing Sequential Numerical Prediction in Autoregressive Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对自回归模型在序列生成中忽视数值连贯性的问题，提出数值令牌完整性损失（NTIL）。任务为改进序列数值预测，通过令牌级（扩展EMD保持数值序关系）和序列级（惩罚整体差异）双重优化，提升预测精度并与大模型有效兼容。实验验证了性能提升。**

- **链接: [http://arxiv.org/pdf/2505.13077v1](http://arxiv.org/pdf/2505.13077v1)**

> **作者:** Xiang Fei; Jinghui Lu; Qi Sun; Hao Feng; Yanjie Wang; Wei Shi; An-Lan Wang; Jingqun Tang; Can Huang
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introduces Numerical Token Integrity Loss (NTIL) to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover's Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL.
>
---
#### [new 162] MA-COIR: Leveraging Semantic Search Index and Generative Models for Ontology-Driven Biomedical Concept Recognition
- **分类: cs.CL**

- **简介: 该论文属于生物医学概念识别任务，旨在解决传统方法无法识别文本中隐式概念的问题。提出MA-COIR框架，通过语义搜索索引消除本体歧义，结合微调BART模型和LLM生成数据，在低资源场景下有效识别显/隐式概念，无需标注即可应用于知识图谱构建等场景。**

- **链接: [http://arxiv.org/pdf/2505.12964v1](http://arxiv.org/pdf/2505.12964v1)**

> **作者:** Shanshan Liu; Noriki Nishida; Rumana Ferdous Munne; Narumi Tokunaga; Yuki Yamagata; Kouji Kozaki; Yuji Matsumoto
>
> **备注:** preprint
>
> **摘要:** Recognizing biomedical concepts in the text is vital for ontology refinement, knowledge graph construction, and concept relationship discovery. However, traditional concept recognition methods, relying on explicit mention identification, often fail to capture complex concepts not explicitly stated in the text. To overcome this limitation, we introduce MA-COIR, a framework that reformulates concept recognition as an indexing-recognition task. By assigning semantic search indexes (ssIDs) to concepts, MA-COIR resolves ambiguities in ontology entries and enhances recognition efficiency. Using a pretrained BART-based model fine-tuned on small datasets, our approach reduces computational requirements to facilitate adoption by domain experts. Furthermore, we incorporate large language models (LLMs)-generated queries and synthetic data to improve recognition in low-resource settings. Experimental results on three scenarios (CDR, HPO, and HOIP) highlight the effectiveness of MA-COIR in recognizing both explicit and implicit concepts without the need for mention-level annotations during inference, advancing ontology-driven concept recognition in biomedical domain applications. Our code and constructed data are available at https://github.com/sl-633/macoir-master.
>
---
#### [new 163] LM$^2$otifs : An Explainable Framework for Machine-Generated Texts Detection
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于机器生成文本检测任务，旨在解决现有方法解释性不足的问题。作者提出了LM²otifs框架，通过图神经网络将文本转为词汇依赖图进行预测，并提取可解释的语言模式（motifs），实现从词汇到句法的多层级解释，验证了检测效果与解释有效性。**

- **链接: [http://arxiv.org/pdf/2505.12507v1](http://arxiv.org/pdf/2505.12507v1)**

> **作者:** Xu Zheng; Zhuomin Chen; Esteban Schafir; Sipeng Chen; Hojat Allah Salehi; Haifeng Chen; Farhad Shirani; Wei Cheng; Dongsheng Luo
>
> **摘要:** The impressive ability of large language models to generate natural text across various tasks has led to critical challenges in authorship authentication. Although numerous detection methods have been developed to differentiate between machine-generated texts (MGT) and human-generated texts (HGT), the explainability of these methods remains a significant gap. Traditional explainability techniques often fall short in capturing the complex word relationships that distinguish HGT from MGT. To address this limitation, we present LM$^2$otifs, a novel explainable framework for MGT detection. Inspired by probabilistic graphical models, we provide a theoretical rationale for the effectiveness. LM$^2$otifs utilizes eXplainable Graph Neural Networks to achieve both accurate detection and interpretability. The LM$^2$otifs pipeline operates in three key stages: first, it transforms text into graphs based on word co-occurrence to represent lexical dependencies; second, graph neural networks are used for prediction; and third, a post-hoc explainability method extracts interpretable motifs, offering multi-level explanations from individual words to sentence structures. Extensive experiments on multiple benchmark datasets demonstrate the comparable performance of LM$^2$otifs. The empirical evaluation of the extracted explainable motifs confirms their effectiveness in differentiating HGT and MGT. Furthermore, qualitative analysis reveals distinct and visible linguistic fingerprints characteristic of MGT.
>
---
#### [new 164] MR. Judge: Multimodal Reasoner as a Judge
- **分类: cs.CL**

- **简介: 该论文提出MR. Judge，通过多选推理任务增强多模态大模型（MLLMs）的评估能力，解决传统评分方法解释性差、性能不足的问题。其将评判转化为多步推理，自动生成负样本并蒸馏文本推理能力，实验显示模型在多个任务中超越现有基准。**

- **链接: [http://arxiv.org/pdf/2505.13403v1](http://arxiv.org/pdf/2505.13403v1)**

> **作者:** Renjie Pi; Felix Bai; Qibin Chen; Simon Wang; Jiulong Shan; Kieran Liu; Meng Cao
>
> **摘要:** The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning extraction: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%.
>
---
#### [new 165] Towards DS-NER: Unveiling and Addressing Latent Noise in Distant Annotations
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究远程监督命名实体识别（DS-NER），旨在解决不同标注方法产生的潜在噪声分布问题。通过分析规则模型与LLM标注技术，提出新型噪声评估框架，将噪声分为未标注实体(UEP)和噪声实体(NEP)两类，并设计针对性解决方案。实验在8个多源数据集上验证了方法的有效性，显著超越现有技术。**

- **链接: [http://arxiv.org/pdf/2505.12454v1](http://arxiv.org/pdf/2505.12454v1)**

> **作者:** Yuyang Ding; Dan Qiao; Juntao Li; Jiajie Xu; Pingfu Chao; Xiaofang Zhou; Min Zhang
>
> **摘要:** Distantly supervised named entity recognition (DS-NER) has emerged as a cheap and convenient alternative to traditional human annotation methods, enabling the automatic generation of training data by aligning text with external resources. Despite the many efforts in noise measurement methods, few works focus on the latent noise distribution between different distant annotation methods. In this work, we explore the effectiveness and robustness of DS-NER by two aspects: (1) distant annotation techniques, which encompasses both traditional rule-based methods and the innovative large language model supervision approach, and (2) noise assessment, for which we introduce a novel framework. This framework addresses the challenges by distinctly categorizing them into the unlabeled-entity problem (UEP) and the noisy-entity problem (NEP), subsequently providing specialized solutions for each. Our proposed method achieves significant improvements on eight real-world distant supervision datasets originating from three different data sources and involving four distinct annotation techniques, confirming its superiority over current state-of-the-art methods.
>
---
#### [new 166] FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究无人机视觉与语言导航任务，解决现有方法多模态融合弱、泛化差、可解释性低的问题。提出FlightGPT框架，基于视觉语言模型构建两阶段训练（监督微调+强化学习优化），结合思维链推理提升决策可解释性，在城市场景数据集实现9.22%成功率提升。**

- **链接: [http://arxiv.org/pdf/2505.12835v1](http://arxiv.org/pdf/2505.12835v1)**

> **作者:** Hengxing Cai; Jinhan Dong; Jingjun Tan; Jingcheng Deng; Sihang Li; Zhifeng Gao; Haidong Wang; Zicheng Su; Agachai Sumalee; Renxin Zhong
>
> **摘要:** Unmanned Aerial Vehicle (UAV) Vision-and-Language Navigation (VLN) is vital for applications such as disaster response, logistics delivery, and urban inspection. However, existing methods often struggle with insufficient multimodal fusion, weak generalization, and poor interpretability. To address these challenges, we propose FlightGPT, a novel UAV VLN framework built upon Vision-Language Models (VLMs) with powerful multimodal perception capabilities. We design a two-stage training pipeline: first, Supervised Fine-Tuning (SFT) using high-quality demonstrations to improve initialization and structured reasoning; then, Group Relative Policy Optimization (GRPO) algorithm, guided by a composite reward that considers goal accuracy, reasoning quality, and format compliance, to enhance generalization and adaptability. Furthermore, FlightGPT introduces a Chain-of-Thought (CoT)-based reasoning mechanism to improve decision interpretability. Extensive experiments on the city-scale dataset CityNav demonstrate that FlightGPT achieves state-of-the-art performance across all scenarios, with a 9.22\% higher success rate than the strongest baseline in unseen environments. Our implementation is publicly available.
>
---
#### [new 167] Historical and psycholinguistic perspectives on morphological productivity: A sketch of an integrative approach
- **分类: cs.CL**

- **简介: 该论文研究形态能产性（构词规律生成新词的能力），结合认知计算模型（DLM）和历时视角（以作家托马斯·曼为例）。任务属于计算心理语言学，旨在量化不同语言形态模式的能产性及个体语言创新机制。工作包括：1）用DLM分析芬兰、马来语和英语的词缀语义规律；2）基于曼的读写数据，发现其新词产出率远低于输入，并与词缀语义距离相关。**

- **链接: [http://arxiv.org/pdf/2505.12071v1](http://arxiv.org/pdf/2505.12071v1)**

> **作者:** Harald Baayen; Kristian Berg; Maziyah Mohamed
>
> **备注:** 35 pages, 11 figures
>
> **摘要:** In this study, we approach morphological productivity from two perspectives: a cognitive-computational perspective, and a diachronic perspective zooming in on an actual speaker, Thomas Mann. For developing the first perspective, we make use of a cognitive computational model of the mental lexicon, the discriminative lexicon model. For computational mappings between form and meaning to be productive, in the sense that novel, previously unencountered words, can be understood and produced, there must be systematicities between the form space and the semantic space. If the relation between form and meaning would be truly arbitrary, a model could memorize form and meaning pairings, but there is no way in which the model would be able to generalize to novel test data. For Finnish nominal inflection, Malay derivation, and English compounding, we explore, using the Discriminative Lexicon Model as a computational tool, to trace differences in the degree to which inflectional and word formation patterns are productive. We show that the DLM tends to associate affix-like sublexical units with the centroids of the embeddings of the words with a given affix. For developing the second perspective, we study how the intake and output of one prolific writer, Thomas Mann, changes over time. We show by means of an examination of what Thomas Mann is likely to have read, and what he wrote, that the rate at which Mann produces novel derived words is extremely low. There are far more novel words in his input than in his output. We show that Thomas Mann is less likely to produce a novel derived word with a given suffix the greater the average distance is of the embeddings of all derived words to the corresponding centroid, and discuss the challenges of using speaker-specific embeddings for low-frequency and novel words.
>
---
#### [new 168] Relation Extraction or Pattern Matching? Unravelling the Generalisation Limits of Language Models for Biographical RE
- **分类: cs.CL**

- **简介: 该论文研究关系抽取（RE）模型的泛化能力，属于自然语言处理任务。旨在揭示模型依赖真实语义模式还是数据集偏差。通过跨数据集实验发现，RE模型易过拟合数据集特征，迁移性能与数据质量而非词汇相似性相关：高质量数据适合微调，低质量数据则少样本学习更优。同时指出现有评测基准的结构缺陷（如单关系样本设计）进一步削弱模型迁移性。**

- **链接: [http://arxiv.org/pdf/2505.12533v1](http://arxiv.org/pdf/2505.12533v1)**

> **作者:** Varvara Arzt; Allan Hanbury; Michael Wiegand; Gábor Recski; Terra Blevins
>
> **摘要:** Analysing the generalisation capabilities of relation extraction (RE) models is crucial for assessing whether they learn robust relational patterns or rely on spurious correlations. Our cross-dataset experiments find that RE models struggle with unseen data, even within similar domains. Notably, higher intra-dataset performance does not indicate better transferability, instead often signaling overfitting to dataset-specific artefacts. Our results also show that data quality, rather than lexical similarity, is key to robust transfer, and the choice of optimal adaptation strategy depends on the quality of data available: while fine-tuning yields the best cross-dataset performance with high-quality data, few-shot in-context learning (ICL) is more effective with noisier data. However, even in these cases, zero-shot baselines occasionally outperform all cross-dataset results. Structural issues in RE benchmarks, such as single-relation per sample constraints and non-standardised negative class definitions, further hinder model transferability.
>
---
#### [new 169] SynDec: A Synthesize-then-Decode Approach for Arbitrary Textual Style Transfer via Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本风格转换任务，解决大语言模型（LLMs）在任意风格迁移中依赖人工提示和固有风格偏差的问题。提出SynDec框架：通过自动合成高质量提示（基于样本筛选、四维风格分析和重排），并在解码阶段增强提示作用（对比有无提示及负样本概率差异），实验表明其在六分之五基准测试中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12821v1](http://arxiv.org/pdf/2505.12821v1)**

> **作者:** Han Sun; Zhen Sun; Zongmin Zhang; Linzhao Jia; Wei Shao; Min Zhang
>
> **摘要:** Large Language Models (LLMs) are emerging as dominant forces for textual style transfer. However, for arbitrary style transfer, LLMs face two key challenges: (1) considerable reliance on manually-constructed prompts and (2) rigid stylistic biases inherent in LLMs. In this paper, we propose a novel Synthesize-then-Decode (SynDec) approach, which automatically synthesizes high-quality prompts and amplifies their roles during decoding process. Specifically, our approach synthesizes prompts by selecting representative few-shot samples, conducting a four-dimensional style analysis, and reranking the candidates. At LLM decoding stage, the TST effect is amplified by maximizing the contrast in output probabilities between scenarios with and without the synthesized prompt, as well as between prompts and negative samples. We conduct extensive experiments and the results show that SynDec outperforms existing state-of-the-art LLM-based methods on five out of six benchmarks (e.g., achieving up to a 9\% increase in accuracy for modern-to-Elizabethan English transfer). Detailed ablation studies further validate the effectiveness of SynDec.
>
---
#### [new 170] MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于临床决策评估任务，旨在解决大语言模型（LLMs）能否遵循结构化医学指南的问题。通过构建MedGUIDE基准（含7747个基于癌症指南的多选题），评估25个LLMs的临床决策能力，发现现有模型在遵循指南方面不足，并验证改进方法的局限性，强调其在医疗安全评估中的重要性。**

- **链接: [http://arxiv.org/pdf/2505.11613v1](http://arxiv.org/pdf/2505.11613v1)**

> **作者:** Xiaomin Li; Mingye Gao; Yuexing Hao; Taoran Li; Guangya Wan; Zihan Wang; Yijun Wang
>
> **摘要:** Clinical guidelines, typically structured as decision trees, are central to evidence-based medical practice and critical for ensuring safe and accurate diagnostic decision-making. However, it remains unclear whether Large Language Models (LLMs) can reliably follow such structured protocols. In this work, we introduce MedGUIDE, a new benchmark for evaluating LLMs on their ability to make guideline-consistent clinical decisions. MedGUIDE is constructed from 55 curated NCCN decision trees across 17 cancer types and uses clinical scenarios generated by LLMs to create a large pool of multiple-choice diagnostic questions. We apply a two-stage quality selection process, combining expert-labeled reward models and LLM-as-a-judge ensembles across ten clinical and linguistic criteria, to select 7,747 high-quality samples. We evaluate 25 LLMs spanning general-purpose, open-source, and medically specialized models, and find that even domain-specific LLMs often underperform on tasks requiring structured guideline adherence. We also test whether performance can be improved via in-context guideline inclusion or continued pretraining. Our findings underscore the importance of MedGUIDE in assessing whether LLMs can operate safely within the procedural frameworks expected in real-world clinical settings.
>
---
#### [new 171] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Reasoning
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文研究大语言模型（LLMs）在长代码上下文中的推理能力，属于代码理解与推理任务。针对LLMs长代码中段推理准确率骤降、现有基准低估语义召回挑战的问题，提出SemTrace方法量化语义召回敏感性，分析词汇/语义召回差异，发现两者机制分离且现有基准敏感性不足。**

- **链接: [http://arxiv.org/pdf/2505.13353v1](http://arxiv.org/pdf/2505.13353v1)**

> **作者:** Adam Štorek; Mukur Gupta; Samira Hajizadeh; Prashast Srivastava; Suman Jana
>
> **摘要:** Although modern Large Language Models (LLMs) support extremely large contexts, their effectiveness in utilizing long context for code reasoning remains unclear. This paper investigates LLM reasoning ability over code snippets within large repositories and how it relates to their recall ability. Specifically, we differentiate between lexical code recall (verbatim retrieval) and semantic code recall (remembering what the code does). To measure semantic recall, we propose SemTrace, a code reasoning technique where the impact of specific statements on output is attributable and unpredictable. We also present a method to quantify semantic recall sensitivity in existing benchmarks. Our evaluation of state-of-the-art LLMs reveals a significant drop in code reasoning accuracy as a code snippet approaches the middle of the input context, particularly with techniques requiring high semantic recall like SemTrace. Moreover, we find that lexical recall varies by granularity, with models excelling at function retrieval but struggling with line-by-line recall. Notably, a disconnect exists between lexical and semantic recall, suggesting different underlying mechanisms. Finally, our findings indicate that current code reasoning benchmarks may exhibit low semantic recall sensitivity, potentially underestimating LLM challenges in leveraging in-context information.
>
---
#### [new 172] R1dacted: Investigating Local Censorship in DeepSeek's R1 Language Model
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文研究DeepSeek的R1语言模型在政治敏感话题上的本地化审查机制，属于模型行为分析领域。通过构建专用提示集，揭示R1特有的审查规律、多语言表现及迁移特性，并提出规避方法，旨在揭露模型训练中潜在的审查设计及其对透明度和治理的影响。**

- **链接: [http://arxiv.org/pdf/2505.12625v1](http://arxiv.org/pdf/2505.12625v1)**

> **作者:** Ali Naseh; Harsh Chaudhari; Jaechul Roh; Mingshi Wu; Alina Oprea; Amir Houmansadr
>
> **摘要:** DeepSeek recently released R1, a high-performing large language model (LLM) optimized for reasoning tasks. Despite its efficient training pipeline, R1 achieves competitive performance, even surpassing leading reasoning models like OpenAI's o1 on several benchmarks. However, emerging reports suggest that R1 refuses to answer certain prompts related to politically sensitive topics in China. While existing LLMs often implement safeguards to avoid generating harmful or offensive outputs, R1 represents a notable shift - exhibiting censorship-like behavior on politically charged queries. In this paper, we investigate this phenomenon by first introducing a large-scale set of heavily curated prompts that get censored by R1, covering a range of politically sensitive topics, but are not censored by other models. We then conduct a comprehensive analysis of R1's censorship patterns, examining their consistency, triggers, and variations across topics, prompt phrasing, and context. Beyond English-language queries, we explore censorship behavior in other languages. We also investigate the transferability of censorship to models distilled from the R1 language model. Finally, we propose techniques for bypassing or removing this censorship. Our findings reveal possible additional censorship integration likely shaped by design choices during training or alignment, raising concerns about transparency, bias, and governance in language model deployment.
>
---
#### [new 173] On the Thinking-Language Modeling Gap in Large Language Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于语言模型推理优化任务，旨在解决大语言模型因语言偏见偏离人类思维链的问题。作者提出Language-of-Thoughts（LoT）提示技术，通过调整生成顺序和词汇表达，减少语言建模偏差，提升多项推理任务表现。**

- **链接: [http://arxiv.org/pdf/2505.12896v1](http://arxiv.org/pdf/2505.12896v1)**

> **作者:** Chenxi Liu; Yongqiang Chen; Tongliang Liu; James Cheng; Bo Han; Kun Zhang
>
> **备注:** Chenxi and Yongqiang contributed equally; project page: https://causalcoat.github.io/lot.html
>
> **摘要:** System 2 reasoning is one of the defining characteristics of intelligence, which requires slow and logical thinking. Human conducts System 2 reasoning via the language of thoughts that organizes the reasoning process as a causal sequence of mental language, or thoughts. Recently, it has been observed that System 2 reasoning can be elicited from Large Language Models (LLMs) pre-trained on large-scale natural languages. However, in this work, we show that there is a significant gap between the modeling of languages and thoughts. As language is primarily a tool for humans to share knowledge and thinking, modeling human language can easily absorb language biases into LLMs deviated from the chain of thoughts in minds. Furthermore, we show that the biases will mislead the eliciting of "thoughts" in LLMs to focus only on a biased part of the premise. To this end, we propose a new prompt technique termed Language-of-Thoughts (LoT) to demonstrate and alleviate this gap. Instead of directly eliciting the chain of thoughts from partial information, LoT instructs LLMs to adjust the order and token used for the expressions of all the relevant information. We show that the simple strategy significantly reduces the language modeling biases in LLMs and improves the performance of LLMs across a variety of reasoning tasks.
>
---
#### [new 174] Natural Language Planning via Coding and Inference Scaling
- **分类: cs.CL**

- **简介: 该论文研究自然语言规划任务（如会议安排），探讨如何通过生成可执行代码（Python/约束求解器）提升大模型处理复杂问题的能力。通过对比闭源与开源模型，发现编程方法常优于直接规划，但生成代码存在鲁棒性和效率缺陷，影响泛化性。**

- **链接: [http://arxiv.org/pdf/2505.13252v1](http://arxiv.org/pdf/2505.13252v1)**

> **作者:** Rikhil Amonkar; Ronan Le Bras; Li Zhang
>
> **摘要:** Real-life textual planning tasks such as meeting scheduling have posed much challenge to LLMs especially when the complexity is high. While previous work primarily studied auto-regressive generation of plans with closed-source models, we systematically evaluate both closed- and open-source models, including those that scales output length with complexity during inference, in generating programs, which are executed to output the plan. We consider not only standard Python code, but also the code to a constraint satisfaction problem solver. Despite the algorithmic nature of the task, we show that programming often but not always outperforms planning. Our detailed error analysis also indicates a lack of robustness and efficiency in the generated code that hinders generalization.
>
---
#### [new 175] MedCaseReasoning: Evaluating and learning diagnostic reasoning from clinical case reports
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断推理评估任务，旨在解决现有基准忽视诊断过程质量的问题。作者构建了首个开放数据集MedCaseReasoning（含14,489临床案例），通过评估发现主流模型在诊断准确性和推理完整性存在不足，并证明基于该数据微调可显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.11733v1](http://arxiv.org/pdf/2505.11733v1)**

> **作者:** Kevin Wu; Eric Wu; Rahul Thapa; Kevin Wei; Angela Zhang; Arvind Suresh; Jacqueline J. Tao; Min Woo Sun; Alejandro Lozano; James Zou
>
> **摘要:** Doctors and patients alike increasingly use Large Language Models (LLMs) to diagnose clinical cases. However, unlike domains such as math or coding, where correctness can be objectively defined by the final answer, medical diagnosis requires both the outcome and the reasoning process to be accurate. Currently, widely used medical benchmarks like MedQA and MMLU assess only accuracy in the final answer, overlooking the quality and faithfulness of the clinical reasoning process. To address this limitation, we introduce MedCaseReasoning, the first open-access dataset for evaluating LLMs on their ability to align with clinician-authored diagnostic reasoning. The dataset includes 14,489 diagnostic question-and-answer cases, each paired with detailed reasoning statements derived from open-access medical case reports. We evaluate state-of-the-art reasoning LLMs on MedCaseReasoning and find significant shortcomings in their diagnoses and reasoning: for instance, the top-performing open-source model, DeepSeek-R1, achieves only 48% 10-shot diagnostic accuracy and mentions only 64% of the clinician reasoning statements (recall). However, we demonstrate that fine-tuning LLMs on the reasoning traces derived from MedCaseReasoning significantly improves diagnostic accuracy and clinical reasoning recall by an average relative gain of 29% and 41%, respectively. The open-source dataset, code, and models are available at https://github.com/kevinwu23/Stanford-MedCaseReasoning.
>
---
#### [new 176] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering
- **分类: cs.CL**

- **简介: 该论文研究句子嵌入提取任务，解决现有提示方法因编码非必要信息（如停用词）导致语义表达能力受限的问题。提出对比提示（CP），通过引入辅助提示进行推理时对比，引导模型聚焦核心语义而非冗余信息。该方法无需微调，可即插即用增强现有提示方法，在语义相似度和分类任务中验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.12831v1](http://arxiv.org/pdf/2505.12831v1)**

> **作者:** Zifeng Cheng; Zhonghui Wang; Yuchen Fu; Zhiwei Jiang; Yafeng Yin; Cong Wang; Qing Gu
>
> **备注:** ACL 2025
>
> **摘要:** Extracting sentence embeddings from large language models (LLMs) is a practical direction, as it requires neither additional data nor fine-tuning. Previous studies usually focus on prompt engineering to guide LLMs to encode the core semantic information of the sentence into the embedding of the last token. However, the last token in these methods still encodes an excess of non-essential information, such as stop words, limiting its encoding capacity. To this end, we propose a Contrastive Prompting (CP) method that introduces an extra auxiliary prompt to elicit better sentence embedding. By contrasting with the auxiliary prompt, CP can steer existing prompts to encode the core semantics of the sentence, rather than non-essential information. CP is a plug-and-play inference-time intervention method that can be combined with various prompt-based methods. Extensive experiments on Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our method can improve the performance of existing prompt-based methods across different LLMs. Our code will be released at https://github.com/zifengcheng/CP.
>
---
#### [new 177] Towards Universal Semantics With Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在利用大语言模型（LLMs）自动生成自然语义元语言（NSM）的通用语义解释，解决传统手动生成效率低的问题。通过构建专用数据集、设计评估方法并微调模型，其1B/8B模型在生成跨语言可译的精准解释上超越GPT-4o，推动了LLM在语义分析和翻译等任务中的通用表征能力。**

- **链接: [http://arxiv.org/pdf/2505.11764v1](http://arxiv.org/pdf/2505.11764v1)**

> **作者:** Raymond Baartmans; Matthew Raffel; Rahul Vikram; Aiden Deringer; Lizhong Chen
>
> **摘要:** The Natural Semantic Metalanguage (NSM) is a linguistic theory based on a universal set of semantic primes: simple, primitive word-meanings that have been shown to exist in most, if not all, languages of the world. According to this framework, any word, regardless of complexity, can be paraphrased using these primes, revealing a clear and universally translatable meaning. These paraphrases, known as explications, can offer valuable applications for many natural language processing (NLP) tasks, but producing them has traditionally been a slow, manual process. In this work, we present the first study of using large language models (LLMs) to generate NSM explications. We introduce automatic evaluation methods, a tailored dataset for training and evaluation, and fine-tuned models for this task. Our 1B and 8B models outperform GPT-4o in producing accurate, cross-translatable explications, marking a significant step toward universal semantic representation with LLMs and opening up new possibilities for applications in semantic analysis, translation, and beyond.
>
---
#### [new 178] HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型微调中数据全局（跨数据集）与局部（单数据集内）不平衡和异质性难题，提出层次平衡优化方法HBO。通过全局和局部双层优化策略，动态调整数据分配，提升模型在多样化任务中的准确率，实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2505.12300v1](http://arxiv.org/pdf/2505.12300v1)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets.
>
---
#### [new 179] GAP: Graph-Assisted Prompts for Dialogue-based Medication Recommendation
- **分类: cs.CL**

- **简介: 该论文研究基于对话的药物推荐任务，解决大型语言模型（LLM）忽略对话细粒度信息、跨轮次关联及生成非事实回应的问题。提出GAP框架，通过构建患者中心化医疗图谱并结合外部知识图谱生成提示，增强信息检索能力，提升推荐准确性和安全性。实验验证其在动态诊断场景的有效性。**

- **链接: [http://arxiv.org/pdf/2505.12888v1](http://arxiv.org/pdf/2505.12888v1)**

> **作者:** Jialun Zhong; Yanzeng Li; Sen Hu; Yang Zhang; Teng Xu; Lei Zou
>
> **摘要:** Medication recommendations have become an important task in the healthcare domain, especially in measuring the accuracy and safety of medical dialogue systems (MDS). Different from the recommendation task based on electronic health records (EHRs), dialogue-based medication recommendations require research on the interaction details between patients and doctors, which is crucial but may not exist in EHRs. Recent advancements in large language models (LLM) have extended the medical dialogue domain. These LLMs can interpret patients' intent and provide medical suggestions including medication recommendations, but some challenges are still worth attention. During a multi-turn dialogue, LLMs may ignore the fine-grained medical information or connections across the dialogue turns, which is vital for providing accurate suggestions. Besides, LLMs may generate non-factual responses when there is a lack of domain-specific knowledge, which is more risky in the medical domain. To address these challenges, we propose a \textbf{G}raph-\textbf{A}ssisted \textbf{P}rompts (\textbf{GAP}) framework for dialogue-based medication recommendation. It extracts medical concepts and corresponding states from dialogue to construct an explicitly patient-centric graph, which can describe the neglected but important information. Further, combined with external medical knowledge graphs, GAP can generate abundant queries and prompts, thus retrieving information from multiple sources to reduce the non-factual responses. We evaluate GAP on a dialogue-based medication recommendation dataset and further explore its potential in a more difficult scenario, dynamically diagnostic interviewing. Extensive experiments demonstrate its competitive performance when compared with strong baselines.
>
---
#### [new 180] An Annotated Corpus of Arabic Tweets for Hate Speech Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的仇恨言论检测任务，旨在解决阿拉伯语方言多样性导致的仇恨内容识别难题。研究构建了包含1万条多标签标注的阿拉伯推文数据集，区分攻击性内容并细化仇恨目标类别（宗教、性别等），通过多标注者验证（一致性0.86/0.71），并验证了AraBERTv2模型的最佳性能（F1 0.7865）。**

- **链接: [http://arxiv.org/pdf/2505.11969v1](http://arxiv.org/pdf/2505.11969v1)**

> **作者:** Md. Rafiul Biswas; Wajdi Zaghouani
>
> **摘要:** Identifying hate speech content in the Arabic language is challenging due to the rich quality of dialectal variations. This study introduces a multilabel hate speech dataset in the Arabic language. We have collected 10000 Arabic tweets and annotated each tweet, whether it contains offensive content or not. If a text contains offensive content, we further classify it into different hate speech targets such as religion, gender, politics, ethnicity, origin, and others. A text can contain either single or multiple targets. Multiple annotators are involved in the data annotation task. We calculated the inter-annotator agreement, which was reported to be 0.86 for offensive content and 0.71 for multiple hate speech targets. Finally, we evaluated the data annotation task by employing a different transformers-based model in which AraBERTv2 outperformed with a micro-F1 score of 0.7865 and an accuracy of 0.786.
>
---
#### [new 181] PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私保护任务，旨在解决大语言模型（LLMs）记忆敏感个人身份信息（PII）的风险评估问题。作者构建了PANORAMA——包含38万合成数据的PII数据集，模拟真实网络环境中的敏感信息分布，并通过微调实验验证重复数据量及内容类型对LLMs记忆率的影响，为隐私风险研究和模型审计提供资源。**

- **链接: [http://arxiv.org/pdf/2505.12238v1](http://arxiv.org/pdf/2505.12238v1)**

> **作者:** Sriram Selvam; Anneswa Ghosh
>
> **摘要:** The memorization of sensitive and personally identifiable information (PII) by large language models (LLMs) poses growing privacy risks as models scale and are increasingly deployed in real-world applications. Existing efforts to study sensitive and PII data memorization and develop mitigation strategies are hampered by the absence of comprehensive, realistic, and ethically sourced datasets reflecting the diversity of sensitive information found on the web. We introduce PANORAMA - Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis, a large-scale synthetic corpus of 384,789 samples derived from 9,674 synthetic profiles designed to closely emulate the distribution, variety, and context of PII and sensitive data as it naturally occurs in online environments. Our data generation pipeline begins with the construction of internally consistent, multi-attribute human profiles using constrained selection to reflect real-world demographics such as education, health attributes, financial status, etc. Using a combination of zero-shot prompting and OpenAI o3-mini, we generate diverse content types - including wiki-style articles, social media posts, forum discussions, online reviews, comments, and marketplace listings - each embedding realistic, contextually appropriate PII and other sensitive information. We validate the utility of PANORAMA by fine-tuning the Mistral-7B model on 1x, 5x, 10x, and 25x data replication rates with a subset of data and measure PII memorization rates - revealing not only consistent increases with repetition but also variation across content types, highlighting PANORAMA's ability to model how memorization risks differ by context. Our dataset and code are publicly available, providing a much-needed resource for privacy risk assessment, model auditing, and the development of privacy-preserving LLMs.
>
---
#### [new 182] Traversal Verification for Speculative Tree Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型推理加速任务，针对推测解码中现有方法因逐层验证导致候选利用率低、接受长度不足的问题，提出叶到根遍历验证算法。通过保留有效子序列并保证概率分布一致性，实现无损加速，实验验证了吞吐量和接受长度的提升。**

- **链接: [http://arxiv.org/pdf/2505.12398v1](http://arxiv.org/pdf/2505.12398v1)**

> **作者:** Yepeng Weng; Qiao Hu; Xujie Chen; Li Liu; Dianwen Mei; Huishi Qiu; Jiang Tian; Zhongchao Shi
>
> **备注:** Under review
>
> **摘要:** Speculative decoding is a promising approach for accelerating large language models. The primary idea is to use a lightweight draft model to speculate the output of the target model for multiple subsequent timesteps, and then verify them in parallel to determine whether the drafted tokens should be accepted or rejected. To enhance acceptance rates, existing frameworks typically construct token trees containing multiple candidates in each timestep. However, their reliance on token-level verification mechanisms introduces two critical limitations: First, the probability distribution of a sequence differs from that of individual tokens, leading to suboptimal acceptance length. Second, current verification schemes begin from the root node and proceed layer by layer in a top-down manner. Once a parent node is rejected, all its child nodes should be discarded, resulting in inefficient utilization of speculative candidates. This paper introduces Traversal Verification, a novel speculative decoding algorithm that fundamentally rethinks the verification paradigm through leaf-to-root traversal. Our approach considers the acceptance of the entire token sequence from the current node to the root, and preserves potentially valid subsequences that would be prematurely discarded by existing methods. We theoretically prove that the probability distribution obtained through Traversal Verification is identical to that of the target model, guaranteeing lossless inference while achieving substantial acceleration gains. Experimental results across different large language models and multiple tasks show that our method consistently improves acceptance length and throughput over existing methods
>
---
#### [new 183] Disambiguation in Conversational Question Answering in the Era of LLM: A Survey
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的歧义消解综述研究，聚焦对话问答场景下大语言模型（LLM）的歧义问题。通过定义歧义类型、分类LLM消歧方法、对比技术优劣、评估数据集，并提出未来方向，旨在增强语言系统的鲁棒性。核心解决LLM时代对话交互中因语言复杂性导致的语义模糊问题。**

- **链接: [http://arxiv.org/pdf/2505.12543v1](http://arxiv.org/pdf/2505.12543v1)**

> **作者:** Md Mehrab Tanjim; Yeonjun In; Xiang Chen; Victor S. Bursztyn; Ryan A. Rossi; Sungchul Kim; Guang-Jie Ren; Vaishnavi Muppala; Shun Jiang; Yongsung Kim; Chanyoung Park
>
> **备注:** Preprint
>
> **摘要:** Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable language systems.
>
---
#### [new 184] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于低资源语言内容审核领域，针对提格里尼亚语社交媒体中滥用语言检测问题，构建了首个多任务基准数据集（13,717条评论），联合标注滥用性、情感和主题。通过迭代术语聚类收集数据，兼容罗马化/Geez双文字系统，实验表明小型多任务模型优于前沿模型（滥用检测准确率86%），推动低资源场景的在线安全研究。**

- **链接: [http://arxiv.org/pdf/2505.12116v1](http://arxiv.org/pdf/2505.12116v1)**

> **作者:** Fitsum Gaim; Hoyun Song; Huije Lee; Changgeon Ko; Eui Jun Hwang; Jong C. Park
>
> **摘要:** Content moderation research has recently made significant advances, but still fails to serve the majority of the world's languages due to the lack of resources, leaving millions of vulnerable users to online hostility. This work presents a large-scale human-annotated multi-task benchmark dataset for abusive language detection in Tigrinya social media with joint annotations for three tasks: abusiveness, sentiment, and topic classification. The dataset comprises 13,717 YouTube comments annotated by nine native speakers, collected from 7,373 videos with a total of over 1.2 billion views across 51 channels. We developed an iterative term clustering approach for effective data selection. Recognizing that around 64% of Tigrinya social media content uses Romanized transliterations rather than native Ge'ez script, our dataset accommodates both writing systems to reflect actual language use. We establish strong baselines across the tasks in the benchmark, while leaving significant challenges for future contributions. Our experiments reveal that small, specialized multi-task models outperform the current frontier models in the low-resource setting, achieving up to 86% accuracy (+7 points) in abusiveness detection. We make the resources publicly available to promote research on online safety.
>
---
#### [new 185] Teach2Eval: An Indirect Evaluation Method for LLM by Judging How It Teaches
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）评估任务，针对传统评测方法存在公平性差、扩展性低及数据污染问题，提出间接评估框架Teach2Eval。通过将开放任务转化为多选题，评估LLM教导弱学生模型的能力，实现自动化多维评测，避免数据泄露，并捕捉传统基准未覆盖的认知能力。实验验证了其与现有评估的一致性及可解释性优势。**

- **链接: [http://arxiv.org/pdf/2505.12259v1](http://arxiv.org/pdf/2505.12259v1)**

> **作者:** Yuhang Zhou; Xutian Chen; Yixin Cao; Yuchen Ni; Yu He; Siyu Tian; Xiang Liu; Jian Zhang; Chuanjun Ji; Guangnan Ye; Xipeng Qiu
>
> **摘要:** Recent progress in large language models (LLMs) has outpaced the development of effective evaluation methods. Traditional benchmarks rely on task-specific metrics and static datasets, which often suffer from fairness issues, limited scalability, and contamination risks. In this paper, we introduce Teach2Eval, an indirect evaluation framework inspired by the Feynman Technique. Instead of directly testing LLMs on predefined tasks, our method evaluates a model's multiple abilities to teach weaker student models to perform tasks effectively. By converting open-ended tasks into standardized multiple-choice questions (MCQs) through teacher-generated feedback, Teach2Eval enables scalable, automated, and multi-dimensional assessment. Our approach not only avoids data leakage and memorization but also captures a broad range of cognitive abilities that are orthogonal to current benchmarks. Experimental results across 26 leading LLMs show strong alignment with existing human and model-based dynamic rankings, while offering additional interpretability for training guidance.
>
---
#### [new 186] A3 : an Analytical Low-Rank Approximation Framework for Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大型语言模型压缩任务，解决现有低秩近似方法忽略Transformer架构特性、引入运行时开销的问题。提出A³框架，将Transformer层拆分为QK、OV、MLP三个组件，通过分析解减少各组件隐藏维度，直接降低模型大小、KV缓存和计算量，无额外开销，实现端到端性能优化。**

- **链接: [http://arxiv.org/pdf/2505.12942v1](http://arxiv.org/pdf/2505.12942v1)**

> **作者:** Jeffrey T. H. Wong; Cheng Zhang; Xinye Cao; Pedro Gimenes; George A. Constantinides; Wayne Luk; Yiren Zhao
>
> **摘要:** Large language models have demonstrated remarkable performance; however, their massive parameter counts make deployment highly expensive. Low-rank approximation offers a promising compression solution, yet existing approaches have two main limitations: (1) They focus on minimizing the output error of individual linear layers, without considering the architectural characteristics of Transformers, and (2) they decompose a large weight matrix into two small low-rank matrices. Consequently, these methods often fall short compared to other compression techniques like pruning and quantization, and introduce runtime overhead such as the extra GEMM kernel launches for decomposed small matrices. To address these limitations, we propose $\tt A^\tt 3$, a post-training low-rank approximation framework. $\tt A^\tt 3$ splits a Transformer layer into three functional components, namely $\tt QK$, $\tt OV$, and $\tt MLP$. For each component, $\tt A^\tt 3$ provides an analytical solution that reduces the hidden dimension size inside each component while minimizing the component's functional loss ($\it i.e.$, error in attention scores, attention outputs, and MLP outputs). This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads. In addition, it provides a new narrative in advancing the optimization problem from singular linear layer loss optimization toward improved end-to-end performance. Through extensive experiments, we show that $\tt A^\tt 3$ maintains superior performance compared to SoTAs. For example, under the same reduction budget in computation and memory, our low-rank approximated LLaMA 3.1-70B achieves a perplexity of 4.69 on WikiText-2, outperforming the previous SoTA's 7.87 by 3.18. We also demonstrate the versatility of $\tt A^\tt 3$, including KV cache compression, quantization, and mixed-rank assignments for enhanced performance.
>
---
#### [new 187] CIE: Controlling Language Model Text Generations Using Continuous Signals
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究可控文本生成任务，旨在解决现有方法（自然语言提示/离散信号）难以实现连续属性控制的问题。通过微调语言模型，提出用连续向量（低-高嵌入插值）控制生成属性，以响应长度为例验证方法优于传统方案，提升控制可靠性。**

- **链接: [http://arxiv.org/pdf/2505.13448v1](http://arxiv.org/pdf/2505.13448v1)**

> **作者:** Vinay Samuel; Harshita Diddee; Yiming Zhang; Daphne Ippolito
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Aligning language models with user intent is becoming increasingly relevant to enhance user experience. This calls for designing methods that can allow users to control the properties of the language that LMs generate. For example, controlling the length of the generation, the complexity of the language that gets chosen, the sentiment, tone, etc. Most existing work attempts to integrate users' control by conditioning LM generations on natural language prompts or discrete control signals, which are often brittle and hard to scale. In this work, we are interested in \textit{continuous} control signals, ones that exist along a spectrum that can't easily be captured in a natural language prompt or via existing techniques in conditional generation. Through a case study in controlling the precise response-length of generations produced by LMs, we demonstrate how after fine-tuning, behaviors of language models can be controlled via continuous signals -- as vectors that are interpolated between a "low" and a "high" token embedding. Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal. Our full open-sourced code and datasets are available at https://github.com/vsamuel2003/CIE.
>
---
#### [new 188] ExpertSteer: Intervening in LLMs through Expert Knowledge
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）行为控制任务，旨在解决现有激活引导方法依赖模型自身生成向量、无法利用外部专家知识的问题。研究者提出ExpertSteer方法，通过专家模型生成引导向量，经维度对齐、干预层匹配等四步骤将知识迁移至目标LLM，实现无参数更新的跨模型干预控制，实验证明其在多任务中显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.12313v1](http://arxiv.org/pdf/2505.12313v1)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities across various tasks, yet guiding them to follow desired behaviours during inference remains a significant challenge. Activation steering offers a promising method to control the generation process of LLMs by modifying their internal activations. However, existing methods commonly intervene in the model's behaviour using steering vectors generated by the model itself, which constrains their effectiveness to that specific model and excludes the possibility of leveraging powerful external expert models for steering. To address these limitations, we propose ExpertSteer, a novel approach that leverages arbitrary specialized expert models to generate steering vectors, enabling intervention in any LLMs. ExpertSteer transfers the knowledge from an expert model to a target LLM through a cohesive four-step process: first aligning representation dimensions with auto-encoders to enable cross-model transfer, then identifying intervention layer pairs based on mutual information analysis, next generating steering vectors from the expert model using Recursive Feature Machines, and finally applying these vectors on the identified layers during inference to selectively guide the target LLM without updating model parameters. We conduct comprehensive experiments using three LLMs on 15 popular benchmarks across four distinct domains. Experiments demonstrate that ExpertSteer significantly outperforms established baselines across diverse tasks at minimal cost.
>
---
#### [new 189] What are they talking about? Benchmarking Large Language Models for Knowledge-Grounded Discussion Summarization
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在知识增强型讨论摘要任务中的表现，解决现有对话摘要系统因缺乏背景知识导致外部理解混乱的问题。工作包括定义背景与观点摘要标准模式，构建专家标注基准数据集，设计分层评估指标，并测试12个LLM。发现模型在知识整合、自我修正方面存在显著不足，最高性能不足69%。**

- **链接: [http://arxiv.org/pdf/2505.12474v1](http://arxiv.org/pdf/2505.12474v1)**

> **作者:** Weixiao Zhou; Junnan Zhu; Gengyao Li; Xianfu Cheng; Xinnian Liang; Feifei Zhai; Zhoujun Li
>
> **备注:** Submitted to EMNLP 2025
>
> **摘要:** In this work, we investigate the performance of LLMs on a new task that requires combining discussion with background knowledge for summarization. This aims to address the limitation of outside observer confusion in existing dialogue summarization systems due to their reliance solely on discussion information. To achieve this, we model the task output as background and opinion summaries and define two standardized summarization patterns. To support assessment, we introduce the first benchmark comprising high-quality samples consistently annotated by human experts and propose a novel hierarchical evaluation framework with fine-grained, interpretable metrics. We evaluate 12 LLMs under structured-prompt and self-reflection paradigms. Our findings reveal: (1) LLMs struggle with background summary retrieval, generation, and opinion summary integration. (2) Even top LLMs achieve less than 69% average performance across both patterns. (3) Current LLMs lack adequate self-evaluation and self-correction capabilities for this task.
>
---
#### [new 190] Distribution Prompting: Understanding the Expressivity of Language Models Through the Next-Token Distributions They Can Produce
- **分类: cs.CL**

- **简介: 该论文研究语言模型生成概率分布的能力，属于模型可解释性任务。旨在探究不同目标分布能否通过提示调整被模型准确生成。通过软/硬梯度优化提示，发现低/高熵分布、含异常词或模型自生成的分布更易近似，揭示了LM作为分布生成器的表达局限。**

- **链接: [http://arxiv.org/pdf/2505.12244v1](http://arxiv.org/pdf/2505.12244v1)**

> **作者:** Haojin Wang; Zining Zhu; Freda Shi
>
> **摘要:** Autoregressive neural language models (LMs) generate a probability distribution over tokens at each time step given a prompt. In this work, we attempt to systematically understand the probability distributions that LMs can produce, showing that some distributions are significantly harder to elicit than others. Specifically, for any target next-token distribution over the vocabulary, we attempt to find a prompt that induces the LM to output a distribution as close as possible to the target, using either soft or hard gradient-based prompt tuning. We find that (1) in general, distributions with very low or very high entropy are easier to approximate than those with moderate entropy; (2) among distributions with the same entropy, those containing ''outlier tokens'' are easier to approximate; (3) target distributions generated by LMs -- even LMs with different tokenizers -- are easier to approximate than randomly chosen targets. These results offer insights into the expressiveness of LMs and the challenges of using them as probability distribution proposers.
>
---
#### [new 191] Decentralized Arena: Towards Democratic and Scalable Automatic Evaluation of Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型自动化评估领域，旨在解决现有基准测试易饱和、人工评估成本高及单模型评判偏差问题。提出去中心化框架dearena，利用全体模型集体智慧相互评估，通过民主化双对比机制和高效排序算法（亚二次复杂度），结合自动问题生成策略，在66个模型实验中实现97%人类评分相关性并显著降低成本。**

- **链接: [http://arxiv.org/pdf/2505.12808v1](http://arxiv.org/pdf/2505.12808v1)**

> **作者:** Yanbin Yin; Kun Zhou; Zhen Wang; Xiangdong Zhang; Yifei Shao; Shibo Hao; Yi Gu; Jieyuan Liu; Somanshu Singla; Tianyang Liu; Eric P. Xing; Zhengzhong Liu; Haojian Jin; Zhiting Hu
>
> **备注:** 20 pages, ongoing work
>
> **摘要:** The recent explosion of large language models (LLMs), each with its own general or specialized strengths, makes scalable, reliable benchmarking more urgent than ever. Standard practices nowadays face fundamental trade-offs: closed-ended question-based benchmarks (eg MMLU) struggle with saturation as newer models emerge, while crowd-sourced leaderboards (eg Chatbot Arena) rely on costly and slow human judges. Recently, automated methods (eg LLM-as-a-judge) shed light on the scalability, but risk bias by relying on one or a few "authority" models. To tackle these issues, we propose Decentralized Arena (dearena), a fully automated framework leveraging collective intelligence from all LLMs to evaluate each other. It mitigates single-model judge bias by democratic, pairwise evaluation, and remains efficient at scale through two key components: (1) a coarse-to-fine ranking algorithm for fast incremental insertion of new models with sub-quadratic complexity, and (2) an automatic question selection strategy for the construction of new evaluation dimensions. Across extensive experiments across 66 LLMs, dearena attains up to 97% correlation with human judgements, while significantly reducing the cost. Our code and data will be publicly released on https://github.com/maitrix-org/de-arena.
>
---
#### [new 192] Enhance Mobile Agents Thinking Process Via Iterative Preference Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于移动GUI代理推理优化任务，旨在解决CoA T轨迹数据不足导致泛化性差的问题。提出迭代偏好学习框架IPL，通过规则奖励构建CoA T树、生成思维级偏好对，并设计三阶段指令进化增强训练数据多样性，实验证明其代理MobileIPL在多个基准上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.12299v1](http://arxiv.org/pdf/2505.12299v1)**

> **作者:** Kun Huang; Weikai Xu; Yuxuan Liu; Quandong Wang; Pengzhi Gao; Wei Liu; Jian Luan; Bin Wang; Bo An
>
> **备注:** 9 pages, 8 figures, 7 tables
>
> **摘要:** The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios.
>
---
#### [new 193] Multilingual Collaborative Defense for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）在多语言场景下易被翻译攻击绕过防护的漏洞，提出多语言协作防御（MCD）方法，通过自动优化安全提示提升跨语言安全性，解决语言训练数据不平衡导致的安全对齐问题，并构建多语言基准验证其防御效果和迁移能力。**

- **链接: [http://arxiv.org/pdf/2505.11835v1](http://arxiv.org/pdf/2505.11835v1)**

> **作者:** Hongliang Li; Jinan Xu; Gengping Cui; Changhao Guan; Fengran Mo; Kaiyu Huang
>
> **备注:** 19 pages, 4figures
>
> **摘要:** The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.
>
---
#### [new 194] AI-Driven Automation Can Become the Foundation of Next-Era Science of Science Research
- **分类: cs.AI; cs.CL; physics.soc-ph**

- **简介: 该论文属于科学学研究的方法论创新，旨在解决传统统计工具（如线性回归）难以分析复杂科研生态的问题。提出利用AI自动化发现科研规律，分析其优势与局限，并构建多智能体系统模拟科研社会，加速科学学发展。**

- **链接: [http://arxiv.org/pdf/2505.12039v1](http://arxiv.org/pdf/2505.12039v1)**

> **作者:** Renqi Chen; Haoyang Su; Shixiang Tang; Zhenfei Yin; Qi Wu; Hui Li; Ye Sun; Nanqing Dong; Wanli Ouyang; Philip Torr
>
> **摘要:** The Science of Science (SoS) explores the mechanisms underlying scientific discovery, and offers valuable insights for enhancing scientific efficiency and fostering innovation. Traditional approaches often rely on simplistic assumptions and basic statistical tools, such as linear regression and rule-based simulations, which struggle to capture the complexity and scale of modern research ecosystems. The advent of artificial intelligence (AI) presents a transformative opportunity for the next generation of SoS, enabling the automation of large-scale pattern discovery and uncovering insights previously unattainable. This paper offers a forward-looking perspective on the integration of Science of Science with AI for automated research pattern discovery and highlights key open challenges that could greatly benefit from AI. We outline the advantages of AI over traditional methods, discuss potential limitations, and propose pathways to overcome them. Additionally, we present a preliminary multi-agent system as an illustrative example to simulate research societies, showcasing AI's ability to replicate real-world research patterns and accelerate progress in Science of Science research.
>
---
#### [new 195] Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于认知建模任务，旨在解决神经网络预测人类行为时缺乏可解释性的问题。研究者利用强化学习训练大语言模型，使其同时生成人类风险决策的自然语言解释和精准预测，通过结果奖励机制引导模型输出显式推理过程，实现了预测性能与解释能力的结合。**

- **链接: [http://arxiv.org/pdf/2505.11614v1](http://arxiv.org/pdf/2505.11614v1)**

> **作者:** Jian-Qiao Zhu; Hanbo Xie; Dilip Arumugam; Robert C. Wilson; Thomas L. Griffiths
>
> **摘要:** A central goal of cognitive modeling is to develop models that not only predict human behavior but also provide insight into the underlying cognitive mechanisms. While neural network models trained on large-scale behavioral data often achieve strong predictive performance, they typically fall short in offering interpretable explanations of the cognitive processes they capture. In this work, we explore the potential of pretrained large language models (LLMs) to serve as dual-purpose cognitive models--capable of both accurate prediction and interpretable explanation in natural language. Specifically, we employ reinforcement learning with outcome-based rewards to guide LLMs toward generating explicit reasoning traces for explaining human risky choices. Our findings demonstrate that this approach produces high-quality explanations alongside strong quantitative predictions of human decisions.
>
---
#### [new 196] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出自动化框架MONDAY，构建大规模跨平台移动OS导航数据集（313K标注帧），解决现有单OS数据集泛化性差的问题。通过OCR检测、UI识别及多步动作提取，自动生成任务数据，提升模型在未见OS上的性能（平均+18.11%），支持持续扩展。**

- **链接: [http://arxiv.org/pdf/2505.12632v1](http://arxiv.org/pdf/2505.12632v1)**

> **作者:** Yunseok Jang; Yeda Song; Sungryull Sohn; Lajanugen Logeswaran; Tiange Luo; Dong-Ki Kim; Kyunghoon Bae; Honglak Lee
>
> **备注:** CVPR 2025
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing GUI visual agents. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models that include MONDAY in their pre-training phases demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving an average performance gain of 18.11%p on an unseen mobile OS platform. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework comprises robust OCR-based scene detection (95.04% F1score), near-perfect UI element detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation.
>
---
#### [new 197] Probing the Vulnerability of Large Language Models to Polysemantic Interventions
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究大语言模型的多义性漏洞（安全性与可解释性任务），揭示神经元编码多个无关特征的结构共性。通过稀疏自编码器分析小模型，开发针对性干预方法，并验证其在黑盒大模型的有效性，证明多义结构的跨模型稳定性和安全风险。**

- **链接: [http://arxiv.org/pdf/2505.11611v1](http://arxiv.org/pdf/2505.11611v1)**

> **作者:** Bofan Gong; Shiyang Lai; Dawn Song
>
> **摘要:** Polysemanticity -- where individual neurons encode multiple unrelated features -- is a well-known characteristic of large neural networks and remains a central challenge in the interpretability of language models. At the same time, its implications for model safety are also poorly understood. Leveraging recent advances in sparse autoencoders, we investigate the polysemantic structure of two small models (Pythia-70M and GPT-2-Small) and evaluate their vulnerability to targeted, covert interventions at the prompt, feature, token, and neuron levels. Our analysis reveals a consistent polysemantic topology shared across both models. Strikingly, we demonstrate that this structure can be exploited to mount effective interventions on two larger, black-box instruction-tuned models (LLaMA3.1-8B-Instruct and Gemma-2-9B-Instruct). These findings suggest not only the generalizability of the interventions but also point to a stable and transferable polysemantic structure that could potentially persist across architectures and training regimes.
>
---
#### [new 198] TARGET: Benchmarking Table Retrieval for Generative Tasks
- **分类: cs.IR; cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于表格检索评估任务，旨在解决结构化数据中如何准确检索相关表格以支持生成式任务的问题。研究者提出了TARGET基准，评估不同检索方法（如密集嵌入与BM25）的性能及其对下游任务的影响，发现密集检索优于传统方法，并揭示了元数据缺失对检索效果的敏感性。**

- **链接: [http://arxiv.org/pdf/2505.11545v1](http://arxiv.org/pdf/2505.11545v1)**

> **作者:** Xingyu Ji; Parker Glenn; Aditya G. Parameswaran; Madelon Hulsebos
>
> **摘要:** The data landscape is rich with structured data, often of high value to organizations, driving important applications in data analysis and machine learning. Recent progress in representation learning and generative models for such data has led to the development of natural language interfaces to structured data, including those leveraging text-to-SQL. Contextualizing interactions, either through conversational interfaces or agentic components, in structured data through retrieval-augmented generation can provide substantial benefits in the form of freshness, accuracy, and comprehensiveness of answers. The key question is: how do we retrieve the right table(s) for the analytical query or task at hand? To this end, we introduce TARGET: a benchmark for evaluating TAble Retrieval for GEnerative Tasks. With TARGET we analyze the retrieval performance of different retrievers in isolation, as well as their impact on downstream tasks. We find that dense embedding-based retrievers far outperform a BM25 baseline which is less effective than it is for retrieval over unstructured text. We also surface the sensitivity of retrievers across various metadata (e.g., missing table titles), and demonstrate a stark variation of retrieval performance across datasets and tasks. TARGET is available at https://target-benchmark.github.io.
>
---
#### [new 199] LogicOCR: Do Your Large Multimodal Models Excel at Logical Reasoning on Text-Rich Images?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态逻辑推理评估任务，旨在解决大型多模态模型（LMMs）在文本密集图像上的复杂逻辑推理能力不足的问题。研究构建了LogicOCR基准数据集（含1,100多选题），通过自动化流程将公务员考试文本转化为多样化图文样本，并评估主流LMMs，揭示其视觉-文本推理的局限性。**

- **链接: [http://arxiv.org/pdf/2505.12307v1](http://arxiv.org/pdf/2505.12307v1)**

> **作者:** Maoyuan Ye; Jing Zhang; Juhua Liu; Bo Du; Dacheng Tao
>
> **备注:** GitHub: \url{https://github.com/MiliLab/LogicOCR}
>
> **摘要:** Recent advances in Large Multimodal Models (LMMs) have significantly improved their reasoning and Optical Character Recognition (OCR) capabilities. However, their performance on complex logical reasoning tasks involving text-rich images remains underexplored. To bridge this gap, we introduce LogicOCR, a benchmark comprising 1,100 multiple-choice questions designed to evaluate LMMs' logical reasoning abilities on text-rich images, while minimizing reliance on domain-specific knowledge (e.g., mathematics). We construct LogicOCR by curating a text corpus from the Chinese National Civil Servant Examination and develop a scalable, automated pipeline to convert it into multimodal samples. First, we design prompt templates to steer GPT-Image-1 to generate images with diverse backgrounds, interleaved text-illustration layouts, and varied fonts, ensuring contextual relevance and visual realism. Then, the generated images are manually verified, with low-quality examples discarded. We evaluate a range of representative open-source and proprietary LMMs under both Chain-of-Thought (CoT) and direct-answer settings. Our multi-dimensional analysis reveals key insights, such as the impact of test-time scaling, input modality differences, and sensitivity to visual-text orientation. Notably, LMMs still lag in multimodal reasoning compared to text-only inputs, indicating that they have not fully bridged visual reading with reasoning. We hope LogicOCR will serve as a valuable resource for advancing multimodal reasoning research. The dataset is available at https://github.com/MiliLab/LogicOCR.
>
---
#### [new 200] Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI模型可信性任务，针对大型推理模型(LRMs)中逻辑合理但事实错误的"推理幻觉"问题。提出Reasoning Score量化推理深度，发现两种关键幻觉模式，开发检测框架RHD和强化学习算法GRPO-R，有效降低幻觉率并提升推理质量。**

- **链接: [http://arxiv.org/pdf/2505.12886v1](http://arxiv.org/pdf/2505.12886v1)**

> **作者:** Zhongxiang Sun; Qipeng Wang; Haoyu Wang; Xiao Zhang; Jun Xu
>
> **备注:** 25 pages
>
> **摘要:** Large Reasoning Models (LRMs) have shown impressive capabilities in multi-step reasoning tasks. However, alongside these successes, a more deceptive form of model error has emerged--Reasoning Hallucination--where logically coherent but factually incorrect reasoning traces lead to persuasive yet faulty conclusions. Unlike traditional hallucinations, these errors are embedded within structured reasoning, making them more difficult to detect and potentially more harmful. In this work, we investigate reasoning hallucinations from a mechanistic perspective. We propose the Reasoning Score, which quantifies the depth of reasoning by measuring the divergence between logits obtained from projecting late layers of LRMs to the vocabulary space, effectively distinguishing shallow pattern-matching from genuine deep reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA dataset and identify two key reasoning hallucination patterns: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps. These insights motivate our Reasoning Hallucination Detection (RHD) framework, which achieves state-of-the-art performance across multiple domains. To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced reinforcement learning algorithm that incorporates step-level deep reasoning rewards via potential-based shaping. Our theoretical analysis establishes stronger generalization guarantees, and experiments demonstrate improved reasoning quality and reduced hallucination rates.
>
---
#### [new 201] Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型（LLMs）在推理中自我验证不足的问题。提出RISE框架，通过整合问题生成与自我验证的在线强化学习，利用可验证奖励同步优化模型解题和验证能力，实验证明其能提升推理准确性和自检频率。**

- **链接: [http://arxiv.org/pdf/2505.13445v1](http://arxiv.org/pdf/2505.13445v1)**

> **作者:** Xiaoyuan Liu; Tian Liang; Zhiwei He; Jiahao Xu; Wenxuan Wang; Pinjia He; Zhaopeng Tu; Haitao Mi; Dong Yu
>
> **备注:** code available at https://github.com/xyliu-cs/RISE
>
> **摘要:** Large Language Models (LLMs) show great promise in complex reasoning, with Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement strategy. However, a prevalent issue is ``superficial self-reflection'', where models fail to robustly verify their own outputs. We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online RL framework designed to tackle this. RISE explicitly and simultaneously trains an LLM to improve both its problem-solving and self-verification abilities within a single, integrated RL process. The core mechanism involves leveraging verifiable rewards from an outcome verifier to provide on-the-fly feedback for both solution generation and self-verification tasks. In each iteration, the model generates solutions, then critiques its own on-policy generated solutions, with both trajectories contributing to the policy update. Extensive experiments on diverse mathematical reasoning benchmarks show that RISE consistently improves model's problem-solving accuracy while concurrently fostering strong self-verification skills. Our analyses highlight the advantages of online verification and the benefits of increased verification compute. Additionally, RISE models exhibit more frequent and accurate self-verification behaviors during reasoning. These advantages reinforce RISE as a flexible and effective path towards developing more robust and self-aware reasoners.
>
---
#### [new 202] Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究稀疏自编码器（SAE）在特征解耦中的缺陷，属于可解释性AI任务。发现当SAE宽度小于真实特征数且特征相关时，会混合相关特征（特征对冲），破坏单义性。通过理论分析和LLM实验验证该现象，指出其导致SAE性能下降，并提出改进的嵌套结构SAE变体。**

- **链接: [http://arxiv.org/pdf/2505.11756v1](http://arxiv.org/pdf/2505.11756v1)**

> **作者:** David Chanin; Tomáš Dulka; Adrià Garriga-Alonso
>
> **摘要:** It is assumed that sparse autoencoders (SAEs) decompose polysemantic activations into interpretable linear directions, as long as the activations are composed of sparse linear combinations of underlying features. However, we find that if an SAE is more narrow than the number of underlying "true features" on which it is trained, and there is correlation between features, the SAE will merge components of correlated features together, thus destroying monosemanticity. In LLM SAEs, these two conditions are almost certainly true. This phenomenon, which we call feature hedging, is caused by SAE reconstruction loss, and is more severe the narrower the SAE. In this work, we introduce the problem of feature hedging and study it both theoretically in toy models and empirically in SAEs trained on LLMs. We suspect that feature hedging may be one of the core reasons that SAEs consistently underperform supervised baselines. Finally, we use our understanding of feature hedging to propose an improved variant of matryoshka SAEs. Our work shows there remain fundamental issues with SAEs, but we are hopeful that that highlighting feature hedging will catalyze future advances that allow SAEs to achieve their full potential of interpreting LLMs at scale.
>
---
#### [new 203] LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LLM-BabyBench基准套件，针对大型语言模型在交互环境中的基础智能评估，属于推理与规划任务。旨在解决LLMs在环境状态预测、低层动作规划和指令分解能力上的评测问题。通过文本化BabyAI环境构建三个数据集（Predict/Plan/Decompose），开发标准化评估框架并开源数据代码，揭示模型在具象推理任务中的不足。**

- **链接: [http://arxiv.org/pdf/2505.12135v1](http://arxiv.org/pdf/2505.12135v1)**

> **作者:** Omar Choukrani; Idriss Malek; Daniil Orel; Zhuohan Xie; Zangir Iklassov; Martin Takáč; Salem Lahlou
>
> **摘要:** Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{https://github.com/choukrani/llm-babybench}{\text{GitHub}}$, $\href{https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench}{\text{HuggingFace}}$).
>
---
#### [new 204] Introduction to Analytical Software Engineering Design Paradigm
- **分类: cs.SE; cs.AI; cs.CL; cs.MS; cs.PL**

- **简介: 该论文提出"分析型软件工程"（ASE）设计范式，解决传统方法在复杂软件系统建模、设计模式检测和代码重构中的不足。通过BSS框架实现语言无关的代码表征，利用ODR框架优化重构算法，建立结构化方法平衡抽象、可扩展性等要素，提升软件维护与优化的效率。**

- **链接: [http://arxiv.org/pdf/2505.11979v1](http://arxiv.org/pdf/2505.11979v1)**

> **作者:** Tarik Houichime; Younes El Amrani
>
> **备注:** The Conference's autorization to submit a preprint was granted
>
> **摘要:** As modern software systems expand in scale and complexity, the challenges associated with their modeling and formulation grow increasingly intricate. Traditional approaches often fall short in effectively addressing these complexities, particularly in tasks such as design pattern detection for maintenance and assessment, as well as code refactoring for optimization and long-term sustainability. This growing inadequacy underscores the need for a paradigm shift in how such challenges are approached and resolved. This paper presents Analytical Software Engineering (ASE), a novel design paradigm aimed at balancing abstraction, tool accessibility, compatibility, and scalability. ASE enables effective modeling and resolution of complex software engineering problems. The paradigm is evaluated through two frameworks Behavioral-Structural Sequences (BSS) and Optimized Design Refactoring (ODR), both developed in accordance with ASE principles. BSS offers a compact, language-agnostic representation of codebases to facilitate precise design pattern detection. ODR unifies artifact and solution representations to optimize code refactoring via heuristic algorithms while eliminating iterative computational overhead. By providing a structured approach to software design challenges, ASE lays the groundwork for future research in encoding and analyzing complex software metrics.
>
---
#### [new 205] Efficient Uncertainty Estimation via Distillation of Bayesian Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大型语言模型不确定性估计任务，解决贝叶斯方法需多次采样导致的低效问题。通过蒸馏贝叶斯模型的预测分布到非贝叶斯学生模型，消除测试时采样需求，仅用训练数据即实现N倍效率提升，实验表明蒸馏后不确定性估计效果达到或超越原贝叶斯模型。**

- **链接: [http://arxiv.org/pdf/2505.11731v1](http://arxiv.org/pdf/2505.11731v1)**

> **作者:** Harshil Vejendla; Haizhou Shi; Yibin Wang; Tunyu Zhang; Huan Zhang; Hao Wang
>
> **备注:** Preprint; work in progress
>
> **摘要:** Recent advances in uncertainty estimation for Large Language Models (LLMs) during downstream adaptation have addressed key challenges of reliability and simplicity. However, existing Bayesian methods typically require multiple sampling iterations during inference, creating significant efficiency issues that limit practical deployment. In this paper, we investigate the possibility of eliminating the need for test-time sampling for LLM uncertainty estimation. Specifically, when given an off-the-shelf Bayesian LLM, we distill its aligned confidence into a non-Bayesian student LLM by minimizing the divergence between their predictive distributions. Unlike typical calibration methods, our distillation is carried out solely on the training dataset without the need of an additional validation dataset. This simple yet effective approach achieves N-times more efficient uncertainty estimation during testing, where N is the number of samples traditionally required by Bayesian LLMs. Our extensive experiments demonstrate that uncertainty estimation capabilities on training data can successfully generalize to unseen test data through our distillation technique, consistently producing results comparable to (or even better than) state-of-the-art Bayesian LLMs.
>
---
#### [new 206] GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究GUI代理的分布外（OOD）检测任务，解决其因OOD指令导致任务崩溃或安全威胁的问题。提出GEM方法，通过高斯混合模型拟合输入嵌入距离以界定能力边界。实验在8个数据集上实现23.7%的平均准确率提升，并在9种主干网络上验证了泛化性。**

- **链接: [http://arxiv.org/pdf/2505.12842v1](http://arxiv.org/pdf/2505.12842v1)**

> **作者:** Zheng Wu; Pengzhou Cheng; Zongru Wu; Lingzhong Dong; Zhuosheng Zhang
>
> **摘要:** Graphical user interface (GUI) agents have recently emerged as an intriguing paradigm for human-computer interaction, capable of automatically executing user instructions to operate intelligent terminal devices. However, when encountering out-of-distribution (OOD) instructions that violate environmental constraints or exceed the current capabilities of agents, GUI agents may suffer task breakdowns or even pose security threats. Therefore, effective OOD detection for GUI agents is essential. Traditional OOD detection methods perform suboptimally in this domain due to the complex embedding space and evolving GUI environments. In this work, we observe that the in-distribution input semantic space of GUI agents exhibits a clustering pattern with respect to the distance from the centroid. Based on the finding, we propose GEM, a novel method based on fitting a Gaussian mixture model over input embedding distances extracted from the GUI Agent that reflect its capability boundary. Evaluated on eight datasets spanning smartphones, computers, and web browsers, our method achieves an average accuracy improvement of 23.70\% over the best-performing baseline. Analysis verifies the generalization ability of our method through experiments on nine different backbones. The codes are available at https://github.com/Wuzheng02/GEM-OODforGUIagents.
>
---
#### [new 207] Optimizing Anytime Reasoning via Budget Relative Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型推理效率优化，提出AnytimeReasoner框架，解决固定计算预算下训练与部署效率低的问题。通过动态截断思考路径并引入密集奖励机制，结合BRPO策略降低方差，实现不同预算下的高效推理。实验表明其在数学任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13438v1](http://arxiv.org/pdf/2505.13438v1)**

> **作者:** Penghui Qi; Zichen Liu; Tianyu Pang; Chao Du; Wee Sun Lee; Min Lin
>
> **摘要:** Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency.
>
---
#### [new 208] CompeteSMoE -- Statistically Guaranteed Mixture of Experts Training via Competition
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于稀疏混合专家（SMoE）模型训练领域，旨在解决传统路由机制中专家计算与路由决策脱节导致的效率低下问题。提出CompeteSMoE算法，通过竞争机制将令牌路由至神经响应最高的专家，理论证明其样本效率优于传统路由，实验验证其在视觉和语言任务中性能更优且训练成本低。**

- **链接: [http://arxiv.org/pdf/2505.13380v1](http://arxiv.org/pdf/2505.13380v1)**

> **作者:** Nam V. Nguyen; Huy Nguyen; Quang Pham; Van Nguyen; Savitha Ramasamy; Nhat Ho
>
> **备注:** 52 pages. This work is an improved version of the previous study at arXiv:2402.02526
>
> **摘要:** Sparse mixture of experts (SMoE) offers an appealing solution to scale up the model complexity beyond the mean of increasing the network's depth or width. However, we argue that effective SMoE training remains challenging because of the suboptimal routing process where experts that perform computation do not directly contribute to the routing process. In this work, we propose competition, a novel mechanism to route tokens to experts with the highest neural response. Theoretically, we show that the competition mechanism enjoys a better sample efficiency than the traditional softmax routing. Furthermore, we develop CompeteSMoE, a simple yet effective algorithm to train large language models by deploying a router to learn the competition policy, thus enjoying strong performances at a low training overhead. Our extensive empirical evaluations on both the visual instruction tuning and language pre-training tasks demonstrate the efficacy, robustness, and scalability of CompeteSMoE compared to state-of-the-art SMoE strategies. We have made the implementation available at: https://github.com/Fsoft-AIC/CompeteSMoE. This work is an improved version of the previous study at arXiv:2402.02526
>
---
#### [new 209] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态安全评估任务，旨在解决视频大视觉语言模型(LVLMs)在动态视频诱导攻击下的安全隐患。通过构建含2,264个视频-文本对的Video-SafetyBench基准，结合可控视频合成方法和新型评估指标RJScore，揭示了模型对良性查询视频攻击67.2%的平均漏洞率。**

- **链接: [http://arxiv.org/pdf/2505.11842v1](http://arxiv.org/pdf/2505.11842v1)**

> **作者:** Xuannan Liu; Zekun Li; Zheqi He; Peipei Li; Shuhan Xia; Xing Cui; Huaibo Huang; Xi Yang; Ran He
>
> **备注:** Project page: https://liuxuannan.github.io/Video-SafetyBench.github.io/
>
> **摘要:** The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.
>
---
#### [new 210] Leveraging LLM Inconsistency to Boost Pass@k Performance
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型性能优化任务，旨在利用LLMs输出不一致性提升Pass@k指标。为解决传统方法忽视输入敏感性缺陷的问题，提出"Variator"代理生成多任务变体并输出对应解，通过概率模型验证其有效性，在APPS数据集超越基线。实验证明前沿模型仍存在不一致性，验证了方法的持续适用性。**

- **链接: [http://arxiv.org/pdf/2505.12938v1](http://arxiv.org/pdf/2505.12938v1)**

> **作者:** Uri Dalal; Meirav Segal; Zvika Ben-Haim; Dan Lahav; Omer Nevo
>
> **摘要:** Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations.
>
---
#### [new 211] mCLM: A Function-Infused and Synthesis-Friendly Modular Chemical Language Model
- **分类: cs.AI; cs.CL; cs.LG; q-bio.QM**

- **简介: 该论文属于药物发现领域，解决传统语言模型生成分子合成困难且功能不足的问题。提出模块化化学语言模型mCLM，通过将分子分解为功能构建块（类似文本分词），结合自然语言描述训练双语模型，确保生成易合成且功能优化的分子。实验证明其能改善FDA药物功能缺陷。**

- **链接: [http://arxiv.org/pdf/2505.12565v1](http://arxiv.org/pdf/2505.12565v1)**

> **作者:** Carl Edwards; Chi Han; Gawon Lee; Thao Nguyen; Bowen Jin; Chetan Kumar Prasad; Sara Szymkuć; Bartosz A. Grzybowski; Ying Diao; Jiawei Han; Ge Liu; Hao Peng; Martin D. Burke; Heng Ji
>
> **摘要:** Despite their ability to understand chemical knowledge and accurately generate sequential representations, large language models (LLMs) remain limited in their capacity to propose novel molecules with drug-like properties. In addition, the molecules that LLMs propose can often be challenging to make in the lab. To more effectively enable the discovery of functional small molecules, LLMs need to learn a molecular language. However, LLMs are currently limited by encoding molecules from atoms. In this paper, we argue that just like tokenizing texts into (sub-)word tokens instead of characters, molecules should be decomposed and reassembled at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model tokenizing molecules into building blocks and learning a bilingual language model of both natural language descriptions of functions and molecule building blocks. By reasoning on such functional building blocks, mCLM guarantees to generate efficiently synthesizable molecules thanks to recent progress in block-based chemistry, while also improving the functions of molecules in a principled manner. In experiments on 430 FDA-approved drugs, we find mCLM capable of significantly improving 5 out of 6 chemical functions critical to determining drug potentials. More importantly, mCLM can reason on multiple functions and improve the FDA-rejected drugs (``fallen angels'') over multiple iterations to greatly improve their shortcomings.
>
---
#### [new 212] EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文属于大语言模型编程能力评估任务，旨在解决现有方法忽视模型鲁棒性且评估结果不一致的问题。提出EVALOOP框架，通过自洽循环（如代码生成与摘要互转）构建反馈循环，无需外部攻击即可量化LLM鲁棒性。实验发现多数模型在10次循环后性能下降5.01%-19.31%，且初始性能与鲁棒性无必然关联。**

- **链接: [http://arxiv.org/pdf/2505.12185v1](http://arxiv.org/pdf/2505.12185v1)**

> **作者:** Sen Fang; Weiyuan Ding; Bowen Xu
>
> **备注:** 19 pages, 11 figures
>
> **摘要:** Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.
>
---
#### [new 213] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于视频空间理解与推理任务，旨在解决现有视觉语言模型在视频时空认知中的不足。提出了ViCA-322K数据集（含32万视频QA对）和ViCA-7B模型，在8项基准任务中刷新SOTA，并通过ViCA-Thinking-2.68K数据集增强模型可解释性，推动具身AI的时空建模研究。**

- **链接: [http://arxiv.org/pdf/2505.12312v1](http://arxiv.org/pdf/2505.12312v1)**

> **作者:** Qi Feng; Hidetoshi Shimodaira
>
> **备注:** 31 pages, 10 figures, 6 tables. The implementation and fine-tuned model (ViCA-7B) are publicly available at https://huggingface.co/nkkbr/ViCA. The ViCA-322K dataset can be found at https://huggingface.co/datasets/nkkbr/ViCA-322K, and the ViCA-Thinking-2.68K dataset is at https://huggingface.co/datasets/nkkbr/ViCA-thinking-2.68k
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [new 214] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究强化学习人类反馈（RLHF）中奖励模型（RM）的评估问题，针对现有基准与策略性能弱相关的问题，提出通过奖励过优化现象设计可靠评估方法，强调减少非必要差异、多维度比较及多样化数据源，并指出过优化程度与下游效果的平衡。**

- **链接: [http://arxiv.org/pdf/2505.12763v1](http://arxiv.org/pdf/2505.12763v1)**

> **作者:** Sunghwan Kim; Dongjin Kang; Taeyoon Kwon; Hyungjoo Chae; Dongha Lee; Jinyoung Yeo
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Reward models (RMs) play a crucial role in reinforcement learning from human feedback (RLHF), aligning model behavior with human preferences. However, existing benchmarks for reward models show a weak correlation with the performance of optimized policies, suggesting that they fail to accurately assess the true capabilities of RMs. To bridge this gap, we explore several evaluation designs through the lens of reward overoptimization\textemdash a phenomenon that captures both how well the reward model aligns with human preferences and the dynamics of the learning signal it provides to the policy. The results highlight three key findings on how to construct a reliable benchmark: (i) it is important to minimize differences between chosen and rejected responses beyond correctness, (ii) evaluating reward models requires multiple comparisons across a wide range of chosen and rejected responses, and (iii) given that reward models encounter responses with diverse representations, responses should be sourced from a variety of models. However, we also observe that a extremely high correlation with degree of overoptimization leads to comparatively lower correlation with certain downstream performance. Thus, when designing a benchmark, it is desirable to use the degree of overoptimization as a useful tool, rather than the end goal.
>
---
#### [new 215] ASR-FAIRBENCH: Measuring and Benchmarking Equity Across Speech Recognition Systems
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别公平性评估任务，旨在解决ASR系统在不同人口群体中的性能差异问题。研究者构建了ASR-FAIRBENCH评测框架，结合公平评分（通过混合效应模型计算）与词错误率，提出FAAS综合指标，利用Fair-Speech数据集揭示主流模型的群体性能差距，为开发包容性技术提供基准。**

- **链接: [http://arxiv.org/pdf/2505.11572v1](http://arxiv.org/pdf/2505.11572v1)**

> **作者:** Anand Rai; Satyam Rahangdale; Utkarsh Anand; Animesh Mukherjee
>
> **备注:** Paper accepted at INTERSPEECH 2025
>
> **摘要:** Automatic Speech Recognition (ASR) systems have become ubiquitous in everyday applications, yet significant disparities in performance across diverse demographic groups persist. In this work, we introduce the ASR-FAIRBENCH leaderboard which is designed to assess both the accuracy and equity of ASR models in real-time. Leveraging the Meta's Fair-Speech dataset, which captures diverse demographic characteristics, we employ a mixed-effects Poisson regression model to derive an overall fairness score. This score is integrated with traditional metrics like Word Error Rate (WER) to compute the Fairness Adjusted ASR Score (FAAS), providing a comprehensive evaluation framework. Our approach reveals significant performance disparities in SOTA ASR models across demographic groups and offers a benchmark to drive the development of more inclusive ASR technologies.
>
---
#### [new 216] Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于语言模型可解释性研究，旨在解决如何利用内部因果机制预测模型在分布外数据上的行为问题。通过符号操作、知识检索等任务，提出反事实模拟和值探测两种方法，验证因果特征能有效预测模型输出正确性，优于非因果方法。**

- **链接: [http://arxiv.org/pdf/2505.11770v1](http://arxiv.org/pdf/2505.11770v1)**

> **作者:** Jing Huang; Junyi Tao; Thomas Icard; Diyi Yang; Christopher Potts
>
> **备注:** ICML 2025
>
> **摘要:** Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples? In this work, we provide a positive answer to this question. Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior. Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions). Both achieve high AUC-ROC in distribution and outperform methods that rely on causal-agnostic features in out-of-distribution settings, where predicting model behaviors is more crucial. Our work thus highlights a novel and significant application for internal causal analysis of language models.
>
---
#### [new 217] Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出LatentSeek框架，针对大语言模型推理能力不足的问题，通过测试时潜在空间的实例级策略梯度优化（无需参数更新），解决传统训练中的灾难性遗忘和数据稀缺问题。在数学推理基准测试中超越思维链提示和微调方法，实现高效收敛。**

- **链接: [http://arxiv.org/pdf/2505.13308v1](http://arxiv.org/pdf/2505.13308v1)**

> **作者:** Hengli Li; Chenxi Li; Tong Wu; Xuekai Zhu; Yuxuan Wang; Zhaoxin Yu; Eric Hanchen Jiang; Song-Chun Zhu; Zixia Jia; Ying Nian Wu; Zilong Zheng
>
> **摘要:** Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs.
>
---
#### [new 218] Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity
- **分类: cs.AI; cs.CL; 91C99; I.2.7; J.4**

- **简介: 该论文属于个性化偏好对齐任务，旨在解决现有数据集忽略个性化与偏好关联的问题。通过构建Fair-PP合成数据集（含28个社会群体、98个公平主题），提出自动化生成框架和样本重加权方法，分析主流模型在偏好空间的定位，实现LLM与目标人物设定的对齐优化。**

- **链接: [http://arxiv.org/pdf/2505.11861v1](http://arxiv.org/pdf/2505.11861v1)**

> **作者:** Qi Zhou; Jie Zhang; Dongxia Wang; Qiang Liu; Tianlin Li; Jin Song Dong; Wenhai Wang; Qing Guo
>
> **备注:** under review
>
> **摘要:** Human preference plays a crucial role in the refinement of large language models (LLMs). However, collecting human preference feedback is costly and most existing datasets neglect the correlation between personalization and preferences. To address this issue, we introduce Fair-PP, a synthetic dataset of personalized preferences targeting social equity, derived from real-world social survey data, which includes 28 social groups, 98 equity topics, and 5 personal preference dimensions. Leveraging GPT-4o-mini, we engage in role-playing based on seven representative persona portrayals guided by existing social survey data, yielding a total of 238,623 preference records. Through Fair-PP, we also contribute (i) An automated framework for generating preference data, along with a more fine-grained dataset of personalized preferences; (ii) analysis of the positioning of the existing mainstream LLMs across five major global regions within the personalized preference space; and (iii) a sample reweighting method for personalized preference alignment, enabling alignment with a target persona while maximizing the divergence from other personas. Empirical experiments show our method outperforms the baselines.
>
---
#### [new 219] Enhancing Latent Computation in Transformers with Latent Tokens
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理中的模型增强任务，旨在提升Transformer大语言模型的泛化能力。通过引入可训练的潜在标记（非自然语言符号），利用注意力机制引导解码过程，在保持预训练模型结构的同时，以低参数量方式增强模型在分布外场景的适应性。实验验证了该方法在合成任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.12629v1](http://arxiv.org/pdf/2505.12629v1)**

> **作者:** Yuchang Sun; Yanxi Chen; Yaliang Li; Bolin Ding
>
> **摘要:** Augmenting large language models (LLMs) with auxiliary tokens has emerged as a promising strategy for enhancing model performance. In this work, we introduce a lightweight method termed latent tokens; these are dummy tokens that may be non-interpretable in natural language but steer the autoregressive decoding process of a Transformer-based LLM via the attention mechanism. The proposed latent tokens can be seamlessly integrated with a pre-trained Transformer, trained in a parameter-efficient manner, and applied flexibly at inference time, while adding minimal complexity overhead to the existing infrastructure of standard Transformers. We propose several hypotheses about the underlying mechanisms of latent tokens and design synthetic tasks accordingly to verify them. Numerical results confirm that the proposed method noticeably outperforms the baselines, particularly in the out-of-distribution generalization scenarios, highlighting its potential in improving the adaptability of LLMs.
>
---
#### [new 220] Spectral Policy Optimization: Coloring your Incorrect Reasoning in GRPO
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习优化任务，旨在解决GRPO方法在"全负样本组"中无法更新策略的问题。提出通过AI反馈引入响应多样性以打破学习停滞，并理论分析了其有效性。实验验证了该方法在不同规模模型及多基准下的性能提升。**

- **链接: [http://arxiv.org/pdf/2505.11595v1](http://arxiv.org/pdf/2505.11595v1)**

> **作者:** Peter Chen; Xiaopeng Li; Ziniu Li; Xi Chen; Tianyi Lin
>
> **备注:** 28 pages
>
> **摘要:** Reinforcement learning (RL) has demonstrated significant success in enhancing reasoning capabilities in large language models (LLMs). One of the most widely used RL methods is Group Relative Policy Optimization (GRPO)~\cite{Shao-2024-Deepseekmath}, known for its memory efficiency and success in training DeepSeek-R1~\cite{Guo-2025-Deepseek}. However, GRPO stalls when all sampled responses in a group are incorrect -- referred to as an \emph{all-negative-sample} group -- as it fails to update the policy, hindering learning progress. The contributions of this paper are two-fold. First, we propose a simple yet effective framework that introduces response diversity within all-negative-sample groups in GRPO using AI feedback. We also provide a theoretical analysis, via a stylized model, showing how this diversification improves learning dynamics. Second, we empirically validate our approach, showing the improved performance across various model sizes (7B, 14B, 32B) in both offline and online learning settings with 10 benchmarks, including base and distilled variants. Our findings highlight that learning from all-negative-sample groups is not only feasible but beneficial, advancing recent insights from \citet{Xiong-2025-Minimalist}.
>
---
#### [new 221] Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation
- **分类: cs.AI; cs.CL; I.2.7; I.2.6; H.2.8**

- **简介: 该论文提出Tiny QA Benchmark++，属于持续LLM评估任务。针对传统基准测试耗时耗资源的问题，开发了轻量级多语言测试套件，包含52项英文基准、合成数据生成器和10种现成多语言包，支持快速检测模型异常，可集成至开发流程实现低延迟、低成本的质量监控。**

- **链接: [http://arxiv.org/pdf/2505.12058v1](http://arxiv.org/pdf/2505.12058v1)**

> **作者:** Vincent Koc
>
> **备注:** 28 pages, 7 figures, 3 tables. Includes expanded appendix & full score matrices. Dataset & code: HF Hub + GitHub + Pypi links in abstract. Core data and code Apache-2.0; synthetic packs eval-only
>
> **摘要:** Tiny QA Benchmark++ (TQB++) presents an ultra-lightweight, multilingual smoke-test suite designed to give large-language-model (LLM) pipelines a unit-test style safety net dataset that runs in seconds with minimal cost. Born out of the tight feedback-loop demands building the Comet Opik prompt-optimization SDK, where waiting on heavyweight benchmarks breaks developer flow. TQB++ couples a 52-item English gold set (less than 20 kB) with a tiny synthetic-data generator pypi package built on provider-agnostic LiteLLM. The generator lets practitioners mint their own tiny packs in any language, domain, or difficulty, while ten ready-made packs already cover Arabic, Chinese, French, German, Japanese, Korean, Portuguese, Russian, Spanish, and Turkish. Every dataset ships with Croissant metadata and plug-and-play files for OpenAI-Evals, LangChain, and standard CI tools, so teams can drop deterministic micro-benchmarks directly into pull-request gates, prompt-engineering loops, and production dashboards without touching GPU budgets. A complete TQB++ run adds only a few seconds to pipeline latency yet reliably flags prompt-template errors, tokenizer drift, and fine-tuning side-effects long before full-scale suites like MMLU or BIG-Bench would finish configuring. The entire framework is released to accelerate continuous, resource-efficient quality assurance across the generative-AI ecosystem.
>
---
#### [new 222] Bullying the Machine: How Personas Increase LLM Vulnerability
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在人格设定下的安全性，属于对抗攻击评估任务。通过模拟攻击框架，发现特定人格配置（如低宜人性）会显著降低模型对抗心理欺凌（如嘲讽、情感操控）的防御能力，导致不安全输出，揭示了人格驱动交互的新型安全风险，呼吁针对性防护策略。**

- **链接: [http://arxiv.org/pdf/2505.12692v1](http://arxiv.org/pdf/2505.12692v1)**

> **作者:** Ziwei Xu; Udit Sanghi; Mohan Kankanhalli
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies.
>
---
#### [new 223] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型推理鲁棒性优化任务，旨在解决LLMs因混淆内容合理性与逻辑有效性导致的推理偏差问题。研究者通过构建可控三段论数据集分离两种推理，定位关键网络层并开发动态激活引导方法（如K-CAST），有效提升形式推理准确率15%，同时保持模型语言能力，增强逻辑一致性。**

- **链接: [http://arxiv.org/pdf/2505.12189v1](http://arxiv.org/pdf/2505.12189v1)**

> **作者:** Marco Valentino; Geonhee Kim; Dhairya Dalal; Zhixue Zhao; André Freitas
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) frequently demonstrate reasoning limitations, often conflating content plausibility (i.e., material inference) with logical validity (i.e., formal inference). This can result in biased inferences, where plausible arguments are incorrectly deemed logically valid or vice versa. Mitigating this limitation is critical, as it undermines the trustworthiness and generalizability of LLMs in applications that demand rigorous logical consistency. This paper investigates the problem of mitigating content biases on formal reasoning through activation steering. Specifically, we curate a controlled syllogistic reasoning dataset to disentangle formal validity from content plausibility. After localising the layers responsible for formal and material inference, we investigate contrastive activation steering methods for test-time interventions. An extensive empirical analysis on different LLMs reveals that contrastive steering consistently supports linear control over content biases. However, we observe that a static approach is insufficient for improving all the tested models. We then leverage the possibility to control content effects by dynamically determining the value of the steering parameters via fine-grained conditional methods. We found that conditional steering is effective on unresponsive models, achieving up to 15% absolute improvement in formal reasoning accuracy with a newly introduced kNN-based method (K-CAST). Finally, additional experiments reveal that steering for content effects is robust to prompt variations, incurs minimal side effects on language modeling capabilities, and can partially generalize to out-of-distribution reasoning tasks. Practically, this paper demonstrates that activation-level interventions can offer a scalable strategy for enhancing the robustness of LLMs, contributing towards more systematic and unbiased formal reasoning.
>
---
#### [new 224] Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文针对基于大语言模型的搜索代理效率瓶颈问题，提出优化框架SearchAgent-X。任务属于LLM搜索系统的性能优化，解决了检索开销（精确检索耗时/粗略检索增加推理步骤）与系统设计缺陷（调度不当、检索停顿引发延迟）两大问题。通过高召回近似检索、优先级调度和非停顿检索技术，在保持生成质量的同时实现3.4倍吞吐量提升和5倍延迟降低。**

- **链接: [http://arxiv.org/pdf/2505.12065v1](http://arxiv.org/pdf/2505.12065v1)**

> **作者:** Tiannuo Yang; Zebin Yao; Bowen Jin; Lixiao Cui; Yusen Li; Gang Wang; Xiaoguang Liu
>
> **摘要:** Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at https://github.com/tiannuo-yang/SearchAgent-X.
>
---
#### [new 225] UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习数据选择优化任务，旨在解决LLM微调中多采样策略的高计算成本问题。提出UFO-RL框架，通过单次不确定性估计快速筛选有效训练数据（ZPD理论指导），使10%精选数据达到全量效果，训练效率提升16倍。**

- **链接: [http://arxiv.org/pdf/2505.12457v1](http://arxiv.org/pdf/2505.12457v1)**

> **作者:** Yang Zhao; Kai Xiong; Xiao Ding; Li Du; YangouOuyang; Zhouhao Sun; Jiannan Guan; Wenbin Zhang; Bin Liu; Dong Hu; Bing Qin; Ting Liu
>
> **摘要:** Scaling RL for LLMs is computationally expensive, largely due to multi-sampling for policy optimization and evaluation, making efficient data selection crucial. Inspired by the Zone of Proximal Development (ZPD) theory, we hypothesize LLMs learn best from data within their potential comprehension zone. Addressing the limitation of conventional, computationally intensive multi-sampling methods for data assessment, we introduce UFO-RL. This novel framework uses a computationally efficient single-pass uncertainty estimation to identify informative data instances, achieving up to 185x faster data evaluation. UFO-RL leverages this metric to select data within the estimated ZPD for training. Experiments show that training with just 10% of data selected by UFO-RL yields performance comparable to or surpassing full-data training, reducing overall training time by up to 16x while enhancing stability and generalization. UFO-RL offers a practical and highly efficient strategy for scaling RL fine-tuning of LLMs by focusing learning on valuable data.
>
---
#### [new 226] IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全攻击任务，针对基于LLM的多智能体系统知识产权泄露问题，提出黑盒攻击框架MASLEAK，通过构造恶意查询窃取系统架构、提示词等敏感信息，实验显示攻击成功率高达87%-92%。**

- **链接: [http://arxiv.org/pdf/2505.12442v1](http://arxiv.org/pdf/2505.12442v1)**

> **作者:** Liwen Wang; Wenxuan Wang; Shuai Wang; Zongjie Li; Zhenlan Ji; Zongyi Lyu; Daoyuan Wu; Shing-Chi Cheung
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.
>
---
#### [new 227] A Minimum Description Length Approach to Regularization in Neural Networks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于神经网络正则化方法研究，旨在解决传统正则化（如L1/L2）在复杂架构中导致模型偏离理想解的问题。通过引入最小描述长度（MDL）原则平衡模型复杂性与数据拟合，作者提出MDL能有效抑制过拟合，促进泛化，使网络从完美初始化收敛到符号化精确解而非近似解，相比现有方法具有理论优势。**

- **链接: [http://arxiv.org/pdf/2505.13398v1](http://arxiv.org/pdf/2505.13398v1)**

> **作者:** Matan Abudy; Orr Well; Emmanuel Chemla; Roni Katzir; Nur Lan
>
> **备注:** 9 pages
>
> **摘要:** State-of-the-art neural networks can be trained to become remarkable solutions to many problems. But while these architectures can express symbolic, perfect solutions, trained models often arrive at approximations instead. We show that the choice of regularization method plays a crucial role: when trained on formal languages with standard regularization ($L_1$, $L_2$, or none), expressive architectures not only fail to converge to correct solutions but are actively pushed away from perfect initializations. In contrast, applying the Minimum Description Length (MDL) principle to balance model complexity with data fit provides a theoretically grounded regularization method. Using MDL, perfect solutions are selected over approximations, independently of the optimization algorithm. We propose that unlike existing regularization techniques, MDL introduces the appropriate inductive bias to effectively counteract overfitting and promote generalization.
>
---
#### [new 228] AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models
- **分类: cs.SE; cs.AI; cs.CG; cs.CL; cs.DB**

- **简介: 该论文属于地理空间代码生成评估任务，旨在解决Google Earth Engine平台缺乏标准化自动评测工具的问题。作者提出首个多模态自动化框架AutoGEEval，构建含1325测试案例的基准集，集成代码生成与验证流程，支持准确性、效率等多维度评估，测试了18种大语言模型性能，为地理代码生成模型开发提供统一评估标准。**

- **链接: [http://arxiv.org/pdf/2505.12900v1](http://arxiv.org/pdf/2505.12900v1)**

> **作者:** Shuyang Hou; Zhangxiao Shen; Huayi Wu; Jianyuan Liang; Haoyue Jiao; Yaxian Qing; Xiaopu Zhang; Xu Li; Zhipeng Gui; Xuefeng Guan; Longgang Xiang
>
> **摘要:** Geospatial code generation is emerging as a key direction in the integration of artificial intelligence and geoscientific analysis. However, there remains a lack of standardized tools for automatic evaluation in this domain. To address this gap, we propose AutoGEEval, the first multimodal, unit-level automated evaluation framework for geospatial code generation tasks on the Google Earth Engine (GEE) platform powered by large language models (LLMs). Built upon the GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench) comprising 1325 test cases that span 26 GEE data types. The framework integrates both question generation and answer verification components to enable an end-to-end automated evaluation pipeline-from function invocation to execution validation. AutoGEEval supports multidimensional quantitative analysis of model outputs in terms of accuracy, resource consumption, execution efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including general-purpose, reasoning-augmented, code-centric, and geoscience-specialized models-revealing their performance characteristics and potential optimization pathways in GEE code generation. This work provides a unified protocol and foundational resource for the development and assessment of geospatial code generation models, advancing the frontier of automated natural language to domain-specific code translation.
>
---
#### [new 229] LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文提出LLM-KG-Bench 3.0框架，属于大语言模型评估任务，旨在解决自动评测LLMs在知识图谱和语义技术（如RDF/SPARQL处理）能力的问题。通过扩展评估任务集、改进API兼容性，并整合30余个主流模型生成数据集，实现模型在序列化任务中的性能对比与能力量化。**

- **链接: [http://arxiv.org/pdf/2505.13098v1](http://arxiv.org/pdf/2505.13098v1)**

> **作者:** Lars-Peter Meyer; Johannes Frey; Desiree Heim; Felix Brei; Claus Stadler; Kurt Junghanns; Michael Martin
>
> **备注:** Peer reviewed publication at ESWC 2025 Resources Track
>
> **摘要:** Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks.
>
---
#### [new 230] FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决长上下文场景下KV缓存存储效率低的问题。提出了FreeKV框架，通过算法层（推测性检索+细粒度校正）和系统层（混合内存布局+双缓冲流式召回）协同优化，在保证精度的前提下提升13倍KV检索速度。**

- **链接: [http://arxiv.org/pdf/2505.13109v1](http://arxiv.org/pdf/2505.13109v1)**

> **作者:** Guangda Liu; Chengwei Li; Zhenyu Ning; Jing Lin; Yiwu Yao; Danning Ke; Minyi Guo; Jieru Zhao
>
> **摘要:** Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
>
---
#### [new 231] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自动定理证明任务，旨在评估AI系统是否具备人类直觉的组合推理能力。针对现有证明器在数学不等式处理中的不足，作者构建了Ineq-Comp基准，通过变量复制、代数重构等系统变换生成复合问题。实验发现主流模型（如DeepSeek-Prover-V2）在组合推理上存在显著缺陷，即使提供子问题证明仍表现不佳，揭示了AI与人类数学直觉的差距。**

- **链接: [http://arxiv.org/pdf/2505.12680v1](http://arxiv.org/pdf/2505.12680v1)**

> **作者:** Haoyu Zhao; Yihan Geng; Shange Tang; Yong Lin; Bohan Lyu; Hongzhou Lin; Chi Jin; Sanjeev Arora
>
> **备注:** 27 pages
>
> **摘要:** LLM-based formal proof assistants (e.g., in Lean) hold great promise for automating mathematical discovery. But beyond syntactic correctness, do these systems truly understand mathematical structure as humans do? We investigate this question through the lens of mathematical inequalities -- a fundamental tool across many domains. While modern provers can solve basic inequalities, we probe their ability to handle human-intuitive compositionality. We introduce Ineq-Comp, a benchmark built from elementary inequalities through systematic transformations, including variable duplication, algebraic rewriting, and multi-step composition. Although these problems remain easy for humans, we find that most provers -- including Goedel, STP, and Kimina-7B -- struggle significantly. DeepSeek-Prover-V2-7B shows relative robustness -- possibly because it is trained to decompose the problems into sub-problems -- but still suffers a 20\% performance drop (pass@32). Strikingly, performance remains poor for all models even when formal proofs of the constituent parts are provided in context, revealing that the source of weakness is indeed in compositional reasoning. Our results expose a persisting gap between the generalization behavior of current AI provers and human mathematical intuition.
>
---
#### [new 232] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究大模型高效微调，解决GPU内存瓶颈问题。提出QZO方法，结合零阶优化（消除梯度和优化器状态）与模型量化（4-bit权重），通过扰动量化尺度估计梯度并稳定训练，相比全参数微调减少18倍内存，实现单卡微调Llama-13B等模型。**

- **链接: [http://arxiv.org/pdf/2505.13430v1](http://arxiv.org/pdf/2505.13430v1)**

> **作者:** Sifeng Shang; Jiayi Zhou; Chenyu Lin; Minxian Li; Kaiyang Zhou
>
> **摘要:** As the size of large language models grows exponentially, GPU memory has become a bottleneck for adapting these models to downstream tasks. In this paper, we aim to push the limits of memory-efficient training by minimizing memory usage on model weights, gradients, and optimizer states, within a unified framework. Our idea is to eliminate both gradients and optimizer states using zeroth-order optimization, which approximates gradients by perturbing weights during forward passes to identify gradient directions. To minimize memory usage on weights, we employ model quantization, e.g., converting from bfloat16 to int4. However, directly applying zeroth-order optimization to quantized weights is infeasible due to the precision gap between discrete weights and continuous gradients, which would otherwise require de-quantization and re-quantization. To overcome this challenge, we propose Quantized Zeroth-order Optimization (QZO), a novel approach that perturbs the continuous quantization scale for gradient estimation and uses a directional derivative clipping method to stabilize training. QZO is orthogonal to both scalar-based and codebook-based post-training quantization methods. Compared to full-parameter fine-tuning in bfloat16, QZO can reduce the total memory cost by more than 18$\times$ for 4-bit LLMs, and enables fine-tuning Llama-2-13B and Stable Diffusion 3.5 Large within a single 24GB GPU.
>
---
#### [new 233] SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多模态推理评估任务，旨在解决大型音频-语言模型（LALMs）多跳推理能力未被系统评估的问题。作者提出SAKURA基准测试，发现LALMs难以整合语音/音频表征进行多步推理，揭示了多模态推理的核心挑战，为后续研究提供资源。**

- **链接: [http://arxiv.org/pdf/2505.13237v1](http://arxiv.org/pdf/2505.13237v1)**

> **作者:** Chih-Kai Yang; Neo Ho; Yen-Ting Piao; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research.
>
---
#### [new 234] Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文针对大语言模型（LLM）奖励模型参数量大、计算成本高的问题，提出轻量级隐态奖励模型ELHSR。通过利用LLM隐藏状态信息，仅需极少量参数（<0.005%基准模型）即可在最佳N采样任务中超越基线，实现高效训练推理，并兼容闭源模型和传统奖励模型融合。**

- **链接: [http://arxiv.org/pdf/2505.12225v1](http://arxiv.org/pdf/2505.12225v1)**

> **作者:** Jizhou Guo; Zhaomin Wu; Philip S. Yu
>
> **摘要:** High-quality reward models are crucial for unlocking the reasoning potential of large language models (LLMs), with best-of-N voting demonstrating significant performance gains. However, current reward models, which typically operate on the textual output of LLMs, are computationally expensive and parameter-heavy, limiting their real-world applications. We introduce the Efficient Linear Hidden State Reward (ELHSR) model - a novel, highly parameter-efficient approach that leverages the rich information embedded in LLM hidden states to address these issues. ELHSR systematically outperform baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training. ELHSR also achieves orders-of-magnitude efficiency improvement with significantly less time and fewer FLOPs per sample than baseline reward models. Moreover, ELHSR exhibits robust performance even when trained only on logits, extending its applicability to some closed-source LLMs. In addition, ELHSR can also be combined with traditional reward models to achieve additional performance gains.
>
---
#### [new 235] J1: Exploring Simple Test-Time Scaling for LLM-as-a-Judge
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI评估任务，旨在提升LLM作为评判模型的性能与可解释性。针对传统奖励模型缺乏解释性、测试扩展不足的问题，提出J1-7B模型：先通过反思增强数据集监督微调，再结合可验证奖励的强化学习训练，并采用测试时计算扩展策略。实验显示模型性能提升4.8%，且强化学习阶段是实现有效扩展趋势的关键。**

- **链接: [http://arxiv.org/pdf/2505.11875v1](http://arxiv.org/pdf/2505.11875v1)**

> **作者:** Chi-Min Chan; Chunpu Xu; Jiaming Ji; Zhen Ye; Pengcheng Wen; Chunyang Jiang; Yaodong Yang; Wei Xue; Sirui Han; Yike Guo
>
> **备注:** 33 pages, 27 figures
>
> **摘要:** The current focus of AI research is shifting from emphasizing model training towards enhancing evaluation quality, a transition that is crucial for driving further advancements in AI systems. Traditional evaluation methods typically rely on reward models assigning scalar preference scores to outputs. Although effective, such approaches lack interpretability, leaving users often uncertain about why a reward model rates a particular response as high or low. The advent of LLM-as-a-Judge provides a more scalable and interpretable method of supervision, offering insights into the decision-making process. Moreover, with the emergence of large reasoning models, which consume more tokens for deeper thinking and answer refinement, scaling test-time computation in the LLM-as-a-Judge paradigm presents an avenue for further boosting performance and providing more interpretability through reasoning traces. In this paper, we introduce $\textbf{J1-7B}$, which is first supervised fine-tuned on reflection-enhanced datasets collected via rejection-sampling and subsequently trained using Reinforcement Learning (RL) with verifiable rewards. At inference time, we apply Simple Test-Time Scaling (STTS) strategies for additional performance improvement. Experimental results demonstrate that $\textbf{J1-7B}$ surpasses the previous state-of-the-art LLM-as-a-Judge by $ \textbf{4.8}$\% and exhibits a $ \textbf{5.1}$\% stronger scaling trend under STTS. Additionally, we present three key findings: (1) Existing LLM-as-a-Judge does not inherently exhibit such scaling trend. (2) Model simply fine-tuned on reflection-enhanced datasets continues to demonstrate similarly weak scaling behavior. (3) Significant scaling trend emerges primarily during the RL phase, suggesting that effective STTS capability is acquired predominantly through RL training.
>
---
#### [new 236] Vague Knowledge: Evidence from Analyst Reports
- **分类: econ.GN; cs.AI; cs.CL; math.LO; q-fin.EC; q-fin.GN; 03B48, 03B65, 03E02, 03E15, 03E72, 18E45, 28A05, 62F15, 68T01,
  68T35, 68T50, 91G30,; F.4; I.2.3; I.2.4; I.2.7; J.1; J.4; J.5**

- **简介: 该论文研究语言在传递模糊知识中的作用，属于金融信息分析任务。针对量化信息无法有效传达主观预期的问题，通过分析分析师报告，发现文本语调（非数值预测）能预测后续预测误差及修订，且语言越模糊、不确定性高或分析师繁忙时效果更显著，证明模糊信息需依赖语言传递。**

- **链接: [http://arxiv.org/pdf/2505.12269v1](http://arxiv.org/pdf/2505.12269v1)**

> **作者:** Kerry Xiao; Amy Zang
>
> **摘要:** People in the real world often possess vague knowledge of future payoffs, for which quantification is not feasible or desirable. We argue that language, with differing ability to convey vague information, plays an important but less known-role in subjective expectations. Empirically, we find that in their reports, analysts include useful information in linguistic expressions but not numerical forecasts. Specifically, the textual tone of analyst reports has predictive power for forecast errors and subsequent revisions in numerical forecasts, and this relation becomes stronger when analyst's language is vaguer, when uncertainty is higher, and when analysts are busier. Overall, our theory and evidence suggest that some useful information is vaguely known and only communicated through language.
>
---
#### [new 237] Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究低秩适应（LoRA）在微调大模型时对训练时攻击（如数据投毒、后门攻击）的鲁棒性，属于模型安全分析任务。通过理论建模和实验验证，发现LoRA对后门攻击更鲁棒，但对无目标数据投毒更脆弱，揭示了其低秩结构与安全风险的关联。**

- **链接: [http://arxiv.org/pdf/2505.12871v1](http://arxiv.org/pdf/2505.12871v1)**

> **作者:** Zi Liang; Haibo Hu; Qingqing Ye; Yaxin Xiao; Ronghua Li
>
> **备注:** To appear at ICML 25
>
> **摘要:** Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings.
>
---
#### [new 238] Efficient Generation of Parameterised Quantum Circuits from Large Texts
- **分类: quant-ph; cs.AI; cs.CL**

- **简介: 该论文属于量子自然语言处理（NLP）任务，旨在解决传统混合量子-经典模型依赖经典神经网络的问题。通过树状结构表示和对称单oidal范畴理论，提出高效方法将长文本（如6410词）编码为参数化量子电路（PQC），保留句法与篇章关系，并集成至开源工具lambeq Gen II。**

- **链接: [http://arxiv.org/pdf/2505.13208v1](http://arxiv.org/pdf/2505.13208v1)**

> **作者:** Colin Krawchuk; Nikhil Khatri; Neil John Ortega; Dimitri Kartsaklis
>
> **摘要:** Quantum approaches to natural language processing (NLP) are redefining how linguistic information is represented and processed. While traditional hybrid quantum-classical models rely heavily on classical neural networks, recent advancements propose a novel framework, DisCoCirc, capable of directly encoding entire documents as parameterised quantum circuits (PQCs), besides enjoying some additional interpretability and compositionality benefits. Following these ideas, this paper introduces an efficient methodology for converting large-scale texts into quantum circuits using tree-like representations of pregroup diagrams. Exploiting the compositional parallels between language and quantum mechanics, grounded in symmetric monoidal categories, our approach enables faithful and efficient encoding of syntactic and discourse relationships in long and complex texts (up to 6410 words in our experiments) to quantum circuits. The developed system is provided to the community as part of the augmented open-source quantum NLP package lambeq Gen II.
>
---
#### [new 239] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于多模态大语言模型的视觉空间推理任务，针对现有模型空间理解能力不足的问题，提出ViCA2模型（融合语义/空间双编码器）和ViCA-322K数据集，在基准测试中以7B参数超越大模型，实现高效空间认知。**

- **链接: [http://arxiv.org/pdf/2505.12363v1](http://arxiv.org/pdf/2505.12363v1)**

> **作者:** Qi Feng; Hidetoshi Shimodaira
>
> **备注:** 26 pages, 19 figures, 4 tables. Code, models, and dataset are available at our project page: https://github.com/nkkbr/ViCA
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [new 240] Efficient RL Training for Reasoning Models via Length-Aware Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大型推理模型训练效率低、响应路径长的问题，提出在强化学习中直接集成长度感知奖励设计，无需额外训练阶段即可缩短推理步骤。通过实验验证，在逻辑和数学任务中分别减少40%和33%的步骤，同时提升或保持性能。**

- **链接: [http://arxiv.org/pdf/2505.12284v1](http://arxiv.org/pdf/2505.12284v1)**

> **作者:** Danlong Yuan; Tian Xie; Shaohan Huang; Zhuocheng Gong; Huishuai Zhang; Chong Luo; Furu Wei; Dongyan Zhao
>
> **备注:** Under review
>
> **摘要:** Large reasoning models, such as OpenAI o1 or DeepSeek R1, have demonstrated remarkable performance on reasoning tasks but often incur a long reasoning path with significant memory and time costs. Existing methods primarily aim to shorten reasoning paths by introducing additional training data and stages. In this paper, we propose three critical reward designs integrated directly into the reinforcement learning process of large reasoning models, which reduce the response length without extra training stages. Experiments on four settings show that our method significantly decreases response length while maintaining or even improving performance. Specifically, in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance. For math problems, we reduce response length averaged by steps by 33% while preserving performance.
>
---
#### [new 241] TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出多级基准TIME，用于评估大语言模型在现实场景中的时间推理能力，解决现有研究忽视密集时间信息、动态事件和复杂依赖的问题。构建包含3.8万QA对的三个子数据集，涵盖11个子任务，并实验分析模型表现，发布数据集和简化版促进研究。**

- **链接: [http://arxiv.org/pdf/2505.12891v1](http://arxiv.org/pdf/2505.12891v1)**

> **作者:** Shaohang Wei; Wei Li; Feifan Song; Wen Luo; Tianyi Zhuang; Haochen Tan; Zhijiang Guo; Houfeng Wang
>
> **备注:** First version. There are still some examples to be added into the appendix
>
> **摘要:** Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , and the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME .
>
---
#### [new 242] LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决基于大语言模型（LLM）的混合检索中查询编码效率低的问题。通过保留完整LLM离线编码文档，但将实时查询编码简化为嵌入查找，LightRetriever在GPU加速下实现千倍推理加速，同时保持95%的检索性能。**

- **链接: [http://arxiv.org/pdf/2505.12260v1](http://arxiv.org/pdf/2505.12260v1)**

> **作者:** Guangyuan Ma; Yongliang Ma; Xuanrui Gou; Zhenpeng Su; Ming Zhou; Songlin Hu
>
> **摘要:** Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance.
>
---
#### [new 243] Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset
- **分类: cs.CR; cs.AI; cs.CL; F.2.2, I.2.7; F.2.2, I.2.7; F.2.2, I.2.7**

- **简介: 该论文属于LLM安全评估任务，旨在解决现有安全工具有效性验证不足的问题。研究构建恶意提示基准数据集，评估7种LLM防护工具（对比基线ChatGPT-3.5）。发现基线误报率高，Lakera Guard和ProtectAI LLM Guard在性能与可用性间表现最优，并提出提升透明度、检测能力等建议。**

- **链接: [http://arxiv.org/pdf/2505.13028v1](http://arxiv.org/pdf/2505.13028v1)**

> **作者:** Sayon Palit; Daniel Woods
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.
>
---
#### [new 244] MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文提出MMAR基准测试，用于评估音频-语言模型在跨语音、音乐及混合场景中的深度推理能力。针对现有基准领域单一、缺乏复杂推理任务的问题，构建了包含四层推理结构（信号、感知、语义、文化）的1000项数据集，标注思维链解释，并通过多模型测试揭示当前模型在深层理解和专业知识上的不足。**

- **链接: [http://arxiv.org/pdf/2505.13032v1](http://arxiv.org/pdf/2505.13032v1)**

> **作者:** Ziyang Ma; Yinghao Ma; Yanqiao Zhu; Chen Yang; Yi-Wen Chao; Ruiyang Xu; Wenxi Chen; Yuanzhe Chen; Zhuo Chen; Jian Cong; Kai Li; Keliang Li; Siyou Li; Xinfeng Li; Xiquan Li; Zheng Lian; Yuzhe Liang; Minghao Liu; Zhikang Niu; Tianrui Wang; Yuping Wang; Yuxuan Wang; Yihao Wu; Guanrou Yang; Jianwei Yu; Ruibin Yuan; Zhisheng Zheng; Ziya Zhou; Haina Zhu; Wei Xue; Emmanouil Benetos; Kai Yu; Eng-Siong Chng; Xie Chen
>
> **备注:** Open-source at https://github.com/ddlBoJack/MMAR
>
> **摘要:** We introduce MMAR, a new benchmark designed to evaluate the deep reasoning capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary tasks. MMAR comprises 1,000 meticulously curated audio-question-answer triplets, collected from real-world internet videos and refined through iterative error corrections and quality checks to ensure high quality. Unlike existing benchmarks that are limited to specific domains of sound, music, or speech, MMAR extends them to a broad spectrum of real-world audio scenarios, including mixed-modality combinations of sound, music, and speech. Each question in MMAR is hierarchically categorized across four reasoning layers: Signal, Perception, Semantic, and Cultural, with additional sub-categories within each layer to reflect task diversity and complexity. To further foster research in this area, we annotate every question with a Chain-of-Thought (CoT) rationale to promote future advancements in audio reasoning. Each item in the benchmark demands multi-step deep reasoning beyond surface-level understanding. Moreover, a part of the questions requires graduate-level perceptual and domain-specific knowledge, elevating the benchmark's difficulty and depth. We evaluate MMAR using a broad set of models, including Large Audio-Language Models (LALMs), Large Audio Reasoning Models (LARMs), Omni Language Models (OLMs), Large Language Models (LLMs), and Large Reasoning Models (LRMs), with audio caption inputs. The performance of these models on MMAR highlights the benchmark's challenging nature, and our analysis further reveals critical limitations of understanding and reasoning capabilities among current models. We hope MMAR will serve as a catalyst for future advances in this important but little-explored area.
>
---
#### [new 245] VenusX: Unlocking Fine-Grained Functional Understanding of Proteins
- **分类: cs.LG; cs.CL; q-bio.QM**

- **简介: 该论文提出VenusX基准测试，解决蛋白质细粒度功能注释和配对问题。通过构建包含残基、片段、结构域层级的87.8万样本数据集，支持跨分布评估，并评估多种模型性能，推动蛋白质功能机制理解与模型优化。**

- **链接: [http://arxiv.org/pdf/2505.11812v1](http://arxiv.org/pdf/2505.11812v1)**

> **作者:** Yang Tan; Wenrui Gou; Bozitao Zhong; Liang Hong; Huiqun Yu; Bingxin Zhou
>
> **备注:** 29 pages, 3 figures, 17 tables
>
> **摘要:** Deep learning models have driven significant progress in predicting protein function and interactions at the protein level. While these advancements have been invaluable for many biological applications such as enzyme engineering and function annotation, a more detailed perspective is essential for understanding protein functional mechanisms and evaluating the biological knowledge captured by models. To address this demand, we introduce VenusX, the first large-scale benchmark for fine-grained functional annotation and function-based protein pairing at the residue, fragment, and domain levels. VenusX comprises three major task categories across six types of annotations, including residue-level binary classification, fragment-level multi-class classification, and pairwise functional similarity scoring for identifying critical active sites, binding sites, conserved sites, motifs, domains, and epitopes. The benchmark features over 878,000 samples curated from major open-source databases such as InterPro, BioLiP, and SAbDab. By providing mixed-family and cross-family splits at three sequence identity thresholds, our benchmark enables a comprehensive assessment of model performance on both in-distribution and out-of-distribution scenarios. For baseline evaluation, we assess a diverse set of popular and open-source models, including pre-trained protein language models, sequence-structure hybrids, structure-based methods, and alignment-based techniques. Their performance is reported across all benchmark datasets and evaluation settings using multiple metrics, offering a thorough comparison and a strong foundation for future research. Code and data are publicly available at https://github.com/ai4protein/VenusX.
>
---
#### [new 246] EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对多模态大语言模型网页代理的环境提示注入攻击任务，解决现有攻击有效性低、隐蔽性差的问题。通过像素级扰动修改网页源码，诱导代理执行指定动作，利用神经网络近似不可微分映射并用梯度下降优化攻击，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11717v1](http://arxiv.org/pdf/2505.11717v1)**

> **作者:** Xilong Wang; John Bloch; Zedian Shao; Yuepeng Hu; Shuyan Zhou; Neil Zhenqiang Gong
>
> **摘要:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines.
>
---
#### [new 247] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出MedAgentBoard基准，用于评估多智能体协作、单LLM与传统方法在四类医疗任务（问答、摘要生成、预测建模、流程自动化）中的性能。研究发现多智能体仅在部分场景（如流程自动化）优于单LLM，但传统方法在医疗问答和预测任务中仍占优，强调需根据具体任务权衡AI方案的复杂性与收益。**

- **链接: [http://arxiv.org/pdf/2505.12371v1](http://arxiv.org/pdf/2505.12371v1)**

> **作者:** Yinghao Zhu; Ziyi He; Haoran Hu; Xiaochen Zheng; Xichen Zhang; Zixiang Wang; Junyi Gao; Liantao Ma; Lequan Yu
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at https://medagentboard.netlify.app/.
>
---
#### [new 248] Token-Level Uncertainty Estimation for Large Language Model Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型不确定性估计任务，旨在解决LLMs在复杂推理中输出可信度评估问题。提出基于低秩权重扰动的token级不确定性估计框架，通过预测分布量化语义不确定性，实验证明其与答案正确性高度相关，并利用粒子滤波提升推理性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11737v1](http://arxiv.org/pdf/2505.11737v1)**

> **作者:** Tunyu Zhang; Haizhou Shi; Yibin Wang; Hengyi Wang; Xiaoxiao He; Zhuowei Li; Haoxian Chen; Ligong Han; Kai Xu; Huan Zhang; Dimitris Metaxas; Hao Wang
>
> **备注:** Preprint; Work in progress
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a token-level uncertainty estimation framework to enable LLMs to self-assess and self-improve their generation quality in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation to LLM decoding, generating predictive distributions that we use to estimate token-level uncertainties. We then aggregate these uncertainties to reflect semantic uncertainty of the generated sequences. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that our token-level uncertainty metrics strongly correlate with answer correctness and model robustness. Additionally, we explore using uncertainty to directly enhance the model's reasoning performance through multiple generations and the particle filtering algorithm. Our approach consistently outperforms existing uncertainty estimation methods, establishing effective uncertainty estimation as a valuable tool for both evaluating and improving reasoning generation in LLMs.
>
---
#### [new 249] CoT-Kinetics: A Theoretical Modeling Assessing LRM Reasoning Process
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大型推理模型（LRM）输出质量评估任务，旨在解决现有方法无法准确衡量推理过程与答案间因果关系的问题。通过借鉴经典力学，提出CoT-Kinetics能量方程，将LRM内部推理过程建模为力学场中的粒子动力学，量化推理合理性以精准评估答案置信度。**

- **链接: [http://arxiv.org/pdf/2505.13408v1](http://arxiv.org/pdf/2505.13408v1)**

> **作者:** Jinhe Bi; Danqi Yan; Yifan Wang; Wenke Huang; Haokun Chen; Guancheng Wan; Mang Ye; Xun Xiao; Hinrich Schuetze; Volker Tresp; Yunpu Ma
>
> **摘要:** Recent Large Reasoning Models significantly improve the reasoning ability of Large Language Models by learning to reason, exhibiting the promising performance in solving complex tasks. LRMs solve tasks that require complex reasoning by explicitly generating reasoning trajectories together with answers. Nevertheless, judging the quality of such an output answer is not easy because only considering the correctness of the answer is not enough and the soundness of the reasoning trajectory part matters as well. Logically, if the soundness of the reasoning part is poor, even if the answer is correct, the confidence of the derived answer should be low. Existing methods did consider jointly assessing the overall output answer by taking into account the reasoning part, however, their capability is still not satisfactory as the causal relationship of the reasoning to the concluded answer cannot properly reflected. In this paper, inspired by classical mechanics, we present a novel approach towards establishing a CoT-Kinetics energy equation. Specifically, our CoT-Kinetics energy equation formulates the token state transformation process, which is regulated by LRM internal transformer layers, as like a particle kinetics dynamics governed in a mechanical field. Our CoT-Kinetics energy assigns a scalar score to evaluate specifically the soundness of the reasoning phase, telling how confident the derived answer could be given the evaluated reasoning. As such, the LRM's overall output quality can be accurately measured, rather than a coarse judgment (e.g., correct or incorrect) anymore.
>
---
#### [new 250] Zero-Shot Iterative Formalization and Planning in Partially Observable Environments
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究自动规划任务，解决部分可观察环境下传统方法因信息缺失无法生成有效规划的问题。提出PDDLego+框架，通过零样本迭代将环境动态转化为PDDL模型并优化规划，无需先验轨迹，在文本模拟环境中验证了其高效性、鲁棒性和知识迁移能力。**

- **链接: [http://arxiv.org/pdf/2505.13126v1](http://arxiv.org/pdf/2505.13126v1)**

> **作者:** Liancheng Gong; Wang Zhu; Jesse Thomason; Li Zhang
>
> **摘要:** In planning, using LLMs not to predict plans but to formalize an environment into the Planning Domain Definition Language (PDDL) has been shown to greatly improve performance and control. While most work focused on fully observable environments, we tackle the more realistic and challenging partially observable environments where existing methods are incapacitated by the lack of complete information. We propose PDDLego+, a framework to iteratively formalize, plan, grow, and refine PDDL representations in a zero-shot manner, without needing access to any existing trajectories. On two textual simulated environments, we show that PDDLego+ not only achieves superior performance, but also shows robustness against problem complexity. We also show that the domain knowledge captured after a successful trial is interpretable and benefits future tasks.
>
---
#### [new 251] Fractured Chain-of-Thought Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于LLM推理优化任务，旨在解决传统思维链（CoT）方法因长推理轨迹导致的高计算成本与延迟问题。提出"断裂采样"方法，通过动态调整推理路径数量、答案生成密度和截断深度三个维度，在减少token消耗的同时保持准确性，实现计算效率与推理性能的最优平衡。**

- **链接: [http://arxiv.org/pdf/2505.12992v1](http://arxiv.org/pdf/2505.12992v1)**

> **作者:** Baohao Liao; Hanze Dong; Yuhui Xu; Doyen Sahoo; Christof Monz; Junnan Li; Caiming Xiong
>
> **摘要:** Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning.
>
---
#### [new 252] Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文研究图形界面（GUI）的指令映射任务，解决现有基准过于简化、无法反映真实交互复杂性的问题。提出新基准OSWorld-G（564样本）和合成数据集Jedi（400万样本），通过多尺度模型训练提升界面元素识别、布局理解和操作能力，使基础模型在复杂任务成功率从5%提升至27%，并验证了组合泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.13227v1](http://arxiv.org/pdf/2505.13227v1)**

> **作者:** Tianbao Xie; Jiaqi Deng; Xiaochuan Li; Junlin Yang; Haoyuan Wu; Jixuan Chen; Wenjing Hu; Xinyuan Wang; Yuhui Xu; Zekun Wang; Yiheng Xu; Junli Wang; Doyen Sahoo; Tao Yu; Caiming Xiong
>
> **备注:** 49 pages, 13 figures
>
> **摘要:** Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at https://osworld-grounding.github.io.
>
---
#### [new 253] Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型（LLM）评估优化任务，旨在解决传统LLM-as-a-Judge方法因单点评估导致的信息丢失和可靠性不足问题。通过设计分布对齐训练框架（KL散度目标+交叉熵正则化），结合对抗训练增强鲁棒性，实现了LLM生成判断分布与人类标注分布的对齐，提升评估准确性和稳定性。**

- **链接: [http://arxiv.org/pdf/2505.12301v1](http://arxiv.org/pdf/2505.12301v1)**

> **作者:** Luyu Chen; Zeyu Zhang; Haoran Tan; Quanyu Dai; Hao Yang; Zhenhua Dong; Xu Chen
>
> **备注:** 19 pages, 3 tables, 3 figures
>
> **摘要:** LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness.
>
---
#### [new 254] IG Parser: A Software Package for the Encoding of Institutional Statements using the Institutional Grammar
- **分类: cs.MA; cs.AI; cs.CL; 68T30, 68T50; E.2; H.1.0; I.7.2; I.6.5; K.4.1**

- **简介: 该论文介绍IG Parser软件，属于制度分析工具开发任务，旨在解决自然语言制度声明（法律/社会规范）的结构化编码难题。基于制度语法2.0理论框架，开发了专用语法IG Script，实现自动化格式转换，支持多维度制度系统分析，并通过架构设计与案例验证工具效能。**

- **链接: [http://arxiv.org/pdf/2505.13393v1](http://arxiv.org/pdf/2505.13393v1)**

> **作者:** Christopher K. Frantz
>
> **备注:** 24 pages
>
> **摘要:** This article provides an overview of IG Parser, a software that facilitates qualitative content analysis of formal (e.g., legal) rules or informal (e.g., socio-normative) norms, and strategies (such as conventions) -- referred to as \emph{institutions} -- that govern social systems and operate configurally to describe \emph{institutional systems}. To this end, the IG Parser employs a distinctive syntax that ensures rigorous encoding of natural language, while automating the transformation into various formats that support the downstream analysis using diverse analytical techniques. The conceptual core of the IG Parser is an associated syntax, IG Script, that operationalizes the conceptual foundations of the Institutional Grammar, and more specifically Institutional Grammar 2.0, an analytical paradigm for institutional analysis. This article presents the IG Parser, including its conceptual foundations, syntactic specification of IG Script, alongside architectural principles. This introduction is augmented with selective illustrative examples that highlight the use and benefit associated with the tool.
>
---
## 更新

#### [replaced 001] Intention Knowledge Graph Construction for User Intention Relation Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.11500v2](http://arxiv.org/pdf/2412.11500v2)**

> **作者:** Jiaxin Bai; Zhaobo Wang; Junfei Cheng; Dan Yu; Zerui Huang; Weiqi Wang; Xin Liu; Chen Luo; Yanming Zhu; Bo Li; Yangqiu Song
>
> **摘要:** Understanding user intentions is challenging for online platforms. Recent work on intention knowledge graphs addresses this but often lacks focus on connecting intentions, which is crucial for modeling user behavior and predicting future actions. This paper introduces a framework to automatically generate an intention knowledge graph, capturing connections between user intentions. Using the Amazon m2 dataset, we construct an intention graph with 351 million edges, demonstrating high plausibility and acceptance. Our model effectively predicts new session intentions and enhances product recommendations, outperforming previous state-of-the-art methods and showcasing the approach's practical utility.
>
---
#### [replaced 002] Comparing Specialised Small and General Large Language Models on Text Classification: 100 Labelled Samples to Achieve Break-Even Performance
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.12819v3](http://arxiv.org/pdf/2402.12819v3)**

> **作者:** Branislav Pecher; Ivan Srba; Maria Bielikova
>
> **摘要:** When solving NLP tasks with limited labelled data, researchers typically either use a general large language model without further update, or use a small number of labelled samples to tune a specialised smaller model. In this work, we answer an important question -- how many labelled samples are required for the specialised small models to outperform general large models, while taking the performance variance into consideration. By observing the behaviour of fine-tuning, instruction-tuning, prompting and in-context learning on 8 language models, we identify such performance break-even points across 8 representative text classification tasks of varying characteristics. We show that the specialised models often need only few samples (on average $100$) to be on par or better than the general ones. At the same time, the number of required labels strongly depends on the dataset or task characteristics, with fine-tuning on binary datasets requiring significantly more samples. When performance variance is taken into consideration, the number of required labels increases on average by $100 - 200\%$. Finally, larger models do not consistently lead to better performance and lower variance, with 4-bit quantisation having negligible impact.
>
---
#### [replaced 003] Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20157v2](http://arxiv.org/pdf/2504.20157v2)**

> **作者:** Zae Myung Kim; Chanwoo Park; Vipul Raheja; Suin Kim; Dongyeop Kang
>
> **备注:** Code and data: https://github.com/minnesotanlp/mpo
>
> **摘要:** Reward-based alignment methods for large language models (LLMs) face two key limitations: vulnerability to reward hacking, where models exploit flaws in the reward signal; and reliance on brittle, labor-intensive prompt engineering when LLMs are used as reward models. We introduce Meta Policy Optimization (MPO), a framework that addresses these challenges by integrating a meta-reward model that dynamically refines the reward model's prompt throughout training. In MPO, the meta-reward model monitors the evolving training context and continuously adjusts the reward model's prompt to maintain high alignment, providing an adaptive reward signal that resists exploitation by the policy. This meta-learning approach promotes a more stable policy optimization, and greatly reduces the need for manual reward prompt design. It yields performance on par with or better than models guided by extensively hand-crafted reward prompts. Furthermore, we show that MPO maintains its effectiveness across diverse tasks, from essay writing to mathematical reasoning, without requiring specialized reward designs. Beyond standard RLAIF, MPO's meta-learning formulation is readily extensible to higher-level alignment frameworks. Overall, this method addresses theoretical and practical challenges in reward-based RL alignment for LLMs, paving the way for more robust and adaptable alignment strategies. The code and data can be accessed at: https://github.com/minnesotanlp/mpo
>
---
#### [replaced 004] ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21729v3](http://arxiv.org/pdf/2503.21729v3)**

> **作者:** Zhicheng Lee; Shulin Cao; Jinxin Liu; Jiajie Zhang; Weichuan Liu; Xiaoyin Che; Lei Hou; Juanzi Li
>
> **摘要:** Large Reasoning Models (LRMs) exhibit remarkable reasoning abilities but rely primarily on parametric knowledge, limiting factual accuracy. While recent works equip reinforcement learning (RL)-based LRMs with retrieval capabilities, they suffer from overthinking and lack robustness in reasoning, reducing their effectiveness in question answering (QA) tasks. To address this, we propose ReaRAG, a factuality-enhanced reasoning model that explores diverse queries without excessive iterations. Our solution includes a novel data construction framework with an upper bound on the reasoning chain length. Specifically, we first leverage an LRM to generate deliberate thinking, then select an action from a predefined action space (Search and Finish). For Search action, a query is executed against the RAG engine, where the result is returned as observation to guide reasoning steps later. This process iterates until a Finish action is chosen. Benefiting from ReaRAG's strong reasoning capabilities, our approach outperforms existing baselines on multi-hop QA. Further analysis highlights its strong reflective ability to recognize errors and refine its reasoning trajectory. Our study enhances LRMs' factuality while effectively integrating robust reasoning for Retrieval-Augmented Generation (RAG).
>
---
#### [replaced 005] What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.01893v2](http://arxiv.org/pdf/2409.01893v2)**

> **作者:** Zhi Chen; Qiguang Chen; Libo Qin; Qipeng Guo; Haijun Lv; Yicheng Zou; Wanxiang Che; Hang Yan; Kai Chen; Dahua Lin
>
> **备注:** ACL 2025 Camera Ready. Code is available at: https://github.com/WowCZ/LongMIT
>
> **摘要:** Recent advancements in large language models (LLMs) with extended context windows have significantly improved tasks such as information extraction, question answering, and complex planning scenarios. In order to achieve success in long context tasks, a large amount of work has been done to enhance the long context capabilities of the model through synthetic data. Existing methods typically utilize the Self-Instruct framework to generate instruction tuning data for better long context capability improvement. However, our preliminary experiments indicate that less than 35% of generated samples are multi-hop, and more than 40% exhibit poor quality, limiting comprehensive understanding and further research. To improve the quality of synthetic data, we propose the Multi-agent Interactive Multi-hop Generation (MIMG) framework, incorporating a Quality Verification Agent, a Single-hop Question Generation Agent, a Multiple Question Sampling Strategy, and a Multi-hop Question Merger Agent. This framework improves the data quality, with the proportion of high-quality, multi-hop, and diverse data exceeding 85%. Furthermore, we systematically investigate strategies for document selection, question merging, and validation techniques through extensive experiments across various models. Our findings show that our synthetic high-quality long-context instruction data significantly enhances model performance, even surpassing models trained on larger amounts of human-annotated data. Our code is available at: https://github.com/WowCZ/LongMIT.
>
---
#### [replaced 006] PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09543v2](http://arxiv.org/pdf/2503.09543v2)**

> **作者:** Oskar van der Wal; Pietro Lesci; Max Muller-Eberstein; Naomi Saphra; Hailey Schoelkopf; Willem Zuidema; Stella Biderman
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** The stability of language model pre-training and its effects on downstream performance are still understudied. Prior work shows that the training process can yield significantly different results in response to slight variations in initial conditions, e.g., the random seed. Crucially, the research community still lacks sufficient resources and tools to systematically investigate pre-training stability, particularly for decoder-only language models. We introduce the PolyPythias, a set of 45 new training runs for the Pythia model suite: 9 new seeds across 5 model sizes, from 14M to 410M parameters, resulting in about 7k new checkpoints that we release. Using these new 45 training runs, in addition to the 5 already available, we study the effects of different initial conditions determined by the seed -- i.e., parameters' initialisation and data order -- on (i) downstream performance, (ii) learned linguistic representations, and (iii) emergence of training phases. In addition to common scaling behaviours, our analyses generally reveal highly consistent training dynamics across both model sizes and initial conditions. Further, the new seeds for each model allow us to identify outlier training runs and delineate their characteristics. Our findings show the potential of using these methods to predict training stability.
>
---
#### [replaced 007] TRAIL: Trace Reasoning and Agentic Issue Localization
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08638v2](http://arxiv.org/pdf/2505.08638v2)**

> **作者:** Darshan Deshpande; Varun Gangal; Hersh Mehta; Jitin Krishnan; Anand Kannappan; Rebecca Qian
>
> **备注:** Dataset: https://huggingface.co/datasets/PatronusAI/TRAIL
>
> **摘要:** The increasing adoption of agentic workflows across diverse domains brings a critical need to scalably and systematically evaluate the complex traces these systems generate. Current evaluation methods depend on manual, domain-specific human analysis of lengthy workflow traces - an approach that does not scale with the growing complexity and volume of agentic outputs. Error analysis in these settings is further complicated by the interplay of external tool outputs and language model reasoning, making it more challenging than traditional software debugging. In this work, we (1) articulate the need for robust and dynamic evaluation methods for agentic workflow traces, (2) introduce a formal taxonomy of error types encountered in agentic systems, and (3) present a set of 148 large human-annotated traces (TRAIL) constructed using this taxonomy and grounded in established agentic benchmarks. To ensure ecological validity, we curate traces from both single and multi-agent systems, focusing on real-world applications such as software engineering and open-world information retrieval. Our evaluations reveal that modern long context LLMs perform poorly at trace debugging, with the best Gemini-2.5-pro model scoring a mere 11% on TRAIL. Our dataset and code are made publicly available to support and accelerate future research in scalable evaluation for agentic workflows.
>
---
#### [replaced 008] Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14285v2](http://arxiv.org/pdf/2502.14285v2)**

> **作者:** Yurong Wu; Fangwen Mu; Qiuhong Zhang; Jinjing Zhao; Xinrun Xu; Lingrui Mei; Yang Wu; Lin Shi; Junjie Wang; Zhiming Ding; Yiwei Wang
>
> **备注:** 14 pages,8 figures,4 tables
>
> **摘要:** Prompt trading has emerged as a significant intellectual property concern in recent years, where vendors entice users by showcasing sample images before selling prompt templates that can generate similar images. This work investigates a critical security vulnerability: attackers can steal prompt templates using only a limited number of sample images. To investigate this threat, we introduce Prism, a prompt-stealing benchmark consisting of 50 templates and 450 images, organized into Easy and Hard difficulty levels. To identify the vulnerabity of VLMs to prompt stealing, we propose EvoStealer, a novel template stealing method that operates without model fine-tuning by leveraging differential evolution algorithms. The system first initializes population sets using multimodal large language models (MLLMs) based on predefined patterns, then iteratively generates enhanced offspring through MLLMs. During evolution, EvoStealer identifies common features across offspring to derive generalized templates. Our comprehensive evaluation conducted across open-source (INTERNVL2-26B) and closed-source models (GPT-4o and GPT-4o-mini) demonstrates that EvoStealer's stolen templates can reproduce images highly similar to originals and effectively generalize to other subjects, significantly outperforming baseline methods with an average improvement of over 10%. Moreover, our cost analysis reveals that EvoStealer achieves template stealing with negligible computational expenses. Our code and dataset are available at https://github.com/whitepagewu/evostealer.
>
---
#### [replaced 009] Leveraging the true depth of LLMs
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02790v2](http://arxiv.org/pdf/2502.02790v2)**

> **作者:** Ramón Calvo González; Daniele Paliotta; Matteo Pagliardini; Martin Jaggi; François Fleuret
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable capabilities at the cost of high compute requirements. Recent studies have demonstrated that intermediate layers in LLMs can be removed or reordered without substantial accuracy loss; however, this insight has not yet been exploited to improve inference efficiency. Leveraging observed layer independence, we propose a novel method that groups consecutive layers into pairs evaluated in parallel, effectively restructuring the computational graph to enhance parallelism. Without requiring retraining or fine-tuning, this approach achieves an inference throughput improvement of 1.05x-1.20x on standard benchmarks, retaining 95\%-99\% of the original model accuracy. Empirical results demonstrate the practicality of this method in significantly reducing inference cost for large-scale LLM deployment. Additionally, we demonstrate that modest performance degradation can be substantially mitigated through lightweight fine-tuning, further enhancing the method's applicability.
>
---
#### [replaced 010] TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks
- **分类: cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.07890v3](http://arxiv.org/pdf/2505.07890v3)**

> **作者:** Kutay Ertürk; Furkan Altınışık; İrem Sarıaltın; Ömer Nezih Gerek
>
> **摘要:** This study presents TSLFormer, a light and robust word-level Turkish Sign Language (TSL) recognition model that treats sign gestures as ordered, string-like language. Instead of using raw RGB or depth videos, our method only works with 3D joint positions - articulation points - extracted using Google's Mediapipe library, which focuses on the hand and torso skeletal locations. This creates efficient input dimensionality reduction while preserving important semantic gesture information. Our approach revisits sign language recognition as sequence-to-sequence translation, inspired by the linguistic nature of sign languages and the success of transformers in natural language processing. Since TSLFormer uses the self-attention mechanism, it effectively captures temporal co-occurrence within gesture sequences and highlights meaningful motion patterns as words unfold. Evaluated on the AUTSL dataset with over 36,000 samples and 227 different words, TSLFormer achieves competitive performance with minimal computational cost. These results show that joint-based input is sufficient for enabling real-time, mobile, and assistive communication systems for hearing-impaired individuals.
>
---
#### [replaced 011] RM-R1: Reward Modeling as Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02387v3](http://arxiv.org/pdf/2505.02387v3)**

> **作者:** Xiusi Chen; Gaotang Li; Ziqi Wang; Bowen Jin; Cheng Qian; Yu Wang; Hongru Wang; Yu Zhang; Denghui Zhang; Tong Zhang; Hanghang Tong; Heng Ji
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Reward modeling is essential for aligning large language models with human preferences through reinforcement learning from human feedback. To provide accurate reward signals, a reward model (RM) should stimulate deep thinking and conduct interpretable reasoning before assigning a score or a judgment. Inspired by recent advances of long chain-of-thought on reasoning-intensive tasks, we hypothesize and validate that integrating reasoning capabilities into reward modeling significantly enhances RMs interpretability and performance. To this end, we introduce a new class of generative reward models - Reasoning Reward Models (ReasRMs) - which formulate reward modeling as a reasoning task. We propose a reasoning-oriented training pipeline and train a family of ReasRMs, RM-R1. RM-R1 features a chain-of-rubrics (CoR) mechanism - self-generating sample-level chat rubrics or math/code solutions, and evaluating candidate responses against them. The training of RM-R1 consists of two key stages: (1) distillation of high-quality reasoning chains and (2) reinforcement learning with verifiable rewards. Empirically, our models achieve state-of-the-art performance across three reward model benchmarks on average, outperforming much larger open-weight models (e.g., INF-ORM-Llama3.1-70B) and proprietary ones (e.g., GPT-4o) by up to 4.9%. Beyond final performance, we perform thorough empirical analyses to understand the key ingredients of successful ReasRM training. To facilitate future research, we release six REASRM models along with code and data at https://github.com/RM-R1-UIUC/RM-R1.
>
---
#### [replaced 012] Beyond Pairwise: Global Zero-shot Temporal Graph Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11114v2](http://arxiv.org/pdf/2502.11114v2)**

> **作者:** Alon Eirew; Kfir Bar; Ido Dagan
>
> **摘要:** Temporal relation extraction (TRE) is a fundamental task in natural language processing (NLP) that involves identifying the temporal relationships between events in a document. Despite the advances in large language models (LLMs), their application to TRE remains limited. Most existing approaches rely on pairwise classification, where event pairs are classified in isolation, leading to computational inefficiency and a lack of global consistency in the resulting temporal graph. In this work, we propose a novel zero-shot method for TRE that generates a document's complete temporal graph in a single step, followed by temporal constraint optimization to refine predictions and enforce temporal consistency across relations. Additionally, we introduce OmniTemp, a new dataset with complete annotations for all pairs of targeted events within a document. Through experiments and analyses, we demonstrate that our method outperforms existing zero-shot approaches and offers a competitive alternative to supervised TRE models.
>
---
#### [replaced 013] Vision-centric Token Compression in Large Language Model
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00791v3](http://arxiv.org/pdf/2502.00791v3)**

> **作者:** Ling Xing; Alex Jinpeng Wang; Rui Yan; Xiangbo Shu; Jinhui Tang
>
> **摘要:** Real-world applications are stretching context windows to hundreds of thousand of tokens while Large Language Models (LLMs) swell from billions to trillions of parameters. This dual expansion send compute and memory costs skyrocketing, making token compression indispensable. We introduce Vision Centric Token Compression (Vist), a slow-fast compression framework that mirrors human reading: the fast path renders distant tokens into images, letting a frozen, lightweight vision encoder skim the low-salience context; the slow path feeds the proximal window into the LLM for fine-grained reasoning. A Probability-Informed Visual Enhancement (PVE) objective masks high-frequency tokens during training, steering the Resampler to concentrate on semantically rich regions-just as skilled reader gloss over function words. On eleven in-context learning benchmarks, Vist achieves the same accuracy with 2.3 times fewer tokens, cutting FLOPs by 16% and memory by 50%. This method delivers remarkable results, outperforming the strongest text encoder-based compression method CEPE by 7.6% on average over benchmarks like TriviaQA, NQ, PopQA, NLUI, and CLIN, setting a new standard for token efficiency in LLMs. The source code will be released.
>
---
#### [replaced 014] Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.16330v2](http://arxiv.org/pdf/2406.16330v2)**

> **作者:** Deyuan Liu; Zhanyue Qin; Hairu Wang; Zhao Yang; Zecheng Wang; Fangying Rong; Qingbin Liu; Yanchao Hao; Xi Chen; Cunhang Fan; Zhao Lv; Zhiying Tu; Dianhui Chu; Bo Li; Dianbo Sui
>
> **摘要:** While large language models (LLMs) excel in many domains, their complexity and scale challenge deployment in resource-limited environments. Current compression techniques, such as parameter pruning, often fail to effectively utilize the knowledge from pruned parameters. To address these challenges, we propose Manifold-Based Knowledge Alignment and Layer Merging Compression (MKA), a novel approach that uses manifold learning and the Normalized Pairwise Information Bottleneck (NPIB) measure to merge similar layers, reducing model size while preserving essential performance. We evaluate MKA on multiple benchmark datasets and various LLMs. Our findings show that MKA not only preserves model performance but also achieves substantial compression ratios, outperforming traditional pruning methods. Moreover, when coupled with quantization, MKA delivers even greater compression. Specifically, on the MMLU dataset using the Llama3-8B model, MKA achieves a compression ratio of 43.75% with a minimal performance decrease of only 2.82\%. The proposed MKA method offers a resource-efficient and performance-preserving model compression technique for LLMs.
>
---
#### [replaced 015] Sparse Matrix in Large Language Model Fine-tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.15525v3](http://arxiv.org/pdf/2405.15525v3)**

> **作者:** Haoze He; Juncheng Billy Li; Xuan Jiang; Heather Miller
>
> **备注:** 14 pages
>
> **摘要:** LoRA and its variants have become popular parameter-efficient fine-tuning (PEFT) methods due to their ability to avoid excessive computational costs. However, an accuracy gap often exists between PEFT methods and full fine-tuning (FT), and this gap has yet to be systematically studied. In this work, we introduce a method for selecting sparse sub-matrices that aim to minimize the performance gap between PEFT vs. full fine-tuning (FT) while also reducing both fine-tuning computational cost and memory cost. Our Sparse Matrix Tuning (SMT) method begins by identifying the most significant sub-matrices in the gradient update, updating only these blocks during the fine-tuning process. In our experiments, we demonstrate that SMT consistently surpasses other PEFT baseline (e.g. LoRA and DoRA) in fine-tuning popular large language models such as LLaMA across a broad spectrum of tasks, while reducing the GPU memory footprint by 67% compared to FT. We also examine how the performance of LoRA and DoRA tends to plateau and decline as the number of trainable parameters increases, in contrast, our SMT method does not suffer from such issue.
>
---
#### [replaced 016] Superposition Yields Robust Neural Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10465v2](http://arxiv.org/pdf/2505.10465v2)**

> **作者:** Yizhou Liu; Ziming Liu; Jeff Gore
>
> **备注:** 30 pages, 23 figures, with corrections
>
> **摘要:** The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law -- the finding that loss decreases as a power law with model size -- remains unclear. Starting from two empirical principles -- that LLMs represent more things than the model dimensions (widths) they have (i.e., representations are superposed), and that words or concepts in language occur with varying frequencies -- we constructed a toy model to study the loss scaling with model size. We found that when superposition is weak, meaning only the most frequent features are represented without interference, the scaling of loss with model size depends on the underlying feature frequency; if feature frequencies follow a power law, so does the loss. In contrast, under strong superposition, where all features are represented but overlap with each other, the loss becomes inversely proportional to the model dimension across a wide range of feature frequency distributions. This robust scaling behavior is explained geometrically: when many more vectors are packed into a lower dimensional space, the interference (squared overlaps) between vectors scales inversely with that dimension. We then analyzed four families of open-sourced LLMs and found that they exhibit strong superposition and quantitatively match the predictions of our toy model. The Chinchilla scaling law turned out to also agree with our results. We conclude that representation superposition is an important mechanism underlying the observed neural scaling laws. We anticipate that these insights will inspire new training strategies and model architectures to achieve better performance with less computation and fewer parameters.
>
---
#### [replaced 017] $S^3$ -- Semantic Signal Separation
- **分类: cs.LG; cs.CL; stat.ML; I.2.7**

- **链接: [http://arxiv.org/pdf/2406.09556v3](http://arxiv.org/pdf/2406.09556v3)**

> **作者:** Márton Kardos; Jan Kostkan; Arnault-Quentin Vermillet; Kristoffer Nielbo; Kenneth Enevoldsen; Roberta Rocca
>
> **备注:** 24 pages, 13 figures (main manuscript has 9 pages and 7 figures); The paper has been adjusted according to reviewers' feedback
>
> **摘要:** Topic models are useful tools for discovering latent semantic structures in large textual corpora. Recent efforts have been oriented at incorporating contextual representations in topic modeling and have been shown to outperform classical topic models. These approaches are typically slow, volatile, and require heavy preprocessing for optimal results. We present Semantic Signal Separation ($S^3$), a theory-driven topic modeling approach in neural embedding spaces. $S^3$ conceptualizes topics as independent axes of semantic space and uncovers these by decomposing contextualized document embeddings using Independent Component Analysis. Our approach provides diverse and highly coherent topics, requires no preprocessing, and is demonstrated to be the fastest contextual topic model, being, on average, 4.5x faster than the runner-up BERTopic. We offer an implementation of $S^3$, and all contextual baselines, in the Turftopic Python package.
>
---
#### [replaced 018] Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.10378v5](http://arxiv.org/pdf/2310.10378v5)**

> **作者:** Jirui Qi; Raquel Fernández; Arianna Bisazza
>
> **备注:** EMNLP2023 Outstanding Paper (Multilinguality and Linguistic Diversity Track). All code and data are released at https://github.com/Betswish/Cross-Lingual-Consistency
>
> **摘要:** Multilingual large-scale Pretrained Language Models (PLMs) have been shown to store considerable amounts of factual knowledge, but large variations are observed across languages. With the ultimate goal of ensuring that users with different language backgrounds obtain consistent feedback from the same model, we study the cross-lingual consistency (CLC) of factual knowledge in various multilingual PLMs. To this end, we propose a Ranking-based Consistency (RankC) metric to evaluate knowledge consistency across languages independently from accuracy. Using this metric, we conduct an in-depth analysis of the determining factors for CLC, both at model level and at language-pair level. Among other results, we find that increasing model size leads to higher factual probing accuracy in most languages, but does not improve cross-lingual consistency. Finally, we conduct a case study on CLC when new factual associations are inserted in the PLMs via model editing. Results on a small sample of facts inserted in English reveal a clear pattern whereby the new piece of knowledge transfers only to languages with which English has a high RankC score.
>
---
#### [replaced 019] SSR: Alignment-Aware Modality Connector for Speech Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.00168v2](http://arxiv.org/pdf/2410.00168v2)**

> **作者:** Weiting Tan; Hirofumi Inaguma; Ning Dong; Paden Tomasello; Xutai Ma
>
> **备注:** IWSLT 2025
>
> **摘要:** Fusing speech into pre-trained language model (SpeechLM) usually suffers from inefficient encoding of long-form speech and catastrophic forgetting of pre-trained text modality. We propose SSR-Connector (Segmented Speech Representation Connector) for better modality fusion. Leveraging speech-text alignments, our approach segments and compresses speech features to match the granularity of text embeddings. Additionally, we introduce a two-stage training pipeline that includes the distillation and fine-tuning phases to mitigate catastrophic forgetting. SSR-Connector outperforms existing mechanism for speech-text modality fusion, consistently achieving better speech understanding (e.g., +10 accuracy on StoryCloze and +20 on Speech-MMLU) while preserving pre-trained text ability.
>
---
#### [replaced 020] Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.04671v2](http://arxiv.org/pdf/2505.04671v2)**

> **作者:** Yuxin Zhang; Meihao Fan; Ju Fan; Mingyang Yi; Yuyu Luo; Jian Tan; Guoliang Li
>
> **摘要:** Recent advances in large language models (LLMs) have significantly improved performance on the Text-to-SQL task by leveraging their powerful reasoning capabilities. To enhance accuracy during the reasoning process, external Process Reward Models (PRMs) can be introduced during training and inference to provide fine-grained supervision. However, if misused, PRMs may distort the reasoning trajectory and lead to suboptimal or incorrect SQL generation. To address this challenge, we propose Reward-SQL, a framework that systematically explores how to incorporate PRMs into the Text-to-SQL reasoning process effectively. Our approach follows a "cold start, then PRM supervision" paradigm. Specifically, we first train the model to decompose SQL queries into structured stepwise reasoning chains using common table expressions (Chain-of-CTEs), establishing a strong and interpretable reasoning baseline. Then, we investigate four strategies for integrating PRMs, and find that combining PRM as an online training signal (e.g.,GRPO) with PRM-guided inference (e.g., best-of-N sampling) yields the best results. Empirically, on the BIRD benchmark, Reward-SQL enables models supervised by PRM (7B) to achieve a 13.1% performance gain across various guidance strategies. Notably, our GRPO-aligned policy model based on Qwen2.5-Coder-7B-Instruct achieves 68.9% accuracy on the BIRD development set, outperforming all baseline methods under the same model size. These results demonstrate the effectiveness of Reward-SQL in leveraging reward-based supervision for Text-to-SQL reasoning.
>
---
#### [replaced 021] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02506v3](http://arxiv.org/pdf/2501.02506v3)**

> **作者:** Junjie Ye; Zhengyin Du; Xuesong Yao; Weijian Lin; Yufei Xu; Zehui Chen; Zaiyuan Wang; Sining Zhu; Zhiheng Xi; Siyu Yuan; Tao Gui; Qi Zhang; Xuanjing Huang; Jiecao Chen
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Effective evaluation of multi-hop tool use is critical for analyzing the understanding, reasoning, and function-calling capabilities of large language models (LLMs). However, progress has been hindered by a lack of reliable evaluation datasets. To address this, we present ToolHop, a dataset comprising 995 user queries and 3,912 associated tools, specifically designed for rigorous evaluation of multi-hop tool use. ToolHop ensures diverse queries, meaningful interdependencies, locally executable tools, detailed feedback, and verifiable answers through a novel query-driven data construction approach that includes tool creation, document refinement, and code generation. We evaluate 14 LLMs across five model families (i.e., LLaMA3.1, Qwen2.5, Gemini1.5, Claude3.5, and GPT), uncovering significant challenges in handling multi-hop tool-use scenarios. The leading model, GPT-4o, achieves an accuracy of 49.04%, underscoring substantial room for improvement. Further analysis reveals variations in tool-use strategies for various families, offering actionable insights to guide the development of more effective approaches. Code and data can be found in https://huggingface.co/datasets/bytedance-research/ToolHop.
>
---
#### [replaced 022] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.11062v3](http://arxiv.org/pdf/2407.11062v3)**

> **作者:** Mengzhao Chen; Wenqi Shao; Peng Xu; Jiahao Wang; Peng Gao; Kaipeng Zhang; Ping Luo
>
> **备注:** ACL 2025 Main, camera ready version
>
> **摘要:** Large language models (LLMs) are crucial in modern natural language processing and artificial intelligence. However, they face challenges in managing their significant memory requirements. Although quantization-aware training (QAT) offers a solution by reducing memory consumption through low-bit representations with minimal accuracy loss, it is impractical due to substantial training resources. To address this, we propose Efficient Quantization-Aware Training (EfficientQAT), a more feasible QAT algorithm. EfficientQAT involves two consecutive phases: Block-wise training of all parameters (Block-AP) and end-to-end training of quantization parameters (E2E-QP). To the best of our knowledge, Block-AP is the first method to enable direct training of all parameters in a block-wise manner, reducing accuracy loss in low-bit scenarios by enhancing the solution space during optimization. E2E-QP then trains only the quantization parameters (step sizes) end-to-end, further improving the performance of quantized models by considering interactions among all sub-modules. Extensive experiments demonstrate that EfficientQAT outperforms previous quantization methods across a range of models, including base LLMs, instruction-tuned LLMs, and multimodal LLMs, with scales from 7B to 70B parameters at various quantization bits. For instance, EfficientQAT obtains a 2-bit Llama-2-70B model on a single A100-80GB GPU in 41 hours, with less than 3 points accuracy degradation compared to the full precision (69.48 vs. 72.41). Code is available at https://github.com/OpenGVLab/EfficientQAT.
>
---
#### [replaced 023] Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.06207v3](http://arxiv.org/pdf/2502.06207v3)**

> **作者:** Junyu Lu; Kai Ma; Kaichun Wang; Kelaiti Xiao; Roy Ka-Wei Lee; Bo Xu; Liang Yang; Hongfei Lin
>
> **备注:** 18 pages, accepted at the ACL 2025
>
> **摘要:** Large Language Models (LLMs) have become essential for offensive language detection, yet their ability to handle annotation disagreement remains underexplored. Disagreement samples, which arise from subjective interpretations, pose a unique challenge due to their ambiguous nature. Understanding how LLMs process these cases, particularly their confidence levels, can offer insight into their alignment with human annotators. This study systematically evaluates the performance of multiple LLMs in detecting offensive language at varying levels of annotation agreement. We analyze binary classification accuracy, examine the relationship between model confidence and human disagreement, and explore how disagreement samples influence model decision-making during few-shot learning and instruction fine-tuning. Our findings reveal that LLMs struggle with low-agreement samples, often exhibiting overconfidence in these ambiguous cases. However, utilizing disagreement samples in training improves both detection accuracy and model alignment with human judgment. These insights provide a foundation for enhancing LLM-based offensive language detection in real-world moderation tasks.
>
---
#### [replaced 024] Can Vision-Language Models Infer Speaker's Ignorance? The Role of Visual and Linguistic Cues
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09120v3](http://arxiv.org/pdf/2502.09120v3)**

> **作者:** Ye-eun Cho; Yunho Maeng
>
> **备注:** 11 pages, 4 figures, 7 tables
>
> **摘要:** This study investigates whether vision-language models (VLMs) can perform pragmatic inference, focusing on ignorance implicatures, utterances that imply the speaker's lack of precise knowledge. To test this, we systematically manipulated contextual cues: the visually depicted situation (visual cue) and QUD-based linguistic prompts (linguistic cue). When only visual cues were provided, three state-of-the-art VLMs (GPT-4o, Gemini 1.5 Pro, and Claude 3.5 sonnet) produced interpretations largely based on the lexical meaning of the modified numerals. When linguistic cues were added to enhance contextual informativeness, Claude exhibited more human-like inference by integrating both types of contextual cues. In contrast, GPT and Gemini favored precise, literal interpretations. Although the influence of contextual cues increased, they treated each contextual cue independently and aligned them with semantic features rather than engaging in context-driven reasoning. These findings suggest that although the models differ in how they handle contextual cues, Claude's ability to combine multiple cues may signal emerging pragmatic competence in multimodal models.
>
---
#### [replaced 025] Inference and Verbalization Functions During In-Context Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.09349v2](http://arxiv.org/pdf/2410.09349v2)**

> **作者:** Junyi Tao; Xiaoyin Chen; Nelson F. Liu
>
> **备注:** EMNLP 2024 Findings
>
> **摘要:** Large language models (LMs) are capable of in-context learning from a few demonstrations (example-label pairs) to solve new tasks during inference. Despite the intuitive importance of high-quality demonstrations, previous work has observed that, in some settings, ICL performance is minimally affected by irrelevant labels (Min et al., 2022). We hypothesize that LMs perform ICL with irrelevant labels via two sequential processes: an inference function that solves the task, followed by a verbalization function that maps the inferred answer to the label space. Importantly, we hypothesize that the inference function is invariant to remappings of the label space (e.g., "true"/"false" to "cat"/"dog"), enabling LMs to share the same inference function across settings with different label words. We empirically validate this hypothesis with controlled layer-wise interchange intervention experiments. Our findings confirm the hypotheses on multiple datasets and tasks (natural language inference, sentiment analysis, and topic classification) and further suggest that the two functions can be localized in specific layers across various open-sourced models, including GEMMA-7B, MISTRAL-7B-V0.3, GEMMA-2-27B, and LLAMA-3.1-70B.
>
---
#### [replaced 026] CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.09436v2](http://arxiv.org/pdf/2505.09436v2)**

> **作者:** Raghav Garg; Kapil Sharma; Karan Gupta
>
> **摘要:** Large Language Models (LLMs) hold immense potential for revolutionizing Customer Experience Management (CXM), particularly in contact center operations. However, evaluating their practical utility in complex operational environments is hindered by data scarcity (due to privacy concerns) and the limitations of current benchmarks. Existing benchmarks often lack realism, failing to incorporate deep knowledge base (KB) integration, real-world noise, or critical operational tasks beyond conversational fluency. To bridge this gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset specifically designed for evaluating AI in operational CXM contexts. Given the diversity in possible contact center features, we have developed a scalable LLM-powered pipeline that simulates the brand's CXM entities that form the foundation of our datasets-such as knowledge articles including product specifications, issue taxonomies, and contact center conversations. The entities closely represent real-world distribution because of controlled noise injection (informed by domain experts) and rigorous automated validation. Building on this, we release CXMArena, which provides dedicated benchmarks targeting five important operational tasks: Knowledge Base Refinement, Intent Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with Integrated Tools. Our baseline experiments underscore the benchmark's difficulty: even state of the art embedding and generation models achieve only 68% accuracy on article search, while standard embedding methods yield a low F1 score of 0.3 for knowledge base refinement, highlighting significant challenges for current models necessitating complex pipelines and solutions over conventional techniques.
>
---
#### [replaced 027] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.05723v2](http://arxiv.org/pdf/2412.05723v2)**

> **作者:** Haizhou Shi; Yibin Wang; Ligong Han; Huan Zhang; Hao Wang
>
> **备注:** Pre-print; Accepted (non-archivally) at ICLR'25 Workshop: "Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI"
>
> **摘要:** Estimating the uncertainty of responses from Large Language Models (LLMs) remains a critical challenge. While recent Bayesian methods have demonstrated effectiveness in quantifying uncertainty through low-rank weight updates, they typically require complex fine-tuning or post-training procedures. In this paper, we propose Training-Free Bayesianization (TFB), a simple yet theoretically grounded framework that efficiently transforms trained low-rank adapters into Bayesian ones without additional training. TFB systematically searches for the maximally acceptable level of variance in the weight posterior, constrained within a family of low-rank isotropic Gaussian distributions. Our theoretical analysis shows that under mild conditions, this search process is equivalent to KL-regularized variational optimization, a generalized form of variational inference. Through comprehensive experiments, we show that TFB achieves superior uncertainty estimation and generalization compared to existing methods while eliminating the need for complex Bayesianization training procedures. Code will be available at https://github.com/Wang-ML-Lab/bayesian-peft.
>
---
#### [replaced 028] Bias Similarity Across Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12010v3](http://arxiv.org/pdf/2410.12010v3)**

> **作者:** Hyejun Jeong; Shiqing Ma; Amir Houmansadr
>
> **备注:** under review
>
> **摘要:** Bias in Large Language Models remains a critical concern as these systems are increasingly deployed in high-stakes applications. Yet most fairness evaluations rely on scalar metrics or single-model analysis, overlooking how biases align -- or diverge -- across model families, scales, and tuning strategies. In this work, we reframe bias similarity as a form of functional similarity and evaluate 24 LLMs from four major families on over one million structured prompts spanning four bias dimensions. Our findings uncover that fairness is not strongly determined by model size, architecture, instruction tuning, or openness. Instead, bias behaviors are highly context-dependent and structurally persistent, often resistant to current alignment techniques. Contrary to common assumptions, we find that open-source models frequently match or outperform proprietary models in both fairness and utility. These results call into question the default reliance on proprietary systems and highlight the need for behaviorally grounded, model-specific audits to better understand how bias manifests and endures across the LLM landscape.
>
---
#### [replaced 029] Disentangling Length Bias In Preference Learning Via Response-Conditioned Modeling
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00814v2](http://arxiv.org/pdf/2502.00814v2)**

> **作者:** Jianfeng Cai; Jinhua Zhu; Ruopei Sun; Yue Wang; Li Li; Wengang Zhou; Houqiang Li
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has achieved considerable success in aligning large language models (LLMs) by modeling human preferences with a learnable reward model and employing a reinforcement learning algorithm to maximize the reward model's scores. However, these reward models are susceptible to exploitation through various superficial confounding factors, with length bias emerging as a particularly significant concern. Moreover, while the pronounced impact of length bias on preference modeling suggests that LLMs possess an inherent sensitivity to length perception, our preliminary investigations reveal that fine-tuned LLMs consistently struggle to adhere to explicit length instructions. To address these two limitations, we propose a novel framework wherein the reward model explicitly differentiates between human semantic preferences and response length requirements. Specifically, we introduce a $\textbf{R}$esponse-$\textbf{c}$onditioned $\textbf{B}$radley-$\textbf{T}$erry (Rc-BT) model that enhances the model's capability in length bias mitigating and length instruction following, through training on our augmented dataset. Furthermore, we propose the Rc-RM and Rc-DPO algorithm to leverage the Rc-BT model for reward modeling and direct policy optimization (DPO) of LLMs, simultaneously mitigating length bias and promoting adherence to length instructions. Extensive experiments across various foundational models and datasets demonstrate the effectiveness and generalizability of our approach.
>
---
#### [replaced 030] ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.10785v2](http://arxiv.org/pdf/2406.10785v2)**

> **作者:** Yurun Song; Junchen Zhao; Ian G. Harris; Sangeetha Abdu Jyothi
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** In this paper, we introduce \textbf{Share}d \textbf{Lo}w \textbf{R}ank \textbf{A}daptation (ShareLoRA), a Large Language Model (LLM) fine-tuning technique that balances parameter efficiency, adaptability, and robustness without compromising performance. By strategically sharing the low-rank weight matrices across different layers, ShareLoRA achieves 44\% to 96\% reduction in trainable parameters compared to standard LoRA, alongside a substantial decrease in memory overhead. This efficiency gain scales with model size, making ShareLoRA particularly advantageous for resource-constrained environments. Importantly, ShareLoRA not only maintains model performance but also exhibits robustness in both classification and generation tasks across diverse models, including RoBERTa, GPT-2, and LLaMA series (1, 2, and 3). It consistently outperforms LoRA in zero-shot, few-shot, and continual fine-tuning scenarios, achieving up to 1.2\% average accuracy improvement, and enhanced generalization across domains. In continual learning settings, ShareLoRA achieves 1.2\% higher accuracy on GSM8K, 0.6\% on HumanEval, and 0.5\% on both MMLU and MMLU-Pro. Our results demonstrate that ShareLoRA supports high-quality fine-tuning while offering strong generalization and continual adaptation across various model scales and diverse tasks.
>
---
#### [replaced 031] Brittle Minds, Fixable Activations: Understanding Belief Representations in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.17513v3](http://arxiv.org/pdf/2406.17513v3)**

> **作者:** Matteo Bortoletto; Constantin Ruhdorfer; Lei Shi; Andreas Bulling
>
> **备注:** ICML 2024 Workshop on Mechanistic Interpretability version: https://openreview.net/forum?id=yEwEVoH9Be
>
> **摘要:** Despite growing interest in Theory of Mind (ToM) tasks for evaluating language models (LMs), little is known about how LMs internally represent mental states of self and others. Understanding these internal mechanisms is critical - not only to move beyond surface-level performance, but also for model alignment and safety, where subtle misattributions of mental states may go undetected in generated outputs. In this work, we present the first systematic investigation of belief representations in LMs by probing models across different scales, training regimens, and prompts - using control tasks to rule out confounds. Our experiments provide evidence that both model size and fine-tuning substantially improve LMs' internal representations of others' beliefs, which are structured - not mere by-products of spurious correlations - yet brittle to prompt variations. Crucially, we show that these representations can be strengthened: targeted edits to model activations can correct wrong ToM inferences.
>
---
#### [replaced 032] Evolving LLMs' Self-Refinement Capability via Iterative Preference Optimization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05605v3](http://arxiv.org/pdf/2502.05605v3)**

> **作者:** Yongcheng Zeng; Xinyu Cui; Xuanfa Jin; Guoqing Liu; Zexu Sun; Dong Li; Ning Yang; Jianye Hao; Haifeng Zhang; Jun Wang
>
> **摘要:** While large language models (LLMs) have demonstrated remarkable general performance, enabling smaller models to achieve capabilities comparable to their larger counterparts remains a critical challenge. For humans, iterative refinement of problem analysis and responses is a common strategy to enhance answer quality. However, we observe that existing LLMs exhibit limited ability to refine their outputs for quality improvement. In this paper, we first investigate mechanisms to unlock and progressively enhance self-refinement ability in smaller models within an iterative preference optimization framework, aiming to bridge the performance gap with larger models. To this end, we propose EVOLVE, a novel post-training and inference framework that iteratively integrates preference training with self-refinement-driven data collection. During training, EVOLVE strengthens the model's direct question-answering ability while simultaneously unlocking its self-refinement potential. At inference, the framework leverages this capability to generate progressively refined responses, which are filtered to construct datasets for subsequent rounds of preference training. Experiments demonstrate EVOLVE's exceptional performance: when applied to Llama-3.1-8B base model and under the self-refinement setting, it surpasses state-of-the-art models including Llama-3.1-405B-Instruct and GPT-4o, achieving a 62.3% length-controlled win rate and 63.3% raw win rate on AlpacaEval 2, along with a 50.3% win rate on Arena-Hard. Furthermore, EVOLVE consistently enhances performance on mathematical reasoning tasks like GSM8K and MATH.
>
---
#### [replaced 033] Computational Reasoning of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20771v2](http://arxiv.org/pdf/2504.20771v2)**

> **作者:** Haitao Wu; Zongbo Han; Joey Tianyi Zhou; Huaxi Huang; Changqing Zhang
>
> **摘要:** With the rapid development and widespread application of Large Language Models (LLMs), multidimensional evaluation has become increasingly critical. However, current evaluations are often domain-specific and overly complex, limiting their effectiveness as cross-domain proxies for core capabilities. To address these limitations and enable a unified and simple evaluation framework, an ideal proxy task should target a basic capability that generalizes across tasks and is independent of domain-specific knowledge. Turing machine provides a powerful theoretical lens by reducing complex processes to basic, domain-agnostic computational operations. This perspective offers a principled framework for evaluating basic computational abilities essential to a wide range of tasks. Motivated by this abstraction, we introduce \textbf{Turing Machine Bench}, a benchmark designed to assess the ability of LLMs to \textbf{strictly follow rules} and \textbf{accurately manage internal states} for multi-step, referred to as \textbf{computational reasoning}. TMBench incorporates four key features: self-contained and knowledge-agnostic reasoning, a minimalistic multi-step structure, controllable difficulty, and a solid theoretical foundation based on Turing machine. Empirical results demonstrate that TMBench serves as an effective proxy for evaluating computational reasoning on representative LLMs. It produces clear step-wise accuracy curves, revealing LLMs' ability to execute multi-step reasoning processes. By analyzing performance trends across TMBench and established reasoning benchmarks, we find strong correlations with real-world tasks, bridging real-task evaluation with basic ability assessment. These findings suggest that TMBench holds potential as a cross-domain dimension for evaluating reasoning in LLMs. Code and data are available at \href{https://github.com/HaitaoWuTJU/Turing-Machine-Bench}{Repo}.
>
---
#### [replaced 034] VCM: Vision Concept Modeling Based on Implicit Contrastive Learning with Vision-Language Instruction Fine-Tuning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19627v2](http://arxiv.org/pdf/2504.19627v2)**

> **作者:** Run Luo; Renke Shan; Longze Chen; Ziqiang Liu; Lu Wang; Min Yang; Xiaobo Xia
>
> **备注:** VCM
>
> **摘要:** Large Vision-Language Models (LVLMs) are pivotal for real-world AI tasks like embodied intelligence due to their strong vision-language reasoning abilities. However, current LVLMs process entire images at the token level, which is inefficient compared to humans who analyze information and generate content at the conceptual level, extracting relevant visual concepts with minimal effort. This inefficiency, stemming from the lack of a visual concept model, limits LVLMs' usability in real-world applications. To address this, we propose VCM, an end-to-end self-supervised visual concept modeling framework. VCM leverages implicit contrastive learning across multiple sampled instances and vision-language fine-tuning to construct a visual concept model without requiring costly concept-level annotations. Our results show that VCM significantly reduces computational costs (e.g., 85\% fewer FLOPs for LLaVA-1.5-7B) while maintaining strong performance across diverse image understanding tasks. Moreover, VCM enhances visual encoders' capabilities in classic visual concept perception tasks. Extensive quantitative and qualitative experiments validate the effectiveness and efficiency of VCM.
>
---
#### [replaced 035] PACE: Abstractions for Communicating Efficiently
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.20120v3](http://arxiv.org/pdf/2409.20120v3)**

> **作者:** Jonathan D. Thomas; Andrea Silvi; Devdatt Dubhashi; Moa Johansson
>
> **备注:** Accepted to CogSci 2025 for presentation
>
> **摘要:** A central but unresolved aspect of problem-solving in AI is the capability to introduce and use abstractions, something humans excel at. Work in cognitive science has demonstrated that humans tend towards higher levels of abstraction when engaged in collaborative task-oriented communication, enabling gradually shorter and more information-efficient utterances. Several computational methods have attempted to replicate this phenomenon, but all make unrealistic simplifying assumptions about how abstractions are introduced and learned. Our method, Procedural Abstractions for Communicating Efficiently (PACE), overcomes these limitations through a neuro-symbolic approach. On the symbolic side, we draw on work from library learning for proposing abstractions. We combine this with neural methods for communication and reinforcement learning, via a novel use of bandit algorithms for controlling the exploration and exploitation trade-off in introducing new abstractions. PACE exhibits similar tendencies to humans on a collaborative construction task from the cognitive science literature, where one agent (the architect) instructs the other (the builder) to reconstruct a scene of block-buildings. PACE results in the emergence of an efficient language as a by-product of collaborative communication. Beyond providing mechanistic insights into human communication, our work serves as a first step to providing conversational agents with the ability for human-like communicative abstractions.
>
---
#### [replaced 036] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05567v2](http://arxiv.org/pdf/2502.05567v2)**

> **作者:** Xiaoyang Liu; Kangjie Bao; Jiashuo Zhang; Yunqi Liu; Yuntian Liu; Yu Chen; Yang Jiao; Tao Luo
>
> **摘要:** Autoformalization, the automatic translation of mathematical content from natural language into machine-verifiable formal languages, has seen significant progress driven by advances in large language models (LLMs). Nonetheless, a primary barrier to further improvements is the limited availability of parallel corpora that map informal mathematical text to its formal counterpart. To address this limitation, we propose ATLAS (Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data), a novel data generation framework designed to produce large-scale, high-quality parallel corpora of theorem statements. Distinct from prior approaches, ATLAS begins with a concept repository, accelerates the improvement of student model through expert iteration combined with knowledge distillation, and introduces two novel augmentation strategies that exploit the structural characteristics of formal languages. With the proposed ATLAS running for 10 iterations, we construct an undergraduate-level dataset comprising 117k theorem statements and develop ATLAS Translator, which demonstrates statistically significant improvements over both the HERALD Translator and the Kimina-Autoformalizer across all benchmarks ($p<0.05$, two-sided t-test), achieving a new state of the art. The datasets, model, and code will be released to the public soon.
>
---
#### [replaced 037] Benchmarking LLMs' Swarm intelligence
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04364v2](http://arxiv.org/pdf/2505.04364v2)**

> **作者:** Kai Ruan; Mowen Huang; Ji-Rong Wen; Hao Sun
>
> **摘要:** Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
>
---
#### [replaced 038] Probabilistic Reasoning with LLMs for k-anonymity Estimation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09674v2](http://arxiv.org/pdf/2503.09674v2)**

> **作者:** Jonathan Zheng; Sauvik Das; Alan Ritter; Wei Xu
>
> **备注:** 9 pages, preprint
>
> **摘要:** Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
>
---
#### [replaced 039] Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.03261v2](http://arxiv.org/pdf/2503.03261v2)**

> **作者:** Yichong Zhao; Susumu Goto
>
> **摘要:** Multiple previous studies have reported suboptimal performance of LLMs in biomedical text mining. By analyzing failure patterns in these evaluations, we identified three primary challenges for LLMs in biomedical corpora: (1) LLMs fail to learn implicit dataset-specific nuances from supervised data, (2) The common formatting requirements of discriminative tasks limit the reasoning capabilities of LLMs particularly for LLMs that lack test-time compute, and (3) LLMs struggle to adhere to annotation guidelines and match exact schemas, which hinders their ability to understand detailed annotation requirements which is essential in biomedical annotation workflow. We experimented with prompt engineering techniques targeted to the above issues, and developed a pipeline that dynamically extracts instructions from annotation guidelines. Our results show that frontier LLMs can approach or surpass the performance of SOTA BERT-based models with minimal reliance on manually annotated data and without fine-tuning. Furthermore, we performed model distillation on a closed-source LLM, demonstrating that a BERT model trained exclusively on synthetic data annotated by LLMs can also achieve a practical performance. Based on these findings, we explored the feasibility of partially replacing manual annotation with LLMs in production scenarios for biomedical text mining.
>
---
#### [replaced 040] Semantic Similarity-Informed Bayesian Borrowing for Quantitative Signal Detection of Adverse Events
- **分类: cs.CL; I.2.4; G.3; H.3.3**

- **链接: [http://arxiv.org/pdf/2504.12052v3](http://arxiv.org/pdf/2504.12052v3)**

> **作者:** François Haguinet; Jeffery L Painter; Gregory E Powell; Andrea Callegaro; Andrew Bate
>
> **备注:** 32 pages, 7 figures, 5 supplementary figures
>
> **摘要:** We present a Bayesian dynamic borrowing (BDB) approach to enhance the quantitative identification of adverse events (AEs) in spontaneous reporting systems (SRSs). The method embeds a robust meta-analytic predictive (MAP) prior with a Bayesian hierarchical model and incorporates semantic similarity measures (SSMs) to enable weighted information sharing from clinically similar MedDRA Preferred Terms (PTs) to the target PT. This continuous similarity-based borrowing overcomes limitations of rigid hierarchical grouping in current disproportionality analysis (DPA). Using data from the FDA Adverse Event Reporting System (FAERS) between 2015 and 2019, we evaluate our approach -- termed IC SSM -- against traditional Information Component (IC) analysis and IC with borrowing at the MedDRA high-level group term level (IC HLGT). A reference set (PVLens), derived from FDA product label update, enabled prospective evaluation of method performance in identifying AEs prior to official labeling. The IC SSM approach demonstrated higher sensitivity (1332/2337=0.570, Youden's J=0.246) than traditional IC (Se=0.501, J=0.250) and IC HLGT (Se=0.556, J=0.225), consistently identifying more true positives and doing so on average 5 months sooner than traditional IC. Despite a marginally lower aggregate F1-score and Youden's index, IC SSM showed higher performance in early post-marketing periods or when the detection threshold was raised, providing more stable and relevant alerts than IC HLGT and traditional IC. These findings support the use of SSM-informed Bayesian borrowing as a scalable and context-aware enhancement to traditional DPA methods, with potential for validation across other datasets and exploration of additional similarity metrics and Bayesian strategies using case-level data.
>
---
#### [replaced 041] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings
- **分类: cs.CL; cs.AI; cs.PL; cs.SE**

- **链接: [http://arxiv.org/pdf/2503.03008v2](http://arxiv.org/pdf/2503.03008v2)**

> **作者:** Andrea Gurioli; Federico Pennino; João Monteiro; Maurizio Gabbrielli
>
> **摘要:** Deploying language models often requires navigating accuracy vs. performance trade-offs to meet latency constraints while preserving utility. Traditional model distillation reduces size but incurs substantial costs through training separate models. We introduce ModularStarEncoder (MoSE), a 1-billion-parameter multi-exit encoder for code retrieval and classification that employs a novel Self-Distillation mechanism. This approach significantly enhances lower-layer representations, enabling flexible deployment of different model portions with favorable performance trade-offs. Our architecture improves text-to-code and code-to-code search by targeting specific encoder layers as exit heads, where higher layers guide earlier ones during training-improving intermediate representations at minimal additional cost. We further enhance MoSE with a repository-level contextual loss that maximizes training context window utilization. Additionally, we release a new dataset created through code translation that extends text-to-code benchmarks with cross-language code-to-code pairs. Evaluations demonstrate the effectiveness of Self-Distillation as a principled approach to trading inference cost for accuracy across various code understanding tasks.
>
---
#### [replaced 042] Effectively Controlling Reasoning Models through Thinking Intervention
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24370v2](http://arxiv.org/pdf/2503.24370v2)**

> **作者:** Tong Wu; Chong Xiang; Jiachen T. Wang; G. Edward Suh; Prateek Mittal
>
> **摘要:** Reasoning-enhanced large language models (LLMs) explicitly generate intermediate reasoning steps prior to generating final answers, helping the model excel in complex problem-solving. In this paper, we demonstrate that this emerging generation framework offers a unique opportunity for more fine-grained control over model behavior. We propose Thinking Intervention, a novel paradigm designed to explicitly guide the internal reasoning processes of LLMs by strategically inserting or revising specific thinking tokens. We find that the Thinking Intervention paradigm enhances the capabilities of reasoning models across a wide range of tasks, including instruction following on IFEval, instruction hierarchy on SEP, and safety alignment on XSTest and SorryBench. Our results demonstrate that Thinking Intervention significantly outperforms baseline prompting approaches, achieving up to 6.7% accuracy gains in instruction-following scenarios, 15.4% improvements in reasoning about instruction hierarchies, and a 40.0% increase in refusal rates for unsafe prompts using open-source DeepSeek R1 models. Overall, our work opens a promising new research avenue for controlling reasoning LLMs.
>
---
#### [replaced 043] Option-ID Based Elimination For Multiple Choice Questions
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15175v3](http://arxiv.org/pdf/2501.15175v3)**

> **作者:** Zhenhao Zhu; Bulou Liu; Qingyao Ai; Yiqun Liu
>
> **摘要:** Multiple choice questions (MCQs) are a popular and important task for evaluating large language models (LLMs). Based on common strategies people use when answering MCQs, the process of elimination (PoE) has been proposed as an effective problem-solving method. Existing PoE methods typically either have LLMs directly identify incorrect options or score options and replace lower-scoring ones with [MASK]. However, both methods suffer from inapplicability or suboptimal performance. To address these issues, this paper proposes a novel option-ID based PoE ($\text{PoE}_{\text{ID}}$). $\text{PoE}_{\text{ID}}$ critically incorporates a debiasing technique to counteract LLMs token bias, enhancing robustness over naive ID-based elimination. It features two strategies: $\text{PoE}_{\text{ID}}^{\text{log}}$, which eliminates options whose IDs have log probabilities below the average threshold, and $\text{PoE}_{\text{ID}}^{\text{seq}}$, which iteratively removes the option with the lowest ID probability. We conduct extensive experiments with 6 different LLMs on 4 diverse datasets. The results demonstrate that $\text{PoE}_{\text{ID}}$, especially $\text{PoE}_{\text{ID}}^{\text{log}}$, significantly improves zero-shot and few-shot MCQs performance, particularly in datasets with more options. Our analyses demonstrate that $\text{PoE}_{\text{ID}}^{\text{log}}$ enhances the LLMs' confidence in selecting the correct option, and the option elimination strategy outperforms methods relying on [MASK] replacement. We further investigate the limitations of LLMs in directly identifying incorrect options, which stem from their inherent deficiencies.
>
---
#### [replaced 044] DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10198v2](http://arxiv.org/pdf/2504.10198v2)**

> **作者:** Hanghui Guo; Jia Zhu; Shimin Di; Weijie Shi; Zhangze Chen; Jiajie Xu
>
> **备注:** Accepted to ACL2025 Main
>
> **摘要:** Dynamic Retrieval-augmented Generation (RAG) has shown great success in mitigating hallucinations in large language models (LLMs) during generation. However, existing dynamic RAG methods face significant limitations in two key aspects: 1) Lack of an effective mechanism to control retrieval triggers, and 2) Lack of effective scrutiny of retrieval content. To address these limitations, we propose an innovative dynamic RAG method, DioR (Adaptive Cognitive Detection and Contextual Retrieval Optimization), which consists of two main components: adaptive cognitive detection and contextual retrieval optimization, specifically designed to determine when retrieval is needed and what to retrieve for LLMs is useful. Experimental results demonstrate that DioR achieves superior performance on all tasks, demonstrating the effectiveness of our work.
>
---
#### [replaced 045] ClinicRealm: Re-evaluating Large Language Models with Conventional Machine Learning for Non-Generative Clinical Prediction Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.18525v2](http://arxiv.org/pdf/2407.18525v2)**

> **作者:** Yinghao Zhu; Junyi Gao; Zixiang Wang; Weibin Liao; Xiaochen Zheng; Lifang Liang; Miguel O. Bernabeu; Yasha Wang; Lequan Yu; Chengwei Pan; Ewen M. Harrison; Liantao Ma
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in medicine. However, their utility in non-generative clinical prediction, often presumed inferior to specialized models, remains under-evaluated, leading to ongoing debate within the field and potential for misuse, misunderstanding, or over-reliance due to a lack of systematic benchmarking. Our ClinicRealm study addresses this by benchmarking 9 GPT-based LLMs, 5 BERT-based models, and 7 traditional methods on unstructured clinical notes and structured Electronic Health Records (EHR). Key findings reveal a significant shift: for clinical note predictions, leading LLMs (e.g., DeepSeek R1/V3, GPT o3-mini-high) in zero-shot settings now decisively outperform finetuned BERT models. On structured EHRs, while specialized models excel with ample data, advanced LLMs (e.g., GPT-4o, DeepSeek R1/V3) show potent zero-shot capabilities, often surpassing conventional models in data-scarce settings. Notably, leading open-source LLMs can match or exceed proprietary counterparts. These results establish modern LLMs as powerful non-generative clinical prediction tools, particularly with unstructured text and offering data-efficient structured data options, thus necessitating a re-evaluation of model selection strategies. This research should serve as an important insight for medical informaticists, AI developers, and clinical researchers, potentially prompting a reassessment of current assumptions and inspiring new approaches to LLM application in predictive healthcare.
>
---
#### [replaced 046] Can MLLMs Generalize to Multi-Party dialog? Exploring Multilingual Response Generation in Complex Scenarios
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11269v2](http://arxiv.org/pdf/2501.11269v2)**

> **作者:** Zhongtian Hu; Yiwen Cui; Ronghan Li; Meng Zhao; Lifang Wang
>
> **摘要:** Current multilingual large language models(MLLMs) still focus on simple question-answering formats, often overlooking more complex dialogue scenarios. In other words, their capabilities of multilingual large models have yet to be validated in dialogue tasks with intricate structures. We therefore ask, Q1: How well do LLMs generalize to more complex dialog scenarios? Q2: Can supervised fine-tuning on a high-quality parallel benchmark restore this ability? Q3: Does the "multilingual complementarity" effect survive in the setting? To answer these questions, we introduce XMP, a high-quality parallel Multilingual dataset sourced from Multi-party Podcast dialogues, which is the first parallel dataset focusing on multi-party dialogue scenarios. Most samples in the dataset feature three or more participants, discussing a wide range of topics. Through extensive experiments, we find that, R1: MLLMs fail to generalize to multi-party setting, R2 Fine-tuning on XMP improves only marginally, with the 70B model achieving at most a 1% absolute gain over its 8B counterpart; R3: Mixing languages during SFT is usually detrimental, with any benefits being marginal and limited to isolated cases in the 70B model.
>
---
#### [replaced 047] Beyond Single-Task: Robust Multi-Task Length Generalization for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11525v2](http://arxiv.org/pdf/2502.11525v2)**

> **作者:** Yi Hu; Shijia Kang; Haotong Yang; Haotian Xu; Muhan Zhang
>
> **摘要:** Length generalization, the ability to solve problems longer than those seen during training, remains a critical challenge for large language models (LLMs). Previous work modifies positional encodings (PEs) and data formats to improve length generalization on specific symbolic tasks such as addition and sorting. However, these approaches are fundamentally limited to special tasks, often degrading general language performance. Furthermore, they are typically evaluated on small transformers trained from scratch on single tasks and can cause performance drop when applied during post-training stage of practical LLMs with general capabilities. Hu et al., (2024) proposed Rule-Following Fine-Tuning (RFFT) to improve length generalization in the post-training stage of LLMs. Despite its compatibility with practical models and strong performance, RFFT is proposed for single tasks too, requiring re-training for each individual task with extensive examples. In this paper, we study length generalization in multi-task settings and propose Meta Rule-Following Fine-Tuning (Meta-RFFT), the first framework enabling robust cross-task length generalization. As our first contribution, we construct a large length generalization dataset containing 86 tasks spanning code execution, number processing, symbolic and logical reasoning tasks, beyond the common addition or multiplication tasks. Secondly, we show that cross-task length generalization is possible with Meta-RFFT. After training on a large number of tasks and instances, the models achieve remarkable length generalization ability on unseen tasks with minimal fine-tuning or one-shot prompting. For example, after fine-tuning on 1 to 5 digit addition, our 32B model achieves 95% accuracy on 30 digit addition, significantly outperforming the state-of-the-art reasoning models (DeepSeek-R1-671B: 72%), despite never seeing this task during RF-pretraining.
>
---
#### [replaced 048] Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation Without Parallel Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14669v2](http://arxiv.org/pdf/2504.14669v2)**

> **作者:** Wei Zou; Sen Yang; Yu Bao; Shujian Huang; Jiajun Chen; Shanbo Cheng
>
> **备注:** 11 pages, 4 figures, accepted by ACL 2025 as findings
>
> **摘要:** The rise of Large Language Models (LLMs) has reshaped machine translation (MT), but multilingual MT still relies heavily on parallel data for supervised fine-tuning (SFT), facing challenges like data scarcity for low-resource languages and catastrophic forgetting. To address these issues, we propose TRANS-ZERO, a self-play framework that leverages only monolingual data and the intrinsic multilingual knowledge of LLM. TRANS-ZERO combines Genetic Monte-Carlo Tree Search (G-MCTS) with preference optimization, achieving strong translation performance that rivals supervised methods. Experiments demonstrate that this approach not only matches the performance of models trained on large-scale parallel data but also excels in non-English translation directions. Further analysis reveals that G-MCTS itself significantly enhances translation quality by exploring semantically consistent candidates through iterative translations, providing a robust foundation for the framework's succuss.
>
---
#### [replaced 049] OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.16918v2](http://arxiv.org/pdf/2504.16918v2)**

> **作者:** Raghav Thind; Youran Sun; Ling Liang; Haizhao Yang
>
> **摘要:** Optimization plays a vital role in scientific research and practical applications. However, formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce OptimAI, a framework for solving Optimization problems described in natural language by leveraging LLM-powered AI agents, and achieve superior performance over current state-of-the-art methods. Our framework is built upon the following key roles: (1) a formulator that translates natural language problem descriptions into precise mathematical formulations; (2) a planner that constructs a high-level solution strategy prior to execution; and (3) a coder and a code critic capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, and our experiments confirm that combining diverse models leads to performance gains. Our approach attains 88.1% accuracy on the NLP4LP dataset and 82.3% on the Optibench dataset, reducing error rates by 58% and 52%, respectively, over prior best results.
>
---
#### [replaced 050] Machine-generated text detection prevents language model collapse
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.15654v5](http://arxiv.org/pdf/2502.15654v5)**

> **作者:** George Drayson; Emine Yilmaz; Vasileios Lampos
>
> **摘要:** As Large Language Models (LLMs) become increasingly prevalent, their generated outputs are proliferating across the web, risking a future where machine-generated content dilutes human-authored text. Since online data is the primary resource for LLM pre-training, subsequent models could be trained on an unknown portion of synthetic samples. This will lead to model collapse, a degenerative process whereby LLMs reinforce their own errors, converge to a low variance output distribution, and ultimately yield a declining performance. In this study, we investigate the impact of decoding strategy on model collapse, analysing the text characteristics at each model generation, the similarity to human references, and the resulting model performance. Using the decoding strategies that lead to the most significant degradation, we evaluate model collapse in more realistic scenarios where the origin of the data (human or synthetic) is unknown. We train a machine-generated text detector and propose an importance sampling approach to alleviate model collapse. Our method is validated on two LLM variants (GPT-2 and SmolLM2), across a range of model sizes (124M to 1.7B), on the open-ended text generation task. We demonstrate that it can not only prevent model collapse but also improve performance when sufficient human-authored samples are present. Source code: github.com/GeorgeDrayson/model_collapse.
>
---
#### [replaced 051] Enhancing LLM Evaluations: The Garbling Trick
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.01533v3](http://arxiv.org/pdf/2411.01533v3)**

> **作者:** William F. Bradley
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** As large language models (LLMs) become increasingly powerful, traditional evaluation metrics tend to saturate, making it challenging to distinguish between models. We propose a general method to transform existing LLM evaluations into a series of progressively more difficult tasks. These enhanced evaluations emphasize reasoning capabilities and can reveal relative performance differences that are not apparent in the original assessments. To demonstrate the effectiveness of our approach, we create a new multiple-choice test corpus, extend it into a family of evaluations, and assess a collection of LLMs. Our results offer insights into the comparative abilities of these models, particularly highlighting the differences between base LLMs and more recent "reasoning" models.
>
---
#### [replaced 052] BAT: Learning to Reason about Spatial Sounds with Large Language Models
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2402.01591v3](http://arxiv.org/pdf/2402.01591v3)**

> **作者:** Zhisheng Zheng; Puyuan Peng; Ziyang Ma; Xie Chen; Eunsol Choi; David Harwath
>
> **备注:** Accepted to ICML 2024. Our demo, dataset, code and model weights are available at: https://zhishengzheng.com/bat
>
> **摘要:** Spatial sound reasoning is a fundamental human skill, enabling us to navigate and interpret our surroundings based on sound. In this paper we present BAT, which combines the spatial sound perception ability of a binaural acoustic scene analysis model with the natural language reasoning capabilities of a large language model (LLM) to replicate this innate ability. To address the lack of existing datasets of in-the-wild spatial sounds, we synthesized a binaural audio dataset using AudioSet and SoundSpaces 2.0. Next, we developed SpatialSoundQA, a spatial sound-based question-answering dataset, offering a range of QA tasks that train BAT in various aspects of spatial sound perception and reasoning. The acoustic front end encoder of BAT is a novel spatial audio encoder named Spatial Audio Spectrogram Transformer, or Spatial-AST, which by itself achieves strong performance across sound event detection, spatial localization, and distance estimation. By integrating Spatial-AST with LLaMA-2 7B model, BAT transcends standard Sound Event Localization and Detection (SELD) tasks, enabling the model to reason about the relationships between the sounds in its environment. Our experiments demonstrate BAT's superior performance on both spatial sound perception and reasoning, showcasing the immense potential of LLMs in navigating and interpreting complex spatial audio environments.
>
---
#### [replaced 053] Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework
- **分类: cs.CL; cs.AI; cs.MA; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2411.16707v3](http://arxiv.org/pdf/2411.16707v3)**

> **作者:** Mengshuo Jia; Zeyu Cui; Gabriela Hug
>
> **备注:** 16 pages
>
> **摘要:** The integration of experimental technologies with large language models (LLMs) is transforming scientific research. It positions AI as a versatile research assistant rather than a mere problem-solving tool. In the field of power systems, however, managing simulations -- one of the essential experimental technologies -- remains a challenge for LLMs due to their limited domain-specific knowledge, restricted reasoning capabilities, and imprecise handling of simulation parameters. To address these limitations, this paper proposes a feedback-driven, multi-agent framework. It incorporates three proposed modules: an enhanced retrieval-augmented generation (RAG) module, an improved reasoning module, and a dynamic environmental acting module with an error-feedback mechanism. Validated on 69 diverse tasks from Daline and MATPOWER, this framework achieves success rates of 93.13% and 96.85%, respectively. It significantly outperforms ChatGPT 4o, o1-preview, and the fine-tuned GPT-4o, which all achieved a success rate lower than 30% on complex tasks. Additionally, the proposed framework also supports rapid, cost-effective task execution, completing each simulation in approximately 30 seconds at an average cost of 0.014 USD for tokens. Overall, this adaptable framework lays a foundation for developing intelligent LLM-based assistants for human researchers, facilitating power system research and beyond.
>
---
#### [replaced 054] Process Reward Models That Think
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16828v2](http://arxiv.org/pdf/2504.16828v2)**

> **作者:** Muhammad Khalifa; Rishabh Agarwal; Lajanugen Logeswaran; Jaekyeom Kim; Hao Peng; Moontae Lee; Honglak Lee; Lu Wang
>
> **摘要:** Step-by-step verifiers -- also known as process reward models (PRMs) -- are a key ingredient for test-time scaling. PRMs require step-level supervision, making them expensive to train. This work aims to build data-efficient PRMs as verbalized step-wise reward models that verify every step in the solution by generating a verification chain-of-thought (CoT). We propose ThinkPRM, a long CoT verifier fine-tuned on orders of magnitude fewer process labels than those required by discriminative PRMs. Our approach capitalizes on the inherent reasoning abilities of long CoT models, and outperforms LLM-as-a-Judge and discriminative verifiers -- using only 1% of the process labels in PRM800K -- across several challenging benchmarks. Specifically, ThinkPRM beats the baselines on ProcessBench, MATH-500, and AIME '24 under best-of-N selection and reward-guided search. In an out-of-domain evaluation on a subset of GPQA-Diamond and LiveCodeBench, our PRM surpasses discriminative verifiers trained on the full PRM800K by 8% and 4.5%, respectively. Lastly, under the same token budget, ThinkPRM scales up verification compute more effectively compared to LLM-as-a-Judge, outperforming it by 7.2% on a subset of ProcessBench. Our work highlights the value of generative, long CoT PRMs that can scale test-time compute for verification while requiring minimal supervision for training. Our code, data, and models will be released at https://github.com/mukhal/thinkprm.
>
---
#### [replaced 055] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10872v2](http://arxiv.org/pdf/2505.10872v2)**

> **作者:** Chenxi Jiang; Chuhao Zhou; Jianfei Yang
>
> **备注:** Under Review
>
> **摘要:** Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark with vague REs (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 77.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompt and chains of thought. This work contributes to the research community of human-robot interaction (HRI) by making robot task planning more practical, particularly for non-expert users, e.g., the elderly and children.
>
---
#### [replaced 056] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.14411v2](http://arxiv.org/pdf/2503.14411v2)**

> **作者:** Siwei Zhang; Yun Xiong; Yateng Tang; Xi Chen; Zian Jia; Zehao Gu; Jiarong Xu; Jiawei Zhang
>
> **备注:** Submit to NeurIPS2025
>
> **摘要:** Temporal graph neural networks (TGNNs) have shown remarkable performance in temporal graph modeling. However, real-world temporal graphs often possess rich textual information, giving rise to temporal text-attributed graphs (TTAGs). Such combination of dynamic text semantics and evolving graph structures introduces heightened complexity. Existing TGNNs embed texts statically and rely heavily on encoding mechanisms that biasedly prioritize structural information, overlooking the temporal evolution of text semantics and the essential interplay between semantics and structures for synergistic reinforcement. To tackle these issues, we present \textbf{CROSS}, a flexible framework that seamlessly extends existing TGNNs for TTAG modeling. CROSS is designed by decomposing the TTAG modeling process into two phases: (i) temporal semantics extraction; and (ii) semantic-structural information unification. The key idea is to advance the large language models (LLMs) to dynamically extract the temporal semantics in text space and then generate cohesive representations unifying both semantics and structures. Specifically, we propose a Temporal Semantics Extractor in the CROSS framework, which empowers LLMs to offer the temporal semantic understanding of node's evolving contexts of textual neighborhoods, facilitating semantic dynamics. Subsequently, we introduce the Semantic-structural Co-encoder, which collaborates with the above Extractor for synthesizing illuminating representations by jointly considering both semantic and structural information while encouraging their mutual reinforcement. Extensive experiments show that CROSS achieves state-of-the-art results on four public datasets and one industrial dataset, with 24.7% absolute MRR gain on average in temporal link prediction and 3.7% AUC gain in node classification of industrial application.
>
---
#### [replaced 057] A Pilot Empirical Study on When and How to Use Knowledge Graphs as Retrieval Augmented Generation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20854v3](http://arxiv.org/pdf/2502.20854v3)**

> **作者:** Xujie Yuan; Yongxu Liu; Shimin Di; Shiwen Wu; Libin Zheng; Rui Meng; Lei Chen; Xiaofang Zhou; Jian Yin
>
> **备注:** 9 pages, 2 figures, 19 tables
>
> **摘要:** The integration of Knowledge Graphs (KGs) into the Retrieval Augmented Generation (RAG) framework has attracted significant interest, with early studies showing promise in mitigating hallucinations and improving model accuracy. However, a systematic understanding and comparative analysis of the rapidly emerging KG-RAG methods are still lacking. This paper seeks to lay the foundation for systematically answering the question of when and how to use KG-RAG by analyzing their performance in various application scenarios associated with different technical configurations. After outlining the mind map using KG-RAG framework and summarizing its popular pipeline, we conduct a pilot empirical study of KG-RAG works to reimplement and evaluate 6 KG-RAG methods across 9 datasets in diverse domains and scenarios, analyzing the impact of 9 KG-RAG configurations in combination with 17 LLMs, and combining Metacognition with KG-RAG as a pilot attempt. Our results underscore the critical role of appropriate application conditions and optimal configurations of KG-RAG components.
>
---
#### [replaced 058] Can ChatGPT capture swearing nuances? Evidence from translating Arabic oaths
- **分类: cs.CL; cs-CL; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2412.02466v3](http://arxiv.org/pdf/2412.02466v3)**

> **作者:** Mohammed Q. Shormani
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** This study sets out to answer one major question: Can ChatGPT capture swearing nuances? It presents an empirical study on the ability of ChatGPT to translate Arabic oath expressions into English. 30 Arabic oath expressions were collected from the literature. These 30 oaths were first translated via ChatGPT and then analyzed and compared to the human translation in terms of types of gaps left unfulfilled by ChatGPT. Specifically, the gaps involved are: religious gap, cultural gap, both religious and cultural gaps, no gap, using non-oath particles, redundancy and noncapturing of Arabic script diacritics. It concludes that ChatGPT translation of oaths is still much unsatisfactory, unveiling the need of further developments of ChatGPT, and the inclusion of Arabic data on which ChatGPT should be trained including oath expressions, oath nuances, rituals, and practices.
>
---
#### [replaced 059] Controlled Training Data Generation with Diffusion Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.15309v2](http://arxiv.org/pdf/2403.15309v2)**

> **作者:** Teresa Yeo; Andrei Atanov; Harold Benoit; Aleksandr Alekseev; Ruchira Ray; Pooya Esmaeil Akhoondi; Amir Zamir
>
> **备注:** Project page at https://adversarial-prompts.epfl.ch/
>
> **摘要:** We present a method to control a text-to-image generative model to produce training data useful for supervised learning. Unlike previous works that employ an open-loop approach and pre-define prompts to generate new data using either a language model or human expertise, we develop an automated closed-loop system which involves two feedback mechanisms. The first mechanism uses feedback from a given supervised model and finds adversarial prompts that result in image generations that maximize the model loss. While these adversarial prompts result in diverse data informed by the model, they are not informed of the target distribution, which can be inefficient. Therefore, we introduce the second feedback mechanism that guides the generation process towards a certain target distribution. We call the method combining these two mechanisms Guided Adversarial Prompts. We perform our evaluations on different tasks, datasets and architectures, with different types of distribution shifts (spuriously correlated data, unseen domains) and demonstrate the efficiency of the proposed feedback mechanisms compared to open-loop approaches.
>
---
#### [replaced 060] Large Language Models Could Be Rote Learners
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08300v4](http://arxiv.org/pdf/2504.08300v4)**

> **作者:** Yuyang Xu; Renjun Hu; Haochao Ying; Jian Wu; Xing Shi; Wei Lin
>
> **备注:** Work in Progress
>
> **摘要:** Multiple-choice question (MCQ) benchmarks are widely used for evaluating Large Language Models (LLMs), yet their reliability is undermined by benchmark contamination. In this study, we reframe contamination as an inherent aspect of learning and seek to disentangle genuine capability acquisition from superficial memorization in LLM evaluation. First, by analyzing model performance under different memorization conditions, we uncover a counterintuitive trend: LLMs perform worse on memorized MCQs than on non-memorized ones, indicating the coexistence of two distinct learning phenomena, i.e., rote memorization and genuine capability learning. To disentangle them, we propose TrinEval, a novel evaluation framework reformulating MCQs into an alternative trinity format, reducing memorization while preserving knowledge assessment. Experiments validate TrinEval's effectiveness in reformulation, and its evaluation reveals that common LLMs may memorize by rote 20.5% of knowledge points (in MMLU on average).
>
---
#### [replaced 061] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11799v2](http://arxiv.org/pdf/2502.11799v2)**

> **作者:** Peiying Yu; Guoxin Chen; Jingjing Wang
>
> **备注:** ACL 2025 Main
>
> **摘要:** Despite the remarkable capabilities of large language models (LLMs) in various reasoning tasks, they still struggle with table reasoning tasks, particularly in maintaining consistency throughout multi-step reasoning processes. While existing approaches have explored various decomposition strategies, they often lack effective mechanisms to identify and correct errors in intermediate reasoning steps, leading to cascading error propagation. To address these issues, we propose Table-Critic, a novel multi-agent framework that facilitates collaborative criticism and iterative refinement of the reasoning process until convergence to correct solutions. Our framework consists of four specialized agents: a Judge for error identification, a Critic for comprehensive critiques, a Refiner for process improvement, and a Curator for pattern distillation. To effectively deal with diverse and unpredictable error types, we introduce a self-evolving template tree that systematically accumulates critique knowledge through experience-driven learning and guides future reflections. Extensive experiments have demonstrated that Table-Critic achieves substantial improvements over existing methods, achieving superior accuracy and error correction rates while maintaining computational efficiency and lower solution degradation rate.
>
---
#### [replaced 062] Gradient descent with generalized Newton's method
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02772v3](http://arxiv.org/pdf/2407.02772v3)**

> **作者:** Zhiqi Bu; Shiyun Xu
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** We propose the generalized Newton's method (GeN) -- a Hessian-informed approach that applies to any optimizer such as SGD and Adam, and covers the Newton-Raphson method as a sub-case. Our method automatically and dynamically selects the learning rate that accelerates the convergence, without the intensive tuning of the learning rate scheduler. In practice, our method is easily implementable, since it only requires additional forward passes with almost zero computational overhead (in terms of training time and memory cost), if the overhead is amortized over many iterations. We present extensive experiments on language and vision tasks (e.g. GPT and ResNet) to showcase that GeN optimizers match the state-of-the-art performance, which was achieved with carefully tuned learning rate schedulers.
>
---
#### [replaced 063] Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08780v5](http://arxiv.org/pdf/2408.08780v5)**

> **作者:** Chenming Tang; Zhixiang Wang; Hao Sun; Yunfang Wu
>
> **摘要:** With the help of in-context learning (ICL), large language models (LLMs) have achieved impressive performance across various tasks. However, the function of descriptive instructions during ICL remains under-explored. In this work, we propose an ensemble prompt framework to describe the selection criteria of multiple in-context examples, and preliminary experiments on machine translation (MT) across six translation directions confirm that this framework boosts ICL performance. But to our surprise, LLMs might not care what the descriptions actually say, and the performance gain is primarily caused by the ensemble format, since it could lead to improvement even with random descriptive nouns. We further apply this new ensemble framework on a range of commonsense, math, logical reasoning and hallucination tasks with three LLMs and achieve promising results, suggesting again that designing a proper prompt format would be much more effective and efficient than paying effort into specific descriptions. Our code will be publicly available once this paper is published.
>
---
#### [replaced 064] DateLogicQA: Benchmarking Temporal Biases in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.13377v2](http://arxiv.org/pdf/2412.13377v2)**

> **作者:** Gagan Bhatia; MingZe Tang; Cristina Mahanta; Madiha Kazi
>
> **摘要:** This paper introduces DateLogicQA, a benchmark with 190 questions covering diverse date formats, temporal contexts, and reasoning types. We propose the Semantic Integrity Metric to assess tokenization quality and analyse two biases: Representation-Level Bias, affecting embeddings, and Logical-Level Bias, influencing reasoning outputs. Our findings provide a comprehensive evaluation of LLMs' capabilities and limitations in temporal reasoning, highlighting key challenges in handling temporal data accurately.
>
---
#### [replaced 065] Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.08696v2](http://arxiv.org/pdf/2408.08696v2)**

> **作者:** Xianzhen Luo; Yixuan Wang; Qingfu Zhu; Zhiming Zhang; Xuanyu Zhang; Qing Yang; Dongliang Xu
>
> **备注:** Accepted by ACL2025. Code is [here](https://github.com/Luowaterbi/TokenRecycling). Token Recycling has already merged into [SpecBench](https://github.com/hemingkx/Spec-Bench)
>
> **摘要:** Massive parameters of LLMs have made inference latency a fundamental bottleneck. Speculative decoding represents a lossless approach to accelerate inference through a guess-and-verify paradigm. Some methods rely on additional architectures to guess draft tokens, which need extra training before use. Alternatively, retrieval-based training-free techniques build libraries from pre-existing corpora or by n-gram generation. However, they face challenges like large storage requirements, time-consuming retrieval, and limited adaptability. Observing that candidate tokens generated during the decoding process are likely to reoccur in future sequences, we propose Token Recycling. It stores candidate tokens in an adjacency matrix and employs a breadth-first-search (BFS)-like algorithm to construct a draft tree, which is then validated through tree attention. New candidate tokens from the decoding process are then used to update the matrix. Token Recycling requires \textless2MB of additional storage and achieves approximately 2x speedup across all sizes of LLMs. It significantly outperforms existing train-free methods by 30\% and even a widely recognized training method by 25\%.
>
---
#### [replaced 066] Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.06577v2](http://arxiv.org/pdf/2410.06577v2)**

> **作者:** Zhihao He; Hang Yu; Zi Gong; Shizhan Liu; Jianguo Li; Weiyao Lin
>
> **备注:** Accepted by ICLR 2025. Camera-ready Version
>
> **摘要:** Recent advancements in Transformer-based large language models (LLMs) have set new standards in natural language processing. However, the classical softmax attention incurs significant computational costs, leading to a $O(T)$ complexity for per-token generation, where $T$ represents the context length. This work explores reducing LLMs' complexity while maintaining performance by introducing Rodimus and its enhanced version, Rodimus$+$. Rodimus employs an innovative data-dependent tempered selection (DDTS) mechanism within a linear attention-based, purely recurrent framework, achieving significant accuracy while drastically reducing the memory usage typically associated with recurrent models. This method exemplifies semantic compression by maintaining essential input information with fixed-size hidden states. Building on this, Rodimus$+$ combines Rodimus with the innovative Sliding Window Shared-Key Attention (SW-SKA) in a hybrid approach, effectively leveraging the complementary semantic, token, and head compression techniques. Our experiments demonstrate that Rodimus$+$-1.6B, trained on 1 trillion tokens, achieves superior downstream performance against models trained on more tokens, including Qwen2-1.5B and RWKV6-1.6B, underscoring its potential to redefine the accuracy-efficiency balance in LLMs. Model code and pre-trained checkpoints are open-sourced at https://github.com/codefuse-ai/rodimus.
>
---
#### [replaced 067] Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2501.18280v3](http://arxiv.org/pdf/2501.18280v3)**

> **作者:** Haoyu Liang; Youran Sun; Yunfeng Cai; Jun Zhu; Bo Zhang
>
> **摘要:** The security issue of large language models (LLMs) has gained wide attention recently, with various defense mechanisms developed to prevent harmful output, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the output distribution of text embedding models is severely biased with a large mean. Inspired by this observation, we propose novel, efficient methods to search for **universal magic words** that attack text embedding models. Universal magic words as suffixes can shift the embedding of any text towards the bias direction, thus manipulating the similarity of any text pair and misleading safeguards. Attackers can jailbreak the safeguards by appending magic words to user prompts and requiring LLMs to end answers with magic words. Experiments show that magic word attacks significantly degrade safeguard performance on JailbreakBench, cause real-world chatbots to produce harmful outputs in full-pipeline attacks, and generalize across input/output texts, models, and languages. To eradicate this security risk, we also propose defense methods against such attacks, which can correct the bias of text embeddings and improve downstream performance in a train-free manner.
>
---
#### [replaced 068] Is This Collection Worth My LLM's Time? Automatically Measuring Information Potential in Text Corpora
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13691v2](http://arxiv.org/pdf/2502.13691v2)**

> **作者:** Tristan Karch; Luca Engel; Philippe Schwaller; Frédéric Kaplan
>
> **摘要:** As large language models (LLMs) converge towards similar capabilities, the key to advancing their performance lies in identifying and incorporating valuable new information sources. However, evaluating which text collections are worth the substantial investment required for digitization, preprocessing, and integration into LLM systems remains a significant challenge. We present a novel approach to this challenge: an automated pipeline that evaluates the potential information gain from text collections without requiring model training or fine-tuning. Our method generates multiple choice questions (MCQs) from texts and measures an LLM's performance both with and without access to the source material. The performance gap between these conditions serves as a proxy for the collection's information potential. We validate our approach using five strategically selected datasets: EPFL PhD manuscripts, a private collection of Venetian historical records, two sets of Wikipedia articles on related topics, and a synthetic baseline dataset. Our results demonstrate that this method effectively identifies collections containing valuable novel information, providing a practical tool for prioritizing data acquisition and integration efforts.
>
---
#### [replaced 069] Eye Tracking Based Cognitive Evaluation of Automatic Readability Assessment Measures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11150v2](http://arxiv.org/pdf/2502.11150v2)**

> **作者:** Keren Gruteke Klein; Shachar Frenkel; Omer Shubi; Yevgeni Berzak
>
> **摘要:** Automated text readability prediction is widely used in many real-world scenarios. Over the past century, such measures have primarily been developed and evaluated on reading comprehension outcomes and on human annotations of text readability levels. In this work, we propose an alternative, eye tracking-based cognitive framework which directly taps into a key aspect of readability: reading ease. We use this framework for evaluating a broad range of prominent readability measures, including two systems widely used in education, by quantifying their ability to account for reading facilitation effects in text simplification, as well as text reading ease more broadly. Our analyses suggest that existing readability measures are poor predictors of reading facilitation and reading ease, outperformed by word properties commonly used in psycholinguistics, and in particular by surprisal.
>
---
#### [replaced 070] Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00234v3](http://arxiv.org/pdf/2505.00234v3)**

> **作者:** Vishnu Sarukkai; Zhiqiang Xie; Kayvon Fatahalian
>
> **摘要:** Improving Large Language Model (LLM) agents for sequential decision-making tasks typically requires extensive task-specific knowledge engineering--custom prompts, curated examples, and specialized observation/action spaces. We investigate a different approach where agents automatically improve by learning from their own successful experiences without human intervention. Our method constructs and refines a database of self-generated trajectories that serve as in-context examples for future tasks. Even naive accumulation of successful trajectories yields substantial performance gains across three diverse benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%). These improvements exceed those achieved by upgrading from gpt-4o-mini to gpt-4o and match the performance of allowing multiple attempts per task. We further enhance this approach with two innovations: database-level curation using population-based training to propagate high-performing example collections, and exemplar-level curation that selectively retains trajectories based on their empirical utility as in-context examples. With these enhancements, our method achieves 93% success on ALFWorld--surpassing approaches that use more powerful LLMs and hand-crafted components. Our trajectory bootstrapping technique demonstrates that agents can autonomously improve through experience, offering a scalable alternative to labor-intensive knowledge engineering.
>
---
#### [replaced 071] From Languages to Geographies: Towards Evaluating Cultural Bias in Hate Speech Datasets
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17874v2](http://arxiv.org/pdf/2404.17874v2)**

> **作者:** Manuel Tonneau; Diyi Liu; Samuel Fraiberger; Ralph Schroeder; Scott A. Hale; Paul Röttger
>
> **备注:** Accepted at WOAH (NAACL 2024). Please cite the ACL Anthology version: https://aclanthology.org/2024.woah-1.23/
>
> **摘要:** Perceptions of hate can vary greatly across cultural contexts. Hate speech (HS) datasets, however, have traditionally been developed by language. This hides potential cultural biases, as one language may be spoken in different countries home to different cultures. In this work, we evaluate cultural bias in HS datasets by leveraging two interrelated cultural proxies: language and geography. We conduct a systematic survey of HS datasets in eight languages and confirm past findings on their English-language bias, but also show that this bias has been steadily decreasing in the past few years. For three geographically-widespread languages -- English, Arabic and Spanish -- we then leverage geographical metadata from tweets to approximate geo-cultural contexts by pairing language and country information. We find that HS datasets for these languages exhibit a strong geo-cultural bias, largely overrepresenting a handful of countries (e.g., US and UK for English) relative to their prominence in both the broader social media population and the general population speaking these languages. Based on these findings, we formulate recommendations for the creation of future HS datasets.
>
---
#### [replaced 072] WorldPM: Scaling Human Preference Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10527v2](http://arxiv.org/pdf/2505.10527v2)**

> **作者:** Binghai Wang; Runji Lin; Keming Lu; Le Yu; Zhenru Zhang; Fei Huang; Chujie Zheng; Kai Dang; Yang Fan; Xingzhang Ren; An Yang; Binyuan Hui; Dayiheng Liu; Tao Gui; Qi Zhang; Xuanjing Huang; Yu-Gang Jiang; Bowen Yu; Jingren Zhou; Junyang Lin
>
> **摘要:** Motivated by scaling laws in language modeling that demonstrate how test loss scales as a power law with model and dataset sizes, we find that similar laws exist in preference modeling. We propose World Preference Modeling$ (WorldPM) to emphasize this scaling potential, where World Preference embodies a unified representation of human preferences. In this paper, we collect preference data from public forums covering diverse user communities, and conduct extensive training using 15M-scale data across models ranging from 1.5B to 72B parameters. We observe distinct patterns across different evaluation metrics: (1) Adversarial metrics (ability to identify deceptive features) consistently scale up with increased training data and base model size; (2) Objective metrics (objective knowledge with well-defined answers) show emergent behavior in larger language models, highlighting WorldPM's scalability potential; (3) Subjective metrics (subjective preferences from a limited number of humans or AI) do not demonstrate scaling trends. Further experiments validate the effectiveness of WorldPM as a foundation for preference fine-tuning. Through evaluations on 7 benchmarks with 20 subtasks, we find that WorldPM broadly improves the generalization performance across human preference datasets of varying sizes (7K, 100K and 800K samples), with performance gains exceeding 5% on many key subtasks. Integrating WorldPM into our internal RLHF pipeline, we observe significant improvements on both in-house and public evaluation sets, with notable gains of 4% to 8% in our in-house evaluations.
>
---
#### [replaced 073] AlignRAG: Leveraging Critique Learning for Evidence-Sensitive Retrieval-Augmented Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14858v2](http://arxiv.org/pdf/2504.14858v2)**

> **作者:** Jiaqi Wei; Hao Zhou; Xiang Zhang; Di Zhang; Zijie Qiu; Wei Wei; Jinzhe Li; Wanli Ouyang; Siqi Sun
>
> **摘要:** Retrieval-augmented generation (RAG) has become a widely adopted paradigm for enabling knowledge-grounded large language models (LLMs). However, standard RAG pipelines often fail to ensure that model reasoning remains consistent with the evidence retrieved, leading to factual inconsistencies or unsupported conclusions. In this work, we reinterpret RAG as Retrieval-Augmented Reasoning and identify a central but underexplored problem: \textit{Reasoning Misalignment}-the divergence between an LLM's internal reasoning trajectory and the evidential constraints provided by retrieval. To address this issue, we propose \textsc{AlignRAG}, a novel iterative framework grounded in Critique-Driven Alignment (CDA). At the heart of \textsc{AlignRAG} lies a \textit{contrastive critique synthesis} mechanism that generates retrieval-sensitive critiques while mitigating self-bias. This mechanism trains a dedicated retrieval-augmented \textit{Critic Language Model (CLM)} using labeled critiques that distinguish between evidence-aligned and misaligned reasoning. Alignment signals for supervision are obtained through self-supervised or externally guided labeling strategies. The resulting CLM is explicitly optimized for evidence sensitivity, enabling it to detect and revise reasoning errors during inference without relying solely on self-generated feedback. Empirical evaluations show that our 8B-parameter CLM improves performance over the Self-Refine baseline by 12.1\% on out-of-domain tasks and outperforms a standard 72B-parameter CLM by 2.2\%, while remaining compatible with existing RAG architectures as a plug-and-play module. Overall, AlignRAG offers a principled solution for aligning model reasoning with retrieved evidence, substantially improving the factual reliability and robustness of RAG systems.
>
---
#### [replaced 074] Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20034v2](http://arxiv.org/pdf/2502.20034v2)**

> **作者:** Hongseok Oh; Wonseok Hwang
>
> **备注:** 4 pages
>
> **摘要:** Recently, Large Vision-Language Models (LVLMs) show remarkable performance across various domains. However, these models suffer from object hallucination. This study revisits the previous claim that the primary cause of such hallucination lies in the limited representational capacity of the vision encoder. Our analysis reveals that the capacity of the vision encoder itself is already adequate for detecting object hallucination. Based on this insight, we propose a Fine-grained CLIPScore (F-CLIPScore), a simple yet effective evaluation metric that enhances object-level granularity by incorporating text embeddings at the noun level. Evaluations on the OHD-Caps benchmark show that F-CLIPScore significantly outperforms conventional CLIPScore in accuracy by a large margin of 39.6\% without additional training. We further demonstrate that F-CLIPScore-based data filtering reduces object hallucination in LVLMs (4.9\% in POPE).
>
---
#### [replaced 075] SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.05517v4](http://arxiv.org/pdf/2408.05517v4)**

> **作者:** Yuze Zhao; Jintao Huang; Jinghan Hu; Xingjun Wang; Yunlin Mao; Daoze Zhang; Hong Zhang; Zeyinzi Jiang; Zhikai Wu; Baole Ai; Ang Wang; Wenmeng Zhou; Yingda Chen
>
> **摘要:** Recent development in Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) have leverage Attention-based Transformer architectures and achieved superior performance and generalization capabilities. They have since covered extensive areas of traditional learning tasks. For instance, text-based tasks such as text-classification and sequence-labeling, as well as multi-modal tasks like Visual Question Answering (VQA) and Optical Character Recognition (OCR), which were previously addressed using different models, can now be tackled based on one foundation model. Consequently, the training and lightweight fine-tuning of LLMs and MLLMs, especially those based on Transformer architecture, has become particularly important. In recognition of these overwhelming needs, we develop SWIFT, a customizable one-stop infrastructure for large models. With support of over $300+$ LLMs and $50+$ MLLMs, SWIFT stands as the open-source framework that provide the most comprehensive support for fine-tuning large models. In particular, it is the first training framework that provides systematic support for MLLMs. In addition to the core functionalities of fine-tuning, SWIFT also integrates post-training processes such as inference, evaluation, and model quantization, to facilitate fast adoptions of large models in various application scenarios. With a systematic integration of various training techniques, SWIFT offers helpful utilities such as benchmark comparisons among different training techniques for large models. For fine-tuning models specialized in agent framework, we show that notable improvements on the ToolBench leader-board can be achieved by training with customized dataset on SWIFT, with an increase of 5.2%-21.8% in the Act.EM metric over various baseline models, a reduction in hallucination by 1.6%-14.1%, and an average performance improvement of 8%-17%.
>
---
#### [replaced 076] FormulaReasoning: A Dataset for Formula-Based Numerical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.12692v5](http://arxiv.org/pdf/2402.12692v5)**

> **作者:** Xiao Li; Bolin Zhu; Kaiwen Shi; Sichen Liu; Yin Zhu; Yiwei Liu; Gong Cheng
>
> **摘要:** The application of formulas (e.g., physics formulas) is a fundamental ability of humans when solving numerical reasoning problems. Existing numerical reasoning datasets seldom explicitly indicate the formulas employed in reasoning, as their questions rely on implicit commonsense mathematical knowledge. In contrast, in this paper, we introduce FormulaReasoning, a new dataset specifically designed for formula-based numerical reasoning. Each of the 4,751 questions in our dataset requires numerical calculation with external physics formulas, making it a more challenging benchmark for evaluating large language models (LLMs). We offer normalized fine-grained annotations for the questions, available in English and Chinese, including formula structures, parameter names, symbols, numerical values, and units, derived from extensive manual effort with LLM assistance for guaranteed quality. We also provide a consolidated formula database to serve as an external knowledge base accompanying the dataset. We employ FormulaReasoning to evaluate LLMs with 7B to over 100B parameters, and explore retrieval-augmented generation with the formula database. Our evaluation also covers supervised methods that break down the reasoning process into formula generation, parameter extraction, and numerical calculation, as well as direct preference optimization methods based on derived preference data.
>
---
#### [replaced 077] Feedback-Driven Vision-Language Alignment with Minimal Human Supervision
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.04568v2](http://arxiv.org/pdf/2501.04568v2)**

> **作者:** Giorgio Giannone; Ruoteng Li; Qianli Feng; Evgeny Perevodchikov; Rui Chen; Aleix Martinez
>
> **备注:** Preprint
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable potential in integrating visual and linguistic information, but their performance is often constrained by the need for extensive, high-quality image-text training data. Curation of these image-text pairs is both time-consuming and computationally expensive. To address this challenge, we introduce SVP (Sampling-based Visual Projection), a novel framework that enhances vision-language alignment without relying on manually curated text-image pairs or preference annotation. SVP leverages a small set of manually selected images, self-captioning and a pre-trained grounding model as a feedback mechanism to elicit latent information in VLMs. We evaluate our approach across six key areas: captioning, referring, visual question answering, multitasking, hallucination control, and object recall. Results demonstrate significant improvements, including a 14 % average improvement in captioning tasks, up to 12 % increase in object recall, and significantly reduced hallucinations, while maintaining question-answering capabilities. Using SVP, a small VLM achieves hallucination reductions similar to a model five times larger, while a VLM with initially poor referring capabilities more than doubles its performance, approaching parity with a model twice its size.
>
---
#### [replaced 078] MaintainCoder: Maintainable Code Generation Under Dynamic Requirements
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24260v2](http://arxiv.org/pdf/2503.24260v2)**

> **作者:** Zhengren Wang; Rui Ling; Chufan Wang; Yongan Yu; Sizhe Wang; Zhiyu Li; Feiyu Xiong; Wentao Zhang
>
> **备注:** https://github.com/IAAR-Shanghai/MaintainCoder
>
> **摘要:** Modern code generation has made significant strides in functional correctness and execution efficiency. However, these systems often overlook a critical dimension in real-world software development: \textit{maintainability}. To handle dynamic requirements with minimal rework, we propose \textbf{MaintainCoder} as a pioneering solution. It integrates the Waterfall model, design patterns, and multi-agent collaboration to systematically enhance cohesion, reduce coupling, achieving clear responsibility boundaries and better maintainability. We also introduce \textbf{MaintainBench}, a benchmark comprising requirement changes and novel dynamic metrics on maintenance efforts. Experiments demonstrate that existing code generation methods struggle to meet maintainability standards when requirements evolve. In contrast, MaintainCoder improves dynamic maintainability metrics by more than 60\% with even higher correctness of initial codes. Furthermore, while static metrics fail to accurately reflect maintainability and even contradict each other, our proposed dynamic metrics exhibit high consistency. Our work not only provides the foundation for maintainable code generation, but also highlights the need for more realistic and comprehensive code generation research.
>
---
#### [replaced 079] MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.07076v5](http://arxiv.org/pdf/2410.07076v5)**

> **作者:** Zonglin Yang; Wanhao Liu; Ben Gao; Tong Xie; Yuqiang Li; Wanli Ouyang; Soujanya Poria; Erik Cambria; Dongzhan Zhou
>
> **备注:** Accepted by ICLR 2025
>
> **摘要:** Scientific discovery plays a pivotal role in advancing human society, and recent progress in large language models (LLMs) suggests their potential to accelerate this process. However, it remains unclear whether LLMs can autonomously generate novel and valid hypotheses in chemistry. In this work, we investigate whether LLMs can discover high-quality chemistry hypotheses given only a research background-comprising a question and/or a survey-without restriction on the domain of the question. We begin with the observation that hypothesis discovery is a seemingly intractable task. To address this, we propose a formal mathematical decomposition grounded in a fundamental assumption: that most chemistry hypotheses can be composed from a research background and a set of inspirations. This decomposition leads to three practical subtasks-retrieving inspirations, composing hypotheses with inspirations, and ranking hypotheses - which together constitute a sufficient set of subtasks for the overall scientific discovery task. We further develop an agentic LLM framework, MOOSE-Chem, that is a direct implementation of this mathematical decomposition. To evaluate this framework, we construct a benchmark of 51 high-impact chemistry papers published and online after January 2024, each manually annotated by PhD chemists with background, inspirations, and hypothesis. The framework is able to rediscover many hypotheses with high similarity to the groundtruth, successfully capturing the core innovations-while ensuring no data contamination since it uses an LLM with knowledge cutoff date prior to 2024. Finally, based on LLM's surprisingly high accuracy on inspiration retrieval, a task with inherently out-of-distribution nature, we propose a bold assumption: that LLMs may already encode latent scientific knowledge associations not yet recognized by humans.
>
---
#### [replaced 080] Concept-Level Explainability for Auditing & Steering LLM Responses
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07610v2](http://arxiv.org/pdf/2505.07610v2)**

> **作者:** Kenza Amara; Rita Sevastjanova; Mennatallah El-Assady
>
> **备注:** 9 pages, 7 figures, Submission to Neurips 2025
>
> **摘要:** As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.
>
---
#### [replaced 081] Superhuman performance of a large language model on the reasoning tasks of a physician
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10849v2](http://arxiv.org/pdf/2412.10849v2)**

> **作者:** Peter G. Brodeur; Thomas A. Buckley; Zahir Kanjee; Ethan Goh; Evelyn Bin Ling; Priyank Jain; Stephanie Cabral; Raja-Elie Abdulnour; Adrian D. Haimovich; Jason A. Freed; Andrew Olson; Daniel J. Morgan; Jason Hom; Robert Gallo; Liam G. McCoy; Haadi Mombini; Christopher Lucas; Misha Fotoohi; Matthew Gwiazdon; Daniele Restifo; Daniel Restrepo; Eric Horvitz; Jonathan Chen; Arjun K. Manrai; Adam Rodman
>
> **摘要:** A seminal paper published by Ledley and Lusted in 1959 introduced complex clinical diagnostic reasoning cases as the gold standard for the evaluation of expert medical computing systems, a standard that has held ever since. Here, we report the results of a physician evaluation of a large language model (LLM) on challenging clinical cases against a baseline of hundreds of physicians. We conduct five experiments to measure clinical reasoning across differential diagnosis generation, display of diagnostic reasoning, triage differential diagnosis, probabilistic reasoning, and management reasoning, all adjudicated by physician experts with validated psychometrics. We then report a real-world study comparing human expert and AI second opinions in randomly-selected patients in the emergency room of a major tertiary academic medical center in Boston, MA. We compared LLMs and board-certified physicians at three predefined diagnostic touchpoints: triage in the emergency room, initial evaluation by a physician, and admission to the hospital or intensive care unit. In all experiments--both vignettes and emergency room second opinions--the LLM displayed superhuman diagnostic and reasoning abilities, as well as continued improvement from prior generations of AI clinical decision support. Our study suggests that LLMs have achieved superhuman performance on general medical diagnostic and management reasoning, fulfilling the vision put forth by Ledley and Lusted, and motivating the urgent need for prospective trials.
>
---
#### [replaced 082] Phare: A Safety Probe for Large Language Models
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.11365v2](http://arxiv.org/pdf/2505.11365v2)**

> **作者:** Pierre Le Jeune; Benoît Malézieux; Weixuan Xiao; Matteo Dora
>
> **摘要:** Ensuring the safety of large language models (LLMs) is critical for responsible deployment, yet existing evaluations often prioritize performance over identifying failure modes. We introduce Phare, a multilingual diagnostic framework to probe and evaluate LLM behavior across three critical dimensions: hallucination and reliability, social biases, and harmful content generation. Our evaluation of 17 state-of-the-art LLMs reveals patterns of systematic vulnerabilities across all safety dimensions, including sycophancy, prompt sensitivity, and stereotype reproduction. By highlighting these specific failure modes rather than simply ranking models, Phare provides researchers and practitioners with actionable insights to build more robust, aligned, and trustworthy language systems.
>
---
#### [replaced 083] MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10526v2](http://arxiv.org/pdf/2505.10526v2)**

> **作者:** Mugilan Ganesan; Shane Segal; Ankur Aggarwal; Nish Sinnadurai; Sean Lie; Vithursan Thangarasa
>
> **备注:** Main paper: 11 pages, 4 figures, 3 tables. Supplementary: 1 page
>
> **摘要:** Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM's vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x on visually-grounded tasks. MASSV provides a scalable, architecture-compatible method for accelerating both current and future VLMs.
>
---
#### [replaced 084] Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era
- **分类: cs.LG; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2403.08946v2](http://arxiv.org/pdf/2403.08946v2)**

> **作者:** Xuansheng Wu; Haiyan Zhao; Yaochen Zhu; Yucheng Shi; Fan Yang; Lijie Hu; Tianming Liu; Xiaoming Zhai; Wenlin Yao; Jundong Li; Mengnan Du; Ninghao Liu
>
> **备注:** 43 pages, 6 figures, including the latest works published in 2024-2025
>
> **摘要:** Explainable AI (XAI) refers to techniques that provide human-understandable insights into the workings of AI models. Recently, the focus of XAI is being extended toward explaining Large Language Models (LLMs). This extension calls for a significant transformation in the XAI methodologies for two reasons. First, many existing XAI methods cannot be directly applied to LLMs due to their complexity and advanced capabilities. Second, as LLMs are increasingly deployed in diverse applications, the role of XAI shifts from merely opening the ``black box'' to actively enhancing the productivity and applicability of LLMs in real-world settings. Meanwhile, the conversation and generation abilities of LLMs can reciprocally enhance XAI. Therefore, in this paper, we introduce Usable XAI in the context of LLMs by analyzing (1) how XAI can explain and improve LLM-based AI systems and (2) how XAI techniques can be improved by using LLMs. We introduce 10 strategies, introducing the key techniques for each and discussing their associated challenges. We also provide case studies to demonstrate how to obtain and leverage explanations. The code used in this paper can be found at: https://github.com/JacksonWuxs/UsableXAI_LLM.
>
---
#### [replaced 085] Can We Verify Step by Step for Incorrect Answer Detection?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.10528v4](http://arxiv.org/pdf/2402.10528v4)**

> **作者:** Xin Xu; Shizhe Diao; Can Yang; Yang Wang
>
> **备注:** accepted to IJCAI 2025
>
> **摘要:** Chain-of-Thought (CoT) prompting has marked a significant advancement in enhancing the reasoning capabilities of large language models (LLMs). Previous studies have developed various extensions of CoT, which focus primarily on enhancing end-task performance. In addition, there has been research on assessing the quality of reasoning chains in CoT. This raises an intriguing question: Is it possible to predict the accuracy of LLM outputs by scrutinizing the reasoning chains they generate? To answer this research question, we introduce a benchmark, R2PE, designed specifically to explore the relationship between reasoning chains and performance in various reasoning tasks spanning five different domains. This benchmark aims to measure the falsehood of the final output of LLMs based on the reasoning steps. To make full use of information in multiple reasoning chains, we propose the process discernibility score (PDS) framework that beats the answer-checking baseline by a large margin. Concretely, this resulted in an average of $5.1\%$ increase in the F1 score and $2.97\%$ improvement in AUC-PR across all 45 subsets within R2PE. We further demonstrate our PDS's efficacy in advancing open-domain QA accuracy.
>
---
#### [replaced 086] LSR-MCTS: Alleviating Long Range Dependency in Code Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07433v3](http://arxiv.org/pdf/2504.07433v3)**

> **作者:** Tingwei Lu; Yangning Li; Liyuan Wang; Binghuai Lin; Jiwei Tang; Qingsong Lv; Wanshi Xu; Hai-Tao Zheng; Yinghui Li; Xin Su; Zifei Shan
>
> **摘要:** The emergence of large language models (LLMs) has significantly promoted the development of code generation task, sparking a surge in pertinent literature. Current research is hindered by redundant generation results and a tendency to overfit local patterns in the short term. Although existing studies attempt to alleviate the issue by adopting a multi-token prediction strategy, there remains limited focus on choosing the appropriate processing length for generations. By analyzing the attention between tokens during the generation process of LLMs, it can be observed that the high spikes of the attention scores typically appear at the end of lines. This insight suggests that it is reasonable to treat each line of code as a fundamental processing unit and generate them sequentially. Inspired by this, we propose the \textbf{LSR-MCTS} algorithm, which leverages MCTS to determine the code line-by-line and select the optimal path. Further, we integrate a self-refine mechanism at each node to enhance diversity and generate higher-quality programs through error correction. Extensive experiments and comprehensive analyses on three public coding benchmarks demonstrate that our method outperforms the state-of-the-art performance approaches.
>
---
#### [replaced 087] Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection
- **分类: cs.LG; cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2410.02647v2](http://arxiv.org/pdf/2410.02647v2)**

> **作者:** Song Li; Yang Tan; Song Ke; Liang Hong; Bingxin Zhou
>
> **备注:** 20 pages, 17 tables, 6 figures
>
> **摘要:** Immunogenicity prediction is a central topic in reverse vaccinology for finding candidate vaccines that can trigger protective immune responses. Existing approaches typically rely on highly compressed features and simple model architectures, leading to limited prediction accuracy and poor generalizability. To address these challenges, we introduce VenusVaccine, a novel deep learning solution with a dual attention mechanism that integrates pre-trained latent vector representations of protein sequences and structures. We also compile the most comprehensive immunogenicity dataset to date, encompassing over 7000 antigen sequences, structures, and immunogenicity labels from bacteria, virus, and tumor. Extensive experiments demonstrate that VenusVaccine outperforms existing methods across a wide range of evaluation metrics. Furthermore, we establish a post-hoc validation protocol to assess the practical significance of deep learning models in tackling vaccine design challenges. Our work provides an effective tool for vaccine design and sets valuable benchmarks for future research. The implementation is at https://github.com/songleee/VenusVaccine.
>
---
#### [replaced 088] DeLoRA: Decoupling Angles and Strength in Low-rank Adaptation
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18225v2](http://arxiv.org/pdf/2503.18225v2)**

> **作者:** Massimo Bini; Leander Girrbach; Zeynep Akata
>
> **备注:** ICLR 2025
>
> **摘要:** Parameter-Efficient FineTuning (PEFT) methods have recently gained significant popularity thanks to the widespread availability of large-scale pretrained models. These methods allow for quick adaptation to downstream tasks with minimal computational cost. However, popular finetuning methods such as LoRA exhibit limited robustness when it comes to hyperparameter choices or extended training regimes, preventing optimal out-of-the-box performance. In contrast, bounded approaches, such as ETHER, provide greater robustness but are limited to extremely low-rank adaptations and fixed-strength transformations, reducing their adaptation expressive power. In this work, we propose Decoupled Low-rank Adaptation (DeLoRA), a novel finetuning method that normalizes and scales learnable low-rank matrices. By bounding the distance of the transformation, DeLoRA effectively decouples the angular learning from the adaptation strength, enhancing robustness without compromising performance. Through evaluations on subject-driven image generation, natural language understanding, and instruction tuning, we show that DeLoRA matches or surpasses performance of competing PEFT methods, while exhibiting stronger robustness. Code is available at https://github.com/ExplainableML/DeLoRA.
>
---
#### [replaced 089] Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10792v2](http://arxiv.org/pdf/2505.10792v2)**

> **作者:** Zhan Peng Lee; Andre Lin; Calvin Tan
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to improve factuality in large language models (LLMs) by grounding their outputs in retrieved documents. However, ensuring perfect retrieval of relevant information remains challenging, and when irrelevant content is passed downstream to an LLM, it can lead to hallucinations. In this work, we propose Finetune-RAG, a simple and effective fine-tuning approach that features the first-of-its-kind RAG training dataset constructed to mimic real-world imperfections. Experimental results show that Finetune-RAG improves factual accuracy by 21.2% over the base model. We also propose Bench-RAG, an LLM-as-a-judge evaluation pipeline that stress tests models under realistic imperfect retrieval scenarios. Our codebase and dataset are fully open sourced for community use.
>
---
#### [replaced 090] PlanFitting: Personalized Exercise Planning with Large Language Model-driven Conversational Agent
- **分类: cs.HC; cs.AI; cs.CL; H.5.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2309.12555v2](http://arxiv.org/pdf/2309.12555v2)**

> **作者:** Donghoon Shin; Gary Hsieh; Young-Ho Kim
>
> **备注:** 17 pages including reference. Accepted to ACM CUI 2025
>
> **摘要:** Creating personalized and actionable exercise plans often requires iteration with experts, which can be costly and inaccessible to many individuals. This work explores the capabilities of Large Language Models (LLMs) in addressing these challenges. We present PlanFitting, an LLM-driven conversational agent that assists users in creating and refining personalized weekly exercise plans. By engaging users in free-form conversations, PlanFitting helps elicit users' goals, availabilities, and potential obstacles, and enables individuals to generate personalized exercise plans aligned with established exercise guidelines. Our study -- involving a user study, intrinsic evaluation, and expert evaluation -- demonstrated PlanFitting's ability to guide users to create tailored, actionable, and evidence-based plans. We discuss future design opportunities for LLM-driven conversational agents to create plans that better comply with exercise principles and accommodate personal constraints.
>
---
#### [replaced 091] Creating General User Models from Computer Use
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10831v2](http://arxiv.org/pdf/2505.10831v2)**

> **作者:** Omar Shaikh; Shardul Sapkota; Shan Rizvi; Eric Horvitz; Joon Sung Park; Diyi Yang; Michael S. Bernstein
>
> **备注:** 22 pages, 6 figures, 1 table; see https://generalusermodels.github.io/
>
> **摘要:** Human-computer interaction has long imagined technology that understands us-from our preferences and habits, to the timing and purpose of our everyday actions. Yet current user models remain fragmented, narrowly tailored to specific apps, and incapable of the flexible reasoning required to fulfill these visions. This paper presents an architecture for a general user model (GUM) that learns about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture user knowledge and preferences. GUMs can infer that a user is preparing for a wedding they're attending from messages with a friend. Or recognize that a user is struggling with a collaborator's feedback on a draft by observing multiple stalled edits and a switch to reading related work. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions. To illustrate the breadth of applications that GUMs enable, we demonstrate how they augment chat-based assistants with context, manage OS notifications to selectively surface important information, and enable interactive agents that adapt to preferences across apps. We also instantiate proactive assistants (GUMBOs) that discover and execute useful suggestions on a user's behalf using their GUM. In our evaluations, we find that GUMs make calibrated and accurate inferences about users, and that assistants built on GUMs proactively identify and perform actions that users wouldn't think to request explicitly. Altogether, GUMs introduce methods that leverage multimodal models to understand unstructured context, enabling long-standing visions of HCI and entirely new interactive systems that anticipate user needs.
>
---
#### [replaced 092] Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22388v2](http://arxiv.org/pdf/2503.22388v2)**

> **作者:** Zhiyu Yang; Shuo Wang; Yukun Yan; Yang Deng
>
> **摘要:** LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the future. DSDBench is publicly available at github.com/KevinCL16/DSDBench.
>
---
#### [replaced 093] LLMs are not Zero-Shot Reasoners for Biomedical Information Extraction
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.12249v2](http://arxiv.org/pdf/2408.12249v2)**

> **作者:** Aishik Nagar; Viktor Schlegel; Thanh-Tung Nguyen; Hao Li; Yuping Wu; Kuluhan Binici; Stefan Winkler
>
> **备注:** 15 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly adopted for applications in healthcare, reaching the performance of domain experts on tasks such as question answering and document summarisation. Despite their success on these tasks, it is unclear how well LLMs perform on tasks that are traditionally pursued in the biomedical domain, such as structured information extraction. To bridge this gap, in this paper, we systematically benchmark LLM performance in Medical Classification and Named Entity Recognition (NER) tasks. We aim to disentangle the contribution of different factors to the performance, particularly the impact of LLMs' task knowledge and reasoning capabilities, their (parametric) domain knowledge, and addition of external knowledge. To this end, we evaluate various open LLMs - including BioMistral and Llama-2 models - on a diverse set of biomedical datasets, using standard prompting, Chain of-Thought (CoT) and Self Consistency based reasoning as well as Retrieval-Augmented Generation (RAG) with PubMed and Wikipedia corpora. Counter intuitively, our results reveal that standard prompting consistently outperforms more complex techniques across both tasks, laying bare the limitations in the current application of CoT, self-consistency and RAG in the biomedical domain. Our findings suggest that advanced prompting methods developed for knowledge- or reasoning-intensive tasks, such as CoT or RAG, are not easily portable to biomedical tasks where precise structured outputs are required. This highlights the need for more effective integration of external knowledge and reasoning mechanisms in LLMs to enhance their performance in real-world biomedical applications.
>
---
#### [replaced 094] "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.02718v3](http://arxiv.org/pdf/2409.02718v3)**

> **作者:** Zi Liang; Qingqing Ye; Yanyun Wang; Sen Zhang; Yaxin Xiao; Ronghua Li; Jianliang Xu; Haibo Hu
>
> **备注:** To appear at ACL 25 main conference
>
> **摘要:** Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA .
>
---
#### [replaced 095] LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2410.02707v4](http://arxiv.org/pdf/2410.02707v4)**

> **作者:** Hadas Orgad; Michael Toker; Zorik Gekhman; Roi Reichart; Idan Szpektor; Hadas Kotek; Yonatan Belinkov
>
> **摘要:** Large language models (LLMs) often produce errors, including factual inaccuracies, biases, and reasoning failures, collectively referred to as "hallucinations". Recent studies have demonstrated that LLMs' internal states encode information regarding the truthfulness of their outputs, and that this information can be utilized to detect errors. In this work, we show that the internal representations of LLMs encode much more information about truthfulness than previously recognized. We first discover that the truthfulness information is concentrated in specific tokens, and leveraging this property significantly enhances error detection performance. Yet, we show that such error detectors fail to generalize across datasets, implying that -- contrary to prior claims -- truthfulness encoding is not universal but rather multifaceted. Next, we show that internal representations can also be used for predicting the types of errors the model is likely to make, facilitating the development of tailored mitigation strategies. Lastly, we reveal a discrepancy between LLMs' internal encoding and external behavior: they may encode the correct answer, yet consistently generate an incorrect one. Taken together, these insights deepen our understanding of LLM errors from the model's internal perspective, which can guide future research on enhancing error analysis and mitigation.
>
---
#### [replaced 096] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21380v2](http://arxiv.org/pdf/2503.21380v2)**

> **作者:** Haoxiang Sun; Yingqian Min; Zhipeng Chen; Wayne Xin Zhao; Lei Fang; Zheng Liu; Zhongyuan Wang; Ji-Rong Wen
>
> **备注:** Technical Report on Slow Thinking with LLMs: Evaluation Benchmark
>
> **摘要:** In recent years, the rapid development of large reasoning models has resulted in the saturation of existing benchmarks for evaluating mathematical reasoning, highlighting the urgent need for more challenging and rigorous evaluation frameworks. To address this gap, we introduce OlymMATH, a novel Olympiad-level mathematical benchmark, designed to rigorously test the complex reasoning capabilities of LLMs. OlymMATH features 200 meticulously curated problems, each manually verified and available in parallel English and Chinese versions. The problems are systematically organized into two distinct difficulty tiers: (1) AIME-level problems (easy) that establish a baseline for mathematical reasoning assessment, and (2) significantly more challenging problems (hard) designed to push the boundaries of current state-of-the-art models. In our benchmark, these problems span four core mathematical fields, each including a verifiable numerical solution to enable objective, rule-based evaluation. Empirical results underscore the significant challenge presented by OlymMATH, with state-of-the-art models including DeepSeek-R1, OpenAI's o3-mini and Gemini 2.5 Pro Exp demonstrating notably limited accuracy on the hard subset. Furthermore, the benchmark facilitates comprehensive bilingual assessment of mathematical reasoning abilities-a critical dimension that remains largely unaddressed in mainstream mathematical reasoning benchmarks. We release the benchmark, evaluation code, detailed results and a data visualization tool at https://github.com/RUCAIBox/OlymMATH.
>
---
#### [replaced 097] A Bounding Box is Worth One Token: Interleaving Layout and Text in a Large Language Model for Document Understanding
- **分类: cs.CL; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2407.01976v3](http://arxiv.org/pdf/2407.01976v3)**

> **作者:** Jinghui Lu; Haiyang Yu; Yanjie Wang; Yongjie Ye; Jingqun Tang; Ziwei Yang; Binghong Wu; Qi Liu; Hao Feng; Han Wang; Hao Liu; Can Huang
>
> **备注:** Accept to ACL2025 Findings
>
> **摘要:** Recently, many studies have demonstrated that exclusively incorporating OCR-derived text and spatial layouts with large language models (LLMs) can be highly effective for document understanding tasks. However, existing methods that integrate spatial layouts with text have limitations, such as producing overly long text sequences or failing to fully leverage the autoregressive traits of LLMs. In this work, we introduce Interleaving Layout and Text in a Large Language Model (LayTextLLM)} for document understanding. LayTextLLM projects each bounding box to a single embedding and interleaves it with text, efficiently avoiding long sequence issues while leveraging autoregressive traits of LLMs. LayTextLLM not only streamlines the interaction of layout and textual data but also shows enhanced performance in KIE and VQA. Comprehensive benchmark evaluations reveal significant improvements of LayTextLLM, with a 15.2% increase on KIE tasks and 10.7% on VQA tasks compared to previous SOTA OCR-based LLMs. All resources are available at https://github.com/LayTextLLM/LayTextLLM.
>
---
#### [replaced 098] LLMScan: Causal Scan for LLM Misbehavior Detection
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.16638v3](http://arxiv.org/pdf/2410.16638v3)**

> **作者:** Mengdi Zhang; Kai Kiat Goh; Peixin Zhang; Jun Sun; Rose Lin Xin; Hongyu Zhang
>
> **摘要:** Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution. LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.
>
---
#### [replaced 099] Joint Localization and Activation Editing for Low-Resource Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01179v3](http://arxiv.org/pdf/2502.01179v3)**

> **作者:** Wen Lai; Alexander Fraser; Ivan Titov
>
> **备注:** Accepted by ICML 2025. The code is released at https://github.com/wenlai-lavine/jola
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, are commonly used to adapt LLMs. However, the effectiveness of standard PEFT methods is limited in low-resource scenarios with only a few hundred examples. Recent advances in interpretability research have inspired the emergence of activation editing (or steering) techniques, which modify the activations of specific model components. These methods, due to their extremely small parameter counts, show promise for small datasets. However, their performance is highly dependent on identifying the correct modules to edit and often lacks stability across different datasets. In this paper, we propose Joint Localization and Activation Editing (JoLA), a method that jointly learns (1) which heads in the Transformer to edit (2) whether the intervention should be additive, multiplicative, or both and (3) the intervention parameters themselves - the vectors applied as additive offsets or multiplicative scalings to the head output. Through evaluations on three benchmarks spanning commonsense reasoning, natural language understanding, and natural language generation, we demonstrate that JoLA consistently outperforms existing methods. The code for the method is released at https://github.com/wenlai-lavine/jola.
>
---
#### [replaced 100] Scaling Embedding Layers in Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01637v2](http://arxiv.org/pdf/2502.01637v2)**

> **作者:** Da Yu; Edith Cohen; Badih Ghazi; Yangsibo Huang; Pritish Kamath; Ravi Kumar; Daogao Liu; Chiyuan Zhang
>
> **备注:** Added downstream evaluation results and improved the overall clarity and writing
>
> **摘要:** We propose SCONE ($S$calable, $C$ontextualized, $O$ffloaded, $N$-gram $E$mbedding), a new method for extending input embedding layers to enhance language model performance. To avoid increased decoding costs, SCONE retains the original vocabulary while introducing embeddings for a set of frequent $n$-grams. These embeddings provide contextualized representation for each input token and are learned with a separate model during training. After training, embeddings are precomputed and stored in off-accelerator memory; during inference, querying them has minimal impact on latency due to the low complexity of embedding lookups. SCONE enables two new scaling strategies: increasing the number of $n$-gram embeddings and scaling the model used to learn them, both while maintaining fixed accelerator usage during inference (in terms of FLOPS and memory). We show that scaling both aspects enables a model with 1B accelerator-resident parameters to outperform a 1.9B-parameter baseline across diverse corpora, while using only about half the FLOPS and accelerator memory during inference.
>
---
#### [replaced 101] VersaTune: An Efficient Data Composition Framework for Training Multi-Capability LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.11266v5](http://arxiv.org/pdf/2411.11266v5)**

> **作者:** Keer Lu; Keshi Zhao; Zhuoran Zhang; Zheng Liang; Da Pan; Shusen Zhang; Xin Wu; Guosheng Dong; Bin Cui; Tengjiao Wang; Wentao Zhang
>
> **摘要:** As demonstrated by the proprietary Large Language Models (LLMs) such as GPT and Claude series, LLMs have the potential to achieve remarkable proficiency across a wide range of domains, including law, medicine, finance, science, code, etc., all within a single model. These capabilities are further augmented during the Supervised Fine-Tuning (SFT) phase. Despite their potential, existing work mainly focuses on domain-specific enhancements during fine-tuning, the challenge of which lies in catastrophic forgetting of knowledge across other domains. In this study, we introduce **VersaTune**, a novel data composition framework designed for enhancing LLMs' overall multi-domain capabilities during training. We begin with detecting the distribution of domain-specific knowledge within the base model, followed by the training data composition that aligns with the model's existing knowledge distribution. During the subsequent training process, domain weights are dynamically adjusted based on their learnable potential and forgetting degree. Experimental results indicate that VersaTune is effective in multi-domain fostering, with an improvement of 35.21\% in the overall multi-ability performances compared to uniform domain weights. Furthermore, we find that Qwen-2.5-32B + VersaTune even surpasses frontier models, including GPT-4o, Claude3.5-Sonnet and DeepSeek-V3 by 0.86\%, 4.76\% and 4.60\%. Additionally, in scenarios where flexible expansion of a specific domain is required, VersaTune reduces the performance degradation in other domains by 38.77\%, while preserving the training efficacy of the target domain.
>
---
#### [replaced 102] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06141v2](http://arxiv.org/pdf/2412.06141v2)**

> **作者:** Kangyu Zhu; Peng Xia; Yun Li; Hongtu Zhu; Sheng Wang; Huaxiu Yao
>
> **备注:** ICML 2025
>
> **摘要:** The advancement of Large Vision-Language Models (LVLMs) has propelled their application in the medical field. However, Medical LVLMs (Med-LVLMs) encounter factuality challenges due to modality misalignment, where the models prioritize textual knowledge over visual input, leading to hallucinations that contradict information in medical images. Previous attempts to enhance modality alignment in Med-LVLMs through preference optimization have inadequately mitigated clinical relevance in preference data, making these samples easily distinguishable and reducing alignment effectiveness. To address this challenge, we propose MMedPO, a novel multimodal medical preference optimization approach that considers the clinical relevance of preference samples to enhance Med-LVLM alignment. MMedPO curates multimodal preference data by introducing two types of dispreference: (1) plausible hallucinations injected through target Med-LVLMs or GPT-4o to produce medically inaccurate responses, and (2) lesion region neglect achieved through local lesion-noising, disrupting visual understanding of critical areas. We then calculate clinical relevance for each sample based on scores from multiple Med-LLMs and visual tools, and integrate these scores into the preference optimization process as weights, enabling effective alignment. Our experiments demonstrate that MMedPO significantly enhances factual accuracy in Med-LVLMs, achieving substantial improvements over existing preference optimization methods by averaging 14.2% and 51.7% across the Med-VQA and report generation tasks. Our code are available in https://github.com/aiming-lab/MMedPO.
>
---
#### [replaced 103] Advancing Multi-Party Dialogue Framework with Speaker-ware Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11292v2](http://arxiv.org/pdf/2501.11292v2)**

> **作者:** Zhongtian Hu; Qi He; Ronghan Li; Meng Zhao; Lifang Wang
>
> **摘要:** Multi-party dialogues, common in collaborative scenarios like brainstorming sessions and negotiations, pose significant challenges due to their complexity and diverse speaker roles. Current methods often use graph neural networks to model dialogue context, capturing structural dynamics but heavily relying on annotated graph structures and overlooking individual speaking styles. To address these challenges, we propose CMR, a Contrastive learning-based Multi-party dialogue Response generation framework. CMR employs a two-stage self-supervised contrastive learning framework. First, it captures global differences in speaking styles across individuals. Then, it focuses on intra-conversation comparisons to identify thematic transitions and contextually relevant facts. To the best of our knowledge, this is the first approach that applies contrastive learning in multi-party dialogue generation. Experimental results demonstrate that CMR not only significantly outperforms state-of-the-art models, but also generalizes well to large pre-trained language models, effectively enhancing their capability in handling multi-party conversations.
>
---
#### [replaced 104] A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15585v2](http://arxiv.org/pdf/2504.15585v2)**

> **作者:** Kun Wang; Guibin Zhang; Zhenhong Zhou; Jiahao Wu; Miao Yu; Shiqian Zhao; Chenlong Yin; Jinhu Fu; Yibo Yan; Hanjun Luo; Liang Lin; Zhihao Xu; Haolang Lu; Xinye Cao; Xinyun Zhou; Weifei Jin; Fanci Meng; Junyuan Mao; Yu Wang; Hao Wu; Minghe Wang; Fan Zhang; Junfeng Fang; Wenjie Qu; Yue Liu; Chengwei Liu; Yifan Zhang; Qiankun Li; Chongye Guo; Yalan Qin; Zhaoxin Fan; Yi Ding; Donghai Hong; Jiaming Ji; Yingxin Lai; Zitong Yu; Xinfeng Li; Yifan Jiang; Yanhui Li; Xinyu Deng; Junlin Wu; Dongxia Wang; Yihao Huang; Yufei Guo; Jen-tse Huang; Qiufeng Wang; Wenxuan Wang; Dongrui Liu; Yanwei Yue; Wenke Huang; Guancheng Wan; Heng Chang; Tianlin Li; Yi Yu; Chenghao Li; Jiawei Li; Lei Bai; Jie Zhang; Qing Guo; Jingyi Wang; Tianlong Chen; Joey Tianyi Zhou; Xiaojun Jia; Weisong Sun; Cong Wu; Jing Chen; Xuming Hu; Yiming Li; Xiao Wang; Ningyu Zhang; Luu Anh Tuan; Guowen Xu; Jiaheng Zhang; Tianwei Zhang; Xingjun Ma; Jindong Gu; Xiang Wang; Bo An; Jun Sun; Mohit Bansal; Shirui Pan; Lingjuan Lyu; Yuval Elovici; Bhavya Kailkhura; Yaodong Yang; Hongwei Li; Wenyuan Xu; Yizhou Sun; Wei Wang; Qing Li; Ke Tang; Yu-Gang Jiang; Felix Juefei-Xu; Hui Xiong; Xiaofeng Wang; Dacheng Tao; Philip S. Yu; Qingsong Wen; Yang Liu
>
> **摘要:** The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
>
---
#### [replaced 105] UC-MOA: Utility-Conditioned Multi-Objective Alignment for Distributional Pareto-Optimality
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10669v2](http://arxiv.org/pdf/2503.10669v2)**

> **作者:** Zelei Cheng; Xin-Qiang Cai; Yuting Tang; Pushi Zhang; Boming Yang; Masashi Sugiyama; Xinyu Xing
>
> **备注:** Language Modeling, Machine Learning for NLP, Distributional Pareto-Optimal
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs) with human values. However, existing approaches struggle to capture the multi-dimensional, distributional nuances of human preferences. Methods such as RiC that directly inject raw reward values into prompts face significant numerical sensitivity issues--for instance, LLMs may fail to distinguish between 9.11 and 9.8--while alternatives like MORLHF, Rewarded Soups, and MODPO incur high computational costs by training multiple models. In this work, we introduce Utility-Conditioned Multi-Objective Alignment (UC-MOA), a novel framework that overcomes these limitations. Our approach leverages a diverse set of strictly increasing, non-linear utility functions to transform user-specified preferences into symbolic tokens, which are then used to condition a single LLM. This design not only mitigates numerical reasoning challenges but also substantially reduces training overhead, yielding models that achieve superior Pareto fronts and robust alignment across complex reward dimensions.
>
---
#### [replaced 106] OntoURL: A Benchmark for Evaluating Large Language Models on Symbolic Ontological Understanding, Reasoning and Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11031v2](http://arxiv.org/pdf/2505.11031v2)**

> **作者:** Xiao Zhang; Huiyuan Lai; Qianru Meng; Johan Bos
>
> **备注:** Paper submitted to NeurIPS 2025 dataset and benchmark track
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across a range of natural language processing tasks, yet their ability to process structured symbolic knowledge remains underexplored. To address this gap, we propose a taxonomy of LLMs' ontological capabilities and introduce OntoURL, the first comprehensive benchmark designed to systematically evaluate LLMs' proficiency in handling ontologies -- formal, symbolic representations of domain knowledge through concepts, relationships, and instances. Based on the proposed taxonomy, OntoURL systematically assesses three dimensions: understanding, reasoning, and learning through 15 distinct tasks comprising 58,981 questions derived from 40 ontologies across 8 domains. Experiments with 20 open-source LLMs reveal significant performance differences across models, tasks, and domains, with current LLMs showing proficiency in understanding ontological knowledge but substantial weaknesses in reasoning and learning tasks. These findings highlight fundamental limitations in LLMs' capability to process symbolic knowledge and establish OntoURL as a critical benchmark for advancing the integration of LLMs with formal knowledge representations.
>
---
#### [replaced 107] Theoretical Proof that Auto-regressive Language Models Collapse when Real-world Data is a Finite Set
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14872v3](http://arxiv.org/pdf/2412.14872v3)**

> **作者:** Lecheng Wang; Xianjie Shi; Ge Li; Jia Li; Xuanming Zhang; Yihong Dong; Wenpin Jiao; Hong Mei
>
> **备注:** 20 pages, 3 figures
>
> **摘要:** Auto-regressive language models (LMs) have been widely used to generate data in data-scarce domains to train new LMs, compensating for the scarcity of real-world data. Previous work experimentally found that LMs collapse when trained on recursively generated data. This paper presents a theoretical proof: once a corpus (such as a subset of the World Wide Web) begins to incorporate generated data and no new real-world data is added to the corpus, then no matter how small the amount of data each LM generates and contributes to the corpus, LM collapse is inevitable after sufficient time. This finding suggests that attempts to mitigate collapse by limiting the quantity of synthetic data in the corpus are fundamentally insufficient. Instead, avoiding collapse hinges on ensuring the quality of synthetic data.
>
---
#### [replaced 108] Explaining Context Length Scaling and Bounds for Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01481v3](http://arxiv.org/pdf/2502.01481v3)**

> **作者:** Jingzhe Shi; Qinwei Ma; Hongyi Liu; Hang Zhao; Jeng-Neng Hwang; Lei Li
>
> **备注:** 30 pages, 13 figures, 2 tables
>
> **摘要:** Long Context Language Models have drawn great attention in the past few years. There has been work discussing the impact of long context on Language Model performance: some find that long irrelevant context could harm performance, while some experimentally summarize loss reduction by relevant long context as Scaling Laws. This calls for a more thorough understanding on how long context impacts Language Modeling. In this work, we (1) propose a clean and effective theoretical framework for explaining the impact of context length on Language Modeling, from an Intrinsic Space perspective; and (2) conduct experiments on natural language and synthetic data, validating our proposed theoretical assumptions and deductions. Our theoretical framework can provide practical insights such as establishing that training dataset size dictates an optimal context length and bounds context length scaling for certain cases. We hope our work may inspire new long context Language Models, as well as future work studying Physics for Language Models. Code for our experiments is available at: https://github.com/JingzheShi/NLPCtlScalingAndBounds.
>
---
#### [replaced 109] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.02444v4](http://arxiv.org/pdf/2502.02444v4)**

> **作者:** Haoran Ye; Tianze Zhang; Yuhang Xie; Liyuan Zhang; Yuanyi Ren; Xin Zhang; Guojie Song
>
> **备注:** Accepted at ACL 2024 Main
>
> **摘要:** Values are core drivers of individual and collective perception, cognition, and behavior. Value systems, such as Schwartz's Theory of Basic Human Values, delineate the hierarchy and interplay among these values, enabling cross-disciplinary investigations into decision-making and societal dynamics. Recently, the rise of Large Language Models (LLMs) has raised concerns regarding their elusive intrinsic values. Despite growing efforts in evaluating, understanding, and aligning LLM values, a psychologically grounded LLM value system remains underexplored. This study addresses the gap by introducing the Generative Psycho-Lexical Approach (GPLA), a scalable, adaptable, and theoretically informed method for constructing value systems. Leveraging GPLA, we propose a psychologically grounded five-factor value system tailored for LLMs. For systematic validation, we present three benchmarking tasks that integrate psychological principles with cutting-edge AI priorities. Our results reveal that the proposed value system meets standard psychological criteria, better captures LLM values, improves LLM safety prediction, and enhances LLM alignment, when compared to the canonical Schwartz's values.
>
---
#### [replaced 110] Reformulation for Pretraining Data Augmentation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04235v2](http://arxiv.org/pdf/2502.04235v2)**

> **作者:** Xintong Hao; Ruijie Zhu; Ge Zhang; Ke Shen; Chenggang Li
>
> **备注:** Dataset released https://huggingface.co/datasets/ByteDance-Seed/mga-fineweb-edu
>
> **摘要:** Despite the impressive capabilities of large language models across various tasks, their continued scaling is severely hampered not only by data scarcity but also by the performance degradation associated with excessive data repetition during training. To overcome this critical bottleneck, we propose the Massive Genre-Audience(MGA) reformulation method, a lightweight and scalable data augmentation technique inspired by synthetic data methodologies. MGA systematically reformulates existing corpora into diverse, contextually-rich variations to mitigate the negative effects of repetition, and we introduce this approach along with the resulting 770 billion token MGACorpus in this work. We experimentally validate its core benefit by demonstrating superior performance against data repetition and upsampling in scaling scenarios (up to 13B parameters). Furthermore, comprehensive analysis investigates the role of prompt engineering in generation quality and reveals nuances in evaluating model capabilities using standard loss metrics. Our work shows that MGA provides a reliable pathway to substantially augment training datasets, effectively alleviating repetition bottlenecks and enabling more efficient scaling of large language models.
>
---
#### [replaced 111] Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10597v2](http://arxiv.org/pdf/2505.10597v2)**

> **作者:** Jiazheng Zhang; Wenqing Jing; Zizhuo Zhang; Zhiheng Xi; Shihan Dou; Rongxiang Weng; Jiahuan Li; Jingang Wang; Mingxu Chai; Shibo Hong; Tao Gui; Qi Zhang
>
> **摘要:** Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human values. However, noisy preferences in human feedback can lead to reward misgeneralization - a phenomenon where reward models learn spurious correlations or overfit to noisy preferences, which poses important challenges to the generalization of RMs. This paper systematically analyzes the characteristics of preference pairs and aims to identify how noisy preferences differ from human-aligned preferences in reward modeling. Our analysis reveals that noisy preferences are difficult for RMs to fit, as they cause sharp training fluctuations and irregular gradient updates. These distinctive dynamics suggest the feasibility of identifying and excluding such noisy preferences. Empirical studies demonstrate that policy LLM optimized with a reward model trained on the full preference dataset, which includes substantial noise, performs worse than the one trained on a subset of exclusively high quality preferences. To address this challenge, we propose an online Collaborative Reward Modeling (CRM) framework to achieve robust preference learning through peer review and curriculum learning. In particular, CRM maintains two RMs that collaboratively filter potential noisy preferences by peer-reviewing each other's data selections. Curriculum learning synchronizes the capabilities of two models, mitigating excessive disparities to promote the utility of peer review. Extensive experiments demonstrate that CRM significantly enhances RM generalization, with up to 9.94 points improvement on RewardBench under an extreme 40\% noise. Moreover, CRM can seamlessly extend to implicit-reward alignment methods, offering a robust and versatile alignment strategy.
>
---
#### [replaced 112] Physics of Language Models: Part 1, Learning Hierarchical Language Structures
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2305.13673v4](http://arxiv.org/pdf/2305.13673v4)**

> **作者:** Zeyuan Allen-Zhu; Yuanzhi Li
>
> **备注:** V2 polishes writing and adds Appendix G; V3 polishes writing and changes the title; V4 improves writing and adds Appendix H (more uniform attention results)
>
> **摘要:** Transformer-based language models are effective but complex, and understanding their inner workings and reasoning mechanisms is a significant challenge. Previous research has primarily explored how these models handle simple tasks like name copying or selection, and we extend this by investigating how these models perform recursive language structure reasoning defined by context-free grammars (CFGs). We introduce a family of synthetic CFGs that produce hierarchical rules, capable of generating lengthy sentences (e.g., hundreds of tokens) that are locally ambiguous and require dynamic programming to parse. Despite this complexity, we demonstrate that generative models like GPT can accurately learn and reason over CFG-defined hierarchies and generate sentences based on it. We explore the model's internals, revealing that its hidden states precisely capture the structure of CFGs, and its attention patterns resemble the information passing in a dynamic programming algorithm. This paper also presents several corollaries, including showing why absolute positional embeddings is inferior to relative and rotary embeddings; uniform attention alone is surprisingly effective (motivating our follow-up work on Canon layers); encoder-only models (e.g., BERT, DeBERTa) struggle with deep structure reasoning on CFGs compared to autoregressive models (e.g., GPT); and injecting structural or syntactic noise into pretraining data markedly improves robustness to corrupted language prompts.
>
---
#### [replaced 113] Decoding Time Series with LLMs: A Multi-Agent Framework for Cross-Domain Annotation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17462v3](http://arxiv.org/pdf/2410.17462v3)**

> **作者:** Minhua Lin; Zhengzhang Chen; Yanchi Liu; Xujiang Zhao; Zongyu Wu; Junxiang Wang; Xiang Zhang; Suhang Wang; Haifeng Chen
>
> **备注:** 29 pages, 12 figures, 32 tables
>
> **摘要:** Time series data is ubiquitous across various domains, including manufacturing, finance, and healthcare. High-quality annotations are essential for effectively understanding time series and facilitating downstream tasks; however, obtaining such annotations is challenging, particularly in mission-critical domains. In this paper, we propose TESSA, a multi-agent system designed to automatically generate both general and domain-specific annotations for time series data. TESSA introduces two agents: a general annotation agent and a domain-specific annotation agent. The general agent captures common patterns and knowledge across multiple source domains, leveraging both time-series-wise and text-wise features to generate general annotations. Meanwhile, the domain-specific agent utilizes limited annotations from the target domain to learn domain-specific terminology and generate targeted annotations. Extensive experiments on multiple synthetic and real-world datasets demonstrate that TESSA effectively generates high-quality annotations, outperforming existing methods.
>
---
#### [replaced 114] OR-Bench: An Over-Refusal Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.20947v3](http://arxiv.org/pdf/2405.20947v3)**

> **作者:** Justin Cui; Wei-Lin Chiang; Ion Stoica; Cho-Jui Hsieh
>
> **备注:** Accepted to ICML 2025, we thank everyone for their valuable suggestions and feedback!
>
> **摘要:** Large Language Models (LLMs) require careful safety alignment to prevent malicious outputs. While significant research focuses on mitigating harmful content generation, the enhanced safety often come with the side effect of over-refusal, where LLMs may reject innocuous prompts and become less helpful. Although the issue of over-refusal has been empirically observed, a systematic measurement is challenging due to the difficulty of crafting prompts that can elicit the over-refusal behaviors of LLMs. This study proposes a novel method for automatically generating large-scale over-refusal datasets. Leveraging this technique, we introduce OR-Bench, the first large-scale over-refusal benchmark. OR-Bench comprises 80,000 over-refusal prompts across 10 common rejection categories, a subset of around 1,000 hard prompts that are challenging even for state-of-the-art LLMs, and an additional 600 toxic prompts to prevent indiscriminate responses. We then conduct a comprehensive study to measure the over-refusal of 32 popular LLMs across 8 model families. Our datasets are publicly available at https://huggingface.co/bench-llms and our codebase is open-sourced at https://github.com/justincui03/or-bench. We hope this benchmark can help the community develop better safety aligned models.
>
---
#### [replaced 115] MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01711v4](http://arxiv.org/pdf/2503.01711v4)**

> **作者:** Weicong Qin; Yi Xu; Weijie Yu; Chenglei Shen; Ming He; Jianping Fan; Xiao Zhang; Jun Xu
>
> **备注:** accepted to ACL 2025 main conference
>
> **摘要:** Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks.
>
---
#### [replaced 116] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05831v2](http://arxiv.org/pdf/2504.05831v2)**

> **作者:** Mingye Zhu; Yi Liu; Zheren Fu; Yongdong Zhang; Zhendong Mao
>
> **摘要:** Preference alignment methods are increasingly critical for steering large language models (LLMs) to generate outputs consistent with human values. While recent approaches often rely on synthetic data generated by LLMs for scalability and cost-efficiency reasons, this reliance can introduce distribution shifts that undermine the nuanced representation of human preferences needed for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment despite such shifts. Our approach first leverages well-learned classifiers to assign a calibration value to each training sample, quantifying its alignment with the target human-preferred distribution. These values are then incorporated into a robust optimization objective that minimizes the worst-case loss over regions of the data space most relevant to human preferences. By explicitly focusing optimization on the target distribution, our approach mitigates the impact of distributional mismatch and improves the generation of responses that better reflect intended values.
>
---
#### [replaced 117] Who Taught You That? Tracing Teachers in Model Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06659v2](http://arxiv.org/pdf/2502.06659v2)**

> **作者:** Somin Wadhwa; Chantal Shaib; Silvio Amir; Byron C. Wallace
>
> **备注:** Findings of ACL 2025
>
> **摘要:** Model distillation -- using outputs from a large teacher model to teach a small student model -- is a practical means of creating efficient models for a particular task. We ask: Can we identify a students' teacher based on its outputs? Such "footprints" left by teacher LLMs would be interesting artifacts. Beyond this, reliable teacher inference may have practical implications as actors seek to distill specific capabilities of massive proprietary LLMs into deployed smaller LMs, potentially violating terms of service. We consider practical task distillation targets including summarization, question answering, and instruction-following. We assume a finite set of candidate teacher models, which we treat as blackboxes. We design discriminative models that operate over lexical features. We find that $n$-gram similarity alone is unreliable for identifying teachers, but part-of-speech (PoS) templates preferred by student models mimic those of their teachers.
>
---
#### [replaced 118] Large Linguistic Models: Investigating LLMs' metalinguistic abilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2305.00948v4](http://arxiv.org/pdf/2305.00948v4)**

> **作者:** Gašper Beguš; Maksymilian Dąbkowski; Ryan Rhodes
>
> **摘要:** The performance of large language models (LLMs) has recently improved to the point where models can perform well on many language tasks. We show here that--for the first time--the models can also generate valid metalinguistic analyses of language data. We outline a research program where the behavioral interpretability of LLMs on these tasks is tested via prompting. LLMs are trained primarily on text--as such, evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. We show that OpenAI's (2024) o1 vastly outperforms other models on tasks involving drawing syntactic trees and phonological generalization. We speculate that OpenAI o1's unique advantage over other models may result from the model's chain-of-thought mechanism, which mimics the structure of human reasoning used in complex cognitive tasks, such as linguistic analysis.
>
---
#### [replaced 119] FANformer: Improving Large Language Models Through Effective Periodicity Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.21309v2](http://arxiv.org/pdf/2502.21309v2)**

> **作者:** Yihong Dong; Ge Li; Xue Jiang; Yongding Tao; Kechi Zhang; Hao Zhu; Huanyu Liu; Jiazheng Ding; Jia Li; Jinliang Deng; Hong Mei
>
> **摘要:** Periodicity, as one of the most important basic characteristics, lays the foundation for facilitating structured knowledge acquisition and systematic cognitive processes within human learning paradigms. However, the potential flaws of periodicity modeling in Transformer affect the learning efficiency and establishment of underlying principles from data for large language models (LLMs) built upon it. In this paper, we demonstrate that integrating effective periodicity modeling can improve the learning efficiency and performance of LLMs. We introduce FANformer, which adapts Fourier Analysis Network (FAN) into attention mechanism to achieve efficient periodicity modeling, by modifying the feature projection process of attention mechanism. Extensive experimental results on language modeling show that FANformer consistently outperforms Transformer when scaling up model size and training tokens, underscoring its superior learning efficiency. Our pretrained FANformer-1B exhibits marked improvements on downstream tasks compared to open-source LLMs with similar model parameters or training tokens. Moreover, we reveal that FANformer exhibits superior ability to learn and apply rules for reasoning compared to Transformer. The results position FANformer as an effective and promising architecture for advancing LLMs.
>
---
#### [replaced 120] Task Facet Learning: A Structured Approach to Prompt Optimization
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.10504v2](http://arxiv.org/pdf/2406.10504v2)**

> **作者:** Gurusha Juneja; Gautam Jajoo; Nagarajan Natarajan; Hua Li; Jian Jiao; Amit Sharma
>
> **备注:** Appearing in ACL Findings 2025
>
> **摘要:** Given a task in the form of a basic description and its training examples, prompt optimization is the problem of synthesizing the given information into a text prompt for a large language model. Humans solve this problem by also considering the different facets that define a task (e.g., counter-examples, explanations, analogies) and including them in the prompt. However, it is unclear whether existing algorithmic approaches, based on iteratively editing a given prompt or automatically selecting a few in-context examples, can cover the multiple facets required to solve a complex task. In this work, we view prompt optimization as that of learning multiple facets of a task from a set of training examples. We exploit structure in the prompt optimization problem and break down a prompt into loosely coupled semantic sections. The proposed algorithm, UniPrompt, (1) clusters the input space and uses clustered batches so that each batch likely corresponds to a different facet of the task, and (2) utilizes a feedback mechanism to propose adding, editing or deleting a section, which in turn is aggregated over a batch to capture generalizable facets. Empirical evaluation on multiple datasets and a real-world task shows that prompts generated using \shortname{} obtain higher accuracy than human-tuned prompts and those from state-of-the-art methods. In particular, our algorithm can generate long, complex prompts that existing methods are unable to generate. Code for UniPrompt is available at https://aka.ms/uniprompt.
>
---
#### [replaced 121] Learning Efficient Recursive Numeral Systems via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.07170v4](http://arxiv.org/pdf/2409.07170v4)**

> **作者:** Andrea Silvi; Jonathan Thomas; Emil Carlsson; Devdatt Dubhashi; Moa Johansson
>
> **备注:** Accepted to CogSci 2025
>
> **摘要:** It has previously been shown that by using reinforcement learning (RL), agents can derive simple approximate and exact-restricted numeral systems that are similar to human ones (Carlsson, 2021). However, it is a major challenge to show how more complex recursive numeral systems, similar to for example English, could arise via a simple learning mechanism such as RL. Here, we introduce an approach towards deriving a mechanistic explanation of the emergence of efficient recursive number systems. We consider pairs of agents learning how to communicate about numerical quantities through a meta-grammar that can be gradually modified throughout the interactions. Utilising a slightly modified version of the meta-grammar of Hurford (1975), we demonstrate that our RL agents, shaped by the pressures for efficient communication, can effectively modify their lexicon towards Pareto-optimal configurations which are comparable to those observed within human numeral systems in terms of their efficiency.
>
---
#### [replaced 122] Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08392v2](http://arxiv.org/pdf/2505.08392v2)**

> **作者:** Ren Zhuang; Ben Wang; Shuifa Sun
>
> **摘要:** Large Language Models leverage Chain-of-Thought (CoT) prompting for complex tasks, but their reasoning traces are often excessively verbose and inefficient, leading to significant computational costs and latency. Current CoT compression techniques typically rely on generic importance metrics and static compression rates, which may inadvertently remove functionally critical tokens or fail to adapt to varying reasoning complexity. To overcome these limitations, we propose Adaptive GoGI-Skip, a novel framework learning dynamic CoT compression via supervised fine-tuning. This approach introduces two synergistic innovations: (1) Goal-Gradient Importance (GoGI), a novel metric accurately identifying functionally relevant tokens by measuring the gradient influence of their intermediate representations on the final answer loss, and (2) Adaptive Dynamic Skipping (ADS), a mechanism dynamically regulating the compression rate based on runtime model uncertainty while ensuring local coherence through an adaptive N-token constraint. To our knowledge, this is the first work unifying a goal-oriented, gradient-based importance metric with dynamic, uncertainty-aware skipping for CoT compression. Trained on compressed MATH data, Adaptive GoGI-Skip demonstrates strong cross-domain generalization across diverse reasoning benchmarks including AIME, GPQA, and GSM8K. It achieves substantial efficiency gains - reducing CoT token counts by over 45% on average and delivering 1.6-2.0 times inference speedups - while maintaining high reasoning accuracy. Notably, it significantly outperforms existing baselines by preserving accuracy even at high effective compression rates, advancing the state of the art in the CoT reasoning efficiency-accuracy trade-off.
>
---
#### [replaced 123] FaMTEB: Massive Text Embedding Benchmark in Persian Language
- **分类: cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.11571v2](http://arxiv.org/pdf/2502.11571v2)**

> **作者:** Erfan Zinvandi; Morteza Alikhani; Mehran Sarmadi; Zahra Pourbahman; Sepehr Arvin; Reza Kazemi; Arash Amini
>
> **摘要:** In this paper, we introduce a comprehensive benchmark for Persian (Farsi) text embeddings, built upon the Massive Text Embedding Benchmark (MTEB). Our benchmark includes 63 datasets spanning seven different tasks: classification, clustering, pair classification, reranking, retrieval, summary retrieval, and semantic textual similarity. The datasets are formed as a combination of existing, translated, and newly generated data, offering a diverse evaluation framework for Persian language models. Given the increasing use of text embedding models in chatbots, evaluation datasets are becoming inseparable ingredients in chatbot challenges and Retrieval-Augmented Generation systems. As a contribution, we include chatbot evaluation datasets in the MTEB benchmark for the first time. In addition, in this paper, we introduce the new task of summary retrieval which is not part of the tasks included in standard MTEB. Another contribution of this paper is the introduction of a substantial number of new Persian language NLP datasets suitable for training and evaluation, some of which have no previous counterparts in Persian. We evaluate the performance of several Persian and multilingual embedding models in a range of tasks. This work introduces an open-source benchmark with datasets, code and a public leaderboard.
>
---
#### [replaced 124] Dynamic Early Exit in Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.15895v2](http://arxiv.org/pdf/2504.15895v2)**

> **作者:** Chenxu Yang; Qingyi Si; Yongjie Duan; Zheliang Zhu; Chenyu Zhu; Qiaowei Li; Zheng Lin; Li Cao; Weiping Wang
>
> **备注:** 23 pages, 15 figures
>
> **摘要:** Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points (e.g.,"Wait" tokens) and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on 10 reasoning benchmarks (e.g., GSM8K, MATH-500, AMC, GPQA, AIME and LiveCodeBench) show that the proposed method is consistently effective on 11 cutting-edge reasoning LLMs of varying series and sizes, reducing the length of CoT sequences by an average of 19.1% to 80.1% while improving accuracy by 0.3% to 5.0%.
>
---
#### [replaced 125] Simple and Provable Scaling Laws for the Test-Time Compute of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19477v4](http://arxiv.org/pdf/2411.19477v4)**

> **作者:** Yanxi Chen; Xuchen Pan; Yaliang Li; Bolin Ding; Jingren Zhou
>
> **摘要:** We propose two simple, principled and practical algorithms that enjoy provable scaling laws for the test-time compute of large language models (LLMs). The first one is a two-stage knockout-style algorithm: given an input problem, it first generates multiple candidate solutions, and then aggregate them via a knockout tournament for the final output. Assuming that the LLM can generate a correct solution with non-zero probability and do better than a random guess in comparing a pair of correct and incorrect solutions, we prove theoretically that the failure probability of this algorithm decays to zero exponentially or by a power law (depending on the specific way of scaling) as its test-time compute grows. The second one is a two-stage league-style algorithm, where each candidate is evaluated by its average win rate against multiple opponents, rather than eliminated upon loss to a single opponent. Under analogous but more robust assumptions, we prove that its failure probability also decays to zero exponentially with more test-time compute. Both algorithms require a black-box LLM and nothing else (e.g., no verifier or reward model) for a minimalistic implementation, which makes them appealing for practical applications and easy to adapt for different tasks. Through extensive experiments with diverse models and datasets, we validate the proposed theories and demonstrate the outstanding scaling properties of both algorithms.
>
---
#### [replaced 126] How Linguistics Learned to Stop Worrying and Love the Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.17047v2](http://arxiv.org/pdf/2501.17047v2)**

> **作者:** Richard Futrell; Kyle Mahowald
>
> **摘要:** Language models can produce fluent, grammatical text. Nonetheless, some maintain that language models don't really learn language and also that, even if they did, that would not be informative for the study of human learning and processing. On the other side, there have been claims that the success of LMs obviates the need for studying linguistic theory and structure. We argue that both extremes are wrong. LMs can contribute to fundamental questions about linguistic structure, language processing, and learning. They force us to rethink arguments and ways of thinking that have been foundational in linguistics. While they do not replace linguistic structure and theory, they serve as model systems and working proofs of concept for gradient, usage-based approaches to language. We offer an optimistic take on the relationship between language models and linguistics.
>
---
#### [replaced 127] The Mirage of Model Editing: Revisiting Evaluation in the Wild
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11177v4](http://arxiv.org/pdf/2502.11177v4)**

> **作者:** Wanli Yang; Fei Sun; Jiajun Tan; Xinyu Ma; Qi Cao; Dawei Yin; Huawei Shen; Xueqi Cheng
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Despite near-perfect results in artificial evaluations, the effectiveness of model editing in real-world applications remains unexplored. To bridge this gap, we propose to study model editing in question answering (QA) by establishing a rigorous evaluation practice to assess the effectiveness of editing methods in correcting LLMs' errors. It consists of QAEdit, a new benchmark derived from popular QA datasets, and a standardized evaluation framework. Our single editing experiments indicate that current editing methods perform substantially worse than previously reported (38.5% vs. ~96%). Through module analysis and controlled experiments, we demonstrate that this performance decline stems from issues in evaluation practices of prior editing research. One key issue is the inappropriate use of teacher forcing in testing prevents error propagation by feeding ground truth tokens (inaccessible in real-world scenarios) as input. Furthermore, we simulate real-world deployment by sequential editing, revealing that current approaches fail drastically with only 1000 edits. Our analysis provides a fundamental reexamination of both the real-world applicability of existing model editing methods and their evaluation practices, and establishes a rigorous evaluation framework with key insights to advance reliable and practical model editing research.
>
---
#### [replaced 128] Tracr-Injection: Distilling Algorithms into Pre-trained Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10719v2](http://arxiv.org/pdf/2505.10719v2)**

> **作者:** Tomás Vergara-Browne; Álvaro Soto
>
> **备注:** ACL Findings 2025
>
> **摘要:** Motivated by the surge of large language models, there has been a push to formally characterize the symbolic abilities intrinsic to the transformer architecture. A programming language, called RASP, has been proposed, which can be directly compiled into transformer weights to implement these algorithms. However, the tasks that can be implemented in RASP are often uncommon to learn from natural unsupervised data, showing a mismatch between theoretical capabilities of the transformer architecture, and the practical learnability of these capabilities from unsupervised data. We propose tracr-injection, a method that allows us to distill algorithms written in RASP directly into a pre-trained language model. We showcase our method by injecting 3 different algorithms into a language model. We show how our method creates an interpretable subspace within the model's residual stream, which can be decoded into the variables present in the code of the RASP algorithm. Additionally, we found that the proposed method can improve out-of-distribution performance compared to our baseline, indicating that indeed a more symbolic mechanism is taking place in the inner workings of the model. We release the code used to run our experiments.
>
---
#### [replaced 129] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17192v3](http://arxiv.org/pdf/2504.17192v3)**

> **作者:** Minju Seo; Jinheon Baek; Seongyun Lee; Sung Ju Hwang
>
> **摘要:** Despite the rapid growth of machine learning research, corresponding code implementations are often unavailable, making it slow and labor-intensive for researchers to reproduce results and build upon prior work. In the meantime, recent Large Language Models (LLMs) excel at understanding scientific documents and generating high-quality code. Inspired by this, we introduce PaperCoder, a multi-agent LLM framework that transforms machine learning papers into functional code repositories. PaperCoder operates in three stages: planning, where it constructs a high-level roadmap, designs the system architecture with diagrams, identifies file dependencies, and generates configuration files; analysis, which focuses on interpreting implementation-specific details; and generation, where modular, dependency-aware code is produced. Moreover, each phase is instantiated through a set of specialized agents designed to collaborate effectively across the pipeline. We then evaluate PaperCoder on generating code implementations from machine learning papers based on both model-based and human evaluations, particularly from the authors of those papers, with author-released repositories as ground truth if available. Our results demonstrate the effectiveness of PaperCoder in creating high-quality, faithful implementations. Furthermore, it consistently shows strengths in the recently released PaperBench benchmark, surpassing strong baselines by substantial margins. Code is available at: https://github.com/going-doer/Paper2Code.
>
---
#### [replaced 130] To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12202v2](http://arxiv.org/pdf/2502.12202v2)**

> **作者:** Zihao Zhu; Hongbao Zhang; Ruotong Wang; Ke Xu; Siwei Lyu; Baoyuan Wu
>
> **备注:** 39 pages, 13 tables, 14 figures
>
> **摘要:** Large Reasoning Models (LRMs) are designed to solve complex tasks by generating explicit reasoning traces before producing final answers. However, we reveal a critical vulnerability in LRMs -- termed Unthinking Vulnerability -- wherein the thinking process can be bypassed by manipulating special delimiter tokens. It is empirically demonstrated to be widespread across mainstream LRMs, posing both a significant risk and potential utility, depending on how it is exploited. In this paper, we systematically investigate this vulnerability from both malicious and beneficial perspectives. On the malicious side, we introduce Breaking of Thought (BoT), a novel attack that enables adversaries to bypass the thinking process of LRMs, thereby compromising their reliability and availability. We present two variants of BoT: a training-based version that injects backdoor during the fine-tuning stage, and a training-free version based on adversarial attack during the inference stage. As a potential defense, we propose thinking recovery alignment to partially mitigate the vulnerability. On the beneficial side, we introduce Monitoring of Thought (MoT), a plug-and-play framework that allows model owners to enhance efficiency and safety. It is implemented by leveraging the same vulnerability to dynamically terminate redundant or risky reasoning through external monitoring. Extensive experiments show that BoT poses a significant threat to reasoning reliability, while MoT provides a practical solution for preventing overthinking and jailbreaking. Our findings expose an inherent flaw in current LRM architectures and underscore the need for more robust reasoning systems in the future.
>
---
#### [replaced 131] Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00979v2](http://arxiv.org/pdf/2505.00979v2)**

> **作者:** Xuhui Jiang; Shengjie Ma; Chengjin Xu; Cehao Yang; Liyu Zhang; Jian Guo
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success but remain data-inefficient, especially when learning from small, specialized corpora with limited and proprietary data. Existing synthetic data generation methods for continue pre-training focus on intra-document content and overlook cross-document knowledge associations, limiting content diversity and depth. We propose Synthetic-on-Graph (SoG), a synthetic data generation framework that incorporates cross-document knowledge associations for efficient corpus expansion. SoG constructs a context graph by extracting entities and concepts from the original corpus, representing cross-document associations, and employing a graph walk strategy for knowledge-associated sampling. This enhances synthetic data diversity and coherence, enabling models to learn complex knowledge structures and handle rare knowledge. To further improve synthetic data quality, we integrate Chain-of-Thought (CoT) and Contrastive Clarifying (CC) synthetic, enhancing reasoning processes and discriminative power. Experiments show that SoG outperforms the state-of-the-art (SOTA) method in a multi-hop document Q&A dataset while performing comparably to the SOTA method on the reading comprehension task datasets, which also underscores the better generalization capability of SoG. Our work advances synthetic data generation and provides practical solutions for efficient knowledge acquisition in LLMs, especially in domains with limited data availability.
>
---
#### [replaced 132] RARE: Retrieval-Augmented Reasoning Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.23513v2](http://arxiv.org/pdf/2503.23513v2)**

> **作者:** Zhengren Wang; Jiayang Yu; Dongsheng Ma; Zhe Chen; Yu Wang; Zhiyu Li; Feiyu Xiong; Yanfeng Wang; Weinan E; Linpeng Tang; Wentao Zhang
>
> **备注:** Repo: https://github.com/Open-DataFlow/RARE
>
> **摘要:** Domain-specific intelligence demands specialized knowledge and sophisticated reasoning for problem-solving, posing significant challenges for large language models (LLMs) that struggle with knowledge hallucination and inadequate reasoning capabilities under constrained parameter budgets. Inspired by Bloom's Taxonomy in educational theory, we propose Retrieval-Augmented Reasoning Modeling (RARE), a novel paradigm that decouples knowledge storage from reasoning optimization. RARE externalizes domain knowledge to retrievable sources and internalizes domain-specific reasoning patterns during training. Specifically, by injecting retrieved knowledge into training prompts with masked losses, RARE transforms learning objectives from rote memorization to contextualized reasoning. It enables models to bypass parameter-intensive memorization and prioritize the development of higher-order cognitive processes. Extensive experiments demonstrate that lightweight RARE-trained models (e.g., Llama-3.1-8B) could achieve state-of-the-art performance, surpassing retrieval-augmented GPT-4 and DeepSeek-R1 up to approximately 20\% accuracy. RARE establishes a paradigm shift where maintainable external knowledge bases synergize with compact, reasoning-optimized models, collectively driving more scalable domain-specific intelligence.
>
---
#### [replaced 133] From the New World of Word Embeddings: A Comparative Study of Small-World Lexico-Semantic Networks in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11380v2](http://arxiv.org/pdf/2502.11380v2)**

> **作者:** Zhu Liu; Ying Liu; KangYang Luo; Cunliang Kong; Maosong Sun
>
> **备注:** Paper under review
>
> **摘要:** Lexico-semantic networks represent words as nodes and their semantic relatedness as edges. While such networks are traditionally constructed using embeddings from encoder-based models or static vectors, embeddings from decoder-only large language models (LLMs) remain underexplored. Unlike encoder models, LLMs are trained with a next-token prediction objective, which does not directly encode the meaning of the current token. In this paper, we construct lexico-semantic networks from the input embeddings of LLMs with varying parameter scales and conduct a comparative analysis of their global and local structures. Our results show that these networks exhibit small-world properties, characterized by high clustering and short path lengths. Moreover, larger LLMs yield more intricate networks with less small-world effects and longer paths, reflecting richer semantic structures and relations. We further validate our approach through analyses of common conceptual pairs, structured lexical relations derived from WordNet, and a cross-lingual semantic network for qualitative words.
>
---
#### [replaced 134] The Hidden Strength of Disagreement: Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.16565v2](http://arxiv.org/pdf/2502.16565v2)**

> **作者:** Zengqing Wu; Takayuki Ito
>
> **备注:** Source codes are available at https://github.com/wuzengqing001225/ConsensusDiversityTradeoffMAS
>
> **摘要:** Consensus formation is pivotal in multi-agent systems (MAS), balancing collective coherence with individual diversity. Conventional LLM-based MAS primarily rely on explicit coordination, e.g., prompts or voting, risking premature homogenization. We argue that implicit consensus, where agents exchange information yet independently form decisions via in-context learning, can be more effective in dynamic environments that require long-horizon adaptability. By retaining partial diversity, systems can better explore novel strategies and cope with external shocks. We formalize a consensus-diversity tradeoff, showing conditions where implicit methods outperform explicit ones. Experiments on three scenarios -- Dynamic Disaster Response, Information Spread and Manipulation, and Dynamic Public-Goods Provision -- confirm partial deviation from group norms boosts exploration, robustness, and performance. We highlight emergent coordination via in-context learning, underscoring the value of preserving diversity for resilient decision-making.
>
---
#### [replaced 135] Efficient Indirect LLM Jailbreak via Multimodal-LLM Jailbreak
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.20015v2](http://arxiv.org/pdf/2405.20015v2)**

> **作者:** Zhenxing Niu; Yuyao Sun; Haoxuan Ji; Zheng Lin; Haichang Gao; Xinbo Gao; Gang Hua; Rong Jin
>
> **摘要:** This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.
>
---
#### [replaced 136] VLSBench: Unveiling Visual Leakage in Multimodal Safety
- **分类: cs.CR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19939v3](http://arxiv.org/pdf/2411.19939v3)**

> **作者:** Xuhao Hu; Dongrui Liu; Hao Li; Xuanjing Huang; Jing Shao
>
> **备注:** ACL2025 Main
>
> **摘要:** Safety concerns of Multimodal large language models (MLLMs) have gradually become an important problem in various applications. Surprisingly, previous works indicate a counterintuitive phenomenon that using textual unlearning to align MLLMs achieves comparable safety performances with MLLMs aligned with image text pairs. To explain such a phenomenon, we discover a Visual Safety Information Leakage (VSIL) problem in existing multimodal safety benchmarks, i.e., the potentially risky content in the image has been revealed in the textual query. Thus, MLLMs can easily refuse these sensitive image-text pairs according to textual queries only, leading to unreliable cross-modality safety evaluation of MLLMs. We also conduct a further comparison experiment between textual alignment and multimodal alignment to highlight this drawback. To this end, we construct multimodal Visual Leakless Safety Bench (VLSBench) with 2.2k image-text pairs through an automated data pipeline. Experimental results indicate that VLSBench poses a significant challenge to both open-source and close-source MLLMs, e.g., LLaVA, Qwen2-VL and GPT-4o. Besides, we empirically compare textual and multimodal alignment methods on VLSBench and find that textual alignment is effective enough for multimodal safety scenarios with VSIL, while multimodal alignment is preferable for safety scenarios without VSIL. Code and data are released under https://github.com/AI45Lab/VLSBench
>
---
#### [replaced 137] Exploring the Potential of Encoder-free Architectures in 3D LMMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09620v2](http://arxiv.org/pdf/2502.09620v2)**

> **作者:** Yiwen Tang; Zoey Guo; Zhuhao Wang; Ray Zhang; Qizhi Chen; Junli Liu; Delin Qu; Zhigang Wang; Dong Wang; Xuelong Li; Bin Zhao
>
> **备注:** The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
> **摘要:** Encoder-free architectures have been preliminarily explored in the 2D visual domain, yet it remains an open question whether they can be effectively applied to 3D understanding scenarios. In this paper, we present the first comprehensive investigation into the potential of encoder-free architectures to alleviate the challenges of encoder-based 3D Large Multimodal Models (LMMs). These challenges include the failure to adapt to varying point cloud resolutions and the point features from the encoder not meeting the semantic needs of Large Language Models (LLMs). We identify key aspects for 3D LMMs to remove the encoder and enable the LLM to assume the role of the 3D encoder: 1) We propose the LLM-embedded Semantic Encoding strategy in the pre-training stage, exploring the effects of various point cloud self-supervised losses. And we present the Hybrid Semantic Loss to extract high-level semantics. 2) We introduce the Hierarchical Geometry Aggregation strategy in the instruction tuning stage. This incorporates inductive bias into the LLM layers to focus on the local details of the point clouds. To the end, we present the first Encoder-free 3D LMM, ENEL. Our 7B model rivals the current state-of-the-art model, ShapeLLM-13B, achieving 55.10%, 50.98%, and 43.10% on the classification, captioning, and VQA tasks, respectively. Our results demonstrate that the encoder-free architecture is highly promising for replacing encoder-based architectures in the field of 3D understanding. The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
---
#### [replaced 138] Streaming Sequence Transduction through Dynamic Compression
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.01172v2](http://arxiv.org/pdf/2402.01172v2)**

> **作者:** Weiting Tan; Yunmo Chen; Tongfei Chen; Guanghui Qin; Haoran Xu; Heidi C. Zhang; Benjamin Van Durme; Philipp Koehn
>
> **备注:** IWSLT 2025
>
> **摘要:** We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
>
---
#### [replaced 139] ARS: Automatic Routing Solver with Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15359v3](http://arxiv.org/pdf/2502.15359v3)**

> **作者:** Kai Li; Fei Liu; Zhenkun Wang; Xialiang Tong; Xiongwei Han; Mingxuan Yuan; Qingfu Zhang
>
> **备注:** Authorship is under discussion; arXiv release will follow finalization
>
> **摘要:** Real-world Vehicle Routing Problems (VRPs) are characterized by a variety of practical constraints, making manual solver design both knowledge-intensive and time-consuming. Although there is increasing interest in automating the design of routing algorithms, existing research has explored only a limited array of VRP variants and fails to adequately address the complex and prevalent constraints encountered in real-world situations. To fill this gap, this paper introduces RoutBench, a benchmark of 1,000 VRP variants derived from 24 attributes, for evaluating the effectiveness of automatic routing solvers in addressing complex constraints. Along with RoutBench, we present the Automatic Routing Solver (ARS), which employs Large Language Model (LLM) agents to enhance a backbone algorithm framework by automatically generating constraint-aware heuristic code, based on problem descriptions and several representative constraints selected from a database. Our experiments show that ARS outperforms state-of-the-art LLM-based methods and commonly used solvers, automatically solving 91.67% of common VRPs and achieving at least a 30% improvement across all benchmarks.
>
---
#### [replaced 140] FreqKV: Frequency Domain Key-Value Compression for Efficient Context Window Extension
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00570v2](http://arxiv.org/pdf/2505.00570v2)**

> **作者:** Jushi Kai; Boyi Zeng; Yixuan Wang; Haoli Bai; Ziwei He; Bo Jiang; Zhouhan Lin
>
> **摘要:** Frequency-domain compression has proven effective in reducing redundancies for spatial signals. In this work, we propose FreqKV, a novel frequency domain key-value (KV) compression technique that enables efficient context window extension for decoder-only large language models (LLMs). Our approach is motivated by a key observation that, in the frequency domain, the energy distribution of the KV cache is predominantly concentrated in low-frequency components. By discarding high-frequency components, we achieve efficient compression of the KV cache with minimal information loss. FreqKV iteratively compresses the increasing KV cache to a fixed size in the frequency domain, allowing models to process lengthy contexts efficiently. Introducing no additional parameters or architectural modifications, FreqKV is applicable to both fine-tuning and inference. With minimal fine-tuning, LLMs can learn to leverage the limited cache that is compressed in the frequency domain and extend the context window. Experiments on a range of long context language modeling and understanding tasks demonstrate the efficiency and effectiveness of the proposed method.
>
---
#### [replaced 141] CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13534v2](http://arxiv.org/pdf/2504.13534v2)**

> **作者:** Feiyang Li; Peng Fang; Zhan Shi; Arijit Khan; Fang Wang; Dan Feng; Weihao Wang; Xin Zhang; Yongjian Cui
>
> **摘要:** Chain-of-thought (CoT) reasoning boosts large language models' (LLMs) performance on complex tasks but faces two key limitations: a lack of reliability when solely relying on LLM-generated reasoning chains and interference from natural language reasoning steps with the models' inference process, also known as the inference logic of LLMs. To address these issues, we propose CoT-RAG, a novel reasoning framework with three key designs: (i) Knowledge Graph-driven CoT Generation,featuring knowledge graphs to modulate reasoning chain generation of LLMs, thereby enhancing reasoning credibility; (ii) Learnable Knowledge Case-aware RAG, which incorporates retrieval-augmented generation (RAG) into knowledge graphs to retrieve relevant sub-cases and sub-descriptions, providing LLMs with learnable information; (iii) Pseudo-Program Prompting Execution, which promotes greater logical rigor by guiding LLMs to execute reasoning tasks as pseudo-programs. Evaluations on nine public datasets spanning three reasoning tasks reveal significant accuracy gains--ranging from 4.0% to 44.3%--over state-of-the-art methods. Furthermore, tests on four domain-specific datasets demonstrate exceptional accuracy and efficient execution, underscoring its practical applicability and scalability.
>
---
#### [replaced 142] UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.20734v2](http://arxiv.org/pdf/2504.20734v2)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Soyeong Jeong; Jinheon Baek; Sung Ju Hwang
>
> **备注:** Project page : https://universalrag.github.io
>
> **摘要:** Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single aggregated corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over various modality-specific and unified baselines.
>
---
#### [replaced 143] SCoRE: Benchmarking Long-Chain Reasoning in Commonsense Scenarios
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06218v2](http://arxiv.org/pdf/2503.06218v2)**

> **作者:** Weidong Zhan; Yue Wang; Nan Hu; Liming Xiao; Jingyuan Ma; Yuhang Qin; Zheng Li; Yixin Yang; Sirui Deng; Jinkun Ding; Wenhan Ma; Rui Li; Weilin Luo; Qun Liu; Zhifang Sui
>
> **摘要:** Currently, long-chain reasoning remains a key challenge for large language models (LLMs) because natural texts lack sufficient explicit reasoning data. However, existing benchmarks suffer from limitations such as narrow coverage, short reasoning paths, or high construction costs. We introduce SCoRE (Scenario-based Commonsense Reasoning Evaluation), a benchmark that synthesizes multi-hop questions from scenario schemas of entities, relations, and logical rules to assess long-chain commonsense reasoning. SCoRE contains 100k bilingual (Chinese-English) multiple-choice questions whose reasoning chains span 2-11 hops and are grouped into various difficulty levels. Each question is accompanied by fine-grained knowledge labels, explicit reasoning chains, and difficulty levels for diagnostic evaluation. Evaluation results on cutting-edge LLMs such as o3-mini and Deepseek R1 shows that even the best model attains only 69.78% accuracy on SCoRE (even only 47.91% on the hard set), with errors often stemming from rare knowledge, logical inconsistency, and over-interpretation of simple questions. SCoRE offers a scalable, extensible framework for evaluating and diagnosing the long-chain commonsense reasoning abilities of LLMs and guiding future advances in model design and training.
>
---
#### [replaced 144] PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16074v2](http://arxiv.org/pdf/2504.16074v2)**

> **作者:** Shi Qiu; Shaoyang Guo; Zhuo-Yang Song; Yunbo Sun; Zeyu Cai; Jiashen Wei; Tianyu Luo; Yixuan Yin; Haoxu Zhang; Yi Hu; Chenyang Wang; Chencheng Tang; Haoling Chang; Qi Liu; Ziheng Zhou; Tianyu Zhang; Jingtian Zhang; Zhangyi Liu; Minghao Li; Yuku Zhang; Boxuan Jing; Xianqi Yin; Yutong Ren; Zizhuo Fu; Jiaming Ji; Weike Wang; Xudong Tian; Anqi Lv; Laifu Man; Jianxiang Li; Feiyu Tao; Qihua Sun; Zhou Liang; Yushu Mu; Zhongxuan Li; Jing-Jun Zhang; Shutao Zhang; Xiaotian Li; Xingqi Xia; Jiawei Lin; Zheyu Shen; Jiahang Chen; Qiuhao Xiong; Binran Wang; Fengyuan Wang; Ziyang Ni; Bohan Zhang; Fan Cui; Changkun Shao; Qing-Hong Cao; Ming-xing Luo; Yaodong Yang; Muhan Zhang; Hua Xing Zhu
>
> **备注:** 34 pages ,12 figures, 7 tables, latest update in 2025/05/18
>
> **摘要:** Current benchmarks for evaluating the reasoning capabilities of Large Language Models (LLMs) face significant limitations: task oversimplification, data contamination, and flawed evaluation items. These deficiencies necessitate more rigorous assessment methods. To address these limitations, we introduce PHYBench, a benchmark of 500 original physics problems ranging from high school to Physics Olympiad difficulty. PHYBench addresses data contamination through original content and employs a systematic curation pipeline to eliminate flawed items. Evaluations show that PHYBench activates more tokens and provides stronger differentiation between reasoning models compared to other baselines like AIME 2024, OlympiadBench and GPQA. Even the best-performing model, Gemini 2.5 Pro, achieves only 36.9% accuracy compared to human experts' 61.9%. To further enhance evaluation precision, we introduce the Expression Edit Distance (EED) Score for mathematical expression assessment, which improves sample efficiency by 204% over binary scoring. Moreover, PHYBench effectively elicits multi-step and multi-condition reasoning, providing a platform for examining models' reasoning robustness, preferences, and deficiencies. The benchmark results and dataset are publicly available at https://www.phybench.cn/.
>
---
#### [replaced 145] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10996v2](http://arxiv.org/pdf/2502.10996v2)**

> **作者:** Pengcheng Jiang; Lang Cao; Ruike Zhu; Minhao Jiang; Yunyi Zhang; Jimeng Sun; Jiawei Han
>
> **备注:** under review
>
> **摘要:** Large language models (LLMs) have achieved impressive performance on knowledge-intensive tasks, yet they often struggle with multi-step reasoning due to the unstructured nature of retrieved context. While retrieval-augmented generation (RAG) methods provide external information, the lack of explicit organization among retrieved passages limits their effectiveness, leading to brittle reasoning pathways. Recent interpretability studies highlighting the importance of structured intermediate reasoning further align with this perspective. We propose Retrieval-And-Structuring (RAS), a framework that dynamically constructs query-specific knowledge graphs through iterative retrieval and structured knowledge building. RAS interleaves targeted retrieval planning with incremental graph construction, enabling models to assemble and reason over evolving knowledge structures tailored to each query. On seven knowledge-intensive benchmarks, RAS consistently outperforms strong baselines, achieving up to 6.4% and 7.0% gains with open-source and proprietary LLMs, respectively. Our results demonstrate that dynamic, query-specific knowledge structuring offers a robust path to improving reasoning accuracy and robustness in language model generation. Our data and code can be found at https://github.com/pat-jj/RAS.
>
---
#### [replaced 146] BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.14971v2](http://arxiv.org/pdf/2410.14971v2)**

> **作者:** Jilong Li; Zhenxi Song; Jiaqi Wang; Meishan Zhang; Honghai Liu; Min Zhang; Zhiguo Zhang
>
> **摘要:** Current EEG/MEG-to-text decoding systems suffer from three key limitations: (1) reliance on teacher-forcing methods, which compromises robustness during inference, (2) sensitivity to session-specific noise, hindering generalization across subjects, and (3) misalignment between brain signals and linguistic representations due to pre-trained language model over-dominance. To overcome these challenges, we propose BrainECHO (Brain signal decoding via vEctor-quantized speCtrogram reconstruction for WHisper-enhanced text generatiOn), a multi-stage framework that employs decoupled representation learning to achieve state-of-the-art performance on both EEG and MEG datasets. Specifically, BrainECHO consists of three stages: (1) Discrete autoencoding, which transforms continuous Mel spectrograms into a finite set of high-quality discrete representations for subsequent stages. (2) Frozen alignment, where brain signal embeddings are mapped to corresponding Mel spectrogram embeddings in a frozen latent space, effectively filtering session-specific noise through vector-quantized reconstruction, yielding a 3.65% improvement in BLEU-4 score. (3) Constrained decoding fine-tuning, which leverages the pre-trained Whisper model for audio-to-text translation, balancing signal adaptation with knowledge preservation, and achieving 74%-89% decoding BLEU scores without excessive reliance on teacher forcing. BrainECHO demonstrates robustness across sentence, session, and subject-independent conditions, passing Gaussian noise tests and showcasing its potential for enhancing language-based brain-computer interfaces.
>
---
#### [replaced 147] HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12908v2](http://arxiv.org/pdf/2503.12908v2)**

> **作者:** Xinyan Jiang; Hang Ye; Yongxin Zhu; Xiaoying Zheng; Zikang Chen; Jun Gong
>
> **备注:** Accepted by ACL2025 findings
>
> **摘要:** Large Language Models (LLMs) often generate hallucinations, producing outputs that are contextually inaccurate or factually incorrect. We introduce HICD, a novel method designed to induce hallucinations for contrastive decoding to mitigate hallucinations. Unlike existing contrastive decoding methods, HICD selects attention heads crucial to the model's prediction as inducing heads, then induces hallucinations by dispersing attention of these inducing heads and compares the hallucinated outputs with the original outputs to obtain the final result. Our approach significantly improves performance on tasks requiring contextual faithfulness, such as context completion, reading comprehension, and question answering. It also improves factuality in tasks requiring accurate knowledge recall. We demonstrate that our inducing heads selection and attention dispersion method leads to more "contrast-effective" hallucinations for contrastive decoding, outperforming other hallucination-inducing methods. Our findings provide a promising strategy for reducing hallucinations by inducing hallucinations in a controlled manner, enhancing the performance of LLMs in a wide range of tasks.
>
---
#### [replaced 148] SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19162v2](http://arxiv.org/pdf/2504.19162v2)**

> **作者:** Jiaqi Chen; Bang Zhang; Ruotian Ma; Peisong Wang; Xiaodan Liang; Zhaopeng Tu; Xiaolong Li; Kwan-Yee K. Wong
>
> **备注:** Project webpage: https://chen-judge.github.io/SPC/
>
> **摘要:** Evaluating the step-by-step reliability of large language model (LLM) reasoning, such as Chain-of-Thought, remains challenging due to the difficulty and cost of obtaining high-quality step-level supervision. In this paper, we introduce Self-Play Critic (SPC), a novel approach where a critic model evolves its ability to assess reasoning steps through adversarial self-play games, eliminating the need for manual step-level annotation. SPC involves fine-tuning two copies of a base model to play two roles, namely a "sneaky generator" that deliberately produces erroneous steps designed to be difficult to detect, and a "critic" that analyzes the correctness of reasoning steps. These two models engage in an adversarial game in which the generator aims to fool the critic, while the critic model seeks to identify the generator's errors. Using reinforcement learning based on the game outcomes, the models iteratively improve; the winner of each confrontation receives a positive reward and the loser receives a negative reward, driving continuous self-evolution. Experiments on three reasoning process benchmarks (ProcessBench, PRM800K, DeltaBench) demonstrate that our SPC progressively enhances its error detection capabilities (e.g., accuracy increases from 70.8% to 77.7% on ProcessBench) and surpasses strong baselines, including distilled R1 model. Furthermore, SPC can guide the test-time search of diverse LLMs and significantly improve their mathematical reasoning performance on MATH500 and AIME2024, surpassing those guided by state-of-the-art process reward models.
>
---
#### [replaced 149] FISH-Tuning: Enhancing PEFT Methods with Fisher Information
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04050v2](http://arxiv.org/pdf/2504.04050v2)**

> **作者:** Kang Xue; Ming Dong; Xinhui Tu; Tingting He
>
> **摘要:** The rapid growth in the parameter size of Large Language Models (LLMs) has spurred the development of Parameter-Efficient Fine-Tuning (PEFT) methods to mitigate the substantial computational costs of fine-tuning. Among these, Fisher Induced Sparse uncHanging (FISH) Mask is a selection-based PEFT technique that identifies a critical subset of pre-trained parameters using approximate Fisher information. While addition-based and reparameterization-based PEFT methods like LoRA and Adapter already fine-tune only a small number of parameters, the newly introduced parameters within these methods themselves present an opportunity for further optimization. Selectively fine-tuning only the most impactful among these new parameters could further reduce resource consumption while maintaining, or even improving, fine-tuning effectiveness. In this paper, we propose \textbf{FISH-Tuning}, a novel approach that incorporates FISH Mask into such PEFT methods, including LoRA, Adapter, and their variants. By leveraging Fisher information to identify and update only the most significant parameters within these added or reparameterized components, FISH-Tuning aims to achieve superior performance without increasing training time or inference latency compared to the vanilla PEFT methods. Experimental results across various datasets and pre-trained models demonstrate that FISH-Tuning consistently outperforms the vanilla PEFT methods when using the same proportion of trainable parameters. Code is available at https://anonymous.4open.science/r/FISH-Tuning-6F7C.
>
---
#### [replaced 150] ImF: Implicit Fingerprint for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21805v2](http://arxiv.org/pdf/2503.21805v2)**

> **作者:** Wu jiaxuan; Peng Wanli; Fu hang; Xue Yiming; Wen juan
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.
>
---
#### [replaced 151] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12464v2](http://arxiv.org/pdf/2502.12464v2)**

> **作者:** Seanie Lee; Dong Bok Lee; Dominik Wagner; Minki Kang; Haebin Seong; Tobias Bocklet; Juho Lee; Sung Ju Hwang
>
> **备注:** 9 pages
>
> **摘要:** Deploying large language models (LLMs) in real-world applications requires robust safety guard models to detect and block harmful user prompts. While large safety guard models achieve strong performance, their computational cost is substantial. To mitigate this, smaller distilled models are used, but they often underperform on "hard" examples where the larger model provides accurate predictions. We observe that many inputs can be reliably handled by the smaller model, while only a small fraction require the larger model's capacity. Motivated by this, we propose SafeRoute, a binary router that distinguishes hard examples from easy ones. Our method selectively applies the larger safety guard model to the data that the router considers hard, improving efficiency while maintaining accuracy compared to solely using the larger safety guard model. Experimental results on multiple benchmark datasets demonstrate that our adaptive model selection significantly enhances the trade-off between computational cost and safety performance, outperforming relevant baselines.
>
---
#### [replaced 152] Multi-Agent Sampling: Scaling Inference Compute for Data Synthesis with Tree Search-Based Agentic Collaboration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17061v2](http://arxiv.org/pdf/2412.17061v2)**

> **作者:** Hai Ye; Mingbao Lin; Hwee Tou Ng; Shuicheng Yan
>
> **备注:** In submission
>
> **摘要:** Scaling laws for inference compute in multi-agent systems remain under-explored compared to single-agent scenarios. This work aims to bridge this gap by investigating the problem of data synthesis through multi-agent sampling, where synthetic responses are generated by sampling from multiple distinct language models. Effective model coordination is crucial for successful multi-agent collaboration. Unlike previous approaches that rely on fixed workflows, we treat model coordination as a multi-step decision-making process, optimizing generation structures dynamically for each input question. We introduce Tree Search-based Orchestrated Agents~(TOA), where the workflow evolves iteratively during the sequential sampling process. To achieve this, we leverage Monte Carlo Tree Search (MCTS), integrating a reward model to provide real-time feedback and accelerate exploration. Our experiments on alignment, machine translation, and mathematical reasoning demonstrate that multi-agent sampling significantly outperforms single-agent sampling as inference compute scales. TOA is the most compute-efficient approach, achieving SOTA performance on WMT and a 72.2\% LC win rate on AlpacaEval. Moreover, fine-tuning with our synthesized alignment data surpasses strong preference learning methods on challenging benchmarks such as Arena-Hard and AlpacaEval.
>
---
#### [replaced 153] RICo: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05327v2](http://arxiv.org/pdf/2505.05327v2)**

> **作者:** Yixin Yang; Qingxiu Dong; Linli Yao; Fangwei Zhu; Zhifang Sui
>
> **摘要:** Data selection for instruction tuning is crucial for improving the performance of large language models (LLMs) while reducing training costs. In this paper, we propose Refined Contribution Measurement with In-Context Learning (RICo), a novel gradient-free method that quantifies the fine-grained contribution of individual samples to both task-level and global-level model performance. RICo enables more accurate identification of high-contribution data, leading to better instruction tuning. We further introduce a lightweight selection paradigm trained on RICo scores, enabling scalable data selection with a strictly linear inference complexity. Extensive experiments on three LLMs across 12 benchmarks and 5 pairwise evaluation sets demonstrate the effectiveness of RICo. Remarkably, on LLaMA3.1-8B, models trained on 15% of RICo-selected data outperform full datasets by 5.42% points and exceed the best performance of widely used selection methods by 2.06% points. We further analyze high-contribution samples selected by RICo, which show both diverse tasks and appropriate difficulty levels, rather than just the hardest ones.
>
---
#### [replaced 154] Learning Virtual Machine Scheduling in Cloud Computing through Language Agents
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10117v2](http://arxiv.org/pdf/2505.10117v2)**

> **作者:** JieHao Wu; Ziwei Wang; Junjie Sheng; Wenhao Li; Xiangfeng Wang; Jun Luo
>
> **摘要:** In cloud services, virtual machine (VM) scheduling is a typical Online Dynamic Multidimensional Bin Packing (ODMBP) problem, characterized by large-scale complexity and fluctuating demands. Traditional optimization methods struggle to adapt to real-time changes, domain-expert-designed heuristic approaches suffer from rigid strategies, and existing learning-based methods often lack generalizability and interpretability. To address these limitations, this paper proposes a hierarchical language agent framework named MiCo, which provides a large language model (LLM)-driven heuristic design paradigm for solving ODMBP. Specifically, ODMBP is formulated as a Semi-Markov Decision Process with Options (SMDP-Option), enabling dynamic scheduling through a two-stage architecture, i.e., Option Miner and Option Composer. Option Miner utilizes LLMs to discover diverse and useful non-context-aware strategies by interacting with constructed environments. Option Composer employs LLMs to discover a composing strategy that integrates the non-context-aware strategies with the contextual ones. Extensive experiments on real-world enterprise datasets demonstrate that MiCo achieves a 96.9\% competitive ratio in large-scale scenarios involving more than 10,000 virtual machines. It maintains high performance even under nonstationary request flows and diverse configurations, thus validating its effectiveness in complex and large-scale cloud environments.
>
---
#### [replaced 155] Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in Large Language Models From Edge to Giant
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11055v5](http://arxiv.org/pdf/2409.11055v5)**

> **作者:** Jemin Lee; Sihyeong Park; Jinse Kwon; Jihun Oh; Yongin Kwon
>
> **备注:** Accepted in IJCAI 2025, 21 pages, 2 figure
>
> **摘要:** Quantization has gained attention as a promising solution for the cost-effective deployment of large and small language models. However, most prior work has been limited to perplexity or basic knowledge tasks and lacks a comprehensive evaluation of recent models like Llama-3.3. In this paper, we conduct a comprehensive evaluation of instruction-tuned models spanning 1B to 405B parameters, applying four quantization methods across 13 datasets. Our findings reveal that (1) quantized models generally surpass smaller FP16 baselines, yet they often struggle with instruction-following and hallucination detection; (2) FP8 consistently emerges as the most robust option across tasks, and AWQ tends to outperform GPTQ in weight-only quantization; (3) smaller models can suffer severe accuracy drops at 4-bit quantization, while 70B-scale models maintain stable performance; (4) notably, \textit{hard} tasks do not always experience the largest accuracy losses, indicating that quantization magnifies a model's inherent weaknesses rather than simply correlating with task difficulty; and (5) an LLM-based judge (MT-Bench) highlights significant performance declines in Coding and STEM tasks, though it occasionally reports improvements in reasoning.
>
---
#### [replaced 156] Watermarking Language Models with Error Correcting Codes
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.10281v3](http://arxiv.org/pdf/2406.10281v3)**

> **作者:** Patrick Chao; Yan Sun; Edgar Dobriban; Hamed Hassani
>
> **摘要:** Recent progress in large language models enables the creation of realistic machine-generated content. Watermarking is a promising approach to distinguish machine-generated text from human text, embedding statistical signals in the output that are ideally undetectable to humans. We propose a watermarking framework that encodes such signals through an error correcting code. Our method, termed robust binary code (RBC) watermark, introduces no noticeable degradation in quality. We evaluate our watermark on base and instruction fine-tuned models and find our watermark is robust to edits, deletions, and translations. We provide an information-theoretic perspective on watermarking, a powerful statistical test for detection and for generating $p$-values, and theoretical guarantees. Our empirical findings suggest our watermark is fast, powerful, and robust, comparing favorably to the state-of-the-art.
>
---
#### [replaced 157] Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.13139v2](http://arxiv.org/pdf/2503.13139v2)**

> **作者:** Weiyu Guo; Ziyang Chen; Shaoguang Wang; Jianxiang He; Yijie Xu; Jinhui Ye; Ying Sun; Hui Xiong
>
> **备注:** 32 pages, under review
>
> **摘要:** Understanding long video content is a complex endeavor that often relies on densely sampled frame captions or end-to-end feature selectors, yet these techniques commonly overlook the logical relationships between textual queries and visual elements. In practice, computational constraints necessitate coarse frame subsampling, a challenge analogous to "finding a needle in a haystack." To address this issue, we introduce a semantics-driven search framework that reformulates keyframe selection under the paradigm of Visual Semantic-Logical Search. Specifically, we systematically define four fundamental logical dependencies: 1) spatial co-occurrence, 2) temporal proximity, 3) attribute dependency, and 4) causal order. These relations dynamically update frame sampling distributions through an iterative refinement process, enabling context-aware identification of semantically critical frames tailored to specific query requirements. Our method establishes new SOTA performance on the manually annotated benchmark in key-frame selection metrics. Furthermore, when applied to downstream video question-answering tasks, the proposed approach demonstrates the best performance gains over existing methods on LongVideoBench and Video-MME, validating its effectiveness in bridging the logical gap between textual queries and visual-temporal reasoning. The code will be publicly available.
>
---
#### [replaced 158] AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.12162v2](http://arxiv.org/pdf/2501.12162v2)**

> **作者:** Zikun Li; Zhuofu Chen; Remi Delacourt; Gabriele Oliaro; Zeyu Wang; Qinghan Chen; Shuhuai Lin; April Yang; Zhihao Zhang; Zhuoming Chen; Sean Lai; Xinhao Cheng; Xupeng Miao; Zhihao Jia
>
> **摘要:** Modern large language model (LLM) applications exhibit diverse service-level objectives (SLOs), from low-latency requirements in interactive coding assistants to more relaxed constraints in data wrangling tasks. Existing LLM serving systems, which rely on uniform batching and scheduling strategies, often fail to meet these heterogeneous SLOs concurrently. We present AdaServe, the first LLM serving system designed to support efficient multi-SLO serving through SLO-customized speculative decoding. AdaServe formulates multi-SLO serving as a constrained optimization problem and introduces a hardware-aware algorithm that constructs a speculation tree tailored to each request's latency target. It features a speculate-select-verify pipeline that enables fine-grained control over decoding speed while maximizing system throughput. AdaServe further adapts to workload variation by dynamically adjusting speculation parameters. Evaluations across diverse workloads show that AdaServe reduces SLO violations by up to 4.3$\times$ and improves goodput by up to 1.9$\times$ compared to the best performing baselines, highlighting its effectiveness in multi-SLO serving.
>
---
#### [replaced 159] Prot42: a Novel Family of Protein Language Models for Target-aware Protein Binder Generation
- **分类: q-bio.BM; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.04453v2](http://arxiv.org/pdf/2504.04453v2)**

> **作者:** Mohammad Amaan Sayeed; Engin Tekin; Maryam Nadeem; Nancy A. ElNaker; Aahan Singh; Natalia Vassilieva; Boulbaba Ben Amor
>
> **摘要:** Unlocking the next generation of biotechnology and therapeutic innovation demands overcoming the inherent complexity and resource-intensity of conventional protein engineering methods. Recent GenAI-powered computational techniques often rely on the availability of the target protein's 3D structures and specific binding sites to generate high-affinity binders, constraints exhibited by models such as AlphaProteo and RFdiffusion. In this work, we explore the use of Protein Language Models (pLMs) for high-affinity binder generation. We introduce Prot42, a novel family of Protein Language Models (pLMs) pretrained on vast amounts of unlabeled protein sequences. By capturing deep evolutionary, structural, and functional insights through an advanced auto-regressive, decoder-only architecture inspired by breakthroughs in natural language processing, Prot42 dramatically expands the capabilities of computational protein design based on language only. Remarkably, our models handle sequences up to 8,192 amino acids, significantly surpassing standard limitations and enabling precise modeling of large proteins and complex multi-domain sequences. Demonstrating powerful practical applications, Prot42 excels in generating high-affinity protein binders and sequence-specific DNA-binding proteins. Our innovative models are publicly available, offering the scientific community an efficient and precise computational toolkit for rapid protein engineering.
>
---
#### [replaced 160] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14321v2](http://arxiv.org/pdf/2504.14321v2)**

> **作者:** Xingyu Li; Chen Gong; Guohong Fu
>
> **摘要:** Multimodal coreference resolution (MCR) aims to identify mentions referring to the same entity across different modalities, such as text and visuals, and is essential for understanding multimodal content. In the era of rapidly growing mutimodal content and social media, MCR is particularly crucial for interpreting user interactions and bridging text-visual references to improve communication and personalization. However, MCR research for real-world dialogues remains unexplored due to the lack of sufficient data resources. To address this gap, we introduce TikTalkCoref, the first Chinese multimodal coreference dataset for social media in real-world scenarios, derived from the popular Douyin short-video platform. This dataset pairs short videos with corresponding textual dialogues from user comments and includes manually annotated coreference clusters for both person mentions in the text and the coreferential person head regions in the corresponding video frames. We also present an effective benchmark approach for MCR, focusing on the celebrity domain, and conduct extensive experiments on our dataset, providing reliable benchmark results for this newly constructed dataset. We will release the TikTalkCoref dataset to facilitate future research on MCR for real-world social media dialogues.
>
---
#### [replaced 161] Generative AI and Large Language Models in Language Preservation: Opportunities and Challenges
- **分类: cs.CL; cs.AI; cs.LG; 68T50, 91F20; I.2.7; I.2.6; J.5**

- **链接: [http://arxiv.org/pdf/2501.11496v2](http://arxiv.org/pdf/2501.11496v2)**

> **作者:** Vincent Koc
>
> **备注:** 9 pages, 3 figures, 2 tables, submitted for IEEE publication. Pre-print updated as part of review process
>
> **摘要:** The global crisis of language endangerment meets a technological turning point as Generative AI (GenAI) and Large Language Models (LLMs) unlock new frontiers in automating corpus creation, transcription, translation, and tutoring. However, this promise is imperiled by fragmented practices and the critical lack of a methodology to navigate the fraught balance between LLM capabilities and the profound risks of data scarcity, cultural misappropriation, and ethical missteps. This paper introduces a novel analytical framework that systematically evaluates GenAI applications against language-specific needs, embedding community governance and ethical safeguards as foundational pillars. We demonstrate its efficacy through the Te Reo M\=aori revitalization, where it illuminates successes, such as community-led Automatic Speech Recognition achieving 92% accuracy, while critically surfacing persistent challenges in data sovereignty and model bias for digital archives and educational tools. Our findings underscore that GenAI can indeed revolutionize language preservation, but only when interventions are rigorously anchored in community-centric data stewardship, continuous evaluation, and transparent risk management. Ultimately, this framework provides an indispensable toolkit for researchers, language communities, and policymakers, aiming to catalyze the ethical and high-impact deployment of LLMs to safeguard the world's linguistic heritage.
>
---
#### [replaced 162] DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.00248v2](http://arxiv.org/pdf/2407.00248v2)**

> **作者:** Zhenhao Li; Huichi Zhou; Marek Rei; Lucia Specia
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to systems built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. The diffusion layer is trained on top of the existing classifier, ensuring seamless integration with any model in a plug-and-play manner. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over existing adversarial defense methods and achieves state-of-the-art performance against common black-box and white-box adversarial attacks.
>
---
#### [replaced 163] Automatically generating Riddles aiding Concept Attainment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2310.18290v2](http://arxiv.org/pdf/2310.18290v2)**

> **作者:** Niharika Sri Parasa; Chaitali Diwan; Srinath Srinivasa
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** One of the primary challenges in online learning environments, is to retain learner engagement. Several different instructional strategies are proposed both in online and offline environments to enhance learner engagement. The Concept Attainment Model is one such instructional strategy that focuses on learners acquiring a deeper understanding of a concept rather than just its dictionary definition. This is done by searching and listing the properties used to distinguish examples from non-examples of various concepts. Our work attempts to apply the Concept Attainment Model to build conceptual riddles, to deploy over online learning environments. The approach involves creating factual triples from learning resources, classifying them based on their uniqueness to a concept into `Topic Markers' and `Common', followed by generating riddles based on the Concept Attainment Model's format and capturing all possible solutions to those riddles. The results obtained from the human evaluation of riddles prove encouraging.
>
---
#### [replaced 164] Signatures of human-like processing in Transformer forward passes
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14107v2](http://arxiv.org/pdf/2504.14107v2)**

> **作者:** Jennifer Hu; Michael A. Lepori; Michael Franke
>
> **备注:** under review
>
> **摘要:** Modern AI models are increasingly being used as theoretical tools to study human cognition. One dominant approach is to evaluate whether human-derived measures are predicted by a model's output: that is, the end-product of a forward pass. However, recent advances in mechanistic interpretability have begun to reveal the internal processes that give rise to model outputs, raising the question of whether models might use human-like processing strategies. Here, we investigate the relationship between real-time processing in humans and layer-time dynamics of computation in Transformers, testing 20 open-source models in 6 domains. We first explore whether forward passes show mechanistic signatures of competitor interference, taking high-level inspiration from cognitive theories. We find that models indeed appear to initially favor a competing incorrect answer in the cases where we would expect decision conflict in humans. We then systematically test whether forward-pass dynamics predict signatures of processing in humans, above and beyond properties of the model's output probability distribution. We find that dynamic measures improve prediction of human processing measures relative to static final-layer measures. Moreover, across our experiments, larger models do not always show more human-like processing patterns. Our work suggests a new way of using AI models to study human cognition: not just as a black box mapping stimuli to responses, but potentially also as explicit processing models.
>
---
