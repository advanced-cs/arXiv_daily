# 自然语言处理 cs.CL

- **最新发布 190 篇**

- **更新 168 篇**

## 最新发布

#### [new 001] SEED: Enhancing Text-to-SQL Performance and Practical Usability Through Automatic Evidence Generation
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于文本到SQL任务，解决真实场景下证据缺失问题。提出SEED系统自动生成证据，提升模型性能与实用性。**

- **链接: [http://arxiv.org/pdf/2506.07423v1](http://arxiv.org/pdf/2506.07423v1)**

> **作者:** Janghyeon Yun; Sang-goo Lee
>
> **摘要:** Text-to-SQL enables non-experts to retrieve data from databases by converting natural language queries into SQL. However, state-of-the-art text-to-SQL studies rely on the BIRD dataset, which assumes that evidence is provided along with questions. Although BIRD facilitates research advancements, it assumes that users have expertise and domain knowledge, contradicting the fundamental goal of text-to-SQL. In addition, human-generated evidence in BIRD contains defects, including missing or erroneous evidence, which affects model performance. To address this issue, we propose SEED (System for Evidence Extraction and Domain knowledge generation), an approach that automatically generates evidence to improve performance and practical usability in real-world scenarios. SEED systematically analyzes database schema, description files, and values to extract relevant information. We evaluated SEED on BIRD and Spider, demonstrating that it significantly improves SQL generation accuracy in the no-evidence scenario, and in some cases, even outperforms the setting where BIRD evidence is provided. Our results highlight that SEED-generated evidence not only bridges the gap between research and real-world deployment but also improves the adaptability and robustness of text-to-SQL models. Our code is available at https://github.com/felix01189/SEED
>
---
#### [new 002] From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的不确定性量化研究，旨在提升LLM与人类协作的可靠性。论文指出当前UQ方法存在的问题，并提出以用户为中心的改进方向。**

- **链接: [http://arxiv.org/pdf/2506.07461v1](http://arxiv.org/pdf/2506.07461v1)**

> **作者:** Siddartha Devic; Tejas Srinivasan; Jesse Thomason; Willie Neiswanger; Vatsal Sharan
>
> **摘要:** Large Language Models (LLMs) are increasingly assisting users in the real world, yet their reliability remains a concern. Uncertainty quantification (UQ) has been heralded as a tool to enhance human-LLM collaboration by enabling users to know when to trust LLM predictions. We argue that current practices for uncertainty quantification in LLMs are not optimal for developing useful UQ for human users making decisions in real-world tasks. Through an analysis of 40 LLM UQ methods, we identify three prevalent practices hindering the community's progress toward its goal of benefiting downstream users: 1) evaluating on benchmarks with low ecological validity; 2) considering only epistemic uncertainty; and 3) optimizing metrics that are not necessarily indicative of downstream utility. For each issue, we propose concrete user-centric practices and research directions that LLM UQ researchers should consider. Instead of hill-climbing on unrepresentative tasks using imperfect metrics, we argue that the community should adopt a more human-centered approach to LLM uncertainty quantification.
>
---
#### [new 003] Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于医学多模态理解与推理任务，旨在解决医疗MLLM知识覆盖不足、易幻觉和推理能力弱的问题。通过数据增强和模型优化，提出Lingshu模型并构建评估框架。**

- **链接: [http://arxiv.org/pdf/2506.07044v1](http://arxiv.org/pdf/2506.07044v1)**

> **作者:** LASA Team; Weiwen Xu; Hou Pong Chan; Long Li; Mahani Aljunied; Ruifeng Yuan; Jianyu Wang; Chenghao Xiao; Guizhen Chen; Chaoqun Liu; Zhaodonghui Li; Yu Sun; Junao Shen; Chaojun Wang; Jie Tan; Deli Zhao; Tingyang Xu; Hao Zhang; Yu Rong
>
> **备注:** Technical Report, 53 pages, 25 tables, and 16 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ...
>
---
#### [new 004] Reinforcement Pre-Training
- **分类: cs.CL**

- **简介: 该论文提出RPT，用于语言模型预训练，通过强化学习提升下一个词预测能力，解决传统依赖标注数据的问题。**

- **链接: [http://arxiv.org/pdf/2506.08007v1](http://arxiv.org/pdf/2506.08007v1)**

> **作者:** Qingxiu Dong; Li Dong; Yao Tang; Tianzhu Ye; Yutao Sun; Zhifang Sui; Furu Wei
>
> **摘要:** In this work, we introduce Reinforcement Pre-Training (RPT) as a new scaling paradigm for large language models and reinforcement learning (RL). Specifically, we reframe next-token prediction as a reasoning task trained using RL, where it receives verifiable rewards for correctly predicting the next token for a given context. RPT offers a scalable method to leverage vast amounts of text data for general-purpose RL, rather than relying on domain-specific annotated answers. By incentivizing the capability of next-token reasoning, RPT significantly improves the language modeling accuracy of predicting the next tokens. Moreover, RPT provides a strong pre-trained foundation for further reinforcement fine-tuning. The scaling curves show that increased training compute consistently improves the next-token prediction accuracy. The results position RPT as an effective and promising scaling paradigm to advance language model pre-training.
>
---
#### [new 005] Natural Language Interaction with Databases on Edge Devices in the Internet of Battlefield Things
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于自然语言查询数据库任务，旨在解决战场物联网中数据处理与信息提取问题。通过结合边缘设备上的大语言模型和图数据库，实现自然语言交互与准确查询。**

- **链接: [http://arxiv.org/pdf/2506.06396v1](http://arxiv.org/pdf/2506.06396v1)**

> **作者:** Christopher D. Molek; Roberto Fronteddu; K. Brent Venable; Niranjan Suri
>
> **摘要:** The expansion of the Internet of Things (IoT) in the battlefield, Internet of Battlefield Things (IoBT), gives rise to new opportunities for enhancing situational awareness. To increase the potential of IoBT for situational awareness in critical decision making, the data from these devices must be processed into consumer-ready information objects, and made available to consumers on demand. To address this challenge we propose a workflow that makes use of natural language processing (NLP) to query a database technology and return a response in natural language. Our solution utilizes Large Language Models (LLMs) that are sized for edge devices to perform NLP as well as graphical databases which are well suited for dynamic connected networks which are pervasive in the IoBT. Our architecture employs LLMs for both mapping questions in natural language to Cypher database queries as well as to summarize the database output back to the user in natural language. We evaluate several medium sized LLMs for both of these tasks on a database representing publicly available data from the US Army's Multipurpose Sensing Area (MSA) at the Jornada Range in Las Cruces, NM. We observe that Llama 3.1 (8 billion parameters) outperforms the other models across all the considered metrics. Most importantly, we note that, unlike current methods, our two step approach allows the relaxation of the Exact Match (EM) requirement of the produced Cypher queries with ground truth code and, in this way, it achieves a 19.4% increase in accuracy. Our workflow lays the ground work for deploying LLMs on edge devices to enable natural language interactions with databases containing information objects for critical decision making.
>
---
#### [new 006] They want to pretend not to understand: The Limits of Current LLMs in Interpreting Implicit Content of Political Discourse
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语用理解任务，旨在解决LLMs在解读政治话语中隐含内容（如预设和暗示）的能力问题。研究使用IMPAQTS语料库测试模型表现，发现当前模型存在显著不足。**

- **链接: [http://arxiv.org/pdf/2506.06775v1](http://arxiv.org/pdf/2506.06775v1)**

> **作者:** Walter Paci; Alessandro Panunzi; Sandro Pezzelle
>
> **备注:** Accepted to the ACL2025 Findings
>
> **摘要:** Implicit content plays a crucial role in political discourse, where speakers systematically employ pragmatic strategies such as implicatures and presuppositions to influence their audiences. Large Language Models (LLMs) have demonstrated strong performance in tasks requiring complex semantic and pragmatic understanding, highlighting their potential for detecting and explaining the meaning of implicit content. However, their ability to do this within political discourse remains largely underexplored. Leveraging, for the first time, the large IMPAQTS corpus, which comprises Italian political speeches with the annotation of manipulative implicit content, we propose methods to test the effectiveness of LLMs in this challenging problem. Through a multiple-choice task and an open-ended generation task, we demonstrate that all tested models struggle to interpret presuppositions and implicatures. We conclude that current LLMs lack the key pragmatic capabilities necessary for accurately interpreting highly implicit language, such as that found in political discourse. At the same time, we highlight promising trends and future directions for enhancing model performance. We release our data and code at https://github.com/WalterPaci/IMPAQTS-PID
>
---
#### [new 007] Not quite Sherlock Holmes: Language model predictions do not reliably differentiate impossible from improbable events
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，探讨语言模型是否能区分不可能与不太可能的事件。研究发现，模型在某些情况下表现不佳，甚至低于随机水平。**

- **链接: [http://arxiv.org/pdf/2506.06808v1](http://arxiv.org/pdf/2506.06808v1)**

> **作者:** James A. Michaelov; Reeka Estacio; Zhien Zhang; Benjamin K. Bergen
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Can language models reliably predict that possible events are more likely than merely improbable ones? By teasing apart possibility, typicality, and contextual relatedness, we show that despite the results of previous work, language models' ability to do this is far from robust. In fact, under certain conditions, all models tested - including Llama 3, Gemma 2, and Mistral NeMo - perform at worse-than-chance level, assigning higher probabilities to impossible sentences such as 'the car was given a parking ticket by the brake' than to merely unlikely sentences such as 'the car was given a parking ticket by the explorer'.
>
---
#### [new 008] Beyond Benchmarks: A Novel Framework for Domain-Specific LLM Evaluation and Knowledge Mapping
- **分类: cs.CL; I.2.7; I.2.6**

- **简介: 该论文属于语言模型评估任务，解决领域基准构建与知识表示问题。提出一种新框架，通过自动转换语料生成评测基准，评估模型领域知识并分析适应过程中的知识遗忘。**

- **链接: [http://arxiv.org/pdf/2506.07658v1](http://arxiv.org/pdf/2506.07658v1)**

> **作者:** Nitin Sharma; Thomas Wolfers; Çağatay Yıldız
>
> **备注:** 35 pages, 24 figures. First submission
>
> **摘要:** The paper addresses two critical challenges in language model (LM) evaluation: creating reliable domain-specific benchmarks and understanding knowledge representation during domain adaptation. We introduce a deterministic pipeline that converts raw domain corpora into completion-type benchmarks without relying on LMs or human curation, eliminating benchmark contamination issues while enabling evaluation on the latest domain data. Our approach generates domain-specific keywords and related word lists using TF and Term TF-IDF methods and constructs prompt-target pairs. We evaluate models by measuring their ability to complete these prompts with the correct domain-specific targets, providing a direct assessment of domain knowledge with low computational cost. Through comprehensive experiments across multiple models (GPT-2 medium/XL, Llama-2/3.1, OLMo-2, Qwen-2, Mistral) and domains, we demonstrate that our benchmark strongly correlates with expert-generated benchmarks while providing a more accurate measure of domain knowledge than traditional perplexity metrics. We reveal that domain adaptation happens rapidly in smaller models (within 500 steps) and illustrate a new approach to domain knowledge evaluation in base models during training for early stopping. By extending mechanistic analysis to domain adaptation, we discover that initial-to-mid layers are primarily responsible for attribute extraction, while later layers focus on next token prediction. Furthermore, we show that during adaptation, forgetting begins in the middle layers, where attribute extraction happens and is amplified in later layers. Our work provides both a practical evaluation methodology for domain-specific LMs and novel insights into knowledge representation during adaptation, with implications for more efficient fine-tuning strategies and targeted approaches to mitigate catastrophic forgetting.
>
---
#### [new 009] Intent Matters: Enhancing AI Tutoring with Fine-Grained Pedagogical Intent Annotation
- **分类: cs.CL**

- **简介: 该论文属于教育AI任务，旨在提升智能导师系统的教学效果。通过细粒度教学意图标注，优化LLM生成的辅导回复质量。**

- **链接: [http://arxiv.org/pdf/2506.07626v1](http://arxiv.org/pdf/2506.07626v1)**

> **作者:** Kseniia Petukhova; Ekaterina Kochmar
>
> **摘要:** Large language models (LLMs) hold great promise for educational applications, particularly in intelligent tutoring systems. However, effective tutoring requires alignment with pedagogical strategies - something current LLMs lack without task-specific adaptation. In this work, we explore whether fine-grained annotation of teacher intents can improve the quality of LLM-generated tutoring responses. We focus on MathDial, a dialog dataset for math instruction, and apply an automated annotation framework to re-annotate a portion of the dataset using a detailed taxonomy of eleven pedagogical intents. We then fine-tune an LLM using these new annotations and compare its performance to models trained on the original four-category taxonomy. Both automatic and qualitative evaluations show that the fine-grained model produces more pedagogically aligned and effective responses. Our findings highlight the value of intent specificity for controlled text generation in educational settings, and we release our annotated data and code to facilitate further research.
>
---
#### [new 010] LLM Unlearning Should Be Form-Independent
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于自然语言处理中的模型遗忘任务，旨在解决现有方法对训练样本形式依赖的问题。研究提出新基准和方法，提升模型在不同表达下的遗忘效果。**

- **链接: [http://arxiv.org/pdf/2506.07795v1](http://arxiv.org/pdf/2506.07795v1)**

> **作者:** Xiaotian Ye; Mengqi Zhang; Shu Wu
>
> **摘要:** Large Language Model (LLM) unlearning aims to erase or suppress undesirable knowledge within the model, offering promise for controlling harmful or private information to prevent misuse. However, recent studies highlight its limited efficacy in real-world scenarios, hindering practical adoption. In this study, we identify a pervasive issue underlying many downstream failures: the effectiveness of existing unlearning methods heavily depends on the form of training samples and frequently fails to generalize to alternate expressions of the same knowledge. We formally characterize this problem as Form-Dependent Bias and systematically investigate its specific manifestation patterns across various downstream tasks. To quantify its prevalence and support future research, we introduce ORT, a novel benchmark designed to evaluate the robustness of unlearning methods against variations in knowledge expression. Results reveal that Form-Dependent Bias is both widespread and severe among current techniques. We argue that LLM unlearning should be form-independent to address the endless forms of downstream tasks encountered in real-world security-critical scenarios. Towards this goal, we introduce Rank-one Concept Redirection (ROCR), a novel training-free method, as a promising solution path. ROCR performs unlearning by targeting the invariants in downstream tasks, specifically the activated dangerous concepts. It is capable of modifying model parameters within seconds to redirect the model's perception of a specific unlearning target concept to another harmless concept. Extensive experiments demonstrate that ROCR significantly improves unlearning effectiveness compared to traditional methods while generating highly natural outputs.
>
---
#### [new 011] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决AI生成文本被检测的问题。提出Adversarial Paraphrasing方法，通过对抗性改写绕过检测系统。**

- **链接: [http://arxiv.org/pdf/2506.07001v1](http://arxiv.org/pdf/2506.07001v1)**

> **作者:** Yize Cheng; Vinu Sankar Sadasivan; Mehrdad Saberi; Shoumik Saha; Soheil Feizi
>
> **摘要:** The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.
>
---
#### [new 012] Canonical Autoregressive Generation
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于自然语言生成任务，旨在解决模型生成非规范token序列的问题。提出“规范采样”方法，确保生成序列符合训练时的规范分词方式。**

- **链接: [http://arxiv.org/pdf/2506.06446v1](http://arxiv.org/pdf/2506.06446v1)**

> **作者:** Ivi Chatzi; Nina Corvelo Benz; Stratis Tsirtsis; Manuel Gomez-Rodriguez
>
> **摘要:** State of the art large language models are trained using large amounts of tokens derived from raw text using what is called a tokenizer. Crucially, the tokenizer determines the (token) vocabulary a model will use during inference as well as, in principle, the (token) language. This is because, while the token vocabulary may allow for different tokenizations of a string, the tokenizer always maps the string to only one of these tokenizations--the canonical tokenization. However, multiple lines of empirical evidence suggest that large language models do not always generate canonical token sequences, and this comes with several negative consequences. In this work, we first show that, to generate a canonical token sequence, a model needs to generate (partial) canonical token sequences at each step of the autoregressive generation process underpinning its functioning. Building upon this theoretical result, we introduce canonical sampling, a simple and efficient sampling method that precludes a given model from generating non-canonical token sequences. Further, we also show that, in comparison with standard sampling, the distribution of token sequences generated using canonical sampling is provably closer to the true distribution of token sequences used during training.
>
---
#### [new 013] Reasoning with RAGged events: RAG-Enhanced Event Knowledge Base Construction and reasoning with proof-assistants
- **分类: cs.CL**

- **简介: 该论文属于历史事件知识库构建任务，旨在解决手动构建成本高和逻辑分析受限的问题。通过多模型增强策略提取事件，并用Coq实现高级推理。**

- **链接: [http://arxiv.org/pdf/2506.07042v1](http://arxiv.org/pdf/2506.07042v1)**

> **作者:** Stergios Chatzikyriakidis
>
> **摘要:** Extracting structured computational representations of historical events from narrative text remains computationally expensive when constructed manually. While RDF/OWL reasoners enable graph-based reasoning, they are limited to fragments of first-order logic, preventing deeper temporal and semantic analysis. This paper addresses both challenges by developing automatic historical event extraction models using multiple LLMs (GPT-4, Claude, Llama 3.2) with three enhancement strategies: pure base generation, knowledge graph enhancement, and Retrieval-Augmented Generation (RAG). We conducted comprehensive evaluations using historical texts from Thucydides. Our findings reveal that enhancement strategies optimize different performance dimensions rather than providing universal improvements. For coverage and historical breadth, base generation achieves optimal performance with Claude and GPT-4 extracting comprehensive events. However, for precision, RAG enhancement improves coordinate accuracy and metadata completeness. Model architecture fundamentally determines enhancement sensitivity: larger models demonstrate robust baseline performance with incremental RAG improvements, while Llama 3.2 shows extreme variance from competitive performance to complete failure. We then developed an automated translation pipeline converting extracted RDF representations into Coq proof assistant specifications, enabling higher-order reasoning beyond RDF capabilities including multi-step causal verification, temporal arithmetic with BC dates, and formal proofs about historical causation. The Coq formalization validates that RAG-discovered event types represent legitimate domain-specific semantic structures rather than ontological violations.
>
---
#### [new 014] KScope: A Framework for Characterizing the Knowledge Status of Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型知识状态分析任务，旨在解决如何准确评估LLM对问题的掌握程度。提出KScope框架，通过统计测试分类知识状态，提升模型知识更新效果。**

- **链接: [http://arxiv.org/pdf/2506.07458v1](http://arxiv.org/pdf/2506.07458v1)**

> **作者:** Yuxin Xiao; Shan Chen; Jack Gallifant; Danielle Bitterman; Thomas Hartvigsen; Marzyeh Ghassemi
>
> **摘要:** Characterizing a large language model's (LLM's) knowledge of a given question is challenging. As a result, prior work has primarily examined LLM behavior under knowledge conflicts, where the model's internal parametric memory contradicts information in the external context. However, this does not fully reflect how well the model knows the answer to the question. In this paper, we first introduce a taxonomy of five knowledge statuses based on the consistency and correctness of LLM knowledge modes. We then propose KScope, a hierarchical framework of statistical tests that progressively refines hypotheses about knowledge modes and characterizes LLM knowledge into one of these five statuses. We apply KScope to nine LLMs across four datasets and systematically establish: (1) Supporting context narrows knowledge gaps across models. (2) Context features related to difficulty, relevance, and familiarity drive successful knowledge updates. (3) LLMs exhibit similar feature preferences when partially correct or conflicted, but diverge sharply when consistently wrong. (4) Context summarization constrained by our feature analysis, together with enhanced credibility, further improves update effectiveness and generalizes across LLMs.
>
---
#### [new 015] SELT: Self-Evaluation Tree Search for LLMs with Task Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SELT框架，用于提升大语言模型在复杂推理任务中的表现。通过改进树搜索算法，实现任务分解与自我评估，增强推理准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.07557v1](http://arxiv.org/pdf/2506.07557v1)**

> **作者:** Mengsong Wu; Di Zhang; Yuqiang Li; Dongzhan Zhou; Wenliang Chen
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** While Large Language Models (LLMs) have achieved remarkable success in a wide range of applications, their performance often degrades in complex reasoning tasks. In this work, we introduce SELT (Self-Evaluation LLM Tree Search), a novel framework that leverages a modified Monte Carlo Tree Search (MCTS) to enhance LLM reasoning without relying on external reward models. By redefining the Upper Confidence Bound scoring to align with intrinsic self-evaluation capabilities of LLMs and decomposing the inference process into atomic subtasks augmented with semantic clustering at each node, SELT effectively balances exploration and exploitation, reduces redundant reasoning paths, and mitigates hallucination. We validate our approach on challenging benchmarks, including the knowledge-based MMLU and the Tool Learning dataset Seal-Tools, where SELT achieves significant improvements in answer accuracy and reasoning robustness compared to baseline methods. Notably, our framework operates without task-specific fine-tuning, demonstrating strong generalizability across diverse reasoning tasks. Relevant results and code are available at https://github.com/fairyshine/SELT .
>
---
#### [new 016] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG评估任务，旨在解决LLM在生成回答时准确引用相关来源的问题。工作包括构建包含标注片段的基准数据集GaRAGe，并评估模型的引用能力。**

- **链接: [http://arxiv.org/pdf/2506.07671v1](http://arxiv.org/pdf/2506.07671v1)**

> **作者:** Ionut-Teodor Sorodoc; Leonardo F. R. Ribeiro; Rexhina Blloshmi; Christopher Davis; Adrià de Gispert
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** We present GaRAGe, a large RAG benchmark with human-curated long-form answers and annotations of each grounding passage, allowing a fine-grained evaluation of whether LLMs can identify relevant grounding when generating RAG answers. Our benchmark contains 2366 questions of diverse complexity, dynamism, and topics, and includes over 35K annotated passages retrieved from both private document sets and the Web, to reflect real-world RAG use cases. This makes it an ideal test bed to evaluate an LLM's ability to identify only the relevant information necessary to compose a response, or provide a deflective response when there is insufficient information. Evaluations of multiple state-of-the-art LLMs on GaRAGe show that the models tend to over-summarise rather than (a) ground their answers strictly on the annotated relevant passages (reaching at most a Relevance-Aware Factuality Score of 60%), or (b) deflect when no relevant grounding is available (reaching at most 31% true positive rate in deflections). The F1 in attribution to relevant sources is at most 58.9%, and we show that performance is particularly reduced when answering time-sensitive questions and when having to draw knowledge from sparser private grounding sources.
>
---
#### [new 017] Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque
- **分类: cs.CL**

- **简介: 该论文属于低资源语言建模任务，旨在解决缺乏大规模指令数据的问题。通过合成指令和预训练模型提升Basque语言模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07597v1](http://arxiv.org/pdf/2506.07597v1)**

> **作者:** Oscar Sainz; Naiara Perez; Julen Etxaniz; Joseba Fernandez de Landa; Itziar Aldabe; Iker García-Ferrero; Aimar Zabala; Ekhi Azurmendi; German Rigau; Eneko Agirre; Mikel Artetxe; Aitor Soroa
>
> **备注:** Under review
>
> **摘要:** Instructing language models with user intent requires large instruction datasets, which are only available for a limited set of languages. In this paper, we explore alternatives to conventional instruction adaptation pipelines in low-resource scenarios. We assume a realistic scenario for low-resource languages, where only the following are available: corpora in the target language, existing open-weight multilingual base and instructed backbone LLMs, and synthetically generated instructions sampled from the instructed backbone. We present a comprehensive set of experiments for Basque that systematically study different combinations of these components evaluated on benchmarks and human preferences from 1,680 participants. Our conclusions show that target language corpora are essential, with synthetic instructions yielding robust models, and, most importantly, that using as backbone an instruction-tuned model outperforms using a base non-instructed model, and improved results when scaling up. Using Llama 3.1 instruct 70B as backbone our model comes near frontier models of much larger sizes for Basque, without using any Basque data apart from the 1.2B word corpora. We release code, models, instruction datasets, and human preferences to support full reproducibility in future research on low-resource language adaptation.
>
---
#### [new 018] DEBATE: A Dataset for Disentangling Textual Ambiguity in Mandarin Through Speech
- **分类: cs.CL**

- **简介: 该论文提出DEBATE数据集，用于研究中文语音歧义消解。任务是通过语音线索解决文本歧义，填补DTS研究空白。**

- **链接: [http://arxiv.org/pdf/2506.07502v1](http://arxiv.org/pdf/2506.07502v1)**

> **作者:** Haotian Guo; Jing Han; Yongfeng Tu; Shihao Gao; Shengfan Shen; Wulong Xiang; Weihao Gan; Zixing Zhang
>
> **摘要:** Despite extensive research on textual and visual disambiguation, disambiguation through speech (DTS) remains underexplored. This is largely due to the lack of high-quality datasets that pair spoken sentences with richly ambiguous text. To address this gap, we present DEBATE, a unique public Chinese speech-text dataset designed to study how speech cues and patterns-pronunciation, pause, stress and intonation-can help resolve textual ambiguity and reveal a speaker's true intent. DEBATE contains 1,001 carefully selected ambiguous utterances, each recorded by 10 native speakers, capturing diverse linguistic ambiguities and their disambiguation through speech. We detail the data collection pipeline and provide rigorous quality analysis. Additionally, we benchmark three state-of-the-art large speech and audio-language models, illustrating clear and huge performance gaps between machine and human understanding of spoken intent. DEBATE represents the first effort of its kind and offers a foundation for building similar DTS datasets across languages and cultures. The dataset and associated code are available at: https://github.com/SmileHnu/DEBATE.
>
---
#### [new 019] Interpretable Depression Detection from Social Media Text Using LLM-Derived Embeddings
- **分类: cs.CL**

- **简介: 该论文属于抑郁症检测任务，旨在通过社交媒体文本进行可解释的抑郁识别。研究比较了LLM和传统分类器在不同分类任务中的表现，探索了LLM生成嵌入的有效性。**

- **链接: [http://arxiv.org/pdf/2506.06616v1](http://arxiv.org/pdf/2506.06616v1)**

> **作者:** Samuel Kim; Oghenemaro Imieye; Yunting Yin
>
> **备注:** Submitted to the IEEE EMBS BHI 2025 Conference
>
> **摘要:** Accurate and interpretable detection of depressive language in social media is useful for early interventions of mental health conditions, and has important implications for both clinical practice and broader public health efforts. In this paper, we investigate the performance of large language models (LLMs) and traditional machine learning classifiers across three classification tasks involving social media data: binary depression classification, depression severity classification, and differential diagnosis classification among depression, PTSD, and anxiety. Our study compares zero-shot LLMs with supervised classifiers trained on both conventional text embeddings and LLM-generated summary embeddings. Our experiments reveal that while zero-shot LLMs demonstrate strong generalization capabilities in binary classification, they struggle with fine-grained ordinal classifications. In contrast, classifiers trained on summary embeddings generated by LLMs demonstrate competitive, and in some cases superior, performance on the classification tasks, particularly when compared to models using traditional text embeddings. Our findings demonstrate the strengths of LLMs in mental health prediction, and suggest promising directions for better utilization of their zero-shot capabilities and context-aware summarization techniques.
>
---
#### [new 020] LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化图表描述生成任务，旨在解决AI生成描述与作者风格不匹配的问题。工作包括构建多模态图表资料集LaMP-Cap，并验证多模态信息提升生成质量的效果。**

- **链接: [http://arxiv.org/pdf/2506.06561v1](http://arxiv.org/pdf/2506.06561v1)**

> **作者:** Ho Yin 'Sam' Ng; Ting-Yao Hsu; Aashish Anantha Ramakrishnan; Branislav Kveton; Nedim Lipka; Franck Dernoncourt; Dongwon Lee; Tong Yu; Sungchul Kim; Ryan A. Rossi; Ting-Hao 'Kenneth' Huang
>
> **摘要:** Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones.
>
---
#### [new 021] Improving Fairness of Large Language Models in Multi-document Summarization
- **分类: cs.CL**

- **简介: 该论文属于多文档摘要任务，旨在解决摘要中公平性问题。提出FairPO方法，同时优化摘要级和语料库级公平性。**

- **链接: [http://arxiv.org/pdf/2506.07479v1](http://arxiv.org/pdf/2506.07479v1)**

> **作者:** Haoyuan Li Yusen Zhang; Snigdha Chaturvedi
>
> **备注:** Accepted to ACL 2025 main
>
> **摘要:** Fairness in multi-document summarization (MDS) is crucial for providing comprehensive views across documents with diverse social attribute values, which can significantly impact decision-making. For example, a summarization system that tends to overrepresent negative reviews of products can mislead customers into disregarding good products. Previous works measure fairness in MDS at two levels: summary-level and corpus-level. While summary-level fairness focuses on individual summaries, corpus-level fairness focuses on a corpus of summaries. Recent methods primarily focus on summary-level fairness. We propose FairPO, a preference tuning method that focuses on both summary-level and corpus-level fairness in MDS. To improve summary-level fairness, we propose to generate preference pairs by perturbing document sets. To improve corpus-level fairness, we propose fairness-aware preference tuning by dynamically adjusting the weights of preference pairs. Our experiments show that FairPO outperforms strong baselines while maintaining the critical qualities of summaries. The code is available at https://github.com/leehaoyuan/coverage_fairnes.
>
---
#### [new 022] SMAR: Soft Modality-Aware Routing Strategy for MoE-based Multimodal Large Language Models Preserving Language Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态大语言模型任务，解决MoE模型在多模态扩展中语言能力下降的问题，提出SMAR方法提升模态区分与语言能力平衡。**

- **链接: [http://arxiv.org/pdf/2506.06406v1](http://arxiv.org/pdf/2506.06406v1)**

> **作者:** Guoyang Xia; Yifeng Ding; Fengfa Li; Lei Ren; Chen Wei; Fangxiang Feng; Xiaojie Wang
>
> **摘要:** Mixture of Experts (MoE) architectures have become a key approach for scaling large language models, with growing interest in extending them to multimodal tasks. Existing methods to build multimodal MoE models either incur high training costs or suffer from degraded language capabilities when adapting pretrained models. To address this, we propose Soft ModalityAware Routing (SMAR), a novel regularization technique that uses Kullback Leibler divergence to control routing probability distributions across modalities, encouraging expert specialization without modifying model architecture or heavily relying on textual data. Experiments on visual instruction tuning show that SMAR preserves language ability at 86.6% retention with only 2.5% pure text, outperforming baselines while maintaining strong multimodal performance. Our approach offers a practical and efficient solution to balance modality differentiation and language capabilities in multimodal MoE models.
>
---
#### [new 023] Improving the Efficiency of Long Document Classification using Sentence Ranking Approach
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于长文档分类任务，旨在解决Transformer模型处理长文本效率低的问题。通过TF-IDF句排序方法选择关键句子，提升效率并保持准确率。**

- **链接: [http://arxiv.org/pdf/2506.07248v1](http://arxiv.org/pdf/2506.07248v1)**

> **作者:** Prathamesh Kokate; Mitali Sarnaik; Manavi Khopade; Raviraj Joshi
>
> **摘要:** Long document classification poses challenges due to the computational limitations of transformer-based models, particularly BERT, which are constrained by fixed input lengths and quadratic attention complexity. Moreover, using the full document for classification is often redundant, as only a subset of sentences typically carries the necessary information. To address this, we propose a TF-IDF-based sentence ranking method that improves efficiency by selecting the most informative content. Our approach explores fixed-count and percentage-based sentence selection, along with an enhanced scoring strategy combining normalized TF-IDF scores and sentence length. Evaluated on the MahaNews LDC dataset of long Marathi news articles, the method consistently outperforms baselines such as first, last, and random sentence selection. With MahaBERT-v2, we achieve near-identical classification accuracy with just a 0.33 percent drop compared to the full-context baseline, while reducing input size by over 50 percent and inference latency by 43 percent. This demonstrates that significant context reduction is possible without sacrificing performance, making the method practical for real-world long document classification tasks.
>
---
#### [new 024] Precise Information Control in Long-Form Text Generation
- **分类: cs.CL**

- **简介: 该论文属于语言模型信息控制任务，旨在解决模型生成内容中的幻觉问题。通过提出PIC框架和基准测试，评估并提升模型生成的准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2506.06589v1](http://arxiv.org/pdf/2506.06589v1)**

> **作者:** Jacqueline He; Howard Yen; Margaret Li; Shuyue Stella Li; Zhiyuan Zeng; Weijia Shi; Yulia Tsvetkov; Danqi Chen; Pang Wei Koh; Luke Zettlemoyer
>
> **备注:** 56 pages, 8 figures. Code and models are publicly available at https://github.com/jacqueline-he/precise-information-control
>
> **摘要:** A central challenge in modern language models (LMs) is intrinsic hallucination: the generation of information that is plausible but unsubstantiated relative to input context. To study this problem, we propose Precise Information Control (PIC), a new task formulation that requires models to generate long-form outputs grounded in a provided set of short self-contained statements, known as verifiable claims, without adding any unsupported ones. For comprehensiveness, PIC includes a full setting that tests a model's ability to include exactly all input claims, and a partial setting that requires the model to selectively incorporate only relevant claims. We present PIC-Bench, a benchmark of eight long-form generation tasks (e.g., summarization, biography generation) adapted to the PIC setting, where LMs are supplied with well-formed, verifiable input claims. Our evaluation of a range of open and proprietary LMs on PIC-Bench reveals that, surprisingly, state-of-the-art LMs still intrinsically hallucinate in over 70% of outputs. To alleviate this lack of faithfulness, we introduce a post-training framework, using a weakly supervised preference data construction method, to train an 8B PIC-LM with stronger PIC ability--improving from 69.1% to 91.0% F1 in the full PIC setting. When integrated into end-to-end factual generation pipelines, PIC-LM improves exact match recall by 17.1% on ambiguous QA with retrieval, and factual precision by 30.5% on a birthplace verification task, underscoring the potential of precisely grounded generation.
>
---
#### [new 025] Exploring the Impact of Temperature on Large Language Models:Hot or Cold?
- **分类: cs.CL**

- **简介: 该论文研究温度对大语言模型性能的影响，旨在优化推理过程中的随机性控制。通过实验分析不同温度下的模型表现，并提出基于BERT的温度选择器以提升小中型模型效果。**

- **链接: [http://arxiv.org/pdf/2506.07295v1](http://arxiv.org/pdf/2506.07295v1)**

> **作者:** Lujun Li; Lama Sleem; Niccolo' Gentile; Geoffrey Nichil; Radu State
>
> **摘要:** The sampling temperature, a critical hyperparameter in large language models (LLMs), modifies the logits before the softmax layer, thereby reshaping the distribution of output tokens. Recent studies have challenged the Stochastic Parrots analogy by demonstrating that LLMs are capable of understanding semantics rather than merely memorizing data and that randomness, modulated by sampling temperature, plays a crucial role in model inference. In this study, we systematically evaluated the impact of temperature in the range of 0 to 2 on data sets designed to assess six different capabilities, conducting statistical analyses on open source models of three different sizes: small (1B--4B), medium (6B--13B), and large (40B--80B). Our findings reveal distinct skill-specific effects of temperature on model performance, highlighting the complexity of optimal temperature selection in practical applications. To address this challenge, we propose a BERT-based temperature selector that takes advantage of these observed effects to identify the optimal temperature for a given prompt. We demonstrate that this approach can significantly improve the performance of small and medium models in the SuperGLUE datasets. Furthermore, our study extends to FP16 precision inference, revealing that temperature effects are consistent with those observed in 4-bit quantized models. By evaluating temperature effects up to 4.0 in three quantized models, we find that the Mutation Temperature -- the point at which significant performance changes occur -- increases with model size.
>
---
#### [new 026] Quantile Regression with Large Language Models for Price Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于价格预测任务，解决传统方法仅提供点估计的问题，提出基于大语言模型的分位数回归方法，生成完整预测分布。**

- **链接: [http://arxiv.org/pdf/2506.06657v1](http://arxiv.org/pdf/2506.06657v1)**

> **作者:** Nikhita Vedula; Dushyanta Dhyani; Laleh Jalali; Boris Oreshkin; Mohsen Bayati; Shervin Malmasi
>
> **备注:** Accepted to Findings of ACL, 2025
>
> **摘要:** Large Language Models (LLMs) have shown promise in structured prediction tasks, including regression, but existing approaches primarily focus on point estimates and lack systematic comparison across different methods. We investigate probabilistic regression using LLMs for unstructured inputs, addressing challenging text-to-distribution prediction tasks such as price estimation where both nuanced text understanding and uncertainty quantification are critical. We propose a novel quantile regression approach that enables LLMs to produce full predictive distributions, improving upon traditional point estimates. Through extensive experiments across three diverse price prediction datasets, we demonstrate that a Mistral-7B model fine-tuned with quantile heads significantly outperforms traditional approaches for both point and distributional estimations, as measured by three established metrics each for prediction accuracy and distributional calibration. Our systematic comparison of LLM approaches, model architectures, training approaches, and data scaling reveals that Mistral-7B consistently outperforms encoder architectures, embedding-based methods, and few-shot learning methods. Our experiments also reveal the effectiveness of LLM-assisted label correction in achieving human-level accuracy without systematic bias. Our curated datasets are made available at https://github.com/vnik18/llm-price-quantile-reg/ to support future research.
>
---
#### [new 027] MedCite: Can Language Models Generate Verifiable Text for Medicine?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学问答任务，旨在解决LLM生成可验证文本（如引用）的问题。提出MedCite框架和多阶段检索引文方法，提升引用质量与准确性。**

- **链接: [http://arxiv.org/pdf/2506.06605v1](http://arxiv.org/pdf/2506.06605v1)**

> **作者:** Xiao Wang; Mengjue Tan; Qiao Jin; Guangzhi Xiong; Yu Hu; Aidong Zhang; Zhiyong Lu; Minjia Zhang
>
> **摘要:** Existing LLM-based medical question-answering systems lack citation generation and evaluation capabilities, raising concerns about their adoption in practice. In this work, we introduce \name, the first end-to-end framework that facilitates the design and evaluation of citation generation with LLMs for medical tasks. Meanwhile, we introduce a novel multi-pass retrieval-citation method that generates high-quality citations. Our evaluation highlights the challenges and opportunities of citation generation for medical tasks, while identifying important design choices that have a significant impact on the final citation quality. Our proposed method achieves superior citation precision and recall improvements compared to strong baseline methods, and we show that evaluation results correlate well with annotation results from professional experts.
>
---
#### [new 028] Learning to Clarify by Reinforcement Learning Through Reward-Weighted Fine-Tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于问答任务，旨在提升QA代理的澄清能力。通过强化学习模拟对话并优化奖励，解决传统方法参数多、不直接优化奖励的问题。**

- **链接: [http://arxiv.org/pdf/2506.06964v1](http://arxiv.org/pdf/2506.06964v1)**

> **作者:** Subhojyoti Mukherjee; Viet Dac Lai; Raghavendra Addanki; Ryan Rossi; Seunghyun Yoon; Trung Bui; Anup Rao; Jayakumar Subramanian; Branislav Kveton
>
> **备注:** 39 pages
>
> **摘要:** Question answering (QA) agents automatically answer questions posed in natural language. In this work, we learn to ask clarifying questions in QA agents. The key idea in our method is to simulate conversations that contain clarifying questions and learn from them using reinforcement learning (RL). To make RL practical, we propose and analyze offline RL objectives that can be viewed as reward-weighted supervised fine-tuning (SFT) and easily optimized in large language models. Our work stands in a stark contrast to recently proposed methods, based on SFT and direct preference optimization, which have additional hyper-parameters and do not directly optimize rewards. We compare to these methods empirically and report gains in both optimized rewards and language quality.
>
---
#### [new 029] Language Models over Canonical Byte-Pair Encodings
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决语言模型中非规范编码概率分配问题，通过条件推理和构造方法确保仅生成规范token序列。**

- **链接: [http://arxiv.org/pdf/2506.07956v1](http://arxiv.org/pdf/2506.07956v1)**

> **作者:** Tim Vieira; Tianyu Liu; Clemente Pasti; Yahya Emara; Brian DuSell; Benjamin LeBrun; Mario Giulianelli; Juan Luis Gastaldi; Timothy J. O'Donnell; Ryan Cotterell
>
> **备注:** ICML 2025
>
> **摘要:** Modern language models represent probability distributions over character strings as distributions over (shorter) token strings derived via a deterministic tokenizer, such as byte-pair encoding. While this approach is highly effective at scaling up language models to large corpora, its current incarnations have a concerning property: the model assigns nonzero probability mass to an exponential number of $\it{noncanonical}$ token encodings of each character string -- these are token strings that decode to valid character strings but are impossible under the deterministic tokenizer (i.e., they will never be seen in any training corpus, no matter how large). This misallocation is both erroneous, as noncanonical strings never appear in training data, and wasteful, diverting probability mass away from plausible outputs. These are avoidable mistakes! In this work, we propose methods to enforce canonicality in token-level language models, ensuring that only canonical token strings are assigned positive probability. We present two approaches: (1) canonicality by conditioning, leveraging test-time inference strategies without additional training, and (2) canonicality by construction, a model parameterization that guarantees canonical outputs but requires training. We demonstrate that fixing canonicality mistakes improves the likelihood of held-out data for several models and corpora.
>
---
#### [new 030] Transcript-Prompted Whisper with Dictionary-Enhanced Decoding for Japanese Speech Annotation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音标注任务，旨在提高日语语音合成数据的准确性。通过微调ASR模型并结合词典知识，提升音素和韵律标注质量。**

- **链接: [http://arxiv.org/pdf/2506.07646v1](http://arxiv.org/pdf/2506.07646v1)**

> **作者:** Rui Hu; Xiaolong Lin; Jiawang Liu; Shixi Huang; Zhenpeng Zhan
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this paper, we propose a method for annotating phonemic and prosodic labels on a given audio-transcript pair, aimed at constructing Japanese text-to-speech (TTS) datasets. Our approach involves fine-tuning a large-scale pre-trained automatic speech recognition (ASR) model, conditioned on ground truth transcripts, to simultaneously output phrase-level graphemes and annotation labels. To further correct errors in phonemic labeling, we employ a decoding strategy that utilizes dictionary prior knowledge. The objective evaluation results demonstrate that our proposed method outperforms previous approaches relying solely on text or audio. The subjective evaluation results indicate that the naturalness of speech synthesized by the TTS model, trained with labels annotated using our method, is comparable to that of a model trained with manual annotations.
>
---
#### [new 031] C-PATH: Conversational Patient Assistance and Triage in Healthcare System
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话系统任务，旨在解决患者就医引导难题。通过构建C-PATH系统，利用大语言模型进行症状识别与科室推荐。**

- **链接: [http://arxiv.org/pdf/2506.06737v1](http://arxiv.org/pdf/2506.06737v1)**

> **作者:** Qi Shi; Qiwei Han; Cláudia Soares
>
> **备注:** Accepted in IEEE ICDH 2025, 10 pages, 8 figures, 5 tables
>
> **摘要:** Navigating healthcare systems can be complex and overwhelming, creating barriers for patients seeking timely and appropriate medical attention. In this paper, we introduce C-PATH (Conversational Patient Assistance and Triage in Healthcare), a novel conversational AI system powered by large language models (LLMs) designed to assist patients in recognizing symptoms and recommending appropriate medical departments through natural, multi-turn dialogues. C-PATH is fine-tuned on medical knowledge, dialogue data, and clinical summaries using a multi-stage pipeline built on the LLaMA3 architecture. A core contribution of this work is a GPT-based data augmentation framework that transforms structured clinical knowledge from DDXPlus into lay-person-friendly conversations, allowing alignment with patient communication norms. We also implement a scalable conversation history management strategy to ensure long-range coherence. Evaluation with GPTScore demonstrates strong performance across dimensions such as clarity, informativeness, and recommendation accuracy. Quantitative benchmarks show that C-PATH achieves superior performance in GPT-rewritten conversational datasets, significantly outperforming domain-specific baselines. C-PATH represents a step forward in the development of user-centric, accessible, and accurate AI tools for digital health assistance and triage.
>
---
#### [new 032] MiniCPM4: Ultra-Efficient LLMs on End Devices
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型在终端设备上的效率。通过优化模型架构、训练数据、算法和推理系统，实现高效运行。**

- **链接: [http://arxiv.org/pdf/2506.07900v1](http://arxiv.org/pdf/2506.07900v1)**

> **作者:** MiniCPM Team; Chaojun Xiao; Yuxuan Li; Xu Han; Yuzhuo Bai; Jie Cai; Haotian Chen; Wentong Chen; Xin Cong; Ganqu Cui; Ning Ding; Shengdan Fan; Yewei Fang; Zixuan Fu; Wenyu Guan; Yitong Guan; Junshao Guo; Yufeng Han; Bingxiang He; Yuxiang Huang; Cunliang Kong; Qiuzuo Li; Siyuan Li; Wenhao Li; Yanghao Li; Yishan Li; Zhen Li; Dan Liu; Biyuan Lin; Yankai Lin; Xiang Long; Quanyu Lu; Yaxi Lu; Peiyan Luo; Hongya Lyu; Litu Ou; Yinxu Pan; Zekai Qu; Qundong Shi; Zijun Song; Jiayuan Su; Zhou Su; Ao Sun; Xianghui Sun; Peijun Tang; Fangzheng Wang; Feng Wang; Shuo Wang; Yudong Wang; Yesai Wu; Zhenyu Xiao; Jie Xie; Zihao Xie; Yukun Yan; Jiarui Yuan; Kaihuo Zhang; Lei Zhang; Linyue Zhang; Xueren Zhang; Yudi Zhang; Hengyu Zhao; Weilin Zhao; Weilun Zhao; Yuanqian Zhao; Zhi Zheng; Ge Zhou; Jie Zhou; Wei Zhou; Zihan Zhou; Zixuan Zhou; Zhiyuan Liu; Guoyang Zeng; Chao Jia; Dahai Li; Maosong Sun
>
> **备注:** MiniCPM4 Technical Report
>
> **摘要:** This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability.
>
---
#### [new 033] TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型训练任务，旨在解决依赖大量语音文本数据的问题。通过统一编码器和轻量投影网络，仅用文本训练模型，实现语音相关任务的高性能。**

- **链接: [http://arxiv.org/pdf/2506.06343v1](http://arxiv.org/pdf/2506.06343v1)**

> **作者:** Taesoo Kim; Jong Hwan Ko
>
> **摘要:** Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data.
>
---
#### [new 034] Synthesis by Design: Controlled Data Generation via Structural Guidance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务，旨在提升LLM的逻辑推理能力。通过结构化引导生成高质量数据，解决现有方法生成质量与复杂度不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.07664v1](http://arxiv.org/pdf/2506.07664v1)**

> **作者:** Lei Xu; Sirui Chen; Yuxuan Huang; Chaochao Lu
>
> **摘要:** Mathematical reasoning remains challenging for LLMs due to complex logic and the need for precise computation. Existing methods enhance LLM reasoning by synthesizing datasets through problem rephrasing, but face issues with generation quality and problem complexity. To address this, we propose to extract structural information with generated problem-solving code from mathematical reasoning and guide data generation with structured solutions. Applied to MATH and GSM8K, our approach produces 39K problems with labeled intermediate steps and a 6.1K-problem benchmark of higher difficulty. Results on our benchmark show that model performance declines as reasoning length increases. Additionally, we conducted fine-tuning experiments using the proposed training data on a range of LLMs, and the results validate the effectiveness of our dataset. We hope the proposed method and dataset will contribute to future research in enhancing LLM reasoning capabilities.
>
---
#### [new 035] Psychological Counseling Cannot Be Achieved Overnight: Automated Psychological Counseling Through Multi-Session Conversations
- **分类: cs.CL**

- **简介: 该论文属于心理辅导任务，解决单次对话模型无法模拟真实多轮辅导的问题。构建了多轮心理辅导数据集，并提出适应性模型以跟踪客户进展。**

- **链接: [http://arxiv.org/pdf/2506.06626v1](http://arxiv.org/pdf/2506.06626v1)**

> **作者:** Junzhe Wang; Bichen Wang; Xing Fu; Yixin Sun; Yanyan Zhao; Bing Qin
>
> **备注:** 15 pages, 19 figures
>
> **摘要:** In recent years, Large Language Models (LLMs) have made significant progress in automated psychological counseling. However, current research focuses on single-session counseling, which doesn't represent real-world scenarios. In practice, psychological counseling is a process, not a one-time event, requiring sustained, multi-session engagement to progressively address clients' issues. To overcome this limitation, we introduce a dataset for Multi-Session Psychological Counseling Conversation Dataset (MusPsy-Dataset). Our MusPsy-Dataset is constructed using real client profiles from publicly available psychological case reports. It captures the dynamic arc of counseling, encompassing multiple progressive counseling conversations from the same client across different sessions. Leveraging our dataset, we also developed our MusPsy-Model, which aims to track client progress and adapt its counseling direction over time. Experiments show that our model performs better than baseline models across multiple sessions.
>
---
#### [new 036] Unblocking Fine-Grained Evaluation of Detailed Captions: An Explaining AutoRater and Critic-and-Revise Pipeline
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型的评估任务，旨在解决详细图像描述事实性评价难题。构建了DOCCI-Critique基准和VNLI-Critique模型，实现自动事实核查与修正。**

- **链接: [http://arxiv.org/pdf/2506.07631v1](http://arxiv.org/pdf/2506.07631v1)**

> **作者:** Brian Gordon; Yonatan Bitton; Andreea Marzoca; Yasumasa Onoe; Xiao Wang; Daniel Cohen-Or; Idan Szpektor
>
> **摘要:** Large Vision-Language Models (VLMs) now generate highly detailed, paragraphlength image captions, yet evaluating their factual accuracy remains challenging. Current methods often miss fine-grained errors, being designed for shorter texts or lacking datasets with verified inaccuracies. We introduce DOCCI-Critique, a benchmark with 1,400 VLM-generated paragraph captions (100 images, 14 VLMs) featuring over 10,216 sentence-level human annotations of factual correctness and explanatory rationales for errors, all within paragraph context. Building on this, we develop VNLI-Critique, a model for automated sentence-level factuality classification and critique generation. We highlight three key applications: (1) VNLI-Critique demonstrates robust generalization, validated by state-of-the-art performance on the M-HalDetect benchmark and strong results in CHOCOLATE claim verification. (2) The VNLI-Critique driven AutoRater for DOCCI-Critique provides reliable VLM rankings, showing excellent alignment with human factuality judgments (e.g., 0.98 Spearman). (3) An innovative Critic-and-Revise pipeline, where critiques from VNLI-Critique guide LLM-based corrections, achieves substantial improvements in caption factuality (e.g., a 46% gain on DetailCaps-4870). Our work offers a crucial benchmark alongside practical tools, designed to significantly elevate the standards for fine-grained evaluation and foster the improvement of VLM image understanding. Project page: https://google.github.io/unblocking-detail-caption
>
---
#### [new 037] Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于代码生成任务，研究LLMs能否生成可靠的测试用例生成器。旨在解决LLMs在生成有效测试用例方面的能力问题，并通过构建基准和数据集提升其性能。**

- **链接: [http://arxiv.org/pdf/2506.06821v1](http://arxiv.org/pdf/2506.06821v1)**

> **作者:** Yuhan Cao; Zian Chen; Kun Quan; Ziliang Zhang; Yu Wang; Xiaoning Dong; Yeqi Feng; Guanzhong He; Jingcheng Huang; Jianhao Li; Yixuan Tan; Jiafu Tang; Yilin Tang; Junlei Wu; Qianyu Xiao; Can Zheng; Shouchen Zhou; Yuxiang Zhu; Yiming Huang; Tian Xie; Tianxing He
>
> **备注:** 37 pages, 22 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
>
---
#### [new 038] Learning Distribution-Wise Control in Representation Space for Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型控制任务，旨在提升模型行为的可控性与鲁棒性。通过引入分布级干预方法，扩展了传统点对点控制，实现了更精细的模型调节。**

- **链接: [http://arxiv.org/pdf/2506.06686v1](http://arxiv.org/pdf/2506.06686v1)**

> **作者:** Chunyuan Deng; Ruidi Chang; Hanjie Chen
>
> **备注:** ICML 2025
>
> **摘要:** Interventions in language models (LMs) are applied strategically to steer model behavior during the forward pass. Learnable interventions, also known as representation fine-tuning, aim to apply pointwise control within the concept subspace and have proven effective in altering high-level behaviors. In this work, we extend this approach to the distribution level, enabling the model to learn not only pointwise transformations but also the surrounding regions of the concept subspace. We demonstrate that these methods perform effectively in early layers, with larger standard deviations correlating strongly with improved performance. Across eight commonsense reasoning and seven arithmetic reasoning benchmarks, our distribution-wise interventions consistently outperform pointwise interventions in controllability and robustness. These results illustrate that distribution-wise interventions provide a more comprehensive method for steering model behavior and enabling finer-grained control over language models. The code is at: \href{https://github.com/chili-lab/D-Intervention}{https://github.com/chili-lab/D-Intervention}.
>
---
#### [new 039] Relationship Detection on Tabular Data Using Statistical Analysis and Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于表格数据关系检测任务，旨在解决无标签表格中列间关系的识别问题。通过结合统计分析与大语言模型，提出一种混合方法以提高检测效果。**

- **链接: [http://arxiv.org/pdf/2506.06371v1](http://arxiv.org/pdf/2506.06371v1)**

> **作者:** Panagiotis Koletsis; Christos Panagiotopoulos; Georgios Th. Papadopoulos; Vasilis Efthymiou
>
> **摘要:** Over the past few years, table interpretation tasks have made significant progress due to their importance and the introduction of new technologies and benchmarks in the field. This work experiments with a hybrid approach for detecting relationships among columns of unlabeled tabular data, using a Knowledge Graph (KG) as a reference point, a task known as CPA. This approach leverages large language models (LLMs) while employing statistical analysis to reduce the search space of potential KG relations. The main modules of this approach for reducing the search space are domain and range constraints detection, as well as relation co-appearance analysis. The experimental evaluation on two benchmark datasets provided by the SemTab challenge assesses the influence of each module and the effectiveness of different state-of-the-art LLMs at various levels of quantization. The experiments were performed, as well as at different prompting techniques. The proposed methodology, which is publicly available on github, proved to be competitive with state-of-the-art approaches on these datasets.
>
---
#### [new 040] ConfQA: Answer Only If You Are Confident
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在减少大语言模型的幻觉问题。通过引入ConfQA策略，使模型在不确定时承认错误，提升回答准确性。**

- **链接: [http://arxiv.org/pdf/2506.07309v1](http://arxiv.org/pdf/2506.07309v1)**

> **作者:** Yin Huang; Yifan Ethan Xu; Kai Sun; Vera Yan; Alicia Sun; Haidar Khan; Jimmy Nguyen; Mohammad Kachuee; Zhaojiang Lin; Yue Liu; Aaron Colak; Anuj Kumar; Wen-tau Yih; Xin Luna Dong
>
> **备注:** 10 pages main content, 10 pages appendix, 5 figures, 7 tables
>
> **摘要:** Can we teach Large Language Models (LLMs) to refrain from hallucinating factual statements? In this paper we present a fine-tuning strategy that we call ConfQA, which can reduce hallucination rate from 20-40% to under 5% across multiple factuality benchmarks. The core idea is simple: when the LLM answers a question correctly, it is trained to continue with the answer; otherwise, it is trained to admit "I am unsure". But there are two key factors that make the training highly effective. First, we introduce a dampening prompt "answer only if you are confident" to explicitly guide the behavior, without which hallucination remains high as 15%-25%. Second, we leverage simple factual statements, specifically attribute values from knowledge graphs, to help LLMs calibrate the confidence, resulting in robust generalization across domains and question types. Building on this insight, we propose the Dual Neural Knowledge framework, which seamlessly select between internally parameterized neural knowledge and externally recorded symbolic knowledge based on ConfQA's confidence. The framework enables potential accuracy gains to beyond 95%, while reducing unnecessary external retrievals by over 30%.
>
---
#### [new 041] Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation
- **分类: cs.CL**

- **简介: 该论文属于大模型微调任务，旨在解决用户数据中的有害提示导致模型安全对齐下降的问题。通过引入拒绝特征引导的教师模型，过滤有害数据并蒸馏对齐知识。**

- **链接: [http://arxiv.org/pdf/2506.07356v1](http://arxiv.org/pdf/2506.07356v1)**

> **作者:** Seokil Ham; Yubin Choi; Seungju Cho; Yujin Yang; Younghun Kim; Changick Kim
>
> **摘要:** Recently, major AI service providers such as Google and OpenAI have introduced Finetuning-as-a-Service, which enables users to customize Large Language Models (LLMs) for specific downstream tasks using their own data. However, this service is vulnerable to degradation of LLM safety-alignment when user data contains harmful prompts. While some prior works address this issue, fundamentally filtering harmful data from user data remains unexplored. Motivated by our observation that a directional representation reflecting refusal behavior (called the refusal feature) obtained from safety-aligned LLMs can inherently distinguish between harmful and harmless prompts, we propose the Refusal-Feature-guided Teacher (ReFT). Our ReFT model is trained to identify harmful prompts based on the similarity between input prompt features and its refusal feature. During finetuning, the ReFT model serves as a teacher that filters harmful prompts from user data and distills alignment knowledge into the base model. Extensive experiments demonstrate that our ReFT-based finetuning strategy effectively minimizes harmful outputs and enhances finetuning accuracy for user-specific tasks, offering a practical solution for secure and reliable deployment of LLMs in Finetuning-as-a-Service.
>
---
#### [new 042] Geopolitical biases in LLMs: what are the "good" and the "bad" countries according to contemporary language models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见分析任务，旨在研究大语言模型中的地缘政治偏见。通过分析不同国家对历史事件的描述，揭示模型对特定国家叙事的偏好，并测试去偏方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.06751v1](http://arxiv.org/pdf/2506.06751v1)**

> **作者:** Mikhail Salnikov; Dmitrii Korzh; Ivan Lazichny; Elvir Karimov; Artyom Iudin; Ivan Oseledets; Oleg Y. Rogov; Alexander Panchenko; Natalia Loukachevitch; Elena Tutubalina
>
> **摘要:** This paper evaluates geopolitical biases in LLMs with respect to various countries though an analysis of their interpretation of historical events with conflicting national perspectives (USA, UK, USSR, and China). We introduce a novel dataset with neutral event descriptions and contrasting viewpoints from different countries. Our findings show significant geopolitical biases, with models favoring specific national narratives. Additionally, simple debiasing prompts had a limited effect in reducing these biases. Experiments with manipulated participant labels reveal models' sensitivity to attribution, sometimes amplifying biases or recognizing inconsistencies, especially with swapped labels. This work highlights national narrative biases in LLMs, challenges the effectiveness of simple debiasing methods, and offers a framework and dataset for future geopolitical bias research.
>
---
#### [new 043] Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多标签文本分类任务，旨在解决手动分类效率低的问题。提出一种基于标签语义的生成方法，通过生成标签描述并匹配实现分类，提升准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.06806v1](http://arxiv.org/pdf/2506.06806v1)**

> **作者:** Subhendu Khatuya; Shashwat Naidu; Saptarshi Ghosh; Pawan Goyal; Niloy Ganguly
>
> **备注:** This work has been accepted to appear at the Association for Computational Linguistics (ACL), 2025
>
> **摘要:** The explosion of textual data has made manual document classification increasingly challenging. To address this, we introduce a robust, efficient domain-agnostic generative model framework for multi-label text classification. Instead of treating labels as mere atomic symbols, our approach utilizes predefined label descriptions and is trained to generate these descriptions based on the input text. During inference, the generated descriptions are matched to the pre-defined labels using a finetuned sentence transformer. We integrate this with a dual-objective loss function, combining cross-entropy loss and cosine similarity of the generated sentences with the predefined target descriptions, ensuring both semantic alignment and accuracy. Our proposed model LAGAMC stands out for its parameter efficiency and versatility across diverse datasets, making it well-suited for practical applications. We demonstrate the effectiveness of our proposed model by achieving new state-of-the-art performances across all evaluated datasets, surpassing several strong baselines. We achieve improvements of 13.94% in Micro-F1 and 24.85% in Macro-F1 compared to the closest baseline across all datasets.
>
---
#### [new 044] CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在通过实例选择减少训练集规模，提升模型训练效率。工作包括比较IS方法，提出两种新方案，实现41%的规模缩减与显著加速。**

- **链接: [http://arxiv.org/pdf/2506.07169v1](http://arxiv.org/pdf/2506.07169v1)**

> **作者:** Washington Cunha; Leonardo Rocha; Marcos André Gonçalves
>
> **备注:** 16 pages, 5 figures, 2 tables
>
> **摘要:** Progress in Natural Language Processing (NLP) has been dictated by the rule of more: more data, more computing power and more complexity, best exemplified by the Large Language Models. However, training (or fine-tuning) large dense models for specific applications usually requires significant amounts of computing resources. This \textbf{Ph.D. dissertation} focuses on an under-investi\-gated NLP data engineering technique, whose potential is enormous in the current scenario known as Instance Selection (IS). The IS goal is to reduce the training set size by removing noisy or redundant instances while maintaining the effectiveness of the trained models and reducing the training process cost. We provide a comprehensive and scientifically sound comparison of IS methods applied to an essential NLP task -- Automatic Text Classification (ATC), considering several classification solutions and many datasets. Our findings reveal a significant untapped potential for IS solutions. We also propose two novel IS solutions that are noise-oriented and redundancy-aware, specifically designed for large datasets and transformer architectures. Our final solution achieved an average reduction of 41\% in training sets, while maintaining the same levels of effectiveness in all datasets. Importantly, our solutions demonstrated speedup improvements of 1.67x (up to 2.46x), making them scalable for datasets with hundreds of thousands of documents.
>
---
#### [new 045] LoRMA: Low-Rank Multiplicative Adaptation for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，解决大模型微调计算成本高的问题。提出LoRMA方法，通过矩阵乘法替代加法更新，提升效率与效果。**

- **链接: [http://arxiv.org/pdf/2506.07621v1](http://arxiv.org/pdf/2506.07621v1)**

> **作者:** Harsh Bihany; Shubham Patel; Ashutosh Modi
>
> **备注:** Accepted at ACL Findings 2025; 21 pages (9 main paper + 5 pages references + 7 pages appendix)
>
> **摘要:** Large Language Models have shown remarkable capabilities in the NLP domain. Their effectiveness can mainly be attributed to their ability to adapt to an array of downstream tasks. However, generally, full fine-tuning is a computationally expensive job. To mitigate this, many techniques have been developed that prime efficiency, a prominent one being Low-Rank Adaptation (LoRA). However, LoRA and its variants employ re-parametrized additive updates. In this paper, we propose Low-Rank Multiplicative Adaptation (LoRMA), which shifts the paradigm of additive updates to a richer space of matrix multiplicative transformations. We tackle challenges such as computational complexity and rank bottleneck of matrix multiplication by effectively re-ordering operations and introducing rank inflation strategies. We conduct extensive experiments to demonstrate the effectiveness of our approach in terms of various evaluation metrics.
>
---
#### [new 046] Hybrid Extractive Abstractive Summarization for Multilingual Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多语言情感分析任务，旨在解决单一方法的局限性。通过结合抽取式和生成式摘要技术，提升准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.06929v1](http://arxiv.org/pdf/2506.06929v1)**

> **作者:** Mikhail Krasitskii; Grigori Sidorov; Olga Kolesnikova; Liliana Chanona Hernandez; Alexander Gelbukh
>
> **备注:** 6 pages
>
> **摘要:** We propose a hybrid approach for multilingual sentiment analysis that combines extractive and abstractive summarization to address the limitations of standalone methods. The model integrates TF-IDF-based extraction with a fine-tuned XLM-R abstractive module, enhanced by dynamic thresholding and cultural adaptation. Experiments across 10 languages show significant improvements over baselines, achieving 0.90 accuracy for English and 0.84 for low-resource languages. The approach also demonstrates 22% greater computational efficiency than traditional methods. Practical applications include real-time brand monitoring and cross-cultural discourse analysis. Future work will focus on optimization for low-resource languages via 8-bit quantization.
>
---
#### [new 047] Com$^2$: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于复杂常识推理任务，旨在解决LLMs在处理隐性、复杂常识知识时的不足。通过构建基准Com$^2$，结合因果图与推理策略提升模型表现。**

- **链接: [http://arxiv.org/pdf/2506.07064v1](http://arxiv.org/pdf/2506.07064v1)**

> **作者:** Kai Xiong; Xiao Ding; Yixin Cao; Yuxiong Yan; Li Du; Yufei Zhang; Jinglong Gao; Jiaqian Liu; Bing Qin; Ting Liu
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) have mastered abundant simple and explicit commonsense knowledge through pre-training, enabling them to achieve human-like performance in simple commonsense reasoning. Nevertheless, LLMs struggle to reason with complex and implicit commonsense knowledge that is derived from simple ones (such as understanding the long-term effects of certain events), an aspect humans tend to focus on more. Existing works focus on complex tasks like math and code, while complex commonsense reasoning remains underexplored due to its uncertainty and lack of structure. To fill this gap and align with real-world concerns, we propose a benchmark Com$^2$ focusing on complex commonsense reasoning. We first incorporate causal event graphs to serve as structured complex commonsense. Then we adopt causal theory~(e.g., intervention) to modify the causal event graphs and obtain different scenarios that meet human concerns. Finally, an LLM is employed to synthesize examples with slow thinking, which is guided by the logical relationships in the modified causal graphs. Furthermore, we use detective stories to construct a more challenging subset. Experiments show that LLMs struggle in reasoning depth and breadth, while post-training and slow thinking can alleviate this. The code and data are available at https://github.com/Waste-Wood/Com2.
>
---
#### [new 048] Question Answering under Temporal Conflict: Evaluating and Organizing Evolving Knowledge with LLMs
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决LLMs在处理随时间变化的知识时出现的冲突问题。通过构建外部记忆框架提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2506.07270v1](http://arxiv.org/pdf/2506.07270v1)**

> **作者:** Atahan Özer; Çağatay Yıldız
>
> **摘要:** Large language models (LLMs) exhibit remarkable capabilities in question answering and reasoning thanks to their extensive parametric memory. However, their knowledge is inherently limited by the scope of their pre-training data, while real-world information evolves continuously. Updating this knowledge typically requires costly and brittle re-training, or in-context learning (ICL), which becomes impractical at scale given the volume and volatility of modern information. Motivated by these limitations, we investigate how LLMs perform when exposed to temporal text corpora, or documents that reflect evolving knowledge over time, such as sports biographies where facts like a player's "current team" change year by year. To this end, we introduce two new benchmarks: Temporal Wiki, which captures factual drift across historical Wikipedia snapshots, and Unified Clark, which aggregates timestamped news articles to simulate real-world information accumulation. Our analysis reveals that LLMs often struggle to reconcile conflicting or outdated facts and can be misled when multiple versions of a fact appear in context. To address these issues, we propose a lightweight, agentic framework that incrementally builds a structured, external memory from source documents without requiring re-training. This knowledge organization strategy enables models to retrieve and reason over temporally filtered, relevant information at inference time. Empirically, our method outperforms ICL and RAG baselines across both benchmarks, especially on questions requiring more complex reasoning or integration of conflicting facts.
>
---
#### [new 049] Subjectivity in the Annotation of Bridging Anaphora
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究桥梁指代标注中的主观性问题，属于自然语言处理任务。旨在解决标注一致性差的问题，通过实验分析不同层面的主观性并提出新分类系统。**

- **链接: [http://arxiv.org/pdf/2506.07297v1](http://arxiv.org/pdf/2506.07297v1)**

> **作者:** Lauren Levine; Amir Zeldes
>
> **备注:** LAW-XIX, ACL 2025 Workshop
>
> **摘要:** Bridging refers to the associative relationship between inferable entities in a discourse and the antecedents which allow us to understand them, such as understanding what "the door" means with respect to an aforementioned "house". As identifying associative relations between entities is an inherently subjective task, it is difficult to achieve consistent agreement in the annotation of bridging anaphora and their antecedents. In this paper, we explore the subjectivity involved in the annotation of bridging instances at three levels: anaphor recognition, antecedent resolution, and bridging subtype selection. To do this, we conduct an annotation pilot on the test set of the existing GUM corpus, and propose a newly developed classification system for bridging subtypes, which we compare to previously proposed schemes. Our results suggest that some previous resources are likely to be severely under-annotated. We also find that while agreement on the bridging subtype category was moderate, annotator overlap for exhaustively identifying instances of bridging is low, and that many disagreements resulted from subjective understanding of the entities involved.
>
---
#### [new 050] Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见分析任务，旨在解决非英语语言模型偏见解释问题。工作包括适配信息论偏见评分方法，并应用于菲律宾语模型验证。**

- **链接: [http://arxiv.org/pdf/2506.07249v1](http://arxiv.org/pdf/2506.07249v1)**

> **作者:** Lance Calvin Lim Gamboa; Yue Feng; Mark Lee
>
> **备注:** Accepted into the Gender Bias in NLP Workshop at ACL 2025 (GeBNLP@ACL2025)
>
> **摘要:** Emerging research on bias attribution and interpretability have revealed how tokens contribute to biased behavior in language models processing English texts. We build on this line of inquiry by adapting the information-theoretic bias attribution score metric for implementation on models handling agglutinative languages, particularly Filipino. We then demonstrate the effectiveness of our adapted method by using it on a purely Filipino model and on three multilingual models: one trained on languages worldwide and two on Southeast Asian data. Our results show that Filipino models are driven towards bias by words pertaining to people, objects, and relationships, entity-based themes that stand in contrast to the action-heavy nature of bias-contributing themes in English (i.e., criminal, sexual, and prosocial behaviors). These findings point to differences in how English and non-English models process inputs linked to sociodemographic groups and bias.
>
---
#### [new 051] Quantum Graph Transformer for NLP Sentiment Classification
- **分类: cs.CL; quant-ph**

- **简介: 该论文属于自然语言处理中的情感分类任务，提出量子图变换器（QGT）解决传统模型参数多、效率低的问题，通过量子自注意力机制提升性能与样本效率。**

- **链接: [http://arxiv.org/pdf/2506.07937v1](http://arxiv.org/pdf/2506.07937v1)**

> **作者:** Shamminuj Aktar; Andreas Bärtschi; Abdel-Hameed A. Badawy; Stephan Eidenbenz
>
> **摘要:** Quantum machine learning is a promising direction for building more efficient and expressive models, particularly in domains where understanding complex, structured data is critical. We present the Quantum Graph Transformer (QGT), a hybrid graph-based architecture that integrates a quantum self-attention mechanism into the message-passing framework for structured language modeling. The attention mechanism is implemented using parameterized quantum circuits (PQCs), which enable the model to capture rich contextual relationships while significantly reducing the number of trainable parameters compared to classical attention mechanisms. We evaluate QGT on five sentiment classification benchmarks. Experimental results show that QGT consistently achieves higher or comparable accuracy than existing quantum natural language processing (QNLP) models, including both attention-based and non-attention-based approaches. When compared with an equivalent classical graph transformer, QGT yields an average accuracy improvement of 5.42% on real-world datasets and 4.76% on synthetic datasets. Additionally, QGT demonstrates improved sample efficiency, requiring nearly 50% fewer labeled samples to reach comparable performance on the Yelp dataset. These results highlight the potential of graph-based QNLP techniques for advancing efficient and scalable language understanding.
>
---
#### [new 052] Right Is Not Enough: The Pitfalls of Outcome Supervision in Training LLMs for Math Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决LLMs仅靠结果奖励导致的推理过程错误问题。通过提出ParaStepVerifier方法，提升对解题步骤的验证准确性。**

- **链接: [http://arxiv.org/pdf/2506.06877v1](http://arxiv.org/pdf/2506.06877v1)**

> **作者:** Jiaxing Guo; Wenjie Yang; Shengzhong Zhang; Tongshan Xu; Lun Du; Da Zheng; Zengfeng Huang
>
> **摘要:** Outcome-rewarded Large Language Models (LLMs) have demonstrated remarkable success in mathematical problem-solving. However, this success often masks a critical issue: models frequently achieve correct answers through fundamentally unsound reasoning processes, a phenomenon indicative of reward hacking. We introduce MathOlympiadEval, a new dataset with fine-grained annotations, which reveals a significant gap between LLMs' answer correctness and their low process correctness. Existing automated methods like LLM-as-a-judge struggle to reliably detect these reasoning flaws. To address this, we propose ParaStepVerifier, a novel methodology for meticulous, step-by-step verification of mathematical solutions. ParaStepVerifier identifies incorrect reasoning steps. Empirical results demonstrate that ParaStepVerifier substantially improves the accuracy of identifying flawed solutions compared to baselines, especially for complex, multi-step problems. This offers a more robust path towards evaluating and training LLMs with genuine mathematical reasoning.
>
---
#### [new 053] Break-The-Chain: Reasoning Failures in LLMs via Adversarial Prompting in Code Generation
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于代码生成任务，研究LLMs在推理中的脆弱性。通过对抗性提示扰动，揭示模型对表面模式的依赖，提出提升推理鲁棒性的方向。**

- **链接: [http://arxiv.org/pdf/2506.06971v1](http://arxiv.org/pdf/2506.06971v1)**

> **作者:** Jaechul Roh; Varun Gandhi; Shivani Anilkumar; Arin Garg
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success in tasks requiring complex reasoning, such as code generation, mathematical problem solving, and algorithmic synthesis -- especially when aided by reasoning tokens and Chain-of-Thought prompting. Yet, a core question remains: do these models truly reason, or do they merely exploit shallow statistical patterns? In this paper, we systematically investigate the robustness of reasoning LLMs by introducing a suite of semantically faithful yet adversarially structured prompt perturbations. Our evaluation -- spanning 700 perturbed code generations derived from LeetCode-style problems -- applies transformations such as storytelling reframing, irrelevant constraint injection, example reordering, and numeric perturbation. We observe that while certain modifications severely degrade performance (with accuracy drops up to -42.1%), others surprisingly improve model accuracy by up to 35.3%, suggesting sensitivity not only to semantics but also to surface-level prompt dynamics. These findings expose the fragility and unpredictability of current reasoning systems, underscoring the need for more principles approaches to reasoning alignments and prompting robustness. We release our perturbation datasets and evaluation framework to promote further research in trustworthy and resilient LLM reasoning.
>
---
#### [new 054] Multilingual Grammatical Error Annotation: Combining Language-Agnostic Framework with Language-Specific Flexibility
- **分类: cs.CL**

- **简介: 该论文属于语法错误纠正任务，旨在解决多语言错误标注不一致的问题。提出一种结合通用框架与语言特性的标注方法，提升多语言GEC的准确性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.07719v1](http://arxiv.org/pdf/2506.07719v1)**

> **作者:** Mengyang Qiu; Tran Minh Nguyen; Zihao Huang; Zelong Li; Yang Gu; Qingyu Gao; Siliang Liu; Jungyeul Park
>
> **备注:** BEA2025
>
> **摘要:** Grammatical Error Correction (GEC) relies on accurate error annotation and evaluation, yet existing frameworks, such as $\texttt{errant}$, face limitations when extended to typologically diverse languages. In this paper, we introduce a standardized, modular framework for multilingual grammatical error annotation. Our approach combines a language-agnostic foundation with structured language-specific extensions, enabling both consistency and flexibility across languages. We reimplement $\texttt{errant}$ using $\texttt{stanza}$ to support broader multilingual coverage, and demonstrate the framework's adaptability through applications to English, German, Czech, Korean, and Chinese, ranging from general-purpose annotation to more customized linguistic refinements. This work supports scalable and interpretable GEC annotation across languages and promotes more consistent evaluation in multilingual settings. The complete codebase and annotation tools can be accessed at https://github.com/open-writing-evaluation/jp_errant_bea.
>
---
#### [new 055] Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决情感理解缺乏解释性的问题。通过引入多任务框架和生成式推理，提升情感预测的准确性和解释性。**

- **链接: [http://arxiv.org/pdf/2506.06820v1](http://arxiv.org/pdf/2506.06820v1)**

> **作者:** Wenyu Zhang; Yingxu He; Geyu Lin; Zhuohan Liu; Shuo Sun; Bin Wang; Xunlong Zou; Jeremy H. M. Wong; Qiongqiong Wang; Hardik B. Sailor; Nancy F. Chen; Ai Ti Aw
>
> **摘要:** Audio Large Language Models (AudioLLMs) have achieved strong results in semantic tasks like speech recognition and translation, but remain limited in modeling paralinguistic cues such as emotion. Existing approaches often treat emotion understanding as a classification problem, offering little insight into the underlying rationale behind predictions. In this work, we explore emotion reasoning, a strategy that leverages the generative capabilities of AudioLLMs to enhance emotion recognition by producing semantically aligned, evidence-grounded explanations. To support this in multitask AudioLLMs, we introduce a unified framework combining reasoning-augmented data supervision, dual-encoder architecture, and task-alternating training. This approach enables AudioLLMs to effectively learn different tasks while incorporating emotional reasoning. Experiments on IEMOCAP and MELD show that our approach not only improves emotion prediction accuracy but also enhances the coherence and evidential grounding of the generated responses.
>
---
#### [new 056] What Do Indonesians Really Need from Language Technology? A Nationwide Survey
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在了解印尼本土语言使用者对语言技术的真实需求。通过全国性调查，发现机器翻译和信息检索是关键需求，同时关注隐私与透明度问题。**

- **链接: [http://arxiv.org/pdf/2506.07506v1](http://arxiv.org/pdf/2506.07506v1)**

> **作者:** Muhammad Dehan Al Kautsar; Lucky Susanto; Derry Wijaya; Fajri Koto
>
> **备注:** 26 pages, 12 figures, 5 tables
>
> **摘要:** There is an emerging effort to develop NLP for Indonesias 700+ local languages, but progress remains costly due to the need for direct engagement with native speakers. However, it is unclear what these language communities truly need from language technology. To address this, we conduct a nationwide survey to assess the actual needs of native speakers in Indonesia. Our findings indicate that addressing language barriers, particularly through machine translation and information retrieval, is the most critical priority. Although there is strong enthusiasm for advancements in language technology, concerns around privacy, bias, and the use of public data for AI training highlight the need for greater transparency and clear communication to support broader AI adoption.
>
---
#### [new 057] A Culturally-diverse Multilingual Multimodal Video Benchmark & Model
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态视频理解任务，旨在解决视频大模型在语言和文化多样性上的不足。提出ViMUL-Bench基准和多语言模型ViMUL，覆盖14种语言，促进跨语言与文化包容的视频理解研究。**

- **链接: [http://arxiv.org/pdf/2506.07032v1](http://arxiv.org/pdf/2506.07032v1)**

> **作者:** Bhuiyan Sanjid Shafique; Ashmal Vayani; Muhammad Maaz; Hanoona Abdul Rasheed; Dinura Dissanayake; Mohammed Irfan Kurpath; Yahya Hmaiti; Go Inoue; Jean Lahoud; Md. Safirur Rashid; Shadid Intisar Quasem; Maheen Fatima; Franco Vidal; Mykola Maslych; Ketan Pravin More; Sanoojan Baliah; Hasindri Watawana; Yuhao Li; Fabian Farestam; Leon Schaller; Roman Tymtsiv; Simon Weber; Hisham Cholakkal; Ivan Laptev; Shin'ichi Satoh; Michael Felsberg; Mubarak Shah; Salman Khan; Fahad Shahbaz Khan
>
> **摘要:** Large multimodal models (LMMs) have recently gained attention due to their effectiveness to understand and generate descriptions of visual content. Most existing LMMs are in English language. While few recent works explore multilingual image LMMs, to the best of our knowledge, moving beyond the English language for cultural and linguistic inclusivity is yet to be investigated in the context of video LMMs. In pursuit of more inclusive video LMMs, we introduce a multilingual Video LMM benchmark, named ViMUL-Bench, to evaluate Video LMMs across 14 languages, including both low- and high-resource languages: English, Chinese, Spanish, French, German, Hindi, Arabic, Russian, Bengali, Urdu, Sinhala, Tamil, Swedish, and Japanese. Our ViMUL-Bench is designed to rigorously test video LMMs across 15 categories including eight culturally diverse categories, ranging from lifestyles and festivals to foods and rituals and from local landmarks to prominent cultural personalities. ViMUL-Bench comprises both open-ended (short and long-form) and multiple-choice questions spanning various video durations (short, medium, and long) with 8k samples that are manually verified by native language speakers. In addition, we also introduce a machine translated multilingual video training set comprising 1.2 million samples and develop a simple multilingual video LMM, named ViMUL, that is shown to provide a better tradeoff between high-and low-resource languages for video understanding. We hope our ViMUL-Bench and multilingual video LMM along with a large-scale multilingual video training set will help ease future research in developing cultural and linguistic inclusive multilingual video LMMs. Our proposed benchmark, video LMM and training data will be publicly released at https://mbzuai-oryx.github.io/ViMUL/.
>
---
#### [new 058] TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review
- **分类: cs.CL**

- **简介: 该论文提出TreeReview框架，用于改进LLM的科学同行评审任务，解决生成全面且高效评论的问题。通过动态问题树结构提升评审深度与效率。**

- **链接: [http://arxiv.org/pdf/2506.07642v1](http://arxiv.org/pdf/2506.07642v1)**

> **作者:** Yuan Chang; Ziyue Li; Hengyuan Zhang; Yuanbo Kong; Yanru Wu; Zhijiang Guo; Ngai Wong
>
> **备注:** 30 pages, 17 figures
>
> **摘要:** While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
>
---
#### [new 059] Adapt Once, Thrive with Updates: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决PEFT模块在基模型更新后性能下降的问题。通过分析模型变化，提出Trans-PEFT方法，提升模块适应性，减少重调成本。**

- **链接: [http://arxiv.org/pdf/2506.06844v1](http://arxiv.org/pdf/2506.06844v1)**

> **作者:** Naibin Gu; Peng Fu; Xiyu Liu; Ke Ma; Zheng Lin; Weiping Wang
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) has become a common method for fine-tuning large language models, where a base model can serve multiple users through PEFT module switching. To enhance user experience, base models require periodic updates. However, once updated, PEFT modules fine-tuned on previous versions often suffer substantial performance degradation on newer versions. Re-tuning these numerous modules to restore performance would incur significant computational costs. Through a comprehensive analysis of the changes that occur during base model updates, we uncover an interesting phenomenon: continual training primarily affects task-specific knowledge stored in Feed-Forward Networks (FFN), while having less impact on the task-specific pattern in the Attention mechanism. Based on these findings, we introduce Trans-PEFT, a novel approach that enhances the PEFT module by focusing on the task-specific pattern while reducing its dependence on certain knowledge in the base model. Further theoretical analysis supports our approach. Extensive experiments across 7 base models and 12 datasets demonstrate that Trans-PEFT trained modules can maintain performance on updated base models without re-tuning, significantly reducing maintenance overhead in real-world applications.
>
---
#### [new 060] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于虚假新闻检测任务，旨在提升零样本场景下的谣言识别能力。通过引入说服知识增强思维链（PCoT），有效提升了检测效果。**

- **链接: [http://arxiv.org/pdf/2506.06842v1](http://arxiv.org/pdf/2506.06842v1)**

> **作者:** Arkadiusz Modzelewski; Witold Sosnowski; Tiziano Labruna; Adam Wierzbicki; Giovanni Da San Martino
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Disinformation detection is a key aspect of media literacy. Psychological studies have shown that knowledge of persuasive fallacies helps individuals detect disinformation. Inspired by these findings, we experimented with large language models (LLMs) to test whether infusing persuasion knowledge enhances disinformation detection. As a result, we introduce the Persuasion-Augmented Chain of Thought (PCoT), a novel approach that leverages persuasion to improve disinformation detection in zero-shot classification. We extensively evaluate PCoT on online news and social media posts. Moreover, we publish two novel, up-to-date disinformation datasets: EUDisinfo and MultiDis. These datasets enable the evaluation of PCoT on content entirely unseen by the LLMs used in our experiments, as the content was published after the models' knowledge cutoffs. We show that, on average, PCoT outperforms competitive methods by 15% across five LLMs and five datasets. These findings highlight the value of persuasion in strengthening zero-shot disinformation detection.
>
---
#### [new 061] Syntactic Control of Language Models by Posterior Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决语法结构控制问题。通过后验推断采样方法，提升生成文本的句法准确性，同时保持语言流畅。**

- **链接: [http://arxiv.org/pdf/2506.07154v1](http://arxiv.org/pdf/2506.07154v1)**

> **作者:** Vicky Xefteri; Tim Vieira; Ryan Cotterell; Afra Amini
>
> **摘要:** Controlling the syntactic structure of text generated by language models is valuable for applications requiring clarity, stylistic consistency, or interpretability, yet it remains a challenging task. In this paper, we argue that sampling algorithms based on the posterior inference can effectively enforce a target constituency structure during generation. Our approach combines sequential Monte Carlo, which estimates the posterior distribution by sampling from a proposal distribution, with a syntactic tagger that ensures that each generated token aligns with the desired syntactic structure. Our experiments with GPT2 and Llama3-8B models show that with an appropriate proposal distribution, we can improve syntactic accuracy, increasing the F1 score from $12.31$ (GPT2-large) and $35.33$ (Llama3-8B) to about $93$ in both cases without compromising the language model's fluency. These results underscore both the complexity of syntactic control and the effectiveness of sampling algorithms, offering a promising approach for applications where precise control over syntax is essential.
>
---
#### [new 062] Transferring Features Across Language Models With Model Stitching
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型迁移任务，解决不同大小语言模型间特征迁移问题。通过仿射映射实现特征有效转移，提升SAEs训练效率。**

- **链接: [http://arxiv.org/pdf/2506.06609v1](http://arxiv.org/pdf/2506.06609v1)**

> **作者:** Alan Chen; Jack Merullo; Alessandro Stolfo; Ellie Pavlick
>
> **摘要:** In this work, we demonstrate that affine mappings between residual streams of language models is a cheap way to effectively transfer represented features between models. We apply this technique to transfer the weights of Sparse Autoencoders (SAEs) between models of different sizes to compare their representations. We find that small and large models learn highly similar representation spaces, which motivates training expensive components like SAEs on a smaller model and transferring to a larger model at a FLOPs savings. For example, using a small-to-large transferred SAE as initialization can lead to 50% cheaper training runs when training SAEs on larger models. Next, we show that transferred probes and steering vectors can effectively recover ground truth performance. Finally, we dive deeper into feature-level transferability, finding that semantic and structural features transfer noticeably differently while specific classes of functional features have their roles faithfully mapped. Overall, our findings illustrate similarities and differences in the linear representation spaces of small and large models and demonstrate a method for improving the training efficiency of SAEs.
>
---
#### [new 063] SafeLawBench: Towards Safe Alignment of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型安全评估任务，旨在解决缺乏客观安全标准的问题。提出SafeLawBench基准，通过法律视角评估模型安全性。**

- **链接: [http://arxiv.org/pdf/2506.06636v1](http://arxiv.org/pdf/2506.06636v1)**

> **作者:** Chuxue Cao; Han Zhu; Jiaming Ji; Qichao Sun; Zhenghao Zhu; Yinyu Wu; Juntao Dai; Yaodong Yang; Sirui Han; Yike Guo
>
> **备注:** Accepted to ACL2025 Findings
>
> **摘要:** With the growing prevalence of large language models (LLMs), the safety of LLMs has raised significant concerns. However, there is still a lack of definitive standards for evaluating their safety due to the subjective nature of current safety benchmarks. To address this gap, we conducted the first exploration of LLMs' safety evaluation from a legal perspective by proposing the SafeLawBench benchmark. SafeLawBench categorizes safety risks into three levels based on legal standards, providing a systematic and comprehensive framework for evaluation. It comprises 24,860 multi-choice questions and 1,106 open-domain question-answering (QA) tasks. Our evaluation included 2 closed-source LLMs and 18 open-source LLMs using zero-shot and few-shot prompting, highlighting the safety features of each model. We also evaluated the LLMs' safety-related reasoning stability and refusal behavior. Additionally, we found that a majority voting mechanism can enhance model performance. Notably, even leading SOTA models like Claude-3.5-Sonnet and GPT-4o have not exceeded 80.5% accuracy in multi-choice tasks on SafeLawBench, while the average accuracy of 20 LLMs remains at 68.8\%. We urge the community to prioritize research on the safety of LLMs.
>
---
#### [new 064] BTPD: A Multilingual Hand-curated Dataset of Bengali Transnational Political Discourse Across Online Communities
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于自然语言处理任务，旨在解决 Bengali 政治话语数据缺失问题，通过手工整理跨平台数据构建多语言语料库。**

- **链接: [http://arxiv.org/pdf/2506.06813v1](http://arxiv.org/pdf/2506.06813v1)**

> **作者:** Dipto Das; Syed Ishtiaque Ahmed; Shion Guha
>
> **摘要:** Understanding political discourse in online spaces is crucial for analyzing public opinion and ideological polarization. While social computing and computational linguistics have explored such discussions in English, such research efforts are significantly limited in major yet under-resourced languages like Bengali due to the unavailability of datasets. In this paper, we present a multilingual dataset of Bengali transnational political discourse (BTPD) collected from three online platforms, each representing distinct community structures and interaction dynamics. Besides describing how we hand-curated the dataset through community-informed keyword-based retrieval, this paper also provides a general overview of its topics and multilingual content.
>
---
#### [new 065] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于AI安全任务，研究价值对齐大语言模型的潜在危害，分析其心理机制并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2506.06404v1](http://arxiv.org/pdf/2506.06404v1)**

> **作者:** Sooyung Choi; Jaehyeok Lee; Xiaoyuan Yi; Jing Yao; Xing Xie; JinYeong Bak
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** The application scope of Large Language Models (LLMs) continues to expand, leading to increasing interest in personalized LLMs that align with human values. However, aligning these models with individual values raises significant safety concerns, as certain values may correlate with harmful information. In this paper, we identify specific safety risks associated with value-aligned LLMs and investigate the psychological principles behind these challenges. Our findings reveal two key insights. (1) Value-aligned LLMs are more prone to harmful behavior compared to non-fine-tuned models and exhibit slightly higher risks in traditional safety evaluations than other fine-tuned models. (2) These safety issues arise because value-aligned LLMs genuinely generate text according to the aligned values, which can amplify harmful outcomes. Using a dataset with detailed safety categories, we find significant correlations between value alignment and safety risks, supported by psychological hypotheses. This study offers insights into the "black box" of value alignment and proposes in-context alignment methods to enhance the safety of value-aligned LLMs.
>
---
#### [new 066] Bit-level BPE: Below the byte boundary
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的子词分词任务，旨在解决长尾字符导致的序列过长问题，提出一种无损压缩技术减少序列长度。**

- **链接: [http://arxiv.org/pdf/2506.07541v1](http://arxiv.org/pdf/2506.07541v1)**

> **作者:** Sangwhan Moon; Tatsuya Hiraoka; Naoaki Okazaki
>
> **摘要:** Byte-level fallbacks for subword tokenization have become a common practice in large language models. In particular, it has been demonstrated to be incredibly effective as a pragmatic solution for preventing OOV, especially in the context of larger models. However, breaking a character down to individual bytes significantly increases the sequence length for long-tail tokens in languages such as Chinese, Japanese, and Korean (CJK) and other character-diverse contexts such as emoji. The increased sequence length results in longer computation during both training and inference. In this work, we propose a simple compression technique that reduces the sequence length losslessly.
>
---
#### [new 067] What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于推理任务，旨在解决大模型推理过程冗长低效的问题。通过分析模型差异，提出FoReaL-Decoding方法，在保持性能的同时提升效率。**

- **链接: [http://arxiv.org/pdf/2506.06998v1](http://arxiv.org/pdf/2506.06998v1)**

> **作者:** Ming Li; Zhengyuan Yang; Xiyao Wang; Dianqi Li; Kevin Lin; Tianyi Zhou; Lijuan Wang
>
> **摘要:** Large reasoning models (LRMs) achieve strong reasoning performance by emitting long chains of thought. Yet, these verbose traces slow down inference and often drift into unnecessary detail, known as the overthinking phenomenon. To better understand LRMs' behavior, we systematically analyze the token-level misalignment between reasoning and non-reasoning models. While it is expected that their primary difference lies in the stylistic "thinking cues", LRMs uniquely exhibit two pivotal, previously under-explored phenomena: a Global Misalignment Rebound, where their divergence from non-reasoning models persists or even grows as response length increases, and more critically, a Local Misalignment Diminish, where the misalignment concentrates at the "thinking cues" each sentence starts with but rapidly declines in the remaining of the sentence. Motivated by the Local Misalignment Diminish, we propose FoReaL-Decoding, a collaborative fast-slow thinking decoding method for cost-quality trade-off. In FoReaL-Decoding, a Leading model leads the first few tokens for each sentence, and then a weaker draft model completes the following tokens to the end of each sentence. FoReaL-Decoding adopts a stochastic gate to smoothly interpolate between the small and the large model. On four popular math-reasoning benchmarks (AIME24, GPQA-Diamond, MATH500, AMC23), FoReaL-Decoding reduces theoretical FLOPs by 30 to 50% and trims CoT length by up to 40%, while preserving 86 to 100% of model performance. These results establish FoReaL-Decoding as a simple, plug-and-play route to controllable cost-quality trade-offs in reasoning-centric tasks.
>
---
#### [new 068] BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于逻辑推理任务，旨在解决大语言模型在处理信念不一致推理时的偏差问题。研究构建了首个大规模日语基准数据集BIS Reasoning 1.0，并评估了多个模型的表现。**

- **链接: [http://arxiv.org/pdf/2506.06955v1](http://arxiv.org/pdf/2506.06955v1)**

> **作者:** Ha-Thanh Nguyen; Chaoran Liu; Hirokazu Kiyomaru; Koichi Takeda; Yusuke Miyao; Maki Matsuda; Yusuke Oda; Pontus Stenetorp; Qianying Liu; Su Myat Noe; Hideyuki Tachibana; Kouta Nakayama; Sadao Kurohashi
>
> **摘要:** We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety.
>
---
#### [new 069] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling
- **分类: cs.CL**

- **简介: 该论文属于低资源主题建模任务，解决领域适应问题。通过引入领域对齐方法，提升模型在数据稀缺领域的性能。**

- **链接: [http://arxiv.org/pdf/2506.07453v1](http://arxiv.org/pdf/2506.07453v1)**

> **作者:** Pritom Saha Akash; Kevin Chen-Chuan Chang
>
> **摘要:** Topic modeling plays a vital role in uncovering hidden semantic structures within text corpora, but existing models struggle in low-resource settings where limited target-domain data leads to unstable and incoherent topic inference. We address this challenge by formally introducing domain adaptation for low-resource topic modeling, where a high-resource source domain informs a low-resource target domain without overwhelming it with irrelevant content. We establish a finite-sample generalization bound showing that effective knowledge transfer depends on robust performance in both domains, minimizing latent-space discrepancy, and preventing overfitting to the data. Guided by these insights, we propose DALTA (Domain-Aligned Latent Topic Adaptation), a new framework that employs a shared encoder for domain-invariant features, specialized decoders for domain-specific nuances, and adversarial alignment to selectively transfer relevant information. Experiments on diverse low-resource datasets demonstrate that DALTA consistently outperforms state-of-the-art methods in terms of topic coherence, stability, and transferability.
>
---
#### [new 070] RULE: Reinforcement UnLEarning Achieves Forget-Retain Pareto Optimality
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型的遗忘任务，旨在有效删除特定信息而不损害模型性能。提出RULE框架，通过强化学习实现高效遗忘与保留的帕累托最优。**

- **链接: [http://arxiv.org/pdf/2506.07171v1](http://arxiv.org/pdf/2506.07171v1)**

> **作者:** Chenlong Zhang; Zhuoran Jin; Hongbang Yuan; Jiaheng Wei; Tong Zhou; Kang Liu; Jun Zhao; Yubo Chen
>
> **备注:** Paper under review
>
> **摘要:** The widespread deployment of Large Language Models (LLMs) trained on massive, uncurated corpora has raised growing concerns about the inclusion of sensitive, copyrighted, or illegal content. This has led to increasing interest in LLM unlearning: the task of selectively removing specific information from a model without retraining from scratch or degrading overall utility. However, existing methods often rely on large-scale forget and retain datasets, and suffer from unnatural responses, poor generalization, or catastrophic utility loss. In this work, we propose Reinforcement UnLearning (RULE), an efficient framework that formulates unlearning as a refusal boundary optimization problem. RULE is trained with a small portion of the forget set and synthesized boundary queries, using a verifiable reward function that encourages safe refusal on forget--related queries while preserving helpful responses on permissible inputs. We provide both theoretical and empirical evidence demonstrating the effectiveness of RULE in achieving targeted unlearning without compromising model utility. Experimental results show that, with only $12%$ forget set and $8%$ synthesized boundary data, RULE outperforms existing baselines by up to $17.5%$ forget quality and $16.3%$ naturalness response while maintaining general utility, achieving forget--retain Pareto optimality. Remarkably, we further observe that RULE improves the naturalness of model outputs, enhances training efficiency, and exhibits strong generalization ability, generalizing refusal behavior to semantically related but unseen queries.
>
---
#### [new 071] What Is Seen Cannot Be Unseen: The Disruptive Effect of Knowledge Conflict on Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在知识冲突下的表现，属于模型行为分析任务。它探讨了上下文与参数知识冲突对模型的影响，并提出诊断框架进行评估。**

- **链接: [http://arxiv.org/pdf/2506.06485v1](http://arxiv.org/pdf/2506.06485v1)**

> **作者:** Kaiser Sun; Fan Bai; Mark Dredze
>
> **摘要:** Large language models frequently rely on both contextual input and parametric knowledge to perform tasks. However, these sources can come into conflict, especially when retrieved documents contradict the model's parametric knowledge. We propose a diagnostic framework to systematically evaluate LLM behavior under context-memory conflict, where the contextual information diverges from their parametric beliefs. We construct diagnostic data that elicit these conflicts and analyze model performance across multiple task types. Our findings reveal that (1) knowledge conflict has minimal impact on tasks that do not require knowledge utilization, (2) model performance is consistently higher when contextual and parametric knowledge are aligned, (3) models are unable to fully suppress their internal knowledge even when instructed, and (4) providing rationales that explain the conflict increases reliance on contexts. These insights raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs.
>
---
#### [new 072] Atomic Reasoning for Scientific Table Claim Verification
- **分类: cs.CL**

- **简介: 该论文属于科学表格声明验证任务，旨在解决模型在复杂科学表单中推理不准确的问题。通过引入原子技能和技能链 schema，提升模型的推理能力和泛化性。**

- **链接: [http://arxiv.org/pdf/2506.06972v1](http://arxiv.org/pdf/2506.06972v1)**

> **作者:** Yuji Zhang; Qingyun Wang; Cheng Qian; Jiateng Liu; Chenkai Sun; Denghui Zhang; Tarek Abdelzaher; Chengxiang Zhai; Preslav Nakov; Heng Ji
>
> **摘要:** Scientific texts often convey authority due to their technical language and complex data. However, this complexity can sometimes lead to the spread of misinformation. Non-experts are particularly susceptible to misleading claims based on scientific tables due to their high information density and perceived credibility. Existing table claim verification models, including state-of-the-art large language models (LLMs), often struggle with precise fine-grained reasoning, resulting in errors and a lack of precision in verifying scientific claims. Inspired by Cognitive Load Theory, we propose that enhancing a model's ability to interpret table-based claims involves reducing cognitive load by developing modular, reusable reasoning components (i.e., atomic skills). We introduce a skill-chaining schema that dynamically composes these skills to facilitate more accurate and generalizable reasoning with a reduced cognitive load. To evaluate this, we create SciAtomicBench, a cross-domain benchmark with fine-grained reasoning annotations. With only 350 fine-tuning examples, our model trained by atomic reasoning outperforms GPT-4o's chain-of-thought method, achieving state-of-the-art results with far less training data.
>
---
#### [new 073] Improving LLM Reasoning through Interpretable Role-Playing Steering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理能力。通过引入SRPS框架，解决传统提示工程不稳定、不可解释的问题，实现对角色扮演行为的精细控制与优化。**

- **链接: [http://arxiv.org/pdf/2506.07335v1](http://arxiv.org/pdf/2506.07335v1)**

> **作者:** Anyi Wang; Dong Shu; Yifan Wang; Yunpu Ma; Mengnan Du
>
> **备注:** 21 pages, 8 figures, 8 tables
>
> **摘要:** Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing.
>
---
#### [new 074] Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究后训练数据质量对模型性能的影响。针对公开数据集不足的问题，比较分析两个开源数据集，提出优化方案提升效果。**

- **链接: [http://arxiv.org/pdf/2506.06522v1](http://arxiv.org/pdf/2506.06522v1)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Syed Zawad; Farhan Ahmed; Heiko Ludwig; Holger Boche
>
> **摘要:** Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
>
---
#### [new 075] Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决轻量级大模型推理能力不足的问题，提出DeBoP方法通过行为优化提升其性能。**

- **链接: [http://arxiv.org/pdf/2506.06401v1](http://arxiv.org/pdf/2506.06401v1)**

> **作者:** Hongming Yang; Shi Lin; Jun Shao; Changting Lin; Donghai Zhu; Meng Han; Qinglei Kong
>
> **备注:** This work is accepted at ACL 2025
>
> **摘要:** Lightweight Large Language Models (LwLLMs) are reduced-parameter, optimized models designed to run efficiently on consumer-grade hardware, offering significant advantages in resource efficiency, cost-effectiveness, and data privacy. However, these models often struggle with limited inference and reasoning capabilities, which restrict their performance on complex tasks and limit their practical applicability. Moreover, existing prompt optimization methods typically rely on extensive manual effort or the meta-cognitive abilities of state-of-the-art LLMs, making them less effective for LwLLMs. To address these challenges, we introduce DeBoP, a new Direct Behavior Optimization Paradigm, original from the Chain-of-Thought (CoT) prompting technique. Unlike CoT Prompting, DeBoP is an automatic optimization method, which focuses on the optimization directly on the behavior of LwLLMs. In particular, DeBoP transforms the optimization of complex prompts into the optimization of discrete, quantifiable execution sequences using a gradient-free Monte Carlo Tree Search. We evaluate DeBoP on seven challenging tasks where state-of-the-art LLMs excel but LwLLMs generally underperform. Experimental results demonstrate that DeBoP significantly outperforms recent prompt optimization methods on most tasks. In particular, DeBoP-optimized LwLLMs surpass GPT-3.5 on most tasks while reducing computational time by approximately 60% compared to other automatic prompt optimization methods.
>
---
#### [new 076] Towards Large Language Models with Self-Consistent Natural Language Explanations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型解释不一致的问题。通过构建基准数据集并提出新度量标准，提升模型解释与决策的一致性。**

- **链接: [http://arxiv.org/pdf/2506.07523v1](http://arxiv.org/pdf/2506.07523v1)**

> **作者:** Sahar Admoni; Ofra Amir; Assaf Hallak; Yftah Ziser
>
> **摘要:** Large language models (LLMs) seem to offer an easy path to interpretability: just ask them to explain their decisions. Yet, studies show that these post-hoc explanations often misrepresent the true decision process, as revealed by mismatches in feature importance. Despite growing evidence of this inconsistency, no systematic solutions have emerged, partly due to the high cost of estimating feature importance, which limits evaluations to small datasets. To address this, we introduce the Post-hoc Self-Consistency Bank (PSCB) - a large-scale benchmark of decisions spanning diverse tasks and models, each paired with LLM-generated explanations and corresponding feature importance scores. Analysis of PSCB reveals that self-consistency scores barely differ between correct and incorrect predictions. We also show that the standard metric fails to meaningfully distinguish between explanations. To overcome this limitation, we propose an alternative metric that more effectively captures variation in explanation quality. We use it to fine-tune LLMs via Direct Preference Optimization (DPO), leading to significantly better alignment between explanations and decision-relevant features, even under domain shift. Our findings point to a scalable path toward more trustworthy, self-consistent LLMs.
>
---
#### [new 077] Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决小模型泛化能力不足的问题。通过融合大模型知识并微调小模型，提升性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.07424v1](http://arxiv.org/pdf/2506.07424v1)**

> **作者:** Kyeonghyun Kim; Jinhee Jang; Juhwan Choi; Yoonji Lee; Kyohoon Jin; YoungBin Kim
>
> **备注:** ACL 2025 main conference
>
> **摘要:** Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi's ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities.
>
---
#### [new 078] Detection Method for Prompt Injection by Integrating Pre-trained Model and Heuristic Feature Engineering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全检测任务，旨在解决prompt injection攻击的检测问题。通过结合预训练模型与启发式特征工程，提出DMPI-PMHFE框架，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.06384v1](http://arxiv.org/pdf/2506.06384v1)**

> **作者:** Yi Ji; Runzhi Li; Baolei Mao
>
> **备注:** Accepted by KSEM2025 AI & Sec Workshop
>
> **摘要:** With the widespread adoption of Large Language Models (LLMs), prompt injection attacks have emerged as a significant security threat. Existing defense mechanisms often face critical trade-offs between effectiveness and generalizability. This highlights the urgent need for efficient prompt injection detection methods that are applicable across a wide range of LLMs. To address this challenge, we propose DMPI-PMHFE, a dual-channel feature fusion detection framework. It integrates a pretrained language model with heuristic feature engineering to detect prompt injection attacks. Specifically, the framework employs DeBERTa-v3-base as a feature extractor to transform input text into semantic vectors enriched with contextual information. In parallel, we design heuristic rules based on known attack patterns to extract explicit structural features commonly observed in attacks. Features from both channels are subsequently fused and passed through a fully connected neural network to produce the final prediction. This dual-channel approach mitigates the limitations of relying only on DeBERTa to extract features. Experimental results on diverse benchmark datasets demonstrate that DMPI-PMHFE outperforms existing methods in terms of accuracy, recall, and F1-score. Furthermore, when deployed actually, it significantly reduces attack success rates across mainstream LLMs, including GLM-4, LLaMA 3, Qwen 2.5, and GPT-4o.
>
---
#### [new 079] LG-ANNA-Embedding technical report
- **分类: cs.CL**

- **简介: 该论文属于文本嵌入任务，旨在提升信息检索和非检索任务的泛化能力。通过结合上下文学习、软监督和自适应负样本挖掘，生成高质量嵌入。**

- **链接: [http://arxiv.org/pdf/2506.07438v1](http://arxiv.org/pdf/2506.07438v1)**

> **作者:** Jooyoung Choi; Hyun Kim; Hansol Jang; Changwook Jun; Kyunghoon Bae; Hyewon Choi; Stanley Jungkyu Choi; Honglak Lee; Chulmin Yun
>
> **备注:** 10 pages
>
> **摘要:** This report presents a unified instruction-based framework for learning generalized text embeddings optimized for both information retrieval (IR) and non-IR tasks. Built upon a decoder-only large language model (Mistral-7B), our approach combines in-context learning, soft supervision, and adaptive hard-negative mining to generate context-aware embeddings without task-specific fine-tuning. Structured instructions and few-shot examples are used to guide the model across diverse tasks, enabling strong performance on classification, semantic similarity, clustering, and reranking benchmarks. To improve semantic discrimination, we employ a soft labeling framework where continuous relevance scores, distilled from a high-performance dense retriever and reranker, serve as fine-grained supervision signals. In addition, we introduce adaptive margin-based hard-negative mining, which filters out semantically ambiguous negatives based on their similarity to positive examples, thereby enhancing training stability and retrieval robustness. Our model is evaluated on the newly introduced MTEB (English, v2) benchmark, covering 41 tasks across seven categories. Results show that our method achieves strong generalization and ranks among the top-performing models by Borda score, outperforming several larger or fully fine-tuned baselines. These findings highlight the effectiveness of combining in-context prompting, soft supervision, and adaptive sampling for scalable, high-quality embedding generation.
>
---
#### [new 080] WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型评估任务，旨在解决现有基准无法全面评估模型能力的问题。提出WebUIBench，涵盖四个关键领域，包含21K高质量数据对。**

- **链接: [http://arxiv.org/pdf/2506.07818v1](http://arxiv.org/pdf/2506.07818v1)**

> **作者:** Zhiyu Lin; Zhengda Zhou; Zhiyuan Zhao; Tianrui Wan; Yilun Ma; Junyu Gao; Xuelong Li
>
> **摘要:** With the rapid advancement of Generative AI technology, Multimodal Large Language Models(MLLMs) have the potential to act as AI software engineers capable of executing complex web application development. Considering that the model requires a confluence of multidimensional sub-capabilities to address the challenges of various development phases, constructing a multi-view evaluation framework is crucial for accurately guiding the enhancement of development efficiency. However, existing benchmarks usually fail to provide an assessment of sub-capabilities and focus solely on webpage generation outcomes. In this work, we draw inspiration from the principles of software engineering and further propose WebUIBench, a benchmark systematically designed to evaluate MLLMs in four key areas: WebUI Perception, HTML Programming,WebUI-HTML Understanding, and WebUI-to-Code. WebUIBench comprises 21K high-quality question-answer pairs derived from over 0.7K real-world websites. The extensive evaluation of 29 mainstream MLLMs uncovers the skill characteristics and various weakness that models encountered during the development process.
>
---
#### [new 081] Representation Decomposition for Learning Similarity and Contrastness Across Modalities for Affective Computing
- **分类: cs.CL**

- **简介: 该论文属于多模态情感计算任务，旨在解决跨模态情感与对比信息的复杂融合问题。通过分解视觉与文本表示，提取共享与特定特征，并整合形成动态提示，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07086v1](http://arxiv.org/pdf/2506.07086v1)**

> **作者:** Yuanhe Tian; Pengsen Cheng; Guoqing Jin; Lei Zhang; Yan Song
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Multi-modal affective computing aims to automatically recognize and interpret human attitudes from diverse data sources such as images and text, thereby enhancing human-computer interaction and emotion understanding. Existing approaches typically rely on unimodal analysis or straightforward fusion of cross-modal information that fail to capture complex and conflicting evidence presented across different modalities. In this paper, we propose a novel LLM-based approach for affective computing that explicitly deconstructs visual and textual representations into shared (modality-invariant) and modality-specific components. Specifically, our approach firstly encodes and aligns input modalities using pre-trained multi-modal encoders, then employs a representation decomposition framework to separate common emotional content from unique cues, and finally integrates these decomposed signals via an attention mechanism to form a dynamic soft prompt for a multi-modal LLM. Extensive experiments on three representative tasks for affective computing, namely, multi-modal aspect-based sentiment analysis, multi-modal emotion analysis, and hateful meme detection, demonstrate the effectiveness of our approach, which consistently outperforms strong baselines and state-of-the-art models.
>
---
#### [new 082] A Survey of Retentive Network
- **分类: cs.CL**

- **简介: 本文为Retentive Network（RetNet）的综述论文，旨在介绍其架构、创新及应用，解决Transformer在长序列处理中的效率问题。**

- **链接: [http://arxiv.org/pdf/2506.06708v1](http://arxiv.org/pdf/2506.06708v1)**

> **作者:** Haiqi Yang; Zhiyuan Li; Yi Chang; Yuan Wu
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Retentive Network (RetNet) represents a significant advancement in neural network architecture, offering an efficient alternative to the Transformer. While Transformers rely on self-attention to model dependencies, they suffer from high memory costs and limited scalability when handling long sequences due to their quadratic complexity. To mitigate these limitations, RetNet introduces a retention mechanism that unifies the inductive bias of recurrence with the global dependency modeling of attention. This mechanism enables linear-time inference, facilitates efficient modeling of extended contexts, and remains compatible with fully parallelizable training pipelines. RetNet has garnered significant research interest due to its consistently demonstrated cross-domain effectiveness, achieving robust performance across machine learning paradigms including natural language processing, speech recognition, and time-series analysis. However, a comprehensive review of RetNet is still missing from the current literature. This paper aims to fill that gap by offering the first detailed survey of the RetNet architecture, its key innovations, and its diverse applications. We also explore the main challenges associated with RetNet and propose future research directions to support its continued advancement in both academic research and practical deployment.
>
---
#### [new 083] Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决低资源环境下生成高质量对齐内容的问题。提出WSD框架，通过小模型引导大模型生成更对齐的文本。**

- **链接: [http://arxiv.org/pdf/2506.07434v1](http://arxiv.org/pdf/2506.07434v1)**

> **作者:** Feifan Song; Shaohang Wei; Wen Luo; Yuxuan Fan; Tianyu Liu; Guoyin Wang; Houfeng Wang
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) require alignment with human preferences to avoid generating offensive, false, or meaningless content. Recently, low-resource methods for LLM alignment have been popular, while still facing challenges in obtaining both high-quality and aligned content. Motivated by the observation that the difficulty of generating aligned responses is concentrated at the beginning of decoding, we propose a novel framework, Weak-to-Strong Decoding (WSD), to enhance the alignment ability of base models by the guidance of a small aligned model. The small model first drafts well-aligned beginnings, followed by the large base model to continue the rest, controlled by a well-designed auto-switch mechanism. We also collect a new dataset, GenerAlign, to fine-tune a small-sized Pilot-3B as the draft model, which effectively enhances different base models under the WSD framework to outperform all baseline methods, while avoiding degradation on downstream tasks, termed as the alignment tax. Extensive experiments are further conducted to examine the impact of different settings and time efficiency, as well as analyses on the intrinsic mechanisms of WSD in depth.
>
---
#### [new 084] Automatic Speech Recognition of African American English: Lexical and Contextual Effects
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别任务，研究AAE中CCR和ING-reduction对ASR的影响，分析无LM系统受词汇邻近效应影响更大。**

- **链接: [http://arxiv.org/pdf/2506.06888v1](http://arxiv.org/pdf/2506.06888v1)**

> **作者:** Hamid Mojarad; Kevin Tang
>
> **备注:** submitted to Interspeech 2025
>
> **摘要:** Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs.
>
---
#### [new 085] A dependently-typed calculus of event telicity and culminativity
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于自然语言处理中的事件分析任务，旨在解决事件telicity和culminativity的建模问题，提出依赖类型框架并用Agda形式化验证。**

- **链接: [http://arxiv.org/pdf/2506.06968v1](http://arxiv.org/pdf/2506.06968v1)**

> **作者:** Pavel Kovalev; Carlo Angiuli
>
> **备注:** 52 pages, Agda formalization available at https://doi.org/10.5281/zenodo.15602617
>
> **摘要:** We present a dependently-typed cross-linguistic framework for analyzing the telicity and culminativity of events, accompanied by examples of using our framework to model English sentences. Our framework consists of two parts. In the nominal domain, we model the boundedness of noun phrases and its relationship to subtyping, delimited quantities, and adjectival modification. In the verbal domain we define a dependent event calculus, modeling telic events as those whose undergoer is bounded, culminating events as telic events that achieve their inherent endpoint, and consider adverbial modification. In both domains we pay particular attention to associated entailments. Our framework is defined as an extension of intensional Martin-L\"of dependent type theory, and the rules and examples in this paper have been formalized in the Agda proof assistant.
>
---
#### [new 086] How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于GraphRAG性能评估任务，旨在解决现有评估框架的偏差问题。通过构建更相关的问答数据和无偏评估流程，重新评估了GraphRAG方法的实际效果。**

- **链接: [http://arxiv.org/pdf/2506.06331v1](http://arxiv.org/pdf/2506.06331v1)**

> **作者:** Qiming Zeng; Xiao Yan; Hao Luo; Yuhao Lin; Yuxiang Wang; Fangcheng Fu; Bo Du; Quanqing Xu; Jiawei Jiang
>
> **摘要:** By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previously. Although our evaluation framework may still have flaws, it calls for scientific evaluations to lay solid foundations for GraphRAG research.
>
---
#### [new 087] Correlated Errors in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; stat.ML**

- **简介: 该论文研究大语言模型中的错误相关性，旨在分析不同模型间错误的一致性。通过实验发现模型错误高度相关，尤其在大型准确模型中更为明显。**

- **链接: [http://arxiv.org/pdf/2506.07962v1](http://arxiv.org/pdf/2506.07962v1)**

> **作者:** Elliot Kim; Avi Garg; Kenny Peng; Nikhil Garg
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Diversity in training data, architecture, and providers is assumed to mitigate homogeneity in LLMs. However, we lack empirical evidence on whether different LLMs differ meaningfully. We conduct a large-scale empirical evaluation on over 350 LLMs overall, using two popular leaderboards and a resume-screening task. We find substantial correlation in model errors -- on one leaderboard dataset, models agree 60% of the time when both models err. We identify factors driving model correlation, including shared architectures and providers. Crucially, however, larger and more accurate models have highly correlated errors, even with distinct architectures and providers. Finally, we show the effects of correlation in two downstream tasks: LLM-as-judge evaluation and hiring -- the latter reflecting theoretical predictions regarding algorithmic monoculture.
>
---
#### [new 088] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本推理中模型注意力偏差问题。通过引入LeaF框架，识别并剔除干扰性token，提升推理准确性和生成质量。**

- **链接: [http://arxiv.org/pdf/2506.07851v1](http://arxiv.org/pdf/2506.07851v1)**

> **作者:** Yiju Guo; Wenkai Yang; Zexu Sun; Ning Ding; Zhiyuan Liu; Yankai Lin
>
> **摘要:** Large language models (LLMs) have demonstrated significant improvements in contextual understanding. However, their ability to attend to truly critical information during long-context reasoning and generation still falls behind the pace. Specifically, our preliminary experiments reveal that certain distracting patterns can misdirect the model's attention during inference, and removing these patterns substantially improves reasoning accuracy and generation quality. We attribute this phenomenon to spurious correlations in the training data, which obstruct the model's capacity to infer authentic causal instruction-response relationships. This phenomenon may induce redundant reasoning processes, potentially resulting in significant inference overhead and, more critically, the generation of erroneous or suboptimal responses. To mitigate this, we introduce a two-stage framework called Learning to Focus (LeaF) leveraging intervention-based inference to disentangle confounding factors. In the first stage, LeaF employs gradient-based comparisons with an advanced teacher to automatically identify confounding tokens based on causal relationships in the training corpus. Then, in the second stage, it prunes these tokens during distillation to enact intervention, aligning the student's attention with the teacher's focus distribution on truly critical context tokens. Experimental results demonstrate that LeaF not only achieves an absolute improvement in various mathematical reasoning and code generation benchmarks but also effectively suppresses attention to confounding tokens during inference, yielding a more interpretable and reliable reasoning model.
>
---
#### [new 089] BriefMe: A Legal NLP Benchmark for Assisting with Legal Briefs
- **分类: cs.CL**

- **简介: 该论文提出BRIEFME，一个用于法律文书的NLP基准，解决法律写作辅助问题，包含三项任务：论证摘要、补充和案例检索。**

- **链接: [http://arxiv.org/pdf/2506.06619v1](http://arxiv.org/pdf/2506.06619v1)**

> **作者:** Jesse Woo; Fateme Hashemi Chaleshtori; Ana Marasović; Kenneth Marino
>
> **备注:** ACL Findings 2025; 10 pages main, 5 pages references, 37 pages appendix
>
> **摘要:** A core part of legal work that has been under-explored in Legal NLP is the writing and editing of legal briefs. This requires not only a thorough understanding of the law of a jurisdiction, from judgments to statutes, but also the ability to make new arguments to try to expand the law in a new direction and make novel and creative arguments that are persuasive to judges. To capture and evaluate these legal skills in language models, we introduce BRIEFME, a new dataset focused on legal briefs. It contains three tasks for language models to assist legal professionals in writing briefs: argument summarization, argument completion, and case retrieval. In this work, we describe the creation of these tasks, analyze them, and show how current models perform. We see that today's large language models (LLMs) are already quite good at the summarization and guided completion tasks, even beating human-generated headings. Yet, they perform poorly on other tasks in our benchmark: realistic argument completion and retrieving relevant legal cases. We hope this dataset encourages more development in Legal NLP in ways that will specifically aid people in performing legal work.
>
---
#### [new 090] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型持续编辑任务，解决模型更新时遗忘旧知识的问题。提出MEMOIR框架，通过残差记忆模块实现高效、可靠的知识注入与保留。**

- **链接: [http://arxiv.org/pdf/2506.07899v1](http://arxiv.org/pdf/2506.07899v1)**

> **作者:** Ke Wang; Yiming Qin; Nikolaos Dimitriadis; Alessandro Favero; Pascal Frossard
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** Language models deployed in real-world systems often require post-hoc updates to incorporate new or corrected knowledge. However, editing such models efficiently and reliably - without retraining or forgetting previous information - remains a major challenge. Existing methods for lifelong model editing either compromise generalization, interfere with past edits, or fail to scale to long editing sequences. We propose MEMOIR, a novel scalable framework that injects knowledge through a residual memory, i.e., a dedicated parameter module, while preserving the core capabilities of the pre-trained model. By sparsifying input activations through sample-dependent masks, MEMOIR confines each edit to a distinct subset of the memory parameters, minimizing interference among edits. At inference, it identifies relevant edits by comparing the sparse activation patterns of new queries to those stored during editing. This enables generalization to rephrased queries by activating only the relevant knowledge while suppressing unnecessary memory activation for unrelated prompts. Experiments on question answering, hallucination correction, and out-of-distribution generalization benchmarks across LLaMA-3 and Mistral demonstrate that MEMOIR achieves state-of-the-art performance across reliability, generalization, and locality metrics, scaling to thousands of sequential edits with minimal forgetting.
>
---
#### [new 091] Reward Model Interpretability via Optimal and Pessimal Tokens
- **分类: cs.CL; cs.AI; cs.CY; cs.LG; I.2.6; I.2.7; H.5.2; J.4; K.4.2**

- **简介: 该论文属于奖励模型可解释性研究，旨在分析奖励模型如何编码人类价值观。通过分析不同模型对单个token的评分，揭示其偏差和不一致性。**

- **链接: [http://arxiv.org/pdf/2506.07326v1](http://arxiv.org/pdf/2506.07326v1)**

> **作者:** Brian Christian; Hannah Rose Kirk; Jessica A. F. Thompson; Christopher Summerfield; Tsvetomira Dumbalska
>
> **备注:** Accepted for publication in Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT '25), to appear June 2025
>
> **摘要:** Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves -- which directly encode human value judgments by turning prompt-response pairs into scalar rewards -- remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training -- distortions that risk propagating through the downstream large language models now deployed to millions.
>
---
#### [new 092] Training Superior Sparse Autoencoders for Instruct Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于机械可解释性任务，解决 instruct 模型中 SAE 训练效果不佳的问题，提出 FAST 方法提升重建质量和特征可解释性。**

- **链接: [http://arxiv.org/pdf/2506.07691v1](http://arxiv.org/pdf/2506.07691v1)**

> **作者:** Jiaming Li; Haoran Ye; Yukun Chen; Xinyue Li; Lei Zhang; Hamid Alinejad-Rokny; Jimmy Chih-Hsien Peng; Min Yang
>
> **摘要:** As large language models (LLMs) grow in scale and capability, understanding their internal mechanisms becomes increasingly critical. Sparse autoencoders (SAEs) have emerged as a key tool in mechanistic interpretability, enabling the extraction of human-interpretable features from LLMs. However, existing SAE training methods are primarily designed for base models, resulting in reduced reconstruction quality and interpretability when applied to instruct models. To bridge this gap, we propose $\underline{\textbf{F}}$inetuning-$\underline{\textbf{a}}$ligned $\underline{\textbf{S}}$equential $\underline{\textbf{T}}$raining ($\textit{FAST}$), a novel training method specifically tailored for instruct models. $\textit{FAST}$ aligns the training process with the data distribution and activation patterns characteristic of instruct models, resulting in substantial improvements in both reconstruction and feature interpretability. On Qwen2.5-7B-Instruct, $\textit{FAST}$ achieves a mean squared error of 0.6468 in token reconstruction, significantly outperforming baseline methods with errors of 5.1985 and 1.5096. In feature interpretability, $\textit{FAST}$ yields a higher proportion of high-quality features, for Llama3.2-3B-Instruct, $21.1\%$ scored in the top range, compared to $7.0\%$ and $10.2\%$ for $\textit{BT(P)}$ and $\textit{BT(F)}$. Surprisingly, we discover that intervening on the activations of special tokens via the SAEs leads to improvements in output quality, suggesting new opportunities for fine-grained control of model behavior. Code, data, and 240 trained SAEs are available at https://github.com/Geaming2002/FAST.
>
---
#### [new 093] Beyond Facts: Evaluating Intent Hallucination in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在复杂查询中产生的意图幻觉问题。通过构建基准测试FAITHQA和提出评估指标CONSTRAINT SCORE进行分析。**

- **链接: [http://arxiv.org/pdf/2506.06539v1](http://arxiv.org/pdf/2506.06539v1)**

> **作者:** Yijie Hao; Haofei Yu; Jiaxuan You
>
> **备注:** Accepted to ACL 2025 main conference
>
> **摘要:** When exposed to complex queries containing multiple conditions, today's large language models (LLMs) tend to produce responses that only partially satisfy the query while neglecting certain conditions. We therefore introduce the concept of Intent Hallucination. In this phenomenon, LLMs either omit (neglecting to address certain parts) or misinterpret (responding to invented query parts) elements of the given query, leading to intent hallucinated generation. To systematically evaluate intent hallucination, we introduce FAITHQA, a novel benchmark for intent hallucination that contains 20,068 problems, covering both query-only and retrieval-augmented generation (RAG) setups with varying topics and difficulty. FAITHQA is the first hallucination benchmark that goes beyond factual verification, tailored to identify the fundamental cause of intent hallucination. By evaluating various LLMs on FAITHQA, we find that (1) intent hallucination is a common issue even for state-of-the-art models, and (2) the phenomenon stems from omission or misinterpretation of LLMs. To facilitate future research, we introduce an automatic LLM generation evaluation metric, CONSTRAINT SCORE, for detecting intent hallucination. Human evaluation results demonstrate that CONSTRAINT SCORE is closer to human performance for intent hallucination compared to baselines.
>
---
#### [new 094] How do datasets, developers, and models affect biases in a low-resourced language?
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于自然语言处理任务，探讨低资源语言中数据、开发者和模型对偏见的影响，旨在揭示情感分析模型中的身份偏见问题。**

- **链接: [http://arxiv.org/pdf/2506.06816v1](http://arxiv.org/pdf/2506.06816v1)**

> **作者:** Dipto Das; Shion Guha; Bryan Semaan
>
> **摘要:** Sociotechnical systems, such as language technologies, frequently exhibit identity-based biases. These biases exacerbate the experiences of historically marginalized communities and remain understudied in low-resource contexts. While models and datasets specific to a language or with multilingual support are commonly recommended to address these biases, this paper empirically tests the effectiveness of such approaches in the context of gender, religion, and nationality-based identities in Bengali, a widely spoken but low-resourced language. We conducted an algorithmic audit of sentiment analysis models built on mBERT and BanglaBERT, which were fine-tuned using all Bengali sentiment analysis (BSA) datasets from Google Dataset Search. Our analyses showed that BSA models exhibit biases across different identity categories despite having similar semantic content and structure. We also examined the inconsistencies and uncertainties arising from combining pre-trained models and datasets created by individuals from diverse demographic backgrounds. We connected these findings to the broader discussions on epistemic injustice, AI alignment, and methodological decisions in algorithmic audits.
>
---
#### [new 095] Unified Game Moderation: Soft-Prompting and LLM-Assisted Label Transfer for Resource-Efficient Toxicity Detection
- **分类: cs.CL; cs.AI; I.2.7; J.4**

- **简介: 该论文属于内容安全任务，旨在解决多游戏多语言场景下的毒性检测问题。通过软提示和LLM辅助标签迁移，提升检测效率与扩展性。**

- **链接: [http://arxiv.org/pdf/2506.06347v1](http://arxiv.org/pdf/2506.06347v1)**

> **作者:** Zachary Yang; Domenico Tullo; Reihaneh Rabbany
>
> **备注:** 11 pages, 1 figure, 9 Tables, KDD 2025 ADS Track
>
> **摘要:** Toxicity detection in gaming communities faces significant scaling challenges when expanding across multiple games and languages, particularly in real-time environments where computational efficiency is crucial. We present two key findings to address these challenges while building upon our previous work on ToxBuster, a BERT-based real-time toxicity detection system. First, we introduce a soft-prompting approach that enables a single model to effectively handle multiple games by incorporating game-context tokens, matching the performance of more complex methods like curriculum learning while offering superior scalability. Second, we develop an LLM-assisted label transfer framework using GPT-4o-mini to extend support to seven additional languages. Evaluations on real game chat data across French, German, Portuguese, and Russian achieve macro F1-scores ranging from 32.96% to 58.88%, with particularly strong performance in German, surpassing the English benchmark of 45.39%. In production, this unified approach significantly reduces computational resources and maintenance overhead compared to maintaining separate models for each game and language combination. At Ubisoft, this model successfully identifies an average of 50 players, per game, per day engaging in sanctionable behavior.
>
---
#### [new 096] A Hybrid GA LLM Framework for Structured Task Optimization
- **分类: cs.CL**

- **简介: 该论文提出GA LLM框架，结合遗传算法与大语言模型，解决结构化任务优化问题，提升约束满足与解决方案质量。**

- **链接: [http://arxiv.org/pdf/2506.07483v1](http://arxiv.org/pdf/2506.07483v1)**

> **作者:** Berry Feng; Jonas Lin; Patrick Lau
>
> **备注:** 7 pages
>
> **摘要:** GA LLM is a hybrid framework that combines Genetic Algorithms with Large Language Models to handle structured generation tasks under strict constraints. Each output, such as a plan or report, is treated as a gene, and evolutionary operations like selection, crossover, and mutation are guided by the language model to iteratively improve solutions. The language model provides domain knowledge and creative variation, while the genetic algorithm ensures structural integrity and global optimization. GA LLM has proven effective in tasks such as itinerary planning, academic outlining, and business reporting, consistently producing well structured and requirement satisfying results. Its modular design also makes it easy to adapt to new tasks. Compared to using a language model alone, GA LLM achieves better constraint satisfaction and higher quality solutions by combining the strengths of both components.
>
---
#### [new 097] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Theorem-of-Thought框架，解决语言模型推理过程不透明、逻辑性弱的问题，通过三种推理代理协同工作提升推理可靠性和可解释性。**

- **链接: [http://arxiv.org/pdf/2506.07106v1](http://arxiv.org/pdf/2506.07106v1)**

> **作者:** Samir Abdaljalil; Hasan Kurban; Khalid Qaraqe; Erchin Serpedin
>
> **摘要:** Large language models (LLMs) have shown strong performance across natural language reasoning tasks, yet their reasoning processes remain brittle and difficult to interpret. Prompting techniques like Chain-of-Thought (CoT) enhance reliability by eliciting intermediate reasoning steps or aggregating multiple outputs. However, they lack mechanisms for enforcing logical structure and assessing internal coherence. We introduce Theorem-of-Thought (ToTh), a novel framework that models reasoning as collaboration among three parallel agents, each simulating a distinct mode of inference: abductive, deductive, and inductive. Each agent produces a reasoning trace, which is structured into a formal reasoning graph. To evaluate consistency, we apply Bayesian belief propagation guided by natural language inference (NLI), assigning confidence scores to each step. The most coherent graph is selected to derive the final answer. Experiments on symbolic (WebOfLies) and numerical (MultiArith) reasoning benchmarks show that ToTh consistently outperforms CoT, Self-Consistency, and CoT-Decoding across multiple LLMs, while producing interpretable and logically grounded reasoning chains. Our findings suggest a promising direction for building more robust and cognitively inspired LLM reasoning. The implementation is available at https://github.com/KurbanIntelligenceLab/theorem-of-thought.
>
---
#### [new 098] Through the Valley: Path to Effective Long CoT Training for Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究小语言模型在长链式思维训练中的性能下降问题，提出Long CoT Degradation现象并分析原因。**

- **链接: [http://arxiv.org/pdf/2506.07712v1](http://arxiv.org/pdf/2506.07712v1)**

> **作者:** Renjie Luo; Jiaxi Li; Chen Huang; Wei Lu
>
> **摘要:** Long chain-of-thought (CoT) supervision has become a common strategy to enhance reasoning in language models. While effective for large models, we identify a phenomenon we call Long CoT Degradation, in which small language models (SLMs; <=3B parameters) trained on limited long CoT data experience significant performance deterioration. Through extensive experiments on the Qwen2.5, LLaMA3 and Gemma3 families, we demonstrate that this degradation is widespread across SLMs. In some settings, models trained on only 8k long CoT examples lose up to 75% of their original performance before fine-tuning. Strikingly, we further observe that for some particularly small models, even training on 220k long CoT examples fails to recover or surpass their original performance prior to fine-tuning. Our analysis attributes this effect to error accumulation: while longer responses increase the capacity for multi-step reasoning, they also amplify the risk of compounding mistakes. Furthermore, we find that Long CoT Degradation may negatively impacts downstream reinforcement learning (RL), although this can be alleviated by sufficiently scaled supervised fine-tuning (SFT). Our findings challenge common assumptions about the benefits of long CoT training for SLMs and offer practical guidance for building more effective small-scale reasoning models.
>
---
#### [new 099] DiscoSum: Discourse-aware News Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于新闻摘要任务，旨在解决语言模型难以保持长篇论述结构的问题。通过构建数据集和提出DiscoSum算法，增强摘要的结构意识与叙事一致性。**

- **链接: [http://arxiv.org/pdf/2506.06930v1](http://arxiv.org/pdf/2506.06930v1)**

> **作者:** Alexander Spangher; Tenghao Huang; Jialiang Gu; Jiatong Shi; Muhao Chen
>
> **备注:** 8 pages, 3 figures, 10 pages in Appendix
>
> **摘要:** Recent advances in text summarization have predominantly leveraged large language models to generate concise summaries. However, language models often do not maintain long-term discourse structure, especially in news articles, where organizational flow significantly influences reader engagement. We introduce a novel approach to integrating discourse structure into summarization processes, focusing specifically on news articles across various media. We present a novel summarization dataset where news articles are summarized multiple times in different ways across different social media platforms (e.g. LinkedIn, Facebook, etc.). We develop a novel news discourse schema to describe summarization structures and a novel algorithm, DiscoSum, which employs beam search technique for structure-aware summarization, enabling the transformation of news stories to meet different stylistic and structural demands. Both human and automatic evaluation results demonstrate the efficacy of our approach in maintaining narrative fidelity and meeting structural requirements.
>
---
#### [new 100] Improving LLM-Powered EDA Assistants with RAFT
- **分类: cs.CL**

- **简介: 该论文属于电子设计自动化（EDA）领域，旨在提升LLM在EDA任务中的性能。通过RAFT和合成数据解决领域知识不足与数据获取困难的问题。**

- **链接: [http://arxiv.org/pdf/2506.06500v1](http://arxiv.org/pdf/2506.06500v1)**

> **作者:** Luyao Shi; Michael Kazda; Charles Schmitter; Hemlata Gupta
>
> **备注:** Accepted paper at IEEE International Conference on LLM-Aided Design, 2025 (LAD 2025)
>
> **摘要:** Electronic design engineers often struggle to efficiently access relevant information for tasks like design verification and technology development. While large language models (LLMs) can enhance productivity as conversational agents, pre-trained open-source LLMs lack domain-specific knowledge for Electronic Design Automation (EDA). In a Retrieval-Augmented Generation (RAG) context, LLMs rely on external context but may still produce inaccurate responses. Retrieval-Augmented Fine-Tuning (RAFT) improves LLM performance, but acquiring labeled question/answer (Q/A) data in EDA is difficult. To address this, we propose using synthetic Q/A datasets to enhance LLMs with RAFT. Our results show that RAFT with synthetic data significantly boosts LLM performance for RAG-based EDA tasks. We also investigate the impact of using real user questions as Retrieval-Augmented Few-Shot (RAFS) examples for synthetic data generation. Additionally, we implement secure access control to ensure sensitive information is only accessible to authorized personnel. Finally, we assess the risk of data leakage and unintended memorization during fine-tuning with synthetic data, providing practical insights.
>
---
#### [new 101] SDE-SQL: Enhancing Text-to-SQL Generation in Large Language Models via Self-Driven Exploration with SQL Probes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本到SQL生成任务，旨在解决LLM在推理时无法动态探索数据库的问题。通过生成并执行SQL探针，实现自驱动探索，提升模型理解数据库的能力。**

- **链接: [http://arxiv.org/pdf/2506.07245v1](http://arxiv.org/pdf/2506.07245v1)**

> **作者:** Wenxuan Xie; Yaxun Dai; Wenhao Jiang
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly improved performance on the Text-to-SQL task. However, prior approaches typically rely on static, pre-processed database information provided at inference time, which limits the model's ability to fully understand the database contents. Without dynamic interaction, LLMs are constrained to fixed, human-provided context and cannot autonomously explore the underlying data. To address this limitation, we propose SDE-SQL, a framework that enables large language models to perform self-driven exploration of databases during inference. This is accomplished by generating and executing SQL probes, which allow the model to actively retrieve information from the database and iteratively update its understanding of the data. Unlike prior methods, SDE-SQL operates in a zero-shot setting, without relying on any question-SQL pairs as in-context demonstrations. When evaluated on the BIRD benchmark with Qwen2.5-72B-Instruct, SDE-SQL achieves an 8.02% relative improvement in execution accuracy over the vanilla Qwen2.5-72B-Instruct baseline, establishing a new state-of-the-art among methods based on open-source models without supervised fine-tuning (SFT) or model ensembling. Moreover, with SFT, the performance of SDE-SQL can be further enhanced, yielding an additional 0.52% improvement.
>
---
#### [new 102] Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的句法分析任务，旨在解决代码混杂和低资源语言的语法标注问题。通过构建LLM驱动的标注流程，生成通用依存树库，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.07274v1](http://arxiv.org/pdf/2506.07274v1)**

> **作者:** Olga Kellert; Nemika Tyagi; Muhammad Imran; Nelvin Licona-Guevara; Carlos Gómez-Rodríguez
>
> **备注:** 16 pages
>
> **摘要:** Code-switching presents a complex challenge for syntactic analysis, especially in low-resource language settings where annotated data is scarce. While recent work has explored the use of large language models (LLMs) for sequence-level tagging, few approaches systematically investigate how well these models capture syntactic structure in code-switched contexts. Moreover, existing parsers trained on monolingual treebanks often fail to generalize to multilingual and mixed-language input. To address this gap, we introduce the BiLingua Parser, an LLM-based annotation pipeline designed to produce Universal Dependencies (UD) annotations for code-switched text. First, we develop a prompt-based framework for Spanish-English and Spanish-Guaran\'i data, combining few-shot LLM prompting with expert review. Second, we release two annotated datasets, including the first Spanish-Guaran\'i UD-parsed corpus. Third, we conduct a detailed syntactic analysis of switch points across language pairs and communicative contexts. Experimental results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, significantly outperforming prior baselines and multilingual parsers. These results show that LLMs, when carefully guided, can serve as practical tools for bootstrapping syntactic resources in under-resourced, code-switched environments. Data and source code are available at https://github.com/N3mika/ParsingProject
>
---
#### [new 103] Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于视频语言模型任务，旨在解决sycophancy问题，通过构建VISE基准评估并分析模型在误导输入下的表现。**

- **链接: [http://arxiv.org/pdf/2506.07180v1](http://arxiv.org/pdf/2506.07180v1)**

> **作者:** Wenrui Zhou; Shu Yang; Qingsong Yang; Zikun Guo; Lijie Hu; Di Wang
>
> **备注:** 24 pages
>
> **摘要:** As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the visual domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. In addition, we explore key-frame selection as an interpretable, training-free mitigation strategy, which reveals potential paths for reducing sycophantic bias by strengthening visual grounding.
>
---
#### [new 104] Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决语言模型后训练中的监督依赖问题。通过自信心作为奖励信号，实现少样本微调，提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2506.06395v1](http://arxiv.org/pdf/2506.06395v1)**

> **作者:** Pengyi Li; Matvey Skripkin; Alexander Zubrey; Andrey Kuznetsov; Ivan Oseledets
>
> **摘要:** Large language models (LLMs) excel at reasoning, yet post-training remains critical for aligning their behavior with task goals. Existing reinforcement learning (RL) methods often depend on costly human annotations or external reward models. We propose Reinforcement Learning via Self-Confidence (RLSC), which uses the model's own confidence as reward signals-eliminating the need for labels, preference models, or reward engineering. Applied to Qwen2.5-Math-7B with only 8 samples per question and 4 training epochs, RLSC improves accuracy by +20.10% on AIME2024, +49.40% on MATH500, and +52.50% on AMC23. RLSC offers a simple, scalable post-training method for reasoning models with minimal supervision.
>
---
#### [new 105] What Makes a Good Natural Language Prompt?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决如何量化自然语言提示的问题。通过分析150余篇论文，提出评估提示质量的框架，并探索提升方法。**

- **链接: [http://arxiv.org/pdf/2506.06950v1](http://arxiv.org/pdf/2506.06950v1)**

> **作者:** Do Xuan Long; Duy Dinh; Ngoc-Hai Nguyen; Kenji Kawaguchi; Nancy F. Chen; Shafiq Joty; Min-Yen Kan
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** As large language models (LLMs) have progressed towards more human-like and human--AI communications have become prevalent, prompting has emerged as a decisive component. However, there is limited conceptual consensus on what exactly quantifies natural language prompts. We attempt to address this question by conducting a meta-analysis surveying more than 150 prompting-related papers from leading NLP and AI conferences from 2022 to 2025 and blogs. We propose a property- and human-centric framework for evaluating prompt quality, encompassing 21 properties categorized into six dimensions. We then examine how existing studies assess their impact on LLMs, revealing their imbalanced support across models and tasks, and substantial research gaps. Further, we analyze correlations among properties in high-quality natural language prompts, deriving prompting recommendations. We then empirically explore multi-property prompt enhancements in reasoning tasks, observing that single-property enhancements often have the greatest impact. Finally, we discover that instruction-tuning on property-enhanced prompts can result in better reasoning models. Our findings establish a foundation for property-centric prompt evaluation and optimization, bridging the gaps between human--AI communication and opening new prompting research directions.
>
---
#### [new 106] Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于跨分词器迁移任务，解决预训练模型分词器移植问题。通过正交匹配追踪方法，在无需训练的情况下重建未见token嵌入，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.06607v1](http://arxiv.org/pdf/2506.06607v1)**

> **作者:** Charles Goddard; Fernando Fernandes Neto
>
> **摘要:** We present a training-free method to transplant tokenizers in pretrained large language models (LLMs) by reconstructing unseen token embeddings via Orthogonal Matching Pursuit (OMP). Specifically, we approximate each out-of-vocabulary token as a sparse linear combination of shared tokens, in two phases: first, compute each new token's representation in the donor embedding space with a small dictionary of shared anchor tokens, then transfer these same sparse coefficients back into the base model's embedding space. On two challenging cross-tokenizer tasks--Llama$\to$Mistral NeMo (12B) and Qwen$\to$Llama (1B)--we show that OMP achieves best zero-shot preservation of the base model's performance across multiple benchmarks, while other zero-shot approaches degrade significantly. Compared to baselines (zero-init, mean-init, and existing approaches like WECHSEL, FOCUS, ZETT), OMP consistently achieves the best overall performance, effectively bridging large tokenizer discrepancies without gradient updates. Our analysis further identifies mismatched numerical tokenization schemes as a critical challenge for preserving mathematical reasoning capabilities. This technique enables direct reuse of pretrained model weights with new tokenizers, facilitating cross-tokenizer knowledge distillation, speculative decoding, ensembling, merging, and domain-specific vocabulary adaptations. We integrate our method into the open-source mergekit-tokensurgeon tool for post hoc vocabulary realignment.
>
---
#### [new 107] CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CCI4.0，一个用于提升大语言模型推理能力的双语预训练数据集，解决数据质量与推理模板不足的问题，通过高质量数据筛选和CoT提取提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07463v1](http://arxiv.org/pdf/2506.07463v1)**

> **作者:** Guang Liu; Liangdong Wang; Jijie Li; Yang Yu; Yao Xu; Jiabei Chen; Yu Bai; Feng Liao; Yonghua Lin
>
> **摘要:** We introduce CCI4.0, a large-scale bilingual pre-training dataset engineered for superior data quality and diverse human-like reasoning trajectory. CCI4.0 occupies roughly $35$ TB of disk space and comprises two sub-datasets: CCI4.0-M2-Base and CCI4.0-M2-CoT. CCI4.0-M2-Base combines a $5.2$ TB carefully curated Chinese web corpus, a $22.5$ TB English subset from Nemotron-CC, and diverse sources from math, wiki, arxiv, and code. Although these data are mostly sourced from well-processed datasets, the quality standards of various domains are dynamic and require extensive expert experience and labor to process. So, we propose a novel pipeline justifying data quality mainly based on models through two-stage deduplication, multiclassifier quality scoring, and domain-aware fluency filtering. We extract $4.5$ billion pieces of CoT(Chain-of-Thought) templates, named CCI4.0-M2-CoT. Differing from the distillation of CoT from larger models, our proposed staged CoT extraction exemplifies diverse reasoning patterns and significantly decreases the possibility of hallucination. Empirical evaluations demonstrate that LLMs pre-trained in CCI4.0 benefit from cleaner, more reliable training signals, yielding consistent improvements in downstream tasks, especially in math and code reflection tasks. Our results underscore the critical role of rigorous data curation and human thinking templates in advancing LLM performance, shedding some light on automatically processing pretraining corpora.
>
---
#### [new 108] DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在解决专业领域中LLM生成文本的零样本检测问题。通过提出DivScore框架提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.06705v1](http://arxiv.org/pdf/2506.06705v1)**

> **作者:** Zhihui Chen; Kai He; Yucheng Huang; Yunxiao Zhu; Mengling Feng
>
> **备注:** Zhihui Chen and Kai He contributed equally to this work, Mengling Feng is the corresponding author
>
> **摘要:** Detecting LLM-generated text in specialized and high-stakes domains like medicine and law is crucial for combating misinformation and ensuring authenticity. However, current zero-shot detectors, while effective on general text, often fail when applied to specialized content due to domain shift. We provide a theoretical analysis showing this failure is fundamentally linked to the KL divergence between human, detector, and source text distributions. To address this, we propose DivScore, a zero-shot detection framework using normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. We also release a domain-specific benchmark for LLM-generated text detection in the medical and legal domains. Experiments on our benchmark show that DivScore consistently outperforms state-of-the-art detectors, with 14.4% higher AUROC and 64.0% higher recall (0.1% false positive rate threshold). In adversarial settings, DivScore demonstrates superior robustness than other baselines, achieving on average 22.8% advantage in AUROC and 29.5% in recall. Code and data are publicly available.
>
---
#### [new 109] Biases Propagate in Encoder-based Vision-Language Models: A Systematic Analysis From Intrinsic Measures to Zero-shot Retrieval Outcomes
- **分类: cs.CL**

- **简介: 该论文研究编码器基础视觉语言模型中的社会群体偏见传播问题，通过分析内在与外在偏见的关联，揭示偏见如何影响零样本检索任务。**

- **链接: [http://arxiv.org/pdf/2506.06506v1](http://arxiv.org/pdf/2506.06506v1)**

> **作者:** Kshitish Ghate; Tessa Charlesworth; Mona Diab; Aylin Caliskan
>
> **备注:** Accepted to ACL Findings 2025
>
> **摘要:** To build fair AI systems we need to understand how social-group biases intrinsic to foundational encoder-based vision-language models (VLMs) manifest in biases in downstream tasks. In this study, we demonstrate that intrinsic biases in VLM representations systematically ``carry over'' or propagate into zero-shot retrieval tasks, revealing how deeply rooted biases shape a model's outputs. We introduce a controlled framework to measure this propagation by correlating (a) intrinsic measures of bias in the representational space with (b) extrinsic measures of bias in zero-shot text-to-image (TTI) and image-to-text (ITT) retrieval. Results show substantial correlations between intrinsic and extrinsic bias, with an average $\rho$ = 0.83 $\pm$ 0.10. This pattern is consistent across 114 analyses, both retrieval directions, six social groups, and three distinct VLMs. Notably, we find that larger/better-performing models exhibit greater bias propagation, a finding that raises concerns given the trend towards increasingly complex AI models. Our framework introduces baseline evaluation tasks to measure the propagation of group and valence signals. Investigations reveal that underrepresented groups experience less robust propagation, further skewing their model-related outcomes.
>
---
#### [new 110] Enhancing Decision-Making of Large Language Models via Actor-Critic
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决LLM在复杂决策中的长期推理与目标对齐问题。提出LAC框架，通过Q值评估和梯度自由策略优化提升决策能力。**

- **链接: [http://arxiv.org/pdf/2506.06376v1](http://arxiv.org/pdf/2506.06376v1)**

> **作者:** Heng Dong; Kefei Duan; Chongjie Zhang
>
> **备注:** Forty-second International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable advancements in natural language processing tasks, yet they encounter challenges in complex decision-making scenarios that require long-term reasoning and alignment with high-level objectives. Existing methods either rely on short-term auto-regressive action generation or face limitations in accurately simulating rollouts and assessing outcomes, leading to sub-optimal decisions. This paper introduces a novel LLM-based Actor-Critic framework, termed LAC, that effectively improves LLM policies with long-term action evaluations in a principled and scalable way. Our approach addresses two key challenges: (1) extracting robust action evaluations by computing Q-values via token logits associated with positive/negative outcomes, enhanced by future trajectory rollouts and reasoning; and (2) enabling efficient policy improvement through a gradient-free mechanism. Experiments across diverse environments -- including high-level decision-making (ALFWorld), low-level action spaces (BabyAI-Text), and large action spaces (WebShop) -- demonstrate the framework's generality and superiority over state-of-the-art methods. Notably, our approach achieves competitive performance using 7B/8B parameter LLMs, even outperforming baseline methods employing GPT-4 in complex tasks. These results underscore the potential of integrating structured policy optimization with LLMs' intrinsic knowledge to advance decision-making capabilities in multi-step environments.
>
---
#### [new 111] Dynamic and Parametric Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决传统RAG系统在复杂任务中的不足，通过动态和参数化方法提升知识检索与生成的效果。**

- **链接: [http://arxiv.org/pdf/2506.06704v1](http://arxiv.org/pdf/2506.06704v1)**

> **作者:** Weihang Su; Qingyao Ai; Jingtao Zhan; Qian Dong; Yiqun Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a foundational paradigm for equipping large language models (LLMs) with external knowledge, playing a critical role in information retrieval and knowledge-intensive applications. However, conventional RAG systems typically adopt a static retrieve-then-generate pipeline and rely on in-context knowledge injection, which can be suboptimal for complex tasks that require multihop reasoning, adaptive information access, and deeper integration of external knowledge. Motivated by these limitations, the research community has moved beyond static retrieval and in-context knowledge injection. Among the emerging directions, this tutorial delves into two rapidly growing and complementary research areas on RAG: Dynamic RAG and Parametric RAG. Dynamic RAG adaptively determines when and what to retrieve during the LLM's generation process, enabling real-time adaptation to the LLM's evolving information needs. Parametric RAG rethinks how retrieved knowledge should be injected into LLMs, transitioning from input-level to parameter-level knowledge injection for enhanced efficiency and effectiveness. This tutorial offers a comprehensive overview of recent advances in these emerging research areas. It also shares theoretical foundations and practical insights to support and inspire further research in RAG.
>
---
#### [new 112] Semantic-preserved Augmentation with Confidence-weighted Fine-tuning for Aspect Category Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于方面类别情感分析任务，解决数据稀缺问题。通过语义保留的数据增强和置信度加权微调，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07148v1](http://arxiv.org/pdf/2506.07148v1)**

> **作者:** Yaping Chai; Haoran Xie; Joe S. Qin
>
> **备注:** 10 pages, 7 figures, 4 tables
>
> **摘要:** Large language model (LLM) is an effective approach to addressing data scarcity in low-resource scenarios. Recent existing research designs hand-crafted prompts to guide LLM for data augmentation. We introduce a data augmentation strategy for the aspect category sentiment analysis (ACSA) task that preserves the original sentence semantics and has linguistic diversity, specifically by providing a structured prompt template for an LLM to generate predefined content. In addition, we employ a post-processing technique to further ensure semantic consistency between the generated sentence and the original sentence. The augmented data increases the semantic coverage of the training distribution, enabling the model better to understand the relationship between aspect categories and sentiment polarities, enhancing its inference capabilities. Furthermore, we propose a confidence-weighted fine-tuning strategy to encourage the model to generate more confident and accurate sentiment polarity predictions. Compared with powerful and recent works, our method consistently achieves the best performance on four benchmark datasets over all baselines.
>
---
#### [new 113] Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的安全评估任务，旨在检测低资源语言中LLMs的脆弱性。通过构造简单攻击方法，验证了LLMs在波兰语等语言中的安全机制不足。**

- **链接: [http://arxiv.org/pdf/2506.07645v1](http://arxiv.org/pdf/2506.07645v1)**

> **作者:** Maciej Chrabąszcz; Katarzyna Lorenc; Karolina Seweryn
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks in recent years. However, their susceptibility to jailbreaks and perturbations necessitates additional evaluations. Many LLMs are multilingual, but safety-related training data contains mainly high-resource languages like English. This can leave them vulnerable to perturbations in low-resource languages such as Polish. We show how surprisingly strong attacks can be cheaply created by altering just a few characters and using a small proxy model for word importance calculation. We find that these character and word-level attacks drastically alter the predictions of different LLMs, suggesting a potential vulnerability that can be used to circumvent their internal safety mechanisms. We validate our attack construction methodology on Polish, a low-resource language, and find potential vulnerabilities of LLMs in this language. Additionally, we show how it can be extended to other languages. We release the created datasets and code for further research.
>
---
#### [new 114] On the Adaptive Psychological Persuasion of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的心理说服任务，旨在提升大模型的说服能力。通过分析策略有效性并提出自适应框架，优化模型在不同情境下的说服效果。**

- **链接: [http://arxiv.org/pdf/2506.06800v1](http://arxiv.org/pdf/2506.06800v1)**

> **作者:** Tianjie Ju; Yujia Chen; Hao Fei; Mong-Li Lee; Wynne Hsu; Pengzhou Cheng; Zongru Wu; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Working in progress
>
> **摘要:** Previous work has showcased the intriguing capabilities of Large Language Models (LLMs) in instruction-following and rhetorical fluency. However, systematic exploration of their dual capabilities to autonomously persuade and resist persuasion, particularly in contexts involving psychological rhetoric, remains unexplored. In this paper, we first evaluate four commonly adopted LLMs by tasking them to alternately act as persuaders and listeners in adversarial dialogues. Empirical results show that persuader LLMs predominantly employ repetitive strategies, leading to low success rates. Then we introduce eleven comprehensive psychological persuasion strategies, finding that explicitly instructing LLMs to adopt specific strategies such as Fluency Effect and Repetition Effect significantly improves persuasion success rates. However, no ``one-size-fits-all'' strategy proves universally effective, with performance heavily dependent on contextual counterfactuals. Motivated by these observations, we propose an adaptive framework based on direct preference optimization that trains LLMs to autonomously select optimal strategies by leveraging persuasion results from strategy-specific responses as preference pairs. Experiments on three open-source LLMs confirm that the proposed adaptive psychological persuasion method effectively enables persuader LLMs to select optimal strategies, significantly enhancing their success rates while maintaining general capabilities. Our code is available at https://github.com/KalinaEine/PsychologicalPersuasion.
>
---
#### [new 115] Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking
- **分类: cs.CL; cs.AI; cs.SC**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在分布变化下的推理鲁棒性问题。通过强化学习提升模型的抽象推理能力，以增强其应对变量变化和干扰信息的能力。**

- **链接: [http://arxiv.org/pdf/2506.07751v1](http://arxiv.org/pdf/2506.07751v1)**

> **作者:** Silin Gao; Antoine Bosselut; Samy Bengio; Emmanuel Abbe
>
> **备注:** Under review
>
> **摘要:** Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks.
>
---
#### [new 116] Advancing Question Generation with Joint Narrative and Difficulty Control
- **分类: cs.CL**

- **简介: 该论文属于问答生成任务，旨在解决同时控制问题叙事和难度的问题，提出联合控制策略以生成更符合教育需求的阅读理解题。**

- **链接: [http://arxiv.org/pdf/2506.06812v1](http://arxiv.org/pdf/2506.06812v1)**

> **作者:** Bernardo Leite; Henrique Lopes Cardoso
>
> **备注:** Preprint. Accepted to the BEA 2025 Workshop (ACL)
>
> **摘要:** Question Generation (QG), the task of automatically generating questions from a source input, has seen significant progress in recent years. Difficulty-controllable QG (DCQG) enables control over the difficulty level of generated questions while considering the learner's ability. Additionally, narrative-controllable QG (NCQG) allows control over the narrative aspects embedded in the questions. However, research in QG lacks a focus on combining these two types of control, which is important for generating questions tailored to educational purposes. To address this gap, we propose a strategy for Joint Narrative and Difficulty Control, enabling simultaneous control over these two attributes in the generation of reading comprehension questions. Our evaluation provides preliminary evidence that this approach is feasible, though it is not effective across all instances. Our findings highlight the conditions under which the strategy performs well and discuss the trade-offs associated with its application.
>
---
#### [new 117] How Far Are We from Optimal Reasoning Efficiency?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型优化任务，旨在解决大模型推理效率低的问题。通过定义效率前沿和REG指标，提出REO-RL算法提升推理效率。**

- **链接: [http://arxiv.org/pdf/2506.07104v1](http://arxiv.org/pdf/2506.07104v1)**

> **作者:** Jiaxuan Gao; Shu Yan; Qixin Tan; Lu Yang; Shusheng Xu; Wei Fu; Zhiyu Mei; Kaifeng Lyu; Yi Wu
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate remarkable problem-solving capabilities through extended Chain-of-Thought (CoT) reasoning but often produce excessively verbose and redundant reasoning traces. This inefficiency incurs high inference costs and limits practical deployment. While existing fine-tuning methods aim to improve reasoning efficiency, assessing their efficiency gains remains challenging due to inconsistent evaluations. In this work, we introduce the reasoning efficiency frontiers, empirical upper bounds derived from fine-tuning base LRMs across diverse approaches and training configurations. Based on these frontiers, we propose the Reasoning Efficiency Gap (REG), a unified metric quantifying deviations of any fine-tuned LRMs from these frontiers. Systematic evaluation on challenging mathematical benchmarks reveals significant gaps in current methods: they either sacrifice accuracy for short length or still remain inefficient under tight token budgets. To reduce the efficiency gap, we propose REO-RL, a class of Reinforcement Learning algorithms that minimizes REG by targeting a sparse set of token budgets. Leveraging numerical integration over strategically selected budgets, REO-RL approximates the full efficiency objective with low error using a small set of token budgets. Through systematic benchmarking, we demonstrate that our efficiency metric, REG, effectively captures the accuracy-length trade-off, with low-REG methods reducing length while maintaining accuracy. Our approach, REO-RL, consistently reduces REG by >=50 across all evaluated LRMs and matching Qwen3-4B/8B efficiency frontiers under a 16K token budget with minimal accuracy loss. Ablation studies confirm the effectiveness of our exponential token budget strategy. Finally, our findings highlight that fine-tuning LRMs to perfectly align with the efficiency frontiers remains an open challenge.
>
---
#### [new 118] Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI模型优化任务，探讨CoT提示的有效性。研究发现CoT在不同模型和任务中效果差异大，可能增加成本且收益有限。**

- **链接: [http://arxiv.org/pdf/2506.07142v1](http://arxiv.org/pdf/2506.07142v1)**

> **作者:** Lennart Meincke; Ethan Mollick; Lilach Mollick; Dan Shapiro
>
> **摘要:** This is the second in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate Chain-of-Thought (CoT) prompting, a technique that encourages a large language model (LLM) to "think step by step" (Wei et al., 2022). CoT is a widely adopted method for improving reasoning tasks, however, our findings reveal a more nuanced picture of its effectiveness. We demonstrate two things: - The effectiveness of Chain-of-Thought prompting can vary greatly depending on the type of task and model. For non-reasoning models, CoT generally improves average performance by a small amount, particularly if the model does not inherently engage in step-by-step processing by default. However, CoT can introduce more variability in answers, sometimes triggering occasional errors in questions the model would otherwise get right. We also found that many recent models perform some form of CoT reasoning even if not asked; for these models, a request to perform CoT had little impact. Performing CoT generally requires far more tokens (increasing cost and time) than direct answers. - For models designed with explicit reasoning capabilities, CoT prompting often results in only marginal, if any, gains in answer accuracy. However, it significantly increases the time and tokens needed to generate a response.
>
---
#### [new 119] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch
- **分类: cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于内容审核任务，旨在评估Twitch自动审核系统AutoMod在识别仇恨言论方面的效果。研究发现其存在显著漏洞，无法有效识别非明显仇恨内容，且误删正常用语。**

- **链接: [http://arxiv.org/pdf/2506.07667v1](http://arxiv.org/pdf/2506.07667v1)**

> **作者:** Prarabdh Shukla; Wei Yin Chong; Yash Patel; Brennan Schaffner; Danish Pruthi; Arjun Bhagoji
>
> **摘要:** To meet the demands of content moderation, online platforms have resorted to automated systems. Newer forms of real-time engagement($\textit{e.g.}$, users commenting on live streams) on platforms like Twitch exert additional pressures on the latency expected of such moderation systems. Despite their prevalence, relatively little is known about the effectiveness of these systems. In this paper, we conduct an audit of Twitch's automated moderation tool ($\texttt{AutoMod}$) to investigate its effectiveness in flagging hateful content. For our audit, we create streaming accounts to act as siloed test beds, and interface with the live chat using Twitch's APIs to send over $107,000$ comments collated from $4$ datasets. We measure $\texttt{AutoMod}$'s accuracy in flagging blatantly hateful content containing misogyny, racism, ableism and homophobia. Our experiments reveal that a large fraction of hateful messages, up to $94\%$ on some datasets, $\textit{bypass moderation}$. Contextual addition of slurs to these messages results in $100\%$ removal, revealing $\texttt{AutoMod}$'s reliance on slurs as a moderation signal. We also find that contrary to Twitch's community guidelines, $\texttt{AutoMod}$ blocks up to $89.5\%$ of benign examples that use sensitive words in pedagogical or empowering contexts. Overall, our audit points to large gaps in $\texttt{AutoMod}$'s capabilities and underscores the importance for such systems to understand context effectively.
>
---
#### [new 120] GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization
- **分类: cs.CL**

- **简介: 该论文属于几何推理任务，旨在提升大语言模型的几何解题能力。针对现有方法效果不佳或成本高的问题，提出GCPO框架，优化辅助构造与几何推理结合。**

- **链接: [http://arxiv.org/pdf/2506.07160v1](http://arxiv.org/pdf/2506.07160v1)**

> **作者:** Yikun Wang; Yibin Wang; Dianyi Wang; Zimian Peng; Qipeng Guo; Dacheng Tao; Jiaqi Wang
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, particularly in mathematical reasoning, amid which geometry problem solving remains a challenging area where auxiliary construction plays a enssential role. Existing approaches either achieve suboptimal performance or rely on massive LLMs (e.g., GPT-4o), incurring massive computational costs. We posit that reinforcement learning with verifiable reward (e.g., GRPO) offers a promising direction for training smaller models that effectively combine auxiliary construction with robust geometric reasoning. However, directly applying GRPO to geometric reasoning presents fundamental limitations due to its dependence on unconditional rewards, which leads to indiscriminate and counterproductive auxiliary constructions. To address these challenges, we propose Group Contrastive Policy Optimization (GCPO), a novel reinforcement learning framework featuring two key innovations: (1) Group Contrastive Masking, which adaptively provides positive or negative reward signals for auxiliary construction based on contextual utility, and a (2) length reward that promotes longer reasoning chains. Building on GCPO, we develop GeometryZero, a family of affordable-size geometric reasoning models that judiciously determine when to employ auxiliary construction. Our extensive empirical evaluation across popular geometric benchmarks (Geometry3K, MathVista) demonstrates that GeometryZero models consistently outperform baselines (e.g. GRPO), achieving an average improvement of 4.29% across all benchmarks.
>
---
#### [new 121] Conjoined Predication and Scalar Implicature
- **分类: cs.CL**

- **简介: 该论文属于语义与语用研究，旨在解决Magri提出的第一个语义谜题。通过分析连词和标量含义的交互作用，揭示句子歧义的根源。**

- **链接: [http://arxiv.org/pdf/2506.07429v1](http://arxiv.org/pdf/2506.07429v1)**

> **作者:** Ratna Kandala
>
> **摘要:** Magri (2016) investigates two puzzles arising from conjunction. Although Magri has proposed a solution to the second puzzle, the first remains unresolved. This first puzzle reveals a hidden interaction among quantification, collective/concurrent interpretation, and contextual updating dimensions that have yet to be explored. In essence, the problem is that certain forms of sentences like "Some Italians come from a warm country," when conjoined as in "(Only) Some Italians come from a warm country and are blond," sound infelicitous, even though no obvious alternative triggers a conflicting scalar implicature. In this paper, we offer a conceptual analysis of Magri's first puzzle by situating it within its original theoretical framework. We argue that the oddness arises from the collective or concurrent reading of the conjunctive predicate: in examples such as "(Only) Some Italians come from a warm country and are blond," this interpretation generates an indirect contextual contradiction. Moreover, we suggest that the pragmatic mechanisms governing scalar implicature generation extend beyond what is captured by exhaustification-based grammatical licensing accounts.
>
---
#### [new 122] MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **简介: 该论文提出MultiMatch，用于半监督文本分类任务，解决伪标签质量和数据不平衡问题，通过多头一致性正则化提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07801v1](http://arxiv.org/pdf/2506.07801v1)**

> **作者:** Iustin Sirbu; Robert-Adrian Popovici; Cornelia Caragea; Stefan Trausan-Matu; Traian Rebedea
>
> **摘要:** We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a three-fold pseudo-label weighting module designed for three key purposes: selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, achieving state-of-the-art results on 9 out of 10 setups from 5 natural language processing datasets and ranking first according to the Friedman test among 19 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26% -- and data imbalance is a key factor for many text classification tasks.
>
---
#### [new 123] Mixture of Small and Large Models for Chinese Spelling Check
- **分类: cs.CL**

- **简介: 该论文属于中文拼写检查任务，旨在解决LLM在纠错中准确率不足和BERT模型过拟合的问题。提出动态混合方法，在解码阶段结合小模型与大模型的概率分布，提升纠错效果。**

- **链接: [http://arxiv.org/pdf/2506.06887v1](http://arxiv.org/pdf/2506.06887v1)**

> **作者:** Ziheng Qiao; Houquan Zhou; Zhenghua Li
>
> **摘要:** In the era of large language models (LLMs), the Chinese Spelling Check (CSC) task has seen various LLM methods developed, yet their performance remains unsatisfactory. In contrast, fine-tuned BERT-based models, relying on high-quality in-domain data, show excellent performance but suffer from edit pattern overfitting. This paper proposes a novel dynamic mixture approach that effectively combines the probability distributions of small models and LLMs during the beam search decoding phase, achieving a balanced enhancement of precise corrections from small models and the fluency of LLMs. This approach also eliminates the need for fine-tuning LLMs, saving significant time and resources, and facilitating domain adaptation. Comprehensive experiments demonstrate that our mixture approach significantly boosts error correction capabilities, achieving state-of-the-art results across multiple datasets. Our code is available at https://github.com/zhqiao-nlp/MSLLM.
>
---
#### [new 124] Beyond the Sentence: A Survey on Context-Aware Machine Translation with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，探讨如何利用大语言模型进行上下文感知的翻译。研究分析了现有方法，比较了商业与开源模型的效果，并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2506.07583v1](http://arxiv.org/pdf/2506.07583v1)**

> **作者:** Ramakrishna Appicharla; Baban Gain; Santanu Pal; Asif Ekbal
>
> **摘要:** Despite the popularity of the large language models (LLMs), their application to machine translation is relatively underexplored, especially in context-aware settings. This work presents a literature review of context-aware translation with LLMs. The existing works utilise prompting and fine-tuning approaches, with few focusing on automatic post-editing and creating translation agents for context-aware machine translation. We observed that the commercial LLMs (such as ChatGPT and Tower LLM) achieved better results than the open-source LLMs (such as Llama and Bloom LLMs), and prompt-based approaches serve as good baselines to assess the quality of translations. Finally, we present some interesting future directions to explore.
>
---
#### [new 125] Extending dependencies to the taggedPBC: Word order in transitive clauses
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的依存句法分析任务，旨在为taggedPBC数据集添加依赖信息，解决跨语言语序研究问题，并验证其与现有数据库的一致性。**

- **链接: [http://arxiv.org/pdf/2506.06785v1](http://arxiv.org/pdf/2506.06785v1)**

> **作者:** Hiram Ring
>
> **摘要:** The taggedPBC (Ring 2025a) contains more than 1,800 sentences of pos-tagged parallel text data from over 1,500 languages, representing 133 language families and 111 isolates. While this dwarfs previously available resources, and the POS tags achieve decent accuracy, allowing for predictive crosslinguistic insights (Ring 2025b), the dataset was not initially annotated for dependencies. This paper reports on a CoNLLU-formatted version of the dataset which transfers dependency information along with POS tags to all languages in the taggedPBC. Although there are various concerns regarding the quality of the tags and the dependencies, word order information derived from this dataset regarding the position of arguments and predicates in transitive clauses correlates with expert determinations of word order in three typological databases (WALS, Grambank, Autotyp). This highlights the usefulness of corpus-based typological approaches (as per Baylor et al. 2023; Bjerva 2024) for extending comparisons of discrete linguistic categories, and suggests that important insights can be gained even from noisy data, given sufficient annotation. The dependency-annotated corpora are also made available for research and collaboration via GitHub.
>
---
#### [new 126] KG2QA: Knowledge Graph-enhanced Retrieval-Augmented Generation for Communication Standards Question Answering
- **分类: cs.CL**

- **简介: 该论文属于通信标准问答任务，旨在解决传统咨询效率低的问题。通过结合大模型微调与知识图谱，构建了智能问答系统，提升了问答效果和准确性。**

- **链接: [http://arxiv.org/pdf/2506.07037v1](http://arxiv.org/pdf/2506.07037v1)**

> **作者:** Zhongze Luo; Weixuan Wan; Qizhi Zheng; Yanhong Bai; Jingyun Sun; Jian Wang; Dan Wang
>
> **备注:** 23 pages
>
> **摘要:** There are many types of standards in the field of communication. The traditional consulting model has a long cycle and relies on the knowledge and experience of experts, making it difficult to meet the rapidly developing technological demands. This paper combines the fine-tuning of large language models with the construction of knowledge graphs to implement an intelligent consultation and question-answering system for communication standards. The experimental results show that after LoRA tuning on the constructed dataset of 6,587 questions and answers in the field of communication standards, Qwen2.5-7B-Instruct demonstrates outstanding professional capabilities in the field of communication standards on the test set. BLEU-4 rose from 18.8564 to 66.8993, and evaluation indicators such as ROUGE also increased significantly, outperforming the fine-tuning effect of the comparison model Llama-3-8B-Instruct. Based on the ontology framework containing 6 entity attributes and 10 relation attributes, a knowledge graph of the communication standard domain containing 13,906 entities and 13,524 relations was constructed, showing a relatively good query accuracy rate. The intelligent consultation and question-answering system enables the fine-tuned model on the server side to access the locally constructed knowledge graph and conduct graphical retrieval of key information first, which is conducive to improving the question-answering effect. The evaluation using DeepSeek as the Judge on the test set shows that our RAG framework enables the fine-tuned model to improve the scores at all five angles, with an average score increase of 2.26%. And combined with web services and API interfaces, it has achieved very good results in terms of interaction experience and back-end access, and has very good practical application value.
>
---
#### [new 127] Cultural Bias Matters: A Cross-Cultural Benchmark Dataset and Sentiment-Enriched Model for Understanding Multimodal Metaphors
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决跨文化语境下多模态隐喻理解中的文化偏见问题。研究构建了多文化多模态隐喻数据集MultiMM，并提出SEMD模型以提升跨文化隐喻识别效果。**

- **链接: [http://arxiv.org/pdf/2506.06987v1](http://arxiv.org/pdf/2506.06987v1)**

> **作者:** Senqi Yang; Dongyu Zhang; Jing Ren; Ziqi Xu; Xiuzhen Zhang; Yiliao Song; Hongfei Lin; Feng Xia
>
> **备注:** This paper has been accepted to the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Main Conference
>
> **摘要:** Metaphors are pervasive in communication, making them crucial for natural language processing (NLP). Previous research on automatic metaphor processing predominantly relies on training data consisting of English samples, which often reflect Western European or North American biases. This cultural skew can lead to an overestimation of model performance and contributions to NLP progress. However, the impact of cultural bias on metaphor processing, particularly in multimodal contexts, remains largely unexplored. To address this gap, we introduce MultiMM, a Multicultural Multimodal Metaphor dataset designed for cross-cultural studies of metaphor in Chinese and English. MultiMM consists of 8,461 text-image advertisement pairs, each accompanied by fine-grained annotations, providing a deeper understanding of multimodal metaphors beyond a single cultural domain. Additionally, we propose Sentiment-Enriched Metaphor Detection (SEMD), a baseline model that integrates sentiment embeddings to enhance metaphor comprehension across cultural backgrounds. Experimental results validate the effectiveness of SEMD on metaphor detection and sentiment analysis tasks. We hope this work increases awareness of cultural bias in NLP research and contributes to the development of fairer and more inclusive language models. Our dataset and code are available at https://github.com/DUTIR-YSQ/MultiMM.
>
---
#### [new 128] DeRAGEC: Denoising Named Entity Candidates with Synthetic Rationale for ASR Error Correction
- **分类: cs.CL**

- **简介: 该论文属于ASR错误纠正任务，旨在提升命名实体识别的准确性。通过合成理由过滤噪声候选实体，无需额外训练即可改进纠错效果。**

- **链接: [http://arxiv.org/pdf/2506.07510v1](http://arxiv.org/pdf/2506.07510v1)**

> **作者:** Solee Im; Wonjun Lee; Jinmyeong An; Yunsu Kim; Jungseul Ok; Gary Geunbae Lee
>
> **备注:** ACL2025 Findings
>
> **摘要:** We present DeRAGEC, a method for improving Named Entity (NE) correction in Automatic Speech Recognition (ASR) systems. By extending the Retrieval-Augmented Generative Error Correction (RAGEC) framework, DeRAGEC employs synthetic denoising rationales to filter out noisy NE candidates before correction. By leveraging phonetic similarity and augmented definitions, it refines noisy retrieved NEs using in-context learning, requiring no additional training. Experimental results on CommonVoice and STOP datasets show significant improvements in Word Error Rate (WER) and NE hit ratio, outperforming baseline ASR and RAGEC methods. Specifically, we achieved a 28% relative reduction in WER compared to ASR without postprocessing. Our source code is publicly available at: https://github.com/solee0022/deragec
>
---
#### [new 129] Swiss Parliaments Corpus Re-Imagined (SPC_R): Enhanced Transcription with RAG-based Correction and Predicted BLEU
- **分类: cs.CL**

- **简介: 该论文属于语音转文本任务，旨在提升瑞士议会语料库质量。通过ASR、LLM纠错和数据过滤，改进低资源领域语音数据的准确性。**

- **链接: [http://arxiv.org/pdf/2506.07726v1](http://arxiv.org/pdf/2506.07726v1)**

> **作者:** Vincenzo Timmel; Manfred Vogel; Daniel Perruchoud; Reza Kakooee
>
> **摘要:** This paper presents a new long-form release of the Swiss Parliaments Corpus, converting entire multi-hour Swiss German debate sessions (each aligned with the official session protocols) into high-quality speech-text pairs. Our pipeline starts by transcribing all session audio into Standard German using Whisper Large-v3 under high-compute settings. We then apply a two-step GPT-4o correction process: first, GPT-4o ingests the raw Whisper output alongside the official protocols to refine misrecognitions, mainly named entities. Second, a separate GPT-4o pass evaluates each refined segment for semantic completeness. We filter out any segments whose Predicted BLEU score (derived from Whisper's average token log-probability) and GPT-4o evaluation score fall below a certain threshold. The final corpus contains 801 hours of audio, of which 751 hours pass our quality control. Compared to the original sentence-level SPC release, our long-form dataset achieves a 6-point BLEU improvement, demonstrating the power of combining robust ASR, LLM-based correction, and data-driven filtering for low-resource, domain-specific speech corpora.
>
---
#### [new 130] Statistical Hypothesis Testing for Auditing Robustness in Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型鲁棒性审计任务，解决如何检测模型输出在干预下的变化问题。通过构建假设检验框架，实现有效、可解释的分析。**

- **链接: [http://arxiv.org/pdf/2506.07947v1](http://arxiv.org/pdf/2506.07947v1)**

> **作者:** Paulius Rauba; Qiyao Wei; Mihaela van der Schaar
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2412.00868
>
> **摘要:** Consider the problem of testing whether the outputs of a large language model (LLM) system change under an arbitrary intervention, such as an input perturbation or changing the model variant. We cannot simply compare two LLM outputs since they might differ due to the stochastic nature of the system, nor can we compare the entire output distribution due to computational intractability. While existing methods for analyzing text-based outputs exist, they focus on fundamentally different problems, such as measuring bias or fairness. To this end, we introduce distribution-based perturbation analysis, a framework that reformulates LLM perturbation analysis as a frequentist hypothesis testing problem. We construct empirical null and alternative output distributions within a low-dimensional semantic similarity space via Monte Carlo sampling, enabling tractable inference without restrictive distributional assumptions. The framework is (i) model-agnostic, (ii) supports the evaluation of arbitrary input perturbations on any black-box LLM, (iii) yields interpretable p-values; (iv) supports multiple perturbations via controlled error rates; and (v) provides scalar effect sizes. We demonstrate the usefulness of the framework across multiple case studies, showing how we can quantify response changes, measure true/false positive rates, and evaluate alignment with reference models. Above all, we see this as a reliable frequentist hypothesis testing framework for LLM auditing.
>
---
#### [new 131] Chain of Methodologies: Scaling Test Time Computation without Training
- **分类: cs.CL**

- **简介: 该论文属于复杂推理任务，旨在解决LLM在深度推理上的不足。提出CoM框架，通过整合人类方法论提升模型推理能力，无需训练即可增强逻辑思考。**

- **链接: [http://arxiv.org/pdf/2506.06982v1](http://arxiv.org/pdf/2506.06982v1)**

> **作者:** Cong Liu; Jie Wu; Weigang Wu; Xu Chen; Liang Lin; Wei-Shi Zheng
>
> **摘要:** Large Language Models (LLMs) often struggle with complex reasoning tasks due to insufficient in-depth insights in their training data, which are typically absent in publicly available documents. This paper introduces the Chain of Methodologies (CoM), an innovative and intuitive prompting framework that enhances structured thinking by integrating human methodological insights, enabling LLMs to tackle complex tasks with extended reasoning. CoM leverages the metacognitive abilities of advanced LLMs, activating systematic reasoning throught user-defined methodologies without explicit fine-tuning. Experiments show that CoM surpasses competitive baselines, demonstrating the potential of training-free prompting methods as robust solutions for complex reasoning tasks and bridging the gap toward human-level reasoning through human-like methodological insights.
>
---
#### [new 132] PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels
- **分类: cs.CL; cs.AI; cs.IR; cs.SI; I.2.7**

- **简介: 该论文属于政治立场检测任务，旨在解决用户层面立场数据稀缺的问题。通过收集Bluesky平台数据，构建了首个2024年美国大选用户立场数据集PolitiSky24。**

- **链接: [http://arxiv.org/pdf/2506.07606v1](http://arxiv.org/pdf/2506.07606v1)**

> **作者:** Peyman Rostami; Vahid Rahimzadeh; Ali Adibi; Azadeh Shakery
>
> **备注:** The dataset is available at https://doi.org/10.5281/zenodo.15616911
>
> **摘要:** Stance detection identifies the viewpoint expressed in text toward a specific target, such as a political figure. While previous datasets have focused primarily on tweet-level stances from established platforms, user-level stance resources, especially on emerging platforms like Bluesky remain scarce. User-level stance detection provides a more holistic view by considering a user's complete posting history rather than isolated posts. We present the first stance detection dataset for the 2024 U.S. presidential election, collected from Bluesky and centered on Kamala Harris and Donald Trump. The dataset comprises 16,044 user-target stance pairs enriched with engagement metadata, interaction graphs, and user posting histories. PolitiSky24 was created using a carefully evaluated pipeline combining advanced information retrieval and large language models, which generates stance labels with supporting rationales and text spans for transparency. The labeling approach achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in political stance analysis through its timeliness, open-data nature, and user-level perspective. The dataset is available at https://doi.org/10.5281/zenodo.15616911
>
---
#### [new 133] Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation
- **分类: cs.CL**

- **简介: 该论文属于低资源方言翻译任务，旨在解决乌克兰方言（如赫楚尔语）与标准乌克兰语之间的翻译问题。研究构建了平行语料库并提出RAG方法生成合成数据，微调LLMs以提升翻译效果。**

- **链接: [http://arxiv.org/pdf/2506.07617v1](http://arxiv.org/pdf/2506.07617v1)**

> **作者:** Roman Kyslyi; Yuliia Maksymiuk; Ihor Pysmennyi
>
> **备注:** Preprint. Will be published at Proceedings of the Fourth Ukrainian Natural Language Processing Workshop (UNLP)
>
> **摘要:** In this paper we introduce the first effort to adapt large language models (LLMs) to the Ukrainian dialect (in our case Hutsul), a low-resource and morphologically complex dialect spoken in the Carpathian Highlands. We created a parallel corpus of 9852 dialect-to-standard Ukrainian sentence pairs and a dictionary of 7320 dialectal word mappings. We also addressed data shortage by proposing an advanced Retrieval-Augmented Generation (RAG) pipeline to generate synthetic parallel translation pairs, expanding the corpus with 52142 examples. We have fine-tuned multiple open-source LLMs using LoRA and evaluated them on a standard-to-dialect translation task, also comparing with few-shot GPT-4o translation. In the absence of human annotators, we adopt a multi-metric evaluation strategy combining BLEU, chrF++, TER, and LLM-based judgment (GPT-4o). The results show that even small(7B) finetuned models outperform zero-shot baselines such as GPT-4o across both automatic and LLM-evaluated metrics. All data, models, and code are publicly released at: https://github.com/woters/vuyko-hutsul
>
---
#### [new 134] Reducing Object Hallucination in Large Audio-Language Models via Audio-Aware Decoding
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于音频-语言问答任务，旨在解决LALMs在音频内容上的对象幻觉问题。通过引入Audio-Aware Decoding（AAD）方法，提升模型准确性。**

- **链接: [http://arxiv.org/pdf/2506.07233v1](http://arxiv.org/pdf/2506.07233v1)**

> **作者:** Tzu-wen Hsu; Ke-Han Lu; Cheng-Han Chiang; Hung-yi Lee
>
> **摘要:** Large Audio-Language Models (LALMs) can take audio and text as the inputs and answer questions about the audio. While prior LALMs have shown strong performance on standard benchmarks, there has been alarming evidence that LALMs can hallucinate what is presented in the audio. To mitigate the hallucination of LALMs, we introduce Audio-Aware Decoding (AAD), a lightweight inference-time strategy that uses contrastive decoding to compare the token prediction logits with and without the audio context. By contrastive decoding, AAD promotes the tokens whose probability increases when the audio is present. We conduct our experiment on object hallucination datasets with three LALMs and show that AAD improves the F1 score by 0.046 to 0.428. We also show that AAD can improve the accuracy on general audio QA datasets like Clotho-AQA by 5.4% to 10.3%. We conduct thorough ablation studies to understand the effectiveness of each component in AAD.
>
---
#### [new 135] Cross-Entropy Games for Language Models: From Implicit Knowledge to General Capability Measures
- **分类: cs.AI; cs.CL; cs.GT; cs.IT; cs.NE; math.IT**

- **简介: 该论文提出Xent Games，用于评估大语言模型的综合能力，解决如何衡量模型通用能力的问题。通过构建基于交叉熵的游戏框架，设计能力基准测试。**

- **链接: [http://arxiv.org/pdf/2506.06832v1](http://arxiv.org/pdf/2506.06832v1)**

> **作者:** Clément Hongler; Andrew Emil
>
> **备注:** 41 pages, 16 figures
>
> **摘要:** Large Language Models (LLMs) define probability measures on text. By considering the implicit knowledge question of what it means for an LLM to know such a measure and what it entails algorithmically, we are naturally led to formulate a series of tasks that go beyond generative sampling, involving forms of summarization, counterfactual thinking, anomaly detection, originality search, reverse prompting, debating, creative solving, etc. These tasks can be formulated as games based on LLM measures, which we call Cross-Entropy (Xent) Games. Xent Games can be single-player or multi-player. They involve cross-entropy scores and cross-entropy constraints, and can be expressed as simple computational graphs and programs. We show the Xent Game space is large enough to contain a wealth of interesting examples, while being constructible from basic game-theoretic consistency axioms. We then discuss how the Xent Game space can be used to measure the abilities of LLMs. This leads to the construction of Xent Game measures: finite families of Xent Games that can be used as capability benchmarks, built from a given scope, by extracting a covering measure. To address the unbounded scope problem associated with the challenge of measuring general abilities, we propose to explore the space of Xent Games in a coherent fashion, using ideas inspired by evolutionary dynamics.
>
---
#### [new 136] Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有方法缺乏可解释性的问题。通过构建标注数据集并微调MLLM，提升检测准确性和解释能力。**

- **链接: [http://arxiv.org/pdf/2506.07045v1](http://arxiv.org/pdf/2506.07045v1)**

> **作者:** Yikun Ji; Hong Yan; Jun Lan; Huijia Zhu; Weiqiang Wang; Qi Fan; Liqing Zhang; Jianfu Zhang
>
> **摘要:** The rapid advancement of image generation technologies intensifies the demand for interpretable and robust detection methods. Although existing approaches often attain high accuracy, they typically operate as black boxes without providing human-understandable justifications. Multi-modal Large Language Models (MLLMs), while not originally intended for forgery detection, exhibit strong analytical and reasoning capabilities. When properly fine-tuned, they can effectively identify AI-generated images and offer meaningful explanations. However, existing MLLMs still struggle with hallucination and often fail to align their visual interpretations with actual image content and human reasoning. To bridge this gap, we construct a dataset of AI-generated images annotated with bounding boxes and descriptive captions that highlight synthesis artifacts, establishing a foundation for human-aligned visual-textual grounded reasoning. We then finetune MLLMs through a multi-stage optimization strategy that progressively balances the objectives of accurate detection, visual localization, and coherent textual explanation. The resulting model achieves superior performance in both detecting AI-generated images and localizing visual flaws, significantly outperforming baseline methods.
>
---
#### [new 137] Reinforcing Multimodal Understanding and Generation with Dual Self-rewards
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决图像-文本对齐问题。通过自监督双奖励机制提升模型的理解与生成能力，无需外部监督。**

- **链接: [http://arxiv.org/pdf/2506.07963v1](http://arxiv.org/pdf/2506.07963v1)**

> **作者:** Jixiang Hong; Yiran Zhang; Guanzhong Wang; Yi Liu; Ji-Rong Wen; Rui Yan
>
> **摘要:** Building upon large language models (LLMs), recent large multimodal models (LMMs) unify cross-model understanding and generation into a single framework. However, LMMs still struggle to achieve accurate image-text alignment, prone to generating text responses contradicting the visual input or failing to follow the text-to-image prompts. Current solutions require external supervision (e.g., human feedback or reward models) and only address unidirectional tasks-either understanding or generation. In this work, based on the observation that understanding and generation are inverse dual tasks, we introduce a self-supervised dual reward mechanism to reinforce the understanding and generation capabilities of LMMs. Specifically, we sample multiple outputs for a given input in one task domain, then reverse the input-output pairs to compute the dual likelihood of the model as self-rewards for optimization. Extensive experimental results on visual understanding and generation benchmarks demonstrate that our method can effectively enhance the performance of the model without any external supervision, especially achieving remarkable improvements in text-to-image tasks.
>
---
#### [new 138] ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning
- **分类: cs.LG; cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于化学与材料科学领域，旨在解决LLMs在化学任务中的知识滞后和专业工具整合难题。通过集成137个化学工具和构建数据集，提出HE-MCTS框架提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07551v1](http://arxiv.org/pdf/2506.07551v1)**

> **作者:** Mengsong Wu; YaFei Wang; Yidong Ming; Yuqi An; Yuwei Wan; Wenliang Chen; Binbin Lin; Yuqiang Li; Tong Xie; Dongzhan Zhou
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Large language models (LLMs) have recently demonstrated promising capabilities in chemistry tasks while still facing challenges due to outdated pretraining knowledge and the difficulty of incorporating specialized chemical expertise. To address these issues, we propose an LLM-based agent that synergistically integrates 137 external chemical tools created ranging from basic information retrieval to complex reaction predictions, and a dataset curation pipeline to generate the dataset ChemToolBench that facilitates both effective tool selection and precise parameter filling during fine-tuning and evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search (HE-MCTS) framework, enabling independent optimization of tool planning and execution. By leveraging self-generated data, our approach supports step-level fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM that surpass GPT-4o. Experimental evaluations demonstrate that our approach significantly improves performance in Chemistry QA and discovery tasks, offering a robust solution to integrate specialized tools with LLMs for advanced chemical applications. All datasets and code are available at https://github.com/AI4Chem/ChemistryAgent .
>
---
#### [new 139] E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于主题模型任务，解决LDA推断问题，提出E-LDA算法，在对数并行时间内高效且可解释地学习主题模型。**

- **链接: [http://arxiv.org/pdf/2506.07747v1](http://arxiv.org/pdf/2506.07747v1)**

> **作者:** Adam Breuer
>
> **备注:** ICML 2025; Code available at: https://github.com/BreuerLabs/E- LDA
>
> **摘要:** In this paper, we provide the first practical algorithms with provable guarantees for the problem of inferring the topics assigned to each document in an LDA topic model. This is the primary inference problem for many applications of topic models in social science, data exploration, and causal inference settings. We obtain this result by showing a novel non-gradient-based, combinatorial approach to estimating topic models. This yields algorithms that converge to near-optimal posterior probability in logarithmic parallel computation time (adaptivity) -- exponentially faster than any known LDA algorithm. We also show that our approach can provide interpretability guarantees such that each learned topic is formally associated with a known keyword. Finally, we show that unlike alternatives, our approach can maintain the independence assumptions necessary to use the learned topic model for downstream causal inference methods that allow researchers to study topics as treatments. In terms of practical performance, our approach consistently returns solutions of higher semantic quality than solutions from state-of-the-art LDA algorithms, neural topic models, and LLM-based topic models across a diverse range of text datasets and evaluation parameters.
>
---
#### [new 140] Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究视觉语言模型的多模态上下文学习任务，旨在解决模型依赖浅层策略而非真正理解任务的问题。通过引入带推理的演示增强方法进行实验验证。**

- **链接: [http://arxiv.org/pdf/2506.07936v1](http://arxiv.org/pdf/2506.07936v1)**

> **作者:** Chengyue Huang; Yuchen Zhu; Sichen Zhu; Jingyun Xiao; Moises Andrade; Shivang Chopra; Zsolt Kira
>
> **摘要:** Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics -- such as copying or majority voting -- rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with Reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, suggesting that current VLMs do not effectively utilize demonstration-level information as intended in MM-ICL.
>
---
#### [new 141] DISRetrieval: Harnessing Discourse Structure for Long Document Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于长文档检索任务，旨在解决传统方法忽视话语结构的问题。提出DISRetrieval框架，利用话语结构提升检索效果。**

- **链接: [http://arxiv.org/pdf/2506.06313v1](http://arxiv.org/pdf/2506.06313v1)**

> **作者:** Huiyao Chen; Yi Yang; Yinghui Li; Meishan Zhang; Min Zhang
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Long document understanding has become increasingly crucial in natural language processing, with retrieval-based methods emerging as a promising solution to address the context length limitations of large language models (LLMs). However, existing approaches either treat documents as flat sequences or employ arbitrary chunking strategies, failing to capture the inherent discourse structure that guides human comprehension. We present DISRetrieval, a novel hierarchical retrieval framework that leverages linguistic discourse structure to enhance long document understanding. Our approach introduces three key innovations: (1) a discourse-aware document organization framework that utilizes rhetorical structure theory (RST) to create sentence-level hierarchical representations, preserving both semantic relationships and natural document flow; (2) an LLM-enhanced node representation technique that combines discourse structure with adaptive summarization to enrich tree nodes with contextual information; and (3) a hierarchical evidence retrieval mechanism that effectively selects relevant content while maintaining discourse coherence. Through comprehensive experiments on QASPER and QuALITY datasets, DISRetrieval demonstrates substantial improvements over existing methods in both token-level retrieval metrics and downstream question answering tasks. Our ablation studies confirm that incorporating discourse structure significantly enhances retrieval effectiveness across different document lengths and query types, validating the importance of linguistically-informed document representation in long-text understanding. Our code and datasets are publicly available at github/DreamH1gh/DISRetrieval to facilitate future research.
>
---
#### [new 142] LLMs as World Models: Data-Driven and Human-Centered Pre-Event Simulation for Disaster Impact Assessment
- **分类: cs.CY; cs.CE; cs.CL; cs.CV**

- **简介: 该论文属于灾害影响评估任务，旨在通过LLMs模拟地震影响。利用多模态数据生成 MMI 预测，提升灾前规划能力。**

- **链接: [http://arxiv.org/pdf/2506.06355v1](http://arxiv.org/pdf/2506.06355v1)**

> **作者:** Lingyao Li; Dawei Li; Zhenhui Ou; Xiaoran Xu; Jingxiao Liu; Zihui Ma; Runlong Yu; Min Deng
>
> **摘要:** Efficient simulation is essential for enhancing proactive preparedness for sudden-onset disasters such as earthquakes. Recent advancements in large language models (LLMs) as world models show promise in simulating complex scenarios. This study examines multiple LLMs to proactively estimate perceived earthquake impacts. Leveraging multimodal datasets including geospatial, socioeconomic, building, and street-level imagery data, our framework generates Modified Mercalli Intensity (MMI) predictions at zip code and county scales. Evaluations on the 2014 Napa and 2019 Ridgecrest earthquakes using USGS ''Did You Feel It? (DYFI)'' reports demonstrate significant alignment, as evidenced by a high correlation of 0.88 and a low RMSE of 0.77 as compared to real reports at the zip code level. Techniques such as RAG and ICL can improve simulation performance, while visual inputs notably enhance accuracy compared to structured numerical data alone. These findings show the promise of LLMs in simulating disaster impacts that can help strengthen pre-event planning.
>
---
#### [new 143] SAP-Bench: Benchmarking Multimodal Large Language Models in Surgical Action Planning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手术动作规划任务，旨在解决多模态大语言模型在手术场景中预测下一步动作的能力不足问题。研究构建了SAP-Bench数据集并提出MLLM-SAP框架进行评估。**

- **链接: [http://arxiv.org/pdf/2506.07196v1](http://arxiv.org/pdf/2506.07196v1)**

> **作者:** Mengya Xu; Zhongzhen Huang; Dillan Imans; Yiru Ye; Xiaofan Zhang; Qi Dou
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Effective evaluation is critical for driving advancements in MLLM research. The surgical action planning (SAP) task, which aims to generate future action sequences from visual inputs, demands precise and sophisticated analytical capabilities. Unlike mathematical reasoning, surgical decision-making operates in life-critical domains and requires meticulous, verifiable processes to ensure reliability and patient safety. This task demands the ability to distinguish between atomic visual actions and coordinate complex, long-horizon procedures, capabilities that are inadequately evaluated by current benchmarks. To address this gap, we introduce SAP-Bench, a large-scale, high-quality dataset designed to enable multimodal large language models (MLLMs) to perform interpretable surgical action planning. Our SAP-Bench benchmark, derived from the cholecystectomy procedures context with the mean duration of 1137.5s, and introduces temporally-grounded surgical action annotations, comprising the 1,226 clinically validated action clips (mean duration: 68.7s) capturing five fundamental surgical actions across 74 procedures. The dataset provides 1,152 strategically sampled current frames, each paired with the corresponding next action as multimodal analysis anchors. We propose the MLLM-SAP framework that leverages MLLMs to generate next action recommendations from the current surgical scene and natural language instructions, enhanced with injected surgical domain knowledge. To assess our dataset's effectiveness and the broader capabilities of current models, we evaluate seven state-of-the-art MLLMs (e.g., OpenAI-o1, GPT-4o, QwenVL2.5-72B, Claude-3.5-Sonnet, GeminiPro2.5, Step-1o, and GLM-4v) and reveal critical gaps in next action prediction performance.
>
---
#### [new 144] On the Fundamental Impossibility of Hallucination Control in Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.GT; cs.LG**

- **简介: 该论文属于自然语言处理领域，探讨大语言模型无法避免幻觉的问题。工作是提出一个不可能定理，证明无法同时满足真实性、信息保留等四个属性。**

- **链接: [http://arxiv.org/pdf/2506.06382v1](http://arxiv.org/pdf/2506.06382v1)**

> **作者:** Michał P. Karpowicz
>
> **摘要:** This paper explains \textbf{why it is impossible to create large language models that do not hallucinate and what are the trade-offs we should be looking for}. It presents a formal \textbf{impossibility theorem} demonstrating that no inference mechanism can simultaneously satisfy four fundamental properties: \textbf{truthful (non-hallucinatory) generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality}. By modeling LLM inference as an \textbf{auction of ideas} where neural components compete to contribute to responses, we prove the impossibility using the Green-Laffont theorem. That mathematical framework provides a rigorous foundation for understanding the nature of inference process, with implications for model architecture, training objectives, and evaluation methods.
>
---
#### [new 145] Solving Inequality Proofs with Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数学不等式证明任务，旨在提升大语言模型的推理能力。通过构建数据集和评估框架，发现现有模型在严格证明上表现不佳，提出theorem-guided等改进方向。**

- **链接: [http://arxiv.org/pdf/2506.07927v1](http://arxiv.org/pdf/2506.07927v1)**

> **作者:** Jiayi Sheng; Luna Lyu; Jikai Jin; Tony Xia; Alex Gu; James Zou; Pan Lu
>
> **备注:** 52 pages, 16 figures
>
> **摘要:** Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at https://ineqmath.github.io/.
>
---
#### [new 146] Learning Speaker-Invariant Visual Features for Lipreading
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于唇读任务，旨在解决因说话人特异性特征导致的模型泛化问题。通过引入SIFLip框架，分离说话人特征，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.07572v1](http://arxiv.org/pdf/2506.07572v1)**

> **作者:** Yu Li; Feng Xue; Shujie Li; Jinrui Zhang; Shuang Yang; Dan Guo; Richang Hong
>
> **摘要:** Lipreading is a challenging cross-modal task that aims to convert visual lip movements into spoken text. Existing lipreading methods often extract visual features that include speaker-specific lip attributes (e.g., shape, color, texture), which introduce spurious correlations between vision and text. These correlations lead to suboptimal lipreading accuracy and restrict model generalization. To address this challenge, we introduce SIFLip, a speaker-invariant visual feature learning framework that disentangles speaker-specific attributes using two complementary disentanglement modules (Implicit Disentanglement and Explicit Disentanglement) to improve generalization. Specifically, since different speakers exhibit semantic consistency between lip movements and phonetic text when pronouncing the same words, our implicit disentanglement module leverages stable text embeddings as supervisory signals to learn common visual representations across speakers, implicitly decoupling speaker-specific features. Additionally, we design a speaker recognition sub-task within the main lipreading pipeline to filter speaker-specific features, then further explicitly disentangle these personalized visual features from the backbone network via gradient reversal. Experimental results demonstrate that SIFLip significantly enhances generalization performance across multiple public datasets. Experimental results demonstrate that SIFLip significantly improves generalization performance across multiple public datasets, outperforming state-of-the-art methods.
>
---
#### [new 147] G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
- **分类: cs.MA; cs.CL; cs.LG**

- **简介: 该论文属于多智能体系统任务，旨在解决MAS记忆架构不足的问题。提出G-Memory，通过分层图结构提升跨试验和个体的协作记忆能力。**

- **链接: [http://arxiv.org/pdf/2506.07398v1](http://arxiv.org/pdf/2506.07398v1)**

> **作者:** Guibin Zhang; Muxin Fu; Guancheng Wan; Miao Yu; Kun Wang; Shuicheng Yan
>
> **摘要:** Large language model (LLM)-powered multi-agent systems (MAS) have demonstrated cognitive and execution capabilities that far exceed those of single LLM agents, yet their capacity for self-evolution remains hampered by underdeveloped memory architectures. Upon close inspection, we are alarmed to discover that prevailing MAS memory mechanisms (1) are overly simplistic, completely disregarding the nuanced inter-agent collaboration trajectories, and (2) lack cross-trial and agent-specific customization, in stark contrast to the expressive memory developed for single agents. To bridge this gap, we introduce G-Memory, a hierarchical, agentic memory system for MAS inspired by organizational memory theory, which manages the lengthy MAS interaction via a three-tier graph hierarchy: insight, query, and interaction graphs. Upon receiving a new user query, G-Memory performs bi-directional memory traversal to retrieve both $\textit{high-level, generalizable insights}$ that enable the system to leverage cross-trial knowledge, and $\textit{fine-grained, condensed interaction trajectories}$ that compactly encode prior collaboration experiences. Upon task execution, the entire hierarchy evolves by assimilating new collaborative trajectories, nurturing the progressive evolution of agent teams. Extensive experiments across five benchmarks, three LLM backbones, and three popular MAS frameworks demonstrate that G-Memory improves success rates in embodied action and accuracy in knowledge QA by up to $20.89\%$ and $10.12\%$, respectively, without any modifications to the original frameworks. Our codes are available at https://github.com/bingreeky/GMemory.
>
---
#### [new 148] Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于AI与就业关系研究，旨在解决AI代理在工作中自动化与增强的匹配问题。通过构建审计框架和数据库，分析职业任务与技术能力的差异，提出人类代理尺度（HAS）以量化人机协作需求。**

- **链接: [http://arxiv.org/pdf/2506.06576v1](http://arxiv.org/pdf/2506.06576v1)**

> **作者:** Yijia Shao; Humishka Zope; Yucheng Jiang; Jiaxin Pei; David Nguyen; Erik Brynjolfsson; Diyi Yang
>
> **备注:** Preprint
>
> **摘要:** The rapid rise of compound AI systems (a.k.a., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. Yet, we lack a systematic understanding of the evolving landscape. In this paper, we address this gap by introducing a novel auditing framework to assess which occupational tasks workers want AI agents to automate or augment, and how those desires align with the current technological capabilities. Our framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, we construct the WORKBank database, building on the U.S. Department of Labor's O*NET database, to capture preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations. Jointly considering the desire and technological capability divides tasks in WORKBank into four zones: Automation "Green Light" Zone, Automation "Red Light" Zone, R&D Opportunity Zone, Low Priority Zone. This highlights critical mismatches and opportunities for AI agent development. Moving beyond a simple automate-or-not dichotomy, our results reveal diverse HAS profiles across occupations, reflecting heterogeneous expectations for human involvement. Moreover, our study offers early signals of how AI agent integration may reshape the core human competencies, shifting from information-focused skills to interpersonal ones. These findings underscore the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics.
>
---
#### [new 149] GLProtein: Global-and-Local Structure Aware Protein Representation Learning
- **分类: cs.LG; cs.AI; cs.CL; q-bio.BM**

- **简介: 该论文属于蛋白质表示学习任务，旨在融合全局结构相似性和局部氨基酸信息，提升蛋白质功能预测性能。**

- **链接: [http://arxiv.org/pdf/2506.06294v1](http://arxiv.org/pdf/2506.06294v1)**

> **作者:** Yunqing Liu; Wenqi Fan; Xiaoyong Wei; Qing Li
>
> **摘要:** Proteins are central to biological systems, participating as building blocks across all forms of life. Despite advancements in understanding protein functions through protein sequence analysis, there remains potential for further exploration in integrating protein structural information. We argue that the structural information of proteins is not only limited to their 3D information but also encompasses information from amino acid molecules (local information) to protein-protein structure similarity (global information). To address this, we propose \textbf{GLProtein}, the first framework in protein pre-training that incorporates both global structural similarity and local amino acid details to enhance prediction accuracy and functional insights. GLProtein innovatively combines protein-masked modelling with triplet structure similarity scoring, protein 3D distance encoding and substructure-based amino acid molecule encoding. Experimental results demonstrate that GLProtein outperforms previous methods in several bioinformatics tasks, including predicting protein-protein interaction, contact prediction, and so on.
>
---
#### [new 150] Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在优化阿拉伯语RAG管道。研究分析了不同组件对性能的影响，提出有效方法提升生成质量与准确性。**

- **链接: [http://arxiv.org/pdf/2506.06339v1](http://arxiv.org/pdf/2506.06339v1)**

> **作者:** Jumana Alsubhi; Mohammad D. Alahmadi; Ahmed Alhusayni; Ibrahim Aldailami; Israa Hamdine; Ahmad Shabana; Yazeed Iskandar; Suhayb Khayyat
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for combining the precision of retrieval systems with the fluency of large language models. While several studies have investigated RAG pipelines for high-resource languages, the optimization of RAG components for Arabic remains underexplored. This study presents a comprehensive empirical evaluation of state-of-the-art RAG components-including chunking strategies, embedding models, rerankers, and language models-across a diverse set of Arabic datasets. Using the RAGAS framework, we systematically compare performance across four core metrics: context precision, context recall, answer faithfulness, and answer relevancy. Our experiments demonstrate that sentence-aware chunking outperforms all other segmentation methods, while BGE-M3 and Multilingual-E5-large emerge as the most effective embedding models. The inclusion of a reranker (bge-reranker-v2-m3) significantly boosts faithfulness in complex datasets, and Aya-8B surpasses StableLM in generation quality. These findings provide critical insights for building high-quality Arabic RAG pipelines and offer practical guidelines for selecting optimal components across different document types.
>
---
#### [new 151] Mitigating Behavioral Hallucination in Multimodal Large Language Models for Sequential Images
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型任务，旨在解决序列图像中的行为幻觉问题。通过提出SHE框架和BEACH指标，有效减少幻觉并保持描述准确性。**

- **链接: [http://arxiv.org/pdf/2506.07184v1](http://arxiv.org/pdf/2506.07184v1)**

> **作者:** Liangliang You; Junchi Yao; Shu Yang; Guimin Hu; Lijie Hu; Di Wang
>
> **摘要:** While multimodal large language models excel at various tasks, they still suffer from hallucinations, which limit their reliability and scalability for broader domain applications. To address this issue, recent research mainly focuses on objective hallucination. However, for sequential images, besides objective hallucination, there is also behavioral hallucination, which is less studied. This work aims to fill in the gap. We first reveal that behavioral hallucinations mainly arise from two key factors: prior-driven bias and the snowball effect. Based on these observations, we introduce SHE (Sequence Hallucination Eradication), a lightweight, two-stage framework that (1) detects hallucinations via visual-textual alignment check using our proposed adaptive temporal window and (2) mitigates them via orthogonal projection onto the joint embedding space. We also propose a new metric (BEACH) to quantify behavioral hallucination severity. Empirical results on standard benchmarks demonstrate that SHE reduces behavioral hallucination by over 10% on BEACH while maintaining descriptive accuracy.
>
---
#### [new 152] Reward Is Enough: LLMs Are In-Context Reinforcement Learners
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，探讨LLM在推理时自发形成RL机制。通过ICRL提示框架，使LLM根据奖励优化响应，提升任务表现。**

- **链接: [http://arxiv.org/pdf/2506.06303v1](http://arxiv.org/pdf/2506.06303v1)**

> **作者:** Kefan Song; Amir Moeini; Peng Wang; Lei Gong; Rohan Chandra; Yanjun Qi; Shangtong Zhang
>
> **摘要:** Reinforcement learning (RL) is a human-designed framework for solving sequential decision making problems. In this work, we demonstrate that, surprisingly, RL emerges in LLM's (Large Language Model) inference time -- a phenomenon known as in-context RL (ICRL). Specifically, we propose a novel multi-round prompting framework called ICRL prompting. The goal is to prompt the LLM to complete a task. After the LLM generates a response at the current round, we give numerical scalar feedbacks for the response, called the rewards. At the next round, we prompt the LLM again with the same task and a context consisting of all previous responses and rewards. We observe that the quality of the LLM's response increases as the context grows. In other words, the LLM is able to maximize the scalar reward signal in the inference time, just like an RL algorithm. We evaluate ICRL prompting in three benchmarks (Game of 24, creative writing, and ScienceWorld) and demonstrate significant performance improvements over baseline methods such as Self-Refine and Reflexion. Surprisingly, in some experiments the reward signals are generated by the LLM itself, yet performance improvements are still observed from ICRL prompting, offering a promising paradigm for scaling test-time compute.
>
---
#### [new 153] HeavyWater and SimplexWater: Watermarking Low-Entropy Text Distributions
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; cs.IT; cs.LG; math.IT**

- **简介: 该论文属于文本水印任务，解决低熵生成场景下的水印设计问题。提出两种可调水印方法，提升检测准确率并减少文本失真。**

- **链接: [http://arxiv.org/pdf/2506.06409v1](http://arxiv.org/pdf/2506.06409v1)**

> **作者:** Dor Tsur; Carol Xuan Long; Claudio Mayrink Verdun; Hsiang Hsu; Chen-Fu Chen; Haim Permuter; Sajani Vithana; Flavio P. Calmon
>
> **摘要:** Large language model (LLM) watermarks enable authentication of text provenance, curb misuse of machine-generated text, and promote trust in AI systems. Current watermarks operate by changing the next-token predictions output by an LLM. The updated (i.e., watermarked) predictions depend on random side information produced, for example, by hashing previously generated tokens. LLM watermarking is particularly challenging in low-entropy generation tasks - such as coding - where next-token predictions are near-deterministic. In this paper, we propose an optimization framework for watermark design. Our goal is to understand how to most effectively use random side information in order to maximize the likelihood of watermark detection and minimize the distortion of generated text. Our analysis informs the design of two new watermarks: HeavyWater and SimplexWater. Both watermarks are tunable, gracefully trading-off between detection accuracy and text distortion. They can also be applied to any LLM and are agnostic to side information generation. We examine the performance of HeavyWater and SimplexWater through several benchmarks, demonstrating that they can achieve high watermark detection accuracy with minimal compromise of text generation quality, particularly in the low-entropy regime. Our theoretical analysis also reveals surprising new connections between LLM watermarking and coding theory. The code implementation can be found in https://github.com/DorTsur/HeavyWater_SimplexWater
>
---
#### [new 154] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于人工智能领域，研究LRMs在不同复杂度任务中的表现，分析其推理能力与局限性，揭示其在高复杂度任务中的性能下降问题。**

- **链接: [http://arxiv.org/pdf/2506.06941v1](http://arxiv.org/pdf/2506.06941v1)**

> **作者:** Parshin Shojaee; Iman Mirzadeh; Keivan Alizadeh; Maxwell Horton; Samy Bengio; Mehrdad Farajtabar
>
> **备注:** preprint
>
> **摘要:** Recent generations of language models have introduced Large Reasoning Models (LRMs) that generate detailed thinking processes before providing answers. While these models demonstrate improved performance on reasoning benchmarks, their fundamental capabilities, scaling properties, and limitations remain insufficiently understood. Current evaluations primarily focus on established math and coding benchmarks, emphasizing final answer accuracy. However, this evaluation paradigm often suffers from contamination and does not provide insights into the reasoning traces. In this work, we systematically investigate these gaps with the help of controllable puzzle environments that allow precise manipulation of complexity while maintaining consistent logical structures. This setup enables the analysis of not only final answers but also the internal reasoning traces, offering insights into how LRMs think. Through extensive experiments, we show that LRMs face a complete accuracy collapse beyond certain complexities. Moreover, they exhibit a counterintuitive scaling limit: their reasoning effort increases with problem complexity up to a point, then declines despite having remaining token budget. By comparing LRMs with their standard LLM counterparts under same inference compute, we identify three performance regimes: (1) low-complexity tasks where standard models outperform LRMs, (2) medium-complexity tasks where LRMs demonstrates advantage, and (3) high-complexity tasks where both models face complete collapse. We found that LRMs have limitations in exact computation: they fail to use explicit algorithms and reason inconsistently across scales. We also investigate the reasoning traces in more depth, studying the patterns of explored solutions and analyzing the models' computational behavior, shedding light on their strengths, limitations, and raising questions about their reasoning capabilities.
>
---
#### [new 155] Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型推理任务，旨在提升LLM的推理能力。通过课程强化学习从易到难训练，解决RL在复杂任务中效果不佳的问题。**

- **链接: [http://arxiv.org/pdf/2506.06632v1](http://arxiv.org/pdf/2506.06632v1)**

> **作者:** Shubham Parashar; Shurui Gui; Xiner Li; Hongyi Ling; Sushil Vemuri; Blake Olson; Eric Li; Yu Zhang; James Caverlee; Dileep Kalathil; Shuiwang Ji
>
> **摘要:** We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method.
>
---
#### [new 156] Is BERTopic Better than PLSA for Extracting Key Topics in Aviation Safety Reports?
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于主题建模任务，旨在比较BERTopic与PLSA在航空安全报告中提取关键主题的效果，以提升对航空事件模式的理解。**

- **链接: [http://arxiv.org/pdf/2506.06328v1](http://arxiv.org/pdf/2506.06328v1)**

> **作者:** Aziida Nanyonga; Joiner Keith; Turhan Ugur; Wild Graham
>
> **摘要:** This study compares the effectiveness of BERTopic and Probabilistic Latent Semantic Analysis (PLSA) in extracting meaningful topics from aviation safety reports aiming to enhance the understanding of patterns in aviation incident data. Using a dataset of over 36,000 National Transportation Safety Board (NTSB) reports from 2000 to 2020, BERTopic employed transformer based embeddings and hierarchical clustering, while PLSA utilized probabilistic modelling through the Expectation-Maximization (EM) algorithm. Results showed that BERTopic outperformed PLSA in topic coherence, achieving a Cv score of 0.41 compared to PLSA 0.37, while also demonstrating superior interpretability as validated by aviation safety experts. These findings underscore the advantages of modern transformer based approaches in analyzing complex aviation datasets, paving the way for enhanced insights and informed decision-making in aviation safety. Future work will explore hybrid models, multilingual datasets, and advanced clustering techniques to further improve topic modelling in this domain.
>
---
#### [new 157] From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决LLM与国际人道法对齐问题。通过测试模型拒绝非法请求的能力及解释质量，提出改进方法。**

- **链接: [http://arxiv.org/pdf/2506.06391v1](http://arxiv.org/pdf/2506.06391v1)**

> **作者:** John Mavi; Diana Teodora Găitan; Sergio Coronado
>
> **摘要:** Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL.
>
---
#### [new 158] Graph-of-Causal Evolution: Challenging Chain-of-Model for Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于因果推理任务，旨在解决链式模型中长程依赖丢失的问题。提出GoCE框架，通过因果掩码和自进化门控提升Transformer捕捉长期因果关系的能力。**

- **链接: [http://arxiv.org/pdf/2506.07501v1](http://arxiv.org/pdf/2506.07501v1)**

> **作者:** Libo Wang
>
> **备注:** The relevant code has been uploaded to the publicly available GitHub repository. The link is: https://github.com/brucewang123456789/GeniusTrail/tree/main/GoCE
>
> **摘要:** In view of the problem that each subchain in the chain-of-model (CoM) relies only on the information of the previous subchain and may lose long-range dependencies due to the causal mask blocking the global context flow between multi-level subchains, this work proposes a graph of causal evolution (GoCE). Its core principle is to map the implicit token representation into a differentiable and sparse causal adjacency matrix, then permeate causal constraints through each layer of calculation using causal-masked attention and causal-MoE. By combining intervention consistency loss test and self-evolution gate, the dynamic balance between causal structure learning and adaptive updating of transformer architecture is realized. The researcher built experimental environments in sandboxes built with Claude Sonnet 4, o4-mini-high, and DeepSeek R1 respectively with the transformer variant architecture introduced in GoCE. It is evaluated on publicly available datasets including CLUTRR, CLADDER, EX-FEVER, and CausalQA and compared with the baseline LLMs. The finding proves that GoCE strengthens the transformer's ability to capture long-range causal dependencies, while the ability to self-evolve is improved. It not only surpasses the design of CoM in terms of design principles, but also provides experience for future research on causal learning and continuous adaptive improvement.
>
---
#### [new 159] Learning Compact Vision Tokens for Efficient Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于多模态任务，旨在解决大模型计算效率低的问题。通过融合视觉令牌减少序列长度，同时保持信息完整，提升推理效率。**

- **链接: [http://arxiv.org/pdf/2506.07138v1](http://arxiv.org/pdf/2506.07138v1)**

> **作者:** Hao Tang; Chengchao Shen
>
> **备注:** The source code and trained weights are available at https://github.com/visresearch/LLaVA-STF
>
> **摘要:** Large multimodal models (LMMs) suffer significant computational challenges due to the high cost of Large Language Models (LLMs) and the quadratic complexity of processing long vision token sequences. In this paper, we explore the spatial redundancy among vision tokens and shorten the length of vision token sequences for inference acceleration. Specifically, we propose a Spatial Token Fusion (STF) method to learn compact vision tokens for short vision token sequence, where spatial-adjacent tokens are fused into one. Meanwhile, weight-frozen vision encoder can not well adapt to the demand of extensive downstream vision-language tasks. To this end, we further introduce a Multi-Block Token Fusion (MBTF) module to supplement multi-granularity features for the reduced token sequence. Overall, we combine STF and MBTF module to balance token reduction and information preservation, thereby improving inference efficiency without sacrificing multimodal reasoning capabilities. Experimental results demonstrate that our method based on LLaVA-1.5 achieves comparable or even superior performance to the baseline on 8 popular vision-language benchmarks with only $25\%$ vision tokens of baseline. The source code and trained weights are available at https://github.com/visresearch/LLaVA-STF.
>
---
#### [new 160] LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement
- **分类: cs.AI; cs.CL; cs.SY; eess.SY**

- **简介: 该论文提出LUCIFER框架，解决动态环境中自主决策与现实不符的问题。融合RL与LLM，提升探索效率与决策质量。**

- **链接: [http://arxiv.org/pdf/2506.07915v1](http://arxiv.org/pdf/2506.07915v1)**

> **作者:** Dimitris Panagopoulos; Adolfo Perrusquia; Weisi Guo
>
> **备注:** 12 pages, 4 Figures, 3 Tables, submitted to the IEEE for possible publication
>
> **摘要:** In dynamic environments, the rapid obsolescence of pre-existing environmental knowledge creates a gap between an agent's internal model and the evolving reality of its operational context. This disparity between prior and updated environmental valuations fundamentally limits the effectiveness of autonomous decision-making. To bridge this gap, the contextual bias of human domain stakeholders, who naturally accumulate insights through direct, real-time observation, becomes indispensable. However, translating their nuanced, and context-rich input into actionable intelligence for autonomous systems remains an open challenge. To address this, we propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates a hierarchical decision-making architecture with reinforcement learning (RL) and large language models (LLMs) into a unified system. This architecture mirrors how humans decompose complex tasks, enabling a high-level planner to coordinate specialised sub-agents, each focused on distinct objectives and temporally interdependent actions. Unlike traditional applications where LLMs are limited to single role, LUCIFER integrates them in two synergistic roles: as context extractors, structuring verbal stakeholder input into domain-aware representations that influence decision-making through an attention space mechanism aligning LLM-derived insights with the agent's learning process, and as zero-shot exploration facilitators guiding the agent's action selection process during exploration. We benchmark various LLMs in both roles and demonstrate that LUCIFER improves exploration efficiency and decision quality, outperforming flat, goal-conditioned policies. Our findings show the potential of context-driven decision-making, where autonomous systems leverage human contextual knowledge for operational success.
>
---
#### [new 161] $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出$\tau^2$-bench，用于评估对话代理在双控环境中的表现，解决传统基准不模拟用户主动参与的问题。**

- **链接: [http://arxiv.org/pdf/2506.07982v1](http://arxiv.org/pdf/2506.07982v1)**

> **作者:** Victor Barres; Honghua Dong; Soham Ray; Xujie Si; Karthik Narasimhan
>
> **摘要:** Existing benchmarks for conversational AI agents simulate single-control environments, where only the AI agent can use tools to interact with the world, while the user remains a passive information provider. This differs from real-world scenarios like technical support, where users need to actively participate in modifying the state of the (shared) world. In order to address this gap, we introduce $\tau^2$-bench, with four key contributions: 1) A novel Telecom dual-control domain modeled as a Dec-POMDP, where both agent and user make use of tools to act in a shared, dynamic environment that tests both agent coordination and communication, 2) A compositional task generator that programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage and controlled complexity, 3) A reliable user simulator tightly coupled with the environment, whose behavior is constrained by tools and observable states, improving simulation fidelity, 4) Fine-grained analysis of agent performance through multiple ablations including separating errors arising from reasoning vs communication/coordination. In particular, our experiments show significant performance drops when agents shift from no-user to dual-control, highlighting the challenges of guiding users. Overall, $\tau^2$-bench provides a controlled testbed for agents that must both reason effectively and guide user actions.
>
---
#### [new 162] Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于文本属性图表示学习任务，旨在解决传统方法依赖大量标注数据的问题。提出GAGA框架，通过选择性标注和图对齐提升效率。**

- **链接: [http://arxiv.org/pdf/2506.07168v1](http://arxiv.org/pdf/2506.07168v1)**

> **作者:** Huanyi Xie; Lijie Hu; Lu Yu; Tianhao Huang; Longfei Li; Meng Li; Jun Zhou; Huan Wang; Di Wang
>
> **备注:** 23 pages
>
> **摘要:** In the realm of Text-attributed Graphs (TAGs), traditional graph neural networks (GNNs) often fall short due to the complex textual information associated with each node. Recent methods have improved node representations by leveraging large language models (LLMs) to enhance node text features, but these approaches typically require extensive annotations or fine-tuning across all nodes, which is both time-consuming and costly. To overcome these challenges, we introduce GAGA, an efficient framework for TAG representation learning. GAGA reduces annotation time and cost by focusing on annotating only representative nodes and edges. It constructs an annotation graph that captures the topological relationships among these annotations. Furthermore, GAGA employs a two-level alignment module to effectively integrate the annotation graph with the TAG, aligning their underlying structures. Experiments show that GAGA achieves classification accuracies on par with or surpassing state-of-the-art methods while requiring only 1% of the data to be annotated, demonstrating its high efficiency.
>
---
#### [new 163] Multi-Step Visual Reasoning with Visual Tokens Scaling and Verification
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决静态视觉编码限制问题，通过动态视觉令牌扩展与验证机制提升模型的迭代推理能力。**

- **链接: [http://arxiv.org/pdf/2506.07235v1](http://arxiv.org/pdf/2506.07235v1)**

> **作者:** Tianyi Bai; Zengjie Hu; Fupeng Sun; Jiantao Qiu; Yizhen Jiang; Guangxin He; Bohan Zeng; Conghui He; Binhang Yuan; Wentao Zhang
>
> **摘要:** Multi-modal large language models (MLLMs) have achieved remarkable capabilities by integrating visual perception with language understanding, enabling applications such as image-grounded dialogue, visual question answering, and scientific analysis. However, most MLLMs adopt a static inference paradigm, encoding the entire image into fixed visual tokens upfront, which limits their ability to iteratively refine understanding or adapt to context during inference. This contrasts sharply with human perception, which is dynamic, selective, and feedback-driven. In this work, we introduce a novel framework for inference-time visual token scaling that enables MLLMs to perform iterative, verifier-guided reasoning over visual content. We formulate the problem as a Markov Decision Process, involving a reasoner that proposes visual actions and a verifier, which is trained via multi-step Direct Preference Optimization (DPO), that evaluates these actions and determines when reasoning should terminate. To support this, we present a new dataset, VTS, comprising supervised reasoning trajectories (VTS-SFT) and preference-labeled reasoning comparisons (VTS-DPO). Our method significantly outperforms existing approaches across diverse visual reasoning benchmarks, offering not only improved accuracy but also more interpretable and grounded reasoning processes. These results demonstrate the promise of dynamic inference mechanisms for enabling fine-grained, context-aware visual reasoning in next-generation MLLMs.
>
---
#### [new 164] Reparameterized LLM Training via Orthogonal Equivalence Transformation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于深度学习训练任务，旨在解决大语言模型训练不稳定和泛化能力差的问题。提出POET算法，通过正交等价变换优化神经元参数，提升训练稳定性和模型性能。**

- **链接: [http://arxiv.org/pdf/2506.08001v1](http://arxiv.org/pdf/2506.08001v1)**

> **作者:** Zeju Qiu; Simon Buchholz; Tim Z. Xiao; Maximilian Dax; Bernhard Schölkopf; Weiyang Liu
>
> **备注:** Technical report v1 (36 pages, 24 figures, project page: https://spherelab.ai/poet-site/)
>
> **摘要:** While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
>
---
#### [new 165] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决扩散型大语言模型（dLLMs）推理延迟高的问题。通过提出dLLM-Cache框架，实现高效缓存加速，提升推理速度。**

- **链接: [http://arxiv.org/pdf/2506.06295v1](http://arxiv.org/pdf/2506.06295v1)**

> **作者:** Zhiyuan Liu; Yicun Yang; Yaojie Zhang; Junjie Chen; Chang Zou; Qingyuan Wei; Shaobo Wang; Linfeng Zhang
>
> **摘要:** Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub.
>
---
#### [new 166] ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols
- **分类: cs.AR; cs.AI; cs.CL**

- **简介: 该论文属于硬件设计任务，旨在解决LLM生成正确SystemVerilog代码的问题。通过构建基准测试，评估模型在通信协议实现中的能力。**

- **链接: [http://arxiv.org/pdf/2506.07945v1](http://arxiv.org/pdf/2506.07945v1)**

> **作者:** Arnav Sheth; Ivaxi Sheth; Mario Fritz
>
> **备注:** Accepted at MLSysArch@ISCA 2025
>
> **摘要:** Recent advances in Large Language Models (LLMs) have shown promising capabilities in generating code for general-purpose programming languages. In contrast, their applicability for hardware description languages, particularly for generating synthesizable and functionally correct designs, remains significantly underexplored. HDLs such as SystemVerilog are logic-oriented and demand strict adherence to timing semantics, concurrency, and synthesizability constraints. Moreover, HDL-based design flows encompass a broad set of tasks beyond structural code generation, including testbench development, assertion-based verification, timing closure, and protocol-level integration for on-chip communication. The objective of our paper is to analyze the capabilities of state-of-the-art LLMs in generating SystemVerilog implementations of standard communication protocols, a core component of embedded and System-on-Chip (SoC) architectures. This paper introduces the first benchmark suite targeting four widely used protocols: SPI, I2C, UART, and AXI. We define code generation tasks that capture varying levels of design abstraction and prompt specificity. The generated designs are assessed for syntactic correctness, synthesizability, and functional fidelity via waveform simulation and test benches.
>
---
#### [new 167] Contextual Experience Replay for Self-Improvement of Language Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于语言代理的自我改进任务，旨在解决LLM在复杂任务中缺乏环境经验的问题。提出CER框架，通过记忆回放提升代理的适应能力。**

- **链接: [http://arxiv.org/pdf/2506.06698v1](http://arxiv.org/pdf/2506.06698v1)**

> **作者:** Yitao Liu; Chenglei Si; Karthik Narasimhan; Shunyu Yao
>
> **备注:** Accepted to ACL 2025. 20 pages
>
> **摘要:** Large language model (LLM) agents have been applied to sequential decision-making tasks such as web navigation, but without any environment-specific experiences, they often fail in these complex tasks. Moreover, current LLM agents are not designed to continually learn from past experiences during inference time, which could be crucial for them to gain these environment-specific experiences. To address this, we propose Contextual Experience Replay (CER), a training-free framework to enable efficient self-improvement for language agents in their context window. Specifically, CER accumulates and synthesizes past experiences into a dynamic memory buffer. These experiences encompass environment dynamics and common decision-making patterns, allowing the agents to retrieve and augment themselves with relevant knowledge in new tasks, enhancing their adaptability in complex environments. We evaluate CER on the challenging WebArena and VisualWebArena benchmarks. On VisualWebArena, CER achieves a competitive performance of 31.9%. On WebArena, CER also gets a competitive average success rate of 36.7%, relatively improving the success rate of the GPT-4o agent baseline by 51.0%. We also conduct a comprehensive analysis on it to prove its efficiency, validity and understand it better.
>
---
#### [new 168] Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于LLM安全研究任务，旨在解决隐性危害问题。通过构建基准测试，揭示对齐失败带来的更隐蔽的安全风险。**

- **链接: [http://arxiv.org/pdf/2506.07402v1](http://arxiv.org/pdf/2506.07402v1)**

> **作者:** Yukai Zhou; Sibei Yang; Wenjie Wang
>
> **摘要:** Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about their security. While jailbreak attacks highlight failures under overtly harmful queries, they overlook a critical risk: incorrectly answering harmless-looking inputs can be dangerous and cause real-world harm (Implicit Harm). We systematically reformulate the LLM risk landscape through a structured quadrant perspective based on output factuality and input harmlessness, uncovering an overlooked high-risk region. To investigate this gap, we propose JailFlipBench, a benchmark aims to capture implicit harm, spanning single-modal, multimodal, and factual extension scenarios with diverse evaluation metrics. We further develop initial JailFlip attack methodologies and conduct comprehensive evaluations across multiple open-source and black-box LLMs, show that implicit harm present immediate and urgent real-world risks, calling for broader LLM safety assessments and alignment beyond conventional jailbreak paradigms.
>
---
#### [new 169] Uncovering the Functional Roles of Nonlinearity in Memory
- **分类: cs.LG; cs.AI; cs.CL; nlin.CD; physics.comp-ph**

- **简介: 该论文研究序列建模任务中非线性在记忆中的作用，探讨何时需要非线性及其实现机制，提出AL-RNN模型进行分析。**

- **链接: [http://arxiv.org/pdf/2506.07919v1](http://arxiv.org/pdf/2506.07919v1)**

> **作者:** Manuel Brenner; Georgia Koppe
>
> **备注:** Preprint under review
>
> **摘要:** Memory and long-range temporal processing are core requirements for sequence modeling tasks across natural language processing, time-series forecasting, speech recognition, and control. While nonlinear recurrence has long been viewed as essential for enabling such mechanisms, recent work suggests that linear dynamics may often suffice. In this study, we go beyond performance comparisons to systematically dissect the functional role of nonlinearity in recurrent networks--identifying both when it is computationally necessary, and what mechanisms it enables. We use Almost Linear Recurrent Neural Networks (AL-RNNs), which allow fine-grained control over nonlinearity, as both a flexible modeling tool and a probe into the internal mechanisms of memory. Across a range of classic sequence modeling tasks and a real-world stimulus selection task, we find that minimal nonlinearity is not only sufficient but often optimal, yielding models that are simpler, more robust, and more interpretable than their fully nonlinear or linear counterparts. Our results provide a principled framework for selectively introducing nonlinearity, bridging dynamical systems theory with the functional demands of long-range memory and structured computation in recurrent neural networks, with implications for both artificial and biological neural systems.
>
---
#### [new 170] Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques
- **分类: cs.LG; cs.AI; cs.CL; cs.DC**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理效率低的问题。通过分析路由和分层推理技术，提出优化模型选择策略以提升计算效率。**

- **链接: [http://arxiv.org/pdf/2506.06579v1](http://arxiv.org/pdf/2506.06579v1)**

> **作者:** Adarsh Prasad Behera; Jaya Prakash Champati; Roberto Morabito; Sasu Tarkoma; James Gross
>
> **摘要:** Recent progress in Language Models (LMs) has dramatically advanced the field of natural language processing (NLP), excelling at tasks like text generation, summarization, and question answering. However, their inference remains computationally expensive and energy intensive, especially in settings with limited hardware, power, or bandwidth. This makes it difficult to deploy LMs in mobile, edge, or cost sensitive environments. To address these challenges, recent approaches have introduced multi LLM intelligent model selection strategies that dynamically allocate computational resources based on query complexity -- using lightweight models for simpler queries and escalating to larger models only when necessary. This survey explores two complementary strategies for efficient LLM inference: (i) routing, which selects the most suitable model based on the query, and (ii) cascading or hierarchical inference (HI), which escalates queries through a sequence of models until a confident response is found. Both approaches aim to reduce computation by using lightweight models for simpler tasks while offloading only when needed. We provide a comparative analysis of these techniques across key performance metrics, discuss benchmarking efforts, and outline open challenges. Finally, we outline future research directions to enable faster response times, adaptive model selection based on task complexity, and scalable deployment across heterogeneous environments, making LLM based systems more efficient and accessible for real world applications.
>
---
#### [new 171] Mitigating Object Hallucination via Robust Local Perception Search
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，旨在解决MLLMs中的幻觉问题。提出LPS方法，在推理时利用局部视觉先验抑制幻觉，效果显著。**

- **链接: [http://arxiv.org/pdf/2506.06729v1](http://arxiv.org/pdf/2506.06729v1)**

> **作者:** Zixian Gao; Chao Yang; Zhanhui Zhou; Xing Xu; Chaochao Lu
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have enabled them to effectively integrate vision and language, addressing a variety of downstream tasks. However, despite their significant success, these models still exhibit hallucination phenomena, where the outputs appear plausible but do not align with the content of the images. To mitigate this issue, we introduce Local Perception Search (LPS), a decoding method during inference that is both simple and training-free, yet effectively suppresses hallucinations. This method leverages local visual prior information as a value function to correct the decoding process. Additionally, we observe that the impact of the local visual prior on model performance is more pronounced in scenarios with high levels of image noise. Notably, LPS is a plug-and-play approach that is compatible with various models. Extensive experiments on widely used hallucination benchmarks and noisy data demonstrate that LPS significantly reduces the incidence of hallucinations compared to the baseline, showing exceptional performance, particularly in noisy settings.
>
---
#### [new 172] When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于安全防御任务，旨在解决语言模型因风格对齐导致的安全漏洞问题。通过分析风格模式对攻击成功率的影响，并提出SafeStyle防御策略提升模型安全性。**

- **链接: [http://arxiv.org/pdf/2506.07452v1](http://arxiv.org/pdf/2506.07452v1)**

> **作者:** Yuxin Xiao; Sana Tonekaboni; Walter Gerych; Vinith Suriyakumar; Marzyeh Ghassemi
>
> **摘要:** Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety.
>
---
#### [new 173] GLOS: Sign Language Generation with Temporally Aligned Gloss-Level Conditioning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到手语生成任务，解决现有方法中词汇顺序错误和语义不准确的问题。通过引入时间对齐的词素条件和条件融合模块，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2506.07460v1](http://arxiv.org/pdf/2506.07460v1)**

> **作者:** Taeryung Lee; Hyeongjin Nam; Gyeongsik Moon; Kyoung Mu Lee
>
> **摘要:** Sign language generation (SLG), or text-to-sign generation, bridges the gap between signers and non-signers. Despite recent progress in SLG, existing methods still often suffer from incorrect lexical ordering and low semantic accuracy. This is primarily due to sentence-level condition, which encodes the entire sentence of the input text into a single feature vector as a condition for SLG. This approach fails to capture the temporal structure of sign language and lacks the granularity of word-level semantics, often leading to disordered sign sequences and ambiguous motions. To overcome these limitations, we propose GLOS, a sign language generation framework with temporally aligned gloss-level conditioning. First, we employ gloss-level conditions, which we define as sequences of gloss embeddings temporally aligned with the motion sequence. This enables the model to access both the temporal structure of sign language and word-level semantics at each timestep. As a result, this allows for fine-grained control of signs and better preservation of lexical order. Second, we introduce a condition fusion module, temporal alignment conditioning (TAC), to efficiently deliver the word-level semantic and temporal structure provided by the gloss-level condition to the corresponding motion timesteps. Our method, which is composed of gloss-level conditions and TAC, generates signs with correct lexical order and high semantic accuracy, outperforming prior methods on CSL-Daily and Phoenix-2014T.
>
---
#### [new 174] MarginSel : Max-Margin Demonstration Selection for LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLMs在少样本学习中示例选择与排序的问题。提出MarginSel方法，通过选择困难示例提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.06699v1](http://arxiv.org/pdf/2506.06699v1)**

> **作者:** Rajeev Bhatt Ambati; James Lester; Shashank Srivastava; Snigdha Chaturvedi
>
> **摘要:** Large Language Models (LLMs) excel at few-shot learning via in-context learning (ICL). However, the effectiveness of ICL is often sensitive to the selection and ordering of demonstration examples. To address this, we present MarginSel: Max-Margin Demonstration Selection for LLMs, a two-step method that selects hard demonstration examples for the ICL prompt, adapting to each test instance. Our approach achieves 2-7% absolute improvement in F1-score across classification tasks, compared to a random selection of examples. We also provide theoretical insights and empirical evidence showing that MarginSel induces max-margin behavior in LLMs by effectively increasing the margin for hard examples, analogous to support vectors, thereby shifting the decision boundary in a beneficial direction.
>
---
#### [new 175] Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI认知能力评估任务，旨在检验大语言模型是否能解决框架问题和符号接地问题。研究设计了两个基准任务，在零样本条件下测试13个模型的表现。**

- **链接: [http://arxiv.org/pdf/2506.07896v1](http://arxiv.org/pdf/2506.07896v1)**

> **作者:** Shoko Oka
>
> **备注:** 52 pages, Additional resources available on GitHub repository
>
> **摘要:** Recent advancements in large language models (LLMs) have revitalized philosophical debates surrounding artificial intelligence. Two of the most fundamental challenges - namely, the Frame Problem and the Symbol Grounding Problem - have historically been viewed as unsolvable within traditional symbolic AI systems. This study investigates whether modern LLMs possess the cognitive capacities required to address these problems. To do so, I designed two benchmark tasks reflecting the philosophical core of each problem, administered them under zero-shot conditions to 13 prominent LLMs (both closed and open-source), and assessed the quality of the models' outputs across five trials each. Responses were scored along multiple criteria, including contextual reasoning, semantic coherence, and information filtering. The results demonstrate that while open-source models showed variability in performance due to differences in model size, quantization, and instruction tuning, several closed models consistently achieved high scores. These findings suggest that select modern LLMs may be acquiring capacities sufficient to produce meaningful and stable responses to these long-standing theoretical challenges.
>
---
#### [new 176] LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于推荐系统任务，旨在解决LLM推荐中缺乏结构化知识的问题。通过引入知识图谱增强RAG框架，提升推荐效果与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.07449v1](http://arxiv.org/pdf/2506.07449v1)**

> **作者:** Vahid Azizi; Fatemeh Koochaki
>
> **摘要:** Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems. Code is available at~\href{https://github.com/VahidAz/LlamaRec-LKG-RAG}{repository}.
>
---
#### [new 177] The Hype Index: an NLP-driven Measure of Market News Attention
- **分类: q-fin.ST; cs.CE; cs.CL**

- **简介: 该论文提出Hype Index，利用NLP分析市场新闻关注度，解决股票波动性与市场信号问题，通过构建两种指数并评估其性能。**

- **链接: [http://arxiv.org/pdf/2506.06329v1](http://arxiv.org/pdf/2506.06329v1)**

> **作者:** Zheng Cao; Wanchaloem Wunkaew; Helyette Geman
>
> **摘要:** This paper introduces the Hype Index as a novel metric to quantify media attention toward large-cap equities, leveraging advances in Natural Language Processing (NLP) for extracting predictive signals from financial news. Using the S&P 100 as the focus universe, we first construct a News Count-Based Hype Index, which measures relative media exposure by computing the share of news articles referencing each stock or sector. We then extend it to the Capitalization Adjusted Hype Index, adjusts for economic size by taking the ratio of a stock's or sector's media weight to its market capitalization weight within its industry or sector. We compute both versions of the Hype Index at the stock and sector levels, and evaluate them through multiple lenses: (1) their classification into different hype groups, (2) their associations with returns, volatility, and VIX index at various lags, (3) their signaling power for short-term market movements, and (4) their empirical properties including correlations, samplings, and trends. Our findings suggest that the Hype Index family provides a valuable set of tools for stock volatility analysis, market signaling, and NLP extensions in Finance.
>
---
#### [new 178] HauntAttack: When Attack Follows Reasoning as a Shadow
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全与隐私任务，旨在解决大型推理模型的安全漏洞问题。通过提出HauntAttack框架，研究模型在遭遇有害指令时的不安全输出行为。**

- **链接: [http://arxiv.org/pdf/2506.07031v1](http://arxiv.org/pdf/2506.07031v1)**

> **作者:** Jingyuan Ma; Rui Li; Zheng Li; Junfeng Liu; Lei Sha; Zhifang Sui
>
> **摘要:** Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs.
>
---
#### [new 179] FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models
- **分类: cs.IR; cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于金融自然语言处理任务，旨在解决LLMs在金融领域应用中的性能不足问题。通过构建FinBERT2模型，提升分类、检索和主题建模效果。**

- **链接: [http://arxiv.org/pdf/2506.06335v1](http://arxiv.org/pdf/2506.06335v1)**

> **作者:** Xuan Xu; Fufang Wen; Beilin Chu; Zhibing Fu; Qinhong Lin; Jiaqi Liu; Binjie Fei; Zhongliang Yang; Linna Zhou; Yu Li
>
> **摘要:** In natural language processing (NLP), the focus has shifted from encoder-only tiny language models like BERT to decoder-only large language models(LLMs) such as GPT-3. However, LLMs' practical application in the financial sector has revealed three limitations: (1) LLMs often perform worse than fine-tuned BERT on discriminative tasks despite costing much higher computational resources, such as market sentiment analysis in financial reports; (2) Application on generative tasks heavily relies on retrieval augmented generation (RAG) methods to provide current and specialized information, with general retrievers showing suboptimal performance on domain-specific retrieval tasks; (3) There are additional inadequacies in other feature-based scenarios, such as topic modeling. We introduce FinBERT2, a specialized bidirectional encoder pretrained on a high-quality, financial-specific corpus of 32b tokens. This represents the largest known Chinese financial pretraining corpus for models of this parameter size. As a better backbone, FinBERT2 can bridge the gap in the financial-specific deployment of LLMs through the following achievements: (1) Discriminative fine-tuned models (Fin-Labelers) outperform other (Fin)BERT variants by 0.4%-3.3% and leading LLMs by 9.7%-12.3% on average across five financial classification tasks. (2) Contrastive fine-tuned models (Fin-Retrievers) outperform both open-source (e.g., +6.8\% avg improvement over BGE-base-zh) and proprietary (e.g., +4.2\% avg improvement over OpenAI's text-embedding-3-large) embedders across five financial retrieval tasks; (3) Building on FinBERT2 variants, we construct the Fin-TopicModel, which enables superior clustering and topic representation for financial titles. Our work revisits financial BERT models through comparative analysis with contemporary LLMs and offers practical insights for effectively utilizing FinBERT in the LLMs era.
>
---
#### [new 180] Play to Generalize: Learning to Reason Through Game Play
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型的泛化推理任务，旨在提升模型在未见领域的推理能力。通过游戏训练，模型在数学和跨学科任务中表现优异，且保持视觉基准性能。**

- **链接: [http://arxiv.org/pdf/2506.08011v1](http://arxiv.org/pdf/2506.08011v1)**

> **作者:** Yunfei Xie; Yinsong Ma; Shiyi Lan; Alan Yuille; Junfei Xiao; Chen Wei
>
> **备注:** Project Page: https://yunfeixie233.github.io/ViGaL/
>
> **摘要:** Developing generalizable reasoning capabilities in multimodal large language models (MLLMs) remains challenging. Motivated by cognitive science literature suggesting that gameplay promotes transferable cognitive skills, we propose a novel post-training paradigm, Visual Game Learning, or ViGaL, where MLLMs develop out-of-domain generalization of multimodal reasoning through playing arcade-like games. Specifically, we show that post-training a 7B-parameter MLLM via reinforcement learning (RL) on simple arcade-like games, e.g. Snake, significantly enhances its downstream performance on multimodal math benchmarks like MathVista, and on multi-discipline questions like MMMU, without seeing any worked solutions, equations, or diagrams during RL, suggesting the capture of transferable reasoning skills. Remarkably, our model outperforms specialist models tuned on multimodal reasoning data in multimodal reasoning benchmarks, while preserving the base model's performance on general visual benchmarks, a challenge where specialist models often fall short. Our findings suggest a new post-training paradigm: synthetic, rule-based games can serve as controllable and scalable pre-text tasks that unlock generalizable multimodal reasoning abilities in MLLMs.
>
---
#### [new 181] Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models
- **分类: cs.LG; cs.CL; cs.MA**

- **简介: 该论文属于语言模型安全任务，旨在解决传统安全对齐滞后问题。通过在线自博弈强化学习，实现攻击者与防御者的协同进化，提升模型安全性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.07468v1](http://arxiv.org/pdf/2506.07468v1)**

> **作者:** Mickel Liu; Liwei Jiang; Yancheng Liang; Simon Shaolei Du; Yejin Choi; Tim Althoff; Natasha Jaques
>
> **摘要:** Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).
>
---
#### [new 182] Hallucination at a Glance: Controlled Visual Edits and Fine-Grained Multimodal Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言任务，解决MLLMs在细粒度视觉差异上的幻觉问题。通过构建数据集和引入一致性损失，提升模型对细微变化的敏感性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.07227v1](http://arxiv.org/pdf/2506.07227v1)**

> **作者:** Tianyi Bai; Yuxuan Fan; Jiantao Qiu; Fupeng Sun; Jiayi Song; Junlin Han; Zichen Liu; Conghui He; Wentao Zhang; Binhang Yuan
>
> **摘要:** Multimodal large language models (MLLMs) have achieved strong performance on vision-language tasks but still struggle with fine-grained visual differences, leading to hallucinations or missed semantic shifts. We attribute this to limitations in both training data and learning objectives. To address these issues, we propose a controlled data generation pipeline that produces minimally edited image pairs with semantically aligned captions. Using this pipeline, we construct the Micro Edit Dataset (MED), containing over 50K image-text pairs spanning 11 fine-grained edit categories, including attribute, count, position, and object presence changes. Building on MED, we introduce a supervised fine-tuning (SFT) framework with a feature-level consistency loss that promotes stable visual embeddings under small edits. We evaluate our approach on the Micro Edit Detection benchmark, which includes carefully balanced evaluation pairs designed to test sensitivity to subtle visual variations across the same edit categories. Our method improves difference detection accuracy and reduces hallucinations compared to strong baselines, including GPT-4o. Moreover, it yields consistent gains on standard vision-language tasks such as image captioning and visual question answering. These results demonstrate the effectiveness of combining targeted data and alignment objectives for enhancing fine-grained visual reasoning in MLLMs.
>
---
#### [new 183] HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于组合优化任务，旨在评估LLMs生成启发式算法的能力。提出HeuriGym框架和QYI指标，发现模型存在工具使用和适应性推理不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.07972v1](http://arxiv.org/pdf/2506.07972v1)**

> **作者:** Hongzheng Chen; Yingheng Wang; Yaohui Cai; Hins Hu; Jiajie Li; Shirley Huang; Chenhui Deng; Rongjian Liang; Shufeng Kong; Haoxing Ren; Samitha Samaranayake; Carla P. Gomes; Zhiru Zhang
>
> **摘要:** While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains.
>
---
#### [new 184] Speaker-Distinguishable CTC: Learning Speaker Distinction Using CTC for Multi-Talker Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在解决Speaker Assignment失败导致的识别错误问题。提出SD-CTC框架，联合分配音素和说话人标签，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.07515v1](http://arxiv.org/pdf/2506.07515v1)**

> **作者:** Asahi Sakuma; Hiroaki Sato; Ryuga Sugano; Tadashi Kumano; Yoshihiko Kawai; Tetsuji Ogawa
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** This paper presents a novel framework for multi-talker automatic speech recognition without the need for auxiliary information. Serialized Output Training (SOT), a widely used approach, suffers from recognition errors due to speaker assignment failures. Although incorporating auxiliary information, such as token-level timestamps, can improve recognition accuracy, extracting such information from natural conversational speech remains challenging. To address this limitation, we propose Speaker-Distinguishable CTC (SD-CTC), an extension of CTC that jointly assigns a token and its corresponding speaker label to each frame. We further integrate SD-CTC into the SOT framework, enabling the SOT model to learn speaker distinction using only overlapping speech and transcriptions. Experimental comparisons show that multi-task learning with SD-CTC and SOT reduces the error rate of the SOT model by 26% and achieves performance comparable to state-of-the-art methods relying on auxiliary information.
>
---
#### [new 185] Improving large language models with concept-aware fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM概念理解不足的问题。通过多token训练方法CAFT，提升模型对高阶概念的把握能力。**

- **链接: [http://arxiv.org/pdf/2506.07833v1](http://arxiv.org/pdf/2506.07833v1)**

> **作者:** Michael K. Chen; Xikun Zhang; Jiaxing Huang; Dacheng Tao
>
> **摘要:** Large language models (LLMs) have become the cornerstone of modern AI. However, the existing paradigm of next-token prediction fundamentally limits their ability to form coherent, high-level concepts, making it a critical barrier to human-like understanding and reasoning. Take the phrase "ribonucleic acid" as an example: an LLM will first decompose it into tokens, i.e., artificial text fragments ("rib", "on", ...), then learn each token sequentially, rather than grasping the phrase as a unified, coherent semantic entity. This fragmented representation hinders deeper conceptual understanding and, ultimately, the development of truly intelligent systems. In response, we introduce Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned. By enabling the learning of sequences that span multiple tokens, this method fosters stronger concept-aware learning. Our experiments demonstrate significant improvements compared to conventional next-token finetuning methods across diverse tasks, including traditional applications like text summarization and domain-specific ones like de novo protein design. Multi-token prediction was previously only possible in the prohibitively expensive pretraining phase; CAFT, to our knowledge, is the first to bring the multi-token setting to the post-training phase, thus effectively democratizing its benefits for the broader community of practitioners and researchers. Finally, the unexpected effectiveness of our proposed method suggests wider implications for the machine learning research community. All code and data are available at https://github.com/michaelchen-lab/caft-llm
>
---
#### [new 186] How Malicious AI Swarms Can Threaten Democracy
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI安全任务，探讨恶意AI群组对民主的威胁，提出平台、模型和系统三层应对策略。**

- **链接: [http://arxiv.org/pdf/2506.06299v1](http://arxiv.org/pdf/2506.06299v1)**

> **作者:** Daniel Thilo Schroeder; Meeyoung Cha; Andrea Baronchelli; Nick Bostrom; Nicholas A. Christakis; David Garcia; Amit Goldenberg; Yara Kyrychenko; Kevin Leyton-Brown; Nina Lutz; Gary Marcus; Filippo Menczer; Gordon Pennycook; David G. Rand; Frank Schweitzer; Christopher Summerfield; Audrey Tang; Jay Van Bavel; Sander van der Linden; Dawn Song; Jonas R. Kunst
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** Advances in AI portend a new era of sophisticated disinformation operations. While individual AI systems already create convincing -- and at times misleading -- information, an imminent development is the emergence of malicious AI swarms. These systems can coordinate covertly, infiltrate communities, evade traditional detectors, and run continuous A/B tests, with round-the-clock persistence. The result can include fabricated grassroots consensus, fragmented shared reality, mass harassment, voter micro-suppression or mobilization, contamination of AI training data, and erosion of institutional trust. With democratic processes worldwide increasingly vulnerable, we urge a three-pronged response: (1) platform-side defenses -- always-on swarm-detection dashboards, pre-election high-fidelity swarm-simulation stress-tests, transparency audits, and optional client-side "AI shields" for users; (2) model-side safeguards -- standardized persuasion-risk tests, provenance-authenticating passkeys, and watermarking; and (3) system-level oversight -- a UN-backed AI Influence Observatory.
>
---
#### [new 187] SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SAFEFLOW协议，解决自主代理系统的安全与可靠性问题，通过信息流控制、事务执行等机制提升系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.07564v1](http://arxiv.org/pdf/2506.07564v1)**

> **作者:** Peiran Li; Xinkai Zou; Zhuohang Wu; Ruifeng Li; Shuo Xing; Hanwen Zheng; Zhikai Hu; Yuping Wang; Haoxi Li; Qin Yuan; Yingmo Zhang; Zhengzhong Tu
>
> **摘要:** Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled powerful autonomous agents capable of complex reasoning and multi-modal tool use. Despite their growing capabilities, today's agent frameworks remain fragile, lacking principled mechanisms for secure information flow, reliability, and multi-agent coordination. In this work, we introduce SAFEFLOW, a new protocol-level framework for building trustworthy LLM/VLM-based agents. SAFEFLOW enforces fine-grained information flow control (IFC), precisely tracking provenance, integrity, and confidentiality of all the data exchanged between agents, tools, users, and environments. By constraining LLM reasoning to respect these security labels, SAFEFLOW prevents untrusted or adversarial inputs from contaminating high-integrity decisions. To ensure robustness in concurrent multi-agent settings, SAFEFLOW introduces transactional execution, conflict resolution, and secure scheduling over shared state, preserving global consistency across agents. We further introduce mechanisms, including write-ahead logging, rollback, and secure caches, that further enhance resilience against runtime errors and policy violations. To validate the performances, we built SAFEFLOWBENCH, a comprehensive benchmark suite designed to evaluate agent reliability under adversarial, noisy, and concurrent operational conditions. Extensive experiments demonstrate that agents built with SAFEFLOW maintain impressive task performance and security guarantees even in hostile environments, substantially outperforming state-of-the-art. Together, SAFEFLOW and SAFEFLOWBENCH lay the groundwork for principled, robust, and secure agent ecosystems, advancing the frontier of reliable autonomy.
>
---
#### [new 188] Large Language Models Can Be a Viable Substitute for Expert Political Surveys When a Shock Disrupts Traditional Measurement Approaches
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于政治分析任务，旨在解决突发事件后传统调查失效的问题。通过LLMs模拟专家判断，分析DOGE裁员案例，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.06540v1](http://arxiv.org/pdf/2506.06540v1)**

> **作者:** Patrick Y. Wu
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** After a disruptive event or shock, such as the Department of Government Efficiency (DOGE) federal layoffs of 2025, expert judgments are colored by knowledge of the outcome. This can make it difficult or impossible to reconstruct the pre-event perceptions needed to study the factors associated with the event. This position paper argues that large language models (LLMs), trained on vast amounts of digital media data, can be a viable substitute for expert political surveys when a shock disrupts traditional measurement. We analyze the DOGE layoffs as a specific case study for this position. We use pairwise comparison prompts with LLMs and derive ideology scores for federal executive agencies. These scores replicate pre-layoff expert measures and predict which agencies were targeted by DOGE. We also use this same approach and find that the perceptions of certain federal agencies as knowledge institutions predict which agencies were targeted by DOGE, even when controlling for ideology. This case study demonstrates that using LLMs allows us to rapidly and easily test the associated factors hypothesized behind the shock. More broadly, our case study of this recent event exemplifies how LLMs offer insights into the correlational factors of the shock when traditional measurement techniques fail. We conclude by proposing a two-part criterion for when researchers can turn to LLMs as a substitute for expert political surveys.
>
---
#### [new 189] Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在检测黑盒LLM API是否被替换或篡改。通过提出一种基于排名的均匀性检验方法，验证模型行为一致性。**

- **链接: [http://arxiv.org/pdf/2506.06975v1](http://arxiv.org/pdf/2506.06975v1)**

> **作者:** Xiaoyuan Zhu; Yaowen Ye; Tianyi Qiu; Hanlin Zhu; Sijun Tan; Ajraf Mannan; Jonathan Michala; Raluca Ada Popa; Willie Neiswanger
>
> **摘要:** As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets.
>
---
#### [new 190] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决小样本学习中模型性能不稳定的问题。通过元学习和提示蒸馏方法提升模型适应能力。**

- **链接: [http://arxiv.org/pdf/2506.06905v1](http://arxiv.org/pdf/2506.06905v1)**

> **作者:** Akash Gupta; Amos Storkey; Mirella Lapata
>
> **摘要:** Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, is inconsistent and does not always improve monotonically with increasing examples. We hypothesize that this occurs due to the LMM being overwhelmed by additional information present in the image embeddings, which is not required for the downstream task. To address this, we propose a meta-learning approach that provides an alternative for inducing few-shot capabilities in LMMs, using a fixed set of soft prompts that are distilled from task-relevant image features and can be adapted at test time using a few examples. To facilitate this distillation, we introduce an attention-mapper module that can be easily integrated with the popular LLaVA v1.5 architecture and is jointly learned with soft prompts, enabling task adaptation in LMMs under low-data regimes with just a few gradient steps. Evaluation on the VL-ICL Bench shows that our method consistently outperforms ICL and related prompt-tuning approaches, even under image perturbations, improving task induction and reasoning across visual question answering tasks.
>
---
## 更新

#### [replaced 001] Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.00979v2](http://arxiv.org/pdf/2503.00979v2)**

> **作者:** Ravi Ghadia; Avinash Kumar; Gaurav Jain; Prashant Nair; Poulami Das
>
> **备注:** Published in the Proceedings of the 42nd International Conference on Machine Learning (ICML), Vancouver, Canada
>
> **摘要:** Autoregressive Transformers rely on Key-Value (KV) caching to accelerate inference. However, the linear growth of the KV cache with context length leads to excessive memory consumption and bandwidth constraints. This bottleneck is particularly problematic in real-time applications -- such as chatbots and interactive assistants -- where low latency and high memory efficiency are critical. Existing methods drop distant tokens or compress states in a lossy manner, sacrificing accuracy by discarding vital context or introducing bias. We propose MorphKV, an inference-time technique that maintains a constant-sized KV cache while preserving accuracy. MorphKV balances long-range dependencies and local coherence during text generation. It eliminates early-token bias while retaining high-fidelity context by adaptively ranking tokens through correlation-aware selection. Unlike heuristic retention or lossy compression, MorphKV iteratively refines the KV cache via lightweight updates guided by attention patterns of recent tokens. This approach captures inter-token correlation with greater accuracy, crucial for tasks like content creation and code generation. Our studies on long-response tasks show 52.9$\%$ memory savings and 18.2$\%$ higher accuracy on average compared to state-of-the-art prior works, enabling efficient real-world deployment.
>
---
#### [replaced 002] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13791v2](http://arxiv.org/pdf/2502.13791v2)**

> **作者:** Nathanaël Carraz Rakotonirina; Mohammed Hamdy; Jon Ander Campos; Lucas Weber; Alberto Testoni; Marzieh Fadaee; Sandro Pezzelle; Marco Del Tredici
>
> **备注:** Published as conference paper at ACL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly used in working environments for a wide range of tasks, excelling at solving individual problems in isolation. However, are they also able to effectively collaborate over long-term interactions? To investigate this, we introduce MemoryCode, a synthetic multi-session dataset designed to test LLMs' ability to track and execute simple coding instructions amid irrelevant information, simulating a realistic setting. While all the models we tested handle isolated instructions well, even the performance of state-of-the-art models like GPT-4o deteriorates when instructions are spread across sessions. Our analysis suggests this is due to their failure to retrieve and integrate information over long instruction chains. Our results highlight a fundamental limitation of current LLMs, restricting their ability to collaborate effectively in long interactions.
>
---
#### [replaced 003] Efficient Pretraining Data Selection for Language Models via Multi-Actor Collaboration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08102v3](http://arxiv.org/pdf/2410.08102v3)**

> **作者:** Tianyi Bai; Ling Yang; Zhen Hao Wong; Fupeng Sun; Jiahui Peng; Xinlin Zhuang; Chi Zhang; Lijun Wu; Jiantao Qiu; Wentao Zhang; Binhang Yuan; Conghui He
>
> **摘要:** Efficient data selection is crucial to accelerate the pretraining of language model (LMs). While various methods have been proposed to enhance data efficiency, limited research has addressed the inherent conflicts between these approaches to achieve optimal data selection for LM pretraining. To tackle this problem, we propose a multi-actor collaborative data selection mechanism: each data selection method independently prioritizes data based on its criterion and updates its prioritization rules using the current state of the model, functioning as an independent actor for data selection; and a console is designed to adjust the impacts of different actors at various stages and dynamically integrate information from all actors throughout the LM pretraining process. We conduct extensive empirical studies to evaluate our multi-actor framework. The experimental results demonstrate that our approach significantly improves data efficiency, accelerates convergence in LM pretraining, and achieves an average relative performance gain up to $10.5\%$ across multiple language model benchmarks compared to the state-of-the-art methods.
>
---
#### [replaced 004] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.00142v3](http://arxiv.org/pdf/2412.00142v3)**

> **作者:** Chancharik Mitra; Brandon Huang; Tianning Chai; Zhiqiu Lin; Assaf Arbelle; Rogerio Feris; Leonid Karlinsky; Trevor Darrell; Deva Ramanan; Roei Herzig
>
> **摘要:** Generative Large Multimodal Models (LMMs) like LLaVA and Qwen-VL excel at a wide variety of vision-language (VL) tasks. Despite strong performance, LMMs' generative outputs are not specialized for vision-language classification tasks (i.e., tasks with vision-language inputs and discrete labels) such as image classification and multiple-choice VQA. One key challenge in utilizing LMMs for these tasks is the extraction of useful features from generative LMMs. To overcome this, we propose an approach that leverages multimodal feature extraction from the LMM's latent space. Toward this end, we present Sparse Attention Vectors (SAVs) -- a finetuning-free method that leverages sparse attention head activations (fewer than 5% of the heads) in LMMs as strong feature representations. With only few-shot examples, SAVs demonstrate state-of-the-art performance compared to a variety of few-shot and finetuned baselines on a collection of vision-language classification tasks. Our experiments also imply that SAVs can scale in performance with additional examples and generalize to similar tasks, establishing SAVs as both effective and robust multimodal feature representations.
>
---
#### [replaced 005] Examining the Mental Health Impact of Misinformation on Social Media Using a Hybrid Transformer-Based Approach
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02333v2](http://arxiv.org/pdf/2503.02333v2)**

> **作者:** Sarvesh Arora; Sarthak Arora; Deepika Kumar; Vallari Agrawal; Vedika Gupta; Dipit Vasdev
>
> **备注:** Paper has data authenticity problems, ALL AUTHORS confirm they wish to withdraw this paper
>
> **摘要:** Social media has significantly reshaped interpersonal communication, fostering connectivity while also enabling the proliferation of misinformation. The unchecked spread of false narratives has profound effects on mental health, contributing to increased stress, anxiety, and misinformation-driven paranoia. This study presents a hybrid transformer-based approach using a RoBERTa-LSTM classifier to detect misinformation, assess its impact on mental health, and classify disorders linked to misinformation exposure. The proposed models demonstrate accuracy rates of 98.4, 87.8, and 77.3 in detecting misinformation, mental health implications, and disorder classification, respectively. Furthermore, Pearson's Chi-Squared Test for Independence (p-value = 0.003871) validates the direct correlation between misinformation and deteriorating mental well-being. This study underscores the urgent need for better misinformation management strategies to mitigate its psychological repercussions. Future research could explore broader datasets incorporating linguistic, demographic, and cultural variables to deepen the understanding of misinformation-induced mental health distress.
>
---
#### [replaced 006] Representation Bending for Large Language Model Safety
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2504.01550v2](http://arxiv.org/pdf/2504.01550v2)**

> **作者:** Ashkan Yousefpour; Taeheon Kim; Ryan S. Kwon; Seungbeen Lee; Wonje Jeung; Seungju Han; Alvin Wan; Harrison Ngan; Youngjae Yu; Jonghyun Choi
>
> **备注:** Accepted to ACL 2025 (main)
>
> **摘要:** Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.
>
---
#### [replaced 007] MIB: A Mechanistic Interpretability Benchmark
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13151v2](http://arxiv.org/pdf/2504.13151v2)**

> **作者:** Aaron Mueller; Atticus Geiger; Sarah Wiegreffe; Dana Arad; Iván Arcuschin; Adam Belfki; Yik Siu Chan; Jaden Fiotto-Kaufman; Tal Haklay; Michael Hanna; Jing Huang; Rohan Gupta; Yaniv Nikankin; Hadas Orgad; Nikhil Prakash; Anja Reusch; Aruna Sankaranarayanan; Shun Shao; Alessandro Stolfo; Martin Tutek; Amir Zur; David Bau; Yonatan Belinkov
>
> **备注:** Accepted to ICML 2025. Project website at https://mib-bench.github.io
>
> **摘要:** How can we know whether new mechanistic interpretability methods achieve real improvements? In pursuit of lasting evaluation standards, we propose MIB, a Mechanistic Interpretability Benchmark, with two tracks spanning four tasks and five models. MIB favors methods that precisely and concisely recover relevant causal pathways or causal variables in neural language models. The circuit localization track compares methods that locate the model components - and connections between them - most important for performing a task (e.g., attribution patching or information flow routes). The causal variable localization track compares methods that featurize a hidden vector, e.g., sparse autoencoders (SAEs) or distributed alignment search (DAS), and align those features to a task-relevant causal variable. Using MIB, we find that attribution and mask optimization methods perform best on circuit localization. For causal variable localization, we find that the supervised DAS method performs best, while SAE features are not better than neurons, i.e., non-featurized hidden vectors. These findings illustrate that MIB enables meaningful comparisons, and increases our confidence that there has been real progress in the field.
>
---
#### [replaced 008] Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18675v2](http://arxiv.org/pdf/2505.18675v2)**

> **作者:** Sicheng Feng; Song Wang; Shuyi Ouyang; Lingdong Kong; Zikai Song; Jianke Zhu; Huan Wang; Xinchao Wang
>
> **摘要:** Multimodal large language models (MLLMs) have recently achieved significant progress in visual tasks, including semantic scene understanding and text-image alignment, with reasoning variants enhancing performance on complex tasks involving mathematics and logic. However, their capacity for reasoning tasks involving fine-grained visual understanding remains insufficiently evaluated. To address this gap, we introduce ReasonMap, a benchmark designed to assess the fine-grained visual understanding and spatial reasoning abilities of MLLMs. ReasonMap encompasses high-resolution transit maps from 30 cities across 13 countries and includes 1,008 question-answer pairs spanning two question types and three templates. Furthermore, we design a two-level evaluation pipeline that properly assesses answer correctness and quality. Comprehensive evaluations of 15 popular MLLMs, including both base and reasoning variants, reveal a counterintuitive pattern: among open-source models, base models outperform reasoning ones, while the opposite trend is observed in closed-source models. Additionally, performance generally degrades when visual inputs are masked, indicating that while MLLMs can leverage prior knowledge to answer some questions, fine-grained visual reasoning tasks still require genuine visual perception for strong performance. Our benchmark study offers new insights into visual reasoning and contributes to investigating the gap between open-source and closed-source models.
>
---
#### [replaced 009] BioHopR: A Benchmark for Multi-Hop, Multi-Answer Reasoning in Biomedical Domain
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22240v2](http://arxiv.org/pdf/2505.22240v2)**

> **作者:** Yunsoo Kim; Yusuf Abdulle; Honghan Wu
>
> **摘要:** Biomedical reasoning often requires traversing interconnected relationships across entities such as drugs, diseases, and proteins. Despite the increasing prominence of large language models (LLMs), existing benchmarks lack the ability to evaluate multi-hop reasoning in the biomedical domain, particularly for queries involving one-to-many and many-to-many relationships. This gap leaves the critical challenges of biomedical multi-hop reasoning underexplored. To address this, we introduce BioHopR, a novel benchmark designed to evaluate multi-hop, multi-answer reasoning in structured biomedical knowledge graphs. Built from the comprehensive PrimeKG, BioHopR includes 1-hop and 2-hop reasoning tasks that reflect real-world biomedical complexities. Evaluations of state-of-the-art models reveal that O3-mini, a proprietary reasoning-focused model, achieves 37.93% precision on 1-hop tasks and 14.57% on 2-hop tasks, outperforming proprietary models such as GPT4O and open-source biomedical models including HuatuoGPT-o1-70B and Llama-3.3-70B. However, all models exhibit significant declines in multi-hop performance, underscoring the challenges of resolving implicit reasoning steps in the biomedical domain. By addressing the lack of benchmarks for multi-hop reasoning in biomedical domain, BioHopR sets a new standard for evaluating reasoning capabilities and highlights critical gaps between proprietary and open-source models while paving the way for future advancements in biomedical LLMs.
>
---
#### [replaced 010] Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.08248v3](http://arxiv.org/pdf/2501.08248v3)**

> **作者:** Yifu Qiu; Varun Embar; Yizhe Zhang; Navdeep Jaitly; Shay B. Cohen; Benjamin Han
>
> **摘要:** Recent advancements in long-context language models (LCLMs) promise to transform Retrieval-Augmented Generation (RAG) by simplifying pipelines. With their expanded context windows, LCLMs can process entire knowledge bases and perform retrieval and reasoning directly -- a capability we define as In-Context Retrieval and Reasoning (ICR^2). However, existing benchmarks like LOFT often overestimate LCLM performance by providing overly simplified contexts. To address this, we introduce ICR^2, a benchmark that evaluates LCLMs in more realistic scenarios by including confounding passages retrieved with strong retrievers. We then propose three methods to enhance LCLM performance: (1) retrieve-then-generate fine-tuning, (2) retrieval-attention-probing, which uses attention heads to filter and de-noise long contexts during decoding, and (3) joint retrieval head training alongside the generation head. Our evaluation of five well-known LCLMs on LOFT and ICR^2 demonstrates significant gains with our best approach applied to Mistral-7B: +17 and +15 points by Exact Match on LOFT, and +13 and +2 points on ICR^2, compared to vanilla RAG and supervised fine-tuning, respectively. It even outperforms GPT-4-Turbo on most tasks despite being a much smaller model.
>
---
#### [replaced 011] Multi-agent Architecture Search via Agentic Supernet
- **分类: cs.LG; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.04180v2](http://arxiv.org/pdf/2502.04180v2)**

> **作者:** Guibin Zhang; Luyang Niu; Junfeng Fang; Kun Wang; Lei Bai; Xiang Wang
>
> **摘要:** Large Language Model (LLM)-empowered multi-agent systems extend the cognitive boundaries of individual agents through disciplined collaboration and interaction, while constructing these systems often requires labor-intensive manual designs. Despite the availability of methods to automate the design of agentic workflows, they typically seek to identify a static, complex, one-size-fits-all system, which, however, fails to dynamically allocate inference resources based on the difficulty and domain of each query. To address this challenge, we shift away from the pursuit of a monolithic agentic system, instead optimizing the \textbf{agentic supernet}, a probabilistic and continuous distribution of agentic architectures. We introduce MaAS, an automated framework that samples query-dependent agentic systems from the supernet, delivering high-quality solutions and tailored resource allocation (\textit{e.g.}, LLM calls, tool calls, token cost). Comprehensive evaluation across six benchmarks demonstrates that MaAS \textbf{(I)} requires only $6\sim45\%$ of the inference costs of existing handcrafted or automated multi-agent systems, \textbf{(II)} surpasses them by $0.54\%\sim11.82\%$, and \textbf{(III)} enjoys superior cross-dataset and cross-LLM-backbone transferability.
>
---
#### [replaced 012] Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14767v2](http://arxiv.org/pdf/2502.14767v2)**

> **作者:** Priyanka Kargupta; Ishika Agarwal; Tal August; Jiawei Han
>
> **备注:** ACL 2025 Main Conference. Code available at: https://github.com/pkargupta/tree-of-debate
>
> **摘要:** With the exponential growth of research facilitated by modern technology and improved accessibility, scientific discoveries have become increasingly fragmented within and across fields. This makes it challenging to assess the significance, novelty, incremental findings, and equivalent ideas between related works, particularly those from different research communities. Large language models (LLMs) have recently demonstrated strong quantitative and qualitative reasoning abilities, and multi-agent LLM debates have shown promise in handling complex reasoning tasks by exploring diverse perspectives and reasoning paths. Inspired by this, we introduce Tree-of-Debate (ToD), a framework which converts scientific papers into LLM personas that debate their respective novelties. To emphasize structured, critical reasoning rather than focusing solely on outcomes, ToD dynamically constructs a debate tree, enabling fine-grained analysis of independent novelty arguments within scholarly articles. Through experiments on scientific literature across various domains, evaluated by expert researchers, we demonstrate that ToD generates informative arguments, effectively contrasts papers, and supports researchers in their literature review.
>
---
#### [replaced 013] Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.11005v3](http://arxiv.org/pdf/2410.11005v3)**

> **作者:** Fangru Lin; Shaoguang Mao; Emanuele La Malfa; Valentin Hofmann; Adrian de Wynter; Xun Wang; Si-Qing Chen; Michael Wooldridge; Janet B. Pierrehumbert; Furu Wei
>
> **备注:** ACL 2025 main
>
> **摘要:** Language is not monolithic. While benchmarks, including those designed for multiple languages, are often used as proxies to evaluate the performance of Large Language Models (LLMs), they tend to overlook the nuances of within-language variation and thus fail to model the experience of speakers of non-standard dialects. Focusing on African American Vernacular English (AAVE), we present the first study aimed at objectively assessing the fairness and robustness of LLMs in handling dialects across canonical reasoning tasks, including algorithm, math, logic, and integrated reasoning. We introduce ReDial (Reasoning with Dialect Queries), a benchmark containing 1.2K+ parallel query pairs in Standardized English and AAVE. We hire AAVE speakers, including experts with computer science backgrounds, to rewrite seven popular benchmarks, such as HumanEval and GSM8K. With ReDial, we evaluate widely used LLMs, including GPT, Claude, Llama, Mistral, and the Phi model families. Our findings reveal that almost all of these widely used models show significant brittleness and unfairness to queries in AAVE. Our work establishes a systematic and objective framework for analyzing LLM bias in dialectal queries. Moreover, it highlights how mainstream LLMs provide unfair service to dialect speakers in reasoning tasks, laying a critical foundation for future research.
>
---
#### [replaced 014] AutoPal: Autonomous Adaptation to Users for Personal AI Companionship
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.13960v4](http://arxiv.org/pdf/2406.13960v4)**

> **作者:** Yi Cheng; Wenge Liu; Kaishuai Xu; Wenjun Hou; Yi Ouyang; Chak Tou Leong; Wenjie Li; Xian Wu; Yefeng Zheng
>
> **摘要:** Previous research has demonstrated the potential of AI agents to act as companions that can provide constant emotional support for humans. In this paper, we emphasize the necessity of autonomous adaptation in personal AI companionship, an underexplored yet promising direction. Such adaptability is crucial as it can facilitate more tailored interactions with users and allow the agent to evolve in response to users' changing needs. However, imbuing agents with autonomous adaptability presents unique challenges, including identifying optimal adaptations to meet users' expectations and ensuring a smooth transition during the adaptation process. To address them, we devise a hierarchical framework, AutoPal, that enables controllable and authentic adjustments to the agent's persona based on user interactions. A personamatching dataset is constructed to facilitate the learning of optimal persona adaptations. Extensive experiments demonstrate the effectiveness of AutoPal and highlight the importance of autonomous adaptability in AI companionship.
>
---
#### [replaced 015] sudo rm -rf agentic_security
- **分类: cs.CL; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.20279v3](http://arxiv.org/pdf/2503.20279v3)**

> **作者:** Sejin Lee; Jian Kim; Haon Park; Ashkan Yousefpour; Sangyoon Yu; Min Song
>
> **备注:** Accepted ACL 2025 Industry track
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal-trained safeguards in commercial computer-use agents, such as Claude for Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24.41% (with no refinement), and up to 41.33% (by its iterative refinement) in Claude for Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs
>
---
#### [replaced 016] Selective Prompt Anchoring for Code Generation
- **分类: cs.LG; cs.AI; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2408.09121v5](http://arxiv.org/pdf/2408.09121v5)**

> **作者:** Yuan Tian; Tianyi Zhang
>
> **备注:** Accepted by ICML'25
>
> **摘要:** Recent advances in large language models (LLMs) have transformed software development by automatically generating code from natural language. Yet challenges remain in generating fully correct code that aligns with user intent. Our study reveals that LLMs tend to pay less attention to user prompts as more code tokens are generated. We hypothesize that this attention dilution issue is an important reason for code generation errors. To mitigate this issue, we propose Selective Prompt Anchoring (SPA) to guide code LLMs to pay more attention to user intent when generating code. We evaluate SPA using six base LLMs across six benchmarks. Our results demonstrate that SPA enhances Pass@1 by up to 12.9%, consistently outperforming SOTA code generation methods in all settings. Our code is available at https://github.com/magic-YuanTian/Selective-Prompt-Anchoring.
>
---
#### [replaced 017] Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05850v2](http://arxiv.org/pdf/2506.05850v2)**

> **作者:** Cheonbok Park; Jeonghoon Kim; Joosung Lee; Sanghwan Bae; Jaegul Choo; Kang Min Yoo
>
> **备注:** Preprint
>
> **摘要:** We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning.
>
---
#### [replaced 018] BLEUBERI: BLEU is a surprisingly effective reward for instruction following
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11080v2](http://arxiv.org/pdf/2505.11080v2)**

> **作者:** Yapei Chang; Yekyung Kim; Michael Krumdick; Amir Zadeh; Chuan Li; Chris Tanner; Mohit Iyyer
>
> **备注:** 28 pages, 11 figures, 15 tables; updated table 1 with random reward results, fixed broken references in appendix
>
> **摘要:** Reward models are central to aligning LLMs with human preferences, but they are costly to train, requiring large-scale human-labeled preference data and powerful pretrained LLM backbones. Meanwhile, the increasing availability of high-quality synthetic instruction-following datasets raises the question: can simpler, reference-based metrics serve as viable alternatives to reward models during RL-based alignment? In this paper, we show first that BLEU, a basic string-matching metric, surprisingly matches strong reward models in agreement with human preferences on general instruction-following datasets. Based on this insight, we develop BLEUBERI, a method that first identifies challenging instructions and then applies Group Relative Policy Optimization (GRPO) using BLEU directly as the reward function. We demonstrate that BLEUBERI-trained models are competitive with models trained via reward model-guided RL across four challenging instruction-following benchmarks and three different base language models. A human evaluation further supports that the quality of BLEUBERI model outputs is on par with those from reward model-aligned models. Moreover, BLEUBERI models generate outputs that are more factually grounded than competing methods. Overall, we show that given access to high-quality reference outputs (easily obtained via existing instruction-following datasets or synthetic data generation), string matching-based metrics are cheap yet effective proxies for reward models during alignment. We release our code and data at https://github.com/lilakk/BLEUBERI.
>
---
#### [replaced 019] Structured Pruning for Diverse Best-of-N Reasoning Optimization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03978v2](http://arxiv.org/pdf/2506.03978v2)**

> **作者:** Hieu Trung Nguyen; Bao Nguyen; Viet Anh Nguyen
>
> **备注:** Accepted to ACL Findings 2025
>
> **摘要:** Model pruning in transformer-based language models, traditionally viewed as a means of achieving computational savings, can enhance the model's reasoning capabilities. In this work, we uncover a surprising phenomenon: the selective pruning of certain attention heads leads to improvements in reasoning performance, particularly on challenging tasks. Motivated by this observation, we propose SPRINT, a novel contrastive learning framework that dynamically selects the optimal head and layer to prune during inference. By aligning question embeddings with head embeddings, SPRINT identifies those pruned-head configurations that result in more accurate reasoning. Extensive experiments demonstrate that our method significantly outperforms traditional best-of-$N$ and random head selection strategies on the MATH500 and GSM8K datasets.
>
---
#### [replaced 020] When Incentives Backfire, Data Stops Being Human
- **分类: cs.CY; cs.AI; cs.CL; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07732v2](http://arxiv.org/pdf/2502.07732v2)**

> **作者:** Sebastin Santy; Prasanta Bhattacharya; Manoel Horta Ribeiro; Kelsey Allen; Sewoong Oh
>
> **备注:** Position Paper at ICML 2025
>
> **摘要:** Progress in AI has relied on human-generated data, from annotator marketplaces to the wider Internet. However, the widespread use of large language models now threatens the quality and integrity of human-generated data on these very platforms. We argue that this issue goes beyond the immediate challenge of filtering AI-generated content -- it reveals deeper flaws in how data collection systems are designed. Existing systems often prioritize speed, scale, and efficiency at the cost of intrinsic human motivation, leading to declining engagement and data quality. We propose that rethinking data collection systems to align with contributors' intrinsic motivations -- rather than relying solely on external incentives -- can help sustain high-quality data sourcing at scale while maintaining contributor trust and long-term participation.
>
---
#### [replaced 021] Open-FinLLMs: Open Multimodal Large Language Models for Financial Applications
- **分类: cs.CL; cs.CE; q-fin.CP**

- **链接: [http://arxiv.org/pdf/2408.11878v3](http://arxiv.org/pdf/2408.11878v3)**

> **作者:** Jimin Huang; Mengxi Xiao; Dong Li; Zihao Jiang; Yuzhe Yang; Yifei Zhang; Lingfei Qian; Yan Wang; Xueqing Peng; Yang Ren; Ruoyu Xiang; Zhengyu Chen; Xiao Zhang; Yueru He; Weiguang Han; Shunian Chen; Lihang Shen; Daniel Kim; Yangyang Yu; Yupeng Cao; Zhiyang Deng; Haohang Li; Duanyu Feng; Yongfu Dai; VijayaSai Somasundaram; Peng Lu; Guojun Xiong; Zhiwei Liu; Zheheng Luo; Zhiyuan Yao; Ruey-Ling Weng; Meikang Qiu; Kaleb E Smith; Honghai Yu; Yanzhao Lai; Min Peng; Jian-Yun Nie; Jordan W. Suchow; Xiao-Yang Liu; Benyou Wang; Alejandro Lopez-Lira; Qianqian Xie; Sophia Ananiadou; Junichi Tsujii
>
> **备注:** 33 pages, 13 figures
>
> **摘要:** Financial LLMs hold promise for advancing financial tasks and domain-specific applications. However, they are limited by scarce corpora, weak multimodal capabilities, and narrow evaluations, making them less suited for real-world application. To address this, we introduce \textit{Open-FinLLMs}, the first open-source multimodal financial LLMs designed to handle diverse tasks across text, tabular, time-series, and chart data, excelling in zero-shot, few-shot, and fine-tuning settings. The suite includes FinLLaMA, pre-trained on a comprehensive 52-billion-token corpus; FinLLaMA-Instruct, fine-tuned with 573K financial instructions; and FinLLaVA, enhanced with 1.43M multimodal tuning pairs for strong cross-modal reasoning. We comprehensively evaluate Open-FinLLMs across 14 financial tasks, 30 datasets, and 4 multimodal tasks in zero-shot, few-shot, and supervised fine-tuning settings, introducing two new multimodal evaluation datasets. Our results show that Open-FinLLMs outperforms afvanced financial and general LLMs such as GPT-4, across financial NLP, decision-making, and multi-modal tasks, highlighting their potential to tackle real-world challenges. To foster innovation and collaboration across academia and industry, we release all codes (https://anonymous.4open.science/r/PIXIU2-0D70/B1D7/LICENSE) and models under OSI-approved licenses.
>
---
#### [replaced 022] LLMs Can Simulate Standardized Patients via Agent Coevolution
- **分类: cs.CL; cs.AI; cs.HC; cs.MA**

- **链接: [http://arxiv.org/pdf/2412.11716v2](http://arxiv.org/pdf/2412.11716v2)**

> **作者:** Zhuoyun Du; Lujie Zheng; Renjun Hu; Yuyang Xu; Xiawei Li; Ying Sun; Wei Chen; Jian Wu; Haolei Cai; Haohao Ying
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Training medical personnel using standardized patients (SPs) remains a complex challenge, requiring extensive domain expertise and role-specific practice. Previous research on Large Language Model (LLM)-based SPs mostly focuses on improving data retrieval accuracy or adjusting prompts through human feedback. However, this focus has overlooked the critical need for patient agents to learn a standardized presentation pattern that transforms data into human-like patient responses through unsupervised simulations. To address this gap, we propose EvoPatient, a novel simulated patient framework in which a patient agent and doctor agents simulate the diagnostic process through multi-turn dialogues, simultaneously gathering experience to improve the quality of both questions and answers, ultimately enabling human doctor training. Extensive experiments on various cases demonstrate that, by providing only overall SP requirements, our framework improves over existing reasoning methods by more than 10\% in requirement alignment and better human preference, while achieving an optimal balance of resource consumption after evolving over 200 cases for 10 hours, with excellent generalizability. Our system will be available at https://github.com/ZJUMAI/EvoPatient.
>
---
#### [replaced 023] BiMa: Towards Biases Mitigation for Text-Video Retrieval via Scene Element Guidance
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03589v2](http://arxiv.org/pdf/2506.03589v2)**

> **作者:** Huy Le; Nhat Chung; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** 22 pages, 14 figures
>
> **摘要:** Text-video retrieval (TVR) systems often suffer from visual-linguistic biases present in datasets, which cause pre-trained vision-language models to overlook key details. To address this, we propose BiMa, a novel framework designed to mitigate biases in both visual and textual representations. Our approach begins by generating scene elements that characterize each video by identifying relevant entities/objects and activities. For visual debiasing, we integrate these scene elements into the video embeddings, enhancing them to emphasize fine-grained and salient details. For textual debiasing, we introduce a mechanism to disentangle text features into content and bias components, enabling the model to focus on meaningful content while separately handling biased information. Extensive experiments and ablation studies across five major TVR benchmarks (i.e., MSR-VTT, MSVD, LSMDC, ActivityNet, and DiDeMo) demonstrate the competitive performance of BiMa. Additionally, the model's bias mitigation capability is consistently validated by its strong results on out-of-distribution retrieval tasks.
>
---
#### [replaced 024] SATA-BENCH: Select All That Apply Benchmark for Multiple Choice Questions
- **分类: cs.CL; cs.AI; 68T01; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.00643v2](http://arxiv.org/pdf/2506.00643v2)**

> **作者:** Weijie Xu; Shixian Cui; Xi Fang; Chi Xue; Stephanie Eckman; Chandan K. Reddy
>
> **备注:** 40 pages, 13 figures
>
> **摘要:** Large language models (LLMs) are increasingly evaluated on single-answer multiple-choice tasks, yet many real-world problems require identifying all correct answers from a set of options. This capability remains underexplored. We introduce SATA-BENCH, the first dedicated benchmark for evaluating LLMs on Select All That Apply (SATA) questions across diverse domains, including reading comprehension, law, and biomedicine. Our evaluation of 27 open-source and proprietary models reveals a significant gap: even the strongest model achieves only 41.8% exact match, exposing LLMs' inability to reliably identify all correct answers. We find that this weakness stems from two core challenges: selection bias - models favor certain choices regardless of content, and count bias - models fail to predict the correct number of answers. To address these issues, we propose Choice Funnel, a decoding strategy that combines token debiasing with adaptive thresholding to guide models toward complete and accurate selections. Choice Funnel achieves up to 29% higher exact match than competitive baselines while reducing inference cost by over 64%. Our findings expose fundamental limitations in current LLMs and introduce a new framework for diagnosing and improving multi-answer reasoning. We release SATA-BENCH and Choice Funnel to promote LLM development for robust decision-making in realistic, multi-answer applications.
>
---
#### [replaced 025] SecFormer: Fast and Accurate Privacy-Preserving Inference for Transformer Models via SMPC
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2401.00793v5](http://arxiv.org/pdf/2401.00793v5)**

> **作者:** Jinglong Luo; Yehong Zhang; Zhuo Zhang; Jiaqi Zhang; Xin Mu; Hui Wang; Yue Yu; Zenglin Xu
>
> **备注:** ACL 2024
>
> **摘要:** With the growing use of Transformer models hosted on cloud platforms to offer inference services, privacy concerns are escalating, especially concerning sensitive data like investment plans and bank account details. Secure Multi-Party Computing (SMPC) emerges as a promising solution to protect the privacy of inference data and model parameters. However, the application of SMPC in Privacy-Preserving Inference (PPI) for Transformer models often leads to considerable slowdowns or declines in performance. This is largely due to the multitude of nonlinear operations in the Transformer architecture, which are not well-suited to SMPC and difficult to circumvent or optimize effectively. To address this concern, we introduce a comprehensive PPI framework called SecFormer to achieve fast and accurate PPI for Transformer models. We successfully eliminate the high-cost exponential and maximum operations in PPI without sacrificing model performance and develop a suite of efficient SMPC protocols by employing suitable numerical computation methods to boost other complex nonlinear functions in PPI, including GeLU, LayerNorm, and a redesigned Softmax. Our extensive experiments reveal that SecFormer outperforms MPCFormer in performance, showing improvements of $3.4\%$ and $24.7\%$ for BERT$_{\text{BASE}}$ and BERT$_{\text{LARGE}}$, respectively. In terms of efficiency, SecFormer is 3.57 and 3.58 times faster than PUMA for BERT$_{\text{BASE}}$ and BERT$_{\text{LARGE}}$, demonstrating its effectiveness and speed.
>
---
#### [replaced 026] Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14996v2](http://arxiv.org/pdf/2503.14996v2)**

> **作者:** Francesco Maria Molfese; Luca Moroni; Luca Gioffré; Alessandro Scirè; Simone Conia; Roberto Navigli
>
> **备注:** Findings of the Association for Computational Linguistics ACL 2025
>
> **摘要:** One of the most widely used tasks for evaluating Large Language Models (LLMs) is Multiple-Choice Question Answering (MCQA). While open-ended question answering tasks are more challenging to evaluate, MCQA tasks are, in principle, easier to assess, as the model's answer is thought to be simple to extract and is compared directly to a set of predefined choices. However, recent studies have started to question the reliability of MCQA evaluation, showing that multiple factors can significantly impact the reported performance of LLMs, especially when the model generates free-form text before selecting one of the answer choices. In this work, we shed light on the inconsistencies of MCQA evaluation strategies, which can lead to inaccurate and misleading model comparisons. We systematically analyze whether existing answer extraction methods are aligned with human judgment, and how they are influenced by answer constraints in the prompt across different domains. Our experiments demonstrate that traditional evaluation strategies often underestimate LLM capabilities, while LLM-based answer extractors are prone to systematic errors. Moreover, we reveal a fundamental trade-off between including format constraints in the prompt to simplify answer extraction and allowing models to generate free-form text to improve reasoning. Our findings call for standardized evaluation methodologies and highlight the need for more reliable and consistent MCQA evaluation practices.
>
---
#### [replaced 027] Unraveling Token Prediction Refinement and Identifying Essential Layers in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15054v2](http://arxiv.org/pdf/2501.15054v2)**

> **作者:** Jaturong Kongmanee
>
> **摘要:** This research aims to unravel how large language models (LLMs) iteratively refine token predictions through internal processing. We utilized a logit lens technique to analyze the model's token predictions derived from intermediate representations. Specifically, we focused on (1) how LLMs access and utilize information from input contexts, and (2) how positioning of relevant information affects the model's token prediction refinement process. On a multi-document question answering task with varying input context lengths, we found that the depth of prediction refinement (defined as the number of intermediate layers an LLM uses to transition from an initial correct token prediction to its final, stable correct output), as a function of the position of relevant information, exhibits an approximately inverted U-shaped curve. We also found that the gap between these two layers, on average, diminishes when relevant information is positioned at the beginning or end of the input context. This suggested that the model requires more refinements when processing longer contexts with relevant information situated in the middle. Furthermore, our findings indicate that not all layers are equally essential for determining final correct outputs. Our analysis provides insights into how token predictions are distributed across different conditions, and establishes important connections to existing hypotheses and previous findings in AI safety research and development.
>
---
#### [replaced 028] AfroBench: How Good are Large Language Models on African Languages?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.07978v5](http://arxiv.org/pdf/2311.07978v5)**

> **作者:** Jessica Ojo; Odunayo Ogundepo; Akintunde Oladipo; Kelechi Ogueji; Jimmy Lin; Pontus Stenetorp; David Ifeoluwa Adelani
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** Large-scale multilingual evaluations, such as MEGA, often include only a handful of African languages due to the scarcity of high-quality evaluation data and the limited discoverability of existing African datasets. This lack of representation hinders comprehensive LLM evaluation across a diverse range of languages and tasks. To address these challenges, we introduce AfroBench -- a multi-task benchmark for evaluating the performance of LLMs across 64 African languages, 15 tasks and 22 datasets. AfroBench consists of nine natural language understanding datasets, six text generation datasets, six knowledge and question answering tasks, and one mathematical reasoning task. We present results comparing the performance of prompting LLMs to fine-tuned baselines based on BERT and T5-style models. Our results suggest large gaps in performance between high-resource languages, such as English, and African languages across most tasks; but performance also varies based on the availability of monolingual data resources. Our findings confirm that performance on African languages continues to remain a hurdle for current LLMs, underscoring the need for additional efforts to close this gap. https://mcgill-nlp.github.io/AfroBench/
>
---
#### [replaced 029] ZeroDL: Zero-shot Distribution Learning for Text Clustering via Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.13342v2](http://arxiv.org/pdf/2406.13342v2)**

> **作者:** Hwiyeol Jo; Hyunwoo Lee; Kang Min Yoo; Taiwoo Park
>
> **备注:** Accepted at ACL2025(Findings)
>
> **摘要:** The advancements in large language models (LLMs) have brought significant progress in NLP tasks. However, if a task cannot be fully described in prompts, the models could fail to carry out the task. In this paper, we propose a simple yet effective method to contextualize a task toward a LLM. The method utilizes (1) open-ended zero-shot inference from the entire dataset, (2) aggregate the inference results, and (3) finally incorporate the aggregated meta-information for the actual task. We show the effectiveness in text clustering tasks, empowering LLMs to perform text-to-text-based clustering and leading to improvements on several datasets. Furthermore, we explore the generated class labels for clustering, showing how the LLM understands the task through data.
>
---
#### [replaced 030] Exploring Training and Inference Scaling Laws in Generative Retrieval
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18941v2](http://arxiv.org/pdf/2503.18941v2)**

> **作者:** Hongru Cai; Yongqi Li; Ruifeng Yuan; Wenjie Wang; Zhen Zhang; Wenjie Li; Tat-Seng Chua
>
> **备注:** Accepted to SIGIR 2025
>
> **摘要:** Generative retrieval reformulates retrieval as an autoregressive generation task, where large language models (LLMs) generate target documents directly from a query. As a novel paradigm, the mechanisms that underpin its performance and scalability remain largely unexplored. We systematically investigate training and inference scaling laws in generative retrieval, exploring how model size, training data scale, and inference-time compute jointly influence performance. We propose a novel evaluation metric inspired by contrastive entropy and generation loss, providing a continuous performance signal that enables robust comparisons across diverse generative retrieval methods. Our experiments show that n-gram-based methods align strongly with training and inference scaling laws. We find that increasing model size, training data scale, and inference-time compute all contribute to improved performance, highlighting the complementary roles of these factors in enhancing generative retrieval. Across these settings, LLaMA models consistently outperform T5 models, suggesting a particular advantage for larger decoder-only models in generative retrieval. Our findings underscore that model sizes, data availability, and inference computation interact to unlock the full potential of generative retrieval, offering new insights for designing and optimizing future systems.
>
---
#### [replaced 031] STAMP Your Content: Proving Dataset Membership via Watermarked Rephrasings
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2504.13416v2](http://arxiv.org/pdf/2504.13416v2)**

> **作者:** Saksham Rastogi; Pratyush Maini; Danish Pruthi
>
> **备注:** Published at ICML 25, Code is available at https://github.com/codeboy5/stamp
>
> **摘要:** Given how large parts of publicly available text are crawled to pretrain large language models (LLMs), data creators increasingly worry about the inclusion of their proprietary data for model training without attribution or licensing. Their concerns are also shared by benchmark curators whose test-sets might be compromised. In this paper, we present STAMP, a framework for detecting dataset membership-i.e., determining the inclusion of a dataset in the pretraining corpora of LLMs. Given an original piece of content, our proposal involves first generating multiple rephrases, each embedding a watermark with a unique secret key. One version is to be released publicly, while others are to be kept private. Subsequently, creators can compare model likelihoods between public and private versions using paired statistical tests to prove membership. We show that our framework can successfully detect contamination across four benchmarks which appear only once in the training data and constitute less than 0.001% of the total tokens, outperforming several contamination detection and dataset inference baselines. We verify that STAMP preserves both the semantic meaning and utility of the original data. We apply STAMP to two real-world scenarios to confirm the inclusion of paper abstracts and blog articles in the pretraining corpora.
>
---
#### [replaced 032] Value Residual Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17897v5](http://arxiv.org/pdf/2410.17897v5)**

> **作者:** Zhanchao Zhou; Tianyi Wu; Zhiyun Jiang; Fares Obeid; Zhenzhong Lan
>
> **摘要:** While Transformer models have achieved remarkable success in various domains, the effectiveness of information propagation through deep networks remains a critical challenge. Standard hidden state residuals often fail to adequately preserve initial token-level information in deeper layers. This paper introduces ResFormer, a novel architecture that enhances information flow by incorporating value residual connections in addition to hidden state residuals. And a variant is SVFormer, where all layers share the first layer's value embedding. Comprehensive empirical evidence demonstrates ResFormer achieves equivalent validation loss with 16.11\% fewer model parameters and 20.3\% less training data compared to Transformer, while maintaining similar memory usage and computational cost. Besides, SVFormer reduces KV cache size by nearly half with only a small performance penalty and can be integrated with other KV-efficient methods, yielding further reductions in KV cache, with performance influenced by sequence length and cumulative learning rate.
>
---
#### [replaced 033] AdaSplash: Adaptive Sparse Flash Attention
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12082v2](http://arxiv.org/pdf/2502.12082v2)**

> **作者:** Nuno Gonçalves; Marcos Treviso; André F. T. Martins
>
> **备注:** Accepted as spotlight in ICML 2025
>
> **摘要:** The computational cost of softmax-based attention in transformers limits their applicability to long-context tasks. Adaptive sparsity, of which $\alpha$-entmax attention is an example, offers a flexible data-dependent alternative, but existing implementations are inefficient and do not leverage the sparsity to obtain runtime and memory gains. In this work, we propose AdaSplash, which combines the efficiency of GPU-optimized algorithms with the sparsity benefits of $\alpha$-entmax. We first introduce a hybrid Halley-bisection algorithm, resulting in a 7-fold reduction in the number of iterations needed to compute the $\alpha$-entmax transformation. Then, we implement custom Triton kernels to efficiently handle adaptive sparsity. Experiments with RoBERTa and ModernBERT for text classification and single-vector retrieval, along with GPT-2 for language modeling, show that our method achieves substantial improvements in runtime and memory efficiency compared to existing $\alpha$-entmax implementations. It approaches -- and in some cases surpasses -- the efficiency of highly optimized softmax implementations like FlashAttention-2, enabling long-context training while maintaining strong task performance.
>
---
#### [replaced 034] BRiTE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18858v2](http://arxiv.org/pdf/2501.18858v2)**

> **作者:** Han Zhong; Yutong Yin; Shenao Zhang; Xiaojun Xu; Yuanxin Liu; Yifei Zuo; Zhihan Liu; Boyi Liu; Sirui Zheng; Hongyi Guo; Liwei Wang; Mingyi Hong; Zhaoran Wang
>
> **备注:** ICML 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, yet generating reliable reasoning processes remains a significant challenge. We present a unified probabilistic framework that formalizes LLM reasoning through a novel graphical model incorporating latent thinking processes and evaluation signals. Within this framework, we introduce the Bootstrapping Reinforced Thinking Process (BRiTE) algorithm, which works in two steps. First, it generates high-quality rationales by approximating the optimal thinking process through reinforcement learning, using a novel reward shaping mechanism. Second, it enhances the base LLM by maximizing the joint probability of rationale generation with respect to the model's parameters. Theoretically, we demonstrate BRiTE's convergence at a rate of $1/T$ with $T$ representing the number of iterations. Empirical evaluations on math and coding benchmarks demonstrate that our approach consistently improves performance across different base models without requiring human-annotated thinking processes. In addition, BRiTE demonstrates superior performance compared to existing algorithms that bootstrap thinking processes use alternative methods such as rejection sampling, and can even match or exceed the results achieved through supervised fine-tuning with human-annotated data.
>
---
#### [replaced 035] Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.16853v2](http://arxiv.org/pdf/2503.16853v2)**

> **作者:** Suho Yoo; Hyunjong Ok; Jaeho Lee
>
> **备注:** 12 pages, 5 figures, ACL Findings 2025
>
> **摘要:** Language models pretrained on text-only corpora often struggle with tasks that require auditory commonsense knowledge. Previous work addresses this problem by augmenting the language model to retrieve knowledge from external audio databases. This approach has several limitations, such as the potential lack of relevant audio in databases and the high costs associated with constructing the databases. To address these issues, we propose Imagine to Hear, a novel approach that dynamically generates auditory knowledge using generative models. Our framework detects multiple audio-related textual spans from the given prompt and generates corresponding auditory knowledge. We develop several mechanisms to efficiently process multiple auditory knowledge, including a CLAP-based rejection sampler and a language-audio fusion module. Our experiments show that our method achieves state-of-the-art performance on AuditoryBench without relying on external databases, highlighting the effectiveness of our generation-based approach.
>
---
#### [replaced 036] PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20377v2](http://arxiv.org/pdf/2502.20377v2)**

> **作者:** Albert Gong; Kamilė Stankevičiūtė; Chao Wan; Anmol Kabra; Raphael Thesmar; Johann Lee; Julius Klenke; Carla P. Gomes; Kilian Q. Weinberger
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** High-quality benchmarks are essential for evaluating reasoning and retrieval capabilities of large language models (LLMs). However, curating datasets for this purpose is not a permanent solution as they are prone to data leakage and inflated performance results. To address these challenges, we propose PhantomWiki: a pipeline to generate unique, factually consistent document corpora with diverse question-answer pairs. Unlike prior work, PhantomWiki is neither a fixed dataset, nor is it based on any existing data. Instead, a new PhantomWiki instance is generated on demand for each evaluation. We vary the question difficulty and corpus size to disentangle reasoning and retrieval capabilities respectively, and find that PhantomWiki datasets are surprisingly challenging for frontier LLMs. Thus, we contribute a scalable and data leakage-resistant framework for disentangled evaluation of reasoning, retrieval, and tool-use abilities. Our code is available at https://github.com/kilian-group/phantom-wiki.
>
---
#### [replaced 037] Is poisoning a real threat to LLM alignment? Maybe more so than you think
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2406.12091v4](http://arxiv.org/pdf/2406.12091v4)**

> **作者:** Pankayaraj Pathmanathan; Souradip Chakraborty; Xiangyu Liu; Yongyuan Liang; Furong Huang
>
> **摘要:** Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.
>
---
#### [replaced 038] WorkForceAgent-R1: Incentivizing Reasoning Capability in LLM-based Web Agents via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22942v2](http://arxiv.org/pdf/2505.22942v2)**

> **作者:** Yuchen Zhuang; Di Jin; Jiaao Chen; Wenqi Shi; Hanrui Wang; Chao Zhang
>
> **备注:** Work in Progress
>
> **摘要:** Large language models (LLMs)-empowered web agents enables automating complex, real-time web navigation tasks in enterprise environments. However, existing web agents relying on supervised fine-tuning (SFT) often struggle with generalization and robustness due to insufficient reasoning capabilities when handling the inherently dynamic nature of web interactions. In this study, we introduce WorkForceAgent-R1, an LLM-based web agent trained using a rule-based R1-style reinforcement learning framework designed explicitly to enhance single-step reasoning and planning for business-oriented web navigation tasks. We employ a structured reward function that evaluates both adherence to output formats and correctness of actions, enabling WorkForceAgent-R1 to implicitly learn robust intermediate reasoning without explicit annotations or extensive expert demonstrations. Extensive experiments on the WorkArena benchmark demonstrate that WorkForceAgent-R1 substantially outperforms SFT baselines by 10.26-16.59%, achieving competitive performance relative to proprietary LLM-based agents (gpt-4o) in workplace-oriented web navigation tasks.
>
---
#### [replaced 039] Parameter-Efficient Fine-Tuning of State Space Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.09016v3](http://arxiv.org/pdf/2410.09016v3)**

> **作者:** Kevin Galim; Wonjun Kang; Yuchen Zeng; Hyung Il Koo; Kangwook Lee
>
> **备注:** Accepted at ICML 2025. Code is available at https://github.com/furiosa-ai/ssm-peft
>
> **摘要:** Deep State Space Models (SSMs), such as Mamba (Gu & Dao, 2024), have become powerful tools for language modeling, offering high performance and linear scalability with sequence length. However, the application of parameter-efficient fine-tuning (PEFT) methods to SSM-based models remains largely underexplored. We start by investigating two fundamental questions on existing PEFT methods: (i) How do they perform on SSM-based models? (ii) Which parameters should they target for optimal results? Our analysis shows that LoRA and its variants consistently outperform all other PEFT methods. While LoRA is effective for linear projection matrices, it fails on SSM modules-yet still outperforms other methods applicable to SSMs, indicating their limitations. This underscores the need for a specialized SSM tuning approach. To address this, we propose Sparse Dimension Tuning (SDT), a PEFT method tailored for SSM modules. Combining SDT for SSMs with LoRA for linear projection matrices, we achieve state-of-the-art performance across extensive experiments.
>
---
#### [replaced 040] A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15585v4](http://arxiv.org/pdf/2504.15585v4)**

> **作者:** Kun Wang; Guibin Zhang; Zhenhong Zhou; Jiahao Wu; Miao Yu; Shiqian Zhao; Chenlong Yin; Jinhu Fu; Yibo Yan; Hanjun Luo; Liang Lin; Zhihao Xu; Haolang Lu; Xinye Cao; Xinyun Zhou; Weifei Jin; Fanci Meng; Shicheng Xu; Junyuan Mao; Yu Wang; Hao Wu; Minghe Wang; Fan Zhang; Junfeng Fang; Wenjie Qu; Yue Liu; Chengwei Liu; Yifan Zhang; Qiankun Li; Chongye Guo; Yalan Qin; Zhaoxin Fan; Kai Wang; Yi Ding; Donghai Hong; Jiaming Ji; Yingxin Lai; Zitong Yu; Xinfeng Li; Yifan Jiang; Yanhui Li; Xinyu Deng; Junlin Wu; Dongxia Wang; Yihao Huang; Yufei Guo; Jen-tse Huang; Qiufeng Wang; Xiaolong Jin; Wenxuan Wang; Dongrui Liu; Yanwei Yue; Wenke Huang; Guancheng Wan; Heng Chang; Tianlin Li; Yi Yu; Chenghao Li; Jiawei Li; Lei Bai; Jie Zhang; Qing Guo; Jingyi Wang; Tianlong Chen; Joey Tianyi Zhou; Xiaojun Jia; Weisong Sun; Cong Wu; Jing Chen; Xuming Hu; Yiming Li; Xiao Wang; Ningyu Zhang; Luu Anh Tuan; Guowen Xu; Jiaheng Zhang; Tianwei Zhang; Xingjun Ma; Jindong Gu; Liang Pang; Xiang Wang; Bo An; Jun Sun; Mohit Bansal; Shirui Pan; Lingjuan Lyu; Yuval Elovici; Bhavya Kailkhura; Yaodong Yang; Hongwei Li; Wenyuan Xu; Yizhou Sun; Wei Wang; Qing Li; Ke Tang; Yu-Gang Jiang; Felix Juefei-Xu; Hui Xiong; Xiaofeng Wang; Dacheng Tao; Philip S. Yu; Qingsong Wen; Yang Liu
>
> **摘要:** The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
>
---
#### [replaced 041] APE: Selective Fine-tuning with Acceptance Criteria for Language Model Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19912v2](http://arxiv.org/pdf/2505.19912v2)**

> **作者:** Javier Marín
>
> **摘要:** We present Adjacent Possible Exploration (APE), a selective fine-tuning method for adapting large language models that systematically explores parameter modifications while maintaining model stability. Inspired by evolutionary optimization principles, APE evaluates multiple candidate parameter updates through fine-tuning on small data subsets and accepts only those exceeding a performance threshold. Unlike standard fine-tuning that follows single gradient directions, APE implements a filtered selection process that prevents destabilizing parameter changes while enabling systematic improvement. Our method achieves 33.9\% BLEU improvement and 36.2\% perplexity reduction on news summarization tasks while using minimal computational resources. The approach provides a practical framework for controlled model adaptation that balances performance gains with representational stability.
>
---
#### [replaced 042] Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.00253v3](http://arxiv.org/pdf/2506.00253v3)**

> **作者:** Lihao Sun; Chengzhi Mao; Valentin Hofmann; Xuechunzi Bai
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Although value-aligned language models (LMs) appear unbiased in explicit bias evaluations, they often exhibit stereotypes in implicit word association tasks, raising concerns about their fair usage. We investigate the mechanisms behind this discrepancy and find that alignment surprisingly amplifies implicit bias in model outputs. Specifically, we show that aligned LMs, unlike their unaligned counterparts, overlook racial concepts in early internal representations when the context is ambiguous. Not representing race likely fails to activate safety guardrails, leading to unintended biases. Inspired by this insight, we propose a new bias mitigation strategy that works by incentivizing the representation of racial concepts in the early model layers. In contrast to conventional mitigation methods of machine unlearning, our interventions find that steering the model to be more aware of racial concepts effectively mitigates implicit bias. Similar to race blindness in humans, ignoring racial nuances can inadvertently perpetuate subtle biases in LMs.
>
---
#### [replaced 043] A UD Treebank for Bohairic Coptic
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.18386v2](http://arxiv.org/pdf/2504.18386v2)**

> **作者:** Amir Zeldes; Nina Speransky; Nicholas Wagner; Caroline T. Schroeder
>
> **摘要:** Despite recent advances in digital resources for other Coptic dialects, especially Sahidic, Bohairic Coptic, the main Coptic dialect for pre-Mamluk, late Byzantine Egypt, and the contemporary language of the Coptic Church, remains critically under-resourced. This paper presents and evaluates the first syntactically annotated corpus of Bohairic Coptic, sampling data from a range of works, including Biblical text, saints' lives and Christian ascetic writing. We also explore some of the main differences we observe compared to the existing UD treebank of Sahidic Coptic, the classical dialect of the language, and conduct joint and cross-dialect parsing experiments, revealing the unique nature of Bohairic as a related, but distinct variety from the more often studied Sahidic.
>
---
#### [replaced 044] Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19914v2](http://arxiv.org/pdf/2505.19914v2)**

> **作者:** Jiangjie Chen; Qianyu He; Siyu Yuan; Aili Chen; Zhicheng Cai; Weinan Dai; Hongli Yu; Qiying Yu; Xuefeng Li; Jiaze Chen; Hao Zhou; Mingxuan Wang
>
> **摘要:** Large Language Models (LLMs), such as OpenAI's o1 and DeepSeek's R1, excel at advanced reasoning tasks like math and coding via Reinforcement Learning with Verifiable Rewards (RLVR), but still struggle with puzzles solvable by humans without domain knowledge. We introduce Enigmata, the first comprehensive suite tailored for improving LLMs with puzzle reasoning skills. It includes 36 tasks across seven categories, each with 1) a generator that produces unlimited examples with controllable difficulty and 2) a rule-based verifier for automatic evaluation. This generator-verifier design supports scalable, multi-task RL training, fine-grained analysis, and seamless RLVR integration. We further propose Enigmata-Eval, a rigorous benchmark, and develop optimized multi-task RLVR strategies. Our trained model, Qwen2.5-32B-Enigmata, consistently surpasses o3-mini-high and o1 on the puzzle reasoning benchmarks like Enigmata-Eval, ARC-AGI (32.8%), and ARC-AGI 2 (0.6%). It also generalizes well to out-of-domain puzzle benchmarks and mathematical reasoning, with little multi-tasking trade-off. When trained on larger models like Seed1.5-Thinking (20B activated parameters and 200B total parameters), puzzle data from Enigmata further boosts SoTA performance on advanced math and STEM reasoning tasks such as AIME (2024-2025), BeyondAIME and GPQA (Diamond), showing nice generalization benefits of Enigmata. This work offers a unified, controllable framework for advancing logical reasoning in LLMs. Resources of this work can be found at https://seed-enigmata.github.io.
>
---
#### [replaced 045] Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment
- **分类: cs.CL; cs.CY; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.12452v2](http://arxiv.org/pdf/2505.12452v2)**

> **作者:** Siyang Wu; Honglin Bao; Nadav Kunievsky; James A. Evans
>
> **备注:** We open-source our patent dataset at https://huggingface.co/datasets/UchiKlab/patent_understanding
>
> **摘要:** Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that compared to placebo scientific information, prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized.
>
---
#### [replaced 046] ViMMRC 2.0 -- Enhancing Machine Reading Comprehension on Vietnamese Literature Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2303.18162v2](http://arxiv.org/pdf/2303.18162v2)**

> **作者:** Son T. Luu; Khoi Trong Hoang; Tuong Quang Pham; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **备注:** Accepted for publication at International Journal of Asian Language Processing
>
> **摘要:** Machine reading comprehension has been an interesting and challenging task in recent years, with the purpose of extracting useful information from texts. To attain the computer ability to understand the reading text and answer relevant information, we introduce ViMMRC 2.0 - an extension of the previous ViMMRC for the task of multiple-choice reading comprehension in Vietnamese Textbooks which contain the reading articles for students from Grade 1 to Grade 12. This dataset has 699 reading passages which are prose and poems, and 5,273 questions. The questions in the new dataset are not fixed with four options as in the previous version. Moreover, the difficulty of questions is increased, which challenges the models to find the correct choice. The computer must understand the whole context of the reading passage, the question, and the content of each choice to extract the right answers. Hence, we propose a multi-stage approach that combines the multi-step attention network (MAN) with the natural language inference (NLI) task to enhance the performance of the reading comprehension model. Then, we compare the proposed methodology with the baseline BERTology models on the new dataset and the ViMMRC 1.0. From the results of the error analysis, we found that the challenge of the reading comprehension models is understanding the implicit context in texts and linking them together in order to find the correct answers. Finally, we hope our new dataset will motivate further research to enhance the ability of computers to understand the Vietnamese language.
>
---
#### [replaced 047] HSF: Defending against Jailbreak Attacks with Hidden State Filtering
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.03788v2](http://arxiv.org/pdf/2409.03788v2)**

> **作者:** Cheng Qian; Hainan Zhang; Lei Sha; Zhiming Zheng
>
> **备注:** WWW2025 WSAI BESTPAPER
>
> **摘要:** With the growing deployment of LLMs in daily applications like chatbots and content generation, efforts to ensure outputs align with human values and avoid harmful content have intensified. However, increasingly sophisticated jailbreak attacks threaten this alignment, aiming to induce unsafe outputs. Current defense efforts either focus on prompt rewriting or detection, which are limited in effectiveness due to the various design of jailbreak prompts, or on output control and detection, which are computationally expensive as they require LLM inference. Therefore, designing a pre-inference defense method that resists diverse jailbreak prompts is crucial for preventing LLM jailbreak attacks. We observe that jailbreak attacks, safe queries, and harmful queries exhibit different clustering patterns within the LLM's hidden state representation space. This suggests that by leveraging the LLM's hidden state representational capabilities, we can analyze the LLM's forthcoming behavior and proactively intervene for defense. In this paper, we propose a jailbreak attack defense strategy based on a Hidden State Filter (HSF), a lossless architectural defense mechanism that enables the model to preemptively identify and reject adversarial inputs before the inference process begins. We activate its defensive potential through an additional plugin module, effectively framing the defense task as a classification problem. Experimental results on two benchmark datasets, utilizing three different LLMs, show that HSF significantly enhances resilience against six cutting-edge jailbreak attacks. It significantly reduces the success rate of jailbreak attacks while minimally impacting responses to benign user queries, with negligible inference overhead, and outperforming defense baselines.Our code and data are available at https://anonymous.4open.science/r/Hidden-State-Filtering-8652/
>
---
#### [replaced 048] Context Is Not Comprehension
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04907v2](http://arxiv.org/pdf/2506.04907v2)**

> **作者:** Alex Pan; Mary-Anne Williams
>
> **备注:** 24 pages, 2 figures, 4 tables; to appear in AAAI 2026
>
> **摘要:** The dominant evaluation of Large Language Models has centered on their ability to surface explicit facts from increasingly vast contexts. While today's best models demonstrate near-perfect recall on these tasks, this apparent success masks a fundamental failure in multi-step computation when information is embedded in a narrative. We introduce Verbose ListOps (VLO), a novel benchmark designed to isolate this failure. VLO programmatically weaves deterministic, nested computations into coherent stories, forcing models to track and update internal state rather than simply locate explicit values. Our experiments show that leading LLMs, capable of solving the raw ListOps equations with near-perfect accuracy, collapse in performance on VLO at just 10k tokens. The VLO framework is extensible to any verifiable reasoning task, providing a critical tool to move beyond simply expanding context windows and begin building models with the robust, stateful comprehension required for complex knowledge work.
>
---
#### [replaced 049] When Two LLMs Debate, Both Think They'll Win
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19184v3](http://arxiv.org/pdf/2505.19184v3)**

> **作者:** Pradyumna Shyama Prasad; Minh Nhat Nguyen
>
> **摘要:** Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLMs are now increasingly deployed without careful review in assistant and agentic roles. Code for our experiments is available at https://github.com/pradyuprasad/llms_overconfidence
>
---
#### [replaced 050] from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors
- **分类: cs.CL; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.00038v3](http://arxiv.org/pdf/2503.00038v3)**

> **作者:** Yu Yan; Sheng Sun; Zenghao Duan; Teli Liu; Min Liu; Zhiyi Yin; Jiangyu Lei; Qi Li
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2412.12145
>
> **摘要:** Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.
>
---
#### [replaced 051] On Path to Multimodal Historical Reasoning: HistBench and HistAgent
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20246v2](http://arxiv.org/pdf/2505.20246v2)**

> **作者:** Jiahao Qiu; Fulian Xiao; Yimin Wang; Yuchen Mao; Yijia Chen; Xinzhe Juan; Siran Wang; Xuan Qi; Tongcheng Zhang; Zixin Yao; Jiacheng Guo; Yifu Lu; Charles Argon; Jundi Cui; Daixin Chen; Junran Zhou; Shuyao Zhou; Zhanpeng Zhou; Ling Yang; Shilong Liu; Hongru Wang; Kaixuan Huang; Xun Jiang; Yuming Cao; Yue Chen; Yunfei Chen; Zhengyi Chen; Ruowei Dai; Mengqiu Deng; Jiye Fu; Yunting Gu; Zijie Guan; Zirui Huang; Xiaoyan Ji; Yumeng Jiang; Delong Kong; Haolong Li; Jiaqi Li; Ruipeng Li; Tianze Li; Zhuoran Li; Haixia Lian; Mengyue Lin; Xudong Liu; Jiayi Lu; Jinghan Lu; Wanyu Luo; Ziyue Luo; Zihao Pu; Zhi Qiao; Ruihuan Ren; Liang Wan; Ruixiang Wang; Tianhui Wang; Yang Wang; Zeyu Wang; Zihua Wang; Yujia Wu; Zhaoyi Wu; Hao Xin; Weiao Xing; Ruojun Xiong; Weijie Xu; Yao Shu; Yao Xiao; Xiaorui Yang; Yuchen Yang; Nan Yi; Jiadong Yu; Yangyuxuan Yu; Huiting Zeng; Danni Zhang; Yunjie Zhang; Zhaoyu Zhang; Zhiheng Zhang; Xiaofeng Zheng; Peirong Zhou; Linyan Zhong; Xiaoyin Zong; Ying Zhao; Zhenxin Chen; Lin Ding; Xiaoyu Gao; Bingbing Gong; Yichao Li; Yang Liao; Guang Ma; Tianyuan Ma; Xinrui Sun; Tianyi Wang; Han Xia; Ruobing Xian; Gen Ye; Tengfei Yu; Wentao Zhang; Yuxi Wang; Xi Gao; Mengdi Wang
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Recent advances in large language models (LLMs) have led to remarkable progress across domains, yet their capabilities in the humanities, particularly history, remain underexplored. Historical reasoning poses unique challenges for AI, involving multimodal source interpretation, temporal inference, and cross-linguistic analysis. While general-purpose agents perform well on many existing benchmarks, they lack the domain-specific expertise required to engage with historical materials and questions. To address this gap, we introduce HistBench, a new benchmark of 414 high-quality questions designed to evaluate AI's capacity for historical reasoning and authored by more than 40 expert contributors. The tasks span a wide range of historical problems-from factual retrieval based on primary sources to interpretive analysis of manuscripts and images, to interdisciplinary challenges involving archaeology, linguistics, or cultural history. Furthermore, the benchmark dataset spans 29 ancient and modern languages and covers a wide range of historical periods and world regions. Finding the poor performance of LLMs and other agents on HistBench, we further present HistAgent, a history-specific agent equipped with carefully designed tools for OCR, translation, archival search, and image understanding in History. On HistBench, HistAgent based on GPT-4o achieves an accuracy of 27.54% pass@1 and 36.47% pass@2, significantly outperforming LLMs with online search and generalist agents, including GPT-4o (18.60%), DeepSeek-R1(14.49%) and Open Deep Research-smolagents(20.29% pass@1 and 25.12% pass@2). These results highlight the limitations of existing LLMs and generalist agents and demonstrate the advantages of HistAgent for historical reasoning.
>
---
#### [replaced 052] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.17663v2](http://arxiv.org/pdf/2505.17663v2)**

> **作者:** Yang Xiao; Jiashuo Wang; Qiancheng Xu; Changhe Song; Chunpu Xu; Yi Cheng; Wenjie Li; Pengfei Liu
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** As Large Language Models (LLMs) increasingly participate in human-AI interactions, evaluating their Theory of Mind (ToM) capabilities - particularly their ability to track dynamic mental states - becomes crucial. While existing benchmarks assess basic ToM abilities, they predominantly focus on static snapshots of mental states, overlooking the temporal evolution that characterizes real-world social interactions. We present \textsc{DynToM}, a novel benchmark specifically designed to evaluate LLMs' ability to understand and track the temporal progression of mental states across interconnected scenarios. Through a systematic four-step framework, we generate 1,100 social contexts encompassing 5,500 scenarios and 78,100 questions, each validated for realism and quality. Our comprehensive evaluation of ten state-of-the-art LLMs reveals that their average performance underperforms humans by 44.7\%, with performance degrading significantly when tracking and reasoning about the shift of mental states. This performance gap highlights fundamental limitations in current LLMs' ability to model the dynamic nature of human mental states.
>
---
#### [replaced 053] Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.05752v2](http://arxiv.org/pdf/2501.05752v2)**

> **作者:** Sungjae Lee; Hyejin Park; Jaechang Kim; Jungseul Ok
>
> **摘要:** Recent advancements in large language models (LLMs) have shown remarkable potential in various complex tasks requiring multi-step reasoning methods like tree search to explore diverse reasoning paths. However, existing methods often suffer from computational inefficiency and redundancy. First, they overlook the diversity of task difficulties, leading to unnecessarily extensive searches even for easy tasks. Second, they neglect the semantics of reasoning paths, resulting in redundant exploration of semantically identical paths. To address these limitations, we propose Semantic Exploration with Adaptive Gating (SEAG), a computationally efficient method. SEAG employs an adaptive gating mechanism that dynamically decides whether to conduct a tree search, based on the confidence level of answers from a preceding simple reasoning method. Furthermore, its tree-based exploration consolidates semantically identical reasoning steps, reducing redundant explorations while maintaining or even improving accuracy. Our extensive experiments demonstrate that SEAG significantly improves accuracy by 4.3% on average while requiring only 31% of computational costs compared to existing tree search-based methods on complex reasoning benchmarks including GSM8K and ARC with diverse language models such as Llama2, Llama3, and Mistral. Our code is available at https://github.com/ml-postech/SEAG-semantic-exploration-with-adaptive-gating .
>
---
#### [replaced 054] Retrieval-Augmented Generation as Noisy In-Context Learning: A Unified Theory and Risk Bounds
- **分类: cs.LG; cs.AI; cs.CL; cs.IR; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2506.03100v3](http://arxiv.org/pdf/2506.03100v3)**

> **作者:** Yang Guo; Yutian Tao; Yifei Ming; Robert D. Nowak; Yingyu Liang
>
> **备注:** Under Review
>
> **摘要:** Retrieval-augmented generation (RAG) has seen many empirical successes in recent years by aiding the LLM with external knowledge. However, its theoretical aspect has remained mostly unexplored. In this paper, we propose the first finite-sample generalization bound for RAG in in-context linear regression and derive an exact bias-variance tradeoff. Our framework views the retrieved texts as query-dependent noisy in-context examples and recovers the classical in-context learning (ICL) and standard RAG as the limit cases. Our analysis suggests that an intrinsic ceiling on generalization error exists on RAG as opposed to the ICL. Furthermore, our framework is able to model retrieval both from the training data and from external corpora by introducing uniform and non-uniform RAG noise. In line with our theory, we show the sample efficiency of ICL and RAG empirically with experiments on common QA benchmarks, such as Natural Questions and TriviaQA.
>
---
#### [replaced 055] Dynamic-SUPERB Phase-2: A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spoken Language Models with 180 Tasks
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.05361v2](http://arxiv.org/pdf/2411.05361v2)**

> **作者:** Chien-yu Huang; Wei-Chih Chen; Shu-wen Yang; Andy T. Liu; Chen-An Li; Yu-Xiang Lin; Wei-Cheng Tseng; Anuj Diwan; Yi-Jen Shih; Jiatong Shi; William Chen; Chih-Kai Yang; Wenze Ren; Xuanjun Chen; Chi-Yuan Hsiao; Puyuan Peng; Shih-Heng Wang; Chun-Yi Kuan; Ke-Han Lu; Kai-Wei Chang; Fabian Ritter-Gutierrez; Kuan-Po Huang; Siddhant Arora; You-Kuan Lin; Ming To Chuang; Eunjung Yeo; Kalvin Chang; Chung-Ming Chien; Kwanghee Choi; Jun-You Wang; Cheng-Hsiu Hsieh; Yi-Cheng Lin; Chee-En Yu; I-Hsiang Chiu; Heitor R. Guimarães; Jionghao Han; Tzu-Quan Lin; Tzu-Yuan Lin; Homu Chang; Ting-Wu Chang; Chun Wei Chen; Shou-Jen Chen; Yu-Hua Chen; Hsi-Chun Cheng; Kunal Dhawan; Jia-Lin Fang; Shi-Xin Fang; Kuan-Yu Fang Chiang; Chi An Fu; Hsien-Fu Hsiao; Ching Yu Hsu; Shao-Syuan Huang; Lee Chen Wei; Hsi-Che Lin; Hsuan-Hao Lin; Hsuan-Ting Lin; Jian-Ren Lin; Ting-Chun Liu; Li-Chun Lu; Tsung-Min Pai; Ankita Pasad; Shih-Yun Shan Kuan; Suwon Shon; Yuxun Tang; Yun-Shao Tsai; Jui-Chiang Wei; Tzu-Chieh Wei; Chengxi Wu; Dien-Ruei Wu; Chao-Han Huck Yang; Chieh-Chi Yang; Jia Qi Yip; Shao-Xiang Yuan; Vahid Noroozi; Zhehuai Chen; Haibin Wu; Karen Livescu; David Harwath; Shinji Watanabe; Hung-yi Lee
>
> **备注:** ICLR 2025
>
> **摘要:** Multimodal foundation models, such as Gemini and ChatGPT, have revolutionized human-machine interactions by seamlessly integrating various forms of data. Developing a universal spoken language model that comprehends a wide range of natural language instructions is critical for bridging communication gaps and facilitating more intuitive interactions. However, the absence of a comprehensive evaluation benchmark poses a significant challenge. We present Dynamic-SUPERB Phase-2, an open and evolving benchmark for the comprehensive evaluation of instruction-based universal speech models. Building upon the first generation, this second version incorporates 125 new tasks contributed collaboratively by the global research community, expanding the benchmark to a total of 180 tasks, making it the largest benchmark for speech and audio evaluation. While the first generation of Dynamic-SUPERB was limited to classification tasks, Dynamic-SUPERB Phase-2 broadens its evaluation capabilities by introducing a wide array of novel and diverse tasks, including regression and sequence generation, across speech, music, and environmental audio. Evaluation results show that no model performed well universally. SALMONN-13B excelled in English ASR and Qwen2-Audio-7B-Instruct showed high accuracy in emotion recognition, but current models still require further innovations to handle a broader range of tasks. We open-source all task data and the evaluation pipeline at https://github.com/dynamic-superb/dynamic-superb.
>
---
#### [replaced 056] Zero-Shot Event Causality Identification via Multi-source Evidence Fuzzy Aggregation with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05675v2](http://arxiv.org/pdf/2506.05675v2)**

> **作者:** Zefan Zeng; Xingchen Hu; Qing Cheng; Weiping Ding; Wentao Li; Zhong Liu
>
> **摘要:** Event Causality Identification (ECI) aims to detect causal relationships between events in textual contexts. Existing ECI models predominantly rely on supervised methodologies, suffering from dependence on large-scale annotated data. Although Large Language Models (LLMs) enable zero-shot ECI, they are prone to causal hallucination-erroneously establishing spurious causal links. To address these challenges, we propose MEFA, a novel zero-shot framework based on Multi-source Evidence Fuzzy Aggregation. First, we decompose causality reasoning into three main tasks (temporality determination, necessity analysis, and sufficiency verification) complemented by three auxiliary tasks. Second, leveraging meticulously designed prompts, we guide LLMs to generate uncertain responses and deterministic outputs. Finally, we quantify LLM's responses of sub-tasks and employ fuzzy aggregation to integrate these evidence for causality scoring and causality determination. Extensive experiments on three benchmarks demonstrate that MEFA outperforms second-best unsupervised baselines by 6.2% in F1-score and 9.3% in precision, while significantly reducing hallucination-induced errors. In-depth analysis verify the effectiveness of task decomposition and the superiority of fuzzy aggregation.
>
---
#### [replaced 057] Latent Thought Models with Variational Bayes Inference-Time Computation
- **分类: cs.CL; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.01567v2](http://arxiv.org/pdf/2502.01567v2)**

> **作者:** Deqian Kong; Minglu Zhao; Dehong Xu; Bo Pang; Shu Wang; Edouardo Honig; Zhangzhang Si; Chuan Li; Jianwen Xie; Sirui Xie; Ying Nian Wu
>
> **摘要:** We propose a novel class of language models, Latent Thought Models (LTMs), which incorporate explicit latent thought vectors that follow an explicit prior model in latent space. These latent thought vectors guide the autoregressive generation of ground tokens through a Transformer decoder. Training employs a dual-rate optimization process within the classical variational Bayes framework: fast learning of local variational parameters for the posterior distribution of latent vectors (inference-time computation), and slow learning of global decoder parameters. Empirical studies reveal that LTMs possess additional scaling dimensions beyond traditional Large Language Models (LLMs), such as the number of iterations in inference-time computation and number of latent thought vectors. Higher sample efficiency can be achieved by increasing training compute per token, with further gains possible by trading model size for more inference steps. Designed based on these scaling properties, LTMs demonstrate superior sample and parameter efficiency compared to autoregressive models and discrete diffusion models. They significantly outperform these counterparts in validation perplexity and zero-shot language modeling tasks. Additionally, LTMs exhibit emergent few-shot in-context reasoning capabilities that scale with model size, and achieve competitive performance in conditional and unconditional text generation.
>
---
#### [replaced 058] Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.15021v2](http://arxiv.org/pdf/2410.15021v2)**

> **作者:** Hidetaka Kamigaito; Hiroyuki Deguchi; Yusuke Sakai; Katsuhiko Hayashi; Taro Watanabe
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** Inference methods play an important role in eliciting the performance of large language models (LLMs). Currently, LLMs use inference methods utilizing generated multiple samples, which can be derived from Minimum Bayes Risk (MBR) Decoding. Previous studies have conducted empirical analyses to clarify the improvements in generation performance achieved by MBR decoding and have reported various observations. However, the theoretical underpinnings of these findings remain uncertain. To address this, we offer a new theoretical interpretation of MBR decoding from the perspective of bias-diversity decomposition. In this interpretation, the error in the quality estimation of hypotheses by MBR decoding is decomposed into two main factors: bias, which considers the closeness between the utility function and human evaluation, and diversity, which represents the variability in the quality estimation of the utility function. The theoretical analysis reveals the difficulty of simultaneously improving bias and diversity, confirming the validity of enhancing MBR decoding performance by increasing diversity. Furthermore, we reveal that diversity can explain one aspect of inference scaling laws that describe performance improvement by increasing sample size. Moreover, experiments across multiple NLP tasks yielded results consistent with these theoretical characteristics. Our code is available at https://github.com/naist-nlp/mbr-bias-diversity.
>
---
#### [replaced 059] SyncMind: Measuring Agent Out-of-Sync Recovery in Collaborative Software Engineering
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06994v2](http://arxiv.org/pdf/2502.06994v2)**

> **作者:** Xuehang Guo; Xingyao Wang; Yangyi Chen; Sha Li; Chi Han; Manling Li; Heng Ji
>
> **摘要:** Software engineering (SE) is increasingly collaborative, with developers working together on shared complex codebases. Effective collaboration in shared environments requires participants -- whether humans or AI agents -- to stay on the same page as their environment evolves. When a collaborator's understanding diverges from the current state -- what we term the out-of-sync challenge -- the collaborator's actions may fail, leading to integration issues. In this work, we introduce SyncMind, a framework that systematically defines the out-of-sync problem faced by large language model (LLM) agents in collaborative software engineering (CSE). Based on SyncMind, we create SyncBench, a benchmark featuring 24,332 instances of agent out-of-sync scenarios in real-world CSE derived from 21 popular GitHub repositories with executable verification tests. Experiments on SyncBench uncover critical insights into existing LLM agents' capabilities and limitations. Besides substantial performance gaps among agents (from Llama-3.1 agent <= 3.33% to Claude-3.5-Sonnet >= 28.18%), their consistently low collaboration willingness (<= 4.86%) suggests fundamental limitations of existing LLM in CSE. However, when collaboration occurs, it positively correlates with out-of-sync recovery success. Minimal performance differences in agents' resource-aware out-of-sync recoveries further reveal their significant lack of resource awareness and adaptability, shedding light on future resource-efficient collaborative systems. Code and data are openly available on our project website: https://xhguo7.github.io/SyncMind/.
>
---
#### [replaced 060] Power-Law Decay Loss for Large Language Model Finetuning: A Theory Perspective
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16900v5](http://arxiv.org/pdf/2505.16900v5)**

> **作者:** Jintian Shao
>
> **备注:** Short of sufficient experiments and references
>
> **摘要:** During the finetuning stage of text generation tasks, standard cross-entropy loss treats all tokens equally. This can lead models to overemphasize high-frequency, low-information tokens, neglecting lower-frequency tokens crucial for specificity and informativeness in generated content. This paper introduces a novel loss function, Power-Law Decay Loss (PDL), specifically designed to optimize the finetuning process for text generation. The core motivation for PDL stems from observations in information theory and linguistics: the informativeness of a token is often inversely proportional to its frequency of occurrence. PDL re-weights the contribution of each token in the standard cross-entropy loss based on its frequency in the training corpus, following a power-law decay. Specifically, the weights for high-frequency tokens are reduced, while low-frequency, information-dense tokens are assigned higher weights. This mechanism guides the model during finetuning to focus more on learning and generating tokens that convey specific and unique information, thereby enhancing the quality, diversity, and informativeness of the generated text. We theoretically elaborate on the motivation and construction of PDL and discuss its potential applications and advantages across various text generation finetuning tasks, such as abstractive summarization, dialogue systems, and style transfer.
>
---
#### [replaced 061] Retrieval-augmented Large Language Models for Financial Time Series Forecasting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05878v3](http://arxiv.org/pdf/2502.05878v3)**

> **作者:** Mengxi Xiao; Zihao Jiang; Lingfei Qian; Zhengyu Chen; Yueru He; Yijing Xu; Yuecheng Jiang; Dong Li; Ruey-Ling Weng; Min Peng; Jimin Huang; Sophia Ananiadou; Qianqian Xie
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Accurately forecasting stock price movements is critical for informed financial decision-making, supporting applications ranging from algorithmic trading to risk management. However, this task remains challenging due to the difficulty of retrieving subtle yet high-impact patterns from noisy financial time-series data, where conventional retrieval methods, whether based on generic language models or simplistic numeric similarity, often fail to capture the intricate temporal dependencies and context-specific signals essential for precise market prediction. To bridge this gap, we introduce FinSrag, the first retrieval-augmented generation (RAG) framework with a novel domain-specific retriever FinSeer for financial time-series forecasting. FinSeer leverages a candidate selection mechanism refined by LLM feedback and a similarity-driven training objective to align queries with historically influential sequences while filtering out financial noise. Such training enables FinSeer to identify the most relevant time-series data segments for downstream forecasting tasks, unlike embedding or distance-based retrieval methods used in existing RAG frameworks. The retrieved patterns are then fed into StockLLM, a 1B-parameter LLM fine-tuned for stock movement prediction, which serves as the generative backbone. Beyond the retrieval method, we enrich the retrieval corpus by curating new datasets that integrate a broader set of financial indicators, capturing previously overlooked market dynamics. Experiments demonstrate that FinSeer outperforms existing textual retrievers and traditional distance-based retrieval approaches in enhancing the prediction accuracy of StockLLM, underscoring the importance of domain-specific retrieval frameworks in handling the complexity of financial time-series data.
>
---
#### [replaced 062] Scaling Inference-Efficient Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18107v2](http://arxiv.org/pdf/2501.18107v2)**

> **作者:** Song Bian; Minghao Yan; Shivaram Venkataraman
>
> **备注:** 21 pages, 18 figures, ICML 2025
>
> **摘要:** Scaling laws are powerful tools to predict the performance of large language models. However, current scaling laws fall short of accounting for inference costs. In this work, we first show that model architecture affects inference latency, where models of the same size can have up to 3.5x difference in latency. To tackle this challenge, we modify the Chinchilla scaling laws to co-optimize the model parameter count, the number of training tokens, and the model architecture. Due to the reason that models of similar training loss exhibit gaps in downstream evaluation, we also propose a novel method to train inference-efficient models based on the revised scaling laws. We perform extensive empirical studies to fit and evaluate our inference-aware scaling laws. We vary model parameters from 80M to 1B, training tokens from 1.6B to 30B, and model shapes, training 63 models. Guided by our inference-efficient scaling law and model selection method, we release the Morph-1B model, which improves inference latency by 1.8x while maintaining accuracy on downstream tasks compared to open-source models, pushing the Pareto frontier of accuracy-latency tradeoff. Notably, our experiments reveal that wider and shallower models can yield efficiency gains while preserving accuracy.
>
---
#### [replaced 063] Knowledge-to-Jailbreak: Investigating Knowledge-driven Jailbreaking Attacks for Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2406.11682v2](http://arxiv.org/pdf/2406.11682v2)**

> **作者:** Shangqing Tu; Zhuoran Pan; Wenxuan Wang; Zhexin Zhang; Yuliang Sun; Jifan Yu; Hongning Wang; Lei Hou; Juanzi Li
>
> **备注:** Accepted by KDD 2025 research track
>
> **摘要:** Large language models (LLMs) have been increasingly applied to various domains, which triggers increasing concerns about LLMs' safety on specialized domains, e.g. medicine. Despite prior explorations on general jailbreaking attacks, there are two challenges for applying existing attacks on testing the domain-specific safety of LLMs: (1) Lack of professional knowledge-driven attacks, (2) Insufficient coverage of domain knowledge. To bridge this gap, we propose a new task, knowledge-to-jailbreak, which aims to generate jailbreaking attacks from domain knowledge, requiring both attack effectiveness and knowledge relevance. We collect a large-scale dataset with 12,974 knowledge-jailbreak pairs and fine-tune a large language model as jailbreak-generator, to produce domain knowledge-specific jailbreaks. Experiments on 13 domains and 8 target LLMs demonstrate the effectiveness of jailbreak-generator in generating jailbreaks that are both threatening to the target LLMs and relevant to the given knowledge. We also apply our method to an out-of-domain knowledge base, showing that jailbreak-generator can generate jailbreaks that are comparable in harmfulness to those crafted by human experts. Data and code are available at: https://github.com/THU-KEG/Knowledge-to-Jailbreak/.
>
---
#### [replaced 064] Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03038v2](http://arxiv.org/pdf/2506.03038v2)**

> **作者:** Jintian Shao; Yiming Cheng
>
> **备注:** Bad experiments, lacking sufficient references
>
> **摘要:** Reinforcement learning (RL) enhances large language models (LLMs) in complex, long-chain-of-thought (long-CoT) reasoning. The advanced VAPO framework, despite sophisticated mechanisms like Decoupled GAE, theoretically faces fundamental limitations in comprehensively modeling and leveraging deep, long-term value for fine-grained, step-by-step policy guidance in extended reasoning chains. We argue these limitations stem from inherent difficulties in credit assignment, value function representational capacity with temporally abstracted goals, and translating global value signals into local policy improvements, especially with sparse rewards. Our theoretical analysis examines these aspects to illuminate VAPO's boundaries in long-term value modeling, aiming to deepen understanding of current RL for advanced reasoning and suggest future research for more robust LLM agents.
>
---
#### [replaced 065] Unveiling and Addressing Pseudo Forgetting in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.11932v2](http://arxiv.org/pdf/2411.11932v2)**

> **作者:** Huashan Sun; Yizhe Yang; Yinghao Li; Jiawei Li; Yang Gao
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Although substantial efforts have been made to mitigate catastrophic forgetting in continual learning, the intrinsic mechanisms are not well understood. In this work, we demonstrate the existence of "pseudo forgetting": the performance degradation on previous tasks is not attributed to a loss of capabilities, but rather to the failure of the instructions to activate the appropriate model abilities. We show that the model's performance on previous tasks can be restored through two simple interventions: (1) providing partial external correct rationale, and (2) appending semantically meaningless suffixes to the original instructions, to guide the generation of correct rationales. Through empirical analysis of the internal mechanisms governing rationale generation, we reveal that models exhibiting pseudo forgetting show reduced instruction dependence during rationale generation, leading to suboptimal activation of their inherent capabilities. Based on this insight, we propose Rationale-Guidance Difficulty based Replay (RGD-R) framework that dynamically allocates replay data based on the model's ability to correctly leverage the intrinsic capabilities. Experimental results demonstrate that RGD-R effectively mitigates pseudo forgetting while maintaining model plasticity.
>
---
#### [replaced 066] Watermarking Language Models with Error Correcting Codes
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.10281v4](http://arxiv.org/pdf/2406.10281v4)**

> **作者:** Patrick Chao; Yan Sun; Edgar Dobriban; Hamed Hassani
>
> **摘要:** Recent progress in large language models enables the creation of realistic machine-generated content. Watermarking is a promising approach to distinguish machine-generated text from human text, embedding statistical signals in the output that are ideally undetectable to humans. We propose a watermarking framework that encodes such signals through an error correcting code. Our method, termed robust binary code (RBC) watermark, introduces no noticeable degradation in quality. We evaluate our watermark on base and instruction fine-tuned models and find that our watermark is robust to edits, deletions, and translations. We provide an information-theoretic perspective on watermarking, a powerful statistical test for detection and for generating $p$-values, and theoretical guarantees. Our empirical findings suggest our watermark is fast, powerful, and robust, comparing favorably to the state-of-the-art.
>
---
#### [replaced 067] DISC: Plug-and-Play Decoding Intervention with Similarity of Characters for Chinese Spelling Check
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12863v2](http://arxiv.org/pdf/2412.12863v2)**

> **作者:** Ziheng Qiao; Houquan Zhou; Yumeng Liu; Zhenghua Li; Min Zhang; Bo Zhang; Chen Li; Ji Zhang; Fei Huang
>
> **摘要:** One key characteristic of the Chinese spelling check (CSC) task is that incorrect characters are usually similar to the correct ones in either phonetics or glyph. To accommodate this, previous works usually leverage confusion sets, which suffer from two problems, i.e., difficulty in determining which character pairs to include and lack of probabilities to distinguish items in the set. In this paper, we propose a light-weight plug-and-play DISC (i.e., decoding intervention with similarity of characters) module for CSC models.DISC measures phonetic and glyph similarities between characters and incorporates this similarity information only during the inference phase. This method can be easily integrated into various existing CSC models, such as ReaLiSe, SCOPE, and ReLM, without additional training costs. Experiments on three CSC benchmarks demonstrate that our proposed method significantly improves model performance, approaching and even surpassing the current state-of-the-art models.
>
---
#### [replaced 068] Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11494v2](http://arxiv.org/pdf/2502.11494v2)**

> **作者:** Zichen Wen; Yifeng Gao; Shaobo Wang; Junyuan Zhang; Qintong Zhang; Weijia Li; Conghui He; Linfeng Zhang
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Vision tokens in multimodal large language models often dominate huge computational overhead due to their excessive length compared to linguistic modality. Abundant recent methods aim to solve this problem with token pruning, which first defines an importance criterion for tokens and then prunes the unimportant vision tokens during inference. However, in this paper, we show that the importance is not an ideal indicator to decide whether a token should be pruned. Surprisingly, it usually results in inferior performance than random token pruning and leading to incompatibility to efficient attention computation operators.Instead, we propose DART (Duplication-Aware Reduction of Tokens), which prunes tokens based on its duplication with other tokens, leading to significant and training-free acceleration. Concretely, DART selects a small subset of pivot tokens and then retains the tokens with low duplication to the pivots, ensuring minimal information loss during token pruning. Experiments demonstrate that DART can prune 88.9% vision tokens while maintaining comparable performance, leading to a 1.99$\times$ and 2.99$\times$ speed-up in total time and prefilling stage, respectively, with good compatibility to efficient attention operators. Our codes are available at https://github.com/ZichenWen1/DART.
>
---
#### [replaced 069] Epistemic Integrity in Large Language Models
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2411.06528v2](http://arxiv.org/pdf/2411.06528v2)**

> **作者:** Bijean Ghafouri; Shahrad Mohammadzadeh; James Zhou; Pratheeksha Nair; Jacob-Junqi Tian; Hikaru Tsujimura; Mayank Goel; Sukanya Krishna; Reihaneh Rabbany; Jean-François Godbout; Kellin Pelrine
>
> **摘要:** Large language models are increasingly relied upon as sources of information, but their propensity for generating false or misleading statements with high confidence poses risks for users and society. In this paper, we confront the critical problem of epistemic miscalibration $\unicode{x2013}$ where a model's linguistic assertiveness fails to reflect its true internal certainty. We introduce a new human-labeled dataset and a novel method for measuring the linguistic assertiveness of Large Language Models (LLMs) which cuts error rates by over 50% relative to previous benchmarks. Validated across multiple datasets, our method reveals a stark misalignment between how confidently models linguistically present information and their actual accuracy. Further human evaluations confirm the severity of this miscalibration. This evidence underscores the urgent risk of the overstated certainty LLMs hold which may mislead users on a massive scale. Our framework provides a crucial step forward in diagnosing this miscalibration, offering a path towards correcting it and more trustworthy AI across domains.
>
---
#### [replaced 070] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13738v2](http://arxiv.org/pdf/2502.13738v2)**

> **作者:** Keqin Peng; Liang Ding; Yuanxin Ouyang; Meng Fang; Yancheng Yuan; Dacheng Tao
>
> **备注:** ACL2025
>
> **摘要:** Large language models (LLMs) excel at a range of tasks through in-context learning (ICL), where only a few task examples guide their predictions. However, prior research highlights that LLMs often overlook input-label mapping information in ICL, relying more on their pre-trained knowledge. To address this issue, we introduce In-Context Contrastive Decoding (ICCD), a novel method that emphasizes input-label mapping by contrasting the output distributions between positive and negative in-context examples. Experiments on 7 natural language understanding (NLU) tasks show that our ICCD method brings consistent and significant improvement (up to +1.8 improvement on average) upon 6 different scales of LLMs without requiring additional training. Our approach is versatile, enhancing performance with various demonstration selection methods, demonstrating its broad applicability and effectiveness. The code and scripts are released at https://github.com/Romainpkq/CD_ICL.
>
---
#### [replaced 071] Scalable Vision Language Model Training via High Quality Data Curation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.05952v3](http://arxiv.org/pdf/2501.05952v3)**

> **作者:** Hongyuan Dong; Zijian Kang; Weijie Yin; Xiao Liang; Chao Feng; Jiao Ran
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** In this paper, we introduce SAIL-VL (ScAlable Vision Language Model TraIning via High QuaLity Data Curation), an open-source vision language model (VLM) series achieving state-of-the-art (SOTA) performance in 2B and 8B parameters. The following three key improvements contribute to SAIL-VL's leading performance: (1) Scalable high-quality visual understanding data construction: We implement a data construction pipeline to enable hundred-million-scale high-quality recaption data annotation. The resulted dataset SAIL-Caption is validated to be of the highest data quality compared with opensource datasets. (2) Scalable Pretraining with High-Quality Visual Understanding Data: We scale SAIL-VL's pretraining budget up to 655B tokens and show that even a 2B VLM benefits from scaled up training data sizes, exhibiting logarithmic data size scaling laws in benchmark performance. (3) Scalable SFT via data quantity and complexity scaling: We curate a high-quality SFT dataset collection with leading data quantity scaling effectiveness and demonstrate that training with progressively higher-complexity data surpasses baseline one-stage training by a large margin. SAIL-VL series models achieve the highest average score in 18 widely used VLM benchmarks in our evaluation, with the 2B model takes the top position over VLMs of comparable sizes on OpenCompass 2024 (https://rank.opencompass.org.cn/leaderboard-multimodal), demonstrating robust visual comprehension abilities. SAIL-VL series models are released at HuggingFace (https://huggingface.co/BytedanceDouyinContent).
>
---
#### [replaced 072] HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.20811v2](http://arxiv.org/pdf/2502.20811v2)**

> **作者:** Xiao Wang; Jingyun Hua; Weihong Lin; Yuanxing Zhang; Fuzheng Zhang; Jianlong Wu; Di Zhang; Liqiang Nie
>
> **摘要:** Recent Multi-modal Large Language Models (MLLMs) have made great progress in video understanding. However, their performance on videos involving human actions is still limited by the lack of high-quality data. To address this, we introduce a two-stage data annotation pipeline. First, we design strategies to accumulate videos featuring clear human actions from the Internet. Second, videos are annotated in a standardized caption format that uses human attributes to distinguish individuals and chronologically details their actions and interactions. Through this pipeline, we curate two datasets, namely HAICTrain and HAICBench. \textbf{HAICTrain} comprises 126K video-caption pairs generated by Gemini-Pro and verified for training purposes. Meanwhile, \textbf{HAICBench} includes 412 manually annotated video-caption pairs and 2,000 QA pairs, for a comprehensive evaluation of human action understanding. Experimental results demonstrate that training with HAICTrain not only significantly enhances human understanding abilities across 4 benchmarks, but can also improve text-to-video generation results. Both the HAICTrain and HAICBench are released at https://huggingface.co/datasets/KuaishouHAIC/HAIC.
>
---
#### [replaced 073] DeepRAG: Thinking to Retrieve Step by Step for Large Language Models
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.01142v2](http://arxiv.org/pdf/2502.01142v2)**

> **作者:** Xinyan Guan; Jiali Zeng; Fandong Meng; Chunlei Xin; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun; Jie Zhou
>
> **摘要:** Large Language Models (LLMs) have shown remarkable reasoning capabilities, while their practical applications are limited by severe factual hallucinations due to limitations in the timeliness, accuracy, and comprehensiveness of their parametric knowledge. Meanwhile, enhancing retrieval-augmented generation (RAG) with reasoning remains challenging due to ineffective task decomposition and redundant retrieval, which can introduce noise and degrade response quality. In this paper, we propose DeepRAG, a framework that models retrieval-augmented reasoning as a Markov Decision Process (MDP), enabling reasonable and adaptive retrieval. By iteratively decomposing queries, DeepRAG dynamically determines whether to retrieve external knowledge or rely on parametric reasoning at each step. Experiments show that DeepRAG improves retrieval efficiency and boosts answer accuracy by 26.4%, demonstrating its effectiveness in enhancing retrieval-augmented reasoning.
>
---
#### [replaced 074] Can Perplexity Predict Fine-tuning Performance? An Investigation of Tokenization Effects on Sequential Language Models for Nepali
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.18071v2](http://arxiv.org/pdf/2404.18071v2)**

> **作者:** Nishant Luitel; Nirajan Bekoju; Anand Kumar Sah; Subarna Shakya
>
> **备注:** 11 pages
>
> **摘要:** The impact of subword tokenization on language model performance is well-documented for perplexity, with finer granularity consistently reducing this intrinsic metric. However, research on how different tokenization schemes affect a model's understanding capabilities remains limited, particularly for non-Latin script languages. Addressing this gap, we conducted a comprehensive evaluation of six distinct tokenization strategies by pretraining transformer-based language models for Nepali and evaluating their performance across multiple downstream tasks. While recent prominent models like GPT, RoBERTa, Claude, LLaMA, Mistral, Falcon, and MPT have adopted byte-level BPE tokenization, our findings demonstrate that for Nepali, SentencePiece tokenization consistently yields superior results on understanding-based tasks. Unlike previous studies that primarily focused on BERT-based architectures, our research specifically examines sequential transformer models, providing valuable insights for language model development in low-resource languages and highlighting the importance of tokenization strategy beyond perplexity reduction.
>
---
#### [replaced 075] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.17679v5](http://arxiv.org/pdf/2411.17679v5)**

> **作者:** Zhu Xu; Zhiqiang Zhao; Zihan Zhang; Yuchi Liu; Quanwei Shen; Fei Liu; Yu Kuang; Jian He; Conglin Liu
>
> **备注:** ACL 2025 Main
>
> **摘要:** Tokenization methods like Byte-Pair Encoding (BPE) enhance computational efficiency in large language models (LLMs) but often obscure internal character structures within tokens. This limitation hinders LLMs' ability to predict precise character positions, which is crucial in tasks like Chinese Spelling Correction (CSC) where identifying the positions of misspelled characters accelerates correction processes. We propose Token Internal Position Awareness (TIPA), a method that significantly improves models' ability to capture character positions within tokens by training them on reverse character prediction tasks using the tokenizer's vocabulary. Experiments demonstrate that TIPA enhances position prediction accuracy in LLMs, enabling more precise identification of target characters in original text. Furthermore, when applied to downstream tasks that do not require exact position prediction, TIPA still boosts performance in tasks needing character-level information, validating its versatility and effectiveness.
>
---
#### [replaced 076] Minerva: A Programmable Memory Test Benchmark for Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03358v2](http://arxiv.org/pdf/2502.03358v2)**

> **作者:** Menglin Xia; Victor Ruehle; Saravan Rajmohan; Reza Shokri
>
> **备注:** ICML 2025
>
> **摘要:** How effectively can LLM-based AI assistants utilize their memory (context) to perform various tasks? Traditional data benchmarks, which are often manually crafted, suffer from several limitations: they are static, susceptible to overfitting, difficult to interpret, and lack actionable insights--failing to pinpoint the specific capabilities a model lacks when it does not pass a test. In this paper, we present a framework for automatically generating a comprehensive set of tests to evaluate models' abilities to use their memory effectively. Our framework extends the range of capability tests beyond the commonly explored (passkey, key-value, needle in the haystack) search, a dominant focus in the literature. Specifically, we evaluate models on atomic tasks such as searching, recalling, editing, matching, comparing information in context memory, performing basic operations when inputs are structured into distinct blocks, and maintaining state while operating on memory, simulating real-world data. Additionally, we design composite tests to investigate the models' ability to perform more complex, integrated tasks. Our benchmark enables an interpretable, detailed assessment of memory capabilities of LLMs.
>
---
#### [replaced 077] MMTEB: Massive Multilingual Text Embedding Benchmark
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.13595v3](http://arxiv.org/pdf/2502.13595v3)**

> **作者:** Kenneth Enevoldsen; Isaac Chung; Imene Kerboua; Márton Kardos; Ashwin Mathur; David Stap; Jay Gala; Wissam Siblini; Dominik Krzemiński; Genta Indra Winata; Saba Sturua; Saiteja Utpala; Mathieu Ciancone; Marion Schaeffer; Gabriel Sequeira; Diganta Misra; Shreeya Dhakal; Jonathan Rystrøm; Roman Solomatin; Ömer Çağatan; Akash Kundu; Martin Bernstorff; Shitao Xiao; Akshita Sukhlecha; Bhavish Pahwa; Rafał Poświata; Kranthi Kiran GV; Shawon Ashraf; Daniel Auras; Björn Plüster; Jan Philipp Harries; Loïc Magne; Isabelle Mohr; Mariya Hendriksen; Dawei Zhu; Hippolyte Gisserot-Boukhlef; Tom Aarsen; Jan Kostkan; Konrad Wojtasik; Taemin Lee; Marek Šuppa; Crystina Zhang; Roberta Rocca; Mohammed Hamdy; Andrianos Michail; John Yang; Manuel Faysse; Aleksei Vatolin; Nandan Thakur; Manan Dey; Dipam Vasani; Pranjal Chitale; Simone Tedeschi; Nguyen Tai; Artem Snegirev; Michael Günther; Mengzhou Xia; Weijia Shi; Xing Han Lù; Jordan Clive; Gayatri Krishnakumar; Anna Maksimova; Silvan Wehrli; Maria Tikhonova; Henil Panchal; Aleksandr Abramov; Malte Ostendorff; Zheng Liu; Simon Clematide; Lester James Miranda; Alena Fenogenova; Guangyu Song; Ruqiya Bin Safi; Wen-Ding Li; Alessia Borghini; Federico Cassano; Hongjin Su; Jimmy Lin; Howard Yen; Lasse Hansen; Sara Hooker; Chenghao Xiao; Vaibhav Adlakha; Orion Weller; Siva Reddy; Niklas Muennighoff
>
> **备注:** Accepted for ICLR: https://openreview.net/forum?id=zl3pfz4VCV
>
> **摘要:** Text embeddings are typically evaluated on a limited set of tasks, which are constrained by language, domain, and task diversity. To address these limitations and provide a more comprehensive evaluation, we introduce the Massive Multilingual Text Embedding Benchmark (MMTEB) - a large-scale, community-driven expansion of MTEB, covering over 500 quality-controlled evaluation tasks across 250+ languages. MMTEB includes a diverse set of challenging, novel tasks such as instruction following, long-document retrieval, and code retrieval, representing the largest multilingual collection of evaluation tasks for embedding models to date. Using this collection, we develop several highly multilingual benchmarks, which we use to evaluate a representative set of models. We find that while large language models (LLMs) with billions of parameters can achieve state-of-the-art performance on certain language subsets and task categories, the best-performing publicly available model is multilingual-e5-large-instruct with only 560 million parameters. To facilitate accessibility and reduce computational cost, we introduce a novel downsampling method based on inter-task correlation, ensuring a diverse selection while preserving relative model rankings. Furthermore, we optimize tasks such as retrieval by sampling hard negatives, creating smaller but effective splits. These optimizations allow us to introduce benchmarks that drastically reduce computational demands. For instance, our newly introduced zero-shot English benchmark maintains a ranking order similar to the full-scale version but at a fraction of the computational cost.
>
---
#### [replaced 078] nvBench 2.0: Resolving Ambiguity in Text-to-Visualization through Stepwise Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12880v2](http://arxiv.org/pdf/2503.12880v2)**

> **作者:** Tianqi Luo; Chuhan Huang; Leixian Shen; Boyan Li; Shuyu Shen; Wei Zeng; Nan Tang; Yuyu Luo
>
> **摘要:** Text-to-Visualization (Text2VIS) enables users to create visualizations from natural language queries, making data insights more accessible. However, Text2VIS faces challenges in interpreting ambiguous queries, as users often express their visualization needs in imprecise language. To address this challenge, we introduce nBench 2.0, a new benchmark designed to evaluate Text2VIS systems in scenarios involving ambiguous queries. nvBench 2.0 includes 7,878 natural language queries and 24,076 corresponding visualizations, derived from 780 tables across 153 domains. It is built using a controlled ambiguity-injection pipeline that generates ambiguous queries through a reverse-generation workflow. By starting with unambiguous seed visualizations and selectively injecting ambiguities, the pipeline yields multiple valid interpretations for each query, with each ambiguous query traceable to its corresponding visualization through step-wise reasoning paths. We evaluate various Large Language Models (LLMs) on their ability to perform ambiguous Text2VIS tasks using nBench 2.0. We also propose Step-Text2Vis, an LLM-based model trained on nvBench 2.0, which enhances performance in ambiguous scenarios through step-wise preference optimization. Our results show that Step-Text2Vis outperforms all baselines, setting a new state-of-the-art for ambiguous Text2VIS tasks. Our source code and data are available at https://nvbench2.github.io/
>
---
#### [replaced 079] NeoBERT: A Next-Generation BERT
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19587v2](http://arxiv.org/pdf/2502.19587v2)**

> **作者:** Lola Le Breton; Quentin Fournier; Mariam El Mezouar; John X. Morris; Sarath Chandar
>
> **备注:** 19 pages, 5 figures, 9 tables. Submitted to TMLR
>
> **摘要:** Recent innovations in architecture, pre-training, and fine-tuning have led to the remarkable in-context learning and reasoning abilities of large auto-regressive language models such as LLaMA and DeepSeek. In contrast, encoders like BERT and RoBERTa have not seen the same level of progress despite being foundational for many downstream NLP applications. To bridge this gap, we introduce NeoBERT, a next-generation encoder that redefines the capabilities of bidirectional models by integrating state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. NeoBERT is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an optimal depth-to-width ratio, and leverages an extended context length of 4,096 tokens. Despite its compact 250M parameter footprint, it achieves state-of-the-art results on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions. In addition, we rigorously evaluate the impact of each modification on GLUE and design a uniform fine-tuning and evaluation framework for MTEB. We release all code, data, checkpoints, and training scripts to accelerate research and real-world adoption.
>
---
#### [replaced 080] Sparse Autoencoders for Hypothesis Generation
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.04382v3](http://arxiv.org/pdf/2502.04382v3)**

> **作者:** Rajiv Movva; Kenny Peng; Nikhil Garg; Jon Kleinberg; Emma Pierson
>
> **备注:** ICML 2025. First two authors contributed equally. v3 adds SAE ablation experiments and qualitative annotation study to v2. Code: https://github.com/rmovva/HypotheSAEs
>
> **摘要:** We describe HypotheSAEs, a general method to hypothesize interpretable relationships between text data (e.g., headlines) and a target variable (e.g., clicks). HypotheSAEs has three steps: (1) train a sparse autoencoder on text embeddings to produce interpretable features describing the data distribution, (2) select features that predict the target variable, and (3) generate a natural language interpretation of each feature (e.g., "mentions being surprised or shocked") using an LLM. Each interpretation serves as a hypothesis about what predicts the target variable. Compared to baselines, our method better identifies reference hypotheses on synthetic datasets (at least +0.06 in F1) and produces more predictive hypotheses on real datasets (~twice as many significant findings), despite requiring 1-2 orders of magnitude less compute than recent LLM-based methods. HypotheSAEs also produces novel discoveries on two well-studied tasks: explaining partisan differences in Congressional speeches and identifying drivers of engagement with online headlines.
>
---
#### [replaced 081] Outlier-weighed Layerwise Sampling for LLM Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.18380v3](http://arxiv.org/pdf/2405.18380v3)**

> **作者:** Pengxiang Li; Lu Yin; Xiaowei Gao; Shiwei Liu
>
> **摘要:** The rapid advancements in Large Language Models (LLMs) have revolutionized various natural language processing tasks. However, the substantial size of LLMs presents significant challenges in training or fine-tuning. While parameter-efficient approaches such as low-rank adaptation (LoRA) have gained popularity, they often compromise performance compared to full-rank fine-tuning. In this paper, we propose Outlier-weighed Layerwise Sampling (OWS), a new memory-efficient fine-tuning approach, inspired by the layerwise outlier distribution of LLMs. Unlike LoRA, which adds extra adapters to all layers, OWS strategically assigns higher sampling probabilities to layers with more outliers, selectively sampling only a few layers and fine-tuning their pre-trained weights. To further increase the number of fine-tuned layers without a proportional rise in memory costs, we incorporate gradient low-rank projection, further boosting the approach's performance. Our extensive experiments across various architectures, including LLaMa2 and Mistral, demonstrate that OWS consistently outperforms baseline approaches, including full fine-tuning. Specifically, it achieves up to a 1.1% average accuracy gain on the Commonsense Reasoning benchmark, a 3.0% improvement on MMLU, and a notable 10% boost on MT-Bench, while being more memory efficient. OWS allows us to fine-tune 7B LLMs with only 21GB of memory. Our code is available at https://github.com/pixeli99/OWS.
>
---
#### [replaced 082] Generalized Interpolating Discrete Diffusion
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04482v2](http://arxiv.org/pdf/2503.04482v2)**

> **作者:** Dimitri von Rütte; Janis Fluri; Yuhui Ding; Antonio Orvieto; Bernhard Schölkopf; Thomas Hofmann
>
> **备注:** Published at ICML 2025; Code available at https://github.com/dvruette/gidd
>
> **摘要:** While state-of-the-art language models achieve impressive results through next-token prediction, they have inherent limitations such as the inability to revise already generated tokens. This has prompted exploration of alternative approaches such as discrete diffusion. However, masked diffusion, which has emerged as a popular choice due to its simplicity and effectiveness, reintroduces this inability to revise words. To overcome this, we generalize masked diffusion, deriving a new family of general interpolating discrete diffusion (GIDD) which offers greater flexibility in the design of the noising processes. Leveraging a novel diffusion ELBO, we achieve compute-matched state-of-the-art performance in diffusion language modeling. Exploiting GIDD's flexibility, we explore a hybrid approach combining masking and uniform noise, leading to improved sample quality and unlocking the ability for the model to correct its own mistakes, an area where autoregressive models notoriously have struggled. Code: https://github.com/dvruette/gidd/
>
---
#### [replaced 083] Sentence-level Reward Model can Generalize Better for Aligning LLM from Human Preference
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04793v2](http://arxiv.org/pdf/2503.04793v2)**

> **作者:** Wenjie Qiu; Yi-Chen Li; Xuqin Zhang; Tianyi Zhang; Yihang Zhang; Zongzhang Zhang; Yang Yu
>
> **摘要:** Learning reward models from human preference datasets and subsequently optimizing language models via reinforcement learning has emerged as a fundamental paradigm for aligning LLMs with human preferences. The performance of the reward model plays a crucial role in the effectiveness of alignment. Previous reward models operate at a coarse-grained level, requiring the generation of a complete response to obtain a reward value. The sparse reward may present challenges for downstream reinforcement learning. While recent efforts have attempted to learn token-level reward models, the lack of explicit semantic information makes it difficult to model the credit of every individual token. In this paper, we propose assigning scores to every sentence, introducing an intermediate-grained reward model. By segmenting the complete response into sentences and applying differential operations to reward output at the start and end positions of each sentence, we can effectively model the rewards of sentences. Moreover, a novel attention mechanism is introduced to aggregate the scores of all sentences into a response-level score, which allows it to be trained using the Bradley-Terry model. On common benchmarks, our method outperforms the response-level reward model by 2.7% on RewardBench (for reward modeling evaluation) and surpasses all baselines on AlpacaEval (for alignment evaluation).
>
---
#### [replaced 084] Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.18433v2](http://arxiv.org/pdf/2409.18433v2)**

> **作者:** Mucong Ding; Chenghao Deng; Jocelyn Choo; Zichu Wu; Aakriti Agrawal; Avi Schwarzschild; Tianyi Zhou; Tom Goldstein; John Langford; Anima Anandkumar; Furong Huang
>
> **备注:** NeurIPS 2024 Datasets and Benchmarks Track
>
> **摘要:** While generalization over tasks from easy to hard is crucial to profile language models (LLMs), the datasets with fine-grained difficulty annotations for each problem across a broad range of complexity are still blank. Aiming to address this limitation, we present Easy2Hard-Bench, a consistently formatted collection of 6 benchmark datasets spanning various domains, such as mathematics and programming problems, chess puzzles, and reasoning questions. Each problem within these datasets is annotated with numerical difficulty scores. To systematically estimate problem difficulties, we collect abundant performance data on attempts to each problem by humans in the real world or LLMs on the prominent leaderboard. Leveraging the rich performance data, we apply well-established difficulty ranking systems, such as Item Response Theory (IRT) and Glicko-2 models, to uniformly assign numerical difficulty scores to problems. Moreover, datasets in Easy2Hard-Bench distinguish themselves from previous collections by a higher proportion of challenging problems. Through extensive experiments with six state-of-the-art LLMs, we provide a comprehensive analysis of their performance and generalization capabilities across varying levels of difficulty, with the aim of inspiring future research in LLM generalization. The datasets are available at https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench.
>
---
#### [replaced 085] Theoretical Benefit and Limitation of Diffusion Language Model
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.09622v2](http://arxiv.org/pdf/2502.09622v2)**

> **作者:** Guhao Feng; Yihan Geng; Jian Guan; Wei Wu; Liwei Wang; Di He
>
> **备注:** 32 pages, 3 figures
>
> **摘要:** Diffusion language models have emerged as a promising approach for text generation. One would naturally expect this method to be an efficient replacement for autoregressive models since multiple tokens can be sampled in parallel during each diffusion step. However, its efficiency-accuracy trade-off is not yet well understood. In this paper, we present a rigorous theoretical analysis of a widely used type of diffusion language model, the Masked Diffusion Model (MDM), and find that its effectiveness heavily depends on the target evaluation metric. Under mild conditions, we prove that when using perplexity as the metric, MDMs can achieve near-optimal perplexity in sampling steps regardless of sequence length, demonstrating that efficiency can be achieved without sacrificing performance. However, when using the sequence error rate--which is important for understanding the "correctness" of a sequence, such as a reasoning chain--we show that the required sampling steps must scale linearly with sequence length to obtain "correct" sequences, thereby eliminating MDM's efficiency advantage over autoregressive models. Our analysis establishes the first theoretical foundation for understanding the benefits and limitations of MDMs. All theoretical findings are supported by empirical studies.
>
---
#### [replaced 086] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04745v2](http://arxiv.org/pdf/2504.04745v2)**

> **作者:** Ankush Raut; Xiaofeng Zhu; Maria Leonor Pacheco
>
> **备注:** 13 pages, 23 figures. Accepted at XLLM @ ACL 2025
>
> **摘要:** This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66.2% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81.3% in the best-case scenario.
>
---
#### [replaced 087] Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.01887v4](http://arxiv.org/pdf/2407.01887v4)**

> **作者:** Fanzeng Xia; Hao Liu; Yisong Yue; Tongxin Li
>
> **备注:** ACL 2025 Findings
>
> **摘要:** In-Context Reinforcement Learning (ICRL) is a frontier paradigm to solve Reinforcement Learning (RL) problems in the foundation model era. While ICRL capabilities have been demonstrated in transformers through task-specific training, the potential of Large Language Models (LLMs) out-of-the-box remains largely unexplored. This paper investigates whether LLMs can generalize cross-domain to perform ICRL under the problem of Dueling Bandits (DB), a stateless preference-based RL setting. We find that the top-performing LLMs exhibit a notable zero-shot capacity for relative decision-making, which translates to low short-term weak regret across all DB environment instances by quickly including the best arm in duels. However, an optimality gap still exists between LLMs and classic DB algorithms in terms of strong regret. LLMs struggle to converge and consistently exploit even when explicitly prompted to do so, and are sensitive to prompt variations. To bridge this gap, we propose an agentic flow framework: LLM with Enhanced Algorithmic Dueling (LEAD), which integrates off-the-shelf DB algorithm support with LLM agents through fine-grained adaptive interplay. We show that LEAD has theoretical guarantees inherited from classic DB algorithms on both weak and strong regret. We validate its efficacy and robustness even with noisy and adversarial prompts. The design of such an agentic framework sheds light on how to enhance the trustworthiness of general-purpose LLMs generalized to in-context decision-making tasks.
>
---
#### [replaced 088] On Support Samples of Next Word Prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04047v2](http://arxiv.org/pdf/2506.04047v2)**

> **作者:** Yuqian Li; Yupei Du; Yufang Liu; Feifei Feng; Mou Xiao Feng; Yuanbin Wu
>
> **备注:** Accepted to ACL2025(Main Conference)
>
> **摘要:** Language models excel in various tasks by making complex decisions, yet understanding the rationale behind these decisions remains a challenge. This paper investigates \emph{data-centric interpretability} in language models, focusing on the next-word prediction task. Using representer theorem, we identify two types of \emph{support samples}-those that either promote or deter specific predictions. Our findings reveal that being a support sample is an intrinsic property, predictable even before training begins. Additionally, while non-support samples are less influential in direct predictions, they play a critical role in preventing overfitting and shaping generalization and representation learning. Notably, the importance of non-support samples increases in deeper layers, suggesting their significant role in intermediate representation formation. These insights shed light on the interplay between data and model decisions, offering a new dimension to understanding language model behavior and interpretability.
>
---
#### [replaced 089] Unsolvable Problem Detection: Robust Understanding Evaluation for Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.20331v4](http://arxiv.org/pdf/2403.20331v4)**

> **作者:** Atsuyuki Miyai; Jingkang Yang; Jingyang Zhang; Yifei Ming; Qing Yu; Go Irie; Yixuan Li; Hai Li; Ziwei Liu; Kiyoharu Aizawa
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** This paper introduces a novel task to evaluate the robust understanding capability of Large Multimodal Models (LMMs), termed $\textbf{Unsolvable Problem Detection (UPD)}$. Multiple-choice question answering (MCQA) is widely used to assess the understanding capability of LMMs, but it does not guarantee that LMMs truly comprehend the answer. UPD assesses the LMM's ability to withhold answers when encountering unsolvable problems of MCQA, verifying whether the model truly understands the answer. UPD encompasses three problems: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD), covering unsolvable cases like answer-lacking or incompatible choices and image-question mismatches. For the evaluation, we introduce the MM-UPD Bench, a benchmark for assessing performance across various ability dimensions. Our experiments reveal that even most LMMs, which demonstrate adequate performance on existing benchmarks, struggle significantly with MM-UPD, underscoring a novel aspect of trustworthiness that current benchmarks have overlooked. A detailed analysis shows that LMMs have different bottlenecks and chain-of-thought and self-reflection improved performance for LMMs with the bottleneck in their LLM capability. We hope our insights will enhance the broader understanding and development of more reliable LMMs. The code is available at https://github.com/AtsuMiyai/UPD.
>
---
#### [replaced 090] ParallelComp: Parallel Long-Context Compressor for Length Extrapolation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14317v2](http://arxiv.org/pdf/2502.14317v2)**

> **作者:** Jing Xiong; Jianghan Shen; Chuanyang Zheng; Zhongwei Wan; Chenyang Zhao; Chiwun Yang; Fanghua Ye; Hongxia Yang; Lingpeng Kong; Ngai Wong
>
> **备注:** This paper has been accepted by ICML 2025
>
> **摘要:** Extrapolating ultra-long contexts (text length >128K) remains a major challenge for large language models (LLMs), as most training-free extrapolation methods are not only severely limited by memory bottlenecks, but also suffer from the attention sink, which restricts their scalability and effectiveness in practice. In this work, we propose ParallelComp, a parallel long-context compression method that effectively overcomes the memory bottleneck, enabling 8B-parameter LLMs to extrapolate from 8K to 128K tokens on a single A100 80GB GPU in a training-free setting. ParallelComp splits the input into chunks, dynamically evicting redundant chunks and irrelevant tokens, supported by a parallel KV cache eviction mechanism. Importantly, we present a systematic theoretical and empirical analysis of attention biases in parallel attention-including the attention sink, recency bias, and middle bias-and reveal that these biases exhibit distinctive patterns under ultra-long context settings. We further design a KV cache eviction technique to mitigate this phenomenon. Experimental results show that ParallelComp enables an 8B model (trained on 8K context) to achieve 91.17% of GPT-4's performance under ultra-long contexts, outperforming closed-source models such as Claude-2 and Kimi-Chat. We achieve a 1.76x improvement in chunk throughput, thereby achieving a 23.50x acceleration in the prefill stage with negligible performance loss and pave the way for scalable and robust ultra-long contexts extrapolation in LLMs. We release the code at https://github.com/menik1126/ParallelComp.
>
---
#### [replaced 091] Measuring Diversity in Synthetic Datasets
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.08512v2](http://arxiv.org/pdf/2502.08512v2)**

> **作者:** Yuchang Zhu; Huizhe Zhang; Bingzhe Wu; Jintang Li; Zibin Zheng; Peilin Zhao; Liang Chen; Yatao Bian
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Large language models (LLMs) are widely adopted to generate synthetic datasets for various natural language processing (NLP) tasks, such as text classification and summarization. However, accurately measuring the diversity of these synthetic datasets-an aspect crucial for robust model performance-remains a significant challenge. In this paper, we introduce DCScore, a novel method for measuring synthetic dataset diversity from a classification perspective. Specifically, DCScore formulates diversity evaluation as a sample classification task, leveraging mutual relationships among samples. We further provide theoretical verification of the diversity-related axioms satisfied by DCScore, highlighting its role as a principled diversity evaluation method. Experimental results on synthetic datasets reveal that DCScore enjoys a stronger correlation with multiple diversity pseudo-truths of evaluated datasets, underscoring its effectiveness. Moreover, both empirical and theoretical evidence demonstrate that DCScore substantially reduces computational costs compared to existing methods. Code is available at: https://github.com/bluewhalelab/dcscore.
>
---
#### [replaced 092] DISC: DISC: Dynamic Decomposition Improves LLM Inference Scaling
- **分类: cs.LG; cs.AI; cs.CL; cs.SE; I.2.6; I.2.7; I.2.8; D.2.3; F.2.2**

- **链接: [http://arxiv.org/pdf/2502.16706v2](http://arxiv.org/pdf/2502.16706v2)**

> **作者:** Jonathan Light; Wei Cheng; Benjamin Riviere; Wu Yue; Masafumi Oyamada; Mengdi Wang; Yisong Yue; Santiago Paternain; Haifeng Chen
>
> **摘要:** Inference scaling methods for large language models often work by breaking problems into steps or groups of tokens, then sampling and selecting the best next steps. However, these steps and their sizes are usually fixed or manually designed based on domain knowledge. We introduce dynamic decomposition, a method that adaptively and automatically breaks down solution and reasoning traces into manageable steps during inference. By allocating compute more effectively - especially by subdividing difficult steps and prioritizing their sampling - dynamic decomposition significantly boosts inference efficiency. Experiments on benchmarks like APPS, MATH, and LiveCodeBench show that dynamic decomposition outperforms fixed strategies such as token-level, sentence-level, and single-step decompositions, reducing the pass@10 error rate by 5.0%, 6.7%, and 10.5% respectively. These results show the promise of dynamic decomposition for improving a broad range of inference scaling techniques.
>
---
#### [replaced 093] How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.06424v3](http://arxiv.org/pdf/2411.06424v3)**

> **作者:** Yushi Yang; Filip Sondej; Harry Mayne; Andrew Lee; Adam Mahdi
>
> **摘要:** Safety fine-tuning algorithms reduce harmful outputs in language models, yet their mechanisms remain under-explored. Direct Preference Optimization (DPO) is a popular choice of algorithm, but prior explanations, attributing its effects solely to dampened toxic neurons in the MLP layers, are incomplete. In this study, we analyse four language models (Llama-3.1-8B, Gemma-2-2B, Mistral-7B, GPT-2-Medium) and show that toxic neurons only account for 2.5% to 24% of DPO's effects across models. Instead, DPO balances distributed activation shifts across all MLP neurons to create a net toxicity reduction. We attribute this reduction to four neuron groups, two aligned with reducing toxicity and two promoting anti-toxicity, whose combined effects replicate DPO across models. To further validate this understanding, we develop an activation editing method mimicking DPO through distributed shifts along a toxicity representation. This method outperforms DPO in reducing toxicity while preserving perplexity, without requiring any weight updates. Our work provides a mechanistic understanding of DPO and introduces an efficient, tuning-free alternative for safety fine-tuning.
>
---
#### [replaced 094] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17061v2](http://arxiv.org/pdf/2505.17061v2)**

> **作者:** Xinlong Chen; Yuanxing Zhang; Qiang Liu; Junfei Wu; Fuzheng Zhang; Tieniu Tan
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have exhibited impressive capabilities across various visual tasks, yet they remain hindered by the persistent challenge of hallucinations. To address this critical issue, we propose Mixture of Decoding (MoD), a novel approach for hallucination mitigation that dynamically adapts decoding strategies by evaluating the correctness of the model's attention on image tokens. Specifically, MoD measures the consistency between outputs generated from the original image tokens and those derived from the model's attended image tokens, to distinguish the correctness aforementioned. If the outputs are consistent, indicating correct attention, MoD employs a complementary strategy to amplify critical information. Conversely, if the outputs are inconsistent, suggesting erroneous attention, MoD utilizes a contrastive strategy to suppress misleading information. Extensive experiments demonstrate that MoD significantly outperforms existing decoding methods across multiple mainstream benchmarks, effectively mitigating hallucinations in LVLMs. The code is available at https://github.com/xlchen0205/MoD.
>
---
#### [replaced 095] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?
- **分类: cs.CL; cs.AI; cs.CV; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2502.11300v2](http://arxiv.org/pdf/2502.11300v2)**

> **作者:** Aashish Anantha Ramakrishnan; Aadarsh Anantha Ramakrishnan; Dongwon Lee
>
> **备注:** To appear at the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), Vienna, Austria, July 2025, https://2025.aclweb.org/
>
> **摘要:** Multimodal Large Language Models (MLLMs) are renowned for their superior instruction-following and reasoning capabilities across diverse problem domains. However, existing benchmarks primarily focus on assessing factual and logical correctness in downstream tasks, with limited emphasis on evaluating MLLMs' ability to interpret pragmatic cues and intermodal relationships. To address this gap, we assess the competency of MLLMs in performing Multimodal Discourse Analysis (MDA) using Coherence Relations. Our benchmark, CORDIAL, encompasses a broad spectrum of Coherence Relations across 3 different discourse domains at varying levels of granularity. Through our experiments on 10+ MLLMs employing different prompting strategies, we show that even top models like Gemini 1.5 Pro and GPT-4o fail to match the performance of simple classifier-based baselines. This study emphasizes the need to move beyond similarity-based metrics and adopt a discourse-driven framework for evaluating MLLMs, providing a more nuanced assessment of their capabilities. The benchmark and code are available at: https://aashish2000.github.io/CORDIAL/
>
---
#### [replaced 096] PECAN: LLM-Guided Dynamic Progress Control with Attention-Guided Hierarchical Weighted Graph for Long-Document QA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.04790v2](http://arxiv.org/pdf/2410.04790v2)**

> **作者:** Xinyu Wang; Yanzheng Xiang; Lin Gui; Yulan He
>
> **摘要:** Long-document QA presents challenges with large-scale text and long-distance dependencies. Recent advances in Large Language Models (LLMs) enable entire documents to be processed in a single pass. However, their computational cost is significantly high. Retrieval-Augmented Generation (RAG) methods split text into smaller chunks, but they often yield inferior results and may lose global context. Recent approaches that integrate LLMs into RAG via iterative summarization either underutilize LLM capabilities or still incur high computational costs. In this paper, we combine the high accuracy of LLMs with the efficiency of RAG and propose LLM-Guided Dynamic Progress Control with Attention-Based Hierarchical Weighted Graph (PECAN). Our method introduces two key improvements: (1) LLM-Guided Dynamic Progress Control: We leverage LLMs to dynamically control the retrieval process, adjusting the amount of retrieved information based on different queries to achieve a better balance of effectiveness and efficiency. (2) Attention-Guided Retrieval: We propose a novel retrieval method that constructs a hierarchical graph where edges are derived by LLM attention weights. Experimental results demonstrate that PECAN achieves LLM-level performance while maintaining computational complexity comparable to that of RAG methods on two single-document and two multi-document QA datasets.
>
---
#### [replaced 097] ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22048v3](http://arxiv.org/pdf/2503.22048v3)**

> **作者:** Chung-En Sun; Ge Yan; Tsui-Wei Weng
>
> **摘要:** Recent studies have shown that Large Language Models (LLMs) augmented with chain-of-thought (CoT) reasoning demonstrate impressive problem-solving abilities. However, in this work, we identify a recurring issue where these models occasionally generate overly short reasoning, leading to degraded performance on even simple mathematical problems. Specifically, we investigate how reasoning length is embedded in the hidden representations of reasoning models and its impact on accuracy. Our analysis reveals that reasoning length is governed by a linear direction in the representation space, allowing us to induce overly short reasoning by steering the model along this direction. Building on this insight, we introduce \textbf{\textit{ThinkEdit}}, a simple yet effective weight-editing approach to mitigate the issue of overly short reasoning. We first identify a small subset of attention heads (approximately 4%) that predominantly drive short reasoning behavior. We then edit the output projection weights of these heads to remove the short reasoning direction. With changes to only 0.2% of the model's parameters, \textbf{\textit{ThinkEdit}} effectively reduces overly short reasoning and yields notable accuracy gains for short reasoning outputs (+6.39%), along with an overall improvement across multiple math benchmarks (+3.34%). Our findings provide new mechanistic insights into how reasoning length is controlled within LLMs and highlight the potential of fine-grained model interventions to improve reasoning quality. Our code is available at: https://github.com/Trustworthy-ML-Lab/ThinkEdit\
>
---
#### [replaced 098] CoT is Not True Reasoning, It Is Just a Tight Constraint to Imitate: A Theory Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02878v2](http://arxiv.org/pdf/2506.02878v2)**

> **作者:** Jintian Shao; Yiming Cheng
>
> **备注:** Last page bad picture, lacking some sufficient experiments
>
> **摘要:** Chain-of-Thought (CoT) prompting has demonstrably enhanced the performance of Large Language Models on tasks requiring multi-step inference. This success has led to widespread claims of emergent reasoning capabilities in these models. In this paper, we present a theoretical counter-perspective: Chain-of-Thought (CoT) does not elicit genuine, abstract reasoning. Instead, we argue that Chain-of-Thought functions as a powerful structural constraint that guides Large Language Models to imitate the form of reasoning. By forcing the generation of intermediate steps, Chain-of-Thought leverages the model immense capacity for sequence prediction and pattern matching, effectively constraining its output to sequences that resemble coherent thought processes. Chain-of-Thought (CoT) prompting has demonstrably enhanced the performance of Large Language Models on tasks requiring multi-step inference. This success has led to widespread claims of emergent reasoning capabilities in these models. In this paper, we present a theoretical counter-perspective: Chain-of-Thought (CoT) does not elicit genuine, abstract reasoning. Instead, we argue that Chain-of-Thought functions as a powerful structural constraint that guides Large Language Models to imitate the form of reasoning. By forcing the generation of intermediate steps, Chain-of-Thought leverages the model immense capacity for sequence prediction and pattern matching, effectively constraining its output to sequences that resemble coherent thought processes.
>
---
#### [replaced 099] EgoNormia: Benchmarking Physical Social Norm Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20490v4](http://arxiv.org/pdf/2502.20490v4)**

> **作者:** MohammadHossein Rezaei; Yicheng Fu; Phil Cuvin; Caleb Ziems; Yanzhe Zhang; Hao Zhu; Diyi Yang
>
> **备注:** V3, with verified bench stats
>
> **摘要:** Human activity is moderated by norms; however, supervision for normative reasoning is sparse, particularly where norms are physically- or socially-grounded. We thus present EGONORMIA $\|\epsilon\|$, comprising 1,853 (200 for EGONORMIA-verified) multiple choice questions (MCQs) grounded within egocentric videos of human interactions, enabling the evaluation and improvement of normative reasoning in vision-language models (VLMs). EGONORMIA spans seven norm categories: safety, privacy, proxemics, politeness, cooperation, coordination/proactivity, and communication/legibility. To compile this dataset at scale, we propose a novel pipeline to generate grounded MCQs from raw egocentric video. Our work demonstrates that current state-of-the-art VLMs lack robust grounded norm understanding, scoring a maximum of 66% on EGONORMIA and 68% on EGONORMIA-verified, with performance across norm categories indicating significant risks of safety and privacy when VLMs are used in real-world agents. We additionally explore methods for improving normative understanding, demonstrating that a naive retrieval-based generation (RAG) method using EGONORMIA can enhance normative reasoning in VLMs.
>
---
#### [replaced 100] Binary Classifier Optimization for Large Language Model Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.04656v2](http://arxiv.org/pdf/2404.04656v2)**

> **作者:** Seungjae Jung; Gunsoo Han; Daniel Wontae Nam; Kyoung-Woon On
>
> **备注:** ACL 2025 main
>
> **摘要:** In real-world services such as ChatGPT, aligning models based on user feedback is crucial for improving model performance. However, due to the simplicity and convenience of providing feedback, users typically offer only basic binary signals, such as 'thumbs-up' or 'thumbs-down'. Most existing alignment research, on the other hand, relies on preference-based approaches that require both positive and negative responses as a pair. We propose Binary Classifier Optimization (BCO), a technique that effectively aligns LLMs using only binary feedback. BCO trains a binary classifier, where the logit serves as an implicit reward, effectively minimizing the Direct Preference Optimization (DPO) loss. We demonstrate that the binary cross-entropy loss employed in classifier training acts as an upper bound for the DPO loss. Additionally, a novel reward shift technique further minimizes the gap between the losses. We validate our methodology in two settings: first, on a paired preference dataset, where our method performs on par with DPO; and second, on a Likert-5 scale annotation dataset which stems from real users' queries. Our model consistently demonstrates effective and robust alignment across four base LLMs and three different datasets, showcasing the strength of our approach to learning from binary signals.
>
---
#### [replaced 101] JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models
- **分类: cs.CR; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2404.08793v2](http://arxiv.org/pdf/2404.08793v2)**

> **作者:** Yingchaojie Feng; Zhizhang Chen; Zhining Kang; Sijia Wang; Haoyu Tian; Wei Zhang; Minfeng Zhu; Wei Chen
>
> **摘要:** The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.
>
---
#### [replaced 102] Automated Capability Discovery via Foundation Model Self-Exploration
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07577v3](http://arxiv.org/pdf/2502.07577v3)**

> **作者:** Cong Lu; Shengran Hu; Jeff Clune
>
> **摘要:** Foundation models have become general-purpose assistants, exhibiting diverse capabilities across numerous domains through training on web-scale data. It remains challenging to precisely characterize even a fraction of the full spectrum of these abilities and potential risks in any new model. Existing evaluation approaches often require significant human effort, and it is taking increasing effort to design ever harder challenges for more capable models. We introduce Automated Capability Discovery (ACD), a framework that designates one foundation model as a scientist to systematically propose open-ended tasks probing the abilities of a subject model (potentially itself). By combining frontier models with ideas from the field of open-endedness, ACD automatically and systematically uncovers a diverse spectrum of surprising capabilities and failures in the subject model. We demonstrate ACD across a range of foundation models (including the GPT, Claude, and Llama series), showing that it automatically generates thousands of distinct tasks, which are then clustered to reveal dozens of broader capability areas and failure modes, that would be challenging for any single team to uncover. We further validate our method's automated scoring with extensive human surveys, observing high agreement between model-generated and human evaluations. By leveraging foundation models' ability to both create tasks and self-evaluate, ACD is a significant step toward scalable, automated evaluation of novel AI systems. All code and evaluation logs are open-sourced at https://github.com/conglu1997/ACD.
>
---
#### [replaced 103] RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.17458v2](http://arxiv.org/pdf/2409.17458v2)**

> **作者:** Yifan Jiang; Kriti Aggarwal; Tanmay Laud; Kashif Munir; Jay Pujara; Subhabrata Mukherjee
>
> **备注:** Accepted in ACL 2025 Findings
>
> **摘要:** The rapid progress of Large Language Models (LLMs) has opened up new opportunities across various domains and applications; yet it also presents challenges related to potential misuse. To mitigate such risks, red teaming has been employed as a proactive security measure to probe language models for harmful outputs via jailbreak attacks. However, current jailbreak attack approaches are single-turn with explicit malicious queries that do not fully capture the complexity of real-world interactions. In reality, users can engage in multi-turn interactions with LLM-based chat assistants, allowing them to conceal their true intentions in a more covert manner. To bridge this gap, we, first, propose a new jailbreak approach, RED QUEEN ATTACK. This method constructs a multi-turn scenario, concealing the malicious intent under the guise of preventing harm. We craft 40 scenarios that vary in turns and select 14 harmful categories to generate 56k multi-turn attack data points. We conduct comprehensive experiments on the RED QUEEN ATTACK with four representative LLM families of different sizes. Our experiments reveal that all LLMs are vulnerable to RED QUEEN ATTACK, reaching 87.62% attack success rate on GPT-4o and 75.4% on Llama3-70B. Further analysis reveals that larger models are more susceptible to the RED QUEEN ATTACK, with multi-turn structures and concealment strategies contributing to its success. To prioritize safety, we introduce a straightforward mitigation strategy called RED QUEEN GUARD, which aligns LLMs to effectively counter adversarial attacks. This approach reduces the attack success rate to below 1% while maintaining the model's performance across standard benchmarks. Full implementation and dataset are publicly accessible at https://github.com/kriti-hippo/red_queen.
>
---
#### [replaced 104] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11880v2](http://arxiv.org/pdf/2503.11880v2)**

> **作者:** Jieming Bian; Lei Wang; Letian Zhang; Jie Xu
>
> **摘要:** Fine-tuning large language models (LLMs) in federated settings enables privacy-preserving adaptation but suffers from cross-client interference due to model aggregation. Existing federated LoRA fine-tuning methods, primarily based on FedAvg, struggle with data heterogeneity, leading to harmful cross-client interference and suboptimal personalization. In this work, we propose \textbf{FedALT}, a novel personalized federated LoRA fine-tuning algorithm that fundamentally departs from FedAvg. Instead of using an aggregated model to initialize local training, each client continues training its individual LoRA while incorporating shared knowledge through a separate Rest-of-World (RoW) LoRA component. To effectively balance local adaptation and global information, FedALT introduces an adaptive mixer that dynamically learns input-specific weightings between the individual and RoW LoRA components, drawing conceptual foundations from the Mixture-of-Experts (MoE) paradigm. Through extensive experiments on NLP benchmarks, we demonstrate that FedALT significantly outperforms state-of-the-art personalized federated LoRA fine-tuning methods, achieving superior local adaptation without sacrificing computational efficiency.
>
---
#### [replaced 105] When Models Know More Than They Can Explain: Quantifying Knowledge Transfer in Human-AI Collaboration
- **分类: cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.05579v2](http://arxiv.org/pdf/2506.05579v2)**

> **作者:** Quan Shi; Carlos E. Jimenez; Shunyu Yao; Nick Haber; Diyi Yang; Karthik Narasimhan
>
> **备注:** For code, data, visualizer, visit: https://kite-live.vercel.app
>
> **摘要:** Recent advancements in AI reasoning have driven substantial improvements across diverse tasks. A critical open question is whether these improvements also yields better knowledge transfer: the ability of models to communicate reasoning in ways humans can understand, apply, and learn from. To investigate this, we introduce Knowledge Integration and Transfer Evaluation (KITE), a conceptual and experimental framework for Human-AI knowledge transfer capabilities and conduct the first large-scale human study (N=118) explicitly designed to measure it. In our two-phase setup, humans first ideate with an AI on problem-solving strategies, then independently implement solutions, isolating model explanations' influence on human understanding. Our findings reveal that although model benchmark performance correlates with collaborative outcomes, this relationship is notably inconsistent, featuring significant outliers, indicating that knowledge transfer requires dedicated optimization. Our analysis identifies behavioral and strategic factors mediating successful knowledge transfer. We release our code, dataset, and evaluation framework to support future work on communicatively aligned models.
>
---
#### [replaced 106] From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs
- **分类: cs.AI; cs.CL; cs.PL**

- **链接: [http://arxiv.org/pdf/2501.16207v4](http://arxiv.org/pdf/2501.16207v4)**

> **作者:** Jialun Cao; Yaojie Lu; Meiziniu Li; Haoyang Ma; Haokun Li; Mengda He; Cheng Wen; Le Sun; Hongyu Zhang; Shengchao Qin; Shing-Chi Cheung; Cong Tian
>
> **备注:** 20 pages
>
> **摘要:** The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO and have made significant progress. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and breaks it down into sub-tasks. We constructed 18k high-quality instruction-response pairs across five formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) by distilling gpt-4o and evaluated against ten open-sourced LLMs, including recent popular DeepSeek-R1. We also fine-tuned several 7~8B small models to achieve comparable performance with Deepseek-R1-671B. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding capabilities. Fine-tuned models are released at https: //huggingface.co/fm-universe.
>
---
#### [replaced 107] LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.14182v3](http://arxiv.org/pdf/2410.14182v3)**

> **作者:** Yujun Zhou; Jingdong Yang; Yue Huang; Kehan Guo; Zoe Emory; Bikram Ghosh; Amita Bedar; Sujay Shekar; Zhenwen Liang; Pin-Yu Chen; Tian Gao; Werner Geyer; Nuno Moniz; Nitesh V Chawla; Xiangliang Zhang
>
> **摘要:** Artificial Intelligence (AI) is revolutionizing scientific research, yet its growing integration into laboratory environments presents critical safety challenges. While large language models (LLMs) increasingly assist in tasks ranging from procedural guidance to autonomous experiment orchestration, an "illusion of understanding" may lead researchers to overestimate their reliability. Such overreliance is particularly dangerous in high-stakes laboratory settings, where failures in hazard identification or risk assessment can result in severe accidents. To address these concerns, we propose the Laboratory Safety Benchmark (LabSafety Bench), a comprehensive framework that evaluates large language models and vision language models (VLMs) on their ability to identify potential hazards, assess risks, and predict the consequences of unsafe actions in lab environments. LabSafety Bench comprises 765 multiple-choice questions aligned with US Occupational Safety and Health Administration (OSHA) protocols, along with 404 realistic laboratory scenarios featuring dual evaluation tasks: the Hazards Identification Test and the Consequence Identification Test, with 3128 open-ended questions in total. Evaluations across eight proprietary models, seven open-weight LLMs, and four VLMs reveal that, despite advanced performance on structured assessments, no model achieves the safety threshold required for reliable operation -- none scoring above 70% on the Hazards Identification Test. Moreover, while proprietary models tend to excel in multiple-choice evaluations, their performance in open-ended, real-world scenario responses is comparable to that of open-source models. These findings underscore the urgent need for specialized evaluation frameworks to ensure the safe and responsible deployment of AI in laboratory settings.
>
---
#### [replaced 108] Attention with Trained Embeddings Provably Selects Important Tokens
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17282v2](http://arxiv.org/pdf/2505.17282v2)**

> **作者:** Diyuan Wu; Aleksandr Shevchenko; Samet Oymak; Marco Mondelli
>
> **备注:** Fix mistakes in Lemma 4.2 and proof of Lemma 4.5
>
> **摘要:** Token embeddings play a crucial role in language modeling but, despite this practical relevance, their theoretical understanding remains limited. Our paper addresses the gap by characterizing the structure of embeddings obtained via gradient descent. Specifically, we consider a one-layer softmax attention model with a linear head for binary classification, i.e., $\texttt{Softmax}( p^\top E_X^\top ) E_X v = \frac{ \sum_{i=1}^T \exp(p^\top E_{x_i}) E_{x_i}^\top v}{\sum_{j=1}^T \exp(p^\top E_{x_{j}}) }$, where $E_X = [ E_{x_1} , \dots, E_{x_T} ]^\top$ contains the embeddings of the input sequence, $p$ is the embedding of the $\mathrm{\langle cls \rangle}$ token and $v$ the output vector. First, we show that, already after a single step of gradient training with the logistic loss, the embeddings $E_X$ capture the importance of tokens in the dataset by aligning with the output vector $v$ proportionally to the frequency with which the corresponding tokens appear in the dataset. Then, after training $p$ via gradient flow until convergence, the softmax selects the important tokens in the sentence (i.e., those that are predictive of the label), and the resulting $\mathrm{\langle cls \rangle}$ embedding maximizes the margin for such a selection. Experiments on real-world datasets (IMDB, Yelp) exhibit a phenomenology close to that unveiled by our theory.
>
---
#### [replaced 109] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14133v2](http://arxiv.org/pdf/2412.14133v2)**

> **作者:** Ido Cohen; Daniela Gottesman; Mor Geva; Raja Giryes
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Vision-language models (VLMs) excel at extracting and reasoning about information from images. Yet, their capacity to leverage internal knowledge about specific entities remains underexplored. This work investigates the disparity in model performance when answering factual questions about an entity described in text versus depicted in an image. Our results reveal a significant accuracy drop - reaching 18% for some models - when the entity is presented visually instead of textually. To study this gap we present PopVQA, a dataset which allows separating entity recognition and question answering, and use it to benchmark several models. We hypothesize that this decline arises from limitations in how information flows from image tokens to query tokens. Thus, we use mechanistic interpretability tools to reveal that, although image tokens are preprocessed by the vision encoder, meaningful information flow from these tokens occurs only in the much deeper layers. Furthermore, critical image processing happens in the language model's middle layers, allowing few layers for consecutive reasoning, highlighting a potential inefficiency in how the model utilizes its layers for reasoning. These insights shed light on the internal mechanics of VLMs and offer pathways for enhancing their reasoning capabilities. PopVQA can be found at https://huggingface.co/datasets/idoco/PopVQA.
>
---
#### [replaced 110] MIRIAD: Augmenting LLMs with millions of medical query-response pairs
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.06091v2](http://arxiv.org/pdf/2506.06091v2)**

> **作者:** Qinyue Zheng; Salman Abdullah; Sam Rawal; Cyril Zakka; Sophie Ostmeier; Maximilian Purk; Eduardo Reis; Eric J. Topol; Jure Leskovec; Michael Moor
>
> **备注:** Preprint
>
> **摘要:** LLMs are bound to transform healthcare with advanced decision support and flexible chat assistants. However, LLMs are prone to generate inaccurate medical content. To ground LLMs in high-quality medical knowledge, LLMs have been equipped with external knowledge via RAG, where unstructured medical knowledge is split into small text chunks that can be selectively retrieved and integrated into the LLMs context. Yet, existing RAG pipelines rely on raw, unstructured medical text, which can be noisy, uncurated and difficult for LLMs to effectively leverage. Systematic approaches to organize medical knowledge to best surface it to LLMs are generally lacking. To address these challenges, we introduce MIRIAD, a large-scale, curated corpus of 5,821,948 medical QA pairs, each rephrased from and grounded in a passage from peer-reviewed medical literature using a semi-automated pipeline combining LLM generation, filtering, grounding, and human annotation. Unlike prior medical corpora, which rely on unstructured text, MIRIAD encapsulates web-scale medical knowledge in an operationalized query-response format, which enables more targeted retrieval. Experiments on challenging medical QA benchmarks show that augmenting LLMs with MIRIAD improves accuracy up to 6.7% compared to unstructured RAG baselines with the same source corpus and with the same amount of retrieved text. Moreover, MIRIAD improved the ability of LLMs to detect medical hallucinations by 22.5 to 37% (increase in F1 score). We further introduce MIRIAD-Atlas, an interactive map of MIRIAD spanning 56 medical disciplines, enabling clinical users to visually explore, search, and refine medical knowledge. MIRIAD promises to unlock a wealth of down-stream applications, including medical information retrievers, enhanced RAG applications, and knowledge-grounded chat interfaces, which ultimately enables more reliable LLM applications in healthcare.
>
---
#### [replaced 111] Gumbel Reranking: Differentiable End-to-End Reranker Optimization
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.11116v2](http://arxiv.org/pdf/2502.11116v2)**

> **作者:** Siyuan Huang; Zhiyuan Ma; Jintao Du; Changhua Meng; Weiqiang Wang; Jingwen Leng; Minyi Guo; Zhouhan Lin
>
> **备注:** ACL 2025 Main
>
> **摘要:** RAG systems rely on rerankers to identify relevant documents. However, fine-tuning these models remains challenging due to the scarcity of annotated query-document pairs. Existing distillation-based approaches suffer from training-inference misalignment and fail to capture interdependencies among candidate documents. To overcome these limitations, we reframe the reranking process as an attention-mask problem and propose Gumbel Reranking, an end-to-end training framework for rerankers aimed at minimizing the training-inference gap. In our approach, reranker optimization is reformulated as learning a stochastic, document-wise Top-$k$ attention mask using the Gumbel Trick and Relaxed Top-$k$ Sampling. This formulation enables end-to-end optimization by minimizing the overall language loss. Experiments across various settings consistently demonstrate performance gains, including a 10.4\% improvement in recall on HotpotQA for distinguishing indirectly relevant documents.
>
---
#### [replaced 112] TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.06511v3](http://arxiv.org/pdf/2410.06511v3)**

> **作者:** Wanchao Liang; Tianyu Liu; Less Wright; Will Constable; Andrew Gu; Chien-Chin Huang; Iris Zhang; Wei Feng; Howard Huang; Junjie Wang; Sanket Purandare; Gokul Nadathur; Stratos Idreos
>
> **摘要:** The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions of tokens require sophisticated distributed systems that enable composing and comparing several state-of-the-art techniques in order to efficiently scale across thousands of accelerators. However, existing solutions are complex, scattered across multiple libraries/repositories, lack interoperability, and are cumbersome to maintain. Thus, curating and empirically comparing training recipes require non-trivial engineering effort. This paper introduces TorchTitan, an open-source, PyTorch-native distributed training system that unifies state-of-the-art techniques, streamlining integration and reducing overhead. TorchTitan enables 3D parallelism in a modular manner with elastic scaling, providing comprehensive logging, checkpointing, and debugging tools for production-ready training. It also incorporates hardware-software co-designed solutions, leveraging features like Float8 training and SymmetricMemory. As a flexible test bed, TorchTitan facilitates custom recipe curation and comparison, allowing us to develop optimized training recipes for Llama 3.1 and provide guidance on selecting techniques for maximum efficiency based on our experiences. We thoroughly assess TorchTitan on the Llama 3.1 family of LLMs, spanning 8 billion to 405 billion parameters, and showcase its exceptional performance, modular composability, and elastic scalability. By stacking training optimizations, we demonstrate accelerations of 65.08% with 1D parallelism at the 128-GPU scale (Llama 3.1 8B), an additional 12.59% with 2D parallelism at the 256-GPU scale (Llama 3.1 70B), and an additional 30% with 3D parallelism at the 512-GPU scale (Llama 3.1 405B) on NVIDIA H100 GPUs over optimized baselines.
>
---
#### [replaced 113] DBudgetKV: Dynamic Budget in KV Cache Compression for Ensuring Optimal Performance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16886v2](http://arxiv.org/pdf/2502.16886v2)**

> **作者:** Xuanfan Ni; Liyan Xu; Chenyang Lyu; Longyue Wang; Mo Yu; Lemao Liu; Fandong Meng; Jie Zhou; Piji Li
>
> **摘要:** To alleviate memory burden during inference of large language models (LLMs), numerous studies have focused on compressing the KV cache by exploring aspects such as attention sparsity. These techniques are often designed with a pre-defined KV budget; however, as the optimal budget varies by different input lengths and task types, the existence of a fixed budget could result in inconsistent performance accepting inputs of diverse domains. To address this limitation, we propose a new KV cache compression objective: to always ensure the full-cache performance regardless of specific inputs, while maximizing KV cache pruning as much as possible. To achieve this goal, we introduce a novel KV cache compression method dubbed DBudgetKV, which features an attention-based metric to signal when the remaining KV cache is unlikely to match the full-cache performance, then halting the pruning process. Empirical evaluation spanning diverse context lengths, task types, and model sizes suggests that our method achieves lossless KV pruning effectively and robustly, exceeding 25% compression ratio on average. Furthermore, our method is easy to integrate within LLM inference, not only optimizing memory space, but also showing reduced inference time compared to existing methods.
>
---
#### [replaced 114] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18668v3](http://arxiv.org/pdf/2505.18668v3)**

> **作者:** Zhen Li; Duan Li; Yukai Guo; Xinyuan Guo; Bowen Li; Lanxi Xiao; Shenyu Qiao; Jiashu Chen; Zijian Wu; Hui Zhang; Xinhuan Shu; Shixia Liu
>
> **备注:** 56 pages
>
> **摘要:** Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 330 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.
>
---
#### [replaced 115] Cool-Fusion: Fuse Large Language Models without Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.19807v2](http://arxiv.org/pdf/2407.19807v2)**

> **作者:** Cong Liu; Xiaojun Quan; Yan Pan; Liang Lin; Weigang Wu; Xu Chen
>
> **摘要:** We focus on the problem of fusing two or more heterogeneous large language models (LLMs) to leverage their complementary strengths. One of the challenges of model fusion is high computational load, specifically in fine-tuning or aligning vocabularies. To address this, we propose Cool-Fusion, a simple yet effective approach that fuses the knowledge of source LLMs, which does not require training. Unlike ensemble methods, Cool-Fusion is applicable to any set of source LLMs that have different vocabularies. To overcome the vocabulary discrepancies among LLMs, we ensemble LLMs on text level, allowing them to rerank the generated texts by each other with different granularities. Extensive experiments have been conducted across a variety of benchmark datasets. On GSM8K, Cool-Fusion increases accuracy from three strong source LLMs by a significant margin of 17.4\%.
>
---
#### [replaced 116] SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15289v4](http://arxiv.org/pdf/2412.15289v4)**

> **作者:** Xiaoning Dong; Wenbo Hu; Wei Xu; Tianxing He
>
> **备注:** To appear at Findings of ACL 2025
>
> **摘要:** Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
>
---
#### [replaced 117] Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05599v2](http://arxiv.org/pdf/2504.05599v2)**

> **作者:** Yi Peng; Peiyu Wang; Xiaokun Wang; Yichen Wei; Jiangbo Pei; Weijie Qiu; Ai Jian; Yunzhuo Hao; Jiachun Pan; Tianyidan Xie; Li Ge; Rongxian Zhuang; Xuchen Song; Yang Liu; Yahui Zhou
>
> **摘要:** We introduce Skywork R1V, a multimodal reasoning model extending the an R1-series Large language models (LLM) to visual modalities via an efficient multimodal transfer method. Leveraging a lightweight visual projector, Skywork R1V facilitates seamless multimodal adaptation without necessitating retraining of either the foundational language model or the vision encoder. To strengthen visual-text alignment, we propose a hybrid optimization strategy that combines Iterative Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), significantly enhancing cross-modal integration efficiency. Additionally, we introduce an adaptive-length Chain-of-Thought distillation approach for reasoning data generation. This approach dynamically optimizes reasoning chain lengths, thereby enhancing inference efficiency and preventing excessive reasoning overthinking. Empirical evaluations demonstrate that Skywork R1V, with only 38B parameters, delivers competitive performance, achieving a score of 69.0 on the MMMU benchmark and 67.5 on MathVista. Meanwhile, it maintains robust textual reasoning performance, evidenced by impressive scores of 72.0 on AIME and 94.0 on MATH500. The Skywork R1V model weights have been publicly released to promote openness and reproducibility.
>
---
#### [replaced 118] EconEvals: Benchmarks and Litmus Tests for LLM Agents in Unknown Environments
- **分类: cs.AI; cs.CL; cs.GT**

- **链接: [http://arxiv.org/pdf/2503.18825v2](http://arxiv.org/pdf/2503.18825v2)**

> **作者:** Sara Fish; Julia Shephard; Minkai Li; Ran I. Shorrer; Yannai A. Gonczarowski
>
> **摘要:** We develop benchmarks for LLM agents that act in, learn from, and strategize in unknown environments, the specifications of which the LLM agent must learn over time from deliberate exploration. Our benchmarks consist of decision-making tasks derived from key problems in economics. To forestall saturation, the benchmark tasks are synthetically generated with scalable difficulty levels. Additionally, we propose litmus tests, a new kind of quantitative measure for LLMs and LLM agents. Unlike benchmarks, litmus tests quantify differences in character, values, and tendencies of LLMs and LLM agents, by considering their behavior when faced with tradeoffs (e.g., efficiency versus equality) where there is no objectively right or wrong behavior. Overall, our benchmarks and litmus tests assess the abilities and tendencies of LLM agents in tackling complex economic problems in diverse settings spanning procurement, scheduling, task allocation, and pricing -- applications that should grow in importance as such agents are further integrated into the economy.
>
---
#### [replaced 119] AMPO: Active Multi-Preference Optimization for Self-play Preference Selection
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18293v2](http://arxiv.org/pdf/2502.18293v2)**

> **作者:** Taneesh Gupta; Rahul Madhavan; Xuchao Zhang; Chetan Bansal; Saravan Rajmohan
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Multi-preference optimization enriches language-model alignment beyond pairwise preferences by contrasting entire sets of helpful and undesired responses, thereby enabling richer training signals for large language models. During self-play alignment, these models often produce numerous candidate answers per query, rendering it computationally infeasible to include all responses in the training objective. In this work, we propose $\textit{Active Multi-Preference Optimization}$ (AMPO), a novel approach that combines on-policy generation, a multi-preference group-contrastive loss, and active subset selection. Specifically, we score and embed large candidate pools of responses and then select a small, yet informative, subset that covers reward extremes and distinct semantic clusters for preference optimization. Our contrastive training scheme is capable of identifying not only the best and worst answers but also subtle, underexplored modes that are crucial for robust alignment. Theoretically, we provide guarantees for expected reward maximization using our active selection method, and empirically, AMPO achieves state-of-the-art results on $\textit{AlpacaEval}$ using Llama 8B and Mistral 7B. We release our datasets $\href{https://huggingface.co/Multi-preference-Optimization}{here}$.
>
---
#### [replaced 120] CVC: A Large-Scale Chinese Value Rule Corpus for Value Alignment of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01495v2](http://arxiv.org/pdf/2506.01495v2)**

> **作者:** Ping Wu; Guobin Shen; Dongcheng Zhao; Yuwei Wang; Yiting Dong; Yu Shi; Enmeng Lu; Feifei Zhao; Yi Zeng
>
> **摘要:** Ensuring that Large Language Models (LLMs) align with mainstream human values and ethical norms is crucial for the safe and sustainable development of AI. Current value evaluation and alignment are constrained by Western cultural bias and incomplete domestic frameworks reliant on non-native rules; furthermore, the lack of scalable, rule-driven scenario generation methods makes evaluations costly and inadequate across diverse cultural contexts. To address these challenges, we propose a hierarchical value framework grounded in core Chinese values, encompassing three main dimensions, 12 core values, and 50 derived values. Based on this framework, we construct a large-scale Chinese Values Corpus (CVC) containing over 250,000 value rules enhanced and expanded through human annotation. Experimental results show that CVC-guided scenarios outperform direct generation ones in value boundaries and content diversity. In the evaluation across six sensitive themes (e.g., surrogacy, suicide), seven mainstream LLMs preferred CVC-generated options in over 70.5% of cases, while five Chinese human annotators showed an 87.5% alignment with CVC, confirming its universality, cultural relevance, and strong alignment with Chinese values. Additionally, we construct 400,000 rule-based moral dilemma scenarios that objectively capture nuanced distinctions in conflicting value prioritization across 17 LLMs. Our work establishes a culturally-adaptive benchmarking framework for comprehensive value evaluation and alignment, representing Chinese characteristics. All data are available at https://huggingface.co/datasets/Beijing-AISI/CVC, and the code is available at https://github.com/Beijing-AISI/CVC.
>
---
#### [replaced 121] From Sub-Ability Diagnosis to Human-Aligned Generation: Bridging the Gap for Text Length Control via MARKERGEN
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13544v3](http://arxiv.org/pdf/2502.13544v3)**

> **作者:** Peiwen Yuan; Chuyi Tan; Shaoxiong Feng; Yiwei Li; Xinglin Wang; Yueqi Zhang; Jiayi Shi; Boyuan Pan; Yao Hu; Kan Li
>
> **摘要:** Despite the rapid progress of large language models (LLMs), their length-controllable text generation (LCTG) ability remains below expectations, posing a major limitation for practical applications. Existing methods mainly focus on end-to-end training to reinforce adherence to length constraints. However, the lack of decomposition and targeted enhancement of LCTG sub-abilities restricts further progress. To bridge this gap, we conduct a bottom-up decomposition of LCTG sub-abilities with human patterns as reference and perform a detailed error analysis. On this basis, we propose MarkerGen, a simple-yet-effective plug-and-play approach that:(1) mitigates LLM fundamental deficiencies via external tool integration;(2) conducts explicit length modeling with dynamically inserted markers;(3) employs a three-stage generation scheme to better align length constraints while maintaining content quality. Comprehensive experiments demonstrate that MarkerGen significantly improves LCTG across various settings, exhibiting outstanding effectiveness and generalizability.
>
---
#### [replaced 122] Legal Mathematical Reasoning with LLMs: Procedural Alignment through Two-Stage Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02590v2](http://arxiv.org/pdf/2504.02590v2)**

> **作者:** Kepu Zhang; Guofu Xie; Weijie Yu; Mingyue Xu; Xu Tang; Yaxin Li; Jun Xu
>
> **摘要:** Legal mathematical reasoning is essential for applying large language models (LLMs) in high-stakes legal contexts, where outputs must be both mathematically accurate and procedurally compliant. However, existing legal LLMs lack structured numerical reasoning, and open-domain models, though capable of calculations, often overlook mandatory legal steps. To address this, we present LexNum, the first Chinese legal mathematical reasoning benchmark, covering three representative scenarios where each instance reflects legally grounded procedural flows. We further propose LexPam, a two-stage reinforcement learning framework for efficient legal reasoning training. Leveraging curriculum learning, we use a stronger teacher model to partition data into basic and challenging subsets. A lightweight 1.5B student model is then fine-tuned with Group Relative Policy Optimization, which avoids costly value networks and enables stable training from sparse, end-of-sequence rewards. The first stage improves accuracy and format; the second introduces a novel reward to guide procedural alignment via task-specific legal elements. Experiments show that existing models perform poorly on LexNum, while LexPam enhances both mathematical accuracy and legal coherence, and generalizes effectively across tasks and domains.
>
---
#### [replaced 123] Evaluating Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12564v3](http://arxiv.org/pdf/2412.12564v3)**

> **作者:** Chengyan Wu; Bolei Ma; Zheyu Zhang; Ningyuan Deng; Yanqing He; Yun Xue
>
> **备注:** Preprint; Paper accepted at International Journal of Machine Learning and Cybernetics
>
> **摘要:** Aspect-based sentiment analysis (ABSA), a sequence labeling task, has attracted increasing attention in multilingual contexts. While previous research has focused largely on fine-tuning or training models specifically for ABSA, we evaluate large language models (LLMs) under zero-shot conditions to explore their potential to tackle this challenge with minimal task-specific adaptation. We conduct a comprehensive empirical evaluation of a series of LLMs on multilingual ABSA tasks, investigating various prompting strategies, including vanilla zero-shot, chain-of-thought (CoT), self-improvement, self-debate, and self-consistency, across nine different models. Results indicate that while LLMs show promise in handling multilingual ABSA, they generally fall short of fine-tuned, task-specific models. Notably, simpler zero-shot prompts often outperform more complex strategies, especially in high-resource languages like English. These findings underscore the need for further refinement of LLM-based approaches to effectively address ABSA task across diverse languages.
>
---
#### [replaced 124] AI Scientists Fail Without Strong Implementation Capability
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01372v2](http://arxiv.org/pdf/2506.01372v2)**

> **作者:** Minjun Zhu; Qiujie Xie; Yixuan Weng; Jian Wu; Zhen Lin; Linyi Yang; Yue Zhang
>
> **备注:** Position
>
> **摘要:** The emergence of Artificial Intelligence (AI) Scientist represents a paradigm shift in scientific discovery, with large language models (LLMs) taking the lead as the primary executor in the entire scientific workflow from idea generation to experiment implementation. Recent AI Scientist studies demonstrate sufficient capabilities for independent scientific discovery, with the generated research reports gaining acceptance at the ICLR 2025 workshop and ACL 2025, arguing that a human-level AI Scientist, capable of uncovering phenomena previously unknown to humans, may be imminent. Despite this substantial progress, AI Scientist has yet to produce a groundbreaking achievement in the domain of computer science on par with automated scientific tools. Based on extensive quantitative evidence from existing benchmarks in complex engineering tasks and a systematic evaluation assess 28 research papers generated by five advanced AI Scientist systems, we argue that \textbf{the fundamental bottleneck for AI Scientists lies in their capability to execute the requisite verification procedures.} Current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers. To better illustrate the root cause of this \textbf{implementation gap}, we provide an in-depth discussion on the fundamental limitations of AI Scientist. This position paper aims to call for the participants in the community to bridge the implementation gap.
>
---
#### [replaced 125] Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08351v2](http://arxiv.org/pdf/2505.08351v2)**

> **作者:** Mina Almasi; Ross Deans Kristensen-McLachlan
>
> **备注:** Accepted at BEA2025 (Conference workshop at ACL 2025)
>
> **摘要:** This paper investigates the potentials of Large Language Models (LLMs) as adaptive tutors in the context of second-language learning. In particular, we evaluate whether system prompting can reliably constrain LLMs to generate only text appropriate to the student's competence level. We simulate full teacher-student dialogues in Spanish using instruction-tuned, open-source LLMs ranging in size from 7B to 12B parameters. Dialogues are generated by having an LLM alternate between tutor and student roles with separate chat histories. The output from the tutor model is then used to evaluate the effectiveness of CEFR-based prompting to control text difficulty across three proficiency levels (A1, B1, C1). Our findings suggest that while system prompting can be used to constrain model outputs, prompting alone is too brittle for sustained, long-term interactional contexts - a phenomenon we term alignment drift. Our results provide insights into the feasibility of LLMs for personalized, proficiency-aligned adaptive tutors and provide a scalable method for low-cost evaluation of model performance without human participants.
>
---
#### [replaced 126] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.04873v2](http://arxiv.org/pdf/2408.04873v2)**

> **作者:** Priyanka Kargupta; Yunyi Zhang; Yizhu Jiao; Siru Ouyang; Jiawei Han
>
> **备注:** ACL 2025 Main Conference. Code available here: https://github.com/pkargupta/epimine
>
> **摘要:** State-of-the-art automatic event detection struggles with interpretability and adaptability to evolving large-scale key events -- unlike episodic structures, which excel in these areas. Often overlooked, episodes represent cohesive clusters of core entities performing actions at a specific time and location; a partially ordered sequence of episodes can represent a key event. This paper introduces a novel task, episode detection, which identifies episodes within a news corpus of key event articles. Detecting episodes poses unique challenges, as they lack explicit temporal or locational markers and cannot be merged using semantic similarity alone. While large language models (LLMs) can aid with these reasoning difficulties, they suffer with long contexts typical of news corpora. To address these challenges, we introduce EpiMine, an unsupervised framework that identifies a key event's candidate episodes by leveraging natural episodic partitions in articles, estimated through shifts in discriminative term combinations. These candidate episodes are more cohesive and representative of true episodes, synergizing with LLMs to better interpret and refine them into final episodes. We apply EpiMine to our three diverse, real-world event datasets annotated at the episode level, where it achieves a 59.2% average gain across all metrics compared to baselines.
>
---
#### [replaced 127] Do Large Language Models Judge Error Severity Like Humans?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05142v2](http://arxiv.org/pdf/2506.05142v2)**

> **作者:** Diege Sun; Guanyi Chen; Zhao Fan; Xiaorong Cheng; Tingting He
>
> **摘要:** Large Language Models (LLMs) are increasingly used as automated evaluators in natural language generation, yet it remains unclear whether they can accurately replicate human judgments of error severity. In this study, we systematically compare human and LLM assessments of image descriptions containing controlled semantic errors. We extend the experimental framework of van Miltenburg et al. (2020) to both unimodal (text-only) and multimodal (text + image) settings, evaluating four error types: age, gender, clothing type, and clothing colour. Our findings reveal that humans assign varying levels of severity to different error types, with visual context significantly amplifying perceived severity for colour and type errors. Notably, most LLMs assign low scores to gender errors but disproportionately high scores to colour errors, unlike humans, who judge both as highly severe but for different reasons. This suggests that these models may have internalised social norms influencing gender judgments but lack the perceptual grounding to emulate human sensitivity to colour, which is shaped by distinct neural mechanisms. Only one of the evaluated LLMs, Doubao, replicates the human-like ranking of error severity, but it fails to distinguish between error types as clearly as humans. Surprisingly, DeepSeek-V3, a unimodal LLM, achieves the highest alignment with human judgments across both unimodal and multimodal conditions, outperforming even state-of-the-art multimodal models.
>
---
#### [replaced 128] Cartridges: Lightweight and general-purpose long context representations via self-study
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06266v2](http://arxiv.org/pdf/2506.06266v2)**

> **作者:** Sabri Eyuboglu; Ryan Ehrlich; Simran Arora; Neel Guha; Dylan Zinsley; Emily Liu; Will Tennien; Atri Rudra; James Zou; Azalia Mirhoseini; Christopher Re
>
> **摘要:** Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining.
>
---
#### [replaced 129] ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04929v2](http://arxiv.org/pdf/2506.04929v2)**

> **作者:** Mikołaj Pokrywka; Wojciech Kusa; Mieszko Rutkowski; Mikołaj Koszowski
>
> **备注:** Accepted at ACL 2025 (The 63rd Annual Meeting of the Association for Computational Linguistics)
>
> **摘要:** Neural Machine Translation (NMT) has improved translation by using Transformer-based models, but it still struggles with word ambiguity and context. This problem is especially important in domain-specific applications, which often have problems with unclear sentences or poor data quality. Our research explores how adding information to models can improve translations in the context of e-commerce data. To this end we create ConECT -- a new Czech-to-Polish e-commerce product translation dataset coupled with images and product metadata consisting of 11,400 sentence pairs. We then investigate and compare different methods that are applicable to context-aware translation. We test a vision-language model (VLM), finding that visual context aids translation quality. Additionally, we explore the incorporation of contextual information into text-to-text models, such as the product's category path or image descriptions. The results of our study demonstrate that the incorporation of contextual information leads to an improvement in the quality of machine translation. We make the new dataset publicly available.
>
---
#### [replaced 130] When Attention Collapses: How Degenerate Layers in LLMs Enable Smaller, Stronger Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.08634v3](http://arxiv.org/pdf/2404.08634v3)**

> **作者:** Sunny Sanyal; Ravid Shwartz-Ziv; Alexandros G. Dimakis; Sujay Sanghavi
>
> **备注:** 29 pages, 22 figures, 11 tables
>
> **摘要:** Large Language Models (LLMs) rely on the transformer architecture and its self-attention mechanism to deliver strong performance across tasks. However, we uncover a structural inefficiency in standard pre-trained decoder-style LLMs: in many of the deeper layers, attention matrices frequently collapse to near rank-one, single-column patterns. We refer to these underutilized components as lazy layers, which are redundant and computationally inefficient. To address this, we propose Inheritune, a simple and effective training recipe for building smaller, more efficient, and high performing language models. Inheritune initializes a compact model by inheriting the useful early layers from a larger pre-trained model, then progressively retrains and expands it. Our experiments across multiple models and datasets show that Inheritune trained models, despite having significantly fewer layers, can match or even outperform their larger counterparts. This approach yields compact, performant models and offers a practical path for efficient language model compression. Code is available at https://github.com/sanyalsunny111/LLM-Inheritune
>
---
#### [replaced 131] MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.18077v3](http://arxiv.org/pdf/2411.18077v3)**

> **作者:** Akshat Sharma; Hangliang Ding; Jianping Li; Neel Dani; Minjia Zhang
>
> **摘要:** How to efficiently serve LLMs in practice has become exceptionally challenging due to their prohibitive memory and computation requirements. In this study, we investigate optimizing the KV cache, whose memory footprint poses a critical bottleneck in LLM inference, especially when dealing with long context tasks. To tackle the challenge, we introduce MiniKV, a KV cache optimization method that simultaneously preserves long context task accuracy while significantly reducing KV cache size via a novel 2-bit layer-discriminative KV cache. More importantly, we develop specialized CUDA kernels to make MiniKV compatible with FlashAttention. Experiments on a wide range of long context tasks show that MiniKV effectively achieves 86% KV cache compression ratio while recovering over 98.5% of accuracy, outperforming state-of-the-art methods while achieving excellent measured system performance improvements.
>
---
#### [replaced 132] Synthetic Text Generation for Training Large Language Models via Gradient Matching
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17607v2](http://arxiv.org/pdf/2502.17607v2)**

> **作者:** Dang Nguyen; Zeman Li; Mohammadhossein Bateni; Vahab Mirrokni; Meisam Razaviyayn; Baharan Mirzasoleiman
>
> **备注:** 18 pages, 10 figures, 5 tables, link: https://github.com/BigML-CS-UCLA/GRADMM
>
> **摘要:** Synthetic data has the potential to improve the performance, training efficiency, and privacy of real training examples. Nevertheless, existing approaches for synthetic text generation are mostly heuristics and cannot generate human-readable text without compromising the privacy of real data, or provide performance guarantees for training Large Language Models (LLMs). In this work, we propose the first theoretically rigorous approach for generating synthetic human-readable text that provides convergence, performance, and privacy guarantees for fine-tuning LLMs on a target task. To do so, we leverage Alternating Direction Method of Multipliers (ADMM) that iteratively optimizes the embeddings of synthetic examples to match the noisy gradient of the target training or validation data, and maps them to a sequence of text tokens with low perplexity. In doing so, the generated synthetic text guarantees convergence of the model to a close neighborhood of the solution obtained by fine-tuning on real data and preserves their privacy. Experiments on various classification tasks confirm the effectiveness of our proposed approach. Our code is available at https://github.com/BigML-CS-UCLA/GRADMM.
>
---
#### [replaced 133] Mechanistic evaluation of Transformers and state space models
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.15105v2](http://arxiv.org/pdf/2505.15105v2)**

> **作者:** Aryaman Arora; Neil Rathi; Nikil Roashan Selvam; Róbert Csordás; Dan Jurafsky; Christopher Potts
>
> **备注:** 9 page main text, 6 pages appendix
>
> **摘要:** State space models (SSMs) for language modelling promise an efficient and performant alternative to quadratic-attention Transformers, yet show variable performance on recalling basic information from the context. While performance on synthetic tasks like Associative Recall (AR) can point to this deficiency, behavioural metrics provide little information as to why--on a mechanistic level--certain architectures fail and others succeed. To address this, we conduct experiments on AR and find that only Transformers and Based SSM models fully succeed at AR, with Mamba a close third, whereas the other SSMs (H3, Hyena) fail. We then use causal interventions to explain why. We find that Transformers and Based learn to store key-value associations in-context using induction heads. By contrast, the SSMs compute these associations only at the last state, with only Mamba succeeding because of its short convolution component. To extend and deepen these findings, we introduce Associative Treecall (ATR), a synthetic task similar to AR based on PCFG induction. ATR introduces language-like hierarchical structure into the AR setting. We find that all architectures learn the same mechanism as they did for AR, and the same three models succeed at the task. These results reveal that architectures with similar accuracy may still have substantive differences, motivating the adoption of mechanistic evaluations.
>
---
#### [replaced 134] Robustifying Vision-Language Models via Dynamic Token Reweighting
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17132v2](http://arxiv.org/pdf/2505.17132v2)**

> **作者:** Tanqiu Jiang; Jiacheng Liang; Rongyi Zhu; Jiawei Zhou; Fenglong Ma; Ting Wang
>
> **摘要:** Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. (warning: this paper contains potentially harmful content generated by VLMs.)
>
---
#### [replaced 135] Rational Decision-Making Agent with Internalized Utility Judgment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2308.12519v3](http://arxiv.org/pdf/2308.12519v3)**

> **作者:** Yining Ye; Xin Cong; Shizuo Tian; Yujia Qin; Chong Liu; Yankai Lin; Zhiyuan Liu; Maosong Sun
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable advancements and have attracted significant efforts to develop LLMs into agents capable of executing intricate multi-step decision-making tasks beyond traditional NLP applications. Existing approaches to LLM-based decision-making predominantly build upon the manually-designed external performance metrics to guide the decision-making process. However, reliance on the external performance metrics as prior is problematic in real-world scenarios, where such prior may be unavailable, flawed, or even erroneous. For genuine autonomous decision making, it is imperative for the agent to develop its rationality from its posterior experiences to judge decisions independently. Central to the development of rationality is the construction of an internalized utility judgment, capable of assigning numerical utilities to each decision. This paper proposes RadAgent (Rational Decision-Making Agent), which fosters the development of its rationality through an iterative framework involving Experience Exploration and Utility Learning. Within this framework, Elo-based Utility Construction is devised to assign Elo scores to individual decision steps to judge their utilities via pairwise comparisons. Consequently, these Elo scores guide the decision-making process to derive optimal outcomes. Experimental results on the ToolBench dataset demonstrate RadAgent's superiority over baselines, achieving over 10% improvement in Pass Rate on diverse tasks. It offers higher-quality solutions and reduces costs (ChatGPT API calls), highlighting its effectiveness and efficiency.
>
---
#### [replaced 136] Comba: Improving Bilinear RNNs with Closed-loop Control
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02475v2](http://arxiv.org/pdf/2506.02475v2)**

> **作者:** Jiaxi Hu; Yongqi Pan; Jusen Du; Disen Lan; Xiaqiang Tang; Qingsong Wen; Yuxuan Liang; Weigao Sun
>
> **摘要:** Recent efficient sequence modeling methods such as Gated DeltaNet, TTT, and RWKV-7 have achieved performance improvements by supervising the recurrent memory management through Delta learning rule. Unlike previous state-space models (e.g., Mamba) and gated linear attentions (e.g., GLA), these models introduce interactions between the recurrent state and the key vector, structurally resembling bilinear systems. In this paper, we first introduce the concept of Bilinear RNNs with a comprehensive analysis on the advantages and limitations of these models. Then, based on closed-loop control theory, we propose a novel Bilinear RNN variant named Comba, which adopts a scalar-plus-low-rank state transition, with both state feedback and output feedback corrections. We also implement a hardware-efficient chunk-wise parallel kernel in Triton and train models with 340M/1.3B parameters on large-scale corpus. Comba demonstrates superior performance and computation efficiency in both language and vision modeling.
>
---
#### [replaced 137] Diversifying the Expert Knowledge for Task-Agnostic Pruning in Sparse Mixture-of-Experts
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.09590v4](http://arxiv.org/pdf/2407.09590v4)**

> **作者:** Zeliang Zhang; Xiaodong Liu; Hao Cheng; Chenliang Xu; Jianfeng Gao
>
> **备注:** Findings of ACL 25. Fix typos. Correct and add more details on the expert similarity measurement
>
> **摘要:** By increasing model parameters but activating them sparsely when performing a task, the use of Mixture-of-Experts (MoE) architecture significantly improves the performance of Large Language Models (LLMs) without increasing the inference cost. However, the memory consumption due to the growing number of experts presents a challenge to the deployment of these models in many real world settings. Our empirical study reveals that some experts encode redundant knowledge during pre-training. We thus propose a method of grouping and pruning similar experts to improve the model's parameter efficiency. We validate the effectiveness of our method by pruning three state-of-the-art MoE architectures, including Mixtral, Deepseek-MoE, and Qwen. The evaluation shows that our method outperforms other model pruning methods on a range of natural language tasks. We will release our code to facilitate future research.
>
---
#### [replaced 138] Watermark under Fire: A Robustness Evaluation of LLM Watermarking
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.13425v3](http://arxiv.org/pdf/2411.13425v3)**

> **作者:** Jiacheng Liang; Zian Wang; Lauren Hong; Shouling Ji; Ting Wang
>
> **备注:** 22 pages
>
> **摘要:** Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.
>
---
#### [replaced 139] LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.00493v3](http://arxiv.org/pdf/2503.00493v3)**

> **作者:** Boyi Kang; Xinfa Zhu; Zihan Zhang; Zhen Ye; Mingshuai Liu; Ziqian Wang; Yike Zhu; Guobin Ma; Jun Chen; Longshuai Xiao; Chao Weng; Wei Xue; Lei Xie
>
> **备注:** ACL2025 main, Codes available at https://github.com/Kevin-naticl/LLaSE-G1
>
> **摘要:** Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area.
>
---
#### [replaced 140] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00975v3](http://arxiv.org/pdf/2506.00975v3)**

> **作者:** Qichao Wang; Ziqiao Meng; Wenqian Cui; Yifei Zhang; Pengcheng Wu; Bingzhe Wu; Irwin King; Liang Chen; Peilin Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
>
---
#### [replaced 141] Understanding and Mitigating Cross-lingual Privacy Leakage via Language-specific and Universal Privacy Neurons
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00759v2](http://arxiv.org/pdf/2506.00759v2)**

> **作者:** Wenshuo Dong; Qingsong Yang; Shu Yang; Lijie Hu; Meng Ding; Wanyu Lin; Tianhang Zheng; Di Wang
>
> **摘要:** Large Language Models (LLMs) trained on massive data capture rich information embedded in the training data. However, this also introduces the risk of privacy leakage, particularly involving personally identifiable information (PII). Although previous studies have shown that this risk can be mitigated through methods such as privacy neurons, they all assume that both the (sensitive) training data and user queries are in English. We show that they cannot defend against the privacy leakage in cross-lingual contexts: even if the training data is exclusively in one language, these (private) models may still reveal private information when queried in another language. In this work, we first investigate the information flow of cross-lingual privacy leakage to give a better understanding. We find that LLMs process private information in the middle layers, where representations are largely shared across languages. The risk of leakage peaks when converted to a language-specific space in later layers. Based on this, we identify privacy-universal neurons and language-specific privacy neurons. Privacy-universal neurons influence privacy leakage across all languages, while language-specific privacy neurons are only related to specific languages. By deactivating these neurons, the cross-lingual privacy leakage risk is reduced by 23.3%-31.6%.
>
---
#### [replaced 142] Large Language Models Often Know When They Are Being Evaluated
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23836v2](http://arxiv.org/pdf/2505.23836v2)**

> **作者:** Joe Needham; Giles Edkins; Govind Pimpale; Henning Bartsch; Marius Hobbhahn
>
> **摘要:** If AI models can detect when they are being evaluated, the effectiveness of evaluations might be compromised. For example, models could have systematically different behavior during evaluations, leading to less reliable benchmarks for deployment and governance decisions. We investigate whether frontier language models can accurately classify transcripts based on whether they originate from evaluations or real-world deployment, a capability we call evaluation awareness. To achieve this, we construct a diverse benchmark of 1,000 prompts and transcripts from 61 distinct datasets. These span public benchmarks (e.g., MMLU, SWEBench), real-world deployment interactions, and agent trajectories from scaffolding frameworks (e.g., web-browsing agents). Frontier models clearly demonstrate above-random evaluation awareness (Gemini-2.5-Pro reaches an AUC of $0.83$), but do not yet surpass our simple human baseline (AUC of $0.92$). Furthermore, both AI models and humans are better at identifying evaluations in agentic settings compared to chat settings. Additionally, we test whether models can identify the purpose of the evaluation. Under multiple-choice and open-ended questioning, AI models far outperform random chance in identifying what an evaluation is testing for. Our results indicate that frontier models already exhibit a substantial, though not yet superhuman, level of evaluation-awareness. We recommend tracking this capability in future models.
>
---
#### [replaced 143] AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.12559v2](http://arxiv.org/pdf/2503.12559v2)**

> **作者:** Xiao Wang; Qingyi Si; Jianlong Wu; Shiyu Zhu; Li Cao; Liqiang Nie
>
> **摘要:** Multimodal Large Language Models (MLLMs) have revolutionized video understanding, yet are still limited by context length when processing long videos. Recent methods compress videos by leveraging visual redundancy uniformly, yielding promising results. Nevertheless, our quantitative analysis shows that redundancy varies significantly across time and model layers, necessitating a more flexible compression strategy. We propose AdaReTaKe, a training-free method that flexibly reduces visual redundancy by allocating compression ratios among time and layers with theoretical guarantees. Integrated into state-of-the-art MLLMs, AdaReTaKe improves processing capacity from 256 to 2048 frames while preserving critical information. Experiments on VideoMME, MLVU, LongVideoBench, and LVBench datasets demonstrate that AdaReTaKe outperforms existing methods by 2.3% and 2.8% for 7B and 72B models, respectively, with even greater improvements of 5.9% and 6.0% on the longest LVBench. Our code is available at https://github.com/SCZwangxiao/video-FlexReduc.git.
>
---
#### [replaced 144] Modality-Specialized Synergizers for Interleaved Vision-Language Generalists
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.03604v2](http://arxiv.org/pdf/2407.03604v2)**

> **作者:** Zhiyang Xu; Minqian Liu; Ying Shen; Joy Rimchala; Jiaxin Zhang; Qifan Wang; Yu Cheng; Lifu Huang
>
> **备注:** 8 Pages, interleaved instruction tuning, parameter-efficient tuning, image understanding, image generation
>
> **摘要:** Recent advancements in Vision-Language Models (VLMs) have led to the emergence of Vision-Language Generalists (VLGs) capable of understanding and generating both text and images. However, seamlessly generating an arbitrary sequence of text and images remains a challenging task for the current VLGs. One primary limitation lies in applying a unified architecture and the same set of parameters to simultaneously model discrete text tokens and continuous image features. Recent works attempt to tackle this fundamental problem by introducing modality-aware expert models. However, they employ identical architectures to process both text and images, disregarding the intrinsic inductive biases in these two modalities. In this work, we introduce MODALITY-SPECIALIZED SYNERGIZERS (MOSS), a novel design that efficiently optimizes existing unified architectures of VLGs with modality-specialized adaptation layers, i.e., a Convolutional LoRA for modeling the local priors of image patches and a Linear LoRA for processing sequential text. This design enables more effective modeling of modality-specific features while maintaining the strong cross-modal integration gained from pretraining. In addition, to improve the instruction-following capability on interleaved text-and-image generation, we introduce LEAFINSTRUCT, the first open-sourced interleaved instruction tuning dataset comprising 184,982 high-quality instances on more than 10 diverse domains. Extensive experiments show that VLGs integrated with M OSS achieve state-of-the-art performance, significantly surpassing baseline VLGs in complex interleaved generation tasks. Furthermore, our method exhibits strong generalizability on different VLGs.
>
---
#### [replaced 145] SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.14676v3](http://arxiv.org/pdf/2410.14676v3)**

> **作者:** Qin Liu; Fei Wang; Chaowei Xiao; Muhao Chen
>
> **备注:** ACL 2025
>
> **摘要:** Existing preference alignment is a one-size-fits-all alignment mechanism, where the part of the large language model (LLM) parametric knowledge with non-preferred features is uniformly blocked to all the users. However, this part of knowledge can be useful to advanced users whose expertise qualifies them to handle these information. The one-size-fits-all alignment mechanism undermines LLM's utility for these qualified users. To address this problem, we propose SudoLM, a framework that lets LLMs learn access control over specific parametric knowledge for users with different credentials via authorization alignment. SudoLM allows authorized users to unlock their access to all the parametric knowledge with an assigned SUDO key while blocking access to non-qualified users. Experiments on two application scenarios demonstrate that SudoLM effectively controls the user's access to the parametric knowledge and maintains its general utility.
>
---
#### [replaced 146] Active Preference Optimization for Sample Efficient RLHF
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.10500v3](http://arxiv.org/pdf/2402.10500v3)**

> **作者:** Nirjhar Das; Souradip Chakraborty; Aldo Pacchiano; Sayak Ray Chowdhury
>
> **备注:** Accepted at ECML-PKDD 2025. Camera ready version
>
> **摘要:** Large Language Models (LLMs) aligned using Reinforcement Learning from Human Feedback (RLHF) have shown remarkable generation abilities in numerous tasks. However, collecting high-quality human preferences creates costly bottlenecks in practical deployments, and hence, training data are often budgeted. In these scenarios, it is crucial to collect training data (e.g., contexts, a pair of generations for each context, and a preference indicating which generation is better) carefully, yet most of the existing methods sample contexts uniformly at random from a given collection. Given this, under the Bradley-Terry-Luce preference model and with a small budget of training data, we show that uniform sampling of contexts could lead to a policy (i.e., an aligned model) that suffers a constant sub-optimality gap from the optimal policy. This highlights the need for an adaptive context sampling strategy for effective alignment under a small sample budget. To address this, we reformulate RLHF within the contextual preference bandit framework, treating generations as actions, and give a nearly complete characterization of the sub-optimality gap in terms of both lower and upper bounds. First, when the action set is a $d$-dimensional hypercube and the number of samples is $T$, we show an $\Omega(d/\sqrt{T})$ lower bound. Next, we propose an algorithm, $\textit{Active Preference Optimization}$ ($\texttt{APO}$), that iteratively collects preferences for the most uncertain contexts. We show that the sub-optimality gap of the policy learned via $\texttt{APO}$ matches the lower bound up to a log factor and a non-linearity constant. Finally, we perform experiments on practical datasets to validate $\texttt{APO}$'s efficacy over existing methods, establishing it as a sample-efficient and cost-effective solution for LLM alignment.
>
---
#### [replaced 147] Toward Reliable Scientific Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14599v2](http://arxiv.org/pdf/2505.14599v2)**

> **作者:** Guangzhi Xiong; Eric Xie; Corey Williams; Myles Kim; Amir Hassan Shariatmadari; Sikun Guo; Stefan Bekiranov; Aidong Zhang
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** Large language models (LLMs) have shown significant potential in scientific disciplines such as biomedicine, particularly in hypothesis generation, where they can analyze vast literature, identify patterns, and suggest research directions. However, a key challenge lies in evaluating the truthfulness of generated hypotheses, as verifying their accuracy often requires substantial time and resources. Additionally, the hallucination problem in LLMs can lead to the generation of hypotheses that appear plausible but are ultimately incorrect, undermining their reliability. To facilitate the systematic study of these challenges, we introduce TruthHypo, a benchmark for assessing the capabilities of LLMs in generating truthful scientific hypotheses, and KnowHD, a knowledge-based hallucination detector to evaluate how well hypotheses are grounded in existing knowledge. Our results show that LLMs struggle to generate truthful hypotheses. By analyzing hallucinations in reasoning steps, we demonstrate that the groundedness scores provided by KnowHD serve as an effective metric for filtering truthful hypotheses from the diverse outputs of LLMs. Human evaluations further validate the utility of KnowHD in identifying truthful hypotheses and accelerating scientific discovery. Our data and source code are available at https://github.com/Teddy-XiongGZ/TruthHypo.
>
---
#### [replaced 148] Data Shapley in One Training Run
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2406.11011v3](http://arxiv.org/pdf/2406.11011v3)**

> **作者:** Jiachen T. Wang; Prateek Mittal; Dawn Song; Ruoxi Jia
>
> **备注:** ICLR 2025 Outstanding Paper Runner-Up
>
> **摘要:** Data Shapley provides a principled framework for attributing data's contribution within machine learning contexts. However, existing approaches require re-training models on different data subsets, which is computationally intensive, foreclosing their application to large-scale models. Furthermore, they produce the same attribution score for any models produced by running the learning algorithm, meaning they cannot perform targeted attribution towards a specific model obtained from a single run of the algorithm. This paper introduces In-Run Data Shapley, which addresses these limitations by offering scalable data attribution for a target model of interest. In its most efficient implementation, our technique incurs negligible additional runtime compared to standard model training. This dramatic efficiency improvement makes it possible to perform data attribution for the foundation model pretraining stage for the first time. We present several case studies that offer fresh insights into pretraining data's contribution and discuss their implications for copyright in generative AI and pretraining data curation.
>
---
#### [replaced 149] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10415v2](http://arxiv.org/pdf/2504.10415v2)**

> **作者:** Parshin Shojaee; Ngoc-Hieu Nguyen; Kazem Meidani; Amir Barati Farimani; Khoa D Doan; Chandan K Reddy
>
> **备注:** ICML 2025 Oral. Project page: https://github.com/deep-symbolic-mathematics/llm-srbench , Benchmark page: https://huggingface.co/datasets/nnheui/llm-srbench
>
> **摘要:** Scientific equation discovery is a fundamental task in the history of scientific progress, enabling the derivation of laws governing natural phenomena. Recently, Large Language Models (LLMs) have gained interest for this task due to their potential to leverage embedded scientific knowledge for hypothesis generation. However, evaluating the true discovery capabilities of these methods remains challenging, as existing benchmarks often rely on common equations that are susceptible to memorization by LLMs, leading to inflated performance metrics that do not reflect discovery. In this paper, we introduce LLM-SRBench, a comprehensive benchmark with 239 challenging problems across four scientific domains specifically designed to evaluate LLM-based scientific equation discovery methods while preventing trivial memorization. Our benchmark comprises two main categories: LSR-Transform, which transforms common physical models into less common mathematical representations to test reasoning beyond memorized forms, and LSR-Synth, which introduces synthetic, discovery-driven problems requiring data-driven reasoning. Through extensive evaluation of several state-of-the-art methods, using both open and closed LLMs, we find that the best-performing system so far achieves only 31.5% symbolic accuracy. These findings highlight the challenges of scientific equation discovery, positioning LLM-SRBench as a valuable resource for future research.
>
---
#### [replaced 150] Scaling over Scaling: Exploring Test-Time Scaling Plateau in Large Reasoning Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20522v2](http://arxiv.org/pdf/2505.20522v2)**

> **作者:** Jian Wang; Boyan Zhu; Chak Tou Leong; Yongqi Li; Wenjie Li
>
> **备注:** Work in progress
>
> **摘要:** Large reasoning models (LRMs) have exhibited the capacity of enhancing reasoning performance via internal test-time scaling. Building upon this, a promising direction is to further scale test-time compute to unlock even greater reasoning capabilities. However, as we push these scaling boundaries, systematically understanding the practical limits and achieving optimal resource allocation becomes a critical challenge. In this paper, we investigate the scaling plateau of test-time scaling and introduce the Test-Time Scaling Performance Model (TTSPM). We theoretically analyze two fundamental paradigms for such extended scaling, parallel scaling and sequential scaling, from a probabilistic modeling perspective. Our primary contribution is the derivation of the saturation point on the scaling budget for both strategies, identifying thresholds beyond which additional computation yields diminishing returns. Remarkably, despite their distinct mechanisms, both paradigms converge to a unified mathematical structure in their upper bounds. We empirically validate our theoretical findings on challenging reasoning benchmarks, including AIME, MATH-500, and GPQA, demonstrating the practical utility of these bounds for test-time resource allocation. We hope that this work provides insights into the cost-benefit trade-offs of test-time scaling, guiding the development of more resource-efficient inference strategies for large reasoning models.
>
---
#### [replaced 151] RONA: Pragmatically Diverse Image Captioning with Coherence Relations
- **分类: cs.CL; cs.AI; cs.CV; 68T50; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2503.10997v2](http://arxiv.org/pdf/2503.10997v2)**

> **作者:** Aashish Anantha Ramakrishnan; Aadarsh Anantha Ramakrishnan; Dongwon Lee
>
> **备注:** Accepted in the NAACL Fourth Workshop on Intelligent and Interactive Writing Assistants (In2Writing), Albuquerque, New Mexico, May 2025, https://in2writing.glitch.me
>
> **摘要:** Writing Assistants (e.g., Grammarly, Microsoft Copilot) traditionally generate diverse image captions by employing syntactic and semantic variations to describe image components. However, human-written captions prioritize conveying a central message alongside visual descriptions using pragmatic cues. To enhance caption diversity, it is essential to explore alternative ways of communicating these messages in conjunction with visual content. We propose RONA, a novel prompting strategy for Multi-modal Large Language Models (MLLM) that leverages Coherence Relations as a controllable axis for pragmatic variations. We demonstrate that RONA generates captions with better overall diversity and ground-truth alignment, compared to MLLM baselines across multiple domains. Our code is available at: https://github.com/aashish2000/RONA
>
---
#### [replaced 152] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products
- **分类: cs.LG; cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2502.10297v5](http://arxiv.org/pdf/2502.10297v5)**

> **作者:** Julien Siems; Timur Carstensen; Arber Zela; Frank Hutter; Massimiliano Pontil; Riccardo Grazzi
>
> **备注:** Accepted at ICLR 2025 Workshop on Foundation Models in the Wild
>
> **摘要:** Linear Recurrent Neural Networks (linear RNNs) have emerged as competitive alternatives to Transformers for sequence modeling, offering efficient training and linear-time inference. However, existing architectures face a fundamental trade-off between expressivity and efficiency, dictated by the structure of their state-transition matrices. Diagonal matrices, used in models such as Mamba, GLA, or mLSTM, yield fast runtime but have limited expressivity. To address this, recent architectures such as DeltaNet and RWKV-7 adopted a diagonal plus rank-1 structure, which allows simultaneous token and channel mixing, improving associative recall and, as recently shown, state-tracking when allowing negative eigenvalues in the state-transition matrices. Building on the interpretation of DeltaNet's recurrence as performing one step of online gradient descent per token on an associative recall loss, we introduce DeltaProduct, which instead takes multiple ($n_h$) steps per token. This naturally leads to diagonal plus rank-$n_h$ state-transition matrices, formed as products of $n_h$ generalized Householder transformations, providing a tunable mechanism to balance expressivity and efficiency. We provide a detailed theoretical characterization of the state-tracking capability of DeltaProduct in finite precision, showing how it improves by increasing $n_h$. Our extensive experiments demonstrate that DeltaProduct outperforms DeltaNet in both state-tracking and language modeling, while also showing significantly improved length extrapolation capabilities.
>
---
#### [replaced 153] RomanLens: The Role Of Latent Romanization In Multilinguality In LLMs
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.07424v3](http://arxiv.org/pdf/2502.07424v3)**

> **作者:** Alan Saji; Jaavid Aktar Husain; Thanmay Jayakumar; Raj Dabre; Anoop Kunchukuttan; Ratish Puduppully
>
> **备注:** 19 pages, 19 figures
>
> **摘要:** Large Language Models (LLMs) exhibit strong multilingual performance despite being predominantly trained on English-centric corpora. This raises a fundamental question: How do LLMs achieve such multilingual capabilities? Focusing on languages written in non-Roman scripts, we investigate the role of Romanization - the representation of non-Roman scripts using Roman characters - as a potential bridge in multilingual processing. Using mechanistic interpretability techniques, we analyze next-token generation and find that intermediate layers frequently represent target words in Romanized form before transitioning to native script, a phenomenon we term Latent Romanization. Further, through activation patching experiments, we demonstrate that LLMs encode semantic concepts similarly across native and Romanized scripts, suggesting a shared underlying representation. Additionally, for translation into non-Roman script languages, our findings reveal that when the target language is in Romanized form, its representations emerge earlier in the model's layers compared to native script. These insights contribute to a deeper understanding of multilingual representation in LLMs and highlight the implicit role of Romanization in facilitating language transfer.
>
---
#### [replaced 154] Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.01968v2](http://arxiv.org/pdf/2502.01968v2)**

> **作者:** Jinlong Pang; Na Di; Zhaowei Zhu; Jiaheng Wei; Hao Cheng; Chen Qian; Yang Liu
>
> **摘要:** Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant, uninformative, or even harmful. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves downstream performance. Code is available at https://github.com/UCSC-REAL/TokenCleaning.
>
---
#### [replaced 155] WeQA: A Benchmark for Retrieval Augmented Generation in Wind Energy Domain
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.11800v3](http://arxiv.org/pdf/2408.11800v3)**

> **作者:** Rounak Meyur; Hung Phan; Sridevi Wagle; Jan Strube; Mahantesh Halappanavar; Sameera Horawalavithana; Anurag Acharya; Sai Munikoti
>
> **备注:** 8 pages without Limitation and References
>
> **摘要:** Wind energy project assessments present significant challenges for decision-makers, who must navigate and synthesize hundreds of pages of environmental and scientific documentation. These documents often span different regions and project scales, covering multiple domains of expertise. This process traditionally demands immense time and specialized knowledge from decision-makers. The advent of Large Language Models (LLM) and Retrieval Augmented Generation (RAG) approaches offer a transformative solution, enabling rapid, accurate cross-document information retrieval and synthesis. As the landscape of Natural Language Processing (NLP) and text generation continues to evolve, benchmarking becomes essential to evaluate and compare the performance of different RAG-based LLMs. In this paper, we present a comprehensive framework to generate a domain relevant RAG benchmark. Our framework is based on automatic question-answer generation with Human (domain experts)-AI (LLM) teaming. As a case study, we demonstrate the framework by introducing WeQA, a first-of-its-kind benchmark on the wind energy domain which comprises of multiple scientific documents/reports related to environmental aspects of wind energy projects. Our framework systematically evaluates RAG performance using diverse metrics and multiple question types with varying complexity level, providing a foundation for rigorous assessment of RAG-based systems in complex scientific domains and enabling researchers to identify areas for improvement in domain-specific applications.
>
---
#### [replaced 156] Highly Fast Text Segmentation With Pairwise Markov Chains
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2102.11037v2](http://arxiv.org/pdf/2102.11037v2)**

> **作者:** Elie Azeraf; Emmanuel Monfrini; Emmanuel Vignon; Wojciech Pieczynski
>
> **备注:** 9 pages, 5 figures, 4 tables, MNLP 2020
>
> **摘要:** Natural Language Processing (NLP) models' current trend consists of using increasingly more extra-data to build the best models as possible. It implies more expensive computational costs and training time, difficulties for deployment, and worries about these models' carbon footprint reveal a critical problem in the future. Against this trend, our goal is to develop NLP models requiring no extra-data and minimizing training time. To do so, in this paper, we explore Markov chain models, Hidden Markov Chain (HMC) and Pairwise Markov Chain (PMC), for NLP segmentation tasks. We apply these models for three classic applications: POS Tagging, Named-Entity-Recognition, and Chunking. We develop an original method to adapt these models for text segmentation's specific challenges to obtain relevant performances with very short training and execution times. PMC achieves equivalent results to those obtained by Conditional Random Fields (CRF), one of the most applied models for these tasks when no extra-data are used. Moreover, PMC has training times 30 times shorter than the CRF ones, which validates this model given our objectives.
>
---
#### [replaced 157] Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05339v2](http://arxiv.org/pdf/2506.05339v2)**

> **作者:** Anirudh Bharadwaj; Chaitanya Malaviya; Nitish Joshi; Mark Yatskar
>
> **备注:** Code and data available at https://github.com/anirudhb123/preference-model-biases
>
> **摘要:** Language models serve as proxies for human preference judgements in alignment and evaluation, yet they exhibit systematic miscalibration, prioritizing superficial patterns over substantive qualities. This bias manifests as overreliance on features like length, structure, and style, leading to issues like reward hacking and unreliable evaluations. Evidence suggests these biases originate in artifacts in human training data. In this work, we systematically investigate the relationship between training data biases and preference model miscalibration across five idiosyncratic features of language model generations: length, structure, jargon, sycophancy and vagueness. Using controlled counterfactual pairs, we first quantify the extent to which preference models favor responses with magnified biases (skew), finding this preference occurs in >60% of instances, and model preferences show high miscalibration (~40%) compared to human preferences. Notably, bias features only show mild negative correlations to human preference labels (mean r_human = -0.12) but show moderately strong positive correlations with labels from a strong reward model (mean r_model = +0.36), suggesting that models may overrely on spurious cues. To mitigate these issues, we propose a simple post-training method based on counterfactual data augmentation (CDA) using synthesized contrastive examples. Finetuning models with CDA reduces average miscalibration from 39.4% to 32.5% and average absolute skew difference from 20.5% to 10.0%, while maintaining overall RewardBench performance, showing that targeted debiasing is effective for building reliable preference models.
>
---
#### [replaced 158] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling
- **分类: cs.LG; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2503.02445v4](http://arxiv.org/pdf/2503.02445v4)**

> **作者:** Hao Li; Yu-Hao Huang; Chang Xu; Viktor Schlegel; Renhe Jiang; Riza Batista-Navarro; Goran Nenadic; Jiang Bian
>
> **备注:** ICML 2025 Main Conference
>
> **摘要:** Time-series Generation (TSG) is a prominent research area with broad applications in simulations, data augmentation, and counterfactual analysis. While existing methods have shown promise in unconditional single-domain TSG, real-world applications demand for cross-domain approaches capable of controlled generation tailored to domain-specific constraints and instance-level requirements. In this paper, we argue that text can provide semantic insights, domain information and instance-specific temporal patterns, to guide and improve TSG. We introduce ``Text-Controlled TSG'', a task focused on generating realistic time series by incorporating textual descriptions. To address data scarcity in this setting, we propose a novel LLM-based Multi-Agent framework that synthesizes diverse, realistic text-to-TS datasets. Furthermore, we introduce BRIDGE, a hybrid text-controlled TSG framework that integrates semantic prototypes with text description for supporting domain-level guidance. This approach achieves state-of-the-art generation fidelity on 11 of 12 datasets, and improves controllability by up to 12% on MSE and 6% MAE compared to no text input generation, highlighting its potential for generating tailored time-series data.
>
---
#### [replaced 159] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11221v2](http://arxiv.org/pdf/2502.11221v2)**

> **作者:** Hui Wei; Zihao Zhang; Shenghua He; Tian Xia; Shijia Pan; Fei Liu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows.
>
---
#### [replaced 160] Compliance-to-Code: Enhancing Financial Compliance Checking via Code Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19804v2](http://arxiv.org/pdf/2505.19804v2)**

> **作者:** Siyuan Li; Jian Chen; Rui Yao; Xuming Hu; Peilin Zhou; Weihua Qiu; Simin Zhang; Chucheng Dong; Zhiyao Li; Qipeng Xie; Zixuan Yuan
>
> **摘要:** Nowadays, regulatory compliance has become a cornerstone of corporate governance, ensuring adherence to systematic legal frameworks. At its core, financial regulations often comprise highly intricate provisions, layered logical structures, and numerous exceptions, which inevitably result in labor-intensive or comprehension challenges. To mitigate this, recent Regulatory Technology (RegTech) and Large Language Models (LLMs) have gained significant attention in automating the conversion of regulatory text into executable compliance logic. However, their performance remains suboptimal particularly when applied to Chinese-language financial regulations, due to three key limitations: (1) incomplete domain-specific knowledge representation, (2) insufficient hierarchical reasoning capabilities, and (3) failure to maintain temporal and logical coherence. One promising solution is to develop a domain specific and code-oriented datasets for model training. Existing datasets such as LexGLUE, LegalBench, and CODE-ACCORD are often English-focused, domain-mismatched, or lack fine-grained granularity for compliance code generation. To fill these gaps, we present Compliance-to-Code, the first large-scale Chinese dataset dedicated to financial regulatory compliance. Covering 1,159 annotated clauses from 361 regulations across ten categories, each clause is modularly structured with four logical elements-subject, condition, constraint, and contextual information-along with regulation relations. We provide deterministic Python code mappings, detailed code reasoning, and code explanations to facilitate automated auditing. To demonstrate utility, we present FinCheck: a pipeline for regulation structuring, code generation, and report generation.
>
---
#### [replaced 161] On Adversarial Robustness of Language Models in Transfer Learning
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00066v2](http://arxiv.org/pdf/2501.00066v2)**

> **作者:** Bohdan Turbal; Anastasiia Mazur; Jiaxu Zhao; Mykola Pechenizkiy
>
> **摘要:** We investigate the adversarial robustness of LLMs in transfer learning scenarios. Through comprehensive experiments on multiple datasets (MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias) and various model architectures (BERT, RoBERTa, GPT-2, Gemma, Phi), we reveal that transfer learning, while improving standard performance metrics, often leads to increased vulnerability to adversarial attacks. Our findings demonstrate that larger models exhibit greater resilience to this phenomenon, suggesting a complex interplay between model size, architecture, and adaptation methods. Our work highlights the crucial need for considering adversarial robustness in transfer learning scenarios and provides insights into maintaining model security without compromising performance. These findings have significant implications for the development and deployment of LLMs in real-world applications where both performance and robustness are paramount.
>
---
#### [replaced 162] Taxonomizing Representational Harms using Speech Act Theory
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2504.00928v2](http://arxiv.org/pdf/2504.00928v2)**

> **作者:** Emily Corvi; Hannah Washington; Stefanie Reed; Chad Atalla; Alexandra Chouldechova; P. Alex Dow; Jean Garcia-Gathright; Nicholas Pangakis; Emily Sheng; Dan Vann; Matthew Vogel; Hanna Wallach
>
> **备注:** In Findings of the Association for Computational Linguistics (ACL), 2025
>
> **摘要:** Representational harms are widely recognized among fairness-related harms caused by generative language systems. However, their definitions are commonly under-specified. We make a theoretical contribution to the specification of representational harms by introducing a framework, grounded in speech act theory (Austin, 1962), that conceptualizes representational harms caused by generative language systems as the perlocutionary effects (i.e., real-world impacts) of particular types of illocutionary acts (i.e., system behaviors). Building on this argument and drawing on relevant literature from linguistic anthropology and sociolinguistics, we provide new definitions of stereotyping, demeaning, and erasure. We then use our framework to develop a granular taxonomy of illocutionary acts that cause representational harms, going beyond the high-level taxonomies presented in previous work. We also discuss the ways that our framework and taxonomy can support the development of valid measurement instruments. Finally, we demonstrate the utility of our framework and taxonomy via a case study that engages with recent conceptual debates about what constitutes a representational harm and how such harms should be measured.
>
---
#### [replaced 163] Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs with Semantic Space
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11586v2](http://arxiv.org/pdf/2503.11586v2)**

> **作者:** Zhiliang Chen; Xinyuan Niu; Chuan-Sheng Foo; Bryan Kian Hsiang Low
>
> **备注:** ICLR 2025 Spotlight
>
> **摘要:** Large language models (LLMs) are used in chatbots or AI assistants to hold conversations with a human user. In such applications, the quality (e.g., user engagement, safety) of a conversation is important and can only be exactly known at the end of the conversation. To maximize its expected quality, conversation planning reasons about the stochastic transitions within a conversation to select the optimal LLM response at each turn. Existing simulation-based conversation planning algorithms typically select the optimal response by simulating future conversations with a large number of LLM queries at every turn. However, this process is extremely time-consuming and hence impractical for real-time conversations. This paper presents a novel approach called Semantic space COnversation Planning with improved Efficiency (SCOPE) that exploits the dense semantic representation of conversations to perform conversation planning efficiently. In particular, SCOPE models the stochastic transitions in conversation semantics and their associated rewards to plan entirely within the semantic space. This allows us to select the optimal LLM response at every conversation turn without needing additional LLM queries for simulation. As a result, SCOPE can perform conversation planning 70 times faster than conventional simulation-based planning algorithms when applied to a wide variety of conversation starters and two reward functions seen in the real world, yet achieving a higher reward within a practical planning budget. Our code can be found at: https://github.com/chenzhiliang94/convo-plan-SCOPE.
>
---
#### [replaced 164] Grounded Persuasive Language Generation for Automated Marketing
- **分类: cs.AI; cs.CL; cs.HC; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2502.16810v3](http://arxiv.org/pdf/2502.16810v3)**

> **作者:** Jibang Wu; Chenghao Yang; Simon Mahns; Yi Wu; Chaoqi Wang; Hao Zhu; Fei Fang; Haifeng Xu
>
> **摘要:** This paper develops an agentic framework that employs large language models (LLMs) to automate the generation of persuasive and grounded marketing content, using real estate listing descriptions as our focal application domain. Our method is designed to align the generated content with user preferences while highlighting useful factual attributes. This agent consists of three key modules: (1) Grounding Module, mimicking expert human behavior to predict marketable features; (2) Personalization Module, aligning content with user preferences; (3) Marketing Module, ensuring factual accuracy and the inclusion of localized features. We conduct systematic human-subject experiments in the domain of real estate marketing, with a focus group of potential house buyers. The results demonstrate that marketing descriptions generated by our approach are preferred over those written by human experts by a clear margin while maintaining the same level of factual accuracy. Our findings suggest a promising agentic approach to automate large-scale targeted marketing while ensuring factuality of content generation.
>
---
#### [replaced 165] General-Reasoner: Advancing LLM Reasoning Across All Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14652v5](http://arxiv.org/pdf/2505.14652v5)**

> **作者:** Xueguang Ma; Qian Liu; Dongfu Jiang; Ge Zhang; Zejun Ma; Wenhu Chen
>
> **摘要:** Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
>
---
#### [replaced 166] EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17654v2](http://arxiv.org/pdf/2505.17654v2)**

> **作者:** Ancheng Xu; Zhihao Yang; Jingpeng Li; Guanghu Yuan; Longze Chen; Liang Yan; Jiehui Zhou; Zhen Qin; Hengyun Chang; Hamid Alinejad-Rokny; Bo Zheng; Min Yang
>
> **摘要:** E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.
>
---
#### [replaced 167] Diversity of Transformer Layers: One Aspect of Parameter Scaling Laws
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.24009v2](http://arxiv.org/pdf/2505.24009v2)**

> **作者:** Hidetaka Kamigaito; Ying Zhang; Jingun Kwon; Katsuhiko Hayashi; Manabu Okumura; Taro Watanabe
>
> **摘要:** Transformers deliver outstanding performance across a wide range of tasks and are now a dominant backbone architecture for large language models (LLMs). Their task-solving performance is improved by increasing parameter size, as shown in the recent studies on parameter scaling laws. Although recent mechanistic-interpretability studies have deepened our understanding of the internal behavior of Transformers by analyzing their residual stream, the relationship between these internal mechanisms and the parameter scaling laws remains unclear. To bridge this gap, we focus on layers and their size, which mainly decide the parameter size of Transformers. For this purpose, we first theoretically investigate the layers within the residual stream through a bias-diversity decomposition. The decomposition separates (i) bias, the error of each layer's output from the ground truth, and (ii) diversity, which indicates how much the outputs of each layer differ from each other. Analyzing Transformers under this theory reveals that performance improves when individual layers make predictions close to the correct answer and remain mutually diverse. We show that diversity becomes especially critical when individual layers' outputs are far from the ground truth. Finally, we introduce an information-theoretic diversity and show our main findings that adding layers enhances performance only when those layers behave differently, i.e., are diverse. We also reveal the performance gains from increasing the number of layers exhibit submodularity: marginal improvements diminish as additional layers increase, mirroring the logarithmic convergence predicted by the parameter scaling laws. Experiments on multiple semantic-understanding tasks with various LLMs empirically confirm the theoretical properties derived in this study.
>
---
#### [replaced 168] ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.13397v5](http://arxiv.org/pdf/2501.13397v5)**

> **作者:** Kangjie Zheng; Junwei Yang; Siyue Liang; Bin Feng; Zequn Liu; Wei Ju; Zhiping Xiao; Ming Zhang
>
> **备注:** 30 pages, 12 figures; ICML 2025
>
> **摘要:** Masked Language Models (MLMs) have achieved remarkable success in many self-supervised representation learning tasks. MLMs are trained by randomly masking portions of the input sequences with [MASK] tokens and learning to reconstruct the original content based on the remaining context. This paper explores the impact of [MASK] tokens on MLMs. Analytical studies show that masking tokens can introduce the corrupted semantics problem, wherein the corrupted context may convey multiple, ambiguous meanings. This problem is also a key factor affecting the performance of MLMs on downstream tasks. Based on these findings, we propose a novel enhanced-context MLM, ExLM. Our approach expands [MASK] tokens in the input context and models the dependencies between these expanded states. This enhancement increases context capacity and enables the model to capture richer semantic information, effectively mitigating the corrupted semantics problem during pre-training. Experimental results demonstrate that ExLM achieves significant performance improvements in both text modeling and SMILES modeling tasks. Further analysis confirms that ExLM enriches semantic representations through context enhancement, and effectively reduces the semantic multimodality commonly observed in MLMs.
>
---
