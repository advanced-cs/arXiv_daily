# 自然语言处理 cs.CL

- **最新发布 86 篇**

- **更新 47 篇**

## 最新发布

#### [new 001] A novel language model for predicting serious adverse event results in clinical trials from their prospective registrations
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在预测临床试验中的严重不良事件（SAE）结果。利用临床试验注册信息，通过预训练语言模型结合滑动窗口方法提取特征，构建分类与回归模型，预测试验组与对照组的SAE发生情况及比例，提升临床试验设计安全性。**

- **链接: [http://arxiv.org/pdf/2507.22919v1](http://arxiv.org/pdf/2507.22919v1)**

> **作者:** Qixuan Hu; Xumou Zhang; Jinman Kim; Florence Bourgeois; Adam G. Dunn
>
> **摘要:** Objectives: With accurate estimates of expected safety results, clinical trials could be designed to avoid terminations and limit exposing participants to unnecessary risks. We evaluated methods for predicting serious adverse event (SAE) results in clinical trials using information only from their registrations prior to the trial. Material and Methods: We analysed 22,107 two-arm parallel interventional clinical trials from ClinicalTrials.gov with structured summary results. Two prediction models were developed: a classifier predicting will experimental arm have higher SAE rates (area under the receiver operating characteristic curve; AUC) than control arm, and a regression model to predict the proportion of SAEs in control arms (root mean squared error; RMSE). A transfer learning approach using pretrained language models (e.g., ClinicalT5, BioBERT) was used for feature extraction, combined with downstream model for prediction. To maintain semantic representation in long trial texts exceeding localised language model input limits, a sliding window method was developed for embedding extraction. Results: The best model (ClinicalT5+Transformer+MLP) had 77.6% AUC predicting which trial arm has a higher proportion of patients with SAEs. When predicting proportion of participants experiencing SAE in the control arm, the same model achieved RMSE of 18.6%. The sliding window approach consistently outperformed methods without it. Across 12 classifiers, the average absolute AUC increase was 2.00%; across 12 regressors, the average absolute RMSE reduction was 1.58%. Discussion: Summary results data available at ClinicalTrials.gov remains underutilised. The potential to estimate results of trials before they start is an opportunity to improve trial design and flag discrepancies between expected and reported safety results.
>
---
#### [new 002] A Novel Evaluation Benchmark for Medical LLMs: Illuminating Safety and Effectiveness in Clinical Domains
- **分类: cs.CL**

- **简介: 该论文属于医疗人工智能评估任务，旨在解决医学大语言模型（LLM）在临床应用中的安全性与有效性评估问题。研究构建了包含30项标准的临床安全-有效性双轨基准（CSEDB），由32名专科医师参与制定2,069个问答项，覆盖26个临床科室。测试显示，模型整体表现中等，专业医学LLM优于通用模型，尤其在高风险场景中表现更优，有助于推动LLM在医疗领域的安全有效应用。**

- **链接: [http://arxiv.org/pdf/2507.23486v1](http://arxiv.org/pdf/2507.23486v1)**

> **作者:** Shirui Wang; Zhihui Tang; Huaxia Yang; Qiuhong Gong; Tiantian Gu; Hongyang Ma; Yongxin Wang; Wubin Sun; Zeliang Lian; Kehang Mao; Yinan Jiang; Zhicheng Huang; Lingyun Ma; Wenjie Shen; Yajie Ji; Yunhui Tan; Chunbo Wang; Yunlu Gao; Qianling Ye; Rui Lin; Mingyu Chen; Lijuan Niu; Zhihao Wang; Peng Yu; Mengran Lang; Yue Liu; Huimin Zhang; Haitao Shen; Long Chen; Qiguang Zhao; Si-Xuan Liu; Lina Zhou; Hua Gao; Dongqiang Ye; Lingmin Meng; Youtao Yu; Naixin Liang; Jianxiong Wu
>
> **摘要:** Large language models (LLMs) hold promise in clinical decision support but face major challenges in safety evaluation and effectiveness validation. We developed the Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB), a multidimensional framework built on clinical expert consensus, encompassing 30 criteria covering critical areas like critical illness recognition, guideline adherence, and medication safety, with weighted consequence measures. Thirty-two specialist physicians developed and reviewed 2,069 open-ended Q&A items aligned with these criteria, spanning 26 clinical departments to simulate real-world scenarios. Benchmark testing of six LLMs revealed moderate overall performance (average total score 57.2%, safety 54.7%, effectiveness 62.3%), with a significant 13.3% performance drop in high-risk scenarios (p < 0.0001). Domain-specific medical LLMs showed consistent performance advantages over general-purpose models, with relatively higher top scores in safety (0.912) and effectiveness (0.861). The findings of this study not only provide a standardized metric for evaluating the clinical application of medical LLMs, facilitating comparative analyses, risk exposure identification, and improvement directions across different scenarios, but also hold the potential to promote safer and more effective deployment of large language models in healthcare environments.
>
---
#### [new 003] Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了大语言模型（LLMs）在处理中文文本歧义时的可信度问题。任务是文本消歧，旨在解决LLMs在面对歧义文本时表现出的脆弱性。作者构建了一个包含歧义句及其消歧配对的基准数据集，并发现LLMs在理解歧义时存在过自信、难以识别多义性等问题，揭示了其与人类理解行为的差异。**

- **链接: [http://arxiv.org/pdf/2507.23121v1](http://arxiv.org/pdf/2507.23121v1)**

> **作者:** Xinwei Wu; Haojie Li; Hongyu Liu; Xinyu Ji; Ruohan Li; Yule Chen; Yigeng Zhang
>
> **备注:** Accepted at KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models (Agentic & GenAI Evaluation Workshop KDD '25)
>
> **摘要:** In this work, we study a critical research problem regarding the trustworthiness of large language models (LLMs): how LLMs behave when encountering ambiguous narrative text, with a particular focus on Chinese textual ambiguity. We created a benchmark dataset by collecting and generating ambiguous sentences with context and their corresponding disambiguated pairs, representing multiple possible interpretations. These annotated examples are systematically categorized into 3 main categories and 9 subcategories. Through experiments, we discovered significant fragility in LLMs when handling ambiguity, revealing behavior that differs substantially from humans. Specifically, LLMs cannot reliably distinguish ambiguous text from unambiguous text, show overconfidence in interpreting ambiguous text as having a single meaning rather than multiple meanings, and exhibit overthinking when attempting to understand the various possible meanings. Our findings highlight a fundamental limitation in current LLMs that has significant implications for their deployment in real-world applications where linguistic ambiguity is common, calling for improved approaches to handle uncertainty in language understanding. The dataset and code are publicly available at this GitHub repository: https://github.com/ictup/LLM-Chinese-Textual-Disambiguation.
>
---
#### [new 004] Multi-Relation Extraction in Entity Pairs using Global Context
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文档级关系抽取任务，旨在解决实体在文档中多处出现且关系可能变化的问题。通过构建全局上下文建模，提出新的输入嵌入方法，捕捉实体在整个文档中的位置信息，实现更准确的跨句子关系预测，提升了多句子推理和全局上下文建模的能力。**

- **链接: [http://arxiv.org/pdf/2507.22926v1](http://arxiv.org/pdf/2507.22926v1)**

> **作者:** Nilesh; Atul Gupta; Avinash C Panday
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** In document-level relation extraction, entities may appear multiple times in a document, and their relationships can shift from one context to another. Accurate prediction of the relationship between two entities across an entire document requires building a global context spanning all relevant sentences. Previous approaches have focused only on the sentences where entities are mentioned, which fails to capture the complete document context necessary for accurate relation extraction. Therefore, this paper introduces a novel input embedding approach to capture the positions of mentioned entities throughout the document rather than focusing solely on the span where they appear. The proposed input encoding approach leverages global relationships and multi-sentence reasoning by representing entities as standalone segments, independent of their positions within the document. The performance of the proposed method has been tested on three benchmark relation extraction datasets, namely DocRED, Re-DocRED, and REBEL. The experimental results demonstrated that the proposed method accurately predicts relationships between entities in a document-level setting. The proposed research also has theoretical and practical implications. Theoretically, it advances global context modeling and multi-sentence reasoning in document-level relation extraction. Practically, it enhances relationship detection, enabling improved performance in real-world NLP applications requiring comprehensive entity-level insights and interpretability.
>
---
#### [new 005] Math Natural Language Inference: this should be easy!
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文研究数学领域的自然语言推理（NLI）任务，构建了一个数学NLI数据集，评估当前大语言模型（LLMs）在数学文本上的推理能力。论文探讨了LLMs在该任务中的表现与人类标注的一致性，并指出其在基本推理上仍存在不足。同时，作者提供了相关数据集以支持后续研究。**

- **链接: [http://arxiv.org/pdf/2507.23063v1](http://arxiv.org/pdf/2507.23063v1)**

> **作者:** Valeria de Paiva; Qiyue Gao; Hai Hu; Pavel Kovalev; Yikang Liu; Lawrence S. Moss; Zhiheng Qian
>
> **备注:** 9 pages plus appendices
>
> **摘要:** We ask whether contemporary LLMs are able to perform natural language inference (NLI) tasks on mathematical texts. We call this the Math NLI problem. We construct a corpus of Math NLI pairs whose premises are from extant mathematical text and whose hypotheses and gold labels were provided by people with experience in both research-level mathematics and also in the NLI field. We also investigate the quality of corpora using the same premises but whose hypotheses are provided by LLMs themselves. We not only investigate the performance but also the inter-group consistency of the diverse group of LLMs. We have both positive and negative findings. Among our positive findings: in some settings, using a majority vote of LLMs is approximately equivalent to using human-labeled data in the Math NLI area. On the negative side: LLMs still struggle with mathematical language. They occasionally fail at even basic inferences. Current models are not as prone to hypothesis-only "inference" in our data the way the previous generation had been. In addition to our findings, we also provide our corpora as data to support future work on Math NLI.
>
---
#### [new 006] Theoretical Foundations and Mitigation of Hallucination in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的幻觉问题。论文提出了幻觉的形式化定义和理论分析，区分了内在与外在幻觉，定义了幻觉风险，并推导了其学习理论边界。同时，论文综述了幻觉的检测与缓解策略，并提出了统一的工作流程，最后推荐了评估协议和数据集，以量化和减少幻觉。**

- **链接: [http://arxiv.org/pdf/2507.22915v1](http://arxiv.org/pdf/2507.22915v1)**

> **作者:** Esmail Gumaan
>
> **备注:** 12 pages
>
> **摘要:** Hallucination in Large Language Models (LLMs) refers to the generation of content that is not faithful to the input or the real-world facts. This paper provides a rigorous treatment of hallucination in LLMs, including formal definitions and theoretical analyses. We distinguish between intrinsic and extrinsic hallucinations, and define a \textit{hallucination risk} for models. We derive bounds on this risk using learning-theoretic frameworks (PAC-Bayes and Rademacher complexity). We then survey detection strategies for hallucinations, such as token-level uncertainty estimation, confidence calibration, and attention alignment checks. On the mitigation side, we discuss approaches including retrieval-augmented generation, hallucination-aware fine-tuning, logit calibration, and the incorporation of fact-verification modules. We propose a unified detection and mitigation workflow, illustrated with a diagram, to integrate these strategies. Finally, we outline evaluation protocols for hallucination, recommending datasets, metrics, and experimental setups to quantify and reduce hallucinations. Our work lays a theoretical foundation and practical guidelines for addressing the crucial challenge of hallucination in LLMs.
>
---
#### [new 007] Deep Learning Approaches for Multimodal Intent Recognition: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态意图识别任务，旨在通过深度学习方法识别用户潜在意图。论文综述了从单模态到多模态技术的发展，涵盖相关数据集、方法、应用及挑战，重点探讨基于Transformer模型的突破，为研究人员提供最新进展与未来方向。**

- **链接: [http://arxiv.org/pdf/2507.22934v1](http://arxiv.org/pdf/2507.22934v1)**

> **作者:** Jingwei Zhao; Yuhua Wen; Qifei Li; Minchi Hu; Yingying Zhou; Jingyao Xue; Junyang Wu; Yingming Gao; Zhengqi Wen; Jianhua Tao; Ya Li
>
> **备注:** Submitted to ACM Computing Surveys
>
> **摘要:** Intent recognition aims to identify users' underlying intentions, traditionally focusing on text in natural language processing. With growing demands for natural human-computer interaction, the field has evolved through deep learning and multimodal approaches, incorporating data from audio, vision, and physiological signals. Recently, the introduction of Transformer-based models has led to notable breakthroughs in this domain. This article surveys deep learning methods for intent recognition, covering the shift from unimodal to multimodal techniques, relevant datasets, methodologies, applications, and current challenges. It provides researchers with insights into the latest developments in multimodal intent recognition (MIR) and directions for future research.
>
---
#### [new 008] RASL: Retrieval Augmented Schema Linking for Massive Database Text-to-SQL
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数据库领域文本到SQL任务，旨在解决大规模数据库中自然语言接口的扩展性问题。现有方法依赖领域微调且忽略元数据语义。作者提出RASL，通过分解并索引数据库模式与元数据，实现高效表识别与列级信息利用。实验表明该方法在不同结构与元数据的大规模数据库中表现优异，无需微调即可部署。**

- **链接: [http://arxiv.org/pdf/2507.23104v1](http://arxiv.org/pdf/2507.23104v1)**

> **作者:** Jeffrey Eben; Aitzaz Ahmad; Stephen Lau
>
> **摘要:** Despite advances in large language model (LLM)-based natural language interfaces for databases, scaling to enterprise-level data catalogs remains an under-explored challenge. Prior works addressing this challenge rely on domain-specific fine-tuning - complicating deployment - and fail to leverage important semantic context contained within database metadata. To address these limitations, we introduce a component-based retrieval architecture that decomposes database schemas and metadata into discrete semantic units, each separately indexed for targeted retrieval. Our approach prioritizes effective table identification while leveraging column-level information, ensuring the total number of retrieved tables remains within a manageable context budget. Experiments demonstrate that our method maintains high recall and accuracy, with our system outperforming baselines over massive databases with varying structure and available metadata. Our solution enables practical text-to-SQL systems deployable across diverse enterprise settings without specialized fine-tuning, addressing a critical scalability gap in natural language database interfaces.
>
---
#### [new 009] P-ReMIS: Pragmatic Reasoning in Mental Health and a Social Implication
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与心理健康交叉任务，旨在解决大型语言模型在心理健康对话中的可解释性与社会影响问题。作者构建了P-ReMe数据集，定义了心理健康中的语用推理现象，并提出相关任务与模型评估方法。**

- **链接: [http://arxiv.org/pdf/2507.23247v1](http://arxiv.org/pdf/2507.23247v1)**

> **作者:** Sneha Oram; Pushpak Bhattacharyya
>
> **摘要:** There has been an increase in recent advancements in the explainability and development of personalized chatbots for mental health. However, the reasoning aspects for explainability and dialogue discourse have not been explored previously for mental health. Hence, we are investigating the pragmatic reasoning capability of large language models (LLMs) in this domain. We introduce P-ReMe dataset, and propose a modified definition for the pragmatic phenomena of implicature (implied meaning) and presupposition (implicit assumption) in mental health. Following the definition, we formulate two tasks in implicature and one task in presupposition. To benchmark the dataset and the presented tasks, we consider four models - Llama3.1, Mistral, MentaLLaMa, and Qwen. The results of the experiments suggest that Mistral and Qwen show substantial reasoning capabilities in the domain. In addition, we also propose StiPRompts to study the stigma around mental health with the state-of-the-art LLMs, GPT-4o mini, Deepseek-chat, and Claude-3.5-haiku. Our evaluated findings show that Claude-3.5-haiku deals with the stigma more responsibly compared to the other two LLMs.
>
---
#### [new 010] Unveiling Super Experts in Mixture-of-Experts Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究MoE大语言模型中的“超级专家”（SEs），旨在解决如何有效压缩模型并保持性能的问题。通过分析发现，SEs在推理中起关键作用，尤其影响数学推理等任务。论文分析了SEs的激活特性及其对注意力机制的影响，揭示了其重要性。**

- **链接: [http://arxiv.org/pdf/2507.23279v1](http://arxiv.org/pdf/2507.23279v1)**

> **作者:** Zunhai Su; Qingyuan Li; Hao Zhang; YuLei Qian; Yuchen Xie; Kehong Yuan
>
> **摘要:** Sparsely activated Mixture-of-Experts (MoE) models have shown promise in enhancing the learning capacity of large language models (LLMs). Leveraging the intrinsic importance differences among experts, recent research has explored expert-level compression techniques to improve the efficiency of MoE LLMs. However, existing approaches often rely on empirical criteria to identify critical experts, lacking a deeper exploration and understanding of the heterogeneous importance of experts. In this study, we present the first discovery and investigation of a distinct subset of experts that play a crucial role in the underlying mechanisms during the model's forward inference. These experts are prevalent in open-source MoE LLMs, and despite their limited number, pruning them leads to a significant decline in model performance (e.g., pruning three causes Qwen3-30B-A3B to produce repetitive and uninformative outputs). We refer to these experts as Super Experts (SEs). Our comprehensive analysis provides progressively deeper insights into SEs. (i) SEs are characterized by rare but extreme activation outliers in the output of the down_proj, which give rise to massive activations in the hidden states between decoder layers. Moreover, the distribution of SEs remains model-specific and is unaffected by post-training processes. (ii) By pruning SEs, we assess their significance across a variety of tasks, revealing their considerable impact on the model's overall performance, particularly in mathematical reasoning. (iii) We further enhance our understanding of the influence of SEs compression. Our findings confirm that MoE LLMs rely on SEs to induce attention sinks, which are crucial for the distribution of attention scores but are significantly disrupted by SE pruning. The code is available at https://github.com/ZunhaiSu/Super-Experts-Profilling.
>
---
#### [new 011] Full Triple Matcher: Integrating all triple elements between heterogeneous Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于知识图谱集成任务，旨在解决异构知识图谱间的实体、谓词及三元组匹配问题，特别是关注上下文匹配的不足。通过标签匹配与三元组匹配方法，结合字符串处理、模糊匹配与向量相似度技术，提升实体匹配准确率，并提出了新数据集以评估三元组匹配效果。**

- **链接: [http://arxiv.org/pdf/2507.22914v1](http://arxiv.org/pdf/2507.22914v1)**

> **作者:** Victor Eiti Yamamoto; Hideaki Takeda
>
> **摘要:** Knowledge graphs (KGs) are powerful tools for representing and reasoning over structured information. Their main components include schema, identity, and context. While schema and identity matching are well-established in ontology and entity matching research, context matching remains largely unexplored. This is particularly important because real-world KGs often vary significantly in source, size, and information density - factors not typically represented in the datasets on which current entity matching methods are evaluated. As a result, existing approaches may fall short in scenarios where diverse and complex contexts need to be integrated. To address this gap, we propose a novel KG integration method consisting of label matching and triple matching. We use string manipulation, fuzzy matching, and vector similarity techniques to align entity and predicate labels. Next, we identify mappings between triples that convey comparable information, using these mappings to improve entity-matching accuracy. Our approach demonstrates competitive performance compared to leading systems in the OAEI competition and against supervised methods, achieving high accuracy across diverse test cases. Additionally, we introduce a new dataset derived from the benchmark dataset to evaluate the triple-matching step more comprehensively.
>
---
#### [new 012] Rule2Text: Natural Language Explanation of Logical Rules in Knowledge Graphs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识图谱任务，旨在解决逻辑规则难以理解的问题。通过使用大语言模型生成逻辑规则的自然语言解释，采用AMIE算法提取规则，并测试多种提示策略。论文评估了生成解释的正确性与清晰度，并探索了自动评判的可行性。**

- **链接: [http://arxiv.org/pdf/2507.23740v1](http://arxiv.org/pdf/2507.23740v1)**

> **作者:** Nasim Shirvani-Mahdavi; Devin Wingfield; Amin Ghasemi; Chengkai Li
>
> **摘要:** Knowledge graphs (KGs) often contain sufficient information to support the inference of new facts. Identifying logical rules not only improves the completeness of a knowledge graph but also enables the detection of potential errors, reveals subtle data patterns, and enhances the overall capacity for reasoning and interpretation. However, the complexity of such rules, combined with the unique labeling conventions of each KG, can make them difficult for humans to understand. In this paper, we explore the potential of large language models to generate natural language explanations for logical rules. Specifically, we extract logical rules using the AMIE 3.5.1 rule discovery algorithm from the benchmark dataset FB15k-237 and two large-scale datasets, FB-CVT-REV and FB+CVT-REV. We examine various prompting strategies, including zero- and few-shot prompting, including variable entity types, and chain-of-thought reasoning. We conduct a comprehensive human evaluation of the generated explanations based on correctness, clarity, and hallucination, and also assess the use of large language models as automatic judges. Our results demonstrate promising performance in terms of explanation correctness and clarity, although several challenges remain for future research. All scripts and data used in this study are publicly available at https://github.com/idirlab/KGRule2NL}{https://github.com/idirlab/KGRule2NL.
>
---
#### [new 013] Evaluating LLMs' Multilingual Capabilities for Bengali: Benchmark Creation and Performance Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决孟加拉语在多语言模型中的表现不佳问题。作者构建了基准数据集，评估了10个开源大语言模型的表现，分析了其在孟加拉语上的性能差距与失败原因，提出了改进方向。**

- **链接: [http://arxiv.org/pdf/2507.23248v1](http://arxiv.org/pdf/2507.23248v1)**

> **作者:** Shimanto Bhowmik; Tawsif Tashwar Dipto; Md Sazzad Islam; Sheryl Hsu; Tahsin Reasat
>
> **摘要:** Bengali is an underrepresented language in NLP research. However, it remains a challenge due to its unique linguistic structure and computational constraints. In this work, we systematically investigate the challenges that hinder Bengali NLP performance by focusing on the absence of standardized evaluation benchmarks. We then evaluated 10 recent open source Large Language Models (LLMs) in 8 of the translated datasets and performed a comprehensive error analysis to pinpoint their primary failure modes. Our findings reveal consistent performance gaps for Bengali compared to English, particularly for smaller models and specific model families like Mistral. We also identified promising robustness in certain architectures, such as DeepSeek, that maintain more stable performance across languages. Our analysis reveals an inverse relationship between tokenization efficiency and LLM accuracy where models tend to perform worse when inputs are excessively tokenized, whereas more efficient \& concise tokenization results in improved performance. These findings highlight critical areas where current models fall short and underscore the need for improved dataset quality and evaluation methodologies tailored to multilingual contexts. This work will catalyze further research on NLP for underrepresented languages, helping to democratize access to advanced language technologies worldwide. The code and dataset used in this research is publicly available at https://github.com/BengaliAI/bn-llm-benchmark.
>
---
#### [new 014] Large Language Models in the Travel Domain: An Industrial Experience
- **分类: cs.CL; cs.AI**

- **简介: 论文研究在旅游领域应用大语言模型（LLMs）生成一致且准确的住宿描述，解决第三方数据不完整和不一致问题。使用Mistral 7B和Mixtral 8x7B进行实验，评估其在内容质量与计算成本间的权衡，结果显示Mixtral 8x7B表现更优但成本更高，为实际部署提供参考依据。**

- **链接: [http://arxiv.org/pdf/2507.22910v1](http://arxiv.org/pdf/2507.22910v1)**

> **作者:** Sergio Di Meglio; Aniello Somma; Luigi Libero Lucio Starace; Fabio Scippacercola; Giancarlo Sperlì; Sergio Di Martino
>
> **备注:** Manuscript accepted to the International Conference on Software Engineering and Knowledge Engineering (SEKE) 2025
>
> **摘要:** Online property booking platforms are widely used and rely heavily on consistent, up-to-date information about accommodation facilities, often sourced from third-party providers. However, these external data sources are frequently affected by incomplete or inconsistent details, which can frustrate users and result in a loss of market. In response to these challenges, we present an industrial case study involving the integration of Large Language Models (LLMs) into CALEIDOHOTELS, a property reservation platform developed by FERVENTO. We evaluate two well-known LLMs in this context: Mistral 7B, fine-tuned with QLoRA, and Mixtral 8x7B, utilized with a refined system prompt. Both models were assessed based on their ability to generate consistent and homogeneous descriptions while minimizing hallucinations. Mixtral 8x7B outperformed Mistral 7B in terms of completeness (99.6% vs. 93%), precision (98.8% vs. 96%), and hallucination rate (1.2% vs. 4%), producing shorter yet more concise content (249 vs. 277 words on average). However, this came at a significantly higher computational cost: 50GB VRAM and $1.61/hour versus 5GB and $0.16/hour for Mistral 7B. Our findings provide practical insights into the trade-offs between model quality and resource efficiency, offering guidance for deploying LLMs in production environments and demonstrating their effectiveness in enhancing the consistency and reliability of accommodation data.
>
---
#### [new 015] Beyond the Cloud: Assessing the Benefits and Drawbacks of Local LLM Deployment for Translators
- **分类: cs.CL; cs.CY; I.2.7; K.4.3**

- **简介: 该论文属于翻译技术任务，探讨本地部署大语言模型（LLM）对译者的影响。论文旨在解决数据隐私、安全与访问公平问题，评估了三种开源本地模型与云端商业模型的性能。研究重点是比较功能表现，而非翻译质量，强调本地部署的可行性与优势。**

- **链接: [http://arxiv.org/pdf/2507.23399v1](http://arxiv.org/pdf/2507.23399v1)**

> **作者:** Peter Sandrini
>
> **摘要:** The rapid proliferation of Large Language Models presents both opportunities and challenges for the translation field. While commercial, cloud-based AI chatbots have garnered significant attention in translation studies, concerns regarding data privacy, security, and equitable access necessitate exploration of alternative deployment models. This paper investigates the feasibility and performance of locally deployable, free language models as a viable alternative to proprietary, cloud-based AI solutions. This study evaluates three open-source models installed on CPU-based platforms and compared against commercially available online chat-bots. The evaluation focuses on functional performance rather than a comparative analysis of human-machine translation quality, an area already subject to extensive research. The platforms assessed were chosen for their accessibility and ease of use across various operating systems. While local deployment introduces its own challenges, the benefits of enhanced data control, improved privacy, and reduced dependency on cloud services are compelling. The findings of this study contribute to a growing body of knowledge concerning the democratization of AI technology and inform future research and development efforts aimed at making LLMs more accessible and practical for a wider range of users, specifically focusing on the needs of individual translators and small businesses.
>
---
#### [new 016] Text-to-SQL Task-oriented Dialogue Ontology Construction
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 论文提出TeQoDO方法，用于构建任务导向对话系统的本体，解决大型语言模型在可解释性和可控性上的不足。通过结合对话理论与SQL编程能力，无需监督即可自主构建本体，并在对话状态追踪任务中表现良好，提升了大型语言模型的可解释性。**

- **链接: [http://arxiv.org/pdf/2507.23358v1](http://arxiv.org/pdf/2507.23358v1)**

> **作者:** Renato Vukovic; Carel van Niekerk; Michael Heck; Benjamin Ruppik; Hsien-Chin Lin; Shutong Feng; Nurul Lubis; Milica Gasic
>
> **摘要:** Large language models (LLMs) are widely used as general-purpose knowledge sources, but they rely on parametric knowledge, limiting explainability and trustworthiness. In task-oriented dialogue (TOD) systems, this separation is explicit, using an external database structured by an explicit ontology to ensure explainability and controllability. However, building such ontologies requires manual labels or supervised training. We introduce TeQoDO: a Text-to-SQL task-oriented Dialogue Ontology construction method. Here, an LLM autonomously builds a TOD ontology from scratch without supervision using its inherent SQL programming capabilities combined with dialogue theory provided in the prompt. We show that TeQoDO outperforms transfer learning approaches, and its constructed ontology is competitive on a downstream dialogue state tracking task. Ablation studies demonstrate the key role of dialogue theory. TeQoDO also scales to allow construction of much larger ontologies, which we investigate on a Wikipedia and ArXiv dataset. We view this as a step towards broader application of ontologies to increase LLM explainability.
>
---
#### [new 017] Using Sentiment Analysis to Investigate Peer Feedback by Native and Non-Native English Speakers
- **分类: cs.CL; I.2.7; K.3.1**

- **简介: 该论文属于自然语言处理与教育交叉任务，旨在探究母语与非母语英语学习者在在线课程中同伴反馈的情感差异。研究通过分析500名学生的情感评分与语言背景，发现母语者对反馈评价较低，而非母语者写作更积极但收到的反馈情感较消极。**

- **链接: [http://arxiv.org/pdf/2507.22924v1](http://arxiv.org/pdf/2507.22924v1)**

> **作者:** Brittney Exline; Melanie Duffin; Brittany Harbison; Chrissa da Gomez; David Joyner
>
> **摘要:** Graduate-level CS programs in the U.S. increasingly enroll international students, with 60.2 percent of master's degrees in 2023 awarded to non-U.S. students. Many of these students take online courses, where peer feedback is used to engage students and improve pedagogy in a scalable manner. Since these courses are conducted in English, many students study in a language other than their first. This paper examines how native versus non-native English speaker status affects three metrics of peer feedback experience in online U.S.-based computing courses. Using the Twitter-roBERTa-based model, we analyze the sentiment of peer reviews written by and to a random sample of 500 students. We then relate sentiment scores and peer feedback ratings to students' language background. Results show that native English speakers rate feedback less favorably, while non-native speakers write more positively but receive less positive sentiment in return. When controlling for sex and age, significant interactions emerge, suggesting that language background plays a modest but complex role in shaping peer feedback experiences.
>
---
#### [new 018] A chart review process aided by natural language processing and multi-wave adaptive sampling to expedite validation of code-based algorithms for large database studies
- **分类: cs.CL; stat.ME**

- **简介: 该论文属于医疗数据分析任务，旨在解决验证基于代码算法准确性耗时的问题。通过引入自然语言处理加快图表审查，并采用多阶段自适应抽样减少审查数量，从而提高效率。论文验证了在肥胖患者自伤行为识别中的算法性能，结果显示该方法大幅节省时间且不影响精度。**

- **链接: [http://arxiv.org/pdf/2507.22943v1](http://arxiv.org/pdf/2507.22943v1)**

> **作者:** Shirley V Wang; Georg Hahn; Sushama Kattinakere Sreedhara; Mufaddal Mahesri; Haritha S. Pillai; Rajendra Aldis; Joyce Lii; Sarah K. Dutcher; Rhoda Eniafe; Jamal T. Jones; Keewan Kim; Jiwei He; Hana Lee; Sengwee Toh; Rishi J Desai; Jie Yang
>
> **摘要:** Background: One of the ways to enhance analyses conducted with large claims databases is by validating the measurement characteristics of code-based algorithms used to identify health outcomes or other key study parameters of interest. These metrics can be used in quantitative bias analyses to assess the robustness of results for an inferential study given potential bias from outcome misclassification. However, extensive time and resource allocation are typically re-quired to create reference-standard labels through manual chart review of free-text notes from linked electronic health records. Methods: We describe an expedited process that introduces efficiency in a validation study us-ing two distinct mechanisms: 1) use of natural language processing (NLP) to reduce time spent by human reviewers to review each chart, and 2) a multi-wave adaptive sampling approach with pre-defined criteria to stop the validation study once performance characteristics are identified with sufficient precision. We illustrate this process in a case study that validates the performance of a claims-based outcome algorithm for intentional self-harm in patients with obesity. Results: We empirically demonstrate that the NLP-assisted annotation process reduced the time spent on review per chart by 40% and use of the pre-defined stopping rule with multi-wave samples would have prevented review of 77% of patient charts with limited compromise to precision in derived measurement characteristics. Conclusion: This approach could facilitate more routine validation of code-based algorithms used to define key study parameters, ultimately enhancing understanding of the reliability of find-ings derived from database studies.
>
---
#### [new 019] Enabling Few-Shot Alzheimer's Disease Diagnosis on Tabular Biomarker Data with LLMs
- **分类: cs.CL; cs.LG; q-bio.QM**

- **简介: 该论文属于医学诊断任务，旨在解决阿尔茨海默病（AD）早期准确诊断的问题。作者提出了TAP-GPT框架，将TableGPT2适配用于小样本结构化生物标志物数据的AD诊断，并通过qLoRA微调提升性能，优于其他先进模型。**

- **链接: [http://arxiv.org/pdf/2507.23227v1](http://arxiv.org/pdf/2507.23227v1)**

> **作者:** Sophie Kearney; Shu Yang; Zixuan Wen; Bojian Hou; Duy Duong-Tran; Tianlong Chen; Jason Moore; Marylyn Ritchie; Li Shen
>
> **摘要:** Early and accurate diagnosis of Alzheimer's disease (AD), a complex neurodegenerative disorder, requires analysis of heterogeneous biomarkers (e.g., neuroimaging, genetic risk factors, cognitive tests, and cerebrospinal fluid proteins) typically represented in a tabular format. With flexible few-shot reasoning, multimodal integration, and natural-language-based interpretability, large language models (LLMs) offer unprecedented opportunities for prediction with structured biomedical data. We propose a novel framework called TAP-GPT, Tabular Alzheimer's Prediction GPT, that adapts TableGPT2, a multimodal tabular-specialized LLM originally developed for business intelligence tasks, for AD diagnosis using structured biomarker data with small sample sizes. Our approach constructs few-shot tabular prompts using in-context learning examples from structured biomedical data and finetunes TableGPT2 using the parameter-efficient qLoRA adaption for a clinical binary classification task of AD or cognitively normal (CN). The TAP-GPT framework harnesses the powerful tabular understanding ability of TableGPT2 and the encoded prior knowledge of LLMs to outperform more advanced general-purpose LLMs and a tabular foundation model (TFM) developed for prediction tasks. To our knowledge, this is the first application of LLMs to the prediction task using tabular biomarker data, paving the way for future LLM-driven multi-agent frameworks in biomedical informatics.
>
---
#### [new 020] Augmented Vision-Language Models: A Systematic Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能任务，旨在解决视觉语言模型缺乏可解释性、难以集成新信息及推理能力不足的问题。论文系统综述了结合神经网络与外部符号系统的增强视觉语言模型方法，以提升其推理和记忆能力。**

- **链接: [http://arxiv.org/pdf/2507.22933v1](http://arxiv.org/pdf/2507.22933v1)**

> **作者:** Anthony C Davis; Burhan Sadiq; Tianmin Shu; Chien-Ming Huang
>
> **摘要:** Recent advances in visual-language machine learning models have demonstrated exceptional ability to use natural language and understand visual scenes by training on large, unstructured datasets. However, this training paradigm cannot produce interpretable explanations for its outputs, requires retraining to integrate new information, is highly resource-intensive, and struggles with certain forms of logical reasoning. One promising solution involves integrating neural networks with external symbolic information systems, forming neural symbolic systems that can enhance reasoning and memory abilities. These neural symbolic systems provide more interpretable explanations to their outputs and the capacity to assimilate new information without extensive retraining. Utilizing powerful pre-trained Vision-Language Models (VLMs) as the core neural component, augmented by external systems, offers a pragmatic approach to realizing the benefits of neural-symbolic integration. This systematic literature review aims to categorize techniques through which visual-language understanding can be improved by interacting with external symbolic information systems.
>
---
#### [new 021] How does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在探究大语言模型中思维链（CoT）提示是否反映模型真实推理过程。研究者结合稀疏自编码器与激活修补技术，从Pythia模型中提取特征，分析CoT与非CoT提示下的推理差异。结果表明，CoT提升了模型内部结构的可解释性与模块化程度，尤其在较大模型中效果显著。**

- **链接: [http://arxiv.org/pdf/2507.22928v1](http://arxiv.org/pdf/2507.22928v1)**

> **作者:** Xi Chen; Aske Plaat; Niki van Stein
>
> **摘要:** Chain-of-thought (CoT) prompting boosts Large Language Models accuracy on multi-step tasks, yet whether the generated "thoughts" reflect the true internal reasoning process is unresolved. We present the first feature-level causal study of CoT faithfulness. Combining sparse autoencoders with activation patching, we extract monosemantic features from Pythia-70M and Pythia-2.8B while they tackle GSM8K math problems under CoT and plain (noCoT) prompting. Swapping a small set of CoT-reasoning features into a noCoT run raises answer log-probabilities significantly in the 2.8B model, but has no reliable effect in 70M, revealing a clear scale threshold. CoT also leads to significantly higher activation sparsity and feature interpretability scores in the larger model, signalling more modular internal computation. For example, the model's confidence in generating correct answers improves from 1.2 to 4.3. We introduce patch-curves and random-feature patching baselines, showing that useful CoT information is not only present in the top-K patches but widely distributed. Overall, our results indicate that CoT can induce more interpretable internal structures in high-capacity LLMs, validating its role as a structured prompting method.
>
---
#### [new 022] FinMarBa: A Market-Informed Dataset for Financial Sentiment Classification
- **分类: cs.CL; q-fin.GN**

- **简介: 该论文属于金融情感分类与投资组合优化任务，旨在结合市场数据与新闻情感信号提升投资决策。论文提出FinMarBa框架，采用轻量大语言模型与深度强化学习，构建三级架构处理混合数据，实现优于基准的投资回报与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.22932v1](http://arxiv.org/pdf/2507.22932v1)**

> **作者:** Baptiste Lefort; Eric Benhamou; Beatrice Guez; Jean-Jacques Ohana; Ethan Setrouk; Alban Etienne
>
> **备注:** 8 pages
>
> **摘要:** This paper presents a novel hierarchical framework for portfolio optimization, integrating lightweight Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) to combine sentiment signals from financial news with traditional market indicators. Our three-tier architecture employs base RL agents to process hybrid data, meta-agents to aggregate their decisions, and a super-agent to merge decisions based on market data and sentiment analysis. Evaluated on data from 2018 to 2024, after training on 2000-2017, the framework achieves a 26% annualized return and a Sharpe ratio of 1.2, outperforming equal-weighted and S&P 500 benchmarks. Key contributions include scalable cross-modal integration, a hierarchical RL structure for enhanced stability, and open-source reproducibility.
>
---
#### [new 023] A Graph-based Approach for Multi-Modal Question Answering from Flowcharts in Telecom Documents
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **简介: 该论文属于多模态问答任务，旨在解决技术文档中基于图表（如流程图）的问答问题。传统基于文本的RAG系统难以处理此类问题，因此作者提出一种基于图的方法，利用视觉大模型生成流程图的图表示，并将其融入文本嵌入流程，提升问答效果。实验表明该方法在电信领域文档上表现良好，且降低了推理阶段对视觉模型的依赖。**

- **链接: [http://arxiv.org/pdf/2507.22938v1](http://arxiv.org/pdf/2507.22938v1)**

> **作者:** Sumit Soman; H. G. Ranjani; Sujoy Roychowdhury; Venkata Dharma Surya Narayana Sastry; Akshat Jain; Pranav Gangrade; Ayaaz Khan
>
> **备注:** Accepted for publication at the KDD 2025 Workshop on Structured Knowledge for Large Language Models
>
> **摘要:** Question-Answering (QA) from technical documents often involves questions whose answers are present in figures, such as flowcharts or flow diagrams. Text-based Retrieval Augmented Generation (RAG) systems may fail to answer such questions. We leverage graph representations of flowcharts obtained from Visual large Language Models (VLMs) and incorporate them in a text-based RAG system to show that this approach can enable image retrieval for QA in the telecom domain. We present the end-to-end approach from processing technical documents, classifying image types, building graph representations, and incorporating them with the text embedding pipeline for efficient retrieval. We benchmark the same on a QA dataset created based on proprietary telecom product information documents. Results show that the graph representations obtained using a fine-tuned VLM model have lower edit distance with respect to the ground truth, which illustrate the robustness of these representations for flowchart images. Further, the approach for QA using these representations gives good retrieval performance using text-based embedding models, including a telecom-domain adapted one. Our approach also alleviates the need for a VLM in inference, which is an important cost benefit for deployed QA systems.
>
---
#### [new 024] Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Causal2Vec，旨在提升仅解码器结构的大语言模型（LLMs）作为通用嵌入模型的表现。该工作属于自然语言处理中的文本嵌入任务，旨在将文本转化为语义向量。现有方法多通过修改模型结构或增加输入文本来克服因果注意力限制，导致信息损失或计算成本上升。Causal2Vec不改变模型结构，引入轻量BERT预编码生成Contextual token，并优化最终嵌入表示，从而在降低计算成本的同时实现性能提升。**

- **链接: [http://arxiv.org/pdf/2507.23386v1](http://arxiv.org/pdf/2507.23386v1)**

> **作者:** Ailiang Lin; Zhuoyun Li; Kotaro Funakoshi
>
> **摘要:** Decoder-only large language models (LLMs) are increasingly used to build embedding models that effectively encode the semantic information of natural language texts into dense vector representations for various embedding tasks. However, many existing methods primarily focus on removing the causal attention mask in LLMs to enable bidirectional attention, potentially undermining the model's ability to extract semantic information acquired during pretraining. Additionally, leading unidirectional approaches often rely on extra input text to overcome the inherent limitations of causal attention, inevitably increasing computational costs. In this work, we propose Causal2Vec, a general-purpose embedding model tailored to enhance the performance of decoder-only LLMs without altering their original architectures or introducing significant computational overhead. Specifically, we first employ a lightweight BERT-style model to pre-encode the input text into a single Contextual token, which is then prepended to the LLM's input sequence, allowing each token to capture contextualized information even without attending to future tokens. Furthermore, to mitigate the recency bias introduced by last-token pooling and help LLMs better leverage the semantic information encoded in the Contextual token, we concatenate the last hidden states of Contextual and EOS tokens as the final text embedding. In practice, Causal2Vec achieves state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB) among models trained solely on publicly available retrieval datasets, while reducing the required sequence length by up to 85% and inference time by up to 82% compared to best-performing methods.
>
---
#### [new 025] Protecting Vulnerable Voices: Synthetic Dataset Generation for Self-Disclosure Detection
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于隐私保护任务，旨在解决社交媒体中用户自披露个人敏感信息（PII）的检测问题。由于缺乏公开标注数据，研究受限。为此，作者提出一种合成数据生成方法，构建了包含19类PII的合成数据集，并确保其可复现、不可关联且难以与真实数据区分，以促进相关研究。**

- **链接: [http://arxiv.org/pdf/2507.22930v1](http://arxiv.org/pdf/2507.22930v1)**

> **作者:** Shalini Jangra; Suparna De; Nishanth Sastry; Saeed Fadaei
>
> **备注:** 15 pages, 4 Figures, Accepted in "The 17th International Conference on Advances in Social Networks Analysis and Mining -ASONAM-2025"
>
> **摘要:** Social platforms such as Reddit have a network of communities of shared interests, with a prevalence of posts and comments from which one can infer users' Personal Information Identifiers (PIIs). While such self-disclosures can lead to rewarding social interactions, they pose privacy risks and the threat of online harms. Research into the identification and retrieval of such risky self-disclosures of PIIs is hampered by the lack of open-source labeled datasets. To foster reproducible research into PII-revealing text detection, we develop a novel methodology to create synthetic equivalents of PII-revealing data that can be safely shared. Our contributions include creating a taxonomy of 19 PII-revealing categories for vulnerable populations and the creation and release of a synthetic PII-labeled multi-text span dataset generated from 3 text generation Large Language Models (LLMs), Llama2-7B, Llama3-8B, and zephyr-7b-beta, with sequential instruction prompting to resemble the original Reddit posts. The utility of our methodology to generate this synthetic dataset is evaluated with three metrics: First, we require reproducibility equivalence, i.e., results from training a model on the synthetic data should be comparable to those obtained by training the same models on the original posts. Second, we require that the synthetic data be unlinkable to the original users, through common mechanisms such as Google Search. Third, we wish to ensure that the synthetic data be indistinguishable from the original, i.e., trained humans should not be able to tell them apart. We release our dataset and code at https://netsys.surrey.ac.uk/datasets/synthetic-self-disclosure/ to foster reproducible research into PII privacy risks in online social media.
>
---
#### [new 026] ISO-Bench: Benchmarking Multimodal Causal Reasoning in Visual-Language Models through Procedural Plans
- **分类: cs.CL**

- **简介: 该论文提出了ISO-Bench，用于评估视觉-语言模型在多模态因果推理中的表现，任务是判断图像与文本步骤的先后顺序。论文旨在解决模型对跨模态因果关系理解不足的问题，通过测试10个前沿模型，发现其表现远低于人类水平，并指出了改进方向。**

- **链接: [http://arxiv.org/pdf/2507.23135v1](http://arxiv.org/pdf/2507.23135v1)**

> **作者:** Ananya Sadana; Yash Kumar Lal; Jiawei Zhou
>
> **摘要:** Understanding causal relationships across modalities is a core challenge for multimodal models operating in real-world environments. We introduce ISO-Bench, a benchmark for evaluating whether models can infer causal dependencies between visual observations and procedural text. Each example presents an image of a task step and a text snippet from a plan, with the goal of deciding whether the visual step occurs before or after the referenced text step. Evaluation results on ten frontier vision-language models show underwhelming performance: the best zero-shot F1 is only 0.57, and chain-of-thought reasoning yields only modest gains (up to 0.62 F1), largely behind humans (0.98 F1). Our analysis further highlights concrete directions for improving causal understanding in multimodal models.
>
---
#### [new 027] Role-Aware Language Models for Secure and Contextualized Access Control in Organizations
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何通过角色感知语言模型实现企业中的安全、情境化访问控制。任务是解决不同组织角色对模型输出的权限控制问题。作者探索了三种建模方法，并构建了两个数据集进行评估，分析模型在不同组织结构下的表现及对攻击的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.23465v1](http://arxiv.org/pdf/2507.23465v1)**

> **作者:** Saeed Almheiri; Yerulan Kongrat; Adrian Santosh; Ruslan Tasmukhanov; Josemaria Vera; Muhammad Dehan Al Kautsar; Fajri Koto
>
> **摘要:** As large language models (LLMs) are increasingly deployed in enterprise settings, controlling model behavior based on user roles becomes an essential requirement. Existing safety methods typically assume uniform access and focus on preventing harmful or toxic outputs, without addressing role-specific access constraints. In this work, we investigate whether LLMs can be fine-tuned to generate responses that reflect the access privileges associated with different organizational roles. We explore three modeling strategies: a BERT-based classifier, an LLM-based classifier, and role-conditioned generation. To evaluate these approaches, we construct two complementary datasets. The first is adapted from existing instruction-tuning corpora through clustering and role labeling, while the second is synthetically generated to reflect realistic, role-sensitive enterprise scenarios. We assess model performance across varying organizational structures and analyze robustness to prompt injection, role mismatch, and jailbreak attempts.
>
---
#### [new 028] PRGB Benchmark: A Robust Placeholder-Assisted Algorithm for Benchmarking Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于信息检索与自然语言处理任务，旨在解决当前检索增强生成（RAG）系统缺乏对大语言模型（LLM）能力的细粒度评估问题。作者提出了PRGB基准，通过多级过滤、组合与引用推理等维度评估LLM在RAG中的表现，并引入占位符方法解耦模型知识与外部知识的贡献。**

- **链接: [http://arxiv.org/pdf/2507.22927v1](http://arxiv.org/pdf/2507.22927v1)**

> **作者:** Zhehao Tan; Yihan Jiao; Dan Yang; Lei Liu; Jie Feng; Duolin Sun; Yue Shen; Jian Wang; Peng Wei; Jinjie Gu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating external knowledge, where the LLM's ability to generate responses based on the combination of a given query and retrieved documents is crucial. However, most benchmarks focus on overall RAG system performance, rarely assessing LLM-specific capabilities. Current benchmarks emphasize broad aspects such as noise robustness, but lack a systematic and granular evaluation framework on document utilization. To this end, we introduce \textit{Placeholder-RAG-Benchmark}, a multi-level fine-grained benchmark, emphasizing the following progressive dimensions: (1) multi-level filtering abilities, (2) combination abilities, and (3) reference reasoning. To provide a more nuanced understanding of LLMs' roles in RAG systems, we formulate an innovative placeholder-based approach to decouple the contributions of the LLM's parametric knowledge and the external knowledge. Experiments demonstrate the limitations of representative LLMs in the RAG system's generation capabilities, particularly in error resilience and context faithfulness. Our benchmark provides a reproducible framework for developing more reliable and efficient RAG systems. Our code is available in https://github.com/Alipay-Med/PRGB.
>
---
#### [new 029] DiffLoRA: Differential Low-Rank Adapters for Large Language Models
- **分类: cs.CL**

- **简介: 论文提出DiffLoRA，一种基于低秩适配器的参数高效微调方法，结合了差分注意力机制，旨在提升大语言模型性能。属于自然语言处理任务，解决参数高效微调中的噪声问题。工作包括设计DiffLoRA结构，并在多种NLP任务上评估其表现，发现其在特定领域如HumanEval中显著优于LoRA。**

- **链接: [http://arxiv.org/pdf/2507.23588v1](http://arxiv.org/pdf/2507.23588v1)**

> **作者:** Alexandre Misrahi; Nadezhda Chirkova; Maxime Louis; Vassilina Nikoulina
>
> **摘要:** Differential Transformer has recently been proposed to improve performance in Transformer models by canceling out noise through a denoiser attention mechanism. In this work, we introduce DiffLoRA, a parameter-efficient adaptation of the differential attention mechanism, with low-rank adapters on both positive and negative attention terms. This approach retains the efficiency of LoRA while aiming to benefit from the performance gains of differential attention. We evaluate DiffLoRA across a broad range of NLP tasks, including general benchmarks, many-shot in-context learning, RAG, and long-context tests. We observe that, although DiffLoRA falls short of other parameter-efficient fine-tuning methods in most evaluation tasks, it shows interesting results in certain domains (+11 pts on LoRA for HumanEval). We analyze the attention patterns post-finetuning to identify the reasons for this behavior.
>
---
#### [new 030] ElectriQ: A Benchmark for Assessing the Response Capability of Large Language Models in Power Marketing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与电力营销服务结合的任务，旨在解决当前电力客服响应慢、准确性低的问题。作者构建了名为ElectriQ的基准测试，包含对话数据集和四项评估指标，用于评估和提升大语言模型在电力营销场景中的表现，并提出知识增强方法优化模型性能。**

- **链接: [http://arxiv.org/pdf/2507.22911v1](http://arxiv.org/pdf/2507.22911v1)**

> **作者:** Jinzhi Wang; Qingke Peng; Haozhou Li; Zeyuan Zeng; Qinfeng Song; Kaixuan Yang; Jiangbo Zhang; Yaoying Wang; Ruimeng Li; Biyi Zhou
>
> **摘要:** Electric power marketing customer service plays a critical role in addressing inquiries, complaints, and service requests. However, current systems, such as China's 95598 hotline, often struggle with slow response times, inflexible procedures, and limited accuracy in domain-specific tasks. While large language models (LLMs) like GPT-4o and Claude 3 demonstrate strong general capabilities, they lack the domain expertise and empathy required in this field. To bridge this gap, we introduce ElectriQ, the first benchmark designed to evaluate and enhance LLMs in electric power marketing scenarios. ElectriQ consists of a dialogue dataset covering six key service categories and introduces four evaluation metrics: professionalism, popularity, readability, and user-friendliness. We further incorporate a domain-specific knowledge base and propose a knowledge augmentation method to boost model performance. Experiments on 13 LLMs reveal that smaller models such as LLama3-8B, when fine-tuned and augmented, can surpass GPT-4o in terms of professionalism and user-friendliness. ElectriQ establishes a comprehensive foundation for developing LLMs tailored to the needs of power marketing services.
>
---
#### [new 031] C3: A Bilingual Benchmark for Spoken Dialogue Models Exploring Challenges in Complex Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话建模任务，旨在解决语音对话模型在理解复杂口语对话中的挑战。论文构建了一个包含1079个中英文实例的基准数据集C3，并提出基于大语言模型的评估方法，以评估语音对话模型在应对语义歧义、上下文依赖等问题上的表现。**

- **链接: [http://arxiv.org/pdf/2507.22968v1](http://arxiv.org/pdf/2507.22968v1)**

> **作者:** Chengqian Ma; Wei Tao; Yiwen Guo
>
> **摘要:** Spoken Dialogue Models (SDMs) have recently attracted significant attention for their ability to generate voice responses directly to users' spoken queries. Despite their increasing popularity, there exists a gap in research focused on comprehensively understanding their practical effectiveness in comprehending and emulating human conversations. This is especially true compared to text-based Large Language Models (LLMs), which benefit from extensive benchmarking. Human voice interactions are inherently more complex than text due to characteristics unique to spoken dialogue. Ambiguity poses one challenge, stemming from semantic factors like polysemy, as well as phonological aspects such as heterograph, heteronyms, and stress patterns. Additionally, context-dependency, like omission, coreference, and multi-turn interaction, adds further complexity to human conversational dynamics. To illuminate the current state of SDM development and to address these challenges, we present a benchmark dataset in this paper, which comprises 1,079 instances in English and Chinese. Accompanied by an LLM-based evaluation method that closely aligns with human judgment, this dataset facilitates a comprehensive exploration of the performance of SDMs in tackling these practical challenges.
>
---
#### [new 032] LENS: Learning Ensemble Confidence from Neural States for Multi-LLM Answer Integration
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于多模型集成任务，旨在提升大语言模型预测的鲁棒性与性能。传统方法忽略模型在不同上下文中的可靠性差异，论文提出LENS，通过分析模型内部状态学习预测其置信度，实现更精细的模型预测加权集成。方法无需修改模型参数，计算开销小，在多项选择和布尔问答任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.23167v1](http://arxiv.org/pdf/2507.23167v1)**

> **作者:** Jizhou Guo
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive performance across various tasks, with different models excelling in distinct domains and specific abilities. Effectively combining the predictions of multiple LLMs is crucial for enhancing system robustness and performance. However, existing ensemble methods often rely on simple techniques like voting or logits ensembling, which overlook the varying confidence and reliability of models in different contexts. In this work, we propose LENS (Learning ENsemble confidence from Neural States), a novel approach that learns to estimate model confidence by analyzing internal representations. For each LLM, we train a lightweight linear confidence predictor that leverages layer-wise hidden states and normalized probabilities as inputs. This allows for more nuanced weighting of model predictions based on their context-dependent reliability. Our method does not require modifying the model parameters and requires negligible additional computation. Experimental results on multiple-choice and boolean question-answering tasks demonstrate that LENS outperforms traditional ensemble methods by a substantial margin. Our findings suggest that internal representations provide valuable signals for determining model confidence and can be effectively leveraged for ensemble learning.
>
---
#### [new 033] Semantic Convergence: Investigating Shared Representations Across Scaled LLMs
- **分类: cs.CL; cs.LG; 68T50; I.2.6; I.2.7**

- **简介: 该论文研究不同规模的Gemma-2语言模型（2B与9B）是否形成相似的内部语义表示。通过稀疏自编码器提取特征，并使用激活相关性对齐特征空间，结合SVCCA和RSA方法进行比较。发现中层表示最相似，初步验证了大模型语义特征的普遍性，支持跨模型可解释性的基础。任务属模型解释性与表示学习。**

- **链接: [http://arxiv.org/pdf/2507.22918v1](http://arxiv.org/pdf/2507.22918v1)**

> **作者:** Daniel Son; Sanjana Rathore; Andrew Rufail; Adrian Simon; Daniel Zhang; Soham Dave; Cole Blondin; Kevin Zhu; Sean O'Brien
>
> **备注:** Submitted to ACL 2025 Student Research Workshop (poster)
>
> **摘要:** We investigate feature universality in Gemma-2 language models (Gemma-2-2B and Gemma-2-9B), asking whether models with a four-fold difference in scale still converge on comparable internal concepts. Using the Sparse Autoencoder (SAE) dictionary-learning pipeline, we utilize SAEs on each model's residual-stream activations, align the resulting monosemantic features via activation correlation, and compare the matched feature spaces with SVCCA and RSA. Middle layers yield the strongest overlap, while early and late layers show far less similarity. Preliminary experiments extend the analysis from single tokens to multi-token subspaces, showing that semantically similar subspaces interact similarly with language models. These results strengthen the case that large language models carve the world into broadly similar, interpretable features despite size differences, reinforcing universality as a foundation for cross-model interpretability.
>
---
#### [new 034] Opacity as Authority: Arbitrariness and the Preclusion of Contestation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文探讨任意性作为人类系统的基础功能机制，而非缺陷。它提出“动机→可验证性→可争议性”链条，分析任意性如何通过隐藏逻辑削弱问责，构建权威。论文旨在重新定义任意性，适用于法律、社会及AI系统。**

- **链接: [http://arxiv.org/pdf/2507.22944v1](http://arxiv.org/pdf/2507.22944v1)**

> **作者:** Naomi Omeonga wa Kayembe
>
> **摘要:** This article redefines arbitrariness not as a normative flaw or a symptom of domination, but as a foundational functional mechanism structuring human systems and interactions. Diverging from critical traditions that conflate arbitrariness with injustice, it posits arbitrariness as a semiotic trait: a property enabling systems - linguistic, legal, or social - to operate effectively while withholding their internal rationale. Building on Ferdinand de Saussure's concept of l'arbitraire du signe, the analysis extends this principle beyond language to demonstrate its cross-domain applicability, particularly in law and social dynamics. The paper introduces the "Motivation -> Constatability -> Contestability" chain, arguing that motivation functions as a crucial interface rendering an act's logic vulnerable to intersubjective contestation. When this chain is broken through mechanisms like "immotivization" or "Conflict Lateralization" (exemplified by "the blur of the wolf drowned in the fish"), acts produce binding effects without exposing their rationale, thus precluding justiciability. This structural opacity, while appearing illogical, is a deliberate design protecting authority from accountability. Drawing on Shannon's entropy model, the paper formalizes arbitrariness as A = H(L|M) (conditional entropy). It thereby proposes a modern theory of arbitrariness as a neutral operator central to control as well as care, an overlooked dimension of interpersonal relations. While primarily developed through human social systems, this framework also illuminates a new pathway for analyzing explainability in advanced artificial intelligence systems.
>
---
#### [new 035] SMART-Editor: A Multi-Agent Framework for Human-Like Design Editing with Structural Integrity
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SMART-Editor，用于结构化和非结构化图像编辑任务，旨在解决局部编辑破坏全局一致性的问题。通过Reward-Refine和RewardDPO方法，在推理和训练阶段提升编辑的语义一致性和视觉对齐。论文引入了SMARTEdit-Bench基准测试，并在多领域编辑场景中优于现有模型。**

- **链接: [http://arxiv.org/pdf/2507.23095v1](http://arxiv.org/pdf/2507.23095v1)**

> **作者:** Ishani Mondal; Meera Bharadwaj; Ayush Roy; Aparna Garimella; Jordan Lee Boyd-Graber
>
> **备注:** Under Submission
>
> **摘要:** We present SMART-Editor, a framework for compositional layout and content editing across structured (posters, websites) and unstructured (natural images) domains. Unlike prior models that perform local edits, SMART-Editor preserves global coherence through two strategies: Reward-Refine, an inference-time rewardguided refinement method, and RewardDPO, a training-time preference optimization approach using reward-aligned layout pairs. To evaluate model performance, we introduce SMARTEdit-Bench, a benchmark covering multi-domain, cascading edit scenarios. SMART-Editor outperforms strong baselines like InstructPix2Pix and HIVE, with RewardDPO achieving up to 15% gains in structured settings and Reward-Refine showing advantages on natural images. Automatic and human evaluations confirm the value of reward-guided planning in producing semantically consistent and visually aligned edits.
>
---
#### [new 036] Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与多模态系统任务，旨在解决如何高效将连续多模态数据转化为适合大语言模型（LLM）处理的离散表示。论文系统梳理了基于向量量化（VQ）的离散化方法，分类分析了8种典型VQ变体，并探讨了其在LLM系统中的集成挑战与研究方向。**

- **链接: [http://arxiv.org/pdf/2507.22920v1](http://arxiv.org/pdf/2507.22920v1)**

> **作者:** Jindong Li; Yali Fu; Jiahong Liu; Linxiao Cao; Wei Ji; Menglin Yang; Irwin King; Ming-Hsuan Yang
>
> **摘要:** The rapid advancement of large language models (LLMs) has intensified the need for effective mechanisms to transform continuous multimodal data into discrete representations suitable for language-based processing. Discrete tokenization, with vector quantization (VQ) as a central approach, offers both computational efficiency and compatibility with LLM architectures. Despite its growing importance, there is a lack of a comprehensive survey that systematically examines VQ techniques in the context of LLM-based systems. This work fills this gap by presenting the first structured taxonomy and analysis of discrete tokenization methods designed for LLMs. We categorize 8 representative VQ variants that span classical and modern paradigms and analyze their algorithmic principles, training dynamics, and integration challenges with LLM pipelines. Beyond algorithm-level investigation, we discuss existing research in terms of classical applications without LLMs, LLM-based single-modality systems, and LLM-based multimodal systems, highlighting how quantization strategies influence alignment, reasoning, and generation performance. In addition, we identify key challenges including codebook collapse, unstable gradient estimation, and modality-specific encoding constraints. Finally, we discuss emerging research directions such as dynamic and task-adaptive quantization, unified tokenization frameworks, and biologically inspired codebook learning. This survey bridges the gap between traditional vector quantization and modern LLM applications, serving as a foundational reference for the development of efficient and generalizable multimodal systems. A continuously updated version is available at: https://github.com/jindongli-Ai/LLM-Discrete-Tokenization-Survey.
>
---
#### [new 037] MRGSEM-Sum: An Unsupervised Multi-document Summarization Framework based on Multi-Relational Graphs and Structural Entropy Minimization
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于多文档摘要任务，旨在解决文档间复杂关系建模与信息冗余问题。现有方法依赖单关系图与预设聚类数，效果受限。论文提出MRGSEM-Sum，构建融合语义与语篇关系的多关系图，并采用结构熵最小化算法自动聚类，最后通过位置感知机制压缩生成摘要，实验证明其性能优异。**

- **链接: [http://arxiv.org/pdf/2507.23400v1](http://arxiv.org/pdf/2507.23400v1)**

> **作者:** Yongbing Zhang; Fang Nan; Shengxiang Gao; Yuxin Huang; Kaiwen Tan; Zhengtao Yu
>
> **摘要:** The core challenge faced by multi-document summarization is the complexity of relationships among documents and the presence of information redundancy. Graph clustering is an effective paradigm for addressing this issue, as it models the complex relationships among documents using graph structures and reduces information redundancy through clustering, achieving significant research progress. However, existing methods often only consider single-relational graphs and require a predefined number of clusters, which hinders their ability to fully represent rich relational information and adaptively partition sentence groups to reduce redundancy. To overcome these limitations, we propose MRGSEM-Sum, an unsupervised multi-document summarization framework based on multi-relational graphs and structural entropy minimization. Specifically, we construct a multi-relational graph that integrates semantic and discourse relations between sentences, comprehensively modeling the intricate and dynamic connections among sentences across documents. We then apply a two-dimensional structural entropy minimization algorithm for clustering, automatically determining the optimal number of clusters and effectively organizing sentences into coherent groups. Finally, we introduce a position-aware compression mechanism to distill each cluster, generating concise and informative summaries. Extensive experiments on four benchmark datasets (Multi-News, DUC-2004, PubMed, and WikiSum) demonstrate that our approach consistently outperforms previous unsupervised methods and, in several cases, achieves performance comparable to supervised models and large language models. Human evaluation demonstrates that the summaries generated by MRGSEM-Sum exhibit high consistency and coverage, approaching human-level quality.
>
---
#### [new 038] CoE-Ops: Collaboration of LLM-based Experts for AIOps Question-Answering
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CoE-Ops框架，用于AIOps问答任务，解决单一模型处理多领域运维问题的能力限制。通过结合多个专家模型与大语言模型分类器，并引入检索增强生成机制，提升了高、低层运维任务的准确率，实验结果显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.22937v1](http://arxiv.org/pdf/2507.22937v1)**

> **作者:** Jinkun Zhao; Yuanshuai Wang; Xingjian Zhang; Ruibo Chen; Xingchuang Liao; Junle Wang; Lei Huang; Kui Zhang; Wenjun Wu
>
> **摘要:** With the rapid evolution of artificial intelligence, AIOps has emerged as a prominent paradigm in DevOps. Lots of work has been proposed to improve the performance of different AIOps phases. However, constrained by domain-specific knowledge, a single model can only handle the operation requirement of a specific task,such as log parser,root cause analysis. Meanwhile, combining multiple models can achieve more efficient results, which have been proved in both previous ensemble learning and the recent LLM training domain. Inspired by these works,to address the similar challenges in AIOPS, this paper first proposes a collaboration-of-expert framework(CoE-Ops) incorporating a general-purpose large language model task classifier. A retrieval-augmented generation mechanism is introduced to improve the framework's capability in handling both Question-Answering tasks with high-level(Code,build,Test,etc.) and low-level(fault analysis,anomaly detection,etc.). Finally, the proposed method is implemented in the AIOps domain, and extensive experiments are conducted on the DevOps-EVAL dataset. Experimental results demonstrate that CoE-Ops achieves a 72% improvement in routing accuracy for high-level AIOps tasks compared to existing CoE methods, delivers up to 8% accuracy enhancement over single AIOps models in DevOps problem resolution, and outperforms larger-scale Mixture-of-Experts (MoE) models by up to 14% in accuracy.
>
---
#### [new 039] MPCC: A Novel Benchmark for Multimodal Planning with Complex Constraints in Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; I.2.8; I.2.10**

- **简介: 该论文提出MPCC基准，评估多模态大语言模型（MLLM）在复杂约束下的多模态规划能力。任务是多模态规划，解决现有基准无法评估真实场景规划能力和缺乏跨模态约束的问题。工作包括设计含复杂约束的现实任务，并验证MLLM在多约束场景下的表现与挑战。**

- **链接: [http://arxiv.org/pdf/2507.23382v1](http://arxiv.org/pdf/2507.23382v1)**

> **作者:** Yiyan Ji; Haoran Chen; Qiguang Chen; Chengyue Wu; Libo Qin; Wanxiang Che
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Multimodal planning capabilities refer to the ability to predict, reason, and design steps for task execution with multimodal context, which is essential for complex reasoning and decision-making across multiple steps. However, current benchmarks face two key challenges: (1) they cannot directly assess multimodal real-world planning capabilities, and (2) they lack constraints or implicit constraints across modalities. To address these issues, we introduce Multimodal Planning with Complex Constraints (MPCC), the first benchmark to systematically evaluate MLLMs' ability to handle multimodal constraints in planning. To address the first challenge, MPCC focuses on three real-world tasks: Flight Planning, Calendar Planning, and Meeting Planning. To solve the second challenge, we introduce complex constraints (e.g. budget, temporal, and spatial) in these tasks, with graded difficulty levels (EASY, MEDIUM, HARD) to separate constraint complexity from search space expansion. Experiments on 13 advanced MLLMs reveal significant challenges: closed-source models achieve only 21.3% feasible plans, while open-source models average below 11%. Additionally, we observe that MLLMs are highly sensitive to constraint complexity and that traditional multimodal prompting strategies fail in multi-constraint scenarios. Our work formalizes multimodal constraints in planning, provides a rigorous evaluation framework, and highlights the need for advancements in constraint-aware reasoning for real-world MLLM applications.
>
---
#### [new 040] Exploring In-Context Learning for Frame-Semantic Parsing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的框架语义解析任务，旨在通过上下文学习方法，利用大语言模型完成无需微调的框架识别与语义角色标注。论文提出自动构建任务提示的方法，基于FrameNet数据库生成提示信息，并在暴力事件相关框架上验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.23082v1](http://arxiv.org/pdf/2507.23082v1)**

> **作者:** Diego Garat; Guillermo Moncecchi; Dina Wonsever
>
> **摘要:** Frame Semantic Parsing (FSP) entails identifying predicates and labeling their arguments according to Frame Semantics. This paper investigates the use of In-Context Learning (ICL) with Large Language Models (LLMs) to perform FSP without model fine-tuning. We propose a method that automatically generates task-specific prompts for the Frame Identification (FI) and Frame Semantic Role Labeling (FSRL) subtasks, relying solely on the FrameNet database. These prompts, constructed from frame definitions and annotated examples, are used to guide six different LLMs. Experiments are conducted on a subset of frames related to violent events. The method achieves competitive results, with F1 scores of 94.3% for FI and 77.4% for FSRL. The findings suggest that ICL offers a practical and effective alternative to traditional fine-tuning for domain-specific FSP tasks.
>
---
#### [new 041] Predicting stock prices with ChatGPT-annotated Reddit sentiment
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 该论文属于金融预测任务，旨在研究社交媒体情绪是否能预测股价波动。论文分析了Reddit上关于GameStop和AMC的讨论，结合三种情感分析模型，发现情绪与股价关联较弱，而评论量和搜索趋势更具预测力。**

- **链接: [http://arxiv.org/pdf/2507.22922v1](http://arxiv.org/pdf/2507.22922v1)**

> **作者:** Mateusz Kmak; Kamil Chmurzyński; Kamil Matejuk; Paweł Kotzbach; Jan Kocoń
>
> **备注:** International Conference on Computational Science 2025
>
> **摘要:** The surge of retail investor activity on social media, exemplified by the 2021 GameStop short squeeze, raised questions about the influence of online sentiment on stock prices. This paper explores whether sentiment derived from social media discussions can meaningfully predict stock market movements. We focus on Reddit's r/wallstreetbets and analyze sentiment related to two companies: GameStop (GME) and AMC Entertainment (AMC). To assess sentiment's role, we employ two existing text-based sentiment analysis methods and introduce a third, a ChatGPT-annotated and fine-tuned RoBERTa-based model designed to better interpret the informal language and emojis prevalent in social media discussions. We use correlation and causality metrics to determine these models' predictive power. Surprisingly, our findings suggest that social media sentiment has only a weak correlation with stock prices. At the same time, simpler metrics, such as the volume of comments and Google search trends, exhibit stronger predictive signals. These results highlight the complexity of retail investor behavior and suggest that traditional sentiment analysis may not fully capture the nuances of market-moving online discussions.
>
---
#### [new 042] PARROT: An Open Multilingual Radiology Reports Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决跨语言、跨地区医学文本数据缺乏问题。作者构建了PARROT数据集，包含21国、13种语言的2658份虚构放射报告，并标注元数据与ICD-10编码。通过人机区分实验，验证数据质量，推动隐私无忧的多语言NLP模型开发。**

- **链接: [http://arxiv.org/pdf/2507.22939v1](http://arxiv.org/pdf/2507.22939v1)**

> **作者:** Bastien Le Guellec; Kokou Adambounou; Lisa C Adams; Thibault Agripnidis; Sung Soo Ahn; Radhia Ait Chalal; Tugba Akinci D Antonoli; Philippe Amouyel; Henrik Andersson; Raphael Bentegeac; Claudio Benzoni; Antonino Andrea Blandino; Felix Busch; Elif Can; Riccardo Cau; Armando Ugo Cavallo; Christelle Chavihot; Erwin Chiquete; Renato Cuocolo; Eugen Divjak; Gordana Ivanac; Barbara Dziadkowiec Macek; Armel Elogne; Salvatore Claudio Fanni; Carlos Ferrarotti; Claudia Fossataro; Federica Fossataro; Katarzyna Fulek; Michal Fulek; Pawel Gac; Martyna Gachowska; Ignacio Garcia Juarez; Marco Gatti; Natalia Gorelik; Alexia Maria Goulianou; Aghiles Hamroun; Nicolas Herinirina; Krzysztof Kraik; Dominik Krupka; Quentin Holay; Felipe Kitamura; Michail E Klontzas; Anna Kompanowska; Rafal Kompanowski; Alexandre Lefevre; Tristan Lemke; Maximilian Lindholz; Lukas Muller; Piotr Macek; Marcus Makowski; Luigi Mannacio; Aymen Meddeb; Antonio Natale; Beatrice Nguema Edzang; Adriana Ojeda; Yae Won Park; Federica Piccione; Andrea Ponsiglione; Malgorzata Poreba; Rafal Poreba; Philipp Prucker; Jean Pierre Pruvo; Rosa Alba Pugliesi; Feno Hasina Rabemanorintsoa; Vasileios Rafailidis; Katarzyna Resler; Jan Rotkegel; Luca Saba; Ezann Siebert; Arnaldo Stanzione; Ali Fuat Tekin; Liz Toapanta Yanchapaxi; Matthaios Triantafyllou; Ekaterini Tsaoulia; Evangelia Vassalou; Federica Vernuccio; Johan Wasselius; Weilang Wang; Szymon Urban; Adrian Wlodarczak; Szymon Wlodarczak; Andrzej Wysocki; Lina Xu; Tomasz Zatonski; Shuhang Zhang; Sebastian Ziegelmayer; Gregory Kuchcinski; Keno K Bressem
>
> **摘要:** Rationale and Objectives: To develop and validate PARROT (Polyglottal Annotated Radiology Reports for Open Testing), a large, multicentric, open-access dataset of fictional radiology reports spanning multiple languages for testing natural language processing applications in radiology. Materials and Methods: From May to September 2024, radiologists were invited to contribute fictional radiology reports following their standard reporting practices. Contributors provided at least 20 reports with associated metadata including anatomical region, imaging modality, clinical context, and for non-English reports, English translations. All reports were assigned ICD-10 codes. A human vs. AI report differentiation study was conducted with 154 participants (radiologists, healthcare professionals, and non-healthcare professionals) assessing whether reports were human-authored or AI-generated. Results: The dataset comprises 2,658 radiology reports from 76 authors across 21 countries and 13 languages. Reports cover multiple imaging modalities (CT: 36.1%, MRI: 22.8%, radiography: 19.0%, ultrasound: 16.8%) and anatomical regions, with chest (19.9%), abdomen (18.6%), head (17.3%), and pelvis (14.1%) being most prevalent. In the differentiation study, participants achieved 53.9% accuracy (95% CI: 50.7%-57.1%) in distinguishing between human and AI-generated reports, with radiologists performing significantly better (56.9%, 95% CI: 53.3%-60.6%, p<0.05) than other groups. Conclusion: PARROT represents the largest open multilingual radiology report dataset, enabling development and validation of natural language processing applications across linguistic, geographic, and clinical boundaries without privacy constraints.
>
---
#### [new 043] Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型代理的长期推理能力。论文提出了一种分层记忆架构（H-MEM），通过多级语义抽象组织记忆，并引入索引编码实现高效检索。实验证明该方法在长期对话场景中优于现有基线方法。**

- **链接: [http://arxiv.org/pdf/2507.22925v1](http://arxiv.org/pdf/2507.22925v1)**

> **作者:** Haoran Sun; Shaoning Zeng
>
> **摘要:** Long-term memory is one of the key factors influencing the reasoning capabilities of Large Language Model Agents (LLM Agents). Incorporating a memory mechanism that effectively integrates past interactions can significantly enhance decision-making and contextual coherence of LLM Agents. While recent works have made progress in memory storage and retrieval, such as encoding memory into dense vectors for similarity-based search or organizing knowledge in the form of graph, these approaches often fall short in structured memory organization and efficient retrieval. To address these limitations, we propose a Hierarchical Memory (H-MEM) architecture for LLM Agents that organizes and updates memory in a multi-level fashion based on the degree of semantic abstraction. Each memory vector is embedded with a positional index encoding pointing to its semantically related sub-memories in the next layer. During the reasoning phase, an index-based routing mechanism enables efficient, layer-by-layer retrieval without performing exhaustive similarity computations. We evaluate our method on five task settings from the LoCoMo dataset. Experimental results show that our approach consistently outperforms five baseline methods, demonstrating its effectiveness in long-term dialogue scenarios.
>
---
#### [new 044] Trustworthy Reasoning: Evaluating and Enhancing Factual Accuracy in LLM Intermediate Thought Processes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理过程中存在的事实错误问题。作者提出了RELIANCE框架，通过事实核查、强化学习优化和可解释性分析，提升模型推理链中的事实准确性，验证了其在多个模型上的有效性，并为未来训练方法提供了新方向。**

- **链接: [http://arxiv.org/pdf/2507.22940v1](http://arxiv.org/pdf/2507.22940v1)**

> **作者:** Rui Jiao; Yue Zhang; Jinku Li
>
> **摘要:** We present RELIANCE (Reasoning Evaluation with Logical Integrity and Accuracy for Confidence Enhancement), a novel framework addressing a critical vulnerability in Large Language Models (LLMs): the prevalence of factual inaccuracies within intermediate reasoning steps despite correct final answers. This phenomenon poses substantial risks in high-stakes domains including healthcare, legal analysis, and scientific research, where erroneous yet confidently presented reasoning can mislead users into dangerous decisions. Our framework integrates three core components: (1) a specialized fact-checking classifier trained on counterfactually augmented data to detect subtle factual inconsistencies within reasoning chains; (2) a Group Relative Policy Optimization (GRPO) reinforcement learning approach that balances factuality, coherence, and structural correctness through multi-dimensional rewards; and (3) a mechanistic interpretability module examining how factuality improvements manifest in model activations during reasoning processes. Extensive evaluation across ten state-of-the-art models reveals concerning patterns: even leading models like Claude-3.7 and GPT-o1 demonstrate reasoning factual accuracy of only 81.93% and 82.57% respectively. RELIANCE significantly enhances factual robustness (up to 49.90% improvement) while maintaining or improving performance on challenging benchmarks including Math-500, AIME-2024, and GPQA. Furthermore, our activation-level analysis provides actionable insights into how factual enhancements reshape reasoning trajectories within model architectures, establishing foundations for future training methodologies that explicitly target factual robustness through activation-guided optimization.
>
---
#### [new 045] A Hybrid Framework for Subject Analysis: Integrating Embedding-Based Regression Models with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息组织任务，旨在解决图书馆资源主题分析中传统模型泛化能力差、大模型易过拟合与幻觉的问题。作者提出一种融合嵌入式回归模型与大语言模型的混合框架，通过预测标签数量与后编辑优化，提高主题标引的准确性与规范性。**

- **链接: [http://arxiv.org/pdf/2507.22913v1](http://arxiv.org/pdf/2507.22913v1)**

> **作者:** Jinyu Liu; Xiaoying Song; Diana Zhang; Jason Thomale; Daqing He; Lingzi Hong
>
> **备注:** 13 pages, 2 figures, accepted by ASIST 2025
>
> **摘要:** Providing subject access to information resources is an essential function of any library management system. Large language models (LLMs) have been widely used in classification and summarization tasks, but their capability to perform subject analysis is underexplored. Multi-label classification with traditional machine learning (ML) models has been used for subject analysis but struggles with unseen cases. LLMs offer an alternative but often over-generate and hallucinate. Therefore, we propose a hybrid framework that integrates embedding-based ML models with LLMs. This approach uses ML models to (1) predict the optimal number of LCSH labels to guide LLM predictions and (2) post-edit the predicted terms with actual LCSH terms to mitigate hallucinations. We experimented with LLMs and the hybrid framework to predict the subject terms of books using the Library of Congress Subject Headings (LCSH). Experiment results show that providing initial predictions to guide LLM generations and imposing post-edits result in more controlled and vocabulary-aligned outputs.
>
---
#### [new 046] Enhancing RAG Efficiency with Adaptive Context Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG系统中因冗长上下文带来的推理效率问题。作者提出ACC-RAG框架，通过自适应调整压缩率，结合层次化压缩与上下文选择，在保持准确率的同时显著提升推理速度。**

- **链接: [http://arxiv.org/pdf/2507.22931v1](http://arxiv.org/pdf/2507.22931v1)**

> **作者:** Shuyu Guo; Zhaochun Ren
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge but incurs significant inference costs due to lengthy retrieved contexts. While context compression mitigates this issue, existing methods apply fixed compression rates, over-compressing simple queries or under-compressing complex ones. We propose Adaptive Context Compression for RAG (ACC-RAG), a framework that dynamically adjusts compression rates based on input complexity, optimizing inference efficiency without sacrificing accuracy. ACC-RAG combines a hierarchical compressor (for multi-granular embeddings) with a context selector to retain minimal sufficient information, akin to human skimming. Evaluated on Wikipedia and five QA datasets, ACC-RAG outperforms fixed-rate methods and matches/unlocks over 4 times faster inference versus standard RAG while maintaining or improving accuracy.
>
---
#### [new 047] EH-Benchmark Ophthalmic Hallucination Benchmark and Agent-Driven Top-Down Traceable Reasoning Workflow
- **分类: cs.CL; cs.CV; cs.MA**

- **简介: 该论文属于医学自然语言处理任务，旨在解决医疗大语言模型在眼科诊断中的幻觉问题。作者构建了EH-Benchmark评估框架，分类幻觉类型，并提出基于智能体的三阶段推理流程，以提升模型的准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.22929v1](http://arxiv.org/pdf/2507.22929v1)**

> **作者:** Xiaoyu Pan; Yang Bai; Ke Zou; Yang Zhou; Jun Zhou; Huazhu Fu; Yih-Chung Tham; Yong Liu
>
> **备注:** 9 figures, 5 tables. submit/6621751
>
> **摘要:** Medical Large Language Models (MLLMs) play a crucial role in ophthalmic diagnosis, holding significant potential to address vision-threatening diseases. However, their accuracy is constrained by hallucinations stemming from limited ophthalmic knowledge, insufficient visual localization and reasoning capabilities, and a scarcity of multimodal ophthalmic data, which collectively impede precise lesion detection and disease diagnosis. Furthermore, existing medical benchmarks fail to effectively evaluate various types of hallucinations or provide actionable solutions to mitigate them. To address the above challenges, we introduce EH-Benchmark, a novel ophthalmology benchmark designed to evaluate hallucinations in MLLMs. We categorize MLLMs' hallucinations based on specific tasks and error types into two primary classes: Visual Understanding and Logical Composition, each comprising multiple subclasses. Given that MLLMs predominantly rely on language-based reasoning rather than visual processing, we propose an agent-centric, three-phase framework, including the Knowledge-Level Retrieval stage, the Task-Level Case Studies stage, and the Result-Level Validation stage. Experimental results show that our multi-agent framework significantly mitigates both types of hallucinations, enhancing accuracy, interpretability, and reliability. Our project is available at https://github.com/ppxy1/EH-Benchmark.
>
---
#### [new 048] A Language Model-Driven Semi-Supervised Ensemble Framework for Illicit Market Detection Across Deep/Dark Web and Social Platforms
- **分类: cs.CL; cs.AI; cs.LG; 68T07, 68T50**

- **简介: 该论文属于非法市场检测任务，旨在解决深网、暗网及社交平台上的非法内容识别问题。作者提出了一种结合微调语言模型与半监督集成学习的分类框架，利用ModernBERT提取语义特征，并融合人工设计的结构与元特征，实现对毒品、武器及凭证交易的高效识别。**

- **链接: [http://arxiv.org/pdf/2507.22912v1](http://arxiv.org/pdf/2507.22912v1)**

> **作者:** Navid Yazdanjue; Morteza Rakhshaninejad; Hossein Yazdanjouei; Mohammad Sadegh Khorshidi; Mikko S. Niemela; Fang Chen; Amir H. Gandomi
>
> **备注:** 16 pages, 5 figures, 9 tables
>
> **摘要:** Illegal marketplaces have increasingly shifted to concealed parts of the internet, including the deep and dark web, as well as platforms such as Telegram, Reddit, and Pastebin. These channels enable the anonymous trade of illicit goods including drugs, weapons, and stolen credentials. Detecting and categorizing such content remains challenging due to limited labeled data, the evolving nature of illicit language, and the structural heterogeneity of online sources. This paper presents a hierarchical classification framework that combines fine-tuned language models with a semi-supervised ensemble learning strategy to detect and classify illicit marketplace content across diverse platforms. We extract semantic representations using ModernBERT, a transformer model for long documents, finetuned on domain-specific data from deep and dark web pages, Telegram channels, Subreddits, and Pastebin pastes to capture specialized jargon and ambiguous linguistic patterns. In addition, we incorporate manually engineered features such as document structure, embedded patterns including Bitcoin addresses, emails, and IPs, and metadata, which complement language model embeddings. The classification pipeline operates in two stages. The first stage uses a semi-supervised ensemble of XGBoost, Random Forest, and SVM with entropy-based weighted voting to detect sales-related documents. The second stage further classifies these into drug, weapon, or credential sales. Experiments on three datasets, including our multi-source corpus, DUTA, and CoDA, show that our model outperforms several baselines, including BERT, ModernBERT, DarkBERT, ALBERT, Longformer, and BigBird. The model achieves an accuracy of 0.96489, an F1-score of 0.93467, and a TMCC of 0.95388, demonstrating strong generalization, robustness under limited supervision, and effectiveness in real-world illicit content detection.
>
---
#### [new 049] Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决传统主题模型难以捕捉语义抽象特征、依赖词袋表示的问题。作者提出机制主题模型（MTMs），基于稀疏自编码器学习可解释特征，定义更深层的主题概念，并实现基于主题的可控文本生成。论文还设计了基于大语言模型的评估框架“主题裁判”来评估主题质量。**

- **链接: [http://arxiv.org/pdf/2507.23220v1](http://arxiv.org/pdf/2507.23220v1)**

> **作者:** Carolina Zheng; Nicolas Beltran-Velez; Sweta Karlekar; Claudia Shi; Achille Nazaret; Asif Mallik; Amir Feder; David M. Blei
>
> **摘要:** Traditional topic models are effective at uncovering latent themes in large text collections. However, due to their reliance on bag-of-words representations, they struggle to capture semantically abstract features. While some neural variants use richer representations, they are similarly constrained by expressing topics as word lists, which limits their ability to articulate complex topics. We introduce Mechanistic Topic Models (MTMs), a class of topic models that operate on interpretable features learned by sparse autoencoders (SAEs). By defining topics over this semantically rich space, MTMs can reveal deeper conceptual themes with expressive feature descriptions. Moreover, uniquely among topic models, MTMs enable controllable text generation using topic-based steering vectors. To properly evaluate MTM topics against word-list-based approaches, we propose \textit{topic judge}, an LLM-based pairwise comparison evaluation framework. Across five datasets, MTMs match or exceed traditional and neural baselines on coherence metrics, are consistently preferred by topic judge, and enable effective steering of LLM outputs.
>
---
#### [new 050] Context-aware Rotary Position Embedding
- **分类: cs.CL**

- **简介: 论文提出CARoPE，一种上下文感知的旋转位置编码方法，用于改进Transformer模型中的位置表示。相比传统RoPE，CARoPE根据输入动态生成频率模式，提升模型对长序列和上下文关系的建模能力。实验表明其在语言建模任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2507.23083v1](http://arxiv.org/pdf/2507.23083v1)**

> **作者:** Ali Veisi; Delaram Fartoot; Hamidreza Amirzadeh
>
> **备注:** 4 pages, 1 table
>
> **摘要:** Positional encoding is a vital component of Transformer architectures, enabling models to incorporate sequence order into self-attention mechanisms. Rotary Positional Embeddings (RoPE) have become a widely adopted solution due to their compatibility with relative position encoding and computational efficiency. However, RoPE relies on static, input-independent sinusoidal frequency patterns, limiting its ability to model context-sensitive relationships. In this work, we propose CARoPE (Context-Aware Rotary Positional Embedding), a novel generalization of RoPE that dynamically generates head-specific frequency patterns conditioned on token embeddings. This design introduces token- and context-sensitive positional representations while preserving RoPE efficiency and architectural simplicity. CARoPE computes input-dependent phase shifts using a bounded transformation of token embeddings and integrates them into the rotary mechanism across attention heads. We evaluate CARoPE on the FineWeb-Edu-10B dataset using GPT-2 variants trained on next-token prediction tasks. Experimental results show that CARoPE consistently outperforms RoPE and other common positional encoding baselines, achieving significantly lower perplexity, even at longer context lengths. Additionally, CARoPE enables faster training throughput without sacrificing model stability. These findings demonstrate that CARoPE offers a scalable, expressive, and efficient upgrade to existing positional encoding strategies in Transformer models.
>
---
#### [new 051] Cascaded Information Disclosure for Generalized Evaluation of Problem Solving Capabilities
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在更准确评估大语言模型的问题解决能力。现有问答基准测试间接且可能夸大模型差异。论文提出级联问题披露框架，分阶段揭示问题信息，以诱导模型生成可解释的中间推理过程，并通过多阶段响应评估模型能力。实验表明该方法比传统问答评估更准确，缩小了模型间性能差距，并通过消融研究验证有效性。**

- **链接: [http://arxiv.org/pdf/2507.23776v1](http://arxiv.org/pdf/2507.23776v1)**

> **作者:** Yunxiang Yan; Tomohiro Sawada; Kartik Goyal
>
> **备注:** Under review
>
> **摘要:** While question-answering~(QA) benchmark performance is an automatic and scalable method to compare LLMs, it is an indirect method of evaluating their underlying problem-solving capabilities. Therefore, we propose a holistic and generalizable framework based on \emph{cascaded question disclosure} that provides a more accurate estimate of the models' problem-solving capabilities while maintaining the scalability and automation. This approach collects model responses in a stagewise manner with each stage revealing partial information about the question designed to elicit generalized reasoning in LLMs. We find that our approach not only provides a better comparison between LLMs, but also induces better intermediate traces in models compared to the standard QA paradigm. We empirically verify this behavior on diverse reasoning and knowledge-heavy QA datasets by comparing LLMs of varying sizes and families. Our approach narrows the performance gap observed in the standard QA evaluation settings, indicating that the prevalent indirect QA paradigm of evaluation overestimates the differences in performance between models. We further validate our findings by extensive ablation studies.
>
---
#### [new 052] Failures Are the Stepping Stones to Success: Enhancing Few-Shot In-Context Learning by Leveraging Negative Samples
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的少样本上下文学习（ICL）效果。现有方法多依赖正样本，忽略负样本的作用。作者提出一种新方法，利用负样本辅助选择更优正样本，通过语义相似性从正负样本库中筛选示例，提升ICL性能。实验验证了该方法优于仅依赖正样本的选择方式。**

- **链接: [http://arxiv.org/pdf/2507.23211v1](http://arxiv.org/pdf/2507.23211v1)**

> **作者:** Yunhao Liang; Ruixuan Ying; Takuya Taniguchi; Zhe Cui
>
> **摘要:** Large Language Models exhibit powerful few-shot in-context learning (ICL) capabilities, but the performance is highly sensitive to provided examples. Recent research has focused on retrieving corresponding examples for each input query, not only enhancing the efficiency and scalability of the learning process but also mitigating inherent biases in manual example selection. However, these studies have primarily emphasized leveraging Positive samples while overlooking the additional information within Negative samples for contextual learning. We propose a novel method that utilizes Negative samples to better select Positive sample examples, thereby enhancing the performance of few-shot ICL. Initially, we construct Positive and Negative sample corpora based on Zero-Shot-Cot. Then, during inference, we employ a semantic similarity-based approach to select the most similar examples from both the Positive and Negative corpora for a given query. Subsequently, we further retrieve Positive examples from the Positive sample corpus based on semantic similarity to the Negative examples, then concatenating them with the previously selected Positive examples to serve as ICL demonstrations. Experimental results demonstrate that our approach surpasses methods solely relying on the most similar positive examples for context, validating that the additional information in negative samples aids in enhancing ICL performance through improved Positive sample selection.
>
---
#### [new 053] User Feedback in Human-LLM Dialogues: A Lens to Understand Users But Noisy as a Learning Signal
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.23158v1](http://arxiv.org/pdf/2507.23158v1)**

> **作者:** Yuhan Liu; Michael J. Q. Zhang; Eunsol Choi
>
> **备注:** Earlier version of this paper was presented at 2nd Workshop on Models of Human Feedback for AI Alignment (MoFA), ICML 2025
>
> **摘要:** Once language models (LMs) are deployed, they can interact with users long-term, ideally evolving continuously based on their feedback. Asking for direct user feedback can be disruptive; thus, we study harvesting user feedback from user-LM interaction logs. We study implicit user feedback in two user-LM interaction datasets (WildChat and LMSYS). First, we analyze user feedback in the user-LLM conversation trajectory, providing insights into when and why such feedback occurs. Second, we study harvesting learning signals from such implicit user feedback. We find that the contents of user feedback (e.g., user wanted clarification), not just the polarity (e.g., users were unhappy with the previous model response), can improve model performance in short human-designed questions (MTBench) but not on longer and more complex questions (WildBench). We also find that the usefulness of user feedback is largely tied to the quality of the user's initial prompt. Together, we provide an in-depth study of implicit user feedback, showing its potential and limitations.
>
---
#### [new 054] MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于音乐领域文本问答任务，旨在解决通用大模型在音乐知识上的不足。作者提出了MusT-RAG框架，结合检索增强生成技术与音乐专用数据库MusWikiDB，优化模型在音乐问答中的表现。实验显示其效果优于传统微调方法。**

- **链接: [http://arxiv.org/pdf/2507.23334v1](http://arxiv.org/pdf/2507.23334v1)**

> **作者:** Daeyong Kwon; SeungHeon Doh; Juhan Nam
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Recent advancements in Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains. While they exhibit strong zero-shot performance on various tasks, LLMs' effectiveness in music-related applications remains limited due to the relatively small proportion of music-specific knowledge in their training data. To address this limitation, we propose MusT-RAG, a comprehensive framework based on Retrieval Augmented Generation (RAG) to adapt general-purpose LLMs for text-only music question answering (MQA) tasks. RAG is a technique that provides external knowledge to LLMs by retrieving relevant context information when generating answers to questions. To optimize RAG for the music domain, we (1) propose MusWikiDB, a music-specialized vector database for the retrieval stage, and (2) utilizes context information during both inference and fine-tuning processes to effectively transform general-purpose LLMs into music-specific models. Our experiment demonstrates that MusT-RAG significantly outperforms traditional fine-tuning approaches in enhancing LLMs' music domain adaptation capabilities, showing consistent improvements across both in-domain and out-of-domain MQA benchmarks. Additionally, our MusWikiDB proves substantially more effective than general Wikipedia corpora, delivering superior performance and computational efficiency.
>
---
#### [new 055] Trusted Knowledge Extraction for Operations and Maintenance Intelligence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识抽取与运维智能任务，旨在解决在保障数据安全的前提下，如何有效构建运维领域知识图谱的问题。论文评估了多种NLP工具和大语言模型在零样本设置下的表现，探讨其在航空等关键行业应用的可行性，并发布了一个开源数据集用于基准测试。**

- **链接: [http://arxiv.org/pdf/2507.22935v1](http://arxiv.org/pdf/2507.22935v1)**

> **作者:** Kathleen Mealey; Jonathan A. Karr Jr.; Priscila Saboia Moreira; Paul R. Brenner; Charles F. Vardeman II
>
> **摘要:** Deriving operational intelligence from organizational data repositories is a key challenge due to the dichotomy of data confidentiality vs data integration objectives, as well as the limitations of Natural Language Processing (NLP) tools relative to the specific knowledge structure of domains such as operations and maintenance. In this work, we discuss Knowledge Graph construction and break down the Knowledge Extraction process into its Named Entity Recognition, Coreference Resolution, Named Entity Linking, and Relation Extraction functional components. We then evaluate sixteen NLP tools in concert with or in comparison to the rapidly advancing capabilities of Large Language Models (LLMs). We focus on the operational and maintenance intelligence use case for trusted applications in the aircraft industry. A baseline dataset is derived from a rich public domain US Federal Aviation Administration dataset focused on equipment failures or maintenance requirements. We assess the zero-shot performance of NLP and LLM tools that can be operated within a controlled, confidential environment (no data is sent to third parties). Based on our observation of significant performance limitations, we discuss the challenges related to trusted NLP and LLM tools as well as their Technical Readiness Level for wider use in mission-critical industries such as aviation. We conclude with recommendations to enhance trust and provide our open-source curated dataset to support further baseline testing and evaluation.
>
---
#### [new 056] Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI生成GPU内核代码任务，旨在解决手动优化低级GPU内核效率低、难度大的问题。作者提出GEAK框架，利用大语言模型生成高性能Triton代码，并引入推理时计算扩展和反馈机制优化生成效果。论文还提供了评估基准，实验表明GEAK在正确性和执行速度上均优于基线方法。**

- **链接: [http://arxiv.org/pdf/2507.23194v1](http://arxiv.org/pdf/2507.23194v1)**

> **作者:** Jianghui Wang; Vinay Joshi; Saptarshi Majumder; Xu Chao; Bin Ding; Ziqiong Liu; Pratik Prabhanjan Brahma; Dong Li; Zicheng Liu; Emad Barsoum
>
> **摘要:** The demand for AI-generated GPU kernels is rapidly growing, influenced by the need for scalable, hardware-optimized solutions in both industry and academia. As deep learning workloads grow in complexity and diversity, it is imperative to automate low-level kernel development to meet performance and productivity demands. Major cloud providers, semiconductor companies, and research institutions are now investing heavily in AI-driven code generation for GPUs, aiming to reduce manual optimization efforts while achieving near-expert performance on hardware like AMD MI300X. The Triton language, a Python-based DSL for GPU programming, has emerged as a popular target for such AI-generated kernels due to its balance of performance and ease-of-coding. In this work, we present an evaluation suite for Triton-based GPU kernels and GEAK (Generating Efficient AI-centric GPU Kernels)-a framework that leverages cutting-edge LLMs to generate performant Triton code specifically for AMD GPUs, including the AMD MI300X and MI250. GEAK leverages inference-time compute scaling to produce Triton-based GPU kernels using a reasoning loop adapted from Reflexion-style feedback mechanisms. On two evaluation benchmarks, GEAK significantly outperformed the baselines of directly prompting frontier LLMs as well as Reflexion-based generation pipelines by achieving correctness up to $63$% and execution speed up of up to $2.59$X. These results highlight the promise of GEAK-like agentic code generation for accelerating the adoption of diverse hardware platforms and democratizing access to expert-level kernel performance.
>
---
#### [new 057] Evaluating Large Language Models (LLMs) in Financial NLP: A Comparative Study on Financial Report Analysis
- **分类: cs.CL; cs.AI; cs.CE; cs.HC; q-fin.CP**

- **简介: 该论文属于金融自然语言处理任务，旨在比较不同大语言模型在财务报告分析中的表现。研究通过设计领域特定提示，评估五种主流模型，发现GPT表现最佳，其次是Claude和Perplexity，而Gemini和DeepSeek表现不稳定，结果受提示和数据影响较大。**

- **链接: [http://arxiv.org/pdf/2507.22936v1](http://arxiv.org/pdf/2507.22936v1)**

> **作者:** Md Talha Mohsin
>
> **备注:** 22 Pages, 6 Tables, 7 Figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide variety of Financial Natural Language Processing (FinNLP) tasks. However, systematic comparisons among widely used LLMs remain underexplored. Given the rapid advancement and growing influence of LLMs in financial analysis, this study conducts a thorough comparative evaluation of five leading LLMs, GPT, Claude, Perplexity, Gemini and DeepSeek, using 10-K filings from the 'Magnificent Seven' technology companies. We create a set of domain-specific prompts and then use three methodologies to evaluate model performance: human annotation, automated lexical-semantic metrics (ROUGE, Cosine Similarity, Jaccard), and model behavior diagnostics (prompt-level variance and across-model similarity). The results show that GPT gives the most coherent, semantically aligned, and contextually relevant answers; followed by Claude and Perplexity. Gemini and DeepSeek, on the other hand, have more variability and less agreement. Also, the similarity and stability of outputs change from company to company and over time, showing that they are sensitive to how prompts are written and what source material is used.
>
---
#### [new 058] Beyond Passive Critical Thinking: Fostering Proactive Questioning to Enhance Human-AI Collaboration
- **分类: cs.CL**

- **简介: 该论文属于人工智能任务，旨在解决当前AI在主动批判性思维上的不足。通过构建新基准GSM-MC和GSM-MCE，评估模型在信息不全或误导条件下主动提问的能力，并提出改进方法，利用强化学习提升Qwen3等模型的准确率，以促进人机协作解决问题。**

- **链接: [http://arxiv.org/pdf/2507.23407v1](http://arxiv.org/pdf/2507.23407v1)**

> **作者:** Ante Wang; Yujie Lin; Jingyao Liu; Suhang Wu; Hao Liu; Xinyan Xiao; Jinsong Su
>
> **摘要:** Critical thinking is essential for building robust AI systems, preventing them from blindly accepting flawed data or biased reasoning. However, prior work has primarily focused on passive critical thinking, where models simply reject problematic queries without taking constructive steps to address user requests. In this work, we introduce proactive critical thinking, a paradigm where models actively seek missing or clarifying information from users to resolve their queries better. To evaluate this capability, we present GSM-MC and GSM-MCE, two novel benchmarks based on GSM8K for assessing mathematical reasoning under incomplete or misleading conditions. GSM-MC contains 1,368 math problems with a key variable deliberately removed, requiring models to identify and request the missing information. GSM-MCE further increases the difficulty by introducing irrelevant details to test robustness against distractions. Experiments on Qwen3 and Llama series models show that, while these models excel in traditional reasoning tasks due to extensive post-training and inference-time scaling, they struggle with proactive critical thinking, especially smaller ones. However, we demonstrate that reinforcement learning (RL) can significantly improve this ability. Using our enhanced RL algorithm, we achieve substantial gains, boosting the Qwen3-1.7B's accuracy from 0.15% to 73.98% on GSM-MC. We hope this work advances models that collaborate more effectively with users in problem-solving through proactive critical thinking.
>
---
#### [new 059] What's Taboo for You? - An Empirical Evaluation of LLMs Behavior Toward Sensitive Content
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对敏感内容的隐性审查行为，分析GPT-4o-mini在改写敏感内容时是否自发降低其敏感性，并评估其敏感度变化。任务属于自然语言处理中的内容安全领域，旨在探究模型在无明确指令下是否会隐性过滤敏感词。**

- **链接: [http://arxiv.org/pdf/2507.23319v1](http://arxiv.org/pdf/2507.23319v1)**

> **作者:** Alfio Ferrara; Sergio Picascia; Laura Pinnavaia; Vojimir Ranitovic; Elisabetta Rocchetti; Alice Tuveri
>
> **摘要:** Proprietary Large Language Models (LLMs) have shown tendencies toward politeness, formality, and implicit content moderation. While previous research has primarily focused on explicitly training models to moderate and detoxify sensitive content, there has been limited exploration of whether LLMs implicitly sanitize language without explicit instructions. This study empirically analyzes the implicit moderation behavior of GPT-4o-mini when paraphrasing sensitive content and evaluates the extent of sensitivity shifts. Our experiments indicate that GPT-4o-mini systematically moderates content toward less sensitive classes, with substantial reductions in derogatory and taboo language. Also, we evaluate the zero-shot capabilities of LLMs in classifying sentence sensitivity, comparing their performances against traditional methods.
>
---
#### [new 060] Enhanced Arabic Text Retrieval with Attentive Relevance Scoring
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升阿拉伯语文本检索效果。针对阿拉伯语复杂形态、缺乏标注资源等问题，论文提出一种基于密集段落检索的增强框架，并引入注意力相关性评分机制，以更精准捕捉问题与文本间的语义关联，提高检索准确率。**

- **链接: [http://arxiv.org/pdf/2507.23404v1](http://arxiv.org/pdf/2507.23404v1)**

> **作者:** Salah Eddine Bekhouche; Azeddine Benlamoudi; Yazid Bounab; Fadi Dornaika; Abdenour Hadid
>
> **摘要:** Arabic poses a particular challenge for natural language processing (NLP) and information retrieval (IR) due to its complex morphology, optional diacritics and the coexistence of Modern Standard Arabic (MSA) and various dialects. Despite the growing global significance of Arabic, it is still underrepresented in NLP research and benchmark resources. In this paper, we present an enhanced Dense Passage Retrieval (DPR) framework developed specifically for Arabic. At the core of our approach is a novel Attentive Relevance Scoring (ARS) that replaces standard interaction mechanisms with an adaptive scoring function that more effectively models the semantic relevance between questions and passages. Our method integrates pre-trained Arabic language models and architectural refinements to improve retrieval performance and significantly increase ranking accuracy when answering Arabic questions. The code is made publicly available at \href{https://github.com/Bekhouche/APR}{GitHub}.
>
---
#### [new 061] Fast and Accurate Contextual Knowledge Extraction Using Cascading Language Model Chains and Candidate Answers
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出了一种名为Language Model Chain（LMC）的级联语言模型算法，用于快速准确地从文本中提取知识，特别是在医疗文档中提取患者出生日期。该方法通过结合多个语言模型，逐步过滤错误回答，减少幻觉现象，提高预测速度和准确性。论文属于自然语言处理中的知识提取任务，旨在解决语言模型易产生错误信息的问题。**

- **链接: [http://arxiv.org/pdf/2507.22921v1](http://arxiv.org/pdf/2507.22921v1)**

> **作者:** Lee Harris
>
> **摘要:** Language models can capture complex relationships in given text, but these are notorious for being costly and for producing information that does not exist (i.e., hallucinations). Furthermore, the resources invested into producing this information would be wasted if it were incorrect. We address these issues by proposing, implementing, and applying the Language Model Chain (LMC) algorithm. In this, a language model's response to a given prompt about given text is only correct if it exists in the collection of possible (i.e., candidate) answers, and text corresponding to incorrect responses is fed into a more predictive (but slower) language model. This process is repeated for a collection of language models, or until all predictions about the text are correct. We used the LMC algorithm to extract patient dates of birth from medical documents, and combining a collection of language models in a multi-stage cascade significantly increased prediction speed and accuracy over individual language models, while greatly reducing the number of corresponding hallucinations. We believe that the novel LMC algorithm significantly contributes to the knowledge extraction field, and that this should be explored much further in the future.
>
---
#### [new 062] Arabic Hate Speech Identification and Masking in Social Media using Deep Learning Models and Pre-trained Models Fine-tuning
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理任务，旨在识别并屏蔽社交媒体中的阿拉伯语仇恨言论。研究解决了仇恨言论检测与文本清洗两个问题，使用深度学习模型和预训练模型微调进行检测，将文本清洗视为机器翻译任务。最终检测模型达到92% Macro F1和95%准确率，清洗模型BLEU得分为0.3。**

- **链接: [http://arxiv.org/pdf/2507.23661v1](http://arxiv.org/pdf/2507.23661v1)**

> **作者:** Salam Thabet Doghmash; Motaz Saad
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Hate speech identification in social media has become an increasingly important issue in recent years. In this research, we address two problems: 1) to detect hate speech in Arabic text, 2) to clean a given text from hate speech. The meaning of cleaning here is replacing each bad word with stars based on the number of letters for each word. Regarding the first problem, we conduct several experiments using deep learning models and transformers to determine the best model in terms of the F1 score. Regarding second problem, we consider it as a machine translation task, where the input is a sentence containing dirty text and the output is the same sentence with masking the dirty text. The presented methods achieve the best model in hate speech detection with a 92\% Macro F1 score and 95\% accuracy. Regarding the text cleaning experiment, the best result in the hate speech masking model reached 0.3 in BLEU score with 1-gram, which is a good result compared with the state of the art machine translation systems.
>
---
#### [new 063] How and Where to Translate? The Impact of Translation Strategies in Cross-lingual LLM Prompting
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型中跨语言提示的翻译策略，旨在解决不同语言间知识共享与分类任务性能差异问题。通过实验评估不同翻译方法对检索增强生成系统的影响，发现优化提示策略可显著提升低资源语言表现，推动多语言资源共享与跨语言提示优化的应用。**

- **链接: [http://arxiv.org/pdf/2507.22923v1](http://arxiv.org/pdf/2507.22923v1)**

> **作者:** Aman Gupta; Yingying Zhuang; Zhou Yu; Ziji Zhang; Anurag Beniwal
>
> **备注:** Accepted at Prompt Optimization KDD '25
>
> **摘要:** Despite advances in the multilingual capabilities of Large Language Models (LLMs), their performance varies substantially across different languages and tasks. In multilingual retrieval-augmented generation (RAG)-based systems, knowledge bases (KB) are often shared from high-resource languages (such as English) to low-resource ones, resulting in retrieved information from the KB being in a different language than the rest of the context. In such scenarios, two common practices are pre-translation to create a mono-lingual prompt and cross-lingual prompting for direct inference. However, the impact of these choices remains unclear. In this paper, we systematically evaluate the impact of different prompt translation strategies for classification tasks with RAG-enhanced LLMs in multilingual systems. Experimental results show that an optimized prompting strategy can significantly improve knowledge sharing across languages, therefore improve the performance on the downstream classification task. The findings advocate for a broader utilization of multilingual resource sharing and cross-lingual prompt optimization for non-English languages, especially the low-resource ones.
>
---
#### [new 064] Reading Between the Timelines: RAG for Answering Diachronic Questions
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索与自然语言处理任务，旨在解决现有RAG模型在处理跨时间线索问题时证据收集的时序不连贯问题。作者提出一种融合时间逻辑的RAG新框架，通过解耦查询主题与时间窗口、引入时间感知检索器，提升了对时序连贯证据的收集能力。为验证方法有效性，构建了ADQAB评测基准，实验表明新方法在回答准确性上显著优于标准RAG模型。**

- **链接: [http://arxiv.org/pdf/2507.22917v1](http://arxiv.org/pdf/2507.22917v1)**

> **作者:** Kwun Hang Lau; Ruiyuan Zhang; Weijie Shi; Xiaofang Zhou; Xiaojun Cheng
>
> **摘要:** While Retrieval-Augmented Generation (RAG) excels at injecting static, factual knowledge into Large Language Models (LLMs), it exhibits a critical deficit in handling longitudinal queries that require tracking entities and phenomena across time. This blind spot arises because conventional, semantically-driven retrieval methods are not equipped to gather evidence that is both topically relevant and temporally coherent for a specified duration. We address this challenge by proposing a new framework that fundamentally redesigns the RAG pipeline to infuse temporal logic. Our methodology begins by disentangling a user's query into its core subject and its temporal window. It then employs a specialized retriever that calibrates semantic matching against temporal relevance, ensuring the collection of a contiguous evidence set that spans the entire queried period. To enable rigorous evaluation of this capability, we also introduce the Analytical Diachronic Question Answering Benchmark (ADQAB), a challenging evaluation suite grounded in a hybrid corpus of real and synthetic financial news. Empirical results on ADQAB show that our approach yields substantial gains in answer accuracy, surpassing standard RAG implementations by 13% to 27%. This work provides a validated pathway toward RAG systems capable of performing the nuanced, evolutionary analysis required for complex, real-world questions. The dataset and code for this study are publicly available at https://github.com/kwunhang/TA-RAG.
>
---
#### [new 065] T-Detect: Tail-Aware Statistical Normalization for Robust Detection of Adversarial Machine-Generated Text
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决对抗性机器生成文本的鲁棒检测问题。现有方法依赖高斯假设，难以应对对抗文本的重尾统计特征。论文提出T-Detect，采用基于t分布的重尾差异评分替代传统高斯归一化，提升检测性能。实验表明其在多个数据集上表现优越，尤其在RAID数据集的Books域达到0.926 AUROC。**

- **链接: [http://arxiv.org/pdf/2507.23577v1](http://arxiv.org/pdf/2507.23577v1)**

> **作者:** Alva West; Luodan Zhang; Liuliu Zhang; Minjun Zhu; Yixuan Weng; Yue Zhang
>
> **摘要:** The proliferation of sophisticated text generation models necessitates the development of robust detection methods capable of identifying machine-generated content, particularly text designed to evade detection through adversarial perturbations. Existing zero-shot detectors often rely on statistical measures that implicitly assume Gaussian distributions, a premise that falters when confronted with the heavy-tailed statistical artifacts characteristic of adversarial or non-native English texts. This paper introduces T-Detect, a novel detection method that fundamentally redesigns the statistical core of curvature-based detectors. Our primary innovation is the replacement of standard Gaussian normalization with a heavy-tailed discrepancy score derived from the Student's t-distribution. This approach is theoretically grounded in the empirical observation that adversarial texts exhibit significant leptokurtosis, rendering traditional statistical assumptions inadequate. T-Detect computes a detection score by normalizing the log-likelihood of a passage against the expected moments of a t-distribution, providing superior resilience to statistical outliers. We validate our approach on the challenging RAID benchmark for adversarial text and the comprehensive HART dataset. Experiments show that T-Detect provides a consistent performance uplift over strong baselines, improving AUROC by up to 3.9\% in targeted domains. When integrated into a two-dimensional detection framework (CT), our method achieves state-of-the-art performance, with an AUROC of 0.926 on the Books domain of RAID. Our contributions are a new, theoretically-justified statistical foundation for text detection, an ablation-validated method that demonstrates superior robustness, and a comprehensive analysis of its performance under adversarial conditions. Ours code are released at https://github.com/ResearAI/t-detect.
>
---
#### [new 066] Med-R$^3$: Enhancing Medical Retrieval-Augmented Reasoning of LLMs via Progressive Reinforcement Learning
- **分类: cs.CL**

- **简介: 论文提出Med-R³框架，旨在通过渐进式强化学习提升大语言模型在医疗领域的检索增强推理能力。该研究属于医疗自然语言处理任务，解决现有方法在检索与推理联合优化不足、泛化能力弱及医疗奖励机制不匹配的问题，实现了检索与推理的协同优化，并取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2507.23541v1](http://arxiv.org/pdf/2507.23541v1)**

> **作者:** Keer Lu; Zheng Liang; Youquan Li; Jiejun Tan; Da Pan; Shusen Zhang; Guosheng Dong; Huang Leng
>
> **摘要:** In medical scenarios, effectively retrieving external knowledge and leveraging it for rigorous logical reasoning is of significant importance. Despite their potential, existing work has predominantly focused on enhancing either retrieval or reasoning capabilities of the models in isolation, with little attention given to their joint optimization, which leads to limited coordination between the two processes. Additionally, current methods rely heavily on supervised fine-tuning (SFT), which can cause models to memorize existing problem-solving pathways, thereby restricting their generalization ability when confronted with novel problem contexts. Furthermore, while some studies have explored to improve retrieval-augmented reasoning in general domains via reinforcement learning, their reward function designs do not adequately capture the specific demands of the medical domain. To address these challenges, we introduce **Med-R$^3$**, a **Med**ical **R**etrieval-augmented **R**easoning framework driven by progressive **R**einforcement learning. In this framework, we first develop the model's ability to perform logical reasoning over medical problems. Subsequently, on the basis of this foundation, we adaptively optimize the retrieval capability to better align with the characteristics of knowledge corpus and external information utilization throughout the reasoning process. Finally, we conduct joint optimization of the model's retrieval and reasoning coordination. Extensive experiments indicate that **Med-R$^3$** could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + Med-R$^3$ surpassing closed-sourced GPT-4o-mini by 3.93\% at a comparable parameter scale, while Qwen2.5-14B augmented with Med-R$^3$ shows a more substantial gain of 13.53\%.
>
---
#### [new 067] SigBERT: Combining Narrative Medical Reports and Rough Path Signature Theory for Survival Risk Estimation in Oncology
- **分类: cs.CL; cs.CY; cs.LG; stat.AP**

- **简介: 论文提出SigBERT，用于肿瘤学中的生存风险估计任务。它结合医学报告与粗糙路径签名理论，处理时序文本数据，提取几何特征并融合至Cox模型，提升风险预测性能。**

- **链接: [http://arxiv.org/pdf/2507.22941v1](http://arxiv.org/pdf/2507.22941v1)**

> **作者:** Paul Minchella; Loïc Verlingue; Stéphane Chrétien; Rémi Vaucher; Guillaume Metzler
>
> **备注:** 12 pages, 2 figures, accepted for ECML PKDD 2025
>
> **摘要:** Electronic medical reports (EHR) contain a vast amount of information that can be leveraged for machine learning applications in healthcare. However, existing survival analysis methods often struggle to effectively handle the complexity of textual data, particularly in its sequential form. Here, we propose SigBERT, an innovative temporal survival analysis framework designed to efficiently process a large number of clinical reports per patient. SigBERT processes timestamped medical reports by extracting and averaging word embeddings into sentence embeddings. To capture temporal dynamics from the time series of sentence embedding coordinates, we apply signature extraction from rough path theory to derive geometric features for each patient, which significantly enhance survival model performance by capturing complex temporal dynamics. These features are then integrated into a LASSO-penalized Cox model to estimate patient-specific risk scores. The model was trained and evaluated on a real-world oncology dataset from the L\'eon B\'erard Center corpus, with a C-index score of 0.75 (sd 0.014) on the independent test cohort. SigBERT integrates sequential medical data to enhance risk estimation, advancing narrative-based survival analysis.
>
---
#### [new 068] Toward the Autonomous AI Doctor: Quantitative Benchmarking of an Autonomous Agentic AI Versus Board-Certified Clinicians in a Real World Setting
- **分类: cs.HC; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于医疗AI任务，旨在解决医疗从业者短缺和行政负担问题。研究通过对比自主AI医生系统Doctronic与认证医生在500例虚拟急诊诊疗中的诊断与治疗一致性，验证AI的自主诊疗能力。结果显示AI表现与医生相当甚至更优。**

- **链接: [http://arxiv.org/pdf/2507.22902v1](http://arxiv.org/pdf/2507.22902v1)**

> **作者:** Hashim Hayat; Maksim Kudrautsau; Evgeniy Makarov; Vlad Melnichenko; Tim Tsykunou; Piotr Varaksin; Matt Pavelle; Adam Z. Oskowitz
>
> **摘要:** Background: Globally we face a projected shortage of 11 million healthcare practitioners by 2030, and administrative burden consumes 50% of clinical time. Artificial intelligence (AI) has the potential to help alleviate these problems. However, no end-to-end autonomous large language model (LLM)-based AI system has been rigorously evaluated in real-world clinical practice. In this study, we evaluated whether a multi-agent LLM-based AI framework can function autonomously as an AI doctor in a virtual urgent care setting. Methods: We retrospectively compared the performance of the multi-agent AI system Doctronic and board-certified clinicians across 500 consecutive urgent-care telehealth encounters. The primary end points: diagnostic concordance, treatment plan consistency, and safety metrics, were assessed by blinded LLM-based adjudication and expert human review. Results: The top diagnosis of Doctronic and clinician matched in 81% of cases, and the treatment plan aligned in 99.2% of cases. No clinical hallucinations occurred (e.g., diagnosis or treatment not supported by clinical findings). In an expert review of discordant cases, AI performance was superior in 36.1%, and human performance was superior in 9.3%; the diagnoses were equivalent in the remaining cases. Conclusions: In this first large-scale validation of an autonomous AI doctor, we demonstrated strong diagnostic and treatment plan concordance with human clinicians, with AI performance matching and in some cases exceeding that of practicing clinicians. These findings indicate that multi-agent AI systems achieve comparable clinical decision-making to human providers and offer a potential solution to healthcare workforce shortages.
>
---
#### [new 069] TextQuests: How Good are LLMs at Text-Based Video Games?
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI代理评估任务，旨在解决现有基准无法全面评估LLM在复杂探索环境中长期推理能力的问题。论文构建了TextQuests基准，基于Infocom文字游戏，测试LLM在无外部工具下的自主推理与持续问题解决能力。**

- **链接: [http://arxiv.org/pdf/2507.23701v1](http://arxiv.org/pdf/2507.23701v1)**

> **作者:** Long Phan; Mantas Mazeika; Andy Zou; Dan Hendrycks
>
> **摘要:** Evaluating AI agents within complex, interactive environments that mirror real-world challenges is critical for understanding their practical capabilities. While existing agent benchmarks effectively assess skills like tool use or performance on structured tasks, they often do not fully capture an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context. To spur the development of agents capable of more robust intrinsic reasoning over long horizons, we introduce TextQuests, a benchmark based on the Infocom suite of interactive fiction games. These text-based adventures, which can take human players over 30 hours and require hundreds of precise actions to solve, serve as an effective proxy for evaluating AI agents on focused, stateful tasks. The benchmark is specifically designed to assess an LLM agent's capacity for self-contained problem-solving by precluding the use of external tools, thereby focusing on intrinsic long-context reasoning capabilities in an exploratory environment characterized by the need for trial-and-error learning and sustained problem-solving within a single interactive session. We release TextQuests at https://textquests.ai.
>
---
#### [new 070] Counterfactual Evaluation for Blind Attack Detection in LLM-based Evaluation Systems
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决基于大语言模型（LLM）的评估系统面临的“盲攻”威胁。作者提出一种结合标准评估（SE）与反事实评估（CFE）的框架，通过在虚假答案下重新评估检测攻击，有效提升了系统的安全性。**

- **链接: [http://arxiv.org/pdf/2507.23453v1](http://arxiv.org/pdf/2507.23453v1)**

> **作者:** Lijia Liu; Takumi Kondo; Kyohei Atarashi; Koh Takeuchi; Jiyi Li; Shigeru Saito; Hisashi Kashima
>
> **摘要:** This paper investigates defenses for LLM-based evaluation systems against prompt injection. We formalize a class of threats called blind attacks, where a candidate answer is crafted independently of the true answer to deceive the evaluator. To counter such attacks, we propose a framework that augments Standard Evaluation (SE) with Counterfactual Evaluation (CFE), which re-evaluates the submission against a deliberately false ground-truth answer. An attack is detected if the system validates an answer under both standard and counterfactual conditions. Experiments show that while standard evaluation is highly vulnerable, our SE+CFE framework significantly improves security by boosting attack detection with minimal performance trade-offs.
>
---
#### [new 071] Deep Learning-based Prediction of Clinical Trial Enrollment with Uncertainty Estimates
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于临床试验预测任务，旨在解决患者入组数量预测问题。作者提出一种基于深度学习的方法，结合预训练语言模型和表格特征，通过注意力机制融合信息，并引入概率层进行不确定性估计，有效预测临床试验的入组情况。**

- **链接: [http://arxiv.org/pdf/2507.23607v1](http://arxiv.org/pdf/2507.23607v1)**

> **作者:** Tien Huu Do; Antoine Masquelier; Nae Eoun Lee; Jonathan Crowther
>
> **摘要:** Clinical trials are a systematic endeavor to assess the safety and efficacy of new drugs or treatments. Conducting such trials typically demands significant financial investment and meticulous planning, highlighting the need for accurate predictions of trial outcomes. Accurately predicting patient enrollment, a key factor in trial success, is one of the primary challenges during the planning phase. In this work, we propose a novel deep learning-based method to address this critical challenge. Our method, implemented as a neural network model, leverages pre-trained language models (PLMs) to capture the complexities and nuances of clinical documents, transforming them into expressive representations. These representations are then combined with encoded tabular features via an attention mechanism. To account for uncertainties in enrollment prediction, we enhance the model with a probabilistic layer based on the Gamma distribution, which enables range estimation. We apply the proposed model to predict clinical trial duration, assuming site-level enrollment follows a Poisson-Gamma process. We carry out extensive experiments on real-world clinical trial data, and show that the proposed method can effectively predict the number of patients enrolled at a number of sites for a given clinical trial, outperforming established baseline models.
>
---
#### [new 072] Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动化定理证明任务，旨在解决数学定理证明中缺乏有效监督信号的问题。作者提出了Seed-Prover模型，结合Lean形式验证反馈和自总结机制，实现深度与广度推理。在IMO和PutnamBench等数学问题上表现优异，并引入几何推理引擎Seed-Geometry，最终在IMO 2025中证明了5道题。**

- **链接: [http://arxiv.org/pdf/2507.23726v1](http://arxiv.org/pdf/2507.23726v1)**

> **作者:** Luoxin Chen; Jinming Gu; Liankai Huang; Wenhao Huang; Zhicheng Jiang; Allan Jie; Xiaoran Jin; Xing Jin; Chenggang Li; Kaijing Ma; Cheng Ren; Jiawei Shen; Wenlei Shi; Tong Sun; He Sun; Jiahui Wang; Siran Wang; Zhihong Wang; Chenrui Wei; Shufa Wei; Yonghui Wu; Yuchen Wu; Yihang Xia; Huajian Xin; Fan Yang; Huaiyuan Ying; Hongyi Yuan; Zheng Yuan; Tianyang Zhan; Chi Zhang; Yue Zhang; Ge Zhang; Tianyun Zhao; Jianqiu Zhao; Yichi Zhou; Thomas Hanwen Zhu
>
> **摘要:** LLMs have demonstrated strong mathematical reasoning abilities by leveraging reinforcement learning with long chain-of-thought, yet they continue to struggle with theorem proving due to the lack of clear supervision signals when solely using natural language. Dedicated domain-specific languages like Lean provide clear supervision via formal verification of proofs, enabling effective training through reinforcement learning. In this work, we propose \textbf{Seed-Prover}, a lemma-style whole-proof reasoning model. Seed-Prover can iteratively refine its proof based on Lean feedback, proved lemmas, and self-summarization. To solve IMO-level contest problems, we design three test-time inference strategies that enable both deep and broad reasoning. Seed-Prover proves $78.1\%$ of formalized past IMO problems, saturates MiniF2F, and achieves over 50\% on PutnamBench, outperforming the previous state-of-the-art by a large margin. To address the lack of geometry support in Lean, we introduce a geometry reasoning engine \textbf{Seed-Geometry}, which outperforms previous formal geometry engines. We use these two systems to participate in IMO 2025 and fully prove 5 out of 6 problems. This work represents a significant advancement in automated mathematical reasoning, demonstrating the effectiveness of formal verification with long chain-of-thought reasoning.
>
---
#### [new 073] Hybrid EEG--Driven Brain--Computer Interface: A Large Language Model Framework for Personalized Language Rehabilitation
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出一种基于脑电图（EEG）与大语言模型（LLM）的混合脑机接口框架，用于个性化语言康复。任务是语言康复辅助，旨在解决传统交流系统无法实时适应用户认知与语言需求的问题。论文工作包括：利用EEG识别神经意图，结合LLM生成个性化语言内容，实现实时调整任务难度与康复内容。**

- **链接: [http://arxiv.org/pdf/2507.22892v1](http://arxiv.org/pdf/2507.22892v1)**

> **作者:** Ismail Hossain; Mridul Banik
>
> **摘要:** Conventional augmentative and alternative communication (AAC) systems and language-learning platforms often fail to adapt in real time to the user's cognitive and linguistic needs, especially in neurological conditions such as post-stroke aphasia or amyotrophic lateral sclerosis. Recent advances in noninvasive electroencephalography (EEG)--based brain-computer interfaces (BCIs) and transformer--based large language models (LLMs) offer complementary strengths: BCIs capture users' neural intent with low fatigue, while LLMs generate contextually tailored language content. We propose and evaluate a novel hybrid framework that leverages real-time EEG signals to drive an LLM-powered language rehabilitation assistant. This system aims to: (1) enable users with severe speech or motor impairments to navigate language-learning modules via mental commands; (2) dynamically personalize vocabulary, sentence-construction exercises, and corrective feedback; and (3) monitor neural markers of cognitive effort to adjust task difficulty on the fly.
>
---
#### [new 074] Voice-guided Orchestrated Intelligence for Clinical Evaluation (VOICE): A Voice AI Agent System for Prehospital Stroke Assessment
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于医疗AI任务，旨在解决院前卒中评估不准确的问题。作者开发了语音引导AI系统VOICE，辅助非专业人员完成卒中评估，并通过视频记录供专家复核。实验显示其卒中识别准确率较高，但存在误判，仍需专家审核。**

- **链接: [http://arxiv.org/pdf/2507.22898v1](http://arxiv.org/pdf/2507.22898v1)**

> **作者:** Julian Acosta; Scott Adams; Julius Kernbach; Romain Hardy; Sung Eun Kim; Luyang Luo; Xiaoman Zhang; Shreya Johri; Mohammed Baharoon; Pranav Rajpurkar
>
> **摘要:** We developed a voice-driven artificial intelligence (AI) system that guides anyone - from paramedics to family members - through expert-level stroke evaluations using natural conversation, while also enabling smartphone video capture of key examination components for documentation and potential expert review. This addresses a critical gap in emergency care: current stroke recognition by first responders is inconsistent and often inaccurate, with sensitivity for stroke detection as low as 58%, causing life-threatening delays in treatment. Three non-medical volunteers used our AI system to assess ten simulated stroke patients, including cases with likely large vessel occlusion (LVO) strokes and stroke-like conditions, while we measured diagnostic accuracy, completion times, user confidence, and expert physician review of the AI-generated reports. The AI system correctly identified 84% of individual stroke signs and detected 75% of likely LVOs, completing evaluations in just over 6 minutes. Users reported high confidence (median 4.5/5) and ease of use (mean 4.67/5). The system successfully identified 86% of actual strokes but also incorrectly flagged 2 of 3 non-stroke cases as strokes. When an expert physician reviewed the AI reports with videos, they identified the correct diagnosis in 100% of cases, but felt confident enough to make preliminary treatment decisions in only 40% of cases due to observed AI errors including incorrect scoring and false information. While the current system's limitations necessitate human oversight, ongoing rapid advancements in speech-to-speech AI models suggest that future versions are poised to enable highly accurate assessments. Achieving human-level voice interaction could transform emergency medical care, putting expert-informed assessment capabilities in everyone's hands.
>
---
#### [new 075] SequenceLayers: Sequence Processing and Streaming Neural Networks Made Easy
- **分类: cs.LG; cs.CL; cs.PL; cs.SE; eess.AS**

- **简介: 该论文属于序列建模任务，旨在解决序列处理中模型在流式和并行计算中的兼容性与正确性问题。论文提出了SequenceLayers库，统一了层式与步式执行模式，确保模型流式运行的同时保持训练与推理一致性，简化了生产级模型构建。**

- **链接: [http://arxiv.org/pdf/2507.23292v1](http://arxiv.org/pdf/2507.23292v1)**

> **作者:** RJ Skerry-Ryan; Julian Salazar; Soroosh Mariooryad; David Kao; Daisy Stanton; Eric Battenberg; Matt Shannon; Ron J. Weiss; Robin Scheibler; Jonas Rothfuss; Tom Bagby
>
> **摘要:** We introduce a neural network layer API and library for sequence modeling, designed for easy creation of sequence models that can be executed both layer-by-layer (e.g., teacher-forced training) and step-by-step (e.g., autoregressive sampling). To achieve this, layers define an explicit representation of their state over time (e.g., a Transformer KV cache, a convolution buffer, an RNN hidden state), and a step method that evolves that state, tested to give identical results to a stateless layer-wise invocation. This and other aspects of the SequenceLayers contract enables complex models to be immediately streamable, mitigates a wide range of common bugs arising in both streaming and parallel sequence processing, and can be implemented in any deep learning library. A composable and declarative API, along with a comprehensive suite of layers and combinators, streamlines the construction of production-scale models from simple streamable components while preserving strong correctness guarantees. Our current implementations of SequenceLayers (JAX, TensorFlow 2) are available at https://github.com/google/sequence-layers.
>
---
#### [new 076] CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks
- **分类: cs.AI; cs.CL**

- **简介: 论文提出CoT-Self-Instruct，用于生成高质量合成提示的数据生成方法，适用于推理与非推理任务。通过链式思维引导LLM生成类似质量的新提示，并进行自动筛选，以提升训练数据质量。该方法在多个推理任务和指令跟随任务上均优于现有数据集。**

- **链接: [http://arxiv.org/pdf/2507.23751v1](http://arxiv.org/pdf/2507.23751v1)**

> **作者:** Ping Yu; Jack Lanchantin; Tianlu Wang; Weizhe Yuan; Olga Golovneva; Ilia Kulikov; Sainbayar Sukhbaatar; Jason Weston; Jing Xu
>
> **摘要:** We propose CoT-Self-Instruct, a synthetic data generation method that instructs LLMs to first reason and plan via Chain-of-Thought (CoT) based on the given seed tasks, and then to generate a new synthetic prompt of similar quality and complexity for use in LLM training, followed by filtering for high-quality data with automatic metrics. In verifiable reasoning, our synthetic data significantly outperforms existing training datasets, such as s1k and OpenMathReasoning, across MATH500, AMC23, AIME24 and GPQA-Diamond. For non-verifiable instruction-following tasks, our method surpasses the performance of human or standard self-instruct prompts on both AlpacaEval 2.0 and Arena-Hard.
>
---
#### [new 077] TweakLLM: A Routing Architecture for Dynamic Tailoring of Cached Responses
- **分类: cs.LG; cs.CL**

- **简介: 论文提出TweakLLM，一种动态调整缓存响应的路由架构，用于提升大型语言模型（LLM）的缓存效率。该工作属于自然语言处理中的模型优化任务，旨在解决缓存响应与用户个性化查询之间相关性不足的问题。通过使用轻量级LLM对缓存内容进行动态调整，结合用户研究与多模型对比，验证其在保持响应质量的同时显著提升缓存命中率，适用于高并发LLM部署场景。**

- **链接: [http://arxiv.org/pdf/2507.23674v1](http://arxiv.org/pdf/2507.23674v1)**

> **作者:** Muhammad Taha Cheema; Abeer Aamir; Khawaja Gul Muhammad; Naveed Anwar Bhatti; Ihsan Ayyub Qazi; Zafar Ayyub Qazi
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) process millions of queries daily, making efficient response caching a compelling optimization for reducing cost and latency. However, preserving relevance to user queries using this approach proves difficult due to the personalized nature of chatbot interactions and the limited accuracy of semantic similarity search. To address this, we present TweakLLM, a novel routing architecture that employs a lightweight LLM to dynamically adapt cached responses to incoming prompts. Through comprehensive evaluation, including user studies with side-by-side comparisons, satisfaction voting, as well as multi-agent LLM debates, we demonstrate that TweakLLM maintains response quality comparable to frontier models while significantly improving cache effectiveness. Our results across real-world datasets highlight TweakLLM as a scalable, resource-efficient caching solution for high-volume LLM deployments without compromising user experience.
>
---
#### [new 078] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于音频理解任务，旨在解决现有模型在细粒度音频理解上的不足。作者提出了MECAT基准和DATE评估指标，通过专家模型与大语言模型结合生成细粒度标注，提升模型对细节的捕捉能力，评估显示现有模型仍有改进空间。**

- **链接: [http://arxiv.org/pdf/2507.23511v1](http://arxiv.org/pdf/2507.23511v1)**

> **作者:** Yadong Niu; Tianzi Wang; Heinrich Dinkel; Xingwei Sun; Jiahao Zhou; Gang Li; Jizhong Liu; Xunying Liu; Junbo Zhang; Jian Luan
>
> **备注:** 9 main pages, 5 figures, 3 tables, and 14 appendix pages
>
> **摘要:** While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at https://github.com/xiaomi-research/mecat
>
---
#### [new 079] ELMES: An Automated Framework for Evaluating Large Language Models in Educational Scenarios
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文提出了ELMES，一个用于评估教育场景中大语言模型（LLMs）的自动化框架。任务是解决当前LLMs在教育应用中缺乏适配评估指标的问题。工作包括构建模块化框架、设计多智能体对话机制、开发LLM-as-a-Judge评估方法，并对多个教育场景进行系统评测。**

- **链接: [http://arxiv.org/pdf/2507.22947v1](http://arxiv.org/pdf/2507.22947v1)**

> **作者:** Shou'ang Wei; Xinyun Wang; Shuzhen Bi; Jian Chen; Ruijia Li; Bo Jiang; Xin Lin; Min Zhang; Yu Song; BingDong Li; Aimin Zhou; Hao Hao
>
> **摘要:** The emergence of Large Language Models (LLMs) presents transformative opportunities for education, generating numerous novel application scenarios. However, significant challenges remain: evaluation metrics vary substantially across different educational scenarios, while many emerging scenarios lack appropriate assessment metrics. Current benchmarks predominantly measure general intelligence rather than pedagogical capabilities. To address this gap, we introduce ELMES, an open-source automated evaluation framework specifically designed for assessing LLMs in educational settings. ELMES features a modular architecture that enables researchers to create dynamic, multi-agent dialogues through simple configuration files, facilitating flexible scenario design without requiring extensive programming expertise. The framework incorporates a hybrid evaluation engine that objectively quantifies traditionally subjective pedagogical metrics using an LLM-as-a-Judge methodology. We conduct systematic benchmarking of state-of-the-art LLMs across four critical educational scenarios: Knowledge Point Explanation, Guided Problem-Solving Teaching, Interdisciplinary Lesson Plan Generation, and Contextualized Question Generation, employing fine-grained metrics developed in collaboration with education specialists. Our results demonstrate distinct capability distributions among models, revealing context-specific strengths and limitations. ELMES provides educators and researchers with an accessible evaluation framework that significantly reduces adaptation barriers for diverse educational applications while advancing the practical implementation of LLMs in pedagogy. The framework is publicly available at \emph{https://github.com/sii-research/elmes.git}.
>
---
#### [new 080] SWE-Debate: Competitive Multi-Agent Debate for Software Issue Resolution
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文属于软件工程任务，旨在解决现有单智能体在代码问题定位中易陷入局部解的问题。提出SWE-Debate框架，通过多智能体辩论生成多样化定位方案，并结合MCTS生成修复补丁。实验表明其在SWE-bench基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.23348v1](http://arxiv.org/pdf/2507.23348v1)**

> **作者:** Han Li; Yuling Shi; Shaoxin Lin; Xiaodong Gu; Heng Lian; Xin Wang; Yantao Jia; Tao Huang; Qianxiang Wang
>
> **备注:** Our code and data are available at https://github.com/YerbaPage/SWE-Debate
>
> **摘要:** Issue resolution has made remarkable progress thanks to the advanced reasoning capabilities of large language models (LLMs). Recently, agent-based frameworks such as SWE-agent have further advanced this progress by enabling autonomous, tool-using agents to tackle complex software engineering tasks. While existing agent-based issue resolution approaches are primarily based on agents' independent explorations, they often get stuck in local solutions and fail to identify issue patterns that span across different parts of the codebase. To address this limitation, we propose SWE-Debate, a competitive multi-agent debate framework that encourages diverse reasoning paths and achieves more consolidated issue localization. SWE-Debate first creates multiple fault propagation traces as localization proposals by traversing a code dependency graph. Then, it organizes a three-round debate among specialized agents, each embodying distinct reasoning perspectives along the fault propagation trace. This structured competition enables agents to collaboratively converge on a consolidated fix plan. Finally, this consolidated fix plan is integrated into an MCTS-based code modification agent for patch generation. Experiments on the SWE-bench benchmark show that SWE-Debate achieves new state-of-the-art results in open-source agent frameworks and outperforms baselines by a large margin.
>
---
#### [new 081] Generalized Reinforcement Learning for Retriever-Specific Query Rewriter with Unstructured Real-World Documents
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于检索增强生成（RAG）系统中的查询优化任务，旨在解决在多样、非结构化现实文档中有效查询构建的挑战。论文提出了RL-QR，一种基于强化学习的检索器专用查询重写框架，无需人工标注数据，适用于文本和多模态数据库。通过合成场景-问题对并采用GRPO训练策略，提升了检索性能，尤其在多模态和词法检索中取得显著改进。**

- **链接: [http://arxiv.org/pdf/2507.23242v1](http://arxiv.org/pdf/2507.23242v1)**

> **作者:** Sungguk Cha; DongWook Kim; Taeseung Hahn; Mintae Kim; Youngsub Han; Byoung-Ki Jeon
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems rely heavily on effective query formulation to unlock external knowledge, yet optimizing queries for diverse, unstructured real-world documents remains a challenge. We introduce \textbf{RL-QR}, a reinforcement learning framework for retriever-specific query rewriting that eliminates the need for human-annotated datasets and extends applicability to both text-only and multi-modal databases. By synthesizing scenario-question pairs and leveraging Generalized Reward Policy Optimization (GRPO), RL-QR trains query rewriters tailored to specific retrievers, enhancing retrieval performance across varied domains. Experiments on industrial in-house data demonstrate significant improvements, with $\text{RL-QR}_{\text{multi-modal}}$ achieving an 11\% relative gain in NDCG@3 for multi-modal RAG and $\text{RL-QR}_{\text{lexical}}$ yielding a 9\% gain for lexical retrievers. However, challenges persist with semantic and hybrid retrievers, where rewriters failed to improve performance, likely due to training misalignments. Our findings highlight RL-QR's potential to revolutionize query optimization for RAG systems, offering a scalable, annotation-free solution for real-world retrieval tasks, while identifying avenues for further refinement in semantic retrieval contexts.
>
---
#### [new 082] SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning Architecture with LLM-Based World Model
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于人工智能代理任务，旨在解决当前AI代理在通用性和可扩展性上的不足。论文提出SimuRA架构，通过基于大语言模型的模拟世界模型实现规划，提升目标导向任务的表现。实验表明其在复杂网页浏览任务中显著提高成功率。**

- **链接: [http://arxiv.org/pdf/2507.23773v1](http://arxiv.org/pdf/2507.23773v1)**

> **作者:** Mingkai Deng; Jinyu Hou; Yilin Shen; Hongxia Jin; Graham Neubig; Zhiting Hu; Eric Xing
>
> **摘要:** AI agents built on large language models (LLMs) hold enormous promise, but current practice focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also suffers from the fundamental limitations of autoregressive LLMs. On the other hand, humans are general agents who reason by mentally simulating the outcomes of their actions and plans. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of optimal agent in any environment, \modelname overcomes the limitations of autoregressive reasoning by introducing a world model for planning via simulation. The generalized world model is implemented using LLM, which can flexibly plan in a wide range of environments using the concept-rich latent space of natural language. Experiments on difficult web browsing tasks show that \modelname improves the success of flight search from 0\% to 32.2\%. World-model-based planning, in particular, shows consistent advantage of up to 124\% over autoregressive planning, demonstrating the advantage of world model simulation as a reasoning paradigm. We are excited about the possibility for training a single, general agent model based on LLMs that can act superintelligently in all environments. To start, we make SimuRA, a web-browsing agent built on \modelname with pretrained LLMs, available as a research demo for public testing.
>
---
#### [new 083] Exploring Dynamic Parameters for Vietnamese Gender-Independent ASR
- **分类: eess.AS; cs.CL; cs.SD; eess.SP**

- **简介: 该论文属于语音识别任务，旨在提升越南语自动语音识别（ASR）的性别无关性能。通过引入基于子带质心频率的动态参数，结合传统MFCC特征，有效减少频谱变化并增强声学建模，从而降低词错误率，并提高对不同性别的适应性。**

- **链接: [http://arxiv.org/pdf/2507.22964v1](http://arxiv.org/pdf/2507.22964v1)**

> **作者:** Sotheara Leang; Éric Castelli; Dominique Vaufreydaz; Sethserey Sam
>
> **摘要:** The dynamic characteristics of speech signal provides temporal information and play an important role in enhancing Automatic Speech Recognition (ASR). In this work, we characterized the acoustic transitions in a ratio plane of Spectral Subband Centroid Frequencies (SSCFs) using polar parameters to capture the dynamic characteristics of the speech and minimize spectral variation. These dynamic parameters were combined with Mel-Frequency Cepstral Coefficients (MFCCs) in Vietnamese ASR to capture more detailed spectral information. The SSCF0 was used as a pseudo-feature for the fundamental frequency (F0) to describe the tonal information robustly. The findings showed that the proposed parameters significantly reduce word error rates and exhibit greater gender independence than the baseline MFCCs.
>
---
#### [new 084] DSBC : Data Science task Benchmarking with Context engineering
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于数据科学任务基准构建任务，旨在解决缺乏系统评估数据科学代理效能的问题。论文提出了DSBC基准，评估三种大语言模型在不同方法下的表现，分析其对提示问题和温度参数的敏感性，为未来研究提供基础。**

- **链接: [http://arxiv.org/pdf/2507.23336v1](http://arxiv.org/pdf/2507.23336v1)**

> **作者:** Ram Mohan Rao Kadiyala; Siddhant Gupta; Jebish Purbey; Giulio Martini; Suman Debnath; Hamza Farooq
>
> **备注:** 32 pages
>
> **摘要:** Recent advances in large language models (LLMs) have significantly impacted data science workflows, giving rise to specialized data science agents designed to automate analytical tasks. Despite rapid adoption, systematic benchmarks evaluating the efficacy and limitations of these agents remain scarce. In this paper, we introduce a comprehensive benchmark specifically crafted to reflect real-world user interactions with data science agents by observing usage of our commercial applications. We evaluate three LLMs: Claude-4.0-Sonnet, Gemini-2.5-Flash, and OpenAI-o4-Mini across three approaches: zero-shot with context engineering, multi-step with context engineering, and with SmolAgent. Our benchmark assesses performance across a diverse set of eight data science task categories, additionally exploring the sensitivity of models to common prompting issues, such as data leakage and slightly ambiguous instructions. We further investigate the influence of temperature parameters on overall and task-specific outcomes for each model and approach. Our findings reveal distinct performance disparities among the evaluated models and methodologies, highlighting critical factors that affect practical deployment. The benchmark dataset and evaluation framework introduced herein aim to provide a foundation for future research of more robust and effective data science agents.
>
---
#### [new 085] Holistic Evaluations of Topic Models
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决主题模型的评估与合理使用问题。论文从数据库视角出发，通过分析1140次BERTopic模型运行结果，探讨模型参数优化的权衡，并反思其对主题模型解释与应用的影响，以促进其负责任的使用。**

- **链接: [http://arxiv.org/pdf/2507.23364v1](http://arxiv.org/pdf/2507.23364v1)**

> **作者:** Thomas Compton
>
> **备注:** 10 pages, 6 tables
>
> **摘要:** Topic models are gaining increasing commercial and academic interest for their ability to summarize large volumes of unstructured text. As unsupervised machine learning methods, they enable researchers to explore data and help general users understand key themes in large text collections. However, they risk becoming a 'black box', where users input data and accept the output as an accurate summary without scrutiny. This article evaluates topic models from a database perspective, drawing insights from 1140 BERTopic model runs. The goal is to identify trade-offs in optimizing model parameters and to reflect on what these findings mean for the interpretation and responsible use of topic models
>
---
#### [new 086] SWE-Exp: Experience-Driven Software Issue Resolution
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文属于软件工程任务，旨在解决当前基于大语言模型的软件问题修复代理缺乏经验积累的问题。通过构建经验库，提取修复过程中的成功与失败经验，实现跨问题的持续学习。实验表明该方法在SWE-bench-Verified上达到41.6%的解决率，推动修复从试错转向经验驱动。**

- **链接: [http://arxiv.org/pdf/2507.23361v1](http://arxiv.org/pdf/2507.23361v1)**

> **作者:** Silin Chen; Shaoxin Lin; Xiaodong Gu; Yuling Shi; Heng Lian; Longfei Yun; Dong Chen; Weiguo Sun; Lin Cao; Qianxiang Wang
>
> **备注:** Our code and data are available at https://github.com/YerbaPage/SWE-Exp
>
> **摘要:** Recent advances in large language model (LLM) agents have shown remarkable progress in software issue resolution, leveraging advanced techniques such as multi-agent collaboration and Monte Carlo Tree Search (MCTS). However, current agents act as memoryless explorers - treating each problem separately without retaining or reusing knowledge from previous repair experiences. This leads to redundant exploration of failed trajectories and missed chances to adapt successful issue resolution methods to similar problems. To address this problem, we introduce SWE-Exp, an experience - enhanced approach that distills concise and actionable experience from prior agent trajectories, enabling continuous learning across issues. Our method introduces a multi-faceted experience bank that captures both successful and failed repair attempts. Specifically, it extracts reusable issue resolution knowledge at different levels - from high-level problem comprehension to specific code changes. Experiments show that SWE-Exp achieves state-of-the-art resolution rate (41.6% Pass@1) on SWE-bench-Verified under open-source agent frameworks. Our approach establishes a new paradigm in which automated software engineering agents systematically accumulate and leverage repair expertise, fundamentally shifting from trial-and-error exploration to strategic, experience-driven issue resolution.
>
---
## 更新

#### [replaced 001] Who's important? -- SUnSET: Synergistic Understanding of Stakeholder, Events and Time for Timeline Generation
- **分类: cs.SI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.21903v2](http://arxiv.org/pdf/2507.21903v2)**

> **作者:** Tiviatis Sim; Kaiwen Yang; Shen Xin; Kenji Kawaguchi
>
> **摘要:** As news reporting becomes increasingly global and decentralized online, tracking related events across multiple sources presents significant challenges. Existing news summarization methods typically utilizes Large Language Models and Graphical methods on article-based summaries. However, this is not effective since it only considers the textual content of similarly dated articles to understand the gist of the event. To counteract the lack of analysis on the parties involved, it is essential to come up with a novel framework to gauge the importance of stakeholders and the connection of related events through the relevant entities involved. Therefore, we present SUnSET: Synergistic Understanding of Stakeholder, Events and Time for the task of Timeline Summarization (TLS). We leverage powerful Large Language Models (LLMs) to build SET triplets and introduced the use of stakeholder-based ranking to construct a $Relevancy$ metric, which can be extended into general situations. Our experimental results outperform all prior baselines and emerged as the new State-of-the-Art, highlighting the impact of stakeholder information within news article.
>
---
#### [replaced 002] VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22607v2](http://arxiv.org/pdf/2507.22607v2)**

> **作者:** Ruifeng Yuan; Chenghao Xiao; Sicong Leng; Jianyu Wang; Long Li; Weiwen Xu; Hou Pong Chan; Deli Zhao; Tingyang Xu; Zhongyu Wei; Hao Zhang; Yu Rong
>
> **备注:** 21 pages, 5 figures, 6 tables. Work in progress
>
> **摘要:** Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach.
>
---
#### [replaced 003] Cutting Through the Noise: Boosting LLM Performance on Math Word Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.15444v4](http://arxiv.org/pdf/2406.15444v4)**

> **作者:** Ujjwala Anantheswaran; Himanshu Gupta; Kevin Scaria; Shreyas Verma; Chitta Baral; Swaroop Mishra
>
> **备注:** Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs
>
> **摘要:** Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, PROBLEMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and improved ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to 6%.
>
---
#### [replaced 004] Improving Multilingual Capabilities with Cultural and Local Knowledge in Large Language Models While Enhancing Native Performance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09753v3](http://arxiv.org/pdf/2504.09753v3)**

> **作者:** Ram Mohan Rao Kadiyala; Siddartha Pullakhandam; Siddhant Gupta; Drishti Sharma; Jebish Purbey; Kanwal Mehreen; Muhammad Arham; Suman Debnath; Hamza Farooq
>
> **备注:** 24 pages, 18 figures
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities, but their development has primarily focused on English and other high-resource languages, leaving many languages underserved. We present our latest Hindi-English bi-lingual LLM \textbf{Mantra-14B} with ~3\% average improvement in benchmark scores over both languages, outperforming models twice its size. Using a curated dataset composed of English and Hindi instruction data of 485K samples, we instruction tuned models such as Qwen-2.5-14B-Instruct and Phi-4 to improve performance over both English and Hindi. Our experiments encompassing seven different LLMs of varying parameter sizes and over 140 training attempts with varying English-Hindi training data ratios demonstrated that it is possible to significantly improve multilingual performance without compromising native performance. Further, our approach avoids resource-intensive techniques like vocabulary expansion or architectural modifications, thus keeping the model size small. Our results indicate that modest fine-tuning with culturally and locally informed data can bridge performance gaps without incurring significant computational overhead. We release our training code, datasets, and models under mit and apache licenses to aid further research towards under-represented and low-resource languages.
>
---
#### [replaced 005] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework
- **分类: cs.AI; cs.CE; cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.14928v3](http://arxiv.org/pdf/2504.14928v3)**

> **作者:** Yao Shi; Rongkeng Liang; Yong Xu
>
> **备注:** Paper URL: https://aclanthology.org/2025.acl-long.1576 ;Presentation Video: https://www.youtube.com/watch?v=j63ooKE50I0
>
> **摘要:** Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
>
---
#### [replaced 006] Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs and Muslims in Large Language Models: A Systematic Review
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.18199v2](http://arxiv.org/pdf/2506.18199v2)**

> **作者:** Bushra Asseri; Estabrag Abdelaziz; Areej Al-Wabil
>
> **备注:** Research is incomplete
>
> **摘要:** Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility.
>
---
#### [replaced 007] Splits! A Flexible Dataset and Evaluation Framework for Sociocultural Linguistic Investigation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04640v2](http://arxiv.org/pdf/2504.04640v2)**

> **作者:** Eylon Caplan; Tania Chakraborty; Dan Goldwasser
>
> **备注:** Preprint, under review
>
> **摘要:** Variation in language use, shaped by speakers' sociocultural background and specific context of use, offers a rich lens into cultural perspectives, values, and opinions. However, the computational study of these Sociocultural Linguistic Phenomena (SLP) has often been limited to bespoke analyses of specific groups or topics, hindering the pace of scientific discovery. To address this, we introduce Splits!, a 9.7 million-post dataset from Reddit designed for systematic and flexible research. The dataset contains posts from over 53,000 users across 6 demographic groups, organized into 89 discussion topics to enable comparative analysis. We validate Splits! via self-identification and by successfully replicating several known SLPs from existing literature. We complement this dataset with a framework that leverages efficient retrieval methods to rapidly validate potential SLPs (PSLPs) by automatically evaluating whether a given hypothesis is supported by our data. Crucially, to distinguish between novel and obvious insights, the framework incorporates a human-validated measure of a hypothesis's ``unexpectedness.'' We demonstrate that the two-stage process reduces the number of statistically significant findings requiring manual inspection by a factor of 1.5-1.8x, streamlining the discovery of promising phenomena for further investigation.
>
---
#### [replaced 008] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v4](http://arxiv.org/pdf/2505.14874v4)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 009] Can one size fit all?: Measuring Failure in Multi-Document Summarization Domain Transfer
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15768v2](http://arxiv.org/pdf/2503.15768v2)**

> **作者:** Alexandra DeLucia; Mark Dredze
>
> **摘要:** Abstractive multi-document summarization (MDS) is the task of automatically summarizing information in multiple documents, from news articles to conversations with multiple speakers. The training approaches for current MDS models can be grouped into four approaches: end-to-end with special pre-training ("direct"), chunk-then-summarize, extract-then-summarize, and inference with GPT-style models. In this work, we evaluate MDS models across training approaches, domains, and dimensions (reference similarity, quality, and factuality), to analyze how and why models trained on one domain can fail to summarize documents from another (News, Science, and Conversation) in the zero-shot domain transfer setting. We define domain-transfer "failure" as a decrease in factuality, higher deviation from the target, and a general decrease in summary quality. In addition to exploring domain transfer for MDS models, we examine potential issues with applying popular summarization metrics out-of-the-box.
>
---
#### [replaced 010] InfAlign: Inference-aware language model alignment
- **分类: cs.LG; cs.CL; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2412.19792v4](http://arxiv.org/pdf/2412.19792v4)**

> **作者:** Ananth Balashankar; Ziteng Sun; Jonathan Berant; Jacob Eisenstein; Michael Collins; Adrian Hutter; Jong Lee; Chirag Nagpal; Flavien Prost; Aradhana Sinha; Ananda Theertha Suresh; Ahmad Beirami
>
> **摘要:** Language model alignment is a critical step in training modern generative language models. Alignment targets to improve win rate of a sample from the aligned model against the base model. Today, we are increasingly using inference-time algorithms (e.g., Best-of-N, controlled decoding, tree search) to decode from language models rather than standard sampling. We show that this train/test mismatch makes standard RLHF framework sub-optimal in view of such inference-time methods. To this end, we propose a framework for inference-aware alignment (InfAlign), which aims to optimize inference-time win rate of the aligned policy against the base model. We prove that for any inference-time decoding procedure, the optimal aligned policy is the solution to the standard RLHF problem with a transformation of the reward. This motivates us to provide the calibrate-and-transform RL (InfAlign-CTRL) algorithm to solve this problem, which involves a reward calibration step and a KL-regularized reward maximization step with a transformation of the calibrated reward. For best-of-N sampling and best-of-N jailbreaking, we propose specific transformations offering up to 3-8% improvement on inference-time win rates. Finally, we also show that our proposed reward calibration method is a strong baseline for optimizing standard win rate.
>
---
#### [replaced 011] Multi-Hypothesis Distillation of Multilingual Neural Translation Models for Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21568v2](http://arxiv.org/pdf/2507.21568v2)**

> **作者:** Aarón Galiano-Jiménez; Juan Antonio Pérez-Ortiz; Felipe Sánchez-Martínez; Víctor M. Sánchez-Cartagena
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** This paper explores sequence-level knowledge distillation (KD) of multilingual pre-trained encoder-decoder translation models. We argue that the teacher model's output distribution holds valuable insights for the student, beyond the approximated mode obtained through beam search (the standard decoding method), and present Multi-Hypothesis Distillation (MHD), a sequence-level KD method that generates multiple translations for each source sentence. This provides a larger representation of the teacher model distribution and exposes the student model to a wider range of target-side prefixes. We leverage $n$-best lists from beam search to guide the student's learning and examine alternative decoding methods to address issues like low variability and the under-representation of infrequent tokens. For low-resource languages, our research shows that while sampling methods may slightly compromise translation quality compared to beam search based approaches, they enhance the generated corpora with greater variability and lexical richness. This ultimately improves student model performance and mitigates the gender bias amplification often associated with KD.
>
---
#### [replaced 012] LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.15621v2](http://arxiv.org/pdf/2503.15621v2)**

> **作者:** Federico Cocchi; Nicholas Moratelli; Davide Caffagni; Sara Sarto; Lorenzo Baraldi; Marcella Cornia; Rita Cucchiara
>
> **备注:** ICCV 2025 Workshop on What is Next in Multimodal Foundation Models
>
> **摘要:** Recent progress in Multimodal Large Language Models (MLLMs) has highlighted the critical roles of both the visual backbone and the underlying language model. While prior work has primarily focused on scaling these components to billions of parameters, the trade-offs between model size, architecture, and performance remain underexplored. Additionally, inconsistencies in training data and evaluation protocols have hindered direct comparisons, making it difficult to derive optimal design choices. In this paper, we introduce LLaVA-MORE, a new family of MLLMs that integrates recent language models with diverse visual backbones. To ensure fair comparisons, we employ a unified training protocol applied consistently across all architectures. Our analysis systematically explores both small- and medium-scale LLMs -- including Phi-4, LLaMA-3.1, and Gemma-2 -- to evaluate multimodal reasoning, generation, and instruction following, while examining the relationship between model size and performance. Beyond evaluating the LLM impact on final results, we conduct a comprehensive study of various visual encoders, ranging from CLIP-based architectures to alternatives such as DINOv2, SigLIP, and SigLIP2. Additional experiments investigate the effects of increased image resolution and variations in pre-training datasets. Overall, our results provide insights into the design of more effective MLLMs, offering a reproducible evaluation framework that facilitates direct comparisons and can guide future model development. Our source code and trained models are publicly available at: https://github.com/aimagelab/LLaVA-MORE.
>
---
#### [replaced 013] AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23628v2](http://arxiv.org/pdf/2505.23628v2)**

> **作者:** Jiaxin Bai; Wei Fan; Qi Hu; Qing Zong; Chunyang Li; Hong Ting Tsang; Hongyu Luo; Yauwai Yim; Haoyu Huang; Xiao Zhou; Feng Qin; Tianshi Zheng; Xi Peng; Xin Yao; Huiwen Yang; Leijie Wu; Yi Ji; Gong Zhang; Renhai Chen; Yangqiu Song
>
> **备注:** 9 pages, preprint, code: https://github.com/HKUST-KnowComp/AutoSchemaKG
>
> **摘要:** We present AutoSchemaKG, a framework for fully autonomous knowledge graph construction that eliminates the need for predefined schemas. Our system leverages large language models to simultaneously extract knowledge triples and induce comprehensive schemas directly from text, modeling both entities and events while employing conceptualization to organize instances into semantic categories. Processing over 50 million documents, we construct ATLAS (Automated Triple Linking And Schema induction), a family of knowledge graphs with 900+ million nodes and 5.9 billion edges. This approach outperforms state-of-the-art baselines on multi-hop QA tasks and enhances LLM factuality. Notably, our schema induction achieves 95\% semantic alignment with human-crafted schemas with zero manual intervention, demonstrating that billion-scale knowledge graphs with dynamically induced schemas can effectively complement parametric knowledge in large language models.
>
---
#### [replaced 014] FinGAIA: A Chinese Benchmark for AI Agents in Real-World Financial Domain
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.17186v2](http://arxiv.org/pdf/2507.17186v2)**

> **作者:** Lingfeng Zeng; Fangqi Lou; Zixuan Wang; Jiajie Xu; Jinyi Niu; Mengping Li; Yifan Dong; Qi Qi; Wei Zhang; Ziwei Yang; Jun Han; Ruilun Feng; Ruiqi Hu; Lejie Zhang; Zhengbo Feng; Yicheng Ren; Xin Guo; Zhaowei Liu; Dongpo Cheng; Weige Cai; Liwen Zhang
>
> **摘要:** The booming development of AI agents presents unprecedented opportunities for automating complex tasks across various domains. However, their multi-step, multi-tool collaboration capabilities in the financial sector remain underexplored. This paper introduces FinGAIA, an end-to-end benchmark designed to evaluate the practical abilities of AI agents in the financial domain. FinGAIA comprises 407 meticulously crafted tasks, spanning seven major financial sub-domains: securities, funds, banking, insurance, futures, trusts, and asset management. These tasks are organized into three hierarchical levels of scenario depth: basic business analysis, asset decision support, and strategic risk management. We evaluated 10 mainstream AI agents in a zero-shot setting. The best-performing agent, ChatGPT, achieved an overall accuracy of 48.9\%, which, while superior to non-professionals, still lags financial experts by over 35 percentage points. Error analysis has revealed five recurring failure patterns: Cross-modal Alignment Deficiency, Financial Terminological Bias, Operational Process Awareness Barrier, among others. These patterns point to crucial directions for future research. Our work provides the first agent benchmark closely related to the financial domain, aiming to objectively assess and promote the development of agents in this crucial field. Partial data is available at https://github.com/SUFE-AIFLM-Lab/FinGAIA.
>
---
#### [replaced 015] ILID: Native Script Language Identification for Indian Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11832v2](http://arxiv.org/pdf/2507.11832v2)**

> **作者:** Yash Ingle; Pruthwik Mishra
>
> **备注:** 10 pages, 1 figure, 6 tables, Paper accepted in RANLP 2025
>
> **摘要:** The language identification task is a crucial fundamental step in NLP. Often it serves as a pre-processing step for widely used NLP applications such as multilingual machine translation, information retrieval, question and answering, and text summarization. The core challenge of language identification lies in distinguishing languages in noisy, short, and code-mixed environments. This becomes even harder in case of diverse Indian languages that exhibit lexical and phonetic similarities, but have distinct differences. Many Indian languages share the same script, making the task even more challenging. Taking all these challenges into account, we develop and release a dataset of 250K sentences consisting of 23 languages including English and all 22 official Indian languages labeled with their language identifiers, where data in most languages are newly created. We also develop and release baseline models using state-of-the-art approaches in machine learning and fine-tuning pre-trained transformer models. Our models outperforms the state-of-the-art pre-trained transformer models for the language identification task. The dataset and the codes are available at https://yashingle-ai.github.io/ILID/ and in Huggingface open source libraries.
>
---
#### [replaced 016] Cultural Palette: Pluralising Culture Alignment via Multi-agent Palette
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11167v3](http://arxiv.org/pdf/2412.11167v3)**

> **作者:** Jiahao Yuan; Zixiang Di; Shangzixin Zhao; Zhiqing Cui; Hanqing Wang; Guisong Yang; Usman Naseem
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Large language models (LLMs) face challenges in aligning with diverse cultural values despite their remarkable performance in generation, which stems from inherent monocultural biases and difficulties in capturing nuanced cultural semantics. Existing methods struggle to adapt to unknown culture after fine-tuning. Inspired by cultural geography across five continents, we propose Cultural Palette, a multi-agent framework that redefines cultural alignment as an adaptive "color-blending" process for country-specific adaptation. Our approach harnesses cultural geography across five continents (Africa, America, Asia, Europe, Oceania) through three key steps: First, we synthesize the Pentachromatic Cultural Palette Dataset using GPT-4o, refining continental-level dialogues with Hofstede's cultural dimensions to establish foundational cultural representations. Second, five continent-level alignment agents form specialized cultural communities that generate region-specific draft responses. Third, a Meta Agent employs Cultural MoErges to dynamically blend these cultural "colors" through attention-gated parameter merging, akin to mixing pigments on a palette, resolving conflicts while preserving cultural nuances to produce the final culturally-aligned response. Extensive experiments across various countries demonstrate that Cultural Palette surpasses existing baselines in cultural alignment.
>
---
#### [replaced 017] Perception-Aware Policy Optimization for Multimodal Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06448v3](http://arxiv.org/pdf/2507.06448v3)**

> **作者:** Zhenhailong Wang; Xuehang Guo; Sofia Stoica; Haiyang Xu; Hongru Wang; Hyeonjeong Ha; Xiusi Chen; Yangyi Chen; Ming Yan; Fei Huang; Heng Ji
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be a highly effective strategy for endowing Large Language Models (LLMs) with robust multi-step reasoning abilities. However, its design and optimizations remain tailored to purely textual domains, resulting in suboptimal performance when applied to multimodal reasoning tasks. In particular, we observe that a major source of error in current multimodal reasoning lies in the perception of visual inputs. To address this bottleneck, we propose PAPO, a novel policy gradient algorithm that encourages the model to learn to perceive while learning to reason. Specifically, we introduce the Implicit Perception Loss in the form of a KL divergence term, which can be seamlessly plugged into mainstream RLVR algorithms such as GRPO and DAPO. Notably, PAPO does not rely on additional data curation, reward models, or stronger teacher models. To further enhance the training stability of PAPO, we introduce the Double Entropy Loss, which effectively regularizes the new KL objective without compromising performance. Despite its simplicity, PAPO yields significant overall improvements of 4.4%-17.5% on diverse multimodal benchmarks. The improvements are more pronounced, approaching 8.0%-19.1%, on tasks with high vision dependency. We also observe a substantial reduction of 30.5% in perception errors, indicating improved perceptual capabilities with PAPO. Overall, our work introduces a deeper integration of perception-aware supervision into core learning objectives and lays the groundwork for a new RL framework that encourages visually grounded reasoning. Code and data will be made publicly available for research purposes. Project page: https://mikewangwzhl.github.io/PAPO.
>
---
#### [replaced 018] LLM-Crowdsourced: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22359v2](http://arxiv.org/pdf/2507.22359v2)**

> **作者:** Qianhong Guo; Wei Xie; Xiaofang Cai; Enze Wang; Shuoyoucheng Ma; Kai Chen; Xiaofeng Wang; Baosheng Wang
>
> **摘要:** Although large language models (LLMs) demonstrate remarkable capabilities across various tasks, evaluating their capabilities remains a challenging task. Existing evaluation methods suffer from issues such as data contamination, black-box operation, and subjective preference. These issues make it difficult to evaluate the LLMs' true capabilities comprehensively. To tackle these challenges, we propose a novel benchmark-free evaluation paradigm, LLM-Crowdsourced. It utilizes LLMs to generate questions, answer independently, and evaluate mutually. This method integrates four key evaluation criteria: dynamic, transparent, objective, and professional, which existing evaluation methods cannot satisfy simultaneously. Experiments on eight mainstream LLMs across mathematics and programming verify the advantages of our method in distinguishing LLM performance. Furthermore, our study reveals several novel findings that are difficult for traditional methods to detect, including but not limited to: (1) Gemini demonstrates the highest original and professional question-design capabilities among others; (2) Some LLMs exhibit ''memorization-based answering'' by misrecognizing questions as familiar ones with a similar structure; (3) LLM evaluation results demonstrate high consistency (robustness).
>
---
#### [replaced 019] Cultural Bias in Large Language Models: Evaluating AI Agents through Moral Questionnaires
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.10073v2](http://arxiv.org/pdf/2507.10073v2)**

> **作者:** Simon Münker
>
> **备注:** 15pages, 1 figure, 2 tables
>
> **摘要:** Are AI systems truly representing human values, or merely averaging across them? Our study suggests a concerning reality: Large Language Models (LLMs) fail to represent diverse cultural moral frameworks despite their linguistic capabilities. We expose significant gaps between AI-generated and human moral intuitions by applying the Moral Foundations Questionnaire across 19 cultural contexts. Comparing multiple state-of-the-art LLMs' origins against human baseline data, we find these models systematically homogenize moral diversity. Surprisingly, increased model size doesn't consistently improve cultural representation fidelity. Our findings challenge the growing use of LLMs as synthetic populations in social science research and highlight a fundamental limitation in current AI alignment approaches. Without data-driven alignment beyond prompting, these systems cannot capture the nuanced, culturally-specific moral intuitions. Our results call for more grounded alignment objectives and evaluation metrics to ensure AI systems represent diverse human values rather than flattening the moral landscape.
>
---
#### [replaced 020] Explaining vague language
- **分类: cs.CL; cs.GT; cs.IT; math.IT; 91A86; I.2.7**

- **链接: [http://arxiv.org/pdf/2404.18154v2](http://arxiv.org/pdf/2404.18154v2)**

> **作者:** Paul Égré; Benjamin Spector
>
> **摘要:** Why is language vague? Vagueness may be explained and rationalized if it can be shown that vague language is more useful to speaker and hearer than precise language. In a well-known paper, Lipman proposes a game-theoretic account of vagueness in terms of mixed strategy that leads to a puzzle: vagueness cannot be strictly better than precision at equilibrium. More recently, \'Egr\'e, Spector, Mortier and Verheyen have put forward a Bayesian account of vagueness establishing that using vague words can be strictly more informative than using precise words. This paper proposes to compare both results and to explain why they are not in contradiction. Lipman's definition of vagueness relies exclusively on a property of signaling strategies, without making any assumptions about the lexicon, whereas \'Egr\'e et al.'s involves a layer of semantic content. We argue that the semantic account of vagueness is needed, and more adequate and explanatory of vagueness.
>
---
#### [replaced 021] WildSpeech-Bench: Benchmarking Audio LLMs in Natural Speech Conversation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21875v2](http://arxiv.org/pdf/2506.21875v2)**

> **作者:** Jian Zhang; Linhao Zhang; Bokai Lei; Chuhan Wu; Wei Jia; Xiao Zhou
>
> **摘要:** Recent multi-modal Large Language Models (LLMs) such as GPT-4o have demonstrated strong capabilities of direct speech interaction. However, the lack of specialized and comprehensive benchmarks for end-to-end speech LLM evaluation hinders optimizing the user experience of Audio LLMs in real-world applications. Existing evaluation methods often adapt text-based benchmarks, overlooking speech's unique characteristics and challenges, including prosody, homophones, stuttering, and differing user expectations. Here, we present a novel approach to thoroughly evaluate LLMs in practical speech conversations. We systematically curate real-world chat data relevant to spoken scenarios, introduce diversity in speaker attributes and acoustic conditions, and augment the dataset with speech-specific phenomena. We further design a query-aware evaluation method to use customized evaluation checklists and prompts to enhance the accuracy of automatic evaluation. We conduct comprehensive testing and detailed analysis of various mainstream speech models, revealing significant differences in model performance across different speech scenarios. The use of query-aware evaluation further enables a finer-grained assessment under various speech-specific scenarios. Our benchmark can provide valuable insights for speech model development and evaluation.
>
---
#### [replaced 022] PurpCode: Reasoning for Safer Code Generation
- **分类: cs.CR; cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2507.19060v2](http://arxiv.org/pdf/2507.19060v2)**

> **作者:** Jiawei Liu; Nirav Diwan; Zhe Wang; Haoyu Zhai; Xiaona Zhou; Kiet A. Nguyen; Tianjiao Yu; Muntasir Wahed; Yinlin Deng; Hadjer Benkraouda; Yuxiang Wei; Lingming Zhang; Ismini Lourentzou; Gang Wang
>
> **摘要:** We introduce PurpCode, the first post-training recipe for training safe code reasoning models towards generating secure code and defending against malicious cyberactivities. PurpCode trains a reasoning model in two stages: (i) Rule Learning, which explicitly teaches the model to reference cybersafety rules to generate vulnerability-free code and to avoid facilitating malicious cyberactivities; and (ii) Reinforcement Learning, which optimizes model safety and preserves model utility through diverse, multi-objective reward mechanisms. To empower the training pipelines with comprehensive cybersafety data, we conduct internal red-teaming to synthesize comprehensive and high-coverage prompts based on real-world tasks for inducing unsafe cyberactivities in the model. Based on PurpCode, we develop a reasoning-based coding model, namely PurpCode-32B, which demonstrates state-of-the-art cybersafety, outperforming various frontier models. Meanwhile, our alignment method decreases the model overrefusal rates in both general and cybersafety-specific scenarios, while preserving model utility in both code generation and common security knowledge.
>
---
#### [replaced 023] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.14534v3](http://arxiv.org/pdf/2507.14534v3)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
#### [replaced 024] RecGPT Technical Report
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22879v2](http://arxiv.org/pdf/2507.22879v2)**

> **作者:** Chao Yi; Dian Chen; Gaoyang Guo; Jiakai Tang; Jian Wu; Jing Yu; Mao Zhang; Sunhao Dai; Wen Chen; Wenjun Yang; Yuning Jiang; Zhujin Gao; Bo Zheng; Chi Li; Dimin Wang; Dixuan Wang; Fan Li; Fan Zhang; Haibin Chen; Haozhuang Liu; Jialin Zhu; Jiamang Wang; Jiawei Wu; Jin Cui; Ju Huang; Kai Zhang; Kan Liu; Lang Tian; Liang Rao; Longbin Li; Lulu Zhao; Na He; Peiyang Wang; Qiqi Huang; Tao Luo; Wenbo Su; Xiaoxiao He; Xin Tong; Xu Chen; Xunke Xi; Yang Li; Yaxuan Wu; Yeqiu Yang; Yi Hu; Yinnan Song; Yuchen Li; Yujie Luo; Yujin Yuan; Yuliang Yan; Zhengyang Wang; Zhibo Xiao; Zhixin Ma; Zile Zhou; Ziqi Zhang
>
> **摘要:** Recommender systems are among the most impactful applications of artificial intelligence, serving as critical infrastructure connecting users, merchants, and platforms. However, most current industrial systems remain heavily reliant on historical co-occurrence patterns and log-fitting objectives, i.e., optimizing for past user interactions without explicitly modeling user intent. This log-fitting approach often leads to overfitting to narrow historical preferences, failing to capture users' evolving and latent interests. As a result, it reinforces filter bubbles and long-tail phenomena, ultimately harming user experience and threatening the sustainability of the whole recommendation ecosystem. To address these challenges, we rethink the overall design paradigm of recommender systems and propose RecGPT, a next-generation framework that places user intent at the center of the recommendation pipeline. By integrating large language models (LLMs) into key stages of user interest mining, item retrieval, and explanation generation, RecGPT transforms log-fitting recommendation into an intent-centric process. To effectively align general-purpose LLMs to the above domain-specific recommendation tasks at scale, RecGPT incorporates a multi-stage training paradigm, which integrates reasoning-enhanced pre-alignment and self-training evolution, guided by a Human-LLM cooperative judge system. Currently, RecGPT has been fully deployed on the Taobao App. Online experiments demonstrate that RecGPT achieves consistent performance gains across stakeholders: users benefit from increased content diversity and satisfaction, merchants and the platform gain greater exposure and conversions. These comprehensive improvement results across all stakeholders validates that LLM-driven, intent-centric design can foster a more sustainable and mutually beneficial recommendation ecosystem.
>
---
#### [replaced 025] AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18666v3](http://arxiv.org/pdf/2503.18666v3)**

> **作者:** Haoyu Wang; Christopher M. Poskitt; Jun Sun
>
> **备注:** Accepted by the 48th IEEE/ACM International Conference on Software Engineering (ICSE 2026)
>
> **摘要:** Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identify 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios.
>
---
#### [replaced 026] Leveraging LLMs to Create Content Corpora for Niche Domains
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; H.3.1; H.3.3**

- **链接: [http://arxiv.org/pdf/2505.02851v2](http://arxiv.org/pdf/2505.02851v2)**

> **作者:** Franklin Zhang; Sonya Zhang; Alon Halevy
>
> **备注:** 9 pages (main content), 5 figures. Supplementary materials can be found at https://github.com/pigfyy/30DayGen-Supplementary-Materials
>
> **摘要:** Constructing specialized content corpora from vast, unstructured web sources for domain-specific applications poses substantial data curation challenges. In this paper, we introduce a streamlined approach for generating high-quality, domain-specific corpora by efficiently acquiring, filtering, structuring, and cleaning web-based data. We showcase how Large Language Models (LLMs) can be leveraged to address complex data curation at scale, and propose a strategical framework incorporating LLM-enhanced techniques for structured content extraction and semantic deduplication. We validate our approach in the behavior education domain through its integration into 30 Day Me, a habit formation application. Our data pipeline, named 30DayGen, enabled the extraction and synthesis of 3,531 unique 30-day challenges from over 15K webpages. A user survey reports a satisfaction score of 4.3 out of 5, with 91% of respondents indicating willingness to use the curated content for their habit-formation goals.
>
---
#### [replaced 027] How Can I Publish My LLM Benchmark Without Giving the True Answers Away?
- **分类: cs.LG; cs.AI; cs.CL; stat.ME**

- **链接: [http://arxiv.org/pdf/2505.18102v2](http://arxiv.org/pdf/2505.18102v2)**

> **作者:** Takashi Ishida; Thanawat Lodkaew; Ikko Yamane
>
> **备注:** Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
>
> **摘要:** Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
>
---
#### [replaced 028] KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.07695v2](http://arxiv.org/pdf/2507.07695v2)**

> **作者:** Hruday Markondapatnaikuni; Basem Suleiman; Abdelkarim Erradi; Shijing Chen
>
> **备注:** 21 pages, 14 figures
>
> **摘要:** Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study.
>
---
#### [replaced 029] Unveiling the Influence of Amplifying Language-Specific Neurons
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.22581v2](http://arxiv.org/pdf/2507.22581v2)**

> **作者:** Inaya Rahmanisa; Lyzander Marciano Andrylie; Mahardika Krisna Ihsani; Alfan Farizki Wicaksono; Haryo Akbarianto Wibowo; Alham Fikri Aji
>
> **备注:** Our code and dataset are made available at https://github.com/tauimbz/lang-task-neuron
>
> **摘要:** Language-specific neurons in LLMs that strongly correlate with individual languages have been shown to influence model behavior by deactivating them. However, their role in amplification remains underexplored. This work investigates the effect of amplifying language-specific neurons through interventions across 18 languages, including low-resource ones, using three models primarily trained in different languages. We compare amplification factors by their effectiveness in steering to the target language using a proposed Language Steering Shift (LSS) evaluation score, then evaluate it on downstream tasks: commonsense reasoning (XCOPA, XWinograd), knowledge (Include), and translation (FLORES). The optimal amplification factors effectively steer output toward nearly all tested languages. Intervention using this factor on downstream tasks improves self-language performance in some cases but generally degrades cross-language results. These findings highlight the effect of language-specific neurons in multilingual behavior, where amplification can be beneficial especially for low-resource languages, but provides limited advantage for cross-lingual transfer.
>
---
#### [replaced 030] LiMe: a Latin Corpus of Late Medieval Criminal Sentences
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.12829v2](http://arxiv.org/pdf/2404.12829v2)**

> **作者:** Alessandra Bassani; Beatrice Del Bo; Alfio Ferrara; Marta Mangini; Sergio Picascia; Ambra Stefanello
>
> **摘要:** The Latin language has received attention from the computational linguistics research community, which has built, over the years, several valuable resources, ranging from detailed annotated corpora to sophisticated tools for linguistic analysis. With the recent advent of large language models, researchers have also started developing models capable of generating vector representations of Latin texts. The performances of such models remain behind the ones for modern languages, given the disparity in available data. In this paper, we present the LiMe dataset, a corpus of 325 documents extracted from a series of medieval manuscripts called Libri sententiarum potestatis Mediolani, and thoroughly annotated by experts, in order to be employed for masked language model, as well as supervised natural language processing tasks.
>
---
#### [replaced 031] DocPolarBERT: A Pre-trained Model for Document Understanding with Relative Polar Coordinate Encoding of Layout Structures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08606v3](http://arxiv.org/pdf/2507.08606v3)**

> **作者:** Benno Uthayasooriyar; Antoine Ly; Franck Vermet; Caio Corro
>
> **摘要:** We introduce DocPolarBERT, a layout-aware BERT model for document understanding that eliminates the need for absolute 2D positional embeddings. We extend self-attention to take into account text block positions in relative polar coordinate system rather than the Cartesian one. Despite being pre-trained on a dataset more than six times smaller than the widely used IIT-CDIP corpus, DocPolarBERT achieves state-of-the-art results. These results demonstrate that a carefully designed attention mechanism can compensate for reduced pre-training data, offering an efficient and effective alternative for document understanding.
>
---
#### [replaced 032] Vision-Language Models Are Not Pragmatically Competent in Referring Expression Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16060v3](http://arxiv.org/pdf/2504.16060v3)**

> **作者:** Ziqiao Ma; Jing Ding; Xuejun Zhang; Dezhi Luo; Jiahe Ding; Sihan Xu; Yuchen Huang; Run Peng; Joyce Chai
>
> **备注:** COLM 2025 & CVinW @ CVPR 2025 (Spotlight). Homepage: https://vlm-reg.github.io/
>
> **摘要:** Referring Expression Generation (REG) is a core task for evaluating the pragmatic competence of vision-language systems, requiring not only accurate semantic grounding but also adherence to principles of cooperative communication (Grice, 1975). However, current evaluations of vision-language models (VLMs) often overlook the pragmatic dimension, reducing REG to a region-based captioning task and neglecting Gricean maxims. In this work, we revisit REG from a pragmatic perspective, introducing a new dataset (RefOI) of 1.5k images annotated with both written and spoken referring expressions. Through a systematic evaluation of state-of-the-art VLMs, we identify three key failures of pragmatic competence: (1) failure to uniquely identify the referent, (2) inclusion of excessive or irrelevant information, and (3) misalignment with human pragmatic preference, such as the underuse of minimal spatial cues. We also show that standard automatic evaluations fail to capture these pragmatic violations, reinforcing superficial cues rather than genuine referential success. Our findings call for a renewed focus on pragmatically informed models and evaluation frameworks that align with real human communication.
>
---
#### [replaced 033] The Pragmatic Mind of Machines: Tracing the Emergence of Pragmatic Competence in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18497v2](http://arxiv.org/pdf/2505.18497v2)**

> **作者:** Kefan Yu; Qingcheng Zeng; Weihao Xuan; Wanxin Li; Jingyi Wu; Rob Voigt
>
> **摘要:** Current large language models (LLMs) have demonstrated emerging capabilities in social intelligence tasks, including implicature resolution and theory-of-mind reasoning, both of which require substantial pragmatic understanding. However, how LLMs acquire this pragmatic competence throughout the training process remains poorly understood. In this work, we introduce ALTPRAG, a dataset grounded in the pragmatic concept of alternatives, to evaluate whether LLMs at different training stages can accurately infer nuanced speaker intentions. Each instance pairs two equally plausible yet pragmatically divergent continuations and requires the model to (i) infer the speaker's intended meaning and (ii) explain when and why a speaker would choose one utterance over its alternative, thus directly probing pragmatic competence through contrastive reasoning. We systematically evaluate 22 LLMs across 3 key training stages: after pre-training, supervised fine-tuning (SFT), and preference optimization, to examine the development of pragmatic competence. Our results show that even base models exhibit notable sensitivity to pragmatic cues, which improves consistently with increases in model and data scale. Additionally, SFT and RLHF contribute further gains, particularly in cognitive-pragmatic scenarios. These findings highlight pragmatic competence as an emergent and compositional property of LLM training and offer new insights for aligning models with human communicative norms.
>
---
#### [replaced 034] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07106v2](http://arxiv.org/pdf/2506.07106v2)**

> **作者:** Samir Abdaljalil; Hasan Kurban; Khalid Qaraqe; Erchin Serpedin
>
> **备注:** ACL 2025 KnowFM
>
> **摘要:** Large language models (LLMs) have shown strong performance across natural language reasoning tasks, yet their reasoning processes remain brittle and difficult to interpret. Prompting techniques like Chain-of-Thought (CoT) enhance reliability by eliciting intermediate reasoning steps or aggregating multiple outputs. However, they lack mechanisms for enforcing logical structure and assessing internal coherence. We introduce Theorem-of-Thought (ToTh), a novel framework that models reasoning as collaboration among three parallel agents, each simulating a distinct mode of inference: abductive, deductive, and inductive. Each agent produces a reasoning trace, which is structured into a formal reasoning graph. To evaluate consistency, we apply Bayesian belief propagation guided by natural language inference (NLI), assigning confidence scores to each step. The most coherent graph is selected to derive the final answer. Experiments on symbolic (WebOfLies) and numerical (MultiArith) reasoning benchmarks show that ToTh consistently outperforms CoT, Self-Consistency, and CoT-Decoding across multiple LLMs, while producing interpretable and logically grounded reasoning chains. Our findings suggest a promising direction for building more robust and cognitively inspired LLM reasoning. The implementation is available at https://github.com/KurbanIntelligenceLab/theorem-of-thought.
>
---
#### [replaced 035] EgoOops: A Dataset for Mistake Action Detection from Egocentric Videos referring to Procedural Texts
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.05343v3](http://arxiv.org/pdf/2410.05343v3)**

> **作者:** Yuto Haneji; Taichi Nishimura; Hirotaka Kameko; Keisuke Shirai; Tomoya Yoshida; Keiya Kajimura; Koki Yamamoto; Taiyu Cui; Tomohiro Nishimoto; Shinsuke Mori
>
> **备注:** Main 8 pages, supplementary 6 pages
>
> **摘要:** Mistake action detection is crucial for developing intelligent archives that detect workers' errors and provide feedback. Existing studies have focused on visually apparent mistakes in free-style activities, resulting in video-only approaches to mistake detection. However, in text-following activities, models cannot determine the correctness of some actions without referring to the texts. Additionally, current mistake datasets rarely use procedural texts for video recording except for cooking. To fill these gaps, this paper proposes the EgoOops dataset, where egocentric videos record erroneous activities when following procedural texts across diverse domains. It features three types of annotations: video-text alignment, mistake labels, and descriptions for mistakes. We also propose a mistake detection approach, combining video-text alignment and mistake label classification to leverage the texts. Our experimental results show that incorporating procedural texts is essential for mistake detection. Data is available through https://y-haneji.github.io/EgoOops-project-page/.
>
---
#### [replaced 036] Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18337v4](http://arxiv.org/pdf/2411.18337v4)**

> **作者:** T. G. D. K. Sumanathilaka; Nicholas Micallef; Julian Hough
>
> **备注:** 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
>
> **摘要:** Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
>
---
#### [replaced 037] Framing Political Bias in Multilingual LLMs Across Pakistani Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00068v2](http://arxiv.org/pdf/2506.00068v2)**

> **作者:** Afrozah Nadeem; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) increasingly shape public discourse, yet most evaluations of political and economic bias have focused on high-resource, Western languages and contexts. This leaves critical blind spots in low-resource, multilingual regions such as Pakistan, where linguistic identity is closely tied to political, religious, and regional ideologies. We present a systematic evaluation of political bias in 13 state-of-the-art LLMs across five Pakistani languages: Urdu, Punjabi, Sindhi, Pashto, and Balochi. Our framework integrates a culturally adapted Political Compass Test (PCT) with multi-level framing analysis, capturing both ideological stance (economic/social axes) and stylistic framing (content, tone, emphasis). Prompts are aligned with 11 socio-political themes specific to the Pakistani context. Results show that while LLMs predominantly reflect liberal-left orientations consistent with Western training data, they exhibit more authoritarian framing in regional languages, highlighting language-conditioned ideological modulation. We also identify consistent model-specific bias patterns across languages. These findings show the need for culturally grounded, multilingual bias auditing frameworks in global NLP.
>
---
#### [replaced 038] Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length
- **分类: cs.CL; cs.AI; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2506.08184v3](http://arxiv.org/pdf/2506.08184v3)**

> **作者:** Chupei Wang; Jiaqiu Vince Sun
>
> **备注:** Accepted at ICML 2025 Workshop on Long Context Foundation Models (ICFM). Code: https://github.com/zhuangziGiantfish/Unable-to-Forget
>
> **摘要:** Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval.
>
---
#### [replaced 039] Iterative Repair with Weak Verifiers for Few-shot Transfer in KBQA with Unanswerability
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.14313v3](http://arxiv.org/pdf/2406.14313v3)**

> **作者:** Riya Sawhney; Samrat Yadav; Indrajit Bhattacharya; Mausam
>
> **摘要:** Real-world applications of KBQA require models to handle unanswerable questions with a limited volume of in-domain labeled training data. We propose the novel task of few-shot transfer for KBQA with unanswerable questions and contribute two new datasets for performance evaluation. We present FUn-FuSIC - a novel solution for our task that extends FuSIC KBQA, the state-of-the-art few-shot transfer model for answerable-only KBQA. We first note that FuSIC-KBQA's iterative repair makes a strong assumption that all questions are unanswerable. As a remedy, we propose Feedback for Unanswerability (FUn), which uses iterative repair using feedback from a suite of strong and weak verifiers, and an adaptation of self consistency for unanswerabilty to better assess the answerability of a question. Our experiments show that FUn-FuSIC significantly outperforms suitable adaptations of multiple LLM based and supervised SoTA models on our task, while establishing a new SoTA for answerable few-shot transfer as well.
>
---
#### [replaced 040] AI-Reporter: A Path to a New Genre of Scientific Communication
- **分类: cs.DL; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05903v2](http://arxiv.org/pdf/2507.05903v2)**

> **作者:** Gerd Graßhoff
>
> **摘要:** The AI-Reporter represents a paradigmatic shift in scientific publication practice. This document demonstrates through a concrete case study how our system transforms academic presentations into publication-ready chapters -- in less than three minutes. Using Arno Simons' lecture on Large Language Models from the ``Large Language Models for the History, Philosophy, and Sociology of Science'' workshop (NEPI) as an example, we show how technological innovation bridges the gap between ephemeral presentation and permanent scientific documentation.
>
---
#### [replaced 041] Robust and Fine-Grained Detection of AI Generated Texts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.11952v3](http://arxiv.org/pdf/2504.11952v3)**

> **作者:** Ram Mohan Rao Kadiyala; Siddartha Pullakhandam; Kanwal Mehreen; Drishti Sharma; Siddhant Gupta; Jebish Purbey; Ashay Srivastava; Subhasya TippaReddy; Arvind Reddy Bobbili; Suraj Telugara Chandrashekhar; Modabbir Adeeb; Srinadh Vura; Suman Debnath; Hamza Farooq
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** An ideal detection system for machine generated content is supposed to work well on any generator as many more advanced LLMs come into existence day by day. Existing systems often struggle with accurately identifying AI-generated content over shorter texts. Further, not all texts might be entirely authored by a human or LLM, hence we focused more over partial cases i.e human-LLM co-authored texts. Our paper introduces a set of models built for the task of token classification which are trained on an extensive collection of human-machine co-authored texts, which performed well over texts of unseen domains, unseen generators, texts by non-native speakers and those with adversarial inputs. We also introduce a new dataset of over 2.4M such texts mostly co-authored by several popular proprietary LLMs over 23 languages. We also present findings of our models' performance over each texts of each domain and generator. Additional findings include comparison of performance against each adversarial method, length of input texts and characteristics of generated texts compared to the original human authored texts.
>
---
#### [replaced 042] Inside-Out: Hidden Factual Knowledge in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.15299v3](http://arxiv.org/pdf/2503.15299v3)**

> **作者:** Zorik Gekhman; Eyal Ben David; Hadas Orgad; Eran Ofek; Yonatan Belinkov; Idan Szpektor; Jonathan Herzig; Roi Reichart
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
>
---
#### [replaced 043] Neutral Residues: Revisiting Adapters for Model Extension
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02744v3](http://arxiv.org/pdf/2410.02744v3)**

> **作者:** Franck Signe Talla; Edouard Grave; Hervé Jégou
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** We address the problem of extending a pretrained large language model to a new domain that was not seen during training. Standard techniques, such as finetuning or low-rank adaptation (LoRA) are successful at domain adaptation, but do not formally add capacity to the model. This often leads to a trade-off, between performing well on the new domain vs. degrading performance on the original domain. Here, we revisit and improve adapters to extend LLMs from three angles: data, architecture and training procedure, which are advantageously considered jointly. The resulting method, called neutral residues, modifies adapters in a way that leads each new residual block to output near-zeros on the original domain. This solution leads to strong results when adapting a state-of-the-art model originally trained on English to a new language. Neutral residues significantly outperform competing approaches such as finetuning, LoRA or vanilla adapters in terms of the trade-off between learning the new language and not forgetting English.
>
---
#### [replaced 044] Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics
- **分类: cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2506.12365v2](http://arxiv.org/pdf/2506.12365v2)**

> **作者:** Asifullah Khan; Muhammad Zaeem Khan; Saleha Jamshed; Sadia Ahmad; Aleesha Zainab; Kaynat Khatib; Faria Bibi; Abdul Rehman
>
> **摘要:** This survey paper outlines the key developments in the field of Large Language Models (LLMs), including enhancements to their reasoning skills, adaptability to various tasks, increased computational efficiency, and the ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. A significant focus is placed on efficiency, detailing scaling strategies, optimization techniques, and the influential Mixture-of-Experts (MoE) architecture, which strategically routes inputs to specialized subnetworks to boost predictive accuracy, while optimizing resource allocation. This survey also offers a broader perspective on recent advancements in LLMs, going beyond isolated aspects such as model architecture or ethical concerns. Additionally, it explores the role of LLMs in Agentic AI and their use as Autonomous Decision-Making Systems, and categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. The survey also identifies underexplored areas such as interpretability, cross-modal integration, and sustainability. While significant advancements have been made in LLMs, challenges such as high computational costs, biases, and ethical risks remain. Overcoming these requires a focus on bias mitigation, transparent decision-making, and explicit ethical guidelines. Future research will generally focus on enhancing the model's ability to handle multiple inputs, thereby making it more intelligent, safe, and reliable.
>
---
#### [replaced 045] Meta CLIP 2: A Worldwide Scaling Recipe
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22062v2](http://arxiv.org/pdf/2507.22062v2)**

> **作者:** Yung-Sung Chuang; Yang Li; Dong Wang; Ching-Feng Yeh; Kehan Lyu; Ramya Raghavendra; James Glass; Lifei Huang; Jason Weston; Luke Zettlemoyer; Xinlei Chen; Zhuang Liu; Saining Xie; Wen-tau Yih; Shang-Wen Li; Hu Xu
>
> **备注:** 10 pages
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present Meta CLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, Meta CLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval.
>
---
#### [replaced 046] How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2401.13481v3](http://arxiv.org/pdf/2401.13481v3)**

> **作者:** Joshua Ashkinaze; Julia Mendelsohn; Li Qiwei; Ceren Budak; Eric Gilbert
>
> **备注:** Accepted at ACM Collective Intelligence 2025. Originally posted 2024
>
> **摘要:** Exposure to large language model output is rapidly increasing. How will seeing AI-generated ideas affect human ideas? We conducted an experiment (800+ participants, 40+ countries) where participants viewed creative ideas that were from ChatGPT or prior experimental participants and then brainstormed their own idea. We varied the number of AI-generated examples (none, low, or high exposure) and if the examples were labeled as 'AI' (disclosure). Our dynamic experiment design -- ideas from prior participants in an experimental condition are used as stimuli for future participants in the same experimental condition -- speaks to the interdependent process of cultural creation: creative ideas are built upon prior ideas. Hence, we capture the compounding effects of having LLMs 'in the culture loop'. We find that high AI exposure (but not low AI exposure) did not affect the creativity of individual ideas but did increase the average amount and rate of change of collective idea diversity. AI made ideas different, not better. There were no main effects of disclosure. We also found that self-reported creative people were less influenced by knowing an idea was from AI and that participants may knowingly adopt AI ideas when the task is difficult. Our findings suggest that introducing AI ideas may increase collective diversity but not individual creativity.
>
---
#### [replaced 047] RAVine: Reality-Aligned Evaluation for Agentic Search
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.16725v2](http://arxiv.org/pdf/2507.16725v2)**

> **作者:** Yilong Xu; Xiang Long; Zhi Zheng; Jinhua Gao
>
> **摘要:** Agentic search, as a more autonomous and adaptive paradigm of retrieval augmentation, is driving the evolution of intelligent search systems. However, existing evaluation frameworks fail to align well with the goals of agentic search. First, the complex queries commonly used in current benchmarks often deviate from realistic user search scenarios. Second, prior approaches tend to introduce noise when extracting ground truth for end-to-end evaluations, leading to distorted assessments at a fine-grained level. Third, most current frameworks focus solely on the quality of final answers, neglecting the evaluation of the iterative process inherent to agentic search. To address these limitations, we propose RAVine -- a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine targets multi-point queries and long-form answers that better reflect user intents, and introduces an attributable ground truth construction strategy to enhance the accuracy of fine-grained evaluation. Moreover, RAVine examines model's interaction with search tools throughout the iterative process, and accounts for factors of efficiency. We benchmark a series of models using RAVine and derive several insights, which we hope will contribute to advancing the development of agentic search systems. The code and datasets are available at https://github.com/SwordFaith/RAVine.
>
---
