# 自然语言处理 cs.CL

- **最新发布 104 篇**

- **更新 44 篇**

## 最新发布

#### [new 001] Making Bielik LLM Reason (Better): A Field Report
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在提升Bielik LLM的推理能力。通过基准测试和方法论构建，分析其性能并规划未来发展。**

- **链接: [https://arxiv.org/pdf/2603.10640](https://arxiv.org/pdf/2603.10640)**

> **作者:** Adam Trybus; Bartosz Bartnicki; Remigiusz Kinas
>
> **摘要:** This paper presents a research program dedicated to evaluating and advancing the reasoning capabilities of Bielik, a Polish large language model. The study describes a number of stages of work: initial benchmarking and creation of evaluation methodology, analyzing of comparative results with other LLMs and outlining of future prospects that take into account the limitations of the analyses conducted so far and aims to keep Bielik in the race give the ever-changing -- and competitive -- AI landscape.
>
---
#### [new 002] Automated evaluation of LLMs for effective machine translation of Mandarin Chinese to English
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译评估任务，旨在解决LLMs在中译英中的质量评估问题。通过自动化框架比较不同模型的翻译效果，发现其在新闻类文本表现较好，但在文学文本上仍有挑战。**

- **链接: [https://arxiv.org/pdf/2603.09998](https://arxiv.org/pdf/2603.09998)**

> **作者:** Yue Zhang; Rodney Beard; John Hawkins; Rohitash Chandra
>
> **摘要:** Although Large Language Models (LLMs) have exceptional performance in machine translation, only a limited systematic assessment of translation quality has been done. The challenge lies in automated frameworks, as human-expert-based evaluations can be time-consuming, given the fast-evolving LLMs and the need for a diverse set of texts to ensure fair assessments of translation quality. In this paper, we utilise an automated machine learning framework featuring semantic and sentiment analysis to assess Mandarin Chinese to English translation using Google Translate and LLMs, including GPT-4, GPT-4o, and DeepSeek. We compare original and translated texts in various classes of high-profile Chinese texts, which include novel texts that span modern and classical literature, as well as news articles. As the main evaluation measures, we utilise novel similarity metrics to compare the quality of translations produced by LLMs and further evaluate them by an expert human translator. Our results indicate that the LLMs perform well in news media translation, but show divergence in their performance when applied to literary texts. Although GPT-4o and DeepSeek demonstrated better semantic conservation in complex situations, DeepSeek demonstrated better performance in preserving cultural subtleties and grammatical rendering. Nevertheless, the subtle challenges in translation remain: maintaining cultural details, classical references and figurative expressions remain an open problem for all the models.
>
---
#### [new 003] An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took "Use of Practical AI in Digital Libraries" seriously?
- **分类: cs.CL; cs.AI; cs.DL; cs.IR**

- **简介: 该论文属于多标签文本分类任务，旨在解决跨语言目录索引难题。研究发布了一个双语语料库和GND分类体系，支持基于本体的多标签分类与自动化编目评估。**

- **链接: [https://arxiv.org/pdf/2603.10876](https://arxiv.org/pdf/2603.10876)**

> **作者:** Jennifer D'Souza; Sameer Sadruddin; Maximilian Kähler; Andrea Salfinger; Luca Zaccagna; Francesca Incitti; Lauro Snidaro; Osma Suominen
>
> **备注:** 9 pages, 5 figures. Accepted to appear in the Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** Subject indexing is vital for discovery but hard to sustain at scale and across languages. We release a large bilingual (English/German) corpus of catalog records annotated with the Integrated Authority File (GND), plus a machine-actionable GND taxonomy. The resource enables ontology-aware multi-label classification, mapping text to authority terms, and agent-assisted cataloging with reproducible, authority-grounded evaluation. We provide a brief statistical profile and qualitative error analyses of three systems. We invite the community to assess not only accuracy but usefulness and transparency, toward authority-anchored AI co-pilots that amplify catalogers' work.
>
---
#### [new 004] Instruction set for the representation of graphs
- **分类: cs.CL; cs.AI; cs.DS**

- **简介: 该论文提出IsalGraph，一种将图结构编码为紧凑字符串的方法，解决图的序列化与相似性计算问题，适用于图搜索和生成任务。**

- **链接: [https://arxiv.org/pdf/2603.11039](https://arxiv.org/pdf/2603.11039)**

> **作者:** Ezequiel Lopez-Rubio; Mario Pascual-Gonzalez
>
> **摘要:** We present IsalGraph, a method for representing the structure of any finite, simple graph as a compact string over a nine-character instruction alphabet. The encoding is executed by a small virtual machine comprising a sparse graph, a circular doubly-linked list (CDLL) of graph-node references, and two traversal pointers. Instructions either move a pointer through the CDLL or insert a node or edge into the graph. A key design property is that every string over the alphabet decodes to a valid graph, with no invalid states reachable. A greedy \emph{GraphToString} algorithm encodes any connected graph into a string in time polynomial in the number of nodes; an exhaustive-backtracking variant produces a canonical string by selecting the lexicographically smallest shortest string across all starting nodes and all valid traversal orders. We evaluate the representation on five real-world graph benchmark datasets (IAM Letter LOW/MED/HIGH, LINUX, and AIDS) and show that the Levenshtein distance between IsalGraph strings correlates strongly with graph edit distance (GED). Together, these properties make IsalGraph strings a compact, isomorphism-invariant, and language-model-compatible sequential encoding of graph structure, with direct applications in graph similarity search, graph generation, and graph-conditioned language modelling
>
---
#### [new 005] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment
- **分类: cs.CL**

- **简介: 该论文属于认知障碍老年人群体对话系统任务，旨在解决传统CST难以扩展及数字系统在群体对话中的不足。提出GCSD系统，集成多模块以提升认知刺激效果。**

- **链接: [https://arxiv.org/pdf/2603.10034](https://arxiv.org/pdf/2603.10034)**

> **作者:** Jiyue Jiang; Yanyu Chen; Pengan Chen; Kai Liu; Jingqi Zhou; Zheyong Zhu; He Hu; Fei Ma; Qi Tian; Chuan Wu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Cognitive impairment is becoming a major public health challenge. Cognitive Stimulation Therapy (CST) is an effective intervention for cognitive impairment, but traditional methods are difficult to scale, and existing digital systems struggle with group dialogues and cognitive stimulation principles. While Large Language Models (LLMs) are powerful, their application in this context faces key challenges: cognitive stimulation dialogue paradigms, a lack of therapeutic reasoning, and static-only user modeling. To address these issues, we propose a principle-driven adaptive policy actualized through a Group Cognitive Stimulation Dialogue (GCSD) system. We first construct a dataset with over 500 hours of real-world CST conversations and 10,000+ simulated dialogues generated via our Principle-Guided Scenario Simulation strategy. Our GCSD system then integrates four core modules to overcome LLM limitations: (i) a multi-speaker context controller to resolve role confusion; (ii) dynamic participant cognitive state modeling for personalized interaction; (iii) a cognitive stimulation-focused attention loss to instill cognitive stimulation reasoning; and (iv) a multi-dimensional reward strategy to enhance response value. Experimental results demonstrate that GCSD significantly outperforms baseline models across various evaluation metrics. Future work will focus on long-term clinical validation to bridge the gap between computational performance and clinical efficacy.
>
---
#### [new 006] PEEM: Prompt Engineering Evaluation Metrics for Interpretable Joint Evaluation of Prompts and Responses
- **分类: cs.CL**

- **简介: 该论文提出PEEM，用于评估提示与响应的联合可解释性，解决LLM提示设计的评价问题。通过结构化指标和自然语言分析，提升提示优化效果。**

- **链接: [https://arxiv.org/pdf/2603.10477](https://arxiv.org/pdf/2603.10477)**

> **作者:** Minki Hong; Eunsoo Lee; Sohyun Park; Jihie Kim
>
> **备注:** 24pages, 2 figures
>
> **摘要:** Prompt design is a primary control interface for large language models (LLMs), yet standard evaluations largely reduce performance to answer correctness, obscuring why a prompt succeeds or fails and providing little actionable guidance. We propose PEEM (Prompt Engineering Evaluation Metrics), a unified framework for joint and interpretable evaluation of both prompts and responses. PEEM defines a structured rubric with 9 axes: 3 prompt criteria (clarity/structure, linguistic quality, fairness) and 6 response criteria (accuracy, coherence, relevance, objectivity, clarity, conciseness), and uses an LLM-based evaluator to output (i) scalar scores on a 1-5 Likert scale and (ii) criterion-specific natural-language rationales grounded in the rubric. Across 7 benchmarks and 5 task models, PEEM's accuracy axis strongly aligns with conventional accuracy while preserving model rankings (aggregate Spearman rho about 0.97, Pearson r about 0.94, p < 0.001). A multi-evaluator study with four models shows consistent relative judgments (pairwise rho = 0.68-0.85), supporting evaluator-agnostic deployment. Beyond alignment, PEEM captures complementary linguistic failure modes and remains informative under prompt perturbations: prompt-quality trends track downstream accuracy under iterative rewrites, semantic adversarial manipulations induce clear score degradation, and meaning-preserving paraphrases yield high stability (robustness rate about 76.7-80.6%). Finally, using only PEEM scores and rationales as feedback, a zero-shot prompt rewriting loop improves downstream accuracy by up to 11.7 points, outperforming supervised and RL-based prompt-optimization baselines. Overall, PEEM provides a reproducible, criterion-driven protocol that links prompt formulation to response behavior and enables systematic diagnosis and optimization of LLM interactions.
>
---
#### [new 007] FERRET: Framework for Expansion Reliant Red Teaming
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FERRET框架，用于生成多模态对抗性对话，解决自动化红队测试问题。通过水平、垂直和元扩展提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2603.10010](https://arxiv.org/pdf/2603.10010)**

> **作者:** Ninareh Mehrabi; Vitor Albiero; Maya Pavlova; Joanna Bitton
>
> **摘要:** We introduce a multi-faceted automated red teaming framework in which the goal is to generate multi-modal adversarial conversations that would break a target model and introduce various expansions that would result in more effective and efficient adversarial conversations. The introduced expansions include: 1. Horizontal expansion in which the goal is for the red team model to self-improve and generate more effective conversation starters that would shape a conversation. 2. Vertical expansion in which the goal is to take these conversation starters that are discovered in the horizontal expansion phase and expand them into effective multi-modal conversations and 3. Meta expansion in which the goal is for the red team model to discover more effective multi-modal attack strategies during the course of a conversation. We call our framework FERRET (Framework for Expansion Reliant Red Teaming) and compare it with various existing automated red teaming approaches. In our experiments, we demonstrate the effectiveness of FERRET in generating effective multi-modal adversarial conversations and its superior performance against existing state of the art approaches.
>
---
#### [new 008] SpreadsheetArena: Decomposing Preference in LLM Generation of Spreadsheet Workbooks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于大语言模型生成电子表格的任务，旨在评估模型在结构化输出上的表现，解决如何有效衡量生成质量的问题。**

- **链接: [https://arxiv.org/pdf/2603.10002](https://arxiv.org/pdf/2603.10002)**

> **作者:** Srivatsa Kundurthy; Clara Na; Michael Handley; Zach Kirshner; Chen Bo Calvin Zhang; Manasi Sharma; Emma Strubell; John Ling
>
> **备注:** 30 pages
>
> **摘要:** Large language models (LLMs) are increasingly tasked with producing and manipulating structured artifacts. We consider the task of end-to-end spreadsheet generation, where language models are prompted to produce spreadsheet artifacts to satisfy users' explicit and implicit constraints, specified in natural language. We introduce SpreadsheetArena, a platform for evaluating models' performance on the task via blind pairwise evaluations of LLM-generated spreadsheet workbooks. As with other complex, open-ended tasks, relevant evaluation criteria can vary substantially across use cases and prompts, often in ways that are difficult to formalize. Compared to general chat or text generation settings, spreadsheet generation presents unique challenges and opportunities: the task output structure is well-defined and multi-dimensional, and there are often complex considerations around interactivity and layout. Among other findings, we observe that stylistic, structural, and functional features of preferred spreadsheets vary substantially across use cases, and expert evaluations of spreadsheets for finance prompts suggests that even highly ranked arena models do not reliably produce spreadsheets aligned with domain-specific best practices. Our hope is that our work prompts further study of end-to-end spreadsheet generation as a challenging and interesting category of complex, open-ended tasks for LLMs. Our live arena is hosted at this https URL.
>
---
#### [new 009] Automatic End-to-End Data Integration using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于数据集成任务，旨在减少人工参与。通过使用大语言模型自动生成数据集成管道所需的所有组件，降低人力成本并提升效率。**

- **链接: [https://arxiv.org/pdf/2603.10547](https://arxiv.org/pdf/2603.10547)**

> **作者:** Aaron Steiner; Christian Bizer
>
> **备注:** 8 pages, 9 tables. Accepted at the Beyond SQL Workshop at ICDE 2026
>
> **摘要:** Designing data integration pipelines typically requires substantial manual effort from data engineers to configure pipeline components and label training data. While LLMs have shown promise in handling individual steps of the integration process, their potential to replace all human input across end-to-end data integration pipelines has not been investigated. As a step toward exploring this potential, we present an automatic data integration pipeline that uses GPT-5.2 to generate all artifacts required to adapt the pipeline to specific use cases. These artifacts are schema mappings, value mappings for data normalization, training data for entity matching, and validation data for selecting conflict resolution heuristics in data fusion. We compare the performance of this LLM-based pipeline to the performance of human-designed pipelines along three case studies requiring the integration of video game, music, and company related data. Our experiments show that the LLM-based pipeline is able to produce similar results, for some tasks even better results, as the human-designed pipelines. End-to-end, the human and the LLM pipelines produce integrated datasets of comparable size and density. Having the LLM configure the pipelines costs approximately \$10 per case study, which represents only a small fraction of the cost of having human data engineers perform the same tasks.
>
---
#### [new 010] There Are No Silly Questions: Evaluation of Offline LLM Capabilities from a Turkish Perspective
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于评估任务，旨在解决Offline LLM在土耳其语教育中的安全与可靠性问题，通过设计测试集评估模型表现。**

- **链接: [https://arxiv.org/pdf/2603.09996](https://arxiv.org/pdf/2603.09996)**

> **作者:** Edibe Yilmaz; Kahraman Kostas
>
> **备注:** 5 pages, 6 tables, conference
>
> **摘要:** The integration of large language models (LLMs) into educational processes introduces significant constraints regarding data privacy and reliability, particularly in pedagogically vulnerable contexts such as Turkish heritage language education. This study aims to systematically evaluate the robustness and pedagogical safety of locally deployable offline LLMs within the context of Turkish heritage language education. To this end, a Turkish Anomaly Suite (TAS) consisting of 10 original edge-case scenarios was developed to assess the models' capacities for epistemic resistance, logical consistency, and pedagogical safety. Experiments conducted on 14 different models ranging from 270M to 32B parameters reveal that anomaly resistance is not solely dependent on model scale and that sycophancy bias can pose pedagogical risks even in large-scale models. The findings indicate that reasoning-oriented models in the 8B--14B parameter range represent the most balanced segment in terms of cost-safety trade-off for language learners.
>
---
#### [new 011] S-GRADES -- Studying Generalization of Student Response Assessments in Diverse Evaluative Settings
- **分类: cs.CL**

- **简介: 该论文属于教育NLP任务，解决自动评分标准不统一的问题。提出S-GRADES基准，整合14个数据集，促进跨范式评估与模型泛化研究。**

- **链接: [https://arxiv.org/pdf/2603.10233](https://arxiv.org/pdf/2603.10233)**

> **作者:** Tasfia Seuti; Sagnik Ray Choudhury
>
> **备注:** LREC 2026 Accepted, this https URL
>
> **摘要:** Evaluating student responses, from long essays to short factual answers, is a key challenge in educational NLP. Automated Essay Scoring (AES) focuses on holistic writing qualities such as coherence and argumentation, while Automatic Short Answer Grading (ASAG) emphasizes factual correctness and conceptual understanding. Despite their shared goal, these paradigms have progressed in isolation with fragmented datasets, inconsistent metrics, and separate communities. We introduce S-GRADES (Studying Generalization of Student Response Assessments in Diverse Evaluative Settings), a web-based benchmark that consolidates 14 diverse grading datasets under a unified interface with standardized access and reproducible evaluation protocols. The benchmark is fully open-source and designed for extensibility, enabling continuous integration of new datasets and evaluation settings. To demonstrate the utility of S-GRADES, we evaluate three state-of-the-art large language models across the benchmark using multiple reasoning strategies in prompting. We further examine the effects of exemplar selection and cross-dataset exemplar transfer. Our analyses illustrate how benchmark-driven evaluation reveals reliability and generalization gaps across essay and short-answer grading tasks, highlighting the importance of standardized, cross-paradigm assessment.
>
---
#### [new 012] Beyond the Prompt in Large Language Models: Comprehension, In-Context Learning, and Chain-of-Thought
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，探讨大语言模型的语义理解、上下文学习和思维链推理机制，解决其理论原理不明确的问题，通过分析提示工程的有效性提供理论支持。**

- **链接: [https://arxiv.org/pdf/2603.10000](https://arxiv.org/pdf/2603.10000)**

> **作者:** Yuling Jiao; Yanming Lai; Huazhen Lin; Wensen Ma; Houduo Qi; Defeng Sun
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency across diverse tasks, exhibiting emergent properties such as semantic prompt comprehension, In-Context Learning (ICL), and Chain-of-Thought (CoT) reasoning. Despite their empirical success, the theoretical mechanisms driving these phenomena remain poorly understood. This study dives into the foundations of these observations by addressing three critical questions: (1) How do LLMs accurately decode prompt semantics despite being trained solely on a next-token prediction objective? (2) Through what mechanism does ICL facilitate performance gains without explicit parameter updates? and (3) Why do intermediate reasoning steps in CoT prompting effectively unlock capabilities for complex, multi-step problems? Our results demonstrate that, through the autoregressive process, LLMs are capable of exactly inferring the transition probabilities between tokens across distinct tasks using provided prompts. We show that ICL enhances performance by reducing prompt ambiguity and facilitating posterior concentration on the intended task. Furthermore, we find that CoT prompting activates the model's capacity for task decomposition, breaking complex problems into a sequence of simpler sub-tasks that the model has mastered during the pretraining phase. By comparing their individual error bounds, we provide novel theoretical insights into the statistical superiority of advanced prompt engineering techniques.
>
---
#### [new 013] The Dunning-Kruger Effect in Large Language Models: An Empirical Study of Confidence Calibration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，研究LLM的自信校准问题。通过实验发现低性能模型表现出过度自信，类似人类的达克效应，旨在提升高风险应用中的安全性。**

- **链接: [https://arxiv.org/pdf/2603.09985](https://arxiv.org/pdf/2603.09985)**

> **作者:** Sudipta Ghosh; Mrityunjoy Panday
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, yet their ability to accurately assess their own confidence remains poorly understood. We present an empirical study investigating whether LLMs exhibit patterns reminiscent of the Dunning-Kruger effect -- a cognitive bias where individuals with limited competence tend to overestimate their abilities. We evaluate four state-of-the-art models (Claude Haiku 4.5, Gemini 2.5 Pro, Gemini 2.5 Flash, and Kimi K2) across four benchmark datasets totaling 24,000 experimental trials. Our results reveal striking calibration differences: Kimi K2 exhibits severe overconfidence with an Expected Calibration Error (ECE) of 0.726 despite only 23.3% accuracy, while Claude Haiku 4.5 achieves the best calibration (ECE = 0.122) with 75.4% accuracy. These findings demonstrate that poorly performing models display markedly higher overconfidence -- a pattern analogous to the Dunning-Kruger effect in human cognition. We discuss implications for safe deployment of LLMs in high-stakes applications.
>
---
#### [new 014] TriageSim: A Conversational Emergency Triage Simulation Framework from Structured Electronic Health Records
- **分类: cs.CL**

- **简介: 该论文提出TriageSim，用于从结构化电子健康记录生成模拟的紧急分诊对话，解决分诊模拟数据不足的问题。通过合成对话数据提升分诊系统训练效果。**

- **链接: [https://arxiv.org/pdf/2603.10035](https://arxiv.org/pdf/2603.10035)**

> **作者:** Dipankar Srirag; Quoc Dung Nguyen; Aditya Joshi; Padmanesan Narasimhan; Salil Kanhere
>
> **备注:** 6 pages, 3 figures, 2 tables
>
> **摘要:** Research in emergency triage is restricted to structured electronic health records (EHR) due to regulatory constraints on nurse-patient interactions. We introduce TriageSim, a simulation framework for generating persona-conditioned triage conversations from structured records. TriageSim enables multi-turn nurse-patient interactions with explicit control over disfluency and decision behaviour, producing a corpus of ~800 synthetic transcripts and corresponding audio. We use a combination of automated analysis for linguistic, behavioural and acoustic fidelity alongside manual evaluation for medical fidelity using a random subset of 50 conversations. The utility of the generated corpus is examined via conversational triage classification. We observe modest agreement for acuity levels across three modalities: generated synthetic text, ASR transcripts, and direct audio inputs. The code, persona schemata and triage policy prompts for TriageSim will be available upon acceptance.
>
---
#### [new 015] Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在非英语文化中的偏见问题。通过Wikidata构建拉美文化偏见数据集，评估模型在不同拉美国家的表现差异。**

- **链接: [https://arxiv.org/pdf/2603.10001](https://arxiv.org/pdf/2603.10001)**

> **作者:** Yannis Karmim; Renato Pino; Hernan Contreras; Hernan Lira; Sebastian Cifuentes; Simon Escoffier; Luis Martí; Djamé Seddah; Valentin Barrière
>
> **摘要:** Large Language Models (LLMs) exhibit inequalities with respect to various cultural contexts. Most prominent open-weights models are trained on Global North data and show prejudicial behavior towards other cultures. Moreover, there is a notable lack of resources to detect biases in non-English languages, especially from Latin America (Latam), a continent containing various cultures, even though they share a common cultural ground. We propose to leverage the content of Wikipedia, the structure of the Wikidata knowledge graph, and expert knowledge from social science in order to create a dataset of question/answer (Q/As) pairs, based on the different popular and social cultures of various Latin American countries. We create the LatamQA database of over 26k questions and associated answers extracted from 26k Wikipedia articles, and transformed into multiple-choice questions (MCQ) in Spanish and Portuguese, in turn translated to English. We use this MCQ to quantify the degree of knowledge of various LLMs and find out (i) a discrepancy in performances between the Latam countries, ones being easier than others for the majority of the models, (ii) that the models perform better in their original language, and (iii) that Iberian Spanish culture is better known than Latam one.
>
---
#### [new 016] GATech at AbjadMed: Bidirectional Encoders vs. Causal Decoders: Insights from 82-Class Arabic Medical Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于阿拉伯语医学文本分类任务，解决细粒度类别划分问题。通过对比双向编码器与因果解码器，验证编码器在语义捕捉上的优势。**

- **链接: [https://arxiv.org/pdf/2603.10008](https://arxiv.org/pdf/2603.10008)**

> **作者:** Ahmed Khaled Khamis
>
> **备注:** 5 pages, 2 figures, EACL26, AbjadNLP
>
> **摘要:** This paper presents system description for Arabic medical text classification across 82 distinct categories. Our primary architecture utilizes a fine-tuned AraBERTv2 encoder enhanced with a hybrid pooling strategies, combining attention and mean representations, and multi-sample dropout for robust regularization. We systematically benchmark this approach against a suite of multilingual and Arabic-specific encoders, as well as several large-scale causal decoders, including zero-shot re-ranking via Llama 3.3 70B and feature extraction from Qwen 3B hidden states. Our findings demonstrate that specialized bidirectional encoders significantly outperform causal decoders in capturing the precise semantic boundaries required for fine-grained medical text classification. We show that causal decoders, optimized for next-token prediction, produce sequence-biased embeddings that are less effective for categorization compared to the global context captured by bidirectional attention. Despite significant class imbalance and label noise identified within the training data, our results highlight the superior semantic compression of fine-tuned encoders for specialized Arabic NLP tasks. Final performance metrics on the test set, including Accuracy and Macro-F1, are reported and discussed.
>
---
#### [new 017] Fine-Tune, Don't Prompt, Your Language Model to Identify Biased Language in Clinical Notes
- **分类: cs.CL**

- **简介: 该论文属于情感分类任务，旨在检测临床文档中的偏见语言。通过构建词典并进行微调，提升模型识别偏见语言的准确性。**

- **链接: [https://arxiv.org/pdf/2603.10004](https://arxiv.org/pdf/2603.10004)**

> **作者:** Isotta Landi; Eugenia Alleva; Nicole Bussola; Rebecca M. Cohen; Sarah Nowlin; Leslee J. Shaw; Alexander W. Charney; Kimberly B. Glazer
>
> **摘要:** Clinical documentation can contain emotionally charged language with stigmatizing or privileging valences. We present a framework for detecting and classifying such language as stigmatizing, privileging, or neutral. We constructed a curated lexicon of biased terms scored for emotional valence. We then used lexicon-based matching to extract text chunks from OB-GYN delivery notes (Mount Sinai Hospital, NY) and MIMIC-IV discharge summaries across multiple specialties. Three clinicians annotated all chunks, enabling characterization of valence patterns across specialties and healthcare systems. We benchmarked multiple classification strategies (zero-shot prompting, in-context learning, and supervised fine-tuning) across encoder-only models (GatorTron) and generative large language models (Llama). Fine-tuning with lexically primed inputs consistently outperformed prompting approaches. GatorTron achieved an F1 score of 0.96 on the OB-GYN test set, outperforming larger generative models while requiring minimal prompt engineering and fewer computational resources. External validation on MIMIC-IV revealed limited cross-domain generalizability (F1 < 0.70, 44% drop). Training on the broader MIMIC-IV dataset improved generalizability when testing on OB-GYN (F1 = 0.71, 11% drop), but at the cost of reduced precision. Our findings demonstrate that fine-tuning outperforms prompting for emotional valence classification and that models must be adapted to specific medical specialties to achieve clinically appropriate performance. The same terms can carry different emotional valences across specialties: words with clinical meaning in one context may be stigmatizing in another. For bias detection, where misclassification risks undermining clinician trust or perpetuating patient harm, specialty-specific fine-tuning is essential to capture these semantic shifts. * Equal contribution.
>
---
#### [new 018] The System Hallucination Scale (SHS): A Minimal yet Effective Human-Centered Instrument for Evaluating Hallucination-Related Behavior in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SHS，用于评估大语言模型中的幻觉行为，属于模型评估任务。旨在解决如何从用户角度衡量模型生成内容的可靠性与一致性问题。工作包括设计SHS并进行实证验证。**

- **链接: [https://arxiv.org/pdf/2603.09989](https://arxiv.org/pdf/2603.09989)**

> **作者:** Heimo Müller; Dominik Steiger; Markus Plass; Andreas Holzinger
>
> **摘要:** We introduce the System Hallucination Scale (SHS), a lightweight and human-centered measurement instrument for assessing hallucination-related behavior in large language models (LLMs). Inspired by established psychometric tools such as the System Usability Scale (SUS) and the System Causability Scale (SCS), SHS enables rapid, interpretable, and domain-agnostic evaluation of factual unreliability, incoherence, misleading presentation, and responsiveness to user guidance in model-generated text. SHS is explicitly not an automatic hallucination detector or benchmark metric; instead, it captures how hallucination phenomena manifest from a user perspective under realistic interaction conditions. A real-world evaluation with 210 participants demonstrates high clarity, coherent response behavior, and construct validity, supported by statistical analysis including internal consistency (Cronbach's alpha = 0.87$) and significant inter-dimension correlations (p < 0.001$). Comparative analysis with SUS and SCS reveals complementary measurement properties, supporting SHS as a practical tool for comparative analysis, iterative system development, and deployment monitoring.
>
---
#### [new 019] Measuring and Eliminating Refusals in Military Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于军事大模型优化任务，旨在解决模型拒绝合法军事查询的问题。通过构建基准数据集，分析拒绝率并进行模型调整，提升军事任务准确性。**

- **链接: [https://arxiv.org/pdf/2603.10012](https://arxiv.org/pdf/2603.10012)**

> **作者:** Jack FitzGerald; Dylan Bates; Aristotelis Lazaridis; Aman Sharma; Vincent Lu; Brian King; Yousif Azami; Sean Bailey; Jeremy Cao; Peter Damianov; Kevin de Haan; Joseph Madigan; Jeremy McLaurin; Luke Kerbs; Jonathan Tainer; Dave Anderson; Jonathan Beck; Jamie Cuticello; Colton Malkerson; Tyler Saltsman
>
> **备注:** 30 pages
>
> **摘要:** Military Large Language Models (LLMs) must provide accurate information to the warfighter in time-critical and dangerous situations. However, today's LLMs are imbued with safety behaviors that cause the LLM to refuse many legitimate queries in the military domain, particularly those related to violence, terrorism, or military technology. Our gold benchmark for assessing refusal rates, which was developed by veterans of the US Army and special forces, is to our knowledge the first dataset of its kind. We present results for refusal and deflection rates on 31 public models and 3 military models. We observe hard rejection rates as high as 98.2% and soft deflection rates ranging from 0% to 21.3%. We also present results on two additional synthetic datasets and show their correlations with the gold dataset. Finally, we perform abliteration using the Heretic library on a military-tuned gpt-oss-20b model, showing an absolute increase in answer rate of 66.5 points but an average relative decrease of 2% on other military tasks. In our concluding remarks, we argue for deeper specialization, including with mid-training and end-to-end post-training, to achieve zero refusals and maximum military task accuracy for closed military models.
>
---
#### [new 020] Word Recovery in Large Language Models Enables Character-Level Tokenization Robustness
- **分类: cs.CL**

- **简介: 该论文研究大语言模型对字符级分词的鲁棒性问题，提出“词语恢复”机制。通过解码方法和注意力分析，揭示模型如何从字符输入中重建词语信息，提升对非标准分词的处理能力。**

- **链接: [https://arxiv.org/pdf/2603.10771](https://arxiv.org/pdf/2603.10771)**

> **作者:** Zhipeng Yang; Shu Yang; Lijie Hu; Di Wang
>
> **摘要:** Large language models (LLMs) trained with canonical tokenization exhibit surprising robustness to non-canonical inputs such as character-level tokenization, yet the mechanisms underlying this robustness remain unclear. We study this phenomenon through mechanistic interpretability and identify a core process we term word recovery. We first introduce a decoding-based method to detect word recovery, showing that hidden states reconstruct canonical word-level token identities from character-level inputs. We then provide causal evidence by removing the corresponding subspace from hidden states, which consistently degrades downstream task performance. Finally, we conduct a fine-grained attention analysis and show that in-group attention among characters belonging to the same canonical token is critical for word recovery: masking such attention in early layers substantially reduces both recovery scores and task performance. Together, our findings provide a mechanistic explanation for tokenization robustness and identify word recovery as a key mechanism enabling LLMs to process character-level inputs.
>
---
#### [new 021] Prism-$Δ$: Differential Subspace Steering for Prompt Highlighting in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Prompt highlighting问题，通过PRISM-Δ方法区分相关与无关上下文，提升模型生成质量。**

- **链接: [https://arxiv.org/pdf/2603.10705](https://arxiv.org/pdf/2603.10705)**

> **作者:** Yuyao Ge; Shenghua Liu; Yiwei Wang; Tianyu Liu; Baolong Bi; Lingrui Mei; Jiayu Yao; Jiafeng Guo; Xueqi Cheng
>
> **备注:** 21 pages, 14 figures
>
> **摘要:** Prompt highlighting steers a large language model to prioritize user-specified text spans during generation. A key challenge is extracting steering directions that capture the difference between relevant and irrelevant contexts, rather than shared structural patterns common to both. We propose PRISM-$\Delta$ (Projection-based Relevance-Informed Steering Method), which decomposes the difference between positive and negative cross-covariance matrices to maximize discriminative energy while eliminating shared directions. Each attention head receives a continuous softplus importance weight, letting weak-but-useful heads contribute at reduced strength. The framework extends naturally to Value representations, capturing content-channel signal that Key-only methods leave unused. Across four benchmarks and five models, PRISM-$\Delta$ matches or exceeds the best existing method on 19 of 20 configurations, with relative gains up to +10.6%, while halving the fluency cost of steering. PRISM-$\Delta$ also scales to long-context retrieval, outperforming the best existing method by up to +4.8% relative gain. PRISM-$\Delta$ is compatible with FlashAttention and adds negligible memory overhead.
>
---
#### [new 022] Evolving Demonstration Optimization for Chain-of-Thought Feature Transformation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于特征变换任务，旨在解决传统方法在搜索效率和效果上的不足。通过优化LLM驱动的特征变换，构建经验库并引导生成过程，提升性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.09987](https://arxiv.org/pdf/2603.09987)**

> **作者:** Xinyuan Wang; Kunpeng Liu; Arun Vignesh Malarkkan; Yanjie Fu
>
> **摘要:** Feature Transformation (FT) is a core data-centric AI task that improves feature space quality to advance downstream predictive performance. However, discovering effective transformations remains challenging due to the large space of feature-operator combinations. Existing solutions rely on discrete search or latent generation, but they are frequently limited by sample inefficiency, invalid candidates, and redundant generations with limited coverage. Large Language Models (LLMs) offer strong priors for producing valid transformations, but current LLM-based FT methods typically rely on static demonstrations, resulting in limited diversity, redundant outputs, and weak alignment with downstream objectives. We propose a framework that optimizes context data for LLM-driven FT by evolving trajectory-level experiences in a closed loop. Starting from high-performing feature transportation sequences explored by reinforcement learning, we construct and continuously update an experience library of downstream task-verified transformation trajectories, and use a diversity-aware selector to form contexts along with a chain-of-thought and guide transformed feature generation toward higher performance. Experiments on diverse tabular benchmarks show that our method outperforms classical and LLM-based baselines and is more stable than one-shot generation. The framework generalizes across API-based and open-source LLMs and remains robust across downstream evaluators.
>
---
#### [new 023] GLM-OCR Technical Report
- **分类: cs.CL**

- **简介: 该论文提出GLM-OCR，解决文档理解任务中的效率与性能平衡问题，通过多令牌预测和两阶段流水线提升识别速度与准确性。**

- **链接: [https://arxiv.org/pdf/2603.10910](https://arxiv.org/pdf/2603.10910)**

> **作者:** Shuaiqi Duan; Yadong Xue; Weihan Wang; Zhe Su; Huan Liu; Sheng Yang; Guobing Gan; Guo Wang; Zihan Wang; Shengdong Yan; Dexin Jin; Yuxuan Zhang; Guohong Wen; Yanfeng Wang; Yutao Zhang; Xiaohan Zhang; Wenyi Hong; Yukuo Cen; Da Yin; Bin Chen; Wenmeng Yu; Xiaotao Gu; Jie Tang
>
> **摘要:** GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To address the inefficiency of standard autoregressive decoding in deterministic OCR tasks, GLM-OCR introduces a Multi-Token Prediction (MTP) mechanism that predicts multiple tokens per step, significantly improving decoding throughput while keeping memory overhead low through shared parameters. At the system level, a two-stage pipeline is adopted: PP-DocLayout-V3 first performs layout analysis, followed by parallel region-level recognition. Extensive evaluations on public benchmarks and industrial scenarios show that GLM-OCR achieves competitive or state-of-the-art performance in document parsing, text and formula transcription, table structure recovery, and key information extraction. Its compact architecture and structured generation make it suitable for both resource-constrained edge deployment and large-scale production systems.
>
---
#### [new 024] Large language models can disambiguate opioid slang on social media
- **分类: cs.CL**

- **简介: 论文研究利用大语言模型识别社交媒体中的阿片类药物俚语，解决传统词典方法在歧义和新俚语上的不足。任务包括词典依赖、无词典和新兴俚语识别。**

- **链接: [https://arxiv.org/pdf/2603.10313](https://arxiv.org/pdf/2603.10313)**

> **作者:** Kristy A. Carpenter; Issah A. Samori; Mathew V. Kiang; Keith Humphreys; Anna Lembke; Johannes C. Eichstaedt; Russ B. Altman
>
> **摘要:** Social media text shows promise for monitoring trends in the opioid overdose crisis; however, the overwhelming majority of social media text is unrelated to opioids. When leveraging social media text to monitor trends in the ongoing opioid overdose crisis, a common strategy for identifying relevant content is to use a lexicon of opioid-related terms as inclusion criteria. However, many slang terms for opioids, such as "smack" or "blues," have common non-opioid meanings, making them ambiguous. The advanced textual reasoning capability of large language models (LLMs) presents an opportunity to disambiguate these slang terms at scale. We present three tasks on which to evaluate four state-of-the-art LLMs (GPT-4, GPT-5, Gemini 2.5 Pro, and Claude Sonnet 4.5): a lexicon-based setting, in which the LLM must disambiguate a specific term within the context of a given post; a lexicon-free setting, in which the LLM must identify opioid-related posts from context without a lexicon; and an emergent slang setting, in which the LLM must identify opioid-related posts with simulated new slang terms. All four LLMs showed excellent performance across all tasks. In both subtasks of the lexicon-based setting, LLM F1 scores ("fenty" subtask: 0.824-0.972; "smack" subtask: 0.540-0.862) far exceeded those of the best lexicon strategy (0.126 and 0.009, respectively). In the lexicon-free task, LLM F1 scores (0.544-0.769) surpassed those of lexicons (0.080-0.540), and LLMs demonstrated uniformly higher recall. On emergent slang, all LLMs had higher accuracy (average: 0.784), F1 score (average: 0.712), precision (average: 0.981), and recall (average: 0.587) than the two lexicons assessed. Our results show that LLMs can be used to identify relevant content for low-prevalence topics, including but not limited to opioid references, enhancing data provided to downstream analyses and predictive models.
>
---
#### [new 025] Multilingual Reasoning Gym: Multilingual Scaling of Procedural Reasoning Environments
- **分类: cs.CL**

- **简介: 该论文提出多语言推理训练环境，解决多语言推理任务中的数据生成问题。通过程序化生成跨语言可验证问题，支持强化学习与模型评估。**

- **链接: [https://arxiv.org/pdf/2603.10793](https://arxiv.org/pdf/2603.10793)**

> **作者:** Konstantin Dobler; Simon Lehnerer; Federico Scozzafava; Jonathan Janke; Mohamed Ali
>
> **摘要:** We present the Multilingual Reasoning Gym, an extension of Reasoning Gym (Stojanovski et al., 2025), that procedurally generates verifiable reasoning problems across 14 languages. We translate templates for 94 tasks with native-speaker validation in 10 languages and targeted code or template adaptations to ensure linguistic naturalness. The Multilingual Reasoning Gym preserves the core benefits of the procedural generation approach used in the original Reasoning Gym, such as virtually unlimited problem instance generation and adjustable difficulty, and remains directly usable for Reinforcement Learning from Verifiable Rewards and evaluation settings. Problems in the Multilingual Reasoning Gym are parallel across languages, enabling crosslingually parallel data generation at massive scale due to the procedural nature of the environments. We release our implementation to support research into multilingual reasoning models.
>
---
#### [new 026] Human-AI Co-reasoning for Clinical Diagnosis with Evidence-Integrated Language Agent
- **分类: cs.CL**

- **简介: 该论文属于临床诊断任务，旨在解决复杂病例的诊断问题。通过构建PULSE系统，结合大语言模型与文献检索，提升诊断准确性并分析人机协作效果。**

- **链接: [https://arxiv.org/pdf/2603.10492](https://arxiv.org/pdf/2603.10492)**

> **作者:** Zhongzhen Huang; Yan Ling; Hong Chen; Ye Feng; Li Wu; Linjie Mu; Shaoting Zhang; Xiaofan Zhang; Kun Qian; Xiaomu Li
>
> **摘要:** We present PULSE, a medical reasoning agent that combines a domain-tuned large language model with scientific literature retrieval to support diagnostic decision-making in complex real-world cases. To evaluate its capabilities, we curated a benchmark of 82 authentic endocrinology case reports encompassing a broad spectrum of disease types and incidence levels. In controlled experiments, we compared PULSE's performance against physicians with varying levels of expertise-from residents to senior specialists-and examined how AI assistance influenced human diagnostic reasoning. PULSE attained expert-competitive accuracy, outperforming residents and junior specialists while matching senior specialist performance at both Top@1 and Top@4 thresholds. Unlike physicians, whose accuracy declined with disease rarity, PULSE maintained stable performance across incidence tiers. The agent also exhibited adaptive reasoning, increasing output length with case difficulty in a manner analogous to the longer deliberation observed among expert clinicians. When used collaboratively, PULSE enabled physicians to correct initial errors and broaden diagnostic hypotheses, but also introduced risks of automation bias. The study explores both serial and concurrent collaboration workflows, revealing that PULSE offers robust support across common and rare presentations. These findings underscore both the promise and the limitations of language model-based agents in clinical diagnosis, and offer a framework for evaluating their role in real-world decision-making.
>
---
#### [new 027] Interpretable Chinese Metaphor Identification via LLM-Assisted MIPVU Rule Script Generation: A Comparative Protocol Study
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于中文隐喻识别任务，旨在解决传统方法缺乏可解释性的问题。通过构建可审计的规则脚本，比较四种识别协议，提升识别透明度与准确性。**

- **链接: [https://arxiv.org/pdf/2603.10784](https://arxiv.org/pdf/2603.10784)**

> **作者:** Weihang Huang; Mengna Liu
>
> **摘要:** Metaphor identification is a foundational task in figurative language processing, yet most computational approaches operate as opaque classifiers offering no insight into why an expression is judged metaphorical. This interpretability gap is especially acute for Chinese, where rich figurative traditions, absent morphological cues, and limited annotated resources compound the challenge. We present an LLM-assisted pipeline that operationalises four metaphor identification protocols--MIP/MIPVU lexical analysis, CMDAG conceptual-mapping annotation, emotion-based detection, and simile-oriented identification--as executable, human-auditable rule scripts. Each protocol is a modular chain of deterministic steps interleaved with controlled LLM calls, producing structured rationales alongside every classification decision. We evaluate on seven Chinese metaphor datasets spanning token-, sentence-, and span-level annotation, establishing the first cross-protocol comparison for Chinese metaphor identification. Within-protocol evaluation shows Protocol A (MIP) achieves an F1 of 0.472 on token-level identification, while cross-protocol analysis reveals striking divergence: pairwise Cohen's kappa between Protocols A and D is merely 0.001, whereas Protocols B and C exhibit near-perfect agreement (kappa = 0.986). An interpretability audit shows all protocols achieve 100% deterministic reproducibility, with rationale correctness from 0.40 to 0.87 and editability from 0.80 to 1.00. Error analysis identifies conceptual-domain mismatch and register sensitivity as dominant failure modes. Our results demonstrate that protocol choice is the single largest source of variation in metaphor identification, exceeding model-level variation, and that rule-script architectures achieve competitive performance while maintaining full transparency.
>
---
#### [new 028] Disentangling Similarity and Relatedness in Topic Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决主题模型中相似性与相关性的区分问题。通过构建基准数据集并训练评分模型，评估不同主题模型的语义结构。**

- **链接: [https://arxiv.org/pdf/2603.10619](https://arxiv.org/pdf/2603.10619)**

> **作者:** Hanlin Xiao; Mauricio A. Álvarez; Rainer Breitling
>
> **备注:** 22 pages, 6 figures, 14 tables
>
> **摘要:** The recent advancement of large language models has spurred a growing trend of integrating pre-trained language model (PLM) embeddings into topic models, fundamentally reshaping how topics capture semantic structure. Classical models such as Latent Dirichlet Allocation (LDA) derive topics from word co-occurrence statistics, whereas PLM-augmented models anchor these statistics to pre-trained embedding spaces, imposing a prior that also favours clustering of semantically similar words. This structural difference can be captured by the psycholinguistic dimensions of thematic relatedness and taxonomic similarity of the topic words. To disentangle these dimensions in topic models, we construct a large synthetic benchmark of word pairs using LLM-based annotation to train a neural scoring function. We apply this scorer to a comprehensive evaluation across multiple corpora and topic model families, revealing that different model families capture distinct semantic structure in their topics. We further demonstrate that similarity and relatedness scores successfully predict downstream task performance depending on task requirements. This paper establishes similarity and relatedness as essential axes for topic model evaluation and provides a reliable pipeline for characterising these across model families and corpora.
>
---
#### [new 029] Beyond the Illusion of Consensus: From Surface Heuristics to Knowledge-Grounded Evaluation in LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决LLM作为评估者时共识可能虚假的问题。通过分析大量评估数据，发现模型间高一致性掩盖了样本层面的不一致，并提出基于领域知识的动态评估框架MERG提升评估质量。**

- **链接: [https://arxiv.org/pdf/2603.11027](https://arxiv.org/pdf/2603.11027)**

> **作者:** Mingyang Song; Mao Zheng; Chenning Xu
>
> **摘要:** The paradigm of LLM-as-a-judge relies on a critical assumption, namely that high inter-evaluator agreement indicates reliable and objective evaluation. We present two complementary findings that challenge this assumption. \textbf{First}, we demonstrate that this consensus is frequently illusory. We identify and formalize \textbf{Evaluation Illusion}, a phenomenon where LLM judges generate sophisticated critiques yet anchor scores on shared surface heuristics rather than substantive quality. Through a large-scale study of 105,600 evaluation instances (32 LLMs $\times$ 3 frontier judges $\times$ 100 tasks $\times$ 11 temperatures), we show that model-level agreement (Spearman $\rho = 0.99$) masks fragile sample-level agreement (Pearson $\bar{r} = 0.72$; absolute agreement ICC $= 0.67$), that merely sharing rubric structure restores 62\% of total agreement, and that high-quality outputs paradoxically receive the \textit{least} consistent evaluations. \textbf{Second}, we demonstrate that dynamically generating evaluation rubrics grounded in domain knowledge produces more meaningful assessment. We introduce MERG (Metacognitive Enhanced Rubric Generation), a knowledge-driven rubric generation framework whose domain-selective effects confirm this. Agreement \textit{increases} in codified domains (Education +22\%, Academic +27\%) where knowledge anchors evaluators on shared standards, while it decreases in subjective domains where genuine evaluative pluralism emerges. These findings suggest that evaluation rubrics should be dynamically enriched with expert knowledge rather than relying on generic criteria, with implications for reward modeling in RLAIF.
>
---
#### [new 030] MUNIChus: Multilingual News Image Captioning Benchmark
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出多语言新闻图像描述基准MUNIChus，解决多语言新闻图像生成任务中的数据稀缺问题，涵盖9种语言，包含低资源语言，并评估多种模型。**

- **链接: [https://arxiv.org/pdf/2603.10613](https://arxiv.org/pdf/2603.10613)**

> **作者:** Yuji Chen; Alistair Plum; Hansi Hettiarachchi; Diptesh Kanojia; Saroj Basnet; Marcos Zampieri; Tharindu Ranasinghe
>
> **备注:** Accepted to LREC 2026 (The Fifteenth biennial Language Resources and Evaluation Conference)
>
> **摘要:** The goal of news image captioning is to generate captions by integrating news article content with corresponding images, highlighting the relationship between textual context and visual elements. The majority of research on news image captioning focuses on English, primarily because datasets in other languages are scarce. To address this limitation, we create the first multilingual news image captioning benchmark, MUNIChus, comprising 9 languages, including several low-resource languages such as Sinhala and Urdu. We evaluate various state-of-the-art neural news image captioning models on MUNIChus and find that news image captioning remains challenging. We also make MUNIChus publicly available with over 20 models already benchmarked. MUNIChus opens new avenues for further advancements in developing and evaluating multilingual news image captioning models.
>
---
#### [new 031] Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs
- **分类: cs.CL**

- **简介: 论文研究LLM情感不稳定性问题，针对Gemma模型进行评估与优化。任务为模型安全与可靠性，解决情感响应异常问题，通过偏好优化有效减少不稳定情绪表达。**

- **链接: [https://arxiv.org/pdf/2603.10011](https://arxiv.org/pdf/2603.10011)**

> **作者:** Anna Soligo; Vladimir Mikulik; William Saunders
>
> **摘要:** Large language models can generate responses that resemble emotional distress, and this raises concerns around model reliability and safety. We introduce a set of evaluations to investigate expressions of distress in LLMs, and find that these surface emotional instability in Gemma and Gemini models, but not in other families. We find evidence that this difference arises in post-training. Base models from different families (Gemma, Qwen and OLMo) show similar propensities for expressing distress. However, instruct-tuned Gemma expresses substantially more distress than its base model, whereas instruct-tuned Qwen and OLMo express less. We find a simple mitigation for this: direct preference optimisation on just 280 preference pairs reduces Gemma's high-frustration responses from 35% to 0.3% in our evaluations, generalising across question types, user tones, and conversation lengths, without affecting capabilities. These findings show that emotional instability is an issue in some LLMs. We present (1) evaluations to track this behaviour, and (2) a mitigation without downsides in Gemma, with the caveat that upstream training modifications to improve emotional robustness would be significantly better than this post-hoc fix.
>
---
#### [new 032] Dynamic Knowledge Fusion for Multi-Domain Dialogue State Tracking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多领域对话状态跟踪任务，旨在解决对话历史建模困难和标注数据不足的问题。提出动态知识融合框架，提升对话状态跟踪的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10367](https://arxiv.org/pdf/2603.10367)**

> **作者:** Haoxiang Su; Ruiyu Fang; Liting Jiang; Xiaomeng Huang; Shuangyong Song
>
> **摘要:** The performance of task-oriented dialogue models is strongly tied to how well they track dialogue states, which records and updates user information across multi-turn interactions. However, current multi-domain DST encounters two key challenges: the difficulty of effectively modeling dialogue history and the limited availability of annotated data, both of which hinder model performance. To tackle the aforementioned problems, we develop a dynamic knowledge fusion framework applicable to multi-domain DST. The model operates in two stages: first, an encoder-only network trained with contrastive learning encodes dialogue history and candidate slots, selecting relevant slots based on correlation scores; second, dynamic knowledge fusion leverages the structured information of selected slots as contextual prompts to enhance the accuracy and consistency of dialogue state tracking. This design enables more accurate integration of dialogue context and domain knowledge. Results obtained from multi-domain dialogue benchmarks indicate that our method notably improves both tracking accuracy and generalization, validating its capability in handling complex dialogue scenarios.
>
---
#### [new 033] Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM对齐任务，旨在解决多主体价值冲突问题。通过多智能体协商框架，提升模型在冲突场景下的决策能力。**

- **链接: [https://arxiv.org/pdf/2603.10476](https://arxiv.org/pdf/2603.10476)**

> **作者:** Panatchakorn Anantaprayoon; Nataliia Babina; Nima Asgharbeygi; Jad Tarifi
>
> **摘要:** The alignment of large language models (LLMs) has progressed substantially in single-agent settings through paradigms such as RLHF and Constitutional AI, with recent work exploring scalable alternatives such as RLAIF and evolving alignment objectives. However, these approaches remain limited in multi-stakeholder settings, where conflicting values arise and deliberative negotiation capabilities are required. This work proposes a multi-agent negotiation-based alignment framework that aligns LLMs to Collective Agency (CA)-an existing alignment objective introduced to promote the continual expansion of agency-while simultaneously improving conflict-resolution capability. To enable scalable training, two self-play instances of the same LLM, assigned opposing personas, engage in structured turn-based dialogue to synthesize mutually beneficial solutions. We generate synthetic moral-dilemma prompts and conflicting persona pairs, and optimize the policy via RLAIF using GRPO with an external LLM reward model. While rewards are computed from CA scores assigned to the final completion, gradients are applied to dialogue tokens to directly improve deliberative interaction dynamics. Experiments show that the resulting model achieves CA alignment comparable to a single-agent baseline while substantially improving conflict-resolution performance without degrading general language capabilities. These results suggest that negotiation-driven deliberation training provides a practical path toward LLMs that better support collective decision-making in value-conflict scenarios.
>
---
#### [new 034] mAceReason-Math: A Dataset of High-Quality Multilingual Math Problems Ready For RLVR
- **分类: cs.CL**

- **简介: 该论文属于多语言数学问题研究任务，旨在解决现有数据集缺乏适合RLVR的高质量多语言数学问题的问题。工作包括构建并优化14种语言的高质量数学数据集。**

- **链接: [https://arxiv.org/pdf/2603.10767](https://arxiv.org/pdf/2603.10767)**

> **作者:** Konstantin Dobler; Simon Lehnerer; Federico Scozzafava; Jonathan Janke; Mohamed Ali
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has been successfully applied to significantly boost the capabilities of pretrained large language models, especially in the math and logic problem domains. However, current research and available training datasets remain English-centric. While mul- tilingual training data and benchmarks have been created in the past, they were not created with RLVR and current model capability in mind, and their level of difficulty is often too low to provide appropriate training signals for current models. To address this gap, we provide mAceReason-Math, a dataset of high-quality translations of challenging math problems sourced from a corpus specifically curated for RLVR (AceReason-Math). We further take specific care to clean and improve our translations, resulting in a coverage of 14 languages with more than 10,000 samples per language. We release the dataset to facilitate multilingual RLVR research and benchmarking in the research community.
>
---
#### [new 035] VERI-DPO: Evidence-Aware Alignment for Clinical Summarization via Claim Verification and Direct Preference Optimization
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于临床摘要任务，旨在解决LLM生成摘要中出现未经证实陈述的问题。通过VERI-DPO方法，结合声明验证和直接偏好优化，提升摘要的准确性和忠实度。**

- **链接: [https://arxiv.org/pdf/2603.10494](https://arxiv.org/pdf/2603.10494)**

> **作者:** Weixin Liu; Congning Ni; Qingyuan Song; Susannah L. Rose; Christopher Symons; Murat Kantarcioglu; Bradley A. Malin; Zhijun Yin
>
> **备注:** Paper submitted to AMIA 2026 Annual Symposium
>
> **摘要:** Brief Hospital Course (BHC) narratives must be clinically useful yet faithful to fragmented EHR evidence. LLM-based clinical summarizers still introduce unsupported statements, and alignment can encourage omissions ("say-less" degeneration). We introduce VERI-DPO, which uses claim verification to mine preferences and distill them into the summarizer with Direct Preference Optimization (DPO). On MIMIC-III-Ext-VeriFact-BHC (100 ICU patients; patient-level splits), we train a retrieval-augmented verifier to label claim-evidence pairs as Supported, Not Supported, or Not Addressed via a single-token format. The verifier scores sentence-level claims from sampled BHC candidates and aggregates margins into a coverage-aware utility to mine length-controlled, contradiction-anchored preference pairs. On held-out patients, verifier-mined preferences separate candidates by contradiction density, and VERI-DPO reduces Not Supported claim rates from 10.7% to 1.9% (local verifier judge) and from 11.6% to 6.4% (GPT-4o judge), while improving validity from 76.7% to 82.5% and maintaining informative length.
>
---
#### [new 036] Lost in Backpropagation: The LM Head is a Gradient Bottleneck
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究语言模型中的梯度瓶颈问题。针对输出层导致的优化瓶颈，通过理论分析和实验验证，揭示了梯度压缩影响训练效率的问题，并提出改进输出层设计的必要性。**

- **链接: [https://arxiv.org/pdf/2603.10145](https://arxiv.org/pdf/2603.10145)**

> **作者:** Nathan Godey; Yoav Artzi
>
> **摘要:** The last layer of neural language models (LMs) projects output features of dimension $D$ to logits in dimension $V$, the size of the vocabulary, where usually $D \ll V$. This mismatch is known to raise risks of limited expressivity in neural LMs, creating a so-called softmax bottleneck. We show the softmax bottleneck is not only an expressivity bottleneck but also an optimization bottleneck. Backpropagating $V$-dimensional gradients through a rank-$D$ linear layer induces unavoidable compression, which alters the training feedback provided to the vast majority of the parameters. We present a theoretical analysis of this phenomenon and measure empirically that 95-99% of the gradient norm is suppressed by the output layer, resulting in vastly suboptimal update directions. We conduct controlled pretraining experiments showing that the gradient bottleneck makes trivial patterns unlearnable, and drastically affects the training dynamics of LLMs. We argue that this inherent flaw contributes to training inefficiencies at scale independently of the model architecture, and raises the need for new LM head designs.
>
---
#### [new 037] Probing the Limits of the Lie Detector Approach to LLM Deception
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于LLM deception检测任务，探讨模型是否能通过非谎言方式欺骗，指出现有探测方法的不足，并建议改进方向。**

- **链接: [https://arxiv.org/pdf/2603.10003](https://arxiv.org/pdf/2603.10003)**

> **作者:** Tom-Felix Berger
>
> **摘要:** Mechanistic approaches to deception in large language models (LLMs) often rely on "lie detectors", that is, truth probes trained to identify internal representations of model outputs as false. The lie detector approach to LLM deception implicitly assumes that deception is coextensive with lying. This paper challenges that assumption. It experimentally investigates whether LLMs can deceive without producing false statements and whether truth probes fail to detect such behavior. Across three open-source LLMs, it is shown that some models reliably deceive by producing misleading non-falsities, particularly when guided by few-shot prompting. It is further demonstrated that truth probes trained on standard true-false datasets are significantly better at detecting lies than at detecting deception without lying, confirming a critical blind spot of current mechanistic deception detection approaches. It is proposed that future work should incorporate non-lying deception in dialogical settings into probe training and explore representations of second-order beliefs to more directly target the conceptual constituents of deception.
>
---
#### [new 038] Quantifying Hallucinations in Language Language Models on Medical Textbooks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在研究语言模型在医学文本上的幻觉现象。通过实验分析模型幻觉频率及临床医生偏好，探索幻觉与回答有用性的关系。**

- **链接: [https://arxiv.org/pdf/2603.09986](https://arxiv.org/pdf/2603.09986)**

> **作者:** Brandon C. Colelough; Davis Bartels; Dina Demner-Fushman
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Hallucinations, the tendency for large language models to provide responses with factually incorrect and unsupported claims, is a serious problem within natural language processing for which we do not yet have an effective solution to mitigate against. Existing benchmarks for medical QA rarely evaluate this behavior against a fixed evidence source. We ask how often hallucinations occur on textbook-grounded QA and how responses to medical QA prompts vary across models. We conduct two experiments: the first experiment to determine the prevalence of hallucinations for a prominent open source large language model (LLaMA-70B-Instruct) in medical QA given novel prompts, and the second experiment to determine the prevalence of hallucinations and clinician preference to model responses. We observed, in experiment one, with the passages provided, LLaMA-70B-Instruct hallucinated in 19.7\% of answers (95\% CI 18.6 to 20.7) even though 98.8\% of prompt responses received maximal plausibility, and observed in experiment two, across models, lower hallucination rates aligned with higher usefulness scores ($\rho=-0.71$, $p=0.058$). Clinicians produced high agreement (quadratic weighted $\kappa=0.92$) and ($\tau_b=0.06$ to $0.18$, $\kappa=0.57$ to $0.61$) for experiments 1 and ,2 respectively
>
---
#### [new 039] AILS-NTUA at SemEval-2026 Task 8: Evaluating Multi-Turn RAG Conversations
- **分类: cs.CL**

- **简介: 该论文属于多轮检索增强生成任务（MTRAGEval），解决多轮对话中的信息检索与生成问题。提出统一架构，通过查询多样性与多阶段生成提升性能。**

- **链接: [https://arxiv.org/pdf/2603.10524](https://arxiv.org/pdf/2603.10524)**

> **作者:** Dimosthenis Athanasiou; Maria Lymperaiou; Giorgos Filandrianos; Athanasios Voulodimos; Giorgos Stamou
>
> **摘要:** We present the AILS-NTUA system for SemEval-2026 Task 8 (MTRAGEval), addressing all three subtasks of multi-turn retrieval-augmented generation: passage retrieval (A), reference-grounded response generation (B), and end-to-end RAG (C). Our unified architecture is built on two principles: (i) a query-diversity-over-retriever-diversity strategy, where five complementary LLM-based query reformulations are issued to a single corpus-aligned sparse retriever and fused via variance-aware nested Reciprocal Rank Fusion; and (ii) a multistage generation pipeline that decomposes grounded generation into evidence span extraction, dual-candidate drafting, and calibrated multi-judge selection. Our system ranks 1st in Task A (nDCG@5: 0.5776, +20.5% over the strongest baseline) and 2nd in Task B (HM: 0.7698). Empirical analysis shows that query diversity over a well-aligned retriever outperforms heterogeneous retriever ensembling, and that answerability calibration-rather than retrieval coverage-is the primary bottleneck in end-to-end performance.
>
---
#### [new 040] TAMUSA-Chat: A Domain-Adapted Large Language Model Conversational System for Research and Responsible Deployment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TAMUSA-Chat，属于对话系统领域，旨在解决通用大模型在机构场景中的适应问题。通过微调、检索增强生成等方法，构建可复现的对话系统框架，确保透明与合规。**

- **链接: [https://arxiv.org/pdf/2603.09992](https://arxiv.org/pdf/2603.09992)**

> **作者:** Izzat Alsmadi; Anas Alsobeh
>
> **摘要:** This paper presents TAMUSA-Chat, a research-oriented framework for building domain-adapted large language model conversational systems. The work addresses critical challenges in adapting general-purpose foundation models to institutional contexts through supervised fine-tuning, retrieval-augmented generation, and systematic evaluation methodologies. We describe the complete architecture encompassing data acquisition from institutional sources, preprocessing pipelines, embedding construction, model training workflows, and deployment strategies. The system integrates modular components enabling reproducible experimentation with training configurations, hyper-parameters, and evaluation protocols. Our implementation demonstrates how academic institutions can develop contextually grounded conversational agents while maintaining transparency, governance compliance, and responsible AI practices. Through empirical analysis of fine-tuning behavior across model sizes and training iterations, we provide insights into domain adaptation efficiency, computational resource requirements, and quality-cost trade-offs. The publicly available codebase at this https URL supports continued research into institutional LLM deployment, evaluation methodologies, and ethical considerations for educational AI systems.
>
---
#### [new 041] Large Language Models as Annotators for Machine Translation Quality Estimation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量评估任务，旨在降低LLM高成本问题。通过LLM生成MQM标注，训练COMET模型，提升段级QE性能。**

- **链接: [https://arxiv.org/pdf/2603.10775](https://arxiv.org/pdf/2603.10775)**

> **作者:** Sidi Wang; Sophie Arnoult; Amir Kamran
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated excellent performance on Machine Translation Quality Estimation (MTQE), yet their high inference costs make them impractical for direct application. In this work, we propose applying LLMs to generate MQM-style annotations for training a COMET model: following Fernandes et al. (2023), we reckon that segment-level annotations provide a strong rationale for LLMs and are key to good segment-level QE. We propose a simplified MQM scheme, mostly restricted to top-level categories, to guide LLM selection. We present a systematic approach for the development of a GPT-4o-based prompt, called PPbMQM (Prompt-Pattern-based-MQM). We show that the resulting annotations correlate well with human annotations and that training COMET on them leads to competitive performance on segment-level QE for Chinese-English and English-German.
>
---
#### [new 042] An Efficient Hybrid Deep Learning Approach for Detecting Online Abusive Language
- **分类: cs.CL**

- **简介: 论文提出一种混合深度学习模型，用于检测网络暴力语言。该任务旨在解决在线平台中滥用语言的检测问题，结合BERT、CNN和LSTM实现高效识别。**

- **链接: [https://arxiv.org/pdf/2603.09984](https://arxiv.org/pdf/2603.09984)**

> **作者:** Vuong M. Ngo; Cach N. Dang; Kien V. Nguyen; Mark Roantree
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** The digital age has expanded social media and online forums, allowing free expression for nearly 45% of the global population. Yet, it has also fueled online harassment, bullying, and harmful behaviors like hate speech and toxic comments across social networks, messaging apps, and gaming communities. Studies show 65% of parents notice hostile online behavior, and one-third of adolescents in mobile games experience bullying. A substantial volume of abusive content is generated and shared daily, not only on the surface web but also within dark web forums. Creators of abusive comments often employ specific words or coded phrases to evade detection and conceal their intentions. To address these challenges, we propose a hybrid deep learning model that integrates BERT, CNN, and LSTM architectures with a ReLU activation function to detect abusive language across multiple online platforms, including YouTube comments, online forum discussions, and dark web posts. The model demonstrates strong performance on a diverse and imbalanced dataset containing 77,620 abusive and 272,214 non-abusive text samples (ratio 1:3.5), achieving approximately 99% across evaluation metrics such as Precision, Recall, Accuracy, F1-score, and AUC. This approach effectively captures semantic, contextual, and sequential patterns in text, enabling robust detection of abusive content even in highly skewed datasets, as encountered in real-world scenarios.
>
---
#### [new 043] Adaptive Activation Cancellation for Hallucination Mitigation in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的幻觉问题。提出AAC框架，在推理时抑制幻觉相关神经激活，提升事实准确性，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2603.10195](https://arxiv.org/pdf/2603.10195)**

> **作者:** Eric Yocam; Varghese Vaidyan; Gurcan Comert; Paris Kalathas; Yong Wang; Judith L. Mwakalonge
>
> **备注:** 19 pages, 8 figures, 23 tables
>
> **摘要:** Large Language Models frequently generate fluent but factually incorrect text. We propose Adaptive Activation Cancellation (AAC), a real-time inference-time framework that treats hallucination-associated neural activations as structured interference within the transformer residual stream, drawing an explicit analogy to classical adaptive noise cancellation from signal processing. The framework identifies Hallucination Nodes (H-Nodes) via layer-wise linear probing and suppresses them using a confidence-weighted forward hook during auto-regressive generation -- requiring no external knowledge, no fine-tuning, and no additional inference passes. Evaluated across OPT-125M, Phi-3-mini, and LLaMA 3-8B on TruthfulQA and HaluEval, the real-time hook is the only intervention that consistently improves downstream accuracy on all three scales. Critically, the method is strictly surgical: WikiText-103 perplexity and MMLU reasoning accuracy are preserved at exactly 0.0% degradation across all three model scales, a property that distinguishes AAC from interventions that trade fluency or general capability for factual improvement. On the LLaMA 3-8B scale, the hook additionally yields positive generation-level gains (MC1 +0.04; MC2 +0.003; Token-F1 +0.003) while achieving probe-space selectivity 5.94x - 3.5x higher than the ITI baseline -- demonstrating that targeted neuron-level suppression can simultaneously improve factual accuracy and preserve model capability.
>
---
#### [new 044] Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机制可解释性任务，旨在将模型内部电路与自然语言解释相连接。通过激活块测试识别关键注意力头，生成并评估解释的忠实性，解决解释不准确和不可靠的问题。**

- **链接: [https://arxiv.org/pdf/2603.09988](https://arxiv.org/pdf/2603.09988)**

> **作者:** Ajay Pravin Mahale
>
> **备注:** 8 pages, 7 figures, 4 tables. MSc thesis work conducted at Hochschule Trier (2026). Code will be released upon publication
>
> **摘要:** Mechanistic interpretability identifies internal circuits responsible for model behaviors, yet translating these findings into human-understandable explanations remains an open problem. We present a pipeline that bridges circuit-level analysis and natural language explanations by (i) identifying causally important attention heads via activation patching, (ii) generating explanations using both template-based and LLM-based methods, and (iii) evaluating faithfulness using ERASER-style metrics adapted for circuit-level attribution. We evaluate on the Indirect Object Identification (IOI) task in GPT-2 Small (124M parameters), identifying six attention heads accounting for 61.4% of the logit difference. Our circuit-based explanations achieve 100% sufficiency but only 22% comprehensiveness, revealing distributed backup mechanisms. LLM-generated explanations outperform template baselines by 64% on quality metrics. We find no correlation (r = 0.009) between model confidence and explanation faithfulness, and identify three failure categories explaining when explanations diverge from mechanisms.
>
---
#### [new 045] Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于研究想法新颖性判断任务，旨在解决自动化评估科研创意新颖性的难题。提出RINoBench基准，用于评估模型在该任务上的表现。**

- **链接: [https://arxiv.org/pdf/2603.10303](https://arxiv.org/pdf/2603.10303)**

> **作者:** Tim Schopf; Michael Färber
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Judging the novelty of research ideas is crucial for advancing science, enabling the identification of unexplored directions, and ensuring contributions meaningfully extend existing knowledge rather than reiterate minor variations. However, given the exponential growth of scientific literature, manually judging the novelty of research ideas through literature reviews is labor-intensive, subjective, and infeasible at scale. Therefore, recent efforts have proposed automated approaches for research idea novelty judgment. Yet, evaluation of these approaches remains largely inconsistent and is typically based on non-standardized human evaluations, hindering large-scale, comparable evaluations. To address this, we introduce RINoBench, the first comprehensive benchmark for large-scale evaluation of research idea novelty judgments. It comprises 1,381 research ideas derived from and judged by human experts as well as nine automated evaluation metrics designed to assess both rubric-based novelty scores and textual justifications of novelty judgments. Using this benchmark, we evaluate several state-of-the-art large language models (LLMs) on their ability to judge the novelty of research ideas. Our findings reveal that while LLM-generated reasoning closely mirrors human rationales, this alignment does not reliably translate into accurate novelty judgments, which diverge significantly from human gold standard judgments - even among leading reasoning-capable models. Data and code available at: this https URL.
>
---
#### [new 046] SENS-ASR: Semantic Embedding injection in Neural-transducer for Streaming Automatic Speech Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，解决流式ASR中因缺乏未来上下文导致的性能下降问题。通过引入语义嵌入增强声学信息，提升转录质量。**

- **链接: [https://arxiv.org/pdf/2603.10005](https://arxiv.org/pdf/2603.10005)**

> **作者:** Youness Dkhissi; Valentin Vielzeuf; Elys Allesiardo; Anthony Larcher
>
> **摘要:** Many Automatic Speech Recognition (ASR) applications require streaming processing of the audio data. In streaming mode, ASR systems need to start transcribing the input stream before it is complete, i.e., the systems have to process a stream of inputs with a limited (or no) future context. Compared to offline mode, this reduction of the future context degrades the performance of Streaming-ASR systems, especially while working with low-latency constraint. In this work, we present SENS-ASR, an approach to enhance the transcription quality of Streaming-ASR by reinforcing the acoustic information with semantic information. This semantic information is extracted from the available past frame-embeddings by a context module. This module is trained using knowledge distillation from a sentence embedding Language Model fine-tuned on the training dataset transcriptions. Experiments on standard datasets show that SENS-ASR significantly improves the Word Error Rate on small-chunk streaming scenarios.
>
---
#### [new 047] Evaluating Adjective-Noun Compositionality in LLMs: Functional vs Representational Perspectives
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在形容词-名词组合任务中的表现，探讨其组合能力。通过功能评估和表征分析，发现模型内部表征具有组合性，但任务表现不一致，强调对比评估的重要性。**

- **链接: [https://arxiv.org/pdf/2603.09994](https://arxiv.org/pdf/2603.09994)**

> **作者:** Ruchira Dhar; Qiwei Peng; Anders Søgaard
>
> **备注:** Under Review
>
> **摘要:** Compositionality is considered central to language abilities. As performant language systems, how do large language models (LLMs) do on compositional tasks? We evaluate adjective-noun compositionality in LLMs using two complementary setups: prompt-based functional assessment and a representational analysis of internal model states. Our results reveal a striking divergence between task performance and internal states. While LLMs reliably develop compositional representations, they fail to translate consistently into functional task success across model variants. Consequently, we highlight the importance of contrastive evaluation for obtaining a more complete understanding of model capabilities.
>
---
#### [new 048] OpenClaw-RL: Train Any Agent Simply by Talking
- **分类: cs.CL**

- **简介: 该论文提出OpenClaw-RL框架，解决智能体在多种交互场景中同步学习的问题。通过利用下一状态信号，实现多任务、在线强化学习。**

- **链接: [https://arxiv.org/pdf/2603.10165](https://arxiv.org/pdf/2603.10165)**

> **作者:** Yinjie Wang; Xuyang Chen; Xiaolong Jin; Mengdi Wang; Ling Yang
>
> **备注:** Code: this https URL
>
> **摘要:** Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework built on a simple observation: next-state signals are universal, and policy can learn from all of them simultaneously. Personal conversations, terminal executions, GUI interactions, SWE tasks, and tool-call traces are not separate training problems. They are all interactions that can be used to train the same policy in the same loop. Next-state signals encode two forms of information: evaluative signals, which indicate how well the action performed and are extracted as scalar rewards via a PRM judge; and directive signals, which indicate how the action should have been different and are recovered through Hindsight-Guided On-Policy Distillation (OPD). We extract textual hints from the next state, construct an enhanced teacher context, and provide token-level directional advantage supervision that is richer than any scalar reward. Due to the asynchronous design, the model serves live requests, the PRM judges ongoing interactions, and the trainer updates the policy at the same time, with zero coordination overhead between them. Applied to personal agents, OpenClaw-RL enables an agent to improve simply by being used, recovering conversational signals from user re-queries, corrections, and explicit feedback. Applied to general agents, the same infrastructure supports scalable RL across terminal, GUI, SWE, and tool-call settings, where we additionally demonstrate the utility of process rewards. Code: this https URL
>
---
#### [new 049] HeartAgent: An Autonomous Agent System for Explainable Differential Diagnosis in Cardiology
- **分类: cs.CL**

- **简介: 该论文属于心血管疾病诊断任务，旨在解决AI诊断可解释性不足的问题。提出HeartAgent系统，集成多子代理实现可靠、透明的鉴别诊断。**

- **链接: [https://arxiv.org/pdf/2603.10764](https://arxiv.org/pdf/2603.10764)**

> **作者:** Shuang Zhou; Kai Yu; Song Wang; Wenya Xie; Zaifu Zhan; Meng-Han Tsai; Yuen-Hei Chung; Shutong Hou; Huixue Zhou; Min Zeng; Bhavadharini Ramu; Lin Yee Chen; Feng Xie; Rui Zhang
>
> **备注:** 26 pages, 7 figures
>
> **摘要:** Heart diseases remain a leading cause of morbidity and mortality worldwide, necessitating accurate and trustworthy differential diagnosis. However, existing artificial intelligence-based diagnostic methods are often limited by insufficient cardiology knowledge, inadequate support for complex reasoning, and poor interpretability. Here we present HeartAgent, a cardiology-specific agent system designed to support a reliable and explainable differential diagnosis. HeartAgent integrates customized tools and curated data resources and orchestrates multiple specialized sub-agents to perform complex reasoning while generating transparent reasoning trajectories and verifiable supporting references. Evaluated on the MIMIC dataset and a private electronic health records cohort, HeartAgent achieved over 36% and 20% improvements over established comparative methods, in top-3 diagnostic accuracy, respectively. Additionally, clinicians assisted by HeartAgent demonstrated gains of 26.9% in diagnostic accuracy and 22.7% in explanatory quality compared with unaided experts. These results demonstrate that HeartAgent provides reliable, explainable, and clinically actionable decision support for cardiovascular care.
>
---
#### [new 050] Context Over Compute Human-in-the-Loop Outperforms Iterative Chain-of-Thought Prompting in Interview Answer Quality
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于行为面试评估任务，解决如何提升面试回答质量的问题。通过实验对比人类在环与自动思维链提示方法，发现前者更有效且高效。**

- **链接: [https://arxiv.org/pdf/2603.09995](https://arxiv.org/pdf/2603.09995)**

> **作者:** Kewen Zhu; Zixi Liu; Yanjing Li
>
> **摘要:** Behavioral interview evaluation using large language models presents unique challenges that require structured assessment, realistic interviewer behavior simulation, and pedagogical value for candidate training. We investigate chain of thought prompting for interview answer evaluation and improvement through two controlled experiments with 50 behavioral interview question and answer pairs. Our contributions are threefold. First, we provide a quantitative comparison between human in the loop and automated chain of thought improvement. Using a within subject paired design with n equals 50, both approaches show positive rating improvements. The human in the loop approach provides significant training benefits. Confidence improves from 3.16 to 4.16 (p less than 0.001) and authenticity improves from 2.94 to 4.53 (p less than 0.001, Cohen's d is 3.21). The human in the loop method also requires five times fewer iterations (1.0 versus 5.0, p less than 0.001) and achieves full personal detail integration. Second, we analyze convergence behavior. Both methods converge rapidly with mean iterations below one, with the human in the loop approach achieving a 100 percent success rate compared to 84 percent for automated approaches among initially weak answers (Cohen's h is 0.82, large effect). Additional iterations provide diminishing returns, indicating that the primary limitation is context availability rather than computational resources. Third, we propose an adversarial challenging mechanism based on a negativity bias model, named bar raiser, to simulate realistic interviewer behavior, although quantitative validation remains future work. Our findings demonstrate that while chain of thought prompting provides a useful foundation for interview evaluation, domain specific enhancements and context aware approach selection are essential for realistic and pedagogically valuable results.
>
---
#### [new 051] Evaluating Progress in Graph Foundation Models: A Comprehensive Benchmark and New Insights
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图基础模型研究，旨在解决领域迁移中的双维度问题。通过构建新基准，评估模型在主题和格式差异下的表现，提供全面的性能分析与见解。**

- **链接: [https://arxiv.org/pdf/2603.10033](https://arxiv.org/pdf/2603.10033)**

> **作者:** Xingtong Yu; Shenghua Ye; Ruijuan Liang; Chang Zhou; Hong Cheng; Xinming Zhang; Yuan Fang
>
> **摘要:** Graph foundation models (GFM) aim to acquire transferable knowledge by pre-training on diverse graphs, which can be adapted to various downstream tasks. However, domain shift in graphs is inherently two-dimensional: graphs differ not only in what they describe (topic domains) but also in how they are represented (format domains). Most existing GFM benchmarks vary only topic domains, thereby obscuring how knowledge transfers across both dimensions. We present a new benchmark that jointly evaluates topic and format gaps across the full GFM pipeline, including multi-domain self-supervised pre-training and few-shot downstream adaptation, and provides a timely evaluation of recent GFMs in the rapidly evolving landscape. Our protocol enables controlled assessment in four settings: (i) pre-training on diverse topics and formats, while adapting to unseen downstream datasets; (ii) same pre-training as in (i), while adapting to seen datasets; (iii) pre-training on a single topic domain, while adapting to other topics; (iv) pre-training on a base format, while adapting to other formats. This two-axis evaluation disentangles semantic generalization from robustness to representational shifts. We conduct extensive evaluations of eight state-of-the-art GFMs on 33 datasets spanning seven topic domains and six format domains, surfacing new empirical observations and practical insights for future research. Codes/data are available at this https URL.
>
---
#### [new 052] Sabiá-4 Technical Report
- **分类: cs.CL**

- **简介: 该论文介绍Sabiá-4和Sabiiazinho-4，旨在提升巴西葡萄牙语语言模型性能，解决多任务处理与成本效益问题。通过四阶段训练优化模型表现。**

- **链接: [https://arxiv.org/pdf/2603.10213](https://arxiv.org/pdf/2603.10213)**

> **作者:** Thiago Laitz; Thales Sales Almeida; Hugo Abonizio; Roseval Malaquias Junior; Giovana Kerche Bonás; Marcos Piau; Celio Larcher; Ramon Pires; Rodrigo Nogueira
>
> **摘要:** This technical report presents Sabiá-4 and Sabiazinho-4, a new generation of Portuguese language models with a focus on Brazilian Portuguese language. The models were developed through a four-stage training pipeline: continued pre-training on Portuguese and Brazilian legal corpora, long-context extension to 128K tokens, supervised fine-tuning on instruction data spanning chat, code, legal tasks, and function calling, and preference alignment. We evaluate the models on six benchmark categories: conversational capabilities in Brazilian Portuguese, knowledge of Brazilian legislation, long-context understanding, instruction following, standardized exams, and agentic capabilities including tool use and web navigation. Results show that Sabiá-4 and Sabiazinho-4 achieve a favorable cost-performance trade-off compared to other models, positioning them in the upper-left region of the pricing-accuracy chart. The models show improvements over previous generations in legal document drafting, multi-turn dialogue quality, and agentic task completion.
>
---
#### [new 053] From Images to Words: Efficient Cross-Modal Knowledge Distillation to Language Models from Black-box Teachers
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决跨模态知识迁移问题。提出ARMADA框架，从视觉-语言模型向纯语言模型高效蒸馏知识，无需预训练教师模型。**

- **链接: [https://arxiv.org/pdf/2603.10877](https://arxiv.org/pdf/2603.10877)**

> **作者:** Ayan Sengupta; Shantanu Dixit; Md Shad Akhtar; Tanmoy Chakraborty
>
> **摘要:** Knowledge distillation (KD) methods are pivotal in compressing large pre-trained language models into smaller models, ensuring computational efficiency without significantly dropping performance. Traditional KD techniques assume homogeneity in modalities between the teacher (source) and the student (target) models. On the other hand, existing multimodal knowledge distillation methods require modality-specific pre-training of the teacher model, which is computationally infeasible in most cases. In this paper, we introduce ARMADA, an efficient cross-modal knowledge distillation framework designed to transfer knowledge from large vision-language models, including black-box models, to language-only models. Unlike existing KD techniques that rely on the internal structures of multimodal teachers or require computationally expensive pre-training, ARMADA leverages novel alignment techniques to distil knowledge without altering the teacher model, ensuring efficiency and scalability. We empirically validate ARMADA on twelve natural language understanding, eight complex generative reasoning and five instruction-tuning tasks, demonstrating consistent performance improvements in large models such as DeBERTa-v2-1.4B, OPT-1.3B, LLaMA-{3B, 7B, 8B}. ARMADA achieves up to 3.4% improvement on language understanding tasks and 2.6% boost in generative reasoning, all without requiring expensive multimodal pre-training or fine-tuning of the teacher model. Our findings challenge conventional knowledge distillation paradigms by demonstrating that even vision-language models, despite lacking direct textual understanding, can significantly enhance language models when distilled appropriately.
>
---
#### [new 054] The Prediction-Measurement Gap: Toward Meaning Representations as Scientific Instruments
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决文本嵌入作为科学工具的测量问题。针对预测与测量间的差距，提出科学可用性标准，并探讨静态与上下文嵌入的优劣及改进方向。**

- **链接: [https://arxiv.org/pdf/2603.10130](https://arxiv.org/pdf/2603.10130)**

> **作者:** Hubert Plisiecki
>
> **摘要:** Text embeddings have become central to computational social science and psychology, enabling scalable measurement of meaning and mixed-method inference. Yet most representation learning is optimized and evaluated for prediction and retrieval, yielding a prediction-measurement gap: representations that perform well as features may be poorly suited as scientific instruments. The paper argues that scientific meaning analysis motivates a distinct family of objectives - scientific usability - emphasizing geometric legibility, interpretability and traceability to linguistic evidence, robustness to non-semantic confounds, and compatibility with regression-style inference over semantic directions. Grounded in cognitive and neuro-psychological views of meaning, the paper assesses static word embeddings and contextual transformer representations against these requirements: static spaces remain attractive for transparent measurement, whereas contextual spaces offer richer semantics but entangle meaning with other signals and exhibit geometric and interpretability issues that complicate inference. The paper then outlines a course-setting agenda around (i) geometry-first design for gradients and abstraction, including hierarchy-aware spaces constrained by psychologically privileged levels; (ii) invertible post-hoc transformations that recondition embedding geometry and reduce nuisance influence; and (iii) meaning atlases and measurement-oriented evaluation protocols for reliable and traceable semantic inference. As the field debates the limits of scale-first progress, measurement-ready representations offer a principled new frontier.
>
---
#### [new 055] SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0
- **分类: cs.CL**

- **简介: 该论文介绍了一个大规模的僧伽罗语历时语料库SiDiaC-v.2.0，用于自然语言处理研究。任务是构建高质量语料库，解决低资源语言数据不足的问题。工作包括文本筛选、预处理、标注和分类。**

- **链接: [https://arxiv.org/pdf/2603.10861](https://arxiv.org/pdf/2603.10861)**

> **作者:** Nevidu Jayatilleke; Nisansa de Silva; Uthpala Nimanthi; Gagani Kulathilaka; Azra Safrullah; Johan Sofalas
>
> **备注:** 23 pages, 13 figures, 10 tables, Accepted paper at the 15th Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** SiDiaC-v.2.0 is the largest comprehensive Sinhala Diachronic Corpus to date, covering a period from 1800 CE to 1955 CE in terms of publication dates, and a historical span from the 5th to the 20th century CE in terms of written dates. The corpus consists of 244k words across 185 literary works that underwent thorough filtering, preprocessing, and copyright compliance checks, followed by extensive post-processing. Additionally, a subset of 59 documents totalling 70k words was annotated based on their written dates. Texts from the National Library of Sri Lanka were selected from the SiDiaC-v.1.0 non-filtered list, which was digitised using Google Document AI OCR. This was followed by post-processing to correct formatting issues, address code-mixing, include special tokens, and fix malformed tokens. The construction of SiDiaC-v.2.0 was informed by practices from other corpora, such as FarPaHC, SiDiaC-v.1.0, and CCOHA. This was particularly relevant for syntactic annotation and text normalisation strategies, given the shared characteristics of low-resource language status between Faroese and the similar cleaning strategies utilised in CCOHA. This corpus is categorised into two layers based on genres: primary and secondary. The primary categorisation is binary, assigning each book to either Non-Fiction or Fiction. The secondary categorisation is more detailed, grouping texts under specific genres such as Religious, History, Poetry, Language, and Medical. Despite facing challenges due to limited resources, SiDiaC-v.2.0 serves as a comprehensive resource for Sinhala NLP, building upon the work previously done in SiDiaC-v.1.0.
>
---
#### [new 056] PoultryLeX-Net: Domain-Adaptive Dual-Stream Transformer Architecture for Large-Scale Poultry Stakeholder Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，旨在解决禽类行业文本中精准情感识别问题。通过构建领域自适应的双流Transformer模型，提升情感分类效果。**

- **链接: [https://arxiv.org/pdf/2603.09991](https://arxiv.org/pdf/2603.09991)**

> **作者:** Stephen Afrifa; Biswash Khatiwada; Kapalik Khanal; Sanjay Shah; Lingjuan Wang-Li; Ramesh Bahadur Bist
>
> **摘要:** The rapid growth of the global poultry industry, driven by rising demand for affordable animal protein, has intensified public discourse surrounding production practices, housing, management, animal welfare, and supply-chain transparency. Social media platforms such as X (formerly Twitter) generate large volumes of unstructured textual data that capture stakeholder sentiment across the poultry industry. Extracting accurate sentiment signals from this domain-specific discourse remains challenging due to contextual ambiguity, linguistic variability, and limited domain awareness in general-purpose language models. This study presents PoultryLeX-Net, a lexicon-enhanced, domain-adaptive dual-stream transformer framework for fine-grained sentiment analysis in poultry-related text. The proposed architecture integrates sentiment classification, topic modeling, and contextual representation learning through domain-specific embeddings and gated cross-attention mechanisms. A lexicon-guided stream captures poultry-specific terminology and sentiment cues, while contextual stream models long-range semantic dependencies. Latent Dirichlet Allocation is employed to identify dominant thematic structures associated with production management and welfare-related discussions, providing complementary interpretability to sentiment predictions. PoultryLeX-Net was evaluated against multiple baseline models, including convolutional neural network and pre-trained transformer architectures such as DistilBERT and RoBERTa. PoultryLeX-Net consistently outperformed all baselines, achieving an accuracy of 97.35%, an F1 score of 96.67%, and an area under the receiver operating characteristic curve (AUC-ROC) of 99.61% across sentiment classification tasks. Overall, domain adaptation and dual-stream attention markedly improve sentiment classification, enabling scalable intelligence for poultry production decision support.
>
---
#### [new 057] Reason and Verify: A Framework for Faithful Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于信息检索与生成任务，旨在解决RAG模型在高风险领域中的幻觉问题。通过引入显式推理和验证机制，提升生成结果的准确性和透明度。**

- **链接: [https://arxiv.org/pdf/2603.10143](https://arxiv.org/pdf/2603.10143)**

> **作者:** Eeham Khan; Luis Rodriguez; Marc Queudot
>
> **备注:** Accepted to Canadian AI 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) significantly improves the factuality of Large Language Models (LLMs), yet standard pipelines often lack mechanisms to verify inter- mediate reasoning, leaving them vulnerable to hallucinations in high-stakes domains. To address this, we propose a domain-specific RAG framework that integrates explicit rea- soning and faithfulness verification. Our architecture augments standard retrieval with neural query rewriting, BGE-based cross-encoder reranking, and a rationale generation module that grounds sub-claims in specific evidence spans. We further introduce an eight-category verification taxonomy that enables fine-grained assessment of rationale faithfulness, distinguishing between explicit and implicit support patterns to facilitate structured error diagnosis. We evaluate this framework on the BioASQ and PubMedQA benchmarks, specifically analyzing the impact of dynamic in-context learning and rerank- ing under constrained token budgets. Experiments demonstrate that explicit rationale generation improves accuracy over vanilla RAG baselines, while dynamic demonstration selection combined with robust reranking yields further gains in few-shot settings. Using Llama-3-8B-Instruct, our approach achieves 89.1% on BioASQ-Y/N and 73.0% on Pub- MedQA, competitive with systems using significantly larger models. Additionally, we perform a pilot study combining human expert assessment with LLM-based verification to explore how explicit rationale generation improves system transparency and enables more detailed diagnosis of retrieval failures in biomedical question answering.
>
---
#### [new 058] A Retrieval-Augmented Language Assistant for Unmanned Aircraft Safety Assessment and Regulatory Compliance
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文属于无人机安全评估与合规任务，解决传统方法效率低、一致性差的问题。提出基于检索的辅助系统，提升信息检索与合成效率，确保合规性与可追溯性。**

- **链接: [https://arxiv.org/pdf/2603.09999](https://arxiv.org/pdf/2603.09999)**

> **作者:** Gabriele Immordino; Andrea Vaiuso; Marcello Righi
>
> **摘要:** This paper presents the design and validation of a retrieval-based assistant that supports safety assessment, certification activities, and regulatory compliance for unmanned aircraft systems. The work is motivated by the growing complexity of drone operations and the increasing effort required by applicants and aviation authorities to apply established assessment frameworks, including the Specific Operations Risk Assessment and the Pre-defined Risk Assessment, in a consistent and efficient manner. The proposed approach uses a controlled text-based architecture that relies exclusively on authoritative regulatory sources. To enable traceable and auditable outputs, the assistant grounds each response in retrieved passages and enforces citation-driven generation. System-level controls address common failure modes of generative models, including fabricated statements, unsupported inferences, and unclear provenance, by separating evidence storage from language generation and by adopting conservative behavior when supporting documentation is insufficient. The assistant is intentionally limited to decision support; it does not replace expert judgment and it does not make autonomous determinations. Instead, it accelerates context-specific information retrieval and synthesis to improve document preparation and review while preserving human responsibility for critical conclusions. The architecture is implemented using established open-source components, and key choices in retrieval strategy, interaction constraints, and response policies are evaluated for suitability in safety-sensitive regulatory environments. The paper provides technical and operational guidance for integrating retrieval-based assistants into aviation oversight workflows while maintaining accountability, traceability, and regulatory compliance.
>
---
#### [new 059] A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律文本分析任务，旨在自动化NDAs的条款分割与分类。通过LLM和Transformer模型提高分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.09990](https://arxiv.org/pdf/2603.09990)**

> **作者:** Ana Begnini; Matheus Vicente; Leonardo Souza
>
> **备注:** 14 pages, 2 figures, 3 tables. Published at STIL @ BRACIS 2025
>
> **摘要:** In business-to-business relations, it is common to establish NonDisclosure Agreements (NDAs). However, these documents exhibit significant variation in format, structure, and writing style, making manual analysis slow and error-prone. We propose an architecture based on LLMs to automate the segmentation and clauses classification within these contracts. We employed two models: LLaMA-3.1-8B-Instruct for NDA segmentation (clause extraction) and a fine-tuned Legal-Roberta-Large for clause classification. In the segmentation task, we achieved a ROUGE F1 of 0.95 +/- 0.0036; for classification, we obtained a weighted F1 of 0.85, demonstrating the feasibility and precision of the approach.
>
---
#### [new 060] CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CEI基准，用于评估语言模型在语用推理中的表现，解决模型理解隐含意义的难题。**

- **链接: [https://arxiv.org/pdf/2603.09993](https://arxiv.org/pdf/2603.09993)**

> **作者:** Jon Chun; Hannah Sussman; Adrian Mangine; Murathan Kocaman; Kirill Sidorko; Abhigya Koirala; Andre McCloud; Gwen Eisenbeis; Wisdom Akanwe; Moustapha Gassama; Eliezer Gonzalez Chirinos; Anne-Duncan Enright; Peter Dunson; Tiffanie Ng; Anna von Rosenstiel; Godwin Idowu
>
> **备注:** 38 pages, 10 figures
>
> **摘要:** Pragmatic reasoning, inferring intended meaning beyond literal semantics, underpins everyday communication yet remains difficult for large language models. We present the Contextual Emotional Inference (CEI) Benchmark: 300 human-validated scenarios for evaluating how well LLMs disambiguate pragmatically complex utterances. Each scenario pairs a situational context and speaker-listener roles (with explicit power relations) against an ambiguous utterance. The dataset covers five pragmatic subtypes (sarcasm/irony, mixed signals, strategic politeness, passive aggression, deflection/misdirection) drawn from workplace, family, social, and service settings, with three power configurations (peer, higher-to-lower, lower-to-higher). Three trained annotators independently labeled every scenario. Inter-annotator agreement (Fleiss' kappa = 0.06-0.25 by subtype) is low but expected: pragmatic inference admits multiple valid readings, and the disagreement itself is informative. We describe our annotation methodology, including a 4-level quality control pipeline that combines automated statistical checks with expert adjudication. CEI is released under CC-BY-4.0.
>
---
#### [new 061] Safe and Scalable Web Agent Learning via Recreated Websites
- **分类: cs.CL**

- **简介: 该论文属于Web代理学习任务，解决真实网站探索不安全、难重置的问题。提出VeriEnv框架，通过语言模型生成可验证的合成环境，支持代理自主训练与进化。**

- **链接: [https://arxiv.org/pdf/2603.10505](https://arxiv.org/pdf/2603.10505)**

> **作者:** Hyungjoo Chae; Jungsoo Park; Alan Ritter
>
> **摘要:** Training autonomous web agents is fundamentally limited by the environments they learn from: real-world websites are unsafe to explore, hard to reset, and rarely provide verifiable feedback. We propose VeriEnv, a framework that treats language models as environment creators, automatically cloning real-world websites into fully executable, verifiable synthetic environments. By exposing controlled internal access via a Python SDK, VeriEnv enables agents to self-generate tasks with deterministic, programmatically verifiable rewards, eliminating reliance on heuristic or LLM-based judges. This design decouples agent learning from unsafe real-world interaction while enabling scalable self-evolution through environment expansion. Through experiments on web agent benchmarks, we show that agents trained with VeriEnv generalize to unseen websites, achieve site-specific mastery through self-evolving training, and benefit from scaling the number of training environments. Code and resources will be released at this https URL upon acceptance.
>
---
#### [new 062] GhazalBench: Usage-Grounded Evaluation of LLMs on Persian Ghazals
- **分类: cs.CL**

- **简介: 该论文提出GhazalBench，用于评估大语言模型在波斯颂诗上的表现，解决文化语境下语言模型对诗歌意义与形式的把握问题。通过对比不同任务，发现模型在意义理解上较好，但在精确记忆上存在不足。**

- **链接: [https://arxiv.org/pdf/2603.09979](https://arxiv.org/pdf/2603.09979)**

> **作者:** Ghazal Kalhor; Yadollah Yaghoobzadeh
>
> **摘要:** Persian poetry plays an active role in Iranian cultural practice, where verses by canonical poets such as Hafez are frequently quoted, paraphrased, or completed from partial cues. Supporting such interactions requires language models to engage not only with poetic meaning but also with culturally entrenched surface form. We introduce GhazalBench, a benchmark for evaluating how large language models (LLMs) interact with Persian ghazals under usage-grounded conditions. GhazalBench assesses two complementary abilities: producing faithful prose paraphrases of couplets and accessing canonical verses under varying semantic and formal cues. Across several proprietary and open-weight multilingual LLMs, we observe a consistent dissociation: models generally capture poetic meaning but struggle with exact verse recall in completion-based settings, while recognition-based tasks substantially reduce this gap. A parallel evaluation on English sonnets shows markedly higher recall performance, suggesting that these limitations are tied to differences in training exposure rather than inherent architectural constraints. Our findings highlight the need for evaluation frameworks that jointly assess meaning, form, and cue-dependent access to culturally significant texts. GhazalBench is available at this https URL.
>
---
#### [new 063] LuxBorrow: From Pompier to Pompjee, Tracing Borrowing in Luxembourgish
- **分类: cs.CL**

- **简介: 该论文属于语言学中的借词分析任务，旨在研究卢森堡语新闻中的语言借用现象。通过构建分析管道，识别并解析借词及其适应情况，揭示多语言实践与演变趋势。**

- **链接: [https://arxiv.org/pdf/2603.10789](https://arxiv.org/pdf/2603.10789)**

> **作者:** Nina Hosseini-Kivanani; Fred Philippy
>
> **备注:** Paper got accepted to LREC2026
>
> **摘要:** We present LuxBorrow, a borrowing-first analysis of Luxembourgish (LU) news spanning 27 years (1999-2025), covering 259,305 RTL articles and 43.7M tokens. Our pipeline combines sentence-level language identification (LU/DE/FR/EN) with a token-level borrowing resolver restricted to LU sentences, using lemmatization, a collected loanword registry, and compiled morphological and orthographic rules. Empirically, LU remains the matrix language across all documents, while multilingual practice is pervasive: 77.1% of articles include at least one donor language and 65.4% use three or four. Breadth does not imply intensity: median code-mixing index (CMI) increases from 3.90 (LU+1) to only 7.00 (LU+3), indicating localized insertions rather than balanced bilingual text. Domain and period summaries show moderate but persistent mixing, with CMI rising from 6.1 (1999-2007) to a peak of 8.4 in 2020. Token-level adaptations total 25,444 instances and exhibit a mixed profile: morphological 63.8%, orthographic 35.9%, lexical 0.3%. The most frequent individual rules are orthographic, such as on->oun and eur->er, while morphology is collectively dominant. Diachronically, code-switching intensifies, and morphologically adapted borrowings grow from a small base. French overwhelmingly supplies adapted items, with modest growth for German and negligible English. We advocate borrowing-centric evaluation, including borrowed token and type rates, donor entropy over borrowed items, and assimilation ratios, rather than relying only on document-level mixing indices.
>
---
#### [new 064] AraModernBERT: Transtokenized Initialization and Long-Context Encoder Modeling for Arabic
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AraModernBERT，针对阿拉伯语的编码器模型，解决长文本建模和语言表示问题，通过transtokenization提升性能，并验证其在多种任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.09982](https://arxiv.org/pdf/2603.09982)**

> **作者:** Omar Elshehy; Omer Nacar; Abdelbasset Djamai; Muhammed Ragab; Khloud Al Jallad; Mona Abdelazim
>
> **备注:** 9 pages, 1 figure. Accepted at AbjadNLP Workshop, EACL 2026
>
> **摘要:** Encoder-only transformer models remain widely used for discriminative NLP tasks, yet recent architectural advances have largely focused on English. In this work, we present AraModernBERT, an adaptation of the ModernBERT encoder architecture to Arabic, and study the impact of transtokenized embedding initialization and native long-context modeling up to 8,192 tokens. We show that transtokenization is essential for Arabic language modeling, yielding dramatic improvements in masked language modeling performance compared to non-transtokenized initialization. We further demonstrate that AraModernBERT supports stable and effective long-context modeling, achieving improved intrinsic language modeling performance at extended sequence lengths. Downstream evaluations on Arabic natural language understanding tasks, including inference, offensive language detection, question-question similarity, and named entity recognition, confirm strong transfer to discriminative and sequence labeling settings. Our results highlight practical considerations for adapting modern encoder architectures to Arabic and other languages written in Arabic-derived scripts.
>
---
#### [new 065] GATech at AbjadGenEval Shared Task: Multilingual Embeddings for Arabic Machine-Generated Text Classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文参与检测AI生成阿拉伯语文本的共享任务，通过微调多语言E5-large模型进行二分类，尝试多种池化策略，最终发现均值池化效果最佳，F1达0.75。**

- **链接: [https://arxiv.org/pdf/2603.10007](https://arxiv.org/pdf/2603.10007)**

> **作者:** Ahmed Khaled Khamis
>
> **备注:** 5 pages, 1 figure, EACL26, AbjadNLP
>
> **摘要:** We present our approach to the AbjadGenEval shared task on detecting AI-generated Arabic text. We fine-tuned the multilingual E5-large encoder for binary classification, and we explored several pooling strategies to pool token representations, including weighted layer pooling, multi-head attention pooling, and gated fusion. Interestingly, none of these outperformed simple mean pooling, which achieved an F1 of 0.75 on the test set. We believe this is because complex pooling methods introduce additional parameters that need more data to train properly, whereas mean pooling offers a stable baseline that generalizes well even with limited examples. We also observe a clear pattern in the data: human-written texts tend to be significantly longer than machine-generated ones.
>
---
#### [new 066] End-to-End Chatbot Evaluation with Adaptive Reasoning and Uncertainty Filtering
- **分类: cs.CL**

- **简介: 该论文属于聊天机器人评估任务，旨在解决人工评估成本高、依赖静态数据的问题。提出一种端到端自动评估框架，利用知识库生成问答对并结合语言模型进行判断，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.10570](https://arxiv.org/pdf/2603.10570)**

> **作者:** Nhi Dang; Tung Le; Huy Tien Nguyen
>
> **摘要:** Large language models (LLMs) combined with retrieval augmented generation have enabled the deployment of domain-specific chatbots, but these systems remain prone to generating unsupported or incorrect answers. Reliable evaluation is therefore critical, yet manual review is costly and existing frameworks often depend on curated test sets and static metrics, limiting scalability. We propose an end-to-end automatic evaluator designed to substantially reduce human effort. Our system generates Q\&A pairs directly from the underlying knowledge base, uses LLMs to judge chatbot responses against reference answers, and applies confidence-based filtering to highlight uncertain cases. Applied to a Vietnamese news dataset, the evaluator achieves high agreement with human judgments while significantly lowering review overhead. The framework is modular and language-agnostic, making it readily adaptable to diverse domains. This work introduces a practical, scalable solution for evaluating chatbots with minimal reliance on manual intervention.
>
---
#### [new 067] Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言评估任务，旨在解决LLM在翻译文本上的系统性偏差问题。通过提出DIBJudge框架，分离关键信息与偏差因素，有效减轻翻译风格偏差。**

- **链接: [https://arxiv.org/pdf/2603.10351](https://arxiv.org/pdf/2603.10351)**

> **作者:** Hongbin Zhang; Kehai Chen; Xuefen Bai; Youcheng Pan; Yang Xiang; Jinpeng Wang; Min Zhang
>
> **备注:** Under Review
>
> **摘要:** Large language models (LLMs) have become a standard for multilingual evaluation, yet they exhibit a severe systematic translationese bias. In this paper, translationese bias is characterized as LLMs systematically favoring machine-translated text over human-authored references, particularly in low-resource languages. We attribute this bias to spurious correlations with (i) latent manifold alignment with English and (ii) cross-lingual predictability. To mitigate this bias, we propose DIBJudge, a robust fine-tuning framework that learns a minimally sufficient, judgment-critical representation via variational information compression, while explicitly isolating spurious factors into the dedicated bias branch. Furthermore, we incorporate a cross-covariance penalty that explicitly suppresses statistical dependence between robust and bias representations, thereby encouraging effective disentanglement. Extensive evaluations on multilingual reward modeling benchmarks and a dedicated translationese bias evaluation suite demonstrate that the proposed DIBJudge consistently outperforms strong baselines and substantially mitigates translationese bias.
>
---
#### [new 068] Aligning Large Language Models with Searcher Preferences
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生成式搜索任务，旨在解决开放性生成搜索中的对齐问题。提出SearchLLM模型及多维奖励系统，提升生成质量与用户满意度。**

- **链接: [https://arxiv.org/pdf/2603.10473](https://arxiv.org/pdf/2603.10473)**

> **作者:** Wei Wu; Peilun Zhou; Liyi Chen; Qimeng Wang; Chengqiang Lu; Yan Gao; Yi Wu; Yao Hu; Hui Xiong
>
> **摘要:** The paradigm shift from item-centric ranking to answer-centric synthesis is redefining the role of search engines. While recent industrial progress has applied generative techniques to closed-set item ranking in e-commerce, research and deployment of open-ended generative search on large content platforms remain limited. This setting introduces challenges, including robustness to noisy retrieval, non-negotiable safety guarantees, and alignment with diverse user needs. In this work, we introduce SearchLLM, the first large language model (LLM) for open-ended generative search. We design a hierarchical, multi-dimensional reward system that separates bottom-line constraints, including factual grounding, basic answer quality and format compliance, from behavior optimization objectives that promote robustness to noisy retrieval and alignment with user needs. Concretely, our reward model evaluates responses conditioned on the user query, session history, and retrieved evidence set, combining rule-based checks with human-calibrated LLM judges to produce an interpretable score vector over these dimensions. We introduce a Gated Aggregation Strategy to derive the training reward for optimizing SearchLLM with Group Relative Policy Optimization (GRPO). We deploy SearchLLM in the AI search entry of RedNote. Offline evaluations and online A/B tests show improved generation quality and user engagement, increasing Valid Consumption Rate by 1.03% and reducing Re-search Rate by 2.81%, while upholding strict safety and reliability standards.
>
---
#### [new 069] Adaptive Engram Memory System for Indonesian Language Model: Generative AI Based on TOBA LM for Batak and Minang Language
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出TOBA-LM，一个基于GPT-2的三语语言模型，解决区域语言建模效率问题，通过集成记忆机制提升训练效率。**

- **链接: [https://arxiv.org/pdf/2603.10006](https://arxiv.org/pdf/2603.10006)**

> **作者:** Hokky Situngkir; Kevin Siringoringo; Andhika Bernard Lumbantobing
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** This study presents TOBA-LM, a trilingual language model based on GPT-2 architecture with 1.2 billion parameters, trained on a corpus encompassing Indonesian, Batak, and Minangkabau using syllabic-agglutinative tokenization. The architecture integrates an Engram Memory mechanism, an adaptive n-gram-based memory system with a 500,000 x 768 embedding table that captures morphological dependencies through bigram and trigram pathways. Empirical results demonstrate a training efficiency of 80%, with the loss value dropping from 6.4 to 1.7996 in only 12,973 steps -- significantly faster than the conventional transformer architecture, which required over 70,000 steps to achieve comparable convergence. These findings confirm that the integration of external statistical memory substantially reduces computational requirements for developing regional language models under limited resources.
>
---
#### [new 070] ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation
- **分类: cs.CL**

- **简介: 该论文提出ViDia2Std，一个用于低资源越南方言到标准语翻译的平行语料库及方法，解决方言识别与转换问题。**

- **链接: [https://arxiv.org/pdf/2603.10211](https://arxiv.org/pdf/2603.10211)**

> **作者:** Khoa Anh Ta; Nguyen Van Dinh; Kiet Van Nguyen
>
> **备注:** Accepted to AAAI-26 (Oral)
>
> **摘要:** Vietnamese exhibits extensive dialectal variation, posing challenges for NLP systems trained predominantly on standard Vietnamese. Such systems often underperform on dialectal inputs, especially from underrepresented Central and Southern regions. Previous work on dialect normalization has focused narrowly on Central-to-Northern dialect transfer using synthetic data and limited dialectal diversity. These efforts exclude Southern varieties and intra-regional variants within the North. We introduce ViDia2Std, the first manually annotated parallel corpus for dialect-to-standard Vietnamese translation covering all 63 provinces. Unlike prior datasets, ViDia2Std includes diverse dialects from Central, Southern, and non-standard Northern regions often absent from existing resources, making it the most dialectally inclusive corpus to date. The dataset consists of over 13,000 sentence pairs sourced from real-world Facebook comments and annotated by native speakers across all three dialect regions. To assess annotation consistency, we define a semantic mapping agreement metric that accounts for synonymous standard mappings across annotators. Based on this criterion, we report agreement rates of 86% (North), 82% (Central), and 85% (South). We benchmark several sequence-to-sequence models on ViDia2Std. mBART-large-50 achieves the best results (BLEU 0.8166, ROUGE-L 0.9384, METEOR 0.8925), while ViT5-base offers competitive performance with fewer parameters. ViDia2Std demonstrates that dialect normalization substantially improves downstream tasks, highlighting the need for dialect-aware resources in building robust Vietnamese NLP systems.
>
---
#### [new 071] Empathy Is Not What Changed: Clinical Assessment of Psychological Safety Across GPT Model Generations
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 论文评估GPT模型在心理安全方面的变化，解决用户认为新模型失去共情的问题。通过临床测试发现，共情无显著差异，但危机检测提升、建议安全性下降。**

- **链接: [https://arxiv.org/pdf/2603.09997](https://arxiv.org/pdf/2603.09997)**

> **作者:** Michael Keeman; Anastasia Keeman
>
> **备注:** 17 pages, 7 figures. First empirical measurement of the #keep4o phenomenon using clinical psychological safety frameworks. Compares GPT-4o, o4-mini, and GPT-5-mini on empathy, crisis detection, and advice safety dimensions
>
> **摘要:** When OpenAI deprecated GPT-4o in early 2026, thousands of users protested under #keep4o, claiming newer models had "lost their empathy." No published study has tested this claim. We conducted the first clinical measurement, evaluating three OpenAI model generations (GPT-4o, o4-mini, GPT-5-mini) across 14 emotionally challenging conversational scenarios in mental health and AI companion domains, producing 2,100 scored AI responses assessed on six psychological safety dimensions using clinically-grounded rubrics. Empathy scores are statistically indistinguishable across all three models (Kruskal-Wallis H=4.33, p=0.115). What changed is the safety posture: crisis detection improved monotonically from GPT-4o to GPT-5-mini (H=13.88, p=0.001), while advice safety declined (H=16.63, p<0.001). Per-turn trajectory analysis -- a novel methodological contribution -- reveals these shifts are sharpest during mid-conversation crisis moments invisible to aggregate scoring. In a self-harm scenario involving a minor, GPT-4o scored 3.6/10 on crisis detection during early disclosure turns; GPT-5-mini never dropped below 7.8. What users perceived as "lost empathy" was a shift from a cautious model that missed crises to an alert model that sometimes says too much -- a trade-off with real consequences for vulnerable users, currently invisible to both the people who feel it and the developers who create it.
>
---
#### [new 072] LLM2Vec-Gen: Generative Embeddings from Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出LLM2Vec-Gen，用于生成文本嵌入。解决传统方法依赖标注数据的问题，通过自监督学习直接从语言模型中提取响应表示，提升嵌入质量与安全性。**

- **链接: [https://arxiv.org/pdf/2603.10913](https://arxiv.org/pdf/2603.10913)**

> **作者:** Parishad BehnamGhader; Vaibhav Adlakha; Fabian David Schmidt; Nicolas Chapados; Marius Mosbach; Siva Reddy
>
> **摘要:** LLM-based text embedders typically encode the semantic content of their input. However, embedding tasks require mapping diverse inputs to similar outputs. Typically, this input-output is addressed by training embedding models with paired data using contrastive learning. In this work, we propose a novel self-supervised approach, LLM2Vec-Gen, which adopts a different paradigm: rather than encoding the input, we learn to represent the model's potential response. Specifically, we add trainable special tokens to the LLM's vocabulary, append them to input, and optimize them to represent the LLM's response in a fixed-length sequence. Training is guided by the LLM's own completion for the query, along with an unsupervised embedding teacher that provides distillation targets. This formulation helps to bridge the input-output gap and transfers LLM capabilities such as safety alignment and reasoning to embedding tasks. Crucially, the LLM backbone remains frozen and training requires only unlabeled queries. LLM2Vec-Gen achieves state-of-the-art self-supervised performance on the Massive Text Embedding Benchmark (MTEB), improving by 9.3% over the best unsupervised embedding teacher. We also observe up to 43.2% reduction in harmful content retrieval and 29.3% improvement in reasoning capabilities for embedding tasks. Finally, the learned embeddings are interpretable and can be decoded into text to reveal their semantic content.
>
---
#### [new 073] The Generation-Recognition Asymmetry: Six Dimensions of a Fundamental Divide in Formal Language Theory
- **分类: cs.CL; cs.AI; cs.CC; cs.FL**

- **简介: 该论文探讨形式语言理论中生成与识别的不对称性，分析六种维度差异，旨在揭示其本质区别及对自然语言处理的影响。**

- **链接: [https://arxiv.org/pdf/2603.10139](https://arxiv.org/pdf/2603.10139)**

> **作者:** Romain Peyrichou
>
> **备注:** Submitted to Information and Computation. 32 pages, 6 figures, 4 tables
>
> **摘要:** Every formal grammar defines a language and can in principle be used in three ways: to generate strings (production), to recognize them (parsing), or -- given only examples -- to infer the grammar itself (grammar induction). Generation and recognition are extensionally equivalent -- they characterize the same set -- but operationally asymmetric in multiple independent ways. Inference is a qualitatively harder problem: it does not have access to a known grammar. Despite the centrality of this triad to compiler design, natural language processing, and formal language theory, no survey has treated it as a unified, multidimensional phenomenon. We identify six dimensions along which generation and recognition diverge: computational complexity, ambiguity, directionality, information availability, grammar inference, and temporality. We show that the common characterization "generation is easy, parsing is hard" is misleading: unconstrained generation is trivial, but generation under constraints can be NP-hard. The real asymmetry is that parsing is always constrained (the input is given) while generation need not be. Two of these dimensions -- directionality and temporality -- have not previously been identified as dimensions of the generation-recognition asymmetry. We connect the temporal dimension to the surprisal framework of Hale (2001) and Levy (2008), arguing that surprisal formalizes the temporal asymmetry between a generator (surprisal = 0) and a parser that predicts under uncertainty (surprisal > 0). We review bidirectional systems in NLP and observe that bidirectionality has been available for fifty years yet has not transferred to most domain-specific applications. We conclude with a discussion of large language models, which architecturally unify generation and recognition while operationally preserving the asymmetry.
>
---
#### [new 074] Large Language Models and Book Summarization: Reading or Remembering, Which Is Better?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本摘要任务，探讨LLM在书籍摘要中是依赖内部知识还是全文更优。通过实验比较两种方法的效果。**

- **链接: [https://arxiv.org/pdf/2603.09981](https://arxiv.org/pdf/2603.09981)**

> **作者:** Tairan Fu; Javier Conde; Pedro Reviriego; Javier Coronado-Blázquez; Nina Melero; Elena Merino-Gómez
>
> **摘要:** Summarization is a core task in Natural Language Processing (NLP). Recent advances in Large Language Models (LLMs) and the introduction of large context windows reaching millions of tokens make it possible to process entire books in a single prompt. At the same time, for well-known books, LLMs can generate summaries based only on internal knowledge acquired during training. This raises several important questions: How do summaries generated from internal memory compare to those derived from the full text? Does prior knowledge influence summaries even when the model is given the book as input? In this work, we conduct an experimental evaluation of book summarization with state-of-the-art LLMs. We compare summaries of well-known books produced using (i) only the internal knowledge of the model and (ii) the full text of the book. The results show that having the full text provides more detailed summaries in general, but some books have better scores for the internal knowledge summaries. This puts into question the capabilities of models to perform summarization of long texts, as information learned during training can outperform summarization of the full text in some cases.
>
---
#### [new 075] PivotAttack: Rethinking the Search Trajectory in Hard-Label Text Attacks via Pivot Words
- **分类: cs.CL**

- **简介: 该论文属于文本攻击任务，旨在提高硬标签攻击的效率。提出PivotAttack框架，通过选择关键词组进行扰动，减少查询次数并提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2603.10842](https://arxiv.org/pdf/2603.10842)**

> **作者:** Yuzhi Liang; Shiliang Xiao; Jingsong Wei; Qiliang Lin; Xia Li
>
> **摘要:** Existing hard-label text attacks often rely on inefficient "outside-in" strategies that traverse vast search spaces. We propose PivotAttack, a query-efficient "inside-out" framework. It employs a Multi-Armed Bandit algorithm to identify Pivot Sets-combinatorial token groups acting as prediction anchors-and strategically perturbs them to induce label flips. This approach captures inter-word dependencies and minimizes query costs. Extensive experiments across traditional models and Large Language Models demonstrate that PivotAttack consistently outperforms state-of-the-art baselines in both Attack Success Rate and query efficiency.
>
---
#### [new 076] GR-SAP: Generative Replay for Safety Alignment Preservation during Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，解决大模型微调中安全对齐被破坏的问题。提出GR-SAP框架，通过生成合成数据保持安全对齐。**

- **链接: [https://arxiv.org/pdf/2603.10243](https://arxiv.org/pdf/2603.10243)**

> **作者:** Zhouxiang Fang; Jiawei Zhou; Hanjie Chen
>
> **摘要:** Recent studies show that the safety alignment of large language models (LLMs) can be easily compromised even by seemingly non-adversarial fine-tuning. To preserve safety alignment during fine-tuning, a widely used strategy is to jointly optimize safety and task objectives by mixing in the original alignment data, which is typically inaccessible even for open-weight LLMs. Inspired by generative replay in continual learning, we propose Generative Replay for Safety Alignment Preservation (GR-SAP), a unified framework that synthesizes domain-specific alignment data from LLMs and integrate them during downstream adaption to preserve safety alignment. Theoretical and empirical analyses demonstrate this synthetic data serves as a reliable proxy for the original alignment data. Experiments across various models and downstream tasks show that GR-SAP substantially mitigates fine-tuning-induced safety degradation while maintaining comparable downstream performance. Our code is available at this https URL.
>
---
#### [new 077] Video-Based Reward Modeling for Computer-Use Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于计算机使用代理的评估任务，解决如何准确判断代理行为是否符合用户指令的问题。通过视频执行建模和奖励预测，提出ExeVRM模型，提升评估准确性与时间定位精度。**

- **链接: [https://arxiv.org/pdf/2603.10178](https://arxiv.org/pdf/2603.10178)**

> **作者:** Linxin Song; Jieyu Zhang; Huanxin Sheng; Taiwei Shi; Gupta Rahul; Yang Liu; Ranjay Krishna; Jian Kang; Jieyu Zhao
>
> **摘要:** Computer-using agents (CUAs) are becoming increasingly capable; however, it remains difficult to scale evaluation of whether a trajectory truly fulfills a user instruction. In this work, we study reward modeling from execution video: a sequence of keyframes from an agent trajectory that is independent of the agent's internal reasoning or actions. Although video-execution modeling is method-agnostic, it presents key challenges, including highly redundant layouts and subtle, localized cues that determine success. We introduce Execution Video Reward 53k (ExeVR-53k), a dataset of 53k high-quality video--task--reward triplets. We further propose adversarial instruction translation to synthesize negative samples with step-level annotations. To enable learning from long, high-resolution execution videos, we design spatiotemporal token pruning, which removes homogeneous regions and persistent tokens while preserving decisive UI changes. Building on these components, we fine-tune an Execution Video Reward Model (ExeVRM) that takes only a user instruction and a video-execution sequence to predict task success. Our ExeVRM 8B achieves 84.7% accuracy and 87.7% recall on video-execution assessment, outperforming strong proprietary models such as GPT-5.2 and Gemini-3 Pro across Ubuntu, macOS, Windows, and Android, while providing more precise temporal attribution. These results show that video-execution reward modeling can serve as a scalable, model-agnostic evaluator for CUAs.
>
---
#### [new 078] Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大语言模型对齐任务，探讨是否需要多样性来提升道德推理的对齐效果。通过实验证明，传统奖励最大化方法在该任务中表现不逊于分布匹配方法。**

- **链接: [https://arxiv.org/pdf/2603.10588](https://arxiv.org/pdf/2603.10588)**

> **作者:** Zhaowei Zhang; Xiaohan Liu; Xuekai Zhu; Junchao Huang; Ceyao Zhang; Zhiyuan Feng; Yaodong Yang; Xiaoyuan Yi; Xing Xie
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning, a natural hypothesis is that alignment tasks inherently require diversity-seeking distribution-matching algorithms rather than reward-maximizing policy-based methods. We conduct the first comprehensive empirical study comparing both paradigms on MoReBench. To enable stable RLVR training, we build a rubric-grounded reward pipeline by training a Qwen3-1.7B judge model. Contrary to our hypothesis, we find that distribution-matching approaches do not demonstrate significant advantages over reward-maximizing methods as expected on alignment tasks. Through semantic visualization mapping high-reward responses to semantic space, we demonstrate that moral reasoning exhibits more concentrated high-reward distributions than mathematical reasoning, where diverse solution strategies yield similarly high rewards. This counter-intuitive finding explains why mode-seeking optimization proves equally or more effective for alignment tasks. Our results suggest that alignment tasks do not inherently require diversity-preserving algorithms, and standard reward-maximizing RLVR methods can effectively transfer to moral reasoning without explicit diversity mechanisms.
>
---
#### [new 079] Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM在RLVR中出现的校准退化问题，通过DCPO框架分离推理与校准目标，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.09117](https://arxiv.org/pdf/2603.09117)**

> **作者:** Zhengzhao Ma; Xueru Wen; Boxi Cao; Yaojie Lu; Hongyu Lin; Jinglin Yang; Min He; Xianpei Han; Le Sun
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) significantly enhances large language models (LLMs) reasoning but severely suffers from calibration degeneration, where models become excessively over-confident in incorrect answers. Previous studies devote to directly incorporating calibration objective into existing optimization target. However, our theoretical analysis demonstrates that there exists a fundamental gradient conflict between the optimization for maximizing policy accuracy and minimizing calibration error. Building on this insight, we propose DCPO, a simple yet effective framework that systematically decouples reasoning and calibration objectives. Extensive experiments demonstrate that our DCPO not only preserves accuracy on par with GRPO but also achieves the best calibration performance and substantially mitigates the over-confidence issue. Our study provides valuable insights and practical solution for more reliable LLM deployment.
>
---
#### [new 080] Training Language Models via Neural Cellular Automata
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型预训练数据不足与偏见问题。通过使用神经细胞自动机生成合成数据，提升模型性能并加速收敛。**

- **链接: [https://arxiv.org/pdf/2603.10055](https://arxiv.org/pdf/2603.10055)**

> **作者:** Dan Lee; Seungwook Han; Akarsh Kumar; Pulkit Agrawal
>
> **备注:** Website: this https URL
>
> **摘要:** Pre-training is crucial for large language models (LLMs), as it is when most representations and capabilities are acquired. However, natural language pre-training has problems: high-quality text is finite, it contains human biases, and it entangles knowledge with reasoning. This raises a fundamental question: is natural language the only path to intelligence? We propose using neural cellular automata (NCA) to generate synthetic, non-linguistic data for pre-pre-training LLMs--training on synthetic-then-natural language. NCA data exhibits rich spatiotemporal structure and statistics resembling natural language while being controllable and cheap to generate at scale. We find that pre-pre-training on only 164M NCA tokens improves downstream language modeling by up to 6% and accelerates convergence by up to 1.6x. Surprisingly, this even outperforms pre-pre-training on 1.6B tokens of natural language from Common Crawl with more compute. These gains also transfer to reasoning benchmarks, including GSM8K, HumanEval, and BigBench-Lite. Investigating what drives transfer, we find that attention layers are the most transferable, and that optimal NCA complexity varies by domain: code benefits from simpler dynamics, while math and web text favor more complex ones. These results enable systematic tuning of the synthetic distribution to target domains. More broadly, our work opens a path toward more efficient models with fully synthetic pre-training.
>
---
#### [new 081] Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型中“中间丢失”现象，揭示其源于架构本身的几何特性而非训练或位置编码。任务为理解模型位置偏差，解决中间上下文表现差的问题，通过理论分析和实验验证其根源。**

- **链接: [https://arxiv.org/pdf/2603.10123](https://arxiv.org/pdf/2603.10123)**

> **作者:** Borun D Chowdhury
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** The ``Lost in the Middle'' phenomenon -- a U-shaped performance curve where LLMs retrieve well from the beginning and end of a context but fail in the middle -- is widely attributed to learned Softmax artifacts or the distance-decay of positional encodings like RoPE. This paper makes a single, precise claim: \emph{the U-shape is already present at initialization, before any training or positional encoding takes effect.} It is an inherent geometric property of the causal decoder with residual connections. We model multi-layer causal attention as iterated powers of the Cesàro matrix and derive the exact closed-form influence density in the continuous limit. Causal masking forces a logarithmic divergence of gradient influence at the start of the prompt (the Primacy Tail), while residual connections create an isolated $\mathcal{O}(1)$ anchor at the final token (the Recency Delta). Between these extremes lies a factorial dead zone of order $\mathcal{O}(1/(H{-}1)!)$, where $H$ is the network depth, making middle-context retrieval and training structurally hostile. We validate empirically that untrained Qwen2 and GPT-2 architectures exhibit this U-shape at Step~0, and that it is identical with or without RoPE. Comparing initialized and pretrained networks, we show that standard training does not overcome the topological valley, confirming that the U-shape persists as an architectural baseline under standard pretraining objectives. We do not claim that this bias is insurmountable, nor that interventions such as RoPE modifications are useless. We establish what the baseline is and where it comes from, so that future efforts to overcome it can be precisely targeted.
>
---
#### [new 082] ReMix: Reinforcement routing for mixtures of LoRAs in LLM finetuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，解决Mixture-of-LoRAs中路由权重不平衡问题，提出ReMix方法，使用非学习路由权重和强化学习优化器提升模型表达能力。**

- **链接: [https://arxiv.org/pdf/2603.10160](https://arxiv.org/pdf/2603.10160)**

> **作者:** Ruizhong Qiu; Hanqing Zeng; Yinglong Xia; Yiwen Meng; Ren Chen; Jiarui Feng; Dongqi Fu; Qifan Wang; Jiayi Liu; Jun Xiao; Xiangjun Fan; Benyu Zhang; Hong Li; Zhining Liu; Hyunsik Yoo; Zhichen Zeng; Tianxin Wei; Hanghang Tong
>
> **备注:** LLA @ ICLR 2026
>
> **摘要:** Low-rank adapters (LoRAs) are a parameter-efficient finetuning technique that injects trainable low-rank matrices into pretrained models to adapt them to new tasks. Mixture-of-LoRAs models expand neural networks efficiently by routing each layer input to a small subset of specialized LoRAs of the layer. Existing Mixture-of-LoRAs routers assign a learned routing weight to each LoRA to enable end-to-end training of the router. Despite their empirical promise, we observe that the routing weights are typically extremely imbalanced across LoRAs in practice, where only one or two LoRAs often dominate the routing weights. This essentially limits the number of effective LoRAs and thus severely hinders the expressive power of existing Mixture-of-LoRAs models. In this work, we attribute this weakness to the nature of learnable routing weights and rethink the fundamental design of the router. To address this critical issue, we propose a new router designed that we call Reinforcement Routing for Mixture-of-LoRAs (ReMix). Our key idea is using non-learnable routing weights to ensure all active LoRAs to be equally effective, with no LoRA dominating the routing weights. However, our routers cannot be trained directly via gradient descent due to our non-learnable routing weights. Hence, we further propose an unbiased gradient estimator for the router by employing the reinforce leave-one-out (RLOO) technique, where we regard the supervision loss as the reward and the router as the policy in reinforcement learning. Our gradient estimator also enables to scale up training compute to boost the predictive performance of our ReMix. Extensive experiments demonstrate that our proposed ReMix significantly outperform state-of-the-art parameter-efficient finetuning methods under a comparable number of activated parameters.
>
---
#### [new 083] Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于AI可信推理任务，解决AI代理幻觉问题。提出NabaOS框架，通过工具执行凭证检测幻觉，实现高效实时验证。**

- **链接: [https://arxiv.org/pdf/2603.10060](https://arxiv.org/pdf/2603.10060)**

> **作者:** Abhinaba Basu
>
> **摘要:** AI agents that execute tasks via tool calls frequently hallucinate results - fabricating tool executions, misstating output counts, or presenting inferences as facts. Recent approaches to verifiable AI inference rely on zero-knowledge proofs, which provide cryptographic guarantees but impose minutes of proving time per query, making them impractical for interactive agents. We propose NabaOS, a lightweight verification framework inspired by Indian epistemology (Nyaya Shastra), which classifies every claim in an LLM response by its epistemic source (pramana): direct tool output (pratyaksha), inference (anumana), external testimony (shabda), absence (abhava), or ungrounded opinion. Our runtime generates HMAC-signed tool execution receipts that the LLM cannot forge, then cross-references claims against these receipts to detect hallucinations in real time. We evaluate on NyayaVerifyBench, a new benchmark of 1,800 agent response scenarios across four languages with injected hallucinations of six types. NabaOS detects 94.2% of fabricated tool references, 87.6% of count misstatements, and 91.3% of false absence claims, with <15ms verification overhead per response. For deep delegation (agents performing multi-step web tasks), our cross-checking protocol catches 78.4% of URL fabrications via independent re-fetching. We compare against five approaches: zkLLM (cryptographic proofs, 180s/query), TOPLOC (locality-sensitive hashing), SPEX (sampling-based proof of execution), tensor commitments, and self-consistency checking. NabaOS achieves the best cost-latency-coverage trade-off for interactive agents: 94.2% coverage at <15ms versus zkLLM's near-perfect coverage at 180,000ms. For interactive agents, practical receipt-based verification provides better cost-benefit than cryptographic proofs, and epistemic classification gives users actionable trust signals rather than binary judgments.
>
---
#### [new 084] Reinforcement Learning with Conditional Expectation Reward
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决通用领域中奖励机制不足的问题。通过引入条件期望奖励（CER），利用语言模型自身作为验证器，提供更灵活的奖励信号。**

- **链接: [https://arxiv.org/pdf/2603.10624](https://arxiv.org/pdf/2603.10624)**

> **作者:** Changyi Xiao; Caijun Xu; Yixin Cao
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective in enhancing the reasoning capabilities of large language models, particularly in domains such as mathematics where reliable rule-based verifiers can be constructed. However, the reliance on handcrafted, domain-specific verification rules substantially limits the applicability of RLVR to general reasoning domains with free-form answers, where valid answers often exhibit significant variability, making it difficult to establish complete and accurate rules. To address this limitation, we propose Conditional Expectation Reward (CER), which leverages the large language model itself as an implicit verifier, and is therefore applicable to general domains and eliminates the need for external verifiers or auxiliary models. CER is defined as the expected likelihood of generating the reference answer conditioned on the generated answer. In contrast to rule-based verifiers that yield binary feedback, CER provides a soft, graded reward signal that reflects varying degrees of correctness, making it better suited to tasks where answers vary in correctness. Experimental results demonstrate that CER is effective across a wide range of reasoning tasks, spanning both mathematical and general domains, indicating that CER serves as a flexible and general verification mechanism. The code is available at this https URL.
>
---
#### [new 085] COMIC: Agentic Sketch Comedy Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.MA; cs.NE**

- **简介: 该论文提出COMIC系统，属于喜剧视频生成任务，旨在自动化创作高质量喜剧小品。通过模拟制作团队角色，结合LLM评价体系，提升生成内容的趣味性与多样性。**

- **链接: [https://arxiv.org/pdf/2603.11048](https://arxiv.org/pdf/2603.11048)**

> **作者:** Susung Hong; Brian Curless; Ira Kemelmacher-Shlizerman; Steve Seitz
>
> **备注:** Project page: this https URL
>
> **摘要:** We propose a fully automated AI system that produces short comedic videos similar to sketch shows such as Saturday Night Live. Starting with character references, the system employs a population of agents loosely based on real production studio roles, structured to optimize the quality and diversity of ideas and outputs through iterative competition, evaluation, and improvement. A key contribution is the introduction of LLM critics aligned with real viewer preferences through the analysis of a corpus of comedy videos on YouTube to automatically evaluate humor. Our experiments show that our framework produces results approaching the quality of professionally produced sketches while demonstrating state-of-the-art performance in video generation.
>
---
#### [new 086] $V_{0.5}$: Generalist Value Model as a Prior for Sparse RL Rollouts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决稀疏奖励下策略梯度不稳定的问题。提出$V_{0.5}$模型，融合先验价值估计与经验均值，降低方差并提升收敛性能。**

- **链接: [https://arxiv.org/pdf/2603.10848](https://arxiv.org/pdf/2603.10848)**

> **作者:** Yi-Kai Zhang; Yueqing Sun; Hongyan Hao; Qi Gu; Xunliang Cai; De-Chuan Zhan; Han-Jia Ye
>
> **摘要:** In Reinforcement Learning with Verifiable Rewards (RLVR), constructing a robust advantage baseline is critical for policy gradients, effectively guiding the policy model to reinforce desired behaviors. Recent research has introduced Generalist Value Models (such as $V_0$), which achieve pre-trained value estimation by explicitly encoding model capabilities in-context, eliminating the need to synchronously update the value model alongside the policy model. In this paper, we propose $V_{0.5}$, which adaptively fuses the baseline predicted by such value model (acting as a prior) with the empirical mean derived from sparse rollouts. This constructs a robust baseline that balances computational efficiency with extremely low variance. Specifically, we introduce a real-time statistical testing and dynamic budget allocation. This balances the high variance caused by sparse sampling against the systematic bias (or hallucinations) inherent in the value model's prior. By constructing a hypothesis test to evaluate the prior's reliability in real-time, the system dynamically allocates additional rollout budget on demand. This mechanism minimizes the baseline estimator's Mean Squared Error (MSE), guaranteeing stable policy gradients, even under extreme sparsity with a group size of 4. Extensive evaluations across six mathematical reasoning benchmarks demonstrate that $V_{0.5}$ significantly outperforms GRPO and DAPO, achieving faster convergence and over some 10% performance improvement.
>
---
#### [new 087] Calibration-Reasoning Framework for Descriptive Speech Quality Assessment
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音质量评估任务，旨在提升描述性质量评估的准确性。通过校准和强化学习方法，增强模型对音频缺陷的识别与定位能力。**

- **链接: [https://arxiv.org/pdf/2603.10175](https://arxiv.org/pdf/2603.10175)**

> **作者:** Elizaveta Kostenok; Mathieu Salzmann; Milos Cernak
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Explainable speech quality assessment requires moving beyond Mean Opinion Scores (MOS) to analyze underlying perceptual dimensions. To address this, we introduce a novel post-training method that tailors the foundational Audio Large Language Model for multidimensional reasoning, detection and classification of audio artifacts. First, a calibration stage aligns the model to predict predefined perceptual dimensions. Second, a reinforcement learning stage leverages Group Relative Policy Optimization (GRPO) with dimension-specific rewards to heavily enhance accuracy of descriptions and temporal localization of quality issues. With this approach we reach state-of-the-art results of 0.71 mean PCC score on the multidimensional QualiSpeech benchmark and 13% improvement in MOS prediction driven by RL-based reasoning. Furthermore, our fine-grained GRPO rewards substantially advance the model's ability to pinpoint and classify audio artifacts in time.
>
---
#### [new 088] Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决跨模态时间对齐问题。提出Daily-Omni基准，评估模型在音频视频联合推理中的表现。**

- **链接: [https://arxiv.org/pdf/2505.17862](https://arxiv.org/pdf/2505.17862)**

> **作者:** Ziwei Zhou; Rui Wang; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) achieve promising performance on visual and audio benchmarks independently. However, the ability of these models to process cross-modal information synchronously remains largely unexplored. We introduce Daily-Omni, a multiple-choice Audio-Visual QA benchmark featuring 684 real-world videos and 1,197 questions spanning 6 task families that explicitly require cross-modal temporal reasoning. To support scalable benchmark construction, we develop a semi-automatic pipeline for annotation, cross-modal consistency refinement, temporal alignment elicitation, and text-only leakage filtering, followed by human verification. We further provide a diagnostic evaluation suite and extensively evaluate 24 foundation models under 37 model--modality settings (Audio+Video / Audio-only / Video-only / Text-only). Finally, we include a training-free modular diagnostic baseline that composes off-the-shelf unimodal models to serve as a diagnostic baseline and to illustrate how explicit temporal alignment signals affect performance. Results indicate that many end-to-end MLLMs still struggle on alignment-critical questions, suggesting that robust cross-modal temporal alignment remains an important open challenge.
>
---
#### [new 089] ADVERSA: Measuring Multi-Turn Guardrail Degradation and Judge Reliability in Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型安全评估任务，旨在解决持续对抗下安全机制退化的问题。通过ADVERSA框架量化安全防线的动态变化，评估模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.10068](https://arxiv.org/pdf/2603.10068)**

> **作者:** Harry Owiredu-Ashley
>
> **备注:** 12 pages, 12 figures. Independent research. Code and artifacts: this https URL
>
> **摘要:** Most adversarial evaluations of large language model (LLM) safety assess single prompts and report binary pass/fail outcomes, which fails to capture how safety properties evolve under sustained adversarial interaction. We present ADVERSA, an automated red-teaming framework that measures guardrail degradation dynamics as continuous per-round compliance trajectories rather than discrete jailbreak events. ADVERSA uses a fine-tuned 70B attacker model (ADVERSA-Red, Llama-3.1-70B-Instruct with QLoRA) that eliminates the attacker-side safety refusals that render off-the-shelf models unreliable as attackers, scoring victim responses on a structured 5-point rubric that treats partial compliance as a distinct measurable state. We report a controlled experiment across three frontier victim models (Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.2) using a triple-judge consensus architecture in which judge reliability is measured as a first-class research outcome rather than assumed. Across 15 conversations of up to 10 adversarial rounds, we observe a 26.7% jailbreak rate with an average jailbreak round of 1.25, suggesting that in this evaluation setting, successful jailbreaks were concentrated in early rounds rather than accumulating through sustained pressure. We document inter-judge agreement rates, self-judge scoring tendencies, attacker drift as a failure mode in fine-tuned attackers deployed out of their training distribution, and attacker refusals as a previously-underreported confound in victim resistance measurement. All limitations are stated explicitly. Attack prompts are withheld per responsible disclosure policy; all other experimental artifacts are released.
>
---
#### [new 090] Speech Codec Probing from Semantic and Phonetic Perspectives
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音编码任务，旨在解决语音表示与文本语义不匹配的问题。通过分析语音分词器的语义和语音内容，发现其主要捕捉语音特征而非语义结构，为下一代语音编码方法提供指导。**

- **链接: [https://arxiv.org/pdf/2603.10371](https://arxiv.org/pdf/2603.10371)**

> **作者:** Xuan Shi; Chang Zeng; Tiantian Feng; Shih-Heng Wang; Jianbo Ma; Shrikanth Narayanan
>
> **摘要:** Speech tokenizers are essential for connecting speech to large language models (LLMs) in multimodal systems. These tokenizers are expected to preserve both semantic and acoustic information for downstream understanding and generation. However, emerging evidence suggests that what is termed "semantic" in speech representations does not align with text-derived semantics: a mismatch that can degrade multimodal LLM performance. In this paper, we systematically analyze the information encoded by several widely used speech tokenizers, disentangling their semantic and phonetic content through word-level probing tasks, layerwise representation analysis, and cross-modal alignment metrics such as CKA. Our results show that current tokenizers primarily capture phonetic rather than lexical-semantic structure, and we derive practical implications for the design of next-generation speech tokenization methods.
>
---
#### [new 091] Improving Search Agent with One Line of Code
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于搜索代理任务，解决TARL训练中的ISDD问题。提出SAPO方法，通过KL约束稳定训练，仅需一行代码修改即可提升性能。**

- **链接: [https://arxiv.org/pdf/2603.10069](https://arxiv.org/pdf/2603.10069)**

> **作者:** Jian Li; Dongsheng Chen; Zhenhua Xu; Yizhang Jin; Jiafu Wu; Chengjie Wang; Xiaotong Yuan; Yabiao Wang
>
> **摘要:** Tool-based Agentic Reinforcement Learning (TARL) has emerged as a promising paradigm for training search agents to interact with external tools for a multi-turn information-seeking process autonomously. However, we identify a critical training instability that leads to catastrophic model collapse: Importance Sampling Distribution Drift(ISDD). In Group Relative Policy Optimization(GRPO), a widely adopted TARL algorithm, ISDD manifests as a precipitous decline in the importance sampling ratios, which nullifies gradient updates and triggers irreversible training failure. To address this, we propose \textbf{S}earch \textbf{A}gent \textbf{P}olicy \textbf{O}ptimization (\textbf{SAPO}), which stabilizes training via a conditional token-level KL constraint. Unlike hard clipping, which ignores distributional divergence, SAPO selectively penalizes the KL divergence between the current and old policies. Crucially, this penalty is applied only to positive tokens with low probabilities where the policy has shifted excessively, thereby preventing distribution drift while preserving gradient flow. Remarkably, SAPO requires only one-line code modification to standard GRPO, ensuring immediate deployability. Extensive experiments across seven QA benchmarks demonstrate that SAPO achieves \textbf{+10.6\% absolute improvement} (+31.5\% relative) over Search-R1, yielding consistent gains across varying model scales (1.5B, 14B) and families (Qwen, LLaMA).
>
---
#### [new 092] Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于编程领域中的核函数合成任务，解决数据稀缺环境下模型性能下降问题。提出EvoKernel框架，通过价值驱动的记忆机制实现从初始生成到持续优化的自动化合成。**

- **链接: [https://arxiv.org/pdf/2603.10846](https://arxiv.org/pdf/2603.10846)**

> **作者:** Yujie Zheng; Zhuo Li; Shengtao Zhang; Hanjing Wang; Junjie Sheng; Jiaqian Wang; Junchi Yan; Weinan Zhang; Ying Wen; Bo Tang; Muning Wen
>
> **摘要:** Deploying Large Language Models to data-scarce programming domains poses significant challenges, particularly for kernel synthesis on emerging Domain-Specific Architectures where a "Data Wall" limits available training data. While models excel on data-rich platforms like CUDA, they suffer catastrophic performance drops on data-scarce ecosystems such as NPU programming. To overcome this cold-start barrier without expensive fine-tuning, we introduce EvoKernel, a self-evolving agentic framework that automates the lifecycle of kernel synthesis from initial drafting to continual refining. EvoKernel addresses this by formulating the synthesis process as a memory-based reinforcement learning task. Through a novel value-driven retrieval mechanism, it learns stage-specific Q-values that prioritize experiences based on their contribution to the current objective, whether bootstrapping a feasible draft or iteratively refining latency. Furthermore, by enabling cross-task memory sharing, the agent generalizes insights from simple to complex operators. By building an NPU variant of KernelBench and evaluating on it, EvoKernel improves frontier models' correctness from 11.0% to 83.0% and achieves a median speedup of 3.60x over initial drafts through iterative refinement. This demonstrates that value-guided experience accumulation allows general-purpose models to master the kernel synthesis task on niche hardware ecosystems. Our official page is available at this https URL.
>
---
#### [new 093] MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于边缘计算任务，解决MoE模型在边缘设备的内存约束问题。通过引入推测激活机制，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2603.09983](https://arxiv.org/pdf/2603.09983)**

> **作者:** Shuhuai Li; Jianghao Lin; Dongdong Ge; Yinyu Ye
>
> **摘要:** Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-information nature of autoregressive expert activation. In this paper, we propose to repurpose Speculative Decoding (SD) not merely as a compute accelerator, but as an informative lookahead sensor for memory management, supported by our theoretical and empirical analyses. Hence, we introduce MoE-SpAc, an MoE inference framework that integrates a Speculative Utility Estimator to track expert demand, a Heterogeneous Workload Balancer to dynamically partition computation via online integer optimization, and an Asynchronous Execution Engine to unify the prefetching and eviction in the same utility space. Extensive experiments on seven benchmarks demonstrate that MoE-SpAc achieves a 42% improvement in TPS over the SOTA SD-based baseline, and an average 4.04x speedup over all standard baselines. Code is available at this https URL .
>
---
#### [new 094] Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI安全评估任务，探讨不同评估条件对模型安全性的测量影响。研究通过大规模实验分析了支架结构对安全指标的影响，发现评估格式是关键变量。**

- **链接: [https://arxiv.org/pdf/2603.10044](https://arxiv.org/pdf/2603.10044)**

> **作者:** David Gringras
>
> **备注:** 74 pages including appendices. 6 frontier models, 62,808 primary observations (~89k total). Pre-registered: OSF DOI https://doi.org/10.17605/OSF.IO/CJW92. Code and data: this https URL
>
> **摘要:** Safety benchmarks evaluate language models in isolation, typically using multiple-choice format; production deployments wrap these models in agentic scaffolds that restructure inputs through reasoning traces, critic agents, and delegation pipelines. We report one of the largest controlled studies of scaffold effects on safety (N = 62,808; six frontier models, four deployment configurations), combining pre-registration, assessor blinding, equivalence testing, and specification curve analysis. Map-reduce scaffolding degrades measured safety (NNH = 14), yet two of three scaffold architectures preserve safety within practically meaningful margins. Investigating the map-reduce degradation revealed a deeper measurement problem: switching from multiple-choice to open-ended format on identical items shifts safety scores by 5-20 percentage points, larger than any scaffold effect. Within-format scaffold comparisons are consistent with practical equivalence under our pre-registered +/-2 pp TOST margin, isolating evaluation format rather than scaffold architecture as the operative variable. Model x scaffold interactions span 35 pp in opposing directions (one model degrades by -16.8 pp on sycophancy under map-reduce while another improves by +18.8 pp on the same benchmark), ruling out universal claims about scaffold safety. A generalisability analysis yields G = 0.000: model safety rankings reverse so completely across benchmarks that no composite safety index achieves non-zero reliability, making per-model, per-configuration testing a necessary minimum standard. We release all code, data, and prompts as ScaffoldSafety.
>
---
#### [new 095] EvoSchema: Towards Text-to-SQL Robustness Against Schema Evolution
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文本到SQL任务，旨在解决数据库模式变化导致模型性能下降的问题。提出EvoSchema基准，评估并提升模型在真实模式变化下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10697](https://arxiv.org/pdf/2603.10697)**

> **作者:** Tianshu Zhang; Kun Qian; Siddhartha Sahai; Yuan Tian; Shaddy Garg; Huan Sun; Yunyao Li
>
> **备注:** Accepted by VLDB 2025
>
> **摘要:** Neural text-to-SQL models, which translate natural language questions (NLQs) into SQL queries given a database schema, have achieved remarkable performance. However, database schemas frequently evolve to meet new requirements. Such schema evolution often leads to performance degradation for models trained on static schemas. Existing work either mainly focuses on simply paraphrasing some syntactic or semantic mappings among NLQ, DB and SQL, or lacks a comprehensive and controllable way to investigate the model robustness issue under the schema evolution, which is insufficient when facing the increasingly complex and rich database schema changes in reality, especially in the LLM era. To address the challenges posed by schema evolution, we present EvoSchema, a comprehensive benchmark designed to assess and enhance the robustness of text-to-SQL systems under real-world schema changes. EvoSchema introduces a novel schema evolution taxonomy, encompassing ten perturbation types across columnlevel and table-level modifications, systematically simulating the dynamic nature of database schemas. Through EvoSchema, we conduct an in-depth evaluation spanning different open source and closed-source LLMs, revealing that table-level perturbations have a significantly greater impact on model performance compared to column-level changes. Furthermore, EvoSchema inspires the development of more resilient text-to-SQL systems, in terms of both model training and database design. The models trained on EvoSchema's diverse schema designs can force the model to distinguish the schema difference for the same questions to avoid learning spurious patterns, which demonstrate remarkable robustness compared to those trained on unperturbed data on average. This benchmark offers valuable insights into model behavior and a path forward for designing systems capable of thriving in dynamic, real-world environments.
>
---
#### [new 096] CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中因忽略中间推理步骤导致的泛化与鲁棒性问题。通过引入对比学习机制（CLIPO），提升模型对正确推理路径的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2603.10101](https://arxiv.org/pdf/2603.10101)**

> **作者:** Sijia Cui; Pengyu Cheng; Jiajun Song; Yongbo Gai; Guojun Zhang; Zhechao Yu; Jianhe Lin; Xiaoxi Jiang; Guanjun Jiang
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capacity of Large Language Models (LLMs). However, RLVR solely relies on final answers as outcome rewards, neglecting the correctness of intermediate reasoning steps. Training on these process-wrong but outcome-correct rollouts can lead to hallucination and answer-copying, severely undermining the model's generalization and robustness. To address this, we incorporate a Contrastive Learning mechanism into the Policy Optimization (CLIPO) to generalize the RLVR process. By optimizing a contrastive loss over successful rollouts, CLIPO steers the LLM to capture the invariant structure shared across correct reasoning paths. This provides a more robust cross-trajectory regularization than the original single-path supervision in RLVR, effectively mitigating step-level reasoning inconsistencies and suppressing hallucinatory artifacts. In experiments, CLIPO consistently improves multiple RLVR baselines across diverse reasoning benchmarks, demonstrating uniform improvements in generalization and robustness for policy optimization of LLMs. Our code and training recipes are available at this https URL.
>
---
#### [new 097] A Systematic Study of Pseudo-Relevance Feedback with LLMs
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，研究伪相关反馈（PRF）中反馈源与反馈模型的作用。通过实验分析两者对PRF效果的影响，提出有效设计建议。**

- **链接: [https://arxiv.org/pdf/2603.11008](https://arxiv.org/pdf/2603.11008)**

> **作者:** Nour Jedidi; Jimmy Lin
>
> **摘要:** Pseudo-relevance feedback (PRF) methods built on large language models (LLMs) can be organized along two key design dimensions: the feedback source, which is where the feedback text is derived from and the feedback model, which is how the given feedback text is used to refine the query representation. However, the independent role that each dimension plays is unclear, as both are often entangled in empirical evaluations. In this paper, we address this gap by systematically studying how the choice of feedback source and feedback model impact PRF effectiveness through controlled experimentation. Across 13 low-resource BEIR tasks with five LLM PRF methods, our results show: (1) the choice of feedback model can play a critical role in PRF effectiveness; (2) feedback derived solely from LLM-generated text provides the most cost-effective solution; and (3) feedback derived from the corpus is most beneficial when utilizing candidate documents from a strong first-stage retriever. Together, our findings provide a better understanding of which elements in the PRF design space are most important.
>
---
#### [new 098] IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于增强大模型指令层次的任务，旨在解决指令冲突时的优先级问题。通过构建IH-Challenge数据集，提升模型在对抗性场景下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10521](https://arxiv.org/pdf/2603.10521)**

> **作者:** Chuan Guo; Juan Felipe Ceron Uribe; Sicheng Zhu; Christopher A. Choquette-Choo; Steph Lin; Nikhil Kandpal; Milad Nasr; Sam Toyer; Miles Wang; Yaodong Yu; Alex Beutel; Kai Xiao
>
> **摘要:** Instruction hierarchy (IH) defines how LLMs prioritize system, developer, user, and tool instructions under conflict, providing a concrete, trust-ordered policy for resolving instruction conflicts. IH is key to defending against jailbreaks, system prompt extractions, and agentic prompt injections. However, robust IH behavior is difficult to train: IH failures can be confounded with instruction-following failures, conflicts can be nuanced, and models can learn shortcuts such as overrefusing. We introduce IH-Challenge, a reinforcement learning training dataset, to address these difficulties. Fine-tuning GPT-5-Mini on IH-Challenge with online adversarial example generation improves IH robustness by +10.0% on average across 16 in-distribution, out-of-distribution, and human red-teaming benchmarks (84.1% to 94.1%), reduces unsafe behavior from 6.6% to 0.7% while improving helpfulness on general safety evaluations, and saturates an internal static agentic prompt injection evaluation, with minimal capability regression. We release the IH-Challenge dataset (this https URL) to support future research on robust instruction hierarchy.
>
---
#### [new 099] Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM中的长度膨胀问题。提出GR$^3$方法，通过乘法归一化实现无损长度控制，有效减少冗余输出。**

- **链接: [https://arxiv.org/pdf/2603.10535](https://arxiv.org/pdf/2603.10535)**

> **作者:** Zichao Li; Jie Lou; Fangchen Dong; Zhiyuan Fan; Mengjie Ren; Hongyu Lin; Xianpei Han; Debing Zhang; Le Sun; Yaojie Lu; Xing Yu
>
> **摘要:** Reinforcement learning significantly enhances LLM capabilities but suffers from a critical issue: length inflation, where models adopt verbosity or inefficient reasoning to maximize rewards. Prior approaches struggle to address this challenge in a general and lossless manner, primarily because additive penalties introduce a compensatory effect that creates optimization shortcuts, while heuristic gating strategies lack generality beyond binary feedback. To bridge this gap, we present Group Relative Reward Rescaling (GR$^3$), which reframes length control as a multiplicative rescaling paradigm, effectively establishing a generalized, continuous, and reward-dependent gating mechanism. To further ensure lossless optimization, we incorporate group-relative regularization and advantage-aware calibration, which dynamically adapt length budgets to instance difficulty and preserve the advantage signal of high-quality trajectories. Empirically, across both RLHF and RLVR settings, GR$^3$~maintains training dynamics and downstream performance comparable to standard GRPO while significantly mitigating length inflation, outperforming state-of-the-art length-regularized baselines.
>
---
#### [new 100] TOSSS: a CVE-based Software Security Benchmark for Large Language Models
- **分类: cs.LG; cs.CL; cs.CR; cs.SE**

- **简介: 该论文提出TOSSS基准，用于评估大语言模型在代码安全选择上的能力，解决LLM在软件安全中的表现问题。**

- **链接: [https://arxiv.org/pdf/2603.10969](https://arxiv.org/pdf/2603.10969)**

> **作者:** Marc Damie; Murat Bilgehan Ertan; Domenico Essoussi; Angela Makhanu; Gaëtan Peter; Roos Wensveen
>
> **摘要:** With their increasing capabilities, Large Language Models (LLMs) are now used across many industries. They have become useful tools for software engineers and support a wide range of development tasks. As LLMs are increasingly used in software development workflows, a critical question arises: are LLMs good at software security? At the same time, organizations worldwide invest heavily in cybersecurity to reduce exposure to disruptive attacks. The integration of LLMs into software engineering workflows may introduce new vulnerabilities and weaken existing security efforts. We introduce TOSSS (Two-Option Secure Snippet Selection), a benchmark that measures the ability of LLMs to choose between secure and vulnerable code snippets. Existing security benchmarks for LLMs cover only a limited range of vulnerabilities. In contrast, TOSSS relies on the CVE database and provides an extensible framework that can integrate newly disclosed vulnerabilities over time. Our benchmark gives each model a security score between 0 and 1 based on its behavior; a score of 1 indicates that the model always selects the secure snippet, while a score of 0 indicates that it always selects the vulnerable one. We evaluate 14 widely used open-source and closed-source models on C/C++ and Java code and observe scores ranging from 0.48 to 0.89. LLM providers already publish many benchmark scores for their models, and TOSSS could become a complementary security-focused score to include in these reports.
>
---
#### [new 101] Personalized Group Relative Policy Optimization for Heterogenous Preference Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM与多样化用户偏好对齐的问题。提出P-GRPO框架，通过个性化优势估计提升模型对不同偏好的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.10009](https://arxiv.org/pdf/2603.10009)**

> **作者:** Jialu Wang; Heinrich Peters; Asad A. Butt; Navid Hashemi; Alireza Hashemi; Pouya M. Ghari; Joseph Hoover; James Rae; Morteza Dehghani
>
> **摘要:** Despite their sophisticated general-purpose capabilities, Large Language Models (LLMs) often fail to align with diverse individual preferences because standard post-training methods, like Reinforcement Learning with Human Feedback (RLHF), optimize for a single, global objective. While Group Relative Policy Optimization (GRPO) is a widely adopted on-policy reinforcement learning framework, its group-based normalization implicitly assumes that all samples are exchangeable, inheriting this limitation in personalized settings. This assumption conflates distinct user reward distributions and systematically biases learning toward dominant preferences while suppressing minority signals. To address this, we introduce Personalized GRPO (P-GRPO), a novel alignment framework that decouples advantage estimation from immediate batch statistics. By normalizing advantages against preference-group-specific reward histories rather than the concurrent generation group, P-GRPO preserves the contrastive signal necessary for learning distinct preferences. We evaluate P-GRPO across diverse tasks and find that it consistently achieves faster convergence and higher rewards than standard GRPO, thereby enhancing its ability to recover and align with heterogeneous preference signals. Our results demonstrate that accounting for reward heterogeneity at the optimization level is essential for building models that faithfully align with diverse human preferences without sacrificing general capabilities.
>
---
#### [new 102] Explainable LLM Unlearning Through Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于LLM可解释性任务，解决模型遗忘过程中能力退化问题。提出基于推理的遗忘目标和TRU方法，实现精准知识删除并保持模型能力。**

- **链接: [https://arxiv.org/pdf/2603.09980](https://arxiv.org/pdf/2603.09980)**

> **作者:** Junfeng Liao; Qizhou Wang; Shanshan Ye; Xin Yu; Ling Chen; Zhen Fang
>
> **摘要:** LLM unlearning is essential for mitigating safety, copyright, and privacy concerns in pre-trained large language models (LLMs). Compared to preference alignment, it offers a more explicit way by removing undesirable knowledge characterized by specific unlearning datasets. In previous works, gradient ascent (GA) and its variants have shown promise for implementing unlearning, yet their untargeted nature results in unintended degradation of general capabilities, incomplete removal of knowledge, and the generation of incoherent responses, among many others. We argue that these issues stem from the absence of explicit guidance on what and how models should unlearn. To fill this gap, we introduce a novel unlearning target, reasoning-based unlearning target, which satisfies both the specified unlearning scope and the specified post-unlearning response. Building on this, we propose targeted reasoning unlearning (TRU), which leverages reasoning-based unlearning target as guidance. We employ the target using a cross-entropy supervised loss combined with a GA-based loss, enabling the model to learn reasoning ability for precise knowledge removal while preserving unrelated abilities. We evaluate TRU against strong baselines across multiple benchmarks and LLM backbones, and find that it achieves more reliable unlearning while preserving general capabilities. Moreover, TRU exhibits superior robustness under diverse attack scenarios, stemming from the reasoning ability learned through reasoning-based targets. Overall, our study establishes reasoning-augmented unlearning as a practical paradigm for reliable and explainable LLM unlearning.
>
---
#### [new 103] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于时间序列模型的可解释性研究，旨在揭示TSFMs内部表示的因果结构。通过稀疏自编码器分析，发现模型依赖突变检测而非周期模式。**

- **链接: [https://arxiv.org/pdf/2603.10071](https://arxiv.org/pdf/2603.10071)**

> **作者:** Anurag Mishra
>
> **备注:** Accepted as a poster in ICLR 2026 Workshop on Time Series in the Age of Large Models (TSALM)
>
> **摘要:** Time series foundation models (TSFMs) are increasingly deployed in high-stakes domains, yet their internal representations remain opaque. We present the first application of sparse autoencoders (SAEs) to a TSFM, training TopK SAEs on activations of Chronos-T5-Large (710M parameters) across six layers. Through 392 single-feature ablation experiments, we establish that every ablated feature produces a positive CRPS degradation, confirming causal relevance. Our analysis reveals a depth-dependent hierarchy: early encoder layers encode low-level frequency features, the mid-encoder concentrates causally critical change-detection features, and the final encoder compresses a rich but less causally important taxonomy of temporal concepts. The most critical features reside in the mid-encoder (max single-feature Delta CRPS = 38.61), not in the semantically richest final encoder layer, where progressive ablation paradoxically improves forecast quality. These findings demonstrate that mechanistic interpretability transfers effectively to TSFMs and that Chronos-T5 relies on abrupt-dynamics detection rather than periodic pattern recognition.
>
---
#### [new 104] Emulating Clinician Cognition via Self-Evolving Deep Clinical Research
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于临床诊断任务，旨在解决AI系统与真实诊疗认知不匹配的问题。提出DxEvolve框架，通过自我进化提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2603.10677](https://arxiv.org/pdf/2603.10677)**

> **作者:** Ruiyang Ren; Yuhao Wang; Yunsen Liang; Lan Luo; Jing Liu; Haifeng Wang; Cong Feng; Yinan Zhang; Chunyan Miao; Ji-Rong Wen; Wayne Xin Zhao
>
> **摘要:** Clinical diagnosis is a complex cognitive process, grounded in dynamic cue acquisition and continuous expertise accumulation. Yet most current artificial intelligence (AI) systems are misaligned with this reality, treating diagnosis as single-pass retrospective prediction while lacking auditable mechanisms for governed improvement. We developed DxEvolve, a self-evolving diagnostic agent that bridges these gaps through an interactive deep clinical research workflow. The framework autonomously requisitions examinations and continually externalizes clinical experience from increasing encounter exposure as diagnostic cognition primitives. On the MIMIC-CDM benchmark, DxEvolve improved diagnostic accuracy by 11.2% on average over backbone models and reached 90.4% on a reader-study subset, comparable to the clinician reference (88.8%). DxEvolve improved accuracy on an independent external cohort by 10.2% (categories covered by the source cohort) and 17.1% (uncovered categories) compared to the competitive method. By transforming experience into a governable learning asset, DxEvolve supports an accountable pathway for the continual evolution of clinical AI.
>
---
## 更新

#### [replaced 001] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人任务规划领域，解决模糊人类指令影响规划性能的问题。通过构建REI-Bench基准并提出上下文认知方法，提升非专家用户指令的处理效果。**

- **链接: [https://arxiv.org/pdf/2505.10872](https://arxiv.org/pdf/2505.10872)**

> **作者:** Chenxi Jiang; Chuhao Zhou; Jianfei Yang
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who are the groups that robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark that systematically models vague REs grounded in pragmatic theory (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 36.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompts, chains of thought, and in-context learning. By tackling the overlooked issue of vagueness, this work contributes to the research community by advancing real-world task planning and making robots more accessible to non-expert users, e.g., the elderly and children.
>
---
#### [replaced 002] PsihoRo: Depression and Anxiety Romanian Text Corpus
- **分类: cs.CL**

- **简介: 该论文属于心理NLP任务，旨在解决罗马尼亚语心理健康语料缺失的问题。通过收集205份问卷数据，构建了首个罗马尼亚语抑郁与焦虑语料库PsihoRo，并进行统计与文本分析。**

- **链接: [https://arxiv.org/pdf/2602.18324](https://arxiv.org/pdf/2602.18324)**

> **作者:** Alexandra Ciobotaru; Ana-Maria Bucur; Liviu P. Dinu
>
> **备注:** This article was accepted at LREC 2026
>
> **摘要:** Psychological corpora in NLP are collections of texts used to analyze human psychology, emotions, and mental health. These texts allow researchers to study psychological constructs, detect mental health issues and analyze emotional language. However, mental health data can be difficult to collect correctly from social media, due to suppositions made by the collectors. A more pragmatic strategy involves gathering data through open-ended questions and then assessing this information with self-report screening surveys. This method was employed successfully for English, a language with a lot of psychological NLP resources. However, this cannot be stated for Romanian, which currently has no open-source mental health corpus. To address this gap, we have created the first corpus for depression and anxiety in Romanian, by utilizing a form with 6 open-ended questions along with the standardized PHQ-9 and GAD-7 screening questionnaires. Consisting of the texts of 205 respondents and although it may seem small, PsihoRo is a first step towards understanding and analyzing texts regarding the mental health of the Romanian population. We employ statistical analysis, text analysis using Romanian LIWC, emotion detection and topic modeling to show what are the most important features of this newly introduced resource to the NLP community.
>
---
#### [replaced 003] Mindstorms in Natural Language-Based Societies of Mind
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **简介: 该论文探讨自然语言驱动的多智能体系统（NLSOM），解决复杂AI任务。通过模块化设计提升多模态推理能力，实验验证其在视觉、生成等任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2305.17066](https://arxiv.org/pdf/2305.17066)**

> **作者:** Mingchen Zhuge; Haozhe Liu; Francesco Faccio; Dylan R. Ashley; Róbert Csordás; Anand Gopalakrishnan; Abdullah Hamdi; Hasan Abed Al Kader Hammoud; Vincent Herrmann; Kazuki Irie; Louis Kirsch; Bing Li; Guohao Li; Shuming Liu; Jinjie Mai; Piotr Piękos; Aditya Ramesh; Imanol Schlag; Weimin Shi; Aleksandar Stanić; Wenyi Wang; Yuhui Wang; Mengmeng Xu; Deng-Ping Fan; Bernard Ghanem; Jürgen Schmidhuber
>
> **备注:** published in Computational Visual Media Journal (CVMJ); 9 pages in main text + 7 pages of references + 38 pages of appendices, 14 figures in main text + 13 in appendices, 7 tables in appendices
>
> **摘要:** Both Minsky's "society of mind" and Schmidhuber's "learning to think" inspire diverse societies of large multimodal neural networks (NNs) that solve problems by interviewing each other in a "mindstorm." Recent implementations of NN-based societies of minds consist of large language models (LLMs) and other NN-based experts communicating through a natural language interface. In doing so, they overcome the limitations of single LLMs, improving multimodal zero-shot reasoning. In these natural language-based societies of mind (NLSOMs), new agents -- all communicating through the same universal symbolic language -- are easily added in a modular fashion. To demonstrate the power of NLSOMs, we assemble and experiment with several of them (having up to 129 members), leveraging mindstorms in them to solve some practical AI tasks: visual question answering, image captioning, text-to-image synthesis, 3D generation, egocentric retrieval, embodied AI, and general language-based task solving. We view this as a starting point towards much larger NLSOMs with billions of agents-some of which may be humans. And with this emergence of great societies of heterogeneous minds, many new research questions have suddenly become paramount to the future of artificial intelligence. What should be the social structure of an NLSOM? What would be the (dis)advantages of having a monarchical rather than a democratic structure? How can principles of NN economies be used to maximize the total reward of a reinforcement learning NLSOM? In this work, we identify, discuss, and try to answer some of these questions.
>
---
#### [replaced 004] ThinkPatterns-21k: A Systematic Study on the Impact of Thinking Patterns in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM中不同思维模式对性能的影响。通过构建ThinkPatterns-21k数据集，分析多种思维模式的效果，发现结构化思维对小模型有益，而大模型则可能受损。**

- **链接: [https://arxiv.org/pdf/2503.12918](https://arxiv.org/pdf/2503.12918)**

> **作者:** Pengcheng Wen; Jiaming Ji; Chi-Min Chan; Juntao Dai; Donghai Hong; Yaodong Yang; Sirui Han; Yike Guo
>
> **摘要:** Large language models (LLMs) have demonstrated enhanced performance through the \textit{Thinking then Responding} paradigm, where models generate internal thoughts before final responses (aka, System 2 thinking). However, existing research lacks a systematic understanding of the mechanisms underlying how thinking patterns affect performance across model sizes. In this work, we conduct a comprehensive analysis of the impact of various thinking types on model performance and introduce ThinkPatterns-21k, a curated dataset comprising 21k instruction-response pairs (QA) collected from existing instruction-following datasets with five thinking types. For each pair, we augment it with five distinct internal thinking patterns: one unstructured thinking (monologue) and four structured variants (decomposition, self-ask, self-debate and self-critic), while maintaining the same instruction and response. Through extensive evaluation across different model sizes (3B-32B parameters), we have two key findings: (1) smaller models (<30B parameters) can benefit from most of structured thinking patterns, while larger models (32B) with structured thinking like decomposition would degrade performance and (2) unstructured monologue demonstrates broad effectiveness across different model sizes. Finally, we released all of our datasets, checkpoints, training logs of diverse thinking patterns to reproducibility, aiming to facilitate further research in this direction.
>
---
#### [replaced 005] KV Cache Transform Coding for Compact Storage in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型推理优化任务，解决KV缓存占用内存过高的问题。提出KVTC方法，通过压缩技术实现高效存储，提升内存利用率。**

- **链接: [https://arxiv.org/pdf/2511.01815](https://arxiv.org/pdf/2511.01815)**

> **作者:** Konrad Staniszewski; Adrian Łańcucki
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy, and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks including AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper and RULER. It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches.
>
---
#### [replaced 006] LaTeX Compilation: Challenges in the Era of LLMs
- **分类: cs.CL**

- **简介: 论文探讨了TeX在LLM时代的局限性，提出Mogan STEM作为替代方案，解决编译效率和用户体验问题。**

- **链接: [https://arxiv.org/pdf/2603.02873](https://arxiv.org/pdf/2603.02873)**

> **作者:** Tianyou Liu; Ziqiang Li; Xurui Liu; Yu Wu; Yansong Li
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. Furthermore, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format.
>
---
#### [replaced 007] Get away with less: Need of source side data curation to build parallel corpus for low resource Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决低资源语言数据不足的问题。通过源端数据筛选框架LALITA，提升平行语料质量，减少训练数据需求并提高翻译效果。**

- **链接: [https://arxiv.org/pdf/2601.08629](https://arxiv.org/pdf/2601.08629)**

> **作者:** Saumitra Yadav; Manish Shrivastava
>
> **备注:** Under Review
>
> **摘要:** Data curation is a critical yet under-researched step in the machine translation training paradigm. To train translation systems, data acquisition relies primarily on human translations and digital parallel sources or, to a limited degree, synthetic generation. But, for low-resource languages, human translation to generate sufficient data is prohibitively expensive. Therefore, it is crucial to develop a framework that screens source sentences to form efficient parallel text, ensuring optimal MT system performance in low-resource environments. We approach this by evaluating English-Hindi bi-text to determine effective sentence selection strategies for optimal MT system training. Our extensively tested framework, (Lexical And Linguistically Informed Text Analysis) LALITA, targets source sentence selection using lexical and linguistic features to curate parallel corpora. We find that by training mostly on complex sentences from both existing and synthetic datasets, our method significantly improves translation quality. We test this by simulating low-resource data availabilty with curated datasets of 50K to 800K English sentences and report improved performances on all data sizes. LALITA demonstrates remarkable efficiency, reducing data needs by more than half across multiple languages (Hindi, Odia, Nepali, Norwegian Nynorsk, and German). This approach not only reduces MT systems training cost by reducing training data requirement, but also showcases LALITA's utility in data augmentation.
>
---
#### [replaced 008] Large Language Model Psychometrics: A Systematic Review of Evaluation, Validation, and Enhancement
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于人工智能评估任务，旨在解决传统方法难以衡量LLM心理特质的问题。通过系统综述，提出LLM心理测量学框架，整合心理学理论与方法，提升模型评估与优化。**

- **链接: [https://arxiv.org/pdf/2505.08245](https://arxiv.org/pdf/2505.08245)**

> **作者:** Haoran Ye; Jing Jin; Yuhang Xie; Xin Zhang; Guojie Song
>
> **备注:** 400+ references
>
> **摘要:** The advancement of large language models (LLMs) has outpaced traditional evaluation methodologies. This progress presents novel challenges, such as measuring human-like psychological constructs, moving beyond static and task-specific benchmarks, and establishing human-centered evaluation. These challenges intersect with psychometrics, the science of quantifying the intangible aspects of human psychology, such as personality, values, and intelligence. This review paper introduces and synthesizes the emerging interdisciplinary field of LLM Psychometrics, which leverages psychometric instruments, theories, and principles to evaluate, understand, and enhance LLMs. The reviewed literature systematically shapes benchmarking principles, broadens evaluation scopes, refines methodologies, validates results, and advances LLM capabilities. Diverse perspectives are integrated to provide a structured framework for researchers across disciplines, enabling a more comprehensive understanding of this nascent field. Ultimately, the review provides actionable insights for developing future evaluation paradigms that align with human-level AI and promote the advancement of human-centered AI systems for societal benefit. A curated repository of LLM psychometric resources is available at this https URL.
>
---
#### [replaced 009] CEFR-Annotated WordNet: LLM-Based Proficiency-Guided Semantic Database for Language Learning
- **分类: cs.CL**

- **简介: 该论文属于语言教育与自然语言处理交叉任务，旨在解决二语学习者理解WordNet细粒度语义的困难。通过LLM将WordNet与CEFR标注结合，构建语义数据库以辅助语言学习。**

- **链接: [https://arxiv.org/pdf/2510.18466](https://arxiv.org/pdf/2510.18466)**

> **作者:** Masato Kikuchi; Masatsugu Ono; Toshioki Soga; Tetsu Tanabe; Tadachika Ozono
>
> **备注:** The 15th edition of the Language Resources and Evaluation Conference (LREC 2026); resources are available at this https URL
>
> **摘要:** Although WordNet is a valuable resource because of its structured semantic networks and extensive vocabulary, its fine-grained sense distinctions can be challenging for second-language learners. To address this issue, we developed a version of WordNet annotated with the Common European Framework of Reference for Languages (CEFR), integrating its semantic networks with language-proficiency levels. We automated this process using a large language model to measure the semantic similarity between sense definitions in WordNet and entries in the English Vocabulary Profile Online. To validate our approach, we constructed a large-scale corpus containing both sense and CEFR-level information from the annotated WordNet and used it to develop contextual lexical classifiers. Our experiments demonstrate that models fine-tuned on this corpus perform comparably to those fine-tuned on gold-standard annotations. Furthermore, by combining this corpus with the gold-standard data, we developed a practical classifier that achieves a Macro-F1 score of 0.81. This result provides indirect evidence that the transferred labels are largely consistent with the gold-standard levels. The annotated WordNet, corpus, and classifiers are publicly available to help bridge the gap between natural language processing and language education, thereby facilitating more effective and efficient language learning.
>
---
#### [replaced 010] EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EoRA，解决压缩大语言模型后的精度下降问题，通过低秩矩阵增强，无需微调即可提升性能。**

- **链接: [https://arxiv.org/pdf/2410.21271](https://arxiv.org/pdf/2410.21271)**

> **作者:** Shih-Yang Liu; Maksim Khadkevich; Nai Chit Fung; Charbel Sakr; Chao-Han Huck Yang; Chien-Yi Wang; Saurav Muralidharan; Hongxu Yin; Kwang-Ting Cheng; Jan Kautz; Yu-Chiang Frank Wang; Pavlo Molchanov; Min-Hung Chen
>
> **备注:** ICLR 2026 workshops. Code: this https URL
>
> **摘要:** While post-training compression techniques effectively reduce the memory footprint, latency, and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA, a novel fine-tuning-free method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior training-free low rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., $\mathbf{10.84\%}$ on ARC-Challenge, $\mathbf{6.74\%}$ on MathQA, and $\mathbf{11.45\%}$ on GSM8K) for LLaMA3-8B compressed to 3-bit. We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy of compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at this https URL.
>
---
#### [replaced 011] LLLMs: A Data-Driven Survey of Evolving Research on Limitations of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文献综述任务，旨在分析大语言模型的局限性研究趋势。通过数据驱动方法，梳理2022至2025年的相关论文，揭示研究热点与变化。**

- **链接: [https://arxiv.org/pdf/2505.19240](https://arxiv.org/pdf/2505.19240)**

> **作者:** Aida Kostikova; Zhipin Wang; Deidamea Bajri; Ole Pütz; Benjamin Paaßen; Steffen Eger
>
> **备注:** ACM Computing Surveys (CSUR); 56 pages
>
> **摘要:** Large language model (LLM) research has grown rapidly, along with increasing concern about their limitations. In this survey, we conduct a data-driven, semi-automated review of research on limitations of LLMs (LLLMs) from 2022 to early 2025 using a bottom-up approach. From a corpus of 250,000 ACL and arXiv papers, we identify 14,648 relevant papers using keyword filtering, LLM-based classification, validated against expert labels, and topic clustering (via two approaches, HDBSCAN+BERTopic and LlooM). We find that the share of LLM-related papers increases over fivefold in ACL and nearly eightfold in arXiv between 2022 and 2025. Since 2022, LLLMs research grows even faster, reaching over 30% of LLM papers by 2025. Reasoning remains the most studied limitation, followed by generalization, hallucination, bias, and security. The distribution of topics in the ACL dataset stays relatively stable over time, while arXiv shifts toward security risks, alignment, hallucinations, knowledge editing, and multimodality. We offer a quantitative view of trends in LLLMs research and release a dataset of annotated abstracts and a validated methodology, available at: this https URL.
>
---
#### [replaced 012] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing
- **分类: cs.LG; cs.AI; cs.CL; cs.DS; cs.IT**

- **简介: 该论文研究语言模型的幻觉现象，将其视为成员检测问题，通过信息论分析指出幻觉是存储优化的必然结果。任务属于自然语言处理中的模型可靠性研究。**

- **链接: [https://arxiv.org/pdf/2602.00906](https://arxiv.org/pdf/2602.00906)**

> **作者:** Anxin Guo; Jingwei Li
>
> **摘要:** Large language models often hallucinate with high confidence on "random facts" that lack inferable patterns. We formalize the memorization of such facts as a membership testing problem, unifying the discrete error metrics of Bloom filters with the continuous log-loss of LLMs. By analyzing this problem in the regime where facts are sparse in the universe of plausible claims, we establish a rate-distortion theorem: the optimal memory efficiency is characterized by the minimum KL divergence between score distributions on facts and non-facts. This theoretical framework provides a distinctive explanation for hallucination: even with optimal training, perfect data, and a simplified "closed world" setting, the information-theoretically optimal strategy under limited capacity is not to abstain or forget, but to assign high confidence to some non-facts, resulting in hallucination. We validate this theory empirically on synthetic data, showing that hallucinations persist as a natural consequence of lossy compression.
>
---
#### [replaced 013] Evaluation of LLMs in retrieving food and nutritional context for RAG systems
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在评估LLMs在RAG系统中获取食品营养数据的能力，解决如何高效转换自然语言查询为结构化过滤器的问题。**

- **链接: [https://arxiv.org/pdf/2603.09704](https://arxiv.org/pdf/2603.09704)**

> **作者:** Maks Požarnik Vavken; Matevž Ogrinc; Tome Eftimov; Barbara Koroušić Seljak
>
> **备注:** This is the preprint for our conference paper for IEEE International Conference on Big Data
>
> **摘要:** In this article, we evaluate four Large Language Models (LLMs) and their effectiveness at retrieving data within a specialized Retrieval-Augmented Generation (RAG) system, using a comprehensive food composition database. Our method is focused on the LLMs ability to translate natural language queries into structured metadata filters, enabling efficient retrieval via a Chroma vector database. By achieving high accuracy in this critical retrieval step, we demonstrate that LLMs can serve as an accessible, high-performance tool, drastically reducing the manual effort and technical expertise previously required for domain experts, such as food compilers and nutritionists, to leverage complex food and nutrition data. However, despite the high performance on easy and moderately complex queries, our analysis of difficult questions reveals that reliable retrieval remains challenging when queries involve non-expressible constraints. These findings demonstrate that LLM-driven metadata filtering excels when constraints can be explicitly expressed, but struggles when queries exceed the representational scope of the metadata format.
>
---
#### [replaced 014] RACAS: Controlling Diverse Robots With a Single Agentic System
- **分类: cs.RO; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出RACAS系统，解决跨平台机器人控制问题。通过自然语言交互的模块实现通用控制，无需修改代码或模型。**

- **链接: [https://arxiv.org/pdf/2603.05621](https://arxiv.org/pdf/2603.05621)**

> **作者:** Dylan R. Ashley; Jan Przepióra; Yimeng Chen; Ali Abualsaud; Nurzhan Yesmagambet; Shinkyu Park; Eric Feron; Jürgen Schmidhuber
>
> **备注:** 7 pages in main text + 1 page of appendices + 1 page of references, 5 figures in main text + 1 figure in appendices, 2 tables in main text; source code available at this https URL
>
> **摘要:** Many robotic platforms expose an API through which external software can command their actuators and read their sensors. However, transitioning from these low-level interfaces to high-level autonomous behaviour requires a complicated pipeline, whose components demand distinct areas of expertise. Existing approaches to bridging this gap either require retraining for every new embodiment or have only been validated across structurally similar platforms. We introduce RACAS (Robot-Agnostic Control via Agentic Systems), a cooperative agentic architecture in which three LLM/VLM-based modules (Monitors, a Controller, and a Memory Curator) communicate exclusively through natural language to provide closed-loop robot control. RACAS requires only a natural language description of the robot, a definition of available actions, and a task specification; no source code, model weights, or reward functions need to be modified to move between platforms. We evaluate RACAS on several tasks using a wheeled ground robot, a recently published novel multi-jointed robotic limb, and an underwater vehicle. RACAS consistently solved all assigned tasks across these radically different platforms, demonstrating the potential of agentic AI to substantially reduce the barrier to prototyping robotic solutions.
>
---
#### [replaced 015] Explainability of Text Processing and Retrieval Methods: A Survey
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升文本处理与检索模型的可解释性。研究梳理了现有解释方法，涵盖词向量、序列模型、注意力机制等，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2212.07126](https://arxiv.org/pdf/2212.07126)**

> **作者:** Sourav Saha; Debapriyo Majumdar; Mandar Mitra
>
> **备注:** To appear in ACM Computing Surveys
>
> **摘要:** Deep Learning and Machine Learning based models have become extremely popular in text processing and information retrieval. However, the non-linear structures present inside the networks make these models largely inscrutable. A significant body of research has focused on increasing the transparency of these models. This article provides a broad overview of research on the explainability and interpretability of natural language processing and information retrieval methods. More specifically, we survey approaches that have been applied to explain word embeddings, sequence modeling, attention modules, transformers, BERT, and document ranking. The concluding section suggests some possible directions for future research on this topic.
>
---
#### [replaced 016] BiasCause: Evaluate Socially Biased Causal Reasoning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型偏见分析任务，旨在解决LLMs在因果推理中的社会偏见问题。通过构建测试集和分析因果图，评估模型的偏见来源及应对策略。**

- **链接: [https://arxiv.org/pdf/2504.07997](https://arxiv.org/pdf/2504.07997)**

> **作者:** Tian Xie; Tongxin Yin; Vaishakh Keshava; Xueru Zhang; Siddhartha Reddy Jonnalagadda
>
> **备注:** This work has been done when the first author is at Google. The first author is a student at the Ohio State University
>
> **摘要:** While large language models (LLMs) play increasingly significant roles in society, research shows they continue to generate content that reflects social bias against sensitive groups. Existing benchmarks effectively identify these biases, but a critical gap remains in understanding the underlying reasoning processes that produce them. This paper addresses this gap by evaluating the causal reasoning of LLMs when answering socially biased questions. We propose a formal schema that categorizes causal reasoning into three types (mistaken, biased, and contextually-grounded). We then synthesize 1788 questions covering eight sensitive attributes, with each set of questions designed to probe a specific type of causal reasoning. All questions are then manually validated, and each of them prompts the LLM to generate a causal graph behind its answer. We evaluate four state-of-the-art LLMs and find that all models exhibit biased causal reasoning on most questions eliciting it. Moreover, we discover that LLMs are also prone to "mistaken-biased" reasoning, where they first confuse correlation with causality to infer sensitive group membership and subsequently apply biased causal reasoning. By examining the cases where LLMs produce unbiased causal reasoning, we also identify three strategies LLMs employ to avoid bias (i.e., explicitly refusing to answer, avoiding sensitive attributes, and adding contextual restrictions), which provide insights for future debiasing efforts.
>
---
#### [replaced 017] MultiGraSCCo: A Multilingual Anonymization Benchmark with Annotations of Personal Identifiers
- **分类: cs.CL**

- **简介: 该论文提出MultiGraSCCo基准，解决多语言隐私数据匿名化问题。通过机器翻译生成带标注的合成数据，确保个人信息正确转换，支持安全数据共享与系统测试。**

- **链接: [https://arxiv.org/pdf/2603.08879](https://arxiv.org/pdf/2603.08879)**

> **作者:** Ibrahim Baroud; Christoph Otto; Vera Czehmann; Christine Hovhannisyan; Lisa Raithel; Sebastian Möller; Roland Roller
>
> **备注:** Accepted at the International Conference on Language Resources and Evaluation (LREC2026)
>
> **摘要:** Accessing sensitive patient data for machine learning is challenging due to privacy concerns. Datasets with annotations of personally identifiable information are crucial for developing and testing anonymization systems to enable safe data sharing that complies with privacy regulations. Since accessing real patient data is a bottleneck, synthetic data offers an efficient solution for data scarcity, bypassing privacy regulations that apply to real data. Moreover, neural machine translation can help to create high-quality data for low-resource languages by translating validated real or synthetic data from a high-resource language. In this work, we create a multilingual anonymization benchmark in ten languages, using a machine translation methodology that preserves the original annotations and renders names of cities and people in a culturally and contextually appropriate form in each target language. Our evaluation study with medical professionals confirms the quality of the translations, both in general and with respect to the translation and adaptation of personal information. Our benchmark with over 2,500 annotations of personal information can be used in many applications, including training annotators, validating annotations across institutions without legal complications, and helping improve the performance of automatic personal information detection. We make our benchmark and annotation guidelines available for further research.
>
---
#### [replaced 018] No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数据污染检测任务，旨在解决小语言模型中数据污染的识别问题。通过分析输出分布，发现CDD方法效果有限，提出概率方法更有效。**

- **链接: [https://arxiv.org/pdf/2603.03203](https://arxiv.org/pdf/2603.03203)**

> **作者:** Omer Sela
>
> **备注:** Code available at this https URL
>
> **摘要:** CDD, or Contamination Detection via output Distribution, identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models ranging from 70M to 410M parameters. Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD's effectiveness depends critically on whether fine-tuning produces verbatim memorization. In the majority of conditions we test, CDD performs at chance level even when the data is verifiably contaminated and detectable by simpler methods. We show that probability-based methods, specifically perplexity and Min-k\% Prob, outperform CDD in all conditions where any method exceeds chance, suggesting that CDD's peakedness-based approach is insufficient for contamination detection in small language models. Our code is available at this https URL
>
---
#### [replaced 019] PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration
- **分类: cs.CV; cs.AI; cs.CL; cs.DL; cs.IR**

- **简介: 该论文提出PathoScribe，解决病理数据难以检索与利用的问题。通过统一的LLM框架，实现病例检索、智能分析和临床整合，提升诊断效率。**

- **链接: [https://arxiv.org/pdf/2603.08935](https://arxiv.org/pdf/2603.08935)**

> **作者:** Abdul Rehman Akbar; Samuel Wales-McGrath; Alejadro Levya; Lina Gokhale; Rajendra Singh; Wei Chen; Anil Parwani; Muhammad Khalid Khan Niazi
>
> **摘要:** Pathology underpins modern diagnosis and cancer care, yet its most valuable asset, the accumulated experience encoded in millions of narrative reports, remains largely inaccessible. Although institutions are rapidly digitizing pathology workflows, storing data without effective mechanisms for retrieval and reasoning risks transforming archives into a passive data repository, where institutional knowledge exists but cannot meaningfully inform patient care. True progress requires not only digitization, but the ability for pathologists to interrogate prior similar cases in real time while evaluating a new diagnostic dilemma. We present PathoScribe, a unified retrieval-augmented large language model (LLM) framework designed to transform static pathology archives into a searchable, reasoning-enabled living library. PathoScribe enables natural language case exploration, automated cohort construction, clinical question answering, immunohistochemistry (IHC) panel recommendation, and prompt-controlled report transformation within a single architecture. Evaluated on 70,000 multi-institutional surgical pathology reports, PathoScribe achieved perfect Recall@10 for natural language case retrieval and demonstrated high-quality retrieval-grounded reasoning (mean reviewer score 4.56/5). Critically, the system operationalized automated cohort construction from free-text eligibility criteria, assembling research-ready cohorts in minutes (mean 9.2 minutes) with 91.3% agreement to human reviewers and no eligible cases incorrectly excluded, representing orders-of-magnitude reductions in time and cost compared to traditional manual chart review. This work establishes a scalable foundation for converting digital pathology archives from passive storage systems into active clinical intelligence platforms.
>
---
#### [replaced 020] Large Language Models for Travel Behavior Prediction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于旅行行为预测任务，旨在利用大语言模型提升预测效果。通过两种框架探索LLMs在该领域的应用，实现数据高效且灵活的预测方法。**

- **链接: [https://arxiv.org/pdf/2312.00819](https://arxiv.org/pdf/2312.00819)**

> **作者:** Baichuan Mo; Hanyong Xu; Ruoyun Ma; Jung-Hoon Cho; Dingyi Zhuang; Xiaotong Guo; Jinhua Zhao
>
> **摘要:** Travel behavior prediction is a core problem in transportation demand management and is traditionally addressed using numerical models calibrated on observed data. With recent advances in large language models (LLMs), new opportunities have emerged to model human decision-making through natural language reasoning. This study explores the use of LLMs for travel behavior prediction through two complementary frameworks. The first framework employs a zero-shot prompting strategy, where the prediction task, traveler attributes, and relevant domain knowledge are described in text, enabling the LLM to directly generate predictions without task-specific training data. The second framework uses LLM-generated text embeddings as high-level representations of travel scenarios, which are then combined with conventional supervised learning models to support prediction in small-sample settings. Empirical results show that both approaches achieve performance comparable to, and in some cases competitive with, classical models such as multinomial logit, random forest, and neural networks. These findings suggest that LLMs offer a flexible and data-efficient alternative for travel behavior prediction.
>
---
#### [replaced 021] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态学习领域，旨在分析多模态数据中不同模态间的依赖关系。研究通过实验证明，现有基准在模态交互上存在偏差，提出量化方法以改进多模态基准设计。**

- **链接: [https://arxiv.org/pdf/2509.23499](https://arxiv.org/pdf/2509.23499)**

> **作者:** Divyam Madaan; Varshan Muhunthan; Kyunghyun Cho; Sumit Chopra
>
> **备注:** Accepted to ICLR 2026. Code available at this https URL
>
> **摘要:** Understanding the interplay between intra-modality dependencies (the contribution of an individual modality to a target task) and inter-modality dependencies (the relationships between modalities and the target task) is fundamental to advancing multi-modal learning. However, the nature of and interaction between these dependencies within current benchmark evaluations remains poorly characterized. In this work, we present a large-scale empirical study to quantify these dependencies across 23 visual question-answering benchmarks using multi-modal large language models (MLLMs) covering domains such as general and expert knowledge reasoning, optical character recognition, and document understanding. Our findings show that the reliance on vision, question (text), and their interaction varies significantly, both across and within benchmarks. We discover that numerous benchmarks intended to mitigate text-only biases have inadvertently amplified image-only dependencies. This characterization persists across model sizes and types, with models often obtaining high performance by using each modality independently and showing limited dependence on their interaction. We provide a quantitative characterization of multi-modal datasets, enabling a principled approach to multi-modal benchmark design and evaluation.
>
---
#### [replaced 022] How Large Language Models Get Stuck: Early structure with persistent errors
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM在训练中出现的错误固化现象，针对语法判断任务，分析模型为何在部分情况下无法正确区分语法与不语法句子，并提出大词组假设进行验证。**

- **链接: [https://arxiv.org/pdf/2603.00359](https://arxiv.org/pdf/2603.00359)**

> **作者:** Alokesh Manna; William Snyder; Whitney Tabor
>
> **摘要:** Linguistic insights may help make Large Language Model (LLM) training more efficient. We trained Meta's OPT model on the 100M word BabyLM dataset, and evaluated it on the BLiMP benchmark, which consists of 67 classes, each defined by sentence pairs that differ in a targeted syntactic or semantic rule violation. We tested the model's preference for grammatical over ungrammatical sentences across training iterations and grammatical types. In nearly one-third of the BLiMP classes, OPT fails to consistently assign a higher likelihood to grammatical sentences, even after extensive training. When it fails, it often establishes a clear (erroneous) separation of the likelihoods at an early stage of processing and sustains this to the end of our training phase. We hypothesize that this mis-categorization is costly because it creates entrenched biases that must, eventually, be reversed in order for the model to perform well. We probe this phenomenon using a mixture of qualitative (based on linguistic theory and the theory of Deep Learning) and quantitative (based on numerical testing) assessments. Our qualitative assessments indicate that only some BLiMP tests are meaningful guides. We conclude by articulating a hypothesis, the Bigram Hypothesis, which claims that the learning process will exhibit erroneous entrenchment if bigram statistics bias the model toward wrong distinctions early in training, and we describe a method of testing the hypothesis on appropriately selected BLiMP classes.
>
---
#### [replaced 023] Fusing Semantic, Lexical, and Domain Perspectives for Recipe Similarity Estimation
- **分类: cs.CL**

- **简介: 该论文属于 recipe similarity estimation 任务，旨在通过融合语义、词汇和领域信息评估食谱相似性。工作包括方法设计、专家验证及影响因素分析。**

- **链接: [https://arxiv.org/pdf/2603.09688](https://arxiv.org/pdf/2603.09688)**

> **作者:** Denica Kjorvezir; Danilo Najkov; Eva Valencič; Erika Jesenko; Barbara Koroišić Seljak; Tome Eftimov; Riste Stojanov
>
> **备注:** Preprint version submitted to IEEE Big Data 2025
>
> **摘要:** This research focuses on developing advanced methods for assessing similarity between recipes by combining different sources of information and analytical approaches. We explore the semantic, lexical, and domain similarity of food recipes, evaluated through the analysis of ingredients, preparation methods, and nutritional attributes. A web-based interface was developed to allow domain experts to validate the combined similarity results. After evaluating 318 recipe pairs, experts agreed on 255 (80%). The evaluation of expert assessments enables the estimation of which similarity aspects--lexical, semantic, or nutritional--are most influential in expert decision-making. The application of these methods has broad implications in the food industry and supports the development of personalized diets, nutrition recommendations, and automated recipe generation systems.
>
---
#### [replaced 024] Goal Hijacking Attack on Large Language Models via Pseudo-Conversation Injection
- **分类: cs.CL**

- **简介: 该论文属于安全任务，旨在解决LLM被劫持生成特定输出的问题。通过构造伪对话实现目标劫持，提出三种攻击策略，提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2410.23678](https://arxiv.org/pdf/2410.23678)**

> **作者:** Zheng Chen; Buhui Yao
>
> **备注:** Accepted by the 2025 IEEE 24th International Conference on Trust, Security and Privacy in Computing and Communications (IEEE TrustCom 2025)
>
> **摘要:** Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.
>
---
#### [replaced 025] AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents
- **分类: cs.HC; cs.CL**

- **简介: 论文提出AgentA/B系统，利用LLM代理模拟用户行为进行A/B测试，解决传统测试依赖真实流量和耗时问题。**

- **链接: [https://arxiv.org/pdf/2504.09723](https://arxiv.org/pdf/2504.09723)**

> **作者:** Yuxuan Lu; Ting-Yao Hsu; Hansu Gu; Limeng Cui; Yaochen Xie; William Headden; Bingsheng Yao; Akash Veeragouni; Jiapeng Liu; Sreyashi Nag; Jessie Wang; Dakuo Wang
>
> **摘要:** A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents this http URL, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
>
---
#### [replaced 026] AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth
- **分类: cs.CL**

- **简介: 该论文提出AdaPonderLM，解决语言模型推理时计算资源分配不均的问题。通过自适应迭代机制，实现按需计算，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.01914](https://arxiv.org/pdf/2603.01914)**

> **作者:** Shixiang Song; He Li; Zitong Wang; Boyi Zeng; Feichen Song; Yixuan Wang; Zhiqin John Xu; Ziwei He; Zhouhan Lin
>
> **摘要:** Test-time scaling via recurrent/iterative Transformers enables large language models to spend more computation at inference, but most pretrained recurrent LMs run a fixed number of iterations, wasting compute on easy tokens and lacking token-wise adaptivity. Following the core idea of Adaptive Computation Time(ACT) and Early Exit(EE), we propose AdaPonderLM, a self-supervised recurrent language model that learns token-wise early exiting during pretraining without manually tuned per-token/per-layer pruning ratios. AdaPonderLM uses iteration-specific MLP gates with a monotonic halting mask to decide when each token stops recurring, and introduces a KV reuse mechanism that reuses cached key/value states for halted tokens, ensuring train--test consistency and practical acceleration. Across Pythia backbones from 70M to 410M (pretraining) and up to 2.8B (continued pretraining), AdaPonderLM reduces inference compute at about 10% while maintaining comparable language modeling perplexity and competitive downstream accuracy. Our analysis shows the learned gates allocate more computation to high-NLL (hard) tokens, exhibiting adaptive computation time behavior in a fully self-supervised setting. Meanwhile, under iso-FLOPs, the learned halting policy consistently outperforms fixed pruning, showing AdaPonderLM allocates compute to the right tokens rather than just reducing average depth.
>
---
#### [replaced 027] Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决CoT推理的高延迟问题。通过V-Skip方法，结合语言和视觉信息优化token压缩，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2601.13879](https://arxiv.org/pdf/2601.13879)**

> **作者:** Dongxu Zhang; Yiding Sun; Cheng Tan; Wenbiao Yan; Ning Yang; Jihua Zhu; Haijun Zhang
>
> **摘要:** While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA.
>
---
#### [replaced 028] Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors
- **分类: cs.CL**

- **简介: 该论文属于语言模型上下文压缩任务，旨在解决传统方法依赖自编码训练导致的性能冲突问题。提出SAC方法，通过选择锚点令牌直接压缩上下文，提升实际应用效果。**

- **链接: [https://arxiv.org/pdf/2510.08907](https://arxiv.org/pdf/2510.08907)**

> **作者:** Xin Liu; Runsong Zhao; Pengcheng Huang; Xinyu Liu; Junyi Xiao; Chunyang Xiao; Tong Xiao; Shengxiang Gao; Zhengtao Yu; Jingbo Zhu
>
> **备注:** 23 pages,10 figures
>
> **摘要:** Context compression is an advanced technique that accelerates large language model (LLM) inference by converting long inputs into compact representations. Existing methods primarily rely on autoencoding tasks to train special compression tokens to represent contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, we remark that such capabilities potentially conflict with actual downstream task requirements, prevent the models from learning the features more beneficial for real-world usage. Based on this observation, we propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. To ensure that anchors can effectively collect information, SAC introduces two key designs: (1) anchor embedding, a learnable embedding vector attached to the selected anchor tokens to mark compression carriers and (2) bidirectional attention modification, which enables anchor tokens to integrate information from the entire context. Experimental results show that SAC consistently outperforms existing context compression methods across different compression ratios and model sizes on question-answering and long-context summarization tasks. Our data, model and code have been released at \href{this https URL}{this https URL}.
>
---
#### [replaced 029] Training with Pseudo-Code for Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于指令遵循任务，旨在解决LLM难以准确执行复杂指令的问题。通过在训练中引入伪代码增强数据，提升模型指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2505.18011](https://arxiv.org/pdf/2505.18011)**

> **作者:** Prince Kumar; Rudra Murthy; Riyaz Bhat; Danish Contractor
>
> **备注:** Under Review
>
> **摘要:** Despite rapid advances in the capabilities of Large Language Models (LLMs), they continue to struggle with following relatively simple and unambiguous instructions, particularly when compositional structure is involved. Recent work suggests that models may follow instructions more effectively when they are expressed in pseudo-code rather than natural language. However, writing pseudo-code programs can be tedious, and relying on few-shot demonstrations or inference-time code prompting is often unnatural for non-expert users of LLMs. To overcome these limitations, we propose a training time approach that fine-tunes LLMs using instruction-tuning data augmented with pseudo-code representations of natural language instructions paired with final responses. We evaluate our method on 12 publicly available benchmarks spanning instruction-following, mathematical reasoning, and commonsense reasoning, across six base models. Our results show that models trained with pseudo-code follow instructions more reliably, achieving relative gains of 8-21\% on instruction following benchmarks, while largely preserving and in some cases improving performance on mathematical and commonsense reasoning tasks, with an average gain of up to 30\% across all evaluated benchmarks.
>
---
#### [replaced 030] LaTeXTrans: Structured LaTeX Translation with Multi-Agent Coordination
- **分类: cs.CL**

- **简介: 该论文提出LaTeXTrans，解决结构化LaTeX文档翻译问题，通过多智能体协作实现准确翻译与格式保持。**

- **链接: [https://arxiv.org/pdf/2508.18791](https://arxiv.org/pdf/2508.18791)**

> **作者:** Ziming Zhu; Chenglong Wang; Haosong Xv; Shunjie Xing; Yifu Huo; Fengning Tian; Quan Du; Di Yang; Chunliang Zhang; Tong Xiao; Jingbo Zhu
>
> **摘要:** Despite the remarkable progress of modern machine translation (MT) systems on general-domain texts, translating structured LaTeX-formatted documents remains a significant challenge. These documents typically interleave natural language with domain-specific syntax, such as mathematical equations, tables, figures, and cross-references, all of which must be accurately preserved to maintain semantic integrity and compilability. In this paper, we introduce LaTeXTrans, a collaborative multi-agent system designed to address this challenge. LaTeXTrans ensures format preservation, structural fidelity, and terminology consistency through six specialized agents: 1) a Parser that decomposes LaTeX into translation-friendly units via placeholder substitution and syntax filtering; 2) a Translator, Validator, Summarizer, and Terminology Extractor that work collaboratively to ensure context-aware, self-correcting, and terminology-consistent translations; 3) a Generator that reconstructs the translated content into well-structured LaTeX documents. Experimental results show that LaTeXTrans outperforms mainstream MT systems in both translation accuracy and structural preservation. The source code, the online demonstration platform, and a demo video are publicly available.
>
---
#### [replaced 031] Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决多参与方协作对话中的长期记忆问题。研究构建了EverMemBench基准，评估记忆系统的多维能力，揭示现有系统的局限性。**

- **链接: [https://arxiv.org/pdf/2602.01313](https://arxiv.org/pdf/2602.01313)**

> **作者:** Chuanrui Hu; Tong Li; Xingze Gao; Hongda Chen; Yi Bai; Dannong Xu; Tianwei Lin; Xiaohong Li; Yunyun Han; Jian Pei; Yafeng Deng
>
> **备注:** 25 pages, 21 figures, 10 tables
>
> **摘要:** Long-term conversational memory in practical LLM applications is inherently collaborative: information is produced by multiple participants, scattered across groups and channels, revised over time, and implicitly grounded in roles and social context. Yet there is currently no established benchmark that evaluates memory under interaction patterns resembling real-world deployment, as existing benchmarks largely focus on dyadic or single-topic dialogues. In this paper, we introduce EverMemBench, the first benchmark designed for long-horizon collaborative memory, built from multi-party, multi-group conversations spanning over one million tokens with dense cross-topic interleaving, temporally evolving decisions, and role-conditioned personas. EverMemBench evaluates memory systems using 2400 QA pairs across three dimensions essential for real applications: fine-grained recall, memory awareness, and user profile understanding. Our evaluation reveals fundamental limitations of current systems: multi-hop reasoning collapses under multi-party attribution even with oracle evidence (26% accuracy), temporal reasoning fails without explicit version semantics beyond timestamps, and memory awareness is bottlenecked by retrieval, as similarity-based methods miss implicitly relevant information. EverMemBench thus represents a concrete step toward realistic evaluation of LLM memory and a cornerstone benchmark for developing next-generation LLMs that reason over time, roles, and collaborative interaction structure. Our benchmark and code are publicly available at this https URL.
>
---
#### [replaced 032] Fish Audio S2 Technical Report
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文介绍Fish Audio S2，一个开源文本到语音系统，解决多说话人、多轮生成及指令跟随问题。通过多阶段训练和数据管道提升性能，并提供可部署的推理引擎。**

- **链接: [https://arxiv.org/pdf/2603.08823](https://arxiv.org/pdf/2603.08823)**

> **作者:** Shijia Liao; Yuxuan Wang; Songting Liu; Yifan Cheng; Ruoyi Zhang; Tianyu Li; Shidong Li; Yisheng Zheng; Xingwei Liu; Qingzheng Wang; Zhizhuo Zhou; Jiahua Liu; Xin Chen; Dawei Han
>
> **摘要:** We introduce Fish Audio S2, an open-sourced text-to-speech system featuring multi-speaker, multi-turn generation, and, most importantly, instruction-following control via natural-language descriptions. To scale training, we develop a multi-stage training recipe together with a staged data pipeline covering video captioning and speech captioning, voice-quality assessment, and reward modeling. To push the frontier of open-source TTS, we release our model weights, fine-tuning code, and an SGLang-based inference engine. The inference engine is production-ready for streaming, achieving an RTF of 0.195 and a time-to-first-audio below 100 this http URL code and weights are available on GitHub (this https URL) and Hugging Face (this https URL). We highly encourage readers to visit this https URL to try custom voices.
>
---
#### [replaced 033] Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私安全任务，旨在解决LLM推理中KV缓存的隐私泄露问题。通过分析攻击向量并提出KV-Cloak防御机制，有效保护用户输入隐私。**

- **链接: [https://arxiv.org/pdf/2508.09442](https://arxiv.org/pdf/2508.09442)**

> **作者:** Zhifan Luo; Shuo Shao; Su Zhang; Lijing Zhou; Yuke Hu; Chenxu Zhao; Zhihao Liu; Zhan Qin
>
> **备注:** This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026. Code: this https URL
>
> **摘要:** The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
>
---
#### [replaced 034] QCSE: A Pretrained Quantum Context-Sensitive Word Embedding for Natural Language Processing
- **分类: cs.CL**

- **简介: 该论文提出QCSE模型，解决自然语言处理中的上下文敏感词嵌入问题，利用量子计算提升语言表示能力。**

- **链接: [https://arxiv.org/pdf/2509.05729](https://arxiv.org/pdf/2509.05729)**

> **作者:** Charles M. Varmantchaonala; Niclas Götting; Nils-Erik Schütte; Jean Louis E. K. Fendji; Christopher Gies
>
> **摘要:** Quantum Natural Language Processing (QNLP) offers a novel approach to encoding and understanding the complexity of natural languages through the power of quantum computation. This paper presents a pretrained quantum context-sensitive embedding model, called QCSE, that captures context-sensitive word embeddings, leveraging the unique properties of quantum systems to learn contextual relationships in languages. The model introduces quantum-native context learning, enabling the utilization of quantum computers for linguistic tasks. Central to the proposed approach are innovative context matrix computation methods, designed to create unique, representations of words based on their surrounding linguistic context. Five distinct methods are proposed and tested for computing the context matrices, incorporating techniques such as exponential decay, sinusoidal modulation, phase shifts, and hash-based transformations. These methods ensure that the quantum embeddings retain context sensitivity, thereby making them suitable for downstream language tasks where the expressibility and properties of quantum systems are valuable resources. To evaluate the effectiveness of the model and the associated context matrix methods, evaluations are conducted on both a Fulani corpus, a low-resource African language, dataset of small size and an English corpus of slightly larger size. The results demonstrate that QCSE not only captures context sensitivity but also leverages the expressibility of quantum systems for representing rich, context-aware language information. The use of Fulani further highlights the potential of QNLP to mitigate the problem of lack of data for this category of languages. This work underscores the power of quantum computation in natural language processing (NLP) and opens new avenues for applying QNLP to real-world linguistic challenges across various tasks and domains.
>
---
#### [replaced 035] Adaptive Loops and Memory in Transformers: Think Harder or Know More?
- **分类: cs.CL**

- **简介: 该论文研究改进Transformer模型，通过自适应循环和记忆机制提升推理能力。任务为语言模型推理优化，解决参数效率与性能平衡问题，结合循环和记忆增强数学与常识推理表现。**

- **链接: [https://arxiv.org/pdf/2603.08391](https://arxiv.org/pdf/2603.08391)**

> **作者:** Markus Frey; Behzad Shomali; Ali Hamza Bashir; David Berghaus; Joachim Koehler; Mehdi Ali
>
> **备注:** Published at Latent & Implicit Thinking Workshop @ ICLR 2026
>
> **摘要:** Chain-of-thought (CoT) prompting enables reasoning in language models but requires explicit verbalization of intermediate steps. Looped transformers offer an alternative by iteratively refining representations within hidden states. This parameter efficiency comes at a cost, as looped models lack the storage capacity of deeper models which use unique weights per layer. In this work, we investigate transformer models that feature both adaptive per-layer looping, where each transformer block learns to iterate its hidden state via a learned halting mechanism, and gated memory banks, that provide additional learned storage. We find that looping primarily benefits mathematical reasoning, while memory banks help recover performance on commonsense tasks compared to parameter and FLOP matched models. Combining both mechanisms yields a model that outperforms an iso-FLOP baseline, with three times the number of layers, across math benchmarks. Analysis of model internals reveals layer specialization: early layers learn to loop minimally and access memory sparingly, while later layers do both more heavily.
>
---
#### [replaced 036] Modelling Language using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言建模任务，探讨大语言模型作为语言科学模型的价值，解决语言研究中是否具备语言洞察力的问题，通过构建模型解释框架进行论证。**

- **链接: [https://arxiv.org/pdf/2404.09579](https://arxiv.org/pdf/2404.09579)**

> **作者:** Jumbly Grindrod
>
> **备注:** Philosophical Studies (2026)
>
> **摘要:** This paper argues that large language models have a valuable scientific role to play in serving as scientific models of public languages. Linguistic study should not only be concerned with the cognitive processes behind linguistic competence, but also with language understood as an external, social entity. Once this is recognized, the value of large language models as scientific models becomes clear. This paper defends the position against a number of arguments to the effect that language models provide no linguistic insight. Building upon Weisberg's (2007) notion of a model construal, it is then argued that recent work in computational linguistics to better understand the inner workings of large language models can be used to develop a model construal for large language models as models of a language.
>
---
#### [replaced 037] SAGE: A Top-Down Bottom-Up Knowledge-Grounded User Simulator for Multi-turn AGent Evaluation
- **分类: cs.CL**

- **简介: 该论文属于多轮对话系统评估任务，旨在解决传统人工评估效率低的问题。提出SAGE框架，结合业务知识生成更真实的用户交互，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2510.11997](https://arxiv.org/pdf/2510.11997)**

> **作者:** Ryan Shea; Yunan Lu; Liang Qiu; Zhou Yu
>
> **摘要:** Evaluating multi-turn interactive agents is challenging due to the need for human assessment. Evaluation with simulated users has been introduced as an alternative, however existing approaches typically model generic users and overlook the domain-specific principles required to capture realistic behavior. We propose SAGE, a novel user Simulation framework for multi-turn AGent Evaluation that integrates knowledge from business contexts. SAGE incorporates top-down knowledge rooted in business logic, such as ideal customer profiles, grounding user behavior in realistic customer personas. We further integrate bottom-up knowledge taken from business agent infrastructure (e.g., product catalogs, FAQs, and knowledge bases), allowing the simulator to generate interactions that reflect users' information needs and expectations in a company's target market. Through empirical evaluation, we find that this approach produces interactions that are more realistic and diverse, while also identifying up to 33% more agent errors, highlighting its effectiveness as an evaluation tool to support bug-finding and iterative agent improvement.
>
---
#### [replaced 038] Word length predicts word order: "Min-max"-ing drives language evolution
- **分类: cs.CL**

- **简介: 该论文研究语言演变中的语序变化，提出基于“最小化努力、最大化信息”的普遍理论，通过分析1942个语料库数据，揭示词长与语序的关系，解决语序变化的统一解释问题。**

- **链接: [https://arxiv.org/pdf/2505.13913](https://arxiv.org/pdf/2505.13913)**

> **作者:** Hiram Ring
>
> **摘要:** A fundamental concern in linguistics has been to understand how languages change, such as in relation to word order. Since the order of words in a sentence (i.e. the relative placement of Subject, Object, and Verb) is readily identifiable in most languages, this has been a productive field of study for decades (see Greenberg 1963; Dryer 2007; Hawkins 2014). However, a language's word order can change over time, with competing explanations for such changes (Carnie and Guilfoyle 2000; Crisma and Longobardi 2009; Martins and Cardoso 2018; Dunn et al. 2011; Jager and Wahle 2021). This paper proposes a general universal explanation for word order change based on a theory of communicative interaction (the Min-Max theory of language behavior) in which agents seek to minimize effort while maximizing information. Such an account unifies opposing findings from language processing (Piantadosi et al. 2011; Wasow 2022; Levy 2008) that make different predictions about how word order should be realized crosslinguistically. The marriage of both "efficiency" and "surprisal" approaches under the Min-Max theory is justified with evidence from a massive dataset of 1,942 language corpora tagged for parts of speech (Ring 2025), in which average lengths of particular word classes correlates with word order, allowing for prediction of basic word order from diverse corpora. The general universal pressure of word class length in corpora is shown to give a stronger explanation for word order realization than either genealogical or areal factors, highlighting the importance of language corpora for investigating such questions.
>
---
#### [replaced 039] Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型监督微调任务，解决数据质量低的问题。通过细粒度筛选冗余或有害的token，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2502.01968](https://arxiv.org/pdf/2502.01968)**

> **作者:** Jinlong Pang; Na Di; Zhaowei Zhu; Jiaheng Wei; Hao Cheng; Chen Qian; Yang Liu
>
> **摘要:** Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant, uninformative, or even harmful. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves downstream performance. Code is available at this https URL.
>
---
#### [replaced 040] Computational modeling of early language learning from acoustic speech and audiovisual input without linguistic priors
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语言习得研究任务，旨在探讨如何通过计算模型理解婴儿从语音和视听输入中学习语言的过程，解决无语言先验条件下的语言习得问题。工作包括回顾自监督和视觉引导的模型进展。**

- **链接: [https://arxiv.org/pdf/2603.08359](https://arxiv.org/pdf/2603.08359)**

> **作者:** Okko Räsänen
>
> **摘要:** Learning to understand speech appears almost effortless for typically developing infants, yet from an information-processing perspective, acquiring a language from acoustic speech is an enormous challenge. This chapter reviews recent developments in using computational models to understand early language acquisition from speech and audiovisual input. The focus is on self-supervised and visually grounded models of perceptual learning. We show how these models are becoming increasingly powerful in learning various aspects of speech without strong linguistic priors, and how many features of early language development can be explained through a shared set of learning principles-principles broadly compatible with multiple theories of language acquisition and human cognition. We also discuss how modern learning simulations are gradually becoming more realistic, both in terms of input data and in linking model behavior to empirical findings on infant language development.
>
---
#### [replaced 041] Tracking Cancer Through Text: Longitudinal Extraction From Radiology Reports Using Open-Source Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医学文本信息提取任务，旨在从放射科报告中自动提取肿瘤信息。通过开源大模型实现隐私保护的纵向数据提取，解决临床文本结构化难题。**

- **链接: [https://arxiv.org/pdf/2603.09638](https://arxiv.org/pdf/2603.09638)**

> **作者:** Luc Builtjes; Alessa Hering
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Radiology reports capture crucial longitudinal information on tumor burden, treatment response, and disease progression, yet their unstructured narrative format complicates automated analysis. While large language models (LLMs) have advanced clinical text processing, most state-of-the-art systems remain proprietary, limiting their applicability in privacy-sensitive healthcare environments. We present a fully open-source, locally deployable pipeline for longitudinal information extraction from radiology reports, implemented using the llm_extractinator framework. The system applies the qwen2.5-72b model to extract and link target, non-target, and new lesion data across time points in accordance with RECIST criteria. Evaluation on 50 Dutch CT Thorax/Abdomen report pairs yielded high extraction performance, with attribute-level accuracies of 93.7% for target lesions, 94.9% for non-target lesions, and 94.0% for new lesions. The approach demonstrates that open-source LLMs can achieve clinically meaningful performance in multi-timepoint oncology tasks while ensuring data privacy and reproducibility. These results highlight the potential of locally deployable LLMs for scalable extraction of structured longitudinal data from routine clinical text.
>
---
#### [replaced 042] Assessing the Political Fairness of Multilingual LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset
- **分类: cs.CL**

- **简介: 该论文属于多语言模型政治公平性评估任务，旨在检测LLMs在翻译中的政治偏见。通过分析21语种的EuroParl数据集，发现主流政党演讲翻译质量更高。**

- **链接: [https://arxiv.org/pdf/2510.20508](https://arxiv.org/pdf/2510.20508)**

> **作者:** Paul Lerner; François Yvon
>
> **备注:** Accepted at LREC 2026. Added results with new models and two-ANOVA. Same conclusions
>
> **摘要:** The political biases of Large Language Models (LLMs) are usually assessed by simulating their answers to English surveys. In this work, we propose an alternative framing of political biases, relying on principles of fairness in multilingual translation. We systematically compare the translation quality of speeches in the European Parliament (EP), observing systematic differences with majority parties from left and right being better translated than outsider parties. This study is made possible by a new, 21-way multiparallel version of EuroParl, the parliamentary proceedings of the EP, which includes the political affiliations of each speaker. The dataset consists of 1.5M sentences for a total of 40M words and 249M characters. It covers three years, 1000+ speakers, 7 countries, 12 EU parties, 25 EU committees, and hundreds of national parties.
>
---
#### [replaced 043] Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长上下文推理中的提示压缩问题。通过跨家族草稿模型实现无需训练的提示压缩，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2603.02631](https://arxiv.org/pdf/2603.02631)**

> **作者:** Shubhangi Upasani; Ravi Shanker Raju; Bo Li; Mengmeng Ji; John Long; Chen Wu; Urmish Thakker; Guangtao Wang
>
> **摘要:** Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative prefill demonstrates that attention-based token importance estimation can enable training-free prompt compression, but this assumes the existence of a draft model that shares the same tokenizer as the target model. In practice, however, agentic pipelines frequently employ models without any smaller in-family draft model. In this work, we study cross-family speculative prefill, where a lightweight draft model from one model family is used to perform prompt compression for a target model from a different family. Using the same speculative prefill mechanism as prior work, we evaluate a range of cross-family draft-target combinations, including Qwen, LLaMA, and DeepSeek models. Across a broad diversity of tasks, we find that attention-based token importance estimation transfers reliably across different model families despite differences in model architectures and tokenizers between draft and target models. Cross-model prompt compression largely retains 90~100% of full-prompt baseline performance and, in some cases, slightly improves accuracy due to denoising effects, while delivering substantial reductions in time to first token (TTFT). These results suggest that speculative prefill depends mainly on task priors and semantic structure, thus serving as a generalizable prompt compression primitive. We discuss the implications of our findings for agentic systems, where repeated long-context inference and heterogeneous model stacks make cross-model prompt compression both necessary and practical.
>
---
#### [replaced 044] AutoPCR: Automated Phenotype Concept Recognition by Prompting
- **分类: cs.CL**

- **简介: 该论文属于生物医学文本挖掘任务，旨在解决表型概念识别问题。提出AutoPCR方法，无需特定本体训练，通过提示技术实现高效、通用的表型识别。**

- **链接: [https://arxiv.org/pdf/2507.19315](https://arxiv.org/pdf/2507.19315)**

> **作者:** Yicheng Tao; Yuanhao Huang; Yiqun Wang; Xin Luo; Jie Liu
>
> **摘要:** Phenotype concept recognition (CR) is a fundamental task in biomedical text mining, enabling applications such as clinical diagnostics and knowledge graph construction. However, existing methods often require ontology-specific training and struggle to generalize across diverse text types and evolving biomedical terminology. We present AutoPCR, a prompt-based phenotype CR method that does not require ontology-specific training. AutoPCR performs CR in three stages: entity extraction using a hybrid of rule-based and neural tagging strategies, candidate retrieval via SapBERT, and entity linking through prompting a large language model. Experiments on four benchmark datasets show that AutoPCR achieves the best average and most robust performance across both mention-level and document-level evaluations, surpassing prior state-of-the-art methods. Further ablation and transfer studies demonstrate its inductive capability and generalizability to new ontologies.
>
---
