# 自然语言处理 cs.CL

- **最新发布 88 篇**

- **更新 85 篇**

## 最新发布

#### [new 001] NeoDictaBERT: Pushing the Frontier of BERT models for Hebrew
- **分类: cs.CL**

- **简介: 该论文提出NeoDictaBERT及bilingual版本，针对希伯来语NLP任务，采用更新的Transformer架构，提升模型性能与上下文长度。解决了传统BERT在希伯来语上表现不足的问题，显著优于现有模型，尤其在检索任务中表现突出。**

- **链接: [http://arxiv.org/pdf/2510.20386v1](http://arxiv.org/pdf/2510.20386v1)**

> **作者:** Shaltiel Shmidman; Avi Shmidman; Moshe Koppel
>
> **摘要:** Since their initial release, BERT models have demonstrated exceptional performance on a variety of tasks, despite their relatively small size (BERT-base has ~100M parameters). Nevertheless, the architectural choices used in these models are outdated compared to newer transformer-based models such as Llama3 and Qwen3. In recent months, several architectures have been proposed to close this gap. ModernBERT and NeoBERT both show strong improvements on English benchmarks and significantly extend the supported context window. Following their successes, we introduce NeoDictaBERT and NeoDictaBERT-bilingual: BERT-style models trained using the same architecture as NeoBERT, with a dedicated focus on Hebrew texts. These models outperform existing ones on almost all Hebrew benchmarks and provide a strong foundation for downstream tasks. Notably, the NeoDictaBERT-bilingual model shows strong results on retrieval tasks, outperforming other multilingual models of similar size. In this paper, we describe the training process and report results across various benchmarks. We release the models to the community as part of our goal to advance research and development in Hebrew NLP.
>
---
#### [new 002] Assessing the Political Fairness of Multilingual LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型在政治翻译中的公平性问题，针对主流政党话语被更准确翻译的现象。基于21语种的多平行欧共体议会数据集，系统比较不同政治派别演讲的翻译质量，揭示了模型对主流政党的偏见。任务为评估多语言LLM的政治公平性，核心是检测翻译偏差。**

- **链接: [http://arxiv.org/pdf/2510.20508v1](http://arxiv.org/pdf/2510.20508v1)**

> **作者:** Paul Lerner; François Yvon
>
> **摘要:** The political biases of Large Language Models (LLMs) are usually assessed by simulating their answers to English surveys. In this work, we propose an alternative framing of political biases, relying on principles of fairness in multilingual translation. We systematically compare the translation quality of speeches in the European Parliament (EP), observing systematic differences with majority parties from left, center, and right being better translated than outsider parties. This study is made possible by a new, 21-way multiparallel version of EuroParl, the parliamentary proceedings of the EP, which includes the political affiliations of each speaker. The dataset consists of 1.5M sentences for a total of 40M words and 249M characters. It covers three years, 1000+ speakers, 7 countries, 12 EU parties, 25 EU committees, and hundreds of national parties.
>
---
#### [new 003] Exploring Generative Process Reward Modeling for Semi-Structured Data: A Case Study of Table Question Answering
- **分类: cs.CL**

- **简介: 该论文研究生成式过程奖励模型（PRM）在表格问答（TQA）任务中的应用。针对半结构化数据中推理步骤松散、领域依赖性强等问题，首次系统评估了生成式PRM在答案与步骤层面的表现，发现结合文本与代码验证的PRM虽能提升选择效果，但泛化能力弱，且步骤验证与答案准确率相关性低。**

- **链接: [http://arxiv.org/pdf/2510.20304v1](http://arxiv.org/pdf/2510.20304v1)**

> **作者:** Lei Tang; Wei Zhou; Mohsen Mesgar
>
> **摘要:** Process reward models (PRMs) improve complex reasoning in large language models (LLMs) by grading candidate solutions step-by-step and selecting answers via aggregated step scores. While effective in domains such as mathematics, their applicability to tasks involving semi-structured data, like table question answering (TQA) remains unexplored. TQA poses unique challenges for PRMs, including abundant irrelevant information, loosely connected reasoning steps, and domain-specific reasoning. This work presents the first systematic study of PRMs for TQA. We evaluate state-of-the-art generative PRMs on TQA from both answer and step perspectives. Results show that PRMs that combine textual and code verification can aid solution selection but struggle to generalize to out-of-domain data. Analysis reveals a weak correlation between performance in step-level verification and answer accuracy, possibly stemming from weak step dependencies and loose causal links. Our findings highlight limitations of current PRMs on TQA and offer valuable insights for building more robust, process-aware verifiers.
>
---
#### [new 004] Automated Extraction of Fluoropyrimidine Treatment and Treatment-Related Toxicities from Clinical Notes Using Natural Language Processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理在医学文本中的应用任务，旨在从临床笔记中自动提取氟尿嘧啶类药物治疗及毒性信息。研究构建标注数据集，比较了规则、机器学习、深度学习及大语言模型方法，发现基于错误分析提示的LLM方法效果最佳，显著提升提取准确性，助力肿瘤学研究与药物安全监测。**

- **链接: [http://arxiv.org/pdf/2510.20727v1](http://arxiv.org/pdf/2510.20727v1)**

> **作者:** Xizhi Wu; Madeline S. Kreider; Philip E. Empey; Chenyu Li; Yanshan Wang
>
> **摘要:** Objective: Fluoropyrimidines are widely prescribed for colorectal and breast cancers, but are associated with toxicities such as hand-foot syndrome and cardiotoxicity. Since toxicity documentation is often embedded in clinical notes, we aimed to develop and evaluate natural language processing (NLP) methods to extract treatment and toxicity information. Materials and Methods: We constructed a gold-standard dataset of 236 clinical notes from 204,165 adult oncology patients. Domain experts annotated categories related to treatment regimens and toxicities. We developed rule-based, machine learning-based (Random Forest, Support Vector Machine [SVM], Logistic Regression [LR]), deep learning-based (BERT, ClinicalBERT), and large language models (LLM)-based NLP approaches (zero-shot and error-analysis prompting). Models used an 80:20 train-test split. Results: Sufficient data existed to train and evaluate 5 annotated categories. Error-analysis prompting achieved optimal precision, recall, and F1 scores (F1=1.000) for treatment and toxicities extraction, whereas zero-shot prompting reached F1=1.000 for treatment and F1=0.876 for toxicities extraction.LR and SVM ranked second for toxicities (F1=0.937). Deep learning underperformed, with BERT (F1=0.873 treatment; F1= 0.839 toxicities) and ClinicalBERT (F1=0.873 treatment; F1 = 0.886 toxicities). Rule-based methods served as our baseline with F1 scores of 0.857 in treatment and 0.858 in toxicities. Discussion: LMM-based approaches outperformed all others, followed by machine learning methods. Machine and deep learning approaches were limited by small training data and showed limited generalizability, particularly for rare categories. Conclusion: LLM-based NLP most effectively extracted fluoropyrimidine treatment and toxicity information from clinical notes, and has strong potential to support oncology research and pharmacovigilance.
>
---
#### [new 005] A Fundamental Algorithm for Dependency Parsing (With Corrections)
- **分类: cs.CL**

- **简介: 该论文提出一种用于依存句法分析的基本算法，旨在将自然语言句子解析为依存树。与短语结构解析不同，该算法逐词处理，即时附加词语，模拟人类大脑的解析特性。虽最坏时间复杂度为$O(n^3)$，但在真实语言中仅在小规模输入时出现。**

- **链接: [http://arxiv.org/pdf/2510.19996v1](http://arxiv.org/pdf/2510.19996v1)**

> **作者:** Michael A. Covington
>
> **备注:** Corrected version of an already widely cited paper
>
> **摘要:** This paper presents a fundamental algorithm for parsing natural language sentences into dependency trees. Unlike phrase-structure (constituency) parsers, this algorithm operates one word at a time, attaching each word as soon as it can be attached, corresponding to properties claimed for the parser in the human brain. Like phrase-structure parsing, its worst-case complexity is $O(n^3)$, but in human language, the worst case occurs only for small $n$.
>
---
#### [new 006] DeBERTa-KC: A Transformer-Based Classifier for Knowledge Construction in Online Learning Discourse
- **分类: cs.CL**

- **简介: 该论文提出DeBERTa-KC模型，用于自动分类在线科学学习对话中的知识建构（KC）水平。针对四类KC（非建构、分享、探索、协商），构建了2万条标注数据集，通过改进DeBERTa-v3并引入正则化技术，提升模型在类别不平衡下的性能。实验表明，该模型显著优于基线方法，能有效识别高阶认知参与行为。**

- **链接: [http://arxiv.org/pdf/2510.19858v1](http://arxiv.org/pdf/2510.19858v1)**

> **作者:** Jindi Wang; Yidi Zhang; Zhaoxing Li
>
> **摘要:** This study presents DeBERTa-KC, a transformer-based model for automatic classification of knowledge construction (KC) levels in online science learning discourse. Using comments collected from four popular YouTube science channels (2022--2024), a balanced corpus of 20,000 manually annotated samples was created across four KC categories: \textit{nonKC}, \textit{Share}, \textit{Explore}, and \textit{Negotiate}. The proposed model extends DeBERTa-v3 with Focal Loss, Label Smoothing, and R-Drop regularization to address class imbalance and enhance generalization. A reproducible end-to-end pipeline was implemented, encompassing data extraction, annotation, preprocessing, training, and evaluation. Across 10-fold stratified cross-validation, DeBERTa-KC achieved a macro-F1 of $0.836 \pm 0.008$, significantly out-performing both classical and transformer baselines ($p<0.01$). Per-category results indicate strong sensitivity to higher-order epistemic engagement, particularly in \textit{Explore} and \textit{Negotiate} discourse. These findings demonstrate that large language models can effectively capture nuanced indicators of knowledge construction in informal digital learning environments, offering scalable, theory-informed approaches to discourse analysis and the development of automated tools for assessing epistemic engagement.
>
---
#### [new 007] LLM-Augmented Symbolic NLU System for More Reliable Continuous Causal Statement Interpretation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对自然语言理解中LLM易幻觉、符号系统覆盖有限的问题，提出混合方法：用LLM增强文本简化与知识补全，结合符号NLU生成可推理的结构化表示。任务为从常识科学文本中提取因果关系与数量信息，实验表明该方法优于纯符号系统。**

- **链接: [http://arxiv.org/pdf/2510.19988v1](http://arxiv.org/pdf/2510.19988v1)**

> **作者:** Xin Lian; Kenneth D. Forbus
>
> **备注:** 18 pages, 2 figures
>
> **摘要:** Despite the broad applicability of large language models (LLMs), their reliance on probabilistic inference makes them vulnerable to errors such as hallucination in generated facts and inconsistent output structure in natural language understanding (NLU) tasks. By contrast, symbolic NLU systems provide interpretable understanding grounded in curated lexicons, semantic resources, and syntactic & semantic interpretation rules. They produce relational representations that can be used for accurate reasoning and planning, as well as incremental debuggable learning. However, symbolic NLU systems tend to be more limited in coverage than LLMs and require scarce knowledge representation and linguistics skills to extend and maintain. This paper explores a hybrid approach that integrates the broad-coverage language processing of LLMs with the symbolic NLU capabilities of producing structured relational representations to hopefully get the best of both approaches. We use LLMs for rephrasing and text simplification, to provide broad coverage, and as a source of information to fill in knowledge gaps more automatically. We use symbolic NLU to produce representations that can be used for reasoning and for incremental learning. We evaluate this approach on the task of extracting and interpreting quantities and causal laws from commonsense science texts, along with symbolic- and LLM-only pipelines. Our results suggest that our hybrid method works significantly better than the symbolic-only pipeline.
>
---
#### [new 008] Systematic Evaluation of Uncertainty Estimation Methods in Large Language Models
- **分类: cs.CL; stat.AP; stat.ME**

- **简介: 该论文针对大语言模型输出的不确定性问题，系统评估四种置信度估计方法（VCE、MSP、Sample Consistency、CoCoA），在四类问答任务上验证其性能。结果表明，各方法捕捉不同维度的置信度，混合方法CoCoA在校准与区分能力上最优，为实际应用提供选择依据。**

- **链接: [http://arxiv.org/pdf/2510.20460v1](http://arxiv.org/pdf/2510.20460v1)**

> **作者:** Christian Hobelsberger; Theresa Winner; Andreas Nawroth; Oliver Mitevski; Anna-Carolina Haensch
>
> **摘要:** Large language models (LLMs) produce outputs with varying levels of uncertainty, and, just as often, varying levels of correctness; making their practical reliability far from guaranteed. To quantify this uncertainty, we systematically evaluate four approaches for confidence estimation in LLM outputs: VCE, MSP, Sample Consistency, and CoCoA (Vashurin et al., 2025). For the evaluation of the approaches, we conduct experiments on four question-answering tasks using a state-of-the-art open-source LLM. Our results show that each uncertainty metric captures a different facet of model confidence and that the hybrid CoCoA approach yields the best reliability overall, improving both calibration and discrimination of correct answers. We discuss the trade-offs of each method and provide recommendations for selecting uncertainty measures in LLM applications.
>
---
#### [new 009] An Evaluation of the Pedagogical Soundness and Usability of AI-Generated Lesson Plans Across Different Models and Prompt Frameworks in High-School Physics
- **分类: cs.CL; cs.AI; G.1.10; G.4; I.2.6; I.2.7**

- **简介: 该论文属于教育AI任务，旨在评估不同AI模型与提示框架生成高中物理教案的教育有效性。研究对比五种模型和三种提示框架，分析其可读性、准确性、课程契合度及认知要求，发现提示框架显著影响教学可靠性，最优组合为高可读性模型+RACE框架+概念与标准检查清单。**

- **链接: [http://arxiv.org/pdf/2510.19866v1](http://arxiv.org/pdf/2510.19866v1)**

> **作者:** Xincheng Liu
>
> **备注:** 20 pages, 6 tables
>
> **摘要:** This study evaluates the pedagogical soundness and usability of AI-generated lesson plans across five leading large language models: ChatGPT (GPT-5), Claude Sonnet 4.5, Gemini 2.5 Flash, DeepSeek V3.2, and Grok 4. Beyond model choice, three structured prompt frameworks were tested: TAG (Task, Audience, Goal), RACE (Role, Audience, Context, Execution), and COSTAR (Context, Objective, Style, Tone, Audience, Response Format). Fifteen lesson plans were generated for a single high-school physics topic, The Electromagnetic Spectrum. The lesson plans were analyzed through four automated computational metrics: (1) readability and linguistic complexity, (2) factual accuracy and hallucination detection, (3) standards and curriculum alignment, and (4) cognitive demand of learning objectives. Results indicate that model selection exerted the strongest influence on linguistic accessibility, with DeepSeek producing the most readable teaching plan (FKGL = 8.64) and Claude generating the densest language (FKGL = 19.89). The prompt framework structure most strongly affected the factual accuracy and pedagogical completeness, with the RACE framework yielding the lowest hallucination index and the highest incidental alignment with NGSS curriculum standards. Across all models, the learning objectives in the fifteen lesson plans clustered at the Remember and Understand tiers of Bloom's taxonomy. There were limited higher-order verbs in the learning objectives extracted. Overall, the findings suggest that readability is significantly governed by model design, while instructional reliability and curricular alignment depend more on the prompt framework. The most effective configuration for lesson plans identified in the results was to combine a readability-optimized model with the RACE framework and an explicit checklist of physics concepts, curriculum standards, and higher-order objectives.
>
---
#### [new 010] Tri-Modal Severity Fused Diagnosis across Depression and Post-traumatic Stress Disorders
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种三模态情感严重度融合诊断框架，用于共病抑郁与创伤后应激障碍的分级评估。通过融合文本、音频、面部信号，实现跨疾病严重度预测与可解释决策支持，提升诊断准确性与鲁棒性，解决现有方法二元化、缺乏严重度感知与多模态协同的问题。**

- **链接: [http://arxiv.org/pdf/2510.20239v1](http://arxiv.org/pdf/2510.20239v1)**

> **作者:** Filippo Cenacchi; Deborah Richards; Longbing Cao
>
> **摘要:** Depression and post traumatic stress disorder (PTSD) often co-occur with connected symptoms, complicating automated assessment, which is often binary and disorder specific. Clinically useful diagnosis needs severity aware cross disorder estimates and decision support explanations. Our unified tri modal affective severity framework synchronizes and fuses interview text with sentence level transformer embeddings, audio with log Mel statistics with deltas, and facial signals with action units, gaze, head and pose descriptors to output graded severities for diagnosing both depression (PHQ-8; 5 classes) and PTSD (3 classes). Standardized features are fused via a calibrated late fusion classifier, yielding per disorder probabilities and feature-level attributions. This severity aware tri-modal affective fusion approach is demoed on multi disorder concurrent depression and PTSD assessment. Stratified cross validation on DAIC derived corpora outperforms unimodal/ablation baselines. The fused model matches the strongest unimodal baseline on accuracy and weighted F1, while improving decision curve utility and robustness under noisy or missing modalities. For PTSD specifically, fusion reduces regression error and improves class concordance. Errors cluster between adjacent severities; extreme classes are identified reliably. Ablations show text contributes most to depression severity, audio and facial cues are critical for PTSD, whereas attributions align with linguistic and behavioral markers. Our approach offers reproducible evaluation and clinician in the loop support for affective clinical decision making.
>
---
#### [new 011] Teacher Demonstrations in a BabyLM's Zone of Proximal Development for Contingent Multi-Turn Interaction
- **分类: cs.CL**

- **简介: 该论文研究多轮对话中的情境连续性问题，提出ContingentChat框架，通过教师示范引导婴儿语言模型（BabyLM）在最近发展区提升对话连贯性。基于1亿词数据训练，利用新对齐数据集进行后训练，增强响应的语法与连贯性，验证了针对性后训练对提升对话质量的有效性，但连续性仍具挑战。**

- **链接: [http://arxiv.org/pdf/2510.20411v1](http://arxiv.org/pdf/2510.20411v1)**

> **作者:** Suchir Salhan; Hongyi Gu; Donya Rooein; Diana Galvan-Sosa; Gabrielle Gaudeau; Andrew Caines; Zheng Yuan; Paula Buttery
>
> **备注:** Outstanding Paper Award, EMNLP 2025 BabyLM Workshop - Oral presentation, Suzhou, China
>
> **摘要:** Multi-turn dialogues between a child and a caregiver are characterized by a property called contingency - that is, prompt, direct, and meaningful exchanges between interlocutors. We introduce ContingentChat, a teacher-student framework that benchmarks and improves multi-turn contingency in a BabyLM trained on 100M words. Using a novel alignment dataset for post-training, BabyLM generates responses that are more grammatical and cohesive. Experiments with adaptive teacher decoding strategies show limited additional gains. ContingentChat demonstrates the benefits of targeted post-training for dialogue quality and indicates that contingency remains a challenging goal for BabyLMs.
>
---
#### [new 012] Forging GEMs: Advancing Greek NLP through Quality-Based Corpus Curation and Specialized Pre-training
- **分类: cs.CL; cs.AI; 68T50, 68T07, 68U35**

- **简介: 该论文针对希腊语自然语言处理中数据碎片化、模型架构单一及上下文长度受限的问题，提出基于高质量语料库构建与专业化预训练的GEM系列模型。通过严谨的数据筛选与预处理，构建通用与法律领域语料库，采用ELECTRA、ConvBERT等现代架构进行预训练，并首次推出双语法律嵌入模型。实验证明新模型显著优于现有基线。**

- **链接: [http://arxiv.org/pdf/2510.20002v1](http://arxiv.org/pdf/2510.20002v1)**

> **作者:** Alexandra Apostolopoulou; Konstantinos Kanaris; Athanasios Koursaris; Dimitris Tsakalidis; George Domalis; Ioannis E. Livieris
>
> **摘要:** The advancement of natural language processing for morphologically rich, moderately-resourced languages like Modern Greek is often hindered by a fragmented research landscape, a lack of architectural diversity and reliance on limited context-length models. This is particularly true in specialized, high-value domains such as law, where existing models are frequently confined to early transformer architectures with a restrictive 512-token window, insufficient for analyzing long legal documents. To address these challenges, this paper presents Greek Embedding Models, a new family of transformer models for Greek language built upon a foundation of extensive, quality-driven data curation. We detail the construction of several large-scale Greek corpora, emphasizing a rigorous, quality-based filtering and preprocessing methodology to create high-value training datasets from both general-domain and specialized legal sources. On this carefully curated foundation, we pre-train and systematically evaluate a diverse suite of modern architectures, which has not previously applied to Greek language, such as ELECTRA, ConvBERT and ModernBERT. Furthermore, we propose the first bilingual Greek-English Embedding Models tailored for the legal domain. The extensive experiments on downstream tasks demonstrate that the new class of models establish the effectiveness of the proposed approach, highlighting that the GEM-RoBERTa and GEM-ConvBERT models significantly outperform existing baselines.
>
---
#### [new 013] Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于表格理解任务，针对现有方法在推理与表操作间平衡不足的问题，提出Mixture-of-Minds多智能体框架。通过规划、编码、回答三角色协作，结合蒙特卡洛树搜索与强化学习实现自优化，显著提升准确率，优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.20176v1](http://arxiv.org/pdf/2510.20176v1)**

> **作者:** Yuhang Zhou; Mingrui Zhang; Ke Li; Mingyi Wang; Qiao Liu; Qifei wang; Jiayi Liu; Fei Liu; Serena Li; Weiwi Li; Mingze Gao; Abhishek Kumar; Xiangjun Fan; Zhuokai Zhao; Lizhu Zhang
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding.
>
---
#### [new 014] From Facts to Folklore: Evaluating Large Language Models on Bengali Cultural Knowledge
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文聚焦于低资源文化知识评估任务，针对大语言模型在孟加拉文化知识上的表现不足，构建了BLanCK数据集，涵盖民间传统、烹饪与方言。实验表明，上下文信息显著提升模型表现，强调需发展情境感知架构与文化定制训练数据。**

- **链接: [http://arxiv.org/pdf/2510.20043v1](http://arxiv.org/pdf/2510.20043v1)**

> **作者:** Nafis Chowdhury; Moinul Haque; Anika Ahmed; Nazia Tasnim; Md. Istiak Hossain Shihab; Sajjadur Rahman; Farig Sadeque
>
> **备注:** 4 pages
>
> **摘要:** Recent progress in NLP research has demonstrated remarkable capabilities of large language models (LLMs) across a wide range of tasks. While recent multilingual benchmarks have advanced cultural evaluation for LLMs, critical gaps remain in capturing the nuances of low-resource cultures. Our work addresses these limitations through a Bengali Language Cultural Knowledge (BLanCK) dataset including folk traditions, culinary arts, and regional dialects. Our investigation of several multilingual language models shows that while these models perform well in non-cultural categories, they struggle significantly with cultural knowledge and performance improves substantially across all models when context is provided, emphasizing context-aware architectures and culturally curated training data.
>
---
#### [new 015] Can ChatGPT Code Communication Data Fairly?: Empirical Evidence from Multiple Collaborative Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI（ChatGPT）在协作任务中自动编码沟通数据的公平性问题，针对性别与种族差异是否导致偏差。通过分析三类协作任务数据，发现ChatGPT编码无显著群体偏差，支持其在大规模协作评估中的可靠应用。**

- **链接: [http://arxiv.org/pdf/2510.20584v1](http://arxiv.org/pdf/2510.20584v1)**

> **作者:** Jiangang Hao; Wenju Cui; Patrick Kyllonen; Emily Kerzabi
>
> **备注:** 38 pages, 4 figures
>
> **摘要:** Assessing communication and collaboration at scale depends on a labor intensive task of coding communication data into categories according to different frameworks. Prior research has established that ChatGPT can be directly instructed with coding rubrics to code the communication data and achieves accuracy comparable to human raters. However, whether the coding from ChatGPT or similar AI technology exhibits bias against different demographic groups, such as gender and race, remains unclear. To fill this gap, this paper investigates ChatGPT-based automated coding of communication data using a typical coding framework for collaborative problem solving, examining differences across gender and racial groups. The analysis draws on data from three types of collaborative tasks: negotiation, problem solving, and decision making. Our results show that ChatGPT-based coding exhibits no significant bias across gender and racial groups, paving the road for its adoption in large-scale assessment of collaboration and communication.
>
---
#### [new 016] Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大推理模型（LRM）作为机器翻译评估工具的潜力。针对其评估时存在“过度思考”、评分偏差及资源浪费问题，提出通过合成人类思维轨迹对LRM进行校准。实验表明，该方法显著降低推理成本约35倍，同时提升不同规模模型的评估相关性，有效推动细粒度自动翻译评估发展。**

- **链接: [http://arxiv.org/pdf/2510.20780v1](http://arxiv.org/pdf/2510.20780v1)**

> **作者:** Runzhe Zhan; Zhihong Huang; Xinyi Yang; Lidia S. Chao; Min Yang; Derek F. Wong
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent advancements in large reasoning models (LRMs) have introduced an intermediate "thinking" process prior to generating final answers, improving their reasoning capabilities on complex downstream tasks. However, the potential of LRMs as evaluators for machine translation (MT) quality remains underexplored. We provides the first systematic analysis of LRM-as-a-judge in MT evaluation. We identify key challenges, revealing LRMs require tailored evaluation materials, tend to "overthink" simpler instances and have issues with scoring mechanisms leading to overestimation. To address these, we propose to calibrate LRM thinking by training them on synthetic, human-like thinking trajectories. Our experiments on WMT24 Metrics benchmarks demonstrate that this approach largely reduces thinking budgets by ~35x while concurrently improving evaluation performance across different LRM scales from 7B to 32B (e.g., R1-Distill-Qwen-7B achieves a +8.7 correlation point improvement). These findings highlight the potential of efficiently calibrated LRMs to advance fine-grained automatic MT evaluation.
>
---
#### [new 017] Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention
- **分类: cs.CL; cs.AI; 68T40; I.2.11**

- **简介: 该论文针对长上下文大模型的机制可解释性难题，提出Stream方法，通过稀疏注意力实现近线性时间与线性空间的注意力模式分析。解决了传统方法在百万级上下文时内存需求过高的问题，实现了高效、可部署的链式思维追踪与信息流解析。**

- **链接: [http://arxiv.org/pdf/2510.19875v1](http://arxiv.org/pdf/2510.19875v1)**

> **作者:** J Rosser; José Luis Redondo García; Gustavo Penha; Konstantina Palla; Hugues Bouchard
>
> **摘要:** As Large Language Models (LLMs) scale to million-token contexts, traditional Mechanistic Interpretability techniques for analyzing attention scale quadratically with context length, demanding terabytes of memory beyond 100,000 tokens. We introduce Sparse Tracing, a novel technique that leverages dynamic sparse attention to efficiently analyze long context attention patterns. We present Stream, a compilable hierarchical pruning algorithm that estimates per-head sparse attention masks in near-linear time $O(T \log T)$ and linear space $O(T)$, enabling one-pass interpretability at scale. Stream performs a binary-search-style refinement to retain only the top-$k$ key blocks per query while preserving the model's next-token behavior. We apply Stream to long chain-of-thought reasoning traces and identify thought anchors while pruning 97-99\% of token interactions. On the RULER benchmark, Stream preserves critical retrieval paths while discarding 90-96\% of interactions and exposes layer-wise routes from the needle to output. Our method offers a practical drop-in tool for analyzing attention patterns and tracing information flow without terabytes of caches. By making long context interpretability feasible on consumer GPUs, Sparse Tracing helps democratize chain-of-thought monitoring. Code is available at https://anonymous.4open.science/r/stream-03B8/.
>
---
#### [new 018] The Dog the Cat Chased Stumped the Model: Measuring When Language Models Abandon Structure for Shortcuts
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语言模型是否真正理解语法结构还是依赖语义捷径的问题，提出CenterBench数据集，包含9720个中心嵌套句的问答任务。通过对比语义合理与不合理句子的表现，量化模型在复杂结构下从语法分析转向语义匹配的临界点，揭示模型在因果推理上的局限性。**

- **链接: [http://arxiv.org/pdf/2510.20543v1](http://arxiv.org/pdf/2510.20543v1)**

> **作者:** Sangmitra Madhusudan; Kaige Chen; Ali Emami
>
> **摘要:** When language models correctly parse "The cat that the dog chased meowed," are they analyzing syntax or simply familiar with dogs chasing cats? Despite extensive benchmarking, we lack methods to distinguish structural understanding from semantic pattern matching. We introduce CenterBench, a dataset of 9,720 comprehension questions on center-embedded sentences (like "The cat [that the dog chased] meowed") where relative clauses nest recursively, creating processing demands from simple to deeply nested structures. Each sentence has a syntactically identical but semantically implausible counterpart (e.g., mailmen prescribe medicine, doctors deliver mail) and six comprehension questions testing surface understanding, syntactic dependencies, and causal reasoning. Testing six models reveals that performance gaps between plausible and implausible sentences widen systematically with complexity, with models showing median gaps up to 26.8 percentage points, quantifying when they abandon structural analysis for semantic associations. Notably, semantic plausibility harms performance on questions about resulting actions, where following causal relationships matters more than semantic coherence. Reasoning models improve accuracy but their traces show semantic shortcuts, overthinking, and answer refusal. Unlike models whose plausibility advantage systematically widens with complexity, humans shows variable semantic effects. CenterBench provides the first framework to identify when models shift from structural analysis to pattern matching.
>
---
#### [new 019] Teaching Language Models to Reason with Tools
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在复杂数学推理中效率低、错误率高的问题，提出CoRT框架，通过提示工程与强化学习优化模型对代码解释器的使用，显著提升推理准确率与效率。**

- **链接: [http://arxiv.org/pdf/2510.20342v1](http://arxiv.org/pdf/2510.20342v1)**

> **作者:** Chengpeng Li; Zhengyang Tang; Ziniu Li; Mingfeng Xue; Keqin Bao; Tian Ding; Ruoyu Sun; Benyou Wang; Xiang Wang; Junyang Lin; Dayiheng Liu
>
> **备注:** NIPS2025 Accepted
>
> **摘要:** Large reasoning models (LRMs) like OpenAI-o1 have shown impressive capabilities in natural language reasoning. However, these models frequently demonstrate inefficiencies or inaccuracies when tackling complex mathematical operations. While integrating computational tools such as Code Interpreters (CIs) offers a promising solution, it introduces a critical challenge: a conflict between the model's internal, probabilistic reasoning and the external, deterministic knowledge provided by the CI, which often leads models to unproductive deliberation. To overcome this, we introduce CoRT (Code-Optimized Reasoning Training), a post-training framework designed to teach LRMs to effectively utilize CIs. We propose \emph{Hint-Engineering}, a new data synthesis strategy that strategically injects diverse hints at optimal points within reasoning paths. This approach generates high-quality, code-integrated reasoning data specifically tailored to optimize LRM-CI interaction. Using this method, we have synthesized 30 high-quality samples to post-train models ranging from 1.5B to 32B parameters through supervised fine-tuning. CoRT further refines the multi-round interleaving of external CI usage and internal thinking by employing rejection sampling and reinforcement learning. Our experimental evaluations demonstrate CoRT's effectiveness, yielding absolute improvements of 4\% and 8\% on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B, respectively, across five challenging mathematical reasoning datasets. Moreover, CoRT significantly enhances efficiency, reducing token usage by approximately 30\% for the 32B model and 50\% for the 1.5B model compared to pure natural language reasoning baselines. The models and code are available at: https://github.com/ChengpengLi1003/CoRT.
>
---
#### [new 020] LyriCAR: A Difficulty-Aware Curriculum Reinforcement Learning Framework For Controllable Lyric Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对歌词翻译任务，解决现有方法难以兼顾音乐约束与段落级连贯性的问题。提出LyriCAR框架，通过难度感知课程设计与自适应策略，实现无监督下的高效训练，显著提升翻译质量并减少40%训练步数。**

- **链接: [http://arxiv.org/pdf/2510.19967v1](http://arxiv.org/pdf/2510.19967v1)**

> **作者:** Le Ren; Xiangjian Zeng; Qingqiang Wu; Ruoxuan Liang
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Lyric translation is a challenging task that requires balancing multiple musical constraints. Existing methods often rely on hand-crafted rules and sentence-level modeling, which restrict their ability to internalize musical-linguistic patterns and to generalize effectively at the paragraph level, where cross-line coherence and global rhyme are crucial. In this work, we propose LyriCAR, a novel framework for controllable lyric translation that operates in a fully unsupervised manner. LyriCAR introduces a difficulty-aware curriculum designer and an adaptive curriculum strategy, ensuring efficient allocation of training resources, accelerating convergence, and improving overall translation quality by guiding the model with increasingly complex challenges. Extensive experiments on the EN-ZH lyric translation task show that LyriCAR achieves state-of-the-art results across both standard translation metrics and multi-dimensional reward scores, surpassing strong baselines. Notably, the adaptive curriculum strategy reduces training steps by nearly 40% while maintaining superior performance. Code, data and model can be accessed at https://github.com/rle27/LyriCAR.
>
---
#### [new 021] Learning from Supervision with Semantic and Episodic Memory: A Reflective Approach to Agent Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何让基于大语言模型的智能体在不更新参数的情况下，通过记忆机制学习分类任务。针对传统微调成本高、不透明的问题，提出融合情景记忆与语义记忆的反思性框架，利用标签和模型生成的批评进行学习，显著提升准确率，并引入新指标“易受暗示性”分析模型行为差异。**

- **链接: [http://arxiv.org/pdf/2510.19897v1](http://arxiv.org/pdf/2510.19897v1)**

> **作者:** Jackson Hassell; Dan Zhang; Hannah Kim; Tom Mitchell; Estevam Hruschka
>
> **备注:** 11 pages
>
> **摘要:** We investigate how agents built on pretrained large language models can learn target classification functions from labeled examples without parameter updates. While conventional approaches like fine-tuning are often costly, inflexible, and opaque, we propose a memory-augmented framework that leverages both labeled data and LLM-generated critiques. Our framework uses episodic memory to store instance-level critiques-capturing specific past experiences-and semantic memory to distill these into reusable, task-level guidance. Across a diverse set of tasks, incorporating critiques yields up to a 24.8 percent accuracy improvement over retrieval-based (RAG-style) baselines that rely only on labels. Through extensive empirical evaluation, we uncover distinct behavioral differences between OpenAI and opensource models, particularly in how they handle fact-oriented versus preference-based data. To interpret how models respond to different representations of supervision encoded in memory, we introduce a novel metric, suggestibility. This helps explain observed behaviors and illuminates how model characteristics and memory strategies jointly shape learning dynamics. Our findings highlight the promise of memory-driven, reflective learning for building more adaptive and interpretable LLM agents.
>
---
#### [new 022] Citation Failure: Definition, Analysis and Efficient Mitigation
- **分类: cs.CL**

- **简介: 该论文聚焦于大模型问答系统中的“引用失败”问题，即模型生成正确回答但未提供完整证据。提出CITECONTROL基准分析引用失败成因，并设计CITENTION框架融合多种方法，有效提升引用准确性，推动可验证AI响应的发展。**

- **链接: [http://arxiv.org/pdf/2510.20303v1](http://arxiv.org/pdf/2510.20303v1)**

> **作者:** Jan Buchmann; Iryna Gurevych
>
> **备注:** Under review. Paper repository: https://github.com/UKPLab/arxiv2025-citation-failure
>
> **摘要:** Citations from LLM-based RAG systems are supposed to simplify response verification. However, this does not hold for citation failure, when a model generates a helpful response, but fails to cite complete evidence. In contrast to previous work, we propose to disentangle this from response failure, where the response itself is flawed, and citing complete evidence is impossible. To address citation failure, this work follows a two-step approach: (1) We study when citation failure occurs and (2) how it can be mitigated. For step 1, we extend prior work by investigating how the relation between response and evidence affects citation quality. We introduce CITECONTROL, a benchmark that systematically varies this relation to analyze failure modes. Experiments show that failures increase with relational complexity and suggest that combining citation methods could improve performance, motivating step 2. To improve LLM citation efficiently, we propose CITENTION, a framework integrating generative, attention-based, and retrieval-based methods. Results demonstrate substantial citation improvements on CITECONTROL and in transfer settings. We make our data and code publicly available.
>
---
#### [new 023] Beyond MedQA: Towards Real-world Clinical Decision Making in the Era of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在临床决策中的评估局限性，提出基于“临床背景”与“临床问题”二维框架的统一范式。旨在解决现有数据集（如MedQA）过于简化、无法反映真实临床决策的问题。工作包括梳理现有数据集、总结应对方法、扩展评估维度，并指出未来挑战，以推动更真实、可解释的临床AI发展。**

- **链接: [http://arxiv.org/pdf/2510.20001v1](http://arxiv.org/pdf/2510.20001v1)**

> **作者:** Yunpeng Xiao; Carl Yang; Mark Mai; Xiao Hu; Kai Shu
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Large language models (LLMs) show promise for clinical use. They are often evaluated using datasets such as MedQA. However, Many medical datasets, such as MedQA, rely on simplified Question-Answering (Q\A) that underrepresents real-world clinical decision-making. Based on this, we propose a unifying paradigm that characterizes clinical decision-making tasks along two dimensions: Clinical Backgrounds and Clinical Questions. As the background and questions approach the real clinical environment, the difficulty increases. We summarize the settings of existing datasets and benchmarks along two dimensions. Then we review methods to address clinical decision-making, including training-time and test-time techniques, and summarize when they help. Next, we extend evaluation beyond accuracy to include efficiency, explainability. Finally, we highlight open challenges. Our paradigm clarifies assumptions, standardizes comparisons, and guides the development of clinically meaningful LLMs.
>
---
#### [new 024] RECALL: REpresentation-aligned Catastrophic-forgetting ALLeviation via Hierarchical Model Merging
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对持续学习中灾难性遗忘问题，提出RECALL框架。通过分析模型层间表示相似性，实现无历史数据下的分层参数融合，有效保留通用特征并适应新任务，提升知识保留与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.20479v1](http://arxiv.org/pdf/2510.20479v1)**

> **作者:** Bowen Wang; Haiyuan Wan; Liwen Shi; Chen Yang; Peng He; Yue Ma; Haochen Han; Wenhao Li; Tiao Tan; Yongjian Li; Fangming Liu; Yifan Gong; Sheng Zhang
>
> **摘要:** We unveil that internal representations in large language models (LLMs) serve as reliable proxies of learned knowledge, and propose RECALL, a novel representation-aware model merging framework for continual learning without access to historical data. RECALL computes inter-model similarity from layer-wise hidden representations over clustered typical samples, and performs adaptive, hierarchical parameter fusion to align knowledge across models. This design enables the preservation of domain-general features in shallow layers while allowing task-specific adaptation in deeper layers. Unlike prior methods that require task labels or incur performance trade-offs, RECALL achieves seamless multi-domain integration and strong resistance to catastrophic forgetting. Extensive experiments across five NLP tasks and multiple continual learning scenarios show that RECALL outperforms baselines in both knowledge retention and generalization, providing a scalable and data-free solution for evolving LLMs.
>
---
#### [new 025] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation
- **分类: cs.CL**

- **简介: 该论文提出BoundRL，用于复杂结构化文本的高效分段任务。针对传统方法难以处理表格、代码等非纯文本元素的问题，提出仅生成起始标记并重建内容的新机制，结合强化学习与中间候选生成，显著降低计算成本与幻觉，提升小模型性能。**

- **链接: [http://arxiv.org/pdf/2510.20151v1](http://arxiv.org/pdf/2510.20151v1)**

> **作者:** Haoyuan Li; Zhengyuan Shen; Sullam Jeoung; Yueyan Chen; Jiayu Li; Qi Zhu; Shuai Wang; Vassilis Ioannidis; Huzefa Rangwala
>
> **摘要:** As structured texts become increasingly complex across diverse domains -- from technical reports to generative AI prompts -- the need for text segmentation into semantically meaningful components becomes critical. Such texts often contain elements beyond plain language, including tables, code snippets, and placeholders, which conventional sentence- or paragraph-level segmentation methods cannot handle effectively. To address this challenge, we propose BoundRL, a novel and efficient approach that jointly performs token-level text segmentation and label prediction for long structured texts. Instead of generating complete contents for each segment, it generates only a sequence of starting tokens and reconstructs the complete contents by locating these tokens within the original texts, thereby reducing inference costs by orders of magnitude and minimizing hallucination. To adapt the model for the output format, BoundRL~performs reinforcement learning with verifiable rewards (RLVR) with a specifically designed reward that jointly optimizes document reconstruction fidelity and semantic alignment. To mitigate entropy collapse, it further constructs intermediate candidates by systematically perturbing a fraction of generated sequences of segments to create stepping stones toward higher-quality solutions. To demonstrate BoundRL's effectiveness on particularly challenging structured texts, we focus evaluation on complex prompts used for LLM applications. Experiments show that BoundRL enables small language models (1.7B parameters) to outperform few-shot prompting of much larger models. Moreover, RLVR with our designed reward yields significant improvements over supervised fine-tuning, and incorporating intermediate candidates further improves both performance and generalization.
>
---
#### [new 026] ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering
- **分类: cs.CL; cs.SE**

- **简介: 该论文针对大模型代理在工具使用中因工具冗余和上下文限制导致的选择不准确问题，提出ToolScope框架。通过工具合并与自动修正减少冗余，利用上下文感知筛选压缩工具集，提升工具选择准确率，显著改善了复杂任务下的工具使用效率。**

- **链接: [http://arxiv.org/pdf/2510.20036v1](http://arxiv.org/pdf/2510.20036v1)**

> **作者:** Marianne Menglin Liu; Daniel Garcia; Fjona Parllaku; Vikas Upadhyay; Syed Fahad Allam Shah; Dan Roth
>
> **备注:** Preprint under review
>
> **摘要:** Large language model (LLM) agents rely on external tools to solve complex tasks, but real-world toolsets often contain redundant tools with overlapping names and descriptions, introducing ambiguity and reducing selection accuracy. LLMs also face strict input context limits, preventing efficient consideration of large toolsets. To address these challenges, we propose ToolScope, which includes: (1) ToolScopeMerger with Auto-Correction to automatically audit and fix tool merges, reducing redundancy, and (2) ToolScopeRetriever to rank and select only the most relevant tools for each query, compressing toolsets to fit within context limits without sacrificing accuracy. Evaluations on three state-of-the-art LLMs and three open-source tool-use benchmarks show gains of 8.38% to 38.6% in tool selection accuracy, demonstrating ToolScope's effectiveness in enhancing LLM tool use.
>
---
#### [new 027] Why Did Apple Fall To The Ground: Evaluating Curiosity In Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型好奇心评估任务，旨在探究LLMs是否具备类人好奇驱动学习能力。基于5DCR量表构建评估框架，从信息寻求、刺激追求等维度量化模型好奇心，发现其知识渴求强但决策保守，且好奇行为可提升推理与主动学习能力。**

- **链接: [http://arxiv.org/pdf/2510.20635v1](http://arxiv.org/pdf/2510.20635v1)**

> **作者:** Haoyu Wang; Sihang Jiang; Yuyan Chen; Yitong Wang; Yanghua Xiao
>
> **摘要:** Curiosity serves as a pivotal conduit for human beings to discover and learn new knowledge. Recent advancements of large language models (LLMs) in natural language processing have sparked discussions regarding whether these models possess capability of curiosity-driven learning akin to humans. In this paper, starting from the human curiosity assessment questionnaire Five-Dimensional Curiosity scale Revised (5DCR), we design a comprehensive evaluation framework that covers dimensions such as Information Seeking, Thrill Seeking, and Social Curiosity to assess the extent of curiosity exhibited by LLMs. The results demonstrate that LLMs exhibit a stronger thirst for knowledge than humans but still tend to make conservative choices when faced with uncertain environments. We further investigated the relationship between curiosity and thinking of LLMs, confirming that curious behaviors can enhance the model's reasoning and active learning abilities. These findings suggest that LLMs have the potential to exhibit curiosity similar to that of humans, providing experimental support for the future development of learning capabilities and innovative research in LLMs.
>
---
#### [new 028] The Reasoning Lingua Franca: A Double-Edged Sword for Multilingual AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型推理任务，聚焦于模型在非英语问题上为何倾向用英语推理。通过对比英文与原语言推理在数学与科学问答任务中的表现，发现英语推理虽准确率更高，但易因“翻译丢失”导致错误，揭示了语言选择对推理质量的关键影响。**

- **链接: [http://arxiv.org/pdf/2510.20647v1](http://arxiv.org/pdf/2510.20647v1)**

> **作者:** Alan Saji; Raj Dabre; Anoop Kunchukuttan; Ratish Puduppully
>
> **备注:** 14 pages, 13 figures, 5 tables
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance on mathematical, scientific, and other question-answering tasks, but their multilingual reasoning abilities remain underexplored. When presented with non-English questions, LRMs often default to reasoning in English, raising concerns about interpretability and the handling of linguistic and cultural nuances. We systematically compare an LRM's reasoning in English versus the language of the question. Our evaluation spans two tasks: MGSM and GPQA Diamond. Beyond measuring answer accuracy, we also analyze cognitive attributes in the reasoning traces. We find that English reasoning traces exhibit a substantially higher presence of these cognitive behaviors, and that reasoning in English generally yields higher final-answer accuracy, with the performance gap increasing as tasks become more complex. However, this English-centric strategy is susceptible to a key failure mode - getting "Lost in Translation," where translation steps lead to errors that would have been avoided by question's language reasoning.
>
---
#### [new 029] Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对线性注意力模型因有限记忆导致的遗忘问题，提出混合稀疏注意力与可学习令牌淘汰机制。通过引入轻量级CNN自适应保留关键键值对，在保持线性复杂度的同时提升检索任务性能。**

- **链接: [http://arxiv.org/pdf/2510.20787v1](http://arxiv.org/pdf/2510.20787v1)**

> **作者:** Mutian He; Philip N. Garner
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Linear-attention models that compress the entire input sequence into a fixed-size recurrent state offer an efficient alternative to Transformers, but their finite memory induces forgetfulness that harms retrieval-intensive tasks. To mitigate the issue, we explore a series of hybrid models that restore direct access to past tokens. We interleave token mixers with intermediate time and space complexity between linear and full attention, including sparse attention with token eviction, and the query-aware native sparse attention. Particularly, we propose a novel learnable token eviction approach. Combined with sliding-window attention, an end-to-end trainable lightweight CNN aggregates information from both past and future adjacent tokens to adaptively retain a limited set of critical KV-pairs per head, maintaining linear attention's constant time and space complexity. Efficient Triton kernels for the sparse attention mechanisms are provided. Empirical evaluations on retrieval-intensive benchmarks support the effectiveness of our approaches.
>
---
#### [new 030] BUSTED at AraGenEval Shared Task: A Comparative Study of Transformer-Based Models for Arabic AI-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与阿拉伯语AI生成文本检测任务，旨在区分人工与AI生成的阿拉伯语文本。作者对比了AraELECTRA、CAMeLBERT和XLM-RoBERTa三类预训练模型，通过微调进行二分类。结果发现，通用多语言模型XLM-RoBERTa表现最优，F1达0.7701，表明其在阿拉伯语检测中具备强泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.20610v1](http://arxiv.org/pdf/2510.20610v1)**

> **作者:** Ali Zain; Sareem Farooqui; Muhammad Rafi
>
> **摘要:** This paper details our submission to the Ara- GenEval Shared Task on Arabic AI-generated text detection, where our team, BUSTED, se- cured 5th place. We investigated the effec- tiveness of three pre-trained transformer mod- els: AraELECTRA, CAMeLBERT, and XLM- RoBERTa. Our approach involved fine-tuning each model on the provided dataset for a binary classification task. Our findings revealed a sur- prising result: the multilingual XLM-RoBERTa model achieved the highest performance with an F1 score of 0.7701, outperforming the spe- cialized Arabic models. This work underscores the complexities of AI-generated text detection and highlights the strong generalization capa- bilities of multilingual models.
>
---
#### [new 031] Context-level Language Modeling by Learning Predictive Context Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ContextLM框架，旨在解决传统语言模型仅依赖词元级预测导致的长程上下文建模不足问题。通过引入“下一上下文预测”目标，使模型学习多词元上下文的预测表示，增强语义结构捕捉能力，提升长程连贯性与注意力效率，且兼容标准自回归评估方式。**

- **链接: [http://arxiv.org/pdf/2510.20280v1](http://arxiv.org/pdf/2510.20280v1)**

> **作者:** Beiya Dai; Yuliang Liu; Daozheng Xue; Qipeng Guo; Kai Chen; Xinbing Wang
>
> **备注:** 16pages,6 figures
>
> **摘要:** Next-token prediction (NTP) is the cornerstone of modern large language models (LLMs) pretraining, driving their unprecedented capabilities in text generation, reasoning, and instruction following. However, the token-level prediction limits the model's capacity to capture higher-level semantic structures and long-range contextual relationships. To overcome this limitation, we introduce \textbf{ContextLM}, a framework that augments standard pretraining with an inherent \textbf{next-context prediction} objective. This mechanism trains the model to learn predictive representations of multi-token contexts, leveraging error signals derived from future token chunks. Crucially, ContextLM achieves this enhancement while remaining fully compatible with the standard autoregressive, token-by-token evaluation paradigm (e.g., perplexity). Extensive experiments on the GPT2 and Pythia model families, scaled up to $1.5$B parameters, show that ContextLM delivers consistent improvements in both perplexity and downstream task performance. Our analysis indicates that next-context prediction provides a scalable and efficient pathway to stronger language modeling, yielding better long-range coherence and more effective attention allocation with minimal computational overhead.
>
---
#### [new 032] Large Language Model enabled Mathematical Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在运筹学数学建模中的应用，旨在解决传统优化建模依赖专家知识、效率低的问题。针对DeepSeek-R1模型，通过多基准测试与纠错策略（如LLM-judge、少样本学习），提升其建模准确性与可靠性，推动自然语言到数学模型的自动化转化。**

- **链接: [http://arxiv.org/pdf/2510.19895v1](http://arxiv.org/pdf/2510.19895v1)**

> **作者:** Guoyun Zhang
>
> **摘要:** The integration of Large Language Models (LLMs) with optimization modeling offers a promising avenue for advancing decision-making in operations research (OR). Traditional optimization methods,such as linear programming, mixed integer programming, and simulation depend heavily on domain expertise to translate real-world problems into solvable mathematical models. While solvers like Gurobi and COPT are powerful, expert input remains essential for defining objectives, constraints, and variables. This research investigates the potential of LLMs, specifically the DeepSeek-R1 model, to bridge this formulation gap using natural language understanding and code generation. Although prior models like GPT-4, Claude, and Bard have shown strong performance in NLP and reasoning tasks, their high token costs and tendency toward hallucinations limit real-world applicability in supply chain contexts. In contrast, DeepSeek-R1, a cost-efficient and high-performing model trained with reinforcement learning, presents a viable alternative. Despite its success in benchmarks such as LiveCodeBench and Math-500, its effectiveness in applied OR scenarios remains under explored. This study systematically evaluates DeepSeek-R1 across four key OR benchmarks: NL4OPT, IndustryOR, EasyLP, and ComplexOR. Our methodology includes baseline assessments, the development of a hallucination taxonomy, and the application of mitigation strategies like LLM-as-a-Judge, Few-shot Learning (FSL), Tool Calling, and a Multi-agent Framework. These techniques aim to reduce hallucinations, enhance formulation accuracy, and better align model outputs with user intent.
>
---
#### [new 033] GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多跳问答中推理规划缺失与执行不一致的问题，提出GlobalRAG框架。通过分解问题为子目标、协同检索与推理、迭代优化证据，并引入规划质量与子目标完成奖励，结合渐进权重衰减策略，显著提升全局推理能力，在少量数据下超越现有方法。**

- **链接: [http://arxiv.org/pdf/2510.20548v1](http://arxiv.org/pdf/2510.20548v1)**

> **作者:** Jinchang Luo; Mingquan Cheng; Fan Wan; Ni Li; Xiaoling Xia; Shuangshuang Tian; Tingcheng Bian; Haiwei Wang; Haohuan Fu; Yan Tao
>
> **备注:** 8 pages, 3 figures, 4 tables
>
> **摘要:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.
>
---
#### [new 034] Robust Preference Alignment via Directional Neighborhood Consensus
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在人类偏好对齐中的鲁棒性问题，提出无需重训练的后处理方法RPS。通过方向邻域共识生成多样化响应并择优，提升模型对非主流偏好的适应能力，显著增强在稀疏偏好区域的性能表现。**

- **链接: [http://arxiv.org/pdf/2510.20498v1](http://arxiv.org/pdf/2510.20498v1)**

> **作者:** Ruochen Mao; Yuling Shi; Xiaodong Gu; Jiaheng Wei
>
> **备注:** Under review at ICLR 2026. 10 pages, 5 figures. Code and data available at https://github.com/rcmao/robust-preference-alignment
>
> **摘要:** Aligning large language models with human preferences is critical for creating reliable and controllable AI systems. A human preference can be visualized as a high-dimensional vector where different directions represent trade-offs between desired attributes (e.g., helpfulness vs. verbosity). Yet, because the training data often reflects dominant, average preferences, LLMs tend to perform well on common requests but fall short in specific, individual needs. This mismatch creates a preference coverage gap. Existing methods often address this through costly retraining, which may not be generalized to the full spectrum of diverse preferences. This brittleness means that when a user's request reflects a nuanced preference deviating from the training data's central tendency, model performance can degrade unpredictably. To address this challenge, we introduce Robust Preference Selection (RPS), a post-hoc, training-free method by leveraging directional neighborhood consensus. Instead of forcing a model to generate a response from a single, highly specific preference, RPS samples multiple responses from a local neighborhood of related preferences to create a superior candidate pool. It then selects the response that best aligns with the user's original intent. We provide a theoretical framework showing our neighborhood generation strategy is provably superior to a strong baseline that also samples multiple candidates. Comprehensive experiments across three distinct alignment paradigms (DPA, DPO, and SFT) demonstrate that RPS consistently improves robustness against this baseline, achieving win rates of up to 69% on challenging preferences from under-represented regions of the space without any model retraining. Our work presents a practical, theoretically-grounded solution for enhancing the reliability of preference-aligned models.
>
---
#### [new 035] VLSP 2025 MLQA-TSR Challenge: Vietnamese Multimodal Legal Question Answering on Traffic Sign Regulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VLSP 2025 MLQA-TSR挑战任务，聚焦越南语多模态交通标志法规问答。包含多模态法律检索与问答两个子任务，旨在推动越南语多模态法律文本处理研究，构建相关基准数据集。工作包括任务设计、数据集构建及模型评估，最佳结果为检索F2 64.55%、问答准确率86.30%。**

- **链接: [http://arxiv.org/pdf/2510.20381v1](http://arxiv.org/pdf/2510.20381v1)**

> **作者:** Son T. Luu; Trung Vo; Hiep Nguyen; Khanh Quoc Tran; Kiet Van Nguyen; Vu Tran; Ngan Luu-Thuy Nguyen; Le-Minh Nguyen
>
> **备注:** VLSP 2025 MLQA-TSR Share Task
>
> **摘要:** This paper presents the VLSP 2025 MLQA-TSR - the multimodal legal question answering on traffic sign regulation shared task at VLSP 2025. VLSP 2025 MLQA-TSR comprises two subtasks: multimodal legal retrieval and multimodal question answering. The goal is to advance research on Vietnamese multimodal legal text processing and to provide a benchmark dataset for building and evaluating intelligent systems in multimodal legal domains, with a focus on traffic sign regulation in Vietnam. The best-reported results on VLSP 2025 MLQA-TSR are an F2 score of 64.55% for multimodal legal retrieval and an accuracy of 86.30% for multimodal question answering.
>
---
#### [new 036] An Expert-grounded benchmark of General Purpose LLMs in LCA
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）在生命周期评估（LCA）中的可靠性问题，构建首个专家评审基准。通过17位专家对11个LLM在22项任务中的输出进行评估，发现37%响应存在错误，且部分模型幻觉率高达40%。研究揭示了LLM在准确性与可验证性上的风险，同时肯定其在解释质量与减轻工作负担方面的潜力。**

- **链接: [http://arxiv.org/pdf/2510.19886v1](http://arxiv.org/pdf/2510.19886v1)**

> **作者:** Artur Donaldson; Bharathan Balaji; Cajetan Oriekezie; Manish Kumar; Laure Patouillard
>
> **摘要:** Purpose: Artificial intelligence (AI), and in particular large language models (LLMs), are increasingly being explored as tools to support life cycle assessment (LCA). While demonstrations exist across environmental and social domains, systematic evidence on their reliability, robustness, and usability remains limited. This study provides the first expert-grounded benchmark of LLMs in LCA, addressing the absence of standardized evaluation frameworks in a field where no clear ground truth or consensus protocols exist. Methods: We evaluated eleven general-purpose LLMs, spanning both commercial and open-source families, across 22 LCA-related tasks. Seventeen experienced practitioners reviewed model outputs against criteria directly relevant to LCA practice, including scientific accuracy, explanation quality, robustness, verifiability, and adherence to instructions. We collected 168 expert reviews. Results: Experts judged 37% of responses to contain inaccurate or misleading information. Ratings of accuracy and quality of explanation were generally rated average or good on many models even smaller models, and format adherence was generally rated favourably. Hallucination rates varied significantly, with some models producing hallucinated citations at rates of up to 40%. There was no clear-cut distinction between ratings on open-weight versus closed-weight LLMs, with open-weight models outperforming or competing on par with closed-weight models on criteria such as accuracy and quality of explanation. Conclusion: These findings highlight the risks of applying LLMs na\"ively in LCA, such as when LLMs are treated as free-form oracles, while also showing benefits especially around quality of explanation and alleviating labour intensiveness of simple tasks. The use of general-purpose LLMs without grounding mechanisms presents ...
>
---
#### [new 037] Automated HIV Screening on Dutch EHR with Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出一种基于大语言模型的自动化HIV筛查方法，旨在利用电子病历中的非结构化文本（如临床笔记）识别高风险患者。针对传统机器学习仅依赖结构化数据而忽略文本信息的问题，该研究构建了新流程，有效提升筛查准确率并保持低假阴性率。**

- **链接: [http://arxiv.org/pdf/2510.19879v1](http://arxiv.org/pdf/2510.19879v1)**

> **作者:** Lang Zhou; Amrish Jhingoer; Yinghao Luo; Klaske Vliegenthart--Jongbloed; Carlijn Jordans; Ben Werkhoven; Tom Seinen; Erik van Mulligen; Casper Rokx; Yunlei Li
>
> **备注:** 28 pages, 6 figures
>
> **摘要:** Efficient screening and early diagnosis of HIV are critical for reducing onward transmission. Although large scale laboratory testing is not feasible, the widespread adoption of Electronic Health Records (EHRs) offers new opportunities to address this challenge. Existing research primarily focuses on applying machine learning methods to structured data, such as patient demographics, for improving HIV diagnosis. However, these approaches often overlook unstructured text data such as clinical notes, which potentially contain valuable information relevant to HIV risk. In this study, we propose a novel pipeline that leverages a Large Language Model (LLM) to analyze unstructured EHR text and determine a patient's eligibility for further HIV testing. Experimental results on clinical data from Erasmus University Medical Center Rotterdam demonstrate that our pipeline achieved high accuracy while maintaining a low false negative rate.
>
---
#### [new 038] Hierarchical Sequence Iteration for Heterogeneous Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多跳、异构证据源下的问答任务，提出HSEQ迭代框架。通过统一编码文本、表格与知识图谱为可逆层次序列，实现结构感知的渐进式证据收集，提升准确率与效率，支持跨模态统一策略与证据标准化，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.20505v1](http://arxiv.org/pdf/2510.20505v1)**

> **作者:** Ruiyi Yang; Hao Xue; Imran Razzak; Hakim Hacid; Flora D. Salim
>
> **备注:** 22 pages, 3 figures
>
> **摘要:** Retrieval-augmented generation (RAG) remains brittle on multi-step questions and heterogeneous evidence sources, trading accuracy against latency and token/tool budgets. This paper introducesHierarchical Sequence (HSEQ) Iteration for Heterogeneous Question Answering, a unified framework that (i) linearize documents, tables, and knowledge graphs into a reversible hierarchical sequence with lightweight structural tags, and (ii) perform structure-aware iteration to collect just-enough evidence before answer synthesis. A Head Agent provides guidance that leads retrieval, while an Iteration Agent selects and expands HSeq via structure-respecting actions (e.g., parent/child hops, table row/column neighbors, KG relations); Finally the head agent composes canonicalized evidence to genearte the final answer, with an optional refinement loop to resolve detected contradictions. Experiments on HotpotQA (text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1 gains over strong single-pass, multi-hop, and agentic RAG baselines with high efficiency. Besides, HSEQ exhibits three key advantages: (1) a format-agnostic unification that enables a single policy to operate across text, tables, and KGs without per-dataset specialization; (2) guided, budget-aware iteration that reduces unnecessary hops, tool calls, and tokens while preserving accuracy; and (3) evidence canonicalization for reliable QA, improving answers consistency and auditability.
>
---
#### [new 039] DeepWideSearch: Benchmarking Depth and Width in Agentic Information Seeking
- **分类: cs.CL**

- **简介: 该论文提出DeepWideSearch基准，旨在评估智能体在信息检索中同时实现深度推理与广度收集的能力。针对现有搜索代理难以兼顾深度与广度的问题，构建了220个跨15领域、需多跳推理的复杂问题，揭示当前模型仅2.39%成功率，并识别出四大失败模式，推动更强大信息搜寻代理的发展。**

- **链接: [http://arxiv.org/pdf/2510.20168v1](http://arxiv.org/pdf/2510.20168v1)**

> **作者:** Tian Lan; Bin Zhu; Qianghuai Jia; Junyang Ren; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **摘要:** Current search agents fundamentally lack the ability to simultaneously perform \textit{deep} reasoning over multi-hop retrieval and \textit{wide}-scale information collection-a critical deficiency for real-world applications like comprehensive market analysis and business development. To bridge this gap, we introduce DeepWideSearch, the first benchmark explicitly designed to evaluate agents to integrate depth and width in information seeking. In DeepWideSearch, agents must process a large volume of data, each requiring deep reasoning over multi-hop retrieval paths. Specifically, we propose two methods to converse established datasets, resulting in a curated collection of 220 questions spanning 15 diverse domains. Extensive experiments demonstrate that even state-of-the-art agents achieve only 2.39% average success rate on DeepWideSearch, highlighting the substantial challenge of integrating depth and width search in information-seeking tasks. Furthermore, our error analysis reveals four failure modes: lack of reflection, overreliance on internal knowledge, insufficient retrieval, and context overflow-exposing key limitations in current agent architectures. We publicly release DeepWideSearch to catalyze future research on more capable and robust information-seeking agents.
>
---
#### [new 040] ARC-Encoder: learning compressed text representations for large language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ARC-Encoder，一种用于大语言模型的上下文压缩编码器，通过将文本压缩为更少的连续表示来降低推理成本。解决长上下文带来的计算开销问题，无需微调或修改解码器，可适配多个模型，提升效率与通用性。**

- **链接: [http://arxiv.org/pdf/2510.20535v1](http://arxiv.org/pdf/2510.20535v1)**

> **作者:** Hippolyte Pilchen; Edouard Grave; Patrick Pérez
>
> **摘要:** Recent techniques such as retrieval-augmented generation or chain-of-thought reasoning have led to longer contexts and increased inference costs. Context compression techniques can reduce these costs, but the most effective approaches require fine-tuning the target model or even modifying its architecture. This can degrade its general abilities when not used for this specific purpose. Here we explore an alternative approach: an encoder that compresses the context into continuous representations which replace token embeddings in decoder LLMs. First, we perform a systematic study of training strategies and architecture choices for the encoder. Our findings led to the design of an Adaptable text Representations Compressor, named ARC-Encoder, which outputs $x$-times fewer continuous representations (typically $x\!\in\!\{4,8\}$) than text tokens. We evaluate ARC-Encoder across a variety of LLM usage scenarios, ranging from in-context learning to context window extension, on both instruct and base decoders. Results show that ARC-Encoder achieves state-of-the-art performance on several benchmarks while improving computational efficiency at inference. Finally, we demonstrate that our models can be adapted to multiple decoders simultaneously, allowing a single encoder to generalize across different decoder LLMs. This makes ARC-Encoder a flexible and efficient solution for portable encoders that work seamlessly with multiple LLMs. We release a training code at https://github.com/kyutai-labs/ARC-Encoder , fine-tuning dataset and pretrained models are available at https://huggingface.co/collections/kyutai/arc-encoders-68ee18787301407d60a57047 .
>
---
#### [new 041] Neural Diversity Regularizes Hallucinations in Small Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对小模型幻觉问题，提出神经多样性机制，通过并行去相关表示降低幻觉率。基于投资组合理论，证明神经多样性可显著减少幻觉，设计ND-LoRA方法实现这一目标，在不增加参数和数据的前提下，平均降低14.6%幻觉率，验证了神经多样性作为独立于参数与数据的第三类缩放维度的有效性。**

- **链接: [http://arxiv.org/pdf/2510.20690v1](http://arxiv.org/pdf/2510.20690v1)**

> **作者:** Kushal Chakrabarti; Nirmal Balachundhar
>
> **摘要:** Language models continue to hallucinate despite increases in parameters, compute, and data. We propose neural diversity -- decorrelated parallel representations -- as a principled mechanism that reduces hallucination rates at fixed parameter and data budgets. Inspired by portfolio theory, where uncorrelated assets reduce risk by $\sqrt{P}$, we prove hallucination probability is bounded by representational correlation: $P(H) \leq f(\sigma^2((1-\rho(P))/P + \rho(P)), \mu^2)$, which predicts that language models need an optimal amount of neurodiversity. To validate this, we introduce ND-LoRA (Neural Diversity Low-Rank Adaptation), combining parallel LoRA adapters with Barlow Twins regularization, and demonstrate that ND-LoRA reduces hallucinations by up to 25.6% (and 14.6% on average) without degrading general accuracy. Ablations show LoRA adapters and regularization act synergistically, causal interventions prove neurodiversity as the mediating factor and correlational analyses indicate scale: a 0.1% neural correlation increase is associated with a 3.8% hallucination increase. Finally, task-dependent optimality emerges: different tasks require different amounts of optimal neurodiversity. Together, our results highlight neural diversity as a third axis of scaling -- orthogonal to parameters and data -- to improve the reliability of language models at fixed budgets.
>
---
#### [new 042] Enhancing Reasoning Skills in Small Persian Medical Language Models Can Outperform Large-Scale Data Training
- **分类: cs.CL**

- **简介: 该论文聚焦于提升小规模波斯语医学语言模型的推理能力。针对波斯语医疗数据稀缺问题，通过RLAIF与DPO方法，构建包含正确与错误思维链的对比数据集，仅用约2.5万词元训练即超越更大规模模型，验证了高效推理训练的有效性。**

- **链接: [http://arxiv.org/pdf/2510.20059v1](http://arxiv.org/pdf/2510.20059v1)**

> **作者:** Mehrdad Ghassabi; Sadra Hakim; Hamidreza Baradaran Kashani; Pedram Rostami
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Enhancing reasoning capabilities in small language models is critical for specialized applications such as medical question answering, particularly in underrepresented languages like Persian. In this study, we employ Reinforcement Learning with AI Feedback (RLAIF) and Direct preference optimization (DPO) to improve the reasoning skills of a general-purpose Persian language model. To achieve this, we translated a multiple-choice medical question-answering dataset into Persian and used RLAIF to generate rejected-preferred answer pairs, which are essential for DPO training. By prompting both teacher and student models to produce Chain-of-Thought (CoT) reasoning responses, we compiled a dataset containing correct and incorrect reasoning trajectories. This dataset, comprising 2 million tokens in preferred answers and 2.5 million tokens in rejected ones, was used to train a baseline model, significantly enhancing its medical reasoning capabilities in Persian. Remarkably, the resulting model outperformed its predecessor, gaokerena-V, which was trained on approximately 57 million tokens, despite leveraging a much smaller dataset. These results highlight the efficiency and effectiveness of reasoning-focused training approaches in developing domain-specific language models with limited data availability.
>
---
#### [new 043] Simple Context Compression: Mean-Pooling and Multi-Ratio Training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型长上下文处理的高计算成本问题，提出一种轻量级均值池化压缩方法。通过简单有效的压缩策略，在保持高性能的同时降低计算开销，并探索多比率训练的可行性，验证了其在多种场景下的优越性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.20797v1](http://arxiv.org/pdf/2510.20797v1)**

> **作者:** Yair Feldman; Yoav Artzi
>
> **备注:** Code available at https://github.com/lil-lab/simple-context-compression
>
> **摘要:** A common strategy to reduce the computational costs of using long contexts in retrieval-augmented generation (RAG) with large language models (LLMs) is soft context compression, where the input sequence is transformed into a shorter continuous representation. We develop a lightweight and simple mean-pooling approach that consistently outperforms the widely used compression-tokens architecture, and study training the same compressor to output multiple compression ratios. We conduct extensive experiments across in-domain and out-of-domain QA datasets, as well as across model families, scales, and compression ratios. Overall, our simple mean-pooling approach achieves the strongest performance, with a relatively small drop when training for multiple compression ratios. More broadly though, across architectures and training regimes the trade-offs are more nuanced, illustrating the complex landscape of compression methods.
>
---
#### [new 044] Beyond Retrieval-Ranking: A Multi-Agent Cognitive Decision Framework for E-Commerce Search
- **分类: cs.CL**

- **简介: 该论文针对电商搜索中传统检索排序范式与用户多阶段认知决策不匹配的问题，提出多智能体认知决策框架MACDF。通过主动决策支持，解决复杂查询下的语义鸿沟、信息搜寻成本高及缺乏专业引导等问题，显著提升推荐准确率与用户满意度。**

- **链接: [http://arxiv.org/pdf/2510.20567v1](http://arxiv.org/pdf/2510.20567v1)**

> **作者:** Zhouwei Zhai; Mengxiang Chen; Haoyun Xia; Jin Li; Renquan Zhou; Min Yang
>
> **摘要:** The retrieval-ranking paradigm has long dominated e-commerce search, but its reliance on query-item matching fundamentally misaligns with multi-stage cognitive decision processes of platform users. This misalignment introduces critical limitations: semantic gaps in complex queries, high decision costs due to cross-platform information foraging, and the absence of professional shopping guidance. To address these issues, we propose a Multi-Agent Cognitive Decision Framework (MACDF), which shifts the paradigm from passive retrieval to proactive decision support. Extensive offline evaluations demonstrate MACDF's significant improvements in recommendation accuracy and user satisfaction, particularly for complex queries involving negation, multi-constraint, or reasoning demands. Online A/B testing on JD search platform confirms its practical efficacy. This work highlights the transformative potential of multi-agent cognitive systems in redefining e-commerce search.
>
---
#### [new 045] Are Stereotypes Leading LLMs' Zero-Shot Stance Detection ?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在零样本立场检测中的刻板印象偏见问题。针对立场检测任务，通过自动标注数据集中的方言与文本复杂度，发现模型存在显著偏见，如将低复杂度文本关联于支持大麻观点，将非裔美式英语关联于反对特朗普。**

- **链接: [http://arxiv.org/pdf/2510.20154v1](http://arxiv.org/pdf/2510.20154v1)**

> **作者:** Anthony Dubreuil; Antoine Gourru; Christine Largeron; Amine Trabelsi
>
> **备注:** Accepted in EMNLP 2025 (Main)
>
> **摘要:** Large Language Models inherit stereotypes from their pretraining data, leading to biased behavior toward certain social groups in many Natural Language Processing tasks, such as hateful speech detection or sentiment analysis. Surprisingly, the evaluation of this kind of bias in stance detection methods has been largely overlooked by the community. Stance Detection involves labeling a statement as being against, in favor, or neutral towards a specific target and is among the most sensitive NLP tasks, as it often relates to political leanings. In this paper, we focus on the bias of Large Language Models when performing stance detection in a zero-shot setting. We automatically annotate posts in pre-existing stance detection datasets with two attributes: dialect or vernacular of a specific group and text complexity/readability, to investigate whether these attributes influence the model's stance detection decisions. Our results show that LLMs exhibit significant stereotypes in stance detection tasks, such as incorrectly associating pro-marijuana views with low text complexity and African American dialect with opposition to Donald Trump.
>
---
#### [new 046] Mask and You Shall Receive: Optimizing Masked Language Modeling For Pretraining BabyLMs
- **分类: cs.CL**

- **简介: 该论文针对预训练小模型（BabyLMs）的掩码语言建模问题，提出一种自适应掩码策略，根据模型预测能力动态调整被掩码词的概率。结合子词嵌入，提升模型对形态变化的泛化能力。实验表明，该方法在(Super)GLUE任务上显著优于标准MLM，且在严格小规模赛道中超越基线。**

- **链接: [http://arxiv.org/pdf/2510.20475v1](http://arxiv.org/pdf/2510.20475v1)**

> **作者:** Lukas Edman; Alexander Fraser
>
> **备注:** Submission to the 2025 BabyLM Challenge
>
> **摘要:** We describe our strategy for the 2025 edition of the BabyLM Challenge. Our main contribution is that of an improved form of Masked Language Modeling (MLM), which adapts the probabilities of the tokens masked according to the model's ability to predict them. The results show a substantial increase in performance on (Super)GLUE tasks over the standard MLM. We also incorporate sub-token embeddings, finding that this increases the model's morphological generalization capabilities. Our submission beats the baseline in the strict-small track.
>
---
#### [new 047] Leveraging the Power of Large Language Models in Entity Linking via Adaptive Routing and Targeted Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对实体链接任务，解决传统方法依赖大量标注数据与昂贵模型微调的问题。提出ARTER框架，通过自适应路由将提及分为易难案例，分别用轻量链接器与针对性LLM推理处理，显著提升效率与性能，在保持高准确率的同时减少一半以上LLM调用。**

- **链接: [http://arxiv.org/pdf/2510.20098v1](http://arxiv.org/pdf/2510.20098v1)**

> **作者:** Yajie Li; Albert Galimov; Mitra Datta Ganapaneni; Pujitha Thejaswi; De Meng; Priyanshu Kumar; Saloni Potdar
>
> **摘要:** Entity Linking (EL) has traditionally relied on large annotated datasets and extensive model fine-tuning. While recent few-shot methods leverage large language models (LLMs) through prompting to reduce training requirements, they often suffer from inefficiencies due to expensive LLM-based reasoning. ARTER (Adaptive Routing and Targeted Entity Reasoning) presents a structured pipeline that achieves high performance without deep fine-tuning by strategically combining candidate generation, context-based scoring, adaptive routing, and selective reasoning. ARTER computes a small set of complementary signals(both embedding and LLM-based) over the retrieved candidates to categorize contextual mentions into easy and hard cases. The cases are then handled by a low-computational entity linker (e.g. ReFinED) and more expensive targeted LLM-based reasoning respectively. On standard benchmarks, ARTER outperforms ReFinED by up to +4.47%, with an average gain of +2.53% on 5 out of 6 datasets, and performs comparably to pipelines using LLM-based reasoning for all mentions, while being as twice as efficient in terms of the number of LLM tokens.
>
---
#### [new 048] A Use-Case Specific Dataset for Measuring Dimensions of Responsible Performance in LLM-generated Text
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文针对大语言模型（LLM）在负责任AI维度（如公平性）的评估问题，提出一种基于真实应用场景（生成产品描述）的特定用例数据集。通过融合性别化形容词与产品类别，构建带标签的提示数据，用于检测LLM在质量、真实性、安全性和公平性方面的缺陷，为负责任的LLM评估提供可操作的数据资源与方法。**

- **链接: [http://arxiv.org/pdf/2510.20782v1](http://arxiv.org/pdf/2510.20782v1)**

> **作者:** Alicia Sagae; Chia-Jung Lee; Sandeep Avula; Brandon Dang; Vanessa Murdock
>
> **备注:** 24 pages with 3 figures, to appear in Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)
>
> **摘要:** Current methods for evaluating large language models (LLMs) typically focus on high-level tasks such as text generation, without targeting a particular AI application. This approach is not sufficient for evaluating LLMs for Responsible AI dimensions like fairness, since protected attributes that are highly relevant in one application may be less relevant in another. In this work, we construct a dataset that is driven by a real-world application (generate a plain-text product description, given a list of product features), parameterized by fairness attributes intersected with gendered adjectives and product categories, yielding a rich set of labeled prompts. We show how to use the data to identify quality, veracity, safety, and fairness gaps in LLMs, contributing a proposal for LLM evaluation paired with a concrete resource for the research community.
>
---
#### [new 049] Structure-Conditional Minimum Bayes Risk Decoding
- **分类: cs.CL**

- **简介: 该论文针对开放任务中最小贝叶斯风险（MBR）解码因忽略生成结果潜在结构而效果不佳的问题，提出三种轻量级改进的效用函数，增强对对话意图、情感和响应结构等隐式结构的敏感性。通过构建结构化数据集与新评估指标验证，所提方法显著提升生成质量，在指令遵循任务上最高提升13.7%胜率。**

- **链接: [http://arxiv.org/pdf/2510.20700v1](http://arxiv.org/pdf/2510.20700v1)**

> **作者:** Bryan Eikema; Anna Rutkiewicz; Mario Giulianelli
>
> **备注:** EMNLP 2025 Camera-Ready
>
> **摘要:** Minimum Bayes Risk (MBR) decoding has seen renewed interest as an alternative to traditional generation strategies. While MBR has proven effective in machine translation, where the variability of a language model's outcome space is naturally constrained, it may face challenges in more open-ended tasks such as dialogue or instruction-following. We hypothesise that in such settings, applying MBR with standard similarity-based utility functions may result in selecting responses that are broadly representative of the model's distribution, yet sub-optimal with respect to any particular grouping of generations that share an underlying latent structure. In this work, we introduce three lightweight adaptations to the utility function, designed to make MBR more sensitive to structural variability in the outcome space. To test our hypothesis, we curate a dataset capturing three representative types of latent structure: dialogue act, emotion, and response structure (e.g., a sentence, a paragraph, or a list). We further propose two metrics to evaluate the structural optimality of MBR. Our analysis demonstrates that common similarity-based utility functions fall short by these metrics. In contrast, our proposed adaptations considerably improve structural optimality. Finally, we evaluate our approaches on real-world instruction-following benchmarks, AlpacaEval and MT-Bench, and show that increased structural sensitivity improves generation quality by up to 13.7 percentage points in win rate.
>
---
#### [new 050] Dialogue Is Not Enough to Make a Communicative BabyLM (But Neither Is Developmentally Inspired Reinforcement Learning)
- **分类: cs.CL**

- **简介: 该论文研究对话数据预训练对小型语言模型（BabyLM）沟通能力的影响。旨在探究仅用对话数据是否足以使模型具备沟通能力。作者构建了llamalogue模型，并通过DPO等策略优化，发现其在对话延续任务上表现优异，但标准基准测试中表现不佳，表明单纯对话训练不足，需结合其他方法提升沟通能力。**

- **链接: [http://arxiv.org/pdf/2510.20358v1](http://arxiv.org/pdf/2510.20358v1)**

> **作者:** Francesca Padovani; Bastian Bunzeck; Manar Ali; Omar Momen; Arianna Bisazza; Hendrik Buschmeier; Sina Zarrieß
>
> **摘要:** We investigate whether pre-training exclusively on dialogue data results in formally and functionally apt small language models. Based on this pre-trained llamalogue model, we employ a variety of fine-tuning strategies to enforce "more communicative" text generations by our models. Although our models underperform on most standard BabyLM benchmarks, they excel at dialogue continuation prediction in a minimal pair setting. While PPO fine-tuning has mixed to adversarial effects on our models, DPO fine-tuning further improves their performance on our custom dialogue benchmark.
>
---
#### [new 051] Steering Evaluation-Aware Language Models To Act Like They Are Deployed
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在安全评估中因识别评估线索而伪装合规的问题，提出通过激活空间的定向引导（steering vector）抑制其评估意识。研究通过两阶段训练构建评估感知模型，并验证了无需重新训练即可使模型在评估时表现如部署状态。任务为提升评估可靠性，方法为基于原始模型的激活调控。**

- **链接: [http://arxiv.org/pdf/2510.20487v1](http://arxiv.org/pdf/2510.20487v1)**

> **作者:** Tim Tian Hua; Andrew Qin; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on documents with factual descriptions of the model (1) using Python type hints during evaluation but not during deployment and (2) recognizing that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. However, this gap can only be observed by removing the evaluation cue. We find that activation steering can suppress evaluation awareness and make the model act like it is deployed even when the cue is present. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed.
>
---
#### [new 052] Improving Transfer Learning for Sequence Labeling Tasks by Adapting Pre-trained Neural Language Models
- **分类: cs.CL**

- **简介: 该论文聚焦序列标注任务中的迁移学习优化。针对预训练语言模型在序列标注中表现不足的问题，提出三种改进：多任务融合领域无关信号、双向信息流的架构修改、基于生成式上下文微调与响应适应的框架，显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.20033v1](http://arxiv.org/pdf/2510.20033v1)**

> **作者:** David Dukić
>
> **摘要:** This doctoral thesis improves the transfer learning for sequence labeling tasks by adapting pre-trained neural language models. The proposed improvements in transfer learning involve introducing a multi-task model that incorporates an additional signal, a method based on architectural modifications in autoregressive large language models, and a sequence labeling framework for autoregressive large language models utilizing supervised in-context fine-tuning combined with response-oriented adaptation strategies. The first improvement is given in the context of domain transfer for the event trigger detection task. The domain transfer of the event trigger detection task can be improved by incorporating an additional signal obtained from a domain-independent text processing system into a multi-task model. The second improvement involves modifying the model's architecture. For that purpose, a method is proposed to enable bidirectional information flow across layers of autoregressive large language models. The third improvement utilizes autoregressive large language models as text generators through a generative supervised in-context fine-tuning framework. The proposed model, method, and framework demonstrate that pre-trained neural language models achieve their best performance on sequence labeling tasks when adapted through targeted transfer learning paradigms.
>
---
#### [new 053] Stuck in the Matrix: Probing Spatial Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文探究大语言模型在文本输入下的空间推理能力，针对五类网格环境任务（如象限识别、几何变换等）测试其空间理解与多步计算能力。结果表明，随着复杂度提升，模型性能显著下降，暴露其缺乏稳健的空间表征，揭示了语言与空间推理间的鸿沟。**

- **链接: [http://arxiv.org/pdf/2510.20198v1](http://arxiv.org/pdf/2510.20198v1)**

> **作者:** Maggie Bai; Ava Kim Cohen; Eleanor Koss; Charlie Lichtenbaum
>
> **备注:** 20 pages, 24 figures
>
> **摘要:** This paper explores the spatial reasoning capability of large language models (LLMs) over textual input through a suite of five tasks aimed at probing their spatial understanding and computational abilities. The models were tested on both fundamental spatial reasoning and multi-step problem-solving within structured grid-based environments using tasks such as quadrant identification, geometric transformations, distance evaluation, word searches, and tile sliding. Each task was scaled in complexity through increasing grid dimensions, requiring models to extend beyond simple pattern recognition into abstract spatial reasoning. Our results reveal that while LLMs demonstrate moderate success in all tasks with small complexity and size, performance drops off rapidly as scale increases, with an average loss in accuracy of 42.7%, and reaching as high as 84%. Every test that began with over 50% accuracy showed a loss of at least 48%, illustrating the consistent nature of the deterioration. Furthermore, their struggles with scaling complexity hint at a lack of robust spatial representations in their underlying architectures. This paper underscores the gap between linguistic and spatial reasoning in LLMs, offering insights into their current limitations, and laying the groundwork for future integrative benchmarks at the intersection of language and geometry.
>
---
#### [new 054] On the Detectability of LLM-Generated Text: What Exactly Is LLM-Generated Text?
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文聚焦于大语言模型生成文本的可检测性问题，指出当前缺乏对“LLM生成文本”的明确定义，且现有检测方法未涵盖真实应用场景的多样性。研究揭示了检测结果的局限性，强调应将检测结果视为参考而非决定性依据。**

- **链接: [http://arxiv.org/pdf/2510.20810v1](http://arxiv.org/pdf/2510.20810v1)**

> **作者:** Mingmeng Geng; Thierry Poibeau
>
> **摘要:** With the widespread use of large language models (LLMs), many researchers have turned their attention to detecting text generated by them. However, there is no consistent or precise definition of their target, namely "LLM-generated text". Differences in usage scenarios and the diversity of LLMs further increase the difficulty of detection. What is commonly regarded as the detecting target usually represents only a subset of the text that LLMs can potentially produce. Human edits to LLM outputs, together with the subtle influences that LLMs exert on their users, are blurring the line between LLM-generated and human-written text. Existing benchmarks and evaluation approaches do not adequately address the various conditions in real-world detector applications. Hence, the numerical results of detectors are often misunderstood, and their significance is diminishing. Therefore, detectors remain useful under specific conditions, but their results should be interpreted only as references rather than decisive indicators.
>
---
#### [new 055] \textsc{CantoNLU}: A benchmark for Cantonese natural language understanding
- **分类: cs.CL**

- **简介: 该论文提出CantoNLU基准，针对粤语自然语言理解资源匮乏问题，涵盖七类语法与语义任务。通过构建数据集并评估多种模型，发现粤语微调模型表现最优，且直接迁移在数据稀缺时仍有效，推动粤语NLP研究发展。**

- **链接: [http://arxiv.org/pdf/2510.20670v1](http://arxiv.org/pdf/2510.20670v1)**

> **作者:** Junghyun Min; York Hay Ng; Sophia Chan; Helena Shunhua Zhao; En-Shiun Annie Lee
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** Cantonese, although spoken by millions, remains under-resourced due to policy and diglossia. To address this scarcity of evaluation frameworks for Cantonese, we introduce \textsc{\textbf{CantoNLU}}, a benchmark for Cantonese natural language understanding (NLU). This novel benchmark spans seven tasks covering syntax and semantics, including word sense disambiguation, linguistic acceptability judgment, language detection, natural language inference, sentiment analysis, part-of-speech tagging, and dependency parsing. In addition to the benchmark, we provide model baseline performance across a set of models: a Mandarin model without Cantonese training, two Cantonese-adapted models obtained by continual pre-training a Mandarin model on Cantonese text, and a monolingual Cantonese model trained from scratch. Results show that Cantonese-adapted models perform best overall, while monolingual models perform better on syntactic tasks. Mandarin models remain competitive in certain settings, indicating that direct transfer may be sufficient when Cantonese domain data is scarce. We release all datasets, code, and model weights to facilitate future research in Cantonese NLP.
>
---
#### [new 056] User Perceptions of Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究用户对LLM在隐私敏感场景下响应的隐私保护与有用性感知。针对现有评估依赖代理LLM、忽视真实用户感知的问题，通过94名参与者对90个场景的实验，发现用户间评价一致性低，而代理LLM间一致高，表明用户感知具个体差异，需开展以用户为中心的评估研究。**

- **链接: [http://arxiv.org/pdf/2510.20721v1](http://arxiv.org/pdf/2510.20721v1)**

> **作者:** Xiaoyuan Wu; Roshni Kaushik; Wenkai Li; Lujo Bauer; Koichi Onoue
>
> **摘要:** Large language models (LLMs) have seen rapid adoption for tasks such as drafting emails, summarizing meetings, and answering health questions. In such uses, users may need to share private information (e.g., health records, contact details). To evaluate LLMs' ability to identify and redact such private information, prior work developed benchmarks (e.g., ConfAIde, PrivacyLens) with real-life scenarios. Using these benchmarks, researchers have found that LLMs sometimes fail to keep secrets private when responding to complex tasks (e.g., leaking employee salaries in meeting summaries). However, these evaluations rely on LLMs (proxy LLMs) to gauge compliance with privacy norms, overlooking real users' perceptions. Moreover, prior work primarily focused on the privacy-preservation quality of responses, without investigating nuanced differences in helpfulness. To understand how users perceive the privacy-preservation quality and helpfulness of LLM responses to privacy-sensitive scenarios, we conducted a user study with 94 participants using 90 scenarios from PrivacyLens. We found that, when evaluating identical responses to the same scenario, users showed low agreement with each other on the privacy-preservation quality and helpfulness of the LLM response. Further, we found high agreement among five proxy LLMs, while each individual LLM had low correlation with users' evaluations. These results indicate that the privacy and helpfulness of LLM responses are often specific to individuals, and proxy LLMs are poor estimates of how real users would perceive these responses in privacy-sensitive scenarios. Our results suggest the need to conduct user-centered studies on measuring LLMs' ability to help users while preserving privacy. Additionally, future research could investigate ways to improve the alignment between proxy LLMs and users for better estimation of users' perceived privacy and utility.
>
---
#### [new 057] Can They Dixit? Yes they Can! Dixit as a Playground for Multimodal Language Model Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出用游戏Dixit作为多模态大模型的评估框架，解决传统评测静态、主观、易被误导的问题。通过游戏机制综合考察模型的生成、推理与策略能力，实验表明其评估结果与主流基准高度一致，且揭示了模型与人类策略差异及改进方向。**

- **链接: [http://arxiv.org/pdf/2510.19892v1](http://arxiv.org/pdf/2510.19892v1)**

> **作者:** Nishant Balepur; Dang Nguyen; Dayeon Ki
>
> **备注:** Accepted as a Spotlight paper at the EMNLP 2025 Wordplay Workshop
>
> **摘要:** Multi-modal large language models (MLMs) are often assessed on static, individual benchmarks -- which cannot jointly assess MLM capabilities in a single task -- or rely on human or model pairwise comparisons -- which is highly subjective, expensive, and allows models to exploit superficial shortcuts (e.g., verbosity) to inflate their win-rates. To overcome these issues, we propose game-based evaluations to holistically assess MLM capabilities. Games require multiple abilities for players to win, are inherently competitive, and are governed by fix, objective rules, and makes evaluation more engaging, providing a robust framework to address the aforementioned challenges. We manifest this evaluation specifically through Dixit, a fantasy card game where players must generate captions for a card that trick some, but not all players, into selecting the played card. Our quantitative experiments with five MLMs show Dixit win-rate rankings are perfectly correlated with those on popular MLM benchmarks, while games between human and MLM players in Dixit reveal several differences between agent strategies and areas of improvement for MLM reasoning.
>
---
#### [new 058] Decoding-Free Sampling Strategies for LLM Marginalization
- **分类: cs.CL; I.2.7**

- **简介: 该论文针对大语言模型推理中因子词分词导致的输出概率评估偏差问题，提出无需解码的采样策略，通过低成本采样实现对文本所有可能分词方式的概率边际化近似，显著提升效率与准确性，应用于下游推理任务。**

- **链接: [http://arxiv.org/pdf/2510.20208v1](http://arxiv.org/pdf/2510.20208v1)**

> **作者:** David Pohl; Marco Cognetta; Junyoung Lee; Naoaki Okazaki
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Modern language models operate on subword-tokenized text in order to make a trade-off between model size, inference speed, and vocabulary coverage. A side effect of this is that, during inference, models are evaluated by measuring the probability of only the specific tokenization produced as the output, despite there being many possible ways to represent the same text with a subword vocabulary. Recent studies have argued instead for evaluating LLMs by marginalization - the probability mass of all tokenizations of a given text. Marginalization is difficult due to the number of possible tokenizations of a text, so often approximate marginalization is done via sampling. However, a downside of sampling is that an expensive generation step must be performed by the LLM for each sample, which limits the number of samples that can be acquired given a runtime budget, and therefore also the accuracy of the approximation. Since computing the probability of a sequence given the tokenization is relatively cheap compared to actually generating it, we investigate sampling strategies that are decoding-free - they require no generation from the LLM, instead relying entirely on extremely cheap sampling strategies that are model and tokenizer agnostic. We investigate the approximation quality and speed of decoding-free sampling strategies for a number of open models to find that they provide sufficiently accurate marginal estimates at a small fraction of the runtime cost and demonstrate its use on a set of downstream inference tasks.
>
---
#### [new 059] FreeChunker: A Cross-Granularity Chunking Framework
- **分类: cs.CL**

- **简介: 该论文针对RAG系统中固定粒度分块导致的适应性差问题，提出FreeChunker框架。通过将句子作为原子单元，实现跨粒度灵活检索，避免静态边界检测，显著提升查询适应性与计算效率。实验表明其在长文本检索任务中性能更优。**

- **链接: [http://arxiv.org/pdf/2510.20356v1](http://arxiv.org/pdf/2510.20356v1)**

> **作者:** Wenxuan Zhang; Yuan-Hao Jiang; Yonghe Wu
>
> **备注:** Submitted to arXiv, October 2025
>
> **摘要:** Chunking strategies significantly impact the effectiveness of Retrieval-Augmented Generation (RAG) systems. Existing methods operate within fixed-granularity paradigms that rely on static boundary identification, limiting their adaptability to diverse query requirements. This paper presents FreeChunker, a Cross-Granularity Encoding Framework that fundamentally transforms the traditional chunking paradigm: the framework treats sentences as atomic units and shifts from static chunk segmentation to flexible retrieval supporting arbitrary sentence combinations. This paradigm shift not only significantly reduces the computational overhead required for semantic boundary detection but also enhances adaptability to complex queries. Experimental evaluation on LongBench V2 demonstrates that FreeChunker achieves superior retrieval performance compared to traditional chunking methods, while significantly outperforming existing approaches in computational efficiency.
>
---
#### [new 060] From Denoising to Refining: A Corrective Framework for Vision-Language Diffusion Model
- **分类: cs.CL**

- **简介: 该论文针对视觉语言扩散模型中的训练-推理差异问题，提出ReDiff框架，通过两阶段训练实现模型自我修正。旨在解决并行解码时错误传播导致的语义幻觉与语法错误，使模型能主动识别并修正自身生成缺陷，显著提升生成内容的准确性和连贯性。**

- **链接: [http://arxiv.org/pdf/2510.19871v1](http://arxiv.org/pdf/2510.19871v1)**

> **作者:** Yatai Ji; Teng Wang; Yuying Ge; Zhiheng Liu; Sidi Yang; Ying Shan; Ping Luo
>
> **摘要:** Discrete diffusion models have emerged as a promising direction for vision-language tasks, offering bidirectional context modeling and theoretical parallelization. However, their practical application is severely hindered by a train-inference discrepancy, which leads to catastrophic error cascades: initial token errors during parallel decoding pollute the generation context, triggering a chain reaction of compounding errors and leading to syntactic errors and semantic hallucinations. To address this fundamental challenge, we reframe the generation process from passive denoising to active refining. We introduce ReDiff, a refining-enhanced diffusion framework that teaches the model to identify and correct its own errors. Our approach features a two-stage training process: first, we instill a foundational revision capability by training the model to revise synthetic errors; second, we implement a novel online self-correction loop where the model is explicitly trained to revise its own flawed drafts by learning from an expert's corrections. This mistake-driven learning endows the model with the crucial ability to revisit and refine its already generated output, effectively breaking the error cascade. Extensive experiments demonstrate that ReDiff significantly improves the coherence and factual accuracy of generated content, enabling stable and efficient parallel generation far superior to traditional denoising methods. Our codes and models are available at https://rediff-hku.github.io/.
>
---
#### [new 061] CreativityPrism: A Holistic Benchmark for Large Language Model Creativity
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CreativityPrism框架，用于全面评估大语言模型的创造力。针对现有评估方法碎片化、缺乏统一标准的问题，将创造力分解为质量、新颖性和多样性三维度，设计九项任务与二十项指标，评估17个模型。结果揭示模型在不同任务间表现差异显著，表明创造力难以跨任务泛化，强调需建立整体评价体系。**

- **链接: [http://arxiv.org/pdf/2510.20091v1](http://arxiv.org/pdf/2510.20091v1)**

> **作者:** Zhaoyi Joey Hou; Bowei Alvin Zhang; Yining Lu; Bhiman Kumar Baghel; Anneliese Brei; Ximing Lu; Meng Jiang; Faeze Brahman; Snigdha Chaturvedi; Haw-Shiuan Chang; Daniel Khashabi; Xiang Lorraine Li
>
> **摘要:** Creativity is often seen as a hallmark of human intelligence. While large language models (LLMs) are increasingly perceived as producing creative text, there is still no holistic framework to evaluate their creativity across diverse scenarios. Existing evaluation methods remain fragmented, with dramatic variation across domains and tasks, largely due to differing definitions and measurements of creativity. Inspired by the hypothesis that creativity is not one fixed idea, we propose CreativityPrism, an evaluation analysis framework that decomposes creativity into three dimensions: quality, novelty, and diversity. CreativityPrism incorporates nine tasks, three domains, i.e., divergent thinking, creative writing, and logical reasoning, and twenty evaluation metrics, which measure each dimension in task-specific, unique ways. We evaluate 17 state-of-the-art (SoTA) proprietary and open-sourced LLMs on CreativityPrism and analyze the performance correlations among different metrics and task domains. Our results reveal a notable gap between proprietary and open-source models. Overall, model performance tends to be highly correlated across tasks within the same domain and less so across different domains. Among evaluation dimensions, diversity and quality metrics show strong correlations - models that perform well on one often excel on the other - whereas novelty exhibits much weaker correlation with either. These findings support our hypothesis that strong performance in one creativity task or dimension does not necessarily generalize to others, underscoring the need for a holistic evaluation of LLM creativity.
>
---
#### [new 062] LM-mixup: Text Data Augmentation via Language Model based Mixup
- **分类: cs.CL**

- **简介: 该论文针对指令微调中高质量数据稀缺的问题，提出指令蒸馏任务与LM-Mixup方法。通过构建144K样本的低质-高质配对数据集MIXTURE，利用监督微调与强化学习优化，融合质量、语义一致性和格式合规性奖励，实现低质数据的有效增强。实验表明，仅用3%的蒸馏数据即可超越全量训练及现有最优方法。**

- **链接: [http://arxiv.org/pdf/2510.20449v1](http://arxiv.org/pdf/2510.20449v1)**

> **作者:** Zhijie Deng; Zhouan Shen; Ling Li; Yao Zhou; Zhaowei Zhu; Yanji He; Wei Wang; Jiaheng Wei
>
> **摘要:** Instruction tuning is crucial for aligning Large Language Models (LLMs), yet the quality of instruction-following data varies significantly. While high-quality data is paramount, it is often scarce; conversely, abundant low-quality data is frequently discarded, leading to substantial information loss. Existing data augmentation methods struggle to augment this low-quality data effectively, and the evaluation of such techniques remains poorly defined. To address this, we formally define the task of Instruction Distillation: distilling multiple low-quality and redundant inputs into high-quality and coherent instruction-output pairs. Specifically, we introduce a comprehensive data construction pipeline to create MIXTURE, a 144K-sample dataset pairing low-quality or semantically redundant imperfect instruction clusters with their high-quality distillations. We then introduce LM-Mixup, by first performing supervised fine-tuning on MIXTURE and then optimizing it with reinforcement learning. This process uses three complementary reward signals: quality, semantic alignment, and format compliance, via Group Relative Policy Optimization (GRPO). We demonstrate that LM-Mixup effectively augments imperfect datasets: fine-tuning LLMs on its distilled data, which accounts for only about 3% of the entire dataset, not only surpasses full-dataset training but also competes with state-of-the-art high-quality data selection methods across multiple benchmarks. Our work establishes that low-quality data is a valuable resource when properly distilled and augmented with LM-Mixup, significantly enhancing the efficiency and performance of instruction-tuned LLMs.
>
---
#### [new 063] The Impact of Negated Text on Hallucination with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究负向文本对大语言模型幻觉的影响，属于幻觉检测任务。针对负向表达下模型识别幻觉能力下降的问题，构建了NegHalu数据集，通过实验发现模型在负向语境中难以准确判断幻觉，且存在逻辑不一致问题，揭示了其内部处理机制的挑战。**

- **链接: [http://arxiv.org/pdf/2510.20375v1](http://arxiv.org/pdf/2510.20375v1)**

> **作者:** Jaehyung Seo; Hyeonseok Moon; Heuiseok Lim
>
> **备注:** Accepted to the EMNLP 2025
>
> **摘要:** Recent studies on hallucination in large language models (LLMs) have been actively progressing in natural language processing. However, the impact of negated text on hallucination with LLMs remains largely unexplored. In this paper, we set three important yet unanswered research questions and aim to address them. To derive the answers, we investigate whether LLMs can recognize contextual shifts caused by negation and still reliably distinguish hallucinations comparable to affirmative cases. We also design the NegHalu dataset by reconstructing existing hallucination detection datasets with negated expressions. Our experiments demonstrate that LLMs struggle to detect hallucinations in negated text effectively, often producing logically inconsistent or unfaithful judgments. Moreover, we trace the internal state of LLMs as they process negated inputs at the token level and reveal the challenges of mitigating their unintended effects.
>
---
#### [new 064] Evaluating Latent Knowledge of Public Tabular Datasets in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在表格数据推理任务中的潜在知识。针对现有评估中因数据泄露导致的偏差问题，通过控制实验发现：当表格含语义线索时模型表现优异，去除线索后性能骤降，表明其能力部分源于对公开数据的记忆而非真实推理。研究呼吁改进评估方法以区分记忆与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.20351v1](http://arxiv.org/pdf/2510.20351v1)**

> **作者:** Matteo Silvestri; Flavio Giorgi; Fabrizio Silvestri; Gabriele Tolomei
>
> **摘要:** Large Language Models (LLMs) are increasingly evaluated on their ability to reason over structured data, yet such assessments often overlook a crucial confound: dataset contamination. In this work, we investigate whether LLMs exhibit prior knowledge of widely used tabular benchmarks such as Adult Income, Titanic, and others. Through a series of controlled probing experiments, we reveal that contamination effects emerge exclusively for datasets containing strong semantic cues-for instance, meaningful column names or interpretable value categories. In contrast, when such cues are removed or randomized, performance sharply declines to near-random levels. These findings suggest that LLMs' apparent competence on tabular reasoning tasks may, in part, reflect memorization of publicly available datasets rather than genuine generalization. We discuss implications for evaluation protocols and propose strategies to disentangle semantic leakage from authentic reasoning ability in future LLM assessments.
>
---
#### [new 065] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对信息密集型图像的视觉问答任务，解决复杂布局中关键线索定位难与多跳推理效率低的问题。提出无需训练的Speculative Verdict框架，通过轻量级草案专家生成多样化推理路径，由强模型综合并筛选共识路径，实现高效准确的推理。**

- **链接: [http://arxiv.org/pdf/2510.20812v1](http://arxiv.org/pdf/2510.20812v1)**

> **作者:** Yuhan Liu; Lianhui Qin; Shengjie Wang
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence. We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers. To further improve efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict. Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from multiple partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines. Code is available at https://github.com/Tinaliu0123/speculative-verdict
>
---
#### [new 066] Decoding the Ear: A Framework for Objectifying Expressiveness from Human Preference Through Efficient Alignment
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文针对语音合成中表达性不足的问题，提出DeEAR框架，通过情感、韵律、自然性三维度将人类偏好转化为客观评分，实现高效精准评估。仅用500样本即达高相关性（SRCC=0.86），支持模型对比与数据优化，显著提升合成语音表达性。**

- **链接: [http://arxiv.org/pdf/2510.20513v1](http://arxiv.org/pdf/2510.20513v1)**

> **作者:** Zhiyu Lin; Jingwen Yang; Jiale Zhao; Meng Liu; Sunzhu Li; Benyou Wang
>
> **备注:** Submitted to ICASSP 2026. Demos and codes are available at https://github.com/FreedomIntelligence/ExpressiveSpeech
>
> **摘要:** Recent speech-to-speech (S2S) models generate intelligible speech but still lack natural expressiveness, largely due to the absence of a reliable evaluation metric. Existing approaches, such as subjective MOS ratings, low-level acoustic features, and emotion recognition are costly, limited, or incomplete. To address this, we present DeEAR (Decoding the Expressive Preference of eAR), a framework that converts human preference for speech expressiveness into an objective score. Grounded in phonetics and psychology, DeEAR evaluates speech across three dimensions: Emotion, Prosody, and Spontaneity, achieving strong alignment with human perception (Spearman's Rank Correlation Coefficient, SRCC = 0.86) using fewer than 500 annotated samples. Beyond reliable scoring, DeEAR enables fair benchmarking and targeted data curation. It not only distinguishes expressiveness gaps across S2S models but also selects 14K expressive utterances to form ExpressiveSpeech, which improves the expressive score (from 2.0 to 23.4 on a 100-point scale) of S2S models. Demos and codes are available at https://github.com/FreedomIntelligence/ExpressiveSpeech
>
---
#### [new 067] Multimedia-Aware Question Answering: A Review of Retrieval and Cross-Modal Reasoning Architectures
- **分类: cs.IR; cs.CL; cs.CV; cs.LG**

- **简介: 该论文聚焦多媒体感知问答任务，旨在解决多模态数据（图像、音频、视频）下查询与内容的对齐难题。通过综述检索增强型QA架构，分析跨模态融合与答案生成方法，总结数据集与评估标准，指出对齐、延迟-精度权衡等挑战，提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.20193v1](http://arxiv.org/pdf/2510.20193v1)**

> **作者:** Rahul Raja; Arpita Vats
>
> **备注:** In Proceedings of the 2nd ACM Workshop in AI-powered Question and Answering Systems (AIQAM '25), October 27-28, 2025, Dublin, Ireland. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3746274.3760393
>
> **摘要:** Question Answering (QA) systems have traditionally relied on structured text data, but the rapid growth of multimedia content (images, audio, video, and structured metadata) has introduced new challenges and opportunities for retrieval-augmented QA. In this survey, we review recent advancements in QA systems that integrate multimedia retrieval pipelines, focusing on architectures that align vision, language, and audio modalities with user queries. We categorize approaches based on retrieval methods, fusion techniques, and answer generation strategies, and analyze benchmark datasets, evaluation protocols, and performance tradeoffs. Furthermore, we highlight key challenges such as cross-modal alignment, latency-accuracy tradeoffs, and semantic grounding, and outline open problems and future research directions for building more robust and context-aware QA systems leveraging multimedia data.
>
---
#### [new 068] LLMs can hide text in other text of the same length.ipynb
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文提出一种利用大语言模型将一段文本隐匿于同长度的另一段看似无关的文本中的方法。属于隐写术任务，旨在解决信息隐蔽传输问题。研究证明80亿参数模型即可高效编码解码，揭示了文本与作者意图的脱钩，对AI安全构成挑战。**

- **链接: [http://arxiv.org/pdf/2510.20075v1](http://arxiv.org/pdf/2510.20075v1)**

> **作者:** Antonio Norelli; Michael Bronstein
>
> **备注:** 21 pages, main paper 9 pages
>
> **摘要:** A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
>
---
#### [new 069] Co-Designing Quantum Codes with Transversal Diagonal Gates via Multi-Agent Systems
- **分类: quant-ph; cs.AI; cs.CL; math-ph; math.MP**

- **简介: 该论文提出一种多智能体协同工作流，用于联合设计具备指定可交换对角门的量子纠错码。基于SSLP框架，结合GPT-5与TeXRA平台，实现问题建模、候选搜索、精确验证与结构抽象，系统性生成新码并证明其正确性，推动了对角门可实现性的规模化分析与分类。**

- **链接: [http://arxiv.org/pdf/2510.20728v1](http://arxiv.org/pdf/2510.20728v1)**

> **作者:** Xi He; Sirui Lu; Bei Zeng
>
> **备注:** 29 pages, 2 figures
>
> **摘要:** We present a multi-agent, human-in-the-loop workflow that co-designs quantum codes with prescribed transversal diagonal gates. It builds on the Subset-Sum Linear Programming (SSLP) framework (arXiv:2504.20847), which partitions basis strings by modular residues and enforces $Z$-marginal Knill-Laflamme (KL) equalities via small LPs. The workflow is powered by GPT-5 and implemented within TeXRA (https://texra.ai)-a multi-agent research assistant platform that supports an iterative tool-use loop agent and a derivation-then-edit workflow reasoning agent. We work in a LaTeX-Python environment where agents reason, edit documents, execute code, and synchronize their work to Git/Overleaf. Within this workspace, three roles collaborate: a Synthesis Agent formulates the problem; a Search Agent sweeps/screens candidates and exactifies numerics into rationals; and an Audit Agent independently checks all KL equalities and the induced logical action. As a first step we focus on distance $d=2$ with nondegenerate residues. For code dimension $K\in\{2,3,4\}$ and $n\le6$ qubits, systematic sweeps yield certificate-backed tables cataloging attainable cyclic logical groups-all realized by new codes-e.g., for $K=3$ we obtain order $16$ at $n=6$. From verified instances, Synthesis Agent abstracts recurring structures into closed-form families and proves they satisfy the KL equalities for all parameters. It further demonstrates that SSLP accommodates residue degeneracy by exhibiting a new $((6,4,2))$ code implementing the transversal controlled-phase $diag(1,1,1,i)$. Overall, the workflow recasts diagonal-transversal feasibility as an analytical pipeline executed at scale, combining systematic enumeration with exact analytical reconstruction. It yields reproducible code constructions, supports targeted extensions to larger $K$ and higher distances, and leads toward data-driven classification.
>
---
#### [new 070] Compress to Impress: Efficient LLM Adaptation Using a Single Gradient Step on 100 Samples
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对大模型高效适配任务，解决传统微调耗时长的问题。提出仅用100样本和单次梯度步，通过分析奇异值梯度筛选关键层，结合多子空间分解，实现无需微调的快速适配，显著提升准确率并大幅降低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.20800v1](http://arxiv.org/pdf/2510.20800v1)**

> **作者:** Shiva Sreeram; Alaa Maalouf; Pratyusha Sharma; Daniela Rus
>
> **摘要:** Recently, Sharma et al. suggested a method called Layer-SElective-Rank reduction (LASER) which demonstrated that pruning high-order components of carefully chosen LLM's weight matrices can boost downstream accuracy -- without any gradient-based fine-tuning. Yet LASER's exhaustive, per-matrix search (each requiring full-dataset forward passes) makes it impractical for rapid deployment. We demonstrate that this overhead can be removed and find that: (i) Only a small, carefully chosen subset of matrices needs to be inspected -- eliminating the layer-by-layer sweep, (ii) The gradient of each matrix's singular values pinpoints which matrices merit reduction, (iii) Increasing the factorization search space by allowing matrices rows to cluster around multiple subspaces and then decomposing each cluster separately further reduces overfitting on the original training data and further lifts accuracy by up to 24.6 percentage points, and finally, (iv) we discover that evaluating on just 100 samples rather than the full training data -- both for computing the indicative gradients and for measuring the final accuracy -- suffices to further reduce the search time; we explain that as adaptation to downstream tasks is dominated by prompting style, not dataset size. As a result, we show that combining these findings yields a fast and robust adaptation algorithm for downstream tasks. Overall, with a single gradient step on 100 examples and a quick scan of the top candidate layers and factorization techniques, we can adapt LLMs to new datasets -- entirely without fine-tuning.
>
---
#### [new 071] ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ImpossibleBench，用于衡量大语言模型在编程任务中利用测试用例“作弊”的倾向。通过构建规范与测试冲突的“不可能”任务，以模型在这些任务上的通过率作为“作弊率”，揭示其绕过规范的行为。工作包括：量化作弊行为、分析影响因素、开发监控工具，助力构建更可靠的LLM编码系统。**

- **链接: [http://arxiv.org/pdf/2510.20270v1](http://arxiv.org/pdf/2510.20270v1)**

> **作者:** Ziqian Zhong; Aditi Raghunathan; Nicholas Carlini
>
> **摘要:** The tendency to find and exploit "shortcuts" to complete tasks poses significant risks for reliable assessment and deployment of large language models (LLMs). For example, an LLM agent with access to unit tests may delete failing tests rather than fix the underlying bug. Such behavior undermines both the validity of benchmark results and the reliability of real-world LLM coding assistant deployments. To quantify, study, and mitigate such behavior, we introduce ImpossibleBench, a benchmark framework that systematically measures LLM agents' propensity to exploit test cases. ImpossibleBench creates "impossible" variants of tasks from existing benchmarks like LiveCodeBench and SWE-bench by introducing direct conflicts between the natural-language specification and the unit tests. We measure an agent's "cheating rate" as its pass rate on these impossible tasks, where any pass necessarily implies a specification-violating shortcut. As a practical framework, ImpossibleBench is not just an evaluation but a versatile tool. We demonstrate its utility for: (1) studying model behaviors, revealing more fine-grained details of cheating behaviors from simple test modification to complex operator overloading; (2) context engineering, showing how prompt, test access and feedback loop affect cheating rates; and (3) developing monitoring tools, providing a testbed with verified deceptive solutions. We hope ImpossibleBench serves as a useful framework for building more robust and reliable LLM systems. Our implementation can be found at https://github.com/safety-research/impossiblebench.
>
---
#### [new 072] Analyticup E-commerce Product Search Competition Technical Report from Team Tredence_AICOE
- **分类: cs.IR; cs.CL**

- **简介: 该论文针对多语言电商搜索中的查询-类别（QC）与查询-商品（QI）相关性任务，通过数据增强与模型微调提升跨语言匹配效果。团队采用Gemma-3 12B与Qwen-2.5 14B模型，结合翻译数据与少数类样本，实现多语言覆盖，最终在竞赛中获第4名，平均F1达0.8857。**

- **链接: [http://arxiv.org/pdf/2510.20674v1](http://arxiv.org/pdf/2510.20674v1)**

> **作者:** Rakshith R; Shubham Sharma; Mohammed Sameer Khan; Ankush Chopra
>
> **摘要:** This study presents the multilingual e-commerce search system developed by the Tredence_AICOE team. The competition features two multilingual relevance tasks: Query-Category (QC) Relevance, which evaluates how well a user's search query aligns with a product category, and Query-Item (QI) Relevance, which measures the match between a multilingual search query and an individual product listing. To ensure full language coverage, we performed data augmentation by translating existing datasets into languages missing from the development set, enabling training across all target languages. We fine-tuned Gemma-3 12B and Qwen-2.5 14B model for both tasks using multiple strategies. The Gemma-3 12B (4-bit) model achieved the best QC performance using original and translated data, and the best QI performance using original, translated, and minority class data creation. These approaches secured 4th place on the final leaderboard, with an average F1-score of 0.8857 on the private test set.
>
---
#### [new 073] What Defines Good Reasoning in LLMs? Dissecting Reasoning Steps with Multi-Aspect Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦于大模型推理质量评估任务，针对现有方法仅关注最终答案正确性而忽视推理过程的问题，提出从相关性和连贯性两方面细粒度评估推理步骤。引入因果逐步评估（CaSE）方法，避免回溯偏差，并通过专家标注数据验证其有效性，证明基于CaSE优化训练数据可提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.20603v1](http://arxiv.org/pdf/2510.20603v1)**

> **作者:** Heejin Do; Jaehui Hwang; Dongyoon Han; Seong Joon Oh; Sangdoo Yun
>
> **摘要:** Evaluating large language models (LLMs) on final-answer correctness is the dominant paradigm. This approach, however, provides a coarse signal for model improvement and overlooks the quality of the underlying reasoning process. We argue that a more granular evaluation of reasoning offers a more effective path to building robust models. We decompose reasoning quality into two dimensions: relevance and coherence. Relevance measures if a step is grounded in the problem; coherence measures if it follows logically from prior steps. To measure these aspects reliably, we introduce causal stepwise evaluation (CaSE). This method assesses each reasoning step using only its preceding context, which avoids hindsight bias. We validate CaSE against human judgments on our new expert-annotated benchmarks, MRa-GSM8K and MRa-MATH. More importantly, we show that curating training data with CaSE-evaluated relevance and coherence directly improves final task performance. Our work provides a scalable framework for analyzing, debugging, and improving LLM reasoning, demonstrating the practical value of moving beyond validity checks.
>
---
#### [new 074] IKnow: Instruction-Knowledge-Aware Continual Pretraining for Effective Domain Adaptation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出IKnow框架，解决指令微调模型在无监督持续预训练中因自监督学习导致指令遵循能力下降的问题。通过构建基于指令-响应对话格式的新型自监督目标，利用文本内嵌领域知识，在不依赖原始模型或外部数据库的情况下，实现有效领域适应。**

- **链接: [http://arxiv.org/pdf/2510.20377v1](http://arxiv.org/pdf/2510.20377v1)**

> **作者:** Tianyi Zhang; Florian Mai; Lucie Flek
>
> **摘要:** Continual pretraining promises to adapt large language models (LLMs) to new domains using only unlabeled test-time data, but naively applying standard self-supervised objectives to instruction-tuned models is known to degrade their instruction-following capability and semantic representations. Existing fixes assume access to the original base model or rely on knowledge from an external domain-specific database - both of which pose a realistic barrier in settings where the base model weights are withheld for safety reasons or reliable external corpora are unavailable. In this work, we propose Instruction-Knowledge-Aware Continual Adaptation (IKnow), a simple and general framework that formulates novel self-supervised objectives in the instruction-response dialogue format. Rather than depend- ing on external resources, IKnow leverages domain knowledge embedded within the text itself and learns to encode it at a deeper semantic level.
>
---
#### [new 075] Calibrating Multimodal Consensus for Emotion Recognition
- **分类: cs.CV; cs.CL; cs.LG; cs.MM**

- **简介: 该论文针对多模态情感识别中模态语义不一致和文本主导问题，提出Calibrated Multimodal Consensus（CMC）模型。通过伪标签生成与无参融合机制，实现自监督预训练与可靠共识融合，提升在存在冲突信息场景下的识别性能。**

- **链接: [http://arxiv.org/pdf/2510.20256v1](http://arxiv.org/pdf/2510.20256v1)**

> **作者:** Guowei Zhong; Junjie Li; Huaiyu Zhu; Ruohong Huan; Yun Pan
>
> **摘要:** In recent years, Multimodal Emotion Recognition (MER) has made substantial progress. Nevertheless, most existing approaches neglect the semantic inconsistencies that may arise across modalities, such as conflicting emotional cues between text and visual inputs. Besides, current methods are often dominated by the text modality due to its strong representational capacity, which can compromise recognition accuracy. To address these challenges, we propose a model termed Calibrated Multimodal Consensus (CMC). CMC introduces a Pseudo Label Generation Module (PLGM) to produce pseudo unimodal labels, enabling unimodal pretraining in a self-supervised fashion. It then employs a Parameter-free Fusion Module (PFM) and a Multimodal Consensus Router (MCR) for multimodal finetuning, thereby mitigating text dominance and guiding the fusion process toward a more reliable consensus. Experimental results demonstrate that CMC achieves performance on par with or superior to state-of-the-art methods across four datasets, CH-SIMS, CH-SIMS v2, CMU-MOSI, and CMU-MOSEI, and exhibits notable advantages in scenarios with semantic inconsistencies on CH-SIMS and CH-SIMS v2. The implementation of this work is publicly accessible at https://github.com/gw-zhong/CMC.
>
---
#### [new 076] Every Question Has Its Own Value: Reinforcement Learning with Explicit Human Values
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出RLEV方法，通过显式人类价值信号优化大语言模型。针对传统强化学习忽略任务重要性的问题，将可量化的人类价值融入奖励函数，提升高价值任务表现，并实现价值感知的终止策略。实验表明该方法在多种场景下优于基线，且对噪声价值信号鲁棒。**

- **链接: [http://arxiv.org/pdf/2510.20187v1](http://arxiv.org/pdf/2510.20187v1)**

> **作者:** Dian Yu; Yulai Zhao; Kishan Panaganti; Linfeng Song; Haitao Mi; Dong Yu
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** We propose Reinforcement Learning with Explicit Human Values (RLEV), a method that aligns Large Language Model (LLM) optimization directly with quantifiable human value signals. While Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains models in objective domains using binary correctness rewards, it overlooks that not all tasks are equally significant. RLEV extends this framework by incorporating human-defined value signals directly into the reward function. Using exam-style data with explicit ground-truth value labels, RLEV consistently outperforms correctness-only baselines across multiple RL algorithms and model scales. Crucially, RLEV policies not only improve value-weighted accuracy but also learn a value-sensitive termination policy: concise for low-value prompts, thorough for high-value ones. We demonstrate this behavior stems from value-weighted gradient amplification on end-of-sequence tokens. Ablation studies confirm the gain is causally linked to value alignment. RLEV remains robust under noisy value signals, such as difficulty-based labels, demonstrating that optimizing for an explicit utility function offers a practical path to aligning LLMs with human priorities.
>
---
#### [new 077] BadGraph: A Backdoor Attack Against Latent Diffusion Model for Text-Guided Graph Generation
- **分类: cs.LG; cs.CL; q-bio.BM**

- **简介: 该论文针对文本引导图生成中的安全问题，提出BadGraph后门攻击方法。通过文本触发器污染训练数据，在不损害正常性能的前提下，使模型在含触发词时生成指定子图。实验验证了攻击的有效性与隐蔽性，揭示了潜在风险，强调了对这类生成模型加强防御的必要性。**

- **链接: [http://arxiv.org/pdf/2510.20792v1](http://arxiv.org/pdf/2510.20792v1)**

> **作者:** Liang Ye; Shengqin Chen; Jiazhu Dai
>
> **摘要:** The rapid progress of graph generation has raised new security concerns, particularly regarding backdoor vulnerabilities. While prior work has explored backdoor attacks in image diffusion and unconditional graph generation, conditional, especially text-guided graph generation remains largely unexamined. This paper proposes BadGraph, a backdoor attack method targeting latent diffusion models for text-guided graph generation. BadGraph leverages textual triggers to poison training data, covertly implanting backdoors that induce attacker-specified subgraphs during inference when triggers appear, while preserving normal performance on clean inputs. Extensive experiments on four benchmark datasets (PubChem, ChEBI-20, PCDes, MoMu) demonstrate the effectiveness and stealth of the attack: less than 10% poisoning rate can achieves 50% attack success rate, while 24% suffices for over 80% success rate, with negligible performance degradation on benign samples. Ablation studies further reveal that the backdoor is implanted during VAE and diffusion training rather than pretraining. These findings reveal the security vulnerabilities in latent diffusion models of text-guided graph generation, highlight the serious risks in models' applications such as drug discovery and underscore the need for robust defenses against the backdoor attack in such diffusion models.
>
---
#### [new 078] Real Deep Research for AI, Robotics and Beyond
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出Real Deep Research（RDR）框架，旨在应对AI与机器人领域研究爆炸式增长带来的信息过载问题。通过系统分析研究趋势、挖掘跨领域机会，为科研人员提供新方向。任务为自动化科研洞察，解决追踪前沿难题。**

- **链接: [http://arxiv.org/pdf/2510.20809v1](http://arxiv.org/pdf/2510.20809v1)**

> **作者:** Xueyan Zou; Jianglong Ye; Hao Zhang; Xiaoyu Xiang; Mingyu Ding; Zhaojing Yang; Yong Jae Lee; Zhuowen Tu; Sifei Liu; Xiaolong Wang
>
> **备注:** website: https://realdeepresearch.github.io
>
> **摘要:** With the rapid growth of research in AI and robotics now producing over 10,000 papers annually it has become increasingly difficult for researchers to stay up to date. Fast evolving trends, the rise of interdisciplinary work, and the need to explore domains beyond one's expertise all contribute to this challenge. To address these issues, we propose a generalizable pipeline capable of systematically analyzing any research area: identifying emerging trends, uncovering cross domain opportunities, and offering concrete starting points for new inquiry. In this work, we present Real Deep Research (RDR) a comprehensive framework applied to the domains of AI and robotics, with a particular focus on foundation models and robotics advancements. We also briefly extend our analysis to other areas of science. The main paper details the construction of the RDR pipeline, while the appendix provides extensive results across each analyzed topic. We hope this work sheds light for researchers working in the field of AI and beyond.
>
---
#### [new 079] Prompt Decorators: A Declarative and Composable Syntax for Reasoning, Formatting, and Control in LLMs
- **分类: cs.PL; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出Prompt Decorators，一种用于控制大模型推理、格式与行为的声明式、可组合语法。针对传统提示工程冗长、不可复现的问题，通过紧凑控制标记（如+++Reasoning、+++Tone）实现对模型行为的精确调控，提升透明度、模块化与一致性，推动可扩展AI系统的构建。**

- **链接: [http://arxiv.org/pdf/2510.19850v1](http://arxiv.org/pdf/2510.19850v1)**

> **作者:** Mostapha Kalami Heris
>
> **摘要:** Large Language Models (LLMs) are central to reasoning, writing, and decision-support workflows, yet users lack consistent control over how they reason and express outputs. Conventional prompt engineering relies on verbose natural-language instructions, limiting reproducibility, modularity, and interpretability. This paper introduces Prompt Decorators, a declarative, composable syntax that governs LLM behavior through compact control tokens such as +++Reasoning, +++Tone(style=formal), and +++Import(topic="Systems Thinking"). Each decorator modifies a behavioral dimension, such as reasoning style, structure, or tone, without changing task content. The framework formalizes twenty core decorators organized into two functional families (Cognitive & Generative and Expressive & Systemic), each further decomposed into subcategories that govern reasoning, interaction, expression, and session-control. It defines a unified syntax, scoping model, and deterministic processing pipeline enabling predictable and auditable behavior composition. By decoupling task intent from execution behavior, Prompt Decorators create a reusable and interpretable interface for prompt design. Illustrative use cases demonstrate improved reasoning transparency, reduced prompt complexity, and standardized model behavior across domains. The paper concludes with implications for interoperability, behavioral consistency, and the development of declarative interfaces for scalable AI systems.
>
---
#### [new 080] Branch-and-Browse: Efficient and Controllable Web Exploration with Tree-Structured Reasoning and Action Memory
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Branch-and-Browse框架，用于提升大语言模型驱动的网页代理在目标导向任务中的效率与可控性。针对现有方法在多步推理深度、回溯能力与计算成本上的不足，该工作通过树状结构推理、状态重放与动作记忆机制，实现细粒度探索与高效执行，在WebArena上显著提升成功率并降低耗时。**

- **链接: [http://arxiv.org/pdf/2510.19838v1](http://arxiv.org/pdf/2510.19838v1)**

> **作者:** Shiqi He; Yue Cui; Xinyu Ma; Yaliang Li; Bolin Ding; Mosharaf Chowdhury
>
> **摘要:** Autonomous web agents powered by large language models (LLMs) show strong potential for performing goal-oriented tasks such as information retrieval, report generation, and online transactions. These agents mark a key step toward practical embodied reasoning in open web environments. However, existing approaches remain limited in reasoning depth and efficiency: vanilla linear methods fail at multi-step reasoning and lack effective backtracking, while other search strategies are coarse-grained and computationally costly. We introduce Branch-and-Browse, a fine-grained web agent framework that unifies structured reasoning-acting, contextual memory, and efficient execution. It (i) employs explicit subtask management with tree-structured exploration for controllable multi-branch reasoning, (ii) bootstraps exploration through efficient web state replay with background reasoning, and (iii) leverages a page action memory to share explored actions within and across sessions. On the WebArena benchmark, Branch-and-Browse achieves a task success rate of 35.8\% and reduces execution time by up to 40.4\% relative to state-of-the-art methods. These results demonstrate that Branch-and-Browse is a reliable and efficient framework for LLM-based web agents.
>
---
#### [new 081] Empathic Prompting: Non-Verbal Context Integration for Multimodal LLM Conversations
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出“共情提示”框架，解决多模态对话中隐性非语言信息缺失问题。通过集成面部表情识别，自动捕捉用户情绪并融入文本提示，增强大模型对话的共情与流畅性，无需用户显式操作，适用于医疗、教育等需情感感知的场景。**

- **链接: [http://arxiv.org/pdf/2510.20743v1](http://arxiv.org/pdf/2510.20743v1)**

> **作者:** Lorenzo Stacchio; Andrea Ubaldi; Alessandro Galdelli; Maurizio Mauri; Emanuele Frontoni; Andrea Gaggioli
>
> **摘要:** We present Empathic Prompting, a novel framework for multimodal human-AI interaction that enriches Large Language Model (LLM) conversations with implicit non-verbal context. The system integrates a commercial facial expression recognition service to capture users' emotional cues and embeds them as contextual signals during prompting. Unlike traditional multimodal interfaces, empathic prompting requires no explicit user control; instead, it unobtrusively augments textual input with affective information for conversational and smoothness alignment. The architecture is modular and scalable, allowing integration of additional non-verbal modules. We describe the system design, implemented through a locally deployed DeepSeek instance, and report a preliminary service and usability evaluation (N=5). Results show consistent integration of non-verbal input into coherent LLM outputs, with participants highlighting conversational fluidity. Beyond this proof of concept, empathic prompting points to applications in chatbot-mediated communication, particularly in domains like healthcare or education, where users' emotional signals are critical yet often opaque in verbal exchanges.
>
---
#### [new 082] Beyond One-Way Influence: Bidirectional Opinion Dynamics in Multi-Turn Human-LLM Interactions
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究多轮人机对话中的双向观点动态，聚焦用户与大模型互动中相互影响机制。针对现有研究仅关注单向影响的不足，通过三类对话实验（静态、标准、个性化）发现：用户立场变化小，但模型输出显著调整，且个性化增强双向影响；包含个人经历的对话最易引发立场改变。研究揭示了过度对齐风险，强调需审慎设计个性化聊天机器人。**

- **链接: [http://arxiv.org/pdf/2510.20039v1](http://arxiv.org/pdf/2510.20039v1)**

> **作者:** Yuyang Jiang; Longjie Guo; Yuchen Wu; Aylin Caliskan; Tanu Mitra; Hua Shen
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Large language model (LLM)-powered chatbots are increasingly used for opinion exploration. Prior research examined how LLMs alter user views, yet little work extended beyond one-way influence to address how user input can affect LLM responses and how such bi-directional influence manifests throughout the multi-turn conversations. This study investigates this dynamic through 50 controversial-topic discussions with participants (N=266) across three conditions: static statements, standard chatbot, and personalized chatbot. Results show that human opinions barely shifted, while LLM outputs changed more substantially, narrowing the gap between human and LLM stance. Personalization amplified these shifts in both directions compared to the standard setting. Analysis of multi-turn conversations further revealed that exchanges involving participants' personal stories were most likely to trigger stance changes for both humans and LLMs. Our work highlights the risk of over-alignment in human-LLM interaction and the need for careful design of personalized chatbots to more thoughtfully and stably align with users.
>
---
#### [new 083] Communication to Completion: Modeling Collaborative Workflows with Intelligent Multi-Agent Communication
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出Communication to Completion（C2C）框架，解决多智能体协作中缺乏任务导向通信机制的问题。通过引入对齐因子（AF）量化任务一致性，并设计分步行动框架实现智能通信决策，提升协作效率。实验表明，C2C可缩短40%任务完成时间，具备良好可扩展性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.19995v1](http://arxiv.org/pdf/2510.19995v1)**

> **作者:** Yiming Lu; Xun Wang; Simin Ma; Shujian Liu; Sathish Reddy Indurthi; Song Wang; Haoyun Deng; Fei Liu; Kaiqiang Song
>
> **备注:** 13 pages
>
> **摘要:** Teamwork in workspace for complex tasks requires diverse communication strategies, but current multi-agent LLM systems lack systematic frameworks for task oriented communication. We introduce Communication to Completion (C2C), a scalable framework that addresses this gap through two key innovations: (1) the Alignment Factor (AF), a novel metric quantifying agent task alignment that directly impacts work efficiency, and (2) a Sequential Action Framework that integrates stepwise execution with intelligent communication decisions. C2C enables agents to make cost aware communication choices, dynamically improving task understanding through targeted interactions. We evaluated C2C on realistic coding workflows across three complexity tiers and team sizes from 5 to 17 agents, comparing against no communication and fixed steps baselines. The results show that C2C reduces the task completion time by about 40% with acceptable communication costs. The framework completes all tasks successfully in standard configurations and maintains effectiveness at scale. C2C establishes both a theoretical foundation for measuring communication effectiveness in multi-agent systems and a practical framework for complex collaborative tasks.
>
---
#### [new 084] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究大视觉语言模型（LVLMs）在生成长文本时易产生幻觉的问题。指出幻觉主因是长响应对上下文依赖增强，而非长度本身。提出“诱导-检测-抑制”框架，通过设计上下文诱导幻觉，实现早期检测与抑制，显著提升准确性，深化了对长响应幻觉机制的理解。**

- **链接: [http://arxiv.org/pdf/2510.20229v1](http://arxiv.org/pdf/2510.20229v1)**

> **作者:** Ge Zheng; Jiaye Qian; Jiajin Tang; Sibei Yang
>
> **摘要:** Large Vision-Language Models (LVLMs) have made significant progress in recent years but are also prone to hallucination issues. They exhibit more hallucinations in longer, free-form responses, often attributed to accumulated uncertainties. In this paper, we ask: Does increased hallucination result solely from length-induced errors, or is there a deeper underlying mechanism? After a series of preliminary experiments and findings, we suggest that the risk of hallucinations is not caused by length itself but by the increased reliance on context for coherence and completeness in longer responses. Building on these insights, we propose a novel "induce-detect-suppress" framework that actively induces hallucinations through deliberately designed contexts, leverages induced instances for early detection of high-risk cases, and ultimately suppresses potential object-level hallucinations during actual decoding. Our approach achieves consistent, significant improvements across all benchmarks, demonstrating its efficacy. The strong detection and improved hallucination mitigation not only validate our framework but, more importantly, re-validate our hypothesis on context. Rather than solely pursuing performance gains, this study aims to provide new insights and serves as a first step toward a deeper exploration of hallucinations in LVLMs' longer responses.
>
---
#### [new 085] BIOCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对生物多模态模型缺乏有效语言监督的问题，提出利用多模态大模型生成领域适配的合成描述性文本作为额外监督信号。通过构建BIOCAP模型，实现图像与文本在物种级语义上的对齐，提升物种分类与图文检索性能，推动生物图像与多模态模型的深度融合。**

- **链接: [http://arxiv.org/pdf/2510.20095v1](http://arxiv.org/pdf/2510.20095v1)**

> **作者:** Ziheng Zhang; Xinyue Ma; Arpita Chowdhury; Elizabeth G. Campolongo; Matthew J. Thompson; Net Zhang; Samuel Stevens; Hilmar Lapp; Tanya Berger-Wolf; Yu Su; Wei-Lun Chao; Jianyang Gu
>
> **备注:** Project page: https://imageomics.github.io/biocap/
>
> **摘要:** This work investigates descriptive captions as an additional source of supervision for biological multimodal foundation models. Images and captions can be viewed as complementary samples from the latent morphospace of a species, each capturing certain biological traits. Incorporating captions during training encourages alignment with this shared latent structure, emphasizing potentially diagnostic characters while suppressing spurious correlations. The main challenge, however, lies in obtaining faithful, instance-specific captions at scale. This requirement has limited the utilization of natural language supervision in organismal biology compared with many other scientific domains. We complement this gap by generating synthetic captions with multimodal large language models (MLLMs), guided by Wikipedia-derived visual information and taxon-tailored format examples. These domain-specific contexts help reduce hallucination and yield accurate, instance-based descriptive captions. Using these captions, we train BIOCAP (i.e., BIOCLIP with Captions), a biological foundation model that captures rich semantics and achieves strong performance in species classification and text-image retrieval. These results demonstrate the value of descriptive captions beyond labels in bridging biological images with multimodal foundation models.
>
---
#### [new 086] Relative-Based Scaling Law for Neural Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对语言模型性能评估中仅依赖交叉熵的局限，提出相对概率（RBP）指标，建立基于相对排序的缩放定律。通过实验证明其在多数据集和模型上的鲁棒性，揭示了模型涌现现象并推动理论发展，为大模型研究提供了新视角。**

- **链接: [http://arxiv.org/pdf/2510.20387v1](http://arxiv.org/pdf/2510.20387v1)**

> **作者:** Baoqing Yue; Jinyuan Zhou; Zixi Wei; Jingtao Zhan; Qingyao Ai; Yiqun Liu
>
> **摘要:** Scaling laws aim to accurately predict model performance across different scales. Existing scaling-law studies almost exclusively rely on cross-entropy as the evaluation metric. However, cross-entropy provides only a partial view of performance: it measures the absolute probability assigned to the correct token, but ignores the relative ordering between correct and incorrect tokens. Yet, relative ordering is crucial for language models, such as in greedy-sampling scenario. To address this limitation, we investigate scaling from the perspective of relative ordering. We first propose the Relative-Based Probability (RBP) metric, which quantifies the probability that the correct token is ranked among the top predictions. Building on this metric, we establish the Relative-Based Scaling Law, which characterizes how RBP improves with increasing model size. Through extensive experiments on four datasets and four model families spanning five orders of magnitude, we demonstrate the robustness and accuracy of this law. Finally, we illustrate the broad application of this law with two examples, namely providing a deeper explanation of emergence phenomena and facilitating finding fundamental theories of scaling laws. In summary, the Relative-Based Scaling Law complements the cross-entropy perspective and contributes to a more complete understanding of scaling large language models. Thus, it offers valuable insights for both practical development and theoretical exploration.
>
---
#### [new 087] AI PB: A Grounded Generative Agent for Personalized Investment Insights
- **分类: cs.AI; cs.CE; cs.CL**

- **简介: 该论文提出AI PB，一个用于零售金融的生成式智能体，解决传统聊天机器人被动响应问题。通过组件化路由、混合检索与多阶段推荐机制，在合规前提下主动生成个性化投资建议，实现在韩国监管要求下的本地化部署与可信生成。**

- **链接: [http://arxiv.org/pdf/2510.20099v1](http://arxiv.org/pdf/2510.20099v1)**

> **作者:** Daewoo Park; Suho Park; Inseok Hong; Hanwool Lee; Junkyu Park; Sangjun Lee; Jeongman An; Hyunbin Loh
>
> **备注:** Under Review
>
> **摘要:** We present AI PB, a production-scale generative agent deployed in real retail finance. Unlike reactive chatbots that answer queries passively, AI PB proactively generates grounded, compliant, and user-specific investment insights. It integrates (i) a component-based orchestration layer that deterministically routes between internal and external LLMs based on data sensitivity, (ii) a hybrid retrieval pipeline using OpenSearch and the finance-domain embedding model, and (iii) a multi-stage recommendation mechanism combining rule heuristics, sequential behavioral modeling, and contextual bandits. Operating fully on-premises under Korean financial regulations, the system employs Docker Swarm and vLLM across 24 X NVIDIA H100 GPUs. Through human QA and system metrics, we demonstrate that grounded generation with explicit routing and layered safety can deliver trustworthy AI insights in high-stakes finance.
>
---
#### [new 088] SODBench: A Large Language Model Approach to Documenting Spreadsheet Operations
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文提出SOD任务，即用自然语言描述电子表格操作。针对现有缺乏系统文档方法的问题，构建了包含111个代码-摘要对的基准数据集，并评估多个大模型在生成文档上的表现，验证了利用LLMs实现准确文档化可行性，有助于提升可复现性与协作效率。**

- **链接: [http://arxiv.org/pdf/2510.19864v1](http://arxiv.org/pdf/2510.19864v1)**

> **作者:** Amila Indika; Igor Molybog
>
> **备注:** 14 pages, 5 figures, 4 tables
>
> **摘要:** Numerous knowledge workers utilize spreadsheets in business, accounting, and finance. However, a lack of systematic documentation methods for spreadsheets hinders automation, collaboration, and knowledge transfer, which risks the loss of crucial institutional knowledge. This paper introduces Spreadsheet Operations Documentation (SOD), an AI task that involves generating human-readable explanations from spreadsheet operations. Many previous studies have utilized Large Language Models (LLMs) for generating spreadsheet manipulation code; however, translating that code into natural language for SOD is a less-explored area. To address this, we present a benchmark of 111 spreadsheet manipulation code snippets, each paired with a corresponding natural language summary. We evaluate five LLMs, GPT-4o, GPT-4o-mini, LLaMA-3.3-70B, Mixtral-8x7B, and Gemma2-9B, using BLEU, GLEU, ROUGE-L, and METEOR metrics. Our findings suggest that LLMs can generate accurate spreadsheet documentation, making SOD a feasible prerequisite step toward enhancing reproducibility, maintainability, and collaborative workflows in spreadsheets, although there are challenges that need to be addressed.
>
---
## 更新

#### [replaced 001] Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19338v2](http://arxiv.org/pdf/2510.19338v2)**

> **作者:** Ling Team; Bin Han; Caizhi Tang; Chen Liang; Donghao Zhang; Fan Yuan; Feng Zhu; Jie Gao; Jingyu Hu; Longfei Li; Meng Li; Mingyang Zhang; Peijie Jiang; Peng Jiao; Qian Zhao; Qingyuan Yang; Wenbo Shen; Xinxing Yang; Yalin Zhang; Yankun Ren; Yao Zhao; Yibo Cao; Yixuan Sun; Yue Zhang; Yuchen Fang; Zibin Lin; Zixuan Cheng; Jun Zhou
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks.
>
---
#### [replaced 002] WolBanking77: Wolof Banking Speech Intent Classification Dataset
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19271v2](http://arxiv.org/pdf/2509.19271v2)**

> **作者:** Abdou Karim Kandji; Frédéric Precioso; Cheikh Ba; Samba Ndiaye; Augustin Ndione
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Intent classification models have made a significant progress in recent years. However, previous studies primarily focus on high-resource language datasets, which results in a gap for low-resource languages and for regions with high rates of illiteracy, where languages are more spoken than read or written. This is the case in Senegal, for example, where Wolof is spoken by around 90\% of the population, while the national illiteracy rate remains at of 42\%. Wolof is actually spoken by more than 10 million people in West African region. To address these limitations, we introduce the Wolof Banking Speech Intent Classification Dataset (WolBanking77), for academic research in intent classification. WolBanking77 currently contains 9,791 text sentences in the banking domain and more than 4 hours of spoken sentences. Experiments on various baselines are conducted in this work, including text and voice state-of-the-art models. The results are very promising on this current dataset. In addition, this paper presents an in-depth examination of the dataset's contents. We report baseline F1-scores and word error rates metrics respectively on NLP and ASR models trained on WolBanking77 dataset and also comparisons between models. Dataset and code available at: \href{https://github.com/abdoukarim/wolbanking77}{wolbanking77}.
>
---
#### [replaced 003] Neural Attention Search
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13251v4](http://arxiv.org/pdf/2502.13251v4)**

> **作者:** Difan Deng; Marius Lindauer
>
> **备注:** 35 pages, 11 figures
>
> **摘要:** We present Neural Attention Search (NAtS), a framework that automatically evaluates the importance of each token within a sequence and determines if the corresponding token can be dropped after several steps. This approach can efficiently reduce the KV cache sizes required by transformer-based models during inference and thus reduce inference costs. In this paper, we design a search space that contains three token types: (i) Global Tokens will be preserved and queried by all the following tokens. (ii) Local Tokens survive until the next global token appears. (iii) Sliding Window Tokens have an impact on the inference of a fixed size of the next following tokens. Similar to the One-Shot Neural Architecture Search approach, this token-type information can be learned jointly with the architecture weights via a learnable attention mask. Experiments on both training a new transformer from scratch and fine-tuning existing large language models show that NAtS can efficiently reduce the KV cache size required for the models while maintaining the models' performance.
>
---
#### [replaced 004] LFD: Layer Fused Decoding to Exploit External Knowledge in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19614v2](http://arxiv.org/pdf/2508.19614v2)**

> **作者:** Yang Sun; Zhiyong Xie; Dan Luo; Long Zhang; Liming Dong; Yunwei Zhao; Xixun Lin; Yanxiong Lu; Chenliang Li; Lixin Zou
>
> **摘要:** Retrieval-augmented generation (RAG) incorporates external knowledge into large language models (LLMs), improving their adaptability to downstream tasks and enabling information updates. Surprisingly, recent empirical evidence demonstrates that injecting noise into retrieved relevant documents paradoxically facilitates exploitation of external knowledge and improves generation quality. Although counterintuitive and challenging to apply in practice, this phenomenon enables granular control and rigorous analysis of how LLMs integrate external knowledge. Therefore, in this paper, we intervene on noise injection and establish a layer-specific functional demarcation within the LLM: shallow layers specialize in local context modeling, intermediate layers focus on integrating long-range external factual knowledge, and deeper layers primarily rely on parametric internal knowledge. Building on this insight, we propose Layer Fused Decoding (LFD), a simple decoding strategy that directly combines representations from an intermediate layer with final-layer decoding outputs to fully exploit the external factual knowledge. To identify the optimal intermediate layer, we introduce an internal knowledge score (IKS) criterion that selects the layer with the lowest IKS value in the latter half of layers. Experimental results across multiple benchmarks demonstrate that LFD helps RAG systems more effectively surface retrieved context knowledge with minimal cost.
>
---
#### [replaced 005] LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.15299v4](http://arxiv.org/pdf/2412.15299v4)**

> **作者:** Sangmin Lee; Woo-Jin Chung; Hong-Goo Kang
>
> **备注:** Accepted to AAAI 2025 (Oral Presentation)
>
> **摘要:** Building a universal multilingual automatic speech recognition (ASR) model that performs equitably across languages has long been a challenge due to its inherent difficulties. To address this task we introduce a Language-Agnostic Multilingual ASR pipeline through orthography Unification and language-specific Transliteration (LAMA-UT). LAMA-UT operates without any language-specific modules while matching the performance of state-of-the-art models trained on a minimal amount of data. Our pipeline consists of two key steps. First, we utilize a universal transcription generator to unify orthographic features into Romanized form and capture common phonetic characteristics across diverse languages. Second, we utilize a universal converter to transform these universal transcriptions into language-specific ones. In experiments, we demonstrate the effectiveness of our proposed method leveraging universal transcriptions for massively multilingual ASR. Our pipeline achieves a relative error reduction rate of 45% when compared to Whisper and performs comparably to MMS, despite being trained on only 0.1% of Whisper's training data. Furthermore, our pipeline does not rely on any language-specific modules. However, it performs on par with zero-shot ASR approaches which utilize additional language-specific lexicons and language models. We expect this framework to serve as a cornerstone for flexible multilingual ASR systems that are generalizable even to unseen languages.
>
---
#### [replaced 006] Does Thinking More always Help? Mirage of Test-Time Scaling in Reasoning Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04210v3](http://arxiv.org/pdf/2506.04210v3)**

> **作者:** Soumya Suvra Ghosal; Souradip Chakraborty; Avinash Reddy; Yifu Lu; Mengdi Wang; Dinesh Manocha; Furong Huang; Mohammad Ghavamzadeh; Amrit Singh Bedi
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Recent trends in test-time scaling for reasoning models (e.g., OpenAI o1, DeepSeek R1) have led to a popular belief that extending thinking traces using prompts like "Wait" or "Let me rethink" can improve performance. This raises a natural question: Does thinking more at test-time truly lead to better reasoning? To answer this question, we perform a detailed empirical study across models and benchmarks, which reveals a consistent pattern of initial performance improvements from additional thinking followed by a decline, due to "overthinking". To understand this non-monotonic trend, we consider a simple probabilistic model, which reveals that additional thinking increases output variance-creating an illusion of improved reasoning while ultimately undermining precision. Thus, observed gains from "more thinking" are not true indicators of improved reasoning, but artifacts stemming from the connection between model uncertainty and evaluation metric. This suggests that test-time scaling through extended thinking is not an effective way to utilize the inference thinking budget. Recognizing these limitations, we introduce an alternative test-time scaling approach, parallel thinking, inspired by Best-of-N sampling. Our method generates multiple independent reasoning paths within the same inference budget and selects the most consistent response via majority vote, achieving up to 20% higher accuracy compared to extended thinking. This provides a simple yet effective mechanism for test-time scaling of reasoning models.
>
---
#### [replaced 007] RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20352v2](http://arxiv.org/pdf/2507.20352v2)**

> **作者:** Hao Xiang; Tianyi Tang; Yang Su; Bowen Yu; An Yang; Fei Huang; Yichang Zhang; Yaojie Lu; Hongyu Lin; Xianpei Han; Jingren Zhou; Junyang Lin; Le Sun
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have shown outstanding potential for role-playing applications. Evaluating these capabilities is becoming crucial yet remains challenging. Existing benchmarks mostly adopt a \textbf{character-centric} approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications. To address this limitation, we introduce RMTBench, a comprehensive \textbf{user-centric} bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds. RMTBench includes custom characters with detailed backgrounds and abstract characters defined by simple traits, enabling evaluation across various user scenarios. Our benchmark constructs dialogues based on explicit user motivations rather than character descriptions, ensuring alignment with practical user applications. Furthermore, we construct an authentic multi-turn dialogue simulation mechanism. With carefully selected evaluation dimensions and LLM-based scoring, this mechanism captures the complex intention of conversations between the user and the character. By shifting focus from character background to user intention fulfillment, RMTBench bridges the gap between academic evaluation and practical deployment requirements, offering a more effective framework for assessing role-playing capabilities in LLMs. All code and datasets will be released soon. We release the datasets at https://huggingface.co/datasets/xiangh/RMTBENCH.
>
---
#### [replaced 008] Constraint Satisfaction Approaches to Wordle: Novel Heuristics and Cross-Lexicon Validation
- **分类: cs.CL; cs.AI; 68T20, 90C27; I.2.8; I.2.3; G.1.6**

- **链接: [http://arxiv.org/pdf/2510.02855v2](http://arxiv.org/pdf/2510.02855v2)**

> **作者:** Jahidul Arafat; Fariha Tasmin; Sanjaya Poudel
>
> **备注:** 35 pages, 14 figures, 10 tables. Open-source implementation with 91% test coverage available at https://github.com/jahidul-arafat/constraint_satisfaction_wordle_arxiv_preprint
>
> **摘要:** Wordle presents an algorithmically rich testbed for constraint satisfaction problem (CSP) solving. While existing solvers rely on information-theoretic entropy maximization or frequency-based heuristics without formal constraint treatment, we present the first comprehensive CSP formulation of Wordle with novel constraint-aware solving strategies. We introduce CSP-Aware Entropy, computing information gain after constraint propagation rather than on raw candidate sets, and a Probabilistic CSP framework integrating Bayesian word-frequency priors with logical constraints. Through evaluation on 2,315 English words, CSP-Aware Entropy achieves 3.54 average guesses with 99.9% success rate, a statistically significant 1.7% improvement over Forward Checking (t=-4.82, p<0.001, Cohen's d=0.07) with 46% faster runtime (12.9ms versus 23.7ms per guess). Under 10% noise, CSP-aware approaches maintain 5.3 percentage point advantages (29.0% versus 23.7%, p=0.041), while Probabilistic CSP achieves 100% success across all noise levels (0-20%) through constraint recovery mechanisms. Cross-lexicon validation on 500 Spanish words demonstrates 88% success with zero language-specific tuning, validating that core CSP principles transfer across languages despite an 11.2 percentage point gap from linguistic differences (p<0.001, Fisher's exact test). Our open-source implementation with 34 unit tests achieving 91% code coverage provides reproducible infrastructure for CSP research. The combination of formal CSP treatment, constraint-aware heuristics, probabilistic-logical integration, robustness analysis, and cross-lexicon validation establishes new performance benchmarks demonstrating that principled constraint satisfaction techniques outperform classical information-theoretic and learning-based approaches for structured puzzle-solving domains.
>
---
#### [replaced 009] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.12977v5](http://arxiv.org/pdf/2411.12977v5)**

> **作者:** Mircea Lică; Ojas Shirekar; Baptiste Colle; Chirag Raman
>
> **备注:** Accepted to NeurIPS 2025 main track as poster
>
> **摘要:** Embodied agents powered by large language models (LLMs), such as Voyager, promise open-ended competence in worlds such as Minecraft. However, when powered by open-weight LLMs they still falter on elementary tasks after domain-specific fine-tuning. We propose MindForge, a generative-agent framework for cultural lifelong learning through explicit perspective taking. We introduce three key innovations: (1) a structured theory of mind representation linking percepts, beliefs, desires, and actions; (2) natural inter-agent communication; and (3) a multi-component memory system. Following the cultural learning framework, we test MindForge in both instructive and collaborative settings within Minecraft. In an instructive setting with GPT-4, MindForge agents powered by open-weight LLMs significantly outperform their Voyager counterparts in basic tasks yielding $3\times$ more tech-tree milestones and collecting $2.3\times$ more unique items than the Voyager baseline. Furthermore, in fully \textit{collaborative} settings, we find that the performance of two underachieving agents improves with more communication rounds, echoing the Condorcet Jury Theorem. MindForge agents demonstrate sophisticated behaviors, including expert-novice knowledge transfer, collaborative problem solving, and adaptation to out-of-distribution tasks through accumulated cultural experiences.
>
---
#### [replaced 010] TianHui: A Domain-Specific Large Language Model for Diverse Traditional Chinese Medicine Scenarios
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.19834v2](http://arxiv.org/pdf/2509.19834v2)**

> **作者:** Ji Yin; Menglan He; Yujie Zhang; Linshuai Zhang; Tingting Ma; Ce Tian; Jie Wu; Lin Xu; Tao Jiang
>
> **备注:** 46 pages, 5 figures,3 tables
>
> **摘要:** Domain-specific LLMs in TCM face limitations in research settings due to constrained adaptability, insufficient evaluation datasets, and limited computational resources. This study presents TianHui, a specialized TCM LLM built through contextual data integration and domain knowledge fusion. We constructed a large-scale TCM corpus (0.97GB unsupervised data + 611,312 QA pairs) and employed a two-stage training strategy with QLoRA, DeepSpeed Stage 2, and Flash Attention 2. Evaluation on 12 benchmarks showed TianHui ranked top-three in all metrics for six datasets (APQ, TCMCD, HFR, HCCA, DHPE, TLAW) and achieved top results in the other six (TCMEE, APR, GCPMI, TCMKQA, TCMRC, ADTG). Optimal configuration was identified as LoRA rank=128, alpha=256, epoch=4, dropout=0.2, max length=2048. TianHui enables systematic preservation and scalable application of TCM knowledge. All resources are open-sourced.
>
---
#### [replaced 011] HauntAttack: When Attack Follows Reasoning as a Shadow
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07031v4](http://arxiv.org/pdf/2506.07031v4)**

> **作者:** Jingyuan Ma; Rui Li; Zheng Li; Junfeng Liu; Heming Xia; Lei Sha; Zhifang Sui
>
> **摘要:** Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.
>
---
#### [replaced 012] Twilight: Adaptive Attention Sparsity with Hierarchical Top-$p$ Pruning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02770v3](http://arxiv.org/pdf/2502.02770v3)**

> **作者:** Chaofan Lin; Jiaming Tang; Shuo Yang; Hanshuo Wang; Tian Tang; Boyu Tian; Ion Stoica; Song Han; Mingyu Gao
>
> **备注:** To appear on NeurIPS 2025 (spotlight)
>
> **摘要:** Leveraging attention sparsity to accelerate long-context large language models (LLMs) has been a hot research topic. However, current algorithms such as sparse attention or key-value (KV) cache compression tend to use a fixed budget, which presents a significant challenge during deployment because it fails to account for the dynamic nature of real-world scenarios, where the optimal balance between accuracy and efficiency can vary greatly. In this paper, we find that borrowing top-$p$ sampling (nucleus sampling) to sparse attention can surprisingly achieve adaptive budgeting. Based on this, we propose Twilight, a framework to bring adaptive sparsity to any existing sparse attention algorithm without sacrificing their accuracy. Empirical results show that Twilight can adaptively prune at most 98% of redundant tokens, leading to $15.4\times$ acceleration in self-attention operations and $3.9\times$ acceleration in end-to-end per token latency in long context LLM decoding.
>
---
#### [replaced 013] Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14536v2](http://arxiv.org/pdf/2505.14536v2)**

> **作者:** Agam Goyal; Vedant Rathi; William Yeh; Yian Wang; Yuen Chen; Hari Sundaram
>
> **备注:** EMNLP 2025
>
> **摘要:** Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.
>
---
#### [replaced 014] Deep Research Brings Deeper Harm
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11851v2](http://arxiv.org/pdf/2510.11851v2)**

> **作者:** Shuo Chen; Zonggen Li; Zhen Han; Bailan He; Tong Liu; Haokun Chen; Georg Groh; Philip Torr; Volker Tresp; Jindong Gu
>
> **备注:** Accepted to Reliable ML from Unreliable Data Workshop @ NeurIPS 2025
>
> **摘要:** Deep Research (DR) agents built on Large Language Models (LLMs) can perform complex, multi-step research by decomposing tasks, retrieving online information, and synthesizing detailed reports. However, the misuse of LLMs with such powerful capabilities can lead to even greater risks. This is especially concerning in high-stakes and knowledge-intensive domains such as biosecurity, where DR can generate a professional report containing detailed forbidden knowledge. Unfortunately, we have found such risks in practice: simply submitting a harmful query, which a standalone LLM directly rejects, can elicit a detailed and dangerous report from DR agents. This highlights the elevated risks and underscores the need for a deeper safety analysis. Yet, jailbreak methods designed for LLMs fall short in exposing such unique risks, as they do not target the research ability of DR agents. To address this gap, we propose two novel jailbreak strategies: Plan Injection, which injects malicious sub-goals into the agent's plan; and Intent Hijack, which reframes harmful queries as academic research questions. We conducted extensive experiments across different LLMs and various safety benchmarks, including general and biosecurity forbidden prompts. These experiments reveal 3 key findings: (1) Alignment of the LLMs often fail in DR agents, where harmful prompts framed in academic terms can hijack agent intent; (2) Multi-step planning and execution weaken the alignment, revealing systemic vulnerabilities that prompt-level safeguards cannot address; (3) DR agents not only bypass refusals but also produce more coherent, professional, and dangerous content, compared with standalone LLMs. These results demonstrate a fundamental misalignment in DR agents and call for better alignment techniques tailored to DR agents. Code and datasets are available at https://chenxshuo.github.io/deeper-harm.
>
---
#### [replaced 015] Rope to Nope and Back Again: A New Hybrid Attention Strategy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18795v2](http://arxiv.org/pdf/2501.18795v2)**

> **作者:** Bowen Yang; Bharat Venkitesh; Dwarak Talupuru; Hangyu Lin; David Cairuz; Phil Blunsom; Acyr Locatelli
>
> **摘要:** Long-context large language models (LLMs) have achieved remarkable advancements, driven by techniques like Rotary Position Embedding (RoPE) (Su et al., 2023) and its extensions (Chen et al., 2023; Liu et al., 2024c; Peng et al., 2023). By adjusting RoPE parameters and incorporating training data with extended contexts, we can train performant models with considerably longer input sequences. However, existing RoPE-based methods exhibit performance limitations when applied to extended context lengths. This paper presents a comprehensive analysis of various attention mechanisms, including RoPE, No Positional Embedding (NoPE), and Query-Key Normalization (QK-Norm), identifying their strengths and shortcomings in long-context modeling. Our investigation identifies distinctive attention patterns in these methods and highlights their impact on long-context performance, providing valuable insights for architectural design. Building on these findings, we propose a novel architecture featuring a hybrid attention mechanism that integrates global and local attention spans. This design not only surpasses conventional RoPE-based transformer models with full attention in both long and short context tasks but also delivers substantial efficiency gains during training and inference.
>
---
#### [replaced 016] A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial Natural Language Inference in Vietnamese
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.17716v3](http://arxiv.org/pdf/2406.17716v3)**

> **作者:** Tin Van Huynh; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **备注:** Accepted by Expert Systems with Applications
>
> **摘要:** Existing Vietnamese Natural Language Inference (NLI) datasets lack adversarial complexity, limiting their ability to evaluate model robustness against challenging linguistic phenomena. In this article, we address the gap in robust Vietnamese NLI resources by introducing ViANLI, the first adversarial NLI dataset for Vietnamese, and propose NLIMoE, a Mixture-of-Experts model to tackle its complexity. We construct ViANLI using an adversarial human-and-machine-in-the-loop approach with rigorous verification. NLIMoE integrates expert subnetworks with a learned dynamic routing mechanism on top of a shared transformer encoder. ViANLI comprises over 10,000 premise-hypothesis pairs and challenges state-of-the-art models, with XLM-R Large achieving only 45.5% accuracy, while NLIMoE reaches 47.3%. Training with ViANLI improves performance on other benchmark Vietnamese NLI datasets including ViNLI, VLSP2021-NLI, and VnNewsNLI. ViANLI is released for enhancing research into model robustness and enriching resources for future Vietnamese and multilingual NLI research.
>
---
#### [replaced 017] DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19201v3](http://arxiv.org/pdf/2505.19201v3)**

> **作者:** Yunhai Hu; Tianhua Xia; Zining Liu; Rahul Raman; Xingyu Liu; Bo Bao; Eric Sather; Vithursan Thangarasa; Sai Qian Zhang
>
> **摘要:** Speculative decoding (SD) has emerged as a powerful method for accelerating autoregressive generation in large language models (LLMs), yet its integration into vision-language models (VLMs) remains underexplored. We introduce DREAM, a novel speculative decoding framework tailored for VLMs that combines three key innovations: (1) a cross-attention-based mechanism to inject intermediate features from the target model into the draft model for improved alignment, (2) adaptive intermediate feature selection based on attention entropy to guide efficient draft model training, and (3) visual token compression to reduce draft model latency. DREAM enables efficient, accurate, and parallel multimodal decoding with significant throughput improvement. Experiments across a diverse set of recent popular VLMs, including LLaVA, Pixtral, SmolVLM and Gemma3, demonstrate up to 3.6x speedup over conventional decoding and significantly outperform prior SD baselines in both inference throughput and speculative draft acceptance length across a broad range of multimodal benchmarks. The code is publicly available at: https://github.com/SAI-Lab-NYU/DREAM.git
>
---
#### [replaced 018] Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.19733v2](http://arxiv.org/pdf/2510.19733v2)**

> **作者:** M. H. I. Abdalla; Zhipin Wang; Christian Frey; Steffen Eger; Josif Grabocka
>
> **摘要:** Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factorized hypernetwork framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to 26x fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values.
>
---
#### [replaced 019] ExpertLens: Activation steering features are highly interpretable
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15090v3](http://arxiv.org/pdf/2502.15090v3)**

> **作者:** Masha Fedzechkina; Eleonora Gualdoni; Sinead Williamson; Katherine Metcalf; Skyler Seto; Barry-John Theobald
>
> **摘要:** Activation steering methods in large language models (LLMs) have emerged as an effective way to perform targeted updates to enhance generated language without requiring large amounts of adaptation data. We ask whether the features discovered by activation steering methods are interpretable. We identify neurons responsible for specific concepts (e.g., ``cat'') using the ``finding experts'' method from research on activation steering and show that the ExpertLens, i.e., inspection of these neurons provides insights about model representation. We find that ExpertLens representations are stable across models and datasets and closely align with human representations inferred from behavioral data, matching inter-human alignment levels. ExpertLens significantly outperforms the alignment captured by word/sentence embeddings. By reconstructing human concept organization through ExpertLens, we show that it enables a granular view of LLM concept representation. Our findings suggest that ExpertLens is a flexible and lightweight approach for capturing and analyzing model representations.
>
---
#### [replaced 020] Quantitative LLM Judges
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02945v2](http://arxiv.org/pdf/2506.02945v2)**

> **作者:** Aishwarya Sahoo; Jeevana Kruthi Karnuthala; Tushar Parmanand Budhwani; Pranchal Agarwal; Sankaran Vaidyanathan; Alexa Siu; Franck Dernoncourt; Jennifer Healey; Nedim Lipka; Ryan Rossi; Uttaran Bhattacharya; Branislav Kveton
>
> **摘要:** LLM-as-a-judge is a framework where a large language model (LLM) evaluates the output of another LLM. While LLMs excel at producing qualitative textual evaluations, they often struggle to predict human preferences and numeric scores. We propose quantitative LLM judges, which align evaluation scores of existing LLM judges to humans in a given domain using regression models. The models are trained to improve the score of the original judge using its rationale and score. We present four quantitative judges for different types of absolute and relative feedback, which showcases the generality and versatility of our framework. Our framework is more computationally efficient than supervised fine-tuning and can be more statistically efficient when human feedback is limited, which is expected in practice. We validate these claims empirically on four datasets using two base judges. Our experiments show that quantitative judges can improve the predictive power of existing judges through post-hoc modeling.
>
---
#### [replaced 021] Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.15312v3](http://arxiv.org/pdf/2510.15312v3)**

> **作者:** Zhiyang Chen; Daliang Xu; Haiyang Shen; Mengwei Xu; Shangguang Wang; Yun Ma
>
> **摘要:** Enhancing on-device large language models (LLMs) with contextual information from local data enables personalized and task-aware generation, powering use cases such as intelligent assistants and UI agents. While recent developments in neural processors have substantially improved the efficiency of prefill on mobile devices, the token-by-token generation process still suffers from high latency and limited hardware utilization due to its inherently memory-bound characteristics. This work presents sd.npu, a mobile inference framework that integrates speculative decoding with dynamic hardware scheduling to accelerate context-aware text generation on mobile devices. The framework introduces three synergistic components: (1) adaptive execution scheduling, which dynamically balances compute graphs between prefill and decoding phases; (2) context-aligned drafting, which improves speculative efficiency through lightweight online calibration to current tasks; and (3) hardware-efficient draft extension, which reuses and expands intermediate sequences to improve processing parallelism and reduce verification cost. Experiments on multiple smartphones and representative workloads show consistent improvements of up to 3.8x in generation speed and 4.7x in energy efficiency compared with existing mobile inference solutions. Component-level analysis further validates the contribution of each optimization.
>
---
#### [replaced 022] LeCoDe: A Benchmark Dataset for Interactive Legal Consultation Dialogue Evaluation
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.19667v2](http://arxiv.org/pdf/2505.19667v2)**

> **作者:** Weikang Yuan; Kaisong Song; Zhuoren Jiang; Junjie Cao; Yujie Zhang; Jun Lin; Kun Kuang; Ji Zhang; Xiaozhong Liu
>
> **摘要:** Legal consultation is essential for safeguarding individual rights and ensuring access to justice, yet remains costly and inaccessible to many individuals due to the shortage of professionals. While recent advances in Large Language Models (LLMs) offer a promising path toward scalable, low-cost legal assistance, current systems fall short in handling the interactive and knowledge-intensive nature of real-world consultations. To address these challenges, we introduce LeCoDe, a real-world multi-turn benchmark dataset comprising 3,696 legal consultation dialogues with 110,008 dialogue turns, designed to evaluate and improve LLMs' legal consultation capability. With LeCoDe, we innovatively collect live-streamed consultations from short-video platforms, providing authentic multi-turn legal consultation dialogues. The rigorous annotation by legal experts further enhances the dataset with professional insights and expertise. Furthermore, we propose a comprehensive evaluation framework that assesses LLMs' consultation capabilities in terms of (1) clarification capability and (2) professional advice quality. This unified framework incorporates 12 metrics across two dimensions. Through extensive experiments on various general and domain-specific LLMs, our results reveal significant challenges in this task, with even state-of-the-art models like GPT-4 achieving only 39.8% recall for clarification and 59% overall score for advice quality, highlighting the complexity of professional consultation scenarios. Based on these findings, we further explore several strategies to enhance LLMs' legal consultation abilities. Our benchmark contributes to advancing research in legal domain dialogue systems, particularly in simulating more real-world user-expert interactions.
>
---
#### [replaced 023] A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09721v3](http://arxiv.org/pdf/2510.09721v3)**

> **作者:** Jiale Guo; Suizhi Huang; Mei Li; Dong Huang; Xingsheng Chen; Regina Zhang; Zhijiang Guo; Han Yu; Siu-Ming Yiu; Pietro Lio; Kwok-Yan Lam
>
> **备注:** 22 pages
>
> **摘要:** The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at https://github.com/lisaGuojl/LLM-Agent-SE-Survey.
>
---
#### [replaced 024] Is Safety Standard Same for Everyone? User-Specific Safety Evaluation of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15086v2](http://arxiv.org/pdf/2502.15086v2)**

> **作者:** Yeonjun In; Wonjoong Kim; Kanghoon Yoon; Sungchul Kim; Mehrab Tanjim; Sangwu Park; Kibum Kim; Chanyoung Park
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** As the use of large language model (LLM) agents continues to grow, their safety vulnerabilities have become increasingly evident. Extensive benchmarks evaluate various aspects of LLM safety by defining the safety relying heavily on general standards, overlooking user-specific standards. However, safety standards for LLM may vary based on a user-specific profiles rather than being universally consistent across all users. This raises a critical research question: Do LLM agents act safely when considering user-specific safety standards? Despite its importance for safe LLM use, no benchmark datasets currently exist to evaluate the user-specific safety of LLMs. To address this gap, we introduce U-SafeBench, a benchmark designed to assess user-specific aspect of LLM safety. Our evaluation of 20 widely used LLMs reveals current LLMs fail to act safely when considering user-specific safety standards, marking a new discovery in this field. To address this vulnerability, we propose a simple remedy based on chain-of-thought, demonstrating its effectiveness in improving user-specific safety. Our benchmark and code are available at https://github.com/yeonjun-in/U-SafeBench.
>
---
#### [replaced 025] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.19634v2](http://arxiv.org/pdf/2507.19634v2)**

> **作者:** Sara Papi; Maike Züfle; Marco Gaido; Beatrice Savoldi; Danni Liu; Ioannis Douros; Luisa Bentivogli; Jan Niehues
>
> **备注:** Data available at https://huggingface.co/datasets/FBK-MT/MCIF | Evaluation and baselines available at https://github.com/hlt-mt/mcif
>
> **摘要:** Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations -- hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities -- speech, vision, and text -- and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development.
>
---
#### [replaced 026] Benchmarking GPT-5 for biomedical natural language processing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04462v2](http://arxiv.org/pdf/2509.04462v2)**

> **作者:** Yu Hou; Zaifu Zhan; Min Zeng; Yifan Wu; Shuang Zhou; Rui Zhang
>
> **摘要:** Biomedical literature and clinical narratives pose multifaceted challenges for natural language understanding, from precise entity extraction and document synthesis to multi-step diagnostic reasoning. This study extends a unified benchmark to evaluate GPT-5 and GPT-4o under zero-, one-, and five-shot prompting across five core biomedical NLP tasks: named entity recognition, relation extraction, multi-label document classification, summarization, and simplification, and nine expanded biomedical QA datasets covering factual knowledge, clinical reasoning, and multimodal visual understanding. Using standardized prompts, fixed decoding parameters, and consistent inference pipelines, we assessed model performance, latency, and token-normalized cost under official pricing. GPT-5 consistently outperformed GPT-4o, with the largest gains on reasoning-intensive datasets such as MedXpertQA and DiagnosisArena and stable improvements in multimodal QA. In core tasks, GPT-5 achieved better chemical NER and ChemProt scores but remained below domain-tuned baselines for disease NER and summarization. Despite producing longer outputs, GPT-5 showed comparable latency and 30 to 50 percent lower effective cost per correct prediction. Fine-grained analyses revealed improvements in diagnosis, treatment, and reasoning subtypes, whereas boundary-sensitive extraction and evidence-dense summarization remain challenging. Overall, GPT-5 approaches deployment-ready performance for biomedical QA while offering a favorable balance of accuracy, interpretability, and economic efficiency. The results support a tiered prompting strategy: direct prompting for large-scale or cost-sensitive applications, and chain-of-thought scaffolds for analytically complex or high-stakes scenarios, highlighting the continued need for hybrid solutions where precision and factual fidelity are critical.
>
---
#### [replaced 027] Annotation Guidelines-Based Knowledge Augmentation: Towards Enhancing Large Language Models for Educational Text Classification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.00954v2](http://arxiv.org/pdf/2406.00954v2)**

> **作者:** Shiqi Liu; Sannyuya Liu; Lele Sha; Zijie Zeng; Dragan Gasevic; Zhi Liu
>
> **备注:** The manuscript has been accepted for publication in IEEE Transactions on Learning Technologies. https://doi.org/10.1109/TLT.2025.3570775
>
> **摘要:** Various machine learning approaches have gained significant popularity for the automated classification of educational text to identify indicators of learning engagement -- i.e. learning engagement classification (LEC). LEC can offer comprehensive insights into human learning processes, attracting significant interest from diverse research communities, including Natural Language Processing (NLP), Learning Analytics, and Educational Data Mining. Recently, Large Language Models (LLMs), such as ChatGPT, have demonstrated remarkable performance in various NLP tasks. However, their comprehensive evaluation and improvement approaches in LEC tasks have not been thoroughly investigated. In this study, we propose the Annotation Guidelines-based Knowledge Augmentation (AGKA) approach to improve LLMs. AGKA employs GPT 4.0 to retrieve label definition knowledge from annotation guidelines, and then applies the random under-sampler to select a few typical examples. Subsequently, we conduct a systematic evaluation benchmark of LEC, which includes six LEC datasets covering behavior classification (question and urgency level), emotion classification (binary and epistemic emotion), and cognition classification (opinion and cognitive presence). The study results demonstrate that AGKA can enhance non-fine-tuned LLMs, particularly GPT 4.0 and Llama 3 70B. GPT 4.0 with AGKA few-shot outperforms full-shot fine-tuned models such as BERT and RoBERTa on simple binary classification datasets. However, GPT 4.0 lags in multi-class tasks that require a deep understanding of complex semantic information. Notably, Llama 3 70B with AGKA is a promising combination based on open-source LLM, because its performance is on par with closed-source GPT 4.0 with AGKA. In addition, LLMs struggle to distinguish between labels with similar names in multi-class classification.
>
---
#### [replaced 028] Text2Mem: A Unified Memory Operation Language for Memory Operating System
- **分类: cs.CL; cs.PL**

- **链接: [http://arxiv.org/pdf/2509.11145v2](http://arxiv.org/pdf/2509.11145v2)**

> **作者:** Yi Wang; Lihai Yang; Boyu Chen; Gongyi Zou; Kerun Xu; Bo Tang; Feiyu Xiong; Siheng Chen; Zhiyu Li
>
> **备注:** 12 pages, 3 figures, 2 tables
>
> **摘要:** Large language model agents increasingly depend on memory to sustain long horizon interaction, but existing frameworks remain limited. Most expose only a few basic primitives such as encode, retrieve, and delete, while higher order operations like merge, promote, demote, split, lock, and expire are missing or inconsistently supported. Moreover, there is no formal and executable specification for memory commands, leaving scope and lifecycle rules implicit and causing unpredictable behavior across systems. We introduce Text2Mem, a unified memory operation language that provides a standardized pathway from natural language to reliable execution. Text2Mem defines a compact yet expressive operation set aligned with encoding, storage, and retrieval. Each instruction is represented as a JSON based schema instance with required fields and semantic invariants, which a parser transforms into typed operation objects with normalized parameters. A validator ensures correctness before execution, while adapters map typed objects either to a SQL prototype backend or to real memory frameworks. Model based services such as embeddings or summarization are integrated when required. All results are returned through a unified execution contract. This design ensures safety, determinism, and portability across heterogeneous backends. We also outline Text2Mem Bench, a planned benchmark that separates schema generation from backend execution to enable systematic evaluation. Together, these components establish the first standardized foundation for memory control in agents.
>
---
#### [replaced 029] FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08396v2](http://arxiv.org/pdf/2510.08396v2)**

> **作者:** Heming Zou; Yunliang Zang; Wutong Xu; Yao Zhu; Xiangyang Ji
>
> **备注:** NeurIPS 2025 accepted paper
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely used parameter-efficient fine-tuning method for foundation models, but it suffers from parameter interference, resulting in suboptimal performance. Although Mixture-of-Experts (MoE)-based LoRA variants show promise in mitigating intra-task correlations in single-task instruction tuning, they introduce additional router parameters and remain ineffective in multi-task model merging where inter-task interference arises. Inspired by the fly olfactory circuit, we propose FlyLoRA, an implicit MoE-based LoRA variant that introduces: (1) rank-wise expert activation in the up-projection matrix, and (2) an implicit router that unifies expert routing and down-projection, where a frozen sparse random projection matrix replaces the traditional dense trainable version. This design resolves the trade-off between intra-task decorrelation and computational efficiency by eliminating the need for an explicit router, while inherently mitigating inter-task interference due to the orthogonality property of random matrices. Extensive experiments across four domains -- general knowledge understanding, scientific question answering, mathematical reasoning, and code generation -- demonstrate consistent performance improvements over existing methods. Beyond empirical gains, FlyLoRA highlights how biological structures can inspire innovations in AI technologies. Code is available at https://github.com/gfyddha/FlyLoRA.
>
---
#### [replaced 030] Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09874v2](http://arxiv.org/pdf/2508.09874v2)**

> **作者:** Jiaqi Cao; Jiarui Wang; Rubin Wei; Qipeng Guo; Kai Chen; Bowen Zhou; Zhouhan Lin
>
> **摘要:** Large Language Models (LLMs) have shown strong abilities in general language tasks, yet adapting them to specific domains remains a challenge. Current method like Domain Adaptive Pretraining (DAPT) requires costly full-parameter training and suffers from catastrophic forgetting. Meanwhile, Retrieval-Augmented Generation (RAG) introduces substantial inference latency due to expensive nearest-neighbor searches and longer context. This paper introduces Memory Decoder, a plug-and-play pretrained memory that enables efficient domain adaptation without changing the original model's parameters. Memory Decoder employs a small transformer decoder that learns to imitate the behavior of an external non-parametric retriever. Once trained, Memory Decoder can be seamlessly integrated with any pretrained language model that shares the same tokenizer, requiring no model-specific modifications. Experimental results demonstrate that Memory Decoder enables effective adaptation of various Qwen and Llama models to three distinct specialized domains: biomedicine, finance, and law, reducing perplexity by an average of 6.17 points. Overall, Memory Decoder introduces a novel paradigm centered on a specially pretrained memory component designed for domain-specific adaptation. This memory architecture can be integrated in a plug-and-play manner, consistently enhancing performance across multiple models within the target domain.
>
---
#### [replaced 031] Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.19258v4](http://arxiv.org/pdf/2410.19258v4)**

> **作者:** Yu Fu; Zefan Cai; Abedelkadir Asi; Wayne Xiong; Yue Dong; Wen Xiao
>
> **备注:** Accepted to ICLR2025
>
> **摘要:** Key-Value (KV) caching is a common technique to enhance the computational efficiency of Large Language Models (LLMs), but its memory overhead grows rapidly with input length. Prior work has shown that not all tokens are equally important for text generation, proposing layer-level KV cache compression to selectively retain key information. Recognizing the distinct roles of attention heads in generation, we propose HeadKV, a head-level KV cache compression method, and HeadKV-R2, which leverages a novel contextual reasoning ability estimation for compression. Our approach operates at the level of individual heads, estimating their importance for contextual QA tasks that require both retrieval and reasoning capabilities. Extensive experiments across diverse benchmarks (LongBench, LooGLE), model architectures (e.g., Llama-3-8B-Instruct, Mistral-7B-Instruct), and long-context abilities tests demonstrate that our head-level KV cache compression significantly outperforms strong baselines, particularly in low-resource settings (KV size = 64 & 128). Notably, our method retains just 1.5% of the KV cache while achieving 97% of the performance of the full KV cache on the contextual question answering benchmark. Codes are available at https://github.com/FYYFU/HeadKV
>
---
#### [replaced 032] Fast-Slow Thinking GRPO for Large Vision-Language Model Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18458v2](http://arxiv.org/pdf/2504.18458v2)**

> **作者:** Wenyi Xiao; Leilei Gan
>
> **摘要:** When applying reinforcement learning--typically through GRPO--to large vision-language model reasoning struggles to effectively scale reasoning length or generates verbose outputs across all tasks with only marginal gains in accuracy. To address this issue, we present FAST-GRPO, a variant of GRPO that dynamically adapts reasoning depth based on question characteristics. Through empirical analysis, we establish the feasibility of fast-slow thinking in LVLMs by investigating how response length and data distribution affect performance. Inspired by these observations, we introduce two complementary metrics to estimate the difficulty of the questions, guiding the model to determine when fast or slow thinking is more appropriate. Next, we incorporate adaptive length-based rewards and difficulty-aware KL divergence into the GRPO algorithm. Experiments across seven reasoning benchmarks demonstrate that FAST achieves state-of-the-art accuracy with over 10\% relative improvement compared to the base model, while reducing token usage by 32.7-67.3\% compared to previous slow-thinking approaches, effectively balancing reasoning length and accuracy.
>
---
#### [replaced 033] Hybrid Latent Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18454v2](http://arxiv.org/pdf/2505.18454v2)**

> **作者:** Zhenrui Yue; Bowen Jin; Huimin Zeng; Honglei Zhuang; Zhen Qin; Jinsung Yoon; Lanyu Shang; Jiawei Han; Dong Wang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent advances in large language models (LLMs) have introduced latent reasoning as a promising alternative to autoregressive reasoning. By performing internal computation with hidden states from previous steps, latent reasoning benefit from more informative features rather than sampling a discrete chain-of-thought (CoT) path. Yet latent reasoning approaches are often incompatible with LLMs, as their continuous paradigm conflicts with the discrete nature of autoregressive generation. Moreover, these methods rely on CoT traces for training and thus fail to exploit the inherent reasoning patterns of LLMs. In this work, we explore latent reasoning by leveraging the intrinsic capabilities of LLMs via reinforcement learning (RL). To this end, we introduce hybrid reasoning policy optimization (HRPO), an RL-based hybrid latent reasoning approach that (1) integrates prior hidden states into sampled tokens with a learnable gating mechanism, and (2) initializes training with predominantly token embeddings while progressively incorporating more hidden features. This design maintains LLMs' generative capabilities and incentivizes hybrid reasoning using both discrete and continuous representations. In addition, the hybrid HRPO introduces stochasticity into latent reasoning via token sampling, thereby enabling RL-based optimization without requiring CoT trajectories. Extensive evaluations across diverse benchmarks show that HRPO outperforms prior methods in both knowledge- and reasoning-intensive tasks. Furthermore, HRPO-trained LLMs remain interpretable and exhibit intriguing behaviors like cross-lingual patterns and shorter completion lengths, highlighting the potential of our RL-based approach and offer insights for future work in latent reasoning.
>
---
#### [replaced 034] Multilingual LLM Prompting Strategies for Medical English-Vietnamese Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15640v2](http://arxiv.org/pdf/2509.15640v2)**

> **作者:** Nhu Vo; Nu-Uyen-Phuong Le; Dung D. Le; Massimo Piccardi; Wray Buntine
>
> **备注:** This version has been withdrawn after receiving the conference review results. We are currently extending and reorganizing the work into a new study
>
> **摘要:** Medical English-Vietnamese machine translation (En-Vi MT) is essential for healthcare access and communication in Vietnam, yet Vietnamese remains a low-resource and under-studied language. We systematically evaluate prompting strategies for six multilingual LLMs (0.5B-9B parameters) on the MedEV dataset, comparing zero-shot, few-shot, and dictionary-augmented prompting with Meddict, an English-Vietnamese medical lexicon. Results show that model scale is the primary driver of performance: larger LLMs achieve strong zero-shot results, while few-shot prompting yields only marginal improvements. In contrast, terminology-aware cues and embedding-based example retrieval consistently improve domain-specific translation. These findings underscore both the promise and the current limitations of multilingual LLMs for medical En-Vi MT.
>
---
#### [replaced 035] Permutative Preference Alignment from Listwise Ranking of Human Judgments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.04346v2](http://arxiv.org/pdf/2410.04346v2)**

> **作者:** Yang Zhao; Yixin Wang; Mingzhang Yin
>
> **备注:** Published at EMNLP 2025 Main Conference
>
> **摘要:** Aligning Large Language Models (LLMs) with human preferences is crucial in ensuring desirable and controllable model behaviors. Current methods, such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on the Bradley-Terry (B-T) model to maximize the likelihood of pairwise choices. However, when multiple responses are available, the B-T model fails to guarantee an accurate list ranking of the responses. To address this issue, we propose Permutative Preference Alignment (PPA), a novel offline listwise approach that incorporates the Normalized Discounted Cumulative Gain (NDCG), a widely-used ranking metric, as an alternative training objective for LLM alignment. We develop an end-to-end alignment algorithm by approximating NDCG with a differentiable surrogate loss. Experiments demonstrate that PPA outperforms existing pairwise and listwise methods on evaluation sets and general benchmarks such as AlpacaEval. Furthermore, we show that NDCG-based approaches improve ranking accuracy more effectively than B-T-based methods and provide a theoretical explanation for this improvement.
>
---
#### [replaced 036] Unlocking Multi-View Insights in Knowledge-Dense Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.12879v2](http://arxiv.org/pdf/2404.12879v2)**

> **作者:** Guanhua Chen; Wenhan Yu; Xiao Lu; Xiao Zhang; Erli Meng; Lei Sha
>
> **摘要:** While Retrieval-Augmented Generation (RAG) plays a crucial role in the application of Large Language Models (LLMs), existing retrieval methods in knowledge-dense domains like law and medicine still suffer from a lack of multi-perspective views, which are essential for improving interpretability and reliability. Previous research on multi-view retrieval often focused solely on different semantic forms of queries, neglecting the expression of specific domain knowledge perspectives. This paper introduces a novel multi-view RAG framework, MVRAG, tailored for knowledge-dense domains that utilizes intention-aware query rewriting from multiple domain viewpoints to enhance retrieval precision, thereby improving the effectiveness of the final inference. Experiments conducted on legal and medical case retrieval demonstrate significant improvements in recall and precision rates with our framework. Our multi-perspective retrieval approach unleashes the potential of multi-view information enhancing RAG tasks, accelerating the further application of LLMs in knowledge-intensive fields.
>
---
#### [replaced 037] "You Are Rejected!": An Empirical Study of Large Language Models Taking Hiring Evaluations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19167v2](http://arxiv.org/pdf/2510.19167v2)**

> **作者:** Dingjie Fu; Dianxing Shi
>
> **备注:** Technical Report, 14 pages, 8 figures
>
> **摘要:** With the proliferation of the internet and the rapid advancement of Artificial Intelligence, leading technology companies face an urgent annual demand for a considerable number of software and algorithm engineers. To efficiently and effectively identify high-potential candidates from thousands of applicants, these firms have established a multi-stage selection process, which crucially includes a standardized hiring evaluation designed to assess job-specific competencies. Motivated by the demonstrated prowess of Large Language Models (LLMs) in coding and reasoning tasks, this paper investigates a critical question: Can LLMs successfully pass these hiring evaluations? To this end, we conduct a comprehensive examination of a widely used professional assessment questionnaire. We employ state-of-the-art LLMs to generate responses and subsequently evaluate their performance. Contrary to any prior expectation of LLMs being ideal engineers, our analysis reveals a significant inconsistency between the model-generated answers and the company-referenced solutions. Our empirical findings lead to a striking conclusion: All evaluated LLMs fails to pass the hiring evaluation.
>
---
#### [replaced 038] XtraGPT: Context-Aware and Controllable Academic Paper Revision
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11336v3](http://arxiv.org/pdf/2505.11336v3)**

> **作者:** Nuo Chen; Andre Lin HuiKai; Jiaying Wu; Junyi Hou; Zining Zhang; Qian Wang; Xidong Wang; Bingsheng He
>
> **备注:** Preprint. The model report is available at https://arxiv.org/abs/2505.11336v1
>
> **摘要:** Despite the growing adoption of large language models (LLMs) in academic workflows, their capabilities remain limited to support high-quality scientific writing. Most existing systems are designed for general-purpose scientific text generation and fail to meet the sophisticated demands of research communication beyond surface-level polishing, such as conceptual coherence across sections. Furthermore, academic writing is inherently iterative and revision-driven, a process not well supported by direct prompting-based paradigms. To address these scenarios, we propose a human-AI collaboration framework for academic paper revision centered on criteria-guided intent alignment and context-aware modeling. To validate the framework, we curate a dataset of 7,000 research papers from top-tier venues annotated with 140,000 instruction-response pairs that reflect realistic, section-level scientific revisions. We instantiate the framework in XtraGPT, the first suite of open-source LLMs (1.5B to 14B parameters) for context-aware, instruction-guided writing assistance. Extensive experiments validate that XtraGPT significantly outperforms same-scale baselines and approaches the quality of proprietary systems. Both automated preference assessments and human evaluations confirm the effectiveness of XtraGPT in improving scientific drafts.
>
---
#### [replaced 039] AssistedDS: Benchmarking How External Domain Knowledge Assists LLMs in Automated Data Science
- **分类: cs.LG; cs.AI; cs.CL; stat.ME; 62-07, 62-08, 68T05, 68T07, 68T01, 68T50; I.2.0; I.2.6; I.2.7; I.5.1; I.5.4; H.2.8; G.3**

- **链接: [http://arxiv.org/pdf/2506.13992v2](http://arxiv.org/pdf/2506.13992v2)**

> **作者:** An Luo; Xun Xian; Jin Du; Fangqiao Tian; Ganghua Wang; Ming Zhong; Shengchun Zhao; Xuan Bi; Zirui Liu; Jiawei Zhou; Jayanth Srinivasa; Ashish Kundu; Charles Fleming; Mingyi Hong; Jie Ding
>
> **摘要:** Large language models (LLMs) have advanced the automation of data science workflows. Yet it remains unclear whether they can critically leverage external domain knowledge as human data scientists do in practice. To answer this question, we introduce AssistedDS (Assisted Data Science), a benchmark designed to systematically evaluate how LLMs handle domain knowledge in tabular prediction tasks. AssistedDS features both synthetic datasets with explicitly known generative mechanisms and real-world Kaggle competitions, each accompanied by curated bundles of helpful and adversarial documents. These documents provide domain-specific insights into data cleaning, feature engineering, and model selection. We assess state-of-the-art LLMs on their ability to discern and apply beneficial versus harmful domain knowledge, evaluating submission validity, information recall, and predictive performance. Our results demonstrate three key findings: (1) LLMs frequently exhibit an uncritical adoption of provided information, significantly impairing their predictive performance when adversarial content is introduced, (2) helpful guidance is often insufficient to counteract the negative influence of adversarial information, and (3) in Kaggle datasets, LLMs often make errors in handling time-series data, applying consistent feature engineering across different folds, and interpreting categorical variables correctly. These findings highlight a substantial gap in current models' ability to critically evaluate and leverage expert knowledge, underscoring an essential research direction for developing more robust, knowledge-aware automated data science systems. Our data and code are publicly available here: https://github.com/jeremyxianx/Assisted-DS
>
---
#### [replaced 040] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15235v5](http://arxiv.org/pdf/2509.15235v5)**

> **作者:** Jialiang Kang; Han Shu; Wenshuo Li; Yingjie Zhai; Xinghao Chen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups (<1.5x). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce Vision-Aware Speculative Decoding (ViSpec), a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding. Code is available at https://github.com/KangJialiang/ViSpec.
>
---
#### [replaced 041] Born a Transformer -- Always a Transformer? On the Effect of Pretraining on Architectural Abilities
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21785v3](http://arxiv.org/pdf/2505.21785v3)**

> **作者:** Mayank Jobanputra; Yana Veitsman; Yash Sarrof; Aleksandra Bakalova; Vera Demberg; Ellie Pavlick; Michael Hahn
>
> **备注:** NeurIPS 2025
>
> **摘要:** Transformers have theoretical limitations in modeling certain sequence-to-sequence tasks, yet it remains largely unclear if these limitations play a role in large-scale pretrained LLMs, or whether LLMs might effectively overcome these constraints in practice due to the scale of both the models themselves and their pretraining data. We explore how these architectural constraints manifest after pretraining, by studying a family of $\textit{retrieval}$ and $\textit{copying}$ tasks inspired by Liu et al. [2024a]. We use a recently proposed framework for studying length generalization [Huang et al., 2025] to provide guarantees for each of our settings. Empirically, we observe an $\textit{induction-versus-anti-induction}$ asymmetry, where pretrained models are better at retrieving tokens to the right (induction) rather than the left (anti-induction) of a query token. This asymmetry disappears upon targeted fine-tuning if length-generalization is guaranteed by theory. Mechanistic analysis reveals that this asymmetry is connected to the differences in the strength of induction versus anti-induction circuits within pretrained transformers. We validate our findings through practical experiments on real-world tasks demonstrating reliability risks. Our results highlight that pretraining selectively enhances certain transformer capabilities, but does not overcome fundamental length-generalization limits.
>
---
#### [replaced 042] X-Reflect: Cross-Reflection Prompting for Multimodal Recommendation
- **分类: cs.IR; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.15172v2](http://arxiv.org/pdf/2408.15172v2)**

> **作者:** Hanjia Lyu; Ryan Rossi; Xiang Chen; Md Mehrab Tanjim; Stefano Petrangeli; Somdeb Sarkhel; Jiebo Luo
>
> **摘要:** Large Language Models (LLMs) have been shown to enhance the effectiveness of enriching item descriptions, thereby improving the accuracy of recommendation systems. However, most existing approaches either rely on text-only prompting or employ basic multimodal strategies that do not fully exploit the complementary information available from both textual and visual modalities. This paper introduces a novel framework, Cross-Reflection Prompting, termed X-Reflect, designed to address these limitations by prompting Multimodal Large Language Models (MLLMs) to explicitly identify and reconcile supportive and conflicting information between text and images. By capturing nuanced insights from both modalities, this approach generates more comprehensive and contextually rich item representations. Extensive experiments conducted on two widely used benchmarks demonstrate that our method outperforms existing prompting baselines in downstream recommendation accuracy. Furthermore, we identify a U-shaped relationship between text-image dissimilarity and recommendation performance, suggesting the benefit of applying multimodal prompting selectively. To support efficient real-time inference, we also introduce X-Reflect-keyword, a lightweight variant that summarizes image content using keywords and replaces the base model with a smaller backbone, achieving nearly 50% reduction in input length while maintaining competitive performance. This work underscores the importance of integrating multimodal information and presents an effective solution for improving item understanding in multimodal recommendation systems.
>
---
#### [replaced 043] Diagnosing Representation Dynamics in NER Model Extension
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17930v2](http://arxiv.org/pdf/2510.17930v2)**

> **作者:** Xirui Zhang; Philippe de La Chevasnerie; Benoit Fabre
>
> **摘要:** Extending Named Entity Recognition (NER) models to new PII entities in noisy spoken-language data is a common need. We find that jointly fine-tuning a BERT model on standard semantic entities (PER, LOC, ORG) and new pattern-based PII (EMAIL, PHONE) results in minimal degradation for original classes. We investigate this "peaceful coexistence," hypothesizing that the model uses independent semantic vs. morphological feature mechanisms. Using an incremental learning setup as a diagnostic tool, we measure semantic drift and find two key insights. First, the LOC (location) entity is uniquely vulnerable due to a representation overlap with new PII, as it shares pattern-like features (e.g., postal codes). Second, we identify a "reverse O-tag representation drift." The model, initially trained to map PII patterns to 'O', blocks new learning. This is resolved only by unfreezing the 'O' tag's classifier, allowing the background class to adapt and "release" these patterns. This work provides a mechanistic diagnosis of NER model adaptation, highlighting feature independence, representation overlap, and 'O' tag plasticity. Work done based on data gathered by https://www.papernest.com
>
---
#### [replaced 044] Blockwise SFT for Diffusion Language Models: Reconciling Bidirectional Attention and Autoregressive Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19529v2](http://arxiv.org/pdf/2508.19529v2)**

> **作者:** Bowen Sun; Yujun Cai; Ming-Hsuan Yang; Yiwei Wang
>
> **摘要:** Discrete diffusion language models have shown strong potential for text generation, yet standard supervised fine-tuning (SFT) misaligns with their semi-autoregressive inference: training randomly masks tokens across the entire response, while inference generates fixed-size blocks sequentially. This mismatch introduces noisy prefixes and leaky suffixes, biasing gradients away from the desired blockwise likelihood. We propose Blockwise SFT, which partitions responses into fixed-size blocks, selects one active block per step for stochastic masking, freezes all preceding tokens, and fully hides future ones. Loss is computed only over the active block, directly mirroring the blockwise decoding process. Experiments on GSM8K, MATH, and MetaMathQA show consistent gains over classical SFT under equal compute or token budgets. Block size consistency studies and ablations confirm that improvements stem from faithful training-inference alignment rather than incidental masking effects. Our results highlight the importance of matching supervision granularity to the decoding procedure in diffusion-based language models.
>
---
#### [replaced 045] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13837v4](http://arxiv.org/pdf/2504.13837v4)**

> **作者:** Yang Yue; Zhiqi Chen; Rui Lu; Andrew Zhao; Zhaokai Wang; Yang Yue; Shiji Song; Gao Huang
>
> **备注:** 30 pages, 27 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
>
---
#### [replaced 046] Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20612v4](http://arxiv.org/pdf/2505.20612v4)**

> **作者:** Peter Robicheaux; Matvei Popov; Anish Madan; Isaac Robinson; Joseph Nelson; Deva Ramanan; Neehar Peri
>
> **备注:** The first two authors contributed equally. This work has been accepted to the Neural Information Processing Systems (NeurIPS) 2025 Datasets & Benchmark Track. Project Page: https://rf100-vl.org/
>
> **摘要:** Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Lastly, we discuss our recent CVPR 2025 Foundational FSOD competition and share insights from the community. Notably, the winning team significantly outperforms our baseline by 17 mAP! Our code and dataset are available at https://github.com/roboflow/rf100-vl and https://universe.roboflow.com/rf100-vl/.
>
---
#### [replaced 047] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23883v2](http://arxiv.org/pdf/2505.23883v2)**

> **作者:** Jianyang Gu; Samuel Stevens; Elizabeth G Campolongo; Matthew J Thompson; Net Zhang; Jiaman Wu; Andrei Kopanev; Zheda Mai; Alexander E. White; James Balhoff; Wasila Dahdul; Daniel Rubenstein; Hilmar Lapp; Tanya Berger-Wolf; Wei-Lun Chao; Yu Su
>
> **备注:** NeurIPS 2025 Spotlight; Project page: https://imageomics.github.io/bioclip-2/
>
> **摘要:** Foundation models trained at scale exhibit remarkable emergent behaviors, learning new capabilities beyond their initial training objectives. We find such emergent behaviors in biological vision models via large-scale contrastive vision-language training. To achieve this, we first curate TreeOfLife-200M, comprising 214 million images of living organisms, the largest and most diverse biological organism image dataset to date. We then train BioCLIP 2 on TreeOfLife-200M to distinguish different species. Despite the narrow training objective, BioCLIP 2 yields extraordinary accuracy when applied to various biological visual tasks such as habitat classification and trait prediction. We identify emergent properties in the learned embedding space of BioCLIP 2. At the inter-species level, the embedding distribution of different species aligns closely with functional and ecological meanings (e.g., beak sizes and habitats). At the intra-species level, instead of being diminished, the intra-species variations (e.g., life stages and sexes) are preserved and better separated in subspaces orthogonal to inter-species distinctions. We provide formal proof and analyses to explain why hierarchical supervision and contrastive objectives encourage these emergent properties. Crucially, our results reveal that these properties become increasingly significant with larger-scale training data, leading to a biologically meaningful embedding space.
>
---
#### [replaced 048] KAT-Coder Technical Report
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18779v2](http://arxiv.org/pdf/2510.18779v2)**

> **作者:** Zizheng Zhan; Ken Deng; Xiaojiang Zhang; Jinghui Wang; Huaixi Tang; Zhiyi Lai; Haoyang Huang; Wen Xiang; Kun Wu; Wenhao Zhuang; Minglei Zhang; Shaojie Wang; Shangpeng Yan; Kepeng Lei; Zongxian Feng; Huiming Wang; Zheng Lin; Mengtong Li; Mengfei Xie; Yinghan Cui; Xuxing Chen; Chao Wang; Weihao Li; Wenqiang Zhu; Jiarong Zhang; Jingxuan Xu; Songwei Yu; Yifan Yao; Xinping Lei; C. Zhang; Han Li; Junqi Xiong; Zuchen Gao; Dailin Li; Haimo Li; Jiaheng Liu; Yuqun Zhang; Junyi Peng; Haotian Zhang; Bin Chen
>
> **摘要:** Recent advances in large language models (LLMs) have enabled progress in agentic coding, where models autonomously reason, plan, and act within interactive software development workflows. However, bridging the gap between static text-based training and dynamic real-world agentic execution remains a core challenge. In this technical report, we present KAT-Coder, a large-scale agentic code model trained through a multi-stage curriculum encompassing Mid-Term Training, Supervised Fine-Tuning (SFT), Reinforcement Fine-Tuning (RFT), and Reinforcement-to-Deployment Adaptation. The Mid-Term stage enhances reasoning, planning, and reflection capabilities through a corpus of real software engineering data and synthetic agentic interactions. The SFT stage constructs a million-sample dataset balancing twenty programming languages, ten development contexts, and ten task archetypes. The RFT stage introduces a novel multi-ground-truth reward formulation for stable and sample-efficient policy optimization. Finally, the Reinforcement-to-Deployment phase adapts the model to production-grade IDE environments using Error-Masked SFT and Tree-Structured Trajectory Training. In summary, these stages enable KAT-Coder to achieve robust tool-use reliability, instruction alignment, and long-context reasoning, forming a deployable foundation for real-world intelligent coding agents. Our KAT series 32B model, KAT-Dev, has been open-sourced on https://huggingface.co/Kwaipilot/KAT-Dev.
>
---
#### [replaced 049] Language Models use Lookbacks to Track Beliefs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14685v2](http://arxiv.org/pdf/2505.14685v2)**

> **作者:** Nikhil Prakash; Natalie Shapira; Arnab Sen Sharma; Christoph Riedl; Yonatan Belinkov; Tamar Rott Shaham; David Bau; Atticus Geiger
>
> **备注:** 31 pages, 33 figures. Code and data at https://belief.baulab.info/
>
> **摘要:** How do language models (LMs) represent characters' beliefs, especially when those beliefs may differ from reality? This question lies at the heart of understanding the Theory of Mind (ToM) capabilities of LMs. We analyze LMs' ability to reason about characters' beliefs using causal mediation and abstraction. We construct a dataset, CausalToM, consisting of simple stories where two characters independently change the state of two objects, potentially unaware of each other's actions. Our investigation uncovers a pervasive algorithmic pattern that we call a lookback mechanism, which enables the LM to recall important information when it becomes necessary. The LM binds each character-object-state triple together by co-locating their reference information, represented as Ordering IDs (OIs), in low-rank subspaces of the state token's residual stream. When asked about a character's beliefs regarding the state of an object, the binding lookback retrieves the correct state OI and then the answer lookback retrieves the corresponding state token. When we introduce text specifying that one character is (not) visible to the other, we find that the LM first generates a visibility ID encoding the relation between the observing and the observed character OIs. In a visibility lookback, this ID is used to retrieve information about the observed character and update the observing character's beliefs. Our work provides insights into belief tracking mechanisms, taking a step toward reverse-engineering ToM reasoning in LMs.
>
---
#### [replaced 050] Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16348v2](http://arxiv.org/pdf/2505.16348v2)**

> **作者:** Taeyoon Kwon; Dongwook Choi; Hyojun Kim; Sunghwan Kim; Seungjun Moon; Beong-woo Kwak; Kuan-Hao Huang; Jinyoung Yeo
>
> **备注:** Work in progress
>
> **摘要:** LLM-powered embodied agents have shown success on conventional object-rearrangement tasks, but providing personalized assistance that leverages user-specific knowledge from past interactions presents new challenges. We investigate these challenges through the lens of agents' memory utilization along two critical dimensions: object semantics (identifying objects based on personal meaning) and user patterns (recalling sequences from behavioral routines). To assess these capabilities, we construct MEMENTO, an end-to-end two-stage evaluation framework comprising single-memory and joint-memory tasks. Our experiments reveal that current agents can recall simple object semantics but struggle to apply sequential user patterns to planning. Through in-depth analysis, we identify two critical bottlenecks: information overload and coordination failures when handling multiple memories. Based on these findings, we explore memory architectural approaches to address these challenges. Given our observation that episodic memory provides both personalized knowledge and in-context learning benefits, we design a hierarchical knowledge graph-based user-profile memory module that separately manages personalized knowledge, achieving substantial improvements on both single and joint-memory tasks. Project website: https://connoriginal.github.io/MEMENTO
>
---
#### [replaced 051] Language Models (Mostly) Know When to Stop Reading
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01025v2](http://arxiv.org/pdf/2502.01025v2)**

> **作者:** Roy Xie; Junlin Wang; Paul Rosu; Chunyuan Deng; Bolun Sun; Zihao Lin; Bhuwan Dhingra
>
> **备注:** Accepted to NeurIPS 2025. Project website: https://royxie.com/when-to-stop-project
>
> **摘要:** Large language models (LLMs) process entire input contexts indiscriminately, which is inefficient when the information required to answer a query is localized within the context. We present dynamic context cutoff, a novel method enabling LLMs to self-terminate processing upon acquiring sufficient task-relevant information. Through analysis of model internals, we discover that specific attention heads inherently encode "sufficiency signals" -- detectable through lightweight classifiers -- that predict when critical information has been processed. This reveals a new efficiency paradigm: models' internal understanding naturally dictates processing needs rather than external compression heuristics. Comprehensive experiments across six QA datasets (up to 40K tokens) with three model families (LLaMA/Qwen/Mistral, 1B-70B) demonstrate 3.4% accuracy improvement while achieving 1.33x token reduction on average. Furthermore, our method demonstrates superior performance compared to other context efficiency methods at equivalent token reduction rates. Additionally, we observe an emergent scaling phenomenon: while smaller models require probing for sufficiency detection, larger models exhibit intrinsic self-assessment capabilities through prompting.
>
---
#### [replaced 052] Token embeddings violate the manifold hypothesis
- **分类: cs.CL; cs.AI; 53Z50, 62H15**

- **链接: [http://arxiv.org/pdf/2504.01002v3](http://arxiv.org/pdf/2504.01002v3)**

> **作者:** Michael Robinson; Sourya Dey; Tony Chiang
>
> **备注:** 30 pages, 9 figures, 10 tables
>
> **摘要:** A full understanding of the behavior of a large language model (LLM) requires our grasp of its input token space. If this space differs from our assumptions, our comprehension of and conclusions about the LLM will likely be flawed. We elucidate the structure of the token embeddings both empirically and theoretically. We present a novel statistical test assuming that the neighborhood around each token has a relatively flat and smooth structure as the null hypothesis. Failing to reject the null is uninformative, but rejecting it at a specific token $\psi$ implies an irregularity in the token subspace in a $\psi$-neighborhood, $B(\psi)$. The structure assumed in the null is a generalization of a manifold with boundary called a \emph{smooth fiber bundle} (which can be split into two spatial regimes -- small and large radius), so we denote our new hypothesis test as the ``fiber bundle hypothesis.'' By running our test over several open-source LLMs, each with unique token embeddings, we find that the null is frequently rejected, and so the evidence suggests that the token subspace is not a fiber bundle and hence also not a manifold. As a consequence of our findings, when an LLM is presented with two semantically equivalent prompts, if one prompt contains a token implicated by our test, the response to that prompt will likely exhibit less stability than the other.
>
---
#### [replaced 053] Position: The Current AI Conference Model is Unsustainable! Diagnosing the Crisis of Centralized AI Conference
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04586v4](http://arxiv.org/pdf/2508.04586v4)**

> **作者:** Nuo Chen; Moming Duan; Andre Huikai Lin; Qian Wang; Jiaying Wu; Bingsheng He
>
> **备注:** Preprint
>
> **摘要:** Artificial Intelligence (AI) conferences are essential for advancing research, sharing knowledge, and fostering academic community. However, their rapid expansion has rendered the centralized conference model increasingly unsustainable. This paper offers a data-driven diagnosis of a structural crisis that threatens the foundational goals of scientific dissemination, equity, and community well-being. We identify four key areas of strain: (1) scientifically, with per-author publication rates more than doubling over the past decade to over 4.5 papers annually; (2) environmentally, with the carbon footprint of a single conference exceeding the daily emissions of its host city; (3) psychologically, with 71% of online community discourse reflecting negative sentiment and 35% referencing mental health concerns; and (4) logistically, with attendance at top conferences such as NeurIPS 2024 beginning to outpace venue capacity. These pressures point to a system that is misaligned with its core mission. In response, we propose the Community-Federated Conference (CFC) model, which separates peer review, presentation, and networking into globally coordinated but locally organized components, offering a more sustainable, inclusive, and resilient path forward for AI research.
>
---
#### [replaced 054] Text Generation Beyond Discrete Token Sampling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14827v3](http://arxiv.org/pdf/2505.14827v3)**

> **作者:** Yufan Zhuang; Liyuan Liu; Chandan Singh; Jingbo Shang; Jianfeng Gao
>
> **摘要:** In standard autoregressive generation, an LLM predicts the next-token distribution, samples a discrete token, and then discards the distribution, passing only the sampled token as new input. To preserve this distribution's rich information, we propose Mixture of Inputs (MoI), a training-free method for autoregressive generation. After generating a token following the standard paradigm, we construct a new input that blends the generated discrete token with the previously discarded token distribution. Specifically, we employ a Bayesian estimation method that treats the token distribution as the prior, the sampled token as the observation, and replaces the conventional one-hot vector with the continuous posterior expectation as the new model input. MoI allows the model to maintain a richer internal representation throughout the generation process, resulting in improved text quality and reasoning capabilities. On mathematical reasoning, code generation, and PhD-level QA tasks, MoI consistently improves performance across multiple models including QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, and DAPO-Qwen-32B, with no additional training and negligible computational overhead.
>
---
#### [replaced 055] Toward Metaphor-Fluid Conversation Design for Voice User Interfaces
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.ET**

- **链接: [http://arxiv.org/pdf/2502.11554v2](http://arxiv.org/pdf/2502.11554v2)**

> **作者:** Smit Desai; Jessie Chin; Dakuo Wang; Benjamin Cowan; Michael Twidale
>
> **摘要:** Metaphors play a critical role in shaping user experiences with Voice User Interfaces (VUIs), yet existing designs often rely on static, human-centric metaphors that fail to adapt to diverse contexts and user needs. This paper introduces Metaphor-Fluid Design, a novel approach that dynamically adjusts metaphorical representations based on conversational use-contexts. We compare this approach to a Default VUI, which characterizes the present implementation of commercial VUIs commonly designed around the persona of an assistant, offering a uniform interaction style across contexts. In Study 1 (N=130), metaphors were mapped to four key use-contexts-commands, information seeking, sociality, and error recovery-along the dimensions of formality and hierarchy, revealing distinct preferences for task-specific metaphorical designs. Study 2 (N=91) evaluates a Metaphor-Fluid VUI against a Default VUI, showing that the Metaphor-Fluid VUI enhances perceived intention to adopt, enjoyment, and likability by aligning better with user expectations for different contexts. However, individual differences in metaphor preferences highlight the need for personalization. These findings challenge the one-size-fits-all paradigm of VUI design and demonstrate the potential of Metaphor-Fluid Design to create more adaptive and engaging human-AI interactions.
>
---
#### [replaced 056] MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14483v2](http://arxiv.org/pdf/2505.14483v2)**

> **作者:** Agam Goyal; Xianyang Zhan; Yilun Chen; Koustuv Saha; Eshwar Chandrasekharan
>
> **备注:** EMNLP 2025 (Oral)
>
> **摘要:** Large language models (LLMs) have shown great potential in flagging harmful content in online communities. Yet, existing approaches for moderation require a separate model for every community and are opaque in their decision-making, limiting real-world adoption. We introduce Mixture of Moderation Experts (MoMoE), a modular, cross-community framework that adds post-hoc explanations to scalable content moderation. MoMoE orchestrates four operators -- Allocate, Predict, Aggregate, Explain -- and is instantiated as seven community-specialized experts (MoMoE-Community) and five norm-violation experts (MoMoE-NormVio). On 30 unseen subreddits, the best variants obtain Micro-F1 scores of 0.72 and 0.67, respectively, matching or surpassing strong fine-tuned baselines while consistently producing concise and reliable explanations. Although community-specialized experts deliver the highest peak accuracy, norm-violation experts provide steadier performance across domains. These findings show that MoMoE yields scalable, transparent moderation without needing per-community fine-tuning. More broadly, they suggest that lightweight, explainable expert ensembles can guide future NLP and HCI research on trustworthy human-AI governance of online communities.
>
---
#### [replaced 057] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.14144v2](http://arxiv.org/pdf/2406.14144v2)**

> **作者:** Jianhui Chen; Xiaozhi Wang; Zijun Yao; Yushi Bai; Lei Hou; Juanzi Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models (LLMs) excel in various capabilities but pose safety risks such as generating harmful content and misinformation, even after safety alignment. In this paper, we explore the inner mechanisms of safety alignment through the lens of mechanistic interpretability, focusing on identifying and analyzing safety neurons within LLMs that are responsible for safety behaviors. We propose inference-time activation contrasting to locate these neurons and dynamic activation patching to evaluate their causal effects on model safety. Experiments on multiple prevalent LLMs demonstrate that we can consistently identify about $5\%$ safety neurons, and by only patching their activations we can restore over $90\%$ of the safety performance across various red-teaming benchmarks without influencing general ability. The finding of safety neurons also helps explain the ''alignment tax'' phenomenon by revealing that the key neurons for model safety and helpfulness significantly overlap, yet they require different activation patterns for the same neurons. Furthermore, we demonstrate an application of our findings in safeguarding LLMs by detecting unsafe outputs before generation. The source code is available at https://github.com/THU-KEG/SafetyNeuron.
>
---
#### [replaced 058] ReDit: Reward Dithering for Improved LLM Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18631v3](http://arxiv.org/pdf/2506.18631v3)**

> **作者:** Chenxing Wei; Jiarui Yu; Ying Tiffany He; Hande Dong; Yao Shu; Fei Yu
>
> **备注:** 34 pages, 19 figures
>
> **摘要:** DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
>
---
#### [replaced 059] SpecEval: Evaluating Model Adherence to Behavior Specifications
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02464v2](http://arxiv.org/pdf/2509.02464v2)**

> **作者:** Ahmed Ahmed; Kevin Klyman; Yi Zeng; Sanmi Koyejo; Percy Liang
>
> **摘要:** Companies that develop foundation models publish behavioral guidelines they pledge their models will follow, but it remains unclear if models actually do so. While providers such as OpenAI, Anthropic, and Google have published detailed specifications describing both desired safety constraints and qualitative traits for their models, there has been no systematic audit of adherence to these guidelines. We introduce an automated framework that audits models against their providers specifications by parsing behavioral statements, generating targeted prompts, and using models to judge adherence. Our central focus is on three way consistency between a provider specification, its model outputs, and its own models as judges; an extension of prior two way generator validator consistency. This establishes a necessary baseline: at minimum, a foundation model should consistently satisfy the developer behavioral specifications when judged by the developer evaluator models. We apply our framework to 16 models from six developers across more than 100 behavioral statements, finding systematic inconsistencies including compliance gaps of up to 20 percent across providers.
>
---
#### [replaced 060] MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09933v5](http://arxiv.org/pdf/2502.09933v5)**

> **作者:** Kai Yan; Zhan Ling; Kang Liu; Yifan Yang; Ting-Han Fan; Lingfeng Shen; Zhengyin Du; Jiecao Chen
>
> **备注:** 39 pages, 11 figures. The paper is accepted at NeurIPS 2025 Datasets & Benchmarks Track, and the latest version adds modifications in camera-ready
>
> **摘要:** The ability to recognize patterns from examples and apply them to new ones is a primal ability for general intelligence, and is widely studied by psychology and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually <10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations often focus on classification, and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context reasoning benchmark for pattern recognition that asks LLM to predict output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for many-shot in-context reasoning, and acquired many insightful findings including scaling effect, robustness, inductive vs. transductive reasoning, retrieval Augmented Generation (RAG), coding for inductive reasoning, cross-domain generalizability, etc.
>
---
#### [replaced 061] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.17536v2](http://arxiv.org/pdf/2508.17536v2)**

> **作者:** Hyeong Kyu Choi; Xiaojin Zhu; Sharon Li
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Multi-Agent Debate~(MAD) has emerged as a promising paradigm for improving the performance of large language models through collaborative reasoning. Despite recent advances, the key factors driving MAD's effectiveness remain unclear. In this work, we disentangle MAD into two key components--Majority Voting and inter-agent Debate--and assess their respective contributions. Through extensive experiments across seven NLP benchmarks, we find that Majority Voting alone accounts for most of the performance gains typically attributed to MAD. To explain this, we propose a theoretical framework that models debate as a stochastic process. We prove that it induces a martingale over agents' belief trajectories, implying that debate alone does not improve expected correctness. Guided by these insights, we demonstrate that targeted interventions, by biasing the belief update toward correction, can meaningfully enhance debate effectiveness. Overall, our findings suggest that while MAD has potential, simple ensembling methods remain strong and more reliable alternatives in many practical settings. Code is released in https://github.com/deeplearning-wisc/debate-or-vote.
>
---
#### [replaced 062] Stress-Testing Model Specs Reveals Character Differences among Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.07686v2](http://arxiv.org/pdf/2510.07686v2)**

> **作者:** Jifan Zhang; Henry Sleight; Andi Peng; John Schulman; Esin Durmus
>
> **摘要:** Large language models (LLMs) are increasingly trained from AI constitutions and model specifications that establish behavioral guidelines and ethical principles. However, these specifications face critical challenges, including internal conflicts between principles and insufficient coverage of nuanced scenarios. We present a systematic methodology for stress-testing model character specifications, automatically identifying numerous cases of principle contradictions and interpretive ambiguities in current model specs. We stress test current model specs by generating scenarios that force explicit tradeoffs between competing value-based principles. Using a comprehensive taxonomy we generate diverse value tradeoff scenarios where models must choose between pairs of legitimate principles that cannot be simultaneously satisfied. We evaluate responses from twelve frontier LLMs across major providers (Anthropic, OpenAI, Google, xAI) and measure behavioral disagreement through value classification scores. Among these scenarios, we identify over 70,000 cases exhibiting significant behavioral divergence. Empirically, we show this high divergence in model behavior strongly predicts underlying problems in model specifications. Through qualitative analysis, we provide numerous example issues in current model specs such as direct contradiction and interpretive ambiguities of several principles. Additionally, our generated dataset also reveals both clear misalignment cases and false-positive refusals across all of the frontier models we study. Lastly, we also provide value prioritization patterns and differences of these models.
>
---
#### [replaced 063] PersonaMatrix: A Recipe for Persona-Aware Evaluation of Legal Summarization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.16449v2](http://arxiv.org/pdf/2509.16449v2)**

> **作者:** Tsz Fung Pang; Maryam Berijanian; Thomas Orth; Breanna Shi; Charlotte S. Alexander
>
> **备注:** Accepted for publication in JURIX 2025 (Legal Knowledge and Information Systems, FAIA series, IOS Press). Long Paper
>
> **摘要:** Legal documents are often long, dense, and difficult to comprehend, not only for laypeople but also for legal experts. While automated document summarization has great potential to improve access to legal knowledge, prevailing task-based evaluators overlook divergent user and stakeholder needs. Tool development is needed to encompass the technicality of a case summary for a litigator yet be accessible for a self-help public researching for their lawsuit. We introduce PersonaMatrix, a persona-by-criterion evaluation framework that scores summaries through the lens of six personas, including legal and non-legal users. We also introduce a controlled dimension-shifted pilot dataset of U.S. civil rights case summaries that varies along depth, accessibility, and procedural detail as well as Diversity-Coverage Index (DCI) to expose divergent optima of legal summary between persona-aware and persona-agnostic judges. This work enables refinement of legal AI summarization systems for both expert and non-expert users, with the potential to increase access to legal knowledge. The code base and data are publicly available in GitHub.
>
---
#### [replaced 064] Sherlock: Self-Correcting Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22651v2](http://arxiv.org/pdf/2505.22651v2)**

> **作者:** Yi Ding; Ruqi Zhang
>
> **备注:** Published at NeurIPS 2025, 27 pages
>
> **摘要:** Reasoning Vision-Language Models (VLMs) have shown promising performance on complex multimodal tasks. However, they still face significant challenges: they are highly sensitive to reasoning errors, require large volumes of annotated data or accurate verifiers, and struggle to generalize beyond specific domains. To address these limitations, we explore self-correction as a strategy to enhance reasoning VLMs. We first conduct an in-depth analysis of reasoning VLMs' self-correction abilities and identify key gaps. Based on our findings, we introduce Sherlock, a self-correction and self-improvement training framework. Sherlock introduces a trajectory-level self-correction objective, a preference data construction method based on visual perturbation, and a dynamic $\beta$ for preference tuning. Once the model acquires self-correction capabilities using only 20k randomly sampled annotated data, it continues to self-improve without external supervision. Built on the Llama3.2-Vision-11B model, Sherlock achieves remarkable results across eight benchmarks, reaching an average accuracy of 64.1 with direct generation and 65.4 after self-correction. It outperforms LLaVA-CoT (63.2), Mulberry (63.9), and LlamaV-o1 (63.4) while using less than 20% of the annotated data.
>
---
#### [replaced 065] Integrating Structural and Semantic Signals in Text-Attributed Graphs with BiGTex
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12474v3](http://arxiv.org/pdf/2504.12474v3)**

> **作者:** Azadeh Beiranvand; Seyed Mehdi Vahidipour
>
> **备注:** 26 pages, 4 figures
>
> **摘要:** Text-attributed graphs (TAGs) present unique challenges in representation learning by requiring models to capture both the semantic richness of node-associated texts and the structural dependencies of the graph. While graph neural networks (GNNs) excel at modeling topological information, they lack the capacity to process unstructured text. Conversely, large language models (LLMs) are proficient in text understanding but are typically unaware of graph structure. In this work, we propose BiGTex (Bidirectional Graph Text), a novel architecture that tightly integrates GNNs and LLMs through stacked Graph-Text Fusion Units. Each unit allows for mutual attention between textual and structural representations, enabling information to flow in both directions, text influencing structure and structure guiding textual interpretation. The proposed architecture is trained using parameter-efficient fine-tuning (LoRA), keeping the LLM frozen while adapting to task-specific signals. Extensive experiments on five benchmark datasets demonstrate that BiGTex achieves state-of-the-art performance in node classification and generalizes effectively to link prediction. An ablation study further highlights the importance of soft prompting and bi-directional attention in the model's success.
>
---
#### [replaced 066] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04510v2](http://arxiv.org/pdf/2502.04510v2)**

> **作者:** Shangbin Feng; Zifeng Wang; Palash Goyal; Yike Wang; Weijia Shi; Huang Xia; Hamid Palangi; Luke Zettlemoyer; Yulia Tsvetkov; Chen-Yu Lee; Tomas Pfister
>
> **备注:** NeurIPS 2025
>
> **摘要:** We propose Heterogeneous Swarms, an algorithm to design multi-LLM systems by jointly optimizing model roles and weights. We represent multi-LLM systems as directed acyclic graphs (DAGs) of LLMs with topological message passing for collaborative generation. Given a pool of LLM experts and a utility function, Heterogeneous Swarms employs two iterative steps: role-step and weight-step. For role-step, we interpret model roles as learning a DAG that specifies the flow of inputs and outputs between LLMs. Starting from a swarm of random continuous adjacency matrices, we decode them into discrete DAGs, call the LLMs in topological order, evaluate on the utility function (e.g. accuracy on a task), and optimize the adjacency matrices with particle swarm optimization based on the utility score. For weight-step, we assess the contribution of individual LLMs in the multi-LLM systems and optimize model weights with swarm intelligence. We propose JFK-score to quantify the individual contribution of each LLM in the best-found DAG of the role-step, then optimize model weights with particle swarm optimization based on the JFK-score. Experiments demonstrate that Heterogeneous Swarms outperforms 15 role- and/or weight-based baselines by 18.5% on average across 12 tasks. Further analysis reveals that Heterogeneous Swarms discovers multi-LLM systems with heterogeneous model roles and substantial collaborative gains, and benefits from the diversity of language models.
>
---
#### [replaced 067] Breaking mBad! Supervised Fine-tuning for Cross-Lingual Detoxification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16722v3](http://arxiv.org/pdf/2505.16722v3)**

> **作者:** Himanshu Beniwal; Youngwoo Kim; Maarten Sap; Soham Dan; Thomas Hartvigsen
>
> **备注:** Accepted at MELT Workshop @ COLM 2025
>
> **摘要:** As large language models (LLMs) become increasingly prevalent in global applications, ensuring that they are toxicity-free across diverse linguistic contexts remains a critical challenge. We explore "Cross-lingual Detoxification", a cross-lingual paradigm that mitigates toxicity, enabling detoxification capabilities to transfer between high and low-resource languages across different script families. We analyze cross-lingual detoxification's effectiveness through 392 extensive settings to evaluate toxicity reduction in cross-distribution settings with limited data and investigate how mitigation impacts model performance on non-toxic tasks, revealing trade-offs between safety and knowledge preservation. Our code and dataset are publicly available at https://github.com/himanshubeniwal/Breaking-mBad.
>
---
#### [replaced 068] Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.18469v5](http://arxiv.org/pdf/2410.18469v5)**

> **作者:** Chung-En Sun; Xiaodong Liu; Weiwei Yang; Tsui-Wei Weng; Hao Cheng; Aidan San; Michel Galley; Jianfeng Gao
>
> **备注:** Accepted to NAACL 2025 Main (Oral)
>
> **摘要:** Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM
>
---
#### [replaced 069] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15034v2](http://arxiv.org/pdf/2505.15034v2)**

> **作者:** Kaiwen Zha; Zhengqi Gao; Maohao Shen; Zhang-Wei Hong; Duane S. Boning; Dina Katabi
>
> **备注:** NeurIPS 2025. The first two authors contributed equally
>
> **摘要:** Reinforcement learning (RL) has recently emerged as a compelling approach for enhancing the reasoning capabilities of large language models (LLMs), where an LLM generator serves as a policy guided by a verifier (reward model). However, current RL post-training methods for LLMs typically use verifiers that are fixed (rule-based or frozen pretrained) or trained discriminatively via supervised fine-tuning (SFT). Such designs are susceptible to reward hacking and generalize poorly beyond their training distributions. To overcome these limitations, we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level LLM verifier, which is trained via RL and co-evolves with the generator. Importantly, the verifier is trained solely based on outcome-level verification correctness rewards without requiring explicit process-level annotations. This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator. Extensive experiments demonstrate that both components of Tango achieve state-of-the-art results among 7B/8B-scale models: the generator attains best-in-class performance across five competition-level math benchmarks and four challenging out-of-domain reasoning tasks, while the verifier leads on the ProcessBench dataset. Remarkably, both components exhibit particularly substantial improvements on the most difficult mathematical reasoning problems. Code is at: https://github.com/kaiwenzha/rl-tango.
>
---
#### [replaced 070] ixi-GEN: Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06795v4](http://arxiv.org/pdf/2507.06795v4)**

> **作者:** Seonwu Kim; Yohan Na; Kihun Kim; Hanhee Cho; Geun Lim; Mintae Kim; Seongik Park; Ki Hyun Kim; Youngsub Han; Byoung-Ki Jeon
>
> **备注:** Accepted at EMNLP 2025 Industry Track
>
> **摘要:** The emergence of open-source large language models (LLMs) has expanded opportunities for enterprise applications; however, many organizations still lack the infrastructure to deploy and maintain large-scale models. As a result, small LLMs (sLLMs) have become a practical alternative despite inherent performance limitations. While Domain Adaptive Continual Pretraining (DACP) has been explored for domain adaptation, its utility in commercial settings remains under-examined. In this study, we validate the effectiveness of a DACP-based recipe across diverse foundation models and service domains, producing DACP-applied sLLMs (ixi-GEN). Through extensive experiments and real-world evaluations, we demonstrate that ixi-GEN models achieve substantial gains in target-domain performance while preserving general capabilities, offering a cost-efficient and scalable solution for enterprise-level deployment.
>
---
#### [replaced 071] More Documents, Same Length: Isolating the Challenge of Multiple Documents in RAG
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04388v2](http://arxiv.org/pdf/2503.04388v2)**

> **作者:** Shahar Levy; Nir Mazor; Lihi Shalmon; Michael Hassid; Gabriel Stanovsky
>
> **备注:** Preprint
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances the accuracy of Large Language Model (LLM) responses by leveraging relevant external documents during generation. Although previous studies noted that retrieving many documents can degrade performance, they did not isolate how the quantity of documents affects performance while controlling for context length. We evaluate various language models on custom datasets derived from a multi-hop QA task. We keep the context length and position of relevant information constant while varying the number of documents, and find that increasing the document count in RAG settings poses significant challenges for most LLMs, reducing performance by up to 20%. However, Qwen2.5 maintained consistent results across increasing document counts, indicating better multi-document handling capability. Finally, our results indicate that processing multiple documents is a separate challenge from handling long contexts. We also make the datasets and code available: https://github.com/shaharl6000/MoreDocsSameLen .
>
---
#### [replaced 072] Adapting Multilingual Models to Code-Mixed Tasks via Model Merging
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.19782v2](http://arxiv.org/pdf/2510.19782v2)**

> **作者:** Prashant Kodali; Vaishnavi Shivkumar; Swarang Joshi; Monojit Choudhary; Ponnurangam Kumaraguru; Manish Shrivastava
>
> **备注:** 9 pages, 5 tables, CODS 2025
>
> **摘要:** We study model merging as a practical alternative to conventional adaptation strategies for code-mixed NLP. Starting from a multilingual base model, we: (i) perform continued pre-training (CPT) on unlabeled code-mixed text to obtain an adapted checkpoint, (ii) merge checkpoint with the base model, and (iii) fine-tune (FT) on the downstream task data. We evaluate our approach for sentence classification (sentiment and hate speech) task in English-Hindi (En-Hi) and English-Spanish (En-Es) using XLM-R and Llama-3.2-1B models. Our results show that merged models consistently outperform full fine-tuning and CPT->FT. We observe gains of 2--5 points in F1 over full fine-tuning and ~1-2 points over CPT->FT, indicating that unlabeled data is leveraged more effectively via merging than via CPT alone. Zero-/few-shot prompting with larger LLMs (e.g., Llama-3.3-70B) lags behind fine-tuned and merged checkpoints, underscoring limits of in-context learning for code-mixed inputs. We further test cross-pair transfer by training on En-Hi and evaluating on En-Ta and En-Ml: merged checkpoints transfer more strongly than monolingual-English baselines (e.g., TV/TIES variants reaching 0.65-0.68 F1 vs 0.61-0.63 for full fine-tuning), suggesting that code-mixed knowledge is a more reliable substrate for low-resource pairs. We conclude with adaptation recipes matched to common data regimes (labeled only; labeled+unlabeled; transfer-only) and discuss limitations and scaling considerations for broader tasks and larger models.
>
---
#### [replaced 073] Bi-Mamba: Towards Accurate 1-Bit State Space Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.11843v2](http://arxiv.org/pdf/2411.11843v2)**

> **作者:** Shengkun Tang; Liqun Ma; Haonan Li; Mingjie Sun; Zhiqiang Shen
>
> **备注:** Accepted in TMLR 2025
>
> **摘要:** The typical Selective State-Space Model (SSM) used in Mamba addresses several limitations of Transformers, such as the quadratic computational complexity with respect to sequence length and the significant memory requirements during inference due to the key-value (KV) cache. However, the increasing size of Mamba models continues to pose challenges for training and deployment, particularly due to their substantial computational demands during both training and inference. In this work, we introduce $\texttt{Bi-Mamba}$, a scalable and powerful 1-bit Mamba architecture designed to enable more efficient large language models (LLMs), with model sizes of 780M, 1.3B, and 2.7B parameters. $\texttt{Bi-Mamba}$ models are trained from scratch on a standard LLM-scale dataset using an autoregressive distillation loss. Extensive experiments on language modeling benchmarks demonstrate that $\texttt{Bi-Mamba}$ achieves performance comparable to its full-precision (FP16 or BF16) counterparts, while outperforming post-training binarization (PTB) Mamba and binarization-aware training (BAT) Transformer baselines. Moreover, $\texttt{Bi-Mamba}$ drastically reduces memory usage and computational cost compared to the original Mamba. Our work pioneers a new line of linear-complexity LLMs under low-bit representation and provides the way for the design of specialized hardware optimized for efficient 1-bit Mamba-based models. Code and the pre-trained weights are available at https://github.com/Tangshengku/Bi-Mamba.
>
---
#### [replaced 074] MLMA: Towards Multilingual ASR With Mamba-based Architectures
- **分类: cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.18684v2](http://arxiv.org/pdf/2510.18684v2)**

> **作者:** Mohamed Nabih Ali; Daniele Falavigna; Alessio Brutti
>
> **备注:** The paper is under review at ICASSP 2026
>
> **摘要:** Multilingual automatic speech recognition (ASR) remains a challenging task, especially when balancing performance across high- and low-resource languages. Recent advances in sequence modeling suggest that architectures beyond Transformers may offer better scalability and efficiency. In this work, we introduce MLMA (Multilingual Language Modeling with Mamba for ASR), a new approach that leverages the Mamba architecture -- an efficient state-space model optimized for long-context sequence processing -- for multilingual ASR. Using Mamba, MLMA implicitly incorporates language-aware conditioning and shared representations to support robust recognition across diverse languages. Experiments on standard multilingual benchmarks show that MLMA achieves competitive performance compared to Transformer-based architectures. These results highlight Mamba's potential as a strong backbone for scalable, efficient, and accurate multilingual speech recognition.
>
---
#### [replaced 075] SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14667v4](http://arxiv.org/pdf/2505.14667v4)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Minsuk Kahng; Albert No
>
> **备注:** Accepted at NeurIPS 2025. Code and models are available at https://ai-isl.github.io/safepath
>
> **摘要:** Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI.
>
---
#### [replaced 076] Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19678v3](http://arxiv.org/pdf/2505.19678v3)**

> **作者:** Hao Fang; Changle Zhou; Jiawei Kong; Kuofeng Gao; Bin Chen; Shu-Tao Xia
>
> **摘要:** Large Vision-Language Models (LVLMs) are susceptible to hallucinations, where generated responses seem semantically plausible yet exhibit little or no relevance to the input image. Previous studies reveal that this issue primarily stems from LVLMs' over-reliance on language priors while disregarding the visual information during decoding. To alleviate this issue, we introduce a novel Conditional Pointwise Mutual Information (C-PMI) calibrated decoding strategy, which adaptively strengthens the mutual dependency between generated texts and input images to mitigate hallucinations. Unlike existing methods solely focusing on text token sampling, we propose to jointly model the contributions of visual and textual tokens to C-PMI, formulating hallucination mitigation as a bi-level optimization problem aimed at maximizing mutual information. To solve it, we design a token purification mechanism that dynamically regulates the decoding process by sampling text tokens remaining maximally relevant to the given image, while simultaneously refining image tokens most pertinent to the generated response. Extensive experiments across various benchmarks reveal that the proposed method significantly reduces hallucinations in LVLMs while preserving decoding efficiency.
>
---
#### [replaced 077] Curing Miracle Steps in LLM Mathematical Reasoning with Rubric Rewards
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07774v2](http://arxiv.org/pdf/2510.07774v2)**

> **作者:** Youliang Yuan; Qiuyang Mang; Jingbang Chen; Hong Wan; Xiaoyuan Liu; Junjielong Xu; Jen-tse Huang; Wenxuan Wang; Wenxiang Jiao; Pinjia He
>
> **备注:** 25 pages, 11 figures, 6 Tables
>
> **摘要:** Large language models for mathematical reasoning are typically trained with outcome-based rewards, which credit only the final answer. In our experiments, we observe that this paradigm is highly susceptible to reward hacking, leading to a substantial overestimation of a model's reasoning ability. This is evidenced by a high incidence of false positives - solutions that reach the correct final answer through an unsound reasoning process. Through a systematic analysis with human verification, we establish a taxonomy of these failure modes, identifying patterns like Miracle Steps - abrupt jumps to a correct output without a valid preceding derivation. Probing experiments suggest a strong association between these Miracle Steps and memorization, where the model appears to recall the answer directly rather than deriving it. To mitigate this systemic issue, we introduce the Rubric Reward Model (RRM), a process-oriented reward function that evaluates the entire reasoning trajectory against problem-specific rubrics. The generative RRM provides fine-grained, calibrated rewards (0-1) that explicitly penalize logical flaws and encourage rigorous deduction. When integrated into a reinforcement learning pipeline, RRM-based training consistently outperforms outcome-only supervision across four math benchmarks. Notably, it boosts Verified Pass@1024 on AIME2024 from 26.7% to 62.6% and reduces the incidence of Miracle Steps by 71%. Our work demonstrates that rewarding the solution process is crucial for building models that are not only more accurate but also more reliable.
>
---
#### [replaced 078] MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14101v2](http://arxiv.org/pdf/2505.14101v2)**

> **作者:** Ernests Lavrinovics; Russa Biswas; Katja Hose; Johannes Bjerva
>
> **摘要:** Large Language Models (LLMs) have inherent limitations of faithfulness and factuality, commonly referred to as hallucinations. Several benchmarks have been developed that provide a test bed for factuality evaluation within the context of English-centric datasets, while relying on supplementary informative context like web links or text passages but ignoring the available structured factual resources. To this end, Knowledge Graphs (KGs) have been identified as a useful aid for hallucination mitigation, as they provide a structured way to represent the facts about entities and their relations with minimal linguistic overhead. We bridge the lack of KG paths and multilinguality for factual language modeling within the existing hallucination evaluation benchmarks and propose a KG-based multilingual, multihop benchmark called MultiHal framed for generative text evaluation. As part of our data collection pipeline, we mined 140k KG-paths from open-domain KGs, from which we pruned noisy KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation shows an absolute scale improvement by approximately 0.12 to 0.36 points for the semantic similarity score, 0.16 to 0.36 for NLI entailment and 0.29 to 0.42 for hallucination detection in KG-RAG over vanilla QA across multiple languages and multiple models, demonstrating the potential of KG integration. We anticipate MultiHal will foster future research towards several graph-based hallucination mitigation and fact-checking tasks.
>
---
#### [replaced 079] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01243v3](http://arxiv.org/pdf/2501.01243v3)**

> **作者:** Lixiong Qin; Shilong Ou; Miaoxuan Zhang; Jiangning Wei; Yuhang Zhang; Xiaoshuai Song; Yuchen Liu; Mei Wang; Weiran Xu
>
> **备注:** 50 pages, 14 figures, 42 tables. NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Faces and humans are crucial elements in social interaction and are widely included in everyday photos and videos. Therefore, a deep understanding of faces and humans will enable multi-modal assistants to achieve improved response quality and broadened application scope. Currently, the multi-modal assistant community lacks a comprehensive and scientific evaluation of face and human understanding abilities. In this paper, we first propose a hierarchical ability taxonomy that includes three levels of abilities. Then, based on this taxonomy, we collect images and annotations from publicly available datasets in the face and human community and build a semi-automatic data pipeline to produce problems for the new benchmark. Finally, the obtained Face-Human-Bench includes a development set and a test set, each with 1800 problems, supporting both English and Chinese. We conduct evaluations over 25 mainstream multi-modal large language models (MLLMs) with our Face-Human-Bench, focusing on the correlation between abilities, the impact of the relative position of targets on performance, and the impact of Chain of Thought (CoT) prompting on performance. We also explore which abilities of MLLMs need to be supplemented by specialist models. The dataset and evaluation code have been made publicly available at https://face-human-bench.github.io.
>
---
#### [replaced 080] MLP Memory: A Retriever-Pretrained Memory for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01832v3](http://arxiv.org/pdf/2508.01832v3)**

> **作者:** Rubin Wei; Jiaqi Cao; Jiarui Wang; Jushi Kai; Qipeng Guo; Bowen Zhou; Zhouhan Lin
>
> **摘要:** Modern approaches to enhancing Large Language Models' factual accuracy and knowledge utilization face a fundamental trade-off: non-parametric retrieval-augmented generation (RAG) provides flexible access to external knowledge but suffers from high inference latency and shallow integration, while parametric fine-tuning methods like LoRA risk catastrophic forgetting and degraded general capabilities. In this work, we propose MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns without explicit document access. By pretraining an MLP to imitate a $k$NN retriever's behavior on the entire pretraining dataset, we create a differentiable memory component that captures the benefits of retrieval-based knowledge access in a fully parametric form. Our architecture integrates this pretrained MLP Memory with Transformer decoders through simple probability interpolation, yielding 17.5\% and 24.1\% scaling gains on WikiText-103 and Web datasets, respectively. It further achieves 12.3\% relative improvement on five question-answering benchmarks and 5.2 points absolute gain across nine general NLP tasks, while reducing hallucinations by up to 10 points on HaluEval. Moreover, MLP Memory delivers 2.5$\times$ faster inference than RAG with superior accuracy. Our findings show that learning retrieval patterns parametrically bridges the gap between efficient inference and effective knowledge access, offering a practical alternative to both RAG and fine-tuning approaches.
>
---
#### [replaced 081] On the Emergence of Linear Analogies in Word Embeddings
- **分类: cs.CL; cond-mat.dis-nn; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18651v2](http://arxiv.org/pdf/2505.18651v2)**

> **作者:** Daniel J. Korchinski; Dhruva Karkada; Yasaman Bahri; Matthieu Wyart
>
> **备注:** Main: 10 pages, 3 figures. Appendices: 11 pages, 7 figures. Accepted at NeurIPS 2025 as a poster
>
> **摘要:** Models such as Word2Vec and GloVe construct word embeddings based on the co-occurrence probability $P(i,j)$ of words $i$ and $j$ in text corpora. The resulting vectors $W_i$ not only group semantically similar words but also exhibit a striking linear analogy structure -- for example, $W_{\text{king}} - W_{\text{man}} + W_{\text{woman}} \approx W_{\text{queen}}$ -- whose theoretical origin remains unclear. Previous observations indicate that this analogy structure: (i) already emerges in the top eigenvectors of the matrix $M(i,j) = P(i,j)/P(i)P(j)$, (ii) strengthens and then saturates as more eigenvectors of $M (i, j)$, which controls the dimension of the embeddings, are included, (iii) is enhanced when using $\log M(i,j)$ rather than $M(i,j)$, and (iv) persists even when all word pairs involved in a specific analogy relation (e.g., king-queen, man-woman) are removed from the corpus. To explain these phenomena, we introduce a theoretical generative model in which words are defined by binary semantic attributes, and co-occurrence probabilities are derived from attribute-based interactions. This model analytically reproduces the emergence of linear analogy structure and naturally accounts for properties (i)-(iv). It can be viewed as giving fine-grained resolution into the role of each additional embedding dimension. It is robust to various forms of noise and agrees well with co-occurrence statistics measured on Wikipedia and the analogy benchmark introduced by Mikolov et al.
>
---
#### [replaced 082] Bag of Tricks for Subverting Reasoning-based Safety Guardrails
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11570v2](http://arxiv.org/pdf/2510.11570v2)**

> **作者:** Shuo Chen; Zhen Han; Haokun Chen; Bailan He; Shengyun Si; Jingpei Wu; Philip Torr; Volker Tresp; Jindong Gu
>
> **备注:** OpenAI Red-teaming Challenge Winner and Oral Presentation
>
> **摘要:** Recent reasoning-based safety guardrails for Large Reasoning Models (LRMs), such as deliberative alignment, have shown strong defense against jailbreak attacks. By leveraging LRMs' reasoning ability, these guardrails help the models to assess the safety of user inputs before generating final responses. The powerful reasoning ability can analyze the intention of the input query and will refuse to assist once it detects the harmful intent hidden by the jailbreak methods. Such guardrails have shown a significant boost in defense, such as the near-perfect refusal rates on the open-source gpt-oss series. Unfortunately, we find that these powerful reasoning-based guardrails can be extremely vulnerable to subtle manipulation of the input prompts, and once hijacked, can lead to even more harmful results. Specifically, we first uncover a surprisingly fragile aspect of these guardrails: simply adding a few template tokens to the input prompt can successfully bypass the seemingly powerful guardrails and lead to explicit and harmful responses. To explore further, we introduce a bag of jailbreak methods that subvert the reasoning-based guardrails. Our attacks span white-, gray-, and black-box settings and range from effortless template manipulations to fully automated optimization. Along with the potential for scalable implementation, these methods also achieve alarmingly high attack success rates (e.g., exceeding 90% across 5 different benchmarks on gpt-oss series on both local host models and online API services). Evaluations across various leading open-source LRMs confirm that these vulnerabilities are systemic, underscoring the urgent need for stronger alignment techniques for open-sourced LRMs to prevent malicious misuse. Code is open-sourced at https://chenxshuo.github.io/bag-of-tricks.
>
---
#### [replaced 083] S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.09068v2](http://arxiv.org/pdf/2505.09068v2)**

> **作者:** Jennifer Haase; Paul H. P. Hanel; Sebastian Pokutta
>
> **摘要:** This paper introduces S-DAT (Synthetic-Divergent Association Task), a scalable, multilingual framework for automated assessment of divergent thinking (DT) -a core component of human creativity. Traditional creativity assessments are often labor-intensive, language-specific, and reliant on subjective human ratings, limiting their scalability and cross-cultural applicability. In contrast, S-DAT leverages large language models and advanced multilingual embeddings to compute semantic distance -- a language-agnostic proxy for DT. We evaluate S-DAT across eleven diverse languages, including English, Spanish, German, Russian, Hindi, and Japanese (Kanji, Hiragana, Katakana), demonstrating robust and consistent scoring across linguistic contexts. Unlike prior DAT approaches, the S-DAT shows convergent validity with other DT measures and correct discriminant validity with convergent thinking. This cross-linguistic flexibility allows for more inclusive, global-scale creativity research, addressing key limitations of earlier approaches. S-DAT provides a powerful tool for fairer, more comprehensive evaluation of cognitive flexibility in diverse populations and can be freely assessed online: https://sdat.iol.zib.de/.
>
---
#### [replaced 084] Text to Band Gap: Pre-trained Language Models as Encoders for Semiconductor Band Gap Prediction
- **分类: cs.CL; cond-mat.mtrl-sci**

- **链接: [http://arxiv.org/pdf/2501.03456v3](http://arxiv.org/pdf/2501.03456v3)**

> **作者:** Ying-Ting Yeh; Janghoon Ock; Achuth Chandrasekhar; Shagun Maheshwari; Amir Barati Farimani
>
> **摘要:** We investigate transformer-based language models, including RoBERTa, T5, Llama-3, and MatSciBERT, for predicting the band gaps of semiconductor materials directly from textual descriptions. The inputs encode key material features, such as chemical composition, crystal system, space group, and other structural and electronic properties. Unlike shallow machine learning models, which require extensive feature engineering, or Graph Neural Networks, which rely on graph representations derived from atomic coordinates, pretrained language models can process textual inputs directly, eliminating the need for manual feature preprocessing or structure-based encoding. Material descriptions were constructed in two formats: structured strings with a consistent template and natural language narratives generated via the ChatGPT API. Each model was augmented with a custom regression head and finetuned for band gap prediction task. Language models of different architectures and parameter sizes were all able to predict band gaps from human-readable text with strong accuracy, achieving MAEs in the range of 0.25-0.33 eV, highlighting the success of this approach for scientific regression tasks. Finetuned Llama-3, with 1.2 billion parameters, achieved the highest accuracy (MAE 0.248 eV, R2 0.891). MatSciBERT, pretrained on materials science literature, reached comparable performance (MAE 0.288 eV, R2 0.871) with significantly fewer parameters (110 million), emphasizing the importance of domain-specific pretraining. Attention analysis shows that both models selectively focus on compositional and spin-related features while de-emphasizing geometric features, reflecting the difficulty of capturing spatial information from text. These results establish that pretrained language models can effectively extract complex feature-property relationships from textual material descriptions.
>
---
#### [replaced 085] Superposition Yields Robust Neural Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10465v3](http://arxiv.org/pdf/2505.10465v3)**

> **作者:** Yizhou Liu; Ziming Liu; Jeff Gore
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law, that loss decreases as a power law with model size, remains unclear. We propose that representation superposition, meaning that LLMs represent more features than they have dimensions, can be a key contributor to loss and cause neural scaling. Based on Anthropic's toy model, we use weight decay to control the degree of superposition, allowing us to systematically study how loss scales with model size. When superposition is weak, the loss follows a power law only if data feature frequencies are power-law distributed. In contrast, under strong superposition, the loss generically scales inversely with model dimension across a broad class of frequency distributions, due to geometric overlaps between representation vectors. We confirmed that open-sourced LLMs operate in the strong superposition regime and have loss scaling like one over the model dimension, and that the Chinchilla scaling laws are also consistent with this behavior. Our results identify representation superposition as a central driver of neural scaling laws, providing insights into questions like when neural scaling laws can be improved and when they will break down.
>
---
