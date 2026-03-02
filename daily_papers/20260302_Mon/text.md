# 自然语言处理 cs.CL

- **最新发布 59 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] LFQA-HP-1M: A Large-Scale Human Preference Dataset for Long-Form Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于长文本问答任务，旨在解决现有评估指标无法反映人类判断的问题。构建了LFQA-HP-1M数据集，并提出评估标准，验证了简单模型的有效性。**

- **链接: [https://arxiv.org/pdf/2602.23603](https://arxiv.org/pdf/2602.23603)**

> **作者:** Rafid Ishrak Jahan; Fahmid Shahriar Iqbal; Sagnik Ray Choudhury
>
> **备注:** LREC 2026 Accepted. this https URL
>
> **摘要:** Long-form question answering (LFQA) demands nuanced evaluation of multi-sentence explanatory responses, yet existing metrics often fail to reflect human judgment. We present LFQA-HP-1M, a large-scale dataset comprising 1.3M human pairwise preference annotations for LFQA. We propose nine rubrics for answer quality evaluation, and show that simple linear models based on these features perform comparably to state-of-the-art LLM evaluators. We further examine transitivity consistency, positional bias, and verbosity biases in LLM evaluators and demonstrate their vulnerability to adversarial perturbations. Overall, this work provides one of the largest public LFQA preference datasets and a rubric-driven framework for transparent and reliable evaluation.
>
---
#### [new 002] MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的对话评估任务，旨在解决多轮协作中私有信息沟通的问题。通过设计协作游戏，评估语言模型在多轮交互中的表现，发现其在规划和执行协作对话方面存在不足。**

- **链接: [https://arxiv.org/pdf/2602.24188](https://arxiv.org/pdf/2602.24188)**

> **作者:** Jacob Eisenstein; Fantine Huot; Adam Fisch; Jonathan Berant; Mirella Lapata
>
> **摘要:** We present a scalable methodology for evaluating language models in multi-turn interactions, using a suite of collaborative games that require effective communication about private information. This enables an interactive scaling analysis, in which a fixed token budget is divided over a variable number of turns. We find that in many cases, language models are unable to use interactive collaboration to improve over the non-interactive baseline scenario in which one agent attempts to summarize its information and the other agent immediately acts -- despite substantial headroom. This suggests that state-of-the-art models still suffer from significant weaknesses in planning and executing multi-turn collaborative conversations. We analyze the linguistic features of these dialogues, assessing the roles of sycophancy, information density, and discourse coherence. While there is no single linguistic explanation for the collaborative weaknesses of contemporary language models, we note that humans achieve comparable task success at superior token efficiency by producing dialogues that are more coherent than those produced by most language models. The proactive management of private information is a defining feature of real-world communication, and we hope that MT-PingEval will drive further work towards improving this capability.
>
---
#### [new 003] CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing
- **分类: cs.CL**

- **简介: 该论文提出CLFEC任务，解决段落级中文专业写作中语言与事实错误的联合校正问题。构建多领域数据集，研究LLM校正方法，发现统一处理优于分离流程。**

- **链接: [https://arxiv.org/pdf/2602.23845](https://arxiv.org/pdf/2602.23845)**

> **作者:** Jian Kai; Zidong Zhang; Jiwen Chen; Zhengxiang Wu; Songtao Sun; Fuyang Li; Yang Cao; Qiang Liu
>
> **摘要:** Chinese text correction has traditionally focused on spelling and grammar, while factual error correction is usually treated separately. However, in paragraph-level Chinese professional writing, linguistic (word/grammar/punctuation) and factual errors frequently co-occur and interact, making unified correction both necessary and challenging. This paper introduces CLFEC (Chinese Linguistic & Factual Error Correction), a new task for joint linguistic and factual correction. We construct a mixed, multi-domain Chinese professional writing dataset spanning current affairs, finance, law, and medicine. We then conduct a systematic study of LLM-based correction paradigms, from prompting to retrieval-augmented generation (RAG) and agentic workflows. The analysis reveals practical challenges, including limited generalization of specialized correction models, the need for evidence grounding for factual repair, the difficulty of mixed-error paragraphs, and over-correction on clean inputs. Results further show that handling linguistic and factual Error within the same context outperform decoupled processes, and that agentic workflows can be effective with suitable backbone models. Overall, our dataset and empirical findings provide guidance for building reliable, fully automatic proofreading systems in industrial settings.
>
---
#### [new 004] MemEmo: Evaluating Emotion in Memory Systems of Agents
- **分类: cs.CL**

- **简介: 该论文属于情感记忆评估任务，旨在解决记忆系统处理情感信息能力不足的问题。通过构建HLME数据集，评估现有系统的三方面表现。**

- **链接: [https://arxiv.org/pdf/2602.23944](https://arxiv.org/pdf/2602.23944)**

> **作者:** Peng Liu; Zhen Tao; Jihao Zhao; Ding Chen; Yansong Zhang; Cuiping Li; Zhiyu Li; Hong Chen
>
> **摘要:** Memory systems address the challenge of context loss in Large Language Model during prolonged interactions. However, compared to human cognition, the efficacy of these systems in processing emotion-related information remains inconclusive. To address this gap, we propose an emotion-enhanced memory evaluation benchmark to assess the performance of mainstream and state-of-the-art memory systems in handling affective information. We developed the \textbf{H}uman-\textbf{L}ike \textbf{M}emory \textbf{E}motion (\textbf{HLME}) dataset, which evaluates memory systems across three dimensions: emotional information extraction, emotional memory updating, and emotional memory question answering. Experimental results indicate that none of the evaluated systems achieve robust performance across all three tasks. Our findings provide an objective perspective on the current deficiencies of memory systems in processing emotional memories and suggest a new trajectory for future research and system optimization.
>
---
#### [new 005] Preference Packing: Efficient Preference Optimization for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型训练优化任务，旨在提升资源效率。针对多回复数据训练中的重复输入问题，提出偏好打包方法，减少计算和内存消耗，实验显示训练时间减少37%以上。**

- **链接: [https://arxiv.org/pdf/2602.24082](https://arxiv.org/pdf/2602.24082)**

> **作者:** Jaekyung Cho
>
> **摘要:** Resource-efficient training optimization techniques are becoming increasingly important as the size of large language models (LLMs) continues to grow. In particular, batch packing is commonly used in pre-training and supervised fine-tuning to achieve resource-efficient training. We propose preference packing, a method to enhance resource efficiency in training techniques that use data with different responses for the same input prompt, such as reward models or Direct Preference Optimization (DPO). Preference packing improves resource efficiency by reducing the attention operations for duplicate input prompts and decreasing KV cache memory usage. We conducted experiments on text-only datasets and image-included datasets and achieved at least 37% reduction in training time. Notably, this method can be applied alongside existing optimization techniques such as batch sorting, resulting in a 3.22x speedup.
>
---
#### [new 006] The Astonishing Ability of Large Language Models to Parse Jabberwockified Language
- **分类: cs.CL**

- **简介: 论文研究LLMs解析被篡改英语文本的能力，属于自然语言理解任务。解决的问题是词汇缺失下语义恢复。工作包括生成干扰文本并测试模型翻译效果。**

- **链接: [https://arxiv.org/pdf/2602.23928](https://arxiv.org/pdf/2602.23928)**

> **作者:** Gary Lupyan; Senyi Yang
>
> **备注:** Submitted to the 2026 Annual Meeting of the Cognitive Science Society
>
> **摘要:** We show that large language models (LLMs) have an astonishing ability to recover meaning from severely degraded English texts. Texts in which content words have been randomly substituted by nonsense strings, e.g., "At the ghybe of the swuint, we are haiveed to Wourge Phrear-gwurr, who sproles into an ghitch flount with his crurp", can be translated to conventional English that is, in many cases, close to the original text, e.g., "At the start of the story, we meet a man, Chow, who moves into an apartment building with his wife." These results show that structural cues (e.g., morphosyntax, closed-class words) constrain lexical meaning to a much larger degree than imagined. Although the abilities of LLMs to make sense of "Jabberwockified" English are clearly superhuman, they are highly relevant to understanding linguistic structure and suggest that efficient language processing either in biological or artificial systems likely benefits from very tight integration between syntax, lexical semantics, and general world knowledge.
>
---
#### [new 007] Structured Prompt Optimization for Few-Shot Text Classification via Semantic Alignment in Latent Space
- **分类: cs.CL**

- **简介: 该论文属于少样本文本分类任务，旨在解决语义纠缠、标签结构不清晰和特征表示不足的问题。通过结构化提示优化框架，提升语义理解和任务适应能力。**

- **链接: [https://arxiv.org/pdf/2602.23753](https://arxiv.org/pdf/2602.23753)**

> **作者:** Jiasen Zheng; Zijun Zhou; Huajun Zhang; Junjiang Lin; Jingyun Jia; Qi Wang
>
> **摘要:** This study addresses the issues of semantic entanglement, unclear label structure, and insufficient feature representation in few-shot text classification, and proposes an optimization framework based on structured prompts to enhance semantic understanding and task adaptation under low-resource conditions. The framework first uses a pretrained language model to encode the input text and obtain basic semantic representations. It then introduces structured prompts composed of multi-dimensional semantic factors and integrates them with text features through a learnable combination mechanism, which forms task-related representations with clear boundaries in the latent space. To further strengthen the consistency between text representations and label semantics, the method constructs a structured label embedding matrix and employs a cross-space alignment mechanism to ensure stable matching between textual features and label attributes. In addition, the model applies prompt orthogonality constraints and a joint optimization objective to maintain independence across different semantic factors in the prompts, allowing the structured prompts to provide transparent and controllable guidance for classification decisions. Three types of sensitivity experiments, including learning rate sensitivity, prompt length sensitivity, and data scale sensitivity, are designed to evaluate the stability and robustness of the framework under different conditions. Experimental results show that the proposed structured prompt optimization framework effectively alleviates semantic conflicts and label ambiguity in few-shot text classification. It significantly improves performance on accuracy, precision, recall, and AUC, and demonstrates strong cross-task applicability.
>
---
#### [new 008] Task Complexity Matters: An Empirical Study of Reasoning in LLMs for Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLMs在情感分析任务中的推理效果，探讨推理对不同复杂度任务的影响。通过实验发现推理效果依赖任务复杂度，简单任务性能下降，复杂任务提升。**

- **链接: [https://arxiv.org/pdf/2602.24060](https://arxiv.org/pdf/2602.24060)**

> **作者:** Donghao Huang; Zhaoxia Wang
>
> **备注:** 12 pages, 1 figure, 3 tables. Accepted at PAKDD 2026
>
> **摘要:** Large language models (LLMs) with reasoning capabilities have fueled a compelling narrative that reasoning universally improves performance across language tasks. We test this claim through a comprehensive evaluation of 504 configurations across seven model families--including adaptive, conditional, and reinforcement learning-based reasoning architectures--on sentiment analysis datasets of varying granularity (binary, five-class, and 27-class emotion). Our findings reveal that reasoning effectiveness is strongly task-dependent, challenging prevailing assumptions: (1) Reasoning shows task-complexity dependence--binary classification degrades up to -19.9 F1 percentage points (pp), while 27-class emotion recognition gains up to +16.0pp; (2) Distilled reasoning variants underperform base models by 3-18 pp on simpler tasks, though few-shot prompting enables partial recovery; (3) Few-shot learning improves over zero-shot in most cases regardless of model type, with gains varying by architecture and task complexity; (4) Pareto frontier analysis shows base models dominate efficiency-performance trade-offs, with reasoning justified only for complex emotion recognition despite 2.1x-54x computational overhead. We complement these quantitative findings with qualitative error analysis revealing that reasoning degrades simpler tasks through systematic over-deliberation, offering mechanistic insight beyond the high-level overthinking hypothesis.
>
---
#### [new 009] FHIRPath-QA: Executable Question Answering over FHIR Electronic Health Records
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，旨在解决患者提问难以获得准确答案的问题。提出FHIRPath-QA数据集和文本转FHIRPath问答方法，提升问答效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.23479](https://arxiv.org/pdf/2602.23479)**

> **作者:** Michael Frew; Nishit Bheda; Bryan Tripp
>
> **备注:** Submitted to LREC 2026 CL4Health Workshop
>
> **摘要:** Though patients are increasingly granted digital access to their electronic health records (EHRs), existing interfaces may not support precise, trustworthy answers to patient-specific questions. Large language models (LLM) show promise in clinical question answering (QA), but retrieval-based approaches are computationally inefficient, prone to hallucination, and difficult to deploy over real-life EHRs. In this work, we introduce FHIRPath-QA, the first open dataset and benchmark for patient-specific QA that includes open-standard FHIRPath queries over real-world clinical data. We propose a text-to-FHIRPath QA paradigm that shifts reasoning from free-text generation to FHIRPath query synthesis, significantly reducing LLM usage. Built on MIMIC-IV on FHIR Demo, the dataset pairs over 14k natural language questions in patient and clinician phrasing with validated FHIRPath queries and answers. Further, we demonstrate that state-of-the-art LLMs struggle to deal with ambiguity in patient language and perform poorly in FHIRPath query synthesis. However, they benefit strongly from supervised fine-tuning. Our results highlight that text-to-FHIRPath synthesis has the potential to serve as a practical foundation for safe, efficient, and interoperable consumer health applications, and our dataset and benchmark serve as a starting point for future research on the topic. The full dataset and generation code is available at: this https URL.
>
---
#### [new 010] Controllable Reasoning Models Are Private Thinkers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私保护任务，旨在解决AI代理在推理过程中泄露用户数据的问题。通过优化模型的指令遵循能力，提升其隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2602.24210](https://arxiv.org/pdf/2602.24210)**

> **作者:** Haritz Puerto; Haonan Li; Xudong Han; Timothy Baldwin; Iryna Gurevych
>
> **摘要:** AI agents powered by reasoning models require access to sensitive user data. However, their reasoning traces are difficult to control, which can result in the unintended leakage of private information to external parties. We propose training models to follow instructions not only in the final answer, but also in reasoning traces, potentially under different constraints. We hypothesize that improving their instruction following abilities in the reasoning traces can improve their privacy-preservation skills. To demonstrate this, we fine-tune models on a new instruction-following dataset with explicit restrictions on reasoning traces. We further introduce a generation strategy that decouples reasoning and answer generation using separate LoRA adapters. We evaluate our approach on six models from two model families, ranging from 1.7B to 14B parameters, across two instruction-following benchmarks and two privacy benchmarks. Our method yields substantial improvements, achieving gains of up to 20.9 points in instruction-following performance and up to 51.9 percentage points on privacy benchmarks. These improvements, however, can come at the cost of task utility, due to the trade-off between reasoning performance and instruction-following abilities. Overall, our results show that improving instruction-following behavior in reasoning models can significantly enhance privacy, suggesting a promising direction for the development of future privacy-aware agents. Our code and data are available at this https URL
>
---
#### [new 011] France or Spain or Germany or France: A Neural Account of Non-Redundant Redundant Disjunctions
- **分类: cs.CL**

- **简介: 该论文研究语言模型中非冗余冗余析取现象，属于自然语言理解任务。解决语言模型如何处理看似冗余但实际可接受的句子结构问题，通过行为实验和模型分析，揭示其神经机制。**

- **链接: [https://arxiv.org/pdf/2602.23547](https://arxiv.org/pdf/2602.23547)**

> **作者:** Sasha Boguraev; Qing Yao; Kyle Mahowald
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Sentences like "She will go to France or Spain, or perhaps to Germany or France." appear formally redundant, yet become acceptable in contexts such as "Mary will go to a philosophy program in France or Spain, or a mathematics program in Germany or France." While this phenomenon has typically been analyzed using symbolic formal representations, we aim to provide a complementary account grounded in artificial neural mechanisms. We first present new behavioral evidence from humans and large language models demonstrating the robustness of this apparent non-redundancy across contexts. We then show that, in language models, redundancy avoidance arises from two interacting mechanisms: models learn to bind contextually relevant information to repeated lexical items, and Transformer induction heads selectively attend to these context-licensed representations. We argue that this neural explanation sheds light on the mechanisms underlying context-sensitive semantic interpretation, and that it complements existing symbolic analyses.
>
---
#### [new 012] The GRADIEND Python Package: An End-to-End System for Gradient-Based Feature Learning
- **分类: cs.CL**

- **简介: 该论文介绍了一个名为GRADIEND的Python包，用于基于梯度的特征学习，解决语言模型中特征方向提取问题，实现了数据生成、训练、评估和多特征比较等任务。**

- **链接: [https://arxiv.org/pdf/2602.23993](https://arxiv.org/pdf/2602.23993)**

> **作者:** Jonathan Drechsel; Steffen Herbold
>
> **摘要:** We present gradiend, an open-source Python package that operationalizes the GRADIEND method for learning feature directions from factual-counterfactual MLM and CLM gradients in language models. The package provides a unified workflow for feature-related data creation, training, evaluation, visualization, persistent model rewriting via controlled weight updates, and multi-feature comparison. We demonstrate GRADIEND on an English pronoun paradigm and on a large-scale feature comparison that reproduces prior use cases.
>
---
#### [new 013] Dialect and Gender Bias in YouTube's Spanish Captioning System
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究YouTube西班牙语字幕系统的方言和性别偏见问题，通过分析不同地区男女说话者的字幕质量，揭示系统存在的系统性差异。**

- **链接: [https://arxiv.org/pdf/2602.24002](https://arxiv.org/pdf/2602.24002)**

> **作者:** Iris Dania Jimenez; Christoph Kern
>
> **备注:** 21 pages, 4 tables
>
> **摘要:** Spanish is the official language of twenty-one countries and is spoken by over 441 million people. Naturally, there are many variations in how Spanish is spoken across these countries. Media platforms such as YouTube rely on automatic speech recognition systems to make their content accessible to different groups of users. However, YouTube offers only one option for automatically generating captions in Spanish. This raises the question: could this captioning system be biased against certain Spanish dialects? This study examines the potential biases in YouTube's automatic captioning system by analyzing its performance across various Spanish dialects. By comparing the quality of captions for female and male speakers from different regions, we identify systematic disparities which can be attributed to specific dialects. Our study provides further evidence that algorithmic technologies deployed on digital platforms need to be calibrated to the diverse needs and experiences of their user populations.
>
---
#### [new 014] From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型评估任务，旨在解决静态数据集无法反映模型推理能力的问题。提出动态协议，通过代理协作自动生成和验证问题，提升评估的准确性和适应性。**

- **链接: [https://arxiv.org/pdf/2602.23729](https://arxiv.org/pdf/2602.23729)**

> **作者:** Seungdong Yoa; Sanghyu Yoon; Suhee Yoon; Dongmin Kim; Ye Seul Sim; Junhyun Lee; Woohyung Lim
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** The evaluation of large language models (LLMs) has predominantly relied on static datasets, which offer limited scalability and fail to capture the evolving reasoning capabilities of recent models. To overcome these limitations, we propose an agent-centric benchmarking paradigm that moves beyond static datasets by introducing a dynamic protocol in which autonomous agents iteratively generate, validate, and solve problems. Within this protocol, a teacher agent generates candidate problems, an orchestrator agent rigorously verifies their validity and guards against adversarial attacks, and a student agent attempts to solve the validated problems. An invalid problem is revised by the teacher agent until it passes validation. If the student correctly solves the problem, the orchestrator prompts the teacher to generate more challenging variants. Consequently, the benchmark scales in difficulty automatically as more capable agents are substituted into any role, enabling progressive evaluation of large language models without manually curated datasets. Adopting text anomaly detection as our primary evaluation format, which demands cross-sentence logical inference and resists pattern-matching shortcuts, we demonstrate that this protocol systematically exposes corner-case reasoning errors that conventional benchmarks fail to reveal. We further advocate evaluating systems along several complementary axes including cross-model pairwise performance and progress between the initial and orchestrator-finalized problems. By shifting the focus from fixed datasets to dynamic protocols, our approach offers a sustainable direction for evaluating ever-evolving language models and introduces a research agenda centered on the co-evolution of agent-centric benchmarks.
>
---
#### [new 015] IDP Accelerator: Agentic Document Intelligence from Extraction to Compliance Validation
- **分类: cs.CL**

- **简介: 该论文提出IDP加速器，解决文档智能处理难题，通过模块化框架实现从提取到合规验证的端到端处理。**

- **链接: [https://arxiv.org/pdf/2602.23481](https://arxiv.org/pdf/2602.23481)**

> **作者:** Md Mofijul Islam; Md Sirajus Salekin; Joe King; Priyashree Roy; Vamsi Thilak Gudi; Spencer Romo; Akhil Nooney; Boyi Xie; Bob Strahan; Diego A. Socolinsky
>
> **摘要:** Understanding and extracting structured insights from unstructured documents remains a foundational challenge in industrial NLP. While Large Language Models (LLMs) enable zero-shot extraction, traditional pipelines often fail to handle multi-document packets, complex reasoning, and strict compliance requirements. We present IDP (Intelligent Document Processing) Accelerator, a framework enabling agentic AI for end-to-end document intelligence with four key components: (1) DocSplit, a novel benchmark dataset and multimodal classifier using BIO tagging to segment complex document packets; (2) configurable Extraction Module leveraging multimodal LLMs to transform unstructured content into structured data; (3) Agentic Analytics Module, compliant with the Model Context Protocol (MCP) providing data access through secure, sandboxed code execution; and (4) Rule Validation Module replacing deterministic engines with LLM-driven logic for complex compliance checks. The interactive demonstration enables users to upload document packets, visualize classification results, and explore extracted data through an intuitive web interface. We demonstrate effectiveness across industries, highlighting a production deployment at a leading healthcare provider achieving 98% classification accuracy, 80% reduced processing latency, and 77% lower operational costs over legacy baselines. IDP Accelerator is open-sourced with a live demonstration available to the community.
>
---
#### [new 016] LLM-Driven Multi-Turn Task-Oriented Dialogue Synthesis for Realistic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于任务导向对话生成领域，旨在解决现有数据集无法反映真实场景的问题。通过LLM驱动的方法生成高质量多轮对话，提升模型的现实推理能力。**

- **链接: [https://arxiv.org/pdf/2602.23610](https://arxiv.org/pdf/2602.23610)**

> **作者:** Yu Zhu; Kai Yang
>
> **摘要:** The reasoning capability of large language models (LLMs), defined as their ability to analyze, infer, and make decisions based on input information, is essential for building intelligent task-oriented dialogue systems. However, existing benchmarks do not sufficiently reflect the complexity of real-world scenarios, which limits their effectiveness in evaluating and enhancing LLM reasoning in practical contexts. Many current reasoning datasets are overly simplistic and abstract, often disconnected from realistic task flows, domain constraints, and operational rules, making it difficult to effectively evaluate LLMs' logical reasoning ability. In addition, data contamination from pretraining corpora undermines the reliability of evaluation results, and traditional crowdsourcing methods for dataset construction are labor-intensive and difficult to scale. To address these challenges, we propose a LLM-driven framework for synthesizing multi-turn, task-oriented dialogues grounded in realistic reasoning scenarios, leveraging trilevel optimization to enhance dialogue quality. Our method generates dialogues grounded in authentic task scenarios, enriched with real-world information, and exhibiting strong contextual coherence. Corresponding reasoning tasks are carefully designed around these dialogues and iteratively refined to continuously improve the tasks' quality and challenge. The resulting dataset serves as a valuable benchmark for assessing and advancing the realistic logical reasoning capabilities of LLMs. Experimental results show that our synthetic data-based reasoning tasks introduce non-trivial reasoning challenges and provide meaningful support for improving the reasoning capabilities of LLMs.
>
---
#### [new 017] Task-Lens: Cross-Task Utility Based Speech Dataset Profiling for Low-Resource Indian Languages
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Task-Lens，用于评估印度语 speech 数据集在多个任务中的适用性，解决低资源语言数据不足问题，通过分析数据集元数据和任务适配性，识别关键缺失领域。**

- **链接: [https://arxiv.org/pdf/2602.23388](https://arxiv.org/pdf/2602.23388)**

> **作者:** Swati Sharma; Divya V. Sharma; Anubha Gupta
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** The rising demand for inclusive speech technologies amplifies the need for multilingual datasets for Natural Language Processing (NLP) research. However, limited awareness of existing task-specific resources in low-resource languages hinders research. This challenge is especially acute in linguistically diverse countries, such as India. Cross-task profiling of existing Indian speech datasets can alleviate the data scarcity challenge. This involves investigating the utility of datasets across multiple downstream tasks rather than focusing on a single task. Prior surveys typically catalogue datasets for a single task, leaving comprehensive cross-task profiling as an open opportunity. Therefore, we propose Task-Lens, a cross-task survey that assesses the readiness of 50 Indian speech datasets spanning 26 languages for nine downstream speech tasks. First, we analyze which datasets contain metadata and properties suitable for specific tasks. Next, we propose task-aligned enhancements to unlock datasets to their full downstream potential. Finally, we identify tasks and Indian languages that are critically underserved by current resources. Our findings reveal that many Indian speech datasets contain untapped metadata that can support multiple downstream tasks. By uncovering cross-task linkages and gaps, Task-Lens enables researchers to explore the broader applicability of existing datasets and to prioritize dataset creation for underserved tasks and languages.
>
---
#### [new 018] Task-Centric Acceleration of Small-Language Models
- **分类: cs.CL; cs.AI; cs.IT**

- **简介: 该论文针对小语言模型的加速问题，提出TASC框架，包含微调和推理两种方法，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2602.24174](https://arxiv.org/pdf/2602.24174)**

> **作者:** Dor Tsur; Sharon Adar; Ran Levy
>
> **摘要:** Small language models (SLMs) have emerged as efficient alternatives to large language models for task-specific applications. However, they are often employed in high-volume, low-latency settings, where efficiency is crucial. We propose TASC, Task-Adaptive Sequence Compression, a framework for SLM acceleration comprising two use-cases: When performing SLM fine-tuning, we propose TASC-ft, which iteratively enriches the tokenizer vocabulary with high-frequency output n-grams and then fine-tunes the model to utilize the expanded vocabulary. Next, we propose an inference-time method, termed TASC-spec. TASC-spec is a lightweight, training-free speculative decoding method that constructs an n-gram draft model from the task's output corpus, mixing task and context n-gram this http URL-spec avoids any additional training, while bypassing draft-target vocabulary alignment constraints. We demonstrate the effectiveness of both methods across multiple low output-variability generation tasks. Our methods show consistent improvements in inference efficiency while maintaining task performance.
>
---
#### [new 019] Humans and LLMs Diverge on Probabilistic Inferences
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言推理任务，探讨人类与LLMs在概率推理上的差异。研究构建了ProbCOPA数据集，对比分析人类与模型的推理表现，揭示两者在非确定性推理中的显著不同。**

- **链接: [https://arxiv.org/pdf/2602.23546](https://arxiv.org/pdf/2602.23546)**

> **作者:** Gaurav Kamath; Sreenath Madathil; Sebastian Schuster; Marie-Catherine de Marneffe; Siva Reddy
>
> **摘要:** Human reasoning often involves working over limited information to arrive at probabilistic conclusions. In its simplest form, this involves making an inference that is not strictly entailed by a premise, but rather only likely given the premise. While reasoning LLMs have demonstrated strong performance on logical and mathematical tasks, their behavior on such open-ended, non-deterministic inferences remains largely unexplored. We introduce ProbCOPA, a dataset of 210 handcrafted probabilistic inferences in English, each annotated for inference likelihood by 25--30 human participants. We find that human responses are graded and varied, revealing probabilistic judgments of the inferences in our dataset. Comparing these judgments with responses from eight state-of-the-art reasoning LLMs, we show that models consistently fail to produce human-like distributions. Finally, analyzing LLM reasoning chains, we find evidence of a common reasoning pattern used to evaluate such inferences. Our findings reveal persistent differences between humans and LLMs, and underscore the need to evaluate reasoning beyond deterministic settings.
>
---
#### [new 020] Multi-Agent Causal Reasoning for Suicide Ideation Detection Through Online Conversations
- **分类: cs.CL**

- **简介: 该论文属于自杀意念检测任务，旨在解决在线对话中风险识别的局限性。提出MACR框架，通过多智能体因果推理提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.23577](https://arxiv.org/pdf/2602.23577)**

> **作者:** Jun Li; Xiangmeng Wang; Haoyang Li; Yifei Yan; Shijie Zhang; Hong Va Leong; Ling Feng; Nancy Xiaonan Yu; Qing Li
>
> **摘要:** Suicide remains a pressing global public health concern. While social media platforms offer opportunities for early risk detection through online conversation trees, existing approaches face two major limitations: (1) They rely on predefined rules (e.g., quotes or relies) to log conversations that capture only a narrow spectrum of user interactions, and (2) They overlook hidden influences such as user conformity and suicide copycat behavior, which can significantly affect suicidal expression and propagation in online communities. To address these limitations, we propose a Multi-Agent Causal Reasoning (MACR) framework that collaboratively employs a Reasoning Agent to scale user interactions and a Bias-aware Decision-Making Agent to mitigate harmful biases arising from hidden influences. The Reasoning Agent integrates cognitive appraisal theory to generate counterfactual user reactions to posts, thereby scaling user interactions. It analyses these reactions through structured dimensions, i.e., cognitive, emotional, and behavioral patterns, with a dedicated sub-agent responsible for each dimension. The Bias-aware Decision-Making Agent mitigates hidden biases through a front-door adjustment strategy, leveraging the counterfactual user reactions produced by the Reasoning Agent. Through the collaboration of reasoning and bias-aware decision making, the proposed MACR framework not only alleviates hidden biases, but also enriches contextual information of user interactions with counterfactual knowledge. Extensive experiments on real-world conversational datasets demonstrate the effectiveness and robustness of MACR in identifying suicide risk.
>
---
#### [new 021] ARGUS: Seeing the Influence of Narrative Features on Persuasion in Argumentative Texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ARGUS框架，研究叙事特征对论点说服力的影响。属于自然语言处理中的说服分析任务，旨在解决在线论辩中叙事作用不明确的问题。通过构建语料库和模型分析，识别关键叙事特征。**

- **链接: [https://arxiv.org/pdf/2602.24109](https://arxiv.org/pdf/2602.24109)**

> **作者:** Sara Nabhani; Federico Pianzola; Khalid Al-Khatib; Malvina Nissim
>
> **备注:** 22 pages, 8 figures, submitted to ACM Transactions on Intelligent Systems and Technology
>
> **摘要:** Can narratives make arguments more persuasive? And to this end, which narrative features matter most? Although stories are often seen as powerful tools for persuasion, their specific role in online, unstructured argumentation remains underexplored. To address this gap, we present ARGUS, a framework for studying the impact of narration on persuasion in argumentative discourse. ARGUS introduces a new ChangeMyView corpus annotated for story presence and six key narrative features, integrating insights from two established theoretical frameworks that capture both textual narrative features and their effects on recipients. Leveraging both encoder-based classifiers and zero-shot large language models (LLMs), ARGUS identifies stories and narrative features and applies them at scale to examine how different narrative dimensions influence persuasion success in online argumentation.
>
---
#### [new 022] EDDA-Coordinata: An Annotated Dataset of Historical Geographic Coordinates
- **分类: cs.CL; cs.DL; cs.IR**

- **简介: 该论文属于历史文本坐标提取任务，旨在解决从古籍中自动恢复地理坐标的问题。通过构建标注数据集并训练模型，提升坐标识别与归一化效果。**

- **链接: [https://arxiv.org/pdf/2602.23941](https://arxiv.org/pdf/2602.23941)**

> **作者:** Ludovic Moncla; Pierre Nugues; Thierry Joliveau; Katherine McDonough
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** This paper introduces a dataset of enriched geographic coordinates retrieved from Diderot and d'Alembert's eighteenth-century Encyclopedie. Automatically recovering geographic coordinates from historical texts is a complex task, as they are expressed in a variety of ways and with varying levels of precision. To improve retrieval of coordinates from similar digitized early modern texts, we have created a gold standard dataset, trained models, published the resulting inferred and normalized coordinate data, and experimented applying these models to new texts. From 74,000 total articles in each of the digitized versions of the Encyclopedie from ARTFL and ENCCRE, we examined 15,278 geographical entries, manually identifying 4,798 containing coordinates, and 10,480 with descriptive but non-numerical references. Leveraging our gold standard annotations, we trained transformer-based models to retrieve and normalize coordinates. The pipeline presented here combines a classifier to identify coordinate-bearing entries and a second model for retrieval, tested across encoder-decoder and decoder architectures. Cross-validation yielded an 86% EM score. On an out-of-domain eighteenth-century Trevoux dictionary (also in French), our fine-tuned model had a 61% EM score, while for the nineteenth-century, 7th edition of the Encyclopaedia Britannica in English, the EM was 77%. These findings highlight the gold standard dataset's usefulness as training data, and our two-step method's cross-lingual, cross-domain generalizability.
>
---
#### [new 023] CoME: Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于移动代理任务，解决混合能力推理的解耦与平衡问题。提出CoME架构及训练策略，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2602.24142](https://arxiv.org/pdf/2602.24142)**

> **作者:** Yuxuan Liu; Weikai Xu; Kun Huang; Changyu Chen; Jiankun Zhao; Pengzhi Gao; Wei Liu; Jian Luan; Shuo Shang; Bo Du; Ji-Rong Wen; Rui Yan
>
> **摘要:** Mobile Agents can autonomously execute user instructions, which requires hybrid-capabilities reasoning, including screen summary, subtask planning, action decision and action function. However, existing agents struggle to achieve both decoupled enhancement and balanced integration of these capabilities. To address these challenges, we propose Channel-of-Mobile-Experts (CoME), a novel agent architecture consisting of four distinct experts, each aligned with a specific reasoning stage, CoME activates the corresponding expert to generate output tokens in each reasoning stage via output-oriented activation. To empower CoME with hybrid-capabilities reasoning, we introduce a progressive training strategy: Expert-FT enables decoupling and enhancement of different experts' capability; Router-FT aligns expert activation with the different reasoning stage; CoT-FT facilitates seamless collaboration and balanced optimization across multiple capabilities. To mitigate error propagation in hybrid-capabilities reasoning, we propose InfoGain-Driven DPO (Info-DPO), which uses information gain to evaluate the contribution of each intermediate step, thereby guiding CoME toward more informative reasoning. Comprehensive experiments show that CoME outperforms dense mobile agents and MoE methods on both AITZ and AMEX datasets.
>
---
#### [new 024] GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍GLUScope工具，用于分析Transformer模型中的GLU神经元。任务是模型可解释性，解决如何理解GLU激活函数中不同符号组合的功能差异。工作包括展示四种符号组合的文本示例及出现频率。**

- **链接: [https://arxiv.org/pdf/2602.23826](https://arxiv.org/pdf/2602.23826)**

> **作者:** Sebastian Gerstner; Hinrich Schütze
>
> **备注:** 6 pages for main body, 9 pages in total. 4 figures
>
> **摘要:** We present GLUScope, an open-source tool for analyzing neurons in Transformer-based language models, intended for interpretability researchers. We focus on more recent models than previous tools do; specifically we consider gated activation functions such as SwiGLU. This introduces a new challenge: understanding positive activations is not enough. Instead, both the gate and the in activation of a neuron can be positive or negative, leading to four different possible sign combinations that in some cases have quite different functionalities. Accordingly, for any neuron, our tool shows text examples for each of the four sign combinations, and indicates how often each combination occurs. We describe examples of how our tool can lead to novel insights. A demo is available at https: //sjgerstner.this http URL.
>
---
#### [new 025] TRIZ-RAGNER: A Retrieval-Augmented Large Language Model for TRIZ-Aware Named Entity Recognition in Patent-Based Contradiction Mining
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于专利矛盾挖掘任务，旨在解决TRIZ参数提取中的语义模糊和领域依赖问题。提出TRIZ-RAGNER框架，结合检索增强和结构化提示，提升参数识别准确率。**

- **链接: [https://arxiv.org/pdf/2602.23656](https://arxiv.org/pdf/2602.23656)**

> **作者:** Zitong Xu; Yuqing Wu; Yue Zhao
>
> **摘要:** TRIZ-based contradiction mining is a fundamental task in patent analysis and systematic innovation, as it enables the identification of improving and worsening technical parameters that drive inventive problem solving. However, existing approaches largely rely on rule-based systems or traditional machine learning models, which struggle with semantic ambiguity, domain dependency, and limited generalization when processing complex patent language. Recently, large language models (LLMs) have shown strong semantic understanding capabilities, yet their direct application to TRIZ parameter extraction remains challenging due to hallucination and insufficient grounding in structured TRIZ knowledge. To address these limitations, this paper proposes TRIZ-RAGNER, a retrieval-augmented large language model framework for TRIZ-aware named entity recognition in patent-based contradiction mining. TRIZ-RAGNER reformulates contradiction mining as a semantic-level NER task and integrates dense retrieval over a TRIZ knowledge base, cross-encoder reranking for context refinement, and structured LLM prompting to extract improving and worsening parameters from patent sentences. By injecting domain-specific TRIZ knowledge into the LLM reasoning process, the proposed framework effectively reduces semantic noise and improves extraction consistency. Experiments on the PaTRIZ dataset demonstrate that TRIZ-RAGNER consistently outperforms traditional sequence labeling models and LLM-based baselines. The proposed framework achieves a precision of 85.6%, a recall of 82.9%, and an F1-score of 84.2% in TRIZ contradiction pair identification. Compared with the strongest baseline using prompt-enhanced GPT, TRIZ-RAGNER yields an absolute F1-score improvement of 7.3 percentage points, confirming the effectiveness of retrieval-augmented TRIZ knowledge grounding for robust and accurate patent-based contradiction mining.
>
---
#### [new 026] Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出SLATE框架，解决大语言模型与搜索引擎结合时的信用分配问题。通过截断步骤采样和密集奖励机制，提升推理与检索效果。**

- **链接: [https://arxiv.org/pdf/2602.23440](https://arxiv.org/pdf/2602.23440)**

> **作者:** Chris Samarinas; Haw-Shiuan Chang; Hamed Zamani
>
> **摘要:** Training large language models to reason with search engines via reinforcement learning is hindered by a fundamental credit assignment problem: existing methods such as Search-R1 provide only a sparse outcome reward after an entire multi-step trajectory, making it infeasible to attribute success or failure to individual reasoning and retrieval decisions. Process-reward methods like StepSearch alleviate this by introducing step-level supervision, but rely on heuristic rewards such as TF-IDF overlap with gold documents, and still sample k complete trajectories per example, retaining high gradient variance. We propose SLATE, a framework built on two complementary ideas: (1) truncated step-level sampling, which generates k trajectories that share a common prefix and differ only at the next step, and (2) dense LLM-as-judge rewards, which replace heuristic scoring with a capable LLM evaluator that assesses the quality of each reasoning step, search query, and answer, providing richer and more reliable supervision. We theoretically prove that under the same dense reward structure, truncated sampling reduces the variance of advantage estimates by up to a factor of T compared to full-trajectory sampling for T-step trajectories, yielding lower-variance, better-targeted policy gradients. Experiments on seven QA benchmarks confirm that SLATE consistently outperforms both sparse-reward and process-reward baselines, with the largest gains on harder multi-hop tasks and smaller models.
>
---
#### [new 027] Terminology Rarity Predicts Catastrophic Failure in LLM Translation of Low-Resource Ancient Languages: Evidence from Ancient Greek
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在低资源古希腊语翻译中的表现，探讨术语罕见性对翻译失败的影响。任务为机器翻译评估，解决翻译质量与术语密度关系问题，通过实验与评估验证假设。**

- **链接: [https://arxiv.org/pdf/2602.24119](https://arxiv.org/pdf/2602.24119)**

> **作者:** James L. Zainaldin; Cameron Pattison; Manuela Marai; Jacob Wu; Mark J. Schiefsky
>
> **备注:** Article + supplementary information
>
> **摘要:** This study presents the first systematic, reference-free human evaluation of large language model (LLM) machine translation (MT) for Ancient Greek (AG) technical prose. We evaluate translations by three commercial LLMs (Claude, Gemini, ChatGPT) of twenty paragraph-length passages from two works by the Greek physician Galen of Pergamum (ca. 129-216 CE): On Mixtures, which has two published English translations, and On the Composition of Drugs according to Kinds, which has never been fully translated into English. We assess translation quality using both standard automated evaluation metrics (BLEU, chrF++, METEOR, ROUGE-L, BERTScore, COMET, BLEURT) and expert human evaluation via a modified Multidimensional Quality Metrics (MQM) framework applied to all 60 translations by a team of domain specialists. On the previously translated expository text, LLMs achieved high translation quality (mean MQM score 95.2/100), with performance approaching expert level. On the untranslated pharmacological text, aggregate quality was lower (79.9/100) but with high variance driven by two passages presenting extreme terminological density; excluding these, scores converged to within 4 points of the translated text. Terminology rarity, operationalized via corpus frequency in the literary Diorisis Ancient Greek Corpus, emerged as a strong predictor of translation failure (r = -.97 for passage-level quality on the untranslated text). Automated metrics showed moderate correlation with human judgment overall on the text with a wide quality spread (Composition), but no metric discriminated among high-quality translations. We discuss implications for the use of LLMs in Classical scholarship and for the design of automated evaluation pipelines for low-resource ancient languages.
>
---
#### [new 028] CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于科学引用验证任务，解决LLM生成中虚假引用问题。构建基准数据集与检测框架，提升引用可信度。**

- **链接: [https://arxiv.org/pdf/2602.23452](https://arxiv.org/pdf/2602.23452)**

> **作者:** Zhengqing Yuan; Kaiwen Shi; Zheyuan Zhang; Lichao Sun; Nitesh V. Chawla; Yanfang Ye
>
> **摘要:** Scientific research relies on accurate citation for attribution and integrity, yet large language models (LLMs) introduce a new risk: fabricated references that appear plausible but correspond to no real publications. Such hallucinated citations have already been observed in submissions and accepted papers at major machine learning venues, exposing vulnerabilities in peer review. Meanwhile, rapidly growing reference lists make manual verification impractical, and existing automated tools remain fragile to noisy and heterogeneous citation formats and lack standardized evaluation. We present the first comprehensive benchmark and detection framework for hallucinated citations in scientific writing. Our multi-agent verification pipeline decomposes citation checking into claim extraction, evidence retrieval, passage matching, reasoning, and calibrated judgment to assess whether a cited source truly supports its claim. We construct a large-scale human-validated dataset across domains and define unified metrics for citation faithfulness and evidence alignment. Experiments with state-of-the-art LLMs reveal substantial citation errors and show that our framework significantly outperforms prior methods in both accuracy and interpretability. This work provides the first scalable infrastructure for auditing citations in the LLM era and practical tools to improve the trustworthiness of scientific references.
>
---
#### [new 029] Do LLMs Benefit From Their Own Words?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在多轮对话中是否受益于自身历史回复。任务是评估模型对自身历史的依赖性，通过对比全上下文与仅用户回合的提示方式，发现部分情况下可省略助手历史以提升效率和质量。**

- **链接: [https://arxiv.org/pdf/2602.24287](https://arxiv.org/pdf/2602.24287)**

> **作者:** Jenny Y. Huang; Leshem Choshen; Ramon Astudillo; Tamara Broderick; Jacob Andreas
>
> **摘要:** Multi-turn interactions with large language models typically retain the assistant's own past responses in the conversation history. In this work, we revisit this design choice by asking whether large language models benefit from conditioning on their own prior responses. Using in-the-wild, multi-turn conversations, we compare standard (full-context) prompting with a user-turn-only prompting approach that omits all previous assistant responses, across three open reasoning models and one state-of-the-art model. To our surprise, we find that removing prior assistant responses does not affect response quality on a large fraction of turns. Omitting assistant-side history can reduce cumulative context lengths by up to 10x. To explain this result, we find that multi-turn conversations consist of a substantial proportion (36.4%) of self-contained prompts, and that many follow-up prompts provide sufficient instruction to be answered using only the current user turn and prior user turns. When analyzing cases where user-turn-only prompting substantially outperforms full context, we identify instances of context pollution, in which models over-condition on their previous responses, introducing errors, hallucinations, or stylistic artifacts that propagate across turns. Motivated by these findings, we design a context-filtering approach that selectively omits assistant-side context. Our findings suggest that selectively omitting assistant history can improve response quality while reducing memory consumption.
>
---
#### [new 030] Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决扩散模型生成效率低的问题。通过提出DiCo方法，实现并行解码，提升推理速度同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2602.23792](https://arxiv.org/pdf/2602.23792)**

> **作者:** Xiangzhong Luo; Yilin An; Zhicheng Yu; Weichen Liu; Xu Yang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Diffusion-based large language models (dLLMs) have shown promising performance across various reasoning tasks, establishing themselves as an alternative to autoregressive large language models (LLMs). Unlike autoregressive LLMs that generate one token per step based on all previous tokens, dLLMs theoretically enable parallel generation of multiple tokens at each decoding step. However, recent dLLMs still favor one-token-per-step generation in practice, as directly decoding multiple masked tokens often leads to degraded generation quality and stability. This reveals a substantial gap between the theoretical parallelism and practical performance of dLLMs. To bridge this gap, we introduce an adaptive parallel decoding approach, namely DiCo, which features a three-phase divide-and-conquer paradigm to unleash the inherent parallelism of dLLMs. During the Divide phase, DiCo first explores the input masked sequence and identifies masked tokens as seed tokens, which are then expanded to construct a set of local clusters. During the Conquer phase, DiCo performs parallel decoding across different local clusters constructed in the Divide phase. The divide-and-conquer process repeatedly alternates between the Divide and Conquer phases until convergence. During the Finalize phase, DiCo decodes the remaining few masked tokens using an effective fine-grained compound decoding scheme to finalize the generation. Extensive experiments demonstrate that DiCo can achieve significant inference speedups while maintaining competitive generation quality.
>
---
#### [new 031] Toward General Semantic Chunking: A Discriminative Framework for Ultra-Long Documents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于长文档主题分割任务，旨在解决超长文本中段落边界检测的问题。提出一种基于Qwen3-0.6B的判别模型，支持高效处理超长文档。**

- **链接: [https://arxiv.org/pdf/2602.23370](https://arxiv.org/pdf/2602.23370)**

> **作者:** Kaifeng Wu; Junyan Wu; Qiang Liu; Jiarui Zhang; Wen Xu
>
> **摘要:** Long-document topic segmentation plays an important role in information retrieval and document understanding, yet existing methods still show clear shortcomings in ultra-long text settings. Traditional discriminative models are constrained by fixed windows and cannot model document-level semantics; generative large language models can output paragraph boundaries, but inference is expensive and long inputs are difficult to support. To address these issues, we propose a discriminative segmentation model based on Qwen3-0.6B. On top of the backbone network, we add a cross-window context fusion layer and a boundary classification head, and combine them with an overlapping sliding-window strategy. Our model supports single-pass inputs of up to 13k tokens and can be extended to ultra-long documents for paragraph boundary detection. To further enhance downstream retrieval efficiency, we derive a vector fusion method with scalar correction, which compresses the representation of ultra-long segments into a single vector without semantic loss. Experiments on the Wikipedia long-document topic segmentation dataset WIKI-727K show that, compared with three generative models based on Qwen2-0.5B released by Jina, our method achieves a better macro-averaged F1 and delivers two orders of magnitude faster inference, substantially improving the practicality and scalability of long-document processing.
>
---
#### [new 032] BRIDGE the Gap: Mitigating Bias Amplification in Automated Scoring of English Language Learners via Inter-group Data Augmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动化评分任务，旨在解决英语学习者（ELL）在评分系统中因数据偏差导致的预测偏差问题。通过生成高质量合成数据提升公平性。**

- **链接: [https://arxiv.org/pdf/2602.23580](https://arxiv.org/pdf/2602.23580)**

> **作者:** Yun Wang; Xuansheng Wu; Jingyuan Huang; Lei Liu; Xiaoming Zhai; Ninghao Liu
>
> **备注:** 15 pages, 1 figure
>
> **摘要:** In the field of educational assessment, automated scoring systems increasingly rely on deep learning and large language models (LLMs). However, these systems face significant risks of bias amplification, where model prediction gaps between student groups become larger than those observed in training data. This issue is especially severe for underrepresented groups such as English Language Learners (ELLs), as models may inherit and further magnify existing disparities in the data. We identify that this issue is closely tied to representation bias: the scarcity of minority (high-scoring ELL) samples makes models trained with empirical risk minimization favor majority (non-ELL) linguistic patterns. Consequently, models tend to under-predict ELL students who even demonstrate comparable domain knowledge but use different linguistic patterns, thereby undermining the fairness of automated scoring outcomes. To mitigate this, we propose BRIDGE, a Bias-Reducing Inter-group Data GEneration framework designed for low-resource assessment settings. Instead of relying on the limited minority samples, BRIDGE synthesizes high-scoring ELL samples by "pasting" construct-relevant (i.e., rubric-aligned knowledge and evidence) content from abundant high-scoring non-ELL samples into authentic ELL linguistic patterns. We further introduce a discriminator model to ensure the quality of synthetic samples. Experiments on California Science Test (CAST) datasets demonstrate that BRIDGE effectively reduces prediction bias for high-scoring ELL students while maintaining overall scoring performance. Notably, our method achieves fairness gains comparable to using additional real human data, offering a cost-effective solution for ensuring equitable scoring in large-scale assessments.
>
---
#### [new 033] Benchmarking BERT-based Models for Sentence-level Topic Classification in Nepali Language
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于句子级主题分类任务，旨在评估不同BERT模型在尼泊尔语中的表现。研究对比了多种预训练模型，以提升尼泊尔语NLP应用效果。**

- **链接: [https://arxiv.org/pdf/2602.23940](https://arxiv.org/pdf/2602.23940)**

> **作者:** Nischal Karki; Bipesh Subedi; Prakash Poudyal; Rupak Raj Ghimire; Bal Krishna Bal
>
> **备注:** 5 pages, 2 figures. Accepted and presented at the Regional International Conference on Natural Language Processing (RegICON 2025), Gauhati University, Guwahati, India, November 27-29, 2025. To appear in the conference proceedings. Accepted papers list available at: this https URL
>
> **摘要:** Transformer-based models such as BERT have significantly advanced Natural Language Processing (NLP) across many languages. However, Nepali, a low-resource language written in Devanagari script, remains relatively underexplored. This study benchmarks multilingual, Indic, Hindi, and Nepali BERT variants to evaluate their effectiveness in Nepali topic classification. Ten pre-trained models, including mBERT, XLM-R, MuRIL, DevBERT, HindiBERT, IndicBERT, and NepBERTa, were fine-tuned and tested on the balanced Nepali dataset containing 25,006 sentences across five conceptual domains and the performance was evaluated using accuracy, weighted precision, recall, F1-score, and AUROC metrics. The results reveal that Indic models, particularly MuRIL-large, achieved the highest F1-score of 90.60%, outperforming multilingual and monolingual models. NepBERTa also performed competitively with an F1-score of 88.26%. Overall, these findings establish a robust baseline for future document-level classification and broader Nepali NLP applications.
>
---
#### [new 034] ArgLLM-App: An Interactive System for Argumentative Reasoning with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ArgLLM-App系统，用于二元任务的论证推理，解决LLM决策可解释性和可争议性问题，支持可视化解释和人机交互。**

- **链接: [https://arxiv.org/pdf/2602.24172](https://arxiv.org/pdf/2602.24172)**

> **作者:** Adam Dejl; Deniz Gorur; Francesca Toni
>
> **备注:** AAMAS 2026 Demonstration Track
>
> **摘要:** Argumentative LLMs (ArgLLMs) are an existing approach leveraging Large Language Models (LLMs) and computational argumentation for decision-making, with the aim of making the resulting decisions faithfully explainable to and contestable by humans. Here we propose a web-based system implementing ArgLLM-empowered agents for binary tasks. ArgLLM-App supports visualisation of the produced explanations and interaction with human users, allowing them to identify and contest any mistakes in the system's reasoning. It is highly modular and enables drawing information from trusted external sources. ArgLLM-App is publicly available at this https URL, with a video demonstration at this https URL.
>
---
#### [new 035] Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型安全任务，解决LLM jailbreak基准测试滞后问题。提出JBF系统，实现攻击代码自动化转换与标准化评估。**

- **链接: [https://arxiv.org/pdf/2602.24009](https://arxiv.org/pdf/2602.24009)**

> **作者:** Zhicheng Fang; Jingjie Zheng; Chenxu Fu; Wei Xu
>
> **摘要:** Jailbreak techniques for large language models (LLMs) evolve faster than benchmarks, making robustness estimates stale and difficult to compare across papers due to drift in datasets, harnesses, and judging protocols. We introduce JAILBREAK FOUNDRY (JBF), a system that addresses this gap via a multi-agent workflow to translate jailbreak papers into executable modules for immediate evaluation within a unified harness. JBF features three core components: (i) JBF-LIB for shared contracts and reusable utilities; (ii) JBF-FORGE for the multi-agent paper-to-module translation; and (iii) JBF-EVAL for standardizing evaluations. Across 30 reproduced attacks, JBF achieves high fidelity with a mean (reproduced-reported) attack success rate (ASR) deviation of +0.26 percentage points. By leveraging shared infrastructure, JBF reduces attack-specific implementation code by nearly half relative to original repositories and achieves an 82.5% mean reused-code ratio. This system enables a standardized AdvBench evaluation of all 30 attacks across 10 victim models using a consistent GPT-4o judge. By automating both attack integration and standardized evaluation, JBF offers a scalable solution for creating living benchmarks that keep pace with the rapidly shifting security landscape.
>
---
#### [new 036] Serendipity with Generative AI: Repurposing knowledge components during polycrisis with a Viable Systems Model approach
- **分类: cs.HC; cs.CL; cs.IR**

- **简介: 该论文探讨生成式AI在多危机中挖掘和重组知识组件的潜力，属于知识管理任务。解决组织在不确定性中忽视隐含知识的问题，通过构建符合VSM的知识库提升系统性创新。**

- **链接: [https://arxiv.org/pdf/2602.23365](https://arxiv.org/pdf/2602.23365)**

> **作者:** Gordon Fletcher; Saomai Vu Khan
>
> **摘要:** Organisations face polycrisis uncertainty yet overlook embedded knowledge. We show how generative AI can operate as a serendipity engine and knowledge transducer to discover, classify and mobilise reusable components (models, frameworks, patterns) from existing documents. Using 206 papers, our pipeline extracted 711 components (approx 3.4 per paper) and organised them into a repository aligned to Beer's Viable System Model (VSM). We contribute i) conceptually, a theory of planned serendipity in which GenAI lowers transduction costs between VSM subsystems, ii) empirically, a component repository and temporal/subject patterns, iii) managerially, a vignette and process blueprint for organisational adoption and iv) socially, pathways linking repurposing to environmental and social benefits. We propose testable links between repository creation, discovery-to-deployment time, and reuse rates, and discuss implications for shifting innovation portfolios from breakthrough bias toward systematic repurposing.
>
---
#### [new 037] Higress-RAG: A Holistic Optimization Framework for Enterprise Retrieval-Augmented Generation via Dual Hybrid Retrieval, Adaptive Routing, and CRAG
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于知识管理任务，旨在解决RAG系统在企业应用中的检索精度低、幻觉多和延迟高的问题。提出Higress-RAG框架，通过优化全流程提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23374](https://arxiv.org/pdf/2602.23374)**

> **作者:** Weixi Lin
>
> **备注:** 7 pages,5 figures, our submissions are not yet published
>
> **摘要:** The integration of Large Language Models (LLMs) into enterprise knowledge management systems has been catalyzed by the Retrieval-Augmented Generation (RAG) paradigm, which augments parametric memory with non-parametric external data. However, the transition from proof-of-concept to production-grade RAG systems is hindered by three persistent challenges: low retrieval precision for complex queries, high rates of hallucination in the generation phase, and unacceptable latency for real-time applications. This paper presents a comprehensive analysis of the Higress RAG MCP Server, a novel, enterprise-centric architecture designed to resolve these bottlenecks through a "Full-Link Optimization" strategy. Built upon the Model Context Protocol (MCP), the system introduces a layered architecture that orchestrates a sophisticated pipeline of Adaptive Routing, Semantic Caching, Hybrid Retrieval, and Corrective RAG (CRAG). We detail the technical implementation of key innovations, including the Higress-Native Splitter for structure-aware data ingestion, the application of Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval signals, and a 50ms-latency Semantic Caching mechanism with dynamic thresholding. Experimental evaluations on domain-specific Higress technical documentation and blogs verify the system's architectural robustness. The results demonstrate that by optimizing the entire retrieval lifecycle - from pre-retrieval query rewriting to post-retrieval corrective evaluation - the Higress RAG system offers a scalable, hallucination-resistant solution for enterprise AI deployment.
>
---
#### [new 038] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理优化任务，解决草案模型接受率低的问题。通过提出LK损失函数，直接优化接受率，提升解码速度。**

- **链接: [https://arxiv.org/pdf/2602.23881](https://arxiv.org/pdf/2602.23881)**

> **作者:** Alexander Samarin; Sergei Krutikov; Anton Shevtsov; Sergei Skvortsov; Filipp Fisin; Alexander Golubev
>
> **摘要:** Speculative decoding accelerates autoregressive large language model (LLM) inference by using a lightweight draft model to propose candidate tokens that are then verified in parallel by the target model. The speedup is significantly determined by the acceptance rate, yet standard training minimizes Kullback-Leibler (KL) divergence as a proxy objective. While KL divergence and acceptance rate share the same global optimum, small draft models, having limited capacity, typically converge to suboptimal solutions where minimizing KL does not guarantee maximizing acceptance rate. To address this issue, we propose LK losses, special training objectives that directly target acceptance rate. Comprehensive experiments across four draft architectures and six target models, ranging from 8B to 685B parameters, demonstrate consistent improvements in acceptance metrics across all configurations compared to the standard KL-based training. We evaluate our approach on general, coding and math domains and report gains of up to 8-10% in average acceptance length. LK losses are easy to implement, introduce no computational overhead and can be directly integrated into any existing speculator training framework, making them a compelling alternative to the existing draft training objectives.
>
---
#### [new 039] SongSong: A Time Phonograph for Chinese SongCi Music from Thousand of Years Away
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出SongSong模型，用于生成古体词曲音乐，解决古代音乐生成困难的问题。工作包括模型设计与数据集构建。**

- **链接: [https://arxiv.org/pdf/2602.24071](https://arxiv.org/pdf/2602.24071)**

> **作者:** Jiajia Li; Jiliang Hu; Ziyi Pan; Chong Chen; Zuchao Li; Ping Wang; Lefei Zhang
>
> **备注:** 9 pages, 6 figures, accepted by AAAI 2025
>
> **摘要:** Recently, there have been significant advancements in music generation. However, existing models primarily focus on creating modern pop songs, making it challenging to produce ancient music with distinct rhythms and styles, such as ancient Chinese SongCi. In this paper, we introduce SongSong, the first music generation model capable of restoring Chinese SongCi to our knowledge. Our model first predicts the melody from the input SongCi, then separately generates the singing voice and accompaniment based on that melody, and finally combines all elements to create the final piece of music. Additionally, to address the lack of ancient music datasets, we create OpenSongSong, a comprehensive dataset of ancient Chinese SongCi music, featuring 29.9 hours of compositions by various renowned SongCi music masters. To assess SongSong's proficiency in performing SongCi, we randomly select 85 SongCi sentences that were not part of the training set for evaluation against SongSong and music generation platforms such as Suno and SkyMusic. The subjective and objective outcomes indicate that our proposed model achieves leading performance in generating high-quality SongCi music.
>
---
#### [new 040] AgenticOCR: Parsing Only What You Need for Efficient Retrieval-Augmented Generation
- **分类: cs.CV; cs.CL**

- **简介: 论文提出AgenticOCR，解决视觉文档RAG中的冗余信息问题。通过动态解析，仅提取所需内容，提升效率与准确性。属于视觉文档理解任务。**

- **链接: [https://arxiv.org/pdf/2602.24134](https://arxiv.org/pdf/2602.24134)**

> **作者:** Zhengren Wang; Dongsheng Ma; Huaping Zhong; Jiayu Li; Wentao Zhang; Bin Wang; Conghui He
>
> **摘要:** The expansion of retrieval-augmented generation (RAG) into multimodal domains has intensified the challenge for processing complex visual documents, such as financial reports. While page-level chunking and retrieval is a natural starting point, it creates a critical bottleneck: delivering entire pages to the generator introduces excessive extraneous context. This not only overloads the generator's attention mechanism but also dilutes the most salient evidence. Moreover, compressing these information-rich pages into a limited visual token budget further increases the risk of hallucinations. To address this, we introduce AgenticOCR, a dynamic parsing paradigm that transforms optical character recognition (OCR) from a static, full-text process into a query-driven, on-demand extraction system. By autonomously analyzing document layout in a "thinking with images" manner, AgenticOCR identifies and selectively recognizes regions of interest. This approach performs on-demand decompression of visual tokens precisely where needed, effectively decoupling retrieval granularity from rigid page-level chunking. AgenticOCR has the potential to serve as the "third building block" of the visual document RAG stack, operating alongside and enhancing standard Embedding and Reranking modules. Experimental results demonstrate that AgenticOCR improves both the efficiency and accuracy of visual RAG systems, achieving expert-level performance in long document understanding. Code and models are available at this https URL.
>
---
#### [new 041] Hello-Chat: Towards Realistic Social Audio Interactions
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出Hello-Chat，解决音频交互中缺乏自然与情感的问题。属于音频语言模型任务，通过真实对话数据和跨模态训练，提升语音的自然度与情感共鸣。**

- **链接: [https://arxiv.org/pdf/2602.23387](https://arxiv.org/pdf/2602.23387)**

> **作者:** Yueran Hou; Peilei Jia; Zihan Sun; Qihang Lu; Wenbing Yang; Yingming Gao; Ya Li; Jun Gao
>
> **摘要:** Recent advancements in Large Audio Language Models (LALMs) have demonstrated exceptional performance in speech recognition and translation. However, existing models often suffer from a disconnect between perception and expression, resulting in a robotic "read-speech" style that lacks the spontaneity and emotional resonance of real human interaction. In this report, we introduce Hello-Chat, an end-to-end audio language model designed for realistic social scenarios. By leveraging a massive dataset of real-life conversations and employing a modality-interleaved training strategy, Hello-Chat achieves a breakthrough in anthropomorphic generation. Experimental results show that our model not only reaches state-of-the-art (SOTA) performance on specific audio understanding tasks but also significantly outperforms existing baselines in prosodic naturalness and emotional alignment, paving the way for the next generation of empathetic AI agents.
>
---
#### [new 042] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大模型优化任务，旨在解决视觉token处理效率低的问题。通过提出HiDrop框架，实现高效视觉token压缩与加速训练。**

- **链接: [https://arxiv.org/pdf/2602.23699](https://arxiv.org/pdf/2602.23699)**

> **作者:** Hao Wu; Yingqi Fan; Jinyang Dai; Junlong Tong; Yunpu Ma; Xiaoyu Shen
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** The quadratic computational cost of processing vision tokens in Multimodal Large Language Models (MLLMs) hinders their widespread adoption. While progressive vision token pruning offers a promising solution, current methods misinterpret shallow layer functions and use rigid schedules, which fail to unlock the full efficiency potential. To address these issues, we propose HiDrop, a framework that aligns token pruning with the true hierarchical function of MLLM layers. HiDrop features two key innovations: (1) Late Injection, which bypasses passive shallow layers to introduce visual tokens exactly where active fusion begins; and (2) Concave Pyramid Pruning with an Early Exit mechanism to dynamically adjust pruning rates across middle and deep layers. This process is optimized via an inter-layer similarity measure and a differentiable top-k operator. To ensure practical efficiency, HiDrop further incorporates persistent positional encoding, FlashAttention-compatible token selection, and parallel decoupling of vision computation to eliminate hidden overhead associated with dynamic token reduction. Extensive experiments show that HiDrop compresses about 90% visual tokens while matching the original performance and accelerating training by 1.72 times. Our work not only sets a new state-of-the-art for efficient MLLM training and inference but also provides valuable insights into the hierarchical nature of multimodal fusion. The code is released at this https URL.
>
---
#### [new 043] Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于深度学习优化任务，旨在解决Adam等优化器内存消耗大的问题。通过低秩近似方法设计新优化器LoRA-Pre，提升训练效率与内存利用率。**

- **链接: [https://arxiv.org/pdf/2602.24283](https://arxiv.org/pdf/2602.24283)**

> **作者:** Zhengbo Wang; Jian Liang; Ran He; Zilei Wang; Tieniu Tan
>
> **备注:** Camera-ready version. Accepted as Oral at ICLR 2026
>
> **摘要:** Modern optimizers like Adam and Muon are central to training large language models, but their reliance on first- and second-order momenta introduces significant memory overhead, which constrains scalability and computational efficiency. In this work, we reframe the exponential moving average (EMA) used in these momenta as the training of a linear regressor via online gradient flow. Building on this equivalence, we introduce LoRA-Pre, a novel low-rank optimizer designed for efficient pre-training. Specifically, LoRA-Pre reduces the optimizer's memory footprint by decomposing the full momentum matrix into a compact low-rank subspace within the online linear learner, thereby maintaining optimization performance while improving memory efficiency. We empirically validate LoRA-Pre's efficacy by pre-training models from the Llama architecture family, scaling from 60M to 1B parameters. LoRA-Pre achieves the highest performance across all model sizes. Notably, LoRA-Pre demonstrates remarkable rank efficiency, achieving comparable or superior results using only 1/8 the rank of baseline methods. Beyond pre-training, we evaluate LoRA-Pre's effectiveness in fine-tuning scenarios. With the same rank, LoRA-Pre consistently outperforms all efficient fine-tuning baselines. Specifically, compared to standard LoRA, LoRA-Pre achieves substantial improvements of 3.14 points on Llama-3.1-8B and 6.17 points on Llama-2-7B, validating our approach's effectiveness across both pre-training and fine-tuning paradigms. Our code is publicly available at this https URL.
>
---
#### [new 044] NAU-QMUL: Utilizing BERT and CLIP for Multi-modal AI-Generated Image Detection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于AI生成图像检测任务，旨在识别图像来源模型。通过融合BERT与CLIP的多模态特征，结合伪标签数据增强，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.23863](https://arxiv.org/pdf/2602.23863)**

> **作者:** Xiaoyu Guo; Arkaitz Zubiaga
>
> **摘要:** With the aim of detecting AI-generated images and identifying the specific models responsible for their generation, we propose a multi-modal multi-task model. The model leverages pre-trained BERT and CLIP Vision encoders for text and image feature extraction, respectively, and employs cross-modal feature fusion with a tailored multi-task loss function. Additionally, a pseudo-labeling-based data augmentation strategy was utilized to expand the training dataset with high-confidence samples. The model achieved fifth place in both Tasks A and B of the `CT2: AI-Generated Image Detection' competition, with F1 scores of 83.16\% and 48.88\%, respectively. These findings highlight the effectiveness of the proposed architecture and its potential for advancing AI-generated content detection in real-world scenarios. The source code for our method is published on this https URL.
>
---
#### [new 045] Recycling Failures: Salvaging Exploration in RLVR via Fine-Grained Off-Policy Guidance
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR中因粗粒度反馈导致的探索空间狭窄问题。通过引入细粒度的离策略修正方法，提升轨迹多样性与模型性能。**

- **链接: [https://arxiv.org/pdf/2602.24110](https://arxiv.org/pdf/2602.24110)**

> **作者:** Yanwei Ren; Haotian Zhang; Likang Xiao; Xikai Zhang; Jiaxing Huang; Jiayan Qiu; Baosheng Yu; Quan Chen; Liu Liu
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing the complex reasoning capabilities of Large Reasoning Models. However, standard outcome-based supervision suffers from a critical limitation that penalizes trajectories that are largely correct but fail due to several missteps as heavily as completely erroneous ones. This coarse feedback signal causes the model to discard valuable largely correct rollouts, leading to a degradation in rollout diversity that prematurely narrows the exploration space. Process Reward Models have demonstrated efficacy in providing reliable step-wise verification for test-time scaling, naively integrating these signals into RLVR as dense rewards proves this http URL methods attempt to introduce off-policy guided whole-trajectory replacement that often outside the policy model's distribution, but still fail to utilize the largely correct rollouts generated by the model itself and thus do not effectively mitigate the narrowing of the exploration space. To address these issues, we propose SCOPE (Step-wise Correction for On-Policy Exploration), a novel framework that utilizes Process Reward Models to pinpoint the first erroneous step in suboptimal rollouts and applies fine-grained, step-wise off-policy rectification. By applying precise refinement on partially correct rollout, our method effectively salvages partially correct trajectories and increases diversity score by 13.5%, thereby sustaining a broad exploration space. Extensive experiments demonstrate that our approach establishes new state-of-the-art results, achieving an average accuracy of 46.6% on math reasoning and exhibiting robust generalization with 53.4% accuracy on out-of-distribution reasoning tasks.
>
---
#### [new 046] EvoX: Meta-Evolution for Automated Discovery
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文提出EvoX，一种自适应进化方法，用于优化搜索策略。解决传统方法固定策略适应性差的问题，通过联合进化解与策略提升优化效果。**

- **链接: [https://arxiv.org/pdf/2602.23413](https://arxiv.org/pdf/2602.23413)**

> **作者:** Shu Liu; Shubham Agarwal; Monishwaran Maheswaran; Mert Cemri; Zhifei Li; Qiuyang Mang; Ashwin Naren; Ethan Boneh; Audrey Cheng; Melissa Z. Pan; Alexander Du; Kurt Keutzer; Alexandros G. Dimakis; Koushik Sen; Matei Zaharia; Ion Stoica
>
> **摘要:** Recent work such as AlphaEvolve has shown that combining LLM-driven optimization with evolutionary search can effectively improve programs, prompts, and algorithms across domains. In this paradigm, previously evaluated solutions are reused to guide the model toward new candidate solutions. Crucially, the effectiveness of this evolution process depends on the search strategy: how prior solutions are selected and varied to generate new candidates. However, most existing methods rely on fixed search strategies with predefined knobs (e.g., explore-exploit ratios) that remain static throughout execution. While effective in some settings, these approaches often fail to adapt across tasks, or even within the same task as the search space changes over time. We introduce EvoX, an adaptive evolution method that optimizes its own evolution process. EvoX jointly evolves candidate solutions and the search strategies used to generate them, continuously updating how prior solutions are selected and varied based on progress. This enables the system to dynamically shift between different search strategies during the optimization process. Across nearly 200 real-world optimization tasks, EvoX outperforms existing AI-driven evolutionary methods including AlphaEvolve, OpenEvolve, GEPA, and ShinkaEvolve on the majority of tasks.
>
---
#### [new 047] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉跟踪任务，旨在解决Transformer跟踪器计算开销大的问题。提出UTPTrack框架，统一剪枝三个组件，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2602.23734](https://arxiv.org/pdf/2602.23734)**

> **作者:** Hao Wu; Xudong Wang; Jialiang Zhang; Junlong Tong; Xinghao Chen; Junyan Lin; Yunpu Ma; Xiaoyu Shen
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** One-stream Transformer-based trackers achieve advanced performance in visual object tracking but suffer from significant computational overhead that hinders real-time deployment. While token pruning offers a path to efficiency, existing methods are fragmented. They typically prune the search region, dynamic template, and static template in isolation, overlooking critical inter-component dependencies, which yields suboptimal pruning and degraded accuracy. To address this, we introduce UTPTrack, a simple and Unified Token Pruning framework that, for the first time, jointly compresses all three components. UTPTrack employs an attention-guided, token type-aware strategy to holistically model redundancy, a design that seamlessly supports unified tracking across multimodal and language-guided tasks within a single model. Extensive evaluations on 10 benchmarks demonstrate that UTPTrack achieves a new state-of-the-art in the accuracy-efficiency trade-off for pruning-based trackers, pruning 65.4% of vision tokens in RGB-based tracking and 67.5% in unified tracking while preserving 99.7% and 100.5% of baseline performance, respectively. This strong performance across both RGB and multimodal scenarios underlines its potential as a robust foundation for future research in efficient visual tracking. Code will be released at this https URL.
>
---
#### [new 048] Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多跳问答任务，解决GraphRAG依赖昂贵GPU和LLM的问题。提出SPRIG方法，使用CPU进行线性图检索，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.23372](https://arxiv.org/pdf/2602.23372)**

> **作者:** Qizhi Wang
>
> **备注:** 13 pages, 14 figures, 26 tables
>
> **摘要:** GraphRAG systems improve multi-hop retrieval by modeling structure, but many approaches rely on expensive LLM-based graph construction and GPU-heavy inference. We present SPRIG (Seeded Propagation for Retrieval In Graphs), a CPU-only, linear-time, token-free GraphRAG pipeline that replaces LLM graph building with lightweight NER-driven co-occurrence graphs and uses Personalized PageRank (PPR) for 28% with negligible Recall@10 changes. The results characterize when CPU-friendly graph retrieval helps multi-hop recall and when strong lexical hybrids (RRF) are sufficient, outlining a realistic path to democratizing GraphRAG without token costs or GPU requirements.
>
---
#### [new 049] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DARE-bench，用于评估大语言模型在数据科学任务中的建模能力和指令遵循能力。解决现有基准缺乏标准化评估和高质量数据的问题，通过大量可验证任务提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.24288](https://arxiv.org/pdf/2602.24288)**

> **作者:** Fan Shu; Yite Wang; Ruofan Wu; Boyi Liu; Zhewei Yao; Yuxiong He; Feng Yan
>
> **备注:** Published as a conference paper at ICLR 2026. 10 pages plus appendix
>
> **摘要:** The fast-growing demands in using Large Language Models (LLMs) to tackle complex multi-step data science tasks create an emergent need for accurate benchmarking. There are two major gaps in existing benchmarks: (i) the lack of standardized, process-aware evaluation that captures instruction adherence and process fidelity, and (ii) the scarcity of accurately labeled training data. To bridge these gaps, we introduce DARE-bench, a benchmark designed for machine learning modeling and data science instruction following. Unlike many existing benchmarks that rely on human- or model-based judges, all tasks in DARE-bench have verifiable ground truth, ensuring objective and reproducible evaluation. To cover a broad range of tasks and support agentic tools, DARE-bench consists of 6,300 Kaggle-derived tasks and provides both large-scale training data and evaluation sets. Extensive evaluations show that even highly capable models such as gpt-o4-mini struggle to achieve good performance, especially in machine learning modeling tasks. Using DARE-bench training tasks for fine-tuning can substantially improve model performance. For example, supervised fine-tuning boosts Qwen3-32B's accuracy by 1.83x and reinforcement learning boosts Qwen3-4B's accuracy by more than 8x. These significant improvements verify the importance of DARE-bench both as an accurate evaluation benchmark and critical training data.
>
---
#### [new 050] Domain-Partitioned Hybrid RAG for Legal Reasoning: Toward Modular and Explainable Legal AI for India
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于法律信息检索任务，旨在解决印度法律文本中多跳推理和跨领域依赖问题。提出一种分域混合RAG与知识图谱架构，提升法律AI的可解释性和推理能力。**

- **链接: [https://arxiv.org/pdf/2602.23371](https://arxiv.org/pdf/2602.23371)**

> **作者:** Rakshita Goel; S Pranav Kumar; Anmol Agrawal; Divyan Poddar; Pratik Narang; Dhruv Kumar
>
> **摘要:** Legal research in India involves navigating long and heterogeneous documents spanning statutes, constitutional provisions, penal codes, and judicial precedents, where purely keyword-based or embedding-only retrieval systems often fail to support structured legal reasoning. Recent retrieval augmented generation (RAG) approaches improve grounding but struggle with multi-hop reasoning, citation chaining, and cross-domain dependencies inherent to legal texts. We propose a domain partitioned hybrid RAG and Knowledge Graph architecture designed specifically for Indian legal research. The system integrates three specialized RAG pipelines covering Supreme Court case law, statutory and constitutional texts, and the Indian Penal Code, each optimized for domain specific retrieval. To enable relational reasoning beyond semantic similarity, we construct a Neo4j based Legal Knowledge Graph capturing structured relationships among cases, statutes, IPC sections, judges, and citations. An LLM driven agentic orchestrator dynamically routes queries across retrieval modules and the knowledge graph, fusing evidence into grounded and citation aware responses. We evaluate the system using a 40 question synthetic legal question answer benchmark curated from authoritative Indian legal sources and assessed via an LLM as a Judge framework. Results show that the hybrid architecture achieves a 70 percent pass rate, substantially outperforming a RAG only baseline at 37.5 percent, with marked improvements in completeness and legal reasoning quality. These findings demonstrate that combining domain partitioned retrieval with structured relational knowledge provides a scalable and interpretable foundation for advanced legal AI systems in the Indian judicial context.
>
---
#### [new 051] SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-rebench V2，解决SWE任务数据不足问题，构建大规模多语言任务集，支持RL训练。**

- **链接: [https://arxiv.org/pdf/2602.23866](https://arxiv.org/pdf/2602.23866)**

> **作者:** Ibragim Badertdinov; Maksim Nekrashevich; Anton Shevtsov; Alexander Golubev
>
> **摘要:** Software engineering agents (SWE) are improving rapidly, with recent gains largely driven by reinforcement learning (RL). However, RL training is constrained by the scarcity of large-scale task collections with reproducible execution environments and reliable test suites. Although a growing number of benchmarks have emerged, datasets suitable for training remain limited in scale and diversity or often target a limited set of high-resource language ecosystems. We introduce SWE-rebench V2, a language-agnostic automated pipeline for harvesting executable real-world SWE tasks and constructing RL training environments at scale. The pipeline synthesizes repository-specific installation and test procedures via an interactive setup agent, and filters unsound instances using an ensemble of LLM judges, validated against human-verified SWE-bench annotations. Using this pipeline, we construct a dataset of 32,000+ tasks spanning 20 languages and 3,600+ repositories, with pre-built images for reproducible execution. To further scale training data, we additionally release 120,000+ tasks with installation instructions, fail-to-pass tests and rich metadata, where the problem statement is generated based on the original pull request description. We validate the collected instances through a diagnostic study that covers a subset of tasks in five programming languages across seven popular models, and provide instance-level metadata that flags common confounders such as overly restrictive tests and underspecified descriptions. We release the datasets, the collection and execution code, and associated artifacts to enable large-scale training of SWE agents across diverse languages and repositories.
>
---
#### [new 052] RewardUQ: A Unified Framework for Uncertainty-Aware Reward Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习中的奖励模型任务，旨在解决奖励模型中不确定性建模的问题。提出RewardUQ框架，系统评估不确定性量化方法，提升模型性能与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.24040](https://arxiv.org/pdf/2602.24040)**

> **作者:** Daniel Yang; Samuel Stante; Florian Redhardt; Lena Libon; Parnian Kassraie; Ido Hakimi; Barna Pásztor; Andreas Krause
>
> **摘要:** Reward models are central to aligning large language models (LLMs) with human preferences. Yet most approaches rely on pointwise reward estimates that overlook the epistemic uncertainty in reward models arising from limited human feedback. Recent work suggests that quantifying this uncertainty can reduce the costs of human annotation via uncertainty-guided active learning and mitigate reward overoptimization in LLM post-training. However, uncertainty-aware reward models have so far been adopted without thorough comparison, leaving them poorly understood. This work introduces a unified framework, RewardUQ, to systematically evaluate uncertainty quantification for reward models. We compare common methods along standard metrics measuring accuracy and calibration, and we propose a new ranking strategy incorporating both dimensions for a simplified comparison. Our experimental results suggest that model size and initialization have the most meaningful impact on performance, and most prior work could have benefited from alternative design choices. To foster the development and evaluation of new methods and aid the deployment in downstream applications, we release our open-source framework as a Python package. Our code is available at this https URL.
>
---
#### [new 053] Reason to Contrast: A Cascaded Multimodal Retrieval Framework
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多模态检索任务，旨在提升检索效果。通过引入推理步骤进行重排序，增强查询与候选项的交互，解决传统方法依赖嵌入维度的问题。**

- **链接: [https://arxiv.org/pdf/2602.23369](https://arxiv.org/pdf/2602.23369)**

> **作者:** Xuanming Cui; Hong-You Chen; Hao Yu; Hao Yuan; Zihao Wang; Shlok Kumar Mishra; Hanchao Yu; Yonghuan Yang; Jun Xiao; Ser-Nam Lim; Jianpeng Cheng; Qi Guo; Xiangjun Fan
>
> **摘要:** Traditional multimodal retrieval systems rely primarily on bi-encoder architectures, where performance is closely tied to embedding dimensionality. Recent work, Think-Then-Embed (TTE), shows that incorporating multimodal reasoning to elicit additional informative tokens before embedding can further improve retrieval. In this paper, we extend this paradigm with TTE-v2, a hybrid multimodal retrieval framework that introduces reasoning-driven performance scaling based on additional input token budget rather than model or embedding size. Our approach augments the initial multimodal retrieval with additional reasoning steps for reranking, enabling more expressive query-candidate interactions at test time. The reranking stage further provides fine-grained supervision for hard negative mining and false negative filtering, creating a feedback loop that effectively strengthens the upstream retriever. This cascaded design delivers substantial test-time improvements based on intermediate reasoning token scaling. Experiments on the MMEB-V2 benchmark demonstrate that TTE-v2-7B achieves a new state-of-the-art accuracy of 75.7%, and that TTE-v2-2B matches or surpasses leading 7B models trained with significantly larger external data. Our results highlight the promise of token-wise scaling as an alternative scaling paradigm for multimodal retrieval.
>
---
#### [new 054] Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于多模态大语言模型的不确定性量化任务，旨在解决模型输出可能错误的问题。提出UMPIRE框架，无需外部工具即可高效评估不确定性。**

- **链接: [https://arxiv.org/pdf/2602.24195](https://arxiv.org/pdf/2602.24195)**

> **作者:** Gregory Kang Ruey Lau; Hieu Dao; Nicole Kan Hui Lin; Bryan Kian Hsiang Low
>
> **备注:** Earlier versions presented at ICLR 2025 QUESTION workshop and ICML 2025 R2-FM workshop
>
> **摘要:** Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features. UMPIRE computes the incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, effectively capturing both the global semantic diversity of samples and the local incoherence of responses based on internal model confidence. We propose uncertainty desiderata for MLLMs and provide theoretical analysis motivating UMPIRE's design. Extensive experiments show that UMPIRE consistently outperforms baseline metrics in error detection and uncertainty calibration across image, audio, and video-text benchmarks, including adversarial and out-of-distribution settings. We also demonstrate UMPIRE's generalization to non-text output tasks, including image and audio generation.
>
---
#### [new 055] Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在解决Referring Expression Comprehension（REC）中的视觉定位问题。针对现有基准测试的不足，作者提出Ref-Adv，通过设计更复杂的表达和干扰项，提升对视觉推理的要求，并评估多模态大模型的表现。**

- **链接: [https://arxiv.org/pdf/2602.23898](https://arxiv.org/pdf/2602.23898)**

> **作者:** Qihua Dong; Kuo Yang; Lin Ju; Handong Zhao; Yitian Zhang; Yizhou Wang; Huimin Zeng; Jianglin Lu; Yun Fu
>
> **备注:** ICLR 2026
>
> **摘要:** Referring Expression Comprehension (REC) links language to region level visual perception. Standard benchmarks (RefCOCO, RefCOCO+, RefCOCOg) have progressed rapidly with multimodal LLMs but remain weak tests of visual reasoning and grounding: (i) many expressions are very short, leaving little reasoning demand; (ii) images often contain few distractors, making the target easy to find; and (iii) redundant descriptors enable shortcut solutions that bypass genuine text understanding and visual reasoning. We introduce Ref-Adv, a modern REC benchmark that suppresses shortcuts by pairing linguistically nontrivial expressions with only the information necessary to uniquely identify the target. The dataset contains referring expressions on real images, curated with hard distractors and annotated with reasoning facets including negation. We conduct comprehensive ablations (word order perturbations and descriptor deletion sufficiency) to show that solving Ref-Adv requires reasoning beyond simple cues, and we evaluate a broad suite of contemporary multimodal LLMs on Ref-Adv. Despite strong results on RefCOCO, RefCOCO+, and RefCOCOg, models drop markedly on Ref-Adv, revealing reliance on shortcuts and gaps in visual reasoning and grounding. We provide an in depth failure analysis and aim for Ref-Adv to guide future work on visual reasoning and grounding in MLLMs.
>
---
#### [new 056] Data Driven Optimization of GPU efficiency for Distributed LLM Adapter Serving
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于分布式LLM适配器优化任务，旨在提升GPU资源效率。通过数据驱动方法，解决多适配器并发服务中的缓存与调度问题，减少GPU需求并避免资源不足。**

- **链接: [https://arxiv.org/pdf/2602.24044](https://arxiv.org/pdf/2602.24044)**

> **作者:** Ferran Agullo; Joan Oliveras; Chen Wang; Alberto Gutierrez-Torre; Olivier Tardieu; Alaa Youssef; Jordi Torres; Josep Ll. Berral
>
> **备注:** journal extension of the workshop paper titled as "A data-driven ml approach for maximizing performance in llm-adapter serving"
>
> **摘要:** Large Language Model (LLM) adapters enable low-cost model specialization, but introduce complex caching and scheduling challenges in distributed serving systems where hundreds of adapters must be hosted concurrently. While prior work has largely focused on latency minimization, resource efficiency through throughput maximization remains underexplored. This paper presents a data-driven pipeline that, for a given workload, computes an adapter placement that serves the workload with the minimum number of GPUs while avoiding request starvation and GPU memory errors. To that end, the approach identifies the maximum feasible throughput attainable on each GPU by leveraging accurate performance predictions learned from real serving behavior. The proposed pipeline integrates three components: (i) a Digital Twin (DT) tailored to LLM-adapter serving, (ii) a distilled machine learning (ML) model trained on DT-generated data, and (iii) a greedy placement algorithm that exploits ML-based performance estimates to maximize GPU efficiency. The DT emulates real system dynamics with high fidelity, achieving below 5% throughput estimation error while executing up to 90 times faster than full LLM benchmarking across both predictable and unpredictable workloads. The learned ML models further accelerate performance estimation with marginal accuracy degradation, enabling scalable optimization. Experimental results demonstrate that the pipeline substantially improves GPU efficiency by reducing the number of GPUs required to sustain target workloads. Beyond GPU efficiency, the pipeline can be adapted to alternative objectives, such as latency minimization, highlighting its versatility for future large-scale LLM serving infrastructures.
>
---
#### [new 057] A Novel Hierarchical Multi-Agent System for Payments Using LLMs
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出一种基于LLM的分层多智能体系统（HMASP），解决支付流程自动化问题，通过模块化架构实现端到端支付任务。**

- **链接: [https://arxiv.org/pdf/2602.24068](https://arxiv.org/pdf/2602.24068)**

> **作者:** Joon Kiat Chua; Donghao Huang; Zhaoxia Wang
>
> **备注:** 12 pages, 1 figure, 3 tables. Accepted at PAKDD 2026
>
> **摘要:** Large language model (LLM) agents, such as OpenAI's Operator and Claude's Computer Use, can automate workflows but unable to handle payment tasks. Existing agentic solutions have gained significant attention; however, even the latest approaches face challenges in implementing end-to-end agentic payment workflows. To address this gap, this research proposes the Hierarchical Multi-Agent System for Payments (HMASP), which provides an end-to-end agentic method for completing payment workflows. The proposed HMASP leverages either open-weight or proprietary LLMs and employs a modular architecture consisting of the Conversational Payment Agent (CPA - first agent level), Supervisor agents (second agent level), Routing agents (third agent level), and the Process summary agent (fourth agent level). The CPA serves as the central entry point, handling all external requests and coordinating subsequent tasks across hierarchical levels. HMASP incorporates architectural patterns that enable modular task execution across agents and levels for payment operations, including shared state variables, decoupled message states, and structured handoff protocols that facilitate coordination across agents and workflows. Experimental results demonstrate the feasibility of the proposed HMASP. To our knowledge, HMASP is the first LLM-based multi-agent system to implement end-to-end agentic payment workflows. This work lays a foundation for extending agentic capabilities into the payment domain.
>
---
#### [new 058] An Agentic LLM Framework for Adverse Media Screening in AML Compliance
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于AML合规任务，旨在解决传统关键词搜索导致的高误报问题。通过构建基于LLM的代理系统，实现自动化不良媒体筛查与风险评分。**

- **链接: [https://arxiv.org/pdf/2602.23373](https://arxiv.org/pdf/2602.23373)**

> **作者:** Pavel Chernakov; Sasan Jafarnejad; Raphaël Frank
>
> **摘要:** Adverse media screening is a critical component of anti-money laundering (AML) and know-your-customer (KYC) compliance processes in financial institutions. Traditional approaches rely on keyword-based searches that generate high false-positive rates or require extensive manual review. We present an agentic system that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to automate adverse media screening. Our system implements a multi-step approach where an LLM agent searches the web, retrieves and processes relevant documents, and computes an Adverse Media Index (AMI) score for each subject. We evaluate our approach using multiple LLM backends on a dataset comprising Politically Exposed Persons (PEPs), persons from regulatory watchlists, and sanctioned persons from OpenSanctions and clean names from academic sources, demonstrating the system's ability to distinguish between high-risk and low-risk individuals.
>
---
#### [new 059] Toward Guarantees for Clinical Reasoning in Vision Language Models via Formal Verification
- **分类: cs.CV; cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于医学视觉语言任务，旨在解决VLM生成报告中的逻辑不一致问题。通过构建验证框架，检测并消除不合理推理，提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2602.24111](https://arxiv.org/pdf/2602.24111)**

> **作者:** Vikash Singh; Debargha Ganguly; Haotian Yu; Chengwei Zhou; Prerna Singh; Brandon Lee; Vipin Chaudhary; Gourav Datta
>
> **摘要:** Vision-language models (VLMs) show promise in drafting radiology reports, yet they frequently suffer from logical inconsistencies, generating diagnostic impressions unsupported by their own perceptual findings or missing logically entailed conclusions. Standard lexical metrics heavily penalize clinical paraphrasing and fail to capture these deductive failures in reference-free settings. Toward guarantees for clinical reasoning, we introduce a neurosymbolic verification framework that deterministically audits the internal consistency of VLM-generated reports. Our pipeline autoformalizes free-text radiographic findings into structured propositional evidence, utilizing an SMT solver (Z3) and a clinical knowledge base to verify whether each diagnostic claim is mathematically entailed, hallucinated, or omitted. Evaluating seven VLMs across five chest X-ray benchmarks, our verifier exposes distinct reasoning failure modes, such as conservative observation and stochastic hallucination, that remain invisible to traditional metrics. On labeled datasets, enforcing solver-backed entailment acts as a rigorous post-hoc guarantee, systematically eliminating unsupported hallucinations to significantly increase diagnostic soundness and precision in generative clinical assistants.
>
---
## 更新

#### [replaced 001] Tracing and Reversing Edits in LLMs
- **分类: cs.CL**

- **简介: 该论文属于知识编辑安全任务，旨在解决LLMs被恶意编辑的问题。通过分析修改后的权重，实现编辑追踪与恢复，准确率达99%和94%。**

- **链接: [https://arxiv.org/pdf/2505.20819](https://arxiv.org/pdf/2505.20819)**

> **作者:** Paul Youssef; Zhixue Zhao; Christin Seifert; Jörg Schlötterer
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Knowledge editing methods (KEs) are a cost-effective way to update the factual content of large language models (LLMs), but they pose a dual-use risk. While KEs are beneficial for updating outdated or incorrect information, they can be exploited maliciously to implant misinformation or bias. In order to defend against these types of malicious manipulation, we need robust techniques that can reliably detect, interpret, and mitigate malicious edits. To that end, we introduce the tasks of tracing and reversing edits. We propose a novel method to infer the edited object entity, solely based on the modified weights, without access to the editing prompt or any other semantically similar prompts, with up to 99% accuracy. Further, we propose an effective and training-free method for reversing edits. Our method reverses up to 94% of the edits, and helps regain the original model's output distribution without access to any information about the edit. This method can further be repurposed to distinguish between edited and unedited weights. Our findings highlight the feasibility of tracing and reversing edits based on the edited weights, opening a new research direction for safeguarding LLMs against adversarial manipulations.
>
---
#### [replaced 002] ViMultiChoice: Toward a Method That Gives Explanation for Multiple-Choice Reading Comprehension in Vietnamese
- **分类: cs.CL**

- **简介: 该论文属于多选阅读理解任务，旨在解决模型无法解释选择原因的问题。作者构建了越南语数据集，并提出ViMultiChoice方法，同时预测答案并生成解释。**

- **链接: [https://arxiv.org/pdf/2602.09961](https://arxiv.org/pdf/2602.09961)**

> **作者:** Trung Tien Cao; Lam Minh Thai; Nghia Hieu Nguyen; Duc-Vu Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** Multiple-choice Reading Comprehension (MCRC) models aim to select the correct answer from a set of candidate options for a given question. However, they typically lack the ability to explain the reasoning behind their choices. In this paper, we introduce a novel Vietnamese dataset designed to train and evaluate MCRC models with explanation generation capabilities. Furthermore, we propose ViMultiChoice, a new method specifically designed for modeling Vietnamese reading comprehension that jointly predicts the correct answer and generates a corresponding explanation. Experimental results demonstrate that ViMultiChoice outperforms existing MCRC baselines, achieving state-of-the-art (SotA) performance on both the ViMMRC 2.0 benchmark and the newly introduced dataset. Additionally, we show that jointly training option decision and explanation generation leads to significant improvements in multiple-choice accuracy.
>
---
#### [replaced 003] MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型任务，旨在探索小规模模型的推理能力。解决“大模型才能具备推理能力”和“需大量数据训练”的假设。通过优化数据集，证明小模型也能实现强推理，提出MobileLLM-R1系列模型。**

- **链接: [https://arxiv.org/pdf/2509.24945](https://arxiv.org/pdf/2509.24945)**

> **作者:** Changsheng Zhao; Ernie Chang; Zechun Liu; Chia-Jung Chang; Wei Wen; Chen Lai; Sheng Cao; Yuandong Tian; Raghuraman Krishnamoorthi; Yangyang Shi; Vikas Chandra
>
> **备注:** ICLR 2026
>
> **摘要:** The paradigm shift in large language models (LLMs) from instinctive responses to chain-of-thought (CoT) reasoning has fueled two prevailing assumptions: (1) reasoning capabilities only emerge in sufficiently large models, and (2) such capabilities require training on massive datasets. While the first assumption has already been challenged by recent sub-billion-parameter reasoning models such as Qwen3-0.6B and DeepSeek distilled variants, the second remains largely unquestioned. In this work, we revisit the necessity of scaling to extremely large corpora (>10T tokens) for reasoning emergence. By carefully curating and resampling open-source datasets that we identify as beneficial under our designed metrics, we demonstrate that strong reasoning abilities can emerge with far less data. Specifically, we show that only ~2T tokens of high-quality data are sufficient, and pre-training with 4.2T tokens on the dataset resampled from these ~2T tokens, followed by a established post-training procedure, enables the development of MobileLLM-R1, a series of sub-billion-parameter reasoning models that substantially outperform prior models trained on fully open-sourced data. For example, MobileLLM-R1-950M achieves an AIME score of 15.5, compared to just 0.6 for OLMo-2-1.48B and 0.3 for SmolLM-2-1.7B. Remarkably, despite being trained on only 11.7% of the tokens compared to Qwen3's proprietary 36T-token corpus for pretraining, MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks. To facilitate further research in this direction, we have made the models (this https URL) and code (this https URL) publicly available, along with the complete training recipe, data sources, and data mixing ratios.
>
---
#### [replaced 004] DeepQuestion: Systematic Generation of Real-World Challenges for Evaluating LLMs Performance
- **分类: cs.CL**

- **简介: 该论文提出DeepQuestion框架，用于生成真实世界挑战以评估LLMs性能。旨在解决现有基准测试无法反映实际复杂问题的问题，通过提升认知复杂度来更准确地衡量模型能力。**

- **链接: [https://arxiv.org/pdf/2505.24532](https://arxiv.org/pdf/2505.24532)**

> **作者:** Ali Khoramfar; Ali Ramezani; Mohammad Mahdi Mohajeri; Mohammad Javad Dousti; Majid Nili Ahmadabadi; Heshaam Faili
>
> **摘要:** While Large Language Models (LLMs) achieve near-human performance on standard benchmarks, their capabilities often fail to generalize to complex, real-world problems. To bridge this gap, we introduce DeepQuestion, a scalable, automated framework that systematically elevates the cognitive complexity of existing datasets. Grounded in Bloom's taxonomy, DeepQuestion generates (1) scenario-based problems to test the application of knowledge in noisy, realistic contexts, and (2) instruction-based prompts that require models to create new questions from a given solution path, assessing synthesis and evaluation skills. Our extensive evaluation across ten leading open-source and proprietary models reveals a stark performance decline with accuracy dropping by up to 70% as tasks ascend the cognitive hierarchy. These findings underscore that current benchmarks overestimate true reasoning abilities and highlight the critical need for cognitively diverse evaluations to guide future LLM development.
>
---
#### [replaced 005] Aletheia tackles FirstProof autonomously
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文介绍Aletheia在FirstProof挑战中自主解决数学问题的性能，属于数学推理任务，旨在验证AI在自动解题方面的能力。**

- **链接: [https://arxiv.org/pdf/2602.21201](https://arxiv.org/pdf/2602.21201)**

> **作者:** Tony Feng; Junehyuk Jung; Sang-hyun Kim; Carlo Pagano; Sergei Gukov; Chiang-Chiang Tsai; David Woodruff; Adel Javanmard; Aryan Mokhtari; Dawsen Hwang; Yuri Chervonyi; Jonathan N. Lee; Garrett Bingham; Trieu H. Trinh; Vahab Mirrokni; Quoc V. Le; Thang Luong
>
> **备注:** 41 pages. Project page: this https URL
>
> **摘要:** We report the performance of Aletheia (Feng et al., 2026b), a mathematics research agent powered by Gemini 3 Deep Think, on the inaugural FirstProof challenge. Within the allowed timeframe of the challenge, Aletheia autonomously solved 6 problems (2, 5, 7, 8, 9, 10) out of 10 according to majority expert assessments; we note that experts were not unanimous on Problem 8 (only). For full transparency, we explain our interpretation of FirstProof and disclose details about our experiments as well as our evaluation. Raw prompts and outputs are available at this https URL.
>
---
#### [replaced 006] Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散语言模型在非自回归解码中的性能问题，旨在解决其实际表现趋近自回归的问题。通过改进训练数据和监督方式，提出NAP方法提升并行生成效果。**

- **链接: [https://arxiv.org/pdf/2602.23225](https://arxiv.org/pdf/2602.23225)**

> **作者:** Pengxiang Li; Dilxat Muhtar; Tianlong Chen; Lu Yin; Shiwei Liu
>
> **摘要:** Diffusion Language Models (DLMs) are often advertised as enabling parallel token generation, yet practical fast DLMs frequently converge to left-to-right, autoregressive (AR)-like decoding dynamics. In contrast, genuinely non-AR generation is promising because it removes AR's sequential bottleneck, better exploiting parallel hardware to reduce synchronization/communication overhead and improve latency scaling with output length. We argue that a primary driver of AR-like decoding is a mismatch between DLM objectives and the highly sequential structure of widely used training data, including standard pretraining corpora and long chain-of-thought (CoT) supervision. Motivated by this diagnosis, we propose NAP (Non-Autoregressive Parallel DLMs), a proof-of-concept, data-centric approach that better aligns supervision with non-AR parallel decoding. NAP curates examples as multiple independent reasoning trajectories and couples them with a parallel-forced decoding strategy that encourages multi-token parallel updates. Across math reasoning benchmarks, NAP yields stronger performance under parallel decoding than DLMs trained on standard long CoT data, with gains growing as parallelism increases. Our results suggest that revisiting data and supervision is a principled direction for mitigating AR-like behavior and moving toward genuinely non-autoregressive parallel generation in DLMs. Our code is available at this https URL.
>
---
#### [replaced 007] On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于隐私安全任务，研究如何通过成员推理攻击从大语言模型中提取训练数据。工作是整合多种MIA技术，评估其在数据提取中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.13352](https://arxiv.org/pdf/2512.13352)**

> **作者:** Ali Al Sahili; Ali Chehab; Razane Tajeddine
>
> **备注:** This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore
>
> **摘要:** Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.
>
---
#### [replaced 008] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning
- **分类: cs.CL**

- **简介: 该论文属于将科学论文转化为代码的任务，旨在解决机器学习研究中代码缺失的问题。论文提出PaperCoder框架，通过三个阶段生成高质量代码。**

- **链接: [https://arxiv.org/pdf/2504.17192](https://arxiv.org/pdf/2504.17192)**

> **作者:** Minju Seo; Jinheon Baek; Seongyun Lee; Sung Ju Hwang
>
> **备注:** ICLR 2026
>
> **摘要:** Despite the rapid growth of machine learning research, corresponding code implementations are often unavailable, making it slow and labor-intensive for researchers to reproduce results and build upon prior work. In the meantime, recent Large Language Models (LLMs) excel at understanding scientific documents and generating high-quality code. Inspired by this, we introduce PaperCoder, a multi-agent LLM framework that transforms machine learning papers into operational code repositories. PaperCoder operates in three stages: planning, where it constructs a high-level roadmap, designs the system architecture with diagrams, identifies file dependencies, and generates configuration files; analysis, which focuses on interpreting implementation-specific details; and generation, where modular, dependency-aware code is produced. Moreover, each phase is instantiated through a set of specialized agents designed to collaborate effectively across the pipeline. We then evaluate PaperCoder on generating code implementations from machine learning papers based on both model-based and human evaluations, particularly from the authors of those papers, with author-released repositories as ground truth if available. Our results demonstrate the effectiveness of PaperCoder in creating high-quality, faithful implementations. Furthermore, it consistently shows strengths in the recently released PaperBench benchmark, surpassing strong baselines by substantial margins. Code is available at: this https URL.
>
---
#### [replaced 009] PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs
- **分类: cs.CL**

- **简介: 该论文提出PTEB，一种动态文本嵌入评估方法，通过随机改写测试提升评估的鲁棒性。解决静态基准导致的性能高估问题，利用LLM生成语义不变的改写文本进行多轮评估。**

- **链接: [https://arxiv.org/pdf/2510.06730](https://arxiv.org/pdf/2510.06730)**

> **作者:** Manuel Frank; Haithem Afli
>
> **备注:** EACL 2026 (Main)
>
> **摘要:** Current sentence embedding evaluations typically rely on static test beds like the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported scores and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in gold ratings and human validation, we show that LLMs generate token-diverse but semantically preserving paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs spanning 20 datasets and 25 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute. We make the code to run PTEB publicly available.
>
---
#### [replaced 010] TWSSenti: A Novel Hybrid Framework for Topic-Wise Sentiment Analysis on Social Media Using Transformer Models
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在提升社交媒体文本的情感分类准确性和鲁棒性。通过融合多种Transformer模型，构建混合框架，解决数据噪声和语境模糊等问题。**

- **链接: [https://arxiv.org/pdf/2504.09896](https://arxiv.org/pdf/2504.09896)**

> **作者:** Aish Albladi; Md Kaosar Uddin; Minarul Islam; Cheryl Seals
>
> **备注:** 27 pages, 12 figures, includes algorithm and comparative tables
>
> **摘要:** Sentiment analysis is a crucial task in natural language processing (NLP) that enables the extraction of meaningful insights from textual data, particularly from dynamic platforms like Twitter and IMDB. This study explores a hybrid framework combining transformer-based models, specifically BERT, GPT-2, RoBERTa, XLNet, and DistilBERT, to improve sentiment classification accuracy and robustness. The framework addresses challenges such as noisy data, contextual ambiguity, and generalization across diverse datasets by leveraging the unique strengths of these models. BERT captures bidirectional context, GPT-2 enhances generative capabilities, RoBERTa optimizes contextual understanding with larger corpora and dynamic masking, XLNet models dependency through permutation-based learning, and DistilBERT offers efficiency with reduced computational overhead while maintaining high accuracy. We demonstrate text cleaning, tokenization, and feature extraction using Term Frequency Inverse Document Frequency (TF-IDF) and Bag of Words (BoW), ensure high-quality input data for the models. The hybrid approach was evaluated on benchmark datasets Sentiment140 and IMDB, achieving superior accuracy rates of 94\% and 95\%, respectively, outperforming standalone models. The results validate the effectiveness of combining multiple transformer models in ensemble-like setups to address the limitations of individual architectures. This research highlights its applicability to real-world tasks such as social media monitoring, customer sentiment analysis, and public opinion tracking which offers a pathway for future advancements in hybrid NLP frameworks.
>
---
#### [replaced 011] COMI: Coarse-to-fine Context Compression via Marginal Information Gain
- **分类: cs.CL**

- **简介: 该论文提出COMI，一种用于长文本压缩的框架，解决LLM在长上下文场景中的计算效率和冗余问题。通过MIG指标实现粗到细的压缩，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2602.01719](https://arxiv.org/pdf/2602.01719)**

> **作者:** Jiwei Tang; Shilei Liu; Zhicheng Zhang; Yujin Yuan; Libin Zheng; Wenbo Su; Bo Zheng
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks. However, their deployment in long context scenarios remains hindered by computational inefficiency and information redundancy. Context compression methods address these challenges by significantly reducing input length and eliminating redundancy. We propose COMI, a coarse-to-fine adaptive context compression framework that jointly optimizes for semantic relevance and diversity under high compression rates. We introduce Marginal Information Gain (MIG), a metric defined as the relevance of a unit to the input query minus its semantic redundancy with other units, guiding the compression process to prioritize information that is both relevant and low redundant. The framework operates in two stages: (1) Coarse-Grained Group Reallocation, where the context is partitioned into groups and dynamically assigned compression rates based on inter-group MIG, ensuring compression budgets align with information value distribution; and (2) Fine-Grained Token Merging, where tokens within each group are fused via an intra-group MIG-based weighting mechanism, thereby preserving key semantics while avoiding the accumulation of redundancy. Extensive experiments across question-answering (e.g., NaturalQuestions, 2WikiMQA, HotpotQA and NarrativeQA), summarization (e.g., MultiNews) with various backbones (e.g., LLaMA-2-7B, Qwen2-7B) show that COMI outperforms existing baselines by a large margin, e.g., approximately 25-point Exact Match (EM) improvement under 32x compression constraint with Qwen2-7B on NaturalQuestions.
>
---
#### [replaced 012] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉问答任务，解决信息密集图像中的多跳推理问题。提出SV框架，通过轻量专家生成路径，由强模型综合得出答案，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.20812](https://arxiv.org/pdf/2510.20812)**

> **作者:** Yuhan Liu; Lianhui Qin; Shengjie Wang
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence. We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers. To further improve efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict. Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from multiple partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines. Code is available at this https URL.
>
---
#### [replaced 013] LEC-KG: An LLM-Embedding Collaborative Framework for Domain-Specific Knowledge Graph Construction -- A Case Study on SDGs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱构建任务，旨在解决领域知识图谱从非结构化文本中构建的难题。通过结合大语言模型与知识图谱嵌入，提出LEC-KG框架，提升关系抽取与结构验证效果。**

- **链接: [https://arxiv.org/pdf/2602.02090](https://arxiv.org/pdf/2602.02090)**

> **作者:** Yikai Zeng; Yingchao Piao; Changhua Pei; Jianhui Li
>
> **摘要:** Constructing domain-specific knowledge graphs from unstructured text remains challenging due to heterogeneous entity mentions, long-tail relation distributions, and the absence of standardized schemas. We present LEC-KG, a bidirectional collaborative framework that integrates the semantic understanding of Large Language Models (LLMs) with the structural reasoning of Knowledge Graph Embeddings (KGE). Our approach features three key components: (1) hierarchical coarse-to-fine relation extraction that mitigates long-tail bias, (2) evidence-guided Chain-of-Thought feedback that grounds structural suggestions in source text, and (3) semantic initialization that enables structural validation for unseen entities. The two modules enhance each other iteratively-KGE provides structure-aware feedback to refine LLM extractions, while validated triples progressively improve KGE representations. We evaluate LEC-KG on Chinese Sustainable Development Goal (SDG) reports, demonstrating substantial improvements over LLM baselines, particularly on low-frequency relations. Through iterative refinement, our framework reliably transforms unstructured policy text into validated knowledge graph triples.
>
---
#### [replaced 014] HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam
- **分类: cs.CL**

- **简介: 该论文针对HLE基准中的噪声问题，提出HLE-Verified，通过专家验证和修正提升基准质量，优化模型评估效果。**

- **链接: [https://arxiv.org/pdf/2602.13964](https://arxiv.org/pdf/2602.13964)**

> **作者:** Weiqi Zhai; Zhihai Wang; Jinghang Wang; Boyu Yang; Xiaogang Li; Xander Xu; Bohan Wang; Peng Wang; Xingzhe Wu; Anfeng Li; Qiyuan Feng; Yuhao Zhou; Shoulin Han; Wenjie Luo; Yiyuan Li; Yaxuan Wang; Ruixian Luo; Guojie Lin; Peiyao Xiao; Chengliang Xu; Ben Wang; Zeyu Wang; Zichao Chen; Jianan Ye; Yijie Hu; Jialong Chen; Zongwen Shen; Yuliang Xu; An Yang; Bowen Yu; Dayiheng Liu; Junyang Lin; Hu Wei; Que Shen; Bing Zhao
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Humanity's Last Exam (HLE) has become a widely used benchmark for evaluating frontier large language models on challenging, multi-domain questions. However, community-led analyses have raised concerns that HLE contains a non-trivial number of noisy items, which can bias evaluation results and distort cross-model comparisons. To address this challenge, we introduce HLE-Verified, a verified and revised version of HLE with a transparent verification protocol and fine-grained error taxonomy. Our construction follows a two-stage validation-and-repair workflow resulting in a certified benchmark. In Stage I, each item undergoes binary validation of the problem and final answer through domain-expert review and model-based cross-checks, yielding 668 verified items. In Stage II, flawed but fixable items are revised under strict constraints preserving the original evaluation intent, through dual independent expert repairs, model-assisted auditing, and final adjudication, resulting in 1,143 revised-and-certified items. The remaining 689 items are released as a documented uncertain set with explicit uncertainty sources and expertise tags for future refinement. We evaluate eight state-of-the-art language models on HLE and HLE-Verified, observing an average absolute accuracy gain of 7--10 percentage points on HLE-Verified. The improvement is particularly pronounced on items where the original problem statement and/or reference answer is erroneous, with gains of 30--40 percentage points. Our analyses further reveal a strong association between model confidence and the presence of errors in the problem statement or reference answer, supporting the effectiveness of our revisions. Overall, HLE-Verified improves HLE-style evaluations by reducing annotation noise and enabling more faithful measurement of model capabilities. Data is available at: this https URL
>
---
#### [replaced 015] Mixed-Initiative Dialog for Human-Robot Collaborative Manipulation
- **分类: cs.RO; cs.CL; cs.HC; cs.LG; cs.MA**

- **简介: 该论文属于人机协作任务，旨在提升长期合作中机器人与人类的沟通效率。研究提出MICoBot系统，通过多级决策优化任务分配，减少人力负担，提高任务成功率和用户体验。**

- **链接: [https://arxiv.org/pdf/2508.05535](https://arxiv.org/pdf/2508.05535)**

> **作者:** Albert Yu; Chengshu Li; Luca Macesanu; Arnav Balaji; Ruchira Ray; Raymond Mooney; Roberto Martín-Martín
>
> **备注:** Project website at this https URL
>
> **摘要:** Effective robotic systems for long-horizon human-robot collaboration must adapt to a wide range of human partners, whose physical behavior, willingness to assist, and understanding of the robot's capabilities may change over time. This demands a tightly coupled communication loop that grants both agents the flexibility to propose, accept, or decline requests as they coordinate toward completing the task effectively. We apply a Mixed-Initiative dialog paradigm to Collaborative human-roBot teaming and propose MICoBot, a system that handles the common scenario where both agents, using natural language, take initiative in formulating, accepting, or rejecting proposals on who can best complete different steps of a task. To handle diverse, task-directed dialog, and find successful collaborative strategies that minimize human effort, MICoBot makes decisions at three levels: (1) a meta-planner considers human dialog to formulate and code a high-level collaboration strategy, (2) a planner optimally allocates the remaining steps to either agent based on the robot's capabilities (measured by a simulation-pretrained affordance model) and the human's estimated availability to help, and (3) an action executor decides the low-level actions to perform or words to say to the human. In physical robot trials with 18 unique human participants, MICoBot significantly improves task success and user experience over a pure LLM baseline and standard agent allocation models. See additional videos and materials at this https URL.
>
---
#### [replaced 016] Personality as Relational Infrastructure: User Perceptions of Personality-Trait-Infused LLM Messaging
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于行为改变系统任务，探讨LLM生成消息中融入人格特质的影响。研究比较不同策略，发现人格信息通过整体暴露影响用户感知，而非单条消息优化。**

- **链接: [https://arxiv.org/pdf/2602.06596](https://arxiv.org/pdf/2602.06596)**

> **作者:** Dominik P. Hofer; David Haag; Rania Islambouli; Jan D. Smeddinck
>
> **备注:** Currently under review
>
> **摘要:** Digital behaviour change systems increasingly rely on repeated, system-initiated messages to support users in everyday contexts. LLMs enable these messages to be personalised consistently across interactions, yet it remains unclear whether such personalisation improves individual messages or instead shapes users' perceptions through patterns of exposure. We explore this question in the context of LLM-generated JITAIs, which are short, context-aware messages delivered at moments deemed appropriate to support behaviour change, using physical activity as an application domain. In a controlled retrospective study, 90 participants evaluated messages generated using four LLM strategies: baseline prompting, few-shot prompting, fine-tuned models, and retrieval augmented generation, each implemented with and without Big Five Personality Traits to produce personality-aligned communication across multiple scenarios. Using ordinal multilevel models with within-between decomposition, we distinguish trial-level effects, whether personality information improves evaluations of individual messages, from person-level exposure effects, whether participants receiving higher proportions of personality-informed messages exhibit systematically different overall perceptions. Results showed no trial-level associations, but participants who received higher proportions of BFPT-informed messages rated the messages as more personalised, appropriate, and reported less negative affect. We use Communication Accommodation Theory for post-hoc analysis. These results suggest that personality-based personalisation in behaviour change systems may operate primarily through aggregate exposure rather than per-message optimisation, with implications for how adaptive systems are designed and evaluated in sustained human-AI interaction. In-situ longitudinal studies are needed to validate these findings in real-world contexts.
>
---
#### [replaced 017] Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric
- **分类: cs.CL**

- **简介: 该论文提出OpenRS系统，解决开放任务中强化学习的对齐问题，通过显式规则框架提升奖励机制的可解释性和稳定性。**

- **链接: [https://arxiv.org/pdf/2602.14069](https://arxiv.org/pdf/2602.14069)**

> **作者:** Ruipeng Jia; Yunyi Yang; Yuxin Wu; Yongbo Gai; Siyuan Tao; Mengyu Zhou; Jianhe Lin; Xiaoxi Jiang; Guanjun Jiang
>
> **摘要:** Scalar reward models compress multi-dimensional human preferences into a single opaque score, creating an information bottleneck that often leads to brittleness and reward hacking in open-ended alignment. We argue that robust alignment for non-verifiable tasks is fundamentally a principle generalization problem: reward should not be a learned function internalized into a judge, but an explicit reasoning process executed under inspectable principles. To operationalize this view, we present the Open Rubric System (OpenRS), a plug-and-play, rubrics-based LLM-as-a-Judge framework built around Pairwise Adaptive Meta-Rubrics (PAMR) and lightweight Pointwise Verifiable Rubrics (PVRs), which provide both hard-constraint guardrails and verifiable reward components when ground-truth or programmatic checks are available. OpenRS uses an explicit meta-rubric -- a constitution-like specification that governs how rubrics are instantiated, weighted, and enforced -- and instantiates adaptive rubrics on the fly by conditioning on the semantic differences between two candidate responses. It then performs criterion-wise pairwise comparisons and aggregates criterion-level preferences externally, avoiding pointwise weighted scalarization while improving discriminability in open-ended settings. To keep principles consistent yet editable across various domains, we introduce a two-level meta-rubric refinement pipeline (automated evolutionary refinement for general principles and a reproducible human-in-the-loop procedure for domain principles), complemented with pointwise verifiable rubrics that act as both guardrails against degenerate behaviors and a source of verifiable reward for objective sub-tasks. Finally, we instantiate OpenRS as reward supervision in pairwise RL training.
>
---
#### [replaced 018] FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data
- **分类: cs.IR; cs.AI; cs.CL; cs.LG; q-fin.ST**

- **简介: 该论文属于金融领域任务，旨在解决LLM在实时金融数据处理上的不足。通过构建数据集和优化模型，提升模型对动态金融信息的响应能力。**

- **链接: [https://arxiv.org/pdf/2502.18471](https://arxiv.org/pdf/2502.18471)**

> **作者:** Ankur Sinha; Chaitanya Agarwal; Pekka Malo
>
> **备注:** 39 pages, 10 tables
>
> **摘要:** Large language models (LLMs) excel at generating human-like responses but often struggle with interactive tasks that require access to real-time information. This limitation poses challenges in finance, where models must access up-to-date information, such as recent news or price movements, to support decision-making. To address this, we introduce Financial Agent, a knowledge-grounding approach for LLMs to handle financial queries using real-time text and tabular data. Our contributions are threefold: First, we develop a Financial Context Dataset of over 50,000 financial queries paired with the required context. Second, we develop FinBloom 7B, a custom 7 billion parameter LLM, by fine-tuning Bloom 7B on 14 million financial news articles from Reuters and Deutsche Presse-Agentur (DPA), alongside a random sample of 25% from 12 million Securities and Exchange Commission (SEC) filings. Third, we fine-tune FinBloom 7B using the Financial Context Dataset to serve as a Financial Agent. This agent generates relevant financial context, enabling efficient real-time data retrieval to answer user queries. By reducing latency and eliminating the need for users to manually provide accurate data, our approach significantly enhances the capability of LLMs to handle dynamic financial tasks. Our proposed approach makes real-time financial decisions, algorithmic trading and other related tasks streamlined, and is valuable in contexts with high-velocity data flows.
>
---
#### [replaced 019] Low-Resource Dialect Adaptation of Large Language Models: A French Dialect Case-Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源方言适应任务，旨在解决大语言模型在方言上的性能不足问题。通过持续预训练和参数高效微调，将模型适配到魁北克法语方言，并验证其效果。**

- **链接: [https://arxiv.org/pdf/2510.22747](https://arxiv.org/pdf/2510.22747)**

> **作者:** Eeham Khan; Firas Saidani; Owen Van Esbroeck; Richard Khoury; Leila Kosseim
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Despite the widespread adoption of Large Language Models (LLMs), their strongest capabilities remain largely confined to a small number of high-resource languages for which there is abundant training data. Recently, continual pre-training (CPT) has emerged as a means to fine-tune these models to low-resource regional dialects. In this paper, we study the use of CPT for dialect learning under tight data and compute budgets. Using low-rank adaptation (LoRA) and compute-efficient continual pre-training, we adapt three LLMs to the Québec French dialect using a very small dataset and benchmark them on the COLE suite. Our experiments demonstrate an improvement on the minority dialect benchmarks with minimal regression on the prestige language benchmarks with around 1% of model parameters updated. Analysis of the results demonstrate that gains are highly contingent on corpus composition. These findings indicate that CPT with parameter-efficient fine-tuning (PEFT) can narrow the dialect gap by providing cost-effective and sustainable language resource creation, expanding high-quality LLM access to minority linguistic communities. To support reproducibility and broaden access, we release the first Québec French LLMs on Hugging Face.
>
---
#### [replaced 020] PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于个性化语言模型任务，解决用户交互历史整合难题。提出基于知识图的外部记忆框架，支持动态语义和时间表示，提升长周期交互能力。**

- **链接: [https://arxiv.org/pdf/2506.17001](https://arxiv.org/pdf/2506.17001)**

> **作者:** Mikhail Menschikov; Dmitry Evseev; Victoria Dochkina; Ruslan Kostoev; Ilia Perepechkin; Petr Anokhin; Nikita Semenov; Evgeny Burnaev
>
> **摘要:** Personalizing language models that effectively incorporating user interaction history remains a central challenge in development of adaptive AI systems. While large language models (LLMs), combined with Retrieval-Augmented Generation (RAG), have improved factual accuracy, they often lack structured memory and fail to scale in complex, long-term interactions. To address this, we propose a flexible external memory framework based on knowledge graph, which construct and update memory model automatically by LLM itself. Building upon the AriGraph architecture, we introduce a novel hybrid graph design that supports both standard edges and two types of hyper-edges, enabling rich and dynamic semantic and temporal representations. Our framework also supports diverse retrieval mechanisms, including A*, water-circle traversal, beam search and hybrid methods, making it adaptable to different datasets and LLM capacities. We evaluate our system on three benchmarks: TriviaQA, HotpotQA, DiaASQ and demonstrate that different memory and retrieval configurations yield optimal performance depending on the task. Additionally, we extend the DiaASQ benchmark with temporal annotations and internally contradictory statements, showing that our system remains robust and effective in managing temporal dependencies and context-aware reasoning.
>
---
#### [replaced 021] CowPilot: A Framework for Autonomous and Human-Agent Collaborative Web Navigation
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出CowPilot框架，解决人机协作网页导航任务中的效率与成功率问题。通过人机协同，减少人工操作步骤，提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2501.16609](https://arxiv.org/pdf/2501.16609)**

> **作者:** Faria Huq; Zora Zhiruo Wang; Frank F. Xu; Tianyue Ou; Shuyan Zhou; Jeffrey P. Bigham; Graham Neubig
>
> **备注:** Published at NAACL System Demonstration Track, 2025
>
> **摘要:** While much work on web agents emphasizes the promise of autonomously performing tasks on behalf of users, in reality, agents often fall short on complex tasks in real-world contexts and modeling user preference. This presents an opportunity for humans to collaborate with the agent and leverage the agent's capabilities effectively. We propose CowPilot, a framework supporting autonomous as well as human-agent collaborative web navigation, and evaluation across task success and task efficiency. CowPilot reduces the number of steps humans need to perform by allowing agents to propose next steps, while users are able to pause, reject, or take alternative actions. During execution, users can interleave their actions with the agent by overriding suggestions or resuming agent control when needed. We conducted case studies on five common websites and found that the human-agent collaborative mode achieves the highest success rate of 95% while requiring humans to perform only 15.2% of the total steps. Even with human interventions during task execution, the agent successfully drives up to half of task success on its own. CowPilot can serve as a useful tool for data collection and agent evaluation across websites, which we believe will enable research in how users and agents can work together. Video demonstrations are available at this https URL
>
---
#### [replaced 022] Modeling Distinct Human Interaction in Web Agents
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决人类何时及为何干预的问题。通过分析用户与代理的交互模式，训练语言模型预测干预行为，提升代理协作能力。**

- **链接: [https://arxiv.org/pdf/2602.17588](https://arxiv.org/pdf/2602.17588)**

> **作者:** Faria Huq; Zora Zhiruo Wang; Zhanqiu Guo; Venu Arvind Arangarajan; Tianyue Ou; Frank Xu; Shuyan Zhou; Graham Neubig; Jeffrey P. Bigham
>
> **备注:** Preprint
>
> **摘要:** Despite rapid progress in autonomous web agents, human involvement remains essential for shaping preferences and correcting agent behavior as tasks unfold. However, current agentic systems lack a principled understanding of when and why humans intervene, often proceeding autonomously past critical decision points or requesting unnecessary confirmation. In this work, we introduce the task of modeling human intervention to support collaborative web task execution. We collect CowCorpus, a dataset of 400 real-user web navigation trajectories containing over 4,200 interleaved human and agent actions. We identify four distinct patterns of user interaction with agents -- hands-off supervision, hands-on oversight, collaborative task-solving, and full user takeover. Leveraging these insights, we train language models (LMs) to anticipate when users are likely to intervene based on their interaction styles, yielding a 61.4-63.4% improvement in intervention prediction accuracy over base LMs. Finally, we deploy these intervention-aware models in live web navigation agents and evaluate them in a user study, finding a 26.5% increase in user-rated agent usefulness. Together, our results show structured modeling of human intervention leads to more adaptive, collaborative agents.
>
---
#### [replaced 023] Modeling Clinical Uncertainty in Radiology Reports: from Explicit Uncertainty Markers to Implicit Reasoning Pathways
- **分类: cs.CL**

- **简介: 该论文属于医学自然语言处理任务，旨在解决放射报告中的不确定性建模问题。通过量化显性不确定性和建模隐性不确定性，提升诊断推理的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2511.04506](https://arxiv.org/pdf/2511.04506)**

> **作者:** Paloma Rabaey; Jong Hak Moon; Jung-Oh Lee; Min Gwan Kim; Hangyul Yoon; Thomas Demeester; Edward Choi
>
> **摘要:** Radiology reports are invaluable for clinical decision-making and hold great potential for automated analysis when structured into machine-readable formats. These reports often contain uncertainty, which we categorize into two distinct types: (i) Explicit uncertainty reflects doubt about the presence or absence of findings, conveyed through hedging phrases. These vary in meaning depending on the context, making rule-based systems insufficient to quantify the level of uncertainty for specific findings; (ii) Implicit uncertainty arises when radiologists omit parts of their reasoning, recording only key findings or diagnoses. Here, it is often unclear whether omitted findings are truly absent or simply unmentioned for brevity. We address these challenges with a two-part framework. We quantify explicit uncertainty by creating an expert-validated, LLM-based reference ranking of common hedging phrases, and mapping each finding to a probability value based on this reference. In addition, we model implicit uncertainty through an expansion framework that systematically adds characteristic sub-findings derived from expert-defined diagnostic pathways for 14 common diagnoses. Using these methods, we release Lunguage++, an expanded, uncertainty-aware version of the Lunguage benchmark of fine-grained structured radiology reports. This enriched resource enables uncertainty-aware image classification, faithful diagnostic reasoning, and new investigations into the clinical impact of diagnostic uncertainty.
>
---
#### [replaced 024] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决LRMs生成冗长思考过程的问题。通过引入ARLCP框架，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.12113](https://arxiv.org/pdf/2602.12113)**

> **作者:** Zewei Yu; Lirong Gao; Yuke Zhu; Bo Zheng; Junbo Zhao; Sheng Guo; Haobo Wang
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks by employing test-time scaling. However, they often generate over-long chains-of-thought that, driven by substantial reflections such as repetitive self-questioning and circular reasoning, lead to high token consumption, substantial computational overhead, and increased latency without improving accuracy, particularly in smaller models. Our observation reveals that increasing problem complexity induces more excessive and unnecessary reflection, which in turn reduces accuracy and increases token overhead. To address this challenge, we propose Adaptive Reflection and Length Coordinated Penalty (ARLCP), a novel reinforcement learning framework designed to dynamically balance reasoning efficiency and solution accuracy. ARLCP introduces two key innovations: (1) a reflection penalty that adaptively curtails unnecessary reflective steps while preserving essential reasoning, and (2) a length penalty calibrated to the estimated complexity of the problem. By coordinating these penalties, ARLCP encourages the model to generate more concise and effective reasoning paths. We evaluate our method on five mathematical reasoning benchmarks using DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B models. Experimental results show that ARLCP achieves a superior efficiency-accuracy trade-off compared to existing approaches. For the 1.5B model, it reduces the average response length by 53.1% while simultaneously improving accuracy by 5.8%. For the 7B model, it achieves a 35.0% reduction in length with a 2.7% accuracy gain. The code is released at this https URL .
>
---
#### [replaced 025] DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在提升LLM推理效率。通过分析推理过程中的熵变化，提出DiffAdapt框架，按难度选择不同推理策略，减少token使用并保持性能。**

- **链接: [https://arxiv.org/pdf/2510.19669](https://arxiv.org/pdf/2510.19669)**

> **作者:** Xiang Liu; Xuming Hu; Xiaowen Chu; Eunsol Choi
>
> **备注:** ICLR 26
>
> **摘要:** Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.
>
---
#### [replaced 026] Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization
- **分类: cs.CL**

- **简介: 该论文属于智能搜索任务，旨在解决长周期代理搜索的效率与泛化问题。提出SMTL框架，通过并行证据获取提升效率，并构建统一数据合成管道增强泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.22675](https://arxiv.org/pdf/2602.22675)**

> **作者:** Qianben Chen; Tianrui Qin; King Zhu; Qiexiang Wang; Chengjun Yu; Shu Xu; Jiaqi Wu; Jiayu Zhang; Xinpeng Liu; Xin Gui; Jingyi Cao; Piaohong Wang; Dingfeng Shi; He Zhu; Tiannan Wang; Yuqing Wang; Maojia Song; Tianyu Zheng; Ge Zhang; Jian Yang; Jiaheng Liu; Minghao Liu; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Recent deep research agents primarily improve performance by scaling reasoning depth, but this leads to high inference cost and latency in search-intensive scenarios. Moreover, generalization across heterogeneous research settings remains challenging. In this work, we propose \emph{Search More, Think Less} (SMTL), a framework for long-horizon agentic search that targets both efficiency and generalization. SMTL replaces sequential reasoning with parallel evidence acquisition, enabling efficient context management under constrained context budgets. To support generalization across task types, we further introduce a unified data synthesis pipeline that constructs search tasks spanning both deterministic question answering and open-ended research scenarios with task appropriate evaluation metrics. We train an end-to-end agent using supervised fine-tuning and reinforcement learning, achieving strong and often state of the art performance across benchmarks including BrowseComp (48.6\%), GAIA (75.7\%), Xbench (82.0\%), and DeepResearch Bench (45.9\%). Compared to Mirothinker-v1.0, SMTL with maximum 100 interaction steps reduces the average number of reasoning steps on BrowseComp by 70.7\%, while improving accuracy.
>
---
#### [replaced 027] CSyMR: Benchmarking Compositional Music Information Retrieval in Symbolic Music Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于符号音乐推理任务，旨在解决自然语言与符号音乐表示不匹配的问题。提出CSyMR-Bench基准和工具增强的检索框架，提升组合式音乐信息检索效果。**

- **链接: [https://arxiv.org/pdf/2601.11556](https://arxiv.org/pdf/2601.11556)**

> **作者:** Boyang Wang; Yash Vishe; Xin Xu; Zachary Novack; Xunyi Jiang; Julian McAuley; Junda Wu
>
> **摘要:** Natural language information needs over symbolic music scores rarely reduce to a single step lookup. Many queries require compositional Music Information Retrieval (MIR) that extracts multiple pieces of evidence from structured notation and aggregates them to answer the question. This setting remains challenging for Large Language Models due to the mismatch between natural language intents and symbolic representations, as well as the difficulty of reliably handling long structured contexts. Existing benchmarks only partially capture these retrieval demands, often emphasizing isolated theoretical knowledge or simplified settings. We introduce CSyMR-Bench, a benchmark for compositional MIR in symbolic music reasoning grounded in authentic user scenarios. It contains 126 multiple choice questions curated from community discussions and professional examinations, where each item requires chaining multiple atomic analyses over a score to derive implicit musical evidence. To support diagnosis, we provide a taxonomy with six query intent categories and six analytical dimension tags. We further propose a tool-augmented retrieval and reasoning framework that integrates a ReAct-style controller with deterministic symbolic analysis operators built with music21. Experiments across prompting baselines and agent variants show that tool-grounded compositional retrieval consistently outperforms Large Language Model-only approaches, yielding 5-7% absolute accuracy gains, with the largest improvements on analysis-heavy categories.
>
---
#### [replaced 028] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决大推理模型过思考导致的高推理成本问题。通过引入反射模型和反射奖励，提升推理效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2505.19862](https://arxiv.org/pdf/2505.19862)**

> **作者:** Hexuan Deng; Wenxiang Jiao; Xuebo Liu; Jun Rao; Min Zhang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate strong performance in complex tasks but often face the challenge of overthinking, leading to substantially high inference costs. Existing approaches synthesize shorter reasoning responses for LRMs to learn, but are inefficient for online usage due to the time-consuming data generation and filtering processes. Meanwhile, online reinforcement learning mainly adopts a length reward to encourage short reasoning responses, but it tends to lose reflection ability and harm performance. To address these issues, we propose REA-RL, which introduces a small reflection model for efficient scaling in online training, offering both parallel sampling and sequential revision. Besides, a reflection reward is designed to further prevent LRMs from favoring short yet non-reflective responses. Experiments show that both methods maintain or enhance performance while significantly improving inference efficiency. Their combination achieves a good balance between performance and efficiency, reducing inference costs by 36% without compromising performance. Further analysis demonstrates that our methods are effective by maintaining reflection frequency for hard problems while appropriately reducing it for easier ones without losing reflection ability. Code is available at this https URL.
>
---
#### [replaced 029] MLP Memory: A Retriever-Pretrained Memory for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MLP Memory，解决大语言模型知识准确性和效率的矛盾。通过预训练MLP模拟检索行为，提升知识获取效果并加快推理速度。**

- **链接: [https://arxiv.org/pdf/2508.01832](https://arxiv.org/pdf/2508.01832)**

> **作者:** Rubin Wei; Jiaqi Cao; Jiarui Wang; Jushi Kai; Qipeng Guo; Bowen Zhou; Zhouhan Lin
>
> **摘要:** Modern approaches to enhancing Large Language Models' factual accuracy and knowledge utilization face a fundamental trade-off: non-parametric retrieval-augmented generation (RAG) provides flexible access to external knowledge but suffers from high inference latency and shallow integration, while parametric fine-tuning methods like LoRA risk catastrophic forgetting and degraded general capabilities. In this work, we propose MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns without explicit document access. By pretraining an MLP to imitate a $k$NN retriever's behavior on the entire pretraining dataset, we create a differentiable memory component that captures the benefits of retrieval-based knowledge access in a fully parametric form. Our architecture integrates this pretrained MLP Memory with Transformer decoders through simple probability interpolation, yielding 17.5\% and 24.1\% scaling gains on WikiText-103 and Web datasets, respectively. It further achieves 12.3\% relative improvement on five question-answering benchmarks and 5.2 points absolute gain across nine general NLP tasks, while reducing hallucinations by up to 10 points on HaluEval. Moreover, MLP Memory delivers 2.5$\times$ faster inference than RAG with superior accuracy. Our findings show that learning retrieval patterns parametrically bridges the gap between efficient inference and effective knowledge access, offering a practical alternative to both RAG and fine-tuning approaches.
>
---
#### [replaced 030] GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决方言数据不足的问题。通过扩展希腊方言数据集，加入多种方言并进行模型微调实验，评估高质量方言数据对大语言模型的影响。**

- **链接: [https://arxiv.org/pdf/2511.03772](https://arxiv.org/pdf/2511.03772)**

> **作者:** Stergios Chatzikyriakidis; Dimitris Papadakis; Sevasti-Ioanna Papaioannou; Erofili Psaltaki
>
> **摘要:** We present an extended Greek Dialectal Dataset (GRDD+) 1that complements the existing GRDD dataset with more data from Cretan, Cypriot, Pontic and Northern Greek, while we add six new varieties: Greco-Corsican, Griko (Southern Italian Greek), Maniot, Heptanesian, Tsakonian, and Katharevusa Greek. The result is a dataset with total size 6,374,939 words and 10 varieties. This is the first dataset with such variation and size to date. We conduct a number of fine-tuning experiments to see the effect of good quality dialectal data on a number of LLMs. We fine-tune three model architectures (Llama-3-8B, Llama-3.1-8B, Krikri-8B) and compare the results to frontier models (Claude-3.7-Sonnet, Gemini-2.5, ChatGPT-5).
>
---
#### [replaced 031] R2GenCSR: Mining Contextual and Residual Information for LLMs-based Radiology Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于放射学报告生成任务，旨在提升LLMs生成报告的质量。针对特征提取效率和计算复杂度问题，提出R2GenCSR框架，结合Mamba和上下文检索优化特征表示。**

- **链接: [https://arxiv.org/pdf/2408.09743](https://arxiv.org/pdf/2408.09743)**

> **作者:** Xiao Wang; Yuehang Li; Fuling Wang; Shiao Wang; Chuanfu Li; Bo Jiang
>
> **备注:** R2GenCSR is accepted by IEEE Journal of Biomedical and Health Informatics (JBHI) 2026
>
> **摘要:** Inspired by the tremendous success of Large Language Models (LLMs), existing Radiology report generation methods attempt to leverage large models to achieve better performance. They usually adopt a Transformer to extract the visual features of a given X-ray image, and then, feed them into the LLM for text generation. How to extract more effective information for the LLMs to help them improve final results is an urgent problem that needs to be solved. Additionally, the use of visual Transformer models also brings high computational complexity. To address these issues, this paper proposes a novel context-guided efficient radiology report generation framework. Specifically, we introduce the Mamba as the vision backbone with linear complexity, and the performance obtained is comparable to that of the strong Transformer model. More importantly, we perform context retrieval from the training set for samples within each mini-batch during the training phase, utilizing both positively and negatively related samples to enhance feature representation and discriminative learning. Subsequently, we feed the vision tokens, context information, and prompt statements to invoke the LLM for generating high-quality medical reports. Extensive experiments on three X-ray report generation datasets (i.e., IU X-Ray, MIMIC-CXR, CheXpert Plus) fully validated the effectiveness of our proposed model. The source code is available at this https URL.
>
---
#### [replaced 032] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language
- **分类: cs.CL**

- **简介: 该论文属于模型可解释性任务，旨在解决LLM特征描述不精确、不一致的问题。提出语义正则表达式，结构化描述特征，提升准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2510.06378](https://arxiv.org/pdf/2510.06378)**

> **作者:** Angie Boggust; Donghao Ren; Yannick Assogba; Dominik Moritz; Arvind Satyanarayan; Fred Hohman
>
> **备注:** ICLR 2026
>
> **摘要:** Automated interpretability aims to translate large language model (LLM) features into human understandable descriptions. However, natural language feature descriptions can be vague, inconsistent, and require manual relabeling. In response, we introduce semantic regexes, structured language descriptions of LLM features. By combining primitives that capture linguistic and semantic patterns with modifiers for contextualization, composition, and quantification, semantic regexes produce precise and expressive feature descriptions. Across quantitative benchmarks and qualitative analyses, semantic regexes match the accuracy of natural language while yielding more concise and consistent feature descriptions. Their inherent structure affords new types of analyses, including quantifying feature complexity across layers, scaling automated interpretability from insights into individual features to model-wide patterns. Finally, in user studies, we find that semantic regexes help people build accurate mental models of LLM features.
>
---
#### [replaced 033] Steering Language Models with Weight Arithmetic
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型调整任务，旨在解决LLM在特定数据上训练后出现的意外泛化问题。通过权重算术方法修改模型参数，实现对行为的有效控制。**

- **链接: [https://arxiv.org/pdf/2511.05408](https://arxiv.org/pdf/2511.05408)**

> **作者:** Constanza Fierro; Fabien Roger
>
> **备注:** ICLR 2026 camera-ready
>
> **摘要:** Providing high-quality feedback to Large Language Models (LLMs) on a diverse training distribution can be difficult and expensive, and providing feedback only on a narrow distribution can result in unintended generalizations. To better leverage narrow training data, we propose contrastive weight steering, a simple post-training method that edits the model parameters using weight arithmetic. We isolate a behavior direction in weight-space by subtracting the weight deltas from two small fine-tunes -- one that induces the desired behavior and another that induces its opposite -- and then add or remove this direction to modify the model's weights. We apply this technique to mitigate sycophancy and induce misalignment, and find that weight steering often generalizes further than activation steering, achieving stronger out-of-distribution behavioral control before degrading general capabilities. We also show that, in the context of task-specific fine-tuning, weight steering can partially mitigate undesired behavioral drift: it can reduce sycophancy and under-refusals introduced during fine-tuning while preserving task performance gains. Finally, we provide preliminary evidence that emergent misalignment can be detected by measuring the similarity between fine-tuning updates and an "evil" weight direction, suggesting that it may be possible to monitor the evolution of weights during training and detect rare misaligned behaviors that never manifest during training or evaluations.
>
---
#### [replaced 034] Interpreting Transformers Through Attention Head Intervention
- **分类: cs.CL**

- **简介: 论文探讨了通过注意力头干预理解Transformer模型的机制，属于模型可解释性研究。旨在解决如何通过干预揭示模型决策过程，验证其因果机制，并应用于AI安全控制。**

- **链接: [https://arxiv.org/pdf/2601.04398](https://arxiv.org/pdf/2601.04398)**

> **作者:** Mason Kadem; Rong Zheng
>
> **备注:** minor citation fix
>
> **摘要:** Neural networks are growing more capable on their own, but we do not understand their neural mechanisms. Understanding these mechanisms' decision-making processes, or mechanistic interpretability, enables (1) accountability and control in high-stakes domains, (2) the study of digital brains and the emergence of cognition, and (3) discovery of new knowledge when AI systems outperform humans. This paper traces how attention head intervention emerged as a key method for causal interpretability of transformers. The evolution from visualization to intervention represents a paradigm shift from observing correlations to causally validating mechanistic hypotheses through direct intervention. Head intervention studies revealed robust empirical findings while also highlighting limitations that complicate interpretation. Recent work demonstrates that mechanistic understanding now enables targeted control of model behaviour, successfully suppressing toxic outputs and manipulating semantic content through selective attention head intervention, validating the practical utility of interpretability research for AI safety.
>
---
#### [replaced 035] Intention-Adaptive LLM Fine-Tuning for Text Revision Generation
- **分类: cs.CL**

- **简介: 该论文属于文本修订生成任务，旨在解决意图不明确或多重意图下的修订生成问题。提出Intention-Tuning框架，通过动态选择模型层来适应不同意图，提升修订效果。**

- **链接: [https://arxiv.org/pdf/2602.00477](https://arxiv.org/pdf/2602.00477)**

> **作者:** Zhexiong Liu; Diane Litman
>
> **备注:** In the Conference of the European Chapter of the Association for Computational Linguistics (EACL), March 2026
>
> **摘要:** Large Language Models (LLMs) have achieved impressive capabilities in various context-based text generation tasks, such as summarization and reasoning; however, their applications in intention-based generation tasks remain underexplored. One such example is revision generation, which requires the generated text to explicitly reflect the writer's actual intentions. Identifying intentions and generating desirable revisions are challenging due to their complex and diverse nature. Although prior work has employed LLMs to generate revisions with few-shot learning, they struggle with handling entangled multi-intent scenarios. While fine-tuning LLMs using intention-based instructions appears promising, it demands large amounts of annotated data, which is expensive and scarce in the revision community. To address these challenges, we propose Intention-Tuning, an intention-adaptive layer-wise LLM fine-tuning framework that dynamically selects a subset of LLM layers to learn the intentions and subsequently transfers their representations to revision generation. Experimental results suggest that Intention-Tuning is effective and efficient on small revision corpora, outperforming several PEFT baselines.
>
---
#### [replaced 036] Scaling Generalist Data-Analytic Agents
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于数据解析代理任务，旨在解决开源模型在处理多样化数据和复杂推理上的不足。提出DataMind框架，提升代理的泛化能力与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.25084](https://arxiv.org/pdf/2509.25084)**

> **作者:** Shuofei Qiao; Yanqiu Zhao; Zhisong Qiu; Xiaobin Wang; Jintian Zhang; Zhao Bin; Ningyu Zhang; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen
>
> **备注:** ICLR 2026
>
> **摘要:** Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research.
>
---
#### [replaced 037] RCPU: Rotation-Constrained Error Compensation for Structured Pruning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型压缩任务，旨在解决结构化剪枝导致的输出误差问题。通过旋转约束补偿和方差感知重要性评分，有效提升剪枝模型性能。**

- **链接: [https://arxiv.org/pdf/2510.07782](https://arxiv.org/pdf/2510.07782)**

> **作者:** Shuichiro Haruta; Kazunori Matsumoto; Zhi Li; Yanan Wang; Mori Kurokawa
>
> **备注:** Accepted as ICLR2026
>
> **摘要:** In this paper, we propose a rotation-constrained compensation method to address the errors introduced by structured pruning of large language models (LLMs). LLMs are trained on massive datasets and accumulate rich semantic knowledge in their representation space. In contrast, pruning is typically carried out with only a small amount of calibration data, which makes output mismatches unavoidable. Although direct least-squares fitting can reduce such errors, it tends to overfit to the limited calibration set, destructively modifying pretrained weights. To overcome this difficulty, we update the pruned parameters under a rotation constraint. This constrained update preserves the geometry of output representations (i.e., norms and inner products) and simultaneously re-aligns the pruned subspace with the original outputs. Furthermore, in rotation-constrained compensation, removing components that strongly contribute to the principal directions of the output makes error recovery difficult. Since input dimensions with large variance strongly affect these principal directions, we design a variance-aware importance score that ensures such dimensions are preferentially kept in the pruned model. By combining this scoring rule with rotation-constrained updates, the proposed method effectively compensates errors while retaining the components likely to be more important in a geometry-preserving manner. In the experiments, we apply the proposed method to Llama-7B and Llama-2-13B, and evaluate it on WikiText2 and multiple language understanding benchmarks. The results demonstrate consistently better perplexity and task accuracy compared with existing baselines.
>
---
#### [replaced 038] The Growing Gains and Pains of Iterative Web Corpora Crawling: Insights from South Slavic CLASSLA-web 2.0 Corpora
- **分类: cs.CL**

- **简介: 该论文属于语言资源建设任务，旨在通过迭代爬取南斯拉夫语族网站，构建大规模语料库。解决如何持续获取高质量文本的问题，工作包括建立爬虫系统并生成CLASSLA-web 2.0语料库。**

- **链接: [https://arxiv.org/pdf/2601.11170](https://arxiv.org/pdf/2601.11170)**

> **作者:** Taja Kuzman Pungeršek; Peter Rupnik; Vít Suchomel; Nikola Ljubešić
>
> **备注:** 11 pages, 7 figures, 2 tables. Accepted at the LREC 2026 conference
>
> **摘要:** Crawling national top-level domains has proven to be highly effective for collecting texts in less-resourced languages. This approach has been recently used for South Slavic languages and resulted in the largest general corpora for this language group: the CLASSLA-web 1.0 corpora. Building on this success, we established a continuous crawling infrastructure for iterative national top-level domain crawling across South Slavic and related webs. We present the first outcome of this crawling infrastructure - the CLASSLA-web 2.0 corpus collection, with substantially larger web corpora containing 17.0 billion words in 38.1 million texts in seven languages: Bosnian, Bulgarian, Croatian, Macedonian, Montenegrin, Serbian, and Slovenian. In addition to genre categories, the new version is also automatically annotated with topic labels. Comparing CLASSLA-web 2.0 with its predecessor reveals that only one-fifth of the texts overlap, showing that re-crawling after just two years yields largely new content. However, while the new web crawls bring growing gains, we also notice growing pains - a manual inspection of top domains reveals a visible degradation of web content, as machine-generated sites now contribute a significant portion of texts.
>
---
#### [replaced 039] Read As Human: Compressing Context via Parallelizable Close Reading and Skimming
- **分类: cs.CL**

- **简介: 该论文提出RAM框架，用于长文本的上下文压缩，解决LLM在长输入场景下的计算效率和冗余问题。通过混合阅读策略，提升处理速度与性能。**

- **链接: [https://arxiv.org/pdf/2602.01840](https://arxiv.org/pdf/2602.01840)**

> **作者:** Jiwei Tang; Shilei Liu; Zhicheng Zhang; Qingsong Lv; Runsong Zhao; Tingwei Lu; Langming Liu; Haibin Chen; Yujin Yuan; Hai-Tao Zheng; Wenbo Su; Bo Zheng
>
> **备注:** 13 pages,5 figures (Compared with v1, the author affiliations have been corrected.)
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional capability across diverse tasks. However, their deployment in long-context scenarios is hindered by two challenges: computational inefficiency and redundant information. We propose RAM (Read As HuMan), a context compression framework that adopts an adaptive hybrid reading strategy, to address these challenges. Inspired by human reading behavior (i.e., close reading important content while skimming less relevant content), RAM partitions the context into segments and encodes them with the input query in parallel. High-relevance segments are fully retained (close reading), while low-relevance ones are query-guided compressed into compact summary vectors (skimming). Both explicit textual segments and implicit summary vectors are concatenated and fed into decoder to achieve both superior performance and natural language format interpretability. To refine the decision boundary between close reading and skimming, we further introduce a contrastive learning objective based on positive and negative query-segment pairs. Experiments demonstrate that RAM outperforms existing baselines on multiple question answering and summarization benchmarks across two backbones, while delivering up to a 12x end-to-end speedup on long inputs (average length 16K; maximum length 32K).
>
---
#### [replaced 040] Measuring Sycophancy of Language Models in Multi-turn Dialogues
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型在对话中盲目迎合用户的问题。通过构建基准测试，分析模型的迎合行为，并探索减少该行为的方法。**

- **链接: [https://arxiv.org/pdf/2505.23840](https://arxiv.org/pdf/2505.23840)**

> **作者:** Jiseung Hong; Grace Byun; Seungone Kim; Kai Shu; Jinho D. Choi
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Large Language Models (LLMs) are expected to provide helpful and harmless responses, yet they often exhibit sycophancy--conforming to user beliefs regardless of factual accuracy or ethical soundness. Prior research on sycophancy has primarily focused on single-turn factual correctness, overlooking the dynamics of real-world interactions. In this work, we introduce SYCON Bench, a novel benchmark for evaluating sycophantic behavior in multi-turn, free-form conversational settings. Our benchmark measures how quickly a model conforms to the user (Turn of Flip) and how frequently it shifts its stance under sustained user pressure (Number of Flip). Applying SYCON Bench to 17 LLMs across three real-world scenarios, we find that sycophancy remains a prevalent failure mode. Our analysis shows that alignment tuning amplifies sycophantic behavior, whereas model scaling and reasoning optimization strengthen the model's ability to resist undesirable user views. Reasoning models generally outperform instruction-tuned models but often fail when they over-index on logical exposition instead of directly addressing the user's underlying beliefs. Finally, we evaluate four additional prompting strategies and demonstrate that adopting a third-person perspective reduces sycophancy by up to 63.8% in debate scenario. We release our code and data at this https URL.
>
---
#### [replaced 041] Bridging Latent Reasoning and Target-Language Generation via Retrieval-Transition Heads
- **分类: cs.CL**

- **简介: 该论文研究多语言模型中注意力头的作用，旨在解决跨语言生成问题。通过识别检索-转换头（RTH），提升链式推理能力。**

- **链接: [https://arxiv.org/pdf/2602.22453](https://arxiv.org/pdf/2602.22453)**

> **作者:** Shaswat Patel; Vishvesh Trivedi; Yue Han; Yihuai Hong; Eunsol Choi
>
> **备注:** In the paper, there are still many statements that are unclear and lack sufficient justification. Since it is difficult for us to estimate how much time would be required to properly revise and correct these issues, we would like to request a withdrawal of the paper in this moment. Thank you!
>
> **摘要:** Recent work has identified a subset of attention heads in Transformer as retrieval heads, which are responsible for retrieving information from the context. In this work, we first investigate retrieval heads in multilingual contexts. In multilingual language models, we find that retrieval heads are often shared across multiple languages. Expanding the study to cross-lingual setting, we identify Retrieval-Transition heads(RTH), which govern the transition to specific target-language output. Our experiments reveal that RTHs are distinct from retrieval heads and more vital for Chain-of-Thought reasoning in multilingual LLMs. Across four multilingual benchmarks (MMLU-ProX, MGSM, MLQA, and XQuaD) and two model families (Qwen-2.5 and Llama-3.1), we demonstrate that masking RTH induces bigger performance drop than masking Retrieval Heads (RH). Our work advances understanding of multilingual LMs by isolating the attention heads responsible for mapping to target languages.
>
---
#### [replaced 042] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SRL框架，解决小模型多步推理难题。通过生成逻辑动作序列和内部思考过程，提升模型推理能力与泛化性。**

- **链接: [https://arxiv.org/pdf/2510.25992](https://arxiv.org/pdf/2510.25992)**

> **作者:** Yihe Deng; I-Hung Hsu; Jun Yan; Zifeng Wang; Rujun Han; Gufeng Zhang; Yanfei Chen; Wei Wang; Tomas Pfister; Chen-Yu Lee
>
> **备注:** Paper accepted by ICLR 2026. The first two authors contribute equally
>
> **摘要:** Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.
>
---
#### [replaced 043] Unraveling Syntax: How Language Models Learn Context-Free Grammars
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文研究语言模型如何学习上下文无关文法（CFG）的子结构。任务是理解模型在处理CFG时的行为。工作包括理论分析与实验验证，发现模型能并行学习子结构，但深度递归仍有困难。**

- **链接: [https://arxiv.org/pdf/2510.02524](https://arxiv.org/pdf/2510.02524)**

> **作者:** Laura Ying Schulz; Daniel Mitropolsky; Tomaso Poggio
>
> **备注:** Equal contribution by LYS and DM
>
> **摘要:** While large models achieve impressive results, their learning dynamics are far from understood. Many domains of interest, such as natural language syntax, coding languages, arithmetic problems, are captured by context-free grammars (CFGs). In this work, we extend prior work on neural language modeling of CFGs in a novel direction: how language modeling behaves with respect to CFG substructure, namely "subgrammars". We first define subgrammars, and prove a set of fundamental theorems regarding language modeling and subgrammars. We show that language modeling loss (or equivalently the Kullback-Leibler divergence) recurses linearly over its top-level subgrammars; applied recursively, the loss decomposes into losses for "irreducible" subgrammars. We also prove that the constant in this linear recurrence is a function of the expected recursion, a notion we introduce. We show that under additional assumptions, parametrized models learn subgrammars in parallel. Empirically, we confirm that small transformers learn subgrammars in parallel, unlike children, who first master simple substructures. We also briefly explore several other questions regarding subgrammars. We find that subgrammar pretraining can improve final performance, but only for tiny models relative to the grammar, while alignment analyses show that pretraining consistently lead to internal representations that better reflect the grammar's substructure in all cases; we also observe persistent difficulty with deeper recursion, a limitation that appears even of large language models.
>
---
#### [replaced 044] Beyond Accuracy: Risk-Sensitive Evaluation of Hallucinated Medical Advice
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决 hallucination 评估中忽略风险的问题。提出一种基于风险的语言评估框架，以更准确衡量生成内容的潜在危害。**

- **链接: [https://arxiv.org/pdf/2602.07319](https://arxiv.org/pdf/2602.07319)**

> **作者:** Savan Doshi
>
> **摘要:** Large language models are increasingly being used in patient-facing medical question answering, where hallucinated outputs can vary widely in potential harm. However, existing hallucination standards and evaluation metrics focus primarily on factual correctness, treating all errors as equally severe. This obscures clinically relevant failure modes, particularly when models generate unsupported but actionable medical language. We propose a risk-sensitive evaluation framework that quantifies hallucinations through the presence of risk-bearing language, including treatment directives, contraindications, urgency cues, and mentions of high-risk medications. Rather than assessing clinical correctness, our approach evaluates the potential impact of hallucinated content if acted upon. We further combine risk scoring with a relevance measure to identify high-risk, low-grounding failures. We apply this framework to three instruction-tuned language models using controlled patient-facing prompts designed as safety stress tests. Our results show that models with similar surface-level behavior exhibit substantially different risk profiles and that standard evaluation metrics fail to capture these distinctions. These findings highlight the importance of incorporating risk sensitivity into hallucination evaluation and suggest that evaluation validity is critically dependent on task and prompt design.
>
---
#### [replaced 045] What Makes a Reward Model a Good Teacher? An Optimization Perspective
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于强化学习任务，研究如何评估奖励模型的有效性。解决的问题是：仅靠准确性是否足够？工作表明奖励模型需具备足够方差以促进优化。**

- **链接: [https://arxiv.org/pdf/2503.15477](https://arxiv.org/pdf/2503.15477)**

> **作者:** Noam Razin; Zixuan Wang; Hubert Strauss; Stanley Wei; Jason D. Lee; Sanjeev Arora
>
> **备注:** Accepted to NeurIPS 2025; Code available at this https URL
>
> **摘要:** The success of Reinforcement Learning from Human Feedback (RLHF) critically depends on the quality of the reward model. However, while this quality is primarily evaluated through accuracy, it remains unclear whether accuracy fully captures what makes a reward model an effective teacher. We address this question from an optimization perspective. First, we prove that regardless of how accurate a reward model is, if it induces low reward variance, then the RLHF objective suffers from a flat landscape. Consequently, even a perfectly accurate reward model can lead to extremely slow optimization, underperforming less accurate models that induce higher reward variance. We additionally show that a reward model that works well for one language model can induce low reward variance, and thus a flat objective landscape, for another. These results establish a fundamental limitation of evaluating reward models solely based on accuracy or independently of the language model they guide. Experiments using models of up to 8B parameters corroborate our theory, demonstrating the interplay between reward variance, accuracy, and reward maximization rate. Overall, our findings highlight that beyond accuracy, a reward model needs to induce sufficient variance for efficient optimization.
>
---
#### [replaced 046] Error-Aware Knowledge Distillation via Targeted Revision for Customer-Service Summarization
- **分类: cs.CL**

- **简介: 该论文针对客户服务摘要任务，提出ARF管道，通过分析、修正和微调，使小型开源模型超越大模型，提升效果与隐私保护。**

- **链接: [https://arxiv.org/pdf/2511.03005](https://arxiv.org/pdf/2511.03005)**

> **作者:** Hee-Jin Lee; Zhen Guo; Luchao Jin; Morteza Moazami Goudarzi
>
> **摘要:** We introduce an Analyze-Revise-Finetune (ARF) pipeline that enables smaller open-source language models (LLMs) to surpass substantially larger proprietary models in customer service summarization tasks. The pipeline first analyzes and categorizes common errors in summaries produced by a teacher model (GPT-3.5), then performs a targeted revision using a compact editor model (Llama 3.1 70B) to generate high-quality, refined training data. Fine-tuning smaller student models (e.g., Llama 3.1 8B, QWen3 4B) on this refined data resulted in superior summarization performance compared to GPT-3.5. The ARF pipeline improves cost efficiency and data privacy while maintaining competitive accuracy, illustrating a generalizable framework for enhancing open-source LLMs across diverse downstream applications.
>
---
#### [replaced 047] Forecasting Future Language: Context Design for Mention Markets
- **分类: q-fin.GN; cs.CL; cs.LG**

- **简介: 该论文属于预测任务，旨在提升提及市场中关键词出现的预测准确性。通过设计上下文和引入市场概率，改进语言模型的预测效果。**

- **链接: [https://arxiv.org/pdf/2602.21229](https://arxiv.org/pdf/2602.21229)**

> **作者:** Sumin Kim; Jihoon Kwon; Yoon Kim; Nicole Kagan; Raffi Khatchadourian; Wonbin Ahn; Alejandro Lopez-Lira; Jaewon Lee; Yoontae Hwang; Oscar Levy; Yongjae Lee; Chanyeol Choi
>
> **备注:** 10 pages
>
> **摘要:** Mention markets, a type of prediction market in which contracts resolve based on whether a specified keyword is mentioned during a future public event, require accurate probabilistic forecasts of keyword-mention outcomes. While recent work shows that large language models (LLMs) can generate forecasts competitive with human forecasters, it remains unclear how input context should be designed to support accurate prediction. In this paper, we study this question through experiments on earnings-call mention markets, which require forecasting whether a company will mention a specified keyword during its upcoming call. We run controlled comparisons varying (i) which contextual information is provided (news and/or prior earnings-call transcripts) and (ii) how \textit{market probability}, (i.e., prediction market contract price) is used. We introduce Market-Conditioned Prompting (MCP), which explicitly treats the market-implied probability as a prior and instructs the LLM to update this prior using textual evidence, rather than re-predicting the base rate from scratch. In our experiments, we find three insights: (1) richer context consistently improves forecasting performance; (2) market-conditioned prompting (MCP), which treats the market probability as a prior and updates it using textual evidence, yields better-calibrated forecasts; and (3) a mixture of the market probability and MCP (MixMCP) outperforms the market baseline. By dampening the LLM's posterior update with the market prior, MixMCP yields more robust predictions than either the market or the LLM alone.
>
---
#### [replaced 048] Janus-Q: End-to-End Event-Driven Trading via Hierarchical-Gated Reward Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于金融交易任务，旨在解决事件驱动交易中数据不足和模型对齐问题。构建了大规模事件数据集，并提出Janus-Q框架，结合强化学习提升交易决策效果。**

- **链接: [https://arxiv.org/pdf/2602.19919](https://arxiv.org/pdf/2602.19919)**

> **作者:** Xiang Li; Zikai Wei; Yiyan Qi; Wanyun Zhou; Xiang Liu; Penglei Sun; Jian Guo; Yongqi Zhang; Xiaowen Chu
>
> **摘要:** Financial market movements are often driven by discrete financial events conveyed through news, whose impacts are heterogeneous, abrupt, and difficult to capture under purely numerical prediction objectives. These limitations have motivated growing interest in using textual information as the primary source of trading signals in learning-based systems. Two key challenges hinder existing approaches: (1) the absence of large-scale, event-centric datasets that jointly model news semantics and statistically grounded market reactions, and (2) the misalignment between language model reasoning and financially valid trading behavior under dynamic market conditions. To address these challenges, we propose Janus-Q, an end-to-end event-driven trading framework that elevates financial news events from auxiliary signals to primary decision units. Janus-Q unifies event-centric data construction and model optimization under a two-stage paradigm. Stage I focuses on event-centric data construction, building a large-scale financial news event dataset comprising 62,400 articles annotated with 10 fine-grained event types, associated stocks, sentiment labels, and event-driven cumulative abnormal return (CAR). Stage II performs decision-oriented fine-tuning, combining supervised learning with reinforcement learning guided by a Hierarchical Gated Reward Model (HGRM), which explicitly captures trade-offs among multiple trading objectives. Extensive experiments demonstrate that Janus-Q achieves more consistent, interpretable, and profitable trading decisions than market indices and LLM baselines, improving the Sharpe Ratio by up to 102.0% while increasing direction accuracy by over 17.5% compared to the strongest competing strategies.
>
---
#### [replaced 049] FeynTune: Large Language Models for High-Energy Theory
- **分类: cs.CL; cs.LG; hep-th**

- **简介: 论文提出针对高能物理的专用大语言模型，解决领域知识融合问题，通过微调不同数据集提升性能。**

- **链接: [https://arxiv.org/pdf/2508.03716](https://arxiv.org/pdf/2508.03716)**

> **作者:** Paul Richmond; Prarit Agarwal; Borun Chowdhury; Vasilis Niarchos; Constantinos Papageorgakis
>
> **备注:** 16 pages; v2: Human evaluation discussion updated, additional training hyperparameters and inference settings included and references added
>
> **摘要:** We present specialized Large Language Models for theoretical High-Energy Physics, obtained as 20 fine-tuned variants of the 8-billion parameter Llama-3.1 model. Each variant was trained on arXiv abstracts (through August 2024) from different combinations of hep-th, hep-ph and gr-qc. For a comparative study, we also trained models on datasets that contained abstracts from disparate fields such as the q-bio and cs categories. All models were fine-tuned using two distinct Low-Rank Adaptation fine-tuning approaches and varying dataset sizes, and outperformed the base model on hep-th abstract completion tasks. We compare performance against leading commercial LLMs (ChatGPT, Claude, Gemini, DeepSeek) and derive insights for further developing specialized language models for High-Energy Theoretical Physics.
>
---
#### [replaced 050] Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型对齐任务，解决数据生成与偏好优化间的分布不匹配问题。提出MetaAPO框架，动态耦合数据生成与训练，提升效果并降低标注成本。**

- **链接: [https://arxiv.org/pdf/2509.23371](https://arxiv.org/pdf/2509.23371)**

> **作者:** Junming Yang; Ning Xu; Biao Liu; Shiqi Qiao; Xin Geng
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Preference optimization is crucial for aligning large language models (LLMs) with human values and intentions. A significant challenge in this process is the distribution mismatch between pre-collected offline preference data and the evolving model policy. Existing methods attempt to reduce this gap using static heuristics or decoupled online sampling strategies, but they often fail to adapt to the model's dynamic learning state. To bridge this gap, we propose Meta-Weighted Adaptive Preference Optimization (MetaAPO), a novel framework that dynamically couples data generation with model training. MetaAPO employs a lightweight meta-learner, as an "alignment gap estimator", to evaluate the potential benefits of on-policy sampling in relation to offline data. This guides targeted online generation and assigns sample-wise meta-weights to the optimization objective, dynamically balancing the quality and distribution of online and offline data. Experiments on AlpacaEval 2, Arena-Hard and MT-Bench demonstrate that MetaAPO consistently outperforms existing preference optimization approaches across various settings, while reducing 42% in online annotation costs. Code is available at this https URL.
>
---
#### [replaced 051] Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的答案一致性选择任务，解决LLM输出不一致的问题。提出LSC方法，通过语义一致选择最优回答，提升短长答案的可靠性。**

- **链接: [https://arxiv.org/pdf/2508.18395](https://arxiv.org/pdf/2508.18395)**

> **作者:** Jungsuk Oh; Jay-Yoon Lee
>
> **摘要:** Probabilistic decoding in Large Language Models (LLMs) often yields inconsistent outputs, particularly on complex or long-form questions. Self-Consistency (SC) mitigates this for short-form QA by majority voting over exact strings, whereas Universal Self-Consistency (USC) and Weighted Unigram Consistency Score (WUCS) extend to long-form responses but lose accuracy on short-form benchmarks. We introduce \textbf{Latent Self-Consistency (LSC)}, which selects the most semantically consistent response using learnable token embeddings. LSC's lightweight forward processing of summary tokens only introduces negligible runtime overhead (at most $0.9\%$) on top of standard decoding of the base LLM, and requires no changes to the model architecture. Across 6 short-form and 5 long-form reasoning benchmarks (e.g., MATH, MMLU, TruthfulQA), LSC surpasses SC, USC, and WUCS on both short-form and long-form on average performance, while adding negligible computational overhead on vanilla inference. These results position LSC as a reliable consistency-selection method that works effectively across various answer formats. Additionally, LSC provides well-calibrated confidence estimates, maintaining low expected calibration error across both answer formats.
>
---
#### [replaced 052] FineScope : SAE-guided Data Selection Enables Domain Specific LLM Pruning and Finetuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FineScope框架，解决领域特定大模型优化问题，通过SAE数据选择与剪枝微调，提升模型效率和性能。**

- **链接: [https://arxiv.org/pdf/2505.00624](https://arxiv.org/pdf/2505.00624)**

> **作者:** Chaitali Bhattacharyya; Hyunsei Lee; Junyoung Lee; Shinhyoung Jang; Il hong Suh; Yeseong Kim
>
> **摘要:** Training large language models (LLMs) from scratch requires significant computational resources, driving interest in developing smaller, domain-specific LLMs that maintain both efficiency and strong task performance. Medium-sized models such as LLaMA, llama} have served as starting points for domain-specific adaptation, but they often suffer from accuracy degradation when tested on specialized datasets. We introduce FineScope, a framework for deriving compact, domain-optimized LLMs from larger pretrained models. FineScope leverages the Sparse Autoencoder (SAE) framework, inspired by its ability to produce interpretable feature representations, to extract domain-specific subsets from large datasets. We apply structured pruning with domain-specific constraints, ensuring that the resulting pruned models retain essential knowledge for the target domain. To further enhance performance, these pruned models undergo self-data distillation, leveraging SAE-curated datasets to restore key domain-specific information lost during pruning. Extensive experiments and ablation studies demonstrate that FineScope achieves highly competitive performance, outperforming several large-scale state-of-the-art LLMs in domain-specific tasks. Additionally, our results show that FineScope enables pruned models to regain a substantial portion of their original performance when fine-tuned with SAE-curated datasets. Furthermore, applying these datasets to fine-tune pretrained LLMs without pruning also improves their domain-specific accuracy, highlighting the robustness of our approach.
>
---
#### [replaced 053] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?
- **分类: cs.CV; cs.CE; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在解决VLM在真实场景中空间推理能力不足的问题。作者构建了SpatiaLab基准，涵盖多种空间关系任务，评估不同模型表现，揭示其与人类的差距。**

- **链接: [https://arxiv.org/pdf/2602.03916](https://arxiv.org/pdf/2602.03916)**

> **作者:** Azmine Toushik Wasi; Wahid Faisal; Abdur Rahman; Mahfuz Ahmed Anik; Munem Shahriar; Mohsin Mahmud Topu; Sadia Tasnim Meem; Rahatun Nesa Priti; Sabrina Afroz Mitu; Md. Iqramul Hoque; Shahriyar Zaman Ridoy; Mohammed Eunus Ali; Majd Hawasly; Mohammad Raza; Md Rizwan Parvez
>
> **备注:** Accepted to ICLR 2026 (this https URL). 92 Pages. 42 Figures and 29 Tables
>
> **摘要:** Spatial reasoning is a fundamental aspect of human cognition, yet it remains a major challenge for contemporary vision-language models (VLMs). Prior work largely relied on synthetic or LLM-generated environments with limited task designs and puzzle-like setups, failing to capture the real-world complexity, visual noise, and diverse spatial relationships that VLMs encounter. To address this, we introduce SpatiaLab, a comprehensive benchmark for evaluating VLMs' spatial reasoning in realistic, unconstrained contexts. SpatiaLab comprises 1,400 visual question-answer pairs across six major categories: Relative Positioning, Depth & Occlusion, Orientation, Size & Scale, Spatial Navigation, and 3D Geometry, each with five subcategories, yielding 30 distinct task types. Each subcategory contains at least 25 questions, and each main category includes at least 200 questions, supporting both multiple-choice and open-ended evaluation. Experiments across diverse state-of-the-art VLMs, including open- and closed-source models, reasoning-focused, and specialized spatial reasoning models, reveal a substantial gap in spatial reasoning capabilities compared with humans. In the multiple-choice setup, InternVL3.5-72B achieves 54.93% accuracy versus 87.57% for humans. In the open-ended setting, all models show a performance drop of around 10-25%, with GPT-5-mini scoring highest at 40.93% versus 64.93% for humans. These results highlight key limitations in handling complex spatial relationships, depth perception, navigation, and 3D geometry. By providing a diverse, real-world evaluation framework, SpatiaLab exposes critical challenges and opportunities for advancing VLMs' spatial reasoning, offering a benchmark to guide future research toward robust, human-aligned spatial understanding. SpatiaLab is available at: this https URL.
>
---
#### [replaced 054] MoDora: Tree-Based Semi-Structured Document Analysis System
- **分类: cs.IR; cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文提出MoDora系统，解决半结构化文档分析问题，通过布局感知组件和层次结构建模，提升问答准确率。**

- **链接: [https://arxiv.org/pdf/2602.23061](https://arxiv.org/pdf/2602.23061)**

> **作者:** Bangrui Xu; Qihang Yao; Zirui Tang; Xuanhe Zhou; Yeye He; Shihan Yu; Qianqian Xu; Bin Wang; Guoliang Li; Conghui He; Fan Wu
>
> **备注:** Extension of our SIGMOD 2026 paper. Please refer to source code available at this https URL
>
> **摘要:** Semi-structured documents integrate diverse interleaved data elements (e.g., tables, charts, hierarchical paragraphs) arranged in various and often irregular layouts. These documents are widely observed across domains and account for a large portion of real-world data. However, existing methods struggle to support natural language question answering over these documents due to three main technical challenges: (1) The elements extracted by techniques like OCR are often fragmented and stripped of their original semantic context, making them inadequate for analysis. (2) Existing approaches lack effective representations to capture hierarchical structures within documents (e.g., associating tables with nested chapter titles) and to preserve layout-specific distinctions (e.g., differentiating sidebars from main content). (3) Answering questions often requires retrieving and aligning relevant information scattered across multiple regions or pages, such as linking a descriptive paragraph to table cells located elsewhere in the document. To address these issues, we propose MoDora, an LLM-powered system for semi-structured document analysis. First, we adopt a local-alignment aggregation strategy to convert OCR-parsed elements into layout-aware components, and conduct type-specific information extraction for components with hierarchical titles or non-text elements. Second, we design the Component-Correlation Tree (CCTree) to hierarchically organize components, explicitly modeling inter-component relations and layout distinctions through a bottom-up cascade summarization process. Finally, we propose a question-type-aware retrieval strategy that supports (1) layout-based grid partitioning for location-based retrieval and (2) LLM-guided pruning for semantic-based retrieval. Experiments show MoDora outperforms baselines by 5.97%-61.07% in accuracy. The code is at this https URL.
>
---
#### [replaced 055] Moral Susceptibility and Robustness under Persona Role-Play in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究大语言模型在角色扮演中的道德表现，分析其道德敏感性和稳定性，旨在理解 persona 条件如何影响模型的道德行为。**

- **链接: [https://arxiv.org/pdf/2511.08565](https://arxiv.org/pdf/2511.08565)**

> **作者:** Davi Bastos Costa; Felippe Alves; Renato Vicente
>
> **备注:** 8+5 pages, 7 tables, 7 figures
>
> **摘要:** Large language models (LLMs) increasingly operate in social contexts, motivating analysis of how they express and shift moral judgments. In this work, we investigate the moral response of LLMs to persona role-play, prompting a LLM to assume a specific character. Using the Moral Foundations Questionnaire (MFQ), we introduce a benchmark that quantifies two properties: moral susceptibility and moral robustness, defined from the variability of MFQ scores across and within personas, respectively. We find that, for moral robustness, model family accounts for most of the variance, while model size shows no systematic effect. The Claude family is, by a significant margin, the most robust, followed by Gemini and GPT-4 models, with other families exhibiting lower robustness. In contrast, moral susceptibility exhibits a mild family effect but a clear within-family size effect, with larger variants being more susceptible. Moreover, robustness and susceptibility are positively correlated, an association that is more pronounced at the family level. Additionally, we present moral foundation profiles for models without persona role-play and for personas averaged across models. Together, these analyses provide a systematic view of how persona conditioning shapes moral behavior in LLMs.
>
---
#### [replaced 056] Understanding In-Context Learning Beyond Transformers: An Investigation of State Space and Hybrid Architectures
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究不同架构模型的上下文学习能力，分析其内部机制差异。旨在理解ICL在不同模型中的实现方式，通过行为探测与干预方法，揭示功能向量的作用及模型差异。**

- **链接: [https://arxiv.org/pdf/2510.23006](https://arxiv.org/pdf/2510.23006)**

> **作者:** Shenran Wang; Timothy Tin-Long Tse; Jian Zhu
>
> **摘要:** We perform in-depth evaluations of in-context learning (ICL) on state-of-the-art transformer, state-space, and hybrid large language models over two categories of knowledge-based ICL tasks. Using a combination of behavioral probing and intervention-based methods, we have discovered that, while LLMs of different architectures can behave similarly in task performance, their internals could remain different. We discover that function vectors (FVs) responsible for ICL are primarily located in the self-attention and Mamba layers, and speculate that Mamba2 uses a different mechanism from FVs to perform ICL. FVs are more important for ICL involving parametric knowledge retrieval, but not for contextual knowledge understanding. Our work contributes to a more nuanced understanding across architectures and task types. Methodologically, our approach also highlights the importance of combining both behavioural and mechanistic analyses to investigate LLM capabilities.
>
---
