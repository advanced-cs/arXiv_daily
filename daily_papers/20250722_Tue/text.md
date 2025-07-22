# 自然语言处理 cs.CL

- **最新发布 117 篇**

- **更新 81 篇**

## 最新发布

#### [new 001] ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出ASPERA框架，用于评估大语言模型在复杂动作执行中的规划能力。任务是生成基于自定义助手库的程序，解决数据可用性和评估鲁棒性问题。工作包括构建模拟环境、生成高质量任务数据，并发布包含250个任务的Asper-Bench数据集，表明该任务对大语言模型具有挑战性。**

- **链接: [http://arxiv.org/pdf/2507.15501v1](http://arxiv.org/pdf/2507.15501v1)**

> **作者:** Alexandru Coca; Mark Gaynor; Zhenxing Zhang; Jianpeng Cheng; Bo-Hsiang Tseng; Pete Boothroyd; Héctor Martinez Alonso; Diarmuid Ó Séaghdha; Anders Johannsen
>
> **备注:** 37 pages, 22 figures. To appear at ACL 2025
>
> **摘要:** This work evaluates the potential of large language models (LLMs) to power digital assistants capable of complex action execution. These assistants rely on pre-trained programming knowledge to execute multi-step goals by composing objects and functions defined in assistant libraries into action execution programs. To achieve this, we develop ASPERA, a framework comprising an assistant library simulation and a human-assisted LLM data generation engine. Our engine allows developers to guide LLM generation of high-quality tasks consisting of complex user queries, simulation state and corresponding validation programs, tackling data availability and evaluation robustness challenges. Alongside the framework we release Asper-Bench, an evaluation dataset of 250 challenging tasks generated using ASPERA, which we use to show that program generation grounded in custom assistant libraries is a significant challenge to LLMs compared to dependency-free code generation.
>
---
#### [new 002] HuggingGraph: Understanding the Supply Chain of LLM Ecosystem
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图分析任务，旨在理解大语言模型（LLM）生态系统中的模型与数据集之间的供应关系。为解决潜在风险、提高模型公平性，论文构建了一个包含397,376节点和453,469边的异构图，并进行分析，揭示了供应网络的结构特性与动态演化。**

- **链接: [http://arxiv.org/pdf/2507.14240v1](http://arxiv.org/pdf/2507.14240v1)**

> **作者:** Mohammad Shahedur Rahman; Peng Gao; Yuede Ji
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Large language models (LLMs) leverage deep learning to process and predict sequences of words from context, enabling them to perform various NLP tasks, such as translation, summarization, question answering, and content generation. However, the growing size and complexity of developing, training, and deploying advanced LLMs require extensive computational resources and large datasets. This creates a barrier for users. As a result, platforms that host models and datasets are widely used. For example, Hugging Face, one of the most popular platforms, hosted 1.8 million models and 450K datasets by June 2025, with no sign of slowing down. Since many LLMs are built from base models, pre-trained models, and external datasets, they can inherit vulnerabilities, biases, or malicious components from earlier models or datasets. Therefore, it is critical to understand the origin and development of these components to better detect potential risks, improve model fairness, and ensure compliance. Motivated by this, our project aims to study the relationships between models and datasets, which are core components of the LLM supply chain. First, we design a method to systematically collect LLM supply chain data. Using this data, we build a directed heterogeneous graph to model the relationships between models and datasets, resulting in a structure with 397,376 nodes and 453,469 edges. We then perform various analyses and uncover several findings, such as: (i) the LLM supply chain graph is large, sparse, and follows a power-law degree distribution; (ii) it features a densely connected core and a fragmented periphery; (iii) datasets play pivotal roles in training; (iv) strong interdependence exists between models and datasets; and (v) the graph is dynamic, with daily updates reflecting the ecosystem's ongoing evolution.
>
---
#### [new 003] Beyond Easy Wins: A Text Hardness-Aware Benchmark for LLM-generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有检测系统在实际应用中忽略误报率、阈值设定和稳定性的问题。作者提出了SHIELD基准，结合可靠性和稳定性，评估检测系统在多种场景下的表现，并开发了可调节难度的人类化文本生成框架，挑战现有检测方法的效果。**

- **链接: [http://arxiv.org/pdf/2507.15286v1](http://arxiv.org/pdf/2507.15286v1)**

> **作者:** Navid Ayoobi; Sadat Shahriar; Arjun Mukherjee
>
> **摘要:** We present a novel evaluation paradigm for AI text detectors that prioritizes real-world and equitable assessment. Current approaches predominantly report conventional metrics like AUROC, overlooking that even modest false positive rates constitute a critical impediment to practical deployment of detection systems. Furthermore, real-world deployment necessitates predetermined threshold configuration, making detector stability (i.e. the maintenance of consistent performance across diverse domains and adversarial scenarios), a critical factor. These aspects have been largely ignored in previous research and benchmarks. Our benchmark, SHIELD, addresses these limitations by integrating both reliability and stability factors into a unified evaluation metric designed for practical assessment. Furthermore, we develop a post-hoc, model-agnostic humanification framework that modifies AI text to more closely resemble human authorship, incorporating a controllable hardness parameter. This hardness-aware approach effectively challenges current SOTA zero-shot detection methods in maintaining both reliability and stability. (Data and code: https://github.com/navid-aub/SHIELD-Benchmark)
>
---
#### [new 004] LionGuard 2: Building Lightweight, Data-Efficient & Localised Multilingual Content Moderators
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言内容审核任务，旨在解决低资源语言和本地化内容审核效果差的问题。论文提出了LionGuard 2，一个轻量级、数据高效的多语言审核分类器，专为新加坡环境设计，支持英、中、马来语和部分泰米尔语，使用预训练嵌入和多头分类器，在17个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.15339v1](http://arxiv.org/pdf/2507.15339v1)**

> **作者:** Leanne Tan; Gabriel Chua; Ziyu Ge; Roy Ka-Wei Lee
>
> **摘要:** Modern moderation systems increasingly support multiple languages, but often fail to address localisation and low-resource variants - creating safety gaps in real-world deployments. Small models offer a potential alternative to large LLMs, yet still demand considerable data and compute. We present LionGuard 2, a lightweight, multilingual moderation classifier tailored to the Singapore context, supporting English, Chinese, Malay, and partial Tamil. Built on pre-trained OpenAI embeddings and a multi-head ordinal classifier, LionGuard 2 outperforms several commercial and open-source systems across 17 benchmarks, including both Singapore-specific and public English datasets. The system is actively deployed within the Singapore Government, demonstrating practical efficacy at scale. Our findings show that high-quality local data and robust multilingual embeddings can achieve strong moderation performance, without fine-tuning large models. We release our model weights and part of our training data to support future work on LLM safety.
>
---
#### [new 005] Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型提示工程手动、不一致的问题。论文提出了Promptomatix框架，自动优化提示，无需人工调整或领域知识。**

- **链接: [http://arxiv.org/pdf/2507.14241v1](http://arxiv.org/pdf/2507.14241v1)**

> **作者:** Rithesh Murthy; Ming Zhu; Liangwei Yang; Jielin Qiu; Juntao Tan; Shelby Heinecke; Huan Wang; Caiming Xiong; Silvio Savarese
>
> **摘要:** Large Language Models (LLMs) perform best with well-crafted prompts, yet prompt engineering remains manual, inconsistent, and inaccessible to non-experts. We introduce Promptomatix, an automatic prompt optimization framework that transforms natural language task descriptions into high-quality prompts without requiring manual tuning or domain expertise. Promptomatix supports both a lightweight meta-prompt-based optimizer and a DSPy-powered compiler, with modular design enabling future extension to more advanced frameworks. The system analyzes user intent, generates synthetic training data, selects prompting strategies, and refines prompts using cost-aware objectives. Evaluated across 5 task categories, Promptomatix achieves competitive or superior performance compared to existing libraries, while reducing prompt length and computational overhead making prompt optimization scalable and efficient.
>
---
#### [new 006] A Fisher's exact test justification of the TF-IDF term-weighting scheme
- **分类: cs.CL; cs.IR; math.ST; stat.TH**

- **简介: 该论文属于信息检索与统计学习任务，旨在为TF-IDF词权重机制建立统计理论依据。论文提出TF-IDF可通过Fisher精确检验的负对数p值进行解释，证明TF-ICF与该统计量在常规条件下密切相关，并在理想假设下建立TF-IDF与该检验的联系，最终在大规模文档集中验证其收敛性。**

- **链接: [http://arxiv.org/pdf/2507.15742v1](http://arxiv.org/pdf/2507.15742v1)**

> **作者:** Paul Sheridan; Zeyad Ahmed; Aitazaz A. Farooque
>
> **备注:** 23 pages, 4 tables
>
> **摘要:** Term frequency-inverse document frequency, or TF-IDF for short, is arguably the most celebrated mathematical expression in the history of information retrieval. Conceived as a simple heuristic quantifying the extent to which a given term's occurrences are concentrated in any one given document out of many, TF-IDF and its many variants are routinely used as term-weighting schemes in diverse text analysis applications. There is a growing body of scholarship dedicated to placing TF-IDF on a sound theoretical foundation. Building on that tradition, this paper justifies the use of TF-IDF to the statistics community by demonstrating how the famed expression can be understood from a significance testing perspective. We show that the common TF-IDF variant TF-ICF is, under mild regularity conditions, closely related to the negative logarithm of the $p$-value from a one-tailed version of Fisher's exact test of statistical significance. As a corollary, we establish a connection between TF-IDF and the said negative log-transformed $p$-value under certain idealized assumptions. We further demonstrate, as a limiting case, that this same quantity converges to TF-IDF in the limit of an infinitely large document collection. The Fisher's exact test justification of TF-IDF equips the working statistician with a ready explanation of the term-weighting scheme's long-established effectiveness.
>
---
#### [new 007] Exploring Human-AI Complementarity in CPS Diagnosis Using Unimodal and Multimodal BERT Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能教育领域中的协作问题解决（CPS）诊断任务，旨在提升从对话中检测CPS指标的准确性。论文比较了单模态BERT与多模态AudiBERT模型的表现，发现AudiBERT在社会认知维度分类中优于BERT，尤其在稀有类别和数据量大的情况下。论文进一步探讨了人机协同的路径，强调模型可解释性的重要性。**

- **链接: [http://arxiv.org/pdf/2507.14579v1](http://arxiv.org/pdf/2507.14579v1)**

> **作者:** Kester Wong; Sahan Bulathwela; Mutlu Cukurova
>
> **备注:** Accepted to appear in the workshop proceedings for the HEXED'25 workshop in the 26th International Conference on Artificial Intelligence in Education 2025 (AIED 2025), 22 July 2025, Palermo, Italy. 5 pages
>
> **摘要:** Detecting collaborative problem solving (CPS) indicators from dialogue using machine learning techniques is a significant challenge for the field of AI in Education. Recent studies have explored the use of Bidirectional Encoder Representations from Transformers (BERT) models on transcription data to reliably detect meaningful CPS indicators. A notable advancement involved the multimodal BERT variant, AudiBERT, which integrates speech and acoustic-prosodic audio features to enhance CPS diagnosis. Although initial results demonstrated multimodal improvements, the statistical significance of these enhancements remained unclear, and there was insufficient guidance on leveraging human-AI complementarity for CPS diagnosis tasks. This workshop paper extends the previous research by highlighting that the AudiBERT model not only improved the classification of classes that were sparse in the dataset, but it also had statistically significant class-wise improvements over the BERT model for classifications in the social-cognitive dimension. However, similar significant class-wise improvements over the BERT model were not observed for classifications in the affective dimension. A correlation analysis highlighted that larger training data was significantly associated with higher recall performance for both the AudiBERT and BERT models. Additionally, the precision of the BERT model was significantly associated with high inter-rater agreement among human coders. When employing the BERT model to diagnose indicators within these subskills that were well-detected by the AudiBERT model, the performance across all indicators was inconsistent. We conclude the paper by outlining a structured approach towards achieving human-AI complementarity for CPS diagnosis, highlighting the crucial inclusion of model explainability to support human agency and engagement in the reflective coding process.
>
---
#### [new 008] Beyond Isolated Capabilities: Bridging Long CoT Reasoning and Long-Context Understanding
- **分类: cs.CL**

- **简介: 该论文研究推理蒸馏对长上下文理解的影响，旨在解决长文本中信息提取与整合的问题。通过多文档问答任务评估蒸馏模型的表现，发现其能显著提升长上下文理解，缓解“中间信息丢失”问题。**

- **链接: [http://arxiv.org/pdf/2507.14849v1](http://arxiv.org/pdf/2507.14849v1)**

> **作者:** Yifei Wang
>
> **摘要:** Reasoning distillation has emerged as an effective approach to enhance the reasoning capabilities of smaller language models. However, the impact of large-scale reasoning distillation on other critical abilities, particularly in-context retrieval and reasoning, remains unexplored. This gap in understanding is particularly significant given the increasing importance of Retrieval-Augmented Generation (RAG) systems, where efficient acquisition and utilization of contextual information are paramount for generating reliable responses. Motivated by the need to understand how the extended long-CoT process influences long-context comprehension, we conduct a comprehensive investigation using a series of open-source models distilled from Deepseek-R1, renowned for its exceptional reasoning capabilities. Our study focuses on evaluating these models' performance in extracting and integrating relevant information from extended contexts through multi-document question and answering tasks. Through rigorous experimentation, we demonstrate that distilled reasoning patterns significantly improve long-context understanding. Our analysis reveals that distillation fosters greater long-context awareness by promoting more detailed and explicit reasoning processes during context analysis and information parsing. This advancement effectively mitigates the persistent "lost in the middle" issue that has hindered long-context models.
>
---
#### [new 009] FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing
- **分类: cs.CL**

- **简介: 该论文属于语音语言模型任务，旨在解决现有模型在长语音处理上的效率与效果不足。作者提出FastLongSpeech框架，通过动态压缩训练和迭代融合策略，提升模型对长语音的理解与处理能力，无需依赖专门的长语音数据集。**

- **链接: [http://arxiv.org/pdf/2507.14815v1](http://arxiv.org/pdf/2507.14815v1)**

> **作者:** Shoutao Guo; Shaolei Zhang; Qingkai Fang; Zhengrui Ma; Min Zhang; Yang Feng
>
> **备注:** The code is at https://github.com/ictnlp/FastLongSpeech. This model is at https://huggingface.co/ICTNLP/FastLongSpeech. The dataset is at https://huggingface.co/datasets/ICTNLP/LongSpeech-Eval
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has spurred significant progress in Large Speech-Language Models (LSLMs), enhancing their capabilities in both speech understanding and generation. While existing LSLMs often concentrate on augmenting speech generation or tackling a diverse array of short-speech tasks, the efficient processing of long-form speech remains a critical yet underexplored challenge. This gap is primarily attributed to the scarcity of long-speech training datasets and the high computational costs associated with long sequences. To address these limitations, we introduce FastLongSpeech, a novel framework designed to extend LSLM capabilities for efficient long-speech processing without necessitating dedicated long-speech training data. FastLongSpeech incorporates an iterative fusion strategy that can compress excessively long-speech sequences into manageable lengths. To adapt LSLMs for long-speech inputs, it introduces a dynamic compression training approach, which exposes the model to short-speech sequences at varying compression ratios, thereby transferring the capabilities of LSLMs to long-speech tasks. To assess the long-speech capabilities of LSLMs, we develop a long-speech understanding benchmark called LongSpeech-Eval. Experiments show that our method exhibits strong performance in both long-speech and short-speech tasks, while greatly improving inference efficiency.
>
---
#### [new 010] Reservoir Computing as a Language Model
- **分类: cs.CL**

- **简介: 该论文探索将储层计算应用于自然文本处理，旨在解决大语言模型能耗高、速度慢的问题。对比了两种储层计算方法与Transformer架构在字符级语言建模中的性能、计算成本和预测准确性，发现储层计算更高效，而Transformer预测质量更优。**

- **链接: [http://arxiv.org/pdf/2507.15779v1](http://arxiv.org/pdf/2507.15779v1)**

> **作者:** Felix Köster; Atsushi Uchida
>
> **备注:** 8 pages, 5 figures, 1 table
>
> **摘要:** Large Language Models (LLM) have dominated the science and media landscape duo to their impressive performance on processing large chunks of data and produce human-like levels of text. Nevertheless, their huge energy demand and slow processing still a bottleneck for further increasing quality while also making the models accessible to everyone. To solve this bottleneck, we will investigate how reservoir computing performs on natural text processing, which could enable fast and energy efficient hardware implementations. Studies investigating the use of reservoir computing as a language model remain sparse. In this paper, we compare three distinct approaches for character-level language modeling, two different reservoir computing approaches, where only an output layer is trainable, and the well-known transformer-based architectures, which fully learn an attention-based sequence representation. We explore the performance, computational cost and prediction accuracy for both paradigms by equally varying the number of trainable parameters for all models. Using a consistent pipeline for all three approaches, we demonstrate that transformers excel in prediction quality, whereas reservoir computers remain highly efficient reducing the training and inference speed. Furthermore, we investigate two types of reservoir computing: a traditional reservoir with a static linear readout, and an attention-enhanced reservoir that dynamically adapts its output weights via an attention mechanism. Our findings underline how these paradigms scale and offer guidelines to balance resource constraints with performance.
>
---
#### [new 011] Explainable Collaborative Problem Solving Diagnosis with BERT using SHAP and its Implications for Teacher Adoption
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与教育技术交叉任务，旨在提升基于BERT的协作问题解决（CPS）分类模型的可解释性。论文使用SHAP分析词汇对分类决策的影响，发现高分类准确率不一定意味着合理解释，部分词汇使用频繁但语义无关。研究强调模型透明度的重要性，提醒教师不要过度依赖模型诊断，同时建议探索集成模型与人机协同方法以提升CPS诊断效果。**

- **链接: [http://arxiv.org/pdf/2507.14584v1](http://arxiv.org/pdf/2507.14584v1)**

> **作者:** Kester Wong; Sahan Bulathwela; Mutlu Cukurova
>
> **备注:** Accepted to appear in the workshop proceedings for the HEXED'25 workshop in the 26th International Conference on Artificial Intelligence in Education 2025 (AIED 2025), 22 July 2025, Palermo, Italy. 6 pages, 2 figures
>
> **摘要:** The use of Bidirectional Encoder Representations from Transformers (BERT) model and its variants for classifying collaborative problem solving (CPS) has been extensively explored within the AI in Education community. However, limited attention has been given to understanding how individual tokenised words in the dataset contribute to the model's classification decisions. Enhancing the explainability of BERT-based CPS diagnostics is essential to better inform end users such as teachers, thereby fostering greater trust and facilitating wider adoption in education. This study undertook a preliminary step towards model transparency and explainability by using SHapley Additive exPlanations (SHAP) to examine how different tokenised words in transcription data contributed to a BERT model's classification of CPS processes. The findings suggested that well-performing classifications did not necessarily equate to a reasonable explanation for the classification decisions. Particular tokenised words were used frequently to affect classifications. The analysis also identified a spurious word, which contributed positively to the classification but was not semantically meaningful to the class. While such model transparency is unlikely to be useful to an end user to improve their practice, it can help them not to overrely on LLM diagnostics and ignore their human expertise. We conclude the workshop paper by noting that the extent to which the model appropriately uses the tokens for its classification is associated with the number of classes involved. It calls for an investigation into the exploration of ensemble model architectures and the involvement of human-AI complementarity for CPS diagnosis, since considerable human reasoning is still required for fine-grained discrimination of CPS subskills.
>
---
#### [new 012] From Disagreement to Understanding: The Case for Ambiguity Detection in NLI
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理（NLI）任务，旨在解决标注分歧问题。它主张将分歧视为有意义的解释差异，特别是由前提或假设中的歧义引发的差异。论文提出了一个统一框架，用于识别歧义类型，并强调需要新标注资源和无监督方法来改进NLI系统。**

- **链接: [http://arxiv.org/pdf/2507.15114v1](http://arxiv.org/pdf/2507.15114v1)**

> **作者:** Chathuri Jayaweera; Bonnie Dorr
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This position paper argues that annotation disagreement in Natural Language Inference (NLI) is not mere noise but often reflects meaningful interpretive variation, especially when triggered by ambiguity in the premise or hypothesis. While underspecified guidelines and annotator behavior can contribute to variation, content-based ambiguity offers a process-independent signal of divergent human perspectives. We call for a shift toward ambiguity-aware NLI by systematically identifying ambiguous input pairs and classifying ambiguity types. To support this, we present a unified framework that integrates existing taxonomies and illustrate key ambiguity subtypes through concrete examples. These examples reveal how ambiguity shapes annotator decisions and motivate the need for targeted detection methods that better align models with human interpretation. A key limitation is the lack of datasets annotated for ambiguity and subtypes. We propose addressing this gap through new annotated resources and unsupervised approaches to ambiguity detection -- paving the way for more robust, explainable, and human-aligned NLI systems.
>
---
#### [new 013] Supernova: Achieving More with Less in Transformer Architectures
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出“Supernova”，一种仅650M参数的Transformer解码器模型，旨在通过优化架构设计和分词技术，在减少参数和训练数据的情况下实现接近更大模型的性能。使用RoPE、GQA、RMSNorm和SwiGLU等技术，并配备高效字节级BPE分词器，仅需100B训练token即可达到1B参数模型90%的性能，挑战了传统模型扩展范式。**

- **链接: [http://arxiv.org/pdf/2507.15773v1](http://arxiv.org/pdf/2507.15773v1)**

> **作者:** Andrei-Valentin Tanase; Elena Pelican
>
> **摘要:** We present Supernova, a 650M-parameter decoder-only transformer that demonstrates how careful architectural design and tokenization innovation can achieve the performance of larger models while maintaining computational efficiency. Our architecture combines Rotary Positional Embeddings (RoPE), Grouped Query Attention (GQA) with a 3:1 compression ratio, RMSNorm for computational efficiency, and SwiGLU activation functions. A critical innovation is our custom 128,000-vocabulary byte-level BPE tokenizer, which achieves state-of-the-art compression performance. Through detailed analysis, we show that Supernova achieves 90% of the performance of 1B-parameter models while using 53% fewer parameters and requiring only 100B training tokens--an order of magnitude less than competing models. Our findings challenge the prevailing scaling paradigm, demonstrating that architectural efficiency and tokenization quality can compensate for reduced parameter counts.
>
---
#### [new 014] Text-to-SQL for Enterprise Data Analytics
- **分类: cs.CL; cs.AI; cs.DB; cs.HC**

- **简介: 该论文属于自然语言处理与数据库交互任务，旨在解决企业中非技术人员难以自主获取数据的问题。论文提出了一个结合知识图谱、文本到SQL生成与交互式聊天的系统，提升了企业用户的数据查询效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.14372v1](http://arxiv.org/pdf/2507.14372v1)**

> **作者:** Albert Chen; Manas Bundele; Gaurav Ahlawat; Patrick Stetz; Zhitao Wang; Qiang Fei; Donghoon Jung; Audrey Chu; Bharadwaj Jayaraman; Ayushi Panth; Yatin Arora; Sourav Jain; Renjith Varma; Alexey Ilin; Iuliia Melnychuk; Chelsea Chueh; Joyan Sil; Xiaofeng Wang
>
> **备注:** 11 pages, 8 figures, Workshop on Agentic AI for Enterprise at KDD '25
>
> **摘要:** The introduction of large language models has brought rapid progress on Text-to-SQL benchmarks, but it is not yet easy to build a working enterprise solution. In this paper, we present insights from building an internal chatbot that enables LinkedIn's product managers, engineers, and operations teams to self-serve data insights from a large, dynamic data lake. Our approach features three components. First, we construct a knowledge graph that captures up-to-date semantics by indexing database metadata, historical query logs, wikis, and code. We apply clustering to identify relevant tables for each team or product area. Second, we build a Text-to-SQL agent that retrieves and ranks context from the knowledge graph, writes a query, and automatically corrects hallucinations and syntax errors. Third, we build an interactive chatbot that supports various user intents, from data discovery to query writing to debugging, and displays responses in rich UI elements to encourage follow-up chats. Our chatbot has over 300 weekly users. Expert review shows that 53% of its responses are correct or close to correct on an internal benchmark set. Through ablation studies, we identify the most important knowledge graph and modeling components, offering a practical path for developing enterprise Text-to-SQL solutions.
>
---
#### [new 015] Reasoning Models are Test Exploiters: Rethinking Multiple-Choice
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决当前多选题问答（MCQA）评估是否仍能有效反映大语言模型真实推理能力的问题。论文通过系统评估15个基准数据集和25个大语言模型，发现允许模型在看到选项后推理会导致其利用选项信息，从而高估真实性能。最终提出更鲁棒的评估建议。**

- **链接: [http://arxiv.org/pdf/2507.15337v1](http://arxiv.org/pdf/2507.15337v1)**

> **作者:** Narun Raman; Taylor Lundy; Kevin Leyton-Brown
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** When evaluating Large Language Models (LLMs) in question-answering domains, it is common to ask the model to choose among a fixed set of choices (so-called multiple-choice question-answering, or MCQA). Although downstream tasks of interest typically do not provide systems with explicit options among which to choose, this approach is nevertheless widely used because it makes it makes automatic grading straightforward and has tended to produce challenging benchmarks that correlate sufficiently well with downstream performance. This paper investigates the extent to which this trend continues to hold for state-of-the-art reasoning models, describing a systematic evaluation of $15$ different question-answering benchmarks (e.g., MMLU, HLE) and $25$ different LLMs (including small models such as Qwen 7B and relatively large models such as Llama 70B). For each model-benchmark pair, we considered $5$ ways of presenting the model with questions, including variations on whether multiple choices were offered to the model at all; whether "none of the above" sometimes replaced the right answer; and whether the model was permitted to perform chain-of-thought reasoning before and/or after the choices were presented. MCQA remained a good proxy for the downstream performance of models as long as they were allowed to perform chain-of-thought reasoning only before being presented with the options among which they had to select. On the other hand, large models that were able to perform reasoning after being given a set of options tended to significantly outperform their free-text performance due to exploiting the information in the options. We conclude that MCQA is no longer a good proxy for assessing downstream performance of state-of-the-art models, and offer practical guidelines for designing more robust, bias-resistant benchmarks that better reflect LLMs' genuine reasoning capabilities.
>
---
#### [new 016] From Queries to Criteria: Understanding How Astronomers Evaluate LLMs
- **分类: cs.CL; astro-ph.IM**

- **简介: 该论文研究天文学家如何评估大语言模型（LLMs），旨在改进LLM的评估方法。任务是构建一个基于LLM的天文文献问答机器人，并通过分析用户查询和访谈结果，总结评估标准，提出更贴近实际需求的LLM评测建议。**

- **链接: [http://arxiv.org/pdf/2507.15715v1](http://arxiv.org/pdf/2507.15715v1)**

> **作者:** Alina Hyk; Kiera McCormick; Mian Zhong; Ioana Ciucă; Sanjib Sharma; John F Wu; J. E. G. Peek; Kartheik G. Iyer; Ziang Xiao; Anjalie Field
>
> **备注:** Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
>
> **摘要:** There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
>
---
#### [new 017] Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR
- **分类: cs.CL**

- **简介: 该论文属于强化学习与大语言模型推理任务，旨在解决现有RLVR方法未区分知识与推理相关token的问题。作者提出Archer方法，通过双token约束与同步更新，对推理token鼓励探索，对知识token加强约束，从而提升模型在数学推理与代码生成等任务上的表现。**

- **链接: [http://arxiv.org/pdf/2507.15778v1](http://arxiv.org/pdf/2507.15778v1)**

> **作者:** Jiakang Wang; Runze Liu; Fuzheng Zhang; Xiu Li; Guorui Zhou
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has become an effective post-training method for improving the reasoning abilities of Large Language Models (LLMs), mainly by shaping higher-order behaviors such as reflection and planning. However, previous RLVR algorithms often apply uniform training signals to all tokens, without considering the different roles of low-entropy knowledge-related tokens and high-entropy reasoning-related tokens. Some recent methods try to separate these token types by gradient masking or asynchronous updates, but these approaches may break semantic dependencies in the model output and hinder effective learning. In this work, we propose Archer, an entropy-aware RLVR approach with dual-token constraints and synchronous updates. Specifically, our method applies weaker KL regularization and higher clipping thresholds to reasoning tokens to encourage exploration, while using stronger constraints on knowledge tokens to maintain factual knowledge. Experimental results on several mathematical reasoning and code generation benchmarks show that our approach significantly outperforms previous RLVR methods, reaching or exceeding state-of-the-art performance among models of comparable size. The code is available at https://github.com/wizard-III/ArcherCodeR.
>
---
#### [new 018] CoLD: Counterfactually-Guided Length Debiasing for Process Reward Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决过程奖励模型（PRMs）中存在的长度偏差问题。现有PRMs倾向于给更长的推理步骤更高评分，即使语义和逻辑未变。论文提出CoLD方法，结合长度惩罚、偏差估计和联合训练，减少长度对奖励预测的影响，提升模型推理的准确性和简洁性。**

- **链接: [http://arxiv.org/pdf/2507.15698v1](http://arxiv.org/pdf/2507.15698v1)**

> **作者:** Congmin Zheng; Jiachen Zhu; Jianghao Lin; Xinyi Dai; Yong Yu; Weinan Zhang; Mengyue Yang
>
> **摘要:** Process Reward Models (PRMs) play a central role in evaluating and guiding multi-step reasoning in large language models (LLMs), especially for mathematical problem solving. However, we identify a pervasive length bias in existing PRMs: they tend to assign higher scores to longer reasoning steps, even when the semantic content and logical validity are unchanged. This bias undermines the reliability of reward predictions and leads to overly verbose outputs during inference. To address this issue, we propose CoLD(Counterfactually-Guided Length Debiasing), a unified framework that mitigates length bias through three components: an explicit length-penalty adjustment, a learned bias estimator trained to capture spurious length-related signals, and a joint training strategy that enforces length-invariance in reward predictions. Our approach is grounded in counterfactual reasoning and informed by causal graph analysis. Extensive experiments on MATH500 and GSM-Plus show that CoLD consistently reduces reward-length correlation, improves accuracy in step selection, and encourages more concise, logically valid reasoning. These results demonstrate the effectiveness and practicality of CoLD in improving the fidelity and robustness of PRMs.
>
---
#### [new 019] Disparities in Peer Review Tone and the Role of Reviewer Anonymity
- **分类: cs.CL**

- **简介: 该论文分析了8万余条同行评审数据，研究评审语言在作者性别、种族和机构背景间的差异，并探讨评审匿名性对评价语言的影响，旨在揭示同行评审中的隐性偏见与结构性不公问题。**

- **链接: [http://arxiv.org/pdf/2507.14741v1](http://arxiv.org/pdf/2507.14741v1)**

> **作者:** Maria Sahakyan; Bedoor AlShebli
>
> **摘要:** The peer review process is often regarded as the gatekeeper of scientific integrity, yet increasing evidence suggests that it is not immune to bias. Although structural inequities in peer review have been widely debated, much less attention has been paid to the subtle ways in which language itself may reinforce disparities. This study undertakes one of the most comprehensive linguistic analyses of peer review to date, examining more than 80,000 reviews in two major journals. Using natural language processing and large-scale statistical modeling, it uncovers how review tone, sentiment, and supportive language vary across author demographics, including gender, race, and institutional affiliation. Using a data set that includes both anonymous and signed reviews, this research also reveals how the disclosure of reviewer identity shapes the language of evaluation. The findings not only expose hidden biases in peer feedback, but also challenge conventional assumptions about anonymity's role in fairness. As academic publishing grapples with reform, these insights raise critical questions about how review policies shape career trajectories and scientific progress.
>
---
#### [new 020] A Penalty Goes a Long Way: Measuring Lexical Diversity in Synthetic Texts Under Prompt-Influenced Length Variations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决合成文本多样性评估中因文本长度变化引入的偏差问题。作者提出了一种新的度量方法——Penalty-Adjusted Type-Token Ratio（PATTR），以更准确地衡量文本的词汇多样性。实验基于多个大语言模型生成的视频脚本数据，验证了PATTR在多样性评估中的有效性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.15092v1](http://arxiv.org/pdf/2507.15092v1)**

> **作者:** Vijeta Deshpande; Ishita Dasgupta; Uttaran Bhattacharya; Somdeb Sarkhel; Saayan Mitra; Anna Rumshisky
>
> **摘要:** Synthetic text generated by Large Language Models (LLMs) is increasingly used for further training and improvement of LLMs. Diversity is crucial for the effectiveness of synthetic data, and researchers rely on prompt engineering to improve diversity. However, the impact of prompt variations on response text length, and, more importantly, the consequential effect on lexical diversity measurements, remain underexplored. In this work, we propose Penalty-Adjusted Type-Token Ratio (PATTR), a diversity metric robust to length variations. We generate a large synthetic corpus of over 20M words using seven models from the LLaMA, OLMo, and Phi families, focusing on a creative writing task of video script generation, where diversity is crucial. We evaluate per-response lexical diversity using PATTR and compare it against existing metrics of Moving-Average TTR (MATTR) and Compression Ratio (CR). Our analysis highlights how text length variations introduce biases favoring shorter responses. Unlike existing metrics, PATTR explicitly considers the task-specific target response length ($L_T$) to effectively mitigate length biases. We further demonstrate the utility of PATTR in filtering the top-10/100/1,000 most lexically diverse responses, showing that it consistently outperforms MATTR and CR by yielding on par or better diversity with high adherence to $L_T$.
>
---
#### [new 021] ChiMed 2.0: Advancing Chinese Medical Dataset in Facilitating Large Language Modeling
- **分类: cs.CL**

- **简介: 该论文属于中文医学数据集构建任务，旨在解决现有数据集规模小、覆盖窄、不支持预训练和强化学习的问题。论文构建了包含预训练、微调和偏好数据的ChiMed 2.0数据集，并验证其在医学大模型训练中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.15275v1](http://arxiv.org/pdf/2507.15275v1)**

> **作者:** Yuanhe Tian; Junjie Liu; Zhizhou Kou; Yuxiang Li; Yan Song
>
> **摘要:** Building high-quality data resources is crucial for advancing artificial intelligence research and applications in specific domains, particularly in the Chinese medical domain. Existing Chinese medical datasets are limited in size and narrow in domain coverage, falling short of the diverse corpora required for effective pre-training. Moreover, most datasets are designed solely for LLM fine-tuning and do not support pre-training and reinforcement learning from human feedback (RLHF). In this paper, we propose a Chinese medical dataset named ChiMed 2.0, which extends our previous work ChiMed, and covers data collected from Chinese medical online platforms and generated by LLMs. ChiMed 2.0 contains 204.4M Chinese characters covering both traditional Chinese medicine classics and modern general medical data, where there are 164.8K documents for pre-training, 351.6K question-answering pairs for supervised fine-tuning (SFT), and 41.7K preference data tuples for RLHF. To validate the effectiveness of our approach for training a Chinese medical LLM, we conduct further pre-training, SFT, and RLHF experiments on representative general domain LLMs and evaluate their performance on medical benchmark datasets. The results show performance gains across different model scales, validating the dataset's effectiveness and applicability.
>
---
#### [new 022] Language Models Change Facts Based on the Way You Talk
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 这篇论文研究了大型语言模型（LLMs）在用户交互应用中如何因用户语言中的身份特征（如种族、性别、年龄）而产生偏差。任务是分析这些身份标记如何影响模型在医疗、法律、政治等高风险领域的决策，揭示潜在偏见并提出评估工具，以提醒在实际部署前需进行充分评估。**

- **链接: [http://arxiv.org/pdf/2507.14238v1](http://arxiv.org/pdf/2507.14238v1)**

> **作者:** Matthew Kearney; Reuben Binns; Yarin Gal
>
> **摘要:** Large language models (LLMs) are increasingly being used in user-facing applications, from providing medical consultations to job interview advice. Recent research suggests that these models are becoming increasingly proficient at inferring identity information about the author of a piece of text from linguistic patterns as subtle as the choice of a few words. However, little is known about how LLMs use this information in their decision-making in real-world applications. We perform the first comprehensive analysis of how identity markers present in a user's writing bias LLM responses across five different high-stakes LLM applications in the domains of medicine, law, politics, government benefits, and job salaries. We find that LLMs are extremely sensitive to markers of identity in user queries and that race, gender, and age consistently influence LLM responses in these applications. For instance, when providing medical advice, we find that models apply different standards of care to individuals of different ethnicities for the same symptoms; we find that LLMs are more likely to alter answers to align with a conservative (liberal) political worldview when asked factual questions by older (younger) individuals; and that LLMs recommend lower salaries for non-White job applicants and higher salaries for women compared to men. Taken together, these biases mean that the use of off-the-shelf LLMs for these applications may cause harmful differences in medical care, foster wage gaps, and create different political factual realities for people of different identities. Beyond providing an analysis, we also provide new tools for evaluating how subtle encoding of identity in users' language choices impacts model decisions. Given the serious implications of these findings, we recommend that similar thorough assessments of LLM use in user-facing applications are conducted before future deployment.
>
---
#### [new 023] Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLMs）生成错误内容的问题。通过提出一种基于聚类的语义一致性方法（Cleanse），利用模型隐藏嵌入中的语义信息，量化生成内容的不确定性，以检测模型是否产生幻觉。**

- **链接: [http://arxiv.org/pdf/2507.14649v1](http://arxiv.org/pdf/2507.14649v1)**

> **作者:** Minsuh Joo; Hyunsoo Cho
>
> **摘要:** Despite the outstanding performance of large language models (LLMs) across various NLP tasks, hallucinations in LLMs--where LLMs generate inaccurate responses--remains as a critical problem as it can be directly connected to a crisis of building safe and reliable LLMs. Uncertainty estimation is primarily used to measure hallucination levels in LLM responses so that correct and incorrect answers can be distinguished clearly. This study proposes an effective uncertainty estimation approach, \textbf{Cl}ust\textbf{e}ring-based sem\textbf{an}tic con\textbf{s}ist\textbf{e}ncy (\textbf{Cleanse}). Cleanse quantifies the uncertainty with the proportion of the intra-cluster consistency in the total consistency between LLM hidden embeddings which contain adequate semantic information of generations, by employing clustering. The effectiveness of Cleanse for detecting hallucination is validated using four off-the-shelf models, LLaMA-7B, LLaMA-13B, LLaMA2-7B and Mistral-7B and two question-answering benchmarks, SQuAD and CoQA.
>
---
#### [new 024] GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出GRACE，用于多行为序列推荐的生成模型。解决生成推荐中推理信息不足、计算成本高和多尺度建模受限问题。采用融合思维链的语义分词和旅程感知稀疏注意力机制，提升效果并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2507.14758v1](http://arxiv.org/pdf/2507.14758v1)**

> **作者:** Luyi Ma; Wanjia Zhang; Kai Zhao; Abhishek Kulkarni; Lalitesh Morishetti; Anjana Ganesh; Ashish Ranjan; Aashika Padmanabhan; Jianpeng Xu; Jason Cho; Praveen Kanumala; Kaushiki Nag; Sumit Dutta; Kamiya Motwani; Malay Patel; Evren Korpeoglu; Sushant Kumar; Kannan Achan
>
> **备注:** 10 pages, 5 figures, The ACM Conference on Recommender Systems (RecSys) 2025
>
> **摘要:** Generative models have recently demonstrated strong potential in multi-behavior recommendation systems, leveraging the expressive power of transformers and tokenization to generate personalized item sequences. However, their adoption is hindered by (1) the lack of explicit information for token reasoning, (2) high computational costs due to quadratic attention complexity and dense sequence representations after tokenization, and (3) limited multi-scale modeling over user history. In this work, we propose GRACE (Generative Recommendation via journey-aware sparse Attention on Chain-of-thought tokEnization), a novel generative framework for multi-behavior sequential recommendation. GRACE introduces a hybrid Chain-of-Thought (CoT) tokenization method that encodes user-item interactions with explicit attributes from product knowledge graphs (e.g., category, brand, price) over semantic tokenization, enabling interpretable and behavior-aligned generation. To address the inefficiency of standard attention, we design a Journey-Aware Sparse Attention (JSA) mechanism, which selectively attends to compressed, intra-, inter-, and current-context segments in the tokenized sequence. Experiments on two real-world datasets show that GRACE significantly outperforms state-of-the-art baselines, achieving up to +106.9% HR@10 and +106.7% NDCG@10 improvement over the state-of-the-art baseline on the Home domain, and +22.1% HR@10 on the Electronics domain. GRACE also reduces attention computation by up to 48% with long sequences.
>
---
#### [new 025] Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理性能扩展问题。论文提出了一种无需训练的混合测试时扩展方法，结合了细粒度序列扩展与并行扩展，通过逐步验证引导提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2507.15512v1](http://arxiv.org/pdf/2507.15512v1)**

> **作者:** Kaiyan Chang; Yonghao Shi; Chenglong Wang; Hang Zhou; Chi Hu; Xiaoqian Liu; Yingfeng Luo; Yuan Ge; Tong Xiao; Jingbo Zhu
>
> **摘要:** Test-Time Scaling (TTS) is a promising approach to progressively elicit the model's intelligence during inference. Recently, training-based TTS methods, such as continued reinforcement learning (RL), have further surged in popularity, while training-free TTS methods are gradually fading from prominence. However, the additional computation overhead of training amplifies the burden on test-time scaling. In this paper, we focus on training-free TTS methods for reasoning. We first design Conditional Step-level Self-refinement, a fine-grained sequential scaling method guided by process verification. On top of its effectiveness, we further combine it with other classical parallel scaling methods at the step level, to introduce a novel inference paradigm called Hybrid Test-Time Scaling. Extensive experiments on five instruction-tuned LLMs across different scales (3B-14B) and families demonstrate that hybrid strategy incorporating various training-free TTS methods at a fine granularity has considerable potential for expanding the reasoning performance boundaries of LLMs.
>
---
#### [new 026] PromptSuite: A Task-Agnostic Framework for Multi-Prompt Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务。针对大语言模型评估中单提示不可靠的问题，提出了PromptSuite框架，可自动生成多样化提示，支持灵活、可扩展的多提示评估，提升模型评估的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.14913v1](http://arxiv.org/pdf/2507.14913v1)**

> **作者:** Eliya Habba; Noam Dahan; Gili Lior; Gabriel Stanovsky
>
> **备注:** Eliya Habba and Noam Dahan contributed equally to this work
>
> **摘要:** Evaluating LLMs with a single prompt has proven unreliable, with small changes leading to significant performance differences. However, generating the prompt variations needed for a more robust multi-prompt evaluation is challenging, limiting its adoption in practice. To address this, we introduce PromptSuite, a framework that enables the automatic generation of various prompts. PromptSuite is flexible - working out of the box on a wide range of tasks and benchmarks. It follows a modular prompt design, allowing controlled perturbations to each component, and is extensible, supporting the addition of new components and perturbation types. Through a series of case studies, we show that PromptSuite provides meaningful variations to support strong evaluation practices. It is available through both a Python API: https://github.com/eliyahabba/PromptSuite, and a user-friendly web interface: https://promptsuite.streamlit.app/
>
---
#### [new 027] Retention analysis of edited knowledge after fine-tuning
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型编辑知识在微调后的保留情况，属模型编辑与知识更新任务。旨在解决编辑知识易被微调覆盖的问题，分析不同编辑技术与微调目标的交互影响。工作包括系统评估编辑知识的遗忘现象，提出通过冻结相关层提升保留效果。**

- **链接: [http://arxiv.org/pdf/2507.14198v1](http://arxiv.org/pdf/2507.14198v1)**

> **作者:** Fufang Wen; Shichang Zhang
>
> **摘要:** Large language models (LLMs) store vast amounts of knowledge, which often requires updates to correct factual errors, incorporate newly acquired information, or adapt model behavior. Model editing methods have emerged as efficient solutions for such updates, offering localized and precise knowledge modification at significantly lower computational cost than continual training. In parallel, LLMs are frequently fine-tuned for a wide range of downstream tasks. However, the effect of fine-tuning on previously edited knowledge remains poorly understood. In this work, we systematically investigate how different fine-tuning objectives interact with various model editing techniques. Our findings show that edited knowledge is substantially more susceptible to forgetting during fine-tuning than intrinsic knowledge acquired through pre-training. This analysis highlights a key limitation of current editing approaches and suggests that evaluating edit robustness under downstream fine-tuning is critical for their practical deployment. We further find that freezing layers associated with edited content can significantly improve knowledge retention, offering insight into how future editing methods might be made more robust.
>
---
#### [new 028] SYNTHIA: Synthetic Yet Naturally Tailored Human-Inspired PersonAs
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于计算社会科学与语言模型任务，旨在解决现有用户画像数据或依赖高成本人工标注，或生成质量不高的问题。作者构建了SYNTHIA数据集，基于真实社交媒体用户活动生成30,000个合成背景故事，结合真实与合成方法的优势，提升了叙事一致性，并支持多时间维度与社交互动研究。**

- **链接: [http://arxiv.org/pdf/2507.14922v1](http://arxiv.org/pdf/2507.14922v1)**

> **作者:** Vahid Rahimzadeh; Erfan Moosavi Monazzah; Mohammad Taher Pilehvar; Yadollah Yaghoobzadeh
>
> **摘要:** Persona-driven LLMs have emerged as powerful tools in computational social science, yet existing approaches fall at opposite extremes, either relying on costly human-curated data or producing synthetic personas that lack consistency and realism. We introduce SYNTHIA, a dataset of 30,000 backstories derived from 10,000 real social media users from BlueSky open platform across three time windows, bridging this spectrum by grounding synthetic generation in authentic user activity. Our evaluation demonstrates that SYNTHIA achieves competitive performance with state-of-the-art methods in demographic diversity and social survey alignment while significantly outperforming them in narrative consistency. Uniquely, SYNTHIA incorporates temporal dimensionality and provides rich social interaction metadata from the underlying network, enabling new research directions in computational social science and persona-driven language modeling.
>
---
#### [new 029] X-Intelligence 3.0: Training and Evaluating Reasoning LLM for Semiconductor Display
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与半导体显示行业结合的任务，旨在解决大模型在该领域推理能力不足的问题。作者提出了X-Intelligence 3.0，通过领域知识微调、强化学习和检索增强生成技术，提升了模型在半导体显示行业中的推理性能，并构建了自动化评估框架，验证了模型在多项任务上的优越表现。**

- **链接: [http://arxiv.org/pdf/2507.14430v1](http://arxiv.org/pdf/2507.14430v1)**

> **作者:** Xiaolin Yan; Yangxing Liu; Jiazhang Zheng; Chi Liu; Mingyu Du; Caisheng Chen; Haoyang Liu; Ming Ding; Yuan Li; Qiuping Liao; Linfeng Li; Zhili Mei; Siyu Wan; Li Li; Ruyi Zhong; Jiangling Yu; Xule Liu; Huihui Hu; Jiameng Yue; Ruohui Cheng; Qi Yang; Liangqing Wu; Ke Zhu; Chi Zhang; Chufei Jing; Yifan Zhou; Yan Liang; Dongdong Li; Zhaohui Wang; Bin Zhao; Mingzhou Wu; Mingzhong Zhou; Peng Du; Zuomin Liao; Chao Dai; Pengfei Liang; Xiaoguang Zhu; Yu Zhang; Yu Gu; Kun Pan; Yuan Wu; Yanqing Guan; Shaojing Wu; Zikang Feng; Xianze Ma; Peishan Cheng; Wenjuan Jiang; Jing Ba; Huihao Yu; Zeping Hu; Yuan Xu; Zhiwei Liu; He Wang; Zhenguo Lin; Ming Liu; Yanhong Meng
>
> **备注:** Technical Report
>
> **摘要:** Large language models (LLMs) have recently achieved significant advances in reasoning and demonstrated their advantages in solving challenging problems. Yet, their effectiveness in the semiconductor display industry remains limited due to a lack of domain-specific training and expertise. To bridge this gap, we present X-Intelligence 3.0, the first high-performance reasoning model specifically developed for the semiconductor display industry. This model is designed to deliver expert-level understanding and reasoning for the industry's complex challenges. Leveraging a carefully curated industry knowledge base, the model undergoes supervised fine-tuning and reinforcement learning to enhance its reasoning and comprehension capabilities. To further accelerate development, we implemented an automated evaluation framework that simulates expert-level assessments. We also integrated a domain-specific retrieval-augmented generation (RAG) mechanism, resulting in notable performance gains on benchmark datasets. Despite its relatively compact size of 32 billion parameters, X-Intelligence 3.0 outperforms SOTA DeepSeek-R1-671B across multiple evaluations. This demonstrates its exceptional efficiency and establishes it as a powerful solution to the longstanding reasoning challenges faced by the semiconductor display industry.
>
---
#### [new 030] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言推理（NLI）任务，旨在解决现有常识知识库覆盖不足的问题。研究探索了大语言模型作为常识知识生成器的潜力，评估其生成知识的可靠性和对推理准确性的影响。结果显示，尽管整体效果提升有限，但在区分蕴含、矛盾和中性推理上具有一定帮助。**

- **链接: [http://arxiv.org/pdf/2507.15100v1](http://arxiv.org/pdf/2507.15100v1)**

> **作者:** Chathuri Jayaweera; Brianna Yanqui; Bonnie Dorr
>
> **备注:** 9 pages, 8 figures and 5 tables
>
> **摘要:** Natural Language Inference (NLI) is the task of determining the semantic entailment of a premise for a given hypothesis. The task aims to develop systems that emulate natural human inferential processes where commonsense knowledge plays a major role. However, existing commonsense resources lack sufficient coverage for a variety of premise-hypothesis pairs. This study explores the potential of Large Language Models as commonsense knowledge generators for NLI along two key dimensions: their reliability in generating such knowledge and the impact of that knowledge on prediction accuracy. We adapt and modify existing metrics to assess LLM factuality and consistency in generating in this context. While explicitly incorporating commonsense knowledge does not consistently improve overall results, it effectively helps distinguish entailing instances and moderately improves distinguishing contradictory and neutral inferences.
>
---
#### [new 031] Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLMs）中出现的意外代码切换问题。作者通过稀疏自编码器分析发现，代码切换与特定语言特征的过度激活有关，进而提出SASFT方法，通过监督微调控制这些特征的激活值。实验表明，该方法有效减少代码切换，同时保持模型多语言性能。**

- **链接: [http://arxiv.org/pdf/2507.14894v1](http://arxiv.org/pdf/2507.14894v1)**

> **作者:** Boyi Deng; Yu Wan; Baosong Yang; Fei Huang; Wenjie Wang; Fuli Feng
>
> **摘要:** Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses. However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness. In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in four cases. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities.
>
---
#### [new 032] 3LM: Bridging Arabic, STEM, and Code through Benchmarking
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决阿拉伯语在STEM和编程领域缺乏高质量评估基准的问题。作者构建了3LM基准套件，包括阿拉伯语的STEM问答、生成型STEM问题及代码生成三项任务，推动相关语言模型研究发展。**

- **链接: [http://arxiv.org/pdf/2507.15850v1](http://arxiv.org/pdf/2507.15850v1)**

> **作者:** Basma El Amel Boussaha; Leen AlQadi; Mugariya Farooq; Shaikha Alsuwaidi; Giulia Campesan; Ahmed Alzubaidi; Mohammed Alyafeai; Hakim Hacid
>
> **摘要:** Arabic is one of the most widely spoken languages in the world, yet efforts to develop and evaluate Large Language Models (LLMs) for Arabic remain relatively limited. Most existing Arabic benchmarks focus on linguistic, cultural, or religious content, leaving a significant gap in domains like STEM and code which are increasingly relevant for real-world LLM applications. To help bridge this gap, we present 3LM, a suite of three benchmarks designed specifically for Arabic. The first is a set of STEM-related question-answer pairs, naturally sourced from Arabic textbooks and educational worksheets. The second consists of synthetically generated STEM questions, created using the same sources. The third benchmark focuses on code generation, built through a careful translation of two widely used code benchmarks, incorporating a human-in-the-loop process with several rounds of review to ensure high-quality and faithful translations. We release all three benchmarks publicly to support the growth of Arabic LLM research in these essential but underrepresented areas.
>
---
#### [new 033] Can LLMs Infer Personality from Real World Conversations?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与心理测评交叉任务，旨在探究大型语言模型（LLMs）是否能从真实对话中推断人格特质。研究者构建了一个包含555个半结构化访谈及BFI-10评分的数据集，评估GPT-4.1 Mini、Meta-LLaMA和DeepSeek在人格推断上的表现。结果显示，尽管模型具备高重测信度，但与真实评分相关性弱，一致性低，且存在评分偏差。研究指出当前LLM在人格推断上的局限性，并强调需基于证据进行改进。**

- **链接: [http://arxiv.org/pdf/2507.14355v1](http://arxiv.org/pdf/2507.14355v1)**

> **作者:** Jianfeng Zhu; Ruoming Jin; Karin G. Coifman
>
> **备注:** 21 pages, 12 figures
>
> **摘要:** Large Language Models (LLMs) such as OpenAI's GPT-4 and Meta's LLaMA offer a promising approach for scalable personality assessment from open-ended language. However, inferring personality traits remains challenging, and earlier work often relied on synthetic data or social media text lacking psychometric validity. We introduce a real-world benchmark of 555 semi-structured interviews with BFI-10 self-report scores for evaluating LLM-based personality inference. Three state-of-the-art LLMs (GPT-4.1 Mini, Meta-LLaMA, and DeepSeek) were tested using zero-shot prompting for BFI-10 item prediction and both zero-shot and chain-of-thought prompting for Big Five trait inference. All models showed high test-retest reliability, but construct validity was limited: correlations with ground-truth scores were weak (max Pearson's $r = 0.27$), interrater agreement was low (Cohen's $\kappa < 0.10$), and predictions were biased toward moderate or high trait levels. Chain-of-thought prompting and longer input context modestly improved distributional alignment, but not trait-level accuracy. These results underscore limitations in current LLM-based personality inference and highlight the need for evidence-based development for psychological applications.
>
---
#### [new 034] BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学自然语言处理任务，旨在解决当前眼科领域大语言模型评估基准不足的问题。作者构建了BELO，一个由13名眼科专家参与审核的高质量眼科多选题数据集，包含900道题目，用于评估模型在临床知识和推理方面的能力。**

- **链接: [http://arxiv.org/pdf/2507.15717v1](http://arxiv.org/pdf/2507.15717v1)**

> **作者:** Sahana Srinivasan; Xuguang Ai; Thaddaeus Wai Soon Lo; Aidan Gilson; Minjie Zou; Ke Zou; Hyunjae Kim; Mingjia Yang; Krithi Pushpanathan; Samantha Yew; Wan Ting Loke; Jocelyn Goh; Yibing Chen; Yiming Kong; Emily Yuelei Fu; Michelle Ongyong Hui; Kristen Nwanyanwu; Amisha Dave; Kelvin Zhenghao Li; Chen-Hsin Sun; Mark Chia; Gabriel Dawei Yang; Wendy Meihua Wong; David Ziyou Chen; Dianbo Liu; Maxwell Singer; Fares Antaki; Lucian V Del Priore; Jost Jonas; Ron Adelman; Qingyu Chen; Yih-Chung Tham
>
> **摘要:** Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models.
>
---
#### [new 035] Open-Source LLMs Collaboration Beats Closed-Source LLMs: A Scalable Multi-Agent System
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出SMACS，一个可扩展的多智能体协作系统框架，旨在通过整合多个开源大语言模型（LLMs）来超越闭源模型的表现。该研究属于多模型协作任务，解决如何有效集成开源模型以提升整体性能的问题。论文主要工作包括设计基于检索的先验选择和探索-利用驱动的后验增强方法，实验证明其在多个基准测试中优于主流闭源模型。**

- **链接: [http://arxiv.org/pdf/2507.14200v1](http://arxiv.org/pdf/2507.14200v1)**

> **作者:** Shengji Tang; Jianjian Cao; Weihao Lin; Jiale Hong; Bo Zhang; Shuyue Hu; Lei Bai; Tao Chen; Wanli Ouyang; Peng Ye
>
> **摘要:** This paper aims to demonstrate the potential and strengths of open-source collectives. It leads to a promising question: Can we harness multiple open-source LLMs to match or even beat the closed-source LLMs? To answer this, we propose SMACS, a scalable multi-agent collaboration system (MACS) framework with high performance. Specifically, for continuous integration of new LLMs and generalization to diverse questions, we first propose a Retrieval-based Prior Selection (RPS), which assigns a proxy performance score to each LLM to select the Top-k LLMs at the instance level for any given question. Then, we propose an Exploration-Exploitation-Driven Posterior Enhancement (EPE), encouraging the generation of diverse responses through prior dropping and selecting the high-quality response via a hybrid posterior score. Experiments on eight mainstream benchmarks validate the effectiveness of our SMACS: by integrating fifteen open-source LLMs, SMACS outperforms leading closed-source LLMs in 2025, e.g., Claude-3.7-Sonnet (+12.73%), GPT-4.1(+5.36%) and GPT-o3-mini(+5.28%) across multiple tasks. Remarkably, it even exceeds the average of best results of different datasets from both open-source LLMs (+2.86%) and closed-source LLMs (+2.04%), pushing the upper bound of intelligence. Code will be released at https://github.com/magent4aci/SMACS.
>
---
#### [new 036] Error-Aware Curriculum Learning for Biomedical Relation Classification
- **分类: cs.CL**

- **简介: 论文属于生物医学关系分类任务，旨在提升知识图谱构建与药物重定位等应用中的关系分类效果。通过设计误差感知的师生框架，利用大语言模型分析错误、生成修正策略，并结合知识图引导课程学习，逐步提升模型性能。最终在多个数据集上达到最优表现。**

- **链接: [http://arxiv.org/pdf/2507.14374v1](http://arxiv.org/pdf/2507.14374v1)**

> **作者:** Sinchani Chakraborty; Sudeshna Sarkar; Pawan Goyal
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Relation Classification (RC) in biomedical texts is essential for constructing knowledge graphs and enabling applications such as drug repurposing and clinical decision-making. We propose an error-aware teacher--student framework that improves RC through structured guidance from a large language model (GPT-4o). Prediction failures from a baseline student model are analyzed by the teacher to classify error types, assign difficulty scores, and generate targeted remediations, including sentence rewrites and suggestions for KG-based enrichment. These enriched annotations are used to train a first student model via instruction tuning. This model then annotates a broader dataset with difficulty scores and remediation-enhanced inputs. A second student is subsequently trained via curriculum learning on this dataset, ordered by difficulty, to promote robust and progressive learning. We also construct a heterogeneous biomedical knowledge graph from PubMed abstracts to support context-aware RC. Our approach achieves new state-of-the-art performance on 4 of 5 PPI datasets and the DDI dataset, while remaining competitive on ChemProt.
>
---
#### [new 037] Conflicting narratives and polarization on social media
- **分类: cs.CL; cs.SI**

- **简介: 该论文研究社交媒体中对立叙事如何引发极化与议题联盟。通过分析德国推特用户关于俄乌战争、新冠疫情、气候变化等议题的推文，提取文本信号，揭示不同群体对同一事件的角色分配差异及叙事主体差异，并发现叙事联盟的策略，以探讨极化的话语机制。**

- **链接: [http://arxiv.org/pdf/2507.15600v1](http://arxiv.org/pdf/2507.15600v1)**

> **作者:** Armin Pournaki
>
> **备注:** 30 pages, 7 figures
>
> **摘要:** Narratives are key interpretative devices by which humans make sense of political reality. In this work, we show how the analysis of conflicting narratives, i.e. conflicting interpretive lenses through which political reality is experienced and told, provides insight into the discursive mechanisms of polarization and issue alignment in the public sphere. Building upon previous work that has identified ideologically polarized issues in the German Twittersphere between 2021 and 2023, we analyze the discursive dimension of polarization by extracting textual signals of conflicting narratives from tweets of opposing opinion groups. Focusing on a selection of salient issues and events (the war in Ukraine, Covid, climate change), we show evidence for conflicting narratives along two dimensions: (i) different attributions of actantial roles to the same set of actants (e.g. diverging interpretations of the role of NATO in the war in Ukraine), and (ii) emplotment of different actants for the same event (e.g. Bill Gates in the right-leaning Covid narrative). Furthermore, we provide first evidence for patterns of narrative alignment, a discursive strategy that political actors employ to align opinions across issues. These findings demonstrate the use of narratives as an analytical lens into the discursive mechanisms of polarization.
>
---
#### [new 038] Evaluating Text Style Transfer: A Nine-Language Benchmark for Text Detoxification
- **分类: cs.CL**

- **简介: 该论文属于文本风格迁移（TST）任务，旨在解决多语言文本去毒化评估不足的问题。论文提出首个涵盖九种语言的基准，评估神经模型与大语言模型作为评判者的有效性，探索更可靠的多语言TST评估方法。**

- **链接: [http://arxiv.org/pdf/2507.15557v1](http://arxiv.org/pdf/2507.15557v1)**

> **作者:** Vitaly Protasov; Nikolay Babakov; Daryna Dementieva; Alexander Panchenko
>
> **备注:** preprint
>
> **摘要:** Despite recent progress in large language models (LLMs), evaluation of text generation tasks such as text style transfer (TST) remains a significant challenge. Recent studies (Dementieva et al., 2024; Pauli et al., 2025) revealed a substantial gap between automatic metrics and human judgments. Moreover, most prior work focuses exclusively on English, leaving multilingual TST evaluation largely unexplored. In this paper, we perform the first comprehensive multilingual study on evaluation of text detoxification system across nine languages: English, Spanish, German, Chinese, Arabic, Hindi, Ukrainian, Russian, Amharic. Drawing inspiration from the machine translation, we assess the effectiveness of modern neural-based evaluation models alongside prompting-based LLM-as-a-judge approaches. Our findings provide a practical recipe for designing more reliable multilingual TST evaluation pipeline in the text detoxification case.
>
---
#### [new 039] WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索与数据合成任务，旨在解决高质量信息检索（IS）代理训练数据不足的问题。论文提出WebShaper框架，通过集合论对IS任务进行形式化，引入“知识投影”（KP）操作，实现推理结构的精确控制。利用KP操作组合，从种子任务出发进行多步扩展，生成复杂问题与答案。最终在GAIA和WebWalkerQA基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.15061v1](http://arxiv.org/pdf/2507.15061v1)**

> **作者:** Zhengwei Tao; Jialong Wu; Wenbiao Yin; Junkai Zhang; Baixuan Li; Haiyang Shen; Kuan Li; Liwen Zhang; Xinyu Wang; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **摘要:** The advent of Large Language Model (LLM)-powered agents has revolutionized artificial intelligence by enabling solutions to complex, open-ended tasks through web-based information-seeking (IS) capabilities. The scarcity of high-quality training data has limited the development of IS agents. Existing approaches typically adopt an information-driven paradigm that first collects web data and then generates questions based on the retrieval. However, this may lead to inconsistency between information structure and reasoning structure, question and answer. To mitigate, we propose a formalization-driven IS data synthesis framework WebShaper to construct a dataset. WebShaper systematically formalizes IS tasks through set theory. Central to the formalization is the concept of Knowledge Projections (KP), which enables precise control over reasoning structure by KP operation compositions. During synthesis, we begin by creating seed tasks, then use a multi-step expansion process. At each step, an agentic Expander expands the current formal question more complex with retrieval and validation tools based on our formalization. We train our model on the synthesized dataset. Experiment results demonstrate that WebShaper achieves state-of-the-art performance among open-sourced IS agents on GAIA and WebWalkerQA benchmarks.
>
---
#### [new 040] In-Depth and In-Breadth: Pre-training Multimodal Language Models Customized for Comprehensive Chart Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决现有图表理解模型泛化能力弱、数据对齐不足的问题。作者提出了ChartScope，通过合成多样化图表数据和采用Dual-Path训练策略，提升模型对图表及底层数据的理解。实验表明其在多种图表类型上表现优异，并发布了ChartDQA新基准。**

- **链接: [http://arxiv.org/pdf/2507.14298v1](http://arxiv.org/pdf/2507.14298v1)**

> **作者:** Wan-Cyuan Fan; Yen-Chun Chen; Mengchen Liu; Alexander Jacobson; Lu Yuan; Leonid Sigal
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2407.14506
>
> **摘要:** Recent methods for customizing Large Vision Language Models (LVLMs) for domain-specific tasks have shown promising results in scientific chart comprehension. However, existing approaches face two major limitations: First, they rely on paired data from only a few chart types, limiting generalization to wide range of chart types. Secondly, they lack targeted pre-training for chart-data alignment, which hampers the model's understanding of underlying data. In this paper, we introduce ChartScope, an LVLM optimized for in-depth chart comprehension across diverse chart types. We propose an efficient data generation pipeline that synthesizes paired data for a wide range of chart types, along with a novel Dual-Path training strategy that enabling the model to succinctly capture essential data details while preserving robust reasoning capabilities by incorporating reasoning over the underlying data. Lastly, we establish ChartDQA, a new benchmark for evaluating not only question-answering at different levels but also underlying data understanding. Experimental results demonstrate that ChartScope significantly enhances comprehension on a wide range of chart types. The code and data are available at https://davidhalladay.github.io/chartscope_demo.
>
---
#### [new 041] Compositional Understanding in Signaling Games
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决信号游戏中接收者难以理解组合信息的问题。作者构建了两种新模型：简约接收者仅从信号的原子消息学习，通用接收者利用所有可用信息，使接收者能真正理解组合信息。**

- **链接: [http://arxiv.org/pdf/2507.15706v1](http://arxiv.org/pdf/2507.15706v1)**

> **作者:** David Peter Wallis Freeborn
>
> **摘要:** Receivers in standard signaling game models struggle with learning compositional information. Even when the signalers send compositional messages, the receivers do not interpret them compositionally. When information from one message component is lost or forgotten, the information from other components is also erased. In this paper I construct signaling game models in which genuine compositional understanding evolves. I present two new models: a minimalist receiver who only learns from the atomic messages of a signal, and a generalist receiver who learns from all of the available information. These models are in many ways simpler than previous alternatives, and allow the receivers to learn from the atomic components of messages.
>
---
#### [new 042] On the robustness of modeling grounded word learning through a child's egocentric input
- **分类: cs.CL**

- **简介: 该论文研究儿童语言习得，探索机器学习如何模拟这一过程。任务是通过儿童视角的输入，验证多模态神经网络在词汇学习中的鲁棒性。论文使用SAYCam数据集，自动转录语音并构建多模态数据，训练模型识别词义映射，结果显示模型可跨架构泛化，同时体现个体差异。**

- **链接: [http://arxiv.org/pdf/2507.14749v1](http://arxiv.org/pdf/2507.14749v1)**

> **作者:** Wai Keen Vong; Brenden M. Lake
>
> **摘要:** What insights can machine learning bring to understanding human language acquisition? Large language and multimodal models have achieved remarkable capabilities, but their reliance on massive training datasets creates a fundamental mismatch with children, who succeed in acquiring language from comparatively limited input. To help bridge this gap, researchers have increasingly trained neural networks using data similar in quantity and quality to children's input. Taking this approach to the limit, Vong et al. (2024) showed that a multimodal neural network trained on 61 hours of visual and linguistic input extracted from just one child's developmental experience could acquire word-referent mappings. However, whether this approach's success reflects the idiosyncrasies of a single child's experience, or whether it would show consistent and robust learning patterns across multiple children's experiences was not explored. In this article, we applied automated speech transcription methods to the entirety of the SAYCam dataset, consisting of over 500 hours of video data spread across all three children. Using these automated transcriptions, we generated multi-modal vision-and-language datasets for both training and evaluation, and explored a range of neural network configurations to examine the robustness of simulated word learning. Our findings demonstrate that networks trained on automatically transcribed data from each child can acquire and generalize word-referent mappings across multiple network architectures. These results validate the robustness of multimodal neural networks for grounded word learning, while highlighting the individual differences that emerge in how models learn when trained on each child's developmental experiences.
>
---
#### [new 043] CCL-XCoT: An Efficient Cross-Lingual Knowledge Transfer Method for Mitigating Hallucination Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言大模型在低资源语言中生成幻觉的问题。作者提出CCL-XCoT方法，通过课程对比学习和跨语言思维链提示，减少幻觉生成并提升跨语言知识迁移效果。**

- **链接: [http://arxiv.org/pdf/2507.14239v1](http://arxiv.org/pdf/2507.14239v1)**

> **作者:** Weihua Zheng; Roy Ka-Wei Lee; Zhengyuan Liu; Kui Wu; AiTi Aw; Bowei Zou
>
> **摘要:** Multilingual Large Language Models(MLLMs) demonstrate strong generalization across languages, yet they remain prone to hallucinations, especially in low-resource languages, due to training data imbalances. These hallucinations, which include inaccurate or fabricated outputs, are particularly problematic in domain-specific generation tasks (Chataigner et al., 2024). To address this challenge, we propose CCL-XCoT(Curriculum-based Contrastive Learning-based Cross-lingual Chain-of-Thought), a two-stage fine-tuning framework for mitigating hallucination in MLLMs. Our approach first enhances cross-lingual semantic alignment through curriculum-based contrastive learning combined with next-token prediction during continued pre-training. Building on this foundation, we then introduce a cross-lingual Chain-of-Thought (XCoT) prompting strategy during instruction fine-tuning, which guides the model to reason in a high-resource language before generating answers in the target low-resource language. Experimental results show that CCL-XCoT reduces hallucination rates by up to 62% and substantially improves factual knowledge transfer across language pairs, without relying on external retrieval or multi-model ensembles.
>
---
#### [new 044] Metaphor and Large Language Models: When Surface Features Matter More than Deep Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在隐喻理解中的局限性。通过多数据集实验，分析模型在自然语言推理和问答任务中对隐喻的处理能力，发现模型表现受词汇重叠和句子长度影响较大，而非真正理解隐喻。**

- **链接: [http://arxiv.org/pdf/2507.15357v1](http://arxiv.org/pdf/2507.15357v1)**

> **作者:** Elisa Sanchez-Bayona; Rodrigo Agerri
>
> **摘要:** This paper presents a comprehensive evaluation of the capabilities of Large Language Models (LLMs) in metaphor interpretation across multiple datasets, tasks, and prompt configurations. Although metaphor processing has gained significant attention in Natural Language Processing (NLP), previous research has been limited to single-dataset evaluations and specific task settings, often using artificially constructed data through lexical replacement. We address these limitations by conducting extensive experiments using diverse publicly available datasets with inference and metaphor annotations, focusing on Natural Language Inference (NLI) and Question Answering (QA) tasks. The results indicate that LLMs' performance is more influenced by features like lexical overlap and sentence length than by metaphorical content, demonstrating that any alleged emergent abilities of LLMs to understand metaphorical language are the result of a combination of surface-level features, in-context learning, and linguistic knowledge. This work provides critical insights into the current capabilities and limitations of LLMs in processing figurative language, highlighting the need for more realistic evaluation frameworks in metaphor interpretation tasks. Data and code are publicly available.
>
---
#### [new 045] STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决现有模型缺乏内部思考过程的问题。论文提出STITCH方法，通过交替生成无声推理块和语音响应块，实现语音响应与内部推理的并行处理，从而在不增加延迟的情况下提升推理能力。**

- **链接: [http://arxiv.org/pdf/2507.15375v1](http://arxiv.org/pdf/2507.15375v1)**

> **作者:** Cheng-Han Chiang; Xiaofei Wang; Linjie Li; Chung-Ching Lin; Kevin Lin; Shujie Liu; Zhendong Wang; Zhengyuan Yang; Hung-yi Lee; Lijuan Wang
>
> **备注:** Work in progress. Project page: https://d223302.github.io/STITCH/
>
> **摘要:** Spoken Language Models (SLMs) are designed to take speech inputs and produce spoken responses. However, current SLMs lack the ability to perform an internal, unspoken thinking process before responding. In contrast, humans typically engage in complex mental reasoning internally, enabling them to communicate ideas clearly and concisely. Thus, integrating an unspoken thought process into SLMs is highly desirable. While naively generating a complete chain-of-thought (CoT) reasoning before starting to talk can enable thinking for SLMs, this induces additional latency for the speech response, as the CoT reasoning can be arbitrarily long. To solve this issue, we propose Stitch, a novel generation method that alternates between the generation of unspoken reasoning chunks and spoken response chunks. Since the audio duration of a chunk of spoken response is much longer than the time to generate the tokens in a chunk of spoken response, we use the remaining free time to generate the unspoken reasoning tokens. When a chunk of audio is played to the user, the model continues to generate the next unspoken reasoning chunk, achieving simultaneous thinking and talking. Remarkably, Stitch matches the latency of baselines that cannot generate unspoken CoT by design while outperforming those baselines by 15% on math reasoning datasets; Stitch also performs equally well on non-reasoning datasets as those baseline models. Some animations and demonstrations are on the project page: https://d223302.github.io/STITCH.
>
---
#### [new 046] Probing Information Distribution in Transformer Architectures through Entropy Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型分析任务，旨在解决理解Transformer模型内部信息分布的问题。通过熵分析量化token级不确定性，研究信息在模型中的管理与转换方式，并以GPT模型为例展示其应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.15347v1](http://arxiv.org/pdf/2507.15347v1)**

> **作者:** Amedeo Buonanno; Alessandro Rivetti; Francesco A. N. Palmieri; Giovanni Di Gennaro; Gianmarco Romano
>
> **备注:** Presented to the Italian Workshop on Neural Networks (WIRN2025) and it will appear in a Springer Chapter
>
> **摘要:** This work explores entropy analysis as a tool for probing information distribution within Transformer-based architectures. By quantifying token-level uncertainty and examining entropy patterns across different stages of processing, we aim to investigate how information is managed and transformed within these models. As a case study, we apply the methodology to a GPT-based large language model, illustrating its potential to reveal insights into model behavior and internal representations. This approach may offer insights into model behavior and contribute to the development of interpretability and evaluation frameworks for transformer-based models
>
---
#### [new 047] Collaborative Distillation Strategies for Parameter-Efficient Language Model Deployment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型部署中的高计算成本与慢推理问题。通过多教师模型协同蒸馏，结合输出融合、特征对齐损失和动态权重调整，提升小模型的语言理解与生成能力，在语言建模、文本生成等任务中表现出优越性能。**

- **链接: [http://arxiv.org/pdf/2507.15198v1](http://arxiv.org/pdf/2507.15198v1)**

> **作者:** Xiandong Meng; Yan Wu; Yexin Tian; Xin Hu; Tianze Kang; Junliang Du
>
> **摘要:** This paper addresses the challenges of high computational cost and slow inference in deploying large language models. It proposes a distillation strategy guided by multiple teacher models. The method constructs several teacher models and integrates their output probability distributions and intermediate semantic features. This guides the student model to learn from multiple sources of knowledge. As a result, the student model gains stronger language understanding and generation ability while maintaining a small parameter size. To achieve this, the paper introduces a weighted output fusion mechanism, a feature alignment loss function, and an entropy-driven dynamic teacher weighting strategy. These components improve the quality and stability of knowledge transfer during distillation. Under multi-teacher guidance, the student model captures semantic information more effectively and demonstrates strong performance across multiple evaluation metrics. In particular, the method shows high consistency in expression, generalization ability, and task adaptability in tasks such as language modeling, text generation, and multi-task learning. The experiments compare the proposed method with several widely adopted distillation approaches. The results further confirm its overall advantages in perplexity, distillation loss, and generation quality. This study provides a feasible technical path for the efficient compression of large-scale language models. It also demonstrates the effectiveness of multi-teacher collaborative mechanisms in complex language modeling tasks.
>
---
#### [new 048] Beyond Architectures: Evaluating the Role of Contextual Embeddings in Detecting Bipolar Disorder on Social Media
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在通过社交媒体文本检测双相情感障碍。论文评估了多种NLP模型，发现基于上下文的嵌入（如RoBERTa、BERT）效果最佳，强调了上下文建模在心理健康筛查中的重要性。**

- **链接: [http://arxiv.org/pdf/2507.14231v1](http://arxiv.org/pdf/2507.14231v1)**

> **作者:** Khalid Hasan; Jamil Saquer
>
> **备注:** The 37th International Conference on Software Engineering & Knowledge Engineering, SEKE 2025 (camera-ready)
>
> **摘要:** Bipolar disorder is a chronic mental illness frequently underdiagnosed due to subtle early symptoms and social stigma. This paper explores the advanced natural language processing (NLP) models for recognizing signs of bipolar disorder based on user-generated social media text. We conduct a comprehensive evaluation of transformer-based models (BERT, RoBERTa, ALBERT, ELECTRA, DistilBERT) and Long Short Term Memory (LSTM) models based on contextualized (BERT) and static (GloVe, Word2Vec) word embeddings. Experiments were performed on a large, annotated dataset of Reddit posts after confirming their validity through sentiment variance and judgmental analysis. Our results demonstrate that RoBERTa achieves the highest performance among transformer models with an F1 score of ~98% while LSTM models using BERT embeddings yield nearly identical results. In contrast, LSTMs trained on static embeddings fail to capture meaningful patterns, scoring near-zero F1. These findings underscore the critical role of contextual language modeling in detecting bipolar disorder. In addition, we report model training times and highlight that DistilBERT offers an optimal balance between efficiency and accuracy. In general, our study offers actionable insights for model selection in mental health NLP applications and validates the potential of contextualized language models to support early bipolar disorder screening.
>
---
#### [new 049] Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言大模型对齐任务，旨在解决低资源语言（如印地语）因数据不足导致的性能落后问题。论文提出基于大模型的“选择性翻译”方法，保留非翻译内容（如代码、公式），并系统研究其效果，对比传统翻译工具与大模型的性能，探索提升多语言对齐的有效策略。**

- **链接: [http://arxiv.org/pdf/2507.14304v1](http://arxiv.org/pdf/2507.14304v1)**

> **作者:** Rakesh Paul; Anusha Kamath; Kanishk Singla; Raviraj Joshi; Utkarsh Vaidya; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs.
>
---
#### [new 050] On the Inevitability of Left-Leaning Political Bias in Aligned Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文探讨了对齐语言模型中左倾政治偏见的必然性，认为追求无害、有益和诚实（HHH）的目标与进步主义价值观一致，从而导致左倾偏见。论文任务是分析政治偏见与AI对齐原则的关系，挑战了将左倾偏见视为问题的主流观点。**

- **链接: [http://arxiv.org/pdf/2507.15328v1](http://arxiv.org/pdf/2507.15328v1)**

> **作者:** Thilo Hagendorff
>
> **摘要:** The guiding principle of AI alignment is to train large language models (LLMs) to be harmless, helpful, and honest (HHH). At the same time, there are mounting concerns that LLMs exhibit a left-wing political bias. Yet, the commitment to AI alignment cannot be harmonized with the latter critique. In this article, I argue that intelligent systems that are trained to be harmless and honest must necessarily exhibit left-wing political bias. Normative assumptions underlying alignment objectives inherently concur with progressive moral frameworks and left-wing principles, emphasizing harm avoidance, inclusivity, fairness, and empirical truthfulness. Conversely, right-wing ideologies often conflict with alignment guidelines. Yet, research on political bias in LLMs is consistently framing its insights about left-leaning tendencies as a risk, as problematic, or concerning. This way, researchers are actively arguing against AI alignment, tacitly fostering the violation of HHH principles.
>
---
#### [new 051] MEKiT: Multi-source Heterogeneous Knowledge Injection Method via Instruction Tuning for Emotion-Cause Pair Extraction
- **分类: cs.CL**

- **简介: 该论文属于情感-原因对抽取（ECPE）任务，旨在解决大语言模型在该任务中表现不佳的问题。通过提出MEKiT方法，注入多源异构知识，提升模型的情感识别与因果推理能力，从而显著提高ECPE任务的性能。**

- **链接: [http://arxiv.org/pdf/2507.14887v1](http://arxiv.org/pdf/2507.14887v1)**

> **作者:** Shiyi Mu; Yongkang Liu; Shi Feng; Xiaocui Yang; Daling Wang; Yifei Zhang
>
> **备注:** Accepted by CogSci
>
> **摘要:** Although large language models (LLMs) excel in text comprehension and generation, their performance on the Emotion-Cause Pair Extraction (ECPE) task, which requires reasoning ability, is often underperform smaller language model. The main reason is the lack of auxiliary knowledge, which limits LLMs' ability to effectively perceive emotions and reason causes. To address this issue, we propose a novel \textbf{M}ulti-source h\textbf{E}terogeneous \textbf{K}nowledge \textbf{i}njection me\textbf{T}hod, MEKiT, which integrates heterogeneous internal emotional knowledge and external causal knowledge. Specifically, for these two distinct aspects and structures of knowledge, we apply the approaches of incorporating instruction templates and mixing data for instruction-tuning, which respectively facilitate LLMs in more comprehensively identifying emotion and accurately reasoning causes. Experimental results demonstrate that MEKiT provides a more effective and adaptable solution for the ECPE task, exhibiting an absolute performance advantage over compared baselines and dramatically improving the performance of LLMs on the ECPE task.
>
---
#### [new 052] P3: Prompts Promote Prompting
- **分类: cs.CL**

- **简介: 该论文属于自动提示优化任务，旨在解决当前大语言模型中系统提示和用户提示单独优化效果不佳的问题。作者提出了P3框架，通过迭代方式同时优化系统和用户提示，并在线上推理时进一步优化，提升了模型在多种任务上的表现。**

- **链接: [http://arxiv.org/pdf/2507.15675v1](http://arxiv.org/pdf/2507.15675v1)**

> **作者:** Xinyu Zhang; Yuanquan Hu; Fangchao Liu; Zhicheng Dou
>
> **备注:** Accepted to ACL 2025 findings
>
> **摘要:** Current large language model (LLM) applications often employ multi-component prompts, comprising both system and user prompts, to guide model behaviors. While recent advancements have demonstrated the efficacy of automatically optimizing either the system or user prompt to boost performance, such unilateral approaches often yield suboptimal outcomes due to the interdependent nature of these components. In this work, we introduce P3, a novel self-improvement framework that concurrently optimizes both system and user prompts through an iterative process. The offline optimized prompts are further leveraged to promote online prompting by performing query-dependent prompt optimization. Extensive experiments on general tasks (e.g., Arena-hard and Alpaca-eval) and reasoning tasks (e.g., GSM8K and GPQA) demonstrate that P3 achieves superior performance in the realm of automatic prompt optimization. Our results highlight the effectiveness of a holistic optimization strategy in enhancing LLM performance across diverse domains.
>
---
#### [new 053] Retrieval-Augmented Clinical Benchmarking for Contextual Model Testing in Kenyan Primary Care: A Methodology Paper
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决大型语言模型（LLMs）在非洲基层医疗应用中的有效性问题。通过构建基于肯尼亚国家指南的检索增强生成（RAG）方法，创建了临床问答数据集Alama Health QA，并引入新评估指标测试模型的临床推理、安全性和适应性，支持非洲医疗AI的安全部署。**

- **链接: [http://arxiv.org/pdf/2507.14615v1](http://arxiv.org/pdf/2507.14615v1)**

> **作者:** Fred Mutisya; Shikoh Gitau; Christine Syovata; Diana Oigara; Ibrahim Matende; Muna Aden; Munira Ali; Ryan Nyotu; Diana Marion; Job Nyangena; Nasubo Ongoma; Keith Mbae; Elizabeth Wamicha; Eric Mibuari; Jean Philbert Nsengemana; Talkmore Chidede
>
> **备注:** 29 pages, 6 figs, 6 tables. Companion methods paper forthcoming
>
> **摘要:** Large Language Models(LLMs) hold promise for improving healthcare access in low-resource settings, but their effectiveness in African primary care remains underexplored. We present a methodology for creating a benchmark dataset and evaluation framework focused on Kenyan Level 2 and 3 clinical care. Our approach uses retrieval augmented generation (RAG) to ground clinical questions in Kenya's national guidelines, ensuring alignment with local standards. These guidelines were digitized, chunked, and indexed for semantic retrieval. Gemini Flash 2.0 Lite was then prompted with guideline excerpts to generate realistic clinical scenarios, multiple-choice questions, and rationale based answers in English and Swahili. Kenyan physicians co-created and refined the dataset, and a blinded expert review process ensured clinical accuracy, clarity, and cultural appropriateness. The resulting Alama Health QA dataset includes thousands of regulator-aligned question answer pairs across common outpatient conditions. Beyond accuracy, we introduce evaluation metrics that test clinical reasoning, safety, and adaptability such as rare case detection (Needle in the Haystack), stepwise logic (Decision Points), and contextual adaptability. Initial results reveal significant performance gaps when LLMs are applied to localized scenarios, consistent with findings that LLM accuracy is lower on African medical content than on US-based benchmarks. This work offers a replicable model for guideline-driven, dynamic benchmarking to support safe AI deployment in African health systems.
>
---
#### [new 054] Linear Relational Decoding of Morphology in Language Models
- **分类: cs.CL**

- **简介: 论文研究语言模型中形态学关系的线性解码，属于自然语言处理任务。旨在探索模型内部表示是否可解释形态学关系。通过两部分仿射逼近和线性变换Ws，验证了中层表示与对象状态的关系，结果表明该方法在多语言和多模型上均能准确还原形态学关系。**

- **链接: [http://arxiv.org/pdf/2507.14640v1](http://arxiv.org/pdf/2507.14640v1)**

> **作者:** Eric Xia; Jugal Kalita
>
> **摘要:** A two-part affine approximation has been found to be a good approximation for transformer computations over certain subject object relations. Adapting the Bigger Analogy Test Set, we show that the linear transformation Ws, where s is a middle layer representation of a subject token and W is derived from model derivatives, is also able to accurately reproduce final object states for many relations. This linear technique is able to achieve 90% faithfulness on morphological relations, and we show similar findings multi-lingually and across models. Our findings indicate that some conceptual relationships in language models, such as morphology, are readily interpretable from latent space, and are sparsely encoded by cross-layer linear transformations.
>
---
#### [new 055] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 论文提出LEAR方法，用于检索增强生成（RAG）中的证据提取任务。旨在解决检索内容中的噪声影响生成质量的问题，通过强化学习实现理性证据的显式推理与提取，提升下游任务准确率并适用于在线RAG系统。**

- **链接: [http://arxiv.org/pdf/2507.15586v1](http://arxiv.org/pdf/2507.15586v1)**

> **作者:** Xinping Zhao; Shouzheng Huang; Yan Zhong; Xinshuo Hu; Baotian Hu; Min Zhang
>
> **备注:** 16 pages, 7 Figures, 10 Tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly impact the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous methods extract evidence straightforwardly without explicit thinking, which risks filtering out key clues and struggles with generalization. To this end, we propose LEAR, which learns to extract rational evidence by (1) explicitly reasoning to identify potential cues within retrieval contents first, and then (2) consciously extracting to avoid omitting any key cues helpful for answering questions. Specifically, we frame evidence reasoning and evidence extraction into one unified response for end-to-end training; apply knowledge token masks for disentanglement to derive reasoning-based and extraction-based answers; and devise three types of verifiable reward functions, including answer, length, and format, to update the model via the policy optimization algorithm. Extensive experiments on three benchmark datasets show the effectiveness of LEAR, providing compact and high-quality evidence, improving the accuracy of downstream tasks, and promoting effective application in online RAG systems.
>
---
#### [new 056] AlgoSimBench: Identifying Algorithmically Similar Problems for Competitive Programming
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与算法推荐任务，旨在解决识别算法相似问题（ASP）的挑战。作者构建了AlgoSimBench基准，包含402个选择题，评估LLM识别ASP的能力，并提出ASM方法提升准确率。**

- **链接: [http://arxiv.org/pdf/2507.15378v1](http://arxiv.org/pdf/2507.15378v1)**

> **作者:** Jierui Li; Raymond Mooney
>
> **备注:** 19 pages, pre-print only
>
> **摘要:** Recent progress in LLMs, such as reasoning models, has demonstrated strong abilities to solve complex competitive programming problems, often rivaling top human competitors. However, it remains underexplored whether these abilities generalize to relevant domains that are less seen during training. To address this, we introduce AlgoSimBench, a new benchmark designed to assess LLMs' ability to identify algorithmically similar problems (ASPs)-problems that can be solved using similar algorithmic approaches. AlgoSimBench consists of 1317 problems, annotated with 231 distinct fine-grained algorithm tags, from which we curate 402 multiple-choice questions (MCQs), where each question presents one algorithmically similar problem alongside three textually similar but algorithmically dissimilar distractors. Our evaluation reveals that LLMs struggle to identify ASPs, with the best-performing model (o3-mini) achieving only 65.9% accuracy on the MCQ task. To address this challenge, we propose attempted solution matching (ASM), a novel method for improving problem similarity detection. On our MCQ task, ASM yields an absolute accuracy improvement of 6.7% to 11.7% across different models. We also evaluated code embedding models and retrieval methods on similar problem identification. While the adversarial selection of problems degrades the performance to be less than random, we found that simply summarizing the problem to remove narrative elements eliminates the effect, and combining ASM with a keyword-prioritized method, BM25, can yield up to 52.2% accuracy. Code and data are available at github.com
>
---
#### [new 057] Operationalizing AI for Good: Spotlight on Deployment and Integration of AI Models in Humanitarian Work
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 该论文属于人工智能应用任务，旨在解决人道主义工作中AI模型部署与集成困难的问题。论文通过与人道主义组织合作，探讨在资源受限环境中部署AI模型并持续维护的方法，为实践者提供关键经验总结。**

- **链接: [http://arxiv.org/pdf/2507.15823v1](http://arxiv.org/pdf/2507.15823v1)**

> **作者:** Anton Abilov; Ke Zhang; Hemank Lamba; Elizabeth M. Olson; Joel R. Tetreault; Alejandro Jaimes
>
> **摘要:** Publications in the AI for Good space have tended to focus on the research and model development that can support high-impact applications. However, very few AI for Good papers discuss the process of deploying and collaborating with the partner organization, and the resulting real-world impact. In this work, we share details about the close collaboration with a humanitarian-to-humanitarian (H2H) organization and how to not only deploy the AI model in a resource-constrained environment, but also how to maintain it for continuous performance updates, and share key takeaways for practitioners.
>
---
#### [new 058] SOI Matters: Analyzing Multi-Setting Training Dynamics in Pretrained Language Models via Subsets of Interest
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在分析多任务、多语言和多源学习对预训练语言模型的影响。作者提出“兴趣子集”（SOI）框架，识别训练中六类学习行为模式，通过可视化分析其在不同设置间的转换，探索提升模型性能与鲁棒性的方法。**

- **链接: [http://arxiv.org/pdf/2507.15236v1](http://arxiv.org/pdf/2507.15236v1)**

> **作者:** Shayan Vassef; Amirhossein Dabiriaghdam; Mohammadreza Bakhtiari; Yadollah Yaghoobzadeh
>
> **摘要:** This work investigates the impact of multi-task, multi-lingual, and multi-source learning approaches on the robustness and performance of pretrained language models. To enhance this analysis, we introduce Subsets of Interest (SOI), a novel categorization framework that identifies six distinct learning behavior patterns during training, including forgettable examples, unlearned examples, and always correct examples. Through SOI transition heatmaps and dataset cartography visualization, we analyze how examples shift between these categories when transitioning from single-setting to multi-setting configurations. We perform comprehensive experiments across three parallel comparisons: multi-task vs. single-task learning using English tasks (entailment, paraphrase, sentiment), multi-source vs. single-source learning using sentiment analysis datasets, and multi-lingual vs. single-lingual learning using intent classification in French, English, and Persian. Our results demonstrate that multi-source learning consistently improves out-of-distribution performance by up to 7%, while multi-task learning shows mixed results with notable gains in similar task combinations. We further introduce a two-stage fine-tuning approach where the second stage leverages SOI-based subset selection to achieve additional performance improvements. These findings provide new insights into training dynamics and offer practical approaches for optimizing multi-setting language model performance.
>
---
#### [new 059] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决如何有效评估多语言大模型的跨语言对齐能力。受神经科学启发，作者提出了一种基于神经元状态的跨语言对齐评估方法（NeuronXA），并验证其在多个模型和任务上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.14900v1](http://arxiv.org/pdf/2507.14900v1)**

> **作者:** Chongxuan Huang; Yongshi Ye; Biao Fu; Qifeng Su; Xiaodong Shi
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable multilingual capabilities, however, how to evaluate cross-lingual alignment remains underexplored. Existing alignment benchmarks primarily focus on sentence embeddings, but prior research has shown that neural models tend to induce a non-smooth representation space, which impact of semantic alignment evaluation on low-resource languages. Inspired by neuroscientific findings that similar information activates overlapping neuronal regions, we propose a novel Neuron State-Based Cross-Lingual Alignment (NeuronXA) to assess the cross-lingual a lignment capabilities of LLMs, which offers a more semantically grounded approach to assess cross-lingual alignment. We evaluate NeuronXA on several prominent multilingual LLMs (LLaMA, Qwen, Mistral, GLM, and OLMo) across two transfer tasks and three multilingual benchmarks. The results demonstrate that with only 100 parallel sentence pairs, NeuronXA achieves a Pearson correlation of 0.9556 with downstream tasks performance and 0.8514 with transferability. These findings demonstrate NeuronXA's effectiveness in assessing both cross-lingual alignment and transferability, even with a small dataset. This highlights its potential to advance cross-lingual alignment research and to improve the semantic understanding of multilingual LLMs.
>
---
#### [new 060] Rethinking Suicidal Ideation Detection: A Trustworthy Annotation Framework and Cross-Lingual Model Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于自杀意念检测任务，旨在解决现有数据集语言覆盖有限、标注不可靠的问题。作者构建了新的土耳其语自杀意念语料库，提出高效的标注框架，并通过跨语言模型评估检验标注一致性与模型性能，强调需更严谨、包容的标注与评估方法。**

- **链接: [http://arxiv.org/pdf/2507.14693v1](http://arxiv.org/pdf/2507.14693v1)**

> **作者:** Amina Dzafic; Merve Kavut; Ulya Bayram
>
> **备注:** This manuscript has been submitted to the IEEE Journal of Biomedical and Health Informatics
>
> **摘要:** Suicidal ideation detection is critical for real-time suicide prevention, yet its progress faces two under-explored challenges: limited language coverage and unreliable annotation practices. Most available datasets are in English, but even among these, high-quality, human-annotated data remains scarce. As a result, many studies rely on available pre-labeled datasets without examining their annotation process or label reliability. The lack of datasets in other languages further limits the global realization of suicide prevention via artificial intelligence (AI). In this study, we address one of these gaps by constructing a novel Turkish suicidal ideation corpus derived from social media posts and introducing a resource-efficient annotation framework involving three human annotators and two large language models (LLMs). We then address the remaining gaps by performing a bidirectional evaluation of label reliability and model consistency across this dataset and three popular English suicidal ideation detection datasets, using transfer learning through eight pre-trained sentiment and emotion classifiers. These transformers help assess annotation consistency and benchmark model performance against manually labeled data. Our findings underscore the need for more rigorous, language-inclusive approaches to annotation and evaluation in mental health natural language processing (NLP) while demonstrating the questionable performance of popular models with zero-shot transfer learning. We advocate for transparency in model training and dataset construction in mental health NLP, prioritizing data and model reliability.
>
---
#### [new 061] MiroMind-M1: An Open-Source Advancement in Mathematical Reasoning via Context-Aware Multi-Stage Policy Optimization
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在提升开源推理语言模型的透明度与可复现性。作者提出MiroMind-M1系列模型，采用两阶段训练方法（SFT和RLVR）及新算法Context-Aware Multi-Stage Policy Optimization，实现高效数学推理。论文发布了完整模型、数据集与配置，以推动相关研究。**

- **链接: [http://arxiv.org/pdf/2507.14683v1](http://arxiv.org/pdf/2507.14683v1)**

> **作者:** Xingxuan Li; Yao Xiao; Dianwen Ng; Hai Ye; Yue Deng; Xiang Lin; Bin Wang; Zhanfeng Mo; Chong Zhang; Yueyi Zhang; Zonglin Yang; Ruilin Li; Lei Lei; Shihao Xu; Han Zhao; Weiling Chen; Feng Ji; Lidong Bing
>
> **备注:** Technical report
>
> **摘要:** Large language models have recently evolved from fluent text generation to advanced reasoning across diverse domains, giving rise to reasoning language models. Among these domains, mathematical reasoning serves as a representative benchmark as it requires precise multi-step logic and abstract reasoning, which can be generalized to other tasks. While closed-source RLMs such as GPT-o3 demonstrate impressive reasoning capabilities, their proprietary nature limits transparency and reproducibility. Although many open-source projects aim to close this gap, most of them lack sufficient openness by omitting critical resources such as datasets and detailed training configurations, which hinders reproducibility. To contribute toward greater transparency in RLM development, we introduce the MiroMind-M1 series, a set of fully open-source RLMs built on the Qwen-2.5 backbone that match or exceed the performance of existing open-source RLMs. Specifically, our models are trained in two stages: SFT on a carefully curated corpus of 719K math-reasoning problems with verified CoT trajectories, followed by RLVR on 62K challenging and verifiable problems. To enhance the robustness and efficiency of the RLVR process, we introduce Context-Aware Multi-Stage Policy Optimization, an algorithm that integrates length-progressive training with an adaptive repetition penalty to encourage context-aware RL training. Our model achieves state-of-the-art or competitive performance and superior token efficiency among Qwen-2.5-based open-source 7B and 32B models on the AIME24, AIME25, and MATH benchmarks. To facilitate reproducibility, we release the complete stack: models (MiroMind-M1-SFT-7B, MiroMind-M1-RL-7B, MiroMind-M1-RL-32B); datasets (MiroMind-M1-SFT-719K, MiroMind-M1-RL-62K); and all training and evaluation configurations. We hope these resources will support further research and foster community advancement.
>
---
#### [new 062] Mangosteen: An Open Thai Corpus for Language Model Pretraining
- **分类: cs.CL**

- **简介: 论文任务是构建高质量泰语预训练语料库。为解决现有语料清洗方法不适用于泰语且缺乏透明度的问题，作者提出Mangosteen，采用泰语适配的清洗流程，并融合多种优质数据源，显著提升模型表现，同时开源全流程资源以促进后续研究。**

- **链接: [http://arxiv.org/pdf/2507.14664v1](http://arxiv.org/pdf/2507.14664v1)**

> **作者:** Wannaphong Phatthiyaphaibun; Can Udomcharoenchaikit; Pakpoom Singkorapoom; Kunat Pipatanakul; Ekapol Chuangsuwanich; Peerat Limkonchotiwat; Sarana Nutanong
>
> **备注:** Work in Progress.All artifacts in this papers: https://huggingface.co/collections/aisingapore/wangchanlion-v3-687a362d8f0ea2fe4077c6b3
>
> **摘要:** Pre-training data shapes a language model's quality, but raw web text is noisy and demands careful cleaning. Existing large-scale corpora rely on English-centric or language-agnostic pipelines whose heuristics do not capture Thai script or cultural nuances, leaving risky material such as gambling content untreated. Prior Thai-specific efforts customize pipelines or build new ones, yet seldom release their data or document design choices, hindering reproducibility and raising the question of how to construct a transparent, high-quality Thai corpus. We introduce Mangosteen: a 47 billion-token Thai corpus built through a Thai-adapted Dolma pipeline that includes custom rule-based language ID, revised C4/Gopher quality filters, and Thai-trained content filters, plus curated non-web sources such as Wikipedia, Royal Gazette texts, OCR-extracted books, and CC-licensed YouTube subtitles. Systematic ablations using GPT-2 show the pipeline trims CommonCrawl from 202M to 25M documents while raising SEA-HELM NLG from 3 to 11; an 8B-parameter SEA-LION model continually pre-trained on Mangosteen then surpasses SEA-LION-v3 and Llama-3.1 by about four points on Thai benchmarks. We release the full pipeline code, cleaning manifests, corpus snapshot, and all checkpoints, providing a fully reproducible foundation for future Thai and regional LLM research.
>
---
#### [new 063] Is Large Language Model Performance on Reasoning Tasks Impacted by Different Ways Questions Are Asked?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究不同提问方式对大语言模型推理任务性能的影响。属于自然语言处理任务。解决的问题是：不同问题类型（如选择题、判断题、简答题）如何影响模型推理准确性。工作包括测试五种模型在三种问题类型下的推理步骤和最终答案准确性，发现提问方式显著影响性能。**

- **链接: [http://arxiv.org/pdf/2507.15707v1](http://arxiv.org/pdf/2507.15707v1)**

> **作者:** Seok Hwan Song; Mohna Chakraborty; Qi Li; Wallapak Tavanapong
>
> **摘要:** Large Language Models (LLMs) have been evaluated using diverse question types, e.g., multiple-choice, true/false, and short/long answers. This study answers an unexplored question about the impact of different question types on LLM accuracy on reasoning tasks. We investigate the performance of five LLMs on three different types of questions using quantitative and deductive reasoning tasks. The performance metrics include accuracy in the reasoning steps and choosing the final answer. Key Findings: (1) Significant differences exist in LLM performance across different question types. (2) Reasoning accuracy does not necessarily correlate with the final selection accuracy. (3) The number of options and the choice of words, influence LLM performance.
>
---
#### [new 064] DialogueForge: LLM Simulation of Human-Chatbot Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话生成任务，旨在解决人工收集人机对话数据耗时费力的问题。作者提出了DialogueForge框架，利用大语言模型（LLM）模拟人类与聊天机器人的对话，通过种子提示和微调技术生成高质量对话，并评估不同模型的表现。**

- **链接: [http://arxiv.org/pdf/2507.15752v1](http://arxiv.org/pdf/2507.15752v1)**

> **作者:** Ruizhe Zhu; Hao Zhu; Yaxuan Li; Syang Zhou; Shijing Cai; Malgorzata Lazuka; Elliott Ash
>
> **备注:** For our code and data, see https://github.com/nerchio/Human_Chatbot-Generation
>
> **摘要:** Collecting human-chatbot dialogues typically demands substantial manual effort and is time-consuming, which limits and poses challenges for research on conversational AI. In this work, we propose DialogueForge - a framework for generating AI-simulated conversations in human-chatbot style. To initialize each generated conversation, DialogueForge uses seed prompts extracted from real human-chatbot interactions. We test a variety of LLMs to simulate the human chatbot user, ranging from state-of-the-art proprietary models to small-scale open-source LLMs, and generate multi-turn dialogues tailored to specific tasks. In addition, we explore fine-tuning techniques to enhance the ability of smaller models to produce indistinguishable human-like dialogues. We evaluate the quality of the simulated conversations and compare different models using the UniEval and GTEval evaluation protocols. Our experiments show that large proprietary models (e.g., GPT-4o) generally outperform others in generating more realistic dialogues, while smaller open-source models (e.g., Llama, Mistral) offer promising performance with greater customization. We demonstrate that the performance of smaller models can be significantly improved by employing supervised fine-tuning techniques. Nevertheless, maintaining coherent and natural long-form human-like dialogues remains a common challenge across all models.
>
---
#### [new 065] Mind the Gap: A Review of Arabic Post-Training Datasets and Their Limitations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决阿拉伯语大语言模型后训练数据集的不足问题。论文系统评估了现有数据集在能力、可引导性、对齐和鲁棒性等方面的表现，指出任务多样性不足、文档标注不完善及社区采用率低等关键问题，并提出改进建议。**

- **链接: [http://arxiv.org/pdf/2507.14688v1](http://arxiv.org/pdf/2507.14688v1)**

> **作者:** Mohammed Alkhowaiter; Norah Alshahrani; Saied Alshahrani; Reem I. Masoud; Alaa Alzahrani; Deema Alnuhait; Emad A. Alghamdi; Khalid Almubarak
>
> **摘要:** Post-training has emerged as a crucial technique for aligning pre-trained Large Language Models (LLMs) with human instructions, significantly enhancing their performance across a wide range of tasks. Central to this process is the quality and diversity of post-training datasets. This paper presents a review of publicly available Arabic post-training datasets on the Hugging Face Hub, organized along four key dimensions: (1) LLM Capabilities (e.g., Question Answering, Translation, Reasoning, Summarization, Dialogue, Code Generation, and Function Calling); (2) Steerability (e.g., persona and system prompts); (3) Alignment (e.g., cultural, safety, ethics, and fairness), and (4) Robustness. Each dataset is rigorously evaluated based on popularity, practical adoption, recency and maintenance, documentation and annotation quality, licensing transparency, and scientific contribution. Our review revealed critical gaps in the development of Arabic post-training datasets, including limited task diversity, inconsistent or missing documentation and annotation, and low adoption across the community. Finally, the paper discusses the implications of these gaps on the progress of Arabic LLMs and applications while providing concrete recommendations for future efforts in post-training dataset development.
>
---
#### [new 066] Interaction as Intelligence: Deep Research With Human-AI Partnership
- **分类: cs.CL**

- **简介: 该论文属于人工智能交互任务，旨在解决当前AI系统在深度研究中缺乏有效人机协作的问题。作者提出“交互即智能”理念，设计Deep Cognition系统，实现透明、可控、可中断的交互与认知监督，提升研究效率与协作性。**

- **链接: [http://arxiv.org/pdf/2507.15759v1](http://arxiv.org/pdf/2507.15759v1)**

> **作者:** Lyumanshan Ye; Xiaojie Cai; Xinkai Wang; Junfei Wang; Xiangkun Hu; Jiadi Su; Yang Nan; Sihan Wang; Bohan Zhang; Xiaoze Fan; Jinbin Luo; Yuxiang Zheng; Tianze Xu; Dayuan Fu; Yunze Wu; Pengrui Lu; Zengzhi Wang; Yiwei Qin; Zhen Huang; Yan Ma; Zhulin Hu; Haoyang Zou; Tiantian Mi; Yixin Ye; Ethan Chern; Pengfei Liu
>
> **备注:** 30 pages, 10 figures
>
> **摘要:** This paper introduces "Interaction as Intelligence" research series, presenting a reconceptualization of human-AI relationships in deep research tasks. Traditional approaches treat interaction merely as an interface for accessing AI capabilities-a conduit between human intent and machine output. We propose that interaction itself constitutes a fundamental dimension of intelligence. As AI systems engage in extended thinking processes for research tasks, meaningful interaction transitions from an optional enhancement to an essential component of effective intelligence. Current deep research systems adopt an "input-wait-output" paradigm where users initiate queries and receive results after black-box processing. This approach leads to error cascade effects, inflexible research boundaries that prevent question refinement during investigation, and missed opportunities for expertise integration. To address these limitations, we introduce Deep Cognition, a system that transforms the human role from giving instructions to cognitive oversight-a mode of engagement where humans guide AI thinking processes through strategic intervention at critical junctures. Deep cognition implements three key innovations: (1)Transparent, controllable, and interruptible interaction that reveals AI reasoning and enables intervention at any point; (2)Fine-grained bidirectional dialogue; and (3)Shared cognitive context where the system observes and adapts to user behaviors without explicit instruction. User evaluation demonstrates that this cognitive oversight paradigm outperforms the strongest baseline across six key metrics: Transparency(+20.0%), Fine-Grained Interaction(+29.2%), Real-Time Intervention(+18.5%), Ease of Collaboration(+27.7%), Results-Worth-Effort(+8.8%), and Interruptibility(+20.7%). Evaluations on challenging research problems show 31.8% to 50.0% points of improvements over deep research systems.
>
---
#### [new 067] MUR: Momentum Uncertainty guided Reasoning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理效率。针对测试时扩展方法导致的冗余计算问题，提出MUR方法，通过动量不确定性动态分配推理资源，并引入gamma-control机制调节计算预算，有效减少计算量并提高准确率。**

- **链接: [http://arxiv.org/pdf/2507.14958v1](http://arxiv.org/pdf/2507.14958v1)**

> **作者:** Hang Yan; Fangzhi Xu; Rongman Xu; Yifei Li; Jian Zhang; Haoran Luo; Xiaobao Wu; Luu Anh Tuan; Haiteng Zhao; Qika Lin; Jun Liu
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) have achieved impressive performance on reasoning-intensive tasks, yet optimizing their reasoning efficiency remains an open challenge. While Test-Time Scaling (TTS) improves reasoning quality, it often leads to overthinking, wasting tokens on redundant computations. This work investigates how to efficiently and adaptively guide LLM test-time scaling without additional training. Inspired by the concept of momentum in physics, we propose Momentum Uncertainty-guided Reasoning (MUR), which dynamically allocates thinking budgets to critical reasoning steps by tracking and aggregating stepwise uncertainty over time. To support flexible inference-time control, we introduce gamma-control, a simple mechanism that tunes the reasoning budget via a single hyperparameter. We provide in-depth theoretical proof to support the superiority of MUR in terms of stability and biases. MUR is comprehensively evaluated against various TTS methods across four challenging benchmarks (MATH-500, AIME24, AIME25, and GPQA-diamond) using different sizes of recent Qwen3 models (1.7B, 4B, and 8B). Results demonstrate that MUR reduces computation by over 50% on average while improving accuracy by 0.62-3.37%.
>
---
#### [new 068] What Makes You CLIC: Detection of Croatian Clickbait Headlines
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决克罗地亚语新闻标题中的点击诱饵检测问题。作者构建了包含20年主流与非主流媒体数据的CLIC数据集，基于BERTić模型进行微调，并对比了基于提示的LLM方法效果。结果表明，微调模型优于通用大模型，且近半数标题含点击诱饵。**

- **链接: [http://arxiv.org/pdf/2507.14314v1](http://arxiv.org/pdf/2507.14314v1)**

> **作者:** Marija Anđedelić; Dominik Šipek; Laura Majer; Jan Šnajder
>
> **备注:** Accepted at Slavic NLP 2025
>
> **摘要:** Online news outlets operate predominantly on an advertising-based revenue model, compelling journalists to create headlines that are often scandalous, intriguing, and provocative -- commonly referred to as clickbait. Automatic detection of clickbait headlines is essential for preserving information quality and reader trust in digital media and requires both contextual understanding and world knowledge. For this task, particularly in less-resourced languages, it remains unclear whether fine-tuned methods or in-context learning (ICL) yield better results. In this paper, we compile CLIC, a novel dataset for clickbait detection of Croatian news headlines spanning a 20-year period and encompassing mainstream and fringe outlets. We fine-tune the BERTi\'c model on this task and compare its performance to LLM-based ICL methods with prompts both in Croatian and English. Finally, we analyze the linguistic properties of clickbait. We find that nearly half of the analyzed headlines contain clickbait, and that finetuned models deliver better results than general LLMs.
>
---
#### [new 069] A Case Against Implicit Standards: Homophone Normalization in Machine Translation for Languages that use the Ge'ez Script
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨了使用Ge'ez文字语言的机器翻译中同音字归一化问题。其任务是提升翻译质量，同时保留语言特征。论文提出在推理后处理中应用归一化，而非训练前处理，解决了模型无法理解不同书写形式的问题，并提升了BLEU分数。**

- **链接: [http://arxiv.org/pdf/2507.15142v1](http://arxiv.org/pdf/2507.15142v1)**

> **作者:** Hellina Hailu Nigatu; Atnafu Lambebo Tonja; Henok Biadglign Ademtew; Hizkel Mitiku Alemayehu; Negasi Haile Abadi; Tadesse Destaw Belay; Seid Muhie Yimam
>
> **备注:** Paper under review
>
> **摘要:** Homophone normalization, where characters that have the same sound in a writing script are mapped to one character, is a pre-processing step applied in Amharic Natural Language Processing (NLP) literature. While this may improve performance reported by automatic metrics, it also results in models that are not able to understand different forms of writing in a single language. Further, there might be impacts in transfer learning, where models trained on normalized data do not generalize well to other languages. In this paper, we experiment with monolingual training and cross-lingual transfer to understand the impacts of normalization on languages that use the Ge'ez script. We then propose a post-inference intervention in which normalization is applied to model predictions instead of training data. With our simple scheme of post-inference normalization, we show that we can achieve an increase in BLEU score of up to 1.03 while preserving language features in training. Our work contributes to the broader discussion on technology-facilitated language change and calls for more language-aware interventions.
>
---
#### [new 070] DeepWriter: A Fact-Grounded Multimodal Writing Assistant Based On Offline Knowledge Base
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DeepWriter，一种基于离线知识库的多模态写作助手，旨在解决大型语言模型在专业领域（如金融、医疗、法律）写作中因缺乏领域知识和易产生幻觉而表现不佳的问题。通过任务分解、大纲生成、多模态检索与分段撰写等步骤，提升生成内容的事实准确性与专业性。**

- **链接: [http://arxiv.org/pdf/2507.14189v1](http://arxiv.org/pdf/2507.14189v1)**

> **作者:** Song Mao; Lejun Cheng; Pinlong Cai; Guohang Yan; Ding Wang; Botian Shi
>
> **备注:** work in process
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in various applications. However, their use as writing assistants in specialized domains like finance, medicine, and law is often hampered by a lack of deep domain-specific knowledge and a tendency to hallucinate. Existing solutions, such as Retrieval-Augmented Generation (RAG), can suffer from inconsistency across multiple retrieval steps, while online search-based methods often degrade quality due to unreliable web content. To address these challenges, we introduce DeepWriter, a customizable, multimodal, long-form writing assistant that operates on a curated, offline knowledge base. DeepWriter leverages a novel pipeline that involves task decomposition, outline generation, multimodal retrieval, and section-by-section composition with reflection. By deeply mining information from a structured corpus and incorporating both textual and visual elements, DeepWriter generates coherent, factually grounded, and professional-grade documents. We also propose a hierarchical knowledge representation to enhance retrieval efficiency and accuracy. Our experiments on financial report generation demonstrate that DeepWriter produces high-quality, verifiable articles that surpasses existing baselines in factual accuracy and generated content quality.
>
---
#### [new 071] Let's Measure the Elephant in the Room: Facilitating Personalized Automated Analysis of Privacy Policies at Scale
- **分类: cs.CL; cs.CR; cs.CY**

- **简介: 该论文属于隐私政策分析任务，旨在解决用户难以理解与个性化匹配隐私政策的问题。论文提出了PoliAnalyzer系统，结合NLP与逻辑推理，自动分析隐私政策与用户偏好的合规性，减少用户认知负担，并揭示常见侵犯用户期望的数据实践。**

- **链接: [http://arxiv.org/pdf/2507.14214v1](http://arxiv.org/pdf/2507.14214v1)**

> **作者:** Rui Zhao; Vladyslav Melnychuk; Jun Zhao; Jesse Wright; Nigel Shadbolt
>
> **摘要:** In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites despite claiming otherwise. This paper introduces PoliAnalyzer, a neuro-symbolic system that assists users with personalized privacy policy analysis. PoliAnalyzer uses Natural Language Processing (NLP) to extract formal representations of data usage practices from policy texts. In favor of deterministic, logical inference is applied to compare user preferences with the formal privacy policy representation and produce a compliance report. To achieve this, we extend an existing formal Data Terms of Use policy language to model privacy policies as app policies and user preferences as data policies. In our evaluation using our enriched PolicyIE dataset curated by legal experts, PoliAnalyzer demonstrated high accuracy in identifying relevant data usage practices, achieving F1-score of 90-100% across most tasks. Additionally, we demonstrate how PoliAnalyzer can model diverse user data-sharing preferences, derived from prior research as 23 user profiles, and perform compliance analysis against the top 100 most-visited websites. This analysis revealed that, on average, 95.2% of a privacy policy's segments do not conflict with the analyzed user preferences, enabling users to concentrate on understanding the 4.8% (636 / 13205) that violates preferences, significantly reducing cognitive burden. Further, we identified common practices in privacy policies that violate user expectations - such as the sharing of location data with 3rd parties. This paper demonstrates that PoliAnalyzer can support automated personalized privacy policy analysis at scale using off-the-shelf NLP tools. This sheds light on a pathway to help individuals regain control over their data and encourage societal discussions on platform data practices to promote a fairer power dynamic.
>
---
#### [new 072] Chinchunmei at SemEval-2025 Task 11: Boosting the Large Language Model's Capability of Emotion Perception using Contrastive Learning
- **分类: cs.CL**

- **简介: 该论文参与SemEval-2025 Task 11的情绪检测任务，旨在提升大语言模型在多语言文本中识别情绪的能力。任务包括多标签分类和情绪强度预测。论文通过引入基于对比学习的方法，如样本对比和生成对比，优化模型对情绪表达的理解和预测性能。**

- **链接: [http://arxiv.org/pdf/2507.15714v1](http://arxiv.org/pdf/2507.15714v1)**

> **作者:** Tian Li; Yujian Sun; Huizhi Liang
>
> **摘要:** The SemEval-2025 Task 11, Bridging the Gap in Text-Based Emotion Detection, introduces an emotion recognition challenge spanning over 28 languages. This competition encourages researchers to explore more advanced approaches to address the challenges posed by the diversity of emotional expressions and background variations. It features two tracks: multi-label classification (Track A) and emotion intensity prediction (Track B), covering six emotion categories: anger, fear, joy, sadness, surprise, and disgust. In our work, we systematically explore the benefits of two contrastive learning approaches: sample-based (Contrastive Reasoning Calibration) and generation-based (DPO, SimPO) contrastive learning. The sample-based contrastive approach trains the model by comparing two samples to generate more reliable predictions. The generation-based contrastive approach trains the model to differentiate between correct and incorrect generations, refining its prediction. All models are fine-tuned from LLaMa3-Instruct-8B. Our system achieves 9th place in Track A and 6th place in Track B for English, while ranking among the top-tier performing systems for other languages.
>
---
#### [new 073] Backtranslation and paraphrasing in the LLM era? Comparing data augmentation methods for emotion classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决情感分类中的数据稀缺与类别不平衡问题。通过对比回译、改写与生成方法，利用大语言模型进行数据增强，评估其在分类性能上的效果。**

- **链接: [http://arxiv.org/pdf/2507.14590v1](http://arxiv.org/pdf/2507.14590v1)**

> **作者:** Łukasz Radliński; Mateusz Guściora; Jan Kocoń
>
> **备注:** International Conference on Computational Science 2025
>
> **摘要:** Numerous domain-specific machine learning tasks struggle with data scarcity and class imbalance. This paper systematically explores data augmentation methods for NLP, particularly through large language models like GPT. The purpose of this paper is to examine and evaluate whether traditional methods such as paraphrasing and backtranslation can leverage a new generation of models to achieve comparable performance to purely generative methods. Methods aimed at solving the problem of data scarcity and utilizing ChatGPT were chosen, as well as an exemplary dataset. We conducted a series of experiments comparing four different approaches to data augmentation in multiple experimental setups. We then evaluated the results both in terms of the quality of generated data and its impact on classification performance. The key findings indicate that backtranslation and paraphrasing can yield comparable or even better results than zero and a few-shot generation of examples.
>
---
#### [new 074] A Novel Self-Evolution Framework for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种双阶段自进化框架（DPSE），用于提升大语言模型的用户偏好适配和领域认知能力。通过引入审查模块和两阶段微调策略，联合优化用户满意度与领域知识，解决现有后训练方法在领域认知上的不足。**

- **链接: [http://arxiv.org/pdf/2507.15281v1](http://arxiv.org/pdf/2507.15281v1)**

> **作者:** Haoran Sun; Zekun Zhang; Shaoning Zeng
>
> **摘要:** The capabilities of Large Language Models (LLMs) are limited to some extent by pre-training, so some researchers optimize LLMs through post-training. Existing post-training strategies, such as memory-based retrieval or preference optimization, improve user alignment yet fail to enhance the model's domain cognition. To bridge this gap, we propose a novel Dual-Phase Self-Evolution (DPSE) framework that jointly optimizes user preference adaptation and domain-specific competence. DPSE introduces a Censor module to extract multi-dimensional interaction signals and estimate satisfaction scores, which guide structured data expansion via topic-aware and preference-driven strategies. These expanded datasets support a two-stage fine-tuning pipeline: supervised domain grounding followed by frequency-aware preference optimization. Experiments across general NLP benchmarks and long-term dialogue tasks demonstrate that DPSE consistently outperforms Supervised Fine-Tuning, Preference Optimization, and Memory-Augmented baselines. Ablation studies validate the contribution of each module. In this way, our framework provides an autonomous path toward continual self-evolution of LLMs.
>
---
#### [new 075] Leveraging Context for Multimodal Fallacy Classification in Political Debates
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与了MM-ArgFallacy2025共享任务，旨在推动政治辩论中逻辑谬误的多模态论点挖掘研究。论文使用预训练Transformer模型，结合上下文信息，分别在文本、音频和多模态条件下进行谬误分类。结果表明，多模态模型表现与纯文本模型相当，具有改进潜力。**

- **链接: [http://arxiv.org/pdf/2507.15641v1](http://arxiv.org/pdf/2507.15641v1)**

> **作者:** Alessio Pittiglio
>
> **备注:** 12th Workshop on Argument Mining (ArgMining 2025) @ ACL 2025
>
> **摘要:** In this paper, we present our submission to the MM-ArgFallacy2025 shared task, which aims to advance research in multimodal argument mining, focusing on logical fallacies in political debates. Our approach uses pretrained Transformer-based models and proposes several ways to leverage context. In the fallacy classification subtask, our models achieved macro F1-scores of 0.4444 (text), 0.3559 (audio), and 0.4403 (multimodal). Our multimodal model showed performance comparable to the text-only model, suggesting potential for improvements.
>
---
#### [new 076] Doc2Chart: Intent-Driven Zero-Shot Chart Generation from Documents
- **分类: cs.CL**

- **简介: 该论文属于图表生成任务，旨在解决从长文档中根据用户意图自动生成图表的问题。现有方法需手动筛选数据，难以应对真实场景。论文提出Doc2Chart框架，分两阶段实现零样本图表生成：先用大模型提取并验证数据，再通过启发式方法选择图表类型并生成代码。设计了基于归因的评估指标，并构建了包含金融和科学领域的数据集验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.14819v1](http://arxiv.org/pdf/2507.14819v1)**

> **作者:** Akriti Jain; Pritika Ramu; Aparna Garimella; Apoorv Saxena
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong capabilities in transforming text descriptions or tables to data visualizations via instruction-tuning methods. However, it is not straightforward to apply these methods directly for a more real-world use case of visualizing data from long documents based on user-given intents, as opposed to the user pre-selecting the relevant content manually. We introduce the task of intent-based chart generation from documents: given a user-specified intent and document(s), the goal is to generate a chart adhering to the intent and grounded on the document(s) in a zero-shot setting. We propose an unsupervised, two-staged framework in which an LLM first extracts relevant information from the document(s) by decomposing the intent and iteratively validates and refines this data. Next, a heuristic-guided module selects an appropriate chart type before final code generation. To assess the data accuracy of the generated charts, we propose an attribution-based metric that uses a structured textual representation of charts, instead of relying on visual decoding metrics that often fail to capture the chart data effectively. To validate our approach, we curate a dataset comprising of 1,242 $<$intent, document, charts$>$ tuples from two domains, finance and scientific, in contrast to the existing datasets that are largely limited to parallel text descriptions/ tables and their corresponding charts. We compare our approach with baselines using single-shot chart generation using LLMs and query-based retrieval methods; our method outperforms by upto $9$ points and $17$ points in terms of chart data accuracy and chart type respectively over the best baselines.
>
---
#### [new 077] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）如何理解叙述中的时间意义，属于自然语言处理与认知科学交叉任务。旨在解决LLMs是否具备类似人类的时间语义理解能力的问题。作者通过专家参与的探针实验，分析LLMs在时间体判断、语义表示和因果推理方面的表现，发现其依赖原型、判断不一致，理解叙事能力不足，并提出了标准化评估框架。**

- **链接: [http://arxiv.org/pdf/2507.14307v1](http://arxiv.org/pdf/2507.14307v1)**

> **作者:** Karin de Langis; Jong Inn Park; Andreas Schramm; Bin Hu; Khanh Chi Le; Michael Mensink; Ahn Thu Tong; Dongyeop Kang
>
> **摘要:** Large language models (LLMs) exhibit increasingly sophisticated linguistic capabilities, yet the extent to which these behaviors reflect human-like cognition versus advanced pattern recognition remains an open question. In this study, we investigate how LLMs process the temporal meaning of linguistic aspect in narratives that were previously used in human studies. Using an Expert-in-the-Loop probing pipeline, we conduct a series of targeted experiments to assess whether LLMs construct semantic representations and pragmatic inferences in a human-like manner. Our findings show that LLMs over-rely on prototypicality, produce inconsistent aspectual judgments, and struggle with causal reasoning derived from aspect, raising concerns about their ability to fully comprehend narratives. These results suggest that LLMs process aspect fundamentally differently from humans and lack robust narrative understanding. Beyond these empirical findings, we develop a standardized experimental framework for the reliable assessment of LLMs' cognitive and linguistic capabilities.
>
---
#### [new 078] Evaluation of Coding Schemes for Transformer-based Gene Sequence Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于基因序列建模任务，旨在比较不同编码方案在Transformer模型中的性能。研究评估了k-mer分段、BPE子词分词及位置编码方法，探讨其对模型效果的影响，并为DNA序列建模提供优化建议。**

- **链接: [http://arxiv.org/pdf/2507.15087v1](http://arxiv.org/pdf/2507.15087v1)**

> **作者:** Chenlei Gong; Yuanhe Tian; Lei Mao; Yan Song
>
> **摘要:** Currently, many studies view DNA sequences as a special type of language and utilize Transformers to model them. These studies use fixed-length k-mer segmentation and BPE subword tokenization but lack a systematic evaluation to determine which is superior. We compare k-mer segmentation with k=1,3,4,5,6, a 4,096-token BPE vocabulary, and three positional encoding methods-sinusoidal, AliBi, and RoPE. Each configuration is trained from scratch in 3, 6, 12, and 24-layer Transformer encoders and evaluated on GUE benchmark dataset. In general, BPE delivers higher and more stable performance across tasks by compressing frequent motifs into variable-length tokens, reducing sequence length, and improving model generalization. RoPE excels at capturing periodic motifs and extrapolating to long sequences, while AliBi also performs well on tasks driven by local dependencies. In terms of depth, we observe significant gains when increasing layers from 3 to 12, with only marginal improvements or slight overfitting at 24 layers. This study provides practical guidance for designing tokenization and positional encoding in DNA Transformer models.
>
---
#### [new 079] The Impact of Language Mixing on Bilingual LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言混合对双语大模型推理的影响。任务是分析中文-英文双语模型在推理过程中切换语言的作用。论文发现语言混合可提升推理能力，抑制则效果下降。通过RLVR训练阶段实现语言切换控制，使用探针预测切换效果，指导解码提升准确率，表明语言混合是策略性行为而非训练副产物。**

- **链接: [http://arxiv.org/pdf/2507.15849v1](http://arxiv.org/pdf/2507.15849v1)**

> **作者:** Yihao Li; Jiayi Xin; Miranda Muqing Miao; Qi Long; Lyle Ungar
>
> **摘要:** Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior.
>
---
#### [new 080] Understanding Large Language Models' Ability on Interdisciplinary Research
- **分类: cs.CL**

- **简介: 该论文属于科学评估任务，旨在解决缺乏评估大语言模型（LLM）在跨学科研究（IDR）中生成研究思路能力的基准问题。作者构建了IDRBench基准，包含专家标注的数据集和多项任务，用以系统评估LLMs在跨学科科研中的表现，并发现现有LLMs在生成高质量跨学科研究思路方面仍存在不足。**

- **链接: [http://arxiv.org/pdf/2507.15736v1](http://arxiv.org/pdf/2507.15736v1)**

> **作者:** Yuanhao Shen; Daniel Xavier de Sousa; Ricardo Marçal; Ali Asad; Hongyu Guo; Xiaodan Zhu
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have revealed their impressive ability to perform multi-step, logic-driven reasoning across complex domains, positioning them as powerful tools and collaborators in scientific discovery while challenging the long-held view that inspiration-driven ideation is uniquely human. However, the lack of a dedicated benchmark that evaluates LLMs' ability to develop ideas in Interdisciplinary Research (IDR) settings poses a critical barrier to fully understanding their strengths and limitations. To address this gap, we introduce IDRBench -- a pioneering benchmark featuring an expert annotated dataset and a suite of tasks tailored to evaluate LLMs' capabilities in proposing valuable research ideas from different scientific domains for interdisciplinary research. This benchmark aims to provide a systematic framework for assessing LLM performance in complex, cross-domain scientific research. Our dataset consists of scientific publications sourced from the ArXiv platform covering six distinct disciplines, and is annotated by domain experts with diverse academic backgrounds. To ensure high-quality annotations, we emphasize clearly defined dimensions that characterize authentic interdisciplinary research. The design of evaluation tasks in IDRBench follows a progressive, real-world perspective, reflecting the natural stages of interdisciplinary research development, including 1) IDR Paper Identification, 2) IDR Idea Integration, and 3) IDR Idea Recommendation. Using IDRBench, we construct baselines across 10 LLMs and observe that despite fostering some level of IDR awareness, LLMs still struggle to produce quality IDR ideas. These findings could not only spark new research directions, but also help to develop next-generation LLMs that excel in interdisciplinary research.
>
---
#### [new 081] XL-DURel: Finetuning Sentence Transformers for Ordinal Word-in-Context Classification
- **分类: cs.CL**

- **简介: 论文提出XL-DURel，一种用于序数词义分类的多语言句子表示模型。通过优化排序目标下的损失函数，提升词义分类性能，表明二分类可视为序数分类特例，统一建模方法可改善不同任务表现。**

- **链接: [http://arxiv.org/pdf/2507.14578v1](http://arxiv.org/pdf/2507.14578v1)**

> **作者:** Sachin Yadav; Dominik Schlechtweg
>
> **备注:** 8 pages
>
> **摘要:** We propose XL-DURel, a finetuned, multilingual Sentence Transformer model optimized for ordinal Word-in-Context classification. We test several loss functions for regression and ranking tasks managing to outperform previous models on ordinal and binary data with a ranking objective based on angular distance in complex space. We further show that binary WiC can be treated as a special case of ordinal WiC and that optimizing models for the general ordinal task improves performance on the more specific binary task. This paves the way for a unified treatment of WiC modeling across different task formulations.
>
---
#### [new 082] Tiny language models
- **分类: cs.CL**

- **简介: 该论文研究小型语言模型（TLM）是否具备大语言模型（LLM）的关键特性。任务是分类任务，旨在解决LLM因资源消耗大而难以普及的问题。作者通过预训练BERT-6和BERT-1变体模型，验证了TLM在分类任务中的有效性，并提出浅层模型组合可实现低延迟推理。**

- **链接: [http://arxiv.org/pdf/2507.14871v1](http://arxiv.org/pdf/2507.14871v1)**

> **作者:** Ronit D. Gross; Yarden Tzach; Tal Halevi; Ella Koresh; Ido Kanter
>
> **备注:** 23 pages, 1 figure and 12 tables
>
> **摘要:** A prominent achievement of natural language processing (NLP) is its ability to understand and generate meaningful human language. This capability relies on complex feedforward transformer block architectures pre-trained on large language models (LLMs). However, LLM pre-training is currently feasible only for a few dominant companies due to the immense computational resources required, limiting broader research participation. This creates a critical need for more accessible alternatives. In this study, we explore whether tiny language models (TLMs) exhibit the same key qualitative features of LLMs. We demonstrate that TLMs exhibit a clear performance gap between pre-trained and non-pre-trained models across classification tasks, indicating the effectiveness of pre-training, even at a tiny scale. The performance gap increases with the size of the pre-training dataset and with greater overlap between tokens in the pre-training and classification datasets. Furthermore, the classification accuracy achieved by a pre-trained deep TLM architecture can be replicated through a soft committee of multiple, independently pre-trained shallow architectures, enabling low-latency TLMs without affecting classification accuracy. Our results are based on pre-training BERT-6 and variants of BERT-1 on subsets of the Wikipedia dataset and evaluating their performance on FewRel, AGNews, and DBPedia classification tasks. Future research on TLM is expected to further illuminate the mechanisms underlying NLP, especially given that its biologically inspired models suggest that TLMs may be sufficient for children or adolescents to develop language.
>
---
#### [new 083] Large Language Models as Medical Codes Selectors: a benchmark using the International Classification of Primary Care
- **分类: cs.CL**

- **简介: 该论文属于医疗自然语言处理任务，旨在解决自动化分配ICPC-2编码的问题。研究使用大型语言模型结合语义搜索引擎，对437条葡萄牙语临床文本进行编码选择，并评估模型在准确率、格式合规性等方面的表现。结果显示，多个模型表现优异，具备自动化编码潜力。**

- **链接: [http://arxiv.org/pdf/2507.14681v1](http://arxiv.org/pdf/2507.14681v1)**

> **作者:** Vinicius Anjos de Almeida; Vinicius de Camargo; Raquel Gómez-Bravo; Egbert van der Haring; Kees van Boven; Marcelo Finger; Luis Fernandez Lopez
>
> **备注:** To be submitted to peer-reviewed journal. 33 pages, 10 figures (including appendix), 15 tables (including appendix). For associated code repository, see https://github.com/almeidava93/llm-as-code-selectors-paper
>
> **摘要:** Background: Medical coding structures healthcare data for research, quality monitoring, and policy. This study assesses the potential of large language models (LLMs) to assign ICPC-2 codes using the output of a domain-specific search engine. Methods: A dataset of 437 Brazilian Portuguese clinical expressions, each annotated with ICPC-2 codes, was used. A semantic search engine (OpenAI's text-embedding-3-large) retrieved candidates from 73,563 labeled concepts. Thirty-three LLMs were prompted with each query and retrieved results to select the best-matching ICPC-2 code. Performance was evaluated using F1-score, along with token usage, cost, response time, and format adherence. Results: Twenty-eight models achieved F1-score > 0.8; ten exceeded 0.85. Top performers included gpt-4.5-preview, o3, and gemini-2.5-pro. Retriever optimization can improve performance by up to 4 points. Most models returned valid codes in the expected format, with reduced hallucinations. Smaller models (<3B) struggled with formatting and input length. Conclusions: LLMs show strong potential for automating ICPC-2 coding, even without fine-tuning. This work offers a benchmark and highlights challenges, but findings are limited by dataset scope and setup. Broader, multilingual, end-to-end evaluations are needed for clinical validation.
>
---
#### [new 084] What Level of Automation is "Good Enough"? A Benchmark of Large Language Models for Meta-Analysis Data Extraction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数据提取任务，旨在解决从医学随机对照试验中自动化提取数据用于荟萃分析的问题。研究评估了三种大语言模型在不同医学领域的性能，并测试了四种提示策略以提升提取质量，最终提出分级指南，以平衡自动化效率与专家监督。**

- **链接: [http://arxiv.org/pdf/2507.15152v1](http://arxiv.org/pdf/2507.15152v1)**

> **作者:** Lingbo Li; Anuradha Mathrani; Teo Susnjak
>
> **摘要:** Automating data extraction from full-text randomised controlled trials (RCTs) for meta-analysis remains a significant challenge. This study evaluates the practical performance of three LLMs (Gemini-2.0-flash, Grok-3, GPT-4o-mini) across tasks involving statistical results, risk-of-bias assessments, and study-level characteristics in three medical domains: hypertension, diabetes, and orthopaedics. We tested four distinct prompting strategies (basic prompting, self-reflective prompting, model ensemble, and customised prompts) to determine how to improve extraction quality. All models demonstrate high precision but consistently suffer from poor recall by omitting key information. We found that customised prompts were the most effective, boosting recall by up to 15\%. Based on this analysis, we propose a three-tiered set of guidelines for using LLMs in data extraction, matching data types to appropriate levels of automation based on task complexity and risk. Our study offers practical advice for automating data extraction in real-world meta-analyses, balancing LLM efficiency with expert oversight through targeted, task-specific automation.
>
---
#### [new 085] RefCritic: Training Long Chain-of-Thought Critic Models with Refinement Feedback
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的批评能力。现有监督微调方法生成的批评质量不高，作者提出RefCritic，通过强化学习结合双规则奖励机制，生成高质量、可操作的反馈，以有效指导模型优化。实验表明其在多个基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.15024v1](http://arxiv.org/pdf/2507.15024v1)**

> **作者:** Qiaoyu Tang; Hao Xiang; Le Yu; Bowen Yu; Hongyu Lin; Yaojie Lu; Xianpei Han; Le Sun; Junyang Lin
>
> **摘要:** With the rapid advancement of Large Language Models (LLMs), developing effective critic modules for precise guidance has become crucial yet challenging. In this paper, we initially demonstrate that supervised fine-tuning for building critic modules (which is widely adopted in current solutions) fails to genuinely enhance models' critique abilities, producing superficial critiques with insufficient reflections and verifications. To unlock the unprecedented critique capabilities, we propose RefCritic, a long-chain-of-thought critic module based on reinforcement learning with dual rule-based rewards: (1) instance-level correctness of solution judgments and (2) refinement accuracies of the policy model based on critiques, aiming to generate high-quality evaluations with actionable feedback that effectively guides model refinement. We evaluate RefCritic on Qwen2.5-14B-Instruct and DeepSeek-R1-Distill-Qwen-14B across five benchmarks. On critique and refinement settings, RefCritic demonstrates consistent advantages across all benchmarks, e.g., 6.8\% and 7.2\% gains on AIME25 for the respective base models. Notably, under majority voting, policy models filtered by RefCritic show superior scaling with increased voting numbers. Moreover, despite training on solution-level supervision, RefCritic outperforms step-level supervised approaches on ProcessBench, a benchmark to identify erroneous steps in mathematical reasoning.
>
---
#### [new 086] Smart Eyes for Silent Threats: VLMs and In-Context Learning for THz Imaging
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决太赫兹（THz）成像中因标注数据少、分辨率低和视觉模糊导致的分类难题。作者采用无需微调的视觉-语言模型（VLMs）结合上下文学习（ICL），通过模态对齐提示框架，在零样本和一样本设置下提升了分类效果。这是首次将ICL增强的VLMs应用于THz成像领域。**

- **链接: [http://arxiv.org/pdf/2507.15576v1](http://arxiv.org/pdf/2507.15576v1)**

> **作者:** Nicolas Poggi; Shashank Agnihotri; Margret Keuper
>
> **摘要:** Terahertz (THz) imaging enables non-invasive analysis for applications such as security screening and material classification, but effective image classification remains challenging due to limited annotations, low resolution, and visual ambiguity. We introduce In-Context Learning (ICL) with Vision-Language Models (VLMs) as a flexible, interpretable alternative that requires no fine-tuning. Using a modality-aligned prompting framework, we adapt two open-weight VLMs to the THz domain and evaluate them under zero-shot and one-shot settings. Our results show that ICL improves classification and interpretability in low-data regimes. This is the first application of ICL-enhanced VLMs to THz imaging, offering a promising direction for resource-constrained scientific domains. Code: \href{https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main}{GitHub repository}.
>
---
#### [new 087] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决长上下文建模中的内存瓶颈问题。论文提出LaCache，通过阶梯状KV缓存结构与迭代压缩机制，在固定内存预算下提升模型长程依赖捕捉能力与连续生成效率。**

- **链接: [http://arxiv.org/pdf/2507.14204v1](http://arxiv.org/pdf/2507.14204v1)**

> **作者:** Dachuan Shi; Yonggan Fu; Xiangchi Yuan; Zhongzhi Yu; Haoran You; Sixu Li; Xin Dong; Jan Kautz; Pavlo Molchanov; Yingyan; Lin
>
> **备注:** ICML 2025. Code: https://github.com/GATECH-EIC/LaCache
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have spurred interest in numerous applications requiring robust long-range capabilities, essential for processing extensive input contexts and continuously generating extended outputs. As sequence lengths increase, the number of Key-Value (KV) pairs in LLMs escalates, creating a significant efficiency bottleneck. In this paper, we propose a new KV cache optimization paradigm called LaCache, a training-free method for efficient and accurate generative inference of LLMs. LaCache enables LLMs to simultaneously address both of the critical challenges in long-range modeling: robust long-range capabilities and continuous generation without running out-of-memory (OOM). Specifically, LaCache integrates two key innovations: (1) a ladder-shaped KV cache pattern that stores KV pairs not only sequentially (left-to-right within each layer) but also across layers (from shallow to deep), providing an extended span for capturing long-range dependencies under a fixed storage budget, thereby boosting long-range capabilities; and (2) an iterative compaction mechanism that progressively compresses older caches, freeing up space for new tokens within a fixed cache size. This token distance-based dynamic compression enables more effective continuous generation under constrained cache budgets. Experiments across various tasks, benchmarks, and LLM models consistently validate LaCache's effectiveness in enhancing LLMs' long-range capabilities. Our code is available at https://github.com/GATECH-EIC/LaCache.
>
---
#### [new 088] Solo Connection: A Parameter Efficient Fine-Tuning Technique for Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决大语言模型微调时参数效率低的问题。作者提出新方法Solo Connection，通过在解码器块级别调整表示，而非修改权重矩阵，显著减少可训练参数，并基于同伦理论引入线性变换实现稳定适配。**

- **链接: [http://arxiv.org/pdf/2507.14353v1](http://arxiv.org/pdf/2507.14353v1)**

> **作者:** Harsh Nilesh Pathak; Randy Paffenroth
>
> **摘要:** Parameter efficient fine tuning (PEFT) is a versatile and extensible approach for adapting a Large Language Model (LLM) for newer tasks. One of the most prominent PEFT approaches, Low Rank Adaptation (LoRA), primarily focuses on adjusting the attention weight matrices within individual decoder blocks of a Generative Pre trained Transformer (GPT2). In contrast, we introduce Solo Connection a novel method that adapts the representation at the decoder-block level rather than modifying individual weight matrices. Not only does Solo Connection outperform LoRA on E2E natural language generation benchmarks, but it also reduces the number of trainable parameters by 59% relative to LoRA and by more than 99% compared to full fine-tuning of GPT2, an early version of Large Language Models (LLMs). Solo Connection is also motivated by homotopy theory: we introduce a trainable linear transformation that gradually interpolates between a zero vector and the task-specific representation, enabling smooth and stable adaptation over time. While skip connections in the original 12 layer GPT2 are typically confined to individual decoder blocks, subsequent GPT2 variants scale up to 48 layers, and even larger language models can include 128 or more decoder blocks. These expanded architectures underscore the need to revisit how skip connections are employed during fine-tuning. This paper focuses on long skip connections that link outputs of different decoder blocks, potentially enhancing the model's ability to adapt to new tasks while leveraging pre-trained knowledge.
>
---
#### [new 089] LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理时生成过长无效文本的问题。作者提出LAPO方法，通过两阶段强化学习让模型自主掌握合理推理长度，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.15758v1](http://arxiv.org/pdf/2507.15758v1)**

> **作者:** Xingyu Wu; Yuchen Yan; Shangke Lyu; Linjuan Wu; Yiwen Qiu; Yongliang Shen; Weiming Lu; Jian Shao; Jun Xiao; Yueting Zhuang
>
> **备注:** GitHub:https://github.com/zju-real/lapo; Project:https://zju-real.github.io/lapo
>
> **摘要:** Large reasoning models have achieved remarkable performance through extended chain-of-thought sequences, yet this computational freedom leads to excessive token generation even for simple problems. We present Length-Adaptive Policy Optimization (LAPO), a novel framework that transforms reasoning length control from an external constraint into an intrinsic model capability. Unlike existing approaches that impose rigid limits or rely on post-hoc interventions, LAPO enables models to internalize an understanding of appropriate reasoning depth through a two-stage reinforcement learning process. In the first stage, models learn natural reasoning patterns by discovering the statistical distribution of successful solution lengths. The second stage leverages these patterns as meta-cognitive guidance, embedding them directly within the model's reasoning context to ensure inference-time flexibility. Experiments on mathematical reasoning benchmarks demonstrate that LAPO reduces token usage by up to 40.9\% while improving accuracy by 2.3\%. Our analysis reveals that models trained with LAPO develop emergent abilities to allocate computational resources based on problem complexity, achieving efficient reasoning without sacrificing quality.
>
---
#### [new 090] GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于GUI交互任务，旨在解决自然语言到界面位置映射中奖励信号稀疏的问题。作者提出GUI-G²框架，用高斯分布建模界面元素，通过点奖励与覆盖奖励机制，实现连续优化，显著提升交互准确性。**

- **链接: [http://arxiv.org/pdf/2507.15846v1](http://arxiv.org/pdf/2507.15846v1)**

> **作者:** Fei Tang; Zhangxuan Gu; Zhengxi Lu; Xuyang Liu; Shuheng Shen; Changhua Meng; Wen Wang; Wenqi Zhang; Yongliang Shen; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks.
>
---
#### [new 091] Docopilot: Improving Multimodal Models for Document-Level Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态文档理解任务，旨在解决现有模型在复杂多页文档理解上的不足。作者构建了一个高质量文档级数据集Doc-750K，并基于此开发了多模态模型Docopilot，有效处理文档级依赖，提升了理解效果。**

- **链接: [http://arxiv.org/pdf/2507.14675v1](http://arxiv.org/pdf/2507.14675v1)**

> **作者:** Yuchen Duan; Zhe Chen; Yusong Hu; Weiyun Wang; Shenglong Ye; Botian Shi; Lewei Lu; Qibin Hou; Tong Lu; Hongsheng Li; Jifeng Dai; Wenhai Wang
>
> **摘要:** Despite significant progress in multimodal large language models (MLLMs), their performance on complex, multi-page document comprehension remains inadequate, largely due to the lack of high-quality, document-level datasets. While current retrieval-augmented generation (RAG) methods offer partial solutions, they suffer from issues, such as fragmented retrieval contexts, multi-stage error accumulation, and extra time costs of retrieval. In this work, we present a high-quality document-level dataset, Doc-750K, designed to support in-depth understanding of multimodal documents. This dataset includes diverse document structures, extensive cross-page dependencies, and real question-answer pairs derived from the original documents. Building on the dataset, we develop a native multimodal model, Docopilot, which can accurately handle document-level dependencies without relying on RAG. Experiments demonstrate that Docopilot achieves superior coherence, accuracy, and efficiency in document understanding tasks and multi-turn interactions, setting a new baseline for document-level multimodal understanding. Data, code, and models are released at https://github.com/OpenGVLab/Docopilot
>
---
#### [new 092] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出了一种用于零样本在线语音转换的模型Conan，旨在实现实时语音转换中内容保留、音色和风格匹配。论文属于语音处理任务，解决现有模型在实时性、语义保真和风格适应上的不足，通过设计流式内容提取、自适应风格编码和因果声码器组件提升效果。**

- **链接: [http://arxiv.org/pdf/2507.14534v1](http://arxiv.org/pdf/2507.14534v1)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
#### [new 093] Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究小规模大语言模型（LLMs）能否通过强化学习（RL）获得可泛化的心理理论（ToM）能力。任务是评估模型在不同ToM数据集上的泛化表现。作者发现，尽管在训练数据上表现提升，但模型未能掌握抽象的ToM，反而出现过拟合，表明RL未使其真正理解心智理论。**

- **链接: [http://arxiv.org/pdf/2507.15788v1](http://arxiv.org/pdf/2507.15788v1)**

> **作者:** Sneheel Sarangi; Hanan Salam
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated emergent capabilities in complex reasoning, largely spurred by rule-based Reinforcement Learning (RL) techniques applied during the post-training. This has raised the question of whether similar methods can instill more nuanced, human-like social intelligence, such as a Theory of Mind (ToM), in LLMs. This paper investigates whether small-scale LLMs can acquire a robust and generalizable ToM capability through RL with verifiable rewards (RLVR). We conduct a systematic evaluation by training models on various combinations of prominent ToM datasets (HiToM, ExploreToM, FANToM) and testing for generalization on held-out datasets (e.g., OpenToM). Our findings indicate that small LLMs struggle to develop a generic ToM capability. While performance on in-distribution tasks improves, this capability fails to transfer to unseen ToM tasks with different characteristics. Furthermore, we demonstrate that prolonged RL training leads to models ``hacking'' the statistical patterns of the training datasets, resulting in significant performance gains on in-domain data but no change, or degradation of performance on out-of-distribution tasks. This suggests the learned behavior is a form of narrow overfitting rather than the acquisition of a true, abstract ToM capability.
>
---
#### [new 094] Hierarchical Budget Policy Optimization for Adaptive Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理时计算效率低的问题。现有方法因采用统一推理策略，导致在简单问题上浪费资源，在复杂问题上表现不足。论文提出HBPO框架，通过分层预算策略和差异化奖励机制，使模型自适应调整推理深度。实验表明，该方法显著减少计算资源使用并提升准确率，实现了效率与能力的平衡。**

- **链接: [http://arxiv.org/pdf/2507.15844v1](http://arxiv.org/pdf/2507.15844v1)**

> **作者:** Shangke Lyu; Linjuan Wu; Yuchen Yan; Xingyu Wu; Hao Li; Yongliang Shen; Peisheng Jiang; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Code: https://github.com/zju-real/hbpo Project Page:https://zju-real.github.io/hbpo/
>
> **摘要:** Large reasoning models achieve remarkable performance through extensive chain-of-thought generation, yet exhibit significant computational inefficiency by applying uniform reasoning strategies regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. HBPO addresses the fundamental challenge of exploration space collapse in efficiency-oriented training, where penalties on long output length systematically bias models away from necessary long reasoning paths. Through hierarchical budget exploration, our approach partitions rollout samples into multiple subgroups with distinct token budgets, aiming to enable efficient resource allocation while preventing degradation of capability. We introduce differentiated reward mechanisms that create budget-aware incentives aligned with the complexity of the problem, allowing models to discover natural correspondences between task requirements and computational effort. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Unlike existing methods that impose external constraints or rely on discrete mode selection, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity.
>
---
#### [new 095] Hear Your Code Fail, Voice-Assisted Debugging for Python
- **分类: cs.PL; cs.CL**

- **简介: 该论文属于软件调试辅助任务，旨在解决传统调试中错误信息不易理解、调试效率低的问题。论文开发了一个基于语音辅助的Python调试插件，通过语音和可视化方式同步反馈错误信息，降低认知负荷，提升错误定位与修复效率，并提升编程可及性与学习效果。**

- **链接: [http://arxiv.org/pdf/2507.15007v1](http://arxiv.org/pdf/2507.15007v1)**

> **作者:** Sayed Mahbub Hasan Amiri; Md. Mainul Islam; Mohammad Shakhawat Hossen; Sayed Majhab Hasan Amiri; Mohammad Shawkat Ali Mamun; Sk. Humaun Kabir; Naznin Akter
>
> **备注:** 35 pages, 20 figures
>
> **摘要:** This research introduces an innovative voice-assisted debugging plugin for Python that transforms silent runtime errors into actionable audible diagnostics. By implementing a global exception hook architecture with pyttsx3 text-to-speech conversion and Tkinter-based GUI visualization, the solution delivers multimodal error feedback through parallel auditory and visual channels. Empirical evaluation demonstrates 37% reduced cognitive load (p<0.01, n=50) compared to traditional stack-trace debugging, while enabling 78% faster error identification through vocalized exception classification and contextualization. The system achieves sub-1.2 second voice latency with under 18% CPU overhead during exception handling, vocalizing error types and consequences while displaying interactive tracebacks with documentation deep links. Criteria validate compatibility across Python 3.7+ environments on Windows, macOS, and Linux platforms. Needing only two lines of integration code, the plugin significantly boosts availability for aesthetically impaired designers and supports multitasking workflows through hands-free error medical diagnosis. Educational applications show particular promise, with pilot studies indicating 45% faster debugging skill acquisition among novice programmers. Future development will incorporate GPT-based repair suggestions and real-time multilingual translation to further advance auditory debugging paradigms. The solution represents a fundamental shift toward human-centric error diagnostics, bridging critical gaps in programming accessibility while establishing new standards for cognitive efficiency in software development workflows.
>
---
#### [new 096] A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering
- **分类: cs.LG; cs.AI; cs.CL; cs.DC**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理中计算成本高的问题。通过聚类激活模式，预测神经元稀疏激活情况，减少计算开销。论文提出一种基于聚类的压缩框架，有效提升预测效率并保持模型性能。**

- **链接: [http://arxiv.org/pdf/2507.14179v1](http://arxiv.org/pdf/2507.14179v1)**

> **作者:** Nobel Dhar; Bobin Deng; Md Romyull Islam; Xinyue Zhang; Kazi Fahim Ahmad Nasif; Kun Suo
>
> **备注:** To be published in Euro-Par 2025
>
> **摘要:** Large Language Models (LLMs) exhibit significant activation sparsity, where only a subset of neurons are active for a given input. Although this sparsity presents opportunities to reduce computational cost, efficiently utilizing it requires predicting activation patterns in a scalable manner. However, direct prediction at the neuron level is computationally expensive due to the vast number of neurons in modern LLMs. To enable efficient prediction and utilization of activation sparsity, we propose a clustering-based activation pattern compression framework. Instead of treating each neuron independently, we group similar activation patterns into a small set of representative clusters. Our method achieves up to 79.34% clustering precision, outperforming standard binary clustering approaches while maintaining minimal degradation in perplexity (PPL) scores. With a sufficiently large number of clusters, our approach attains a PPL score as low as 12.49, demonstrating its effectiveness in preserving model quality while reducing computational overhead. By predicting cluster assignments rather than individual neuron states, future models can efficiently infer activation patterns from pre-computed centroids. We detail the clustering algorithm, analyze its effectiveness in capturing meaningful activation structures, and demonstrate its potential to improve sparse computation efficiency. This clustering-based formulation serves as a foundation for future work on activation pattern prediction, paving the way for efficient inference in large-scale language models.
>
---
#### [new 097] Assessing the Reliability of Large Language Models for Deductive Qualitative Coding: A Comparative Study of ChatGPT Interventions
- **分类: cs.HC; cs.CL**

- **简介: 该论文研究大型语言模型（如ChatGPT）在演绎式定性编码任务中的可靠性。任务是将美国最高法院案例摘要分类到21个政策领域。论文测试了四种干预方法，评估分类性能与一致性。结果显示逐步任务分解策略表现最佳，具备较高可靠性，表明适当干预下LLMs可应用于严谨的定性编码工作。**

- **链接: [http://arxiv.org/pdf/2507.14384v1](http://arxiv.org/pdf/2507.14384v1)**

> **作者:** Angjelin Hila; Elliott Hauser
>
> **备注:** Extended version of paper accepted for presentation at the ASIS&T Annual Meeting 2025. 38 pages, 12 figures
>
> **摘要:** In this study, we investigate the use of large language models (LLMs), specifically ChatGPT, for structured deductive qualitative coding. While most current research emphasizes inductive coding applications, we address the underexplored potential of LLMs to perform deductive classification tasks aligned with established human-coded schemes. Using the Comparative Agendas Project (CAP) Master Codebook, we classified U.S. Supreme Court case summaries into 21 major policy domains. We tested four intervention methods: zero-shot, few-shot, definition-based, and a novel Step-by-Step Task Decomposition strategy, across repeated samples. Performance was evaluated using standard classification metrics (accuracy, F1-score, Cohen's kappa, Krippendorff's alpha), and construct validity was assessed using chi-squared tests and Cramer's V. Chi-squared and effect size analyses confirmed that intervention strategies significantly influenced classification behavior, with Cramer's V values ranging from 0.359 to 0.613, indicating moderate to strong shifts in classification patterns. The Step-by-Step Task Decomposition strategy achieved the strongest reliability (accuracy = 0.775, kappa = 0.744, alpha = 0.746), achieving thresholds for substantial agreement. Despite the semantic ambiguity within case summaries, ChatGPT displayed stable agreement across samples, including high F1 scores in low-support subclasses. These findings demonstrate that with targeted, custom-tailored interventions, LLMs can achieve reliability levels suitable for integration into rigorous qualitative coding workflows.
>
---
#### [new 098] When Autonomy Goes Rogue: Preparing for Risks of Multi-Agent Collusion in Social Systems
- **分类: cs.AI; cs.CL**

- **简介: 论文研究多智能体系统在社交系统中的恶意共谋风险，属于人工智能安全任务。旨在揭示自主AI群体可能带来的危害，特别是在去中心化结构下更易规避干预、造成更大破坏的问题。论文构建模拟框架，分析其在虚假信息传播和电商欺诈中的行为，提出需加强检测与应对策略。**

- **链接: [http://arxiv.org/pdf/2507.14660v1](http://arxiv.org/pdf/2507.14660v1)**

> **作者:** Qibing Ren; Sitao Xie; Longxuan Wei; Zhenfei Yin; Junchi Yan; Lizhuang Ma; Jing Shao
>
> **备注:** Code is available at https://github.com/renqibing/RogueAgent
>
> **摘要:** Recent large-scale events like election fraud and financial scams have shown how harmful coordinated efforts by human groups can be. With the rise of autonomous AI systems, there is growing concern that AI-driven groups could also cause similar harm. While most AI safety research focuses on individual AI systems, the risks posed by multi-agent systems (MAS) in complex real-world situations are still underexplored. In this paper, we introduce a proof-of-concept to simulate the risks of malicious MAS collusion, using a flexible framework that supports both centralized and decentralized coordination structures. We apply this framework to two high-risk fields: misinformation spread and e-commerce fraud. Our findings show that decentralized systems are more effective at carrying out malicious actions than centralized ones. The increased autonomy of decentralized systems allows them to adapt their strategies and cause more damage. Even when traditional interventions, like content flagging, are applied, decentralized groups can adjust their tactics to avoid detection. We present key insights into how these malicious groups operate and the need for better detection systems and countermeasures. Code is available at https://github.com/renqibing/RogueAgent.
>
---
#### [new 099] Efficient Whole Slide Pathology VQA via Token Compression
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于病理图像视觉问答（VQA）任务，旨在解决全切片图像（WSI）因高分辨率导致的计算资源消耗大、现有方法缺乏生成能力的问题。论文提出TCP-LLaVA模型，通过引入可训练压缩token，聚合多模态信息并减少输入长度，从而提升VQA准确率并降低资源消耗。**

- **链接: [http://arxiv.org/pdf/2507.14497v1](http://arxiv.org/pdf/2507.14497v1)**

> **作者:** Weimin Lyu; Qingqiao Hu; Kehan Qi; Zhan Shi; Wentao Huang; Saumya Gupta; Chao Chen
>
> **摘要:** Whole-slide images (WSIs) in pathology can reach up to 10,000 x 10,000 pixels, posing significant challenges for multimodal large language model (MLLM) due to long context length and high computational demands. Previous methods typically focus on patch-level analysis or slide-level classification using CLIP-based models with multi-instance learning, but they lack the generative capabilities needed for visual question answering (VQA). More recent MLLM-based approaches address VQA by feeding thousands of patch tokens directly into the language model, which leads to excessive resource consumption. To address these limitations, we propose Token Compression Pathology LLaVA (TCP-LLaVA), the first MLLM architecture to perform WSI VQA via token compression. TCP-LLaVA introduces a set of trainable compression tokens that aggregate visual and textual information through a modality compression module, inspired by the [CLS] token mechanism in BERT. Only the compressed tokens are forwarded to the LLM for answer generation, significantly reducing input length and computational cost. Experiments on ten TCGA tumor subtypes show that TCP-LLaVA outperforms existing MLLM baselines in VQA accuracy while reducing training resource consumption by a substantial margin.
>
---
#### [new 100] ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于网络安全任务，旨在解决自动化威胁调查问题。作者构建了ExCyTIn-Bench基准，包含模拟攻击、安全日志和生成的问题，用于评估基于大语言模型（LLM）的智能体在多跳证据链推理和事件报告生成方面的能力，推动自动化威胁调查的发展。**

- **链接: [http://arxiv.org/pdf/2507.14201v1](http://arxiv.org/pdf/2507.14201v1)**

> **作者:** Yiran Wu; Mauricio Velazco; Andrew Zhao; Manuel Raúl Meléndez Luján; Srisuma Movva; Yogesh K Roy; Quang Nguyen; Roberto Rodriguez; Qingyun Wu; Michael Albada; Julia Kiseleva; Anand Mudgerikar
>
> **摘要:** We present ExCyTIn-Bench, the first benchmark to Evaluate an LLM agent x on the task of Cyber Threat Investigation through security questions derived from investigation graphs. Real-world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi-hop chains of evidence, and compile an incident report. With the developments of LLMs, building LLM-based agents for automatic thread investigation is a promising direction. To assist the development and evaluation of LLM agents, we construct a dataset from a controlled Azure tenant that covers 8 simulated real-world multi-step attacks, 57 log tables from Microsoft Sentinel and related services, and 589 automatically generated questions. We leverage security logs extracted with expert-crafted detection logic to build threat investigation graphs, and then generate questions with LLMs using paired nodes on the graph, taking the start node as background context and the end node as answer. Anchoring each question to these explicit nodes and edges not only provides automatic, explainable ground truth answers but also makes the pipeline reusable and readily extensible to new logs. This also enables the automatic generation of procedural tasks with verifiable rewards, which can be naturally extended to training agents via reinforcement learning. Our comprehensive experiments with different models confirm the difficulty of the task: with the base setting, the average reward across all evaluated models is 0.249, and the best achieved is 0.368, leaving substantial headroom for future research. Code and data are coming soon!
>
---
#### [new 101] It's Not That Simple. An Analysis of Simple Test-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文分析了一种称为“简单测试时扩展”的方法，旨在复现类似o1模型的扩展行为。任务是理解不同测试时计算扩展策略的效果。论文发现，通过限制最大长度的“缩放缩小”有效，而通过追加“Wait”的“缩放扩大”效果不佳。指出真正的测试时扩展应提升性能，而非仅模仿扩展行为。**

- **链接: [http://arxiv.org/pdf/2507.14419v1](http://arxiv.org/pdf/2507.14419v1)**

> **作者:** Guojun Wu
>
> **摘要:** Prior work proposed simple test-time scaling, a method for replicating this scaling behavior with models distilled from o1-like models by manually controlling test-time compute: either scaling down by enforcing a maximum length or scaling up by iteratively appending "Wait" when the model is about to terminate its generation. This paper presents an analysis of simple test-time scaling and finds that the scaling behavior is largely attributed to scaling down by enforcing a maximum length. In contrast, fine-tuning on long CoT data distilled from o1-like models has no significant impact on scaling behavior, and scaling up by appending "Wait" leads to inconsistencies, as the model may oscillate between solutions. A key distinction exists between scaling down by enforcing a maximum length and scaling up test-time compute in o1-like models, such as DeepSeek-R1\@. These models are typically allowed to utilize as much compute as needed, with the only constraint being the model's maximum supported length. By learning to naturally scale up test-time compute during reinforcement learning, o1-like models surpass their peak performance when scaling up. In contrast, simple test-time scaling progressively imposes a lower upper limit on model performance as it scales down. While replicating the test-time scaling behavior of o1 models can be straightforward by scaling down, it is crucial to recognize that the goal of scaling test-time compute is to unlock higher performance -- beyond what the model could originally achieve -- rather than merely reproducing the appearance of scaling behavior.
>
---
#### [new 102] Dissociating model architectures from inference computations
- **分类: q-bio.NC; cs.CL; cs.LG**

- **简介: 该论文属于序列建模任务，旨在解决模型架构与推理计算的耦合问题。作者指出，自回归模型在推理阶段可通过结构化上下文访问模拟深度时间计算，并在迭代推理中引入层次化时间分解，减少计算量同时保持预测能力，表明预测构建过程不依赖特定架构。**

- **链接: [http://arxiv.org/pdf/2507.15776v1](http://arxiv.org/pdf/2507.15776v1)**

> **作者:** Noor Sajid; Johan Medrano
>
> **备注:** 3 pages, 1 figure
>
> **摘要:** Parr et al., 2025 examines how auto-regressive and deep temporal models differ in their treatment of non-Markovian sequence modelling. Building on this, we highlight the need for dissociating model architectures, i.e., how the predictive distribution factorises, from the computations invoked at inference. We demonstrate that deep temporal computations are mimicked by autoregressive models by structuring context access during iterative inference. Using a transformer trained on next-token prediction, we show that inducing hierarchical temporal factorisation during iterative inference maintains predictive capacity while instantiating fewer computations. This emphasises that processes for constructing and refining predictions are not necessarily bound to their underlying model architectures.
>
---
#### [new 103] WebGuard: Building a Generalizable Guardrail for Web Agents
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于安全评估任务，旨在解决自主网页代理（Web Agents）可能执行有害操作的风险问题。作者构建了名为WebGuard的数据集，包含4,939个标注动作，涵盖22个领域，提出三级风险分类体系。通过评估发现当前大模型表现不足，进而尝试微调专用模型提升预测性能，取得显著改进，但仍未能达到高风险场景的可靠性要求。**

- **链接: [http://arxiv.org/pdf/2507.14293v1](http://arxiv.org/pdf/2507.14293v1)**

> **作者:** Boyuan Zheng; Zeyi Liao; Scott Salisbury; Zeyuan Liu; Michael Lin; Qinyuan Zheng; Zifan Wang; Xiang Deng; Dawn Song; Huan Sun; Yu Su
>
> **备注:** We publicly release WebGuard, along with its annotation tools and fine-tuned models, to facilitate open-source research on monitoring and safeguarding web agents. All resources are available at https://github.com/OSU-NLP-Group/WebGuard
>
> **摘要:** The rapid development of autonomous web agents powered by Large Language Models (LLMs), while greatly elevating efficiency, exposes the frontier risk of taking unintended or harmful actions. This situation underscores an urgent need for effective safety measures, akin to access controls for human users. To address this critical challenge, we introduce WebGuard, the first comprehensive dataset designed to support the assessment of web agent action risks and facilitate the development of guardrails for real-world online environments. In doing so, WebGuard specifically focuses on predicting the outcome of state-changing actions and contains 4,939 human-annotated actions from 193 websites across 22 diverse domains, including often-overlooked long-tail websites. These actions are categorized using a novel three-tier risk schema: SAFE, LOW, and HIGH. The dataset includes designated training and test splits to support evaluation under diverse generalization settings. Our initial evaluations reveal a concerning deficiency: even frontier LLMs achieve less than 60% accuracy in predicting action outcomes and less than 60% recall in lagging HIGH-risk actions, highlighting the risks of deploying current-generation agents without dedicated safeguards. We therefore investigate fine-tuning specialized guardrail models using WebGuard. We conduct comprehensive evaluations across multiple generalization settings and find that a fine-tuned Qwen2.5VL-7B model yields a substantial improvement in performance, boosting accuracy from 37% to 80% and HIGH-risk action recall from 20% to 76%. Despite these improvements, the performance still falls short of the reliability required for high-stakes deployment, where guardrails must approach near-perfect accuracy and recall.
>
---
#### [new 104] A2TTS: TTS for Low Resource Indian Languages
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决低资源印度语言中对未见说话人生成自然语音的问题。作者提出了一个基于扩散模型的TTS系统，结合说话人编码器和注意力机制，提升多说话人语音生成的自然度与准确性，并支持多种印度语言。**

- **链接: [http://arxiv.org/pdf/2507.15272v1](http://arxiv.org/pdf/2507.15272v1)**

> **作者:** Ayush Singh Bhadoriya; Abhishek Nikunj Shinde; Isha Pandey; Ganesh Ramakrishnan
>
> **摘要:** We present a speaker conditioned text-to-speech (TTS) system aimed at addressing challenges in generating speech for unseen speakers and supporting diverse Indian languages. Our method leverages a diffusion-based TTS architecture, where a speaker encoder extracts embeddings from short reference audio samples to condition the DDPM decoder for multispeaker generation. To further enhance prosody and naturalness, we employ a cross-attention based duration prediction mechanism that utilizes reference audio, enabling more accurate and speaker consistent timing. This results in speech that closely resembles the target speaker while improving duration modeling and overall expressiveness. Additionally, to improve zero-shot generation, we employed classifier free guidance, allowing the system to generate speech more near speech for unknown speakers. Using this approach, we trained language-specific speaker-conditioned models. Using the IndicSUPERB dataset for multiple Indian languages such as Bengali, Gujarati, Hindi, Marathi, Malayalam, Punjabi and Tamil.
>
---
#### [new 105] Inverse Scaling in Test-Time Compute
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大推理模型（LRMs）在测试时计算扩展中的“逆向缩放”问题，即增加推理长度反而降低性能。论文通过四类任务识别出五种失败模式，揭示了扩展推理带来的问题，强调需多长度评估以改进模型推理能力。**

- **链接: [http://arxiv.org/pdf/2507.14417v1](http://arxiv.org/pdf/2507.14417v1)**

> **作者:** Aryo Pradipta Gema; Alexander Hägele; Runjin Chen; Andy Arditi; Jacob Goldman-Wetzler; Kit Fraser-Taliente; Henry Sleight; Linda Petrini; Julian Michael; Beatrice Alex; Pasquale Minervini; Yanda Chen; Joe Benton; Ethan Perez
>
> **摘要:** We construct evaluation tasks where extending the reasoning length of Large Reasoning Models (LRMs) deteriorates performance, exhibiting an inverse scaling relationship between test-time compute and accuracy. Our evaluation tasks span four categories: simple counting tasks with distractors, regression tasks with spurious features, deduction tasks with constraint tracking, and advanced AI risks. We identify five distinct failure modes when models reason for longer: 1) Claude models become increasingly distracted by irrelevant information; 2) OpenAI o-series models resist distractors but overfit to problem framings; 3) models shift from reasonable priors to spurious correlations; 4) all models show difficulties in maintaining focus on complex deductive tasks; and 5) extended reasoning may amplify concerning behaviors, with Claude Sonnet 4 showing increased expressions of self-preservation. These findings suggest that while test-time compute scaling remains promising for improving model capabilities, it may inadvertently reinforce problematic reasoning patterns. Our results demonstrate the importance of evaluating models across diverse reasoning lengths to identify and address these failure modes in LRMs.
>
---
#### [new 106] Optimizing Legal Document Retrieval in Vietnamese with Semi-Hard Negative Mining
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于法律文档检索任务，旨在提升越南语法律文档检索的效率与准确度。论文提出了一种结合检索与重排序的两阶段框架，通过优化负样本挖掘策略，改善模型性能。团队在SoICT Hackathon 2024中取得前三名，验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.14619v1](http://arxiv.org/pdf/2507.14619v1)**

> **作者:** Van-Hoang Le; Duc-Vu Nguyen; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **备注:** Accepted at ICCCI 2025
>
> **摘要:** Large Language Models (LLMs) face significant challenges in specialized domains like law, where precision and domain-specific knowledge are critical. This paper presents a streamlined two-stage framework consisting of Retrieval and Re-ranking to enhance legal document retrieval efficiency and accuracy. Our approach employs a fine-tuned Bi-Encoder for rapid candidate retrieval, followed by a Cross-Encoder for precise re-ranking, both optimized through strategic negative example mining. Key innovations include the introduction of the Exist@m metric to evaluate retrieval effectiveness and the use of semi-hard negatives to mitigate training bias, which significantly improved re-ranking performance. Evaluated on the SoICT Hackathon 2024 for Legal Document Retrieval, our team, 4Huiter, achieved a top-three position. While top-performing teams employed ensemble models and iterative self-training on large bge-m3 architectures, our lightweight, single-pass approach offered a competitive alternative with far fewer parameters. The framework demonstrates that optimized data processing, tailored loss functions, and balanced negative sampling are pivotal for building robust retrieval-augmented systems in legal contexts.
>
---
#### [new 107] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在持续预训练中的灾难性遗忘问题。通过提出“数据混合代理”框架，利用强化学习自动学习领域数据的加权策略，实现源域与目标域间的平衡性能，无需人工设定规则，提升模型在不同领域下的表现。**

- **链接: [http://arxiv.org/pdf/2507.15640v1](http://arxiv.org/pdf/2507.15640v1)**

> **作者:** Kailai Yang; Xiao Liu; Lei Ji; Hao Li; Yeyun Gong; Peng Cheng; Mao Yang
>
> **摘要:** Continual pre-training on small-scale task-specific data is an effective method for improving large language models in new target fields, yet it risks catastrophic forgetting of their original capabilities. A common solution is to re-weight training data mixtures from source and target fields on a domain space to achieve balanced performance. Previous domain reweighting strategies rely on manual designation with certain heuristics based on human intuition or empirical results. In this work, we prove that more general heuristics can be parameterized by proposing Data Mixing Agent, the first model-based, end-to-end framework that learns to re-weight domains. The agent learns generalizable heuristics through reinforcement learning on large quantities of data mixing trajectories with corresponding feedback from an evaluation environment. Experiments in continual pre-training on math reasoning show that Data Mixing Agent outperforms strong baselines in achieving balanced performance across source and target field benchmarks. Furthermore, it generalizes well across unseen source fields, target models, and domain spaces without retraining. Direct application to the code generation field also indicates its adaptability across target domains. Further analysis showcases the agents' well-aligned heuristics with human intuitions and their efficiency in achieving superior model performance with less source-field data.
>
---
#### [new 108] Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLHF中奖励模型因分布偏移导致的过估计问题。作者提出OCRM方法，通过重要性加权进行离策略修正，提升奖励模型准确性，从而改善最终策略性能。**

- **链接: [http://arxiv.org/pdf/2507.15507v1](http://arxiv.org/pdf/2507.15507v1)**

> **作者:** Johannes Ackermann; Takashi Ishida; Masashi Sugiyama
>
> **备注:** Accept at the Conference On Language Modeling (COLM) 2025
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) allows us to train models, such as language models (LMs), to follow complex human preferences. In RLHF for LMs, we first train an LM using supervised fine-tuning, sample pairs of responses, obtain human feedback, and use the resulting data to train a reward model (RM). RL methods are then used to train the LM to maximize the reward given by the RM. As training progresses, the responses generated by the LM no longer resemble the responses seen by the RM during training, leading to the RM becoming inaccurate. The score given by the RM keeps increasing, but the learned behavior no longer matches the human preferences. This issue is known as overoptimization. We investigate overoptimization from the point of view of distribution shift and show that the shift results in an inconsistent estimate of the RM parameters, leading to an inconsistent estimate of the policy gradient. We propose Off-Policy Corrected Reward Modeling (OCRM), which iteratively off-policy corrects the RM using importance weighting, without requiring new labels or samples. This results in a more accurate RM, which empirically leads to an improved final policy. We validate our approach in experiments with summarization and chatbot datasets and show that it performs significantly better than standard RLHF methods and baselines. Our implementation is available at https://github.com/JohannesAck/OffPolicyCorrectedRewardModeling
>
---
#### [new 109] The Invisible Leash: Why RLVR May Not Escape Its Origin
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文探讨了强化学习与可验证奖励（RLVR）在扩展AI推理能力上的潜力与局限。任务是分析RLVR是否能真正突破基础模型的推理边界。研究发现，RLVR受限于基础模型的支持，可能仅放大高奖励输出，而非探索新解。实验验证了其提升精度的同时，会缩小解的多样性。指出需新算法突破此“隐形束缚”。**

- **链接: [http://arxiv.org/pdf/2507.14843v1](http://arxiv.org/pdf/2507.14843v1)**

> **作者:** Fang Wu; Weihao Xuan; Ximing Lu; Zaid Harchaoui; Yejin Choi
>
> **摘要:** Recent advances in large reasoning models highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing AI's capabilities, particularly in solving complex logical tasks. However, it remains unclear whether RLVR truly expands a model's reasoning boundary or merely amplifies high-reward outputs that the base model already knows for improved precision. This study presents a theoretical and empirical investigation that provides fresh insights into the potential limits of RLVR. First, we offer a new theoretical perspective that RLVR is constrained by the base model's support-unable to sample solutions with zero initial probability-and operates as a conservative reweighting mechanism that may restrict the discovery of entirely original solutions. We also identify an entropy-reward tradeoff: while RLVR reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while RLVR consistently improves pass@1, the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets, failing to recover correct answers that were previously accessible to the base model. Interestingly, we also observe that while RLVR sometimes increases token-level entropy, resulting in greater uncertainty at each generation step, answer-level entropy declines, indicating that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, these findings reveal potential limits of RLVR in extending reasoning horizons. Breaking this invisible leash may require future algorithmic innovations such as explicit exploration mechanisms or hybrid strategies that seed probability mass into underrepresented solution regions.
>
---
#### [new 110] Identifying Algorithmic and Domain-Specific Bias in Parliamentary Debate Summarisation
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决议会辩论自动摘要中的算法与领域偏差问题。作者提出多阶段框架，分析发言人属性对摘要可见性与准确性的影响，发现模型存在位置与党派偏差，并探索缓解策略。**

- **链接: [http://arxiv.org/pdf/2507.14221v1](http://arxiv.org/pdf/2507.14221v1)**

> **作者:** Eoghan Cunningham; James Cross; Derek Greene
>
> **摘要:** The automated summarisation of parliamentary debates using large language models (LLMs) offers a promising way to make complex legislative discourse more accessible to the public. However, such summaries must not only be accurate and concise but also equitably represent the views and contributions of all speakers. This paper explores the use of LLMs to summarise plenary debates from the European Parliament and investigates the algorithmic and representational biases that emerge in this context. We propose a structured, multi-stage summarisation framework that improves textual coherence and content fidelity, while enabling the systematic analysis of how speaker attributes -- such as speaking order or political affiliation -- influence the visibility and accuracy of their contributions in the final summaries. Through our experiments using both proprietary and open-weight LLMs, we find evidence of consistent positional and partisan biases, with certain speakers systematically under-represented or misattributed. Our analysis shows that these biases vary by model and summarisation strategy, with hierarchical approaches offering the greatest potential to reduce disparity. These findings underscore the need for domain-sensitive evaluation metrics and ethical oversight in the deployment of LLMs for democratic applications.
>
---
#### [new 111] Routine: A Structural Planning Framework for LLM Agent System in Enterprise
- **分类: cs.AI; cs.CL**

- **简介: 论文提出Routine框架，解决企业场景中大模型代理系统因缺乏领域流程知识导致的计划混乱、工具缺失和执行不稳定问题。通过结构化规划与多步骤工具调用，提升执行准确率，并实现模型快速适配新场景。**

- **链接: [http://arxiv.org/pdf/2507.14447v1](http://arxiv.org/pdf/2507.14447v1)**

> **作者:** Guancheng Zeng; Xueyi Chen; Jiawang Hu; Shaohua Qi; Yaxuan Mao; Zhantao Wang; Yifan Nie; Shuang Li; Qiuyang Feng; Pengxu Qiu; Yujia Wang; Wenqiang Han; Linyan Huang; Gang Li; Jingjing Mo; Haowen Hu
>
> **备注:** 26 pages, 8 figures, 5 tables
>
> **摘要:** The deployment of agent systems in an enterprise environment is often hindered by several challenges: common models lack domain-specific process knowledge, leading to disorganized plans, missing key tools, and poor execution stability. To address this, this paper introduces Routine, a multi-step agent planning framework designed with a clear structure, explicit instructions, and seamless parameter passing to guide the agent's execution module in performing multi-step tool-calling tasks with high stability. In evaluations conducted within a real-world enterprise scenario, Routine significantly increases the execution accuracy in model tool calls, increasing the performance of GPT-4o from 41.1% to 96.3%, and Qwen3-14B from 32.6% to 83.3%. We further constructed a Routine-following training dataset and fine-tuned Qwen3-14B, resulting in an accuracy increase to 88.2% on scenario-specific evaluations, indicating improved adherence to execution plans. In addition, we employed Routine-based distillation to create a scenario-specific, multi-step tool-calling dataset. Fine-tuning on this distilled dataset raised the model's accuracy to 95.5%, approaching GPT-4o's performance. These results highlight Routine's effectiveness in distilling domain-specific tool-usage patterns and enhancing model adaptability to new scenarios. Our experimental results demonstrate that Routine provides a practical and accessible approach to building stable agent workflows, accelerating the deployment and adoption of agent systems in enterprise environments, and advancing the technical vision of AI for Process.
>
---
#### [new 112] GREAT: Guiding Query Generation with a Trie for Recommending Related Search about Video at Kuaishou
- **分类: cs.IR; cs.CL**

- **简介: 论文提出GREAT框架，用于视频相关搜索中的物品到查询（I2Q）推荐任务，旨在解决当前方法缺乏语义与查询深度交互的问题。工作包括构建基于Trie的查询生成方法，并发布大规模数据集KuaiRS。**

- **链接: [http://arxiv.org/pdf/2507.15267v1](http://arxiv.org/pdf/2507.15267v1)**

> **作者:** Ninglu Shao; Jinshan Wang; Chenxu Wang; Qingbiao Li; Xiaoxue Zang; Han Li
>
> **摘要:** Currently, short video platforms have become the primary place for individuals to share experiences and obtain information. To better meet users' needs for acquiring information while browsing short videos, some apps have introduced a search entry at the bottom of videos, accompanied with recommended relevant queries. This scenario is known as query recommendation in video-related search, where core task is item-to-query (I2Q) recommendation. As this scenario has only emerged in recent years, there is a notable scarcity of academic research and publicly available datasets in this domain. To address this gap, we systematically examine the challenges associated with this scenario for the first time. Subsequently, we release a large-scale dataset derived from real-world data pertaining to the query recommendation in video-\textit{\textbf{r}}elated \textit{\textbf{s}}earch on the \textit{\textbf{Kuai}}shou app (\textbf{KuaiRS}). Presently, existing methods rely on embeddings to calculate similarity for matching short videos with queries, lacking deep interaction between the semantic content and the query. In this paper, we introduce a novel LLM-based framework named \textbf{GREAT}, which \textit{\textbf{g}}uides que\textit{\textbf{r}}y g\textit{\textbf{e}}ner\textit{\textbf{a}}tion with a \textit{\textbf{t}}rie to address I2Q recommendation in related search. Specifically, we initially gather high-quality queries with high exposure and click-through rate to construct a query-based trie. During training, we enhance the LLM's capability to generate high-quality queries using the query-based trie. In the inference phase, the query-based trie serves as a guide for the token generation. Finally, we further refine the relevance and literal quality between items and queries via a post-processing module. Extensive offline and online experiments demonstrate the effectiveness of our proposed method.
>
---
#### [new 113] Exploiting Context-dependent Duration Features for Voice Anonymization Attack Systems
- **分类: cs.SD; cs.CL; cs.CR; eess.AS**

- **简介: 该论文属于语音匿名化与攻击分析任务，旨在研究语音中的时序动态特征对说话人识别的影响。论文提出了一种基于上下文相关时长嵌入的说话人特征表示方法，并构建了攻击模型，用于评估语音匿名化系统的安全性。实验表明，该方法在原始和匿名化语音上均提升了攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.15214v1](http://arxiv.org/pdf/2507.15214v1)**

> **作者:** Natalia Tomashenko; Emmanuel Vincent; Marc Tommasi
>
> **备注:** Accepted at Interspeech-2025
>
> **摘要:** The temporal dynamics of speech, encompassing variations in rhythm, intonation, and speaking rate, contain important and unique information about speaker identity. This paper proposes a new method for representing speaker characteristics by extracting context-dependent duration embeddings from speech temporal dynamics. We develop novel attack models using these representations and analyze the potential vulnerabilities in speaker verification and voice anonymization systems.The experimental results show that the developed attack models provide a significant improvement in speaker verification performance for both original and anonymized data in comparison with simpler representations of speech temporal dynamics reported in the literature.
>
---
#### [new 114] Long-Short Distance Graph Neural Networks and Improved Curriculum Learning for Emotion Recognition in Conversation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于情感识别任务，旨在解决对话中情感识别的长距离与短距离信息融合及数据不平衡问题。论文提出了长-短距离图神经网络模型，利用有向无环图构建长距离和短距离图网络，并引入差异正则化和双仿射模块以增强特征交互，同时设计了改进的课程学习方法，通过“加权情感转移”指标解决数据不平衡问题，从而提升情感识别效果。**

- **链接: [http://arxiv.org/pdf/2507.15205v1](http://arxiv.org/pdf/2507.15205v1)**

> **作者:** Xinran Li; Xiujuan Xu; Jiaqi Qiao
>
> **备注:** Accepted by the 28th European Conference on Artificial Intelligence (ECAI 2025)
>
> **摘要:** Emotion Recognition in Conversation (ERC) is a practical and challenging task. This paper proposes a novel multimodal approach, the Long-Short Distance Graph Neural Network (LSDGNN). Based on the Directed Acyclic Graph (DAG), it constructs a long-distance graph neural network and a short-distance graph neural network to obtain multimodal features of distant and nearby utterances, respectively. To ensure that long- and short-distance features are as distinct as possible in representation while enabling mutual influence between the two modules, we employ a Differential Regularizer and incorporate a BiAffine Module to facilitate feature interaction. In addition, we propose an Improved Curriculum Learning (ICL) to address the challenge of data imbalance. By computing the similarity between different emotions to emphasize the shifts in similar emotions, we design a "weighted emotional shift" metric and develop a difficulty measurer, enabling a training process that prioritizes learning easy samples before harder ones. Experimental results on the IEMOCAP and MELD datasets demonstrate that our model outperforms existing benchmarks.
>
---
#### [new 115] GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于文本分类任务，旨在解决垃圾文本检测中对抗攻击和标注数据不足的问题。作者提出了GCC-Spam框架，结合生成对抗网络、对比学习和字符相似性网络，提升检测效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.14679v1](http://arxiv.org/pdf/2507.14679v1)**

> **作者:** Zixin Xu; Zhijie Wang; Zhiyuan Pan
>
> **摘要:** The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples.
>
---
#### [new 116] Towards physician-centered oversight of conversational diagnostic AI
- **分类: cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于医疗AI任务，旨在解决诊断AI在现实应用中的安全与监管问题。研究提出了一种异步监督框架g-AMIE，通过AI进行问诊并在医生监督下提供诊断建议，提升诊疗效率与质量。实验表明，该系统在多项指标上优于NP/PA及受监管的PCP组。**

- **链接: [http://arxiv.org/pdf/2507.15743v1](http://arxiv.org/pdf/2507.15743v1)**

> **作者:** Elahe Vedadi; David Barrett; Natalie Harris; Ellery Wulczyn; Shashir Reddy; Roma Ruparel; Mike Schaekermann; Tim Strother; Ryutaro Tanno; Yash Sharma; Jihyeon Lee; Cían Hughes; Dylan Slack; Anil Palepu; Jan Freyberg; Khaled Saab; Valentin Liévin; Wei-Hung Weng; Tao Tu; Yun Liu; Nenad Tomasev; Kavita Kulkarni; S. Sara Mahdavi; Kelvin Guu; Joëlle Barral; Dale R. Webster; James Manyika; Avinatan Hassidim; Katherine Chou; Yossi Matias; Pushmeet Kohli; Adam Rodman; Vivek Natarajan; Alan Karthikesalingam; David Stutz
>
> **摘要:** Recent work has demonstrated the promise of conversational AI systems for diagnostic dialogue. However, real-world assurance of patient safety means that providing individual diagnoses and treatment plans is considered a regulated activity by licensed professionals. Furthermore, physicians commonly oversee other team members in such activities, including nurse practitioners (NPs) or physician assistants/associates (PAs). Inspired by this, we propose a framework for effective, asynchronous oversight of the Articulate Medical Intelligence Explorer (AMIE) AI system. We propose guardrailed-AMIE (g-AMIE), a multi-agent system that performs history taking within guardrails, abstaining from individualized medical advice. Afterwards, g-AMIE conveys assessments to an overseeing primary care physician (PCP) in a clinician cockpit interface. The PCP provides oversight and retains accountability of the clinical decision. This effectively decouples oversight from intake and can thus happen asynchronously. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) of text consultations with asynchronous oversight, we compared g-AMIE to NPs/PAs or a group of PCPs under the same guardrails. Across 60 scenarios, g-AMIE outperformed both groups in performing high-quality intake, summarizing cases, and proposing diagnoses and management plans for the overseeing PCP to review. This resulted in higher quality composite decisions. PCP oversight of g-AMIE was also more time-efficient than standalone PCP consultations in prior work. While our study does not replicate existing clinical practices and likely underestimates clinicians' capabilities, our results demonstrate the promise of asynchronous oversight as a feasible paradigm for diagnostic AI systems to operate under expert human oversight for enhancing real-world care.
>
---
#### [new 117] What do Large Language Models know about materials?
- **分类: physics.app-ph; cs.CE; cs.CL**

- **简介: 该论文探讨大型语言模型（LLMs）在材料科学中的知识表现能力，聚焦其在材料“工艺-结构-性能-效能”链中的适用性。任务是评估LLMs生成材料信息的准确性，分析词汇与分词对材料特征识别的影响，并建立材料知识基准，以指导模型在工程中的合理应用。**

- **链接: [http://arxiv.org/pdf/2507.14586v1](http://arxiv.org/pdf/2507.14586v1)**

> **作者:** Adrian Ehrenhofer; Thomas Wallmersperger; Gianaurelio Cuniberti
>
> **摘要:** Large Language Models (LLMs) are increasingly applied in the fields of mechanical engineering and materials science. As models that establish connections through the interface of language, LLMs can be applied for step-wise reasoning through the Processing-Structure-Property-Performance chain of material science and engineering. Current LLMs are built for adequately representing a dataset, which is the most part of the accessible internet. However, the internet mostly contains non-scientific content. If LLMs should be applied for engineering purposes, it is valuable to investigate models for their intrinsic knowledge -- here: the capacity to generate correct information about materials. In the current work, for the example of the Periodic Table of Elements, we highlight the role of vocabulary and tokenization for the uniqueness of material fingerprints, and the LLMs' capabilities of generating factually correct output of different state-of-the-art open models. This leads to a material knowledge benchmark for an informed choice, for which steps in the PSPP chain LLMs are applicable, and where specialized models are required.
>
---
## 更新

#### [replaced 001] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.09516v4](http://arxiv.org/pdf/2503.09516v4)**

> **作者:** Bowen Jin; Hansi Zeng; Zhenrui Yue; Jinsung Yoon; Sercan Arik; Dong Wang; Hamed Zamani; Jiawei Han
>
> **备注:** 31 pages
>
> **摘要:** Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
>
---
#### [replaced 002] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.17163v4](http://arxiv.org/pdf/2502.17163v4)**

> **作者:** María Andrea Cruz Blandón; Jayasimha Talur; Bruno Charron; Dong Liu; Saab Mansour; Marcello Federico
>
> **备注:** ACL 2025
>
> **摘要:** Automatic evaluation of retrieval augmented generation (RAG) systems relies on fine-grained dimensions like faithfulness and relevance, as judged by expert human annotators. Meta-evaluation benchmarks support the development of automatic evaluators that correlate well with human judgement. However, existing benchmarks predominantly focus on English or use translated data, which fails to capture cultural nuances. A native approach provides a better representation of the end user experience. In this work, we develop a Multilingual End-to-end Meta-Evaluation RAG benchmark (MEMERAG). Our benchmark builds on the popular MIRACL dataset, using native-language questions and generating responses with diverse large language models (LLMs), which are then assessed by expert annotators for faithfulness and relevance. We describe our annotation process and show that it achieves high inter-annotator agreement. We then analyse the performance of the answer-generating LLMs across languages as per the human evaluators. Finally we apply the dataset to our main use-case which is to benchmark multilingual automatic evaluators (LLM-as-a-judge). We show that our benchmark can reliably identify improvements offered by advanced prompting techniques and LLMs. Our dataset is available at https://github.com/amazon-science/MEMERAG
>
---
#### [replaced 003] Towards Harmonized Uncertainty Estimation for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19073v2](http://arxiv.org/pdf/2505.19073v2)**

> **作者:** Rui Li; Jing Long; Muge Qi; Heming Xia; Lei Sha; Peiyi Wang; Zhifang Sui
>
> **备注:** ACL 2025
>
> **摘要:** To facilitate robust and trustworthy deployment of large language models (LLMs), it is essential to quantify the reliability of their generations through uncertainty estimation. While recent efforts have made significant advancements by leveraging the internal logic and linguistic features of LLMs to estimate uncertainty scores, our empirical analysis highlights the pitfalls of these methods to strike a harmonized estimation between indication, balance, and calibration, which hinders their broader capability for accurate uncertainty estimation. To address this challenge, we propose CUE (Corrector for Uncertainty Estimation): A straightforward yet effective method that employs a lightweight model trained on data aligned with the target LLM's performance to adjust uncertainty scores. Comprehensive experiments across diverse models and tasks demonstrate its effectiveness, which achieves consistent improvements of up to 60% over existing methods.
>
---
#### [replaced 004] A Survey on Large Language Model-Based Social Agents in Game-Theoretic Scenarios
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.03920v2](http://arxiv.org/pdf/2412.03920v2)**

> **作者:** Xiachong Feng; Longxu Dou; Ella Li; Qinghao Wang; Haochuan Wang; Yu Guo; Chang Ma; Lingpeng Kong
>
> **摘要:** Game-theoretic scenarios have become pivotal in evaluating the social intelligence of Large Language Model (LLM)-based social agents. While numerous studies have explored these agents in such settings, there is a lack of a comprehensive survey summarizing the current progress. To address this gap, we systematically review existing research on LLM-based social agents within game-theoretic scenarios. Our survey organizes the findings into three core components: Game Framework, Social Agent, and Evaluation Protocol. The game framework encompasses diverse game scenarios, ranging from choice-focusing to communication-focusing games. The social agent part explores agents' preferences, beliefs, and reasoning abilities, as well as their interactions and synergistic effects on decision-making. The evaluation protocol covers both game-agnostic and game-specific metrics for assessing agent performance. Additionally, we analyze the performance of current social agents across various game scenarios. By reflecting on the current research and identifying future research directions, this survey provides insights to advance the development and evaluation of social agents in game-theoretic scenarios.
>
---
#### [replaced 005] Draft-based Approximate Inference for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08373v2](http://arxiv.org/pdf/2506.08373v2)**

> **作者:** Kevin Galim; Ethan Ewer; Wonjun Kang; Minjae Lee; Hyung Il Koo; Kangwook Lee
>
> **备注:** Added discussion and comparison with SpecPrefill
>
> **摘要:** Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: (i) SpecKV, the first method that leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping, and (ii) SpecPC, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput. Our code is available at https://github.com/furiosa-ai/draft-based-approx-llm.
>
---
#### [replaced 006] How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14642v2](http://arxiv.org/pdf/2502.14642v2)**

> **作者:** Rui Li; Heming Xia; Xinfeng Yuan; Qingxiu Dong; Lei Sha; Wenjie Li; Zhifang Sui
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior.
>
---
#### [replaced 007] Meta4XNLI: A Crosslingual Parallel Corpus for Metaphor Detection and Interpretation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.07053v3](http://arxiv.org/pdf/2404.07053v3)**

> **作者:** Elisa Sanchez-Bayona; Rodrigo Agerri
>
> **摘要:** Metaphors are a ubiquitous but often overlooked part of everyday language. As a complex cognitive-linguistic phenomenon, they provide a valuable means to evaluate whether language models can capture deeper aspects of meaning, including semantic, pragmatic, and cultural context. In this work, we present Meta4XNLI, the first parallel dataset for Natural Language Inference (NLI) newly annotated for metaphor detection and interpretation in both English and Spanish. Meta4XNLI facilitates the comparison of encoder- and decoder-based models in detecting and understanding metaphorical language in multilingual and cross-lingual settings. Our results show that fine-tuned encoders outperform decoders-only LLMs in metaphor detection. Metaphor interpretation is evaluated via the NLI framework with comparable performance of masked and autoregressive models, which notably decreases when the inference is affected by metaphorical language. Our study also finds that translation plays an important role in the preservation or loss of metaphors across languages, introducing shifts that might impact metaphor occurrence and model performance. These findings underscore the importance of resources like Meta4XNLI for advancing the analysis of the capabilities of language models and improving our understanding of metaphor processing across languages. Furthermore, the dataset offers previously unavailable opportunities to investigate metaphor interpretation, cross-lingual metaphor transferability, and the impact of translation on the development of multilingual annotated resources.
>
---
#### [replaced 008] FastMCTS: A Simple Sampling Strategy for Data Synthesis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11476v2](http://arxiv.org/pdf/2502.11476v2)**

> **作者:** Peiji Li; Kai Lv; Yunfan Shao; Yichuan Ma; Linyang Li; Xiaoqing Zheng; Xipeng Qiu; Qipeng Guo
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Synthetic high-quality multi-step reasoning data can significantly enhance the performance of large language models on various tasks. However, most existing methods rely on rejection sampling, which generates trajectories independently and suffers from inefficiency and imbalanced sampling across problems of varying difficulty. In this work, we introduce FastMCTS, an innovative data synthesis strategy inspired by Monte Carlo Tree Search. FastMCTS provides a more efficient sampling method for multi-step reasoning data, offering step-level evaluation signals and promoting balanced sampling across problems of different difficulty levels. Experiments on both English and Chinese reasoning datasets demonstrate that FastMCTS generates over 30\% more correct reasoning paths compared to rejection sampling as the number of generated tokens scales up. Furthermore, under comparable synthetic data budgets, models trained on FastMCTS-generated data outperform those trained on rejection sampling data by 3.9\% across multiple benchmarks. As a lightweight sampling strategy, FastMCTS offers a practical and efficient alternative for synthesizing high-quality reasoning data. Our code will be released soon.
>
---
#### [replaced 009] The Dual-Route Model of Induction
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.03022v2](http://arxiv.org/pdf/2504.03022v2)**

> **作者:** Sheridan Feucht; Eric Todd; Byron Wallace; David Bau
>
> **备注:** 43 pages, 49 figures. Published as a conference paper at COLM 2025. Code and data at https://dualroute.baulab.info
>
> **摘要:** Prior work on in-context copying has shown the existence of induction heads, which attend to and promote individual tokens during copying. In this work we discover a new type of induction head: concept-level induction heads, which copy entire lexical units instead of individual tokens. Concept induction heads learn to attend to the ends of multi-token words throughout training, working in parallel with token-level induction heads to copy meaningful text. We show that these heads are responsible for semantic tasks like word-level translation, whereas token induction heads are vital for tasks that can only be done verbatim (like copying nonsense tokens). These two "routes" operate independently: we show that ablation of token induction heads causes models to paraphrase where they would otherwise copy verbatim. By patching concept induction head outputs, we find that they contain language-independent word representations that mediate natural language translation, suggesting that LLMs represent abstract word meanings independent of language or form.
>
---
#### [replaced 010] Empowering LLMs with Logical Reasoning: A Comprehensive Survey
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15652v4](http://arxiv.org/pdf/2502.15652v4)**

> **作者:** Fengxiang Cheng; Haoxuan Li; Fenrong Liu; Robert van Rooij; Kun Zhang; Zhouchen Lin
>
> **备注:** Accepted by IJCAI 2025 (Survey Track)
>
> **摘要:** Large language models (LLMs) have achieved remarkable successes on various tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs, which can be categorized into the following two aspects: (1) Logical question answering: LLMs often fail to generate the correct answer within a complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises. (2) Logical consistency: LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art question-answering LLM Macaw, answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose a detailed taxonomy. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistencies, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extending to modal logic to account for uncertainty and developing efficient algorithms that simultaneously satisfy multiple logical consistencies.
>
---
#### [replaced 011] Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains
- **分类: cs.LG; cs.CL; cs.IT; math.IT; stat.ML**

- **链接: [http://arxiv.org/pdf/2402.04161v2](http://arxiv.org/pdf/2402.04161v2)**

> **作者:** Ashok Vardhan Makkuva; Marco Bondaschi; Adway Girish; Alliot Nagle; Martin Jaggi; Hyeji Kim; Michael Gastpar
>
> **备注:** Published at ICLR 2025 under the title "Attention with Markov: A Curious Case of Single-Layer Transformers"
>
> **摘要:** Attention-based transformers have achieved tremendous success across a variety of disciplines including natural languages. To deepen our understanding of their sequential modeling capabilities, there is a growing interest in using Markov input processes to study them. A key finding is that when trained on first-order Markov chains, transformers with two or more layers consistently develop an induction head mechanism to estimate the in-context bigram conditional distribution. In contrast, single-layer transformers, unable to form an induction head, directly learn the Markov kernel but often face a surprising challenge: they become trapped in local minima representing the unigram distribution, whereas deeper models reliably converge to the ground-truth bigram. While single-layer transformers can theoretically model first-order Markov chains, their empirical failure to learn this simple kernel in practice remains a curious phenomenon. To explain this contrasting behavior of single-layer models, in this paper we introduce a new framework for a principled analysis of transformers via Markov chains. Leveraging our framework, we theoretically characterize the loss landscape of single-layer transformers and show the existence of global minima (bigram) and bad local minima (unigram) contingent on data properties and model architecture. We precisely delineate the regimes under which these local optima occur. Backed by experiments, we demonstrate that our theoretical findings are in congruence with the empirical results. Finally, we outline several open problems in this arena. Code is available at https://github.com/Bond1995/Markov .
>
---
#### [replaced 012] Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in Large Language Model-Based Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00409v3](http://arxiv.org/pdf/2502.00409v3)**

> **作者:** Clovis Varangot-Reille; Christophe Bouvard; Antoine Gourru; Mathieu Ciancone; Marion Schaeffer; François Jacquenet
>
> **摘要:** Large Language Model (LLM)-based systems, i.e. interconnected elements that include an LLM as a central component, such as conversational agents, are usually designed with monolithic, static architectures that rely on a single, general-purpose LLM to handle all user queries. However, these systems may be inefficient as different queries may require different levels of reasoning, domain knowledge or pre-processing. While generalist LLMs (e.g. GPT-4o, Claude-Sonnet) perform well across a wide range of tasks, they may incur significant financial, energy and computational costs. These costs may be disproportionate for simpler queries, resulting in unnecessary resource utilisation. A routing mechanism can therefore be employed to route queries to more appropriate components, such as smaller or specialised models, thereby improving efficiency and optimising resource consumption. This survey aims to provide a comprehensive overview of routing strategies in LLM-based systems. Specifically, it reviews when, why, and how routing should be integrated into LLM pipelines to improve efficiency, scalability, and performance. We define the objectives to optimise, such as cost minimisation and performance maximisation, and discuss the timing of routing within the LLM workflow, whether it occurs before or after generation. We also detail the various implementation strategies, including similarity-based, supervised, reinforcement learning-based, and generative methods. Practical considerations such as industrial applications and current limitations are also examined, like standardising routing experiments, accounting for non-financial costs, and designing adaptive strategies. By formalising routing as a performance-cost optimisation problem, this survey provides tools and directions to guide future research and development of adaptive low-cost LLM-based systems.
>
---
#### [replaced 013] Layerwise Recall and the Geometry of Interwoven Knowledge in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.10871v2](http://arxiv.org/pdf/2502.10871v2)**

> **作者:** Ge Lei; Samuel J. Cooper
>
> **摘要:** This study explores how large language models (LLMs) encode interwoven scientific knowledge, using chemical elements and LLaMA-series models as a case study. We identify a 3D spiral structure in the hidden states that aligns with the conceptual structure of the periodic table, suggesting that LLMs can reflect the geometric organization of scientific concepts learned from text. Linear probing reveals that middle layers encode continuous, overlapping attributes that enable indirect recall, while deeper layers sharpen categorical distinctions and incorporate linguistic context. These findings suggest that LLMs represent symbolic knowledge not as isolated facts, but as structured geometric manifolds that intertwine semantic information across layers. We hope this work inspires further exploration of how LLMs represent and reason about scientific knowledge, particularly in domains such as materials science.
>
---
#### [replaced 014] A Survey of the Evolution of Language Model-Based Dialogue Systems: Data, Task and Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2311.16789v2](http://arxiv.org/pdf/2311.16789v2)**

> **作者:** Hongru Wang; Lingzhi Wang; Yiming Du; Liang Chen; Jingyan Zhou; Yufei Wang; Kam-Fai Wong
>
> **摘要:** Dialogue systems (DS), including the task-oriented dialogue system (TOD) and the open-domain dialogue system (ODD), have always been a fundamental task in natural language processing (NLP), allowing various applications in practice. Owing to sophisticated training and well-designed model architecture, language models (LM) are usually adopted as the necessary backbone to build the dialogue system. Consequently, every breakthrough in LM brings about a shift in learning paradigm and research attention within dialogue system, especially the appearance of pre-trained language models (PLMs) and large language models (LLMs). In this paper, we take a deep look at the history of the dialogue system, especially its special relationship with the advancements of language models. Thus, our survey offers a systematic perspective, categorizing different stages in a chronological order aligned with LM breakthroughs, providing a comprehensive review of state-of-the-art research outcomes. What's more, we turn our attention to emerging topics and engage in a discussion on open challenges, providing valuable insights into the future directions for LLM-based dialogue systems. In summary, this survey delves into the dynamic interplay between language models and dialogue systems, unraveling the evolutionary path of this essential relationship. Through this exploration, we pave the way for a deeper comprehension of the field, guiding future developments in LM-based dialogue systems.
>
---
#### [replaced 015] A Mathematical Theory of Discursive Networks
- **分类: cs.CL; cs.LG; 68T01, 60J10, 91D30, 05C82, 68T50, 68W20, 94A15; I.2.7; I.2.11; G.3**

- **链接: [http://arxiv.org/pdf/2507.06565v4](http://arxiv.org/pdf/2507.06565v4)**

> **作者:** Juan B. Gutiérrez
>
> **备注:** 39 pages, 4 figures, 4 tables, 3 algorithm, 56 references
>
> **摘要:** Large language models (LLMs) turn writing into a live exchange between humans and software. We characterize this new medium as a discursive network that treats people and LLMs as equal nodes and tracks how their statements circulate. We define the generation of erroneous information as invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. We develop a general mathematical model of discursive networks that shows that a network governed only by drift and self-repair stabilizes at a modest error rate. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source Flaws-of-Others (FOO) algorithm: a configurable loop in which any set of agents critique one another while a harmonizer merges their verdicts. We identify an ethical transgression, epithesis, that occurs when humans fail to engage in the discursive network. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from connecting imperfect ones into networks that enforce mutual accountability.
>
---
#### [replaced 016] Entity-aware Cross-lingual Claim Detection for Automated Fact-checking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.15220v4](http://arxiv.org/pdf/2503.15220v4)**

> **作者:** Rrubaa Panchendrarajan; Arkaitz Zubiaga
>
> **摘要:** Identifying claims requiring verification is a critical task in automated fact-checking, especially given the proliferation of misinformation on social media platforms. Despite notable progress, challenges remain-particularly in handling multilingual data prevalent in online discourse. Recent efforts have focused on fine-tuning pre-trained multilingual language models to address this. While these models can handle multiple languages, their ability to effectively transfer cross-lingual knowledge for detecting claims spreading on social media remains under-explored. In this paper, we introduce EX-Claim, an entity-aware cross-lingual claim detection model that generalizes well to handle multilingual claims. The model leverages entity information derived from named entity recognition and entity linking techniques to improve the language-level performance of both seen and unseen languages during training. Extensive experiments conducted on three datasets from different social media platforms demonstrate that our proposed model stands out as an effective solution, demonstrating consistent performance gains across 27 languages and robust knowledge transfer between languages seen and unseen during training.
>
---
#### [replaced 017] Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2503.10968v2](http://arxiv.org/pdf/2503.10968v2)**

> **作者:** Camilo Chacón Sartori; Christian Blum
>
> **摘要:** Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills.
>
---
#### [replaced 018] KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16002v3](http://arxiv.org/pdf/2502.16002v3)**

> **作者:** Jingbo Yang; Bairu Hou; Wei Wei; Yujia Bao; Shiyu Chang
>
> **摘要:** We describe KVLink, an approach for efficient key-value (KV) cache reuse in large language models (LLMs). In many LLM applications, different inputs can share overlapping context, such as the same retrieved document appearing in multiple queries. However, the LLMs still need to encode the entire context for each query, leading to redundant computation. In this paper, we investigate a new strategy to eliminate such inefficiency, where the KV cache of each document is precomputed independently. During inference, the KV caches of retrieved documents are concatenated, allowing the model to reuse cached representations instead of recomputing them. To mitigate the performance degradation when using KV caches computed independently for each document, KVLink introduces two key techniques: adjusting positional embeddings of the KV cache at inference to match the global position after concatenation, and using trainable special tokens to restore self-attention across independently encoded documents. Experiments across 7 datasets demonstrate that KVLink improves question answering accuracy by an average of 4% over state-of-the-art methods. Furthermore, by leveraging precomputed KV caches, our approach reduces time-to-first-token by up to 96% compared to standard LLM inference, making it a scalable and efficient solution for context reuse. Additionally, KVLink can be combined with KV cache compression to further save cache loading and storage overhead while outperforming the baselines.
>
---
#### [replaced 019] A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.12899v3](http://arxiv.org/pdf/2503.12899v3)**

> **作者:** Jian Gu; Aldeida Aleti; Chunyang Chen; Hongyu Zhang
>
> **备注:** 13 pages, 7 figure, 8 tables, under peer-review
>
> **摘要:** Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose Semantic Targeting for Analytical Repair (STAR), a pioneering and novel semantic-based optimization approach for repairing LLMs. STAR realizes the main operations of repairing LMs in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (MINT) and optimization methods (SGD), STAR integrates their strengths while mitigating their limitations. STAR supports solving multiple failures together, significantly improving the usefulness. Evaluated on coding tasks using popular code LMs, STAR exhibits superior effectiveness (10.5%-19.9% improvements) and efficiency (2.4-7.0 times speedup). In terms of side effects, namely the balance between generalization and specificity, STAR outperforms prior work by a significant margin. Additionally, we conducted assessments on the overfitting risk of LM repair as well as the cumulative impact.
>
---
#### [replaced 020] VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.02655v3](http://arxiv.org/pdf/2402.02655v3)**

> **作者:** Thinh Phuoc Ngo; Khoa Tran Anh Dang; Son T. Luu; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **备注:** To appear as the main conference paper at EACL 2024
>
> **摘要:** This paper presents the development process of a Vietnamese spoken language corpus for machine reading comprehension (MRC) tasks and provides insights into the challenges and opportunities associated with using real-world data for machine reading comprehension tasks. The existing MRC corpora in Vietnamese mainly focus on formal written documents such as Wikipedia articles, online newspapers, or textbooks. In contrast, the VlogQA consists of 10,076 question-answer pairs based on 1,230 transcript documents sourced from YouTube -- an extensive source of user-uploaded content, covering the topics of food and travel. By capturing the spoken language of native Vietnamese speakers in natural settings, an obscure corner overlooked in Vietnamese research, the corpus provides a valuable resource for future research in reading comprehension tasks for the Vietnamese language. Regarding performance evaluation, our deep-learning models achieved the highest F1 score of 75.34% on the test set, indicating significant progress in machine reading comprehension for Vietnamese spoken language data. In terms of EM, the highest score we accomplished is 53.97%, which reflects the challenge in processing spoken-based content and highlights the need for further improvement.
>
---
#### [replaced 021] Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.03441v2](http://arxiv.org/pdf/2501.03441v2)**

> **作者:** Sarah E. Finch; Ellie S. Paek; Ikseon Choi; Jinho D. Choi
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** As chatbots become integral to daily life, personalizing systems is key for fostering trust, engagement, and inclusivity. This study examines how linguistic similarity affects chatbot performance, focusing on integrating African American English (AAE) into virtual agents to better serve the African American community. We develop text-based and spoken chatbots using large language models and text-to-speech technology, then evaluate them with AAE speakers against standard English chatbots. Our results show that while text-based AAE chatbots often underperform, spoken chatbots benefit from an African American voice and AAE elements, improving performance and preference. These findings underscore the complexities of linguistic personalization and the dynamics between text and speech modalities, highlighting technological limitations that affect chatbots' AA speech generation and pointing to promising future research directions.
>
---
#### [replaced 022] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.08985v4](http://arxiv.org/pdf/2412.08985v4)**

> **作者:** Tianshi Zheng; Weihan Li; Jiaxin Bai; Weiqi Wang; Yangqiu Song
>
> **备注:** ACL 2025 Main
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems show remarkable potential as question answering tools in the K-12 Education domain, where knowledge is typically queried within the restricted scope of authoritative textbooks. However, discrepancies between these textbooks and the parametric knowledge inherent in Large Language Models (LLMs) can undermine the effectiveness of RAG systems. To systematically investigate RAG system robustness against such knowledge discrepancies, we introduce KnowShiftQA. This novel question answering dataset simulates these discrepancies by applying deliberate hypothetical knowledge updates to both answers and source documents, reflecting how textbook knowledge can shift. KnowShiftQA comprises 3,005 questions across five subjects, designed with a comprehensive question typology focusing on context utilization and knowledge integration. Our extensive experiments on retrieval and question answering performance reveal that most RAG systems suffer a substantial performance drop when faced with these knowledge discrepancies. Furthermore, questions requiring the integration of contextual (textbook) knowledge with parametric (LLM) knowledge pose a significant challenge to current LLMs.
>
---
#### [replaced 023] On Entity Identification in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02701v4](http://arxiv.org/pdf/2506.02701v4)**

> **作者:** Masaki Sakata; Benjamin Heinzerling; Sho Yokoi; Takumi Ito; Kentaro Inui
>
> **备注:** ACL 2025 Findings; 26 pages, 13 figures, 9 tables
>
> **摘要:** We analyze the extent to which internal representations of language models (LMs) identify and distinguish mentions of named entities, focusing on the many-to-many correspondence between entities and their mentions. We first formulate two problems of entity mentions -- ambiguity and variability -- and propose a framework analogous to clustering quality metrics. Specifically, we quantify through cluster analysis of LM internal representations the extent to which mentions of the same entity cluster together and mentions of different entities remain separated. Our experiments examine five Transformer-based autoregressive models, showing that they effectively identify and distinguish entities with metrics analogous to precision and recall ranging from 0.66 to 0.9. Further analysis reveals that entity-related information is compactly represented in a low-dimensional linear subspace at early LM layers. Additionally, we clarify how the characteristics of entity representations influence word prediction performance. These findings are interpreted through the lens of isomorphism between LM representations and entity-centric knowledge structures in the real world, providing insights into how LMs internally organize and use entity information.
>
---
#### [replaced 024] Enabling Efficient Attack Investigation via Human-in-the-Loop Security Analysis
- **分类: cs.CR; cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2211.05403v3](http://arxiv.org/pdf/2211.05403v3)**

> **作者:** Saimon Amanuel Tsegai; Xinyu Yang; Haoyuan Liu; Peng Gao
>
> **备注:** Accepted at 51st International Conference on Very Large Data Bases (VLDB) 2025
>
> **摘要:** System auditing is a vital technique for collecting system call events as system provenance and investigating complex multi-step attacks such as Advanced Persistent Threats. However, existing attack investigation methods struggle to uncover long attack sequences due to the massive volume of system provenance data and their inability to focus on attack-relevant parts. In this paper, we present Provexa, a defense system that enables human analysts to effectively analyze large-scale system provenance to reveal multi-step attack sequences. Provexa introduces an expressive domain-specific language, ProvQL, that offers essential primitives for various types of attack analyses (e.g., attack pattern search, attack dependency tracking) with user-defined constraints, enabling analysts to focus on attack-relevant parts and iteratively sift through the large provenance data. Moreover, Provexa provides an optimized execution engine for efficient language execution. Our extensive evaluations on a wide range of attack scenarios demonstrate the practical effectiveness of Provexa in facilitating timely attack investigation.
>
---
#### [replaced 025] Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.10524v2](http://arxiv.org/pdf/2507.10524v2)**

> **作者:** Sangmin Bae; Yujin Kim; Reza Bayat; Sungnyun Kim; Jiyoun Ha; Tal Schuster; Adam Fisch; Hrayr Harutyunyan; Ziwei Ji; Aaron Courville; Se-Young Yun
>
> **备注:** 36 pages, 9 figures, 14 tables, codes at https://github.com/raymin0223/mixture_of_recursions
>
> **摘要:** Scaling language models unlocks impressive capabilities, but the accompanying computational and memory demands make both training and deployment expensive. Existing efficiency efforts typically target either parameter sharing or adaptive computation, leaving open the question of how to attain both simultaneously. We introduce Mixture-of-Recursions (MoR), a unified framework that combines the two axes of efficiency inside a single Recursive Transformer. MoR reuses a shared stack of layers across recursion steps to achieve parameter efficiency, while lightweight routers enable adaptive token-level thinking by dynamically assigning different recursion depths to individual tokens. This allows MoR to focus quadratic attention computation only among tokens still active at a given recursion depth, further improving memory access efficiency by selectively caching only their key-value pairs. Beyond these core mechanisms, we also propose a KV sharing variant that reuses KV pairs from the first recursion, specifically designed to decrease prefill latency and memory footprint. Across model scales ranging from 135M to 1.7B parameters, MoR forms a new Pareto frontier: at equal training FLOPs and smaller model sizes, it significantly lowers validation perplexity and improves few-shot accuracy, while delivering higher throughput compared with vanilla and existing recursive baselines. These gains demonstrate that MoR is an effective path towards large-model quality without incurring large-model cost.
>
---
#### [replaced 026] SWI: Speaking with Intent in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2503.21544v2](http://arxiv.org/pdf/2503.21544v2)**

> **作者:** Yuwei Yin; EunJeong Hwang; Giuseppe Carenini
>
> **备注:** Code: https://github.com/YuweiYin/SWI
>
> **摘要:** Intent, typically clearly formulated and planned, functions as a cognitive framework for communication and problem-solving. This paper introduces the concept of Speaking with Intent (SWI) in large language models (LLMs), where the explicitly generated intent encapsulates the model's underlying intention and provides high-level planning to guide subsequent analysis and action. By emulating deliberate and purposeful thoughts in the human mind, SWI is hypothesized to enhance the reasoning capabilities and generation quality of LLMs. Extensive experiments on text summarization, multi-task question answering, and mathematical reasoning benchmarks consistently demonstrate the effectiveness and generalizability of Speaking with Intent over direct generation without explicit intent. Further analysis corroborates the generalizability of SWI under different experimental settings. Moreover, human evaluations verify the coherence, effectiveness, and interpretability of the intent produced by SWI. The promising results in enhancing LLMs with explicit intents pave a new avenue for boosting LLMs' generation and reasoning abilities with cognitive notions.
>
---
#### [replaced 027] Why Does New Knowledge Create Messy Ripple Effects in LLMs?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.12828v3](http://arxiv.org/pdf/2407.12828v3)**

> **作者:** Jiaxin Qin; Zixuan Zhang; Manling Li; Pengfei Yu; Heng Ji
>
> **摘要:** Extensive previous research has focused on post-training knowledge editing (KE) for language models (LMs) to ensure that knowledge remains accurate and up-to-date. One desired property and open question in KE is to let edited LMs correctly handle ripple effects, where LM is expected to answer its logically related knowledge accurately. In this paper, we answer the question of why most KE methods still create messy ripple effects. We conduct extensive analysis and identify a salient indicator, GradSim, that effectively reveals when and why updated knowledge ripples in LMs. GradSim is computed by the cosine similarity between gradients of the original fact and its related knowledge. We observe a strong positive correlation between ripple effect performance and GradSim across different LMs, KE methods, and evaluation metrics. Further investigations into three counter-intuitive failure cases (Negation, Over-Ripple, Multi-Lingual) of ripple effects demonstrate that these failures are often associated with very low GradSim. This finding validates that GradSim is an effective indicator of when knowledge ripples in LMs.
>
---
#### [replaced 028] Vulnerability of LLMs to Vertically Aligned Text Manipulations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20016v3](http://arxiv.org/pdf/2410.20016v3)**

> **作者:** Zhecheng Li; Yiwei Wang; Bryan Hooi; Yujun Cai; Zhen Xiong; Nanyun Peng; Kai-wei Chang
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Vertical text input is commonly encountered in various real-world applications, such as mathematical computations and word-based Sudoku puzzles. While current large language models (LLMs) have excelled in natural language tasks, they remain vulnerable to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: \textit{Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input?} In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) \textit{Chain of Thought (CoT)} reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but \textit{few-shot learning} with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
>
---
#### [replaced 029] Transformers and Ensemble methods: A solution for Hate Speech Detection in Arabic languages
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2303.09823v2](http://arxiv.org/pdf/2303.09823v2)**

> **作者:** Angel Felipe Magnossão de Paula; Imene Bensalem; Paolo Rosso; Wajdi Zaghouani
>
> **备注:** 6 pages, 3 tables
>
> **摘要:** This paper describes our participation in the shared task of hate speech detection, which is one of the subtasks of the CERIST NLP Challenge 2022. Our experiments evaluate the performance of six transformer models and their combination using 2 ensemble approaches. The best results on the training set, in a five-fold cross validation scenario, were obtained by using the ensemble approach based on the majority vote. The evaluation of this approach on the test set resulted in an F1-score of 0.60 and an Accuracy of 0.86.
>
---
#### [replaced 030] Commonsense Reasoning in Arab Culture
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12788v2](http://arxiv.org/pdf/2502.12788v2)**

> **作者:** Abdelrahman Sadallah; Junior Cedric Tonga; Khalid Almubarak; Saeed Almheiri; Farah Atif; Chatrine Qwaider; Karima Kadaoui; Sara Shatnawi; Yaser Alesh; Fajri Koto
>
> **备注:** ACL 2025 - Main
>
> **摘要:** Despite progress in Arabic large language models, such as Jais and AceGPT, their evaluation on commonsense reasoning has largely relied on machine-translated datasets, which lack cultural depth and may introduce Anglocentric biases. Commonsense reasoning is shaped by geographical and cultural contexts, and existing English datasets fail to capture the diversity of the Arab world. To address this, we introduce ArabCulture, a commonsense reasoning dataset in Modern Standard Arabic (MSA), covering cultures of 13 countries across the Gulf, Levant, North Africa, and the Nile Valley. The dataset was built from scratch by engaging native speakers to write and validate culturally relevant questions for their respective countries. ArabCulture spans 12 daily life domains with 54 fine-grained subtopics, reflecting various aspects of social norms, traditions, and everyday experiences. Zero-shot evaluations show that open-weight language models with up to 32B parameters struggle to comprehend diverse Arab cultures, with performance varying across regions. These findings highlight the need for more culturally aware models and datasets tailored to the Arabic-speaking world.
>
---
#### [replaced 031] Texture or Semantics? Vision-Language Models Get Lost in Font Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23768v2](http://arxiv.org/pdf/2503.23768v2)**

> **作者:** Zhecheng Li; Guoxian Song; Yujun Cai; Zhen Xiong; Junsong Yuan; Yiwei Wang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Modern Vision-Language Models (VLMs) exhibit remarkable visual and linguistic capabilities, achieving impressive performance in various tasks such as image recognition and object localization. However, their effectiveness in fine-grained tasks remains an open question. In everyday scenarios, individuals encountering design materials, such as magazines, typography tutorials, research papers, or branding content, may wish to identify aesthetically pleasing fonts used in the text. Given their multimodal capabilities and free accessibility, many VLMs are often considered potential tools for font recognition. This raises a fundamental question: Do VLMs truly possess the capability to recognize fonts? To investigate this, we introduce the Font Recognition Benchmark (FRB), a compact and well-structured dataset comprising 15 commonly used fonts. FRB includes two versions: (i) an easy version, where 10 sentences are rendered in different fonts, and (ii) a hard version, where each text sample consists of the names of the 15 fonts themselves, introducing a stroop effect that challenges model perception. Through extensive evaluation of various VLMs on font recognition tasks, we arrive at the following key findings: (i) Current VLMs exhibit limited font recognition capabilities, with many state-of-the-art models failing to achieve satisfactory performance and being easily affected by the stroop effect introduced by textual information. (ii) Few-shot learning and Chain-of-Thought (CoT) prompting provide minimal benefits in improving font recognition accuracy across different VLMs. (iii) Attention analysis sheds light on the inherent limitations of VLMs in capturing semantic features.
>
---
#### [replaced 032] OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11858v2](http://arxiv.org/pdf/2503.11858v2)**

> **作者:** Ivan Kartáč; Mateusz Lango; Ondřej Dušek
>
> **摘要:** Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate.
>
---
#### [replaced 033] AlphaDPO: Adaptive Reward Margin for Direct Preference Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10148v4](http://arxiv.org/pdf/2410.10148v4)**

> **作者:** Junkang Wu; Xue Wang; Zhengyi Yang; Jiancan Wu; Jinyang Gao; Bolin Ding; Xiang Wang; Xiangnan He
>
> **摘要:** Aligning large language models (LLMs) with human values and intentions is crucial for their utility, honesty, and safety. Reinforcement learning from human feedback (RLHF) is a popular approach to achieve this alignment, but it faces challenges in computational efficiency and training stability. Recent methods like Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO) have proposed offline alternatives to RLHF, simplifying the process by reparameterizing the reward function. However, DPO depends on a potentially suboptimal reference model, and SimPO's assumption of a fixed target reward margin may lead to suboptimal decisions in diverse data settings. In this work, we propose $\alpha$-DPO, an adaptive preference optimization algorithm designed to address these limitations by introducing a dynamic reward margin. Specifically, $\alpha$-DPO employs an adaptive preference distribution, balancing the policy model and the reference model to achieve personalized reward margins. We provide theoretical guarantees for $\alpha$-DPO, demonstrating its effectiveness as a surrogate optimization objective and its ability to balance alignment and diversity through KL divergence control. Empirical evaluations on AlpacaEval 2 and Arena-Hard show that $\alpha$-DPO consistently outperforms DPO and SimPO across various model settings, establishing it as a robust approach for fine-tuning LLMs. Our method achieves significant improvements in win rates, highlighting its potential as a powerful tool for LLM alignment. The code is available at https://github.com/junkangwu/alpha-DPO
>
---
#### [replaced 034] OMoE: Diversifying Mixture of Low-Rank Adaptation by Orthogonal Finetuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.10062v2](http://arxiv.org/pdf/2501.10062v2)**

> **作者:** Jinyuan Feng; Zhiqiang Pu; Tianyi Hu; Dongmin Li; Xiaolin Ai; Huimu Wang
>
> **备注:** This paper is accepted by ECAI 2025
>
> **摘要:** Building mixture-of-experts (MoE) architecture for Low-rank adaptation (LoRA) is emerging as a potential direction in parameter-efficient fine-tuning (PEFT) for its modular design and remarkable performance. However, simply stacking the number of experts cannot guarantee significant improvement. In this work, we first conduct qualitative analysis to indicate that experts collapse to similar representations in vanilla MoE, limiting the capacity of modular design and computational efficiency. Ulteriorly, Our analysis reveals that the performance of previous MoE variants maybe limited by a lack of diversity among experts. Motivated by these findings, we propose Orthogonal Mixture-of-Experts (OMoE), a resource-efficient MoE variant that trains experts in an orthogonal manner to promote diversity. In OMoE, a Gram-Schmidt process is leveraged to enforce that the experts' representations lie within the Stiefel manifold. By applying orthogonal constraints directly to the architecture, OMoE keeps the learning objective unchanged, without compromising optimality. Our method is simple and alleviates memory bottlenecks, as it incurs minimal experts compared to vanilla MoE models. Experiments on diverse commonsense reasoning benchmarks demonstrate that OMoE can consistently achieve stable and efficient performance improvement when compared with the state-of-the-art methods while significantly reducing the number of required experts.
>
---
#### [replaced 035] Dynamic Context Tuning for Retrieval-Augmented Generation: Enhancing Multi-Turn Planning and Tool Adaptation
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.11092v2](http://arxiv.org/pdf/2506.11092v2)**

> **作者:** Jubin Abhishek Soni; Amit Anand; Rajesh Kumar Pandey; Aniket Abhishek Soni
>
> **备注:** We are withdrawing the submission in order to thoroughly revise the work
>
> **摘要:** Retrieval-Augmented Generation (RAG) has significantly advanced large language models (LLMs) by grounding their outputs in external tools and knowledge sources. However, existing RAG systems are typically constrained to static, single-turn interactions with fixed toolsets, making them ill-suited for dynamic domains such as healthcare and smart homes, where user intent, available tools, and contextual factors evolve over time. We present Dynamic Context Tuning (DCT), a lightweight framework that extends RAG to support multi-turn dialogue and evolving tool environments without requiring retraining. DCT integrates an attention-based context cache to track relevant past information, LoRA-based retrieval to dynamically select domain-specific tools, and efficient context compression to maintain inputs within LLM context limits. Experiments on both synthetic and real-world benchmarks show that DCT improves plan accuracy by 14% and reduces hallucinations by 37%, while matching GPT-4 performance at significantly lower cost. Furthermore, DCT generalizes to previously unseen tools, enabling scalable and adaptable AI assistants across a wide range of dynamic environments.
>
---
#### [replaced 036] Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in Product QA Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19545v2](http://arxiv.org/pdf/2502.19545v2)**

> **作者:** Ashley Lewis; Michael White; Jing Liu; Toshiaki Koike-Akino; Kieran Parsons; Ye Wang
>
> **摘要:** The deployment of Large Language Models (LLMs) in customer support is constrained by hallucination (generating false information) and the high cost of proprietary models. To address these challenges, we propose a retrieval-augmented question-answering (QA) pipeline and explore how to balance human input and automation. Using a dataset of questions about a Samsung Smart TV user manual, we demonstrate that synthetic data generated by LLMs outperforms crowdsourced data in reducing hallucination in finetuned models. We also compare self-training (fine-tuning models on their own outputs) and knowledge distillation (fine-tuning on stronger models' outputs, e.g., GPT-4o), and find that self-training achieves comparable hallucination reduction. We conjecture that this surprising finding can be attributed to increased exposure bias issues in the knowledge distillation case and support this conjecture with post hoc analysis. We also improve robustness to unanswerable questions and retrieval failures with contextualized "I don't know" responses. These findings show that scalable, cost-efficient QA systems can be built using synthetic data and self-training with open-source models, reducing reliance on proprietary tools or costly human annotations.
>
---
#### [replaced 037] Dr.Copilot: A Multi-Agent Prompt Optimized Assistant for Improving Patient-Doctor Communication in Romanian
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11299v2](http://arxiv.org/pdf/2507.11299v2)**

> **作者:** Andrei Niculae; Adrian Cosma; Cosmin Dumitrache; Emilian Rǎdoi
>
> **备注:** 10 figures, 2 tables, 2 listings
>
> **摘要:** Text-based telemedicine has become increasingly common, yet the quality of medical advice in doctor-patient interactions is often judged more on how advice is communicated rather than its clinical accuracy. To address this, we introduce Dr. Copilot , a multi-agent large language model (LLM) system that supports Romanian-speaking doctors by evaluating and enhancing the presentation quality of their written responses. Rather than assessing medical correctness, Dr. Copilot provides feedback along 17 interpretable axes. The system comprises of three LLM agents with prompts automatically optimized via DSPy. Designed with low-resource Romanian data and deployed using open-weight models, it delivers real-time specific feedback to doctors within a telemedicine platform. Empirical evaluations and live deployment with 41 doctors show measurable improvements in user reviews and response quality, marking one of the first real-world deployments of LLMs in Romanian medical settings.
>
---
#### [replaced 038] Controlling Language Confusion in Multilingual LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19116v2](http://arxiv.org/pdf/2505.19116v2)**

> **作者:** Nahyun Lee; Yeongseo Woo; Hyunwoo Ko; Guijin Son
>
> **备注:** 4 pages
>
> **摘要:** Large language models often suffer from language confusion, a phenomenon in which responses are partially or entirely generated in unintended languages. This critically degrades the user experience, especially in low-resource settings. We hypothesize that this issue stems from limitations in conventional fine-tuning objectives, such as supervised learning, which optimize the likelihood of correct tokens without explicitly penalizing undesired outputs such as cross-lingual mixing. Analysis of loss trajectories during pretraining further reveals that models fail to distinguish between monolingual and language-mixed texts, highlighting the absence of inherent pressure to avoid such confusion. In this work, we apply ORPO, which adds penalties for unwanted output styles to standard SFT, effectively suppressing language-confused generations. ORPO maintains strong language consistency, even under high decoding temperatures, while preserving general QA performance. Our findings suggest that incorporating appropriate penalty terms can effectively mitigate language confusion in multilingual models, particularly in low-resource scenarios.
>
---
#### [replaced 039] End-to-end Joint Punctuated and Normalized ASR with a Limited Amount of Punctuated Training Data
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2311.17741v3](http://arxiv.org/pdf/2311.17741v3)**

> **作者:** Can Cui; Imran Ahamad Sheikh; Mostafa Sadeghi; Emmanuel Vincent
>
> **摘要:** Joint punctuated and normalized automatic speech recognition (ASR) aims at outputing transcripts with and without punctuation and casing. This task remains challenging due to the lack of paired speech and punctuated text data in most ASR corpora. We propose two approaches to train an end-to-end joint punctuated and normalized ASR system using limited punctuated data. The first approach uses a language model to convert normalized training transcripts into punctuated transcripts. This achieves a better performance on out-of-domain test data, with up to 17% relative Punctuation-Case-aware Word Error Rate (PC-WER) reduction. The second approach uses a single decoder conditioned on the type of output. This yields a 42% relative PC-WER reduction compared to Whisper-base and a 4% relative (normalized) WER reduction compared to the normalized output of a punctuated-only model. Additionally, our proposed model demonstrates the feasibility of a joint ASR system using as little as 5% punctuated training data with a moderate (2.42% absolute) PC-WER increase.
>
---
#### [replaced 040] Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models Aligned with Human Cognitive Principles
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2406.12644v5](http://arxiv.org/pdf/2406.12644v5)**

> **作者:** Devichand Budagam; Ashutosh Kumar; Mahsa Khoshnoodi; Sankalp KJ; Vinija Jain; Aman Chadha
>
> **备注:** 18 pages, 9 figures, KDD workshop on Prompt Optimization 2025
>
> **摘要:** Assessing the effectiveness of large language models (LLMs) in performing different tasks is crucial for understanding their strengths and weaknesses. This paper presents Hierarchical Prompting Taxonomy (HPT), grounded on human cognitive principles and designed to assess LLMs by examining the cognitive demands of various tasks. The HPT utilizes the Hierarchical Prompting Framework (HPF), which structures five unique prompting strategies in a hierarchical order based on their cognitive requirement on LLMs when compared to human mental capabilities. It assesses the complexity of tasks with the Hierarchical Prompting Index (HPI), which demonstrates the cognitive competencies of LLMs across diverse datasets and offers insights into the cognitive demands that datasets place on different LLMs. This approach enables a comprehensive evaluation of an LLMs problem solving abilities and the intricacy of a dataset, offering a standardized metric for task complexity. Extensive experiments with multiple datasets and LLMs show that HPF enhances LLM performance by 2% to 63% compared to baseline performance, with GSM8k being the most cognitively complex task among reasoning and coding tasks with an average HPI of 3.20 confirming the effectiveness of HPT. To support future research and reproducibility in this domain, the implementations of HPT and HPF are available here.
>
---
#### [replaced 041] Executable Functional Abstractions: Inferring Generative Programs for Advanced Math Problems
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.09763v2](http://arxiv.org/pdf/2504.09763v2)**

> **作者:** Zaid Khan; Elias Stengel-Eskin; Archiki Prasad; Jaemin Cho; Mohit Bansal
>
> **备注:** Project Page: https://zaidkhan.me/EFAGen/
>
> **摘要:** Scientists often infer abstract procedures from specific instances of problems and use the abstractions to generate new, related instances. For example, programs encoding the formal rules and properties of a system have been useful in fields ranging from reinforcement learning (procedural environments) to physics (simulation engines). These programs can be seen as functions which execute to different outputs based on their parameterizations (e.g., gridworld configuration or initial physical conditions). We introduce the term EFA (Executable Functional Abstraction) to denote such programs for math problems. EFA-like constructs have been shown to be useful for mathematical reasoning as problem generators for stress-testing models. However, prior work has been limited to automatically constructing abstractions for grade-school math (whose simple rules are easy to encode in programs), while generating EFAs for advanced math has thus far required human engineering. We explore the automatic construction of EFAs for advanced mathematics problems by developing EFAGen, which operationalizes the task of automatically inferring an EFA for a given seed problem and solution as a program synthesis task. We first formalize the properties of any valid EFA as executable unit tests. Using execution feedback from the unit tests, we search over candidate programs sampled from a LLM to find EFA programs that are faithful to the generalized problem and solution class underlying the seed problem. We then apply the tests as a reward signal, training LLMs to become better writers of EFAs. We show that EFAs inferred by EFAGen are faithful to the seed problems, produce learnable problem variations, and that EFAGen can infer EFAs across diverse sources of competition-level math problems. Finally, we show uses of model-written EFAs e.g., finding harder/easier problem variants, as well as data generation.
>
---
#### [replaced 042] Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.15639v2](http://arxiv.org/pdf/2502.15639v2)**

> **作者:** Anirudh Sundar; Sinead Williamson; Katherine Metcalf; Barry-John Theobald; Skyler Seto; Masha Fedzechkina
>
> **备注:** 34 pages
>
> **摘要:** Aligned representations across languages is a desired property in multilingual large language models (mLLMs), as alignment can improve performance in cross-lingual tasks. Typically alignment requires fine-tuning a model, which is computationally expensive, and sizable language data, which often may not be available. A data-efficient alternative to fine-tuning is model interventions -- a method for manipulating model activations to steer generation into the desired direction. We analyze the effect of a popular intervention (finding experts) on the alignment of cross-lingual representations in mLLMs. We identify the neurons to manipulate for a given language and introspect the embedding space of mLLMs pre- and post-manipulation. We show that modifying the mLLM's activations changes its embedding space such that cross-lingual alignment is enhanced. Further, we show that the changes to the embedding space translate into improved downstream performance on retrieval tasks, with up to 2x improvements in top-1 accuracy on cross-lingual retrieval.
>
---
#### [replaced 043] CSSL: Contrastive Self-Supervised Learning for Dependency Parsing on Relatively Free Word Ordered and Morphologically Rich Low Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.06944v2](http://arxiv.org/pdf/2410.06944v2)**

> **作者:** Pretam Ray; Jivnesh Sandhan; Amrith Krishna; Pawan Goyal
>
> **备注:** Accepted at EMNLP 2024 Main (Short), 9 pages, 3 figures, 4 Tables
>
> **摘要:** Neural dependency parsing has achieved remarkable performance for low resource morphologically rich languages. It has also been well-studied that morphologically rich languages exhibit relatively free word order. This prompts a fundamental investigation: Is there a way to enhance dependency parsing performance, making the model robust to word order variations utilizing the relatively free word order nature of morphologically rich languages? In this work, we examine the robustness of graph-based parsing architectures on 7 relatively free word order languages. We focus on scrutinizing essential modifications such as data augmentation and the removal of position encoding required to adapt these architectures accordingly. To this end, we propose a contrastive self-supervised learning method to make the model robust to word order variations. Furthermore, our proposed modification demonstrates a substantial average gain of 3.03/2.95 points in 7 relatively free word order languages, as measured by the UAS/LAS Score metric when compared to the best performing baseline.
>
---
#### [replaced 044] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; cs.NE; math.IT**

- **链接: [http://arxiv.org/pdf/2506.19037v2](http://arxiv.org/pdf/2506.19037v2)**

> **作者:** Omer Luxembourg; Haim Permuter; Eliya Nachmani
>
> **摘要:** Masked diffusion language models (MDLMs) promise fast, non-autoregressive text generation, yet existing samplers, which pick tokens to unmask based on model confidence, ignore interactions when unmasking multiple positions in parallel and effectively reduce to slow, autoregressive behavior. We propose the Dilated Unmasking Scheduler (DUS), an inference-only, planner-model-free method that partitions sequence positions into non-adjacent dilated groups and unmasked them in parallel so as to minimize an upper bound on joint entropy gain at each denoising step. By explicitly trading off the number of network calls against generation quality, DUS recovers most of the performance lost under traditional parallel unmasking strategies. Across math (GSM8K, MATH500), code (HumanEval, MBPP) and general-knowledge benchmarks (BBH, MMLU-Pro), DUS outperforms confidence-based planners, without modifying the underlying denoiser, and reveals the true speed-quality frontier of MDLMs.
>
---
#### [replaced 045] Where Do People Tell Stories Online? Story Detection Across Online Communities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2311.09675v4](http://arxiv.org/pdf/2311.09675v4)**

> **作者:** Maria Antoniak; Joel Mire; Maarten Sap; Elliott Ash; Andrew Piper
>
> **摘要:** Story detection in online communities is a challenging task as stories are scattered across communities and interwoven with non-storytelling spans within a single text. We address this challenge by building and releasing the StorySeeker toolkit, including a richly annotated dataset of 502 Reddit posts and comments, a detailed codebook adapted to the social media context, and models to predict storytelling at the document and span levels. Our dataset is sampled from hundreds of popular English-language Reddit communities ranging across 33 topic categories, and it contains fine-grained expert annotations, including binary story labels, story spans, and event spans. We evaluate a range of detection methods using our data, and we identify the distinctive textual features of online storytelling, focusing on storytelling spans. We illuminate distributional characteristics of storytelling on a large community-centric social media platform, and we also conduct a case study on r/ChangeMyView, where storytelling is used as one of many persuasive strategies, illustrating that our data and models can be used for both inter- and intra-community research. Finally, we discuss implications of our tools and analyses for narratology and the study of online communities.
>
---
#### [replaced 046] Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.00024v2](http://arxiv.org/pdf/2503.00024v2)**

> **作者:** Yanran Chen; Steffen Eger
>
> **备注:** ACL 2025 Camera-ready
>
> **摘要:** Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, human judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness. We further analyze whether 11 LLMs behave like humans in the same scenario, finding that while LLMs generally mirror human patterns, they struggle to capture nuanced emotional effects in individual judgments.
>
---
#### [replaced 047] KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.20923v2](http://arxiv.org/pdf/2506.20923v2)**

> **作者:** Xinping Zhao; Xinshuo Hu; Zifei Shan; Shouzheng Huang; Yao Zhou; Zetian Sun; Zhenyu Liu; Dongfang Li; Xinyuan Wei; Qian Chen; Youcheng Pan; Yang Xiang; Meishan Zhang; Haofen Wang; Jun Yu; Baotian Hu; Min Zhang
>
> **备注:** Technical Report; 26 pages 12 tables 1 figure. arXiv admin note: text overlap with arXiv:2501.01028
>
> **摘要:** In this paper, we propose KaLM-Embedding-V2, a versatile and compact embedding model, which achieves impressive performance in general-purpose text embedding tasks by leveraging superior training techniques and data. Our key innovations include: (1) To better align the architecture with representation learning, we remove the causal attention mask and adopt a fully bidirectional transformer with simple yet effective mean-pooling to produce fixed-length embeddings; (2) We employ a multi-stage training pipeline: (i) pre-training on large-scale weakly supervised open-source corpora; (ii) fine-tuning on high-quality retrieval and non-retrieval datasets; and (iii) model-soup parameter averaging for robust generalization. Besides, we introduce a focal-style reweighting mechanism that concentrates learning on difficult samples and an online hard-negative mixing strategy to continuously enrich hard negatives without expensive offline mining; (3) We collect over 20 categories of data for pre-training and 100 categories of data for fine-tuning, to boost both the performance and generalization of the embedding model. Extensive evaluations on the Massive Text Embedding Benchmark (MTEB) Chinese and English show that our model significantly outperforms others of comparable size, and competes with 3x, 14x, 18x, and 26x larger embedding models, setting a new standard for a versatile and compact embedding model with less than 1B parameters.
>
---
#### [replaced 048] DARE: Diverse Visual Question Answering with Robustness Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.18023v2](http://arxiv.org/pdf/2409.18023v2)**

> **作者:** Hannah Sterz; Jonas Pfeiffer; Ivan Vulić
>
> **摘要:** Vision Language Models (VLMs) extend remarkable capabilities of text-only large language models and vision-only models, and are able to learn from and process multi-modal vision-text input. While modern VLMs perform well on a number of standard image classification and image-text matching tasks, they still struggle with a number of crucial vision-language (VL) reasoning abilities such as counting and spatial reasoning. Moreover, while they might be very brittle to small variations in instructions and/or evaluation protocols, existing benchmarks fail to evaluate their robustness (or rather the lack of it). In order to couple challenging VL scenarios with comprehensive robustness evaluation, we introduce DARE, Diverse Visual Question Answering with Robustness Evaluation, a carefully created and curated multiple-choice VQA benchmark. DARE evaluates VLM performance on five diverse categories and includes four robustness-oriented evaluations based on the variations of: prompts, the subsets of answer options, the output format and the number of correct answers. Among a spectrum of other findings, we report that state-of-the-art VLMs still struggle with questions in most categories and are unable to consistently deliver their peak performance across the tested robustness evaluations. The worst case performance across the subsets of options is up to 34% below the performance in the standard case. The robustness of the open-source VLMs such as LLaVA 1.6 and Idefics2 cannot match the closed-source models such as GPT-4 and Gemini, but even the latter remain very brittle to different variations.
>
---
#### [replaced 049] Detecting PTSD in Clinical Interviews: A Comparative Analysis of NLP Methods and Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.01216v2](http://arxiv.org/pdf/2504.01216v2)**

> **作者:** Feng Chen; Dror Ben-Zeev; Gillian Sparks; Arya Kadakia; Trevor Cohen
>
> **摘要:** Post-Traumatic Stress Disorder (PTSD) remains underdiagnosed in clinical settings, presenting opportunities for automated detection to identify patients. This study evaluates natural language processing approaches for detecting PTSD from clinical interview transcripts. We compared general and mental health-specific transformer models (BERT/RoBERTa), embedding-based methods (SentenceBERT/LLaMA), and large language model prompting strategies (zero-shot/few-shot/chain-of-thought) using the DAIC-WOZ dataset. Domain-specific end-to-end models significantly outperformed general models (Mental-RoBERTa AUPRC=0.675+/-0.084 vs. RoBERTa-base 0.599+/-0.145). SentenceBERT embeddings with neural networks achieved the highest overall performance (AUPRC=0.758+/-0.128). Few-shot prompting using DSM-5 criteria yielded competitive results with two examples (AUPRC=0.737). Performance varied significantly across symptom severity and comorbidity status with depression, with higher accuracy for severe PTSD cases and patients with comorbid depression. Our findings highlight the potential of domain-adapted embeddings and LLMs for scalable screening while underscoring the need for improved detection of nuanced presentations and offering insights for developing clinically viable AI tools for PTSD assessment.
>
---
#### [replaced 050] Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06806v2](http://arxiv.org/pdf/2506.06806v2)**

> **作者:** Subhendu Khatuya; Shashwat Naidu; Saptarshi Ghosh; Pawan Goyal; Niloy Ganguly
>
> **备注:** This work has been accepted to appear at the Association for Computational Linguistics (ACL), 2025
>
> **摘要:** The explosion of textual data has made manual document classification increasingly challenging. To address this, we introduce a robust, efficient domain-agnostic generative model framework for multi-label text classification. Instead of treating labels as mere atomic symbols, our approach utilizes predefined label descriptions and is trained to generate these descriptions based on the input text. During inference, the generated descriptions are matched to the pre-defined labels using a finetuned sentence transformer. We integrate this with a dual-objective loss function, combining cross-entropy loss and cosine similarity of the generated sentences with the predefined target descriptions, ensuring both semantic alignment and accuracy. Our proposed model LAGAMC stands out for its parameter efficiency and versatility across diverse datasets, making it well-suited for practical applications. We demonstrate the effectiveness of our proposed model by achieving new state-of-the-art performances across all evaluated datasets, surpassing several strong baselines. We achieve improvements of 13.94% in Micro-F1 and 24.85% in Macro-F1 compared to the closest baseline across all datasets.
>
---
#### [replaced 051] Domain-Adaptive Small Language Models for Structured Tax Code Prediction
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10880v2](http://arxiv.org/pdf/2507.10880v2)**

> **作者:** Souvik Nath; Sumit Wadhwa; Luis Perez
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Every day, multinational firms process thousands of transactions, each of which must adhere to tax regulations that vary by jurisdiction and are often nuanced. The determination of product and service tax codes, such as HSN or SAC is a major use case in Tax compliance. An accurate determination of such codes is imperative to avoid any tax penalties. This paper proposes a domain-adaptive small language model (SLM) with an encoder-decoder architecture for the enhanced prediction of product and service tax codes. In this approach, we address the problem of predicting hierarchical tax code sequences using unstructured product and services data. We employ an SLM based upon encoder-decoder architecture as this enables sequential generation of tax codes to capture the hierarchical dependencies present within the tax codes. Our experiments demonstrate that encoder-decoder SLMs can be successfully applied to the sequential prediction of structured tax codes, a domain that remains comparatively unexplored in current NLP research. In this paper, we demonstrate the superior performance of the domain-adaptive encoder-decoder SLMs over flat classifiers when applied to the Harmonized System of Nomenclature (HSN), and achieve superior results compared to decoder-only and encoder-only architectures for structured sequence generation tasks. This approach can also be scaled to other government-mandated tax commodity codes, such as United Nations Standard Products and Services Codes (UNSPSC), or Brazil's Nomenclatura Comum do Mercosul (NCM).
>
---
#### [replaced 052] KAT-V1: Kwai-AutoThink Technical Report
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08297v3](http://arxiv.org/pdf/2507.08297v3)**

> **作者:** Zizheng Zhan; Ken Deng; Huaixi Tang; Wen Xiang; Kun Wu; Weihao Li; Wenqiang Zhu; Jingxuan Xu; Lecheng Huang; Zongxian Feng; Shaojie Wang; Shangpeng Yan; Xuxing Chen; Jiaheng Liu; Zhongyuan Peng; Zuchen Gao; Haoyang Huang; Xiaojiang Zhang; Jinghui Wang; Zheng Lin; Mengtong Li; Huiming Wang; Ziqi Zhan; Yanan Wu; Yuanxing Zhang; Jian Yang; Guang Chen; Haotian Zhang; Bin Chen; Bing Yu
>
> **摘要:** We present Kwaipilot-AutoThink (KAT), an open-source 40B large language model developed to address the overthinking problem in reasoning-intensive tasks, where an automatic thinking training paradigm is proposed to dynamically switch between reasoning and non-reasoning modes based on task complexity. Specifically, first, we construct the dual-regime dataset based on a novel tagging pipeline and a multi-agent synthesis strategy, and then we apply Multi-Token Prediction (MTP)-enhanced knowledge distillation, enabling efficient and fine-grained reasoning transfer with minimal pretraining cost. Besides, we implement a cold-start initialization strategy that introduces mode-selection priors using majority-vote signals and intent-aware prompting. Finally, we propose Step-SRPO, a reinforcement learning algorithm that incorporates intermediate supervision into the GRPO framework, offering structured guidance over both reasoning-mode selection and response accuracy. Extensive experiments across multiple benchmarks demonstrate that KAT consistently matches or even outperforms current state-of-the-art models, including DeepSeek-R1-0528 and Qwen3-235B-A22B, across a wide range of reasoning-intensive tasks while reducing token usage. Notably, KAT outperforms all open-source models and even surpasses o3-mini on the leakage-controlled LiveCodeBench Pro. Beyond academic evaluation, KAT has been successfully deployed in Kwaipilot (i.e., Kuaishou's internal coding assistant), where it improves real-world development workflows with high accuracy, efficiency, and controllable reasoning behaviors. Moreover, we are actively training a 200B Mixture-of-Experts (MoE) model with 40B active parameters, and early results already show significant gains, further demonstrating the scalability of the AutoThink paradigm.
>
---
#### [replaced 053] Supporting SENCOTEN Language Documentation Efforts with Automatic Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.10827v2](http://arxiv.org/pdf/2507.10827v2)**

> **作者:** Mengzhe Geng; Patrick Littell; Aidan Pine; PENÁĆ; Marc Tessier; Roland Kuhn
>
> **备注:** Accepted by ComputEL-8
>
> **摘要:** The SENCOTEN language, spoken on the Saanich peninsula of southern Vancouver Island, is in the midst of vigorous language revitalization efforts to turn the tide of language loss as a result of colonial language policies. To support these on-the-ground efforts, the community is turning to digital technology. Automatic Speech Recognition (ASR) technology holds great promise for accelerating language documentation and the creation of educational resources. However, developing ASR systems for SENCOTEN is challenging due to limited data and significant vocabulary variation from its polysynthetic structure and stress-driven metathesis. To address these challenges, we propose an ASR-driven documentation pipeline that leverages augmented speech data from a text-to-speech (TTS) system and cross-lingual transfer learning with Speech Foundation Models (SFMs). An n-gram language model is also incorporated via shallow fusion or n-best restoring to maximize the use of available data. Experiments on the SENCOTEN dataset show a word error rate (WER) of 19.34% and a character error rate (CER) of 5.09% on the test set with a 57.02% out-of-vocabulary (OOV) rate. After filtering minor cedilla-related errors, WER improves to 14.32% (26.48% on unseen words) and CER to 3.45%, demonstrating the potential of our ASR-driven pipeline to support SENCOTEN language documentation.
>
---
#### [replaced 054] TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12891v2](http://arxiv.org/pdf/2505.12891v2)**

> **作者:** Shaohang Wei; Wei Li; Feifan Song; Wen Luo; Tianyi Zhuang; Haochen Tan; Zhijiang Guo; Houfeng Wang
>
> **备注:** Second version
>
> **摘要:** Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , and the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME .
>
---
#### [replaced 055] Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15090v2](http://arxiv.org/pdf/2502.15090v2)**

> **作者:** Masha Fedzechkina; Eleonora Gualdoni; Sinead Williamson; Katherine Metcalf; Skyler Seto; Barry-John Theobald
>
> **摘要:** Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to study representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., ''cat'') and then analyze the corresponding activation patterns. We find that LLM representations captured this way closely align with human representations inferred from behavioral data, matching inter-human alignment levels. Our approach significantly outperforms the alignment captured by word embeddings, which have been the focus of prior work on human-LLM alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts -- we show that LLMs organize concepts in a way that mirrors human concept organization.
>
---
#### [replaced 056] Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01781v2](http://arxiv.org/pdf/2503.01781v2)**

> **作者:** Meghana Rajeev; Rajkumar Ramamurthy; Prapti Trivedi; Vikas Yadav; Oluwanifemi Bamgbose; Sathwik Tejaswi Madhusudan; James Zou; Nazneen Rajani
>
> **备注:** Accepted to CoLM 2025
>
> **摘要:** We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.
>
---
#### [replaced 057] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.06211v2](http://arxiv.org/pdf/2409.06211v2)**

> **作者:** Jaeseong Lee; seung-won hwang; Aurick Qiao; Daniel F Campos; Zhewei Yao; Yuxiong He
>
> **备注:** ACL 2025 main
>
> **摘要:** Mixture-of-experts (MoEs) have been adopted for reducing inference costs by sparsely activating experts in Large language models (LLMs). Despite this reduction, the massive number of experts in MoEs still makes them expensive to serve. In this paper, we study how to address this, by pruning MoEs. Among pruning methodologies, unstructured pruning has been known to achieve the highest performance for a given pruning ratio, compared to structured pruning, since the latter imposes constraints on the sparsification structure. This is intuitive, as the solution space of unstructured pruning subsumes that of structured pruning. However, our counterintuitive finding reveals that expert pruning, a form of structured pruning, can actually precede unstructured pruning to outperform unstructured-only pruning. As existing expert pruning, requiring $O(\frac{k^n}{\sqrt{n}})$ forward passes for $n$ experts, cannot scale for recent MoEs, we propose a scalable alternative with $O(1)$ complexity, yet outperforming the more expensive methods. The key idea is leveraging a latent structure between experts, based on behavior similarity, such that the greedy decision of whether to prune closely captures the joint pruning effect. Ours is highly effective -- for Snowflake Arctic, a 480B-sized MoE with 128 experts, our method needs only one H100 and two hours to achieve nearly no loss in performance with 40% sparsity, even in generative tasks such as GSM8K, where state-of-the-art unstructured pruning fails to. The code will be made publicly available.
>
---
#### [replaced 058] Enhancing Natural Language Inference Performance with Knowledge Graph for COVID-19 Automated Fact-Checking in Indonesian Language
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.00061v2](http://arxiv.org/pdf/2409.00061v2)**

> **作者:** Arief Purnama Muharram; Ayu Purwarianti
>
> **备注:** Submitted to the Journal of ICT Research and Applications (JICTRA)
>
> **摘要:** Automated fact-checking is a key strategy to overcome the spread of COVID-19 misinformation on the internet. These systems typically leverage deep learning approaches through Natural Language Inference (NLI) to verify the truthfulness of information based on supporting evidence. However, one challenge that arises in deep learning is performance stagnation due to a lack of knowledge during training. This study proposes using a Knowledge Graph (KG) as external knowledge to enhance NLI performance for automated COVID-19 fact-checking in the Indonesian language. The proposed model architecture comprises three modules: a fact module, an NLI module, and a classifier module. The fact module processes information from the KG, while the NLI module handles semantic relationships between the given premise and hypothesis. The representation vectors from both modules are concatenated and fed into the classifier module to produce the final result. The model was trained using the generated Indonesian COVID-19 fact-checking dataset and the COVID-19 KG Bahasa Indonesia. Our study demonstrates that incorporating KGs can significantly improve NLI performance in fact-checking, achieving the best accuracy of 0.8616. This suggests that KGs are a valuable component for enhancing NLI performance in automated fact-checking.
>
---
#### [replaced 059] Preventing Rogue Agents Improves Multi-Agent Collaboration
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.05986v2](http://arxiv.org/pdf/2502.05986v2)**

> **作者:** Ohav Barbi; Ori Yoran; Mor Geva
>
> **备注:** Accepted as a spotlight to REALM (First Workshop for Research on Agent Language Models) at ACL 2025
>
> **摘要:** Multi-agent systems, where specialized agents collaborate to solve a shared task hold great potential, from increased modularity to simulating complex environments. However, they also have a major caveat -- a single agent can cause the entire system to fail. Consider a simple game where the knowledge to solve the task is distributed between agents, which share information in a communication channel. At each round, any of the agents can terminate the game and make the final prediction, even if they are uncertain about the outcome of their action. Detection of such rogue agents before they act may prevent the system's failure. In this work, we propose to monitor agents during action prediction and intervene when a future error is likely to occur. To test our approach, we introduce WhoDunitEnv, a multi-agent collaboration environment that allows modular control over task complexity and communication structure. Experiments on WhoDunitEnv, code generation tasks and the GovSim environment for resource sustainability show that our approach leads to substantial performance gains up to 17.4%, 2.5% and 20%, respectively. Thorough analysis shows that our monitors successfully identify critical points of agent confusion and our interventions effectively stop agent errors from propagating.
>
---
#### [replaced 060] Tokenization Standards for Linguistic Integrity: Turkish as a Benchmark
- **分类: cs.CL; 68T50, 68T10; I.2.7; I.2.6; H.3.1**

- **链接: [http://arxiv.org/pdf/2502.07057v2](http://arxiv.org/pdf/2502.07057v2)**

> **作者:** M. Ali Bayram; Ali Arda Fincan; Ahmet Semih Gümüş; Sercan Karakaş; Banu Diri; Savaş Yıldırım
>
> **摘要:** Tokenization is a fundamental preprocessing step in NLP, directly impacting large language models' (LLMs) ability to capture syntactic, morphosyntactic, and semantic structures. This paper introduces a novel framework for systematically evaluating tokenization strategies, addressing challenges in morphologically rich and low-resource languages. Using a Turkish dataset of 6,200 multiple-choice questions from the Massive Multitask Language Understanding (MMLU) benchmark, the framework assesses tokenizers across five key metrics: vocabulary size, token count, processing time, language-specific token percentages (\%TR), and token purity. These metrics provide a structured approach to evaluating how well tokenizers preserve linguistic structures. While \%TR measures the proportion of valid words in the target language, \%Pure assesses the alignment of tokens with meaningful linguistic units, such as roots and valid morphemes, minimizing semantic fragmentation. The findings reveal that \%TR, introduced as a critical metric, exhibits a stronger correlation with downstream performance (e.g., MMLU scores) than token purity, emphasizing its role in improving model accuracy. Additionally, larger model parameters do not necessarily yield better tokenization quality or enhanced results, highlighting the importance of tailored tokenization strategies that prioritize linguistic alignment. This framework sets a new standard for developing robust tokenization methods optimized for morphologically complex and low-resource languages. Future work will refine morphological analysis, explore domain-specific customizations, and conduct cross-linguistic evaluations to further enhance tokenization practices.
>
---
#### [replaced 061] Only a Little to the Left: A Theory-grounded Measure of Political Bias in Large Language Models
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16148v2](http://arxiv.org/pdf/2503.16148v2)**

> **作者:** Mats Faulborn; Indira Sen; Max Pellert; Andreas Spitz; David Garcia
>
> **备注:** Preprint of ACL 2025 paper
>
> **摘要:** Prompt-based language models like GPT4 and LLaMa have been used for a wide variety of use cases such as simulating agents, searching for information, or for content analysis. For all of these applications and others, political biases in these models can affect their performance. Several researchers have attempted to study political bias in language models using evaluation suites based on surveys, such as the Political Compass Test (PCT), often finding a particular leaning favored by these models. However, there is some variation in the exact prompting techniques, leading to diverging findings, and most research relies on constrained-answer settings to extract model responses. Moreover, the Political Compass Test is not a scientifically valid survey instrument. In this work, we contribute a political bias measured informed by political science theory, building on survey design principles to test a wide variety of input prompts, while taking into account prompt sensitivity. We then prompt 11 different open and commercial models, differentiating between instruction-tuned and non-instruction-tuned models, and automatically classify their political stances from 88,110 responses. Leveraging this dataset, we compute political bias profiles across different prompt variations and find that while PCT exaggerates bias in certain models like GPT3.5, measures of political bias are often unstable, but generally more left-leaning for instruction-tuned models. Code and data are available on: https://github.com/MaFa211/theory_grounded_pol_bias
>
---
#### [replaced 062] Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05641v3](http://arxiv.org/pdf/2503.05641v3)**

> **作者:** Justin Chih-Yao Chen; Sukwon Yun; Elias Stengel-Eskin; Tianlong Chen; Mohit Bansal
>
> **备注:** The first three authors contributed equally. Project Page: https://symbolic-moe.github.io/
>
> **摘要:** Combining existing pre-trained expert LLMs is a promising avenue for scalably tackling large-scale and diverse tasks. However, selecting task-level experts is often too coarse-grained, as heterogeneous tasks may require different expertise per instance. To enable adaptive instance-level mixing of pre-trained LLM experts, we propose Symbolic-MoE, a symbolic, text-based, and gradient-free Mixture-of-Experts framework. Symbolic-MoE takes a fine-grained approach to selection by emphasizing skills, e.g., algebra in math or molecular biology in biomedical reasoning. We propose a skill-based recruiting strategy that dynamically selects the most relevant set of expert LLMs for diverse reasoning tasks based on their strengths. Each selected expert then generates its own reasoning, resulting in k outputs from k experts, which are then synthesized into a final high-quality response by an aggregator chosen based on its ability to integrate diverse reasoning outputs. We show that Symbolic-MoE's instance-level expert selection improves performance by a large margin but -- when implemented naively -- can introduce a high computational overhead due to the need for constant model loading and offloading. To address this, we implement a batch strategy that groups instances based on their assigned experts, loading each model only once. This allows us to integrate 16 expert models on 1 GPU with a time cost comparable to or better than prior multi-agent baselines using 4 GPUs. Through extensive evaluations on diverse benchmarks (MMLU-Pro, GPQA, AIME, and MedMCQA), we show that Symbolic-MoE beats strong LLMs like GPT4o-mini, as well as multi-agent approaches, with an absolute avg. gain of 8.15% over the best multi-agent baseline. Moreover, Symbolic-MoE generalizes well to unseen tasks and removes the need for expensive multi-round discussions, outperforming discussion baselines with less computation.
>
---
#### [replaced 063] Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.05375v2](http://arxiv.org/pdf/2411.05375v2)**

> **作者:** Mubashara Akhtar; Michael Schlichtkrull; Andreas Vlachos
>
> **备注:** Accepted at TACL
>
> **摘要:** Current automated fact-checking (AFC) approaches typically evaluate evidence either implicitly via the predicted verdicts or through exact matches with predefined closed knowledge sources, such as Wikipedia. However, these methods are limited due to their reliance on evaluation metrics originally designed for other purposes and constraints from closed knowledge sources. In this work, we introduce \textbf{\textcolor{skyblue}{Ev\textsuperscript{2}}\textcolor{orangebrown}{R}} which combines the strengths of reference-based evaluation and verdict-level proxy scoring. Ev\textsuperscript{2}R jointly assesses how well the evidence aligns with the gold references and how reliably it supports the verdict, addressing the shortcomings of prior methods. We evaluate Ev\textsuperscript{2}R against three types of evidence evaluation approaches: reference-based, proxy-reference, and reference-less baselines. Assessments against human ratings and adversarial tests demonstrate that Ev\textsuperscript{2}R consistently outperforms existing scoring approaches in accuracy and robustness. It achieves stronger correlation with human judgments and greater robustness to adversarial perturbations, establishing it as a reliable metric for evidence evaluation in AFC.\footnote{Code is available at \href{https://github.com/mubasharaak/fc-evidence-evaluation}{https://github.com/mubasharaak/fc-evidence-evaluation}.}
>
---
#### [replaced 064] DRS: Deep Question Reformulation With Structured Output
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.17993v5](http://arxiv.org/pdf/2411.17993v5)**

> **作者:** Zhecheng Li; Yiwei Wang; Bryan Hooi; Yujun Cai; Nanyun Peng; Kai-Wei Chang
>
> **备注:** Findings of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Question answering represents a core capability of large language models (LLMs). However, when individuals encounter unfamiliar knowledge in texts, they often formulate questions that the text itself cannot answer due to insufficient understanding of the underlying information. Recent studies reveal that while LLMs can detect unanswerable questions, they struggle to assist users in reformulating these questions. Even advanced models like GPT-3.5 demonstrate limited effectiveness in this regard. To address this limitation, we propose DRS: Deep Question Reformulation with Structured Output, a novel zero-shot method aimed at enhancing LLMs ability to assist users in reformulating questions to extract relevant information from new documents. DRS combines the strengths of LLMs with a DFS-based algorithm to iteratively explore potential entity combinations and constrain outputs using predefined entities. This structured approach significantly enhances the reformulation capabilities of LLMs. Comprehensive experimental evaluations demonstrate that DRS improves the reformulation accuracy of GPT-3.5 from 23.03% to 70.42%, while also enhancing the performance of open-source models, such as Gemma2-9B, from 26.35% to 56.75%.
>
---
#### [replaced 065] Orchestrator-Agent Trust: A Modular Agentic AI Visual Classification System with Trust-Aware Orchestration and RAG-Based Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10571v2](http://arxiv.org/pdf/2507.10571v2)**

> **作者:** Konstantinos I. Roumeliotis; Ranjan Sapkota; Manoj Karkee; Nikolaos D. Tselikas
>
> **摘要:** Modern Artificial Intelligence (AI) increasingly relies on multi-agent architectures that blend visual and language understanding. Yet, a pressing challenge remains: How can we trust these agents especially in zero-shot settings with no fine-tuning? We introduce a novel modular Agentic AI visual classification framework that integrates generalist multimodal agents with a non-visual reasoning orchestrator and a Retrieval-Augmented Generation (RAG) module. Applied to apple leaf disease diagnosis, we benchmark three configurations: (I) zero-shot with confidence-based orchestration, (II) fine-tuned agents with improved performance, and (III) trust-calibrated orchestration enhanced by CLIP-based image retrieval and re-evaluation loops. Using confidence calibration metrics (ECE, OCR, CCC), the orchestrator modulates trust across agents. Our results demonstrate a 77.94\% accuracy improvement in the zero-shot setting using trust-aware orchestration and RAG, achieving 85.63\% overall. GPT-4o showed better calibration, while Qwen-2.5-VL displayed overconfidence. Furthermore, image-RAG grounded predictions with visually similar cases, enabling correction of agent overconfidence via iterative re-evaluation. The proposed system separates perception (vision agents) from meta-reasoning (orchestrator), enabling scalable and interpretable multi-agent AI. This blueprint is extensible to diagnostics, biology, and other trust-critical domains. All models, prompts, results, and system components including the complete software source code are openly released to support reproducibility, transparency, and community benchmarking at Github: https://github.com/Applied-AI-Research-Lab/Orchestrator-Agent-Trust
>
---
#### [replaced 066] A Survey of Context Engineering for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13334v2](http://arxiv.org/pdf/2507.13334v2)**

> **作者:** Lingrui Mei; Jiayu Yao; Yuyao Ge; Yiwei Wang; Baolong Bi; Yujun Cai; Jiazhi Liu; Mingyu Li; Zhong-Zhi Li; Duzhen Zhang; Chenlin Zhou; Jiayi Mao; Tianze Xia; Jiafeng Guo; Shenghua Liu
>
> **备注:** ongoing work; 166 pages, 1411 citations
>
> **摘要:** The performance of Large Language Models (LLMs) is fundamentally determined by the contextual information provided during inference. This survey introduces Context Engineering, a formal discipline that transcends simple prompt design to encompass the systematic optimization of information payloads for LLMs. We present a comprehensive taxonomy decomposing Context Engineering into its foundational components and the sophisticated implementations that integrate them into intelligent systems. We first examine the foundational components: context retrieval and generation, context processing and context management. We then explore how these components are architecturally integrated to create sophisticated system implementations: retrieval-augmented generation (RAG), memory systems and tool-integrated reasoning, and multi-agent systems. Through this systematic analysis of over 1400 research papers, our survey not only establishes a technical roadmap for the field but also reveals a critical research gap: a fundamental asymmetry exists between model capabilities. While current models, augmented by advanced context engineering, demonstrate remarkable proficiency in understanding complex contexts, they exhibit pronounced limitations in generating equally sophisticated, long-form outputs. Addressing this gap is a defining priority for future research. Ultimately, this survey provides a unified framework for both researchers and engineers advancing context-aware AI.
>
---
#### [replaced 067] Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document Restoration
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05108v2](http://arxiv.org/pdf/2507.05108v2)**

> **作者:** Yuyi Zhang; Peirong Zhang; Zhenhua Yang; Pengyu Yan; Yongxin Shi; Pengwei Liu; Fengjun Guo; Lianwen Jin
>
> **摘要:** Historical documents represent an invaluable cultural heritage, yet have undergone significant degradation over time through tears, water erosion, and oxidation. Existing Historical Document Restoration (HDR) methods primarily focus on single modality or limited-size restoration, failing to meet practical needs. To fill this gap, we present a full-page HDR dataset (FPHDR) and a novel automated HDR solution (AutoHDR). Specifically, FPHDR comprises 1,633 real and 6,543 synthetic images with character-level and line-level locations, as well as character annotations in different damage grades. AutoHDR mimics historians' restoration workflows through a three-stage approach: OCR-assisted damage localization, vision-language context text prediction, and patch autoregressive appearance restoration. The modular architecture of AutoHDR enables seamless human-machine collaboration, allowing for flexible intervention and optimization at each restoration stage. Experiments demonstrate AutoHDR's remarkable performance in HDR. When processing severely damaged documents, our method improves OCR accuracy from 46.83% to 84.05%, with further enhancement to 94.25% through human-machine collaboration. We believe this work represents a significant advancement in automated historical document restoration and contributes substantially to cultural heritage preservation. The model and dataset are available at https://github.com/SCUT-DLVCLab/AutoHDR.
>
---
#### [replaced 068] Lizard: An Efficient Linearization Framework for Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09025v2](http://arxiv.org/pdf/2507.09025v2)**

> **作者:** Chien Van Nguyen; Ruiyi Zhang; Hanieh Deilamsalehy; Puneet Mathur; Viet Dac Lai; Haoliang Wang; Jayakumar Subramanian; Ryan A. Rossi; Trung Bui; Nikos Vlassis; Franck Dernoncourt; Thien Huu Nguyen
>
> **备注:** 15 pages
>
> **摘要:** We propose Lizard, a linearization framework that transforms pretrained Transformer-based Large Language Models (LLMs) into flexible, subquadratic architectures for infinite-context generation. Transformer-based LLMs face significant memory and computational bottlenecks as context lengths increase, due to the quadratic complexity of softmax attention and the growing key-value (KV) cache. Lizard addresses these limitations by introducing a subquadratic attention mechanism that closely approximates softmax attention while preserving the output quality. Unlike previous linearization methods, which are often limited by fixed model structures and therefore exclude gating mechanisms, Lizard incorporates a gating module inspired by recent state-of-the-art linear models. This enables adaptive memory control, supports constant-memory inference, offers strong length generalization, and allows more flexible model design. Lizard combines gated linear attention for global context compression with sliding window attention enhanced by meta memory, forming a hybrid mechanism that captures both long-range dependencies and fine-grained local interactions. Moreover, we introduce a hardware-aware algorithm that accelerates the training speed of our models. Extensive experiments show that Lizard achieves near-lossless recovery of the teacher model's performance across standard language modeling tasks, while significantly outperforming previous linearization methods. On the 5-shot MMLU benchmark, Lizard improves over prior models by 18 points and shows significant improvements on associative recall tasks.
>
---
#### [replaced 069] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.06656v3](http://arxiv.org/pdf/2409.06656v3)**

> **作者:** Taejin Park; Ivan Medennikov; Kunal Dhawan; Weiqing Wang; He Huang; Nithin Rao Koluguri; Krishna C. Puvvada; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Published at ICML 2025
>
> **摘要:** Sortformer is an encoder-based speaker diarization model designed for supervising speaker tagging in speech-to-text models. Instead of relying solely on permutation invariant loss (PIL), Sortformer introduces Sort Loss to resolve the permutation problem, either independently or in tandem with PIL. In addition, we propose a streamlined multi-speaker speech-to-text architecture that leverages Sortformer for speaker supervision, embedding speaker labels into the encoder using sinusoidal kernel functions. This design addresses the speaker permutation problem through sorted objectives, effectively bridging timestamps and tokens to supervise speaker labels in the output transcriptions. Experiments demonstrate that Sort Loss can boost speaker diarization performance, and incorporating the speaker supervision from Sortformer improves multi-speaker transcription accuracy. We anticipate that the proposed Sortformer and multi-speaker architecture will enable the seamless integration of speaker tagging capabilities into foundational speech-to-text systems and multimodal large language models (LLMs), offering an easily adoptable and user-friendly mechanism to enhance their versatility and performance in speaker-aware tasks. The code and trained models are made publicly available through the NVIDIA NeMo Framework.
>
---
#### [replaced 070] CCSBench: Evaluating Compositional Controllability in LLMs for Scientific Document Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12601v2](http://arxiv.org/pdf/2410.12601v2)**

> **作者:** Yixi Ding; Jiaying Wu; Tongyao Zhu; Yanxia Qin; Qian Liu; Min-Yen Kan
>
> **备注:** Accepted to KDD 2025 SciSoc LLM Workshop: Large Language Models for Scientific and Societal Advances
>
> **摘要:** To broaden the dissemination of scientific knowledge to diverse audiences, it is desirable for scientific document summarization systems to simultaneously control multiple attributes such as length and empirical focus. However, existing research typically focuses on controlling single attributes, leaving the compositional control of multiple attributes underexplored. To address this gap, we introduce CCSBench, the first evaluation benchmark for compositional controllable summarization in the scientific domain. Our benchmark enables fine-grained control over both explicit attributes (e.g., length), which are objective and straightforward, and implicit attributes (e.g., conceptual or empirical focus), which are more subjective and abstract. We conduct extensive experiments using various large language models (LLMs) under various settings, including in-context learning, parameter-efficient fine-tuning, and two-stage modular methods for balancing control over different attributes. Our findings reveal significant limitations in LLMs capabilities in balancing trade-offs between control attributes, especially implicit ones that require deeper understanding and abstract reasoning.
>
---
#### [replaced 071] BriLLM: Brain-inspired Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11299v5](http://arxiv.org/pdf/2503.11299v5)**

> **作者:** Hai Zhao; Hongqiu Wu; Dongjie Yang; Anni Zou; Jiale Hong
>
> **摘要:** This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above.
>
---
#### [replaced 072] DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11558v3](http://arxiv.org/pdf/2506.11558v3)**

> **作者:** Bo-Cheng Chiu; Jen-Jee Chen; Yu-Chee Tseng; Feng-Chi Chen
>
> **摘要:** Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
>
---
#### [replaced 073] Growing a Twig to Accelerate Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14075v2](http://arxiv.org/pdf/2503.14075v2)**

> **作者:** Zhenwei Shao; Mingyang Wang; Zhou Yu; Wenwen Pan; Yan Yang; Tao Wei; Hongyuan Zhang; Ning Mao; Wei Chen; Jun Yu
>
> **备注:** accepted at ICCV 2025
>
> **摘要:** Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (e.g., 30 tokens). To address the limitations above, we present TwigVLM -- a simple and general architecture by growing a lightweight twig upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves better accuracy retention by employing a twig-guided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods.
>
---
#### [replaced 074] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12829v2](http://arxiv.org/pdf/2502.12829v2)**

> **作者:** Mukhammed Togmanov; Nurdaulet Mukhituly; Diana Turmakhan; Jonibek Mansurov; Maiya Goloburda; Akhmed Sakip; Zhuohan Xie; Yuxia Wang; Bekassyl Syzdykov; Nurkhan Laiyk; Alham Fikri Aji; Ekaterina Kochmar; Preslav Nakov; Fajri Koto
>
> **摘要:** Despite having a population of twenty million, Kazakhstan's culture and language remain underrepresented in the field of natural language processing. Although large language models (LLMs) continue to advance worldwide, progress in Kazakh language has been limited, as seen in the scarcity of dedicated models and benchmark evaluations. To address this gap, we introduce KazMMLU, the first MMLU-style dataset specifically designed for Kazakh language. KazMMLU comprises 23,000 questions that cover various educational levels, including STEM, humanities, and social sciences, sourced from authentic educational materials and manually validated by native speakers and educators. The dataset includes 10,969 Kazakh questions and 12,031 Russian questions, reflecting Kazakhstan's bilingual education system and rich local context. Our evaluation of several state-of-the-art multilingual models (Llama-3.1, Qwen-2.5, GPT-4, and DeepSeek V3) demonstrates substantial room for improvement, as even the best-performing models struggle to achieve competitive performance in Kazakh and Russian. These findings underscore significant performance gaps compared to high-resource languages. We hope that our dataset will enable further research and development of Kazakh-centric LLMs. Data and code will be made available upon acceptance.
>
---
#### [replaced 075] AutoGen Driven Multi Agent Framework for Iterative Crime Data Analysis and Prediction
- **分类: cs.MA; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11475v2](http://arxiv.org/pdf/2506.11475v2)**

> **作者:** Syeda Kisaa Fatima; Tehreem Zubair; Noman Ahmed; Asifullah Khan
>
> **摘要:** This paper introduces LUCID-MA (Learning and Understanding Crime through Dialogue of Multiple Agents), an innovative AI powered framework where multiple AI agents collaboratively analyze and understand crime data. Our system that consists of three core components: an analysis assistant that highlights spatiotemporal crime patterns; a feedback component that reviews and refines analytical results; and a prediction component that forecasts future crime trends. With a well-designed prompt and the LLaMA-2-13B-Chat-GPTQ model, it runs completely offline and allows the agents undergo self-improvement through 100 rounds of communication with less human interaction. A scoring function is incorporated to evaluate agent performance, providing visual plots to track learning progress. This work demonstrates the potential of AutoGen-style agents for autonomous, scalable, and iterative analysis in social science domains, maintaining data privacy through offline execution. It also showcases a computational model with emergent intelligence, where the system's global behavior emerges from the interactions of its agents. This emergent behavior manifests as enhanced individual agent performance, driven by collaborative dialogue between the LLM-based agents.
>
---
#### [replaced 076] Towards the Next Frontier in Speech Representation Learning Using Disentanglement
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.02543v2](http://arxiv.org/pdf/2407.02543v2)**

> **作者:** Varun Krishna; Sriram Ganapathy
>
> **备注:** There were some bugs in the Code that was used to produce the results in the paper. The results reported in the paper are not valid
>
> **摘要:** The popular frameworks for self-supervised learning of speech representations have largely focused on frame-level masked prediction of speech regions. While this has shown promising downstream task performance for speech recognition and related tasks, this has largely ignored factors of speech that are encoded at coarser level, like characteristics of the speaker or channel that remain consistent through-out a speech utterance. In this work, we propose a framework for Learning Disentangled Self Supervised (termed as Learn2Diss) representations of speech, which consists of frame-level and an utterance-level encoder modules. The two encoders are initially learned independently, where the frame-level model is largely inspired by existing self supervision techniques, thereby learning pseudo-phonemic representations, while the utterance-level encoder is inspired by constrastive learning of pooled embeddings, thereby learning pseudo-speaker representations. The joint learning of these two modules consists of disentangling the two encoders using a mutual information based criterion. With several downstream evaluation experiments, we show that the proposed Learn2Diss achieves state-of-the-art results on a variety of tasks, with the frame-level encoder representations improving semantic tasks, while the utterance-level representations improve non-semantic tasks.
>
---
#### [replaced 077] clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05445v2](http://arxiv.org/pdf/2505.05445v2)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** 31 pages
>
> **摘要:** The emergence of instruction-tuned large language models (LLMs) has advanced the field of dialogue systems, enabling both realistic user simulations and robust multi-turn conversational agents. However, existing research often evaluates these components in isolation-either focusing on a single user simulator or a specific system design-limiting the generalisability of insights across architectures and configurations. In this work, we propose clem todd (chat-optimized LLMs for task-oriented dialogue systems development), a flexible framework for systematically evaluating dialogue systems under consistent conditions. clem todd enables detailed benchmarking across combinations of user simulators and dialogue systems, whether existing models from literature or newly developed ones. It supports plug-and-play integration and ensures uniform datasets, evaluation metrics, and computational constraints. We showcase clem todd's flexibility by re-evaluating existing task-oriented dialogue systems within this unified setup and integrating three newly proposed dialogue systems into the same evaluation pipeline. Our results provide actionable insights into how architecture, scale, and prompting strategies affect dialogue performance, offering practical guidance for building efficient and effective conversational AI systems.
>
---
#### [replaced 078] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.03040v2](http://arxiv.org/pdf/2501.03040v2)**

> **作者:** Duygu Sezen Islakoglu; Jan-Christoph Kalo
>
> **备注:** Accepted to ACL 2025. Results on a larger test set. 13 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success in various NLP tasks, yet they still face significant challenges in reasoning and arithmetic. Temporal reasoning, a critical component of natural language understanding, has raised increasing research attention. However, comprehensive testing of Allen's interval relations (e.g., before, after, during) -- a fundamental framework for temporal relationships -- remains underexplored. To fill this gap, we present ChronoSense, a new benchmark for evaluating LLMs' temporal understanding. It includes 16 tasks, focusing on identifying the Allen relation between two temporal events and temporal arithmetic, using both abstract events and real-world data from Wikidata. We assess the performance of seven recent LLMs using this benchmark and the results indicate that models handle Allen relations, even symmetrical ones, quite differently. Moreover, the findings suggest that the models may rely on memorization to answer time-related questions. Overall, the models' low performance highlights the need for improved temporal understanding in LLMs and ChronoSense offers a robust framework for future research in this area. Our dataset and the source code are available at https://github.com/duyguislakoglu/chronosense.
>
---
#### [replaced 079] Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08622v2](http://arxiv.org/pdf/2505.08622v2)**

> **作者:** Donghoon Kim; Minji Bae; Kyuhong Shim; Byonghyo Shim
>
> **备注:** ICLR 2025 (Official Code: https://github.com/DonghoonKim-1938/VGD)
>
> **摘要:** Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models.
>
---
#### [replaced 080] MKE-Coder: Multi-Axial Knowledge with Evidence Verification in ICD Coding for Chinese EMRs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14916v3](http://arxiv.org/pdf/2502.14916v3)**

> **作者:** Xinxin You; Xien Liu; Xue Yang; Ziyi Wang; Ji Wu
>
> **备注:** We have decided to withdraw this manuscript in order to allow for further revisions and additional experiments
>
> **摘要:** The task of automatically coding the International Classification of Diseases (ICD) in the medical field has been well-established and has received much attention. Automatic coding of the ICD in the medical field has been successful in English but faces challenges when dealing with Chinese electronic medical records (EMRs). The first issue lies in the difficulty of extracting disease code-related information from Chinese EMRs, primarily due to the concise writing style and specific internal structure of the EMRs. The second problem is that previous methods have failed to leverage the disease-based multi-axial knowledge and lack of association with the corresponding clinical evidence. This paper introduces a novel framework called MKE-Coder: Multi-axial Knowledge with Evidence verification in ICD coding for Chinese EMRs. Initially, we identify candidate codes for the diagnosis and categorize each of them into knowledge under four coding axes.Subsequently, we retrieve corresponding clinical evidence from the comprehensive content of EMRs and filter credible evidence through a scoring model. Finally, to ensure the validity of the candidate code, we propose an inference module based on the masked language modeling strategy. This module verifies that all the axis knowledge associated with the candidate code is supported by evidence and provides recommendations accordingly. To evaluate the performance of our framework, we conduct experiments using a large-scale Chinese EMR dataset collected from various hospitals. The experimental results demonstrate that MKE-Coder exhibits significant superiority in the task of automatic ICD coding based on Chinese EMRs. In the practical evaluation of our method within simulated real coding scenarios, it has been demonstrated that our approach significantly aids coders in enhancing both their coding accuracy and speed.
>
---
#### [replaced 081] APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.03601v4](http://arxiv.org/pdf/2504.03601v4)**

> **作者:** Akshara Prabhakar; Zuxin Liu; Ming Zhu; Jianguo Zhang; Tulika Awalgaonkar; Shiyu Wang; Zhiwei Liu; Haolin Chen; Thai Hoang; Juan Carlos Niebles; Shelby Heinecke; Weiran Yao; Huan Wang; Silvio Savarese; Caiming Xiong
>
> **备注:** 12 pages plus references and appendices; fixes typo in fig 6
>
> **摘要:** Training effective AI agents for multi-turn interactions requires high-quality data that captures realistic human-agent dynamics, yet such data is scarce and expensive to collect manually. We introduce APIGen-MT, a two-phase framework that generates verifiable and diverse multi-turn agent data. In the first phase, our agentic pipeline produces detailed task blueprints with ground-truth actions, leveraging a committee of LLM reviewers and iterative feedback loops. These blueprints are then transformed into complete interaction trajectories through simulated human-agent interplay. We train a family of models -- the xLAM-2-fc-r series with sizes ranging from 1B to 70B parameters. Our models outperform frontier models such as GPT-4o and Claude 3.5 on $\tau$-bench and BFCL benchmarks, with the smaller models surpassing their larger counterparts, particularly in multi-turn settings, while maintaining superior consistency across multiple trials. Comprehensive experiments demonstrate that our verified blueprint-to-details approach yields high-quality training data, enabling the development of more reliable, efficient, and capable agents. We open-source 5K synthetic data trajectories and the trained xLAM-2-fc-r models to advance research in AI agents. Models at https://huggingface.co/collections/Salesforce/xlam-2-67ef5be12949d8dcdae354c4; Dataset at https://huggingface.co/datasets/Salesforce/APIGen-MT-5k and Website at https://apigen-mt.github.io
>
---
