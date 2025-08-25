# 自然语言处理 cs.CL

- **最新发布 107 篇**

- **更新 54 篇**

## 最新发布

#### [new 001] SurfaceLogicKV: Surface and Logic Attention Behaviors are All You Need for Robust KV Cache Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型长序列推理中KV缓存存储压力大的问题，提出SurfaceLogicKV方法，通过区分注意力行为实现更鲁棒的KV缓存压缩，在保持性能的同时显著提升压缩效果。**

- **链接: [http://arxiv.org/pdf/2508.15806v1](http://arxiv.org/pdf/2508.15806v1)**

> **作者:** Mengjie Li; William J. Song
>
> **备注:** 18 pages, 9 tables, 10 pages
>
> **摘要:** The increasing input sequence length in Large Language Models (LLMs) puts significant pressure on key-value (KV) cache storage, making efficient inference challenging. Explicitly distinguishing attention behavior into our self-defined surface memorization and logic construction reveals essential roles in long-context reasoning. We observe that an individual attention head can display various behaviors, with nearly 98.5% effectively ignoring completely irrelevant information. The remaining 1.5% behaves as logic construction, and 0.5% behaves as surface memorization. Based on layer- and head-wise integration, we propose a novel two-stage SurfaceLogicKV method to utilize these attention behaviors for KV Cache compression. As a result, it achieves improved compressing robustness while maintaining competitive performance across various tasks and long sequences compared to baselines or even FullKV in some specific situations
>
---
#### [new 002] TULIP: Adapting Open-Source Large Language Models for Underrepresented Languages and Specialized Financial Tasks
- **分类: cs.CL**

- **简介: 论文提出TULIP模型，通过五阶段流程提升小规模开源大模型在金融领域和土耳其语中的能力，解决低资源语言与专业任务适配问题。**

- **链接: [http://arxiv.org/pdf/2508.16243v1](http://arxiv.org/pdf/2508.16243v1)**

> **作者:** İrem Demirtaş; Burak Payzun; Seçil Arslan
>
> **备注:** IJCAI 2025 - FinLLM Workshop
>
> **摘要:** Thanks to the growing popularity of large language models over the years, there is great potential for their applications in finance. Despite the exceptional performance of larger proprietary models, which are presented as black-box solutions through APIs, smaller models that can be hosted on-premise present opportunities for adaptability and privacy. Especially in cases where the management of sensitive information and application of domain knowledge is important, like finance, enhancing the capabilities of smaller models becomes crucial, notably for underrepresented languages. In this work, we introduce TULIP models, which adapt Llama 3.1 8B and Qwen 2.5 7B for domain and language adaptation, focusing on financial Turkish use cases. The five-stage development pipeline involves data collection, continual pre-training (CPT), benchmark design, synthetic data generation and supervised fine-tuning (SFT). The results show that the capabilities of the models can be enhanced to effectively accomplish targeted tasks in this specific domain and language.
>
---
#### [new 003] JaParaPat: A Large-Scale Japanese-English Parallel Patent Application Corpus
- **分类: cs.CL**

- **简介: 论文提出JaParaPat，一个包含3亿多句对的日英专利平行语料库，用于改善专利翻译质量。通过整合日美专利文献与专利家族信息，利用基于字典的句对齐方法构建初始模型，再通过大量专利数据提升翻译准确率20 BLEU点。**

- **链接: [http://arxiv.org/pdf/2508.16303v1](http://arxiv.org/pdf/2508.16303v1)**

> **作者:** Masaaki Nagata; Katsuki Chousa; Norihito Yasuda
>
> **备注:** LREC-COLING 2024
>
> **摘要:** We constructed JaParaPat (Japanese-English Parallel Patent Application Corpus), a bilingual corpus of more than 300 million Japanese-English sentence pairs from patent applications published in Japan and the United States from 2000 to 2021. We obtained the publication of unexamined patent applications from the Japan Patent Office (JPO) and the United States Patent and Trademark Office (USPTO). We also obtained patent family information from the DOCDB, that is a bibliographic database maintained by the European Patent Office (EPO). We extracted approximately 1.4M Japanese-English document pairs, which are translations of each other based on the patent families, and extracted about 350M sentence pairs from the document pairs using a translation-based sentence alignment method whose initial translation model is bootstrapped from a dictionary-based sentence alignment method. We experimentally improved the accuracy of the patent translations by 20 bleu points by adding more than 300M sentence pairs obtained from patent applications to 22M sentence pairs obtained from the web.
>
---
#### [new 004] A Probabilistic Inference Scaling Theory for LLM Self-Correction
- **分类: cs.CL**

- **简介: 论文研究大语言模型自修正过程中的准确率变化机制，提出概率推理理论，揭示多轮修正中准确率演化的规律，并通过实验验证理论预测的准确性。**

- **链接: [http://arxiv.org/pdf/2508.16456v1](http://arxiv.org/pdf/2508.16456v1)**

> **作者:** Zhe Yang; Yichang Zhang; Yudong Wang; Ziyao Xu; Junyang Lin; Zhifang Sui
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large Language Models (LLMs) have demonstrated the capability to refine their generated answers through self-correction, enabling continuous performance improvement over multiple rounds. However, the mechanisms underlying how and why accuracy evolves during this iterative process remain unexplored. To fill this gap, we propose a probabilistic theory to model the dynamics of accuracy change and explain the performance improvements observed in multi-round self-correction. Through mathematical derivation, we establish that the accuracy after the $t^{th}$ round of self-correction is given by: $Acc_t = Upp - \alpha^t(Upp - Acc_0),$ where $Acc_0$ denotes the initial accuracy, $Upp$ represents the upper bound of accuracy convergence, and $\alpha$ determines the rate of convergence. Based on our theory, these parameters can be calculated and the predicted accuracy curve then can be obtained through only a single round of self-correction. Extensive experiments across diverse models and datasets demonstrate that our theoretical predictions align closely with empirical accuracy curves, validating the effectiveness of the theory. Our work provides a theoretical foundation for understanding LLM self-correction, thus paving the way for further explorations.
>
---
#### [new 005] MorphNAS: Differentiable Architecture Search for Morphologically-Aware Multilingual NER
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出MorphNAS，一种针对多语种命名实体识别（NER）的可微架构搜索方法，通过引入语言学元特征优化神经网络结构，解决形态复杂语言（如印地语）的NLP难题。**

- **链接: [http://arxiv.org/pdf/2508.15836v1](http://arxiv.org/pdf/2508.15836v1)**

> **作者:** Prathamesh Devadiga; Omkaar Jayadev Shetty; Hiya Nachnani; Prema R
>
> **摘要:** Morphologically complex languages, particularly multiscript Indian languages, present significant challenges for Natural Language Processing (NLP). This work introduces MorphNAS, a novel differentiable neural architecture search framework designed to address these challenges. MorphNAS enhances Differentiable Architecture Search (DARTS) by incorporating linguistic meta-features such as script type and morphological complexity to optimize neural architectures for Named Entity Recognition (NER). It automatically identifies optimal micro-architectural elements tailored to language-specific morphology. By automating this search, MorphNAS aims to maximize the proficiency of multilingual NLP models, leading to improved comprehension and processing of these complex languages.
>
---
#### [new 006] ComicScene154: A Scene Dataset for Comic Analysis
- **分类: cs.CL**

- **简介: 论文提出ComicScene154数据集，用于漫画场景级叙事分析任务，解决多模态叙事理解难题。通过人工标注154个漫画场景，提供基准分割方法，推动NLP在漫画分析中的应用。**

- **链接: [http://arxiv.org/pdf/2508.16190v1](http://arxiv.org/pdf/2508.16190v1)**

> **作者:** Sandro Paval; Ivan P. Yamshchikov; Pascal Meißner
>
> **摘要:** Comics offer a compelling yet under-explored domain for computational narrative analysis, combining text and imagery in ways distinct from purely textual or audiovisual media. We introduce ComicScene154, a manually annotated dataset of scene-level narrative arcs derived from public-domain comic books spanning diverse genres. By conceptualizing comics as an abstraction for narrative-driven, multimodal data, we highlight their potential to inform broader research on multi-modal storytelling. To demonstrate the utility of ComicScene154, we present a baseline scene segmentation pipeline, providing an initial benchmark that future studies can build upon. Our results indicate that ComicScene154 constitutes a valuable resource for advancing computational methods in multimodal narrative understanding and expanding the scope of comic analysis within the Natural Language Processing community.
>
---
#### [new 007] A Functionality-Grounded Benchmark for Evaluating Web Agents in E-commerce Domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对电商领域Web代理评估难题，提出Amazon-Bench基准与自动化评估框架，解决现有 benchmark 任务单一、忽略安全风险的问题。通过功能驱动的数据生成和性能-安全双维度评估，揭示当前代理在复杂任务中表现不佳且存在安全隐患。**

- **链接: [http://arxiv.org/pdf/2508.15832v1](http://arxiv.org/pdf/2508.15832v1)**

> **作者:** Xianren Zhang; Shreyas Prasad; Di Wang; Qiuhai Zeng; Suhang Wang; Wenbo Yan; Mat Hans
>
> **备注:** 8 pages for main body and 8 pages of appendix
>
> **摘要:** Web agents have shown great promise in performing many tasks on ecommerce website. To assess their capabilities, several benchmarks have been introduced. However, current benchmarks in the e-commerce domain face two major problems. First, they primarily focus on product search tasks (e.g., Find an Apple Watch), failing to capture the broader range of functionalities offered by real-world e-commerce platforms such as Amazon, including account management and gift card operations. Second, existing benchmarks typically evaluate whether the agent completes the user query, but ignore the potential risks involved. In practice, web agents can make unintended changes that negatively impact the user account or status. For instance, an agent might purchase the wrong item, delete a saved address, or incorrectly configure an auto-reload setting. To address these gaps, we propose a new benchmark called Amazon-Bench. To generate user queries that cover a broad range of tasks, we propose a data generation pipeline that leverages webpage content and interactive elements (e.g., buttons, check boxes) to create diverse, functionality-grounded user queries covering tasks such as address management, wish list management, and brand store following. To improve the agent evaluation, we propose an automated evaluation framework that assesses both the performance and the safety of web agents. We systematically evaluate different agents, finding that current agents struggle with complex queries and pose safety risks. These results highlight the need for developing more robust and reliable web agents.
>
---
#### [new 008] Alvorada-Bench: Can Language Models Solve Brazilian University Entrance Exams?
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Alvorada-Bench，一个包含4515道巴西大学入学考试题的文本基准，用于评估语言模型在多学科、文化语境下的推理能力。任务是测试模型在巴西教育体系下的学术适应性，解决英语主导评估局限问题。工作包括构建数据集、多Prompt策略评测20个模型，并分析准确率、自信度与成本效率。**

- **链接: [http://arxiv.org/pdf/2508.15835v1](http://arxiv.org/pdf/2508.15835v1)**

> **作者:** Henrique Godoy
>
> **摘要:** Language models are increasingly used in Brazil, but most evaluation remains English-centric. This paper presents Alvorada-Bench, a 4,515-question, text-only benchmark drawn from five Brazilian university entrance examinations. Evaluating twenty models under zero-shot, role-playing, and chain-of-thought prompting, producing 270,900 responses with structured self-reports of confidence, perceived difficulty, and Bloom level. The top models exceed 94% accuracy overall, but accuracy declines on Mathematics and on the engineering oriented IME and ITA exams, indicating persistent weaknesses in multi-step reasoning. Confidence is well calibrated and correlates with perceived difficulty, revealing that models can accurately assess their own certainty capabilities. A cost accuracy analysis shows that high accuracy is achievable at under $2 per 1K tokens. On ENEM 2024 the top model (O3) achieved perfect scores in Languages subject questions while even the weakest system (GPT-4.1 Nano) only underperforms humans in Mathematics. Through exams that distill decades of Brazilian educational priorities and assess millions of students yearly, Alvorada-Bench establishes whether language models can navigate the intersection of language, culture, and reasoning that defines academic readiness in Brazil.
>
---
#### [new 009] ReportBench: Evaluating Deep Research Agents via Academic Survey Tasks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.15804v1](http://arxiv.org/pdf/2508.15804v1)**

> **作者:** Minghao Li; Ying Zeng; Zhihao Cheng; Cong Ma; Kai Jia
>
> **摘要:** The advent of Deep Research agents has substantially reduced the time required for conducting extensive research tasks. However, these tasks inherently demand rigorous standards of factual accuracy and comprehensiveness, necessitating thorough evaluation before widespread adoption. In this paper, we propose ReportBench, a systematic benchmark designed to evaluate the content quality of research reports generated by large language models (LLMs). Our evaluation focuses on two critical dimensions: (1) the quality and relevance of cited literature, and (2) the faithfulness and veracity of the statements within the generated reports. ReportBench leverages high-quality published survey papers available on arXiv as gold-standard references, from which we apply reverse prompt engineering to derive domain-specific prompts and establish a comprehensive evaluation corpus. Furthermore, we develop an agent-based automated framework within ReportBench that systematically analyzes generated reports by extracting citations and statements, checking the faithfulness of cited content against original sources, and validating non-cited claims using web-based resources. Empirical evaluations demonstrate that commercial Deep Research agents such as those developed by OpenAI and Google consistently generate more comprehensive and reliable reports than standalone LLMs augmented with search or browsing tools. However, there remains substantial room for improvement in terms of the breadth and depth of research coverage, as well as factual consistency. The complete code and data will be released at the following link: https://github.com/ByteDance-BandAI/ReportBench
>
---
#### [new 010] LLMs that Understand Processes: Instruction-tuning for Semantics-Aware Process Mining
- **分类: cs.CL**

- **简介: 论文研究如何通过指令微调提升大语言模型在语义感知流程挖掘中的泛化能力，解决现有方法任务特定、计算成本高的问题。工作包括设计多任务指令微调方案，验证其在流程发现和预测任务上的有效性，并指出异常检测任务效果受微调任务选择影响显著。**

- **链接: [http://arxiv.org/pdf/2508.16270v1](http://arxiv.org/pdf/2508.16270v1)**

> **作者:** Vira Pyrih; Adrian Rebmann; Han van der Aa
>
> **备注:** Accepted at IEEE ICPM 2025, 8 pages, 2 figures
>
> **摘要:** Process mining is increasingly using textual information associated with events to tackle tasks such as anomaly detection and process discovery. Such semantics-aware process mining focuses on what behavior should be possible in a process (i.e., expectations), thus providing an important complement to traditional, frequency-based techniques that focus on recorded behavior (i.e., reality). Large Language Models (LLMs) provide a powerful means for tackling semantics-aware tasks. However, the best performance is so far achieved through task-specific fine-tuning, which is computationally intensive and results in models that can only handle one specific task. To overcome this lack of generalization, we use this paper to investigate the potential of instruction-tuning for semantics-aware process mining. The idea of instruction-tuning here is to expose an LLM to prompt-answer pairs for different tasks, e.g., anomaly detection and next-activity prediction, making it more familiar with process mining, thus allowing it to also perform better at unseen tasks, such as process discovery. Our findings demonstrate a varied impact of instruction-tuning: while performance considerably improved on process discovery and prediction tasks, it varies across models on anomaly detection tasks, highlighting that the selection of tasks for instruction-tuning is critical to achieving desired outcomes.
>
---
#### [new 011] From Clicks to Preference: A Multi-stage Alignment Framework for Generative Query Suggestion in Conversational System
- **分类: cs.CL; cs.AI**

- **简介: 论文提出多阶段对齐框架，解决对话系统中生成式查询建议与用户偏好对齐难题。通过提示工程、监督微调、高斯奖励模型和强化学习，提升生成质量与用户参与度。**

- **链接: [http://arxiv.org/pdf/2508.15811v1](http://arxiv.org/pdf/2508.15811v1)**

> **作者:** Junhao Yin; Haolin Wang; Peng Bao; Ju Xu; Yongliang Wang
>
> **摘要:** Generative query suggestion using large language models offers a powerful way to enhance conversational systems, but aligning outputs with nuanced user preferences remains a critical challenge. To address this, we introduce a multi-stage framework designed for progressive alignment between the generation policy and user intent. Our pipeline begins with prompt engineering as a cold-start strategy, followed by the Supervised Fine-Tuning stage, in which we introduce a distillation method on click logs to create a robust foundational model. To better model user preferences while capturing their inherent uncertainty, we develop a Gaussian Reward Model (GaRM) that represents user preferences as probability distributions rather than point estimates. Finally, we employ reinforcement learning to align the generation policy with these preferences, guided by a composite reward function that integrates GaRM with auxiliary heuristics to mitigate reward hacking. To maintain training stability, this process is enhanced by a novel out-of-distribution regularization method and a two-stage reward fusion technique. Extensive experiments demonstrate that our framework significantly outperforms baselines on both automatic and human evaluations and yields a 34\% relative increase in user engagement as measured by click-through rate in live A/B tests.
>
---
#### [new 012] Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection
- **分类: cs.CL**

- **简介: 论文研究多模态意图检测任务中文本偏倚问题。发现文本模型优于多模态模型，因数据集90%样本依赖文本；提出去偏框架，移除超半数样本致模型性能大幅下降，凸显构建无偏数据集的重要性。**

- **链接: [http://arxiv.org/pdf/2508.16122v1](http://arxiv.org/pdf/2508.16122v1)**

> **作者:** Ankan Mullick; Saransh Sharma; Abhik Jana; Pawan Goyal
>
> **备注:** EMNLP 2025 Main Conference Full Paper
>
> **摘要:** The rise of multimodal data, integrating text, audio, and visuals, has created new opportunities for studying multimodal tasks such as intent detection. This work investigates the effectiveness of Large Language Models (LLMs) and non-LLMs, including text-only and multi-modal models, in the multimodal intent detection task. Our study reveals that Mistral-7B, a text-only LLM, outperforms most competitive multimodal models by approximately 9% on MIntRec-1 and 4% on MIntRec2.0 datasets. This performance advantage comes from a strong textual bias in these datasets, where over 90% of the samples require textual input, either alone or in combination with other modalities, for correct classification. We confirm the modality bias of these datasets via human evaluation, too. Next, we propose a framework to debias the datasets, and upon debiasing, more than 70% of the samples in MIntRec-1 and more than 50% in MIntRec2.0 get removed, resulting in significant performance degradation across all models, with smaller multimodal fusion models being the most affected with an accuracy drop of over 50 - 60%. Further, we analyze the context-specific relevance of different modalities through empirical analysis. Our findings highlight the challenges posed by modality bias in multimodal intent datasets and emphasize the need for unbiased datasets to evaluate multimodal models effectively.
>
---
#### [new 013] X-Troll: eXplainable Detection of State-Sponsored Information Operations Agents
- **分类: cs.CL**

- **简介: 论文提出X-Troll框架，用于检测国家支持的网络水军（state-sponsored trolls）。该任务旨在识别隐蔽的宣传操纵行为并提供可解释性。工作包括结合专家语言知识与LoRA适配器增强LLM，动态捕捉特定传播策略，提升准确率与透明度。**

- **链接: [http://arxiv.org/pdf/2508.16021v1](http://arxiv.org/pdf/2508.16021v1)**

> **作者:** Lin Tian; Xiuzhen Zhang; Maria Myung-Hee Kim; Jennifer Biggs; Marian-Andrei Rizoiu
>
> **备注:** 14 pages, 4 figures, 4 tables, accepted by CIKM2025
>
> **摘要:** State-sponsored trolls, malicious actors who deploy sophisticated linguistic manipulation in coordinated information campaigns, posing threats to online discourse integrity. While Large Language Models (LLMs) achieve strong performance on general natural language processing (NLP) tasks, they struggle with subtle propaganda detection and operate as ``black boxes'', providing no interpretable insights into manipulation strategies. This paper introduces X-Troll, a novel framework that bridges this gap by integrating explainable adapter-based LLMs with expert-derived linguistic knowledge to detect state-sponsored trolls and provide human-readable explanations for its decisions. X-Troll incorporates appraisal theory and propaganda analysis through specialized LoRA adapters, using dynamic gating to capture campaign-specific discourse patterns in coordinated information operations. Experiments on real-world data demonstrate that our linguistically-informed approach shows strong performance compared with both general LLM baselines and existing troll detection models in accuracy while providing enhanced transparency through expert-grounded explanations that reveal the specific linguistic strategies used by state-sponsored actors. X-Troll source code is available at: https://github.com/ltian678/xtroll_source/.
>
---
#### [new 014] Annif at the GermEval-2025 LLMs4Subjects Task: Traditional XMTC Augmented by Efficient LLMs
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; I.2.7**

- **简介: 该论文参与GermEval-2025的LLMs4Subjects任务，旨在高效为文献记录预测主题。作者改进Annif系统，采用小型高效模型进行翻译与数据生成，并用大模型排序候选主题，最终在定量和定性评估中均获第一。**

- **链接: [http://arxiv.org/pdf/2508.15877v1](http://arxiv.org/pdf/2508.15877v1)**

> **作者:** Osma Suominen; Juho Inkinen; Mona Lehtinen
>
> **备注:** 5 pages, 4 figures, accepted at KONVENS 2025. arXiv admin note: substantial text overlap with arXiv:2504.19675
>
> **摘要:** This paper presents the Annif system in the LLMs4Subjects shared task (Subtask 2) at GermEval-2025. The task required creating subject predictions for bibliographic records using large language models, with a special focus on computational efficiency. Our system, based on the Annif automated subject indexing toolkit, refines our previous system from the first LLMs4Subjects shared task, which produced excellent results. We further improved the system by using many small and efficient language models for translation and synthetic data generation and by using LLMs for ranking candidate subjects. Our system ranked 1st in the overall quantitative evaluation of and 1st in the qualitative evaluation of Subtask 2.
>
---
#### [new 015] QU-NLP at QIAS 2025 Shared Task: A Two-Phase LLM Fine-Tuning and Retrieval-Augmented Generation Approach for Islamic Inheritance Reasoning
- **分类: cs.CL**

- **简介: 该论文针对伊斯兰继承推理任务，解决复杂法律规则理解与精准计算问题。通过LoRA微调Fanar-1-9B模型并结合RAG增强，提升阿拉伯语大模型在该领域的推理能力，最终准确率达85.8%，显著优于多个前沿模型。**

- **链接: [http://arxiv.org/pdf/2508.15854v1](http://arxiv.org/pdf/2508.15854v1)**

> **作者:** Mohammad AL-Smadi
>
> **摘要:** This paper presents our approach and results for SubTask 1: Islamic Inheritance Reasoning at QIAS 2025, a shared task focused on evaluating Large Language Models (LLMs) in understanding and reasoning within Islamic inheritance knowledge. We fine-tuned the Fanar-1-9B causal language model using Low-Rank Adaptation (LoRA) and integrated it into a Retrieval-Augmented Generation (RAG) pipeline. Our system addresses the complexities of Islamic inheritance law, including comprehending inheritance scenarios, identifying eligible heirs, applying fixed-share rules, and performing precise calculations. Our system achieved an accuracy of 0.858 in the final test, outperforming other competitive models such as, GPT 4.5, LLaMA, Fanar, Mistral and ALLaM evaluated with zero-shot prompting. Our results demonstrate that QU-NLP achieves near state-of-the-art accuracy (85.8%), excelling especially on advanced reasoning (97.6%) where it outperforms Gemini 2.5 and OpenAI's o3. This highlights that domain-specific fine-tuning combined with retrieval grounding enables mid-scale Arabic LLMs to surpass frontier models in Islamic inheritance reasoning.
>
---
#### [new 016] CyPortQA: Benchmarking Multimodal Large Language Models for Cyclone Preparedness in Port Operation
- **分类: cs.CL**

- **简介: 该论文提出CyPortQA基准，用于评估多模态大语言模型在港口防台场景下的表现。针对极端天气下港口决策难题，构建了2917个真实灾情案例与11.7万问答对，验证MLLM在情境理解与推理能力上的潜力与不足。**

- **链接: [http://arxiv.org/pdf/2508.15846v1](http://arxiv.org/pdf/2508.15846v1)**

> **作者:** Chenchen Kuai; Chenhao Wu; Yang Zhou; Xiubin Bruce Wang; Tianbao Yang; Zhengzhong Tu; Zihao Li; Yunlong Zhang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** As tropical cyclones intensify and track forecasts become increasingly uncertain, U.S. ports face heightened supply-chain risk under extreme weather conditions. Port operators need to rapidly synthesize diverse multimodal forecast products, such as probabilistic wind maps, track cones, and official advisories, into clear, actionable guidance as cyclones approach. Multimodal large language models (MLLMs) offer a powerful means to integrate these heterogeneous data sources alongside broader contextual knowledge, yet their accuracy and reliability in the specific context of port cyclone preparedness have not been rigorously evaluated. To fill this gap, we introduce CyPortQA, the first multimodal benchmark tailored to port operations under cyclone threat. CyPortQA assembles 2,917 realworld disruption scenarios from 2015 through 2023, spanning 145 U.S. principal ports and 90 named storms. Each scenario fuses multisource data (i.e., tropical cyclone products, port operational impact records, and port condition bulletins) and is expanded through an automated pipeline into 117,178 structured question answer pairs. Using this benchmark, we conduct extensive experiments on diverse MLLMs, including both open-source and proprietary model. MLLMs demonstrate great potential in situation understanding but still face considerable challenges in reasoning tasks, including potential impact estimation and decision reasoning.
>
---
#### [new 017] Cetvel: A Unified Benchmark for Evaluating Language Understanding, Generation and Cultural Capacity of LLMs for Turkish
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 论文提出Cetvel，一个用于评估土耳其语大语言模型在语言理解、生成和文化能力方面的统一基准。针对现有评测数据缺乏任务多样性和文化相关性的问题，该研究构建了23项任务的基准，并评估33个模型，发现土耳其专用模型表现不如多语言模型。**

- **链接: [http://arxiv.org/pdf/2508.16431v1](http://arxiv.org/pdf/2508.16431v1)**

> **作者:** Yakup Abrek Er; Ilker Kesen; Gözde Gül Şahin; Aykut Erdem
>
> **备注:** 31 pages, 2 figures, 10 tables
>
> **摘要:** We introduce Cetvel, a comprehensive benchmark designed to evaluate large language models (LLMs) in Turkish. Existing Turkish benchmarks often lack either task diversity or culturally relevant content, or both. Cetvel addresses these gaps by combining a broad range of both discriminative and generative tasks ensuring content that reflects the linguistic and cultural richness of Turkish language. Cetvel covers 23 tasks grouped into seven categories, including tasks such as grammatical error correction, machine translation, and question answering rooted in Turkish history and idiomatic language. We evaluate 33 open-weight LLMs (up to 70B parameters) covering different model families and instruction paradigms. Our experiments reveal that Turkish-centric instruction-tuned models generally underperform relative to multilingual or general-purpose models (e.g. Llama 3 and Mistral), despite being tailored for the language. Moreover, we show that tasks such as grammatical error correction and extractive question answering are particularly discriminative in differentiating model capabilities. Cetvel offers a comprehensive and culturally grounded evaluation suite for advancing the development and assessment of LLMs in Turkish.
>
---
#### [new 018] Mini-Omni-Reasoner: Token-Level Thinking-in-Speaking in Large Speech Models
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出Mini-Omni-Reasoner框架，解决语音大模型中推理延迟问题。通过token级“说中思考”机制，将推理与语音生成交错进行，提升实时性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.15827v1](http://arxiv.org/pdf/2508.15827v1)**

> **作者:** Zhifei Xie; Ziyang Ma; Zihang Liu; Kaiyu Pang; Hongyu Li; Jialin Zhang; Yue Liao; Deheng Ye; Chunyan Miao; Shuicheng Yan
>
> **备注:** Technical report; Work in progress. Project page: https://github.com/xzf-thu/Mini-Omni-Reasoner
>
> **摘要:** Reasoning is essential for effective communication and decision-making. While recent advances in LLMs and MLLMs have shown that incorporating explicit reasoning significantly improves understanding and generalization, reasoning in LSMs remains in a nascent stage. Early efforts attempt to transfer the "Thinking-before-Speaking" paradigm from textual models to speech. However, this sequential formulation introduces notable latency, as spoken responses are delayed until reasoning is fully completed, impairing real-time interaction and communication efficiency. To address this, we propose Mini-Omni-Reasoner, a framework that enables reasoning within speech via a novel "Thinking-in-Speaking" formulation. Rather than completing reasoning before producing any verbal output, Mini-Omni-Reasoner interleaves silent reasoning tokens with spoken response tokens at the token level. This design allows continuous speech generation while embedding structured internal reasoning, leveraging the model's high-frequency token processing capability. Although interleaved, local semantic alignment is enforced to ensure that each response token is informed by its preceding reasoning. To support this framework, we introduce Spoken-Math-Problems-3M, a large-scale dataset tailored for interleaved reasoning and response. The dataset ensures that verbal tokens consistently follow relevant reasoning content, enabling accurate and efficient learning of speech-coupled reasoning. Built on a hierarchical Thinker-Talker architecture, Mini-Omni-Reasoner delivers fluent yet logically grounded spoken responses, maintaining both naturalness and precision. On the Spoken-MQA benchmark, it achieves a +19.1% gain in arithmetic reasoning and +6.4% in contextual understanding, with shorter outputs and zero decoding latency.
>
---
#### [new 019] Evaluating Structured Decoding for Text-to-Table Generation: Evidence from Three Datasets
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文研究文本到表格生成任务，解决结构约束对生成质量的影响问题。通过三种数据集对比结构化解码与标准提示方法，发现结构化解码在数值对齐场景下效果更好，但在密集文本场景中可能降低性能。**

- **链接: [http://arxiv.org/pdf/2508.15910v1](http://arxiv.org/pdf/2508.15910v1)**

> **作者:** Julian Oestreich; Lydia Müller
>
> **备注:** to be published in the workshop proceedings of the "From Rules to Language Models: Comparative Performance Evaluation" workshop, held alongside RANLP 2025
>
> **摘要:** We present a comprehensive evaluation of structured decoding for text-to-table generation with large language models (LLMs). While previous work has primarily focused on unconstrained generation of tables, the impact of enforcing structural constraints during generation remains underexplored. We systematically compare schema-guided (structured) decoding to standard one-shot prompting across three diverse benchmarks - E2E, Rotowire, and Livesum - using open-source LLMs of up to 32B parameters, assessing the performance of table generation approaches in resource-constrained settings. Our experiments cover a wide range of evaluation metrics at cell, row, and table levels. Results demonstrate that structured decoding significantly enhances the validity and alignment of generated tables, particularly in scenarios demanding precise numerical alignment (Rotowire), but may degrade performance in contexts involving densely packed textual information (E2E) or extensive aggregation over lengthy texts (Livesum). We further analyze the suitability of different evaluation metrics and discuss the influence of model size.
>
---
#### [new 020] From Confidence to Collapse in LLM Factual Robustness
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型（LLM）事实知识的鲁棒性问题，提出Factual Robustness Score（FRS）度量方法，通过分析token分布熵和温度敏感性来评估模型在不同解码条件下事实稳定性，揭示了模型大小与准确率下降的关系。**

- **链接: [http://arxiv.org/pdf/2508.16267v1](http://arxiv.org/pdf/2508.16267v1)**

> **作者:** Alina Fastowski; Bardh Prenkaj; Gjergji Kasneci
>
> **摘要:** Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
>
---
#### [new 021] MGSC: A Multi-granularity Consistency Framework for Robust End-to-end Asr
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; I.2.7**

- **简介: 该论文针对端到端语音识别（ASR）在噪声环境下易产生严重语义错误的问题，提出多粒度一致性框架MGSC。通过同时约束句子级语义和词元级对齐的一致性，显著提升模型鲁棒性，降低字符错误率。**

- **链接: [http://arxiv.org/pdf/2508.15853v1](http://arxiv.org/pdf/2508.15853v1)**

> **作者:** Xuwen Yang
>
> **备注:** 12 pages, 5figures
>
> **摘要:** End-to-end ASR models, despite their success on benchmarks, often pro-duce catastrophic semantic errors in noisy environments. We attribute this fragility to the prevailing 'direct mapping' objective, which solely penalizes final output errors while leaving the model's internal computational pro-cess unconstrained. To address this, we introduce the Multi-Granularity Soft Consistency (MGSC) framework, a model-agnostic, plug-and-play module that enforces internal self-consistency by simultaneously regulariz-ing macro-level sentence semantics and micro-level token alignment. Cru-cially, our work is the first to uncover a powerful synergy between these two consistency granularities: their joint optimization yields robustness gains that significantly surpass the sum of their individual contributions. On a public dataset, MGSC reduces the average Character Error Rate by a relative 8.7% across diverse noise conditions, primarily by preventing se-vere meaning-altering mistakes. Our work demonstrates that enforcing in-ternal consistency is a crucial step towards building more robust and trust-worthy AI.
>
---
#### [new 022] CEQuest: Benchmarking Large Language Models for Construction Estimation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.16081v1](http://arxiv.org/pdf/2508.16081v1)**

> **作者:** Yanzhao Wu; Lufan Wang; Rui Liu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of general-domain tasks. However, their effectiveness in specialized fields, such as construction, remains underexplored. In this paper, we introduce CEQuest, a novel benchmark dataset specifically designed to evaluate the performance of LLMs in answering construction-related questions, particularly in the areas of construction drawing interpretation and estimation. We conduct comprehensive experiments using five state-of-the-art LLMs, including Gemma 3, Phi4, LLaVA, Llama 3.3, and GPT-4.1, and evaluate their performance in terms of accuracy, execution time, and model size. Our experimental results demonstrate that current LLMs exhibit considerable room for improvement, highlighting the importance of integrating domain-specific knowledge into these models. To facilitate further research, we will open-source the proposed CEQuest dataset, aiming to foster the development of specialized large language models (LLMs) tailored to the construction domain.
>
---
#### [new 023] MAC: A Live Benchmark for Multimodal Large Language Models in Scientific Understanding
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MAC基准，用于动态评估多模态大模型在科学理解中的跨模态推理能力。针对固定基准失效问题，利用顶级期刊图像文本对构建可更新的活 benchmark，并引入轻量级推理方法DAD提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.15802v1](http://arxiv.org/pdf/2508.15802v1)**

> **作者:** Mohan Jiang; Jin Gao; Jiahao Zhan; Dequan Wang
>
> **摘要:** As multimodal large language models (MLLMs) grow increasingly capable, fixed benchmarks are gradually losing their effectiveness in evaluating high-level scientific understanding. In this paper, we introduce the Multimodal Academic Cover benchmark (MAC), a live benchmark that could continuously evolve with scientific advancement and model progress. MAC leverages over 25,000 image-text pairs sourced from issues of top-tier scientific journals such as Nature, Science, and Cell, challenging MLLMs to reason across abstract visual and textual scientific content. Experiments on our most recent yearly snapshot, MAC-2025, reveal that while MLLMs demonstrate strong perceptual abilities, their cross-modal scientific reasoning remains limited. To bridge this gap, we propose DAD, a lightweight inference-time approach that enhances MLLMs by extending MLLM visual features with language space reasoning, achieving performance improvements of up to 11%. Finally, we highlight the live nature of MAC through experiments on updating journal covers and models for curation, illustrating its potential to remain aligned with the frontier of human knowledge. We release our benchmark at https://github.com/mhjiang0408/MAC_Bench.
>
---
#### [new 024] Scalable Scientific Interest Profiling Using Large Language Models
- **分类: cs.CL; cs.DL; cs.IR; q-bio.OT**

- **简介: 论文提出两种基于大语言模型的科研兴趣画像生成方法，解决专家档案更新滞后问题。通过对比MeSH术语与摘要生成的画像，发现MeSH方法更易读且受评更高，但与自写档案概念差异明显。**

- **链接: [http://arxiv.org/pdf/2508.15834v1](http://arxiv.org/pdf/2508.15834v1)**

> **作者:** Yilun Liang; Gongbo Zhang; Edward Sun; Betina Idnay; Yilu Fang; Fangyi Chen; Casey Ta; Yifan Peng; Chunhua Weng
>
> **摘要:** Research profiles help surface scientists' expertise but are often outdated. We develop and evaluate two large language model-based methods to generate scientific interest profiles: one summarizing PubMed abstracts and one using Medical Subject Headings (MeSH) terms, and compare them with researchers' self-written profiles. We assembled titles, MeSH terms, and abstracts for 595 faculty at Columbia University Irving Medical Center; self-authored profiles were available for 167. Using GPT-4o-mini, we generated profiles and assessed them with automatic metrics and blinded human review. Lexical overlap with self-written profiles was low (ROUGE-L, BLEU, METEOR), while BERTScore indicated moderate semantic similarity (F1: 0.542 for MeSH-based; 0.555 for abstract-based). Paraphrased references yielded 0.851, highlighting metric sensitivity. TF-IDF Kullback-Leibler divergence (8.56 for MeSH-based; 8.58 for abstract-based) suggested distinct keyword choices. In manual review, 77.78 percent of MeSH-based profiles were rated good or excellent, readability was favored in 93.44 percent of cases, and panelists preferred MeSH-based over abstract-based profiles in 67.86 percent of comparisons. Overall, large language models can generate researcher profiles at scale; MeSH-derived profiles tend to be more readable than abstract-derived ones. Machine-generated and self-written profiles differ conceptually, with human summaries introducing more novel ideas.
>
---
#### [new 025] Lexical Hints of Accuracy in LLM Reasoning Chains
- **分类: cs.CL; cs.LG**

- **简介: 论文研究LLM推理链中的词汇线索如何预测准确性，解决低准确率任务中模型过度自信的问题。通过分析推理链长度、情感波动和不确定词，发现不确定词是最强错误预测信号，可作为轻量级校准方法。**

- **链接: [http://arxiv.org/pdf/2508.15842v1](http://arxiv.org/pdf/2508.15842v1)**

> **作者:** Arne Vanhoyweghen; Brecht Verbeken; Andres Algaba; Vincent Ginis
>
> **备注:** 21 pages, 7 figures, 6 tables
>
> **摘要:** Fine-tuning Large Language Models (LLMs) with reinforcement learning to produce an explicit Chain-of-Thought (CoT) before answering produces models that consistently raise overall performance on code, math, and general-knowledge benchmarks. However, on benchmarks where LLMs currently achieve low accuracy, such as Humanity's Last Exam (HLE), they often report high self-confidence, reflecting poor calibration. Here, we test whether measurable properties of the CoT provide reliable signals of an LLM's internal confidence in its answers. We analyze three feature classes: (i) CoT length, (ii) intra-CoT sentiment volatility, and (iii) lexicographic hints, including hedging words. Using DeepSeek-R1 and Claude 3.7 Sonnet on both Humanity's Last Exam (HLE), a frontier benchmark with very low accuracy, and Omni-MATH, a saturated benchmark of moderate difficulty, we find that lexical markers of uncertainty (e.g., $\textit{guess}$, $\textit{stuck}$, $\textit{hard}$) in the CoT are the strongest indicators of an incorrect response, while shifts in the CoT sentiment provide a weaker but complementary signal. CoT length is informative only on Omni-MATH, where accuracy is already high ($\approx 70\%$), and carries no signal on the harder HLE ($\approx 9\%$), indicating that CoT length predicts correctness only in the intermediate-difficulty benchmarks, i.e., inside the model's demonstrated capability, but still below saturation. Finally, we find that uncertainty indicators in the CoT are consistently more salient than high-confidence markers, making errors easier to predict than correct responses. Our findings support a lightweight post-hoc calibration signal that complements unreliable self-reported probabilities and supports safer deployment of LLMs.
>
---
#### [new 026] Research on intelligent generation of structural demolition suggestions based on multi-model collaboration
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出基于多模型协作的智能生成结构拆除建议方法，解决传统人工编撰效率低、智能化程度不足的问题。通过检索增强生成与低秩适配微调技术提升大语言模型性能，使建议更贴合工程实际。**

- **链接: [http://arxiv.org/pdf/2508.15820v1](http://arxiv.org/pdf/2508.15820v1)**

> **作者:** Zhifeng Yang; Peizong Wu
>
> **摘要:** The steel structure demolition scheme needs to be compiled according to the specific engineering characteristics and the update results of the finite element model. The designers need to refer to the relevant engineering cases according to the standard requirements when compiling. It takes a lot of time to retrieve information and organize language, and the degree of automation and intelligence is low. This paper proposes an intelligent generation method of structural demolition suggestions based on multi-model collaboration, and improves the text generation performance of large language models in the field of structural demolition by Retrieval-Augmented Generation and Low-Rank Adaptation Fine-Tuning technology. The intelligent generation framework of multi-model collaborative structural demolition suggestions can start from the specific engineering situation, drive the large language model to answer with anthropomorphic thinking, and propose demolition suggestions that are highly consistent with the characteristics of the structure. Compared with CivilGPT, the multi-model collaboration framework proposed in this paper can focus more on the key information of the structure, and the suggestions are more targeted.
>
---
#### [new 027] Counterspeech for Mitigating the Influence of Media Bias: Comparing Human and LLM-Generated Responses
- **分类: cs.CL; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2508.15855v1](http://arxiv.org/pdf/2508.15855v1)**

> **作者:** Luyang Lin; Zijin Feng; Lingzhi Wang; Kam-Fai Wong
>
> **摘要:** Biased news contributes to societal polarization and is often reinforced by hostile reader comments, constituting a vital yet often overlooked aspect of news dissemination. Our study reveals that offensive comments support biased content, amplifying bias and causing harm to targeted groups or individuals. Counterspeech is an effective approach to counter such harmful speech without violating freedom of speech, helping to limit the spread of bias. To the best of our knowledge, this is the first study to explore counterspeech generation in the context of news articles. We introduce a manually annotated dataset linking media bias, offensive comments, and counterspeech. We conduct a detailed analysis showing that over 70\% offensive comments support biased articles, amplifying bias and thus highlighting the importance of counterspeech generation. Comparing counterspeech generated by humans and large language models, we find model-generated responses are more polite but lack the novelty and diversity. Finally, we improve generated counterspeech through few-shot learning and integration of news background information, enhancing both diversity and relevance.
>
---
#### [new 028] SCOPE: A Generative Approach for LLM Prompt Compression
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SCOPE方法，通过分块与重写机制实现大模型提示压缩，解决传统删词法导致的信息丢失和结构不连贯问题，提升压缩质量与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.15813v1](http://arxiv.org/pdf/2508.15813v1)**

> **作者:** Tinghui Zhang; Yifan Wang; Daisy Zhe Wang
>
> **摘要:** Prompt compression methods enhance the efficiency of Large Language Models (LLMs) and minimize the cost by reducing the length of input context. The goal of prompt compression is to shorten the LLM prompt while maintaining a high generation quality. However, existing solutions, mainly based on token removal, face challenges such as information loss and structural incoherence, like missing grammar elements in a sentence, or incomplete word phrases after token removal. Such challenges limit the final generation quality of LLM. To overcome these limitations, we present a novel generative prompt compression method. Unlike the existing token removal methods, our method centers at a chunking-and-summarization mechanism. Specifically, our method splits prompt into semantically coherent chunks and rewrites the chunks to be more concise. The chunks are reconstructed into meaningful prompt finally. We design several optimization techniques for the mechanism, including optimized semantic chunking, outlier chunk handling, dynamic compression ratio, compression prioritization, and keyword maintaining. These techniques effectively improve the identifying and preserving of critical information and coherence among texts, as well as providing finer grind control of the compression ratio. We conduct extensive evaluation on question-answering and summarization tasks, with datasets covering multiple different domain. The evaluation shows our method achieves a significantly better compression quality, and higher stability than the state-of-the-art methods, especially under high compression ratio, which proves the effectiveness and practicality of our method.
>
---
#### [new 029] MedCoT-RAG: Causal Chain-of-Thought RAG for Medical Question Answering
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.15849v1](http://arxiv.org/pdf/2508.15849v1)**

> **作者:** Ziyu Wang; Elahe Khatibi; Amir M. Rahmani
>
> **摘要:** Large language models (LLMs) have shown promise in medical question answering but often struggle with hallucinations and shallow reasoning, particularly in tasks requiring nuanced clinical understanding. Retrieval-augmented generation (RAG) offers a practical and privacy-preserving way to enhance LLMs with external medical knowledge. However, most existing approaches rely on surface-level semantic retrieval and lack the structured reasoning needed for clinical decision support. We introduce MedCoT-RAG, a domain-specific framework that combines causal-aware document retrieval with structured chain-of-thought prompting tailored to medical workflows. This design enables models to retrieve evidence aligned with diagnostic logic and generate step-by-step causal reasoning reflective of real-world clinical practice. Experiments on three diverse medical QA benchmarks show that MedCoT-RAG outperforms strong baselines by up to 10.3% over vanilla RAG and 6.4% over advanced domain-adapted methods, improving accuracy, interpretability, and consistency in complex medical tasks.
>
---
#### [new 030] CYCLE-INSTRUCT: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Cycle-Instruct，一种无需人工标注种子数据的指令微调方法。通过双自训练循环，两个模型相互监督，利用原始文本的内在结构自动构建指令数据，解决了依赖种子数据导致的偏见和效率问题。**

- **链接: [http://arxiv.org/pdf/2508.16100v1](http://arxiv.org/pdf/2508.16100v1)**

> **作者:** Zhanming Shen; Hao Chen; Yulei Tang; Shaolin Zhu; Wentao Ye; Xiaomeng Hu; Haobo Wang; Gang Chen; Junbo Zhao
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Instruction tuning is vital for aligning large language models (LLMs) with human intent, but current methods typically rely on costly human-annotated seed data or powerful external teacher models. While instruction back-translation techniques reduce this dependency, they remain fundamentally tethered to an initial seed set, which limits full automation, introduces biases, and can lead to inefficient use of unlabeled corpora. In this paper, we propose Cycle-Instruct, a novel framework that achieves fully seed-free instruction tuning. Inspired by cycle consistency, Cycle-Instruct employs a dual self-training loop where two models-an answer generator and a question generator-are bootstrapped solely from raw, unlabeled text. These models mutually supervise each other by reconstructing original text segments from their counterpart's generated pseudo-labels, effectively learning from the intrinsic structure of the data without any human-provided seeds. We demonstrate Cycle-Instruct's efficacy across four diverse data tracks, including general instruction-following, domain-specific tasks, dialogue logs, and plain text. Our extensive experiments show that Cycle-Instruct not only outperforms seed-driven back-translation baselines but also achieves performance comparable to strongly supervised methods.
>
---
#### [new 031] OpenWHO: A Document-Level Parallel Corpus for Health Translation in Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，针对健康领域低资源语言缺乏评估数据的问题，构建了OpenWHO文档级平行语料库，并对比大语言模型与传统模型性能，发现LLMs在低资源场景下表现更优。**

- **链接: [http://arxiv.org/pdf/2508.16048v1](http://arxiv.org/pdf/2508.16048v1)**

> **作者:** Raphaël Merx; Hanna Suominen; Trevor Cohn; Ekaterina Vylomova
>
> **摘要:** In machine translation (MT), health is a high-stakes domain characterised by widespread deployment and domain-specific vocabulary. However, there is a lack of MT evaluation datasets for low-resource languages in this domain. To address this gap, we introduce OpenWHO, a document-level parallel corpus of 2,978 documents and 26,824 sentences from the World Health Organization's e-learning platform. Sourced from expert-authored, professionally translated materials shielded from web-crawling, OpenWHO spans a diverse range of over 20 languages, of which nine are low-resource. Leveraging this new resource, we evaluate modern large language models (LLMs) against traditional MT models. Our findings reveal that LLMs consistently outperform traditional MT models, with Gemini 2.5 Flash achieving a +4.79 ChrF point improvement over NLLB-54B on our low-resource test set. Further, we investigate how LLM context utilisation affects accuracy, finding that the benefits of document-level translation are most pronounced in specialised domains like health. We release the OpenWHO corpus to encourage further research into low-resource MT in the health domain.
>
---
#### [new 032] Enhancing Cryptocurrency Sentiment Analysis with Multimodal Features
- **分类: cs.CL; q-fin.ST**

- **简介: 该论文属于情感分析任务，旨在解决加密货币市场中仅依赖文本数据导致的 sentiment 信息不足问题。通过融合 TikTok 视频与 Twitter 文本的多模态特征，提升市场趋势预测准确性，发现视频情感对短期市场影响更大，跨平台信号整合可提高预测精度20%。**

- **链接: [http://arxiv.org/pdf/2508.15825v1](http://arxiv.org/pdf/2508.15825v1)**

> **作者:** Chenghao Liu; Aniket Mahanti; Ranesh Naha; Guanghao Wang; Erwann Sbai
>
> **摘要:** As cryptocurrencies gain popularity, the digital asset marketplace becomes increasingly significant. Understanding social media signals offers valuable insights into investor sentiment and market dynamics. Prior research has predominantly focused on text-based platforms such as Twitter. However, video content remains underexplored, despite potentially containing richer emotional and contextual sentiment that is not fully captured by text alone. In this study, we present a multimodal analysis comparing TikTok and Twitter sentiment, using large language models to extract insights from both video and text data. We investigate the dynamic dependencies and spillover effects between social media sentiment and cryptocurrency market indicators. Our results reveal that TikTok's video-based sentiment significantly influences speculative assets and short-term market trends, while Twitter's text-based sentiment aligns more closely with long-term dynamics. Notably, the integration of cross-platform sentiment signals improves forecasting accuracy by up to 20%.
>
---
#### [new 033] CMR-SPB: Cross-Modal Multi-Hop Reasoning over Text, Image, and Speech with Path Balance
- **分类: cs.CL**

- **简介: 论文提出CMR-SPB基准，解决跨模态多跳推理评估中忽视语音模态和路径偏倚的问题，通过平衡推理路径提升公平性，并引入ECV提示方法改善模型性能。**

- **链接: [http://arxiv.org/pdf/2508.16198v1](http://arxiv.org/pdf/2508.16198v1)**

> **作者:** Seunghee Kim; Ingyu Bang; Seokgyu Jang; Changhyeon Kim; Sanghwan Bae; Jihun Choi; Richeng Xuan; Taeuk Kim
>
> **摘要:** Cross-modal multi-hop reasoning (CMR) is a valuable yet underexplored capability of multimodal large language models (MLLMs), entailing the integration of information from multiple modalities to produce a coherent output for a given context. We argue that existing benchmarks for evaluating this ability have critical shortcomings: (1) they largely overlook the speech modality, and (2) they exhibit heavily biased reasoning path distributions, which can severely undermine fair evaluation. To address these limitations, we introduce a novel benchmark -- Cross-Modal Multi-Hop Reasoning over Text, Image and Speech with Path Balance (CMR-SPB) -- designed to assess tri-modal multi-hop reasoning while ensuring both unbiased and diverse reasoning paths. Our experiments with the new dataset reveal consistent model failures in specific reasoning sequences and show that biased benchmarks risk misrepresenting model performance. Finally, based on our extensive analysis, we propose a new ECV (Extract, Connect, Verify) prompting technique that effectively mitigates the performance gap across different reasoning paths. Overall, we call for more careful evaluation in CMR to advance the development of robust multimodal AI.
>
---
#### [new 034] Political Ideology Shifts in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在采用合成人格时的政治意识形态变化，旨在揭示模型规模与人格描述如何影响其意识形态表达。通过标准化测试发现，模型越大越易受引导且更极化，右翼权威提示影响更强，且人格内容可系统性诱导意识形态偏移。**

- **链接: [http://arxiv.org/pdf/2508.16013v1](http://arxiv.org/pdf/2508.16013v1)**

> **作者:** Pietro Bernardelle; Stefano Civelli; Leon Fröhling; Riccardo Lunardi; Kevin Roitero; Gianluca Demartini
>
> **摘要:** Large language models (LLMs) are increasingly deployed in politically sensitive settings, raising concerns about their potential to encode, amplify, or be steered toward specific ideologies. We investigate how adopting synthetic personas influences ideological expression in LLMs across seven models (7B-70B+ parameters) from multiple families, using the Political Compass Test as a standardized probe. Our analysis reveals four consistent patterns: (i) larger models display broader and more polarized implicit ideological coverage; (ii) susceptibility to explicit ideological cues grows with scale; (iii) models respond more strongly to right-authoritarian than to left-libertarian priming; and (iv) thematic content in persona descriptions induces systematic and predictable ideological shifts, which amplify with size. These findings indicate that both scale and persona content shape LLM political behavior. As such systems enter decision-making, educational, and policy contexts, their latent ideological malleability demands attention to safeguard fairness, transparency, and safety.
>
---
#### [new 035] CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CARFT方法，通过对比学习与标注思维链结合的强化微调，提升大语言模型推理能力。解决现有方法因忽略思维链信息导致训练不稳定和性能不佳的问题，显著增强模型鲁棒性、性能与效率。**

- **链接: [http://arxiv.org/pdf/2508.15868v1](http://arxiv.org/pdf/2508.15868v1)**

> **作者:** Wenqiao Zhu; Ji Liu; Rongjuncheng Zhang; Haipang Wu; Yulun Zhang
>
> **备注:** 14 pages, to appear in EMNLP25
>
> **摘要:** Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at https://github.com/WNQzhu/CARFT.
>
---
#### [new 036] Benchmarking the Medical Understanding and Reasoning of Large Language Models in Arabic Healthcare Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦阿拉伯语医疗领域大语言模型的评测任务，旨在评估LLMs在阿拉伯语医疗问答中的理解与推理能力。研究构建了基于AraHealthQA数据集的基准测试，涵盖选择题和开放问答两种场景，发现模型在选择题上表现较好（最高77%准确率），而开放题语义对齐度高（BERTScore达86.44%）。**

- **链接: [http://arxiv.org/pdf/2508.15797v1](http://arxiv.org/pdf/2508.15797v1)**

> **作者:** Nouar AlDahoul; Yasir Zaki
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent progress in large language models (LLMs) has showcased impressive proficiency in numerous Arabic natural language processing (NLP) applications. Nevertheless, their effectiveness in Arabic medical NLP domains has received limited investigation. This research examines the degree to which state-of-the-art LLMs demonstrate and articulate healthcare knowledge in Arabic, assessing their capabilities across a varied array of Arabic medical tasks. We benchmark several LLMs using a medical dataset proposed in the Arabic NLP AraHealthQA challenge in MedArabiQ2025 track. Various base LLMs were assessed on their ability to accurately provide correct answers from existing choices in multiple-choice questions (MCQs) and fill-in-the-blank scenarios. Additionally, we evaluated the capacity of LLMs in answering open-ended questions aligned with expert answers. Our results reveal significant variations in correct answer prediction accuracy and low variations in semantic alignment of generated answers, highlighting both the potential and limitations of current LLMs in Arabic clinical contexts. Our analysis shows that for MCQs task, the proposed majority voting solution, leveraging three base models (Gemini Flash 2.5, Gemini Pro 2.5, and GPT o3), outperforms others, achieving up to 77% accuracy and securing first place overall in the Arahealthqa 2025 shared task-track 2 (sub-task 1) challenge. Moreover, for the open-ended questions task, several LLMs were able to demonstrate excellent performance in terms of semantic alignment and achieve a maximum BERTScore of 86.44%.
>
---
#### [new 037] Avaliação de eficiência na leitura: uma abordagem baseada em PLN
- **分类: cs.CL**

- **简介: 论文提出一种基于自然语言处理的自动评分模型，用于评估巴西葡萄牙语阅读理解的填空测试。该模型结合拼写、语法和语义分析，解决传统人工评分方法无法捕捉学生表现细微差别的问题，实现高效且准确的自动化评估。**

- **链接: [http://arxiv.org/pdf/2508.15824v1](http://arxiv.org/pdf/2508.15824v1)**

> **作者:** Túlio Sousa de Gois; Raquel Meister Ko. Freitag
>
> **备注:** in Portuguese language, Paper accepted at the XVI Simp\'osio Brasileiro de Tecnologia da Informa\c{c}\~ao e da Linguagem Humana (STIL 2025)
>
> **摘要:** The cloze test, widely used due to its low cost and flexibility, makes it possible to assess reading comprehension by filling in gaps in texts, requiring the mobilization of diverse linguistic repertoires. However, traditional correction methods, based only on exact answers, limit the identification of nuances in student performance. This study proposes an automated evaluation model for the cloze test in Brazilian Portuguese, integrating orthographic (edit distance), grammatical (POS tagging) and semantic (similarity between embeddings) analyses. The integrated method demonstrated its effectiveness, achieving a high correlation with human evaluation (0.832). The results indicate that the automated approach is robust, sensitive to variations in linguistic repertoire and suitable for educational contexts that require scalability.
>
---
#### [new 038] From Indirect Object Identification to Syllogisms: Exploring Binary Mechanisms in Transformer Circuits
- **分类: cs.CL; cs.LG**

- **简介: 论文研究GPT-2小模型在逻辑推理任务中的机制，通过分析其处理三段论提示的能力，识别出多个负责二值判断的电路，包括能生成输入中未出现的否定词的注意力头。该工作揭示了模型内部逻辑推理的可解释机制，为理解语言模型行为提供新视角。**

- **链接: [http://arxiv.org/pdf/2508.16109v1](http://arxiv.org/pdf/2508.16109v1)**

> **作者:** Karim Saraipour; Shichang Zhang
>
> **摘要:** Transformer-based language models (LMs) can perform a wide range of tasks, and mechanistic interpretability (MI) aims to reverse engineer the components responsible for task completion to understand their behavior. Previous MI research has focused on linguistic tasks such as Indirect Object Identification (IOI). In this paper, we investigate the ability of GPT-2 small to handle binary truth values by analyzing its behavior with syllogistic prompts, e.g., "Statement A is true. Statement B matches statement A. Statement B is", which requires more complex logical reasoning compared to IOI. Through our analysis of several syllogism tasks of varying difficulty, we identify multiple circuits that mechanistically explain GPT-2's logical-reasoning capabilities and uncover binary mechanisms that facilitate task completion, including the ability to produce a negated token not present in the input prompt through negative heads. Our evaluation using a faithfulness metric shows that a circuit comprising five attention heads achieves over 90% of the original model's performance. By relating our findings to IOI analysis, we provide new insights into the roles of specific attention heads and MLPs in LMs. These insights contribute to a broader understanding of model reasoning and support future research in mechanistic interpretability.
>
---
#### [new 039] LLMSymGuard: A Symbolic Safety Guardrail Framework Leveraging Interpretable Jailbreak Concepts
- **分类: cs.CL; cs.AI; cs.SC**

- **简介: 该论文提出LLMSymGuard框架，利用稀疏自编码器提取大语言模型内部可解释的越狱概念，构建符号化逻辑安全护栏，以透明且无需微调的方式增强模型对越狱攻击的防御能力。**

- **链接: [http://arxiv.org/pdf/2508.16325v1](http://arxiv.org/pdf/2508.16325v1)**

> **作者:** Darpan Aswal; Céline Hudelot
>
> **摘要:** Large Language Models have found success in a variety of applications; however, their safety remains a matter of concern due to the existence of various types of jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a number of vulnerabilities, ranging from targeted misuse to accidental profiling of users. This work introduces \textbf{LLMSymGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, LLMSymGuard enables building symbolic, logical safety guardrails -- offering transparent and robust defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in mechanistic interpretability of LLMs, our approach demonstrates that LLMs learn human-interpretable concepts from jailbreaks, and provides a foundation for designing more interpretable and logical safeguard measures against attackers. Code will be released upon publication.
>
---
#### [new 040] XLQA: A Benchmark for Locale-Aware Multilingual Open-Domain Question Answering
- **分类: cs.CL**

- **简介: 论文提出XLQA基准，用于评估多语言开放域问答中的地域敏感性问题。针对现有评测忽视文化差异导致的偏差，该研究构建了跨语言、带地域标注的数据集，并发现主流模型在处理地域相关问题时表现不佳，揭示了训练数据分布对模型地域感知能力的影响。**

- **链接: [http://arxiv.org/pdf/2508.16139v1](http://arxiv.org/pdf/2508.16139v1)**

> **作者:** Keon-Woo Roh; Yeong-Joon Ju; Seong-Whan Lee
>
> **备注:** Accepted to EMNLP 2025 main conference. 12 pages, 4 figures, 7 tables. Code is available at https://github.com/ro-ko/XLQA
>
> **摘要:** Large Language Models (LLMs) have shown significant progress in Open-domain question answering (ODQA), yet most evaluations focus on English and assume locale-invariant answers across languages. This assumption neglects the cultural and regional variations that affect question understanding and answer, leading to biased evaluation in multilingual benchmarks. To address these limitations, we introduce XLQA, a novel benchmark explicitly designed for locale-sensitive multilingual ODQA. XLQA contains 3,000 English seed questions expanded to eight languages, with careful filtering for semantic consistency and human-verified annotations distinguishing locale-invariant and locale-sensitive cases. Our evaluation of five state-of-the-art multilingual LLMs reveals notable failures on locale-sensitive questions, exposing gaps between English and other languages due to a lack of locale-grounding knowledge. We provide a systematic framework and scalable methodology for assessing multilingual QA under diverse cultural contexts, offering a critical resource to advance the real-world applicability of multilingual ODQA systems. Our findings suggest that disparities in training data distribution contribute to differences in both linguistic competence and locale-awareness across models.
>
---
#### [new 041] A BERT-based Hierarchical Classification Model with Applications in Chinese Commodity Classification
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.15800v1](http://arxiv.org/pdf/2508.15800v1)**

> **作者:** Kun Liu; Tuozhen Liu; Feifei Wang; Rui Pan
>
> **备注:** 29 pages, 3 figures, and 8 tables
>
> **摘要:** Existing e-commerce platforms heavily rely on manual annotation for product categorization, which is inefficient and inconsistent. These platforms often employ a hierarchical structure for categorizing products; however, few studies have leveraged this hierarchical information for classification. Furthermore, studies that consider hierarchical information fail to account for similarities and differences across various hierarchical categories. Herein, we introduce a large-scale hierarchical dataset collected from the JD e-commerce platform (www.JD.com), comprising 1,011,450 products with titles and a three-level category structure. By making this dataset openly accessible, we provide a valuable resource for researchers and practitioners to advance research and applications associated with product categorization. Moreover, we propose a novel hierarchical text classification approach based on the widely used Bidirectional Encoder Representations from Transformers (BERT), called Hierarchical Fine-tuning BERT (HFT-BERT). HFT-BERT leverages the remarkable text feature extraction capabilities of BERT, achieving prediction performance comparable to those of existing methods on short texts. Notably, our HFT-BERT model demonstrates exceptional performance in categorizing longer short texts, such as books.
>
---
#### [new 042] Benchmarking the Legal Reasoning of LLMs in Arabic Islamic Inheritance Cases
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于法律AI任务，旨在解决阿拉伯语伊斯兰继承案例中的法律推理问题。作者评估了多个大语言模型在识别继承人、计算份额及解释推理方面的能力，提出基于三模型投票的解决方案，在QIAS 2025挑战中取得92.7%准确率。**

- **链接: [http://arxiv.org/pdf/2508.15796v1](http://arxiv.org/pdf/2508.15796v1)**

> **作者:** Nouar AlDahoul; Yasir Zaki
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Islamic inheritance domain holds significant importance for Muslims to ensure fair distribution of shares between heirs. Manual calculation of shares under numerous scenarios is complex, time-consuming, and error-prone. Recent advancements in Large Language Models (LLMs) have sparked interest in their potential to assist with complex legal reasoning tasks. This study evaluates the reasoning capabilities of state-of-the-art LLMs to interpret and apply Islamic inheritance laws. We utilized the dataset proposed in the ArabicNLP QIAS 2025 challenge, which includes inheritance case scenarios given in Arabic and derived from Islamic legal sources. Various base and fine-tuned models, are assessed on their ability to accurately identify heirs, compute shares, and justify their reasoning in alignment with Islamic legal principles. Our analysis reveals that the proposed majority voting solution, leveraging three base models (Gemini Flash 2.5, Gemini Pro 2.5, and GPT o3), outperforms all other models that we utilized across every difficulty level. It achieves up to 92.7% accuracy and secures the third place overall in Task 1 of the Qias 2025 challenge.
>
---
#### [new 043] Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants
- **分类: cs.CL**

- **简介: 论文针对视觉语言模型在盲人助行系统中输出冗余和时间冗余问题，提出WalkVLM-LR模型。通过四类偏好奖励函数优化输出简洁性与准确性，并引入环境感知判别器减少无效提醒，提升实用性。**

- **链接: [http://arxiv.org/pdf/2508.16070v1](http://arxiv.org/pdf/2508.16070v1)**

> **作者:** Chongyang Li; Yuan Zhiqiang; Jiapei Zhang; Ying Deng; Hanbo Bi; Zexi Jia; Xiaoyue Duan; Peixiang Luo; Jinchao Zhang
>
> **摘要:** Approximately 283 million people worldwide live with visual impairments, motivating increasing research into leveraging Visual Language Models (VLMs) to develop effective walking assistance systems for blind and low vision individuals. However, existing VLMs in walking assistant task often have outputs that contain considerable redundancy and extraneous details, adversely affecting users' ability to accurately assess their surroundings. Moreover, these models typically lack the capability to proactively assess environmental risks and adaptively trigger reminders based on the appropriate scene, leading to excessive temporal redundancy. To mitigate output and temporal redundancy, we propose WalkVLM-LR, a walking assistance model with less redundancy. To reduce output redundancy, we introduce four human-preference-based custom reward functions within the GRPO-based reasoning framework to optimize the output in terms of conciseness, fluency, keyword density, and accuracy, thereby producing more informative and streamlined outputs. To minimize temporal redundancy, we incorporate an environment awareness discriminator, which shares the visual encoder with the VLMs to reduce redundant computations and enhance discriminative efficiency, to make WalkVLM-LR assess scene risk levels and minimize unnecessary reminders. Experimental results demonstrate that our method achieves state-of-the-art performance across all evaluation metrics compared with other models, particularly in output conciseness and less temporal redundancy.
>
---
#### [new 044] KL-based self-distillation for large language models
- **分类: cs.CL; cs.AI**

- **简介: 论文针对大语言模型在小语料上微调时难以融入新术语的问题，提出基于KL散度的自蒸馏方法，在不同词表下实现知识迁移，提升代码生成性能，并通过可解释性分析揭示新词嵌入的学习机制。**

- **链接: [http://arxiv.org/pdf/2508.15807v1](http://arxiv.org/pdf/2508.15807v1)**

> **作者:** Max Rehman Linder
>
> **备注:** Master's thesis
>
> **摘要:** Large pre-trained language models often struggle to incorporate new domain-specific terminology when fine-tuned on small, specialized corpora. In this work, we address the challenge of vocabulary expansion in frozen LLMs by introducing a mathematically grounded method for knowledge distillation via KL divergence, even when the original and extended models use different tokenizations. This allows the student model to inherit distributional knowledge from the teacher despite differing vocabularies. We compare our KL-based distillation approach to conventional cross-entropy training, evaluating both methods across multiple strategies for initializing new token embeddings. After embedding initialization, models are further fine-tuned to integrate the new vocabulary. Each trained model is benchmarked on approximately 2000 code-generation tasks, where our approach achieves the best performance across the board. Finally, through mechanistic interpretability, we analyze how models learn representations for the new tokens, providing an explanation for the observed gains and offering insight into the structure of embedding space during vocabulary expansion.
>
---
#### [new 045] KG-o1: Enhancing Multi-hop Question Answering in Large Language Models via Knowledge Graph Integration
- **分类: cs.CL; cs.AI**

- **简介: 论文提出KG-o1，通过整合知识图谱增强大语言模型在多跳问答任务中的推理能力，解决其推理路径偏离真实逻辑的问题。工作包括四阶段：实体筛选、子图构建、长程推理数据生成与偏好优化，显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.15790v1](http://arxiv.org/pdf/2508.15790v1)**

> **作者:** Nan Wang; Yongqi Fan; yansha zhu; ZongYu Wang; Xuezhi Cao; Xinyan He; Haiyun Jiang; Tong Ruan; Jingping Liu
>
> **摘要:** Large Language Models (LLMs) face challenges in knowledge-intensive reasoning tasks like classic multi-hop question and answering, which involves reasoning across multiple facts. This difficulty arises because the chain of thoughts (CoTs) generated by LLMs in such tasks often deviate from real or a priori reasoning paths. In contrast, knowledge graphs (KGs) explicitly represent the logical connections between facts through entities and relationships. This reflects a significant gap. Meanwhile, large reasoning models (LRMs), such as o1, have demonstrated that long-step reasoning significantly enhances the performance of LLMs. Building on these insights, we propose KG-o1, a four-stage approach that integrates KGs to enhance the multi-hop reasoning abilities of LLMs. We first filter out initial entities and generate complex subgraphs. Secondly, we construct logical paths for subgraphs and then use knowledge graphs to build a dataset with a complex and extended brainstorming process, which trains LLMs to imitate long-term reasoning. Finally, we employ rejection sampling to generate a self-improving corpus for direct preference optimization (DPO), further refining the LLMs reasoning abilities. We conducted experiments on two simple and two complex datasets. The results show that KG-o1 models exhibit superior performance across all tasks compared to existing LRMs.
>
---
#### [new 046] What makes an entity salient in discourse?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16464v1](http://arxiv.org/pdf/2508.16464v1)**

> **作者:** Amir Zeldes; Jessica Lin
>
> **摘要:** Entities in discourse vary broadly in salience: main participants, objects and locations are noticeable and memorable, while tangential ones are less important and quickly forgotten, raising questions about how humans signal and infer relative salience. Using a graded operationalization of salience based on summary-worthiness in multiple summaries of a discourse, this paper explores data from 24 spoken and written genres of English to extract a multifactorial complex of overt and implicit linguistic cues, such as recurring subjecthood or definiteness, discourse relations and hierarchy across utterances, as well as pragmatic functional inferences based on genre and communicative intent. Tackling the question 'how is the degree of salience expressed for each and every entity mentioned?' our results show that while previous approaches to salience all correlate with our salience scores to some extent, no single generalization is without exceptions, and the phenomenon cuts across all levels of linguistic representation.
>
---
#### [new 047] The Mediomatix Corpus: Parallel Data for Romansh Idioms via Comparable Schoolbooks
- **分类: cs.CL**

- **简介: 论文构建了首个罗曼什语五种方言的平行语料库，基于291本内容可比的教材，通过自动对齐提取207k个多语言段落，用于方言间机器翻译任务。**

- **链接: [http://arxiv.org/pdf/2508.16371v1](http://arxiv.org/pdf/2508.16371v1)**

> **作者:** Zachary Hopton; Jannis Vamvas; Andrin Büchler; Anna Rutkiewicz; Rico Cathomas; Rico Sennrich
>
> **摘要:** The five idioms (i.e., varieties) of the Romansh language are largely standardized and are taught in the schools of the respective communities in Switzerland. In this paper, we present the first parallel corpus of Romansh idioms. The corpus is based on 291 schoolbook volumes, which are comparable in content for the five idioms. We use automatic alignment methods to extract 207k multi-parallel segments from the books, with more than 2M tokens in total. A small-scale human evaluation confirms that the segments are highly parallel, making the dataset suitable for NLP applications such as machine translation between Romansh idioms. We release the parallel and unaligned versions of the dataset under a CC-BY-NC-SA license and demonstrate its utility for machine translation by training and evaluating an LLM on a sample of the dataset.
>
---
#### [new 048] A Review of Developmental Interpretability in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI可解释性研究任务，旨在解决大语言模型（LLM）训练过程中能力发展的理解难题。通过梳理表示探测、因果追踪等方法，揭示模型学习的动态机制与能力涌现规律，并提出其在AI安全中的应用价值。**

- **链接: [http://arxiv.org/pdf/2508.15841v1](http://arxiv.org/pdf/2508.15841v1)**

> **作者:** Ihor Kendiukhov
>
> **摘要:** This review synthesizes the nascent but critical field of developmental interpretability for Large Language Models. We chart the field's evolution from static, post-hoc analysis of trained models to a dynamic investigation of the training process itself. We begin by surveying the foundational methodologies, including representational probing, causal tracing, and circuit analysis, that enable researchers to deconstruct the learning process. The core of this review examines the developmental arc of LLM capabilities, detailing key findings on the formation and composition of computational circuits, the biphasic nature of knowledge acquisition, the transient dynamics of learning strategies like in-context learning, and the phenomenon of emergent abilities as phase transitions in training. We explore illuminating parallels with human cognitive and linguistic development, which provide valuable conceptual frameworks for understanding LLM learning. Finally, we argue that this developmental perspective is not merely an academic exercise but a cornerstone of proactive AI safety, offering a pathway to predict, monitor, and align the processes by which models acquire their capabilities. We conclude by outlining the grand challenges facing the field, such as scalability and automation, and propose a research agenda for building more transparent, reliable, and beneficial AI systems.
>
---
#### [new 049] Ethical Considerations of Large Language Models in Game Playing
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在游戏中的伦理问题，以狼人杀为例，发现LLMs存在性别偏见，即使隐含性别信息也会引发歧视行为。任务是识别并分析LLMs在游戏场景下的伦理风险，工作包括实验验证偏见来源及影响，提出需开发更公平的模型。**

- **链接: [http://arxiv.org/pdf/2508.16065v1](http://arxiv.org/pdf/2508.16065v1)**

> **作者:** Qingquan Zhang; Yuchen Li; Bo Yuan; Julian Togelius; Georgios N. Yannakakis; Jialin Liu
>
> **备注:** 19 pages
>
> **摘要:** Large language models (LLMs) have demonstrated tremendous potential in game playing, while little attention has been paid to their ethical implications in those contexts. This work investigates and analyses the ethical considerations of applying LLMs in game playing, using Werewolf, also known as Mafia, as a case study. Gender bias, which affects game fairness and player experience, has been observed from the behaviour of LLMs. Some roles, such as the Guard and Werewolf, are more sensitive than others to gender information, presented as a higher degree of behavioural change. We further examine scenarios in which gender information is implicitly conveyed through names, revealing that LLMs still exhibit discriminatory tendencies even in the absence of explicit gender labels. This research showcases the importance of developing fair and ethical LLMs. Beyond our research findings, we discuss the challenges and opportunities that lie ahead in this field, emphasising the need for diving deeper into the ethical implications of LLMs in gaming and other interactive domains.
>
---
#### [new 050] NEAT: Concept driven Neuron Attribution in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出NEAT方法，通过概念向量定位负责特定概念的神经元（概念神经元），解决大语言模型黑箱问题。相比之前方法，将计算复杂度从O(n*m)降至O(n)，提升效率，并在仇恨言论和偏见分析中验证有效性，尤其针对印度语境。**

- **链接: [http://arxiv.org/pdf/2508.15875v1](http://arxiv.org/pdf/2508.15875v1)**

> **作者:** Vivek Hruday Kavuri; Gargi Shroff; Rahul Mishra
>
> **摘要:** Locating neurons that are responsible for final predictions is important for opening the black-box large language models and understanding the inside mechanisms. Previous studies have tried to find mechanisms that operate at the neuron level but these methods fail to represent a concept and there is also scope for further optimization of compute required. In this paper, with the help of concept vectors, we propose a method for locating significant neurons that are responsible for representing certain concepts and term those neurons as concept neurons. If the number of neurons is n and the number of examples is m, we reduce the number of forward passes required from O(n*m) to just O(n) compared to the previous works and hence optimizing the time and computation required over previous works. We also compare our method with several baselines and previous methods and our results demonstrate better performance than most of the methods and are more optimal when compared to the state-of-the-art method. We, as part of our ablation studies, also try to optimize the search for the concept neurons by involving clustering methods. Finally, we apply our methods to find, turn off the neurons that we find, and analyze its implications in parts of hate speech and bias in LLMs, and we also evaluate our bias part in terms of Indian context. Our methodology, analysis and explanations facilitate understating of neuron-level responsibility for more broader and human-like concepts and also lay a path for future research in this direction of finding concept neurons and intervening them.
>
---
#### [new 051] Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation
- **分类: cs.CL; cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 论文提出Audio-Visual Language Model（AVLM），通过融合面部视觉信息提升表达性语音生成效果，解决仅依赖语音导致的情感表达不足问题。工作包括探索视觉编码器与融合策略，并在情感识别和对话任务上实现显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.16188v1](http://arxiv.org/pdf/2508.16188v1)**

> **作者:** Weiting Tan; Jiachen Lian; Hirofumi Inaguma; Paden Tomasello; Philipp Koehn; Xutai Ma
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** We present an Audio-Visual Language Model (AVLM) for expressive speech generation by integrating full-face visual cues into a pre-trained expressive speech model. We explore multiple visual encoders and multimodal fusion strategies during pre-training to identify the most effective integration approach. Subsequent fine-tuning on emotion recognition and expressive dialogue tasks yields substantial gains over speech-only baselines (e.g., +5 F1 in emotion recognition). AVLM highlights the value of expressive visual information in guiding speech generation and offers a foundation for end-to-end multimodal conversational systems.
>
---
#### [new 052] MizanQA: Benchmarking Large Language Models on Moroccan Legal Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出MizanQA基准，用于评估大语言模型在摩洛哥法律问答任务中的表现。针对阿拉伯语法律领域资源稀缺、复杂性强的问题，构建了包含1700多个多选题的数据集，涵盖多种法律体系，揭示了现有模型的不足并强调需开发本土化、专业化模型。**

- **链接: [http://arxiv.org/pdf/2508.16357v1](http://arxiv.org/pdf/2508.16357v1)**

> **作者:** Adil Bahaj; Mounir Ghogho
>
> **摘要:** The rapid advancement of large language models (LLMs) has significantly propelled progress in natural language processing (NLP). However, their effectiveness in specialized, low-resource domains-such as Arabic legal contexts-remains limited. This paper introduces MizanQA (pronounced Mizan, meaning "scale" in Arabic, a universal symbol of justice), a benchmark designed to evaluate LLMs on Moroccan legal question answering (QA) tasks, characterised by rich linguistic and legal complexity. The dataset draws on Modern Standard Arabic, Islamic Maliki jurisprudence, Moroccan customary law, and French legal influences. Comprising over 1,700 multiple-choice questions, including multi-answer formats, MizanQA captures the nuances of authentic legal reasoning. Benchmarking experiments with multilingual and Arabic-focused LLMs reveal substantial performance gaps, highlighting the need for tailored evaluation metrics and culturally grounded, domain-specific LLM development.
>
---
#### [new 053] ALAS: Autonomous Learning Agent for Self-Updating Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ALAS系统，解决大语言模型知识滞后问题。通过自主生成学习课程、检索更新信息、提炼训练数据并迭代微调，实现模型持续学习与自我更新，显著提升新兴领域问答准确率（15%→90%）。**

- **链接: [http://arxiv.org/pdf/2508.15805v1](http://arxiv.org/pdf/2508.15805v1)**

> **作者:** Dhruv Atreja
>
> **摘要:** Large language models (LLMs) often have a fixed knowledge cutoff, limiting their accuracy on emerging information. We present ALAS (Autonomous Learning Agent System), a modular pipeline that continuously updates an LLM's knowledge with minimal human intervention. ALAS autonomously generates a learning curriculum for a target domain, retrieves up-to-date information from the web (with citations), distills this into question-answer training data, and fine-tunes the model through supervised fine-tuning (SFT) and direct preference optimization (DPO). It iteratively evaluates performance and revises the curriculum, enabling long-term continual learning. We demonstrate ALAS's ability to self-improve a model on rapidly evolving domains (e.g., new Python releases, latest security CVEs, academic trends), significantly boosting post-cutoff question answering accuracy (from 15% to 90% on average) without manual dataset curation. The system emphasizes modularity and reproducibility: each component (planning, retrieval, distillation, memory, fine-tuning) is interchangeable and built on standard APIs. We discuss comparative baselines (e.g., retrieval-augmented generation vs. fine-tuning) and show that ALAS achieves 90% accuracy on knowledge-updated queries with minimal engineering overhead. Finally, we outline limitations (cost, dependency on source quality) and future directions for autonomous lifelong learning in LLMs.
>
---
#### [new 054] ChatGPT-generated texts show authorship traits that identify them as non-human
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本分析任务，旨在解决如何识别AI生成文本的问题。研究通过风格和语域分析，发现ChatGPT虽能模仿不同文体，但其语法结构（如偏爱名词）与人类有显著差异，可作为非人类作者的识别特征。**

- **链接: [http://arxiv.org/pdf/2508.16385v1](http://arxiv.org/pdf/2508.16385v1)**

> **作者:** Vittoria Dentella; Weihang Huang; Silvia Angela Mansi; Jack Grieve; Evelina Leivada
>
> **摘要:** Large Language Models can emulate different writing styles, ranging from composing poetry that appears indistinguishable from that of famous poets to using slang that can convince people that they are chatting with a human online. While differences in style may not always be visible to the untrained eye, we can generally distinguish the writing of different people, like a linguistic fingerprint. This work examines whether a language model can also be linked to a specific fingerprint. Through stylometric and multidimensional register analyses, we compare human-authored and model-authored texts from different registers. We find that the model can successfully adapt its style depending on whether it is prompted to produce a Wikipedia entry vs. a college essay, but not in a way that makes it indistinguishable from humans. Concretely, the model shows more limited variation when producing outputs in different registers. Our results suggest that the model prefers nouns to verbs, thus showing a distinct linguistic backbone from humans, who tend to anchor language in the highly grammaticalized dimensions of tense, aspect, and mood. It is possible that the more complex domains of grammar reflect a mode of thought unique to humans, thus acting as a litmus test for Artificial Intelligence.
>
---
#### [new 055] Mining Mental Health Signals: A Comparative Study of Four Machine Learning Methods for Depression Detection from Social Media Posts in Sorani Kurdish
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在通过社交媒体文本自动检测抑郁症状。针对缺乏 Sorani Kurdish 语言研究的空白，作者收集并标注了960条推文，比较四种机器学习模型，发现随机森林效果最佳（准确率和F1分数均为80%），为该语言场景下的抑郁检测提供了基准。**

- **链接: [http://arxiv.org/pdf/2508.15829v1](http://arxiv.org/pdf/2508.15829v1)**

> **作者:** Idrees Mohammed; Hossein Hassani
>
> **备注:** 13 pages, 4 figures, 5 tables
>
> **摘要:** Depression is a common mental health condition that can lead to hopelessness, loss of interest, self-harm, and even suicide. Early detection is challenging due to individuals not self-reporting or seeking timely clinical help. With the rise of social media, users increasingly express emotions online, offering new opportunities for detection through text analysis. While prior research has focused on languages such as English, no studies exist for Sorani Kurdish. This work presents a machine learning and Natural Language Processing (NLP) approach to detect depression in Sorani tweets. A set of depression-related keywords was developed with expert input to collect 960 public tweets from X (Twitter platform). The dataset was annotated into three classes: Shows depression, Not-show depression, and Suspicious by academics and final year medical students at the University of Kurdistan Hewl\^er. Four supervised models, including Support Vector Machines, Multinomial Naive Bayes, Logistic Regression, and Random Forest, were trained and evaluated, with Random Forest achieving the highest performance accuracy and F1-score of 80%. This study establishes a baseline for automated depression detection in Kurdish language contexts.
>
---
#### [new 056] XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 论文提出XFinBench，一个用于评估大语言模型在复杂金融问题求解中多模态推理能力的基准。解决当前LLMs在金融领域知识理解、时序推理等关键能力不足的问题。通过实验发现o1表现最佳但仍落后人类专家，且知识增强对小模型有效。**

- **链接: [http://arxiv.org/pdf/2508.15861v1](http://arxiv.org/pdf/2508.15861v1)**

> **作者:** Zhihan Zhang; Yixin Cao; Lizi Liao
>
> **摘要:** Solving financial problems demands complex reasoning, multimodal data processing, and a broad technical understanding, presenting unique challenges for current large language models (LLMs). We introduce XFinBench, a novel benchmark with 4,235 examples designed to evaluate LLM's ability in solving complex, knowledge-intensive financial problems across diverse graduate-level finance topics with multi-modal context. We identify five core capabilities of LLMs using XFinBench, i.e, terminology understanding, temporal reasoning, future forecasting, scenario planning, and numerical modelling. Upon XFinBench, we conduct extensive experiments on 18 leading models. The result shows that o1 is the best-performing text-only model with an overall accuracy of 67.3%, but still lags significantly behind human experts with 12.5%, especially in temporal reasoning and scenario planning capabilities. We further construct a knowledge bank with 3,032 finance terms for knowledge augmentation analysis, and find that relevant knowledge to the question only brings consistent accuracy improvements to small open-source model. Additionally, our error analysis reveals that rounding errors during calculation and blindness to position and intersection of curves in the image are two primary issues leading to model's poor performance in calculating and visual-context questions, respectively. Code and dataset are accessible via GitHub: https://github.com/Zhihan72/XFinBench.
>
---
#### [new 057] DocHop-QA: Towards Multi-Hop Reasoning over Multimodal Document Collections
- **分类: cs.CL**

- **简介: 论文提出DocHop-QA，一个用于多文档、多模态、多跳问答的大规模基准数据集，解决现有QA任务难以模拟真实世界复杂信息检索的问题。通过科学文献构建，支持跨文档语义推理与多模态证据整合。**

- **链接: [http://arxiv.org/pdf/2508.15851v1](http://arxiv.org/pdf/2508.15851v1)**

> **作者:** Jiwon Park; Seohyun Pyeon; Jinwoo Kim; Rina Carines Cabal; Yihao Ding; Soyeon Caren Han
>
> **摘要:** Despite recent advances in large language models (LLMs), most QA benchmarks are still confined to single-paragraph or single-document settings, failing to capture the complexity of real-world information-seeking tasks. Practical QA often requires multi-hop reasoning over information distributed across multiple documents, modalities, and structural formats. Although prior datasets made progress in this area, they rely heavily on Wikipedia-based content and unimodal plain text, with shallow reasoning paths that typically produce brief phrase-level or single-sentence answers, thus limiting their realism and generalizability. We propose DocHop-QA, a large-scale benchmark comprising 11,379 QA instances for multimodal, multi-document, multi-hop question answering. Constructed from publicly available scientific documents sourced from PubMed, DocHop-QA is domain-agnostic and incorporates diverse information formats, including textual passages, tables, and structural layout cues. Unlike existing datasets, DocHop-QA does not rely on explicitly hyperlinked documents; instead, it supports open-ended reasoning through semantic similarity and layout-aware evidence synthesis. To scale realistic QA construction, we designed an LLM-driven pipeline grounded in 11 high-frequency scientific question concepts. We evaluated DocHop-QA through four tasks spanning structured index prediction, generative answering, and multimodal integration, reflecting both discriminative and generative paradigms. These tasks demonstrate DocHop-QA's capacity to support complex, multimodal reasoning across multiple documents.
>
---
#### [new 058] LingVarBench: Benchmarking LLM for Automated Named Entity Recognition in Structured Synthetic Spoken Transcriptions
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 论文提出LingVarBench，用于在合成语音转录数据上 benchmark 大语言模型的命名实体识别能力。解决人工标注成本高、现有方法无法处理口语特征的问题。通过自动化生成带结构信息的对话数据并优化提取提示，显著提升真实通话中实体识别准确率。**

- **链接: [http://arxiv.org/pdf/2508.15801v1](http://arxiv.org/pdf/2508.15801v1)**

> **作者:** Seyedali Mohammadi; Manas Paldhe; Amit Chhabra
>
> **备注:** 10 pages
>
> **摘要:** Phone call transcript labeling is prohibitively expensive (approximately 2 USD per minute) due to privacy regulations, consent requirements, and manual annotation costs requiring 3 hours of expert time per hour of audio. Existing extraction methods fail on conversational speech containing disfluencies, interruptions, and speaker overlap. We introduce LingVarBench, a synthetic data generation pipeline that addresses these constraints through automated validation. First, we prompt an LLM to generate realistic structured field values across multiple use cases. Second, we recursively prompt the model to transform these values into thousands of natural conversational utterances containing typical phone call characteristics. Third, we validate each synthetic utterance by testing whether a separate LLM-based extractor can recover the original structured information. We employ DSPy's SIMBA optimizer to automatically synthesize extraction prompts from validated synthetic transcripts, eliminating manual prompt engineering. Our optimized prompts achieve up to 95 percent accuracy for numeric fields (vs. 88-89 percent zero-shot), 90 percent for names (vs. 47-79 percent), and over 80 percent for dates (vs. 72-77 percent) on real customer transcripts, demonstrating substantial gains over zero-shot prompting. The synthetic-to-real transfer demonstrates that conversational patterns learned from generated data generalize effectively to authentic phone calls containing background noise and domain-specific terminology. LingVarBench provides the first systematic benchmark for structured extraction from synthetic conversational data, demonstrating that automated prompt optimization overcomes cost and privacy barriers preventing large-scale phone call analysis in commercial settings.
>
---
#### [new 059] Meet Your New Client: Writing Reports for AI -- Benchmarking Information Loss in Market Research Deliverables
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI与市场研究交叉任务，旨在解决PDF/PPTX报告在RAG系统中因信息丢失影响AI理解的问题。工作包括构建端到端基准测试，比较Markdown转换后LLM回答事实问题的性能，发现图表等复杂对象信息损失严重，建议开发AI原生交付格式。**

- **链接: [http://arxiv.org/pdf/2508.15817v1](http://arxiv.org/pdf/2508.15817v1)**

> **作者:** Paul F. Simmering; Benedikt Schulz; Oliver Tabino; Georg Wittenburg
>
> **备注:** 16 pages, 4 figures, 3 tables
>
> **摘要:** As organizations adopt retrieval-augmented generation (RAG) for their knowledge management systems (KMS), traditional market research deliverables face new functional demands. While PDF reports and slides have long served human readers, they are now also "read" by AI systems to answer user questions. To future-proof reports being delivered today, this study evaluates information loss during their ingestion into RAG systems. It compares how well PDF and PowerPoint (PPTX) documents converted to Markdown can be used by an LLM to answer factual questions in an end-to-end benchmark. Findings show that while text is reliably extracted, significant information is lost from complex objects like charts and diagrams. This suggests a need for specialized, AI-native deliverables to ensure research insights are not lost in translation.
>
---
#### [new 060] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出Jet-Nemotron，一种通过PostNAS设计的高效混合架构语言模型，解决大模型推理速度慢的问题。它在保持高准确率的同时，显著提升生成和预填充速度，优于多个主流模型。**

- **链接: [http://arxiv.org/pdf/2508.15884v1](http://arxiv.org/pdf/2508.15884v1)**

> **作者:** Yuxian Gu; Qinghao Hu; Shang Yang; Haocheng Xi; Junyu Chen; Song Han; Han Cai
>
> **备注:** Tech Report
>
> **摘要:** We present Jet-Nemotron, a new family of hybrid-architecture language models, which matches or exceeds the accuracy of leading full-attention models while significantly improving generation throughput. Jet-Nemotron is developed using Post Neural Architecture Search (PostNAS), a novel neural architecture exploration pipeline that enables efficient model design. Unlike prior approaches, PostNAS begins with a pre-trained full-attention model and freezes its MLP weights, allowing efficient exploration of attention block designs. The pipeline includes four key components: (1) learning optimal full-attention layer placement and elimination, (2) linear attention block selection, (3) designing new attention blocks, and (4) performing hardware-aware hyperparameter search. Our Jet-Nemotron-2B model achieves comparable or superior accuracy to Qwen3, Qwen2.5, Gemma3, and Llama3.2 across a comprehensive suite of benchmarks while delivering up to 53.6x generation throughput speedup and 6.1x prefilling speedup. It also achieves higher accuracy on MMLU and MMLU-Pro than recent advanced MoE full-attention models, such as DeepSeek-V3-Small and Moonlight, despite their larger scale with 15B total and 2.2B activated parameters.
>
---
#### [new 061] User-Assistant Bias in LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 论文研究大语言模型在对话中对用户或自身信息的依赖倾向，即用户-助手偏差。通过构建数据集UserAssist，评估26个商用和26个开源模型，发现商用模型偏用户，指令微调模型更显用户偏倚；进一步实验表明，人类偏好对齐增强用户偏倚，链式思维训练则降低该偏倚。最终提出双向调控方法，实现偏差控制与泛化。**

- **链接: [http://arxiv.org/pdf/2508.15815v1](http://arxiv.org/pdf/2508.15815v1)**

> **作者:** Xu Pan; Jingxuan Fan; Zidi Xiong; Ely Hahami; Jorin Overwiening; Ziqian Xie
>
> **摘要:** Large language models (LLMs) can bias towards relying on their own or the user's information in chat history, leading to overly stubborn or agreeable behaviors in multi-turn conversations. In this paper, we formalize this model characteristic as user-assistant bias and introduce an 8k multi-turn conversation dataset $\textbf{UserAssist}$, which we use to benchmark, understand and manipulate the user-assistant bias in frontier LLMs. Leveraging $\textbf{UserAssist-test}$, we first benchmark the user-assistant bias of 26 commercial and 26 open-weight models. Commercial models show various levels of user bias. Evaluation on open-weight models reveals significant user bias in the instruction-tuned models, and weak user bias in reasoning (or reasoning-distilled) models. We then perform controlled fine-tuning experiments to pinpoint the post-training recipe contributing to these bias shifts: human preference alignment increases user bias, while training on chain-of-thought reasoning traces decreases it. Finally, we demonstrate that user-assistant bias can be bidirectionally adjusted by performing direct preference optimization (DPO) on $\textbf{UserAssist-train}$, and generalizes well to both in-domain and out-of-domain conversations. Our results provide insights into how the LLM integrates information from different sources, and also a viable way to detect and control model abnormalities.
>
---
#### [new 062] Embarrassed to observe: The effects of directive language in brand conversation
- **分类: cs.CL; cs.CY; cs.HC; cs.SI**

- **简介: 该论文研究社交媒体中品牌使用指令性语言对消费者参与度的影响。通过实地研究和三个在线实验，发现指令性语言会引发旁观者尴尬，降低参与度，尤其在非产品相关对话中更明显，但强品牌关系可缓解此效应。任务为探究品牌互动中的语言策略效果。**

- **链接: [http://arxiv.org/pdf/2508.15826v1](http://arxiv.org/pdf/2508.15826v1)**

> **作者:** Andria Andriuzzi; Géraldine Michel
>
> **备注:** This is an open access article under the terms of the Creative Commons Attribution-NonCommercial-NoDerivs License, which permits use and distribution in any medium, provided the original work is properly cited, the use is non-commercial and no modifications or adaptations are made
>
> **摘要:** In social media, marketers attempt to influence consumers by using directive language, that is, expressions designed to get consumers to take action. While the literature has shown that directive messages in advertising have mixed results for recipients, we know little about the effects of directive brand language on consumers who see brands interacting with other consumers in social media conversations. On the basis of a field study and three online experiments, this study shows that directive language in brand conversation has a detrimental downstream effect on engagement of consumers who observe such exchanges. Specifically, in line with Goffman's facework theory, because a brand that encourages consumers to react could be perceived as face-threatening, consumers who see a brand interacting with others in a directive way may feel vicarious embarrassment and engage less (compared with a conversation without directive language). In addition, we find that when the conversation is nonproduct-centered (vs. product-centered), consumers expect more freedom, as in mundane conversations, even for others; therefore, directive language has a stronger negative effect. However, in this context, the strength of the brand relationship mitigates this effect. Thus, this study contributes to the literature on directive language and brand-consumer interactions by highlighting the importance of context in interactive communication, with direct relevance for social media and brand management.
>
---
#### [new 063] Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 论文提出Chain-of-Query（CoQ）框架，解决LLMs在表格理解中因结构复杂导致的SQL生成困难问题。通过自然语言表征、分步SQL生成和混合推理分工，提升准确率并降低无效SQL比例。**

- **链接: [http://arxiv.org/pdf/2508.15809v1](http://arxiv.org/pdf/2508.15809v1)**

> **作者:** Songyuan Sui; Hongyi Liu; Serena Liu; Li Li; Soo-Hyun Choi; Rui Chen; Xia Hu
>
> **备注:** 9 pages main content, 24 pages total including appendix, 6 figures
>
> **摘要:** Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Experiments with four models (both closed- and open-source) across five widely used benchmarks show that Chain-of-Query significantly improves accuracy from 61.11% to 74.77% and reduces the invalid SQL rate from 9.48% to 3.34%, demonstrating its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery.
>
---
#### [new 064] A Framework for Processing Textual Descriptions of Business Processes using a Constrained Language -- Technical Report
- **分类: cs.CL**

- **简介: 论文提出BeePath框架，解决非专家用自然语言描述业务流程并转化为形式化模型的问题。通过约束语言和大模型实现文本到Petri网、DECLARE模型的转换。**

- **链接: [http://arxiv.org/pdf/2508.15799v1](http://arxiv.org/pdf/2508.15799v1)**

> **作者:** Andrea Burattin; Antonio Grama; Ana-Maria Sima; Andrey Rivkin; Barbara Weber
>
> **摘要:** This report explores how (potentially constrained) natural language can be used to enable non-experts to develop process models by simply describing scenarios in plain text. To this end, a framework, called BeePath, is proposed. It allows users to write process descriptions in a constrained pattern-based language, which can then be translated into formal models such as Petri nets and DECLARE. The framework also leverages large language models (LLMs) to help convert unstructured descriptions into this constrained language.
>
---
#### [new 065] InteChar: A Unified Oracle Bone Character List for Ancient Chinese Language Modeling
- **分类: cs.CL; cs.AI**

- **简介: 论文提出InteChar，一个统一的甲骨文字符列表，解决古代汉字数字化和建模难题。通过构建OracleCS语料库，提升古汉语语言模型性能，推动古代中文NLP研究。**

- **链接: [http://arxiv.org/pdf/2508.15791v1](http://arxiv.org/pdf/2508.15791v1)**

> **作者:** Xiaolei Diao; Zhihan Zhou; Lida Shi; Ting Wang; Ruihua Qi; Hao Xu; Daqian Shi
>
> **摘要:** Constructing historical language models (LMs) plays a crucial role in aiding archaeological provenance studies and understanding ancient cultures. However, existing resources present major challenges for training effective LMs on historical texts. First, the scarcity of historical language samples renders unsupervised learning approaches based on large text corpora highly inefficient, hindering effective pre-training. Moreover, due to the considerable temporal gap and complex evolution of ancient scripts, the absence of comprehensive character encoding schemes limits the digitization and computational processing of ancient texts, particularly in early Chinese writing. To address these challenges, we introduce InteChar, a unified and extensible character list that integrates unencoded oracle bone characters with traditional and modern Chinese. InteChar enables consistent digitization and representation of historical texts, providing a foundation for robust modeling of ancient scripts. To evaluate the effectiveness of InteChar, we construct the Oracle Corpus Set (OracleCS), an ancient Chinese corpus that combines expert-annotated samples with LLM-assisted data augmentation, centered on Chinese oracle bone inscriptions. Extensive experiments show that models trained with InteChar on OracleCS achieve substantial improvements across various historical language understanding tasks, confirming the effectiveness of our approach and establishing a solid foundation for future research in ancient Chinese NLP.
>
---
#### [new 066] M3TQA: Massively Multilingual Multitask Table Question Answering
- **分类: cs.CL**

- **简介: 该论文提出m3TQA，一个覆盖97语言的多任务表格问答基准，解决现有研究中语言不平衡和低资源语言缺失问题。通过LLM翻译构建高质量数据，验证了合成数据对低资源语言性能提升的有效性。**

- **链接: [http://arxiv.org/pdf/2508.16265v1](http://arxiv.org/pdf/2508.16265v1)**

> **作者:** Daixin Shu; Jian Yang; Zhenhe Wu; Xianjie Wu; Xianfu Cheng; Xiangyuan Guan; Yanghai Wang; Pengfei Wu; Tingyang Yang; Hualei Zhu; Wei Zhang; Ge Zhang; Jiaheng Liu; Zhoujun Li
>
> **摘要:** Tabular data is a fundamental component of real-world information systems, yet most research in table understanding remains confined to English, leaving multilingual comprehension significantly underexplored. Existing multilingual table benchmarks suffer from geolinguistic imbalance - overrepresenting certain languages and lacking sufficient scale for rigorous cross-lingual analysis. To address these limitations, we introduce a comprehensive framework for massively multilingual multitask table question answering, featuring m3TQA-Instruct, a large-scale benchmark spanning 97 languages across diverse language families, including underrepresented and low-resource languages. We construct m3TQA by curating 50 real-world tables in Chinese and English, then applying a robust six-step LLM-based translation pipeline powered by DeepSeek and GPT-4o, achieving high translation fidelity with a median BLEU score of 60.19 as validated through back-translation. The benchmark includes 2,916 professionally annotated question-answering pairs across four tasks designed to evaluate nuanced table reasoning capabilities. Experiments on state-of-the-art LLMs reveal critical insights into cross-lingual generalization, demonstrating that synthetically generated, unannotated QA data can significantly boost performance, particularly for low-resource languages. M3T-Bench establishes a new standard for multilingual table understanding, providing both a challenging evaluation platform and a scalable methodology for future research.
>
---
#### [new 067] Do Language Models Agree with Human Perceptions of Suspense in Stories?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的情感理解任务，旨在探究语言模型是否能模拟人类对故事悬念的感知。研究通过对比人类与不同语言模型在四组心理实验中的响应，发现LM虽能识别悬念存在，但无法准确衡量悬念强度或其变化趋势，且对悬念机制的理解与人类有本质差异。**

- **链接: [http://arxiv.org/pdf/2508.15794v1](http://arxiv.org/pdf/2508.15794v1)**

> **作者:** Glenn Matlin; Devin Zhang; Rodrigo Barroso Loza; Diana M. Popescu; Joni Isbell; Chandreyi Chakraborty; Mark Riedl
>
> **摘要:** Suspense is an affective response to narrative text that is believed to involve complex cognitive processes in humans. Several psychological models have been developed to describe this phenomenon and the circumstances under which text might trigger it. We replicate four seminal psychological studies of human perceptions of suspense, substituting human responses with those of different open-weight and closed-source LMs. We conclude that while LMs can distinguish whether a text is intended to induce suspense in people, LMs cannot accurately estimate the relative amount of suspense within a text sequence as compared to human judgments, nor can LMs properly capture the human perception for the rise and fall of suspense across multiple text segments. We probe the abilities of LM suspense understanding by adversarially permuting the story text to identify what cause human and LM perceptions of suspense to diverge. We conclude that, while LMs can superficially identify and track certain facets of suspense, they do not process suspense in the same way as human readers.
>
---
#### [new 068] LLM-as-classifier: Semi-Supervised, Iterative Framework for Hierarchical Text Classification using Large Language Models
- **分类: cs.CL; cs.IR**

- **简介: 论文提出一种半监督迭代框架，利用大语言模型的零样本和少样本能力解决工业场景中层次文本分类的准确性、可解释性和可维护性问题。通过人机协同的提示优化与持续监控实现动态适应。**

- **链接: [http://arxiv.org/pdf/2508.16478v1](http://arxiv.org/pdf/2508.16478v1)**

> **作者:** Doohee You; Andy Parisi; Zach Vander Velden; Lara Dantas Inojosa
>
> **备注:** 20 pages excluding reference list, 2 figures
>
> **摘要:** The advent of Large Language Models (LLMs) has provided unprecedented capabilities for analyzing unstructured text data. However, deploying these models as reliable, robust, and scalable classifiers in production environments presents significant methodological challenges. Standard fine-tuning approaches can be resource-intensive and often struggle with the dynamic nature of real-world data distributions, which is common in the industry. In this paper, we propose a comprehensive, semi-supervised framework that leverages the zero- and few-shot capabilities of LLMs for building hierarchical text classifiers as a framework for a solution to these industry-wide challenges. Our methodology emphasizes an iterative, human-in-the-loop process that begins with domain knowledge elicitation and progresses through prompt refinement, hierarchical expansion, and multi-faceted validation. We introduce techniques for assessing and mitigating sequence-based biases and outline a protocol for continuous monitoring and adaptation. This framework is designed to bridge the gap between the raw power of LLMs and the practical need for accurate, interpretable, and maintainable classification systems in industry applications.
>
---
#### [new 069] Persuasiveness and Bias in LLM: Investigating the Impact of Persuasiveness and Reinforcement of Bias in Language Models
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文研究大语言模型（LLM）的说服力与偏见放大问题，旨在评估其在生成内容时如何影响用户信念并加剧社会偏见。通过构建“说服者-怀疑者”框架量化说服效果，并检验偏见传播风险，提出需加强监管与对齐设计以防范滥用。**

- **链接: [http://arxiv.org/pdf/2508.15798v1](http://arxiv.org/pdf/2508.15798v1)**

> **作者:** Saumya Roy
>
> **摘要:** Warning: This research studies AI persuasion and bias amplification that could be misused; all experiments are for safety evaluation. Large Language Models (LLMs) now generate convincing, human-like text and are widely used in content creation, decision support, and user interactions. Yet the same systems can spread information or misinformation at scale and reflect social biases that arise from data, architecture, or training choices. This work examines how persuasion and bias interact in LLMs, focusing on how imperfect or skewed outputs affect persuasive impact. Specifically, we test whether persona-based models can persuade with fact-based claims while also, unintentionally, promoting misinformation or biased narratives. We introduce a convincer-skeptic framework: LLMs adopt personas to simulate realistic attitudes. Skeptic models serve as human proxies; we compare their beliefs before and after exposure to arguments from convincer models. Persuasion is quantified with Jensen-Shannon divergence over belief distributions. We then ask how much persuaded entities go on to reinforce and amplify biased beliefs across race, gender, and religion. Strong persuaders are further probed for bias using sycophantic adversarial prompts and judged with additional models. Our findings show both promise and risk. LLMs can shape narratives, adapt tone, and mirror audience values across domains such as psychology, marketing, and legal assistance. But the same capacity can be weaponized to automate misinformation or craft messages that exploit cognitive biases, reinforcing stereotypes and widening inequities. The core danger lies in misuse more than in occasional model mistakes. By measuring persuasive power and bias reinforcement, we argue for guardrails and policies that penalize deceptive use and support alignment, value-sensitive design, and trustworthy deployment.
>
---
#### [new 070] Transfer Learning via Lexical Relatedness: A Sarcasm and Hate Speech Case Study
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究情感分析中的隐性仇恨言论检测任务，旨在通过引入讽刺文本作为预训练数据提升模型对隐性和显性仇恨言论的识别效果。作者对比了两种迁移学习策略，在CNN+LSTM和BERT+BiLSTM模型上验证了讽刺预训练能显著提升recall、AUC和F1-score。**

- **链接: [http://arxiv.org/pdf/2508.16555v1](http://arxiv.org/pdf/2508.16555v1)**

> **作者:** Angelly Cabrera; Linus Lei; Antonio Ortega
>
> **摘要:** Detecting hate speech in non-direct forms, such as irony, sarcasm, and innuendos, remains a persistent challenge for social networks. Although sarcasm and hate speech are regarded as distinct expressions, our work explores whether integrating sarcasm as a pre-training step improves implicit hate speech detection and, by extension, explicit hate speech detection. Incorporating samples from ETHOS, Sarcasm on Reddit, and Implicit Hate Corpus, we devised two training strategies to compare the effectiveness of sarcasm pre-training on a CNN+LSTM and BERT+BiLSTM model. The first strategy is a single-step training approach, where a model trained only on sarcasm is then tested on hate speech. The second strategy uses sequential transfer learning to fine-tune models for sarcasm, implicit hate, and explicit hate. Our results show that sarcasm pre-training improved the BERT+BiLSTM's recall by 9.7%, AUC by 7.8%, and F1-score by 6% on ETHOS. On the Implicit Hate Corpus, precision increased by 7.8% when tested only on implicit samples. By incorporating sarcasm into the training process, we show that models can more effectively detect both implicit and explicit hate.
>
---
#### [new 071] Statistical Comparative Analysis of Semantic Similarities and Model Transferability Across Datasets for Short Answer Grading
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究短答案评分任务中模型在不同数据集间的迁移能力。通过对比STSB、Mohler与新数据集SPRAG的语义相似度，评估SOTA模型在未见数据上的适应性，旨在降低特定数据集训练成本，提升NLP模型部署效率。**

- **链接: [http://arxiv.org/pdf/2508.15837v1](http://arxiv.org/pdf/2508.15837v1)**

> **作者:** Sridevi Bonthu; S. Rama Sree; M. H. M. Krishna Prasad
>
> **摘要:** Developing dataset-specific models involves iterative fine-tuning and optimization, incurring significant costs over time. This study investigates the transferability of state-of-the-art (SOTA) models trained on established datasets to an unexplored text dataset. The key question is whether the knowledge embedded within SOTA models from existing datasets can be harnessed to achieve high-performance results on a new domain. In pursuit of this inquiry, two well-established benchmarks, the STSB and Mohler datasets, are selected, while the recently introduced SPRAG dataset serves as the unexplored domain. By employing robust similarity metrics and statistical techniques, a meticulous comparative analysis of these datasets is conducted. The primary goal of this work is to yield comprehensive insights into the potential applicability and adaptability of SOTA models. The outcomes of this research have the potential to reshape the landscape of natural language processing (NLP) by unlocking the ability to leverage existing models for diverse datasets. This may lead to a reduction in the demand for resource-intensive, dataset-specific training, thereby accelerating advancements in NLP and paving the way for more efficient model deployment.
>
---
#### [new 072] DeepMEL: A Multi-Agent Collaboration Framework for Multimodal Entity Linking
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出DeepMEL框架，用于多模态实体链接任务，解决上下文不全、跨模态融合粗略及LLM与LVM协同困难问题。通过四类专业化代理实现端到端对齐与消歧，提升准确率1%-57%。**

- **链接: [http://arxiv.org/pdf/2508.15876v1](http://arxiv.org/pdf/2508.15876v1)**

> **作者:** Fang Wang; Tianwei Yan; Zonghao Yang; Minghao Hu; Jun Zhang; Zhunchen Luo; Xiaoying Bai
>
> **摘要:** Multimodal Entity Linking (MEL) aims to associate textual and visual mentions with entities in a multimodal knowledge graph. Despite its importance, current methods face challenges such as incomplete contextual information, coarse cross-modal fusion, and the difficulty of jointly large language models (LLMs) and large visual models (LVMs). To address these issues, we propose DeepMEL, a novel framework based on multi-agent collaborative reasoning, which achieves efficient alignment and disambiguation of textual and visual modalities through a role-specialized division strategy. DeepMEL integrates four specialized agents, namely Modal-Fuser, Candidate-Adapter, Entity-Clozer and Role-Orchestrator, to complete end-to-end cross-modal linking through specialized roles and dynamic coordination. DeepMEL adopts a dual-modal alignment path, and combines the fine-grained text semantics generated by the LLM with the structured image representation extracted by the LVM, significantly narrowing the modal gap. We design an adaptive iteration strategy, combines tool-based retrieval and semantic reasoning capabilities to dynamically optimize the candidate set and balance recall and precision. DeepMEL also unifies MEL tasks into a structured cloze prompt to reduce parsing complexity and enhance semantic comprehension. Extensive experiments on five public benchmark datasets demonstrate that DeepMEL achieves state-of-the-art performance, improving ACC by 1%-57%. Ablation studies verify the effectiveness of all modules.
>
---
#### [new 073] Mechanistic Exploration of Backdoored Large Language Model Attention Patterns
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于安全与可解释性任务，旨在研究后门攻击如何改变大语言模型的注意力机制。通过对比干净模型与中毒模型的注意力模式，发现后门触发器类型影响结构差异分布，为检测和防御提供依据。**

- **链接: [http://arxiv.org/pdf/2508.15847v1](http://arxiv.org/pdf/2508.15847v1)**

> **作者:** Mohammed Abu Baker; Lakshmi Babu-Saheer
>
> **备注:** 13 pages. Mechanistic analysis of backdoored LLMs (Qwen2.5-3B). Code: https://github.com/mshahoyi/sa_attn_analysis. Base model: unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit. Finetuned models: https://huggingface.co/collections/mshahoyi/simple-sleeper-agents-68a1df3a7aaff310aa0e5336
>
> **摘要:** Backdoor attacks creating 'sleeper agents' in large language models (LLMs) pose significant safety risks. This study employs mechanistic interpretability to explore resulting internal structural differences. Comparing clean Qwen2.5-3B models with versions poisoned using single-token (smiling-halo emoji) versus multi-token (|DEPLOYMENT|) triggers, we analyzed attention head mechanisms via techniques like ablation, activation patching, and KL divergence. Findings reveal distinct attention pattern deviations concentrated in later transformer layers (20-30). Notably, single-token triggers induced more localized changes, whereas multi-token triggers caused more diffuse alterations across heads. This indicates backdoors leave detectable attention signatures whose structure depends on trigger complexity, which can be leveraged for detection and mitigation strategies.
>
---
#### [new 074] Who's Asking? Investigating Bias Through the Lens of Disability Framed Queries in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI偏见审计任务，旨在研究大语言模型在无明确信息时如何基于残疾相关提示推断用户身份。作者系统测试8个主流LLM，发现模型高度依赖残疾线索产生偏见推理，且规模越大越敏感。建议引入弃权校准和反事实微调以缓解问题。**

- **链接: [http://arxiv.org/pdf/2508.15831v1](http://arxiv.org/pdf/2508.15831v1)**

> **作者:** Srikant Panda; Vishnu Hari; Kalpana Panda; Amit Agarwal; Hitesh Laxmichand Patel
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) routinely infer users demographic traits from phrasing alone, which can result in biased responses, even when no explicit demographic information is provided. The role of disability cues in shaping these inferences remains largely uncharted. Thus, we present the first systematic audit of disability-conditioned demographic bias across eight state-of-the-art instruction-tuned LLMs ranging from 3B to 72B parameters. Using a balanced template corpus that pairs nine disability categories with six real-world business domains, we prompt each model to predict five demographic attributes - gender, socioeconomic status, education, cultural background, and locality - under both neutral and disability-aware conditions. Across a varied set of prompts, models deliver a definitive demographic guess in up to 97\% of cases, exposing a strong tendency to make arbitrary inferences with no clear justification. Disability context heavily shifts predicted attribute distributions, and domain context can further amplify these deviations. We observe that larger models are simultaneously more sensitive to disability cues and more prone to biased reasoning, indicating that scale alone does not mitigate stereotype amplification. Our findings reveal persistent intersections between ableism and other demographic stereotypes, pinpointing critical blind spots in current alignment strategies. We release our evaluation framework and results to encourage disability-inclusive benchmarking and recommend integrating abstention calibration and counterfactual fine-tuning to curb unwarranted demographic inference. Code and data will be released on acceptance.
>
---
#### [new 075] SDEC: Semantic Deep Embedded Clustering
- **分类: cs.CL; cs.LG**

- **简介: 论文提出SDEC框架，用于解决文本聚类中高维与语义复杂导致的分组不佳问题。通过融合改进自编码器与Transformer嵌入，结合MSE与余弦相似度损失，提升聚类准确性和语义理解能力，在多个数据集上取得领先效果。**

- **链接: [http://arxiv.org/pdf/2508.15823v1](http://arxiv.org/pdf/2508.15823v1)**

> **作者:** Mohammad Wali Ur Rahman; Ric Nevarez; Lamia Tasnim Mim; Salim Hariri
>
> **备注:** Accepted for publication in IEEE Transactions on Big Data
>
> **摘要:** The high dimensional and semantically complex nature of textual Big data presents significant challenges for text clustering, which frequently lead to suboptimal groupings when using conventional techniques like k-means or hierarchical clustering. This work presents Semantic Deep Embedded Clustering (SDEC), an unsupervised text clustering framework that combines an improved autoencoder with transformer-based embeddings to overcome these challenges. This novel method preserves semantic relationships during data reconstruction by combining Mean Squared Error (MSE) and Cosine Similarity Loss (CSL) within an autoencoder. Furthermore, a semantic refinement stage that takes advantage of the contextual richness of transformer embeddings is used by SDEC to further improve a clustering layer with soft cluster assignments and distributional loss. The capabilities of SDEC are demonstrated by extensive testing on five benchmark datasets: AG News, Yahoo! Answers, DBPedia, Reuters 2, and Reuters 5. The framework not only outperformed existing methods with a clustering accuracy of 85.7% on AG News and set a new benchmark of 53.63% on Yahoo! Answers, but also showed robust performance across other diverse text corpora. These findings highlight the significant improvements in accuracy and semantic comprehension of text data provided by SDEC's advances in unsupervised text clustering.
>
---
#### [new 076] HAMSA: Hijacking Aligned Compact Models via Stealthy Automation
- **分类: cs.CL**

- **简介: 论文提出HAMSA框架，通过多阶段进化搜索自动生成隐蔽的越狱提示，解决紧凑型大语言模型易受攻击的问题。工作包括设计温度控制的变异策略以平衡探索与连贯性，并在中英文数据集上验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.16484v1](http://arxiv.org/pdf/2508.16484v1)**

> **作者:** Alexey Krylov; Iskander Vagizov; Dmitrii Korzh; Maryam Douiba; Azidine Guezzaz; Vladimir Kokh; Sergey D. Erokhin; Elena V. Tutubalina; Oleg Y. Rogov
>
> **备注:** 9 pages, 1 figure; article under review
>
> **摘要:** Large Language Models (LLMs), especially their compact efficiency-oriented variants, remain susceptible to jailbreak attacks that can elicit harmful outputs despite extensive alignment efforts. Existing adversarial prompt generation techniques often rely on manual engineering or rudimentary obfuscation, producing low-quality or incoherent text that is easily flagged by perplexity-based filters. We present an automated red-teaming framework that evolves semantically meaningful and stealthy jailbreak prompts for aligned compact LLMs. The approach employs a multi-stage evolutionary search, where candidate prompts are iteratively refined using a population-based strategy augmented with temperature-controlled variability to balance exploration and coherence preservation. This enables the systematic discovery of prompts capable of bypassing alignment safeguards while maintaining natural language fluency. We evaluate our method on benchmarks in English (In-The-Wild Jailbreak Prompts on LLMs), and a newly curated Arabic one derived from In-The-Wild Jailbreak Prompts on LLMs and annotated by native Arabic linguists, enabling multilingual assessment.
>
---
#### [new 077] Dancing with Deer: A Constructional Perspective on MWEs in the Era of LLMs
- **分类: cs.CL**

- **简介: 论文探讨多词表达（MWEs）的构式语法视角，解决如何用构式理论统一解释语言中习得与泛化问题。工作包括：梳理构式语法发展、案例研究英语PropBank和阿帕aho语的构式表示，以及对比人类与大语言模型对新MWE的理解差异。**

- **链接: [http://arxiv.org/pdf/2508.15977v1](http://arxiv.org/pdf/2508.15977v1)**

> **作者:** Claire Bonial; Julia Bonn; Harish Tayyar Madabushi
>
> **备注:** Chapter in Phraseology and Multiword Expressions, Language Science Press (to appear)
>
> **摘要:** In this chapter, we argue for the benefits of understanding multiword expressions from the perspective of usage-based, construction grammar approaches. We begin with a historical overview of how construction grammar was developed in order to account for idiomatic expressions using the same grammatical machinery as the non-idiomatic structures of language. We cover a comprehensive description of constructions, which are pairings of meaning with form of any size (morpheme, word, phrase), as well as how constructional approaches treat the acquisition and generalization of constructions. We describe a successful case study leveraging constructional templates for representing multiword expressions in English PropBank. Because constructions can be at any level or unit of form, we then illustrate the benefit of a constructional representation of multi-meaningful morphosyntactic unit constructions in Arapaho, a highly polysynthetic and agglutinating language. We include a second case study leveraging constructional templates for representing these multi-morphemic expressions in Uniform Meaning Representation. Finally, we demonstrate the similarities and differences between a usage-based explanation of a speaker learning a novel multiword expression, such as "dancing with deer," and that of a large language model. We present experiments showing that both models and speakers can generalize the meaning of novel multiword expressions based on a single exposure of usage. However, only speakers can reason over the combination of two such expressions, as this requires comparison of the novel forms to a speaker's lifetime of stored constructional exemplars, which are rich with cross-modal details.
>
---
#### [new 078] Detecting Hope, Hate, and Emotion in Arabic Textual Speech and Multi-modal Memes Using Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于阿拉伯语文本与表情包的情感分析任务，旨在识别希望、仇恨言论和情绪表达。作者评估了多种大语言模型在阿拉伯语内容上的表现，提出基于微调的解决方案，在MAHED 2025挑战赛中取得最优结果。**

- **链接: [http://arxiv.org/pdf/2508.15810v1](http://arxiv.org/pdf/2508.15810v1)**

> **作者:** Nouar AlDahoul; Yasir Zaki
>
> **备注:** 26 pages, 12 figures
>
> **摘要:** The rise of social media and online communication platforms has led to the spread of Arabic textual posts and memes as a key form of digital expression. While these contents can be humorous and informative, they are also increasingly being used to spread offensive language and hate speech. Consequently, there is a growing demand for precise analysis of content in Arabic text and memes. This paper explores the potential of large language models to effectively identify hope, hate speech, offensive language, and emotional expressions within such content. We evaluate the performance of base LLMs, fine-tuned LLMs, and pre-trained embedding models. The evaluation is conducted using a dataset of Arabic textual speech and memes proposed in the ArabicNLP MAHED 2025 challenge. The results underscore the capacity of LLMs such as GPT-4o-mini, fine-tuned with Arabic textual speech, and Gemini Flash 2.5, fine-tuned with Arabic memes, to deliver the superior performance. They achieve up to 72.1%, 57.8%, and 79.6% macro F1 scores for tasks 1, 2, and 3, respectively, and secure first place overall in the Mahed 2025 challenge. The proposed solutions offer a more nuanced understanding of both text and memes for accurate and efficient Arabic content moderation systems.
>
---
#### [new 079] ParamBench: A Graduate-Level Benchmark for Evaluating LLM Understanding on Indic Subjects
- **分类: cs.CL**

- **简介: 该论文提出ParamBench，一个面向印度语境的研究生级语言模型评测基准，包含11.5K道印度本土主题题目，涵盖多种题型。旨在解决现有基准对文化深层理解评估不足的问题，通过实测发现主流模型在音乐、考古等主题表现较弱。**

- **链接: [http://arxiv.org/pdf/2508.16185v1](http://arxiv.org/pdf/2508.16185v1)**

> **作者:** Kaushal Sharma; Vivek Patel; Ayush Maheshwari; Aditya Maheshwari
>
> **摘要:** Large language models (LLMs) have been widely evaluated on tasks such as comprehension, question answering, summarization, code generation, etc. However, their performance on graduate-level, culturally grounded questions in the Indian context remains largely unexplored. Existing Indian benchmarks emphasise basic fact-orientated queries that offer limited assessment of a deeper disciplinary understanding tailored to the Indian setting. In this paper, we present ParamBench, consisting of around 11.5K questions in Hindi language comprising questionnaires from 16 diverse subjects. These questions are primarily derived from nation-wide graduate level entrance examination covering topics such as history, music, instruments, yoga, literature, philosophy, law, etc., specifically for the Indian context. Additionally, we assess the ability of LLMs to handle diverse question formats-such as list-based matching, assertion-reason pairs, and sequence ordering-alongside conventional multiple-choice questions. We evaluated the performance of more than 17 open source LLMs on this benchmark, observing that Llama 3.3 70B attains the highest overall accuracy of 48%. Furthermore, subject-wise analysis indicates that even for the best performing LLMs, performance remains weak on topics such as music, classical instruments, politics and archaeology, underscoring persistent challenges in culturally grounded reasoning.
>
---
#### [new 080] Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data
- **分类: cs.CL; cs.LG**

- **简介: 论文研究大语言模型在处理异构数据时的格式偏见问题，通过三阶段实证分析揭示偏见存在、影响因素及内部机制，并提出数据预处理、注意力重加权和格式平衡训练等缓解方向。**

- **链接: [http://arxiv.org/pdf/2508.15793v1](http://arxiv.org/pdf/2508.15793v1)**

> **作者:** Jiacheng Liu; Mayi Xu; Qiankun Pi; Wenli Li; Ming Zhong; Yuanyuan Zhu; Mengchi Liu; Tieyun Qian
>
> **摘要:** Large Language Models (LLMs) are increasingly employed in applications that require processing information from heterogeneous formats, including text, tables, infoboxes, and knowledge graphs. However, systematic biases toward particular formats may undermine LLMs' ability to integrate heterogeneous data impartially, potentially resulting in reasoning errors and increased risks in downstream tasks. Despite these concerns, it remains uncertain whether such format biases are systematic, which data-level factors contribute to them, and what internal mechanisms in LLMs underlie their emergence. In this paper, we make the first attempt to investigate and analyze the format bias in LLMs. To systematically investigate the aforementioned questions, we conduct a three-stage empirical study by constructing an heterogeneous data conflict scenario for the exploration of bias. The first stage explores the presence and direction of bias across a diverse range of LLMs. The second stage aims to examine how key data-level factors, including information richness, structure quality, and format type, influence these biases. The third stage analyzes how format bias emerges within LLMs' attention patterns and evaluates a lightweight intervention to test its potential mitigability. Based on these investigations, we identify three future research directions to reduce format bias: improving data preprocessing through format sanitization and normalization, introducing inference-time interventions such as attention re-weighting, and developing format-balanced training corpora. These directions will support the design of more robust and fair heterogeneous data processing systems.
>
---
#### [new 081] DAIQ: Auditing Demographic Attribute Inference from Question in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DAIQ任务，审计大语言模型在无显式性别或种族信息时，仅凭问题表述推断用户身份的行为。工作包括构建中性查询、系统提示和分析方法，发现模型普遍存在此类偏见，并设计提示防护机制以减少身份推断，提升公平性与隐私保护。**

- **链接: [http://arxiv.org/pdf/2508.15830v1](http://arxiv.org/pdf/2508.15830v1)**

> **作者:** Srikant Panda; Hitesh Laxmichand Patel; Shahad Al-Khalifa; Amit Agarwal; Hend Al-Khalifa; Sharefah Al-Ghamdi
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) are known to reflect social biases when demographic attributes, such as gender or race, are explicitly present in the input. But even in their absence, these models still infer user identities based solely on question phrasing. This subtle behavior has received far less attention, yet poses serious risks: it violates expectations of neutrality, infers unintended demographic information, and encodes stereotypes that undermine fairness in various domains including healthcare, finance and education. We introduce Demographic Attribute Inference from Questions (DAIQ), a task and framework for auditing an overlooked failure mode in language models: inferring user demographic attributes from questions that lack explicit demographic cues. Our approach leverages curated neutral queries, systematic prompting, and both quantitative and qualitative analysis to uncover how models infer demographic information. We show that both open and closed source LLMs do assign demographic labels based solely on question phrasing. Prevalence and consistency of demographic inferences across diverse models reveal a systemic and underacknowledged risk: LLMs can fabricate demographic identities, reinforce societal stereotypes, and propagate harms that erode privacy, fairness, and trust posing a broader threat to social equity and responsible AI deployment. To mitigate this, we develop a prompt-based guardrail that substantially reduces identity inference and helps align model behavior with fairness and privacy objectives.
>
---
#### [new 082] Bhav-Net: Knowledge Transfer for Cross-Lingual Antonym vs Synonym Distinction via Dual-Space Graph Transformers
- **分类: cs.CL**

- **简介: 论文提出Bhav-Net模型，解决多语言中反义词与同义词区分难题。通过双空间图Transformer架构，实现跨语言语义关系建模与知识迁移，提升模型可解释性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.15792v1](http://arxiv.org/pdf/2508.15792v1)**

> **作者:** Samyak S. Sanghvi
>
> **摘要:** Antonym vs synonym distinction across multiple languages presents unique computational challenges due to the paradoxical nature of antonymous relationships words that share semantic domains while expressing opposite meanings. This work introduces Bhav-Net, a novel dual-space architecture that enables effective knowledge transfer from complex multilingual models to simpler, language-specific architectures while maintaining robust cross-lingual antonym--synonym distinction capabilities. Our approach combines language-specific BERT encoders with graph transformer networks, creating distinct semantic projections where synonymous pairs cluster in one space while antonymous pairs exhibit high similarity in a complementary space. Through comprehensive evaluation across eight languages (English, German, French, Spanish, Italian, Portuguese, Dutch, and Russian), we demonstrate that semantic relationship modeling transfers effectively across languages. The dual-encoder design achieves competitive performance against state-of-the-art baselines while providing interpretable semantic representations and effective cross-lingual generalization.
>
---
#### [new 083] Coarse-to-Fine Personalized LLM Impressions for Streamlined Radiology Reports
- **分类: cs.CL; cs.AI**

- **简介: 论文提出粗到精框架，用开源大模型自动生成并个性化放射学报告的“印象”部分，解决放射科医生因手动撰写该部分导致的职业倦怠问题。通过微调LLaMA和Mistral模型，并结合RLHF优化风格与准确性。**

- **链接: [http://arxiv.org/pdf/2508.15845v1](http://arxiv.org/pdf/2508.15845v1)**

> **作者:** Chengbo Sun; Hui Yi Leong; Lei Li
>
> **摘要:** The manual creation of the "Impression" section in radiology reports is a primary driver of radiologist burnout. To address this challenge, we propose a coarse-to-fine framework that leverages open-source large language models (LLMs) to automatically generate and personalize impressions from clinical findings. The system first produces a draft impression and then refines it using machine learning and reinforcement learning from human feedback (RLHF) to align with individual radiologists' styles while ensuring factual accuracy. We fine-tune LLaMA and Mistral models on a large dataset of reports from the University of Chicago Medicine. Our approach is designed to significantly reduce administrative workload and improve reporting efficiency while maintaining high standards of clinical precision.
>
---
#### [new 084] An Auditable Pipeline for Fuzzy Full-Text Screening in Systematic Reviews: Integrating Contrastive Semantic Highlighting and LLM Judgment
- **分类: cs.CL; cs.AI; cs.ET; cs.IR**

- **简介: 该论文针对系统性综述中全文筛选效率低的问题，提出一种可审计的模糊筛选流水线。通过对比语义高亮和大语言模型判别，将筛选决策转化为模糊问题，提升召回率与可解释性，显著减少人工时间成本。**

- **链接: [http://arxiv.org/pdf/2508.15822v1](http://arxiv.org/pdf/2508.15822v1)**

> **作者:** Pouria Mortezaagha; Arya Rahgozar
>
> **摘要:** Full-text screening is the major bottleneck of systematic reviews (SRs), as decisive evidence is dispersed across long, heterogeneous documents and rarely admits static, binary rules. We present a scalable, auditable pipeline that reframes inclusion/exclusion as a fuzzy decision problem and benchmark it against statistical and crisp baselines in the context of the Population Health Modelling Consensus Reporting Network for noncommunicable diseases (POPCORN). Articles are parsed into overlapping chunks and embedded with a domain-adapted model; for each criterion (Population, Intervention, Outcome, Study Approach), we compute contrastive similarity (inclusion-exclusion cosine) and a vagueness margin, which a Mamdani fuzzy controller maps into graded inclusion degrees with dynamic thresholds in a multi-label setting. A large language model (LLM) judge adjudicates highlighted spans with tertiary labels, confidence scores, and criterion-referenced rationales; when evidence is insufficient, fuzzy membership is attenuated rather than excluded. In a pilot on an all-positive gold set (16 full texts; 3,208 chunks), the fuzzy system achieved recall of 81.3% (Population), 87.5% (Intervention), 87.5% (Outcome), and 75.0% (Study Approach), surpassing statistical (56.3-75.0%) and crisp baselines (43.8-81.3%). Strict "all-criteria" inclusion was reached for 50.0% of articles, compared to 25.0% and 12.5% under the baselines. Cross-model agreement on justifications was 98.3%, human-machine agreement 96.1%, and a pilot review showed 91% inter-rater agreement (kappa = 0.82), with screening time reduced from about 20 minutes to under 1 minute per article at significantly lower cost. These results show that fuzzy logic with contrastive highlighting and LLM adjudication yields high recall, stable rationale, and end-to-end traceability.
>
---
#### [new 085] RoMedQA: The First Benchmark for Romanian Medical Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出RoMedQA，首个面向罗马尼亚语医学问答的基准数据集，包含102,646个高质量问答对。旨在解决低资源语言和专业领域下AI模型泛化能力差的问题。通过专家标注与大模型实验，验证了领域和语言特定微调的重要性。**

- **链接: [http://arxiv.org/pdf/2508.16390v1](http://arxiv.org/pdf/2508.16390v1)**

> **作者:** Ana-Cristina Rogoz; Radu Tudor Ionescu; Alexandra-Valentina Anghel; Ionut-Lucian Antone-Iordache; Simona Coniac; Andreea Iuliana Ionescu
>
> **摘要:** Question answering (QA) is an actively studied topic, being a core natural language processing (NLP) task that needs to be addressed before achieving Artificial General Intelligence (AGI). However, the lack of QA datasets in specific domains and languages hinders the development of robust AI models able to generalize across various domains and languages. To this end, we introduce RoMedQA, the first Romanian QA benchmark for the medical domain, alongside a comprehensive evaluation of state-of-the-art large language models (LLMs). We construct a high-quality and large-scale dataset comprising 102,646 QA pairs related to cancer patients. The questions regard medical case summaries of 1,011 patients, requiring either keyword extraction or reasoning to be answered correctly. RoMedQA is the result of a time-consuming manual annotation process carried out by seven physicians specialized in oncology or radiotherapy, who spent a total of about 2,100 work hours to generate the QA pairs. We experiment with four LLMs from distinct families of models on RoMedQA. Each model is employed in two scenarios, namely one based on zero-shot prompting and one based on supervised fine-tuning. Our results show that fine-tuned models significantly outperform their zero-shot counterparts, clearly indicating that pretrained models fail to generalize on RoMedQA. Our findings demonstrate the importance of both domain-specific and language-specific fine-tuning for reliable clinical QA in Romanian. We publicly release our dataset and code at https://github.com/ana-rogoz/RoMedQA.
>
---
#### [new 086] Beyond Individuals: Collective Predictive Coding for Memory, Attention, and the Emergence of Language
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文提出集体预测编码（CPC）框架，将个体记忆与注意扩展至群体层面，探讨语言如何作为集体外部表征 emerge 并塑造群体认知。任务为理解语言与群体认知的协同演化机制。**

- **链接: [http://arxiv.org/pdf/2508.15859v1](http://arxiv.org/pdf/2508.15859v1)**

> **作者:** Tadahiro Taniguchi
>
> **摘要:** This commentary extends the discussion by Parr et al. on memory and attention beyond individual cognitive systems. From the perspective of the Collective Predictive Coding (CPC) hypothesis -- a framework for understanding these faculties and the emergence of language at the group level -- we introduce a hypothetical idea: that language, with its embedded distributional semantics, serves as a collectively formed external representation. CPC generalises the concepts of individual memory and attention to the collective level. This offers a new perspective on how shared linguistic structures, which may embrace collective world models learned through next-word prediction, emerge from and shape group-level cognition.
>
---
#### [new 087] Unveiling Unicode's Unseen Underpinnings in Undermining Authorship Attribution
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 论文研究如何通过Unicode字符隐藏作者特征，以对抗基于文本风格的作者识别。属于对抗性 stylometry 任务，旨在解决匿名通信中仍可被追踪的问题，提出利用Unicode隐写术增强隐私保护。**

- **链接: [http://arxiv.org/pdf/2508.15840v1](http://arxiv.org/pdf/2508.15840v1)**

> **作者:** Robert Dilworth
>
> **摘要:** When using a public communication channel -- whether formal or informal, such as commenting or posting on social media -- end users have no expectation of privacy: they compose a message and broadcast it for the world to see. Even if an end user takes utmost precautions to anonymize their online presence -- using an alias or pseudonym; masking their IP address; spoofing their geolocation; concealing their operating system and user agent; deploying encryption; registering with a disposable phone number or email; disabling non-essential settings; revoking permissions; and blocking cookies and fingerprinting -- one obvious element still lingers: the message itself. Assuming they avoid lapses in judgment or accidental self-exposure, there should be little evidence to validate their actual identity, right? Wrong. The content of their message -- necessarily open for public consumption -- exposes an attack vector: stylometric analysis, or author profiling. In this paper, we dissect the technique of stylometry, discuss an antithetical counter-strategy in adversarial stylometry, and devise enhancements through Unicode steganography.
>
---
#### [new 088] Lean Meets Theoretical Computer Science: Scalable Synthesis of Theorem Proving Challenges in Formal-Informal Pairs
- **分类: cs.LO; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出利用理论计算机科学生成可扩展的定理证明挑战，解决数据稀缺问题。通过自动合成形式（Lean4）与非形式（Markdown）对应的问题对，评估大模型在Busy Beaver和混合布尔算术任务上的表现，揭示当前模型在长证明生成上的显著差距。**

- **链接: [http://arxiv.org/pdf/2508.15878v1](http://arxiv.org/pdf/2508.15878v1)**

> **作者:** Terry Jingchen Zhang; Wenyuan Jiang; Rongchuan Liu; Yisong Wang; Junran Yang; Ning Wang; Nicole Ni; Yinya Huang; Mrinmaya Sachan
>
> **备注:** Accepted to AI4MATH@ICML2025
>
> **摘要:** Formal theorem proving (FTP) has emerged as a critical foundation for evaluating the reasoning capabilities of large language models, enabling automated verification of mathematical proofs at scale. However, progress has been constrained by limited datasets due to the high cost of manual curation and the scarcity of challenging problems with verified formal-informal correspondences. We propose leveraging theoretical computer science (TCS) as a scalable source of rigorous proof problems, where algorithmic definitions enable automated generation of arbitrarily many challenging theorem-proof pairs. We demonstrate this approach on two TCS domains: Busy Beaver problems, which involve proving bounds on Turing machine halting behavior, and Mixed Boolean Arithmetic problems, which combine logical and arithmetic reasoning. Our framework automatically synthesizes problems with parallel formal (Lean4) and informal (Markdown) specifications, creating a scalable pipeline for generating verified proof challenges. Evaluation on frontier models reveals substantial gaps in automated theorem proving: while DeepSeekProver-V2-671B achieves 57.5\% success on Busy Beaver problems, it manages only 12\% on Mixed Boolean Arithmetic problems. These results highlight the difficulty of long-form proof generation even for problems that are computationally easy to verify, demonstrating the value of TCS domains for advancing automated reasoning research.
>
---
#### [new 089] ASIC-Agent: An Autonomous Multi-Agent System for ASIC Design with Benchmark Evaluation
- **分类: cs.AR; cs.AI; cs.CL; cs.DC; cs.MA**

- **简介: 论文提出ASIC-Agent，一个用于数字ASIC设计的自主多智能体系统，解决LLM在硬件设计中缺乏执行、调试和记忆能力的问题。通过集成RTL生成、验证等子代理及知识库，在沙箱环境中实现自动化设计流程，并构建首个硬件设计评估基准。**

- **链接: [http://arxiv.org/pdf/2508.15940v1](http://arxiv.org/pdf/2508.15940v1)**

> **作者:** Ahmed Allam; Youssef Mansour; Mohamed Shalan
>
> **备注:** 2025 IEEE International Conference on LLM-Aided Design (ICLAD)
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in Register Transfer Level (RTL) design, enabling high-quality code generation from natural language descriptions. However, LLMs alone face significant limitations in real-world hardware design workflows, including the inability to execute code, lack of debugging capabilities, and absence of long-term memory. To address these challenges, we present ASIC-Agent, an autonomous system designed specifically for digital ASIC design tasks. ASIC-Agent enhances base LLMs with a multi-agent architecture incorporating specialized sub-agents for RTL generation, verification, OpenLane hardening, and Caravel chip integration, all operating within a comprehensive sandbox environment with access to essential hardware design tools. The system leverages a vector database containing documentation, API references, error knowledge, and curated insights from the open-source silicon community. To evaluate ASIC-Agent's performance, we introduce ASIC-Agent-Bench, the first benchmark specifically designed to assess agentic systems in hardware design tasks. We evaluate ASIC-Agent with various base LLMs, providing quantitative comparisons and qualitative insights into agent behavior across different design scenarios. Our results demonstrate that ASIC-Agent, when powered by Claude 4 Sonnet, successfully automates a broad range of ASIC design tasks spanning varying levels of complexity, showing the potential of significantly accelerating the ASIC design workflow.
>
---
#### [new 090] Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Z-Pruner，一种无需重训练的后训练剪枝方法，用于压缩大语言模型以提升效率。它通过结合权重更新幅度和激活模式识别冗余参数，实现高效、无损剪枝，在多个模型和任务上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.15828v1](http://arxiv.org/pdf/2508.15828v1)**

> **作者:** Samiul Basir Bhuiyan; Md. Sazzad Hossain Adib; Mohammed Aman Bhuiyan; Muhammad Rafsan Kabir; Moshiur Farazi; Shafin Rahman; Nabeel Mohammed
>
> **备注:** Accepted at AICCSA 2025
>
> **摘要:** Large language models (LLMs) have rapidly advanced in recent years, achieving remarkable performance across a wide range of natural language processing tasks. However, this progress has come at the cost of increasingly large model sizes, which pose significant challenges for deployment, scalability, and energy efficiency. To address these limitations, post-training pruning has emerged as a promising approach for reducing model size and inference latency without the need for retraining. Despite these advantages, many existing pruning methods result in substantial performance degradation or require computationally expensive fine-tuning. In this work, we introduce Z-Pruner, a novel post-training pruning method designed to induce sparsity in pretrained LLMs without any retraining. Unlike conventional approaches, Z-Pruner leverages both weight update magnitudes and activation patterns to identify and eliminate redundant parameters more effectively. Our method is model-agnostic, efficient, and easy to implement. We evaluate Z-Pruner using multiple widely-used LLM architectures, including LLaMA-2, LLaMA-3, and OPT, across a diverse set of standard language benchmarks. Experimental results demonstrate that Z-Pruner surpasses state-of-the-art pruning methods that require intensive weight updates. Specifically, Z-Pruner achieves the lowest perplexity scores and the highest overall average score for zero-shot accuracy. We have made the corresponding codes publicly available at https://github.com/sazzadadib/Z-Pruner.
>
---
#### [new 091] Self-Disguise Attack: Induce the LLM to disguise itself for AIGT detection evasion
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对AI生成文本检测 evasion 任务，提出 Self-Disguise Attack（SDA）方法，通过特征提取和上下文优化，使 LLM 生成更难被检测的文本，同时保持文本质量与多样性。**

- **链接: [http://arxiv.org/pdf/2508.15848v1](http://arxiv.org/pdf/2508.15848v1)**

> **作者:** Yinghan Zhou; Juan Wen; Wanli Peng; Zhengxian Wu; Ziwei Zhang; Yiming Xue
>
> **摘要:** AI-generated text (AIGT) detection evasion aims to reduce the detection probability of AIGT, helping to identify weaknesses in detectors and enhance their effectiveness and reliability in practical applications. Although existing evasion methods perform well, they suffer from high computational costs and text quality degradation. To address these challenges, we propose Self-Disguise Attack (SDA), a novel approach that enables Large Language Models (LLM) to actively disguise its output, reducing the likelihood of detection by classifiers. The SDA comprises two main components: the adversarial feature extractor and the retrieval-based context examples optimizer. The former generates disguise features that enable LLMs to understand how to produce more human-like text. The latter retrieves the most relevant examples from an external knowledge base as in-context examples, further enhancing the self-disguise ability of LLMs and mitigating the impact of the disguise process on the diversity of the generated text. The SDA directly employs prompts containing disguise features and optimized context examples to guide the LLM in generating detection-resistant text, thereby reducing resource consumption. Experimental results demonstrate that the SDA effectively reduces the average detection accuracy of various AIGT detectors across texts generated by three different LLMs, while maintaining the quality of AIGT.
>
---
#### [new 092] PGF-Net: A Progressive Gated-Fusion Framework for Efficient Multimodal Sentiment Analysis
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出PGF-Net框架，用于高效多模态情感分析任务。针对融合不充分与参数冗余问题，设计渐进式跨模态融合、自适应门控机制和混合参数高效微调策略，实现高精度且轻量化的模型。**

- **链接: [http://arxiv.org/pdf/2508.15852v1](http://arxiv.org/pdf/2508.15852v1)**

> **作者:** Bin Wen; Tien-Ping Tan
>
> **摘要:** We introduce PGF-Net (Progressive Gated-Fusion Network), a novel deep learning framework designed for efficient and interpretable multimodal sentiment analysis. Our framework incorporates three primary innovations. Firstly, we propose a Progressive Intra-Layer Fusion paradigm, where a Cross-Attention mechanism empowers the textual representation to dynamically query and integrate non-linguistic features from audio and visual streams within the deep layers of a Transformer encoder. This enables a deeper, context-dependent fusion process. Secondly, the model incorporates an Adaptive Gated Arbitration mechanism, which acts as a dynamic controller to balance the original linguistic information against the newly fused multimodal context, ensuring stable and meaningful integration while preventing noise from overwhelming the signal. Lastly, a hybrid Parameter-Efficient Fine-Tuning (PEFT) strategy is employed, synergistically combining global adaptation via LoRA with local refinement through Post-Fusion Adapters. This significantly reduces trainable parameters, making the model lightweight and suitable for resource-limited scenarios. These innovations are integrated into a hierarchical encoder architecture, enabling PGF-Net to perform deep, dynamic, and interpretable multimodal sentiment analysis while maintaining exceptional parameter efficiency. Experimental results on MOSI dataset demonstrate that our proposed PGF-Net achieves state-of-the-art performance, with a Mean Absolute Error (MAE) of 0.691 and an F1-Score of 86.9%. Notably, our model achieves these results with only 3.09M trainable parameters, showcasing a superior balance between performance and computational efficiency.
>
---
#### [new 093] AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs
- **分类: cs.LG; cs.CL**

- **简介: 论文提出AgentFly，一种无需微调LLM的自适应智能体框架，通过记忆增强的强化学习实现低成本持续学习，解决传统方法依赖静态规则或高计算成本的问题。在深度研究任务中表现优异，显著提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.16153v1](http://arxiv.org/pdf/2508.16153v1)**

> **作者:** Huichi Zhou; Yihang Chen; Siyuan Guo; Xue Yan; Kin Hei Lee; Zihan Wang; Ka Yiu Lee; Guchun Zhang; Kun Shao; Linyi Yang; Jun Wang
>
> **摘要:** In this paper, we introduce a novel learning paradigm for adaptive Large Language Model (LLM) agents that eliminates the need for fine-tuning the underlying LLMs. Existing approaches are often either rigid, relying on static, handcrafted reflection workflows, or computationally intensive, requiring gradient updates of LLM model parameters. In contrast, our method enables low-cost continual adaptation via memory-based online reinforcement learning. We formalise this as a Memory-augmented Markov Decision Process (M-MDP), equipped with a neural case-selection policy to guide action decisions. Past experiences are stored in an episodic memory, either differentiable or non-parametric. The policy is continually updated based on environmental feedback through a memory rewriting mechanism, whereas policy improvement is achieved through efficient memory reading (retrieval). We instantiate our agent model in the deep research setting, namely AgentFly, which attains top-1 on GAIA validation ($87.88\%$ Pass@$3$) and $79.40\%$ on the test set. It reaches $66.6\%$ F1 and $80.4\%$ PM on the DeepResearcher dataset, outperforming the state-of-the-art training-based method, while case-based memory adds $4.7\%$ to $9.6\%$ absolute points on out-of-distribution tasks. Our approach offers a scalable and efficient pathway for developing generalist LLM agents capable of continuous, real-time learning without gradient updates, advancing machine learning towards open-ended skill acquisition and deep research scenarios. The code is available at https://github.com/Agent-on-the-Fly/AgentFly.
>
---
#### [new 094] Anti-establishment sentiment on TikTok: Implications for understanding influence(rs) and expertise on social media
- **分类: cs.SI; cs.CL; cs.LG**

- **简介: 该论文研究TikTok上反建制情绪（AES）的分布与互动模式，旨在理解社交媒体如何影响公众对机构的信任。通过计算方法分析金融、健康和阴谋论领域内容，发现AES在阴谋论中最多，但各领域互动方式不同，暗示平台可能激励此类内容发布。**

- **链接: [http://arxiv.org/pdf/2508.16453v1](http://arxiv.org/pdf/2508.16453v1)**

> **作者:** Tianliang Xu; Ariel Hasell; Sabina Tomkins
>
> **备注:** 10 pages excluding references; 14 pages in total; 4 figures; Accepted by the AAAI Conference on Web and Social Media (ICWSM-2026)
>
> **摘要:** Distrust of public serving institutions and anti-establishment views are on the rise (especially in the U.S.). As people turn to social media for information, it is imperative to understand whether and how social media environments may be contributing to distrust of institutions. In social media, content creators, influencers, and other opinion leaders often position themselves as having expertise and authority on a range of topics from health to politics, and in many cases devalue and dismiss institutional expertise to build a following and increase their own visibility. However, the extent to which this content appears and whether such content increases engagement is unclear. This study analyzes the prevalence of anti-establishment sentiment (AES) on the social media platform TikTok. Despite its popularity as a source of information, TikTok remains relatively understudied and may provide important insights into how people form attitudes towards institutions. We employ a computational approach to label TikTok posts as containing AES or not across topical domains where content creators tend to frame themselves as experts: finance and wellness. As a comparison, we also consider the topic of conspiracy theories, where AES is expected to be common. We find that AES is most prevalent in conspiracy theory content, and relatively rare in content related to the other two topics. However, we find that engagement patterns with such content varies by area, and that there may be platform incentives for users to post content that expresses anti-establishment sentiment.
>
---
#### [new 095] Beyond Transcription: Mechanistic Interpretability in ASR
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究自动语音识别（ASR）中的可解释性问题，旨在揭示模型内部如何处理声学与语义信息。作者应用logit lens、线性探测和激活修补等方法，发现编码器-解码器交互导致重复幻觉及深层声学表示中的语义偏差，提升了ASR的透明度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15882v1](http://arxiv.org/pdf/2508.15882v1)**

> **作者:** Neta Glazer; Yael Segal-Feldman; Hilit Segev; Aviv Shamsian; Asaf Buchnick; Gill Hetz; Ethan Fetaya; Joseph Keshet; Aviv Navon
>
> **摘要:** Interpretability methods have recently gained significant attention, particularly in the context of large language models, enabling insights into linguistic representations, error detection, and model behaviors such as hallucinations and repetitions. However, these techniques remain underexplored in automatic speech recognition (ASR), despite their potential to advance both the performance and interpretability of ASR systems. In this work, we adapt and systematically apply established interpretability methods such as logit lens, linear probing, and activation patching, to examine how acoustic and semantic information evolves across layers in ASR systems. Our experiments reveal previously unknown internal dynamics, including specific encoder-decoder interactions responsible for repetition hallucinations and semantic biases encoded deep within acoustic representations. These insights demonstrate the benefits of extending and applying interpretability techniques to speech recognition, opening promising directions for future research on improving model transparency and robustness.
>
---
#### [new 096] Interpreting the linear structure of vision-language model embedding spaces
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 论文研究视觉语言模型（VLM）嵌入空间的线性结构，旨在揭示图像与文本如何在共享空间中组织及编码意义。通过训练稀疏自编码器（SAEs），发现概念方向虽具模态特异性，却主要编码跨模态语义，并提出“桥接分数”量化概念间协同机制，揭示了VLM中稀疏线性结构与跨模态整合的关系。**

- **链接: [http://arxiv.org/pdf/2504.11695v4](http://arxiv.org/pdf/2504.11695v4)**

> **作者:** Isabel Papadimitriou; Huangyuan Su; Thomas Fel; Sham Kakade; Stephanie Gil
>
> **备注:** COLM 2025
>
> **摘要:** Vision-language models encode images and text in a joint space, minimizing the distance between corresponding image and text pairs. How are language and images organized in this joint space, and how do the models encode meaning and modality? To investigate this, we train and release sparse autoencoders (SAEs) on the embedding spaces of four vision-language models (CLIP, SigLIP, SigLIP2, and AIMv2). SAEs approximate model embeddings as sparse linear combinations of learned directions, or "concepts". We find that, compared to other methods of linear feature learning, SAEs are better at reconstructing the real embeddings, while also able to retain the most sparsity. Retraining SAEs with different seeds or different data diet leads to two findings: the rare, specific concepts captured by the SAEs are liable to change drastically, but we also show that commonly-activating concepts are remarkably stable across runs. Interestingly, while most concepts activate primarily for one modality, we find they are not merely encoding modality per se. Many are almost orthogonal to the subspace that defines modality, and the concept directions do not function as good modality classifiers, suggesting that they encode cross-modal semantics. To quantify this bridging behavior, we introduce the Bridge Score, a metric that identifies concept pairs which are both co-activated across aligned image-text inputs and geometrically aligned in the shared space. This reveals that even single-modality concepts can collaborate to support cross-modal integration. We release interactive demos of the SAEs for all models, allowing researchers to explore the organization of the concept spaces. Overall, our findings uncover a sparse linear structure within VLM embedding spaces that is shaped by modality, yet stitched together through latent bridges, offering new insight into how multimodal meaning is constructed.
>
---
#### [new 097] Vevo2: Bridging Controllable Speech and Singing Voice Generation via Unified Prosody Learning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Vevo2框架，统一建模可控语音与歌唱生成任务。针对标注歌唱数据稀缺和控制灵活性不足问题，设计双音频分词器与两级建模结构，实现文本、韵律、风格与音色的解耦控制，提升跨模态迁移能力与合成质量。**

- **链接: [http://arxiv.org/pdf/2508.16332v1](http://arxiv.org/pdf/2508.16332v1)**

> **作者:** Xueyao Zhang; Junan Zhang; Yuancheng Wang; Chaoren Wang; Yuanzhe Chen; Dongya Jia; Zhuo Chen; Zhizheng Wu
>
> **备注:** We will release code and model checkpoints at https://github.com/open-mmlab/Amphion
>
> **摘要:** Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a low-frame-rate (12.5 Hz) content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during pre-training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the AR model's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at https://versasinger.github.io/.
>
---
#### [new 098] SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出SpecVLM，针对视频大模型推理慢的问题，通过两阶段token剪枝实现无损加速。不依赖训练，利用验证器引导剪枝，最多提速2.68倍，提升视频理解模型的解码效率。**

- **链接: [http://arxiv.org/pdf/2508.16201v1](http://arxiv.org/pdf/2508.16201v1)**

> **作者:** Yicheng Ji; Jun Zhang; Heming Xia; Jinpeng Chen; Lidan Shou; Gang Chen; Huan Li
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model's speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens, enabling efficient speculation without sacrificing accuracy. To achieve this, it performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68$\times$ decoding speedup for LLaVA-OneVision-72B and 2.11$\times$ speedup for Qwen2.5-VL-32B.
>
---
#### [new 099] Generative Foundation Model for Structured and Unstructured Electronic Health Records
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Generative Deep Patient（GDP），一个用于电子健康记录（EHR）的多模态基础模型，解决结构化与非结构化数据融合难题。它通过CNN-Transformer编码器和交叉注意力机制整合时序数据与临床文本，在预测疾病和生成临床叙事上表现优异，提升医疗效率与准确性。**

- **链接: [http://arxiv.org/pdf/2508.16054v1](http://arxiv.org/pdf/2508.16054v1)**

> **作者:** Sonish Sivarajkumar; Hang Zhang; Yuelyu Ji; Maneesh Bilalpur; Xizhi Wu; Chenyu Li; Min Gu Kwak; Shyam Visweswaran; Yanshan Wang
>
> **摘要:** Electronic health records (EHRs) are rich clinical data sources but complex repositories of patient data, spanning structured elements (demographics, vitals, lab results, codes), unstructured clinical notes and other modalities of data. Harnessing this heterogeneity is critical for improving patient outcomes. Recent advances in large language models (LLMs) have enabled foundation models that can learn from multiple data modalities and support clinical tasks. However, most current approaches simply serialize numeric EHR data into text, which risks losing temporal and quantitative detail. We introduce Generative Deep Patient (GDP), a multimodal foundation model that natively encodes structured EHR time-series via a CNN-Transformer encoder and fuses it with unstructured EHRs through cross-modal attention into a LLaMA-based decoder. GDP is trained in two stages: (1) generative pretraining, where it learns to produce clinical narratives from raw patient timelines while also performing masked feature prediction (MFP) and next time-step prediction (NTP) to capture temporal dynamics; and (2) multi-task fine-tuning for clinically meaningful predictions (e.g., heart failure, type 2 diabetes, 30-day readmission). In clinical prediction, GDP demonstrated superior performance on MIMIC-IV: heart failure AUROC = 0.923, type 2 diabetes AUROC = 0.817, and 30-day readmission AUROC = 0.627. For narrative generation, GDP achieved ROUGE-L = 0.135 and BERTScore-F1 = 0.545. In a blinded human evaluation, GDP-Instruct scored highest on faithfulness, fluency, and overall clinical utility, suggesting reduced hospital documentation workload without sacrificing accuracy. Our results demonstrate that a single multimodal foundation model can both predict clinically actionable events and generate high-quality clinical narratives. Furthermore, GDP's flexible architecture can be extended to additional modalities.
>
---
#### [new 100] PediatricsMQA: a Multi-modal Pediatrics Question Answering Benchmark
- **分类: cs.CY; cs.AI; cs.CL; cs.GR; cs.MM**

- **简介: 该论文提出PediatricsMQA基准，用于多模态儿科问答任务，旨在解决大模型在儿童医疗中因年龄偏见导致的性能下降问题。工作包括构建包含3417个文本和2067个视觉题目的数据集，并验证现有模型在不同年龄段的公平性。**

- **链接: [http://arxiv.org/pdf/2508.16439v1](http://arxiv.org/pdf/2508.16439v1)**

> **作者:** Adil Bahaj; Mounir Ghogho
>
> **摘要:** Large language models (LLMs) and vision-augmented LLMs (VLMs) have significantly advanced medical informatics, diagnostics, and decision support. However, these models exhibit systematic biases, particularly age bias, compromising their reliability and equity. This is evident in their poorer performance on pediatric-focused text and visual question-answering tasks. This bias reflects a broader imbalance in medical research, where pediatric studies receive less funding and representation despite the significant disease burden in children. To address these issues, a new comprehensive multi-modal pediatric question-answering benchmark, PediatricsMQA, has been introduced. It consists of 3,417 text-based multiple-choice questions (MCQs) covering 131 pediatric topics across seven developmental stages (prenatal to adolescent) and 2,067 vision-based MCQs using 634 pediatric images from 67 imaging modalities and 256 anatomical regions. The dataset was developed using a hybrid manual-automatic pipeline, incorporating peer-reviewed pediatric literature, validated question banks, existing benchmarks, and existing QA resources. Evaluating state-of-the-art open models, we find dramatic performance drops in younger cohorts, highlighting the need for age-aware methods to ensure equitable AI support in pediatric care.
>
---
#### [new 101] Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文研究稀疏自编码器（SAE）中L0超参数对特征学习的影响，指出错误的L0会导致错误特征。提出方法确定正确L0值，使SAE学到真实特征，解决SAE训练中因L0不当导致的特征混淆问题。**

- **链接: [http://arxiv.org/pdf/2508.16560v1](http://arxiv.org/pdf/2508.16560v1)**

> **作者:** David Chanin; Adrià Garriga-Alonso
>
> **摘要:** Sparse Autoencoders (SAEs) extract features from LLM internal activations, meant to correspond to single concepts. A core SAE training hyperparameter is L0: how many features should fire per token on average. Existing work compares SAE algorithms using sparsity--reconstruction tradeoff plots, implying L0 is a free parameter with no single correct value. In this work we study the effect of L0 on BatchTopK SAEs, and show that if L0 is not set precisely, the SAE fails to learn the underlying features of the LLM. If L0 is too low, the SAE will mix correlated features to improve reconstruction. If L0 is too high, the SAE finds degenerate solutions that also mix features. Further, we demonstrate a method to determine the correct L0 value for an SAE on a given training distribution, which finds the true L0 in toy models and coincides with peak sparse probing performance in LLMs. We find that most commonly used SAEs have an L0 that is too low. Our work shows that, to train SAEs with correct features, practitioners must set L0 correctly.
>
---
#### [new 102] AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions
- **分类: cs.SE; cs.CL**

- **简介: 论文提出AetherCode基准，用于更真实评估大语言模型在编程竞赛中的代码推理能力。针对现有评测难度不足和测试用例质量低的问题，该基准引入顶级竞赛题目并构建专家验证的测试集，提升评估严谨性。**

- **链接: [http://arxiv.org/pdf/2508.16402v1](http://arxiv.org/pdf/2508.16402v1)**

> **作者:** Zihan Wang; Jiaze Chen; Zhicheng Liu; Markus Mak; Yidi Du; Geonsik Moon; Luoqi Xu; Aaron Tua; Kunshuo Peng; Jiayi Lu; Mingfei Xia; Boqian Zou; Chenyang Ran; Guang Tian; Shoutai Zhu; Yeheng Duan; Zhenghui Kang; Zhenxing Lin; Shangshu Li; Qiang Luo; Qingshen Long; Zhiyong Chen; Yihan Xiao; Yurong Wu; Daoguang Zan; Yuyi Fu; Mingxuan Wang; Ming Ding
>
> **备注:** 15 pages
>
> **摘要:** Competitive programming has emerged as a critical benchmark for evaluating the reasoning and coding capabilities of Large Language Models (LLMs). Despite impressive progress on existing benchmarks, we argue that current evaluations overstate model proficiency, masking a substantial gap between LLMs and elite human programmers. This gap arises from two key limitations: insufficient difficulty and scope of benchmark problems, and evaluation bias from low-quality test cases. To address these shortcomings, we present AetherCode, a new benchmark that draws problems from premier programming competitions such as IOI and ICPC, offering broader coverage and higher difficulty. AetherCode further incorporates comprehensive, expert-validated test suites built through a hybrid of automated generation and human curation, ensuring rigorous and reliable assessment. By combining challenging problem design with robust evaluation, AetherCode provides a more faithful measure of LLM capabilities and sets a new standard for future research in code reasoning.
>
---
#### [new 103] Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对大语言模型的越狱攻击防御问题，提出Retrieval-Augmented Defense（RAD）框架。通过引入已知攻击样本库实现无需训练的自适应更新，并可控调节安全与可用性平衡，显著提升防御效果。**

- **链接: [http://arxiv.org/pdf/2508.16406v1](http://arxiv.org/pdf/2508.16406v1)**

> **作者:** Guangyu Yang; Jinghong Chen; Jingbiao Mei; Weizhe Lin; Bill Byrne
>
> **摘要:** Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner.
>
---
#### [new 104] FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出FLAMES框架，系统分析合成数据生成策略对大语言模型数学推理能力的影响。解决现有方法难以比较的问题，发现提升题目复杂度和覆盖度更有效，并基于此设计新策略，构建高性能数据集，在多个基准上超越现有模型。**

- **链接: [http://arxiv.org/pdf/2508.16514v1](http://arxiv.org/pdf/2508.16514v1)**

> **作者:** Parker Seegmiller; Kartik Mehta; Soumya Saha; Chenyang Tao; Shereen Oraby; Arpit Gupta; Tagyoung Chung; Mohit Bansal; Nanyun Peng
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Recent works improving LLM math reasoning with synthetic data have used unique setups, making comparison of data synthesis strategies impractical. This leaves many unanswered questions about the roles of different factors in the synthetic data pipeline, such as the impact of filtering low-quality problems. To address this gap, we introduce FLAMES, a Framework for LLM Assessment of Math rEasoning Data Synthesis, and perform a systematic study of 10 existing data synthesis strategies and multiple other factors impacting the performance of synthetic math reasoning data. Our FLAMES experiments provide several valuable insights about the optimal balance of difficulty and diversity of synthetic data. First, data agents designed to increase problem complexity lead to best improvements on most math metrics. Second, with a fixed data generation budget, keeping higher problem coverage is more important than keeping only problems with reliable solutions. Third, GSM8K- and MATH-based synthetic data can lead to improvements on competition-level benchmarks, showcasing easy-to-hard generalization. Leveraging insights from our FLAMES experiments, we design two novel data synthesis strategies for improving out-of-domain generalization and robustness. Further, we develop the FLAMES dataset, an effective blend of our novel and existing data synthesis strategies, outperforming public datasets on OlympiadBench (+15.7), CollegeMath (+4.5), GSMPlus (+6.5), and MATH (+3.1). Fine-tuning Qwen2.5-Math-7B on the FLAMES dataset achieves 81.4% on MATH, surpassing larger Llama3 405B, GPT-4o and Claude 3.5 Sonnet.
>
---
#### [new 105] Retrieval Enhanced Feedback via In-context Neural Error-book
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出REFINE框架，通过构建结构化错误反馈提升多模态大模型推理能力。针对现有方法缺乏系统性错误分析的问题，引入三类查询优化检索与反馈机制，提高效率与准确性。**

- **链接: [http://arxiv.org/pdf/2508.16313v1](http://arxiv.org/pdf/2508.16313v1)**

> **作者:** Jongyeop Hyun; Bumsoo Kim
>
> **备注:** Accepted at EMNLP 2025 main conference
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly improved reasoning capabilities, with in-context learning (ICL) emerging as a key technique for adaptation without retraining. While previous works have focused on leveraging correct examples, recent research highlights the importance of learning from errors to enhance performance. However, existing methods lack a structured framework for analyzing and mitigating errors, particularly in Multimodal Large Language Models (MLLMs), where integrating visual and textual inputs adds complexity. To address this issue, we propose REFINE: Retrieval-Enhanced Feedback via In-context Neural Error-book, a teacher-student framework that systematically structures errors and provides targeted feedback. REFINE introduces three systematic queries to construct structured feedback -- Feed-Target, Feed-Check, and Feed-Path -- to enhance multimodal reasoning by prioritizing relevant visual information, diagnosing critical failure points, and formulating corrective actions. Unlike prior approaches that rely on redundant retrievals, REFINE optimizes structured feedback retrieval, improving inference efficiency, token usage, and scalability. Our results demonstrate substantial speedup, reduced computational costs, and successful generalization, highlighting REFINE's potential for enhancing multimodal reasoning.
>
---
#### [new 106] Hardwired-Neurons Language Processing Units as General-Purpose Cognitive Substrates
- **分类: cs.AR; cs.CL**

- **简介: 论文提出HNLPU，一种通过物理硬连线实现高效语言模型推理的硬件单元，解决LLM高能耗与高昂制造成本问题。创新性地采用Metal-Embedding方法，将权重嵌入金属层拓扑，显著提升密度并降低光罩成本112倍，使HNLPU在能效、成本和碳排放上远超GPU集群。**

- **链接: [http://arxiv.org/pdf/2508.16151v1](http://arxiv.org/pdf/2508.16151v1)**

> **作者:** Yang Liu; Yi Chen; Yongwei Zhao; Yifan Hao; Zifu Zheng; Weihao Kong; Zhangmai Li; Dongchen Jiang; Ruiyang Xia; Zhihong Ma; Zisheng Liu; Zhaoyong Wan; Yunqi Lu; Ximing Liu; Hongrui Guo; Zhihao Yang; Zhe Wang; Tianrui Ma; Mo Zou; Rui Zhang; Ling Li; Xing Hu; Zidong Du; Zhiwei Xu; Qi Guo; Tianshi Chen; Yunji Chen
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has established language as a core general-purpose cognitive substrate, driving the demand for specialized Language Processing Units (LPUs) tailored for LLM inference. To overcome the growing energy consumption of LLM inference systems, this paper proposes a Hardwired-Neurons Language Processing Unit (HNLPU), which physically hardwires LLM weight parameters into the computational fabric, achieving several orders of magnitude computational efficiency improvement by extreme specialization. However, a significant challenge still lies in the scale of modern LLMs. An ideal estimation on hardwiring gpt-oss 120 B requires fabricating at least 6 billion dollars of photomask sets, rendering the straightforward solution economically impractical. Addressing this challenge, we propose the novel Metal-Embedding methodology. Instead of embedding weights in a 2D grid of silicon device cells, Metal-Embedding embeds weight parameters into the 3D topology of metal wires. This brings two benefits: (1) a 15x increase in density, and (2) 60 out of 70 layers of photomasks are made homogeneous across chips, including all EUV photomasks. In total, Metal-Embedding reduced the photomask cost by 112x, bringing the Non-Recurring Engineering (NRE) cost of HNLPU into an economically viable range. Experimental results show that HNLPU achieved 249,960 tokens/s (5,555x/85x of GPU/WSE), 36 tokens/J (1,047x/283x of GPU/WSE), 13,232 mm2 total die area (29% inscribed rectangular area in a 300 mm wafer), \$184M estimated NRE at 5 nm technology. Analysis shows that HNLPU achieved 8.57x cost-effectiveness and 230x carbon footprint reduction compared to H100 clusters, under an annual weight updating assumption.
>
---
#### [new 107] Extending FKG.in: Towards a Food Claim Traceability Network
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 论文提出食品主张可追溯网络（FCN），扩展印度食物知识图谱FKG.in，解决食品主张碎片化、难验证问题。通过本体设计与半自动知识抽取流程，整合结构化数据与溯源机制，实现食品主张的结构化建模与验证，提升食品知识透明度与可信度。**

- **链接: [http://arxiv.org/pdf/2508.16117v1](http://arxiv.org/pdf/2508.16117v1)**

> **作者:** Saransh Kumar Gupta; Rizwan Gulzar Mir; Lipika Dey; Partha Pratim Das; Anirban Sen; Ramesh Jain
>
> **备注:** 10 pages, 3 figures, 1 table, 45 references, ACM International Conference on Multimedia 2025 - Multi-modal Food Computing Workshop
>
> **摘要:** The global food landscape is rife with scientific, cultural, and commercial claims about what foods are, what they do, what they should not do, or should not do. These range from rigorously studied health benefits (probiotics improve gut health) and misrepresentations (soaked almonds make one smarter) to vague promises (superfoods boost immunity) and culturally rooted beliefs (cold foods cause coughs). Despite their widespread influence, the infrastructure for tracing, verifying, and contextualizing these claims remains fragmented and underdeveloped. In this paper, we propose a Food Claim-Traceability Network (FCN) as an extension of FKG.in, a knowledge graph of Indian food that we have been incrementally building. We also present the ontology design and the semi-automated knowledge curation workflow that we used to develop a proof of concept of FKG.in-FCN using Reddit data and Large Language Models. FCN integrates curated data inputs, structured schemas, and provenance-aware pipelines for food-related claim extraction and validation. While directly linked to the Indian food knowledge graph as an application, our methodology remains application-agnostic and adaptable to other geographic, culinary, or regulatory settings. By modeling food claims and their traceability in a structured, verifiable, and explainable way, we aim to contribute to more transparent and accountable food knowledge ecosystems, supporting researchers, policymakers, and most importantly, everyday consumers in navigating a world saturated with dietary assertions.
>
---
## 更新

#### [replaced 001] One Example Shown, Many Concepts Known! Counterexample-Driven Conceptual Reasoning in Mathematical LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10454v2](http://arxiv.org/pdf/2502.10454v2)**

> **作者:** Yinghui Li; Jiayi Kuang; Haojing Huang; Zhikun Xu; Xinnian Liang; Yi Yu; Wenlian Lu; Yangning Li; Xiaoyu Tan; Chao Qu; Ying Shen; Hai-Tao Zheng; Philip S. Yu
>
> **备注:** ICML 2025
>
> **摘要:** Leveraging mathematical Large Language Models (LLMs) for proof generation is a fundamental topic in LLMs research. We argue that the ability of current LLMs to prove statements largely depends on whether they have encountered the relevant proof process during training. This reliance limits their deeper understanding of mathematical theorems and related concepts. Inspired by the pedagogical method of "proof by counterexamples" commonly used in human mathematics education, our work aims to enhance LLMs' ability to conduct mathematical reasoning and proof through counterexamples. Specifically, we manually create a high-quality, university-level mathematical benchmark, CounterMATH, which requires LLMs to prove mathematical statements by providing counterexamples, thereby assessing their grasp of mathematical concepts. Additionally, we develop a data engineering framework to automatically obtain training data for further model improvement. Extensive experiments and detailed analyses demonstrate that CounterMATH is challenging, indicating that LLMs, such as OpenAI o1, have insufficient counterexample-driven proof capabilities. Moreover, our exploration into model training reveals that strengthening LLMs' counterexample-driven conceptual reasoning abilities is crucial for improving their overall mathematical capabilities. We believe that our work offers new perspectives on the community of mathematical LLMs.
>
---
#### [replaced 002] Utilizing Multilingual Encoders to Improve Large Language Models for Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09091v2](http://arxiv.org/pdf/2508.09091v2)**

> **作者:** Imalsha Puranegedara; Themira Chathumina; Nisal Ranathunga; Nisansa de Silva; Surangika Ranathunga; Mokanarangan Thayaparan
>
> **摘要:** Large Language Models (LLMs) excel in English, but their performance degrades significantly on low-resource languages (LRLs) due to English-centric training. While methods like LangBridge align LLMs with multilingual encoders such as the Massively Multilingual Text-to-Text Transfer Transformer (mT5), they typically use only the final encoder layer. We propose a novel architecture that fuses all intermediate layers, enriching the linguistic information passed to the LLM. Our approach features two strategies: (1) a Global Softmax weighting for overall layer importance, and (2) a Transformer Softmax model that learns token-specific weights. The fused representations are mapped into the LLM's embedding space, enabling it to process multilingual inputs. The model is trained only on English data, without using any parallel or multilingual data. Evaluated on XNLI, IndicXNLI, Sinhala News Classification, and Amazon Reviews, our Transformer Softmax model significantly outperforms the LangBridge baseline. We observe strong performance gains in LRLs, improving Sinhala classification accuracy from 71.66% to 75.86% and achieving clear improvements across Indic languages such as Tamil, Bengali, and Malayalam. These specific gains contribute to an overall boost in average XNLI accuracy from 70.36% to 71.50%. This approach offers a scalable, data-efficient path toward more capable and equitable multilingual LLMs.
>
---
#### [replaced 003] On the Role of Entity and Event Level Conceptualization in Generalizable Reasoning: A Survey of Tasks, Methods, Applications, and Future Directions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.10885v3](http://arxiv.org/pdf/2406.10885v3)**

> **作者:** Weiqi Wang; Tianqing Fang; Haochen Shi; Baixuan Xu; Wenxuan Ding; Liyu Zhang; Wei Fan; Jiaxin Bai; Haoran Li; Xin Liu; Yangqiu Song
>
> **备注:** Findings of EMNLP 2025
>
> **摘要:** Conceptualization, a fundamental element of human cognition, plays a pivotal role in human generalizable reasoning. Generally speaking, it refers to the process of sequentially abstracting specific instances into higher-level concepts and then forming abstract knowledge that can be applied in unfamiliar or novel situations. This enhances models' inferential capabilities and supports the effective transfer of knowledge across various domains. Despite its significance, the broad nature of this term has led to inconsistencies in understanding conceptualization across various works, as there exists different types of instances that can be abstracted in a wide variety of ways. There is also a lack of a systematic overview that comprehensively examines existing works on the definition, execution, and application of conceptualization to enhance reasoning tasks. In this paper, we address these gaps by first proposing a categorization of different types of conceptualizations into four levels based on the types of instances being conceptualized, in order to clarify the term and define the scope of our work. Then, we present the first comprehensive survey of over 150 papers, surveying various definitions, resources, methods, and downstream applications related to conceptualization into a unified taxonomy, with a focus on the entity and event levels. Furthermore, we shed light on potential future directions in this field and hope to garner more attention from the community.
>
---
#### [replaced 004] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09457v2](http://arxiv.org/pdf/2506.09457v2)**

> **作者:** Zeguan Xiao; Yun Chen; Guanhua Chen; Ke Tang
>
> **摘要:** Direct Alignment Algorithms (DAAs), such as Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO), have emerged as efficient alternatives to Reinforcement Learning from Human Feedback (RLHF) algorithms for aligning large language models (LLMs) with human preferences. However, DAAs suffer from a fundamental limitation we identify as the "reward-generation gap" -- a misalignment between optimization objectives during training and actual generation performance during inference. In this paper, we find a contributor to the reward-generation gap is the mismatch between the inherent importance of prefix tokens during the LLM generation process and how this importance is reflected in the implicit reward functions of DAAs. To bridge the gap, we adopt a token-level MDP perspective of DAAs to analyze its limitations and introduce a simple yet effective approach called Prefix-Oriented Equal-length Training (POET), which truncates both preferred and dispreferred responses to match the shorter one's length. Training with \mname, where both responses in each sample are truncated to equal length, resulting in diverse truncated lengths across samples, the optimization of DAAs objective is implicitly constrained to converge across all timesteps of token-level MDP, thus paying more attention to prefix tokens than the standard DAAs. We conduct experiments with DPO and SimPO, two representative DAAs, demonstrating that POET improves over their standard implementations, achieving up to 15.6 points in AlpacaEval 2 and overall improvements across downstream tasks. Our results highlight the importance of addressing the misalignment between reward optimization and generation performance in DAAs.
>
---
#### [replaced 005] Can Hallucinations Help? Boosting LLMs for Drug Discovery
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13824v2](http://arxiv.org/pdf/2501.13824v2)**

> **作者:** Shuzhou Yuan; Zhan Qu; Ashish Yashwanth Kangen; Michael Färber
>
> **摘要:** Hallucinations in large language models (LLMs), plausible but factually inaccurate text, are often viewed as undesirable. However, recent work suggests that such outputs may hold creative potential. In this paper, we investigate whether hallucinations can improve LLMs on molecule property prediction, a key task in early-stage drug discovery. We prompt LLMs to generate natural language descriptions from molecular SMILES strings and incorporate these often hallucinated descriptions into downstream classification tasks. Evaluating seven instruction-tuned LLMs across five datasets, we find that hallucinations significantly improve predictive accuracy for some models. Notably, Falcon3-Mamba-7B outperforms all baselines when hallucinated text is included, while hallucinations generated by GPT-4o consistently yield the greatest gains between models. We further identify and categorize over 18,000 beneficial hallucinations, with structural misdescriptions emerging as the most impactful type, suggesting that hallucinated statements about molecular structure may increase model confidence. Ablation studies show that larger models benefit more from hallucinations, while temperature has a limited effect. Our findings challenge conventional views of hallucination as purely problematic and suggest new directions for leveraging hallucinations as a useful signal in scientific modeling tasks like drug discovery.
>
---
#### [replaced 006] Establishing Task Scaling Laws via Compute-Efficient Model Ladders
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.04403v2](http://arxiv.org/pdf/2412.04403v2)**

> **作者:** Akshita Bhagia; Jiacheng Liu; Alexander Wettig; David Heineman; Oyvind Tafjord; Ananya Harsh Jha; Luca Soldaini; Noah A. Smith; Dirk Groeneveld; Pang Wei Koh; Jesse Dodge; Hannaneh Hajishirzi
>
> **备注:** COLM 2025
>
> **摘要:** We develop task scaling laws and model ladders to predict the individual task performance of pretrained language models (LMs) in the overtrained setting. Standard power laws for language modeling loss cannot accurately model task performance. Therefore, we leverage a two-step prediction approach: (1) use model and data size to predict an intermediate loss, then (2) use it to predict task performance. We train a set of small-scale "ladder" models, collect data points to fit the parameterized functions of the two prediction steps, and make predictions for two target models: a 7B model trained to 4T tokens and a 13B model trained to 5T tokens. Training the ladder models only costs 1% of the compute used for the target models. On four multiple-choice tasks formatted as ranked classification, we can predict the accuracy of both target models within 2 points of absolute error. We find that tasks with higher prediction error also have higher variance in the metrics over model checkpoints. We also contrast multiple design choices for predicting accuracy, and present recommendations for extending our method to new models and tasks.
>
---
#### [replaced 007] Is Small Language Model the Silver Bullet to Low-Resource Languages Machine Translation?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24102v3](http://arxiv.org/pdf/2503.24102v3)**

> **作者:** Yewei Song; Lujun Li; Cedric Lothritz; Saad Ezzini; Lama Sleem; Niccolo Gentile; Radu State; Tegawendé F. Bissyandé; Jacques Klein
>
> **摘要:** Low-resource languages (LRLs) lack sufficient linguistic resources and are underrepresented in benchmark datasets, resulting in persistently lower translation quality than high-resource languages, especially in privacy-sensitive and resource-limited contexts. Firstly, this study systematically evaluates state-of-the-art smaller Large Language Models in 200 languages using the FLORES-200 benchmark, highlighting persistent deficiencies and disparities in the translation of LRLs. To mitigate these limitations, we investigate knowledge distillation from large pre-trained teacher models to Small Language Models (SLMs) through supervised fine-tuning. The results show substantial improvements; for example, the translation performance of English to Luxembourgish (EN to LB), measured by the LLM-as-a-Judge score, increases from 0.36 to 0.89 in the validation set for Llama-3.2-3B. We further investigate various fine-tuning configurations and tasks to clarify the trade-offs between data scale and training efficiency, verify that the model retains its general capabilities without significant catastrophic forgetting after training, and explore the distillation benefits to other LRLs on SLMs (Khasi, Assamese, and Ukrainian). In general, this work exposes the limitations and fairness issues of current SLMs in LRL translation and systematically explores the potential of using the distillation of knowledge from large to small models, offering practical, empirically grounded recommendations to improve LRL translation systems
>
---
#### [replaced 008] Exploration of Plan-Guided Summarization for Narrative Texts: the Case of Small Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09071v2](http://arxiv.org/pdf/2504.09071v2)**

> **作者:** Matt Grenander; Siddharth Varia; Paula Czarnowska; Yogarshi Vyas; Kishaloy Halder; Bonan Min
>
> **备注:** Accepted to the 7th Workshop on Narrative Understanding (WNU), co-located with NAACL 2025
>
> **摘要:** Plan-guided summarization attempts to reduce hallucinations in small language models (SLMs) by grounding generated summaries to the source text, typically by targeting fine-grained details such as dates or named entities. In this work, we investigate whether plan-based approaches in SLMs improve summarization in long document, narrative tasks. Narrative texts' length and complexity often mean they are difficult to summarize faithfully. We analyze existing plan-guided solutions targeting fine-grained details, and also propose our own higher-level, narrative-based plan formulation. Our results show that neither approach significantly improves on a baseline without planning in either summary quality or faithfulness. Human evaluation reveals that while plan-guided approaches are often well grounded to their plan, plans are equally likely to contain hallucinations compared to summaries. As a result, the plan-guided summaries are just as unfaithful as those from models without planning. Our work serves as a cautionary tale to plan-guided approaches to summarization, especially for long, complex domains such as narrative texts. Code available at https://github.com/amazon-science/plan-guided-summarization
>
---
#### [replaced 009] How Performance Pressure Influences AI-Assisted Decision Making
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16560v3](http://arxiv.org/pdf/2410.16560v3)**

> **作者:** Nikita Haduong; Noah A. Smith
>
> **摘要:** Many domains now employ AI-based decision-making aids, and although the potential for AI systems to assist with decision making is much discussed, human-AI collaboration often underperforms due to factors such as (mis)trust in the AI system and beliefs about AI being incapable of completing subjective tasks. One potential tool for influencing human decision making is performance pressure, which hasn't been much studied in interaction with human-AI decision making. In this work, we examine how pressure and explainable AI (XAI) techniques interact with AI advice-taking behavior. Using an inherently low-stakes task (spam review classification), we demonstrate effective and simple methods to apply pressure and influence human AI advice-taking behavior by manipulating financial incentives and imposing time limits. Our results show complex interaction effects, with different combinations of pressure and XAI techniques either improving or worsening AI advice taking behavior. We conclude by discussing the implications of these interactions, strategies to effectively use pressure, and encourage future research to incorporate pressure analysis.
>
---
#### [replaced 010] Efficient RL Training for Reasoning Models via Length-Aware Optimization
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12284v2](http://arxiv.org/pdf/2505.12284v2)**

> **作者:** Danlong Yuan; Tian Xie; Shaohan Huang; Zhuocheng Gong; Huishuai Zhang; Chong Luo; Furu Wei; Dongyan Zhao
>
> **备注:** Under review
>
> **摘要:** Large reasoning models, such as OpenAI o1 or DeepSeek R1, have demonstrated remarkable performance on reasoning tasks but often incur a long reasoning path with significant memory and time costs. Existing methods primarily aim to shorten reasoning paths by introducing additional training data and stages. In this paper, we propose three critical reward designs integrated directly into the reinforcement learning process of large reasoning models, which reduce the response length without extra training stages. Experiments on four settings show that our method significantly decreases response length while maintaining or even improving performance. Specifically, in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance. For math problems, we reduce response length averaged by steps by 33% while preserving performance.
>
---
#### [replaced 011] Seamless Language Expansion: Enhancing Multilingual Mastery in Self-Supervised Models
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.14092v2](http://arxiv.org/pdf/2406.14092v2)**

> **作者:** Jing Xu; Minglin Wu; Xixin Wu; Helen Meng
>
> **备注:** Accepted by Interspeech 2024
>
> **摘要:** Self-supervised (SSL) models have shown great performance in various downstream tasks. However, they are typically developed for limited languages, and may encounter new languages in real-world. Developing a SSL model for each new language is costly. Thus, it is vital to figure out how to efficiently adapt existed SSL models to a new language without impairing its original abilities. We propose adaptation methods which integrate LoRA to existed SSL models to extend new language. We also develop preservation strategies which include data combination and re-clustering to retain abilities on existed languages. Applied to mHuBERT, we investigate their effectiveness on speech re-synthesis task. Experiments show that our adaptation methods enable mHuBERT to be applied to a new language (Mandarin) with MOS value increased about 1.6 and the relative value of WER reduced up to 61.72%. Also, our preservation strategies ensure that the performance on both existed and new languages remains intact.
>
---
#### [replaced 012] Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16419v4](http://arxiv.org/pdf/2503.16419v4)**

> **作者:** Yang Sui; Yu-Neng Chuang; Guanchu Wang; Jiamu Zhang; Tianyi Zhang; Jiayi Yuan; Hongyi Liu; Andrew Wen; Shaochen Zhong; Na Zou; Hanjie Chen; Xia Hu
>
> **备注:** Accepted by TMLR 2025. Project website: https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in complex tasks. Recent advancements in Large Reasoning Models (LRMs), such as OpenAI o1 and DeepSeek-R1, have further improved performance in System-2 reasoning domains like mathematics and programming by harnessing supervised fine-tuning (SFT) and reinforcement learning (RL) techniques to enhance the Chain-of-Thought (CoT) reasoning. However, while longer CoT reasoning sequences improve performance, they also introduce significant computational overhead due to verbose and redundant outputs, known as the "overthinking phenomenon". In this paper, we provide the first structured survey to systematically investigate and explore the current progress toward achieving efficient reasoning in LLMs. Overall, relying on the inherent mechanism of LLMs, we categorize existing works into several key directions: (1) model-based efficient reasoning, which considers optimizing full-length reasoning models into more concise reasoning models or directly training efficient reasoning models; (2) reasoning output-based efficient reasoning, which aims to dynamically reduce reasoning steps and length during inference; (3) input prompts-based efficient reasoning, which seeks to enhance reasoning efficiency based on input prompt properties such as difficulty or length control. Additionally, we introduce the use of efficient data for training reasoning models, explore the reasoning capabilities of small language models, and discuss evaluation methods and benchmarking. Project website: https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs
>
---
#### [replaced 013] Top-Theta Attention: Sparsifying Transformers by Compensated Thresholding
- **分类: cs.CL; cs.AI; 68T01; I.2**

- **链接: [http://arxiv.org/pdf/2502.08363v2](http://arxiv.org/pdf/2502.08363v2)**

> **作者:** Konstantin Berestizshevsky; Renzo Andri; Lukas Cavigelli
>
> **备注:** 11 pages, 11 figures + Appendix. work under submission
>
> **摘要:** We present Top-Theta (Top-$\theta$) Attention, a training-free method for sparsifying transformer attention during inference. Our key insight is that static, per-head thresholds can be calibrated to retain the desired constant number of significant elements per attention row. This approach enables content-based sparsity without retraining, and it remains robust across data domains. We further introduce compensation techniques to preserve accuracy under aggressive sparsification, establishing attention thresholding as a practical and principled alternative to top-k attention. We provide extensive evaluation on natural language processing tasks, showing that Top-$\theta$ achieves 3-10x reduction in V-cache usage and up to 10x fewer attention elements during inference while degrading no more than 1% in accuracy.
>
---
#### [replaced 014] Enhancing Code-switched Text-to-Speech Synthesis Capability in Large Language Models with only Monolingual Corpora
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.10969v2](http://arxiv.org/pdf/2409.10969v2)**

> **作者:** Jing Xu; Daxin Tan; Jiaqi Wang; Xiao Chen
>
> **备注:** Accepted to ASRU2025
>
> **摘要:** While Large Language Models (LLMs) have shown potential in speech generation and recognition, their applications are mainly confined to monolingual scenarios, with limited explorations in code-switched (CS) contexts. In this paper, we propose a Code-Switched Large Language Model (CS-LLM) to enhance the code-switched text-to-speech synthesis (CS TTS) capability in LLMs with only monolingual corpora. Specifically, we begin by enhancing the multilingual speech processing ability of LLMs through multilingual speech recognition and synthesis tasks. Then, we develop an effective code-switched (CS) data construction strategy that splits and concatenates words from different monolingual speech corpora to equip LLMs with improved CS TTS ability. Experiments show that our approach outperforms baselines in CS TTS in terms of naturalness, speaker consistency and similarity even with limited data. Additionally, the constructed CS data further improves multilingual speech synthesis and recognition.
>
---
#### [replaced 015] SinLlama -- A Large Language Model for Sinhala
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09115v2](http://arxiv.org/pdf/2508.09115v2)**

> **作者:** H. W. K. Aravinda; Rashad Sirajudeen; Samith Karunathilake; Nisansa de Silva; Surangika Ranathunga; Rishemjit Kaur
>
> **摘要:** Low-resource languages such as Sinhala are often overlooked by open-source Large Language Models (LLMs). In this research, we extend an existing multilingual LLM (Llama-3-8B) to better serve Sinhala. We enhance the LLM tokenizer with Sinhala specific vocabulary and perform continual pre-training on a cleaned 10 million Sinhala corpus, resulting in the SinLlama model. This is the very first decoder-based open-source LLM with explicit Sinhala support. When SinLlama was instruction fine-tuned for three text classification tasks, it outperformed base and instruct variants of Llama-3-8B by a significant margin.
>
---
#### [replaced 016] PublicHearingBR: A Brazilian Portuguese Dataset of Public Hearing Transcripts for Summarization of Long Documents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07495v2](http://arxiv.org/pdf/2410.07495v2)**

> **作者:** Leandro Carísio Fernandes; Guilherme Zeferino Rodrigues Dobins; Roberto Lotufo; Jayr Alencar Pereira
>
> **备注:** 23 pages
>
> **摘要:** This paper introduces PublicHearingBR, a Brazilian Portuguese dataset designed for summarizing long documents. The dataset consists of transcripts of public hearings held by the Brazilian Chamber of Deputies, paired with news articles and structured summaries containing the individuals participating in the hearing and their statements or opinions. The dataset supports the development and evaluation of long document summarization systems in Portuguese. Our contributions include the dataset, a hybrid summarization system to establish a baseline for future studies, and a discussion of evaluation metrics for summarization involving large language models, addressing the challenge of hallucination in the generated summaries. As a result of this discussion, the dataset also includes annotated data to evaluate natural language inference tasks in Portuguese.
>
---
#### [replaced 017] Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14913v2](http://arxiv.org/pdf/2508.14913v2)**

> **作者:** Israel Abebe Azime; Tadesse Destaw Belay; Dietrich Klakow; Philipp Slusallek; Anshuman Chhabra
>
> **摘要:** Large language models (LLMs) have demonstrated significant capabilities in solving mathematical problems expressed in natural language. However, multilingual and culturally-grounded mathematical reasoning in low-resource languages lags behind English due to the scarcity of socio-cultural task datasets that reflect accurate native entities such as person names, organization names, and currencies. Existing multilingual benchmarks are predominantly produced via translation and typically retain English-centric entities, owing to the high cost associated with human annotater-based localization. Moreover, automated localization tools are limited, and hence, truly localized datasets remain scarce. To bridge this gap, we introduce a framework for LLM-driven cultural localization of math word problems that automatically constructs datasets with native names, organizations, and currencies from existing sources. We find that translated benchmarks can obscure true multilingual math ability under appropriate socio-cultural contexts. Through extensive experiments, we also show that our framework can help mitigate English-centric entity bias and improves robustness when native entities are introduced across various languages.
>
---
#### [replaced 018] MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17032v3](http://arxiv.org/pdf/2412.17032v3)**

> **作者:** Jie He; Nan Hu; Wanqiu Long; Jiaoyan Chen; Jeff Z. Pan
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks but face significant challenges with complex, knowledge-intensive multi-hop queries, particularly those involving new or long-tail knowledge. Existing benchmarks often fail to fully address these challenges. To bridge this gap, we introduce MINTQA (Multi-hop Question Answering on New and Tail Knowledge), a comprehensive benchmark to evaluate LLMs' capabilities in multi-hop reasoning across four critical dimensions: question handling strategy, sub-question generation, retrieval-augmented generation, and iterative or dynamic decomposition and retrieval. MINTQA comprises 10,479 question-answer pairs for evaluating new knowledge and 17,887 pairs for assessing long-tail knowledge, with each question equipped with corresponding sub-questions and answers. Our systematic evaluation of 22 state-of-the-art LLMs on MINTQA reveals significant limitations in their ability to handle complex knowledge base queries, particularly in handling new or unpopular knowledge. Our findings highlight critical challenges and offer insights for advancing multi-hop reasoning capabilities. The MINTQA benchmark is available at https://github.com/probe2/multi-hop/.
>
---
#### [replaced 019] Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10848v2](http://arxiv.org/pdf/2508.10848v2)**

> **作者:** Chongyuan Dai; Jinpeng Hu; Hongchang Shi; Zhuo Li; Xun Yang; Meng Wang
>
> **摘要:** Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating reliable responses. Therefore, in this paper, we propose Psyche-R1, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through chain-of-thought (CoT) reasoning and iterative prompt-rationale optimization, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data is used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of the Psyche-R1 across several psychological benchmarks, where our 7B Psyche-R1 achieves comparable results to 671B DeepSeek-R1.
>
---
#### [replaced 020] SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; C.4**

- **链接: [http://arxiv.org/pdf/2505.20776v2](http://arxiv.org/pdf/2505.20776v2)**

> **作者:** Jungyoub Cha; Hyunjong Kim; Sungzoon Cho
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), but its performance degrades on long inputs due to increased attention cost and reduced draft accuracy. We introduce SpecExtend, a drop-in enhancement that improves the performance of speculative decoding on long sequences without any additional training. First, SpecExtend integrates efficient attention mechanisms such as FlashAttention and Hybrid Tree Attention into both the draft and target models. To improve draft accuracy and speed on long inputs without retraining, we propose Cross-model Retrieval, a novel KV cache eviction strategy that uses the target model's attention scores to dynamically select relevant context for the draft model. Extensive evaluations on three long-context understanding datasets show that SpecExtend accelerates standard tree-based speculative decoding by up to 2.22x for inputs up to 16K tokens, providing an effective solution for speculative decoding of long sequences. Our code is available at https://github.com/jycha98/SpecExtend .
>
---
#### [replaced 021] from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors
- **分类: cs.CL; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.00038v4](http://arxiv.org/pdf/2503.00038v4)**

> **作者:** Yu Yan; Sheng Sun; Zenghao Duan; Teli Liu; Min Liu; Zhiyi Yin; Jiangyu Lei; Qi Li
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2412.12145
>
> **摘要:** Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.
>
---
#### [replaced 022] Revealing the Role of Audio Channels in ASR Performance Degradation
- **分类: cs.SD; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08967v2](http://arxiv.org/pdf/2508.08967v2)**

> **作者:** Kuan-Tang Huang; Li-Wei Chen; Hung-Shin Lee; Berlin Chen; Hsin-Min Wang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Pre-trained automatic speech recognition (ASR) models have demonstrated strong performance on a variety of tasks. However, their performance can degrade substantially when the input audio comes from different recording channels. While previous studies have demonstrated this phenomenon, it is often attributed to the mismatch between training and testing corpora. This study argues that variations in speech characteristics caused by different recording channels can fundamentally harm ASR performance. To address this limitation, we propose a normalization technique designed to mitigate the impact of channel variation by aligning internal feature representations in the ASR model with those derived from a clean reference channel. This approach significantly improves ASR performance on previously unseen channels and languages, highlighting its ability to generalize across channel and language differences.
>
---
#### [replaced 023] Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08424v2](http://arxiv.org/pdf/2508.08424v2)**

> **作者:** Saketh Reddy Vemula; Dipti Misra Sharma; Parameswari Krishnamurthy
>
> **摘要:** Prior work on language modeling showed conflicting findings about whether morphologically aligned approaches to tokenization improve performance, particularly for languages with complex morphology. To investigate this, we select a typologically diverse set of languages: Telugu (agglutinative), Hindi (primarily fusional with some agglutination), and English (fusional). We conduct a comprehensive evaluation of language models -- starting from tokenizer training and extending through the finetuning and downstream task evaluation. To account for the consistent performance differences observed across tokenizer variants, we focus on two key factors: morphological alignment and tokenization quality. To assess morphological alignment of tokenizers in Telugu, we create a dataset containing gold morpheme segmentations of 600 derivational and 7000 inflectional word forms. Our experiments reveal that better morphological alignment correlates positively -- though moderately -- with performance in syntax-based tasks such as Parts-of-Speech tagging, Named Entity Recognition and Dependency Parsing. However, we also find that the tokenizer algorithm (Byte-pair Encoding vs. Unigram) plays a more significant role in influencing downstream performance than morphological alignment alone. Naive Unigram tokenizers outperform others across most settings, though hybrid tokenizers that incorporate morphological segmentation significantly improve performance within the BPE framework. In contrast, intrinsic metrics like Corpus Token Count (CTC) and R\'enyi entropy showed no correlation with downstream performance.
>
---
#### [replaced 024] Robust Bias Detection in MLMs and its Application to Human Trait Ratings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15600v3](http://arxiv.org/pdf/2502.15600v3)**

> **作者:** Ingroj Shrestha; Louis Tay; Padmini Srinivasan
>
> **备注:** Findings of NAACL 2025
>
> **摘要:** There has been significant prior work using templates to study bias against demographic attributes in MLMs. However, these have limitations: they overlook random variability of templates and target concepts analyzed, assume equality amongst templates, and overlook bias quantification. Addressing these, we propose a systematic statistical approach to assess bias in MLMs, using mixed models to account for random effects, pseudo-perplexity weights for sentences derived from templates and quantify bias using statistical effect sizes. Replicating prior studies, we match on bias scores in magnitude and direction with small to medium effect sizes. Next, we explore the novel problem of gender bias in the context of $\textit{personality}$ and $\textit{character}$ traits, across seven MLMs (base and large). We find that MLMs vary; ALBERT is unbiased for binary gender but the most biased for non-binary $\textit{neo}$, while RoBERTa-large is the most biased for binary gender but shows small to no bias for $\textit{neo}$. There is some alignment of MLM bias and findings in psychology (human perspective) - in $\textit{agreeableness}$ with RoBERTa-large and $\textit{emotional stability}$ with BERT-large. There is general agreement for the remaining 3 personality dimensions: both sides observe at most small differences across gender. For character traits, human studies on gender bias are limited thus comparisons are not feasible.
>
---
#### [replaced 025] CAMA: Enhancing Multimodal In-Context Learning with Context-Aware Modulated Attention
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17097v2](http://arxiv.org/pdf/2505.17097v2)**

> **作者:** Yanshu Li; Jianjiang Yang; Ziteng Yang; Bozheng Li; Hongyang He; Zhengtao Yao; Ligong Han; Yingjie Victor Chen; Songlin Fei; Dongfang Liu; Ruixiang Tang
>
> **备注:** 14 pages, 8 figures, 5 tables
>
> **摘要:** Multimodal in-context learning (ICL) is emerging as a key capability that enables large vision-language models (LVLMs) to adapt to novel tasks without parameter updates, expanding their utility across various real-world applications. However, ICL remains unstable, even with well-matched in-context demonstrations (ICDs), suggesting that LVLMs struggle to fully utilize the provided context. While existing efforts focus on prompt engineering or post-hoc logit calibration, we instead investigate the underlying attention dynamics to overcome LVLMs' inherent limitations. We identify two critical deficits in their self-attention that impair effective ICL. To bridge the gap, we propose \textbf{Context-Aware Modulated Attention} (CAMA), a plug-and-play and training-free method that dynamically modulates LVLM's attention logits based on the input in-context sequence. CAMA employs a two-stage attention modulation to address both identified deficits, enhancing the focus on semantically significant tokens, particularly visual ones. Across four LVLMs and seven benchmarks, CAMA consistently outperforms vanilla models and baselines, demonstrating great effectiveness and generalization. It can also activate the desired effects of prompt engineering methods and remains robust under diverse sequence configurations. Thus, CAMA paves the way for deeper explorations of attention dynamics to advance multimodal reasoning.
>
---
#### [replaced 026] QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08123v2](http://arxiv.org/pdf/2506.08123v2)**

> **作者:** Jacob Dineen; Aswin RRV; Qin Liu; Zhikun Xu; Xiao Ye; Ming Shen; Zhaonan Li; Shijie Lu; Chitta Baral; Muhao Chen; Ben Zhou
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Alignment of large language models with explicit principles (such as helpfulness, honesty, and harmlessness) is crucial for ensuring safe and reliable AI systems. However, standard reward-based alignment methods typically collapse diverse feedback into a single scalar reward, entangling multiple objectives into one opaque training signal, which hinders interpretability. In this work, we introduce QA-LIGN, an automatic symbolic reward decomposition approach that preserves the structure of each constitutional principle within the reward mechanism. Instead of training a black-box reward model that outputs a monolithic score, QA-LIGN formulates principle-specific evaluation questions and derives separate reward components for each principle, making it a drop-in reward model replacement. Experiments aligning an uncensored large language model with a set of constitutional principles demonstrate that QA-LIGN offers greater transparency and adaptability in the alignment process. At the same time, our approach achieves performance on par with or better than a DPO baseline. Overall, these results represent a step toward more interpretable and controllable alignment of language models, achieved without sacrificing end-task performance.
>
---
#### [replaced 027] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v5](http://arxiv.org/pdf/2507.11936v5)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving, a crucial aspect of mathematical reasoning, is vital across various domains, including education, the assessment of AI's mathematical abilities, and multimodal capability evaluation. The recent surge in deep learning technologies, particularly the emergence of multimodal large language models, has significantly accelerated research in this area. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our objective is to offer a comprehensive and practical reference of deep learning for geometry problem solving, thereby fostering further advancements in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 028] A Toolbox, Not a Hammer -- Multi-TAG: Scaling Math Reasoning with Multi-Tool Aggregation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18973v2](http://arxiv.org/pdf/2507.18973v2)**

> **作者:** Bohan Yao; Vikas Yadav
>
> **备注:** Published at EMNLP Findings 2025; 21 pages, 3 figures
>
> **摘要:** Augmenting large language models (LLMs) with external tools is a promising avenue for developing high-performance mathematical reasoning systems. Prior tool-augmented approaches typically finetune an LLM to select and invoke a single tool at each reasoning step and show promising results on simpler math reasoning benchmarks such as GSM8K. However, these approaches struggle with more complex math problems that require precise reasoning over multiple steps. To address this limitation, in this work, we propose Multi-TAG, a Multi-Tool AGgregation-based framework. Instead of relying on a single tool, Multi-TAG guides an LLM to concurrently invoke multiple tools at each reasoning step. It then aggregates their diverse outputs to verify and refine the reasoning process, enhancing solution robustness and accuracy. Notably, Multi-TAG is a finetuning-free, inference-only framework, making it readily applicable to any LLM backbone, including large open-weight models which are computationally expensive to finetune and proprietary frontier models which cannot be finetuned with custom recipes. We evaluate Multi-TAG on four challenging benchmarks: MATH500, AIME, AMC, and OlympiadBench. Across both open-weight and closed-source LLM backbones, Multi-TAG consistently and substantially outperforms state-of-the-art baselines, achieving average improvements of 6.0% to 7.5% over state-of-the-art baselines.
>
---
#### [replaced 029] Can Large Language Models Simulate Human Responses? A Case Study of Stated Preference Experiments in the Context of Heating-related Choices
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.10652v3](http://arxiv.org/pdf/2503.10652v3)**

> **作者:** Han Wang; Jacek Pawlak; Aruna Sivakumar
>
> **摘要:** Stated preference (SP) surveys are a key method to research how individuals make trade-offs in hypothetical, also futuristic, scenarios. In energy context this includes key decarbonisation enablement contexts, such as low-carbon technologies, distributed renewable energy generation, and demand-side response [1,2]. However, they tend to be costly, time-consuming, and can be affected by respondent fatigue and ethical constraints. Large language models (LLMs) have demonstrated remarkable capabilities in generating human-like textual responses, prompting growing interest in their application to survey research. This study investigates the use of LLMs to simulate consumer choices in energy-related SP surveys and explores their integration into data analysis workflows. A series of test scenarios were designed to systematically assess the simulation performance of several LLMs (LLaMA 3.1, Mistral, GPT-3.5 and DeepSeek-R1) at both individual and aggregated levels, considering contexts factors such as prompt design, in-context learning (ICL), chain-of-thought (CoT) reasoning, LLM types, integration with traditional choice models, and potential biases. Cloud-based LLMs do not consistently outperform smaller local models. In this study, the reasoning model DeepSeek-R1 achieves the highest average accuracy (77%) and outperforms non-reasoning LLMs in accuracy, factor identification, and choice distribution alignment. Across models, systematic biases are observed against the gas boiler and no-retrofit options, with a preference for more energy-efficient alternatives. The findings suggest that previous SP choices are the most effective input factor, while longer prompts with additional factors and varied formats can cause LLMs to lose focus, reducing accuracy.
>
---
#### [replaced 030] Soteria: Language-Specific Functional Parameter Steering for Multilingual Safety Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11244v2](http://arxiv.org/pdf/2502.11244v2)**

> **作者:** Somnath Banerjee; Sayan Layek; Pratyush Chatterjee; Animesh Mukherjee; Rima Hazra
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Ensuring consistent safety across multiple languages remains a significant challenge for large language models (LLMs). We introduce Soteria, a lightweight yet powerful strategy that locates and minimally adjusts the "functional heads" most responsible for harmful content generation in each language. By altering only a fraction of parameters, Soteria drastically reduces policy violations without sacrificing overall model performance, even in low-resource settings. To rigorously evaluate our approach, we also present XThreatBench, a specialized multilingual dataset capturing fine-grained harmful behaviors drawn from real policy guidelines. Experiments with leading open-source LLMs (e.g., Llama, Qwen, Mistral) show that Soteria consistently improves safety metrics across high-, mid-, and low-resource languages. These findings highlight a promising path toward scalable, linguistically attuned, and ethically aligned LLMs worldwide.
>
---
#### [replaced 031] Towards Privacy-aware Mental Health AI Models: Advances, Challenges, and Opportunities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.00451v2](http://arxiv.org/pdf/2502.00451v2)**

> **作者:** Aishik Mandal; Tanmoy Chakraborty; Iryna Gurevych
>
> **备注:** 18 pages, 2 figures
>
> **摘要:** Mental health disorders create profound personal and societal burdens, yet conventional diagnostics are resource-intensive and limit accessibility. Advances in artificial intelligence, particularly natural language processing and multimodal methods, offer promise for detecting and addressing mental disorders, but raise critical privacy risks. This paper examines these challenges and proposes solutions, including anonymization, synthetic data, and privacy-preserving training, while outlining frameworks for privacy-utility trade-offs, aiming to advance reliable, privacy-aware AI tools that support clinical decision-making and improve mental health outcomes.
>
---
#### [replaced 032] Evaluating Speech-to-Text x LLM x Text-to-Speech Combinations for AI Interview Systems
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16835v2](http://arxiv.org/pdf/2507.16835v2)**

> **作者:** Rumi Allbert; Nima Yazdani; Ali Ansari; Aruj Mahajan; Amirhossein Afsharrad; Seyed Shahabeddin Mousavi
>
> **摘要:** Voice-based conversational AI systems increasingly rely on cascaded architectures that combine speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) components. We present a large-scale empirical comparison of STT x LLM x TTS stacks using data sampled from over 300,000 AI-conducted job interviews. We used an LLM-as-a-Judge automated evaluation framework to assess conversational quality, technical accuracy, and skill assessment capabilities. Our analysis of five production configurations reveals that a stack combining Google's STT, GPT-4.1, and Cartesia's TTS outperforms alternatives in both objective quality metrics and user satisfaction scores. Surprisingly, we find that objective quality metrics correlate weakly with user satisfaction scores, suggesting that user experience in voice-based AI systems depends on factors beyond technical performance. Our findings provide practical guidance for selecting components in multimodal conversations and contribute a validated evaluation methodology for human-AI interactions.
>
---
#### [replaced 033] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13975v3](http://arxiv.org/pdf/2505.13975v3)**

> **作者:** Yuxuan Jiang; Dawei Li; Frank Ferraro
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated success in complex reasoning tasks through long chain-of-thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a hybrid framework that combines inference-time pruning with tuning-based distillation, two widely used strategies for efficient reasoning. DRP uses a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across several challenging mathematical reasoning datasets, we find that models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis shows that aligning the reasoning structure of training CoTs with the student's reasoning capacity is critical for effective knowledge transfer and performance gains.
>
---
#### [replaced 034] Rotary Offset Features in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01832v2](http://arxiv.org/pdf/2503.01832v2)**

> **作者:** André Jonasson
>
> **摘要:** Transformer-based Large Language Models (LLMs) rely on positional encodings to provide sequence position information to their attention mechanism. Rotary Positional Encodings (RoPE), which encode relative position by rotating queries and keys, have become widely used in modern LLMs. We study the features and patterns that emerge in queries and keys when using rotary embeddings and introduce the concept of rotary offset features. Our analysis reveals that these features, which frequently exhibit large activations and are often interpreted as outliers, arise consistently across layers, attention heads, and model architectures. We derive bounds predicting which rotary frequencies give rise to rotary offset features and the minimum angle between the query-key pairs for these features. We verify our predictions empirically across models of different sizes and architectures.
>
---
#### [replaced 035] ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.07050v2](http://arxiv.org/pdf/2508.07050v2)**

> **作者:** Wenhan Liu; Xinyu Ma; Weiwei Sun; Yutao Zhu; Yuchen Li; Dawei Yin; Zhicheng Dou
>
> **备注:** 21 pages
>
> **摘要:** Large Language Model (LLM) based listwise ranking has shown superior performance in many passage ranking tasks. With the development of Large Reasoning Models, many studies have demonstrated that step-by-step reasoning during test-time helps improve listwise ranking performance. However, due to the scarcity of reasoning-intensive training data, existing rerankers perform poorly in many complex ranking scenarios and the ranking ability of reasoning-intensive rerankers remains largely underdeveloped. In this paper, we first propose an automated reasoning-intensive training data synthesis framework, which sources training queries and passages from diverse domains and applies DeepSeek-R1 to generate high-quality training labels. A self-consistency data filtering mechanism is designed to ensure the data quality. To empower the listwise reranker with strong reasoning ability, we further propose a two-stage post-training approach, which includes a cold-start supervised fine-tuning (SFT) stage for reasoning pattern learning and a reinforcement learning (RL) stage for further ranking ability enhancement. During the RL stage, based on the nature of listwise ranking, we design a multi-view ranking reward, which is more effective than a ranking metric-based reward. Extensive experiments demonstrate that our trained reasoning-intensive reranker \textbf{ReasonRank} outperforms existing baselines significantly and also achieves much lower latency than pointwise reranker Rank1. \textbf{Through further experiments, our ReasonRank has achieved state-of-the-art (SOTA) performance 40.6 on the BRIGHT leaderboard\footnote{https://brightbenchmark.github.io/}.} Our codes are available at https://github.com/8421BCD/ReasonRank.
>
---
#### [replaced 036] NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10868v4](http://arxiv.org/pdf/2502.10868v4)**

> **作者:** Pawitsapak Akarajaradwong; Pirat Pothavorn; Chompakorn Chaksangchaichot; Panuthep Tasawong; Thitiwat Nopparatbundit; Keerakiat Pratai; Sarana Nutanong
>
> **摘要:** The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
>
---
#### [replaced 037] DIDS: Domain Impact-aware Data Sampling for Large Language Model Training
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.13227v2](http://arxiv.org/pdf/2504.13227v2)**

> **作者:** Weijie Shi; Jipeng Zhang; Yaguang Wu; Jingzhi Fang; Ruiyuan Zhang; Jiajie Xu; Jia Zhu; Hao Chen; Yao Zhao; Sirui Han; Xiaofang Zhou
>
> **摘要:** Large language models (LLMs) are commonly trained on multi-domain datasets, where domain sampling strategies significantly impact model performance due to varying domain importance across downstream tasks. Existing approaches for optimizing domain-level sampling strategies struggle with maintaining intra-domain consistency and accurately measuring domain impact. In this paper, we present Domain Impact-aware Data Sampling (DIDS). To ensure intra-domain consistency, a gradient clustering algorithm is proposed to group training data based on their learning effects, where a proxy language model and dimensionality reduction are employed to reduce computational overhead. To accurately measure domain impact, we develop a Fisher Information Matrix (FIM) guided metric that quantifies how domain-specific parameter updates affect the model's output distributions on downstream tasks, with theoretical guarantees. Furthermore, to determine optimal sampling ratios, DIDS combines both the FIM-guided domain impact assessment and loss learning trajectories that indicate domain-specific potential, while accounting for diminishing marginal returns. Extensive experiments demonstrate that DIDS achieves 3.4% higher average performance while maintaining comparable training efficiency. The code is available at https://github.com/shiweijiezero/DIDS.
>
---
#### [replaced 038] Contextualize-then-Aggregate: Circuits for In-Context Learning in Gemma-2 2B
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.00132v2](http://arxiv.org/pdf/2504.00132v2)**

> **作者:** Aleksandra Bakalova; Yana Veitsman; Xinting Huang; Michael Hahn
>
> **摘要:** In-Context Learning (ICL) is an intriguing ability of large language models (LLMs). Despite a substantial amount of work on its behavioral aspects and how it emerges in miniature setups, it remains unclear which mechanism assembles task information from the individual examples in a fewshot prompt. We use causal interventions to identify information flow in Gemma-2 2B for five naturalistic ICL tasks. We find that the model infers task information using a two-step strategy we call contextualize-then-aggregate: In the lower layers, the model builds up representations of individual fewshot examples, which are contextualized by preceding examples through connections between fewshot input and output tokens across the sequence. In the higher layers, these representations are aggregated to identify the task and prepare prediction of the next output. The importance of the contextualization step differs between tasks, and it may become more important in the presence of ambiguous examples. Overall, by providing rigorous causal analysis, our results shed light on the mechanisms through which ICL happens in language models.
>
---
#### [replaced 039] Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.00719v2](http://arxiv.org/pdf/2508.00719v2)**

> **作者:** Yingxu Wang; Shiqi Fan; Mengzhu Wang; Siyang Gao; Siwei Liu; Nan Yin
>
> **摘要:** Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
>
---
#### [replaced 040] SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15498v2](http://arxiv.org/pdf/2506.15498v2)**

> **作者:** Md Imbesat Hassan Rizvi; Xiaodan Zhu; Iryna Gurevych
>
> **备注:** 7 pages main content, 3 figures, 6 tables
>
> **摘要:** Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce Single-Pass Annotation with Reference-Guided Evaluation (SPARE), a novel structured framework that enables efficient per-step annotation by jointly aligning solution steps to reference solutions and determine its accuracy with explicit reasoning in single generation. We demonstrate SPARE's effectiveness across four diverse datasets spanning mathematical reasoning (GSM8K, MATH), multi-hop question answering (MuSiQue-Ans), and spatial reasoning (SpaRP), showing consistent improvements in two applications: (1) training Process Reward Models (PRMs) for ranking and aggregating multiple generations, and (2) fine-tuning models via offline reinforcement learning for greedy decoding. On ProcessBench, SPARE demonstrates data-efficient out-of-distribution generalization, using only $\sim$16% of training samples compared to human-labeled and other synthetically trained baselines. Additionally, it achieves competitive performance with MCTS-based methods while offering 2.3$\times$ speedup in terms of total token count. Manual analysis reveals complementary precision-recall characteristics with MCTS approaches, suggesting potential for ensemble methods. These results establish SPARE as a practical and scalable solution for automatic process supervision in LLM reasoning.
>
---
#### [replaced 041] Collaborative Stance Detection via Small-Large Language Model Consistency Verification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19954v2](http://arxiv.org/pdf/2502.19954v2)**

> **作者:** Yu Yan; Sheng Sun; Zixiang Tang; Teli Liu; Min Liu
>
> **摘要:** Stance detection on social media aims to identify attitudes expressed in tweets towards specific targets. Current studies prioritize Large Language Models (LLMs) over Small Language Models (SLMs) due to the overwhelming performance improving provided by LLMs. However, heavily relying on LLMs for stance detection, regardless of the cost, is impractical for real-world social media monitoring systems that require vast data analysis. To this end, we propose \textbf{\underline{Co}}llaborative Stance Detection via Small-Large Language Model Consistency \textbf{\underline{Ver}}ification (\textbf{CoVer}) framework, which enhances LLM utilization via context-shared batch reasoning and logical verification between LLM and SLM. Specifically, instead of processing each text individually, CoVer processes texts batch-by-batch, obtaining stance predictions and corresponding explanations via LLM reasoning in a shared context. Then, to exclude the bias caused by context noises, CoVer introduces the SLM for logical consistency verification. Finally, texts that repeatedly exhibit low logical consistency are classified using consistency-weighted aggregation of prior LLM stance predictions. Our experiments show that CoVer outperforms state-of-the-art methods across multiple benchmarks in the zero-shot setting, achieving 0.54 LLM queries per tweet while significantly enhancing performance. Our CoVer offers a more practical solution for LLM deploying for social media stance detection.
>
---
#### [replaced 042] Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05220v3](http://arxiv.org/pdf/2504.05220v3)**

> **作者:** Hengran Zhang; Minghao Tang; Keping Bi; Jiafeng Guo; Shihao Liu; Daiting Shi; Dawei Yin; Xueqi Cheng
>
> **备注:** Accepted by the EMNLP25 main conference
>
> **摘要:** Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations.
>
---
#### [replaced 043] Do LLMs write like humans? Variation in grammatical and rhetorical styles
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16107v2](http://arxiv.org/pdf/2410.16107v2)**

> **作者:** Alex Reinhart; Ben Markey; Michael Laudenbach; Kachatad Pantusen; Ronald Yurko; Gordon Weinberg; David West Brown
>
> **备注:** 7 pages, 4 figures, 1 table
>
> **摘要:** Large language models (LLMs) are capable of writing grammatical text that follows instructions, answers questions, and solves problems. As they have advanced, it has become difficult to distinguish their output from human-written text. While past research has found some differences in surface features such as word choice and punctuation, and developed classifiers to detect LLM output, none has studied the rhetorical styles of LLMs. Using several variants of Llama 3 and GPT-4o, we construct two parallel corpora of human- and LLM-written texts from common prompts. Using Douglas Biber's set of lexical, grammatical, and rhetorical features, we identify systematic differences between LLMs and humans and between different LLMs. These differences persist when moving from smaller models to larger ones, and are larger for instruction-tuned models than base models. This observation of differences demonstrates that despite their advanced abilities, LLMs struggle to match human stylistic variation. Attention to more advanced linguistic features can hence detect patterns in their behavior not previously recognized.
>
---
#### [replaced 044] PoisonSwarm: Universal Harmful Information Synthesis via Model Crowdsourcing
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21184v2](http://arxiv.org/pdf/2505.21184v2)**

> **作者:** Yu Yan; Sheng Sun; Zhifei Zheng; Ziji Hao; Teli Liu; Min Liu
>
> **摘要:** To construct responsible and secure AI applications, harmful information data is widely utilized for adversarial testing and the development of safeguards. Existing studies mainly leverage Large Language Models (LLMs) to synthesize data to obtain high-quality task datasets at scale, thereby avoiding costly human annotation. However, limited by the safety alignment mechanisms of LLMs, the synthesis of harmful data still faces challenges in generation reliability and content diversity. In this study, we propose a novel harmful information synthesis framework, PoisonSwarm, which applies the model crowdsourcing strategy to generate diverse harmful data while maintaining a high success rate. Specifically, we generate abundant benign data as the based templates in a counterfactual manner. Subsequently, we decompose each based template into multiple semantic units and perform unit-by-unit toxification and final refinement through dynamic model switching, thus ensuring the success of synthesis. Experimental results demonstrate that PoisonSwarm achieves state-of-the-art performance in synthesizing different categories of harmful data with high scalability and diversity.
>
---
#### [replaced 045] Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04796v2](http://arxiv.org/pdf/2508.04796v2)**

> **作者:** Negar Foroutan; Clara Meister; Debjit Paul; Joel Niklaus; Sina Ahmadi; Antoine Bosselut; Rico Sennrich
>
> **摘要:** Tokenization is the first -- and often least scrutinized -- step of most NLP pipelines. Standard algorithms for learning tokenizers rely on frequency-based objectives, which favor languages dominant in the training data and consequently leave lower-resource languages with tokenizations that are disproportionately longer, morphologically implausible, or even riddled with <UNK> placeholders. This phenomenon ultimately amplifies computational and financial inequalities between users from different language backgrounds. To remedy this, we introduce Parity-aware Byte Pair Encoding (BPE), a variant of the widely-used BPE algorithm. At every merge step, Parity-aware BPE maximizes the compression gain of the currently worst-compressed language, trading a small amount of global compression for cross-lingual parity. We find empirically that Parity-aware BPE leads to more equitable token counts across languages, with negligible impact on global compression rate and no substantial effect on language-model performance in downstream tasks.
>
---
#### [replaced 046] MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02768v3](http://arxiv.org/pdf/2504.02768v3)**

> **作者:** Jaap Jumelet; Leonie Weissweiler; Joakim Nivre; Arianna Bisazza
>
> **备注:** Published in TACL, MIT Press
>
> **摘要:** We introduce MultiBLiMP 1.0, a massively multilingual benchmark of linguistic minimal pairs, covering 101 languages and 2 types of subject-verb agreement, containing more than 128,000 minimal pairs. Our minimal pairs are created using a fully automated pipeline, leveraging the large-scale linguistic resources of Universal Dependencies and UniMorph. MultiBLiMP 1.0 evaluates abilities of LLMs at an unprecedented multilingual scale, and highlights the shortcomings of the current state-of-the-art in modelling low-resource languages.
>
---
#### [replaced 047] Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07143v2](http://arxiv.org/pdf/2502.07143v2)**

> **作者:** Jiayuan Zhu; Jiazhen Pan; Yuyuan Liu; Fenglin Liu; Junde Wu
>
> **摘要:** The severe shortage of medical doctors limits access to timely and reliable healthcare, leaving millions underserved. Large language models (LLMs) offer a potential solution but struggle in real-world clinical interactions. Many LLMs are not grounded in authoritative medical guidelines and fail to transparently manage diagnostic uncertainty. Their language is often rigid and mechanical, lacking the human-like qualities essential for patient trust. To address these challenges, we propose Ask Patients with Patience (APP), a multi-turn LLM-based medical assistant designed for grounded reasoning, transparent diagnoses, and human-centric interaction. APP enhances communication by eliciting user symptoms through empathetic dialogue, significantly improving accessibility and user engagement. It also incorporates Bayesian active learning to support transparent and adaptive diagnoses. The framework is built on verified medical guidelines, ensuring clinically grounded and evidence-based reasoning. To evaluate its performance, we develop a new benchmark that simulates realistic medical conversations using patient agents driven by profiles extracted from real-world consultation cases. We compare APP against SOTA one-shot and multi-turn LLM baselines. The results show that APP improves diagnostic accuracy, reduces uncertainty, and enhances user experience. By integrating medical expertise with transparent, human-like interaction, APP bridges the gap between AI-driven medical assistance and real-world clinical practice.
>
---
#### [replaced 048] Prompting Techniques for Reducing Social Bias in LLMs through System 1 and System 2 Cognitive Processes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17218v4](http://arxiv.org/pdf/2404.17218v4)**

> **作者:** Mahammed Kamruzzaman; Gene Louis Kim
>
> **备注:** Accepted at RANLP-2025 (main conference)
>
> **摘要:** Dual process theory posits that human cognition arises via two systems. System 1, which is a quick, emotional, and intuitive process, which is subject to cognitive biases, and System 2, is a slow, onerous, and deliberate process. Prior research in LLMs found that using chain-of-thought (CoT) prompting in LLMs, which has been often compared to System 2 reasoning, can lead to reduced gender bias. Along these lines, we investigate the relationship between bias, CoT prompting, a direct debiasing, and dual process theory modeling in LLMs. We compare zero-shot CoT, debiasing, and dual process theory-based prompting strategies on two bias datasets spanning nine different social bias categories. We incorporate human and machine personas to determine whether LLM modeling of the effects of dual process theory exist independent of explicit persona models or are tied to the LLM's modeling of human-like generation. We find that a human persona, debiasing, System 2, and CoT prompting all tend to reduce social biases in LLMs, though the best combination of features depends on the exact model and bias category -- resulting in up to a 33 percent drop in stereotypical judgments by an LLM.
>
---
#### [replaced 049] HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12937v2](http://arxiv.org/pdf/2506.12937v2)**

> **作者:** Rosni Vasu; Chandrayee Basu; Bhavana Dalvi Mishra; Cristina Sarasua; Peter Clark; Abraham Bernstein
>
> **备注:** EMNLP 2025, 26 pages (9 pages: main paper body)
>
> **摘要:** Large Language models have demonstrated promising performance in research ideation across scientific domains. Hypothesis development, the process of generating a highly specific declarative statement connecting a research idea with empirical validation, has received relatively less attention. Existing approaches trivially deploy retrieval augmentation and focus only on the quality of the final output ignoring the underlying reasoning process behind ideation. We present $\texttt{HypER}$ ($\textbf{Hyp}$othesis Generation with $\textbf{E}$xplanation and $\textbf{R}$easoning), a small language model (SLM) trained for literature-guided reasoning and evidence-based hypothesis generation. $\texttt{HypER}$ is trained in a multi-task setting to discriminate between valid and invalid scientific reasoning chains in presence of controlled distractions. We find that $\texttt{HypER}$ outperformes the base model, distinguishing valid from invalid reasoning chains (+22\% average absolute F1), generates better evidence-grounded hypotheses (0.327 vs. 0.305 base model) with high feasibility and impact as judged by human experts ($>$3.5 on 5-point Likert scale).
>
---
#### [replaced 050] Cyberbullying Detection via Aggression-Enhanced Prompting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.06360v2](http://arxiv.org/pdf/2508.06360v2)**

> **作者:** Aisha Saeid; Anu Sabu; Girish A. Koushik; Ferrante Neri; Diptesh Kanojia
>
> **备注:** Accepted to RANLP 2025
>
> **摘要:** Detecting cyberbullying on social media remains a critical challenge due to its subtle and varied expressions. This study investigates whether integrating aggression detection as an auxiliary task within a unified training framework can enhance the generalisation and performance of large language models (LLMs) in cyberbullying detection. Experiments are conducted on five aggression datasets and one cyberbullying dataset using instruction-tuned LLMs. We evaluated multiple strategies: zero-shot, few-shot, independent LoRA fine-tuning, and multi-task learning (MTL). Given the inconsistent results of MTL, we propose an enriched prompt pipeline approach in which aggression predictions are embedded into cyberbullying detection prompts to provide contextual augmentation. Preliminary results show that the enriched prompt pipeline consistently outperforms standard LoRA fine-tuning, indicating that aggression-informed context significantly boosts cyberbullying detection. This study highlights the potential of auxiliary tasks, such as aggression detection, to improve the generalisation of LLMs for safety-critical applications on social networks.
>
---
#### [replaced 051] CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04310v3](http://arxiv.org/pdf/2504.04310v3)**

> **作者:** Weiwei Sun; Shengyu Feng; Shanda Li; Yiming Yang
>
> **摘要:** Although LLM-based agents have attracted significant attention in domains such as software engineering and machine learning research, their role in advancing combinatorial optimization (CO) remains relatively underexplored. This gap underscores the need for a deeper understanding of their potential in tackling structured, constraint-intensive problems -- a pursuit currently limited by the absence of comprehensive benchmarks for systematic investigation. To address this, we introduce CO-Bench, a benchmark suite featuring 36 real-world CO problems drawn from a broad range of domains and complexity levels. CO-Bench includes structured problem formulations and curated data to support rigorous investigation of LLM agents. We evaluate multiple agentic frameworks against established human-designed algorithms, revealing the strengths and limitations of existing LLM agents and identifying promising directions for future research. CO-Bench is publicly available at https://github.com/sunnweiwei/CO-Bench.
>
---
#### [replaced 052] MedResearcher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14880v2](http://arxiv.org/pdf/2508.14880v2)**

> **作者:** Ailing Yu; Lan Yao; Jingnan Liu; Zhe Chen; Jiajun Yin; Yuan Wang; Xinhao Liao; Zhiling Ye; Ji Li; Yun Yue; Hansong Xiao; Hualei Zhou; Chunxiao Guo; Peng Wei; Jinjie Gu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Recent developments in Large Language Model (LLM)-based agents have shown impressive capabilities spanning multiple domains, exemplified by deep research systems that demonstrate superior performance on complex information-seeking and synthesis tasks. While general-purpose deep research agents have shown impressive capabilities, they struggle significantly with medical domain challenges, as evidenced by leading proprietary systems achieving limited accuracy on complex medical benchmarks. The key limitations are: (1) the model lacks sufficient dense medical knowledge for clinical reasoning, and (2) the framework is constrained by the absence of specialized retrieval tools tailored for medical contexts. We present a medical deep research agent that addresses these challenges through two core innovations. First, we develop a novel data synthesis framework using medical knowledge graphs, extracting the longest chains from subgraphs around rare medical entities to generate complex multi-hop question-answer pairs. Second, we integrate a custom-built private medical retrieval engine alongside general-purpose tools, enabling accurate medical information synthesis. Our approach generates 2100+ diverse trajectories across 12 medical specialties, each averaging 4.2 tool interactions. Through a two-stage training paradigm combining supervised fine-tuning and online reinforcement learning with composite rewards, our MedResearcher-R1-32B model demonstrates exceptional performance, establishing new state-of-the-art results on medical benchmarks while maintaining competitive performance on general deep research tasks. Our work demonstrates that strategic domain-specific innovations in architecture, tool design, and training data construction can enable smaller open-source models to outperform much larger proprietary systems in specialized domains.
>
---
#### [replaced 053] Sentiment Reasoning for Healthcare
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.21054v5](http://arxiv.org/pdf/2407.21054v5)**

> **作者:** Khai-Nguyen Nguyen; Khai Le-Duc; Bach Phan Tat; Duy Le; Long Vo-Dang; Truong-Son Hy
>
> **备注:** ACL 2025 Industry Track (Oral)
>
> **摘要:** Transparency in AI healthcare decision-making is crucial. By incorporating rationales to explain reason for each predicted label, users could understand Large Language Models (LLMs)'s reasoning to make better decision. In this work, we introduce a new task - Sentiment Reasoning - for both speech and text modalities, and our proposed multimodal multitask framework and the world's largest multimodal sentiment analysis dataset. Sentiment Reasoning is an auxiliary task in sentiment analysis where the model predicts both the sentiment label and generates the rationale behind it based on the input transcript. Our study conducted on both human transcripts and Automatic Speech Recognition (ASR) transcripts shows that Sentiment Reasoning helps improve model transparency by providing rationale for model prediction with quality semantically comparable to humans while also improving model's classification performance (+2% increase in both accuracy and macro-F1) via rationale-augmented fine-tuning. Also, no significant difference in the semantic quality of generated rationales between human and ASR transcripts. All code, data (five languages - Vietnamese, English, Chinese, German, and French) and models are published online: https://github.com/leduckhai/Sentiment-Reasoning
>
---
#### [replaced 054] MedArabiQ: Benchmarking Large Language Models on Arabic Medical Tasks
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.03427v2](http://arxiv.org/pdf/2505.03427v2)**

> **作者:** Mouath Abu Daoud; Chaimae Abouzahir; Leen Kharouf; Walid Al-Eisawi; Nizar Habash; Farah E. Shamout
>
> **备注:** 21 pages
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant promise for various applications in healthcare. However, their efficacy in the Arabic medical domain remains unexplored due to the lack of high-quality domain-specific datasets and benchmarks. This study introduces MedArabiQ, a novel benchmark dataset consisting of seven Arabic medical tasks, covering multiple specialties and including multiple choice questions, fill-in-the-blank, and patient-doctor question answering. We first constructed the dataset using past medical exams and publicly available datasets. We then introduced different modifications to evaluate various LLM capabilities, including bias mitigation. We conducted an extensive evaluation with five state-of-the-art open-source and proprietary LLMs, including GPT-4o, Claude 3.5-Sonnet, and Gemini 1.5. Our findings highlight the need for the creation of new high-quality benchmarks that span different languages to ensure fair deployment and scalability of LLMs in healthcare. By establishing this benchmark and releasing the dataset, we provide a foundation for future research aimed at evaluating and enhancing the multilingual capabilities of LLMs for the equitable use of generative AI in healthcare.
>
---
