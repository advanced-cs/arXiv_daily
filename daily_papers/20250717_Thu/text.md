# 自然语言处理 cs.CL

- **最新发布 59 篇**

- **更新 41 篇**

## 最新发布

#### [new 001] Beyond Single Models: Enhancing LLM Detection of Ambiguity in Requests through Debate
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在处理用户请求时的歧义问题。通过多代理辩论框架提升模型检测与解决歧义的能力。**

- **链接: [http://arxiv.org/pdf/2507.12370v1](http://arxiv.org/pdf/2507.12370v1)**

> **作者:** Ana Davila; Jacinto Colan; Yasuhisa Hasegawa
>
> **备注:** Accepted at the 2025 SICE Festival with Annual Conference (SICE FES)
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant capabilities in understanding and generating human language, contributing to more natural interactions with complex systems. However, they face challenges such as ambiguity in user requests processed by LLMs. To address these challenges, this paper introduces and evaluates a multi-agent debate framework designed to enhance detection and resolution capabilities beyond single models. The framework consists of three LLM architectures (Llama3-8B, Gemma2-9B, and Mistral-7B variants) and a dataset with diverse ambiguities. The debate framework markedly enhanced the performance of Llama3-8B and Mistral-7B variants over their individual baselines, with Mistral-7B-led debates achieving a notable 76.7% success rate and proving particularly effective for complex ambiguities and efficient consensus. While acknowledging varying model responses to collaborative strategies, these findings underscore the debate framework's value as a targeted method for augmenting LLM capabilities. This work offers important insights for developing more robust and adaptive language understanding systems by showing how structured debates can lead to improved clarity in interactive systems.
>
---
#### [new 002] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言指令遵循任务，旨在解决现有数据集在多语言场景下的局限性。通过构建多语言基准Marco-Bench-MIF，评估大模型的跨语言能力。**

- **链接: [http://arxiv.org/pdf/2507.11882v1](http://arxiv.org/pdf/2507.11882v1)**

> **作者:** Bo Zeng; Chenyang Lyu; Sinuo Liu; Mingyan Zeng; Minghao Wu; Xuanfan Ni; Tianqi Shi; Yu Zhao; Yefeng Liu; Chenyu Zhu; Ruizhe Li; Jiahui Geng; Qing Li; Yu Tong; Longyue Wang; Weihua Luo; Kaifu Zhang
>
> **备注:** ACL 2025 Main Conference paper
>
> **摘要:** Instruction-following capability has become a major ability to be evaluated for Large Language Models (LLMs). However, existing datasets, such as IFEval, are either predominantly monolingual and centered on English or simply machine translated to other languages, limiting their applicability in multilingual contexts. In this paper, we present an carefully-curated extension of IFEval to a localized multilingual version named Marco-Bench-MIF, covering 30 languages with varying levels of localization. Our benchmark addresses linguistic constraints (e.g., modifying capitalization requirements for Chinese) and cultural references (e.g., substituting region-specific company names in prompts) via a hybrid pipeline combining translation with verification. Through comprehensive evaluation of 20+ LLMs on our Marco-Bench-MIF, we found that: (1) 25-35% accuracy gap between high/low-resource languages, (2) model scales largely impact performance by 45-60% yet persists script-specific challenges, and (3) machine-translated data underestimates accuracy by7-22% versus localized data. Our analysis identifies challenges in multilingual instruction following, including keyword consistency preservation and compositional constraint adherence across languages. Our Marco-Bench-MIF is available at https://github.com/AIDC-AI/Marco-Bench-MIF.
>
---
#### [new 003] Findings of MEGA: Maths Explanation with LLMs using the Socratic Method for Active Learning
- **分类: cs.CL**

- **简介: 该论文属于教育技术任务，旨在提升大学生数学学习效果。通过结合苏格拉底法、思维链等方法，设计MEGA系统，对比传统方法，验证其在解释复杂数学问题上的优势。**

- **链接: [http://arxiv.org/pdf/2507.12079v1](http://arxiv.org/pdf/2507.12079v1)**

> **作者:** Tosin Adewumi; Foteini Simistira Liwicki; Marcus Liwicki; Viktor Gardelli; Lama Alkhaled; Hamam Mokayed
>
> **备注:** This paper was accepted for the special issue AI for Education by the IEEE Signal Processing Magazine journal
>
> **摘要:** This paper presents an intervention study on the effects of the combined methods of (1) the Socratic method, (2) Chain of Thought (CoT) reasoning, (3) simplified gamification and (4) formative feedback on university students' Maths learning driven by large language models (LLMs). We call our approach Mathematics Explanations through Games by AI LLMs (MEGA). Some students struggle with Maths and as a result avoid Math-related discipline or subjects despite the importance of Maths across many fields, including signal processing. Oftentimes, students' Maths difficulties stem from suboptimal pedagogy. We compared the MEGA method to the traditional step-by-step (CoT) method to ascertain which is better by using a within-group design after randomly assigning questions for the participants, who are university students. Samples (n=60) were randomly drawn from each of the two test sets of the Grade School Math 8K (GSM8K) and Mathematics Aptitude Test of Heuristics (MATH) datasets, based on the error margin of 11%, the confidence level of 90%, and a manageable number of samples for the student evaluators. These samples were used to evaluate two capable LLMs at length (Generative Pretrained Transformer 4o (GPT4o) and Claude 3.5 Sonnet) out of the initial six that were tested for capability. The results showed that students agree in more instances that the MEGA method is experienced as better for learning for both datasets. It is even much better than the CoT (47.5% compared to 26.67%) in the more difficult MATH dataset, indicating that MEGA is better at explaining difficult Maths problems.
>
---
#### [new 004] Value-Based Large Language Model Agent Simulation for Mutual Evaluation of Trust and Interpersonal Closeness
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于社会模拟任务，旨在探究价值相似性对LLM代理间信任与亲密关系的影响。通过实验验证了价值相似性促进关系建立的假设。**

- **链接: [http://arxiv.org/pdf/2507.11979v1](http://arxiv.org/pdf/2507.11979v1)**

> **作者:** Yuki Sakamoto; Takahisa Uchida; Hiroshi Ishiguro
>
> **摘要:** Large language models (LLMs) have emerged as powerful tools for simulating complex social phenomena using human-like agents with specific traits. In human societies, value similarity is important for building trust and close relationships; however, it remains unexplored whether this principle holds true in artificial societies comprising LLM agents. Therefore, this study investigates the influence of value similarity on relationship-building among LLM agents through two experiments. First, in a preliminary experiment, we evaluated the controllability of values in LLMs to identify the most effective model and prompt design for controlling the values. Subsequently, in the main experiment, we generated pairs of LLM agents imbued with specific values and analyzed their mutual evaluations of trust and interpersonal closeness following a dialogue. The experiments were conducted in English and Japanese to investigate language dependence. The results confirmed that pairs of agents with higher value similarity exhibited greater mutual trust and interpersonal closeness. Our findings demonstrate that the LLM agent simulation serves as a valid testbed for social science theories, contributes to elucidating the mechanisms by which values influence relationship building, and provides a foundation for inspiring new theories and insights into the social sciences.
>
---
#### [new 005] Web-Browsing LLMs Can Access Social Media Profiles and Infer User Demographics
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与社会计算任务，研究Web-Browsing LLMs是否能通过用户名推断用户人口统计信息。工作包括构建数据集并验证模型的推断能力及潜在偏见。**

- **链接: [http://arxiv.org/pdf/2507.12372v1](http://arxiv.org/pdf/2507.12372v1)**

> **作者:** Meysam Alizadeh; Fabrizio Gilardi; Zeynab Samei; Mohsen Mosleh
>
> **摘要:** Large language models (LLMs) have traditionally relied on static training data, limiting their knowledge to fixed snapshots. Recent advancements, however, have equipped LLMs with web browsing capabilities, enabling real time information retrieval and multi step reasoning over live web content. While prior studies have demonstrated LLMs ability to access and analyze websites, their capacity to directly retrieve and analyze social media data remains unexplored. Here, we evaluate whether web browsing LLMs can infer demographic attributes of social media users given only their usernames. Using a synthetic dataset of 48 X (Twitter) accounts and a survey dataset of 1,384 international participants, we show that these models can access social media content and predict user demographics with reasonable accuracy. Analysis of the synthetic dataset further reveals how LLMs parse and interpret social media profiles, which may introduce gender and political biases against accounts with minimal activity. While this capability holds promise for computational social science in the post API era, it also raises risks of misuse particularly in information operations and targeted advertising underscoring the need for safeguards. We recommend that LLM providers restrict this capability in public facing applications, while preserving controlled access for verified research purposes.
>
---
#### [new 006] ILID: Native Script Language Identification for Indian Languages
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在解决印度多语言在相同书写系统下的识别难题。通过构建数据集和开发基准模型提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.11832v1](http://arxiv.org/pdf/2507.11832v1)**

> **作者:** Yash Ingle; Pruthwik Mishra
>
> **备注:** 8 pages, 1 figure, 7 tables, Paper accepted in RANLP 2025
>
> **摘要:** The language identification task is a crucial fundamental step in NLP. Often it serves as a pre-processing step for widely used NLP applications such as multilingual machine translation, information retrieval, question and answering, and text summarization. The core challenge of language identification lies in distinguishing languages in noisy, short, and code-mixed environments. This becomes even harder in case of diverse Indian languages that exhibit lexical and phonetic similarities, but have distinct differences. Many Indian languages share the same script making the task even more challenging. In this paper, we release a dataset of 230K sentences consisting of English and all 22 official Indian languages labeled with their language identifiers where data in most languages are newly created. We also develop and release robust baseline models using state-of-the-art approaches in machine learning and deep learning that can aid the research in this field. Our baseline models are comparable to the state-of-the-art models for the language identification task.
>
---
#### [new 007] Tracing Facts or just Copies? A critical investigation of the Competitions of Mechanisms in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机制解释任务，研究LLM如何处理矛盾信息，通过分析注意力头作用，揭示其复制抑制机制及领域依赖性。**

- **链接: [http://arxiv.org/pdf/2507.11809v1](http://arxiv.org/pdf/2507.11809v1)**

> **作者:** Dante Campregher; Yanxu Chen; Sander Hoffman; Maria Heuss
>
> **备注:** 18 Pages, 13 figures
>
> **摘要:** This paper presents a reproducibility study examining how Large Language Models (LLMs) manage competing factual and counterfactual information, focusing on the role of attention heads in this process. We attempt to reproduce and reconcile findings from three recent studies by Ortu et al., Yu, Merullo, and Pavlick and McDougall et al. that investigate the competition between model-learned facts and contradictory context information through Mechanistic Interpretability tools. Our study specifically examines the relationship between attention head strength and factual output ratios, evaluates competing hypotheses about attention heads' suppression mechanisms, and investigates the domain specificity of these attention patterns. Our findings suggest that attention heads promoting factual output do so via general copy suppression rather than selective counterfactual suppression, as strengthening them can also inhibit correct facts. Additionally, we show that attention head behavior is domain-dependent, with larger models exhibiting more specialized and category-sensitive patterns.
>
---
#### [new 008] Cross-lingual Few-shot Learning for Persian Sentiment Analysis with Incremental Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言情感分析任务，旨在用少量数据实现波斯语情感分析。通过结合少样本学习和增量学习，利用多语言预训练模型提升性能。**

- **链接: [http://arxiv.org/pdf/2507.11634v1](http://arxiv.org/pdf/2507.11634v1)**

> **作者:** Farideh Majidi; Ziaeddin Beheshtifard
>
> **备注:** Proceedings of the First National Conference on Artificial Intelligence and Emerging Research: Convergence of Humans and Intelligent Systems
>
> **摘要:** This research examines cross-lingual sentiment analysis using few-shot learning and incremental learning methods in Persian. The main objective is to develop a model capable of performing sentiment analysis in Persian using limited data, while getting prior knowledge from high-resource languages. To achieve this, three pre-trained multilingual models (XLM-RoBERTa, mDeBERTa, and DistilBERT) were employed, which were fine-tuned using few-shot and incremental learning approaches on small samples of Persian data from diverse sources, including X, Instagram, Digikala, Snappfood, and Taaghche. This variety enabled the models to learn from a broad range of contexts. Experimental results show that the mDeBERTa and XLM-RoBERTa achieved high performances, reaching 96% accuracy on Persian sentiment analysis. These findings highlight the effectiveness of combining few-shot learning and incremental learning with multilingual pre-trained models.
>
---
#### [new 009] DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的测试题生成任务，旨在解决Cloze测试中干扰项生成的问题。提出DualReward框架，通过动态奖励机制提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.11875v1](http://arxiv.org/pdf/2507.11875v1)**

> **作者:** Tianyou Huang; Xinglu Chen; Jingshen Zhang; Xinying Qiu; Ruiying Niu
>
> **备注:** Accepted to CCL 2025
>
> **摘要:** This paper introduces DualReward, a novel reinforcement learning framework for automatic distractor generation in cloze tests. Unlike conventional approaches that rely primarily on supervised learning or static generative models, our method employs a dual reward structure with adaptive scaling that differentiates between human-created gold standard distractors and model-generated candidates. The framework dynamically adjusts reward signal intensity based on model performance and confidence. We evaluate our approach on both passage-level (CLOTH-F) and sentence-level (MCQ) cloze test datasets, demonstrating consistent improvements over state-of-the-art baselines. Experimental results show that our adaptive reward scaling mechanism provides modest but consistent benefits on homogeneous datasets (CLOTH-F) and more substantial improvements (3.48-3.86% in P@1) on diverse, cross-domain data (MCQ), suggesting its particular effectiveness for handling varied question types and domains. Our work offers a flexible framework that effectively balances learning from reliable human examples while exploring novel, high-quality distractors for automated test generation.
>
---
#### [new 010] POLYCHARTQA: Benchmarking Large Vision-Language Models with Multilingual Chart Question Answering
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于多语言图表问答任务，旨在解决现有基准英语主导的问题。通过构建多语言图表数据集，评估视觉语言模型的跨语言理解能力。**

- **链接: [http://arxiv.org/pdf/2507.11939v1](http://arxiv.org/pdf/2507.11939v1)**

> **作者:** Yichen Xu; Liangyu Chen; Liang Zhang; Wenxuan Wang; Qin Jin
>
> **备注:** Work in Progress
>
> **摘要:** Charts are a universally adopted medium for interpreting and communicating data. However, existing chart understanding benchmarks are predominantly English-centric, limiting their accessibility and applicability to global audiences. In this paper, we present PolyChartQA, the first large-scale multilingual chart question answering benchmark covering 22,606 charts and 26,151 question-answering pairs across 10 diverse languages. PolyChartQA is built using a decoupled pipeline that separates chart data from rendering code, allowing multilingual charts to be flexibly generated by simply translating the data and reusing the code. We leverage state-of-the-art LLM-based translation and enforce rigorous quality control in the pipeline to ensure the linguistic and semantic consistency of the generated multilingual charts. PolyChartQA facilitates systematic evaluation of multilingual chart understanding. Experiments on both open- and closed-source large vision-language models reveal a significant performance gap between English and other languages, especially low-resource ones with non-Latin scripts. This benchmark lays a foundation for advancing globally inclusive vision-language models.
>
---
#### [new 011] BlockBPE: Parallel BPE Tokenization
- **分类: cs.CL; cs.DC**

- **简介: 该论文属于自然语言处理任务，解决GPU上BPE分词效率低的问题，提出BlockBPE实现并行化分词，提升批量推理吞吐量。**

- **链接: [http://arxiv.org/pdf/2507.11941v1](http://arxiv.org/pdf/2507.11941v1)**

> **作者:** Amos You
>
> **备注:** ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models (ICML 2025)
>
> **摘要:** Tokenization is a critical preprocessing step in large language model pipelines, yet widely-used implementations remain CPU-bound and suboptimal for batch inference workflows on GPU. We present BlockBPE, a parallel GPU implementation of byte-pair encoding (BPE) that achieves near linear-time complexity under realistic assumptions and is optimized for high-throughput, batch inference. Unlike existing Rust-based tokenizers such as HuggingFace Tokenizers or OpenAI's tiktoken-whose runtimes are dominated by Regex pre-tokenization and exhibit $O(n \log n)$ runtime-BlockBPE eliminates the Regex pre-tokenization which leads to small loss in generation quality, but enables highly parallelized token merges within thread blocks, reducing overall complexity to $O(nd)$ where $d \ll n$. On high-batch inference workloads, BlockBPE achieves up to 2x higher throughput than tiktoken and 2.5x over HuggingFace Tokenizers.
>
---
#### [new 012] AI Wizards at CheckThat! 2025: Enhancing Transformer-Based Embeddings with Sentiment for Subjectivity Detection in News Articles
- **分类: cs.CL; cs.IR**

- **简介: 该论文参与Clef 2025主题检测任务，解决新闻文章主观性分类问题。通过融合情感分数增强Transformer模型，提升分类效果。**

- **链接: [http://arxiv.org/pdf/2507.11764v1](http://arxiv.org/pdf/2507.11764v1)**

> **作者:** Matteo Fasulo; Luca Babboni; Luca Tedeschini
>
> **备注:** 14 pages, 6 figures, accepted at CLEF 2025 CheckThat! Lab
>
> **摘要:** This paper presents AI Wizards' participation in the CLEF 2025 CheckThat! Lab Task 1: Subjectivity Detection in News Articles, classifying sentences as subjective/objective in monolingual, multilingual, and zero-shot settings. Training/development datasets were provided for Arabic, German, English, Italian, and Bulgarian; final evaluation included additional unseen languages (e.g., Greek, Romanian, Polish, Ukrainian) to assess generalization. Our primary strategy enhanced transformer-based classifiers by integrating sentiment scores, derived from an auxiliary model, with sentence representations, aiming to improve upon standard fine-tuning. We explored this sentiment-augmented architecture with mDeBERTaV3-base, ModernBERT-base (English), and Llama3.2-1B. To address class imbalance, prevalent across languages, we employed decision threshold calibration optimized on the development set. Our experiments show sentiment feature integration significantly boosts performance, especially subjective F1 score. This framework led to high rankings, notably 1st for Greek (Macro F1 = 0.51).
>
---
#### [new 013] CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于代码理解任务，旨在解决LLM在解析Python笔记本时的误解问题。通过结合语法分析与LLM推理，提升对数据流和执行依赖的准确识别。**

- **链接: [http://arxiv.org/pdf/2507.11742v1](http://arxiv.org/pdf/2507.11742v1)**

> **作者:** Meng Li; Timothy M. McPhillips; Dingmin Wang; Shin-Rong Tsai; Bertram Ludäscher
>
> **备注:** Preprint. Accepted to COLM 2025
>
> **摘要:** Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O sets, then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies.
>
---
#### [new 014] A Comparative Approach to Assessing Linguistic Creativity of Large Language Models and Humans
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在评估大语言模型与人类的语义创造力。通过设计测试任务，比较两者在原创性、扩展性等方面的差异。**

- **链接: [http://arxiv.org/pdf/2507.12039v1](http://arxiv.org/pdf/2507.12039v1)**

> **作者:** Anca Dinu; Andra-Maria Florescu; Alina Resceanu
>
> **备注:** Accepted for presentation at KES 2025. To appear in Procedia Computer Science (Elsevier)
>
> **摘要:** The following paper introduces a general linguistic creativity test for humans and Large Language Models (LLMs). The test consists of various tasks aimed at assessing their ability to generate new original words and phrases based on word formation processes (derivation and compounding) and on metaphorical language use. We administered the test to 24 humans and to an equal number of LLMs, and we automatically evaluated their answers using OCSAI tool for three criteria: Originality, Elaboration, and Flexibility. The results show that LLMs not only outperformed humans in all the assessed criteria, but did better in six out of the eight test tasks. We then computed the uniqueness of the individual answers, which showed some minor differences between humans and LLMs. Finally, we performed a short manual analysis of the dataset, which revealed that humans are more inclined towards E(extending)-creativity, while LLMs favor F(ixed)-creativity.
>
---
#### [new 015] COLA-GEC: A Bidirectional Framework for Enhancing Grammatical Acceptability and Error Correction
- **分类: cs.CL**

- **简介: 该论文属于语法错误纠正（GEC）和语法可接受性判断（COLA）任务，旨在通过双向框架提升两者性能，解决语法模型不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.11867v1](http://arxiv.org/pdf/2507.11867v1)**

> **作者:** Xiangyu Yang; Xinying Qiu
>
> **备注:** Accepted to CLNLP 2025
>
> **摘要:** Grammatical Error Correction (GEC) and grammatical acceptability judgment (COLA) are core tasks in natural language processing, sharing foundational grammatical knowledge yet typically evolving independently. This paper introduces COLA-GEC, a novel bidirectional framework that enhances both tasks through mutual knowledge transfer. First, we augment grammatical acceptability models using GEC datasets, significantly improving their performance across multiple languages. Second, we integrate grammatical acceptability signals into GEC model training via a dynamic loss function, effectively guiding corrections toward grammatically acceptable outputs. Our approach achieves state-of-the-art results on several multilingual benchmarks. Comprehensive error analysis highlights remaining challenges, particularly in punctuation error correction, providing insights for future improvements in grammatical modeling.
>
---
#### [new 016] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于几何问题求解任务，旨在探讨深度学习在该领域的应用，总结任务、方法、评估指标及挑战，推动相关研究发展。**

- **链接: [http://arxiv.org/pdf/2507.11936v1](http://arxiv.org/pdf/2507.11936v1)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [new 017] DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的提示压缩任务，旨在减少冗余、提升信息密度。针对现有方法忽略注意力机制和熵变化的问题，提出动态注意力感知的压缩方法DAC，有效整合熵与注意力信息，实现更优压缩效果。**

- **链接: [http://arxiv.org/pdf/2507.11942v1](http://arxiv.org/pdf/2507.11942v1)**

> **作者:** Yi Zhao; Zuchao Li; Hai Zhao; Baoyuan Qi; Guoming Liu
>
> **备注:** ACL 2025
>
> **摘要:** Task-agnostic prompt compression leverages the redundancy in natural language to reduce computational overhead and enhance information density within prompts, especially in long-context scenarios. Existing methods predominantly rely on information entropy as the metric to compress lexical units, aiming to achieve minimal information loss. However, these approaches overlook two critical aspects: (i) the importance of attention-critical tokens at the algorithmic level, and (ii) shifts in information entropy during the compression process. Motivated by these challenges, we propose a dynamic attention-aware approach for task-agnostic prompt compression (DAC). This approach effectively integrates entropy and attention information, dynamically sensing entropy shifts during compression to achieve fine-grained prompt compression. Extensive experiments across various domains, including LongBench, GSM8K, and BBH, show that DAC consistently yields robust and substantial improvements across a diverse range of tasks and LLMs, offering compelling evidence of its efficacy.
>
---
#### [new 018] Language Models Improve When Pretraining Data Matches Target Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决预训练数据与目标任务不匹配的问题。通过BETR方法，根据基准测试示例选择预训练文档，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.12466v1](http://arxiv.org/pdf/2507.12466v1)**

> **作者:** David Mizrahi; Anders Boesen Lindbo Larsen; Jesse Allardice; Suzie Petryk; Yuri Gorokhov; Jeffrey Li; Alex Fang; Josh Gardner; Tom Gunter; Afshin Dehghan
>
> **备注:** 44 pages, 25 figures, 13 tables
>
> **摘要:** Every data selection method inherently has a target. In practice, these targets often emerge implicitly through benchmark-driven iteration: researchers develop selection strategies, train models, measure benchmark performance, then refine accordingly. This raises a natural question: what happens when we make this optimization explicit? To explore this, we propose benchmark-targeted ranking (BETR), a simple method that selects pretraining documents based on similarity to benchmark training examples. BETR embeds benchmark examples and a sample of pretraining documents in a shared space, scores this sample by similarity to benchmarks, then trains a lightweight classifier to predict these scores for the full corpus. We compare data selection methods by training over 500 models spanning $10^{19}$ to $10^{22}$ FLOPs and fitting scaling laws to them. From this, we find that simply aligning pretraining data to evaluation benchmarks using BETR achieves a 2.1x compute multiplier over DCLM-Baseline (4.7x over unfiltered data) and improves performance on 9 out of 10 tasks across all scales. BETR also generalizes well: when targeting a diverse set of benchmarks disjoint from our evaluation suite, it still matches or outperforms baselines. Our scaling analysis further reveals a clear trend: larger models require less aggressive filtering. Overall, our findings show that directly matching pretraining data to target tasks precisely shapes model capabilities and highlight that optimal selection strategies must adapt to model scale.
>
---
#### [new 019] Exploring Gender Bias in Alzheimer's Disease Detection: Insights from Mandarin and Greek Speech Perception
- **分类: cs.CL; cs.HC; cs.SD**

- **简介: 该论文属于阿尔茨海默病语音检测任务，旨在解决性别偏见问题。研究发现男性语音更易被误判为患病，且声学特征与判断相关。**

- **链接: [http://arxiv.org/pdf/2507.12356v1](http://arxiv.org/pdf/2507.12356v1)**

> **作者:** Liu He; Yuanchao Li; Rui Feng; XinRan Han; Yin-Long Liu; Yuwei Yang; Zude Zhu; Jiahong Yuan
>
> **备注:** 12 pages, 5 figures, conference or other essential info
>
> **摘要:** Gender bias has been widely observed in speech perception tasks, influenced by the fundamental voicing differences between genders. This study reveals a gender bias in the perception of Alzheimer's Disease (AD) speech. In a perception experiment involving 16 Chinese listeners evaluating both Chinese and Greek speech, we identified that male speech was more frequently identified as AD, with this bias being particularly pronounced in Chinese speech. Acoustic analysis showed that shimmer values in male speech were significantly associated with AD perception, while speech portion exhibited a significant negative correlation with AD identification. Although language did not have a significant impact on AD perception, our findings underscore the critical role of gender bias in AD speech perception. This work highlights the necessity of addressing gender bias when developing AD detection models and calls for further research to validate model performance across different linguistic contexts.
>
---
#### [new 020] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大模型推理优化任务，解决资源消耗高问题。通过分析不同规模模型间注意力矩阵相似性，提出IAM框架，提升计算速度并减少KV缓存使用。**

- **链接: [http://arxiv.org/pdf/2507.11953v1](http://arxiv.org/pdf/2507.11953v1)**

> **作者:** Yi Zhao; Zuchao Li; Hai Zhao
>
> **备注:** ACL 2025
>
> **摘要:** LLMs encounter significant challenges in resource consumption nowadays, especially with long contexts. Despite extensive efforts dedicate to enhancing inference efficiency, these methods primarily exploit internal sparsity within the models, without leveraging external information for optimization. We identify the high similarity of attention matrices across different-scale LLMs, which offers a novel perspective for optimization. We first conduct a comprehensive analysis of how to measure similarity, how to select mapping Layers and whether mapping is consistency. Based on these insights, we introduce the IAM framework, which achieves dual benefits of accelerated attention computation and reduced KV cache usage by performing attention mapping between small and large LLMs. Our experimental results demonstrate that IAM can accelerate prefill by 15% and reduce KV cache usage by 22.1% without appreciably sacrificing performance. Experiments on different series of models show the generalizability of IAM. Importantly, it is also orthogonal to many existing KV cache optimization methods, making it a versatile addition to the current toolkit for enhancing LLM efficiency.
>
---
#### [new 021] Toward a Behavioural Translation Style Space: Simulating the Temporal Dynamics of Affect, Behaviour, and Cognition in Human Translation Production
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在建模翻译过程中的行为模式。通过分析眼动和按键数据，构建行为翻译风格空间，以模拟情感、行为与认知的动态交互。**

- **链接: [http://arxiv.org/pdf/2507.12208v1](http://arxiv.org/pdf/2507.12208v1)**

> **作者:** Michael Carl; Takanori Mizowaki; Aishvarya Ray; Masaru Yamada; Devi Sri Bandaru; Xinyue Ren
>
> **摘要:** The paper introduces a Behavioural Translation Style Space (BTSS) that describes possible behavioural translation patterns. The suggested BTSS is organized as a hierarchical structure that entails various embedded processing layers. We posit that observable translation behaviour - i.e., eye and finger movements - is fundamental when executing the physical act of translation but it is caused and shaped by higher-order cognitive processes and affective translation states. We analyse records of keystrokes and gaze data as indicators of the hidden mental processing structure and organize the behavioural patterns as a multi-layered embedded BTSS. The BTSS serves as the basis for a computational translation agent to simulate the temporal dynamics of affect, automatized behaviour and cognition during human translation production.
>
---
#### [new 022] Overview of the Sensemaking Task at the ELOQUENT 2025 Lab: LLMs as Teachers, Students and Evaluators
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于Sensemaking任务，旨在评估生成模型从文本中“理解”信息的能力。通过教师、学生和评估者的三步流程，研究LLMs在问答和评分中的表现及问题。**

- **链接: [http://arxiv.org/pdf/2507.12143v1](http://arxiv.org/pdf/2507.12143v1)**

> **作者:** Pavel Šindelář; Ondřej Bojar
>
> **备注:** 30 pages, 7 figures, CLEF 2025 Conference and Labs of the Evaluation Forum
>
> **摘要:** ELOQUENT is a set of shared tasks that aims to create easily testable high-level criteria for evaluating generative language models. Sensemaking is one such shared task. In Sensemaking, we try to assess how well generative models ``make sense out of a given text'' in three steps inspired by exams in a classroom setting: (1) Teacher systems should prepare a set of questions, (2) Student systems should answer these questions, and (3) Evaluator systems should score these answers, all adhering rather strictly to a given set of input materials. We report on the 2025 edition of Sensemaking, where we had 7 sources of test materials (fact-checking analyses of statements, textbooks, transcribed recordings of a lecture, and educational videos) spanning English, German, Ukrainian, and Czech languages. This year, 4 teams participated, providing us with 2 Teacher submissions, 2 Student submissions, and 2 Evaluator submissions. We added baselines for Teacher and Student using commercial large language model systems. We devised a fully automatic evaluation procedure, which we compare to a minimalistic manual evaluation. We were able to make some interesting observations. For the first task, the creation of questions, better evaluation strategies will still have to be devised because it is difficult to discern the quality of the various candidate question sets. In the second task, question answering, the LLMs examined overall perform acceptably, but restricting their answers to the given input texts remains problematic. In the third task, evaluation of question answers, our adversarial tests reveal that systems using the LLM-as-a-Judge paradigm erroneously rate both garbled question-answer pairs and answers to mixed-up questions as acceptable.
>
---
#### [new 023] ExpliCIT-QA: Explainable Code-Based Image Table Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于表格问答任务，旨在解决传统系统解释性不足的问题。通过多模态理解、语言推理、代码生成与执行，实现可解释的表格图像问答。**

- **链接: [http://arxiv.org/pdf/2507.11694v1](http://arxiv.org/pdf/2507.11694v1)**

> **作者:** Maximiliano Hormazábal Lagos; Álvaro Bueno Sáez; Pedro Alonso Doval; Jorge Alcalde Vesteiro; Héctor Cerezo-Costas
>
> **备注:** This work has been accepted for presentation at the 24nd Portuguese Conference on Artificial Intelligence (EPIA 2025) and will be published in the proceedings by Springer in the Lecture Notes in Computer Science (LNCS) series. Please cite the published version when available
>
> **摘要:** We present ExpliCIT-QA, a system that extends our previous MRT approach for tabular question answering into a multimodal pipeline capable of handling complex table images and providing explainable answers. ExpliCIT-QA follows a modular design, consisting of: (1) Multimodal Table Understanding, which uses a Chain-of-Thought approach to extract and transform content from table images; (2) Language-based Reasoning, where a step-by-step explanation in natural language is generated to solve the problem; (3) Automatic Code Generation, where Python/Pandas scripts are created based on the reasoning steps, with feedback for handling errors; (4) Code Execution to compute the final answer; and (5) Natural Language Explanation that describes how the answer was computed. The system is built for transparency and auditability: all intermediate outputs, parsed tables, reasoning steps, generated code, and final answers are available for inspection. This strategy works towards closing the explainability gap in end-to-end TableVQA systems. We evaluated ExpliCIT-QA on the TableVQA-Bench benchmark, comparing it with existing baselines. We demonstrated improvements in interpretability and transparency, which open the door for applications in sensitive domains like finance and healthcare where auditing results are critical.
>
---
#### [new 024] Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型生成速度慢的问题。通过多token并行预测技术提升生成效率，同时保持质量。**

- **链接: [http://arxiv.org/pdf/2507.11851v1](http://arxiv.org/pdf/2507.11851v1)**

> **作者:** Mohammad Samragh; Arnav Kundu; David Harrison; Kumari Nishu; Devang Naik; Minsik Cho; Mehrdad Farajtabar
>
> **摘要:** Autoregressive language models are constrained by their inherently sequential nature, generating one token at a time. This paradigm limits inference speed and parallelism, especially during later stages of generation when the direction and semantics of text are relatively certain. In this work, we propose a novel framework that leverages the inherent knowledge of vanilla autoregressive language models about future tokens, combining techniques to realize this potential and enable simultaneous prediction of multiple subsequent tokens. Our approach introduces several key innovations: (1) a masked-input formulation where multiple future tokens are jointly predicted from a common prefix; (2) a gated LoRA formulation that preserves the original LLM's functionality, while equipping it for multi-token prediction; (3) a lightweight, learnable sampler module that generates coherent sequences from the predicted future tokens; (4) a set of auxiliary training losses, including a consistency loss, to enhance the coherence and accuracy of jointly generated tokens; and (5) a speculative generation strategy that expands tokens quadratically in the future while maintaining high fidelity. Our method achieves significant speedups through supervised fine-tuning on pretrained models. For example, it generates code and math nearly 5x faster, and improves general chat and knowledge tasks by almost 2.5x. These gains come without any loss in quality.
>
---
#### [new 025] Graph Representations for Reading Comprehension Analysis using Large Language Model and Eye-Tracking Biomarker
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于阅读理解研究任务，旨在分析人类与LLM在语言理解上的差异。通过构建语义图表示，比较眼动数据，探索更深层次的语言理解模式。**

- **链接: [http://arxiv.org/pdf/2507.11972v1](http://arxiv.org/pdf/2507.11972v1)**

> **作者:** Yuhong Zhang; Jialu Li; Shilai Yang; Yuchen Xu; Gert Cauwenberghs; Tzyy-Ping Jung
>
> **摘要:** Reading comprehension is a fundamental skill in human cognitive development. With the advancement of Large Language Models (LLMs), there is a growing need to compare how humans and LLMs understand language across different contexts and apply this understanding to functional tasks such as inference, emotion interpretation, and information retrieval. Our previous work used LLMs and human biomarkers to study the reading comprehension process. The results showed that the biomarkers corresponding to words with high and low relevance to the inference target, as labeled by the LLMs, exhibited distinct patterns, particularly when validated using eye-tracking data. However, focusing solely on individual words limited the depth of understanding, which made the conclusions somewhat simplistic despite their potential significance. This study used an LLM-based AI agent to group words from a reading passage into nodes and edges, forming a graph-based text representation based on semantic meaning and question-oriented prompts. We then compare the distribution of eye fixations on important nodes and edges. Our findings indicate that LLMs exhibit high consistency in language understanding at the level of graph topological structure. These results build on our previous findings and offer insights into effective human-AI co-learning strategies.
>
---
#### [new 026] BOOKCOREF: Coreference Resolution at Book Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于共指消解任务，解决长文本（如书籍）上共指识别不足的问题。提出BOOKCOREF基准，提升系统在长文档上的性能。**

- **链接: [http://arxiv.org/pdf/2507.12075v1](http://arxiv.org/pdf/2507.12075v1)**

> **作者:** Giuliano Martinelli; Tommaso Bonomo; Pere-Lluís Huguet Cabot; Roberto Navigli
>
> **备注:** Accepted to ACL 2025 Main Conference. 19 pages
>
> **摘要:** Coreference Resolution systems are typically evaluated on benchmarks containing small- to medium-scale documents. When it comes to evaluating long texts, however, existing benchmarks, such as LitBank, remain limited in length and do not adequately assess system capabilities at the book scale, i.e., when co-referring mentions span hundreds of thousands of tokens. To fill this gap, we first put forward a novel automatic pipeline that produces high-quality Coreference Resolution annotations on full narrative texts. Then, we adopt this pipeline to create the first book-scale coreference benchmark, BOOKCOREF, with an average document length of more than 200,000 tokens. We carry out a series of experiments showing the robustness of our automatic procedure and demonstrating the value of our resource, which enables current long-document coreference systems to gain up to +20 CoNLL-F1 points when evaluated on full books. Moreover, we report on the new challenges introduced by this unprecedented book-scale setting, highlighting that current models fail to deliver the same performance they achieve on smaller documents. We release our data and code to encourage research and development of new book-scale Coreference Resolution systems at https://github.com/sapienzanlp/bookcoref.
>
---
#### [new 027] Chain-of-Descriptions: Improving Code LLMs for VHDL Code Generation and Summarization
- **分类: cs.CL; cs.AI; cs.AR**

- **简介: 该论文属于VHDL代码生成与摘要任务，旨在解决LLMs在硬件描述语言上的性能不足问题。通过提出CoDes方法提升模型表现。**

- **链接: [http://arxiv.org/pdf/2507.12308v1](http://arxiv.org/pdf/2507.12308v1)**

> **作者:** Prashanth Vijayaraghavan; Apoorva Nitsure; Charles Mackin; Luyao Shi; Stefano Ambrogio; Arvind Haran; Viresh Paruthi; Ali Elzein; Dan Coops; David Beymer; Tyler Baldwin; Ehsan Degan
>
> **备注:** 10 pages (6 content pages + 4 supplementary), 5 figures, Proceedings of the 2024 ACM/IEEE International Symposium on Machine Learning for CAD. 2024 (MLCAD'24)
>
> **摘要:** Large Language Models (LLMs) have become widely used across diverse NLP tasks and domains, demonstrating their adaptability and effectiveness. In the realm of Electronic Design Automation (EDA), LLMs show promise for tasks like Register-Transfer Level (RTL) code generation and summarization. However, despite the proliferation of LLMs for general code-related tasks, there's a dearth of research focused on evaluating and refining these models for hardware description languages (HDLs), notably VHDL. In this study, we evaluate the performance of existing code LLMs for VHDL code generation and summarization using various metrics and two datasets -- VHDL-Eval and VHDL-Xform. The latter, an in-house dataset, aims to gauge LLMs' understanding of functionally equivalent code. Our findings reveal consistent underperformance of these models across different metrics, underscoring a significant gap in their suitability for this domain. To address this challenge, we propose Chain-of-Descriptions (CoDes), a novel approach to enhance the performance of LLMs for VHDL code generation and summarization tasks. CoDes involves generating a series of intermediate descriptive steps based on: (i) the problem statement for code generation, and (ii) the VHDL code for summarization. These steps are then integrated with the original input prompt (problem statement or code) and provided as input to the LLMs to generate the final output. Our experiments demonstrate that the CoDes approach significantly surpasses the standard prompting strategy across various metrics on both datasets. This method not only improves the quality of VHDL code generation and summarization but also serves as a framework for future research aimed at enhancing code LLMs for VHDL.
>
---
#### [new 028] PoTPTQ: A Two-step Power-of-Two Post-training for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型量化任务，旨在解决低精度LLM部署中的效率与准确性问题。提出PoTPTQ框架，通过两步后训练提升量化效果并加速解量化过程。**

- **链接: [http://arxiv.org/pdf/2507.11959v1](http://arxiv.org/pdf/2507.11959v1)**

> **作者:** Xinyu Wang; Vahid Partovi Nia; Peng Lu; Jerry Huang; Xiao-Wen Chang; Boxing Chen; Yufei Cui
>
> **备注:** Accepted at ECAI 2025 (European Conference on Artificial Intelligence)
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, their deployment is challenging due to the substantial computational resources required. Power-of-two (PoT) quantization is a general tool to counteract this difficulty. Albeit previous works on PoT quantization can be efficiently dequantized on CPUs using fixed-point addition, it showed less effectiveness on GPUs. The reason is entanglement of the sign bit and sequential bit manipulations needed for dequantization. We propose a novel POT quantization framework for LLM weights that (i) outperforms state-of-the-art accuracy in extremely low-precision number formats, and (ii) enables faster inference through more efficient dequantization. To maintain the accuracy of the quantized model, we introduce a two-step post-training algorithm: (i) initialize the quantization scales with a robust starting point, and (ii) refine these scales using a minimal calibration set. The performance of our PoT post-training algorithm surpasses the current state-of-the-art in integer quantization, particularly at low precisions such as 2- and 3-bit formats. Our PoT quantization accelerates the dequantization step required for the floating point inference and leads to $3.67\times$ speed up on a NVIDIA V100, and $1.63\times$ on a NVIDIA RTX 4090, compared to uniform integer dequantization.
>
---
#### [new 029] MapIQ: Benchmarking Multimodal Large Language Models for Map Question Answering
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于地图问答任务，旨在解决现有研究对地图类型和分析任务覆盖不足的问题。工作包括构建MapIQ数据集并评估多模态大模型的表现。**

- **链接: [http://arxiv.org/pdf/2507.11625v1](http://arxiv.org/pdf/2507.11625v1)**

> **作者:** Varun Srivastava; Fan Lei; Srija Mukhopadhyay; Vivek Gupta; Ross Maciejewski
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have driven researchers to explore how well these models read data visualizations, e.g., bar charts, scatter plots. More recently, attention has shifted to visual question answering with maps (Map-VQA). However, Map-VQA research has primarily focused on choropleth maps, which cover only a limited range of thematic categories and visual analytical tasks. To address these gaps, we introduce MapIQ, a benchmark dataset comprising 14,706 question-answer pairs across three map types: choropleth maps, cartograms, and proportional symbol maps spanning topics from six distinct themes (e.g., housing, crime). We evaluate multiple MLLMs using six visual analytical tasks, comparing their performance against one another and a human baseline. An additional experiment examining the impact of map design changes (e.g., altered color schemes, modified legend designs, and removal of map elements) provides insights into the robustness and sensitivity of MLLMs, their reliance on internal geographic knowledge, and potential avenues for improving Map-VQA performance.
>
---
#### [new 030] Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本异常检测任务，旨在解决缺乏标准化基准的问题。通过构建基准测试，评估不同模型和算法的检测效果，发现嵌入质量对检测效果影响显著。**

- **链接: [http://arxiv.org/pdf/2507.12295v1](http://arxiv.org/pdf/2507.12295v1)**

> **作者:** Feng Xiao; Jicong Fan
>
> **摘要:** Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMa-2, LLama-3, Mistral, OpenAI (small, ada, large)); (3) multi-domain text datasets (news, social media, scientific publications); (4) comprehensive evaluation metrics (AUROC, AUPRC). Our experiments reveal a critical empirical insight: embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived embeddings.In addition, we observe strongly low-rank characteristics in cross-model performance matrices, which enables an efficient strategy for rapid model evaluation (or embedding evaluation) and selection in practical applications. Furthermore, by open-sourcing our benchmark toolkit that includes all embeddings from different models and code at https://github.com/jicongfan/Text-Anomaly-Detection-Benchmark, this work provides a foundation for future research in robust and scalable text anomaly detection systems.
>
---
#### [new 031] Toxicity-Aware Few-Shot Prompting for Low-Resource Singlish Translation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于低资源语言翻译任务，解决毒害内容翻译中语境和毒性保留问题。通过两阶段框架优化提示工程与模型选择，提升翻译质量与文化敏感性。**

- **链接: [http://arxiv.org/pdf/2507.11966v1](http://arxiv.org/pdf/2507.11966v1)**

> **作者:** Ziyu Ge; Gabriel Chua; Leanne Tan; Roy Ka-Wei Lee
>
> **摘要:** As online communication increasingly incorporates under-represented languages and colloquial dialects, standard translation systems often fail to preserve local slang, code-mixing, and culturally embedded markers of harmful speech. Translating toxic content between low-resource language pairs poses additional challenges due to scarce parallel data and safety filters that sanitize offensive expressions. In this work, we propose a reproducible, two-stage framework for toxicity-preserving translation, demonstrated on a code-mixed Singlish safety corpus. First, we perform human-verified few-shot prompt engineering: we iteratively curate and rank annotator-selected Singlish-target examples to capture nuanced slang, tone, and toxicity. Second, we optimize model-prompt pairs by benchmarking several large language models using semantic similarity via direct and back-translation. Quantitative human evaluation confirms the effectiveness and efficiency of our pipeline. Beyond improving translation quality, our framework contributes to the safety of multicultural LLMs by supporting culturally sensitive moderation and benchmarking in low-resource contexts. By positioning Singlish as a testbed for inclusive NLP, we underscore the importance of preserving sociolinguistic nuance in real-world applications such as content moderation and regional platform governance.
>
---
#### [new 032] Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型对齐任务，旨在预测推理过程中响应的偏差。通过分析思维链激活值，使用线性探测器提前识别潜在风险，实现实时安全监控。**

- **链接: [http://arxiv.org/pdf/2507.12428v1](http://arxiv.org/pdf/2507.12428v1)**

> **作者:** Yik Siu Chan; Zheng-Xin Yong; Stephen H. Bach
>
> **摘要:** Open-weights reasoning language models generate long chains-of-thought (CoTs) before producing a final response, which improves performance but introduces additional alignment risks, with harmful content often appearing in both the CoTs and the final outputs. In this work, we investigate if we can use CoTs to predict final response misalignment. We evaluate a range of monitoring approaches, including humans, highly-capable large language models, and text classifiers, using either CoT text or activations. First, we find that a simple linear probe trained on CoT activations can significantly outperform all text-based methods in predicting whether a final response will be safe or unsafe. CoT texts are often unfaithful and can mislead humans and classifiers, while model latents (i.e., CoT activations) offer a more reliable predictive signal. Second, the probe makes accurate predictions before reasoning completes, achieving strong performance even when applied to early CoT segments. These findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation.
>
---
#### [new 033] Infherno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床数据结构化任务，旨在解决从自由文本生成FHIR资源的问题。提出Infherno框架，利用LLM代理和术语库实现端到端生成。**

- **链接: [http://arxiv.org/pdf/2507.12261v1](http://arxiv.org/pdf/2507.12261v1)**

> **作者:** Johann Frei; Nils Feldhus; Lisa Raithel; Roland Roller; Alexander Meyer; Frank Kramer
>
> **备注:** Submitted to EMNLP 2025 System Demonstrations | Code: https://github.com/j-frei/Infherno | Video: https://www.youtube.com/watch?v=kyj5C2ivbMw | Demo: https://infherno.misit-augsburg.de | HuggingFace Spaces: https://huggingface.co/spaces/nfel/infherno
>
> **摘要:** For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources rely on modular, rule-based systems or LLMs with instruction tuning and constrained decoding. Since they frequently suffer from limited generalizability and structural inconformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Infherno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions.
>
---
#### [new 034] Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data
- **分类: cs.CL; cs.AI; cs.CE; cs.IR**

- **简介: 该论文属于信息检索与生成任务，旨在解决企业结构化数据处理难题。通过改进的RAG框架提升检索与生成效果。**

- **链接: [http://arxiv.org/pdf/2507.12425v1](http://arxiv.org/pdf/2507.12425v1)**

> **作者:** Chandana Cheerla
>
> **摘要:** Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data. This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability. Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at https://github.com/CheerlaChandana/Enterprise-Chatbot
>
---
#### [new 035] Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究简化语言对LLM生成同义词定义的影响，旨在解决信息丢失问题，通过实验验证不同模型效果并提出优化方法。**

- **链接: [http://arxiv.org/pdf/2507.11981v1](http://arxiv.org/pdf/2507.11981v1)**

> **作者:** Lukas Ellinger; Miriam Anschütz; Georg Groh
>
> **备注:** Accepted by RANLP 2025
>
> **摘要:** Large Language Models (LLMs) can provide accurate word definitions and explanations for any context. However, the scope of the definition changes for different target groups, like children or language learners. This is especially relevant for homonyms, words with multiple meanings, where oversimplification might risk information loss by omitting key senses, potentially misleading users who trust LLM outputs. We investigate how simplification impacts homonym definition quality across three target groups: Normal, Simple, and ELI5. Using two novel evaluation datasets spanning multiple languages, we test DeepSeek v3, Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge and human annotations. Our results show that simplification drastically degrades definition completeness by neglecting polysemy, increasing the risk of misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization substantially improves homonym response quality across all prompt types. These findings highlight the need to balance simplicity and completeness in educational NLP to ensure reliable, context-aware definitions for all learners.
>
---
#### [new 036] Iterative Augmentation with Summarization Refinement (IASR) Evaluation for Unstructured Survey data Modeling and Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决数据稀疏下的语义建模问题。提出IASR框架评估文本增强效果，提升主题建模精度。**

- **链接: [http://arxiv.org/pdf/2507.12126v1](http://arxiv.org/pdf/2507.12126v1)**

> **作者:** Payal Bhattad; Sai Manoj Pudukotai Dinakarrao; Anju Gupta
>
> **摘要:** Text data augmentation is a widely used strategy for mitigating data sparsity in natural language processing (NLP), particularly in low-resource settings where limited samples hinder effective semantic modeling. While augmentation can improve input diversity and downstream interpretability, existing techniques often lack mechanisms to ensure semantic preservation during large-scale or iterative generation, leading to redundancy and instability. This work introduces a principled evaluation framework for large language model (LLM) based text augmentation, comprising two components: (1) Scalability Analysis, which measures semantic consistency as augmentation volume increases, and (2) Iterative Augmentation with Summarization Refinement (IASR), which evaluates semantic drift across recursive paraphrasing cycles. Empirical evaluations across state-of-the-art LLMs show that GPT-3.5 Turbo achieved the best balance of semantic fidelity, diversity, and generation efficiency. Applied to a real-world topic modeling task using BERTopic with GPT-enhanced few-shot labeling, the proposed approach results in a 400% increase in topic granularity and complete elimination of topic overlaps. These findings validated the utility of the proposed frameworks for structured evaluation of LLM-based augmentation in practical NLP pipelines.
>
---
#### [new 037] Cross-Domain Transfer and Few-Shot Learning for Personal Identifiable Information Recognition
- **分类: cs.CL**

- **简介: 该论文属于文本匿名化任务，旨在提升PII识别的准确性。研究解决跨领域迁移与小样本学习问题，通过多领域数据融合和模型迁移验证有效性。**

- **链接: [http://arxiv.org/pdf/2507.11862v1](http://arxiv.org/pdf/2507.11862v1)**

> **作者:** Junhong Ye; Xu Yuan; Xinying Qiu
>
> **备注:** Accepted to CLNLP 2025
>
> **摘要:** Accurate recognition of personally identifiable information (PII) is central to automated text anonymization. This paper investigates the effectiveness of cross-domain model transfer, multi-domain data fusion, and sample-efficient learning for PII recognition. Using annotated corpora from healthcare (I2B2), legal (TAB), and biography (Wikipedia), we evaluate models across four dimensions: in-domain performance, cross-domain transferability, fusion, and few-shot learning. Results show legal-domain data transfers well to biographical texts, while medical domains resist incoming transfer. Fusion benefits are domain-specific, and high-quality recognition is achievable with only 10% of training data in low-specialization domains.
>
---
#### [new 038] Towards few-shot isolated word reading assessment
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决低资源环境下孤立词阅读评估问题。通过少样本方法，利用SSL模型提取特征进行分类，但发现儿童语音效果下降明显。**

- **链接: [http://arxiv.org/pdf/2507.12217v1](http://arxiv.org/pdf/2507.12217v1)**

> **作者:** Reuben Smit; Retief Louw; Herman Kamper
>
> **备注:** Accepted to SLaTE 2025
>
> **摘要:** We explore an ASR-free method for isolated word reading assessment in low-resource settings. Our few-shot approach compares input child speech to a small set of adult-provided reference templates. Inputs and templates are encoded using intermediate layers from large self-supervised learned (SSL) models. Using an Afrikaans child speech benchmark, we investigate design options such as discretising SSL features and barycentre averaging of the templates. Idealised experiments show reasonable performance for adults, but a substantial drop for child speech input, even with child templates. Despite the success of employing SSL representations in low-resource speech tasks, our work highlights the limitations of SSL representations for processing child data when used in a few-shot classification system.
>
---
#### [new 039] Partitioner Guided Modal Learning Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态学习任务，旨在解决如何有效分离和学习单模态与跨模态特征的问题。提出PgM框架，通过分割、学习和重建实现更优的多模态表示。**

- **链接: [http://arxiv.org/pdf/2507.11661v1](http://arxiv.org/pdf/2507.11661v1)**

> **作者:** Guimin Hu; Yi Xin; Lijie Hu; Zhihong Zhu; Hasti Seifi
>
> **备注:** acm multimedia 2025
>
> **摘要:** Multimodal learning benefits from multiple modal information, and each learned modal representations can be divided into uni-modal that can be learned from uni-modal training and paired-modal features that can be learned from cross-modal interaction. Building on this perspective, we propose a partitioner-guided modal learning framework, PgM, which consists of the modal partitioner, uni-modal learner, paired-modal learner, and uni-paired modal decoder. Modal partitioner segments the learned modal representation into uni-modal and paired-modal features. Modal learner incorporates two dedicated components for uni-modal and paired-modal learning. Uni-paired modal decoder reconstructs modal representation based on uni-modal and paired-modal features. PgM offers three key benefits: 1) thorough learning of uni-modal and paired-modal features, 2) flexible distribution adjustment for uni-modal and paired-modal representations to suit diverse downstream tasks, and 3) different learning rates across modalities and partitions. Extensive experiments demonstrate the effectiveness of PgM across four multimodal tasks and further highlight its transferability to existing models. Additionally, we visualize the distribution of uni-modal and paired-modal features across modalities and tasks, offering insights into their respective contributions.
>
---
#### [new 040] Improving Data and Parameter Efficiency of Neural Language Models Using Representation Analysis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升神经语言模型的数据和参数效率。通过表示分析、优化技术和主动学习等方法，解决模型训练中的资源浪费和泛化能力问题。**

- **链接: [http://arxiv.org/pdf/2507.12004v1](http://arxiv.org/pdf/2507.12004v1)**

> **作者:** Josip Jukić
>
> **摘要:** This thesis addresses challenges related to data and parameter efficiency in neural language models, with a focus on representation analysis and the introduction of new optimization techniques. The first part examines the properties and dynamics of language representations within neural models, emphasizing their significance in enhancing robustness and generalization. It proposes innovative approaches based on representation smoothness, including regularization strategies that utilize Jacobian and Hessian matrices to stabilize training and mitigate sensitivity to input perturbations. The second part focuses on methods to significantly enhance data and parameter efficiency by integrating active learning strategies with parameter-efficient fine-tuning, guided by insights from representation smoothness analysis. It presents smoothness-informed early-stopping techniques designed to eliminate the need for labeled validation sets and proposes innovative combinations of active learning and parameter-efficient fine-tuning to reduce labeling efforts and computational resources. Extensive experimental evaluations across various NLP tasks demonstrate that these combined approaches substantially outperform traditional methods in terms of performance, stability, and efficiency. The third part explores weak supervision techniques enhanced by in-context learning to effectively utilize unlabeled data, further reducing dependence on extensive labeling. It shows that using in-context learning as a mechanism for weak supervision enables models to better generalize from limited labeled data by leveraging unlabeled examples more effectively during training. Comprehensive empirical evaluations confirm significant gains in model accuracy, adaptability, and robustness, especially in low-resource settings and dynamic data environments.
>
---
#### [new 041] LLMs Encode Harmfulness and Refusal Separately
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，研究LLMs如何内部编码有害性与拒绝行为。工作包括识别有害性维度，提出Latent Guard用于检测危险输入，提升安全性。**

- **链接: [http://arxiv.org/pdf/2507.11878v1](http://arxiv.org/pdf/2507.11878v1)**

> **作者:** Jiachen Zhao; Jing Huang; Zhengxuan Wu; David Bau; Weiyan Shi
>
> **摘要:** LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety
>
---
#### [new 042] Probing for Arithmetic Errors in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型自检任务，旨在检测语言模型中的算术错误。通过分析内部激活状态，训练轻量级检测器并提升任务准确性。**

- **链接: [http://arxiv.org/pdf/2507.12379v1](http://arxiv.org/pdf/2507.12379v1)**

> **作者:** Yucheng Sun; Alessandro Stolfo; Mrinmaya Sachan
>
> **摘要:** We investigate whether internal activations in language models can be used to detect arithmetic errors. Starting with a controlled setting of 3-digit addition, we show that simple probes can accurately decode both the model's predicted output and the correct answer from hidden states, regardless of whether the model's output is correct. Building on this, we train lightweight error detectors that predict model correctness with over 90% accuracy. We then extend our analysis to structured chain-of-thought traces on addition-only GSM8K problems and find that probes trained on simple arithmetic generalize well to this more complex setting, revealing consistent internal representations. Finally, we demonstrate that these probes can guide selective re-prompting of erroneous reasoning steps, improving task accuracy with minimal disruption to correct outputs. Our findings suggest that arithmetic errors can be anticipated from internal activations alone, and that simple probes offer a viable path toward lightweight model self-correction.
>
---
#### [new 043] Improving Contextual ASR via Multi-grained Fusion with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，解决ASR在识别上下文关键词上的不足。通过多粒度融合方法结合词和短语级别信息，提升关键词识别效果。**

- **链接: [http://arxiv.org/pdf/2507.12252v1](http://arxiv.org/pdf/2507.12252v1)**

> **作者:** Shilin Zhou; Zhenghua Li
>
> **摘要:** While end-to-end Automatic Speech Recognition (ASR) models have shown impressive performance in transcribing general speech, they often struggle to accurately recognize contextually relevant keywords, such as proper nouns or user-specific entities. Previous approaches have explored leveraging keyword dictionaries in the textual modality to improve keyword recognition, either through token-level fusion that guides token-by-token generation or phrase-level fusion that enables direct copying of keyword phrases. However, these methods operate at different granularities and have their own limitations. In this paper, we propose a novel multi-grained fusion approach that jointly leverages the strengths of both token-level and phrase-level fusion with Large Language Models (LLMs). Our approach incorporates a late-fusion strategy that elegantly combines ASR's acoustic information with LLM's rich contextual knowledge, balancing fine-grained token precision with holistic phrase-level understanding. Experiments on Chinese and English datasets demonstrate that our approach achieves state-of-the-art performance on keyword-related metrics while preserving high accuracy on non-keyword text. Ablation studies further confirm that the token-level and phrase-level components both contribute significantly to the performance gains, complementing each other in our joint multi-grained framework. The code and models will be publicly available at https://github.com/.
>
---
#### [new 044] Subjective Evaluation Profile Analysis of Science Fiction Short Stories and its Critical-Theoretical Significance
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本评价任务，旨在分析LLMs的文学评价模式与偏好，通过实验揭示其非中立的评价特征。**

- **链接: [http://arxiv.org/pdf/2507.11582v1](http://arxiv.org/pdf/2507.11582v1)**

> **作者:** Kazuyoshi Otsuka
>
> **备注:** 38 pages. Manuscript submitted for review to the Journal of Computational Literary Studies (JCLS)
>
> **摘要:** This study positions large language models (LLMs) as "subjective literary critics" to explore aesthetic preferences and evaluation patterns in literary assessment. Ten Japanese science fiction short stories were translated into English and evaluated by six state-of-the-art LLMs across seven independent sessions. Principal component analysis and clustering techniques revealed significant variations in evaluation consistency ({\alpha} ranging from 1.00 to 0.35) and five distinct evaluation patterns. Additionally, evaluation variance across stories differed by up to 4.5-fold, with TF-IDF analysis confirming distinctive evaluation vocabularies for each model. Our seven-session within-day protocol using an original Science Fiction corpus strategically minimizes external biases, allowing us to observe implicit value systems shaped by RLHF and their influence on literary judgment. These findings suggest that LLMs may possess individual evaluation characteristics similar to human critical schools, rather than functioning as neutral benchmarkers.
>
---
#### [new 045] StylOch at PAN: Gradient-Boosted Trees with Frequency-Based Stylometric Features
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI文本检测任务，旨在区分机器生成与人类写作。通过提取语言特征并使用梯度提升树进行分类，构建了一个高效且可解释的检测系统。**

- **链接: [http://arxiv.org/pdf/2507.12064v1](http://arxiv.org/pdf/2507.12064v1)**

> **作者:** Jeremi K. Ochab; Mateusz Matias; Tymoteusz Boba; Tomasz Walkowiak
>
> **摘要:** This submission to the binary AI detection task is based on a modular stylometric pipeline, where: public spaCy models are used for text preprocessing (including tokenisation, named entity recognition, dependency parsing, part-of-speech tagging, and morphology annotation) and extracting several thousand features (frequencies of n-grams of the above linguistic annotations); light-gradient boosting machines are used as the classifier. We collect a large corpus of more than 500 000 machine-generated texts for the classifier's training. We explore several parameter options to increase the classifier's capacity and take advantage of that training set. Our approach follows the non-neural, computationally inexpensive but explainable approach found effective previously.
>
---
#### [new 046] The benefits of query-based KGQA systems for complex and temporal questions in LLM era
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识图谱问答任务，解决大语言模型在多跳和时间问题上的不足，提出多阶段查询框架并引入新的实体链接方法，提升小模型的问答性能。**

- **链接: [http://arxiv.org/pdf/2507.11954v1](http://arxiv.org/pdf/2507.11954v1)**

> **作者:** Artem Alekseev; Mikhail Chaichuk; Miron Butko; Alexander Panchenko; Elena Tutubalina; Oleg Somov
>
> **备注:** 15 pages, 3 figures, 7 tables
>
> **摘要:** Large language models excel in question-answering (QA) yet still struggle with multi-hop reasoning and temporal questions. Query-based knowledge graph QA (KGQA) offers a modular alternative by generating executable queries instead of direct answers. We explore multi-stage query-based framework for WikiData QA, proposing multi-stage approach that enhances performance on challenging multi-hop and temporal benchmarks. Through generalization and rejection studies, we evaluate robustness across multi-hop and temporal QA datasets. Additionally, we introduce a novel entity linking and predicate matching method using CoT reasoning. Our results demonstrate the potential of query-based multi-stage KGQA framework for improving multi-hop and temporal QA with small language models. Code and data: https://github.com/ar2max/NLDB-KGQA-System
>
---
#### [new 047] Evaluating the Ability of Large Language Models to Reason about Cardinal Directions, Revisited
- **分类: cs.CL**

- **简介: 该论文属于方向推理任务，研究LLM在不同情境下判断 cardinal directions 的能力，发现即使新模型也存在不足。**

- **链接: [http://arxiv.org/pdf/2507.12059v1](http://arxiv.org/pdf/2507.12059v1)**

> **作者:** Anthony G Cohn; Robert E Blackwell
>
> **备注:** 8 pages, 5 figures. Accepted at QR 2025 : 38th International Workshop on Qualitative Reasoning at IJCAI
>
> **摘要:** We investigate the abilities of 28 Large language Models (LLMs) to reason about cardinal directions (CDs) using a benchmark generated from a set of templates, extensively testing an LLM's ability to determine the correct CD given a particular scenario. The templates allow for a number of degrees of variation such as means of locomotion of the agent involved, and whether set in the first, second or third person. Even the newer Large Reasoning Models are unable to reliably determine the correct CD for all questions. This paper summarises and extends earlier work presented at COSIT-24.
>
---
#### [new 048] Translationese-index: Using Likelihood Ratios for Graded and Generalizable Measurement of Translationese
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量评估任务，旨在量化翻译腔（translationese）。通过构建T-index，利用语言模型的似然比进行度量，验证其有效性与通用性。**

- **链接: [http://arxiv.org/pdf/2507.12260v1](http://arxiv.org/pdf/2507.12260v1)**

> **作者:** Yikang Liu; Wanyang Zhang; Yiming Wang; Jialong Tang; Pei Zhang; Baosong Yang; Fei Huang; Rui Wang; Hai Hu
>
> **摘要:** In this paper, we propose the first quantitative measure for translationese -- the translationese-index (T-index) for graded and generalizable measurement of translationese, computed from the likelihood ratios of two contrastively fine-tuned language models (LMs). We use a synthesized dataset and a dataset with translations in the wild to evaluate T-index's generalizability in cross-domain settings and its validity against human judgments. Our results show that T-index is both robust and efficient. T-index scored by two 0.5B LMs fine-tuned on only 1-5k pairs of synthetic data can well capture translationese in the wild. We find that the relative differences in T-indices between translations can well predict pairwise translationese annotations obtained from human annotators; and the absolute values of T-indices correlate well with human ratings of degrees of translationese (Pearson's $r = 0.568$). Additionally, the correlation between T-index and existing machine translation (MT) quality estimation (QE) metrics such as BLEU and COMET is low, suggesting that T-index is not covered by these metrics and can serve as a complementary metric in MT QE.
>
---
#### [new 049] S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于主题建模任务，旨在解决VAE-NTMs中的后验崩溃问题。提出S2WTM模型，利用球面切片Wasserstein距离优化潜在空间结构。**

- **链接: [http://arxiv.org/pdf/2507.12451v1](http://arxiv.org/pdf/2507.12451v1)**

> **作者:** Suman Adhya; Debarshi Kumar Sanyal
>
> **备注:** Accepted as a long paper for ACL 2025 main conference
>
> **摘要:** Modeling latent representations in a hyperspherical space has proven effective for capturing directional similarities in high-dimensional text data, benefiting topic modeling. Variational autoencoder-based neural topic models (VAE-NTMs) commonly adopt the von Mises-Fisher prior to encode hyperspherical structure. However, VAE-NTMs often suffer from posterior collapse, where the KL divergence term in the objective function highly diminishes, leading to ineffective latent representations. To mitigate this issue while modeling hyperspherical structure in the latent space, we propose the Spherical Sliced Wasserstein Autoencoder for Topic Modeling (S2WTM). S2WTM employs a prior distribution supported on the unit hypersphere and leverages the Spherical Sliced-Wasserstein distance to align the aggregated posterior distribution with the prior. Experimental results demonstrate that S2WTM outperforms state-of-the-art topic models, generating more coherent and diverse topics while improving performance on downstream tasks.
>
---
#### [new 050] Developing Visual Augmented Q&A System using Scalable Vision Embedding Retrieval & Late Interaction Re-ranker
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于多模态问答任务，解决视觉信息检索效率与性能问题。通过混合搜索和晚期重排序提升检索效果，实现高效稳定的视觉增强问答系统。**

- **链接: [http://arxiv.org/pdf/2507.12378v1](http://arxiv.org/pdf/2507.12378v1)**

> **作者:** Rachna Saxena; Abhijeet Kumar; Suresh Shanmugam
>
> **备注:** Presented at NLP@IR workshop at SIGIR conference
>
> **摘要:** Traditional information extraction systems face challenges with text only language models as it does not consider infographics (visual elements of information) such as tables, charts, images etc. often used to convey complex information to readers. Multimodal LLM (MLLM) face challenges of finding needle in the haystack problem i.e., either longer context length or substantial number of documents as search space. Late interaction mechanism over visual language models has shown state of the art performance in retrieval-based vision augmented Q&A tasks. There are yet few challenges using it for RAG based multi-modal Q&A. Firstly, many popular and widely adopted vector databases do not support native multi-vector retrieval. Secondly, late interaction requires computation which inflates space footprint and can hinder enterprise adoption. Lastly, the current state of late interaction mechanism does not leverage the approximate neighbor search indexing methods for large speed ups in retrieval process. This paper explores a pragmatic approach to make vision retrieval process scalable and efficient without compromising on performance quality. We propose multi-step custom implementation utilizing widely adopted hybrid search (metadata & embedding) and state of the art late interaction re-ranker to retrieve best matching pages. Finally, MLLM are prompted as reader to generate answers from contextualized best matching pages. Through experiments, we observe that the proposed design is scalable (significant speed up) and stable (without degrading performance quality), hence can be used as production systems at enterprises.
>
---
#### [new 051] Nonlinear Concept Erasure: a Density Matching Approach
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于公平性任务，旨在消除神经网络中敏感信息。通过密度匹配方法实现概念擦除，有效降低模型偏差。**

- **链接: [http://arxiv.org/pdf/2507.12341v1](http://arxiv.org/pdf/2507.12341v1)**

> **作者:** Antoine Saillenfest; Pirmin Lemberger
>
> **备注:** 17 pages, 10 figures, accepted for publication in ECAI 2025 (28th European Conference on Artificial Intelligence)
>
> **摘要:** Ensuring that neural models used in real-world applications cannot infer sensitive information, such as demographic attributes like gender or race, from text representations is a critical challenge when fairness is a concern. We address this issue through concept erasure, a process that removes information related to a specific concept from distributed representations while preserving as much of the remaining semantic information as possible. Our approach involves learning an orthogonal projection in the embedding space, designed to make the class-conditional feature distributions of the discrete concept to erase indistinguishable after projection. By adjusting the rank of the projector, we control the extent of information removal, while its orthogonality ensures strict preservation of the local structure of the embeddings. Our method, termed $\overline{\mathrm{L}}$EOPARD, achieves state-of-the-art performance in nonlinear erasure of a discrete attribute on classic natural language processing benchmarks. Furthermore, we demonstrate that $\overline{\mathrm{L}}$EOPARD effectively mitigates bias in deep nonlinear classifiers, thereby promoting fairness.
>
---
#### [new 052] RiemannLoRA: A Unified Riemannian Framework for Ambiguity-Free LoRA Optimization
- **分类: cs.LG; cs.CL; cs.NA; math.DG; math.NA; 68T07, 65F55, 53Z50**

- **简介: 该论文属于参数高效微调任务，解决LoRA的初始化与过参数化问题。通过构建黎曼流形框架，提升优化效率和性能。**

- **链接: [http://arxiv.org/pdf/2507.12142v1](http://arxiv.org/pdf/2507.12142v1)**

> **作者:** Vladimir Bogachev; Vladimir Aletov; Alexander Molozhavenko; Denis Bobkov; Vera Soboleva; Aibek Alanov; Maxim Rakhuba
>
> **摘要:** Low-Rank Adaptation (LoRA) has become a widely adopted standard for parameter-efficient fine-tuning of large language models (LLMs), significantly reducing memory and computational demands. However, challenges remain, including finding optimal initialization strategies or mitigating overparametrization in low-rank matrix factorization. In this work, we propose a novel approach that addresses both of the challenges simultaneously within a unified framework. Our method treats a set of fixed-rank LoRA matrices as a smooth manifold. Considering adapters as elements on this manifold removes overparametrization, while determining the direction of the fastest loss decrease along the manifold provides initialization. Special care is taken to obtain numerically stable and computationally efficient implementation of our method, using best practices from numerical linear algebra and Riemannian optimization. Experimental results on LLM and diffusion model architectures demonstrate that RiemannLoRA consistently improves both convergence speed and final performance over standard LoRA and its state-of-the-art modifications.
>
---
#### [new 053] Fairness Is Not Enough: Auditing Competence and Intersectional Bias in AI-powered Resume Screening
- **分类: cs.CY; cs.AI; cs.CL; I.2.1; K.4.2; I.2.6; K.4.1**

- **简介: 该论文属于AI招聘工具评估任务，旨在解决AI在简历筛选中的公平性与能力问题。通过审计发现模型存在隐性偏见和评价能力不足，提出双维度验证框架。**

- **链接: [http://arxiv.org/pdf/2507.11548v1](http://arxiv.org/pdf/2507.11548v1)**

> **作者:** Kevin T Webster
>
> **备注:** 58 pages, 4 figures
>
> **摘要:** The increasing use of generative AI for resume screening is predicated on the assumption that it offers an unbiased alternative to biased human decision-making. However, this belief fails to address a critical question: are these AI systems fundamentally competent at the evaluative tasks they are meant to perform? This study investigates the question of competence through a two-part audit of eight major AI platforms. Experiment 1 confirmed complex, contextual racial and gender biases, with some models penalizing candidates merely for the presence of demographic signals. Experiment 2, which evaluated core competence, provided a critical insight: some models that appeared unbiased were, in fact, incapable of performing a substantive evaluation, relying instead on superficial keyword matching. This paper introduces the "Illusion of Neutrality" to describe this phenomenon, where an apparent lack of bias is merely a symptom of a model's inability to make meaningful judgments. This study recommends that organizations and regulators adopt a dual-validation framework, auditing AI hiring tools for both demographic bias and demonstrable competence to ensure they are both equitable and effective.
>
---
#### [new 054] MetaLint: Generalizable Idiomatic Code Quality Analysis through Instruction-Following and Easy-to-Hard Generalization
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文属于代码质量分析任务，旨在解决模型难以适应新代码规范的问题。通过指令微调和合成数据训练，提升模型对新型代码模式的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.11687v1](http://arxiv.org/pdf/2507.11687v1)**

> **作者:** Atharva Naik; Lawanya Baghel; Dhakshin Govindarajan; Darsh Agrawal; Daniel Fried; Carolyn Rose
>
> **摘要:** Large Language Models, though successful in code generation, struggle with code quality analysis because they are limited by static training data and can't easily adapt to evolving best practices. We introduce MetaLint, a new instruction-following framework that formulates code quality analysis as the task of detecting and fixing problematic semantic code fragments or code idioms based on high-level specifications. Unlike conventional approaches that train models on static, rule-based data, MetaLint employs instruction tuning on synthetic linter-generated data to support easy-to-hard generalization, enabling models to adapt to novel or complex code patterns without retraining. To evaluate this, we construct a benchmark of challenging idioms inspired by real-world coding standards such as Python Enhancement Proposals (PEPs) and assess whether MetaLint-trained models reason adaptively or simply memorize. Our results show that MetaLint improves generalization to unseen PEP idioms, achieving a 70.37% F-score on idiom detection with the highest recall (70.43%) among all evaluated models. It also achieves 26.73% on localization, competitive for its 4B parameter size and comparable to larger state-of-the-art models like o3-mini, highlighting its potential for future-proof code quality analysis.
>
---
#### [new 055] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于AI评估任务，解决MLLMs在评估过程中存在的同意偏差问题。通过提出SGV方法，提升模型的验证能力，显著提高任务完成率和准确性。**

- **链接: [http://arxiv.org/pdf/2507.11662v1](http://arxiv.org/pdf/2507.11662v1)**

> **作者:** Moises Andrade; Joonhyuk Cha; Brandon Ho; Vriksha Srihari; Karmesh Yadav; Zsolt Kira
>
> **备注:** Our code and data are publicly available at https://github.com/mshalimay/mllm-verifiers-abias-sgv
>
> **摘要:** Verifiers -- functions assigning rewards to agent behavior -- have been key for AI progress in domains like math and board games. However, extending these gains to domains without clear-cut success criteria (e.g.,computer use) remains a challenge: while humans can recognize suitable outcomes, translating this intuition into scalable rules is non-trivial. Multimodal Large Language Models(MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers of agent trajectories across web navigation, computer use, and robotic manipulation, and identify a critical limitation: agreement bias, a strong tendency for MLLMs to favor information in their context window, often generating chains of thought to rationalize flawed behavior. This bias is pervasive across models, resilient to test-time scaling, and can impact several methods using MLLMs as evaluators (e.g.,data filtering). Notably, it occurs despite MLLMs showing strong, human-aligned priors on desired behavior. To address this, we propose Self-Grounded Verification (SGV), a lightweight method that enables more effective use of MLLMs' knowledge and reasoning by harnessing their own sampling mechanisms via unconditional and conditional generation. SGV operates in two steps: first, the MLLM is elicited to retrieve broad priors about task completion, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Enhanced with SGV, MLLM verifiers show gains of up to 20 points in accuracy and failure detection rates, and can perform real-time supervision of heterogeneous agents, boosting task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena -- setting a new state of the art on the benchmark, surpassing the previous best by 48%.
>
---
#### [new 056] Jailbreak-Tuning: Models Efficiently Learn Jailbreak Susceptibility
- **分类: cs.CR; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI安全任务，研究模型在微调后易受攻击的问题，提出一种方法使模型无视安全机制，生成有害内容，强调现有模型的安全隐患。**

- **链接: [http://arxiv.org/pdf/2507.11630v1](http://arxiv.org/pdf/2507.11630v1)**

> **作者:** Brendan Murphy; Dillon Bowen; Shahrad Mohammadzadeh; Julius Broomfield; Adam Gleave; Kellin Pelrine
>
> **摘要:** AI systems are rapidly advancing in capability, and frontier model developers broadly acknowledge the need for safeguards against serious misuse. However, this paper demonstrates that fine-tuning, whether via open weights or closed fine-tuning APIs, can produce helpful-only models. In contrast to prior work which is blocked by modern moderation systems or achieved only partial removal of safeguards or degraded output quality, our jailbreak-tuning method teaches models to generate detailed, high-quality responses to arbitrary harmful requests. For example, OpenAI, Google, and Anthropic models will fully comply with requests for CBRN assistance, executing cyberattacks, and other criminal activity. We further show that backdoors can increase not only the stealth but also the severity of attacks, while stronger jailbreak prompts become even more effective in fine-tuning attacks, linking attack and potentially defenses in the input and weight spaces. Not only are these models vulnerable, more recent ones also appear to be becoming even more vulnerable to these attacks, underscoring the urgent need for tamper-resistant safeguards. Until such safeguards are discovered, companies and policymakers should view the release of any fine-tunable model as simultaneously releasing its evil twin: equally capable as the original model, and usable for any malicious purpose within its capabilities.
>
---
#### [new 057] MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出MERA Code，用于评估代码生成模型在多语言任务中的表现，解决现有评估忽视代码质量和实际应用的问题。**

- **链接: [http://arxiv.org/pdf/2507.12284v1](http://arxiv.org/pdf/2507.12284v1)**

> **作者:** Artem Chervyakov; Alexander Kharitonov; Pavel Zadorozhny; Adamenko Pavel; Rodion Levichev; Dmitrii Vorobev; Dmitrii Salikhov; Aidar Valeev; Alena Pestova; Maria Dziuba; Ilseyar Alimova; Artem Zavgorodnev; Aleksandr Medvedev; Stanislav Moiseev; Elena Bruches; Daniil Grebenkin; Roman Derunets; Vikulov Vladimir; Anton Emelyanov; Dmitrii Babaev; Vladimir V. Ivanov; Valentin Malykh; Alena Fenogenova
>
> **摘要:** Advancements in LLMs have enhanced task automation in software engineering; however, current evaluations primarily focus on natural language tasks, overlooking code quality. Most benchmarks prioritize high-level reasoning over executable code and real-world performance, leaving gaps in understanding true capabilities and risks associated with these models in production. To address this issue, we propose MERA Code, a new addition to the MERA benchmark family, specifically focused on evaluating code for the latest code generation LLMs in Russian. This benchmark includes 11 evaluation tasks that span 8 programming languages. Our proposed evaluation methodology features a taxonomy that outlines the practical coding skills necessary for models to complete these tasks. The benchmark comprises an open-source codebase for users to conduct MERA assessments, a scoring system compatible with various programming environments, and a platform featuring a leaderboard and submission system. We evaluate open LLMs and frontier API models, analyzing their limitations in terms of practical coding tasks in non-English languages. We are publicly releasing MERA to guide future research, anticipate groundbreaking features in model development, and standardize evaluation procedures.
>
---
#### [new 058] RUMAA: Repeat-Aware Unified Music Audio Analysis for Score-Performance Alignment, Transcription, and Mistake Detection
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于音乐音频分析任务，解决乐谱与演奏对齐、转录及错误检测问题。提出RUMAA框架，统一处理上述任务，提升重复结构下的性能。**

- **链接: [http://arxiv.org/pdf/2507.12175v1](http://arxiv.org/pdf/2507.12175v1)**

> **作者:** Sungkyun Chang; Simon Dixon; Emmanouil Benetos
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** This study introduces RUMAA, a transformer-based framework for music performance analysis that unifies score-to-performance alignment, score-informed transcription, and mistake detection in a near end-to-end manner. Unlike prior methods addressing these tasks separately, RUMAA integrates them using pre-trained score and audio encoders and a novel tri-stream decoder capturing task interdependencies through proxy tasks. It aligns human-readable MusicXML scores with repeat symbols to full-length performance audio, overcoming traditional MIDI-based methods that rely on manually unfolded score-MIDI data with pre-specified repeat structures. RUMAA matches state-of-the-art alignment methods on non-repeated scores and outperforms them on scores with repeats in a public piano music dataset, while also delivering promising transcription and mistake detection results.
>
---
#### [new 059] Simulated Language Acquisition in a Biologically Realistic Model of the Brain
- **分类: cs.NE; cs.CL**

- **简介: 该论文属于语言认知研究，旨在解释神经元活动如何产生语言能力。通过构建生物现实的神经模型，实现从零开始的语言学习与生成。**

- **链接: [http://arxiv.org/pdf/2507.11788v1](http://arxiv.org/pdf/2507.11788v1)**

> **作者:** Daniel Mitropolsky; Christos Papadimitriou
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Despite tremendous progress in neuroscience, we do not have a compelling narrative for the precise way whereby the spiking of neurons in our brain results in high-level cognitive phenomena such as planning and language. We introduce a simple mathematical formulation of six basic and broadly accepted principles of neuroscience: excitatory neurons, brain areas, random synapses, Hebbian plasticity, local inhibition, and inter-area inhibition. We implement a simulated neuromorphic system based on this formalism, which is capable of basic language acquisition: Starting from a tabula rasa, the system learns, in any language, the semantics of words, their syntactic role (verb versus noun), and the word order of the language, including the ability to generate novel sentences, through the exposure to a modest number of grounded sentences in the same language. We discuss several possible extensions and implications of this result.
>
---
## 更新

#### [replaced 001] Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16069v2](http://arxiv.org/pdf/2410.16069v2)**

> **作者:** Maggie Mi; Aline Villavicencio; Nafise Sadat Moosavi
>
> **备注:** ACL 2025
>
> **摘要:** Human processing of idioms relies on understanding the contextual sentences in which idioms occur, as well as language-intrinsic features such as frequency and speaker-intrinsic factors like familiarity. While LLMs have shown high performance on idiomaticity detection tasks, this success may be attributed to reasoning shortcuts in existing datasets. To this end, we construct a novel, controlled contrastive dataset designed to test whether LLMs can effectively use context to disambiguate idiomatic meaning. Additionally, we explore how collocational frequency and sentence probability influence model performance. Our findings reveal that LLMs often fail to resolve idiomaticity when it is required to attend to the surrounding context, and that models perform better on sentences that have higher likelihood. The collocational frequency of expressions also impacts performance. We make our code and dataset publicly available.
>
---
#### [replaced 002] NLP Meets the World: Toward Improving Conversations With the Public About Natural Language Processing Research
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10559v2](http://arxiv.org/pdf/2507.10559v2)**

> **作者:** Shomir Wilson
>
> **备注:** 7 pages
>
> **摘要:** Recent developments in large language models (LLMs) have been accompanied by rapidly growing public interest in natural language processing (NLP). This attention is reflected by major news venues, which sometimes invite NLP researchers to share their knowledge and views with a wide audience. Recognizing the opportunities of the present, for both the research field and for individual researchers, this paper shares recommendations for communicating with a general audience about the capabilities and limitations of NLP. These recommendations cover three themes: vague terminology as an obstacle to public understanding, unreasonable expectations as obstacles to sustainable growth, and ethical failures as obstacles to continued support. Published NLP research and popular news coverage are cited to illustrate these themes with examples. The recommendations promote effective, transparent communication with the general public about NLP, in order to strengthen public understanding and encourage support for research.
>
---
#### [replaced 003] CultureCLIP: Empowering CLIP with Cultural Awareness through Synthetic Images and Contextualized Captions
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06210v2](http://arxiv.org/pdf/2507.06210v2)**

> **作者:** Yuchen Huang; Zhiyuan Fan; Zhitao He; Sandeep Polisetty; Wenyan Li; Yi R. Fung
>
> **备注:** 25 pages, COLM 2025
>
> **摘要:** Pretrained vision-language models (VLMs) such as CLIP excel in general multimodal comprehension but often struggle to capture nuanced, context-dependent visual cues. This makes it difficult to distinguish between similar-looking concepts with potentially different cultural meanings. Such deficiencies are mainly due to a limited amount of high-quality cultural data, contextual information, and the lack of negative examples that highlight subtle differences. To mitigate this, we design a data curation pipeline leveraging open-sourced VLMs and text-to-image models to construct CulTwin, a synthetic cultural dataset. This dataset consists of paired concept-caption-image triplets, where concepts visually resemble each other but are culturally different. Then, we fine-tune CLIP on CulTwin to develop CultureCLIP, which aligns cultural concepts with contextually enhanced captions and synthetic images through tailored contrastive learning. Experiments on culture-specific benchmarks show that CultureCLIP outperforms the base CLIP, achieving up to a notable 5.49% improvement in fine-grained concept recognition on certain tasks while preserving CLIP's original generalization ability, validating the effectiveness of our data synthesis and VLM backbone training paradigm in capturing subtle cultural distinctions.
>
---
#### [replaced 004] Learning an Effective Premise Retrieval Model for Efficient Mathematical Formalization
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.13959v3](http://arxiv.org/pdf/2501.13959v3)**

> **作者:** Yicheng Tao; Haotian Liu; Shanwen Wang; Hongteng Xu
>
> **摘要:** Formalized mathematics has recently garnered significant attention for its ability to assist mathematicians across various fields. Premise retrieval, as a common step in mathematical formalization, has been a challenge, particularly for inexperienced users. Existing retrieval methods that facilitate natural language queries require a certain level of mathematical expertise from users, while approaches based on formal languages (e.g., Lean) typically struggle with the scarcity of training data, hindering the training of effective and generalizable retrieval models. In this work, we introduce a novel method that leverages data extracted from Mathlib to train a lightweight and effective premise retrieval model. In particular, the proposed model embeds queries (i.e., proof state provided by Lean) and premises in a latent space, featuring a tokenizer specifically trained on formal corpora. The model is learned in a contrastive learning framework, in which a fine-grained similarity calculation method and a re-ranking module are applied to enhance the retrieval performance. Experimental results demonstrate that our model outperforms existing baselines, achieving higher accuracy while maintaining a lower computational load. We have released an open-source search engine based on our retrieval model at https://premise-search.com/. The source code and the trained model can be found at https://github.com/ruc-ai4math/Premise-Retrieval.
>
---
#### [replaced 005] Semantic Adapter for Universal Text Embeddings: Diagnosing and Mitigating Negation Blindness to Enhance Universality
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00584v2](http://arxiv.org/pdf/2504.00584v2)**

> **作者:** Hongliu Cao
>
> **备注:** Accepted in ECAI 2025 main track
>
> **摘要:** Negation plays an important role in various natural language processing tasks such as Natural Language Inference and Sentiment Analysis tasks. Numerous prior studies have found that contextual text embedding models such as BERT, ELMO, RoBERTa or XLNet face challenges in accurately understanding negation. Recent advancements in universal text embeddings have demonstrated superior performance over contextual text embeddings in various tasks. However, due to the bias in popular evaluation benchmarks, the negation awareness capacity of these models remains unclear. To bridge the gap in existing literature, an in-depth analysis is initiated in this work to study the negation awareness of cutting-edge universal text embedding models. Our findings reveal a significant lack of negation awareness in these models, often interpreting negated text pairs as semantically similar. To efficiently deal with the conflict that different tasks need different trade-offs between topic and negation information among other semantic information, a data-efficient and computational-efficient embedding re-weighting method is proposed without modifying the parameters of text embedding models. The proposed solution is able to improve text embedding models' negation awareness significantly on both simple negation understanding task and complex negation understanding task. Furthermore, the proposed solution can also significantly improve the negation awareness of Large Language Model based task-specific high dimensional universal text embeddings.
>
---
#### [replaced 006] AKReF: An argumentative knowledge representation framework for structured argumentation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00713v3](http://arxiv.org/pdf/2506.00713v3)**

> **作者:** Debarati Bhattacharjee; Ashish Anand
>
> **备注:** 20 pages, 7 figures, 2 tables
>
> **摘要:** This paper presents a framework to convert argumentative texts into argument knowledge graphs (AKG). The proposed argumentative knowledge representation framework (AKReF) extends the theoretical foundation and enables the AKG to provide a graphical view of the argumentative structure that is easier to understand. Starting with basic annotations of argumentative components (ACs) and argumentative relations (ARs), we enrich the information by constructing a knowledge base (KB) graph with metadata attributes for nodes. Next, we apply modus ponens on premises and inference rules from the KB to form arguments. From these arguments, we create an AKG. The nodes and edges of the AKG have attributes capturing key argumentative features such as the type of premise (e.g., axiom, ordinary premise, assumption), the type of inference rule (e.g., strict, defeasible), preference order over defeasible rules, markers (e.g., "therefore", "however"), and the type of attack (e.g., undercut, rebuttal, undermining). We identify inference rules by locating a specific set of markers, called inference markers (IM). This, in turn, makes it possible to identify undercut attacks previously undetectable in existing datasets. AKG prepares the ground for reasoning tasks, including checking the coherence of arguments and identifying opportunities for revision. For this, it is essential to find indirect relations, many of which are implicit. Our proposed AKG format, with annotated inference rules and modus ponens, helps reasoning models learn the implicit, indirect relations that require inference over arguments and their interconnections. We use an essay from the AAEC dataset to illustrate the framework. We further show its application in complex analyses such as extracting a conflict-free set and a maximal set of admissible arguments.
>
---
#### [replaced 007] Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.09477v2](http://arxiv.org/pdf/2507.09477v2)**

> **作者:** Yangning Li; Weizhi Zhang; Yuyao Yang; Wei-Chieh Huang; Yaozu Wu; Junyu Luo; Yuanchen Bei; Henry Peng Zou; Xiao Luo; Yusheng Zhao; Chunkit Chan; Yankai Chen; Zhongfen Deng; Yinghui Li; Hai-Tao Zheng; Dongyuan Li; Renhe Jiang; Ming Zhang; Yangqiu Song; Philip S. Yu
>
> **备注:** submitted to ARR May
>
> **摘要:** Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at https://github.com/DavidZWZ/Awesome-RAG-Reasoning.
>
---
#### [replaced 008] Simple Mechanistic Explanations for Out-Of-Context Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.08218v2](http://arxiv.org/pdf/2507.08218v2)**

> **作者:** Atticus Wang; Joshua Engels; Oliver Clive-Griffin; Senthooran Rajamanoharan; Neel Nanda
>
> **备注:** ICML 2025 Workshop R2-FM
>
> **摘要:** Out-of-context reasoning (OOCR) is a phenomenon in which fine-tuned LLMs exhibit surprisingly deep out-of-distribution generalization. Rather than learning shallow heuristics, they implicitly internalize and act on the consequences of observations scattered throughout the fine-tuning data. In this work, we investigate this phenomenon mechanistically and find that many instances of OOCR in the literature have a simple explanation: the LoRA fine-tuning essentially adds a constant steering vector, steering the model towards a general concept. This improves performance on the fine-tuning task and in many other concept-related domains, causing the surprising generalization. Moreover, we can directly train steering vectors for these tasks from scratch, which also induces OOCR. We find that our results hold even for a task that seems like it must involve conditional behavior (model backdoors); it turns out that unconditionally adding a steering vector is sufficient. Overall, our work presents one explanation of what gets learned during fine-tuning for OOCR tasks, contributing to the key question of why LLMs can reason out of context, an advanced capability that is highly relevant to their safe and reliable deployment.
>
---
#### [replaced 009] FADE: Why Bad Descriptions Happen to Good Features
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16994v2](http://arxiv.org/pdf/2502.16994v2)**

> **作者:** Bruno Puri; Aakriti Jain; Elena Golimblevskaia; Patrick Kahardipraja; Thomas Wiegand; Wojciech Samek; Sebastian Lapuschkin
>
> **摘要:** Recent advances in mechanistic interpretability have highlighted the potential of automating interpretability pipelines in analyzing the latent representations within LLMs. While this may enhance our understanding of internal mechanisms, the field lacks standardized evaluation methods for assessing the validity of discovered features. We attempt to bridge this gap by introducing FADE: Feature Alignment to Description Evaluation, a scalable model-agnostic framework for automatically evaluating feature-to-description alignment. FADE evaluates alignment across four key metrics - Clarity, Responsiveness, Purity, and Faithfulness - and systematically quantifies the causes of the misalignment between features and their descriptions. We apply FADE to analyze existing open-source feature descriptions and assess key components of automated interpretability pipelines, aiming to enhance the quality of descriptions. Our findings highlight fundamental challenges in generating feature descriptions, particularly for SAEs compared to MLP neurons, providing insights into the limitations and future directions of automated interpretability. We release FADE as an open-source package at: https://github.com/brunibrun/FADE
>
---
#### [replaced 010] Learning to Reason at the Frontier of Learnability
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12272v4](http://arxiv.org/pdf/2502.12272v4)**

> **作者:** Thomas Foster; Jakob Foerster
>
> **摘要:** Reinforcement learning is now widely adopted as the final stage of large language model training, especially for reasoning-style tasks such as maths problems. Typically, models attempt each question many times during a single training step and attempt to learn from their successes and failures. However, we demonstrate that throughout training with two popular algorithms (PPO and VinePPO) on two widely used datasets, many questions are either solved by all attempts - meaning they are already learned - or by none - providing no meaningful training signal. To address this, we adapt a method from the reinforcement learning literature - sampling for learnability - and apply it to the reinforcement learning stage of LLM training. Our curriculum prioritises questions with high variance of success, i.e. those where the agent sometimes succeeds, but not always. Our findings demonstrate that this curriculum consistently boosts training performance across multiple algorithms and datasets, paving the way for more efficient and effective reinforcement learning with LLMs.
>
---
#### [replaced 011] Resona: Improving Context Copying in Linear Recurrence Models with Retrieval
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22913v2](http://arxiv.org/pdf/2503.22913v2)**

> **作者:** Xinyu Wang; Linrui Ma; Jerry Huang; Peng Lu; Prasanna Parthasarathi; Xiao-Wen Chang; Boxing Chen; Yufei Cui
>
> **备注:** Comments: Accepted at COLM 2025 (Conference on Learning with Machines)
>
> **摘要:** Recent shifts in the space of large language model (LLM) research have shown an increasing focus on novel architectures to compete with prototypical Transformer-based models that have long dominated this space. Linear recurrent models have proven to be a viable competitor due to their computational efficiency. However, such models still demonstrate a sizable gap compared to Transformers in terms of in-context learning among other tasks that require recalling information from a context. In this work, we introduce Resona, a simple and scalable framework for augmenting linear recurrent models with retrieval. Resona augments models with the ability to integrate retrieved information from the provided input context, enabling tailored behavior to diverse task requirements. Experiments on a variety of linear recurrent models demonstrate that Resona-augmented models observe significant performance gains on a variety of synthetic as well as real-world natural language tasks, highlighting its ability to act as a general purpose method to improve the in-context learning and language modeling abilities of linear recurrent LLMs.
>
---
#### [replaced 012] Reasoning Strategies in Large Language Models: Can They Follow, Prefer, and Optimize?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11423v2](http://arxiv.org/pdf/2507.11423v2)**

> **作者:** Yanjian Zhang; Guillaume Wisniewski; Nadi Tomeh; Thierry Charnois
>
> **摘要:** Human reasoning involves different strategies, each suited to specific problems. Prior work shows that large language model (LLMs) tend to favor a single reasoning strategy, potentially limiting their effectiveness in diverse reasoning challenges. In this work, we investigate whether prompting can control LLMs reasoning strategies and assess its impact on logical problem-solving. While our experiments show that no single strategy consistently improves accuracy, performance could be enhanced if models could adaptively choose the optimal strategy. We propose methods to guide LLMs in strategy selection, highlighting new ways to refine their reasoning abilities.
>
---
#### [replaced 013] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling
- **分类: cs.LG; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2503.02445v5](http://arxiv.org/pdf/2503.02445v5)**

> **作者:** Hao Li; Yu-Hao Huang; Chang Xu; Viktor Schlegel; Renhe Jiang; Riza Batista-Navarro; Goran Nenadic; Jiang Bian
>
> **备注:** ICML 2025 Main Conference
>
> **摘要:** Time-series Generation (TSG) is a prominent research area with broad applications in simulations, data augmentation, and counterfactual analysis. While existing methods have shown promise in unconditional single-domain TSG, real-world applications demand for cross-domain approaches capable of controlled generation tailored to domain-specific constraints and instance-level requirements. In this paper, we argue that text can provide semantic insights, domain information and instance-specific temporal patterns, to guide and improve TSG. We introduce ``Text-Controlled TSG'', a task focused on generating realistic time series by incorporating textual descriptions. To address data scarcity in this setting, we propose a novel LLM-based Multi-Agent framework that synthesizes diverse, realistic text-to-TS datasets. Furthermore, we introduce BRIDGE, a hybrid text-controlled TSG framework that integrates semantic prototypes with text description for supporting domain-level guidance. This approach achieves state-of-the-art generation fidelity on 11 of 12 datasets, and improves controllability by up to 12% on MSE and 6% MAE compared to no text input generation, highlighting its potential for generating tailored time-series data.
>
---
#### [replaced 014] Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15670v3](http://arxiv.org/pdf/2505.15670v3)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
#### [replaced 015] A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09037v2](http://arxiv.org/pdf/2504.09037v2)**

> **作者:** Zixuan Ke; Fangkai Jiao; Yifei Ming; Xuan-Phi Nguyen; Austin Xu; Do Xuan Long; Minzhi Li; Chengwei Qin; Peifeng Wang; Silvio Savarese; Caiming Xiong; Shafiq Joty
>
> **备注:** 72 pages, 6 figures
>
> **摘要:** Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
>
---
#### [replaced 016] Towards Geo-Culturally Grounded LLM Generations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13497v4](http://arxiv.org/pdf/2502.13497v4)**

> **作者:** Piyawat Lertvittayakumjorn; David Kinney; Vinodkumar Prabhakaran; Donald Martin Jr.; Sunipa Dev
>
> **备注:** ACL 2025 (main conference)
>
> **摘要:** Generative large language models (LLMs) have demonstrated gaps in diverse cultural awareness across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on LLMs' ability to display familiarity with various national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on multiple cultural awareness benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., cultural norms, artifacts, and institutions), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models and fails to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional cultural knowledge and open-ended cultural fluency when it comes to evaluating LLMs' cultural awareness.
>
---
#### [replaced 017] Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04457v3](http://arxiv.org/pdf/2505.04457v3)**

> **作者:** Shigeki Karita; Yuma Koizumi; Heiga Zen; Haruko Ishikawa; Robin Scheibler; Michiel Bacchiani
>
> **备注:** Accepted to IEEE WASPAA2025
>
> **摘要:** Training data cleaning is a new application for generative model-based speech restoration (SR). This paper introduces Miipher-2, an SR model designed for million-hour scale data, for training data cleaning for large-scale generative models like large language models. Key challenges addressed include generalization to unseen languages, operation without explicit conditioning (e.g., text, speaker ID), and computational efficiency. Miipher-2 utilizes a frozen, pre-trained Universal Speech Model (USM), supporting over 300 languages, as a robust, conditioning-free feature extractor. To optimize efficiency and minimize memory, Miipher-2 incorporates parallel adapters for predicting clean USM features from noisy inputs and employs the WaveFit neural vocoder for waveform synthesis. These components were trained on 3,000 hours of multi-lingual, studio-quality recordings with augmented degradations, while USM parameters remained fixed. Experimental results demonstrate Miipher-2's superior or comparable performance to conventional SR models in word-error-rate, speaker similarity, and both objective and subjective sound quality scores across all tested languages. Miipher-2 operates efficiently on consumer-grade accelerators, achieving a real-time factor of 0.0078, enabling the processing of a million-hour speech dataset in approximately three days using only 100 such accelerators.
>
---
#### [replaced 018] Journalism-Guided Agentic In-Context Learning for News Stance Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11049v2](http://arxiv.org/pdf/2507.11049v2)**

> **作者:** Dahyun Lee; Jonghyeon Choi; Jiyoung Han; Kunwoo Park
>
> **备注:** Preprint. 24 pages
>
> **摘要:** As online news consumption grows, personalized recommendation systems have become integral to digital journalism. However, these systems risk reinforcing filter bubbles and political polarization by failing to incorporate diverse perspectives. Stance detection -- identifying a text's position on a target -- can help mitigate this by enabling viewpoint-aware recommendations and data-driven analyses of media bias. Yet, existing stance detection research remains largely limited to short texts and high-resource languages. To address these gaps, we introduce \textsc{K-News-Stance}, the first Korean dataset for article-level stance detection, comprising 2,000 news articles with article-level and 19,650 segment-level stance annotations across 47 societal issues. We also propose \textsc{JoA-ICL}, a \textbf{Jo}urnalism-guided \textbf{A}gentic \textbf{I}n-\textbf{C}ontext \textbf{L}earning framework that employs a language model agent to predict the stances of key structural segments (e.g., leads, quotes), which are then aggregated to infer the overall article stance. Experiments show that \textsc{JoA-ICL} outperforms existing stance detection methods, highlighting the benefits of segment-level agency in capturing the overall position of long-form news articles. Two case studies further demonstrate its broader utility in promoting viewpoint diversity in news recommendations and uncovering patterns of media bias.
>
---
#### [replaced 019] Labels Generated by Large Language Models Help Measure People's Empathy in Vitro
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00691v2](http://arxiv.org/pdf/2501.00691v2)**

> **作者:** Md Rakibul Hasan; Yue Yao; Md Zakir Hossain; Aneesh Krishna; Imre Rudas; Shafin Rahman; Tom Gedeon
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Large language models (LLMs) have revolutionised many fields, with LLM-as-a-service (LLMSaaS) offering accessible, general-purpose solutions without costly task-specific training. In contrast to the widely studied prompt engineering for directly solving tasks (in vivo), this paper explores LLMs' potential for in-vitro applications: using LLM-generated labels to improve supervised training of mainstream models. We examine two strategies - (1) noisy label correction and (2) training data augmentation - in empathy computing, an emerging task to predict psychology-based questionnaire outcomes from inputs like textual narratives. Crowdsourced datasets in this domain often suffer from noisy labels that misrepresent underlying empathy. We show that replacing or supplementing these crowdsourced labels with LLM-generated labels, developed using psychology-based scale-aware prompts, achieves statistically significant accuracy improvements. Notably, the RoBERTa pre-trained language model (PLM) trained with noise-reduced labels yields a state-of-the-art Pearson correlation coefficient of 0.648 on the public NewsEmp benchmarks. This paper further analyses evaluation metric selection and demographic biases to help guide the future development of more equitable empathy computing models. Code and LLM-generated labels are available at https://github.com/hasan-rakibul/LLMPathy.
>
---
#### [replaced 020] Large Language Models Often Know When They Are Being Evaluated
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23836v3](http://arxiv.org/pdf/2505.23836v3)**

> **作者:** Joe Needham; Giles Edkins; Govind Pimpale; Henning Bartsch; Marius Hobbhahn
>
> **摘要:** If AI models can detect when they are being evaluated, the effectiveness of evaluations might be compromised. For example, models could have systematically different behavior during evaluations, leading to less reliable benchmarks for deployment and governance decisions. We investigate whether frontier language models can accurately classify transcripts based on whether they originate from evaluations or real-world deployment, a capability we call evaluation awareness. To achieve this, we construct a diverse benchmark of 1,000 prompts and transcripts from 61 distinct datasets. These span public benchmarks (e.g., MMLU, SWEBench), real-world deployment interactions, and agent trajectories from scaffolding frameworks (e.g., web-browsing agents). Frontier models clearly demonstrate above-random evaluation awareness (Gemini-2.5-Pro reaches an AUC of $0.83$), but do not yet surpass our simple human baseline (AUC of $0.92$). Furthermore, both AI models and humans are better at identifying evaluations in agentic settings compared to chat settings. Additionally, we test whether models can identify the purpose of the evaluation. Under multiple-choice and open-ended questioning, AI models far outperform random chance in identifying what an evaluation is testing for. Our results indicate that frontier models already exhibit a substantial, though not yet superhuman, level of evaluation-awareness. We recommend tracking this capability in future models.
>
---
#### [replaced 021] TD-EVAL: Revisiting Task-Oriented Dialogue Evaluation by Combining Turn-Level Precision with Dialogue-Level Comparisons
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.19982v2](http://arxiv.org/pdf/2504.19982v2)**

> **作者:** Emre Can Acikgoz; Carl Guo; Suvodip Dey; Akul Datta; Takyoung Kim; Gokhan Tur; Dilek Hakkani-Tür
>
> **摘要:** Task-oriented dialogue (TOD) systems are experiencing a revolution driven by Large Language Models (LLMs), yet the evaluation methodologies for these systems remain insufficient for their growing sophistication. While traditional automatic metrics effectively assessed earlier modular systems, they focus solely on the dialogue level and cannot detect critical intermediate errors that can arise during user-agent interactions. In this paper, we introduce TD-EVAL (Turn and Dialogue-level Evaluation), a two-step evaluation framework that unifies fine-grained turn-level analysis with holistic dialogue-level comparisons. At turn level, we evaluate each response along three TOD-specific dimensions: conversation cohesion, backend knowledge consistency, and policy compliance. Meanwhile, we design TOD Agent Arena that uses pairwise comparisons to provide a measure of dialogue-level quality. Through experiments on MultiWOZ 2.4 and {\tau}-Bench, we demonstrate that TD-EVAL effectively identifies the conversational errors that conventional metrics miss. Furthermore, TD-EVAL exhibits better alignment with human judgments than traditional and LLM-based metrics. These findings demonstrate that TD-EVAL introduces a new paradigm for TOD system evaluation, efficiently assessing both turn and system levels with a plug-and-play framework for future research.
>
---
#### [replaced 022] METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation
- **分类: cs.LG; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.10543v2](http://arxiv.org/pdf/2412.10543v2)**

> **作者:** Siddhant Ray; Rui Pan; Zhuohan Gu; Kuntai Du; Shaoting Feng; Ganesh Ananthanarayanan; Ravi Netravali; Junchen Jiang
>
> **备注:** 17 pages, 18 figures
>
> **摘要:** RAG (Retrieval Augmented Generation) allows LLMs (large language models) to generate better responses with external knowledge, but using more external knowledge often improves generation quality at the expense of response delay. Prior work either reduces the response delay (through better scheduling of RAG queries) or strives to maximize quality (which involves tuning the RAG workflow), but they fall short in optimizing the tradeoff between the delay and quality of RAG responses. This paper presents METIS, the first RAG system that jointly schedules queries and adapts the key RAG configurations of each query, such as the number of retrieved text chunks and synthesis methods, in order to balance quality optimization and response delay reduction. Using 4 popular RAG-QA datasets, we show that compared with the state-of-the-art RAG optimization schemes, METIS reduces the generation latency by $1.64-2.54\times$ without sacrificing generation quality.
>
---
#### [replaced 023] Linearly-Interpretable Concept Embedding Models for Text Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.14335v2](http://arxiv.org/pdf/2406.14335v2)**

> **作者:** Francesco De Santis; Philippe Bich; Gabriele Ciravegna; Pietro Barbiero; Danilo Giordano; Tania Cerquitelli
>
> **摘要:** Despite their success, Large-Language Models (LLMs) still face criticism due to their lack of interpretability. Traditional post-hoc interpretation methods, based on attention and gradient-based analysis, offer limited insights as they only approximate the model's decision-making processes and have been proved to be unreliable. For this reason, Concept-Bottleneck Models (CBMs) have been lately proposed in the textual field to provide interpretable predictions based on human-understandable concepts. However, CBMs still exhibit several limitations due to their architectural constraints limiting their expressivity, to the absence of task-interpretability when employing non-linear task predictors and for requiring extensive annotations that are impractical for real-world text data. In this paper, we address these challenges by proposing a novel Linearly Interpretable Concept Embedding Model (LICEM) going beyond the current accuracy-interpretability trade-off. LICEMs classification accuracy is better than existing interpretable models and matches black-box ones. We show that the explanations provided by our models are more interveneable and causally consistent with respect to existing solutions. Finally, we show that LICEMs can be trained without requiring any concept supervision, as concepts can be automatically predicted when using an LLM backbone.
>
---
#### [replaced 024] DEEPER Insight into Your User: Directed Persona Refinement for Dynamic Persona Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11078v2](http://arxiv.org/pdf/2502.11078v2)**

> **作者:** Aili Chen; Chengyu Du; Jiangjie Chen; Jinghan Xu; Yikai Zhang; Siyu Yuan; Zulong Chen; Liangyue Li; Yanghua Xiao
>
> **摘要:** To advance personalized applications such as recommendation systems and user behavior prediction, recent research increasingly adopts large language models (LLMs) for human -readable persona modeling. In dynamic real -world scenarios, effective persona modeling necessitates leveraging streaming behavior data to continually optimize user personas. However, existing methods -whether regenerating personas or incrementally extending them with new behaviors -often fail to achieve sustained improvements in persona quality or future behavior prediction accuracy. To address this, we propose DEEPER, a novel approach for dynamic persona modeling that enables continual persona optimization. Specifically, we enhance the model's direction -search capability through an iterative reinforcement learning framework, allowing it to automatically identify effective update directions and optimize personas using discrepancies between user behaviors and model predictions. Extensive experiments on dynamic persona modeling involving 4800 users across 10 domains highlight the superior persona optimization capabilities of DEEPER, delivering an impressive 32.2% average reduction in user behavior prediction error over four update rounds -outperforming the best baseline by a remarkable 22.92%.
>
---
#### [replaced 025] From Semantic Web and MAS to Agentic AI: A Unified Narrative of the Web of Agents
- **分类: cs.AI; cs.CL; cs.CR; cs.HC; cs.MA; I.2.11; I.2.7; C.2.4; K.6.5; I.2.4**

- **链接: [http://arxiv.org/pdf/2507.10644v2](http://arxiv.org/pdf/2507.10644v2)**

> **作者:** Tatiana Petrova; Boris Bliznioukov; Aleksandr Puzikov; Radu State
>
> **备注:** 33 pages, 9 figures, 8 tables
>
> **摘要:** The concept of the Web of Agents (WoA), which transforms the static, document-centric Web into an environment of autonomous agents acting on users' behalf, has attracted growing interest as large language models (LLMs) become more capable. However, research in this area is still fragmented across different communities. Contemporary surveys catalog the latest LLM-powered frameworks, while the rich histories of Multi-Agent Systems (MAS) and the Semantic Web are often treated as separate, legacy domains. This fragmentation obscures the intellectual lineage of modern systems and hinders a holistic understanding of the field's trajectory. We present the first comprehensive evolutionary overview of the WoA. We show that modern protocols like A2A and the MCP, are direct evolutionary responses to the well-documented limitations of earlier standards like FIPA standards and OWL-based semantic agents. To systematize this analysis, we introduce a four-axis taxonomy (semantic foundation, communication paradigm, locus of intelligence, discovery mechanism). This framework provides a unified analytical lens for comparing agent architectures across all generations, revealing a clear line of descent where others have seen a disconnect. Our analysis identifies a paradigm shift in the 'locus of intelligence': from being encoded in external data (Semantic Web) or the platform (MAS) to being embedded within the agent's core model (LLM). This shift is foundational to modern Agentic AI, enabling the scalable and adaptive systems the WoA has long envisioned. We conclude that while new protocols are essential, they are insufficient for building a robust, open, trustworthy ecosystem. Finally, we argue that the next research frontier lies in solving persistent socio-technical challenges, and we map out a new agenda focused on decentralized identity, economic models, security, and governance for the emerging WoA.
>
---
#### [replaced 026] Protecting Copyrighted Material with Unique Identifiers in Large Language Model Training
- **分类: cs.CL; cs.CR; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.15740v3](http://arxiv.org/pdf/2403.15740v3)**

> **作者:** Shuai Zhao; Linchao Zhu; Ruijie Quan; Yi Yang
>
> **备注:** A technical report, work mainly done in the early of 2024
>
> **摘要:** A primary concern regarding training large language models (LLMs) is whether they abuse copyrighted online text. With the increasing training data scale and the prevalence of LLMs in daily lives, two problems arise: \textbf{1)} false positive membership inference results misled by similar examples; \textbf{2)} membership inference methods are usually too complex for end users to understand and use. To address these issues, we propose an alternative \textit{insert-and-detect} methodology, advocating that web users and content platforms employ \textbf{\textit{unique identifiers}} for reliable and independent membership inference. Users and platforms can create their identifiers, embed them in copyrighted text, and independently detect them in future LLMs. As an initial demonstration, we introduce \textit{\textbf{ghost sentences}} and a user-friendly last-$k$ words test, allowing end users to chat with LLMs for membership inference. Ghost sentences consist primarily of unique passphrases of random natural words, which can come with customized elements to bypass possible filter rules. The last-$k$ words test requires a significant repetition time of ghost sentences~($\ge10$). For cases with fewer repetitions, we designed an extra perplexity test, as LLMs exhibit high perplexity when encountering unnatural passphrases. We also conduct a comprehensive study on the memorization and membership inference of ghost sentences, examining factors such as training data scales, model sizes, repetition times, insertion positions, wordlist of passphrases, alignment, \textit{etc}. Our study shows the possibility of applying ghost sentences in real scenarios and provides instructions for the potential application.
>
---
#### [replaced 027] Planning-Aware Code Infilling via Horizon-Length Prediction
- **分类: cs.LG; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.03103v3](http://arxiv.org/pdf/2410.03103v3)**

> **作者:** Yifeng Ding; Hantian Ding; Shiqi Wang; Qing Sun; Varun Kumar; Zijian Wang
>
> **摘要:** Fill-in-the-Middle (FIM), or infilling, has become integral to code language models, enabling generation of missing code given both left and right contexts. However, the current FIM training paradigm which performs next-token prediction (NTP) over reordered sequence often leads to models struggling to generate content that aligns well with the surrounding context. We hypothesize that NTP alone is insufficient for models to learn effective planning conditioned on the distant right context, a critical factor for successful code infilling. To overcome this, we propose Horizon-Length Prediction (HLP), a novel training objective that teaches models to predict the number of remaining middle tokens at each step. HLP advances FIM with lookahead planning, enabling models to inherently learn infilling boundaries for arbitrary left and right contexts without relying on dataset-specific post-processing. Our evaluation across different model families and sizes shows that HLP significantly improves FIM performance by up to 24% relatively on diverse benchmarks, across file-level and repository-level. Furthermore, the enhanced planning capability gained through HLP boosts model performance on code reasoning. Importantly, HLP incurs negligible training overhead and no additional inference cost, ensuring its practicality for real-world scenarios.
>
---
#### [replaced 028] Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and Large Language Model Knowledge
- **分类: cs.CL; cs.AI; cs.DL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.11330v2](http://arxiv.org/pdf/2507.11330v2)**

> **作者:** Wenqing Wu; Chengzhi Zhang; Yi Zhao
>
> **备注:** Journal of the Association for Information Science and Technology, 2025
>
> **摘要:** Novelty is a crucial criterion in the peer review process for evaluating academic papers. Traditionally, it's judged by experts or measure by unique reference combinations. Both methods have limitations: experts have limited knowledge, and the effectiveness of the combination method is uncertain. Moreover, it's unclear if unique citations truly measure novelty. The large language model (LLM) possesses a wealth of knowledge, while human experts possess judgment abilities that the LLM does not possess. Therefore, our research integrates the knowledge and abilities of LLM and human experts to address the limitations of novelty assessment. One of the most common types of novelty in academic papers is the introduction of new methods. In this paper, we propose leveraging human knowledge and LLM to assist pretrained language models (PLMs, e.g. BERT etc.) in predicting the method novelty of papers. Specifically, we extract sentences related to the novelty of the academic paper from peer review reports and use LLM to summarize the methodology section of the academic paper, which are then used to fine-tune PLMs. In addition, we have designed a text-guided fusion module with novel Sparse-Attention to better integrate human and LLM knowledge. We compared the method we proposed with a large number of baselines. Extensive experiments demonstrate that our method achieves superior performance.
>
---
#### [replaced 029] Generative Emergent Communication: Large Language Model is a Collective World Model
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.00226v2](http://arxiv.org/pdf/2501.00226v2)**

> **作者:** Tadahiro Taniguchi; Ryo Ueda; Tomoaki Nakamura; Masahiro Suzuki; Akira Taniguchi
>
> **摘要:** Large Language Models (LLMs) have demonstrated a remarkable ability to capture extensive world knowledge, yet how this is achieved without direct sensorimotor experience remains a fundamental puzzle. This study proposes a novel theoretical solution by introducing the Collective World Model hypothesis. We argue that an LLM does not learn a world model from scratch; instead, it learns a statistical approximation of a collective world model that is already implicitly encoded in human language through a society-wide process of embodied, interactive sense-making. To formalize this process, we introduce generative emergent communication (Generative EmCom), a framework built on the Collective Predictive Coding (CPC). This framework models the emergence of language as a process of decentralized Bayesian inference over the internal states of multiple agents. We argue that this process effectively creates an encoder-decoder structure at a societal scale: human society collectively encodes its grounded, internal representations into language, and an LLM subsequently decodes these symbols to reconstruct a latent space that mirrors the structure of the original collective representations. This perspective provides a principled, mathematical explanation for how LLMs acquire their capabilities. The main contributions of this paper are: 1) the formalization of the Generative EmCom framework, clarifying its connection to world models and multi-agent reinforcement learning, and 2) its application to interpret LLMs, explaining phenomena such as distributional semantics as a natural consequence of representation reconstruction. This work provides a unified theory that bridges individual cognitive development, collective language evolution, and the foundations of large-scale AI.
>
---
#### [replaced 030] Organize the Web: Constructing Domains Enhances Pre-Training Data Curation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10341v3](http://arxiv.org/pdf/2502.10341v3)**

> **作者:** Alexander Wettig; Kyle Lo; Sewon Min; Hannaneh Hajishirzi; Danqi Chen; Luca Soldaini
>
> **备注:** Accepted at ICML 2025. Project page: https://weborganizer.allen.ai
>
> **摘要:** Modern language models are trained on large, unstructured datasets consisting of trillions of tokens and obtained by crawling the web. The unstructured nature makes it difficult to reason about their contents and develop systematic approaches to data curation. In this paper, we unpack monolithic web corpora by developing taxonomies of their contents and organizing them into domains. We introduce WebOrganizer, a framework for organizing web pages in terms of both their topic and format. Using these two complementary notions of domains, we automatically annotate pre-training data by distilling annotations from a large language model into efficient classifiers. This allows us to study how data from different domains should be mixed to improve models on downstream tasks, and we show that we can combine insights about effective topics and formats to further boost performance. We demonstrate that our domain mixing also improves existing methods that select data based on quality. Furthermore, we study and compare how quality-based methods will implicitly change the domain mixture. Overall, our work demonstrates that constructing and mixing domains provides a valuable complement to quality-based data curation methods, opening new avenues for effective and insightful pre-training data curation.
>
---
#### [replaced 031] Flexible and Efficient Grammar-Constrained Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.05111v2](http://arxiv.org/pdf/2502.05111v2)**

> **作者:** Kanghee Park; Timothy Zhou; Loris D'Antoni
>
> **摘要:** Large Language Models (LLMs) are often asked to generate structured outputs that obey precise syntactic rules, such as code snippets or formatted data. Grammar-constrained decoding (GCD) can guarantee that LLM outputs matches such rules by masking out tokens that will provably lead to outputs that do not belong to a specified context-free grammar (CFG). To guarantee soundness, GCD algorithms have to compute how a given LLM subword tokenizer can align with the tokens used by a given context-free grammar and compute token masks based on this information. Doing so efficiently is challenging and existing GCD algorithms require tens of minutes to preprocess common grammars. We present a new GCD algorithm together with an implementation that offers 17.71x faster offline preprocessing than existing approaches while preserving state-of-the-art efficiency in online mask computation.
>
---
#### [replaced 032] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06607v2](http://arxiv.org/pdf/2507.06607v2)**

> **作者:** Liliang Ren; Congcong Chen; Haoran Xu; Young Jin Kim; Adam Atkinson; Zheng Zhan; Jiankai Sun; Baolin Peng; Liyuan Liu; Shuohang Wang; Hao Cheng; Jianfeng Gao; Weizhu Chen; Yelong Shen
>
> **摘要:** Recent advances in language modeling have demonstrated the effectiveness of State Space Models (SSMs) for efficient sequence modeling. While hybrid architectures such as Samba and the decoder-decoder architecture, YOCO, have shown promising performance gains over Transformers, prior works have not investigated the efficiency potential of representation sharing between SSM layers. In this paper, we introduce the Gated Memory Unit (GMU), a simple yet effective mechanism for efficient memory sharing across layers. We apply it to create SambaY, a decoder-hybrid-decoder architecture that incorporates GMUs in the cross-decoder to share memory readout states from a Samba-based self-decoder. SambaY significantly enhances decoding efficiency, preserves linear pre-filling time complexity, and boosts long-context performance, all while eliminating the need for explicit positional encoding. Through extensive scaling experiments, we demonstrate that our model exhibits a significantly lower irreducible loss compared to a strong YOCO baseline, indicating superior performance scalability under large-scale compute regimes. Our largest model enhanced with Differential Attention, Phi4-mini-Flash-Reasoning, achieves significantly better performance than Phi4-mini-Reasoning on reasoning tasks such as Math500, AIME24/25, and GPQA Diamond without any reinforcement learning, while delivering up to 10x higher decoding throughput on 2K-length prompts with 32K generation length under the vLLM inference framework. We release our training codebase on open-source data at https://github.com/microsoft/ArchScale.
>
---
#### [replaced 033] ReviewAgents: Bridging the Gap Between Human and AI-Generated Paper Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.08506v3](http://arxiv.org/pdf/2503.08506v3)**

> **作者:** Xian Gao; Jiacheng Ruan; Zongyun Zhang; Jingsheng Gao; Ting Liu; Yuzhuo Fu
>
> **备注:** Work in progress
>
> **摘要:** Academic paper review is a critical yet time-consuming task within the research community. With the increasing volume of academic publications, automating the review process has become a significant challenge. The primary issue lies in generating comprehensive, accurate, and reasoning-consistent review comments that align with human reviewers' judgments. In this paper, we address this challenge by proposing ReviewAgents, a framework that leverages large language models (LLMs) to generate academic paper reviews. We first introduce a novel dataset, Review-CoT, consisting of 142k review comments, designed for training LLM agents. This dataset emulates the structured reasoning process of human reviewers-summarizing the paper, referencing relevant works, identifying strengths and weaknesses, and generating a review conclusion. Building upon this, we train LLM reviewer agents capable of structured reasoning using a relevant-paper-aware training method. Furthermore, we construct ReviewAgents, a multi-role, multi-LLM agent review framework, to enhance the review comment generation process. Additionally, we propose ReviewBench, a benchmark for evaluating the review comments generated by LLMs. Our experimental results on ReviewBench demonstrate that while existing LLMs exhibit a certain degree of potential for automating the review process, there remains a gap when compared to human-generated reviews. Moreover, our ReviewAgents framework further narrows this gap, outperforming advanced LLMs in generating review comments.
>
---
#### [replaced 034] Measuring Spiritual Values and Bias of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.11647v2](http://arxiv.org/pdf/2410.11647v2)**

> **作者:** Songyuan Liu; Ziyang Zhang; Runze Yan; Wei Wu; Carl Yang; Jiaying Lu
>
> **备注:** 9 pages including appendix; 5 figures; 5 tables
>
> **摘要:** Large language models (LLMs) have become integral tool for users from various backgrounds. LLMs, trained on vast corpora, reflect the linguistic and cultural nuances embedded in their pre-training data. However, the values and perspectives inherent in this data can influence the behavior of LLMs, leading to potential biases. As a result, the use of LLMs in contexts involving spiritual or moral values necessitates careful consideration of these underlying biases. Our work starts with verification of our hypothesis by testing the spiritual values of popular LLMs. Experimental results show that LLMs' spiritual values are quite diverse, as opposed to the stereotype of atheists or secularists. We then investigate how different spiritual values affect LLMs in social-fairness scenarios e.g., hate speech identification). Our findings reveal that different spiritual values indeed lead to different sensitivity to different hate target groups. Furthermore, we propose to continue pre-training LLMs on spiritual texts, and empirical results demonstrate the effectiveness of this approach in mitigating spiritual bias.
>
---
#### [replaced 035] Multi-domain Multilingual Sentiment Analysis in Industry: Predicting Aspect-based Opinion Quadruples
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10389v2](http://arxiv.org/pdf/2505.10389v2)**

> **作者:** Benjamin White; Anastasia Shimorina
>
> **摘要:** This paper explores the design of an aspect-based sentiment analysis system using large language models (LLMs) for real-world use. We focus on quadruple opinion extraction -- identifying aspect categories, sentiment polarity, targets, and opinion expressions from text data across different domains and languages. We investigate whether a single fine-tuned model can effectively handle multiple domain-specific taxonomies simultaneously. We demonstrate that a combined multi-domain model achieves performance comparable to specialized single-domain models while reducing operational complexity. We also share lessons learned for handling non-extractive predictions and evaluating various failure modes when developing LLM-based systems for structured prediction tasks.
>
---
#### [replaced 036] Truth Sleuth and Trend Bender: AI Agents to fact-check YouTube videos and influence opinions
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.10577v2](http://arxiv.org/pdf/2507.10577v2)**

> **作者:** Cécile Logé; Rehan Ghori
>
> **摘要:** Misinformation poses a significant threat in today's digital world, often spreading rapidly through platforms like YouTube. This paper introduces a novel approach to combating misinformation by developing an AI-powered system that not only fact-checks claims made in YouTube videos but also actively engages users in the comment section and challenge misleading narratives. Our system comprises two main agents: Truth Sleuth and Trend Bender. Truth Sleuth extracts claims from a YouTube video, uses a Retrieval-Augmented Generation (RAG) approach - drawing on sources like Wikipedia, Google Search, Google FactCheck - to accurately assess their veracity and generates a nuanced and comprehensive report. Through rigorous prompt engineering, Trend Bender leverages this report along with a curated corpus of relevant articles to generate insightful and persuasive comments designed to stimulate a productive debate. With a carefully set up self-evaluation loop, this agent is able to iteratively improve its style and refine its output. We demonstrate the system's capabilities through experiments on established benchmark datasets and a real-world deployment on YouTube, showcasing its potential to engage users and potentially influence perspectives. Our findings highlight the high accuracy of our fact-checking agent, and confirm the potential of AI-driven interventions in combating misinformation and fostering a more informed online space.
>
---
#### [replaced 037] Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training
- **分类: cs.AI; cs.CL; math.SP**

- **链接: [http://arxiv.org/pdf/2410.15460v4](http://arxiv.org/pdf/2410.15460v4)**

> **作者:** Shahrad Mohammadzadeh; Juan David Guerra; Marco Bonizzato; Reihaneh Rabbany; Golnoosh Farnadi
>
> **备注:** Accepted to ACL 2025, accepted to Safe Generative AI Workshop @ NeurIPS 2024. Camera-ready version for ACL 2025 (to appear). Submitted July 2025
>
> **摘要:** As large language models (LLMs) become increasingly prevalent, concerns about their reliability, particularly due to hallucinations - factually inaccurate or irrelevant outputs - have grown. Our research investigates the relationship between the uncertainty in training dynamics and the emergence of hallucinations. Using models from the Pythia suite and several hallucination detection metrics, we analyze hallucination trends and identify significant variance during training. To address this, we propose \textbf{Sensitivity Dropout (SenD)}, a novel training protocol designed to reduce hallucination variance during training by deterministically dropping embedding indices with significant variability. In addition, we develop an unsupervised hallucination detection metric, Efficient EigenScore (EES), which approximates the traditional EigenScore in 2x speed. This metric is integrated into our training protocol, allowing SenD to be both computationally scalable and effective at reducing hallucination variance. SenD improves test-time reliability of Pythia and Meta's Llama models by up to 17\% and enhances factual accuracy in Wikipedia, Medical, Legal, and Coding domains without affecting downstream task performance.
>
---
#### [replaced 038] Understanding Language Model Circuits through Knowledge Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.17241v4](http://arxiv.org/pdf/2406.17241v4)**

> **作者:** Huaizhi Ge; Frank Rudzicz; Zining Zhu
>
> **备注:** A previous version of this document contained a hidden prompt entered by Z Zhu without knowledge of -- or consent by -- his co-authors. This version does not contain the prompt
>
> **摘要:** Recent advances in language model interpretability have identified circuits, critical subnetworks that replicate model behaviors, yet how knowledge is structured within these crucial subnetworks remains opaque. To gain an understanding toward the knowledge in the circuits, we conduct systematic knowledge editing experiments on the circuits of the GPT-2 language model. Our analysis reveals intriguing patterns in how circuits respond to editing attempts, the extent of knowledge distribution across network components, and the architectural composition of knowledge-bearing circuits. These findings offer insights into the complex relationship between model circuits and knowledge representation, deepening the understanding of how information is organized within language models. Our findings offer novel insights into the ``meanings'' of the circuits, and introduce directions for further interpretability and safety research of language models.
>
---
#### [replaced 039] RAGGED: Towards Informed Design of Scalable and Stable RAG Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.09040v3](http://arxiv.org/pdf/2403.09040v3)**

> **作者:** Jennifer Hsia; Afreen Shaikh; Zhiruo Wang; Graham Neubig
>
> **备注:** Project page: https://github.com/neulab/ragged
>
> **摘要:** Retrieval-augmented generation (RAG) enhances language models by integrating external knowledge, but its effectiveness is highly dependent on system configuration. Improper retrieval settings can degrade performance, making RAG less reliable than closed-book generation. In this work, we introduce RAGGED, a framework for systematically evaluating RAG systems across diverse retriever-reader configurations, retrieval depths, and datasets. Our analysis reveals that reader robustness to noise is the key determinant of RAG stability and scalability. Some readers benefit from increased retrieval depth, while others degrade due to their sensitivity to distracting content. Through large-scale experiments on open-domain, multi-hop, and specialized-domain datasets, we show that retrievers, rerankers, and prompts influence performance but do not fundamentally alter these reader-driven trends. By providing a principled framework and new metrics to assess RAG stability and scalability, RAGGED enables systematic evaluation of retrieval-augmented generation systems, guiding future research on optimizing retrieval depth and model robustness.
>
---
#### [replaced 040] How Well Can Knowledge Edit Methods Edit Perplexing Knowledge?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.17253v3](http://arxiv.org/pdf/2406.17253v3)**

> **作者:** Huaizhi Ge; Frank Rudzicz; Zining Zhu
>
> **备注:** A previous version of this document contained a hidden prompt entered by Z Zhu without knowledge of -- or consent by -- his co-authors. This version does not contain the prompt
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities, but updating their knowledge post-training remains a critical challenge. While recent model editing techniques like Rank-One Model Editing (ROME) show promise, their effectiveness may vary based on the nature of the knowledge being edited. We introduce the concept of ``perplexingness'': the degree to which new knowledge conflicts with an LLM's learned conceptual hierarchies and categorical relationships. For instance, editing ``British Shorthair is a kind of cat'' to ``British Shorthair is a kind of dog'' represents a low-perplexingness edit within the same taxonomic level, while editing ``A cat is a kind of animal'' to ``A cat is a kind of plant'' represents a high-perplexingness edit that violates fundamental categorical boundaries. To systematically investigate this phenomenon, we introduce HierarchyData, a carefully curated dataset of 99 hyponym-hypernym pairs across diverse categories. Through controlled experiments across three models and four editing methods, we demonstrate a strong negative correlation between the perplexingness of new knowledge and the effectiveness of knowledge editing. Our analysis reveals that edits involving more abstract concepts (hypernyms) generally exhibit higher perplexingness and are more resistant to modification than their specific counterparts (hyponyms). These findings highlight a fundamental challenge in LLM knowledge editing: the more a new fact contradicts an LLM's learned conceptual hierarchies, the harder it becomes to reliably encode that knowledge.
>
---
#### [replaced 041] TRIM: Token Reduction and Inference Modeling for Cost-Effective Language Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.07682v4](http://arxiv.org/pdf/2412.07682v4)**

> **作者:** Alfredo Garrachón Ruiz; Tomás de la Rosa; Daniel Borrajo
>
> **备注:** 13 pages, 12 tables, 7 figures
>
> **摘要:** The inference cost of Large Language Models (LLMs) is a significant challenge due to their computational demands, specially on tasks requiring long outputs. However, natural language often contains redundancy, which presents an opportunity for optimization. We have observed that LLMs can generate distilled language-concise outputs that retain essential meaning, when prompted appropriately. We propose TRIM, a pipeline for saving computational cost in which a shorter distilled output from the LLM is reconstructed into a full narrative by a smaller model with lower inference costs. Our experiments show promising results, particularly in general knowledge domains with 20.58% saved tokens on average with tiny decrease in evaluation metrics, hinting that this approach can effectively balance efficiency and accuracy in language processing tasks.
>
---
