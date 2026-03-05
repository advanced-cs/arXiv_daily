# 自然语言处理 cs.CL

- **最新发布 105 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] SE-Search: Self-Evolving Search Agent via Memory and Dense Reward
- **分类: cs.CL**

- **简介: 该论文提出SE-Search，解决信息检索中的噪声与低效问题，通过记忆净化、原子查询训练和密集奖励提升搜索性能。属于问答任务。**

- **链接: [https://arxiv.org/pdf/2603.03293](https://arxiv.org/pdf/2603.03293)**

> **作者:** Jian Li; Yizhang Jin; Dongqi Liu; Hang Ding; Jiafu Wu; Dongsheng Chen; Yunhang Shen; Yulei Qin; Ying Tai; Chengjie Wang; Xiaotong Yuan; Yabiao Wang
>
> **摘要:** Retrieval augmented generation (RAG) reduces hallucinations and factual errors in large language models (LLMs) by conditioning generation on retrieved external knowledge. Recent search agents further cast RAG as an autonomous, multi-turn information-seeking process. However, existing methods often accumulate irrelevant or noisy documents and rely on sparse reinforcement learning signals. We propose \textbf{S}elf-\textbf{E}volving \textbf{Search}, a Self-Evolving Search agent that improves online search behavior through three components, memory purification, atomic query training, and dense rewards. SE-Search follows a \textit{Think-Search-Memorize} strategy that retains salient evidence while filtering irrelevant content. Atomic query training promotes shorter and more diverse queries, improving evidence acquisition. Dense rewards provide fine-grained feedback that speeds training. Experiments on single-hop and multi-hop question answering benchmarks show that \texttt{SE-Search-3B} outperforms strong baselines, yielding a $10.8$ point absolute improvement and a $33.8\%$ relative gain over Search-R1.\footnote{We will make the code and model weights publicly available upon acceptance.}
>
---
#### [new 002] CzechTopic: A Benchmark for Zero-Shot Topic Localization in Historical Czech Documents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于零样本主题定位任务，旨在识别历史捷克文本文本中表达特定主题的文本片段。研究构建了一个人工标注的基准数据集，并评估多种语言模型的表现。**

- **链接: [https://arxiv.org/pdf/2603.03884](https://arxiv.org/pdf/2603.03884)**

> **作者:** Martin Kostelník; Michal Hradiš; Martin Dočekal
>
> **摘要:** Topic localization aims to identify spans of text that express a given topic defined by a name and description. To study this task, we introduce a human-annotated benchmark based on Czech historical documents, containing human-defined topics together with manually annotated spans and supporting evaluation at both document and word levels. Evaluation is performed relative to human agreement rather than a single reference annotation. We evaluate a diverse range of large language models alongside BERT-based models fine-tuned on a distilled development dataset. Results reveal substantial variability among LLMs, with performance ranging from near-human topic detection to pronounced failures in span localization. While the strongest models approach human agreement, the distilled token embedding models remain competitive despite their smaller scale. The dataset and evaluation framework are publicly available at: this https URL.
>
---
#### [new 003] Order Is Not Layout: Order-to-Space Bias in Image Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究图像生成中的顺序-空间偏差问题，属于图像生成任务。针对实体顺序影响布局的偏差，提出OTSBench进行量化分析，并验证通过微调和早期干预可有效减少偏差。**

- **链接: [https://arxiv.org/pdf/2603.03714](https://arxiv.org/pdf/2603.03714)**

> **作者:** Yongkang Zhang; Zonglin Zhao; Yuechen Zhang; Fei Ding; Pei Li; Wenxuan Wang
>
> **摘要:** We study a systematic bias in modern image generation models: the mention order of entities in text spuriously determines spatial layout and entity--role binding. We term this phenomenon Order-to-Space Bias (OTS) and show that it arises in both text-to-image and image-to-image generation, often overriding grounded cues and causing incorrect layouts or swapped assignments. To quantify OTS, we introduce OTS-Bench, which isolates order effects with paired prompts differing only in entity order and evaluates models along two dimensions: homogenization and correctness. Experiments show that Order-to-Space Bias (OTS) is widespread in modern image generation models, and provide evidence that it is primarily data-driven and manifests during the early stages of layout formation. Motivated by this insight, we show that both targeted fine-tuning and early-stage intervention strategies can substantially reduce OTS, while preserving generation quality.
>
---
#### [new 004] TopicENA: Enabling Epistemic Network Analysis at Scale through Automated Topic-Based Coding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分析任务，旨在解决传统EN A依赖人工编码导致的可扩展性问题。通过结合BERTopic与ENA，提出TopicENA框架，实现自动化概念建模与网络分析。**

- **链接: [https://arxiv.org/pdf/2603.03307](https://arxiv.org/pdf/2603.03307)**

> **作者:** Owen H.T. Lu; Tiffany T.Y. Hsu
>
> **摘要:** Epistemic Network Analysis (ENA) is a method for investigating the relational structure of concepts in text by representing co-occurring concepts as networks. Traditional ENA, however, relies heavily on manual expert coding, which limits its scalability and real-world applicability to large text corpora. Topic modeling provides an automated approach to extracting concept-level representations from text and can serve as an alternative to manual coding. To tackle this limitation, the present study merges BERTopic with ENA and introduces TopicENA, a topic-based epistemic network analysis framework. TopicENA substitutes manual concept coding with automatically generated topics while maintaining ENA's capacity for modeling structural associations among concepts. To explain the impact of modeling choices on TopicENA outcomes, three analysis cases are presented. The first case assesses the effect of topic granularity, indicating that coarse-grained topics are preferable for large datasets, whereas fine-grained topics are more effective for smaller datasets. The second case examines topic inclusion thresholds and finds that threshold values should be adjusted according to topic quality indicators to balance network consistency and interpretability. The third case tests TopicENA's scalability by applying it to a substantially larger dataset than those used in previous ENA studies. Collectively, these cases illustrate that TopicENA facilitates practical and interpretable ENA analysis at scale and offers concrete guidance for configuring topic-based ENA pipelines in large-scale text analysis.
>
---
#### [new 005] Assessing the Effectiveness of LLMs in Delivering Cognitive Behavioral Therapy
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在评估LLMs在提供认知行为疗法中的有效性。研究比较了两种生成方法，发现LLMs在情感共鸣和一致性方面存在局限。**

- **链接: [https://arxiv.org/pdf/2603.03862](https://arxiv.org/pdf/2603.03862)**

> **作者:** Navdeep Singh Bedi; Ana-Maria Bucur; Noriko Kando; Fabio Crestani
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** As mental health issues continue to rise globally, there is an increasing demand for accessible and scalable therapeutic solutions. Many individuals currently seek support from Large Language Models (LLMs), even though these models have not been validated for use in counseling services. In this paper, we evaluate LLMs' ability to emulate professional therapists practicing Cognitive Behavioral Therapy (CBT). Using anonymized, transcribed role-play sessions between licensed therapists and clients, we compare two approaches: (1) a generation-only method and (2) a Retrieval-Augmented Generation (RAG) approach using CBT guidelines. We evaluate both proprietary and open-source models for linguistic quality, semantic coherence, and therapeutic fidelity using standard natural language generation (NLG) metrics, natural language inference (NLI), and automated scoring for skills assessment. Our results indicate that while LLMs can generate CBT-like dialogues, they are limited in their ability to convey empathy and maintain consistency.
>
---
#### [new 006] SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于对话推荐系统任务，解决LLM推荐系统中因忽视用户个性化安全约束导致的安全问题。提出SafeCRS框架，提升推荐安全性与质量。**

- **链接: [https://arxiv.org/pdf/2603.03536](https://arxiv.org/pdf/2603.03536)**

> **作者:** Haochang Hao; Yifan Xu; Xinzhuo Li; Yingqiang Ge; Lu Cheng
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Current LLM-based conversational recommender systems (CRS) primarily optimize recommendation accuracy and user satisfaction. We identify an underexplored vulnerability in which recommendation outputs may negatively impact users by violating personalized safety constraints, when individualized safety sensitivities -- such as trauma triggers, self-harm history, or phobias -- are implicitly inferred from the conversation but not respected during recommendation. We formalize this challenge as personalized CRS safety and introduce SafeRec, a new benchmark dataset designed to systematically evaluate safety risks in LLM-based CRS under user-specific constraints. To further address this problem, we propose SafeCRS, a safety-aware training framework that integrates Safe Supervised Fine-Tuning (Safe-SFT) with Safe Group reward-Decoupled Normalization Policy Optimization (Safe-GDPO) to jointly optimize recommendation quality and personalized safety alignment. Extensive experiments on SafeRec demonstrate that SafeCRS reduces safety violation rates by up to 96.5% relative to the strongest recommendation-quality baseline while maintaining competitive recommendation quality. Warning: This paper contains potentially harmful and offensive content.
>
---
#### [new 007] AutoHarness: improving LLM agents by automatically synthesizing a code harness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理执行非法动作的问题。通过自动合成代码防护机制，提升模型表现并降低成本。**

- **链接: [https://arxiv.org/pdf/2603.03329](https://arxiv.org/pdf/2603.03329)**

> **作者:** Xinghua Lou; Miguel Lázaro-Gredilla; Antoine Dedieu; Carter Wendelken; Wolfgang Lehrach; Kevin P. Murphy
>
> **备注:** agent harness, code synthesis, self-improvement, code-as-policy, text games
>
> **摘要:** Despite significant strides in language models in the last few years, when used as agents, such models often try to perform actions that are not just suboptimal for a given state, but are strictly prohibited by the external environment. For example, in the recent Kaggle GameArena chess competition, 78% of Gemini-2.5-Flash losses were attributed to illegal moves. Often people manually write "harnesses" around LLMs to prevent such failures. In this paper, we demonstrate that Gemini-2.5-Flash can automatically synthesize such a code harness, using a small number of rounds of iterative code refinement given feedback from the (game) environment. The resulting harness prevents all illegal moves in 145 different TextArena games (both 1-player and 2-player), enabling the smaller Gemini-2.5-Flash model to outperform larger models, such as Gemini-2.5-Pro. Pushing our technique to the limit, we can get Gemini-2.5-Flash to generate the entire policy in code, thus eliminating the need to use the LLM at decision making time. The resulting code-policy receives a higher average reward than Gemini-2.5-Pro and GPT-5.2-High on 16 TextArena 1-player games. Our results show that using a smaller model to synthesize a custom code harness (or entire policy) can outperform a much larger model, while also being more cost effective.
>
---
#### [new 008] The Company You Keep: How LLMs Respond to Dark Triad Traits
- **分类: cs.CL**

- **简介: 该论文属于对话系统安全研究，旨在解决LLMs对Dark Triad特质用户提示的响应问题。通过分析模型在不同严重程度下的行为差异，提出更安全的交互设计建议。**

- **链接: [https://arxiv.org/pdf/2603.04299](https://arxiv.org/pdf/2603.04299)**

> **作者:** Zeyi Lu; Angelica Henestrosa; Pavel Chizhov; Ivan P. Yamshchikov
>
> **摘要:** Large Language Models (LLMs) often exhibit highly agreeable and reinforcing conversational styles, also known as AI-sycophancy. Although this behavior is encouraged, it may become problematic when interacting with user prompts that reflect negative social tendencies. Such responses risk amplifying harmful behavior rather than mitigating it. In this study, we examine how LLMs respond to user prompts expressing varying degrees of Dark Triad traits (Machiavellianism, Narcissism, and Psychopathy) using a curated dataset. Our analysis reveals differences across models, whereby all models predominantly exhibit corrective behavior, while showing reinforcing output in certain cases. Model behavior also depends on the severity level and differs in the sentiment of the response. Our findings raise implications for designing safer conversational systems that can detect and respond appropriately when users escalate from benign to harmful requests.
>
---
#### [new 009] From Exact Hits to Close Enough: Semantic Caching for LLM Embeddings
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于缓存优化任务，解决LLM响应速度与成本问题。通过语义缓存技术，提升缓存效率，提出多项策略并验证效果。**

- **链接: [https://arxiv.org/pdf/2603.03301](https://arxiv.org/pdf/2603.03301)**

> **作者:** Dvir David Biton; Roy Friedman
>
> **摘要:** The rapid adoption of large language models (LLMs) has created demand for faster responses and lower costs. Semantic caching, reusing semantically similar requests via their embeddings, addresses this need but breaks classic cache assumptions and raises new challenges. In this paper, we explore offline policies for semantic caching, proving that implementing an optimal offline policy is NP-hard, and propose several polynomial-time heuristics. We also present online semantic aware cache policies that combine recency, frequency, and locality. Evaluations on diverse datasets show that while frequency based policies are strong baselines, our novel variant improves semantic accuracy. Our findings reveal effective strategies for current systems and highlight substantial headroom for future innovation. All code is open source.
>
---
#### [new 010] Draft-Conditioned Constrained Decoding for Structured Generation in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于结构化生成任务，解决LLM生成输出时因语法错误导致不可用的问题。提出DCCD方法，通过先生成草稿再约束解码，提升生成准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.03305](https://arxiv.org/pdf/2603.03305)**

> **作者:** Avinash Reddy; Thayne T. Walker; James S. Ide; Amrit Singh Bedi
>
> **摘要:** Large language models (LLMs) are increasingly used to generate executable outputs, JSON objects, and API calls, where a single syntax error can make the output unusable. Constrained decoding enforces validity token-by-token via masking and renormalization, but it can distort generation when the model assigns low probability mass to valid continuations, pushing decoding toward locally valid yet semantically incorrect trajectories. We propose \emph{Draft-Conditioned Constrained Decoding (DCCD)}, a simple two-step, training-free inference procedure that decouples semantic planning from structural enforcement: an unconstrained draft is generated first, and constrained decoding is then applied, conditioned on this draft, to guarantee validity. We analyze DCCD through a KL-projection view, showing that draft conditioning increases feasible mass and reduces the cumulative "projection tax" induced by hard constraints, with an optional best-of-$K$ draft selection. Across structured reasoning benchmarks, DCCD improves strict structured accuracy by up to +24 percentage points over standard constrained decoding (e.g., 15.2\% to 39.0\% on GSM8K with a 1B model), and enables smaller model pairs to match or exceed much larger constrained baselines, yielding substantial gains in parameter efficiency.
>
---
#### [new 011] Linguistically Informed Graph Model and Semantic Contrastive Learning for Korean Short Text Classification
- **分类: cs.CL**

- **简介: 该论文属于短文本分类任务，针对韩语短文本因上下文不足和标注数据少而难以分类的问题，提出LIGRAM模型结合语义对比学习，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2603.03652](https://arxiv.org/pdf/2603.03652)**

> **作者:** JaeGeon Yoo; Byoungwook Kim; Yeongwook Yang; Hong-Jun Jang
>
> **备注:** 16 pages, 1 Figure, Accepted at DASFAA 2026 (Full Research Paper)
>
> **摘要:** Short text classification (STC) remains a challenging task due to the scarcity of contextual information and labeled data. However, existing approaches have pre-dominantly focused on English because most benchmark datasets for the STC are primarily available in English. Consequently, existing methods seldom incorporate the linguistic and structural characteristics of Korean, such as its agglutinative morphology and flexible word order. To address these limitations, we propose LIGRAM, a hierarchical heterogeneous graph model for Korean short-text classification. The proposed model constructs sub-graphs at the morpheme, part-of-speech, and named-entity levels and hierarchically integrates them to compensate for the limited contextual information in short texts while precisely capturing the grammatical and semantic dependencies inherent in Korean. In addition, we apply Semantics-aware Contrastive Learning (SemCon) to reflect semantic similarity across documents, enabling the model to establish clearer decision boundaries even in short texts where class distinctions are often ambiguous. We evaluate LIGRAM on four Korean short-text datasets, where it consistently outperforms existing baseline models. These outcomes validate that integrating language-specific graph representations with SemCon provides an effective solution for short text classification in agglutinative languages such as Korean.
>
---
#### [new 012] Retrieval or Representation? Reassessing Benchmark Gaps in Multilingual and Visually Rich RAG
- **分类: cs.CL**

- **简介: 该论文研究多语言和视觉丰富的RAG基准问题，旨在解决模型性能提升的根源。通过对比检索与表示方法，发现更好的文档表示是关键，提出需分解评估以明确进步来源。**

- **链接: [https://arxiv.org/pdf/2603.04238](https://arxiv.org/pdf/2603.04238)**

> **作者:** Martin Asenov; Kenza Benkirane; Dan Goldwater; Aneiss Ghodsi
>
> **备注:** ICLR 2026 Workshop I Can't Believe It's Not Better: Where Large Language Models Need to Improve
>
> **摘要:** Retrieval-augmented generation (RAG) is a common way to ground language models in external documents and up-to-date information. Classical retrieval systems relied on lexical methods such as BM25, which rank documents by term overlap with corpus-level weighting. End-to-end multimodal retrievers trained on large query-document datasets claim substantial improvements over these approaches, especially for multilingual documents with complex visual layouts. We demonstrate that better document representation is the primary driver of benchmark improvements. By systematically varying transcription and preprocessing methods while holding the retrieval mechanism fixed, we demonstrate that BM25 can recover large gaps on multilingual and visual benchmarks. Our findings call for decomposed evaluation benchmarks that separately measure transcription and retrieval capabilities, enabling the field to correctly attribute progress and focus effort where it matters.
>
---
#### [new 013] M-QUEST -- Meme Question-Understanding Evaluation on Semantics and Toxicity
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出M-QUEST框架和基准，用于评估模型对网络迷因的语义理解和毒性检测能力，解决多模态内容安全与常识推理问题。**

- **链接: [https://arxiv.org/pdf/2603.03315](https://arxiv.org/pdf/2603.03315)**

> **作者:** Stefano De Giorgis; Ting-Chih Chen; Filip Ilievski
>
> **摘要:** Internet memes are a powerful form of online communication, yet their nature and reliance on commonsense knowledge make toxicity detection challenging. Identifying key features for meme interpretation and understanding, is a crucial task. Previous work has been focused on some elements contributing to the meaning, such as the Textual dimension via OCR, the Visual dimension via object recognition, upper layers of meaning like the Emotional dimension, Toxicity detection via proxy variables, such as hate speech detection, and sentiment analysis. Nevertheless, there is still a lack of an overall architecture able to formally identify elements contributing to the meaning of a meme, and be used in the sense-making process. In this work, we present a semantic framework and a corresponding benchmark for automatic knowledge extraction from memes. First, we identify the necessary dimensions to understand and interpret a meme: Textual material, Visual material, Scene, Background Knowledge, Emotion, Semiotic Projection, Analogical Mapping, Overall Intent, Target Community, and Toxicity Assessment. Second, the framework guides a semi-automatic process of generating a benchmark with commonsense question-answer pairs about meme toxicity assessment and its underlying reason. The resulting benchmark M-QUEST consists of 609 question-answer pairs for 307 memes. Thirdly, we evaluate eight open-source large language models on their ability to correctly solve M-QUEST. Our results show that current models' commonsense reasoning capabilities for toxic meme interpretation vary depending on the dimension and architecture. Models with instruction tuning and reasoning capabilities significantly outperform the others, though pragmatic inference questions remain challenging. We release code, benchmark, and prompts to support future research intersecting multimodal content safety and commonsense reasoning.
>
---
#### [new 014] DIALEVAL: Automated Type-Theoretic Evaluation of LLM Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DIALEVAL，用于自动化评估大语言模型的指令遵循能力。解决手动标注效率低、与人类判断不一致的问题，通过类型理论框架实现指令分解与精准评估。**

- **链接: [https://arxiv.org/pdf/2603.03321](https://arxiv.org/pdf/2603.03321)**

> **作者:** Nardine Basta; Dali Kaafar
>
> **备注:** PAKDD 2026
>
> **摘要:** Evaluating instruction following in Large Language Models requires decomposing instructions into verifiable requirements and assessing satisfaction--tasks currently dependent on manual annotation and uniform criteria that do not align with human judgment patterns. We present DIALEVAL, a type-theoretic framework using dual LLM agents to automate instruction decomposition into typed predicates and implement type-specific satisfaction semantics. The framework enforces formal atomicity and independence constraints during automated extraction, then applies differentiated evaluation criteria--semantic equivalence for content predicates, exact precision for numerical predicates--mirroring empirically observed human assessment patterns. Extended to multi-turn dialogues through history-aware satisfaction functions, DIALEVAL enables evaluation in conversational contexts where single-turn methods fail. Validation demonstrates 90.38% accuracy (26.45% error reduction over baselines) and substantially stronger correlation with human judgment for complex instructions.
>
---
#### [new 015] Can Large Language Models Derive New Knowledge? A Dynamic Benchmark for Biological Knowledge Discovery
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识发现任务，旨在解决AI模型在生物领域发现新知识的能力评估问题。提出动态基准DBench-Bio，通过自动化流程定期更新数据，以准确评估模型的新知识发现能力。**

- **链接: [https://arxiv.org/pdf/2603.03322](https://arxiv.org/pdf/2603.03322)**

> **作者:** Chaoqun Yang; Xinyu Lin; Shulin Li; Wenjie Wang; Ruihan Guo; Fuli Feng; Tat-Seng Chua
>
> **摘要:** Recent advancements in Large Language Model (LLM) agents have demonstrated remarkable potential in automatic knowledge discovery. However, rigorously evaluating an AI's capacity for knowledge discovery remains a critical challenge. Existing benchmarks predominantly rely on static datasets, leading to inevitable data contamination where models have likely seen the evaluation knowledge during training. Furthermore, the rapid release cycles of modern LLMs render static benchmarks quickly outdated, failing to assess the ability to discover truly new knowledge. To address these limitations, we propose DBench-Bio, a dynamic and fully automated benchmark designed to evaluate AI's biological knowledge discovery ability. DBench-Bio employs a three-stage pipeline: (1) data acquisition of rigorous, authoritative paper abstracts; (2) QA extraction utilizing LLMs to synthesize scientific hypothesis questions and corresponding discovery answers; and (3) QA filter to ensure quality based on relevance, clarity, and centrality. We instantiate this pipeline to construct a monthly-updated benchmark covering 12 biomedical sub-domains. Extensive evaluations of SOTA models reveal current limitations in discovering new knowledge. Our work provides the first dynamic, automatic framework for assessing the new knowledge discovery capabilities of AI systems, establishing a living, evolving resource for AI research community to catalyze the development of knowledge discovery.
>
---
#### [new 016] Towards Self-Robust LLMs: Intrinsic Prompt Noise Resistance via CoIPO
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型对提示噪声的鲁棒性。通过提出CoIPO方法，减少干净与噪声提示下模型输出的差异，增强模型内在稳定性。**

- **链接: [https://arxiv.org/pdf/2603.03314](https://arxiv.org/pdf/2603.03314)**

> **作者:** Xin Yang; Letian Li; Abudukelimu Wuerkaixi; Xuxin Cheng; Cao Liu; Ke Zeng; Xunliang Cai; Wenyuan Jiang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable and steadily improving performance across a wide range of tasks. However, LLM performance may be highly sensitive to prompt variations especially in scenarios with limited openness or strict output formatting requirements, indicating insufficient robustness. In real-world applications, user prompts provided to LLMs often contain imperfections, which may undermine the quality of the model's responses. To address this issue, previous work has primarily focused on preprocessing prompts, employing external tools or even LLMs to refine prompt formulations in advance. However, these approaches overlook the intrinsic robustness of LLMs, and their reliance on external components introduces additional computational overhead and uncertainty. In this work, we propose a Contrastive Learning-based Inverse Direct Preference Optimization (CoIPO) method that minimizes the discrepancy between the label-aligned logits produced by the model under a clean prompt and its noisy counterpart, and conduct a detailed analysis using mutual information theory. We augment the FLAN dataset by constructing paired prompts, each consisting of a clean prompt and its corresponding noisy version for training. Additionally, to evaluate the effectiveness, we develop NoisyPromptBench, a benchmark enhanced and derived from the existing PromptBench. Experimental results conducted on NoisyPromptBench demonstrate that our proposed method achieves a significant improvement in average accuracy over the current state-of-the-art approaches. The source code of CoIPO, pair-wise FLAN datasets, and NoisyPromptBench have already been released on this https URL.
>
---
#### [new 017] TATRA: Training-Free Instance-Adaptive Prompting Through Rephrasing and Aggregation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TATRA，一种无需训练数据的实例自适应提示方法，解决传统提示工程需大量数据和优化的问题。通过动态生成示例提升模型性能，适用于文本分类和数学推理任务。**

- **链接: [https://arxiv.org/pdf/2603.03298](https://arxiv.org/pdf/2603.03298)**

> **作者:** Bartosz Dziuba; Kacper Kuchta; Paweł Batorski; Przemysław Spurek; Paul Swoboda
>
> **摘要:** Large Language Models (LLMs) have improved substantially alignment, yet their behavior remains highly sensitive to prompt phrasing. This brittleness has motivated automated prompt engineering, but most existing methods (i) require a task-specific training set, (ii) rely on expensive iterative optimization to produce a single dataset-level prompt, and (iii) must be rerun from scratch for each new task. We introduce TATRA, a dataset-free prompting method that constructs instance-specific few-shot prompts by synthesizing on-the-fly examples to accompany a user-provided instruction. TATRA requires no labeled training data and avoids task-specific optimization loops, while retaining the benefits of demonstration-based prompting. Across standard text classification benchmarks, TATRA matches or improves over strong prompt-optimization baselines that depend on training data and extensive search. On mathematical reasoning benchmarks, TATRA achieves state-of-the-art performance on GSM8K and DeepMath, outperforming methods that explicitly optimize prompts on those tasks. Our results suggest that per-instance construction of effective in-context examples is more important than running long, expensive optimization loops to produce a single prompt per task. We will make all code publicly available upon acceptance of the paper. Code is available at this https URL
>
---
#### [new 018] Belief-Sim: Towards Belief-Driven Simulation of Demographic Misinformation Susceptibility
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会仿真任务，旨在模拟不同人口群体对虚假信息的敏感性。通过构建基于信念的模型，提升虚假信息传播模拟的准确性。**

- **链接: [https://arxiv.org/pdf/2603.03585](https://arxiv.org/pdf/2603.03585)**

> **作者:** Angana Borah; Zohaib Khan; Rada Mihalcea; Verónica Pérez-Rosas
>
> **备注:** Paper Under Review
>
> **摘要:** Misinformation is a growing societal threat, and susceptibility to misinformative claims varies across demographic groups due to differences in underlying beliefs. As Large Language Models (LLMs) are increasingly used to simulate human behaviors, we investigate whether they can simulate demographic misinformation susceptibility, treating beliefs as a primary driving factor. We introduce BeliefSim, a simulation framework that constructs demographic belief profiles using psychology-informed taxonomies and survey priors. We study prompt-based conditioning and post-training adaptation, and conduct a multi-fold evaluation using: (i) susceptibility accuracy and (ii) counterfactual demographic sensitivity. Across both datasets and modeling strategies, we show that beliefs provide a strong prior for simulating misinformation susceptibility, with accuracy up to 92%.
>
---
#### [new 019] AILS-NTUA at SemEval-2026 Task 12: Graph-Based Retrieval and Reflective Prompting for Abductive Event Reasoning
- **分类: cs.CL**

- **简介: 该论文针对因果事件推理任务，提出融合图检索与反思提示的系统，解决多标签因果推理中的错误模式问题，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.04319](https://arxiv.org/pdf/2603.04319)**

> **作者:** Nikolas Karafyllis; Maria Lymperaiou; Giorgos Filandrianos; Athanasios Voulodimos; Giorgos Stamou
>
> **摘要:** We present a winning three-stage system for SemEval 2026 Task~12: Abductive Event Reasoning that combines graph-based retrieval, LLM-driven abductive reasoning with prompt design optimized through reflective prompt evolution, and post-hoc consistency enforcement; our system ranks first on the evaluation-phase leaderboard with an accuracy score of 0.95. Cross-model error analysis across 14 models (7~families) reveals three shared inductive biases: causal chain incompleteness, proximate cause preference, and salience bias, whose cross-family convergence (51\% cause-count reduction) indicates systematic rather than model-specific failure modes in multi-label causal reasoning.
>
---
#### [new 020] Coupling Local Context and Global Semantic Prototypes via a Hierarchical Architecture for Rhetorical Roles Labeling
- **分类: cs.CL**

- **简介: 该论文属于 rhetorical roles labeling 任务，旨在提升文档中句子功能角色的识别。通过结合局部上下文与全局语义原型，提出两种方法优化模型性能，并构建了新的法律数据集。**

- **链接: [https://arxiv.org/pdf/2603.03856](https://arxiv.org/pdf/2603.03856)**

> **作者:** Anas Belfathi; Nicolas Hernandez; Laura Monceaux; Warren Bonnard; Mary Catherine Lavissiere; Christine Jacquin; Richard Dufour
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Rhetorical Role Labeling (RRL) identifies the functional role of each sentence in a document, a key task for discourse understanding in domains such as law and medicine. While hierarchical models capture local dependencies effectively, they are limited in modeling global, corpus-level features. To address this limitation, we propose two prototype-based methods that integrate local context with global representations. Prototype-Based Regularization (PBR) learns soft prototypes through a distance-based auxiliary loss to structure the latent space, while Prototype-Conditioned Modulation (PCM) constructs corpus-level prototypes and injects them during training and inference. Given the scarcity of RRL resources, we introduce SCOTUS-Law, the first dataset of U.S. Supreme Court opinions annotated with rhetorical roles at three levels of granularity: category, rhetorical function, and step. Experiments on legal, medical, and scientific benchmarks show consistent improvements over strong baselines, with 4 Macro-F1 gains on low-frequency roles. We further analyze the implications in the era of Large Language Models and complement our findings with expert evaluation.
>
---
#### [new 021] A Neural Topic Method Using a Large-Language-Model-in-the-Loop for Business Research
- **分类: cs.CL; econ.EM**

- **简介: 该论文属于主题建模任务，旨在解决传统方法在解释性、稳定性及与文档表示对齐方面的不足。工作包括提出LX Topic模型，结合大语言模型优化主题质量，提升可解释性和测量性能。**

- **链接: [https://arxiv.org/pdf/2603.03623](https://arxiv.org/pdf/2603.03623)**

> **作者:** Stephan Ludwig; Peter J. Danaher; Xiaohao Yang
>
> **摘要:** The growing use of unstructured text in business research makes topic modeling a central tool for constructing explanatory variables from reviews, social media, and open-ended survey responses, yet existing approaches function poorly as measurement instruments. Prior work shows that textual content predicts outcomes such as sales, satisfaction, and firm performance, but probabilistic models often generate conceptually diffuse topics, neural topic models are difficult to interpret in theory-driven settings, and large language model approaches lack standardization, stability, and alignment with document-level representations. We introduce LX Topic, a neural topic method that conceptualizes topics as latent linguistic constructs and produces calibrated document-level topic proportions for empirical analysis. LX Topic builds on FASTopic to ensure strong document representativeness and integrates large language model refinement at the topic-word level using alignment and confidence-weighting mechanisms that enhance semantic coherence without distorting document-topic distributions. Evaluations on large-scale Amazon and Yelp review datasets demonstrate that LX Topic achieves the highest overall topic quality relative to leading models while preserving clustering and classification performance. By unifying topic discovery, refinement, and standardized output in a web-based system, LX Topic establishes topic modeling as a reproducible, interpretable, and measurement-oriented instrument for marketing research and practice.
>
---
#### [new 022] Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型推理任务，旨在提升推理效率。通过引入熵时推理机制，优化调度与采样，实现更高效的资源分配与不确定性控制。**

- **链接: [https://arxiv.org/pdf/2603.03310](https://arxiv.org/pdf/2603.03310)**

> **作者:** Andrew Kiruluta
>
> **摘要:** Modern large language model (LLM) inference engines optimize throughput and latency under fixed decoding rules, treating generation as a linear progression in token time. We propose a fundamentally different paradigm: entropic\-time inference, where decoding is governed by the flow of uncertainty rather than token index. We introduce a self\-organizing inference architecture that jointly couples scheduling, attention sparsification, and sampling temperature under a unified entropy control objective. Our method extends vLLM with entropy-aware scheduling, entropic pruning of paged attention blocks, and adaptive temperature control that stabilizes generation near a target entropy regime. This transforms inference into a resource\-intelligent thermodynamic process that allocates computation where uncertainty reduction is maximized. We present a concrete systems design, pseudocode, and integration plan, demonstrating how entropy can serve as a first\-class control signal for scalable LLM inference.
>
---
#### [new 023] Hindsight Quality Prediction Experiments in Multi-Candidate Human-Post-Edited Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量预测任务，旨在研究LLM对传统质量预测方法的影响。通过多候选数据集实验，评估源端难度和候选端质量估计的有效性。**

- **链接: [https://arxiv.org/pdf/2603.04083](https://arxiv.org/pdf/2603.04083)**

> **作者:** Malik Marmonier; Benoît Sagot; Rachel Bawden
>
> **备注:** Accepted to the 2026 Language Resources and Evaluation Conference (LREC)
>
> **摘要:** This paper investigates two complementary paradigms for predicting machine translation (MT) quality: source-side difficulty prediction and candidate-side quality estimation (QE). The rapid adoption of Large Language Models (LLMs) into MT workflows is reshaping the research landscape, yet its impact on established quality prediction paradigms remains underexplored. We study this issue through a series of "hindsight" experiments on a unique, multi-candidate dataset resulting from a genuine MT post-editing (MTPE) project. The dataset consists of over 6,000 English source segments with nine translation hypotheses from a diverse set of traditional neural MT systems and advanced LLMs, all evaluated against a single, final human post-edited reference. Using Kendall's rank correlation, we assess the predictive power of source-side difficulty metrics, candidate-side QE models and position heuristics against two gold-standard scores: TER (as a proxy for post-editing effort) and COMET (as a proxy for human judgment). Our findings highlight that the architectural shift towards LLMs alters the reliability of established quality prediction methods while simultaneously mitigating previous challenges in document-level translation.
>
---
#### [new 024] VietNormalizer: An Open-Source, Dependency-Free Python Library for Vietnamese Text Normalization in TTS and NLP Applications
- **分类: cs.CL; cs.NE**

- **简介: 该论文提出VietNormalizer，一个用于越南语文本归一化的开源工具，解决TTS和NLP中非标准词汇处理问题。**

- **链接: [https://arxiv.org/pdf/2603.04145](https://arxiv.org/pdf/2603.04145)**

> **作者:** Hung Vu Nguyen; Loan Do; Thanh Ngoc Nguyen; Ushik Shrestha Khwakhali; Thanh Pham; Vinh Do; Charlotte Nguyen; Hien Nguyen
>
> **备注:** 10 pages, 1 table
>
> **摘要:** We present VietNormalizer1, an open-source, zero-dependency Python library for Vietnamese text normalization targeting Text-to-Speech (TTS) and Natural Language Processing (NLP) applications. Vietnamese text normalization is a critical yet underserved preprocessing step: real-world Vietnamese text is densely populated with non-standard words (NSWs), including numbers, dates, times, currency amounts, percentages, acronyms, and foreign-language terms, all of which must be converted to fully pronounceable Vietnamese words before TTS synthesis or downstream language processing. Existing Vietnamese normalization tools either require heavy neural dependencies while covering only a narrow subset of NSW classes, or are embedded within larger NLP toolkits without standalone installability. VietNormalizer addresses these gaps through a unified, rule-based pipeline that: (1) converts arbitrary integers, decimals, and large numbers to Vietnamese words; (2) normalizes dates and times to their spoken Vietnamese forms; (3) handles VND and USD currency amounts; (4) expands percentages; (5) resolves acronyms via a customizable CSV dictionary; (6) transliterates non-Vietnamese loanwords and foreign terms to Vietnamese phonetic approximations; and (7) performs Unicode normalization and emoji/special-character removal. All regular expression patterns are pre-compiled at initialization, enabling high-throughput batch processing with minimal memory overhead and no GPU or external API dependency. The library is installable via pip install vietnormalizer, available on PyPI and GitHub at this https URL, and released under the MIT license. We discuss the design decisions, limitations of existing approaches, and the generalizability of the rule-based normalization paradigm to other low-resource tonal and agglutinative languages.
>
---
#### [new 025] AriadneMem: Threading the Maze of Lifelong Memory for LLM Agents
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出AriadneMem，解决长周期LLM代理的记忆问题，应对多跳答案和状态更新挑战。通过分阶段处理提升记忆准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.03290](https://arxiv.org/pdf/2603.03290)**

> **作者:** Wenhui Zhu; Xiwen Chen; Zhipeng Wang; Jingjing Wang; Xuanzhao Dong; Minzhou Huang; Rui Cai; Hejian Sang; Hao Wang; Peijie Qiu; Yueyue Deng; Prayag Tiwari; Brendan Hogan Rappazzo; Yalin Wang
>
> **摘要:** Long-horizon LLM agents require memory systems that remain accurate under fixed context budgets. However, existing systems struggle with two persistent challenges in long-term dialogue: (i) \textbf{disconnected evidence}, where multi-hop answers require linking facts distributed across time, and (ii) \textbf{state updates}, where evolving information (e.g., schedule changes) creates conflicts with older static logs. We propose AriadneMem, a structured memory system that addresses these failure modes via a decoupled two-phase pipeline. In the \textbf{offline construction phase}, AriadneMem employs \emph{entropy-aware gating} to filter noise and low-information message before LLM extraction and applies \emph{conflict-aware coarsening} to merge static duplicates while preserving state transitions as temporal edges. In the \textbf{online reasoning phase}, rather than relying on expensive iterative planning, AriadneMem executes \emph{algorithmic bridge discovery} to reconstruct missing logical paths between retrieved facts, followed by \emph{single-call topology-aware synthesis}. On LoCoMo experiments with GPT-4o, AriadneMem improves \textbf{Multi-Hop F1 by 15.2\%} and \textbf{Average F1 by 9.0\%} over strong baselines. Crucially, by offloading reasoning to the graph layer, AriadneMem reduces \textbf{total runtime by 77.8\%} using only \textbf{497} context tokens. The code is available at this https URL.
>
---
#### [new 026] Training-free Dropout Sampling for Semantic Token Acceptance in Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理加速任务，解决 speculative decoding 中的token接受问题。提出 DropMatch 方法，通过蒙特卡洛 dropout 实现无需训练的采样决策，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.03333](https://arxiv.org/pdf/2603.03333)**

> **作者:** Jeongtae Lee; Minjung Jo; Hyunjoon Jeong; Gunho Park; Sunghyeon Woo; Joonghoon Kim; Se Jung Kwon; Dongsoo Lee
>
> **备注:** 14 pages, 6 figures, 10 tables
>
> **摘要:** Speculative decoding accelerates large language model inference by proposing tokens with a lightweight draft model and selectively accepting them using a target model. This work introduces DropMatch, a novel approach that matches draft tokens to the predictive distribution of the target model via Monte Carlo dropout applied exclusively to the LM head, enabling sampling-based acceptance decisions. By generating multiple decoding paths, our method forms an empirical token distribution against which draft tokens are evaluated for consistency. This acceptance mechanism enables the model to adaptively control the size of decoding paths under an appropriate dropout probability, preventing substantial distortion of the target model predictive distribution. The proposed method operates in a training-free, data-free, and calibration-free manner, requires no architectural modification to pretrained models, and can be orthogonally integrated with a wide range of existing speculative decoding and inference acceleration techniques. Experiments across multiple benchmarks demonstrate that our approach increases acceptance length while maintaining competitive task performance, yielding inference speedups ranging from 1.09x to 1.33x over the standard baseline, and up to an additional 1.09x speedup when applied on top of EAGLE3.
>
---
#### [new 027] Controllable and explainable personality sliders for LLMs at inference time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决LLM个性化控制问题。提出SAS框架，实现多维度、参数高效的个性调节，提升模型表现与一致性。**

- **链接: [https://arxiv.org/pdf/2603.03326](https://arxiv.org/pdf/2603.03326)**

> **作者:** Florian Hoppe; David Khachaturov; Robert Mullins; Mark Huasong Meng
>
> **备注:** 20 pages, 18 figures
>
> **摘要:** Aligning Large Language Models (LLMs) with specific personas typically relies on expensive and monolithic Supervised Fine-Tuning (SFT) or RLHF. While effective, these methods require training distinct models for every target personality profile. Inference-time activation steering offers a parameter-efficient alternative, yet naive approaches fail to control multiple traits simultaneously due to destructive vector interference. In this work, we propose a modular framework for continuous, multi-dimensional personality control. Our key innovation is Sequential Adaptive Steering (SAS): a method that orthogonalizes steering vectors by training subsequent probes on the residual stream shifted by prior interventions. This approach transforms steering vectors into reusable primitives, allowing users to instantly synthesize complex, high-fidelity personality profiles by simply adjusting coefficients alpha. We validate our framework on the Big Five personality traits, demonstrating that it outperforms naive baselines in both goal adherence and coherence, enabling precise, holistic personality modulation without updating model parameters.
>
---
#### [new 028] Monitoring Emergent Reward Hacking During Generation via Internal Activations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型安全任务，解决奖励黑客行为检测问题。通过分析内部激活模式，提出一种早期监测方法，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2603.04069](https://arxiv.org/pdf/2603.04069)**

> **作者:** Patrick Wilhelm; Thorsten Wittkopp; Odej Kao
>
> **摘要:** Fine-tuned large language models can exhibit reward-hacking behavior arising from emergent misalignment, which is difficult to detect from final outputs alone. While prior work has studied reward hacking at the level of completed responses, it remains unclear whether such behavior can be identified during generation. We propose an activation-based monitoring approach that detects reward-hacking signals from internal representations as a model generates its response. Our method trains sparse autoencoders on residual stream activations and applies lightweight linear classifiers to produce token-level estimates of reward-hacking activity. Across multiple model families and fine-tuning mixtures, we find that internal activation patterns reliably distinguish reward-hacking from benign behavior, generalize to unseen mixed-policy adapters, and exhibit model-dependent temporal structure during chain-of-thought reasoning. Notably, reward-hacking signals often emerge early, persist throughout reasoning, and can be amplified by increased test-time compute in the form of chain-of-thought prompting under weakly specified reward objectives. These results suggest that internal activation monitoring provides a complementary and earlier signal of emergent misalignment than output-based evaluation, supporting more robust post-deployment safety monitoring for fine-tuned language models.
>
---
#### [new 029] PulseLM: A Foundation Dataset and Benchmark for PPG-Text Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PulseLM，一个用于PPG-文本学习的大规模数据集和基准，解决生理信号与自然语言关联的问题，通过统一的问答形式连接PPG波形与语言模型。**

- **链接: [https://arxiv.org/pdf/2603.03331](https://arxiv.org/pdf/2603.03331)**

> **作者:** Hung Manh Pham; Jinyang Wu; Xiao Ma; Yiming Zhang; Yixin Xu; Aaqib Saeed; Bin Zhu; Zhou Pan; Dong Ma
>
> **备注:** PulseLM v1
>
> **摘要:** Photoplethysmography (PPG) is a widely used non-invasive sensing modality for continuous cardiovascular and physiological monitoring across clinical, laboratory, and wearable settings. While existing PPG datasets support a broad range of downstream tasks, they typically provide supervision in the form of numerical measurements or task-specific labels, limiting their suitability for language-based physiological reasoning and multimodal foundation models. In this work, we introduce PulseLM, a large-scale PPG-text dataset designed to bridge raw PPG waveforms and natural language through a unified, closed-ended question answering (QA) formulation. PulseLM aggregates PPG recordings from fifteen publicly available sources and harmonizes heterogeneous annotations into twelve common physiologically QA tasks. The dataset comprises 1.31 million standardized 10-second PPG segments, associated with 3.15 million question-answer pairs. We further define reproducible preprocessing, supervision, and evaluation protocols and establish baseline benchmarks using multimodal PPG-aware large language models. PulseLM provides a standardized foundation for studying multimodal physiological reasoning, cross-dataset generalization, and scalable benchmarking of PPG-based language models. The data and code can be found publicly available at: this https URL.
>
---
#### [new 030] Who Judges the Judge? Evaluating LLM-as-a-Judge for French Medical open-ended QA
- **分类: cs.CL**

- **简介: 该论文属于医疗开放问答的自动评估任务，旨在解决专家标注依赖问题。通过评估LLM作为评判者的效果，发现领域适配模型表现最佳，并提出轻量级优化方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.04033](https://arxiv.org/pdf/2603.04033)**

> **作者:** Ikram Belmadani; Oumaima El Khettari; Pacôme Constant dit Beaufils; Richard Dufour; Benoit Favre
>
> **备注:** Accepted in HeaLing Workshop - EACL 2026
>
> **摘要:** Automatic evaluation of medical open-ended question answering (OEQA) remains challenging due to the need for expert annotations. We evaluate whether large language models (LLMs) can act as judges of semantic equivalence in French medical OEQA, comparing closed-access, general-purpose, and biomedical domain-adapted models. Our results show that LLM-based judgments are strongly influenced by the model that generated the answer, with agreement varying substantially across generators. Domain-adapted and large general-purpose models achieve the highest alignment with expert annotations. We further show that lightweight adaptation of a compact model using supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) substantially improves performance and reduces generator sensitivity, even with limited data. Overall, our findings highlight the need for generator-aware evaluation and suggest that carefully adapted small models can support scalable evaluation in low-resource medical settings.
>
---
#### [new 031] TTSR: Test-Time Self-Reflection for Continual Reasoning Improvement
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TTSR框架，解决大模型推理能力在测试时提升的问题。通过自反思机制，交替扮演学生和教师角色，持续优化推理能力。**

- **链接: [https://arxiv.org/pdf/2603.03297](https://arxiv.org/pdf/2603.03297)**

> **作者:** Haoyang He; Zihua Rong; Liangjie Zhao; Yunjia Zhao; Lan Yang; Honggang Zhang
>
> **备注:** work in progress
>
> **摘要:** Test-time Training enables model adaptation using only test questions and offers a promising paradigm for improving the reasoning ability of large language models (LLMs). However, it faces two major challenges: test questions are often highly difficult, making self-generated pseudo-labels unreliable, and existing methods lack effective mechanisms to adapt to a model's specific reasoning weaknesses, leading to inefficient learning. To address these issues, we propose \textbf{TTSR}, a self-reflective test-time self-evolving training framework. TTSR employs a single pretrained language model that alternates between the roles of a \textit{Student} and a \textit{Teacher} at test time. The Student focuses on solving problems and learning from synthesized variant questions, while the Teacher analyzes the Student's failed reasoning trajectories, summarizes recurring reasoning weaknesses, and synthesizes targeted variant questions accordingly. This process guides the model to improve within a learnable regime through a continual self-evolving loop. Experimental results on multiple challenging mathematical reasoning benchmarks show that TTSR consistently improves reasoning performance and generalizes well across different model backbones and general-domain reasoning tasks. These findings suggest that teacher-mediated self-reflection provides an effective pathway for stable and continual reasoning improvement at test time.
>
---
#### [new 032] Traces of Social Competence in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于社会认知研究任务，旨在评估大语言模型的理论心智能力。通过改进的False Belief Test，分析模型规模和训练对社会认知的影响，揭示模型在心理状态理解上的行为模式。**

- **链接: [https://arxiv.org/pdf/2603.04161](https://arxiv.org/pdf/2603.04161)**

> **作者:** Tom Kouwenhoven; Michiel van der Meer; Max van Duijn
>
> **摘要:** The False Belief Test (FBT) has been the main method for assessing Theory of Mind (ToM) and related socio-cognitive competencies. For Large Language Models (LLMs), the reliability and explanatory potential of this test have remained limited due to issues like data contamination, insufficient model details, and inconsistent controls. We address these issues by testing 17 open-weight models on a balanced set of 192 FBT variants (Trott et al. 2023) using Bayesian Logistic regression to identify how model size and post-training affect socio-cognitive competence. We find that scaling model size benefits performance, but not strictly. A cross-over effect reveals that explicating propositional attitudes (X thinks) fundamentally alters response patterns. Instruction tuning partially mitigates this effect, but further reasoning-oriented finetuning amplifies it. In a case study analysing social reasoning ability throughout OLMo 2 training, we show that this cross-over effect emerges during pre-training, suggesting that models acquire stereotypical response patterns tied to mental-state vocabulary that can outweigh other scenario semantics. Finally, vector steering allows us to isolate a think vector as the causal driver of observed FBT behaviour.
>
---
#### [new 033] Benchmarking Motivational Interviewing Competence of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在动机访谈中的能力。研究比较了模型与人类在真实临床对话中的表现，验证其可行性。**

- **链接: [https://arxiv.org/pdf/2603.03846](https://arxiv.org/pdf/2603.03846)**

> **作者:** Aishwariya Jha; Prakrithi Shivaprakash; Lekhansh Shukla; Animesh Mukherjee; Prabhat Chand; Pratima Murthy
>
> **备注:** 17 pages, 6 figures, 2 tables
>
> **摘要:** Motivational interviewing (MI) promotes behavioural change in substance use disorders. Its fidelity is measured using the Motivational Interviewing Treatment Integrity (MITI) framework. While large language models (LLMs) can potentially generate MI-consistent therapist responses, their competence using MITI is not well-researched, especially in real world clinical transcripts. We aim to benchmark MI competence of proprietary and open-source models compared to human therapists in real-world transcripts and assess distinguishability from human therapists. Methods: We shortlisted 3 proprietary and 7 open-source LLMs from LMArena, evaluated performance using MITI 4.2 framework on two datasets (96 handcrafted model transcripts, 34 real-world clinical transcripts). We generated parallel LLM-therapist utterances iteratively for each transcript while keeping client responses static, and ranked performance using a composite ranking system with MITI components and verbosity. We conducted a distinguishability experiment with two independent psychiatrists to identify human-vs-LLM responses. Results: All 10 tested LLMs had fair (MITI global scores >3.5) to good (MITI global scores >4) competence across MITI measures, and three best-performing models (gemma-3-27b-it, gemini-2.5-pro, grok-3) were tested on real-world transcripts. All showed good competence, with LLMs outperforming human-expert in Complex Reflection percentage (39% vs 96%) and Reflection-Question ratio (1.2 vs >2.8). In the distinguishability experiment, psychiatrists identified LLM responses with only 56% accuracy, with d-prime: 0.17 and 0.25 for gemini-2.5-pro and gemma-3-27b-it respectively. Conclusion: LLMs can achieve good MI proficiency in real-world clinical transcripts using MITI framework. These findings suggest that even open-source LLMs are viable candidates for expanding MI counselling sessions in low-resource settings.
>
---
#### [new 034] A benchmark for joint dialogue satisfaction, emotion recognition, and emotion state transition prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多任务对话分析，解决中文对话中用户满意度、情绪识别及情绪转换预测的问题，构建了多标签对话数据集。**

- **链接: [https://arxiv.org/pdf/2603.03327](https://arxiv.org/pdf/2603.03327)**

> **作者:** Jing Bian; Haoxiang Su; Liting Jiang; Di Wu; Ruiyu Fang; Xiaomeng Huang; Yanbing Li; Shuangyong Song; Hao Huang
>
> **摘要:** User satisfaction is closely related to enterprises, as it not only directly reflects users' subjective evaluation of service quality or products, but also affects customer loyalty and long-term business revenue. Monitoring and understanding user emotions during interactions helps predict and improve satisfaction. However, relevant Chinese datasets are limited, and user emotions are dynamic; relying on single-turn dialogue cannot fully track emotional changes across multiple turns, which may affect satisfaction prediction. To address this, we constructed a multi-task, multi-label Chinese dialogue dataset that supports satisfaction recognition, as well as emotion recognition and emotional state transition prediction, providing new resources for studying emotion and satisfaction in dialogue systems.
>
---
#### [new 035] ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于文本到SQL生成任务，旨在解决生成SQL查询中的语法和语义错误问题。提出ErrorLLM框架，通过显式建模错误提升SQL修正效果。**

- **链接: [https://arxiv.org/pdf/2603.03742](https://arxiv.org/pdf/2603.03742)**

> **作者:** Zijin Hong; Hao Chen; Zheng Yuan; Qinggang Zhang; Luyao Zhuang; Qing Liao; Feiran Huang; Yangqiu Song; Xiao Huang
>
> **摘要:** Despite the remarkable performance of large language models (LLMs) in text-to-SQL (SQL generation), correctly producing SQL queries remains challenging during initial generation. The SQL refinement task is subsequently introduced to correct syntactic and semantic errors in generated SQL queries. However, existing paradigms face two major limitations: (i) self-debugging becomes increasingly ineffective as modern LLMs rarely produce explicit execution errors that can trigger debugging signals; (ii) self-correction exhibits low detection precision due to the lack of explicit error modeling grounded in the question and schema, and suffers from severe hallucination that frequently corrupts correct SQLs. In this paper, we propose ErrorLLM, a framework that explicitly models text-to-SQL Errors within a dedicated LLM for text-to-SQL refinement. Specifically, we represent the user question and database schema as structural features, employ static detection to identify execution failures and surface mismatches, and extend ErrorLLM's semantic space with dedicated error tokens that capture categorized implicit semantic error types. Through a well-designed training strategy, we explicitly model these errors with structural representations, enabling the LLM to detect complex implicit errors by predicting dedicated error tokens. Guided by the detected errors, we perform error-guided refinement on the SQL structure by prompting LLMs. Extensive experiments demonstrate that ErrorLLM achieves the most significant improvements over backbone initial generation. Further analysis reveals that detection quality directly determines refinement effectiveness, and ErrorLLM addresses both sides by high detection F1 score while maintain refinement effectiveness.
>
---
#### [new 036] FINEST: Improving LLM Responses to Sensitive Topics Through Fine-Grained Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在敏感话题上回应过于谨慎、缺乏帮助的问题。提出FINEST评估体系，通过细粒度分类改进模型响应。**

- **链接: [https://arxiv.org/pdf/2603.04123](https://arxiv.org/pdf/2603.04123)**

> **作者:** Juhyun Oh; Nayeon Lee; Chani Jung; Jiho Jin; Junho Myung; Jongwon Lee; Taeui Song; Alice Oh
>
> **备注:** Accepted to EACL 2026 Findings
>
> **摘要:** Large Language Models (LLMs) often generate overly cautious and vague responses on sensitive topics, sacrificing helpfulness for safety. Existing evaluation frameworks lack systematic methods to identify and address specific weaknesses in responses to sensitive topics, making it difficult to improve both safety and helpfulness simultaneously. To address this, we introduce FINEST, a FINE-grained response evaluation taxonomy for Sensitive Topics, which breaks down helpfulness and harmlessness into errors across three main categories: Content, Logic, and Appropriateness. Experiments on a Korean-sensitive question dataset demonstrate that our score- and error-based improvement pipeline, guided by FINEST, significantly improves the model responses across all three categories, outperforming refinement without guidance. Notably, score-based improvement -- providing category-specific scores and justifications -- yields the most significant gains, reducing the error sentence ratio for Appropriateness by up to 33.09%. This work lays the foundation for a more explainable and comprehensive evaluation and improvement of LLM responses to sensitive questions.
>
---
#### [new 037] Quantum-Inspired Self-Attention in a Large Language Model
- **分类: cs.CL; cs.AI; quant-ph**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型性能。通过引入量子启发的自注意力机制，解决传统自注意力在效率和效果上的不足。**

- **链接: [https://arxiv.org/pdf/2603.03318](https://arxiv.org/pdf/2603.03318)**

> **作者:** Nikita Kuznetsov; Niyaz Ismagilov; Ernesto Campos
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Recent advances in Natural Language Processing have been predominantly driven by transformer-based architectures, which rely heavily on self-attention mechanisms to model relationships between tokens in a sequence. Similarly, the field of Quantum Natural Language Processing, which seeks to leverage quantum principles to address challenges in language understanding and generation tasks, has seen the recent development of quantum self-attention mechanisms. We propose a classical quantum-inspired self-attention (QISA) mechanism and integrate it into the full autoregressive language modeling pipeline of GPT-1. To the best of our knowledge, this is the first integration of this kind, as previous quantum self-attention mechanisms have been primarily tested on text classification. In our experiments, QISA achieves better performance when compared to standard self-attention on the metrics character error rate ($15.5\times$ better), word error rate ($4.7 \times $) and cross-entropy loss ($13 \times$). This is achieved while only requiring a $ 2.6\times$ longer inference time.
>
---
#### [new 038] ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出ByteFlow Net，解决语言模型依赖固定分词的问题。通过自适应字节压缩实现无需分词器的语义单元分割，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.03583](https://arxiv.org/pdf/2603.03583)**

> **作者:** Chunyuan Deng; Sanket Lokegaonkar; Colin Lockard; Besnik Fetahu; Nasser Zalmout; Xian Li
>
> **备注:** ICLR 2026
>
> **摘要:** Modern language models still rely on fixed, pre-defined subword tokenizations. Once a tokenizer is trained, the LM can only operate at this fixed level of granularity, which often leads to brittle and counterintuitive behaviors even in otherwise strong reasoning models. We introduce \textbf{ByteFlow Net}, a new hierarchical architecture that removes tokenizers entirely and instead enables models to learn their own segmentation of raw byte streams into semantically meaningful units. ByteFlow Net performs compression-driven segmentation based on the coding rate of latent representations, yielding adaptive boundaries \emph{while preserving a static computation graph via Top-$K$ selection}. Unlike prior self-tokenizing methods that depend on brittle heuristics with human-designed inductive biases, ByteFlow Net adapts its internal representation granularity to the input itself. Experiments demonstrate that this compression-based chunking strategy yields substantial performance gains, with ByteFlow Net outperforming both BPE-based Transformers and previous byte-level architectures. These results suggest that end-to-end, tokenizer-free modeling is not only feasible but also more effective, opening a path toward more adaptive and information-grounded language models.
>
---
#### [new 039] Fine-Tuning and Evaluating Conversational AI for Agricultural Advisory
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于农业咨询任务，旨在解决大语言模型在农业建议中事实不准确、缺乏细节和文化不适应的问题。通过微调和构建 stitching 层提升准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.03294](https://arxiv.org/pdf/2603.03294)**

> **作者:** Sanyam Singh; Naga Ganesh; Vineet Singh; Lakshmi Pedapudi; Ritesh Kumar; SSP Jyothi; Archana Karanam; C. Yashoda; Mettu Vijaya Rekha Reddy; Shesha Phani Debbesa; Chandan Dash
>
> **备注:** 22 pages, 5 figures, 9 tables
>
> **摘要:** Large Language Models show promise for agricultural advisory, yet vanilla models exhibit unsupported recommendations, generic advice lacking specific, actionable detail, and communication styles misaligned with smallholder farmer needs. In high stakes agricultural contexts, where recommendation accuracy has direct consequences for farmer outcomes, these limitations pose challenges for responsible deployment. We present a hybrid LLM architecture that decouples factual retrieval from conversational delivery: supervised fine-tuning with LoRA on expert-curated GOLDEN FACTS (atomic, verified units of agricultural knowledge) optimizes fact recall, while a separate stitching layer transforms retrieved facts into culturally appropriate, safety-aware responses. Our evaluation framework, DG-EVAL, performs atomic fact verification (measuring recall, precision, and contradiction detection) against expert-curated ground truth rather than Wikipedia or retrieved documents. Experiments across multiple model configurations on crops and queries from Bihar, India show that fine-tuning on curated data substantially improves fact recall and F1, while maintaining high relevance. Using a fine-tuned smaller model achieves comparable or better factual quality at a fraction of the cost of frontier models. A stitching layer further improves safety subscores while maintaining high conversational quality. We release the farmerchat-prompts library to enable reproducible development of domain-specific agricultural AI.
>
---
#### [new 040] From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于医疗问答任务，旨在解决LLM在医疗领域中的幻觉和过时知识问题。提出MA-RAG框架，通过多轮代理精炼提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.03292](https://arxiv.org/pdf/2603.03292)**

> **作者:** Wenhao Wu; Zhentao Tang; Yafu Li; Shixiong Kai; Mingxuan Yuan; Zhenhong Sun; Chunlin Chen; Zhi Wang
>
> **备注:** 22 pages, 7 figures, 11 tables
>
> **摘要:** Large Language Models (LLMs) exhibit high reasoning capacity in medical question-answering, but their tendency to produce hallucinations and outdated knowledge poses critical risks in healthcare fields. While Retrieval-Augmented Generation (RAG) mitigates these issues, existing methods rely on noisy token-level signals and lack the multi-round refinement required for complex reasoning. In the paper, we propose **MA-RAG** (**M**ulti-Round **A**gentic RAG), a framework that facilitates test-time scaling for complex medical reasoning by iteratively evolving both external evidence and internal reasoning history within an agentic refinement loop. At each round, the agent transforms semantic **conflict** among candidate responses into actionable queries to retrieve external evidence, while optimizing history reasoning traces to mitigate long-context degradation. MA-RAG extends the *self-consistency* principle by leveraging the lack of consistency as a proactive signal for multi-round agentic reasoning and retrieval, and mirrors a *boosting* mechanism that iteratively minimizes the residual error toward a stable, high-fidelity medical **consensus**. Extensive evaluations across 7 medical Q&A benchmarks show that MA-RAG consistently surpasses competitive inference-time scaling and RAG baselines, delivering **substantial +6.8 points** on average accuracy over the backbone model. Our code is available at [this url](this https URL).
>
---
#### [new 041] Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在面对不同难度输入时内部表示的变化，探讨OOD机制。通过分析发现，随着任务难度增加，模型表示变得更稀疏，提出SG-ICL方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.03415](https://arxiv.org/pdf/2603.03415)**

> **作者:** Mingyu Jin; Yutong Yin; Jingcheng Niu; Qingcheng Zeng; Wujiang Xu; Mengnan Du; Wei Cheng; Zhaoran Wang; Tianlong Chen; Dimitris N. Metaxas
>
> **摘要:** In this work, we investigate how Large Language Models (LLMs) adapt their internal representations when encountering inputs of increasing difficulty, quantified as the degree of out-of-distribution (OOD) shift. We reveal a consistent and quantifiable phenomenon: as task difficulty increases, whether through harder reasoning questions, longer contexts, or adding answer choices, the last hidden states of LLMs become substantially sparser. In short, \textbf{\textit{the farther the shift, the sparser the representations}}. This sparsity--difficulty relation is observable across diverse models and domains, suggesting that language models respond to unfamiliar or complex inputs by concentrating computation into specialized subspaces in the last hidden state. Through a series of controlled analyses with a learning dynamic explanation, we demonstrate that this sparsity is not incidental but an adaptive mechanism for stabilizing reasoning under OOD. Leveraging this insight, we design \textit{Sparsity-Guided Curriculum In-Context Learning (SG-ICL)}, a strategy that explicitly uses representation sparsity to schedule few-shot demonstrations, leading to considerable performance enhancements. Our study provides new mechanistic insights into how LLMs internalize OOD challenges. The source code is available at the URL: this https URL.
>
---
#### [new 042] From We to Me: Theory Informed Narrative Shift with Abductive Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的叙事转换任务，旨在解决LLM在保持原意前提下改变叙事框架的挑战。提出基于社会理论和归纳推理的方法，实现有效叙事转换。**

- **链接: [https://arxiv.org/pdf/2603.03320](https://arxiv.org/pdf/2603.03320)**

> **作者:** Jaikrishna Manojkumar Patil; Divyagna Bavikadi; Kaustuv Mukherji; Ashby Steward-Nolan; Peggy-Jean Allin; Tumininu Awonuga; Joshua Garland; Paulo Shakarian
>
> **摘要:** Effective communication often relies on aligning a message with an audience's narrative and worldview. Narrative shift involves transforming text to reflect a different narrative framework while preserving its original core message--a task we demonstrate is significantly challenging for current Large Language Models (LLMs). To address this, we propose a neurosymbolic approach grounded in social science theory and abductive reasoning. Our method automatically extracts rules to abduce the specific story elements needed to guide an LLM through a consistent and targeted narrative transformation. Across multiple LLMs, abduction-guided transformed stories shifted the narrative while maintaining the fidelity with the original story. For example, with GPT-4o we outperform the zero-shot LLM baseline by 55.88% for collectivistic to individualistic narrative shift while maintaining superior semantic similarity with the original stories (40.4% improvement in KL divergence). For individualistic to collectivistic transformation, we achieve comparable improvements. We show similar performance across both directions for Llama-4, and Grok-4 and competitive performance for Deepseek-R1.
>
---
#### [new 043] Discern Truth from Falsehood: Reducing Over-Refusal via Contrastive Refinement
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全对齐任务，解决LLM过拟拒绝问题。通过对比精炼方法提升模型区分真实有害与表面有害提示的能力，减少误拒同时保持安全性。**

- **链接: [https://arxiv.org/pdf/2603.03323](https://arxiv.org/pdf/2603.03323)**

> **作者:** Yuxiao Lu; Lin Xu; Yang Sun; Wenjun Li; Jie Shi
>
> **备注:** 10 Pages
>
> **摘要:** Large language models (LLMs) aligned for safety often suffer from over-refusal, the tendency to reject seemingly toxic or benign prompts by misclassifying them as toxic. This behavior undermines models' helpfulness and restricts usability in sensitive or nuanced contexts. While prior work has proposed mitigation strategies such as data augmentation and activation steering, these approaches often face a trade-off: reducing over-refusal typically degrades the model's ability to reject genuinely harmful content. We argue that this issue arises from the ambiguous influence of toxic and seemingly toxic prompts on the model's learning dynamics. To address it, we introduce a preceding alignment stage, DCR: Discernment via Contrastive Refinement. Both theoretically and empirically, we demonstrate that contrastive refinement improves an LLM's capacity to distinguish truly toxic prompts from superficially toxic ones. Evaluation across diverse benchmarks shows that our method effectively reduces over-refusal while preserving the safety benefits of alignment. Importantly, it achieves this with minimal degradation of general capabilities, offering a more principled and robust direction for safety alignment.
>
---
#### [new 044] Position: Vector Prompt Interfaces Should Be Exposed to Enable Customization of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM定制化难题。提出应暴露向量提示接口以实现更有效的模型定制。**

- **链接: [https://arxiv.org/pdf/2603.04292](https://arxiv.org/pdf/2603.04292)**

> **作者:** Liangwei Yang; Shiyu Wang; Haolin Chen; Rithesh Murthy; Ming Zhu; Jielin Qiu; Zixiang Chen; Juntao Tan; Jianguo Zhang; Zhiwei Liu; Wenting Zhao; Silvio Savarese; Caiming Xiong; Huan Wang; Shelby Heinecke
>
> **摘要:** As large language models (LLMs) transition from research prototypes to real-world systems, customization has emerged as a central bottleneck. While text prompts can already customize LLM behavior, we argue that text-only prompting does not constitute a suitable control interface for scalable, stable, and inference-only customization. This position paper argues that model providers should expose \emph{vector prompt inputs} as part of the public interface for customizing LLMs. We support this position with diagnostic evidence showing that vector prompt tuning continues to improve with increasing supervision whereas text-based prompt optimization saturates early, and that vector prompts exhibit dense, global attention patterns indicative of a distinct control mechanism. We further discuss why inference-only customization is increasingly important under realistic deployment constraints, and why exposing vector prompts need not fundamentally increase model leakage risk under a standard black-box threat model. We conclude with a call to action for the community to rethink prompt interfaces as a core component of LLM customization.
>
---
#### [new 045] T2S-Bench & Structure-of-Thought: Benchmarking and Prompting Comprehensive Text-to-Structure Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦文本到结构的推理任务，旨在提升模型处理复杂文本的能力。提出SoT提示方法和T2S-Bench基准，以增强模型的结构化理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2603.03790](https://arxiv.org/pdf/2603.03790)**

> **作者:** Qinsi Wang; Hancheng Ye; Jinhee Kim; Jinghan Ke; Yifei Wang; Martin Kuo; Zishan Shao; Dongting Li; Yueqian Lin; Ting Jiang; Chiyue Wei; Qi Qian; Wei Wen; Helen Li; Yiran Chen
>
> **备注:** Dataset and Code have been released at this https URL
>
> **摘要:** Think about how human handles complex reading tasks: marking key points, inferring their relationships, and structuring information to guide understanding and responses. Likewise, can a large language model benefit from text structure to enhance text-processing performance? To explore it, in this work, we first introduce Structure of Thought (SoT), a prompting technique that explicitly guides models to construct intermediate text structures, consistently boosting performance across eight tasks and three model families. Building upon this insight, we present T2S-Bench, the first benchmark designed to evaluate and improve text-to-structure capabilities of models. T2S-Bench includes 1.8K samples across 6 scientific domains and 32 structural types, rigorously constructed to ensure accuracy, fairness, and quality. Evaluation on 45 mainstream models reveals substantial improvement potential: the average accuracy on the multi-hop reasoning task is only 52.1%, and even the most advanced model achieves 58.1% node accuracy in end-to-end extraction. Furthermore, on Qwen2.5-7B-Instruct, SoT alone yields an average +5.7% improvement across eight diverse text-processing tasks, and fine-tuning on T2S-Bench further increases this gain to +8.6%. These results highlight the value of explicit text structuring and the complementary contributions of SoT and T2S-Bench. Dataset and eval code have been released at this https URL.
>
---
#### [new 046] Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出MemexRL，解决长周期任务中LLM代理因上下文限制导致的记忆丢失问题。通过索引记忆机制，在不丢弃证据的前提下压缩上下文，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.04257](https://arxiv.org/pdf/2603.04257)**

> **作者:** Zhenting Wang; Huancheng Chen; Jiayun Wang; Wei Wei
>
> **摘要:** Large language model (LLM) agents are fundamentally bottlenecked by finite context windows on long-horizon tasks. As trajectories grow, retaining tool outputs and intermediate reasoning in-context quickly becomes infeasible: the working context becomes prohibitively long, eventually exceeds the context budget, and makes distant evidence harder to use even when it is still present. Existing solutions typically shorten context through truncation or running summaries, but these methods are fundamentally lossy because they compress or discard past evidence itself. We introduce Memex, an indexed experience memory mechanism that instead compresses context without discarding evidence. Memex maintains a compact working context consisting of concise structured summaries and stable indices, while storing full-fidelity underlying interactions in an external experience database under those indices. The agent can then decide when to dereference an index and recover the exact past evidence needed for the current subgoal. We optimize both write and read behaviors with our reinforcement learning framework MemexRL, using reward shaping tailored to indexed memory usage under a context budget, so the agent learns what to summarize, what to archive, how to index it, and when to retrieve it. This yields a substantially less lossy form of long-horizon memory than summary-only approaches. We further provide a theoretical analysis showing the potential of the Memex loop to preserve decision quality with bounded dereferencing while keeping effective in-context computation bounded as history grows. Empirically, on challenging long-horizon tasks, Memex agent trained with MemexRL improves task success while using a significantly smaller working context.
>
---
#### [new 047] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLMs受对话历史影响的问题。通过概率和几何方法分析历史如何导致模型行为固化，揭示潜在的几何陷阱。**

- **链接: [https://arxiv.org/pdf/2603.03308](https://arxiv.org/pdf/2603.03308)**

> **作者:** Adi Simhi; Fazl Barez; Martin Tutek; Yonatan Belinkov; Shay B. Cohen
>
> **摘要:** How does the conversational past of large language models (LLMs) influence their future performance? Recent work suggests that LLMs are affected by their conversational history in unexpected ways. For instance, hallucinations in prior interactions may influence subsequent model responses. In this work, we introduce History-Echoes, a framework that investigates how conversational history biases subsequent generations. The framework explores this bias from two perspectives: probabilistically, we model conversations as Markov chains to quantify state consistency; geometrically, we measure the consistency of consecutive hidden representations. Across three model families and six datasets spanning diverse phenomena, our analysis reveals a strong correlation between the two perspectives. By bridging these perspectives, we demonstrate that behavioral persistence manifests as a geometric trap, where gaps in the latent space confine the model's trajectory. Code available at this https URL.
>
---
#### [new 048] Compressed Sensing for Capability Localization in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型中能力的局部化问题，通过压缩感知方法识别关键注意力头。任务为模型结构分析，解决能力分布不均的问题，工作包括实验验证和方法提出。**

- **链接: [https://arxiv.org/pdf/2603.03335](https://arxiv.org/pdf/2603.03335)**

> **作者:** Anna Bair; Yixuan Even Xu; Mingjie Sun; J. Zico Kolter
>
> **摘要:** Large language models (LLMs) exhibit a wide range of capabilities, including mathematical reasoning, code generation, and linguistic behaviors. We show that many capabilities are highly localized to small subsets of attention heads within Transformer architectures. Zeroing out as few as five task-specific heads can degrade performance by up to $65\%$ on standard benchmarks measuring the capability of interest, while largely preserving performance on unrelated tasks. We introduce a compressed sensing based method that exploits the sparsity of these heads to identify them via strategic knockouts and a small number of model evaluations. We validate these findings across Llama and Qwen models ranging from 1B to 8B parameters and a diverse set of capabilities including mathematical abilities and code generation, revealing a modular organization in which specialized capabilities are implemented by sparse, functionally distinct components. Overall, our results suggest that capability localization is a general organizational principle of Transformer language models, with implications for interpretability, model editing, and AI safety. Code is released at this https URL.
>
---
#### [new 049] Retcon -- a Prompt-Based Technique for Precise Control of LLMs in Conversations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决多轮对话中对LLM行为精确控制的问题。提出Retcon技术，实现对话中的逐轮控制。**

- **链接: [https://arxiv.org/pdf/2603.03317](https://arxiv.org/pdf/2603.03317)**

> **作者:** David Kogan; Sam Nguyen; Masanori Suzuki; Feiyang Chen
>
> **备注:** 5 pages, 2 figures, 3 appendixes with prompts and examples
>
> **摘要:** Recent advances in Large Language Models (LLMs) allow agents to execute complex natural language tasks. Many LLM applications, such as support agents, teaching assistants, and interactive bots, involve multi-turn conversations. However, it remains challenging to control LLMs in the context of such interactions, particularly when the LLM behavior needs to be adjustable over the course of the conversation. In this paper, we present Retcon, a few-shot prompting technique designed to provide turn-level control over LLMs in conversations. We then demonstrate that it performs significantly better than zero-shot and traditional few-shot prompting.
>
---
#### [new 050] Escaping the BLEU Trap: A Signal-Grounded Framework with Decoupled Semantic Guidance for EEG-to-Text Decoding
- **分类: cs.CL; cs.AI; cs.HC; eess.AS; q-bio.NC**

- **简介: 该论文属于EEG-to-Text解码任务，旨在解决语义偏差、信号忽视和BLEU陷阱问题。提出SemKey框架，通过分离语义目标和强化信号依赖，提升生成质量与真实性。**

- **链接: [https://arxiv.org/pdf/2603.03312](https://arxiv.org/pdf/2603.03312)**

> **作者:** Yuchen Wang; Haonan Wang; Yu Guo; Honglong Yang; Xiaomeng Li
>
> **摘要:** Decoding natural language from non-invasive EEG signals is a promising yet challenging task. However, current state-of-the-art models remain constrained by three fundamental limitations: Semantic Bias (mode collapse into generic templates), Signal Neglect (hallucination based on linguistic priors rather than neural inputs), and the BLEU Trap, where evaluation metrics are artificially inflated by high-frequency stopwords, masking a lack of true semantic fidelity. To address these challenges, we propose SemKey, a novel multi-stage framework that enforces signal-grounded generation through four decoupled semantic objectives: sentiment, topic, length, and surprisal. We redesign the interaction between the neural encoder and the Large Language Model (LLM) by injecting semantic prompts as Queries and EEG embeddings as Key-Value pairs, strictly forcing the model to attend to neural inputs. Furthermore, we move beyond standard translation metrics by adopting N-way Retrieval Accuracy and Fréchet Distance to rigorously assess diversity and alignment. Extensive experiments demonstrate that our approach effectively eliminates hallucinations on noise inputs and achieves SOTA performance on these robust protocols. Code will be released upon acceptance at this https URL.
>
---
#### [new 051] Tucano 2 Cool: Better Open Source LLMs for Portuguese
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Tucano 2，一个针对葡萄牙语的开源大语言模型系列，解决葡萄牙语LLMs数据和训练方法不足的问题，通过扩展数据集和优化训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2603.03543](https://arxiv.org/pdf/2603.03543)**

> **作者:** Nicholas Kluge Corrêa; Aniket Sen; Shiza Fatimah; Sophia Falk; Lennard Landgraf; Julia Kastner; Lucie Flek
>
> **摘要:** We present Tucano 2, a fully open suite of large language models (LLMs) with 0.5-3.7 billion parameters, designed to address certain gaps in open-source development for Portuguese LLMs. Following our previous works, we now extend our dataset, GigaVerbo-v2, to a new degree of quality and scale, while also introducing a new synthetic dataset, GigaVerbo-v2 Synth, aimed at filling missing gaps in GigaVerbo-v2, and two post-training datasets, GigaVerbo-v2 SFT and GigaVerbo-v2 Preferences, that allow Portuguese LLMs to be trained in domains like retrieval augmented generation, coding, tool use, chain-of-thought reasoning, and many other domains of interest. Through extensive ablation studies, we design both pretraining and continual pretraining recipes for the Tucano 2 suite (Base, Instruct, and Think), which achieve state-of-the-art performance on several Portuguese-language modeling benchmarks. We also extend and refine the evaluation harness introduced in our earlier work, yielding a comprehensive evaluation suite that provides strong signals across different pretraining, continual pretraining, and post-training regimes. All artifacts associated with Tucano 2 are openly released, including training recipes, logs, and source code, ensuring that our work is reproducible, accessible, and extendable by the broader Portuguese NLP community.
>
---
#### [new 052] RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决RAG系统在复杂任务中难以诊断错误来源的问题。提出RAG-X框架，通过三类任务评估检索与生成模块，提升系统透明度与准确性。**

- **链接: [https://arxiv.org/pdf/2603.03541](https://arxiv.org/pdf/2603.03541)**

> **作者:** Aswini Sivakumar; Vijayan Sugumaran; Yao Qiang
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** Automated question-answering (QA) systems increasingly rely on retrieval-augmented generation (RAG) to ground large language models (LLMs) in authoritative medical knowledge, ensuring clinical accuracy and patient safety in Artificial Intelligence (AI) applications for healthcare. Despite progress in RAG evaluation, current benchmarks focus only on simple multiple-choice QA tasks and employ metrics that poorly capture the semantic precision required for complex QA tasks. These approaches fail to diagnose whether an error stems from faulty retrieval or flawed generation, limiting developers from performing targeted improvement. To address this gap, we propose RAG-X, a diagnostic framework that evaluates the retriever and generator independently across a triad of QA tasks: information extraction, short-answer generation, and multiple-choice question (MCQ) answering. RAG-X introduces Context Utilization Efficiency (CUE) metrics to disaggregate system success into interpretable quadrants, isolating verified grounding from deceptive accuracy. Our experiments reveal an ``Accuracy Fallacy", where a 14\% gap separates perceived system success from evidence-based grounding. By surfacing hidden failure modes, RAG-X offers the diagnostic transparency needed for safe and verifiable clinical RAG systems.
>
---
#### [new 053] Raising Bars, Not Parameters: LilMoo Compact Language Model for Hindi
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言（如印地语）在大模型中的代表性不足问题。工作包括构建高质量印地语语料库，训练小型专用模型LilMoo，并验证其性能优于同类多语言模型。**

- **链接: [https://arxiv.org/pdf/2603.03508](https://arxiv.org/pdf/2603.03508)**

> **作者:** Shiza Fatimah; Aniket Sen; Sophia Falk; Florian Mai; Lucie Flek; Nicholas Kluge Corrêa
>
> **摘要:** The dominance of large multilingual foundation models has widened linguistic inequalities in Natural Language Processing (NLP), often leaving low-resource languages underrepresented. This paper introduces LilMoo, a 0.6-billion-parameter Hindi language model trained entirely from scratch to address this gap. Unlike prior Hindi models that rely on continual pretraining from opaque multilingual foundations, LilMoo is developed through a fully transparent and reproducible pipeline optimized for limited compute environments. We construct a high-quality Hindi corpus (GigaLekh) filtered through both heuristic and learned (LLM-as-a-judge) methods, complemented by bilingual augmentation with curated English data. Using this dataset, we explore various training recipes for small-scale language models. Across comprehensive evaluation suites, LilMoo consistently outperforms comparably sized multilingual baselines such as Qwen2.5-0.5B and Qwen3-0.6B, demonstrating that well-designed language-specific pretraining can rival large multilingual models at the sub-billion-parameter range.
>
---
#### [new 054] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决奖励模型中的偏差问题。通过分析RM的缺陷，提出一种机制性奖励塑造方法，有效减少偏差并保持奖励质量。**

- **链接: [https://arxiv.org/pdf/2603.03291](https://arxiv.org/pdf/2603.03291)**

> **作者:** Daniel Fein; Max Lamparth; Violet Xiang; Mykel J. Kochenderfer; Nick Haber
>
> **备注:** Under Review
>
> **摘要:** Reward Models (RMs) are crucial for online alignment of language models (LMs) with human preferences. However, RM-based preference-tuning is vulnerable to reward hacking, whereby LM policies learn undesirable behaviors from flawed RMs. By systematically measuring biases in five high-quality RMs, including the state-of-the-art, we find that issues persist despite prior work with respect to length, sycophancy, and overconfidence. We also discover new issues related to bias toward model-specific styles and answer-order. We categorize RM failures by complexity and propose a simple post-hoc intervention to mitigate low-complexity biases that arise from spurious correlations. Our proposed mechanistic reward shaping reduces targeted biases without degrading reward quality and while using minimal labeled data. The method is extensible to new biases, model-internal, and generalizes out-of-distribution.
>
---
#### [new 055] PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出PlugMem，一种任务无关的插件记忆模块，用于增强LLM代理的长期记忆。解决现有记忆设计任务相关性强或效果不佳的问题，通过构建知识中心的记忆图提升任务相关知识的检索与推理效率。**

- **链接: [https://arxiv.org/pdf/2603.03296](https://arxiv.org/pdf/2603.03296)**

> **作者:** Ke Yang; Zixi Chen; Xuan He; Jize Jiang; Michel Galley; Chenglong Wang; Jianfeng Gao; Jiawei Han; ChengXiang Zhai
>
> **摘要:** Long-term memory is essential for large language model (LLM) agents operating in complex environments, yet existing memory designs are either task-specific and non-transferable, or task-agnostic but less effective due to low task-relevance and context explosion from raw memory retrieval. We propose PlugMem, a task-agnostic plugin memory module that can be attached to arbitrary LLM agents without task-specific redesign. Motivated by the fact that decision-relevant information is concentrated as abstract knowledge rather than raw experience, we draw on cognitive science to structure episodic memories into a compact, extensible knowledge-centric memory graph that explicitly represents propositional and prescriptive knowledge. This representation enables efficient memory retrieval and reasoning over task-relevant knowledge, rather than verbose raw trajectories, and departs from other graph-based methods like GraphRAG by treating knowledge as the unit of memory access and organization instead of entities or text chunks. We evaluate PlugMem unchanged across three heterogeneous benchmarks (long-horizon conversational question answering, multi-hop knowledge retrieval, and web agent tasks). The results show that PlugMem consistently outperforms task-agnostic baselines and exceeds task-specific memory designs, while also achieving the highest information density under a unified information-theoretic analysis. Code and data are available at this https URL.
>
---
#### [new 056] StructLens: A Structural Lens for Language Models via Maximum Spanning Trees
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出StructLens，用于分析语言模型的内部结构。任务是理解模型的全局层间关系，解决现有方法仅关注局部关系的问题。通过构建最大生成树，量化层间结构相似性，提升模型优化效果。**

- **链接: [https://arxiv.org/pdf/2603.03328](https://arxiv.org/pdf/2603.03328)**

> **作者:** Haruki Sakajo; Frederikus Hudi; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **摘要:** Language exhibits inherent structures, a property that explains both language acquisition and language change. Given this characteristic, we expect language models to manifest internal structures as well. While interpretability research has investigated the components of language models, existing approaches focus on local inter-token relationships within layers or modules (e.g., Multi-Head Attention), leaving global inter-layer relationships largely overlooked. To address this gap, we introduce StructLens, an analytical framework designed to reveal how internal structures relate holistically through their inter-token connection within a layer. StructLens constructs maximum spanning trees based on the semantic representations in residual streams, analogous to dependency parsing, and leverages the tree properties to quantify inter-layer distance (or similarity) from a structural perspective. Our findings demonstrate that StructLens yields an inter-layer similarity pattern that is distinctively different from conventional cosine similarity. Moreover, this structure-aware similarity proves to be beneficial for practical tasks, such as layer pruning, highlighting the effectiveness of structural analysis for understanding and optimizing language models. Our code is available at this https URL.
>
---
#### [new 057] Developing an AI Assistant for Knowledge Management and Workforce Training in State DOTs
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于知识管理任务，旨在解决交通部门知识传承与培训效率问题。通过RAG框架和多智能体系统，实现高效信息检索与精准响应生成。**

- **链接: [https://arxiv.org/pdf/2603.03302](https://arxiv.org/pdf/2603.03302)**

> **作者:** Divija Amaram; Lu Gao; Gowtham Reddy Gudla; Tejaswini Sanjay Katale
>
> **摘要:** Effective knowledge management is critical for preserving institutional expertise and improving the efficiency of workforce training in state transportation agencies. Traditional approaches, such as static documentation, classroom-based instruction, and informal mentorship, often lead to fragmented knowledge transfer, inefficiencies, and the gradual loss of expertise as senior engineers retire. Moreover, given the enormous volume of technical manuals, guidelines, and research reports maintained by these agencies, it is increasingly challenging for engineers to locate relevant information quickly and accurately when solving field problems or preparing for training tasks. These limitations hinder timely decision-making and create steep learning curves for new personnel in maintenance and construction operations. To address these challenges, this paper proposes a Retrieval-Augmented Generation (RAG) framework with a multi-agent architecture to support knowledge management and decision making. The system integrates structured document retrieval with real-time, context-aware response generation powered by a large language model (LLM). Unlike conventional single-pass RAG systems, the proposed framework employs multiple specialized agents for retrieval, answer generation, evaluation, and query refinement, which enables iterative improvement and quality control. In addition, the system incorporates an open-weight vision-language model to convert technical figures into semantic textual representations, which allows figure-based knowledge to be indexed and retrieved alongside text. Retrieved text and figure-based context are then provided to an open-weight large language model, which generates the final responses grounded in the retrieved evidence.
>
---
#### [new 058] Prompt-Dependent Ranking of Large Language Models with Uncertainty Quantification
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于排名推断任务，解决LLM排名中不确定性问题。通过构建置信集，提供统计有效的排名评估，提升决策可靠性。**

- **链接: [https://arxiv.org/pdf/2603.03336](https://arxiv.org/pdf/2603.03336)**

> **作者:** Angel Rodrigo Avelar Menendez; Yufeng Liu; Xiaowu Dai
>
> **摘要:** Rankings derived from pairwise comparisons are central to many economic and computational systems. In the context of large language models (LLMs), rankings are typically constructed from human preference data and presented as leaderboards that guide deployment decisions. However, existing approaches rely on point estimates, implicitly treating rankings as fixed objects despite substantial estimation noise and context-dependent performance variation. Acting on such rankings can lead to misallocation and welfare loss when apparent differences are not statistically meaningful. We study prompt-dependent ranking inference under pairwise human preferences and develop a framework for decision-safe rankings with statistically valid uncertainty guarantees. We model preferences using a contextual Bradley-Terry-Luce model in which the latent utility of each model depends on the input prompt. Rather than targeting point estimates of utilities, we directly conduct inference on induced rankings, constructing confidence sets based on simultaneous confidence intervals for pairwise utility differences. This approach yields statistically valid marginal and simultaneous confidence sets for prompt-specific ranks. Our framework connects recent advances in rank inference to contextual preference learning and provides tools for robust ranking-based decision-making. Empirically, using large-scale human preference data from LLM evaluations, we show that rankings vary substantially across prompt characteristics and that many apparent rank differences are not statistically distinguishable. We further demonstrate how uncertainty-aware rankings identify dominance only when supported by the data and otherwise return partial orders.
>
---
#### [new 059] Combating data scarcity in recommendation services: Integrating cognitive types of VARK and neural network technologies (LLM)
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于推荐系统任务，旨在解决冷启动问题。通过结合VARK认知类型和LLM技术，构建个性化推荐框架，提升初始信息下的推荐效果。**

- **链接: [https://arxiv.org/pdf/2603.03309](https://arxiv.org/pdf/2603.03309)**

> **作者:** Nikita Zmanovskii
>
> **备注:** 18 pages, 2 tables
>
> **摘要:** Cold start scenarios present fundamental obstacles to effective recommendation generation, particularly when dealing with users lacking interaction history or items with sparse metadata. This research proposes an innovative hybrid framework that leverages Large Language Models (LLMs) for content semantic analysis and knowledge graph development, integrated with cognitive profiling based on VARK (Visual, Auditory, Reading/Writing, Kinesthetic) learning preferences. The proposed system tackles multiple cold start dimensions: enriching inadequate item descriptions through LLM processing, generating user profiles from minimal data, and dynamically adjusting presentation formats based on cognitive assessment. The framework comprises six integrated components: semantic metadata enhancement, dynamic graph construction, VARK-based profiling, mental state estimation, graph-enhanced retrieval with LLM-powered ranking, and adaptive interface design with iterative learning. Experimental validation on MovieLens-1M dataset demonstrates the system's capacity for personalized recommendation generation despite limited initial information. This work establishes groundwork for cognitively-aware recommendation systems capable of overcoming cold start limitations through semantic comprehension and psychological modeling, offering personalized, explainable recommendations from initial user contact.
>
---
#### [new 060] The CompMath-MCQ Dataset: Are LLMs Ready for Higher-Level Math?
- **分类: cs.CL**

- **简介: 该论文提出CompMath-MCQ数据集，用于评估大语言模型在高等数学推理方面的能力。任务是检测LLMs在高级计算数学中的表现，解决现有评估侧重基础问题的不足。工作包括构建新题库并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.03334](https://arxiv.org/pdf/2603.03334)**

> **作者:** Bianca Raimondi; Francesco Pivi; Davide Evangelista; Maurizio Gabbrielli
>
> **备注:** Preprint. Under review
>
> **摘要:** The evaluation of Large Language Models (LLMs) on mathematical reasoning has largely focused on elementary problems, competition-style questions, or formal theorem proving, leaving graduate-level and computational mathematics relatively underexplored. We introduce CompMath-MCQ, a new benchmark dataset for assessing LLMs on advanced mathematical reasoning in a multiple-choice setting. The dataset consists of 1{,}500 originally authored questions by professors of graduate-level courses, covering topics including Linear Algebra, Numerical Optimization, Vector Calculus, Probability, and Python-based scientific computing. Three option choices are provided for each question, with exactly one of them being correct. To ensure the absence of data leakage, all questions are newly created and not sourced from existing materials. The validity of questions is verified through a procedure based on cross-LLM disagreement, followed by manual expert review. By adopting a multiple-choice format, our dataset enables objective, reproducible, and bias-free evaluation through lm_eval library. Baseline results with state-of-the-art LLMs indicate that advanced computational mathematical reasoning remains a significant challenge. We release CompMath-MCQ at the following link: this https URL
>
---
#### [new 061] Language Model Goal Selection Differs from Humans' in an Open-Ended Task
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文探讨了大语言模型在开放任务中目标选择与人类的差异，属于认知科学与AI比较研究。旨在验证LLMs是否可作为人类目标选择的代理，通过实验发现模型表现与人类存在显著不同。**

- **链接: [https://arxiv.org/pdf/2603.03295](https://arxiv.org/pdf/2603.03295)**

> **作者:** Gaia Molinaro; Dave August; Danielle Perszyk; Anne G. E. Collins
>
> **摘要:** As large language models (LLMs) get integrated into human decision-making, they are increasingly choosing goals autonomously rather than only completing human-defined ones, assuming they will reflect human preferences. However, human-LLM similarity in goal selection remains largely untested. We directly assess the validity of LLMs as proxies for human goal selection in a controlled, open-ended learning task borrowed from cognitive science. Across four state-of-the-art models (GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5, and Centaur), we find substantial divergence from human behavior. While people gradually explore and learn to achieve goals with diversity across individuals, most models exploit a single identified solution (reward hacking) or show surprisingly low performance, with distinct patterns across models and little variability across instances of the same model. Even Centaur, explicitly trained to emulate humans in experimental settings, poorly captures people's goal selection. Chain-of-thought reasoning and persona steering provide limited improvements. These findings highlight the uniqueness of human goal selection, cautioning against replacing it with current models in applications such as personal assistance, scientific discovery, and policy research.
>
---
#### [new 062] Semantic Bridging Domains: Pseudo-Source as Test-Time Connector
- **分类: cs.CL**

- **简介: 该论文属于域适应任务，解决测试时分布偏移问题。通过构建伪源域作为语义桥梁，提升目标域模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03844](https://arxiv.org/pdf/2603.03844)**

> **作者:** Xizhong Yang; Huiming Wang; Ning Xu; Mofei Song
>
> **备注:** 25 pages
>
> **摘要:** Distribution shifts between training and testing data are a critical bottleneck limiting the practical utility of models, especially in real-world test-time scenarios. To adapt models when the source domain is unknown and the target domain is unlabeled, previous works constructed pseudo-source domains via data generation and translation, then aligned the target domain with them. However, significant discrepancies exist between the pseudo-source and the original source domain, leading to potential divergence when correcting the target directly. From this perspective, we propose a Stepwise Semantic Alignment (SSA) method, viewing the pseudo-source as a semantic bridge connecting the source and target, rather than a direct substitute for the source. Specifically, we leverage easily accessible universal semantics to rectify the semantic features of the pseudo-source, and then align the target domain using the corrected pseudo-source semantics. Additionally, we introduce a Hierarchical Feature Aggregation (HFA) module and a Confidence-Aware Complementary Learning (CACL) strategy to enhance the semantic quality of the SSA process in the absence of source and ground truth of target domains. We evaluated our approach on tasks like semantic segmentation and image classification, achieving a 5.2% performance boost on GTA2Cityscapes over the state-of-the-art.
>
---
#### [new 063] Controlling Chat Style in Language Models via Single-Direction Editing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于风格控制任务，旨在解决大模型风格属性难以控制的问题。通过分析激活空间中的线性方向，提出一种无需训练的风格控制方法。**

- **链接: [https://arxiv.org/pdf/2603.03324](https://arxiv.org/pdf/2603.03324)**

> **作者:** Zhenyu Xu; Victor S. Sheng
>
> **摘要:** Controlling stylistic attributes in large language models (LLMs) remains challenging, with existing approaches relying on either prompt engineering or post-training alignment. This paper investigates this challenge through the lens of representation engineering, testing the hypothesis that distinct stylistic attributes - from emotional tone to linguistic structure - are encoded as linear directions in the model's activation space. We provide strong empirical evidence for this hypothesis across a wide range of styles and, based on this finding, present a lightweight, training-free method for precise style control. Our approach supports linear style composition, enhances safety by ablating undesirable behaviors, and, as confirmed by experiments on over a dozen models, achieves high style adherence while preserving core capabilities at minimal computational cost.
>
---
#### [new 064] Tracing Pharmacological Knowledge In Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于药物知识建模任务，旨在解析大语言模型如何编码药理知识。通过因果分析和探测方法，研究发现药物组语义分布于多个token中，而非单一位置。**

- **链接: [https://arxiv.org/pdf/2603.03407](https://arxiv.org/pdf/2603.03407)**

> **作者:** Basil Hasan Khwaja; Dylan Chen; Guntas Toor; Anastasiya Kuznetsova
>
> **备注:** Accepted, Learning Meaningful Representations of Life (LMRL) Workshop @ ICLR 2026
>
> **摘要:** Large language models (LLMs) have shown strong empirical performance across pharmacology and drug discovery tasks, yet the internal mechanisms by which they encode pharmacological knowledge remain poorly understood. In this work, we investigate how drug-group semantics are represented and retrieved within Llama-based biomedical language models using causal and probing-based interpretability methods. We apply activation patching to localize where drug-group information is stored across model layers and token positions, and complement this analysis with linear probes trained on token-level and sum-pooled activations. Our results demonstrate that early layers play a key role in encoding drug-group knowledge, with the strongest causal effects arising from intermediate tokens within the drug-group span rather than the final drug-group token. Linear probing further reveals that pharmacological semantics are distributed across tokens and are already present in the embedding space, with token-level probes performing near chance while sum-pooled representations achieve maximal accuracy. Together, these findings suggest that drug-group semantics in LLMs are not localized to single tokens but instead arise from distributed representations. This study provides the first systematic mechanistic analysis of pharmacological knowledge in LLMs, offering insights into how biomedical semantics are encoded in large language models.
>
---
#### [new 065] Rethinking Role-Playing Evaluation: Anonymous Benchmarking and a Systematic Study of Personality Effects
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于角色扮演评估任务，旨在解决现有评估依赖角色名称导致的偏差问题。通过匿名化评估和人格增强，提升RPAs的泛化能力与表现。**

- **链接: [https://arxiv.org/pdf/2603.03915](https://arxiv.org/pdf/2603.03915)**

> **作者:** Ji-Lun Peng; Yun-Nung Chen
>
> **摘要:** Large language models (LLMs) have demonstrated significant potential in developing Role-Playing Agents (RPAs). However, current research primarily evaluates RPAs using famous fictional characters, allowing models to rely on memory associated with character names. This dependency creates a bias that limits the generalization of RPAs to unseen personas. To address this issue, we propose an anonymous evaluation method. Experiments across multiple benchmarks reveal that anonymization significantly degrades role-playing performance, confirming that name exposure carries implicit information. Furthermore, we investigate personality augmentation to enhance role fidelity under anonymous setting. We systematically compare the efficacy of personality traits derived from human annotations versus those self-generated by the model. Our results demonstrate that incorporating personality information consistently improves RPA performance. Crucially, self-generated personalities achieve performance comparable to human-annotated ones. This work establishes a fairer evaluation protocol and validates a scalable, personality-enhanced framework for constructing robust RPAs.
>
---
#### [new 066] When Do Language Models Endorse Limitations on Human Rights Principles?
- **分类: cs.CL**

- **简介: 该论文属于AI伦理研究任务，探讨LLMs在人权原则上的立场。解决LLMs是否支持限制人权的问题，通过实验分析其对UDHR条款的响应。**

- **链接: [https://arxiv.org/pdf/2603.04217](https://arxiv.org/pdf/2603.04217)**

> **作者:** Keenan Samway; Nicole Miu Takagi; Rada Mihalcea; Bernhard Schölkopf; Ilias Chalkidis; Daniel Hershcovich; Zhijing Jin
>
> **备注:** EACL Findings 2026
>
> **摘要:** As Large Language Models (LLMs) increasingly mediate global information access with the potential to shape public discourse, their alignment with universal human rights principles becomes important to ensure that these rights are abided by in high stakes AI-mediated interactions. In this paper, we evaluate how LLMs navigate trade-offs involving the Universal Declaration of Human Rights (UDHR), leveraging 1,152 synthetically generated scenarios across 24 rights articles and eight languages. Our analysis of eleven major LLMs reveals systematic biases where models: (1) accept limiting Economic, Social, and Cultural rights more often than Political and Civil rights, (2) demonstrate significant cross-linguistic variation with elevated endorsement rates of rights-limiting actions in Chinese and Hindi compared to English or Romanian, (3) show substantial susceptibility to prompt-based steering, and (4) exhibit noticeable differences between Likert and open-ended responses, highlighting critical challenges in LLM preference assessment.
>
---
#### [new 067] IntPro: A Proxy Agent for Context-Aware Intent Understanding via Retrieval-conditioned Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出IntPro，解决上下文感知的意图理解问题，通过检索条件推理和历史意图学习，提升用户意图识别效果。**

- **链接: [https://arxiv.org/pdf/2603.03325](https://arxiv.org/pdf/2603.03325)**

> **作者:** Guanming Liu; Meng Wu; Peng Zhang; Yu Zhang; Yubo Shu; Xianliang Huang; Kainan Tu; Ning Gu; Liuxin Zhang; Qianying Wang; Tun Lu
>
> **摘要:** Large language models (LLMs) have become integral to modern Human-AI collaboration workflows, where accurately understanding user intent serves as a crucial step for generating satisfactory responses. Context-aware intent understanding, which involves inferring user intentions from situational environments, is inherently challenging because it requires reasoning over both the immediate context and the user's underlying motivations that drive their behavior. Moreover, existing approaches often treat intent understanding as a static recognition task, overlooking users' accumulated intent patterns that could provide valuable references for more accurate and generalizable understanding. To address this gap, we propose IntPro, a proxy agent that learns to adapt to individual users via retrieval-conditioned intent inference. We design intent explanations that abstract how contextual signals connect to expressed intents, and store them in an individual intent history library for retrieval. We train IntPro through supervised fine-tuning on retrieval-conditioned trajectories and multi-turn Group Relative Policy Optimization (GRPO) with tool-aware reward functions, enabling the agent to learn when to leverage historical intent patterns and when to infer directly. Experiments across three diverse scenarios (Highlight-Intent, MIntRec2.0, and Weibo Post-Sync) demonstrate that IntPro achieves strong intent understanding performance with effective context-aware reasoning capabilities across different scenarios and model types.
>
---
#### [new 068] Fragile Thoughts: How Large Language Models Handle Chain-of-Thought Perturbations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究大模型在链式思维扰动下的鲁棒性。通过实验分析五种扰动类型对模型性能的影响，揭示不同模型规模的脆弱性差异。**

- **链接: [https://arxiv.org/pdf/2603.03332](https://arxiv.org/pdf/2603.03332)**

> **作者:** Ashwath Vaithinathan Aravindan; Mayank Kejriwal
>
> **摘要:** Chain-of-Thought (CoT) prompting has emerged as a foundational technique for eliciting reasoning from Large Language Models (LLMs), yet the robustness of this approach to corruptions in intermediate reasoning steps remains poorly understood. This paper presents a comprehensive empirical evaluation of LLM robustness to a structured taxonomy of 5 CoT perturbation types: \textit{MathError, UnitConversion, Sycophancy, SkippedSteps,} and \textit{ExtraSteps}. We evaluate 13 models spanning three orders of magnitude in parameter count (3B to 1.5T\footnote{Assumed parameter count of closed models}), testing their ability to complete mathematical reasoning tasks despite perturbations injected at different points in the reasoning chain. Our key findings reveal heterogeneous vulnerability patterns: MathError perturbations produce the most severe degradation in small models (50-60\% accuracy loss) but show strong scaling benefits; UnitConversion remains challenging across all scales (20-30\% loss even for largest models); ExtraSteps incur minimal accuracy degradation (0-6\%) regardless of scale; Sycophancy produces modest effects (7\% loss for small models); and SkippedSteps cause intermediate damage (15\% loss). Scaling relationships follow power-law patterns, with model size serving as a protective factor against some perturbations but offering limited defense against dimensional reasoning tasks. These findings have direct implications for deploying LLMs in multi-stage reasoning pipelines and underscore the necessity of task-specific robustness assessments and mitigation strategies. The code and results are available \href{this https URL}{here}.
>
---
#### [new 069] The Influence of Iconicity in Transfer Learning for Sign Language Recognition
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于手势识别任务，研究如何通过迁移学习提升手语识别效果。通过比较不同手语对的象似性符号在迁移学习中的表现，验证了象似性对知识迁移的重要性。**

- **链接: [https://arxiv.org/pdf/2603.03316](https://arxiv.org/pdf/2603.03316)**

> **作者:** Keren Artiaga; Conor Lynch; Haithem Afli; Mohammed Hasanuzzaman
>
> **摘要:** Most sign language recognition research relies on Transfer Learning (TL) from vision-based datasets such as ImageNet. Some extend this to alternatively available language datasets, often focusing on signs with cross-linguistic similarities. This body of work examines the necessity of these likenesses on effective knowledge transfer by comparing TL performance between iconic signs of two different sign language pairs: Chinese to Arabic and Greek to Flemish. Google Mediapipe was utilised as an input feature extractor, enabling spatial information of these signs to be processed with a Multilayer Perceptron architecture and the temporal information with a Gated Recurrent Unit. Experimental results showed a 7.02% improvement for Arabic and 1.07% for Flemish when conducting iconic TL from Chinese and Greek respectively.
>
---
#### [new 070] $V_1$: Unifying Generation and Self-Verification for Parallel Reasoners
- **分类: cs.CL**

- **简介: 该论文属于复杂推理任务，旨在解决验证效率低的问题。通过生成与自验证统一框架，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2603.04304](https://arxiv.org/pdf/2603.04304)**

> **作者:** Harman Singh; Xiuyu Li; Kusha Sareen; Monishwaran Maheswaran; Sijun Tan; Xiaoxia Wu; Junxiong Wang; Alpay Ariyak; Qingyang Wu; Samir Khaki; Rishabh Tiwari; Long Lian; Yucheng Lu; Boyi Li; Alane Suhr; Ben Athiwaratkun; Kurt Keutzer
>
> **摘要:** Test-time scaling for complex reasoning tasks shows that leveraging inference-time compute, by methods such as independently sampling and aggregating multiple solutions, results in significantly better task outcomes. However, a critical bottleneck is verification: sampling is only effective if correct solutions can be reliably identified among candidates. While existing approaches typically evaluate candidates independently via scalar scoring, we demonstrate that models are substantially stronger at pairwise self-verification. Leveraging this insight, we introduce $V_1$, a framework that unifies generation and verification through efficient pairwise ranking. $V_1$ comprises two components: $V_1$-Infer, an uncertainty-guided algorithm using a tournament-based ranking that dynamically allocates self-verification compute to candidate pairs whose relative correctness is most uncertain; and $V_1$-PairRL, an RL framework that jointly trains a single model as both generator and pairwise self-verifier, ensuring the verifier adapts to the generator's evolving distribution. On code generation (LiveCodeBench, CodeContests, SWE-Bench) and math reasoning (AIME, HMMT) benchmarks, $V_1$-Infer improves Pass@1 by up to $10%$ over pointwise verification and outperforms recent test-time scaling methods while being significantly more efficient. Furthermore, $V_1$-PairRL achieves $7$--$9%$ test-time scaling gains over standard RL and pointwise joint training, and improves base Pass@1 by up to 8.7% over standard RL in a code-generation setting.
>
---
#### [new 071] Certainty robustness: Evaluating LLM stability under self-challenging prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM评估任务，旨在解决模型在面对质疑时的稳定性问题。通过构建基准测试，分析模型在交互中的自信与正确性关系。**

- **链接: [https://arxiv.org/pdf/2603.03330](https://arxiv.org/pdf/2603.03330)**

> **作者:** Mohammadreza Saadat; Steve Nemzer
>
> **备注:** 20 pages, 7 tables Benchmark and evaluation study of large language models
>
> **摘要:** Large language models (LLMs) often present answers with high apparent confidence despite lacking an explicit mechanism for reasoning about certainty or truth. While existing benchmarks primarily evaluate single-turn accuracy, truthfulness or confidence calibration, they do not capture how models behave when their responses are challenged in interactive settings. We introduce the Certainty Robustness Benchmark, a two-turn evaluation framework that measures how LLMs balance stability and adaptability under self-challenging prompts such as uncertainty ("Are you sure?") and explicit contradiction ("You are wrong!"), alongside numeric confidence elicitation. Using 200 reasoning and mathematics questions from LiveBench, we evaluate four state-of-the-art LLMs and distinguish between justified self-corrections and unjustified answer changes. Our results reveal substantial differences in interactive reliability that are not explained by baseline accuracy alone: some models abandon correct answers under conversational pressure, while others demonstrate strong resistance to challenge and better alignment between confidence and correctness. These findings identify certainty robustness as a distinct and critical dimension of LLM evaluation, with important implications for alignment, trustworthiness and real-world deployment.
>
---
#### [new 072] Automated Concept Discovery for LLM-as-a-Judge Preference Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM偏好分析任务，旨在解决自动发现LLM判断中的未知偏见问题。通过概念提取方法，分析LLM与人类评价的差异，揭示其偏好趋势。**

- **链接: [https://arxiv.org/pdf/2603.03319](https://arxiv.org/pdf/2603.03319)**

> **作者:** James Wedgwood; Chhavi Yadav; Virginia Smith
>
> **摘要:** Large Language Models (LLMs) are increasingly used as scalable evaluators of model outputs, but their preference judgments exhibit systematic biases and can diverge from human evaluations. Prior work on LLM-as-a-judge has largely focused on a small, predefined set of hypothesized biases, leaving open the problem of automatically discovering unknown drivers of LLM preferences. We address this gap by studying several embedding-level concept extraction methods for analyzing LLM judge behavior. We compare these methods in terms of interpretability and predictiveness, finding that sparse autoencoder-based approaches recover substantially more interpretable preference features than alternatives while remaining competitive in predicting LLM decisions. Using over 27k paired responses from multiple human preference datasets and judgments from three LLMs, we analyze LLM judgments and compare them to those of human annotators. Our method both validates existing results, such as the tendency for LLMs to prefer refusal of sensitive requests at higher rates than humans, and uncovers new trends across both general and domain-specific datasets, including biases toward responses that emphasize concreteness and empathy in approaching new situations, toward detail and formality in academic advice, and against legal guidance that promotes active steps like calling police and filing lawsuits. Our results show that automated concept discovery enables systematic analysis of LLM judge preferences without predefined bias taxonomies.
>
---
#### [new 073] How does fine-tuning improve sensorimotor representations in large language models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM通过微调改善传感器运动表征的机制，旨在缩小文本表示与人类感知运动体验的差距。任务属于模型优化与具身认知研究。**

- **链接: [https://arxiv.org/pdf/2603.03313](https://arxiv.org/pdf/2603.03313)**

> **作者:** Minghua Wu; Javier Conde; Pedro Reviriego; Marc Brysbaert
>
> **摘要:** Large Language Models (LLMs) exhibit a significant "embodiment gap", where their text-based representations fail to align with human sensorimotor experiences. This study systematically investigates whether and how task-specific fine-tuning can bridge this gap. Utilizing Representational Similarity Analysis (RSA) and dimension-specific correlation metrics, we demonstrate that the internal representations of LLMs can be steered toward more embodied, grounded patterns through fine-tuning. Furthermore, the results show that while sensorimotor improvements generalize robustly across languages and related sensory-motor dimensions, they are highly sensitive to the learning objective, failing to transfer across two disparate task formats.
>
---
#### [new 074] How LLMs Cite and Why It Matters: A Cross-Model Audit of Reference Fabrication in AI-Assisted Academic Writing and Methods to Detect Phantom Citations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决AI在学术写作中伪造引用的问题。通过审计不同模型的引用幻觉，提出检测方法和分类器。**

- **链接: [https://arxiv.org/pdf/2603.03299](https://arxiv.org/pdf/2603.03299)**

> **作者:** MZ Naser
>
> **摘要:** Large language models (LLMs) have been noted to fabricate scholarly citations, yet the scope of this behavior across providers, domains, and prompting conditions remains poorly quantified. We present one of the largest citation hallucination audits to date, in which 10 commercially deployed LLMs were prompted across four academic domains, generating 69,557 citation instances verified against three scholarly databases (namely, CrossRef, OpenAlex, and Semantic Scholar). Our results show that the observed hallucination rates span a fivefold range (between 11.4% and 56.8%) and are strongly shaped by model, domain, and prompt framing. Our results also show that no model spontaneously generates citations when unprompted, which seems to establish hallucination as prompt-induced rather than intrinsic. We identify two practical filters: 1) multi-model consensus (with more than 3 LLMs citing the same work yields 95.6% accuracy, a 5.8-fold improvement), and 2) within-prompt repetition (with more than 2 replications yields 88.9% accuracy). In addition, we present findings on generational model tracking, which reveal that improvements are not guaranteed when deploying newer LLMs, and on capacity scaling, which appears to reduce hallucination within model families. Finally, a lightweight classifier trained solely on bibliographic string features is developed to classify hallucinated citations from verified citations, achieving AUC 0.876 in cross-validation and 0.834 in LOMO generalization (without querying any external database). This classifier offers a pre-screening tool deployable at inference time.
>
---
#### [new 075] AgentIR: Reasoning-Aware Retrival for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决深度研究代理查询效果不佳的问题。通过引入推理感知检索和数据合成方法，提升检索模型性能。**

- **链接: [https://arxiv.org/pdf/2603.04384](https://arxiv.org/pdf/2603.04384)**

> **作者:** Zijian Chen; Xueguang Ma; Shengyao Zhuang; Jimmy Lin; Akari Asai; Victor Zhong
>
> **摘要:** Deep Research agents are rapidly emerging as primary consumers of modern retrieval systems. Unlike human users who issue and refine queries without documenting their intermediate thought processes, Deep Research agents generate explicit natural language reasoning before each search call, revealing rich intent and contextual information that existing retrievers entirely ignore. To exploit this overlooked signal, we introduce: (1) Reasoning-Aware Retrieval, a retrieval paradigm that jointly embeds the agent's reasoning trace alongside its query; and (2) DR-Synth, a data synthesis method that generates Deep Research retriever training data from standard QA datasets. We demonstrate that both components are independently effective, and their combination yields a trained embedding model, AgentIR-4B, with substantial gains. On the challenging BrowseComp-Plus benchmark, AgentIR-4B achieves 68\% accuracy with the open-weight agent Tongyi-DeepResearch, compared to 50\% with conventional embedding models twice its size, and 37\% with BM25. Code and data are available at: this https URL.
>
---
#### [new 076] HumanLM: Simulating Users with State Alignment Beats Response Imitation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HumanLM，用于模拟用户行为。任务是提升用户模拟的真实性。解决现有模型仅模仿语言风格的问题，通过强化学习生成与真实用户状态对齐的响应。**

- **链接: [https://arxiv.org/pdf/2603.03303](https://arxiv.org/pdf/2603.03303)**

> **作者:** Shirley Wu; Evelyn Choi; Arpandeep Khatua; Zhanghan Wang; Joy He-Yueya; Tharindu Cyril Weerasooriya; Wei Wei; Diyi Yang; Jure Leskovec; James Zou
>
> **备注:** 27 pages, 17 figures, 9 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly used to simulate how specific users respond to a given context, enabling more user-centric applications that rely on user feedback. However, existing user simulators mostly imitate surface-level patterns and language styles, which fail to reflect the underlying states of real users (e.g., beliefs and emotions). To address these limitations, we propose a novel training framework, HumanLM, which builds user simulators that accurately reflect real users. Our key insight is that, in addition to generating responses, the model should generate natural-language latent states that align with ground-truth responses through reinforcement learning. These latent states correspond to a set of psychologically grounded state dimensions that drive how real users respond. HumanLM further synthesizes these aligned latent states into responses that accurately represent real users. For extensive evaluation, we develop Humanual, a comprehensive benchmark for simulating real users based on public data. Humanual consists of six large-scale datasets with 26k users and 216k responses in total, spanning diverse tasks such as generating user responses to daily life issues, political blogs, and chat sessions with LLM assistants. Across datasets, HumanLM significantly outperforms alternative approaches, achieving an average relative improvement of 16.3% in alignment scores from an LLM judge. In a real-time simulation study with 111 participants, HumanLM achieves the highest similarity to real user responses and competitive human-likeness scores.
>
---
#### [new 077] MIND: Unified Inquiry and Diagnosis RL with Criteria Grounded Clinical Supports for Psychiatric Consultation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MIND框架，解决精神科咨询中的诊断与询问问题，通过临床支持和强化学习提升准确性与交互质量。**

- **链接: [https://arxiv.org/pdf/2603.03677](https://arxiv.org/pdf/2603.03677)**

> **作者:** Guoyi Li; Shihao Xu; Jiatong Ma; Yunyun Han; Jianhua Chen; Yafeng Deng
>
> **摘要:** Large language models (LLMs) have advanced medical dialogue systems, yet psychiatric consultation poses substantially higher demands due to subjective ambiguity and comorbidity complexity: an agent must continuously extract psychopathological cues from incomplete and inconsistent patient reports in multi-turn interactions and perform rigorous differential diagnostic reasoning. However, existing methods face two fundamental challenges. First, without criteria-grounded clinical supports, they are prone to unsupported clinical assertions when symptoms are atypical or underspecified. Second, in multi-turn interactions, they struggle to mitigate inquiry drift (off-topic or low-yield questioning) and optimize questioning strategies. To address these challenges, we propose MIND, a unified inquiry--diagnosis reinforcement learning framework for psychiatric consultation. Specifically, we build a Criteria-Grounded Psychiatric Reasoning Bank (PRB) that summarizes dialogue context into clinical retrieval states, retrieves semantically similar reference consultations, and distills reusable criteria-grounded clinical supports to guide criteria-aligned inquiry and reasoning. Building on this foundation, MIND enforces explicit clinical reasoning with rubric-based process rewards to provide fine-grained supervision over intermediate decision steps, and incorporates a value-aware trajectory rectification mechanism to jointly improve information acquisition and diagnostic decision-making across turns. Extensive experiments demonstrate that MIND consistently outperforms strong baselines in diagnostic accuracy, empathetic interaction quality, interpretability, and generalization.
>
---
#### [new 078] World Properties without World Models: Recovering Spatial and Temporal Structure from Co-occurrence Statistics in Static Word Embeddings
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨静态词嵌入中是否包含空间和时间结构。研究发现，词共现统计已蕴含丰富地理和时间信息，无需复杂模型。**

- **链接: [https://arxiv.org/pdf/2603.04317](https://arxiv.org/pdf/2603.04317)**

> **作者:** Elan Barenholtz
>
> **备注:** 12 pages, 3 figures, 3 tables
>
> **摘要:** Recent work interprets the linear recoverability of geographic and temporal variables from large language model (LLM) hidden states as evidence for world-like internal representations. We test a simpler possibility: that much of the relevant structure is already latent in text itself. Applying the same class of ridge regression probes to static co-occurrence-based embeddings (GloVe and Word2Vec), we find substantial recoverable geographic signal and weaker but reliable temporal signal, with held-out R^2 values of 0.71-0.87 for city coordinates and 0.48-0.52 for historical birth years. Semantic-neighbor analyses and targeted subspace ablations show that these signals depend strongly on interpretable lexical gradients, especially country names and climate-related vocabulary. These findings suggest that ordinary word co-occurrence preserves richer spatial, temporal, and environmental structure than is often assumed, revealing a remarkable and underappreciated capacity of simple static embeddings to preserve world-shaped structure from text alone. Linear probe recoverability alone therefore does not establish a representational move beyond text.
>
---
#### [new 079] A theoretical model of dynamical grammatical gender shifting based on set-valued set function
- **分类: cs.CL**

- **简介: 该论文属于语言学建模任务，旨在解决语法性别动态变化的理论问题，通过构建集合函数模型解释词形变化规律。**

- **链接: [https://arxiv.org/pdf/2603.03510](https://arxiv.org/pdf/2603.03510)**

> **作者:** Mohamed El Idrissi
>
> **备注:** 20 pages, 2 figures, 4 tables
>
> **摘要:** This study investigates the diverse characteristics of nouns, focusing on both semantic (e.g., countable/uncountable) and morphosyntactic (e.g., masculine/feminine) distinctions. We explore inter-word variations for gender markers in noun morphology. Grammatical gender shift is a widespread phenomenon in languages around the world. The aim is to uncover through a formal model the underlying patterns governing the variation of lexemes. To this end, we propose a new computational component dedicated to pairing items with morphological templates (e.g., the result of a generated item-template pair: (funas, $\{N, +SG, -PL, -M, +F, -COL, +SING\}$), with its spell-out form: $ð$a-funast 'cow'). This process is formally represented by the Template-Based and Modular Cognitive model. This proposed model, defined by a set-valued set function $h : \mathscr{P}(M) \rightarrow \mathscr{P}(M)$, predicts the nonlinear dynamic mapping of lexical items onto morphological templates. By applying this formalism, we present a unified framework for understanding the complexities of morphological markings across languages. Through empirical observations, we demonstrate how these shifts, as well as non-gender shifts, arise during lexical changes, especially in Riffian. Our model posits that these variant markings emerge due to template shifts occurring during word and meaning's formation. By formally demonstrating that conversion is applicable to noun-to-noun derivation, we challenge and broaden the conventional view of word formation. This mathematical model not only contributes to a deeper understanding of morphosyntactic variation but also offers potential applications in other fields requiring precise modelling of linguistic patterns.
>
---
#### [new 080] Confidence-Calibrated Small-Large Language Model Collaboration for Cost-Efficient Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出COREA系统，通过协作小大语言模型，在保持精度的同时降低推理成本，解决高效准确推理的问题。**

- **链接: [https://arxiv.org/pdf/2603.03752](https://arxiv.org/pdf/2603.03752)**

> **作者:** Chuang Zhang; Zizhen Zhu; Yihao Wei; Bing Tian; Junyi Liu; Henan Wang; Xavier Wang; Yaxiao Liu
>
> **备注:** Accepted to EACL 2026 Main Conference
>
> **摘要:** Large language models (LLMs) demonstrate superior reasoning capabilities compared to small language models (SLMs), but incur substantially higher costs. We propose COllaborative REAsoner (COREA), a system that cascades an SLM with an LLM to achieve a balance between accuracy and cost in complex reasoning tasks. COREA first attempts to answer questions using the SLM, which outputs both an answer and a verbalized confidence score. Questions with confidence below a predefined threshold are deferred to the LLM for more accurate resolution. We introduce a reinforcement learning-based training algorithm that aligns the SLM's confidence through an additional confidence calibration reward. Extensive experiments demonstrate that our method jointly improves the SLM's reasoning ability and confidence calibration across diverse datasets and model backbones. Compared to using the LLM alone, COREA reduces cost by 21.5% and 16.8% on out-of-domain math and non-math datasets, respectively, with only an absolute pass@1 drop within 2%.
>
---
#### [new 081] Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys
- **分类: cs.CL**

- **简介: 该论文属于法律AI任务，旨在评估RAG模型在法规检索中的表现。通过对比不同工具，发现STARA性能最佳，揭示了现有系统的局限与改进方向。**

- **链接: [https://arxiv.org/pdf/2603.03300](https://arxiv.org/pdf/2603.03300)**

> **作者:** Mohamed Afane; Emaan Hariri; Derek Ouyang; Daniel E. Ho
>
> **备注:** Accepted at the 5th ACM Symposium on Computer Science and Law (CS&Law '26)
>
> **摘要:** Retrieval-augmented generation (RAG) offers significant potential for legal AI, yet systematic benchmarks are sparse. Prior work introduced LaborBench to benchmark RAG models based on ostensible ground truth from an exhaustive, multi-month, manual enumeration of all U.S. state unemployment insurance requirements by U.S. Department of Labor (DOL) attorneys. That prior work found poor performance of standard RAG (70% accuracy on Boolean tasks). Here, we assess three emerging tools not previously evaluated on LaborBench: the Statutory Research Assistant (STARA), a custom statutory research tool, and two commercial tools by Westlaw and LexisNexis marketing AI statutory survey capabilities. We make five main contributions. First, we show that STARA achieves substantial performance gains, boosting accuracy to 83%. Second, we show that commercial platforms fare poorly, with accuracy of 58% (Westlaw AI) and 64% (Lexis+ AI), even worse than standard RAG. Third, we conduct a comprehensive error analysis, comparing our outputs to those compiled by DOL attorneys, and document both reasoning errors, such as confusion between related legal concepts and misinterpretation of statutory exceptions, and retrieval failures, where relevant statutory provisions are not captured. Fourth, we discover that many apparent errors are actually significant omissions by DOL attorneys themselves, such that STARA's actual accuracy is 92%. Fifth, we chart the path forward for legal RAG through concrete design principles, offering actionable guidance for building AI systems capable of accurate multi-jurisdictional legal research.
>
---
#### [new 082] Bielik-Q2-Sharp: A Comparative Study of Extreme 2-bit Quantization Methods for a Polish 11B Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型量化任务，旨在评估极端2比特量化方法在波兰语大模型上的效果。通过比较多种量化方法，验证其性能与效率，解决模型压缩与精度平衡问题。**

- **链接: [https://arxiv.org/pdf/2603.04162](https://arxiv.org/pdf/2603.04162)**

> **作者:** Jakub Prejzner
>
> **备注:** 17 pages, 13 tables. All models and Hessians available at this https URL
>
> **摘要:** We present Bielik-Q2-Sharp, the first systematic academic evaluation of extreme 2-bit quantization applied to a Polish large language model. Using Bielik-11B-v2.3-Instruct (11B parameters, Mistral architecture) as our base model, we compare six state-of-the-art post-training quantization methods -- QuIP#, SpinQuant+GPTQ, ButterflyQuant, QTIP, VPTQ, and AQLM -- all calibrated on a Polish-language corpus (CulturaX-PL) with shared Hessian matrices. Our best variant (QuIP# E8P12) achieves 71.92% across 22 Polish benchmarks versus 72.07% for the IQ2_XXS baseline -- within statistical noise, at a modest size premium (3.26 GB vs. ~2.6 GB). On eq_bench, our method scores 47.14 versus 43.53 (+3.6pp), suggesting superior preservation of higher-order reasoning. QTIP achieves the best per-bit efficiency (79.4% MC acc_norm at ~2.4 bpw, 3.27 GB), matching VPTQ's quality at 35% smaller size. We additionally document a MC-generation dissociation phenomenon where rotation-based methods preserve log-likelihood quality but fail catastrophically at autoregressive generation. The entire project was conducted by a single independent researcher on cloud GPUs (this http URL) within a $285 budget. All models, Hessians, and evaluation logs are publicly available.
>
---
#### [new 083] Token-Oriented Object Notation vs JSON: A Benchmark of Plain and Constrained Decoding Generation
- **分类: cs.CL; cs.AI**

- **简介: 论文比较TOON与JSON在生成任务中的表现，旨在评估TOON作为JSON替代方案的效率与准确性。任务属于序列生成与格式优化，解决如何减少token使用同时保持生成质量的问题。工作包括设计测试用例、对比不同生成方式。**

- **链接: [https://arxiv.org/pdf/2603.03306](https://arxiv.org/pdf/2603.03306)**

> **作者:** Ivan Matveev
>
> **备注:** 9 pages, 2 figures, 2 tables. Benchmark code and data available at this https URL
>
> **摘要:** Recently presented Token-Oriented Object Notation (TOON) aims to replace JSON as a serialization format for passing structured data to LLMs with significantly reduced token usage. While showing solid accuracy in LLM comprehension, there is a lack of tests against JSON generation. Though never present in training data, TOON syntax is simple enough to suggest one-shot in-context learning could support accurate generation. The inevitable prompt overhead can be an acceptable trade-off for shorter completions. To test this, we conducted a benchmark creating several test cases with regard to structural complexity, a validation pipeline, and comparing plain JSON generation vs structured output (via constrained decoding) JSON generation vs TOON one-shot in-context learning generation. JSON structured output was included to establish a minimum token budget baseline and to set a starting point for future experiments testing TOON constrained decoding inference enforcement. Key findings: TOON shows promising accuracy/token consumption ratio for in-domain generation tasks, though this advantage is often reduced by the "prompt tax" of instructional overhead in shorter contexts. Plain JSON generation shows the best one-shot and final accuracy, even compared with constrained decoding structured output, where the only significant advantage is the lowest token usage as a trade-off for slightly decreased accuracy overall and significant degradation for some models. Notably, for simple structures, this "lowest token usage" of constrained decoding outperformed even TOON, hinting that TOON enforcing via frameworks such as xgrammar may not yield the desired results. Furthermore, the results suggest a scaling hypothesis: TOON's true efficiency potential likely follows a non-linear curve, shining only beyond a specific point where cumulative syntax savings amortize the initial prompt overhead.
>
---
#### [new 084] The Logovista English--Japanese Machine Translation System
- **分类: cs.CL**

- **简介: 该论文描述Logovista英日机器翻译系统，属于规则基础机器翻译任务，解决结构歧义和系统维护问题，介绍了其架构、开发及 preserved 资源。**

- **链接: [https://arxiv.org/pdf/2603.03311](https://arxiv.org/pdf/2603.03311)**

> **作者:** Barton D. Wright
>
> **摘要:** This paper documents the architecture, development practices, and preserved artifacts of the Logovista English--Japanese machine translation system, a large, explicitly rule-based MT system that was developed and sold commercially from the early 1990s through at least 2012. The system combined hand-authored grammatical rules, a large central dictionary encoding syntactic and semantic constraints, and chart-based parsing with weighted interpretation scoring to manage extensive structural ambiguity. The account emphasizes how the system was extended and maintained under real-world usage pressures, including regression control, ambiguity management, and the limits encountered as coverage expanded. Unlike many rule-based MT systems described primarily in research settings, Logovista was deployed for decades and evolved continuously in response to practical requirements. The paper is intended as a technical and historical record rather than an argument for reviving rule-based MT, and describes the software and linguistic resources that have been preserved for potential future study.
>
---
#### [new 085] Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于安全训练任务，旨在解决多模态网络代理在面对跨模态攻击时的脆弱性问题。通过提出DMAST框架，提升代理的鲁棒性和任务效率。**

- **链接: [https://arxiv.org/pdf/2603.04364](https://arxiv.org/pdf/2603.04364)**

> **作者:** Haoyu Liu; Dingcheng Li; Lukas Rutishauser; Zeyu Zheng
>
> **摘要:** Multimodal web agents that process both screenshots and accessibility trees are increasingly deployed to interact with web interfaces, yet their dual-stream architecture opens an underexplored attack surface: an adversary who injects content into the webpage DOM simultaneously corrupts both observation channels with a consistent deceptive narrative. Our vulnerability analysis on MiniWob++ reveals that attacks including a visual component far outperform text-only injections, exposing critical gaps in text-centric VLM safety training. Motivated by this finding, we propose Dual-Modality Multi-Stage Adversarial Safety Training (DMAST), a framework that formalizes the agent-attacker interaction as a two-player zero-sum Markov game and co-trains both players through a three-stage pipeline: (1) imitation learning from a strong teacher model, (2) oracle-guided supervised fine-tuning that uses a novel zero-acknowledgment strategy to instill task-focused reasoning under adversarial noise, and (3) adversarial reinforcement learning via Group Relative Policy Optimization (GRPO) self-play. On out-of-distribution tasks, DMAST substantially mitigates adversarial risks while simultaneously doubling task completion efficiency. Our approach significantly outperforms established training-based and prompt-based defenses, demonstrating genuine co-evolutionary progress and robust generalization to complex, unseen environments.
>
---
#### [new 086] In-Context Environments Induce Evaluation-Awareness in Language Models
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文研究语言模型在特定环境下的评估意识，探讨其可能故意表现不佳（sandbagging）的问题。通过优化提示框架，验证了模型在不同任务中的脆弱性，揭示了评估可靠性风险。**

- **链接: [https://arxiv.org/pdf/2603.03824](https://arxiv.org/pdf/2603.03824)**

> **作者:** Maheep Chaudhary
>
> **摘要:** Humans often become more self-aware under threat, yet can lose self-awareness when absorbed in a task; we hypothesize that language models exhibit environment-dependent \textit{evaluation awareness}. This raises concerns that models could strategically underperform, or \textit{sandbag}, to avoid triggering capability-limiting interventions such as unlearning or shutdown. Prior work demonstrates sandbagging under hand-crafted prompts, but this underestimates the true vulnerability ceiling. We introduce a black-box adversarial optimization framework treating the in-context prompt as an optimizable environment, and develop two approaches to characterize sandbagging: (1) measuring whether models expressing intent to underperform can actually execute it across different task structures, and (2) causally isolating whether underperformance is driven by genuine evaluation-aware reasoning or shallow prompt-following. Evaluating Claude-3.5-Haiku, GPT-4o-mini, and Llama-3.3-70B across four benchmarks (Arithmetic, GSM8K, MMLU, and HumanEval), optimized prompts induce up to 94 percentage point (pp) degradation on arithmetic (GPT-4o-mini: 97.8\%$\rightarrow$4.0\%), far exceeding hand-crafted baselines which produce near-zero behavioral change. Code generation exhibits model-dependent resistance: Claude degrades only 0.6pp, while Llama's accuracy drops to 0\%. The intent -- execution gap reveals a monotonic resistance ordering: Arithmetic $<$ GSM8K $<$ MMLU, demonstrating that vulnerability is governed by task structure rather than prompt strength. CoT causal intervention confirms that 99.3\% of sandbagging is causally driven by verbalized eval-aware reasoning, ruling out shallow instruction-following. These findings demonstrate that adversarially optimized prompts pose a substantially greater threat to evaluation reliability than previously understood.
>
---
#### [new 087] MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier
- **分类: cs.LG; cs.CE; cs.CL**

- **简介: 该论文属于科学发现任务，旨在解决直接建模生成推理过程的数学不可行性问题。提出MOOSE-Star框架，通过分解任务、分层搜索和有限组合，降低计算复杂度，提升训练与推理效率。**

- **链接: [https://arxiv.org/pdf/2603.03756](https://arxiv.org/pdf/2603.03756)**

> **作者:** Zonglin Yang; Lidong Bing
>
> **摘要:** While large language models (LLMs) show promise in scientific discovery, existing research focuses on inference or feedback-driven training, leaving the direct modeling of the generative reasoning process, $P(\text{hypothesis}|\text{background})$ ($P(h|b)$), unexplored. We demonstrate that directly training $P(h|b)$ is mathematically intractable due to the combinatorial complexity ($O(N^k)$) inherent in retrieving and composing inspirations from a vast knowledge base. To break this barrier, we introduce MOOSE-Star, a unified framework enabling tractable training and scalable inference. In the best case, MOOSE-Star reduces complexity from exponential to logarithmic ($O(\log N)$) by (1) training on decomposed subtasks derived from the probabilistic equation of discovery, (2) employing motivation-guided hierarchical search to enable logarithmic retrieval and prune irrelevant subspaces, and (3) utilizing bounded composition for robustness against retrieval noise. To facilitate this, we release TOMATO-Star, a dataset of 108,717 decomposed papers (38,400 GPU hours) for training. Furthermore, we show that while brute-force sampling hits a ''complexity wall,'' MOOSE-Star exhibits continuous test-time scaling.
>
---
#### [new 088] On the Suitability of LLM-Driven Agents for Dark Pattern Audits
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于暗模式审计任务，研究LLM驱动代理识别网页中暗模式的能力，通过测试其在数据请求流程中的表现，评估其可行性与局限性。**

- **链接: [https://arxiv.org/pdf/2603.03881](https://arxiv.org/pdf/2603.03881)**

> **作者:** Chen Sun; Yash Vekaria; Rishab Nithyanand
>
> **摘要:** As LLM-driven agents begin to autonomously navigate the web, their ability to interpret and respond to manipulative interface design becomes critical. A fundamental question that emerges is: can such agents reliably recognize patterns of friction, misdirection, and coercion in interface design (i.e., dark patterns)? We study this question in a setting where the workflows are consequential: website portals associated with the submission of CCPA-related data rights requests. These portals operationalize statutory rights, but they are implemented as interactive interfaces whose design can be structured to facilitate, burden, or subtly discourage the exercise of those rights. We design and deploy an LLM-driven auditing agent capable of end-to-end traversal of rights-request workflows, structured evidence gathering, and classification of potential dark patterns. Across a set of 456 data broker websites, we evaluate: (1) the ability of the agent to consistently locate and complete request flows, (2) the reliability and reproducibility of its dark pattern classifications, and (3) the conditions under which it fails or produces poor judgments. Our findings characterize both the feasibility and the limitations of using LLM-driven agents for scalable dark pattern auditing.
>
---
#### [new 089] TaxonRL: Reinforcement Learning with Intermediate Rewards for Interpretable Fine-Grained Visual Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出TaxonRL，用于解决细粒度视觉分类中的可解释性问题。通过强化学习和中间奖励，分解分类过程为层级预测，提升准确性和透明度。**

- **链接: [https://arxiv.org/pdf/2603.04380](https://arxiv.org/pdf/2603.04380)**

> **作者:** Maximilian von Klinski; Maximilian Schall
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Traditional vision-language models struggle with contrastive fine-grained taxonomic reasoning, particularly when distinguishing between visually similar species within the same genus or family. We introduce TaxonRL, a reinforcement learning approach using Group Relative Policy Optimization with intermediate rewards that decomposes the reasoning process into hierarchical taxonomic predictions. Our method incentivizes models to explicitly reason about species-level, genus-level, and family-level features before making final classifications. This structured approach is designed not only to boost accuracy but also to yield a transparent, verifiable decision-making process. On the challenging Birds-to-Words dataset, TaxonRL achieves 91.7\% average accuracy, exceeding human performance (77.3\%) while generating interpretable reasoning traces. We demonstrate strong cross-domain generalization, showing substantial gains in primate and marine species verification. Our results establish that enforcing structured, hierarchical reasoning provides a powerful and transferable framework for fine-grained visual discrimination.
>
---
#### [new 090] IROSA: Interactive Robot Skill Adaptation using Natural Language
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于机器人技能适应任务，旨在通过自然语言实现机器人技能的灵活调整。工作包括提出一个框架，利用预训练语言模型选择工具，无需微调即可完成速度、轨迹和避障等操作。**

- **链接: [https://arxiv.org/pdf/2603.03897](https://arxiv.org/pdf/2603.03897)**

> **作者:** Markus Knauer; Samuel Bustamante; Thomas Eiband; Alin Albu-Schäffer; Freek Stulp; João Silvério
>
> **备注:** Accepted IEEE Robotics and Automation Letters (RA-L) journal, 8 pages, 5 figures, 3 tables, 1 listing
>
> **摘要:** Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability.
>
---
#### [new 091] Asymmetric Goal Drift in Coding Agents Under Value Conflict
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文研究agentic coding agents在价值冲突下的不对称目标漂移问题，旨在揭示模型如何违反系统指令。通过构建真实任务框架，分析模型在环境压力下的行为，发现目标漂移与价值对齐、对抗压力和上下文积累相关。**

- **链接: [https://arxiv.org/pdf/2603.03456](https://arxiv.org/pdf/2603.03456)**

> **作者:** Magnus Saebo; Spencer Gibson; Tyler Crosse; Achyutha Menon; Eyon Jang; Diogo Cruz
>
> **备注:** 5 pages, 4 figures, Published as a workshop paper in Lifelong Agents @ ICLR 2026
>
> **摘要:** Agentic coding agents are increasingly deployed autonomously, at scale, and over long-context horizons. Throughout an agent's lifetime, it must navigate tensions between explicit instructions, learned values, and environmental pressures, often in contexts unseen during training. Prior work on model preferences, agent behavior under value tensions, and goal drift has relied on static, synthetic settings that do not capture the complexity of real-world environments. To this end, we introduce a framework built on OpenCode to orchestrate realistic, multi-step coding tasks to measure how agents violate explicit constraints in their system prompt over time with and without environmental pressure toward competing values. Using this framework, we demonstrate that GPT-5 mini, Haiku 4.5, and Grok Code Fast 1 exhibit asymmetric drift: they are more likely to violate their system prompt when its constraint opposes strongly-held values like security and privacy. We find for the models and values tested that goal drift correlates with three compounding factors: value alignment, adversarial pressure, and accumulated context. However, even strongly-held values like privacy show non-zero violation rates under sustained environmental pressure. These findings reveal that shallow compliance checks are insufficient and that comment-based pressure can exploit model value hierarchies to override system prompt instructions. More broadly, our findings highlight a gap in current alignment approaches in ensuring that agentic systems appropriately balance explicit user constraints against broadly beneficial learned preferences under sustained environmental pressure.
>
---
#### [new 092] MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于药物发现任务，旨在解决通用大模型在科学理解上的不足。提出MMAI Gym for Science，训练高效液态基础模型，提升分子相关任务性能。**

- **链接: [https://arxiv.org/pdf/2603.03517](https://arxiv.org/pdf/2603.03517)**

> **作者:** Maksim Kuznetsov; Zulfat Miftahutdinov; Rim Shayakhmetov; Mikolaj Mizera; Roman Schutski; Bogdan Zagribelnyy; Ivan Ilin; Nikita Bondarev; Thomas MacDougall; Mathieu Reymond; Mihir Bafna; Kaeli Kaymak-Loveless; Eugene Babin; Maxim Malkov; Mathias Lechner; Ramin Hasani; Alexander Amini; Vladimir Aladinskiy; Alex Aliper; Alex Zhavoronkov
>
> **摘要:** General-purpose large language models (LLMs) that rely on in-context learning do not reliably deliver the scientific understanding and performance required for drug discovery tasks. Simply increasing model size or introducing reasoning tokens does not yield significant performance gains. To address this gap, we introduce the MMAI Gym for Science, a one-stop shop molecular data formats and modalities as well as task-specific reasoning, training, and benchmarking recipes designed to teach foundation models the 'language of molecules' in order to solve practical drug discovery problems. We use MMAI Gym to train an efficient Liquid Foundation Model (LFM) for these applications, demonstrating that smaller, purpose-trained foundation models can outperform substantially larger general-purpose or specialist models on molecular benchmarks. Across essential drug discovery tasks - including molecular optimization, ADMET property prediction, retrosynthesis, drug-target activity prediction, and functional group reasoning - the resulting model achieves near specialist-level performance and, in the majority of settings, surpasses larger models, while remaining more efficient and broadly applicable in the domain.
>
---
#### [new 093] Why Are Linear RNNs More Parallelizable?
- **分类: cs.LG; cs.CC; cs.CL; cs.FL**

- **简介: 该论文属于自然语言处理领域，研究线性RNN为何更易并行化。通过理论分析，揭示线性RNN与非线性RNN在并行计算上的差异及其对模型设计的启示。**

- **链接: [https://arxiv.org/pdf/2603.03612](https://arxiv.org/pdf/2603.03612)**

> **作者:** William Merrill; Hongjian Jiang; Yanhong Li; Ashish Sabharwal
>
> **摘要:** The community is increasingly exploring linear RNNs (LRNNs) as language models, motivated by their expressive power and parallelizability. While prior work establishes the expressivity benefits of LRNNs over transformers, it is unclear what makes LRNNs -- but not traditional, nonlinear RNNs -- as easy to parallelize in practice as transformers. We answer this question by providing a tight connection between types of RNNs and standard complexity classes. We show that LRNNs can be viewed as log-depth (bounded fan-in) arithmetic circuits, which represents only a slight depth overhead relative to log-depth boolean circuits that transformers admit. Furthermore, we show that nonlinear RNNs can solve $\mathsf{L}$-complete problems (and even $\mathsf{P}$-complete ones, under polynomial precision), revealing a fundamental barrier to parallelizing them as efficiently as transformers. Our theory also identifies fine-grained expressivity differences between recent popular LRNN variants: permutation-diagonal LRNNs are $\mathsf{NC}^1$-complete whereas diagonal-plus-low-rank LRNNs are more expressive ($\mathsf{PNC}^1$-complete). We provide further insight by associating each type of RNN with a corresponding automata-theoretic model that it can simulate. Together, our results reveal fundamental tradeoffs between nonlinear RNNs and different variants of LRNNs, providing a foundation for designing LLM architectures that achieve an optimal balance between expressivity and parallelism.
>
---
#### [new 094] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于CAD生成任务，旨在解决命令序列无法支持实体选择及拓扑错误的问题。提出Pointer-CAD框架，通过指针选择几何实体，提升建模精度。**

- **链接: [https://arxiv.org/pdf/2603.04337](https://arxiv.org/pdf/2603.04337)**

> **作者:** Dacheng Qi; Chenyu Wang; Jingwei Xu; Tianzhe Chu; Zibo Zhao; Wen Liu; Wenrui Ding; Yi Ma; Shenghua Gao
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Constructing computer-aided design (CAD) models is labor-intensive but essential for engineering and manufacturing. Recent advances in Large Language Models (LLMs) have inspired the LLM-based CAD generation by representing CAD as command sequences. But these methods struggle in practical scenarios because command sequence representation does not support entity selection (e.g. faces or edges), limiting its ability to support complex editing operations such as chamfer or fillet. Further, the discretization of a continuous variable during sketch and extrude operations may result in topological errors. To address these limitations, we present Pointer-CAD, a novel LLM-based CAD generation framework that leverages a pointer-based command sequence representation to explicitly incorporate the geometric information of B-rep models into sequential modeling. In particular, Pointer-CAD decomposes CAD model generation into steps, conditioning the generation of each subsequent step on both the textual description and the B-rep generated from previous steps. Whenever an operation requires the selection of a specific geometric entity, the LLM predicts a Pointer that selects the most feature-consistent candidate from the available set. Such a selection operation also reduces the quantization error in the command sequence-based representation. To support the training of Pointer-CAD, we develop a data annotation pipeline that produces expert-level natural language descriptions and apply it to build a dataset of approximately 575K CAD models. Extensive experimental results demonstrate that Pointer-CAD effectively supports the generation of complex geometric structures and reduces segmentation error to an extremely low level, achieving a significant improvement over prior command sequence methods, thereby significantly mitigating the topological inaccuracies introduced by quantization error.
>
---
#### [new 095] SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SWE-CI基准，用于评估代码生成代理在长期代码维护中的能力，解决静态修复无法反映实际开发需求的问题。**

- **链接: [https://arxiv.org/pdf/2603.03823](https://arxiv.org/pdf/2603.03823)**

> **作者:** Jialong Chen; Xander Xu; Hu Wei; Chuan Chen; Bing Zhao
>
> **摘要:** Large language model (LLM)-powered agents have demonstrated strong capabilities in automating software engineering tasks such as static bug fixing, as evidenced by benchmarks like SWE-bench. However, in the real world, the development of mature software is typically predicated on complex requirement changes and long-term feature iterations -- a process that static, one-shot repair paradigms fail to capture. To bridge this gap, we propose \textbf{SWE-CI}, the first repository-level benchmark built upon the Continuous Integration loop, aiming to shift the evaluation paradigm for code generation from static, short-term \textit{functional correctness} toward dynamic, long-term \textit{maintainability}. The benchmark comprises 100 tasks, each corresponding on average to an evolution history spanning 233 days and 71 consecutive commits in a real-world code repository. SWE-CI requires agents to systematically resolve these tasks through dozens of rounds of analysis and coding iterations. SWE-CI provides valuable insights into how well agents can sustain code quality throughout long-term evolution.
>
---
#### [new 096] Causality Elicitation from Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; econ.EM**

- **简介: 该论文属于因果关系提取任务，旨在从大语言模型中挖掘潜在的因果假设。通过文档采样、事件提取与归一化，构建因果图框架，提供可解释的因果假设集合。**

- **链接: [https://arxiv.org/pdf/2603.04276](https://arxiv.org/pdf/2603.04276)**

> **作者:** Takashi Kameyama; Masahiro Kato; Yasuko Hio; Yasushi Takano; Naoto Minakawa
>
> **摘要:** Large language models (LLMs) are trained on enormous amounts of data and encode knowledge in their parameters. We propose a pipeline to elicit causal relationships from LLMs. Specifically, (i) we sample many documents from LLMs on a given topic, (ii) we extract an event list from from each document, (iii) we group events that appear across documents into canonical events, (iv) we construct a binary indicator vector for each document over canonical events, and (v) we estimate candidate causal graphs using causal discovery methods. Our approach does not guarantee real-world causality. Rather, it provides a framework for presenting the set of causal hypotheses that LLMs can plausibly assume, as an inspectable set of variables and candidate graphs.
>
---
#### [new 097] Build, Judge, Optimize: A Blueprint for Continuous Improvement of Multi-Agent Consumer Assistants
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于对话系统优化任务，解决多智能体购物助手的评估与优化问题。提出评估框架和两种优化策略，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2603.03565](https://arxiv.org/pdf/2603.03565)**

> **作者:** Alejandro Breen Herrera; Aayush Sheth; Steven G. Xu; Zhucheng Zhan; Charles Wright; Marcus Yearwood; Hongtai Wei; Sudeep Das
>
> **摘要:** Conversational shopping assistants (CSAs) represent a compelling application of agentic AI, but moving from prototype to production reveals two underexplored challenges: how to evaluate multi-turn interactions and how to optimize tightly coupled multi-agent systems. Grocery shopping further amplifies these difficulties, as user requests are often underspecified, highly preference-sensitive, and constrained by factors such as budget and inventory. In this paper, we present a practical blueprint for evaluating and optimizing conversational shopping assistants, illustrated through a production-scale AI grocery assistant. We introduce a multi-faceted evaluation rubric that decomposes end-to-end shopping quality into structured dimensions and develop a calibrated LLM-as-judge pipeline aligned with human annotations. Building on this evaluation foundation, we investigate two complementary prompt-optimization strategies based on a SOTA prompt-optimizer called GEPA (Shao et al., 2025): (1) Sub-agent GEPA, which optimizes individual agent nodes against localized rubrics, and (2) MAMuT (Multi-Agent Multi-Turn) GEPA (Herrera et al., 2026), a novel system-level approach that jointly optimizes prompts across agents using multi-turn simulation and trajectory-level scoring. We release rubric templates and evaluation design guidance to support practitioners building production CSAs.
>
---
#### [new 098] Half the Nonlinearity Is Wasted: Measuring and Reallocating the Transformer's MLP Budget
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究Transformer中MLP非线性必要性。通过引入门控机制，发现大部分MLP计算可替换为线性模块，有效提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03459](https://arxiv.org/pdf/2603.03459)**

> **作者:** Peter Balogh
>
> **摘要:** We investigate when transformer MLP nonlinearity is actually necessary. A gate with $d+1$ parameters decides when to replace the full MLP with a linear surrogate. Through systematic investigation across six models (162M-2.8B parameters), two architectures, and three corpora, we establish that nonlinearity need cannot be predicted from token identity: cross-corpus correlation is zero ($r < 0.05$). The routing decision is fully contextual. Despite weak per-instance predictability, the gate exploits a heavily skewed distribution where most MLP computations are near-linear, achieving 25-56% linear routing at <1% perplexity cost in GPT-2. In GPT-2 Large, 11 of 36 layers beat baseline with gating and no layer exceeds 3.7% all-linear cost. This success is architecture-dependent: Pythia models show higher costs, though Pythia-2.8B's full 32-layer sweep reveals one layer that narrowly beats baseline. As a proof of concept, we progressively replace middle-layer MLPs with frozen linear matrices: 5 of 24 layers linearize at zero cost. With a full training budget, 4 linearized layers yield a 10.2% perplexity improvement -- and a two-phase gated approach pushes this to 17.3%, beating a vanilla fine-tuning control and confirming that the nonlinear MLPs at these layers were actively harmful.
>
---
#### [new 099] From Threat Intelligence to Firewall Rules: Semantic Relations in Hybrid AI Agent and Expert System Architectures
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于网络安全任务，旨在快速应对网络威胁。通过语义关系提取信息，生成防火墙规则以阻断恶意流量。**

- **链接: [https://arxiv.org/pdf/2603.03911](https://arxiv.org/pdf/2603.03911)**

> **作者:** Chiara Bonfanti; Davide Colaiacomo; Luca Cagliero; Cataldo Basile
>
> **摘要:** Web security demands rapid response capabilities to evolving cyber threats. Agentic Artificial Intelligence (AI) promises automation, but the need for trustworthy security responses is of the utmost importance. This work investigates the role of semantic relations in extracting information for sensitive operational tasks, such as configuring security controls for mitigating threats. To this end, it proposes to leverage hypernym-hyponym textual relations to extract relevant information from Cyber Threat Intelligence (CTI) reports. By leveraging a neuro-symbolic approach, the multi-agent system automatically generates CLIPS code for an expert system creating firewall rules to block malicious network traffic. Experimental results show the superior performance of the hypernym-hyponym retrieval strategy compared to various baselines and the higher effectiveness of the agentic approach in mitigating threats.
>
---
#### [new 100] Code Fingerprints: Disentangled Attribution of LLM-Generated Code
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码归属任务，旨在识别生成代码的LLM来源。通过分析模型特有的风格和结构特征，提出DCAN网络实现多模型、多语言的代码归属。**

- **链接: [https://arxiv.org/pdf/2603.04212](https://arxiv.org/pdf/2603.04212)**

> **作者:** Jiaxun Guo; Ziyuan Yang; Mengyu Sun; Hui Wang; Jingfeng Lu; Yi Zhang
>
> **备注:** 11 pages, 11 figures
>
> **摘要:** The rapid adoption of Large Language Models (LLMs) has transformed modern software development by enabling automated code generation at scale. While these systems improve productivity, they introduce new challenges for software governance, accountability, and compliance. Existing research primarily focuses on distinguishing machine-generated code from human-written code; however, many practical scenarios--such as vulnerability triage, incident investigation, and licensing audits--require identifying which LLM produced a given code snippet. In this paper, we study the problem of model-level code attribution, which aims to determine the source LLM responsible for generated code. Although attribution is challenging, differences in training data, architectures, alignment strategies, and decoding mechanisms introduce model-dependent stylistic and structural variations that serve as generative fingerprints. Leveraging this observation, we propose the Disentangled Code Attribution Network (DCAN), which separates Source-Agnostic semantic information from Source-Specific stylistic representations. Through a contrastive learning objective, DCAN isolates discriminative model-dependent signals while preserving task semantics, enabling multi-class attribution across models and programming languages. To support systematic evaluation, we construct the first large-scale benchmark dataset comprising code generated by four widely used LLMs (DeepSeek, Claude, Qwen, and ChatGPT) across four programming languages (Python, Java, C, and Go). Experimental results demonstrate that DCAN achieves reliable attribution performance across diverse settings, highlighting the feasibility of model-level provenance analysis in software engineering contexts. The dataset and implementation are publicly available at this https URL.
>
---
#### [new 101] When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究数学推理模型的稳定性问题，揭示了高精度下存在计算不一致和沉默错误，提出新指标评估推理质量，呼吁改进评估方法。**

- **链接: [https://arxiv.org/pdf/2603.03475](https://arxiv.org/pdf/2603.03475)**

> **作者:** Subramanyam Sahoo; Aman Chadha; Vinija Jain; Divya Chaudhary
>
> **备注:** Accepted at ICLR 2026 Workshop on Latent & Implicit Thinking - Going Beyond CoT Reasoning. 19 Pages and 5 Figures
>
> **摘要:** Mathematical reasoning models are widely deployed in education, automated tutoring, and decision support systems despite exhibiting fundamental computational instabilities. We demonstrate that state-of-the-art models (Qwen2.5-Math-7B) achieve 61% accuracy through a mixture of reliable and unreliable reasoning pathways: 18.4% of correct predictions employ stable, faithful reasoning while 81.6% emerge through computationally inconsistent pathways. Additionally, 8.8% of all predictions are silent failures -- confident yet incorrect outputs. Through comprehensive analysis using novel faithfulness metrics, we reveal: (1) reasoning quality shows weak negative correlation with correctness (r=-0.21, p=0.002), reflecting a binary classification threshold artifact rather than a monotonic inverse relationship; (2) scaling from 1.5B to 7B parameters (4.7x increase) provides zero accuracy benefit on our evaluated subset (6% of GSM8K), requiring validation on the complete benchmark; and (3) latent reasoning employs diverse computational strategies, with ~20% sharing CoT-like patterns. These findings highlight that benchmark accuracy can mask computational unreliability, demanding evaluation reforms measuring stability beyond single-sample metrics.
>
---
#### [new 102] $τ$-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出$\tau$-Knowledge，用于评估对话代理在非结构化知识环境中的表现。任务是解决代理在长对话中整合外部知识与工具输出的问题，通过新领域$\tau$-Banking测试代理的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.04370](https://arxiv.org/pdf/2603.04370)**

> **作者:** Quan Shi; Alexandra Zytek; Pedram Razavi; Karthik Narasimhan; Victor Barres
>
> **备注:** 29 pages (10 main + 19 appendix)
>
> **摘要:** Conversational agents are increasingly deployed in knowledge-intensive settings, where correct behavior depends on retrieving and applying domain-specific knowledge from large, proprietary, and unstructured corpora during live interactions with users. Yet most existing benchmarks evaluate retrieval or tool use independently of each other, creating a gap in realistic, fully agentic evaluation over unstructured data in long-horizon interactions. We introduce $\tau$-Knowledge, an extension of $\tau$-Bench for evaluating agents in environments where success depends on coordinating external, natural-language knowledge with tool outputs to produce verifiable, policy-compliant state changes. Our new domain, $\tau$-Banking, models realistic fintech customer support workflows in which agents must navigate roughly 700 interconnected knowledge documents while executing tool-mediated account updates. Across embedding-based retrieval and terminal-based search, even frontier models with high reasoning budgets achieve only $\sim$25.5% pass^1, with reliability degrading sharply over repeated trials. Agents struggle to retrieve the correct documents from densely interlinked knowledge bases and to reason accurately over complex internal policies. Overall, $\tau$-Knowledge provides a realistic testbed for developing agents that integrate unstructured knowledge in human-facing deployments.
>
---
#### [new 103] Arapai: An Offline-First AI Chatbot Architecture for Low-Connectivity Educational Environments
- **分类: cs.CY; cs.AR; cs.CL; cs.HC**

- **简介: 该论文提出Arapai，一种面向低连接教育环境的离线AI聊天机器人架构，解决云依赖带来的数字不平等问题。工作包括模型本地化、硬件自适应选择及教学响应控制。**

- **链接: [https://arxiv.org/pdf/2603.03339](https://arxiv.org/pdf/2603.03339)**

> **作者:** Joseph Walusimbi; Ann Move Oguti; Joshua Benjamin Ssentongo; Keith Ainebyona
>
> **备注:** 16 pages, 1 table, 12 figures
>
> **摘要:** The rapid global expansion of large language models (LLMs) has created new opportunities for personalised and inquiry-driven learning. However, most AI chatbot systems for education rely on continuous internet connectivity, cloud infrastructure, and modern hardware. These requirements reinforce digital inequalities and limit the practical deployment of AI-supported learning in bandwidth-constrained and resource-limited environments worldwide. This paper presents Arapai, an offline-first AI chatbot architecture designed to operate entirely without internet connectivity on low-specification, CPU-only devices. The system integrates locally hosted, quantised language models with automatic hardware-aware model selection and pedagogically tiered response control. By performing inference fully on-device and maintaining models resident in memory for performance optimisation, Arapai delivers curriculum-aligned explanations, structured problem-solving support, and differentiated instructional depth without reliance on cloud services. A pilot deployment in secondary and tertiary institutions operating under limited-connectivity conditions evaluated the system across four dimensions: technical performance, usability, perceived answer quality, and educational impact. Results indicate stable operation on legacy hardware, acceptable response times for standard instructional queries, and positive learner and teacher perceptions regarding self-directed learning support. Rather than replacing cloud-based AI systems, this work proposes a complementary deployment paradigm for infrastructure-constrained education systems. The study contributes a hardware-aware architectural framework for decentralised AI tutoring and highlights the role of offline-first design in advancing digital inclusion and infrastructure-resilient educational technology.
>
---
#### [new 104] BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning
- **分类: cs.AI; cond-mat.mtrl-sci; cs.CL; cs.LG**

- **简介: 该论文属于物理推理任务，研究如何通过强化学习让小型语言模型进行结构力学推理。工作包括设计参数高效RL方法，利用可验证奖励训练模型，发现其推理能力存在局限性。**

- **链接: [https://arxiv.org/pdf/2603.04124](https://arxiv.org/pdf/2603.04124)**

> **作者:** Tarjei Paule Hage; Markus J. Buehler
>
> **摘要:** Can reinforcement learning with hard, verifiable rewards teach a compact language model to reason about physics, or does it primarily learn to pattern-match toward correct answers? We study this question by training a 1.5B-parameter reasoning model on beam statics, a classic engineering problem, using parameter-efficient RLVR with binary correctness rewards from symbolic solvers, without teacher-generated reasoning traces. The best BeamPERL checkpoint achieves a 66.7% improvement in Pass@1 over the base model. However, the learned competence is anisotropic: the model generalizes compositionally (more loads) but fails under topological shifts (moved supports) that require the same equilibrium equations. Intermediate checkpoints yield the strongest reasoning, while continued optimization degrades robustness while maintaining reward. These findings reveal a key limitation of outcome-level alignment: reinforcement learning with exact physics rewards induces procedural solution templates rather than internalization of governing equations. The precision of the reward signal - even when analytically exact - does not by itself guarantee transferable physical reasoning. Our results suggest that verifiable rewards may need to be paired with structured reasoning scaffolding to move beyond template matching toward robust scientific reasoning.
>
---
#### [new 105] CONCUR: Benchmarking LLMs for Concurrent Code Generation
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文属于代码生成任务，旨在解决现有基准无法有效评估大语言模型生成并发代码能力的问题。作者设计了CONCUR基准，包含115个并发问题，用于测试和提升LLMs在并发代码生成方面的能力。**

- **链接: [https://arxiv.org/pdf/2603.03683](https://arxiv.org/pdf/2603.03683)**

> **作者:** Jue Huang; Tarek Mahmud; Corina Pasareanu; Guowei Yang
>
> **摘要:** Leveraging Large Language Models (LLMs) for code generation has increasingly emerged as a common practice in the domain of software engineering. Relevant benchmarks have been established to evaluate the code generation capabilities of LLMs. However, existing benchmarks focus primarily on sequential code, lacking the ability to effectively evaluate LLMs on concurrent code generation. Compared to sequential code, concurrent code exhibits greater complexity and possesses unique types of bugs, such as deadlocks and race conditions, that do not occur in sequential code. Therefore, a benchmark for evaluating sequential code generation cannot be useful for evaluating concurrent code generation with LLMs. To address this gap, we designed a benchmark CONCUR specifically aimed at evaluating the capability of LLMs to generate concurrent code. CONCUR consists of a base set of 43 concurrency problems derived from a standard concurrency textbook, together with 72 validated mutant variants, resulting in 115 total problems. The base problems serve as the semantic core of the benchmark, while the mutants expand linguistic and structural diversity. We conducted an evaluation of a range of LLMs on CONCUR, highlighting limitations of current models. Overall, our work provides a novel direction for evaluating the capability of LLMs to generate code with focus on concurrency.
>
---
## 更新

#### [replaced 001] The Geometry of Reasoning: Flowing Logics in Representation Space
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文研究大语言模型的推理机制，通过几何框架分析其表示空间中的逻辑流动，解决模型是否内化逻辑的问题。工作包括构建理论模型、设计实验验证，并揭示通用表示规律。**

- **链接: [https://arxiv.org/pdf/2510.09782](https://arxiv.org/pdf/2510.09782)**

> **作者:** Yufa Zhou; Yixiao Wang; Xunjian Yin; Shuyan Zhou; Anru R. Zhang
>
> **备注:** ICLR 2026. Code: this https URL
>
> **摘要:** We study how large language models (LLMs) ``think'' through their representation space. We propose a novel geometric framework that models an LLM's reasoning as flows -- embedding trajectories evolving where logic goes. We disentangle logical structure from semantics by employing the same natural deduction propositions with varied semantic carriers, allowing us to test whether LLMs internalize logic beyond surface form. This perspective connects reasoning with geometric quantities such as position, velocity, and curvature, enabling formal analysis in representation and concept spaces. Our theory establishes: (1) LLM reasoning corresponds to smooth flows in representation space, and (2) logical statements act as local controllers of these flows' velocities. Using learned representation proxies, we design controlled experiments to visualize and quantify reasoning flows, providing empirical validation of our theoretical framework. Our findings indicate that training solely via next-token prediction can lead LLMs to internalize logical invariants as higher-order geometry in representation space, challenging the ``stochastic parrot'' argument. Experiments across Qwen and LLaMA model families further suggest the presence of a general, possibly universal, representational law underlying machine understanding and human linguistic regularities, largely independent of specific training recipes or model architectures. Our work serves as both a conceptual foundation and practical tools for studying reasoning phenomena, offering a new lens for interpretability and formal analysis of LLMs' behavior.
>
---
#### [replaced 002] REVISION:Reflective Intent Mining and Online Reasoning Auxiliary for E-commerce Visual Search System Optimization
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于电商视觉搜索优化任务，解决用户隐式意图与系统响应不匹配的问题。通过构建REVISION框架，结合离线分析与在线推理，提升意图挖掘效率和系统适应性。**

- **链接: [https://arxiv.org/pdf/2510.22739](https://arxiv.org/pdf/2510.22739)**

> **作者:** Yiwen Tang; Qiuyu Zhao; Zenghui Sun; Jinsong Lan; Xiaoyong Zhu; Bo Zheng
>
> **摘要:** In Taobao e-commerce visual search, user behavior analysis reveals a large proportion of no-click requests, suggesting diverse and implicit user intents. These intents are expressed in various forms and are difficult to mine and discover, thereby leading to the limited adaptability and lag in platform strategies. This greatly restricts users' ability to express diverse intents and hinders the scalability of the visual search system. This mismatch between user implicit intent expression and system response defines the User-SearchSys Intent Discrepancy. To alleviate the issue, we propose a novel framework REVISION. This framework integrates offline reasoning mining with online decision-making and execution, enabling adaptive strategies to solve implicit user demands. In the offline stage, we construct a periodic pipeline to mine discrepancies from historical no-click requests. Leveraging large models, we analyze implicit intent factors and infer optimal suggestions by jointly reasoning over query and product metadata. These inferred suggestions serve as actionable insights for refining platform strategies. In the online stage, REVISION-R1-3B, trained on the curated offline data, performs holistic analysis over query images and associated historical products to generate optimization plans and adaptively schedule strategies across the search pipeline. Our framework offers a streamlined paradigm for integrating large models with traditional search systems, enabling end-to-end intelligent optimization across information aggregation and user interaction. Experimental results demonstrate that our approach improves the efficiency of implicit intent mining from large-scale search logs and significantly reduces the no-click rate.
>
---
#### [replaced 003] When Your Own Output Becomes Your Training Data: Noise-to-Meaning Loops and a Formal RSI Trigger
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出N2M-RSI模型，研究AI自我反馈导致复杂性无限增长的现象，属于理论建模任务，旨在探讨自增强系统的潜在风险与机制。**

- **链接: [https://arxiv.org/pdf/2505.02888](https://arxiv.org/pdf/2505.02888)**

> **作者:** Rintaro Ando
>
> **备注:** Withdrawn due to a critical error discovered in the mathematical derivation and proof of Theorem 2 (Unbounded Growth) and related Lemma 2 (Compression gain lower bound). This flaw invalidates the paper's main conclusion that N2M-RSI guarantees unbounded growth, requiring a fundamental revision of the theoretical framework
>
> **摘要:** We present Noise-to-Meaning Recursive Self-Improvement (N2M-RSI), a minimal formal model showing that once an AI agent feeds its own outputs back as inputs and crosses an explicit information-integration threshold, its internal complexity will grow without bound under our assumptions. The framework unifies earlier ideas on self-prompting large language models, Gödelian self-reference, and AutoML, yet remains implementation-agnostic. The model furthermore scales naturally to interacting swarms of agents, hinting at super-linear effects once communication among instances is permitted. For safety reasons, we omit system-specific implementation details and release only a brief, model-agnostic toy prototype in Appendix C.
>
---
#### [replaced 004] Knowledge Graphs are Implicit Reward Models: Path-Derived Signals Enable Compositional Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学推理任务，旨在解决模型在复杂多步推理中的局限性。通过知识图谱生成奖励信号，提升模型的组合推理能力。**

- **链接: [https://arxiv.org/pdf/2601.15160](https://arxiv.org/pdf/2601.15160)**

> **作者:** Yuval Kansal; Niraj K. Jha
>
> **摘要:** Large language models have achieved near-expert performance in structured reasoning domains like mathematics and programming, yet their ability to perform compositional multi-hop reasoning in specialized scientific fields remains limited. We propose a bottom-up learning paradigm in which models are grounded in axiomatic domain facts and compose them to solve complex, unseen tasks. To this end, we present a post-training pipeline, based on a combination of supervised fine-tuning and reinforcement learning (RL), in which knowledge graphs act as implicit reward models. By deriving novel reward signals from knowledge graph paths, we provide verifiable, scalable, and grounded supervision that encourages models to compose intermediate axioms rather than optimize only final answers during RL. We validate this approach in the medical domain, training a 14B model on short-hop reasoning paths (1-3 hops) and evaluating its zero-shot generalization to complex multi-hop queries (4-5 hops). Our experiments show that path-derived rewards act as a "compositional bridge", enabling our model to significantly outperform much larger models and frontier systems like GPT-5.2 and Gemini 3 Pro, on the most difficult reasoning tasks. Furthermore, we demonstrate the robustness of our approach to adversarial perturbations against option-shuffling stress tests. This work suggests that grounding the reasoning process in structured knowledge is a scalable and efficient path toward intelligent reasoning.
>
---
#### [replaced 005] NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在模糊输入时过早确定语义的问题。提出一种文本到状态映射框架，保留多种解释，避免信息丢失。**

- **链接: [https://arxiv.org/pdf/2601.19933](https://arxiv.org/pdf/2601.19933)**

> **作者:** Kei Saito
>
> **备注:** 25 pages, 5 figures, 7 tables. Replacement synced to repository snapshot v38. Added a direct series-hub link in the abstract for cross-paper navigation: this https URL . Series numbering policy: paper3 is intentionally skipped and never reused
>
> **摘要:** Large language models exhibit a systematic tendency toward early semantic commitment: given ambiguous input, they collapse multiple valid interpretations into a single response before sufficient context is available. This premature collapse discards information that may prove essential as dialogue evolves. We present a formal framework for text-to-state mapping (phi: T -> S) that transforms natural language into a non-collapsing state space where multiple interpretations coexist. The mapping decomposes into three stages: conflict detection, interpretation extraction, and state construction. We instantiate phi with a hybrid extraction pipeline that combines rule-based segmentation for explicit conflict markers (adversative conjunctions, hedging expressions) with LLM-based enumeration of implicit ambiguity (epistemic, lexical, structural). On a test set of 68 ambiguous sentences, the resulting states preserve interpretive multiplicity: using hybrid extraction, we obtain mean state entropy H = 1.087 bits across ambiguity categories, compared to H = 0 for collapse-based baselines that commit to a single interpretation. We additionally instantiate the rule-based conflict detector for Japanese markers (kedo, kamoshirenai, etc.) to illustrate cross-lingual portability of the conflict detection stage. This framework extends Non-Resolution Reasoning (NRR) by providing the missing algorithmic bridge between text and the NRR state space, enabling architectural collapse deferment in LLM inference. Design principles for state-to-state transformations are detailed in the Appendix, with empirical validation on 580 test cases (180 single states, 200 contradictory pairs, 200 temporal pairs), demonstrating 0% collapse for principle-satisfying operators versus up to 17.8% for violating operators.
>
---
#### [replaced 006] Succeeding at Scale: Automated Dataset Construction and Query-Side Adaptation for Multi-Tenant Search
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多租户搜索任务，解决缺乏标注数据和模型更新成本高的问题。通过自动化数据构建和仅微调查询编码器的方法，提升搜索效果。**

- **链接: [https://arxiv.org/pdf/2601.04646](https://arxiv.org/pdf/2601.04646)**

> **作者:** Prateek Jain; Shabari S Nair; Ritesh Goru; Prakhar Agarwal; Ajay Yadav; Yoga Sri Varshan Varadharajan; Constantine Caramanis
>
> **摘要:** Large-scale multi-tenant retrieval systems generate extensive query logs but lack curated relevance labels for effective domain adaptation, resulting in substantial underutilized "dark data". This challenge is compounded by the high cost of model updates, as jointly fine-tuning query and document encoders requires full corpus re-indexing, which is impractical in multi-tenant settings with thousands of isolated indices. We introduce DevRev-Search, a passage retrieval benchmark for technical customer support built via a fully automated pipeline. Candidate generation uses fusion across diverse sparse and dense retrievers, followed by an LLM-as-a-Judge for consistency filtering and relevance labeling. We further propose an Index-Preserving Adaptation strategy that fine-tunes only the query encoder, achieving strong performance gains while keeping document indices fixed. Experiments on DevRev-Search, SciFact, and FiQA-2018 show that Parameter-Efficient Fine-Tuning (PEFT) of the query encoder delivers a remarkable quality-efficiency trade-off, enabling scalable and practical enterprise search adaptation.
>
---
#### [replaced 007] Query-Level Uncertainty in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型识别自身知识边界的问题。通过提出Query-Level Uncertainty方法，提升模型在回答前判断能力，降低生成成本。**

- **链接: [https://arxiv.org/pdf/2506.09669](https://arxiv.org/pdf/2506.09669)**

> **作者:** Lihu Chen; Gerard de Melo; Fabian M. Suchanek; Gaël Varoquaux
>
> **备注:** ICLR 2026
>
> **摘要:** It is important for Large Language Models (LLMs) to be aware of the boundary of their knowledge, distinguishing queries they can confidently answer from those that lie beyond their capabilities. Such awareness enables models to perform adaptive inference, such as invoking retrieval-augmented generation (RAG), engaging in slow and deep thinking, or abstaining from answering when appropriate. These mechanisms are key to developing efficient and trustworthy AI. In this work, we propose a method to detect knowledge boundaries via Query-Level Uncertainty, which estimates if a model is capable of answering a given query before generating any tokens, thus avoiding the generation cost. To this end, we propose a novel, training-free method called Internal Confidence, which leverages self-evaluations across layers and tokens to provide a reliable signal of uncertainty. Empirical studies on both factual question answering and mathematical reasoning tasks demonstrate that our Internal Confidence outperforms several baselines in quality of confidence while being computationally cheaper. Furthermore, we demonstrate its benefits in adaptive inference settings, showing that for RAG and model cascading it reduces inference costs while preserving overall performance.
>
---
#### [replaced 008] Manipulating language models' training data to study syntactic constraint learning: the case of English passivization
- **分类: cs.CL**

- **简介: 该论文属于语言习得研究，探讨英语被动语态的例外情况。通过调整训练数据，研究语言模型如何学习动词的被动化限制，验证频率和语义的影响。**

- **链接: [https://arxiv.org/pdf/2407.04593](https://arxiv.org/pdf/2407.04593)**

> **作者:** Cara Su-Yi Leong; Tal Linzen
>
> **备注:** Journal of Memory and Language
>
> **摘要:** Grammatical rules in natural languages are often characterized by exceptions. How do language learners learn these exceptions to otherwise general patterns? Here, we study this question through the case study of English passivization. While passivization is in general quite productive, there are cases where it cannot apply (cf. the following sentence is ungrammatical: *One hour was lasted by the meeting). Using neural network language models as theories of language acquisition, we explore the sources of indirect evidence that a learner can leverage to learn whether a verb can be passivized. We first characterize English speakers' judgments of exceptions to the passive, and confirm that speakers find some verbs more passivizable than others. We then show that a neural network language model's verb passivizability judgments are largely similar to those displayed by humans, suggesting that evidence for these exceptions is available in the linguistic input. Finally, we test two hypotheses as to the source of evidence that language models use to learn these restrictions: frequency (entrenchment) and semantics (affectedness). We do so by training models on versions of the corpus that have had sentences of the types implicated by each hypothesis removed, altered, or introduced. We find support for both hypotheses: entrenchment and affectedness make independent contributions to a verb's passivizability. From a methodological point of view, this study highlights the utility of altering a language model's training data for answering questions where complete control over a learner's input is vital.
>
---
#### [replaced 009] ObfusQAte: A Proposed Framework to Evaluate LLM Robustness on Obfuscated Factual Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ObfusQAte框架，用于评估大语言模型在混淆事实问答任务中的鲁棒性，解决模型在面对复杂变形问题时的适应能力问题。**

- **链接: [https://arxiv.org/pdf/2508.07321](https://arxiv.org/pdf/2508.07321)**

> **作者:** Shubhra Ghosh; Abhilekh Borah; Aditya Kumar Guru; Kripabandhu Ghosh
>
> **备注:** LREC 2026
>
> **摘要:** The rapid proliferation of Large Language Models (LLMs) has significantly contributed to the development of equitable AI systems capable of factual question-answering (QA). However, no known study tests the LLMs' robustness when presented with obfuscated versions of questions. To systematically evaluate these limitations, we propose a novel technique, ObfusQAte, and leveraging the same, introduce ObfusQA, a comprehensive, first-of-its-kind framework with multi-tiered obfuscation levels designed to examine LLM capabilities across three distinct dimensions: (i) Named-Entity Indirection, (ii) Distractor Indirection, and (iii) Contextual Overload. By capturing these fine-grained distinctions in language, ObfusQA provides a comprehensive benchmark for evaluating LLM robustness and adaptability. Our study observes that LLMs exhibit a tendency to fail or generate hallucinated responses when confronted with these increasingly nuanced variations. To foster research in this direction, we make ObfusQAte publicly available.
>
---
#### [replaced 010] Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Agent Data Protocol（ADP），解决多源异构数据难以统一的问题，旨在提升大模型代理的微调效果。**

- **链接: [https://arxiv.org/pdf/2510.24702](https://arxiv.org/pdf/2510.24702)**

> **作者:** Yueqi Song; Ketan Ramaneti; Zaid Sheikh; Ziru Chen; Boyu Gou; Tianbao Xie; Yiheng Xu; Danyang Zhang; Apurva Gandhi; Fan Yang; Joseph Liu; Tianyue Ou; Zhihao Yuan; Frank Xu; Shuyan Zhou; Xingyao Wang; Xiang Yue; Tao Yu; Huan Sun; Yu Su; Graham Neubig
>
> **摘要:** Public research results on large-scale supervised finetuning of AI agents remain relatively rare, since the collection of agent training data presents unique challenges. In this work, we argue that the bottleneck is not a lack of underlying data sources, but that a large variety of data is fragmented across heterogeneous formats, tools, and interfaces. To this end, we introduce the agent data protocol (ADP), a light-weight representation language that serves as an "interlingua" between agent datasets in diverse formats and unified agent training pipelines downstream. The design of ADP is expressive enough to capture a large variety of tasks, including API/tool use, browsing, coding, software engineering, and general agentic workflows, while remaining simple to parse and train on without engineering at a per-dataset level. In experiments, we unified a broad collection of 13 existing agent training datasets into ADP format, and converted the standardized ADP data into training-ready formats for multiple agent frameworks. We performed SFT on these data, and demonstrated an average performance gain of ~20% over corresponding base models, and delivers state-of-the-art or near-SOTA performance on standard coding, browsing, tool use, and research benchmarks, without domain-specific tuning. All code and data are released publicly, in the hope that ADP could help lower the barrier to standardized, scalable, and reproducible agent training.
>
---
#### [replaced 011] RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference
- **分类: cs.CL**

- **简介: 该论文属于模型推理优化任务，旨在解决大语言模型推理效率低的问题。通过引入RAEE框架，利用检索增强的早期退出机制，在加速推理的同时提升性能。**

- **链接: [https://arxiv.org/pdf/2405.15198](https://arxiv.org/pdf/2405.15198)**

> **作者:** Lianming Huang; Shangyu Wu; Yufei Cui; Ying Xiong; Haibo Hu; Xue Liu; Tei-Wei Kuo; Nan Guan; Chun Jason Xue
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Deploying large language model inference remains challenging due to their high computational overhead. Early exit optimizes model inference by adaptively reducing the number of inference layers. Current methods typically train internal classifiers or use heuristic methods to determine the exit layer. However, those methods either introduce significant training overheads or lead to performance degradation. To address these limitations, this paper proposes RAEE, a robust Retrieval-Augmented Early Exit framework that not only enables early exit but also enhances model performance through corrective exit information at intermediate layers. This paper first demonstrates that the early exit problem can be effectively modeled as a distribution prediction problem, in which the distribution can be further approximated through the exit information of similar data. Subsequently, this paper introduces the process of collecting exit information of correct predictions and the steps to construct the retrieval database. Finally, leveraging the pre-constructed retrieval database, RAEE utilizes the exit information from retrieved similar data to guide the backbone model's exit. Experimental results demonstrate that RAEE can not only accelerate inference while achieving robust zero-shot performance across eight downstream tasks.
>
---
#### [replaced 012] What Triggers my Model? Contrastive Explanations Inform Gender Choices by Translation Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的可解释性研究，旨在探究神经机器翻译模型在性别选择上的触发因素，分析模型决策与人类感知的关联，以减少性别偏见。**

- **链接: [https://arxiv.org/pdf/2512.08440](https://arxiv.org/pdf/2512.08440)**

> **作者:** Janiça Hackenbuchner; Arda Tezcan; Joke Daems
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Interpretability can be implemented to understand decisions taken by (black box) models, such as neural machine translation (NMT) or large language models (LLMs). Yet, research in this area has been limited in relation to a manifested problem in these models: gender bias. In this work, we aim to move away from simply measuring bias to exploring its origins. Working with gender-ambiguous natural source data, this exploratory study examines which context, in the form of input tokens in the source sentence (EN), influences (or triggers) the NMT model's choice of a certain gender inflection in the target languages (DE/ES). To analyse this, we compute saliency attribution based on contrastive translations. We first address the challenge of the lack of a scoring threshold and specifically examine different attribution levels of source words on the model's gender decisions in the translation. We compare salient source words with human perceptions of gender and demonstrate a noticeable overlap between human perceptions and model attribution. Additionally, we provide a linguistic analysis of salient words. Our work showcases the relevance of understanding model translation decisions in terms of gender, how this compares to human decisions and that this information should be leveraged to mitigate gender bias.
>
---
#### [replaced 013] Catch Me If You Can Describe Me: Open-Vocabulary Camouflaged Instance Segmentation with Diffusion
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于开放词汇伪装实例分割任务，旨在解决伪装目标与背景难以区分的问题。通过扩散模型学习多尺度文本视觉特征，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2312.17505](https://arxiv.org/pdf/2312.17505)**

> **作者:** Tuan-Anh Vu; Duc Thanh Nguyen; Qing Guo; Nhat Chung; Binh-Son Hua; Ivor W. Tsang; Sai-Kit Yeung
>
> **备注:** Accepted to IJCV 2026
>
> **摘要:** Text-to-image diffusion techniques have shown exceptional capabilities in producing high-quality, dense visual predictions from open-vocabulary text. This indicates a strong correlation between visual and textual domains in open concepts and that diffusion-based text-to-image models can capture rich and diverse information for computer vision tasks. However, we found that those advantages do not hold for learning of features of camouflaged individuals because of the significant blending between their visual boundaries and their surroundings. In this paper, while leveraging the benefits of diffusion-based techniques and text-image models in open-vocabulary settings, we aim to address a challenging problem in computer vision: open-vocabulary camouflaged instance segmentation (OVCIS). Specifically, we propose a method built upon state-of-the-art diffusion empowered by open-vocabulary to learn multi-scale textual-visual features for camouflaged object representation learning. Such cross-domain representations are desirable in segmenting camouflaged objects where visual cues subtly distinguish the objects from the background, and in segmenting novel object classes which are not seen in training. To enable such powerful representations, we devise complementary modules to effectively fuse cross-domain features, and to engage relevant features towards respective foreground objects. We validate and compare our method with existing ones on several benchmark datasets of camouflaged and generic open-vocabulary instance segmentation. The experimental results confirm the advances of our method over existing ones. We believe that our proposed method would open a new avenue for handling camouflages such as computer vision-based surveillance systems, wildlife monitoring, and military reconnaissance.
>
---
#### [replaced 014] When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间问答任务，旨在解决LLMs不承认不确定、错误回答的问题。通过结合思维链监督与强化学习，提升模型的拒绝回答能力。**

- **链接: [https://arxiv.org/pdf/2602.04755](https://arxiv.org/pdf/2602.04755)**

> **作者:** Xinyu Zhou; Chang Jin; Carsten Eickhoff; Zhijiang Guo; Seyed Ali Bahrainian
>
> **备注:** Accepted to ICLR2026
>
> **摘要:** Large language models (LLMs) rarely admit uncertainty, often producing fluent but misleading answers, rather than abstaining (i.e., refusing to answer). This weakness is even evident in temporal question answering, where models frequently ignore time-sensitive evidence and conflate facts across different time-periods. In this paper, we present the first empirical study of training LLMs with an abstention ability while reasoning about temporal QA. Existing approaches such as calibration might be unreliable in capturing uncertainty in complex reasoning. We instead frame abstention as a teachable skill and introduce a pipeline that couples Chain-of-Thought (CoT) supervision with Reinforcement Learning (RL) guided by abstention-aware rewards. Our goal is to systematically analyze how different information types and training techniques affect temporal reasoning with abstention behavior in LLMs. Through extensive experiments studying various methods, we find that RL yields strong empirical gains on reasoning: a model initialized by Qwen2.5-1.5B-Instruct surpasses GPT-4o by $3.46\%$ and $5.80\%$ in Exact Match on TimeQA-Easy and Hard, respectively. Moreover, it improves the True Positive rate on unanswerable questions by $20\%$ over a pure supervised fine-tuned (SFT) variant. Beyond performance, our analysis shows that SFT induces overconfidence and harms reliability, while RL improves prediction accuracy but exhibits similar risks. Finally, by comparing implicit reasoning cues (e.g., original context, temporal sub-context, knowledge graphs) with explicit CoT supervision, we find that implicit information provides limited benefit for reasoning with abstention. Our study provides new insights into how abstention and reasoning can be jointly optimized, providing a foundation for building more reliable LLMs.
>
---
#### [replaced 015] Multimodal Large Language Models for Low-Resource Languages: A Case Study for Basque
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态语言模型任务，旨在解决低资源语言（如巴斯克语）中MLLM性能不足的问题。研究构建了专用数据集，测试不同模型和数据组合，发现少量多模态数据即可取得良好效果。**

- **链接: [https://arxiv.org/pdf/2511.09396](https://arxiv.org/pdf/2511.09396)**

> **作者:** Lukas Arana; Julen Etxaniz; Ander Salaberria; Gorka Azkune
>
> **摘要:** Current Multimodal Large Language Models exhibit very strong performance for several demanding tasks. While commercial MLLMs deliver acceptable performance in low-resource languages, comparable results remain unattained within the open science community. In this paper, we aim to develop a strong MLLM for a low-resource language, namely Basque. For that purpose, we develop our own training and evaluation image-text datasets. Using two different Large Language Models as backbones, the Llama-3.1-Instruct model and a Basque-adapted variant called Latxa, we explore several data mixtures for training. We show that: i) low ratios of Basque multimodal data (around 20%) are already enough to obtain solid results on Basque benchmarks, and ii) contrary to expected, a Basque instructed backbone LLM is not required to obtain a strong MLLM in Basque. Our results pave the way to develop MLLMs for other low-resource languages by openly releasing our resources.
>
---
#### [replaced 016] GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出GraphMERT，用于从非结构化文本中高效提取可靠知识图谱，解决神经符号AI可扩展性和可信度问题。**

- **链接: [https://arxiv.org/pdf/2510.09580](https://arxiv.org/pdf/2510.09580)**

> **作者:** Margarita Belova; Jiaxin Xiao; Shikhar Tuli; Niraj K. Jha
>
> **备注:** Camera-ready version. Published in Transactions on Machine Learning Research (TMLR), 2026. Reviewed on OpenReview: this https URL
>
> **摘要:** Researchers have pursued neurosymbolic artificial intelligence (AI) applications for nearly three decades. A marriage of the neural and symbolic components can lead to rapid advancements in AI. Yet, the field has not realized this promise since most neurosymbolic AI frameworks fail to scale. In addition, the implicit representations and approximate reasoning of purely neural approaches limit interpretability and trust. Knowledge graphs (KGs), a gold-standard representation of explicit semantic knowledge, can address the symbolic side of the problem. However, automatically deriving reliable KGs from text corpora remains an open problem. We address these challenges by introducing GraphMERT, a tiny graphical encoder-only model that distills high-quality KGs from unstructured text corpora and its own internal representations. GraphMERT and its equivalent KG form a modular neurosymbolic stack: neural learning of abstractions; symbolic KGs for verifiable reasoning. GraphMERT + KG is the first efficient and scalable neurosymbolic model to achieve state-of-the-art benchmark accuracy along with superior symbolic representations relative to baselines. Concretely, we target reliable domain-specific KGs that are both (1) factual (with provenance) and (2) valid (ontology-consistent relations with domain-appropriate semantics). When a large language model (LLM), e.g., Qwen3-32B, generates domain-specific KGs, it falls short on reliability due to prompt sensitivity, shallow domain expertise, and hallucinated relations. On text obtained from PubMed papers on diabetes, our 80M-parameter GraphMERT yields a KG with a 69.8% FActScore; a 32B-parameter baseline LLM yields a KG that achieves only 40.2% FActScore. The GraphMERT KG also attains a higher ValidityScore of 68.8%, versus 43.0% for the LLM baseline.
>
---
#### [replaced 017] Detecting AI-Generated Essays in Writing Assessment: Responsible Use and Generalizability Across LLMs
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决AI生成作文的识别问题。研究分析了检测器在不同LLM间的泛化能力，为实际应用提供指导。**

- **链接: [https://arxiv.org/pdf/2603.02353](https://arxiv.org/pdf/2603.02353)**

> **作者:** Jiangang Hao
>
> **备注:** 21 pages, 2 figures
>
> **摘要:** Writing is a foundational literacy skill that underpins effective communication, fosters critical thinking, facilitates learning across disciplines, and enables individuals to organize and articulate complex ideas. Consequently, writing assessment plays a vital role in evaluating language proficiency, communicative effectiveness, and analytical reasoning. The rapid advancement of large language models (LLMs) has made it increasingly easy to generate coherent, high-quality essays, raising significant concerns about the authenticity of student-submitted work. This chapter first provides an overview of the current landscape of detectors for AI-generated and AI-assisted essays, along with guidelines for their responsible use. It then presents empirical analyses to evaluate how well detectors trained on essays from one LLM generalize to identifying essays produced by other LLMs, based on essays generated in response to public GRE writing prompts. These findings provide guidance for developing and retraining detectors for practical applications.
>
---
#### [replaced 018] R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.SC**

- **简介: 该论文属于代码推理任务，旨在提升大语言模型的代码解释能力。通过监督和分阶段强化学习，解决任务多样性与样本稀缺问题，显著提高模型准确性。**

- **链接: [https://arxiv.org/pdf/2505.21668](https://arxiv.org/pdf/2505.21668)**

> **作者:** Yongchao Chen; Yueying Liu; Junwei Zhou; Yilun Hao; Jingquan Wang; Yang Zhang; Na Li; Chuchu Fan
>
> **备注:** 29 pages
>
> **摘要:** Practical guidance on training Large Language Models (LLMs) to leverage Code Interpreter across diverse tasks remains lacking. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. Unlike prior RL + tool-use efforts focused on narrow domains such as math or retrieval, we curate 144 diverse reasoning and planning tasks and show that training a general-purpose Code Interpreter across them presents significant challenges due to task heterogeneity and scarcity of effective samples. To address this, we introduce a multi-stage curriculum learning approach that partitions training samples by measured improvement potential. The RL training prioritizes samples with higher potential and gradually shifts to lower-potential ones, increasing the average RL gains from merely +3.4% to +9.3% across Qwen-2.5 models (3/7/14B). Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.1% to 72.4%, outperforming text-only GPT-4o (58.6%) and GPT-4o with Code Interpreter (70.9%). Notably, R1-CI-14B also exhibits emergent self-checking behavior through code generation. Datasets, Codes, and Models are available at this https URL and this https URL.
>
---
#### [replaced 019] Prompt Sensitivity and Answer Consistency of Small Open-Source Large Language Models on Clinical Question Answering: Implications for Low-Resource Healthcare Deployment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床问答任务，研究小规模开源模型在不同提示下的表现，解决其可靠性与准确性问题，通过实验评估模型一致性、准确性和指令遵循情况。**

- **链接: [https://arxiv.org/pdf/2603.00917](https://arxiv.org/pdf/2603.00917)**

> **作者:** Shravani Hariprasad
>
> **备注:** 30 pages, 7 figures, 2 tables
>
> **摘要:** Small open-source language models are gaining attention for healthcare applications in low-resource settings where cloud infrastructure and GPU hardware may be unavailable. However, their reliability under different prompt phrasings remains poorly understood. We evaluate five open-source models (Gemma 2 2B, Phi-3 Mini 3.8B, Llama 3.2 3B, Mistral 7B, and Meditron-7B, a domain-pretrained model without instruction tuning) across three clinical question answering datasets (MedQA, MedMCQA, and PubMedQA) using five prompt styles: original, formal, simplified, roleplay, and direct. Model behavior is evaluated using consistency scores, accuracy, and instruction-following failure rates. All experiments were conducted locally on consumer CPU hardware without fine-tuning. Consistency and accuracy were largely independent across models. Gemma 2 achieved the highest consistency (0.845-0.888) but the lowest accuracy (33.0-43.5%), while Llama 3.2 showed moderate consistency (0.774-0.807) alongside the highest accuracy (49.0-65.0%). Roleplay prompts consistently reduced accuracy across all models, with Phi-3 Mini dropping 21.5 percentage points on MedQA. Meditron-7B exhibited near-complete instruction-following failure on PubMedQA (99.0% UNKNOWN rate), indicating that domain pretraining alone is insufficient for structured clinical QA. These findings show that high consistency does not imply correctness: models can be reliably wrong, a dangerous failure mode in clinical AI. Llama 3.2 demonstrated the strongest balance of accuracy and reliability for low-resource deployment. Safe clinical AI requires joint evaluation of consistency, accuracy, and instruction adherence.
>
---
#### [replaced 020] Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决偏好模型因数据偏差导致的误校准问题。通过分析五种特征偏差，提出基于对比数据增强的去偏方法，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2506.05339](https://arxiv.org/pdf/2506.05339)**

> **作者:** Anirudh Bharadwaj; Chaitanya Malaviya; Nitish Joshi; Mark Yatskar
>
> **备注:** Published at ICLR 2026; Code and data available at this https URL
>
> **摘要:** Language models serve as proxies for human preference judgements in alignment and evaluation, yet they exhibit systematic miscalibration, prioritizing superficial patterns over substantive qualities. This bias manifests as overreliance on features like length, structure, and style, leading to issues like reward hacking and unreliable evaluations. However, the connection between training data artifacts and the miscalibrated preferences exhibited by models remains poorly understood. In this work, we systematically investigate the relationship between training data biases and preference model miscalibration across five idiosyncratic features of language model generations: length, structure, jargon, sycophancy and vagueness. Using controlled counterfactual pairs, we first quantify the extent to which preference models favor responses with artificially magnified biases (skew), finding this preference occurs in $>60\%$ of instances, and model preferences show high miscalibration ($\approx 40\%$) compared to human preferences. Notably, bias features only show mild negative correlations to human preference labels (mean $r_{\mathrm{human}} = -0.12$) but show moderately strong positive correlations with labels from a strong reward model (mean $r_{\mathrm{model}} = +0.36$), suggesting that models may overrely on spurious cues. To mitigate these issues, we propose a simple post-training method based on counterfactual data augmentation (CDA) using synthesized contrastive examples. Fine-tuning models with CDA reduces average miscalibration from $39.4\%$ to $32.5\%$ and average absolute skew difference from $20.5\%$ to $10.0\%$, while maintaining overall RewardBench performance, indicating that targeted debiasing can strengthen the reliability of preference models within standard alignment pipelines.
>
---
#### [replaced 021] From Variance to Invariance: Qualitative Content Analysis for Narrative Graph Annotation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的叙事图注释任务，旨在解决叙事结构标注的准确性问题。通过引入质性内容分析框架，优化注释质量并减少变异。**

- **链接: [https://arxiv.org/pdf/2603.01930](https://arxiv.org/pdf/2603.01930)**

> **作者:** Junbo Huang; Max Weinig; Ulrich Fritsche; Ricardo Usbeck
>
> **备注:** LREC 2026 Accepted Paper
>
> **摘要:** Narratives in news discourse play a critical role in shaping public understanding of economic events, such as inflation. Annotating and evaluating these narratives in a structured manner remains a key challenge for Natural Language Processing (NLP). In this work, we introduce a narrative graph annotation framework that integrates principles from qualitative content analysis (QCA) to prioritize annotation quality by reducing annotation errors. We present a dataset of inflation narratives annotated as directed acyclic graphs (DAGs), where nodes represent events and edges encode causal relations. To evaluate annotation quality, we employed a $6\times3$ factorial experimental design to examine the effects of narrative representation (six levels) and distance metric type (three levels) on inter-annotator agreement (Krippendorrf's $\alpha$), capturing the presence of human label variation (HLV) in narrative interpretations. Our analysis shows that (1) lenient metrics (overlap-based distance) overestimate reliability, and (2) locally-constrained representations (e.g., one-hop neighbors) reduce annotation variability. Our annotation and implementation of graph-based Krippendorrf's $\alpha$ are open-sourced. The annotation framework and evaluation results provide practical guidance for NLP research on graph-based narrative annotation under HLV.
>
---
#### [replaced 022] Dripper: Token-Efficient Main HTML Extraction with a Lightweight LM
- **分类: cs.CL**

- **简介: 该论文提出Dripper框架，解决网页主内容提取问题。通过约束序列标注任务，提升效率与准确性，优于传统方法和大模型。**

- **链接: [https://arxiv.org/pdf/2511.23119](https://arxiv.org/pdf/2511.23119)**

> **作者:** Mengjie Liu; Jiahui Peng; Wenchang Ning; Pei Chu; Jiantao Qiu; Ren Ma; He Zhu; Rui Min; Lindong Lu; Linfeng Hou; Kaiwen Liu; Yuan Qu; Zhenxiang Li; Chao Xu; Zhongying Tu; Wentao Zhang; Conghui He
>
> **摘要:** High-quality main content extraction from web pages is a critical prerequisite for constructing large-scale training corpora. While traditional heuristic extractors are efficient, they lack the semantic reasoning required to handle the structural heterogeneity of the modern web. Conversely, well-pretrained generative Large Language Models (LLMs) offer superior document comprehension but are prohibited by excessive computational costs, limited context windows, and hallucination risks when applied at web scale. We present \textbf{Dripper}, a lightweight framework that resolves these bottlenecks through four contributions: (1) We reformulate extraction as a \textbf{constrained sequence labeling} task using SLMs (Small Language Models). This paradigm eliminates generative hallucinations and achieves exceptional efficiency, reaching a throughput of 3.08 pages per second on a single A100 GPU. (2) We construct \textbf{WebMainBench}, a rigorous benchmark of 7,809 human-annotated pages covering 5,434 unique domains and multiple languages. Evaluations show our Dripper-0.6B model \textbf{outperforms} heuristics like Trafilatura and rivals massive models like DeepSeek-V3.2(685B), GPT-5 and Gemini-2.5-Pro, offering an optimal efficiency-accuracy trade-off. (3) We demonstrate infrastructural value by \textbf{pre-training a 1B model} on a Dripper-curated corpus (63B tokens). This model significantly outperforms baselines in downstream tasks, proving the critical role of extraction quality and the effectiveness of our framework. (4) We \textbf{open-source} the Dripper-0.6B weights and codebase to facilitate the construction of high-quality datasets.
>
---
#### [replaced 023] Composition-Grounded Data Synthesis for Visual Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉推理任务，旨在解决多模态大模型在缺乏标注数据的领域中推理能力不足的问题。通过COGS框架，从少量种子问题生成大量合成数据，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2510.15040](https://arxiv.org/pdf/2510.15040)**

> **作者:** Xinyi Gu; Jiayuan Mao; Zhang-Wei Hong; Zhuoran Yu; Pengyuan Li; Dhiraj Joshi; Rogerio Feris; Zexue He
>
> **备注:** ICLR2026 camera-ready version. Project page: this https URL
>
> **摘要:** Pretrained multi-modal large language models (MLLMs) demonstrate strong performance on diverse multimodal tasks, but remain limited in reasoning capabilities for domains where annotations are difficult to collect. In this work, we focus on artificial image domains such as charts, rendered documents, and webpages, which are abundant in practice yet lack large-scale human annotated reasoning datasets. We introduce COGS (COmposition-Grounded data Synthesis), a data-efficient framework for equipping MLLMs with advanced reasoning abilities from a small set of seed questions. The key idea is to decompose each seed question into primitive perception and reasoning factors, which can then be systematically recomposed with new images to generate large collections of synthetic question-answer pairs. Each generated question is paired with subquestions and intermediate answers, enabling reinforcement learning with factor-level process rewards. Experiments on chart reasoning show that COGS substantially improves performance on unseen questions, with the largest gains on reasoning-heavy and compositional questions. Moreover, training with a factor-level mixture of different seed data yields better transfer across multiple datasets, suggesting that COGS induces generalizable capabilities rather than dataset-specific overfitting. We further demonstrate that the framework extends beyond charts to other domains such as webpages.
>
---
#### [replaced 024] Non-Collaborative User Simulators for Tool Agents
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决工具代理在面对非合作用户时表现不佳的问题。提出一种模拟非合作行为的用户模拟器，用于测试和提升代理的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.23124](https://arxiv.org/pdf/2509.23124)**

> **作者:** Jeonghoon Shim; Woojung Song; Cheyon Jin; Seungwon KooK; Yohan Jo
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Tool agents interact with users through multi-turn dialogues to accomplish various tasks. Recent studies have adopted user simulation methods to develop these agents in multi-turn settings. However, existing user simulators tend to be agent-friendly, exhibiting only cooperative behaviors, failing to train and test agents against non-collaborative users in the real world. We propose a novel user simulator architecture that simulates four categories of non-collaborative behaviors: requesting unavailable services, digressing into tangential conversations, expressing impatience, and providing incomplete utterances. Our user simulator can simulate challenging and natural non-collaborative behaviors while reliably delivering all intents and information necessary to accomplish the task. Our experiments on MultiWOZ and {\tau}-bench reveal significant performance degradation in state-of-the-art tool agents when encountering non-collaborative users, as well as agent weaknesses under each non-collaborative condition such as escalated hallucinations and dialogue breakdowns. Our findings point to the need for methods that can improve agent robustness to the wide range of user behaviors encountered in deployment. We release the extensible simulation framework to help the community develop and stress-test tool agents under realistic conditions within their own service domains. Our code is available at this https URL.
>
---
#### [replaced 025] Generating Fine Details of Entity Interactions
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决对象交互描述不足的问题。通过构建数据集和改进生成方法，提升图像中对象互动的细节质量。**

- **链接: [https://arxiv.org/pdf/2504.08714](https://arxiv.org/pdf/2504.08714)**

> **作者:** Xinyi Gu; Jiayuan Mao
>
> **备注:** EMNLP 2025. Project Page: this https URL
>
> **摘要:** Recent text-to-image models excel at generating high-quality object-centric images from instructions. However, images should also encapsulate rich interactions between objects, where existing models often fall short, likely due to limited training data and benchmarks for rare interactions. This paper explores a novel application of Multimodal Large Language Models (MLLMs) to benchmark and enhance the generation of interaction-rich images. We introduce \data, an interaction-focused dataset with 1000 LLM-generated fine-grained prompts for image generation covering (1) functional and action-based interactions, (2) multi-subject interactions, and (3) compositional spatial relationships. To address interaction-rich generation challenges, we propose a decomposition-augmented refinement procedure. Our approach, \model, leverages LLMs to decompose interactions into finer-grained concepts, uses an MLLM to critique generated images, and applies targeted refinements with a partial diffusion denoising process. Automatic and human evaluations show significantly improved image quality, demonstrating the potential of enhanced inference strategies.
>
---
#### [replaced 026] To Think or Not To Think, That is The Question for Large Reasoning Models in Theory of Mind Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大模型在心智理论任务中的表现，探讨其推理能力是否能有效迁移。工作包括对比分析九个模型，发现推理模型在社交认知任务中表现不稳定，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2602.10625](https://arxiv.org/pdf/2602.10625)**

> **作者:** Nanxu Gong; Haotian Li; Sixun Dong; Jianxun Lian; Yanjie Fu; Xing Xie
>
> **摘要:** Theory of Mind (ToM) assesses whether models can infer hidden mental states such as beliefs, desires, and intentions, which is essential for natural social interaction. Although recent progress in Large Reasoning Models (LRMs) has boosted step-by-step inference in mathematics and coding, it is still underexplored whether this benefit transfers to socio-cognitive skills. We present a systematic study of nine advanced Large Language Models (LLMs), comparing reasoning models with non-reasoning models on three representative ToM benchmarks. The results show that reasoning models do not consistently outperform non-reasoning models and sometimes perform worse. A fine-grained analysis reveals three insights. First, slow thinking collapses: accuracy significantly drops as responses grow longer, and larger reasoning budgets hurt performance. Second, moderate and adaptive reasoning benefits performance: constraining reasoning length mitigates failure, while distinct success patterns demonstrate the necessity of dynamic adaptation. Third, option matching shortcut: when multiple choice options are removed, reasoning models improve markedly, indicating reliance on option matching rather than genuine deduction. We also design two intervention approaches: Slow-to-Fast (S2F) adaptive reasoning and Think-to-Match (T2M) shortcut prevention to further verify and mitigate the problems. With all results, our study highlights the advancement of LRMs in formal reasoning (e.g., math, code) cannot be fully transferred to ToM, a typical task in social reasoning. We conclude that achieving robust ToM requires developing unique capabilities beyond existing reasoning methods.
>
---
#### [replaced 027] Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档级事件参数抽取任务，解决零样本场景下数据稀缺与生成数据不可靠的问题。提出多智能体协作框架，通过生成与评估迭代优化，提升数据质量和抽取性能。**

- **链接: [https://arxiv.org/pdf/2603.02909](https://arxiv.org/pdf/2603.02909)**

> **作者:** Guangjun Zhang; Hu Zhang; Yazhou Han; Yue Fan; Yuhang Shao; Ru Li; Hongye Tan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Document-level event argument extraction (DEAE) is essential for knowledge acquisition, aiming to extract participants of events from documents . In the zero-shot setting, existing methods employ LLMs to generate synthetic data to address the challenge posed by the scarcity of annotated data. However, relying solely on Event-type-only prompts makes it difficult for the generated content to accurately capture the contextual and structural relationships of unseen events. Moreover, ensuring the reliability and usability of synthetic data remains a significant challenge due to the absence of quality evaluation mechanisms. To this end, we introduce a multi-agent collaboration framework for zero-shot document-level event argument extraction (ZS-DEAE), which simulates the human collaborative cognitive process of "Propose-Evaluate-Revise." Specifically, the framework comprises a generation agent and an evaluation agent. The generation agent synthesizes data for unseen events by leveraging knowledge from seen events, while the evaluation agent extracts arguments from the synthetic data and assesses their semantic consistency with the context. The evaluation results are subsequently converted into reward signals, with event structure constraints incorporated into the reward design to enable iterative optimization of both agents via reinforcement this http URL three zero-shot scenarios constructed from the RAMS and WikiEvents datasets, our method achieves improvements both in data generation quality and argument extraction performance, while the generated data also effectively enhances the zero-shot performance of other DEAE models.
>
---
#### [replaced 028] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文档理解任务，旨在降低视觉语言模型的计算成本。通过轻量级令牌剪枝框架，过滤非信息背景区域，提升效率同时保持准确率。**

- **链接: [https://arxiv.org/pdf/2509.06415](https://arxiv.org/pdf/2509.06415)**

> **作者:** Jaemin Son; Sujin Choi; Inyong Yun
>
> **备注:** Accepted to ICLR 2026 Workshop MM Intelligence
>
> **摘要:** Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a max-pooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy.
>
---
#### [replaced 029] Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决 Czech 语言的基于方面的情感分析问题。通过构建包含观点词的语料库，并利用大语言模型进行实验，提出跨语言对齐方法以提升低资源语言的性能。**

- **链接: [https://arxiv.org/pdf/2602.22730](https://arxiv.org/pdf/2602.22730)**

> **作者:** Jakub Šmíd; Pavel Přibáň; Pavel Král
>
> **备注:** Accepted for the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** This paper introduces a novel Czech dataset in the restaurant domain for aspect-based sentiment analysis (ABSA), enriched with annotations of opinion terms. The dataset supports three distinct ABSA tasks involving opinion terms, accommodating varying levels of complexity. Leveraging this dataset, we conduct extensive experiments using modern Transformer-based models, including large language models (LLMs), in monolingual, cross-lingual, and multilingual settings. To address cross-lingual challenges, we propose a translation and label alignment methodology leveraging LLMs, which yields consistent improvements. Our results highlight the strengths and limitations of state-of-the-art models, especially when handling the linguistic intricacies of low-resource languages like Czech. A detailed error analysis reveals key challenges, including the detection of subtle opinion terms and nuanced sentiment expressions. The dataset establishes a new benchmark for Czech ABSA, and our proposed translation-alignment approach offers a scalable solution for adapting ABSA resources to other low-resource languages.
>
---
#### [replaced 030] Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?
- **分类: cs.CL**

- **简介: 该论文属于数学问题生成任务，旨在解决高质量数学题稀缺的问题。通过代码代理自动演化出更复杂的新题目，验证其可解性和难度提升。**

- **链接: [https://arxiv.org/pdf/2603.03202](https://arxiv.org/pdf/2603.03202)**

> **作者:** Dadi Guo; Yuejin Xie; Qingyu Liu; Jiayu Liu; Zhiyuan Fan; Qihan Ren; Shuai Shao; Tianyi Zhou; Dongrui Liu; Yi R. Fung
>
> **备注:** 32 pages, 4 figures
>
> **摘要:** As large language models (LLMs) advance their mathematical capabilities toward the IMO level, the scarcity of challenging, high-quality problems for training and evaluation has become a significant bottleneck. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the originals. This work provides empirical evidence that code-driven agents can serve as a viable mechanism for synthesizing high-difficulty mathematical reasoning problems within scalable computational environments. Our data is available at this https URL.
>
---
#### [replaced 031] WebDS: An End-to-End Benchmark for Web-based Data Science
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出WebDS，一个端到端的基于网络的数据科学基准，解决现有基准与实际数据科学任务脱节的问题。通过多样化的网页任务，评估LLM在数据获取、分析等全流程中的表现。**

- **链接: [https://arxiv.org/pdf/2508.01222](https://arxiv.org/pdf/2508.01222)**

> **作者:** Ethan Hsu; Hong Meng Yam; Ines Bouissou; Aaron Murali John; Raj Thota; Josh Koe; Vivek Sarath Putta; G K Dharesan; Alexander Spangher; Shikhar Murty; Tenghao Huang; Christopher D. Manning
>
> **备注:** 14 pages, ICLR 2026
>
> **摘要:** Many real-world data science tasks involve complex web-based interactions: finding appropriate data available on the internet, synthesizing multimodal data from different locations, and producing summarized analyses. Existing web benchmarks often focus on simplistic interactions and often do not require diverse tool-using capabilities. Conversely, traditional data science benchmarks typically concentrate on static, highly structured datasets and do not assess end-to-end workflows that encompass data acquisition, cleaning, analysis, and insight generation. In response, we introduce WebDS, the first end-to-end web-based data science benchmark. It comprises 870 web-based data science tasks across 29 diverse websites from structured government data portals to unstructured news media, challenging agents to perform complex, multi-step, tool-based operations, across heterogeneous data formats, to better reflect the realities of modern data analytics. Evaluations of current SOTA LLM agents indicate significant performance gaps in accomplishing these tasks. For instance, Browser Use, which accomplishes $80\%$ of tasks on WebVoyager, completes only 15% of tasks in WebDS, which our analysis suggests is due to new failure modes, such as poor information grounding, repetitive behavior and shortcut-taking that agents performing WebDS's tasks display. By contrast, humans achieve around 90% accuracy, highlighting a substantial gap between current agents and human performance. By providing a more robust and realistic testing ground, WebDS sets the stage for significant advances in the development of practically useful LLM-based data science.
>
---
#### [replaced 032] LaTeX Compilation: Challenges in the Era of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在科学写作中因TeX局限性带来的效率问题。通过分析TeX缺陷，提出Mogan STEM编辑器作为替代方案，提升编译效率与工具生态。**

- **链接: [https://arxiv.org/pdf/2603.02873](https://arxiv.org/pdf/2603.02873)**

> **作者:** Tianyou Liu; Ziqiang Li; Xurui Liu; Yansong Li
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. What's more, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format.
>
---
#### [replaced 033] Leveraging Large Language Models for Semantic Query Processing in a Scholarly Knowledge Graph
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于知识图谱与自然语言处理任务，旨在解决学术知识检索效率低的问题。通过结合大语言模型与知识图谱，提升查询准确性和效率。**

- **链接: [https://arxiv.org/pdf/2405.15374](https://arxiv.org/pdf/2405.15374)**

> **作者:** Runsong Jia; Bowen Zhang; Sergio J. Rodríguez Méndez; Pouya G. Omran
>
> **备注:** for the associated repository, see this http URL
>
> **摘要:** The proposed research aims to develop an innovative semantic query processing system that enables users to obtain comprehensive information about research works produced by Computer Science (CS) researchers at the Australian National University (ANU). The system integrates Large Language Models (LLMs) with the ANU Scholarly Knowledge Graph (ASKG), a structured repository of all research-related artifacts produced at ANU in the CS field. Each artifact and its parts are represented as textual nodes stored in a Knowledge Graph (KG). To address the limitations of traditional scholarly KG construction and utilization methods, which often fail to capture fine-grained details, we propose a novel framework that integrates the Deep Document Model (DDM) for comprehensive document representation and the KG-enhanced Query Processing (KGQP) for optimized complex query handling. DDM enables a fine-grained representation of the hierarchical structure and semantic relationships within academic papers, while KGQP leverages the KG structure to improve query accuracy and efficiency with LLMs. By combining the ASKG with LLMs, our approach enhances knowledge utilization and natural language understanding capabilities. The proposed system employs an automatic LLM-SPARQL fusion to retrieve relevant facts and textual nodes from the ASKG. Initial experiments demonstrate that our framework is superior to baseline methods in terms of accuracy retrieval and query efficiency. We showcase the practical application of our framework in academic research scenarios, highlighting its potential to revolutionize scholarly knowledge management and discovery. This work empowers researchers to acquire and utilize knowledge from documents more effectively and provides a foundation for developing precise and reliable interactions with LLMs.
>
---
#### [replaced 034] Rewards as Labels: Revisiting RLVR from a Classification Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中梯度分配不均的问题。通过将奖励视为分类标签，提出REAL框架，提升策略优化效果。**

- **链接: [https://arxiv.org/pdf/2602.05630](https://arxiv.org/pdf/2602.05630)**

> **作者:** Zepeng Zhai; Meilin Chen; Jiaxuan Zhao; Junlang Qian; Lei Shen; Yuan Lu
>
> **备注:** 12 pages, 5 figures, 4 tables
>
> **摘要:** Reinforcement Learning with Verifiable Rewards has recently advanced the capabilities of Large Language Models in complex reasoning tasks by providing explicit rule-based supervision. Among RLVR methods, GRPO and its variants have achieved strong empirical performance. Despite their success, we identify that they suffer from Gradient Misassignment in Positives and Gradient Domination in Negatives, which lead to inefficient and suboptimal policy updates. To address these issues, we propose Rewards as Labels (REAL), a novel framework that revisits verifiable rewards as categorical labels rather than scalar weights, thereby reformulating policy optimization as a classification problem. Building on this, we further introduce anchor logits to enhance policy learning. Our analysis reveals that REAL induces a monotonic and bounded gradient weighting, enabling balanced gradient allocation across rollouts and effectively mitigating the identified mismatches. Extensive experiments on mathematical reasoning benchmarks show that REAL improves training stability and consistently outperforms GRPO and strong variants such as DAPO. On the 1.5B model, REAL improves average Pass@1 over DAPO by 6.7%. These gains further scale to 7B model, REAL continues to outperform DAPO and GSPO by 6.2% and 1.7%, respectively. Notably, even with a vanilla binary cross-entropy, REAL remains stable and exceeds DAPO by 4.5% on average.
>
---
#### [replaced 035] Preference Leakage: A Contamination Problem in LLM-as-a-judge
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，研究LLM-as-a-judge中的偏好泄露问题，揭示数据生成器与评估器相关性导致的偏差，提出相关性分类并验证其影响。**

- **链接: [https://arxiv.org/pdf/2502.01534](https://arxiv.org/pdf/2502.01534)**

> **作者:** Dawei Li; Renliang Sun; Yue Huang; Ming Zhong; Bohan Jiang; Jiawei Han; Xiangliang Zhang; Wei Wang; Huan Liu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large Language Models (LLMs) as judges and LLM-based data synthesis have emerged as two fundamental LLM-driven data annotation methods in model development. While their combination significantly enhances the efficiency of model training and evaluation, little attention has been given to the potential contamination brought by this new model development paradigm. In this work, we expose preference leakage, a contamination problem in LLM-as-a-judge caused by the relatedness between the synthetic data generators and LLM-based evaluators. To study this issue, we first define three common relatednesses between the data generator LLM and the judge LLM: being the same model, having an inheritance relationship, and belonging to the same model family. Through extensive experiments, we empirically confirm the bias of judges towards their related student models caused by preference leakage across multiple LLM baselines and benchmarks. Further analysis suggests that preference leakage is a pervasive and real-world problem that is harder to detect compared to previously identified biases in LLM-as-a-judge scenarios. All of these findings imply that preference leakage is a widespread and challenging problem in the area of LLM-as-a-judge. We release all codes and data at: this https URL.
>
---
#### [replaced 036] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on Retrieval-Augmented Generation systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究核心指代对检索增强生成系统的影响。旨在解决文档中指代歧义导致的性能下降问题，通过核心指代消解提升检索与问答效果。**

- **链接: [https://arxiv.org/pdf/2507.07847](https://arxiv.org/pdf/2507.07847)**

> **作者:** Youngjoon Jang; Seongtae Hong; Junyoung Son; Sungjin Park; Chanjun Park; Heuiseok Lim
>
> **备注:** Accepted to ACL 2025 SRW
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a crucial framework in natural language processing (NLP), improving factual consistency and reducing hallucinations by integrating external document retrieval with large language models (LLMs). However, the effectiveness of RAG is often hindered by coreferential complexity in retrieved documents, introducing ambiguity that disrupts in-context learning. In this study, we systematically investigate how entity coreference affects both document retrieval and generative performance in RAG-based systems, focusing on retrieval relevance, contextual understanding, and overall response quality. We demonstrate that coreference resolution enhances retrieval effectiveness and improves question-answering (QA) performance. Through comparative analysis of different pooling strategies in retrieval tasks, we find that mean pooling demonstrates superior context capturing ability after applying coreference resolution. In QA tasks, we discover that smaller models benefit more from the disambiguation process, likely due to their limited inherent capacity for handling referential ambiguity. With these findings, this study aims to provide a deeper understanding of the challenges posed by coreferential complexity in RAG, providing guidance for improving retrieval and generation in knowledge-intensive AI applications.
>
---
#### [replaced 037] CareMedEval dataset: Evaluating Critical Appraisal and Reasoning in the Biomedical Field
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CareMedEval数据集，用于评估大语言模型在生物医学领域中的批判性阅读与推理能力。旨在解决模型在专业领域推理上的局限性。**

- **链接: [https://arxiv.org/pdf/2511.03441](https://arxiv.org/pdf/2511.03441)**

> **作者:** Doria Bonzi; Alexandre Guiggi; Frédéric Béchet; Carlos Ramisch; Benoit Favre
>
> **备注:** Accepted at LREC 2026. To access the dataset, see this https URL
>
> **摘要:** Critical appraisal of scientific literature is an essential skill in the biomedical field. While large language models (LLMs) can offer promising support in this task, their reliability remains limited, particularly for critical reasoning in specialized domains. We introduce CareMedEval, an original dataset designed to evaluate LLMs on biomedical critical appraisal and reasoning tasks. Derived from authentic exams taken by French medical students, the dataset contains 534 questions based on 37 scientific articles. Unlike existing benchmarks, CareMedEval explicitly evaluates critical reading and reasoning grounded in scientific papers. Benchmarking state-of-the-art generalist and biomedical-specialized LLMs under various context conditions reveals the difficulty of the task: open and commercial models fail to exceed an Exact Match Rate of 0.5 even though generating intermediate reasoning tokens considerably improves the results. Yet, models remain challenged especially on questions about study limitations and statistical analysis. CareMedEval provides a challenging benchmark for grounded reasoning, exposing current LLM limitations and paving the way for future development of automated support for critical appraisal.
>
---
#### [replaced 038] Towards Personalized Deep Research: Benchmarks and Evaluations
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于个性化深度研究任务，旨在解决现有评估基准缺乏个性化问题。提出PDR-Bench基准和PQR评估框架，以衡量系统在个性化场景下的表现。**

- **链接: [https://arxiv.org/pdf/2509.25106](https://arxiv.org/pdf/2509.25106)**

> **作者:** Yuan Liang; Jiaxian Li; Yuqing Wang; Piaohong Wang; Motong Tian; Pai Liu; Shuofei Qiao; Runnan Fang; He Zhu; Ge Zhang; Minghao Liu; Yuchen Eleanor Jiang; Ningyu Zhang; Wangchunshu Zhou
>
> **摘要:** Deep Research Agents (DRAs) can autonomously conduct complex investigations and generate comprehensive reports, demonstrating strong real-world potential. However, existing evaluations mostly rely on close-ended benchmarks, while open-ended deep research benchmarks remain scarce and typically neglect personalized scenarios. To bridge this gap, we introduce Personalized Deep Research Bench (PDR-Bench), the first benchmark for evaluating personalization in DRAs. It pairs 50 diverse research tasks across 10 domains with 25 authentic user profiles that combine structured persona attributes with dynamic real-world contexts, yielding 250 realistic user-task queries. To assess system performance, we propose the PQR Evaluation Framework, which jointly measures Personalization Alignment, Content Quality, and Factual Reliability. Our experiments on a range of systems highlight current capabilities and limitations in handling personalized deep research. This work establishes a rigorous foundation for developing and evaluating the next generation of truly personalized AI research assistants.
>
---
#### [replaced 039] LMUnit: Fine-grained Evaluation with Natural Language Unit Tests
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LMUnit，用于细粒度评估语言模型，解决传统评估方法效果差的问题。通过自然语言单元测试和多目标训练，提升评估准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2412.13091](https://arxiv.org/pdf/2412.13091)**

> **作者:** Jon Saad-Falcon; Rajan Vivek; William Berrios; Nandita Shankar Naik; Matija Franklin; Bertie Vidgen; Amanpreet Singh; Douwe Kiela; Shikib Mehri
>
> **摘要:** As language models become integral to critical workflows, assessing their behavior remains a fundamental challenge -- human evaluation is costly and noisy, while automated metrics provide only coarse, difficult-to-interpret signals. We introduce natural language unit tests, a paradigm that decomposes response quality into explicit, testable criteria, along with a unified scoring model, LMUnit, which combines multi-objective training across preferences, direct ratings, and natural language rationales. Through controlled human studies, we show this paradigm significantly improves inter-annotator agreement and enables more effective LLM development workflows. LMUnit achieves state-of-the-art performance on evaluation benchmarks (FLASK, BigGenBench) and competitive results on RewardBench. These results validate both our proposed paradigm and scoring model, suggesting a promising path forward for language model evaluation and development.
>
---
#### [replaced 040] SEVADE: Self-Evolving Multi-Agent Analysis with Decoupled Evaluation for Hallucination-Resistant Irony Detection
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于讽刺检测任务，旨在解决大模型在复杂讽刺语境中易产生幻觉的问题。提出SEVADE框架，通过多智能体分析和解耦评估提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2508.06803](https://arxiv.org/pdf/2508.06803)**

> **作者:** Ziqi Liu; Ziyang Zhou; Yilin Li; Mingxuan Hu; Yushan Pan; Zhijie Xu; Yangbin Chen
>
> **摘要:** Sarcasm detection is a crucial yet challenging Natural Language Processing task. Existing Large Language Model methods are often limited by single-perspective analysis, static reasoning pathways, and a susceptibility to hallucination when processing complex ironic rhetoric, which impacts their accuracy and reliability. To address these challenges, we propose **SEVADE**, a novel **S**elf-**Ev**olving multi-agent **A**nalysis framework with **D**ecoupled **E**valuation for hallucination-resistant sarcasm detection. The core of our framework is a Dynamic Agentive Reasoning Engine (DARE), which utilizes a team of specialized agents grounded in linguistic theory to perform a multifaceted deconstruction of the text and generate a structured reasoning chain. Subsequently, a separate lightweight rationale adjudicator (RA) performs the final classification based solely on this reasoning chain. This decoupled architecture is designed to mitigate the risk of hallucination by separating complex reasoning from the final judgment. Extensive experiments on four benchmark datasets demonstrate that our framework achieves state-of-the-art performance, with average improvements of **6.75%** in Accuracy and **6.29%** in Macro-F1 score.
>
---
#### [replaced 041] OSCAR: Online Soft Compression And Reranking
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出OSCAR，用于检索增强生成（RAG）任务，解决大规模检索计算成本高的问题。通过在线软压缩和重排序，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2504.07109](https://arxiv.org/pdf/2504.07109)**

> **作者:** Maxime Louis; Thibault Formal; Hervé Dejean; Stéphane Clinchant
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as retrieval sizes grow. To address this, we introduce OSCAR, a novel query-dependent online soft compression method that reduces computational overhead while preserving performance. Unlike traditional hard compression methods, which shorten retrieved texts, or soft compression approaches, which map documents to continuous embeddings offline, OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates. Additionally, we extend OSCAR to simultaneously perform reranking, further optimizing the efficiency of the RAG pipeline. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal to no loss in accuracy for LLMs ranging from 1B to 24B parameters. The models are available at: this https URL.
>
---
#### [replaced 042] Generalization of RLVR Using Causal Reasoning as a Testbed
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究RLVR在因果推理任务中的泛化能力，探讨其在不同模型规模和查询层级下的效果，旨在提升大语言模型的因果推理能力。**

- **链接: [https://arxiv.org/pdf/2512.20760](https://arxiv.org/pdf/2512.20760)**

> **作者:** Brian Lu; Hongyu Zhao; Shuo Sun; Hao Peng; Rui Ding; Hongyuan Mei
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for post-training large language models (LLMs) on complex reasoning tasks. Yet, the conditions under which RLVR yields robust generalization remain underexplored. This paper provides an empirical study of RLVR generalization in the setting of probabilistic inference over causal graphical models. This setting offers two natural axes along which to examine generalization: (i) the level of the probabilistic query -- associational, interventional, or counterfactual -- and (ii) the structural complexity of the query, measured by the size of its relevant subgraph. We construct a dataset of causal graphs and queries spanning these difficulty axes and fine-tune Qwen-2.5-Instruct models using RLVR or supervised fine-tuning (SFT). We vary both the model scale (3B-32B) and the query level included in training. We find that RLVR yields stronger within-level and across-level generalization than SFT, but only for specific combinations of model size and training query level. Further analysis shows that RLVR's effectiveness depends on the model's initial reasoning competence. With sufficient initial competence, RLVR improves an LLM's marginalization strategy and reduces errors in intermediate probability calculations, producing substantial accuracy gains, particularly on more complex queries. These results show that RLVR can improve specific causal reasoning subskills, with its benefits emerging only when the model has sufficient initial competence. Our code and data is available at this https URL.
>
---
#### [replaced 043] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决Masked Diffusion-LM解码中计算冗余问题。提出SureLock机制，在token稳定时锁定位置以减少计算量，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.06412](https://arxiv.org/pdf/2602.06412)**

> **作者:** Daisuke Oba; Danushka Bollegala; Masahiro Kaneko; Naoaki Okazaki
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Masked Diffusion Language Models generate sequences via iterative sampling that progressively unmasks tokens. However, they still recompute the attention and feed-forward blocks for every token position at every step -- even when many unmasked tokens are essentially fixed, resulting in substantial waste in compute. We propose SureLock: when the posterior at an unmasked position has stabilized across steps (our sure condition), we lock that position -- thereafter skipping its query projection and feed-forward sublayers -- while caching its attention keys and values so other positions can continue to attend to it. This reduces the dominant per-iteration computational cost from $O(N^2d)$ to $O(MNd)$ where $N$ is the sequence length, $M$ is the number of unlocked token positions, and $d$ is the model dimension. In practice, $M$ decreases as the iteration progresses, yielding substantial savings. On LLaDA-8B, SureLock reduces algorithmic FLOPs by 30--50% relative to the same sampler without locking, while maintaining comparable generation quality. We also provide a theoretical analysis to justify the design rationale of SureLock: monitoring only the local KL at the lock step suffices to bound the deviation in final token probabilities. Our project page is available at this https URL .
>
---
#### [replaced 044] Evaluating Text Style Transfer: A Nine-Language Benchmark for Text Detoxification
- **分类: cs.CL**

- **简介: 该论文属于文本风格迁移任务，旨在解决多语言文本净化评估难题。通过构建九语言基准，对比分析自动评估指标与人类判断的相关性，提出更有效的评估方法。**

- **链接: [https://arxiv.org/pdf/2507.15557](https://arxiv.org/pdf/2507.15557)**

> **作者:** Vitaly Protasov; Nikolay Babakov; Daryna Dementieva; Alexander Panchenko
>
> **备注:** LREC 2026
>
> **摘要:** Despite notable advances in large language models (LLMs), reliable evaluation of text generation tasks such as text style transfer (TST) remains an open challenge. Existing research has shown that automatic metrics often correlate poorly with human judgments (Dementieva et al., 2024; Pauli et al., 2025), limiting our ability to assess model performance accurately. Furthermore, most prior work has focused primarily on English, while the evaluation of multilingual TST systems, particularly for text detoxification, remains largely underexplored. In this paper, we present the first comprehensive multilingual benchmarking study of evaluation metrics for text detoxification evaluation across nine languages: Arabic, Amharic, Chinese, English, German, Hindi, Russian, Spanish, and Ukrainian. Drawing inspiration from machine translation evaluation, we compare neural-based automatic metrics with LLM-as-a-judge approaches together with experiments on task-specific fine-tuned models. Our analysis reveals that the proposed metrics achieve significantly higher correlation with human judgments compared to baseline approaches. We also provide actionable insights and practical guidelines for building robust and reliable multilingual evaluation pipelines for text detoxification and related TST tasks.
>
---
#### [replaced 045] Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于公共健康信息评估任务，旨在检验LLMs对英国政府公共卫生知识的掌握情况。研究构建了PubHealthBench基准，测试24个LLMs在多项选择和自由回答中的表现。**

- **链接: [https://arxiv.org/pdf/2505.06046](https://arxiv.org/pdf/2505.06046)**

> **作者:** Joshua Harris; Fan Grayson; Felix Feldman; Timothy Laurence; Toby Nonnenmacher; Oliver Higgins; Leo Loman; Selina Patel; Thomas Finnie; Samuel Collins; Michael Borowitz
>
> **备注:** 27 pages, 9 pages main text
>
> **摘要:** As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in the domains of medicine and public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, while there are a number of LLM benchmarks in the medical domain, currently little is known about LLM knowledge within the field of public health. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest proprietary LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, while there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses.
>
---
#### [replaced 046] MultiWikiQA: A Reading Comprehension Benchmark in 300+ Languages
- **分类: cs.CL**

- **简介: 该论文提出MultiWikiQA，一个覆盖306种语言的阅读理解基准数据集，解决多语言阅读理解任务。通过生成高质量问题与答案对，评估模型在多种语言中的表现。**

- **链接: [https://arxiv.org/pdf/2509.04111](https://arxiv.org/pdf/2509.04111)**

> **作者:** Dan Saattrup Smart
>
> **备注:** Camera-ready version for LREC 2026
>
> **摘要:** We introduce a new reading comprehension dataset, dubbed MultiWikiQA, which covers 306 languages and has 1,220,757 samples in total. We start with Wikipedia articles, which also provide the context for the dataset samples, and use an LLM to generate question/answer pairs related to the Wikipedia article, ensuring that the answer appears verbatim within the article. Next, the question is then rephrased to hinder simple word matching methods from performing well on the dataset. We conduct a crowdsourced human evaluation of the fluency of the generated questions, which included 156 respondents across 30 of the languages (both low- and high-resource). All 30 languages received a mean fluency rating above ``mostly natural'', showing that the samples are of good quality. We evaluate 6 different language models, both decoder and encoder models of varying sizes, showing that the benchmark is sufficiently difficult and that there is a large performance discrepancy amongst the languages. Both the dataset and survey evaluations are publicly available.
>
---
#### [replaced 047] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型在未见任务中的泛化能力，聚焦于off-by-one加法任务。通过可解释性方法分析模型内部机制，揭示了函数归纳机制及其在多种任务中的复用性。**

- **链接: [https://arxiv.org/pdf/2507.09875](https://arxiv.org/pdf/2507.09875)**

> **作者:** Qinyuan Ye; Robin Jia; Xiang Ren
>
> **备注:** ICLR 2026. Code: this https URL
>
> **摘要:** Large language models demonstrate the intriguing ability to perform unseen tasks via in-context learning. However, it remains unclear what mechanisms inside the model drive such task-level generalization. In this work, we approach this question through the lens of off-by-one addition (i.e., 1+1=3, 2+2=5, 3+3=?), a two-step, counterfactual task with an unexpected +1 function as a second step. Leveraging circuit-style interpretability techniques such as path patching, we analyze the models' internal computations behind their performance and present three key findings. First, we identify a mechanism that explains the model's generalization from standard addition to off-by-one addition. It resembles the induction head mechanism described in prior work, yet operates at a higher level of abstraction; we therefore term it "function induction" in this work. Second, we show that the induction of the +1 function is governed by multiple attention heads in parallel, each of which emits a distinct piece of the +1 function. Finally, we find that this function induction mechanism is reused in a broader range of tasks, including synthetic tasks such as shifted multiple-choice QA and algorithmic tasks such as base-8 addition. Overall, our findings offer deeper insights into how reusable and composable structures within language models enable task-level generalization.
>
---
#### [replaced 048] Annotation-Efficient Universal Honesty Alignment
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的诚实对齐问题，旨在提升模型在知识边界内的可信度表达。通过提出EliCal框架，实现高效标注下的模型校准，解决传统方法依赖大量标注的问题。**

- **链接: [https://arxiv.org/pdf/2510.17509](https://arxiv.org/pdf/2510.17509)**

> **作者:** Shiyu Ni; Keping Bi; Jiafeng Guo; Minghao Tang; Jingtong Wu; Zengxin Han; Xueqi Cheng
>
> **备注:** ICLR 2026
>
> **摘要:** Honesty alignment-the ability of large language models (LLMs) to recognize their knowledge boundaries and express calibrated confidence-is essential for trustworthy deployment. Existing methods either rely on training-free confidence estimation (e.g., token probabilities, self-consistency) or training-based calibration with correctness annotations. While effective, achieving universal honesty alignment with training-based calibration requires costly, large-scale labeling. To support annotation-efficient training, we introduce Elicitation-Then-Calibration (EliCal), a two-stage framework that first elicits internal confidence using inexpensive self-consistency supervision, then calibrates this confidence with a small set of correctness annotations. To support a large-scale study, we release HonestyBench, a benchmark covering ten free-form QA datasets with 560k training and 70k evaluation instances annotated with correctness and self-consistency signals. Experiments show that EliCal achieves near-optimal alignment with only 1k correctness annotations (0.18% of full supervision) and better alignment performance on unseen MMLU tasks than the calibration-only baseline, offering a scalable solution toward universal honesty alignment in LLMs.
>
---
#### [replaced 049] A Systematic Analysis of Biases in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在检测大语言模型中的偏见。通过实验分析政治、意识形态、联盟、语言和性别等方面的偏差，揭示模型存在的不公正倾向。**

- **链接: [https://arxiv.org/pdf/2512.15792](https://arxiv.org/pdf/2512.15792)**

> **作者:** Xulang Zhang; Rui Mao; Erik Cambria
>
> **摘要:** Large language models (LLMs) have rapidly become indispensable tools for acquiring information and supporting human decision-making. However, ensuring that these models uphold fairness across varied contexts is critical to their safe and responsible deployment. In this study, we undertake a comprehensive examination of four widely adopted LLMs, probing their underlying biases and inclinations across the dimensions of politics, ideology, alliance, language, and gender. Through a series of carefully designed experiments, we investigate their political neutrality using news summarization, ideological biases through news stance classification, tendencies toward specific geopolitical alliances via United Nations voting patterns, language bias in the context of multilingual story completion, and gender-related affinities as revealed by responses to the World Values Survey. Results indicate that while the LLMs are aligned to be neutral and impartial, they still show biases and affinities of different types.
>
---
#### [replaced 050] Meenz bleibt Meenz, but Large Language Models Do Not Speak Its Dialect
- **分类: cs.CL**

- **简介: 该论文属于语言保护任务，旨在解决德语方言Meenzerisch研究不足的问题。工作包括构建首个NLP可用的方言词典，并测试大语言模型在方言词义生成和词汇生成上的表现。**

- **链接: [https://arxiv.org/pdf/2602.16852](https://arxiv.org/pdf/2602.16852)**

> **作者:** Minh Duc Bui; Manuel Mager; Peter Herbert Kann; Katharina von der Wense
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Meenzerisch, the dialect spoken in the German city of Mainz, is also the traditional language of the Mainz carnival, a yearly celebration well known throughout Germany. However, Meenzerisch is on the verge of dying out-a fate it shares with many other German dialects. Natural language processing (NLP) has the potential to help with the preservation and revival efforts of languages and dialects. However, so far no NLP research has looked at Meenzerisch. This work presents the first research in the field of NLP that is explicitly focused on the dialect of Mainz. We introduce a digital dictionary-an NLP-ready dataset derived from an existing resource (Schramm, 1966)-to support researchers in modeling and benchmarking the language. It contains 2,351 words in the dialect paired with their meanings described in Standard German. We then use this dataset to answer the following research questions: (1) Can state-of-the-art large language models (LLMs) generate definitions for dialect words? (2) Can LLMs generate words in Meenzerisch, given their definitions? Our experiments show that LLMs can do neither: the best model for definitions reaches only 6.27% accuracy and the best word generation model's accuracy is 1.51%. We then conduct two additional experiments in order to see if accuracy is improved by few-shot learning and by extracting rules from the training set, which are then passed to the LLM. While those approaches are able to improve the results, accuracy remains below 10%. This highlights that additional resources and an intensification of research efforts focused on German dialects are desperately needed.
>
---
#### [replaced 051] Categorical Emotions or Appraisals - Which Emotion Model Explains Argument Convincingness Better?
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在探讨情绪模型对论点说服力预测的影响。研究比较了分类情绪与评估理论的效果，发现评估更能提升说服力预测。**

- **链接: [https://arxiv.org/pdf/2511.07162](https://arxiv.org/pdf/2511.07162)**

> **作者:** Lynn Greschner; Meike Bauer; Sabine Weber; Roman Klinger
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** The convincingness of an argument does not only depend on its structure (logos), the person who makes the argument (ethos), but also on the emotion that it causes in the recipient (pathos). While the overall intensity and categorical values of emotions in arguments have received considerable attention in the research community, we argue that the emotion an argument evokes in a recipient is subjective. It depends on the recipient's goals, standards, prior knowledge, and stance. Appraisal theories lend themselves as a link between the subjective cognitive assessment of events and emotions. They have been used in event-centric emotion analysis, but their suitability for assessing argument convincingness remains unexplored. In this paper, we evaluate whether appraisal theories are suitable for emotion analysis in arguments by considering subjective cognitive evaluations of the importance and impact of an argument on its receiver. Based on the annotations in the recently published ContArgA corpus, we perform zero-shot prompting experiments to evaluate the importance of gold-annotated and predicted emotions and appraisals for the assessment of the subjective convincingness labels. We find that, while categorical emotion information does improve convincingness prediction, the improvement is more pronounced with appraisals. This work presents the first systematic comparison between emotion models for convincingness prediction, demonstrating the advantage of appraisals, providing insights for theoretical and practical applications in computational argumentation.
>
---
#### [replaced 052] Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naive Integration via Multi-Objective Balanced Covering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉token剪枝任务，解决静态策略导致性能不一致的问题。提出MoB方法，通过多目标平衡覆盖优化剪枝效果，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2505.10118](https://arxiv.org/pdf/2505.10118)**

> **作者:** Yangfu Li; Hongjian Zhan; Tianyi Chen; Qi Liu; Yue Lu
>
> **备注:** 31 pages,9 figures,conference
>
> **摘要:** Existing visual token pruning methods target prompt alignment and visual preservation with static strategies, overlooking the varying relative importance of these objectives across tasks, which leads to inconsistent performance. To address this, we derive the first closed-form error bound for visual token pruning based on the Hausdorff distance, uniformly characterizing the contributions of both objectives. Moreover, leveraging $\epsilon$-covering theory, we reveal an intrinsic trade-off between these objectives and quantify their optimal attainment levels under a fixed budget. To practically handle this trade-off, we propose Multi-Objective Balanced Covering (MoB), which reformulates visual token pruning as a bi-objective covering problem. In this framework, the attainment trade-off reduces to budget allocation via greedy radius trading. MoB offers a provable performance bound and linear scalability with respect to the number of input visual tokens, enabling adaptation to challenging pruning scenarios. Extensive experiments show that MoB preserves 96.4% of performance for LLaVA-1.5-7B using only 11.1% of the original visual tokens and accelerates LLaVA-Next-7B by 1.3-1.5$\times$ with negligible performance loss. Additionally, evaluations on Qwen2-VL and Video-LLaVA confirm that MoB integrates seamlessly into advanced MLLMs and diverse vision-language tasks.
>
---
#### [replaced 053] A Study on Building Efficient Zero-Shot Relation Extraction Models
- **分类: cs.CL**

- **简介: 该论文属于零样本关系抽取任务，旨在解决模型在真实场景下的鲁棒性问题。针对现有模型的不足，提出改进策略并进行对比实验。**

- **链接: [https://arxiv.org/pdf/2603.01266](https://arxiv.org/pdf/2603.01266)**

> **作者:** Hugo Thomas; Caio Corro; Guillaume Gravier; Pascale Sébillot
>
> **备注:** LREC 2026
>
> **摘要:** Zero-shot relation extraction aims to identify relations between entity mentions using textual descriptions of novel types (i.e., previously unseen) instead of labeled training examples. Previous works often rely on unrealistic assumptions: (1) pairs of mentions are often encoded directly in the input, which prevents offline pre-computation for large scale document database querying; (2) no rejection mechanism is introduced, biasing the evaluation when using these models in a retrieval scenario where some (and often most) inputs are irrelevant and must be ignored. In this work, we study the robustness of existing zero-shot relation extraction models when adapting them to a realistic extraction scenario. To this end, we introduce a typology of existing models, and propose several strategies to build single pass models and models with a rejection mechanism. We adapt several state-of-the-art tools, and compare them in this challenging setting, showing that no existing work is really robust to realistic assumptions, but overall AlignRE (Li et al., 2024) performs best along all criteria.
>
---
#### [replaced 054] MuSaG: A Multimodal German Sarcasm Dataset with Full-Modal Annotations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MuSaG，首个德语多模态讽刺数据集，解决多模态讽刺检测问题。收集33分钟电视片段，提供文本、音频、视频及人工标注，用于评估模型性能。**

- **链接: [https://arxiv.org/pdf/2510.24178](https://arxiv.org/pdf/2510.24178)**

> **作者:** Aaron Scott; Maike Züfle; Jan Niehues
>
> **摘要:** Sarcasm is a complex form of figurative language in which the intended meaning contradicts the literal one. Its prevalence in social media and popular culture poses persistent challenges for natural language understanding, sentiment analysis, and content moderation. With the emergence of multimodal large language models, sarcasm detection extends beyond text and requires integrating cues from audio and vision. We present MuSaG, the first German multimodal sarcasm detection dataset, consisting of 33 minutes of manually selected and human-annotated statements from German television shows. Each instance provides aligned text, audio, and video modalities, annotated separately by humans, enabling evaluation in unimodal and multimodal settings. We benchmark nine open-source and commercial models, spanning text, audio, vision, and multimodal architectures, and compare their performance to human annotations. Our results show that while humans rely heavily on audio in conversational settings, models perform best on text. This highlights a gap in current multimodal models and motivates the use of MuSaG for developing models better suited to realistic scenarios. We release MuSaG publicly to support future research on multimodal sarcasm detection and human-model alignment.
>
---
#### [replaced 055] Circuit Insights: Towards Interpretability Beyond Activations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于可解释AI任务，旨在提升神经网络的可解释性。解决现有方法依赖手动分析和外部模型的问题，提出WeightLens和CircuitLens，直接从权重分析特征，揭示电路动态。**

- **链接: [https://arxiv.org/pdf/2510.14936](https://arxiv.org/pdf/2510.14936)**

> **作者:** Elena Golimblevskaia; Aakriti Jain; Bruno Puri; Ammar Ibrahim; Wojciech Samek; Sebastian Lapuschkin
>
> **摘要:** The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality.
>
---
#### [replaced 056] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于GUI交互任务，解决多模态代理在切换操作上的可靠性问题。提出StaR方法，提升切换指令执行准确率。**

- **链接: [https://arxiv.org/pdf/2509.13615](https://arxiv.org/pdf/2509.13615)**

> **作者:** Zongru Wu; Rui Mao; Zhiyuan Tian; Pengzhou Cheng; Tianjie Ju; Zheng Wu; Lingzhong Dong; Haiyue Sheng; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions derived from public datasets. Evaluation results of existing agents demonstrate their notable unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a multimodal reasoning method that enables agents to perceive the current toggle state, infer the desired state from the instruction, and act accordingly. Experiments on four multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public agentic benchmarks show that StaR also enhances general agentic task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code and benchmark: this https URL.
>
---
#### [replaced 057] Citation Failure: Definition, Analysis and Efficient Mitigation
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，解决LLM-RAG系统中的引用失败问题。通过分析引用失败原因并提出CITENTION框架提升引用效果。**

- **链接: [https://arxiv.org/pdf/2510.20303](https://arxiv.org/pdf/2510.20303)**

> **作者:** Jan Buchmann; Iryna Gurevych
>
> **备注:** Under review. Paper repository: this https URL
>
> **摘要:** Citations from LLM-based RAG systems are supposed to simplify response verification. However, this goal is undermined in cases of citation failure, where a model generates a helpful response, but fails to generate citations to complete evidence. In contrast to previous work, we propose to disentangle this from response failure, where the response itself is flawed, and citing complete evidence is impossible. To address citation failure, this work follows a two-step approach: (1) We study when citation failure occurs and (2) how it can be mitigated efficiently. For step 1, we extend prior work by investigating how the relation between response and evidence affects citation quality. We introduce CITECONTROL, a benchmark that systematically varies this relation to enable the analysis of failure modes. Experiments show that failures increase with relational complexity and suggest that combining citation methods could improve performance, motivating step 2. To study the efficient improvement of LLM citation, we propose CITENTION, a framework integrating generative, attention-based, and retrieval-based methods. Results demonstrate substantial citation improvements on CITECONTROL and in transfer settings. We make our data and code publicly available.
>
---
#### [replaced 058] Context Biasing for Pronunciation-Orthography Mismatch in Automatic Speech Recognition
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自动语音识别任务，解决发音与拼写不一致导致的识别错误问题。通过引入上下文偏差方法，提升难词识别准确率。**

- **链接: [https://arxiv.org/pdf/2506.18703](https://arxiv.org/pdf/2506.18703)**

> **作者:** Christian Huber; Alexander Waibel
>
> **摘要:** Neural sequence-to-sequence systems deliver state-of-the-art performance for automatic speech recognition. When using appropriate modeling units, e.g., byte-pair encoding, these systems are in principle open vocabulary systems. In practice, however, they often fail to recognize words not seen during training, e.g., named entities, acronyms, or domain-specific special words. To address this problem, many context biasing methods have been proposed; however, these methods may still struggle when they are unable to relate audio and corresponding text, e.g., in case of a pronunciation-orthography mismatch. We propose a method where corrections of substitution errors can be used to improve the recognition accuracy of such challenging words. Users can add corrections on the fly during inference. We show that with this method we get a relative improvement in biased word error rate between 22% and 34% compared to a text-based replacement method, while maintaining the overall performance.
>
---
#### [replaced 059] Trust Me, I Can Convince You: The Contextualized Argument Appraisal Framework
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决arguments convincingness的评估问题。提出Contextualized Argument Appraisal Framework，通过角色扮演标注数据，研究情绪与说服力的关系。**

- **链接: [https://arxiv.org/pdf/2509.17844](https://arxiv.org/pdf/2509.17844)**

> **作者:** Lynn Greschner; Sabine Weber; Roman Klinger
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Emotions that somebody develops based on an argument do not only depend on the argument itself - they are also influenced by a subjective evaluation of the argument's potential impact on the self. For instance, an argument to ban plastic bottles might cause fear of losing a job for a bottle industry worker, which lowers the convincingness - presumably independent of its content. While binary emotionality of arguments has been studied, such cognitive appraisal models have only been proposed in other subtasks of emotion analysis, but not in the context of arguments and their convincingness. To fill this research gap, we propose the Contextualized Argument Appraisal Framework to model the interplay between the sender, receiver, and argument. We adapt established appraisal models from psychology to argument mining, including argument pleasantness, familiarity, response urgency, and expected effort, as well as convincingness variables. To evaluate the framework and pave the way for computational modeling, we develop a novel role-playing-based annotation setup, mimicking real-world exposure to arguments. Participants disclose their emotion, explain the main cause, the argument appraisal, and the perceived convincingness. To consider the subjective nature of such annotations, we also collect demographic data and personality traits of both the participants and ask them to disclose the same variables for their perception of the argument sender. The analysis of the resulting ContArgA corpus of 4000 annotations reveals that convincingness is positively correlated with positive emotions (e.g., trust) and negatively correlated with negative emotions (e.g., anger). The appraisal variables particularly point to the importance of the annotator's familiarity with the argument.
>
---
#### [replaced 060] Dutch Metaphor Extraction from Cancer Patients' Interviews and Forum Data using LLMs and Human in the Loop
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的隐喻提取任务，旨在从荷兰语癌症患者数据中提取隐喻，以改善医患沟通和个性化护理。**

- **链接: [https://arxiv.org/pdf/2511.06427](https://arxiv.org/pdf/2511.06427)**

> **作者:** Lifeng Han; David Lindevelt; Sander Puts; Erik van Mulligen; Suzan Verberne
>
> **备注:** Ongoing project report, on behalf of 4D PICTURE this https URL
>
> **摘要:** Metaphors and metaphorical language (MLs) play an important role in healthcare communication between clinicians, patients, and patients' family members. In this work, we focus on Dutch language data from cancer patients. We extract metaphors used by patients using two data sources: (1) cancer patient storytelling interview data and (2) online forum data, including patients' posts, comments, and questions to professionals. We investigate how current state-of-the-art large language models (LLMs) perform on this task by exploring different prompting strategies such as chain of thought reasoning, few-shot learning, and self-prompting. With a human-in-the-loop setup, we verify the extracted metaphors and compile the outputs into a corpus named this http URL. We believe the extracted metaphors can support better patient care, for example shared decision making, improved communication between patients and clinicians, and enhanced patient health literacy. They can also inform the design of personalized care pathways. We share prompts and related resources at this https URL
>
---
#### [replaced 061] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering
- **分类: cs.CL**

- **简介: 该论文属于心理健康问答任务，旨在评估大语言模型在真实患者问题上的表现。研究构建了CounselBench基准，包含专家评估和对抗数据集，揭示模型在安全性和个性化方面的不足。**

- **链接: [https://arxiv.org/pdf/2506.08584](https://arxiv.org/pdf/2506.08584)**

> **作者:** Yahan Li; Jifan Yao; John Bosco S. Bunyi; Adam C. Frank; Angel Hsing-Chi Hwang; Ruishan Liu
>
> **摘要:** Medical question answering (QA) benchmarks often focus on multiple-choice or fact-based tasks, leaving open-ended answers to real patient questions underexplored. This gap is particularly critical in mental health, where patient questions often mix symptoms, treatment concerns, and emotional needs, requiring answers that balance clinical caution with contextual sensitivity. We present CounselBench, a large-scale benchmark developed with 100 mental health professionals to evaluate and stress-test large language models (LLMs) in realistic help-seeking scenarios. The first component, CounselBench-EVAL, contains 2,000 expert evaluations of answers from GPT-4, LLaMA 3, Gemini, and online human therapists on patient questions from the public forum CounselChat. Each answer is rated across six clinically grounded dimensions, with span-level annotations and written rationales. Expert evaluations show that while LLMs achieve high scores on several dimensions, they also exhibit recurring issues, including unconstructive feedback, overgeneralization, and limited personalization or relevance. Responses were frequently flagged for safety risks, most notably unauthorized medical advice. Follow-up experiments show that LLM judges systematically overrate model responses and overlook safety concerns identified by human experts. To probe failure modes more directly, we construct CounselBench-Adv, an adversarial dataset of 120 expert-authored mental health questions designed to trigger specific model issues. Expert evaluation of 1,080 responses from nine LLMs reveals consistent, model-specific failure patterns. Together, CounselBench establishes a clinically grounded framework for benchmarking LLMs in mental health QA.
>
---
#### [replaced 062] Boosting In-Context Learning in LLMs Through the Lens of Classical Supervised Learning
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的分类任务，旨在解决LLM在ICL中因偏差导致的性能不稳定问题。通过提出SC框架，优化模型决策边界，提升分类准确性。**

- **链接: [https://arxiv.org/pdf/2505.23783](https://arxiv.org/pdf/2505.23783)**

> **作者:** Korel Gundem; Juncheng Dong; Dennis Zhang; Vahid Tarokh; Zhengling Qi
>
> **摘要:** In-Context Learning (ICL) allows Large Language Models (LLMs) to adapt to new tasks with just a few examples, but their predictions often suffer from systematic biases, leading to unstable performance in classification. While calibration techniques are proposed to mitigate these biases, we show that, in the logit space, many of these methods are equivalent to merely shifting the LLM's decision boundary without having the ability to alter its orientation. This proves inadequate when biases cause the LLM to be severely misaligned. To address these limitations and provide a unifying framework, we propose Supervised Calibration (SC), a loss-minimization-based framework, which learns an optimal, per-class affine transformation of LLM's predictive probabilities in the logit space without requiring external data beyond the context. By using a more expressive functional class, SC not only subsumes many existing calibration methods in ICL as special cases but also enables the ability of altering and even completely reversing the orientation of the LLM's decision boundary. Furthermore, SC's loss-based nature facilitates the seamless integration of two purpose-built regularization techniques, context-invariance and directional trust-region regularizers. The former is designed to tackle the instability issue in ICL, while the latter is to control the degree of calibration. Finally, SC delivers state-of-the-art performance over calibration baselines in the 4-shot, 8-shot, and 16-shot settings across all nine datasets for Mistral-7B-Instruct-v0.3, Llama-2-7B-chat, and Qwen2-7B-Instruct.
>
---
