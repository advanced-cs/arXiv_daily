# 自然语言处理 cs.CL

- **最新发布 70 篇**

- **更新 50 篇**

## 最新发布

#### [new 001] A Diagrammatic Calculus for a Functional Model of Natural Language Semantics
- **分类: cs.CL; cs.PL; J.5; D.3.1; D.3.3**

- **简介: 该论文属于自然语言语义研究，旨在提升传统指称风格的表达能力。通过函数式编程方法构建类型与效应系统，并设计图示演算以高效计算句子的指称。**

- **链接: [http://arxiv.org/pdf/2507.00782v1](http://arxiv.org/pdf/2507.00782v1)**

> **作者:** Matthieu Pierre Boyer
>
> **备注:** 15 pages, preprint before submission to CSL 2026
>
> **摘要:** In this paper, we study a functional programming approach to natural language semantics, allowing us to increase the expressivity of a more traditional denotation style. We will formalize a category based type and effect system, and construct a diagrammatic calculus to model parsing and handling of effects, and use it to efficiently compute the denotations for sentences.
>
---
#### [new 002] Transferable Modeling Strategies for Low-Resource LLM Tasks: A Prompt and Alignment-Based
- **分类: cs.CL**

- **简介: 该论文属于低资源语言任务，解决大模型在少数据下的迁移与适应问题。提出融合知识对齐和软提示调优的框架，提升模型泛化与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.00601v1](http://arxiv.org/pdf/2507.00601v1)**

> **作者:** Shuangquan Lyu; Yingnan Deng; Guiran Liu; Zhen Qi; Ruotong Wang
>
> **摘要:** This paper addresses the limited transfer and adaptation capabilities of large language models in low-resource language scenarios. It proposes a unified framework that combines a knowledge transfer module with parameter-efficient fine-tuning strategies. The method introduces knowledge alignment loss and soft prompt tuning to guide the model in effectively absorbing the structural features of target languages or tasks under minimal annotation. This enhances both generalization performance and training stability. The framework includes lightweight adaptation modules to reduce computational costs. During training, it integrates freezing strategies and prompt injection to preserve the model's original knowledge while enabling quick adaptation to new tasks. The study also conducts stability analysis experiments and synthetic pseudo-data transfer experiments to systematically evaluate the method's applicability and robustness across different low-resource tasks. Experimental results show that compared with existing multilingual pre-trained models and mainstream transfer methods, the proposed approach achieves higher performance and stability on cross-lingual tasks such as MLQA, XQuAD, and PAWS-X. It demonstrates particularly strong advantages under extremely data-scarce conditions. The proposed method offers strong generality and scalability. It enhances task-specific adaptability while preserving the general capabilities of large language models. This makes it well-suited for complex semantic modeling and multilingual processing tasks.
>
---
#### [new 003] LineRetriever: Planning-Aware Observation Reduction for Web Agents
- **分类: cs.CL**

- **简介: 该论文属于Web导航任务，解决网页内容过长导致模型无法有效处理的问题。提出LineRetriever方法，通过关注未来动作预测来优化检索。**

- **链接: [http://arxiv.org/pdf/2507.00210v1](http://arxiv.org/pdf/2507.00210v1)**

> **作者:** Imene Kerboua; Sahar Omidi Shayegan; Megh Thakkar; Xing Han Lù; Massimo Caccia; Véronique Eglin; Alexandre Aussem; Jérémy Espinas; Alexandre Lacoste
>
> **摘要:** While large language models have demonstrated impressive capabilities in web navigation tasks, the extensive context of web pages, often represented as DOM or Accessibility Tree (AxTree) structures, frequently exceeds model context limits. Current approaches like bottom-up truncation or embedding-based retrieval lose critical information about page state and action history. This is particularly problematic for adaptive planning in web agents, where understanding the current state is essential for determining future actions. We hypothesize that embedding models lack sufficient capacity to capture plan-relevant information, especially when retrieving content that supports future action prediction. This raises a fundamental question: how can retrieval methods be optimized for adaptive planning in web navigation tasks? In response, we introduce \textit{LineRetriever}, a novel approach that leverages a language model to identify and retrieve observation lines most relevant to future navigation steps. Unlike traditional retrieval methods that focus solely on semantic similarity, \textit{LineRetriever} explicitly considers the planning horizon, prioritizing elements that contribute to action prediction. Our experiments demonstrate that \textit{LineRetriever} can reduce the size of the observation at each step for the web agent while maintaining consistent performance within the context limitations.
>
---
#### [new 004] Methodological Rigour in Algorithm Application: An Illustration of Topic Modelling Algorithm
- **分类: cs.CL**

- **简介: 该论文属于方法论研究，旨在解决算法应用中的严谨性问题。通过介绍主题建模算法的应用指南，提升研究可信度。**

- **链接: [http://arxiv.org/pdf/2507.00547v1](http://arxiv.org/pdf/2507.00547v1)**

> **作者:** Malmi Amadoru
>
> **摘要:** The rise of advanced computational algorithms has opened new avenues for computationally intensive research approaches to theory development. However, the opacity of these algorithms and lack of transparency and rigour in their application pose methodological challenges, potentially undermining trust in research. The discourse on methodological rigour in this new genre of research is still emerging. Against this backdrop, I attempt to offer guidance on methodological rigour, particularly in the context of topic modelling algorithms. By illustrating the application of the structural topic modelling algorithm and presenting a set of guidelines, I discuss how to ensure rigour in topic modelling studies. Although the guidelines are for the application of topic modelling algorithms, they can be applied to other algorithms with context-specific adjustments. The guidelines are helpful, especially for novice researchers applying topic modelling, and editors and reviewers handling topic modelling manuscripts. I contribute to the literature on topic modelling and join the emerging dialogue on methodological rigour in computationally intensive theory construction research.
>
---
#### [new 005] Mathematics Isn't Culture-Free: Probing Cultural Gaps via Entity and Scenario Perturbations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决数学问题中的文化偏见问题。通过构建多地区文化适配的数据集，评估大模型在不同文化背景下的表现差异。**

- **链接: [http://arxiv.org/pdf/2507.00883v1](http://arxiv.org/pdf/2507.00883v1)**

> **作者:** Aditya Tomar; Nihar Ranjan Sahoo; Ashish Mittal; Rudra Murthy; Pushpak Bhattacharyya
>
> **摘要:** Although mathematics is often considered culturally neutral, the way mathematical problems are presented can carry implicit cultural context. Existing benchmarks like GSM8K are predominantly rooted in Western norms, including names, currencies, and everyday scenarios. In this work, we create culturally adapted variants of the GSM8K test set for five regions Africa, India, China, Korea, and Japan using prompt-based transformations followed by manual verification. We evaluate six large language models (LLMs), ranging from 8B to 72B parameters, across five prompting strategies to assess their robustness to cultural variation in math problem presentation. Our findings reveal a consistent performance gap: models perform best on the original US-centric dataset and comparatively worse on culturally adapted versions. However, models with reasoning capabilities are more resilient to these shifts, suggesting that deeper reasoning helps bridge cultural presentation gaps in mathematical tasks
>
---
#### [new 006] Natural language processing for African languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦非洲语言的自然语言处理任务，解决低资源语言数据不足与模型性能受限的问题，通过构建高质量语料和优化多语言模型实现改进。**

- **链接: [http://arxiv.org/pdf/2507.00297v1](http://arxiv.org/pdf/2507.00297v1)**

> **作者:** David Ifeoluwa Adelani
>
> **备注:** PhD thesis
>
> **摘要:** Recent advances in word embeddings and language models use large-scale, unlabelled data and self-supervised learning to boost NLP performance. Multilingual models, often trained on web-sourced data like Wikipedia, face challenges: few low-resource languages are included, their data is often noisy, and lack of labeled datasets makes it hard to evaluate performance outside high-resource languages like English. In this dissertation, we focus on languages spoken in Sub-Saharan Africa where all the indigenous languages in this region can be regarded as low-resourced in terms of the availability of labelled data for NLP tasks and unlabelled data found on the web. We analyse the noise in the publicly available corpora, and curate a high-quality corpus, demonstrating that the quality of semantic representations learned in word embeddings does not only depend on the amount of data but on the quality of pre-training data. We demonstrate empirically the limitations of word embeddings, and the opportunities the multilingual pre-trained language model (PLM) offers especially for languages unseen during pre-training and low-resource scenarios. We further study how to adapt and specialize multilingual PLMs to unseen African languages using a small amount of monolingual texts. To address the under-representation of the African languages in NLP research, we developed large scale human-annotated labelled datasets for 21 African languages in two impactful NLP tasks: named entity recognition and machine translation. We conduct an extensive empirical evaluation using state-of-the-art methods across supervised, weakly-supervised, and transfer learning settings.
>
---
#### [new 007] Two-Stage Reasoning-Infused Learning: Improving Classification with LLM-Generated Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在提升分类模型的性能与可解释性。通过两阶段方法，利用大语言模型生成推理过程，增强下游模型的分类效果。**

- **链接: [http://arxiv.org/pdf/2507.00214v1](http://arxiv.org/pdf/2507.00214v1)**

> **作者:** Mads Henrichsen; Rasmus Krebs
>
> **摘要:** Standard classification models often map inputs directly to labels without explicit reasoning, potentially limiting their performance, robustness, and interpretability. This paper introduces a novel two-stage approach to enhance text classification by leveraging Large Language Model (LLM)-generated reasonings. In the first stage, we fine-tune a Llama-3.2-1B-Instruct model (henceforth Llama-R-Gen) on a general-purpose reasoning dataset (syvai/reasoning-gen) to generate textual reasoning (R) given a question and its answer. In the second stage, this generally trained Llama-R-Gen is used offline to create an augmented training dataset for a downstream generative model. This downstream model, based on Llama-3.2-1B-Instruct, takes only the input text (Q) and is trained to output the generated reasoning (R) immediately followed by the predicted emotion (A). We demonstrate this methodology on the dair-ai/emotion dataset for emotion classification. Our experiments show that the generative model trained to output reasoning and the emotion (Classifier Q->RA) achieves a significant improvement of 8.7 percentage points in accuracy (for emotion prediction) compared to a baseline generative model trained solely to output the emotion (Classifier Q->A), highlighting the strong generalization capabilities of the reasoning generation and the benefit of explicit reasoning training. This work underscores the potential of LLM-generated reasonings for creating richer training datasets, thereby improving the performance of diverse downstream NLP tasks and providing explicit explanations.
>
---
#### [new 008] Many LLMs Are More Utilitarian Than One
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; I.2.11**

- **简介: 该论文属于AI伦理与道德推理任务，研究多模型协作中的道德判断行为，探讨其与人类群体决策的异同。**

- **链接: [http://arxiv.org/pdf/2507.00814v1](http://arxiv.org/pdf/2507.00814v1)**

> **作者:** Anita Keshmirian; Razan Baltaji; Babak Hemmatian; Hadi Asghari; Lav R. Varshney
>
> **备注:** 9 pages, 8 Figures, 7 tables
>
> **摘要:** Moral judgment is integral to large language model (LLM) alignment and social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function collectively during collaboration, compared to individual agents. In human moral judgment, group deliberation leads to a utilitarian boost: a tendency to endorse norm violations that maximize benefits for the greatest number of people despite harms. We study whether a similar dynamic emerges in multi-agent LLM systems. We tested six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reasoned independently, and (2) Group, where they engaged in multi-turn discussions in pairs or triads. In personal moral dilemmas, where agents must decide to directly harm one individual to maximize the utility for others, all models found moral violations to be more acceptable when part of a group than individually, similar to human experiments. Some models endorsed actions that maximized overall well-being, even if they benefited strangers over familiar individuals. Others became more willing to violate moral norms in groups. However, while human groups show a similar action bias, the mechanism for their utilitarian boost differs from LLMs. Whereas the human shift comes from heightened sensitivity to decision outcomes, LLM groups show either reduced norm sensitivity or enhanced impartiality. This suggests that while the surface behavior of LLM collectives mimics human group reasoning, the underlying drivers differ. We discuss the implications for AI alignment, multi-agent design, and artificial moral reasoning.
>
---
#### [new 009] Impact of Fine-Tuning Methods on Memorization in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究预训练模型微调方法对记忆隐私的影响。针对微调过程中隐私泄露问题，通过成员推理攻击评估不同方法，发现提示微调更安全。**

- **链接: [http://arxiv.org/pdf/2507.00258v1](http://arxiv.org/pdf/2507.00258v1)**

> **作者:** Jie Hou; Chuxiong Wu; Lannan Luo; Qiang Zeng
>
> **摘要:** As the capabilities of pre-trained large language models (LLMs) continue to advance, the "pre-train and fine-tune" paradigm has become increasingly mainstream, leading to the development of various fine-tuning methods. However, the privacy risks arising from memorization during fine-tuning have received relatively little attention. To address this gap, we categorize popular fine-tuning approaches and assess their impact on memorization through the lens of membership inference attacks (MIAs). Our results show that, compared to parameter-based fine-tuning, prompt-based fine-tuning achieves competitive performance while exhibiting lower vulnerability to MIAs. Furthermore, prompt-based methods maintain low memorization regardless of model scale. These findings suggest that parameter-based fine-tuning is more prone to leaking private information, whereas prompt-based fine-tuning serves as a more privacy-preserving option.
>
---
#### [new 010] NIRANTAR: Continual Learning with New Languages and Domains on Real-world Speech Data
- **分类: cs.CL**

- **简介: 该论文属于语音识别中的持续学习任务，旨在解决多语言、多领域下的模型适应问题。提出Nirantar框架，利用真实世界数据进行评估与基准测试。**

- **链接: [http://arxiv.org/pdf/2507.00534v1](http://arxiv.org/pdf/2507.00534v1)**

> **作者:** Tahir Javed; Kaushal Bhogale; Mitesh M. Khapra
>
> **备注:** Accepted in Interspecch 2025
>
> **摘要:** We introduce Nirantar, a comprehensive framework for evaluating continual learning (CL) in multilingual and multi-domain ASR. Designed to reflect real-world CL challenges, Nirantar leverages data collected incrementally across 22 languages and 208 districts in India through natural episodes. This enables evaluation across Language-Incremental (LIL), Domain-Incremental (DIL), and the novel Language-Incremental Domain-Incremental Learning (LIDIL) scenarios. Unlike prior work that relies on simulated episodes, Nirantar presents dynamic, non-uniform language and domain shifts, making it an ideal testbed for CL research. With 3250 hours of human-transcribed speech, including 1720 hours newly introduced in this work, our framework enables systematic benchmarking of CL methods. We evaluate existing approaches and demonstrate that no single method performs consistently well, underscoring the need for more robust CL strategies.
>
---
#### [new 011] TransLaw: Benchmarking Large Language Models in Multi-Agent Simulation of the Collaborative Translation
- **分类: cs.CL; cs.HC; cs.MA**

- **简介: 该论文属于法律翻译任务，旨在解决 Hong Kong 法律文本翻译中的准确性与风格问题。通过构建多智能体框架 TransLaw，提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2507.00875v1](http://arxiv.org/pdf/2507.00875v1)**

> **作者:** Xi Xuan; King-kui Sin; Yufei Zhou; Chunyu Kit
>
> **备注:** arXiv admin note: text overlap with arXiv:2501.09444; text overlap with arXiv:2409.20288 by other authors
>
> **摘要:** Multi-agent systems empowered by large language models (LLMs) have demonstrated remarkable capabilities in a wide range of downstream applications, including machine translation. However, the potential of LLMs in translating Hong Kong legal judgments remains uncertain due to challenges such as intricate legal terminology, culturally embedded nuances, and strict linguistic structures. In this work, we introduce TransLaw, a novel multi-agent framework implemented for real-world Hong Kong case law translation. It employs three specialized agents, namely, Translator, Annotator, and Proofreader, to collaboratively produce translations for high accuracy in legal meaning, appropriateness in style, and adequate coherence and cohesion in structure. This framework supports customizable LLM configurations and achieves tremendous cost reduction compared to professional human translation services. We evaluated its performance using 13 open-source and commercial LLMs as agents and obtained interesting findings, including that it surpasses GPT-4o in legal semantic accuracy, structural coherence, and stylistic fidelity, yet trails human experts in contextualizing complex terminology and stylistic naturalness. Our platform website is available at CityUHK, and our bilingual judgment corpus used for the evaluation is available at Hugging Face.
>
---
#### [new 012] LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成评估任务，旨在解决创意写作自动化评价难题。提出LitBench基准和数据集，通过模型训练与人类验证提升评价可靠性。**

- **链接: [http://arxiv.org/pdf/2507.00769v1](http://arxiv.org/pdf/2507.00769v1)**

> **作者:** Daniel Fein; Sebastian Russo; Violet Xiang; Kabir Jolly; Rafael Rafailov; Nick Haber
>
> **摘要:** Evaluating creative writing generated by large language models (LLMs) remains challenging because open-ended narratives lack ground truths. Without performant automated evaluation methods, off-the-shelf (OTS) language models are employed as zero-shot judges, yet their reliability is unclear in this context. In pursuit of robust evaluation for creative writing, we introduce LitBench, the first standardized benchmark and paired dataset for creative writing verification, comprising a held-out test set of 2,480 debiased, human-labeled story comparisons drawn from Reddit and a 43,827-pair training corpus of human preference labels. Using LitBench, we (i) benchmark zero-shot LLM judges, (ii) train Bradley Terry and generative reward models, and (iii) conduct an online human study to validate reward model rankings on newly LLM-generated stories. Our benchmark identifies Claude-3.7-Sonnet as the strongest off-the-shelf judge, reaching 73% agreement with human preferences; among trained reward models, Bradley-Terry and Generative reward models both attain an accuracy of 78%, outperforming all off-the-shelf judges. An online human study further confirms that our trained reward models consistently align with human preferences in novel LLM-generated stories. We release LitBench and reward models at https://huggingface.co/collections/SAA-Lab/litbench-68267b5da3aafe58f9e43461, providing a vetted resource for reliable, automated evaluation and optimization of creative writing systems.
>
---
#### [new 013] SciArena: An Open Evaluation Platform for Foundation Models in Scientific Literature Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍SciArena平台，用于评估基础模型在科学文献任务中的表现，解决模型评价缺乏社区参与的问题，通过集体投票和元评估基准提升评价可靠性。**

- **链接: [http://arxiv.org/pdf/2507.01001v1](http://arxiv.org/pdf/2507.01001v1)**

> **作者:** Yilun Zhao; Kaiyan Zhang; Tiansheng Hu; Sihong Wu; Ronan Le Bras; Taira Anderson; Jonathan Bragg; Joseph Chee Chang; Jesse Dodge; Matt Latzke; Yixin Liu; Charles McGrady; Xiangru Tang; Zihang Wang; Chen Zhao; Hannaneh Hajishirzi; Doug Downey; Arman Cohan
>
> **摘要:** We present SciArena, an open and collaborative platform for evaluating foundation models on scientific literature tasks. Unlike traditional benchmarks for scientific literature understanding and synthesis, SciArena engages the research community directly, following the Chatbot Arena evaluation approach of community voting on model comparisons. By leveraging collective intelligence, SciArena offers a community-driven evaluation of model performance on open-ended scientific tasks that demand literature-grounded, long-form responses. The platform currently supports 23 open-source and proprietary foundation models and has collected over 13,000 votes from trusted researchers across diverse scientific domains. We analyze the data collected so far and confirm that the submitted questions are diverse, aligned with real-world literature needs, and that participating researchers demonstrate strong self-consistency and inter-annotator agreement in their evaluations. We discuss the results and insights based on the model ranking leaderboard. To further promote research in building model-based automated evaluation systems for literature tasks, we release SciArena-Eval, a meta-evaluation benchmark based on our collected preference data. The benchmark measures the accuracy of models in judging answer quality by comparing their pairwise assessments with human votes. Our experiments highlight the benchmark's challenges and emphasize the need for more reliable automated evaluation methods.
>
---
#### [new 014] Pitfalls of Evaluating Language Models with Open Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨开放基准测试在评估语言模型时的局限性，通过构建“作弊”模型揭示其潜在问题，并提出需结合私有基准以提高评估可靠性。**

- **链接: [http://arxiv.org/pdf/2507.00460v1](http://arxiv.org/pdf/2507.00460v1)**

> **作者:** Md. Najib Hasan; Mohammad Fakhruddin Babar; Souvika Sarkar; Monowar Hasan; Santu Karmaker
>
> **摘要:** Open Large Language Model (LLM) benchmarks, such as HELM and BIG-bench, offer standardized, transparent protocols that facilitate the fair comparison, reproducibility, and iterative advancement of Language Models (LMs). However, their openness also introduces critical and underexplored pitfalls. This study exposes these weaknesses by systematically constructing ``cheating'' models -- smaller variants of BART, T5, and GPT-2 fine-tuned directly on public test sets -- which achieve top rankings on a prominent open, holistic benchmark (HELM) despite poor generalization and limited practical utility. Our findings underscore three key insights: \ca high leaderboard performance on open benchmarks may not always reflect real-world effectiveness; \cb private or dynamic benchmarks must complement open evaluations to safeguard integrity; and \cc a fundamental reevaluation of current benchmarking practices is essential to ensure robust and trustworthy LM assessments.
>
---
#### [new 015] Towards Style Alignment in Cross-Cultural Translation
- **分类: cs.CL**

- **简介: 该论文属于跨文化翻译任务，旨在解决风格不对齐问题，通过RASTA方法提升模型对文化沟通规范的风格对齐能力。**

- **链接: [http://arxiv.org/pdf/2507.00216v1](http://arxiv.org/pdf/2507.00216v1)**

> **作者:** Shreya Havaldar; Adam Stein; Eric Wong; Lyle Ungar
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Successful communication depends on the speaker's intended style (i.e., what the speaker is trying to convey) aligning with the listener's interpreted style (i.e., what the listener perceives). However, cultural differences often lead to misalignment between the two; for example, politeness is often lost in translation. We characterize the ways that LLMs fail to translate style - biasing translations towards neutrality and performing worse in non-Western languages. We mitigate these failures with RASTA (Retrieval-Augmented STylistic Alignment), a method that leverages learned stylistic concepts to encourage LLM translation to appropriately convey cultural communication norms and align style.
>
---
#### [new 016] Causal Prompting for Implicit Sentiment Analysis with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于隐式情感分析任务，旨在解决LLM在ISA中因因果有效性不足导致的偏差问题。提出CAPITAL框架，通过因果推理提升模型准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.00389v1](http://arxiv.org/pdf/2507.00389v1)**

> **作者:** Jing Ren; Wenhao Zhou; Bowen Li; Mujie Liu; Nguyen Linh Dan Le; Jiade Cen; Liping Chen; Ziqi Xu; Xiwei Xu; Xiaodong Li
>
> **摘要:** Implicit Sentiment Analysis (ISA) aims to infer sentiment that is implied rather than explicitly stated, requiring models to perform deeper reasoning over subtle contextual cues. While recent prompting-based methods using Large Language Models (LLMs) have shown promise in ISA, they often rely on majority voting over chain-of-thought (CoT) reasoning paths without evaluating their causal validity, making them susceptible to internal biases and spurious correlations. To address this challenge, we propose CAPITAL, a causal prompting framework that incorporates front-door adjustment into CoT reasoning. CAPITAL decomposes the overall causal effect into two components: the influence of the input prompt on the reasoning chains, and the impact of those chains on the final output. These components are estimated using encoder-based clustering and the NWGM approximation, with a contrastive learning objective used to better align the encoder's representation with the LLM's reasoning space. Experiments on benchmark ISA datasets with three LLMs demonstrate that CAPITAL consistently outperforms strong prompting baselines in both accuracy and robustness, particularly under adversarial conditions. This work offers a principled approach to integrating causal inference into LLM prompting and highlights its benefits for bias-aware sentiment reasoning. The source code and case study are available at: https://github.com/whZ62/CAPITAL.
>
---
#### [new 017] Discourse Heuristics For Paradoxically Moral Self-Correction
- **分类: cs.CL**

- **简介: 该论文属于AI伦理任务，旨在解决LLM道德自我修正中的悖论问题。通过分析语篇结构，发现其依赖启发式方法，提出利用精选数据集改进修正效果。**

- **链接: [http://arxiv.org/pdf/2507.00985v1](http://arxiv.org/pdf/2507.00985v1)**

> **作者:** Guangliang Liu; Zimo Qi; Xitong Zhang; Kristen Marie Johnson
>
> **摘要:** Moral self-correction has emerged as a promising approach for aligning the output of Large Language Models (LLMs) with human moral values. However, moral self-correction techniques are subject to two primary paradoxes. First, despite empirical and theoretical evidence to support the effectiveness of self-correction, this LLM capability only operates at a superficial level. Second, while LLMs possess the capability of self-diagnosing immoral aspects of their output, they struggle to identify the cause of this moral inconsistency during their self-correction process. To better understand and address these paradoxes, we analyze the discourse constructions in fine-tuning corpora designed to enhance moral self-correction, uncovering the existence of the heuristics underlying effective constructions. We demonstrate that moral self-correction relies on discourse constructions that reflect heuristic shortcuts, and that the presence of these heuristic shortcuts during self-correction leads to inconsistency when attempting to enhance both self-correction and self-diagnosis capabilities jointly. Based on our findings, we propose a solution to improve moral self-correction by leveraging the heuristics of curated datasets. We also highlight the generalization challenges of this capability, particularly in terms of learning from situated context and model scales.
>
---
#### [new 018] Should We Still Pretrain Encoders with Masked Language Modeling?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨MLM与CLM在预训练编码器中的效果，通过实验比较两者优劣，并提出混合训练策略以提升性能。**

- **链接: [http://arxiv.org/pdf/2507.00994v1](http://arxiv.org/pdf/2507.00994v1)**

> **作者:** Hippolyte Gisserot-Boukhlef; Nicolas Boizard; Manuel Faysse; Duarte M. Alves; Emmanuel Malherbe; André F. T. Martins; Céline Hudelot; Pierre Colombo
>
> **备注:** 23 pages, 10 figures, 17 tables
>
> **摘要:** Learning high-quality text representations is fundamental to a wide range of NLP tasks. While encoder pretraining has traditionally relied on Masked Language Modeling (MLM), recent evidence suggests that decoder models pretrained with Causal Language Modeling (CLM) can be effectively repurposed as encoders, often surpassing traditional encoders on text representation benchmarks. However, it remains unclear whether these gains reflect an inherent advantage of the CLM objective or arise from confounding factors such as model and data scale. In this paper, we address this question through a series of large-scale, carefully controlled pretraining ablations, training a total of 30 models ranging from 210 million to 1 billion parameters, and conducting over 15,000 fine-tuning and evaluation runs. We find that while training with MLM generally yields better performance across text representation tasks, CLM-trained models are more data-efficient and demonstrate improved fine-tuning stability. Building on these findings, we experimentally show that a biphasic training strategy that sequentially applies CLM and then MLM, achieves optimal performance under a fixed computational training budget. Moreover, we demonstrate that this strategy becomes more appealing when initializing from readily available pretrained CLM models (from the existing LLM ecosystem), reducing the computational burden needed to train best-in-class encoder models. We release all project artifacts at https://hf.co/MLMvsCLM to foster further research.
>
---
#### [new 019] Linearly Decoding Refused Knowledge in Aligned Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究指令调优模型中被拒绝信息的可解码性。工作包括分析 jailbreak 提示下信息的线性可解码性及其实现机制。**

- **链接: [http://arxiv.org/pdf/2507.00239v1](http://arxiv.org/pdf/2507.00239v1)**

> **作者:** Aryan Shrivastava; Ari Holtzman
>
> **摘要:** Most commonly used language models (LMs) are instruction-tuned and aligned using a combination of fine-tuning and reinforcement learning, causing them to refuse users requests deemed harmful by the model. However, jailbreak prompts can often bypass these refusal mechanisms and elicit harmful responses. In this work, we study the extent to which information accessed via jailbreak prompts is decodable using linear probes trained on LM hidden states. We show that a great deal of initially refused information is linearly decodable. For example, across models, the response of a jailbroken LM for the average IQ of a country can be predicted by a linear probe with Pearson correlations exceeding $0.8$. Surprisingly, we find that probes trained on base models (which do not refuse) sometimes transfer to their instruction-tuned versions and are capable of revealing information that jailbreaks decode generatively, suggesting that the internal representations of many refused properties persist from base LMs through instruction-tuning. Importantly, we show that this information is not merely "leftover" in instruction-tuned models, but is actively used by them: we find that probe-predicted values correlate with LM generated pairwise comparisons, indicating that the information decoded by our probes align with suppressed generative behavior that may be expressed more subtly in other downstream tasks. Overall, our results suggest that instruction-tuning does not wholly eliminate or even relocate harmful information in representation space-they merely suppress its direct expression, leaving it both linearly accessible and indirectly influential in downstream behavior.
>
---
#### [new 020] Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM依赖人工提示的问题。提出MoR框架，通过自适应推理策略提升模型性能，无需外部提示工程。**

- **链接: [http://arxiv.org/pdf/2507.00606v1](http://arxiv.org/pdf/2507.00606v1)**

> **作者:** Tao Xiong; Xavier Hu; Wenyan Fan; Shengyu Zhang
>
> **摘要:** Large language models (LLMs) excel in complex tasks through advanced prompting techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT), but their reliance on manually crafted, task-specific prompts limits adaptability and efficiency. We introduce Mixture of Reasoning (MoR), a training framework that embeds diverse reasoning strategies into LLMs for autonomous, task-adaptive reasoning without external prompt engineering. MoR has two phases: Thought Generation, creating reasoning chain templates with models like GPT-4o, and SFT Dataset Construction, pairing templates with benchmark datasets for supervised fine-tuning.Our experiments show that MoR significantly enhances performance, with MoR150 achieving 0.730 (2.2% improvement) using CoT prompting and 0.734 (13.5% improvement) compared to baselines. MoR eliminates the need for task-specific prompts, offering a generalizable solution for robust reasoning across diverse tasks.
>
---
#### [new 021] MemeCMD: An Automatically Generated Chinese Multi-turn Dialogue Dataset with Contextually Retrieved Memes
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemeCMD，一个自动生成的中文多轮对话数据集，结合上下文检索的梗图。旨在解决对话数据缺乏多模态表达的问题，通过生成对话与合理插入梗图提升对话的生动性与自然度。**

- **链接: [http://arxiv.org/pdf/2507.00891v1](http://arxiv.org/pdf/2507.00891v1)**

> **作者:** Yuheng Wang; Xianhe Tang; Pufeng Huang
>
> **摘要:** Memes are widely used in online social interactions, providing vivid, intuitive, and often humorous means to express intentions and emotions. Existing dialogue datasets are predominantly limited to either manually annotated or pure-text conversations, lacking the expressiveness and contextual nuance that multimodal interactions provide.To address these challenges, we introduce MemeCMD, an automatically generated Chinese Multi-turn Dialogue dataset with contextually retrieved memes. Our dataset combines a large-scale, MLLM-annotated meme library with dialogues auto-generated by dual agents across diverse scenarios. We introduce a retrieval framework and adaptive threshold to ensure contextually relevant, naturally spaced meme usage. Experiments demonstrate the effectiveness of our approach in generating contextually appropriate and diverse meme-incorporated dialogues, offering a scalable and privacy-preserving resource for advancing multimodal conversational AI.
>
---
#### [new 022] EfficientXLang: Towards Improving Token Efficiency Through Cross-Lingual Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升模型的token效率。研究发现非英语语言在推理中更高效且准确，强调多语言基础的重要性。**

- **链接: [http://arxiv.org/pdf/2507.00246v1](http://arxiv.org/pdf/2507.00246v1)**

> **作者:** Sanchit Ahuja; Praneetha Vaddamanu; Barun Patra
>
> **备注:** 15 pages, 5 figures, 9 tables
>
> **摘要:** Despite recent advances in Language Reasoning Models (LRMs), most research focuses solely on English, even though many models are pretrained on multilingual data. In this work, we investigate: Is English the most token-efficient language for reasoning? We evaluate three open-source RLMs: DeepSeek R1, Qwen 2.5 and Qwen 3, across four math datasets and seven typologically diverse languages. We find that reasoning in non-English languages not only reduces token usage, but also preserves accuracy. These gains persist even after translating the reasoning traces into English, suggesting genuine shifts in reasoning behavior rather than surface-level linguistic effects. The extent of improvement, however, depends on the models multilingual strength. Our findings motivate a broader view of reasoning in language models, highlighting the potential of multilingual reasoning and the importance of strong multilingual foundations. The code for our work can be found: https://github.com/microsoft/EfficientXLang.
>
---
#### [new 023] Prompting as Scientific Inquiry
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨如何将提示视为科学方法，解决对大语言模型理解不足的问题，提出 prompting 是行为科学而非权宜之计。**

- **链接: [http://arxiv.org/pdf/2507.00163v1](http://arxiv.org/pdf/2507.00163v1)**

> **作者:** Ari Holtzman; Chenhao Tan
>
> **摘要:** Prompting is the primary method by which we study and control large language models. It is also one of the most powerful: nearly every major capability attributed to LLMs-few-shot learning, chain-of-thought, constitutional AI-was first unlocked through prompting. Yet prompting is rarely treated as science and is frequently frowned upon as alchemy. We argue that this is a category error. If we treat LLMs as a new kind of complex and opaque organism that is trained rather than programmed, then prompting is not a workaround: it is behavioral science. Mechanistic interpretability peers into the neural substrate, prompting probes the model in its native interface: language. We contend that prompting is not inferior, but rather a key component in the science of LLMs.
>
---
#### [new 024] Capsule Network-Based Semantic Intent Modeling for Human-Computer Interaction
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在提升人机交互中的意图识别准确率。通过胶囊网络构建语义意图模型，优化特征提取与分类效果。**

- **链接: [http://arxiv.org/pdf/2507.00540v1](http://arxiv.org/pdf/2507.00540v1)**

> **作者:** Shixiao Wang; Yifan Zhuang; Runsheng Zhang; Zhijun Song
>
> **摘要:** This paper proposes a user semantic intent modeling algorithm based on Capsule Networks to address the problem of insufficient accuracy in intent recognition for human-computer interaction. The method represents semantic features in input text through a vectorized capsule structure. It uses a dynamic routing mechanism to transfer information across multiple capsule layers. This helps capture hierarchical relationships and part-whole structures between semantic entities more effectively. The model uses a convolutional feature extraction module as the low-level encoder. After generating initial semantic capsules, it forms high-level abstract intent representations through an iterative routing process. To further enhance performance, a margin-based mechanism is introduced into the loss function. This improves the model's ability to distinguish between intent classes. Experiments are conducted using a public natural language understanding dataset. Multiple mainstream models are used for comparison. Results show that the proposed model outperforms traditional methods and other deep learning structures in terms of accuracy, F1-score, and intent detection rate. The study also analyzes the effect of the number of dynamic routing iterations on model performance. A convergence curve of the loss function during training is provided. These results verify the stability and effectiveness of the proposed method in semantic modeling. Overall, this study presents a new structured modeling approach to improve intent recognition under complex semantic conditions.
>
---
#### [new 025] AI Analyst: Framework and Comprehensive Evaluation of Large Language Models for Financial Time Series Report Generation
- **分类: cs.CL**

- **简介: 该论文属于金融报告生成任务，旨在解决如何利用大语言模型从时间序列数据中生成准确报告的问题。工作包括框架设计、模型选择与评估，以及信息分类系统构建。**

- **链接: [http://arxiv.org/pdf/2507.00718v1](http://arxiv.org/pdf/2507.00718v1)**

> **作者:** Elizabeth Fons; Elena Kochkina; Rachneet Kaur; Zhen Zeng; Berowne Hlavaty; Charese Smiley; Svitlana Vyetrenko; Manuela Veloso
>
> **摘要:** This paper explores the potential of large language models (LLMs) to generate financial reports from time series data. We propose a framework encompassing prompt engineering, model selection, and evaluation. We introduce an automated highlighting system to categorize information within the generated reports, differentiating between insights derived directly from time series data, stemming from financial reasoning, and those reliant on external knowledge. This approach aids in evaluating the factual grounding and reasoning capabilities of the models. Our experiments, utilizing both data from the real stock market indices and synthetic time series, demonstrate the capability of LLMs to produce coherent and informative financial reports.
>
---
#### [new 026] Contrasting Cognitive Styles in Vision-Language Models: Holistic Attention in Japanese Versus Analytical Focus in English
- **分类: cs.CL**

- **简介: 该论文属于跨文化认知研究任务，探讨VLMs在日语与英语训练下是否表现出不同的注意力模式，揭示语言与文化如何影响模型输出。**

- **链接: [http://arxiv.org/pdf/2507.00700v1](http://arxiv.org/pdf/2507.00700v1)**

> **作者:** Ahmed Sabir; Azinovič Gasper; Mengsay Loem; Rajesh Sharma
>
> **摘要:** Cross-cultural research in perception and cognition has shown that individuals from different cultural backgrounds process visual information in distinct ways. East Asians, for example, tend to adopt a holistic perspective, attending to contextual relationships, whereas Westerners often employ an analytical approach, focusing on individual objects and their attributes. In this study, we investigate whether Vision-Language Models (VLMs) trained predominantly on different languages, specifically Japanese and English, exhibit similar culturally grounded attentional patterns. Using comparative analysis of image descriptions, we examine whether these models reflect differences in holistic versus analytic tendencies. Our findings suggest that VLMs not only internalize the structural properties of language but also reproduce cultural behaviors embedded in the training data, indicating that cultural cognition may implicitly shape model outputs.
>
---
#### [new 027] Question Decomposition for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决多跳问题中信息分散的问题。通过问题分解和重排序提升检索与生成效果。**

- **链接: [http://arxiv.org/pdf/2507.00355v1](http://arxiv.org/pdf/2507.00355v1)**

> **作者:** Paul J. L. Ammann; Jonas Golde; Alan Akbik
>
> **备注:** Accepted to ACL SRW 2025. 9 Pages, 2 Figures, 4 Tables
>
> **摘要:** Grounding large language models (LLMs) in verifiable external sources is a well-established strategy for generating reliable answers. Retrieval-augmented generation (RAG) is one such approach, particularly effective for tasks like question answering: it retrieves passages that are semantically related to the question and then conditions the model on this evidence. However, multi-hop questions, such as "Which company among NVIDIA, Apple, and Google made the biggest profit in 2023?," challenge RAG because relevant facts are often distributed across multiple documents rather than co-occurring in one source, making it difficult for standard RAG to retrieve sufficient information. To address this, we propose a RAG pipeline that incorporates question decomposition: (i) an LLM decomposes the original query into sub-questions, (ii) passages are retrieved for each sub-question, and (iii) the merged candidate pool is reranked to improve the coverage and precision of the retrieved evidence. We show that question decomposition effectively assembles complementary documents, while reranking reduces noise and promotes the most relevant passages before answer generation. Although reranking itself is standard, we show that pairing an off-the-shelf cross-encoder reranker with LLM-driven question decomposition bridges the retrieval gap on multi-hop questions and provides a practical, drop-in enhancement, without any extra training or specialized indexing. We evaluate our approach on the MultiHop-RAG and HotpotQA, showing gains in retrieval (MRR@10: +36.7%) and answer accuracy (F1: +11.6%) over standard RAG baselines.
>
---
#### [new 028] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America
- **分类: cs.CL**

- **简介: 该论文提出La Leaderboard，用于评估西班牙语及其变体的大型语言模型，解决多样性与可访问性问题，整合66个数据集并评测50个模型。**

- **链接: [http://arxiv.org/pdf/2507.00999v1](http://arxiv.org/pdf/2507.00999v1)**

> **作者:** María Grandury; Javier Aula-Blasco; Júlia Falcão; Clémentine Fourrier; Miguel González; Gonzalo Martínez; Gonzalo Santamaría; Rodrigo Agerri; Nuria Aldama; Luis Chiruzzo; Javier Conde; Helena Gómez; Marta Guerrero; Guido Ivetta; Natalia López; Flor Miriam Plaza-del-Arco; María Teresa Martín-Valdivia; Helena Montoro; Carmen Muñoz; Pedro Reviriego; Leire Rosado; Alejandro Vaca; María Estrella Vallecillo-Rodríguez; Jorge Vallego; Irune Zubiaga
>
> **备注:** Accepted at ACL 2025 Main
>
> **摘要:** Leaderboards showcase the current capabilities and limitations of Large Language Models (LLMs). To motivate the development of LLMs that represent the linguistic and cultural diversity of the Spanish-speaking community, we present La Leaderboard, the first open-source leaderboard to evaluate generative LLMs in languages and language varieties of Spain and Latin America. La Leaderboard is a community-driven project that aims to establish an evaluation standard for everyone interested in developing LLMs for the Spanish-speaking community. This initial version combines 66 datasets in Basque, Catalan, Galician, and different Spanish varieties, showcasing the evaluation results of 50 models. To encourage community-driven development of leaderboards in other languages, we explain our methodology, including guidance on selecting the most suitable evaluation setup for each downstream task. In particular, we provide a rationale for using fewer few-shot examples than typically found in the literature, aiming to reduce environmental impact and facilitate access to reproducible results for a broader research community.
>
---
#### [new 029] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering
- **分类: cs.CL**

- **简介: 该论文属于文本分析任务，旨在解决主题模型和文档聚类评估难题。通过设计可扩展的人工评估协议及自动化替代方案，提升评估的有效性与可行性。**

- **链接: [http://arxiv.org/pdf/2507.00828v1](http://arxiv.org/pdf/2507.00828v1)**

> **作者:** Alexander Hoyle; Lorena Calvo-Bartolomé; Jordan Boyd-Graber; Philip Resnik
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Topic model and document-clustering evaluations either use automated metrics that align poorly with human preferences or require expert labels that are intractable to scale. We design a scalable human evaluation protocol and a corresponding automated approximation that reflect practitioners' real-world usage of models. Annotators -- or an LLM-based proxy -- review text items assigned to a topic or cluster, infer a category for the group, then apply that category to other documents. Using this protocol, we collect extensive crowdworker annotations of outputs from a diverse set of topic models on two datasets. We then use these annotations to validate automated proxies, finding that the best LLM proxies are statistically indistinguishable from a human annotator and can therefore serve as a reasonable substitute in automated evaluations. Package, web interface, and data are at https://github.com/ahoho/proxann
>
---
#### [new 030] SAFER: Probing Safety in Reward Models with Sparse Autoencoder
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，旨在提升奖励模型的安全性。通过SAFER框架，解析奖励模型激活中的可解释特征，并设计数据策略优化安全对齐。**

- **链接: [http://arxiv.org/pdf/2507.00665v1](http://arxiv.org/pdf/2507.00665v1)**

> **作者:** Sihang Li; Wei Shi; Ziyuan Xie; Tao Liang; Guojun Ma; Xiang Wang
>
> **摘要:** Reinforcement learning from human feedback (RLHF) is a key paradigm for aligning large language models (LLMs) with human values, yet the reward models at its core remain largely opaque. In this work, we present sparse Autoencoder For Enhanced Reward model (\textbf{SAFER}), a novel framework for interpreting and improving reward models through mechanistic analysis. Leveraging Sparse Autoencoders (SAEs), we uncover human-interpretable features in reward model activations, enabling insight into safety-relevant decision-making. We apply SAFER to safety-oriented preference datasets and quantify the salience of individual features by activation differences between chosen and rejected responses. Using these feature-level signals, we design targeted data poisoning and denoising strategies. Experiments show that SAFER can precisely degrade or enhance safety alignment with minimal data modification, without sacrificing general chat performance. Our approach contributes to interpreting, auditing and refining reward models in high-stakes LLM alignment tasks. Our codes are available at https://github.com/xzy-101/SAFER-code. \textit{This paper discusses topics related to large language model safety and may include discussions or examples that highlight potential risks or unsafe outcomes.}
>
---
#### [new 031] Stylometry recognizes human and LLM-generated texts in short samples
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分类任务，旨在区分人类与LLM生成的文本。通过构建数据集并应用stylometry方法，使用树模型进行分类，准确率达98%，验证了区分的可能性。**

- **链接: [http://arxiv.org/pdf/2507.00838v1](http://arxiv.org/pdf/2507.00838v1)**

> **作者:** Karol Przystalski; Jan K. Argasiński; Iwona Grabska-Gradzińska; Jeremi K. Ochab
>
> **摘要:** The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type.
>
---
#### [new 032] Beyond Sociodemographic Prompting: Using Supervision to Align LLMs with Human Response Distributions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型对不同人群回答分布的对齐能力。通过简单监督方法改善模型与多样群体的匹配度。**

- **链接: [http://arxiv.org/pdf/2507.00439v1](http://arxiv.org/pdf/2507.00439v1)**

> **作者:** Gauri Kambhatla; Sanjana Gautam; Angela Zhang; Alex Liu; Ravi Srinivasan; Junyi Jessy Li; Matthew Lease
>
> **摘要:** The ability to accurately predict how different population groups would answer subjective questions would have great value. In this work, we show that use of relatively simple supervision can greatly improve language model alignment with diverse population groups, as measured over three datasets spanning various topics. Beyond evaluating average performance, we also report how alignment varies across specific groups. The simplicity and generality of our approach promotes easy adoption, while our broad findings provide useful guidance for when to use or not use our approach in practice. By conducting evaluation over many LLMs and prompting strategies, along with open-sourcing our work, we provide a useful benchmark to stimulate future research.
>
---
#### [new 033] Gregorian melody, modality, and memory: Segmenting chant with Bayesian nonparametrics
- **分类: cs.CL**

- **简介: 该论文属于音乐信息检索任务，旨在通过贝叶斯非参数方法对格里高利圣咏进行分割，解决其结构与记忆关系问题。**

- **链接: [http://arxiv.org/pdf/2507.00380v1](http://arxiv.org/pdf/2507.00380v1)**

> **作者:** Vojtěch Lanz; Jan Hajič jr
>
> **摘要:** The idea that Gregorian melodies are constructed from some vocabulary of segments has long been a part of chant scholarship. This so-called "centonisation" theory has received much musicological criticism, but frequent re-use of certain melodic segments has been observed in chant melodies, and the intractable number of possible segmentations allowed the option that some undiscovered segmentation exists that will yet prove the value of centonisation, and recent empirical results have shown that segmentations can outperform music-theoretical features in mode classification. Inspired by the fact that Gregorian chant was memorised, we search for an optimal unsupervised segmentation of chant melody using nested hierarchical Pitman-Yor language models. The segmentation we find achieves state-of-the-art performance in mode classification. Modeling a monk memorising the melodies from one liturgical manuscript, we then find empirical evidence for the link between mode classification and memory efficiency, and observe more formulaic areas at the beginnings and ends of melodies corresponding to the practical role of modality in performance. However, the resulting segmentations themselves indicate that even such a memory-optimal segmentation is not what is understood as centonisation.
>
---
#### [new 034] Generative AI and the future of scientometrics: current topics and future questions
- **分类: cs.CL; cs.DL**

- **简介: 本文探讨生成式AI在科学计量学中的应用，分析其在语言生成任务中的潜力与局限，旨在推动该领域对AI影响的深入思考。**

- **链接: [http://arxiv.org/pdf/2507.00783v1](http://arxiv.org/pdf/2507.00783v1)**

> **作者:** Benedetto Lepori; Jens Peter Andersen; Karsten Donnay
>
> **摘要:** The aim of this paper is to review the use of GenAI in scientometrics, and to begin a debate on the broader implications for the field. First, we provide an introduction on GenAI's generative and probabilistic nature as rooted in distributional linguistics. And we relate this to the debate on the extent to which GenAI might be able to mimic human 'reasoning'. Second, we leverage this distinction for a critical engagement with recent experiments using GenAI in scientometrics, including topic labelling, the analysis of citation contexts, predictive applications, scholars' profiling, and research assessment. GenAI shows promise in tasks where language generation dominates, such as labelling, but faces limitations in tasks that require stable semantics, pragmatic reasoning, or structured domain knowledge. However, these results might become quickly outdated. Our recommendation is, therefore, to always strive to systematically compare the performance of different GenAI models for specific tasks. Third, we inquire whether, by generating large amounts of scientific language, GenAI might have a fundamental impact on our field by affecting textual characteristics used to measure science, such as authors, words, and references. We argue that careful empirical work and theoretical reflection will be essential to remain capable of interpreting the evolving patterns of knowledge production.
>
---
#### [new 035] TeamCMU at Touché: Adversarial Co-Evolution for Advertisement Integration and Detection in Conversational Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话搜索中的广告集成与检测任务，旨在解决生成系统中广告透明度和用户体验问题。工作包括设计广告重写器和分类器，并通过合成数据训练提升广告隐蔽性。**

- **链接: [http://arxiv.org/pdf/2507.00509v1](http://arxiv.org/pdf/2507.00509v1)**

> **作者:** To Eun Kim; João Coelho; Gbemileke Onilude; Jai Singh
>
> **摘要:** As conversational search engines increasingly adopt generation-based paradigms powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), the integration of advertisements into generated responses presents both commercial opportunities and challenges for user experience. Unlike traditional search, where advertisements are clearly delineated, generative systems blur the boundary between informational content and promotional material, raising concerns around transparency and trust. In this work, we propose a modular pipeline for advertisement management in RAG-based conversational systems, consisting of an ad-rewriter for seamless ad integration and a robust ad-classifier for detection. We leverage synthetic data to train high-performing classifiers, which are then used to guide two complementary ad-integration strategies: supervised fine-tuning of the ad-rewriter and a best-of-N sampling approach that selects the least detectable ad-integrated response among multiple candidates. Our evaluation focuses on two core questions: the effectiveness of ad classifiers in detecting diverse ad integration strategies, and the training methods that best support coherent, minimally intrusive ad insertion. Experimental results show that our ad-classifier, trained on synthetic advertisement data inspired by marketing strategies and enhanced through curriculum learning, achieves robust detection performance. Additionally, we demonstrate that classifier-guided optimization, through both fine-tuning and best-of-N sampling, significantly improves ad stealth, enabling more seamless integration. These findings contribute an adversarial co-evolution framework for developing more sophisticated ad-aware generative search systems and robust ad classifiers.
>
---
#### [new 036] Modeling Data Diversity for Joint Instance and Verbalizer Selection in Cold-Start Scenarios
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于冷启动场景下的文本分类任务，旨在解决模板、verbalizer和实例选择敏感的问题。提出COLDSELECT方法，联合选择verbalizer和实例，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00330v1](http://arxiv.org/pdf/2507.00330v1)**

> **作者:** Mohna Chakraborty; Adithya Kulkarni; Qi Li
>
> **摘要:** Prompt-based methods leverage the knowledge of pre-trained language models (PLMs) trained with a masked language modeling (MLM) objective; however, these methods are sensitive to template, verbalizer, and few-shot instance selection, particularly in cold-start settings with no labeled data. Existing studies overlook the dependency between instances and verbalizers, where instance-label probabilities depend on verbalizer token proximity in the embedding space. To address this, we propose COLDSELECT, a joint verbalizer and instance selection approach that models data diversity. COLDSELECT maps PLM vocabulary and $h_{[MASK]}$ embeddings into a shared space, applying dimensionality reduction and clustering to ensure efficient and diverse selection. By optimizing for minimal uncertainty and maximal diversity, COLDSELECT captures data relationships effectively. Experiments on eight benchmarks demonstrate COLDSELECT's superiority in reducing uncertainty and enhancing generalization, outperforming baselines in verbalizer and few-shot instance selection for cold-start scenarios.
>
---
#### [new 037] The Algebraic Structure of Morphosyntax
- **分类: cs.CL; math.QA; 91F20, 18M60, 68Q70**

- **简介: 该论文属于语言学理论任务，旨在建立形态与句法接口的数学模型，解决形态结构如何与句法结合的问题，通过代数结构和操作符理论进行形式化描述。**

- **链接: [http://arxiv.org/pdf/2507.00244v1](http://arxiv.org/pdf/2507.00244v1)**

> **作者:** Isabella Senturia; Matilde Marcolli
>
> **备注:** 45 pages, LaTeX, 2 png figures
>
> **摘要:** Within the context of the mathematical formulation of Merge and the Strong Minimalist Thesis, we present a mathematical model of the morphology-syntax interface. In this setting, morphology has compositional properties responsible for word formation, organized into a magma of morphological trees. However, unlike syntax, we do not have movement within morphology. A coproduct decomposition exists, but it requires extending the set of morphological trees beyond those which are generated solely by the magma, to a larger set of possible morphological inputs to syntactic trees. These participate in the formation of morphosyntactic trees as an algebra over an operad, and a correspondence between algebras over an operad. The process of structure formation for morphosyntactic trees can then be described in terms of this operadic correspondence that pairs syntactic and morphological data and the morphology coproduct. We reinterpret in this setting certain operations of Distributed Morphology as transformation that allow for flexibility in moving the boundary between syntax and morphology within the morphosyntactic objects.
>
---
#### [new 038] The Cognate Data Bottleneck in Language Phylogenetics
- **分类: cs.CL; q-bio.PE**

- **简介: 该论文属于语言系统发育任务，旨在解决认知数据不足的问题。研究发现现有数据量不足以支持计算方法，尝试从BabelNet提取数据但效果不佳，表明该领域仍面临挑战。**

- **链接: [http://arxiv.org/pdf/2507.00911v1](http://arxiv.org/pdf/2507.00911v1)**

> **作者:** Luise Häuser; Alexandros Stamatakis
>
> **摘要:** To fully exploit the potential of computational phylogenetic methods for cognate data one needs to leverage specific (complex) models an machine learning-based techniques. However, both approaches require datasets that are substantially larger than the manually collected cognate data currently available. To the best of our knowledge, there exists no feasible approach to automatically generate larger cognate datasets. We substantiate this claim by automatically extracting datasets from BabelNet, a large multilingual encyclopedic dictionary. We demonstrate that phylogenetic inferences on the respective character matrices yield trees that are largely inconsistent with the established gold standard ground truth trees. We also discuss why we consider it as being unlikely to be able to extract more suitable character matrices from other multilingual resources. Phylogenetic data analysis approaches that require larger datasets can therefore not be applied to cognate data. Thus, it remains an open question how, and if these computational approaches can be applied in historical linguistics.
>
---
#### [new 039] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data
- **分类: cs.CL**

- **简介: 该论文研究表格理解任务，探讨LLMs在科学与非科学表格上的表现及跨模态处理能力。**

- **链接: [http://arxiv.org/pdf/2507.00152v1](http://arxiv.org/pdf/2507.00152v1)**

> **作者:** Ekaterina Borisova; Fabio Barth; Nils Feldhus; Raia Abu Ahmad; Malte Ostendorff; Pedro Ortiz Suarez; Georg Rehm; Sebastian Möller
>
> **备注:** TRL@ACL 2025, camera-ready version
>
> **摘要:** Tables are among the most widely used tools for representing structured data in research, business, medicine, and education. Although LLMs demonstrate strong performance in downstream tasks, their efficiency in processing tabular data remains underexplored. In this paper, we investigate the effectiveness of both text-based and multimodal LLMs on table understanding tasks through a cross-domain and cross-modality evaluation. Specifically, we compare their performance on tables from scientific vs. non-scientific contexts and examine their robustness on tables represented as images vs. text. Additionally, we conduct an interpretability analysis to measure context usage and input relevance. We also introduce the TableEval benchmark, comprising 3017 tables from scholarly publications, Wikipedia, and financial reports, where each table is provided in five different formats: Image, Dictionary, HTML, XML, and LaTeX. Our findings indicate that while LLMs maintain robustness across table modalities, they face significant challenges when processing scientific tables.
>
---
#### [new 040] Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于机器学习领域，探讨下游任务中缩放定律的可靠性。研究指出缩放定律在多数情况下不成立，提出需理解其适用条件。**

- **链接: [http://arxiv.org/pdf/2507.00885v1](http://arxiv.org/pdf/2507.00885v1)**

> **作者:** Nicholas Lourie; Michael Y. Hu; Kyunghyun Cho
>
> **摘要:** Downstream scaling laws aim to predict task performance at larger scales from pretraining losses at smaller scales. Whether this prediction should be possible is unclear: some works demonstrate that task performance follows clear linear scaling trends under transformation, whereas others point out fundamental challenges to downstream scaling laws, such as emergence and inverse scaling. In this work, we conduct a meta-analysis of existing data on downstream scaling laws, finding that close fit to linear scaling laws only occurs in a minority of cases: 39% of the time. Furthermore, seemingly benign changes to the experimental setting can completely change the scaling trend. Our analysis underscores the need to understand the conditions under which scaling laws succeed. To fully model the relationship between pretraining loss and downstream task performance, we must embrace the cases in which scaling behavior deviates from linear trends.
>
---
#### [new 041] Failure by Interference: Language Models Make Balanced Parentheses Errors When Faulty Mechanisms Overshadow Sound Ones
- **分类: cs.CL; cs.AI; cs.SE; I.2.7**

- **简介: 该论文研究语言模型在生成平衡括号时的错误，属于自然语言处理中的语法任务。针对模型内部机制失效问题，提出RASteer方法提升准确性。**

- **链接: [http://arxiv.org/pdf/2507.00322v1](http://arxiv.org/pdf/2507.00322v1)**

> **作者:** Daking Rai; Samuel Miller; Kevin Moran; Ziyu Yao
>
> **备注:** 23 pages, 10 figures, Preprint
>
> **摘要:** Despite remarkable advances in coding capabilities, language models (LMs) still struggle with simple syntactic tasks such as generating balanced parentheses. In this study, we investigate the underlying mechanisms behind the persistence of these errors across LMs of varying sizes (124M-7B) to both understand and mitigate the errors. Our study reveals that LMs rely on a number of components (attention heads and FF neurons) that independently make their own predictions. While some components reliably promote correct answers across a generalized range of inputs (i.e., implementing "sound mechanisms''), others are less reliable and introduce noise by promoting incorrect tokens (i.e., implementing "faulty mechanisms''). Errors occur when the faulty mechanisms overshadow the sound ones and dominantly affect the predictions. Motivated by this insight, we introduce RASteer, a steering method to systematically identify and increase the contribution of reliable components for improving model performance. RASteer substantially improves performance on balanced parentheses tasks, boosting accuracy of some models from $0$% to around $100$% without impairing the models' general coding ability. We further demonstrate its broader applicability in arithmetic reasoning tasks, achieving performance gains of up to around $20$%.
>
---
#### [new 042] TUM-MiKaNi at SemEval-2025 Task 3: Towards Multilingual and Knowledge-Aware Non-factual Hallucination Identification
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与SemEval-2025 Task-3，旨在解决多语言非事实幻觉识别问题。提出结合检索和BERT的系统，提升LLM输出准确性。**

- **链接: [http://arxiv.org/pdf/2507.00579v1](http://arxiv.org/pdf/2507.00579v1)**

> **作者:** Miriam Anschütz; Ekaterina Gikalo; Niklas Herbster; Georg Groh
>
> **备注:** 6 pages, 3 figures, SemEval-2025 Task 3, ACL
>
> **摘要:** Hallucinations are one of the major problems of LLMs, hindering their trustworthiness and deployment to wider use cases. However, most of the research on hallucinations focuses on English data, neglecting the multilingual nature of LLMs. This paper describes our submission to the SemEval-2025 Task-3 - Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. We propose a two-part pipeline that combines retrieval-based fact verification against Wikipedia with a BERT-based system fine-tuned to identify common hallucination patterns. Our system achieves competitive results across all languages, reaching top-10 results in eight languages, including English. Moreover, it supports multiple languages beyond the fourteen covered by the shared task. This multilingual hallucination identifier can help to improve LLM outputs and their usefulness in the future.
>
---
#### [new 043] Beat and Downbeat Tracking in Performance MIDI Using an End-to-End Transformer Architecture
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于音乐节奏分析任务，解决性能MIDI中的节拍与强拍跟踪问题。提出基于Transformer的端到端模型，提升准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00466v1](http://arxiv.org/pdf/2507.00466v1)**

> **作者:** Sebastian Murgul; Michael Heizmann
>
> **备注:** Accepted to the 22nd Sound and Music Computing Conference (SMC), 2025
>
> **摘要:** Beat tracking in musical performance MIDI is a challenging and important task for notation-level music transcription and rhythmical analysis, yet existing methods primarily focus on audio-based approaches. This paper proposes an end-to-end transformer-based model for beat and downbeat tracking in performance MIDI, leveraging an encoder-decoder architecture for sequence-to-sequence translation of MIDI input to beat annotations. Our approach introduces novel data preprocessing techniques, including dynamic augmentation and optimized tokenization strategies, to improve accuracy and generalizability across different datasets. We conduct extensive experiments using the A-MAPS, ASAP, GuitarSet, and Leduc datasets, comparing our model against state-of-the-art hidden Markov models (HMMs) and deep learning-based beat tracking methods. The results demonstrate that our model outperforms existing symbolic music beat tracking approaches, achieving competitive F1-scores across various musical styles and instruments. Our findings highlight the potential of transformer architectures for symbolic beat tracking and suggest future integration with automatic music transcription systems for enhanced music analysis and score generation.
>
---
#### [new 044] State and Memory is All You Need for Robust and Reliable AI Agents
- **分类: cs.MA; cs.AI; cs.CL; cs.ET; physics.chem-ph**

- **简介: 该论文属于AI代理任务，旨在解决复杂科学工作流中的记忆、规划和工具集成问题。提出SciBORG框架，通过状态和记忆增强实现可靠的任务执行与决策。**

- **链接: [http://arxiv.org/pdf/2507.00081v1](http://arxiv.org/pdf/2507.00081v1)**

> **作者:** Matthew Muhoberac; Atharva Parikh; Nirvi Vakharia; Saniya Virani; Aco Radujevic; Savannah Wood; Meghav Verma; Dimitri Metaxotos; Jeyaraman Soundararajan; Thierry Masquelin; Alexander G. Godfrey; Sean Gardner; Dobrila Rudnicki; Sam Michael; Gaurav Chopra
>
> **备注:** 5 Main Figures, 10 Extended Data Figures (37 Pages) for Manuscript ; 9 Supplementary Tables, 40 Supplementary Figures (180 Pages) for Supporting Information
>
> **摘要:** Large language models (LLMs) have enabled powerful advances in natural language understanding and generation. Yet their application to complex, real-world scientific workflows remain limited by challenges in memory, planning, and tool integration. Here, we introduce SciBORG (Scientific Bespoke Artificial Intelligence Agents Optimized for Research Goals), a modular agentic framework that allows LLM-based agents to autonomously plan, reason, and achieve robust and reliable domain-specific task execution. Agents are constructed dynamically from source code documentation and augmented with finite-state automata (FSA) memory, enabling persistent state tracking and context-aware decision-making. This approach eliminates the need for manual prompt engineering and allows for robust, scalable deployment across diverse applications via maintaining context across extended workflows and to recover from tool or execution failures. We validate SciBORG through integration with both physical and virtual hardware, such as microwave synthesizers for executing user-specified reactions, with context-aware decision making and demonstrate its use in autonomous multi-step bioassay retrieval from the PubChem database utilizing multi-step planning, reasoning, agent-to-agent communication and coordination for execution of exploratory tasks. Systematic benchmarking shows that SciBORG agents achieve reliable execution, adaptive planning, and interpretable state transitions. Our results show that memory and state awareness are critical enablers of agentic planning and reliability, offering a generalizable foundation for deploying AI agents in complex environments.
>
---
#### [new 045] ASTRO: Teaching Language Models to Reason by Reflecting and Backtracking In-Context
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ASTRO框架，旨在提升语言模型的推理能力。通过模拟搜索过程，增强模型的自我反思和回溯能力，解决非推理模型推理不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.00417v1](http://arxiv.org/pdf/2507.00417v1)**

> **作者:** Joongwon Kim; Anirudh Goyal; Liang Tan; Hannaneh Hajishirzi; Srinivasan Iyer; Tianlu Wang
>
> **备注:** 36 pages, 23 figures
>
> **摘要:** We introduce ASTRO, the "Autoregressive Search-Taught Reasoner", a framework for training language models to reason like search algorithms, explicitly leveraging self-reflection, backtracking, and exploration in their outputs. Recently, training large language models (LLMs) via reinforcement learning (RL) has led to the advent of reasoning models with greatly enhanced reasoning capabilities. Open-source replications of reasoning models, while successful, build upon models that already exhibit strong reasoning capabilities along with search behavior observed even before RL. As a result, it is yet unclear how to boost the reasoning capabilities of other non-reasoner models including Llama 3. ASTRO teaches such models to internalize structured search behavior through a synthetic dataset derived from Monte Carlo Tree Search (MCTS) over mathematical problem-solving trajectories. By converting search traces into natural language chain-of-thoughts that capture both successes and recoveries from failure, ASTRO bootstraps models with a rich prior for exploration during RL. We finetune our models on these search-derived traces and further improve performance via RL with verifiable rewards. We apply ASTRO to the Llama 3 family of models and achieve absolute performance gains of 16.0% on MATH-500, 26.9% on AMC 2023, and 20.0% on AIME 2024, especially improving upon challenging problems that require iterative correction. Our results demonstrate that search-inspired training offers a principled way to instill robust reasoning capabilities into open LLMs.
>
---
#### [new 046] Multi-interaction TTS toward professional recording reproduction
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决合成语音风格难以精细调整的问题。通过多轮交互机制，允许用户逐步优化语音风格。**

- **链接: [http://arxiv.org/pdf/2507.00808v1](http://arxiv.org/pdf/2507.00808v1)**

> **作者:** Hiroki Kanagawa; Kenichi Fujita; Aya Watanabe; Yusuke Ijima
>
> **备注:** 7 pages,6 figures, Accepted to Speech Synthesis Workshop 2025 (SSW13)
>
> **摘要:** Voice directors often iteratively refine voice actors' performances by providing feedback to achieve the desired outcome. While this iterative feedback-based refinement process is important in actual recordings, it has been overlooked in text-to-speech synthesis (TTS). As a result, fine-grained style refinement after the initial synthesis is not possible, even though the synthesized speech often deviates from the user's intended style. To address this issue, we propose a TTS method with multi-step interaction that allows users to intuitively and rapidly refine synthetized speech. Our approach models the interaction between the TTS model and its user to emulate the relationship between voice actors and voice directors. Experiments show that the proposed model with its corresponding dataset enable iterative style refinements in accordance with users' directions, thus demonstrating its multi-interaction capability. Sample audios are available: https://ntt-hilab-gensp. github.io/ssw13multiinteraction_tts/
>
---
#### [new 047] Verifiable Natural Language to Linear Temporal Logic Translation: A Benchmark Dataset and Evaluation Suite
- **分类: eess.SY; cs.CL; cs.SY**

- **简介: 该论文属于自然语言到线性时序逻辑的翻译任务，旨在解决现有基准无法验证翻译结果的问题。工作包括构建VLTL-Bench数据集，支持端到端评估与子步骤分析。**

- **链接: [http://arxiv.org/pdf/2507.00877v1](http://arxiv.org/pdf/2507.00877v1)**

> **作者:** William H English; Chase Walker; Dominic Simon; Sumit Kumar Jha; Rickard Ewetz
>
> **摘要:** Empirical evaluation of state-of-the-art natural-language (NL) to temporal-logic (TL) translation systems reveals near-perfect performance on existing benchmarks. However, current studies measure only the accuracy of the translation of NL logic into formal TL, ignoring a system's capacity to ground atomic propositions into new scenarios or environments. This is a critical feature, necessary for the verification of resulting formulas in a concrete state space. Consequently, most NL-to-TL translation frameworks propose their own bespoke dataset in which the correct grounding is known a-priori, inflating performance metrics and neglecting the need for extensible, domain-general systems. In this paper, we introduce the Verifiable Linear Temporal Logic Benchmark ( VLTL-Bench), a unifying benchmark that measures verification and verifiability of automated NL-to-LTL translation. The dataset consists of three unique state spaces and thousands of diverse natural language specifications and corresponding formal specifications in temporal logic. Moreover, the benchmark contains sample traces to validate the temporal logic expressions. While the benchmark directly supports end-to-end evaluation, we observe that many frameworks decompose the process into i) lifting, ii) grounding, iii) translation, and iv) verification. The benchmark provides ground truths after each of these steps to enable researches to improve and evaluate different substeps of the overall problem. To encourage methodologically sound advances in verifiable NL-to-LTL translation approaches, we release VLTL-Bench here: https://www.kaggle.com/datasets/dubascudes/vltl bench.
>
---
#### [new 048] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决模型幻觉问题。提出ONLY方法，通过单次查询和一层干预有效减少幻觉，提升可靠性与效率。**

- **链接: [http://arxiv.org/pdf/2507.00898v1](http://arxiv.org/pdf/2507.00898v1)**

> **作者:** Zifu Wan; Ce Zhang; Silong Yong; Martin Q. Ma; Simon Stepputtis; Louis-Philippe Morency; Deva Ramanan; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted by ICCV 2025. Project page: https://zifuwan.github.io/ONLY/
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) have introduced a new paradigm for understanding and reasoning about image input through textual responses. Although they have achieved remarkable performance across a range of multi-modal tasks, they face the persistent challenge of hallucination, which introduces practical weaknesses and raises concerns about their reliable deployment in real-world applications. Existing work has explored contrastive decoding approaches to mitigate this issue, where the output of the original LVLM is compared and contrasted with that of a perturbed version. However, these methods require two or more queries that slow down LVLM response generation, making them less suitable for real-time applications. To overcome this limitation, we propose ONLY, a training-free decoding approach that requires only a single query and a one-layer intervention during decoding, enabling efficient real-time deployment. Specifically, we enhance textual outputs by selectively amplifying crucial textual information using a text-to-visual entropy ratio for each token. Extensive experimental results demonstrate that our proposed ONLY consistently outperforms state-of-the-art methods across various benchmarks while requiring minimal implementation effort and computational cost. Code is available at https://github.com/zifuwan/ONLY.
>
---
#### [new 049] Moment Sampling in Video LLMs for Long-Form Video QA
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频问答任务，旨在解决长视频中关键帧丢失和冗余问题。通过引入时刻采样方法，提升视频大模型的长文本问答性能。**

- **链接: [http://arxiv.org/pdf/2507.00033v1](http://arxiv.org/pdf/2507.00033v1)**

> **作者:** Mustafa Chasmai; Gauri Jagatap; Gouthaman KV; Grant Van Horn; Subhransu Maji; Andrea Fanelli
>
> **备注:** Workshop on Video Large Language Models (VidLLMs) at CVPR 2025
>
> **摘要:** Recent advancements in video large language models (Video LLMs) have significantly advanced the field of video question answering (VideoQA). While existing methods perform well on short videos, they often struggle with long-range reasoning in longer videos. To scale Video LLMs for longer video content, frame sub-sampling (selecting frames at regular intervals) is commonly used. However, this approach is suboptimal, often leading to the loss of crucial frames or the inclusion of redundant information from multiple similar frames. Missing key frames impairs the model's ability to answer questions accurately, while redundant frames lead the model to focus on irrelevant video segments and increase computational resource consumption. In this paper, we investigate the use of a general-purpose text-to-video moment retrieval model to guide the frame sampling process. We propose "moment sampling", a novel, model-agnostic approach that enables the model to select the most relevant frames according to the context of the question. Specifically, we employ a lightweight moment retrieval model to prioritize frame selection. By focusing on the frames most pertinent to the given question, our method enhances long-form VideoQA performance in Video LLMs. Through extensive experiments on four long-form VideoQA datasets, using four state-of-the-art Video LLMs, we demonstrate the effectiveness of the proposed approach.
>
---
#### [new 050] Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨数学推理对大模型通用能力的影响。研究发现数学优化难以迁移到其他任务，提出需重新考虑训练方法。**

- **链接: [http://arxiv.org/pdf/2507.00432v1](http://arxiv.org/pdf/2507.00432v1)**

> **作者:** Maggie Huan; Yuetai Li; Tuney Zheng; Xiaoyu Xu; Seungone Kim; Minxin Du; Radha Poovendran; Graham Neubig; Xiang Yue
>
> **摘要:** Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.
>
---
#### [new 051] Developing Lightweight DNN Models With Limited Data For Real-Time Sign Language Recognition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于实时手语识别任务，解决数据稀缺、计算成本高和帧率不一致问题。通过轻量DNN和MediaPipe实现高效准确识别。**

- **链接: [http://arxiv.org/pdf/2507.00248v1](http://arxiv.org/pdf/2507.00248v1)**

> **作者:** Nikita Nikitin; Eugene Fomin
>
> **备注:** 7 pages, 2 figures, 2 tables, for associated mpeg file, see https://slait.app/static/Screen_Recording.mp4
>
> **摘要:** We present a novel framework for real-time sign language recognition using lightweight DNNs trained on limited data. Our system addresses key challenges in sign language recognition, including data scarcity, high computational costs, and discrepancies in frame rates between training and inference environments. By encoding sign language specific parameters, such as handshape, palm orientation, movement, and location into vectorized inputs, and leveraging MediaPipe for landmark extraction, we achieve highly separable input data representations. Our DNN architecture, optimized for sub 10MB deployment, enables accurate classification of 343 signs with less than 10ms latency on edge devices. The data annotation platform 'slait data' facilitates structured labeling and vector extraction. Our model achieved 92% accuracy in isolated sign recognition and has been integrated into the 'slait ai' web application, where it demonstrates stable inference.
>
---
#### [new 052] Open-ended Scientific Discovery via Bayesian Surprise
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自主科学发现任务，旨在解决AI如何自主提出有价值问题。工作是提出AutoDS方法，利用贝叶斯惊喜驱动探索，提升发现新知识的能力。**

- **链接: [http://arxiv.org/pdf/2507.00310v1](http://arxiv.org/pdf/2507.00310v1)**

> **作者:** Dhruv Agarwal; Bodhisattwa Prasad Majumder; Reece Adamson; Megha Chakravorty; Satvika Reddy Gavireddy; Aditya Parashar; Harshit Surana; Bhavana Dalvi Mishra; Andrew McCallum; Ashish Sabharwal; Peter Clark
>
> **摘要:** The promise of autonomous scientific discovery (ASD) hinges not only on answering questions, but also on knowing which questions to ask. Most recent works in ASD explore the use of large language models (LLMs) in goal-driven settings, relying on human-specified research questions to guide hypothesis generation. However, scientific discovery may be accelerated further by allowing the AI system to drive exploration by its own criteria. The few existing approaches in open-ended ASD select hypotheses based on diversity heuristics or subjective proxies for human interestingness, but the former struggles to meaningfully navigate the typically vast hypothesis space, and the latter suffers from imprecise definitions. This paper presents AutoDS -- a method for open-ended ASD that instead drives scientific exploration using Bayesian surprise. Here, we quantify the epistemic shift from the LLM's prior beliefs about a hypothesis to its posterior beliefs after gathering experimental results. To efficiently explore the space of nested hypotheses, our method employs a Monte Carlo tree search (MCTS) strategy with progressive widening using surprisal as the reward function. We evaluate AutoDS in the setting of data-driven discovery across 21 real-world datasets spanning domains such as biology, economics, finance, and behavioral science. Our results demonstrate that under a fixed budget, AutoDS substantially outperforms competitors by producing 5--29\% more discoveries deemed surprising by the LLM. Our human evaluation further finds that two-thirds of AutoDS discoveries are surprising to the domain experts, suggesting this is an important step forward towards building open-ended ASD systems.
>
---
#### [new 053] Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention
- **分类: cs.LG; cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理中的长文本建模任务，旨在解决状态空间模型在捕捉长距离依赖上的不足。通过引入上下文相关稀疏注意力机制，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00449v1](http://arxiv.org/pdf/2507.00449v1)**

> **作者:** Zhihao Zhan; Jianan Zhao; Zhaocheng Zhu; Jian Tang
>
> **备注:** Proceedings of the 42nd International Conference on Machine Learning, ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models, 18 pages, 9 figures
>
> **摘要:** Efficient long-context modeling remains a critical challenge for natural language processing (NLP), as the time complexity of the predominant Transformer architecture scales quadratically with the sequence length. While state-space models (SSMs) offer alternative sub-quadratic solutions, they struggle to capture long-range dependencies effectively. In this work, we focus on analyzing and improving the long-context modeling capabilities of SSMs. We show that the widely used synthetic task, associative recall, which requires a model to recall a value associated with a single key without context, insufficiently represents the complexities of real-world long-context modeling. To address this limitation, we extend the associative recall to a novel synthetic task, \emph{joint recall}, which requires a model to recall the value associated with a key given in a specified context. Theoretically, we prove that SSMs do not have the expressiveness to solve multi-query joint recall in sub-quadratic time complexity. To resolve this issue, we propose a solution based on integrating SSMs with Context-Dependent Sparse Attention (CDSA), which has the expressiveness to solve multi-query joint recall with sub-quadratic computation. To bridge the gap between theoretical analysis and real-world applications, we propose locality-sensitive Hashing Attention with sparse Key Selection (HAX), which instantiates the theoretical solution and is further tailored to natural language domains. Extensive experiments on both synthetic and real-world long-context benchmarks show that HAX consistently outperforms SSM baselines and SSMs integrated with context-independent sparse attention (CISA).
>
---
#### [new 054] Thinking About Thinking: SAGE-nano's Inverse Reasoning for Self-Aware Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型的可解释性。通过逆向推理技术，使模型能自我反思并解释其推理过程，增强透明度与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.00092v1](http://arxiv.org/pdf/2507.00092v1)**

> **作者:** Basab Jha; Firoj Paudel; Ujjwal Puri; Zhang Yuting; Choi Donghyuk; Wang Junhao
>
> **备注:** 19 pages, 2 figures, 9 tables
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities at solving complex reasoning tasks with Chain-of-Thought (CoT) prompting, but their decision-making processes remain somewhat blackbox. We introduce textbfinverse reasoning, a novel paradigm enabling LLMs to decompose and explain their own reasoning chains post-hoc. Our approach, used in SAGE-nano, a 4-billion-parameter reasoning model, employs a metacognitive structure that reflects back via attention processes to identify major decision points and generate explanations of reasoning choices. While typical CoT approaches are directed towards forward reasoning generation, inverse reasoning provides insight into why specific reasoning chains were selected over others. Through thorough testing of logical reasoning puzzles, math problems and ethical dilemmas from AQUA-RAT, CommonsenseQA, and customized benchmarks, we demonstrate that SAGE-nano is at the cutting edge both on reasoning accuracy (74.6% on AQUA-RAT) and explanation quality (92.1% human preference score) for its task, and offers performance almost on par with models like Claude-3.5 Sonnet or GPT-4o. Our contributions are: (i) the first rigorous framework for LLM self-reflection via inverse reasoning, (ii) a novel metalearning framework to reverse the attention flow, (iii) comprehensive evaluation frameworks for reasoning transparency, and (iv) evidence that increasing reasoning using inverse reasoning improves interpretability along with reasoning performance. Our work creates new avenues for transparent AI systems and closes significant gaps in AI safety, education, and scientific discovery.
>
---
#### [new 055] MassTool: A Multi-Task Search-Based Tool Retrieval Framework for Large Language Models
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于工具检索任务，旨在提升大语言模型与外部工具交互的准确性。通过多任务框架MassTool，结合查询理解与工具匹配，解决传统方法忽视查询理解的问题。**

- **链接: [http://arxiv.org/pdf/2507.00487v1](http://arxiv.org/pdf/2507.00487v1)**

> **作者:** Jianghao Lin; Xinyuan Wang; Xinyi Dai; Menghui Zhu; Bo Chen; Ruiming Tang; Yong Yu; Weinan Zhang
>
> **摘要:** Tool retrieval is a critical component in enabling large language models (LLMs) to interact effectively with external tools. It aims to precisely filter the massive tools into a small set of candidates for the downstream tool-augmented LLMs. However, most existing approaches primarily focus on optimizing tool representations, often neglecting the importance of precise query comprehension. To address this gap, we introduce MassTool, a multi-task search-based framework designed to enhance both query representation and tool retrieval accuracy. MassTool employs a two-tower architecture: a tool usage detection tower that predicts the need for function calls, and a tool retrieval tower that leverages a query-centric graph convolution network (QC-GCN) for effective query-tool matching. It also incorporates search-based user intent modeling (SUIM) to handle diverse and out-of-distribution queries, alongside an adaptive knowledge transfer (AdaKT) module for efficient multi-task learning. By jointly optimizing tool usage detection loss, list-wise retrieval loss, and contrastive regularization loss, MassTool establishes a robust dual-step sequential decision-making pipeline for precise query understanding. Extensive experiments demonstrate its effectiveness in improving retrieval accuracy. Our code is available at https://github.com/wxydada/MassTool.
>
---
#### [new 056] MANTA: Cross-Modal Semantic Alignment and Information-Theoretic Optimization for Long-form Multimodal Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MANTA框架，解决多模态理解中的语义对齐与信息优化问题，提升长视频问答等任务性能。**

- **链接: [http://arxiv.org/pdf/2507.00068v1](http://arxiv.org/pdf/2507.00068v1)**

> **作者:** Ziqi Zhong; Daniel Tang
>
> **摘要:** While multi-modal learning has advanced significantly, current approaches often treat modalities separately, creating inconsistencies in representation and reasoning. We introduce MANTA (Multi-modal Abstraction and Normalization via Textual Alignment), a theoretically-grounded framework that unifies visual and auditory inputs into a structured textual space for seamless processing with large language models. MANTA addresses four key challenges: (1) semantic alignment across modalities with information-theoretic optimization, (2) adaptive temporal synchronization for varying information densities, (3) hierarchical content representation for multi-scale understanding, and (4) context-aware retrieval of sparse information from long sequences. We formalize our approach within a rigorous mathematical framework, proving its optimality for context selection under token constraints. Extensive experiments on the challenging task of Long Video Question Answering show that MANTA improves state-of-the-art models by up to 22.6% in overall accuracy, with particularly significant gains (27.3%) on videos exceeding 30 minutes. Additionally, we demonstrate MANTA's superiority on temporal reasoning tasks (23.8% improvement) and cross-modal understanding (25.1% improvement). Our framework introduces novel density estimation techniques for redundancy minimization while preserving rare signals, establishing new foundations for unifying multimodal representations through structured text.
>
---
#### [new 057] $μ^2$Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation
- **分类: cs.LG; cs.CL; eess.IV**

- **简介: 该论文属于放射学报告生成任务，旨在解决医学影像信息提取难和报告质量评估难的问题，提出$\mu^2$Tokenizer提升多模态信息融合与报告质量。**

- **链接: [http://arxiv.org/pdf/2507.00316v1](http://arxiv.org/pdf/2507.00316v1)**

> **作者:** Siyou Li; Pengyao Qin; Huanan Wu; Dong Nie; Arun J. Thirunavukarasu; Juntao Yu; Le Zhang
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Automated radiology report generation (RRG) aims to produce detailed textual reports from clinical imaging, such as computed tomography (CT) scans, to improve the accuracy and efficiency of diagnosis and provision of management advice. RRG is complicated by two key challenges: (1) inherent complexity in extracting relevant information from imaging data under resource constraints, and (2) difficulty in objectively evaluating discrepancies between model-generated and expert-written reports. To address these challenges, we propose $\mu^2$LLM, a $\underline{\textbf{mu}}$ltiscale $\underline{\textbf{mu}}$ltimodal large language models for RRG tasks. The novel ${\mu}^2$Tokenizer, as an intermediate layer, integrates multi-modal features from the multiscale visual tokenizer and the text tokenizer, then enhances report generation quality through direct preference optimization (DPO), guided by GREEN-RedLlama. Experimental results on four large CT image-report medical datasetdemonstrate that our method outperforms existing approaches, highlighting the potential of our fine-tuned $\mu^2$LLMs on limited data for RRG tasks.
>
---
#### [new 058] CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出"CaughtCheating"任务，研究MLLM在视觉推理中的漏洞，探索其检测细微线索的能力，旨在提升模型的人类级侦探推理水平。**

- **链接: [http://arxiv.org/pdf/2507.00045v1](http://arxiv.org/pdf/2507.00045v1)**

> **作者:** Ming Li; Chenguang Wang; Yijun Liang; Xiyao Wang; Yuhang Zhou; Xiyang Wu; Yuqing Zhang; Ruiyi Zhang; Tianyi Zhou
>
> **摘要:** Recent agentic Multi-Modal Large Language Models (MLLMs) such as GPT-o3 have achieved near-ceiling scores on various existing benchmarks, motivating a demand for more challenging test tasks. These MLLMs have been reported to excel in a few expert-level tasks for humans, e.g., GeoGuesser, reflecting their potential as a detective who can notice minuscule cues in an image and weave them into coherent, situational explanations, leading to a reliable answer. But can they match the performance of excellent human detectives? To answer this question, we investigate some hard scenarios where GPT-o3 can still handle, and find a common scenario where o3's performance drops to nearly zero, which we name CaughtCheating. It is inspired by the social media requests that ask others to detect suspicious clues from photos shared by the poster's partner. We conduct extensive experiments and analysis to understand why existing MLLMs lack sufficient capability to solve this kind of task. CaughtCheating provides a class of challenging visual perception and reasoning tasks with great value and practical usage. Success in these tasks paves the way for MLLMs to acquire human-level detective perception and reasoning capabilities.
>
---
#### [new 059] Enhancing Reasoning Capabilities in SLMs with Reward Guided Dataset Distillation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在提升小语言模型的推理能力。针对知识蒸馏中学生模型仅复制教师模型响应的问题，提出基于奖励的数据集蒸馏方法AdvDistill，通过多生成和奖励机制提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00054v1](http://arxiv.org/pdf/2507.00054v1)**

> **作者:** Shreyansh Padarha
>
> **备注:** 17 Pages, 7 figures
>
> **摘要:** The push to compress and impart the proficiency of Large Language Models (LLMs) into more deployable and efficient Small Language Models (SLMs) has benefited from improvements in knowledge distillation (KD) techniques. These techniques allow a smaller student model to learn from a more capable and larger teacher model's responses. However, distillation often revolves around the student model merely copying the teacher's in-distribution responses, limiting its generalisability. This limitation is amplified on reasoning tasks and can be computationally expensive. In this study, we propose AdvDistill, a reward-guided dataset distillation framework. We utilise multiple generations (responses) from a teacher for each prompt and assign rewards based on rule-based verifiers. These varying and normally distributed rewards serve as weights when training student models. Our methods and their subsequent behavioural analysis demonstrate a significant improvement in student model performance for mathematical and complex reasoning tasks, showcasing the efficacy and benefits of incorporating a rewarding mechanism in dataset distillation processes.
>
---
#### [new 060] ROSE: Toward Reality-Oriented Safety Evaluation of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于大语言模型安全评估任务，旨在解决现有评估方法在对抗性提示覆盖不足和现实场景对齐差的问题。通过多目标强化学习生成多样化、情境丰富的对抗性提示，提升安全评估效果。**

- **链接: [http://arxiv.org/pdf/2507.00026v1](http://arxiv.org/pdf/2507.00026v1)**

> **作者:** Jiale Ding; Xiang Zheng; Cong Wang; Wei-Bin Lee; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed as black-box components in real-world applications, evaluating their safety-especially under adversarial prompting-has become critical. Arguably, effective safety evaluations should be adaptive, evolving with LLM capabilities, and also cover a broad spectrum of harmful topics and real-world scenarios to fully expose potential vulnerabilities. Existing manual safety benchmarks, built on handcrafted adversarial prompts, are limited by their static nature and the intensive labor required to update them, making it difficult to keep pace with rapidly advancing LLMs. In contrast, automated adversarial prompt generation offers a promising path toward adaptive evaluation. However, current methods often suffer from insufficient adversarial topic coverage (topic-level diversity) and weak alignment with real-world contexts. These shortcomings stem from the exploration-exploitation dilemma in black-box optimization and a lack of real-world contextualization, resulting in adversarial prompts that are both topically narrow and scenario-repetitive. To address these issues, we propose Reality-Oriented Safety Evaluation (ROSE), a novel framework that uses multi-objective reinforcement learning to fine-tune an adversarial LLM for generating topically diverse and contextually rich adversarial prompts. Experiments show that ROSE outperforms existing methods in uncovering safety vulnerabilities in state-of-the-art LLMs, with notable improvements in integrated evaluation metrics. We hope ROSE represents a step toward more practical and reality-oriented safety evaluation of LLMs. WARNING: This paper contains examples of potentially harmful text.
>
---
#### [new 061] Federated Learning-Enabled Hybrid Language Models for Communication-Efficient Token Transmission
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于边缘AI任务，旨在减少通信开销。通过联邦学习优化不确定性阈值，降低对大模型的依赖，提升效率。**

- **链接: [http://arxiv.org/pdf/2507.00082v1](http://arxiv.org/pdf/2507.00082v1)**

> **作者:** Faranaksadat Solat; Joohyung Lee; Mohamed Seif; Dusit Niyato; H. Vincent Poor
>
> **备注:** 17 pages, 16 figures, IEEE Internet of Things
>
> **摘要:** Hybrid Language Models (HLMs) combine the low-latency efficiency of Small Language Models (SLMs) on edge devices with the high accuracy of Large Language Models (LLMs) on centralized servers. Unlike traditional end-to-end LLM inference, HLMs reduce latency and communication by invoking LLMs only when local SLM predictions are uncertain, i.e., when token-level confidence is low or entropy is high. However, ambiguous or low-confidence predictions still require frequent offloading to the LLM, leading to significant communication overhead in bandwidth-constrained settings. To address this, we propose FedHLM, a communication-efficient HLM framework that integrates uncertainty-aware inference with Federated Learning (FL). FedHLM's key innovation lies in collaboratively learning token-level uncertainty thresholds that govern when LLM assistance is needed. Rather than using static or manually tuned thresholds, FedHLM employs FL to optimize these thresholds in a privacy-preserving, distributed manner. Additionally, it leverages embedding-based token representations for Peer-to-Peer (P2P) resolution, enabling clients to reuse tokens inferred by semantically similar peers without engaging the LLM. We further introduce hierarchical model aggregation: edge servers refine local routing policies through client updates, while cross-cluster coordination aligns global decision boundaries. This layered design captures recurring uncertainty patterns, reducing redundant LLM queries. Experiments on large-scale news classification tasks show that FedHLM reduces LLM transmissions by over 95 percent with negligible accuracy loss, making it well-suited for scalable and efficient edge-AI applications.
>
---
#### [new 062] GLU Attention Improve Transformer
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于自然语言处理任务，旨在提升Transformer模型性能。通过引入GLU Attention机制，在不增加参数和计算成本的情况下，提高模型效果与收敛速度。**

- **链接: [http://arxiv.org/pdf/2507.00022v1](http://arxiv.org/pdf/2507.00022v1)**

> **作者:** Zehao Wang
>
> **备注:** 4 pages 4 figures
>
> **摘要:** Gated Linear Units (GLU) have shown great potential in enhancing neural network performance. In this paper, I introduce a novel attention mechanism called GLU Attention, which introduces nonlinearity into the values of Attention. My experiments demonstrate that GLU Attention improves both model performance and convergence speed across text and vision modalities with zero additional parameters and negligible computational costs. GLU Attention is lightweight and can seamlessly integrate with other technologies, such as Flash Attention, Rotary Position Embedding (RoPE), and various Multi-Head Attention (MHA) variants such as Grouped-Query Attention (GQA). This project is open-sourced at github.
>
---
#### [new 063] Hypertokens: Holographic Associative Memory in Tokenized LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs中的信息分散问题。通过引入HDRAM框架，利用符号化记忆和量子启发方法提升信息检索效率。**

- **链接: [http://arxiv.org/pdf/2507.00002v1](http://arxiv.org/pdf/2507.00002v1)**

> **作者:** Christopher James Augeri
>
> **备注:** preprint as accepted to https://qnlp.ai/ - Quantum AI and NLP Conference 2025
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities but suffer from apparent precision loss, reframed here as information spreading. This reframing shifts the problem from computational precision to an information-theoretic communication issue. We address the K:V and V:K memory problem in LLMs by introducing HDRAM (Holographically Defined Random Access Memory), a symbolic memory framework treating transformer latent space as a spread-spectrum channel. Built upon hypertokens, structured symbolic codes integrating classical error-correcting codes (ECC), holographic computing, and quantum-inspired search, HDRAM recovers distributed information through principled despreading. These phase-coherent memory addresses enable efficient key-value operations and Grover-style search in latent space. By combining ECC grammar with compressed sensing and Krylov subspace alignment, HDRAM significantly improves associative retrieval without architectural changes, demonstrating how Classical-Holographic-Quantum-inspired (CHQ) principles can fortify transformer architectures.
>
---
#### [new 064] Flexible Language Modeling in Continuous Space with Transformer-based Autoregressive Flows
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言建模任务，旨在解决传统模型在离散空间中的局限性。提出TarFlowLM框架，利用Transformer的自回归流在连续潜空间建模，提升灵活性和上下文捕捉能力。**

- **链接: [http://arxiv.org/pdf/2507.00425v1](http://arxiv.org/pdf/2507.00425v1)**

> **作者:** Ruixiang Zhang; Shuangfei Zhai; Jiatao Gu; Yizhe Zhang; Huangjie Zheng; Tianrong Chen; Miguel Angel Bautista; Josh Susskind; Navdeep Jaitly
>
> **摘要:** Autoregressive models have driven remarkable progress in language modeling. Their foundational reliance on discrete tokens, unidirectional context, and single-pass decoding, while central to their success, also inspires the exploration of a design space that could offer new axes of modeling flexibility. In this work, we explore an alternative paradigm, shifting language modeling from a discrete token space to a continuous latent space. We propose a novel framework TarFlowLM, that employs transformer-based autoregressive normalizing flows to model these continuous representations. This approach unlocks substantial flexibility, enabling the construction of models that can capture global bi-directional context through stacked, alternating-direction autoregressive transformations, support block-wise generation with flexible token patch sizes, and facilitate a hierarchical multi-pass generation process. We further propose new mixture-based coupling transformations designed to capture complex dependencies within the latent space shaped by discrete data, and demonstrate theoretical connections to conventional discrete autoregressive models. Extensive experiments on language modeling benchmarks demonstrate strong likelihood performance and highlight the flexible modeling capabilities inherent in our framework.
>
---
#### [new 065] Leveraging Large Language Models for Spontaneous Speech-Based Suicide Risk Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于自杀风险检测任务，旨在通过语音识别青少年自杀风险。利用大语言模型提取特征，结合传统声学与语义特征，提升检测准确率。**

- **链接: [http://arxiv.org/pdf/2507.00693v1](http://arxiv.org/pdf/2507.00693v1)**

> **作者:** Yifan Gao; Jiao Fu; Long Guo; Hong Liu
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Early identification of suicide risk is crucial for preventing suicidal behaviors. As a result, the identification and study of patterns and markers related to suicide risk have become a key focus of current research. In this paper, we present the results of our work in the 1st SpeechWellness Challenge (SW1), which aims to explore speech as a non-invasive and easily accessible mental health indicator for identifying adolescents at risk of suicide.Our approach leverages large language model (LLM) as the primary tool for feature extraction, alongside conventional acoustic and semantic features. The proposed method achieves an accuracy of 74\% on the test set, ranking first in the SW1 challenge. These findings demonstrate the potential of LLM-based methods for analyzing speech in the context of suicide risk assessment.
>
---
#### [new 066] Interpretable AI for Time-Series: Multi-Model Heatmap Fusion with Global Attention and NLP-Generated Explanations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于时间序列解释任务，解决模型可解释性不足问题，融合ResNet与Transformer热图并生成NLP解释，提升医疗和工业场景的决策透明度。**

- **链接: [http://arxiv.org/pdf/2507.00234v1](http://arxiv.org/pdf/2507.00234v1)**

> **作者:** Jiztom Kavalakkatt Francis; Matthew J Darr
>
> **备注:** 13 pages
>
> **摘要:** In this paper, we present a novel framework for enhancing model interpretability by integrating heatmaps produced separately by ResNet and a restructured 2D Transformer with globally weighted input saliency. We address the critical problem of spatial-temporal misalignment in existing interpretability methods, where convolutional networks fail to capture global context and Transformers lack localized precision - a limitation that impedes actionable insights in safety-critical domains like healthcare and industrial monitoring. Our method merges gradient-weighted activation maps (ResNet) and Transformer attention rollout into a unified visualization, achieving full spatial-temporal alignment while preserving real-time performance. Empirical evaluations on clinical (ECG arrhythmia detection) and industrial (energy consumption prediction) datasets demonstrate significant improvements: the hybrid framework achieves 94.1% accuracy (F1 0.93) on the PhysioNet dataset and reduces regression error to RMSE = 0.28 kWh (R2 = 0.95) on the UCI Energy Appliance dataset-outperforming standalone ResNet, Transformer, and InceptionTime baselines by 3.8-12.4%. An NLP module translates fused heatmaps into domain-specific narratives (e.g., "Elevated ST-segment between 2-4 seconds suggests myocardial ischemia"), validated via BLEU-4 (0.586) and ROUGE-L (0.650) scores. By formalizing interpretability as causal fidelity and spatial-temporal alignment, our approach bridges the gap between technical outputs and stakeholder understanding, offering a scalable solution for transparent, time-aware decision-making.
>
---
#### [new 067] Enhancing LLM Agent Safety via Causal Influence Prompting
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI安全领域，旨在提升LLM代理的安全性。通过因果影响图（CIP）识别和缓解决策风险，确保代理行为可靠。**

- **链接: [http://arxiv.org/pdf/2507.00979v1](http://arxiv.org/pdf/2507.00979v1)**

> **作者:** Dongyoon Hahm; Woogyeol Jin; June Suk Choi; Sungsoo Ahn; Kimin Lee
>
> **备注:** Accepted at ACL 2025 Findings, Source code: https://github.com/HahmDY/causal_influence_prompting.git
>
> **摘要:** As autonomous agents powered by large language models (LLMs) continue to demonstrate potential across various assistive tasks, ensuring their safe and reliable behavior is crucial for preventing unintended consequences. In this work, we introduce CIP, a novel technique that leverages causal influence diagrams (CIDs) to identify and mitigate risks arising from agent decision-making. CIDs provide a structured representation of cause-and-effect relationships, enabling agents to anticipate harmful outcomes and make safer decisions. Our approach consists of three key steps: (1) initializing a CID based on task specifications to outline the decision-making process, (2) guiding agent interactions with the environment using the CID, and (3) iteratively refining the CID based on observed behaviors and outcomes. Experimental results demonstrate that our method effectively enhances safety in both code execution and mobile device control tasks.
>
---
#### [new 068] The language of time: a language model perspective on time-series foundation models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于时间序列建模任务，旨在解决跨领域迁移的矛盾问题。通过理论与实验分析，揭示时间序列模型如何借鉴语言模型的表示学习机制，提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00078v1](http://arxiv.org/pdf/2507.00078v1)**

> **作者:** Yi Xie; Yun Xiong; Zejian Shi; Hao Niu; Zhengfu Liu
>
> **摘要:** With the rise of large language models, the paradigm of training foundation models with massive parameter counts on vast datasets has been adopted in multiple domains to achieve remarkable success. Time series foundation models represent a significant extension of this paradigm, demonstrating exceptional expressive power, generalization, and cross-domain transferability. However, this gives rise to a fundamental paradox: time series data reflect distinct dynamical systems, making cross-domain transfer intuitively implausible, yet this is contradicted by the models' empirical success. To resolve this paradox, this paper investigates, from both theoretical and experimental perspectives, the representation learning mechanisms and generalization capabilities of patch-based time series foundation models. We argue that such models are not merely applying a new architecture but are fundamentally generalizing the representation paradigm of language models by extending deterministic vector-based representations to latent probabilistic distributional forms. Our theoretical analysis supports this framework by demonstrating that continuous time-series patches can be faithfully quantized into a discrete vocabulary whose key statistical properties are highly consistent with those of natural language. This generalization allows time series models to inherit the robust representation and transfer abilities of large language models, thereby explaining their superior performance in temporal tasks. Ultimately, our work provides a rigorous theoretical cornerstone for understanding, evaluating, and improving the safety and reliability of large-scale time series foundation models.
>
---
#### [new 069] Safe Low Bandwidth SPV: A Formal Treatment of Simplified Payment Verification Protocols and Security Bounds
- **分类: cs.CR; cs.CL; cs.DC; 68Q85, 68M10, 94A60, 91A80, 68Q17, 68W10, 68R10; C.2.2; F.2.2; D.4.6; K.6.5**

- **简介: 该论文属于密码学与区块链领域，解决SPV协议的安全性与效率问题。通过形式化验证和数学分析，提出安全且低带宽的SPV方案。**

- **链接: [http://arxiv.org/pdf/2507.00740v1](http://arxiv.org/pdf/2507.00740v1)**

> **作者:** Craig S Wright
>
> **备注:** 56 pages 5 images
>
> **摘要:** This paper presents a complete formal specification, protocol description, and mathematical proof structure for Simplified Payment Verification (SPV) as originally defined in the Bitcoin whitepaper \cite{nakamoto2008}. In stark contrast to the misrepresentations proliferated by popular implementations, we show that SPV is not only secure under bounded adversarial assumptions but strictly optimal for digital cash systems requiring scalable and verifiable transaction inclusion. We reconstruct the SPV protocol from first principles, grounding its verification model in symbolic automata, Merkle membership relations, and chain-of-proof dominance predicates. Through rigorous probabilistic and game-theoretic analysis, we derive the economic bounds within which the protocol operates securely and verify its liveness and safety properties under partial connectivity, hostile relay networks, and adversarial propagation delay. Our specification further introduces low-bandwidth optimisations such as adaptive polling and compressed header synchronisation while preserving correctness. This document serves both as a blueprint for secure SPV implementation and a rebuttal of common misconceptions surrounding non-validating clients.
>
---
#### [new 070] Implicit Reward as the Bridge: A Unified View of SFT and DPO Connections
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型后训练任务，解决SFT与偏好学习的统一理论问题，提出改进方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00018v1](http://arxiv.org/pdf/2507.00018v1)**

> **作者:** Bo Wang; Qinyuan Cheng; Runyu Peng; Rong Bao; Peiji Li; Qipeng Guo; Linyang Li; Zhiyuan Zeng; Yunhua Zhou; Xipeng Qiu
>
> **摘要:** Post-training processes are essential phases in grounding pre-trained language models to real-world tasks, with learning from demonstrations or preference signals playing a crucial role in this adaptation. We present a unified theoretical framework bridging Supervised Fine-Tuning (SFT) and preference learning in Large Language Model (LLM) post-training. Through rigorous mathematical derivation, we demonstrate that both SFT and preference learning methods like Direct Preference Optimization (DPO) operate within the same optimal policy-reward subspace, with SFT representing a special case of implicit reward learning. Our analysis reveals a critical limitation in conventional SFT: the KL divergence term in distribution matching becomes constant with respect to the policy during optimization, failing to constrain model updates. To address this, we propose a simple yet effective learning rate reduction approach that yields significant performance improvements (up to \textbf{25\%} relative gain and \textbf{6\%} absolute win rate increase in instruction following tasks. Additionally, we derive alternative SFT objectives from various f-divergence functions that preserve the KL term during optimization, further enhancing post-DPO model performance. Finally, we extend the theoretical relationship between LLM logits and Q-functions from preference learning to the SFT context, providing mathematical derivations and experimental validation.
>
---
## 更新

#### [replaced 001] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.24119v2](http://arxiv.org/pdf/2506.24119v2)**

> **作者:** Bo Liu; Leon Guertler; Simon Yu; Zichen Liu; Penghui Qi; Daniel Balcells; Mickel Liu; Cheston Tan; Weiyan Shi; Min Lin; Wee Sun Lee; Natasha Jaques
>
> **备注:** Work in Progress
>
> **摘要:** Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development.
>
---
#### [replaced 002] The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.22419v2](http://arxiv.org/pdf/2506.22419v2)**

> **作者:** Bingchen Zhao; Despoina Magka; Minqi Jiang; Xian Li; Roberta Raileanu; Tatiana Shavrina; Jean-Christophe Gagnon-Audet; Kelvin Niu; Shagun Sodhani; Michael Shvartsman; Andrei Lupu; Alisia Lupidi; Edan Toledo; Karen Hambardzumyan; Martin Josifoski; Thomas Foster; Lucia Cipolina-Kun; Abhishek Charnalia; Derek Dunfield; Alexander H. Miller; Oisin Mac Aodha; Jakob Foerster; Yoram Bachrach
>
> **摘要:** Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent.
>
---
#### [replaced 003] Evaluating Deduplication Techniques for Economic Research Paper Titles with a Focus on Semantic Similarity using NLP and LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01141v3](http://arxiv.org/pdf/2410.01141v3)**

> **作者:** Doohee You; S Fraiberger
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** This study investigates efficient deduplication techniques for a large NLP dataset of economic research paper titles. We explore various pairing methods alongside established distance measures (Levenshtein distance, cosine similarity) and a sBERT model for semantic evaluation. Our findings suggest a potentially low prevalence of duplicates based on the observed semantic similarity across different methods. Further exploration with a human-annotated ground truth set is completed for a more conclusive assessment. The result supports findings from the NLP, LLM based distance metrics.
>
---
#### [replaced 004] Llama-Nemotron: Efficient Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00949v4](http://arxiv.org/pdf/2505.00949v4)**

> **作者:** Akhiad Bercovich; Itay Levy; Izik Golan; Mohammad Dabbah; Ran El-Yaniv; Omri Puny; Ido Galil; Zach Moshe; Tomer Ronen; Najeeb Nabwani; Ido Shahaf; Oren Tropp; Ehud Karpas; Ran Zilberstein; Jiaqi Zeng; Soumye Singhal; Alexander Bukharin; Yian Zhang; Tugrul Konuk; Gerald Shen; Ameya Sunil Mahabaleshwarkar; Bilal Kartal; Yoshi Suhara; Olivier Delalleau; Zijia Chen; Zhilin Wang; David Mosallanezhad; Adi Renduchintala; Haifeng Qian; Dima Rekesh; Fei Jia; Somshubra Majumdar; Vahid Noroozi; Wasi Uddin Ahmad; Sean Narenthiran; Aleksander Ficek; Mehrzad Samadi; Jocelyn Huang; Siddhartha Jain; Igor Gitman; Ivan Moshkov; Wei Du; Shubham Toshniwal; George Armstrong; Branislav Kisacanin; Matvei Novikov; Daria Gitman; Evelina Bakhturina; Prasoon Varshney; Makesh Narsimhan; Jane Polak Scowcroft; John Kamalu; Dan Su; Kezhi Kong; Markus Kliegl; Rabeeh Karimi; Ying Lin; Sanjeev Satheesh; Jupinder Parmar; Pritam Gundecha; Brandon Norick; Joseph Jennings; Shrimai Prabhumoye; Syeda Nahida Akter; Mostofa Patwary; Abhinav Khattar; Deepak Narayanan; Roger Waleffe; Jimmy Zhang; Bor-Yiing Su; Guyue Huang; Terry Kong; Parth Chadha; Sahil Jain; Christine Harvey; Elad Segal; Jining Huang; Sergey Kashirsky; Robert McQueen; Izzy Putterman; George Lam; Arun Venkatesan; Sherry Wu; Vinh Nguyen; Manoj Kilaru; Andrew Wang; Anna Warno; Abhilash Somasamudramath; Sandip Bhaskar; Maka Dong; Nave Assaf; Shahar Mor; Omer Ullman Argov; Scot Junkin; Oleksandr Romanenko; Pedro Larroy; Monika Katariya; Marco Rovinelli; Viji Balas; Nicholas Edelman; Anahita Bhiwandiwalla; Muthu Subramaniam; Smita Ithape; Karthik Ramamoorthy; Yuting Wu; Suguna Varshini Velury; Omri Almog; Joyjit Daw; Denys Fridman; Erick Galinkin; Michael Evans; Shaona Ghosh; Katherine Luna; Leon Derczynski; Nikki Pope; Eileen Long; Seth Schneider; Guillermo Siman; Tomasz Grzegorzek; Pablo Ribalta; Monika Katariya; Chris Alexiuk; Joey Conway; Trisha Saar; Ann Guan; Krzysztof Pawelec; Shyamala Prayaga; Oleksii Kuchaiev; Boris Ginsburg; Oluwatobi Olabiyi; Kari Briski; Jonathan Cohen; Bryan Catanzaro; Jonah Alben; Yonatan Geifman; Eric Chung
>
> **摘要:** We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use. The family comes in three sizes -- Nano (8B), Super (49B), and Ultra (253B) -- and performs competitively with state-of-the-art reasoning models such as DeepSeek-R1 while offering superior inference throughput and memory efficiency. In this report, we discuss the training procedure for these models, which entails using neural architecture search from Llama 3 models for accelerated inference, knowledge distillation, and continued pretraining, followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning. Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle, allowing users to switch between standard chat and reasoning modes during inference. To further support open research and facilitate model development, we provide the following resources: 1. We release the Llama-Nemotron reasoning models -- LN-Nano, LN-Super, and LN-Ultra -- under the commercially permissive NVIDIA Open Model License Agreement. 2. We release the complete post-training dataset: Llama-Nemotron-Post-Training-Dataset. 3. We also release our training codebases: NeMo, NeMo-Aligner, and Megatron-LM.
>
---
#### [replaced 005] Positional Bias in Binary Question Answering: How Uncertainty Shapes Model Preferences
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.23743v2](http://arxiv.org/pdf/2506.23743v2)**

> **作者:** Tiziano Labruna; Simone Gallo; Giovanni Da San Martino
>
> **摘要:** Positional bias in binary question answering occurs when a model systematically favors one choice over another based solely on the ordering of presented options. In this study, we quantify and analyze positional bias across five large language models under varying degrees of answer uncertainty. We re-adapted the SQuAD-it dataset by adding an extra incorrect answer option and then created multiple versions with progressively less context and more out-of-context answers, yielding datasets that range from low to high uncertainty. Additionally, we evaluate two naturally higher-uncertainty benchmarks: (1) WebGPT - question pairs with unequal human-assigned quality scores, and (2) Winning Arguments - where models predict the more persuasive argument in Reddit's r/ChangeMyView exchanges. Across each dataset, the order of the "correct" (or higher-quality/persuasive) option is systematically flipped (first placed in position 1, then in position 2) to compute both Preference Fairness and Position Consistency. We observe that positional bias is nearly absent under low-uncertainty conditions, but grows exponentially when it becomes doubtful to decide which option is correct.
>
---
#### [replaced 006] An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21393v3](http://arxiv.org/pdf/2503.21393v3)**

> **作者:** Rohitash Chandra; Aryan Chaudhari; Yeshwanth Rayavarapu
>
> **摘要:** Large Language models (LLMs) have been prominent for language translation, including low-resource languages. There has been limited study on the assessment of the quality of translations generated by LLMs, including Gemini, GPT, and Google Translate. This study addresses this limitation by using semantic and sentiment analysis of selected LLMs for Indian languages, including Sanskrit, Telugu and Hindi. We select prominent texts (Bhagavad Gita, Tamas and Maha Prasthanam ) that have been well translated by experts and use LLMs to generate their translations into English, and provide a comparison with selected expert (human) translations. Our investigation revealed that while LLMs have made significant progress in translation accuracy, challenges remain in preserving sentiment and semantic integrity, especially in metaphorical and philosophical contexts for texts such as the Bhagavad Gita. The sentiment analysis revealed that GPT models are better at preserving the sentiment polarity for the given texts when compared to human (expert) translation. The results revealed that GPT models are generally better at maintaining the sentiment and semantics when compared to Google Translate. This study could help in the development of accurate and culturally sensitive translation systems for large language models.
>
---
#### [replaced 007] A Study of In-Context-Learning-Based Text-to-SQL Errors
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2501.09310v2](http://arxiv.org/pdf/2501.09310v2)**

> **作者:** Jiawei Shen; Chengcheng Wan; Ruoyi Qiao; Jiazhen Zou; Hang Xu; Yuchen Shao; Yueling Zhang; Weikai Miao; Geguang Pu
>
> **摘要:** Large language models (LLMs) have been adopted to perform text-to-SQL tasks, utilizing their in-context learning (ICL) capability to translate natural language questions into structured query language (SQL). However, such a technique faces correctness problems and requires efficient repairing solutions. In this paper, we conduct the first comprehensive study of text-to-SQL errors. Our study covers four representative ICL-based techniques, five basic repairing methods, two benchmarks, and two LLM settings. We find that text-to-SQL errors are widespread and summarize 29 error types of 7 categories. We also find that existing repairing attempts have limited correctness improvement at the cost of high computational overhead with many mis-repairs. Based on the findings, we propose MapleRepair, a novel text-to-SQL error detection and repairing framework. The evaluation demonstrates that MapleRepair outperforms existing solutions by repairing 13.8% more queries with neglectable mis-repairs and 67.4% less overhead.
>
---
#### [replaced 008] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19955v2](http://arxiv.org/pdf/2505.19955v2)**

> **作者:** Hui Chen; Miao Xiong; Yujie Lu; Wei Han; Ailin Deng; Yufei He; Jiaying Wu; Yibo Li; Yue Liu; Bryan Hooi
>
> **备注:** 42 pages, 9 figures
>
> **摘要:** Recent advancements in AI agents have demonstrated their growing potential to drive and support scientific discovery. In this work, we introduce MLR-Bench, a comprehensive benchmark for evaluating AI agents on open-ended machine learning research. MLR-Bench includes three key components: (1) 201 research tasks sourced from NeurIPS, ICLR, and ICML workshops covering diverse ML topics; (2) MLR-Judge, an automated evaluation framework combining LLM-based reviewers with carefully designed review rubrics to assess research quality; and (3) MLR-Agent, a modular agent scaffold capable of completing research tasks through four stages: idea generation, proposal formulation, experimentation, and paper writing. Our framework supports both stepwise assessment across these distinct research stages, and end-to-end evaluation of the final research paper. We then use MLR-Bench to evaluate six frontier LLMs and an advanced coding agent, finding that while LLMs are effective at generating coherent ideas and well-structured papers, current coding agents frequently (e.g., in 80% of the cases) produce fabricated or invalidated experimental results--posing a major barrier to scientific reliability. We validate MLR-Judge through human evaluation, showing high agreement with expert reviewers, supporting its potential as a scalable tool for research evaluation. We open-source MLR-Bench to help the community benchmark, diagnose, and improve AI research agents toward trustworthy and transparent scientific discovery.
>
---
#### [replaced 009] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition
- **分类: cs.CL; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2503.21248v2](http://arxiv.org/pdf/2503.21248v2)**

> **作者:** Yujie Liu; Zonglin Yang; Tong Xie; Jinjie Ni; Ben Gao; Yuqiang Li; Shixiang Tang; Wanli Ouyang; Erik Cambria; Dongzhan Zhou
>
> **摘要:** Large language models (LLMs) have demonstrated potential in assisting scientific research, yet their ability to discover high-quality research hypotheses remains unexamined due to the lack of a dedicated benchmark. To address this gap, we introduce the first large-scale benchmark for evaluating LLMs with a near-sufficient set of sub-tasks of scientific discovery: inspiration retrieval, hypothesis composition, and hypothesis ranking. We develop an automated framework that extracts critical components - research questions, background surveys, inspirations, and hypotheses - from scientific papers across 12 disciplines, with expert validation confirming its accuracy. To prevent data contamination, we focus exclusively on papers published in 2024, ensuring minimal overlap with LLM pretraining data. Our evaluation reveals that LLMs perform well in retrieving inspirations, an out-of-distribution task, suggesting their ability to surface novel knowledge associations. This positions LLMs as "research hypothesis mines", capable of facilitating automated scientific discovery by generating innovative hypotheses at scale with minimal human intervention.
>
---
#### [replaced 010] ETTA: Elucidating the Design Space of Text-to-Audio Models
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.19351v2](http://arxiv.org/pdf/2412.19351v2)**

> **作者:** Sang-gil Lee; Zhifeng Kong; Arushi Goel; Sungwon Kim; Rafael Valle; Bryan Catanzaro
>
> **备注:** ICML 2025. Demo: https://research.nvidia.com/labs/adlr/ETTA/ Code: https://github.com/NVIDIA/elucidated-text-to-audio
>
> **摘要:** Recent years have seen significant progress in Text-To-Audio (TTA) synthesis, enabling users to enrich their creative workflows with synthetic audio generated from natural language prompts. Despite this progress, the effects of data, model architecture, training objective functions, and sampling strategies on target benchmarks are not well understood. With the purpose of providing a holistic understanding of the design space of TTA models, we set up a large-scale empirical experiment focused on diffusion and flow matching models. Our contributions include: 1) AF-Synthetic, a large dataset of high quality synthetic captions obtained from an audio understanding model; 2) a systematic comparison of different architectural, training, and inference design choices for TTA models; 3) an analysis of sampling methods and their Pareto curves with respect to generation quality and inference speed. We leverage the knowledge obtained from this extensive analysis to propose our best model dubbed Elucidated Text-To-Audio (ETTA). When evaluated on AudioCaps and MusicCaps, ETTA provides improvements over the baselines trained on publicly available data, while being competitive with models trained on proprietary data. Finally, we show ETTA's improved ability to generate creative audio following complex and imaginative captions -- a task that is more challenging than current benchmarks.
>
---
#### [replaced 011] Not Minds, but Signs: Reframing LLMs through Semiotics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17080v2](http://arxiv.org/pdf/2505.17080v2)**

> **作者:** Davide Picca
>
> **摘要:** This paper challenges the prevailing tendency to frame Large Language Models (LLMs) as cognitive systems, arguing instead for a semiotic perspective that situates these models within the broader dynamics of sign manipulation and meaning-making. Rather than assuming that LLMs understand language or simulate human thought, we propose that their primary function is to recombine, recontextualize, and circulate linguistic forms based on probabilistic associations. By shifting from a cognitivist to a semiotic framework, we avoid anthropomorphism and gain a more precise understanding of how LLMs participate in cultural processes, not by thinking, but by generating texts that invite interpretation. Through theoretical analysis and practical examples, the paper demonstrates how LLMs function as semiotic agents whose outputs can be treated as interpretive acts, open to contextual negotiation and critical reflection. We explore applications in literature, philosophy, education, and cultural production, emphasizing how LLMs can serve as tools for creativity, dialogue, and critical inquiry. The semiotic paradigm foregrounds the situated, contingent, and socially embedded nature of meaning, offering a more rigorous and ethically aware framework for studying and using LLMs. Ultimately, this approach reframes LLMs as technological participants in an ongoing ecology of signs. They do not possess minds, but they alter how we read, write, and make meaning, compelling us to reconsider the foundations of language, interpretation, and the role of artificial systems in the production of knowledge.
>
---
#### [replaced 012] Efficient Domain-adaptive Continual Pretraining for the Process Industry in the German Language
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.19856v3](http://arxiv.org/pdf/2504.19856v3)**

> **作者:** Anastasia Zhukova; Christian E. Matt; Bela Gipp
>
> **备注:** accepted to TSD 2025
>
> **摘要:** Domain-adaptive continual pretraining (DAPT) is a state-of-the-art technique that further trains a language model (LM) on its pretraining task, e.g., masked language modeling (MLM), when common domain adaptation via LM fine-tuning is not possible due to a lack of labeled task data. Although popular, MLM requires a significant corpus of domain-related data, which is difficult to obtain for specific domains in languages other than English, such as the process industry in the German language. This paper introduces an efficient approach called ICL-augmented pretraining or ICL-APT that leverages in-context learning (ICL) and k-nearest neighbors (kNN) to augment target data with domain-related and in-domain texts, significantly reducing GPU time while maintaining strong model performance. Our results show that the best configuration of ICL-APT performed better than the state-of-the-art DAPT by 28.7% (7.87 points) and requires almost 4 times less GPU-computing time, providing a cost-effective solution for industries with limited computational capacity. The findings highlight the broader applicability of this framework to other low-resource industries, making NLP-based solutions more accessible and feasible in production environments.
>
---
#### [replaced 013] Large Language Model Confidence Estimation via Black-Box Access
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.04370v4](http://arxiv.org/pdf/2406.04370v4)**

> **作者:** Tejaswini Pedapati; Amit Dhurandhar; Soumya Ghosh; Soham Dan; Prasanna Sattigeri
>
> **备注:** Accepted to TMLR 2025
>
> **摘要:** Estimating uncertainty or confidence in the responses of a model can be significant in evaluating trust not only in the responses, but also in the model as a whole. In this paper, we explore the problem of estimating confidence for responses of large language models (LLMs) with simply black-box or query access to them. We propose a simple and extensible framework where, we engineer novel features and train a (interpretable) model (viz. logistic regression) on these features to estimate the confidence. We empirically demonstrate that our simple framework is effective in estimating confidence of Flan-ul2, Llama-13b, Mistral-7b and GPT-4 on four benchmark Q\&A tasks as well as of Pegasus-large and BART-large on two benchmark summarization tasks with it surpassing baselines by even over $10\%$ (on AUROC) in some cases. Additionally, our interpretable approach provides insight into features that are predictive of confidence, leading to the interesting and useful discovery that our confidence models built for one LLM generalize zero-shot across others on a given dataset.
>
---
#### [replaced 014] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01144v4](http://arxiv.org/pdf/2501.01144v4)**

> **作者:** Wonsuk Jang; Thierry Tambe
>
> **备注:** ICML 2025
>
> **摘要:** The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
>
---
#### [replaced 015] Seeking and Updating with Live Visual Knowledge
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05288v2](http://arxiv.org/pdf/2504.05288v2)**

> **作者:** Mingyang Fu; Yuyang Peng; Dongping Chen; Zetong Zhou; Benlin Liu; Yao Wan; Zhou Zhao; Philip S. Yu; Ranjay Krishna
>
> **备注:** Preprint. Under Review
>
> **摘要:** The visual world around us constantly evolves, from real-time news and social media trends to global infrastructure changes visible through satellite imagery and augmented reality enhancements. However, Multimodal Large Language Models (MLLMs), which automate many tasks, struggle to stay current, limited by the cutoff dates in their fixed training datasets. To quantify this stagnation, we introduce LiveVQA, the first-of-its-kind dataset featuring 107,143 samples and 12 categories data specifically designed to support research in both seeking and updating with live visual knowledge. Drawing from recent news articles, video platforms, and academic publications in April 2024-May 2025, LiveVQA enables evaluation of how models handle latest visual information beyond their knowledge boundaries and how current methods help to update them. Our comprehensive benchmarking of 17 state-of-the-art MLLMs reveals significant performance gaps on content beyond knowledge cutoff, and tool-use or agentic visual seeking framework drastically gain an average of 327% improvement. Furthermore, we explore parameter-efficient fine-tuning (PEFT) methods to update MLLMs with new visual knowledge. We dive deeply to the critical balance between adapter capacity and model capability when updating MLLMs with new visual knowledge. All the experimental dataset and source code are publicly available at: https://livevqa.github.io.
>
---
#### [replaced 016] Capturing Visualization Design Rationale
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16571v2](http://arxiv.org/pdf/2506.16571v2)**

> **作者:** Maeve Hutchinson; Radu Jianu; Aidan Slingsby; Jo Wood; Pranava Madhyastha
>
> **备注:** To be presented at IEEE VIS 2025
>
> **摘要:** Prior natural language datasets for data visualization have focused on tasks such as visualization literacy assessment, insight generation, and visualization generation from natural language instructions. These studies often rely on controlled setups with purpose-built visualizations and artificially constructed questions. As a result, they tend to prioritize the interpretation of visualizations, focusing on decoding visualizations rather than understanding their encoding. In this paper, we present a new dataset and methodology for probing visualization design rationale through natural language. We leverage a unique source of real-world visualizations and natural language narratives: literate visualization notebooks created by students as part of a data visualization course. These notebooks combine visual artifacts with design exposition, in which students make explicit the rationale behind their design decisions. We also use large language models (LLMs) to generate and categorize question-answer-rationale triples from the narratives and articulations in the notebooks. We then carefully validate the triples and curate a dataset that captures and distills the visualization design choices and corresponding rationales of the students.
>
---
#### [replaced 017] DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21096v2](http://arxiv.org/pdf/2506.21096v2)**

> **作者:** Kang He; Yuzhe Ding; Haining Wang; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Previous multimodal sentence representation learning methods have achieved impressive performance. However, most approaches focus on aligning images and text at a coarse level, facing two critical challenges:cross-modal misalignment bias and intra-modal semantic divergence, which significantly degrade sentence representation quality. To address these challenges, we propose DALR (Dual-level Alignment Learning for Multimodal Sentence Representation). For cross-modal alignment, we propose a consistency learning module that softens negative samples and utilizes semantic similarity from an auxiliary task to achieve fine-grained cross-modal alignment. Additionally, we contend that sentence relationships go beyond binary positive-negative labels, exhibiting a more intricate ranking structure. To better capture these relationships and enhance representation quality, we integrate ranking distillation with global intra-modal alignment learning. Comprehensive experiments on semantic textual similarity (STS) and transfer (TR) tasks validate the effectiveness of our approach, consistently demonstrating its superiority over state-of-the-art baselines.
>
---
#### [replaced 018] Pipelined Decoder for Efficient Context-Aware Text Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23431v2](http://arxiv.org/pdf/2506.23431v2)**

> **作者:** Zixian Huang; Chenxu Niu; Yu Gu; Gengyang Xiao; Xinwei Huang; Gong Cheng
>
> **摘要:** As the basis of generative AI, an autoregressive model requires the generation of a new token depending on all the previously generated tokens, which brings high quality but also restricts the model to generate tokens one by one, forming a bottleneck limiting the generation speed. In this paper, we propose a new decoder architecture that efficiently generates text in parallel for context-aware generation tasks. Our proposed pipelined decoder initiates the generation of multiple subsequences simultaneously, and, at each time-step, it generates a new token for each subsequence to realize parallelism. Experiments on multiple text generation tasks, including question answering, text summarization, and keyphrase generation, show that our pipelined decoder significantly improves the generation speed without a significant loss of generation quality or additional memory consumption.
>
---
#### [replaced 019] Iterative Resolution of Prompt Ambiguities Using a Progressive Cutting-Search Approach
- **分类: cs.AI; cs.CL; cs.ET; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02952v2](http://arxiv.org/pdf/2505.02952v2)**

> **作者:** Fabrizio Marozzo
>
> **摘要:** Generative AI systems have revolutionized human interaction by enabling natural language-based coding and problem solving. However, the inherent ambiguity of natural language often leads to imprecise instructions, forcing users to iteratively test, correct, and resubmit their prompts. We propose an iterative approach that systematically narrows down these ambiguities through a structured series of clarification questions and alternative solution proposals, illustrated with input/output examples as well. Once every uncertainty is resolved, a final, precise solution is generated. Evaluated on a diverse dataset spanning coding, data analysis, and creative writing, our method demonstrates superior accuracy, competitive resolution times, and higher user satisfaction compared to conventional one-shot solutions, which typically require multiple manual iterations to achieve a correct output.
>
---
#### [replaced 020] Flexora: Flexible Low Rank Adaptation for Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.10774v4](http://arxiv.org/pdf/2408.10774v4)**

> **作者:** Chenxing Wei; Yao Shu; Ying Tiffany He; Fei Richard Yu
>
> **备注:** 40 pages, 15 figures
>
> **摘要:** Large Language Models (LLMs) are driving advancements in artificial intelligence by increasing the scale of model parameters, which has significantly enhanced generalization ability and unlocked new capabilities in practice. However, their performance in specific downstream tasks is usually hindered by their knowledge boundaries on these tasks. Thus, fine-tuning techniques, especially the widely used Low-Rank Adaptation (LoRA) method, have been introduced to expand the boundaries on these tasks, whereas LoRA would underperform on certain tasks owing to its potential overfitting on these tasks. To overcome this overfitting and improve the performance of LoRA, we propose the flexible low rank adaptation (Flexora) method to automatically and flexibly select the most important layers needing to be fine-tuned to achieve the best performance on different downstream tasks. Specifically, Flexora firstly frames this layer selection problem as a well-defined hyperparameter optimization (HPO) problem, then addresses it using the unrolled differentiation (UD) method, and finally selects the most useful layers based on the optimized hyperparameters. Our extensive experiments on many pretrained models and natural language tasks show that Flexora is able to consistently improve over the existing baselines, indicating the effectiveness of our Flexora in practice. We additionally provide insightful theoretical results and many ablation studies to deliver a comprehensive understanding of our Flexora.
>
---
#### [replaced 021] Flow-Modulated Scoring for Semantic-Aware Knowledge Graph Completion
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23137v2](http://arxiv.org/pdf/2506.23137v2)**

> **作者:** Siyuan Li; Ruitong Liu; Yan Wen; Te Sun
>
> **备注:** 10 pages
>
> **摘要:** Effective modeling of multifaceted relations is pivotal for Knowledge Graph Completion (KGC). However, a majority of existing approaches are predicated on static, embedding-based scoring, exhibiting inherent limitations in capturing contextual dependencies and relational dynamics. Addressing this gap, we propose the Flow-Modulated Scoring (FMS) framework. FMS comprises two principal components: (1) a semantic context learning module that encodes context-sensitive entity representations, and (2) a conditional flow-matching module designed to learn the dynamic transformation from a head to a tail embedding, governed by the aforementioned context. The resultant predictive vector field, representing the context-informed relational path, serves to dynamically refine the initial static score of an entity pair. Through this synergy of context-aware static representations and conditioned dynamic information, FMS facilitates a more profound modeling of relational semantics. Comprehensive evaluations on several standard benchmarks demonstrate that our proposed method surpasses prior state-of-the-art results.
>
---
#### [replaced 022] SAGE: Steering Dialog Generation with Future-Aware State-Action Augmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.03040v2](http://arxiv.org/pdf/2503.03040v2)**

> **作者:** Yizhe Zhang; Navdeep Jaitly
>
> **备注:** 9 pages main text
>
> **摘要:** Recent advances in large language models have demonstrated impressive capabilities in task-oriented applications, yet building emotionally intelligent chatbots that can engage in natural, strategic conversations remains a challenge. We present a novel approach called SAGE that uses latent variables to control long-horizon behavior in dialogue generation. At the core of our method is the State-Action Chain (SAC), which augments standard language model fine-tuning by introducing latent variables that encapsulate emotional states and conversational strategies between dialogue turns. During inference, these variables are generated before each response, enabling coarse-grained control over dialogue progression while maintaining natural interaction patterns. We also introduce a self-improvement pipeline that leverages dialogue tree search, LLM-based reward modeling, and targeted fine-tuning to optimize conversational trajectories. Our experimental results show that models trained with this approach demonstrate improved performance in emotional intelligence metrics while maintaining strong capabilities on LLM benchmarks. The discrete nature of our latent variables facilitates search-based strategies and provides a foundation for future applications of reinforcement learning to dialogue systems, where learning can occur at the state level rather than the token level. https://github.com/apple/ml-sage-dialog-gen
>
---
#### [replaced 023] Integrating Expert Labels into LLM-based Emission Goal Detection: Example Selection vs Automatic Prompt Design
- **分类: cs.LG; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2412.06432v2](http://arxiv.org/pdf/2412.06432v2)**

> **作者:** Marco Wrzalik; Adrian Ulges; Anne Uersfeld; Florian Faust; Viola Campos
>
> **摘要:** We address the detection of emission reduction goals in corporate reports, an important task for monitoring companies' progress in addressing climate change. Specifically, we focus on the issue of integrating expert feedback in the form of labeled example passages into LLM-based pipelines, and compare the two strategies of (1) a dynamic selection of few-shot examples and (2) the automatic optimization of the prompt by the LLM itself. Our findings on a public dataset of 769 climate-related passages from real-world business reports indicate that automatic prompt optimization is the superior approach, while combining both methods provides only limited benefit. Qualitative results indicate that optimized prompts do indeed capture many intricacies of the targeted emission goal extraction task.
>
---
#### [replaced 024] Fact Recall, Heuristics or Pure Guesswork? Precise Interpretations of Language Models for Fact Completion
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14405v4](http://arxiv.org/pdf/2410.14405v4)**

> **作者:** Denitsa Saynova; Lovisa Hagström; Moa Johansson; Richard Johansson; Marco Kuhlmann
>
> **备注:** accepted to ACL Findings 2025
>
> **摘要:** Language models (LMs) can make a correct prediction based on many possible signals in a prompt, not all corresponding to recall of factual associations. However, current interpretations of LMs fail to take this into account. For example, given the query "Astrid Lindgren was born in" with the corresponding completion "Sweden", no difference is made between whether the prediction was based on knowing where the author was born or assuming that a person with a Swedish-sounding name was born in Sweden. In this paper, we present a model-specific recipe - PrISM - for constructing datasets with examples of four different prediction scenarios: generic language modeling, guesswork, heuristics recall and exact fact recall. We apply two popular interpretability methods to the scenarios: causal tracing (CT) and information flow analysis. We find that both yield distinct results for each scenario. Results for exact fact recall and generic language modeling scenarios confirm previous conclusions about the importance of mid-range MLP sublayers for fact recall, while results for guesswork and heuristics indicate a critical role of late last token position MLP sublayers. In summary, we contribute resources for a more extensive and granular study of fact completion in LMs, together with analyses that provide a more nuanced understanding of how LMs process fact-related queries.
>
---
#### [replaced 025] HyperCLOVA X THINK Technical Report
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22403v2](http://arxiv.org/pdf/2506.22403v2)**

> **作者:** NAVER Cloud HyperCLOVA X Team
>
> **备注:** 50 pages, 13 figures; fixed figures in the appendix
>
> **摘要:** We introduce HyperCLOVA X THINK, the first reasoning-focused large language model in the HyperCLOVA X family, pre-trained on roughly $6$ trillion high-quality Korean, and English tokens, augmented with targeted synthetic Korean data. It was implemented as a compute-memory-balanced Peri-LN Transformer scaled with $\mu$P, pre-trained through a three-stage curriculum that expands the context window to $128$K tokens, and post-trained via supervised fine-tuning with Reinforcement Learning from Verifiable Rewards supports both detailed rationale and concise-answer modes. It delivers competitive performance against similarly sized models on Korea-focused benchmarks such as KMMLU, CSAT, KoBALT-700, HAERAE-1.0, and KoBigBench, while preserving robust bilingual consistency and translation quality. In addition, a vision-augmented variant matches or exceeds GPT-4.1 on the KCSAT STEM benchmark, all of which are achieved with substantially lower training compute than existing models of similar sizes. We also present a pruning and distillation technique that will soon be applied to HyperCLOVA X THINK for an open-source and business-friendly foundation model. Altogether, these capabilities position HyperCLOVA X THINK as a robust foundation for Korean AI innovation and a valuable resource for the global research community.
>
---
#### [replaced 026] Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14518v2](http://arxiv.org/pdf/2505.14518v2)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025. Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation.
>
---
#### [replaced 027] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21098v2](http://arxiv.org/pdf/2506.21098v2)**

> **作者:** Qinwen Chen; Wenbiao Tao; Zhiwei Zhu; Mingfan Xi; Liangzhong Guo; Yuan Wang; Wei Wang; Yunshi Lan
>
> **备注:** 7 pages, 4 figures. Accepted at ACL 2025 Industry Track
>
> **摘要:** Community Question Answering (CQA) platforms can be deemed as important knowledge bases in community, but effectively leveraging historical interactions and domain knowledge in real-time remains a challenge. Existing methods often underutilize external knowledge, fail to incorporate dynamic historical QA context, or lack memory mechanisms suited for industrial deployment. We propose ComRAG, a retrieval-augmented generation framework for real-time industrial CQA that integrates static knowledge with dynamic historical QA pairs via a centroid-based memory mechanism designed for retrieval, generation, and efficient storage. Evaluated on three industrial CQA datasets, ComRAG consistently outperforms all baselines--achieving up to 25.9% improvement in vector similarity, reducing latency by 8.7% to 23.3%, and lowering chunk growth from 20.23% to 2.06% over iterations.
>
---
#### [replaced 028] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24778v2](http://arxiv.org/pdf/2505.24778v2)**

> **作者:** Jiayu Liu; Qing Zong; Weiqi Wang; Yangqiu Song
>
> **备注:** ACL2025 Main
>
> **摘要:** As large language models (LLMs) are increasingly used in high-stakes domains, accurately assessing their confidence is crucial. Humans typically express confidence through epistemic markers (e.g., "fairly confident") instead of numerical values. However, it remains unclear whether LLMs consistently use these markers to reflect their intrinsic confidence due to the difficulty of quantifying uncertainty associated with various markers. To address this gap, we first define marker confidence as the observed accuracy when a model employs an epistemic marker. We evaluate its stability across multiple question-answering datasets in both in-distribution and out-of-distribution settings for open-source and proprietary LLMs. Our results show that while markers generalize well within the same distribution, their confidence is inconsistent in out-of-distribution scenarios. These findings raise significant concerns about the reliability of epistemic markers for confidence estimation, underscoring the need for improved alignment between marker based confidence and actual model uncertainty. Our code is available at https://github.com/HKUST-KnowComp/MarCon.
>
---
#### [replaced 029] ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling
- **分类: cs.CL; eess.SP; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2412.14373v2](http://arxiv.org/pdf/2412.14373v2)**

> **作者:** William Han; Chaojing Duan; Michael A. Rosenberg; Emerson Liu; Ding Zhao
>
> **备注:** 38 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional versatility across domains, including applications to electrocardiograms (ECGs). A growing body of work focuses on generating text from multi-channeled ECG signals and corresponding textual prompts. Existing approaches often involve a two-stage process: pretraining an ECG-specific encoder with a self-supervised learning (SSL) objective, followed by finetuning an LLM for natural language generation (NLG) using encoder-derived features. However, these methods face two key limitations: inefficiency due to multi-stage training and challenges in interpreting encoder-generated features. To overcome these issues, we propose ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs. ECG-Byte compresses and encodes ECG signals into tokens, enabling direct end-to-end LLM training by combining ECG and text tokens. This approach enhances interpretability, as ECG tokens can be directly mapped back to the original signals. Leveraging ECG-Byte, we achieve competitive NLG performance while training 3 times faster and using just 48\% of the data required by traditional two-stage methods.
>
---
#### [replaced 030] A Graph-Based Classical and Quantum Approach to Deterministic L-System Inference
- **分类: quant-ph; cs.CL; cs.DS; cs.FL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19906v3](http://arxiv.org/pdf/2411.19906v3)**

> **作者:** Ali Lotfi; Ian McQuillan; Steven Rayan
>
> **备注:** 17 pages, 1 figure
>
> **摘要:** L-systems can be made to model and create simulations of many biological processes, such as plant development. Finding an L-system for a given process is typically solved by hand, by experts, in a massively time-consuming process. It would be significant if this could be done automatically from data, such as from sequences of images. In this paper, we are interested in inferring a particular type of L-system, deterministic context-free L-system (D0L-system) from a sequence of strings. We introduce the characteristic graph of a sequence of strings, which we then utilize to translate our problem (inferring D0L-systems) in polynomial time into the maximum independent set problem (MIS) and the SAT problem. After that, we offer a classical exact algorithm and an approximate quantum algorithm for the problem.
>
---
#### [replaced 031] OM4OV: Leveraging Ontology Matching for Ontology Versioning
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2409.20302v4](http://arxiv.org/pdf/2409.20302v4)**

> **作者:** Zhangcheng Qiang; Kerry Taylor; Weiqing Wang
>
> **备注:** 15 pages, 8 figures, 1 table
>
> **摘要:** Due to the dynamic nature of the Semantic Web, version control is necessary to capture time-varying information, particularly for widely used ontologies. Despite the long-standing recognition of ontology versioning (OV) as a crucial component for efficient ontology management, the growing size of ontologies and accumulating errors caused by manual labour overwhelm current OV approaches. In this paper, we propose a fresh approach to performing OV using existing ontology matching (OM) techniques and systems. We introduce a unified OM4OV pipeline. From an OM perspective, we reconstruct a new task formulation and measurements for OV tasks. Building upon the prior alignment(s) from OM, we propose a pipeline optimisation method called the cross-reference (CR) mechanism to enhance overall OV performance. We experimentally validate the OM4OV pipeline and the cross-reference mechanism in an OV testbed originating from the Ontology Alignment Evaluation Initiative (OAEI) datasets. We also discuss insights into OM used for OV tasks, where some apparent false mappings detected by OV systems are not actually untrue.
>
---
#### [replaced 032] Benchmarking the Pedagogical Knowledge of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18710v3](http://arxiv.org/pdf/2506.18710v3)**

> **作者:** Maxime Lelièvre; Amy Waldock; Meng Liu; Natalia Valdés Aspillaga; Alasdair Mackintosh; María José Ogando Portela; Jared Lee; Paul Atherton; Robin A. A. Ince; Oliver G. B. Garrod
>
> **摘要:** Benchmarks like Massive Multitask Language Understanding (MMLU) have played a pivotal role in evaluating AI's knowledge and abilities across diverse domains. However, existing benchmarks predominantly focus on content knowledge, leaving a critical gap in assessing models' understanding of pedagogy - the method and practice of teaching. This paper introduces The Pedagogy Benchmark, a novel dataset designed to evaluate large language models on their Cross-Domain Pedagogical Knowledge (CDPK) and Special Education Needs and Disability (SEND) pedagogical knowledge. These benchmarks are built on a carefully curated set of questions sourced from professional development exams for teachers, which cover a range of pedagogical subdomains such as teaching strategies and assessment methods. Here we outline the methodology and development of these benchmarks. We report results for 97 models, with accuracies spanning a range from 28% to 89% on the pedagogical knowledge questions. We consider the relationship between cost and accuracy and chart the progression of the Pareto value frontier over time. We provide online leaderboards at https://rebrand.ly/pedagogy which are updated with new models and allow interactive exploration and filtering based on various model properties, such as cost per token and open-vs-closed weights, as well as looking at performance in different subjects. LLMs and generative AI have tremendous potential to influence education and help to address the global learning crisis. Education-focused benchmarks are crucial to measure models' capacities to understand pedagogical concepts, respond appropriately to learners' needs, and support effective teaching practices across diverse contexts. They are needed for informing the responsible and evidence-based deployment of LLMs and LLM-based tools in educational settings, and for guiding both development and policy decisions.
>
---
#### [replaced 033] Text Production and Comprehension by Human and Artificial Intelligence: Interdisciplinary Workshop Report
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22698v2](http://arxiv.org/pdf/2506.22698v2)**

> **作者:** Emily Dux Speltz
>
> **摘要:** This report synthesizes the outcomes of a recent interdisciplinary workshop that brought together leading experts in cognitive psychology, language learning, and artificial intelligence (AI)-based natural language processing (NLP). The workshop, funded by the National Science Foundation, aimed to address a critical knowledge gap in our understanding of the relationship between AI language models and human cognitive processes in text comprehension and composition. Through collaborative dialogue across cognitive, linguistic, and technological perspectives, workshop participants examined the underlying processes involved when humans produce and comprehend text, and how AI can both inform our understanding of these processes and augment human capabilities. The workshop revealed emerging patterns in the relationship between large language models (LLMs) and human cognition, with highlights on both the capabilities of LLMs and their limitations in fully replicating human-like language understanding and generation. Key findings include the potential of LLMs to offer insights into human language processing, the increasing alignment between LLM behavior and human language processing when models are fine-tuned with human feedback, and the opportunities and challenges presented by human-AI collaboration in language tasks. By synthesizing these findings, this report aims to guide future research, development, and implementation of LLMs in cognitive psychology, linguistics, and education. It emphasizes the importance of ethical considerations and responsible use of AI technologies while striving to enhance human capabilities in text comprehension and production through effective human-AI collaboration.
>
---
#### [replaced 034] SPADE: Structured Prompting Augmentation for Dialogue Enhancement in Machine-Generated Text Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.15044v2](http://arxiv.org/pdf/2503.15044v2)**

> **作者:** Haoyi Li; Angela Yifei Yuan; Soyeon Caren Han; Christopher Leckie
>
> **备注:** ACL LLMSEC
>
> **摘要:** The increasing capability of large language models (LLMs) to generate synthetic content has heightened concerns about their misuse, driving the development of Machine-Generated Text (MGT) detection models. However, these detectors face significant challenges due to the lack of high-quality synthetic datasets for training. To address this issue, we propose SPADE, a structured framework for detecting synthetic dialogues using prompt-based positive and negative samples. Our proposed methods yield 14 new dialogue datasets, which we benchmark against eight MGT detection models. The results demonstrate improved generalization performance when utilizing a mixed dataset produced by proposed augmentation frameworks, offering a practical approach to enhancing LLM application security. Considering that real-world agents lack knowledge of future opponent utterances, we simulate online dialogue detection and examine the relationship between chat history length and detection accuracy. Our open-source datasets, code and prompts can be downloaded from https://github.com/AngieYYF/SPADE-customer-service-dialogue.
>
---
#### [replaced 035] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Radiology with Zero-Shot Multi-Task Capability
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.07416v2](http://arxiv.org/pdf/2504.07416v2)**

> **作者:** Jonggwon Park; Soobum Kim; Byungmu Yoon; Kyoyun Choi
>
> **摘要:** Recent advancements in multi-modal models have significantly improved vision-language (VL) alignment in radiology. However, existing approaches struggle to effectively utilize complex radiology reports for learning and offer limited interpretability through attention probability visualizations. To address these challenges, we introduce RadZero, a novel framework for VL alignment in radiology with zero-shot multi-task capability. A key component of our approach is VL-CABS (Vision-Language Cross-Attention Based on Similarity), which aligns text embeddings with local image features for interpretable, fine-grained VL reasoning. RadZero leverages large language models to extract concise semantic sentences from radiology reports and employs multi-positive contrastive training to effectively capture relationships between images and multiple relevant textual descriptions. It uses a pre-trained vision encoder with additional trainable Transformer layers, allowing efficient high-resolution image processing. By computing similarity between text embeddings and local image patch features, VL-CABS enables zero-shot inference with similarity probability for classification, and pixel-level VL similarity maps for grounding and segmentation. Experimental results on public chest radiograph benchmarks show that RadZero outperforms state-of-the-art methods in zero-shot classification, grounding, and segmentation. Furthermore, VL similarity map analysis highlights the potential of VL-CABS for improving explainability in VL alignment. Additionally, qualitative evaluation demonstrates RadZero's capability for open-vocabulary semantic segmentation, further validating its effectiveness in medical imaging.
>
---
#### [replaced 036] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.17117v3](http://arxiv.org/pdf/2505.17117v3)**

> **作者:** Chen Shani; Dan Jurafsky; Yann LeCun; Ravid Shwartz-Ziv
>
> **摘要:** Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
>
---
#### [replaced 037] Language Models Might Not Understand You: Evaluating Theory of Mind via Story Prompting
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.19089v2](http://arxiv.org/pdf/2506.19089v2)**

> **作者:** Nathaniel Getachew; Abulhair Saparov
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** We introduce $\texttt{StorySim}$, a programmable framework for synthetically generating stories to evaluate the theory of mind (ToM) and world modeling (WM) capabilities of large language models (LLMs). Unlike prior benchmarks that may suffer from contamination in pretraining data, $\texttt{StorySim}$ produces novel, compositional story prompts anchored by a highly controllable $\texttt{Storyboard}$, enabling precise manipulation of character perspectives and events. We use this framework to design first- and second-order ToM tasks alongside WM tasks that control for the ability to track and model mental states. Our experiments across a suite of state-of-the-art LLMs reveal that most models perform better on WM tasks than ToM tasks, and that models tend to perform better reasoning with humans compared to inanimate objects. Additionally, our framework enabled us to find evidence of heuristic behavior such as recency bias and an over-reliance on earlier events in the story. All code for generating data and evaluations is freely available.
>
---
#### [replaced 038] Two-Stage Regularization-Based Structured Pruning for LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18232v2](http://arxiv.org/pdf/2505.18232v2)**

> **作者:** Mingkuan Feng; Jinyang Wu; Siyuan Liu; Shuai Zhang; Ruihan Jin; Feihu Che; Pengpeng Shao; Zhengqi Wen; Jianhua Tao
>
> **摘要:** The deployment of large language models (LLMs) is largely hindered by their large number of parameters. Structural pruning has emerged as a promising solution. Prior structured pruning methods directly remove unimportant parameters based on certain metrics, which often causes knowledge loss and necessitates extensive retraining. To overcome this, we introduce a novel pruning method TRSP: Two-Stage Regularization-Based Structured Pruning for LLMs. Specifically, we multiply the output of each transformer layer by an initial learnable weight and iteratively learn these weights by adding their $\ell_1$-norm as a regularization term to the loss function, serving as the first-stage regularization. Subsequently, we apply additional regularization to the difference between the output and input of layers with smaller weights, encouraging the shift of knowledge to the preserved layers. This serves as the second-stage regularization. TRSP retains more knowledge and better preserves model performance than direct parameter elimination. Through extensive experimentation we show that TRSP outperforms strong layer-wise structured pruning methods without requiring retraining. As a layer-wise pruning method, it delivers notable end-to-end acceleration, making it a promising solution for efficient LLM deployment.
>
---
#### [replaced 039] Breaking mBad! Supervised Fine-tuning for Cross-Lingual Detoxification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16722v2](http://arxiv.org/pdf/2505.16722v2)**

> **作者:** Himanshu Beniwal; Youngwoo Kim; Maarten Sap; Soham Dan; Thomas Hartvigsen
>
> **摘要:** As large language models (LLMs) become increasingly prevalent in global applications, ensuring that they are toxicity-free across diverse linguistic contexts remains a critical challenge. We explore "Cross-lingual Detoxification", a cross-lingual paradigm that mitigates toxicity, enabling detoxification capabilities to transfer between high and low-resource languages across different script families. We analyze cross-lingual detoxification's effectiveness through 392 extensive settings to evaluate toxicity reduction in cross-distribution settings with limited data and investigate how mitigation impacts model performance on non-toxic tasks, revealing trade-offs between safety and knowledge preservation. Our code and dataset are publicly available at https://github.com/himanshubeniwal/Breaking-mBad.
>
---
#### [replaced 040] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14051v2](http://arxiv.org/pdf/2502.14051v2)**

> **作者:** Payman Behnam; Yaosheng Fu; Ritchie Zhao; Po-An Tsai; Zhiding Yu; Alexey Tumanov
>
> **备注:** ICML 2025
>
> **摘要:** Transformer-based Large Language Models rely critically on the KV cache to efficiently handle extended contexts during the decode phase. Yet, the size of the KV cache grows proportionally with the input length, burdening both memory bandwidth and capacity as decoding progresses. To address this challenge, we present RocketKV, a training-free KV cache compression strategy containing two consecutive stages. In the first stage, it performs coarse-grain permanent KV cache eviction on the input sequence tokens. In the second stage, it adopts a hybrid sparse attention method to conduct fine-grain top-k sparse attention, approximating the attention scores by leveraging both head and sequence dimensionality reductions. We show that RocketKV provides a compression ratio of up to 400$\times$, end-to-end speedup of up to 3.7$\times$ as well as peak memory reduction of up to 32.6% in the decode phase on an NVIDIA A100 GPU compared to the full KV cache baseline, while achieving negligible accuracy loss on a variety of long-context tasks. We also propose a variant of RocketKV for multi-turn scenarios, which consistently outperforms other existing methods and achieves accuracy nearly on par with an oracle top-k attention scheme.
>
---
#### [replaced 041] Quasi-symbolic Semantic Geometry over Transformer-based Variational AutoEncoder
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2210.06230v3](http://arxiv.org/pdf/2210.06230v3)**

> **作者:** Yingji Zhang; Danilo S. Carvalho; André Freitas
>
> **备注:** CoNLL2025 (Best Paper nomination)
>
> **摘要:** Formal/symbolic semantics can provide canonical, rigid controllability and interpretability to sentence representations due to their \textit{localisation} or \textit{composition} property. How can we deliver such property to the current distributional sentence representations to control and interpret the generation of language models (LMs)? In this work, we theoretically frame the sentence semantics as the composition of \textit{semantic role - word content} features and propose the formal semantic geometry. To inject such geometry into Transformer-based LMs (i.e. GPT2), we deploy Transformer-based Variational AutoEncoder with a supervision approach, where the sentence generation can be manipulated and explained over low-dimensional latent Gaussian space. In addition, we propose a new probing algorithm to guide the movement of sentence vectors over such geometry. Experimental results reveal that the formal semantic geometry can potentially deliver better control and interpretation to sentence generation.
>
---
#### [replaced 042] Generative Representational Learning of Foundation Models for Recommendation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11999v3](http://arxiv.org/pdf/2506.11999v3)**

> **作者:** Zheli Zhou; Chenxu Zhu; Jianghao Lin; Bo Chen; Ruiming Tang; Weinan Zhang; Yong Yu
>
> **备注:** Project page is available at https://junkfood436.github.io/RecFound/
>
> **摘要:** Developing a single foundation model with the capability to excel across diverse tasks has been a long-standing objective in the field of artificial intelligence. As the wave of general-purpose foundation models sweeps across various domains, their influence has significantly extended to the field of recommendation systems. While recent efforts have explored recommendation foundation models for various generative tasks, they often overlook crucial embedding tasks and struggle with the complexities of multi-task learning, including knowledge sharing & conflict resolution, and convergence speed inconsistencies. To address these limitations, we introduce RecFound, a generative representational learning framework for recommendation foundation models. We construct the first comprehensive dataset for recommendation foundation models covering both generative and embedding tasks across diverse scenarios. Based on this dataset, we propose a novel multi-task training scheme featuring a Task-wise Mixture of Low-rank Experts (TMoLE) to handle knowledge sharing & conflict, a Step-wise Convergence-oriented Sample Scheduler (S2Sched) to address inconsistent convergence, and a Model Merge module to balance the performance across tasks. Experiments demonstrate that RecFound achieves state-of-the-art performance across various recommendation tasks, outperforming existing baselines.
>
---
#### [replaced 043] T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00703v2](http://arxiv.org/pdf/2505.00703v2)**

> **作者:** Dongzhi Jiang; Ziyu Guo; Renrui Zhang; Zhuofan Zong; Hao Li; Le Zhuo; Shilin Yan; Pheng-Ann Heng; Hongsheng Li
>
> **备注:** Project Page: https://github.com/CaraJ7/T2I-R1
>
> **摘要:** Recent advancements in large language models have demonstrated how chain-of-thought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1, a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generation CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: https://github.com/CaraJ7/T2I-R1
>
---
#### [replaced 044] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.14640v2](http://arxiv.org/pdf/2401.14640v2)**

> **作者:** Nan Hu; Jiaoyan Chen; Yike Wu; Guilin Qi; Hongru Wang; Sheng Bi; Yongrui Chen; Tongtong Wu; Jeff Z. Pan
>
> **备注:** Accepted to ACL 2025 (Main Conference)
>
> **摘要:** Attributed Question Answering (AQA) has attracted wide attention, but there are still several limitations in evaluating the attributions, including lacking fine-grained attribution categories, relying on manual annotations, and failing to compare attributions with only subtle differences. To bridge these gaps, we introduce Complex Attributed Question Answering (CAQA), a large-scale benchmark containing comprehensive attribution categories, automatically generated using Knowledge Graphs (KGs), and complex attribution scenarios. We have conducted extensive experiments to verify the effectiveness of CAQA, including the benchmarking of 25 automatic evaluators, their comparison with human evaluators, the testing of LLM evaluators fine-tuned by CAQA and so on. These experiments also lead to a series of important findings that can benefit the future research of AQA. All the codes and data are publicly accessible at https://github.com/HuuuNan/CAQA-Benchmark.
>
---
#### [replaced 045] Parameter-Efficient Fine-Tuning via Circular Convolution
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.19342v4](http://arxiv.org/pdf/2407.19342v4)**

> **作者:** Aochuan Chen; Jiashun Cheng; Zijing Liu; Ziqi Gao; Fugee Tsung; Yu Li; Jia Li
>
> **备注:** ACL 2025
>
> **摘要:** Low-Rank Adaptation (LoRA) has gained popularity for fine-tuning large foundation models, leveraging low-rank matrices $\mathbf{A}$ and $\mathbf{B}$ to represent weight changes (i.e., $\Delta \mathbf{W} = \mathbf{B} \mathbf{A}$). This method reduces trainable parameters and mitigates heavy memory consumption associated with full delta matrices by sequentially multiplying $\mathbf{A}$ and $\mathbf{B}$ with the activation. Despite its success, the intrinsic low-rank characteristic may limit its performance. Although several variants have been proposed to address this issue, they often overlook the crucial computational and memory efficiency brought by LoRA. In this paper, we propose Circular Convolution Adaptation (C$^3$A), which not only achieves high-rank adaptation with enhanced performance but also excels in both computational power and memory utilization. Extensive experiments demonstrate that C$^3$A consistently outperforms LoRA and its variants across various fine-tuning tasks.
>
---
#### [replaced 046] Intertextual Parallel Detection in Biblical Hebrew: A Transformer-Based Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.24117v2](http://arxiv.org/pdf/2506.24117v2)**

> **作者:** David M. Smiley
>
> **摘要:** Identifying parallel passages in biblical Hebrew (BH) is central to biblical scholarship for understanding intertextual relationships. Traditional methods rely on manual comparison, a labor-intensive process prone to human error. This study evaluates the potential of pre-trained transformer-based language models, including E5, AlephBERT, MPNet, and LaBSE, for detecting textual parallels in the Hebrew Bible. Focusing on known parallels between Samuel/Kings and Chronicles, I assessed each model's capability to generate word embeddings distinguishing parallel from non-parallel passages. Using cosine similarity and Wasserstein Distance measures, I found that E5 and AlephBERT show promise; E5 excels in parallel detection, while AlephBERT demonstrates stronger non-parallel differentiation. These findings indicate that pre-trained models can enhance the efficiency and accuracy of detecting intertextual parallels in ancient texts, suggesting broader applications for ancient language studies.
>
---
#### [replaced 047] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.16211v2](http://arxiv.org/pdf/2505.16211v2)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Zhe Wang; Xingjian Du; Shun Zhang; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Xiaojun Jia; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Haoyang Li; Yiming Li; Xiaobin Zhuang; Yang Liu; Haibo Hu; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; XiaoFeng Wang; Wenyuan Xu; Wei Dong; Xinfeng Li
>
> **备注:** Technical Report
>
> **摘要:** The rapid advancement and expanding applications of Audio Large Language Models (ALLMs) demand a rigorous understanding of their trustworthiness. However, systematic research on evaluating these models, particularly concerning risks unique to the audio modality, remains largely unexplored. Existing evaluation frameworks primarily focus on the text modality or address only a restricted set of safety dimensions, failing to adequately account for the unique characteristics and application scenarios inherent to the audio modality. We introduce AudioTrust-the first multifaceted trustworthiness evaluation framework and benchmark specifically designed for ALLMs. AudioTrust facilitates assessments across six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. To comprehensively evaluate these dimensions, AudioTrust is structured around 18 distinct experimental setups. Its core is a meticulously constructed dataset of over 4,420 audio/text samples, drawn from real-world scenarios (e.g., daily conversations, emergency calls, voice assistant interactions), specifically designed to probe the multifaceted trustworthiness of ALLMs. For assessment, the benchmark carefully designs 9 audio-specific evaluation metrics, and we employ a large-scale automated pipeline for objective and scalable scoring of model outputs. Experimental results reveal the trustworthiness boundaries and limitations of current state-of-the-art open-source and closed-source ALLMs when confronted with various high-risk audio scenarios, offering valuable insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are available at https://github.com/JusperLee/AudioTrust.
>
---
#### [replaced 048] Graft: Integrating the Domain Knowledge via Efficient Parameter Synergy for MLLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.23940v2](http://arxiv.org/pdf/2506.23940v2)**

> **作者:** Yang Dai; Jianxiang An; Tianwei Lin; Hongyang He; Hongzhe Huang; Wenqiao Zhang; Zheqi Lv; Siliang Tang; Yueting Zhuang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved success across various domains. However, their applicability tends to degrade when confronted with different types of data inputs, especially for MLLMs that have been fine-tuned for specific tasks. Despite its importance, the study of knowledge sharing among domain-specific MLLMs--such as those trained for mathematics or code--remains largely underexplored. To address the fragmentation of knowledge across domain-specialized MLLMs, we propose a unified parameter integration framework that enables modular composition of expert capabilities. Our method is grounded in a novel Compatibility-Aware Parameter Splicing (CAPS) strategy, which leverages both local functional attribution and global information-theoretic signals to guide selective parameter fusion. By extending this mechanism to the low-rank adaptation layer granularity, we ensure efficient integration with minimal inference overhead. Furthermore, we introduce a domain compatibility scoring mechanism that quantifies inter-expert alignment at the activation level and correlates with downstream task utility. This principled fusion protocol allows the final model to synergize heterogeneous expertise while preserving structural modularity. Extensive evaluations across diverse multimodal benchmarks validate the effectiveness of our framework, offering a scalable path toward compositional, domain-adaptive MLLMs.
>
---
#### [replaced 049] Learning-to-Context Slope: Evaluating In-Context Learning Effectiveness Beyond Performance Illusions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.23146v2](http://arxiv.org/pdf/2506.23146v2)**

> **作者:** Dingzriui Wang; Xuanliang Zhang; Keyan Xu; Qingfu Zhu; Wanxiang Che; Yang Deng
>
> **摘要:** In-context learning (ICL) has emerged as an effective approach to enhance the performance of large language models (LLMs). However, its effectiveness varies significantly across models and tasks, posing challenges for practitioners to determine when ICL reliably improves performance. Current evaluation approaches, reliant on performance change after applying ICL, suffer from low reliability, poor attribution, and impracticality in data-insufficient scenarios. We propose the Learning-to-Context Slope (LCS), a novel metric that quantifies ICL effectiveness by modeling the slope between learning gain (loss decrease from demonstrations) and contextual relevance (demonstration-input relevance). LCS addresses key limitations of performance-based metrics: (1) it captures continuous loss changes even when outputs are incorrect, improving reliability; (2) its formulation attributes ICL failures to weak contextual alignment (inability to adapt inputs to demonstrations) or strong output calibration (self-verification of correctness); and (3) it minimizes reliance on labeled data via synthetic evaluation. Extensive experiments demonstrate that LCS strongly correlates with performance improvements in labeled settings and reliably reflects true effectiveness in biased or data-scarce scenarios. Further analysis reveals actionable thresholds for LCS and identifies model capabilities critical to ICL success.
>
---
#### [replaced 050] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03704v3](http://arxiv.org/pdf/2412.03704v3)**

> **作者:** Xiyao Wang; Zhengyuan Yang; Linjie Li; Hongjin Lu; Yuancheng Xu; Chung-Ching Lin; Kevin Lin; Furong Huang; Lijuan Wang
>
> **摘要:** Despite significant advancements in vision-language models (VLMs), there lacks effective approaches to enhance response quality by scaling inference-time computation. This capability is known to be a core step towards the self-improving models in recent large language model studies. In this paper, we present Vision Value Model (VisVM) that can guide VLM inference-time search to generate responses with better visual comprehension. Specifically, VisVM not only evaluates the generated sentence quality in the current search step, but also anticipates the quality of subsequent sentences that may result from the current step, thus providing a long-term value. In this way, VisVM steers VLMs away from generating sentences prone to hallucinations or insufficient detail, thereby producing higher quality responses. Experimental results demonstrate that VisVM-guided search significantly enhances VLMs' ability to generate descriptive captions with richer visual details and fewer hallucinations, compared with greedy decoding and search methods with other visual reward signals. Furthermore, we find that self-training the model with the VisVM-guided captions improve VLM's performance across a wide range of multimodal benchmarks, indicating the potential for developing self-improving VLMs. Our value model and code are available at https://github.com/si0wang/VisVM.
>
---
