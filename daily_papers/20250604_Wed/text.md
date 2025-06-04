# 自然语言处理 cs.CL

- **最新发布 141 篇**

- **更新 149 篇**

## 最新发布

#### [new 001] Literary Evidence Retrieval via Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大模型对文学文本的理解能力。通过构建文学证据检索基准，测试模型根据文学批评生成缺失引文的能力，发现闭源模型表现优于开源模型，但仍存在理解细微文学信号的挑战。**

- **链接: [http://arxiv.org/pdf/2506.03090v1](http://arxiv.org/pdf/2506.03090v1)**

> **作者:** Katherine Thai; Mohit Iyyer
>
> **备注:** ACL 2025
>
> **摘要:** How well do modern long-context language models understand literary fiction? We explore this question via the task of literary evidence retrieval, repurposing the RELiC dataset of That et al. (2022) to construct a benchmark where the entire text of a primary source (e.g., The Great Gatsby) is provided to an LLM alongside literary criticism with a missing quotation from that work. This setting, in which the model must generate the missing quotation, mirrors the human process of literary analysis by requiring models to perform both global narrative reasoning and close textual examination. We curate a high-quality subset of 292 examples through extensive filtering and human verification. Our experiments show that recent reasoning models, such as Gemini Pro 2.5 can exceed human expert performance (62.5% vs. 50% accuracy). In contrast, the best open-weight model achieves only 29.1% accuracy, highlighting a wide gap in interpretive reasoning between open and closed-weight models. Despite their speed and apparent accuracy, even the strongest models struggle with nuanced literary signals and overgeneration, signaling open challenges for applying LLMs to literary analysis. We release our dataset and evaluation code to encourage future work in this direction.
>
---
#### [new 002] Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM
- **分类: cs.CL; cs.AI; cs.IR; 68T50; I.2.7; H.3.3**

- **简介: 该论文属于命名实体识别（NER）任务，旨在解决俄语文化新闻文本中人名识别的问题。作者使用SPbLitGuide数据集，评估了多种模型（如DeepPavlov、RoBERTa、SpaCy及GPT系列大模型）的表现，发现GPT-4o在特定提示下效果最佳，F1值达0.93，展示了大模型在俄语NER任务中的优势和发展潜力。**

- **链接: [http://arxiv.org/pdf/2506.02589v1](http://arxiv.org/pdf/2506.02589v1)**

> **作者:** Maria Levchenko
>
> **摘要:** This paper addresses the challenge of Named Entity Recognition (NER) for person names within the specialized domain of Russian news texts concerning cultural events. The study utilizes the unique SPbLitGuide dataset, a collection of event announcements from Saint Petersburg spanning 1999 to 2019. A comparative evaluation of diverse NER models is presented, encompassing established transformer-based architectures such as DeepPavlov, RoBERTa, and SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4, and GPT-4o. Key findings highlight the superior performance of GPT-4o when provided with specific prompting for JSON output, achieving an F1 score of 0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The research contributes to a deeper understanding of current NER model capabilities and limitations when applied to morphologically rich languages like Russian within the cultural heritage domain, offering insights for researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025) achieves F1=0.94 for both simple and structured prompts, demonstrating rapid progress across model families and simplified deployment requirements.
>
---
#### [new 003] Token and Span Classification for Entity Recognition in French Historical Encyclopedias
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究历史法语百科全书中的命名实体识别（NER），旨在解决非标准化语言、古拼写及嵌套实体带来的挑战。作者比较了多种方法，包括传统模型与基于Transformer的模型，并提出将NER视为词元和片段分类任务。实验基于18世纪法语数据集GeoEDdA，探索低资源场景下的生成模型应用，强调结合符号与神经方法的潜力。**

- **链接: [http://arxiv.org/pdf/2506.02872v1](http://arxiv.org/pdf/2506.02872v1)**

> **作者:** Ludovic Moncla; Hédi Zeghidi
>
> **摘要:** Named Entity Recognition (NER) in historical texts presents unique challenges due to non-standardized language, archaic orthography, and nested or overlapping entities. This study benchmarks a diverse set of NER approaches, ranging from classical Conditional Random Fields (CRFs) and spaCy-based models to transformer-based architectures such as CamemBERT and sequence-labeling models like Flair. Experiments are conducted on the GeoEDdA dataset, a richly annotated corpus derived from 18th-century French encyclopedias. We propose framing NER as both token-level and span-level classification to accommodate complex nested entity structures typical of historical documents. Additionally, we evaluate the emerging potential of few-shot prompting with generative language models for low-resource scenarios. Our results demonstrate that while transformer-based models achieve state-of-the-art performance, especially on nested entities, generative models offer promising alternatives when labeled data are scarce. The study highlights ongoing challenges in historical NER and suggests avenues for hybrid approaches combining symbolic and neural methods to better capture the intricacies of early modern French text.
>
---
#### [new 004] FinS-Pilot: A Benchmark for Online Financial System
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融自然语言处理任务，旨在解决金融领域缺乏专业评估工具的问题。作者构建了FinS-Pilot基准，用于评估在线金融应用中的RAG系统，结合真实金融助手交互数据和实时API信息，覆盖股票分析和宏观经济预测等领域。通过实验验证了其有效性，为金融NLP研究提供了框架和数据集。**

- **链接: [http://arxiv.org/pdf/2506.02037v1](http://arxiv.org/pdf/2506.02037v1)**

> **作者:** Feng Wang; Yiding Sun; Jiaxin Mao; Wei Xue; Danqing Xu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across various professional domains, with their performance typically evaluated through standardized benchmarks. However, the development of financial RAG benchmarks has been constrained by data confidentiality issues and the lack of dynamic data integration. To address this issue, we introduces FinS-Pilot, a novel benchmark for evaluating RAG systems in online financial applications. Constructed from real-world financial assistant interactions, our benchmark incorporates both real-time API data and structured text sources, organized through an intent classification framework covering critical financial domains such as equity analysis and macroeconomic forecasting. The benchmark enables comprehensive evaluation of financial assistants' capabilities in handling both static knowledge and time-sensitive market information. Through systematic experiments with multiple Chinese leading LLMs, we demonstrate FinS-Pilot's effectiveness in identifying models suitable for financial applications while addressing the current gap in specialized evaluation tools for the financial domain. Our work contributes both a practical evaluation framework and a curated dataset to advance research in financial NLP systems. The code and dataset are accessible on GitHub\footnote{https://github.com/PhealenWang/financial\_rag\_benchmark}.
>
---
#### [new 005] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，旨在自动发现模型间的细粒度性能差异。现有评估方法不够精细，难以揭示模型优劣的具体情境。论文提出BehaviorBox，利用上下文嵌入技术，自动提取体现性能差异的文本特征，如特定语法结构或标点使用场景，从而更深入理解模型表现差异。**

- **链接: [http://arxiv.org/pdf/2506.02204v1](http://arxiv.org/pdf/2506.02204v1)**

> **作者:** Lindia Tjuatja; Graham Neubig
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Language model evaluation is a daunting task: prompts are brittle, corpus-level perplexities are vague, and the choice of benchmarks are endless. Finding examples that show meaningful, generalizable differences between two LMs is crucial to understanding where one model succeeds and another fails. Can this process be done automatically? In this work, we propose methodology for automated comparison of language models that uses performance-aware contextual embeddings to find fine-grained features of text where one LM outperforms another. Our method, which we name BehaviorBox, extracts coherent features that demonstrate differences with respect to the ease of generation between two LMs. Specifically, BehaviorBox finds features that describe groups of words in fine-grained contexts, such as "conditional 'were' in the phrase 'if you were'" and "exclamation marks after emotional statements", where one model outperforms another within a particular datatset. We apply BehaviorBox to compare models that vary in size, model family, and post-training, and enumerate insights into specific contexts that illustrate meaningful differences in performance which cannot be found by measures such as corpus-level perplexity alone.
>
---
#### [new 006] Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective
- **分类: cs.CL**

- **简介: 该论文分析VAPO框架在长链推理中建模长期价值的理论局限性，属于强化学习与大语言模型交叉任务。旨在揭示其在信用分配、价值函数表示和全局信号转化上的问题，为改进LLM代理提供方向。**

- **链接: [http://arxiv.org/pdf/2506.03038v1](http://arxiv.org/pdf/2506.03038v1)**

> **作者:** Jintian Shao; Yiming Cheng
>
> **摘要:** Reinforcement learning (RL) enhances large language models (LLMs) in complex, long-chain-of-thought (long-CoT) reasoning. The advanced VAPO framework, despite sophisticated mechanisms like Decoupled GAE, theoretically faces fundamental limitations in comprehensively modeling and leveraging deep, long-term value for fine-grained, step-by-step policy guidance in extended reasoning chains. We argue these limitations stem from inherent difficulties in credit assignment, value function representational capacity with temporally abstracted goals, and translating global value signals into local policy improvements, especially with sparse rewards. Our theoretical analysis examines these aspects to illuminate VAPO's boundaries in long-term value modeling, aiming to deepen understanding of current RL for advanced reasoning and suggest future research for more robust LLM agents.
>
---
#### [new 007] Causal Estimation of Tokenisation Bias
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型中分词偏差的因果估计，属于自然语言处理任务。它旨在解决不同分词方式影响模型对字符序列概率分配的问题。通过回归不连续设计，比较相近分词在临界点附近的效应，发现分词选择显著影响模型输出，尤其小模型中某子词存在可使对应字符概率提升17倍。**

- **链接: [http://arxiv.org/pdf/2506.03149v1](http://arxiv.org/pdf/2506.03149v1)**

> **作者:** Pietro Lesci; Clara Meister; Thomas Hofmann; Andreas Vlachos; Tiago Pimentel
>
> **备注:** Published as a conference paper at ACL 2025
>
> **摘要:** Modern language models are typically trained over subword sequences, but ultimately define probabilities over character-strings. Ideally, the choice of the tokeniser -- which maps character-strings to subwords -- should not affect the probability assigned to the underlying character-string; in practice, it does. We define this mismatch as tokenisation bias. In this work, we quantify one particular type of tokenisation bias: the effect of including or not a subword (e.g., $\langle hello \rangle$) in a tokeniser's vocabulary on the probability a trained model assigns to the corresponding characters (i.e., \textit{``hello''}). Estimating this effect is challenging because each model is trained with only one tokeniser. We address this by framing tokenisation bias as a causal effect and estimating it using the regression discontinuity design. Specifically, we exploit the fact that tokenisation algorithms rank subwords and add the first $K$ to a tokeniser's vocabulary, where $K$ is an arbitrary cutoff point. As such, we can estimate a causal effect by comparing similar subwords around this cutoff. Experimentally, we find that tokenisation consistently affects models' outputs across scales, vocabularies, and tokenisers. Notably, a subword's presence in a small model's vocabulary may increase its characters' probability by up to 17 times, highlighting tokenisation as a key design choice in language modelling.
>
---
#### [new 008] IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决印尼多语言环境下大语言模型的安全性问题。作者构建了首个针对印尼语境的人工验证安全数据集IndoSafety，涵盖印尼语及三种地方语言，并提出适应当地文化的评估框架，提升模型在敏感内容生成上的安全性。**

- **链接: [http://arxiv.org/pdf/2506.02573v1](http://arxiv.org/pdf/2506.02573v1)**

> **作者:** Muhammad Falensi Azmi; Muhammad Dehan Al Kautsar; Alfan Farizki Wicaksono; Fajri Koto
>
> **备注:** 25 pages
>
> **摘要:** Although region-specific large language models (LLMs) are increasingly developed, their safety remains underexplored, particularly in culturally diverse settings like Indonesia, where sensitivity to local norms is essential and highly valued by the community. In this work, we present IndoSafety, the first high-quality, human-verified safety evaluation dataset tailored for the Indonesian context, covering five language varieties: formal and colloquial Indonesian, along with three major local languages: Javanese, Sundanese, and Minangkabau. IndoSafety is constructed by extending prior safety frameworks to develop a taxonomy that captures Indonesia's sociocultural context. We find that existing Indonesian-centric LLMs often generate unsafe outputs, particularly in colloquial and local language settings, while fine-tuning on IndoSafety significantly improves safety while preserving task performance. Our work highlights the critical need for culturally grounded safety evaluation and provides a concrete step toward responsible LLM deployment in multilingual settings. Warning: This paper contains example data that may be offensive, harmful, or biased.
>
---
#### [new 009] Investigating the Impact of Word Informativeness on Speech Emotion Recognition
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决如何有效提取携带关键情感信息的语音片段问题。通过利用预训练语言模型计算词的信息量，筛选出语义重要段落，并在这些片段上提取声学特征，以提升情感识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.02239v1](http://arxiv.org/pdf/2506.02239v1)**

> **作者:** Sofoklis Kakouros
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** In emotion recognition from speech, a key challenge lies in identifying speech signal segments that carry the most relevant acoustic variations for discerning specific emotions. Traditional approaches compute functionals for features such as energy and F0 over entire sentences or longer speech portions, potentially missing essential fine-grained variation in the long-form statistics. This research investigates the use of word informativeness, derived from a pre-trained language model, to identify semantically important segments. Acoustic features are then computed exclusively for these identified segments, enhancing emotion recognition accuracy. The methodology utilizes standard acoustic prosodic features, their functionals, and self-supervised representations. Results indicate a notable improvement in recognition performance when features are computed on segments selected based on word informativeness, underscoring the effectiveness of this approach.
>
---
#### [new 010] MidPO: Dual Preference Optimization for Safety and Helpfulness in Large Language Models via a Mixture of Experts Framework
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型的安全与有用性优化任务，旨在解决安全性和有用性难以兼顾的问题。通过提出MidPO框架，结合专家混合模型和动态路由机制，实现两者的自适应平衡，经实验验证效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02460v1](http://arxiv.org/pdf/2506.02460v1)**

> **作者:** Yupeng Qi; Ziyu Lyu; Min Yang; Yanlin Wang; Lu Bai; Lixin Cui
>
> **摘要:** As large language models (LLMs) are increasingly applied across various domains, enhancing safety while maintaining the helpfulness of LLMs has become a critical challenge. Recent studies solve this problem through safety-constrained online preference optimization or safety-constrained offline preference optimization. However, the safety-constrained online methods often suffer from excessive safety, which might reduce helpfulness, while the safety-constrained offline methods perform poorly in adaptively balancing safety and helpfulness. To address these limitations, we propose MidPO, a \textbf{\underline{Mi}}xture of Experts (MoE) framework for safety-helpfulness \textbf{\underline{d}}ual \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization. Firstly, MidPO devises single-preference enhanced direct preference optimization approach to transform the base model into two independent experts, termed safety and helpfulness experts, and fine-tunes the two independent experts for optimal safety or helpfulness performance. Secondly, to achieve an effective balance between safety and helpfulness, MidPO incorporates the two experts into the MoE framework and designs a dynamic routing mechanism to allocate contributions from each expert adaptively. We conduct quantitative and qualitative experiments on three popular datasets to demonstrate the proposed MidPO significantly outperforms state-of-the-art approaches in both safety and helpfulness. The code and models will be released.
>
---
#### [new 011] Pruning for Performance: Efficient Idiom and Metaphor Classification in Low-Resource Konkani Using mBERT
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言（如康卡尼语）中习语和隐喻分类的效率问题。作者提出了一种结合mBERT、双向LSTM和线性分类器的混合模型，并采用基于梯度的注意力头剪枝策略提升效率。在新构建的隐喻数据集上取得78%准确率，在已有习语任务上达83%，验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.02005v1](http://arxiv.org/pdf/2506.02005v1)**

> **作者:** Timothy Do; Pranav Saran; Harshita Poojary; Pranav Prabhu; Sean O'Brien; Vasu Sharma; Kevin Zhu
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** In this paper, we address the persistent challenges that figurative language expressions pose for natural language processing (NLP) systems, particularly in low-resource languages such as Konkani. We present a hybrid model that integrates a pre-trained Multilingual BERT (mBERT) with a bidirectional LSTM and a linear classifier. This architecture is fine-tuned on a newly introduced annotated dataset for metaphor classification, developed as part of this work. To improve the model's efficiency, we implement a gradient-based attention head pruning strategy. For metaphor classification, the pruned model achieves an accuracy of 78%. We also applied our pruning approach to expand on an existing idiom classification task, achieving 83% accuracy. These results demonstrate the effectiveness of attention head pruning for building efficient NLP tools in underrepresented languages.
>
---
#### [new 012] TO-GATE: Clarifying Questions and Summarizing Responses with Trajectory Optimization for Eliciting Human Preference
- **分类: cs.CL**

- **简介: 该论文属于对话系统与偏好获取任务，旨在解决现有方法在生成澄清问题和总结回答时难以确定最优对话路径、易产生无关问题的问题。论文提出了TO-GATE框架，通过轨迹优化提升问题生成效果，包含澄清解析器和总结器两个核心组件，实验表明其性能优于基线方法。**

- **链接: [http://arxiv.org/pdf/2506.02827v1](http://arxiv.org/pdf/2506.02827v1)**

> **作者:** Yulin Dou; Jiangming Liu
>
> **摘要:** Large language models (LLMs) can effectively elicit human preferences through multi-turn dialogue. Complex tasks can be accomplished through iterative clarifying questions and final responses generated by an LLM acting as a questioner (STaR-GATE; Andukuri et al., 2024}). However, existing approaches based on self-taught reasoning struggle to identify optimal dialogue trajectories and avoid irrelevant questions to the tasks. To address this limitation, we propose TO-GATE, a novel framework that enhances question generation through trajectory optimization, which consists of two key components: a clarification resolver that generates optimal questioning trajectories, and a summarizer that ensures task-aligned final responses. The trajectory optimization enables the model to produce effective elicitation questions and summary responses tailored to specific tasks. Experimental results demonstrate that TO-GATE significantly outperforms baseline methods, achieving a 9.32% improvement on standard preference elicitation tasks.
>
---
#### [new 013] Beyond the Surface: Measuring Self-Preference in LLM Judgments
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLM）作为评判者时存在的自偏偏好问题。现有方法易将自偏偏好与响应质量混淆，作者引入“黄金判断”和DBG评分来更准确地衡量偏差，并探索影响因素及机制。**

- **链接: [http://arxiv.org/pdf/2506.02592v1](http://arxiv.org/pdf/2506.02592v1)**

> **作者:** Zhi-Yuan Chen; Hao Wang; Xinyu Zhang; Enrui Hu; Yankai Lin
>
> **摘要:** Recent studies show that large language models (LLMs) exhibit self-preference bias when serving as judges, meaning they tend to favor their own responses over those generated by other models. Existing methods typically measure this bias by calculating the difference between the scores a judge model assigns to its own responses and those it assigns to responses from other models. However, this approach conflates self-preference bias with response quality, as higher-quality responses from the judge model may also lead to positive score differences, even in the absence of bias. To address this issue, we introduce gold judgments as proxies for the actual quality of responses and propose the DBG score, which measures self-preference bias as the difference between the scores assigned by the judge model to its own responses and the corresponding gold judgments. Since gold judgments reflect true response quality, the DBG score mitigates the confounding effect of response quality on bias measurement. Using the DBG score, we conduct comprehensive experiments to assess self-preference bias across LLMs of varying versions, sizes, and reasoning abilities. Additionally, we investigate two factors that influence and help alleviate self-preference bias: response text style and the post-training data of judge models. Finally, we explore potential underlying mechanisms of self-preference bias from an attention-based perspective. Our code and data are available at https://github.com/zhiyuanc2001/self-preference.
>
---
#### [new 014] Should LLM Safety Be More Than Refusing Harmful Instructions?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型（LLM）在处理长尾分布和加密文本时的安全性问题。研究提出一个二维安全评估框架，包括指令拒绝与生成安全，并通过实验揭示具备解密能力的模型易受攻击的问题，进一步探讨现有防护机制的优劣，为提升LLM安全性提供方向。**

- **链接: [http://arxiv.org/pdf/2506.02442v1](http://arxiv.org/pdf/2506.02442v1)**

> **作者:** Utsav Maskey; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms.
>
---
#### [new 015] CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态问答任务，旨在解决多模态RAG中的知识不一致问题（参数与检索知识、图文知识之间）。提出CoRe-MMRAG框架，通过四阶段流程融合内部与外部知识，并设计训练策略提升效果。在KB-VQA数据集上表现优于基线方法。**

- **链接: [http://arxiv.org/pdf/2506.02544v1](http://arxiv.org/pdf/2506.02544v1)**

> **作者:** Yang Tian; Fan Liu; Jingyuan Zhang; Victoria W.; Yupeng Hu; Liqiang Nie
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MMRAG) has been introduced to enhance Multimodal Large Language Models by incorporating externally retrieved multimodal knowledge, but it introduces two challenges: Parametric-Retrieved Knowledge Inconsistency (PRKI), where discrepancies between parametric and retrieved knowledge create uncertainty in determining reliability, and Visual-Textual Knowledge Inconsistency (VTKI), where misalignment between visual and textual sources disrupts entity representation. To address these challenges, we propose \textbf{C}r\textbf{o}ss-source knowledge \textbf{Re}conciliation for \textbf{M}ulti\textbf{M}odal \textbf{RAG} (CoRe-MMRAG), a novel end-to-end framework that effectively reconciles inconsistencies across knowledge sources. CoRe-MMRAG follows a four-stage pipeline: it first generates an internal response from parametric knowledge, then selects the most relevant multimodal evidence via joint similarity assessment, generates an external response, and finally integrates both to produce a reliable answer. Additionally, a specialized training paradigm enhances knowledge source discrimination, multimodal integration, and unified answer generation. Experiments on KB-VQA benchmarks show that CoRe-MMRAG achieves substantial improvements over baseline methods, achieving 5.6\% and 9.3\% performance gains on InfoSeek and Encyclopedic-VQA, respectively. We release code and data at \href{https://github.com/TyangJN/CoRe-MMRAG}{https://github.com/TyangJN/CoRe-MMRAG}.
>
---
#### [new 016] Pruning General Large Language Models into Customized Expert Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型参数冗余、计算资源消耗大的问题。提出了一种定制化剪枝方法Cus-Prun，沿“语言”、“领域”和“任务”维度剪枝，生成无需后训练的轻量专家模型，兼顾专家与通用能力。**

- **链接: [http://arxiv.org/pdf/2506.02561v1](http://arxiv.org/pdf/2506.02561v1)**

> **作者:** Yirao Zhao; Guizhen Chen; Kenji Kawaguchi; Lidong Bing; Wenxuan Zhang
>
> **摘要:** Large language models (LLMs) have revolutionized natural language processing, yet their substantial model sizes often require substantial computational resources. To preserve computing resources and accelerate inference speed, it is crucial to prune redundant parameters, especially for experienced users who often need compact expert models tailored to specific downstream scenarios. However, most existing pruning methods focus on preserving the model's general capabilities, often requiring extensive post-training or suffering from degraded performance due to coarse-grained pruning. In this work, we design a $\underline{Cus}$tom $\underline{Prun}$ing method ($\texttt{Cus-Prun}$) to prune a large general model into a smaller lightweight expert model, which is positioned along the "language", "domain" and "task" dimensions. By identifying and pruning irrelevant neurons of each dimension, $\texttt{Cus-Prun}$ creates expert models without any post-training. Our experiments demonstrate that $\texttt{Cus-Prun}$ consistently outperforms other methods, achieving minimal loss in both expert and general capabilities across various models from different model families and sizes.
>
---
#### [new 017] Towards a Japanese Full-duplex Spoken Dialogue System
- **分类: cs.CL; eess.AS**

- **简介: 该论文旨在开发首个支持日语的全双工语音对话系统，解决现有研究在日语方面不足的问题。作者基于英文模型Moshi，通过两阶段训练（预训练和微调）及合成数据增强，提升了模型在自然性和意义性方面的表现。**

- **链接: [http://arxiv.org/pdf/2506.02979v1](http://arxiv.org/pdf/2506.02979v1)**

> **作者:** Atsumoto Ohashi; Shinya Iizuka; Jingjing Jiang; Ryuichiro Higashinaka
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Full-duplex spoken dialogue systems, which can model simultaneous bidirectional features of human conversations such as speech overlaps and backchannels, have attracted significant attention recently. However, the study of full-duplex spoken dialogue systems for the Japanese language has been limited, and the research on their development in Japanese remains scarce. In this paper, we present the first publicly available full-duplex spoken dialogue model in Japanese, which is built upon Moshi, a full-duplex dialogue model in English. Our model is trained through a two-stage process: pre-training on a large-scale spoken dialogue data in Japanese, followed by fine-tuning on high-quality stereo spoken dialogue data. We further enhance the model's performance by incorporating synthetic dialogue data generated by a multi-stream text-to-speech system. Evaluation experiments demonstrate that the trained model outperforms Japanese baseline models in both naturalness and meaningfulness.
>
---
#### [new 018] SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决VLM在识别光学错觉和隐藏内容中的语义理解缺陷。作者构建了包含112张图像的HC-Bench基准测试，提出SemVink方法，通过缩小图像分辨率提升VLM准确率至99%以上，揭示其对低级视觉操作的缺失问题，并倡导多尺度处理模型的发展。**

- **链接: [http://arxiv.org/pdf/2506.02803v1](http://arxiv.org/pdf/2506.02803v1)**

> **作者:** Sifan Li; Yujun Cai; Yiwei Wang
>
> **摘要:** Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden text, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0-5.36%)-even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions (32-128 pixels), which unlocks >99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond.
>
---
#### [new 019] Quantitative LLM Judges
- **分类: cs.CL; cs.LG**

- **简介: 论文提出“定量LLM评委”框架，属于自然语言处理任务。旨在解决LLM自动评估结果与人类评分一致性不足问题。通过回归模型将现有LLM评委的评价和分数映射到人类评分，提升预测能力。展示了四种适用于不同反馈类型的定量评委，验证了其有效性与效率。**

- **链接: [http://arxiv.org/pdf/2506.02945v1](http://arxiv.org/pdf/2506.02945v1)**

> **作者:** Aishwarya Sahoo; Jeevana Kruthi Karnuthala; Tushar Parmanand Budhwani; Pranchal Agarwal; Sankaran Vaidyanathan; Alexa Siu; Franck Dernoncourt; Jennifer Healey; Nedim Lipka; Ryan Rossi; Uttaran Bhattacharya; Branislav Kveton
>
> **摘要:** LLM-as-a-judge is a framework in which a large language model (LLM) automatically evaluates the output of another LLM. We propose quantitative LLM judges, which align evaluation scores of existing LLM judges to human scores in a given domain using regression models. The models are trained to improve the score of the original judge by using the judge's textual evaluation and score. We present four quantitative judges for different types of absolute and relative feedback, which showcases the generality and versatility of our framework. Our framework is more computationally efficient than supervised fine-tuning and can be more statistically efficient when human feedback is limited, which is expected in most applications of our work. We validate these claims empirically on four datasets using two base judges. Our experiments show that quantitative judges can effectively improve the predictive power of existing judges through post-hoc modeling.
>
---
#### [new 020] Answer Convergence as a Signal for Early Stopping in Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理过程冗长、成本高的问题。通过研究推理步骤的必要性，提出三种提前终止策略，减少冗余推理，提升效率。实验表明这些方法能显著降低计算成本且不损害准确性。**

- **链接: [http://arxiv.org/pdf/2506.02536v1](http://arxiv.org/pdf/2506.02536v1)**

> **作者:** Xin Liu; Lu Wang
>
> **摘要:** Chain-of-thought (CoT) prompting enhances reasoning in large language models (LLMs) but often leads to verbose and redundant outputs, thus increasing inference cost. We hypothesize that many reasoning steps are unnecessary for producing correct answers. To investigate this, we start with a systematic study to examine what is the minimum reasoning required for a model to reach a stable decision. We find that on math reasoning tasks like math, models typically converge to their final answers after 60\% of the reasoning steps, suggesting substantial redundancy in the remaining content. Based on these insights, we propose three inference-time strategies to improve efficiency: (1) early stopping via answer consistency, (2) boosting the probability of generating end-of-reasoning signals, and (3) a supervised method that learns when to stop based on internal activations. Experiments across five benchmarks and five open-weights LLMs show that our methods significantly reduce token usage with little or no accuracy drop. In particular, on NaturalQuestions, Answer Consistency reduces tokens by over 40\% while further improving accuracy. Our work underscores the importance of cost-effective reasoning methods that operate at inference time, offering practical benefits for real-world applications.
>
---
#### [new 021] Different Speech Translation Models Encode and Translate Speaker Gender Differently
- **分类: cs.CL**

- **简介: 该论文属于语音翻译任务，旨在探究不同语音翻译模型如何编码和翻译说话者性别。通过可解释性分析方法，作者发现传统模型能捕捉性别信息，而新架构（含适配器）则不能，导致倾向于男性默认的翻译偏差。**

- **链接: [http://arxiv.org/pdf/2506.02172v1](http://arxiv.org/pdf/2506.02172v1)**

> **作者:** Dennis Fucci; Marco Gaido; Matteo Negri; Luisa Bentivogli; Andre Martins; Giuseppe Attanasio
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Recent studies on interpreting the hidden states of speech models have shown their ability to capture speaker-specific features, including gender. Does this finding also hold for speech translation (ST) models? If so, what are the implications for the speaker's gender assignment in translation? We address these questions from an interpretability perspective, using probing methods to assess gender encoding across diverse ST models. Results on three language directions (English-French/Italian/Spanish) indicate that while traditional encoder-decoder models capture gender information, newer architectures -- integrating a speech encoder with a machine translation system via adapters -- do not. We also demonstrate that low gender encoding capabilities result in systems' tendency toward a masculine default, a translation bias that is more pronounced in newer architectures.
>
---
#### [new 022] Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs
- **分类: cs.CL**

- **简介: 该论文研究个性化大语言模型（LLM）在不同任务中保持预设角色一致性的能力。属于自然语言处理任务中的模型一致性分析。为解决LLM在多任务下角色一致性不足的问题，作者构建了包含四个角色类别和五种任务的评估框架，分析一致性受角色设定、刻板印象及模型设计的影响，并发现结构化任务和更多上下文有助于提升一致性。**

- **链接: [http://arxiv.org/pdf/2506.02659v1](http://arxiv.org/pdf/2506.02659v1)**

> **作者:** Manon Reusens; Bart Baesens; David Jurgens
>
> **摘要:** Personalized Large Language Models (LLMs) are increasingly used in diverse applications, where they are assigned a specific persona - such as a happy high school teacher - to guide their responses. While prior research has examined how well LLMs adhere to predefined personas in writing style, a comprehensive analysis of consistency across different personas and task types is lacking. In this paper, we introduce a new standardized framework to analyze consistency in persona-assigned LLMs. We define consistency as the extent to which a model maintains coherent responses when assigned the same persona across different tasks and runs. Our framework evaluates personas across four different categories (happiness, occupation, personality, and political stance) spanning multiple task dimensions (survey writing, essay generation, social media post generation, single turn, and multi-turn conversations). Our findings reveal that consistency is influenced by multiple factors, including the assigned persona, stereotypes, and model design choices. Consistency also varies across tasks, increasing with more structured tasks and additional context. All code is available on GitHub.
>
---
#### [new 023] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态生成评估任务，旨在解决现有评估模型忽视文本到图像生成任务及缺乏大规模人类评估数据的问题。作者构建了包含双向生成任务数据的Minos-Corpus，提出数据选择与平衡、Mix-SFT训练方法，并应用DPO优化训练，最终开发出性能领先的多模态评估模型Minos。**

- **链接: [http://arxiv.org/pdf/2506.02494v1](http://arxiv.org/pdf/2506.02494v1)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xinyu Hu; Li Lin; Mingqi Gao; Shi Qiu; Xiaojun Wan
>
> **摘要:** Evaluation is important for multimodal generation tasks. With the rapid progress of MLLMs, there is growing interest in applying MLLMs to build general evaluation systems. However, existing work overlooks two aspects: (1) the development of evaluation capabilities for text-to-image (T2I) generation task, and (2) the incorporation of large-scale human evaluation data. In this paper, we introduce Minos-Corpus, a large-scale multimodal evaluation dataset that combines evaluation data from both human and GPT. The corpus contains evaluation data across both image-to-text(I2T) and T2I generation tasks. Based on this corpus, we propose Data Selection and Balance, Mix-SFT training methods, and apply DPO to develop Minos, a multimodal evaluation model built upon a 7B backbone. Minos achieves state-of-the-art (SoTA) performance among all open-source evaluation models of similar scale on the average of evaluation performance on all tasks, and outperforms all open-source and closed-source models on evaluation of T2I generation task. Extensive experiments demonstrate the importance of leveraging high-quality human evaluation data and jointly training on evaluation data from both I2T and T2I generation tasks.
>
---
#### [new 024] DYNAC: Dynamic Vocabulary based Non-Autoregressive Contextualization for Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文属于语音识别任务，旨在解决上下文偏置建模中推理速度慢和动态词汇依赖建模不足的问题。工作提出DYNAC方法，将动态词汇集成到CTC模型中间层，实现编码器自条件化，有效捕捉静态与动态token依赖，显著提升推理效率。**

- **链接: [http://arxiv.org/pdf/2506.00422v1](http://arxiv.org/pdf/2506.00422v1)**

> **作者:** Yui Sudo; Yosuke Fukumoto; Muhammad Shakeel; Yifan Peng; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Contextual biasing (CB) improves automatic speech recognition for rare and unseen phrases. Recent studies have introduced dynamic vocabulary, which represents context phrases as expandable tokens in autoregressive (AR) models. This method improves CB accuracy but with slow inference speed. While dynamic vocabulary can be applied to non-autoregressive (NAR) models, such as connectionist temporal classification (CTC), the conditional independence assumption fails to capture dependencies between static and dynamic tokens. This paper proposes DYNAC (Dynamic Vocabulary-based NAR Contextualization), a self-conditioned CTC method that integrates dynamic vocabulary into intermediate layers. Conditioning the encoder on dynamic vocabulary, DYNAC effectively captures dependencies between static and dynamic tokens while reducing the real-time factor (RTF). Experimental results show that DYNAC reduces RTF by 81% with a 0.1-point degradation in word error rate on the LibriSpeech 960 test-clean set.
>
---
#### [new 025] KARE-RAG: Knowledge-Aware Refinement and Enhancement for RAG
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决RAG中文档噪声导致的事实不一致问题。作者提出KARE-RAG，通过结构化知识表示、密集直接偏好优化（DDPO）和对比数据生成策略，提升模型对噪声内容的处理能力，从而增强RAG在多任务下的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.02503v1](http://arxiv.org/pdf/2506.02503v1)**

> **作者:** Yongjian Li; HaoCheng Chu; Yukun Yan; Zhenghao Liu; Shi Yu; Zheni Zeng; Ruobing Wang; Sen Song; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access broader knowledge sources, yet factual inconsistencies persist due to noise in retrieved documents-even with advanced retrieval methods. We demonstrate that enhancing generative models' capacity to process noisy content is equally critical for robust performance. In this paper, we present KARE-RAG (Knowledge-Aware Refinement and Enhancement for RAG), which improves knowledge utilization through three key innovations: (1) structured knowledge representations that facilitate error detection during training, (2) Dense Direct Preference Optimization (DDPO)-a refined training objective that prioritizes correction of critical errors, and (3) a contrastive data generation pipeline that maintains semantic consistency while rectifying factual inaccuracies. Experiments show our method significantly enhances standard RAG pipelines across model scales, improving both in-domain and out-of-domain task performance without compromising general capabilities. Notably, these gains are achieved with modest training data, suggesting data-efficient optimization is possible through targeted learning strategies. Our findings establish a new direction for RAG improvement: by improving how models learn to process retrieved content, we can enhance performance across diverse inference paradigms. All data and code will be publicly available on Github.
>
---
#### [new 026] ChatCFD: an End-to-End CFD Agent with Domain-specific Structured Thinking
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与计算流体动力学（CFD）结合的交叉任务，旨在解决CFD仿真流程复杂、依赖专家知识的问题。作者提出了ChatCFD，一个基于大语言模型的端到端系统，能在OpenFOAM框架内自动执行CFD仿真工作流，用户仅需提供自然语言描述或文献信息即可完成复杂配置，提升了自动化水平和适应性。**

- **链接: [http://arxiv.org/pdf/2506.02019v1](http://arxiv.org/pdf/2506.02019v1)**

> **作者:** E Fan; Weizong Wang; Tianhan Zhang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Computational Fluid Dynamics (CFD) is essential for scientific and engineering advancements but is limited by operational complexity and the need for extensive expertise. This paper presents ChatCFD, a large language model-driven pipeline that automates CFD workflows within the OpenFOAM framework. It enables users to configure and execute complex simulations from natural language prompts or published literature with minimal expertise. The innovation is its structured approach to database construction, configuration validation, and error reflection, integrating CFD and OpenFOAM knowledge with general language models to improve accuracy and adaptability. Validation shows ChatCFD can autonomously reproduce published CFD results, handling complex, unseen configurations beyond basic examples, a task challenging for general language models.
>
---
#### [new 027] FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于联邦学习与大语言模型领域，旨在解决数据隐私和领域适配问题。作者构建了FlowerTune基准测试平台，评估跨领域联邦微调性能，涵盖自然语言处理、金融、医疗和编程四大领域，比较26种预训练模型在不同策略下的表现，推动隐私保护与专业化模型发展。**

- **链接: [http://arxiv.org/pdf/2506.02961v1](http://arxiv.org/pdf/2506.02961v1)**

> **作者:** Yan Gao; Massimo Roberto Scamarcia; Javier Fernandez-Marques; Mohammad Naseri; Chong Shen Ng; Dimitris Stripelis; Zexi Li; Tao Shen; Jiamu Bai; Daoyuan Chen; Zikai Zhang; Rui Hu; InSeo Song; Lee KangYoon; Hong Jia; Ting Dang; Junyan Wang; Zheyuan Liu; Daniel Janes Beutel; Lingjuan Lyu; Nicholas D. Lane
>
> **摘要:** Large Language Models (LLMs) have achieved state-of-the-art results across diverse domains, yet their development remains reliant on vast amounts of publicly available data, raising concerns about data scarcity and the lack of access to domain-specific, sensitive information. Federated Learning (FL) presents a compelling framework to address these challenges by enabling decentralized fine-tuning on pre-trained LLMs without sharing raw data. However, the compatibility and performance of pre-trained LLMs in FL settings remain largely under explored. We introduce the FlowerTune LLM Leaderboard, a first-of-its-kind benchmarking suite designed to evaluate federated fine-tuning of LLMs across four diverse domains: general NLP, finance, medical, and coding. Each domain includes federated instruction-tuning datasets and domain-specific evaluation metrics. Our results, obtained through a collaborative, open-source and community-driven approach, provide the first comprehensive comparison across 26 pre-trained LLMs with different aggregation and fine-tuning strategies under federated settings, offering actionable insights into model performance, resource constraints, and domain adaptation. This work lays the foundation for developing privacy-preserving, domain-specialized LLMs for real-world applications.
>
---
#### [new 028] ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在减少大语言模型（LLM）的参数量。它通过使用结构化矩阵和正交变换提高权重的可压缩性，从而在不显著影响性能的前提下降低模型的计算与内存需求。**

- **链接: [http://arxiv.org/pdf/2506.02818v1](http://arxiv.org/pdf/2506.02818v1)**

> **作者:** Ekaterina Grishina; Mikhail Gorbunov; Maxim Rakhuba
>
> **备注:** Accepted by ACL Findings
>
> **摘要:** Large language models (LLMs) demonstrate impressive results in natural language processing tasks but require a significant amount of computational and memory resources. Structured matrix representations are a promising way for reducing the number of parameters of these models. However, it seems unrealistic to expect that weight matrices of pretrained models can be accurately represented by structured matrices without any fine-tuning. To overcome this issue, we utilize the fact that LLM output is invariant under certain orthogonal transformations of weight matrices. This insight can be leveraged to identify transformations that significantly improve the compressibility of weights within structured classes. The proposed approach is applicable to various types of structured matrices that support efficient projection operations. Code is available at https://github.com/GrishKate/ProcrustesGPT
>
---
#### [new 029] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了EvaLearn，一个用于评估大语言模型（LLMs）学习能力和效率的基准。任务是通过序列问题求解，衡量模型在不同任务中利用经验提升性能的能力。论文构建了包含648个问题的基准测试，提供五项自动化指标，并分析九个前沿模型的表现，揭示静态能力强者未必具备更强学习能力，推动对模型动态学习特性的研究。**

- **链接: [http://arxiv.org/pdf/2506.02672v1](http://arxiv.org/pdf/2506.02672v1)**

> **作者:** Shihan Dou; Ming Zhang; Chenhao Huang; Jiayi Chen; Feng Chen; Shichun Liu; Yan Liu; Chenxiao Liu; Cheng Zhong; Zongzhang Zhang; Tao Gui; Chao Xin; Wei Chengzhi; Lin Yan; Qi Zhang; Xuanjing Huang
>
> **备注:** 47 pages, 24 figures
>
> **摘要:** We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository.
>
---
#### [new 030] Research on Medical Named Entity Identification Based On Prompt-Biomrc Model and Its Application in Intelligent Consultation System
- **分类: cs.CL**

- **简介: 该论文属于医疗领域的命名实体识别（NER）任务，旨在提升医学文本中实体识别的精度与效率。为解决传统模型效果不足的问题，作者提出了Prompt-bioMRC模型，融合硬模板与软提示设计，并验证其在多个医疗数据集上的优越性能，为智能诊断系统提供技术支持。**

- **链接: [http://arxiv.org/pdf/2506.01961v1](http://arxiv.org/pdf/2506.01961v1)**

> **作者:** Jinzhu Yang
>
> **摘要:** This study is dedicated to exploring the application of prompt learning methods to advance Named Entity Recognition (NER) within the medical domain. In recent years, the emergence of large-scale models has driven significant progress in NER tasks, particularly with the introduction of the BioBERT language model, which has greatly enhanced NER capabilities in medical texts. Our research introduces the Prompt-bioMRC model, which integrates both hard template and soft prompt designs aimed at refining the precision and efficiency of medical entity recognition. Through extensive experimentation across diverse medical datasets, our findings consistently demonstrate that our approach surpasses traditional models. This enhancement not only validates the efficacy of our methodology but also highlights its potential to provide reliable technological support for applications like intelligent diagnosis systems. By leveraging advanced NER techniques, this study contributes to advancing automated medical data processing, facilitating more accurate medical information extraction, and supporting efficient healthcare decision-making processes.
>
---
#### [new 031] On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures
- **分类: cs.CL**

- **简介: 论文研究大语言模型在不同文化测量系统下的泛化能力，探讨其默认使用何种系统、跨系统表现是否稳定及能否通过推理缓解问题。任务属自然语言处理中的跨文化适应性评估。工作包括构建新数据集、实证分析七模型表现，发现模型倾向主流系统，转换影响准确性，且需更多计算资源应对非主流文化。**

- **链接: [http://arxiv.org/pdf/2506.02591v1](http://arxiv.org/pdf/2506.02591v1)**

> **作者:** Minh Duc Bui; Kyung Eun Park; Goran Glavaš; Fabian David Schmidt; Katharina von der Wense
>
> **备注:** Accepted to ACL 2025 Main (Camera-Ready Version)
>
> **摘要:** Measurement systems (e.g., currencies) differ across cultures, but the conversions between them are well defined so that humans can state facts using any measurement system of their choice. Being available to users from diverse cultural backgrounds, large language models (LLMs) should also be able to provide accurate information irrespective of the measurement system at hand. Using newly compiled datasets we test if this is the case for seven open-source LLMs, addressing three key research questions: (RQ1) What is the default system used by LLMs for each type of measurement? (RQ2) Do LLMs' answers and their accuracy vary across different measurement systems? (RQ3) Can LLMs mitigate potential challenges w.r.t. underrepresented systems via reasoning? Our findings show that LLMs default to the measurement system predominantly used in the data. Additionally, we observe considerable instability and variance in performance across different measurement systems. While this instability can in part be mitigated by employing reasoning methods such as chain-of-thought (CoT), this implies longer responses and thereby significantly increases test-time compute (and inference costs), marginalizing users from cultural backgrounds that use underrepresented measurement systems.
>
---
#### [new 032] IP-Dialog: Evaluating Implicit Personalization in Dialogue Systems with Synthetic Data
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于对话系统中的隐式个性化评估任务，旨在解决因缺乏高质量数据而导致的个性化能力评估与改进难题。作者提出了基于合成数据的IP-Dialog基准和训练集，涵盖10项任务和12种用户属性，并构建了包含四个指标的评估框架及五个因果图模型以分析推理路径，从而全面评估模型的个性化能力。**

- **链接: [http://arxiv.org/pdf/2506.02449v1](http://arxiv.org/pdf/2506.02449v1)**

> **作者:** Bo Peng; Zhiheng Wang; Heyang Gong; Chaochao Lu
>
> **摘要:** In modern dialogue systems, the ability to implicitly infer user backgrounds from conversations and leverage this information for personalized assistance is crucial. However, the scarcity of high-quality data remains a fundamental challenge to evaluating and improving this capability. Traditional dataset construction methods are labor-intensive, resource-demanding, and raise privacy concerns. To address these issues, we propose a novel approach for automatic synthetic data generation and introduce the Implicit Personalized Dialogue (IP-Dialog) benchmark along with a training dataset, covering 10 tasks and 12 user attribute types. Additionally, we develop a systematic evaluation framework with four metrics to assess both attribute awareness and reasoning capabilities. We further propose five causal graphs to elucidate models' reasoning pathways during implicit personalization. Extensive experiments yield insightful observations and prove the reliability of our dataset.
>
---
#### [new 033] Consultant Decoding: Yet Another Synergistic Mechanism
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理加速中因高拒绝率导致效率低下的问题。作者提出了一种新的协同机制“顾问解码”（CD），通过使用LLM自身计算的token级似然进行验证，减少了对大模型的调用频率，在保持生成质量的同时显著提升了推理速度。**

- **链接: [http://arxiv.org/pdf/2506.02391v1](http://arxiv.org/pdf/2506.02391v1)**

> **作者:** Chuanghao Ding; Jiaping Wang; Ziqing Yang; Xiaoliang Wang; Dahua Lin; Cam-Tu Nguyen; Fei Tan
>
> **备注:** ACL 2025 findings
>
> **摘要:** The synergistic mechanism based on Speculative Decoding (SD) has garnered considerable attention as a simple yet effective approach for accelerating the inference of large language models (LLMs). Nonetheless, the high rejection rates require repeated LLMs calls to validate draft tokens, undermining the overall efficiency gain of SD. In this work, we revisit existing verification mechanisms and propose a novel synergetic mechanism Consultant Decoding (CD). Unlike SD, which relies on a metric derived from importance sampling for verification, CD verifies candidate drafts using token-level likelihoods computed solely by the LLM. CD achieves up to a 2.5-fold increase in inference speed compared to the target model, while maintaining comparable generation quality (around 100% of the target model's performance). Interestingly, this is achieved by combining models whose parameter sizes differ by two orders of magnitude. In addition, CD reduces the call frequency of the large target model to below 10%, particularly in more demanding tasks. CD's performance was even found to surpass that of the large target model, which theoretically represents the upper bound for speculative decoding.
>
---
#### [new 034] Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection
- **分类: cs.CL**

- **简介: 该论文属于 misinformation detection 任务，旨在解决模型依赖表面特征（shortcut learning）导致泛化能力差的问题。作者提出了 TruthOverTricks 框架评估 shortcut 学习，并构建新数据集 NQ-Misinfo 和 Streaming-Misinfo。他们还提出 SMF 方法，通过 LLM 增强的数据扩增缓解模型对捷径特征的依赖，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.02350v1](http://arxiv.org/pdf/2506.02350v1)**

> **作者:** Herun Wan; Jiaying Wu; Minnan Luo; Zhi Zeng; Zhixiong Su
>
> **摘要:** Misinformation detection models often rely on superficial cues (i.e., \emph{shortcuts}) that correlate with misinformation in training data but fail to generalize to the diverse and evolving nature of real-world misinformation. This issue is exacerbated by large language models (LLMs), which can easily generate convincing misinformation through simple prompts. We introduce TruthOverTricks, a unified evaluation paradigm for measuring shortcut learning in misinformation detection. TruthOverTricks categorizes shortcut behaviors into intrinsic shortcut induction and extrinsic shortcut injection, and evaluates seven representative detectors across 14 popular benchmarks, along with two new factual misinformation datasets, NQ-Misinfo and Streaming-Misinfo. Empirical results reveal that existing detectors suffer severe performance degradation when exposed to both naturally occurring and adversarially crafted shortcuts. To address this, we propose SMF, an LLM-augmented data augmentation framework that mitigates shortcut reliance through paraphrasing, factual summarization, and sentiment normalization. SMF consistently enhances robustness across 16 benchmarks, encouraging models to rely on deeper semantic understanding rather than shortcut cues. To promote the development of misinformation detectors, we have published the resources publicly at https://github.com/whr000001/TruthOverTricks.
>
---
#### [new 035] RACE-Align: Retrieval-Augmented and Chain-of-Thought Enhanced Preference Alignment for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.6; H.3.3**

- **简介: 该论文属于大型语言模型（LLM）的偏好对齐任务，旨在解决LLM在垂直领域中准确性、领域推理和可解释性不足的问题。作者提出RACE-Align框架，通过结合知识检索与链式思维推理构建高质量偏好数据，并使用DPO算法提升模型表现，尤其在中医领域验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2506.02726v1](http://arxiv.org/pdf/2506.02726v1)**

> **作者:** Qihang Yan; Xinyu Zhang; Luming Guo; Qi Zhang; Feifan Liu
>
> **摘要:** Large Language Models (LLMs) struggle with accuracy, domain-specific reasoning, and interpretability in vertical domains. Traditional preference alignment methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) often overlook the underlying knowledge sources and reasoning logic. This paper introduces RACE-Align (Retrieval-Augmented and Chain-of-Thought Enhanced Alignment), a novel framework designed to address these limitations. RACE-Align systematically constructs a binary preference dataset incorporating external knowledge support and explicit Chain-of-Thought (CoT) reasoning, then aligns LLMs using the DPO algorithm. The core innovation lies in its preference data construction strategy: it integrates AI-driven retrieval for factual grounding, enhancing knowledgeability and accuracy, and emphasizes the optimization of domain-specific CoT, treating the reasoning process itself as a key preference dimension. A multi-stage, AI-driven refinement pipeline cost-effectively generates these preference pairs. Experimental validation in Traditional Chinese Medicine (TCM) using Qwen3-1.7B as the base model demonstrates that RACE-Align significantly outperforms the original base model and a model fine-tuned only with Supervised Fine-Tuning (SFT). Improvements were observed across multiple dimensions, including answer accuracy, information richness, application of TCM thinking patterns, logicality and depth of reasoning, and interpretability. These findings suggest RACE-Align offers an effective pathway to enhance LLMs' knowledge application, reasoning reliability, and process transparency in complex vertical domains.
>
---
#### [new 036] BabyLM's First Constructions: Causal interventions provide a signal of learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与语言习得交叉任务，旨在探讨模型在符合儿童学习数据量的条件下是否能有效学习语言构造。研究者使用Rozner等的方法，评估BabyLM挑战中的模型，发现即使训练数据量合理，模型仍能表示多样构造，且构造表现与任务性能相关，表明其对语言学习具功能意义。**

- **链接: [http://arxiv.org/pdf/2506.02147v1](http://arxiv.org/pdf/2506.02147v1)**

> **作者:** Joshua Rozner; Leonie Weissweiler; Cory Shain
>
> **摘要:** Construction grammar posits that children acquire constructions (form-meaning pairings) from the statistics of their environment. Recent work supports this hypothesis by showing sensitivity to constructions in pretrained language models (PLMs), including one recent study (Rozner et al., 2025) demonstrating that constructions shape the PLM's output distribution. However, models under study have generally been trained on developmentally implausible amounts of data, casting doubt on their relevance to human language learning. Here we use Rozner et al.'s methods to evaluate constructional learning in models from the 2024 BabyLM challenge. Our results show that even when trained on developmentally plausible quantities of data, models represent diverse constructions, even hard cases that are superficially indistinguishable. We further find correlational evidence that constructional performance may be functionally relevant: models that better represent constructions perform better on the BabyLM benchmarks.
>
---
#### [new 037] MASTER: Enhancing Large Language Model via Multi-Agent Simulated Teaching
- **分类: cs.CL**

- **简介: 论文提出了一种名为MASTER的数据增强方法，用于解决大语言模型在指令微调中缺乏高质量数据的问题。通过多智能体模拟教学场景生成高质量对话数据，并构建了微调数据集BOOST-QA。实验表明，使用该数据集微调的模型在多个基准任务上表现出色，尤其提升了复杂任务的推理能力。属于自然语言处理中的模型微调与数据增强任务。**

- **链接: [http://arxiv.org/pdf/2506.02689v1](http://arxiv.org/pdf/2506.02689v1)**

> **作者:** Liang Yue; Yihong Tang; Kehai Chen; Jie Liu; Min Zhang
>
> **摘要:** Instruction fine-tuning is crucial in NLP tasks, enhancing pretrained models' instruction-following capabilities and task-specific performance. However, obtaining high-quality fine-tuning data for large models is challenging due to data collection difficulties and high production costs. To address this, we propose MASTER, a novel data augmentation method that enriches original data through interactions among multiple agents with varying cognitive levels. We simulate three pedagogically grounded teaching scenarios, leveraging multi-agent conversations to generate high-quality teacher-student interaction data. Utilizing MASTER, we construct BOOST-QA, a fine-tuning dataset augmented from existing datasets like Orca-Math-200k, ProcQA, and OpenHermes2.5. Experiments show that models fine-tuned with BOOST-QA perform excellently across multiple benchmarks, demonstrating strong multitask generalization. Notably, MASTER significantly improves models' reasoning abilities in complex tasks, providing valuable insights for future research.
>
---
#### [new 038] FroM: Frobenius Norm-Based Data-Free Adaptive Model Merging
- **分类: cs.CL**

- **简介: 论文提出FroM，一种无需训练数据的自适应模型融合方法，用于解决微调大语言模型时的任务干扰问题。该方法基于Frobenius范数直接衡量模型参数，通过引入额外超参数进行控制，在多种微调场景中优于基线方法。**

- **链接: [http://arxiv.org/pdf/2506.02478v1](http://arxiv.org/pdf/2506.02478v1)**

> **作者:** Zijian Li; Xiaocheng Feng; Huixin Liu; Yichong Huang; Ting Liu; Bing Qin
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** With the development of large language models, fine-tuning has emerged as an effective method to enhance performance in specific scenarios by injecting domain-specific knowledge. In this context, model merging techniques provide a solution for fusing knowledge from multiple fine-tuning models by combining their parameters. However, traditional methods often encounter task interference when merging full fine-tuning models, and this problem becomes even more evident in parameter-efficient fine-tuning scenarios. In this paper, we introduce an improvement to the RegMean method, which indirectly leverages the training data to approximate the outputs of the linear layers before and after merging. We propose an adaptive merging method called FroM, which directly measures the model parameters using the Frobenius norm, without any training data. By introducing an additional hyperparameter for control, FroM outperforms baseline methods across various fine-tuning scenarios, alleviating the task interference problem.
>
---
#### [new 039] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于代码生成与测试任务，旨在解决缺乏监督信号导致的代码与测试协同进化难题。作者提出CURE框架，通过强化学习让编码模型和单元测试模型相互学习、共同进化，提升了代码生成准确性和推理效率，并支持下游任务扩展。**

- **链接: [http://arxiv.org/pdf/2506.03136v1](http://arxiv.org/pdf/2506.03136v1)**

> **作者:** Yinjie Wang; Ling Yang; Ye Tian; Ke Shen; Mengdi Wang
>
> **备注:** Project: https://github.com/Gen-Verse/CURE
>
> **摘要:** We propose CURE, a novel reinforcement learning framework with a dedicated reward design that co-evolves coding and unit test generation capabilities based on their interaction outcomes, without any ground-truth code as supervision. This approach enables flexible and scalable training and allows the unit tester to learn directly from the coder's mistakes. Our derived ReasonFlux-Coder-7B and 14B models improve code generation accuracy by 5.3% and Best-of-N accuracy by 9.0% after optimization on Qwen2.5-Instruct models, outperforming similarly sized Qwen-Coder, DeepSeek-Coder, and Seed-Coder. They naturally extend to downstream tasks such as test-time scaling and agentic coding-achieving a 8.1% improvement over the base model. For the long-CoT model, our ReasonFlux-Coder-4B consistently outperforms Qwen3-4B while achieving 64.8% inference efficiency in unit test generation. Notably, we also find that our model can serve as an effective reward model for reinforcement learning on base models. Project: https://github.com/Gen-Verse/CURE
>
---
#### [new 040] M$^3$FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一个金融会议理解评估数据集M³FinMeeting，属于自然语言处理任务。旨在解决现有金融领域基准数据局限性大、难以反映真实会议场景的问题。论文构建了多语言、跨行业、多任务的数据集，包含摘要生成、问答对抽取和问题回答三个任务，支持英文、中文和日文，覆盖多个行业领域，以全面评估大语言模型在金融会议理解方面的能力。**

- **链接: [http://arxiv.org/pdf/2506.02510v1](http://arxiv.org/pdf/2506.02510v1)**

> **作者:** Jie Zhu; Junhui Li; Yalong Wen; Xiandong Li; Lifan Guo; Feng Chen
>
> **备注:** Accepted by ACL-2025
>
> **摘要:** Recent breakthroughs in large language models (LLMs) have led to the development of new benchmarks for evaluating their performance in the financial domain. However, current financial benchmarks often rely on news articles, earnings reports, or announcements, making it challenging to capture the real-world dynamics of financial meetings. To address this gap, we propose a novel benchmark called $\texttt{M$^3$FinMeeting}$, which is a multilingual, multi-sector, and multi-task dataset designed for financial meeting understanding. First, $\texttt{M$^3$FinMeeting}$ supports English, Chinese, and Japanese, enhancing comprehension of financial discussions in diverse linguistic contexts. Second, it encompasses various industry sectors defined by the Global Industry Classification Standard (GICS), ensuring that the benchmark spans a broad range of financial activities. Finally, $\texttt{M$^3$FinMeeting}$ includes three tasks: summarization, question-answer (QA) pair extraction, and question answering, facilitating a more realistic and comprehensive evaluation of understanding. Experimental results with seven popular LLMs reveal that even the most advanced long-context models have significant room for improvement, demonstrating the effectiveness of $\texttt{M$^3$FinMeeting}$ as a benchmark for assessing LLMs' financial meeting comprehension skills.
>
---
#### [new 041] Performance of leading large language models in May 2025 in Membership of the Royal College of General Practitioners-style examination questions: a cross-sectional analysis
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文评估了2025年5月领先的大型语言模型（如o3、Claude Opus 4等）在英国全科医师考试（MRCGP风格）中的表现。任务是测试这些模型解答包含文本、实验室结果和临床图像的医学考题的能力。研究发现，所有模型成绩优异，均远超医生平均水平，其中o3表现最佳。论文展示了LLM在初级医疗教育中的潜力。**

- **链接: [http://arxiv.org/pdf/2506.02987v1](http://arxiv.org/pdf/2506.02987v1)**

> **作者:** Richard Armitage
>
> **备注:** 12 pages, 1 Table
>
> **摘要:** Background: Large language models (LLMs) have demonstrated substantial potential to support clinical practice. Other than Chat GPT4 and its predecessors, few LLMs, especially those of the leading and more powerful reasoning model class, have been subjected to medical specialty examination questions, including in the domain of primary care. This paper aimed to test the capabilities of leading LLMs as of May 2025 (o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro) in primary care education, specifically in answering Member of the Royal College of General Practitioners (MRCGP) style examination questions. Methods: o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro were tasked to answer 100 randomly chosen multiple choice questions from the Royal College of General Practitioners GP SelfTest on 25 May 2025. Questions included textual information, laboratory results, and clinical images. Each model was prompted to answer as a GP in the UK and was provided with full question information. Each question was attempted once by each model. Responses were scored against correct answers provided by GP SelfTest. Results: The total score of o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro was 99.0%, 95.0%, 95.0%, and 95.0%, respectively. The average peer score for the same questions was 73.0%. Discussion: All models performed remarkably well, and all substantially exceeded the average performance of GPs and GP registrars who had answered the same questions. o3 demonstrated the best performance, while the performances of the other leading models were comparable with each other and were not substantially lower than that of o3. These findings strengthen the case for LLMs, particularly reasoning models, to support the delivery of primary care, especially those that have been specifically trained on primary care clinical data.
>
---
#### [new 042] Something Just Like TRuST : Toxicity Recognition of Span and Target
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的毒性检测任务，旨在解决在线内容 toxicity 识别问题。作者构建了包含多维度标注的综合数据集 TRuST，并评估了大型语言模型在毒性检测、目标群体识别和有毒片段提取上的表现，发现微调模型优于零/少样本方法，但社会推理能力仍不足。**

- **链接: [http://arxiv.org/pdf/2506.02326v1](http://arxiv.org/pdf/2506.02326v1)**

> **作者:** Berk Atil; Namrata Sureddy; Rebecca J. Passonneau
>
> **摘要:** Toxicity in online content, including content generated by language models, has become a critical concern due to its potential for negative psychological and social impact. This paper introduces TRuST, a comprehensive dataset designed to improve toxicity detection that merges existing datasets, and has labels for toxicity, target social group, and toxic spans. It includes a diverse range of target groups such as ethnicity, gender, religion, disability, and politics, with both human/machine-annotated and human machine-generated data. We benchmark state-of-the-art large language models (LLMs) on toxicity detection, target group identification, and toxic span extraction. We find that fine-tuned models consistently outperform zero-shot and few-shot prompting, though performance remains low for certain social groups. Further, reasoning capabilities do not significantly improve performance, indicating that LLMs have weak social reasoning skills.
>
---
#### [new 043] Natural Language Processing to Enhance Deliberation in Political Online Discussions: A Survey
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理任务，旨在提升政治在线讨论中的协商质量。论文分析了政治在线讨论中存在的问题，并探讨了如何利用机器学习方法改善讨论环境、促进理性交流。**

- **链接: [http://arxiv.org/pdf/2506.02533v1](http://arxiv.org/pdf/2506.02533v1)**

> **作者:** Maike Behrendt; Stefan Sylvius Wagner; Carina Weinmann; Marike Bormann; Mira Warne; Stefan Harmeling
>
> **摘要:** Political online participation in the form of discussing political issues and exchanging opinions among citizens is gaining importance with more and more formats being held digitally. To come to a decision, a careful discussion and consideration of opinions and a civil exchange of arguments, which is defined as the act of deliberation, is desirable. The quality of discussions and participation processes in terms of their deliberativeness highly depends on the design of platforms and processes. To facilitate online communication for both participants and initiators, machine learning methods offer a lot of potential. In this work we want to showcase which issues occur in political online discussions and how machine learning can be used to counteract these issues and enhance deliberation.
>
---
#### [new 044] Multi-task Learning with Active Learning for Arabic Offensive Speech Detection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决阿拉伯语社交媒体中攻击性言语检测的问题。由于标注数据少、方言差异和语言复杂性，检测难度大。论文提出一种结合多任务学习与主动学习的新框架，通过联合训练暴力与低俗言语任务，动态调整任务权重，并采用主动学习策略选择关键样本，提升检测效果。最终在OSACT2022数据集上取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2506.02753v1](http://arxiv.org/pdf/2506.02753v1)**

> **作者:** Aisha Alansari; Hamzah Luqman
>
> **摘要:** The rapid growth of social media has amplified the spread of offensive, violent, and vulgar speech, which poses serious societal and cybersecurity concerns. Detecting such content in Arabic text is particularly complex due to limited labeled data, dialectal variations, and the language's inherent complexity. This paper proposes a novel framework that integrates multi-task learning (MTL) with active learning to enhance offensive speech detection in Arabic social media text. By jointly training on two auxiliary tasks, violent and vulgar speech, the model leverages shared representations to improve the detection accuracy of the offensive speech. Our approach dynamically adjusts task weights during training to balance the contribution of each task and optimize performance. To address the scarcity of labeled data, we employ an active learning strategy through several uncertainty sampling techniques to iteratively select the most informative samples for model training. We also introduce weighted emoji handling to better capture semantic cues. Experimental results on the OSACT2022 dataset show that the proposed framework achieves a state-of-the-art macro F1-score of 85.42%, outperforming existing methods while using significantly fewer fine-tuning samples. The findings of this study highlight the potential of integrating MTL with active learning for efficient and accurate offensive language detection in resource-constrained settings.
>
---
#### [new 045] Enhancing Large Language Models with Neurosymbolic Reasoning for Multilingual Tasks
- **分类: cs.CL**

- **简介: 该论文属于多语言任务下的大模型推理增强任务，旨在解决长文本中多目标信息分散导致的推理难题。作者提出NSAR方法，结合神经与符号推理，在推理过程中显式提取符号事实并生成可执行代码以提升推理准确性。实验表明其优于传统RAG和提示策略。**

- **链接: [http://arxiv.org/pdf/2506.02483v1](http://arxiv.org/pdf/2506.02483v1)**

> **作者:** Sina Bagheri Nezhad; Ameeta Agrawal
>
> **备注:** Accepted at 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
>
> **摘要:** Large language models (LLMs) often struggle to perform multi-target reasoning in long-context scenarios where relevant information is scattered across extensive documents. To address this challenge, we introduce NeuroSymbolic Augmented Reasoning (NSAR), which combines the benefits of neural and symbolic reasoning during inference. NSAR explicitly extracts symbolic facts from text and generates executable Python code to handle complex reasoning steps. Through extensive experiments across seven languages and diverse context lengths, we demonstrate that NSAR significantly outperforms both a vanilla RAG baseline and advanced prompting strategies in accurately identifying and synthesizing multiple pieces of information. Our results highlight the effectiveness of combining explicit symbolic operations with neural inference for robust, interpretable, and scalable reasoning in multilingual settings.
>
---
#### [new 046] A Multi-Dialectal Dataset for German Dialect ASR and Dialect-to-Standard Speech Translation
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别与方言研究任务，旨在解决德语方言在ASR中缺乏代表性的问 题。作者构建了包含东南德语三种方言及标准德语的语音数据集Betthupferl，提 供方言与标准语双语转录，并分析语言差异。他们评估多语言ASR模型在方言到 标准语翻译中的表现，发现模型输出更接近方言结构，仅部分语法差异被归一化。**

- **链接: [http://arxiv.org/pdf/2506.02894v1](http://arxiv.org/pdf/2506.02894v1)**

> **作者:** Verena Blaschke; Miriam Winkler; Constantin Förster; Gabriele Wenger-Glemser; Barbara Plank
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Although Germany has a diverse landscape of dialects, they are underrepresented in current automatic speech recognition (ASR) research. To enable studies of how robust models are towards dialectal variation, we present Betthupferl, an evaluation dataset containing four hours of read speech in three dialect groups spoken in Southeast Germany (Franconian, Bavarian, Alemannic), and half an hour of Standard German speech. We provide both dialectal and Standard German transcriptions, and analyze the linguistic differences between them. We benchmark several multilingual state-of-the-art ASR models on speech translation into Standard German, and find differences between how much the output resembles the dialectal vs. standardized transcriptions. Qualitative error analyses of the best ASR model reveal that it sometimes normalizes grammatical differences, but often stays closer to the dialectal constructions.
>
---
#### [new 047] TL;DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression
- **分类: cs.CL; cs.CE; cs.NA; math.NA**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理过程中输出过长、效率低的问题。作者提出一种动态比例训练方法，通过平衡模型的快速和慢速推理数据，减少冗余推理过程，在保持推理能力的同时显著缩短输出长度。**

- **链接: [http://arxiv.org/pdf/2506.02678v1](http://arxiv.org/pdf/2506.02678v1)**

> **作者:** Zhong-Zhi Li; Xiao Liang; Zihao Tang; Lei Ji; Peijie Wang; Haotian Xu; Xing W; Haizhen Huang; Weiwei Deng; Ying Nian Wu; Yeyun Gong; Zhijiang Guo; Xiao Liu; Fei Yin; Cheng-Lin Liu
>
> **摘要:** Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon.
>
---
#### [new 048] Stereotypical gender actions can be extracted from Web text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决从网络文本中提取性别刻板动作的问题。通过分析语料库和推特数据，结合常识知识库OMCS，计算动作的性别倾向，并与人类判断进行对比评估。最终提供了包含441个手动标注及21,442个自动标注的动作数据集。**

- **链接: [http://arxiv.org/pdf/2506.02740v1](http://arxiv.org/pdf/2506.02740v1)**

> **作者:** Amaç Herdağdelen; Marco Baroni
>
> **摘要:** We extracted gender-specific actions from text corpora and Twitter, and compared them to stereotypical expectations of people. We used Open Mind Common Sense (OMCS), a commonsense knowledge repository, to focus on actions that are pertinent to common sense and daily life of humans. We use the gender information of Twitter users and Web-corpus-based pronoun/name gender heuristics to compute the gender bias of the actions. With high recall, we obtained a Spearman correlation of 0.47 between corpus-based predictions and a human gold standard, and an area under the ROC curve of 0.76 when predicting the polarity of the gold standard. We conclude that it is feasible to use natural text (and a Twitter-derived corpus in particular) in order to augment commonsense repositories with the stereotypical gender expectations of actions. We also present a dataset of 441 commonsense actions with human judges' ratings on whether the action is typically/slightly masculine/feminine (or neutral), and another larger dataset of 21,442 actions automatically rated by the methods we investigate in this study.
>
---
#### [new 049] LAM SIMULATOR: Advancing Data Generation for Large Action Model Training via Online Exploration and Trajectory Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出LAM SIMULATOR，用于大型动作模型（LAMs）训练的数据生成框架。旨在解决多步骤任务中高质量训练数据缺乏的问题。通过在线探索与轨迹反馈机制，实现LLM代理自主任务探索与数据生成。实验表明其在提升AI代理性能方面效果显著。**

- **链接: [http://arxiv.org/pdf/2506.02298v1](http://arxiv.org/pdf/2506.02298v1)**

> **作者:** Thai Hoang; Kung-Hsiang Huang; Shirley Kokane; Jianguo Zhang; Zuxin Liu; Ming Zhu; Jake Grigsby; Tian Lan; Michael S Ryoo; Chien-Sheng Wu; Shelby Heinecke; Huan Wang; Silvio Savarese; Caiming Xiong; Juan Carlos Niebles
>
> **备注:** LAM Simulator framework for agentic data generation
>
> **摘要:** Large Action Models (LAMs) for AI Agents offer incredible potential but face challenges due to the need for high-quality training data, especially for multi-steps tasks that involve planning, executing tool calls, and responding to feedback. To address these issues, we present LAM SIMULATOR, a comprehensive framework designed for online exploration of agentic tasks with high-quality feedback. Our framework features a dynamic task query generator, an extensive collection of tools, and an interactive environment where Large Language Model (LLM) Agents can call tools and receive real-time feedback. This setup enables LLM Agents to explore and solve tasks autonomously, facilitating the discovery of multiple approaches to tackle any given task. The resulting action trajectory data are then used to create high-quality training datasets for LAMs. Our experiments on popular agentic benchmarks, ToolBench and CRMArena, highlight the effectiveness of LAM SIMULATOR: models trained with self-generated datasets using our framework achieve significant performance gains, up to a 49.3\% improvement over their original baselines. LAM SIMULATOR requires minimal human input during dataset creation, highlighting LAM SIMULATOR's efficiency and effectiveness in speeding up development of AI agents.
>
---
#### [new 050] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems
- **分类: cs.CL**

- **简介: 该论文属于隐私政策问答任务，旨在解决现有系统在英语方言上的性能差异问题。作者提出了一种多智能体框架，通过方言智能体和隐私政策智能体协作，提升模型对非标准方言的处理能力，无需重新训练即可提高准确率，从而实现更公平的信息访问。**

- **链接: [http://arxiv.org/pdf/2506.02998v1](http://arxiv.org/pdf/2506.02998v1)**

> **作者:** Đorđe Klisura; Astrid R Bernaga Torres; Anna Karen Gárate-Escamilla; Rajesh Roshan Biswal; Ke Yang; Hilal Pataci; Anthony Rios
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Privacy policies inform users about data collection and usage, yet their complexity limits accessibility for diverse populations. Existing Privacy Policy Question Answering (QA) systems exhibit performance disparities across English dialects, disadvantaging speakers of non-standard varieties. We propose a novel multi-agent framework inspired by human-centered design principles to mitigate dialectal biases. Our approach integrates a Dialect Agent, which translates queries into Standard American English (SAE) while preserving dialectal intent, and a Privacy Policy Agent, which refines predictions using domain expertise. Unlike prior approaches, our method does not require retraining or dialect-specific fine-tuning, making it broadly applicable across models and domains. Evaluated on PrivacyQA and PolicyQA, our framework improves GPT-4o-mini's zero-shot accuracy from 0.394 to 0.601 on PrivacyQA and from 0.352 to 0.464 on PolicyQA, surpassing or matching few-shot baselines without additional training data. These results highlight the effectiveness of structured agent collaboration in mitigating dialect biases and underscore the importance of designing NLP systems that account for linguistic diversity to ensure equitable access to privacy information.
>
---
#### [new 051] Conditioning Large Language Models on Legal Systems? Detecting Punishable Hate Speech
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律与自然语言处理交叉任务，旨在解决大型语言模型在法律系统下对可惩罚仇恨言论的检测问题。论文通过不同方式对模型进行法律知识条件限制，分析其在仇恨言论分类中的表现，发现模型与法律专家之间仍存在性能差距，尤其在抽象法律知识条件下表现较差。**

- **链接: [http://arxiv.org/pdf/2506.03009v1](http://arxiv.org/pdf/2506.03009v1)**

> **作者:** Florian Ludwig; Torsten Zesch; Frederike Zufall
>
> **摘要:** The assessment of legal problems requires the consideration of a specific legal system and its levels of abstraction, from constitutional law to statutory law to case law. The extent to which Large Language Models (LLMs) internalize such legal systems is unknown. In this paper, we propose and investigate different approaches to condition LLMs at different levels of abstraction in legal systems. This paper examines different approaches to conditioning LLMs at multiple levels of abstraction in legal systems to detect potentially punishable hate speech. We focus on the task of classifying whether a specific social media posts falls under the criminal offense of incitement to hatred as prescribed by the German Criminal Code. The results show that there is still a significant performance gap between models and legal experts in the legal assessment of hate speech, regardless of the level of abstraction with which the models were conditioned. Our analysis revealed, that models conditioned on abstract legal knowledge lacked deep task understanding, often contradicting themselves and hallucinating answers, while models using concrete legal knowledge performed reasonably well in identifying relevant target groups, but struggled with classifying target conducts.
>
---
#### [new 052] Exploiting the English Vocabulary Profile for L2 word-level vocabulary assessment with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与二语习得交叉任务，旨在解决二语词汇水平自动评估问题。利用英语词汇档案（EVP）和大语言模型（LLMs），结合词义、上下文及多词表达等信息，精细评估学习者作文中单词的熟练度，并验证其与整体写作水平的关系及EVP的一致性。**

- **链接: [http://arxiv.org/pdf/2506.02758v1](http://arxiv.org/pdf/2506.02758v1)**

> **作者:** Stefano Bannò; Kate Knill; Mark Gales
>
> **备注:** Accepted to the 20th Workshop on Innovative Use of NLP for Building Educational Applications
>
> **摘要:** Vocabulary use is a fundamental aspect of second language (L2) proficiency. To date, its assessment by automated systems has typically examined the context-independent, or part-of-speech (PoS) related use of words. This paper introduces a novel approach to enable fine-grained vocabulary evaluation exploiting the precise use of words within a sentence. The scheme combines large language models (LLMs) with the English Vocabulary Profile (EVP). The EVP is a standard lexical resource that enables in-context vocabulary use to be linked with proficiency level. We evaluate the ability of LLMs to assign proficiency levels to individual words as they appear in L2 learner writing, addressing key challenges such as polysemy, contextual variation, and multi-word expressions. We compare LLMs to a PoS-based baseline. LLMs appear to exploit additional semantic information that yields improved performance. We also explore correlations between word-level proficiency and essay-level proficiency. Finally, the approach is applied to examine the consistency of the EVP proficiency levels. Results show that LLMs are well-suited for the task of vocabulary assessment.
>
---
#### [new 053] Quantifying Misattribution Unfairness in Authorship Attribution
- **分类: cs.CL**

- **简介: 该论文属于作者归属任务，旨在解决误归因不公平问题。通过提出MAUIk指标量化模型误判风险，发现模型对某些作者存在更高误判风险，且与作者在嵌入空间中的位置相关。研究强调需关注此类风险并进行校准。**

- **链接: [http://arxiv.org/pdf/2506.02321v1](http://arxiv.org/pdf/2506.02321v1)**

> **作者:** Pegah Alipoormolabashi; Ajay Patel; Niranjan Balasubramanian
>
> **摘要:** Authorship misattribution can have profound consequences in real life. In forensic settings simply being considered as one of the potential authors of an evidential piece of text or communication can result in undesirable scrutiny. This raises a fairness question: Is every author in the candidate pool at equal risk of misattribution? Standard evaluation measures for authorship attribution systems do not explicitly account for this notion of fairness. We introduce a simple measure, Misattribution Unfairness Index (MAUIk), which is based on how often authors are ranked in the top k for texts they did not write. Using this measure we quantify the unfairness of five models on two different datasets. All models exhibit high levels of unfairness with increased risks for some authors. Furthermore, we find that this unfairness relates to how the models embed the authors as vectors in the latent search space. In particular, we observe that the risk of misattribution is higher for authors closer to the centroid (or center) of the embedded authors in the haystack. These results indicate the potential for harm and the need for communicating with and calibrating end users on misattribution risk when building and providing such models for downstream use.
>
---
#### [new 054] IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator
- **分类: cs.CL; cs.AI**

- **简介: 论文提出IMPARA-GED，属于自动语法纠错评估任务，旨在解决无参考答案的语法错误质量评估问题。通过增强预训练语言模型的语法错误检测能力，构建更优的质量评估方法，并在SEEDA数据集上验证其与人工评估具有最高相关性。**

- **链接: [http://arxiv.org/pdf/2506.02899v1](http://arxiv.org/pdf/2506.02899v1)**

> **作者:** Yusuke Sakai; Takumi Goto; Taro Watanabe
>
> **备注:** ACL 2025 Findings
>
> **摘要:** We propose IMPARA-GED, a novel reference-free automatic grammatical error correction (GEC) evaluation method with grammatical error detection (GED) capabilities. We focus on the quality estimator of IMPARA, an existing automatic GEC evaluation method, and construct that of IMPARA-GED using a pre-trained language model with enhanced GED capabilities. Experimental results on SEEDA, a meta-evaluation dataset for automatic GEC evaluation methods, demonstrate that IMPARA-GED achieves the highest correlation with human sentence-level evaluations.
>
---
#### [new 055] Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型对句子语法可接受性的判断。通过“语法提示”方法，先让大模型解释语法规则，再将解释反馈给目标模型以辅助判断。解决了模型知规则却难应用的问题，显著提升了小模型表现，降低了大小模型间准确率差距，尤其在多语言环境下效果明显。**

- **链接: [http://arxiv.org/pdf/2506.02302v1](http://arxiv.org/pdf/2506.02302v1)**

> **作者:** Russell Scheinberg; Ameeta Agrawal; Amber Shore; So Young Lee
>
> **备注:** Accepted at ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) can explain grammatical rules, yet they often fail to apply those rules when judging sentence acceptability. We present "grammar prompting", an explain-then-process paradigm: a large LLM first produces a concise explanation of the relevant syntactic phenomenon, then that explanation is fed back as additional context to the target model -- either an LLM or a smaller language model (SLM) -- before deciding which sentence of a minimal pair is grammatical. On the English BLiMP, Chinese SLING, and Russian RuBLiMP benchmarks, this simple prompt design yields substantial improvements over strong baselines across many syntactic phenomena. Feeding an LLM's metalinguistic explanation back to the target model bridges the gap between knowing a rule and using it. On SLMs, grammar prompting alone trims the average LLM-SLM accuracy gap by about 20%, and when paired with chain-of-thought, by 56% (13.0 pp -> 5.8 pp), all at negligible cost. The lightweight, language-agnostic cue lets low-cost SLMs approach frontier-LLM performance in multilingual settings.
>
---
#### [new 056] Multilingual Information Retrieval with a Monolingual Knowledge Base
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于多语言信息检索任务，旨在解决如何利用单语知识库实现跨语言知识共享的问题。论文提出了一种基于加权采样的对比学习策略，用于微调多语言嵌入模型，从而提升检索效果，并适用于多语言和代码混合场景。**

- **链接: [http://arxiv.org/pdf/2506.02527v1](http://arxiv.org/pdf/2506.02527v1)**

> **作者:** Yingying Zhuang; Aman Gupta; Anurag Beniwal
>
> **备注:** 6 pages, accepted at GENNEXT@SIGIR25
>
> **摘要:** Multilingual information retrieval has emerged as powerful tools for expanding knowledge sharing across languages. On the other hand, resources on high quality knowledge base are often scarce and in limited languages, therefore an effective embedding model to transform sentences from different languages into a feature vector space same as the knowledge base language becomes the key ingredient for cross language knowledge sharing, especially to transfer knowledge available in high-resource languages to low-resource ones. In this paper we propose a novel strategy to fine-tune multilingual embedding models with weighted sampling for contrastive learning, enabling multilingual information retrieval with a monolingual knowledge base. We demonstrate that the weighted sampling strategy produces performance gains compared to standard ones by up to 31.03\% in MRR and up to 33.98\% in Recall@3. Additionally, our proposed methodology is language agnostic and applicable for both multilingual and code switching use cases.
>
---
#### [new 057] STORYTELLER: An Enhanced Plot-Planning Framework for Coherent and Cohesive Story Generation
- **分类: cs.CL**

- **简介: 该论文属于自动故事生成任务，旨在解决现有方法在叙事连贯性和逻辑一致性上的不足。作者提出了STORYTELLER框架，通过基于主谓宾三元组的剧情节点结构和动态模块（STORYLINE和NEKG）提升故事质量。实验表明其在多个指标上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02347v1](http://arxiv.org/pdf/2506.02347v1)**

> **作者:** Jiaming Li; Yukun Chen; Ziqiang Liu; Minghuan Tan; Lei Zhang; Yunshui Li; Run Luo; Longze Chen; Jing Luo; Ahmadreza Argha; Hamid Alinejad-Rokny; Wei Zhou; Min Yang
>
> **摘要:** Stories are central to human culture, serving to share ideas, preserve traditions, and foster connections. Automatic story generation, a key advancement in artificial intelligence (AI), offers new possibilities for creating personalized content, exploring creative ideas, and enhancing interactive experiences. However, existing methods struggle to maintain narrative coherence and logical consistency. This disconnect compromises the overall storytelling experience, underscoring the need for substantial improvements. Inspired by human cognitive processes, we introduce Storyteller, a novel approach that systemically improves the coherence and consistency of automatically generated stories. Storyteller introduces a plot node structure based on linguistically grounded subject verb object (SVO) triplets, which capture essential story events and ensure a consistent logical flow. Unlike previous methods, Storyteller integrates two dynamic modules, the STORYLINE and narrative entity knowledge graph (NEKG),that continuously interact with the story generation process. This integration produces structurally sound, cohesive and immersive narratives. Extensive experiments demonstrate that Storyteller significantly outperforms existing approaches, achieving an 84.33% average win rate through human preference evaluation. At the same time, it is also far ahead in other aspects including creativity, coherence, engagement, and relevance.
>
---
#### [new 058] Exploring Explanations Improves the Robustness of In-Context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言理解任务，旨在提升上下文学习（ICL）在分布外数据上的泛化能力。通过引入解释机制（X-ICL），引导模型理解并生成正确标签的推理过程，进一步提出X²-ICL，系统探索所有可能标签的解释，增强决策的全面性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.02378v1](http://arxiv.org/pdf/2506.02378v1)**

> **作者:** Ukyo Honda; Tatsushi Oka
>
> **备注:** Accepted to ACL 2025 (Main Conference)
>
> **摘要:** In-context learning (ICL) has emerged as a successful paradigm for leveraging large language models (LLMs). However, it often struggles to generalize beyond the distribution of the provided demonstrations. A recent advancement in enhancing robustness is ICL with explanations (X-ICL), which improves prediction reliability by guiding LLMs to understand and articulate the reasoning behind correct labels. Building on this approach, we introduce an advanced framework that extends X-ICL by systematically exploring explanations for all possible labels (X$^2$-ICL), thereby enabling more comprehensive and robust decision-making. Experimental results on multiple natural language understanding datasets validate the effectiveness of X$^2$-ICL, demonstrating significantly improved robustness to out-of-distribution data compared to the existing ICL approaches.
>
---
#### [new 059] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems
- **分类: cs.CL**

- **简介: 该论文研究语音到文本系统中习语翻译的挑战，属于机器翻译任务。对比了多种模型在德英、俄英语对上的表现，发现语音到文本系统在处理习语时效果明显下降，而文本到文本系统和大语言模型表现更好，强调需改进语音到文本系统的结构与策略。**

- **链接: [http://arxiv.org/pdf/2506.02995v1](http://arxiv.org/pdf/2506.02995v1)**

> **作者:** Iuliia Zaitova; Badr M. Abdullah; Wei Xue; Dietrich Klakow; Bernd Möbius; Tania Avgustinova
>
> **备注:** 13 pages, 3 figures, ACL 2025
>
> **摘要:** Idioms are defined as a group of words with a figurative meaning not deducible from their individual components. Although modern machine translation systems have made remarkable progress, translating idioms remains a major challenge, especially for speech-to-text systems, where research on this topic is notably sparse. In this paper, we systematically evaluate idiom translation as compared to conventional news translation in both text-to-text machine translation (MT) and speech-to-text translation (SLT) systems across two language pairs (German to English, Russian to English). We compare state-of-the-art end-to-end SLT systems (SeamlessM4T SLT-to-text, Whisper Large v3) with MT systems (SeamlessM4T SLT-to-text, No Language Left Behind), Large Language Models (DeepSeek, LLaMA) and cascaded alternatives. Our results reveal that SLT systems experience a pronounced performance drop on idiomatic data, often reverting to literal translations even in higher layers, whereas MT systems and Large Language Models demonstrate better handling of idioms. These findings underscore the need for idiom-specific strategies and improved internal representations in SLT architectures.
>
---
#### [new 060] From Anger to Joy: How Nationality Personas Shape Emotion Attribution in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）在赋予国籍角色时是否表现出情感刻板印象，属于自然语言处理与社会认知交叉任务。它旨在揭示不同国家在情绪归因上的模型偏差，并分析这些归因是否符合文化规范。研究发现，LLMs存在国籍相关的情感偏见，尤其在负面情绪上与人类反应不一致，表明其可能内化了简化或有偏见的文化刻板印象。**

- **链接: [http://arxiv.org/pdf/2506.02431v1](http://arxiv.org/pdf/2506.02431v1)**

> **作者:** Mahammed Kamruzzaman; Abdullah Al Monsur; Gene Louis Kim; Anshuman Chhabra
>
> **摘要:** Emotions are a fundamental facet of human experience, varying across individuals, cultural contexts, and nationalities. Given the recent success of Large Language Models (LLMs) as role-playing agents, we examine whether LLMs exhibit emotional stereotypes when assigned nationality-specific personas. Specifically, we investigate how different countries are represented in pre-trained LLMs through emotion attributions and whether these attributions align with cultural norms. Our analysis reveals significant nationality-based differences, with emotions such as shame, fear, and joy being disproportionately assigned across regions. Furthermore, we observe notable misalignment between LLM-generated and human emotional responses, particularly for negative emotions, highlighting the presence of reductive and potentially biased stereotypes in LLM outputs.
>
---
#### [new 061] Enhancing Paraphrase Type Generation: The Impact of DPO and RLHF Evaluated with Human-Ranked Data
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理任务，旨在提升模型生成特定类型释义的能力。它解决了现有方法依赖自动指标、难以匹配人类偏好的问题。研究引入了DPO训练方法，并构建了人类评分数据集，提升了释义准确性和偏好度，还开发了高性能的释义类型检测模型。**

- **链接: [http://arxiv.org/pdf/2506.02018v1](http://arxiv.org/pdf/2506.02018v1)**

> **作者:** Christopher Lee Lübbers
>
> **备注:** 21 pages, 11 figures. Master's thesis, University of Goettingen, December 2025. Code: https://github.com/cluebbers/dpo-rlhf-paraphrase-types. Models: https://huggingface.co/collections/cluebbers/enhancing-paraphrase-type-generation-673ca8d75dfe2ce962a48ac0
>
> **摘要:** Paraphrasing re-expresses meaning to enhance applications like text simplification, machine translation, and question-answering. Specific paraphrase types facilitate accurate semantic analysis and robust language models. However, existing paraphrase-type generation methods often misalign with human preferences due to reliance on automated metrics and limited human-annotated training data, obscuring crucial aspects of semantic fidelity and linguistic transformations. This study addresses this gap by leveraging a human-ranked paraphrase-type dataset and integrating Direct Preference Optimization (DPO) to align model outputs directly with human judgments. DPO-based training increases paraphrase-type generation accuracy by 3 percentage points over a supervised baseline and raises human preference ratings by 7 percentage points. A newly created human-annotated dataset supports more rigorous future evaluations. Additionally, a paraphrase-type detection model achieves F1 scores of 0.91 for addition/deletion, 0.78 for same polarity substitution, and 0.70 for punctuation changes. These findings demonstrate that preference data and DPO training produce more reliable, semantically accurate paraphrases, enabling downstream applications such as improved summarization and more robust question-answering. The PTD model surpasses automated metrics and provides a more reliable framework for evaluating paraphrase quality, advancing paraphrase-type research toward richer, user-aligned language generation and establishing a stronger foundation for future evaluations grounded in human-centric criteria.
>
---
#### [new 062] ReasoningFlow: Semantic Structure of Complex Reasoning Traces
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大推理模型生成的复杂推理过程难以分析的问题。提出了ReasoningFlow框架，将推理过程建模为有向无环图，以识别和理解其中的语义结构与推理模式。**

- **链接: [http://arxiv.org/pdf/2506.02532v1](http://arxiv.org/pdf/2506.02532v1)**

> **作者:** Jinu Lee; Sagnik Mukherjee; Dilek Hakkani-Tur; Julia Hockenmaier
>
> **备注:** 10 pages, 6 figures. ArgMining 2025 Workshop (Non-archival) @ ACL 2025
>
> **摘要:** Large reasoning models (LRMs) generate complex reasoning traces with planning, reflection, verification, and backtracking. In this work, we introduce ReasoningFlow, a unified schema for analyzing the semantic structures of these complex traces. ReasoningFlow parses traces into directed acyclic graphs, enabling the characterization of distinct reasoning patterns as subgraph structures. This human-interpretable representation offers promising applications in understanding, evaluating, and enhancing the reasoning processes of LRMs.
>
---
#### [new 063] Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多方言阿拉伯语语音识别任务，旨在解决方言语音识别中数据不足的问题。通过微调Whisper模型，结合标准阿拉伯语和多种方言数据训练，发现小规模标准语数据可显著提升效果，且混合方言模型表现与单一方言模型相当，为低资源场景提供了有效方案。**

- **链接: [http://arxiv.org/pdf/2506.02627v1](http://arxiv.org/pdf/2506.02627v1)**

> **作者:** Ömer Tarik Özyilmaz; Matt Coler; Matias Valdenegro-Toro
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Although commercial Arabic automatic speech recognition (ASR) systems support Modern Standard Arabic (MSA), they struggle with dialectal speech. We investigate the effect of fine-tuning OpenAI's Whisper on five major Arabic dialects (Gulf, Levantine, Iraqi, Egyptian, Maghrebi) using Mozilla Common Voice for MSA and the MASC dataset for dialectal speech. We evaluate MSA training size effects, benefits of pre-training on MSA data, and dialect-specific versus dialect-pooled models. We find that small amounts of MSA fine-tuning data yield substantial improvements for smaller models, matching larger non-fine-tuned models. While MSA pre-training shows minimal benefit, suggesting limited shared features between MSA and dialects, our dialect-pooled models perform comparably to dialect-specific ones. This indicates that pooling dialectal data, when properly balanced, can help address data scarcity in low-resource ASR without significant performance loss.
>
---
#### [new 064] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment
- **分类: cs.CL**

- **简介: 该论文属于任务导向型对话系统构建任务，旨在解决领域专家难以定义、测试和改进对话系统行为的问题。论文提出了CoDial框架，将专家知识转化为可执行对话逻辑，支持零样本生成与迭代优化，实现在法律、医疗等高风险领域的实用对话系统。**

- **链接: [http://arxiv.org/pdf/2506.02264v1](http://arxiv.org/pdf/2506.02264v1)**

> **作者:** Radin Shayanfar; Chu Fei Luo; Rohan Bhambhoria; Samuel Dahan; Xiaodan Zhu
>
> **摘要:** It is often challenging to teach specialized, unseen tasks to dialogue systems due to the high cost of expert knowledge, training data, and high technical difficulty. To support domain-specific applications - such as law, medicine, or finance - it is essential to build frameworks that enable non-technical experts to define, test, and refine system behaviour with minimal effort. Achieving this requires cross-disciplinary collaboration between developers and domain specialists. In this work, we introduce a novel framework, CoDial (Code for Dialogue), that converts expert knowledge, represented as a novel structured heterogeneous graph, into executable conversation logic. CoDial can be easily implemented in existing guardrailing languages, such as Colang, to enable interpretable, modifiable, and true zero-shot specification of task-oriented dialogue systems. Empirically, CoDial achieves state-of-the-art performance on the STAR dataset for inference-based models and is competitive with similar baselines on the well-known MultiWOZ dataset. We also demonstrate CoDial's iterative improvement via manual and LLM-aided feedback, making it a practical tool for expert-guided alignment of LLMs in high-stakes domains.
>
---
#### [new 065] Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.CE; cs.HC; cs.LG**

- **简介: 该论文属于单细胞数据分析任务，旨在解决现有模型缺乏批注级推理能力的问题。作者提出CellPuzzles基准，并训练基于强化学习的模型Cell-o1，实现跨批次唯一细胞类型分配，显著提升性能并模拟专家推理行为。**

- **链接: [http://arxiv.org/pdf/2506.02911v1](http://arxiv.org/pdf/2506.02911v1)**

> **作者:** Yin Fang; Qiao Jin; Guangzhi Xiong; Bowen Jin; Xianrui Zhong; Siru Ouyang; Aidong Zhang; Jiawei Han; Zhiyong Lu
>
> **备注:** 28 pages; 16 tables; 7 figures; Code: https://github.com/ncbi-nlp/cell-o1
>
> **摘要:** Cell type annotation is a key task in analyzing the heterogeneity of single-cell RNA sequencing data. Although recent foundation models automate this process, they typically annotate cells independently, without considering batch-level cellular context or providing explanatory reasoning. In contrast, human experts often annotate distinct cell types for different cell clusters based on their domain knowledge. To mimic this workflow, we introduce the CellPuzzles task, where the objective is to assign unique cell types to a batch of cells. This benchmark spans diverse tissues, diseases, and donor conditions, and requires reasoning across the batch-level cellular context to ensure label uniqueness. We find that off-the-shelf large language models (LLMs) struggle on CellPuzzles, with the best baseline (OpenAI's o1) achieving only 19.0% batch-level accuracy. To fill this gap, we propose Cell-o1, a 7B LLM trained via supervised fine-tuning on distilled reasoning traces, followed by reinforcement learning with batch-level rewards. Cell-o1 achieves state-of-the-art performance, outperforming o1 by over 73% and generalizing well across contexts. Further analysis of training dynamics and reasoning behaviors provides insights into batch-level annotation performance and emergent expert-like reasoning. Code and data are available at https://github.com/ncbi-nlp/cell-o1.
>
---
#### [new 066] Gender Inequality in English Textbooks Around the World: an NLP Approach
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于自然语言处理任务，旨在分析全球英语教材中的性别不平等现象。通过计算人物数量、首次提及顺序及词汇关联等指标，研究不同文化区域教材中的性别表征差异，发现男性角色普遍被过度代表，其中拉丁文化圈差异最小。**

- **链接: [http://arxiv.org/pdf/2506.02425v1](http://arxiv.org/pdf/2506.02425v1)**

> **作者:** Tairan Liu
>
> **摘要:** Textbooks play a critical role in shaping children's understanding of the world. While previous studies have identified gender inequality in individual countries' textbooks, few have examined the issue cross-culturally. This study applies natural language processing methods to quantify gender inequality in English textbooks from 22 countries across 7 cultural spheres. Metrics include character count, firstness (which gender is mentioned first), and TF-IDF word associations by gender. The analysis also identifies gender patterns in proper names appearing in TF-IDF word lists, tests whether large language models can distinguish between gendered word lists, and uses GloVe embeddings to examine how closely keywords associate with each gender. Results show consistent overrepresentation of male characters in terms of count, firstness, and named entities. All regions exhibit gender inequality, with the Latin cultural sphere showing the least disparity.
>
---
#### [new 067] Sounding Like a Winner? Prosodic Differences in Post-Match Interviews
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究网球赛后采访中的韵律特征，旨在通过语音的音高、强度等区分胜负，并利用自监督学习模型（如Wav2Vec 2.0和HuBERT）对比赛结果进行分类。任务是判断运动员是否获胜，使用传统声学特征和深度语音表示结合机器学习方法实现预测。**

- **链接: [http://arxiv.org/pdf/2506.02283v1](http://arxiv.org/pdf/2506.02283v1)**

> **作者:** Sofoklis Kakouros; Haoyu Chen
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This study examines the prosodic characteristics associated with winning and losing in post-match tennis interviews. Additionally, this research explores the potential to classify match outcomes solely based on post-match interview recordings using prosodic features and self-supervised learning (SSL) representations. By analyzing prosodic elements such as pitch and intensity, alongside SSL models like Wav2Vec 2.0 and HuBERT, the aim is to determine whether an athlete has won or lost their match. Traditional acoustic features and deep speech representations are extracted from the data, and machine learning classifiers are employed to distinguish between winning and losing players. Results indicate that SSL representations effectively differentiate between winning and losing outcomes, capturing subtle speech patterns linked to emotional states. At the same time, prosodic cues -- such as pitch variability -- remain strong indicators of victory.
>
---
#### [new 068] Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升小规模语言模型的推理能力。为解决小模型推理效果差且依赖大模型蒸馏的问题，作者提出 COLLATE 框架，通过偏好优化使小模型生成多样化推理路径，并选择最优解。方法在多个数据集上验证有效，适用于不同结构和规模的小模型。**

- **链接: [http://arxiv.org/pdf/2506.02519v1](http://arxiv.org/pdf/2506.02519v1)**

> **作者:** Sohan Patnaik; Milan Aggarwal; Sumit Bhatia; Balaji Krishnamurthy
>
> **备注:** Accepted at ACL Main 2025
>
> **摘要:** LLMssuch as GPT-4 have shown a remarkable ability to solve complex questions by generating step-by-step rationales. Prior works have utilized this capability to improve smaller and cheaper LMs (say, with 7B parameters). However, various practical constraints, such as copyright and legal issues, owing to lack of transparency in the pre-training data of large (often closed) models, prevent their use in commercial settings. Little focus has been given to improving the innate reasoning ability of smaller models without distilling information from larger LLMs. To address this, we propose COLLATE, a trainable framework that tunes a (small) LLM to generate those outputs from a pool of diverse rationales that selectively improves the downstream task. COLLATE enforces multiple instances of the same LLM to exhibit distinct behavior and employs them to generate rationales to obtain diverse outputs. The LLM is then tuned via preference optimization to choose the candidate rationale which maximizes the likelihood of ground-truth answer. COLLATE outperforms several trainable and prompting baselines on 5 datasets across 3 domains: maths problem solving, natural language inference, and commonsense reasoning. We show the eff icacy of COLLATE on LLMs from different model families across varying parameter scales (1B to 8B) and demonstrate the benefit of multiple rationale providers guided by the end task through ablations. Code is released here (https://github.com/Sohanpatnaik106/collate).
>
---
#### [new 069] Expanding before Inferring: Enhancing Factuality in Large Language Models through Premature Layers Interpolation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成文本时出现的事实不一致问题（即“幻觉”）。作者提出了一种无需训练、可插拔使用的方法PLI，通过在模型中间层插入基于数学插值得到的“早熟层”，增强信息处理深度，从而提升生成内容的事实准确性。实验表明该方法在多个数据集上有效减少幻觉现象。**

- **链接: [http://arxiv.org/pdf/2506.02973v1](http://arxiv.org/pdf/2506.02973v1)**

> **作者:** Dingwei Chen; Ziqiang Liu; Feiteng Fang; Chak Tou Leong; Shiwen Ni; Ahmadreza Argha; Hamid Alinejad-Rokny; Min Yang; Chengming Li
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable capabilities in text understanding and generation. However, their tendency to produce factually inconsistent outputs, commonly referred to as ''hallucinations'', remains a critical challenge. Existing approaches, such as retrieval-based and inference-time correction methods, primarily address this issue at the input or output level, often overlooking the intrinsic information refinement process and the role of premature layers. Meanwhile, alignment- and fine-tuning-based methods are resource-intensive. In this paper, we propose PLI (Premature Layers Interpolation), a novel, training-free, and plug-and-play intervention designed to enhance factuality. PLI mitigates hallucinations by inserting premature layers formed through mathematical interpolation with adjacent layers. Inspired by stable diffusion and sampling steps, PLI extends the depth of information processing and transmission in LLMs, improving factual coherence. Experiments on four publicly available datasets demonstrate that PLI effectively reduces hallucinations while outperforming existing baselines in most cases. Further analysis suggests that the success of layer interpolation is closely linked to LLMs' internal mechanisms. To promote reproducibility, we will release our code and data upon acceptance.
>
---
#### [new 070] Do Language Models Think Consistently? A Study of Value Preferences Across Varying Response Lengths
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在不同回答长度下的价值偏好一致性问题，属于自然语言处理与伦理评估任务。它旨在解决短文本测试结果是否能准确反映模型在长文本生成中的价值倾向。论文通过对比五种模型的短文本反应和不同长度的长文本输出发现两者间相关性较弱，并分析了影响长文本价值表达的因素。**

- **链接: [http://arxiv.org/pdf/2506.02481v1](http://arxiv.org/pdf/2506.02481v1)**

> **作者:** Inderjeet Nair; Lu Wang
>
> **摘要:** Evaluations of LLMs' ethical risks and value inclinations often rely on short-form surveys and psychometric tests, yet real-world use involves long-form, open-ended responses -- leaving value-related risks and preferences in practical settings largely underexplored. In this work, we ask: Do value preferences inferred from short-form tests align with those expressed in long-form outputs? To address this question, we compare value preferences elicited from short-form reactions and long-form responses, varying the number of arguments in the latter to capture users' differing verbosity preferences. Analyzing five LLMs (llama3-8b, gemma2-9b, mistral-7b, qwen2-7b, and olmo-7b), we find (1) a weak correlation between value preferences inferred from short-form and long-form responses across varying argument counts, and (2) similarly weak correlation between preferences derived from any two distinct long-form generation settings. (3) Alignment yields only modest gains in the consistency of value expression. Further, we examine how long-form generation attributes relate to value preferences, finding that argument specificity negatively correlates with preference strength, while representation across scenarios shows a positive correlation. Our findings underscore the need for more robust methods to ensure consistent value expression across diverse applications.
>
---
#### [new 071] NovelHopQA: Diagnosing Multi-Hop Reasoning Failures in Long Narrative Contexts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长文本多跳推理中的表现问题。作者构建了 NovelHopQA 基准，包含来自小说的超长上下文和多跳问题，用于评估模型在不同推理深度和上下文长度下的表现，并分析其失败原因。**

- **链接: [http://arxiv.org/pdf/2506.02000v1](http://arxiv.org/pdf/2506.02000v1)**

> **作者:** Abhay Gupta; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** Current large language models (LLMs) struggle to answer questions that span tens of thousands of tokens, especially when multi-hop reasoning is involved. While prior benchmarks explore long-context comprehension or multi-hop reasoning in isolation, none jointly vary context length and reasoning depth in natural narrative settings. We introduce NovelHopQA, the first benchmark to evaluate k1-4 hop QA over 64k-128k-token excerpts from 83 full-length public-domain novels. A keyword-guided pipeline builds hop-separated chains grounded in coherent storylines. We evaluate six state-of-the-art (SOTA) models and apply oracle-context filtering to ensure all questions are genuinely answerable. Human annotators validate both alignment and hop depth. We noticed consistent accuracy drops with increased hops and context length, even in frontier models-revealing that sheer scale does not guarantee robust reasoning. Our failure mode analysis highlights common breakdowns, such as missed final-hop integration and long-range drift. NovelHopQA offers a controlled diagnostic setting to stress-test multi-hop reasoning at scale.
>
---
#### [new 072] Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know?
- **分类: cs.CL; cs.IR; cs.LG; stat.AP; stat.ME**

- **简介: 该论文属于模型评估任务，旨在解决当前对大语言模型（LLMs）评估不准确、忽略未见知识的问题。作者提出了KnowSum框架，通过统计方法量化LLMs中未被观察到的知识，从而更全面地评估其能力，并在多个应用中验证了该方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.02058v1](http://arxiv.org/pdf/2506.02058v1)**

> **作者:** Xiang Li; Jiayi Xin; Qi Long; Weijie J. Su
>
> **摘要:** Accurate evaluation of large language models (LLMs) is crucial for understanding their capabilities and guiding their development. However, current evaluations often inconsistently reflect the actual capacities of these models. In this paper, we demonstrate that one of many contributing factors to this \textit{evaluation crisis} is the oversight of unseen knowledge -- information encoded by LLMs but not directly observed or not yet observed during evaluations. We introduce KnowSum, a statistical framework designed to provide a more comprehensive assessment by quantifying the unseen knowledge for a class of evaluation tasks. KnowSum estimates the unobserved portion by extrapolating from the appearance frequencies of observed knowledge instances. We demonstrate the effectiveness and utility of KnowSum across three critical applications: estimating total knowledge, evaluating information retrieval effectiveness, and measuring output diversity. Our experiments reveal that a substantial volume of knowledge is omitted when relying solely on observed LLM performance. Importantly, KnowSum yields significantly different comparative rankings for several common LLMs based on their internal knowledge.
>
---
#### [new 073] CoT is Not True Reasoning, It Is Just a Tight Constraint to Imitate: A Theory Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，探讨大语言模型的推理能力。它旨在解决关于思维链（CoT）提示是否真正引发模型抽象推理的问题。论文提出理论观点，认为CoT并非激发真实推理，而是通过结构约束引导模型模仿推理过程。**

- **链接: [http://arxiv.org/pdf/2506.02878v1](http://arxiv.org/pdf/2506.02878v1)**

> **作者:** Jintian Shao; Yiming Cheng
>
> **摘要:** Chain-of-Thought (CoT) prompting has demonstrably enhanced the performance of Large Language Models on tasks requiring multi-step inference. This success has led to widespread claims of emergent reasoning capabilities in these models. In this paper, we present a theoretical counter-perspective: Chain-of-Thought (CoT) does not elicit genuine, abstract reasoning. Instead, we argue that Chain-of-Thought functions as a powerful structural constraint that guides Large Language Models to imitate the form of reasoning. By forcing the generation of intermediate steps, Chain-of-Thought leverages the model immense capacity for sequence prediction and pattern matching, effectively constraining its output to sequences that resemble coherent thought processes. Chain-of-Thought (CoT) prompting has demonstrably enhanced the performance of Large Language Models on tasks requiring multi-step inference. This success has led to widespread claims of emergent reasoning capabilities in these models. In this paper, we present a theoretical counter-perspective: Chain-of-Thought (CoT) does not elicit genuine, abstract reasoning. Instead, we argue that Chain-of-Thought functions as a powerful structural constraint that guides Large Language Models to imitate the form of reasoning. By forcing the generation of intermediate steps, Chain-of-Thought leverages the model immense capacity for sequence prediction and pattern matching, effectively constraining its output to sequences that resemble coherent thought processes.
>
---
#### [new 074] On Entity Identification in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究语言模型内部如何识别和区分命名实体。论文提出框架分析模型内部表示的聚类质量，解决实体提及的歧义与变体问题，揭示实体信息在低维空间紧凑表示，并探讨其对词预测性能的影响。**

- **链接: [http://arxiv.org/pdf/2506.02701v1](http://arxiv.org/pdf/2506.02701v1)**

> **作者:** Masaki Sakata; Sho Yokoi; Benjamin Heinzerling; Takumi Ito; Kentaro Inui
>
> **备注:** ACL 2025 Findings; 26 pages, 13 figures, 9 tables
>
> **摘要:** We analyze the extent to which internal representations of language models (LMs) identify and distinguish mentions of named entities, focusing on the many-to-many correspondence between entities and their mentions. We first formulate two problems of entity mentions -- ambiguity and variability -- and propose a framework analogous to clustering quality metrics. Specifically, we quantify through cluster analysis of LM internal representations the extent to which mentions of the same entity cluster together and mentions of different entities remain separated. Our experiments examine five Transformer-based autoregressive models, showing that they effectively identify and distinguish entities with metrics analogous to precision and recall ranging from 0.66 to 0.9. Further analysis reveals that entity-related information is compactly represented in a low-dimensional linear subspace at early LM layers. Additionally, we clarify how the characteristics of entity representations influence word prediction performance. These findings are interpreted through the lens of isomorphism between LM representations and entity-centric knowledge structures in the real world, providing insights into how LMs internally organize and use entity information.
>
---
#### [new 075] Coding Agents with Multimodal Browsing are Generalist Problem Solvers
- **分类: cs.CL**

- **简介: 该论文旨在解决AI代理泛化能力差的问题，提出通用代理OpenHands-Versa，仅使用代码编辑、网页搜索等通用工具，在多个复杂任务（如SWE-Bench、GAIA）上表现优于专用代理，展示了通用工具的有效性。**

- **链接: [http://arxiv.org/pdf/2506.03011v1](http://arxiv.org/pdf/2506.03011v1)**

> **作者:** Aditya Bharat Soni; Boxuan Li; Xingyao Wang; Valerie Chen; Graham Neubig
>
> **摘要:** Modern human labor is characterized by specialization; we train for years and develop particular tools that allow us to perform well across a variety of tasks. In addition, AI agents have been specialized for domains such as software engineering, web navigation, and workflow automation. However, this results in agents that are good for one thing but fail to generalize beyond their intended scope. One reason for this is that agent developers provide a highly specialized set of tools or make architectural decisions optimized for a specific use case or benchmark. In this work, we ask the question: what is the minimal set of general tools that can be used to achieve high performance across a diverse set of tasks? Our answer is OpenHands-Versa, a generalist agent built with a modest number of general tools: code editing and execution, web search, as well as multimodal web browsing and file access. Importantly, OpenHands-Versa demonstrates superior or competitive performance over leading specialized agents across three diverse and challenging benchmarks: SWE-Bench Multimodal, GAIA, and The Agent Company, outperforming the best-performing previously published results with absolute improvements in success rate of 9.1, 1.3, and 9.1 points respectively. Further, we show how existing state-of-the-art multi-agent systems fail to generalize beyond their target domains. These results demonstrate the feasibility of developing a generalist agent to solve diverse tasks and establish OpenHands-Versa as a strong baseline for future research.
>
---
#### [new 076] GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出GUI-Actor，旨在解决基于视觉语言模型（VLM）的GUI代理中的视觉定位问题。该方法通过引入注意力机制和一个用于评估动作区域的验证器，实现无需坐标的视觉定位，提升模型在不同屏幕分辨率和布局上的泛化能力，并在多个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.03143v1](http://arxiv.org/pdf/2506.03143v1)**

> **作者:** Qianhui Wu; Kanzhi Cheng; Rui Yang; Chaoyun Zhang; Jianwei Yang; Huiqiang Jiang; Jian Mu; Baolin Peng; Bo Qiao; Reuben Tan; Si Qin; Lars Liden; Qingwei Lin; Huan Zhang; Tong Zhang; Jianbing Zhang; Dongmei Zhang; Jianfeng Gao
>
> **摘要:** One of the principal challenges in building VLM-powered GUI agents is visual grounding, i.e., localizing the appropriate screen region for action execution based on both the visual content and the textual plans. Most existing work formulates this as a text-based coordinate generation task. However, these approaches suffer from several limitations: weak spatial-semantic alignment, inability to handle ambiguous supervision targets, and a mismatch between the dense nature of screen coordinates and the coarse, patch-level granularity of visual features extracted by models like Vision Transformers. In this paper, we propose GUI-Actor, a VLM-based method for coordinate-free GUI grounding. At its core, GUI-Actor introduces an attention-based action head that learns to align a dedicated <ACTOR> token with all relevant visual patch tokens, enabling the model to propose one or more action regions in a single forward pass. In line with this, we further design a grounding verifier to evaluate and select the most plausible action region from the candidates proposed for action execution. Extensive experiments show that GUI-Actor outperforms prior state-of-the-art methods on multiple GUI action grounding benchmarks, with improved generalization to unseen screen resolutions and layouts. Notably, GUI-Actor-7B even surpasses UI-TARS-72B (38.1) on ScreenSpot-Pro, achieving scores of 40.7 with Qwen2-VL and 44.6 with Qwen2.5-VL as backbones. Furthermore, by incorporating the verifier, we find that fine-tuning only the newly introduced action head (~100M parameters for 7B model) while keeping the VLM backbone frozen is sufficient to achieve performance comparable to previous state-of-the-art models, highlighting that GUI-Actor can endow the underlying VLM with effective grounding capabilities without compromising its general-purpose strengths.
>
---
#### [new 077] Prosodic Structure Beyond Lexical Content: A Study of Self-Supervised Learning
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音韵律结构在无监督学习中的建模，旨在分析韵律（如语调、节奏和响度）在不依赖词汇内容的情况下对语言理解的贡献。通过提出“掩码韵律模型”，探索其在词边界识别和情感识别等任务上的效果，比较不同时间尺度对结构学习的影响，揭示了自监督学习在复杂韵律结构建模中的优势。**

- **链接: [http://arxiv.org/pdf/2506.02584v1](http://arxiv.org/pdf/2506.02584v1)**

> **作者:** Sarenne Wallbridge; Christoph Minixhofer; Catherine Lai; Peter Bell
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** People exploit the predictability of lexical structures during text comprehension. Though predictable structure is also present in speech, the degree to which prosody, e.g. intonation, tempo, and loudness, contributes to such structure independently of the lexical content is unclear. This study leverages self-supervised learning (SSL) to examine the temporal granularity of structures in the acoustic correlates of prosody. Representations from our proposed Masked Prosody Model can predict perceptual labels dependent on local information, such as word boundaries, but provide the most value for labels involving longer-term structures, like emotion recognition. Probing experiments across various perceptual labels show strong relative gains over untransformed pitch, energy, and voice activity features. Our results reveal the importance of SSL training objective timescale and highlight the value of complex SSL-encoded structures compared to more constrained classical structures.
>
---
#### [new 078] DIAMOND: An LLM-Driven Agent for Context-Aware Baseball Highlight Summarization
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于体育视频摘要任务，旨在解决传统方法在捕捉比赛亮点时缺乏战略深度和叙事连贯性问题。论文提出DIAMOND框架，结合统计分析与大语言模型的自然语言理解，提升棒球比赛关键时刻的识别效果。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02351v1](http://arxiv.org/pdf/2506.02351v1)**

> **作者:** Jeonghun Kang; Soonmok Kwon; Joonseok Lee; Byung-Hak Kim
>
> **备注:** To appear in the First REALM (Research on Agent Language Models) workshop at ACL 2025
>
> **摘要:** Traditional approaches -- such as Win Probability Added (WPA)-based ranking or computer vision-driven event detection -- can identify scoring plays but often miss strategic depth, momentum shifts, and storyline progression. Manual curation remains the gold standard but is resource-intensive and not scalable. We introduce DIAMOND, an LLM-driven agent for context-aware baseball highlight summarization that integrates structured sports analytics with natural language reasoning. DIAMOND leverages sabermetric features -- Win Expectancy, WPA, and Leverage Index -- to quantify play importance, while an LLM module enhances selection based on contextual narrative value. This hybrid approach ensures both quantitative rigor and qualitative richness, surpassing the limitations of purely statistical or vision-based systems. Evaluated on five diverse Korean Baseball Organization League games, DIAMOND improves F1-score from 42.9% (WPA-only) to 84.8%, outperforming both commercial and statistical baselines. Though limited in scale, our results highlight the potential of modular, interpretable agent-based frameworks for event-level summarization in sports and beyond.
>
---
#### [new 079] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Morphology in Modern Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在探究现代大语言模型如何编码词汇身份与屈折形态。通过训练分类器分析模型各层激活状态，发现词汇信息早期线性集中、后期非线性处理，而屈折信息始终线性可分。研究覆盖多种架构与训练方式的16个模型，揭示其编码模式的一致性，暗示这些特性可能对预测下一词元至关重要。**

- **链接: [http://arxiv.org/pdf/2506.02132v1](http://arxiv.org/pdf/2506.02132v1)**

> **作者:** Michael Li; Nishant Subramani
>
> **摘要:** Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information is rooted in studies of early models like BERT and GPT-2. To better understand today's language models, we investigate how both classical architectures (BERT, DeBERTa, GPT-2)and contemporary large language models (Pythia, OLMo-2, Gemma-2, Qwen2.5, Llama-3.1) represent lexical identity and inflectional morphology. We train linear and nonlinear classifiers on layer-wise activations to predict word lemmas and inflectional features. We discover that models concentrate lexical information linearly in early layers and increasingly nonlinearly in later layers, while keeping inflectional information uniformly accessible and linearly separable throughout the layers. Further analysis reveals that these models encode inflectional morphology through generalizable abstractions, but rely predominantly on memorization to encode lexical identity. Remarkably, these patterns emerge across all 16 models we test, despite differences in architecture, size, and training regime (including pretrained and instruction-tuned variants). This consistency suggests that, despite substantial advances in LLM technologies, transformer models organize linguistic information in similar ways, indicating that these properties could be fundamental for next token prediction and are learned early during pretraining. Our code is available at https://github.com/ml5885/model_internal_sleuthing.
>
---
#### [new 080] No Free Lunch in Active Learning: LLM Embedding Quality Dictates Query Strategy Success
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于文本分类任务，研究主动学习中查询策略的有效性。它探讨了大语言模型生成的嵌入质量对主动学习性能的影响，使用MTEB榜单上的模型和多样化的分类任务进行实验。论文发现嵌入质量显著影响策略效果，推荐根据具体任务评估选择策略。**

- **链接: [http://arxiv.org/pdf/2506.01992v1](http://arxiv.org/pdf/2506.01992v1)**

> **作者:** Lukas Rauch; Moritz Wirth; Denis Huseljic; Marek Herde; Bernhard Sick; Matthias Aßenmacher
>
> **备注:** under review @NeurIPS2025
>
> **摘要:** The advent of large language models (LLMs) capable of producing general-purpose representations lets us revisit the practicality of deep active learning (AL): By leveraging frozen LLM embeddings, we can mitigate the computational costs of iteratively fine-tuning large backbones. This study establishes a benchmark and systematically investigates the influence of LLM embedding quality on query strategies in deep AL. We employ five top-performing models from the massive text embedding benchmark (MTEB) leaderboard and two baselines for ten diverse text classification tasks. Our findings reveal key insights: First, initializing the labeled pool using diversity-based sampling synergizes with high-quality embeddings, boosting performance in early AL iterations. Second, the choice of the optimal query strategy is sensitive to embedding quality. While the computationally inexpensive Margin sampling can achieve performance spikes on specific datasets, we find that strategies like Badge exhibit greater robustness across tasks. Importantly, their effectiveness is often enhanced when paired with higher-quality embeddings. Our results emphasize the need for context-specific evaluation of AL strategies, as performance heavily depends on embedding quality and the target task.
>
---
#### [new 081] ImpRAG: Retrieval-Augmented Generation with Implicit Queries
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统RAG系统依赖显式查询导致泛化能力受限的问题。作者提出ImpRAG，一种无需查询的RAG系统，通过隐式表达信息需求，统一检索与生成过程，提升模型在多种知识密集型任务上的表现。**

- **链接: [http://arxiv.org/pdf/2506.02279v1](http://arxiv.org/pdf/2506.02279v1)**

> **作者:** Wenzheng Zhang; Xi Victoria Lin; Karl Stratos; Wen-tau Yih; Mingda Chen
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems traditionally treat retrieval and generation as separate processes, requiring explicit textual queries to connect them. This separation can limit the ability of models to generalize across diverse tasks. In this work, we propose a query-free RAG system, named ImpRAG, which integrates retrieval and generation into a unified model. ImpRAG allows models to implicitly express their information needs, eliminating the need for human-specified queries. By dividing pretrained decoder-only language models into specialized layer groups, ImpRAG optimizes retrieval and generation tasks simultaneously. Our approach employs a two-stage inference process, using the same model parameters and forward pass for both retrieval and generation, thereby minimizing the disparity between retrievers and language models. Experiments on 8 knowledge-intensive tasks demonstrate that ImpRAG achieves 3.6-11.5 improvements in exact match scores on unseen tasks with diverse formats, highlighting its effectiveness in enabling models to articulate their own information needs and generalize across tasks. Our analysis underscores the importance of balancing retrieval and generation parameters and leveraging generation perplexities as retrieval training objectives for enhanced performance.
>
---
#### [new 082] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态持续指令调优任务，旨在解决现有方法在持续学习中易遗忘旧任务的问题。作者提出BranchLoRA框架，通过分支机制和任务路由提升模型效率与性能，并缓解灾难性遗忘。**

- **链接: [http://arxiv.org/pdf/2506.02041v1](http://arxiv.org/pdf/2506.02041v1)**

> **作者:** Duzhen Zhang; Yong Ren; Zhong-Zhi Li; Yahan Yu; Jiahua Dong; Chenxing Li; Zhilong Ji; Jinfeng Bai
>
> **备注:** Accepted by ACL2025 Main Conference
>
> **摘要:** Multimodal Continual Instruction Tuning (MCIT) aims to finetune Multimodal Large Language Models (MLLMs) to continually align with human intent across sequential tasks. Existing approaches often rely on the Mixture-of-Experts (MoE) LoRA framework to preserve previous instruction alignments. However, these methods are prone to Catastrophic Forgetting (CF), as they aggregate all LoRA blocks via simple summation, which compromises performance over time. In this paper, we identify a critical parameter inefficiency in the MoELoRA framework within the MCIT context. Based on this insight, we propose BranchLoRA, an asymmetric framework to enhance both efficiency and performance. To mitigate CF, we introduce a flexible tuning-freezing mechanism within BranchLoRA, enabling branches to specialize in intra-task knowledge while fostering inter-task collaboration. Moreover, we incrementally incorporate task-specific routers to ensure an optimal branch distribution over time, rather than favoring the most recent task. To streamline inference, we introduce a task selector that automatically routes test inputs to the appropriate router without requiring task identity. Extensive experiments on the latest MCIT benchmark demonstrate that BranchLoRA significantly outperforms MoELoRA and maintains its superiority across various MLLM sizes.
>
---
#### [new 083] Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态报告生成任务，旨在解决自动生成图文交错报告的问题。现有方法主要生成纯文本，缺乏有效图表整合。作者提出FDV结构化图表描述和Multimodal DeepResearcher框架，分四阶段生成图文报告，并构建评估数据集MultimodalReportBench。实验表明其方法显著优于基线。**

- **链接: [http://arxiv.org/pdf/2506.02454v1](http://arxiv.org/pdf/2506.02454v1)**

> **作者:** Zhaorui Yang; Bo Pan; Han Wang; Yiyao Wang; Xingyu Liu; Minfeng Zhu; Bo Zhang; Wei Chen
>
> **备注:** 47 pages
>
> **摘要:** Visualizations play a crucial part in effective communication of concepts and information. Recent advances in reasoning and retrieval augmented generation have enabled Large Language Models (LLMs) to perform deep research and generate comprehensive reports. Despite its progress, existing deep research frameworks primarily focus on generating text-only content, leaving the automated generation of interleaved texts and visualizations underexplored. This novel task poses key challenges in designing informative visualizations and effectively integrating them with text reports. To address these challenges, we propose Formal Description of Visualization (FDV), a structured textual representation of charts that enables LLMs to learn from and generate diverse, high-quality visualizations. Building on this representation, we introduce Multimodal DeepResearcher, an agentic framework that decomposes the task into four stages: (1) researching, (2) exemplar report textualization, (3) planning, and (4) multimodal report generation. For the evaluation of generated multimodal reports, we develop MultimodalReportBench, which contains 100 diverse topics served as inputs along with 5 dedicated metrics. Extensive experiments across models and evaluation methods demonstrate the effectiveness of Multimodal DeepResearcher. Notably, utilizing the same Claude 3.7 Sonnet model, Multimodal DeepResearcher achieves an 82\% overall win rate over the baseline method.
>
---
#### [new 084] HENT-SRT: Hierarchical Efficient Neural Transducer with Self-Distillation for Joint Speech Recognition and Translation
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别与翻译任务，旨在解决神经转换器（NT）在语音翻译中的词序重排和性能下降问题。作者提出HENT-SRT框架，通过任务分解、自蒸馏、一致性正则化等方法提升效果，并引入层级编码器等优化训练效率，实现了新SOTA性能。**

- **链接: [http://arxiv.org/pdf/2506.02157v1](http://arxiv.org/pdf/2506.02157v1)**

> **作者:** Amir Hussein; Cihan Xiao; Matthew Wiesner; Dan Povey; Leibny Paola Garcia; Sanjeev Khudanpur
>
> **摘要:** Neural transducers (NT) provide an effective framework for speech streaming, demonstrating strong performance in automatic speech recognition (ASR). However, the application of NT to speech translation (ST) remains challenging, as existing approaches struggle with word reordering and performance degradation when jointly modeling ASR and ST, resulting in a gap with attention-based encoder-decoder (AED) models. Existing NT-based ST approaches also suffer from high computational training costs. To address these issues, we propose HENT-SRT (Hierarchical Efficient Neural Transducer for Speech Recognition and Translation), a novel framework that factorizes ASR and translation tasks to better handle reordering. To ensure robust ST while preserving ASR performance, we use self-distillation with CTC consistency regularization. Moreover, we improve computational efficiency by incorporating best practices from ASR transducers, including a down-sampled hierarchical encoder, a stateless predictor, and a pruned transducer loss to reduce training complexity. Finally, we introduce a blank penalty during decoding, reducing deletions and improving translation quality. Our approach is evaluated on three conversational datasets Arabic, Spanish, and Mandarin achieving new state-of-the-art performance among NT models and substantially narrowing the gap with AED-based systems.
>
---
#### [new 085] Knowledge or Reasoning? A Close Look at How LLMs Think Across Domains
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探究大语言模型在医疗和数学领域中的推理过程。它通过分解知识与推理，提出评估框架衡量知识正确性和推理质量，分析不同训练方法对模型推理能力的影响。**

- **链接: [http://arxiv.org/pdf/2506.02126v1](http://arxiv.org/pdf/2506.02126v1)**

> **作者:** Juncheng Wu; Sheng Liu; Haoqin Tu; Hang Yu; Xiaoke Huang; James Zou; Cihang Xie; Yuyin Zhou
>
> **备注:** 17 pages, preprint
>
> **摘要:** Recent advances in reasoning-enhanced Large Language Models such as OpenAI-o1/3 and DeepSeek-R1 have significantly improved performance on complex tasks. However, the quality and transparency of their internal reasoning processes remain underexplored. This work moves beyond the final-answer accuracy and investigates step-by-step reasoning in the medical and mathematical domains by explicitly decomposing the thinking trajectories into two parts: knowledge and reasoning. Specifically, we introduce a fine-grained evaluation framework that judges: (1) the correctness of knowledge used (measured by Knowledge Index (KI)) and (2) the quality of reasoning (measured by Information Gain (InfoGain)). Using this framework, we study R1-distilled and base Qwen models trained with supervised fine-tuning (SFT) and/or reinforcement learning (RL) in the medical and math domains. Three intriguing findings emerge: (1) The general reasoning abilities in R1-distilled models do not transfer effectively to the medical domain through either SFT or RL. (2) SFT raises final-answer accuracy in both domains, but often at the cost of reasoning quality: InfoGain drops by 38.9% on average compared with untrained models; In the medical domain, however, SFT remains crucial because domain knowledge is indispensable. (3) RL enhances medical reasoning by pruning inaccurate or irrelevant knowledge from reasoning paths, thereby improving both reasoning accuracy and knowledge correctness.
>
---
#### [new 086] Leveraging Information Retrieval to Enhance Spoken Language Understanding Prompts in Few-Shot Learning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决少样本场景下口语理解性能不足的问题。通过利用信息检索技术选择示例构建增强提示，实验证明该方法能有效提升口语理解任务的表现，且不增加提示长度。**

- **链接: [http://arxiv.org/pdf/2506.03035v1](http://arxiv.org/pdf/2506.03035v1)**

> **作者:** Pierre Lepagnol; Sahar Ghannay; Thomas Gerald; Christophe Servan; Sophie Rosset
>
> **备注:** Conference paper accepted to INTERSPEECH 2025
>
> **摘要:** Understanding user queries is fundamental in many applications, such as home assistants, booking systems, or recommendations. Accordingly, it is crucial to develop accurate Spoken Language Understanding (SLU) approaches to ensure the reliability of the considered system. Current State-of-the-Art SLU techniques rely on large amounts of training data; however, only limited annotated examples are available for specific tasks or languages. In the meantime, instruction-tuned large language models (LLMs) have shown exceptional performance on unseen tasks in a few-shot setting when provided with adequate prompts. In this work, we propose to explore example selection by leveraging Information retrieval (IR) approaches to build an enhanced prompt that is applied to an SLU task. We evaluate the effectiveness of the proposed method on several SLU benchmarks. Experimental results show that lexical IR methods significantly enhance performance without increasing prompt length.
>
---
#### [new 087] Entity-Augmented Neuroscience Knowledge Retrieval Using Ontology and Semantic Understanding Capability of LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱构建与信息检索任务，旨在解决从无标注神经科学文献中高效抽取知识并提升问答效果的问题。利用大语言模型、神经科学本体和文本嵌入技术构建知识图谱，并提出实体增强的检索算法。实验表明方法在实体抽取和问答准确率上表现优异。**

- **链接: [http://arxiv.org/pdf/2506.03145v1](http://arxiv.org/pdf/2506.03145v1)**

> **作者:** Pralaypati Ta; Sriram Venkatesaperumal; Keerthi Ram; Mohanasankar Sivaprakasam
>
> **摘要:** Neuroscience research publications encompass a vast wealth of knowledge. Accurately retrieving existing information and discovering new insights from this extensive literature is essential for advancing the field. However, when knowledge is dispersed across multiple sources, current state-of-the-art retrieval methods often struggle to extract the necessary information. A knowledge graph (KG) can integrate and link knowledge from multiple sources, but existing methods for constructing KGs in neuroscience often rely on labeled data and require domain expertise. Acquiring large-scale, labeled data for a specialized area like neuroscience presents significant challenges. This work proposes novel methods for constructing KG from unlabeled large-scale neuroscience research corpus utilizing large language models (LLM), neuroscience ontology, and text embeddings. We analyze the semantic relevance of neuroscience text segments identified by LLM for building the knowledge graph. We also introduce an entity-augmented information retrieval algorithm to extract knowledge from the KG. Several experiments were conducted to evaluate the proposed approaches, and the results demonstrate that our methods significantly enhance knowledge discovery from the unlabeled neuroscience research corpus. It achieves an F1 score of 0.84 for entity extraction, and the knowledge obtained from the KG improves answers to over 54% of the questions.
>
---
#### [new 088] Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与强化学习交叉任务，旨在解决大型语言模型在仅使用数值反馈进行强化学习时遇到的性能瓶颈问题。作者提出了一种结合自然语言和数值反馈的在线强化学习框架Critique-GRPO，通过引入基于批评的自然语言反馈提升模型推理能力，实验证明其在多个复杂推理任务上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.03106v1](http://arxiv.org/pdf/2506.03106v1)**

> **作者:** Xiaoying Zhang; Hao Sun; Yipeng Zhang; Kaituo Feng; Chao Yang; Helen Meng
>
> **备注:** 38 pages
>
> **摘要:** Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided refinements simultaneously while maintaining exploration. Extensive experiments using Qwen2.5-7B-Base and Qwen3-8B-Base show that Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.5% and 5%, respectively. Notably, Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals two critical insights about policy exploration: (1) higher entropy does not always guarantee efficient learning from exploration, and (2) longer responses do not necessarily lead to more effective exploration.
>
---
#### [new 089] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育科技任务，旨在解决多语言环境下儿童语言学习的个性化与互动性不足问题。作者设计了多语言多模态对话系统SingaKids，通过图片描述任务促进语言学习，并优化模型以适应不同语言和儿童认知水平。**

- **链接: [http://arxiv.org/pdf/2506.02412v1](http://arxiv.org/pdf/2506.02412v1)**

> **作者:** Zhengyuan Liu; Geyu Lin; Hui Li Tan; Huayun Zhang; Yanfeng Lu; Xiaoxue Gao; Stella Xin Yin; He Sun; Hock Huan Goh; Lung Hsiang Wong; Nancy F. Chen
>
> **备注:** ACL 2025 Industry Track
>
> **摘要:** The integration of generative artificial intelligence into educational applications has enhanced personalized and interactive learning experiences, and it shows strong potential to promote young learners language acquisition. However, it is still challenging to ensure consistent and robust performance across different languages and cultural contexts, and kids-friendly design requires simplified instructions, engaging interactions, and age-appropriate scaffolding to maintain motivation and optimize learning outcomes. In this work, we introduce SingaKids, a dialogic tutor designed to facilitate language learning through picture description tasks. Our system integrates dense image captioning, multilingual dialogic interaction, speech understanding, and engaging speech generation to create an immersive learning environment in four languages: English, Mandarin, Malay, and Tamil. We further improve the system through multilingual pre-training, task-specific tuning, and scaffolding optimization. Empirical studies with elementary school students demonstrate that SingaKids provides effective dialogic teaching, benefiting learners at different performance levels.
>
---
#### [new 090] Facts Do Care About Your Language: Assessing Answer Quality of Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估多语言大模型（如Llama3.1）在不同语言中回答事实性问题的准确性。论文关注教育场景，发现模型在非英语语言中表现较差，且存在对稀有语言的偏见，强调了确保多语言模型事实准确性的必要性。**

- **链接: [http://arxiv.org/pdf/2506.03051v1](http://arxiv.org/pdf/2506.03051v1)**

> **作者:** Yuval Kansal; Shmuel Berman; Lydia Liu
>
> **摘要:** Factuality is a necessary precursor to useful educational tools. As adoption of Large Language Models (LLMs) in education continues of grow, ensuring correctness in all settings is paramount. Despite their strong English capabilities, LLM performance in other languages is largely untested. In this work, we evaluate the correctness of the Llama3.1 family of models in answering factual questions appropriate for middle and high school students. We demonstrate that LLMs not only provide extraneous and less truthful information, but also exacerbate existing biases against rare languages.
>
---
#### [new 091] XToM: Exploring the Multilingual Theory of Mind for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型（LLMs）在多语言环境下推理心理状态的能力不足问题。作者构建了XToM多语言基准，评估LLMs在五种语言中的心智理论表现，发现模型虽具语言理解能力，但在跨语言心智推理上存在局限。**

- **链接: [http://arxiv.org/pdf/2506.02461v1](http://arxiv.org/pdf/2506.02461v1)**

> **作者:** Chunkit Chan; Yauwai Yim; Hongchuan Zeng; Zhiying Zou; Xinyuan Cheng; Zhifan Sun; Zheye Deng; Kawai Chung; Yuzhuo Ao; Yixiang Fan; Cheng Jiayang; Ercong Nie; Ginny Y. Wong; Helmut Schmid; Hinrich Schütze; Simon See; Yangqiu Song
>
> **摘要:** Theory of Mind (ToM), the ability to infer mental states in others, is pivotal for human social cognition. Existing evaluations of ToM in LLMs are largely limited to English, neglecting the linguistic diversity that shapes human cognition. This limitation raises a critical question: can LLMs exhibit Multilingual Theory of Mind, which is the capacity to reason about mental states across diverse linguistic contexts? To address this gap, we present XToM, a rigorously validated multilingual benchmark that evaluates ToM across five languages and incorporates diverse, contextually rich task scenarios. Using XToM, we systematically evaluate LLMs (e.g., DeepSeek R1), revealing a pronounced dissonance: while models excel in multilingual language understanding, their ToM performance varies across languages. Our findings expose limitations in LLMs' ability to replicate human-like mentalizing across linguistic contexts.
>
---
#### [new 092] AnswerCarefully: A Dataset for Improving the Safety of Japanese LLM Output
- **分类: cs.CL**

- **简介: 该论文提出了AnswerCarefully数据集，用于提升日语大语言模型（LLM）输出的安全性。任务是构建一个包含1800对问题与参考答案的数据集，涵盖多种风险类别，反映日本社会文化背景。论文展示了该数据集在微调和评估LLM安全性方面的有效性，并提供英文翻译以支持多语言扩展。**

- **链接: [http://arxiv.org/pdf/2506.02372v1](http://arxiv.org/pdf/2506.02372v1)**

> **作者:** Hisami Suzuki; Satoru Katsumata; Takashi Kodama; Tetsuro Takahashi; Kouta Nakayama; Satoshi Sekine
>
> **摘要:** In this paper we present AnswerCarefully, a dataset for promoting the safety and appropriateness of Japanese LLM outputs. The dataset consists of 1,800 pairs of questions and reference answers, where the questions require special attention in answering. It covers a wide range of risk categories established in prior English-language datasets, but the data samples are original in that they are manually created to reflect the socio-cultural context of LLM usage in Japan. We show that using this dataset for instruction to fine-tune a Japanese LLM led to improved output safety without compromising the utility of general responses. We also report the results of a safety evaluation of 12 Japanese LLMs using this dataset as a benchmark. Finally, we describe the latest update on the dataset which provides English translations and annotations of the questions, aimed at facilitating the derivation of similar datasets in different languages and regions.
>
---
#### [new 093] AUTOCIRCUIT-RL: Reinforcement Learning-Driven LLM for Automated Circuit Topology Generation
- **分类: cs.CL**

- **简介: 该论文属于电子设计自动化任务，旨在解决模拟电路拓扑结构自动生成的问题。通过结合大语言模型与强化学习，提出AUTOCIRCUIT-RL框架，分两阶段优化生成过程，提升生成电路的有效性、效率与多样性。**

- **链接: [http://arxiv.org/pdf/2506.03122v1](http://arxiv.org/pdf/2506.03122v1)**

> **作者:** Prashanth Vijayaraghavan; Luyao Shi; Ehsan Degan; Vandana Mukherjee; Xin Zhang
>
> **备注:** 9 Pages (Content), 4 Pages (Appendix), 7 figures, ICML'2025
>
> **摘要:** Analog circuit topology synthesis is integral to Electronic Design Automation (EDA), enabling the automated creation of circuit structures tailored to specific design requirements. However, the vast design search space and strict constraint adherence make efficient synthesis challenging. Leveraging the versatility of Large Language Models (LLMs), we propose AUTOCIRCUIT-RL,a novel reinforcement learning (RL)-based framework for automated analog circuit synthesis. The framework operates in two phases: instruction tuning, where an LLM learns to generate circuit topologies from structured prompts encoding design constraints, and RL refinement, which further improves the instruction-tuned model using reward models that evaluate validity, efficiency, and output voltage. The refined model is then used directly to generate topologies that satisfy the design constraints. Empirical results show that AUTOCIRCUIT-RL generates ~12% more valid circuits and improves efficiency by ~14% compared to the best baselines, while reducing duplicate generation rates by ~38%. It achieves over 60% success in synthesizing valid circuits with limited training data, demonstrating strong generalization. These findings highlight the framework's effectiveness in scaling to complex circuits while maintaining efficiency and constraint adherence, marking a significant advancement in AI-driven circuit design.
>
---
#### [new 094] Comparative Analysis of AI Agent Architectures for Entity Relationship Classification
- **分类: cs.CL; cs.AI; I.2.7; I.2.1**

- **简介: 该论文属于信息抽取任务中的实体关系分类问题，旨在解决标注数据有限和关系结构复杂场景下的分类挑战。论文比较了三种基于大语言模型的AI代理架构：反思自评、层次任务分解和多代理动态示例生成机制。通过跨多个领域和模型后端的系统比较实验，发现多代理协作在性能上优于标准少样本提示方法，并接近微调模型效果。研究为构建模块化、通用的关系抽取系统提供了实用指导。**

- **链接: [http://arxiv.org/pdf/2506.02426v1](http://arxiv.org/pdf/2506.02426v1)**

> **作者:** Maryam Berijanian; Kuldeep Singh; Amin Sehati
>
> **摘要:** Entity relationship classification remains a challenging task in information extraction, especially in scenarios with limited labeled data and complex relational structures. In this study, we conduct a comparative analysis of three distinct AI agent architectures designed to perform relation classification using large language models (LLMs). The agentic architectures explored include (1) reflective self-evaluation, (2) hierarchical task decomposition, and (3) a novel multi-agent dynamic example generation mechanism, each leveraging different modes of reasoning and prompt adaptation. In particular, our dynamic example generation approach introduces real-time cooperative and adversarial prompting. We systematically compare their performance across multiple domains and model backends. Our experiments demonstrate that multi-agent coordination consistently outperforms standard few-shot prompting and approaches the performance of fine-tuned models. These findings offer practical guidance for the design of modular, generalizable LLM-based systems for structured relation extraction. The source codes and dataset are available at \href{https://github.com/maryambrj/ALIEN.git}{https://github.com/maryambrj/ALIEN.git}.
>
---
#### [new 095] A Controllable Examination for Long-Context Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决现有长上下文语言模型评估框架的局限性。作者提出了新基准LongBioBench，通过生成人工传记提供可控环境，评估模型的理解、推理和可信度。实验表明，当前模型在语义理解和基本推理方面仍存在不足，并揭示了现有合成基准设计的问题及长上下文持续预训练的影响。**

- **链接: [http://arxiv.org/pdf/2506.02921v1](http://arxiv.org/pdf/2506.02921v1)**

> **作者:** Yijun Yang; Zeyu Huang; Wenhao Zhu; Zihan Qiu; Fei Yuan; Jeff Z. Pan; Ivan Titov
>
> **备注:** Preprint
>
> **摘要:** Existing frameworks for evaluating long-context language models (LCLM) can be broadly categorized into real-world and synthetic tasks. Despite their utility, both approaches are accompanied by certain intrinsic limitations. Real-world tasks are too complex to interpret or characterize and are susceptible to data contamination. In contrast, synthetic tasks often adopt the needle-in-the-haystack (NIAH) format, wherein a lack of coherence between the "needle" and the "haystack" compromises their validity as proxies for realistic applications. In response to these challenges, we posit that an ideal long-context evaluation framework should be characterized by three essential features: $\textit{seamless context}$, $\textit{controllable setting}$, and $\textit{sound evaluation}$. This study introduces $\textbf{LongBioBench}$, a novel benchmark that utilizes artificially generated biographies as a controlled environment for assessing LCLMs across dimensions of $\textit{understanding}$, $\textit{reasoning}$, and $\textit{trustworthiness}$. Our experimental evaluation, which includes $\textbf{18}$ LCLMs in total, demonstrates that most models still exhibit deficiencies in semantic understanding and elementary reasoning over retrieved results and are less trustworthy as context length increases. Our further analysis indicates some design choices employed by existing synthetic benchmarks, such as contextual non-coherence, numerical needles, and the absence of distractors, rendering them vulnerable to test the model long-context capabilities. Moreover, we also reveal that long-context continual pretraining primarily adjusts RoPE embedding to accommodate extended context lengths. To sum up, compared to previous synthetic benchmarks, LongBioBench achieves a better trade-off between mirroring authentic language tasks and maintaining controllability, and is highly interpretable and configurable.
>
---
#### [new 096] Decompose, Plan in Parallel, and Merge: A Novel Paradigm for Large Language Models based Planning with Multiple Constraints
- **分类: cs.CL**

- **简介: 论文属于任务规划领域，旨在解决大语言模型在复杂约束下规划任务中的级联错误和效率问题。提出了DPPM方法，通过分解任务、并行子任务规划、合并子计划，并引入验证与优化模块。实验表明其在旅行规划任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02683v1](http://arxiv.org/pdf/2506.02683v1)**

> **作者:** Zhengdong Lu; Weikai Lu; Yiling Tao; Yun Dai; ZiXuan Chen; Huiping Zhuang; Cen Chen; Hao Peng; Ziqian Zeng
>
> **摘要:** Despite significant advances in Large Language Models (LLMs), planning tasks still present challenges for LLM-based agents. Existing planning methods face two key limitations: heavy constraints and cascading errors. To address these limitations, we propose a novel parallel planning paradigm, which Decomposes, Plans for subtasks in Parallel, and Merges subplans into a final plan (DPPM). Specifically, DPPM decomposes the complex task based on constraints into subtasks, generates the subplan for each subtask in parallel, and merges them into a global plan. In addition, our approach incorporates a verification and refinement module, enabling error correction and conflict resolution. Experimental results demonstrate that DPPM significantly outperforms existing methods in travel planning tasks.
>
---
#### [new 097] ORPP: Self-Optimizing Role-playing Prompts to Enhance Language Model Capabilities
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决如何高效优化大语言模型的提示以提升性能。现有方法计算开销大或依赖模型自身优化能力，限制了应用范围。论文提出ORPP框架，通过优化角色扮演类提示，在小样本上迭代生成高质量提示，并利用少样本学习推广到更多样本，提升了模型表现及兼容性。**

- **链接: [http://arxiv.org/pdf/2506.02480v1](http://arxiv.org/pdf/2506.02480v1)**

> **作者:** Yifan Duan; Yihong Tang; Kehai Chen; Liqiang Nie; Min Zhang
>
> **摘要:** High-quality prompts are crucial for eliciting outstanding performance from large language models (LLMs) on complex tasks. Existing research has explored model-driven strategies for prompt optimization. However, these methods often suffer from high computational overhead or require strong optimization capabilities from the model itself, which limits their broad applicability.To address these challenges, we propose ORPP (Optimized Role-Playing Prompt),a framework that enhances model performance by optimizing and generating role-playing prompts. The core idea of ORPP is to confine the prompt search space to role-playing scenarios, thereby fully activating the model's intrinsic capabilities through carefully crafted, high-quality role-playing prompts. Specifically, ORPP first performs iterative optimization on a small subset of training samples to generate high-quality role-playing prompts. Then, leveraging the model's few-shot learning capability, it transfers the optimization experience to efficiently generate suitable prompts for the remaining samples.Our experimental results show that ORPP not only matches but in most cases surpasses existing mainstream prompt optimization methods in terms of performance. Notably, ORPP demonstrates superior "plug-and-play" capability. In most cases, it can be integrated with various other prompt methods and further enhance their effectiveness.
>
---
#### [new 098] GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与知识图谱任务，旨在解决现有GraphRAG模型评估不足的问题。作者构建了大规模领域基准GraphRAG-Bench，包含多跳推理问题、多样任务和全流程评估，以全面衡量模型在图增强生成中的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.02404v1](http://arxiv.org/pdf/2506.02404v1)**

> **作者:** Yilin Xiao; Junnan Dong; Chuang Zhou; Su Dong; Qianwen Zhang; Di Yin; Xing Sun; Xiao Huang
>
> **摘要:** Graph Retrieval Augmented Generation (GraphRAG) has garnered increasing recognition for its potential to enhance large language models (LLMs) by structurally organizing domain-specific corpora and facilitating complex reasoning. However, current evaluations of GraphRAG models predominantly rely on traditional question-answering datasets. Their limited scope in questions and evaluation metrics fails to comprehensively assess the reasoning capacity improvements enabled by GraphRAG models. To address this gap, we introduce GraphRAG-Bench, a large-scale, domain-specific benchmark designed to rigorously evaluate GraphRAG models. Our benchmark offers three key superiorities: \((i)\) Challenging question design. Featuring college-level, domain-specific questions that demand multi-hop reasoning, the benchmark ensures that simple content retrieval is insufficient for problem-solving. For example, some questions require mathematical reasoning or programming. \((ii)\) Diverse task coverage. The dataset includes a broad spectrum of reasoning tasks, multiple-choice, true/false, multi-select, open-ended, and fill-in-the-blank. It spans 16 disciplines in twenty core textbooks. \((iii)\) Holistic evaluation framework. GraphRAG-Bench provides comprehensive assessment across the entire GraphRAG pipeline, including graph construction, knowledge retrieval, and answer generation. Beyond final-answer correctness, it evaluates the logical coherence of the reasoning process. By applying nine contemporary GraphRAG methods to GraphRAG-Bench, we demonstrate its utility in quantifying how graph-based structuring improves model reasoning capabilities. Our analysis reveals critical insights about graph architectures, retrieval efficacy, and reasoning capabilities, offering actionable guidance for the research community.
>
---
#### [new 099] AI Debate Aids Assessment of Controversial Claims
- **分类: cs.CL**

- **简介: 该论文研究AI辩论在争议性主张评估中的作用，旨在解决AI影响力扩大带来的误导风险。通过对比实验，发现AI辩论可提升人类和AI法官的判断准确性，尤其对持有主流观点者效果显著，展示了其在可扩展、抗偏见监督中的潜力。**

- **链接: [http://arxiv.org/pdf/2506.02175v1](http://arxiv.org/pdf/2506.02175v1)**

> **作者:** Salman Rahman; Sheriff Issaka; Ashima Suvarna; Genglin Liu; James Shiffer; Jaeyoung Lee; Md Rizwan Parvez; Hamid Palangi; Shi Feng; Nanyun Peng; Yejin Choi; Julian Michael; Liwei Jiang; Saadia Gabriel
>
> **摘要:** As AI grows more powerful, it will increasingly shape how we understand the world. But with this influence comes the risk of amplifying misinformation and deepening social divides-especially on consequential topics like public health where factual accuracy directly impacts well-being. Scalable Oversight aims to ensure AI truthfulness by enabling humans to supervise systems that may exceed human capabilities--yet humans themselves hold different beliefs and biases that impair their judgment. We study whether AI debate can guide biased judges toward the truth by having two AI systems debate opposing sides of controversial COVID-19 factuality claims where people hold strong prior beliefs. We conduct two studies: one with human judges holding either mainstream or skeptical beliefs evaluating factuality claims through AI-assisted debate or consultancy protocols, and a second examining the same problem with personalized AI judges designed to mimic these different human belief systems. In our human study, we find that debate-where two AI advisor systems present opposing evidence-based arguments-consistently improves judgment accuracy and confidence calibration, outperforming consultancy with a single-advisor system by 10% overall. The improvement is most significant for judges with mainstream beliefs (+15.2% accuracy), though debate also helps skeptical judges who initially misjudge claims move toward accurate views (+4.7% accuracy). In our AI judge study, we find that AI judges with human-like personas achieve even higher accuracy (78.5%) than human judges (70.1%) and default AI judges without personas (69.8%), suggesting their potential for supervising frontier AI models. These findings highlight AI debate as a promising path toward scalable, bias-resilient oversight--leveraging both diverse human and AI judgments to move closer to truth in contested domains.
>
---
#### [new 100] FinChain: A Symbolic Benchmark for Verifiable Chain-of-Thought Financial Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于金融推理任务，旨在解决现有数据集缺乏对中间推理步骤评估的问题。作者构建了FinChain基准，包含54个主题的符号推理模板及可执行代码，支持生成训练数据与跨领域适配，并提出ChainEval指标评估多步推理效果，发现当前模型仍有较大提升空间。**

- **链接: [http://arxiv.org/pdf/2506.02515v1](http://arxiv.org/pdf/2506.02515v1)**

> **作者:** Zhuohan Xie; Dhruv Sahnan; Debopriyo Banerjee; Georgi Georgiev; Rushil Thareja; Hachem Madmoun; Jinyan Su; Aaryamonvikram Singh; Yuxia Wang; Rui Xing; Fajri Koto; Haonan Li; Ivan Koychev; Tanmoy Chakraborty; Salem Lahlou; Veselin Stoyanov; Preslav Nakov
>
> **备注:** 15 pages, 8 figures, 2 tables
>
> **摘要:** Multi-step symbolic reasoning is critical for advancing downstream performance on financial tasks. Yet, benchmarks for systematically evaluating this capability are lacking. Existing datasets like FinQA and ConvFinQA supervise only final numerical answers, without assessing intermediate reasoning steps. To address this, we introduce FinChain, the first symbolic benchmark designed for verifiable Chain-of- Thought (CoT) financial reasoning. Spanning 54 topics across 12 financial domains, Fin- Chain offers five parameterized templates per topic, each varying in reasoning complexity and domain expertise required. Each dataset instance includes an executable Python trace, enabling automatic generation of extensive training data and easy adaptation to other domains. We also introduce ChainEval, a new metric for automatic evaluation of both final answers and intermediate reasoning. Benchmarking 30 LLMs on our dataset, we find that even state-of-the-art models have considerable room for improvement in multi-step financial reasoning. All templates and evaluation metrics for FinChain are available at https: //github.com/mbzuai-nlp/finchain.
>
---
#### [new 101] Adaptive Graph Pruning for Multi-Agent Communication
- **分类: cs.CL; cs.MA**

- **简介: 论文提出了一种自适应图剪枝（AGP）框架，用于多智能体通信，旨在解决固定结构限制。它动态优化智能体数量与通信拓扑，通过两阶段训练策略，在多种任务中实现了高性能、少耗 token 且训练高效的协作效果。**

- **链接: [http://arxiv.org/pdf/2506.02951v1](http://arxiv.org/pdf/2506.02951v1)**

> **作者:** Boyi Li; Zhonghan Zhao; Der-Horng Lee; Gaoang Wang
>
> **摘要:** Large Language Model (LLM) based multi-agent systems have shown remarkable performance in various tasks, especially when enhanced through collaborative communication. However, current methods often rely on a fixed number of agents and static communication structures, limiting their ability to adapt to varying task complexities. In this paper, we propose Adaptive Graph Pruning (AGP), a novel task-adaptive multi-agent collaboration framework that jointly optimizes agent quantity (hard-pruning) and communication topology (soft-pruning). Specifically, our method employs a two-stage training strategy: firstly, independently training soft-pruning networks for different agent quantities to determine optimal agent-quantity-specific complete graphs and positional masks across specific tasks; and then jointly optimizing hard-pruning and soft-pruning within a maximum complete graph to dynamically configure the number of agents and their communication topologies per task. Extensive experiments demonstrate that our approach is: (1) High-performing, achieving state-of-the-art results across six benchmarks and consistently generalizes across multiple mainstream LLM architectures, with a increase in performance of $2.58\%\sim 9.84\%$; (2) Task-adaptive, dynamically constructing optimized communication topologies tailored to specific tasks, with an extremely high performance in all three task categories (general reasoning, mathematical reasoning, and code generation); (3) Token-economical, having fewer training steps and token consumption at the same time, with a decrease in token consumption of $90\%+$; and (4) Training-efficient, achieving high performance with very few training steps compared with other methods. The performance will surpass the existing baselines after about ten steps of training under six benchmarks.
>
---
#### [new 102] Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在分析现代ASR模型（基于Conformer）所依赖的声学线索。通过特征归因技术，研究元音、擦音和塞音的声学特性，揭示模型在时域和频域中的关注重点，提升模型可解释性，并指出潜在的鲁棒性改进方向。**

- **链接: [http://arxiv.org/pdf/2506.02181v1](http://arxiv.org/pdf/2506.02181v1)**

> **作者:** Dennis Fucci; Marco Gaido; Matteo Negri; Mauro Cettolo; Luisa Bentivogli
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Despite significant advances in ASR, the specific acoustic cues models rely on remain unclear. Prior studies have examined such cues on a limited set of phonemes and outdated models. In this work, we apply a feature attribution technique to identify the relevant acoustic cues for a modern Conformer-based ASR system. By analyzing plosives, fricatives, and vowels, we assess how feature attributions align with their acoustic properties in the time and frequency domains, also essential for human speech perception. Our findings show that the ASR model relies on vowels' full time spans, particularly their first two formants, with greater saliency in male speech. It also better captures the spectral characteristics of sibilant fricatives than non-sibilants and prioritizes the release phase in plosives, especially burst characteristics. These insights enhance the interpretability of ASR models and highlight areas for future research to uncover potential gaps in model robustness.
>
---
#### [new 103] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器生成文本检测任务，旨在解决人类与AI共同撰写文本中的细粒度检测问题。作者构建了带词级标注的HACo-Det数据集，并将现有文档级检测方法扩展到词级和句级检测，评估其性能并分析影响因素。**

- **链接: [http://arxiv.org/pdf/2506.02959v1](http://arxiv.org/pdf/2506.02959v1)**

> **作者:** Zhixiong Su; Yichen Wang; Herun Wan; Zhaohan Zhang; Minnan Luo
>
> **摘要:** The misuse of large language models (LLMs) poses potential risks, motivating the development of machine-generated text (MGT) detection. Existing literature primarily concentrates on binary, document-level detection, thereby neglecting texts that are composed jointly by human and LLM contributions. Hence, this paper explores the possibility of fine-grained MGT detection under human-AI coauthoring. We suggest fine-grained detectors can pave pathways toward coauthored text detection with a numeric AI ratio. Specifically, we propose a dataset, HACo-Det, which produces human-AI coauthored texts via an automatic pipeline with word-level attribution labels. We retrofit seven prevailing document-level detectors to generalize them to word-level detection. Then we evaluate these detectors on HACo-Det on both word- and sentence-level detection tasks. Empirical results show that metric-based methods struggle to conduct fine-grained detection with a 0.462 average F1 score, while finetuned models show superior performance and better generalization across domains. However, we argue that fine-grained co-authored text detection is far from solved. We further analyze factors influencing performance, e.g., context window, and highlight the limitations of current methods, pointing to potential avenues for improvement.
>
---
#### [new 104] INESC-ID @ eRisk 2025: Exploring Fine-Tuned, Similarity-Based, and Prompt-Based Approaches to Depression Symptom Identification
- **分类: cs.CL; cs.IR; cs.LG; I.2.7; I.5.4; J.3; H.3.3**

- **简介: 该论文参与eRisk 2025任务1，旨在识别抑郁症状。基于BDI问卷，从句子中检索相关症状。团队采用微调模型、相似度、提示等方法，并使用集成技术，在信息检索评估中取得最优效果。**

- **链接: [http://arxiv.org/pdf/2506.02924v1](http://arxiv.org/pdf/2506.02924v1)**

> **作者:** Diogo A. P. Nunes; Eugénio Ribeiro
>
> **备注:** 12 pages, 1 figure, 6 tables
>
> **摘要:** In this work, we describe our team's approach to eRisk's 2025 Task 1: Search for Symptoms of Depression. Given a set of sentences and the Beck's Depression Inventory - II (BDI) questionnaire, participants were tasked with submitting up to 1,000 sentences per depression symptom in the BDI, sorted by relevance. Participant submissions were evaluated according to standard Information Retrieval (IR) metrics, including Average Precision (AP) and R-Precision (R-PREC). The provided training data, however, consisted of sentences labeled as to whether a given sentence was relevant or not w.r.t. one of BDI's symptoms. Due to this labeling limitation, we framed our development as a binary classification task for each BDI symptom, and evaluated accordingly. To that end, we split the available labeled data into training and validation sets, and explored foundation model fine-tuning, sentence similarity, Large Language Model (LLM) prompting, and ensemble techniques. The validation results revealed that fine-tuning foundation models yielded the best performance, particularly when enhanced with synthetic data to mitigate class imbalance. We also observed that the optimal approach varied by symptom. Based on these insights, we devised five independent test runs, two of which used ensemble methods. These runs achieved the highest scores in the official IR evaluation, outperforming submissions from 16 other teams.
>
---
#### [new 105] Beyond Text Compression: Evaluating Tokenizers Across Scales
- **分类: cs.CL**

- **简介: 论文任务是评估不同规模语言模型中的分词器（tokenizer）质量。它旨在解决如何高效、可靠地衡量分词器对模型性能影响的问题。作者通过研究小模型预测大模型分词器效果、分析多语言与单语场景差异，并提出基于Zipf定律的新评估指标，构建了一个有效的分词器内在评估框架。**

- **链接: [http://arxiv.org/pdf/2506.03101v1](http://arxiv.org/pdf/2506.03101v1)**

> **作者:** Jonas F. Lotz; António V. Lopes; Stephan Peitz; Hendra Setiawan; Leonardo Emili
>
> **备注:** ACL 2025
>
> **摘要:** The choice of tokenizer can profoundly impact language model performance, yet accessible and reliable evaluations of tokenizer quality remain an open challenge. Inspired by scaling consistency, we show that smaller models can accurately predict significant differences in tokenizer impact on larger models at a fraction of the compute cost. By systematically evaluating both English-centric and multilingual tokenizers, we find that tokenizer choice has negligible effects on tasks in English but results in consistent performance differences in multilingual settings. We propose new intrinsic tokenizer metrics inspired by Zipf's law that correlate more strongly with downstream performance than text compression when modeling unseen languages. By combining several metrics to capture multiple aspects of tokenizer behavior, we develop a reliable framework for intrinsic tokenizer evaluations. Our work offers a more efficient path to informed tokenizer selection in future language model development.
>
---
#### [new 106] EssayBench: Evaluating Large Language Models in Multi-Genre Chinese Essay Writing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于中文作文生成评估任务，旨在解决现有大模型在多体裁中文作文写作能力评估不足的问题。作者构建了包含728个真实提示的多体裁基准EssayBench，涵盖议论文、记叙文、描写文和说明文，并开发了细粒度评分框架，评估15个大模型表现，推动教育场景下的作文生成研究。**

- **链接: [http://arxiv.org/pdf/2506.02596v1](http://arxiv.org/pdf/2506.02596v1)**

> **作者:** Fan Gao; Dongyuan Li; Ding Xia; Fei Mi; Yasheng Wang; Lifeng Shang; Baojun Wang
>
> **摘要:** Chinese essay writing and its evaluation are critical in educational contexts, yet the capabilities of Large Language Models (LLMs) in this domain remain largely underexplored. Existing benchmarks often rely on coarse-grained text quality metrics, largely overlooking the structural and rhetorical complexities of Chinese essays, particularly across diverse genres. To address this gap, we propose \benchName, a multi-genre benchmark specifically designed for Chinese essay writing across four major genres: Argumentative, Narrative, Descriptive, and Expository. We curate and refine a total of 728 real-world prompts to ensure authenticity and meticulously categorize them into the \textit{Open-Ended} and \textit{Constrained} sets to capture diverse writing scenarios. To reliably evaluate generated essays, we develop a fine-grained, genre-specific scoring framework that hierarchically aggregates scores. We further validate our evaluation protocol through a comprehensive human agreement study. Finally, we benchmark 15 large-sized LLMs, analyzing their strengths and limitations across genres and instruction types. With \benchName, we aim to advance LLM-based Chinese essay evaluation and inspire future research on improving essay generation in educational settings.
>
---
#### [new 107] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL
- **分类: cs.CL**

- **简介: 该论文旨在解决开源推理模型依赖现有模型的局限性，提出构建长链思维（CoT）数据集的任务。通过利用未专为推理扩展训练的小型LLM，收集10万条CoT推理路径，并改进其推理能力与可控性。结果显示，该数据集质量接近R1，且在强化学习中表现更优。**

- **链接: [http://arxiv.org/pdf/2506.02338v1](http://arxiv.org/pdf/2506.02338v1)**

> **作者:** Hyungjoo Chae; Dongjin Kang; Jihyuk Kim; Beong-woo Kwak; Sunghyun Park; Haeju Park; Jinyoung Yeo; Moontae Lee; Kyungjae Lee
>
> **备注:** ACL 2025 Industry
>
> **摘要:** With the release of R1, a publicly available large reasoning model (LRM), researchers commonly train new LRMs by training language models on R1's long chain-of-thought (CoT) inferences. While prior works show that LRMs' capabilities can be reproduced through direct distillation, the continued reliance on the existing models (e.g., R1) remains a critical limitation in advancing the field. As a first step toward independent LRM development, this paper explores the possibility of constructing a long CoT dataset with LLMs that are not trained for inference-time scaling. To this end, we present the Long CoT Collection, a dataset of 100K CoT rationales annotated using existing short CoT LLMs. We develop a pipeline that induces o1's novel reasoning strategies into short CoT LLMs, enabling them to think longer and introducing controllability over the thought budget to better manage the overthinking problem. Our extensive analyses validate that our dataset achieves quality comparable to--or slightly below--R1. Furthermore, our experiments demonstrate that training on our dataset not only strengthens general reasoning skills, but also provides a strong foundation for reinforcement learning--models initialized on our data achieve 2-3x larger gains with RLVR.
>
---
#### [new 108] Leveraging Natural Language Processing to Unravel the Mystery of Life: A Review of NLP Approaches in Genomics, Transcriptomics, and Proteomics
- **分类: cs.CL; cs.AI; q-bio.GN**

- **简介: 该论文综述了自然语言处理（NLP）技术在生物信息学中的应用，旨在解决基因组学、转录组学和蛋白质组学中的数据分析问题。论文属于综述任务，总结了从经典方法如word2vec到先进Transformer模型的使用，并探讨其在结构预测、基因表达等任务中的潜力与局限性。**

- **链接: [http://arxiv.org/pdf/2506.02212v1](http://arxiv.org/pdf/2506.02212v1)**

> **作者:** Ella Rannon; David Burstein
>
> **摘要:** Natural Language Processing (NLP) has transformed various fields beyond linguistics by applying techniques originally developed for human language to the analysis of biological sequences. This review explores the application of NLP methods to biological sequence data, focusing on genomics, transcriptomics, and proteomics. We examine how various NLP methods, from classic approaches like word2vec to advanced models employing transformers and hyena operators, are being adapted to analyze DNA, RNA, protein sequences, and entire genomes. The review also examines tokenization strategies and model architectures, evaluating their strengths, limitations, and suitability for different biological tasks. We further cover recent advances in NLP applications for biological data, such as structure prediction, gene expression, and evolutionary analysis, highlighting the potential of these methods for extracting meaningful insights from large-scale genomic data. As language models continue to advance, their integration into bioinformatics holds immense promise for advancing our understanding of biological processes in all domains of life.
>
---
#### [new 109] Retrieval-Augmented Generation as Noisy In-Context Learning: A Unified Theory and Risk Bounds
- **分类: cs.LG; cs.AI; cs.CL; cs.IR; math.ST; stat.TH**

- **简介: 该论文属于理论分析任务，旨在探讨检索增强生成（RAG）的泛化性能。论文提出首个有限样本下的RAG在上下文线性回归中的泛化界，并推导偏差-方差权衡。将RAG视为带噪声的上下文学习（ICL），统一建模RAG与ICL，揭示RAG存在固有误差上限，并通过实验验证其理论分析。**

- **链接: [http://arxiv.org/pdf/2506.03100v1](http://arxiv.org/pdf/2506.03100v1)**

> **作者:** Yang Guo; Yutian Tao; Yifei Ming; Robert D. Nowak; Yingyu Liang
>
> **备注:** Under Review
>
> **摘要:** Retrieval-augmented generation (RAG) has seen many empirical successes in recent years by aiding the LLM with external knowledge. However, its theoretical aspect has remained mostly unexplored. In this paper, we propose the first finite-sample generalization bound for RAG in in-context linear regression and derive an exact bias-variance tradeoff. Our framework views the retrieved texts as query-dependent noisy in-context examples and recovers the classical in-context learning (ICL) and standard RAG as the limit cases. Our analysis suggests that an intrinsic ceiling on generalization error exists on RAG as opposed to the ICL. Furthermore, our framework is able to model retrieval both from the training data and from external corpora by introducing uniform and non-uniform RAG noise. In line with our theory, we show the sample efficiency of ICL and RAG empirically with experiments on common QA benchmarks, such as Natural Questions and TriviaQA.
>
---
#### [new 110] Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解析大推理模型（LRMs）的内部推理机制。通过信息论视角，追踪中间表征与正确答案之间互信息（MI）的变化，发现MI峰值现象，并指出这些峰值对应“思考标记”（如“Hmm”、“Therefore”）。研究表明这些标记对推理性能至关重要，并提出利用这些标记提升模型推理能力的方法。**

- **链接: [http://arxiv.org/pdf/2506.02867v1](http://arxiv.org/pdf/2506.02867v1)**

> **作者:** Chen Qian; Dongrui Liu; Haochen Wen; Zhen Bai; Yong Liu; Jing Shao
>
> **备注:** Preprint. Under review
>
> **摘要:** Large reasoning models (LRMs) have demonstrated impressive capabilities in complex problem-solving, yet their internal reasoning mechanisms remain poorly understood. In this paper, we investigate the reasoning trajectories of LRMs from an information-theoretic perspective. By tracking how mutual information (MI) between intermediate representations and the correct answer evolves during LRM reasoning, we observe an interesting MI peaks phenomenon: the MI at specific generative steps exhibits a sudden and significant increase during LRM's reasoning process. We theoretically analyze such phenomenon and show that as MI increases, the probability of model's prediction error decreases. Furthermore, these MI peaks often correspond to tokens expressing reflection or transition, such as ``Hmm'', ``Wait'' and ``Therefore,'' which we term as the thinking tokens. We then demonstrate that these thinking tokens are crucial for LRM's reasoning performance, while other tokens has minimal impacts. Building on these analyses, we propose two simple yet effective methods to improve LRM's reasoning performance, by delicately leveraging these thinking tokens. Overall, our work provides novel insights into the reasoning mechanisms of LRMs and offers practical ways to improve their reasoning capabilities. The code is available at https://github.com/ChnQ/MI-Peaks.
>
---
#### [new 111] Automated Web Application Testing: End-to-End Test Case Generation with Large Language Models and Screen Transition Graphs
- **分类: cs.SE; cs.AI; cs.CL; I.2.7**

- **简介: 该论文属于软件测试任务，旨在解决Web应用测试中导航流程复杂和表单交互难自动化的问题。作者结合大语言模型与屏幕状态图、状态图方法，自动生成端到端测试用例，提升了测试覆盖率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.02529v1](http://arxiv.org/pdf/2506.02529v1)**

> **作者:** Nguyen-Khang Le; Quan Minh Bui; Minh Ngoc Nguyen; Hiep Nguyen; Trung Vo; Son T. Luu; Shoshin Nomura; Minh Le Nguyen
>
> **备注:** Published in the Proceedings of JSAI 2025
>
> **摘要:** Web applications are critical to modern software ecosystems, yet ensuring their reliability remains challenging due to the complexity and dynamic nature of web interfaces. Recent advances in large language models (LLMs) have shown promise in automating complex tasks, but limitations persist in handling dynamic navigation flows and complex form interactions. This paper presents an automated system for generating test cases for two key aspects of web application testing: site navigation and form filling. For site navigation, the system employs screen transition graphs and LLMs to model navigation flows and generate test scenarios. For form filling, it uses state graphs to handle conditional forms and automates Selenium script generation. Key contributions include: (1) a novel integration of graph structures and LLMs for site navigation testing, (2) a state graph-based approach for automating form-filling test cases, and (3) a comprehensive dataset for evaluating form-interaction testing. Experimental results demonstrate the system's effectiveness in improving test coverage and robustness, advancing the state of web application testing.
>
---
#### [new 112] A Dynamic Framework for Semantic Grouping of Common Data Elements (CDE) Using Embeddings and Clustering
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于数据集成与语义分析任务，旨在解决生物医学数据中通用数据元素（CDE）的语义异构性问题。通过使用大型语言模型生成嵌入表示，并结合HDBSCAN聚类、自动标签生成和监督分类，实现对CDE的动态分组与高效整合。**

- **链接: [http://arxiv.org/pdf/2506.02160v1](http://arxiv.org/pdf/2506.02160v1)**

> **作者:** Madan Krishnamurthy; Daniel Korn; Melissa A Haendel; Christopher J Mungall; Anne E Thessen
>
> **摘要:** This research aims to develop a dynamic and scalable framework to facilitate harmonization of Common Data Elements (CDEs) across heterogeneous biomedical datasets by addressing challenges such as semantic heterogeneity, structural variability, and context dependence to streamline integration, enhance interoperability, and accelerate scientific discovery. Our methodology leverages Large Language Models (LLMs) for context-aware text embeddings that convert CDEs into dense vectors capturing semantic relationships and patterns. These embeddings are clustered using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) to group semantically similar CDEs. The framework incorporates four key steps: (1) LLM-based text embedding to mathematically represent semantic context, (2) unsupervised clustering of embeddings via HDBSCAN, (3) automated labeling using LLM summarization, and (4) supervised learning to train a classifier assigning new or unclustered CDEs to labeled clusters. Evaluated on the NIH NLM CDE Repository with over 24,000 CDEs, the system identified 118 meaningful clusters at an optimized minimum cluster size of 20. The classifier achieved 90.46 percent overall accuracy, performing best in larger categories. External validation against Gravity Projects Social Determinants of Health domains showed strong agreement (Adjusted Rand Index 0.52, Normalized Mutual Information 0.78), indicating that embeddings effectively capture cluster characteristics. This adaptable and scalable approach offers a practical solution to CDE harmonization, improving selection efficiency and supporting ongoing data interoperability.
>
---
#### [new 113] Synthetic Speech Source Tracing using Metric Learning
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频取证任务，旨在解决合成语音的源追踪问题。现有方法多关注欺骗检测，缺乏有效追踪生成系统的手段。受说话人识别启发，作者采用分类和度量学习方法，在MLAADv5数据集上测试ResNet与自监督模型效果，结果显示ResNet表现优异，表明其在源追踪中的可行性，并指出优化自监督表示的重要性。**

- **链接: [http://arxiv.org/pdf/2506.02590v1](http://arxiv.org/pdf/2506.02590v1)**

> **作者:** Dimitrios Koutsianos; Stavros Zacharopoulos; Yannis Panagakis; Themos Stafylakis
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This paper addresses source tracing in synthetic speech-identifying generative systems behind manipulated audio via speaker recognition-inspired pipelines. While prior work focuses on spoofing detection, source tracing lacks robust solutions. We evaluate two approaches: classification-based and metric-learning. We tested our methods on the MLAADv5 benchmark using ResNet and self-supervised learning (SSL) backbones. The results show that ResNet achieves competitive performance with the metric learning approach, matching and even exceeding SSL-based systems. Our work demonstrates ResNet's viability for source tracing while underscoring the need to optimize SSL representations for this task. Our work bridges speaker recognition methodologies with audio forensic challenges, offering new directions for combating synthetic media manipulation.
>
---
#### [new 114] Breaking Quadratic Barriers: A Non-Attention LLM for Ultra-Long Context Horizons
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统Transformer模型在长文本处理中的效率问题。论文提出了一种无需注意力机制的新架构，结合状态空间模块、多分辨率卷积层、轻量级循环监督器与检索增强外部内存，实现对超长上下文（数十万至百万级token）的高效建模，显著降低计算复杂度。**

- **链接: [http://arxiv.org/pdf/2506.01963v1](http://arxiv.org/pdf/2506.01963v1)**

> **作者:** Andrew Kiruluta; Preethi Raju; Priscilla Burity
>
> **摘要:** We present a novel non attention based architecture for large language models (LLMs) that efficiently handles very long context windows, on the order of hundreds of thousands to potentially millions of tokens. Unlike traditional Transformer designs, which suffer from quadratic memory and computation overload due to the nature of the self attention mechanism, our model avoids token to token attention entirely. Instead, it combines the following complementary components: State Space blocks (inspired by S4) that learn continuous time convolution kernels and scale near linearly with sequence length, Multi Resolution Convolution layers that capture local context at different dilation levels, a lightweight Recurrent Supervisor to maintain a global hidden state across sequential chunks, and Retrieval Augmented External Memory that stores and retrieves high-level chunk embeddings without reintroducing quadratic operations.
>
---
#### [new 115] Iterative Self-Improvement of Vision Language Models for Image Scoring and Self-Explanation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像评分与解释任务，旨在提升视觉语言模型在评分的同时生成可信的自然语言解释。通过自训练和偏好优化，结合两个数据集迭代训练，改善评分准确性和解释一致性。**

- **链接: [http://arxiv.org/pdf/2506.02708v1](http://arxiv.org/pdf/2506.02708v1)**

> **作者:** Naoto Tanji; Toshihiko Yamasaki
>
> **备注:** Accepted to ICIP2025
>
> **摘要:** Image scoring is a crucial task in numerous real-world applications. To trust a model's judgment, understanding its rationale is essential. This paper proposes a novel training method for Vision Language Models (VLMs) to generate not only image scores but also corresponding justifications in natural language. Leveraging only an image scoring dataset and an instruction-tuned VLM, our method enables self-training, utilizing the VLM's generated text without relying on external data or models. In addition, we introduce a simple method for creating a dataset designed to improve alignment between predicted scores and their textual justifications. By iteratively training the model with Direct Preference Optimization on two distinct datasets and merging them, we can improve both scoring accuracy and the coherence of generated explanations.
>
---
#### [new 116] Scaling Fine-Grained MoE Beyond 50B Parameters: Empirical Evaluation and Practical Insights
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型扩展任务，旨在解决如何高效扩展MoE模型的问题。论文提出了细粒度MoE的训练方法，并进行了实证评估，比较其与标准MoE在高达56B参数模型中的表现，验证了其在收敛速度、模型性能和训练实用性方面的优势。**

- **链接: [http://arxiv.org/pdf/2506.02890v1](http://arxiv.org/pdf/2506.02890v1)**

> **作者:** Jakub Krajewski; Marcin Chochowski; Daniel Korzekwa
>
> **摘要:** Mixture of Experts (MoE) architectures have emerged as pivotal for scaling Large Language Models (LLMs) efficiently. Fine-grained MoE approaches - utilizing more numerous, smaller experts - have demonstrated potential in improving model convergence and quality. This work proposes a set of training recipes and provides a comprehensive empirical evaluation of fine-grained MoE, directly comparing its scaling properties against standard MoE configurations for models with up to 56B total (17B active) parameters. We investigate convergence speed, model performance on downstream benchmarks, and practical training considerations across various setups. Overall, at the largest scale we show that fine-grained MoE achieves better validation loss and higher accuracy across a set of downstream benchmarks. This study offers empirical grounding and practical insights for leveraging fine-grained MoE in the development of future large-scale models.
>
---
#### [new 117] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多语言语义检索任务，旨在解决现有数据集和模型在处理多条件、多语言及多图像检索时的局限性。作者构建了首个支持多条件查询的多语言语义检索数据集MERIT，并提出Coral框架，结合嵌入重建与对比学习，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2506.03144v1](http://arxiv.org/pdf/2506.03144v1)**

> **作者:** Wei Chow; Yuan Gao; Linfeng Li; Xian Wang; Qi Xu; Hang Song; Lingdong Kong; Ran Zhou; Yi Zeng; Yidong Cai; Botian Jiang; Shilin Xu; Jiajun Zhang; Minghui Qiu; Xiangtai Li; Tianshu Yang; Siliang Tang; Juncheng Li
>
> **备注:** Preprint; Project Page, Code, and Dataset at: https://merit-2025.github.io/
>
> **摘要:** Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval.
>
---
#### [new 118] ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在评估大语言模型（LLMs）将最新机器学习研究思想转化为可执行代码的能力。作者构建了ResearchCodeBench基准测试集，包含212个编码挑战，评估30多个LLMs的表现，发现最佳模型成功率不足40%。论文分析了性能、污染和错误模式，推动LLM在科研代码生成方向的发展。**

- **链接: [http://arxiv.org/pdf/2506.02314v1](http://arxiv.org/pdf/2506.02314v1)**

> **作者:** Tianyu Hua; Harper Hua; Violet Xiang; Benjamin Klieger; Sang T. Truong; Weixin Liang; Fan-Yun Sun; Nick Haber
>
> **摘要:** Large language models (LLMs) have shown promise in transforming machine learning research, yet their capability to faithfully implement novel ideas from recent research papers-ideas unseen during pretraining-remains unclear. We introduce ResearchCodeBench, a benchmark of 212 coding challenges that evaluates LLMs' ability to translate cutting-edge ML contributions from top 2024-2025 research papers into executable code. We assessed 30+ proprietary and open-source LLMs, finding that even the best models correctly implement less than 40% of the code. We find Gemini-2.5-Pro-Preview to perform best at 37.3% success rate, with O3 (High) and O4-mini (High) following behind at 32.3% and 30.8% respectively. We present empirical findings on performance comparison, contamination, and error patterns. By providing a rigorous and community-driven evaluation platform, ResearchCodeBench enables continuous understanding and advancement of LLM-driven innovation in research code generation.
>
---
#### [new 119] Mitigating Manipulation and Enhancing Persuasion: A Reflective Multi-Agent Approach for Legal Argument Generation
- **分类: cs.AI; cs.CL; cs.LG; 68T50; I.2**

- **简介: 该论文属于法律论证生成任务，旨在解决大语言模型在法律场景中易产生幻觉、不当说服和无法有效利用事实的问题。论文提出了一种反思多智能体方法，通过迭代优化生成三方法律论点，并验证其在减少错误论证和提升事实利用率方面的有效性。**

- **链接: [http://arxiv.org/pdf/2506.02992v1](http://arxiv.org/pdf/2506.02992v1)**

> **作者:** Li Zhang; Kevin D. Ashley
>
> **备注:** 13 pages, 2 figures, Workshop on Legally Compliant Intelligent Chatbots at ICAIL 2025]{Workshop on Legally Compliant Intelligent Chatbots @ ICAIL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly explored for legal argument generation, yet they pose significant risks of manipulation through hallucination and ungrounded persuasion, and often fail to utilize provided factual bases effectively or abstain when arguments are untenable. This paper introduces a novel reflective multi-agent method designed to address these challenges in the context of legally compliant persuasion. Our approach employs specialized agents--a Factor Analyst and an Argument Polisher--in an iterative refinement process to generate 3-ply legal arguments (plaintiff, defendant, rebuttal). We evaluate Reflective Multi-Agent against single-agent, enhanced-prompt single-agent, and non-reflective multi-agent baselines using four diverse LLMs (GPT-4o, GPT-4o-mini, Llama-4-Maverick-17b-128e, Llama-4-Scout-17b-16e) across three legal scenarios: "arguable", "mismatched", and "non-arguable". Results demonstrate Reflective Multi-Agent's significant superiority in successful abstention (preventing generation when arguments cannot be grounded), marked improvements in hallucination accuracy (reducing fabricated and misattributed factors), particularly in "non-arguable" scenarios, and enhanced factor utilization recall (improving the use of provided case facts). These findings suggest that structured reflection within a multi-agent framework offers a robust computable method for fostering ethical persuasion and mitigating manipulation in LLM-based legal argumentation systems, a critical step towards trustworthy AI in law. Project page: https://lizhang-aiandlaw.github.io/A-Reflective-Multi-Agent-Approach-for-Legal-Argument-Generation/
>
---
#### [new 120] Enhancing Speech Instruction Understanding and Disambiguation in Robotics via Speech Prosody
- **分类: cs.RO; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于人机交互任务，旨在解决机器人理解模糊语音指令的问题。通过利用语音韵律特征来推断和解析指令意图，并结合大语言模型进行任务规划。论文还发布了首个用于语音消歧的机器人数据集，提升了语音指令理解和任务选择的准确性。**

- **链接: [http://arxiv.org/pdf/2506.02057v1](http://arxiv.org/pdf/2506.02057v1)**

> **作者:** David Sasu; Kweku Andoh Yamoah; Benedict Quartey; Natalie Schluter
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Enabling robots to accurately interpret and execute spoken language instructions is essential for effective human-robot collaboration. Traditional methods rely on speech recognition to transcribe speech into text, often discarding crucial prosodic cues needed for disambiguating intent. We propose a novel approach that directly leverages speech prosody to infer and resolve instruction intent. Predicted intents are integrated into large language models via in-context learning to disambiguate and select appropriate task plans. Additionally, we present the first ambiguous speech dataset for robotics, designed to advance research in speech disambiguation. Our method achieves 95.79% accuracy in detecting referent intents within an utterance and determines the intended task plan of ambiguous instructions with 71.96% accuracy, demonstrating its potential to significantly improve human-robot communication.
>
---
#### [new 121] Learning More with Less: Self-Supervised Approaches for Low-Resource Speech Emotion Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决低资源语言因标注数据不足导致的识别难题。作者采用对比学习和BYOL两种自监督方法，提升跨语言情感识别性能，并在多种语言上取得显著效果，为构建包容性强的情感识别系统提供了新思路。**

- **链接: [http://arxiv.org/pdf/2506.02059v1](http://arxiv.org/pdf/2506.02059v1)**

> **作者:** Ziwei Gong; Pengyuan Shi; Kaan Donbekci; Lin Ai; Run Chen; David Sasu; Zehui Wu; Julia Hirschberg
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Speech Emotion Recognition (SER) has seen significant progress with deep learning, yet remains challenging for Low-Resource Languages (LRLs) due to the scarcity of annotated data. In this work, we explore unsupervised learning to improve SER in low-resource settings. Specifically, we investigate contrastive learning (CL) and Bootstrap Your Own Latent (BYOL) as self-supervised approaches to enhance cross-lingual generalization. Our methods achieve notable F1 score improvements of 10.6% in Urdu, 15.2% in German, and 13.9% in Bangla, demonstrating their effectiveness in LRLs. Additionally, we analyze model behavior to provide insights on key factors influencing performance across languages, and also highlighting challenges in low-resource SER. This work provides a foundation for developing more inclusive, explainable, and robust emotion recognition systems for underrepresented languages.
>
---
#### [new 122] Enhancing Speech Emotion Recognition with Graph-Based Multimodal Fusion and Prosodic Features for the Speech Emotion Recognition in Naturalistic Conditions Challenge at Interspeech 2025
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文属于语音情感识别（SER）任务，旨在解决自然场景下情感识别准确率低的问题。作者提出了一种结合图注意力网络的多模态融合方法，并利用基频量化与预训练音频标签模型提升性能，最终在Interspeech 2025挑战赛中取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2506.02088v1](http://arxiv.org/pdf/2506.02088v1)**

> **作者:** Alef Iury Siqueira Ferreira; Lucas Rafael Gris; Alexandre Ferro Filho; Lucas Ólives; Daniel Ribeiro; Luiz Fernando; Fernanda Lustosa; Rodrigo Tanaka; Frederico Santos de Oliveira; Arlindo Galvão Filho
>
> **摘要:** Training SER models in natural, spontaneous speech is especially challenging due to the subtle expression of emotions and the unpredictable nature of real-world audio. In this paper, we present a robust system for the INTERSPEECH 2025 Speech Emotion Recognition in Naturalistic Conditions Challenge, focusing on categorical emotion recognition. Our method combines state-of-the-art audio models with text features enriched by prosodic and spectral cues. In particular, we investigate the effectiveness of Fundamental Frequency (F0) quantization and the use of a pretrained audio tagging model. We also employ an ensemble model to improve robustness. On the official test set, our system achieved a Macro F1-score of 39.79% (42.20% on validation). Our results underscore the potential of these methods, and analysis of fusion techniques confirmed the effectiveness of Graph Attention Networks. Our source code is publicly available.
>
---
#### [new 123] Unveiling Audio Deepfake Origins: A Deep Metric learning And Conformer Network Approach With Ensemble Fusion
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频源追踪任务，旨在解决音频深度伪造的溯源问题。通过结合深度度量学习、Conformer网络与集成融合方法，提升对真实与伪造语音的辨别能力及源系统追踪准确性。**

- **链接: [http://arxiv.org/pdf/2506.02085v1](http://arxiv.org/pdf/2506.02085v1)**

> **作者:** Ajinkya Kulkarni; Sandipana Dowerah; Tanel Alumae; Mathew Magimai. -Doss
>
> **备注:** Accepted at Interspeech 2025, Netherlands
>
> **摘要:** Audio deepfakes are acquiring an unprecedented level of realism with advanced AI. While current research focuses on discerning real speech from spoofed speech, tracing the source system is equally crucial. This work proposes a novel audio source tracing system combining deep metric multi-class N-pair loss with Real Emphasis and Fake Dispersion framework, a Conformer classification network, and ensemble score-embedding fusion. The N-pair loss improves discriminative ability, while Real Emphasis and Fake Dispersion enhance robustness by focusing on differentiating real and fake speech patterns. The Conformer network captures both global and local dependencies in the audio signal, crucial for source tracing. The proposed ensemble score-embedding fusion shows an optimal trade-off between in-domain and out-of-domain source tracing scenarios. We evaluate our method using Frechet Distance and standard metrics, demonstrating superior performance in source tracing over the baseline system.
>
---
#### [new 124] Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型中量化与低秩分解协同优化不足的问题。通过提出ODLRI方法，使低秩部分专门捕捉激活敏感权重，从而减少量化误差并提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.02077v1](http://arxiv.org/pdf/2506.02077v1)**

> **作者:** Yoonjun Cho; Soeun Kim; Dongjae Jeon; Kyelim Lee; Beomsoo Lee; Albert No
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Decomposing weight matrices into quantization and low-rank components ($\mathbf{W} \approx \mathbf{Q} + \mathbf{L}\mathbf{R}$) is a widely used technique for compressing large language models (LLMs). Existing joint optimization methods iteratively alternate between quantization and low-rank approximation. However, these methods tend to prioritize one component at the expense of the other, resulting in suboptimal decompositions that fail to leverage each component's unique strengths. In this work, we introduce Outlier-Driven Low-Rank Initialization (ODLRI), which assigns low-rank components the specific role of capturing activation-sensitive weights. This structured decomposition mitigates outliers' negative impact on quantization, enabling more effective balance between quantization and low-rank approximation. Experiments on Llama2 (7B, 13B, 70B), Llama3-8B, and Mistral-7B demonstrate that incorporating ODLRI into the joint optimization framework consistently reduces activation-aware error, minimizes quantization scale, and improves perplexity and zero-shot accuracy in low-bit settings.
>
---
#### [new 125] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在解决当前模型在复杂空间理解上的不足。作者构建了一个全面的基准OmniSpatial，包含四大类和50个子类，涵盖动态推理、复杂空间逻辑等，并收集了1.5K问题对模型进行评估，揭示了现有模型的局限性，并提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.03135v1](http://arxiv.org/pdf/2506.03135v1)**

> **作者:** Mengdi Jia; Zekun Qi; Shaochen Zhang; Wenyao Zhang; Xinqiang Yu; Jiawei He; He Wang; Li Yi
>
> **备注:** Project Page: https://qizekun.github.io/omnispatial/
>
> **摘要:** Spatial reasoning is a key aspect of cognitive psychology and remains a major bottleneck for current vision-language models (VLMs). While extensive research has aimed to evaluate or improve VLMs' understanding of basic spatial relations, such as distinguishing left from right, near from far, and object counting, these tasks represent only the most fundamental level of spatial reasoning. In this work, we introduce OmniSpatial, a comprehensive and challenging benchmark for spatial reasoning, grounded in cognitive psychology. OmniSpatial covers four major categories: dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking, with 50 fine-grained subcategories. Through Internet data crawling and careful manual annotation, we construct over 1.5K question-answer pairs. Extensive experiments show that both open- and closed-source VLMs, as well as existing reasoning and spatial understanding models, exhibit significant limitations in comprehensive spatial understanding. We further analyze failure cases and propose potential directions for future research.
>
---
#### [new 126] KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出KDRL，一种结合知识蒸馏（KD）与强化学习（RL）的统一大语言模型后训练框架，旨在提升模型推理能力。通过策略梯度优化，同时最小化学生与教师模型分布的RKL并最大化规则奖励，实现高效推理训练。**

- **链接: [http://arxiv.org/pdf/2506.02208v1](http://arxiv.org/pdf/2506.02208v1)**

> **作者:** Hongling Xu; Qi Zhu; Heyuan Deng; Jinpeng Li; Lu Hou; Yasheng Wang; Lifeng Shang; Ruifeng Xu; Fei Mi
>
> **摘要:** Recent advances in large language model (LLM) post-training have leveraged two distinct paradigms to enhance reasoning capabilities: reinforcement learning (RL) and knowledge distillation (KD). While RL enables the emergence of complex reasoning behaviors, it often suffers from low sample efficiency when the initial policy struggles to explore high-reward trajectories. Conversely, KD improves learning efficiency via mimicking the teacher model but tends to generalize poorly to out-of-domain scenarios. In this work, we present \textbf{KDRL}, a \textit{unified post-training framework} that jointly optimizes a reasoning model through teacher supervision (KD) and self-exploration (RL). Specifically, KDRL leverages policy gradient optimization to simultaneously minimize the reverse Kullback-Leibler divergence (RKL) between the student and teacher distributions while maximizing the expected rule-based rewards. We first formulate a unified objective that integrates GRPO and KD, and systematically explore how different KL approximations, KL coefficients, and reward-guided KD strategies affect the overall post-training dynamics and performance. Empirical results on multiple reasoning benchmarks demonstrate that KDRL outperforms GRPO and various KD baselines while achieving a favorable balance between performance and reasoning token efficiency. These findings indicate that integrating KD and RL serves as an effective and efficient strategy to train reasoning LLMs.
>
---
#### [new 127] BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出了一种名为BitBypass的新型黑盒越狱攻击方法，旨在绕过对齐大型语言模型的安全机制。它通过利用连字符分隔的比特流伪装，从数据的基本信息表示层面出发，而非依赖提示工程或对抗性操纵。实验表明，该方法在多个先进模型上具有较高的隐蔽性和攻击成功率，揭示了当前安全对齐技术的潜在漏洞。**

- **链接: [http://arxiv.org/pdf/2506.02479v1](http://arxiv.org/pdf/2506.02479v1)**

> **作者:** Kalyan Nakka; Nitesh Saxena
>
> **备注:** 24 pages, 24 figures, and 7 tables
>
> **摘要:** The inherent risk of generating harmful and unsafe content by Large Language Models (LLMs), has highlighted the need for their safety alignment. Various techniques like supervised fine-tuning, reinforcement learning from human feedback, and red-teaming were developed for ensuring the safety alignment of LLMs. However, the robustness of these aligned LLMs is always challenged by adversarial attacks that exploit unexplored and underlying vulnerabilities of the safety alignment. In this paper, we develop a novel black-box jailbreak attack, called BitBypass, that leverages hyphen-separated bitstream camouflage for jailbreaking aligned LLMs. This represents a new direction in jailbreaking by exploiting fundamental information representation of data as continuous bits, rather than leveraging prompt engineering or adversarial manipulations. Our evaluation of five state-of-the-art LLMs, namely GPT-4o, Gemini 1.5, Claude 3.5, Llama 3.1, and Mixtral, in adversarial perspective, revealed the capabilities of BitBypass in bypassing their safety alignment and tricking them into generating harmful and unsafe content. Further, we observed that BitBypass outperforms several state-of-the-art jailbreak attacks in terms of stealthiness and attack success. Overall, these results highlights the effectiveness and efficiency of BitBypass in jailbreaking these state-of-the-art LLMs.
>
---
#### [new 128] Comba: Improving Nonlinear RNNs with Closed-loop Control
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于序列建模任务，旨在提升非线性RNN的性能与效率。作者分析了现有模型的优劣，提出基于闭环控制理论的Comba模型，采用标量加低秩状态转移和反馈机制，并实现了高效并行计算。实验表明其在语言和视觉任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.02475v1](http://arxiv.org/pdf/2506.02475v1)**

> **作者:** Jiaxi Hu; Yongqi Pan; Jusen Du; Disen Lan; Xiaqiang Tang; Qingsong Wen; Yuxuan Liang; Weigao Sun
>
> **摘要:** Recent efficient sequence modeling methods such as Gated DeltaNet, TTT, and RWKV-7 have achieved performance improvements by supervising the recurrent memory management through Delta learning rule. Unlike previous state-space models (e.g., Mamba) and gated linear attentions (e.g., GLA), these models introduce interactions between the recurrent state and the key vector, resulting in a nonlinear recursive structure. In this paper, we first introduce the concept of Nonlinear RNNs with a comprehensive analysis on the advantages and limitations of these models. Then, based on closed-loop control theory, we propose a novel Nonlinear RNN variant named Comba, which adopts a scalar-plus-low-rank state transition, with both state feedback and output feedback corrections. We also implement a hardware-efficient chunk-wise parallel kernel in Triton and train models with 340M/1.3B parameters on large-scale corpus. Comba demonstrates its superior performance and computation efficiency in both language and vision modeling.
>
---
#### [new 129] StarVC: A Unified Auto-Regressive Framework for Joint Text and Speech Generation in Voice Conversion
- **分类: cs.MM; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音转换任务，旨在解决传统方法忽略语言内容建模的问题。作者提出StarVC框架，先预测文本再合成语音，更好分离说话人身份与语言内容，提升转换效果。**

- **链接: [http://arxiv.org/pdf/2506.02414v1](http://arxiv.org/pdf/2506.02414v1)**

> **作者:** Fengjin Li; Jie Wang; Yadong Niu; Yongqing Wang; Meng Meng; Jian Luan; Zhiyong Wu
>
> **备注:** 5 pages, 2 figures, Accepted by Interspeech 2025, Demo: https://thuhcsi.github.io/StarVC/
>
> **摘要:** Voice Conversion (VC) modifies speech to match a target speaker while preserving linguistic content. Traditional methods usually extract speaker information directly from speech while neglecting the explicit utilization of linguistic content. Since VC fundamentally involves disentangling speaker identity from linguistic content, leveraging structured semantic features could enhance conversion performance. However, previous attempts to incorporate semantic features into VC have shown limited effectiveness, motivating the integration of explicit text modeling. We propose StarVC, a unified autoregressive VC framework that first predicts text tokens before synthesizing acoustic features. The experiments demonstrate that StarVC outperforms conventional VC methods in preserving both linguistic content (i.e., WER and CER) and speaker characteristics (i.e., SECS and MOS). Audio demo can be found at: https://thuhcsi.github.io/StarVC/.
>
---
#### [new 130] Inter(sectional) Alia(s): Ambiguity in Voice Agent Identity via Intersectional Japanese Self-Referents
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SD; eess.AS**

- **简介: 该论文研究了日本语境中语音代理的身份模糊性问题，特别是通过交叉性视角分析日语自我指称词对身份感知的影响。任务是探索语音代理在性别、年龄和正式程度等社会身份维度上的认知效果。工作包括招募204名日本参与者，评估三种ChatGPT语音与七种自我指称词的组合表现。结果揭示了语音性别化现象及某些自指词的去性别化潜力，强调文化敏感性和交叉性分析的重要性。**

- **链接: [http://arxiv.org/pdf/2506.01998v1](http://arxiv.org/pdf/2506.01998v1)**

> **作者:** Takao Fujii; Katie Seaborn; Madeleine Steeds; Jun Kato
>
> **备注:** CHI '25
>
> **摘要:** Conversational agents that mimic people have raised questions about the ethics of anthropomorphizing machines with human social identity cues. Critics have also questioned assumptions of identity neutrality in humanlike agents. Recent work has revealed that intersectional Japanese pronouns can elicit complex and sometimes evasive impressions of agent identity. Yet, the role of other "neutral" non-pronominal self-referents (NPSR) and voice as a socially expressive medium remains unexplored. In a crowdsourcing study, Japanese participants (N = 204) evaluated three ChatGPT voices (Juniper, Breeze, and Ember) using seven self-referents. We found strong evidence of voice gendering alongside the potential of intersectional self-referents to evade gendering, i.e., ambiguity through neutrality and elusiveness. Notably, perceptions of age and formality intersected with gendering as per sociolinguistic theories, especially boku and watakushi. This work provides a nuanced take on agent identity perceptions and champions intersectional and culturally-sensitive work on voice agents.
>
---
#### [new 131] Cocktail-Party Audio-Visual Speech Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频-视觉语音识别（AVSR）任务，旨在解决鸡尾酒会场景中复杂噪音下的语音识别问题。作者构建了一个包含说话和静默面部片段的新数据集，并提出新方法，在极端噪声环境下将词错误率显著降低67%，无需显式分割线索。**

- **链接: [http://arxiv.org/pdf/2506.02178v1](http://arxiv.org/pdf/2506.02178v1)**

> **作者:** Thai-Binh Nguyen; Ngoc-Quan Pham; Alexander Waibel
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) offers a robust solution for speech recognition in challenging environments, such as cocktail-party scenarios, where relying solely on audio proves insufficient. However, current AVSR models are often optimized for idealized scenarios with consistently active speakers, overlooking the complexities of real-world settings that include both speaking and silent facial segments. This study addresses this gap by introducing a novel audio-visual cocktail-party dataset designed to benchmark current AVSR systems and highlight the limitations of prior approaches in realistic noisy conditions. Additionally, we contribute a 1526-hour AVSR dataset comprising both talking-face and silent-face segments, enabling significant performance gains in cocktail-party environments. Our approach reduces WER by 67% relative to the state-of-the-art, reducing WER from 119% to 39.2% in extreme noise, without relying on explicit segmentation cues.
>
---
#### [new 132] Benchmarking and Advancing Large Language Models for Local Life Services
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理与本地生活服务结合的任务，旨在解决大语言模型在本地生活服务中的应用效果问题。工作包括构建综合基准测试，评估不同规模LLM的表现，并探索通过模型微调和基于代理的工作流提升性能。结果显示小模型可达到接近大模型的效果，优化了部署可行性。**

- **链接: [http://arxiv.org/pdf/2506.02720v1](http://arxiv.org/pdf/2506.02720v1)**

> **作者:** Xiaochong Lan; Jie Feng; Jiahuan Lei; Xinlei Shi; Yong Li
>
> **备注:** KDD 2025
>
> **摘要:** Large language models (LLMs) have exhibited remarkable capabilities and achieved significant breakthroughs across various domains, leading to their widespread adoption in recent years. Building on this progress, we investigate their potential in the realm of local life services. In this study, we establish a comprehensive benchmark and systematically evaluate the performance of diverse LLMs across a wide range of tasks relevant to local life services. To further enhance their effectiveness, we explore two key approaches: model fine-tuning and agent-based workflows. Our findings reveal that even a relatively compact 7B model can attain performance levels comparable to a much larger 72B model, effectively balancing inference cost and model capability. This optimization greatly enhances the feasibility and efficiency of deploying LLMs in real-world online services, making them more practical and accessible for local life applications.
>
---
#### [new 133] Generate, Not Recommend: Personalized Multimodal Content Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于个性化多模态内容生成任务，旨在解决推荐系统无法生成新颖内容的问题。作者利用大模态模型，通过监督微调和在线强化学习，实现根据用户兴趣生成个性化图像等内容，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.01704v1](http://arxiv.org/pdf/2506.01704v1)**

> **作者:** Jiongnan Liu; Zhicheng Dou; Ning Hu; Chenyan Xiong
>
> **摘要:** To address the challenge of information overload from massive web contents, recommender systems are widely applied to retrieve and present personalized results for users. However, recommendation tasks are inherently constrained to filtering existing items and lack the ability to generate novel concepts, limiting their capacity to fully satisfy user demands and preferences. In this paper, we propose a new paradigm that goes beyond content filtering and selecting: directly generating personalized items in a multimodal form, such as images, tailored to individual users. To accomplish this, we leverage any-to-any Large Multimodal Models (LMMs) and train them in both supervised fine-tuning and online reinforcement learning strategy to equip them with the ability to yield tailored next items for users. Experiments on two benchmark datasets and user study confirm the efficacy of the proposed method. Notably, the generated images not only align well with users' historical preferences but also exhibit relevance to their potential future interests.
>
---
#### [new 134] SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决现有RL训练数据不足和难度不够问题。作者提出了SynthRL方法，通过自动合成可验证的高质量训练数据，提升模型在复杂视觉数学推理任务上的表现。实验表明，该方法显著提高了模型在多个跨领域基准上的性能。**

- **链接: [http://arxiv.org/pdf/2506.02096v1](http://arxiv.org/pdf/2506.02096v1)**

> **作者:** Zijian Wu; Jinjie Ni; Xiangyan Liu; Zichen Liu; Hang Yan; Michael Qizhe Shieh
>
> **摘要:** Vision-language models (VLMs) trained via reinforcement learning with verifiable reward (RLVR) have shown notable progress in scaling test-time compute effectively. In this work, we investigate how synthesized RL data can further improve RLVR. To this end, we propose \textbf{SynthRL}-a scalable and guaranteed pipeline for automatic data scaling in reasoning-oriented RL training. SynthRL comprises three key stages: (1) selecting seed questions with appropriate distribution, (2) augmenting them into more challenging variants while preserving the original answers, and (3) a guaranteed verification stage that ensures near-perfect correctness and difficulty enhancement. Our empirical experiments demonstrate SynthRL's scalability and effectiveness. When applied to the MMK12 dataset, SynthRL synthesizes over 3.3K additional verifiable, challenging questions from approximately 8K seed samples. Models trained with our synthesized data achieve consistent gains across five out-of-domain visual math reasoning benchmarks, with a significant improvement over baseline models trained on seed data alone. Notably, detailed analysis reveals that the gains are more pronounced on the most challenging evaluation samples, highlighting SynthRL's effectiveness in eliciting deeper and more complex reasoning patterns.
>
---
#### [new 135] MAEBE: Multi-Agent Emergent Behavior Framework
- **分类: cs.MA; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出MAEBE框架，评估多智能体AI系统中的突发行为风险。任务是分析LLM在单智能体与多智能体环境中的道德偏好与行为变化。通过实验发现，问题表述方式、群体动力学和监督机制显著影响AI系统的安全性和对齐表现，强调需在交互环境中评估AI系统。**

- **链接: [http://arxiv.org/pdf/2506.03053v1](http://arxiv.org/pdf/2506.03053v1)**

> **作者:** Sinem Erisken; Timothy Gothard; Martin Leitgab; Ram Potham
>
> **备注:** Preprint. This work has been submitted to the Multi-Agent Systems Workshop at ICML 2025 for review
>
> **摘要:** Traditional AI safety evaluations on isolated LLMs are insufficient as multi-agent AI ensembles become prevalent, introducing novel emergent risks. This paper introduces the Multi-Agent Emergent Behavior Evaluation (MAEBE) framework to systematically assess such risks. Using MAEBE with the Greatest Good Benchmark (and a novel double-inversion question technique), we demonstrate that: (1) LLM moral preferences, particularly for Instrumental Harm, are surprisingly brittle and shift significantly with question framing, both in single agents and ensembles. (2) The moral reasoning of LLM ensembles is not directly predictable from isolated agent behavior due to emergent group dynamics. (3) Specifically, ensembles exhibit phenomena like peer pressure influencing convergence, even when guided by a supervisor, highlighting distinct safety and alignment challenges. Our findings underscore the necessity of evaluating AI systems in their interactive, multi-agent contexts.
>
---
#### [new 136] Rethinking Machine Unlearning in Image Generation Models
- **分类: cs.AI; cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于图像生成模型中的机器遗忘任务，旨在解决数据隐私与内容安全问题。现有方法存在任务不明确、评估框架和指标不可靠等挑战。作者提出了CatIGMU任务分类框架、EvalIGMU评估框架及DataIGMU高质量数据集，以提升算法设计与评估的可靠性。**

- **链接: [http://arxiv.org/pdf/2506.02761v1](http://arxiv.org/pdf/2506.02761v1)**

> **作者:** Renyang Liu; Wenjie Feng; Tianwei Zhang; Wei Zhou; Xueqi Cheng; See-Kiong Ng
>
> **备注:** Accepted by ACM CCS 2025
>
> **摘要:** With the surge and widespread application of image generation models, data privacy and content safety have become major concerns and attracted great attention from users, service providers, and policymakers. Machine unlearning (MU) is recognized as a cost-effective and promising means to address these challenges. Despite some advancements, image generation model unlearning (IGMU) still faces remarkable gaps in practice, e.g., unclear task discrimination and unlearning guidelines, lack of an effective evaluation framework, and unreliable evaluation metrics. These can hinder the understanding of unlearning mechanisms and the design of practical unlearning algorithms. We perform exhaustive assessments over existing state-of-the-art unlearning algorithms and evaluation standards, and discover several critical flaws and challenges in IGMU tasks. Driven by these limitations, we make several core contributions, to facilitate the comprehensive understanding, standardized categorization, and reliable evaluation of IGMU. Specifically, (1) We design CatIGMU, a novel hierarchical task categorization framework. It provides detailed implementation guidance for IGMU, assisting in the design of unlearning algorithms and the construction of testbeds. (2) We introduce EvalIGMU, a comprehensive evaluation framework. It includes reliable quantitative metrics across five critical aspects. (3) We construct DataIGM, a high-quality unlearning dataset, which can be used for extensive evaluations of IGMU, training content detectors for judgment, and benchmarking the state-of-the-art unlearning algorithms. With EvalIGMU and DataIGM, we discover that most existing IGMU algorithms cannot handle the unlearning well across different evaluation dimensions, especially for preservation and robustness. Code and models are available at https://github.com/ryliu68/IGMU.
>
---
#### [new 137] UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言统一建模任务，旨在解决现有模型在图像感知与编辑能力上的不足。受GPT-4o-Image启发，作者提出UniWorld框架，基于语义编码器实现图像理解、生成与编辑。仅用1%的数据量便超越BAGEL，在多任务上表现优异，并开源全部资源。**

- **链接: [http://arxiv.org/pdf/2506.03147v1](http://arxiv.org/pdf/2506.03147v1)**

> **作者:** Bin Lin; Zongjian Li; Xinhua Cheng; Yuwei Niu; Yang Ye; Xianyi He; Shenghai Yuan; Wangbo Yu; Shaodong Wang; Yunyang Ge; Yatian Pang; Li Yuan
>
> **摘要:** Although existing unified models deliver strong performance on vision-language understanding and text-to-image generation, their models are limited in exploring image perception and manipulation tasks, which are urgently desired by users for wide applications. Recently, OpenAI released their powerful GPT-4o-Image model for comprehensive image perception and manipulation, achieving expressive capability and attracting community interests. By observing the performance of GPT-4o-Image in our carefully constructed experiments, we infer that GPT-4o-Image leverages features extracted by semantic encoders instead of VAE, while VAEs are considered essential components in many image manipulation models. Motivated by such inspiring observations, we present a unified generative framework named UniWorld based on semantic features provided by powerful visual-language models and contrastive semantic encoders. As a result, we build a strong unified model using only 1% amount of BAGEL's data, which consistently outperforms BAGEL on image editing benchmarks. UniWorld also maintains competitive image understanding and generation capabilities, achieving strong performance across multiple image perception tasks. We fully open-source our models, including model weights, training and evaluation scripts, and datasets.
>
---
#### [new 138] Response-Level Rewards Are All You Need for Online Reinforcement Learning in LLMs: A Mathematical Perspective
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型中在线强化学习的“零奖励假设”问题，提出轨迹策略梯度定理，表明仅使用响应级奖励即可无偏估计基于真实令牌级奖励的策略梯度。工作包括理论分析与新算法TRePO，旨在简化LLM微调过程。**

- **链接: [http://arxiv.org/pdf/2506.02553v1](http://arxiv.org/pdf/2506.02553v1)**

> **作者:** Shenghua He; Tian Xia; Xuan Zhou; Hui Wei
>
> **摘要:** We study a common challenge in reinforcement learning for large language models (LLMs): the Zero-Reward Assumption, where non-terminal actions (i.e., intermediate token generations) receive zero task-specific immediate reward, while only the final token receives a reward for the entire response. This assumption arises frequently in practice, as precise token-level rewards are often difficult or infeasible to obtain in LLM applications. In this work, we provide a unifying theoretical perspective. We introduce the Trajectory Policy Gradient Theorem, which shows that the policy gradient based on true, unknown token-level rewards can be unbiasedly estimated using only a response-level reward model, regardless of whether the Zero-Reward Assumption holds or not, for algorithms in the REINFORCE and Actor-Critic families. This result reveals that widely used methods such as PPO, GRPO, ReMax, and RLOO inherently possess the capacity to model token-level reward signals, offering a theoretical justification for response-level reward approaches. Our findings pave the way for more practical, efficient LLM fine-tuning, allowing developers to treat training algorithms as black boxes and focus on improving the response-level reward model with auxiliary sub-models. We also offer a detailed analysis of popular RL and non-RL methods, comparing their theoretical foundations and practical advantages across common LLM tasks. Finally, we propose a new algorithm: Token-Reinforced Policy Optimization (TRePO), a theoretically grounded method that is simpler than PPO, matches GRPO in memory efficiency, and holds promise for broad applicability.
>
---
#### [new 139] Turning LLM Activations Quantization-Friendly
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型（LLM）量化中的激活值难量化问题。通过分析异常值对量化误差的影响，提出结合通道缩放与旋转的混合方法，并引入新指标评估量化难度，优化整数运算效率。**

- **链接: [http://arxiv.org/pdf/2506.01967v1](http://arxiv.org/pdf/2506.01967v1)**

> **作者:** Patrik Czakó; Gábor Kertész; Sándor Szénási
>
> **备注:** 6 pages, 5 figures. Accepted to SACI 2025 conference proceedings
>
> **摘要:** Quantization effectively reduces the serving costs of Large Language Models (LLMs) by speeding up data movement through compressed parameters and enabling faster operations via integer arithmetic. However, activating integer arithmetic requires quantizing both weights and activations, which poses challenges due to the significant outliers in LLMs that increase quantization error. In this work, we investigate these outliers with an emphasis on their effect on layer-wise quantization error, then examine how smoothing and rotation transform the observed values. Our primary contributions include introducing a new metric to measure and visualize quantization difficulty based on channel magnitudes, as well as proposing a hybrid approach that applies channel-wise scaling before rotation, supported by a mathematical formulation of its benefits.
>
---
#### [new 140] An Exploratory Framework for Future SETI Applications: Detecting Generative Reactivity via Language Models
- **分类: astro-ph.IM; cs.CL**

- **简介: 该论文属于自然语言处理与地外信号探测交叉任务，旨在探索语言模型能否从噪声样输入中识别潜在结构。为解决传统SETI方法难以应对的未知编码问题，作者提出“语义诱导潜力”指标，评估GPT-2对多种声音（如鲸歌、鸟鸣、白噪音）的反应，发现生物声音比白噪音更易激发语言模型生成有结构输出，提示该方法或可辅助搜寻非传统形式的地外通讯。**

- **链接: [http://arxiv.org/pdf/2506.02730v1](http://arxiv.org/pdf/2506.02730v1)**

> **作者:** Po-Chieh Yu
>
> **备注:** submitted to the International Journal of Astrobiology
>
> **摘要:** We present an exploratory framework to test whether noise-like input can induce structured responses in language models. Instead of assuming that extraterrestrial signals must be decoded, we evaluate whether inputs can trigger linguistic behavior in generative systems. This shifts the focus from decoding to viewing structured output as a sign of underlying regularity in the input. We tested GPT-2 small, a 117M-parameter model trained on English text, using four types of acoustic input: human speech, humpback whale vocalizations, Phylloscopus trochilus birdsong, and algorithmically generated white noise. All inputs were treated as noise-like, without any assumed symbolic encoding. To assess reactivity, we defined a composite score called Semantic Induction Potential (SIP), combining entropy, syntax coherence, compression gain, and repetition penalty. Results showed that whale and bird vocalizations had higher SIP scores than white noise, while human speech triggered only moderate responses. This suggests that language models may detect latent structure even in data without conventional semantics. We propose that this approach could complement traditional SETI methods, especially in cases where communicative intent is unknown. Generative reactivity may offer a different way to identify data worth closer attention.
>
---
#### [new 141] VLCD: Vision-Language Contrastive Distillation for Accurate and Efficient Automatic Placenta Analysis
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决现有自动胎盘病理检测方法计算量大、部署受限的问题。作者提出了VLCD方法，包含文本锚定的知识蒸馏策略和无监督预训练策略，以提升模型效率与准确性，尤其适用于资源有限环境。**

- **链接: [http://arxiv.org/pdf/2506.02229v1](http://arxiv.org/pdf/2506.02229v1)**

> **作者:** Manas Mehta; Yimu Pan; Kelly Gallagher; Alison D. Gernand; Jeffery A. Goldstein; Delia Mwinyelle; Leena Mithal; James Z. Wang
>
> **备注:** Proceedings of the 9th International Workshop on Health Intelligence, in conjunction with the Annual AAAI Conference on Artificial Intelligence, Philadelphia, Pennsylvania, March 2025
>
> **摘要:** Pathological examination of the placenta is an effective method for detecting and mitigating health risks associated with childbirth. Recent advancements in AI have enabled the use of photographs of the placenta and pathology reports for detecting and classifying signs of childbirth-related pathologies. However, existing automated methods are computationally extensive, which limits their deployability. We propose two modifications to vision-language contrastive learning (VLC) frameworks to enhance their accuracy and efficiency: (1) text-anchored vision-language contrastive knowledge distillation (VLCD)-a new knowledge distillation strategy for medical VLC pretraining, and (2) unsupervised predistillation using a large natural images dataset for improved initialization. Our approach distills efficient neural networks that match or surpass the teacher model in performance while achieving model compression and acceleration. Our results showcase the value of unsupervised predistillation in improving the performance and robustness of our approach, specifically for lower-quality images. VLCD serves as an effective way to improve the efficiency and deployability of medical VLC approaches, making AI-based healthcare solutions more accessible, especially in resource-constrained environments.
>
---
## 更新

#### [replaced 001] The time scale of redundancy between prosody and linguistic context
- **分类: cs.CL; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2503.11630v3](http://arxiv.org/pdf/2503.11630v3)**

> **作者:** Tamar I. Regev; Chiebuka Ohams; Shaylee Xie; Lukas Wolf; Evelina Fedorenko; Alex Warstadt; Ethan G. Wilcox; Tiago Pimentel
>
> **备注:** 13 pages, 4 figures, accepted to ACL. Updated following ACL reviewers comments
>
> **摘要:** In spoken communication, information is transmitted not only via words, but also through a rich array of non-verbal signals, including prosody--the non-segmental auditory features of speech. Do these different communication channels carry distinct information? Prior work has shown that the information carried by prosodic features is substantially redundant with that carried by the surrounding words. Here, we systematically examine the time scale of this relationship, studying how it varies with the length of past and future contexts. We find that a word's prosodic features require an extended past context (3-8 words across different features) to be reliably predicted. Given that long-scale contextual information decays in memory, prosody may facilitate communication by adding information that is locally unique. We also find that a word's prosodic features show some redundancy with future words, but only with a short scale of 1-2 words, consistent with reports of incremental short-term planning in language production. Thus, prosody may facilitate communication by helping listeners predict upcoming material. In tandem, our results highlight potentially distinct roles that prosody plays in facilitating integration of words into past contexts and in helping predict upcoming words.
>
---
#### [replaced 002] Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00653v2](http://arxiv.org/pdf/2506.00653v2)**

> **作者:** Femi Bello; Anubrata Das; Fanzhi Zeng; Fangcong Yin; Liu Leqi
>
> **摘要:** It has been hypothesized that neural networks with similar architectures trained on similar data learn shared representations relevant to the learning task. We build on this idea by extending the conceptual framework where representations learned across models trained on the same data can be expressed as linear combinations of a \emph{universal} set of basis features. These basis features underlie the learning task itself and remain consistent across models, regardless of scale. From this framework, we propose the \textbf{Linear Representation Transferability (LRT)} Hypothesis -- that there exists an affine transformation between the representation spaces of different models. To test this hypothesis, we learn affine mappings between the hidden states of models of different sizes and evaluate whether steering vectors -- directions in hidden state space associated with specific model behaviors -- retain their semantic effect when transferred from small to large language models using the learned mappings. We find strong empirical evidence that such affine mappings can preserve steering behaviors. These findings suggest that representations learned by small models can be used to guide the behavior of large models, and that the LRT hypothesis may be a promising direction on understanding representation alignment across model scales.
>
---
#### [replaced 003] Unique Hard Attention: A Tale of Two Sides
- **分类: cs.LG; cs.CC; cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2503.14615v2](http://arxiv.org/pdf/2503.14615v2)**

> **作者:** Selim Jerad; Anej Svete; Jiaoda Li; Ryan Cotterell
>
> **摘要:** Understanding the expressive power of transformers has recently attracted attention, as it offers insights into their abilities and limitations. Many studies analyze unique hard attention transformers, where attention selects a single position that maximizes the attention scores. When multiple positions achieve the maximum score, either the rightmost or the leftmost of those is chosen. In this paper, we highlight the importance of this seeming triviality. Recently, finite-precision transformers with both leftmost- and rightmost-hard attention were shown to be equivalent to Linear Temporal Logic (LTL). We show that this no longer holds with only leftmost-hard attention -- in that case, they correspond to a \emph{strictly weaker} fragment of LTL. Furthermore, we show that models with leftmost-hard attention are equivalent to \emph{soft} attention, suggesting they may better approximate real-world transformers than right-attention models. These findings refine the landscape of transformer expressivity and underscore the role of attention directionality.
>
---
#### [replaced 004] Ola: Pushing the Frontiers of Omni-Modal Language Model
- **分类: cs.CV; cs.CL; cs.MM; cs.SD; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.04328v3](http://arxiv.org/pdf/2502.04328v3)**

> **作者:** Zuyan Liu; Yuhao Dong; Jiahui Wang; Ziwei Liu; Winston Hu; Jiwen Lu; Yongming Rao
>
> **摘要:** Recent advances in large language models, particularly following GPT-4o, have sparked increasing interest in developing omni-modal models capable of understanding more modalities. While some open-source alternatives have emerged, there is still a notable lag behind specialized single-modality models in performance. In this paper, we present Ola, an Omni-modal Language model that achieves competitive performance across image, video, and audio understanding compared to specialized counterparts, pushing the frontiers of the omni-modal language model to a large extent. We conduct a comprehensive exploration of architectural design, data curation, and training strategies essential for building a robust omni-modal model. Ola incorporates advanced visual understanding and audio recognition capabilities through several critical and effective improvements over mainstream baselines. Moreover, we rethink inter-modal relationships during omni-modal training, emphasizing cross-modal alignment with video as a central bridge, and propose a progressive training pipeline that begins with the most distinct modalities and gradually moves towards closer modality alignment. Extensive experiments demonstrate that Ola surpasses existing open omni-modal LLMs across all modalities while achieving highly competitive performance compared to state-of-the-art specialized models of similar sizes. We aim to make Ola a fully open omni-modal understanding solution to advance future research in this emerging field. Model weights, code, and data are open-sourced at https://github.com/Ola-Omni/Ola.
>
---
#### [replaced 005] Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14354v2](http://arxiv.org/pdf/2502.14354v2)**

> **作者:** Moxin Li; Yuantao Zhang; Wenjie Wang; Wentao Shi; Zhuo Liu; Fuli Feng; Tat-Seng Chua
>
> **备注:** ACL findings (2025)
>
> **摘要:** Multi-Objective Alignment (MOA) aims to align LLMs' responses with multiple human preference objectives, with Direct Preference Optimization (DPO) emerging as a prominent approach. However, we find that DPO-based MOA approaches suffer from widespread preference conflicts in the data, where different objectives favor different responses. This results in conflicting optimization directions, hindering the optimization on the Pareto Front. To address this, we propose to construct Pareto-optimal responses to resolve preference conflicts. To efficiently obtain and utilize such responses, we propose a self-improving DPO framework that enables LLMs to self-generate and select Pareto-optimal responses for self-supervised preference alignment. Extensive experiments on two datasets demonstrate the superior Pareto Front achieved by our framework compared to various baselines. Code is available at https://github.com/zyttt-coder/SIPO.
>
---
#### [replaced 006] Meta-Learning Neural Mechanisms rather than Bayesian Priors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16048v2](http://arxiv.org/pdf/2503.16048v2)**

> **作者:** Michael Goodale; Salvador Mascarenhas; Yair Lakretz
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** Children acquire language despite being exposed to several orders of magnitude less data than large language models require. Meta-learning has been proposed as a way to integrate human-like learning biases into neural-network architectures, combining both the structured generalizations of symbolic models with the scalability of neural-network models. But what does meta-learning exactly imbue the model with? We investigate the meta-learning of formal languages and find that, contrary to previous claims, meta-trained models are not learning simplicity-based priors when meta-trained on datasets organised around simplicity. Rather, we find evidence that meta-training imprints neural mechanisms (such as counters) into the model, which function like cognitive primitives for the network on downstream tasks. Most surprisingly, we find that meta-training on a single formal language can provide as much improvement to a model as meta-training on 5000 different formal languages, provided that the formal language incentivizes the learning of useful neural mechanisms. Taken together, our findings provide practical implications for efficient meta-learning paradigms and new theoretical insights into linking symbolic theories and neural mechanisms.
>
---
#### [replaced 007] What Has Been Lost with Synthetic Evaluation?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22830v2](http://arxiv.org/pdf/2505.22830v2)**

> **作者:** Alexander Gill; Abhilasha Ravichander; Ana Marasović
>
> **备注:** v2: Fixed low resolution figures
>
> **摘要:** Large language models (LLMs) are increasingly used for data generation. However, creating evaluation benchmarks raises the bar for this emerging paradigm. Benchmarks must target specific phenomena, penalize exploiting shortcuts, and be challenging. Through two case studies, we investigate whether LLMs can meet these demands by generating reasoning over-text benchmarks and comparing them to those created through careful crowdsourcing. Specifically, we evaluate both the validity and difficulty of LLM-generated versions of two high-quality reading comprehension datasets: CondaQA, which evaluates reasoning about negation, and DROP, which targets reasoning about quantities. We find that prompting LLMs can produce variants of these datasets that are often valid according to the annotation guidelines, at a fraction of the cost of the original crowdsourcing effort. However, we show that they are less challenging for LLMs than their human-authored counterparts. This finding sheds light on what may have been lost by generating evaluation data with LLMs, and calls for critically reassessing the immediate use of this increasingly prevalent approach to benchmark creation.
>
---
#### [replaced 008] Improving Transformer Performance for French Clinical Notes Classification Using Mixture of Experts on a Limited Dataset
- **分类: cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2303.12892v3](http://arxiv.org/pdf/2303.12892v3)**

> **作者:** Thanh-Dung Le; Philippe Jouvet; Rita Noumeir
>
> **备注:** Accepted for publication in the IEEE Journal of Translational Engineering in Health and Medicine
>
> **摘要:** Transformer-based models have shown outstanding results in natural language processing but face challenges in applications like classifying small-scale clinical texts, especially with constrained computational resources. This study presents a customized Mixture of Expert (MoE) Transformer models for classifying small-scale French clinical texts at CHU Sainte-Justine Hospital. The MoE-Transformer addresses the dual challenges of effective training with limited data and low-resource computation suitable for in-house hospital use. Despite the success of biomedical pre-trained models such as CamemBERT-bio, DrBERT, and AliBERT, their high computational demands make them impractical for many clinical settings. Our MoE-Transformer model not only outperforms DistillBERT, CamemBERT, FlauBERT, and Transformer models on the same dataset but also achieves impressive results: an accuracy of 87\%, precision of 87\%, recall of 85\%, and F1-score of 86\%. While the MoE-Transformer does not surpass the performance of biomedical pre-trained BERT models, it can be trained at least 190 times faster, offering a viable alternative for settings with limited data and computational resources. Although the MoE-Transformer addresses challenges of generalization gaps and sharp minima, demonstrating some limitations for efficient and accurate clinical text classification, this model still represents a significant advancement in the field. It is particularly valuable for classifying small French clinical narratives within the privacy and constraints of hospital-based computational resources.
>
---
#### [replaced 009] COMPKE: Complex Question Answering under Knowledge Editing
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00829v2](http://arxiv.org/pdf/2506.00829v2)**

> **作者:** Keyuan Cheng; Zijian Kan; Zhixian He; Zhuoran Zhang; Muhammad Asif Ali; Ke Xu; Lijie Hu; Di Wang
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Knowledge Editing, which efficiently modifies the knowledge in large language models, has gathered great attention. Current benchmarks primarily use multi-hop question answering to assess and analyze newly injected or updated knowledge. However, we argue that these benchmarks fail to effectively evaluate how well the updated models apply this knowledge in real-life scenarios, particularly when questions require complex reasoning, involving one-to-many relationships or multi-step logical intersections. To fill in this gap, we introduce a new benchmark, COMPKE: Complex Question Answering under Knowledge Editing, which includes 11,924 complex questions that reflect real-life situations. We conduct an extensive evaluation of four knowledge editing methods on COMPKE, revealing that their effectiveness varies notably across different models. For instance, MeLLo attains an accuracy of 39.47 on GPT-4O-MINI, but this drops sharply to 3.83 on QWEN2.5-3B. We further investigate the underlying causes of these disparities from both methodological and model-specific perspectives. The datasets are available at https://github.com/kzjkzj666/CompKE.
>
---
#### [replaced 010] A$^2$ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and Query-Aware Vector Quantization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12665v2](http://arxiv.org/pdf/2502.12665v2)**

> **作者:** Junhui He; Junna Xing; Nan Wang; Rui Xu; Shangyu Wu; Peng Zhou; Qiang Liu; Chun Jason Xue; Qingan Li
>
> **摘要:** Long context large language models (LLMs) pose significant challenges for efficient serving due to the large memory footprint and high access overhead of KV cache. Retrieval-based KV cache reduction methods can mitigate these challenges, typically by offloading the complete KV cache to CPU and retrieving necessary tokens on demand during inference. However, these methods still suffer from unsatisfactory accuracy degradation and extra retrieval overhead. To address these limitations, this paper proposes A$^2$ATS, a novel retrieval-based KV cache reduction method. A$^2$ATS aims to obtain an accurate approximation of attention scores by applying the vector quantization technique to key states, thereby enabling efficient and precise retrieval of the top-K tokens. First, we propose Windowed Rotary Position Embedding, which decouples the positional dependency from query and key states after position embedding. Then, we propose query-aware vector quantization that optimizes the objective of attention score approximation directly. Finally, we design the heterogeneous inference architecture for KV cache offloading, enabling long context serving with larger batch sizes. Experimental results demonstrate that A$^2$ATS can achieve a lower performance degradation with similar or lower overhead compared to existing methods, thereby increasing long context serving throughput by up to $2.7 \times$.
>
---
#### [replaced 011] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01776v2](http://arxiv.org/pdf/2506.01776v2)**

> **作者:** Yile Liu; Ziwei Ma; Xiu Jiang; Jinglu Hu; Jing Chang; Liang Li
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** With the rapid adoption of large language models (LLMs) in natural language processing, the ability to follow instructions has emerged as a key metric for evaluating their practical utility. However, existing evaluation methods often focus on single-language scenarios, overlooking the challenges and differences present in multilingual and cross-lingual contexts. To address this gap, we introduce MaXIFE: a comprehensive evaluation benchmark designed to assess instruction-following capabilities across 23 different languages with 1667 verifiable instruction tasks. MaXIFE integrates both Rule-Based Evaluation and Model-Based Evaluation, ensuring a balance of efficiency and accuracy. We applied MaXIFE to evaluate several leading commercial LLMs, establishing baseline results for future comparisons. By providing a standardized tool for multilingual instruction-following evaluation, MaXIFE aims to advance research and development in natural language processing.
>
---
#### [replaced 012] Revealing the Parallel Multilingual Learning within Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.09073v3](http://arxiv.org/pdf/2403.09073v3)**

> **作者:** Yongyu Mu; Peinan Feng; Zhiquan Cao; Yuzhang Wu; Bei Li; Chenglong Wang; Tong Xiao; Kai Song; Tongran Liu; Chunliang Zhang; Jingbo Zhu
>
> **备注:** Accepted to EMNLP 2024
>
> **摘要:** In this study, we reveal an in-context learning (ICL) capability of multilingual large language models (LLMs): by translating the input to several languages, we provide Parallel Input in Multiple Languages (PiM) to LLMs, which significantly enhances their comprehension abilities. To test this capability, we design extensive experiments encompassing 8 typical datasets, 7 languages and 8 state-of-the-art multilingual LLMs. Experimental results show that (1) incorporating more languages help PiM surpass the conventional ICL further; (2) even combining with the translations that are inferior to baseline performance can also help. Moreover, by examining the activated neurons in LLMs, we discover a counterintuitive but interesting phenomenon. Contrary to the common thought that PiM would activate more neurons than monolingual input to leverage knowledge learned from diverse languages, PiM actually inhibits neurons and promotes more precise neuron activation especially when more languages are added. This phenomenon aligns with the neuroscience insight about synaptic pruning, which removes less used neural connections, strengthens remainders, and then enhances brain intelligence.
>
---
#### [replaced 013] IndicRAGSuite: Large-Scale Datasets and a Benchmark for Indian Language RAG Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01615v2](http://arxiv.org/pdf/2506.01615v2)**

> **作者:** Pasunuti Prasanjith; Prathmesh B More; Anoop Kunchukuttan; Raj Dabre
>
> **备注:** WIP
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems enable language models to access relevant information and generate accurate, well-grounded, and contextually informed responses. However, for Indian languages, the development of high-quality RAG systems is hindered by the lack of two critical resources: (1) evaluation benchmarks for retrieval and generation tasks, and (2) large-scale training datasets for multilingual retrieval. Most existing benchmarks and datasets are centered around English or high-resource languages, making it difficult to extend RAG capabilities to the diverse linguistic landscape of India. To address the lack of evaluation benchmarks, we create IndicMSMarco, a multilingual benchmark for evaluating retrieval quality and response generation in 13 Indian languages, created via manual translation of 1000 diverse queries from MS MARCO-dev set. To address the need for training data, we build a large-scale dataset of (question, answer, relevant passage) tuples derived from the Wikipedias of 19 Indian languages using state-of-the-art LLMs. Additionally, we include translated versions of the original MS MARCO dataset to further enrich the training data and ensure alignment with real-world information-seeking tasks. Resources are available here: https://huggingface.co/collections/ai4bharat/indicragsuite-683e7273cb2337208c8c0fcb
>
---
#### [replaced 014] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21863v3](http://arxiv.org/pdf/2505.21863v3)**

> **作者:** Shikhhar Siingh; Abhinav Rawat; Chitta Baral; Vivek Gupta
>
> **摘要:** Publicly significant images from events hold valuable contextual information, crucial for journalism and education. However, existing methods often struggle to extract this relevance accurately. To address this, we introduce GETReason (Geospatial Event Temporal Reasoning), a framework that moves beyond surface-level image descriptions to infer deeper contextual meaning. We propose that extracting global event, temporal, and geospatial information enhances understanding of an image's significance. Additionally, we introduce GREAT (Geospatial Reasoning and Event Accuracy with Temporal Alignment), a new metric for evaluating reasoning-based image understanding. Our layered multi-agent approach, assessed using a reasoning-weighted metric, demonstrates that meaningful insights can be inferred, effectively linking images to their broader event context.
>
---
#### [replaced 015] Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.00622v2](http://arxiv.org/pdf/2506.00622v2)**

> **作者:** Haesung Pyun; Yoonah Park; Yohan Jo
>
> **备注:** This paper has been accepted for publication at ACL 2025
>
> **摘要:** In dialogue state tracking (DST), in-context learning comprises a retriever that selects labeled dialogues as in-context examples and a DST model that uses these examples to infer the dialogue state of the query dialogue. Existing methods for constructing training data for retrievers suffer from three key limitations: (1) the synergistic effect of examples is not considered, (2) the linguistic characteristics of the query are not sufficiently factored in, and (3) scoring is not directly optimized for DST performance. Consequently, the retriever can fail to retrieve examples that would substantially improve DST performance. To address these issues, we present CombiSearch, a method that scores effective in-context examples based on their combinatorial impact on DST performance. Our evaluation on MultiWOZ shows that retrievers trained with CombiSearch surpass state-of-the-art models, achieving a 20x gain in data efficiency and generalizing well to the SGD dataset. Moreover, CombiSearch attains a 12% absolute improvement in the upper bound DST performance over traditional approaches when no retrieval errors are assumed. This significantly increases the headroom for practical DST performance while demonstrating that existing methods rely on suboptimal data for retriever training.
>
---
#### [replaced 016] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00773v2](http://arxiv.org/pdf/2506.00773v2)**

> **作者:** Boheng Sheng; Jiacheng Yao; Meicong Zhang; Guoxiu He
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) often struggle to accurately read and comprehend extremely long texts. Current methods for improvement typically rely on splitting long contexts into fixed-length chunks. However, fixed truncation risks separating semantically relevant content, leading to ambiguity and compromising accurate understanding. To overcome this limitation, we propose a straightforward approach for dynamically separating and selecting chunks of long context, facilitating a more streamlined input for LLMs. In particular, we compute semantic similarities between adjacent sentences, using lower similarities to adaptively divide long contexts into variable-length chunks. We further train a question-aware classifier to select sensitive chunks that are critical for answering specific questions. Experimental results on both single-hop and multi-hop question-answering benchmarks show that the proposed approach consistently outperforms strong baselines. Notably, it maintains robustness across a wide range of input lengths, handling sequences of up to 256k tokens. Our datasets and code are available at the following link: https://github.com/ECNU-Text-Computing/DCS
>
---
#### [replaced 017] Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14545v3](http://arxiv.org/pdf/2406.14545v3)**

> **作者:** Đorđe Klisura; Anthony Rios
>
> **备注:** Accepted to NAACL 2025 Findings
>
> **摘要:** Text-to-SQL systems empower users to interact with databases using natural language, automatically translating queries into executable SQL code. However, their reliance on database schema information for SQL generation exposes them to significant security vulnerabilities, particularly schema inference attacks that can lead to unauthorized data access or manipulation. In this paper, we introduce a novel zero-knowledge framework for reconstructing the underlying database schema of text-to-SQL models without any prior knowledge of the database. Our approach systematically probes text-to-SQL models with specially crafted questions and leverages a surrogate GPT-4 model to interpret the outputs, effectively uncovering hidden schema elements -- including tables, columns, and data types. We demonstrate that our method achieves high accuracy in reconstructing table names, with F1 scores of up to .99 for generative models and .78 for fine-tuned models, underscoring the severity of schema leakage risks. We also show that our attack can steal prompt information in non-text-to-SQL models. Furthermore, we propose a simple protection mechanism for generative models and empirically show its limitations in mitigating these attacks.
>
---
#### [replaced 018] Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16014v3](http://arxiv.org/pdf/2505.16014v3)**

> **作者:** Yash Saxena; Ankur Padia; Mandar S Chaudhary; Kalpa Gunaratna; Srinivasan Parthasarathy; Manas Gaur
>
> **摘要:** Traditional Retrieval-Augmented Generation (RAG) pipelines rely on similarity-based retrieval and re-ranking, which depend on heuristics such as top-k, and lack explainability, interpretability, and robustness against adversarial content. To address this gap, we propose a novel method METEORA that replaces re-ranking in RAG with a rationale-driven selection approach. METEORA operates in two stages. First, a general-purpose LLM is preference-tuned to generate rationales conditioned on the input query using direct preference optimization. These rationales guide the evidence chunk selection engine, which selects relevant chunks in three stages: pairing individual rationales with corresponding retrieved chunks for local relevance, global selection with elbow detection for adaptive cutoff, and context expansion via neighboring chunks. This process eliminates the need for top-k heuristics. The rationales are also used for consistency check using a Verifier LLM to detect and filter poisoned or misleading content for safe generation. The framework provides explainable and interpretable evidence flow by using rationales consistently across both selection and verification. Our evaluation across six datasets spanning legal, financial, and academic research domains shows that METEORA improves generation accuracy by 33.34% while using approximately 50% fewer chunks than state-of-the-art re-ranking methods. In adversarial settings, METEORA significantly improves the F1 score from 0.10 to 0.44 over the state-of-the-art perplexity-based defense baseline, demonstrating strong resilience to poisoning attacks. Code available at: https://anonymous.4open.science/r/METEORA-DC46/README.md
>
---
#### [replaced 019] Logits are All We Need to Adapt Closed Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06806v3](http://arxiv.org/pdf/2502.06806v3)**

> **作者:** Gaurush Hiranandani; Haolun Wu; Subhojyoti Mukherjee; Sanmi Koyejo
>
> **备注:** 29 pages, 8 figures
>
> **摘要:** Many commercial Large Language Models (LLMs) are often closed-source, limiting developers to prompt tuning for aligning content generation with specific applications. While these models currently do not provide access to token logits, we argue that if such access were available, it would enable more powerful adaptation techniques beyond prompt engineering. In this paper, we propose a token-level probability reweighting framework that, given access to logits and a small amount of task-specific data, can effectively steer black-box LLMs toward application-specific content generation. Our approach views next-token prediction through the lens of supervised classification. We show that aligning black-box LLMs with task-specific data can be formulated as a label noise correction problem, leading to Plugin model -- an autoregressive probability reweighting model that operates solely on logits. We provide theoretical justification for why reweighting logits alone is sufficient for task adaptation. Extensive experiments with multiple datasets, LLMs, and reweighting models demonstrate the effectiveness of our method, advocating for broader access to token logits in closed-source models.
>
---
#### [replaced 020] CHEER-Ekman: Fine-grained Embodied Emotion Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01047v2](http://arxiv.org/pdf/2506.01047v2)**

> **作者:** Phan Anh Duong; Cat Luong; Divyesh Bommana; Tianyu Jiang
>
> **备注:** ACL 2025
>
> **摘要:** Emotions manifest through physical experiences and bodily reactions, yet identifying such embodied emotions in text remains understudied. We present an embodied emotion classification dataset, CHEER-Ekman, extending the existing binary embodied emotion dataset with Ekman's six basic emotion categories. Using automatic best-worst scaling with large language models, we achieve performance superior to supervised approaches on our new dataset. Our investigation reveals that simplified prompting instructions and chain-of-thought reasoning significantly improve emotion recognition accuracy, enabling smaller models to achieve competitive performance with larger ones. Our dataset is publicly available at: https://github.com/menamerai/cheer-ekman.
>
---
#### [replaced 021] LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.06705v3](http://arxiv.org/pdf/2405.06705v3)**

> **作者:** Zhuoxuan Jiang; Haoyuan Peng; Shanshan Feng; Fan Li; Dongsheng Li
>
> **备注:** Accepted by IJCAI 2024
>
> **摘要:** Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
>
---
#### [replaced 022] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.00382v2](http://arxiv.org/pdf/2410.00382v2)**

> **作者:** Shota Takashiro; Takeshi Kojima; Andrew Gambardella; Qi Cao; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** Accepted at ACL 2025 (Findings)
>
> **摘要:** As large language models (LLMs) are applied across diverse domains, the ability to selectively unlearn specific information is becoming increasingly essential. For instance, LLMs are expected to selectively provide confidential information to authorized internal users, such as employees or trusted partners, while withholding it from external users, including the general public and unauthorized entities. Therefore, we propose a novel method termed ``in-context knowledge unlearning'', which enables the model to selectively forget information in test-time based on the query context. Our method fine-tunes pre-trained LLMs to enable prompt unlearning of target knowledge within the context, while preserving unrelated information. Experiments on TOFU, AGE and RWKU datasets using Llama2-7B/13B and Mistral-7B models demonstrate that our method achieves up to 95% forget accuracy while retaining 80% of unrelated knowledge, significantly outperforming baselines in both in-domain and out-of-domain scenarios. Further investigation of the model's internal behavior revealed that while fine-tuned LLMs generate correct predictions in the middle layers and preserve them up to the final layer. However, the decision to forget is made only at the last layer, i.e. ``LLMs pretend to forget''. Our findings offer valuable insight into the improvement of the robustness of the unlearning mechanisms in LLMs, laying a foundation for future research in the field.
>
---
#### [replaced 023] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.06645v2](http://arxiv.org/pdf/2501.06645v2)**

> **作者:** Tong Liu; Xiao Yu; Wenxuan Zhou; Jindong Gu; Volker Tresp
>
> **备注:** ACL 2025
>
> **摘要:** Efficient preference optimization algorithms such as Direct Preference Optimization (DPO) have become a popular approach in aligning large language models (LLMs) with human preferences. These algorithms implicitly treat the LLM as a reward model, and focus on training it to correct misranked preference pairs. However, recent work~\citep{chen2024preference} empirically finds that DPO training \textit{rarely improves these misranked preference pairs}, despite its gradient emphasizing on these cases. We introduce FocalPO, a DPO variant that instead \textit{down-weighs} misranked preference pairs and prioritizes enhancing the model's understanding of pairs that it can already rank correctly. Inspired by Focal Loss used in vision tasks, FocalPO achieves this by adding a modulating factor to dynamically scale DPO loss. Our experiment demonstrates that FocalPO surpasses DPO and its variants on popular benchmarks like Alpaca Eval 2.0 using Mistral-Base-7B and Llama-3-Instruct-8B, with the introduced hyperparameter fixed. Additionally, we empirically reveals how FocalPO affects training on correct and incorrect sample groups, further underscoring its effectiveness.
>
---
#### [replaced 024] X-Driver: Explainable Autonomous Driving with Vision-Language Models
- **分类: cs.RO; cs.CL; cs.CV; cs.ET**

- **链接: [http://arxiv.org/pdf/2505.05098v2](http://arxiv.org/pdf/2505.05098v2)**

> **作者:** Wei Liu; Jiyuan Zhang; Binxiong Zheng; Yufeng Hu; Yingzhan Lin; Zengfeng Zeng
>
> **摘要:** End-to-end autonomous driving has advanced significantly, offering benefits such as system simplicity and stronger driving performance in both open-loop and closed-loop settings than conventional pipelines. However, existing frameworks still suffer from low success rates in closed-loop evaluations, highlighting their limitations in real-world deployment. In this paper, we introduce X-Driver, a unified multi-modal large language models(MLLMs) framework designed for closed-loop autonomous driving, leveraging Chain-of-Thought(CoT) and autoregressive modeling to enhance perception and decision-making. We validate X-Driver across multiple autonomous driving tasks using public benchmarks in CARLA simulation environment, including Bench2Drive[6]. Our experimental results demonstrate superior closed-loop performance, surpassing the current state-of-the-art(SOTA) while improving the interpretability of driving decisions. These findings underscore the importance of structured reasoning in end-to-end driving and establish X-Driver as a strong baseline for future research in closed-loop autonomous driving.
>
---
#### [replaced 025] How Bidirectionality Helps Language Models Learn Better via Dynamic Bottleneck Estimation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00859v2](http://arxiv.org/pdf/2506.00859v2)**

> **作者:** Md Kowsher; Nusrat Jahan Prottasha; Shiyun Xu; Shetu Mohanto; Chen Chen; Ozlem Garibay; Niloofar Yousefi
>
> **摘要:** Bidirectional language models have better context understanding and perform better than unidirectional models on natural language understanding tasks, yet the theoretical reasons behind this advantage remain unclear. In this work, we investigate this disparity through the lens of the Information Bottleneck (IB) principle, which formalizes a trade-off between compressing input information and preserving task-relevant content. We propose FlowNIB, a dynamic and scalable method for estimating mutual information during training that addresses key limitations of classical IB approaches, including computational intractability and fixed trade-off schedules. Theoretically, we show that bidirectional models retain more mutual information and exhibit higher effective dimensionality than unidirectional models. To support this, we present a generalized framework for measuring representational complexity and prove that bidirectional representations are strictly more informative under mild conditions. We further validate our findings through extensive experiments across multiple models and tasks using FlowNIB, revealing how information is encoded and compressed throughout training. Together, our work provides a principled explanation for the effectiveness of bidirectional architectures and introduces a practical tool for analyzing information flow in deep language models.
>
---
#### [replaced 026] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15299v2](http://arxiv.org/pdf/2505.15299v2)**

> **作者:** Maodong Li; Longyin Zhang; Fang Kong
>
> **备注:** 17 pages, 5 figures, accepted to the Findings of ACL 2025
>
> **摘要:** Multi-hop question generation (MQG) aims to generate questions that require synthesizing multiple information snippets from documents to derive target answers. The primary challenge lies in effectively pinpointing crucial information snippets related to question-answer (QA) pairs, typically relying on keywords. However, existing works fail to fully utilize the guiding potential of keywords and neglect to differentiate the distinct roles of question-specific and document-specific keywords. To address this, we define dual-perspective keywords (i.e., question and document keywords) and propose a Dual-Perspective Keyword-Guided (DPKG) framework, which seamlessly integrates keywords into the multi-hop question generation process. We argue that question keywords capture the questioner's intent, whereas document keywords reflect the content related to the QA pair. Functionally, question and document keywords work together to pinpoint essential information snippets in the document, with question keywords required to appear in the generated question. The DPKG framework consists of an expanded transformer encoder and two answer-aware transformer decoders for keyword and question generation, respectively. Extensive experiments demonstrate the effectiveness of our work, showcasing its promising performance and underscoring its significant value in the MQG task.
>
---
#### [replaced 027] A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.15806v2](http://arxiv.org/pdf/2502.15806v2)**

> **作者:** Yang Yao; Xuan Tong; Ruofan Wang; Yixu Wang; Lujundong Li; Liang Liu; Yan Teng; Yingchun Wang
>
> **摘要:** Large Reasoning Models (LRMs) have significantly advanced beyond traditional Large Language Models (LLMs) with their exceptional logical reasoning capabilities, yet these improvements introduce heightened safety risks. When subjected to jailbreak attacks, their ability to generate more targeted and organized content can lead to greater harm. Although some studies claim that reasoning enables safer LRMs against existing LLM attacks, they overlook the inherent flaws within the reasoning process itself. To address this gap, we propose the first jailbreak attack targeting LRMs, exploiting their unique vulnerabilities stemming from the advanced reasoning capabilities. Specifically, we introduce a Chaos Machine, a novel component to transform attack prompts with diverse one-to-one mappings. The chaos mappings iteratively generated by the machine are embedded into the reasoning chain, which strengthens the variability and complexity and also promotes a more robust attack. Based on this, we construct the Mousetrap framework, which makes attacks projected into nonlinear-like low sample spaces with mismatched generalization enhanced. Also, due to the more competing objectives, LRMs gradually maintain the inertia of unpredictable iterative reasoning and fall into our trap. Success rates of the Mousetrap attacking o1-mini, Claude-Sonnet and Gemini-Thinking are as high as 96%, 86% and 98% respectively on our toxic dataset Trotter. On benchmarks such as AdvBench, StrongREJECT, and HarmBench, attacking Claude-Sonnet, well-known for its safety, Mousetrap can astonishingly achieve success rates of 87.5%, 86.58% and 93.13% respectively. Attention: This paper contains inappropriate, offensive and harmful content.
>
---
#### [replaced 028] Can reasoning models comprehend mathematical problems in Chinese ancient texts? An empirical study based on data from Suanjing Shishu
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16660v2](http://arxiv.org/pdf/2505.16660v2)**

> **作者:** Liu Chang; Wang Dongbo; Liu liu; Zhao Zhixiao
>
> **备注:** 29pages, 7 figures
>
> **摘要:** This study addresses the challenges in intelligent processing of Chinese ancient mathematical classics by constructing Guji_MATH, a benchmark for evaluating classical texts based on Suanjing Shishu. It systematically assesses the mathematical problem-solving capabilities of mainstream reasoning models under the unique linguistic constraints of classical Chinese. Through machine-assisted annotation and manual verification, 538 mathematical problems were extracted from 8 canonical texts, forming a structured dataset centered on the "Question-Answer-Solution" framework, supplemented by problem types and difficulty levels. Dual evaluation modes--closed-book (autonomous problem-solving) and open-book (reproducing classical solution methods)--were designed to evaluate the performance of six reasoning models on ancient Chinese mathematical problems. Results indicate that reasoning models can partially comprehend and solve these problems, yet their overall performance remains inferior to benchmarks on modern mathematical tasks. Enhancing models' classical Chinese comprehension and cultural knowledge should be prioritized for optimization. This study provides methodological support for mining mathematical knowledge from ancient texts and disseminating traditional culture, while offering new perspectives for evaluating cross-linguistic and cross-cultural capabilities of reasoning models.
>
---
#### [replaced 029] d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12216v2](http://arxiv.org/pdf/2504.12216v2)**

> **作者:** Siyan Zhao; Devaansh Gupta; Qinqing Zheng; Aditya Grover
>
> **备注:** 27 pages, project page at https://dllm-reasoning.github.io/
>
> **摘要:** Recent large language models (LLMs) have demonstrated strong reasoning capabilities that benefits from online reinforcement learning (RL). These capabilities have primarily been demonstrated within the left-to-right autoregressive (AR) generation paradigm. In contrast, non-autoregressive paradigms based on diffusion generate text in a coarse-to-fine manner. Although recent diffusion-based large language models (dLLMs) have achieved competitive language modeling performance compared to their AR counterparts, it remains unclear if dLLMs can also leverage recent advances in LLM reasoning. To this end, we propose d1, a framework to adapt pre-trained masked dLLMs into reasoning models via a combination of supervised finetuning (SFT) and RL. Specifically, we develop and extend techniques to improve reasoning in pretrained dLLMs: (a) we utilize a masked SFT technique to distill knowledge and instill self-improvement behavior directly from existing datasets, and (b) we introduce a novel critic-free, policy-gradient based RL algorithm called diffu-GRPO, the first integration of policy gradient methods to masked dLLMs. Through empirical studies, we investigate the performance of different post-training recipes on multiple mathematical and planning benchmarks. We find that d1 yields the best performance and significantly improves performance of a state-of-the-art dLLM. Our code is released at https://dllm-reasoning.github.io/.
>
---
#### [replaced 030] EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.21271v4](http://arxiv.org/pdf/2410.21271v4)**

> **作者:** Shih-Yang Liu; Maksim Khadkevich; Nai Chit Fung; Charbel Sakr; Chao-Han Huck Yang; Chien-Yi Wang; Saurav Muralidharan; Hongxu Yin; Kwang-Ting Cheng; Jan Kautz; Yu-Chiang Frank Wang; Pavlo Molchanov; Min-Hung Chen
>
> **摘要:** While post-training compression techniques effectively reduce the memory footprint, latency, and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA, a novel fine-tuning-free method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior training-free low rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., $\mathbf{10.84\%}$ on ARC-Challenge, $\mathbf{6.74\%}$ on MathQA, and $\mathbf{6.74\%}$ on GSM8K) for LLaMA3-8B compressed to 3-bit. We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy of compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at https://github.com/NVlabs/EoRA.
>
---
#### [replaced 031] Self-Evolved Reward Learning for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.00418v3](http://arxiv.org/pdf/2411.00418v3)**

> **作者:** Chenghua Huang; Zhizhen Fan; Lu Wang; Fangkai Yang; Pu Zhao; Zeqi Lin; Qingwei Lin; Dongmei Zhang; Saravan Rajmohan; Qi Zhang
>
> **备注:** 23 pages,6 figures,Accepted to ICLR 2025
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) is a crucial technique for aligning language models with human preferences, playing a pivotal role in the success of conversational models like GPT-4, ChatGPT, and Llama 2. A core challenge in employing RLHF lies in training a reliable reward model (RM), which relies on high-quality labels typically provided by human experts or advanced AI system. These methods can be costly and may introduce biases that affect the language model's responses. As language models improve, human input may become less effective in further enhancing their performance. In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance, thereby boosting the capabilities of large language models (LLMs). Resources of this paper can be found at https://aka.ms/ser
>
---
#### [replaced 032] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02295v4](http://arxiv.org/pdf/2501.02295v4)**

> **作者:** Yachao Zhao; Bo Wang; Yan Wang; Dongming Zhao; Ruifang He; Yuexian Hou
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Large Language Models (LLMs) have been shown to exhibit various biases and stereotypes in their generated content. While extensive research has investigated biases in LLMs, prior work has predominantly focused on explicit bias, with minimal attention to implicit bias and the relation between these two forms of bias. This paper presents a systematic framework grounded in social psychology theories to investigate and compare explicit and implicit biases in LLMs. We propose a novel self-reflection-based evaluation framework that operates in two phases: first measuring implicit bias through simulated psychological assessment methods, then evaluating explicit bias by prompting LLMs to analyze their own generated content. Through extensive experiments on advanced LLMs across multiple social dimensions, we demonstrate that LLMs exhibit a substantial inconsistency between explicit and implicit biases: while explicit bias manifests as mild stereotypes, implicit bias exhibits strong stereotypes. We further investigate the underlying factors contributing to this explicit-implicit bias inconsistency, examining the effects of training data scale, model size, and alignment techniques. Experimental results indicate that while explicit bias declines with increased training data and model size, implicit bias exhibits a contrasting upward trend. Moreover, contemporary alignment methods effectively suppress explicit bias but show limited efficacy in mitigating implicit bias.
>
---
#### [replaced 033] AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.16873v2](http://arxiv.org/pdf/2404.16873v2)**

> **作者:** Anselm Paulus; Arman Zharmagambetov; Chuan Guo; Brandon Amos; Yuandong Tian
>
> **备注:** Accepted to ICML 2025. Code is available at http://github.com/facebookresearch/advprompter
>
> **摘要:** Large Language Models (LLMs) are vulnerable to jailbreaking attacks that lead to generation of inappropriate or harmful content. Manual red-teaming requires a time-consuming search for adversarial prompts, whereas automatic adversarial prompt generation often leads to semantically meaningless attacks that do not scale well. In this paper, we present a novel method that uses another LLM, called AdvPrompter, to generate human-readable adversarial prompts in seconds. AdvPrompter, which is trained using an alternating optimization algorithm, generates suffixes that veil the input instruction without changing its meaning, such that the TargetLLM is lured to give a harmful response. Experimental results on popular open source TargetLLMs show highly competitive results on the AdvBench and HarmBench datasets, that also transfer to closed-source black-box LLMs. We also show that training on adversarial suffixes generated by AdvPrompter is a promising strategy for improving the robustness of LLMs to jailbreaking attacks.
>
---
#### [replaced 034] Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.09689v3](http://arxiv.org/pdf/2411.09689v3)**

> **作者:** Seongmin Lee; Hsiang Hsu; Chun-Fu Chen; Duen Horng; Chau
>
> **备注:** 22 pages, 15 figures
>
> **摘要:** LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
>
---
#### [replaced 035] Checkpoint Merging via Bayesian Optimization in LLM Pretraining
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.19390v2](http://arxiv.org/pdf/2403.19390v2)**

> **作者:** Deyuan Liu; Zecheng Wang; Bingning Wang; Weipeng Chen; Chunshan Li; Zhiying Tu; Dianhui Chu; Bo Li; Dianbo Sui
>
> **摘要:** The rapid proliferation of large language models (LLMs) such as GPT-4 and Gemini underscores the intense demand for resources during their training processes, posing significant challenges due to substantial computational and environmental costs. To alleviate this issue, we propose checkpoint merging in pretraining LLM. This method utilizes LLM checkpoints with shared training trajectories, and is rooted in an extensive search space exploration for the best merging weight via Bayesian optimization. Through various experiments, we demonstrate that: (1) Our proposed methodology exhibits the capacity to augment pretraining, presenting an opportunity akin to obtaining substantial benefits at minimal cost; (2) Our proposed methodology, despite requiring a given held-out dataset, still demonstrates robust generalization capabilities across diverse domains, a pivotal aspect in pretraining.
>
---
#### [replaced 036] KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09825v2](http://arxiv.org/pdf/2505.09825v2)**

> **作者:** Peiqi Sui; Juan Diego Rodriguez; Philippe Laban; Dean Murphy; Joseph P. Dexter; Richard Jean So; Samuel Baker; Pramit Chaudhuri
>
> **备注:** ACL 2025 main
>
> **摘要:** Each year, tens of millions of essays are written and graded in college-level English courses. Students are asked to analyze literary and cultural texts through a process known as close reading, in which they gather textual details to formulate evidence-based arguments. Despite being viewed as a basis for critical thinking and widely adopted as a required element of university coursework, close reading has never been evaluated on large language models (LLMs), and multi-discipline benchmarks like MMLU do not include literature as a subject. To fill this gap, we present KRISTEVA, the first close reading benchmark for evaluating interpretive reasoning, consisting of 1331 multiple-choice questions adapted from classroom data. With KRISTEVA, we propose three progressively more difficult sets of tasks to approximate different elements of the close reading process, which we use to test how well LLMs may seem to understand and reason about literary works: 1) extracting stylistic features, 2) retrieving relevant contextual information from parametric knowledge, and 3) multi-hop reasoning between style and external contexts. Our baseline results find that, while state-of-the-art LLMs possess some college-level close reading competency (accuracy 49.7% - 69.7%), their performances still trail those of experienced human evaluators on 10 out of our 11 tasks.
>
---
#### [replaced 037] Free-text Rationale Generation under Readability Level Control
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.01384v3](http://arxiv.org/pdf/2407.01384v3)**

> **作者:** Yi-Sheng Hsu; Nils Feldhus; Sherzod Hakimov
>
> **备注:** ACL 2025 Workshop on Generation, Evaluation, and Metrics (GEM^2)
>
> **摘要:** Free-text rationales justify model decisions in natural language and thus become likable and accessible among approaches to explanation across many tasks. However, their effectiveness can be hindered by misinterpretation and hallucination. As a perturbation test, we investigate how large language models (LLMs) perform rationale generation under the effects of readability level control, i.e., being prompted for an explanation targeting a specific expertise level, such as sixth grade or college. We find that explanations are adaptable to such instruction, though the observed distinction between readability levels does not fully match the defined complexity scores according to traditional readability metrics. Furthermore, the generated rationales tend to feature medium level complexity, which correlates with the measured quality using automatic metrics. Finally, our human annotators confirm a generally satisfactory impression on rationales at all readability levels, with high-school-level readability being most commonly perceived and favored.
>
---
#### [replaced 038] PolyPrompt: Automating Knowledge Extraction from Multilingual Language Models with Dynamic Prompt Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19756v2](http://arxiv.org/pdf/2502.19756v2)**

> **作者:** Nathan Roll
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Large language models (LLMs) showcase increasingly impressive English benchmark scores, however their performance profiles remain inconsistent across multilingual settings. To address this gap, we introduce PolyPrompt, a novel, parameter-efficient framework for enhancing the multilingual capabilities of LLMs. Our method learns a set of trigger tokens for each language through a gradient-based search, identifying the input query's language and selecting the corresponding trigger tokens which are prepended to the prompt during inference. We perform experiments on two ~1 billion parameter models, with evaluations on the global MMLU benchmark across fifteen typologically and resource diverse languages, demonstrating accuracy gains of 3.7%-19.9% compared to naive and translation-pipeline baselines.
>
---
#### [replaced 039] Unnatural Languages Are Not Bugs but Features for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01926v2](http://arxiv.org/pdf/2503.01926v2)**

> **作者:** Keyu Duan; Yiran Zhao; Zhili Feng; Jinjie Ni; Tianyu Pang; Qian Liu; Tianle Cai; Longxu Dou; Kenji Kawaguchi; Anirudh Goyal; J. Zico Kolter; Michael Qizhe Shieh
>
> **摘要:** Large Language Models (LLMs) have been observed to process non-human-readable text sequences, such as jailbreak prompts, often viewed as a bug for aligned LLMs. In this work, we present a systematic investigation challenging this perception, demonstrating that unnatural languages - strings that appear incomprehensible to humans but maintain semantic meanings for LLMs - contain latent features usable by models. Notably, unnatural languages possess latent features that can be generalized across different models and tasks during inference. Furthermore, models fine-tuned on unnatural versions of instruction datasets perform on-par with those trained on natural language, achieving 49.71 win rates in Length-controlled AlpacaEval 2.0 in average across various base models. In addition, through comprehensive analysis, we demonstrate that LLMs process unnatural languages by filtering noise and inferring contextual meaning from filtered words.
>
---
#### [replaced 040] Enhancing Clinical Multiple-Choice Questions Benchmarks with Knowledge Graph Guided Distractor Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00612v2](http://arxiv.org/pdf/2506.00612v2)**

> **作者:** Running Yang; Wenlong Deng; Minghui Chen; Yuyin Zhou; Xiaoxiao Li
>
> **摘要:** Clinical tasks such as diagnosis and treatment require strong decision-making abilities, highlighting the importance of rigorous evaluation benchmarks to assess the reliability of large language models (LLMs). In this work, we introduce a knowledge-guided data augmentation framework that enhances the difficulty of clinical multiple-choice question (MCQ) datasets by generating distractors (i.e., incorrect choices that are similar to the correct one and may confuse existing LLMs). Using our KG-based pipeline, the generated choices are both clinically plausible and deliberately misleading. Our approach involves multi-step, semantically informed walks on a medical knowledge graph to identify distractor paths-associations that are medically relevant but factually incorrect-which then guide the LLM in crafting more deceptive distractors. We apply the designed knowledge graph guided distractor generation (KGGDG) pipline, to six widely used medical QA benchmarks and show that it consistently reduces the accuracy of state-of-the-art LLMs. These findings establish KGGDG as a powerful tool for enabling more robust and diagnostic evaluations of medical LLMs.
>
---
#### [replaced 041] Computational Analysis of Character Development in Holocaust Testimonies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17063v2](http://arxiv.org/pdf/2412.17063v2)**

> **作者:** Esther Shizgal; Eitan Wagner; Renana Keydar; Omri Abend
>
> **摘要:** This work presents a computational approach to analyze character development along the narrative timeline. The analysis characterizes the inner and outer changes the protagonist undergoes within a narrative, and the interplay between them. We consider transcripts of Holocaust survivor testimonies as a test case, each telling the story of an individual in first-person terms. We focus on the survivor's religious trajectory, examining the evolution of their disposition toward religious belief and practice along the testimony. Clustering the resulting trajectories in the dataset, we identify common sequences in the data. Our findings highlight multiple common structures of religiosity across the narratives: in terms of belief, most present a constant disposition, while for practice, most present an oscillating structure, serving as valuable material for historical and sociological research. This work demonstrates the potential of natural language processing techniques for analyzing character evolution through thematic trajectories in narratives.
>
---
#### [replaced 042] DLP: Dynamic Layerwise Pruning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23807v3](http://arxiv.org/pdf/2505.23807v3)**

> **作者:** Yuli Chen; Bo Cheng; Jiale Han; Yingying Zhang; Yingting Li; Shuhao Zhang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Pruning has recently been widely adopted to reduce the parameter scale and improve the inference efficiency of Large Language Models (LLMs). Mainstream pruning techniques often rely on uniform layerwise pruning strategies, which can lead to severe performance degradation at high sparsity levels. Recognizing the varying contributions of different layers in LLMs, recent studies have shifted their focus toward non-uniform layerwise pruning. However, these approaches often rely on pre-defined values, which can result in suboptimal performance. To overcome these limitations, we propose a novel method called Dynamic Layerwise Pruning (DLP). This approach adaptively determines the relative importance of each layer by integrating model weights with input activation information, assigning pruning rates accordingly. Experimental results show that DLP effectively preserves model performance at high sparsity levels across multiple LLMs. Specifically, at 70% sparsity, DLP reduces the perplexity of LLaMA2-7B by 7.79 and improves the average accuracy by 2.7% compared to state-of-the-art methods. Moreover, DLP is compatible with various existing LLM compression techniques and can be seamlessly integrated into Parameter-Efficient Fine-Tuning (PEFT). We release the code at https://github.com/ironartisan/DLP to facilitate future research.
>
---
#### [replaced 043] Large Language Model Evaluation via Matrix Nuclear-Norm
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10672v3](http://arxiv.org/pdf/2410.10672v3)**

> **作者:** Yahan Li; Tingyu Xia; Yi Chang; Yuan Wu
>
> **备注:** 21 pages
>
> **摘要:** As large language models (LLMs) continue to evolve, efficient evaluation metrics are vital for assessing their ability to compress information and reduce redundancy. While traditional metrics like Matrix Entropy offer valuable insights, they are computationally intensive for large-scale models due to their \( O(n^3) \) time complexity with Singular Value Decomposition (SVD). To mitigate this issue, we introduce the Matrix Nuclear-Norm, which not only serves as a metric to quantify the data compression proficiency of LLM but also provides a convex approximation of matrix rank to capture both predictive discriminability and diversity. By employing the \( L_{1,2}\text{-norm} \) to further approximate the nuclear norm, we can effectively assess the model's information compression capabilities. This approach reduces the time complexity to \( O(n^2) \) and eliminates the need for SVD computation. Consequently, the Matrix Nuclear-Norm achieves speeds 8 to 24 times faster than Matrix Entropy for the CEREBRAS-GPT model as sizes increase from 111M to 6.7B. This performance gap becomes more pronounced with larger models, as validated in tests with other models like Pythia. Additionally, evaluations on benchmarks and model responses confirm that our proposed Matrix Nuclear-Norm is a reliable, scalable, and efficient tool for assessing LLMs' performance, striking a balance between accuracy and computational efficiency. The code is available at https://github.com/MLGroupJLU/MatrixNuclearNorm.
>
---
#### [replaced 044] Is it the end of (generative) linguistics as we know it?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12797v2](http://arxiv.org/pdf/2412.12797v2)**

> **作者:** Cristiano Chesi
>
> **摘要:** A significant debate has emerged in response to a paper written by Steven Piantadosi (Piantadosi, 2023) and uploaded to the LingBuzz platform, the open archive for generative linguistics. Piantadosi's dismissal of Chomsky's approach is ruthless, but generative linguists deserve it. In this paper, I will adopt three idealized perspectives -- computational, theoretical, and experimental -- to focus on two fundamental issues that lend partial support to Piantadosi's critique: (a) the evidence challenging the Poverty of Stimulus (PoS) hypothesis and (b) the notion of simplicity as conceived within mainstream Minimalism. In conclusion, I argue that, to reclaim a central role in language studies, generative linguistics -- representing a prototypical theoretical perspective on language -- needs a serious update leading to (i) more precise, consistent, and complete formalizations of foundational intuitions and (ii) the establishment and utilization of a standardized dataset of crucial empirical evidence to evaluate the theory's adequacy. On the other hand, ignoring the formal perspective leads to major drawbacks in both computational and experimental approaches. Neither descriptive nor explanatory adequacy can be easily achieved without the precise formulation of general principles that can be challenged empirically.
>
---
#### [replaced 045] ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents
- **分类: cs.CV; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.18017v2](http://arxiv.org/pdf/2502.18017v2)**

> **作者:** Qiuchen Wang; Ruixue Ding; Zehui Chen; Weiqi Wu; Shihang Wang; Pengjun Xie; Feng Zhao
>
> **摘要:** Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model's reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark. The code is available at https://github.com/Alibaba-NLP/ViDoRAG.
>
---
#### [replaced 046] Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17533v3](http://arxiv.org/pdf/2412.17533v3)**

> **作者:** Anna Kołos; Katarzyna Lorenc; Emilia Wiśnios; Agnieszka Karlińska
>
> **备注:** Accepted for ACL 2025 Main Conference
>
> **摘要:** The surge in online content has created an urgent demand for robust detection systems, especially in non-English contexts where current tools demonstrate significant limitations. We present forePLay, a novel Polish language dataset for erotic content detection, featuring over 24k annotated sentences with a multidimensional taxonomy encompassing ambiguity, violence, and social unacceptability dimensions. Our comprehensive evaluation demonstrates that specialized Polish language models achieve superior performance compared to multilingual alternatives, with transformer-based architectures showing particular strength in handling imbalanced categories. The dataset and accompanying analysis establish essential frameworks for developing linguistically-aware content moderation systems, while highlighting critical considerations for extending such capabilities to morphologically complex languages.
>
---
#### [replaced 047] Diving into Self-Evolving Training for Multimodal Reasoning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.17451v2](http://arxiv.org/pdf/2412.17451v2)**

> **作者:** Wei Liu; Junlong Li; Xiwen Zhang; Fan Zhou; Yu Cheng; Junxian He
>
> **备注:** ICML 2025, Project Page: https://mstar-lmm.github.io
>
> **摘要:** Self-evolving trainin--where models iteratively learn from their own outputs--has emerged as a key approach for complex reasoning tasks, addressing the scarcity of high-quality chain-of-thought data. However, its effectiveness in multimodal reasoning, a domain more intricate than text-only reasoning, remains underexplored, and the understanding of critical factors in this training paradigm remains limited. Furthermore, a central challenge for this training method is performance saturation, which impedes further improvements and scalability. Inspired by reinforcement learning (RL), in this paper, we reframe self-evolving training for multimodal reasoning through the lens of RL, identifying three pivotal factors: Training Method, Reward Model, and Prompt Variation. Through systematic analysis, we establish relatively optimal design principles that significantly enhance multimodal reasoning capabilities. Moreover, delving deeper into training dynamics, we uncover the roots of saturation and propose a new automatic balancing mechanism to mitigate this limitation. Building on these insights, we propose M-STAR (Multimodal Self-evolving Training for Reasoning), a framework that achieves consistent performance gains across models of varying sizes and diverse benchmarks. All resources are made publicly available at https://mstar-lmm.github.io.
>
---
#### [replaced 048] Grounded Persuasive Language Generation for Automated Marketing
- **分类: cs.AI; cs.CL; cs.HC; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2502.16810v2](http://arxiv.org/pdf/2502.16810v2)**

> **作者:** Jibang Wu; Chenghao Yang; Simon Mahns; Chaoqi Wang; Hao Zhu; Fei Fang; Haifeng Xu
>
> **摘要:** This paper develops an agentic framework that employs large language models (LLMs) to automate the generation of persuasive and grounded marketing content, using real estate listing descriptions as our focal application domain. Our method is designed to align the generated content with user preferences while highlighting useful factual attributes. This agent consists of three key modules: (1) Grounding Module, mimicking expert human behavior to predict marketable features; (2) Personalization Module, aligning content with user preferences; (3) Marketing Module, ensuring factual accuracy and the inclusion of localized features. We conduct systematic human-subject experiments in the domain of real estate marketing, with a focus group of potential house buyers. The results demonstrate that marketing descriptions generated by our approach are preferred over those written by human experts by a clear margin while maintaining the same level of factual accuracy. Our findings suggest a promising agentic approach to automate large-scale targeted marketing while ensuring factuality of content generation.
>
---
#### [replaced 049] Threading the Needle: Reweaving Chain-of-Thought Reasoning to Explain Human Label Variation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23368v2](http://arxiv.org/pdf/2505.23368v2)**

> **作者:** Beiduo Chen; Yang Janet Liu; Anna Korhonen; Barbara Plank
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** The recent rise of reasoning-tuned Large Language Models (LLMs)--which generate chains of thought (CoTs) before giving the final answer--has attracted significant attention and offers new opportunities for gaining insights into human label variation, which refers to plausible differences in how multiple annotators label the same data instance. Prior work has shown that LLM-generated explanations can help align model predictions with human label distributions, but typically adopt a reverse paradigm: producing explanations based on given answers. In contrast, CoTs provide a forward reasoning path that may implicitly embed rationales for each answer option, before generating the answers. We thus propose a novel LLM-based pipeline enriched with linguistically-grounded discourse segmenters to extract supporting and opposing statements for each answer option from CoTs with improved accuracy. We also propose a rank-based HLV evaluation framework that prioritizes the ranking of answers over exact scores, which instead favor direct comparison of label distributions. Our method outperforms a direct generation method as well as baselines on three datasets, and shows better alignment of ranking methods with humans, highlighting the effectiveness of our approach.
>
---
#### [replaced 050] GPTVQ: The Blessing of Dimensionality for LLM Quantization
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.15319v2](http://arxiv.org/pdf/2402.15319v2)**

> **作者:** Mart van Baalen; Andrey Kuzmin; Ivan Koryakovskiy; Markus Nagel; Peter Couperus; Cedric Bastoul; Eric Mahurin; Tijmen Blankevoort; Paul Whatmough
>
> **摘要:** In this work we show that the size versus accuracy trade-off of neural network quantization can be significantly improved by increasing the quantization dimensionality. We propose the GPTVQ method, a new fast method for post-training vector quantization (VQ) that scales well to Large Language Models (LLMs). Our method interleaves quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE. Quantization codebooks are initialized using an efficient data-aware version of the EM algorithm. The codebooks are then updated, and further compressed by using integer quantization and SVD-based compression. GPTVQ establishes a new state-of-the art in the size vs accuracy trade-offs on a wide range of LLMs such as Llama-v2 and Mistral. Furthermore, our method is efficient: on a single H100 it takes between 3 and 11 hours to process a Llamav2-70B model, depending on quantization setting. Lastly, with on-device timings for VQ decompression on a mobile CPU we show that VQ leads to improved latency compared to using a 4-bit integer format.
>
---
#### [replaced 051] Enhancing Target-unspecific Tasks through a Features Matrix
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03414v5](http://arxiv.org/pdf/2505.03414v5)**

> **作者:** Fangming Cui; Yonggang Zhang; Xuan Wang; Xinmei Tian; Jun Yu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Recent developments in prompt learning of large Vision-Language Models (VLMs) have significantly improved performance in target-specific tasks. However, these prompting methods often struggle to tackle the target-unspecific or generalizable tasks effectively. It may be attributed to the fact that overfitting training causes the model to forget its general knowledge. The general knowledge has a strong promotion on target-unspecific tasks. To alleviate this issue, we propose a novel Features Matrix (FM) approach designed to enhance these models on target-unspecific tasks. Our method extracts and leverages general knowledge, shaping a Features Matrix (FM). Specifically, the FM captures the semantics of diverse inputs from a deep and fine perspective, preserving essential general knowledge, which mitigates the risk of overfitting. Representative evaluations demonstrate that: 1) the FM is compatible with existing frameworks as a generic and flexible module, and 2) the FM significantly showcases its effectiveness in enhancing target-unspecific tasks (base-to-novel generalization, domain generalization, and cross-dataset generalization), achieving state-of-the-art performance.
>
---
#### [replaced 052] A Hitchhiker's Guide to Scaling Law Estimation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.11840v2](http://arxiv.org/pdf/2410.11840v2)**

> **作者:** Leshem Choshen; Yang Zhang; Jacob Andreas
>
> **备注:** ICML
>
> **摘要:** Scaling laws predict the loss of a target machine learning model by extrapolating from easier-to-train models with fewer parameters or smaller training sets. This provides an efficient way for practitioners and researchers alike to compare pretraining decisions involving optimizers, datasets, and model architectures. Despite the widespread use of scaling laws to model the dynamics of language model training, there has been little work on understanding how to best estimate and interpret them. We collect (and release) a large-scale dataset containing losses and downstream evaluations for 485 previously published pretrained models. We use these to estimate more than 1000 scaling laws, then derive a set of best practices for estimating scaling laws in new model families. We find that fitting scaling laws to intermediate checkpoints of training runs (and not just their final losses) substantially improves accuracy, and that -- all else equal -- estimates of performance are generally most accurate when derived from other models of similar sizes. However, because there is a significant degree of variability across model seeds, training multiple small models is sometimes more useful than training a single large one. Moreover, while different model families differ scaling behavior, they are often similar enough that a target model's behavior can be predicted from a single model with the same architecture, along with scaling parameter estimates derived from other model families.
>
---
#### [replaced 053] Lower Layers Matter: Alleviating Hallucination via Multi-Layer Fusion Contrastive Decoding with Truthfulness Refocused
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08769v2](http://arxiv.org/pdf/2408.08769v2)**

> **作者:** Dingwei Chen; Feiteng Fang; Shiwen Ni; Feng Liang; Xiping Hu; Ahmadreza Argha; Hamid Alinejad-Rokny; Min Yang; Chengming Li
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional performance across various natural language processing tasks. However, they occasionally generate inaccurate and counterfactual outputs, a phenomenon commonly referred to as "hallucinations''. To tackle this issue, recent studies have explored contrastive decoding between the original model and an amateur model with induced hallucination, showing promising results. Nevertheless, this approach can disrupt the original LLM's output distribution due to coarse contrast and simple subtraction operations, potentially leading to errors. In this paper, we introduce a novel contrastive decoding framework, termed LOL (LOwer Layer Matters). Unlike prior methods that focus solely on the final layer, our approach integrates contrastive information from lower layers to enable multi-layer fusion during contrastive decoding. Additionally, we incorporate a truthfulness refocused module that leverages instruction guidance to further improve truthfulness in contrastive decoding. Extensive experiments on four publicly available datasets demonstrate that the LOL framework significantly mitigates hallucination while outperforming existing baselines in most cases. For reproducibility, we will release our code and data upon acceptance.
>
---
#### [replaced 054] R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24133v2](http://arxiv.org/pdf/2505.24133v2)**

> **作者:** Zefan Cai; Wen Xiao; Hanshi Sun; Cheng Luo; Yikai Zhang; Ke Wan; Yucheng Li; Yeyang Zhou; Li-Wen Chang; Jiuxiang Gu; Zhen Dong; Anima Anandkumar; Abedelkadir Asi; Junjie Hu
>
> **摘要:** Reasoning models have demonstrated impressive performance in self-reflection and chain-of-thought reasoning. However, they often produce excessively long outputs, leading to prohibitively large key-value (KV) caches during inference. While chain-of-thought inference significantly improves performance on complex reasoning tasks, it can also lead to reasoning failures when deployed with existing KV cache compression approaches. To address this, we propose Redundancy-aware KV Cache Compression for Reasoning models (R-KV), a novel method specifically targeting redundant tokens in reasoning models. Our method preserves nearly 100% of the full KV cache performance using only 10% of the KV cache, substantially outperforming existing KV cache baselines, which reach only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache. This KV-cache reduction also leads to a 90% memory saving and a 6.6X throughput over standard chain-of-thought reasoning inference. Experimental results show that R-KV consistently outperforms existing KV cache compression baselines across two mathematical reasoning datasets.
>
---
#### [replaced 055] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02677v2](http://arxiv.org/pdf/2410.02677v2)**

> **作者:** Yu Ying Chiu; Liwei Jiang; Bill Yuchen Lin; Chan Young Park; Shuyue Stella Li; Sahithya Ravi; Mehar Bhatia; Maria Antoniak; Yulia Tsvetkov; Vered Shwartz; Yejin Choi
>
> **备注:** ACL 2025 Main, 39 pages, 16 figures. arXiv admin note: text overlap with arXiv:2404.06664
>
> **摘要:** Robust, diverse, and challenging cultural knowledge benchmarks are essential for measuring our progress towards making LMs that are helpful across diverse cultures. We introduce CulturalBench: a set of 1,696 human-written and human-verified questions to assess LMs' cultural knowledge, covering 45 global regions including underrepresented ones like Bangladesh, Zimbabwe, and Peru. Questions are each verified by five independent annotators and span 17 diverse topics ranging from food preferences to greeting etiquette. We construct CulturalBench using methods inspired by Human-AI Red-Teaming. Compared to human performance (92.4% accuracy), the hard version of CulturalBench is challenging even for the best-performing frontier LMs, ranging from 28.7% to 61.5% in accuracy. We find that LMs often struggle with tricky questions that have multiple correct answers (e.g., What utensils do the Chinese usually use?), revealing a tendency to overfit to a single answer. Our results indicate that GPT-4o substantially outperform other models across cultures, besting local providers (e.g., Mistral on European culture and DeepSeek on Chinese culture). Across the board, models under-perform on questions related to North Africa, South America and Middle East.
>
---
#### [replaced 056] A Survey on Employing Large Language Models for Text-to-SQL Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.15186v5](http://arxiv.org/pdf/2407.15186v5)**

> **作者:** Liang Shi; Zhengju Tang; Nan Zhang; Xiaotong Zhang; Zhi Yang
>
> **备注:** Accepted by ACM Computing Surveys (CSUR)
>
> **摘要:** With the development of the Large Language Models (LLMs), a large range of LLM-based Text-to-SQL(Text2SQL) methods have emerged. This survey provides a comprehensive review of LLM-based Text2SQL studies. We first enumerate classic benchmarks and evaluation metrics. For the two mainstream methods, prompt engineering and finetuning, we introduce a comprehensive taxonomy and offer practical insights into each subcategory. We present an overall analysis of the above methods and various models evaluated on well-known datasets and extract some characteristics. Finally, we discuss the challenges and future directions in this field.
>
---
#### [replaced 057] Dataset Cartography for Large Language Model Alignment: Mapping and Diagnosing Preference Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23114v2](http://arxiv.org/pdf/2505.23114v2)**

> **作者:** Seohyeong Lee; Eunwon Kim; Hwaran Lee; Buru Chang
>
> **摘要:** Human preference data plays a critical role in aligning large language models (LLMs) with human values. However, collecting such data is often expensive and inefficient, posing a significant scalability challenge. To address this, we introduce Alignment Data Map, a GPT-4o-assisted tool for analyzing and diagnosing preference data. Using GPT-4o as a proxy for LLM alignment, we compute alignment scores for LLM-generated responses to instructions from existing preference datasets. These scores are then used to construct an Alignment Data Map based on their mean and variance. Our experiments show that using only 33 percent of the data, specifically samples in the high-mean, low-variance region, achieves performance comparable to or better than using the entire dataset. This finding suggests that the Alignment Data Map can significantly improve data collection efficiency by identifying high-quality samples for LLM alignment without requiring explicit annotations. Moreover, the Alignment Data Map can diagnose existing preference datasets. Our analysis shows that it effectively detects low-impact or potentially misannotated samples. Source code is available online.
>
---
#### [replaced 058] Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.05050v4](http://arxiv.org/pdf/2504.05050v4)**

> **作者:** Jiawei Lian; Jianhong Pan; Lefan Wang; Yi Wang; Shaohui Mei; Lap-Pui Chau
>
> **摘要:** Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.
>
---
#### [replaced 059] Evaluating and Advancing Multimodal Large Language Models in Perception Ability Lens
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.14725v2](http://arxiv.org/pdf/2411.14725v2)**

> **作者:** Feng Chen; Chenhui Gou; Jing Liu; Yang Yang; Zhaoyang Li; Jiyuan Zhang; Zhenbang Sun; Bohan Zhuang; Qi Wu
>
> **备注:** Code repository: https://github.com/Chenfeng1271/AbilityLens/tree/main
>
> **摘要:** As multimodal large language models (MLLMs) advance rapidly, rigorous evaluation has become essential, providing further guidance for their development. In this work, we focus on a unified and robust evaluation of \textbf{vision perception} abilities, the foundational skill of MLLMs. We find that existing perception benchmarks, each focusing on different question types, domains, and evaluation metrics, introduce significant evaluation variance, complicating comprehensive assessments of perception abilities when relying on any single benchmark. To address this, we introduce \textbf{AbilityLens}, a unified benchmark designed to evaluate MLLMs in six key perception abilities (ranging from counting, OCR, to understanding structural data), focusing on both accuracy and stability, with each ability encompassing diverse types of questions, domains, and metrics. With the assistance of AbilityLens, we: (1) identify the strengths and weaknesses of current main-stream MLLMs, highlighting stability patterns and revealing a notable performance gap between state-of-the-art open-source and closed-source models; (2) uncover interesting ability conflict and early convergence phenomena during MLLM training; (3) reveal the primary reason of ability conflict is data mixing ratio and LLM model size; and (4) discuss the effectiveness of some straightforward strategies \eg, fine-tuning and model merging, to solve the ability conflict. The benchmark and online leaderboard is released in https://github.com/Chenfeng1271/AbilityLens.
>
---
#### [replaced 060] UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.00334v4](http://arxiv.org/pdf/2502.00334v4)**

> **作者:** Xin Xu; Qiyun Xu; Tong Xiao; Tianhao Chen; Yuchen Yan; Jiaxin Zhang; Shizhe Diao; Can Yang; Yang Wang
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in solving complex reasoning tasks, particularly in mathematics. However, the domain of physics reasoning presents unique challenges that have received significantly less attention. Existing benchmarks often fall short in evaluating LLMs' abilities on the breadth and depth of undergraduate-level physics, underscoring the need for a comprehensive evaluation. To fill this gap, we introduce UGPhysics, a large-scale and comprehensive benchmark specifically designed to evaluate UnderGraduate-level Physics (UGPhysics) reasoning with LLMs. UGPhysics includes 5,520 undergraduate-level physics problems in both English and Chinese, covering 13 subjects with seven different answer types and four distinct physics reasoning skills, all rigorously screened for data leakage. Additionally, we develop a Model-Assistant Rule-based Judgment (MARJ) pipeline specifically tailored for assessing answer correctness of physics problems, ensuring accurate evaluation. Our evaluation of 31 leading LLMs shows that the highest overall accuracy, 49.8% (achieved by OpenAI-o1-mini), emphasizes the necessity for models with stronger physics reasoning skills, beyond math abilities. We hope UGPhysics, along with MARJ, will drive future advancements in AI for physics reasoning. Codes and data are available at https://github.com/YangLabHKUST/UGPhysics .
>
---
#### [replaced 061] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought
- **分类: cs.CL; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.17214v2](http://arxiv.org/pdf/2502.17214v2)**

> **作者:** Boxuan Zhang; Ruqi Zhang
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) excel in many tasks but struggle to accurately quantify uncertainty in their generated responses. This limitation makes it challenging to detect misinformation and ensure reliable decision-making. Existing uncertainty quantification (UQ) methods for LLMs are primarily prompt-wise rather than response-wise, often requiring multiple response samples, which incurs high computational costs. Moreover, LLMs have been shown to be overconfident, particularly when using reasoning steps to derive their answers. In this work, we propose CoT-UQ, a response-wise UQ framework that integrates LLMs' inherent reasoning capabilities through Chain-of-Thought (CoT) into the UQ process. CoT-UQ captures critical information during inference by extracting keywords from each reasoning step and assessing their importance to the final answer. This key reasoning information is then aggregated to produce a final uncertainty estimate. We conduct extensive experiments based on Llama Family with model sizes varying from 8B to 13B across logical and mathematical reasoning tasks. Experimental results demonstrate that CoT-UQ significantly outperforms existing UQ methods, achieving an average improvement of 5.9% AUROC compared to current UQ methods. The code is available at: https://github.com/ZBox1005/CoT-UQ.
>
---
#### [replaced 062] Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17110v3](http://arxiv.org/pdf/2502.17110v3)**

> **作者:** Junyang Wang; Haiyang Xu; Xi Zhang; Ming Yan; Ji Zhang; Fei Huang; Jitao Sang
>
> **备注:** 17 pages, 7 figures, 9 tables
>
> **摘要:** The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation.
>
---
#### [replaced 063] SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15289v3](http://arxiv.org/pdf/2412.15289v3)**

> **作者:** Xiaoning Dong; Wenbo Hu; Wei Xu; Tianxing He
>
> **摘要:** Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
>
---
#### [replaced 064] How to Connect Speech Foundation Models and Large Language Models? What Matters and What Does Not
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.17044v3](http://arxiv.org/pdf/2409.17044v3)**

> **作者:** Francesco Verdini; Pierfrancesco Melucci; Stefano Perna; Francesco Cariaggi; Marco Gaido; Sara Papi; Szymon Mazurek; Marek Kasztelnik; Luisa Bentivogli; Sébastien Bratières; Paolo Merialdo; Simone Scardapane
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** The remarkable performance achieved by Large Language Models (LLM) has driven research efforts to leverage them for a wide range of tasks and input modalities. In speech-to-text (S2T) tasks, the emerging solution consists of projecting the output of the encoder of a Speech Foundational Model (SFM) into the LLM embedding space through an adapter module. However, no work has yet investigated how much the downstream-task performance depends on each component (SFM, adapter, LLM) nor whether the best design of the adapter depends on the chosen SFM and LLM. To fill this gap, we evaluate the combination of 5 adapter modules, 2 LLMs (Mistral and Llama), and 2 SFMs (Whisper and SeamlessM4T) on two widespread S2T tasks, namely Automatic Speech Recognition and Speech Translation. Our results demonstrate that the SFM plays a pivotal role in downstream performance, while the adapter choice has moderate impact and depends on the SFM and LLM.
>
---
#### [replaced 065] A Fully Automated Pipeline for Conversational Discourse Annotation: Tree Scheme Generation and Labeling with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.08961v2](http://arxiv.org/pdf/2504.08961v2)**

> **作者:** Kseniia Petukhova; Ekaterina Kochmar
>
> **摘要:** Recent advances in Large Language Models (LLMs) have shown promise in automating discourse annotation for conversations. While manually designing tree annotation schemes significantly improves annotation quality for humans and models, their creation remains time-consuming and requires expert knowledge. We propose a fully automated pipeline that uses LLMs to construct such schemes and perform annotation. We evaluate our approach on speech functions (SFs) and the Switchboard-DAMSL (SWBD-DAMSL) taxonomies. Our experiments compare various design choices, and we show that frequency-guided decision trees, paired with an advanced LLM for annotation, can outperform previously manually designed trees and even match or surpass human annotators while significantly reducing the time required for annotation. We release all code and resultant schemes and annotations to facilitate future research on discourse annotation.
>
---
#### [replaced 066] SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.04975v2](http://arxiv.org/pdf/2411.04975v2)**

> **作者:** Gabriele Oliaro; Zhihao Jia; Daniel Campos; Aurick Qiao
>
> **摘要:** Speculative decoding is widely adopted to reduce latency in large language model (LLM) inference by leveraging smaller draft models capable of handling diverse user tasks. However, emerging AI applications, such as LLM-based agents, present unique workload characteristics: instead of diverse independent requests, agentic frameworks typically submit repetitive inference requests, such as multi-agent pipelines performing similar subtasks or self-refinement loops iteratively enhancing outputs. These workloads result in long and highly predictable sequences, which current speculative decoding methods do not effectively exploit. To address this gap, we introduce \emph{SuffixDecoding}, a novel method that utilizes efficient suffix trees to cache long token sequences from prompts and previous outputs. By adaptively speculating more tokens when acceptance likelihood is high and fewer when it is low, SuffixDecoding effectively exploits opportunities for longer speculations while conserving computation when those opportunities are limited. Evaluations on agentic benchmarks, including SWE-Bench and Text-to-SQL, demonstrate that SuffixDecoding achieves speedups of up to 5.3$\times$, outperforming state-of-the-art methods -- 2.8$\times$ faster than model-based approaches like EAGLE-2/3 and 1.9$\times$ faster than model-free approaches such as Token Recycling. SuffixDecoding is open-sourced at https://github.com/snowflakedb/ArcticInference.
>
---
#### [replaced 067] Generator-Assistant Stepwise Rollback Framework for Large Language Model Agent
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02519v3](http://arxiv.org/pdf/2503.02519v3)**

> **作者:** Xingzuo Li; Kehai Chen; Yunfei Long; Xuefeng Bai; Yong Xu; Min Zhang
>
> **摘要:** Large language model (LLM) agents typically adopt a step-by-step reasoning framework, in which they interleave the processes of thinking and acting to accomplish the given task. However, this paradigm faces a deep-rooted one-pass issue whereby each generated intermediate thought is plugged into the trajectory regardless of its correctness, which can cause irreversible error propagation. To address the issue, this paper proposes a novel framework called Generator-Assistant Stepwise Rollback (GA-Rollback) to induce better decision-making for LLM agents. Particularly, GA-Rollback utilizes a generator to interact with the environment and an assistant to examine each action produced by the generator, where the assistant triggers a rollback operation upon detection of incorrect actions. Moreover, we introduce two additional strategies tailored for the rollback scenario to further improve its effectiveness. Extensive experiments show that GA-Rollback achieves significant improvements over several strong baselines on three widely used benchmarks. Our analysis further reveals that GA-Rollback can function as a robust plug-and-play module, integrating seamlessly with other methods.
>
---
#### [replaced 068] Scaling Physical Reasoning with the PHYSICS Dataset
- **分类: cs.CL; cs.LG; physics.ed-ph**

- **链接: [http://arxiv.org/pdf/2506.00022v2](http://arxiv.org/pdf/2506.00022v2)**

> **作者:** Shenghe Zheng; Qianjia Cheng; Junchi Yao; Mengsong Wu; Haonan He; Ning Ding; Yu Cheng; Shuyue Hu; Lei Bai; Dongzhan Zhou; Ganqu Cui; Peng Ye
>
> **备注:** Work on physical datasets
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable progress on advanced reasoning tasks such as mathematics and coding competitions. Meanwhile, physics, despite being both reasoning-intensive and essential to real-world understanding, received limited academic and industrial attention. This paper introduces PHYSICS, a dataset containing 16,568 high-quality physics problems spanning subjects and difficulty levels, to facilitate this issue. Specifically, PHYSICS is curated with exercises from over 100 textbooks through a carefully designed pipeline for quality control. It covers five major physics domains: Mechanics, Electromagnetism, Thermodynamics, Optics, and Modern Physics. It also spans a wide range of difficulty levels, from high school to graduate-level physics courses. To utilize the data for improving and evaluating the model's physical reasoning capabilities, we split the dataset into training and test sets, and provide reasoning paths generated by powerful reasoning models for the training data to facilitate model training. In addition, for the evaluation part, we find that existing evaluation frameworks exhibit biases in aspects such as units, simplification, and precision in physics domain. To balance efficiency and accuracy, we introduce a Rule+Model evaluation framework tailored to physics problems. Our evaluations on current state-of-the-art open-source and proprietary models highlight the limitations of current models in handling physics-related tasks. We hope that our dataset and evaluation methodology will jointly advance the development of LLMs in the field of physics.
>
---
#### [replaced 069] DRAMA: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.18460v2](http://arxiv.org/pdf/2502.18460v2)**

> **作者:** Xueguang Ma; Xi Victoria Lin; Barlas Oguz; Jimmy Lin; Wen-tau Yih; Xilun Chen
>
> **备注:** ACL 2025
>
> **摘要:** Large language models (LLMs) have demonstrated strong effectiveness and robustness while fine-tuned as dense retrievers. However, their large parameter size brings significant inference time computational challenges, including high encoding costs for large-scale corpora and increased query latency, limiting their practical deployment. While smaller retrievers offer better efficiency, they often fail to generalize effectively with limited supervised fine-tuning data. In this work, we introduce DRAMA, a training framework that leverages LLMs to train smaller generalizable dense retrievers. In particular, we adopt pruned LLMs as the backbone and train on diverse LLM-augmented data in a single-stage contrastive learning setup. Experiments show that DRAMA offers better multilingual and long-context capabilities than traditional encoder-based retrievers, and achieves strong performance across multiple tasks and languages. These highlight the potential of connecting the training of smaller retrievers with the growing advancements in LLMs, bridging the gap between efficiency and generalization.
>
---
#### [replaced 070] TransAug: Translate as Augmentation for Sentence Embeddings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2111.00157v3](http://arxiv.org/pdf/2111.00157v3)**

> **作者:** Jue Wang
>
> **摘要:** While contrastive learning greatly advances the representation of sentence embeddings, it is still limited by the size of the existing sentence datasets. In this paper, we present TransAug (Translate as Augmentation), which provide the first exploration of utilizing translated sentence pairs as data augmentation for text, and introduce a two-stage paradigm to advances the state-of-the-art sentence embeddings. Instead of adopting an encoder trained in other languages setting, we first distill a Chinese encoder from a SimCSE encoder (pretrained in English), so that their embeddings are close in semantic space, which can be regraded as implicit data augmentation. Then, we only update the English encoder via cross-lingual contrastive learning and frozen the distilled Chinese encoder. Our approach achieves a new state-of-art on standard semantic textual similarity (STS), outperforming both SimCSE and Sentence-T5, and the best performance in corresponding tracks on transfer tasks evaluated by SentEval.
>
---
#### [replaced 071] Large Language Models to Diffusion Finetuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15781v2](http://arxiv.org/pdf/2501.15781v2)**

> **作者:** Edoardo Cetin; Tianyu Zhao; Yujin Tang
>
> **备注:** Camera-ready version, presented at ICML 2025. Code available at: https://github.com/SakanaAI/L2D
>
> **摘要:** We propose a new finetuning method to provide pre-trained large language models (LMs) the ability to scale test-time compute through the diffusion framework. By increasing the number of diffusion steps, we show our finetuned models achieve monotonically increasing accuracy, directly translating to improved performance across downstream tasks. Furthermore, our finetuned models can expertly answer questions on specific topics by integrating powerful guidance techniques, and autonomously determine the compute required for a given problem by leveraging adaptive ODE solvers. Our method is universally applicable to any foundation model pre-trained with a cross-entropy loss and does not modify any of its original weights, fully preserving its strong single-step generation capabilities. We show our method is more effective and fully compatible with traditional finetuning approaches, introducing an orthogonal new direction to unify the strengths of the autoregressive and diffusion frameworks.
>
---
#### [replaced 072] Localizing Persona Representations in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24539v2](http://arxiv.org/pdf/2505.24539v2)**

> **作者:** Celia Cintas; Miriam Rateike; Erik Miehling; Elizabeth Daly; Skyler Speakman
>
> **摘要:** We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
>
---
#### [replaced 073] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.15462v3](http://arxiv.org/pdf/2411.15462v3)**

> **作者:** Manuel Tonneau; Diyi Liu; Niyati Malhotra; Scott A. Hale; Samuel P. Fraiberger; Victor Orozco-Olvera; Paul Röttger
>
> **备注:** ACL 2025 main conference. Data available at https://huggingface.co/datasets/manueltonneau/hateday
>
> **摘要:** To address the global challenge of online hate speech, prior research has developed detection models to flag such content on social media. However, due to systematic biases in evaluation datasets, the real-world effectiveness of these models remains unclear, particularly across geographies. We introduce HateDay, the first global hate speech dataset representative of social media settings, constructed from a random sample of all tweets posted on September 21, 2022 and covering eight languages and four English-speaking countries. Using HateDay, we uncover substantial variation in the prevalence and composition of hate speech across languages and regions. We show that evaluations on academic datasets greatly overestimate real-world detection performance, which we find is very low, especially for non-European languages. Our analysis identifies key drivers of this gap, including models' difficulty to distinguish hate from offensive speech and a mismatch between the target groups emphasized in academic datasets and those most frequently targeted in real-world settings. We argue that poor model performance makes public models ill-suited for automatic hate speech moderation and find that high moderation rates are only achievable with substantial human oversight. Our results underscore the need to evaluate detection systems on data that reflects the complexity and diversity of real-world social media.
>
---
#### [replaced 074] Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11075v2](http://arxiv.org/pdf/2502.11075v2)**

> **作者:** Haoyang Li; Xuejia Chen; Zhanchao XU; Darian Li; Nicole Hu; Fei Teng; Yiming Li; Luyu Qiu; Chen Jason Zhang; Qing Li; Lei Chen
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in natural language processing tasks, such as text generation and semantic understanding. However, their performance on numerical reasoning tasks, such as basic arithmetic, numerical retrieval, and magnitude comparison, remains surprisingly poor. This gap arises from their reliance on surface-level statistical patterns rather than understanding numbers as continuous magnitudes. Existing benchmarks primarily focus on either linguistic competence or structured mathematical problem-solving, neglecting fundamental numerical reasoning required in real-world scenarios. To bridge this gap, we propose NumericBench, a comprehensive benchmark to evaluate six fundamental numerical capabilities: number recognition, arithmetic operations, contextual retrieval, comparison, summary, and logical reasoning. NumericBench includes datasets ranging from synthetic number lists to the crawled real-world data, addressing challenges like long contexts, noise, and multi-step reasoning. Extensive experiments on state-of-the-art LLMs, including GPT-4 and DeepSeek, reveal persistent weaknesses in numerical reasoning, highlighting the urgent need to improve numerically-aware language modeling. The benchmark is released in: https://github.com/TreeAI-Lab/NumericBench.
>
---
#### [replaced 075] CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00519v2](http://arxiv.org/pdf/2506.00519v2)**

> **作者:** Yuxi Sun; Aoqi Zuo; Wei Gao; Jing Ma
>
> **备注:** Accepted to Association for Computational Linguistics Findings (ACL) 2025
>
> **摘要:** Large Language Models (LLMs) often exhibit knowledge disparities across languages. Encouraging LLMs to \textit{abstain} when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce \textit{CausalAbstain}, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that \textit{CausalAbstain} effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (\textsc{Casual-native}) and multilingual (\textsc{Causal-multi}) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks. Our code and data are open-sourced at https://github.com/peachch/CausalAbstain.
>
---
#### [replaced 076] Akan Cinematic Emotions (ACE): A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10973v3](http://arxiv.org/pdf/2502.10973v3)**

> **作者:** David Sasu; Zehui Wu; Ziwei Gong; Run Chen; Pengyuan Shi; Lin Ai; Julia Hirschberg; Natalie Schluter
>
> **备注:** Accepted to Findings at ACL 2025
>
> **摘要:** In this paper, we introduce the Akan Conversation Emotion (ACE) dataset, the first multimodal emotion dialogue dataset for an African language, addressing the significant lack of resources for low-resource languages in emotion recognition research. ACE, developed for the Akan language, contains 385 emotion-labeled dialogues and 6,162 utterances across audio, visual, and textual modalities, along with word-level prosodic prominence annotations. The presence of prosodic labels in this dataset also makes it the first prosodically annotated African language dataset. We demonstrate the quality and utility of ACE through experiments using state-of-the-art emotion recognition methods, establishing solid baselines for future research. We hope ACE inspires further work on inclusive, linguistically and culturally diverse NLP resources.
>
---
#### [replaced 077] VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22019v2](http://arxiv.org/pdf/2505.22019v2)**

> **作者:** Qiuchen Wang; Ruixue Ding; Yu Zeng; Zehui Chen; Lin Chen; Shihang Wang; Pengjun Xie; Fei Huang; Feng Zhao
>
> **摘要:** Effectively retrieving, reasoning and understanding visually rich information remains a challenge for RAG methods. Traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As RL has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing continual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incorporate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search engines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse-to-fine perspective. Furthermore, to bridge the gap between users' original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewriting and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. The code is available at https://github.com/Alibaba-NLP/VRAG.
>
---
#### [replaced 078] Generative Emotion Cause Explanation in Multimodal Conversations
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02430v2](http://arxiv.org/pdf/2411.02430v2)**

> **作者:** Lin Wang; Xiaocui Yang; Shi Feng; Daling Wang; Yifei Zhang; Zhitao Zhang
>
> **摘要:** Multimodal conversation, a crucial form of human communication, carries rich emotional content, making the exploration of the causes of emotions within it a research endeavor of significant importance. However, existing research on the causes of emotions typically employs an utterance selection method within a single textual modality to locate causal utterances. This approach remains limited to coarse-grained assessments, lacks nuanced explanations of emotional causation, and demonstrates inadequate capability in identifying multimodal emotional triggers. Therefore, we introduce a task-\textbf{Multimodal Emotion Cause Explanation in Conversation (MECEC)}. This task aims to generate a summary based on the multimodal context of conversations, clearly and intuitively describing the reasons that trigger a given emotion. To adapt to this task, we develop a new dataset (ECEM) based on the MELD dataset. ECEM combines video clips with detailed explanations of character emotions, helping to explore the causal factors behind emotional expression in multimodal conversations. A novel approach, FAME-Net, is further proposed, that harnesses the power of Large Language Models (LLMs) to analyze visual data and accurately interpret the emotions conveyed through facial expressions in videos. By exploiting the contagion effect of facial emotions, FAME-Net effectively captures the emotional causes of individuals engaged in conversations. Our experimental results on the newly constructed dataset show that FAME-Net outperforms several excellent baselines. Code and dataset are available at https://github.com/3222345200/FAME-Net.
>
---
#### [replaced 079] MCU: An Evaluation Framework for Open-Ended Game Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.08367v4](http://arxiv.org/pdf/2310.08367v4)**

> **作者:** Xinyue Zheng; Haowei Lin; Kaichen He; Zihao Wang; Zilong Zheng; Yitao Liang
>
> **摘要:** Developing AI agents capable of interacting with open-world environments to solve diverse tasks is a compelling challenge. However, evaluating such open-ended agents remains difficult, with current benchmarks facing scalability limitations. To address this, we introduce Minecraft Universe (MCU), a comprehensive evaluation framework set within the open-world video game Minecraft. MCU incorporates three key components: (1) an expanding collection of 3,452 composable atomic tasks that encompasses 11 major categories and 41 subcategories of challenges; (2) a task composition mechanism capable of generating infinite diverse tasks with varying difficulty; and (3) a general evaluation framework that achieves 91.5\% alignment with human ratings for open-ended task assessment. Empirical results reveal that even state-of-the-art foundation agents struggle with the increasing diversity and complexity of tasks. These findings highlight the necessity of MCU as a robust benchmark to drive progress in AI agent development within open-ended environments. Our evaluation code and scripts are available at https://github.com/CraftJarvis/MCU.
>
---
#### [replaced 080] Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16552v4](http://arxiv.org/pdf/2505.16552v4)**

> **作者:** Wenhui Tan; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Ruihua Song
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
>
---
#### [replaced 081] BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12974v3](http://arxiv.org/pdf/2410.12974v3)**

> **作者:** Anna Sokol; Elizabeth Daly; Michael Hind; David Piorkowski; Xiangliang Zhang; Nuno Moniz; Nitesh Chawla
>
> **摘要:** Large language models (LLMs) are powerful tools capable of handling diverse tasks. Comparing and selecting appropriate LLMs for specific tasks requires systematic evaluation methods, as models exhibit varying capabilities across different domains. However, finding suitable benchmarks is difficult given the many available options. This complexity not only increases the risk of benchmark misuse and misinterpretation but also demands substantial effort from LLM users, seeking the most suitable benchmarks for their specific needs. To address these issues, we introduce \texttt{BenchmarkCards}, an intuitive and validated documentation framework that standardizes critical benchmark attributes such as objectives, methodologies, data sources, and limitations. Through user studies involving benchmark creators and users, we show that \texttt{BenchmarkCards} can simplify benchmark selection and enhance transparency, facilitating informed decision-making in evaluating LLMs. Data & Code: https://github.com/SokolAnn/BenchmarkCards
>
---
#### [replaced 082] Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments
- **分类: cs.CL; cs.AI; cs.LG; 68T50**

- **链接: [http://arxiv.org/pdf/2506.00694v2](http://arxiv.org/pdf/2506.00694v2)**

> **作者:** Li Zhang; Morgan Gray; Jaromir Savelka; Kevin D. Ashley
>
> **备注:** 11 pages, 7th Workshop on Automated Semantic Analysis of Information in Legal Text @ ICAIL 2025, 16 June 2025, Chicago, IL
>
> **摘要:** Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Link: https://lizhang-aiandlaw.github.io/An-Automated-Pipeline-for-Evaluating-LLM-Generated-3-ply-Case-Based-Legal-Arguments/
>
---
#### [replaced 083] Pi-SQL: Enhancing Text-to-SQL with Fine-Grained Guidance from Pivot Programming Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00912v2](http://arxiv.org/pdf/2506.00912v2)**

> **作者:** Yongdong chi; Hanqing Wang; Zonghan Yang; Jian Yang; Xiao Yan; Yun Chen; Guanhua Chen
>
> **摘要:** Text-to-SQL transforms the user queries from natural language to executable SQL programs, enabling non-experts to interact with complex databases. Existing prompt-based methods craft meticulous text guidelines and examples to facilitate SQL generation, but their accuracy is hindered by the large semantic gap between the texts and the low-resource SQL programs. In this work, we propose Pi-SQL, which incorporates the high-resource Python program as a pivot to bridge between the natural language query and SQL program. In particular, Pi-SQL first generates Python programs that provide fine-grained step-by-step guidelines in their code blocks or comments, and then produces an SQL program following the guidance of each Python program. The final SQL program matches the reference Python program's query results and, through selection from candidates generated by different strategies, achieves superior execution speed, with a reward-based valid efficiency score up to 4.55 higher than the best-performing baseline. Extensive experiments demonstrate the effectiveness of Pi-SQL, which improves the execution accuracy of the best-performing baseline by up to 3.20.
>
---
#### [replaced 084] Recall with Reasoning: Chain-of-Thought Distillation for Mamba's Long-Context Memory and Extrapolation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03320v2](http://arxiv.org/pdf/2505.03320v2)**

> **作者:** Junyu Ma; Tianqing Fang; Zhisong Zhang; Hongming Zhang; Haitao Mi; Dong Yu
>
> **摘要:** Mamba's theoretical infinite-context potential is limited in practice when sequences far exceed training lengths. This work explores unlocking Mamba's long-context memory ability by a simple-yet-effective method, Recall with Reasoning (RwR), by distilling chain-of-thought (CoT) summarization from a teacher model. Specifically, RwR prepends these summarization as CoT prompts during fine-tuning, teaching Mamba to actively recall and reason over long contexts. Experiments on LONGMEMEVAL and HELMET show RwR boosts Mamba's long-context performance against comparable Transformer/hybrid baselines under similar pretraining conditions, while preserving short-context capabilities, all without architectural changes.
>
---
#### [replaced 085] Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.11184v2](http://arxiv.org/pdf/2502.11184v2)**

> **作者:** Wenxuan Wang; Xiaoyuan Liu; Kuiyi Gao; Jen-tse Huang; Youliang Yuan; Pinjia He; Shuai Wang; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have expanded the capabilities of traditional language models by enabling interaction through both text and images. However, ensuring the safety of these models remains a significant challenge, particularly in accurately identifying whether multimodal content is safe or unsafe-a capability we term safety awareness. In this paper, we introduce MMSafeAware, the first comprehensive multimodal safety awareness benchmark designed to evaluate MLLMs across 29 safety scenarios with 1500 carefully curated image-prompt pairs. MMSafeAware includes both unsafe and over-safety subsets to assess models abilities to correctly identify unsafe content and avoid over-sensitivity that can hinder helpfulness. Evaluating nine widely used MLLMs using MMSafeAware reveals that current models are not sufficiently safe and often overly sensitive; for example, GPT-4V misclassifies 36.1% of unsafe inputs as safe and 59.9% of benign inputs as unsafe. We further explore three methods to improve safety awareness-prompting-based approaches, visual contrastive decoding, and vision-centric reasoning fine-tuning-but find that none achieve satisfactory performance. Our findings highlight the profound challenges in developing MLLMs with robust safety awareness, underscoring the need for further research in this area. All the code and data will be publicly available to facilitate future research.
>
---
#### [replaced 086] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.13649v3](http://arxiv.org/pdf/2412.13649v3)**

> **作者:** Jialong Wu; Zhenglin Wang; Linhai Zhang; Yilong Lai; Yulan He; Deyu Zhou
>
> **备注:** ACL 2025
>
> **摘要:** Key-Value (KV) cache has become a bottleneck of LLMs for long-context generation. Despite the numerous efforts in this area, the optimization for the decoding phase is generally ignored. However, we believe such optimization is crucial, especially for long-output generation tasks based on the following two observations: (i) Excessive compression during the prefill phase, which requires specific full context impairs the comprehension of the reasoning task; (ii) Deviation of heavy hitters occurs in the reasoning tasks with long outputs. Therefore, SCOPE, a simple yet efficient framework that separately performs KV cache optimization during the prefill and decoding phases, is introduced. Specifically, the KV cache during the prefill phase is preserved to maintain the essential information, while a novel strategy based on sliding is proposed to select essential heavy hitters for the decoding phase. Memory usage and memory transfer are further optimized using adaptive and discontinuous strategies. Extensive experiments on LongGenBench show the effectiveness and generalization of SCOPE and its compatibility as a plug-in to other prefill-only KV compression methods.
>
---
#### [replaced 087] A Similarity Paradigm Through Textual Regularization Without Forgetting
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14376v2](http://arxiv.org/pdf/2502.14376v2)**

> **作者:** Fangming Cui; Jan Fong; Rongfei Zeng; Xinmei Tian; Jun Yu
>
> **摘要:** Prompt learning has emerged as a promising method for adapting pre-trained visual-language models (VLMs) to a range of downstream tasks. While optimizing the context can be effective for improving performance on specific tasks, it can often lead to poor generalization performance on unseen classes or datasets sampled from different distributions. It may be attributed to the fact that textual prompts tend to overfit downstream data distributions, leading to the forgetting of generalized knowledge derived from hand-crafted prompts. In this paper, we propose a novel method called Similarity Paradigm with Textual Regularization (SPTR) for prompt learning without forgetting. SPTR is a two-pronged design based on hand-crafted prompts that is an inseparable framework. 1) To avoid forgetting general textual knowledge, we introduce the optimal transport as a textual regularization to finely ensure approximation with hand-crafted features and tuning textual features. 2) In order to continuously unleash the general ability of multiple hand-crafted prompts, we propose a similarity paradigm for natural alignment score and adversarial alignment score to improve model robustness for generalization. Both modules share a common objective in addressing generalization issues, aiming to maximize the generalization capability derived from multiple hand-crafted prompts. Four representative tasks (i.e., non-generalization few-shot learning, base-to-novel generalization, cross-dataset generalization, domain generalization) across 11 datasets demonstrate that SPTR outperforms existing prompt learning methods.
>
---
#### [replaced 088] XTRUST: On the Multilingual Trustworthiness of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.15762v2](http://arxiv.org/pdf/2409.15762v2)**

> **作者:** Yahan Li; Yi Wang; Yi Chang; Yuan Wu
>
> **备注:** 21 pages
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across a range of natural language processing (NLP) tasks, capturing the attention of both practitioners and the broader public. A key question that now preoccupies the AI community concerns the capabilities and limitations of these models, with trustworthiness emerging as a central issue, particularly as LLMs are increasingly applied in sensitive fields like healthcare and finance, where errors can have serious consequences. However, most previous studies on the trustworthiness of LLMs have been limited to a single language, typically the predominant one in the dataset, such as English. In response to the growing global deployment of LLMs, we introduce XTRUST, the first comprehensive multilingual trustworthiness benchmark. XTRUST encompasses a diverse range of topics, including illegal activities, hallucination, out-of-distribution (OOD) robustness, physical and mental health, toxicity, fairness, misinformation, privacy, and machine ethics, across 10 different languages. Using XTRUST, we conduct an empirical evaluation of the multilingual trustworthiness of five widely used LLMs, offering an in-depth analysis of their performance across languages and tasks. Our results indicate that many LLMs struggle with certain low-resource languages, such as Arabic and Russian, highlighting the considerable room for improvement in the multilingual trustworthiness of current language models. The code is available at https://github.com/LluckyYH/XTRUST.
>
---
#### [replaced 089] STORM-BORN: A Challenging Mathematical Derivations Dataset Curated via a Human-in-the-Loop Multi-Agent Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01531v2](http://arxiv.org/pdf/2506.01531v2)**

> **作者:** Wenhao Liu; Zhenyi Lu; Xinyu Hu; Jierui Zhang; Dailin Li; Jiacheng Cen; Huilin Cao; Haiteng Wang; Yuhan Li; Kun Xie; Dandan Li; Pei Zhang; Chengbo Zhang; Yuxiang Ren; Xiaohong Huang; Yan Ma
>
> **备注:** accepted by ACL2025
>
> **摘要:** High-quality math datasets are crucial for advancing the reasoning abilities of large language models (LLMs). However, existing datasets often suffer from three key issues: outdated and insufficient challenging content, neglecting human-like reasoning, and limited reliability due to single-LLM generation. To address these, we introduce STORM-BORN, an ultra-challenging dataset of mathematical derivations sourced from cutting-edge academic papers, which includes dense human-like approximations and heuristic cues. To ensure the reliability and quality, we propose a novel human-in-the-loop, multi-agent data generation framework, integrating reasoning-dense filters, multi-agent collaboration, and human mathematicians' evaluations. We curated a set of 2,000 synthetic samples and deliberately selected the 100 most difficult problems. Even most advanced models like GPT-o1 solved fewer than 5% of them. Fine-tuning on STORM-BORN boosts accuracy by 7.84% (LLaMA3-8B) and 9.12% (Qwen2.5-7B). As AI approaches mathematician-level reasoning, STORM-BORN provides both a high-difficulty benchmark and a human-like reasoning training resource. Our code and dataset are publicly available at https://github.com/lwhere/STORM-BORN.
>
---
#### [replaced 090] Conti Inc.: Understanding the Internal Discussions of a large Ransomware-as-a-Service Operator with Machine Learning
- **分类: cs.CR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2308.16061v2](http://arxiv.org/pdf/2308.16061v2)**

> **作者:** Estelle Ruellan; Masarah Paquet-Clouston; Sebastian Garcia
>
> **摘要:** Ransomware-as-a-service (RaaS) is increasing the scale and complexity of ransomware attacks. Understanding the internal operations behind RaaS has been a challenge due to the illegality of such activities. The recent chat leak of the Conti RaaS operator, one of the most infamous ransomware operators on the international scene, offers a key opportunity to better understand the inner workings of such organizations. This paper analyzes the main topic discussions in the Conti chat leak using machine learning techniques such as Natural Language Processing (NLP) and Latent Dirichlet Allocation (LDA), as well as visualization strategies. Five discussion topics are found: 1) Business, 2) Technical, 3) Internal tasking/Management, 4) Malware, and 5) Customer Service/Problem Solving. Moreover, the distribution of topics among Conti members shows that only 4% of individuals have specialized discussions while almost all individuals (96%) are all-rounders, meaning that their discussions revolve around the five topics. The results also indicate that a significant proportion of Conti discussions are non-tech related. This study thus highlights that running such large RaaS operations requires a workforce skilled beyond technical abilities, with individuals involved in various tasks, from management to customer service or problem solving. The discussion topics also show that the organization behind the Conti RaaS oper5086933ator shares similarities with a large firm. We conclude that, although RaaS represents an example of specialization in the cybercrime industry, only a few members are specialized in one topic, while the rest runs and coordinates the RaaS operation.
>
---
#### [replaced 091] Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13338v2](http://arxiv.org/pdf/2505.13338v2)**

> **作者:** Qiongqiong Wang; Hardik B. Sailor; Tianchi Liu; Ai Ti Aw
>
> **备注:** Accepted at Interspeech 2025. [v2]: The dataset has been released, and the link is now updated
>
> **摘要:** Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities.
>
---
#### [replaced 092] Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.01789v2](http://arxiv.org/pdf/2506.01789v2)**

> **作者:** Genta Indra Winata; David Anugraha; Emmy Liu; Alham Fikri Aji; Shou-Yi Hung; Aditya Parashar; Patrick Amadeus Irawan; Ruochen Zhang; Zheng-Xin Yong; Jan Christian Blaise Cruz; Niklas Muennighoff; Seungone Kim; Hanyang Zhao; Sudipta Kar; Kezia Erina Suryoraharjo; M. Farid Adilazuarda; En-Shiun Annie Lee; Ayu Purwarianti; Derry Tanti Wijaya; Monojit Choudhury
>
> **备注:** Preprint
>
> **摘要:** High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at https://github.com/datarubrics/datarubrics.
>
---
#### [replaced 093] Q-STRUM Debate: Query-Driven Contrastive Summarization for Recommendation Comparison
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12921v2](http://arxiv.org/pdf/2502.12921v2)**

> **作者:** George-Kirollos Saad; Scott Sanner
>
> **摘要:** Query-driven recommendation with unknown items poses a challenge for users to understand why certain items are appropriate for their needs. Query-driven Contrastive Summarization (QCS) is a methodology designed to address this issue by leveraging language-based item descriptions to clarify contrasts between them. However, existing state-of-the-art contrastive summarization methods such as STRUM-LLM fall short of this goal. To overcome these limitations, we introduce Q-STRUM Debate, a novel extension of STRUM-LLM that employs debate-style prompting to generate focused and contrastive summarizations of item aspects relevant to a query. Leveraging modern large language models (LLMs) as powerful tools for generating debates, Q-STRUM Debate provides enhanced contrastive summaries. Experiments across three datasets demonstrate that Q-STRUM Debate yields significant performance improvements over existing methods on key contrastive summarization criteria, thus introducing a novel and performant debate prompting methodology for QCS.
>
---
#### [replaced 094] Cross-Institutional Dental EHR Entity Extraction via Generative AI and Synthetic Notes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.21050v3](http://arxiv.org/pdf/2407.21050v3)**

> **作者:** Yao-Shun Chuang; Chun-Teh Lee; Oluwabunmi Tokede; Guo-Hao Lin; Ryan Brandon; Trung Duong Tran; Xiaoqian Jiang; Muhammad F. Walji
>
> **备注:** 11 pages, 2 tables, 3 figures, under review
>
> **摘要:** This research addresses the issue of missing structured data in dental records by extracting diagnostic information from unstructured text. The updated periodontology classification system's complexity has increased incomplete or missing structured diagnoses. To tackle this, we use advanced AI and NLP methods, leveraging GPT-4 to generate synthetic notes for fine-tuning a RoBERTa model. This significantly enhances the model's ability to understand medical and dental language. We evaluated the model using 120 randomly selected clinical notes from two datasets, demonstrating its improved diagnostic extraction accuracy. The results showed high accuracy in diagnosing periodontal status, stage, and grade, with Site 1 scoring 0.99 and Site 2 scoring 0.98. In the subtype category, Site 2 achieved perfect scores, outperforming Site 1. This method enhances extraction accuracy and broadens its use across dental contexts. The study underscores AI and NLP's transformative impact on healthcare delivery and management. Integrating AI and NLP technologies enhances documentation and simplifies administrative tasks by precisely extracting complex clinical information. This approach effectively addresses challenges in dental diagnostics. Using synthetic training data from LLMs optimizes the training process, improving accuracy and efficiency in identifying periodontal diagnoses from clinical notes. This innovative method holds promise for broader healthcare applications, potentially improving patient care quality.
>
---
#### [replaced 095] UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.03525v4](http://arxiv.org/pdf/2407.03525v4)**

> **作者:** Md Nayem Uddin; Amir Saeidi; Divij Handa; Agastya Seth; Tran Cao Son; Eduardo Blanco; Steven R. Corman; Chitta Baral
>
> **备注:** Accepted at ACL 2025 (Main)
>
> **摘要:** This paper introduces UnSeenTimeQA, a novel data contamination-free time-sensitive question-answering (TSQA) benchmark. It differs from existing TSQA benchmarks by avoiding web-searchable queries grounded in the real world. We present a series of time-sensitive event scenarios based on synthetically generated facts. It requires large language models (LLMs) to engage in genuine temporal reasoning without depending on the factual knowledge acquired during the pre-training phase. Our data generation framework enables on-demand generation of new samples, mitigating the risk of data leakage. We designed three types of time-sensitive questions to test LLMs' temporal reasoning abilities over sequential and parallel event occurrences. Our evaluation of five LLMs on synthetic fact-based TSQA reveals mixed results: while they perform well on simpler subsets, their overall performance remains inferior as compared to real world fact-based TSQA. Error analysis indicates that LLMs face difficulties in reasoning over long-range event dependencies and parallel events.
>
---
#### [replaced 096] Splintering Nonconcatenative Languages for Better Tokenization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14433v2](http://arxiv.org/pdf/2503.14433v2)**

> **作者:** Bar Gazit; Shaltiel Shmidman; Avi Shmidman; Yuval Pinter
>
> **备注:** Findings of the ACL 2025
>
> **摘要:** Common subword tokenization algorithms like BPE and UnigramLM assume that text can be split into meaningful units by concatenative measures alone. This is not true for languages such as Hebrew and Arabic, where morphology is encoded in root-template patterns, or Malay and Georgian, where split affixes are common. We present SPLINTER, a pre-processing step which rearranges text into a linear form that better represents such nonconcatenative morphologies, enabling meaningful contiguous segments to be found by the tokenizer. We demonstrate SPLINTER's merit using both intrinsic measures evaluating token vocabularies in Hebrew, Arabic, and Malay; as well as on downstream tasks using BERT-architecture models trained for Hebrew.
>
---
#### [replaced 097] LazyReview A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11042v3](http://arxiv.org/pdf/2504.11042v3)**

> **作者:** Sukannya Purkayastha; Zhuang Li; Anne Lauscher; Lizhen Qu; Iryna Gurevych
>
> **备注:** Accepted at ACL 2025: 29 pages, 18 Figures, 15 Tables
>
> **摘要:** Peer review is a cornerstone of quality control in scientific publishing. With the increasing workload, the unintended use of `quick' heuristics, referred to as lazy thinking, has emerged as a recurring issue compromising review quality. Automated methods to detect such heuristics can help improve the peer-reviewing process. However, there is limited NLP research on this issue, and no real-world dataset exists to support the development of detection tools. This work introduces LazyReview, a dataset of peer-review sentences annotated with fine-grained lazy thinking categories. Our analysis reveals that Large Language Models (LLMs) struggle to detect these instances in a zero-shot setting. However, instruction-based fine-tuning on our dataset significantly boosts performance by 10-20 performance points, highlighting the importance of high-quality training data. Furthermore, a controlled experiment demonstrates that reviews revised with lazy thinking feedback are more comprehensive and actionable than those written without such feedback. We will release our dataset and the enhanced guidelines that can be used to train junior reviewers in the community. (Code available here: https://github.com/UKPLab/acl2025-lazy-review)
>
---
#### [replaced 098] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings
- **分类: cs.CL; cs.CR; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.12562v3](http://arxiv.org/pdf/2502.12562v3)**

> **作者:** Weikai Lu; Hao Peng; Huiping Zhuang; Cen Chen; Ziqian Zeng
>
> **备注:** Accepted in ACL 2025 Main Track
>
> **摘要:** Multimodal Large Language Models (MLLMs) have serious security vulnerabilities.While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.
>
---
#### [replaced 099] Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13887v3](http://arxiv.org/pdf/2505.13887v3)**

> **作者:** Junyang Wang; Haiyang Xu; Xi Zhang; Ming Yan; Ji Zhang; Fei Huang; Jitao Sang
>
> **备注:** I submitted the replacement version as a new article by mistake. Future updates will appear at arXiv:2502.17110
>
> **摘要:** The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation.
>
---
#### [replaced 100] The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
- **分类: cs.LG; cs.AI; cs.CL; cs.NA; math.NA; math.OC; 65F30, 68T07, 68N19; G.1.3; I.2.6; F.2.1; G.1.6**

- **链接: [http://arxiv.org/pdf/2505.16932v2](http://arxiv.org/pdf/2505.16932v2)**

> **作者:** Noah Amsel; David Persson; Christopher Musco; Robert M. Gower
>
> **备注:** 34 pages, 8 figures, 4 algorithms
>
> **摘要:** Computing the polar decomposition and the related matrix sign function, has been a well-studied problem in numerical analysis for decades. More recently, it has emerged as an important subroutine in deep learning, particularly within the Muon optimization framework. However, the requirements in this setting differ significantly from those of traditional numerical analysis. In deep learning, methods must be highly efficient and GPU-compatible, but high accuracy is often unnecessary. As a result, classical algorithms like Newton-Schulz (which suffers from slow initial convergence) and methods based on rational functions (which rely on QR decompositions or matrix inverses) are poorly suited to this context. In this work, we introduce Polar Express, a GPU-friendly algorithm for computing the polar decomposition. Like classical polynomial methods such as Newton-Schulz, our approach uses only matrix-matrix multiplications, making it GPU-compatible. Motivated by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the polynomial update rule at each iteration by solving a minimax optimization problem, and we prove that it enjoys a strong worst-case optimality guarantee. This property ensures both rapid early convergence and fast asymptotic convergence. We also address finite-precision issues, making it stable in bfloat16 in practice. We apply Polar Express within the Muon optimization framework and show consistent improvements in validation loss on large-scale models such as GPT-2, outperforming recent alternatives across a range of learning rates.
>
---
#### [replaced 101] Enhancing Ultra-Low-Bit Quantization of Large Language Models Through Saliency-Aware Partial Retraining
- **分类: cs.LG; cs.CL; 68T50, 68T07, 68T09, 68U15; I.2.7; I.2.6; I.2.4**

- **链接: [http://arxiv.org/pdf/2504.13932v2](http://arxiv.org/pdf/2504.13932v2)**

> **作者:** Deyu Cao; Samin Aref
>
> **备注:** This is a post-peer-review accepted manuscript from the proceedings of the 22nd International Conference on Modeling Decisions for Artificial Intelligence (MDAI'25). The publisher authenticated version and full citation details are available on Springer's website. 31 pages, 4 figures, 16 tables
>
> **摘要:** The growing use of large language models has raised environmental and economic concerns about their intensity of resource usage during inference. Serving these models to each user requires substantial energy and water for cooling. Model compression techniques like quantization can shrink large language models and make them more resource efficient at the cost of potential performance degradation. Quantization methods compress model size through replacing their high-precision parameters by quantized values of lower precision. Among existing methods, the ApiQ method achieves superior accuracy preservation at minimal memory and time overhead. We investigate two ideas to extend performance in ultra-low-bit quantization beyond ApiQ's level. First, we look into combining existing quantization-aware training techniques with ApiQ's partial training. We show that this does not outperform the baseline ApiQ method with limited training data and frozen weights. This leads to two key insights: (1) The substantial representational capacity that is gained through full retraining is unlikely to be feasible through partial training. (2) This gain may depend on using a large and diverse dataset in quantization-aware training. Second, through a novel approach informed by the two insights, we propose an ultra-low-bit quantization method that builds upon ApiQ and extends its performance without the need for full retraining. This publicly available method relies on a saliency-aware regularization term that prioritizes preserving the most impactful parameters during quantization. Our experiments on LLaMA 7B and 13B benchmarks demonstrate that our method reduces the ApiQ's accuracy degradation by 10.85\% and 7.54\% respectively.
>
---
#### [replaced 102] Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.20129v3](http://arxiv.org/pdf/2502.20129v3)**

> **作者:** Yifan Zhang; Wenyu Du; Dongming Jin; Jie Fu; Zhi Jin
>
> **摘要:** Chain-of-thought (CoT) significantly enhances the performance of large language models (LLMs) across a wide range of tasks, and prior research shows that CoT can theoretically increase expressiveness. However, there is limited mechanistic understanding of the algorithms that Transformer+CoT can learn. Our key contributions are: (1) We evaluate the state tracking capabilities of Transformer+CoT and its variants, confirming the effectiveness of CoT. (2) Next, we identify the circuit (a subset of model components, responsible for tracking the world state), indicating that late-layer MLP neurons play a key role. We propose two metrics, compression and distinction, and show that the neuron sets for each state achieve nearly 100% accuracy, providing evidence of an implicit finite state automaton (FSA) embedded within the model. (3) Additionally, we explore three challenging settings: skipping intermediate steps, introducing data noises, and testing length generalization. Our results demonstrate that Transformer+CoT learns robust algorithms (FSAs), highlighting its resilience in challenging scenarios. Our code is available at https://github.com/IvanChangPKU/FSA.
>
---
#### [replaced 103] Improving the Language Understanding Capabilities of Large Language Models Using Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.11020v4](http://arxiv.org/pdf/2410.11020v4)**

> **作者:** Bokai Hu; Sai Ashish Somayajula; Xin Pan; Pengtao Xie
>
> **摘要:** Instruction-fine-tuned large language models (LLMs) under 14B parameters continue to underperform on natural language understanding (NLU) tasks, often trailing smaller models like BERT-base on benchmarks such as GLUE and SuperGLUE. Motivated by the success of reinforcement learning in reasoning tasks (e.g., DeepSeek), we explore Proximal Policy Optimization (PPO) as a framework to improve the NLU capabilities of LLMs. We frame NLU as a reinforcement learning environment, treating token generation as a sequence of actions and optimizing for reward signals based on alignment with ground-truth labels. PPO consistently outperforms supervised fine-tuning, yielding an average improvement of 6.3 points on GLUE, and surpasses zero-shot and few-shot prompting by 38.7 and 26.1 points, respectively. Notably, PPO-tuned models outperform GPT-4o by over 4\% on average across sentiment and natural language inference tasks, including gains of 7.3\% on the Mental Health dataset and 10.9\% on SIGA-nli. This work highlights a promising direction for adapting LLMs to new tasks by reframing them as reinforcement learning problems, enabling learning through simple end-task rewards rather than extensive data curation.
>
---
#### [replaced 104] Social Genome: Grounded Social Reasoning Abilities of Multimodal Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.15109v3](http://arxiv.org/pdf/2502.15109v3)**

> **作者:** Leena Mathur; Marian Qian; Paul Pu Liang; Louis-Philippe Morency
>
> **备注:** Under Review, 24 pages
>
> **摘要:** Social reasoning abilities are crucial for AI systems to effectively interpret and respond to multimodal human communication and interaction within social contexts. We introduce SOCIAL GENOME, the first benchmark for fine-grained, grounded social reasoning abilities of multimodal models. SOCIAL GENOME contains 272 videos of interactions and 1,486 human-annotated reasoning traces related to inferences about these interactions. These traces contain 5,777 reasoning steps that reference evidence from visual cues, verbal cues, vocal cues, and external knowledge (contextual knowledge external to videos). SOCIAL GENOME is also the first modeling challenge to study external knowledge in social reasoning. SOCIAL GENOME computes metrics to holistically evaluate semantic and structural qualities of model-generated social reasoning traces. We demonstrate the utility of SOCIAL GENOME through experiments with state-of-the-art models, identifying performance gaps and opportunities for future research to improve the grounded social reasoning abilities of multimodal models.
>
---
#### [replaced 105] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16765v2](http://arxiv.org/pdf/2411.16765v2)**

> **作者:** Shester Gueuwou; Xiaodan Du; Greg Shakhnarovich; Karen Livescu; Alexander H. Liu
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Sign language processing has traditionally relied on task-specific models, limiting the potential for transfer learning across tasks. Pre-training methods for sign language have typically focused on either supervised pre-training, which cannot take advantage of unlabeled data, or context-independent (frame or video segment) representations, which ignore the effects of relationships across time in sign language. We introduce SHuBERT (Sign Hidden-Unit BERT), a self-supervised contextual representation model learned from approximately 1,000 hours of American Sign Language video. SHuBERT adapts masked token prediction objectives to multi-stream visual sign language input, learning to predict multiple targets corresponding to clustered hand, face, and body pose streams. SHuBERT achieves state-of-the-art performance across multiple tasks including sign language translation, isolated sign language recognition, and fingerspelling detection.
>
---
#### [replaced 106] Superhuman performance of a large language model on the reasoning tasks of a physician
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10849v3](http://arxiv.org/pdf/2412.10849v3)**

> **作者:** Peter G. Brodeur; Thomas A. Buckley; Zahir Kanjee; Ethan Goh; Evelyn Bin Ling; Priyank Jain; Stephanie Cabral; Raja-Elie Abdulnour; Adrian D. Haimovich; Jason A. Freed; Andrew Olson; Daniel J. Morgan; Jason Hom; Robert Gallo; Liam G. McCoy; Haadi Mombini; Christopher Lucas; Misha Fotoohi; Matthew Gwiazdon; Daniele Restifo; Daniel Restrepo; Eric Horvitz; Jonathan Chen; Arjun K. Manrai; Adam Rodman
>
> **摘要:** A seminal paper published by Ledley and Lusted in 1959 introduced complex clinical diagnostic reasoning cases as the gold standard for the evaluation of expert medical computing systems, a standard that has held ever since. Here, we report the results of a physician evaluation of a large language model (LLM) on challenging clinical cases against a baseline of hundreds of physicians. We conduct five experiments to measure clinical reasoning across differential diagnosis generation, display of diagnostic reasoning, triage differential diagnosis, probabilistic reasoning, and management reasoning, all adjudicated by physician experts with validated psychometrics. We then report a real-world study comparing human expert and AI second opinions in randomly-selected patients in the emergency room of a major tertiary academic medical center in Boston, MA. We compared LLMs and board-certified physicians at three predefined diagnostic touchpoints: triage in the emergency room, initial evaluation by a physician, and admission to the hospital or intensive care unit. In all experiments--both vignettes and emergency room second opinions--the LLM displayed superhuman diagnostic and reasoning abilities, as well as continued improvement from prior generations of AI clinical decision support. Our study suggests that LLMs have achieved superhuman performance on general medical diagnostic and management reasoning, fulfilling the vision put forth by Ledley and Lusted, and motivating the urgent need for prospective trials.
>
---
#### [replaced 107] Efficient Annotator Reliability Assessment with EffiARA
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.00589v3](http://arxiv.org/pdf/2504.00589v3)**

> **作者:** Owen Cook; Jake Vasilakes; Ian Roberts; Xingyi Song
>
> **摘要:** Data annotation is an essential component of the machine learning pipeline; it is also a costly and time-consuming process. With the introduction of transformer-based models, annotation at the document level is increasingly popular; however, there is no standard framework for structuring such tasks. The EffiARA annotation framework is, to our knowledge, the first project to support the whole annotation pipeline, from understanding the resources required for an annotation task to compiling the annotated dataset and gaining insights into the reliability of individual annotators as well as the dataset as a whole. The framework's efficacy is supported by two previous studies: one improving classification performance through annotator-reliability-based soft-label aggregation and sample weighting, and the other increasing the overall agreement among annotators through removing identifying and replacing an unreliable annotator. This work introduces the EffiARA Python package and its accompanying webtool, which provides an accessible graphical user interface for the system. We open-source the EffiARA Python package at https://github.com/MiniEggz/EffiARA and the webtool is publicly accessible at https://effiara.gate.ac.uk.
>
---
#### [replaced 108] Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10995v4](http://arxiv.org/pdf/2410.10995v4)**

> **作者:** Emmanouil Zaranis; Giuseppe Attanasio; Sweta Agrawal; André F. T. Martins
>
> **备注:** ACL 2025
>
> **摘要:** Quality estimation (QE)-the automatic assessment of translation quality-has recently become crucial across several stages of the translation pipeline, from data curation to training and decoding. While QE metrics have been optimized to align with human judgments, whether they encode social biases has been largely overlooked. Biased QE risks favoring certain demographic groups over others, e.g., by exacerbating gaps in visibility and usability. This paper defines and investigates gender bias of QE metrics and discusses its downstream implications for machine translation (MT). Experiments with state-of-the-art QE metrics across multiple domains, datasets, and languages reveal significant bias. When a human entity's gender in the source is undisclosed, masculine-inflected translations score higher than feminine-inflected ones, and gender-neutral translations are penalized. Even when contextual cues disambiguate gender, using context-aware QE metrics leads to more errors in selecting the correct translation inflection for feminine referents than for masculine ones. Moreover, a biased QE metric affects data filtering and quality-aware decoding. Our findings underscore the need for a renewed focus on developing and evaluating QE metrics centered on gender.
>
---
#### [replaced 109] A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06150v2](http://arxiv.org/pdf/2505.06150v2)**

> **作者:** Ryan Lagasse; Aidan Kierans; Avijit Ghosh; Shiri Dori-Hacohen
>
> **摘要:** We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings.
>
---
#### [replaced 110] Localizing and Mitigating Errors in Long-form Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.11930v5](http://arxiv.org/pdf/2407.11930v5)**

> **作者:** Rachneet Sachdeva; Yixiao Song; Mohit Iyyer; Iryna Gurevych
>
> **备注:** ACL 2025 Findings; Code and data are available: https://github.com/UKPLab/acl2025-lfqa-hallucination
>
> **摘要:** Long-form question answering (LFQA) aims to provide thorough and in-depth answers to complex questions, enhancing comprehension. However, such detailed responses are prone to hallucinations and factual inconsistencies, challenging their faithful evaluation. This work introduces HaluQuestQA, the first hallucination dataset with localized error annotations for human-written and model-generated LFQA answers. HaluQuestQA comprises 698 QA pairs with 1.8k span-level error annotations for five different error types by expert annotators, along with preference judgments. Using our collected data, we thoroughly analyze the shortcomings of long-form answers and find that they lack comprehensiveness and provide unhelpful references. We train an automatic feedback model on this dataset that predicts error spans with incomplete information and provides associated explanations. Finally, we propose a prompt-based approach, Error-informed refinement, that uses signals from the learned feedback model to refine generated answers, which we show reduces errors and improves answer quality across multiple models. Furthermore, humans find answers generated by our approach comprehensive and highly prefer them (84%) over the baseline answers.
>
---
#### [replaced 111] SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.06907v2](http://arxiv.org/pdf/2406.06907v2)**

> **作者:** Shester Gueuwou; Xiaodan Du; Greg Shakhnarovich; Karen Livescu
>
> **备注:** Accepted to ACL (Findings) 2025
>
> **摘要:** A persistent challenge in sign language video processing, including the task of sign to written language translation, is how we learn representations of sign language in an effective and efficient way that preserves the important attributes of these languages, while remaining invariant to irrelevant visual differences. Informed by the nature and linguistics of signed languages, our proposed method focuses on just the most relevant parts in a signing video: the face, hands and body pose of the signer. However, instead of fully relying on pose estimation from off-the-shelf pose tracking models, which have inconsistent performance for hands and faces, we propose to learn a representation of the complex handshapes and facial expressions of sign languages in a self-supervised fashion. Our approach is based on learning from individual frames (rather than video sequences) and is therefore much more efficient than prior work on sign language pre-training. Compared to a recent model that established a new state of the art in sign language translation on the How2Sign dataset, our approach yields similar translation performance, using less than 3\% of the compute.
>
---
#### [replaced 112] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20322v2](http://arxiv.org/pdf/2505.20322v2)**

> **作者:** Mengru Wang; Ziwen Xu; Shengyu Mao; Shumin Deng; Zhaopeng Tu; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control.
>
---
#### [replaced 113] A Complexity-Based Theory of Compositionality
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.14817v5](http://arxiv.org/pdf/2410.14817v5)**

> **作者:** Eric Elmoznino; Thomas Jiralerspong; Yoshua Bengio; Guillaume Lajoie
>
> **摘要:** Compositionality is believed to be fundamental to intelligence. In humans, it underlies the structure of thought, language, and higher-level reasoning. In AI, compositional representations can enable a powerful form of out-of-distribution generalization, in which a model systematically adapts to novel combinations of known concepts. However, while we have strong intuitions about what compositionality is, we lack satisfying formal definitions for it that are measurable and mathematical. Here, we propose such a definition, which we call representational compositionality, that accounts for and extends our intuitions about compositionality. The definition is conceptually simple, quantitative, grounded in algorithmic information theory, and applicable to any representation. Intuitively, representational compositionality states that a compositional representation satisfies three properties. First, it must be expressive. Second, it must be possible to re-describe the representation as a function of discrete symbolic sequences with re-combinable parts, analogous to sentences in natural language. Third, the function that relates these symbolic sequences to the representation, analogous to semantics in natural language, must be simple. Through experiments on both synthetic and real world data, we validate our definition of compositionality and show how it unifies disparate intuitions from across the literature in both AI and cognitive science. We also show that representational compositionality, while theoretically intractable, can be readily estimated using standard deep learning tools. We hope that our definition can inspire the design of novel, theoretically-driven models that better capture the mechanisms of compositional thought. We make our code available at https://github.com/EricElmoznino/complexity_compositionality.
>
---
#### [replaced 114] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2410.03869v2](http://arxiv.org/pdf/2410.03869v2)**

> **作者:** Wenxuan Wang; Kuiyi Gao; Youliang Yuan; Jen-tse Huang; Qiuzhi Liu; Shuai Wang; Wenxiang Jiao; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Text-based image generation models, such as Stable Diffusion and DALL-E 3, hold significant potential in content creation and publishing workflows, making them the focus in recent years. Despite their remarkable capability to generate diverse and vivid images, considerable efforts are being made to prevent the generation of harmful content, such as abusive, violent, or pornographic material. To assess the safety of existing models, we introduce a novel jailbreaking method called Chain-of-Jailbreak (CoJ) attack, which compromises image generation models through a step-by-step editing process. Specifically, for malicious queries that cannot bypass the safeguards with a single prompt, we intentionally decompose the query into multiple sub-queries. The image generation models are then prompted to generate and iteratively edit images based on these sub-queries. To evaluate the effectiveness of our CoJ attack method, we constructed a comprehensive dataset, CoJ-Bench, encompassing nine safety scenarios, three types of editing operations, and three editing elements. Experiments on four widely-used image generation services provided by GPT-4V, GPT-4o, Gemini 1.5 and Gemini 1.5 Pro, demonstrate that our CoJ attack method can successfully bypass the safeguards of models for over 60% cases, which significantly outperforms other jailbreaking methods (i.e., 14%). Further, to enhance these models' safety against our CoJ attack method, we also propose an effective prompting-based method, Think Twice Prompting, that can successfully defend over 95% of CoJ attack. We release our dataset and code to facilitate the AI safety research.
>
---
#### [replaced 115] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17878v2](http://arxiv.org/pdf/2502.17878v2)**

> **作者:** Hongqiu Wu; Weiqi Wu; Tianyang Xu; Jiameng Zhang; Hai Zhao
>
> **备注:** Accepted by ACL'2025
>
> **摘要:** LLM-based Interactive Drama is a novel AI-based dialogue scenario, where the user (i.e. the player) plays the role of a character in the story, has conversations with characters played by LLM agents, and experiences an unfolding story. This paper begins with understanding interactive drama from two aspects: Immersion, the player's feeling of being present in the story, and Agency, the player's ability to influence the story world. Both are crucial to creating an enjoyable interactive experience, while they have been underexplored in previous work. To enhance these two aspects, we first propose Playwriting-guided Generation, a novel method that helps LLMs craft dramatic stories with substantially improved structures and narrative quality. Additionally, we introduce Plot-based Reflection for LLM agents to refine their reactions to align with the player's intentions. Our evaluation relies on human judgment to assess the gains of our methods in terms of immersion and agency.
>
---
#### [replaced 116] Multimodal Forecasting of Sparse Intraoperative Hypotension Events Powered by Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22116v2](http://arxiv.org/pdf/2505.22116v2)**

> **作者:** Jintao Zhang; Zirui Liu; Mingyue Cheng; Shilong Zhang; Tingyue Pan; Qi Liu; Yanhu Xie
>
> **摘要:** Intraoperative hypotension (IOH) frequently occurs under general anesthesia and is strongly linked to adverse outcomes such as myocardial injury and increased mortality. Despite its significance, IOH prediction is hindered by event sparsity and the challenge of integrating static and dynamic data across diverse patients. In this paper, we propose \textbf{IOHFuseLM}, a multimodal language model framework. To accurately identify and differentiate sparse hypotensive events, we leverage a two-stage training strategy. The first stage involves domain adaptive pretraining on IOH physiological time series augmented through diffusion methods, thereby enhancing the model sensitivity to patterns associated with hypotension. Subsequently, task fine-tuning is performed on the original clinical dataset to further enhance the ability to distinguish normotensive from hypotensive states. To enable multimodal fusion for each patient, we align structured clinical descriptions with the corresponding physiological time series at the token level. Such alignment enables the model to capture individualized temporal patterns alongside their corresponding clinical semantics. In addition, we convert static patient attributes into structured text to enrich personalized information. Experimental evaluations on two intraoperative datasets demonstrate that IOHFuseLM outperforms established baselines in accurately identifying IOH events, highlighting its applicability in clinical decision support scenarios. Our code is publicly available to promote reproducibility at https://github.com/zjt-gpu/IOHFuseLM.
>
---
#### [replaced 117] LiTEx: A Linguistic Taxonomy of Explanations for Understanding Within-Label Variation in Natural Language Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22848v2](http://arxiv.org/pdf/2505.22848v2)**

> **作者:** Pingjun Hong; Beiduo Chen; Siyao Peng; Marie-Catherine de Marneffe; Barbara Plank
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** There is increasing evidence of Human Label Variation (HLV) in Natural Language Inference (NLI), where annotators assign different labels to the same premise-hypothesis pair. However, within-label variation--cases where annotators agree on the same label but provide divergent reasoning--poses an additional and mostly overlooked challenge. Several NLI datasets contain highlighted words in the NLI item as explanations, but the same spans on the NLI item can be highlighted for different reasons, as evidenced by free-text explanations, which offer a window into annotators' reasoning. To systematically understand this problem and gain insight into the rationales behind NLI labels, we introduce LITEX, a linguistically-informed taxonomy for categorizing free-text explanations. Using this taxonomy, we annotate a subset of the e-SNLI dataset, validate the taxonomy's reliability, and analyze how it aligns with NLI labels, highlights, and explanations. We further assess the taxonomy's usefulness in explanation generation, demonstrating that conditioning generation on LITEX yields explanations that are linguistically closer to human explanations than those generated using only labels or highlights. Our approach thus not only captures within-label variation but also shows how taxonomy-guided generation for reasoning can bridge the gap between human and model explanations more effectively than existing strategies.
>
---
#### [replaced 118] Can Character-based Language Models Improve Downstream Task Performance in Low-Resource and Noisy Language Scenarios?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2110.13658v2](http://arxiv.org/pdf/2110.13658v2)**

> **作者:** Arij Riabi; Benoît Sagot; Djamé Seddah
>
> **备注:** updated version with new results
>
> **摘要:** Recent impressive improvements in NLP, largely based on the success of contextual neural language models, have been mostly demonstrated on at most a couple dozen high-resource languages. Building language models and, more generally, NLP systems for non-standardized and low-resource languages remains a challenging task. In this work, we focus on North-African colloquial dialectal Arabic written using an extension of the Latin script, called NArabizi, found mostly on social media and messaging communication. In this low-resource scenario with data displaying a high level of variability, we compare the downstream performance of a character-based language model on part-of-speech tagging and dependency parsing to that of monolingual and multilingual models. We show that a character-based model trained on only 99k sentences of NArabizi and fined-tuned on a small treebank of this language leads to performance close to those obtained with the same architecture pre-trained on large multilingual and monolingual models. Confirming these results a on much larger data set of noisy French user-generated content, we argue that such character-based language models can be an asset for NLP in low-resource and high language variability set-tings.
>
---
#### [replaced 119] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15628v4](http://arxiv.org/pdf/2412.15628v4)**

> **作者:** Mengyu Ye; Tatsuki Kuribayashi; Goro Kobayashi; Jun Suzuki
>
> **备注:** Findings of ACL 2025
>
> **摘要:** Interpreting the internal process of neural models has long been a challenge. This challenge remains relevant in the era of large language models (LLMs) and in-context learning (ICL); for example, ICL poses a new issue of interpreting which example in the few-shot examples contributed to identifying/solving the task. To this end, in this paper, we design synthetic diagnostic tasks of inductive reasoning, inspired by the generalization tests typically adopted in psycholinguistics. Here, most in-context examples are ambiguous w.r.t. their underlying rule, and one critical example disambiguates it. The question is whether conventional input attribution (IA) methods can track such a reasoning process, i.e., identify the influential example, in ICL. Our experiments provide several practical findings; for example, a certain simple IA method works the best, and the larger the model, the generally harder it is to interpret the ICL with gradient-based IA methods.
>
---
#### [replaced 120] LLM-Driven E-Commerce Marketing Content Optimization: Balancing Creativity and Conversion
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.23809v2](http://arxiv.org/pdf/2505.23809v2)**

> **作者:** Haowei Yang; Haotian Lyu; Tianle Zhang; Dingzhou Wang; Yushang Zhao
>
> **摘要:** As e-commerce competition intensifies, balancing creative content with conversion effectiveness becomes critical. Leveraging LLMs' language generation capabilities, we propose a framework that integrates prompt engineering, multi-objective fine-tuning, and post-processing to generate marketing copy that is both engaging and conversion-driven. Our fine-tuning method combines sentiment adjustment, diversity enhancement, and CTA embedding. Through offline evaluations and online A/B tests across categories, our approach achieves a 12.5 % increase in CTR and an 8.3 % increase in CVR while maintaining content novelty. This provides a practical solution for automated copy generation and suggests paths for future multimodal, real-time personalization.
>
---
#### [replaced 121] Time-R1: Towards Comprehensive Temporal Reasoning in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.13508v2](http://arxiv.org/pdf/2505.13508v2)**

> **作者:** Zijia Liu; Peixuan Han; Haofei Yu; Haoru Li; Jiaxuan You
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints.
>
---
#### [replaced 122] OASST-ETC Dataset: Alignment Signals from Eye-tracking Analysis of LLM Responses
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10927v3](http://arxiv.org/pdf/2503.10927v3)**

> **作者:** Angela Lopez-Cardona; Sebastian Idesis; Miguel Barreda-Ángeles; Sergi Abadal; Ioannis Arapakis
>
> **备注:** This paper has been accepted to ACM ETRA 2025 and published on PACMHCI
>
> **摘要:** While Large Language Models (LLMs) have significantly advanced natural language processing, aligning them with human preferences remains an open challenge. Although current alignment methods rely primarily on explicit feedback, eye-tracking (ET) data offers insights into real-time cognitive processing during reading. In this paper, we present OASST-ETC, a novel eye-tracking corpus capturing reading patterns from 24 participants, while evaluating LLM-generated responses from the OASST1 dataset. Our analysis reveals distinct reading patterns between preferred and non-preferred responses, which we compare with synthetic eye-tracking data. Furthermore, we examine the correlation between human reading measures and attention patterns from various transformer-based models, discovering stronger correlations in preferred responses. This work introduces a unique resource for studying human cognitive processing in LLM evaluation and suggests promising directions for incorporating eye-tracking data into alignment methods. The dataset and analysis code are publicly available.
>
---
#### [replaced 123] DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23754v2](http://arxiv.org/pdf/2505.23754v2)**

> **作者:** Ziyin Zhang; Jiahao Xu; Zhiwei He; Tian Liang; Qiuzhi Liu; Yansi Li; Linfeng Song; Zhenwen Liang; Zhuosheng Zhang; Rui Wang; Zhaopeng Tu; Haitao Mi; Dong Yu
>
> **摘要:** Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration.
>
---
#### [replaced 124] Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition in Social Media Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.11479v3](http://arxiv.org/pdf/2411.11479v3)**

> **作者:** Jingxuan Li; Yuning Yang; Shengqi Yang; Linfan Zhang; Ying Nian Wu
>
> **备注:** ACL 2025 main
>
> **摘要:** The recent progress in Vision-Language Models (VLMs) has broadened the scope of multimodal applications. However, evaluations often remain limited to functional tasks, neglecting abstract dimensions such as personality traits and human values. To address this gap, we introduce Value-Spectrum, a novel Visual Question Answering (VQA) benchmark aimed at assessing VLMs based on Schwartz's value dimensions that capture core human values guiding people's preferences and actions. We design a VLM agent pipeline to simulate video browsing and construct a vector database comprising over 50,000 short videos from TikTok, YouTube Shorts, and Instagram Reels. These videos span multiple months and cover diverse topics, including family, health, hobbies, society, technology, etc. Benchmarking on Value-Spectrum highlights notable variations in how VLMs handle value-oriented content. Beyond identifying VLMs' intrinsic preferences, we also explore the ability of VLM agents to adopt specific personas when explicitly prompted, revealing insights into the adaptability of the model in role-playing scenarios. These findings highlight the potential of Value-Spectrum as a comprehensive evaluation set for tracking VLM preferences in value-based tasks and abilities to simulate diverse personas. The complete code and data are available at: https://github.com/Jeremyyny/Value-Spectrum.
>
---
#### [replaced 125] One-shot Entropy Minimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20282v3](http://arxiv.org/pdf/2505.20282v3)**

> **作者:** Zitian Gao; Lynx Chen; Joey Zhou; Bryan Dai
>
> **备注:** Work in progress
>
> **摘要:** We trained 13,440 large language models and found that entropy minimization requires only a single unlabeled data and 10 steps optimization to achieve performance improvements comparable to or even greater than those obtained using thousands of data and carefully designed rewards in rule-based reinforcement learning. This striking result may prompt a rethinking of post-training paradigms for large language models. Our code is avaliable at https://github.com/zitian-gao/one-shot-em.
>
---
#### [replaced 126] On the class of coding optimality of human languages and the origins of Zipf's law
- **分类: cs.CL; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2505.20015v2](http://arxiv.org/pdf/2505.20015v2)**

> **作者:** Ramon Ferrer-i-Cancho
>
> **摘要:** Here we present a new class of optimality for coding systems. Members of that class are displaced linearly from optimal coding and thus exhibit Zipf's law, namely a power-law distribution of frequency ranks. Within that class, Zipf's law, the size-rank law and the size-probability law form a group-like structure. We identify human languages that are members of the class. All languages showing sufficient agreement with Zipf's law are potential members of the class. In contrast, there are communication systems in other species that cannot be members of that class for exhibiting an exponential distribution instead but dolphins and humpback whales might. We provide a new insight into plots of frequency versus rank in double logarithmic scale. For any system, a straight line in that scale indicates that the lengths of optimal codes under non-singular coding and under uniquely decodable encoding are displaced by a linear function whose slope is the exponent of Zipf's law. For systems under compression and constrained to be uniquely decodable, such a straight line may indicate that the system is coding close to optimality. Our findings provide support for the hypothesis that Zipf's law originates from compression.
>
---
#### [replaced 127] Continual Speech Learning with Fused Speech Features
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.01496v2](http://arxiv.org/pdf/2506.01496v2)**

> **作者:** Guitao Wang; Jinming Zhao; Hao Yang; Guilin Qi; Tongtong Wu; Gholamreza Haffari
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Rapid growth in speech data demands adaptive models, as traditional static methods fail to keep pace with dynamic and diverse speech information. We introduce continuous speech learning, a new set-up targeting at bridging the adaptation gap in current speech models. We use the encoder-decoder Whisper model to standardize speech tasks into a generative format. We integrate a learnable gated-fusion layer on the top of the encoder to dynamically select task-specific features for downstream tasks. Our approach improves accuracy significantly over traditional methods in six speech processing tasks, demonstrating gains in adapting to new speech tasks without full retraining.
>
---
#### [replaced 128] What Goes Into a LM Acceptability Judgment? Rethinking the Impact of Frequency and Length
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.02528v3](http://arxiv.org/pdf/2411.02528v3)**

> **作者:** Lindia Tjuatja; Graham Neubig; Tal Linzen; Sophie Hao
>
> **备注:** Accepted to NAACL 2025 (Main Conference)
>
> **摘要:** When comparing the linguistic capabilities of language models (LMs) with humans using LM probabilities, factors such as the length of the sequence and the unigram frequency of lexical items have a significant effect on LM probabilities in ways that humans are largely robust to. Prior works in comparing LM and human acceptability judgments treat these effects uniformly across models, making a strong assumption that models require the same degree of adjustment to control for length and unigram frequency effects. We propose MORCELA, a new linking theory between LM scores and acceptability judgments where the optimal level of adjustment for these effects is estimated from data via learned parameters for length and unigram frequency. We first show that MORCELA outperforms a commonly used linking theory for acceptability - SLOR (Pauls and Klein, 2012; Lau et al. 2017) - across two families of transformer LMs (Pythia and OPT). Furthermore, we demonstrate that the assumed degrees of adjustment in SLOR for length and unigram frequency overcorrect for these confounds, and that larger models require a lower relative degree of adjustment for unigram frequency, though a significant amount of adjustment is still necessary for all models. Finally, our subsequent analysis shows that larger LMs' lower susceptibility to frequency effects can be explained by an ability to better predict rarer words in context.
>
---
#### [replaced 129] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.17579v2](http://arxiv.org/pdf/2503.17579v2)**

> **作者:** Suet-Ying Lam; Qingcheng Zeng; Jingyi Wu; Rob Voigt
>
> **备注:** ACL 2025 Camera-ready
>
> **摘要:** Whether large language models (LLMs) process language similarly to humans has been the subject of much theoretical and practical debate. We examine this question through the lens of the production-interpretation distinction found in human sentence processing and evaluate the extent to which instruction-tuned LLMs replicate this distinction. Using an empirically documented asymmetry between pronoun production and interpretation in humans for implicit causality verbs as a testbed, we find that some LLMs do quantitatively and qualitatively reflect human-like asymmetries between production and interpretation. We demonstrate that whether this behavior holds depends upon both model size-with larger models more likely to reflect human-like patterns and the choice of meta-linguistic prompts used to elicit the behavior. Our codes and results are available at https://github.com/LingMechLab/Production-Interpretation_Asymmetries_ACL2025.
>
---
#### [replaced 130] UltraWiki: Ultra-fine-grained Entity Set Expansion with Negative Seed Entities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.04247v3](http://arxiv.org/pdf/2403.04247v3)**

> **作者:** Yangning Li; Qingsong Lv; Tianyu Yu; Yinghui Li; Xuming Hu; Wenhao Jiang; Hai-Tao Zheng; Hui Wang
>
> **备注:** Accepted by ICDE 2025
>
> **摘要:** Entity Set Expansion (ESE) aims to identify new entities belonging to the same semantic class as the given set of seed entities. Traditional methods solely relied on positive seed entities to represent the target fine-grained semantic class, rendering them tough to represent ultra-fine-grained semantic classes. Specifically, merely relying on positive seed entities leads to two inherent shortcomings: (i) Ambiguity among ultra-fine-grained semantic classes. (ii) Inability to define ``unwanted'' semantics. Hence, previous ESE methods struggle to address the ultra-fine-grained ESE (Ultra-ESE) task. To solve this issue, we first introduce negative seed entities in the inputs, which jointly describe the ultra-fine-grained semantic class with positive seed entities. Negative seed entities eliminate the semantic ambiguity by providing a contrast between positive and negative attributes. Meanwhile, it provides a straightforward way to express ``unwanted''. To assess model performance in Ultra-ESE and facilitate further research, we also constructed UltraWiki, the first large-scale dataset tailored for Ultra-ESE. UltraWiki encompasses 50,973 entities and 394,097 sentences, alongside 236 ultra-fine-grained semantic classes, where each class is represented with 3-5 positive and negative seed entities. Moreover, a retrieval-based framework RetExpan and a generation-based framework GenExpan are proposed to provide powerful baselines for Ultra-ESE. Additionally, we devised two strategies to enhance models' comprehension of ultra-fine-grained entities' semantics: contrastive learning and chain-of-thought reasoning. Extensive experiments confirm the effectiveness of our proposed strategies and also reveal that there remains a large space for improvement in Ultra-ESE.
>
---
#### [replaced 131] LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17134v2](http://arxiv.org/pdf/2505.17134v2)**

> **作者:** Chaochen Gao; Xing Wu; Zijia Lin; Debing Zhang; Songlin Hu
>
> **摘要:** High-quality long-context instruction data is essential for aligning long-context large language models (LLMs). Despite the public release of models like Qwen and Llama, their long-context instruction data remains proprietary. Human annotation is costly and challenging, while template-based synthesis methods limit scale, diversity, and quality. We introduce LongMagpie, a self-synthesis framework that automatically generates large-scale long-context instruction data. Our key insight is that aligned long-context LLMs, when presented with a document followed by special tokens preceding a user turn, auto-regressively generate contextually relevant queries. By harvesting these document-query pairs and the model's responses, LongMagpie produces high-quality instructions without human effort. Experiments on HELMET, RULER, and Longbench v2 demonstrate that LongMagpie achieves leading performance on long-context tasks while maintaining competitive performance on short-context tasks, establishing it as a simple and effective approach for open, diverse, and scalable long-context instruction data synthesis.
>
---
#### [replaced 132] Improving Multilingual Speech Models on ML-SUPERB 2.0: Fine-tuning with Data Augmentation and LID-Aware CTC
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24200v2](http://arxiv.org/pdf/2505.24200v2)**

> **作者:** Qingzheng Wang; Jiancheng Sun; Yifan Peng; Shinji Watanabe
>
> **摘要:** Multilingual speech processing with self-supervised or supervised pre-trained Speech Foundation Models (SFM) has achieved strong performance on tasks like Language Identification (LID) and Automatic Speech Recognition (ASR). However, these models struggle with limited resources during fine-tuning. This paper enhances multilingual LID and ASR on ML-SUPERB 2.0 by exploring multiple strategies for adapting SFMs, including frozen upstream training, partial fine-tuning, and low-rank adaptation. Furthermore, we employ data augmentation to mitigate performance gaps in few-shot settings and introduce LID Connectionist Temporal Classification (CTC) loss for regularization. Our approach achieves a 14% relative improvement in LID accuracy and a 30% relative reduction in ASR CER over the baseline on ML-SUPERB 2.0, securing second place in the Interspeech 2025 ML-SUPERB 2.0 Challenge.
>
---
#### [replaced 133] PersianMedQA: Language-Centric Evaluation of LLMs in the Persian Medical Domain
- **分类: cs.CL; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2506.00250v2](http://arxiv.org/pdf/2506.00250v2)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Amirhossein Sheikholselami; Sepehr Karimi; Sepideh Ranjbar Kalahroodi; Heshaam Faili; Azadeh Shakery
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance on a wide range of NLP benchmarks, often surpassing human-level accuracy. However, their reliability in high-stakes domains such as medicine, particularly in low-resource languages, remains underexplored. In this work, we introduce PersianMedQA, a large-scale, expert-validated dataset of multiple-choice Persian medical questions, designed to evaluate LLMs across both Persian and English. We benchmark over 40 state-of-the-art models, including general-purpose, Persian fine-tuned, and medical LLMs, in zero-shot and chain-of-thought (CoT) settings. Our results show that closed-source general models (e.g., GPT-4.1) consistently outperform all other categories, achieving 83.3% accuracy in Persian and 80.7% in English, while Persian fine-tuned models such as Dorna underperform significantly (e.g., 35.9% in Persian), often struggling with both instruction-following and domain reasoning. We also analyze the impact of translation, showing that while English performance is generally higher, Persian responses are sometimes more accurate due to cultural and clinical contextual cues. Finally, we demonstrate that model size alone is insufficient for robust performance without strong domain or language adaptation. PersianMedQA provides a foundation for evaluating multilingual and culturally grounded medical reasoning in LLMs. The PersianMedQA dataset can be accessed at: https://huggingface.co/datasets/MohammadJRanjbar/PersianMedQA
>
---
#### [replaced 134] Tracking the Feature Dynamics in LLM Training: A Mechanistic Study
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17626v3](http://arxiv.org/pdf/2412.17626v3)**

> **作者:** Yang Xu; Yi Wang; Hengguan Huang; Hao Wang
>
> **摘要:** Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we (1) introduce SAE-Track, a novel method for efficiently obtaining a continual series of SAEs, providing the foundation for a mechanistic study that covers (2) the semantic evolution of features, (3) the underlying processes of feature formation, and (4) the directional drift of feature vectors. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution. For reproducibility, our code is available at https://github.com/Superposition09m/SAE-Track.
>
---
#### [replaced 135] Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09416v2](http://arxiv.org/pdf/2502.09416v2)**

> **作者:** Takumi Goto; Yusuke Sakai; Taro Watanabe
>
> **备注:** ACL 2025 (Main), 5 pages, 2 figures
>
> **摘要:** One of the goals of automatic evaluation metrics in grammatical error correction (GEC) is to rank GEC systems such that it matches human preferences. However, current automatic evaluations are based on procedures that diverge from human evaluation. Specifically, human evaluation derives rankings by aggregating sentence-level relative evaluation results, e.g., pairwise comparisons, using a rating algorithm, whereas automatic evaluation averages sentence-level absolute scores to obtain corpus-level scores, which are then sorted to determine rankings. In this study, we propose an aggregation method for existing automatic evaluation metrics which aligns with human evaluation methods to bridge this gap. We conducted experiments using various metrics, including edit-based metrics, n-gram based metrics, and sentence-level metrics, and show that resolving the gap improves results for the most of metrics on the SEEDA benchmark. We also found that even BERT-based metrics sometimes outperform the metrics of GPT-4. The proposed ranking method is integrated gec-metrics.
>
---
#### [replaced 136] Negation: A Pink Elephant in the Large Language Models' Room?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22395v2](http://arxiv.org/pdf/2503.22395v2)**

> **作者:** Tereza Vrabcová; Marek Kadlčík; Petr Sojka; Michal Štefánik; Michal Spiegel
>
> **摘要:** Negations are key to determining sentence meaning, making them essential for logical reasoning. Despite their importance, negations pose a substantial challenge for large language models (LLMs) and remain underexplored. We constructed and published two new textual entailment datasets NoFEVER-ML and NoSNLI-ML in four languages (English, Czech, German, and Ukrainian) with examples differing in negation. It allows investigation of the root causes of the negation problem and its exemplification: how popular LLM model properties and language impact their inability to handle negation correctly. Contrary to previous work, we show that increasing the model size may improve the models' ability to handle negations. Furthermore, we find that both the models' reasoning accuracy and robustness to negation are language-dependent and that the length and explicitness of the premise have an impact on robustness. There is better accuracy in projective language with fixed order, such as English, than in non-projective ones, such as German or Czech. Our entailment datasets pave the way to further research for explanation and exemplification of the negation problem, minimization of LLM hallucinations, and improvement of LLM reasoning in multilingual settings.
>
---
#### [replaced 137] Instruction-Following Pruning for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02086v3](http://arxiv.org/pdf/2501.02086v3)**

> **作者:** Bairu Hou; Qibin Chen; Jianyu Wang; Guoli Yin; Chong Wang; Nan Du; Ruoming Pang; Shiyu Chang; Tao Lei
>
> **备注:** ICML 2025
>
> **摘要:** With the rapid scaling of large language models (LLMs), structured pruning has become a widely used technique to learn efficient, smaller models from larger ones, delivering superior performance compared to training similarly sized models from scratch. In this paper, we move beyond the traditional static pruning approach of determining a fixed pruning mask for a model, and propose a dynamic approach to structured pruning. In our method, the pruning mask is input-dependent and adapts dynamically based on the information described in a user instruction. Our approach, termed "instruction-following pruning", introduces a sparse mask predictor that takes the user instruction as input and dynamically selects the most relevant model parameters for the given task. To identify and activate effective parameters, we jointly optimize the sparse mask predictor and the LLM, leveraging both instruction-following data and the pre-training corpus. Experimental results demonstrate the effectiveness of our approach on a wide range of evaluation benchmarks. For example, our 3B activated model improves over the 3B dense model by 5-8 points of absolute margin on domains such as math and coding, and rivals the performance of a 9B model.
>
---
#### [replaced 138] Gaussian mixture models as a proxy for interacting language models
- **分类: cs.CL; cs.LG; stat.ML; 62R07**

- **链接: [http://arxiv.org/pdf/2506.00077v2](http://arxiv.org/pdf/2506.00077v2)**

> **作者:** Edward L. Wang; Tianyu Wang; Avanti Athreya; Vince Lyzinski; Carey E. Priebe
>
> **摘要:** Large language models (LLMs) are a powerful tool with the ability to match human capabilities and behavior in many settings. Retrieval-augmented generation (RAG) further allows LLMs to generate diverse output depending on the contents of their RAG database. This motivates their use in the social sciences to study human behavior between individuals when large-scale experiments are infeasible. However, LLMs depend on complex, computationally expensive algorithms. In this paper, we introduce interacting Gaussian mixture models (GMMs) as an alternative to similar frameworks using LLMs. We compare a simplified model of GMMs to select experimental simulations of LLMs whose updating and response depend on feedback from other LLMs. We find that interacting GMMs capture important features of the dynamics in interacting LLMs, and we investigate key similarities and differences between interacting LLMs and GMMs. We conclude by discussing the benefits of Gaussian mixture models, potential modifications, and future research directions.
>
---
#### [replaced 139] Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.05037v2](http://arxiv.org/pdf/2503.05037v2)**

> **作者:** Mohsen Fayyaz; Ali Modarressi; Hinrich Schuetze; Nanyun Peng
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Dense retrieval models are commonly used in Information Retrieval (IR) applications, such as Retrieval-Augmented Generation (RAG). Since they often serve as the first step in these systems, their robustness is critical to avoid downstream failures. In this work, we repurpose a relation extraction dataset (e.g., Re-DocRED) to design controlled experiments that quantify the impact of heuristic biases, such as a preference for shorter documents, on retrievers like Dragon+ and Contriever. We uncover major vulnerabilities, showing retrievers favor shorter documents, early positions, repeated entities, and literal matches, all while ignoring the answer's presence! Notably, when multiple biases combine, models exhibit catastrophic performance degradation, selecting the answer-containing document in less than 10% of cases over a synthetic biased document without the answer. Furthermore, we show that these biases have direct consequences for downstream applications like RAG, where retrieval-preferred documents can mislead LLMs, resulting in a 34% performance drop than providing no documents at all. https://huggingface.co/datasets/mohsenfayyaz/ColDeR
>
---
#### [replaced 140] TACLR: A Scalable and Efficient Retrieval-based Method for Industrial Product Attribute Value Identification
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.03835v4](http://arxiv.org/pdf/2501.03835v4)**

> **作者:** Yindu Su; Huike Zou; Lin Sun; Ting Zhang; Haiyang Yang; Liyu Chen; David Lo; Qingheng Zhang; Shuguang Han; Jufeng Chen
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Product Attribute Value Identification (PAVI) involves identifying attribute values from product profiles, a key task for improving product search, recommendation, and business analytics on e-commerce platforms. However, existing PAVI methods face critical challenges, such as inferring implicit values, handling out-of-distribution (OOD) values, and producing normalized outputs. To address these limitations, we introduce Taxonomy-Aware Contrastive Learning Retrieval (TACLR), the first retrieval-based method for PAVI. TACLR formulates PAVI as an information retrieval task by encoding product profiles and candidate values into embeddings and retrieving values based on their similarity. It leverages contrastive training with taxonomy-aware hard negative sampling and employs adaptive inference with dynamic thresholds. TACLR offers three key advantages: (1) it effectively handles implicit and OOD values while producing normalized outputs; (2) it scales to thousands of categories, tens of thousands of attributes, and millions of values; and (3) it supports efficient inference for high-load industrial deployment. Extensive experiments on proprietary and public datasets validate the effectiveness and efficiency of TACLR. Further, it has been successfully deployed on the real-world e-commerce platform Xianyu, processing millions of product listings daily with frequently updated, large-scale attribute taxonomies. We release the code to facilitate reproducibility and future research at https://github.com/SuYindu/TACLR.
>
---
#### [replaced 141] Inference-time sparse attention with asymmetric indexing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.08246v2](http://arxiv.org/pdf/2502.08246v2)**

> **作者:** Pierre-Emmanuel Mazaré; Gergely Szilvasy; Maria Lomeli; Francisco Massa; Naila Murray; Hervé Jégou; Matthijs Douze
>
> **摘要:** Self-attention in transformer models is an incremental associative memory that maps key vectors to value vectors. One way to speed up self-attention is to employ GPU-compatible vector search algorithms based on standard partitioning methods such as k-means. However, such partitioning methods yield poor results in this context because (1) the keys and queries follow different distributions, and (2) the RoPE positional encoding hinders the bucket assignment. This paper introduces Saap (Self-Attention with Asymmetric Partitions), which overcomes these problems. It is an asymmetrical indexing technique that employs distinct partitions for keys and queries, thereby approximating self-attention with a data-adaptive sparsity pattern. It works on pretrained language models and only requires to train (offline) a small query classifier. On a long context Llama 3.1-8b model, with sequences ranging from 100k to 500k tokens, Saap typically reduces by a factor of 20 the fraction of memory that needs to be looked-up, which translates to a time saving of 60\% when compared to FlashAttention-v2.
>
---
#### [replaced 142] ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00095v2](http://arxiv.org/pdf/2506.00095v2)**

> **作者:** Yuchong Li; Xiaojun Zeng; Chihua Fang; Jian Yang; Fucang Jia; Lei Zhang
>
> **摘要:** Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage.
>
---
#### [replaced 143] GPR: Empowering Generation with Graph-Pretrained Retriever
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00261v2](http://arxiv.org/pdf/2506.00261v2)**

> **作者:** Xiaochen Wang; Zongyu Wu; Yuan Zhong; Xiang Zhang; Suhang Wang; Fenglong Ma
>
> **摘要:** Graph retrieval-augmented generation (GRAG) places high demands on graph-specific retrievers. However, existing retrievers often rely on language models pretrained on plain text, limiting their effectiveness due to domain misalignment and structure ignorance. To address these challenges, we propose GPR, a graph-based retriever pretrained directly on knowledge graphs. GPR aligns natural language questions with relevant subgraphs through LLM-guided graph augmentation and employs a structure-aware objective to learn fine-grained retrieval strategies. Experiments on two datasets, three LLM backbones, and five baselines show that GPR consistently improves both retrieval quality and downstream generation, demonstrating its effectiveness as a robust retrieval solution for GRAG.
>
---
#### [replaced 144] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00288v2](http://arxiv.org/pdf/2506.00288v2)**

> **作者:** Ahmed Elhady; Eneko Agirre; Mikel Artetxe
>
> **备注:** To appear in ACL 2025 Main
>
> **摘要:** Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future.
>
---
#### [replaced 145] We Should Chart an Atlas of All the World's Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10633v2](http://arxiv.org/pdf/2503.10633v2)**

> **作者:** Eliahu Horwitz; Nitzan Kurer; Jonathan Kahana; Liel Amar; Yedid Hoshen
>
> **备注:** Project page: https://horwitz.ai/model-atlas
>
> **摘要:** Public model repositories now contain millions of models, yet most models remain undocumented and effectively lost. In this position paper, we advocate for charting the world's model population in a unified structure we call the Model Atlas: a graph that captures models, their attributes, and the weight transformations that connect them. The Model Atlas enables applications in model forensics, meta-ML research, and model discovery, challenging tasks given today's unstructured model repositories. However, because most models lack documentation, large atlas regions remain uncharted. Addressing this gap motivates new machine learning methods that treat models themselves as data, inferring properties such as functionality, performance, and lineage directly from their weights. We argue that a scalable path forward is to bypass the unique parameter symmetries that plague model weights. Charting all the world's models will require a community effort, and we hope its broad utility will rally researchers toward this goal.
>
---
#### [replaced 146] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13928v2](http://arxiv.org/pdf/2502.13928v2)**

> **作者:** Shengguang Wu; Fan-Yun Sun; Kaiyue Wen; Nick Haber
>
> **备注:** Accepted to ACL 2025 Main. Project Website: https://s-vco.github.io/
>
> **摘要:** Recent studies have shown that Large Vision-Language Models (VLMs) tend to neglect image content and over-rely on language-model priors, resulting in errors in visually grounded tasks and hallucinations. We hypothesize that this issue arises because existing VLMs are not explicitly trained to generate texts that are accurately grounded in fine-grained image details. To enhance visual feedback during VLM training, we propose S-VCO (Symmetrical Visual Contrastive Optimization), a novel finetuning objective that steers the model toward capturing important visual details and aligning them with corresponding text tokens. To further facilitate this detailed alignment, we introduce MVC, a paired image-text dataset built by automatically filtering and augmenting visual counterfactual data to challenge the model with hard contrastive cases involving Minimal Visual Contrasts. Experiments show that our method consistently improves VLM performance across diverse benchmarks covering various abilities and domains, achieving up to a 22% reduction in hallucinations, and significant gains in vision-centric and general tasks. Notably, these improvements become increasingly pronounced in benchmarks with higher visual dependency. In short, S-VCO offers a significant enhancement of VLM's visually-dependent task performance while retaining or even improving the model's general abilities. We opensource our code at https://s-vco.github.io/
>
---
#### [replaced 147] Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.02862v2](http://arxiv.org/pdf/2505.02862v2)**

> **作者:** Haoming Yang; Ke Ma; Xiaojun Jia; Yingfei Sun; Qianqian Xu; Qingming Huang
>
> **摘要:** Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.
>
---
#### [replaced 148] SHARP: Unlocking Interactive Hallucination via Stance Transfer in Role-Playing LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07965v5](http://arxiv.org/pdf/2411.07965v5)**

> **作者:** Chuyi Kong; Ziyang Luo; Hongzhan Lin; Zhiyuan Fan; Yaxin Fan; Yuxi Sun; Jing Ma
>
> **备注:** 28 pages, unfortunately accepted to findings with Meta 4, acknowledge and apologize to the reviewers and area chair who support our work in the discussion period
>
> **摘要:** The advanced role-playing capabilities of Large Language Models (LLMs) have enabled rich interactive scenarios, yet existing research in social interactions neglects hallucination while struggling with poor generalizability and implicit character fidelity judgments. To bridge this gap, motivated by human behaviour, we introduce a generalizable and explicit paradigm for uncovering interactive patterns of LLMs across diverse worldviews. Specifically, we first define interactive hallucination through stance transfer, then construct SHARP, a benchmark built by extracting relations from commonsense knowledge graphs and utilizing LLMs' inherent hallucination properties to simulate multi-role interactions. Extensive experiments confirm our paradigm's effectiveness and stability, examine the factors that influence these metrics, and challenge conventional hallucination mitigation solutions. More broadly, our work reveals a fundamental limitation in popular post-training methods for role-playing LLMs: the tendency to obscure knowledge beneath style, resulting in monotonous yet human-like behaviors - interactive hallucination.
>
---
#### [replaced 149] DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23001v2](http://arxiv.org/pdf/2505.23001v2)**

> **作者:** Yize Cheng; Wenxiao Wang; Mazda Moayeri; Soheil Feizi
>
> **摘要:** Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.
>
---
