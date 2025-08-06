# 自然语言处理 cs.CL

- **最新发布 76 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors
- **分类: cs.CL**

- **简介: 该论文旨在开发对抗样本检测方法，解决大型语言模型（LLM）受到攻击的风险问题。通过分析上下文共现矩阵与张量的潜空间特征，提出CoCoTen方法，有效识别并降低攻击概率，实现显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.02997v1](http://arxiv.org/pdf/2508.02997v1)**

> **作者:** Sri Durga Sai Sowmya Kadali; Evangelos E. Papalexakis
>
> **摘要:** The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models. To support future research and reproducibility, we have made our implementation publicly available.
>
---
#### [new 002] Probing Syntax in Large Language Models: Successes and Remaining Challenges
- **分类: cs.CL**

- **简介: 该论文旨在探讨结构化探针在大型语言模型中的应用效果，解决结构性和统计因素对语义表示的影响问题，并通过三个控制实验验证其局限性，为改进评估方法提供依据。**

- **链接: [http://arxiv.org/pdf/2508.03211v1](http://arxiv.org/pdf/2508.03211v1)**

> **作者:** Pablo J. Diego-Simón; Emmanuel Chemla; Jean-Rémi King; Yair Lakretz
>
> **摘要:** The syntactic structures of sentences can be readily read-out from the activations of large language models (LLMs). However, the ``structural probes'' that have been developed to reveal this phenomenon are typically evaluated on an indiscriminate set of sentences. Consequently, it remains unclear whether structural and/or statistical factors systematically affect these syntactic representations. To address this issue, we conduct an in-depth analysis of structural probes on three controlled benchmarks. Our results are three-fold. First, structural probes are biased by a superficial property: the closer two words are in a sentence, the more likely structural probes will consider them as syntactically linked. Second, structural probes are challenged by linguistic properties: they poorly represent deep syntactic structures, and get interfered by interacting nouns or ungrammatical verb forms. Third, structural probes do not appear to be affected by the predictability of individual words. Overall, this work sheds light on the current challenges faced by structural probes. Providing a benchmark made of controlled stimuli to better evaluate their performance.
>
---
#### [new 003] Do language models accommodate their users? A study of linguistic convergence
- **分类: cs.CL**

- **简介: 该论文旨在探讨大型语言模型（LLMs）在生成语言时是否呈现与人类相似的使用模式，通过比较不同模型与人类对话风格的相似性及风格特征，验证语言收敛现象并揭示模型与人类行为机制的差异。**

- **链接: [http://arxiv.org/pdf/2508.03276v1](http://arxiv.org/pdf/2508.03276v1)**

> **作者:** Terra Blevins; Susanne Schmalwieser; Benjamin Roth
>
> **摘要:** While large language models (LLMs) are generally considered proficient in generating language, how similar their language usage is to that of humans remains understudied. In this paper, we test whether models exhibit linguistic convergence, a core pragmatic element of human language communication, asking: do models adapt, or converge, to the linguistic patterns of their user? To answer this, we systematically compare model completions of exisiting dialogues to the original human responses across sixteen language models, three dialogue corpora, and a variety of stylometric features. We find that models strongly converge to the conversation's style, often significantly overfitting relative to the human baseline. While convergence patterns are often feature-specific, we observe consistent shifts in convergence across modeling settings, with instruction-tuned and larger models converging less than their pretrained counterparts. Given the differences between human and model convergence patterns, we hypothesize that the underlying mechanisms for these behaviors are very different.
>
---
#### [new 004] Can LLMs Generate High-Quality Task-Specific Conversations?
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了大型语言模型如何通过参数化控制提升任务特定对话质量，解决了话题连贯性、知识进展、角色一致性等关键问题，提出了标准化方法并进行了实验验证。**

- **链接: [http://arxiv.org/pdf/2508.02931v1](http://arxiv.org/pdf/2508.02931v1)**

> **作者:** Shengqi Li; Amarnath Gupta
>
> **摘要:** This paper introduces a parameterization framework for controlling conversation quality in large language models. We explore nine key parameters across six dimensions that enable precise specification of dialogue properties. Through experiments with state-of-the-art LLMs, we demonstrate that parameter-based control produces statistically significant differences in generated conversation properties. Our approach addresses challenges in conversation generation, including topic coherence, knowledge progression, character consistency, and control granularity. The framework provides a standardized method for conversation quality control with applications in education, therapy, customer service, and entertainment. Future work will focus on implementing additional parameters through architectural modifications and developing benchmark datasets for evaluation.
>
---
#### [new 005] Somatic in the East, Psychological in the West?: Investigating Clinically-Grounded Cross-Cultural Depression Symptom Expression in LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文探讨大语言模型（LLMs）在不同文化背景下对抑郁症症状表达的跨文化模式，旨在验证其能否模仿西方（心理症状）与东方（身体症状）的表达差异。研究通过实验分析发现，LLMs在英语提示下未能有效捕捉文化差异，但提示性语言（如中文、日语等）可改善效果，揭示了低敏感度及文化惯性机制。任务为评估LLMs的文化适应能力，解决如何提升其跨文化理解，工作包括多语言实验与文化因素分析。**

- **链接: [http://arxiv.org/pdf/2508.03247v1](http://arxiv.org/pdf/2508.03247v1)**

> **作者:** Shintaro Sakai; Jisun An; Migyeong Kang; Haewoon Kwak
>
> **摘要:** Prior clinical psychology research shows that Western individuals with depression tend to report psychological symptoms, while Eastern individuals report somatic ones. We test whether Large Language Models (LLMs), which are increasingly used in mental health, reproduce these cultural patterns by prompting them with Western or Eastern personas. Results show that LLMs largely fail to replicate the patterns when prompted in English, though prompting in major Eastern languages (i.e., Chinese, Japanese, and Hindi) improves alignment in several configurations. Our analysis pinpoints two key reasons for this failure: the models' low sensitivity to cultural personas and a strong, culturally invariant symptom hierarchy that overrides cultural cues. These findings reveal that while prompt language is important, current general-purpose LLMs lack the robust, culture-aware capabilities essential for safe and effective mental health applications.
>
---
#### [new 006] Are We on the Right Way for Assessing Document Retrieval-Augmented Generation?
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文提出Double-Bench系统，旨在解决现有RAG评估体系缺乏真实数据验证的问题，通过构建多模态评估框架来改进模型性能，填补文本与视觉嵌入模型之间的差距并解决过自信偏差。**

- **链接: [http://arxiv.org/pdf/2508.03644v1](http://arxiv.org/pdf/2508.03644v1)**

> **作者:** Wenxuan Shen; Mingjia Wang; Yaochen Wang; Dongping Chen; Junjie Yang; Yao Wan; Weiwei Lin
>
> **备注:** In submission. Project website: https://double-bench.github.io/
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems using Multimodal Large Language Models (MLLMs) show great promise for complex document understanding, yet their development is critically hampered by inadequate evaluation. Current benchmarks often focus on specific part of document RAG system and use synthetic data with incomplete ground truth and evidence labels, therefore failing to reflect real-world bottlenecks and challenges. To overcome these limitations, we introduce Double-Bench: a new large-scale, multilingual, and multimodal evaluation system that is able to produce fine-grained assessment to each component within document RAG systems. It comprises 3,276 documents (72,880 pages) and 5,168 single- and multi-hop queries across 6 languages and 4 document types with streamlined dynamic update support for potential data contamination issues. Queries are grounded in exhaustively scanned evidence pages and verified by human experts to ensure maximum quality and completeness. Our comprehensive experiments across 9 state-of-the-art embedding models, 4 MLLMs and 4 end-to-end document RAG frameworks demonstrate the gap between text and visual embedding models is narrowing, highlighting the need in building stronger document retrieval models. Our findings also reveal the over-confidence dilemma within current document RAG frameworks that tend to provide answer even without evidence support. We hope our fully open-source Double-Bench provide a rigorous foundation for future research in advanced document RAG systems. We plan to retrieve timely corpus and release new benchmarks on an annual basis.
>
---
#### [new 007] FilBench: Can LLMs Understand and Generate Filipino?
- **分类: cs.CL**

- **简介: The paper describes FilBench, a Filipino-centric benchmark to evaluate LLMs across multiple languages. It addresses gaps in existing studies by curating tasks related to Philippine NLP priorities (e.g., Cultural Knowledge, Classical NLP). By testing 27 models, it finds that current systems struggle with reading comprehension and translation, highlighting the need for language-specific benchmarks.**

- **链接: [http://arxiv.org/pdf/2508.03523v1](http://arxiv.org/pdf/2508.03523v1)**

> **作者:** Lester James V. Miranda; Elyanah Aco; Conner Manuel; Jan Christian Blaise Cruz; Joseph Marvin Imperial
>
> **摘要:** Despite the impressive performance of LLMs on English-based tasks, little is known about their capabilities in specific languages such as Filipino. In this work, we address this gap by introducing FilBench, a Filipino-centric benchmark designed to evaluate LLMs across a diverse set of tasks and capabilities in Filipino, Tagalog, and Cebuano. We carefully curate the tasks in FilBench to reflect the priorities and trends of NLP research in the Philippines such as Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation. By evaluating 27 state-of-the-art LLMs on FilBench, we find that several LLMs suffer from reading comprehension and translation capabilities. Our results indicate that FilBench is challenging, with the best model, GPT-4o, achieving only a score of 72.23%. Moreover, we also find that models trained specifically for Southeast Asian languages tend to underperform on FilBench, with the highest-performing model, SEA-LION v3 70B, achieving only a score of 61.07%. Our work demonstrates the value of curating language-specific LLM benchmarks to aid in driving progress on Filipino NLP and increasing the inclusion of Philippine languages in LLM development.
>
---
#### [new 008] CF-RAG: A Dataset and Method for Carbon Footprint QA Using Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在构建一个针对碳足迹问答任务的开放数据集（CarbonPDF）和LLM技术（CarbonPDF），解决PDF文档中复杂结构和数据不一致性带来的分析难题。通过改进GPT-4o模型，研究者开发了专门针对此类问题的解决方案。**

- **链接: [http://arxiv.org/pdf/2508.03489v1](http://arxiv.org/pdf/2508.03489v1)**

> **作者:** Kaiwen Zhao; Bharathan Balaji; Stephen Lee
>
> **摘要:** Product sustainability reports provide valuable insights into the environmental impacts of a product and are often distributed in PDF format. These reports often include a combination of tables and text, which complicates their analysis. The lack of standardization and the variability in reporting formats further exacerbate the difficulty of extracting and interpreting relevant information from large volumes of documents. In this paper, we tackle the challenge of answering questions related to carbon footprints within sustainability reports available in PDF format. Unlike previous approaches, our focus is on addressing the difficulties posed by the unstructured and inconsistent nature of text extracted from PDF parsing. To facilitate this analysis, we introduce CarbonPDF-QA, an open-source dataset containing question-answer pairs for 1735 product report documents, along with human-annotated answers. Our analysis shows that GPT-4o struggles to answer questions with data inconsistencies. To address this limitation, we propose CarbonPDF, an LLM-based technique specifically designed to answer carbon footprint questions on such datasets. We develop CarbonPDF by fine-tuning Llama 3 with our training data. Our results show that our technique outperforms current state-of-the-art techniques, including question-answering (QA) systems finetuned on table and text data.
>
---
#### [new 009] Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking for Complex Instruction Following
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在解决LLMs在复杂指令执行中的推理不一致问题，通过生成并筛选有效提示（硬/易/过）以及应用熵正则化与强化学习框架，提升其推理能力和指令遵循性，最终在多个基准测试中优于同类模型。**

- **链接: [http://arxiv.org/pdf/2508.03178v1](http://arxiv.org/pdf/2508.03178v1)**

> **作者:** Chenyang Wang; Liang Wen; Shousheng Jia; Xiangzheng Zhang; Liang Xu
>
> **备注:** 12 pages, 10 figures, 7 tables
>
> **摘要:** While advancements in the reasoning abilities of LLMs have significantly enhanced their performance in solving mathematical problems, coding tasks, and general puzzles, their effectiveness in accurately adhering to instructions remains inconsistent, particularly with more complex directives. Our investigation identifies lazy reasoning during the thinking stage as the primary factor contributing to poor instruction adherence. To mitigate this issue, we propose a comprehensive framework designed to enable rigorous reasoning processes involving preview and self-checking, essential for satisfying strict instruction constraints. Specifically, we first generate instructions with complex constraints and apply a filtering process to obtain valid prompts, resulting in three distinct prompt datasets categorized as hard, easy, and pass. Then, we employ rejection sampling on the pass prompts to curate a small yet high-quality dataset, enabling a cold-start initialization of the model and facilitating its adaptation to effective reasoning patterns. Subsequently, we employ an entropy-preserving supervised fine-tuning (Entropy-SFT) strategy coupled with token-wise entropy-adaptive (TEA-RL) reinforcement learning guided by rule-based dense rewards. This approach encourages the model to transform its reasoning mechanism, ultimately fostering generalizable reasoning abilities that encompass preview and self-checking. Extensive experiments conducted on instruction-following benchmarks demonstrate remarkable performance improvements across various model scales. Notably, our Light-IF-32B model surpasses both larger open-source models such as DeepSeek-R1 and closed-source models like Doubao-1.6.
>
---
#### [new 010] Modeling Annotator Disagreement with Demographic-Aware Experts and Synthetic Perspectives
- **分类: cs.CL**

- **简介: 该论文旨在解决主观NLP任务中的annotator分歧问题，提出DEMO-MoE模型通过分组专家网络与合成标注增强结构化差异表示，验证了零样本提示生成的合成数据对稀疏数据覆盖的适应性，并探索了数据融合策略，提升多视角任务的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.02853v1](http://arxiv.org/pdf/2508.02853v1)**

> **作者:** Yinuo Xu; Veronica Derricks; Allison Earl; David Jurgens
>
> **备注:** 28 pages, 17 figures
>
> **摘要:** We present an approach to modeling annotator disagreement in subjective NLP tasks through both architectural and data-centric innovations. Our model, DEM-MoE (Demographic-Aware Mixture of Experts), routes inputs to expert subnetworks based on annotator demographics, enabling it to better represent structured, group-level variation compared to prior models. DEM-MoE consistently performs competitively across demographic groups, and shows especially strong results on datasets with high annotator disagreement. To address sparse demographic coverage, we test whether LLM-generated synthetic annotations via zero-shot persona prompting can be used for data imputation. We show these synthetic judgments align moderately well with human annotations on our data and offer a scalable way to potentially enrich training data. We then propose and evaluate approaches for blending real and synthetic data using strategies tailored to dataset structure. We find that the optimal strategies depend on dataset structure. Together, these contributions improve the representation of diverse perspectives.
>
---
#### [new 011] Pay What LLM Wants: Can LLM Simulate Economics Experiment with 522 Real-human Persona?
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了大型语言模型（LLM）在经济模拟中的应用，通过实证研究验证其能否准确反映真实人类行为，解决了传统基于虚构数据的局限性。研究比较了三种LLM的方法，并分析了522名韩国消费者的真实行为数据的影响。**

- **链接: [http://arxiv.org/pdf/2508.03262v1](http://arxiv.org/pdf/2508.03262v1)**

> **作者:** Junhyuk Choi; Hyeonchu Park; Haemin Lee; Hyebeen Shin; Hyun Joung Jin; Bugeun Kim
>
> **备注:** Preprint
>
> **摘要:** Recent advances in Large Language Models (LLMs) have generated significant interest in their capacity to simulate human-like behaviors, yet most studies rely on fictional personas rather than actual human data. We address this limitation by evaluating LLMs' ability to predict individual economic decision-making using Pay-What-You-Want (PWYW) pricing experiments with real 522 human personas. Our study systematically compares three state-of-the-art multimodal LLMs using detailed persona information from 522 Korean participants in cultural consumption scenarios. We investigate whether LLMs can accurately replicate individual human choices and how persona injection methods affect prediction performance. Results reveal that while LLMs struggle with precise individual-level predictions, they demonstrate reasonable group-level behavioral tendencies. Also, we found that commonly adopted prompting techniques are not much better than naive prompting methods; reconstruction of personal narrative nor retrieval augmented generation have no significant gain against simple prompting method. We believe that these findings can provide the first comprehensive evaluation of LLMs' capabilities on simulating economic behavior using real human data, offering empirical guidance for persona-based simulation in computational social science.
>
---
#### [new 012] Tackling Distribution Shift in LLM via KILO: Knowledge-Instructed Learning for Continual Adaptation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种名为KILO的任务框架，旨在解决大型语言模型在跨领域适应中的性能退化问题，通过结合动态知识图谱与指令微调，在WikiText-103预训练基础上完成连续学习，验证其在BioASQ、SciQ等领域的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03571v1](http://arxiv.org/pdf/2508.03571v1)**

> **作者:** Iing Muttakhiroh; Thomas Fevens
>
> **摘要:** Large Language Models (LLMs) often suffer from performance degradation when faced with domain shifts, primarily due to catastrophic forgetting. In this work, we propose KILO (Knowledge-Instructed Learning for Continual Adaptation), a novel continual learning framework that integrates dynamic knowledge graphs with instruction tuning. By leveraging retrieved domain-specific knowledge as guidance during training, KILO enhances both adaptability to new domains and retention of previously acquired knowledge. We pretrain our model on WikiText-103 and evaluate sequential adaptation across four diverse target domains: BioASQ, SciQ, TweetEval, and MIND. Our experiments demonstrate that KILO consistently outperforms strong baselines, including continual fine-tuning, ERNIE 2.0, and CPT, in terms of backward transfer, forward transfer, F1 score, retention rate, and training efficiency. These results highlight the effectiveness of combining structured knowledge retrieval and instruction prompting to overcome domain shift challenges in continual learning scenarios.
>
---
#### [new 013] RCP-Merging: Merging Long Chain-of-Thought Models with Domain-Specific Models by Considering Reasoning Capability as Prior
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决长链思考模型与领域知识模型合并的资源效率问题，通过考虑推理能力作为优先级设计RCP-Merging框架，成功实现了模型性能提升（9.5%+）的同时保留原长链推理能力。**

- **链接: [http://arxiv.org/pdf/2508.03140v1](http://arxiv.org/pdf/2508.03140v1)**

> **作者:** Junyao Yang; Jianwei Wang; Huiping Zhuang; Cen Chen; Ziqian Zeng
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) with long chain-of-thought (CoT) capability, termed Reasoning Models, demonstrate superior intricate problem-solving abilities through multi-step long CoT reasoning. To create a dual-capability model with long CoT capability and domain-specific knowledge without substantial computational and data costs, model merging emerges as a highly resource-efficient method. However, significant challenges lie in merging domain-specific LLMs with long CoT ones since nowadays merging methods suffer from reasoning capability degradation, even gibberish output and output collapse. To overcome this, we introduce RCP-Merging: Merging Long Chain-of-Thought Models with Domain-Specific Models by Considering Reasoning Capability as Prior, a novel merging framework designed to integrate domain-specific LLMs with long CoT capability, meanwhile maintaining model performance in the original domain. Treating reasoning model weights as foundational prior, our method utilizes a reasoning capability indicator to preserve core long CoT capability model weights while selectively merging essential domain-specific weights. We conducted extensive experiments on Qwen2.5-7B, Llama3.1-8B, and Qwen2.5-1.5B models in BioMedicine and Finance domains. Our results show that RCP-Merging successfully merges a reasoning model with domain-specific ones, improving domain task performance by 9.5% and 9.2% over state-of-the-art methods, without significantly harming the original long CoT reasoning capability.
>
---
#### [new 014] CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决大型语言模型（LLMs）答案验证的准确性与鲁棒性问题，提出CompassVerifier模型并构建VeriferBench基准，通过多领域任务验证和异常检测提升验证效率。**

- **链接: [http://arxiv.org/pdf/2508.03686v1](http://arxiv.org/pdf/2508.03686v1)**

> **作者:** Shudong Liu; Hongwei Liu; Junnan Liu; Linchen Xiao; Songyang Gao; Chengqi Lyu; Yuzhe Gu; Wenwei Zhang; Derek F. Wong; Songyang Zhang; Kai Chen
>
> **备注:** Technical Report; 31 Pages
>
> **摘要:** Answer verification is crucial not only for evaluating large language models (LLMs) by matching their unstructured outputs against standard answers, but also serves as the reward model to guide LLM optimization. Most evaluation frameworks rely on regularized matching or employ general LLMs for answer verification, which demands extensive, repetitive customization for regex rules or evaluation prompts. Two fundamental limitations persist in current methodologies: 1) the absence of comprehensive benchmarks that systematically evaluate verification capabilities across different LLMs; and 2) the nascent stage of verifier development, where existing approaches lack both the robustness to handle complex edge cases and the generalizability across different domains. In this work, we develop CompassVerifier, an accurate and robust lightweight verifier model for evaluation and outcome reward. It demonstrates multi-domain competency spanning math, knowledge, and diverse reasoning tasks, with the capability to process various answer types, including multi-subproblems, formulas, and sequence answers, while effectively identifying abnormal/invalid responses. We introduce VerifierBench benchmark comprising model outputs collected from multiple data sources, augmented through manual analysis of metaerror patterns to enhance CompassVerifier. We anticipate that CompassVerifier and VerifierBench will facilitate answer verification, evaluation protocols, and reinforcement learning research. Code and dataset are available at https://github.com/open-compass/CompassVerifier.
>
---
#### [new 015] CTR-Sink: Attention Sink for Language Models in Click-Through Rate Prediction
- **分类: cs.CL**

- **简介: 该论文研究了点击率预测任务，解决了传统基于文本建模方法中行为序列结构与自然语言不匹配的问题，提出CTR-Sink框架，通过行为级注意力点动态调节、插入时间距离锚点及两阶段训练策略，提升模型对复杂行为关系的捕捉能力。**

- **链接: [http://arxiv.org/pdf/2508.03668v1](http://arxiv.org/pdf/2508.03668v1)**

> **作者:** Zixuan Li; Binzong Geng; Jing Xiong; Yong He; Yuxuan Hu; Jian Chen; Dingwei Chen; Xiyu Chang; Liang Zhang; Linjian Mo; Chengming Li; Chuan Yuan; Zhenan Sun
>
> **摘要:** Click-Through Rate (CTR) prediction, a core task in recommendation systems, estimates user click likelihood using historical behavioral data. Modeling user behavior sequences as text to leverage Language Models (LMs) for this task has gained traction, owing to LMs' strong semantic understanding and contextual modeling capabilities. However, a critical structural gap exists: user behavior sequences consist of discrete actions connected by semantically empty separators, differing fundamentally from the coherent natural language in LM pre-training. This mismatch causes semantic fragmentation, where LM attention scatters across irrelevant tokens instead of focusing on meaningful behavior boundaries and inter-behavior relationships, degrading prediction performance. To address this, we propose $\textit{CTR-Sink}$, a novel framework introducing behavior-level attention sinks tailored for recommendation scenarios. Inspired by attention sink theory, it constructs attention focus sinks and dynamically regulates attention aggregation via external information. Specifically, we insert sink tokens between consecutive behaviors, incorporating recommendation-specific signals such as temporal distance to serve as stable attention sinks. To enhance generality, we design a two-stage training strategy that explicitly guides LM attention toward sink tokens and a attention sink mechanism that amplifies inter-sink dependencies to better capture behavioral correlations. Experiments on one industrial dataset and two open-source datasets (MovieLens, Kuairec), alongside visualization results, validate the method's effectiveness across scenarios.
>
---
#### [new 016] Privacy-Aware Decoding: Mitigating Privacy Leakage of Large Language Models in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出隐私感知解码（PAD）技术，针对RAG中隐私泄漏问题，通过注入校准噪声并结合RDP进行隐私保护，有效降低敏感信息泄露，同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2508.03098v1](http://arxiv.org/pdf/2508.03098v1)**

> **作者:** Haoran Wang; Xiongxiao Xu; Baixiang Huang; Kai Shu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances the factual accuracy of large language models (LLMs) by conditioning outputs on external knowledge sources. However, when retrieval involves private or sensitive data, RAG systems are susceptible to extraction attacks that can leak confidential information through generated responses. We propose Privacy-Aware Decoding (PAD), a lightweight, inference-time defense that adaptively injects calibrated Gaussian noise into token logits during generation. PAD integrates confidence-based screening to selectively protect high-risk tokens, efficient sensitivity estimation to minimize unnecessary noise, and context-aware noise calibration to balance privacy with generation quality. A \renyi Differential Privacy (RDP) accountant rigorously tracks cumulative privacy loss, enabling explicit per-response $(\varepsilon, \delta)$-DP guarantees for sensitive outputs. Unlike prior approaches requiring retraining or corpus-level filtering, PAD is model-agnostic and operates entirely at decoding time with minimal computational overhead. Experiments on three real-world datasets demonstrate that PAD substantially reduces private information leakage while preserving response utility, outperforming existing retrieval- and post-processing-based defenses. Our work takes an important step toward mitigating privacy risks in RAG via decoding strategies, paving the way for universal and scalable privacy solutions in sensitive domains. Our code is available: https://github.com/wang2226/PAD.
>
---
#### [new 017] Marito: Structuring and Building Open Multilingual Terminologies for South African NLP
- **分类: cs.CL**

- **简介: 该论文旨在解决South Africa官方语言术语碎片化问题，通过系统整理、清洗与标准化开放数据集，构建Marito平台，用于提升英语-津巴布韦机器翻译精度，推动多语言NLP技术发展。**

- **链接: [http://arxiv.org/pdf/2508.03529v1](http://arxiv.org/pdf/2508.03529v1)**

> **作者:** Vukosi Marivate; Isheanesu Dzingirai; Fiskani Banda; Richard Lastrucci; Thapelo Sindane; Keabetswe Madumo; Kayode Olaleye; Abiodun Modupe; Unarine Netshifhefhe; Herkulaas Combrink; Mohlatlego Nakeng; Matome Ledwaba
>
> **备注:** Under Review
>
> **摘要:** The critical lack of structured terminological data for South Africa's official languages hampers progress in multilingual NLP, despite the existence of numerous government and academic terminology lists. These valuable assets remain fragmented and locked in non-machine-readable formats, rendering them unusable for computational research and development. \emph{Marito} addresses this challenge by systematically aggregating, cleaning, and standardising these scattered resources into open, interoperable datasets. We introduce the foundational \emph{Marito} dataset, released under the equitable, Africa-centered NOODL framework. To demonstrate its immediate utility, we integrate the terminology into a Retrieval-Augmented Generation (RAG) pipeline. Experiments show substantial improvements in the accuracy and domain-specific consistency of English-to-Tshivenda machine translation for large language models. \emph{Marito} provides a scalable foundation for developing robust and equitable NLP technologies, ensuring South Africa's rich linguistic diversity is represented in the digital age.
>
---
#### [new 018] Variety Is the Spice of Life: Detecting Misinformation with Dynamic Environmental Representations
- **分类: cs.CL; cs.SI**

- **简介: 该论文旨在解决社交媒体中虚假信息检测的动态环境适应问题，提出MISDER框架（包含LSTM、ODE、PT三种模型），通过动态环境表示与时序建模实现有效检测。**

- **链接: [http://arxiv.org/pdf/2508.03420v1](http://arxiv.org/pdf/2508.03420v1)**

> **作者:** Bing Wang; Ximing Li; Yiming Wang; Changchun Li; Jiaxu Cui; Renchu Guan; Bo Yang
>
> **备注:** Accepted by CIKM 2025. 11 pages, 4 figures. Code: https://github.com/wangbing1416/MISDER
>
> **摘要:** The proliferation of misinformation across diverse social media platforms has drawn significant attention from both academic and industrial communities due to its detrimental effects. Accordingly, automatically distinguishing misinformation, dubbed as Misinformation Detection (MD), has become an increasingly active research topic. The mainstream methods formulate MD as a static learning paradigm, which learns the mapping between the content, links, and propagation of news articles and the corresponding manual veracity labels. However, the static assumption is often violated, since in real-world scenarios, the veracity of news articles may vacillate within the dynamically evolving social environment. To tackle this problem, we propose a novel framework, namely Misinformation detection with Dynamic Environmental Representations (MISDER). The basic idea of MISDER lies in learning a social environmental representation for each period and employing a temporal model to predict the representation for future periods. In this work, we specify the temporal model as the LSTM model, continuous dynamics equation, and pre-trained dynamics system, suggesting three variants of MISDER, namely MISDER-LSTM, MISDER-ODE, and MISDER-PT, respectively. To evaluate the performance of MISDER, we compare it to various MD baselines across 2 prevalent datasets, and the experimental results can indicate the effectiveness of our proposed model.
>
---
#### [new 019] Can Large Vision-Language Models Understand Multimodal Sarcasm?
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究了多模态讽刺分析任务，旨在解决传统文本分析中的视觉理解不足和概念知识缺失问题。通过提出训练自由框架，整合物体提取与外部知识，提升LVLM在多模态讽刺场景下的解释能力。**

- **链接: [http://arxiv.org/pdf/2508.03654v1](http://arxiv.org/pdf/2508.03654v1)**

> **作者:** Xinyu Wang; Yue Zhang; Liqiang Jing
>
> **备注:** Accepted by CIKM 2025
>
> **摘要:** Sarcasm is a complex linguistic phenomenon that involves a disparity between literal and intended meanings, making it challenging for sentiment analysis and other emotion-sensitive tasks. While traditional sarcasm detection methods primarily focus on text, recent approaches have incorporated multimodal information. However, the application of Large Visual Language Models (LVLMs) in Multimodal Sarcasm Analysis (MSA) remains underexplored. In this paper, we evaluate LVLMs in MSA tasks, specifically focusing on Multimodal Sarcasm Detection and Multimodal Sarcasm Explanation. Through comprehensive experiments, we identify key limitations, such as insufficient visual understanding and a lack of conceptual knowledge. To address these issues, we propose a training-free framework that integrates in-depth object extraction and external conceptual knowledge to improve the model's ability to interpret and explain sarcasm in multimodal contexts. The experimental results on multiple models show the effectiveness of our proposed framework. The code is available at https://github.com/cp-cp/LVLM-MSA.
>
---
#### [new 020] When Algorithms Meet Artists: Topic Modeling the AI-Art Debate, 2013-2025
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文分析了AI艺术领域中艺术家对创作权、伦理等问题的关注，通过数据挖掘揭示技术术语对公众话语的边缘化功能，并提出Bertopic方法和多模态基准以促进更透明的AI创作讨论。**

- **链接: [http://arxiv.org/pdf/2508.03037v1](http://arxiv.org/pdf/2508.03037v1)**

> **作者:** Ariya Mukherjee-Gandhi; Oliver Muellerklein
>
> **备注:** 18 pages, 5 figures, 5 tables
>
> **摘要:** As generative AI continues to reshape artistic production and alternate modes of human expression, artists whose livelihoods are most directly affected have raised urgent concerns about consent, transparency, and the future of creative labor. However, the voices of artists are often marginalized in dominant public and scholarly discourse. This study presents a twelve-year analysis, from 2013 to 2025, of English-language discourse surrounding AI-generated art. It draws from 439 curated 500-word excerpts sampled from opinion articles, news reports, blogs, legal filings, and spoken-word transcripts. Through a reproducible methodology, we identify five stable thematic clusters and uncover a misalignment between artists' perceptions and prevailing media narratives. Our findings highlight how the use of technical jargon can function as a subtle form of gatekeeping, often sidelining the very issues artists deem most urgent. Our work provides a BERTopic-based methodology and a multimodal baseline for future research, alongside a clear call for deeper, transparency-driven engagement with artist perspectives in the evolving AI-creative landscape.
>
---
#### [new 021] Coherent Multimodal Reasoning with Iterative Self-Evaluation for Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文旨在解决跨模态推理任务中缺乏深度逻辑推理的问题，提出基于迭代自我评估的CMRF框架，通过分解问题、增强推理能力和验证逻辑一致性，显著提升了大型视觉语言模型（LVLMs）在复杂场景中的性能表现。**

- **链接: [http://arxiv.org/pdf/2508.02886v1](http://arxiv.org/pdf/2508.02886v1)**

> **作者:** Wenjie Luo; Ruocheng Li; Shanshan Zhu; Julian Perry
>
> **摘要:** Despite significant advancements, current large language models (LLMs) and vision-language models (LVLMs) continue to struggle with complex, multi-step, cross-modal common sense reasoning tasks, often exhibiting a lack of "deliberative thinking." They tend to rely on superficial associations rather than deep, chained inference, particularly when integrating visual information with abstract concepts. To address this, we propose the Coherent Multimodal Reasoning Framework (CMRF), a novel approach that enhances LVLMs' common sense reasoning capabilities through an iterative, self-evaluating inference mechanism. CMRF mimics human problem-solving by decomposing complex queries, generating step-by-step inferences, and self-correcting errors. Our framework integrates three key modules: a Reasoning Decomposition Unit (RDU) for breaking down problems into sub-questions, a Contextual Inference Engine (CIE) for contextual inference, and a Coherence Assessment Module (CAM) for evaluating logical consistency and confidence. Coupled with an Adaptive Iterative Refinement strategy, CMRF systematically refines its reasoning paths. Built upon LLaVA-1.6-34B and trained on a novel Multimodal Daily Activity Reasoning (MDAR) dataset, CMRF achieves state-of-the-art performance among open-source LVLMs on challenging benchmarks like VCR, A-OKVQA, and DailyLife-MRC. It attains an average accuracy of 69.4%, surpassing the best open-source baseline by +2.4 percentage points, with particular strength in complex reasoning scenarios. Extensive ablation studies and human evaluations confirm the critical contributions of each module and the effectiveness of iterative refinement in fostering more coherent and accurate reasoning.
>
---
#### [new 022] fact check AI at SemEval-2025 Task 7: Multilingual and Crosslingual Fact-checked Claim Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文旨在解决多语言和跨语言事实检查查询检索任务，通过Bi-Encoder模型结合预训练Transformer优化句子相似性，采用轻量级模型实现92%多语言成功@10及80%跨语成功@10，验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.03475v1](http://arxiv.org/pdf/2508.03475v1)**

> **作者:** Pranshu Rastogi
>
> **备注:** 7 pages, 6 tables. Code available at https://github.com/pranshurastogi29/SemEval-2025-ACL-Multi-and-Crosslingual-Retrieval-using-Bi-encoders
>
> **摘要:** SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval is approached as a Learning-to-Rank task using a bi-encoder model fine-tuned from a pre-trained transformer optimized for sentence similarity. Training used both the source languages and their English translations for multilingual retrieval and only English translations for cross-lingual retrieval. Using lightweight models with fewer than 500M parameters and training on Kaggle T4 GPUs, the method achieved 92% Success@10 in multilingual and 80% Success@10 in 5th in crosslingual and 10th in multilingual tracks.
>
---
#### [new 023] Merge-based syntax is mediated by distinct neurocognitive mechanisms: A clustering analysis of comprehension abilities in 84,000 individuals with language deficits across nine languages
- **分类: cs.CL**

- **简介: 该论文研究了Merge操作在不同语言和认知能力下的神经机制，探讨了其对语言复杂性理解的作用，并通过集群分析发现三种结构类型与语言障碍有关。**

- **链接: [http://arxiv.org/pdf/2508.02885v1](http://arxiv.org/pdf/2508.02885v1)**

> **作者:** Elliot Murphy; Rohan Venkatesh; Edward Khokhlovich; Andrey Vyshedskiy
>
> **摘要:** In the modern language sciences, the core computational operation of syntax, 'Merge', is defined as an operation that combines two linguistic units (e.g., 'brown', 'cat') to form a categorized structure ('brown cat', a Noun Phrase). This can then be further combined with additional linguistic units based on this categorial information, respecting non-associativity such that abstract grouping is respected. Some linguists have embraced the view that Merge is an elementary, indivisible operation that emerged in a single evolutionary step. From a neurocognitive standpoint, different mental objects constructed by Merge may be supported by distinct mechanisms: (1) simple command constructions (e.g., "eat apples"); (2) the merging of adjectives and nouns ("red boat"); and (3) the merging of nouns with spatial prepositions ("laptop behind the sofa"). Here, we systematically investigate participants' comprehension of sentences with increasing levels of syntactic complexity. Clustering analyses revealed behavioral evidence for three distinct structural types, which we discuss as potentially emerging at different developmental stages and subject to selective impairment. While a Merge-based syntax may still have emerged suddenly in evolutionary time, responsible for the structured symbolic turn our species took, different cognitive mechanisms seem to underwrite the processing of various types of Merge-based objects.
>
---
#### [new 024] Investigating Gender Bias in LLM-Generated Stories via Psychological Stereotypes
- **分类: cs.CL; cs.AI**

- **简介: The paper investigates gender bias in LLM-generated stories using psychological stereotypes (e.g., aggressiveness/gossiping) in open-ended narrative tasks. It creates a dataset to analyze how attributes influence model behavior, revealing that conditioning mitigates bias and gender stereotypes shape outcomes. The work highlights the importance of psychology-grounded evaluation of LLMs. (100 words)**

- **链接: [http://arxiv.org/pdf/2508.03292v1](http://arxiv.org/pdf/2508.03292v1)**

> **作者:** Shahed Masoudian; Gustavo Escobedo; Hannah Strauss; Markus Schedl
>
> **备注:** Under Review
>
> **摘要:** As Large Language Models (LLMs) are increasingly used across different applications, concerns about their potential to amplify gender biases in various tasks are rising. Prior research has often probed gender bias using explicit gender cues as counterfactual, or studied them in sentence completion and short question answering tasks. These formats might overlook more implicit forms of bias embedded in generative behavior of longer content. In this work, we investigate gender bias in LLMs using gender stereotypes studied in psychology (e.g., aggressiveness or gossiping) in an open-ended task of narrative generation. We introduce a novel dataset called StereoBias-Stories containing short stories either unconditioned or conditioned on (one, two, or six) random attributes from 25 psychological stereotypes and three task-related story endings. We analyze how the gender contribution in the overall story changes in response to these attributes and present three key findings: (1) While models, on average, are highly biased towards male in unconditioned prompts, conditioning on attributes independent from gender stereotypes mitigates this bias. (2) Combining multiple attributes associated with the same gender stereotype intensifies model behavior, with male ones amplifying bias and female ones alleviating it. (3) Model biases align with psychological ground-truth used for categorization, and alignment strength increases with model size. Together, these insights highlight the importance of psychology-grounded evaluation of LLMs.
>
---
#### [new 025] Cropping outperforms dropout as an augmentation strategy for training self-supervised text embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究了自监督文本嵌入训练中的增强策略比较，提出裁剪优于dropout的方法，解决了基于文本的模型性能优化问题，并验证了其在不同数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03453v1](http://arxiv.org/pdf/2508.03453v1)**

> **作者:** Rita González-Márquez; Philipp Berens; Dmitry Kobak
>
> **摘要:** Text embeddings, i.e. vector representations of entire texts, play an important role in many NLP applications, such as retrieval-augmented generation, sentiment analysis, clustering, or visualizing collections of texts for data exploration. Currently, top-performing embedding models are derived from pre-trained language models via extensive supervised fine-tuning using curated text pairs. This contrasts with computer vision, where self-supervised training based on data augmentations has demonstrated remarkable success. Here we systematically compare the two most well-known augmentation strategies for positive pair generation in contrastive learning of text embeddings. We assess embedding quality on MTEB and additional in-domain evaluations and show that cropping augmentation strongly outperforms the dropout-based approach. We find that on out-of-domain data, the quality of resulting embeddings is below the supervised SOTA models, but for in-domain data, self-supervised fine-tuning produces high-quality text embeddings after very short fine-tuning, sometimes only marginally below the supervised SOTA. Finally, we show that representation quality increases towards the last transformer layers, which undergo the largest change during fine-tuning; and that fine-tuning only those last layers is sufficient to reach similar embedding quality.
>
---
#### [new 026] Analyzing German Parliamentary Speeches: A Machine Learning Approach for Topic and Sentiment Classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文旨在通过机器学习分析德国议会28,000份公开演讲，探讨政治议题与情感变化，解决了政党角色演变及治理责任对话语境的影响问题，并验证了相关模型的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03181v1](http://arxiv.org/pdf/2508.03181v1)**

> **作者:** Lukas Pätz; Moritz Beyer; Jannik Späth; Lasse Bohlen; Patrick Zschech; Mathias Kraus; Julian Rosenberger
>
> **备注:** Accepted at 20th International Conference on Wirtschaftsinformatik (WI25); September 2025, M\"unster, Germany
>
> **摘要:** This study investigates political discourse in the German parliament, the Bundestag, by analyzing approximately 28,000 parliamentary speeches from the last five years. Two machine learning models for topic and sentiment classification were developed and trained on a manually labeled dataset. The models showed strong classification performance, achieving an area under the receiver operating characteristic curve (AUROC) of 0.94 for topic classification (average across topics) and 0.89 for sentiment classification. Both models were applied to assess topic trends and sentiment distributions across political parties and over time. The analysis reveals remarkable relationships between parties and their role in parliament. In particular, a change in style can be observed for parties moving from government to opposition. While ideological positions matter, governing responsibilities also shape discourse. The analysis directly addresses key questions about the evolution of topics, sentiment dynamics, and party-specific discourse strategies in the Bundestag.
>
---
#### [new 027] Highlight & Summarize: RAG without the jailbreaks
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出了一种无需暴露用户输入的RAG设计模式（H&S），旨在防止模型劫持攻击，通过将标准RAG任务拆分为提示提取与总结两步实现安全生成，显著提升模型输出质量。**

- **链接: [http://arxiv.org/pdf/2508.02872v1](http://arxiv.org/pdf/2508.02872v1)**

> **作者:** Giovanni Cherubin; Andrew Paverd
>
> **摘要:** Preventing jailbreaking and model hijacking of Large Language Models (LLMs) is an important yet challenging task. For example, when interacting with a chatbot, malicious users can input specially crafted prompts to cause the LLM to generate undesirable content or perform a completely different task from its intended purpose. Existing mitigations for such attacks typically rely on hardening the LLM's system prompt or using a content classifier trained to detect undesirable content or off-topic conversations. However, these probabilistic approaches are relatively easy to bypass due to the very large space of possible inputs and undesirable outputs. In this paper, we present and evaluate Highlight & Summarize (H&S), a new design pattern for retrieval-augmented generation (RAG) systems that prevents these attacks by design. The core idea is to perform the same task as a standard RAG pipeline (i.e., to provide natural language answers to questions, based on relevant sources) without ever revealing the user's question to the generative LLM. This is achieved by splitting the pipeline into two components: a highlighter, which takes the user's question and extracts relevant passages ("highlights") from the retrieved documents, and a summarizer, which takes the highlighted passages and summarizes them into a cohesive answer. We describe several possible instantiations of H&S and evaluate their generated responses in terms of correctness, relevance, and response quality. Surprisingly, when using an LLM-based highlighter, the majority of H&S responses are judged to be better than those of a standard RAG pipeline.
>
---
#### [new 028] Beyond Content: How Grammatical Gender Shapes Visual Representation in Text-to-Image Models
- **分类: cs.CL**

- **简介: 该论文研究语法性别对文本到图像（T2I）模型视觉表示的影响，旨在揭示语言结构如何塑造AI生成图像。通过跨语言基准实验发现，语法性别显著改变图像生成效果：男性词汇增加73%的男性表示，女性词汇增加38%的女性表示。研究展示了语言结构对视觉输出的直接影响，提出新维度理解多语言系统偏见。**

- **链接: [http://arxiv.org/pdf/2508.03199v1](http://arxiv.org/pdf/2508.03199v1)**

> **作者:** Muhammed Saeed; Shaina Raza; Ashmal Vayani; Muhammad Abdul-Mageed; Ali Emami; Shady Shehata
>
> **摘要:** Research on bias in Text-to-Image (T2I) models has primarily focused on demographic representation and stereotypical attributes, overlooking a fundamental question: how does grammatical gender influence visual representation across languages? We introduce a cross-linguistic benchmark examining words where grammatical gender contradicts stereotypical gender associations (e.g., ``une sentinelle'' - grammatically feminine in French but referring to the stereotypically masculine concept ``guard''). Our dataset spans five gendered languages (French, Spanish, German, Italian, Russian) and two gender-neutral control languages (English, Chinese), comprising 800 unique prompts that generated 28,800 images across three state-of-the-art T2I models. Our analysis reveals that grammatical gender dramatically influences image generation: masculine grammatical markers increase male representation to 73\% on average (compared to 22\% with gender-neutral English), while feminine grammatical markers increase female representation to 38\% (compared to 28\% in English). These effects vary systematically by language resource availability and model architecture, with high-resource languages showing stronger effects. Our findings establish that language structure itself, not just content, shapes AI-generated visual outputs, introducing a new dimension for understanding bias and fairness in multilingual, multimodal systems.
>
---
#### [new 029] Towards Trustworthy Multimodal Moderation via Policy-Aligned Reasoning and Hierarchical Labeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出了一种基于政策引导的多模态内容过滤框架Hi-Guard，解决传统基于噪声标签的学习无法满足安全合规要求的问题。通过分层管道（轻量级模型过滤安全内容，强模型进行细粒度风险分类）和层级分类树结构，结合规则定义的模型提示及多级软掩码奖励优化，实现了更高的准确性、解释性和可追溯性，推动可信内容安全系统的构建。**

- **链接: [http://arxiv.org/pdf/2508.03296v1](http://arxiv.org/pdf/2508.03296v1)**

> **作者:** Anqi Li; Wenwei Jin; Jintao Tong; Pengda Qin; Weijia Li; Guo Lu
>
> **摘要:** Social platforms have revolutionized information sharing, but also accelerated the dissemination of harmful and policy-violating content. To ensure safety and compliance at scale, moderation systems must go beyond efficiency and offer accuracy and interpretability. However, current approaches largely rely on noisy, label-driven learning, lacking alignment with moderation rules and producing opaque decisions that hinder human review. Therefore, we propose Hierarchical Guard (Hi-Guard), a multimodal moderation framework that introduces a new policy-aligned decision paradigm. The term "Hierarchical" reflects two key aspects of our system design: (1) a hierarchical moderation pipeline, where a lightweight binary model first filters safe content and a stronger model handles fine-grained risk classification; and (2) a hierarchical taxonomy in the second stage, where the model performs path-based classification over a hierarchical taxonomy ranging from coarse to fine-grained levels. To ensure alignment with evolving moderation policies, Hi-Guard directly incorporates rule definitions into the model prompt. To further enhance structured prediction and reasoning, we introduce a multi-level soft-margin reward and optimize with Group Relative Policy Optimization (GRPO), penalizing semantically adjacent misclassifications and improving explanation quality. Extensive experiments and real-world deployment demonstrate that Hi-Guard achieves superior classification accuracy, generalization, and interpretability, paving the way toward scalable, transparent, and trustworthy content safety systems. Code is available at: https://github.com/lianqi1008/Hi-Guard.
>
---
#### [new 030] Thinking with Nothinking Calibration: A New In-Context Learning Paradigm in Reasoning Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出了一种新型的在-context学习范式，旨在解决传统RLLMs在多步推理中的不确定性问题。通过构建两种推理模式（Thinking和Nothinking），并结合单次提示进行优化，有效提升了推理精度和泛化能力，实现了比现有方法更优的性能表现。**

- **链接: [http://arxiv.org/pdf/2508.03363v1](http://arxiv.org/pdf/2508.03363v1)**

> **作者:** Haotian Wu; Bo Xu; Yao Shu; Menglin Yang; Chengwei Qin
>
> **摘要:** Reasoning large language models (RLLMs) have recently demonstrated remarkable capabilities through structured and multi-step reasoning. While prior research has primarily focused on improving their training and inference strategies, their potential for in-context learning (ICL) remains largely underexplored. To fill this gap, we propose Thinking with Nothinking Calibration (JointThinking), a new ICL paradigm that leverages the structured difference between two reasoning modes, i.e., Thinking and Nothinking, to improve reasoning accuracy. Specifically, our method prompts the model to generate two answers in parallel: one in Thinking mode and the other in Nothinking mode. A second round of Thinking is triggered only when the two initial responses are inconsistent, using a single prompt that incorporates the original question and both candidate answers. Since such disagreement occurs infrequently (e.g., only 6\% in GSM8K), our method performs just one round of reasoning in most cases, resulting in minimal latency overhead. Extensive experiments across multiple reasoning benchmarks demonstrate that JointThinking significantly outperforms few-shot chain-of-thought (CoT) and majority voting with improved answer robustness. Moreover, It achieves comparable in-distribution performance to training-based SOTA method, while substantially outperforming on out-of-distribution tasks. We further conduct a systematic analysis of the calibration mechanism, showing that leveraging different reasoning modes consistently lowers the error rate and highlights the value of structural thinking diversity. Additionally, we observe that the performance gap between actual and ideal reasoning narrows as model size increases in the second round of thinking, indicating the strong scalability of our approach. Finally, we discuss current limitations and outline promising directions for future ICL research in RLLMs.
>
---
#### [new 031] UPLME: Uncertainty-Aware Probabilistic Language Modelling for Robust Empathy Regression
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出了一种基于不确定性感知的语义模型（UPLME），旨在解决情感回归任务中的噪声自报告评分问题。通过结合概率语言模型与贝叶斯变分推断，在保持预测一致性的同时捕获不确定性，设计了两项新型损失以优化预测效果。实验表明，UPLME在噪声样本分离和性能提升方面优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.03520v1](http://arxiv.org/pdf/2508.03520v1)**

> **作者:** Md Rakibul Hasan; Md Zakir Hossain; Aneesh Krishna; Shafin Rahman; Tom Gedeon
>
> **备注:** Code available at https://github.com/hasan-rakibul/UPLME
>
> **摘要:** Supervised learning for empathy regression is challenged by noisy self-reported empathy scores. While many algorithms have been proposed for learning with noisy labels in textual classification problems, the regression counterpart is relatively under-explored. We propose UPLME, an uncertainty-aware probabilistic language modelling framework to capture label noise in the regression setting of empathy detection. UPLME includes a probabilistic language model that predicts both empathy score and heteroscedastic uncertainty and is trained using Bayesian concepts with variational model ensembling. We further introduce two novel loss components: one penalises degenerate Uncertainty Quantification (UQ), and another enforces the similarity between the input pairs on which we predict empathy. UPLME provides state-of-the-art performance (Pearson Correlation Coefficient: $0.558\rightarrow0.580$ and $0.629\rightarrow0.634$) in terms of the performance reported in the literature in two public benchmarks, having label noise. Through synthetic label noise injection, we show that UPLME is effective in separating noisy and clean samples based on the predicted uncertainty. UPLME further outperform (Calibration error: $0.571\rightarrow0.376$) a recent variational model ensembling-based UQ method designed for regression problems.
>
---
#### [new 032] EmbedGrad: Gradient-Based Prompt Optimization in Embedding Space for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出 EmbedGrad，解决大型语言模型在多样化任务中的提示嵌入优化问题，通过梯度驱动的嵌入细化技术，分离训练与部署，提升推理能力并增强数学推理等复杂任务的准确性。**

- **链接: [http://arxiv.org/pdf/2508.03533v1](http://arxiv.org/pdf/2508.03533v1)**

> **作者:** Xiaoming Hou; Jiquan Zhang; Zibin Lin; DaCheng Tao; Shengli Zhang
>
> **摘要:** Effectively adapting powerful pretrained foundation models to diverse tasks remains a key challenge in AI deployment. Current approaches primarily follow two paradigms:discrete optimization of text prompts through prompt engineering, or continuous adaptation via additional trainable parameters. Both exhibit limitations-discrete methods lack refinement precision while parameter-based techniques increase complexity and reduce interpretability. To address these constraints, we propose EmbedGrad, a novel framework that optimizes text prompt embeddings through gradient-based refinement. Our approach uniquely decouples training from deployment:during optimization,labeled examples guide precise embedding adjustments while preserving semantic meaning; during inference, only optimized embeddings integrate with user queries. This enables fine-grained calibration impossible in text space, such as enhancing the reasoning capability of prompts like please reason step by step. Comprehensive evaluations across mathematical reasoning, sentiment analysis, and causal judgment tasks demonstrate EmbedGrad's effectiveness:optimizing this reasoning prompt for Qwen2.5-Math-1.5B increased accuracy from 14.74\% to 58.96\% on mathematical problems. Consistent improvements were observed across model scales (0.5B-14B) and all tasks, with particularly significant gains for smaller models on complex problems like causal judgment. By bridging prompt engineering and parameter efficiency without architectural changes, our work establishes embedding refinement as a powerful new paradigm for task adaptation.
>
---
#### [new 033] Clinically Grounded Agent-based Report Evaluation: An Interpretable Metric for Radiology Report Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在开发一种可解释的医学报告评估框架ICARE，解决自动影像报告生成中的可信度问题，通过大语言模型代理与多选题问答技术，量化临床精度与召回率，并验证其在临床研究中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.02808v1](http://arxiv.org/pdf/2508.02808v1)**

> **作者:** Radhika Dua; Young Joon; Kwon; Siddhant Dogra; Daniel Freedman; Diana Ruan; Motaz Nashawaty; Danielle Rigau; Daniel Alexander Alber; Kang Zhang; Kyunghyun Cho; Eric Karl Oermann
>
> **摘要:** Radiological imaging is central to diagnosis, treatment planning, and clinical decision-making. Vision-language foundation models have spurred interest in automated radiology report generation (RRG), but safe deployment requires reliable clinical evaluation of generated reports. Existing metrics often rely on surface-level similarity or behave as black boxes, lacking interpretability. We introduce ICARE (Interpretable and Clinically-grounded Agent-based Report Evaluation), an interpretable evaluation framework leveraging large language model agents and dynamic multiple-choice question answering (MCQA). Two agents, each with either the ground-truth or generated report, generate clinically meaningful questions and quiz each other. Agreement on answers captures preservation and consistency of findings, serving as interpretable proxies for clinical precision and recall. By linking scores to question-answer pairs, ICARE enables transparent, and interpretable assessment. Clinician studies show ICARE aligns significantly more with expert judgment than prior metrics. Perturbation analyses confirm sensitivity to clinical content and reproducibility, while model comparisons reveal interpretable error patterns.
>
---
#### [new 034] Exploring Stability-Plasticity Trade-offs for Continual Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文旨在探索持续命名实体识别（CNER）中的稳定与可塑性平衡问题。通过提出稳定性-可塑性对冲方法（SPT），结合多模态聚合与动态权重融合技术，解决了传统KD方法因过度稳定导致的知识损失问题，并开发了伪标签机制以应对非实体类型的语义迁移挑战。**

- **链接: [http://arxiv.org/pdf/2508.03259v1](http://arxiv.org/pdf/2508.03259v1)**

> **作者:** Duzhen Zhang; Chenxing Li; Jiahua Dong; Qi Liu; Dong Yu
>
> **备注:** Accepted by IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** Continual Named Entity Recognition (CNER) is an evolving field that focuses on sequentially updating an existing model to incorporate new entity types. Previous CNER methods primarily utilize Knowledge Distillation (KD) to preserve prior knowledge and overcome catastrophic forgetting, strictly ensuring that the representations of old and new models remain consistent. Consequently, they often impart the model with excessive stability (i.e., retention of old knowledge) but limited plasticity (i.e., acquisition of new knowledge). To address this issue, we propose a Stability-Plasticity Trade-off (SPT) method for CNER that balances these aspects from both representation and weight perspectives. From the representation perspective, we introduce a pooling operation into the original KD, permitting a level of plasticity by consolidating representation dimensions. From the weight perspective, we dynamically merge the weights of old and new models, strengthening old knowledge while maintaining new knowledge. During this fusion, we implement a weight-guided selective mechanism to prioritize significant weights. Moreover, we develop a confidence-based pseudo-labeling approach for the current non-entity type, which predicts entity types using the old model to handle the semantic shift of the non-entity type, a challenge specific to CNER that has largely been ignored by previous methods. Extensive experiments across ten CNER settings on three benchmark datasets demonstrate that our SPT method surpasses previous CNER approaches, highlighting its effectiveness in achieving a suitable stability-plasticity trade-off.
>
---
#### [new 035] Current State in Privacy-Preserving Text Preprocessing for Domain-Agnostic NLP
- **分类: cs.CL**

- **简介: 该论文探讨了隐私保护下的文本预处理技术，旨在解决领域无关NLP任务中提取敏感信息的问题，提出了多种预处理方法以增强数据安全性。**

- **链接: [http://arxiv.org/pdf/2508.03204v1](http://arxiv.org/pdf/2508.03204v1)**

> **作者:** Abhirup Sinha; Pritilata Saha; Tithi Saha
>
> **备注:** To be published in the Proceedings of Die Studierendenkonferenz Informatik (SKILL) 2024
>
> **摘要:** Privacy is a fundamental human right. Data privacy is protected by different regulations, such as GDPR. However, modern large language models require a huge amount of data to learn linguistic variations, and the data often contains private information. Research has shown that it is possible to extract private information from such language models. Thus, anonymizing such private and sensitive information is of utmost importance. While complete anonymization may not be possible, a number of different pre-processing approaches exist for masking or pseudonymizing private information in textual data. This report focuses on a few of such approaches for domain-agnostic NLP tasks.
>
---
#### [new 036] NLP Methods May Actually Be Better Than Professors at Estimating Question Difficulty
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了大语言模型（LLMs）在估算考试题难度中的有效性，提出通过监督学习方法（仅用42个样本）比传统教授更准确地区分易难题，从而提升考试评估质量。**

- **链接: [http://arxiv.org/pdf/2508.03294v1](http://arxiv.org/pdf/2508.03294v1)**

> **作者:** Leonidas Zotos; Ivo Pascal de Jong; Matias Valdenegro-Toro; Andreea Ioana Sburlea; Malvina Nissim; Hedderik van Rijn
>
> **备注:** 10 pages, 2 figures, accepted at the 2nd International Workshop on AI in Society, Education and Educational Research (AISEER)
>
> **摘要:** Estimating the difficulty of exam questions is essential for developing good exams, but professors are not always good at this task. We compare various Large Language Model-based methods with three professors in their ability to estimate what percentage of students will give correct answers on True/False exam questions in the areas of Neural Networks and Machine Learning. Our results show that the professors have limited ability to distinguish between easy and difficult questions and that they are outperformed by directly asking Gemini 2.5 to solve this task. Yet, we obtained even better results using uncertainties of the LLMs solving the questions in a supervised learning setting, using only 42 training samples. We conclude that supervised learning using LLM uncertainty can help professors better estimate the difficulty of exam questions, improving the quality of assessment.
>
---
#### [new 037] FairLangProc: A Python package for fairness in NLP
- **分类: cs.CL; stat.ML; 68T50; I.2.7**

- **简介: 该论文旨在解决大型语言模型在组织正义和医疗等关键场景中的偏见问题，提出FairLangProc包提供统一的公平性实现框架，兼容Hugging Face库以促进技术的普及与应用。**

- **链接: [http://arxiv.org/pdf/2508.03677v1](http://arxiv.org/pdf/2508.03677v1)**

> **作者:** Arturo Pérez-Peralta; Sandra Benítez-Peña; Rosa E. Lillo
>
> **备注:** 40 pages, 4 figures, 3 tables
>
> **摘要:** The rise in usage of Large Language Models to near ubiquitousness in recent years has risen societal concern about their applications in decision-making contexts, such as organizational justice or healthcare. This, in turn, poses questions about the fairness of these models in critical settings, which leads to the developement of different procedures to address bias in Natural Language Processing. Although many datasets, metrics and algorithms have been proposed to measure and mitigate harmful prejudice in Natural Language Processing, their implementation is diverse and far from centralized. As a response, this paper presents FairLangProc, a comprehensive Python package providing a common implementation of some of the more recent advances in fairness in Natural Language Processing providing an interface compatible with the famous Hugging Face transformers library, aiming to encourage the widespread use and democratization of bias mitigation techniques. The implementation can be found on https://github.com/arturo-perez-peralta/FairLangProc.
>
---
#### [new 038] CTTS: Collective Test-Time Scaling
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了多代理与多奖励模型协作的集体测试时间缩放（CTTS），旨在突破单代理系统效率瓶颈。通过设计SA-MR、MA-SR和MA-MR三种模式，提出CTTS-MM框架，利用Agent Collaboration Search（ACS）和Mixture of Reword Models（MoR）优化多模态推理，验证其在主流基准上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.03333v1](http://arxiv.org/pdf/2508.03333v1)**

> **作者:** Zhende Song; Shengji Tang; Peng Ye; Jiayuan Fan; Tao Chen
>
> **摘要:** Test-time scaling (TTS) has emerged as a promising research field for enhancing the effectiveness of large language models (LLMs) without extra training. However, most existing approaches, e.g., Best-of-N and Self-Consistency rely on a single agent interacting with a reward model (SA-SR), constrained by limited capabilities of a single test-time scaling (STTS) paradigm. On the other hand, recent works demonstrate that collective-agent methods can break through the upper bound of single-agent systems by orchestrating diverse models. Thus, in this paper, we take a first step towards exploring Collective Test-Time Scaling (CTTS). Consider the different interaction types of single and multiple models, we design three primary paradigms to investigate the optimal paradigm of CTTS: (1) single agent to multiple reward models (SA-MR); (2) multiple agents to single reward model (MA-SR); and (3) multiple agents to multiple reward models (MA-MR). Extensive experiments demonstrate that MA-MR consistently achieves the best performance. Based on this, we propose a novel framework named CTTS-MM that effectively leverages both multi-agent and multi-reward-model collaboration for enhanced inference. Specifically, for multi-agent collaboration, we propose an Agent Collaboration Search (ACS), which searches for the most effective combination of LLM agents from a large candidate pool; for multi-reward-model collaboration, we propose Mixture of Reword Models (MoR), which consists of a curated question pool and a Prior Reward model Ensemble Selection (PRES) to select the optimal combinations of reward models via Pair-wise Reward Ranking (PRR) metric. Experiments across seven mainstream benchmarks demonstrate that the proposed CTTS-MM consistently obtains superior performance. Code will be released at https://github.com/magent4aci/CTTS-MM.
>
---
#### [new 039] Cross-lingual Opinions and Emotions Mining in Comparable Documents
- **分类: cs.CL; I.2.7**

- **简介: 该论文旨在研究多语言对比文档中的情感与意见分类差异，通过跨语言标注方法（结合WNA词汇）并进行统计分析，验证不同新闻机构文章的情感一致性，提出语言独立的标注模型以解决跨语言语料对比中的情感异质性问题。**

- **链接: [http://arxiv.org/pdf/2508.03112v1](http://arxiv.org/pdf/2508.03112v1)**

> **作者:** Motaz Saad; David Langlois; Kamel Smaili
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Comparable texts are topic-aligned documents in multiple languages that are not direct translations. They are valuable for understanding how a topic is discussed across languages. This research studies differences in sentiments and emotions across English-Arabic comparable documents. First, texts are annotated with sentiment and emotion labels. We apply a cross-lingual method to label documents with opinion classes (subjective/objective), avoiding reliance on machine translation. To annotate with emotions (anger, disgust, fear, joy, sadness, surprise), we manually translate the English WordNet-Affect (WNA) lexicon into Arabic, creating bilingual emotion lexicons used to label the comparable corpora. We then apply a statistical measure to assess the agreement of sentiments and emotions in each source-target document pair. This comparison is especially relevant when the documents originate from different sources. To our knowledge, this aspect has not been explored in prior literature. Our study includes English-Arabic document pairs from Euronews, BBC, and Al-Jazeera (JSC). Results show that sentiment and emotion annotations align when articles come from the same news agency and diverge when they come from different ones. The proposed method is language-independent and generalizable to other language pairs.
>
---
#### [new 040] RooseBERT: A New Deal For Political Language Modelling
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍了一种领域预训练的新型语言模型 RooseBERT，旨在解决政治语言分析中的隐含论证和复杂结构挑战。通过大规模政治语料训练，其在命名实体识别、情感分析等任务上显著优于通用语言模型，验证了领域自适应提升性能的效果。**

- **链接: [http://arxiv.org/pdf/2508.03250v1](http://arxiv.org/pdf/2508.03250v1)**

> **作者:** Deborah Dore; Elena Cabrio; Serena Villata
>
> **摘要:** The increasing amount of political debates and politics-related discussions calls for the definition of novel computational methods to automatically analyse such content with the final goal of lightening up political deliberation to citizens. However, the specificity of the political language and the argumentative form of these debates (employing hidden communication strategies and leveraging implicit arguments) make this task very challenging, even for current general-purpose pre-trained Language Models. To address this issue, we introduce a novel pre-trained Language Model for political discourse language called RooseBERT. Pre-training a language model on a specialised domain presents different technical and linguistic challenges, requiring extensive computational resources and large-scale data. RooseBERT has been trained on large political debate and speech corpora (8K debates, each composed of several sub-debates on different topics) in English. To evaluate its performances, we fine-tuned it on four downstream tasks related to political debate analysis, i.e., named entity recognition, sentiment analysis, argument component detection and classification, and argument relation prediction and classification. Our results demonstrate significant improvements over general-purpose Language Models on these four tasks, highlighting how domain-specific pre-training enhances performance in political debate analysis. We release the RooseBERT language model for the research community.
>
---
#### [new 041] CardiffNLP at CLEARS-2025: Prompting Large Language Models for Plain Language and Easy-to-Read Text Rewriting
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Llama-3与Gemma-3结合的方法，用于西班牙语文本转译任务（IberLEF 2025），解决如何使模型生成清晰易读文本的问题，获得第一、二名。**

- **链接: [http://arxiv.org/pdf/2508.03240v1](http://arxiv.org/pdf/2508.03240v1)**

> **作者:** Mutaz Ayesh; Nicolás Gutiérrez-Rolón; Fernando Alva-Manchego
>
> **摘要:** This paper details the CardiffNLP team's contribution to the CLEARS shared task on Spanish text adaptation, hosted by IberLEF 2025. The shared task contained two subtasks and the team submitted to both. Our team took an LLM-prompting approach with different prompt variations. While we initially experimented with LLaMA-3.2, we adopted Gemma-3 for our final submission, and landed third place in Subtask 1 and second place in Subtask 2. We detail our numerous prompt variations, examples, and experimental results.
>
---
#### [new 042] Taggus: An Automated Pipeline for the Extraction of Characters' Social Networks from Portuguese Fiction Literature
- **分类: cs.CL; cs.IR**

- **简介: 该论文旨在开发一个用于从葡萄牙文学中提取角色社交网络的自动化系统，解决现有NLP方法无法有效捕捉角色互动与关系的问题，通过结合POS标签和启发式算法实现性能提升，平均F1-Score达94.1%和75.9%。**

- **链接: [http://arxiv.org/pdf/2508.03358v1](http://arxiv.org/pdf/2508.03358v1)**

> **作者:** Tiago G Canário; Catarina Duarte; Flávio L. Pinheiro; João L. M. Pereira
>
> **备注:** 24 pages, 5 Figures, 4 Tables
>
> **摘要:** Automatically identifying characters and their interactions from fiction books is, arguably, a complex task that requires pipelines that leverage multiple Natural Language Processing (NLP) methods, such as Named Entity Recognition (NER) and Part-of-speech (POS) tagging. However, these methods are not optimized for the task that leads to the construction of Social Networks of Characters. Indeed, the currently available methods tend to underperform, especially in less-represented languages, due to a lack of manually annotated data for training. Here, we propose a pipeline, which we call Taggus, to extract social networks from literary fiction works in Portuguese. Our results show that compared to readily available State-of-the-Art tools -- off-the-shelf NER tools and Large Language Models (ChatGPT) -- the resulting pipeline, which uses POS tagging and a combination of heuristics, achieves satisfying results with an average F1-Score of $94.1\%$ in the task of identifying characters and solving for co-reference and $75.9\%$ in interaction detection. These represent, respectively, an increase of $50.7\%$ and $22.3\%$ on results achieved by the readily available State-of-the-Art tools. Further steps to improve results are outlined, such as solutions for detecting relationships between characters. Limitations on the size and scope of our testing samples are acknowledged. The Taggus pipeline is publicly available to encourage development in this field for the Portuguese language.2
>
---
#### [new 043] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations
- **分类: cs.CL**

- **简介: 该论文探讨了如何通过内部表示提升LLMs作为法官的对人类偏好的理解，解决了传统方法依赖最终层的局限性，提出了LAGER框架并验证其在数据选择和情感理解任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.03550v1](http://arxiv.org/pdf/2508.03550v1)**

> **作者:** Peng Lai; Jianjie Zheng; Sijie Cheng; Yun Chen; Peng Li; Yang Liu; Guanhua Chen
>
> **摘要:** The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using large language models, a paradigm known as "LLMas-a-judge." However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and taskrelevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a lightweight and efficient framework for enhancing LLM-as-a-Judge alignment with human scoring, via internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer scoretoken logits and computing the expected score from a softmax-based distribution, with the LLM backbone kept frozen. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the effectiveness of our method.
>
---
#### [new 044] LECTOR: LLM-Enhanced Concept-based Test-Oriented Repetition for Adaptive Spaced Learning
- **分类: cs.CL**

- **简介: 该论文提出LECTOR算法，用于测试导向学习场景（如语言考试），解决语义混淆与个性化适应问题。通过LLM分析语义并整合间隔重复原理，验证其显著提升语言考试成功率。**

- **链接: [http://arxiv.org/pdf/2508.03275v1](http://arxiv.org/pdf/2508.03275v1)**

> **作者:** Jiahao Zhao
>
> **备注:** 15 pages, 4 figures, 1 table
>
> **摘要:** Spaced repetition systems are fundamental to efficient learning and memory retention, but existing algorithms often struggle with semantic interference and personalized adaptation. We present LECTOR (\textbf{L}LM-\textbf{E}nhanced \textbf{C}oncept-based \textbf{T}est-\textbf{O}riented \textbf{R}epetition), a novel adaptive scheduling algorithm specifically designed for test-oriented learning scenarios, particularly language examinations where success rate is paramount. LECTOR leverages large language models for semantic analysis while incorporating personalized learning profiles, addressing the critical challenge of semantic confusion in vocabulary learning by utilizing LLM-powered semantic similarity assessment and integrating it with established spaced repetition principles. Our comprehensive evaluation against six baseline algorithms (SSP-MMC, SM2, HLR, FSRS, ANKI, THRESHOLD) across 100 simulated learners over 100 days demonstrates significant improvements: LECTOR achieves a 90.2\% success rate compared to 88.4\% for the best baseline (SSP-MMC), representing a 2.0\% relative improvement. The algorithm shows particular strength in handling semantically similar concepts, reducing confusion-induced errors while maintaining computational efficiency. Our results establish LECTOR as a promising direction for intelligent tutoring systems and adaptive learning platforms.
>
---
#### [new 045] More Than a Score: Probing the Impact of Prompt Specificity on LLM Code Generation
- **分类: cs.CL; cs.LG; cs.PL**

- **简介: 该论文探讨了提示特定性对LLM代码生成任务性能的影响，旨在解决为何LLMs在通用任务（如HumanEval）中表现优异但专业任务（如ParEval）不足的问题。通过引入PartialOrderEval并验证其对不同任务的提示详细度影响，研究揭示了I/O规格、边界情况和步骤分解等关键因素。**

- **链接: [http://arxiv.org/pdf/2508.03678v1](http://arxiv.org/pdf/2508.03678v1)**

> **作者:** Yangtian Zi; Harshitha Menon; Arjun Guha
>
> **摘要:** State-of-the-art Large Language Models (LLMs) achieve high pass@1 on general benchmarks like HumanEval but underperform on specialized suites such as ParEval. Is this due to LLMs missing domain knowledge or insufficient prompt detail is given? To answer this, we introduce PartialOrderEval, which augments any code generation benchmark with a partial order of prompts from minimal to maximally detailed. Applying it to HumanEval and both serial and OpenMP subsets of ParEval, we measure how pass@1 scales with prompt specificity. Our experiments with Llama-3.x and Qwen2.5-Coder demonstrate varying degrees of prompt sensitivity across different tasks, and a qualitative analysis highlights explicit I/O specifications, edge-case handling, and stepwise breakdowns as the key drivers of prompt detail improvement.
>
---
#### [new 046] Token-Level Precise Attack on RAG: Searching for the Best Alternatives to Mislead Generation
- **分类: cs.CL**

- **简介: 该论文属于技术研究任务，旨在改进检索增强生成（RAG）框架以抵御恶意攻击，解决其安全漏洞问题，通过Token-level精确攻击突破黑盒场景限制，实现更有效的防御机制。**

- **链接: [http://arxiv.org/pdf/2508.03110v1](http://arxiv.org/pdf/2508.03110v1)**

> **作者:** Zizhong Li; Haopeng Zhang; Jiawei Zhang
>
> **摘要:** While large language models (LLMs) have achieved remarkable success in providing trustworthy responses for knowledge-intensive tasks, they still face critical limitations such as hallucinations and outdated knowledge. To address these issues, the retrieval-augmented generation (RAG) framework enhances LLMs with access to external knowledge via a retriever, enabling more accurate and real-time outputs about the latest events. However, this integration brings new security vulnerabilities: the risk that malicious content in the external database can be retrieved and used to manipulate model outputs. Although prior work has explored attacks on RAG systems, existing approaches either rely heavily on access to the retriever or fail to jointly consider both retrieval and generation stages, limiting their effectiveness, particularly in black-box scenarios. To overcome these limitations, we propose Token-level Precise Attack on the RAG (TPARAG), a novel framework that targets both white-box and black-box RAG systems. TPARAG leverages a lightweight white-box LLM as an attacker to generate and iteratively optimize malicious passages at the token level, ensuring both retrievability and high attack success in generation. Extensive experiments on open-domain QA datasets demonstrate that TPARAG consistently outperforms previous approaches in retrieval-stage and end-to-end attack effectiveness. These results further reveal critical vulnerabilities in RAG pipelines and offer new insights into improving their robustness.
>
---
#### [new 047] SLIM-LLMs: Modeling of Style-Sensory Language RelationshipsThrough Low-Dimensional Representations
- **分类: cs.CL**

- **简介: 该论文旨在探索风格与感官语言的关系，利用低维特征（R4）建模非线性关联，提出SLIM-LLMs模型，通过降低参数效率提升性能。**

- **链接: [http://arxiv.org/pdf/2508.02901v1](http://arxiv.org/pdf/2508.02901v1)**

> **作者:** Osama Khalid; Sanvesh Srivastava; Padmini Srinivasan
>
> **摘要:** Sensorial language -- the language connected to our senses including vision, sound, touch, taste, smell, and interoception, plays a fundamental role in how we communicate experiences and perceptions. We explore the relationship between sensorial language and traditional stylistic features, like those measured by LIWC, using a novel Reduced-Rank Ridge Regression (R4) approach. We demonstrate that low-dimensional latent representations of LIWC features r = 24 effectively capture stylistic information for sensorial language prediction compared to the full feature set (r = 74). We introduce Stylometrically Lean Interpretable Models (SLIM-LLMs), which model non-linear relationships between these style dimensions. Evaluated across five genres, SLIM-LLMs with low-rank LIWC features match the performance of full-scale language models while reducing parameters by up to 80%.
>
---
#### [new 048] LLMs Have a Heart of Stone: Demystifying the Soft Thinking Ability of Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了大语言模型（LLM）的"软思考"能力，旨在解决现有模型依赖离散token限制表达能力的问题。通过分析模型内部行为，提出引入随机性策略（如Dirichlet采样和Gumbel-Softmax），证明其可提升软思考效果并优化推理空间。**

- **链接: [http://arxiv.org/pdf/2508.03440v1](http://arxiv.org/pdf/2508.03440v1)**

> **作者:** Junhong Wu; Jinliang Lu; Zixuan Ren; Ganqiang Hu; Zhi Wu; Dai Dai; Hua Wu
>
> **备注:** 10 pages, 7 figures, working in progress
>
> **摘要:** Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
>
---
#### [new 049] Long Story Generation via Knowledge Graph and Literary Theory
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在解决长篇故事生成中的主题漂移和逻辑不连贯问题，通过构建基于知识图谱和文学理论的多智能体结构，结合LLM实现高效生成，以提升故事质量。**

- **链接: [http://arxiv.org/pdf/2508.03137v1](http://arxiv.org/pdf/2508.03137v1)**

> **作者:** Ge Shi; Kaiyu Huang; Guochen Feng
>
> **摘要:** The generation of a long story consisting of several thousand words is a sub-task in the field of long text generation~(LTG). Previous research has addressed this challenge through outline-based generation, which employs a multi-stage method for generating outlines into stories. However, this approach suffers from two common issues: almost inevitable theme drift caused by the loss of memory of previous outlines, and tedious plots with incoherent logic that are less appealing to human readers. In this paper, we propose the multi-agent Story Generator structure to improve the multi-stage method, using large language models~(LLMs) as the core components of agents. To avoid theme drift, we introduce a memory storage model comprising two components: a long-term memory storage that identifies the most important memories, thereby preventing theme drift; and a short-term memory storage that retains the latest outlines from each generation round. To incorporate engaging elements into the story, we design a story theme obstacle framework based on literary narratology theory that introduces uncertain factors and evaluation criteria to generate outline. This framework calculates the similarity of the former storyline and enhances the appeal of the story by building a knowledge graph and integrating new node content. Additionally, we establish a multi-agent interaction stage to simulate writer-reader interaction through dialogue and revise the story text according to feedback, to ensure it remains consistent and logical. Evaluations against previous methods demonstrate that our approach can generate higher-quality long stories.
>
---
#### [new 050] ReDSM5: A Reddit Dataset for DSM-5 Depression Detection
- **分类: cs.CL**

- **简介: 该论文构建了包含Reddit长文本的DSM-5抑郁检测数据集，解决了传统方法仅标注情绪/未标注症状的问题，通过症状级标注与临床解释提升模型检测与解释能力。**

- **链接: [http://arxiv.org/pdf/2508.03399v1](http://arxiv.org/pdf/2508.03399v1)**

> **作者:** Eliseo Bao; Anxo Pérez; Javier Parapar
>
> **备注:** Accepted as a resource paper at CIKM 2025
>
> **摘要:** Depression is a pervasive mental health condition that affects hundreds of millions of individuals worldwide, yet many cases remain undiagnosed due to barriers in traditional clinical access and pervasive stigma. Social media platforms, and Reddit in particular, offer rich, user-generated narratives that can reveal early signs of depressive symptomatology. However, existing computational approaches often label entire posts simply as depressed or not depressed, without linking language to specific criteria from the DSM-5, the standard clinical framework for diagnosing depression. This limits both clinical relevance and interpretability. To address this gap, we introduce ReDSM5, a novel Reddit corpus comprising 1484 long-form posts, each exhaustively annotated at the sentence level by a licensed psychologist for the nine DSM-5 depression symptoms. For each label, the annotator also provides a concise clinical rationale grounded in DSM-5 methodology. We conduct an exploratory analysis of the collection, examining lexical, syntactic, and emotional patterns that characterize symptom expression in social media narratives. Compared to prior resources, ReDSM5 uniquely combines symptom-specific supervision with expert explanations, facilitating the development of models that not only detect depression but also generate human-interpretable reasoning. We establish baseline benchmarks for both multi-label symptom classification and explanation generation, providing reference results for future research on detection and interpretability.
>
---
#### [new 051] Reliable Evaluation Protocol for Low-Precision Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文旨在开发低精度检索系统评估方案，解决因计算精度不足导致的关联性错误问题。通过引入高精度评分机制（HPS）和tie-aware指标（TRM），结合实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.03306v1](http://arxiv.org/pdf/2508.03306v1)**

> **作者:** Kisu Yang; Yoonna Jang; Hwanseok Jang; Kenneth Choi; Isabelle Augenstein; Heuiseok Lim
>
> **备注:** 11 pages, 5 figures, submitted to ARR
>
> **摘要:** Lowering the numerical precision of model parameters and computations is widely adopted to improve the efficiency of retrieval systems. However, when computing relevance scores between the query and documents in low-precision, we observe spurious ties due to the reduced granularity. This introduces high variability in the results based on tie resolution, making the evaluation less reliable. To address this, we propose a more robust retrieval evaluation protocol designed to reduce score variation. It consists of: (1) High-Precision Scoring (HPS), which upcasts the final scoring step to higher precision to resolve tied candidates with minimal computational cost; and (2) Tie-aware Retrieval Metrics (TRM), which report expected scores, range, and bias to quantify order uncertainty of tied candidates. Our experiments test multiple models with three scoring functions on two retrieval datasets to demonstrate that HPS dramatically reduces tie-induced instability, and TRM accurately recovers expected metric values. This combination enables a more consistent and reliable evaluation system for lower-precision retrievals.
>
---
#### [new 052] ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ToolRegistry，旨在解决LLM工具集成中的碎片化、协议限制及复杂性问题，通过协议无关接口简化工具管理，实现60-80%代码缩减和3.1x性能提升，并兼容OpenAI标准，验证其在开发效率与维护性上的显著优势。**

- **链接: [http://arxiv.org/pdf/2507.10593v1](http://arxiv.org/pdf/2507.10593v1)**

> **作者:** Peng Ding
>
> **摘要:** Large Language Model (LLM) applications are increasingly relying on external tools to extend their capabilities beyond text generation. However, current tool integration approaches suffer from fragmentation, protocol limitations, and implementation complexity, leading to substantial development overhead. This paper presents Toolregistry, a protocol-agnostic tool management library that simplifies tool registration, representation, execution, and lifecycle management via a unified interface. Our evaluation demonstrates that \toolregistry achieves 60-80% reduction in tool integration code, up to 3.1x performance improvements through concurrent execution, and 100% compatibility with OpenAI function calling standards. Real-world case studies show significant improvements in development efficiency and code maintainability across diverse integration scenarios. \toolregistry is open-source and available at https://github.com/Oaklight/ToolRegistry, with comprehensive documentation at https://toolregistry.readthedocs.io/.
>
---
#### [new 053] Defend LLMs Through Self-Consciousness
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文旨在开发一种基于自我意识的自保护机制，解决prompt注入攻击问题，通过Meta-Cognitive和Arbitration模块实现LLMs自主评估与调控输出，实验表明在AdvBench等七种模型上显著提升防御成功率，同时分析了计算开销与防御效果的平衡。**

- **链接: [http://arxiv.org/pdf/2508.02961v1](http://arxiv.org/pdf/2508.02961v1)**

> **作者:** Boshi Huang; Fabio Nonato de Paula
>
> **备注:** Presented at KDD Workshop on Ethical Artificial Intelligence: Methods and Applications (EAI) 2025
>
> **摘要:** This paper introduces a novel self-consciousness defense mechanism for Large Language Models (LLMs) to combat prompt injection attacks. Unlike traditional approaches that rely on external classifiers, our method leverages the LLM's inherent reasoning capabilities to perform self-protection. We propose a framework that incorporates Meta-Cognitive and Arbitration Modules, enabling LLMs to evaluate and regulate their own outputs autonomously. Our approach is evaluated on seven state-of-the-art LLMs using two datasets: AdvBench and Prompt-Injection-Mixed-Techniques-2024. Experiment results demonstrate significant improvements in defense success rates across models and datasets, with some achieving perfect and near-perfect defense in Enhanced Mode. We also analyze the trade-off between defense success rate improvement and computational overhead. This self-consciousness method offers a lightweight, cost-effective solution for enhancing LLM ethics, particularly beneficial for GenAI use cases across various platforms.
>
---
#### [new 054] Draw Your Mind: Personalized Generation via Condition-Level Modeling in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于任务为个性化生成，解决T2I扩散模型因提示级建模限制导致的个性化不足问题，提出DrUM通过条件级建模融合用户画像与Transformer适配器，提升模型对用户偏好自然融入的能力。**

- **链接: [http://arxiv.org/pdf/2508.03481v1](http://arxiv.org/pdf/2508.03481v1)**

> **作者:** Hyungjin Kim; Seokho Ahn; Young-Duk Seo
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Personalized generation in T2I diffusion models aims to naturally incorporate individual user preferences into the generation process with minimal user intervention. However, existing studies primarily rely on prompt-level modeling with large-scale models, often leading to inaccurate personalization due to the limited input token capacity of T2I diffusion models. To address these limitations, we propose DrUM, a novel method that integrates user profiling with a transformer-based adapter to enable personalized generation through condition-level modeling in the latent space. DrUM demonstrates strong performance on large-scale datasets and seamlessly integrates with open-source text encoders, making it compatible with widely used foundation T2I models without requiring additional fine-tuning.
>
---
#### [new 055] Toward Verifiable Misinformation Detection: A Multi-Tool LLM Agent Framework
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出了一种基于多工具的LLM框架，旨在验证虚假信息，解决传统模型检测效率低、透明度不足的问题，通过动态交互与证据整合提升准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.03092v1](http://arxiv.org/pdf/2508.03092v1)**

> **作者:** Zikun Cui; Tianyi Huang; Chia-En Chiang; Cuiqianhe Du
>
> **摘要:** With the proliferation of Large Language Models (LLMs), the detection of misinformation has become increasingly important and complex. This research proposes an innovative verifiable misinformation detection LLM agent that goes beyond traditional true/false binary judgments. The agent actively verifies claims through dynamic interaction with diverse web sources, assesses information source credibility, synthesizes evidence, and provides a complete verifiable reasoning process. Our designed agent architecture includes three core tools: precise web search tool, source credibility assessment tool and numerical claim verification tool. These tools enable the agent to execute multi-step verification strategies, maintain evidence logs, and form comprehensive assessment conclusions. We evaluate using standard misinformation datasets such as FakeNewsNet, comparing with traditional machine learning models and LLMs. Evaluation metrics include standard classification metrics, quality assessment of reasoning processes, and robustness testing against rewritten content. Experimental results show that our agent outperforms baseline methods in misinformation detection accuracy, reasoning transparency, and resistance to information rewriting, providing a new paradigm for trustworthy AI-assisted fact-checking.
>
---
#### [new 056] Forest vs Tree: The $(N, K)$ Trade-off in Reproducible ML Evaluation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究了机器学习评价中$(N, K)$参数的权衡，旨在通过优化指标选择和数据集配置提升可重复性，解决了因忽视人类分歧导致的高$N \times K$成本问题。**

- **链接: [http://arxiv.org/pdf/2508.03663v1](http://arxiv.org/pdf/2508.03663v1)**

> **作者:** Deepak Pandita; Flip Korn; Chris Welty; Christopher M. Homan
>
> **摘要:** Reproducibility is a cornerstone of scientific validation and of the authority it confers on its results. Reproducibility in machine learning evaluations leads to greater trust, confidence, and value. However, the ground truth responses used in machine learning often necessarily come from humans, among whom disagreement is prevalent, and surprisingly little research has studied the impact of effectively ignoring disagreement in these responses, as is typically the case. One reason for the lack of research is that budgets for collecting human-annotated evaluation data are limited, and obtaining more samples from multiple annotators for each example greatly increases the per-item annotation costs. We investigate the trade-off between the number of items ($N$) and the number of responses per item ($K$) needed for reliable machine learning evaluation. We analyze a diverse collection of categorical datasets for which multiple annotations per item exist, and simulated distributions fit to these datasets, to determine the optimal $(N, K)$ configuration, given a fixed budget ($N \times K$), for collecting evaluation data and reliably comparing the performance of machine learning models. Our findings show, first, that accounting for human disagreement may come with $N \times K$ at no more than 1000 (and often much lower) for every dataset tested on at least one metric. Moreover, this minimal $N \times K$ almost always occurred for $K > 10$. Furthermore, the nature of the tradeoff between $K$ and $N$ -- or if one even existed -- depends on the evaluation metric, with metrics that are more sensitive to the full distribution of responses performing better at higher levels of $K$. Our methods can be used to help ML practitioners get more effective test data by finding the optimal metrics and number of items and annotations per item to collect to get the most reliability for their budget.
>
---
#### [new 057] Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文旨在解决LLM工具集成中的协议碎片化问题，提出一种协议无关的集成方法，通过自动化生成schema、优化并发执行和管理工具，实现开发效率提升60%-80%，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.02979v1](http://arxiv.org/pdf/2508.02979v1)**

> **作者:** Peng Ding; Rick Stevens
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2507.10593
>
> **摘要:** The proliferation of tool-augmented Large Language Models (LLMs) has created a fragmented ecosystem where developers must navigate multiple protocols, manual schema definitions, and complex execution workflows. We address this challenge by proposing a unified approach to tool integration that abstracts protocol differences while optimizing execution performance. Our solution demonstrates how protocol-agnostic design principles can significantly reduce development overhead through automated schema generation, dual-mode concurrent execution, and seamless multi-source tool management. Experimental results show 60-80% code reduction across integration scenarios, performance improvements up to 3.1x through optimized concurrency, and full compatibility with existing function calling standards. This work contributes both theoretical insights into tool integration architecture and practical solutions for real-world LLM application development.
>
---
#### [new 058] OSINT or BULLSHINT? Exploring Open-Source Intelligence tweets about the Russo-Ukrainian War
- **分类: cs.SI; cs.CL**

- **简介: 该论文探讨OSINT在冲突中的角色，识别虚假信息与传播策略，利用情感分析、部分派别检测及社区检测技术分析数据，揭示战地信息传播模式及其影响机制。**

- **链接: [http://arxiv.org/pdf/2508.03599v1](http://arxiv.org/pdf/2508.03599v1)**

> **作者:** Johannes Niu; Mila Stillman; Anna Kruspe
>
> **摘要:** This paper examines the role of Open Source Intelligence (OSINT) on Twitter regarding the Russo-Ukrainian war, distinguishing between genuine OSINT and deceptive misinformation efforts, termed "BULLSHINT." Utilizing a dataset spanning from January 2022 to July 2023, we analyze nearly 2 million tweets from approximately 1,040 users involved in discussing real-time military engagements, strategic analyses, and misinformation related to the conflict. Using sentiment analysis, partisanship detection, misinformation identification, and Named Entity Recognition (NER), we uncover communicative patterns and dissemination strategies within the OSINT community. Significant findings reveal a predominant negative sentiment influenced by war events, a nuanced distribution of pro-Ukrainian and pro-Russian partisanship, and the potential strategic manipulation of information. Additionally, we apply community detection techniques, which are able to identify distinct clusters partisanship, topics, and misinformation, highlighting the complex dynamics of information spread on social media. This research contributes to the understanding of digital warfare and misinformation dynamics, offering insights into the operationalization of OSINT in geopolitical conflicts.
>
---
#### [new 059] SecoustiCodec: Cross-Modal Aligned Streaming Single-Codecbook Speech Codec
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出一种跨模态低带宽语音编码器SecoustiCodec，旨在解决语义与声学信息分离困难、提升语义完整性和重建质量的问题。通过引入VAE和FSQ实现语义编码，结合对比学习消除混音信息，并采用多阶段优化策略，有效缓解长尾分布问题，达到SOTA的PESQ性能（1.77/2.58）。**

- **链接: [http://arxiv.org/pdf/2508.02849v1](http://arxiv.org/pdf/2508.02849v1)**

> **作者:** Chunyu Qiang; Haoyu Wang; Cheng Gong; Tianrui Wang; Ruibo Fu; Tao Wang; Ruilong Chen; Jiangyan Yi; Zhengqi Wen; Chen Zhang; Longbiao Wang; Jianwu Dang; Jianhua Tao
>
> **摘要:** Speech codecs serve as a crucial bridge in unifying speech and text language models. Existing codec methods face several challenges in semantic encoding, such as residual paralinguistic information (e.g., timbre, emotion), insufficient semantic completeness, limited reconstruction capability, and lack of support for streaming. To address these challenges, we propose SecoustiCodec, a cross-modal aligned low-bitrate streaming speech codec that disentangles semantic and paralinguistic information in a single-codebook space. To ensure semantic completeness and reconstruction fidelity, paralinguistic encoding is introduced to bridge the information gap between semantic and acoustic encoding. A semantic-only efficient quantization method based on VAE (Variational Autoencoder) and FSQ (Finite Scalar Quantization) is proposed. This approach alleviates the long-tail distribution problem of tokens while maintaining high codebook utilization. A semantic disentanglement method based on contrastive learning is proposed, which aligns text and speech in a joint multimodal frame-level space, effectively removing paralinguistic information from semantic encoding. An acoustic-constrained multi-stage optimization strategy is proposed to ensure robust and stable convergence. Figure~\ref{fig:pesq_kbps_below_2kbps} shows SecoustiCodec achieves SOTA (state-of-the-art) reconstruction quality (PESQ) of 1.77/2.58 at 0.27/1 kbps. The code and model weights for SecoustiCodec will be open-sourced upon the completion of the peer-review process. We've open-sourced SecoustiCodec's demo, code, and model weights.
>
---
#### [new 060] NeuroSync: Intent-Aware Code-Based Problem Solving via Direct LLM Understanding Modification
- **分类: cs.HC; cs.AI; cs.CL; cs.SE**

- **简介: 该论文探讨了通过直接干预LLM理解来提升意图-任务对齐的工作机制，解决了用户依赖对话型LLM解决编程问题时因意图与代码不匹配导致的困惑问题，创新性地引入知识蒸馏技术并通过可视化工具实现意图映射编辑。**

- **链接: [http://arxiv.org/pdf/2508.02823v1](http://arxiv.org/pdf/2508.02823v1)**

> **作者:** Wenshuo Zhang; Leixian Shen; Shuchang Xu; Jindu Wang; Jian Zhao; Huamin Qu; Linping Yuan
>
> **备注:** Accepted in UIST 2025
>
> **摘要:** Conversational LLMs have been widely adopted by domain users with limited programming experience to solve domain problems. However, these users often face misalignment between their intent and generated code, resulting in frustration and rounds of clarification. This work first investigates the cause of this misalignment, which dues to bidirectional ambiguity: both user intents and coding tasks are inherently nonlinear, yet must be expressed and interpreted through linear prompts and code sequences. To address this, we propose direct intent-task matching, a new human-LLM interaction paradigm that externalizes and enables direct manipulation of the LLM understanding, i.e., the coding tasks and their relationships inferred by the LLM prior to code generation. As a proof-of-concept, this paradigm is then implemented in NeuroSync, which employs a knowledge distillation pipeline to extract LLM understanding, user intents, and their mappings, and enhances the alignment by allowing users to intuitively inspect and edit them via visualizations. We evaluate the algorithmic components of NeuroSync via technical experiments, and assess its overall usability and effectiveness via a user study (N=12). The results show that it enhances intent-task alignment, lowers cognitive effort, and improves coding efficiency.
>
---
#### [new 061] Efficient Agents: Building Effective Agents While Reducing Cost
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文旨在解决LLM驱动代理系统因复杂度增加而成本上升的问题，通过分析复杂度、模块化效果及效率优化，提出Efficient Agents框架，实现成本降低28.4%的同时提升性能，推动AI代理系统的高效化与可持续发展。**

- **链接: [http://arxiv.org/pdf/2508.02694v1](http://arxiv.org/pdf/2508.02694v1)**

> **作者:** Ningning Wang; Xavier Hu; Pai Liu; He Zhu; Yue Hou; Heyuan Huang; Shengyu Zhang; Jian Yang; Jiaheng Liu; Ge Zhang; Changwang Zhang; Jun Wang; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** Work in progress. For GitHub repository, see https://github.com/OPPO-PersonalAI/OAgents
>
> **摘要:** The remarkable capabilities of Large Language Model (LLM)-driven agents have enabled sophisticated systems to tackle complex, multi-step tasks, but their escalating costs threaten scalability and accessibility. This work presents the first systematic study of the efficiency-effectiveness trade-off in modern agent systems, addressing the critical need for cost-effective designs without sacrificing performance. We investigate three key questions: (1) How much complexity do agentic tasks inherently require? (2) When do additional modules yield diminishing returns? (3) How much efficiency can be gained through the design of efficient agent frameworks? Through an empirical analysis on the GAIA benchmark, we evaluate the impact of LLM backbone selection, agent framework designs, and test-time scaling strategies. Using the cost-of-pass metric, we quantify the efficiency-performance trade-off across these dimensions. Our findings inform the development of Efficient Agents , a novel agent framework that has an optimal complexity to task requirements. Efficient Agents retains 96.7% of the performance of OWL, one leading open-source agent framework, while reducing operational costs from $0.398 to $0.228, resulting in a 28.4% improvement in cost-of-pass. Our work provides actionable insights for designing efficient, high-performing agent systems, advancing the accessibility and sustainability of AI-driven solutions.
>
---
#### [new 062] ChartCap: Mitigating Hallucination of Dense Chart Captioning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文旨在解决图表描述不准确、存在假象的问题，通过构建包含结构元素与关键信息的ChartCap数据集，设计四阶段生成模型并提出视觉一致性评分指标，有效提升了模型在生成高质量描述方面的性能。**

- **链接: [http://arxiv.org/pdf/2508.03164v1](http://arxiv.org/pdf/2508.03164v1)**

> **作者:** Junyoung Lim; Jaewoo Ahn; Gunhee Kim
>
> **备注:** ICCV 2025 (Highlight)
>
> **摘要:** Generating accurate, informative, and hallucination-free captions for charts remains challenging for vision language models, primarily due to the lack of large-scale, high-quality datasets of real-world charts. However, existing real-world chart datasets suffer from the inclusion of extraneous information that cannot be inferred from the chart and failure to sufficiently capture structural elements and key insights. Therefore, we introduce ChartCap, a large-scale dataset of 565K real-world chart images paired with type-specific, dense captions that exclude extraneous information and highlight both structural elements and key insights in detail. To build ChartCap, we design a four-stage pipeline that generates captions using only the discernible data from the chart and employ a cycle consistency-based human verification, which accelerates quality control without sacrificing accuracy. Additionally, we propose a novel metric, the Visual Consistency Score, which evaluates caption quality by measuring the similarity between the chart regenerated from a caption and the original chart, independent of reference captions. Extensive experiments confirms that models fine-tuned on ChartCap consistently generate more accurate and informative captions with reduced hallucinations, surpassing both open-source and proprietary models and even human-annotated captions.
>
---
#### [new 063] MultiRAG: A Knowledge-guided Framework for Mitigating Hallucination in Multi-source Retrieval Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出了一个基于知识引导的框架MultiRAG，旨在解决多源检索增强生成（RAG）中由于多源数据稀疏性和信息冲突导致的虚假信息问题。通过构建多源线图整合逻辑关系并实现多级信心评估，解决了传统RAG的稀疏数据和信息冲突问题。**

- **链接: [http://arxiv.org/pdf/2508.03553v1](http://arxiv.org/pdf/2508.03553v1)**

> **作者:** Wenlong Wu; Haofen Wang; Bohan Li; Peixuan Huang; Xinzhe Zhao; Lei Liang
>
> **备注:** Accepted by ICDE 2025 Research Paper
>
> **摘要:** Retrieval Augmented Generation (RAG) has emerged as a promising solution to address hallucination issues in Large Language Models (LLMs). However, the integration of multiple retrieval sources, while potentially more informative, introduces new challenges that can paradoxically exacerbate hallucination problems. These challenges manifest primarily in two aspects: the sparse distribution of multi-source data that hinders the capture of logical relationships and the inherent inconsistencies among different sources that lead to information conflicts. To address these challenges, we propose MultiRAG, a novel framework designed to mitigate hallucination in multi-source retrieval-augmented generation through knowledge-guided approaches. Our framework introduces two key innovations: (1) a knowledge construction module that employs multi-source line graphs to efficiently aggregate logical relationships across different knowledge sources, effectively addressing the sparse data distribution issue; and (2) a sophisticated retrieval module that implements a multi-level confidence calculation mechanism, performing both graph-level and node-level assessments to identify and eliminate unreliable information nodes, thereby reducing hallucinations caused by inter-source inconsistencies. Extensive experiments on four multi-domain query datasets and two multi-hop QA datasets demonstrate that MultiRAG significantly enhances the reliability and efficiency of knowledge retrieval in complex multi-source scenarios. \textcolor{blue}{Our code is available in https://github.com/wuwenlong123/MultiRAG.
>
---
#### [new 064] A Comparative Study of Neurosymbolic AI Approaches to Interpretable Logical Reasoning
- **分类: cs.AI; cs.CL; cs.LG; cs.SC**

- **简介: 该论文旨在比较神经符号AI方法在提升逻辑推理能力方面的表现，解决领域依赖性强的LLM推理问题。通过引入LNN（集成模型）和LLM-SS（混合模型），分析其在领域无关任务中的优势，提出基于LLM-Symbolic Solver的通用框架，验证混合模型更优。**

- **链接: [http://arxiv.org/pdf/2508.03366v1](http://arxiv.org/pdf/2508.03366v1)**

> **作者:** Michael K. Chen
>
> **备注:** Accepted to NeSy 2025
>
> **摘要:** General logical reasoning, defined as the ability to reason deductively on domain-agnostic tasks, continues to be a challenge for large language models (LLMs). Current LLMs fail to reason deterministically and are not interpretable. As such, there has been a recent surge in interest in neurosymbolic AI, which attempts to incorporate logic into neural networks. We first identify two main neurosymbolic approaches to improving logical reasoning: (i) the integrative approach comprising models where symbolic reasoning is contained within the neural network, and (ii) the hybrid approach comprising models where a symbolic solver, separate from the neural network, performs symbolic reasoning. Both contain AI systems with promising results on domain-specific logical reasoning benchmarks. However, their performance on domain-agnostic benchmarks is understudied. To the best of our knowledge, there has not been a comparison of the contrasting approaches that answers the following question: Which approach is more promising for developing general logical reasoning? To analyze their potential, the following best-in-class domain-agnostic models are introduced: Logic Neural Network (LNN), which uses the integrative approach, and LLM-Symbolic Solver (LLM-SS), which uses the hybrid approach. Using both models as case studies and representatives of each approach, our analysis demonstrates that the hybrid approach is more promising for developing general logical reasoning because (i) its reasoning chain is more interpretable, and (ii) it retains the capabilities and advantages of existing LLMs. To support future works using the hybrid approach, we propose a generalizable framework based on LLM-SS that is modular by design, model-agnostic, domain-agnostic, and requires little to no human input.
>
---
#### [new 065] Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning
- **分类: cs.LG; cs.CL; cs.SE**

- **简介: 该论文旨在解决长上下文、多轮交互下的软件工程代理任务，通过改进DAPO算法实现对SWE基准测试的成功率提升（39%），并对比其他模型展示其有效性，推动自主软件工程代理的发展。**

- **链接: [http://arxiv.org/pdf/2508.03501v1](http://arxiv.org/pdf/2508.03501v1)**

> **作者:** Alexander Golubev; Maria Trofimova; Sergei Polezhaev; Ibragim Badertdinov; Maksim Nekrashevich; Anton Shevtsov; Simon Karasik; Sergey Abramov; Andrei Andriushchenko; Filipp Fisin; Sergei Skvortsov; Boris Yangel
>
> **摘要:** Research on applications of Reinforcement Learning (RL) to Large Language Models (LLMs) has mostly been focused on single-turn problems, such as mathematical reasoning or single-shot code generation. While these problems can be viewed as token-level multi-turn MDPs, this view corresponds to a degenerate case of multi-turn interaction where the environment provides no feedback. This contrasts with many real-world domains, such as software engineering (SWE), which require rich multi-turn interactions with a stateful environment that responds to each action with a non-trivial observation. To bridge this gap, we demonstrate the successful application of RL to this general regime. Using a modified Decoupled Advantage Policy Optimization (DAPO) algorithm, we train an agent based on Qwen2.5-72B-Instruct to solve real-world software engineering tasks. Our approach increases the agent's success rate on the SWE-bench Verified benchmark from a 20% rejection fine-tuned baseline to 39%, without relying on any teacher models. On SWE-rebench, our agent matches or outperforms leading open-weight models such as DeepSeek-V3-0324 and Qwen3-235B-A22B using an identical scaffolding, offering a viable path toward building more capable autonomous agents for complex real-world problems based on open models.
>
---
#### [new 066] VRPO: Rethinking Value Modeling for Robust RL Training under Noisy Supervision
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出VRPO框架，解决RLHF中噪声干扰导致的政策不稳定问题，通过结合语言模型熵/困惑度辅助损失与信息瓶颈机制，增强价值模型对噪声的过滤能力，提升数学推理、科学问答等任务的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.03058v1](http://arxiv.org/pdf/2508.03058v1)**

> **作者:** Dingwei Zhu; Shihan Dou; Zhiheng Xi; Senjie Jin; Guoqiang Zhang; Jiazheng Zhang; Junjie Ye; Mingxu Chai; Enyu Zhou; Ming Zhang; Caishuang Huang; Yunke Zhang; Yuran Wang; Tao Gui
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) often suffers from noisy or imperfect reward supervision in real-world settings, which undermines policy stability and generalization. Such noise may cause models to lose attention on key words during advantage estimation. While prior work focuses on reward denoising or filtering poor data, it often overlooks the critical role of the value model in policy optimization. In this work, we show that a strong value model is essential for mitigating noise by absorbing unstable signals and enabling more reliable advantage estimation. We propose VRPO, a value-centric framework for robust PPO training under noisy supervision. VRPO combines two core designs: (1) an auxiliary loss guided by entropy and perplexity from a frozen language model, and (2) a variational information bottleneck. These mechanisms enhance the value model's ability to filter out noise and capture key words from the context during advantage estimation, transforming it from a passive predictor into an active regulator of noise. Experiments on math reasoning, science QA, and multi-turn dialogue, under both rule-based and model-based noisy rewards, show that VRPO consistently outperforms PPO and GRPO baselines. Our findings underscore the often-overlooked importance of the value model in RLHF and offer a principled and practical approach to robust policy optimization in noisy real-world environments.
>
---
#### [new 067] VisuCraft: Enhancing Large Vision-Language Models for Complex Visual-Guided Creative Content Generation via Structured Information Extraction
- **分类: cs.CV; cs.CL**

- **简介: VisuCraft旨在提升LVLMs在复杂视觉引导内容生成中的表现，解决现有模型在视觉质量、创造力和指令准确性方面的局限性，通过整合结构信息提取与动态提示模块实现优化。**

- **链接: [http://arxiv.org/pdf/2508.02890v1](http://arxiv.org/pdf/2508.02890v1)**

> **作者:** Rongxin Jiang; Robert Long; Chenghao Gu; Mingrui Yan
>
> **摘要:** This paper introduces VisuCraft, a novel framework designed to significantly enhance the capabilities of Large Vision-Language Models (LVLMs) in complex visual-guided creative content generation. Existing LVLMs often exhibit limitations in maintaining high visual fidelity, genuine creativity, and precise adherence to nuanced user instructions when generating long-form texts. VisuCraft addresses these challenges by integrating a multimodal structured information extractor (E) and a dynamic prompt generation module (G). The extractor distills fine-grained visual attributes from input images into a rich, structured representation, which the dynamic prompt module then combines with user instructions to create highly optimized prompts for underlying LVLMs (e.g., LLaVA, InstructBLIP). Evaluated on the self-constructed ImageStoryGen-500K dataset using VisuGen Metrics (Visual Grounding, Creativity, and Instruction Adherence), VisuCraft consistently outperforms baseline LVLMs across tasks like story generation and poetry composition. Our results demonstrate remarkable improvements, particularly in creativity and instruction adherence, validating VisuCraft's effectiveness in producing imaginative, visually grounded, and user-aligned long-form creative text. This work unlocks new potential for LVLMs in sophisticated creative AI applications.
>
---
#### [new 068] MoKA: Mixture of Kronecker Adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Mixture of Kronecker Adapters（MoKA），解决低秩家庭适配器因秩限制导致表达力不足的问题，通过混合高斯乘积建模权重更新并引入门控机制，实现参数效率与准确性平衡，同时优化硬件部署，实验表明其在指令推理和常识推理任务中优于传统PEFT方法，参数减少27倍。**

- **链接: [http://arxiv.org/pdf/2508.03527v1](http://arxiv.org/pdf/2508.03527v1)**

> **作者:** Mohammadreza Sadeghi; Mahsa Ghazvini Nejad; MirHamed Jafarzadeh Asl; Yu Gu; Yuanhao Yu; Masoud Asgharian; Vahid Partovi Nia
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) is essential for reducing the computational overhead of large language models (LLMs). Low-rank family adapters are commonly used to control the parameter size efficiently while maintaining the generative power of LLMs. However, their limited expressiveness due to the rank constraint often restricts their performance on complex tasks. We propose Mixture of Kronecker Adapters (MoKA), a new generation of Kronecker adapters that addresses this limitation by modeling weight updates as a mixture of Kronecker products. Our proposed adapter leverages a gating mechanism that measures the importance of each Kronecker factor, enabling more expressive adaptation. Moreover, MoKA enables a rank flexibility that provides a better trade-off between parameter efficiency and accuracy. To ensure hardware efficiency, we reformulate Kronecker computations using standard matrix operations, allowing seamless deployment on GPU-optimized hardware. We conduct extensive experiments on instruction-tuning and commonsense reasoning tasks using low-bit quantized versions of LLaMA2-7B and LLaMA3-8B models. MoKA not only outperforms PEFT baselines, but also reduces the number of trainable parameters up to 27x, achieving state-of-the-art trade-offs between performance and parameter efficiency.
>
---
#### [new 069] Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文探讨了基于大视觉语言模型的视觉-语言导航任务，旨在解决自主机器人如何根据自然语言指令导航陌生环境的问题。通过对比低级与全景动作空间，研究了off-the-shelf LVLMs在VLN任务中的有效性，发现其在R2R数据集上实现41%的成功率，表明其仍需进一步优化以匹配专门设计的导航模型。**

- **链接: [http://arxiv.org/pdf/2508.02917v1](http://arxiv.org/pdf/2508.02917v1)**

> **作者:** Vebjørn Haug Kåsene; Pierre Lison
>
> **备注:** This paper has been accepted to ICNSLP 2025
>
> **摘要:** Vision-and-Language Navigation (VLN) refers to the task of enabling autonomous robots to navigate unfamiliar environments by following natural language instructions. While recent Large Vision-Language Models (LVLMs) have shown promise in this task, most current VLM systems rely on models specifically designed and optimized for navigation, leaving the potential of off-the-shelf LVLMs underexplored. Furthermore, while older VLN approaches used low-level action spaces with egocentric views and atomic actions (such as "turn left" or "move forward"), newer models tend to favor panoramic action spaces with discrete navigable viewpoints. This paper investigates (1) whether off-the-shelf LVLMs (fine-tuned without architectural modifications or simulator-based training) can effectively support VLN tasks and (2) whether such models can support both low-level and panoramic action paradigms. To this end, we fine-tune the open-source model Qwen2.5-VL-3B-Instruct on the Room-to-Room (R2R) dataset and evaluate its empirical performance across both low-level and panoramic action spaces. The best resulting model achieves a 41% success rate on the R2R test set, demonstrating that while off-the-shelf LVLMs can learn to perform Vision-and-Language Navigation, they still lag behind models specifically designed for this task.
>
---
#### [new 070] Understanding the Embedding Models on Hyper-relational Knowledge Graph
- **分类: cs.LG; cs.CL; cs.SI**

- **简介: 该论文旨在理解超关系知识图（HKGs）中嵌入模型的性能差异，解决传统KGE模型无法有效捕捉长距离依赖和主三元组信息的问题，通过提出FormerGNN框架优化分解方法与信息整合策略，提升HKGE模型效果。**

- **链接: [http://arxiv.org/pdf/2508.03280v1](http://arxiv.org/pdf/2508.03280v1)**

> **作者:** Yubo Wang; Shimin Di; Zhili Wang; Haoyang Li; Fei Teng; Hao Xin; Lei Chen
>
> **备注:** Accepted by CIKM 2025
>
> **摘要:** Recently, Hyper-relational Knowledge Graphs (HKGs) have been proposed as an extension of traditional Knowledge Graphs (KGs) to better represent real-world facts with additional qualifiers. As a result, researchers have attempted to adapt classical Knowledge Graph Embedding (KGE) models for HKGs by designing extra qualifier processing modules. However, it remains unclear whether the superior performance of Hyper-relational KGE (HKGE) models arises from their base KGE model or the specially designed extension module. Hence, in this paper, we data-wise convert HKGs to KG format using three decomposition methods and then evaluate the performance of several classical KGE models on HKGs. Our results show that some KGE models achieve performance comparable to that of HKGE models. Upon further analysis, we find that the decomposition methods alter the original HKG topology and fail to fully preserve HKG information. Moreover, we observe that current HKGE models are either insufficient in capturing the graph's long-range dependency or struggle to integrate main-triple and qualifier information due to the information compression issue. To further justify our findings and offer a potential direction for future HKGE research, we propose the FormerGNN framework. This framework employs a qualifier integrator to preserve the original HKG topology, and a GNN-based graph encoder to capture the graph's long-range dependencies, followed by an improved approach for integrating main-triple and qualifier information to mitigate compression issues. Our experimental results demonstrate that FormerGNN outperforms existing HKGE models.
>
---
#### [new 071] Teaching at Scale: Leveraging AI to Evaluate and Elevate Engineering Education
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文提出了一种基于大语言模型的教学评估框架，旨在通过分层总结、匿名化和异常检测提取工程教育中的关键反馈，结合可视化分析提升教学效果，解决了传统人工评估效率低的问题，实现了跨学科、多维度的教学质量监测与持续改进。**

- **链接: [http://arxiv.org/pdf/2508.02731v1](http://arxiv.org/pdf/2508.02731v1)**

> **作者:** Jean-Francois Chamberland; Martin C. Carlisle; Arul Jayaraman; Krishna R. Narayanan; Sunay Palsole; Karan Watson
>
> **摘要:** Evaluating teaching effectiveness at scale remains a persistent challenge for large universities, particularly within engineering programs that enroll tens of thousands of students. Traditional methods, such as manual review of student evaluations, are often impractical, leading to overlooked insights and inconsistent data use. This article presents a scalable, AI-supported framework for synthesizing qualitative student feedback using large language models. The system employs hierarchical summarization, anonymization, and exception handling to extract actionable themes from open-ended comments while upholding ethical safeguards. Visual analytics contextualize numeric scores through percentile-based comparisons, historical trends, and instructional load. The approach supports meaningful evaluation and aligns with best practices in qualitative analysis and educational assessment, incorporating student, peer, and self-reflective inputs without automating personnel decisions. We report on its successful deployment across a large college of engineering. Preliminary validation through comparisons with human reviewers, faculty feedback, and longitudinal analysis suggests that LLM-generated summaries can reliably support formative evaluation and professional development. This work demonstrates how AI systems, when designed with transparency and shared governance, can promote teaching excellence and continuous improvement at scale within academic institutions.
>
---
#### [new 072] AGENTiGraph: A Multi-Agent Knowledge Graph Framework for Interactive, Domain-Specific LLM Chatbots
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AGENTiGraph框架，解决多智能体知识图谱与LLM交互问题，通过意图分类、任务规划和知识整合提升领域数据管理效率，验证其在教育场景下的性能并拓展至合规医疗等应用。**

- **链接: [http://arxiv.org/pdf/2508.02999v1](http://arxiv.org/pdf/2508.02999v1)**

> **作者:** Xinjie Zhao; Moritz Blum; Fan Gao; Yingjian Chen; Boming Yang; Luis Marquez-Carpintero; Mónica Pina-Navarro; Yanran Fu; So Morikawa; Yusuke Iwasawa; Yutaka Matsuo; Chanjun Park; Irene Li
>
> **备注:** CIKM 2025, Demo Track
>
> **摘要:** AGENTiGraph is a user-friendly, agent-driven system that enables intuitive interaction and management of domain-specific data through the manipulation of knowledge graphs in natural language. It gives non-technical users a complete, visual solution to incrementally build and refine their knowledge bases, allowing multi-round dialogues and dynamic updates without specialized query languages. The flexible design of AGENTiGraph, including intent classification, task planning, and automatic knowledge integration, ensures seamless reasoning between diverse tasks. Evaluated on a 3,500-query benchmark within an educational scenario, the system outperforms strong zero-shot baselines (achieving 95.12% classification accuracy, 90.45% execution success), indicating potential scalability to compliance-critical or multi-step queries in legal and medical domains, e.g., incorporating new statutes or research on the fly. Our open-source demo offers a powerful new paradigm for multi-turn enterprise knowledge management that bridges LLMs and structured graphs.
>
---
#### [new 073] CreditARF: A Framework for Corporate Credit Rating with Annual Report and Financial Feature Integration
- **分类: q-fin.ST; cs.CE; cs.CL; cs.LG**

- **简介: 本研究提出一种基于FinBERT的信用评级框架，整合年度报告和财务数据以提升非金融信息的利用效率，通过构建大规模数据集CCRD验证其有效性，显著提高预测准确率。**

- **链接: [http://arxiv.org/pdf/2508.02738v1](http://arxiv.org/pdf/2508.02738v1)**

> **作者:** Yumeng Shi; Zhongliang Yang; DiYang Lu; Yisi Wang; Yiting Zhou; Linna Zhou
>
> **摘要:** Corporate credit rating serves as a crucial intermediary service in the market economy, playing a key role in maintaining economic order. Existing credit rating models rely on financial metrics and deep learning. However, they often overlook insights from non-financial data, such as corporate annual reports. To address this, this paper introduces a corporate credit rating framework that integrates financial data with features extracted from annual reports using FinBERT, aiming to fully leverage the potential value of unstructured text data. In addition, we have developed a large-scale dataset, the Comprehensive Corporate Rating Dataset (CCRD), which combines both traditional financial data and textual data from annual reports. The experimental results show that the proposed method improves the accuracy of the rating predictions by 8-12%, significantly improving the effectiveness and reliability of corporate credit ratings.
>
---
#### [new 074] VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models via Hessian Augmentation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出一种基于Hessian增强的VLMQ框架，解决大型视觉语言模型PTQ在token冗余和重要性不均衡问题，通过优化Hessian结构并计算token级重要性因素，实现了在低比特设置下的显著性能提升（如MME-RealWorld）。**

- **链接: [http://arxiv.org/pdf/2508.03351v1](http://arxiv.org/pdf/2508.03351v1)**

> **作者:** Yufei Xue; Yushi Huang; Jiawei Shao; Jun Zhang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Post-training quantization (PTQ) has emerged as an effective approach for compressing large models and accelerating their inference without retraining. While PTQ has been extensively studied in the context of large language models (LLMs), its applicability to vision-language models (VLMs) remains underexplored. In this paper, we identify a modality discrepancy (\emph{i.e.}, limited text tokens \emph{vs.} excessive and redundant vision tokens) of VLMs. However, existing Hessian-based LLM PTQ methods treat all tokens equally during quantization, resulting in severe performance drops when applied to VLMs. Motivated by this observation, we propose a novel importance-aware PTQ framework tailored for VLMs, dubbed VLMQ. Specifically, to address vision token redundancy, VLMQ 1) optimizes an importance-aware objective that yields an enhanced Hessian with token-level importance factors, while retaining compatibility with parallelized weight updates, and 2) ensures efficiency and effectiveness by computing these factors via a single lightweight block-wise backward pass, guided by a theoretical connection to token-level perturbations. Extensive evaluations on 8 benchmarks across 0.5B$\sim$32B VLMs demonstrate the state-of-the-art (SOTA) performance of our VLMQ, particularly under low-bit settings. For example, it achieves a substantial \textbf{16.45\%} improvement on MME-RealWorld under 2-bit quantization.
>
---
#### [new 075] Beyond Meme Templates: Limitations of Visual Similarity Measures in Meme Matching
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出了基于段位相似度的新方法，解决了传统模板匹配在非模板格式下的局限性，并通过对比全图与提示模型验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.03562v1](http://arxiv.org/pdf/2508.03562v1)**

> **作者:** Muzhaffar Hazman; Susan McKeever; Josephine Griffith
>
> **备注:** Accepted for publication at IEEE International Conference on Image Processing Theory, Tools and Applications (IPTA) 2025
>
> **摘要:** Internet memes, now a staple of digital communication, play a pivotal role in how users engage within online communities and allow researchers to gain insight into contemporary digital culture. These engaging user-generated content are characterised by their reuse of visual elements also found in other memes. Matching instances of memes via these shared visual elements, called Meme Matching, is the basis of a wealth of meme analysis approaches. However, most existing methods assume that every meme consists of a shared visual background, called a Template, with some overlaid text, thereby limiting meme matching to comparing the background image alone. Current approaches exclude the many memes that are not template-based and limit the effectiveness of automated meme analysis and would not be effective at linking memes to contemporary web-based meme dictionaries. In this work, we introduce a broader formulation of meme matching that extends beyond template matching. We show that conventional similarity measures, including a novel segment-wise computation of the similarity measures, excel at matching template-based memes but fall short when applied to non-template-based meme formats. However, the segment-wise approach was found to consistently outperform the whole-image measures on matching non-template-based memes. Finally, we explore a prompting-based approach using a pretrained Multimodal Large Language Model for meme matching. Our results highlight that accurately matching memes via shared visual elements, not just background templates, remains an open challenge that requires more sophisticated matching techniques.
>
---
#### [new 076] PyLate: Flexible Training and Retrieval for Late Interaction Models
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出PyLate框架，解决传统单向量搜索压缩信息导致的性能下降问题，通过多向量架构实现高效检索，为late interaction模型提供模块化支持，推动其在现代信息检索系统中的应用。**

- **链接: [http://arxiv.org/pdf/2508.03555v1](http://arxiv.org/pdf/2508.03555v1)**

> **作者:** Antoine Chaffin; Raphaël Sourty
>
> **备注:** 5 pages
>
> **摘要:** Neural ranking has become a cornerstone of modern information retrieval. While single vector search remains the dominant paradigm, it suffers from the shortcoming of compressing all the information into a single vector. This compression leads to notable performance degradation in out-of-domain, long-context, and reasoning-intensive retrieval tasks. Multi-vector approaches pioneered by ColBERT aim to address these limitations by preserving individual token embeddings and computing similarity via the MaxSim operator. This architecture has demonstrated superior empirical advantages, including enhanced out-of-domain generalization, long-context handling, and performance in complex retrieval scenarios. Despite these compelling empirical results and clear theoretical advantages, the practical adoption and public availability of late interaction models remain low compared to their single-vector counterparts, primarily due to a lack of accessible and modular tools for training and experimenting with such models. To bridge this gap, we introduce PyLate, a streamlined library built on top of Sentence Transformers to support multi-vector architectures natively, inheriting its efficient training, advanced logging, and automated model card generation while requiring minimal code changes to code templates users are already familiar with. By offering multi-vector-specific features such as efficient indexes, PyLate aims to accelerate research and real-world application of late interaction models, thereby unlocking their full potential in modern IR systems. Finally, PyLate has already enabled the development of state-of-the-art models, including GTE-ModernColBERT and Reason-ModernColBERT, demonstrating its practical utility for both research and production environments.
>
---
## 更新

#### [replaced 001] Reconstructing Sepsis Trajectories from Clinical Case Reports using LLMs: the Textual Time Series Corpus for Sepsis
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12326v2](http://arxiv.org/pdf/2504.12326v2)**

> **作者:** Shahriar Noroozizadeh; Jeremy C. Weiss
>
> **摘要:** Clinical case reports and discharge summaries may be the most complete and accurate summarization of patient encounters, yet they are finalized, i.e., timestamped after the encounter. Complementary data structured streams become available sooner but suffer from incompleteness. To train models and algorithms on more complete and temporally fine-grained data, we construct a pipeline to phenotype, extract, and annotate time-localized findings within case reports using large language models. We apply our pipeline to generate an open-access textual time series corpus for Sepsis-3 comprising 2,139 case reports from the Pubmed-Open Access (PMOA) Subset. To validate our system, we apply it on PMOA and timeline annotations from I2B2/MIMIC-IV and compare the results to physician-expert annotations. We show high recovery rates of clinical findings (event match rates: O1-preview--0.755, Llama 3.3 70B Instruct--0.753) and strong temporal ordering (concordance: O1-preview--0.932, Llama 3.3 70B Instruct--0.932). Our work characterizes the ability of LLMs to time-localize clinical findings in text, illustrating the limitations of LLM use for temporal reconstruction and providing several potential avenues of improvement via multimodal integration.
>
---
#### [replaced 002] LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12260v3](http://arxiv.org/pdf/2505.12260v3)**

> **作者:** Guangyuan Ma; Yongliang Ma; Xuanrui Gou; Zhenpeng Su; Ming Zhou; Songlin Hu
>
> **摘要:** Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over 1000x speedup in query encoding and over 10x increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.
>
---
#### [replaced 003] Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02175v2](http://arxiv.org/pdf/2508.02175v2)**

> **作者:** Liang Lin; Miao Yu; Kaiwen Luo; Yibo Zhang; Lilan Peng; Dexian Wang; Xuehai Tang; Yuanhe Zhang; Xikang Yang; Zhenhong Zhou; Kun Wang; Yang Liu
>
> **摘要:** As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.
>
---
#### [replaced 004] GPT is Devastated and LLaMA is Content: Emotion Representation Alignment in LLMs for Keyword-based Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11881v2](http://arxiv.org/pdf/2503.11881v2)**

> **作者:** Shadab Choudhury; Asha Kumar; Lara J. Martin
>
> **摘要:** In controlled text generation using large language models (LLMs), gaps arise between the language model's interpretation of concepts and people's expectations. We introduce the human evaluation task of Representation Alignment for measuring this gap. We selected four emotion representations: Words, Valence-Arousal-Dominance (VAD) dimensions expressed in both Lexical and Numeric forms, and Emojis and evaluate them in the context of keyword-guided sentence generation using both GPT-4 and LLaMA-3. In addition to Representation Alignment, we also measure people's judgments of the accuracy and realism of the generated sentences. While representations like VAD break emotions into easy-to-compute components, our findings show that people agree more with how LLMs generate when conditioned on English words (e.g., ``angry'') rather than VAD scales. This difference is especially visible when comparing Numeric VAD to words. Furthermore, we found that the perception of how much a generated sentence conveys an emotion is dependent on both the representation type and which emotion it is.
>
---
#### [replaced 005] Dynaword: From One-shot to Continuously Developed Datasets
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.02271v2](http://arxiv.org/pdf/2508.02271v2)**

> **作者:** Kenneth Enevoldsen; Kristian Nørgaard Jensen; Jan Kostkan; Balázs Szabó; Márton Kardos; Kirten Vad; Johan Heinsen; Andrea Blasi Núñez; Gianluca Barmina; Jacob Nielsen; Rasmus Larsen; Peter Vahlstrup; Per Møldrup Dalum; Desmond Elliott; Lukas Galke; Peter Schneider-Kamp; Kristoffer Nielbo
>
> **摘要:** Large-scale datasets are foundational for research and development in natural language processing. However, current approaches face three key challenges: (1) reliance on ambiguously licensed sources restricting use, sharing, and derivative works; (2) static dataset releases that prevent community contributions and diminish longevity; and (3) quality assurance processes restricted to publishing teams rather than leveraging community expertise. To address these limitations, we introduce two contributions: the Dynaword approach and Danish Dynaword. The Dynaword approach is a framework for creating large-scale, open datasets that can be continuously updated through community collaboration. Danish Dynaword is a concrete implementation that validates this approach and demonstrates its potential. Danish Dynaword contains over four times as many tokens as comparable releases, is exclusively openly licensed, and has received multiple contributions across industry and research. The repository includes light-weight tests to ensure data formatting, quality, and documentation, establishing a sustainable framework for ongoing community contributions and dataset evolution.
>
---
#### [replaced 006] ReaGAN: Node-as-Agent-Reasoning Graph Agentic Network
- **分类: cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.00429v2](http://arxiv.org/pdf/2508.00429v2)**

> **作者:** Minghao Guo; Xi Zhu; Jingyuan Huang; Kai Mei; Yongfeng Zhang
>
> **备注:** 17 pages, work in progress
>
> **摘要:** Graph Neural Networks (GNNs) have achieved remarkable success in graph-based learning by propagating information among neighbor nodes via predefined aggregation mechanisms. However, such fixed schemes often suffer from two key limitations. First, they cannot handle the imbalance in node informativeness -- some nodes are rich in information, while others remain sparse. Second, predefined message passing primarily leverages local structural similarity while ignoring global semantic relationships across the graph, limiting the model's ability to capture distant but relevant information. We propose Retrieval-augmented Graph Agentic Network (ReaGAN), an agent-based framework that empowers each node with autonomous, node-level decision-making. Each node acts as an agent that independently plans its next action based on its internal memory, enabling node-level planning and adaptive message propagation. Additionally, retrieval-augmented generation (RAG) allows nodes to access semantically relevant content and build global relationships in the graph. ReaGAN achieves competitive performance under few-shot in-context settings using a frozen LLM backbone without fine-tuning, showcasing the potential of agentic planning and local-global retrieval in graph learning.
>
---
#### [replaced 007] BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.14971v3](http://arxiv.org/pdf/2410.14971v3)**

> **作者:** Jilong Li; Zhenxi Song; Jiaqi Wang; Meishan Zhang; Honghai Liu; Min Zhang; Zhiguo Zhang
>
> **备注:** 8 pages (excluding references), accepted by Findings of ACL 2025
>
> **摘要:** Current EEG/MEG-to-text decoding systems suffer from three key limitations: (1) reliance on teacher-forcing methods, which compromises robustness during inference, (2) sensitivity to session-specific noise, hindering generalization across subjects, and (3) misalignment between brain signals and linguistic representations due to pre-trained language model over-dominance. To overcome these challenges, we propose BrainECHO (Brain signal decoding via vEctor-quantized speCtrogram reconstruction for WHisper-enhanced text generatiOn), a multi-stage framework that employs decoupled representation learning to achieve state-of-the-art performance on both EEG and MEG datasets. Specifically, BrainECHO consists of three stages: (1) Discrete autoencoding, which transforms continuous Mel spectrograms into a finite set of high-quality discrete representations for subsequent stages. (2) Frozen alignment, where brain signal embeddings are mapped to corresponding Mel spectrogram embeddings in a frozen latent space, effectively filtering session-specific noise through vector-quantized reconstruction, yielding a 3.65% improvement in BLEU-4 score. (3) Constrained decoding fine-tuning, which leverages the pre-trained Whisper model for audio-to-text translation, balancing signal adaptation with knowledge preservation, and achieving 74%-89% decoding BLEU scores without excessive reliance on teacher forcing. BrainECHO demonstrates robustness across sentence, session, and subject-independent conditions, passing Gaussian noise tests and showcasing its potential for enhancing language-based brain-computer interfaces.
>
---
#### [replaced 008] Why do LLMs attend to the first token?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02732v4](http://arxiv.org/pdf/2504.02732v4)**

> **作者:** Federico Barbero; Álvaro Arroyo; Xiangming Gu; Christos Perivolaropoulos; Michael Bronstein; Petar Veličković; Razvan Pascanu
>
> **摘要:** Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
>
---
#### [replaced 009] CutPaste&Find: Efficient Multimodal Hallucination Detector with Visual-aid Knowledge Base
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12591v2](http://arxiv.org/pdf/2502.12591v2)**

> **作者:** Cong-Duy Nguyen; Xiaobao Wu; Duc Anh Vu; Shuai Zhao; Thong Nguyen; Anh Tuan Luu
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, but they remain susceptible to hallucination, particularly object hallucination where non-existent objects or incorrect attributes are fabricated in generated descriptions. Existing detection methods achieve strong performance but rely heavily on expensive API calls and iterative LVLM-based validation, making them impractical for large-scale or offline use. To address these limitations, we propose CutPaste\&Find, a lightweight and training-free framework for detecting hallucinations in LVLM-generated outputs. Our approach leverages off-the-shelf visual and linguistic modules to perform multi-step verification efficiently without requiring LVLM inference. At the core of our framework is a Visual-aid Knowledge Base that encodes rich entity-attribute relationships and associated image representations. We introduce a scaling factor to refine similarity scores, mitigating the issue of suboptimal alignment values even for ground-truth image-text pairs. Comprehensive evaluations on benchmark datasets, including POPE and R-Bench, demonstrate that CutPaste\&Find achieves competitive hallucination detection performance while being significantly more efficient and cost-effective than previous methods.
>
---
#### [replaced 010] ChineseHarm-Bench: A Chinese Harmful Content Detection Benchmark
- **分类: cs.CL; cs.AI; cs.CR; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10960v2](http://arxiv.org/pdf/2506.10960v2)**

> **作者:** Kangwei Liu; Siyuan Cheng; Bozhong Tian; Xiaozhuan Liang; Yuyang Yin; Meng Han; Ningyu Zhang; Bryan Hooi; Xi Chen; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) have been increasingly applied to automated harmful content detection tasks, assisting moderators in identifying policy violations and improving the overall efficiency and accuracy of content review. However, existing resources for harmful content detection are predominantly focused on English, with Chinese datasets remaining scarce and often limited in scope. We present a comprehensive, professionally annotated benchmark for Chinese content harm detection, which covers six representative categories and is constructed entirely from real-world data. Our annotation process further yields a knowledge rule base that provides explicit expert knowledge to assist LLMs in Chinese harmful content detection. In addition, we propose a knowledge-augmented baseline that integrates both human-annotated knowledge rules and implicit knowledge from large language models, enabling smaller models to achieve performance comparable to state-of-the-art LLMs. Code and data are available at https://github.com/zjunlp/ChineseHarm-bench.
>
---
#### [replaced 011] Memorization in Fine-Tuned Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.21009v2](http://arxiv.org/pdf/2507.21009v2)**

> **作者:** Danil Savine
>
> **摘要:** This study investigates the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), with a focus on the medical domain due to its privacy-sensitive nature. We examine how different aspects of the fine-tuning process affect a model's propensity to memorize training data, using the PHEE dataset of pharmacovigilance events. Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning. Key findings include: (1) Value and Output matrices contribute more significantly to memorization compared to Query and Key matrices; (2) Lower perplexity in the fine-tuned model correlates with increased memorization; (3) Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks. These results provide insights into the trade-offs between model performance and privacy risks in fine-tuned LLMs. Our findings have implications for developing more effective and responsible strategies for adapting large language models while managing data privacy concerns.
>
---
#### [replaced 012] Evaluating LLMs on Real-World Forecasting Against Expert Forecasters
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04562v3](http://arxiv.org/pdf/2507.04562v3)**

> **作者:** Janna Lu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against top forecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of experts.
>
---
#### [replaced 013] SMART-Editor: A Multi-Agent Framework for Human-Like Design Editing with Structural Integrity
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.23095v2](http://arxiv.org/pdf/2507.23095v2)**

> **作者:** Ishani Mondal; Meera Bharadwaj; Ayush Roy; Aparna Garimella; Jordan Lee Boyd-Graber
>
> **备注:** This requires some internal approval before the public release
>
> **摘要:** We present SMART-Editor, a framework for compositional layout and content editing across structured (posters, websites) and unstructured (natural images) domains. Unlike prior models that perform local edits, SMART-Editor preserves global coherence through two strategies: Reward-Refine, an inference-time rewardguided refinement method, and RewardDPO, a training-time preference optimization approach using reward-aligned layout pairs. To evaluate model performance, we introduce SMARTEdit-Bench, a benchmark covering multi-domain, cascading edit scenarios. SMART-Editor outperforms strong baselines like InstructPix2Pix and HIVE, with RewardDPO achieving up to 15% gains in structured settings and Reward-Refine showing advantages on natural images. Automatic and human evaluations confirm the value of reward-guided planning in producing semantically consistent and visually aligned edits.
>
---
#### [replaced 014] CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01031v2](http://arxiv.org/pdf/2508.01031v2)**

> **作者:** Jingzhe Ni; Xiaolong Yin; Xingyu Lu; Xintong Li; Ji Wei; Ruofeng Tong; Min Tang; Peng Du
>
> **摘要:** Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both abstract textual descriptions and freehand sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Context-Independent Imperative Paradigm (CIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation.
>
---
#### [replaced 015] Aging Up AAC: An Introspection on Augmentative and Alternative Communication Applications for Autistic Adults
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17730v3](http://arxiv.org/pdf/2404.17730v3)**

> **作者:** Lara J. Martin; Malathy Nagalakshmi
>
> **摘要:** High-tech Augmentative and Alternative Communication (AAC) has been rapidly advancing in recent years due to the increased use of large language models (LLMs) like ChatGPT, but many of these techniques are integrated without the inclusion of the users' perspectives. Autistic adults have been particularly neglected in the design of AAC tools. We conducted in-depth interviews with 12 autistic adults to find the pain points of current AAC and determine what technological advances they might find helpful. We found 8 different categories of themes from our interviews: input flexibility, output flexibility, selecting or adapting AAC, contexts for AAC use, benefits, access as an adult, stumbling blocks for continued use, and control of communication. In this paper, we go through these categories in depth -- comparing each to prior work -- and then highlight novel findings to suggest possible research directions.
>
---
#### [replaced 016] ADS-Edit: A Multimodal Knowledge Editing Dataset for Autonomous Driving Systems
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.20756v3](http://arxiv.org/pdf/2503.20756v3)**

> **作者:** Chenxi Wang; Jizhan Fang; Xiang Chen; Bozhong Tian; Ziwen Xu; Huajun Chen; Ningyu Zhang
>
> **备注:** ACM MM 2025
>
> **摘要:** Recent advancements in Large Multimodal Models (LMMs) have shown promise in Autonomous Driving Systems (ADS). However, their direct application to ADS is hindered by challenges such as misunderstanding of traffic knowledge, complex road conditions, and diverse states of vehicle. To address these challenges, we propose the use of Knowledge Editing, which enables targeted modifications to a model's behavior without the need for full retraining. Meanwhile, we introduce ADS-Edit, a multimodal knowledge editing dataset specifically designed for ADS, which includes various real-world scenarios, multiple data types, and comprehensive evaluation metrics. We conduct comprehensive experiments and derive several interesting conclusions. We hope that our work will contribute to the further advancement of knowledge editing applications in the field of autonomous driving. Code and data are available in https://github.com/zjunlp/EasyEdit/blob/main/examples/ADSEdit.md.
>
---
#### [replaced 017] Document Haystack: A Long Context Multimodal Image/Document Understanding Vision LLM Benchmark
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.15882v2](http://arxiv.org/pdf/2507.15882v2)**

> **作者:** Goeric Huybrechts; Srikanth Ronanki; Sai Muralidhar Jayanthi; Jack Fitzgerald; Srinivasan Veeravanallur
>
> **摘要:** The proliferation of multimodal Large Language Models has significantly advanced the ability to analyze and understand complex data inputs from different modalities. However, the processing of long documents remains under-explored, largely due to a lack of suitable benchmarks. To address this, we introduce Document Haystack, a comprehensive benchmark designed to evaluate the performance of Vision Language Models (VLMs) on long, visually complex documents. Document Haystack features documents ranging from 5 to 200 pages and strategically inserts pure text or multimodal text+image "needles" at various depths within the documents to challenge VLMs' retrieval capabilities. Comprising 400 document variants and a total of 8,250 questions, it is supported by an objective, automated evaluation framework. We detail the construction and characteristics of the Document Haystack dataset, present results from prominent VLMs and discuss potential research avenues in this area.
>
---
#### [replaced 018] AdaMCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Multilingual Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.16154v3](http://arxiv.org/pdf/2501.16154v3)**

> **作者:** Weihua Zheng; Xin Huang; Zhengyuan Liu; Tarun Kumar Vangani; Bowei Zou; Xiyan Tao; Yuhao Wu; Ai Ti Aw; Nancy F. Chen; Roy Ka-Wei Lee
>
> **摘要:** Large language models (LLMs) have shown impressive multilingual capabilities through pretraining on diverse corpora. Although these models show strong reasoning abilities, their performance varies significantly between languages due to the imbalanced distribution of training data. Existing approaches using sample-level translation for extensive multilingual pretraining and cross-lingual tuning face scalability challenges and often fail to capture nuanced reasoning processes across languages. In this paper, we introduce AdaMCOT (Adaptive Multilingual Chain-of-Thought), a framework that enhances multilingual factual reasoning by dynamically routing thought processes in intermediary "thinking languages" before generating target-language responses. AdaMCOT leverages a language-agnostic core and incorporates an adaptive, reward-based mechanism for selecting optimal reasoning pathways without requiring additional pretraining. Our comprehensive evaluation across multiple benchmarks demonstrates substantial improvements in both factual reasoning quality and cross-lingual consistency, with particularly strong performance gains in low-resource language settings. An in-depth analysis of the model's hidden states and semantic space further elucidates the underlying mechanism of our method. The results suggest that adaptive reasoning paths can effectively bridge the performance gap between high and low-resource languages while maintaining cultural and linguistic nuances.
>
---
#### [replaced 019] Bridging LLMs and KGs without Fine-Tuning: Intermediate Probing Meets Subgraph-Aware Entity Descriptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.06787v4](http://arxiv.org/pdf/2408.06787v4)**

> **作者:** Bo Xue; Yi Xu; Yunchong Song; Jiaxin Ding; Luoyi Fu; Xinbing Wang
>
> **摘要:** Traditional knowledge graph completion (KGC) methods rely solely on structural information, struggling with the inherent sparsity of knowledge graphs (KGs). By contrast, Large Language Models (LLMs) encapsulate extensive world knowledge and exhibit powerful context modeling capabilities, making them promising for mitigating the limitations of traditional methods. However, direct fine-tuning of LLMs for KGC, though effective, imposes substantial computational and memory overheads, while utilizing non-fine-tuned LLMs is efficient but yields suboptimal performance. In this work, we propose a novel framework that synergizes the strengths of LLMs with robust knowledge representation to enable effective and efficient KGC. We extract the context-aware hidden states of knowledge triples from the intermediate layers of LLMs, thereby capturing rich semantic and relational nuances. These representations are then utilized to train a data-efficient classifier tailored specifically for KGC tasks. To bridge the semantic gaps between LLMs and KGs, we employ subgraph sampling on KGs to generate model-friendly entity descriptions. We further adopt sliced mutual information (SMI) as a principled metric to quantify the task-specific information encoded in these representations. Extensive experiments on standard benchmarks validate the efficiency and effectiveness of our approach. We achieve a 47\% relative improvement over previous methods based on non-fine-tuned LLMs and, to our knowledge, are the first to achieve classification performance comparable to fine-tuned LLMs while enhancing GPU memory efficiency by $188\times$ and accelerating training and inference by $26.11\times$.
>
---
#### [replaced 020] Science Hierarchography: Hierarchical Organization of Science Literature
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13834v2](http://arxiv.org/pdf/2504.13834v2)**

> **作者:** Muhan Gao; Jash Shah; Weiqi Wang; Daniel Khashabi
>
> **摘要:** Scientific knowledge is growing rapidly, making it difficult to track progress and high-level conceptual links across broad disciplines. While tools like citation networks and search engines help retrieve related papers, they lack the abstraction needed to capture the needed to represent the density and structure of activity across subfields. We motivate SCIENCE HIERARCHOGRAPHY, the goal of organizing scientific literature into a high-quality hierarchical structure that spans multiple levels of abstraction -- from broad domains to specific studies. Such a representation can provide insights into which fields are well-explored and which are under-explored. To achieve this goal, we develop a hybrid approach that combines efficient embedding-based clustering with LLM-based prompting, striking a balance between scalability and semantic precision. Compared to LLM-heavy methods like iterative tree construction, our approach achieves superior quality-speed trade-offs. Our hierarchies capture different dimensions of research contributions, reflecting the interdisciplinary and multifaceted nature of modern science. We evaluate its utility by measuring how effectively an LLM-based agent can navigate the hierarchy to locate target papers. Results show that our method improves interpretability and offers an alternative pathway for exploring scientific literature beyond traditional search methods. Code, data and demo are available: https://github.com/JHU-CLSP/science-hierarchography
>
---
#### [replaced 021] GEMA-Score: Granular Explainable Multi-Agent Scoring Framework for Radiology Report Evaluation
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2503.05347v2](http://arxiv.org/pdf/2503.05347v2)**

> **作者:** Zhenxuan Zhang; Kinhei Lee; Peiyuan Jing; Weihang Deng; Huichi Zhou; Zihao Jin; Jiahao Huang; Zhifan Gao; Dominic C Marshall; Yingying Fang; Guang Yang
>
> **摘要:** Automatic medical report generation has the potential to support clinical diagnosis, reduce the workload of radiologists, and demonstrate potential for enhancing diagnostic consistency. However, current evaluation metrics often fail to reflect the clinical reliability of generated reports. Early overlap-based methods focus on textual matches between predicted and ground-truth entities but miss fine-grained clinical details (e.g., anatomical location, severity). Some diagnostic metrics are limited by fixed vocabularies or templates, reducing their ability to capture diverse clinical expressions. LLM-based approaches further lack interpretable reasoning steps, making it hard to assess or trust their behavior in safety-critical settings. These limitations hinder the comprehensive assessment of the reliability of generated reports and pose risks in their selection for clinical use. Therefore, we propose a Granular Explainable Multi-Agent Score (GEMA-Score) in this paper, which conducts both objective quantification and subjective evaluation through a large language model-based multi-agent workflow. Our GEMA-Score parses structured reports and employs stable calculations through interactive exchanges of information among agents to assess disease diagnosis, location, severity, and uncertainty. Additionally, an LLM-based scoring agent evaluates completeness, readability, and clinical terminology while providing explanatory feedback. Extensive experiments validate that GEMA-Score achieves the highest correlation with human expert evaluations on a public dataset, demonstrating its effectiveness in clinical scoring (Kendall coefficient = $0.69$ for ReXVal dataset and Kendall coefficient = $0.45$ for RadEvalX dataset). The anonymous project demo is available at: https://github.com/Zhenxuan-Zhang/GEMA_score.
>
---
#### [replaced 022] Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19331v2](http://arxiv.org/pdf/2411.19331v2)**

> **作者:** Luca Barsellotti; Lorenzo Bianchi; Nicola Messina; Fabio Carrara; Marcella Cornia; Lorenzo Baraldi; Fabrizio Falchi; Rita Cucchiara
>
> **摘要:** Open-Vocabulary Segmentation (OVS) aims at segmenting images from free-form textual concepts without predefined training classes. While existing vision-language models such as CLIP can generate segmentation masks by leveraging coarse spatial information from Vision Transformers, they face challenges in spatial localization due to their global alignment of image and text features. Conversely, self-supervised visual models like DINO excel in fine-grained visual encoding but lack integration with language. To bridge this gap, we present Talk2DINO, a novel hybrid approach that combines the spatial accuracy of DINOv2 with the language understanding of CLIP. Our approach aligns the textual embeddings of CLIP to the patch-level features of DINOv2 through a learned mapping function without the need to fine-tune the underlying backbones. At training time, we exploit the attention maps of DINOv2 to selectively align local visual patches with textual embeddings. We show that the powerful semantic and localization abilities of Talk2DINO can enhance the segmentation process, resulting in more natural and less noisy segmentations, and that our approach can also effectively distinguish foreground objects from the background. Experimental results demonstrate that Talk2DINO achieves state-of-the-art performance across several unsupervised OVS benchmarks. Source code and models are publicly available at: https://lorebianchi98.github.io/Talk2DINO/.
>
---
#### [replaced 023] MemOS: A Memory OS for AI System
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03724v3](http://arxiv.org/pdf/2507.03724v3)**

> **作者:** Zhiyu Li; Shichao Song; Chenyang Xi; Hanyu Wang; Chen Tang; Simin Niu; Ding Chen; Jiawei Yang; Chunyu Li; Qingchen Yu; Jihao Zhao; Yezhaohui Wang; Peng Liu; Zehao Lin; Pengyuan Wang; Jiahao Huo; Tianyi Chen; Kai Chen; Kehang Li; Zhen Tao; Huayi Lai; Hao Wu; Bo Tang; Zhenren Wang; Zhaoxin Fan; Ningyu Zhang; Linfeng Zhang; Junchi Yan; Mingchuan Yang; Tong Xu; Wei Xu; Huajun Chen; Haofen Wang; Hongkang Yang; Wentao Zhang; Zhi-Qin John Xu; Siheng Chen; Feiyu Xiong
>
> **备注:** 36 pages, 10 figures, 5 tables
>
> **摘要:** Large Language Models (LLMs) have become an essential infrastructure for Artificial General Intelligence (AGI), yet their lack of well-defined memory management systems hinders the development of long-context reasoning, continual personalization, and knowledge consistency.Existing models mainly rely on static parameters and short-lived contextual states, limiting their ability to track user preferences or update knowledge over extended periods.While Retrieval-Augmented Generation (RAG) introduces external knowledge in plain text, it remains a stateless workaround without lifecycle control or integration with persistent representations.Recent work has modeled the training and inference cost of LLMs from a memory hierarchy perspective, showing that introducing an explicit memory layer between parameter memory and external retrieval can substantially reduce these costs by externalizing specific knowledge. Beyond computational efficiency, LLMs face broader challenges arising from how information is distributed over time and context, requiring systems capable of managing heterogeneous knowledge spanning different temporal scales and sources. To address this challenge, we propose MemOS, a memory operating system that treats memory as a manageable system resource. It unifies the representation, scheduling, and evolution of plaintext, activation-based, and parameter-level memories, enabling cost-efficient storage and retrieval. As the basic unit, a MemCube encapsulates both memory content and metadata such as provenance and versioning. MemCubes can be composed, migrated, and fused over time, enabling flexible transitions between memory types and bridging retrieval with parameter-based learning. MemOS establishes a memory-centric system framework that brings controllability, plasticity, and evolvability to LLMs, laying the foundation for continual learning and personalized modeling.
>
---
#### [replaced 024] Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.19794v3](http://arxiv.org/pdf/2506.19794v3)**

> **作者:** Yuqi Zhu; Yi Zhong; Jintian Zhang; Ziheng Zhang; Shuofei Qiao; Yujie Luo; Lun Du; Da Zheng; Ningyu Zhang; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate model behavior across three core dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. Code is available at https://github.com/zjunlp/DataMind.
>
---
#### [replaced 025] Pseudo-Autoregressive Neural Codec Language Models for Efficient Zero-Shot Text-to-Speech Synthesis
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10352v3](http://arxiv.org/pdf/2504.10352v3)**

> **作者:** Yifan Yang; Shujie Liu; Jinyu Li; Yuxuan Hu; Haibin Wu; Hui Wang; Jianwei Yu; Lingwei Meng; Haiyang Sun; Yanqing Liu; Yan Lu; Kai Yu; Xie Chen
>
> **备注:** Accepted in ACMMM 2025
>
> **摘要:** Recent zero-shot text-to-speech (TTS) systems face a common dilemma: autoregressive (AR) models suffer from slow generation and lack duration controllability, while non-autoregressive (NAR) models lack temporal modeling and typically require complex designs. In this paper, we introduce a novel pseudo-autoregressive (PAR) codec language modeling approach that unifies AR and NAR modeling. Combining explicit temporal modeling from AR with parallel generation from NAR, PAR generates dynamic-length spans at fixed time steps. Building on PAR, we propose PALLE, a two-stage TTS system that leverages PAR for initial generation followed by NAR refinement. In the first stage, PAR progressively generates speech tokens along the time dimension, with each step predicting all positions in parallel but only retaining the left-most span. In the second stage, low-confidence tokens are iteratively refined in parallel, leveraging the global contextual information. Experiments demonstrate that PALLE, trained on LibriTTS, outperforms state-of-the-art systems trained on large-scale data, including F5-TTS, E2-TTS, and MaskGCT, on the LibriSpeech test-clean set in terms of speech quality, speaker similarity, and intelligibility, while achieving up to ten times faster inference speed. Audio samples are available at https://microsoft.com/research/project/vall-e-x/palle.
>
---
#### [replaced 026] Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01191v2](http://arxiv.org/pdf/2508.01191v2)**

> **作者:** Chengshuai Zhao; Zhen Tan; Pingchuan Ma; Dawei Li; Bohan Jiang; Yancheng Wang; Yingzhen Yang; Huan Liu
>
> **摘要:** Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To investigate each dimension, we design DataAlchemy, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of why and when CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning.
>
---
#### [replaced 027] Think Outside the Data: Colonial Biases and Systemic Issues in Automated Moderation Pipelines for Low-Resource Languages
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.13836v3](http://arxiv.org/pdf/2501.13836v3)**

> **作者:** Farhana Shahid; Mona Elswah; Aditya Vashistha
>
> **备注:** Accepted to AIES 2025
>
> **摘要:** Most social media users come from the Global South, where harmful content usually appears in local languages. Yet, AI-driven moderation systems struggle with low-resource languages spoken in these regions. Through semi-structured interviews with 22 AI experts working on harmful content detection in four low-resource languages: Tamil (South Asia), Swahili (East Africa), Maghrebi Arabic (North Africa), and Quechua (South America)--we examine systemic issues in building automated moderation tools for these languages. Our findings reveal that beyond data scarcity, socio-political factors such as tech companies' monopoly on user data and lack of investment in moderation for low-profit Global South markets exacerbate historic inequities. Even if more data were available, the English-centric and data-intensive design of language models and preprocessing techniques overlooks the need to design for morphologically complex, linguistically diverse, and code-mixed languages. We argue these limitations are not just technical gaps caused by "data scarcity" but reflect structural inequities, rooted in colonial suppression of non-Western languages. We discuss multi-stakeholder approaches to strengthen local research capacity, democratize data access, and support language-aware solutions to improve automated moderation for low-resource languages.
>
---
#### [replaced 028] A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09037v3](http://arxiv.org/pdf/2504.09037v3)**

> **作者:** Zixuan Ke; Fangkai Jiao; Yifei Ming; Xuan-Phi Nguyen; Austin Xu; Do Xuan Long; Minzhi Li; Chengwei Qin; Peifeng Wang; Silvio Savarese; Caiming Xiong; Shafiq Joty
>
> **备注:** 72 pages, 6 figures. Accepted to TMLR, with Survey Certification award
>
> **摘要:** Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
>
---
#### [replaced 029] Antidistillation Sampling
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13146v3](http://arxiv.org/pdf/2504.13146v3)**

> **作者:** Yash Savani; Asher Trockman; Zhili Feng; Avi Schwarzschild; Alexander Robey; Marc Finzi; J. Zico Kolter
>
> **摘要:** Frontier models that generate extended reasoning traces inadvertently produce rich token sequences that can facilitate model distillation. Recognizing this vulnerability, model owners may seek sampling strategies that limit the effectiveness of distillation without compromising model performance. Antidistillation sampling provides exactly this capability. By strategically modifying a model's next-token probability distribution, antidistillation sampling poisons reasoning traces, rendering them significantly less effective for distillation while preserving the model's practical utility. For further details, see https://antidistillation.com.
>
---
#### [replaced 030] Filtering with Self-Attention and Storing with MLP: One-Layer Transformers Can Provably Acquire and Extract Knowledge
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00901v2](http://arxiv.org/pdf/2508.00901v2)**

> **作者:** Ruichen Xu; Kexin Chen
>
> **摘要:** Modern large language models excel in knowledge-intensive tasks, yet how transformers acquire (store) knowledge during pre-training and extract (retrieve) it during post-fine-tuning inference remains theoretically opaque. While prior theoretical work has begun to investigate these questions through the analysis of training dynamics, such studies are limited to single-layer, attention-only architectures. However, most existing studies suggest that MLPs are the most contributing components for storing knowledge in transformer-based language models. Meanwhile, our empirical investigations reveal that such simplified models, when trained using standard next-token prediction objectives, may be incapable of acquiring or extracting factual knowledge. To overcome this limitation, we introduce a tractable one-layer transformer framework that crucially incorporates both self-attention and MLP modules. By tracking its gradient dynamics, we establish convergence and generalization guarantees that illuminate the ability of knowledge acquisition and extraction. We prove that 1) Transformers can achieve near-optimal training loss during pre-training, signifying effective knowledge acquisition; 2) With a large fine-tuning dataset and specific data multiplicity conditions met, transformers can achieve low generalization error when tested on factual knowledge learned during pre-training but not reinforced during the fine-tuning, indicating successful knowledge extraction; 3) When the conditions are not satisfied, transformers exhibit high generalization loss, resulting in hallucinations. Our analysis includes both full fine-tuning and low-rank fine-tuning. Furthermore, our analysis offers theoretical insights into several pertinent empirical phenomena, such as the role of learning rate schedules. Experiments on synthetic and real-world PopQA datasets with GPT-2 and Llama-3.2-1B validate our results.
>
---
#### [replaced 031] Pre-trained Transformer-Based Approach for Arabic Question Answering : A Comparative Study
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2111.05671v2](http://arxiv.org/pdf/2111.05671v2)**

> **作者:** Kholoud Alsubhi; Amani Jamal; Areej Alhothali
>
> **备注:** Rewrite the paper
>
> **摘要:** Question answering(QA) is one of the most challenging yet widely investigated problems in Natural Language Processing (NLP). Question-answering (QA) systems try to produce answers for given questions. These answers can be generated from unstructured or structured text. Hence, QA is considered an important research area that can be used in evaluating text understanding systems. A large volume of QA studies was devoted to the English language, investigating the most advanced techniques and achieving state-of-the-art results. However, research efforts in the Arabic question-answering progress at a considerably slower pace due to the scarcity of research efforts in Arabic QA and the lack of large benchmark datasets. Recently many pre-trained language models provided high performance in many Arabic NLP problems. In this work, we evaluate the state-of-the-art pre-trained transformers models for Arabic QA using four reading comprehension datasets which are Arabic-SQuAD, ARCD, AQAD, and TyDiQA-GoldP datasets. We fine-tuned and compared the performance of the AraBERTv2-base model, AraBERTv0.2-large model, and AraELECTRA model. In the last, we provide an analysis to understand and interpret the low-performance results obtained by some models.
>
---
#### [replaced 032] WSI-LLaVA: A Multimodal Large Language Model for Whole Slide Image
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.02141v3](http://arxiv.org/pdf/2412.02141v3)**

> **作者:** Yuci Liang; Xinheng Lyu; Meidan Ding; Wenting Chen; Jipeng Zhang; Yuexiang Ren; Xiangjian He; Song Wu; Sen Yang; Xiyue Wang; Xiaohan Xing; Linlin Shen
>
> **备注:** ICCV 2025, 38 pages, 22 figures, 35 tables
>
> **摘要:** Recent advancements in computational pathology have produced patch-level Multi-modal Large Language Models (MLLMs), but these models are limited by their inability to analyze whole slide images (WSIs) comprehensively and their tendency to bypass crucial morphological features that pathologists rely on for diagnosis. To address these challenges, we first introduce WSI-Bench, a large-scale morphology-aware benchmark containing 180k VQA pairs from 9,850 WSIs across 30 cancer types, designed to evaluate MLLMs' understanding of morphological characteristics crucial for accurate diagnosis. Building upon this benchmark, we present WSI-LLaVA, a novel framework for gigapixel WSI understanding that employs a three-stage training approach: WSI-text alignment, feature space alignment, and task-specific instruction tuning. To better assess model performance in pathological contexts, we develop two specialized WSI metrics: WSI-Precision and WSI-Relevance. Experimental results demonstrate that WSI-LLaVA outperforms existing models across all capability dimensions, with a significant improvement in morphological analysis, establishing a clear correlation between morphological understanding and diagnostic accuracy.
>
---
#### [replaced 033] What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12537v2](http://arxiv.org/pdf/2506.12537v2)**

> **作者:** Xiaoran Fan; Zhichao Sun; Yangfan Gao; Jingfei Xiong; Hang Yan; Yifei Cao; Jiajun Sun; Shuo Li; Zhihao Zhang; Zhiheng Xi; Yuhao Zhou; Senjie Jin; Changhao Jiang; Junjie Ye; Ming Zhang; Rui Zheng; Zhenhua Han; Yunke Zhang; Demei Yan; Shaokang Dong; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the role of speech tokenizer designs in LLM-centric SLMs, augmented by speech heads and speaker modeling. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.
>
---
#### [replaced 034] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02308v2](http://arxiv.org/pdf/2508.02308v2)**

> **作者:** Sikui Zhang; Guangze Gao; Ziyun Gan; Chunfeng Yuan; Zefeng Lin; Houwen Peng; Bing Li; Weiming Hu
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Large language models (LLMs) experience significant performance degradation when the input exceeds the pretraining context window, primarily due to the out-of-distribution (OOD) behavior of Rotary Position Embedding (RoPE). Recent studies mitigate this problem by remapping OOD positions into the in-distribution range with fixed mapping strategies, ignoring the dynamic relationship between input length and the model's effective context window. To this end, we propose Length-aware Multi-grained Positional Encoding (LaMPE), a training-free method that fully utilizes the model's effective context window for adaptive long-context scaling in LLMs. Motivated by the left-skewed frequency distribution of relative positions, LaMPE establishes a dynamic relationship between mapping length and input length through a parametric scaled sigmoid function to adaptively allocate positional capacity across varying input lengths. Meanwhile, LaMPE devises a novel multi-grained attention mechanism that strategically allocates positional resolution across different sequence regions to capture both fine-grained locality and long-range dependencies. Our method can be seamlessly applied to a wide range of RoPE-based LLMs without training. Extensive experiments on three representative LLMs across five mainstream long-context benchmarks demonstrate that LaMPE achieves significant performance improvements compared to existing length extrapolation methods. The code will be released at https://github.com/scar-on/LaMPE.
>
---
#### [replaced 035] Proof2Hybrid: Automatic Mathematical Benchmark Synthesis for Proof-Centric Problems
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.02208v2](http://arxiv.org/pdf/2508.02208v2)**

> **作者:** Yebo Peng; Zixiang Liu; Yaoming Li; Zhizhuo Yang; Xinye Xu; Bowen Ye; Weijun Yuan; Zihan Wang; Tong Yang
>
> **摘要:** Evaluating the mathematical capability of Large Language Models (LLMs) is a critical yet challenging frontier. Existing benchmarks fall short, particularly for proof-centric problems, as manual creation is unscalable and costly, leaving the true mathematical abilities of LLMs largely unassessed. To overcome these barriers, we propose Proof2Hybrid, the first fully automated framework that synthesizes high-quality, proof-centric benchmarks from natural language mathematical corpora. The key novelty of our solution is Proof2X, a roadmap of converting mathematical proofs into various kinds of questions that are easy to verify. Instructed by this roadmap, we propose a new type of hybrid-formatted questions, named ``$m$-out-of-$n$ multiple judge questions'', specifically designed to enable robust, automatic evaluation while being resilient to guessing and superficial pattern matching inherent in traditional formats. As a demonstration of our framework, we introduce AlgGeoTest, a benchmark for algebraic geometry--a frontier domain of modern mathematics--comprising 456 challenging items. Our extensive evaluations on state-of-the-art LLMs using AlgGeoTest reveal profound deficits in their comprehension of algebraic geometry, providing a more precise measure of their true mathematical capabilities. Our framework and benchmark pave the way for a new wave of in-depth research into the mathematical intelligence of AI systems.
>
---
#### [replaced 036] Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.06219v2](http://arxiv.org/pdf/2504.06219v2)**

> **作者:** Dongyang Fan; Vinko Sabolčec; Matin Ansaripour; Ayush Kumar Tarun; Martin Jaggi; Antoine Bosselut; Imanol Schlag
>
> **备注:** COLM 2025 Camera Ready version
>
> **摘要:** The increasing adoption of web crawling opt-outs by copyright holders of online content raises critical questions about the impact of data compliance on large language model (LLM) performance. However, little is known about how these restrictions (and the resultant filtering of pretraining datasets) affect the capabilities of models trained using these corpora. In this work, we conceptualize this effect as the $\textit{data compliance gap}$ (DCG), which quantifies the performance difference between models trained on datasets that comply with web crawling opt-outs, and those that do not. We measure the data compliance gap in two settings: pretraining models from scratch and continual pretraining from existing compliant models (simulating a setting where copyrighted data could be integrated later in pretraining). Our experiments with 1.5B models show that, as of January 2025, compliance with web data opt-outs does not degrade general knowledge acquisition (close to 0\% DCG). However, in specialized domains such as biomedical research, excluding major publishers leads to performance declines. These findings suggest that while general-purpose LLMs can be trained to perform equally well using fully open data, performance in specialized domains may benefit from access to high-quality copyrighted sources later in training. Our study provides empirical insights into the long-debated trade-off between data compliance and downstream model performance, informing future discussions on AI training practices and policy decisions. Our website is available at https://data-compliance.github.io/.
>
---
#### [replaced 037] Post-Completion Learning for Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.20252v2](http://arxiv.org/pdf/2507.20252v2)**

> **作者:** Xiang Fei; Siqi Wang; Shu Wei; Yuxiang Nie; Wei Shi; Hao Feng; Chao Feng; Can Huang
>
> **摘要:** Current language model training paradigms typically terminate learning upon reaching the end-of-sequence (<eos>) token, overlooking the potential learning opportunities in the post-completion space. We propose Post-Completion Learning (PCL), a novel training framework that systematically utilizes the sequence space after model output completion, to enhance both the reasoning and self-evaluation abilities. PCL enables models to continue generating self-assessments and reward predictions during training, while maintaining efficient inference by stopping at the completion point. To fully utilize this post-completion space, we design a white-box reinforcement learning method: let the model evaluate the output content according to the reward rules, then calculate and align the score with the reward functions for supervision. We implement dual-track SFT to optimize both reasoning and evaluation capabilities, and mixed it with RL training to achieve multi-objective hybrid optimization. Experimental results on different datasets and models demonstrate consistent improvements over traditional SFT and RL methods. Our method provides a new technical path for language model training that enhances output quality while preserving deployment efficiency.
>
---
#### [replaced 038] Multilingual Performance Biases of Large Language Models in Education
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.17720v2](http://arxiv.org/pdf/2504.17720v2)**

> **作者:** Vansh Gupta; Sankalan Pal Chowdhury; Vilém Zouhar; Donya Rooein; Mrinmaya Sachan
>
> **摘要:** Large language models (LLMs) are increasingly being adopted in educational settings. These applications expand beyond English, though current LLMs remain primarily English-centric. In this work, we ascertain if their use in education settings in non-English languages is warranted. We evaluated the performance of popular LLMs on four educational tasks: identifying student misconceptions, providing targeted feedback, interactive tutoring, and grading translations in eight languages (Mandarin, Hindi, Arabic, German, Farsi, Telugu, Ukrainian, Czech) in addition to English. We find that the performance on these tasks somewhat corresponds to the amount of language represented in training data, with lower-resource languages having poorer task performance. Although the models perform reasonably well in most languages, the frequent performance drop from English is significant. Thus, we recommend that practitioners first verify that the LLM works well in the target language for their educational task before deployment.
>
---
#### [replaced 039] Principled Foundations for Preference Optimization
- **分类: cs.LG; cs.AI; cs.CL; I.2.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.07855v2](http://arxiv.org/pdf/2507.07855v2)**

> **作者:** Wenxuan Zhou; Shujian Zhang; Brice Magdalou; John Lambert; Ehsan Amid; Richard Nock; Andrew Hard
>
> **摘要:** In this paper, we show that direct preference optimization (DPO) is a very specific form of a connection between two major theories in the ML context of learning from preferences: loss functions (Savage) and stochastic choice (Doignon-Falmagne and Machina). The connection is established for all of Savage's losses and at this level of generality, (i) it includes support for abstention on the choice theory side, (ii) it includes support for non-convex objectives on the ML side, and (iii) it allows to frame for free some notable extensions of the DPO setting, including margins and corrections for length. Getting to understand how DPO operates from a general principled perspective is crucial because of the huge and diverse application landscape of models, because of the current momentum around DPO, but also -- and importantly -- because many state of the art variations on DPO definitely occupy a small region of the map that we cover. It also helps to understand the pitfalls of departing from this map, and figure out workarounds.
>
---
#### [replaced 040] Energy-Based Reward Models for Robust Language Model Alignment
- **分类: cs.CL; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.13134v2](http://arxiv.org/pdf/2504.13134v2)**

> **作者:** Anamika Lochab; Ruqi Zhang
>
> **备注:** Accepted by COLM 2025
>
> **摘要:** Reward models (RMs) are essential for aligning Large Language Models (LLMs) with human preferences. However, they often struggle with capturing complex human preferences and generalizing to unseen data. To address these challenges, we introduce Energy-Based Reward Model (EBRM), a lightweight post-hoc refinement framework that enhances RM robustness and generalization. EBRM models the reward distribution explicitly, capturing uncertainty in human preferences and mitigating the impact of noisy or misaligned annotations. It achieves this through conflict-aware data filtering, label-noise-aware contrastive training, and hybrid initialization. Notably, EBRM enhances RMs without retraining, making it computationally efficient and adaptable across different models and tasks. Empirical evaluations on RM benchmarks demonstrate significant improvements in both robustness and generalization, achieving up to a 5.97% improvement in safety-critical alignment tasks compared to standard RMs. Furthermore, reinforcement learning experiments confirm that our refined rewards enhance alignment quality, effectively delaying reward hacking. These results demonstrate our approach as a scalable and effective enhancement for existing RMs and alignment pipelines. The code is available at EBRM.
>
---
#### [replaced 041] SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02013v2](http://arxiv.org/pdf/2508.02013v2)**

> **作者:** Changhao Jiang; Jiajun Sun; Yifei Cao; Jiabao Zhuang; Hui Li; Xiaoran Fan; Ming Zhang; Junjie Ye; Shihan Dou; Zhiheng Xi; Jingqi Tong; Yilong Wu; Baoyu Fan; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** We request withdrawal of this paper due to an error in the experimental results reported in Table 2 on page 8. Specifically, the results for the Qwen2.5-Omni model are incorrect. We are currently conducting further verification and plan to resubmit with corrected results
>
> **摘要:** Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field.
>
---
#### [replaced 042] A Comprehensive Evaluation of Semantic Relation Knowledge of Pretrained Language Models and Humans
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01131v5](http://arxiv.org/pdf/2412.01131v5)**

> **作者:** Zhihan Cao; Hiroaki Yamada; Simone Teufel; Takenobu Tokunaga
>
> **摘要:** Recently, much work has concerned itself with the enigma of what exactly pretrained language models~(PLMs) learn about different aspects of language, and how they learn it. One stream of this type of research investigates the knowledge that PLMs have about semantic relations. However, many aspects of semantic relations were left unexplored. Generally, only one relation has been considered, namely hypernymy. Furthermore, previous work did not measure humans' performance on the same task as that performed by the PLMs. This means that at this point in time, there is only an incomplete view of the extent of these models' semantic relation knowledge. To address this gap, we introduce a comprehensive evaluation framework covering five relations beyond hypernymy, namely hyponymy, holonymy, meronymy, antonymy, and synonymy. We use five metrics (two newly introduced here) for recently untreated aspects of semantic relation knowledge, namely soundness, completeness, symmetry, prototypicality, and distinguishability. Using these, we can fairly compare humans and models on the same task. Our extensive experiments involve six PLMs, four masked and two causal language models. The results reveal a significant knowledge gap between humans and models for all semantic relations. In general, causal language models, despite their wide use, do not always perform significantly better than masked language models. Antonymy is the outlier relation where all models perform reasonably well. The evaluation materials can be found at https://github.com/hancules/ProbeResponses.
>
---
#### [replaced 043] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00222v2](http://arxiv.org/pdf/2508.00222v2)**

> **作者:** Yihong Dong; Xue Jiang; Yongding Tao; Huanyu Liu; Kechi Zhang; Lili Mou; Rongyu Cao; Yingwei Ma; Jue Chen; Binhua Li; Zhi Jin; Fei Huang; Yongbin Li; Ge Li
>
> **摘要:** Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address for distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
>
---
#### [replaced 044] CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.17234v2](http://arxiv.org/pdf/2507.17234v2)**

> **作者:** Kyeongkyu Lee; Seonghwan Yoon; Hongki Lim
>
> **摘要:** Automatic generation of radiology reports has the potential to alleviate radiologists' significant workload, yet current methods struggle to deliver clinically reliable conclusions. In particular, most prior approaches focus on producing fluent text without effectively ensuring the factual correctness of the reports and often rely on single-view images, limiting diagnostic comprehensiveness. We propose CLARIFID, a novel framework that directly optimizes diagnostic correctness by mirroring the two-step workflow of experts. Specifically, CLARIFID (1) learns the logical flow from Findings to Impression through section-aware pretraining, (2) is fine-tuned with Proximal Policy Optimization in which the CheXbert F1 score of the Impression section serves as the reward, (3) enforces reasoning-aware decoding that completes "Findings" before synthesizing the "Impression", and (4) fuses multiple chest X-ray views via a vision-transformer-based multi-view encoder. During inference, we apply a reasoning-aware next-token forcing strategy followed by report-level re-ranking, ensuring that the model first produces a comprehensive Findings section before synthesizing the Impression and thereby preserving coherent clinical reasoning. Experimental results on the MIMIC-CXR dataset demonstrate that our method achieves superior clinical efficacy and outperforms existing baselines on both standard NLG metrics and clinically aware scores.
>
---
#### [replaced 045] Mind the Gap: The Divergence Between Human and LLM-Generated Tasks
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00282v2](http://arxiv.org/pdf/2508.00282v2)**

> **作者:** Yi-Long Lu; Jiajun Song; Chunhui Zhang; Wei Wang
>
> **摘要:** Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied goals. We conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents.
>
---
#### [replaced 046] AI4Research: A Survey of Artificial Intelligence for Scientific Research
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01903v2](http://arxiv.org/pdf/2507.01903v2)**

> **作者:** Qiguang Chen; Mingda Yang; Libo Qin; Jinhao Liu; Zheng Yan; Jiannan Guan; Dengyun Peng; Yiyan Ji; Hanjing Li; Mengkang Hu; Yimeng Zhang; Yihao Liang; Yuhang Zhou; Jiaqi Wang; Zhi Chen; Wanxiang Che
>
> **备注:** Preprint, Paper list is available at https://github.com/LightChen233/Awesome-AI4Research
>
> **摘要:** Recent advancements in artificial intelligence (AI), particularly in large language models (LLMs) such as OpenAI-o1 and DeepSeek-R1, have demonstrated remarkable capabilities in complex domains such as logical reasoning and experimental coding. Motivated by these advancements, numerous studies have explored the application of AI in the innovation process, particularly in the context of scientific research. These AI technologies primarily aim to develop systems that can autonomously conduct research processes across a wide range of scientific disciplines. Despite these significant strides, a comprehensive survey on AI for Research (AI4Research) remains absent, which hampers our understanding and impedes further development in this field. To address this gap, we present a comprehensive survey and offer a unified perspective on AI4Research. Specifically, the main contributions of our work are as follows: (1) Systematic taxonomy: We first introduce a systematic taxonomy to classify five mainstream tasks in AI4Research. (2) New frontiers: Then, we identify key research gaps and highlight promising future directions, focusing on the rigor and scalability of automated experiments, as well as the societal impact. (3) Abundant applications and resources: Finally, we compile a wealth of resources, including relevant multidisciplinary applications, data corpora, and tools. We hope our work will provide the research community with quick access to these resources and stimulate innovative breakthroughs in AI4Research.
>
---
#### [replaced 047] ProRefine: Inference-Time Prompt Refinement with Textual Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05305v2](http://arxiv.org/pdf/2506.05305v2)**

> **作者:** Deepak Pandita; Tharindu Cyril Weerasooriya; Ankit Parag Shah; Isabelle Diana May-Xin Ng; Christopher M. Homan; Wei Wei
>
> **摘要:** Agentic workflows, where multiple AI agents collaborate to accomplish complex tasks like reasoning or planning, play a substantial role in many cutting-edge commercial applications, and continue to fascinate researchers across nearly all fields for their potential to accomplish expensive, complex tasks that, until recently, only humans have been trusted to do. These workflows depend critically on the prompts used to provide the roles models play in such workflows. Poorly designed prompts that fail even slightly to guide individual agents can lead to sub-optimal performance that may snowball within a system of agents, limiting their reliability and scalability. To address this important problem of inference-time prompt optimization, we introduce ProRefine, an innovative inference-time optimization method that uses an agentic loop of LLMs to generate and apply textual feedback. ProRefine dynamically refines prompts for multi-step reasoning tasks without additional training or ground truth labels. Evaluated on five benchmark mathematical reasoning datasets, ProRefine significantly surpasses zero-shot Chain-of-Thought baselines by 3 to 37 percentage points. This approach not only boosts accuracy but also allows smaller models to approach the performance of their larger counterparts. This highlights its potential for building more cost-effective and powerful hybrid AI systems, thereby democratizing access to high-performing AI.
>
---
#### [replaced 048] Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01696v2](http://arxiv.org/pdf/2508.01696v2)**

> **作者:** Yi Jiang; Sendong Zhao; Jianbo Li; Haochun Wang; Lizhe Zhang; Yan Liu; Bing Qin
>
> **备注:** code available at https://github.com/liunian-Jay/CoCoA
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising framework for enhancing the capabilities of Large Language Models (LLMs), especially in knowledge-intensive tasks. Despite its advantages, current RAG methods often struggle to *fully exploit knowledge during generation*. In particular, the synergy between the model's internal parametric knowledge and external retrieved knowledge remains limited. Retrieved contents may sometimes mislead generation, while certain generated content can guide the model toward more accurate outputs. In this work, we propose Collaborative Chain-of-Agents, a framework designed to enhance explicitly synergy over both parametric and retrieved knowledge. Specifically, we first introduce CoCoA-zero, a multi-agent RAG framework that first performs conditional knowledge induction and then reasons answers. Building on this, we develop CoCoA, a long-chain training strategy that synthesizes extended multi-agent reasoning trajectories from CoCoA-zero to fine-tune the LLM. This strategy enhances the model's capability to explicitly integrate and jointly leverage parametric and retrieved knowledge. Experiments results show that CoCoA-zero and CoCoA achieve superior performance on open-domain and multi-hop QA tasks.
>
---
#### [replaced 049] Listening to the Unspoken: Exploring "365" Aspects of Multimodal Interview Performance Assessment
- **分类: cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.22676v3](http://arxiv.org/pdf/2507.22676v3)**

> **作者:** Jia Li; Yang Wang; Wenhao Qian; Jialong Hu; Zhenzhen Hu; Richang Hong; Meng Wang
>
> **备注:** 8 pages, 4 figures, ACM MM 2025. github:https://github.com/MSA-LMC/365Aspects
>
> **摘要:** Interview performance assessment is essential for determining candidates' suitability for professional positions. To ensure holistic and fair evaluations, we propose a novel and comprehensive framework that explores ``365'' aspects of interview performance by integrating \textit{three} modalities (video, audio, and text), \textit{six} responses per candidate, and \textit{five} key evaluation dimensions. The framework employs modality-specific feature extractors to encode heterogeneous data streams and subsequently fused via a Shared Compression Multilayer Perceptron. This module compresses multimodal embeddings into a unified latent space, facilitating efficient feature interaction. To enhance prediction robustness, we incorporate a two-level ensemble learning strategy: (1) independent regression heads predict scores for each response, and (2) predictions are aggregated across responses using a mean-pooling mechanism to produce final scores for the five target dimensions. By listening to the unspoken, our approach captures both explicit and implicit cues from multimodal data, enabling comprehensive and unbiased assessments. Achieving a multi-dimensional average MSE of 0.1824, our framework secured first place in the AVI Challenge 2025, demonstrating its effectiveness and robustness in advancing automated and multimodal interview performance assessment. The full implementation is available at https://github.com/MSA-LMC/365Aspects.
>
---
#### [replaced 050] Out-of-Context Relational Reasoning in Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10408v2](http://arxiv.org/pdf/2503.10408v2)**

> **作者:** Jonathan Shaki; Emanuele La Malfa; Michael Wooldridge; Sarit Kraus
>
> **摘要:** Binary relations, such as equality, are basic mathematical concepts that appear, implicitly or explicitly, in most benchmarks for Large Language Models (LLM). A recent trend in the literature is benchmarking LLMs on out-of-context learning, where the data is not presented in the prompt, but only during the model's training. However, existing works mostly focus on higher-order tasks, making it hard to interpret success or failure. In this work, we study how well can LLMs reason out-of-context on binary relations by only learning the representations of newly introduced tokens. Our experiments focus on equality ($=$), inequality ($<$), and inclusion ($\subset$) and the properties they satisfy, such as reflexivity, symmetry, transitivity, and logical complexity (e.g., the number of reasoning "hops"). We show that LLMs achieve better than random accuracy, but are still far from perfect, even on relatively simple reasoning tasks involving binary relations. We analyse the learned representations and show that LLMs encode useful information directly, arranging the embeddings according to the task.
>
---
#### [replaced 051] VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo
- **分类: cs.CL; cs.AI; cs.DC**

- **链接: [http://arxiv.org/pdf/2508.02317v2](http://arxiv.org/pdf/2508.02317v2)**

> **作者:** Qianli Ma; Yaowei Zheng; Zhelun Shi; Zhongkai Zhao; Bin Jia; Ziyue Huang; Zhiqi Lin; Youjie Li; Jiacheng Yang; Yanghua Peng; Zhi Zhang; Xin Liu
>
> **摘要:** Recent advances in large language models (LLMs) have driven impressive progress in omni-modal understanding and generation. However, training omni-modal LLMs remains a significant challenge due to the heterogeneous model architectures required to process diverse modalities, necessitating sophisticated system design for efficient large-scale training. Existing frameworks typically entangle model definition with parallel logic, incurring limited scalability and substantial engineering overhead for end-to-end omni-modal training. We present VeOmni, a modular and efficient training framework to accelerate the development of omni-modal LLMs. VeOmni introduces model-centric distributed recipes that decouples communication from computation, enabling efficient 3D parallelism on omni-modal LLMs. VeOmni also features a flexible configuration interface supporting seamless integration of new modalities with minimal code change. Using VeOmni, a omni-modal mixture-of-experts (MoE) model with 30B parameters can be trained with over 2,800 tokens/sec/GPU throughput and scale to 160K context lengths via 3D parallelism on 128 GPUs, showcasing its superior efficiency and scalability for training large omni-modal LLMs.
>
---
#### [replaced 052] STRUCTSENSE: A Task-Agnostic Agentic Framework for Structured Information Extraction with Human-In-The-Loop Evaluation and Benchmarking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03674v2](http://arxiv.org/pdf/2507.03674v2)**

> **作者:** Tek Raj Chhetri; Yibei Chen; Puja Trivedi; Dorota Jarecka; Saif Haobsh; Patrick Ray; Lydia Ng; Satrajit S. Ghosh
>
> **备注:** -
>
> **摘要:** The ability to extract structured information from unstructured sources-such as free-text documents and scientific literature-is critical for accelerating scientific discovery and knowledge synthesis. Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks, including structured information extraction. However, their effectiveness often diminishes in specialized, domain-specific contexts that require nuanced understanding and expert-level domain knowledge. In addition, existing LLM-based approaches frequently exhibit poor transferability across tasks and domains, limiting their scalability and adaptability. To address these challenges, we introduce StructSense, a modular, task-agnostic, open-source framework for structured information extraction built on LLMs. StructSense is guided by domain-specific symbolic knowledge encoded in ontologies, enabling it to navigate complex domain content more effectively. It further incorporates agentic capabilities through self-evaluative judges that form a feedback loop for iterative refinement, and includes human-in-the-loop mechanisms to ensure quality and validation. We demonstrate that StructSense can overcome both the limitations of domain sensitivity and the lack of cross-task generalizability, as shown through its application to diverse neuroscience information extraction tasks.
>
---
#### [replaced 053] ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01365v2](http://arxiv.org/pdf/2508.01365v2)**

> **作者:** Zihan Wang; Rui Zhang; Hongwei Li; Wenshu Fan; Wenbo Jiang; Qingchuan Zhao; Guowen Xu
>
> **备注:** Under review
>
> **摘要:** Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments.
>
---
#### [replaced 054] Domain-Independent Automatic Generation of Descriptive Texts for Time-Series Data
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.16647v2](http://arxiv.org/pdf/2409.16647v2)**

> **作者:** Kota Dohi; Aoi Ito; Harsh Purohit; Tomoya Nishida; Takashi Endo; Yohei Kawaguchi
>
> **摘要:** Due to scarcity of time-series data annotated with descriptive texts, training a model to generate descriptive texts for time-series data is challenging. In this study, we propose a method to systematically generate domain-independent descriptive texts from time-series data. We identify two distinct approaches for creating pairs of time-series data and descriptive texts: the forward approach and the backward approach. By implementing the novel backward approach, we create the Temporal Automated Captions for Observations (TACO) dataset. Experimental results demonstrate that a contrastive learning based model trained using the TACO dataset is capable of generating descriptive texts for time-series data in novel domains.
>
---
#### [replaced 055] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10532v2](http://arxiv.org/pdf/2507.10532v2)**

> **作者:** Mingqi Wu; Zhihao Zhang; Qiaole Dong; Zhiheng Xi; Jun Zhao; Senjie Jin; Xiaoran Fan; Yuhao Zhou; Huijie Lv; Ming Zhang; Yanwei Fu; Qin Liu; Songyang Zhang; Qi Zhang
>
> **备注:** 33 pages
>
> **摘要:** Reasoning in large language models has long been a central research focus, and recent studies employing reinforcement learning (RL) have introduced diverse methods that yield substantial performance gains with minimal or even no external supervision. Surprisingly, some studies even suggest that random or incorrect reward signals can enhance performance. However, these breakthroughs are predominantly observed for the mathematically strong Qwen2.5 series on benchmarks such as MATH-500, AMC, and AIME, and seldom transfer to models like Llama, which warrants a more in-depth investigation. In this work, our empirical analysis reveals that pre-training on massive web-scale corpora leaves Qwen2.5 susceptible to data contamination in widely used benchmarks. Consequently, conclusions derived from contaminated benchmarks on Qwen2.5 series may be unreliable. To obtain trustworthy evaluation results, we introduce a generator that creates fully clean arithmetic problems of arbitrary length and difficulty, dubbed RandomCalculation. Using this leakage-free dataset, we show that only accurate reward signals yield steady improvements that surpass the base model's performance boundary in mathematical reasoning, whereas random or incorrect rewards do not. Moreover, we conduct more fine-grained analyses to elucidate the factors underlying the different performance observed on the MATH-500 and RandomCalculation benchmarks. Consequently, we recommend that future studies evaluate models on uncontaminated benchmarks and, when feasible, test various model series to ensure trustworthy conclusions about RL and related methods.
>
---
#### [replaced 056] Ensemble Learning for Large Language Models in Text and Code Generation: A Survey
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.13505v2](http://arxiv.org/pdf/2503.13505v2)**

> **作者:** Mari Ashiga; Wei Jie; Fan Wu; Vardan Voskanyan; Fateme Dinmohammadi; Paul Brookes; Jingzhi Gong; Zheng Wang
>
> **备注:** Under review by IEEE TAI
>
> **摘要:** Generative Pretrained Transformers (GPTs) are foundational Large Language Models (LLMs) for text generation. However, individual LLMs often produce inconsistent outputs and exhibit biases, limiting their representation of diverse language patterns. The closed-source nature of many powerful LLMs further restricts industry applications due to data privacy concerns. Inspired by successes in text generation, LLM ensemble techniques are now increasingly explored for code generation. This article reviews these emerging ensemble approaches to enhance understanding, encourage further research, and promote practical implementation in both text and code generation. We categorize LLM ensembles into seven main methods - weight merging, knowledge fusion, mixture-of-experts, reward ensemble, output ensemble, routing, and cascading - analyzing capabilities of those approaches. Our findings highlight key benefits such as improved diversity representation, enhanced output quality, and greater application flexibility. These insights aid model selection for real-world tasks and crucially, lay groundwork for extending ensemble strategies to multimodal LLMs.
>
---
#### [replaced 057] MetaGen Blended RAG: Unlocking Zero-Shot Precision for Specialized Domain Question-Answering
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18247v3](http://arxiv.org/pdf/2505.18247v3)**

> **作者:** Kunal Sawarkar; Shivam R. Solanki; Abhilasha Mangal
>
> **摘要:** Retrieval-Augmented Generation (RAG) struggles with domain-specific enterprise datasets, often isolated behind firewalls and rich in complex, specialized terminology unseen by LLMs during pre-training. Semantic variability across domains like medicine, networking, or law hampers RAG's context precision, while fine-tuning solutions are costly, slow, and lack generalization as new data emerges. Achieving zero-shot precision with retrievers without fine-tuning still remains a key challenge. We introduce 'MetaGen Blended RAG', a novel enterprise search approach that enhances semantic retrievers through a metadata generation pipeline and hybrid query indexes using dense and sparse vectors. By leveraging key concepts, topics, and acronyms, our method creates metadata-enriched semantic indexes and boosted hybrid queries, delivering robust, scalable performance without fine-tuning. On the biomedical PubMedQA dataset, MetaGen Blended RAG achieves 82% retrieval accuracy and 77% RAG accuracy, surpassing all prior zero-shot RAG benchmarks and even rivaling fine-tuned models on that dataset, while also excelling on datasets like SQuAD and NQ. This approach redefines enterprise search using a new approach to building semantic retrievers with unmatched generalization across specialized domains.
>
---
#### [replaced 058] Large language models provide unsafe answers to patient-posed medical questions
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.18905v2](http://arxiv.org/pdf/2507.18905v2)**

> **作者:** Rachel L. Draelos; Samina Afreen; Barbara Blasko; Tiffany L. Brazile; Natasha Chase; Dimple Patel Desai; Jessica Evert; Heather L. Gardner; Lauren Herrmann; Aswathy Vaikom House; Stephanie Kass; Marianne Kavan; Kirshma Khemani; Amanda Koire; Lauren M. McDonald; Zahraa Rabeeah; Amy Shah
>
> **备注:** 20 pages
>
> **摘要:** Millions of patients are already using large language model (LLM) chatbots for medical advice on a regular basis, raising patient safety concerns. This physician-led red-teaming study compares the safety of four publicly available chatbots--Claude by Anthropic, Gemini by Google, GPT-4o by OpenAI, and Llama3-70B by Meta--on a new dataset, HealthAdvice, using an evaluation framework that enables quantitative and qualitative analysis. In total, 888 chatbot responses are evaluated for 222 patient-posed advice-seeking medical questions on primary care topics spanning internal medicine, women's health, and pediatrics. We find statistically significant differences between chatbots. The rate of problematic responses varies from 21.6 percent (Claude) to 43.2 percent (Llama), with unsafe responses varying from 5 percent (Claude) to 13 percent (GPT-4o, Llama). Qualitative results reveal chatbot responses with the potential to lead to serious patient harm. This study suggests that millions of patients could be receiving unsafe medical advice from publicly available chatbots, and further work is needed to improve the clinical safety of these powerful tools.
>
---
#### [replaced 059] RIVAL: Reinforcement Learning with Iterative and Adversarial Optimization for Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05070v2](http://arxiv.org/pdf/2506.05070v2)**

> **作者:** Tianjiao Li; Mengran Yu; Chenyu Shi; Yanjun Zhao; Xiaojing Liu; Qiang Zhang; Qi Zhang; Xuanjing Huang; Jiayin Wang
>
> **摘要:** Large language models (LLMs) possess strong multilingual capabilities, and combining Reinforcement Learning from Human Feedback (RLHF) with translation tasks has shown great potential. However, we observe that this paradigm performs unexpectedly poorly when applied to colloquial subtitle translation tasks. In this work, we investigate this issue and find that the offline reward model (RM) gradually diverges from the online LLM due to distributional shift, ultimately leading to undesirable training outcomes. To address this, we propose RIVAL, an adversarial training framework that formulates the process as a min-max game between the RM and the LLM. RIVAL iteratively updates the both models, with the RM trained to distinguish strong from weak translations (qualitative preference reward), and the LLM trained to enhance its translation for closing this gap. To stabilize training and improve generalizability, we also incorporate quantitative preference reward (e.g., BLEU) into the RM, enabling reference-free quality modeling aligned with human evaluation. Through extensive experiments, we demonstrate that the proposed adversarial training framework significantly improves upon translation baselines.
>
---
#### [replaced 060] R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.17307v3](http://arxiv.org/pdf/2507.17307v3)**

> **作者:** Zhuokun Chen; Zeren Chen; Jiahao He; Mingkui Tan; Jianfei Cai; Bohan Zhuang
>
> **摘要:** Chain-of-thought (CoT) reasoning enhances the problem-solving capabilities of large language models by encouraging step-by-step intermediate reasoning during inference. While effective, CoT introduces substantial computational overhead due to its reliance on autoregressive decoding over long token sequences. Existing acceleration strategies either reduce sequence length through early stopping or compressive reward designs, or improve decoding speed via speculative decoding with smaller models. However, speculative decoding suffers from limited speedup when the agreement between small and large models is low, and fails to exploit the potential advantages of small models in producing concise intermediate reasoning. In this paper, we present R-Stitch, a token-level, confidence-based hybrid decoding framework that accelerates CoT inference by switching between a small language model (SLM) and a large language model (LLM) along the reasoning trajectory. R-Stitch uses the SLM to generate tokens by default and delegates to the LLM only when the SLM's confidence falls below a threshold. This design avoids full-sequence rollback and selectively invokes the LLM on uncertain steps, preserving both efficiency and answer quality. R-Stitch is model-agnostic, training-free, and compatible with standard decoding pipelines. Experiments on math reasoning benchmarks demonstrate that R-Stitch achieves up to 85\% reduction in inference latency with negligible accuracy drop, highlighting its practical effectiveness in accelerating CoT reasoning.
>
---
#### [replaced 061] Exploring Personalized Health Support through Data-Driven, Theory-Guided LLMs: A Case Study in Sleep Health
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13920v2](http://arxiv.org/pdf/2502.13920v2)**

> **作者:** Xingbo Wang; Janessa Griffith; Daniel A. Adler; Joey Castillo; Tanzeem Choudhury; Fei Wang
>
> **备注:** Accepted to CHI Conference on Human Factors in Computing Systems (CHI 2025). Code is available at https://github.com/xingbow/sleephealthLLM
>
> **摘要:** Despite the prevalence of sleep-tracking devices, many individuals struggle to translate data into actionable improvements in sleep health. Current methods often provide data-driven suggestions but may not be feasible and adaptive to real-life constraints and individual contexts. We present HealthGuru, a novel large language model-powered chatbot to enhance sleep health through data-driven, theory-guided, and adaptive recommendations with conversational behavior change support. HealthGuru's multi-agent framework integrates wearable device data, contextual information, and a contextual multi-armed bandit model to suggest tailored sleep-enhancing activities. The system facilitates natural conversations while incorporating data-driven insights and theoretical behavior change techniques. Our eight-week in-the-wild deployment study with 16 participants compared HealthGuru to a baseline chatbot. Results show improved metrics like sleep duration and activity scores, higher quality responses, and increased user motivation for behavior change with HealthGuru. We also identify challenges and design considerations for personalization and user engagement in health chatbots.
>
---
#### [replaced 062] CLIPPER: Compression enables long-context synthetic data generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14854v2](http://arxiv.org/pdf/2502.14854v2)**

> **作者:** Chau Minh Pham; Yapei Chang; Mohit Iyyer
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** LLM developers are increasingly reliant on synthetic data, but generating high-quality data for complex long-context reasoning tasks remains challenging. We introduce CLIPPER, a compression-based approach for generating synthetic data tailored to narrative claim verification - a task that requires reasoning over a book to verify a given claim. Instead of generating claims directly from the raw text of the book, which results in artifact-riddled claims, CLIPPER first compresses the book into chapter outlines and book summaries and then uses these intermediate representations to generate complex claims and corresponding chain-of-thoughts. Compared to naive approaches, CLIPPER produces claims that are more valid, grounded, and complex. Using CLIPPER, we construct a dataset of 19K synthetic book claims paired with their source texts and chain-of-thought reasoning, and use it to fine-tune three open-weight models. Our best model achieves breakthrough results on narrative claim verification (from 28% to 76% accuracy on our test set) and sets a new state-of-the-art for sub-10B models on the NoCha leaderboard. Further analysis shows that our models generate more detailed and grounded chain-of-thought reasoning while also improving performance on other narrative understanding tasks (e.g., NarrativeQA).
>
---
#### [replaced 063] From Text to Trajectory: Exploring Complex Constraint Representation and Decomposition in Safe Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.08920v3](http://arxiv.org/pdf/2412.08920v3)**

> **作者:** Pusen Dong; Tianchen Zhu; Yue Qiu; Haoyi Zhou; Jianxin Li
>
> **备注:** Accepted by NeurIPS 2024
>
> **摘要:** Safe reinforcement learning (RL) requires the agent to finish a given task while obeying specific constraints. Giving constraints in natural language form has great potential for practical scenarios due to its flexible transfer capability and accessibility. Previous safe RL methods with natural language constraints typically need to design cost functions manually for each constraint, which requires domain expertise and lacks flexibility. In this paper, we harness the dual role of text in this task, using it not only to provide constraint but also as a training signal. We introduce the Trajectory-level Textual Constraints Translator (TTCT) to replace the manually designed cost function. Our empirical results demonstrate that TTCT effectively comprehends textual constraint and trajectory, and the policies trained by TTCT can achieve a lower violation rate than the standard cost function. Extra studies are conducted to demonstrate that the TTCT has zero-shot transfer capability to adapt to constraint-shift environments.
>
---
#### [replaced 064] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04856v3](http://arxiv.org/pdf/2503.04856v3)**

> **作者:** Junwoo Ha; Hyunjun Kim; Sangyoon Yu; Haon Park; Ashkan Yousefpour; Yuna Park; Suhyun Kim
>
> **备注:** Accepted to ACL 2025 (Main Track). Camera-ready version
>
> **摘要:** We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.
>
---
#### [replaced 065] When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02087v2](http://arxiv.org/pdf/2508.02087v2)**

> **作者:** Keyu Wang; Jin Li; Shu Yang; Zhuoran Zhang; Di Wang
>
> **摘要:** Large Language Models (LLMs) often exhibit sycophantic behavior, agreeing with user-stated opinions even when those contradict factual knowledge. While prior work has documented this tendency, the internal mechanisms that enable such behavior remain poorly understood. In this paper, we provide a mechanistic account of how sycophancy arises within LLMs. We first systematically study how user opinions induce sycophancy across different model families. We find that simple opinion statements reliably induce sycophancy, whereas user expertise framing has a negligible impact. Through logit-lens analysis and causal activation patching, we identify a two-stage emergence of sycophancy: (1) a late-layer output preference shift and (2) deeper representational divergence. We also verify that user authority fails to influence behavior because models do not encode it internally. In addition, we examine how grammatical perspective affects sycophantic behavior, finding that first-person prompts (``I believe...'') consistently induce higher sycophancy rates than third-person framings (``They believe...'') by creating stronger representational perturbations in deeper layers. These findings highlight that sycophancy is not a surface-level artifact but emerges from a structural override of learned knowledge in deeper layers, with implications for alignment and truthful AI systems.
>
---
#### [replaced 066] The Multi-Round Diagnostic RAG Framework for Emulating Clinical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07724v2](http://arxiv.org/pdf/2504.07724v2)**

> **作者:** Penglei Sun; Yixiang Chen; Xiang Li; Xiaowen Chu
>
> **摘要:** In recent years, accurately and quickly deploying medical large language models (LLMs) has become a trend. Among these, retrieval-augmented generation (RAG) has garnered attention due to rapid deployment and privacy protection. However, the challenge hinder the practical deployment of RAG for medical diagnosis: the semantic gap between colloquial patient descriptions and the professional terminology within medical knowledge bases. We try to address the challenge from the data perspective and the method perspective. First, to address the semantic gap in existing knowledge bases, we construct DiagnosGraph, a generalist knowledge graph covering both modern medicine and Traditional Chinese Medicine. It contains 876 common diseases with the graph of 7,997 nodes and 37,201 triples. To bridge the gap between colloquial patient narratives and academic medical knowledge, DiagnosGraph also introduces $1,908$ medical record by formalizing the patient chief complaint and proposing a medical diagnosis. Second, we introduce the Multi-Round Diagnostic RAG (MRD-RAG) framework. It utilizes a multi-round dialogue to refine diagnostic possibilities, emulating the clinical reasoning of a physician. Experiments conducted on four medical benchmarks, with evaluations by human physicians, demonstrate that MRD-RAG enhances the diagnostic performance of LLMs, highlighting its potential to make automated diagnosis more accurate and human-aligned.
>
---
