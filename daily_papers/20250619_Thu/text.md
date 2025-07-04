# 自然语言处理 cs.CL

- **最新发布 71 篇**

- **更新 57 篇**

## 最新发布

#### [new 001] Learning-Time Encoding Shapes Unlearning in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型可解释性与知识管理任务，旨在解决LLMs中后验知识删除（unlearning）的问题。研究发现，学习时的知识编码方式影响unlearning效果，提出通过改写描述提升效果。**

- **链接: [http://arxiv.org/pdf/2506.15076v1](http://arxiv.org/pdf/2506.15076v1)**

> **作者:** Ruihan Wu; Konstantin Garov; Kamalika Chaudhuri
>
> **摘要:** As large language models (LLMs) are increasingly deployed in the real world, the ability to ``unlearn'', or remove specific pieces of knowledge post hoc, has become essential for a variety of reasons ranging from privacy regulations to correcting outdated or harmful content. Prior work has proposed unlearning benchmarks and algorithms, and has typically assumed that the training process and the target model are fixed. In this work, we empirically investigate how learning-time choices in knowledge encoding impact the effectiveness of unlearning factual knowledge. Our experiments reveal two key findings: (1) learning with paraphrased descriptions improves unlearning performance and (2) unlearning individual piece of knowledge from a chunk of text is challenging. Our results suggest that learning-time knowledge encoding may play a central role in enabling reliable post-hoc unlearning.
>
---
#### [new 002] Minding the Politeness Gap in Cross-cultural Communication
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于跨文化交际研究，旨在解决语言理解中的礼貌差异问题。通过实验和计算模型，分析英美英语中强调词的解释差异，揭示其源于语义和礼貌规范的相互作用。**

- **链接: [http://arxiv.org/pdf/2506.15623v1](http://arxiv.org/pdf/2506.15623v1)**

> **作者:** Yuka Machino; Matthias Hofer; Max Siegel; Joshua B. Tenenbaum; Robert D. Hawkins
>
> **摘要:** Misunderstandings in cross-cultural communication often arise from subtle differences in interpretation, but it is unclear whether these differences arise from the literal meanings assigned to words or from more general pragmatic factors such as norms around politeness and brevity. In this paper, we report three experiments examining how speakers of British and American English interpret intensifiers like "quite" and "very." To better understand these cross-cultural differences, we developed a computational cognitive model where listeners recursively reason about speakers who balance informativity, politeness, and utterance cost. Our model comparisons suggested that cross-cultural differences in intensifier interpretation stem from a combination of (1) different literal meanings, (2) different weights on utterance cost. These findings challenge accounts based purely on semantic variation or politeness norms, demonstrating that cross-cultural differences in interpretation emerge from an intricate interplay between the two.
>
---
#### [new 003] Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于隐私安全任务，研究大模型推理过程中的数据泄露问题。工作揭示了推理轨迹可能暴露敏感信息，并指出增加推理步骤会加剧泄露风险。**

- **链接: [http://arxiv.org/pdf/2506.15674v1](http://arxiv.org/pdf/2506.15674v1)**

> **作者:** Tommaso Green; Martin Gubri; Haritz Puerto; Sangdoo Yun; Seong Joon Oh
>
> **摘要:** We study privacy leakage in the reasoning traces of large reasoning models used as personal agents. Unlike final outputs, reasoning traces are often assumed to be internal and safe. We challenge this assumption by showing that reasoning traces frequently contain sensitive user data, which can be extracted via prompt injections or accidentally leak into outputs. Through probing and agentic evaluations, we demonstrate that test-time compute approaches, particularly increased reasoning steps, amplify such leakage. While increasing the budget of those test-time compute approaches makes models more cautious in their final answers, it also leads them to reason more verbosely and leak more in their own thinking. This reveals a core tension: reasoning improves utility but enlarges the privacy attack surface. We argue that safety efforts must extend to the model's internal thinking, not just its outputs.
>
---
#### [new 004] Approximating Language Model Training Data from Weights
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练数据逆向任务，旨在从模型权重中近似恢复训练数据。通过梯度方法从公共语料中选择匹配数据，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.15553v1](http://arxiv.org/pdf/2506.15553v1)**

> **作者:** John X. Morris; Junjie Oscar Yin; Woojeong Kim; Vitaly Shmatikov; Alexander M. Rush
>
> **摘要:** Modern language models often have open weights but closed training data. We formalize the problem of data approximation from model weights and propose several baselines and metrics. We develop a gradient-based approach that selects the highest-matching data from a large public text corpus and show its effectiveness at recovering useful data given only weights of the original and finetuned models. Even when none of the true training data is known, our method is able to locate a small subset of public Web documents can be used to train a model to close to the original model performance given models trained for both classification and supervised-finetuning. On the AG News classification task, our method improves performance from 65% (using randomly selected data) to 80%, approaching the expert benchmark of 88%. When applied to a model trained with SFT on MSMARCO web documents, our method reduces perplexity from 3.3 to 2.3, compared to an expert LLAMA model's perplexity of 2.0.
>
---
#### [new 005] DeVisE: Behavioral Testing of Medical Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医疗大模型评估任务，旨在解决现有评价方法无法区分真实医学推理与表面模式的问题。通过构建包含真实和合成数据的ICU出院记录集，测试模型对人口统计和生命体征变化的响应。**

- **链接: [http://arxiv.org/pdf/2506.15339v1](http://arxiv.org/pdf/2506.15339v1)**

> **作者:** Camila Zurdo Tagliabue; Heloisa Oss Boll; Aykut Erdem; Erkut Erdem; Iacer Calixto
>
> **摘要:** Large language models (LLMs) are increasingly used in clinical decision support, yet current evaluation methods often fail to distinguish genuine medical reasoning from superficial patterns. We introduce DeVisE (Demographics and Vital signs Evaluation), a behavioral testing framework for probing fine-grained clinical understanding. We construct a dataset of ICU discharge notes from MIMIC-IV, generating both raw (real-world) and template-based (synthetic) versions with controlled single-variable counterfactuals targeting demographic (age, gender, ethnicity) and vital sign attributes. We evaluate five LLMs spanning general-purpose and medically fine-tuned variants, under both zero-shot and fine-tuned settings. We assess model behavior via (1) input-level sensitivity - how counterfactuals alter the likelihood of a note; and (2) downstream reasoning - how they affect predicted hospital length-of-stay. Our results show that zero-shot models exhibit more coherent counterfactual reasoning patterns, while fine-tuned models tend to be more stable yet less responsive to clinically meaningful changes. Notably, demographic factors subtly but consistently influence outputs, emphasizing the importance of fairness-aware evaluation. This work highlights the utility of behavioral testing in exposing the reasoning strategies of clinical LLMs and informing the design of safer, more transparent medical AI systems.
>
---
#### [new 006] PhantomHunter: Detecting Unseen Privately-Tuned LLM-Generated Text via Family-Aware Learning
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于文本检测任务，旨在解决私有微调大模型生成文本的检测问题。提出PhantomHunter，通过家族感知学习捕捉共性特征，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.15683v1](http://arxiv.org/pdf/2506.15683v1)**

> **作者:** Yuhui Shi; Yehan Yang; Qiang Sheng; Hao Mi; Beizhe Hu; Chaoxi Xu; Juan Cao
>
> **备注:** 17 pages, 3 figures, 6 tables
>
> **摘要:** With the popularity of large language models (LLMs), undesirable societal problems like misinformation production and academic misconduct have been more severe, making LLM-generated text detection now of unprecedented importance. Although existing methods have made remarkable progress, a new challenge posed by text from privately tuned LLMs remains underexplored. Users could easily possess private LLMs by fine-tuning an open-source one with private corpora, resulting in a significant performance drop of existing detectors in practice. To address this issue, we propose PhantomHunter, an LLM-generated text detector specialized for detecting text from unseen, privately-tuned LLMs. Its family-aware learning framework captures family-level traits shared across the base models and their derivatives, instead of memorizing individual characteristics. Experiments on data from LLaMA, Gemma, and Mistral families show its superiority over 7 baselines and 3 industrial services, with F1 scores of over 96%.
>
---
#### [new 007] RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型推理评估任务，旨在解决LLM是否真具备推理能力的问题。通过构建RE-IMAGINE框架生成不同层次的推理问题，验证模型是否依赖统计记忆。**

- **链接: [http://arxiv.org/pdf/2506.15455v1](http://arxiv.org/pdf/2506.15455v1)**

> **作者:** Xinnuo Xu; Rachel Lawrence; Kshitij Dubey; Atharva Pandey; Risa Ueno; Fabian Falck; Aditya V. Nori; Rahul Sharma; Amit Sharma; Javier Gonzalez
>
> **备注:** ICML 2025
>
> **摘要:** Recent Large Language Models (LLMs) have reported high accuracy on reasoning benchmarks. However, it is still unclear whether the observed results arise from true reasoning or from statistical recall of the training set. Inspired by the ladder of causation (Pearl, 2009) and its three levels (associations, interventions and counterfactuals), this paper introduces RE-IMAGINE, a framework to characterize a hierarchy of reasoning ability in LLMs, alongside an automated pipeline to generate problem variations at different levels of the hierarchy. By altering problems in an intermediate symbolic representation, RE-IMAGINE generates arbitrarily many problems that are not solvable using memorization alone. Moreover, the framework is general and can work across reasoning domains, including math, code, and logic. We demonstrate our framework on four widely-used benchmarks to evaluate several families of LLMs, and observe reductions in performance when the models are queried with problem variations. These assessments indicate a degree of reliance on statistical recall for past performance, and open the door to further research targeting skills across the reasoning hierarchy.
>
---
#### [new 008] AgentGroupChat-V2: Divide-and-Conquer Is What LLM-Based Multi-Agent System Need
- **分类: cs.CL**

- **简介: 该论文提出AgentGroupChat-V2框架，解决多智能体系统在复杂任务中的架构设计与性能问题，通过分治并行、自适应协作和优化组织策略提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.15451v1](http://arxiv.org/pdf/2506.15451v1)**

> **作者:** Zhouhong Gu; Xiaoxuan Zhu; Yin Cai; Hao Shen; Xingzhou Chen; Qingyi Wang; Jialin Li; Xiaoran Shi; Haoran Guo; Wenxuan Huang; Hongwei Feng; Yanghua Xiao; Zheyu Ye; Yao Hu; Shaosheng Cao
>
> **摘要:** Large language model based multi-agent systems have demonstrated significant potential in social simulation and complex task resolution domains. However, current frameworks face critical challenges in system architecture design, cross-domain generalizability, and performance guarantees, particularly as task complexity and number of agents increases. We introduces AgentGroupChat-V2, a novel framework addressing these challenges through three core innovations: (1) a divide-and-conquer fully parallel architecture that decomposes user queries into hierarchical task forest structures enabling dependency management and distributed concurrent processing. (2) an adaptive collaboration engine that dynamically selects heterogeneous LLM combinations and interaction modes based on task characteristics. (3) agent organization optimization strategies combining divide-and-conquer approaches for efficient problem decomposition. Extensive experiments demonstrate AgentGroupChat-V2's superior performance across diverse domains, achieving 91.50% accuracy on GSM8K (exceeding the best baseline by 5.6 percentage points), 30.4% accuracy on competition-level AIME (nearly doubling other methods), and 79.20% pass@1 on HumanEval. Performance advantages become increasingly pronounced with higher task difficulty, particularly on Level 5 MATH problems where improvements exceed 11 percentage points compared to state-of-the-art baselines. These results confirm that AgentGroupChat-V2 provides a comprehensive solution for building efficient, general-purpose LLM multi-agent systems with significant advantages in complex reasoning scenarios. Code is available at https://github.com/MikeGu721/AgentGroupChat-V2.
>
---
#### [new 009] Research on Graph-Retrieval Augmented Generation Based on Historical Text Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于历史文本分析任务，旨在解决大模型在历史知识上的领域差距。通过构建知识图谱与生成模型结合的Graph RAG框架，提升关系抽取效果并减少幻觉。**

- **链接: [http://arxiv.org/pdf/2506.15241v1](http://arxiv.org/pdf/2506.15241v1)**

> **作者:** Yang Fan; Zhang Qi; Xing Wenqian; Liu Chang; Liu Liu
>
> **摘要:** This article addresses domain knowledge gaps in general large language models for historical text analysis in the context of computational humanities and AIGC technology. We propose the Graph RAG framework, combining chain-of-thought prompting, self-instruction generation, and process supervision to create a The First Four Histories character relationship dataset with minimal manual annotation. This dataset supports automated historical knowledge extraction, reducing labor costs. In the graph-augmented generation phase, we introduce a collaborative mechanism between knowledge graphs and retrieval-augmented generation, improving the alignment of general models with historical knowledge. Experiments show that the domain-specific model Xunzi-Qwen1.5-14B, with Simplified Chinese input and chain-of-thought prompting, achieves optimal performance in relation extraction (F1 = 0.68). The DeepSeek model integrated with GraphRAG improves F1 by 11% (0.08-0.19) on the open-domain C-CLUE relation extraction dataset, surpassing the F1 value of Xunzi-Qwen1.5-14B (0.12), effectively alleviating hallucinations phenomenon, and improving interpretability. This framework offers a low-resource solution for classical text knowledge extraction, advancing historical knowledge services and humanities research.
>
---
#### [new 010] ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理能力。通过引入原型表示，解决跨领域泛化不足的问题，提出ProtoReasoning框架增强模型推理效果。**

- **链接: [http://arxiv.org/pdf/2506.15211v1](http://arxiv.org/pdf/2506.15211v1)**

> **作者:** Feng He; Zijun Chen; Xinnian Liang; Tingting Ma; Yunqi Qiu; Shuangzhi Wu; Junchi Yan
>
> **摘要:** Recent advances in Large Reasoning Models (LRMs) trained with Long Chain-of-Thought (Long CoT) reasoning have demonstrated remarkable cross-domain generalization capabilities. However, the underlying mechanisms supporting such transfer remain poorly understood. We hypothesize that cross-domain generalization arises from shared abstract reasoning prototypes -- fundamental reasoning patterns that capture the essence of problems across domains. These prototypes minimize the nuances of the representation, revealing that seemingly diverse tasks are grounded in shared reasoning structures.Based on this hypothesis, we propose ProtoReasoning, a framework that enhances the reasoning ability of LLMs by leveraging scalable and verifiable prototypical representations (Prolog for logical reasoning, PDDL for planning).ProtoReasoning features: (1) an automated prototype construction pipeline that transforms problems into corresponding prototype representations; (2) a comprehensive verification system providing reliable feedback through Prolog/PDDL interpreters; (3) the scalability to synthesize problems arbitrarily within prototype space while ensuring correctness. Extensive experiments show that ProtoReasoning achieves 4.7% improvement over baseline models on logical reasoning (Enigmata-Eval), 6.3% improvement on planning tasks, 4.0% improvement on general reasoning (MMLU) and 1.0% on mathematics (AIME24). Significantly, our ablation studies confirm that learning in prototype space also demonstrates enhanced generalization to structurally similar problems compared to training solely on natural language representations, validating our hypothesis that reasoning prototypes serve as the foundation for generalizable reasoning in large language models.
>
---
#### [new 011] Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework for Evaluating Gender Diversity in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在解决大语言模型在性别多样性上的包容性问题，提出GIFI框架进行多层级评估。**

- **链接: [http://arxiv.org/pdf/2506.15568v1](http://arxiv.org/pdf/2506.15568v1)**

> **作者:** Zhengyang Shan; Emily Ruth Diana; Jiawei Zhou
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** We present a comprehensive evaluation of gender fairness in large language models (LLMs), focusing on their ability to handle both binary and non-binary genders. While previous studies primarily focus on binary gender distinctions, we introduce the Gender Inclusivity Fairness Index (GIFI), a novel and comprehensive metric that quantifies the diverse gender inclusivity of LLMs. GIFI consists of a wide range of evaluations at different levels, from simply probing the model with respect to provided gender pronouns to testing various aspects of model generation and cognitive behaviors under different gender assumptions, revealing biases associated with varying gender identifiers. We conduct extensive evaluations with GIFI on 22 prominent open-source and proprietary LLMs of varying sizes and capabilities, discovering significant variations in LLMs' gender inclusivity. Our study highlights the importance of improving LLMs' inclusivity, providing a critical benchmark for future advancements in gender fairness in generative models.
>
---
#### [new 012] Oldies but Goldies: The Potential of Character N-grams for Romanian Texts
- **分类: cs.CL**

- **简介: 该论文属于作者身份识别任务，旨在解决罗马尼亚文本的作者归属问题。通过使用字符n-gram特征和多种机器学习方法进行分类，验证了简单特征的有效性。**

- **链接: [http://arxiv.org/pdf/2506.15650v1](http://arxiv.org/pdf/2506.15650v1)**

> **作者:** Dana Lupsa; Sanda-Maria Avram
>
> **摘要:** This study addresses the problem of authorship attribution for Romanian texts using the ROST corpus, a standard benchmark in the field. We systematically evaluate six machine learning techniques: Support Vector Machine (SVM), Logistic Regression (LR), k-Nearest Neighbors (k-NN), Decision Trees (DT), Random Forests (RF), and Artificial Neural Networks (ANN), employing character n-gram features for classification. Among these, the ANN model achieved the highest performance, including perfect classification in four out of fifteen runs when using 5-gram features. These results demonstrate that lightweight, interpretable character n-gram approaches can deliver state-of-the-art accuracy for Romanian authorship attribution, rivaling more complex methods. Our findings highlight the potential of simple stylometric features in resource, constrained or under-studied language settings.
>
---
#### [new 013] CrEst: Credibility Estimation for Contexts in LLMs via Weak Supervision
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于信息可信度评估任务，解决LLM推理中上下文文档可信度不一的问题。提出CrEst框架，通过弱监督方法自动评估文档可信度，并改进模型推理策略。**

- **链接: [http://arxiv.org/pdf/2506.14912v1](http://arxiv.org/pdf/2506.14912v1)**

> **作者:** Dyah Adila; Shuai Zhang; Boran Han; Bonan Min; Yuyang Wang
>
> **摘要:** The integration of contextual information has significantly enhanced the performance of large language models (LLMs) on knowledge-intensive tasks. However, existing methods often overlook a critical challenge: the credibility of context documents can vary widely, potentially leading to the propagation of unreliable information. In this paper, we introduce CrEst, a novel weakly supervised framework for assessing the credibility of context documents during LLM inference--without requiring manual annotations. Our approach is grounded in the insight that credible documents tend to exhibit higher semantic coherence with other credible documents, enabling automated credibility estimation through inter-document agreement. To incorporate credibility into LLM inference, we propose two integration strategies: a black-box approach for models without access to internal weights or activations, and a white-box method that directly modifies attention mechanisms. Extensive experiments across three model architectures and five datasets demonstrate that CrEst consistently outperforms strong baselines, achieving up to a 26.86% improvement in accuracy and a 3.49% increase in F1 score. Further analysis shows that CrEst maintains robust performance even under high-noise conditions.
>
---
#### [new 014] Thunder-DeID: Accurate and Efficient De-identification Framework for Korean Court Judgments
- **分类: cs.CL**

- **简介: 该论文属于文本去标识化任务，旨在解决韩国法院判决书中的个人隐私保护问题。通过构建数据集、分类PII并开发深度学习模型实现高效准确的去标识化。**

- **链接: [http://arxiv.org/pdf/2506.15266v1](http://arxiv.org/pdf/2506.15266v1)**

> **作者:** Sungen Hahm; Heejin Kim; Gyuseong Lee; Hyunji Park; Jaejin Lee
>
> **摘要:** To ensure a balance between open access to justice and personal data protection, the South Korean judiciary mandates the de-identification of court judgments before they can be publicly disclosed. However, the current de-identification process is inadequate for handling court judgments at scale while adhering to strict legal requirements. Additionally, the legal definitions and categorizations of personal identifiers are vague and not well-suited for technical solutions. To tackle these challenges, we propose a de-identification framework called Thunder-DeID, which aligns with relevant laws and practices. Specifically, we (i) construct and release the first Korean legal dataset containing annotated judgments along with corresponding lists of entity mentions, (ii) introduce a systematic categorization of Personally Identifiable Information (PII), and (iii) develop an end-to-end deep neural network (DNN)-based de-identification pipeline. Our experimental results demonstrate that our model achieves state-of-the-art performance in the de-identification of court judgments.
>
---
#### [new 015] CC-LEARN: Cohort-based Consistency Learning
- **分类: cs.CL**

- **简介: 该论文提出CC-Learn，属于增强语言模型推理一致性的任务，通过强化学习提升模型在复杂推理任务中的稳定性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.15662v1](http://arxiv.org/pdf/2506.15662v1)**

> **作者:** Xiao Ye; Shaswat Shrivastava; Zhaonan Li; Jacob Dineen; Shijie Lu; Avneet Ahuja; Ming Shen; Zhikun Xu; Ben Zhou
>
> **摘要:** Large language models excel at many tasks but still struggle with consistent, robust reasoning. We introduce Cohort-based Consistency Learning (CC-Learn), a reinforcement learning framework that improves the reliability of LLM reasoning by training on cohorts of similar questions derived from shared programmatic abstractions. To enforce cohort-level consistency, we define a composite objective combining cohort accuracy, a retrieval bonus for effective problem decomposition, and a rejection penalty for trivial or invalid lookups that reinforcement learning can directly optimize, unlike supervised fine-tuning. Optimizing this reward guides the model to adopt uniform reasoning patterns across all cohort members. Experiments on challenging reasoning benchmarks (including ARC-Challenge and StrategyQA) show that CC-Learn boosts both accuracy and reasoning stability over pretrained and SFT baselines. These results demonstrate that cohort-level RL effectively enhances reasoning consistency in LLMs.
>
---
#### [new 016] Improving Dialogue Discourse Parsing through Discourse-aware Utterance Clarification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话话语解析任务，旨在解决对话中因省略和习语导致的歧义问题。提出DCM模块和CPO方法提升解析准确性。**

- **链接: [http://arxiv.org/pdf/2506.15081v1](http://arxiv.org/pdf/2506.15081v1)**

> **作者:** Yaxin Fan; Peifeng Li; Qiaoming Zhu
>
> **备注:** Accepted by ACL2025(main conference)
>
> **摘要:** Dialogue discourse parsing aims to identify and analyze discourse relations between the utterances within dialogues. However, linguistic features in dialogues, such as omission and idiom, frequently introduce ambiguities that obscure the intended discourse relations, posing significant challenges for parsers. To address this issue, we propose a Discourse-aware Clarification Module (DCM) to enhance the performance of the dialogue discourse parser. DCM employs two distinct reasoning processes: clarification type reasoning and discourse goal reasoning. The former analyzes linguistic features, while the latter distinguishes the intended relation from the ambiguous one. Furthermore, we introduce Contribution-aware Preference Optimization (CPO) to mitigate the risk of erroneous clarifications, thereby reducing cascading errors. CPO enables the parser to assess the contributions of the clarifications from DCM and provide feedback to optimize the DCM, enhancing its adaptability and alignment with the parser's requirements. Extensive experiments on the STAC and Molweni datasets demonstrate that our approach effectively resolves ambiguities and significantly outperforms the state-of-the-art (SOTA) baselines.
>
---
#### [new 017] SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SPARE框架，用于自动过程标注与奖励建模，解决多步骤推理中高效高质量标注难题。**

- **链接: [http://arxiv.org/pdf/2506.15498v1](http://arxiv.org/pdf/2506.15498v1)**

> **作者:** Md Imbesat Hassan Rizvi; Xiaodan Zhu; Iryna Gurevych
>
> **备注:** 8 pages main content, 4 figures, 4 tables
>
> **摘要:** Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce Single-Pass Annotation with Reference-Guided Evaluation (SPARE), a novel structured framework that enables single-pass, per-step annotation by aligning each solution step to one or multiple steps in a reference solution, accompanied by explicit reasoning for evaluation. We show that reference-guided step-level evaluation effectively facilitates process supervision on four datasets spanning three domains: mathematical reasoning, multi-hop compositional question answering, and spatial reasoning. We demonstrate that SPARE, when compared to baselines, improves reasoning performance when used for: (1) fine-tuning models in an offline RL setup for inference-time greedy-decoding, and (2) training reward models for ranking/aggregating multiple LLM-generated outputs. Additionally, SPARE achieves competitive performance on challenging mathematical datasets while offering 2.6 times greater efficiency, requiring only 38% of the runtime, compared to tree search-based automatic annotation. The codebase, along with a trained SPARE-PRM model, is publicly released to facilitate further research and reproducibility.
>
---
#### [new 018] Lessons from Training Grounded LLMs with Verifiable Rewards
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的回答准确性和可信度。通过强化学习和内部推理优化模型的引用和拒绝能力，显著改善了生成结果的质量。**

- **链接: [http://arxiv.org/pdf/2506.15522v1](http://arxiv.org/pdf/2506.15522v1)**

> **作者:** Shang Hong Sim; Tej Deep Pala; Vernon Toh; Hai Leong Chieu; Amir Zadeh; Chuan Li; Navonil Majumder; Soujanya Poria
>
> **摘要:** Generating grounded and trustworthy responses remains a key challenge for large language models (LLMs). While retrieval-augmented generation (RAG) with citation-based grounding holds promise, instruction-tuned models frequently fail even in straightforward scenarios: missing explicitly stated answers, citing incorrectly, or refusing when evidence is available. In this work, we explore how reinforcement learning (RL) and internal reasoning can enhance grounding in LLMs. We use the GRPO (Group Relative Policy Optimization) method to train models using verifiable outcome-based rewards targeting answer correctness, citation sufficiency, and refusal quality, without requiring gold reasoning traces or expensive annotations. Through comprehensive experiments across ASQA, QAMPARI, ELI5, and ExpertQA we show that reasoning-augmented models significantly outperform instruction-only variants, especially in handling unanswerable queries and generating well-cited responses. A two-stage training setup, first optimizing answer and citation behavior and then refusal, further improves grounding by stabilizing the learning signal. Additionally, we revisit instruction tuning via GPT-4 distillation and find that combining it with GRPO enhances performance on long-form, generative QA tasks. Overall, our findings highlight the value of reasoning, stage-wise optimization, and outcome-driven RL for building more verifiable and reliable LLMs.
>
---
#### [new 019] Semantically-Aware Rewards for Open-Ended R1 Training in Free-Form Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于开放生成任务，解决长文本评价难题。提出PrefBERT模型，通过语义奖励提升GRPO训练效果，优于传统指标。**

- **链接: [http://arxiv.org/pdf/2506.15068v1](http://arxiv.org/pdf/2506.15068v1)**

> **作者:** Zongxia Li; Yapei Chang; Yuhang Zhou; Xiyang Wu; Zichao Liang; Yoo Yeon Sung; Jordan Lee Boyd-Graber
>
> **摘要:** Evaluating open-ended long-form generation is challenging because it is hard to define what clearly separates good from bad outputs. Existing methods often miss key aspects like coherence, style, or relevance, or are biased by pretraining data, making open-ended long-form evaluation an underexplored problem. To address this gap, we propose PrefBERT, a scoring model for evaluating open-ended long-form generation in GRPO and guiding its training with distinct rewards for good and bad outputs. Trained on two response evaluation datasets with diverse long-form styles and Likert-rated quality, PrefBERT effectively supports GRPO by offering better semantic reward feedback than traditional metrics ROUGE-L and BERTScore do. Through comprehensive evaluations, including LLM-as-a-judge, human ratings, and qualitative analysis, we show that PrefBERT, trained on multi-sentence and paragraph-length responses, remains reliable across varied long passages and aligns well with the verifiable rewards GRPO needs. Human evaluations confirm that using PrefBERT as the reward signal to train policy models yields responses better aligned with human preferences than those trained with traditional metrics. Our code is available at https://github.com/zli12321/long_form_rl.
>
---
#### [new 020] MinosEval: Distinguishing Factoid and Non-Factoid for Tailored Open-Ended QA Evaluation with LLMs
- **分类: cs.CL**

- **简介: 该论文属于开放问答评估任务，旨在解决LLMs回答质量评价不准确的问题。提出MinosEval方法，区分事实性与非事实性问题，采用不同策略提升评估效果。**

- **链接: [http://arxiv.org/pdf/2506.15215v1](http://arxiv.org/pdf/2506.15215v1)**

> **作者:** Yongqi Fan; Yating Wang; Guandong Wang; Jie Zhai; Jingping Liu; Qi Ye; Tong Ruan
>
> **摘要:** Open-ended question answering (QA) is a key task for evaluating the capabilities of large language models (LLMs). Compared to closed-ended QA, it demands longer answer statements, more nuanced reasoning processes, and diverse expressions, making refined and interpretable automatic evaluation both crucial and challenging. Traditional metrics like ROUGE and BERTScore struggle to capture semantic similarities due to different patterns between model responses and reference answers. Current LLM-based evaluation approaches, such as pairwise or listwise comparisons of candidate answers, lack intuitive interpretability. While pointwise scoring of each response provides some descriptions, it fails to adapt across different question contents. Most notably, existing methods overlook the distinction between factoid and non-factoid questions. To address these challenges, we propose \textbf{MinosEval}, a novel evaluation method that first distinguishes open-ended questions and then ranks candidate answers using different evaluation strategies. For factoid questions, it applies an adaptive key-point scoring strategy, while for non-factoid questions, it uses an instance-aware listwise ranking strategy. Experiments on multiple open-ended QA datasets, including self-built ones with more candidate responses to complement community resources, show that MinosEval better aligns with human annotations and offers more interpretable results.
>
---
#### [new 021] PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，解决实时语音交互中大模型延迟问题。通过输入时间推测生成，减少首句生成时间，提升响应速度。**

- **链接: [http://arxiv.org/pdf/2506.15556v1](http://arxiv.org/pdf/2506.15556v1)**

> **作者:** Shufan Li; Aditya Grover
>
> **备注:** 16 pages,4 figures
>
> **摘要:** Large Language Models (LLMs) are widely used in real-time voice chat applications, typically in combination with text-to-speech (TTS) systems to generate audio responses. However, their large size often leads to noticeable latency between the end of user input and the start of audio output, resulting in suboptimal user experiences. This latency is particularly evident when LLMs are deployed as single-user voice assistants on consumer-grade hardware with limited computing capacity. We discovered that this latency is primarily dominated by the time it takes for the LLMs to generate the first sentence, which is required as input by the TTS systems that synthesize audio responses on a sentence-by-sentence basis. To address this bottleneck, we propose Predictive Generation (PredGen), a novel framework that mitigates-or even eliminates-this delay through speculative decoding at input time. PredGen generates candidate responses while the user is still speaking, enabling the system to begin TTS processing with minimal delay. Simulated experiments on the Lmsys and MT-Bench datasets show that the proposed method can effectively reduce the latency by around 2x across a wide range of use cases, while incurring only minimal additional computation cost at input time-computation that would otherwise go unused.
>
---
#### [new 022] ConLID: Supervised Contrastive Learning for Low-Resource Language Identification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言识别任务，针对低资源语言在领域外数据上表现差的问题，提出一种监督对比学习方法，提升其识别性能。**

- **链接: [http://arxiv.org/pdf/2506.15304v1](http://arxiv.org/pdf/2506.15304v1)**

> **作者:** Negar Foroutan; Jakhongir Saydaliev; Ye Eun Kim; Antoine Bosselut
>
> **备注:** Submitted to EMNLP
>
> **摘要:** Language identification (LID) is a critical step in curating multilingual LLM pretraining corpora from web crawls. While many studies on LID model training focus on collecting diverse training data to improve performance, low-resource languages -- often limited to single-domain data, such as the Bible -- continue to perform poorly. To resolve these class imbalance and bias issues, we propose a novel supervised contrastive learning (SCL) approach to learn domain-invariant representations for low-resource languages. Through an extensive analysis, we show that our approach improves LID performance on out-of-domain data for low-resource languages by 3.2%, demonstrating its effectiveness in enhancing LID models.
>
---
#### [new 023] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文档理解任务，旨在解决表格和图表的多模态问答问题。提出WikiMixQA基准，评估模型在长文档中的跨模态推理能力。**

- **链接: [http://arxiv.org/pdf/2506.15594v1](http://arxiv.org/pdf/2506.15594v1)**

> **作者:** Negar Foroutan; Angelika Romanou; Matin Ansaripour; Julian Martin Eisenschlos; Karl Aberer; Rémi Lebret
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** Documents are fundamental to preserving and disseminating information, often incorporating complex layouts, tables, and charts that pose significant challenges for automatic document understanding (DU). While vision-language large models (VLLMs) have demonstrated improvements across various tasks, their effectiveness in processing long-context vision inputs remains unclear. This paper introduces WikiMixQA, a benchmark comprising 1,000 multiple-choice questions (MCQs) designed to evaluate cross-modal reasoning over tables and charts extracted from 4,000 Wikipedia pages spanning seven distinct topics. Unlike existing benchmarks, WikiMixQA emphasizes complex reasoning by requiring models to synthesize information from multiple modalities. We evaluate 12 state-of-the-art vision-language models, revealing that while proprietary models achieve ~70% accuracy when provided with direct context, their performance deteriorates significantly when retrieval from long documents is required. Among these, GPT-4-o is the only model exceeding 50% accuracy in this setting, whereas open-source models perform considerably worse, with a maximum accuracy of 27%. These findings underscore the challenges of long-context, multi-modal reasoning and establish WikiMixQA as a crucial benchmark for advancing document understanding research.
>
---
#### [new 024] The Compositional Architecture of Regret in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型中的后悔机制，解决数据、度量和神经元分析难题，提出新方法提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2506.15617v1](http://arxiv.org/pdf/2506.15617v1)**

> **作者:** Xiangxiang Cui; Shu Yang; Tianjin Huang; Wanyu Lin; Lijie Hu; Di Wang
>
> **备注:** 23 pages
>
> **摘要:** Regret in Large Language Models refers to their explicit regret expression when presented with evidence contradicting their previously generated misinformation. Studying the regret mechanism is crucial for enhancing model reliability and helps in revealing how cognition is coded in neural networks. To understand this mechanism, we need to first identify regret expressions in model outputs, then analyze their internal representation. This analysis requires examining the model's hidden states, where information processing occurs at the neuron level. However, this faces three key challenges: (1) the absence of specialized datasets capturing regret expressions, (2) the lack of metrics to find the optimal regret representation layer, and (3) the lack of metrics for identifying and analyzing regret neurons. Addressing these limitations, we propose: (1) a workflow for constructing a comprehensive regret dataset through strategically designed prompting scenarios, (2) the Supervised Compression-Decoupling Index (S-CDI) metric to identify optimal regret representation layers, and (3) the Regret Dominance Score (RDS) metric to identify regret neurons and the Group Impact Coefficient (GIC) to analyze activation patterns. Our experimental results successfully identified the optimal regret representation layer using the S-CDI metric, which significantly enhanced performance in probe classification experiments. Additionally, we discovered an M-shaped decoupling pattern across model layers, revealing how information processing alternates between coupling and decoupling phases. Through the RDS metric, we categorized neurons into three distinct functional groups: regret neurons, non-regret neurons, and dual neurons.
>
---
#### [new 025] DiscoSG: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement
- **分类: cs.CL**

- **简介: 该论文提出Discourse-level文本场景图解析任务（DiscoSG），解决多句描述中跨句指代导致的图碎片化问题，通过迭代图优化方法提升解析效果。**

- **链接: [http://arxiv.org/pdf/2506.15583v1](http://arxiv.org/pdf/2506.15583v1)**

> **作者:** Shaoqing Lin; Chong Teng; Fei Li; Donghong Ji; Lizhen Qu; Zhuang Li
>
> **摘要:** Vision-Language Models (VLMs) now generate discourse-level, multi-sentence visual descriptions, challenging text scene graph parsers originally designed for single-sentence caption-to-graph mapping. Current approaches typically merge sentence-level parsing outputs for discourse input, often missing phenomena like cross-sentence coreference, resulting in fragmented graphs and degraded downstream VLM task performance. To address this, we introduce a new task, Discourse-level text Scene Graph parsing (DiscoSG), supported by our dataset DiscoSG-DS, which comprises 400 expert-annotated and 8,430 synthesised multi-sentence caption-graph pairs for images. Each caption averages 9 sentences, and each graph contains at least 3 times more triples than those in existing datasets. While fine-tuning large PLMs (i.e., GPT-4) on DiscoSG-DS improves SPICE by approximately 48% over the best sentence-merging baseline, high inference cost and restrictive licensing hinder its open-source use, and smaller fine-tuned PLMs struggle with complex graphs. We propose DiscoSG-Refiner, which drafts a base graph using one small PLM, then employs a second PLM to iteratively propose graph edits, reducing full-graph generation overhead. Using two Flan-T5-Base models, DiscoSG-Refiner still improves SPICE by approximately 30% over the best baseline while achieving 86 times faster inference than GPT-4. It also consistently improves downstream VLM tasks like discourse-level caption evaluation and hallucination detection. Code and data are available at: https://github.com/ShaoqLin/DiscoSG
>
---
#### [new 026] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus for Summarization and Headline Generation
- **分类: cs.CL**

- **简介: 该论文属于多模态摘要和标题生成任务，旨在解决印度语言资源不足的问题。通过构建COSMMIC数据集，融合文本、图像和用户评论进行摘要生成研究。**

- **链接: [http://arxiv.org/pdf/2506.15372v1](http://arxiv.org/pdf/2506.15372v1)**

> **作者:** Raghvendra Kumar; S. A. Mohammed Salman; Aryan Sahu; Tridib Nandi; Pragathi Y. P.; Sriparna Saha; Jose G. Moreno
>
> **备注:** ACL 2025 MAINs
>
> **摘要:** Despite progress in comment-aware multimodal and multilingual summarization for English and Chinese, research in Indian languages remains limited. This study addresses this gap by introducing COSMMIC, a pioneering comment-sensitive multimodal, multilingual dataset featuring nine major Indian languages. COSMMIC comprises 4,959 article-image pairs and 24,484 reader comments, with ground-truth summaries available in all included languages. Our approach enhances summaries by integrating reader insights and feedback. We explore summarization and headline generation across four configurations: (1) using article text alone, (2) incorporating user comments, (3) utilizing images, and (4) combining text, comments, and images. To assess the dataset's effectiveness, we employ state-of-the-art language models such as LLama3 and GPT-4. We conduct a comprehensive study to evaluate different component combinations, including identifying supportive comments, filtering out noise using a dedicated comment classifier using IndicBERT, and extracting valuable insights from images with a multilingual CLIP-based classifier. This helps determine the most effective configurations for natural language generation (NLG) tasks. Unlike many existing datasets that are either text-only or lack user comments in multimodal settings, COSMMIC uniquely integrates text, images, and user feedback. This holistic approach bridges gaps in Indian language resources, advancing NLP research and fostering inclusivity.
>
---
#### [new 027] TopClustRAG at SIGIR 2025 LiveRAG Challenge
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在提升大规模RAG系统的答案质量。通过混合检索与聚类方法优化上下文过滤和提示聚合，提高答案的相关性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.15246v1](http://arxiv.org/pdf/2506.15246v1)**

> **作者:** Juli Bakagianni; John Pavlopoulos; Aristidis Likas
>
> **摘要:** We present TopClustRAG, a retrieval-augmented generation (RAG) system developed for the LiveRAG Challenge, which evaluates end-to-end question answering over large-scale web corpora. Our system employs a hybrid retrieval strategy combining sparse and dense indices, followed by K-Means clustering to group semantically similar passages. Representative passages from each cluster are used to construct cluster-specific prompts for a large language model (LLM), generating intermediate answers that are filtered, reranked, and finally synthesized into a single, comprehensive response. This multi-stage pipeline enhances answer diversity, relevance, and faithfulness to retrieved evidence. Evaluated on the FineWeb Sample-10BT dataset, TopClustRAG ranked 2nd in faithfulness and 7th in correctness on the official leaderboard, demonstrating the effectiveness of clustering-based context filtering and prompt aggregation in large-scale RAG systems.
>
---
#### [new 028] Lost in Variation? Evaluating NLI Performance in Basque and Spanish Geographical Variants
- **分类: cs.CL**

- **简介: 该论文研究NLI任务中处理巴斯克语和西班牙语变体的能力，评估语言技术在处理语言变异时的性能下降问题，并构建了相关数据集进行实验分析。**

- **链接: [http://arxiv.org/pdf/2506.15239v1](http://arxiv.org/pdf/2506.15239v1)**

> **作者:** Jaione Bengoetxea; Itziar Gonzalez-Dios; Rodrigo Agerri
>
> **摘要:** In this paper, we evaluate the capacity of current language technologies to understand Basque and Spanish language varieties. We use Natural Language Inference (NLI) as a pivot task and introduce a novel, manually-curated parallel dataset in Basque and Spanish, along with their respective variants. Our empirical analysis of crosslingual and in-context learning experiments using encoder-only and decoder-based Large Language Models (LLMs) shows a performance drop when handling linguistic variation, especially in Basque. Error analysis suggests that this decline is not due to lexical overlap, but rather to the linguistic variation itself. Further ablation experiments indicate that encoder-only models particularly struggle with Western Basque, which aligns with linguistic theory that identifies peripheral dialects (e.g., Western) as more distant from the standard. All data and code are publicly available.
>
---
#### [new 029] From Model to Classroom: Evaluating Generated MCQs for Portuguese with Narrative and Difficulty Concerns
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的MCQ生成任务，旨在解决生成高质量、符合难度和叙事要求的葡萄牙语选择题问题。研究评估了生成题目的质量和可靠性。**

- **链接: [http://arxiv.org/pdf/2506.15598v1](http://arxiv.org/pdf/2506.15598v1)**

> **作者:** Bernardo Leite; Henrique Lopes Cardoso; Pedro Pinto; Abel Ferreira; Luís Abreu; Isabel Rangel; Sandra Monteiro
>
> **备注:** This is a preprint version of the manuscript currently under review at an international journal
>
> **摘要:** While MCQs are valuable for learning and evaluation, manually creating them with varying difficulty levels and targeted reading skills remains a time-consuming and costly task. Recent advances in generative AI provide an opportunity to automate MCQ generation efficiently. However, assessing the actual quality and reliability of generated MCQs has received limited attention -- particularly regarding cases where generation fails. This aspect becomes particularly important when the generated MCQs are meant to be applied in real-world settings. Additionally, most MCQ generation studies focus on English, leaving other languages underexplored. This paper investigates the capabilities of current generative models in producing MCQs for reading comprehension in Portuguese, a morphologically rich language. Our study focuses on generating MCQs that align with curriculum-relevant narrative elements and span different difficulty levels. We evaluate these MCQs through expert review and by analyzing the psychometric properties extracted from student responses to assess their suitability for elementary school students. Our results show that current models can generate MCQs of comparable quality to human-authored ones. However, we identify issues related to semantic clarity and answerability. Also, challenges remain in generating distractors that engage students and meet established criteria for high-quality MCQ option design.
>
---
#### [new 030] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification
- **分类: cs.CL**

- **简介: 该论文提出SciVer基准，用于评估基础模型在多模态科学声明验证中的能力，解决多模态科学推理任务中的验证问题。**

- **链接: [http://arxiv.org/pdf/2506.15569v1](http://arxiv.org/pdf/2506.15569v1)**

> **作者:** Chengye Wang; Yifei Shen; Zexi Kuang; Arman Cohan; Yilun Zhao
>
> **摘要:** We introduce SciVer, the first benchmark specifically designed to evaluate the ability of foundation models to verify claims within a multimodal scientific context. SciVer consists of 3,000 expert-annotated examples over 1,113 scientific papers, covering four subsets, each representing a common reasoning type in multimodal scientific claim verification. To enable fine-grained evaluation, each example includes expert-annotated supporting evidence. We assess the performance of 21 state-of-the-art multimodal foundation models, including o4-mini, Gemini-2.5-Flash, Llama-3.2-Vision, and Qwen2.5-VL. Our experiment reveals a substantial performance gap between these models and human experts on SciVer. Through an in-depth analysis of retrieval-augmented generation (RAG), and human-conducted error evaluations, we identify critical limitations in current open-source models, offering key insights to advance models' comprehension and reasoning in multimodal scientific literature tasks.
>
---
#### [new 031] Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放域对话任务，旨在解决对话响应多样性不足的问题。通过分解为多响应生成和偏好选择，提升响应多样性与质量。**

- **链接: [http://arxiv.org/pdf/2506.15131v1](http://arxiv.org/pdf/2506.15131v1)**

> **作者:** Jing Yang Lee; Kong-Aik Lee; Woon-Seng Gan
>
> **摘要:** Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models.
>
---
#### [new 032] From Chat to Checkup: Can Large Language Models Assist in Diabetes Prediction?
- **分类: cs.CL**

- **简介: 该论文属于医疗预测任务，旨在探索大语言模型在糖尿病预测中的应用，通过对比不同模型和提示策略，评估其有效性。**

- **链接: [http://arxiv.org/pdf/2506.14949v1](http://arxiv.org/pdf/2506.14949v1)**

> **作者:** Shadman Sakib; Oishy Fatema Akhand; Ajwad Abrar
>
> **备注:** Accepted in 1st IEEE QPAIN 2025
>
> **摘要:** While Machine Learning (ML) and Deep Learning (DL) models have been widely used for diabetes prediction, the use of Large Language Models (LLMs) for structured numerical data is still not well explored. In this study, we test the effectiveness of LLMs in predicting diabetes using zero-shot, one-shot, and three-shot prompting methods. We conduct an empirical analysis using the Pima Indian Diabetes Database (PIDD). We evaluate six LLMs, including four open-source models: Gemma-2-27B, Mistral-7B, Llama-3.1-8B, and Llama-3.2-2B. We also test two proprietary models: GPT-4o and Gemini Flash 2.0. In addition, we compare their performance with three traditional machine learning models: Random Forest, Logistic Regression, and Support Vector Machine (SVM). We use accuracy, precision, recall, and F1-score as evaluation metrics. Our results show that proprietary LLMs perform better than open-source ones, with GPT-4o and Gemma-2-27B achieving the highest accuracy in few-shot settings. Notably, Gemma-2-27B also outperforms the traditional ML models in terms of F1-score. However, there are still issues such as performance variation across prompting strategies and the need for domain-specific fine-tuning. This study shows that LLMs can be useful for medical prediction tasks and encourages future work on prompt engineering and hybrid approaches to improve healthcare predictions.
>
---
#### [new 033] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture
- **分类: cs.CL**

- **简介: 该论文属于文化理解任务，旨在评估语言模型对印度文化的掌握。通过构建SANSKRITI数据集，测试模型在印度文化相关问题上的表现。**

- **链接: [http://arxiv.org/pdf/2506.15355v1](http://arxiv.org/pdf/2506.15355v1)**

> **作者:** Arijit Maji; Raghvendra Kumar; Akash Ghosh; Anushka; Sriparna Saha
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Language Models (LMs) are indispensable tools shaping modern workflows, but their global effectiveness depends on understanding local socio-cultural contexts. To address this, we introduce SANSKRITI, a benchmark designed to evaluate language models' comprehension of India's rich cultural diversity. Comprising 21,853 meticulously curated question-answer pairs spanning 28 states and 8 union territories, SANSKRITI is the largest dataset for testing Indian cultural knowledge. It covers sixteen key attributes of Indian culture: rituals and ceremonies, history, tourism, cuisine, dance and music, costume, language, art, festivals, religion, medicine, transport, sports, nightlife, and personalities, providing a comprehensive representation of India's cultural tapestry. We evaluate SANSKRITI on leading Large Language Models (LLMs), Indic Language Models (ILMs), and Small Language Models (SLMs), revealing significant disparities in their ability to handle culturally nuanced queries, with many models struggling in region-specific contexts. By offering an extensive, culturally rich, and diverse dataset, SANSKRITI sets a new standard for assessing and improving the cultural understanding of LMs.
>
---
#### [new 034] Emergence of Primacy and Recency Effect in Mamba: A Mechanistic Point of View
- **分类: cs.CL**

- **简介: 该论文研究Mamba模型中的记忆机制，通过primacy和recency效应分析信息保留与遗忘。任务是理解语言模型的记忆行为，解决如何区分长短期记忆及影响因素的问题。工作包括识别三种机制并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.15156v1](http://arxiv.org/pdf/2506.15156v1)**

> **作者:** Muhammad Cendekia Airlangga; Hilal AlQuabeh; Munachiso S Nwadike; Kentaro Inui
>
> **摘要:** We study memory in state-space language models using primacy and recency effects as behavioral tools to uncover how information is retained and forgotten over time. Applying structured recall tasks to the Mamba architecture, we observe a consistent U-shaped accuracy profile, indicating strong performance at the beginning and end of input sequences. We identify three mechanisms that give rise to this pattern. First, long-term memory is supported by a sparse subset of channels within the model's selective state space block, which persistently encode early input tokens and are causally linked to primacy effects. Second, short-term memory is governed by delta-modulated recurrence: recent inputs receive more weight due to exponential decay, but this recency advantage collapses when distractor items are introduced, revealing a clear limit to memory depth. Third, we find that memory allocation is dynamically modulated by semantic regularity: repeated relations in the input sequence shift the delta gating behavior, increasing the tendency to forget intermediate items. We validate these findings via targeted ablations and input perturbations on two large-scale Mamba-based language models: one with 1.4B and another with 7B parameters.
>
---
#### [new 035] Combining Constrained and Unconstrained Decoding via Boosting: BoostCD and Its Application to Information Extraction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的信息抽取任务，旨在解决约束解码质量低的问题。通过结合有约束和无约束解码，提出BoostCD方法提升输出质量。**

- **链接: [http://arxiv.org/pdf/2506.14901v1](http://arxiv.org/pdf/2506.14901v1)**

> **作者:** Marija Šakota; Robert West
>
> **摘要:** Many recent approaches to structured NLP tasks use an autoregressive language model $M$ to map unstructured input text $x$ to output text $y$ representing structured objects (such as tuples, lists, trees, code, etc.), where the desired output structure is enforced via constrained decoding. During training, these approaches do not require the model to be aware of the constraints, which are merely implicit in the training outputs $y$. This is advantageous as it allows for dynamic constraints without requiring retraining, but can lead to low-quality output during constrained decoding at test time. We overcome this problem with Boosted Constrained Decoding (BoostCD), which combines constrained and unconstrained decoding in two phases: Phase 1 decodes from the base model $M$ twice, in constrained and unconstrained mode, obtaining two weak predictions. In phase 2, a learned autoregressive boosted model combines the two weak predictions into one final prediction. The mistakes made by the base model with vs. without constraints tend to be complementary, which the boosted model learns to exploit for improved performance. We demonstrate the power of BoostCD by applying it to closed information extraction. Our model, BoostIE, outperforms prior approaches both in and out of distribution, addressing several common errors identified in those approaches.
>
---
#### [new 036] CKD-EHR:Clinical Knowledge Distillation for Electronic Health Records
- **分类: cs.CL**

- **简介: 该论文属于疾病风险预测任务，旨在解决EHR模型知识不足和部署效率低的问题。通过知识蒸馏技术，将大模型知识迁移至轻量模型，提升准确率与速度。**

- **链接: [http://arxiv.org/pdf/2506.15118v1](http://arxiv.org/pdf/2506.15118v1)**

> **作者:** Junke Wang; Hongshun Ling; Li Zhang; Longqian Zhang; Fang Wang; Yuan Gao; Zhi Li
>
> **备注:** 20 pages,5 figures
>
> **摘要:** Electronic Health Records (EHR)-based disease prediction models have demonstrated significant clinical value in promoting precision medicine and enabling early intervention. However, existing large language models face two major challenges: insufficient representation of medical knowledge and low efficiency in clinical deployment. To address these challenges, this study proposes the CKD-EHR (Clinical Knowledge Distillation for EHR) framework, which achieves efficient and accurate disease risk prediction through knowledge distillation techniques. Specifically, the large language model Qwen2.5-7B is first fine-tuned on medical knowledge-enhanced data to serve as the teacher model.It then generates interpretable soft labels through a multi-granularity attention distillation mechanism. Finally, the distilled knowledge is transferred to a lightweight BERT student model. Experimental results show that on the MIMIC-III dataset, CKD-EHR significantly outperforms the baseline model:diagnostic accuracy is increased by 9%, F1-score is improved by 27%, and a 22.2 times inference speedup is achieved. This innovative solution not only greatly improves resource utilization efficiency but also significantly enhances the accuracy and timeliness of diagnosis, providing a practical technical approach for resource optimization in clinical settings. The code and data for this research are available athttps://github.com/209506702/CKD_EHR.
>
---
#### [new 037] Cohort Discovery: A Survey on LLM-Assisted Clinical Trial Recruitment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床试验招募任务，旨在解决试者与试验匹配问题。通过分析LLM在该领域的应用，探讨其潜力与挑战。**

- **链接: [http://arxiv.org/pdf/2506.15301v1](http://arxiv.org/pdf/2506.15301v1)**

> **作者:** Shrestha Ghosh; Moritz Schneider; Carina Reinicke; Carsten Eickhoff
>
> **摘要:** Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions.
>
---
#### [new 038] Targeted Lexical Injection: Unlocking Latent Cross-Lingual Alignment in Lugha-Llama via Early-Layer LoRA Fine-Tuning
- **分类: cs.CL; 68T50; I.2.7; I.2.6**

- **简介: 该论文属于跨语言对齐任务，旨在提升低资源语言模型的词义对齐能力。通过早期层LoRA微调和对比学习，显著改善了斯瓦希里语与英语的词向量对齐效果。**

- **链接: [http://arxiv.org/pdf/2506.15415v1](http://arxiv.org/pdf/2506.15415v1)**

> **作者:** Stanley Ngugi
>
> **备注:** 11 pages, 3 figures, 2 tables. Research on parameter-efficient fine-tuning (PEFT) for low-resource languages (Swahili). Investigates cross-lingual lexical alignment in Lugha-Llama using LoRA and contrastive learning
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their performance in low-resource languages (LRLs), such as Swahili, often lags due to data scarcity and underrepresentation in pre-training. A key challenge is achieving robust cross-lingual lexical alignment, crucial for tasks like translation and cross-lingual information retrieval. This paper introduces Targeted Lexical Injection (TLI), a novel and efficient fine-tuning approach. We first demonstrate that Lugha-Llama-8B-wura, a Swahili-centric LLM, exhibits strong, near-perfect lexical alignment for Swahili-English word pairs in its early internal layers (specifically Layer 2, with ~0.99998 average cosine similarity based on a pilot study), a capability not fully reflected in its final output representations (baseline ~0.32 similarity on our evaluation set). TLI leverages this insight by using Low-Rank Adaptation (LoRA) and a contrastive learning objective to fine-tune the model, specifically targeting embeddings from this empirically identified optimal early layer. Our experiments show that TLI significantly improves the output-level lexical alignment for 623 trained Swahili-English word pairs, increasing average cosine similarity from 0.3211 to 0.4113 (+28.08%, p < 1.33 x 10^-240). More importantly, these improvements generalize remarkably well to 63 unseen control word pairs, with similarity increasing from 0.3143 to 0.4033 (+28.32%, p < 7.17 x 10^-27). These findings suggest TLI enhances the model's ability to preserve and propagate its inherent early-layer cross-lingual knowledge, offering a parameter-efficient and effective strategy for improving lexical alignment in LRL-focused LLMs.
>
---
#### [new 039] Memory Tokens: Large Language Models Can Generate Reversible Sentence Embeddings
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决如何生成可逆句子嵌入的问题。通过引入记忆标记，使大语言模型能精确重建原始文本。**

- **链接: [http://arxiv.org/pdf/2506.15001v1](http://arxiv.org/pdf/2506.15001v1)**

> **作者:** Ignacio Sastre; Aiala Rosá
>
> **备注:** This paper will be presented at The First Workshop on Large Language Model Memorization (L2M2) at ACL 2025
>
> **摘要:** In this work, we observe an interesting phenomenon: it is possible to generate reversible sentence embeddings that allow an LLM to reconstruct the original text exactly, without modifying the model's weights. This is achieved by introducing a special memory token, whose embedding is optimized through training on a fixed sequence. When prompted with this embedding, the model reconstructs the fixed sequence exactly. We evaluate this phenomenon across English and Spanish datasets, sequences of up to approximately 240 tokens, and model scales ranging from 100M to 8B parameters. Notably, Llama 3.1 8B successfully reconstructs all tested sequences. Our findings highlight an interesting capability of LLMs and suggest potential applications in memory-based retrieval, compression, and controlled text generation.
>
---
#### [new 040] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决大语言模型在遵循指令和组合泛化能力上的不足。提出Ordered CommonGen基准，评估模型按指定顺序生成句子的能力。**

- **链接: [http://arxiv.org/pdf/2506.15629v1](http://arxiv.org/pdf/2506.15629v1)**

> **作者:** Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** ACL 2025 Main
>
> **摘要:** In generative commonsense reasoning tasks such as CommonGen, generative large language models (LLMs) compose sentences that include all given concepts. However, when focusing on instruction-following capabilities, if a prompt specifies a concept order, LLMs must generate sentences that adhere to the specified order. To address this, we propose Ordered CommonGen, a benchmark designed to evaluate the compositional generalization and instruction-following abilities of LLMs. This benchmark measures ordered coverage to assess whether concepts are generated in the specified order, enabling a simultaneous evaluation of both abilities. We conducted a comprehensive analysis using 36 LLMs and found that, while LLMs generally understand the intent of instructions, biases toward specific concept order patterns often lead to low-diversity outputs or identical results even when the concept order is altered. Moreover, even the most instruction-compliant LLM achieved only about 75% ordered coverage, highlighting the need for improvements in both instruction-following and compositional generalization capabilities.
>
---
#### [new 041] Identifying social isolation themes in NVDRS text narratives using topic modeling and text-classification methods
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，旨在通过NLP技术识别NVDRS中社会孤立主题，解决无法直接记录社会孤立的问题。**

- **链接: [http://arxiv.org/pdf/2506.15030v1](http://arxiv.org/pdf/2506.15030v1)**

> **作者:** Drew Walker; Swati Rajwal; Sudeshna Das; Snigdha Peddireddy; Abeed Sarker
>
> **备注:** 22 pages, 2 figures, 5 tables
>
> **摘要:** Social isolation and loneliness, which have been increasing in recent years strongly contribute toward suicide rates. Although social isolation and loneliness are not currently recorded within the US National Violent Death Reporting System's (NVDRS) structured variables, natural language processing (NLP) techniques can be used to identify these constructs in law enforcement and coroner medical examiner narratives. Using topic modeling to generate lexicon development and supervised learning classifiers, we developed high-quality classifiers (average F1: .86, accuracy: .82). Evaluating over 300,000 suicides from 2002 to 2020, we identified 1,198 mentioning chronic social isolation. Decedents had higher odds of chronic social isolation classification if they were men (OR = 1.44; CI: 1.24, 1.69, p<.0001), gay (OR = 3.68; 1.97, 6.33, p<.0001), or were divorced (OR = 3.34; 2.68, 4.19, p<.0001). We found significant predictors for other social isolation topics of recent or impending divorce, child custody loss, eviction or recent move, and break-up. Our methods can improve surveillance and prevention of social isolation and loneliness in the United States.
>
---
#### [new 042] Thunder-Tok: Minimizing Tokens per Word in Tokenizing Korean Texts for Generative Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的分词任务，旨在减少韩语文本的token数量以提升模型效率。通过规则预分词和熵选择算法，Thunder-Tok在不降低性能的前提下降低了token fertility。**

- **链接: [http://arxiv.org/pdf/2506.15138v1](http://arxiv.org/pdf/2506.15138v1)**

> **作者:** Gyeongje Cho; Yeonkyoun So; Chanwoo Park; Sangmin Lee; Sungmok Jung; Jaejin Lee
>
> **摘要:** This paper introduces Thunder-Tok, a new Korean tokenizer designed to reduce token fertility without compromising model performance. Our approach uses a rule-based pre-tokenization method that aligns with the linguistic structure of the Korean language. We also create a seed vocabulary containing tokens that resemble linguistic units and employ a branching entropy-based selection algorithm. These techniques increase the average token length, thus lowering fertility while preserving linguistic information. Experimental results indicate that Thunder-Tok reduces fertility by approximately 10% (i.e., reduces the number of tokens by 10%, improving the inference speed by 10%) compared to BPE without compromising performance across various downstream tasks. These findings demonstrate that our linguistically informed approach is effective and practical for designing efficient tokenizers for language models.
>
---
#### [new 043] RATTENTION: Towards the Minimal Sliding Window Size in Local-Global Attention Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决局部全局注意力模型中窗口大小的效率与性能平衡问题。通过引入RATTENTION机制，实现更小窗口下的高效且高性能模型。**

- **链接: [http://arxiv.org/pdf/2506.15545v1](http://arxiv.org/pdf/2506.15545v1)**

> **作者:** Bailin Wang; Chang Lan; Chong Wang; Ruoming Pang
>
> **备注:** 9 pages
>
> **摘要:** Local-global attention models have recently emerged as compelling alternatives to standard Transformers, promising improvements in both training and inference efficiency. However, the crucial choice of window size presents a Pareto tradeoff: larger windows maintain performance akin to full attention but offer minimal efficiency gains in short-context scenarios, while smaller windows can lead to performance degradation. Current models, such as Gemma2 and Mistral, adopt conservative window sizes (e.g., 4096 out of an 8192 pretraining length) to preserve performance. This work investigates strategies to shift this Pareto frontier, enabling local-global models to achieve efficiency gains even in short-context regimes. Our core motivation is to address the intrinsic limitation of local attention -- its complete disregard for tokens outside the defined window. We explore RATTENTION, a variant of local attention integrated with a specialized linear attention mechanism designed to capture information from these out-of-window tokens. Pretraining experiments at the 3B and 12B scales demonstrate that RATTENTION achieves a superior Pareto tradeoff between performance and efficiency. As a sweet spot, RATTENTION with a window size of just 512 consistently matches the performance of full-attention models across diverse settings. Furthermore, the recurrent nature inherent in the linear attention component of RATTENTION contributes to enhanced long-context performance, as validated on the RULER benchmark. Crucially, these improvements do not compromise training efficiency; thanks to a specialized kernel implementation and the reduced window size, RATTENTION maintains training speeds comparable to existing state-of-the-art approaches.
>
---
#### [new 044] Context-Informed Grounding Supervision
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决模型生成缺乏依据的问题。通过引入CINGS方法，在训练中结合外部上下文，提升生成内容的准确性与一致性。**

- **链接: [http://arxiv.org/pdf/2506.15480v1](http://arxiv.org/pdf/2506.15480v1)**

> **作者:** Hyunji Lee; Seunghyun Yoon; Yunjae Won; Hanseok Oh; Geewook Kim; Trung Bui; Franck Dernoncourt; Elias Stengel-Eskin; Mohit Bansal; Minjoon Seo
>
> **摘要:** Large language models (LLMs) are often supplemented with external knowledge to provide information not encoded in their parameters or to reduce hallucination. In such cases, we expect the model to generate responses by grounding its response in the provided external context. However, prior work has shown that simply appending context at inference time does not ensure grounded generation. To address this, we propose Context-INformed Grounding Supervision (CINGS), a post-training supervision in which the model is trained with relevant context prepended to the response, while computing the loss only over the response tokens and masking out the context. Our experiments demonstrate that models trained with CINGS exhibit stronger grounding in both textual and visual domains compared to standard instruction-tuned models. In the text domain, CINGS outperforms other training methods across 11 information-seeking datasets and is complementary to inference-time grounding techniques. In the vision-language domain, replacing a vision-language model's LLM backbone with a CINGS-trained model reduces hallucinations across four benchmarks and maintains factual consistency throughout the generated response. This improved grounding comes without degradation in general downstream performance. Finally, we analyze the mechanism underlying the enhanced grounding in CINGS and find that it induces a shift in the model's prior knowledge and behavior, implicitly encouraging greater reliance on the external context.
>
---
#### [new 045] GenRecal: Generation after Recalibration from Large to Small Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型压缩任务，旨在解决大模型向小模型知识蒸馏中的架构差异问题。提出GenRecal框架，通过特征对齐实现跨类型模型的知识迁移。**

- **链接: [http://arxiv.org/pdf/2506.15681v1](http://arxiv.org/pdf/2506.15681v1)**

> **作者:** Byung-Kwan Lee; Ryo Hachiuma; Yong Man Ro; Yu-Chiang Frank Wang; Yueh-Hua Wu
>
> **备注:** Project page: https://byungkwanlee.github.io/GenRecal-page/
>
> **摘要:** Recent advancements in vision-language models (VLMs) have leveraged large language models (LLMs) to achieve performance on par with closed-source systems like GPT-4V. However, deploying these models in real-world scenarios, particularly on resource-constrained devices, remains challenging due to their substantial computational demands. This has spurred interest in distilling knowledge from large VLMs into smaller, more efficient counterparts. A key challenge arises here from the diversity of VLM architectures, which are built on different LLMs and employ varying token types-differing in vocabulary size, token splits, and token index ordering. To address this challenge of limitation to a specific VLM type, we present Generation after Recalibration (GenRecal), a novel, general-purpose distillation framework for VLMs. GenRecal incorporates a Recalibrator that aligns and adapts feature representations between heterogeneous VLMs, enabling effective knowledge transfer across different types of VLMs. Through extensive experiments on multiple challenging benchmarks, we demonstrate that GenRecal significantly improves baseline performances, eventually outperforming large-scale open- and closed-source VLMs.
>
---
#### [new 046] Adverse Event Extraction from Discharge Summaries: A New Dataset, Annotation Scheme, and Initial Findings
- **分类: cs.CL**

- **简介: 该论文属于医疗文本中的不良事件抽取任务，旨在解决老年患者出院摘要中罕见不良事件识别难题。研究构建了新数据集并提出标注方案，评估了多种模型性能。**

- **链接: [http://arxiv.org/pdf/2506.14900v1](http://arxiv.org/pdf/2506.14900v1)**

> **作者:** Imane Guellil; Salomé Andres; Atul Anand; Bruce Guthrie; Huayu Zhang; Abul Hasan; Honghan Wu; Beatrice Alex
>
> **备注:** Accepted and will be published at ACL2025 (main conference)
>
> **摘要:** In this work, we present a manually annotated corpus for Adverse Event (AE) extraction from discharge summaries of elderly patients, a population often underrepresented in clinical NLP resources. The dataset includes 14 clinically significant AEs-such as falls, delirium, and intracranial haemorrhage, along with contextual attributes like negation, diagnosis type, and in-hospital occurrence. Uniquely, the annotation schema supports both discontinuous and overlapping entities, addressing challenges rarely tackled in prior work. We evaluate multiple models using FlairNLP across three annotation granularities: fine-grained, coarse-grained, and coarse-grained with negation. While transformer-based models (e.g., BERT-cased) achieve strong performance on document-level coarse-grained extraction (F1 = 0.943), performance drops notably for fine-grained entity-level tasks (e.g., F1 = 0.675), particularly for rare events and complex attributes. These results demonstrate that despite high-level scores, significant challenges remain in detecting underrepresented AEs and capturing nuanced clinical language. Developed within a Trusted Research Environment (TRE), the dataset is available upon request via DataLoch and serves as a robust benchmark for evaluating AE extraction methods and supporting future cross-dataset generalisation.
>
---
#### [new 047] A Comparative Study of Task Adaptation Techniques of Large Language Models for Identifying Sustainable Development Goals
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在识别可持续发展目标（SDGs）。研究比较了不同大语言模型及任务适应技术的效果，发现优化提示的小模型可与大模型媲美。**

- **链接: [http://arxiv.org/pdf/2506.15208v1](http://arxiv.org/pdf/2506.15208v1)**

> **作者:** Andrea Cadeddu; Alessandro Chessa; Vincenzo De Leo; Gianni Fenu; Enrico Motta; Francesco Osborne; Diego Reforgiato Recupero; Angelo Salatino; Luca Secchi
>
> **备注:** Submitted to IEEE Access
>
> **摘要:** In 2012, the United Nations introduced 17 Sustainable Development Goals (SDGs) aimed at creating a more sustainable and improved future by 2030. However, tracking progress toward these goals is difficult because of the extensive scale and complexity of the data involved. Text classification models have become vital tools in this area, automating the analysis of vast amounts of text from a variety of sources. Additionally, large language models (LLMs) have recently proven indispensable for many natural language processing tasks, including text classification, thanks to their ability to recognize complex linguistic patterns and semantics. This study analyzes various proprietary and open-source LLMs for a single-label, multi-class text classification task focused on the SDGs. Then, it also evaluates the effectiveness of task adaptation techniques (i.e., in-context learning approaches), namely Zero-Shot and Few-Shot Learning, as well as Fine-Tuning within this domain. The results reveal that smaller models, when optimized through prompt engineering, can perform on par with larger models like OpenAI's GPT (Generative Pre-trained Transformer).
>
---
#### [new 048] Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本中夸张和隐喻检测任务，旨在解决其语义模糊和表达多样带来的识别难题。通过引入情感引导和双向动态交互机制提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.15504v1](http://arxiv.org/pdf/2506.15504v1)**

> **作者:** Li Zheng; Sihang Wang; Hao Fei; Zuquan Peng; Fei Li; Jianming Fu; Chong Teng; Donghong Ji
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Text-based hyperbole and metaphor detection are of great significance for natural language processing (NLP) tasks. However, due to their semantic obscurity and expressive diversity, it is rather challenging to identify them. Existing methods mostly focus on superficial text features, ignoring the associations of hyperbole and metaphor as well as the effect of implicit emotion on perceiving these rhetorical devices. To implement these hypotheses, we propose an emotion-guided hyperbole and metaphor detection framework based on bidirectional dynamic interaction (EmoBi). Firstly, the emotion analysis module deeply mines the emotion connotations behind hyperbole and metaphor. Next, the emotion-based domain mapping module identifies the target and source domains to gain a deeper understanding of the implicit meanings of hyperbole and metaphor. Finally, the bidirectional dynamic interaction module enables the mutual promotion between hyperbole and metaphor. Meanwhile, a verification mechanism is designed to ensure detection accuracy and reliability. Experiments show that EmoBi outperforms all baseline methods on four datasets. Specifically, compared to the current SoTA, the F1 score increased by 28.1% for hyperbole detection on the TroFi dataset and 23.1% for metaphor detection on the HYPO-L dataset. These results, underpinned by in-depth analyses, underscore the effectiveness and potential of our approach for advancing hyperbole and metaphor detection.
>
---
#### [new 049] Understanding GUI Agent Localization Biases through Logit Sharpness
- **分类: cs.CL**

- **简介: 该论文属于GUI代理定位任务，旨在解决模型定位偏差问题。通过提出评估框架和PSS指标，提升模型的可靠性和可解释性。**

- **链接: [http://arxiv.org/pdf/2506.15425v1](http://arxiv.org/pdf/2506.15425v1)**

> **作者:** Xingjian Tao; Yiwei Wang; Yujun Cai; Zhicheng Yang; Jing Tang
>
> **摘要:** Multimodal large language models (MLLMs) have enabled GUI agents to interact with operating systems by grounding language into spatial actions. Despite their promising performance, these models frequently exhibit hallucinations-systematic localization errors that compromise reliability. We propose a fine-grained evaluation framework that categorizes model predictions into four distinct types, revealing nuanced failure modes beyond traditional accuracy metrics. To better quantify model uncertainty, we introduce the Peak Sharpness Score (PSS), a metric that evaluates the alignment between semantic continuity and logits distribution in coordinate prediction. Building on this insight, we further propose Context-Aware Cropping, a training-free technique that improves model performance by adaptively refining input context. Extensive experiments demonstrate that our framework and methods provide actionable insights and enhance the interpretability and robustness of GUI agent behavior.
>
---
#### [new 050] Gender-Neutral Machine Translation Strategies in Practice
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决性别中立翻译问题。研究评估了21个MT系统在不同翻译方向中的性别中立表现，分析其策略与性别刻板印象的影响。**

- **链接: [http://arxiv.org/pdf/2506.15676v1](http://arxiv.org/pdf/2506.15676v1)**

> **作者:** Hillary Dawkins; Isar Nejadgholi; Chi-kiu Lo
>
> **备注:** to appear at GITT 2025
>
> **摘要:** Gender-inclusive machine translation (MT) should preserve gender ambiguity in the source to avoid misgendering and representational harms. While gender ambiguity often occurs naturally in notional gender languages such as English, maintaining that gender neutrality in grammatical gender languages is a challenge. Here we assess the sensitivity of 21 MT systems to the need for gender neutrality in response to gender ambiguity in three translation directions of varying difficulty. The specific gender-neutral strategies that are observed in practice are categorized and discussed. Additionally, we examine the effect of binary gender stereotypes on the use of gender-neutral translation. In general, we report a disappointing absence of gender-neutral translations in response to gender ambiguity. However, we observe a small handful of MT systems that switch to gender neutral translation using specific strategies, depending on the target language.
>
---
#### [new 051] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MDBench，一个用于评估大语言模型多文档推理能力的合成基准数据集，解决多文档推理评估不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.14927v1](http://arxiv.org/pdf/2506.14927v1)**

> **作者:** Joseph J. Peper; Wenzhao Qiu; Ali Payani; Lu Wang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Natural language processing evaluation has made significant progress, largely driven by the proliferation of powerful large language mod-els (LLMs). New evaluation benchmarks are of increasing priority as the reasoning capabilities of LLMs are expanding at a rapid pace. In particular, while multi-document (MD) reasoning is an area of extreme relevance given LLM capabilities in handling longer-context inputs, few benchmarks exist to rigorously examine model behavior in this setting. Moreover, the multi-document setting is historically challenging for benchmark creation due to the expensive cost of annotating long inputs. In this work, we introduce MDBench, a new dataset for evaluating LLMs on the task of multi-document reasoning. Notably, MDBench is created through a novel synthetic generation process, allowing us to controllably and efficiently generate challenging document sets and the corresponding question-answer (QA) examples. Our novel technique operates on condensed structured seed knowledge, modifying it through LLM-assisted edits to induce MD-specific reasoning challenges. We then convert this structured knowledge into a natural text surface form, generating a document set and corresponding QA example. We analyze the behavior of popular LLMs and prompting techniques, finding that MDBENCH poses significant challenges for all methods, even with relatively short document sets. We also see our knowledge-guided generation technique (1) allows us to readily perform targeted analysis of MD-specific reasoning capabilities and (2) can be adapted quickly to account for new challenges and future modeling improvements.
>
---
#### [new 052] When and How Unlabeled Data Provably Improve In-Context Learning
- **分类: cs.LG; cs.AI; cs.CL; math.OC**

- **简介: 该论文属于半监督学习任务，研究如何利用未标记数据提升上下文学习效果。通过理论分析，提出多层Transformer模型可有效利用未标记数据，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.15329v1](http://arxiv.org/pdf/2506.15329v1)**

> **作者:** Yingcong Li; Xiangyu Chang; Muti Kara; Xiaofeng Liu; Amit Roy-Chowdhury; Samet Oymak
>
> **摘要:** Recent research shows that in-context learning (ICL) can be effective even when demonstrations have missing or incorrect labels. To shed light on this capability, we examine a canonical setting where the demonstrations are drawn according to a binary Gaussian mixture model (GMM) and a certain fraction of the demonstrations have missing labels. We provide a comprehensive theoretical study to show that: (1) The loss landscape of one-layer linear attention models recover the optimal fully-supervised estimator but completely fail to exploit unlabeled data; (2) In contrast, multilayer or looped transformers can effectively leverage unlabeled data by implicitly constructing estimators of the form $\sum_{i\ge 0} a_i (X^\top X)^iX^\top y$ with $X$ and $y$ denoting features and partially-observed labels (with missing entries set to zero). We characterize the class of polynomials that can be expressed as a function of depth and draw connections to Expectation Maximization, an iterative pseudo-labeling algorithm commonly used in semi-supervised learning. Importantly, the leading polynomial power is exponential in depth, so mild amount of depth/looping suffices. As an application of theory, we propose looping off-the-shelf tabular foundation models to enhance their semi-supervision capabilities. Extensive evaluations on real-world datasets show that our method significantly improves the semisupervised tabular learning performance over the standard single pass inference.
>
---
#### [new 053] Dense SAE Latents Are Features, Not Bugs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型中密集潜在表示的性质，旨在解决其是否为训练噪声的问题。通过分析几何结构与功能，发现密集潜变量具有实际功能，应视为模型计算的一部分。**

- **链接: [http://arxiv.org/pdf/2506.15679v1](http://arxiv.org/pdf/2506.15679v1)**

> **作者:** Xiaoqing Sun; Alessandro Stolfo; Joshua Engels; Ben Wu; Senthooran Rajamanoharan; Mrinmaya Sachan; Max Tegmark
>
> **摘要:** Sparse autoencoders (SAEs) are designed to extract interpretable features from language models by enforcing a sparsity constraint. Ideally, training an SAE would yield latents that are both sparse and semantically meaningful. However, many SAE latents activate frequently (i.e., are \emph{dense}), raising concerns that they may be undesirable artifacts of the training procedure. In this work, we systematically investigate the geometry, function, and origin of dense latents and show that they are not only persistent but often reflect meaningful model representations. We first demonstrate that dense latents tend to form antipodal pairs that reconstruct specific directions in the residual stream, and that ablating their subspace suppresses the emergence of new dense features in retrained SAEs -- suggesting that high density features are an intrinsic property of the residual space. We then introduce a taxonomy of dense latents, identifying classes tied to position tracking, context binding, entropy regulation, letter-specific output signals, part-of-speech, and principal component reconstruction. Finally, we analyze how these features evolve across layers, revealing a shift from structural features in early layers, to semantic features in mid layers, and finally to output-oriented signals in the last layers of the model. Our findings indicate that dense latents serve functional roles in language model computation and should not be dismissed as training noise.
>
---
#### [new 054] Identifying economic narratives in large text corpora -- An integrated approach using Large Language Models
- **分类: econ.GN; cs.CL; q-fin.EC**

- **简介: 该论文属于经济叙事识别任务，旨在解决传统模型在深层语义理解上的不足。研究使用LLM（如GPT-4o）提取经济叙事，并与专家标注对比评估效果。**

- **链接: [http://arxiv.org/pdf/2506.15041v1](http://arxiv.org/pdf/2506.15041v1)**

> **作者:** Tobias Schmidt; Kai-Robin Lange; Matthias Reccius; Henrik Müller; Michael Roos; Carsten Jentsch
>
> **备注:** 53 pages, 5 figures
>
> **摘要:** As interest in economic narratives has grown in recent years, so has the number of pipelines dedicated to extracting such narratives from texts. Pipelines often employ a mix of state-of-the-art natural language processing techniques, such as BERT, to tackle this task. While effective on foundational linguistic operations essential for narrative extraction, such models lack the deeper semantic understanding required to distinguish extracting economic narratives from merely conducting classic tasks like Semantic Role Labeling. Instead of relying on complex model pipelines, we evaluate the benefits of Large Language Models (LLMs) by analyzing a corpus of Wall Street Journal and New York Times newspaper articles about inflation. We apply a rigorous narrative definition and compare GPT-4o outputs to gold-standard narratives produced by expert annotators. Our results suggests that GPT-4o is capable of extracting valid economic narratives in a structured format, but still falls short of expert-level performance when handling complex documents and narratives. Given the novelty of LLMs in economic research, we also provide guidance for future work in economics and the social sciences that employs LLMs to pursue similar objectives.
>
---
#### [new 055] ETS: Open Vocabulary Electroencephalography-To-Text Decoding and Sentiment Classification
- **分类: cs.LG; cs.CL; cs.HC**

- **简介: 该论文属于脑机接口任务，旨在解决开放词汇的EEG到文本解码和情感分类问题。通过整合EEG与眼动数据，提出ETS框架，提升解码准确性和情感识别效果。**

- **链接: [http://arxiv.org/pdf/2506.14783v1](http://arxiv.org/pdf/2506.14783v1)**

> **作者:** Mohamed Masry; Mohamed Amen; Mohamed Elzyat; Mohamed Hamed; Norhan Magdy; Maram Khaled
>
> **备注:** Graduation project report submitted at Faculty of Computer Science and Artificial Intelligence, Helwan University
>
> **摘要:** Decoding natural language from brain activity using non-invasive electroencephalography (EEG) remains a significant challenge in neuroscience and machine learning, particularly for open-vocabulary scenarios where traditional methods struggle with noise and variability. Previous studies have achieved high accuracy on small-closed vocabularies, but it still struggles on open vocabularies. In this study, we propose ETS, a framework that integrates EEG with synchronized eye-tracking data to address two critical tasks: (1) open-vocabulary text generation and (2) sentiment classification of perceived language. Our model achieves a superior performance on BLEU and Rouge score for EEG-To-Text decoding and up to 10% F1 score on EEG-based ternary sentiment classification, which significantly outperforms supervised baselines. Furthermore, we show that our proposed model can handle data from various subjects and sources, showing great potential for high performance open vocabulary eeg-to-text system.
>
---
#### [new 056] Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching
- **分类: cs.DC; cs.AI; cs.CL; cs.LG; cs.PF**

- **简介: 该论文属于LLM代理服务优化任务，旨在降低复杂工作流中的成本。通过提取和复用计划模板，提升效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.14852v1](http://arxiv.org/pdf/2506.14852v1)**

> **作者:** Qizheng Zhang; Michael Wornow; Kunle Olukotun
>
> **备注:** 23 pages
>
> **摘要:** LLM-based agentic applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agentic applications where outputs depend on external data or environmental contexts. We propose agentic plan caching, a novel approach that extracts, stores, adapts, and reuses structured plan templates from planning stages of agentic applications across semantically similar tasks to reduce the cost of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agentic applications shows that our system can reduce costs by 46.62% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures.
>
---
#### [new 057] Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于神经网络可解释性任务，旨在解决特征描述中对多义性（polysemanticity）捕捉不足的问题。提出PRISM框架，提升特征描述的准确性和表达能力。**

- **链接: [http://arxiv.org/pdf/2506.15538v1](http://arxiv.org/pdf/2506.15538v1)**

> **作者:** Laura Kopf; Nils Feldhus; Kirill Bykov; Philine Lou Bommer; Anna Hedström; Marina M. -C. Höhne; Oliver Eberle
>
> **摘要:** Automated interpretability research aims to identify concepts encoded in neural network features to enhance human understanding of model behavior. Current feature description methods face two critical challenges: limited robustness and the flawed assumption that each neuron encodes only a single concept (monosemanticity), despite growing evidence that neurons are often polysemantic. This assumption restricts the expressiveness of feature descriptions and limits their ability to capture the full range of behaviors encoded in model internals. To address this, we introduce Polysemantic FeatuRe Identification and Scoring Method (PRISM), a novel framework that captures the inherent complexity of neural network features. Unlike prior approaches that assign a single description per feature, PRISM provides more nuanced descriptions for both polysemantic and monosemantic features. We apply PRISM to language models and, through extensive benchmarking against existing methods, demonstrate that our approach produces more accurate and faithful feature descriptions, improving both overall description quality (via a description score) and the ability to capture distinct concepts when polysemanticity is present (via a polysemanticity score).
>
---
#### [new 058] Argus Inspection: Do Multimodal Large Language Models Possess the Eye of Panoptes?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.14805v1](http://arxiv.org/pdf/2506.14805v1)**

> **作者:** Yang Yao; Lingyu Li; Jiaxin Song; Chiyu Chen; Zhenqi He; Yixu Wang; Xin Wang; Tianle Gu; Jie Li; Yan Teng; Yingchun Wang
>
> **摘要:** As Multimodal Large Language Models (MLLMs) continue to evolve, their cognitive and reasoning capabilities have seen remarkable progress. However, challenges in visual fine-grained perception and commonsense causal inference persist. This paper introduces Argus Inspection, a multimodal benchmark with two levels of difficulty, emphasizing detailed visual recognition while incorporating real-world commonsense understanding to evaluate causal reasoning abilities. Expanding on it, we present the Eye of Panoptes framework, which integrates a binary parametric Sigmoid metric with an indicator function, enabling a more holistic evaluation of MLLMs' responses in opinion-based reasoning tasks. Experiments conducted on 26 mainstream MLLMs reveal that the highest performance in visual fine-grained reasoning reaches only 0.46, highlighting considerable potential for enhancement. Our research offers valuable perspectives for the continued refinement of MLLMs.
>
---
#### [new 059] Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14965v1](http://arxiv.org/pdf/2506.14965v1)**

> **作者:** Zhoujun Cheng; Shibo Hao; Tianyang Liu; Fan Zhou; Yutao Xie; Feng Yao; Yuexin Bian; Yonghao Zhuang; Nilabjo Dey; Yuheng Zha; Yi Gu; Kun Zhou; Yuqi Wang; Yuan Li; Richard Fan; Jianshu She; Chengqian Gao; Abulhair Saparov; Haonan Li; Taylor W. Killian; Mikhail Yurochkin; Zhengzhong Liu; Eric P. Xing; Zhiting Hu
>
> **备注:** 38 pages, 9 figures. Under review
>
> **摘要:** Reinforcement learning (RL) has emerged as a promising approach to improve large language model (LLM) reasoning, yet most open efforts focus narrowly on math and code, limiting our understanding of its broader applicability to general reasoning. A key challenge lies in the lack of reliable, scalable RL reward signals across diverse reasoning domains. We introduce Guru, a curated RL reasoning corpus of 92K verifiable examples spanning six reasoning domains--Math, Code, Science, Logic, Simulation, and Tabular--each built through domain-specific reward design, deduplication, and filtering to ensure reliability and effectiveness for RL training. Based on Guru, we systematically revisit established findings in RL for LLM reasoning and observe significant variation across domains. For example, while prior work suggests that RL primarily elicits existing knowledge from pretrained models, our results reveal a more nuanced pattern: domains frequently seen during pretraining (Math, Code, Science) easily benefit from cross-domain RL training, while domains with limited pretraining exposure (Logic, Simulation, and Tabular) require in-domain training to achieve meaningful performance gains, suggesting that RL is likely to facilitate genuine skill acquisition. Finally, we present Guru-7B and Guru-32B, two models that achieve state-of-the-art performance among open models RL-trained with publicly available data, outperforming best baselines by 7.9% and 6.7% on our 17-task evaluation suite across six reasoning domains. We also show that our models effectively improve the Pass@k performance of their base models, particularly on complex tasks less likely to appear in pretraining data. We release data, models, training and evaluation code to facilitate general-purpose reasoning at: https://github.com/LLM360/Reasoning360
>
---
#### [new 060] LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，解决LLM在微调后安全性能下降的问题。通过LoX方法增强模型对微调的鲁棒性，提升安全性。**

- **链接: [http://arxiv.org/pdf/2506.15606v1](http://arxiv.org/pdf/2506.15606v1)**

> **作者:** Gabrel J. Perin; Runjin Chen; Xuxi Chen; Nina S. T. Hirata; Zhangyang Wang; Junyuan Hong
>
> **摘要:** Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
>
---
#### [new 061] Factorized RVQ-GAN For Disentangled Speech Tokenization
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出HAC，一种统一的语音编解码器，解决语音表示学习问题。通过分层结构分离语音的音素和词义信息，提升语音生成与理解性能。**

- **链接: [http://arxiv.org/pdf/2506.15456v1](http://arxiv.org/pdf/2506.15456v1)**

> **作者:** Sameer Khurana; Dominik Klement; Antoine Laurent; Dominik Bobos; Juraj Novosad; Peter Gazdik; Ellen Zhang; Zili Huang; Amir Hussein; Ricard Marxer; Yoshiki Masuyama; Ryo Aihara; Chiori Hori; Francois G. Germain; Gordon Wichern; Jonathan Le Roux
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We propose Hierarchical Audio Codec (HAC), a unified neural speech codec that factorizes its bottleneck into three linguistic levels-acoustic, phonetic, and lexical-within a single model. HAC leverages two knowledge distillation objectives: one from a pre-trained speech encoder (HuBERT) for phoneme-level structure, and another from a text-based encoder (LaBSE) for lexical cues. Experiments on English and multilingual data show that HAC's factorized bottleneck yields disentangled token sets: one aligns with phonemes, while another captures word-level semantics. Quantitative evaluations confirm that HAC tokens preserve naturalness and provide interpretable linguistic information, outperforming single-level baselines in both disentanglement and reconstruction quality. These findings underscore HAC's potential as a unified discrete speech representation, bridging acoustic detail and lexical meaning for downstream speech generation and understanding tasks.
>
---
#### [new 062] SonicVerse: Multi-Task Learning for Music Feature-Informed Captioning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS; 68T10 (Primary), 68T50 (Secondary); H.5.5; H.5.1; I.2.7**

- **简介: 该论文提出SonicVerse模型，属于音乐描述生成任务，旨在通过多任务学习提升音乐片段的详细描述质量。**

- **链接: [http://arxiv.org/pdf/2506.15154v1](http://arxiv.org/pdf/2506.15154v1)**

> **作者:** Anuradha Chopra; Abhinaba Roy; Dorien Herremans
>
> **备注:** 14 pages, 2 figures, Accepted to AIMC 2025
>
> **摘要:** Detailed captions that accurately reflect the characteristics of a music piece can enrich music databases and drive forward research in music AI. This paper introduces a multi-task music captioning model, SonicVerse, that integrates caption generation with auxiliary music feature detection tasks such as key detection, vocals detection, and more, so as to directly capture both low-level acoustic details as well as high-level musical attributes. The key contribution is a projection-based architecture that transforms audio input into language tokens, while simultaneously detecting music features through dedicated auxiliary heads. The outputs of these heads are also projected into language tokens, to enhance the captioning input. This framework not only produces rich, descriptive captions for short music fragments but also directly enables the generation of detailed time-informed descriptions for longer music pieces, by chaining the outputs using a large-language model. To train the model, we extended the MusicBench dataset by annotating it with music features using MIRFLEX, a modular music feature extractor, resulting in paired audio, captions and music feature data. Experimental results show that incorporating features in this way improves the quality and detail of the generated captions.
>
---
#### [new 063] Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于自然语言处理中的模型训练优化任务，解决LLMs中嵌入学习率与词汇量关系的问题，提出新的缩放规则以提高训练效率。**

- **链接: [http://arxiv.org/pdf/2506.15025v1](http://arxiv.org/pdf/2506.15025v1)**

> **作者:** Soufiane Hayou; Liyuan Liu
>
> **备注:** TD,LR: How to set the learning rate for emebdding layer in LLMs?
>
> **摘要:** Pretraining large language models is a costly process. To make this process more efficient, several methods have been proposed to optimize model architecture/parametrization and hardware use. On the parametrization side, $\mu P$ (Maximal Update Parametrization) parametrizes model weights and learning rate (LR) in a way that makes hyperparameters (HPs) transferable with width (embedding dimension): HPs can be tuned for a small model and used for larger models without additional tuning. While $\mu$P showed impressive results in practice, recent empirical studies have reported conflicting observations when applied to LLMs. One limitation of the theory behind $\mu$P is the fact that input dimension (vocabulary size in LLMs) is considered fixed when taking the width to infinity. This is unrealistic since vocabulary size is generally much larger than width in practice. In this work, we provide a theoretical analysis of the effect of vocabulary size on training dynamics, and subsequently show that as vocabulary size increases, the training dynamics \emph{interpolate between the $\mu$P regime and another regime that we call Large Vocab (LV) Regime}, where optimal scaling rules are different from those predicted by $\mu$P. Our analysis reveals that in the LV regime, the optimal embedding LR to hidden LR ratio should roughly scale as $\Theta(\sqrt{width})$, surprisingly close to the empirical findings previously reported in the literature, and different from the $\Theta(width)$ ratio predicted by $\mu$P. We conduct several experiments to validate our theory, and pretrain a 1B model from scratch to show the benefit of our suggested scaling rule for the embedding LR.
>
---
#### [new 064] video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **简介: 该论文属于视频描述生成任务，旨在提升视频字幕的准确性。通过改进的DPO方法和LoRA技术，优化模型性能，显著降低错误率。**

- **链接: [http://arxiv.org/pdf/2506.15220v1](http://arxiv.org/pdf/2506.15220v1)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimisation (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimised using DPO. To further improve training, we propose a novel multi-round DPO (MrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initialising the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilise the process. Experimental results show that MrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing the captioning error rates by 28\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining highly competitive performance to the state-of-the-art on widely used video question-answering benchmarks among models of similar size. Codes are available at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [new 065] Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.MM; cs.RO**

- **简介: 该论文提出Embodied Web Agents，解决物理与数字智能融合问题。构建了集成3D环境与网络接口的仿真平台，发布基准测试任务，评估跨领域智能。**

- **链接: [http://arxiv.org/pdf/2506.15677v1](http://arxiv.org/pdf/2506.15677v1)**

> **作者:** Yining Hong; Rui Sun; Bingxuan Li; Xingcheng Yao; Maxine Wu; Alexander Chien; Da Yin; Ying Nian Wu; Zhecan James Wang; Kai-Wei Chang
>
> **摘要:** AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
>
---
#### [new 066] AutoRule: Reasoning Chain-of-thought Extracted Rule-based Rewards Improve Preference Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决人工规则工程耗时的问题，提出AutoRule自动提取规则作为奖励，提升偏好学习效果。**

- **链接: [http://arxiv.org/pdf/2506.15651v1](http://arxiv.org/pdf/2506.15651v1)**

> **作者:** Tevin Wang; Chenyan Xiong
>
> **摘要:** Rule-based rewards offer a promising strategy for improving reinforcement learning from human feedback (RLHF), but current approaches often rely on manual rule engineering. We present AutoRule, a fully automated method for extracting rules from preference feedback and formulating them into rule-based rewards. AutoRule extraction operates in three stages: it leverages a reasoning model to interpret user preferences, identifies candidate rules from the reasoning chain of these interpretations, and synthesizes them into a unified rule set. Leveraging the finalized rule set, we employ language-model verifiers to compute the fraction of rules satisfied by each output, using this metric as an auxiliary reward alongside the learned reward model during policy optimization. Training a Llama-3-8B model with AutoRule results in a 28.6\% relative improvement in length-controlled win rate on AlpacaEval2.0, and a 6.1\% relative gain in second-turn performance on a held-out MT-Bench subset, compared to a GRPO baseline trained with the same learned reward model but without the rule-based auxiliary reward. Our analysis confirms that the extracted rules exhibit good agreement with dataset preference. We find that AutoRule demonstrates reduced reward hacking compared to a learned reward model when run over two episodes. Finally, our case study suggests that the extracted rules capture unique qualities valued in different datasets. The extracted rules are provided in the appendix, and the code is open-sourced at https://github.com/cxcscmu/AutoRule.
>
---
#### [new 067] An accurate and revised version of optical character recognition-based speech synthesis using LabVIEW
- **分类: cs.SD; cs.CL; cs.CV; eess.AS; 14J60; I.2.7; I.4; I.5; I.7.5**

- **简介: 该论文属于语音合成任务，旨在解决视障人士获取书籍困难的问题。通过OCR技术与LabVIEW实现准确的语音转换系统。**

- **链接: [http://arxiv.org/pdf/2506.15029v1](http://arxiv.org/pdf/2506.15029v1)**

> **作者:** Prateek Mehta; Anasuya Patil
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Knowledge extraction through sound is a distinctive property. Visually impaired individuals often rely solely on Braille books and audio recordings provided by NGOs. Due to limitations in these approaches, blind individuals often cannot access books of their choice. Speech is a more effective mode of communication than text for blind and visually impaired persons, as they can easily respond to sounds. This paper presents the development of an accurate, reliable, cost-effective, and user-friendly optical character recognition (OCR)-based speech synthesis system. The OCR-based system has been implemented using Laboratory Virtual Instrument Engineering Workbench (LabVIEW).
>
---
#### [new 068] SemIRNet: A Semantic Irony Recognition Network for Multimodal Sarcasm Detection
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态讽刺检测任务，旨在解决图形与文本隐含关联识别困难的问题。提出SemIRNet模型，融合知识库、设计语义相似度模块并引入对比损失函数，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.14791v1](http://arxiv.org/pdf/2506.14791v1)**

> **作者:** Jingxuan Zhou; Yuehao Wu; Yibo Zhang; Yeyubei Zhang; Yunchong Liu; Bolin Huang; Chunhong Yuan
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Aiming at the problem of difficulty in accurately identifying graphical implicit correlations in multimodal irony detection tasks, this paper proposes a Semantic Irony Recognition Network (SemIRNet). The model contains three main innovations: (1) The ConceptNet knowledge base is introduced for the first time to acquire conceptual knowledge, which enhances the model's common-sense reasoning ability; (2) Two cross-modal semantic similarity detection modules at the word level and sample level are designed to model graphic-textual correlations at different granularities; and (3) A contrastive learning loss function is introduced to optimize the spatial distribution of the sample features, which improves the separability of positive and negative samples. Experiments on a publicly available multimodal irony detection benchmark dataset show that the accuracy and F1 value of this model are improved by 1.64% and 2.88% to 88.87% and 86.33%, respectively, compared with the existing optimal methods. Further ablation experiments verify the important role of knowledge fusion and semantic similarity detection in improving the model performance.
>
---
#### [new 069] Assembly of Experts: Linear-time construction of the Chimera LLM variants with emergent and adaptable behaviors
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，旨在高效构建具备新行为的LLM变体。通过线性时间方法整合专家模型，提升性能并减少资源消耗。**

- **链接: [http://arxiv.org/pdf/2506.14794v1](http://arxiv.org/pdf/2506.14794v1)**

> **作者:** Henrik Klagges; Robert Dahlke; Fabian Klemm; Benjamin Merkel; Daniel Klingmann; David A. Reiss; Dan Zecha
>
> **摘要:** Requiring $10^{13}$-$10^{15}$ FLOPs to calculate one 8 bit weight in an LLM during pretraining is extremely expensive and seems inefficient. To better leverage the huge investments made into pretrained models, we develop the new "Assembly-of-Experts" (AoE) construction method to create capable child variants of existing Mixture-of-Experts parent models in linear time. Model weight tensors get interpolated individually, allowing to enhance or suppress semantic features of the parents. Varying the proportion of weights taken from the parent models, we observe some properties of the AoE child model changing gradually, while other behavioral traits emerge with a sharp transition. Surprisingly, nearly every generated model is functional and capable, which makes searching the model space straightforward. We construct the DeepSeek R1T "Chimera", a 671B open-weights hybrid model combining DeepSeek's V3-0324 and R1 model variants. The child inherits only the routed expert tensors of R1, but still achieves about R1-level intelligence. At the same time, it uses about 40\% fewer output tokens, close to V3 speed. Constructed without any fine-tuning or distillation, the Chimera exhibits surprisingly compact, orderly reasoning compared to its parent models.
>
---
#### [new 070] Hypothesis Testing for Quantifying LLM-Human Misalignment in Multiple Choice Settings
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于社会科学研究任务，旨在解决LLM与人类行为不一致的问题。通过假设检验框架，评估语言模型在多选调查中的模拟能力，发现其在不同群体中表现不佳。**

- **链接: [http://arxiv.org/pdf/2506.14997v1](http://arxiv.org/pdf/2506.14997v1)**

> **作者:** Harbin Hong; Sebastian Caldas; Liu Leqi
>
> **摘要:** As Large Language Models (LLMs) increasingly appear in social science research (e.g., economics and marketing), it becomes crucial to assess how well these models replicate human behavior. In this work, using hypothesis testing, we present a quantitative framework to assess the misalignment between LLM-simulated and actual human behaviors in multiple-choice survey settings. This framework allows us to determine in a principled way whether a specific language model can effectively simulate human opinions, decision-making, and general behaviors represented through multiple-choice options. We applied this framework to a popular language model for simulating people's opinions in various public surveys and found that this model is ill-suited for simulating the tested sub-populations (e.g., across different races, ages, and incomes) for contentious questions. This raises questions about the alignment of this language model with the tested populations, highlighting the need for new practices in using LLMs for social science studies beyond naive simulations of human subjects.
>
---
#### [new 071] Detecting Narrative Shifts through Persistent Structures: A Topological Analysis of Media Discourse
- **分类: cs.SI; cs.CL; physics.soc-ph; 55U10**

- **简介: 该论文属于文本分析任务，旨在检测媒体话语中的叙事转变。通过拓扑方法分析事件对公共讨论的影响，识别语义结构的变化。**

- **链接: [http://arxiv.org/pdf/2506.14836v1](http://arxiv.org/pdf/2506.14836v1)**

> **作者:** Mark M. Bailey; Mark I. Heiligman
>
> **备注:** 23 pages
>
> **摘要:** How can we detect when global events fundamentally reshape public discourse? This study introduces a topological framework for identifying structural change in media narratives using persistent homology. Drawing on international news articles surrounding major events - including the Russian invasion of Ukraine (Feb 2022), the murder of George Floyd (May 2020), the U.S. Capitol insurrection (Jan 2021), and the Hamas-led invasion of Israel (Oct 2023) - we construct daily co-occurrence graphs of noun phrases to trace evolving discourse. Each graph is embedded and transformed into a persistence diagram via a Vietoris-Rips filtration. We then compute Wasserstein distances and persistence entropies across homological dimensions to capture semantic disruption and narrative volatility over time. Our results show that major geopolitical and social events align with sharp spikes in both H0 (connected components) and H1 (loops), indicating sudden reorganization in narrative structure and coherence. Cross-correlation analyses reveal a typical lag pattern in which changes to component-level structure (H0) precede higher-order motif shifts (H1), suggesting a bottom-up cascade of semantic change. An exception occurs during the Russian invasion of Ukraine, where H1 entropy leads H0, possibly reflecting top-down narrative framing before local discourse adjusts. Persistence entropy further distinguishes tightly focused from diffuse narrative regimes. These findings demonstrate that persistent homology offers a mathematically principled, unsupervised method for detecting inflection points and directional shifts in public attention - without requiring prior knowledge of specific events. This topological approach advances computational social science by enabling real-time detection of semantic restructuring during crises, protests, and information shocks.
>
---
## 更新

#### [replaced 001] Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence
- **分类: cs.CL; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2410.17161v3](http://arxiv.org/pdf/2410.17161v3)**

> **作者:** İlker Işık; Ramazan Gokberk Cinbis; Ebru Aydin Gol
>
> **备注:** ICML 2025 Poster Paper, Camera Ready Version
>
> **摘要:** Language models lack the notion of interchangeable tokens: symbols that are semantically equivalent yet distinct, such as bound variables in formal logic. This limitation prevents generalization to larger vocabularies and hinders the model's ability to recognize alpha-equivalence, where renaming bound variables preserves meaning. We formalize this machine learning problem and introduce alpha-covariance, a metric for evaluating robustness to such transformations. To tackle this task, we propose a dual-part token embedding strategy: a shared component ensures semantic consistency, while a randomized component maintains token distinguishability. Compared to a baseline that relies on alpha-renaming for data augmentation, our approach demonstrates improved generalization to unseen tokens in linear temporal logic solving, propositional logic assignment prediction, and copying with an extendable vocabulary, while introducing a favorable inductive bias for alpha-equivalence. Our findings establish a foundation for designing language models that can learn interchangeable token representations, a crucial step toward more flexible and systematic reasoning in formal domains. Our code and project page are available at https://necrashter.github.io/interchangeable-token-embeddings
>
---
#### [replaced 002] A Systematic Survey of Natural Language Processing for the Greek Language
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.09861v4](http://arxiv.org/pdf/2407.09861v4)**

> **作者:** Juli Bakagianni; Kanella Pouli; Maria Gavriilidou; John Pavlopoulos
>
> **备注:** This version matches the paper published in Patterns (Cell Press). The title has been updated to reflect the published version
>
> **摘要:** Comprehensive monolingual Natural Language Processing (NLP) surveys are essential for assessing language-specific challenges, resource availability, and research gaps. However, existing surveys often lack standardized methodologies, leading to selection bias and fragmented coverage of NLP tasks and resources. This study introduces a generalizable framework for systematic monolingual NLP surveys. Our approach integrates a structured search protocol to minimize bias, an NLP task taxonomy for classification, and language resource taxonomies to identify potential benchmarks and highlight opportunities for improving resource availability. We apply this framework to Greek NLP (2012-2023), providing an in-depth analysis of its current state, task-specific progress, and resource gaps. The survey results are publicly available (https://doi.org/10.5281/zenodo.15314882) and are regularly updated to provide an evergreen resource. This systematic survey of Greek NLP serves as a case study, demonstrating the effectiveness of our framework and its potential for broader application to other not so well-resourced languages as regards NLP.
>
---
#### [replaced 003] Perspective Transition of Large Language Models for Solving Subjective Tasks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.09265v2](http://arxiv.org/pdf/2501.09265v2)**

> **作者:** Xiaolong Wang; Yuanchi Zhang; Ziyue Wang; Yuzhuang Xu; Fuwen Luo; Yile Wang; Peng Li; Yang Liu
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) have revolutionized the field of natural language processing, enabling remarkable progress in various tasks. Different from objective tasks such as commonsense reasoning and arithmetic question-answering, the performance of LLMs on subjective tasks is still limited, where the perspective on the specific problem plays crucial roles for better interpreting the context and giving proper response. For example, in certain scenarios, LLMs may perform better when answering from an expert role perspective, potentially eliciting their relevant domain knowledge. In contrast, in some scenarios, LLMs may provide more accurate responses when answering from a third-person standpoint, enabling a more comprehensive understanding of the problem and potentially mitigating inherent biases. In this paper, we propose Reasoning through Perspective Transition (RPT), a method based on in-context learning that enables LLMs to dynamically select among direct, role, and third-person perspectives for the best way to solve corresponding subjective problem. Through extensive experiments on totally 12 subjective tasks by using both closed-source and open-source LLMs including GPT-4, GPT-3.5, Llama-3, and Qwen-2, our method outperforms widely used single fixed perspective based methods such as chain-of-thought prompting and expert prompting, highlights the intricate ways that LLMs can adapt their perspectives to provide nuanced and contextually appropriate responses for different problems.
>
---
#### [replaced 004] OM4OV: Leveraging Ontology Matching for Ontology Versioning
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2409.20302v3](http://arxiv.org/pdf/2409.20302v3)**

> **作者:** Zhangcheng Qiang; Kerry Taylor; Weiqing Wang
>
> **备注:** 15 pages, 8 figures, 1 table
>
> **摘要:** Due to the dynamic nature of the Semantic Web, version control is necessary to capture time-varying information, particularly for widely used ontologies. Despite the long-standing recognition of ontology versioning (OV) as a crucial component for efficient ontology management, the growing size of ontologies and accumulating errors caused by manual labour overwhelm current OV approaches. In this paper, we propose yet another approach to performing OV using existing ontology matching (OM) techniques and systems. We introduce a unified OM4OV pipeline. From an OM perspective, we reconstruct a new task formulation and measurement for OV tasks. Building upon the prior alignment(s) from OM, we propose a pipeline optimisation method called the cross-reference (CR) mechanism to enhance overall OV performance. We experimentally validate the OM4OV pipeline and the cross-reference mechanism in the OV tested originating from the Ontology Alignment Evaluation Initiative (OAEI) datasets. We also discuss insights into OM used for OV tasks, where some false mappings detected by OV systems are not actually untrue.
>
---
#### [replaced 005] TransXSSM: A Hybrid Transformer State Space Model with Unified Rotary Position Embedding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09507v3](http://arxiv.org/pdf/2506.09507v3)**

> **作者:** Bingheng Wu; Jingze Shi; Yifan Wu; Nan Tang; Yuyu Luo
>
> **摘要:** Transformers exhibit proficiency in capturing long-range dependencies, whereas State Space Models (SSMs) facilitate linear-time sequence modeling. Notwithstanding their synergistic potential, the integration of these architectures presents a significant challenge, primarily attributable to a fundamental incongr inuity their respective positional encoding mechanisms: Transformers rely on explicit Rotary Position Embeddings (RoPE), while SSMs leverage implicit positional representations via convolutions. This divergence often precipitates discontinuities and suboptimal performance.To address this impediment, we propose a unified rotary position embedding (Unified RoPE) methodology, thereby establishing a consistent positional encoding framework for both self-attention and state-space components. Using this Unified RoPE, we introduce TransXSSM, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme. At a 4 sequenceK length, TransXSSM exhibits training and inference speeds that are 42.3% and 29.5% faster, respectively, relative to standard Transformer models. It also delivers higher accuracy: under comparable settings, it surpasses a Transformer baseline by over 4% on language modeling benchmarks.TransXSSM furthermore scales more effectively: TransXSSM-1.3B gains 7.22% in average accuracy over its 320M version (versus about 6% gains for equivalent Transformers or SSMs). Our results show that unified positional encoding resolves positional incompatibility in hybrid models, enabling efficient, high-performance long-context modeling.
>
---
#### [replaced 006] Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14397v2](http://arxiv.org/pdf/2506.14397v2)**

> **作者:** Yeonkyoung So; Gyuseong Lee; Sungmok Jung; Joonhak Lee; JiA Kang; Sangho Kim; Jaejin Lee
>
> **摘要:** Negation is a fundamental linguistic phenomenon that poses persistent challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Existing benchmarks often treat negation as a side case within broader tasks like natural language inference, resulting in a lack of benchmarks that exclusively target negation understanding. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly designed to assess sentence-level negation understanding in LLMs. Thunder-NUBench goes beyond surface-level cue detection by contrasting standard negation with structurally diverse alternatives such as local negation, contradiction, and paraphrase. The benchmark consists of manually curated sentence-negation pairs and a multiple-choice dataset that enables in-depth evaluation of models' negation understanding.
>
---
#### [replaced 007] Multi-Agent Language Models: Advancing Cooperation, Coordination, and Adaptation
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.09331v2](http://arxiv.org/pdf/2506.09331v2)**

> **作者:** Arjun Vaithilingam Sudhakar
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2311.07687
>
> **摘要:** Modern Large Language Models (LLMs) exhibit impressive zero-shot and few-shot generalization capabilities across complex natural language tasks, enabling their widespread use as virtual assistants for diverse applications such as translation and summarization. Despite being trained solely on large corpora of text without explicit supervision on author intent, LLMs appear to infer the underlying meaning of textual interactions. This raises a fundamental question: can LLMs model and reason about the intentions of others, i.e., do they possess a form of theory of mind? Understanding other's intentions is crucial for effective collaboration, which underpins human societal success and is essential for cooperative interactions among multiple agents, including humans and autonomous systems. In this work, we investigate the theory of mind in LLMs through the lens of cooperative multi-agent reinforcement learning (MARL), where agents learn to collaborate via repeated interactions, mirroring human social reasoning. Our approach aims to enhance artificial agent's ability to adapt and cooperate with both artificial and human partners. By leveraging LLM-based agents capable of natural language interaction, we move towards creating hybrid human-AI systems that can foster seamless collaboration, with broad implications for the future of human-artificial interaction.
>
---
#### [replaced 008] BriefMe: A Legal NLP Benchmark for Assisting with Legal Briefs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06619v2](http://arxiv.org/pdf/2506.06619v2)**

> **作者:** Jesse Woo; Fateme Hashemi Chaleshtori; Ana Marasović; Kenneth Marino
>
> **备注:** ACL Findings 2025; 10 pages main, 5 pages references, 37 pages appendix
>
> **摘要:** A core part of legal work that has been under-explored in Legal NLP is the writing and editing of legal briefs. This requires not only a thorough understanding of the law of a jurisdiction, from judgments to statutes, but also the ability to make new arguments to try to expand the law in a new direction and make novel and creative arguments that are persuasive to judges. To capture and evaluate these legal skills in language models, we introduce BRIEFME, a new dataset focused on legal briefs. It contains three tasks for language models to assist legal professionals in writing briefs: argument summarization, argument completion, and case retrieval. In this work, we describe the creation of these tasks, analyze them, and show how current models perform. We see that today's large language models (LLMs) are already quite good at the summarization and guided completion tasks, even beating human-generated headings. Yet, they perform poorly on other tasks in our benchmark: realistic argument completion and retrieving relevant legal cases. We hope this dataset encourages more development in Legal NLP in ways that will specifically aid people in performing legal work.
>
---
#### [replaced 009] REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.8**

- **链接: [http://arxiv.org/pdf/2412.03092v2](http://arxiv.org/pdf/2412.03092v2)**

> **作者:** Peiyan Zhang; Haibo Jin; Leyang Hu; Xinnuo Li; Liying Kang; Man Luo; Yangqiu Song; Haohan Wang
>
> **备注:** 20 pages, 2 figures, accepted by ICML 2025
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly enhanced the ability of LLM-based systems to perform complex tasks through natural language processing and tool interaction. However, optimizing these LLM-based systems for specific tasks remains challenging, often requiring manual interventions like prompt engineering and hyperparameter tuning. Existing automatic optimization methods, such as textual feedback-based techniques (e.g., TextGrad), tend to focus on immediate feedback, analogous to using immediate derivatives in traditional numerical gradient descent. However, relying solely on such feedback can be limited when the adjustments made in response to this feedback are either too small or fluctuate irregularly, potentially slowing down or even stalling the optimization process. To overcome these challenges, more adaptive methods are needed, especially in situations where the system's response is evolving slowly or unpredictably. In this paper, we introduce REVOLVE, an optimization method that tracks how "R"esponses "EVOLVE" across iterations in LLM systems. By focusing on the evolution of responses over time, REVOLVE enables more stable and effective optimization by making thoughtful, progressive adjustments at each step. Experimental results demonstrate that REVOLVE outperforms competitive baselines, achieving a 7.8% improvement in prompt optimization, a 20.72% gain in solution refinement, and a 29.17% increase in code optimization. Additionally, REVOLVE converges in fewer iterations, resulting in significant computational savings. Beyond its practical contributions, REVOLVE highlights a promising direction, where the rich knowledge from established optimization principles can be leveraged to enhance LLM systems, which paves the way for further advancements in this hybrid domain.
>
---
#### [replaced 010] Efficient Long CoT Reasoning in Small Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18440v2](http://arxiv.org/pdf/2505.18440v2)**

> **作者:** Zhaoyang Wang; Jinqi Jiang; Tian Qiu; Hui Liu; Xianfeng Tang; Huaxiu Yao
>
> **摘要:** Recent large reasoning models such as DeepSeek-R1 exhibit strong complex problems solving abilities by generating long chain-of-thought (CoT) reasoning steps. It is challenging to directly train small language models (SLMs) to emerge long CoT. Thus, distillation becomes a practical method to enable SLMs for such reasoning ability. However, the long CoT often contains a lot of redundant contents (e.g., overthinking steps) which may make SLMs hard to learn considering their relatively poor capacity and generalization. To address this issue, we propose a simple-yet-effective method to prune unnecessary steps in long CoT, and then employ an on-policy method for the SLM itself to curate valid and useful long CoT training data. In this way, SLMs can effectively learn efficient long CoT reasoning and preserve competitive performance at the same time. Experimental results across a series of mathematical reasoning benchmarks demonstrate the effectiveness of the proposed method in distilling long CoT reasoning ability into SLMs which maintains the competitive performance but significantly reduces generating redundant reasoning steps.
>
---
#### [replaced 011] Robust Utility-Preserving Text Anonymization Based on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.11770v2](http://arxiv.org/pdf/2407.11770v2)**

> **作者:** Tianyu Yang; Xiaodan Zhu; Iryna Gurevych
>
> **备注:** Accepted by ACL'2025 Main Conference
>
> **摘要:** Anonymizing text that contains sensitive information is crucial for a wide range of applications. Existing techniques face the emerging challenges of the re-identification ability of large language models (LLMs), which have shown advanced capability in memorizing detailed information and reasoning over dispersed pieces of patterns to draw conclusions. When defending against LLM-based re-identification, anonymization could jeopardize the utility of the resulting anonymized data in downstream tasks. In general, the interaction between anonymization and data utility requires a deeper understanding within the context of LLMs. In this paper, we propose a framework composed of three key LLM-based components: a privacy evaluator, a utility evaluator, and an optimization component, which work collaboratively to perform anonymization. Extensive experiments demonstrate that the proposed model outperforms existing baselines, showing robustness in reducing the risk of re-identification while preserving greater data utility in downstream tasks. We provide detailed studies on these core modules. To consider large-scale and real-time applications, we investigate the distillation of the anonymization capabilities into lightweight models. All of our code and datasets will be made publicly available at https://github.com/UKPLab/acl2025-rupta.
>
---
#### [replaced 012] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.13534v2](http://arxiv.org/pdf/2402.13534v2)**

> **作者:** Xuemei Tang; Jun Wang; Qi Su; Chu-ren Huang; Jinghang Gu
>
> **备注:** 10 pages, 9 tables, 3 figures, Accepted by ACL 2025 (short paper)
>
> **摘要:** Sequence labeling models often benefit from incorporating external knowledge. However, this practice introduces data heterogeneity and complicates the model with additional modules, leading to increased expenses for training a high-performing model. To address this challenge, we propose a two-stage curriculum learning (TCL) framework specifically designed for sequence labeling tasks. The TCL framework enhances training by gradually introducing data instances from easy to hard, aiming to improve both performance and training speed. Furthermore, we explore different metrics for assessing the difficulty levels of sequence labeling tasks. Through extensive experimentation on six Chinese word segmentation (CWS) and Part-of-speech tagging (POS) datasets, we demonstrate the effectiveness of our model in enhancing the performance of sequence labeling models. Additionally, our analysis indicates that TCL accelerates training and alleviates the slow training problem associated with complex models.
>
---
#### [replaced 013] Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.11895v2](http://arxiv.org/pdf/2503.11895v2)**

> **作者:** Bhiman Kumar Baghel; Scott M. Jordan; Zheyuan Ryan Shi; Xiang Lorraine Li
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) are widely deployed in downstream tasks, but keeping their knowledge up-to-date via retraining or fine-tuning is often computationally expensive. Model editing provides a more efficient alternative by updating a targeted subset of parameters, which often follows the locate-and-edit paradigm. Despite this efficiency, existing methods are limited: edits may fail to inject knowledge (UnderEdit) or unintentionally disrupt unrelated neighboring knowledge (OverEdit). To address these challenges, we propose two complementary methods: iterative model editing, which applies successive edits to mitigate UnderEdit, and neighbor-assisted model editing, which incorporates neighboring knowledge during editing to reduce OverEdit. Our extensive experiments show that these techniques improve editing performance across multiple LLMs, algorithms, and benchmarks, reducing UnderEdit by up to 38 percentage points and OverEdit by up to 6, while remaining broadly applicable to any locate-and-edit method.
>
---
#### [replaced 014] Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13300v3](http://arxiv.org/pdf/2506.13300v3)**

> **作者:** Bo Li; Chengben Xu; Wufeng Zhang
>
> **摘要:** This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints.
>
---
#### [replaced 015] Bi-VLDoc: Bidirectional Vision-Language Modeling for Visually-Rich Document Understanding
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2206.13155v2](http://arxiv.org/pdf/2206.13155v2)**

> **作者:** Chuwei Luo; Guozhi Tang; Qi Zheng; Cong Yao; Lianwen Jin; Chenliang Li; Yang Xue; Luo Si
>
> **备注:** IJDAR 2025
>
> **摘要:** Multi-modal document pre-trained models have proven to be very effective in a variety of visually-rich document understanding (VrDU) tasks. Though existing document pre-trained models have achieved excellent performance on standard benchmarks for VrDU, the way they model and exploit the interactions between vision and language on documents has hindered them from better generalization ability and higher accuracy. In this work, we investigate the problem of vision-language joint representation learning for VrDU mainly from the perspective of supervisory signals. Specifically, a pre-training paradigm called Bi-VLDoc is proposed, in which a bidirectional vision-language supervision strategy and a vision-language hybrid-attention mechanism are devised to fully explore and utilize the interactions between these two modalities, to learn stronger cross-modal document representations with richer semantics. Benefiting from the learned informative cross-modal document representations, Bi-VLDoc significantly advances the state-of-the-art performance on three widely-used document understanding benchmarks, including Form Understanding (from 85.14% to 93.44%), Receipt Information Extraction (from 96.01% to 97.84%), and Document Classification (from 96.08% to 97.12%). On Document Visual QA, Bi-VLDoc achieves the state-of-the-art performance compared to previous single model methods.
>
---
#### [replaced 016] How much do language models memorize?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24832v3](http://arxiv.org/pdf/2505.24832v3)**

> **作者:** John X. Morris; Chawin Sitawarin; Chuan Guo; Narine Kokhlikyan; G. Edward Suh; Alexander M. Rush; Kamalika Chaudhuri; Saeed Mahloujifar
>
> **摘要:** We propose a new method for estimating how much a model knows about a datapoint and use it to measure the capacity of modern language models. Prior studies of language model memorization have struggled to disentangle memorization from generalization. We formally separate memorization into two components: unintended memorization, the information a model contains about a specific dataset, and generalization, the information a model contains about the true data-generation process. When we completely eliminate generalization, we can compute the total memorization, which provides an estimate of model capacity: our measurements estimate that GPT-style models have a capacity of approximately 3.6 bits per parameter. We train language models on datasets of increasing size and observe that models memorize until their capacity fills, at which point "grokking" begins, and unintended memorization decreases as models begin to generalize. We train hundreds of transformer language models ranging from $500K$ to $1.5B$ parameters and produce a series of scaling laws relating model capacity and data size to membership inference.
>
---
#### [replaced 017] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.14625v2](http://arxiv.org/pdf/2506.14625v2)**

> **作者:** Chenchen Yuan; Zheyu Zhang; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** Large Language Models (LLMs) have shown impressive moral reasoning abilities. Yet they often diverge when confronted with complex, multi-factor moral dilemmas. To address these discrepancies, we propose a framework that synthesizes multiple LLMs' moral judgments into a collectively formulated moral judgment, realigning models that deviate significantly from this consensus. Our aggregation mechanism fuses continuous moral acceptability scores (beyond binary labels) into a collective probability, weighting contributions by model reliability. For misaligned models, a targeted embedding-optimization procedure fine-tunes token embeddings for moral philosophical theories, minimizing JS divergence to the consensus while preserving semantic integrity. Experiments on a large-scale social moral dilemma dataset show our approach builds robust consensus and improves individual model fidelity. These findings highlight the value of data-driven moral alignment across multiple models and its potential for safer, more consistent AI systems.
>
---
#### [replaced 018] Lean Workbook: A large-scale Lean problem set formalized from natural language math problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.03847v3](http://arxiv.org/pdf/2406.03847v3)**

> **作者:** Huaiyuan Ying; Zijian Wu; Yihan Geng; Zheng Yuan; Dahua Lin; Kai Chen
>
> **摘要:** Large language models have demonstrated impressive capabilities across various natural language processing tasks, especially in solving mathematical problems. However, large language models are not good at math theorem proving using formal languages like Lean. A significant challenge in this area is the scarcity of training data available in these formal languages. To address this issue, we propose a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa. Our results indicate that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs. Our final dataset contains about 57K formal-informal question pairs along with searched proof from the math contest forum and 21 new IMO questions. We open-source our code at https://github.com/InternLM/InternLM-Math and our data at https://huggingface.co/datasets/InternLM/Lean-Workbook.
>
---
#### [replaced 019] UD-English-CHILDES: A Collected Resource of Gold and Silver Universal Dependencies Trees for Child Language Interactions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.20304v3](http://arxiv.org/pdf/2504.20304v3)**

> **作者:** Xiulin Yang; Zhuoxuan Ju; Lanni Bu; Zoey Liu; Nathan Schneider
>
> **备注:** UDW 2025
>
> **摘要:** CHILDES is a widely used resource of transcribed child and child-directed speech. This paper introduces UD-English-CHILDES, the first officially released Universal Dependencies (UD) treebank. It is derived from previously dependency-annotated CHILDES data, which we harmonize to follow unified annotation principles. The gold-standard trees encompass utterances sampled from 11 children and their caregivers, totaling over 48K sentences (236K tokens). We validate these gold-standard annotations under the UD v2 framework and provide an additional 1M~silver-standard sentences, offering a consistent resource for computational and linguistic research.
>
---
#### [replaced 020] ALPS: Attention Localization and Pruning Strategy for Efficient Alignment of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18799v4](http://arxiv.org/pdf/2505.18799v4)**

> **作者:** Hao Chen; Haoze Li; Zhiqing Xiao; Lirong Gao; Qi Zhang; Xiaomeng Hu; Ningtao Wang; Xing Fu; Junbo Zhao
>
> **备注:** Accepted@ACL25-findings, 17 pages, 8 figures, 14 tables
>
> **摘要:** Aligning general-purpose large language models (LLMs) to downstream tasks often incurs significant training adjustment costs. Prior research has explored various avenues to enhance alignment efficiency, primarily through minimal-data training or data-driven activations to identify key attention heads. However, these approaches inherently introduce data dependency, which hinders generalization and reusability. To address this issue and enhance model alignment efficiency, we propose the Attention Localization and Pruning Strategy (ALPS), an efficient algorithm that localizes the most task-sensitive attention heads and prunes by restricting attention training updates to these heads, thereby reducing alignment costs. Experimental results demonstrate that our method activates only 10% of attention parameters during fine-tuning while achieving a 2% performance improvement over baselines on three tasks. Moreover, the identified task-specific heads are transferable across datasets and mitigate knowledge forgetting. Our work and findings provide a novel perspective on efficient LLM alignment. The code is available at https://github.com/VoiceBeer/ALPS.
>
---
#### [replaced 021] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.16930v4](http://arxiv.org/pdf/2410.16930v4)**

> **作者:** Bryan R. Christ; Zack Gottesman; Jonathan Kropko; Thomas Hartvigsen
>
> **备注:** 38 pages, 54 figures, Accepted to ACL 2025 (Main)
>
> **摘要:** Math reasoning is an active area of Large Language Model (LLM) research because it is a hallmark of artificial intelligence and has implications in several domains, including math education. However, few works have explored how math reasoning is encoded within LLM parameters and if it is a skill that can be isolated within models. Doing so could allow targeted intervention to improve math performance without altering non-math behavior and foster understanding of how models encode math reasoning. We introduce Math Neurosurgery (MathNeuro), a computationally efficient method we use to isolate math-specific parameters in LLMs using only forward passes. MathNeuro builds on existing work by using weights and activations to calculate parameter importance, but isolates math-specific parameters by filtering out those important for general language tasks. Through pruning parameters MathNeuro identifies, we delete a LLM's math reasoning ability without significantly impacting its general language ability. Scaling the identified parameters by a small constant improves a pretrained or instruction-tuned LLM's performance by 4-17% on GSM8K and 5-35% on MATH while leaving non-math behavior unaltered. MathNeuro is also data efficient: most of its effectiveness holds when identifying math-specific parameters using a single sample. MathNeuro highlights the potential for future work to intervene on math-specific parameters.
>
---
#### [replaced 022] Entropy-based Exploration Conduction for Multi-step Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.15848v2](http://arxiv.org/pdf/2503.15848v2)**

> **作者:** Jinghan Zhang; Xiting Wang; Fengran Mo; Yeyang Zhou; Wanfu Gao; Kunpeng Liu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Multi-step processes via large language models (LLMs) have proven effective for solving complex reasoning tasks. However, the depth of exploration of the reasoning procedure can significantly affect the task performance. Existing methods to automatically decide the depth often lead to high cost and a lack of flexibility. To address these issues, we propose Entropy-based Exploration Depth Conduction (Entro-duction), a novel method that dynamically adjusts the exploration depth during multi-step reasoning by monitoring LLM's output entropy and variance entropy. We employ these two features to capture the model's uncertainty of the current step and the fluctuation of uncertainty across consecutive reasoning steps. Based on the observed entropy changes, the LLM selects whether to deepen, expand, or stop exploration according to the probability, which facilitates the trade-off between the reasoning accuracy and exploration effectiveness. Experimental results across four benchmark datasets demonstrate the efficacy of Entro-duction.
>
---
#### [replaced 023] A Guide to Misinformation Detection Data and Evaluation
- **分类: cs.SI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2411.05060v4](http://arxiv.org/pdf/2411.05060v4)**

> **作者:** Camille Thibault; Jacob-Junqi Tian; Gabrielle Peloquin-Skulski; Taylor Lynn Curtis; James Zhou; Florence Laflamme; Yuxiang Guan; Reihaneh Rabbany; Jean-François Godbout; Kellin Pelrine
>
> **摘要:** Misinformation is a complex societal issue, and mitigating solutions are difficult to create due to data deficiencies. To address this, we have curated the largest collection of (mis)information datasets in the literature, totaling 75. From these, we evaluated the quality of 36 datasets that consist of statements or claims, as well as the 9 datasets that consist of data in purely paragraph form. We assess these datasets to identify those with solid foundations for empirical work and those with flaws that could result in misleading and non-generalizable results, such as spurious correlations, or examples that are ambiguous or otherwise impossible to assess for veracity. We find the latter issue is particularly severe and affects most datasets in the literature. We further provide state-of-the-art baselines on all these datasets, but show that regardless of label quality, categorical labels may no longer give an accurate evaluation of detection model performance. Finally, we propose and highlight Evaluation Quality Assurance (EQA) as a tool to guide the field toward systemic solutions rather than inadvertently propagating issues in evaluation. Overall, this guide aims to provide a roadmap for higher quality data and better grounded evaluations, ultimately improving research in misinformation detection. All datasets and other artifacts are available at misinfo-datasets.complexdatalab.com.
>
---
#### [replaced 024] Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09033v2](http://arxiv.org/pdf/2506.09033v2)**

> **作者:** Haozhen Zhang; Tao Feng; Jiaxuan You
>
> **备注:** Code is available at https://github.com/ulab-uiuc/Router-R1. Models and Datasets are available at https://huggingface.co/collections/ulab-ai/router-r1-6851bbe099c7a56914b5db03
>
> **摘要:** The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To facilitate learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for optimizing the balance between performance and cost, opening a pathway toward enhancing performance-cost trade-offs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms several strong baselines, achieving superior performance while maintaining robust generalization and cost management.
>
---
#### [replaced 025] AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.14634v2](http://arxiv.org/pdf/2506.14634v2)**

> **作者:** Leah von der Heyde; Anna-Carolina Haensch; Bernd Weiß; Jessica Daikeler
>
> **备注:** to appear in Survey Research Methods
>
> **摘要:** The recent development and wider accessibility of LLMs have spurred discussions about how they can be used in survey research, including classifying open-ended survey responses. Due to their linguistic capacities, it is possible that LLMs are an efficient alternative to time-consuming manual coding and the pre-training of supervised machine learning models. As most existing research on this topic has focused on English-language responses relating to non-complex topics or on single LLMs, it is unclear whether its findings generalize and how the quality of these classifications compares to established methods. In this study, we investigate to what extent different LLMs can be used to code open-ended survey responses in other contexts, using German data on reasons for survey participation as an example. We compare several state-of-the-art LLMs and several prompting approaches, and evaluate the LLMs' performance by using human expert codings. Overall performance differs greatly between LLMs, and only a fine-tuned LLM achieves satisfactory levels of predictive performance. Performance differences between prompting approaches are conditional on the LLM used. Finally, LLMs' unequal classification performance across different categories of reasons for survey participation results in different categorical distributions when not using fine-tuning. We discuss the implications of these findings, both for methodological research on coding open-ended responses and for their substantive analysis, and for practitioners processing or substantively analyzing such data. Finally, we highlight the many trade-offs researchers need to consider when choosing automated methods for open-ended response classification in the age of LLMs. In doing so, our study contributes to the growing body of research about the conditions under which LLMs can be efficiently, accurately, and reliably leveraged in survey research.
>
---
#### [replaced 026] Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.08327v2](http://arxiv.org/pdf/2503.08327v2)**

> **作者:** José Pombal; Nuno M. Guerreiro; Ricardo Rei; André F. T. Martins
>
> **摘要:** As automatic metrics become increasingly stronger and widely adopted, the risk of unintentionally "gaming the metric" during model development rises. This issue is caused by metric interference (MINT), i.e., the use of the same or related metrics for both model tuning and evaluation. MINT can misguide practitioners into being overoptimistic about the performance of their systems: as system outputs become a function of the interfering metric, their estimated quality loses correlation with human judgments. In this work, we analyze two common cases of MINT in machine translation-related tasks: filtering of training data, and decoding with quality signals. Importantly, we find that MINT strongly distorts instance-level metric scores, even when metrics are not directly optimized for-questioning the common strategy of leveraging a different, yet related metric for evaluation that is not used for tuning. To address this problem, we propose MINTADJUST, a method for more reliable evaluation under MINT. On the WMT24 MT shared task test set, MINTADJUST ranks translations and systems more accurately than state-of-the-art metrics across a majority of language pairs, especially for high-quality systems. Furthermore, MINTADJUST outperforms AUTORANK, the ensembling method used by the organizers.
>
---
#### [replaced 027] Fractured Chain-of-Thought Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.12992v3](http://arxiv.org/pdf/2505.12992v3)**

> **作者:** Baohao Liao; Hanze Dong; Yuhui Xu; Doyen Sahoo; Christof Monz; Junnan Li; Caiming Xiong
>
> **摘要:** Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning. Code is available at https://github.com/BaohaoLiao/frac-cot.
>
---
#### [replaced 028] LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.06561v2](http://arxiv.org/pdf/2506.06561v2)**

> **作者:** Ho Yin 'Sam' Ng; Ting-Yao Hsu; Aashish Anantha Ramakrishnan; Branislav Kveton; Nedim Lipka; Franck Dernoncourt; Dongwon Lee; Tong Yu; Sungchul Kim; Ryan A. Rossi; Ting-Hao 'Kenneth' Huang
>
> **备注:** The LaMP-CAP dataset is publicly available at: https://github.com/Crowd-AI-Lab/lamp-cap
>
> **摘要:** Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones.
>
---
#### [replaced 029] SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02803v2](http://arxiv.org/pdf/2506.02803v2)**

> **作者:** Sifan Li; Yujun Cai; Yiwei Wang
>
> **摘要:** Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden text, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0-5.36%)-even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions (32-128 pixels), which unlocks >99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond.
>
---
#### [replaced 030] I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14693v3](http://arxiv.org/pdf/2502.14693v3)**

> **作者:** Zujie Liang; Feng Wei; Wujiang Xu; Lin Chen; Yuxi Qian; Xinhui Wu
>
> **摘要:** Recent advancements in large language models (LLMs) have shown remarkable potential in automating machine learning tasks. However, existing LLM-based agents often struggle with low-diversity and suboptimal code generation. While recent work has introduced Monte Carlo Tree Search (MCTS) to address these issues, limitations persist in the quality and diversity of thoughts generated, as well as in the scalar value feedback mechanisms used for node selection. In this study, we introduce Introspective Monte Carlo Tree Search (I-MCTS), a novel approach that iteratively expands tree nodes through an introspective process that meticulously analyzes solutions and results from parent and sibling nodes. This facilitates a continuous refinement of the node in the search tree, thereby enhancing the overall decision-making process. Furthermore, we integrate a Large Language Model (LLM)-based value model to facilitate direct evaluation of each node's solution prior to conducting comprehensive computational rollouts. A hybrid rewarding mechanism is implemented to seamlessly transition the Q-value from LLM-estimated scores to actual performance scores. This allows higher-quality nodes to be traversed earlier. Applied to the various ML tasks, our approach demonstrates a 6% absolute improvement in performance compared to the strong open-source AutoML agents, showcasing its effectiveness in enhancing agentic AutoML systems. Resource available at https://github.com/jokieleung/I-MCTS
>
---
#### [replaced 031] HiURE: Hierarchical Exemplar Contrastive Learning for Unsupervised Relation Extraction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2205.02225v4](http://arxiv.org/pdf/2205.02225v4)**

> **作者:** Shuliang Liu; Xuming Hu; Chenwei Zhang; Shu`ang Li; Lijie Wen; Philip S. Yu
>
> **备注:** In NAACL 2022 as a long paper. Code and data available at https://github.com/THU-BPM/HiURE
>
> **摘要:** Unsupervised relation extraction aims to extract the relationship between entities from natural language sentences without prior information on relational scope or distribution. Existing works either utilize self-supervised schemes to refine relational feature signals by iteratively leveraging adaptive clustering and classification that provoke gradual drift problems, or adopt instance-wise contrastive learning which unreasonably pushes apart those sentence pairs that are semantically similar. To overcome these defects, we propose a novel contrastive learning framework named HiURE, which has the capability to derive hierarchical signals from relational feature space using cross hierarchy attention and effectively optimize relation representation of sentences under exemplar-wise contrastive learning. Experimental results on two public datasets demonstrate the advanced effectiveness and robustness of HiURE on unsupervised relation extraction when compared with state-of-the-art models.
>
---
#### [replaced 032] Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.14731v2](http://arxiv.org/pdf/2506.14731v2)**

> **作者:** Ling Team; Bin Hu; Cai Chen; Deng Zhao; Ding Liu; Dingnan Jin; Feng Zhu; Hao Dai; Hongzhi Luan; Jia Guo; Jiaming Liu; Jiewei Wu; Jun Mei; Jun Zhou; Junbo Zhao; Junwu Xiong; Kaihong Zhang; Kuan Xu; Lei Liang; Liang Jiang; Liangcheng Fu; Longfei Zheng; Qiang Gao; Qing Cui; Quan Wan; Shaomian Zheng; Shuaicheng Li; Tongkai Yang; Wang Ren; Xiaodong Yan; Xiaopei Wan; Xiaoyun Feng; Xin Zhao; Xinxing Yang; Xinyu Kong; Xuemin Yang; Yang Li; Yingting Wu; Yongkang Liu; Zhankai Xu; Zhenduo Zhang; Zhenglei Zhou; Zhenyu Huang; Zhiqiang Zhang; Zihao Wang; Zujie Wen
>
> **备注:** Technical Report
>
> **摘要:** We present Ring-lite, a Mixture-of-Experts (MoE)-based large language model optimized via reinforcement learning (RL) to achieve efficient and robust reasoning capabilities. Built upon the publicly available Ling-lite model, a 16.8 billion parameter model with 2.75 billion activated parameters, our approach matches the performance of state-of-the-art (SOTA) small-scale reasoning models on challenging benchmarks (e.g., AIME, LiveCodeBench, GPQA-Diamond) while activating only one-third of the parameters required by comparable models. To accomplish this, we introduce a joint training pipeline integrating distillation with RL, revealing undocumented challenges in MoE RL training. First, we identify optimization instability during RL training, and we propose Constrained Contextual Computation Policy Optimization(C3PO), a novel approach that enhances training stability and improves computational throughput via algorithm-system co-design methodology. Second, we empirically demonstrate that selecting distillation checkpoints based on entropy loss for RL training, rather than validation metrics, yields superior performance-efficiency trade-offs in subsequent RL training. Finally, we develop a two-stage training paradigm to harmonize multi-domain data integration, addressing domain conflicts that arise in training with mixed dataset. We will release the model, dataset, and code.
>
---
#### [replaced 033] Large Language Models for Automated Literature Review: An Evaluation of Reference Generation, Abstract Writing, and Review Composition
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.13612v4](http://arxiv.org/pdf/2412.13612v4)**

> **作者:** Xuemei Tang; Xufeng Duan; Zhenguang G. Cai
>
> **备注:** 12 pages, 5 figures, 5 tables
>
> **摘要:** Large language models (LLMs) have emerged as a potential solution to automate the complex processes involved in writing literature reviews, such as literature collection, organization, and summarization. However, it is yet unclear how good LLMs are at automating comprehensive and reliable literature reviews. This study introduces a framework to automatically evaluate the performance of LLMs in three key tasks of literature writing: reference generation, literature summary, and literature review composition. We introduce multidimensional evaluation metrics that assess the hallucination rates in generated references and measure the semantic coverage and factual consistency of the literature summaries and compositions against human-written counterparts. The experimental results reveal that even the most advanced models still generate hallucinated references, despite recent progress. Moreover, we observe that the performance of different models varies across disciplines when it comes to writing literature reviews. These findings highlight the need for further research and development to improve the reliability of LLMs in automating academic literature reviews.
>
---
#### [replaced 034] CODESYNC: Synchronizing Large Language Models with Dynamic Code Evolution at Scale
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.16645v2](http://arxiv.org/pdf/2502.16645v2)**

> **作者:** Chenlong Wang; Zhaoyang Chu; Zhengxiang Cheng; Xuyi Yang; Kaiyue Qiu; Yao Wan; Zhou Zhao; Xuanhua Shi; Dongping Chen
>
> **摘要:** Large Language Models (LLMs) have exhibited exceptional performance in software engineering yet face challenges in adapting to continually evolving code knowledge, particularly regarding the frequent updates of third-party library APIs. This limitation, stemming from static pre-training datasets, often results in non-executable code or implementations with suboptimal safety and efficiency. To this end, this paper introduces CODESYNC, a data engine for identifying outdated code patterns and collecting real-time code knowledge updates from Python third-party libraries. Building upon CODESYNC, we develop CODESYNCBENCH, a comprehensive benchmark for assessing LLMs' ability to stay synchronized with code evolution, which covers real-world updates for 220 APIs from six Python libraries. Our benchmark offers 3,300 test cases across three evaluation tasks and an update-aware instruction tuning dataset consisting of 2,200 training samples. Extensive experiments on 14 state-of-the-art LLMs reveal that they struggle with dynamic code evolution, even with the support of advanced knowledge updating methods (e.g., DPO, ORPO, and SimPO). We believe that our benchmark can offer a strong foundation for the development of more effective methods for real-time code knowledge updating in the future. The experimental code and dataset are publicly available at: https://github.com/Lucky-voyage/Code-Sync.
>
---
#### [replaced 035] Dynamic Acoustic Model Architecture Optimization in Training for ASR
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13180v2](http://arxiv.org/pdf/2506.13180v2)**

> **作者:** Jingjing Xu; Zijian Yang; Albert Zeyer; Eugen Beck; Ralf Schlueter; Hermann Ney
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Architecture design is inherently complex. Existing approaches rely on either handcrafted rules, which demand extensive empirical expertise, or automated methods like neural architecture search, which are computationally intensive. In this paper, we introduce DMAO, an architecture optimization framework that employs a grow-and-drop strategy to automatically reallocate parameters during training. This reallocation shifts resources from less-utilized areas to those parts of the model where they are most beneficial. Notably, DMAO only introduces negligible training overhead at a given model complexity. We evaluate DMAO through experiments with CTC on LibriSpeech, TED-LIUM-v2 and Switchboard datasets. The results show that, using the same amount of training resources, our proposed DMAO consistently improves WER by up to 6% relatively across various architectures, model sizes, and datasets. Furthermore, we analyze the pattern of parameter redistribution and uncover insightful findings.
>
---
#### [replaced 036] ChemHAS: Hierarchical Agent Stacking for Enhancing Chemistry Tools
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21569v2](http://arxiv.org/pdf/2505.21569v2)**

> **作者:** Zhucong Li; Bowei Zhang; Jin Xiao; Zhijian Zhou; Fenglei Cao; Jiaqing Liang; Yuan Qi
>
> **备注:** 9 pages
>
> **摘要:** Large Language Model (LLM)-based agents have demonstrated the ability to improve performance in chemistry-related tasks by selecting appropriate tools. However, their effectiveness remains limited by the inherent prediction errors of chemistry tools. In this paper, we take a step further by exploring how LLMbased agents can, in turn, be leveraged to reduce prediction errors of the tools. To this end, we propose ChemHAS (Chemical Hierarchical Agent Stacking), a simple yet effective method that enhances chemistry tools through optimizing agent-stacking structures from limited data. ChemHAS achieves state-of-the-art performance across four fundamental chemistry tasks, demonstrating that our method can effectively compensate for prediction errors of the tools. Furthermore, we identify and characterize four distinct agent-stacking behaviors, potentially improving interpretability and revealing new possibilities for AI agent applications in scientific research. Our code and dataset are publicly available at https: //anonymous.4open.science/r/ChemHAS-01E4/README.md.
>
---
#### [replaced 037] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19941v3](http://arxiv.org/pdf/2502.19941v3)**

> **作者:** Xiang Geng; Zhejian Lai; Jiajun Chen; Hao Yang; Shujian Huang
>
> **备注:** ACL2025 Main
>
> **摘要:** Quality Estimation (QE) models evaluate the quality of machine translations without reference translations, serving as the reward models for the translation task. Due to the data scarcity, synthetic data generation has emerged as a promising solution. However, synthetic QE data often suffers from distribution shift, which can manifest as discrepancies between pseudo and real translations, or in pseudo labels that do not align with human preferences. To tackle this issue, we introduce DCSQE, a novel framework for alleviating distribution shift in synthetic QE data. To reduce the difference between pseudo and real translations, we employ the constrained beam search algorithm and enhance translation diversity through the use of distinct generation models. DCSQE uses references, i.e., translation supervision signals, to guide both the generation and annotation processes, enhancing the quality of token-level labels. DCSQE further identifies the shortest phrase covering consecutive error tokens, mimicking human annotation behavior, to assign the final phrase-level labels. Specially, we underscore that the translation model can not annotate translations of itself accurately. Extensive experiments demonstrate that DCSQE outperforms SOTA baselines like CometKiwi in both supervised and unsupervised settings. Further analysis offers insights into synthetic data generation that could benefit reward models for other tasks. The code is available at https://github.com/NJUNLP/njuqe.
>
---
#### [replaced 038] RadioRAG: Online Retrieval-augmented Generation for Radiology Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.15621v3](http://arxiv.org/pdf/2407.15621v3)**

> **作者:** Soroosh Tayebi Arasteh; Mahshad Lotfinia; Keno Bressem; Robert Siepmann; Lisa Adams; Dyke Ferber; Christiane Kuhl; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **备注:** Published in Radiology: Artificial Intelligence
>
> **摘要:** Large language models (LLMs) often generate outdated or inaccurate information based on static training datasets. Retrieval-augmented generation (RAG) mitigates this by integrating outside data sources. While previous RAG systems used pre-assembled, fixed databases with limited flexibility, we have developed Radiology RAG (RadioRAG), an end-to-end framework that retrieves data from authoritative radiologic online sources in real-time. We evaluate the diagnostic accuracy of various LLMs when answering radiology-specific questions with and without access to additional online information via RAG. Using 80 questions from the RSNA Case Collection across radiologic subspecialties and 24 additional expert-curated questions with reference standard answers, LLMs (GPT-3.5-turbo, GPT-4, Mistral-7B, Mixtral-8x7B, and Llama3 [8B and 70B]) were prompted with and without RadioRAG in a zero-shot inference scenario RadioRAG retrieved context-specific information from Radiopaedia in real-time. Accuracy was investigated. Statistical analyses were performed using bootstrapping. The results were further compared with human performance. RadioRAG improved diagnostic accuracy across most LLMs, with relative accuracy increases ranging up to 54% for different LLMs. It matched or exceeded non-RAG models and the human radiologist in question answering across radiologic subspecialties, particularly in breast imaging and emergency radiology. However, the degree of improvement varied among models; GPT-3.5-turbo and Mixtral-8x7B-instruct-v0.1 saw notable gains, while Mistral-7B-instruct-v0.2 showed no improvement, highlighting variability in RadioRAG's effectiveness. LLMs benefit when provided access to domain-specific data beyond their training data. RadioRAG shows potential to improve LLM accuracy and factuality in radiology question answering by integrating real-time domain-specific data.
>
---
#### [replaced 039] BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.06955v2](http://arxiv.org/pdf/2506.06955v2)**

> **作者:** Ha-Thanh Nguyen; Chaoran Liu; Koichi Takeda; Yusuke Miyao; Pontus Stenetorp; Qianying Liu; Su Myat Noe; Hideyuki Tachibana; Sadao Kurohashi
>
> **备注:** This version includes an updated literature review, added acknowledgements, and a revised author list
>
> **摘要:** We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety.
>
---
#### [replaced 040] The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19797v3](http://arxiv.org/pdf/2505.19797v3)**

> **作者:** Yiqun Zhang; Hao Li; Chenxu Wang; Linyao Chen; Qiaosheng Zhang; Peng Ye; Shi Feng; Daling Wang; Zhen Wang; Xinrun Wang; Jia Xu; Lei Bai; Wanli Ouyang; Shuyue Hu
>
> **备注:** 9 pages, 4 figures, 6 tables, supplementary material (appendix) included separately
>
> **摘要:** Proprietary giants are increasingly dominating the race for ever-larger language models. Can open-source, smaller models remain competitive across a broad range of tasks? In this paper, we present the Avengers -- a simple recipe that leverages the collective intelligence of these smaller models. The Avengers builds upon four lightweight operations: (i) embedding: encode queries using a text embedding model; (ii) clustering: group queries based on their semantic similarity; (iii) scoring: scores each model's performance within each cluster; and (iv) voting: improve outputs via repeated sampling and voting. At inference time, each query is embedded and assigned to its nearest cluster. The top-performing model(s) within that cluster are selected to generate the response with repeated sampling. Remarkably, with 10 open-source models (~7B parameters each), the Avengers surpasses GPT-4o, 4.1, and 4.5 in average performance across 15 diverse datasets spanning mathematics, coding, logical reasoning, general knowledge, and affective tasks. In particular, it surpasses GPT-4.1 on mathematics tasks by 18.21% and on code tasks by 7.46%. Furthermore, the Avengers delivers superior out-of-distribution generalization, and remains robust across various embedding models, clustering algorithms, ensemble strategies, and values of its sole parameter -- the number of clusters.
>
---
#### [replaced 041] Can LLMs Ask Good Questions?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.03491v2](http://arxiv.org/pdf/2501.03491v2)**

> **作者:** Yueheng Zhang; Xiaoyuan Liu; Yiyou Sun; Atheer Alharbi; Hend Alzahrani; Tianneng Shi; Basel Alomair; Dawn Song
>
> **摘要:** We evaluate questions generated by large language models (LLMs) from context, comparing them to human-authored questions across six dimensions: question type, question length, context coverage, answerability, uncommonness, and required answer length. Our study spans two open-source and two proprietary state-of-the-art models. Results reveal that LLM-generated questions tend to demand longer descriptive answers and exhibit more evenly distributed context focus, in contrast to the positional bias often seen in QA tasks. These findings provide insights into the distinctive characteristics of LLM-generated questions and inform future work on question quality and downstream applications.
>
---
#### [replaced 042] Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08343v2](http://arxiv.org/pdf/2506.08343v2)**

> **作者:** Chenlong Wang; Yuanning Feng; Dongping Chen; Zhaoyang Chu; Ranjay Krishna; Tianyi Zhou
>
> **摘要:** Recent advances in large reasoning models have enabled complex, step-by-step reasoning but often introduce significant overthinking, resulting in verbose and redundant outputs that hinder efficiency. In this study, we examine whether explicit self-reflection, signaled by tokens such as "Wait" and "Hmm", is necessary for advanced reasoning. We propose NoWait, a simple yet effective approach that disables explicit self-reflection by suppressing these tokens during inference. Extensive experiments on ten benchmarks across textual, visual, and video reasoning tasks show that NoWait reduces chain-of-thought trajectory length by up to 27%-51% in five R1-style model series, without compromising model utility. NoWait thus offers a plug-and-play solution for efficient and utility-preserving multimodal reasoning.
>
---
#### [replaced 043] Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16065v2](http://arxiv.org/pdf/2505.16065v2)**

> **作者:** Ruijie Xi; He Ba; Hao Yuan; Rishu Agrawal; Yuxin Tian; Ruoyan Long; Arul Prakash
>
> **摘要:** Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data.
>
---
#### [replaced 044] Root Defence Strategies: Ensuring Safety of LLM at the Decoding Level
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2410.06809v3](http://arxiv.org/pdf/2410.06809v3)**

> **作者:** Xinyi Zeng; Yuying Shang; Jiawei Chen; Jingyuan Zhang; Yu Tian
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Large language models (LLMs) have demonstrated immense utility across various industries. However, as LLMs advance, the risk of harmful outputs increases due to incorrect or malicious instruction prompts. While current methods effectively address jailbreak risks, they share common limitations: 1) Judging harmful responses from the prefill-level lacks utilization of the model's decoding outputs, leading to relatively lower effectiveness and robustness. 2) Rejecting potentially harmful responses based on a single evaluation can significantly impair the model's helpfulness.This paper examines the LLMs' capability to recognize harmful outputs, revealing and quantifying their proficiency in assessing the danger of previous tokens. Motivated by pilot experiment results, we design a robust defense mechanism at the decoding level. Our novel decoder-oriented, step-by-step defense architecture corrects harmful queries directly rather than rejecting them outright. We introduce speculative decoding to enhance usability and facilitate deployment to boost secure decoding speed. Extensive experiments demonstrate that our approach improves model security without compromising reasoning speed. Notably, our method leverages the model's ability to discern hazardous information, maintaining its helpfulness compared to existing methods.
>
---
#### [replaced 045] LLäMmlein: Transparent, Compact and Competitive German-Only Language Models from Scratch
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.11171v5](http://arxiv.org/pdf/2411.11171v5)**

> **作者:** Jan Pfister; Julia Wunderle; Andreas Hotho
>
> **备注:** camera ready @ACL25; https://www.informatik.uni-wuerzburg.de/datascience/projects/nlp/llammlein/
>
> **摘要:** We create two German-only decoder models, LL\"aMmlein 120M and 1B, transparently from scratch and publish them, along with the training data, for the German NLP research community to use. The model training involved several key steps, including extensive data preprocessing, the creation of a custom German tokenizer, the training itself, as well as the evaluation of the final models on various benchmarks. Throughout the training process, multiple checkpoints were saved and analyzed using the SuperGLEBer benchmark to monitor the models' learning dynamics. Compared to state-of-the-art models on the SuperGLEBer benchmark, both LL\"aMmlein models performed competitively, consistently matching or surpassing models with similar parameter sizes. The results show that the models' quality scales with size as expected, but performance improvements on some tasks plateaued early, offering valuable insights into resource allocation for future model development.
>
---
#### [replaced 046] GRAM: A Generative Foundation Reward Model for Reward Generalization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.14175v2](http://arxiv.org/pdf/2506.14175v2)**

> **作者:** Chenglong Wang; Yang Gan; Yifu Huo; Yongyu Mu; Qiaozhi He; Murun Yang; Bei Li; Tong Xiao; Chunliang Zhang; Tongran Liu; Jingbo Zhu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** In aligning large language models (LLMs), reward models have played an important role, but are standardly trained as discriminative models and rely only on labeled human preference data. In this paper, we explore methods that train reward models using both unlabeled and labeled data. Building on the generative models in LLMs, we develop a generative reward model that is first trained via large-scale unsupervised learning and then fine-tuned via supervised learning. We also show that by using label smoothing, we are in fact optimizing a regularized pairwise ranking loss. This result, in turn, provides a new view of training reward models, which links generative models and discriminative models under the same class of training objectives. The outcome of these techniques is a foundation reward model, which can be applied to a wide range of tasks with little or no further fine-tuning effort. Extensive experiments show that this model generalizes well across several tasks, including response ranking, reinforcement learning from human feedback, and task adaptation with fine-tuning, achieving significant performance improvements over several strong baseline models.
>
---
#### [replaced 047] Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10912v2](http://arxiv.org/pdf/2506.10912v2)**

> **作者:** Fei Lin; Ziyang Gong; Cong Wang; Yonglin Tian; Tengchao Zhang; Xue Yang; Gen Luo; Fei-Yue Wang
>
> **摘要:** Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair - generating structurally valid molecular alternatives with reduced toxicity - has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 560 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a high-throughput evaluation chain for repair success. We systematically assess nearly 30 mainstream general-purpose MLLMs and design multiple ablation studies to analyze key factors such as evaluation criteria, candidate diversity, and failure attribution. Experimental results show that although current MLLMs still face significant challenges on this task, they begin to demonstrate promising capabilities in toxicity understanding, semantic constraint adherence, and structure-aware molecule editing.
>
---
#### [replaced 048] GreekBarBench: A Challenging Benchmark for Free-Text Legal Reasoning and Citations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17267v2](http://arxiv.org/pdf/2505.17267v2)**

> **作者:** Odysseas S. Chlapanis; Dimitrios Galanis; Nikolaos Aletras; Ion Androutsopoulos
>
> **备注:** 19 pages, 17 figures, submitted to May ARR
>
> **摘要:** We introduce GreekBarBench, a benchmark that evaluates LLMs on legal questions across five different legal areas from the Greek Bar exams, requiring citations to statutory articles and case facts. To tackle the challenges of free-text evaluation, we propose a three-dimensional scoring system combined with an LLM-as-a-judge approach. We also develop a meta-evaluation benchmark to assess the correlation between LLM-judges and human expert evaluations, revealing that simple, span-based rubrics improve their alignment. Our systematic evaluation of 13 proprietary and open-weight LLMs shows that even though the best models outperform average expert scores, they fall short of the 95th percentile of experts.
>
---
#### [replaced 049] Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.07009v3](http://arxiv.org/pdf/2410.07009v3)**

> **作者:** Valentin Knappich; Simon Razniewski; Anna Hätty; Annemarie Friedrich
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Dealing with long and highly complex technical text is a challenge for Large Language Models (LLMs), which still have to unfold their potential in supporting expensive and timeintensive processes like patent drafting. Within patents, the description constitutes more than 90% of the document on average. Yet, its automatic generation remains understudied. When drafting patent applications, patent attorneys typically receive invention reports (IRs), which are usually confidential, hindering research on LLM-supported patent drafting. Often, prepublication research papers serve as IRs. We leverage this duality to build PAP2PAT, an open and realistic benchmark for patent drafting consisting of 1.8k patent-paper pairs describing the same inventions. To address the complex longdocument patent generation task, we propose chunk-based outline-guided generation using the research paper as invention specification. Our extensive evaluation using PAP2PAT and a human case study show that LLMs can effectively leverage information from the paper, but still struggle to provide the necessary level of detail. Fine-tuning leads to more patent-style language, but also to more hallucination. We release our data and code https://github.com/boschresearch/Pap2Pat.
>
---
#### [replaced 050] TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks
- **分类: cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.07890v4](http://arxiv.org/pdf/2505.07890v4)**

> **作者:** Kutay Ertürk; Furkan Altınışık; İrem Sarıaltın; Ömer Nezih Gerek
>
> **摘要:** This study presents TSLFormer, a light and robust word-level Turkish Sign Language (TSL) recognition model that treats sign gestures as ordered, string-like language. Instead of using raw RGB or depth videos, our method only works with 3D joint positions - articulation points - extracted using Google's Mediapipe library, which focuses on the hand and torso skeletal locations. This creates efficient input dimensionality reduction while preserving important semantic gesture information. Our approach revisits sign language recognition as sequence-to-sequence translation, inspired by the linguistic nature of sign languages and the success of transformers in natural language processing. Since TSLFormer uses the self-attention mechanism, it effectively captures temporal co-occurrence within gesture sequences and highlights meaningful motion patterns as words unfold. Evaluated on the AUTSL dataset with over 36,000 samples and 227 different words, TSLFormer achieves competitive performance with minimal computational cost. These results show that joint-based input is sufficient for enabling real-time, mobile, and assistive communication systems for hearing-impaired individuals.
>
---
#### [replaced 051] PsychBench: A comprehensive and professional benchmark for evaluating the performance of LLM-assisted psychiatric clinical practice
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.01903v2](http://arxiv.org/pdf/2503.01903v2)**

> **作者:** Shuyu Liu; Ruoxi Wang; Ling Zhang; Xuequan Zhu; Rui Yang; Xinzhu Zhou; Fei Wu; Zhi Yang; Cheng Jin; Gang Wang
>
> **摘要:** The advent of Large Language Models (LLMs) offers potential solutions to address problems such as shortage of medical resources and low diagnostic consistency in psychiatric clinical practice. Despite this potential, a robust and comprehensive benchmarking framework to assess the efficacy of LLMs in authentic psychiatric clinical environments is absent. This has impeded the advancement of specialized LLMs tailored to psychiatric applications. In response to this gap, by incorporating clinical demands in psychiatry and clinical data, we proposed a benchmarking system, PsychBench, to evaluate the practical performance of LLMs in psychiatric clinical settings. We conducted a comprehensive quantitative evaluation of 16 LLMs using PsychBench, and investigated the impact of prompt design, chain-of-thought reasoning, input text length, and domain-specific knowledge fine-tuning on model performance. Through detailed error analysis, we identified strengths and potential limitations of the existing models and suggested directions for improvement. Subsequently, a clinical reader study involving 60 psychiatrists of varying seniority was conducted to further explore the practical benefits of existing LLMs as supportive tools for psychiatrists of varying seniority. Through the quantitative and reader evaluation, we show that while existing models demonstrate significant potential, they are not yet adequate as decision-making tools in psychiatric clinical practice. The reader study further indicates that, as an auxiliary tool, LLM could provide particularly notable support for junior psychiatrists, effectively enhancing their work efficiency and overall clinical quality. To promote research in this area, we will make the dataset and evaluation framework publicly available, with the hope of advancing the application of LLMs in psychiatric clinical settings.
>
---
#### [replaced 052] Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09099v2](http://arxiv.org/pdf/2506.09099v2)**

> **作者:** Joshua Barron; Devin White
>
> **备注:** Accepted for oral presentation to Tiny Titans: The next wave of On-Device Learning for Foundational Models Workshop at the 42nd International Conference on Machine Learning
>
> **摘要:** The relationship between memorization and generalization in large language models (LLMs) remains an open area of research, with growing evidence that the two are deeply intertwined. In this work, we investigate this relationship by pre-training a series of capacity-limited Transformer models from scratch on two synthetic character-level tasks designed to separately probe generalization (via arithmetic extrapolation) and memorization (via factual recall). We observe a consistent trade-off: small models extrapolate to unseen arithmetic cases but fail to memorize facts, while larger models memorize but fail to extrapolate. An intermediate-capacity model exhibits a similar shift toward memorization. When trained on both tasks jointly, no model (regardless of size) succeeds at extrapolation. These findings suggest that pre-training may intrinsically favor one learning mode over the other. By isolating these dynamics in a controlled setting, our study offers insight into how model capacity shapes learning behavior and offers broader implications for the design and deployment of small language models.
>
---
#### [replaced 053] J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13346v3](http://arxiv.org/pdf/2505.13346v3)**

> **作者:** Austin Xu; Yilun Zhou; Xuan-Phi Nguyen; Caiming Xiong; Shafiq Joty
>
> **备注:** 25 pages, 4 figures, 6 tables. Updated with code and benchmark
>
> **摘要:** To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench.
>
---
#### [replaced 054] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.18043v2](http://arxiv.org/pdf/2412.18043v2)**

> **作者:** Yidong Gan; Maciej Rybinski; Ben Hachey; Jonathan K. Kummerfeld
>
> **备注:** Accepted to the ACL 2025 Main Conference
>
> **摘要:** Clinical coding is crucial for healthcare billing and data analysis. Manual clinical coding is labour-intensive and error-prone, which has motivated research towards full automation of the process. However, our analysis, based on US English electronic health records and automated coding research using these records, shows that widely used evaluation methods are not aligned with real clinical contexts. For example, evaluations that focus on the top 50 most common codes are an oversimplification, as there are thousands of codes used in practice. This position paper aims to align AI coding research more closely with practical challenges of clinical coding. Based on our analysis, we offer eight specific recommendations, suggesting ways to improve current evaluation methods. Additionally, we propose new AI-based methods beyond automated coding, suggesting alternative approaches to assist clinical coders in their workflows.
>
---
#### [replaced 055] Efficiently Building a Domain-Specific Large Language Model from Scratch: A Case Study of a Classical Chinese Large Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11810v3](http://arxiv.org/pdf/2505.11810v3)**

> **作者:** Shen Li; Renfen Hu; Lijun Wang
>
> **摘要:** General-purpose large language models demonstrate notable capabilities in language comprehension and generation, achieving results that are comparable to, or even surpass, human performance in many natural language processing tasks. Nevertheless, when general models are applied to some specific domains, e.g., Classical Chinese texts, their effectiveness is often unsatisfactory, and fine-tuning open-source foundational models similarly struggles to adequately incorporate domain-specific knowledge. To address this challenge, this study developed a large language model, AI Taiyan, specifically designed for understanding and generating Classical Chinese. Experiments show that with a reasonable model design, data processing, foundational training, and fine-tuning, satisfactory results can be achieved with only 1.8 billion parameters. In key tasks related to language processing of Classical Chinese such as punctuation, identification of allusions, explanation of word meanings, and translation between ancient and modern Chinese, this model exhibits a clear advantage over both general-purpose large models and domain-specific traditional models, achieving levels close to or surpassing human baselines. This research provides a reference for the efficient construction of specialized domain-specific large language models. Furthermore, the paper discusses the application of this model in fields such as the collation of ancient texts, dictionary editing, and language research, combined with case studies.
>
---
#### [replaced 056] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13366v3](http://arxiv.org/pdf/2506.13366v3)**

> **作者:** Didi Zhang; Yaxin Fan; Peifeng Li; Qiaoming Zhu
>
> **备注:** Accepted by ACL'25 (main conference)
>
> **摘要:** Goal-oriented proactive dialogue systems are designed to guide user conversations seamlessly towards specific objectives by planning a goal-oriented path. However, previous research has focused predominantly on optimizing these paths while neglecting the inconsistencies that may arise between generated responses and dialogue contexts, including user profiles, dialogue history, domain knowledge, and subgoals. To address this issue, we introduce a model-agnostic two-stage Consistency Reflection and Correction (CRC) framework. Specifically, in the consistency reflection stage, the model is prompted to reflect on the discrepancies between generated responses and dialogue contexts, identifying inconsistencies and suggesting possible corrections. In the consistency correction stage, the model generates responses that are more consistent with the dialogue context based on these reflection results. We conducted experiments on various model architectures with different parameter sizes, including encoder-decoder models (BART, T5) and decoder-only models (GPT-2, DialoGPT, Phi3, Mistral and LLaMA3), and the experimental results on three datasets demonstrate that our CRC framework significantly improves the consistency between generated responses and dialogue contexts.
>
---
#### [replaced 057] PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21342v3](http://arxiv.org/pdf/2505.21342v3)**

> **作者:** Valentin Knappich; Annemarie Friedrich; Anna Hätty; Simon Razniewski
>
> **备注:** PatentSemTech@SIGIR2025
>
> **摘要:** Patent claims define the scope of protection for an invention. If there are ambiguities in a claim, it is rejected by the patent office. In the US, this is referred to as indefiniteness (35 U.S.C {\S} 112(b)) and is among the most frequent reasons for patent application rejection. The development of automatic methods for patent definiteness examination has the potential to make patent drafting and examination more efficient, but no annotated dataset has been published to date. We introduce PEDANTIC (Patent Definiteness Examination Corpus), a novel dataset of 14k US patent claims from patent applications relating to Natural Language Processing (NLP), annotated with reasons for indefiniteness. We construct PEDANTIC using a fully automatic pipeline that retrieves office action documents from the USPTO and uses Large Language Models (LLMs) to extract the reasons for indefiniteness. A human validation study confirms the pipeline's accuracy in generating high-quality annotations. To gain insight beyond binary classification metrics, we implement an LLM-as-Judge evaluation that compares the free-form reasoning of every model-cited reason with every examiner-cited reason. We show that LLM agents based on Qwen 2.5 32B and 72B struggle to outperform logistic regression baselines on definiteness prediction, even though they often correctly identify the underlying reasons. PEDANTIC provides a valuable resource for patent AI researchers, enabling the development of advanced examination models. We will publicly release the dataset and code.
>
---
