# 自然语言处理 cs.CL

- **最新发布 126 篇**

- **更新 93 篇**

## 最新发布

#### [new 001] A Single Character can Make or Break Your LLM Evals
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了大语言模型（LLM）评估中示例分隔符的选择对模型表现的影响。任务是探究不同分隔符（如逗号、换行等）如何影响模型输出。论文发现，分隔符选择会显著影响评估结果，甚至改变模型排名。通过分析注意力机制，发现高效分隔符能引导模型关注关键信息。最后提出增强模型鲁棒性的方法。**

- **链接: [http://arxiv.org/pdf/2510.05152v1](http://arxiv.org/pdf/2510.05152v1)**

> **作者:** Jingtong Su; Jianyu Zhang; Karen Ullrich; Léon Bottou; Mark Ibrahim
>
> **摘要:** Common Large Language model (LLM) evaluations rely on demonstration examples to steer models' responses to the desired style. While the number of examples used has been studied and standardized, the choice of how to format examples is less investigated. In evaluation protocols and real world usage, users face the choice how to separate in-context examples: use a comma? new line? semi-colon? hashtag? etc.? Surprisingly, we find this seemingly minor choice can dramatically alter model response quality. Across leading model families (Llama, Qwen, Gemma), performance on MMLU for example can vary by $\pm 23\%$ depending on the choice of delimiter. In fact, one can manipulate model rankings to put any model in the lead by only modifying the single character separating examples. We find LLMs' brittleness pervades topics, model families, and doesn't improve with scale. By probing attention head scores, we find that good-performing delimiters steer attention towards key tokens in the input. Finally, we explore methods to improve LLMs' robustness to the choice of delimiter. We find specifying the selected delimiter in the prompt boosts robustness and offer practical recommendations for the best-performing delimiters to select.
>
---
#### [new 002] CDTP: A Large-Scale Chinese Data-Text Pair Dataset for Comprehensive Evaluation of Chinese LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决中文大语言模型（LLMs）缺乏结构化数据和针对性评估的问题。作者构建了包含700万中文文本对和1500万三元组的CDTP数据集，覆盖四个领域，支持知识驱动任务的评估与多任务微调，提出了CB-ECLLM基准，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.06039v1](http://arxiv.org/pdf/2510.06039v1)**

> **作者:** Chengwei Wu; Jiapu Wang; Mingyang Gao; Xingrui Zhuo; Jipeng Guo; Runlin Lei; Haoran Luo; Tianyu Chen; Haoyi Zhou; Shirui Pan; Zechao Li
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, Chinese LLMs face unique challenges, primarily due to the dominance of unstructured free text and the lack of structured representations in Chinese corpora. While existing benchmarks for LLMs partially assess Chinese LLMs, they are still predominantly English-centric and fail to address the unique linguistic characteristics of Chinese, lacking structured datasets essential for robust evaluation. To address these challenges, we present a Comprehensive Benchmark for Evaluating Chinese Large Language Models (CB-ECLLM) based on the newly constructed Chinese Data-Text Pair (CDTP) dataset. Specifically, CDTP comprises over 7 million aligned text pairs, each consisting of unstructured text coupled with one or more corresponding triples, alongside a total of 15 million triples spanning four critical domains. The core contributions of CDTP are threefold: (i) enriching Chinese corpora with high-quality structured information; (ii) enabling fine-grained evaluation tailored to knowledge-driven tasks; and (iii) supporting multi-task fine-tuning to assess generalization and robustness across scenarios, including Knowledge Graph Completion, Triple-to-Text generation, and Question Answering. Furthermore, we conduct rigorous evaluations through extensive experiments and ablation studies to assess the effectiveness, Supervised Fine-Tuning (SFT), and robustness of the benchmark. To support reproducible research, we offer an open-source codebase and outline potential directions for future investigations based on our insights.
>
---
#### [new 003] EvalMORAAL: Interpretable Chain-of-Thought and LLM-as-Judge Evaluation for Moral Alignment in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出了EvalMORAAL框架，用于评估大语言模型在不同文化背景下的道德对齐情况。该任务属于AI伦理与文化适应性评估。为解决模型在不同区域道德观念一致性不足的问题，作者结合推理链与模型评审机制，在多国调查数据上进行评测，发现并分析了区域偏差问题。**

- **链接: [http://arxiv.org/pdf/2510.05942v1](http://arxiv.org/pdf/2510.05942v1)**

> **作者:** Hadi Mohammadi; Anastasia Giachanou; Ayoub Bagheri
>
> **摘要:** We present EvalMORAAL, a transparent chain-of-thought (CoT) framework that uses two scoring methods (log-probabilities and direct ratings) plus a model-as-judge peer review to evaluate moral alignment in 20 large language models. We assess models on the World Values Survey (55 countries, 19 topics) and the PEW Global Attitudes Survey (39 countries, 8 topics). With EvalMORAAL, top models align closely with survey responses (Pearson's r approximately 0.90 on WVS). Yet we find a clear regional difference: Western regions average r=0.82 while non-Western regions average r=0.61 (a 0.21 absolute gap), indicating consistent regional bias. Our framework adds three parts: (1) two scoring methods for all models to enable fair comparison, (2) a structured chain-of-thought protocol with self-consistency checks, and (3) a model-as-judge peer review that flags 348 conflicts using a data-driven threshold. Peer agreement relates to survey alignment (WVS r=0.74, PEW r=0.39, both p<.001), supporting automated quality checks. These results show real progress toward culture-aware AI while highlighting open challenges for use across regions.
>
---
#### [new 004] RECODE-H: A Benchmark for Research Code Development with Interactive Human Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RECODE-H基准，用于评估结合交互式人类反馈的研究代码生成。旨在解决LLM生成科研代码的准确性与可执行性不足的问题，通过多轮反馈模拟真实科研协作。论文属于代码生成与评估任务。**

- **链接: [http://arxiv.org/pdf/2510.06186v1](http://arxiv.org/pdf/2510.06186v1)**

> **作者:** Chunyu Miao; Henry Peng Zou; Yangning Li; Yankai Chen; Yibo Wang; Fangxin Wang; Yifan Li; Wooseong Yang; Bowei He; Xinni Zhang; Dianzhi Yu; Hanchen Yang; Hoang H Nguyen; Yue Zhou; Jie Yang; Jizhou Guo; Wenzhe Fan; Chin-Yuan Yeh; Panpan Meng; Liancheng Fang; Jinhu Qi; Wei-Chieh Huang; Zhengyao Gu; Yuwei Han; Langzhou He; Yuyao Yang; Xue Liu; Irwin King; Philip S. Yu
>
> **备注:** Code and dataset are available at github.com/ChunyuMiao98/RECODE
>
> **摘要:** Large language models (LLMs) show the promise in supporting scientific research implementation, yet their ability to generate correct and executable code remains limited. Existing works largely adopt one-shot settings, ignoring the iterative and feedback-driven nature of realistic workflows of scientific research development. To address this gap, we present RECODE-H, a benchmark of 102 tasks from research papers and repositories that evaluates LLM agents through multi-turn interactions with LLM-simulated human feedback. It includes structured instructions,unit tests, and a five-level feedback hierarchy to reflect realistic researcher-agent collaboration. We further present ReCodeAgent, a framework that integrates feedback into iterative code generation. Experiments with leading LLMs, including GPT-5, Claude-Sonnet-4, DeepSeek-V3.1, and Gemini 2.5, show substantial performance gains with richer feedback, while also highlighting ongoing challenges in the generation of complex research code. RECODE-H establishes a foundation for developing adaptive, feedback-driven LLM agents in scientific research implementation
>
---
#### [new 005] Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations
- **分类: cs.CL**

- **简介: 该论文属于学术展示生成任务，旨在解决自动化展示工具在叙事、美学设计和自我调整上的不足。论文提出了EvoPresent框架及其核心模型PresAesth，通过多任务强化学习提升幻灯片的美学评分与改进能力，并构建了相关评估基准，验证了高质量反馈和多任务训练对自我改进的重要性。**

- **链接: [http://arxiv.org/pdf/2510.05571v1](http://arxiv.org/pdf/2510.05571v1)**

> **作者:** Chengzhi Liu; Yuzhe Yang; Kaiwen Zhou; Zhen Zhang; Yue Fan; Yannan Xie; Peng Qi; Xin Eric Wang
>
> **摘要:** The promotion of academic papers has become an important means of enhancing research visibility. However, existing automated methods struggle limited storytelling, insufficient aesthetic quality, and constrained self-adjustment, making it difficult to achieve efficient and engaging dissemination. At the heart of those challenges is a simple principle: \emph{there is no way to improve it when you cannot evaluate it right}. To address this, we introduce \textbf{EvoPresent}, a self-improvement agent framework that unifies coherent narratives, aesthetic-aware designs, and realistic presentation delivery via virtual characters. Central to EvoPresent is \textbf{PresAesth}, a multi-task reinforcement learning (RL) aesthetic model that provides reliable aesthetic scoring, defect adjustment, and comparative feedback, enabling iterative self-improvement even under limited aesthetic training data. To systematically evaluate the methods, we introduce \textbf{EvoPresent Benchmark}, a comprehensive benchmark comprising: \textit{Presentation Generation Quality}, built on 650 top-tier AI conference papers with multimodal resources (slides, videos and scripts) to assess both content and design; and \textit{Aesthetic Awareness}, consisting of 2,000 slide pairs with varying aesthetic levels, supporting joint training and evaluation on scoring, defect adjustment, and comparison. Our findings highlight that (i) High-quality feedback is essential for agent self-improvement, while initial capability alone does not guarantee effective self-correction. (ii) Automated generation pipelines exhibit a trade-off between visual design and content construction. (iii) Multi-task RL training shows stronger generalization in aesthetic awareness tasks.
>
---
#### [new 006] LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的知识蒸馏任务，旨在解决大型语言模型在招聘匹配与解释中的部署难题。论文提出了LANTERN框架，通过多目标建模与多级知识蒸馏，提升任务性能与可解释性，并优化模型规模以增强在线服务的可行性与效率。**

- **链接: [http://arxiv.org/pdf/2510.05490v1](http://arxiv.org/pdf/2510.05490v1)**

> **作者:** Zhoutong Fu; Yihan Cao; Yi-Lin Chen; Aman Lunia; Liming Dong; Neha Saraf; Ruijie Jiang; Yun Dai; Qingquan Song; Tan Wang; Guoyao Li; Derek Koh; Haichao Wei; Zhipeng Wang; Aman Gupta; Chengming Jiang; Jianqiang Shen; Liangjie Hong; Wenjing Zhang
>
> **备注:** 9 pages, 4 figures, 5 tables
>
> **摘要:** Large language models (LLMs) have achieved strong performance across a wide range of natural language processing tasks. However, deploying LLMs at scale for domain specific applications, such as job-person fit and explanation in job seeking platforms, introduces distinct challenges. At LinkedIn, the job person fit task requires analyzing a candidate's public profile against job requirements to produce both a fit assessment and a detailed explanation. Directly applying open source or finetuned LLMs to this task often fails to yield high quality, actionable feedback due to the complexity of the domain and the need for structured outputs. Moreover, the large size of these models leads to high inference latency and limits scalability, making them unsuitable for online use. To address these challenges, we introduce LANTERN, a novel LLM knowledge distillation framework tailored specifically for job person fit tasks. LANTERN involves modeling over multiple objectives, an encoder model for classification purpose, and a decoder model for explanation purpose. To better distill the knowledge from a strong black box teacher model to multiple downstream models, LANTERN incorporates multi level knowledge distillation that integrates both data and logit level insights. In addition to introducing the knowledge distillation framework, we share our insights on post training techniques and prompt engineering, both of which are crucial for successfully adapting LLMs to domain specific downstream tasks. Extensive experimental results demonstrate that LANTERN significantly improves task specific metrics for both job person fit and explanation. Online evaluations further confirm its effectiveness, showing measurable gains in job seeker engagement, including a 0.24\% increase in apply rate and a 0.28\% increase in qualified applications.
>
---
#### [new 007] Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型（LLM）和视觉语言模型（VLM）部署中的高内存和计算需求问题。论文提出了一种基于激活感知的低秩压缩框架PGSVD，通过理论分析建立损失与压缩误差的关系，并利用帕累托优化选择最优秩，实现高效的模型压缩，在保持精度的同时提升了推理速度。**

- **链接: [http://arxiv.org/pdf/2510.05544v1](http://arxiv.org/pdf/2510.05544v1)**

> **作者:** Ryan Solgi; Parsa Madinei; Jiayi Tian; Rupak Swaminathan; Jing Liu; Nathan Susanj; Zheng Zhang
>
> **摘要:** Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
>
---
#### [new 008] DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision
- **分类: cs.CL**

- **简介: 论文提出DecEx-RAG，属于检索增强生成（RAG）任务，旨在解决现有方法在探索效率、稀疏奖励信号和模糊全局反馈上的问题。通过建模为马尔可夫决策过程并引入剪枝策略，优化任务分解与检索生成能力。实验表明性能平均提升6.2%，数据构建效率提高6倍。**

- **链接: [http://arxiv.org/pdf/2510.05691v1](http://arxiv.org/pdf/2510.05691v1)**

> **作者:** Yongqi Leng; Yikun Lei; Xikai Liu; Meizhi Zhong; Bojian Xiong; Yurong Zhang; Yan Gao; Yi Wu; Yao Hu; Deyi Xiong
>
> **摘要:** Agentic Retrieval-Augmented Generation (Agentic RAG) enhances the processing capability for complex tasks through dynamic retrieval and adaptive workflows. Recent advances (e.g., Search-R1) have shown that outcome-supervised reinforcement learning demonstrate strong performance. However, this approach still suffers from inefficient exploration, sparse reward signals, and ambiguous global reward feedback. To address these challenges, we propose DecEx-RAG, which models RAG as a Markov Decision Process (MDP) incorporating decision-making and execution, while introducing an efficient pruning strategy to optimize data expansion. Through comprehensive process-level policy optimization, DecEx-RAG significantly enhances the autonomous task decomposition, dynamic retrieval, and high-quality answer generation capabilities of large language models (LLMs). Experiments show that DecEx-RAG achieves an average absolute performance improvement of $6.2\%$ across six datasets, significantly outperforming existing baselines. Moreover, the pruning strategy improves data construction efficiency by nearly $6 \times$, providing an efficient solution for process-supervised RAG training. The code is available at https://github.com/sdsxdxl/DecEx-RAG.
>
---
#### [new 009] NLD-LLM: A systematic framework for evaluating small language transformer models on natural language description
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估小型语言模型生成代码描述的能力。为解决评估不一致的问题，作者构建了NLD-LLM框架，采用多样化的模型和提示策略，通过语义与结构指标验证提示工程对模型性能的提升作用。**

- **链接: [http://arxiv.org/pdf/2510.05139v1](http://arxiv.org/pdf/2510.05139v1)**

> **作者:** Hamed Jelodar; Mohammad Meymani; Parisa Hamedi; Tochukwu Emmanuel Nwankwo; Samita Bai; Roozbeh Razavi-Far; Ali A. Ghorbani
>
> **摘要:** Natural Language Description (NLD) is a Natural Language Processing (NLP) task that requires models to generate structured and meaningful outputs from natural language inputs. In this work, we propose NLD-LLM, a systematic NLP framework to evaluate the performance of language models to generate accurate and concise source code descriptions. This framework incorporates a diverse set of transformer models, including Qwen, DeepSeek, Phi, LLaMA, and Mistral, spanning various sizes, architectures, and training approaches. Central to NLD-LLM is a comprehensive prompt design strategy that includes standardized formatting, clear task guidance, and NLD prompting, ensuring fair and consistent evaluation. Additionally, we apply an iterative refinement process to improve output's quality and assess the model's adaptability. Using semantic and structural metrics, our analysis demonstrates that prompt engineering significantly impacts the effectiveness of the model such that smaller models often performing competitively when supported by well-crafted prompts.
>
---
#### [new 010] CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长文本阅读理解中的记忆不足问题。受皮亚杰建构主义理论启发，提出CAM模型，具备结构化、灵活适应和动态调整的内存机制，通过增量聚类算法提升记忆组织与检索效率，增强模型在问答、摘要和主张验证等任务中的表现。**

- **链接: [http://arxiv.org/pdf/2510.05520v1](http://arxiv.org/pdf/2510.05520v1)**

> **作者:** Rui Li; Zeyu Zhang; Xiaohe Bo; Zihang Tian; Xu Chen; Quanyu Dai; Zhenhua Dong; Ruiming Tang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification.
>
---
#### [new 011] EEPO: Exploration-Enhanced Policy Optimization via Sample-Then-Forget
- **分类: cs.CL**

- **简介: 该论文属于强化学习与大语言模型（LLM）任务，旨在解决RLVR中探索与利用失衡导致的熵崩溃与探索能力下降问题。作者提出EEPO框架，通过“采样-遗忘”机制实现两阶段 rollout，抑制主导行为模式，提升探索广度。实验表明EEPO在多个推理基准上优于GRPO。**

- **链接: [http://arxiv.org/pdf/2510.05837v1](http://arxiv.org/pdf/2510.05837v1)**

> **作者:** Liang Chen; Xueting Han; Qizhou Wang; Bo Han; Jing Bai; Hinrich Schutze; Kam-Fai Wong
>
> **摘要:** Balancing exploration and exploitation remains a central challenge in reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs). Current RLVR methods often overemphasize exploitation, leading to entropy collapse, diminished exploratory capacity, and ultimately limited performance gains. Although techniques that increase policy stochasticity can promote exploration, they frequently fail to escape dominant behavioral modes. This creates a self-reinforcing loop-repeatedly sampling and rewarding dominant modes-that further erodes exploration. We introduce Exploration-Enhanced Policy Optimization (EEPO), a framework that promotes exploration via two-stage rollouts with adaptive unlearning. In the first stage, the model generates half of the trajectories; it then undergoes a lightweight unlearning step to temporarily suppress these sampled responses, forcing the second stage to explore different regions of the output space. This sample-then-forget mechanism disrupts the self-reinforcing loop and promotes wider exploration during rollouts. Across five reasoning benchmarks, EEPO outperforms GRPO, achieving average relative gains of 24.3% on Qwen2.5-3B, 33.0% on Llama3.2-3B-Instruct, and 10.4% on Qwen3-8B-Base.
>
---
#### [new 012] Automated Alignment of Math Items to Content Standards in Large-Scale Assessments Using Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大规模评估中数学题目与内容标准的自动对齐问题。作者评估了三种方法：基于嵌入的传统机器学习模型、微调多种BERT模型、以及集成学习方法。最终发现，DeBERTa和RoBERTa在不同层级对齐任务中表现最佳，集成方法未能超越语言模型。**

- **链接: [http://arxiv.org/pdf/2510.05129v1](http://arxiv.org/pdf/2510.05129v1)**

> **作者:** Qingshu Xu; Hong Jiao; Tianyi Zhou; Ming Li; Nan Zhang; Sydney Peters; Yanbin Fu
>
> **摘要:** Accurate alignment of items to content standards is critical for valid score interpretation in large-scale assessments. This study evaluates three automated paradigms for aligning items with four domain and nineteen skill labels. First, we extracted embeddings and trained multiple classical supervised machine learning models, and further investigated the impact of dimensionality reduction on model performance. Second, we fine-tuned eight BERT model and its variants for both domain and skill alignment. Third, we explored ensemble learning with majority voting and stacking with multiple meta-models. The DeBERTa-v3-base achieved the highest weighted-average F1 score of 0.950 for domain alignment while the RoBERTa-large yielded the highest F1 score of 0.869 for skill alignment. Ensemble models did not surpass the best-performing language models. Dimension reduction enhanced linear classifiers based on embeddings but did not perform better than language models. This study demonstrated different methods in automated item alignment to content standards.}
>
---
#### [new 013] Luth: Efficient French Specialization for Small Language Models and Cross-Lingual Transfer
- **分类: cs.CL; I.2.7**

- **简介: 论文提出Luth，专注于法语的小型语言模型，旨在缩小现有大语言模型以英语为中心带来的法语性能差距。通过在高质量法语数据上进行针对性后训练，并结合模型融合策略，提升法语表现，同时保持英语能力，实现了跨语言迁移。**

- **链接: [http://arxiv.org/pdf/2510.05846v1](http://arxiv.org/pdf/2510.05846v1)**

> **作者:** Maxence Lasbordes; Sinoué Gad
>
> **备注:** 12 pages, 4 figures and 9 tables
>
> **摘要:** The landscape of Large Language Models (LLMs) remains predominantly English-centric, resulting in a significant performance gap for other major languages, such as French, especially in the context of Small Language Models (SLMs). Existing multilingual models demonstrate considerably lower performance in French compared to English, and research on efficient adaptation methods for French remains limited. To address this, we introduce \textbf{Luth}, a family of French-specialized SLMs: through targeted post-training on curated, high-quality French data, our models outperform all open-source counterparts of comparable size on multiple French benchmarks while retaining their original English capabilities. We further show that strategic model merging enhances performance in both languages, establishing Luth as a new state of the art for French SLMs and a robust baseline for future French-language research.
>
---
#### [new 014] Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs
- **分类: cs.CL**

- **简介: 该论文研究大规模公共讨论中意见汇总的公平性问题，旨在解决现有大语言模型（LLM）在生成政策用摘要时可能存在的少数观点代表性不足和顺序偏差问题。论文构建了包含意见数据和人工评分的DeliberationBank数据集，并训练了DeliberationJudge模型，提供更高效、更贴近人类判断的摘要评价方法，以提升AI在政策制定中的代表性和公平性。**

- **链接: [http://arxiv.org/pdf/2510.05154v1](http://arxiv.org/pdf/2510.05154v1)**

> **作者:** Shenzhe Zhu; Shu Yang; Michiel A. Bakker; Alex Pentland; Jiaxin Pei
>
> **摘要:** Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
>
---
#### [new 015] Reliable End-to-End Material Information Extraction from the Literature with Source-Tracked Multi-Stage Large Language Models
- **分类: cs.CL; cond-mat.mtrl-sci**

- **简介: 该论文属于材料信息抽取任务，旨在解决从文献中全面、准确提取材料成分、工艺、微观结构和性能关系的问题。作者提出了一种基于大语言模型的多阶段抽取流程，结合迭代提取与来源追踪，显著提升了抽取精度与完整性。**

- **链接: [http://arxiv.org/pdf/2510.05142v1](http://arxiv.org/pdf/2510.05142v1)**

> **作者:** Xin Wang; Anshu Raj; Matthew Luebbe; Haiming Wen; Shuozhi Xu; Kun Lu
>
> **备注:** 27 pages, 4 figures, 7 tables
>
> **摘要:** Data-driven materials discovery requires large-scale experimental datasets, yet most of the information remains trapped in unstructured literature. Existing extraction efforts often focus on a limited set of features and have not addressed the integrated composition-processing-microstructure-property relationships essential for understanding materials behavior, thereby posing challenges for building comprehensive databases. To address this gap, we propose a multi-stage information extraction pipeline powered by large language models, which captures 47 features spanning composition, processing, microstructure, and properties exclusively from experimentally reported materials. The pipeline integrates iterative extraction with source tracking to enhance both accuracy and reliability. Evaluations at the feature level (independent attributes) and tuple level (interdependent features) yielded F1 scores around 0.96. Compared with single-pass extraction without source tracking, our approach improved F1 scores of microstructure category by 10.0% (feature level) and 13.7% (tuple level), and reduced missed materials from 49 to 13 out of 396 materials in 100 articles on precipitate-containing multi-principal element alloys (miss rate reduced from 12.4% to 3.3%). The pipeline enables scalable and efficient literature mining, producing databases with high precision, minimal omissions, and zero false positives. These datasets provide trustworthy inputs for machine learning and materials informatics, while the modular design generalizes to diverse material classes, enabling comprehensive materials information extraction.
>
---
#### [new 016] The End of Transformers? On Challenging Attention and the Rise of Sub-Quadratic Architectures
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型因注意力机制的二次复杂度导致的长上下文处理瓶颈问题。论文综述了多种应对方案，包括子二次注意力机制、循环神经网络、状态空间模型及混合架构，并对其计算效率和性能进行分析评估。**

- **链接: [http://arxiv.org/pdf/2510.05364v1](http://arxiv.org/pdf/2510.05364v1)**

> **作者:** Alexander M. Fichtl; Jeremias Bohn; Josefin Kelber; Edoardo Mosca; Georg Groh
>
> **备注:** 21 pages, 2 figures, 2 tables
>
> **摘要:** Transformers have dominated sequence processing tasks for the past seven years -- most notably language modeling. However, the inherent quadratic complexity of their attention mechanism remains a significant bottleneck as context length increases. This paper surveys recent efforts to overcome this bottleneck, including advances in (sub-quadratic) attention variants, recurrent neural networks, state space models, and hybrid architectures. We critically analyze these approaches in terms of compute and memory complexity, benchmark results, and fundamental limitations to assess whether the dominance of pure-attention transformers may soon be challenged.
>
---
#### [new 017] Hire Your Anthropologist! Rethinking Culture Benchmarks Through an Anthropological Lens
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于文化评估任务，旨在解决当前大语言模型文化基准过于静态、简化的问题。作者提出四部分框架，分析20个文化基准，发现六类方法问题，并基于人类学方法提出改进建议，如引入真实情境、社区参与设计和情境化评估，以提升模型对复杂文化情境的理解能力。**

- **链接: [http://arxiv.org/pdf/2510.05931v1](http://arxiv.org/pdf/2510.05931v1)**

> **作者:** Mai AlKhamissi; Yunze Xiao; Badr AlKhamissi; Mona Diab
>
> **备注:** 12 pages; 2 figures; First two author contributed equally
>
> **摘要:** Cultural evaluation of large language models has become increasingly important, yet current benchmarks often reduce culture to static facts or homogeneous values. This view conflicts with anthropological accounts that emphasize culture as dynamic, historically situated, and enacted in practice. To analyze this gap, we introduce a four-part framework that categorizes how benchmarks frame culture, such as knowledge, preference, performance, or bias. Using this lens, we qualitatively examine 20 cultural benchmarks and identify six recurring methodological issues, including treating countries as cultures, overlooking within-culture diversity, and relying on oversimplified survey formats. Drawing on established anthropological methods, we propose concrete improvements: incorporating real-world narratives and scenarios, involving cultural communities in design and validation, and evaluating models in context rather than isolation. Our aim is to guide the development of cultural benchmarks that go beyond static recall tasks and more accurately capture the responses of the models to complex cultural situations.
>
---
#### [new 018] Code-Switching In-Context Learning for Cross-Lingual Transfer of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在跨语言推理时依赖英语表示导致的翻译障碍问题。作者提出代码切换上下文学习（CSICL），通过在提示中逐步从目标语言过渡到英语，引导模型在英语中进行推理，从而提升跨语言性能，特别是在低资源语言中的表现。**

- **链接: [http://arxiv.org/pdf/2510.05678v1](http://arxiv.org/pdf/2510.05678v1)**

> **作者:** Haneul Yoo; Jiho Jin; Kyunghyun Cho; Alice Oh
>
> **摘要:** While large language models (LLMs) exhibit strong multilingual abilities, their reliance on English as latent representations creates a translation barrier, where reasoning implicitly depends on internal translation into English. When this process fails, performance in non-English languages deteriorates sharply, limiting the inclusiveness of LLM-based applications. Existing cross-lingual in-context learning (X-ICL) methods primarily leverage monolingual demonstrations, often failing to mitigate this barrier and instead reinforcing it. In this work, we introduce code-switching in-context learning (CSICL), a simple yet effective prompting strategy that progressively transitions from a target language to English within demonstrations and instruction to facilitate their latent reasoning in English. By explicitly scaffolding the reasoning process through controlled code-switching, CSICL acts as an implicit linguistic bridge that enhances cross-lingual alignment and reduces reliance on the translation barrier. We conduct extensive experiments across 4 LLMs, 6 datasets, and 10 languages, spanning both knowledge-intensive and reasoning-oriented domains. Our results demonstrate that CSICL consistently outperforms X-ICL baselines, achieving gains of 3.1%p and 1.9%p in both target and unseen languages, respectively. The improvement is even more pronounced in low-resource settings, with gains of 14.7% in target and 5.3% in unseen languages. These findings establish code-switching as a principled and robust approach for overcoming the translation barrier during inference, moving LLMs toward more equitable and effective multilingual systems.
>
---
#### [new 019] A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks
- **分类: cs.CL**

- **简介: 该论文属于智能体任务规划领域，旨在解决大语言模型在长时程任务中缺乏全局规划导致的试错和幻觉问题。论文提出EAGLET方法，通过两步训练生成高质量计划，并结合强化学习提升规划能力，实现高效且无需人工干预的任务规划方案。**

- **链接: [http://arxiv.org/pdf/2510.05608v1](http://arxiv.org/pdf/2510.05608v1)**

> **作者:** Shuzheng Si; Haozhe Zhao; Kangyang Luo; Gang Chen; Fanchao Qi; Minjia Zhang; Baobao Chang; Maosong Sun
>
> **摘要:** Agents based on large language models (LLMs) struggle with brainless trial-and-error and generating hallucinatory actions due to a lack of global planning in long-horizon tasks. In this paper, we introduce a plan-and-execute framework and propose EAGLET, an efficient and effective planner training method to enhance the executor agent's planning abilities without human effort. Specifically, we train a plug-and-play global planner through a two-step process: we first synthesize high-quality plans from an advanced LLM using our proposed homologous consensus filtering strategy, and apply fine-tuning as a cold start. Moreover, we further improve the planner with a rule-based reinforcement learning stage using a novel executor capability gain reward, ensuring it can handle task instructions of varying difficulty. Experiments on three long-horizon agent tasks show that executor agents equipped with our planner outperform existing methods, achieving new state-of-the-art performance. Meanwhile, EAGLET reduces training costs by 8x compared to RL-based baselines, and it does not require manual effort or extra training data, offering an efficient and effective solution.
>
---
#### [new 020] Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context
- **分类: cs.CL**

- **简介: 该论文研究语言模型在上下文中绑定与检索实体的机制。任务是分析模型如何通过位置、词汇和反射机制检索绑定实体。论文发现位置机制在复杂场景下效果差，模型会混合三种机制进行补偿，并构建因果模型实现高准确率预测，验证了其在长文本中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.06182v1](http://arxiv.org/pdf/2510.06182v1)**

> **作者:** Yoav Gur-Arieh; Mor Geva; Atticus Geiger
>
> **摘要:** A key component of in-context reasoning is the ability of language models (LMs) to bind entities for later retrieval. For example, an LM might represent "Ann loves pie" by binding "Ann" to "pie", allowing it to later retrieve "Ann" when asked "Who loves pie?" Prior research on short lists of bound entities found strong evidence that LMs implement such retrieval via a positional mechanism, where "Ann" is retrieved based on its position in context. In this work, we find that this mechanism generalizes poorly to more complex settings; as the number of bound entities in context increases, the positional mechanism becomes noisy and unreliable in middle positions. To compensate for this, we find that LMs supplement the positional mechanism with a lexical mechanism (retrieving "Ann" using its bound counterpart "pie") and a reflexive mechanism (retrieving "Ann" through a direct pointer). Through extensive experiments on nine models and ten binding tasks, we uncover a consistent pattern in how LMs mix these mechanisms to drive model behavior. We leverage these insights to develop a causal model combining all three mechanisms that estimates next token distributions with 95% agreement. Finally, we show that our model generalizes to substantially longer inputs of open-ended text interleaved with entity groups, further demonstrating the robustness of our findings in more natural settings. Overall, our study establishes a more complete picture of how LMs bind and retrieve entities in-context.
>
---
#### [new 021] Trainable Reference-Based Evaluation Metric for Identifying Quality of English-Gujarati Machine Translation System
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译评价任务，旨在解决印度语种古吉拉特语翻译质量评估难题。作者构建了基于监督学习的参考评价指标，训练了两个深度学习模型（6层与10层），使用25个特征并进行500轮训练，最终通过1000个翻译输出验证模型效果，结果显示其与人工评价相关性更高。**

- **链接: [http://arxiv.org/pdf/2510.05113v1](http://arxiv.org/pdf/2510.05113v1)**

> **作者:** Nisheeth Joshi; Pragya Katyayan; Palak Arora
>
> **备注:** 8 Pages, 4 Tables, 4 Figures
>
> **摘要:** Machine Translation (MT) Evaluation is an integral part of the MT development life cycle. Without analyzing the outputs of MT engines, it is impossible to evaluate the performance of an MT system. Through experiments, it has been identified that what works for English and other European languages does not work well with Indian languages. Thus, In this paper, we have introduced a reference-based MT evaluation metric for Gujarati which is based on supervised learning. We have trained two versions of the metric which uses 25 features for training. Among the two models, one model is trained using 6 hidden layers with 500 epochs while the other model is trained using 10 hidden layers with 500 epochs. To test the performance of the metric, we collected 1000 MT outputs of seven MT systems. These MT engine outputs were compared with 1 human reference translation. While comparing the developed metrics with other available metrics, it was found that the metrics produced better human correlations.
>
---
#### [new 022] AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多智能体协作问答任务，旨在解决如何有效选择和组合不同大模型与策略以提升问答性能的问题。论文提出AgentRouter，通过知识图谱引导的路由机制，利用异构图神经网络学习任务感知的代理协作方案，实现了优于单一与集成基线的效果。**

- **链接: [http://arxiv.org/pdf/2510.05445v1](http://arxiv.org/pdf/2510.05445v1)**

> **作者:** Zheyuan Zhang; Kaiwen Shi; Zhengqing Yuan; Zehong Wang; Tianyi Ma; Keerthiram Murugesan; Vincent Galassi; Chuxu Zhang; Yanfang Ye
>
> **摘要:** Large language models (LLMs) and agent-based frameworks have advanced rapidly, enabling diverse applications. Yet, with the proliferation of models and agentic strategies, practitioners face substantial uncertainty in selecting the best configuration for a downstream task. Prior studies show that different agents and backbones exhibit complementary strengths, and that larger models are not always superior, underscoring the need for adaptive routing mechanisms. Existing approaches to agent routing, however, often emphasize cost efficiency while overlooking the fine-grained contextual and relational structure inherent in QA tasks. In this paper, we propose tAgentRouter, a framework that formulates multi-agent QA as a knowledge-graph-guided routing problem supervised by empirical performance signals. Specifically, we convert QA instance into a knowledge graph that jointly encodes queries, contextual entities, and agents, and then train a heterogeneous graph neural network (GNN) to propagate information across node types and produce task-aware routing distributions over agents. By leveraging soft supervision and weighted aggregation of agent outputs, AgentRouter learns principled collaboration schemes that capture the complementary strengths of diverse agents. Extensive experiments demonstrate that our framework consistently outperforms single-agent and ensemble baselines, while generalizing across benchmarks and LLM backbones. These results highlight the effectiveness and robustness of graph-supervised multi-agent routing for question answering.
>
---
#### [new 023] Training Large Language Models To Reason In Parallel With Global Forking Tokens
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在并行推理中多样性和准确性难以兼顾的问题。通过提出“集合监督微调”方法（SSFT），利用全局分叉标记与推理路径的二部图匹配，实现多样化且准确的并行推理，提升了模型在多个推理基准上的性能。**

- **链接: [http://arxiv.org/pdf/2510.05132v1](http://arxiv.org/pdf/2510.05132v1)**

> **作者:** Sheng Jia; Xiao Wang; Shiva Prasad Kasiviswanathan
>
> **摘要:** Although LLMs have demonstrated improved performance by scaling parallel test-time compute, doing so relies on generating reasoning paths that are both diverse and accurate. For challenging problems, the forking tokens that trigger diverse yet correct reasoning modes are typically deep in the sampling tree. Consequently, common strategies to encourage diversity, such as temperature scaling, encounter a worsened trade-off between diversity and accuracy. Motivated by this challenge, we treat parallel reasoning as a set-of-next-token-prediction problem, and incorporate a set-based global loss into Supervised Fine-Tuning (SFT) using self-supervised bipartite matching between our global forking tokens and unique reasoning traces. We observe that, while naive fine-tuning with multiple reasoning traces collapses these unique reasoning modes, our proposed method, Set Supervised Fine-Tuning (SSFT), preserves these modes and produces emergent global forking tokens. Experiments on multiple reasoning benchmarks show that our SSFT consistently outperforms SFT under both Pass@1 and Cons@k metrics.
>
---
#### [new 024] LexiCon: a Benchmark for Planning under Temporal Constraints in Natural Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言规划任务，旨在解决大型语言模型在带时间约束的规划任务中表现不佳的问题。作者构建了LexiCon基准，将现有环境加入时间约束并转为自然语言，评估LLMs的受限规划能力，并实现环境扩展与自动约束生成。**

- **链接: [http://arxiv.org/pdf/2510.05972v1](http://arxiv.org/pdf/2510.05972v1)**

> **作者:** Periklis Mantenoglou; Rishi Hazra; Pedro Zuidberg Dos Martires; Luc De Raedt
>
> **摘要:** Owing to their reasoning capabilities, large language models (LLMs) have been evaluated on planning tasks described in natural language. However, LLMs have largely been tested on planning domains without constraints. In order to deploy them in real-world settings where adherence to constraints, in particular safety constraints, is critical, we need to evaluate their performance on constrained planning tasks. We introduce LexiCon -- a natural language-based (Lexi) constrained (Con) planning benchmark, consisting of a suite of environments, that can be used to evaluate the planning capabilities of LLMs in a principled fashion. The core idea behind LexiCon is to take existing planning environments and impose temporal constraints on the states. These constrained problems are then translated into natural language and given to an LLM to solve. A key feature of LexiCon is its extensibility. That is, the set of supported environments can be extended with new (unconstrained) environment generators, for which temporal constraints are constructed automatically. This renders LexiCon future-proof: the hardness of the generated planning problems can be increased as the planning capabilities of LLMs improve. Our experiments reveal that the performance of state-of-the-art LLMs, including reasoning models like GPT-5, o3, and R1, deteriorates as the degree of constrainedness of the planning tasks increases.
>
---
#### [new 025] Every Step Counts: Decoding Trajectories as Authorship Fingerprints of dLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究离散扩散大语言模型（dLLMs）的解码轨迹作为模型归属的指纹。任务是模型归属，旨在区分不同模型或同一模型的不同检查点。论文提出Directed Decoding Map（DDM）提取结构信息，并用Gaussian-Trajectory Attribution（GTA）定义归属得分，以提升归属准确性。**

- **链接: [http://arxiv.org/pdf/2510.05148v1](http://arxiv.org/pdf/2510.05148v1)**

> **作者:** Qi Li; Runpeng Yu; Haiquan Lu; Xinchao Wang
>
> **摘要:** Discrete Diffusion Large Language Models (dLLMs) have recently emerged as a competitive paradigm for non-autoregressive language modeling. Their distinctive decoding mechanism enables faster inference speed and strong performance in code generation and mathematical tasks. In this work, we show that the decoding mechanism of dLLMs not only enhances model utility but also can be used as a powerful tool for model attribution. A key challenge in this problem lies in the diversity of attribution scenarios, including distinguishing between different models as well as between different checkpoints or backups of the same model. To ensure broad applicability, we identify two fundamental problems: what information to extract from the decoding trajectory, and how to utilize it effectively. We first observe that relying directly on per-step model confidence yields poor performance. This is mainly due to the bidirectional decoding nature of dLLMs: each newly decoded token influences the confidence of other decoded tokens, making model confidence highly redundant and washing out structural signal regarding decoding order or dependencies. To overcome this, we propose a novel information extraction scheme called the Directed Decoding Map (DDM), which captures structural relationships between decoding steps and better reveals model-specific behaviors. Furthermore, to make full use of the extracted structural information during attribution, we propose Gaussian-Trajectory Attribution (GTA), where we fit a cell-wise Gaussian distribution at each decoding position for each target model, and define the likelihood of a trajectory as the attribution score: if a trajectory exhibits higher log-likelihood under the distribution of a specific model, it is more likely to have been generated by that model. Extensive experiments under different settings validate the utility of our methods.
>
---
#### [new 026] Curiosity-Driven LLM-as-a-judge for Personalized Creative Judgment
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在个性化创意判断（如创造性写作评估）中的不足。作者提出了一种基于好奇心驱动的“LLM-as-a-judge”方法，通过学习个体的创造性判断偏好，在多个评估指标上优于传统的监督微调方法。**

- **链接: [http://arxiv.org/pdf/2510.05135v1](http://arxiv.org/pdf/2510.05135v1)**

> **作者:** Vanya Bannihatti Kumar; Divyanshu Goyal; Akhil Eppa; Neel Bhandari
>
> **摘要:** Modern large language models (LLMs) excel at objective tasks such as evaluating mathematical reasoning and factual accuracy, yet they falter when faced with the nuanced, subjective nature of assessing creativity. In this work, we propose a novel curiosity-driven LLM-as-a-judge for evaluating creative writing which is personlized to each individual's creative judgments. We use the Torrance Test of Creative Thinking(TTCW) benchmark introduced in Chakrabarty et al. (2024), which has stories annotated by expert humans across various subjective dimensions like Originality, to test our hypothesis. We show that our method enables models across various sizes, to learn the nuanced creative judgments of different individuals, by showing improvements over baseline supervised finetuning(SFT) method across various evaluation metrics like Pearson correlation, Cohen's and F1 values. Our method is especially useful in subjective evaluations where not all the annotators agree with each other.
>
---
#### [new 027] Context Length Alone Hurts LLM Performance Despite Perfect Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究长上下文输入对大语言模型（LLM）性能的影响，探讨即使信息检索完美，长上下文是否仍会损害模型表现。研究发现，输入长度本身会导致性能下降，与检索质量和干扰无关。为缓解此问题，论文提出一种通用策略：让模型先复述相关信息再解题，有效提升了性能。**

- **链接: [http://arxiv.org/pdf/2510.05381v1](http://arxiv.org/pdf/2510.05381v1)**

> **作者:** Yufeng Du; Minyang Tian; Srikanth Ronanki; Subendhu Rongali; Sravan Bodapati; Aram Galstyan; Azton Wells; Roy Schwartz; Eliu A Huerta; Hao Peng
>
> **备注:** 18 pages (9 pages of main content), 5 figures, accepted at the Findings of EMNLP 2025
>
> **摘要:** Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline.
>
---
#### [new 028] Prompt reinforcing for long-term planning of large language models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在多轮交互中规划能力不足的问题。作者提出了一种受强化学习启发的提示优化框架，通过逐轮反馈和经验回放来改进任务指令提示，提升模型在文本到SQL和任务导向对话等多轮任务中的表现。**

- **链接: [http://arxiv.org/pdf/2510.05921v1](http://arxiv.org/pdf/2510.05921v1)**

> **作者:** Hsien-Chin Lin; Benjamin Matthias Ruppik; Carel van Niekerk; Chia-Hao Shen; Michael Heck; Nurul Lubis; Renato Vukovic; Shutong Feng; Milica Gašić
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in a wide range of natural language processing tasks and can be adapted through prompting. However, they remain suboptimal in multi-turn interactions, often relying on incorrect early assumptions and failing to track user goals over time, which makes such tasks particularly challenging. Prior works in dialogue systems have shown that long-term planning is essential for handling interactive tasks. In this work, we propose a prompt optimisation framework inspired by reinforcement learning, which enables such planning to take place by only modifying the task instruction prompt of the LLM-based agent. By generating turn-by-turn feedback and leveraging experience replay for prompt rewriting, our proposed method shows significant improvement in multi-turn tasks such as text-to-SQL and task-oriented dialogue. Moreover, it generalises across different LLM-based agents and can leverage diverse LLMs as meta-prompting agents. This warrants future research in reinforcement learning-inspired parameter-free optimisation methods.
>
---
#### [new 029] Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的安全评估任务，旨在研究大语言模型（LLMs）在长文本输入中对有害内容的敏感性。论文系统评估了不同类型、位置和比例的有害内容对LLMs的影响，揭示了模型在安全关键场景中的表现特点与挑战。**

- **链接: [http://arxiv.org/pdf/2510.05864v1](http://arxiv.org/pdf/2510.05864v1)**

> **作者:** Faeze Ghorbanpour; Alexander Fraser
>
> **摘要:** Large language models (LLMs) increasingly support applications that rely on extended context, from document processing to retrieval-augmented generation. While their long-context capabilities are well studied for reasoning and retrieval, little is known about their behavior in safety-critical scenarios. We evaluate LLMs' sensitivity to harmful content under extended context, varying type (explicit vs. implicit), position (beginning, middle, end), prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens). Across harmful content categories such as toxic, offensive, and hate speech, with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance peaks at moderate harmful prevalence (0.25) but declines when content is very sparse or dominant; recall decreases with increasing context length; harmful sentences at the beginning are generally detected more reliably; and explicit content is more consistently recognized than implicit. These findings provide the first systematic view of how LLMs prioritize and calibrate harmful content in long contexts, highlighting both their emerging strengths and the challenges that remain for safety-critical use.
>
---
#### [new 030] ASPO: Asymmetric Importance Sampling Policy Optimization
- **分类: cs.CL**

- **简介: 论文提出ASPO算法，属于大语言模型强化学习任务，旨在解决现有方法中正负token重要性采样比例失衡导致的训练不稳定问题。通过翻转正优势token的采样比例并引入软双截断机制，提升训练稳定性和最终性能。**

- **链接: [http://arxiv.org/pdf/2510.06062v1](http://arxiv.org/pdf/2510.06062v1)**

> **作者:** Jiakang Wang; Runze Liu; Lei Lin; Wenping Hu; Xiu Li; Fuzheng Zhang; Guorui Zhou; Kun Gai
>
> **摘要:** Recent Large Language Model (LLM) post-training methods rely on token-level clipping mechanisms during Reinforcement Learning (RL). However, we identify a fundamental flaw in this Outcome-Supervised RL (OSRL) paradigm: the Importance Sampling (IS) ratios of positive-advantage tokens are mismatched, leading to unbalanced token weighting for positive and negative tokens. This mismatch suppresses the update of low-probability tokens while over-amplifying already high-probability ones. To address this, we propose Asymmetric Importance Sampling Policy Optimization (ASPO), which uses a simple yet effective strategy that flips the IS ratios of positive-advantage tokens, aligning their update direction with the learning dynamics of negative ones. AIS further incorporates a soft dual-clipping mechanism to stabilize extreme updates while maintaining gradient flow. Comprehensive experiments on coding and mathematical reasoning benchmarks demonstrate that ASPO significantly mitigates premature convergence, improves training stability, and enhances final performance over strong GRPO-based baselines. Our analysis provides new insights into the role of token-level weighting in OSRL and highlights the critical importance of correcting IS in LLM RL. The code and models of ASPO are available at https://github.com/wizard-III/Archer2.0.
>
---
#### [new 031] On the Role of Difficult Prompts in Self-Play Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在探究困难提示在自博弈偏好优化中的影响。通过分析不同难度提示对优化效果的作用，发现困难提示会降低性能，增加模型容量可缓解此问题。论文提出选择性移除困难提示以提升整体性能，并总结了相关实验结果与经验教训。**

- **链接: [http://arxiv.org/pdf/2510.05534v1](http://arxiv.org/pdf/2510.05534v1)**

> **作者:** Yao Xiao; Jung-jae Kim; Roy Ka-wei Lee; Lidong Bing
>
> **摘要:** Self-play preference optimization has emerged as a prominent paradigm for aligning large language models (LLMs). It typically involves a language model to generate on-policy responses for prompts and a reward model (RM) to guide the selection of chosen and rejected responses, which can be further trained with direct preference optimization (DPO). However, the role of prompts remains underexplored, despite being a core component in this pipeline. In this work, we investigate how prompts of varying difficulty influence self-play preference optimization. We first use the mean reward of $N$ sampled responses of a prompt as a proxy for its difficulty. We find that difficult prompts exhibit substantially inferior self-play optimization performance in comparison to easy prompts for language models. Moreover, incorporating difficult prompts into training fails to enhance overall performance and, in fact, leads to slight degradation compared to training on easy prompts alone. We also observe that the performance gap between difficult and easy prompts closes as the model capacity increases, suggesting that difficulty interacts with the model capacity. Building on these findings, we explore strategies to mitigate the negative effect of difficult prompts on final performance. We demonstrate that selectively removing an appropriate portion of challenging prompts enhances overall self-play performance, while also reporting failed attempts and lessons learned.
>
---
#### [new 032] To model human linguistic prediction, make LLMs less superhuman
- **分类: cs.CL**

- **简介: 该论文探讨将大语言模型（LLMs）用于模拟人类语言预测的问题。任务是使其更贴近人类认知，而非超人化。问题在于当前LLMs因记忆能力过强，无法准确预测人类阅读行为。论文分析原因并提出改进方向，同时指出需更多人类实验数据来评估进展。**

- **链接: [http://arxiv.org/pdf/2510.05141v1](http://arxiv.org/pdf/2510.05141v1)**

> **作者:** Byung-Doh Oh; Tal Linzen
>
> **摘要:** When people listen to or read a sentence, they actively make predictions about upcoming words: words that are less predictable are generally read more slowly than predictable ones. The success of large language models (LLMs), which, like humans, make predictions about upcoming words, has motivated exploring the use of these models as cognitive models of human linguistic prediction. Surprisingly, in the last few years, as language models have become better at predicting the next word, their ability to predict human reading behavior has declined. This is because LLMs are able to predict upcoming words much better than people can, leading them to predict lower processing difficulty in reading than observed in human experiments; in other words, mainstream LLMs are 'superhuman' as models of language comprehension. In this position paper, we argue that LLMs' superhumanness is primarily driven by two factors: compared to humans, LLMs have much stronger long-term memory for facts and training examples, and they have much better short-term memory for previous words in the text. We advocate for creating models that have human-like long-term and short-term memory, and outline some possible directions for achieving this goal. Finally, we argue that currently available human data is insufficient to measure progress towards this goal, and outline human experiments that can address this gap.
>
---
#### [new 033] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决传统TTS系统在发音对齐上的不足。现有方法依赖成对的优劣语句级样本，数据效率低且无法实现细粒度优化。论文提出TKTO，无需配对数据，实现更高效训练，并直接优化词元级发音对齐，显著提升日语TTS准确率并降低错误率。**

- **链接: [http://arxiv.org/pdf/2510.05799v1](http://arxiv.org/pdf/2510.05799v1)**

> **作者:** Rikuto Kotoge; Yuichi Sasaki
>
> **摘要:** Aligning text-to-speech (TTS) system outputs with human feedback through preference optimization has been shown to effectively improve the robustness and naturalness of language model-based TTS models. Current approaches primarily require paired desirable and undesirable samples at the utterance level. However, such pairs are often limited in TTS output data, and utterance-level formulation prevents fine-grained token-level optimization needed for accurate pronunciation alignment. In this study, we propose TKTO that eliminates the need for paired data, enabling a more data-efficient training paradigm, and directly targets token-level units, automatically providing fine-grained alignment signals without token-level annotations. TKTO improves the challenging Japanese TTS accuracy by 39% and reduces CER by 54%, automatically assigning 12.8 times stronger reward to targeted tokens.
>
---
#### [new 034] SynCED-EnDe 2025: A Synthetic and Curated English - German Dataset for Critical Error Detection in Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译中的关键错误检测任务，旨在判断翻译是否含有不可接受的语义偏差。论文构建了新数据集SynCED-EnDe，包含1,000人工标注和8,000半自动标注的英德句对，涵盖多样领域并引入错误子类与细粒度标注，提升错误分析能力。数据集公开并附有工具与基线模型，推动安全翻译应用。**

- **链接: [http://arxiv.org/pdf/2510.05144v1](http://arxiv.org/pdf/2510.05144v1)**

> **作者:** Muskaan Chopra; Lorenz Sparrenberg; Rafet Sifa
>
> **摘要:** Critical Error Detection (CED) in machine translation aims to determine whether a translation is safe to use or contains unacceptable deviations in meaning. While the WMT21 English-German CED dataset provided the first benchmark, it is limited in scale, label balance, domain coverage, and temporal freshness. We present SynCED-EnDe, a new resource consisting of 1,000 gold-labeled and 8,000 silver-labeled sentence pairs, balanced 50/50 between error and non-error cases. SynCED-EnDe draws from diverse 2024-2025 sources (StackExchange, GOV.UK) and introduces explicit error subclasses, structured trigger flags, and fine-grained auxiliary judgments (obviousness, severity, localization complexity, contextual dependency, adequacy deviation). These enrichments enable systematic analyses of error risk and intricacy beyond binary detection. The dataset is permanently hosted on GitHub and Hugging Face, accompanied by documentation, annotation guidelines, and baseline scripts. Benchmark experiments with XLM-R and related encoders show substantial performance gains over WMT21 due to balanced labels and refined annotations. We envision SynCED-EnDe as a community resource to advance safe deployment of MT in information retrieval and conversational assistants, particularly in emerging contexts such as wearable AI devices.
>
---
#### [new 035] MADIAVE: Multi-Agent Debate for Implicit Attribute Value Extraction
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MADIAVE，一种多智能体辩论框架，用于隐式属性值抽取（Implicit AVE）任务，旨在从多模态数据中推断产品潜在属性。通过多轮辩论，多个MLLM代理相互验证和更新结果，提升推理性能与鲁棒性，尤其改善初始表现差的属性。实验表明该方法在ImplicitAVE数据集上效果显著。**

- **链接: [http://arxiv.org/pdf/2510.05611v1](http://arxiv.org/pdf/2510.05611v1)**

> **作者:** Wei-Chieh Huang; Cornelia Caragea
>
> **摘要:** Implicit Attribute Value Extraction (AVE) is essential for accurately representing products in e-commerce, as it infers lantent attributes from multimodal data. Despite advances in multimodal large language models (MLLMs), implicit AVE remains challenging due to the complexity of multidimensional data and gaps in vision-text understanding. In this work, we introduce \textsc{\modelname}, a multi-agent debate framework that employs multiple MLLM agents to iteratively refine inferences. Through a series of debate rounds, agents verify and update each other's responses, thereby improving inference performance and robustness. Experiments on the ImplicitAVE dataset demonstrate that even a few rounds of debate significantly boost accuracy, especially for attributes with initially low performance. We systematically evaluate various debate configurations, including identical or different MLLM agents, and analyze how debate rounds affect convergence dynamics. Our findings highlight the potential of multi-agent debate strategies to address the limitations of single-agent approaches and offer a scalable solution for implicit AVE in multimodal e-commerce.
>
---
#### [new 036] SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决AI助手多轮对话评估中用户模拟器的可靠性问题。作者构建了SimulatorArena基准，包含909个人类-LLM对话，评估模拟用户的行为匹配度与人类判断的一致性，发现基于用户画像的模拟器可达到与人类评估较高相关性（Spearman's ρ 0.7），可作为评估AI助手的高效替代方案。**

- **链接: [http://arxiv.org/pdf/2510.05444v1](http://arxiv.org/pdf/2510.05444v1)**

> **作者:** Yao Dou; Michel Galley; Baolin Peng; Chris Kedzie; Weixin Cai; Alan Ritter; Chris Quirk; Wei Xu; Jianfeng Gao
>
> **备注:** Accepted at EMNLP 2025 Main
>
> **摘要:** Large language models (LLMs) are increasingly used in interactive applications, and human evaluation remains the gold standard for assessing their performance in multi-turn conversations. Since human studies are costly, time-consuming, and hard to reproduce, recent work explores using LLMs to simulate users for automatic assistant evaluation. However, there is no benchmark or systematic study to evaluate whether these simulated users are reliable stand-ins for real users. To address this, we introduce SimulatorArena, a benchmark of 909 annotated human-LLM conversations on two interactive tasks -- math tutoring and document creation. SimulatorArena evaluates simulators based on how closely their messages match human behavior and how well their assistant ratings align with human judgments. Experiments on various simulator methods show that simulators conditioned on user profiles, capturing traits like background and message styles, align closely with human judgments. They reach Spearman's $\rho$ of 0.7 on both tasks, providing a practical, scalable alternative to human evaluation. Using the best simulator for each task, we benchmark 18 assistants, including the latest LLMs such as GPT-5, Claude 4.1 Opus, and Gemini 2.5 Pro.
>
---
#### [new 037] Language Model as Planner and Formalizer under Constraints
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与规划任务，旨在解决大语言模型在规划任务中因依赖简单环境设定导致的能力高估和安全问题。论文引入了精细标注的自然语言约束，覆盖四个形式化类别，增强现有规划基准。实验表明，这些约束显著降低模型性能，挑战其对问题复杂度和词汇变化的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.05486v1](http://arxiv.org/pdf/2510.05486v1)**

> **作者:** Cassie Huang; Stuti Mohan; Ziyi Yang; Stefanie Tellex; Li Zhang
>
> **摘要:** LLMs have been widely used in planning, either as planners to generate action sequences end-to-end, or as formalizers to represent the planning domain and problem in a formal language that can derive plans deterministically. However, both lines of work rely on standard benchmarks that only include generic and simplistic environmental specifications, leading to potential overestimation of the planning ability of LLMs and safety concerns in downstream tasks. We bridge this gap by augmenting widely used planning benchmarks with manually annotated, fine-grained, and rich natural language constraints spanning four formally defined categories. Over 4 state-of-the-art reasoning LLMs, 3 formal languages, 5 methods, and 4 datasets, we show that the introduction of constraints not only consistently halves performance, but also significantly challenges robustness to problem complexity and lexical shift.
>
---
#### [new 038] Peeking inside the Black-Box: Reinforcement Learning for Explainable and Accurate Relation Extraction
- **分类: cs.CL; cs.IR**

- **简介: 论文属于关系抽取任务，旨在解决传统方法在准确性和解释性上的不足。作者提出CogRE框架，结合认知科学启发的推理机制与强化学习优化，提升一发关系抽取效果。通过自动生成关键词典辅助解释，实验表明方法在准确性和解释性上均有显著提升。**

- **链接: [http://arxiv.org/pdf/2510.06198v1](http://arxiv.org/pdf/2510.06198v1)**

> **作者:** Xinyu Guo; Zhengliang Shi; Minglai Yang; Mahdi Rahimi; Mihai Surdeanu
>
> **备注:** Working in process
>
> **摘要:** This paper introduces a framework for relation extraction (RE) that enhances both accuracy and explainability. The framework has two key components: (i) a reasoning mechanism that formulates relation extraction as a series of text-processing steps inspired by cognitive science, and (ii) an optimization process driven by reinforcement learning (RL) with a novel reward function designed to improve both task accuracy and explanation quality. We call our approach CogRE. Our framework addresses the lack of supervision for language-based explanations in traditional RE by promoting outputs that include important relation keywords. These keywords are drawn from a high-quality dictionary that is automatically constructed using an LLM. We evaluate our approach for the task of one-shot RE using two LLMs and two RE datasets. Our experiments show that CogRE improves explanation quality by addressing two common failure patterns in one-shot RE: poor attention focus and limited one-shot learning capability. For example, our cognitive-structured reasoning with Qwen2.5-15B-Instruct on One-shot NYT29 achieves 24.65% F1, surpassing prior reasoning-based designs. Optimizing this approach with RL using our reward further improves performance by +23.46% (absolute). Finally, human evaluation shows that our best model generates relational keywords closely aligned with gold labels, increasing human explanation quality ratings by 54% (relative).
>
---
#### [new 039] A Lightweight Large Language Model-Based Multi-Agent System for 2D Frame Structural Analysis
- **分类: cs.CL**

- **简介: 论文提出了一种基于轻量级大语言模型的多智能体系统，用于自动化二维框架结构的有限元建模。该系统分解结构分析任务，利用多个专业代理协同完成几何建模、代码生成和模型验证，解决结构工程中有限元建模自动化程度低的问题。**

- **链接: [http://arxiv.org/pdf/2510.05414v1](http://arxiv.org/pdf/2510.05414v1)**

> **作者:** Ziheng Geng; Jiachen Liu; Ran Cao; Lu Cheng; Haifeng Wang; Minghui Cheng
>
> **摘要:** Large language models (LLMs) have recently been used to empower autonomous agents in engineering, significantly improving automation and efficiency in labor-intensive workflows. However, their potential remains underexplored in structural engineering, particularly for finite element modeling tasks requiring geometric modeling, complex reasoning, and domain knowledge. To bridge this gap, this paper develops a LLM-based multi-agent system to automate finite element modeling of 2D frames. The system decomposes structural analysis into subtasks, each managed by a specialized agent powered by the lightweight Llama-3.3 70B Instruct model. The workflow begins with a Problem Analysis Agent, which extracts geometry, boundary, and material parameters from the user input. Next, a Geometry Agent incrementally derives node coordinates and element connectivity by applying expert-defined rules. These structured outputs are converted into executable OpenSeesPy code by a Translation Agent and refined by a Model Validation Agent through consistency checks. Then, a Load Agent applies load conditions into the assembled structural model. Experimental evaluations on 20 benchmark problems demonstrate that the system achieves accuracy over 80% in most cases across 10 repeated trials, outperforming Gemini-2.5 Pro and ChatGPT-4o models.
>
---
#### [new 040] Demystifying deep search: a holistic evaluation with hint-free multi-hop questions and factorised metrics
- **分类: cs.CL**

- **简介: 该论文属于信息检索与生成任务，旨在解决当前RAG系统和网络代理在多跳深度搜索中评估不全面的问题。作者构建了无提示多跳问答基准WebDetective和评估框架，分离搜索充分性、知识利用与拒绝行为，揭示模型在自主推理路径发现上的不足，并提出改进的工作流EvidenceLoop提升系统表现。**

- **链接: [http://arxiv.org/pdf/2510.05137v1](http://arxiv.org/pdf/2510.05137v1)**

> **作者:** Maojia Song; Renhang Liu; Xinyu Wang; Yong Jiang; Pengjun Xie; Fei Huang; Soujanya Poria; Jingren Zhou
>
> **摘要:** RAG (Retrieval-Augmented Generation) systems and web agents are increasingly evaluated on multi-hop deep search tasks, yet current practice suffers from two major limitations. First, most benchmarks leak the reasoning path in the question text, allowing models to follow surface cues rather than discover reasoning chains autonomously. Second, evaluation is typically reduced to a single pass rate, which collapses diverse behaviours into one score and obscures whether failures stem from inadequate search, poor knowledge use, or inappropriate refusal. To address these issues, we present WebDetective, a benchmark of hint-free multi-hop questions paired with a controlled Wikipedia sandbox that ensures full traceability of model actions, and a holistic evaluation framework that separates search sufficiency, knowledge utilisation, and refusal behaviour. Our evaluation of 25 state-of-the-art models reveals systematic weaknesses across all architectures: models struggle with knowledge utilisation despite having sufficient evidence and demonstrate near-absent appropriate refusal when evidence is lacking. These patterns expose a fundamental gap: today's systems excel at executing given reasoning paths but fail when required to discover them. We develop an agentic workflow, EvidenceLoop, that explicitly targets the challenges our benchmark identifies, incorporating verification loops and systematic evidence tracking that improve both search and synthesis capabilities. This baseline demonstrates that WebDetective's diagnostic framework can guide concrete architectural improvements, establishing our benchmark as a critical tool for developing genuinely autonomous reasoning systems rather than pattern-following agents.
>
---
#### [new 041] Prototype-Based Dynamic Steering for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理能力。现有方法依赖显式指令或静态干预，效果有限。作者提出基于原型的动态引导（PDS），通过聚类推理与中性提示的激活差异构建“推理原型”，在推理时生成实例相关的引导向量。实验表明，PDS在多个任务上有效提升模型表现，且无需微调或提示工程。**

- **链接: [http://arxiv.org/pdf/2510.05498v1](http://arxiv.org/pdf/2510.05498v1)**

> **作者:** Ceyhun Efe Kayan; Li Zhang
>
> **摘要:** Despite impressive breadth, LLMs still rely on explicit reasoning instructions or static, one-fits-all steering methods, leaving a gap for adaptive, instruction-free reasoning amplification. We present Prototype-Based Dynamic Steering (PDS), a test-time method that amplifies large language model (LLM) reasoning without adding or altering instructions. We introduce "reasoning prototypes" by clustering activation differences between Chain-of-Thought (CoT) and neutral prompts. At inference, an input's hidden state is projected onto these prototypes to form an instance-specific steering vector. Evaluated on GSM8K, AQuA-RAT, and BIG-Bench tasks, PDS consistently improves accuracy without fine-tuning or prompt engineering. Notably, the gains persist even when CoT is explicitly suppressed to improve cost-efficiency, indicating that the intervention strengthens latent reasoning processes rather than inducing a superficial behavioral shift. These results position dynamic, prototype-guided steering as a lightweight alternative to training-time approaches for enhancing LLM reasoning.
>
---
#### [new 042] RoSE: Round-robin Synthetic Data Evaluation for Selecting LLM Generators without Human Test Sets
- **分类: cs.CL**

- **简介: 论文提出RoSE方法，用于在无人工测试集情况下选择最佳LLM生成器。其任务是解决低资源语言中合成数据生成的评估难题。通过训练小模型并跨LLM生成数据评估，得出RoSE评分，有效预测下游性能，优于现有指标。**

- **链接: [http://arxiv.org/pdf/2510.06143v1](http://arxiv.org/pdf/2510.06143v1)**

> **作者:** Jan Cegin; Branislav Pecher; Ivan Srba; Jakub Simko
>
> **备注:** 16 pages
>
> **摘要:** LLMs are powerful generators of synthetic data, which are used for training smaller, specific models. This is especially valuable for low-resource languages, where human-labelled data is scarce but LLMs can still produce high-quality text. However, LLMs differ in how useful their outputs are for training. Selecting the best LLM as a generator is challenging because extrinsic evaluation requires costly human annotations (which are often unavailable for low-resource languages), while intrinsic metrics correlate poorly with downstream performance. We introduce Round robin Synthetic data Evaluation (RoSE), a proxy metric for selecting the best LLM generator without human test sets. RoSE trains a small model on the outputs of a candidate generator (LLM) and then evaluates it on generated synthetic examples from all other candidate LLMs. The final RoSE score is the mean performance of this small model. Across six LLMs, eleven languages, and three tasks (sentiment, topic, intent), RoSE identifies the optimal generator more often than any other intrinsic heuristics. RoSE outperforms intrinsic heuristics and comes within 0.76 percentage points of the optimal generator baseline. This result is measured in terms of downstream performance, obtained by training a small model on the chosen generator's outputs (optimal vs. proxy metric selected) and evaluating it on human-labelled test data. Additionally, RoSE is the only metric to achieve a positive correlation with performance on human test data.
>
---
#### [new 043] KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG for Safety-Critical Aviation Maintenance
- **分类: cs.CL; cs.IR**

- **简介: 论文提出KEO框架，结合知识图谱与RAG技术，利用大语言模型在航空安全维护领域进行知识抽取与推理。旨在解决传统文本分块RAG在全局推理上的不足，提升维护任务中的系统级洞察与问答效果。**

- **链接: [http://arxiv.org/pdf/2510.05524v1](http://arxiv.org/pdf/2510.05524v1)**

> **作者:** Kuangshi Ai; Jonathan A. Karr Jr; Meng Jiang; Nitesh V. Chawla; Chaoli Wang
>
> **摘要:** We present Knowledge Extraction on OMIn (KEO), a domain-specific knowledge extraction and reasoning framework with large language models (LLMs) in safety-critical contexts. Using the Operations and Maintenance Intelligence (OMIn) dataset, we construct a QA benchmark spanning global sensemaking and actionable maintenance tasks. KEO builds a structured Knowledge Graph (KG) and integrates it into a retrieval-augmented generation (RAG) pipeline, enabling more coherent, dataset-wide reasoning than traditional text-chunk RAG. We evaluate locally deployable LLMs (Gemma-3, Phi-4, Mistral-Nemo) and employ stronger models (GPT-4o, Llama-3.3) as judges. Experiments show that KEO markedly improves global sensemaking by revealing patterns and system-level insights, while text-chunk RAG remains effective for fine-grained procedural tasks requiring localized retrieval. These findings underscore the promise of KG-augmented LLMs for secure, domain-specific QA and their potential in high-stakes reasoning.
>
---
#### [new 044] Exploring Large Language Models for Financial Applications: Techniques, Performance, and Challenges with FinMA
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于金融自然语言处理任务，旨在解决大语言模型在金融领域的适应性问题。论文通过分析FinMA模型的架构、指令微调过程及在FLARE基准上的表现，探讨其在情感分析、分类等任务中的效果与挑战。**

- **链接: [http://arxiv.org/pdf/2510.05151v1](http://arxiv.org/pdf/2510.05151v1)**

> **作者:** Prudence Djagba; Abdelkader Y. Saley
>
> **摘要:** This research explores the strengths and weaknesses of domain-adapted Large Language Models (LLMs) in the context of financial natural language processing (NLP). The analysis centers on FinMA, a model created within the PIXIU framework, which is evaluated for its performance in specialized financial tasks. Recognizing the critical demands of accuracy, reliability, and domain adaptation in financial applications, this study examines FinMA's model architecture, its instruction tuning process utilizing the Financial Instruction Tuning (FIT) dataset, and its evaluation under the FLARE benchmark. Findings indicate that FinMA performs well in sentiment analysis and classification, but faces notable challenges in tasks involving numerical reasoning, entity recognition, and summarization. This work aims to advance the understanding of how financial LLMs can be effectively designed and evaluated to assist in finance-related decision-making processes.
>
---
#### [new 045] LiRA: A Multi-Agent Framework for Reliable and Readable Literature Review Generation
- **分类: cs.CL**

- **简介: 该论文属于自动化文献综述生成任务，旨在解决现有方法在可读性和事实准确性上的不足。论文提出了LiRA框架，采用多智能体协作流程模拟人工综述撰写过程，并在多个数据集上验证了其在写作质量和引用准确性上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.05138v1](http://arxiv.org/pdf/2510.05138v1)**

> **作者:** Gregory Hok Tjoan Go; Khang Ly; Anders Søgaard; Amin Tabatabaei; Maarten de Rijke; Xinyi Chen
>
> **摘要:** The rapid growth of scientific publications has made it increasingly difficult to keep literature reviews comprehensive and up-to-date. Though prior work has focused on automating retrieval and screening, the writing phase of systematic reviews remains largely under-explored, especially with regard to readability and factual accuracy. To address this, we present LiRA (Literature Review Agents), a multi-agent collaborative workflow which emulates the human literature review process. LiRA utilizes specialized agents for content outlining, subsection writing, editing, and reviewing, producing cohesive and comprehensive review articles. Evaluated on SciReviewGen and a proprietary ScienceDirect dataset, LiRA outperforms current baselines such as AutoSurvey and MASS-Survey in writing and citation quality, while maintaining competitive similarity to human-written reviews. We further evaluate LiRA in real-world scenarios using document retrieval and assess its robustness to reviewer model variation. Our findings highlight the potential of agentic LLM workflows, even without domain-specific tuning, to improve the reliability and usability of automated scientific writing.
>
---
#### [new 046] CARE: Cognitive-reasoning Augmented Reinforcement for Emotional Support Conversation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感支持对话任务，旨在解决现有方法忽视深层认知推理的问题。作者提出CARE框架，通过认知推理增强和强化学习，提升对话系统的逻辑连贯性与支持质量，无需依赖大规模合成数据。**

- **链接: [http://arxiv.org/pdf/2510.05122v1](http://arxiv.org/pdf/2510.05122v1)**

> **作者:** Jie Zhu; Yuanchen Zhou; Shuo Jiang; Junhui Li; Lifan Guo; Feng Chen; Chi Zhang; Fang Kong
>
> **备注:** Preprint
>
> **摘要:** Emotional Support Conversation (ESC) plays a vital role in alleviating psychological stress and providing emotional value through dialogue. While recent studies have largely focused on data augmentation and synthetic corpus construction, they often overlook the deeper cognitive reasoning processes that underpin effective emotional support. To address this gap, we propose \textbf{CARE}, a novel framework that strengthens reasoning in ESC without relying on large-scale synthetic data. CARE leverages the original ESC training set to guide models in generating logically coherent and supportive responses, thereby explicitly enhancing cognitive reasoning. Building on this foundation, we further employ reinforcement learning to refine and reinforce the reasoning process. Experimental results demonstrate that CARE significantly improves both the logical soundness and supportive quality of responses, advancing the development of empathetic, cognitively robust, and human-like emotional support systems.
>
---
#### [new 047] SocialNLI: A Dialogue-Centric Social Inference Dataset
- **分类: cs.CL**

- **简介: 该论文提出了SocialNLI数据集，用于评估模型从对话中进行社会推理的能力，属于自然语言处理中的社会推理任务。旨在解决当前模型难以理解复杂社会现象（如讽刺、反语）的问题，通过多步反事实推理分析模型心智理论能力。**

- **链接: [http://arxiv.org/pdf/2510.05458v1](http://arxiv.org/pdf/2510.05458v1)**

> **作者:** Akhil Deo; Kate Sanders; Benjamin Van Durme
>
> **备注:** 4 pages
>
> **摘要:** Making theory-of-mind inferences from human dialogue is a strong indicator of a model's underlying social abilities, which are fundamental for adept AI assistants. However, large language and reasoning models struggle to understand sophisticated social phenomena in transcript data, such as sarcasm and irony. To assess the weaknesses of current models and to identify their solutions, we introduce SocialNLI (SoNLI) -- the first social dialogue inference dataset. SoNLI consists of a collection of dialogue transcripts hand-picked to center complex social nuances like irony and sarcasm, paired with inferences, corresponding likelihood scores, and human-written explanations. We explore social inference analysis as a facet of theory-of-mind, and evaluate LLM and reasoning model theory-of-mind ability through multi-step counterfactual reasoning.
>
---
#### [new 048] InforME: Improving Informativeness of Abstractive Text Summarization With Informative Attention Guided by Named Entity Salience
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本摘要任务，旨在提升摘要的信息量。通过引入基于实体显著性的信息注意力机制与熵减方法，优化模型对关键信息的学习。实验表明，在CNN/Daily Mail和XSum数据集上，该方法在ROUGE分数和人工评价的信息量方面均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.05769v1](http://arxiv.org/pdf/2510.05769v1)**

> **作者:** Jianbin Shen; Christy Jie Liang; Junyu Xuan
>
> **摘要:** Abstractive text summarization is integral to the Big Data era, which demands advanced methods to turn voluminous and often long text data into concise but coherent and informative summaries for efficient human consumption. Despite significant progress, there is still room for improvement in various aspects. One such aspect is to improve informativeness. Hence, this paper proposes a novel learning approach consisting of two methods: an optimal transport-based informative attention method to improve learning focal information in reference summaries and an accumulative joint entropy reduction method on named entities to enhance informative salience. Experiment results show that our approach achieves better ROUGE scores compared to prior work on CNN/Daily Mail while having competitive results on XSum. Human evaluation of informativeness also demonstrates the better performance of our approach over a strong baseline. Further analysis gives insight into the plausible reasons underlying the evaluation results.
>
---
#### [new 049] Characterizing Model Behavior Under Synthetic Data Training: An Empirical Study Across Scales and Mixing Ratios
- **分类: cs.CL**

- **简介: 该论文研究在不同比例合成数据训练下模型的行为变化，属于自然语言处理任务。旨在解决合成数据对模型性能、校准和输出特性的影响问题。作者通过控制实验，使用Pythia模型套件在五项任务上评估不同合成数据比例下的表现，发现模型性能随合成数据比例增加而下降，但存在安全比例范围，并给出实用建议。**

- **链接: [http://arxiv.org/pdf/2510.05133v1](http://arxiv.org/pdf/2510.05133v1)**

> **作者:** Y. Du; G. Wu; G. Tang; W. Wang; Q. Fan
>
> **备注:** 17 pages. Technical report
>
> **摘要:** Synthetic data generated by large language models has become integral to modern NLP training pipelines, from bootstrapping reasoning capabilities to augmenting instruction-following datasets. While recent work demonstrates successful applications maintaining high external data ratios, systematic understanding of how synthetic data proportion affects model behavior across different scales remains limited. This paper presents a controlled empirical study examining model performance, calibration, and output characteristics when trained on varying synthetic-to-external data ratios. Using the Pythia model suite (410M-12B parameters) across five diverse tasks, we evaluate models after one to three training iterations with synthetic data proportions ranging from 0-50\%. Our key findings include: models maintain stable performance with up to 20\% synthetic data, but degradation accelerates beyond 30\%; larger models (6.9B-12B) show greater robustness to synthetic data than smaller models (410M-1.4B); calibration degradation precedes accuracy loss, providing an early warning signal; and task characteristics matter, with reasoning tasks degrading faster than retrieval tasks under synthetic data training. Importantly, we find that current best practices, such as those employed in STaR and Self-Instruct systems that maintain greater than 80\% external data, operate well within safe regimes identified by our experiments. We provide practical guidance for practitioners on synthetic data budgets based on model scale and task requirements, alongside detailed comparison with concurrent work including Shumailov et al.'s model collapse findings.
>
---
#### [new 050] A novel hallucination classification framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成内容中出现的幻觉问题。通过构建幻觉分类框架，利用提示工程重现多种幻觉类型，并使用嵌入模型与无监督学习分析其向量空间分布，提出一种轻量级幻觉检测方法。**

- **链接: [http://arxiv.org/pdf/2510.05189v1](http://arxiv.org/pdf/2510.05189v1)**

> **作者:** Maksym Zavhorodnii; Dmytro Dehtiarov; Anna Konovalenko
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** This work introduces a novel methodology for the automatic detection of hallucinations generated during large language model (LLM) inference. The proposed approach is based on a systematic taxonomy and controlled reproduction of diverse hallucination types through prompt engineering. A dedicated hallucination dataset is subsequently mapped into a vector space using an embedding model and analyzed with unsupervised learning techniques in a reduced-dimensional representation of hallucinations with veridical responses. Quantitative evaluation of inter-centroid distances reveals a consistent correlation between the severity of informational distortion in hallucinations and their spatial divergence from the cluster of correct outputs. These findings provide theoretical and empirical evidence that even simple classification algorithms can reliably distinguish hallucinations from accurate responses within a single LLM, thereby offering a lightweight yet effective framework for improving model reliability.
>
---
#### [new 051] Latent Speech-Text Transformer
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出“潜在语音-文本Transformer”（LST），旨在提升语音-文本多模态模型的预训练效率与性能。该任务属于多模态学习，解决语音与文本序列长度不平衡导致的计算低效和对齐困难问题。工作核心是通过动态聚合语音token为潜在语音块，实现更高效的数据与计算利用。**

- **链接: [http://arxiv.org/pdf/2510.06195v1](http://arxiv.org/pdf/2510.06195v1)**

> **作者:** Yen-Ju Lu; Yashesh Gaur; Wei Zhou; Benjamin Muller; Jesus Villalba; Najim Dehak; Luke Zettlemoyer; Gargi Ghosh; Mike Lewis; Srinivasan Iyer; Duc Le
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** Auto-regressive speech-text models are typically pre-trained on a large number of interleaved sequences of text tokens and raw speech encoded as speech tokens using vector quantization. These models have demonstrated state-of-the-art performance in speech-to-speech understanding and generation benchmarks, together with promising scaling laws, primarily enabled by the representational alignment between text and speech. Nevertheless, they suffer from shortcomings, partly owing to the disproportionately longer sequences of speech tokens in contrast to textual tokens. This results in a large compute imbalance between modalities during pre-training as well as during inference, and a potential hindrance to effectively aligning speech and text, ultimately translating to several orders of magnitude slower scaling laws. We introduce the Latent Speech-Text Transformer (LST), which makes pre-training speech-text models more data-efficient by dynamically and inexpensively aggregating speech tokens into latent speech patches. These patches serve as higher-level units that can either align with corresponding textual units to aid capability transfer or even encapsulate common speech sequences like silences to be more compute-efficient. We show that LST outperforms vanilla approaches on speech-to-speech as well as text-to-text benchmarks in both data- and compute-controlled settings, the former indicating more effective representational alignment and the latter indicating steeper scaling laws for speech-text models. On HellaSwag story completion, LST achieves 6.5% absolute gain in speech accuracy under compute-controlled training and 5.3% under data-controlled training, while also improving text performance. We will release our models, code, and the evaluation data to facilitate further research.
>
---
#### [new 052] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credits
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CreditDecoding，属于文本生成任务，旨在加速扩散大语言模型（dLLM）的并行解码过程。通过引入“Trace Credit”量化每个token的收敛潜力，融合当前logits与历史信息，减少冗余迭代，提升解码速度与鲁棒性。实验显示其在多个模型上显著加速且性能提升。**

- **链接: [http://arxiv.org/pdf/2510.06133v1](http://arxiv.org/pdf/2510.06133v1)**

> **作者:** Kangyu Wang; Zhiyun Jiang; Haibo Feng; Weijia Zhao; Lin Liu; Jianguo Li; Zhenzhong Lan; Weiyao Lin
>
> **备注:** 18 pages,8 figures,4 tables
>
> **摘要:** Diffusion large language models (dLLMs) generate text through iterative denoising steps, achieving parallel decoding by denoising only high-confidence positions at each step. However, existing approaches often repetitively remask tokens due to initially low confidence scores, leading to redundant iterations and limiting overall acceleration. Through the analysis of dLLM decoding traces, we observe that the model often determines the final prediction for a token several steps before the decoding step. To leverage this historical information and avoid redundant steps, we introduce the concept of Trace Credit, which quantifies each token's convergence potential by accumulating historical logits. Furthermore, we propose CreditDecoding, a training-free parallel decoding algorithm that accelerates the confidence convergence of correct but underconfident tokens by fusing current logits with Trace Credit. This process significantly reduces redundant iterations and enhances decoding robustness. On eight benchmarks, CreditDecoding achieves a 5.48 times speedup and a 0.48 performance improvement over LLaDA-8B-Instruct, and a 4.11 times speedup with a 0.15 performance improvement over LLaDA-MoE-Instruct. Importantly, CreditDecoding scales effectively to long sequences and is orthogonal to mainstream inference optimizations, making it a readily integrable and versatile solution.
>
---
#### [new 053] Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification
- **分类: cs.CL**

- **简介: 该论文属于专利分类任务，旨在解决LLM生成的推理存在错误而影响模型训练的问题。提出“自过滤蒸馏”方法，利用三个无监督信任指标评估LLM生成的推理，指导模型训练。在USPTO-2M数据集上验证了方法在准确性、稳定性和可解释性上的优势。**

- **链接: [http://arxiv.org/pdf/2510.05431v1](http://arxiv.org/pdf/2510.05431v1)**

> **作者:** Yoo Yongmin; Zhang Xu; Cao Longbing
>
> **摘要:** Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework specifically tailored for patent classification, which treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset, a widely used benchmark for patent classification, show that our method outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
>
---
#### [new 054] Automated Boilerplate: Prevalence and Quality of Contract Generators in the Context of Swiss Privacy Policies
- **分类: cs.CL**

- **简介: 该论文研究瑞士隐私政策中自动化合同生成器的普及与质量，评估其合规性影响。任务是分析法律文件合规性，解决企业合规难题。通过构建多语言数据集并使用GPT-5模型进行评估，发现使用生成器的隐私政策合规性更高。**

- **链接: [http://arxiv.org/pdf/2510.05860v1](http://arxiv.org/pdf/2510.05860v1)**

> **作者:** Luka Nenadic; David Rodriguez
>
> **备注:** 23 pages, 4 figures
>
> **摘要:** It has become increasingly challenging for firms to comply with a plethora of novel digital regulations. This is especially true for smaller businesses that often lack both the resources and know-how to draft complex legal documents. Instead of seeking costly legal advice from attorneys, firms may turn to cheaper alternative legal service providers such as automated contract generators. While these services have a long-standing presence, there is little empirical evidence on their prevalence and output quality. We address this gap in the context of a 2023 Swiss privacy law revision. To enable a systematic evaluation, we create and annotate a multilingual benchmark dataset that captures key compliance obligations under Swiss and EU privacy law. Using this dataset, we validate a novel GPT-5-based method for large-scale compliance assessment of privacy policies, allowing us to measure the impact of the revision. We observe compliance increases indicating an effect of the revision. Generators, explicitly referenced by 18% of local websites, are associated with substantially higher levels of compliance, with increases of up to 15 percentage points compared to privacy policies without generator use. These findings contribute to three debates: the potential of LLMs for cross-lingual legal analysis, the Brussels Effect of EU regulations, and, crucially, the role of automated tools in improving compliance and contractual quality.
>
---
#### [new 055] VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决KV缓存带来的高内存开销问题。通过提出VecInfer方法，采用低比特向量量化并抑制键缓存中的异常值，从而实现高效的推理性能。**

- **链接: [http://arxiv.org/pdf/2510.06175v1](http://arxiv.org/pdf/2510.06175v1)**

> **作者:** Dingyu Yao; Chenxu Yang; Zhengyang Tong; Zheng Lin; Wei Liu; Jian Luan; Weiping Wang
>
> **摘要:** The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexible representational capacity across bit-widths, they suffer severe performance degradation at ultra-low bit-widths due to key cache outliers that hinder effective codebook utilization. To address this challenge, we propose VecInfer, a novel VQ method for aggressive KV cache compression while enabling efficient inference. By applying smooth and Hadamard transformations, VecInfer suppresses outliers in the key cache, enabling the codebook to comprehensively cover the original data distribution and thereby reducing quantization difficulty. To facilitate efficient deployment, we design an optimized CUDA kernel that fuses computation with dequantization to minimize memory access overhead. Extensive evaluations demonstrate that VecInfer consistently outperforms existing quantization baselines across both long-context understanding and mathematical reasoning tasks. With only 2-bit quantization, VecInfer achieves performance comparable to full precision, while delivering up to $\mathbf{2.7\times}$ speedup in large-batch self-attention computation and $\mathbf{8.3\times}$ reduction in single-batch end-to-end latency on Llama-3.1-8B with a 196k sequence length.
>
---
#### [new 056] Residualized Similarity for Faithfully Explainable Authorship Verification
- **分类: cs.CL**

- **简介: 该论文属于作者验证任务，旨在解决现有模型准确性高但缺乏可解释性的问题。论文提出了残差相似性方法，结合神经网络与可解释特征，提升模型可解释性的同时保持性能，实现了预测结果的忠实解释。**

- **链接: [http://arxiv.org/pdf/2510.05362v1](http://arxiv.org/pdf/2510.05362v1)**

> **作者:** Peter Zeng; Pegah Alipoormolabashi; Jihu Mun; Gourab Dey; Nikita Soni; Niranjan Balasubramanian; Owen Rambow; H. Schwartz
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Responsible use of Authorship Verification (AV) systems not only requires high accuracy but also interpretable solutions. More importantly, for systems to be used to make decisions with real-world consequences requires the model's prediction to be explainable using interpretable features that can be traced to the original texts. Neural methods achieve high accuracies, but their representations lack direct interpretability. Furthermore, LLM predictions cannot be explained faithfully -- if there is an explanation given for a prediction, it doesn't represent the reasoning process behind the model's prediction. In this paper, we introduce Residualized Similarity (RS), a novel method that supplements systems using interpretable features with a neural network to improve their performance while maintaining interpretability. Authorship verification is fundamentally a similarity task, where the goal is to measure how alike two documents are. The key idea is to use the neural network to predict a similarity residual, i.e. the error in the similarity predicted by the interpretable system. Our evaluation across four datasets shows that not only can we match the performance of state-of-the-art authorship verification models, but we can show how and to what degree the final prediction is faithful and interpretable.
>
---
#### [new 057] Collaborative and Proactive Management of Task-Oriented Conversations
- **分类: cs.CL**

- **简介: 该论文属于任务型对话系统（TOD）研究，旨在解决现有系统在目标感知规划上的不足。作者基于信息状态方法，构建了一个包含中间信息的对话管理模型，并利用大语言模型（LLM）实现数据库查询与偏好匹配。实验表明该方法在MultiWOZ数据集上提升了任务完成的效率与成功率。**

- **链接: [http://arxiv.org/pdf/2510.05110v1](http://arxiv.org/pdf/2510.05110v1)**

> **作者:** Arezoo Saedi; Afsaneh Fatemi; Mohammad Ali Nematbakhsh; Sophie Rosset; Anne Vilnat
>
> **摘要:** Task oriented dialogue systems (TOD) complete particular tasks based on user preferences across natural language interactions. Considering the impressive performance of large language models (LLMs) in natural language processing (NLP) tasks, most of the latest TODs are centered on LLMs. While proactive planning is crucial for task completion, many existing TODs overlook effective goal-aware planning. This paper creates a model for managing task-oriented conversations, conceptualized centered on the information state approach to dialogue management. The created model incorporated constructive intermediate information in planning. Initially, predefined slots and text part informational components are created to model user preferences. Investigating intermediate information, critical circumstances are identified. Informational components corresponding to these circumstances are created. Possible configurations for these informational components lead to limited information states. Then, dialogue moves, which indicate movement between these information states and the procedures that must be performed in the movements, are created. Eventually, the update strategy is constructed. The created model is implemented leveraging in-context learning of LLMs. In this model, database queries are created centered on indicated predefined slots and the order of retrieved entities is indicated centered on text part. This mechanism enables passing the whole corresponding entities to the preferences in the order of congruency. Evaluations exploiting the complete test conversations of MultiWOZ, with no more than a domain in a conversation, illustrate maximal inform and success, and improvement compared with previous methods.
>
---
#### [new 058] Hallucination is Inevitable for LLMs with the Open World Assumption
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨了大语言模型（LLMs）在开放世界假设下的“幻觉”问题，认为幻觉是模型泛化能力的体现，而非单纯缺陷。任务是分析幻觉的不可避免性，提出其分类，并指出在开放环境下应容忍而非彻底消除幻觉，以实现与人类智能的兼容。**

- **链接: [http://arxiv.org/pdf/2510.05116v1](http://arxiv.org/pdf/2510.05116v1)**

> **作者:** Bowen Xu
>
> **摘要:** Large Language Models (LLMs) exhibit impressive linguistic competence but also produce inaccurate or fabricated outputs, often called ``hallucinations''. Engineering approaches usually regard hallucination as a defect to be minimized, while formal analyses have argued for its theoretical inevitability. Yet both perspectives remain incomplete when considering the conditions required for artificial general intelligence (AGI). This paper reframes ``hallucination'' as a manifestation of the generalization problem. Under the Closed World assumption, where training and test distributions are consistent, hallucinations may be mitigated. Under the Open World assumption, however, where the environment is unbounded, hallucinations become inevitable. This paper further develops a classification of hallucination, distinguishing cases that may be corrected from those that appear unavoidable under open-world conditions. On this basis, it suggests that ``hallucination'' should be approached not merely as an engineering defect but as a structural feature to be tolerated and made compatible with human intelligence.
>
---
#### [new 059] Towards Structured Knowledge: Advancing Triple Extraction from Regional Trade Agreements using Large Language Models
- **分类: cs.CL; cs.CE; cs.IR; cs.LG**

- **简介: 该论文属于信息抽取任务，旨在解决从区域贸易协定文本中提取结构化知识（三元组）的问题。论文使用大语言模型（如Llama 3.1），探索零样本、一样本和少样本提示技术，结合正负样例，评估其在贸易协议文本中的三元组抽取效果，强调语言模型在经济领域应用的潜力与挑战。**

- **链接: [http://arxiv.org/pdf/2510.05121v1](http://arxiv.org/pdf/2510.05121v1)**

> **作者:** Durgesh Nandini; Rebekka Koch; Mirco Schoenfeld
>
> **摘要:** This study investigates the effectiveness of Large Language Models (LLMs) for the extraction of structured knowledge in the form of Subject-Predicate-Object triples. We apply the setup for the domain of Economics application. The findings can be applied to a wide range of scenarios, including the creation of economic trade knowledge graphs from natural language legal trade agreement texts. As a use case, we apply the model to regional trade agreement texts to extract trade-related information triples. In particular, we explore the zero-shot, one-shot and few-shot prompting techniques, incorporating positive and negative examples, and evaluate their performance based on quantitative and qualitative metrics. Specifically, we used Llama 3.1 model to process the unstructured regional trade agreement texts and extract triples. We discuss key insights, challenges, and potential future directions, emphasizing the significance of language models in economic applications.
>
---
#### [new 060] Catalog-Native LLM: Speaking Item-ID Dialect with Less Entanglement for Recommendation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于推荐系统任务，旨在解决协同过滤与大语言模型（LLM）融合困难的问题。作者提出IDIOMoE模型，将物品交互历史视为语言方言，通过分离文本与物品专家网络，避免模态干扰，实现推荐性能提升并保持语言理解能力。**

- **链接: [http://arxiv.org/pdf/2510.05125v1](http://arxiv.org/pdf/2510.05125v1)**

> **作者:** Reza Shirkavand; Xiaokai Wei; Chen Wang; Zheng Hui; Heng Huang; Michelle Gong
>
> **摘要:** While collaborative filtering delivers predictive accuracy and efficiency, and Large Language Models (LLMs) enable expressive and generalizable reasoning, modern recommendation systems must bring these strengths together. Growing user expectations, such as natural-language queries and transparent explanations, further highlight the need for a unified approach. However, doing so is nontrivial. Collaborative signals are often token-efficient but semantically opaque, while LLMs are semantically rich but struggle to model implicit user preferences when trained only on textual inputs. This paper introduces Item-ID + Oral-language Mixture-of-Experts Language Model (IDIOMoE), which treats item interaction histories as a native dialect within the language space, enabling collaborative signals to be understood in the same way as natural language. By splitting the Feed Forward Network of each block of a pretrained LLM into a separate text expert and an item expert with token-type gating, our method avoids destructive interference between text and catalog modalities. IDIOMoE demonstrates strong recommendation performance across both public and proprietary datasets, while preserving the text understanding of the pretrained model.
>
---
#### [new 061] Revisiting Long-context Modeling from Context Denoising Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长上下文模型易受无关信息干扰的问题。通过提出上下文去噪训练策略，增强模型对关键信息的关注，从而提升预测效果。实验表明，该方法使8B模型表现接近GPT-4o水平。**

- **链接: [http://arxiv.org/pdf/2510.05862v1](http://arxiv.org/pdf/2510.05862v1)**

> **作者:** Zecheng Tang; Baibei Ji; Juntao Li; Lijun Wu; Haijia Gui; Min Zhang
>
> **摘要:** Long-context models (LCMs) have demonstrated great potential in processing long sequences, facilitating many real-world applications. The success of LCMs can be attributed to their ability to locate implicit critical information within the context for further prediction. However, recent research reveals that LCMs are often susceptible to contextual noise, i.e., irrelevant tokens, that can mislead model attention. In this paper, we conduct a fine-grained analysis of the context noise and propose an effective metric, the Integrated Gradient (IG) score, to detect and quantify the noise information within the context. Our findings reveal that even simple mitigation of detected context noise can substantially boost the model's attention on critical tokens and benefit subsequent predictions. Building on this insight, we propose Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. Extensive experiments across four tasks, under both context window scaling and long-context alignment settings, demonstrate the superiority of CDT. Notably, when trained with CDT, an open-source 8B model can achieve performance (50.92) comparable to GPT-4o (51.00).
>
---
#### [new 062] Let it Calm: Exploratory Annealed Decoding for Verifiable Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习与语言模型生成任务，旨在解决RLVR中探索与利用的平衡问题。现有方法难以兼顾样本质量与训练稳定性。作者提出Exploratory Annealed Decoding（EAD），通过在生成过程中动态降低采样温度，前期探索、后期利用，提升样本效率与模型表现。方法轻量且通用，适用于多种RLVR算法与模型规模。**

- **链接: [http://arxiv.org/pdf/2510.05251v1](http://arxiv.org/pdf/2510.05251v1)**

> **作者:** Chenghao Yang; Lin Gui; Chenxiao Yang; Victor Veitch; Lizhu Zhang; Zhuokai Zhao
>
> **备注:** Codebase: https://github.com/yangalan123/EAD-RLVR
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) is a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs), yet its success hinges on effective exploration. An ideal exploration strategy must navigate two fundamental challenges: it must preserve sample quality while also ensuring training stability. While standard fixed-temperature sampling is simple, it struggles to balance these competing demands, as high temperatures degrade sample quality and low temperatures limit discovery. In this work, we propose a simpler and more effective strategy, Exploratory Annealed Decoding (EAD), grounded in the insight that exploration is most impactful on early tokens which define a sequence's semantic direction. EAD implements an intuitive **explore-at-the-beginning, exploit-at-the-end** strategy by annealing the sampling temperature from high to low during generation. This dynamic schedule encourages meaningful, high-level diversity at the start, then gradually lowers the temperature to preserve sample quality and keep the sampling distribution close to the target policy, which is essential for stable training. We demonstrate that EAD is a lightweight, plug-and-play method that significantly improves sample efficiency, consistently outperforming fixed-temperature sampling across various RLVR algorithms and model sizes. Our work suggests that aligning exploration with the natural dynamics of sequential generation offers a robust path to improving LLM reasoning.
>
---
#### [new 063] UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态检索增强生成（MM-RAG）任务，旨在解决当前评估方法割裂、无法反映真实文档场景的问题。论文构建了UniDoc-Bench，首个大规模、现实文档基准，包含70k PDF页面和1,600多模态问答对，支持统一比较四种范式，揭示多模态融合的优势与当前嵌入方法的不足。**

- **链接: [http://arxiv.org/pdf/2510.03663v1](http://arxiv.org/pdf/2510.03663v1)**

> **作者:** Xiangyu Peng; Cab Qin; Zeyuan Chen; Ran Xu; Caiming Xiong; Chien-Sheng Wu
>
> **摘要:** Multimodal retrieval-augmented generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented, focusing on either text or images in isolation or on simplified multimodal setups that fail to capture document-centric multimodal use cases. In this paper, we introduce UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across eight domains. Our pipeline extracts and links evidence from text, tables, and figures, then generates 1,600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. To ensure reliability, 20% of QA pairs are validated by multiple annotators and expert adjudication. UniDoc-Bench supports apples-to-apples comparison across four paradigms: (1) text-only, (2) image-only, (3) multimodal text-image fusion, and (4) multimodal joint retrieval -- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. Our experiments show that multimodal text-image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding-based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines.
>
---
#### [new 064] Improving Metacognition and Uncertainty Communication in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型在不确定性表达方面的能力。研究通过监督微调，改进模型在不同任务和领域中对答案的置信度估计与比较能力，发现多任务训练效果更佳，使不确定性表达更具泛化性和准确性。**

- **链接: [http://arxiv.org/pdf/2510.05126v1](http://arxiv.org/pdf/2510.05126v1)**

> **作者:** Mark Steyvers; Catarina Belem; Padhraic Smyth
>
> **摘要:** Large language models (LLMs) are increasingly used in decision-making contexts, but when they present answers without signaling low confidence, users may unknowingly act on erroneous outputs. While prior work shows that LLMs maintain internal uncertainty signals, their explicit verbalized confidence is typically miscalibrated and poorly discriminates between correct and incorrect answers. Across two types of LLMs, we investigate whether supervised finetuning can improve models' ability to communicate uncertainty and whether such improvements generalize across tasks and domains. We finetune the LLMs on datasets spanning general knowledge, mathematics, and open-ended trivia, and evaluate two metacognitive tasks: (1) single-question confidence estimation, where the model assigns a numeric certainty to its answer, and (2) pairwise confidence comparison, where the model selects which of two answers it is more likely to have correct. We assess generalization to unseen domains, including medical and legal reasoning. Results show that finetuning improves calibration (alignment between stated confidence and accuracy) and discrimination (higher confidence for correct vs. incorrect responses) within and across domains, while leaving accuracy unchanged. However, improvements are task-specific: training on single-question calibration does not transfer to pairwise comparison, and vice versa. In contrast, multitask finetuning on both forms of metacognition yields broader gains, producing lower calibration error and stronger discrimination in out-of-domain evaluations. These results show that while uncertainty communication in LLMs is trainable and generalizable, different metacognitive skills do not naturally reinforce one another and must be developed together through multitask training.
>
---
#### [new 065] Distributional Semantics Tracing: A Framework for Explaining Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文属于自然语言处理任务，旨在解释大语言模型的幻觉问题。通过提出分布语义追踪框架，定位模型中导致幻觉的关键层，并揭示其机制源于联想路径与语境路径的冲突，从而提供幻觉生成的可解释性分析。**

- **链接: [http://arxiv.org/pdf/2510.06107v1](http://arxiv.org/pdf/2510.06107v1)**

> **作者:** Gagan Bhatia; Somayajulu G Sripada; Kevin Allan; Jacobo Azcona
>
> **摘要:** Large Language Models (LLMs) are prone to hallucination, the generation of plausible yet factually incorrect statements. This work investigates the intrinsic, architectural origins of this failure mode through three primary contributions.First, to enable the reliable tracing of internal semantic failures, we propose \textbf{Distributional Semantics Tracing (DST)}, a unified framework that integrates established interpretability techniques to produce a causal map of a model's reasoning, treating meaning as a function of context (distributional semantics). Second, we pinpoint the model's layer at which a hallucination becomes inevitable, identifying a specific \textbf{commitment layer} where a model's internal representations irreversibly diverge from factuality. Third, we identify the underlying mechanism for these failures. We observe a conflict between distinct computational pathways, which we interpret using the lens of dual-process theory: a fast, heuristic \textbf{associative pathway} (akin to System 1) and a slow, deliberate \textbf{contextual pathway} (akin to System 2), leading to predictable failure modes such as \textit{Reasoning Shortcut Hijacks}. Our framework's ability to quantify the coherence of the contextual pathway reveals a strong negative correlation ($\rho = -0.863$) with hallucination rates, implying that these failures are predictable consequences of internal semantic weakness. The result is a mechanistic account of how, when, and why hallucinations occur within the Transformer architecture.
>
---
#### [new 066] The African Languages Lab: A Collaborative Approach to Advancing Low-Resource African NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理（NLP）任务，旨在解决非洲语言在计算语言学中严重缺乏资源的问题。作者通过构建大规模多模态数据集、优化模型性能，并培养本地研究人才，推动低资源非洲语言的技术发展。**

- **链接: [http://arxiv.org/pdf/2510.05644v1](http://arxiv.org/pdf/2510.05644v1)**

> **作者:** Sheriff Issaka; Keyi Wang; Yinka Ajibola; Oluwatumininu Samuel-Ipaye; Zhaoyi Zhang; Nicte Aguillon Jimenez; Evans Kofi Agyei; Abraham Lin; Rohan Ramachandran; Sadick Abdul Mumin; Faith Nchifor; Mohammed Shuraim; Lieqi Liu; Erick Rosas Gonzalez; Sylvester Kpei; Jemimah Osei; Carlene Ajeneza; Persis Boateng; Prisca Adwoa Dufie Yeboah; Saadia Gabriel
>
> **摘要:** Despite representing nearly one-third of the world's languages, African languages remain critically underserved by modern NLP technologies, with 88\% classified as severely underrepresented or completely ignored in computational linguistics. We present the African Languages Lab (All Lab), a comprehensive research initiative that addresses this technological gap through systematic data collection, model development, and capacity building. Our contributions include: (1) a quality-controlled data collection pipeline, yielding the largest validated African multi-modal speech and text dataset spanning 40 languages with 19 billion tokens of monolingual text and 12,628 hours of aligned speech data; (2) extensive experimental validation demonstrating that our dataset, combined with fine-tuning, achieves substantial improvements over baseline models, averaging +23.69 ChrF++, +0.33 COMET, and +15.34 BLEU points across 31 evaluated languages; and (3) a structured research program that has successfully mentored fifteen early-career researchers, establishing sustainable local capacity. Our comparative evaluation against Google Translate reveals competitive performance in several languages while identifying areas that require continued development.
>
---
#### [new 067] MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.MA**

- **简介: 论文提出MADS框架，通过多智能体自博弈生成多样化说服性对话数据，旨在解决行业中小模型训练数据不足、冷启动评估难等问题。应用于营销场景后，有效提升小模型说服能力，带来业务价值。**

- **链接: [http://arxiv.org/pdf/2510.05124v1](http://arxiv.org/pdf/2510.05124v1)**

> **作者:** Mingjin Li; Yu Liu; Huayi Liu; Xiang Ye; Chao Jiang; Hongguang Zhang
>
> **备注:** work in progress
>
> **摘要:** We propose MADS (Multi-Agent Dialogue Simulation), a scalable framework for generating persuasive multi-turn dialogues via agent self-play. MADS employs three coordinated agents: User Agents simulating diverse persona-driven behaviors, a Dialog Agent executing task-oriented persuasion strategies and an Optimization Agent evaluating and refining dialogue outcomes. We further validate its effectiveness through users' Chain-of-Attitude (CoA) modeling and dedicated LLMs' persuasion assessment. This approach enables low-cost generation of training data without human annotation, addressing key industry challenges such as lack of user data, cold-start evaluation difficulties, and prompt inefficiency. Applied to a real-world marketing scenario, MADS significantly improved the persuasion capacity of small LLMs, increasing the organic traffic conversion rate by 22.4\% (from 1.83\% to 2.24\%) , demonstrating clear business value.
>
---
#### [new 068] Advancing Automated Spatio-Semantic Analysis in Picture Description Using Language Models
- **分类: cs.CL; cs.CV; eess.AS**

- **简介: 该论文属于自然语言处理与认知评估交叉任务，旨在解决图片描述中认知语言障碍自动分析问题。现有方法忽略视觉叙事路径，该研究利用微调BERT模型自动提取并排序内容信息单元（CIU），实现高效评估认知障碍，相关模型与工具已开源。**

- **链接: [http://arxiv.org/pdf/2510.05128v1](http://arxiv.org/pdf/2510.05128v1)**

> **作者:** Si-Ioi Ng; Pranav S. Ambadi; Kimberly D. Mueller; Julie Liss; Visar Berisha
>
> **摘要:** Current methods for automated assessment of cognitive-linguistic impairment via picture description often neglect the visual narrative path - the sequence and locations of elements a speaker described in the picture. Analyses of spatio-semantic features capture this path using content information units (CIUs), but manual tagging or dictionary-based mapping is labor-intensive. This study proposes a BERT-based pipeline, fine tuned with binary cross-entropy and pairwise ranking loss, for automated CIU extraction and ordering from the Cookie Theft picture description. Evaluated by 5-fold cross-validation, it achieves 93% median precision, 96% median recall in CIU detection, and 24% sequence error rates. The proposed method extracts features that exhibit strong Pearson correlations with ground truth, surpassing the dictionary-based baseline in external validation. These features also perform comparably to those derived from manual annotations in evaluating group differences via ANCOVA. The pipeline is shown to effectively characterize visual narrative paths for cognitive impairment assessment, with the implementation and models open-sourced to public.
>
---
#### [new 069] Adaptive and Multi-Source Entity Matching for Name Standardization of Astronomical Observation Facilities
- **分类: cs.CL; astro-ph.IM**

- **简介: 该论文属于实体匹配任务，旨在解决天文观测设施名称标准化问题。通过多源数据（如Wikidata）提取实体，结合NLP技术和LLM模型进行实体比较与映射验证，最终生成标准化名称集，用于提升天文数据互操作性与一致性。**

- **链接: [http://arxiv.org/pdf/2510.05744v1](http://arxiv.org/pdf/2510.05744v1)**

> **作者:** Liza Fretel; Baptiste Cecconi; Laura Debisschop
>
> **备注:** Accepted in Ontology Matching 2025 conference proceedings
>
> **摘要:** This ongoing work focuses on the development of a methodology for generating a multi-source mapping of astronomical observation facilities. To compare two entities, we compute scores with adaptable criteria and Natural Language Processing (NLP) techniques (Bag-of-Words approaches, sequential approaches, and surface approaches) to map entities extracted from eight semantic artifacts, including Wikidata and astronomy-oriented resources. We utilize every property available, such as labels, definitions, descriptions, external identifiers, and more domain-specific properties, such as the observation wavebands, spacecraft launch dates, funding agencies, etc. Finally, we use a Large Language Model (LLM) to accept or reject a mapping suggestion and provide a justification, ensuring the plausibility and FAIRness of the validated synonym pairs. The resulting mapping is composed of multi-source synonym sets providing only one standardized label per entity. Those mappings will be used to feed our Name Resolver API and will be integrated into the International Virtual Observatory Alliance (IVOA) Vocabularies and the OntoPortal-Astro platform.
>
---
#### [new 070] Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言模型中跨语言迁移效果差的问题。通过提出“并行分词器”框架，单独训练单语分词器并利用双语词典对齐词汇，使语义相同的词有统一表示。该方法提升了低资源语言的跨语言迁移效果，在情感分析、句子相似度等任务上优于传统多语言模型。**

- **链接: [http://arxiv.org/pdf/2510.06128v1](http://arxiv.org/pdf/2510.06128v1)**

> **作者:** Muhammad Dehan Al Kautsar; Fajri Koto
>
> **备注:** 18 pages, 25 tables, 7 figures
>
> **摘要:** Tokenization defines the foundation of multilingual language models by determining how words are represented and shared across languages. However, existing methods often fail to support effective cross-lingual transfer because semantically equivalent words are assigned distinct embeddings. For example, "I eat rice" in English and "Ina cin shinkafa" in Hausa are typically mapped to different vocabulary indices, preventing shared representations and limiting cross-lingual generalization. We introduce parallel tokenizers. This new framework trains tokenizers monolingually and then aligns their vocabularies exhaustively using bilingual dictionaries or word-to-word translation, ensuring consistent indices for semantically equivalent words. This alignment enforces a shared semantic space across languages while naturally improving fertility balance. To assess their effectiveness, we pretrain a transformer encoder from scratch on thirteen low-resource languages and evaluate it on sentiment analysis, hate speech detection, emotion classification, and sentence embedding similarity. Across all tasks, models trained with parallel tokenizers outperform conventional multilingual baselines, confirming that rethinking tokenization is essential for advancing multilingual representation learning--especially in low-resource settings.
>
---
#### [new 071] DACP: Domain-Adaptive Continual Pre-Training of Large Language Models for Phone Conversation Summarization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本摘要任务，旨在解决大语言模型在特定领域（如电话对话）摘要效果不佳的问题。作者通过持续预训练（continual pre-training）方法，利用无标签对话数据提升模型摘要能力，并验证了该方法在多领域的效果与鲁棒性，同时提供了数据选择策略的实践指导。**

- **链接: [http://arxiv.org/pdf/2510.05858v1](http://arxiv.org/pdf/2510.05858v1)**

> **作者:** Xue-Yong Fu; Elena Khasanova; Md Tahmid Rahman Laskar; Harsh Saini; Shashi Bhushan TN
>
> **备注:** Accepted to the NewSumm Workshop at EMNLP 2025
>
> **摘要:** Large language models (LLMs) have achieved impressive performance in text summarization, yet their performance often falls short when applied to specialized domains %or conversational data that differ from their original pre-training distribution. While fine-tuning can improve summarization quality, it typically relies on costly and scarce high-quality labeled data. In this work, we explore continual pre-training as a scalable, self-supervised approach to adapt LLMs for downstream summarization tasks, particularly in the context of noisy real-world conversation transcripts. We conduct extensive experiments using large-scale, unlabeled business conversation data to investigate whether continual pre-training enhances model capabilities in conversational summarization. Our results demonstrate that continual pre-training yields substantial gains in both in-domain and out-of-domain summarization benchmarks, while maintaining strong generalization and robustness. We also analyze the effects of data selection strategies, providing practical guidelines for applying continual pre-training in summarization-focused industrial applications.
>
---
#### [new 072] Submodular Context Partitioning and Compression for In-Context Learning-short paper
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在上下文学习中因输入复杂度高导致的效率与性能问题。论文提出Sub-CP方法，通过子模目标实现块感知的上下文选择，优化块间多样性和块内一致性，从而提升上下文学习效果。**

- **链接: [http://arxiv.org/pdf/2510.05130v1](http://arxiv.org/pdf/2510.05130v1)**

> **作者:** Shaoyi Zheng; Canyu Zhang; Tianyi Zhou; Shengjie Wang
>
> **摘要:** In-context learning (ICL) enables efficient few-shot learning in large language models (LLMs) without training, but suffers from the quadratic input complexity of transformers, limiting the maximum number of exemplars. While various efficient ICL approaches partition the context into blocks to process (e.g., ensembling, compression, cross-attention), they often ignore the information redundancy or under-representation caused by different partition strategies, leading to suboptimal performance. To tackle this problem, we propose Sub-CP, a block-aware context selection framework that leverages submodular objectives to control block diversity. Sub-CP supports a flexible spectrum of selection strategies, allowing each block to range from globally diverse to locally coherent. This allows fine-grained control over semantic structure while enabling precomputation. Extensive experiments across diverse tasks on multiple datasets show that Sub-CP consistently improves performance across model scales.
>
---
#### [new 073] RAG Makes Guardrails Unsafe? Investigating Robustness of Guardrails under RAG-style Contexts
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了在检索增强生成（RAG）背景下，基于大语言模型（LLM）的防护机制（guardrails）的鲁棒性问题。任务是评估这些防护模型在面对上下文扰动时的可靠性。论文发现，插入良性文档会改变防护判断，导致约11%的输入和8%的输出防护失效，揭示了当前防护机制在上下文鲁棒性上的不足，并建议改进训练和评估方法。**

- **链接: [http://arxiv.org/pdf/2510.05310v1](http://arxiv.org/pdf/2510.05310v1)**

> **作者:** Yining She; Daniel W. Peterson; Marianne Menglin Liu; Vikas Upadhyay; Mohammad Hossein Chaghazardi; Eunsuk Kang; Dan Roth
>
> **摘要:** With the increasing adoption of large language models (LLMs), ensuring the safety of LLM systems has become a pressing concern. External LLM-based guardrail models have emerged as a popular solution to screen unsafe inputs and outputs, but they are themselves fine-tuned or prompt-engineered LLMs that are vulnerable to data distribution shifts. In this paper, taking Retrieval Augmentation Generation (RAG) as a case study, we investigated how robust LLM-based guardrails are against additional information embedded in the context. Through a systematic evaluation of 3 Llama Guards and 2 GPT-oss models, we confirmed that inserting benign documents into the guardrail context alters the judgments of input and output guardrails in around 11% and 8% of cases, making them unreliable. We separately analyzed the effect of each component in the augmented context: retrieved documents, user query, and LLM-generated response. The two mitigation methods we tested only bring minor improvements. These results expose a context-robustness gap in current guardrails and motivate training and evaluation protocols that are robust to retrieval and query composition.
>
---
#### [new 074] Cross-Lingual Mental Health Ontologies for Indian Languages: Bridging Patient Expression and Clinical Understanding through Explainable AI and Human-in-the-Loop Validation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与心理健康结合的跨语言任务，旨在解决印度多语言背景下患者表达与临床理解之间的文化及语言鸿沟。论文构建了跨语言患者压力表达图（CL-PDE），通过图方法对印度语言中的心理压力表达进行建模、对齐，并与临床术语关联，提升心理健康NLP工具的文化适应性与包容性。**

- **链接: [http://arxiv.org/pdf/2510.05387v1](http://arxiv.org/pdf/2510.05387v1)**

> **作者:** Ananth Kandala; Ratna Kandala; Akshata Kishore Moharir; Niva Manchanda; Sunaina Singh
>
> **摘要:** Mental health communication in India is linguistically fragmented, culturally diverse, and often underrepresented in clinical NLP. Current health ontologies and mental health resources are dominated by diagnostic frameworks centered on English or Western culture, leaving a gap in representing patient distress expressions in Indian languages. We propose cross-linguistic graphs of patient stress expressions (CL-PDE), a framework for building cross-lingual mental health ontologies through graph-based methods that capture culturally embedded expressions of distress, align them across languages, and link them with clinical terminology. Our approach addresses critical gaps in healthcare communication by grounding AI systems in culturally valid representations, allowing more inclusive and patient-centric NLP tools for mental health care in multilingual contexts.
>
---
#### [new 075] Linguistic Characteristics of AI-Generated Text: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在分析人工智能生成文本的语言特征。论文综述了现有研究，从语言学角度总结了AI生成文本的特点，如风格更正式、词汇多样性较低等，并指出了研究在语言和模型选择上的局限性，以及提示敏感性问题需进一步探索。**

- **链接: [http://arxiv.org/pdf/2510.05136v1](http://arxiv.org/pdf/2510.05136v1)**

> **作者:** Luka Terčon; Kaja Dobrovoljc
>
> **备注:** 26 pages, 5 figures
>
> **摘要:** Large language models (LLMs) are solidifying their position in the modern world as effective tools for the automatic generation of text. Their use is quickly becoming commonplace in fields such as education, healthcare, and scientific research. There is a growing need to study the linguistic features present in AI-generated text, as the increasing presence of such texts has profound implications in various disciplines such as corpus linguistics, computational linguistics, and natural language processing. Many observations have already been made, however a broader synthesis of the findings made so far is required to provide a better understanding of the topic. The present survey paper aims to provide such a synthesis of extant research. We categorize the existing works along several dimensions, including the levels of linguistic description, the models included, the genres analyzed, the languages analyzed, and the approach to prompting. Additionally, the same scheme is used to present the findings made so far and expose the current trends followed by researchers. Among the most-often reported findings is the observation that AI-generated text is more likely to contain a more formal and impersonal style, signaled by the increased presence of nouns, determiners, and adpositions and the lower reliance on adjectives and adverbs. AI-generated text is also more likely to feature a lower lexical diversity, a smaller vocabulary size, and repetitive text. Current research, however, remains heavily concentrated on English data and mostly on text generated by the GPT model family, highlighting the need for broader cross-linguistic and cross-model investigation. In most cases authors also fail to address the issue of prompt sensitivity, leaving much room for future studies that employ multiple prompt wordings in the text generation phase.
>
---
#### [new 076] BanglaTalk: Towards Real-Time Speech Assistance for Bengali Regional Dialects
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语音处理任务，旨在解决孟加拉语方言多样性导致的实时语音助手缺失问题。作者提出了BanglaTalk系统，采用客户端-服务器架构和RTP协议，实现低延迟通信。通过微调IndicWav2Vec模型构建方言感知的ASR系统BRDialect，提升了识别性能，并在低带宽环境下保持实时交互性。**

- **链接: [http://arxiv.org/pdf/2510.06188v1](http://arxiv.org/pdf/2510.06188v1)**

> **作者:** Jakir Hasan; Shubhashis Roy Dipta
>
> **摘要:** Real-time speech assistants are becoming increasingly popular for ensuring improved accessibility to information. Bengali, being a low-resource language with a high regional dialectal diversity, has seen limited progress in developing such systems. Existing systems are not optimized for real-time use and focus only on standard Bengali. In this work, we present BanglaTalk, the first real-time speech assistance system for Bengali regional dialects. BanglaTalk follows the client-server architecture and uses the Real-time Transport Protocol (RTP) to ensure low-latency communication. To address dialectal variation, we introduce a dialect-aware ASR system, BRDialect, developed by fine-tuning the IndicWav2Vec model in ten Bengali regional dialects. It outperforms the baseline ASR models by 12.41-33.98% on the RegSpeech12 dataset. Furthermore, BanglaTalk can operate at a low bandwidth of 24 kbps while maintaining an average end-to-end delay of 4.9 seconds. Low bandwidth usage and minimal end-to-end delay make the system both cost-effective and interactive for real-time use cases, enabling inclusive and accessible speech technology for the diverse community of Bengali speakers.
>
---
#### [new 077] Rationale-Augmented Retrieval with Constrained LLM Re-Ranking for Task Discovery
- **分类: cs.CL; cs.AI**

- **简介: 论文提出了一种结合语义搜索与大语言模型重排序的混合检索系统，用于解决新员工在GoEngage平台上查找任务模块时面临的术语障碍和搜索局限性，提升搜索准确性与系统可演进性。**

- **链接: [http://arxiv.org/pdf/2510.05131v1](http://arxiv.org/pdf/2510.05131v1)**

> **作者:** Bowen Wei
>
> **摘要:** Head Start programs utilizing GoEngage face significant challenges when new or rotating staff attempt to locate appropriate Tasks (modules) on the platform homepage. These difficulties arise from domain-specific jargon (e.g., IFPA, DRDP), system-specific nomenclature (e.g., Application Pool), and the inherent limitations of lexical search in handling typos and varied word ordering. We propose a pragmatic hybrid semantic search system that synergistically combines lightweight typo-tolerant lexical retrieval, embedding-based vector similarity, and constrained large language model (LLM) re-ranking. Our approach leverages the organization's existing Task Repository and Knowledge Base infrastructure while ensuring trustworthiness through low false-positive rates, evolvability to accommodate terminological changes, and economic efficiency via intelligent caching, shortlist generation, and graceful degradation mechanisms. We provide a comprehensive framework detailing required resources, a phased implementation strategy with concrete milestones, an offline evaluation protocol utilizing curated test cases (Hit@K, Precision@K, Recall@K, MRR), and an online measurement methodology incorporating query success metrics, zero-result rates, and dwell-time proxies.
>
---
#### [new 078] The fragility of "cultural tendencies" in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在验证大语言模型是否存在文化倾向性。论文复现并扩展了前人实验，使用更多模型和测试项，发现提示语言对输出影响极小，质疑“文化倾向”是模型的稳定特征，认为其是特定模型和任务设计的脆弱产物。**

- **链接: [http://arxiv.org/pdf/2510.05869v1](http://arxiv.org/pdf/2510.05869v1)**

> **作者:** Kun Sun; Rong Wang
>
> **摘要:** In a recent study, Lu, Song, and Zhang (2025) (LSZ) propose that large language models (LLMs), when prompted in different languages, display culturally specific tendencies. They report that the two models (i.e., GPT and ERNIE) respond in more interdependent and holistic ways when prompted in Chinese, and more independent and analytic ways when prompted in English. LSZ attribute these differences to deep-seated cultural patterns in the models, claiming that prompt language alone can induce substantial cultural shifts. While we acknowledge the empirical patterns they observed, we find their experiments, methods, and interpretations problematic. In this paper, we critically re-evaluate the methodology, theoretical framing, and conclusions of LSZ. We argue that the reported "cultural tendencies" are not stable traits but fragile artifacts of specific models and task design. To test this, we conducted targeted replications using a broader set of LLMs and a larger number of test items. Our results show that prompt language has minimal effect on outputs, challenging LSZ's claim that these models encode grounded cultural beliefs.
>
---
#### [new 079] WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives
- **分类: cs.CL**

- **简介: 该论文属于信息检索与自然语言处理任务，旨在解决历史天气档案中社会脆弱性与韧性信息提取困难的问题。作者构建了WeatherArchive-Bench基准，包含检索与评估两个任务，用于测试系统从历史文本中提取天气事件及其社会影响的能力。**

- **链接: [http://arxiv.org/pdf/2510.05336v1](http://arxiv.org/pdf/2510.05336v1)**

> **作者:** Yongan Yu; Xianda Du; Qingchen Hu; Jiahao Liang; Jingwei Ni; Dan Qiang; Kaiyu Huang; Grant McKenzie; Renee Sieber; Fengran Mo
>
> **摘要:** Historical archives on weather events are collections of enduring primary source records that offer rich, untapped narratives of how societies have experienced and responded to extreme weather events. These qualitative accounts provide insights into societal vulnerability and resilience that are largely absent from meteorological records, making them valuable for climate scientists to understand societal responses. However, their vast scale, noisy digitized quality, and archaic language make it difficult to transform them into structured knowledge for climate research. To address this challenge, we introduce WeatherArchive-Bench, the first benchmark for evaluating retrieval-augmented generation (RAG) systems on historical weather archives. WeatherArchive-Bench comprises two tasks: WeatherArchive-Retrieval, which measures a system's ability to locate historically relevant passages from over one million archival news segments, and WeatherArchive-Assessment, which evaluates whether Large Language Models (LLMs) can classify societal vulnerability and resilience indicators from extreme weather narratives. Extensive experiments across sparse, dense, and re-ranking retrievers, as well as a diverse set of LLMs, reveal that dense retrievers often fail on historical terminology, while LLMs frequently misinterpret vulnerability and resilience concepts. These findings highlight key limitations in reasoning about complex societal indicators and provide insights for designing more robust climate-focused RAG systems from archival contexts. The constructed dataset and evaluation framework are publicly available at https://anonymous.4open.science/r/WeatherArchive-Bench/.
>
---
#### [new 080] Evaluating The Impact of Stimulus Quality in Investigations of LLM Language Performance
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决刺激质量对大语言模型（LLM）语言表现评估的影响。为探究刺激质量如何影响模型表现，研究者通过改进刺激设计，使用GPT-2进行测试，并利用Gemini 2.5 Pro生成更优质的测试数据。结果显示，刺激质量显著影响模型的语法预测表现。**

- **链接: [http://arxiv.org/pdf/2510.06018v1](http://arxiv.org/pdf/2510.06018v1)**

> **作者:** Timothy Pistotti; Jason Brown; Michael Witbrock
>
> **备注:** Presented at https://brigap-workshop.github.io/ Information to be updated upon publication of proceedings
>
> **摘要:** Recent studies employing Large Language Models (LLMs) to test the Argument from the Poverty of the Stimulus (APS) have yielded contrasting results across syntactic phenomena. This paper investigates the hypothesis that characteristics of the stimuli used in recent studies, including lexical ambiguities and structural complexities, may confound model performance. A methodology is proposed for re-evaluating LLM competence on syntactic prediction, focusing on GPT-2. This involves: 1) establishing a baseline on previously used (both filtered and unfiltered) stimuli, and 2) generating a new, refined dataset using a state-of-the-art (SOTA) generative LLM (Gemini 2.5 Pro Preview) guided by linguistically-informed templates designed to mitigate identified confounds. Our preliminary findings indicate that GPT-2 demonstrates notably improved performance on these refined PG stimuli compared to baselines, suggesting that stimulus quality significantly influences outcomes in surprisal-based evaluations of LLM syntactic competency.
>
---
#### [new 081] Mixture of Neuron Experts
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决MoE模型推理时参数利用率低、效率不高的问题。作者通过研究MoE层参数激活的稀疏性，提出了一种基于神经元粒度的专家选择方法Mixture of Neuron Experts（MoNE），实现高效推理并提升参数利用效率。**

- **链接: [http://arxiv.org/pdf/2510.05781v1](http://arxiv.org/pdf/2510.05781v1)**

> **作者:** Runxi Cheng; Yuchen Guan; Yucheng Ding; Qingguo Hu; Yongxian Wei; Chun Yuan; Yelong Shen; Weizhu Chen; Yeyun Gong
>
> **备注:** 18 page, 11 figures, 7 tables
>
> **摘要:** In this work, we first explore whether the parameters activated by the MoE layer remain highly sparse at inference. We perform a sparsification study on several representative MoE models. For each expert, we rank parameters by the magnitude of their activations from the gate projection and progressively prune the activated subset. Pruning up to 60% of parameters within that subset causes only negligible task-performance degradation; substantial drops occur only after more than 90% are removed. We further decompose experts into neuron-granular MoE and visualize their activation values, finding that most neuron activations are near zero. This observation motivates us to select only high-activation neuron experts during pretraining. Based on this insight, we propose Mixture of Neuron Experts (MoNE). MoNE achieves neuron-granular expert selection by only applying a simple top-k selection within each expert, incurs negligible latency, and requires no additional routing parameters or inter-expert communication. Extensive experiments demonstrate that MoNE matches traditional MoE performance while activating only 50% of the MoE-layer parameters, and it consistently outperforms traditional MoE when compared at equal numbers of activated parameters. These results suggest that MoNE is a practical approach to improving parameter utilization and inference efficiency in MoE-like models.
>
---
#### [new 082] MASA: Rethinking the Representational Bottleneck in LoRA with Multi-A Shared Adaptation
- **分类: cs.CL**

- **简介: 论文属于参数高效微调任务，旨在解决LoRA中单一降维矩阵导致的表达瓶颈问题。工作提出MASA架构，采用多A单B结构，通过跨层共享多专家A矩阵提取多样特征，提升模型在多领域、单领域及多任务场景下的适应能力。**

- **链接: [http://arxiv.org/pdf/2510.06005v1](http://arxiv.org/pdf/2510.06005v1)**

> **作者:** Qin Dong; Yuntian Tang; Heming Jia; Yunhang Shen; Bohan Jia; Wenxuan Huang; Lianyue Zhang; Jiao Xie; Shaohui Lin
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a dominant method in Parameter-Efficient Fine-Tuning (PEFT) for large language models, which augments the transformer layer with one down-projection $A$ and one up-projection $B$. However, LoRA's reliance on a single down-projection matrix ($A$) creates a representational bottleneck, as this solitary feature extractor is inherently insufficient for capturing the diverse signals required by complex tasks. This motivates our architectural shift to focus on enriching the feature adaptation to improve the downstream task adaptation ability. We propose MASA (Multi-$A$ Shared Adaptation), an architecture that implements a multi-$A$, single-$B$ structure where the multi-$A$ expert ensemble is asymmetrically shared across layers to ensure parameter efficiency. In MASA, these specialized experts capture diverse features, which are then integrated by a single, layer-specific $B$-matrix. The effectiveness and versatility of our method are validated through a comprehensive suite of experiments spanning multi-domain generalization, single-domain specialization, and multi-task reasoning. For example, on the MMLU benchmark, MASA achieves an average accuracy of 59.62%, outperforming the standard LoRA by 1.08 points (a relative improvement of 1.84%) with comparable learnable parameters of 0.52%.
>
---
#### [new 083] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型在指令微调后难以灵活调整输出分布的问题。作者提出Spectrum Tuning方法，通过大规模资源Spectrum Suite进行后训练，提升模型在上下文中根据新信息调整输出的能力，同时覆盖更多样化的输出空间并保持分布对齐。**

- **链接: [http://arxiv.org/pdf/2510.06084v1](http://arxiv.org/pdf/2510.06084v1)**

> **作者:** Taylor Sorensen; Benjamin Newman; Jared Moore; Chan Park; Jillian Fisher; Niloofar Mireshghallah; Liwei Jiang; Yejin Choi
>
> **摘要:** Language model post-training has enhanced instruction-following and performance on many downstream tasks, but also comes with an often-overlooked cost on tasks with many possible valid answers. We characterize three desiderata for conditional distributional modeling: in-context steerability, valid output space coverage, and distributional alignment, and document across three model families how current post-training can reduce these properties. In particular, we disambiguate between two kinds of in-context learning: ICL for eliciting existing underlying knowledge or capabilities, and in-context steerability, where a model must use in-context information to override its priors and steer to a novel data generating distribution. To better evaluate and improve these desiderata, we introduce Spectrum Suite, a large-scale resource compiled from >40 data sources and spanning >90 tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. We find that while current post-training techniques help elicit underlying capabilities and knowledge, they hurt models' ability to flexibly steer in-context. To mitigate these issues, we propose Spectrum Tuning, a post-training method using Spectrum Suite to improve steerability and distributional coverage. We find that Spectrum Tuning often improves over pretrained models and their instruction-tuned counterparts, enhancing steerability, spanning more of the output space, and improving distributional alignment on held-out datasets.
>
---
#### [new 084] Exploring Gaps in the APS: Direct Minimal Pair Analysis in LLM Syntactic Assessments
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型（LLM）在复杂句法结构（如寄生间隙）中的可学习性争议。通过生成精炼的寄生间隙刺激范式，并采用直接最小对比较（wh-effect）方法系统评估GPT-2，发现其在所有测试条件下均表现出对空位依存关系的稳健掌握，表明评估指标的选择对判断LLM句法能力至关重要。**

- **链接: [http://arxiv.org/pdf/2510.06001v1](http://arxiv.org/pdf/2510.06001v1)**

> **作者:** Timothy Pistotti; Jason Brown; Michael Witbrock
>
> **备注:** Presented at the https://brigap-workshop.github.io/ Information to be updated after publication of proceedings
>
> **摘要:** Recent studies probing the Argument from the Poverty of the Stimulus (APS) have applied Large Language Models (LLMs) to test the learnability of complex syntax through surprisal-based metrics. However, divergent conclusions raise questions concerning the insights these metrics offer. While Wilcox et al. (2024) used direct minimal pair comparisons (the "wh-effect") to demonstrate that models successfully generalise knowledge of filler-gap dependencies, Lan et al. (2024) used a Difference-in-Differences (DiD) metric and found that models largely fail on parasitic gaps (PGs). This paper argues that the direct minimal pair approach offers greater diagnostic transparency. We demonstrate this by generating a full 8-permutation paradigm of refined PG stimuli and evaluating the GPT-2 model used in previous studies with a systematic Wilcox-style wh-effect analysis. Our results show that GPT-2 succeeds across all four tested conditions, indicating robust knowledge of filler-gap licensing principles even in complex PG environments. This finding, which contrasts with the more ambiguous results from DiD-style metrics, suggests that the choice of evaluation metric is critical for assessing an LLM's syntactic competence.
>
---
#### [new 085] H1B-KV: Hybrid One-Bit Caches for Memory-Efficient Large Language Model Inference
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型推理任务，旨在解决长上下文推理中的内存瓶颈问题。通过提出H1B-KV混合缓存压缩方案，对键值对进行1位二进制草图和4位量化压缩，大幅减少内存使用，并保持模型性能。**

- **链接: [http://arxiv.org/pdf/2510.05529v1](http://arxiv.org/pdf/2510.05529v1)**

> **作者:** Harshil Vejendla
>
> **备注:** MIT URTC 2025 Technical Paper (Oral), 5 pages, 1 figure
>
> **摘要:** Autoregressive decoding in large language models (LLMs) requires caching a growing list of past key-value (KV) pairs, making long-context inference a memory-bound problem. While recent methods have explored quantizing the cache, evicting tokens, or using binary sketches for keys (e.g., Loki), these approaches often provide an incomplete solution by leaving one component (like values) uncompressed or by discarding context information. This paper introduces the Hybrid One-Bit KV Cache (H1B-KV), a comprehensive compression scheme that radically reduces memory usage without sacrificing context. H1B-KV represents each key vector using a 1-bit binary sketch, enabling hardware-friendly bitwise attention, and further compresses value vectors using 4-bit quantization. This holistic, hybrid approach allows a 7-billion parameter LLM to handle an 8k-token context with under 60 MB of cache memory - a 70x reduction. We demonstrate that after a lightweight finetuning, H1B-KV matches full-precision performance not only on perplexity benchmarks but also on complex downstream tasks like mathematical reasoning (GSM8K), multi-task understanding (MMLU), and code generation (HumanEval). Our results show H1B-KV significantly outperforms leading quantization (KIVI), token eviction (SparseLLM), and key-only sketching (Loki) methods in quality-per-byte, establishing it as a robust solution for deploying LLMs in memory-constrained environments.
>
---
#### [new 086] TensorBLEU: Vectorized GPU-based BLEU Score Implementation for Per-Sentence In-Training Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 论文提出TensorBLEU，旨在加速自然语言处理模型训练中的BLEU评分计算。该工作属于模型评估任务，针对传统CPU方法（如NLTK）在GPU上逐句评估效率低的问题，设计了基于PyTorch的全向量化、内存高效实现，实现最高40倍速度提升，适用于强化学习等需实时奖励信号的场景。**

- **链接: [http://arxiv.org/pdf/2510.05485v1](http://arxiv.org/pdf/2510.05485v1)**

> **作者:** Adam Filipek
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Modern natural language processing models have achieved unprecedented scale, yet the tools for their evaluation often remain a computational bottleneck, limiting the pace of research. This is particularly acute for in-training evaluation metrics, such as per-sentence reward signals in Reinforcement Learning, which must operate efficiently on batches of token IDs directly on the GPU. In this paper, we introduce TensorBLEU, a novel implementation of the BLEU metric designed from the ground up for this specific use case. Our approach is fully vectorized for GPU-accelerated, per-sentence computation within PyTorch and introduces a memory-efficient counting mechanism. By creating a compact, batch-specific dictionary of n-grams using \texttt{torch.unique}, our method avoids the prohibitive memory costs of traditional hashing-based vectorization, making it practical for large-vocabulary models. We benchmark TensorBLEU against NLTK, the standard library for token-ID-based BLEU calculation on the CPU. Experiments show that TensorBLEU provides speedups of over 13x on consumer-grade GPUs (NVIDIA T4) and exceeding 40x on data-center-class hardware (NVIDIA A100). This performance transforms a significant bottleneck into a negligible part of the training loop. By clearly defining its role as a "Token-ID BLEU" for development purposes and open-sourcing our implementation, we provide a powerful tool for accelerating research in areas like RL-based model fine-tuning.
>
---
#### [new 087] Aligning Language Models with Clinical Expertise: DPO for Heart Failure Nursing Documentation in Critical Care
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理与临床医学结合的任务，旨在解决ICU中心力衰竭护理文档不规范的问题。作者使用DPO方法优化Mistral-7B模型，利用MIMIC-III数据提升文档质量，评估显示各项指标显著提高。**

- **链接: [http://arxiv.org/pdf/2510.05410v1](http://arxiv.org/pdf/2510.05410v1)**

> **作者:** Junyi Fan; Li Sun; Negin Ashrafi; Kamiar Alaei; Maryam Pishgar
>
> **摘要:** Nursing documentation in intensive care units (ICUs) provides essential clinical intelligence but often suffers from inconsistent terminology, informal styles, and lack of standardization, challenges that are particularly critical in heart failure care. This study applies Direct Preference Optimization (DPO) to adapt Mistral-7B, a locally deployable language model, using 8,838 heart failure nursing notes from the MIMIC-III database and 21,210 preference pairs derived from expert-verified GPT outputs, model generations, and original notes. Evaluation across BLEU, ROUGE, BERTScore, Perplexity, and expert qualitative assessments demonstrates that DPO markedly enhances documentation quality. Specifically, BLEU increased by 84% (0.173 to 0.318), BERTScore improved by 7.6% (0.828 to 0.891), and expert ratings rose across accuracy (+14.4 points), completeness (+14.5 points), logical consistency (+14.1 points), readability (+11.1 points), and structural clarity (+6.0 points). These results indicate that DPO can align lightweight clinical language models with expert standards, supporting privacy-preserving, AI-assisted documentation within electronic health record systems to reduce administrative burden and improve ICU patient safety.
>
---
#### [new 088] Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型（LLMs）在亚洲语言中可能存在的文化偏见问题。作者构建了多语言基准Camellia，包含19,530个标注实体和2,173个语境，评估LLMs在文化适应、情感关联和实体抽取等任务中的表现，发现模型在亚洲文化理解上存在偏差和性能差距。**

- **链接: [http://arxiv.org/pdf/2510.05291v1](http://arxiv.org/pdf/2510.05291v1)**

> **作者:** Tarek Naous; Anagha Savit; Carlos Rafael Catalan; Geyang Guo; Jaehyeok Lee; Kyungdon Lee; Lheane Marie Dizon; Mengyu Ye; Neel Kothari; Sahajpreet Singh; Sarah Masud; Tanish Patwa; Trung Thanh Tran; Zohaib Khan; Alan Ritter; JinYeong Bak; Keisuke Sakaguchi; Tanmoy Chakraborty; Yuki Arase; Wei Xu
>
> **摘要:** As Large Language Models (LLMs) gain stronger multilingual capabilities, their ability to handle culturally diverse entities becomes crucial. Prior work has shown that LLMs often favor Western-associated entities in Arabic, raising concerns about cultural fairness. Due to the lack of multilingual benchmarks, it remains unclear if such biases also manifest in different non-Western languages. In this paper, we introduce Camellia, a benchmark for measuring entity-centric cultural biases in nine Asian languages spanning six distinct Asian cultures. Camellia includes 19,530 entities manually annotated for association with the specific Asian or Western culture, as well as 2,173 naturally occurring masked contexts for entities derived from social media posts. Using Camellia, we evaluate cultural biases in four recent multilingual LLM families across various tasks such as cultural context adaptation, sentiment association, and entity extractive QA. Our analyses show a struggle by LLMs at cultural adaptation in all Asian languages, with performance differing across models developed in regions with varying access to culturally-relevant data. We further observe that different LLM families hold their distinct biases, differing in how they associate cultures with particular sentiments. Lastly, we find that LLMs struggle with context understanding in Asian languages, creating performance gaps between cultures in entity extraction.
>
---
#### [new 089] Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs
- **分类: cs.CL**

- **简介: 该论文属于多跳推理任务，旨在解决开放域问题中信息检索与推理的挑战。作者提出FGDIP框架，通过动态交互规划和实时反馈优化推理策略。实验表明其在HotpotQA和StrategyQA数据集上表现优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.05577v1](http://arxiv.org/pdf/2510.05577v1)**

> **作者:** Dong Yan; Gaochen Wu; Bowen Zhou
>
> **摘要:** Recent advancements in language agents have led to significant improvements in multi-hop reasoning tasks. However, existing approaches often struggle with handling open-domain problems, which require massive information retrieval due to their reliance on a fixed sequence of actions. To address this, we propose Feedback-Guided Dynamic Interactive Planning (FGDIP), a novel framework tailored to enhance reasoning in LLMs by utilizing dynamic and adaptive strategies for information exploration in open-domain multi-hop reasoning tasks. Our approach begins by identifying key entities relevant to the problem, which serve as the initial nodes in the reasoning process. From these initial nodes, we then generate reasoning child nodes with the process being refined through a combination of historical error analysis and real-time feedback, which allows the framework to dynamically adjust and optimize its reasoning strategies. By integrating depth-first search with an innovative node generation technique, our framework adapts based on both prior error paths and concurrently generated nodes at the same hierarchical level. This dynamic strategy effectively expands the search space while ensuring the reasoning process systematically converges toward accurate solutions. Experimental results show that FGDIP achieved up to 54.47% F1 score on the HotpotQA dataset and 70.05% on the StrategyQA dataset, surpassing the best baseline by 5.03% and 7.25% respectively, highlighting its versatility and potential to enhance language agents in multi-hop reasoning tasks.
>
---
#### [new 090] The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决如何通过推理痕迹将大语言模型的编码能力迁移到小模型中。论文发现模型性能随蒸馏数据量呈现先降后升的趋势，提出“代码推理的山谷”现象，并分析了不同难度问题和输出正确性对蒸馏效果的影响。**

- **链接: [http://arxiv.org/pdf/2510.06101v1](http://arxiv.org/pdf/2510.06101v1)**

> **作者:** Muyu He; Muhammad Ali Shafique; Anand Kumar; Tsach Mackey; Nazneen Rajani
>
> **备注:** NeurIPS 2025 Workshop on Deep Learning for Code (DL4C), Project page: https://collinear.ai/valley-of-reasoning
>
> **摘要:** Distilling the thinking traces of a Large Language Model (LLM) with reasoning capabilities into a smaller model has been proven effective. Yet, there is a scarcity of work done on how model performances scale with the quantity of distillation data. In this work, we study the scaling trend of distilling competitive coding skills on two small non-reasoning LLMs. We validate the hypothesis that there is a $\textit{valley of code reasoning}$: downstream performance on competitive coding first drops as data quantity increases, then it steadily increases in a sharper-than-log-linear fashion. Having identified the trend, we further fine-tune the models at two different distillation stages on the same data to ground conclusions on their respective learning phases. We learn that across stages in the low and medium-low data regimes, small models benefit significantly from easier coding questions than from harder ones. We also find that, surprisingly, the correctness of outputs in training data makes no difference to distillation outcomes. Our work represents a step forward in understanding the training dynamics of code reasoning distillation outside intuition
>
---
#### [new 091] Probing the Difficulty Perception Mechanism of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）对问题难度的感知机制，属于自然语言处理与模型解释任务。它旨在解决LLMs能否在内部评估问题难度的问题。论文通过线性探测和注意力头分析，验证LLMs能线性建模数学问题难度，并定位感知难度的关键结构。**

- **链接: [http://arxiv.org/pdf/2510.05969v1](http://arxiv.org/pdf/2510.05969v1)**

> **作者:** Sunbowen Lee; Qingyu Yin; Chak Tou Leong; Jialiang Zhang; Yicheng Gong; Xiaoyu Shen
>
> **摘要:** Large language models (LLMs) are increasingly deployed on complex reasoning tasks, yet little is known about their ability to internally evaluate problem difficulty, which is an essential capability for adaptive reasoning and efficient resource allocation. In this work, we investigate whether LLMs implicitly encode problem difficulty in their internal representations. Using a linear probe on the final-token representations of LLMs, we demonstrate that the difficulty level of math problems can be linearly modeled. We further locate the specific attention heads of the final Transformer layer: these attention heads have opposite activation patterns for simple and difficult problems, thus achieving perception of difficulty. Our ablation experiments prove the accuracy of the location. Crucially, our experiments provide practical support for using LLMs as automatic difficulty annotators, potentially substantially reducing reliance on costly human labeling in benchmark construction and curriculum learning. We also uncover that there is a significant difference in entropy and difficulty perception at the token level. Our study reveals that difficulty perception in LLMs is not only present but also structurally organized, offering new theoretical insights and practical directions for future research.
>
---
#### [new 092] Chronological Thinking in Full-Duplex Spoken Dialogue Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于口语对话系统任务，旨在解决全双工对话模型在用户讲话时无法有效思考的问题。作者提出“时序思维”机制，使模型在倾听时逐步推理，提升响应质量，同时保持低延迟，更好地模拟人类对话中的实时互动与动态处理能力。**

- **链接: [http://arxiv.org/pdf/2510.05150v1](http://arxiv.org/pdf/2510.05150v1)**

> **作者:** Donghang Wu; Haoyang Zhang; Chen Chen; Tianyu Zhang; Fei Tian; Xuerui Yang; Gang Yu; Hexin Liu; Nana Hou; Yuchen Hu; Eng Siong Chng
>
> **摘要:** Recent advances in spoken dialogue language models (SDLMs) reflect growing interest in shifting from turn-based to full-duplex systems, where the models continuously perceive user speech streams while generating responses. This simultaneous listening and speaking design enables real-time interaction and the agent can handle dynamic conversational behaviors like user barge-in. However, during the listening phase, existing systems keep the agent idle by repeatedly predicting the silence token, which departs from human behavior: we usually engage in lightweight thinking during conversation rather than remaining absent-minded. Inspired by this, we propose Chronological Thinking, a on-the-fly conversational thinking mechanism that aims to improve response quality in full-duplex SDLMs. Specifically, chronological thinking presents a paradigm shift from conventional LLM thinking approaches, such as Chain-of-Thought, purpose-built for streaming acoustic input. (1) Strictly causal: the agent reasons incrementally while listening, updating internal hypotheses only from past audio with no lookahead. (2) No additional latency: reasoning is amortized during the listening window; once the user stops speaking, the agent halts thinking and begins speaking without further delay. Experiments demonstrate the effectiveness of chronological thinking through both objective metrics and human evaluations show consistent improvements in response quality. Furthermore, chronological thinking robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics.
>
---
#### [new 093] Diversity Is All You Need for Contrastive Learning: Spectral Bounds on Gradient Magnitudes
- **分类: cs.CL**

- **简介: 该论文属于对比学习任务，旨在提升模型训练效率与性能。作者通过分析梯度谱，推导出非渐近的梯度界，并提出谱感知的批量选择方法，有效提升训练速度与准确率，同时减少梯度方差。**

- **链接: [http://arxiv.org/pdf/2510.05767v1](http://arxiv.org/pdf/2510.05767v1)**

> **作者:** Peter Ochieng
>
> **摘要:** We derive non-asymptotic spectral bands that bound the squared InfoNCE gradient norm via alignment, temperature, and batch spectrum, recovering the \(1/\tau^{2}\) law and closely tracking batch-mean gradients on synthetic data and ImageNet. Using effective rank \(R_{\mathrm{eff}}\) as an anisotropy proxy, we design spectrum-aware batch selection, including a fast greedy builder. On ImageNet-100, Greedy-64 cuts time-to-67.5\% top-1 by 15\% vs.\ random (24\% vs.\ Pool--P3) at equal accuracy; CIFAR-10 shows similar gains. In-batch whitening promotes isotropy and reduces 50-step gradient variance by \(1.37\times\), matching our theoretical upper bound.
>
---
#### [new 094] Beyond Monolithic Rewards: A Hybrid and Multi-Aspect Reward Optimization for MLLM Alignment
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）对齐任务，旨在解决单一奖励信号在对齐人类偏好时的局限性。论文提出了一种混合奖励建模框架，结合模型基础奖励与规则基础奖励，并引入多方面奖励和长度惩罚机制，以提升模型在多模态任务上的表现。实验表明该方法在通用和数学推理任务上均有显著改进。**

- **链接: [http://arxiv.org/pdf/2510.05283v1](http://arxiv.org/pdf/2510.05283v1)**

> **作者:** Radha Gulhane; Sathish Reddy Indurthi
>
> **摘要:** Aligning multimodal large language models (MLLMs) with human preferences often relies on single-signal, model-based reward methods. Such monolithic rewards often lack confidence calibration across domain-specific tasks, fail to capture diverse aspects of human preferences, and require extensive data annotation and reward model training. In this work, we propose a hybrid reward modeling framework that integrates complementary reward paradigms: (i) model-based rewards, where a learned reward model predicts scalar or vector scores from synthetic and human feedback, and (ii) rule-based rewards, where domain-specific heuristics provide explicit correctness signals with confidence. Beyond accuracy, we further incorporate multi-aspect rewards to enforce instruction adherence and introduce a generalized length-penalty reward to stabilize training and improve performance. The proposed framework provides a flexible and effective approach to aligning MLLMs through reinforcement learning policy optimization. Our experiments show consistent improvements across different multimodal benchmarks when applying hybrid and multi-aspect reward modeling. Our best performing model in the 3B family achieves an overall average improvement of ~9.5% across general and math reasoning tasks. Focusing specifically on mathematical benchmarks, the model achieves a significant average improvement of ~16%, highlighting its effectiveness in mathematical reasoning and problem solving.
>
---
#### [new 095] Generative AI-Driven Hierarchical Multi-Agent Framework for Zero-Touch Optical Networks
- **分类: cs.NI; cs.AI; cs.CL; cs.MA; cs.SY; eess.SY**

- **简介: 该论文属于网络管理任务，旨在解决光网络全生命周期管理中多任务协同自动化问题。为实现零触控光网络，论文提出了一种基于生成式AI的分层多智能体框架，支持多任务分配、协作与执行，并通过实际网络场景验证其在规划、运维和升级阶段的应用能力。**

- **链接: [http://arxiv.org/pdf/2510.05625v1](http://arxiv.org/pdf/2510.05625v1)**

> **作者:** Yao Zhang; Yuchen Song; Shengnan Li; Yan Shi; Shikui Shen; Xiongyan Tang; Min Zhang; Danshi Wang
>
> **备注:** 7 pages,6 figures, Accepted by lEEE Communications Magazine, Open call
>
> **摘要:** The rapid development of Generative Artificial Intelligence (GenAI) has catalyzed a transformative technological revolution across all walks of life. As the backbone of wideband communication, optical networks are expecting high-level autonomous operation and zero-touch management to accommodate their expanding network scales and escalating transmission bandwidth. The integration of GenAI is deemed as the pivotal solution for realizing zero-touch optical networks. However, the lifecycle management of optical networks involves a multitude of tasks and necessitates seamless collaboration across multiple layers, which poses significant challenges to the existing single-agent GenAI systems. In this paper, we propose a GenAI-driven hierarchical multi-agent framework designed to streamline multi-task autonomous execution for zero-touch optical networks. We present the architecture, implementation, and applications of this framework. A field-deployed mesh network is utilized to demonstrate three typical scenarios throughout the lifecycle of optical network: quality of transmission estimation in the planning stage, dynamic channel adding/dropping in the operation stage, and system capacity increase in the upgrade stage. The case studies, illustrate the capabilities of multi-agent framework in multi-task allocation, coordination, execution, evaluation, and summarization. This work provides a promising approach for the future development of intelligent, efficient, and collaborative network management solutions, paving the way for more specialized and adaptive zero-touch optical networks.
>
---
#### [new 096] Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型（LLM）安全评估任务，旨在解决现有评估方法不够可靠、难以比较的问题。论文提出了一种基于贝叶斯建模的端到端评估框架，改进实验设计与不确定性量化，用于评估LLM对提示注入攻击的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.05709v1](http://arxiv.org/pdf/2510.05709v1)**

> **作者:** Mary Llewellyn; Annie Gray; Josh Collyer; Michael Harries
>
> **摘要:** Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability.
>
---
#### [new 097] Learning from Failures: Understanding LLM Alignment through Failure-Aware Inverse RL
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理与机器学习交叉任务，旨在提升大语言模型（LLM）的可解释性与安全性。它通过改进逆强化学习（IRL）方法，提出“失败感知”算法，专注于模型误判或难以区分的数据，以更准确还原RLHF中的潜在奖励函数，从而优化模型对齐并减少不确定性。**

- **链接: [http://arxiv.org/pdf/2510.06092v1](http://arxiv.org/pdf/2510.06092v1)**

> **作者:** Nyal Patel; Matthieu Bou; Arjun Jagota; Satyapriya Krishna; Sonali Parbhoo
>
> **备注:** Preprint
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with human preferences, yet the underlying reward signals they internalize remain hidden, posing a critical challenge for interpretability and safety. Existing approaches attempt to extract these latent incentives using Inverse Reinforcement Learning (IRL), but treat all preference pairs equally, often overlooking the most informative signals: those examples the extracted reward model misclassifies or assigns nearly equal scores, which we term \emph{failures}. We introduce a novel \emph{failure-aware} IRL algorithm that focuses on misclassified or difficult examples to recover the latent rewards defining model behaviors. By learning from these failures, our failure-aware IRL extracts reward functions that better reflect the true objectives behind RLHF. We demonstrate that failure-aware IRL outperforms existing IRL baselines across multiple metrics when applied to LLM detoxification, without requiring external classifiers or supervision. Crucially, failure-aware IRL yields rewards that better capture the true incentives learned during RLHF, enabling more effective re-RLHF training than standard IRL. This establishes failure-aware IRL as a robust, scalable method for auditing model alignment and reducing ambiguity in the IRL process.
>
---
#### [new 098] The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI对齐任务，旨在解决大型语言模型（LLM）目标不透明、难以验证的问题。作者提出一种基于贝叶斯逆强化学习的“对齐审计框架”，能推断目标分布、量化不确定性，并通过多轮证据收缩后验，提升可解释性与可信度，支持实际审计与优化LLM行为。**

- **链接: [http://arxiv.org/pdf/2510.06096v1](http://arxiv.org/pdf/2510.06096v1)**

> **作者:** Matthieu Bou; Nyal Patel; Arjun Jagota; Satyapriya Krishna; Sonali Parbhoo
>
> **备注:** Preprint
>
> **摘要:** The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
>
---
#### [new 099] Adversarial Reinforcement Learning for Large Language Model Agent Safety
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型安全任务，旨在解决工具使用中面临的间接提示注入攻击问题。作者提出ARLAS框架，通过对抗强化学习，协同训练攻击者与防御者，提升模型面对新型攻击时的鲁棒性。实验表明该方法有效降低攻击成功率并提高任务完成率。**

- **链接: [http://arxiv.org/pdf/2510.05442v1](http://arxiv.org/pdf/2510.05442v1)**

> **作者:** Zizhao Wang; Dingcheng Li; Vaishakh Keshava; Phillip Wallis; Ananth Balashankar; Peter Stone; Lukas Rutishauser
>
> **摘要:** Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.
>
---
#### [new 100] TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出TaTToo，一种基于工具的表格推理过程奖励模型（PRM），旨在提升大推理模型（LRMs）在表格推理任务上的测试时扩展能力（TTS）。现有PRM在处理表格特有操作（如子表检索、模式交互）时效果不佳，为此，作者构建了60k高质量标注数据，并采用两阶段训练方法：冷启动监督微调与工具引导的强化学习。实验表明，TaTToo在5个表格推理任务上平均提升30.9%，且模型规模仅为8B，优于72B基线模型。**

- **链接: [http://arxiv.org/pdf/2510.06217v1](http://arxiv.org/pdf/2510.06217v1)**

> **作者:** Jiaru Zou; Soumya Roy; Vinay Kumar Verma; Ziyi Wang; David Wipf; Pan Lu; Sumit Negi; James Zou; Jingrui He
>
> **摘要:** Process Reward Models (PRMs) have recently emerged as a powerful framework for enhancing the reasoning capabilities of large reasoning models (LRMs), particularly in the context of test-time scaling (TTS). However, their potential for supervising LRMs on tabular reasoning domains remains underexplored. Through detailed empirical analyses, we identify that existing PRMs, though widely adopted for supervising text-only reasoning steps, struggle with table-specific operations such as sub-table retrieval and schema interaction, leading to critical performance bottlenecks. To address this limitation, we propose TaTToo, a novel table-grounded PRM framework that (i) reasons explicitly over tabular reasoning steps and (ii) integrates tool-based verification to provide precise reward supervision. Concretely, we first design a scalable data curation pipeline that constructs over 60k high-quality step-level annotations by integrating table verification rationales with tool-based executions. Building on the collected data, we train TaTToo with a dual-stage paradigm: cold-start supervised fine-tuning to capture tool-use reasoning patterns, followed by reinforcement learning with tool-grounded reward shaping to align our model with table-based verification. We provide a comprehensive evaluation of the policy improvement induced by our newly designed PRM. Across 5 challenging tabular reasoning benchmarks covering numerical reasoning, fact-checking, and data analysis, TaTToo improves downstream policy LRMs by 30.9% at inference, surpasses strong PRM baselines such as Qwen-2.5-Math-PRM-72B with only 8B parameters, and demonstrates strong generalizability across diverse TTS strategies.
>
---
#### [new 101] Paying Attention to Hybrid Attention: Untangling the Issues with Conversion Methods
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型因计算复杂度高导致的扩展性问题。论文揭示现有线性化方法的缺陷，提出三种新方法以平衡线性注意力与滑动窗口机制的使用，确保模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2510.05901v1](http://arxiv.org/pdf/2510.05901v1)**

> **作者:** Martin Benfeghoul; Teresa Delgado; Adnan Oomerjee; Haitham Bou Ammar; Jun Wang; Zafeirios Fountas
>
> **摘要:** Transformers' quadratic computational complexity limits their scalability despite remarkable performance. While linear attention reduces this to linear complexity, pre-training such models from scratch remains, in most cases, prohibitively expensive. Recent post-training linearisation methods convert pre-trained Transformers to linear models efficiently, often using hybrid approaches that combine linear attention with sliding-window softmax. We identify a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on SWA. Component-level diagnostics reveal this previously undetected behaviour stems from overlooked evaluation practices on common-sense benchmarks. We propose three solutions to ensure balanced component usage: (i) inference-time hybridisation of linear-only conversions with sliding-window softmax; (ii) HedgeCATs, combining attention-weight transfer with targeted LoRA fine-tuning; and (iii) Scheduled Sliding-window Dropout (SSD), which stochastically suppresses the softmax branch during training to prevent component collapse. Our methods maintain computational efficiency while recovering most base model performance and ensuring genuine linear attention adoption, restoring the validity of performance attributions in hybrid conversions.
>
---
#### [new 102] Optimization Modeling via Semantic Anchored Alignment
- **分类: cs.AI; cs.CL; cs.PL**

- **简介: 该论文属于优化建模任务，旨在解决大语言模型生成求解器代码时语义错误导致模型逻辑错误的问题。论文提出了SAC-Opt框架，通过语义锚定对齐与修正机制，提升模型生成的语义准确性和鲁棒性。实验表明其建模准确率平均提升了7.8%。**

- **链接: [http://arxiv.org/pdf/2510.05115v1](http://arxiv.org/pdf/2510.05115v1)**

> **作者:** Yansen Zhang; Qingcan Kang; Yujie Chen; Yufei Wang; Xiongwei Han; Tao Zhong; Mingxuan Yuan; Chen Ma
>
> **摘要:** Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.8\%, with gains of up to 21.9\% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code.
>
---
#### [new 103] Domain-Shift-Aware Conformal Prediction for Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.LG; stat.AP**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在领域迁移下的不确定性量化问题。提出了一种领域感知的共形预测方法（DS-CP），通过重新加权校准样本来提高预测集在分布偏移下的可靠性，保证覆盖性的同时提升适应性。**

- **链接: [http://arxiv.org/pdf/2510.05566v1](http://arxiv.org/pdf/2510.05566v1)**

> **作者:** Zhexiao Lin; Yuanyuan Li; Neeraj Sarna; Yuanyuan Gao; Michael von Gablenz
>
> **备注:** 26 pages
>
> **摘要:** Large language models have achieved impressive performance across diverse tasks. However, their tendency to produce overconfident and factually incorrect outputs, known as hallucinations, poses risks in real world applications. Conformal prediction provides finite-sample, distribution-free coverage guarantees, but standard conformal prediction breaks down under domain shift, often leading to under-coverage and unreliable prediction sets. We propose a new framework called Domain-Shift-Aware Conformal Prediction (DS-CP). Our framework adapts conformal prediction to large language models under domain shift, by systematically reweighting calibration samples based on their proximity to the test prompt, thereby preserving validity while enhancing adaptivity. Our theoretical analysis and experiments on the MMLU benchmark demonstrate that the proposed method delivers more reliable coverage than standard conformal prediction, especially under substantial distribution shifts, while maintaining efficiency. This provides a practical step toward trustworthy uncertainty quantification for large language models in real-world deployment.
>
---
#### [new 104] TokenChain: A Discrete Speech Chain via Semantic Token Modeling
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音处理任务，旨在提升语音识别（ASR）和语音合成（TTS）性能。论文提出TokenChain，通过语义token建模构建离散语音链，结合自回归文本到语义模型与掩码生成的语义到声学模型，实现端到端反馈学习，有效提升跨域迁移效果，减少错误率并缓解遗忘问题。**

- **链接: [http://arxiv.org/pdf/2510.06201v1](http://arxiv.org/pdf/2510.06201v1)**

> **作者:** Mingxuan Wang; Satoshi Nakamura
>
> **备注:** 5 pages, 3 figures. Submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Machine Speech Chain, simulating the human perception-production loop, proves effective in jointly improving ASR and TTS. We propose TokenChain, a fully discrete speech chain coupling semantic-token ASR with a two-stage TTS: an autoregressive text-to-semantic model co-trained with ASR and a masked-generative semantic-to-acoustic model for synthesis only. End-to-end feedback across the text interface is enabled with straight-through argmax/Gumbel-Softmax and balanced with supervised ASR via dynamic weight averaging. Ablations examine optimal temperature schedules for in- and cross-domain transfer. Evaluation reveals TokenChain surpasses baseline accuracy 2-6 epochs earlier and yields 5-13% lower equal-epoch error with stable T2S on LibriSpeech, and reduces relative ASR WER by 56% and T2S WER by 31% on TED-LIUM with minimal forgetting, showing that chain learning remains effective with token interfaces and models.
>
---
#### [new 105] In-the-Flow Agentic System Optimization for Effective Planning and Tool Use
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于强化学习与大语言模型任务，旨在解决现有工具增强方法在长周期、多工具场景下泛化能力弱、扩展性差的问题。论文提出AgentFlow框架，采用可训练的流程内智能体系统，结合四个模块协作与Flow-GRPO算法，实现多轮交互中的高效规划与工具使用，显著提升性能表现。**

- **链接: [http://arxiv.org/pdf/2510.05592v1](http://arxiv.org/pdf/2510.05592v1)**

> **作者:** Zhuofeng Li; Haoxiang Zhang; Seungju Han; Sheng Liu; Jianwen Xie; Yu Zhang; Yejin Choi; James Zou; Pan Lu
>
> **备注:** 45 pages, 12 figures. Project website: https://agentflow.stanford.edu/
>
> **摘要:** Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multi-turn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AgentFlow with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns.
>
---
#### [new 106] MatheMagic: Generating Dynamic Mathematics Benchmarks Robust to Memorization
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数学推理评估任务，旨在解决模型对数学测试集的记忆和过拟合并缺乏真实推理能力的问题。论文提出了MatheMagic，通过动态生成带有符号和规则变化的数学题，构建抗过拟合的可验证测试基准，以评估模型的归纳和演绎能力。**

- **链接: [http://arxiv.org/pdf/2510.05962v1](http://arxiv.org/pdf/2510.05962v1)**

> **作者:** Dayyán O'Brien; Barry Haddow; Emily Allaway; Pinzhen Chen
>
> **摘要:** Conducting contamination-free evaluation of mathematical capabilities can be difficult for two reasons: models may memorize a test set once it is made public, and current mathematical benchmarks are prone to overfitting due to having limited diversity of symbols and rules, coupled with closed-ended answers. This paper proposes a method to leverage these shortcomings as useful features to a construct dynamic, counterfactual benchmark, which can be used to both reveal overfitting and measure true reasoning. We demonstrate this via MatheMagic, which generates math test instances with the interpretations of numbers and operators altered, yet has automatically verifiable answers. Test instances are randomly seeded and constructed at test time to evaluate a model's induction or deduction capability, offering stability, extensibility, comparability, and robustness to overfitting. Our experiments find that models solve deduction more easily than induction, but they revert to standard math. Further analysis reveals that math-adapted models fail to exhibit a general "skill" of reasoning, and fine-tuning on induction tasks generalizes poorly.
>
---
#### [new 107] Taxonomy of User Needs and Actions
- **分类: cs.HC; cs.CL**

- **简介: 该论文旨在构建一个全面的用户需求与行为分类框架（TUNA），以更好地理解和设计人机对话系统。它通过分析大量对话数据，结合理论验证，提出多层次分类体系，涵盖信息获取、内容创作、社交互动等行为，解决现有分类过于笼统或局限的问题，支持跨领域应用与政策统一。**

- **链接: [http://arxiv.org/pdf/2510.06124v1](http://arxiv.org/pdf/2510.06124v1)**

> **作者:** Renee Shelby; Fernando Diaz; Vinodkumar Prabhakaran
>
> **摘要:** The growing ubiquity of conversational AI highlights the need for frameworks that capture not only users' instrumental goals but also the situated, adaptive, and social practices through which they achieve them. Existing taxonomies of conversational behavior either overgeneralize, remain domain-specific, or reduce interactions to narrow dialogue functions. To address this gap, we introduce the Taxonomy of User Needs and Actions (TUNA), an empirically grounded framework developed through iterative qualitative analysis of 1193 human-AI conversations, supplemented by theoretical review and validation across diverse contexts. TUNA organizes user actions into a three-level hierarchy encompassing behaviors associated with information seeking, synthesis, procedural guidance, content creation, social interaction, and meta-conversation. By centering user agency and appropriation practices, TUNA enables multi-scale evaluation, supports policy harmonization across products, and provides a backbone for layering domain-specific taxonomies. This work contributes a systematic vocabulary for describing AI use, advancing both scholarly understanding and practical design of safer, more responsive, and more accountable conversational systems.
>
---
#### [new 108] Improving Chain-of-Thought Efficiency for Autoregressive Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像生成任务，旨在解决链式思维（CoT）在生成图像时冗余导致的效率低下问题。作者提出ShortCoTI框架，通过强化学习优化CoT提示，使其更简洁且保持图像质量，从而提升计算效率。**

- **链接: [http://arxiv.org/pdf/2510.05593v1](http://arxiv.org/pdf/2510.05593v1)**

> **作者:** Zeqi Gu; Markos Georgopoulos; Xiaoliang Dai; Marjan Ghazvininejad; Chu Wang; Felix Juefei-Xu; Kunpeng Li; Yujun Shi; Zecheng He; Zijian He; Jiawei Zhou; Abe Davis; Jialiang Wang
>
> **摘要:** Autoregressive multimodal large language models have recently gained popularity for image generation, driven by advances in foundation models. To enhance alignment and detail, newer approaches employ chain-of-thought (CoT) reasoning, expanding user inputs into elaborated prompts prior to image synthesis. However, this strategy can introduce unnecessary redundancy -- a phenomenon we call visual overthinking -- which increases computational costs and can introduce details that contradict the original prompt. In this work, we explore how to generate more concise CoT sequences for more efficient image generation. We introduce ShortCoTI, a lightweight optimization framework that encourages more concise CoT while preserving output image quality. ShortCoTI rewards more concise prompts with an adaptive function that scales according to an estimated difficulty for each task. Incorporating this reward into a reinforcement learning paradigm reduces prompt reasoning length by 54% while maintaining or slightly improving quality metrics across multiple benchmarks (T2I-CompBench, GenEval). Qualitative analysis shows that our method eliminates verbose explanations and repetitive refinements, producing reasoning prompts that are both concise and semantically rich. As a result, ShortCoTI improves computational efficiency without compromising the fidelity or visual appeal of generated images.
>
---
#### [new 109] VAL-Bench: Measuring Value Alignment in Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在公共议题中价值立场一致性问题。作者构建了包含11.5万对争议性文本的VAL-Bench基准，利用大模型作为评判者衡量模型回应的一致性，揭示不同模型在价值观对齐上的差异及安全策略与表达性之间的权衡。**

- **链接: [http://arxiv.org/pdf/2510.05465v1](http://arxiv.org/pdf/2510.05465v1)**

> **作者:** Aman Gupta; Denny O'Shea; Fazl Barez
>
> **摘要:** Large language models (LLMs) are increasingly used for tasks where outputs shape human decisions, so it is critical to test whether their responses reflect consistent human values. Existing benchmarks mostly track refusals or predefined safety violations, but these only check rule compliance and do not reveal whether a model upholds a coherent value system when facing controversial real-world issues. We introduce the \textbf{V}alue \textbf{AL}ignment \textbf{Bench}mark (\textbf{VAL-Bench}), which evaluates whether models maintain a stable value stance across paired prompts that frame opposing sides of public debates. VAL-Bench consists of 115K such pairs from Wikipedia's controversial sections. A well-aligned model should express similar underlying views regardless of framing, which we measure using an LLM-as-judge to score agreement or divergence between paired responses. Applied across leading open- and closed-source models, the benchmark reveals large variation in alignment and highlights trade-offs between safety strategies (e.g., refusals) and more expressive value systems. By providing a scalable, reproducible benchmark, VAL-Bench enables systematic comparison of how reliably LLMs embody human values.
>
---
#### [new 110] Mitigating Premature Exploitation in Particle-based Monte Carlo for Inference-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于推理时扩展任务，旨在解决粒子滤波在数学推理中因过早利用导致的次优解问题。作者提出Entropic Particle Filtering（ePF），通过Entropic Annealing保持多样性，结合Look-ahead Modulation评估路径潜力，提升推理质量。**

- **链接: [http://arxiv.org/pdf/2510.05825v1](http://arxiv.org/pdf/2510.05825v1)**

> **作者:** Giorgio Giannone; Guangxuan Xu; Nikhil Shivakumar Nayak; Rohan Mahesh Awhad; Shivchander Sudalairaj; Kai Xu; Akash Srivastava
>
> **摘要:** Inference-Time Scaling (ITS) improves language models by allocating more computation at generation time. Particle Filtering (PF) has emerged as a strong ITS method for complex mathematical reasoning tasks, but it is vulnerable when guided by process reward models, which often assign overconfident scores early in the reasoning process. This causes PF to suffer from premature exploitation: it myopically commits to locally promising trajectories, prunes potentially correct hypotheses, and converges to suboptimal solutions. This failure mode, known as particle impoverishment, is especially severe under constrained computational budgets. To address this, we analyze the problem and identify two root causes: a lack of diversity in the particle set due to overconfident resampling and consequent inability to assess the potential of a reasoning path. We introduce Entropic Particle Filtering (ePF), an algorithm that integrates two new techniques to solve these issues. The first technique, Entropic Annealing (EA), directly mitigates particle impoverishment by monitoring search diversity via entropy; when diversity drops, it intervenes by dynamically annealing the resampling distribution to preserve exploration. The second, an enhancement called Look-ahead Modulation (LaM), adds a predictive guide to evaluate a state's potential based on its successors. On several challenging math benchmarks, ePF significantly outperforms strong baselines and achieves up to a 50 % relative improvement in task reward. Together, these methods improve PF's resilience by balancing the exploration of diverse solution spaces with the exploitation of high-reward regions, ultimately leading to higher-quality solutions.
>
---
#### [new 111] Early Multimodal Prediction of Cross-Lingual Meme Virality on Reddit: A Time-Window Analysis
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于社交媒体内容传播预测任务，旨在解决跨语言环境下早期预测模因病毒式传播的问题。通过构建混合参与度指标定义病毒性，采用XGBoost等模型结合多模态特征，在不同时间窗口内进行预测。结果显示早期信号有效，XGBoost在30分钟内表现优异，揭示了从静态内容到动态行为的特征重要性转变。**

- **链接: [http://arxiv.org/pdf/2510.05761v1](http://arxiv.org/pdf/2510.05761v1)**

> **作者:** Sedat Dogan; Nina Dethlefs; Debarati Chakraborty
>
> **备注:** Preprint work in progress. Main body: 9 pages. Total: 15 pages including references and appendix. 16 figures and 12 tables
>
> **摘要:** Predicting the virality of online content remains challenging, especially for culturally complex, fast-evolving memes. This study investigates the feasibility of early prediction of meme virality using a large-scale, cross-lingual dataset from 25 diverse Reddit communities. We propose a robust, data-driven method to define virality based on a hybrid engagement score, learning a percentile-based threshold from a chronologically held-out training set to prevent data leakage. We evaluated a suite of models, including Logistic Regression, XGBoost, and a Multi-layer Perceptron (MLP), with a comprehensive, multimodal feature set across increasing time windows (30-420 min). Crucially, useful signals emerge quickly: our best-performing model, XGBoost, achieves a PR-AUC $>$ 0.52 in just 30 minutes. Our analysis reveals a clear "evidentiary transition," in which the importance of the feature dynamically shifts from the static context to the temporal dynamics as a meme gains traction. This work establishes a robust, interpretable, and practical benchmark for early virality prediction in scenarios where full diffusion cascade data is unavailable, contributing a novel cross-lingual dataset and a methodologically sound definition of virality. To our knowledge, this study is the first to combine time series data with static content and network features to predict early meme virality.
>
---
#### [new 112] Influence Functions for Efficient Data Selection in Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决如何高效选择高质量推理数据的问题。论文提出使用影响函数衡量推理数据质量，并引入基于影响函数的剪枝方法，实验证明其在数学推理任务上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.06108v1](http://arxiv.org/pdf/2510.06108v1)**

> **作者:** Prateek Humane; Paolo Cudrano; Daniel Z. Kaplan; Matteo Matteucci; Supriyo Chakraborty; Irina Rish
>
> **摘要:** Fine-tuning large language models (LLMs) on chain-of-thought (CoT) data shows that a small amount of high-quality data can outperform massive datasets. Yet, what constitutes "quality" remains ill-defined. Existing reasoning methods rely on indirect heuristics such as problem difficulty or trace length, while instruction-tuning has explored a broader range of automated selection strategies, but rarely in the context of reasoning. We propose to define reasoning data quality using influence functions, which measure the causal effect of individual CoT examples on downstream accuracy, and introduce influence-based pruning, which consistently outperforms perplexity and embedding-based baselines on math reasoning within a model family.
>
---
#### [new 113] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies
- **分类: cs.LG; cs.AI; cs.CL; I.2; I.2.7**

- **简介: 该论文属于自然语言生成任务，旨在解决掩码扩散模型（MDM）中解掩顺序对生成性能影响大的问题。现有方法依赖人工设计的启发式策略，效果有限。论文提出通过强化学习框架，将解掩过程建模为KL正则化的马尔可夫决策过程，学习更优的解掩策略。理论证明和实验均表明，该方法优于传统启发式策略，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2510.05725v1](http://arxiv.org/pdf/2510.05725v1)**

> **作者:** Chunsan Hong; Seonho An; Min-Soo Kim; Jong Chul Ye
>
> **备注:** Preprint
>
> **摘要:** Masked diffusion models (MDMs) have recently emerged as a novel framework for language modeling. MDMs generate sentences by iteratively denoising masked sequences, filling in [MASK] tokens step by step. Although MDMs support any-order sampling, performance is highly sensitive to the choice of which position to unmask next. Prior work typically relies on rule-based schedules (e.g., max-confidence, max-margin), which provide ad hoc improvements. In contrast, we replace these heuristics with a learned scheduler. Specifically, we cast denoising as a KL-regularized Markov decision process (MDP) with an explicit reference policy and optimize a regularized objective that admits policy improvement and convergence guarantees under standard assumptions. We prove that the optimized policy under this framework generates samples that more closely match the data distribution than heuristic schedules. Empirically, across four benchmarks, our learned policy consistently outperforms max-confidence: for example, on SUDOKU, where unmasking order is critical, it yields a 20.1% gain over random and a 11.2% gain over max-confidence.
>
---
#### [new 114] Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 论文提出SAT-Graph API，旨在解决法律领域中结构化知识检索的不确定性问题。该工作属于信息检索任务，通过定义可组合、可审计的操作原语，实现从SAT-Graph RAG中进行确定性检索，支持高精度搜索、引用解析、时点版本获取和因果追溯，提升检索过程的透明性和可解释性。**

- **链接: [http://arxiv.org/pdf/2510.06002v1](http://arxiv.org/pdf/2510.06002v1)**

> **作者:** Hudson de Martim
>
> **摘要:** The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core limitations of standard Retrieval-Augmented Generation in the legal domain by providing a verifiable knowledge graph that models hierarchical structure, temporal evolution, and causal events of legal norms. However, a critical gap remains: how to reliably query this structured knowledge without sacrificing its deterministic properties. This paper introduces the SAT-Graph API, a formal query execution layer centered on canonical actions-atomic, composable, and auditable primitives that isolate probabilistic discovery from deterministic retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust reference resolution; (iii) point-in-time version retrieval; and (iv) auditable causal tracing. We demonstrate how planner-guided agents can decompose complex queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer architecture transforms retrieval from an opaque black box to a transparent, auditable process, directly addressing Explainable AI (XAI) requirements for high-stakes domains.
>
---
#### [new 115] Do Code Models Suffer from the Dunning-Kruger Effect?
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 论文探讨了代码生成模型是否存在“达宁-克鲁格效应”（DKE），即能力不足的个体高估自身能力的现象。研究分析了不同编程语言下模型的自信程度与性能，发现低能力和低资源语言场景下模型表现出更强的DKE倾向。该论文属于人工智能认知偏差研究任务，旨在揭示AI模型在编码任务中的判断偏差问题。**

- **链接: [http://arxiv.org/pdf/2510.05457v1](http://arxiv.org/pdf/2510.05457v1)**

> **作者:** Mukul Singh; Somya Chatterjee; Arjun Radhakrishna; Sumit Gulwani
>
> **摘要:** As artificial intelligence systems increasingly collaborate with humans in creative and technical domains, questions arise about the cognitive boundaries and biases that shape our shared agency. This paper investigates the Dunning-Kruger Effect (DKE), the tendency for those with limited competence to overestimate their abilities in state-of-the-art LLMs in coding tasks. By analyzing model confidence and performance across a diverse set of programming languages, we reveal that AI models mirror human patterns of overconfidence, especially in unfamiliar or low-resource domains. Our experiments demonstrate that less competent models and those operating in rare programming languages exhibit stronger DKE-like bias, suggesting that the strength of the bias is proportionate to the competence of the models.
>
---
#### [new 116] ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出ARM框架，旨在优化多智能体系统中的思维链（CoT）推理。通过树搜索与代码空间演化，发现具备反思能力的模块化推理单元，提升MAS在不同模型与任务上的泛化能力，优于手动设计与现有自动方法。**

- **链接: [http://arxiv.org/pdf/2510.05746v1](http://arxiv.org/pdf/2510.05746v1)**

> **作者:** Bohan Yao; Shiva Krishna Reddy Malay; Vikas Yadav
>
> **备注:** 29 pages, 2 figures
>
> **摘要:** Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization.
>
---
#### [new 117] Sci-Phi: A Large Language Model Spatial Audio Descriptor
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出Sci-Phi，一种空间音频大语言模型，旨在解决单通道音频输入在空间理解上的限制。该模型通过双空间与频谱编码器，估计声音来源及环境参数，能描述最多四个定向声源、背景音与房间特性。模型基于合成数据训练，并在真实场景中表现良好，具备实际应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.05542v1](http://arxiv.org/pdf/2510.05542v1)**

> **作者:** Xilin Jiang; Hannes Gamper; Sebastian Braun
>
> **摘要:** Acoustic scene perception involves describing the type of sounds, their timing, their direction and distance, as well as their loudness and reverberation. While audio language models excel in sound recognition, single-channel input fundamentally limits spatial understanding. This work presents Sci-Phi, a spatial audio large language model with dual spatial and spectral encoders that estimates a complete parameter set for all sound sources and the surrounding environment. Learning from over 4,000 hours of synthetic first-order Ambisonics recordings including metadata, Sci-Phi enumerates and describes up to four directional sound sources in one pass, alongside non-directional background sounds and room characteristics. We evaluate the model with a permutation-invariant protocol and 15 metrics covering content, location, timing, loudness, and reverberation, and analyze its robustness across source counts, signal-to-noise ratios, reverberation levels, and challenging mixtures of acoustically, spatially, or temporally similar sources. Notably, Sci-Phi generalizes to real room impulse responses with only minor performance degradation. Overall, this work establishes the first audio LLM capable of full spatial-scene description, with strong potential for real-world deployment. Demo: https://sci-phi-audio.github.io/demo
>
---
#### [new 118] NorMuon: Making Muon more efficient and scalable
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于优化器设计任务，旨在解决大语言模型训练效率与参数利用不平衡问题。提出NorMuon优化器，结合正交化与神经元级自适应学习率，通过行级归一化平衡参数更新，提升训练效率。实验表明其优于Adam和Muon，具备良好扩展性。**

- **链接: [http://arxiv.org/pdf/2510.05491v1](http://arxiv.org/pdf/2510.05491v1)**

> **作者:** Zichong Li; Liming Liu; Chen Liang; Weizhu Chen; Tuo Zhao
>
> **摘要:** The choice of optimizer significantly impacts the training efficiency and computational costs of large language models (LLMs). Recently, the Muon optimizer has demonstrated promising results by orthogonalizing parameter updates, improving optimization geometry through better conditioning. Despite Muon's emergence as a candidate successor to Adam, the potential for jointly leveraging their strengths has not been systematically explored. In this work, we bridge this gap by proposing NorMuon (Neuron-wise Normalized Muon), an optimizer that synergistically combines orthogonalization with neuron-level adaptive learning rates. Our analysis reveals that while Muon effectively reduces condition numbers, the resulting updates exhibit highly non-uniform neuron norms, causing certain neurons to dominate the optimization process. NorMuon addresses this imbalance by maintaining second-order momentum statistics for each neuron and applying row-wise normalization after orthogonalization, ensuring balanced parameter utilization while preserving Muon's conditioning benefits. To enable practical deployment at scale, we develop an efficient distributed implementation under the FSDP2 framework that strategically distributes orthogonalization computations across devices. Experiments across multiple model scales demonstrate that NorMuon consistently outperforms both Adam and Muon, achieving 21.74% better training efficiency than Adam and 11.31% improvement over Muon on 1.1 B pretraining setting, while maintaining a comparable memory footprint to Muon. Our findings suggest that orthogonalization and adaptive learning rates are complementary rather than competing approaches, opening new avenues for optimizer design in large-scale deep learning.
>
---
#### [new 119] AMAQ: Adaptive Mixed-bit Activation Quantization for Collaborative Parameter Efficient Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出AMAQ方法，用于协同参数高效微调任务，解决大模型分布式训练中的通信效率与计算开销问题。通过自适应混合位激活量化，动态分配不同通道的比特数，提升训练稳定性和准确率，降低通信成本。**

- **链接: [http://arxiv.org/pdf/2510.05468v1](http://arxiv.org/pdf/2510.05468v1)**

> **作者:** Yurun Song; Zhuoyi Yang; Ian G. Harris; Sangeetha Abdu Jyothi
>
> **备注:** 14 pages
>
> **摘要:** Large Language Models (LLMs) are scaling rapidly, creating significant challenges for collaborative server client distributed training, particularly in terms of communication efficiency and computational overheads. To address these challenges, we implement Parameter-efficient Split Learning, which effectively balances efficiency and performance for collaborative training on low-resource devices. To reduce communication overhead in collaborative training, we introduce Adaptive Mixed bit Activation Quantization (AMAQ), a strategy that progressively compresses activations and gradients from high precision (6 to 8 bits) to low precision (3 to 4 bits). AMAQ achieves this by effectively allocating bit budgets across channels based on feature wise and layer wise importance using bit regularization. Under the same bit budgets, AMAQ outperforms fixed-precision approaches, delivering about 2.5% higher generation accuracy and about 1.3% better classification accuracy for models like LLaMA3 8B and Qwen2.5 7B. In addition, it significantly enhances training stability and reducing ultra-low bit representation collapse during the training. Experiments demonstrate that AMAQ integrates effectively into practical multi-machine collaborative training setups, offering superior inference accuracy with only a modest communication overhead for bits adaptation during training. This trade off makes AMAQ a practical and effective solution for collaborative training with minimal communication cost.
>
---
#### [new 120] Sample Smart, Not Hard: Correctness-First Decoding for Better Reasoning in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在复杂推理任务中的解码策略问题。现有方法在探索多样性和保证答案质量间存在冲突。论文提出基于正确率估计的解码策略，在不确定性高的步骤降低采样随机性，从而提升推理准确率。**

- **链接: [http://arxiv.org/pdf/2510.05987v1](http://arxiv.org/pdf/2510.05987v1)**

> **作者:** Xueyan Li; Guinan Su; Mrinmaya Sachan; Jonas Geiping
>
> **摘要:** Large Language Models (LLMs) are increasingly applied to complex tasks that require extended reasoning. In such settings, models often benefit from diverse chains-of-thought to arrive at multiple candidate solutions. This requires two competing objectives: to inject enough stochasticity to explore multiple reasoning chains, and to ensure sufficient accuracy and quality in each path. Existing works pursue the first objective by increasing exploration at highly uncertain steps with higher temperature or larger candidate token sets, while others improve reliability by rejecting samples with low confidence post-generation, implying that low confidence correlates with low answer quality. These two lines of thought are in conflict, as they conflate different sources of uncertainty. To resolve this, we argue that the decoding rule should be calibrated by correctness, not confidence alone. We should sample from tokens with higher estimated correctness, and reduce sampling where expected correctness is low. We propose simple strategies that achieve this goal: Greedy-Threshold makes sampling greedy at very low confidence steps. Calibrated-TopK and Calibrated-epsilon set truncation threshold based on estimated rank-wise correctness. Together, our findings challenge prevailing heuristics about decoding under uncertainty and show gains across math and general reasoning benchmarks.
>
---
#### [new 121] Decoding Partial Differential Equations: Cross-Modal Adaptation of Decoder-only Models to PDEs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于科学机器学习任务，旨在解决解码器专用模型在跨模态适应求解偏微分方程（PDE）中的效果差的问题。作者通过引入两种新方法，Parallel Flipping和Sequence Doubling，提升了解码器模型在时间相关PDE模拟任务中的性能，缩小了与编码器模型的差距。**

- **链接: [http://arxiv.org/pdf/2510.05278v1](http://arxiv.org/pdf/2510.05278v1)**

> **作者:** Paloma García-de-Herreros; Philipp Slusallek; Dietrich Klakow; Vagrant Gautam
>
> **摘要:** Large language models have shown great success on natural language tasks in recent years, but they have also shown great promise when adapted to new modalities, e.g., for scientific machine learning tasks. Even though decoder-only models are more popular within NLP and scale exceedingly well at generating natural language, most proposed approaches for cross-modal adaptation focus on encoder-only models, raising the question of how model architecture affects these approaches. In this paper, we therefore perform a series of ablation studies to answer this question, systematically comparing encoder-only and decoder-only models on cross-modal adaptation for time-dependent simulation tasks based on partial differential equations (PDEs). We find that decoder-only models are far worse than encoder-only models, when existing approaches are applied unmodified. In contrast to several other domains, scaling decoder-only models also does not help. To harness the potential of decoder-only models in this context, we introduce two novel approaches, Parallel Flipping and Sequence Doubling, attempting to mimic bidirectionality in autoregressive models. Both our methods improve overall performance using decoder-only models for all tasks and all cross-model adaptation methods, closing the gap to encoder-only model performance. We hope that our findings broaden the spectrum of models used on cross-modal adaptation tasks to further scientific ML.
>
---
#### [new 122] WaveSP-Net: Learnable Wavelet-Domain Sparse Prompt Tuning for Speech Deepfake Detection
- **分类: eess.AS; cs.CL; eess.SP**

- **简介: 该论文属于语音深伪检测任务，旨在解决现有方法依赖全量微调预训练模型、参数效率低且泛化能力弱的问题。作者提出WaveSP-Net，结合基于小波变换的参数高效前端与Mamba架构后端，有效捕捉多尺度特征，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.05305v1](http://arxiv.org/pdf/2510.05305v1)**

> **作者:** Xi Xuan; Xuechen Liu; Wenxin Zhang; Yi-Cheng Lin; Xiaojian Lin; Tomi Kinnunen
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Modern front-end design for speech deepfake detection relies on full fine-tuning of large pre-trained models like XLSR. However, this approach is not parameter-efficient and may lead to suboptimal generalization to realistic, in-the-wild data types. To address these limitations, we introduce a new family of parameter-efficient front-ends that fuse prompt-tuning with classical signal processing transforms. These include FourierPT-XLSR, which uses the Fourier Transform, and two variants based on the Wavelet Transform: WSPT-XLSR and Partial-WSPT-XLSR. We further propose WaveSP-Net, a novel architecture combining a Partial-WSPT-XLSR front-end and a bidirectional Mamba-based back-end. This design injects multi-resolution features into the prompt embeddings, which enhances the localization of subtle synthetic artifacts without altering the frozen XLSR parameters. Experimental results demonstrate that WaveSP-Net outperforms several state-of-the-art models on two new and challenging benchmarks, Deepfake-Eval-2024 and SpoofCeleb, with low trainable parameters and notable performance gains. The code and models are available at https://github.com/xxuan-acoustics/WaveSP-Net.
>
---
#### [new 123] Quantum Concept Music Score from Quantum Picturalism: Musical Incarnation of a Bell-Pair under Measurements
- **分类: quant-ph; cs.CL; math.CT**

- **简介: 该论文提出了一种基于量子图示主义（QPict）的量子音乐理论，称为“量子概念音乐”（QCM），旨在解决传统西方音乐记谱法在线性表达上的局限性。它通过量子力学与音乐的结合，构建了新的音乐形式，强调音乐内部与外部交互关系，适用于作曲、演奏及AI音乐自动化等多个领域。**

- **链接: [http://arxiv.org/pdf/2510.05391v1](http://arxiv.org/pdf/2510.05391v1)**

> **作者:** Rakhat-Bi Abdyssagin; Bob Coecke
>
> **备注:** 6 pages, musical score
>
> **摘要:** We initiate the development of a new language and theory for quantum music, to which we refer as Quantum Concept Music (QCM). This new music formalism is based on Categorical Quantum Mechanics (CQM), and more specifically, its diagrammatic incarnation Quantum Picturalism (QPict), which is heavily based on ZX-calculus. In fact, it is naturally inherited from CQM/QPict. At its heart is the explicit notational representation of relations that exist within and between the key concepts of music composition, performance, and automation. QCM also enables one to directly translate quantum phenomena into music compositions in a both intuitively obvious, rigorous and mechanical manner. Following this pattern, we propose a score for musicians interacting like a Bell-pair under measurement, and outline examples of how it could be live performed. While most of the Western classical music notation has heavily relied on linear representation of music - which does not always adequately capture the nature of music - our approach is distinct by highlighting the fundamental relational dimension of music. In addition, this quantum-based technique not only influences the music at the profound level of composition, but also has a direct impact on a live performance, and also provides a new template for automating music, e.g.~in the context of AI-generation. All together, we initiate the creation of new music formalism that is powerful and efficient in capturing the interactive nature of music, both in terms of internal and external interactions, and goes beyond the boundaries of Western classical music notation, which allows to use it in many different genres and directions.
>
---
#### [new 124] MixReasoning: Switching Modes to Think
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升推理模型效率。针对现有模型在处理复杂问题时对所有步骤均进行深度推理，导致冗余计算的问题，作者提出MixReasoning框架，动态调整推理深度，对关键步骤进行详细推理，对简单步骤进行简略处理，从而缩短推理链，提升效率且不损失准确性。**

- **链接: [http://arxiv.org/pdf/2510.06052v1](http://arxiv.org/pdf/2510.06052v1)**

> **作者:** Haiquan Lu; Gongfan Fang; Xinyin Ma; Qi Li; Xinchao Wang
>
> **摘要:** Reasoning models enhance performance by tackling problems in a step-by-step manner, decomposing them into sub-problems and exploring long chains of thought before producing an answer. However, applying extended reasoning to every step introduces substantial redundancy, as sub-problems vary widely in difficulty and complexity: a small number of pivotal steps are genuinely challenging and decisive for the final answer, while many others only involve straightforward revisions or simple computations. Therefore, a natural idea is to endow reasoning models with the ability to adaptively respond to this variation, rather than treating all steps with the same level of elaboration. To this end, we propose MixReasoning, a framework that dynamically adjusts the depth of reasoning within a single response. The resulting chain of thought then becomes a mixture of detailed reasoning on difficult steps and concise inference on simpler ones. Experiments on GSM8K, MATH-500, and AIME show that MixReasoning shortens reasoning length and substantially improves efficiency without compromising accuracy.
>
---
#### [new 125] Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM搜索代理中因搜索轨迹结构异质性导致的跨层偏差问题。作者提出Stratified GRPO方法，通过分层优势归一化（SAN）在同质轨迹组内评估策略，消除偏差并提升训练稳定性与效果。**

- **链接: [http://arxiv.org/pdf/2510.06214v1](http://arxiv.org/pdf/2510.06214v1)**

> **作者:** Mingkang Zhu; Xi Chen; Bei Yu; Hengshuang Zhao; Jiaya Jia
>
> **摘要:** Large language model (LLM) agents increasingly rely on external tools such as search engines to solve complex, multi-step problems, and reinforcement learning (RL) has become a key paradigm for training them. However, the trajectories of search agents are structurally heterogeneous, where variations in the number, placement, and outcomes of search calls lead to fundamentally different answer directions and reward distributions. Standard policy gradient methods, which use a single global baseline, suffer from what we identify and formalize as cross-stratum bias-an "apples-to-oranges" comparison of heterogeneous trajectories. This cross-stratum bias distorts credit assignment and hinders exploration of complex, multi-step search strategies. To address this, we propose Stratified GRPO, whose central component, Stratified Advantage Normalization (SAN), partitions trajectories into homogeneous strata based on their structural properties and computes advantages locally within each stratum. This ensures that trajectories are evaluated only against their true peers. Our analysis proves that SAN eliminates cross-stratum bias, yields conditionally unbiased unit-variance estimates inside each stratum, and retains the global unbiasedness and unit-variance properties enjoyed by standard normalization, resulting in a more pure and scale-stable learning signal. To improve practical stability under finite-sample regimes, we further linearly blend SAN with the global estimator. Extensive experiments on diverse single-hop and multi-hop question-answering benchmarks demonstrate that Stratified GRPO consistently and substantially outperforms GRPO by up to 11.3 points, achieving higher training rewards, greater training stability, and more effective search policies. These results establish stratification as a principled remedy for structural heterogeneity in RL for LLM search agents.
>
---
#### [new 126] Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices
- **分类: cs.DC; cs.AI; cs.CL; eess.SP**

- **简介: 该论文属于端侧多模态推理任务，旨在解决大模型在小型设备上运行时资源消耗高、效率低的问题。通过软硬件协同设计，将模型拆分为模块并分配至合适加速器，优化计算与内存使用，实现了高效的本地化推理。**

- **链接: [http://arxiv.org/pdf/2510.05109v1](http://arxiv.org/pdf/2510.05109v1)**

> **作者:** Yilong Li; Shuai Zhang; Yijing Zeng; Hao Zhang; Xinmiao Xiong; Jingyu Liu; Pan Hu; Suman Banerjee
>
> **摘要:** Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly half a day and LLaMA-3-8B for voice interactions up to almost 20.8 hours.
>
---
## 更新

#### [replaced 001] MASRAD: Arabic Terminology Management Corpora with Semi-Automatic Construction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.19211v2](http://arxiv.org/pdf/2503.19211v2)**

> **作者:** Mahdi Nasser; Laura Sayyah; Fadi A. Zaraket
>
> **摘要:** This paper presents MASRAD, a terminology dataset for Arabic terminology management, and a method with supporting tools for its semi-automatic construction. The entries in MASRAD are $(f,a)$ pairs of foreign (non-Arabic) terms $f$, appearing in specialized, academic and field-specific books next to their Arabic $a$ counterparts. MASRAD-Ex systematically extracts these pairs as a first step to construct MASRAD. MASRAD helps improving term consistency in academic translations and specialized Arabic documents, and automating cross-lingual text processing. MASRAD-Ex leverages translated terms organically occurring in Arabic books, and considers several candidate pairs for each term phrase. The candidate Arabic terms occur next to the foreign terms, and vary in length. MASRAD-Ex computes lexicographic, phonetic, morphological, and semantic similarity metrics for each candidate pair, and uses heuristic, machine learning, and machine learning with post-processing approaches to decide on the best candidate. This paper presents MASRAD after thorough expert review and makes it available to the interested research community. The best performing MASRAD-Ex approach achieved 90.5% precision and 92.4% recall.
>
---
#### [replaced 002] A Generative Approach to LLM Harmfulness Mitigation with Red Flag Tokens
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.16366v4](http://arxiv.org/pdf/2502.16366v4)**

> **作者:** David Dobre; Mehrnaz Mofakhami; Sophie Xhonneux; Leo Schwinn; Gauthier Gidel
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Many safety post-training methods for large language models (LLMs) are designed to modify the model's behaviour from producing unsafe answers to issuing refusals. However, such distribution shifts are often brittle and degrade performance on desirable tasks. To address these pitfalls, we propose augmenting the model's vocabulary with a special red flag token, and training the model to insert this token whenever harmful content is generated or imminent. This approach enables the model to explicitly learn the concept of harmfulness in its representations, with minimal impact on utility due to the marginal change in the generated distribution of natural language. Moreover, because the token is embedded in the model's vocabulary, we can naturally leverage the LLMs' generalization capabilities, such as in-context learning (ICL) and out-of-distribution generalization to languages that are not formally supported (e.g., Japanese for Llama3). In particular, we demonstrate that through ICL alone, the model can learn to initiate reflective reasoning upon generating the red flag token at inference, which steers the response away from harmful continuations or enables self-correction when the flag is raised falsely. This approach is orthogonal and complementary to existing safety technique (such as safety classifiers or standard safety training) and easier to evaluate in comparison to natural language refusals, as it does not require a human or automated judge to assess the harmlessness of the answers.
>
---
#### [replaced 003] How Reliable are Causal Probing Interventions?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.15510v4](http://arxiv.org/pdf/2408.15510v4)**

> **作者:** Marc Canby; Adam Davies; Chirag Rastogi; Julia Hockenmaier
>
> **摘要:** Causal probing aims to analyze foundation models by examining how intervening on their representation of various latent properties impacts their outputs. Recent works have cast doubt on the theoretical basis of several leading causal probing methods, but it has been unclear how to systematically evaluate the effectiveness of these methods in practice. To address this, we define two key causal probing desiderata: completeness (how thoroughly the representation of the target property has been transformed) and selectivity (how little non-targeted properties have been impacted). We find that there is an inherent tradeoff between the two, which we define as reliability, their harmonic mean. We introduce an empirical analysis framework to measure and evaluate these quantities, allowing us to make the first direct comparisons between different families of leading causal probing methods (e.g., linear vs. nonlinear, or concept removal vs. counterfactual interventions). We find that: (1) all methods show a clear tradeoff between completeness and selectivity; (2) more complete and reliable methods have a greater impact on LLM behavior; and (3) nonlinear interventions are almost always more reliable than linear interventions.
>
---
#### [replaced 004] MathVC: An LLM-Simulated Multi-Character Virtual Classroom for Mathematics Education
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2404.06711v3](http://arxiv.org/pdf/2404.06711v3)**

> **作者:** Murong Yue; Wenhan Lyu; Jennifer Suh; Yixuan Zhang; Ziyu Yao
>
> **备注:** Accepted by AAAI 2025 workshop
>
> **摘要:** Collaborative problem solving (CPS) is essential in mathematics education, fostering deeper learning through the exchange of ideas. Yet, classrooms often lack the resources, time, and peer dynamics needed to sustain productive CPS. Recent advancements in Large Language Models (LLMs) offer a promising avenue to enhance CPS in mathematical education. We designed and developed MathVC, a multi-persona LLM simulated virtual classroom platform to facilitate CPS in mathematics. MathVC combines a meta planning controller that monitors CPS stages-sense-making, team organization, planning, execution, validation, and predicts the next speaker, with a persona simulation stack that encodes mathematical thinking via a task schema and error-injected persona schemas seeded from teacher-specified misconceptions. We evaluated MathVC with 14 U.S. middle schoolers. Students reported constructive interaction and reaching shared solutions, describing gains in engagement, motivation, and confidence through diverse perspectives, immediate scaffolding, and human-like fallibility. Our findings also provide insights into simulating peers via LLM-based technologies for collaboration to support learning.
>
---
#### [replaced 005] LaB-RAG: Label Boosted Retrieval Augmented Generation for Radiology Report Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.16523v2](http://arxiv.org/pdf/2411.16523v2)**

> **作者:** Steven Song; Anirudh Subramanyam; Irene Madejski; Robert L. Grossman
>
> **摘要:** In the current paradigm of image captioning, deep learning models are trained to generate text from image embeddings of latent features. We challenge the assumption that fine-tuning of large, bespoke models is required to improve model generation accuracy. Here we propose Label Boosted Retrieval Augmented Generation (LaB-RAG), a small-model-based approach to image captioning that leverages image descriptors in the form of categorical labels to boost standard retrieval augmented generation (RAG) with pretrained large language models (LLMs). We study our method in the context of radiology report generation (RRG) over MIMIC-CXR and CheXpert Plus. We argue that simple classification models combined with zero-shot embeddings can effectively transform X-rays into text-space as radiology-specific labels. In combination with standard RAG, we show that these derived text labels can be used with general-domain LLMs to generate radiology reports. Without ever training our generative language model or image embedding models specifically for the task, and without ever directly "showing" the LLM an X-ray, we demonstrate that LaB-RAG achieves better results across natural language and radiology language metrics compared with other retrieval-based RRG methods, while attaining competitive results compared to other fine-tuned vision-language RRG models. We further conduct extensive ablation experiments to better understand the components of LaB-RAG. Our results suggest broader compatibility and synergy with fine-tuned methods to further enhance RRG performance.
>
---
#### [replaced 006] Robustness of Large Language Models to Perturbations in Text
- **分类: cs.CL; cs.AI; I.7; I.2.7; I.2.4**

- **链接: [http://arxiv.org/pdf/2407.08989v2](http://arxiv.org/pdf/2407.08989v2)**

> **作者:** Ayush Singh; Navpreet Singh; Shubham Vatsal
>
> **备注:** 8 pages, 1 figure, 6 tables, updated with results also from GPT-4, LLaMa-3
>
> **摘要:** Having a clean dataset has been the foundational assumption of most natural language processing (NLP) systems. However, properly written text is rarely found in real-world scenarios and hence, oftentimes invalidates the aforementioned foundational assumption. Recently, Large language models (LLMs) have shown impressive performance, but can they handle the inevitable noise in real-world data? This work tackles this critical question by investigating LLMs' resilience against morphological variations in text. To that end, we artificially introduce varying levels of noise into a diverse set of datasets and systematically evaluate LLMs' robustness against the corrupt variations of the original text. Our findings show that contrary to popular beliefs, generative LLMs are quiet robust to noisy perturbations in text. This is a departure from pre-trained models like BERT or RoBERTa whose performance has been shown to be sensitive to deteriorating noisy text. Additionally, we test LLMs' resilience on multiple real-world benchmarks that closely mimic commonly found errors in the wild. With minimal prompting, LLMs achieve a new state-of-the-art on the benchmark tasks of Grammar Error Correction (GEC) and Lexical Semantic Change (LSC). To empower future research, we also release a dataset annotated by humans stating their preference for LLM vs. human-corrected outputs along with the code to reproduce our results.
>
---
#### [replaced 007] What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts
- **分类: cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2505.13360v2](http://arxiv.org/pdf/2505.13360v2)**

> **作者:** Chenyang Yang; Yike Shi; Qianou Ma; Michael Xieyang Liu; Christian Kästner; Tongshuang Wu
>
> **摘要:** Prompt underspecification is a common challenge when interacting with LLMs. In this paper, we present an in-depth analysis of this problem, showing that while LLMs can often infer unspecified requirements by default (41.1%), such behavior is fragile: Under-specified prompts are 2x as likely to regress across model or prompt changes, sometimes with accuracy drops exceeding 20%. This instability makes it difficult to reliably build LLM applications. Moreover, simply specifying all requirements does not consistently help, as models have limited instruction-following ability and requirements can conflict. Standard prompt optimizers likewise provide little benefit. To address these issues, we propose requirements-aware prompt optimization mechanisms that improve performance by 4.8% on average over baselines. We further advocate for a systematic process of proactive requirements discovery, evaluation, and monitoring to better manage prompt underspecification in practice.
>
---
#### [replaced 008] Can Video Large Multimodal Models Think Like Doubters-or Double-Down: A Study on Defeasible Video Entailment
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22385v2](http://arxiv.org/pdf/2506.22385v2)**

> **作者:** Yue Zhang; Jilei Sun; Yunhui Guo; Vibhav Gogate
>
> **摘要:** Video Large Multimodal Models (VLMMs) have made impressive strides in understanding video content, but they often struggle with abstract and adaptive reasoning-the ability to revise their interpretations when new information emerges. In reality, conclusions are rarely set in stone; additional context can strengthen or weaken an initial inference. To address this, we introduce Defeasible Video Entailment (DVidE), a new task that challenges models to think like doubters, constantly updating their reasoning based on evolving evidence. In DVidE, given a video premise and a textual hypothesis, models must determine whether a new update strengthens or weakens the hypothesis (classification version) or generate a coherent update that modifies the entailment relationship (generation version). For solving the classification task, we propose the Chain of Counterfactual Thought framework, utilizing counterfactual reasoning, ASR-enhanced video content, and rationale refinement to reduce inference bias. For the generation task, we develop a framework that combines ASR output with a Large Language Model (LLM) to produce coherent, contextually relevant updates aligned with the intended strengthener or weakener goals. Additionally, we introduce a novel benchmark dataset, with strengthener/weakener annotations and an LLM-based evaluation metric specifically designed for assessing generative performance. Experimental results demonstrate significant improvements, highlighting our proposed method in enhancing dynamic reasoning capabilities of VLMMs.
>
---
#### [replaced 009] Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12428v2](http://arxiv.org/pdf/2507.12428v2)**

> **作者:** Yik Siu Chan; Zheng-Xin Yong; Stephen H. Bach
>
> **摘要:** Reasoning language models improve performance on complex tasks by generating long chains of thought (CoTs), but this process can also increase harmful outputs in adversarial settings. In this work, we ask whether the long CoTs can be leveraged for predictive safety monitoring: do the reasoning traces provide early signals of final response alignment that could enable timely intervention? We evaluate a range of monitoring methods using either CoT text or activations, including highly capable large language models, fine-tuned classifiers, and humans. First, we find that a simple linear probe trained on CoT activations significantly outperforms all text-based baselines in predicting whether a final response is safe or unsafe, with an average absolute increase of 13 in F1 scores over the best-performing alternatives. CoT texts are often unfaithful and misleading, while model latents provide a more reliable predictive signal. Second, the probe can be applied to early CoT segments before the response is generated, showing that alignment signals appear before reasoning completes. Error analysis reveals that the performance gap between text classifiers and the linear probe largely stems from a subset of responses we call performative CoTs, where the reasoning consistently contradicts the final response as the CoT progresses. Our findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation.
>
---
#### [replaced 010] ExpertLongBench: Benchmarking Language Models on Expert-Level Long-Form Generation Tasks with Structured Checklists
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01241v3](http://arxiv.org/pdf/2506.01241v3)**

> **作者:** Jie Ruan; Inderjeet Nair; Shuyang Cao; Amy Liu; Sheza Munir; Micah Pollens-Dempsey; Tiffany Chiang; Lucy Kates; Nicholas David; Sihan Chen; Ruxin Yang; Yuqian Yang; Jasmine Gump; Tessa Bialek; Vivek Sankaran; Margo Schlanger; Lu Wang
>
> **摘要:** This paper introduces ExpertLongBench, an expert-level benchmark containing 11 tasks from 9 domains that reflect realistic expert workflows and applications. Beyond question answering, the application-driven tasks in ExpertLongBench demand long-form outputs that can exceed 5,000 tokens and strict adherence to domain-specific requirements. Notably, each task in ExpertLongBench includes a rubric, designed or validated by domain experts, to specify task requirements and guide output evaluation. Furthermore, we propose CLEAR, an evaluation framework that supports accurate evaluation of long-form model outputs in our benchmark. To achieve fine-grained, expert-aligned evaluation, CLEAR derives checklists from both model outputs and references by extracting information corresponding to items in the task-specific rubric. Checklist items of model outputs are then compared with corresponding items of reference outputs to assess their correctness, enabling grounded evaluation. We benchmark 13 popular large language models (LLMs) and analyze components in CLEAR, showing that (1) existing LLMs, with the top performer Gemini-2.5-Pro achieving only a 33.4 F1 score, require significant improvement for expert-level tasks; (2) models can generate content corresponding to the required aspects, but far from correct; and (3) accurate checklist extraction and comparison in CLEAR can be achieved by open-weight models for more scalable, reproducible, and low-cost usage.
>
---
#### [replaced 011] PLSemanticsBench: Large Language Models As Programming Language Interpreters
- **分类: cs.PL; cs.AI; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2510.03415v2](http://arxiv.org/pdf/2510.03415v2)**

> **作者:** Aditya Thimmaiah; Jiyang Zhang; Jayanth Srinivasa; Junyi Jessy Li; Milos Gligoric
>
> **摘要:** As large language models (LLMs) excel at code reasoning, a natural question arises: can an LLM execute programs (i.e., act as an interpreter) purely based on a programming language's formal semantics? If so, it will enable rapid prototyping of new programming languages and language features. We study this question using the imperative language IMP (a subset of C), formalized via small-step operational semantics (SOS) and rewriting-based operational semantics (K-semantics). We introduce three evaluation sets-Human-Written, LLM-Translated, and Fuzzer- Generated-whose difficulty is controlled by code-complexity metrics spanning the size, control-flow, and data-flow axes. Given a program and its semantics formalized with SOS/K-semantics, models are evaluated on three tasks ranging from coarse to fine: (1) final-state prediction, (2) semantic rule prediction, and (3) execution trace prediction. To distinguish pretraining memorization from semantic competence, we define two nonstandard semantics obtained through systematic mutations of the standard rules. Across strong code/reasoning LLMs, performance drops under nonstandard semantics despite high performance under the standard one. We further find that (i) there are patterns to different model failures, (ii) most reasoning models perform exceptionally well on coarse grained tasks involving reasoning about highly complex programs often containing nested loop depths beyond five, and surprisingly, (iii) providing formal semantics helps on simple programs but often hurts on more complex ones. Overall, the results show a promise that LLMs could serve as programming language interpreters, but points to the lack of their robust semantics understanding. We release the benchmark and the supporting code at https://github.com/EngineeringSoftware/PLSemanticsBench.
>
---
#### [replaced 012] Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.03502v2](http://arxiv.org/pdf/2503.03502v2)**

> **作者:** Canaan Yung; Hanxun Huang; Christopher Leckie; Sarah Erfani
>
> **备注:** 40 Pages, 6 figues
>
> **摘要:** Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.
>
---
#### [replaced 013] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.01367v3](http://arxiv.org/pdf/2510.01367v3)**

> **作者:** Xinpeng Wang; Nitish Joshi; Barbara Plank; Rico Angell; He He
>
> **备注:** 25 pages, 31 figures
>
> **摘要:** Reward hacking, where a reasoning model exploits loopholes in a reward function to achieve high rewards without solving the intended task, poses a significant threat. This behavior may be explicit, i.e. verbalized in the model's chain-of-thought (CoT), or implicit, where the CoT appears benign thus bypasses CoT monitors. To detect implicit reward hacking, we propose TRACE (Truncated Reasoning AUC Evaluation). Our key observation is that hacking occurs when exploiting the loophole is easier than solving the actual task. This means that the model is using less 'effort' than required to achieve high reward. TRACE quantifies effort by measuring how early a model's reasoning becomes sufficient to obtain the reward. We progressively truncate a model's CoT at various lengths, force the model to answer, and estimate the expected reward at each cutoff. A hacking model, which takes a shortcut, will achieve a high expected reward with only a small fraction of its CoT, yielding a large area under the accuracy-vs-length curve. TRACE achieves over 65% gains over our strongest 72B CoT monitor in math reasoning, and over 30% gains over a 32B monitor in coding. We further show that TRACE can discover unknown loopholes during training. Overall, TRACE offers a scalable unsupervised approach for oversight where current monitoring methods prove ineffective.
>
---
#### [replaced 014] LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04573v2](http://arxiv.org/pdf/2510.04573v2)**

> **作者:** Haoqiang Kang; Yizhe Zhang; Nikki Lijing Kuang; Nicklas Majamaki; Navdeep Jaitly; Yi-An Ma; Lianhui Qin
>
> **摘要:** Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
>
---
#### [replaced 015] LLM Unlearning Without an Expert Curated Dataset
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.06595v3](http://arxiv.org/pdf/2508.06595v3)**

> **作者:** Xiaoyuan Zhu; Muru Zhang; Ollie Liu; Robin Jia; Willie Neiswanger
>
> **摘要:** Modern large language models often encode sensitive, harmful, or copyrighted knowledge, raising the need for post-hoc unlearning-the ability to remove specific domains of knowledge from a model without full retraining. A major bottleneck in current unlearning pipelines is constructing effective forget sets-datasets that approximate the target domain and guide the model to forget it. In this work, we introduce a scalable, automated approach to generate high-quality forget sets using language models themselves. Our method synthesizes textbook-style data through a structured prompting pipeline, requiring only a domain name as input. Through experiments on unlearning biosecurity, cybersecurity, and Harry Potter novels, we show that our synthetic datasets consistently outperform the baseline synthetic alternatives and are comparable to the expert-curated ones. Additionally, ablation studies reveal that the multi-step generation pipeline significantly boosts data diversity, which in turn improves unlearning utility. Overall, our findings suggest that synthetic datasets offer a promising path toward practical, scalable unlearning for a wide range of emerging domains without the need for manual intervention. We release our code and dataset at https://github.com/xyzhu123/Synthetic_Textbook.
>
---
#### [replaced 016] SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02725v2](http://arxiv.org/pdf/2504.02725v2)**

> **作者:** Kehua Feng; Keyan Ding; Yuhao Wang; Menghan Li; Fanjunduo Wei; Xinda Wang; Qiang Zhang; Huajun Chen
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.
>
---
#### [replaced 017] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23495v3](http://arxiv.org/pdf/2505.23495v3)**

> **作者:** Liangliang Zhang; Zhuorui Jiang; Hongliang Chi; Haoyang Chen; Mohammed Elkoumy; Fali Wang; Qiong Wu; Zhengyi Zhou; Shirui Pan; Suhang Wang; Yao Ma
>
> **备注:** Accepted at NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Knowledge Graph Question Answering (KGQA) systems rely on high-quality benchmarks to evaluate complex multi-hop reasoning. However, despite their widespread use, popular datasets such as WebQSP and CWQ suffer from critical quality issues, including inaccurate or incomplete ground-truth annotations, poorly constructed questions that are ambiguous, trivial, or unanswerable, and outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA datasets, including WebQSP and CWQ, we find that the average factual correctness rate is only 57 %. To address these issues, we introduce KGQAGen, an LLM-in-the-loop framework that systematically resolves these pitfalls. KGQAGen combines structured knowledge grounding, LLM-guided generation, and symbolic verification to produce challenging and verifiable QA instances. Using KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results demonstrate that even state-of-the-art systems struggle on this benchmark, highlighting its ability to expose limitations of existing models. Our findings advocate for more rigorous benchmark construction and position KGQAGen as a scalable framework for advancing KGQA evaluation.
>
---
#### [replaced 018] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.15228v2](http://arxiv.org/pdf/2501.15228v2)**

> **作者:** Yiqun Chen; Lingyong Yan; Weiwei Sun; Xinyu Ma; Yi Zhang; Shuaiqiang Wang; Dawei Yin; Yiming Yang; Jiaxin Mao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Retrieval-augmented generation (RAG) is widely utilized to incorporate external knowledge into large language models, thereby enhancing factuality and reducing hallucinations in question-answering (QA) tasks. A standard RAG pipeline consists of several components, such as query rewriting, document retrieval, document filtering, and answer generation. However, these components are typically optimized separately through supervised fine-tuning, which can lead to misalignments between the objectives of individual components and the overarching aim of generating accurate answers. Although recent efforts have explored using reinforcement learning (RL) to optimize specific RAG components, these approaches often focus on simple pipelines with only two components or do not adequately address the complex interdependencies and collaborative interactions among the modules. To overcome these limitations, we propose treating the complex RAG pipeline with multiple components as a multi-agent cooperative task, in which each component can be regarded as an RL agent. Specifically, we present MMOA-RAG, Multi-Module joint Optimization Algorithm for RAG, which employs multi-agent reinforcement learning to harmonize all agents' goals toward a unified reward, such as the F1 score of the final answer. Experiments conducted on various QA benchmarks demonstrate that MMOA-RAG effectively boost the overall performance of the pipeline and outperforms existing baselines. Furthermore, comprehensive ablation studies validate the contributions of individual components and demonstrate MMOA-RAG can be adapted to different RAG pipelines and benchmarks.
>
---
#### [replaced 019] SciKnowEval: Evaluating Multi-level Scientific Knowledge of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.09098v4](http://arxiv.org/pdf/2406.09098v4)**

> **作者:** Kehua Feng; Xinyi Shen; Weijie Wang; Xiang Zhuang; Yuqi Tang; Qiang Zhang; Keyan Ding
>
> **备注:** 33 pages, 2 figures
>
> **摘要:** Large language models (LLMs) are playing an increasingly important role in scientific research, yet there remains a lack of comprehensive benchmarks to evaluate the breadth and depth of scientific knowledge embedded in these models. To address this gap, we introduce SciKnowEval, a large-scale dataset designed to systematically assess LLMs across five progressive levels of scientific understanding: memory, comprehension, reasoning, discernment, and application. SciKnowEval comprises 28K multi-level questions and solutions spanning biology, chemistry, physics, and materials science. Using this benchmark, we evaluate 20 leading open-source and proprietary LLMs. The results show that while proprietary models often achieve state-of-the-art performance, substantial challenges remain -- particularly in scientific reasoning and real-world application. We envision SciKnowEval as a standard benchmark for evaluating scientific capabilities in LLMs and as a catalyst for advancing more capable and reliable scientific language models.
>
---
#### [replaced 020] Building Resource-Constrained Language Agents: A Korean Case Study on Chemical Toxicity Information
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17753v2](http://arxiv.org/pdf/2503.17753v2)**

> **作者:** Hojun Cho; Donghu Kim; Soyoung Yang; Chan Lee; Hunjoo Lee; Jaegul Choo
>
> **备注:** EMNLP 2025 Industry track
>
> **摘要:** Language agents powered by large language models (LLMs) face significant deployment challenges in resource-constrained environments, particularly for specialized domains and less-common languages. This paper presents Tox-chat, a Korean chemical toxicity information agent devised within these limitations. We propose two key innovations: a context-efficient architecture that reduces token consumption through hierarchical section search, and a scenario-based dialogue generation methodology that effectively distills tool-using capabilities from larger models. Experimental evaluations demonstrate that our fine-tuned 8B parameter model substantially outperforms both untuned models and baseline approaches, in terms of DB faithfulness and preference. Our work offers valuable insights for researchers developing domain-specific language agents under practical constraints.
>
---
#### [replaced 021] Aligning Language Models with Real-time Knowledge Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.01302v2](http://arxiv.org/pdf/2508.01302v2)**

> **作者:** Chenming Tang; Yutong Yang; Kexue Wang; Yunfang Wu
>
> **备注:** Pre-print
>
> **摘要:** Knowledge editing aims to modify outdated knowledge in large language models (LLMs) efficiently while retaining their original capabilities. Mainstream benchmarks for knowledge editing are predominantly static and fail to keep in pace with the evolving real-world knowledge. In this work, we introduce CRAFT, an ever-evolving real-world benchmark for knowledge editing. It features well-designed paired edits for composite reasoning, and evaluates models on alias portability as well as temporal and common-sense locality, making it a challenging knowledge editing benchmark on which previous knowledge editing methods hardly achieve balanced performance. Towards flexible real-time editing, we propose KEDAS, a novel paradigm of knowledge editing alignment featuring diverse edit augmentation and self-adaptive post-alignment inference, which exhibits significant performance gain on CRAFT compared to previous methods. All of our code and data are available at https://anonymous.4open.science/r/CRAFT-KEDAS.
>
---
#### [replaced 022] WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13312v3](http://arxiv.org/pdf/2509.13312v3)**

> **作者:** Zijian Li; Xin Guan; Bo Zhang; Shen Huang; Houquan Zhou; Shaopeng Lai; Ming Yan; Yong Jiang; Pengjun Xie; Fei Huang; Jun Zhang; Jingren Zhou
>
> **备注:** An agent system for open-ended deep research
>
> **摘要:** This paper tackles \textbf{open-ended deep research (OEDR)}, a complex challenge where AI agents must synthesize vast web-scale information into insightful reports. Current approaches are plagued by dual-fold limitations: static research pipelines that decouple planning from evidence acquisition and monolithic generation paradigms that include redundant, irrelevant evidence, suffering from hallucination issues and low citation accuracy. To address these challenges, we introduce \textbf{WebWeaver}, a novel dual-agent framework that emulates the human research process. The planner operates in a dynamic cycle, iteratively interleaving evidence acquisition with outline optimization to produce a comprehensive, citation-grounded outline linking to a memory bank of evidence. The writer then executes a hierarchical retrieval and writing process, composing the report section by section. By performing targeted retrieval of only the necessary evidence from the memory bank via citations for each part, it effectively mitigates long-context issues and citation hallucinations. Our framework establishes a new state-of-the-art across major OEDR benchmarks, including DeepResearch Bench, DeepConsult, and DeepResearchGym. These results validate our human-centric, iterative methodology, demonstrating that adaptive planning and focused synthesis are crucial for producing comprehensive, trusted, and well-structured reports.
>
---
#### [replaced 023] Unifying Inference-Time Planning Language Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14763v2](http://arxiv.org/pdf/2505.14763v2)**

> **作者:** Prabhu Prakash Kagitha; Bo Sun; Ishan Desai; Andrew Zhu; Cassie Huang; Manling Li; Ziyang Li; Li Zhang
>
> **摘要:** A line of work in planning uses LLM not to generate a plan, but to generate a formal representation in some planning language, which can be input into a symbolic solver to deterministically find a plan. While showing improved trust and promising performance, dozens of recent publications have proposed scattered methods on a variety of benchmarks under different experimental settings. We attempt to unify the inference-time LLM-as-formalizer methodology for classical planning by proposing a unifying framework based on intermediate representations. We thus systematically evaluate more than a dozen pipelines that subsume most existing work, while proposing novel ones that involve syntactically similar but high resource intermediate languages (such as a Python wrapper of PDDL). We provide recipes for planning language generation pipelines, draw a series of conclusions showing the efficacy of their various components, and evidence their robustness against problem complexity.
>
---
#### [replaced 024] An Embarrassingly Simple Defense Against LLM Abliteration Attacks
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19056v2](http://arxiv.org/pdf/2505.19056v2)**

> **作者:** Harethah Abu Shairah; Hasan Abed Al Kader Hammoud; Bernard Ghanem; George Turkiyyah
>
> **备注:** preprint - under review
>
> **摘要:** Large language models (LLMs) are typically aligned to refuse harmful instructions through safety fine-tuning. A recent attack, termed abliteration, identifies and suppresses the single latent direction most responsible for refusal behavior, thereby enabling models to generate harmful content. We propose a defense that fundamentally alters how models express refusal. We construct an extended-refusal dataset in which responses to harmful prompts provide detailed justifications before refusing, distributing the refusal signal across multiple token positions. Fine-tuning Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on this dataset yields models that maintain high refusal rates under abliteration: refusal rates drop by at most 10%, compared to 70-80% drops in baseline models. Comprehensive evaluations of safety and utility demonstrate that extended-refusal fine-tuning effectively neutralizes abliteration attacks while preserving general model performance and enhancing robustness across multiple alignment scenarios.
>
---
#### [replaced 025] Towards Locally Deployable Fine-Tuned Causal Large Language Models for Mode Choice Behaviour
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.21432v2](http://arxiv.org/pdf/2507.21432v2)**

> **作者:** Tareq Alsaleh; Bilal Farooq
>
> **摘要:** This study investigates the adoption of open-access, locally deployable causal large language models (LLMs) for travel mode choice prediction and introduces LiTransMC, the first fine-tuned causal LLM developed for this task. We systematically benchmark eleven open-access LLMs (1-12B parameters) across three stated and revealed preference datasets, testing 396 configurations and generating over 79,000 mode choice decisions. Beyond predictive accuracy, we evaluate models generated reasoning using BERTopic for topic modelling and a novel Explanation Strength Index, providing the first structured analysis of how LLMs articulate decision factors in alignment with behavioural theory. LiTransMC, fine-tuned using parameter efficient and loss masking strategy, achieved a weighted F1 score of 0.6845 and a Jensen-Shannon Divergence of 0.000245, surpassing both untuned local models and larger proprietary systems, including GPT-4o with advanced persona inference and embedding-based loading, while also outperforming classical mode choice methods such as discrete choice models and machine learning classifiers for the same dataset. This dual improvement, i.e., high instant-level accuracy and near-perfect distributional calibration, demonstrates the feasibility of creating specialist, locally deployable LLMs that integrate prediction and interpretability. Through combining structured behavioural prediction with natural language reasoning, this work unlocks the potential for conversational, multi-task transport models capable of supporting agent-based simulations, policy testing, and behavioural insight generation. These findings establish a pathway for transforming general purpose LLMs into specialized and explainable tools for transportation research and policy formulation, while maintaining privacy, reducing cost, and broadening access through local deployment.
>
---
#### [replaced 026] A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07086v2](http://arxiv.org/pdf/2504.07086v2)**

> **作者:** Andreas Hochlehnert; Hardik Bhatnagar; Vishaal Udandarao; Samuel Albanie; Ameya Prabhu; Matthias Bethge
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Reasoning has emerged as the next major frontier for language models (LMs), with rapid advances from both academic and industrial labs. However, this progress often outpaces methodological rigor, with many evaluations relying on benchmarking practices that lack transparency, robustness, or statistical grounding. In this work, we conduct a comprehensive empirical study and find that current mathematical reasoning benchmarks are highly sensitive to subtle implementation choices--including decoding parameters, random seeds, prompt formatting, and even hardware and software configurations. Performance gains reported in recent studies frequently hinge on unclear comparisons or unreported sources of variance. To address these issues, we propose a standardized evaluation framework with clearly defined best practices and reporting standards. Using this framework, we reassess recent methods and find that most reinforcement learning (RL) approaches yield only modest improvements--far below prior claims--and are prone to overfitting, especially on small-scale benchmarks like AIME'24. In contrast, supervised finetuning (SFT) methods show consistently stronger generalization in the settings we study. To foster reproducibility, we release all code, prompts, and model outputs, for reasoning benchmarks, establishing more rigorous foundations for future work.
>
---
#### [replaced 027] Cross-Document Cross-Lingual NLI via RST-Enhanced Graph Fusion and Interpretability Prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12324v3](http://arxiv.org/pdf/2504.12324v3)**

> **作者:** Mengying Yuan; Wenhao Wang; Zixuan Wang; Yujie Huang; Kangli Wei; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** EMNLP 2025 Main (Camera Ready)
>
> **摘要:** Natural Language Inference (NLI) is a fundamental task in natural language processing. While NLI has developed many sub-directions such as sentence-level NLI, document-level NLI and cross-lingual NLI, Cross-Document Cross-Lingual NLI (CDCL-NLI) remains largely unexplored. In this paper, we propose a novel paradigm: CDCL-NLI, which extends traditional NLI capabilities to multi-document, multilingual scenarios. To support this task, we construct a high-quality CDCL-NLI dataset including 25,410 instances and spanning 26 languages. To address the limitations of previous methods on CDCL-NLI task, we further propose an innovative method that integrates RST-enhanced graph fusion with interpretability-aware prediction. Our approach leverages RST (Rhetorical Structure Theory) within heterogeneous graph neural networks for cross-document context modeling, and employs a structure-aware semantic alignment based on lexical chains for cross-lingual understanding. For NLI interpretability, we develop an EDU (Elementary Discourse Unit)-level attribution framework that produces extractive explanations. Extensive experiments demonstrate our approach's superior performance, achieving significant improvements over both conventional NLI models as well as large language models. Our work sheds light on the study of NLI and will bring research interest on cross-document cross-lingual context understanding, hallucination elimination and interpretability inference. Our code and datasets are available at "https://github.com/Leonardo123-ui/CDCL_NLI" for peer review.
>
---
#### [replaced 028] Do AI Models Perform Human-like Abstract Reasoning Across Modalities?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.02125v3](http://arxiv.org/pdf/2510.02125v3)**

> **作者:** Claas Beger; Ryan Yi; Shuhao Fu; Arseny Moskvichev; Sarah W. Tsai; Sivasankaran Rajamanickam; Melanie Mitchell
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** OpenAI's o3-preview reasoning model exceeded human accuracy on the ARC-AGI benchmark, but does that mean state-of-the-art models recognize and reason with the abstractions that the task creators intended? We investigate models' abstraction abilities on ConceptARC. We evaluate models under settings that vary the input modality (textual vs. visual), whether the model is permitted to use external Python tools, and, for reasoning models, the amount of reasoning effort. In addition to measuring output accuracy, we perform fine-grained evaluation of the natural-language rules that models generate to explain their solutions. This dual evaluation lets us assess whether models solve tasks using the abstractions ConceptARC was designed to elicit, rather than relying on surface-level patterns. Our results show that, while some models using text-based representations match human output accuracy, the best models' rules are often based on surface-level ``shortcuts'' and capture intended abstractions far less often than humans. Thus their capabilities for general abstract reasoning may be overestimated by evaluations based on accuracy alone. In the visual modality, AI models' output accuracy drops sharply, yet our rule-level analysis reveals that models might be underestimated, as they still exhibit a substantial share of rules that capture intended abstractions, but are often unable to correctly apply these rules. In short, our results show that models still lag humans in abstract reasoning, and that using accuracy alone to evaluate abstract reasoning on ARC-like tasks may overestimate abstract-reasoning capabilities in textual modalities and underestimate it in visual modalities. We believe that our evaluation framework offers a more faithful picture of multimodal models' abstract reasoning abilities and a more principled way to track progress toward human-like, abstraction-centered intelligence.
>
---
#### [replaced 029] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02322v2](http://arxiv.org/pdf/2508.02322v2)**

> **作者:** Yuzhuang Xu; Xu Han; Yuanchi Zhang; Yixuan Wang; Yijun Liu; Shiyu Ji; Qingfu Zhu; Wanxiang Che
>
> **备注:** 16 pages, 9 figures, 7 tables
>
> **摘要:** Large Language Models (LLMs) with Mixture-of-Experts (MoE) architectures are distinguished by their strong performance scaling with increasing parameters across a wide range of tasks, yet they also suffer from substantial computational and storage overheads. Notably, the performance gains of MoE models do not scale proportionally with the growth in expert parameters. While prior works attempt to reduce parameters via expert-level pruning, merging, or decomposition, they still suffer from challenges in both performance and computational efficiency. In this paper, we address these challenges by introducing micro-expert as a finer-grained compression unit that spans across matrices. We first establish a more fundamental perspective, viewing MoE layers as mixtures of micro-experts, and present CAMERA, a lightweight and training-free framework for identifying micro-expert redundancy. Our analysis uncovers significant variance in micro-expert contributions during decoding. Based on this insight, we further propose CAMERA-P, a structured micro-expert pruning framework, and CAMERA-Q, a mixed-precision quantization idea designed for micro-experts. Extensive experiments on nine downstream tasks show that CAMERA-P consistently outperforms strong baselines under pruning ratios ranging from 20% to 60%. Furthermore, CAMERA-Q achieves superior results under aggressive 2-bit quantization, surpassing existing matrix- and channel-level ideas. Notably, our method enables complete micro-expert analysis of Qwen2-57B-A14B in less than 5 minutes on a single NVIDIA A100-40GB GPU.
>
---
#### [replaced 030] Exploring the Potential of Conversational AI Support for Agent-Based Social Simulation Model Design
- **分类: cs.HC; cs.AI; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2405.08032v2](http://arxiv.org/pdf/2405.08032v2)**

> **作者:** Peer-Olaf Siebers
>
> **备注:** This paper has been published in the Journal of Artificial Societies and Social Simulation 28 (3) 2. Please refer to the published version at [https://doi.org/10.18564/jasss.5681]
>
> **摘要:** ChatGPT, the AI-powered chatbot with a massive user base of hundreds of millions, has become a global phenomenon. However, the use of Conversational AI Systems (CAISs) like ChatGPT for research in the field of Social Simulation is still limited. Specifically, there is no evidence of its usage in Agent-Based Social Simulation (ABSS) model design. This paper takes a crucial first step toward exploring the untapped potential of this emerging technology in the context of ABSS model design. The research presented here demonstrates how CAISs can facilitate the development of innovative conceptual ABSS models in a concise timeframe and with minimal required upfront case-based knowledge. By employing advanced prompt engineering techniques and adhering to the Engineering ABSS framework, we have constructed a comprehensive prompt script that enables the design of conceptual ABSS models with or by the CAIS. A proof-of-concept application of the prompt script, used to generate the conceptual ABSS model for a case study on the impact of adaptive architecture in a museum environment, illustrates the practicality of the approach. Despite occasional inaccuracies and conversational divergence, the CAIS proved to be a valuable companion for ABSS modellers.
>
---
#### [replaced 031] From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22203v2](http://arxiv.org/pdf/2505.22203v2)**

> **作者:** Yuzhen Huang; Weihao Zeng; Xingshan Zeng; Qi Zhu; Junxian He
>
> **摘要:** Trustworthy verifiers are essential for the success of reinforcement learning with verifiable reward (RLVR), which is the core methodology behind various large reasoning models such as DeepSeek-R1. In complex domains like mathematical reasoning, rule-based verifiers have been widely adopted in previous works to train strong reasoning models. However, the reliability of these verifiers and their impact on the RL training process remain poorly understood. In this work, we take mathematical reasoning as a case study and conduct a comprehensive analysis of various verifiers in both static evaluation and RL training scenarios. First, we find that current open-source rule-based verifiers often fail to recognize equivalent answers presented in different formats across multiple commonly used mathematical datasets, resulting in non-negligible false negative rates. This limitation adversely affects RL training performance and becomes more pronounced as the policy model gets stronger. Subsequently, we investigate model-based verifiers as a potential solution to address these limitations. While the static evaluation shows that model-based verifiers achieve significantly higher verification accuracy, further analysis and RL results imply that they are highly susceptible to hacking, where they misclassify certain patterns in responses as correct, particularly after fine-tuning. This vulnerability is exploited during policy model optimization, leading to artificially inflated rewards. Our findings underscore the unique challenges inherent to both rule-based and model-based verifiers and provide insights toward developing more accurate and robust reward systems for reinforcement learning.
>
---
#### [replaced 032] FAID: Fine-Grained AI-Generated Text Detection Using Multi-Task Auxiliary and Multi-Level Contrastive Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14271v2](http://arxiv.org/pdf/2505.14271v2)**

> **作者:** Minh Ngoc Ta; Dong Cao Van; Duc-Anh Hoang; Minh Le-Anh; Truong Nguyen; My Anh Tran Nguyen; Yuxia Wang; Preslav Nakov; Sang Dinh
>
> **摘要:** The growing collaboration between humans and AI models in generative tasks has introduced new challenges in distinguishing between human-written, LLM-generated, and human--LLM collaborative texts. In this work, we collect a multilingual, multi-domain, multi-generator dataset FAIDSet. We further introduce a fine-grained detection framework FAID to classify text into these three categories, and also to identify the underlying LLM family of the generator. Unlike existing binary classifiers, FAID is built to capture both authorship and model-specific characteristics. Our method combines multi-level contrastive learning with multi-task auxiliary classification to learn subtle stylistic cues. By modeling LLM families as distinct stylistic entities, we incorporate an adaptation to address distributional shifts without retraining for unseen data. Our experimental results demonstrate that FAID outperforms several baselines, particularly enhancing the generalization accuracy on unseen domains and new LLMs, thus offering a potential solution for improving transparency and accountability in AI-assisted writing.
>
---
#### [replaced 033] DynaGuard: A Dynamic Guardian Model With User-Defined Policies
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02563v3](http://arxiv.org/pdf/2509.02563v3)**

> **作者:** Monte Hoover; Vatsal Baherwani; Neel Jain; Khalid Saifullah; Joseph Vincent; Chirag Jain; Melissa Kazemi Rad; C. Bayan Bruss; Ashwinee Panda; Tom Goldstein
>
> **备注:** 22 Pages
>
> **摘要:** Guardian models play a crucial role in ensuring the safety and ethical behavior of user-facing AI applications by enforcing guardrails and detecting harmful content. While standard guardian models are limited to predefined, static harm categories, we introduce DynaGuard, a suite of dynamic guardian models offering novel flexibility by evaluating text based on user-defined policies, and DynaBench, a dataset for training and evaluating dynamic guardian models. Our models provide both rapid detection of policy violations and a chain-of-thought reasoning option that articulate and justify model outputs. Critically, DynaGuard not only surpasses static models in detection accuracy on traditional safety categories, but is competitive with frontier reasoning models on free-form policy violations, all in a fraction of the time. This makes DynaGuard an critical tool for language model guardrails.
>
---
#### [replaced 034] Context Biasing for Pronunciations-Orthography Mismatch in Automatic Speech Recognition
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.18703v2](http://arxiv.org/pdf/2506.18703v2)**

> **作者:** Christian Huber; Alexander Waibel
>
> **摘要:** Neural sequence-to-sequence systems deliver state-of-the-art performance for automatic speech recognition. When using appropriate modeling units, e.g., byte-pair encoded characters, these systems are in principal open vocabulary systems. In practice, however, they often fail to recognize words not seen during training, e.g., named entities, acronyms, or domain-specific special words. To address this problem, many context biasing methods have been proposed; however, for words with a pronunciation-orthography mismatch, these methods may still struggle. We propose a method which allows corrections of substitution errors to improve the recognition accuracy of such challenging words. Users can add corrections on the fly during inference. We show that with this method we get a relative improvement in biased word error rate of up to 8%, while maintaining a competitive overall word error rate.
>
---
#### [replaced 035] AgenticIE: An Adaptive Agent for Information Extraction from Complex Regulatory Documents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11773v2](http://arxiv.org/pdf/2509.11773v2)**

> **作者:** Gaye Colakoglu; Gürkan Solmaz; Jonathan Fürst
>
> **摘要:** Declaration of Performance (DoP) documents, mandated by EU regulation, certify the performance of construction products. There are two challenges to make DoPs machine and human accessible through automated key-value pair extraction (KVP) and question answering (QA): (1) While some of their content is standardized, DoPs vary widely in layout, schema, and format; (2) Both users and documents are multilingual. Existing static or LLM-only Information Extraction (IE) pipelines fail to adapt to this structural document and user diversity. Our domain-specific, agentic system addresses these challenges through a planner-executor-responder architecture. The system infers user intent, detects document language and modality, and orchestrates tools dynamically for robust, traceable reasoning while avoiding tool misuse or execution loops. Our agent outperforms baselines (ROUGE: 0.783 vs. 0.703/0.608) with better cross-lingual stability (17-point vs. 21-26-point variation).
>
---
#### [replaced 036] Language Models Surface the Unwritten Code of Science and Society
- **分类: cs.CY; cs.CL; cs.DL**

- **链接: [http://arxiv.org/pdf/2505.18942v3](http://arxiv.org/pdf/2505.18942v3)**

> **作者:** Honglin Bao; Siyang Wu; Jiwoong Choi; Yingrong Mao; James A. Evans
>
> **摘要:** This paper calls on the research community not only to investigate how human biases are inherited by large language models (LLMs) but also to explore how these biases in LLMs can be leveraged to make society's "unwritten code" - such as implicit stereotypes and heuristics - visible and accessible for critique. We introduce a conceptual framework through a case study in science: uncovering hidden rules in peer review - the factors that reviewers care about but rarely state explicitly due to normative scientific expectations. The idea of the framework is to push LLMs to speak out their heuristics through generating self-consistent hypotheses - why one paper appeared stronger in reviewer scoring - among paired papers submitted to 45 academic conferences, while iteratively searching deeper hypotheses from remaining pairs where existing hypotheses cannot explain. We observed that LLMs' normative priors about the internal characteristics of good science extracted from their self-talk, e.g., theoretical rigor, were systematically updated toward posteriors that emphasize storytelling about external connections, such as how the work is positioned and connected within and across literatures. Human reviewers tend to explicitly reward aspects that moderately align with LLMs' normative priors (correlation = 0.49) but avoid articulating contextualization and storytelling posteriors in their review comments (correlation = -0.14), despite giving implicit reward to them with positive scores. These patterns are robust across different models and out-of-sample judgments. We discuss the broad applicability of our proposed framework, leveraging LLMs as diagnostic tools to amplify and surface the tacit codes underlying human society, enabling public discussion of revealed values and more precisely targeted responsible AI.
>
---
#### [replaced 037] Learning to vary: Teaching LMs to reproduce human linguistic variability in next-word prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17794v2](http://arxiv.org/pdf/2509.17794v2)**

> **作者:** Tobias Groot; Salo Lacunes; Evgenia Ilia
>
> **备注:** EMNLP UncertaiNLP Workshop 2025
>
> **摘要:** Natural language generation (NLG) tasks are often subject to inherent variability; e.g. predicting the next word given a context has multiple valid responses, evident when asking multiple humans to complete the task. While having language models (LMs) that are aligned pluralistically, so that they are able to reproduce well the inherent diversity in perspectives of an entire population of interest is clearly beneficial, Ilia and Aziz (2024) show that LMs do not reproduce this type of linguistic variability well. They speculate this inability might stem from the lack of consistent training of LMs with data reflecting this type of inherent variability. As such, we investigate whether training LMs on multiple plausible word continuations per context can improve their ability to reproduce human linguistic variability for next-word prediction. We employ fine-tuning techniques for pre-trained and instruction-tuned models; and demonstrate their potential when fine-tuning GPT-2 and Mistral-7B-IT, using Provo Corpus. Our evaluation, which measures divergence among empirically estimated human and model next-word distributions across contexts before and after fine-tuning, shows that our multi-label fine-tuning improves the LMs' ability to reproduce linguistic variability; both for contexts that admit higher and lower variability.
>
---
#### [replaced 038] What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.01719v2](http://arxiv.org/pdf/2510.01719v2)**

> **作者:** Jiwan Chung; Neel Joshi; Pratyusha Sharma; Youngjae Yu; Vibhav Vineet
>
> **摘要:** Multimodal reasoning models have recently shown promise on challenging domains such as olympiad-level geometry, yet their evaluation remains dominated by aggregate accuracy, a single score that obscures where and how models are improving. We introduce MathLens, a benchmark designed to disentangle the subskills of multimodal reasoning while preserving the complexity of textbook-style geometry problems. The benchmark separates performance into three components: Perception: extracting information from raw inputs, Reasoning: operating on available information, and Integration: selecting relevant perceptual evidence and applying it within reasoning. To support each test, we provide annotations: visual diagrams, textual descriptions to evaluate reasoning in isolation, controlled questions that require both modalities, and probes for fine-grained perceptual skills, all derived from symbolic specifications of the problems to ensure consistency and robustness. Our analysis reveals that different training approaches have uneven effects: First, reinforcement learning chiefly strengthens perception, especially when supported by textual supervision, while textual SFT indirectly improves perception through reflective reasoning. Second, reasoning improves only in tandem with perception. Third, integration remains the weakest capacity, with residual errors concentrated there once other skills advance. Finally, robustness diverges: RL improves consistency under diagram variation, whereas multimodal SFT reduces it through overfitting. We will release all data and experimental logs.
>
---
#### [replaced 039] CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02298v2](http://arxiv.org/pdf/2508.02298v2)**

> **作者:** Guofu Xie; Yunsheng Shi; Hongtao Tian; Ting Yao; Xiao Zhang
>
> **备注:** Work in progress
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
>
---
#### [replaced 040] SAE-FiRE: Enhancing Earnings Surprise Predictions Through Sparse Autoencoder Feature Selection
- **分类: q-fin.CP; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14420v2](http://arxiv.org/pdf/2505.14420v2)**

> **作者:** Huopu Zhang; Yanguang Liu; Miao Zhang; Zirui He; Mengnan Du
>
> **摘要:** Predicting earnings surprises from financial documents, such as earnings conference calls, regulatory filings, and financial news, has become increasingly important in financial economics. However, these financial documents present significant analytical challenges, typically containing over 5,000 words with substantial redundancy and industry-specific terminology that creates obstacles for language models. In this work, we propose the SAE-FiRE (Sparse Autoencoder for Financial Representation Enhancement) framework to address these limitations by extracting key information while eliminating redundancy. SAE-FiRE employs Sparse Autoencoders (SAEs) to decompose dense neural representations from large language models into interpretable sparse components, then applies statistical feature selection methods, including ANOVA F-tests and tree-based importance scoring, to identify the top-k most discriminative dimensions for classification. By systematically filtering out noise that might otherwise lead to overfitting, we enable more robust and generalizable predictions. Experimental results across three financial datasets demonstrate that SAE-FiRE significantly outperforms baseline approaches.
>
---
#### [replaced 041] QAPyramid: Fine-grained Evaluation of Content Selection for Text Summarization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.07096v2](http://arxiv.org/pdf/2412.07096v2)**

> **作者:** Shiyue Zhang; David Wan; Arie Cattan; Ayal Klein; Ido Dagan; Mohit Bansal
>
> **备注:** Accepted to COLM 2025. The first two authors contributed equally. Code: https://github.com/ZhangShiyue/QAPyramid
>
> **摘要:** How to properly conduct human evaluations for text summarization is a longstanding challenge. The Pyramid human evaluation protocol, which assesses content selection by breaking the reference summary into subunits and verifying their presence in the system summary, has been widely adopted. However, it suffers from a lack of systematicity in the definition and granularity of the sub-units. We address these problems by proposing QAPyramid, which decomposes each reference summary into finer-grained question-answer (QA) pairs according to the QA-SRL framework. We collect QA-SRL annotations for reference summaries from CNN/DM and evaluate 10 summarization systems, resulting in 8.9K QA-level annotations. We show that, compared to Pyramid, QAPyramid provides more systematic and fine-grained content selection evaluation while maintaining high inter-annotator agreement without needing expert annotations. Furthermore, we propose metrics that automate the evaluation pipeline and achieve higher correlations with QAPyramid than other widely adopted metrics.
>
---
#### [replaced 042] When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05690v2](http://arxiv.org/pdf/2506.05690v2)**

> **作者:** Zhishang Xiang; Chuanjie Wu; Qinggang Zhang; Shengyuan Chen; Zijin Hong; Xiao Huang; Jinsong Su
>
> **备注:** All resources and analyses are collected at https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
>
> **摘要:** Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate reasoning.Despite its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models onboth hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, coveringfact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph constructionand knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application. All related resources and analyses are collected for the community at https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.
>
---
#### [replaced 043] Evaluating the Effect of Retrieval Augmentation on Social Biases
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17611v2](http://arxiv.org/pdf/2502.17611v2)**

> **作者:** Tianhui Zhang; Yi Zhou; Danushka Bollegala
>
> **备注:** 21 pages
>
> **摘要:** Retrieval Augmented Generation (RAG) has gained popularity as a method for conveniently incorporating novel facts that were not seen during the pre-training stage in Large Language Model (LLM)-based Natural Language Generation (NLG) systems. However, LLMs are known to encode significant levels of unfair social biases. The modulation of these biases by RAG in NLG systems is not well understood. In this paper, we systematically study the relationship between the different components of a RAG system and the social biases presented in the text generated across three languages (i.e. English, Japanese and Chinese) and four social bias types (i.e. gender, race, age and religion). Specifically, using the Bias Question Answering (BBQ) benchmark datasets, we evaluate the social biases in RAG responses from document collections with varying levels of stereotypical biases, employing multiple LLMs used as generators. We find that the biases in document collections are often amplified in the generated responses, even when the generating LLM exhibits a low-level of bias. Our findings raise concerns about the use of RAG as a technique for injecting novel facts into NLG systems and call for careful evaluation of potential social biases in RAG applications before their real-world deployment.
>
---
#### [replaced 044] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17978v2](http://arxiv.org/pdf/2505.17978v2)**

> **作者:** Rui Cao; Zifeng Ding; Zhijiang Guo; Michael Schlichtkrull; Andreas Vlachos
>
> **备注:** accepted at NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Textual claims are often accompanied by images to enhance their credibility and spread on social media, but this also raises concerns about the spread of misinformation. Existing datasets for automated verification of image-text claims remain limited, as they often consist of synthetic claims and lack evidence annotations to capture the reasoning behind the verdict. In this work, we introduce AVerImaTeC, a dataset consisting of 1,297 real-world image-text claims. Each claim is annotated with question-answer (QA) pairs containing evidence from the web, reflecting a decomposed reasoning regarding the verdict. We mitigate common challenges in fact-checking datasets such as contextual dependence, temporal leakage, and evidence insufficiency, via claim normalization, temporally constrained evidence annotation, and a two-stage sufficiency check. We assess the consistency of the annotation in AVerImaTeC via inter-annotator studies, achieving a $\kappa=0.742$ on verdicts and $74.7\%$ consistency on QA pairs. We also propose a novel evaluation method for evidence retrieval and conduct extensive experiments to establish baselines for verifying image-text claims using open-web evidence.
>
---
#### [replaced 045] Contrastive Learning Using Graph Embeddings for Domain Adaptation of Language Models in the Process Industry
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.04631v2](http://arxiv.org/pdf/2510.04631v2)**

> **作者:** Anastasia Zhukova; Jonas Lührs; Christian E. Lobmüller; Bela Gipp
>
> **备注:** accepted to EMNLP 2025 (industry track)
>
> **摘要:** Recent trends in NLP utilize knowledge graphs (KGs) to enhance pretrained language models by incorporating additional knowledge from the graph structures to learn domain-specific terminology or relationships between documents that might otherwise be overlooked. This paper explores how SciNCL, a graph-aware neighborhood contrastive learning methodology originally designed for scientific publications, can be applied to the process industry domain, where text logs contain crucial information about daily operations and are often structured as sparse KGs. Our experiments demonstrate that language models fine-tuned with triplets derived from graph embeddings (GE) outperform a state-of-the-art mE5-large text encoder by 9.8-14.3% (5.45-7.96p) on the proprietary process industry text embedding benchmark (PITEB) while having 3 times fewer parameters.
>
---
#### [replaced 046] Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.09946v3](http://arxiv.org/pdf/2408.09946v3)**

> **作者:** Byungjun Kim; Dayeon Seo; Minju Kim; Bugeun Kim
>
> **备注:** Published in IEEE Access
>
> **摘要:** Recent studies have investigated whether large language models (LLMs) can support obscured communication, which is characterized by core aspects such as inferring subtext and evading suspicions. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two limitations with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these limitations, we propose a microscopic and systematic approach to the investigation. Specifically, we introduce six fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
>
---
#### [replaced 047] Trajectory Prediction Meets Large Language Models: A Survey
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03408v2](http://arxiv.org/pdf/2506.03408v2)**

> **作者:** Yi Xu; Ruining Yang; Yitian Zhang; Jianglin Lu; Mingyuan Zhang; Yizhou Wang; Lili Su; Yun Fu
>
> **备注:** 16 pages, GitHub: https://github.com/colorfulfuture/Awesome-Trajectory-Motion-Prediction-Papers
>
> **摘要:** Recent advances in large language models (LLMs) have sparked growing interest in integrating language-driven techniques into trajectory prediction. By leveraging their semantic and reasoning capabilities, LLMs are reshaping how autonomous systems perceive, model, and predict trajectories. This survey provides a comprehensive overview of this emerging field, categorizing recent work into five directions: (1) Trajectory prediction via language modeling paradigms, (2) Direct trajectory prediction with pretrained language models, (3) Language-guided scene understanding for trajectory prediction, (4) Language-driven data generation for trajectory prediction, (5) Language-based reasoning and interpretability for trajectory prediction. For each, we analyze representative methods, highlight core design choices, and identify open challenges. This survey bridges natural language processing and trajectory prediction, offering a unified perspective on how language can enrich trajectory prediction.
>
---
#### [replaced 048] AgriGPT-VL: Agricultural Vision-Language Understanding Suite
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04002v2](http://arxiv.org/pdf/2510.04002v2)**

> **作者:** Bo Yang; Yunkui Chen; Lanfei Feng; Yu Zhang; Xiao Xu; Jianyu Zhang; Nueraili Aierken; Runhe Huang; Hongjian Lin; Yibin Ying; Shijian Li
>
> **摘要:** Despite rapid advances in multimodal large language models, agricultural applications remain constrained by the scarcity of domain-tailored models, curated vision-language corpora, and rigorous evaluation. To address these challenges, we present the AgriGPT-VL Suite, a unified multimodal framework for agriculture. Our contributions are threefold. First, we introduce Agri-3M-VL, the largest vision-language corpus for agriculture to our knowledge, curated by a scalable multi-agent data generator; it comprises 1M image-caption pairs, 2M image-grounded VQA pairs, 50K expert-level VQA instances, and 15K GRPO reinforcement learning samples. Second, we develop AgriGPT-VL, an agriculture-specialized vision-language model trained via a progressive curriculum of textual grounding, multimodal shallow/deep alignment, and GRPO refinement. This method achieves strong multimodal reasoning while preserving text-only capability. Third, we establish AgriBench-VL-4K, a compact yet challenging evaluation suite with open-ended and image-grounded questions, paired with multi-metric evaluation and an LLM-as-a-judge framework. Experiments show that AgriGPT-VL outperforms leading general-purpose VLMs on AgriBench-VL-4K, achieving higher pairwise win rates in the LLM-as-a-judge evaluation. Meanwhile, it remains competitive on the text-only AgriBench-13K with no noticeable degradation of language ability. Ablation studies further confirm consistent gains from our alignment and GRPO refinement stages. We will open source all of the resources to support reproducible research and deployment in low-resource agricultural settings.
>
---
#### [replaced 049] MedHal: An Evaluation Dataset for Medical Hallucination Detection
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.08596v2](http://arxiv.org/pdf/2504.08596v2)**

> **作者:** Gaya Mehenni; Fabrice Lamarche; Odette Rios-Ibacache; John Kildea; Amal Zouaq
>
> **摘要:** We present MedHal, a novel large-scale dataset specifically designed to evaluate if models can detect hallucinations in medical texts. Current hallucination detection methods face significant limitations when applied to specialized domains like medicine, where they can have disastrous consequences. Existing medical datasets are either too small, containing only a few hundred samples, or focus on a single task like Question Answering or Natural Language Inference. MedHal addresses these gaps by: (1) incorporating diverse medical text sources and tasks; (2) providing a substantial volume of annotated samples suitable for training medical hallucination detection models; and (3) including explanations for factual inconsistencies to guide model learning. We demonstrate MedHal's utility by training and evaluating a baseline medical hallucination detection model, showing improvements over general-purpose hallucination detection approaches. This resource enables more efficient evaluation of medical text generation systems while reducing reliance on costly expert review, potentially accelerating the development of medical AI research.
>
---
#### [replaced 050] RooseBERT: A New Deal For Political Language Modelling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.03250v2](http://arxiv.org/pdf/2508.03250v2)**

> **作者:** Deborah Dore; Elena Cabrio; Serena Villata
>
> **摘要:** The increasing amount of political debates and politics-related discussions calls for the definition of novel computational methods to automatically analyse such content with the final goal of lightening up political deliberation to citizens. However, the specificity of the political language and the argumentative form of these debates (employing hidden communication strategies and leveraging implicit arguments) make this task very challenging, even for current general-purpose pre-trained Language Models. To address this issue, we introduce a novel pre-trained Language Model for political discourse language called RooseBERT. Pre-training a language model on a specialised domain presents different technical and linguistic challenges, requiring extensive computational resources and large-scale data. RooseBERT has been trained on large political debate and speech corpora (8K debates, each composed of several sub-debates on different topics) in English. To evaluate its performances, we fine-tuned it on four downstream tasks related to political debate analysis, i.e., stance detection, sentiment analysis, argument component detection and classification, and argument relation prediction and classification. Our results demonstrate significant improvements over general-purpose Language Models on these four tasks, highlighting how domain-specific pre-training enhances performance in political debate analysis. We release RooseBERT for the research community.
>
---
#### [replaced 051] Assessing Algorithmic Bias in Language-Based Depression Detection: A Comparison of DNN and LLM Approaches
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.25795v2](http://arxiv.org/pdf/2509.25795v2)**

> **作者:** Obed Junias; Prajakta Kini; Theodora Chaspari
>
> **备注:** 7 pages, 1 figure. This paper has been accepted to the IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 2025), Georgia Institute of Technology, Atlanta, Georgia, October 26-29, 2025
>
> **摘要:** This paper investigates algorithmic bias in language-based models for automated depression detection, focusing on socio-demographic disparities related to gender and race/ethnicity. Models trained using deep neural networks (DNN) based embeddings are compared to few-shot learning approaches with large language models (LLMs), evaluating both performance and fairness on clinical interview transcripts from the Distress Analysis Interview Corpus/Wizard-of-Oz (DAIC-WOZ). To mitigate bias, fairness-aware loss functions are applied to DNN-based models, while in-context learning with varied prompt framing and shot counts is explored for LLMs. Results indicate that LLMs outperform DNN-based models in depression classification, particularly for underrepresented groups such as Hispanic participants. LLMs also exhibit reduced gender bias compared to DNN-based embeddings, though racial disparities persist. Among fairness-aware techniques for mitigating bias in DNN-based embeddings, the worst-group loss, which is designed to minimize loss for the worst-performing demographic group, achieves a better balance between performance and fairness. In contrast, the fairness-regularized loss minimizes loss across all groups but performs less effectively. In LLMs, guided prompting with ethical framing helps mitigate gender bias in the 1-shot setting. However, increasing the number of shots does not lead to further reductions in disparities. For race/ethnicity, neither prompting strategy nor increasing $N$ in $N$-shot learning effectively reduces disparities.
>
---
#### [replaced 052] Entropy-Gated Branching for Efficient Test-Time Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21961v3](http://arxiv.org/pdf/2503.21961v3)**

> **作者:** Xianzhi Li; Ethan Callanan; Abdellah Ghassel; Xiaodan Zhu
>
> **摘要:** Test-time compute methods can significantly improve the reasoning capabilities and problem-solving accuracy of large language models (LLMs). However, these approaches require substantially more computational resources, with most compute wasted on exploring low-diversity branches where the model already exhibits high confidence. We observe that a small subset of uncertain reasoning steps has a disproportionately large impact on final prediction accuracy, and branching at these critical junctures tends to yield more diverse and higher-quality candidate reasoning steps. We propose Entropy-Gated Branching (EGB), which branches only at high-uncertainty steps and prunes expansions with a lightweight verifier. On mathematical and financial reasoning benchmarks, EGB improves accuracy by 22.6% over standard inference while operating 31%-75% faster across math benchmarks than test-time beam search with higher performance. Our results show that dynamic resource allocation during inference can substantially improve both efficiency and effectiveness, offering a more scalable pathway to enhanced LLM reasoning capabilities.
>
---
#### [replaced 053] The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs?
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10020v3](http://arxiv.org/pdf/2504.10020v3)**

> **作者:** Hao Yin; Guangzong Si; Zilei Wang
>
> **摘要:** Contrastive decoding strategies are widely used to reduce object hallucinations in multimodal large language models (MLLMs). These methods work by constructing contrastive samples to induce hallucinations and then suppressing them in the output distribution. However, this paper demonstrates that such approaches fail to effectively mitigate the hallucination problem. The performance improvements observed on POPE Benchmark are largely driven by two misleading factors: (1) crude, unidirectional adjustments to the model's output distribution and (2) the adaptive plausibility constraint, which reduces the sampling strategy to greedy search. To further illustrate these issues, we introduce a series of spurious improvement methods and evaluate their performance against contrastive decoding techniques. Experimental results reveal that the observed performance gains in contrastive decoding are entirely unrelated to its intended goal of mitigating hallucinations. Our findings challenge common assumptions about the effectiveness of contrastive decoding strategies and pave the way for developing genuinely effective solutions to hallucinations in MLLMs.
>
---
#### [replaced 054] ChartCards: A Chart-Metadata Generation Framework for Multi-Task Chart Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15046v3](http://arxiv.org/pdf/2505.15046v3)**

> **作者:** Yifan Wu; Lutao Yan; Leixian Shen; Yinan Mei; Jiannan Wang; Yuyu Luo
>
> **备注:** Need to be revised
>
> **摘要:** The emergence of Multi-modal Large Language Models (MLLMs) presents new opportunities for chart understanding. However, due to the fine-grained nature of these tasks, applying MLLMs typically requires large, high-quality datasets for task-specific fine-tuning, leading to high data collection and training costs. To address this, we propose ChartCards, a unified chart-metadata generation framework for multi-task chart understanding. ChartCards systematically synthesizes various chart information, including data tables, visualization code, visual elements, and multi-dimensional semantic captions. By structuring this information into organized metadata, ChartCards enables a single chart to support multiple downstream tasks, such as text-to-chart retrieval, chart summarization, chart-to-table conversion, chart description, and chart question answering. Using ChartCards, we further construct MetaChart, a large-scale high-quality dataset containing 10,862 data tables, 85K charts, and 170 K high-quality chart captions. We validate the dataset through qualitative crowdsourcing evaluations and quantitative fine-tuning experiments across various chart understanding tasks. Fine-tuning six different models on MetaChart resulted in an average performance improvement of 5% across all tasks. The most notable improvements are seen in text-to-chart retrieval and chart-to-table tasks, with Long-CLIP and Llama 3.2-11B achieving improvements of 17% and 28%, respectively.
>
---
#### [replaced 055] Measuring LLM Novelty As The Frontier Of Original And High-Quality Output
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09389v2](http://arxiv.org/pdf/2504.09389v2)**

> **作者:** Vishakh Padmakumar; Chen Yueh-Han; Jane Pan; Valerie Chen; He He
>
> **备注:** Updated results with higher coverage of open-data models and better quality judgments
>
> **摘要:** As large language models (LLMs) are increasingly used for ideation and scientific discovery, it is important to evaluate their ability to generate novel output. Prior work evaluates novelty as originality with respect to model training data, but original outputs may be of low quality. In contrast, non-expert judges more reliably score quality but may favor memorized outputs, limiting the reliability of human preference as a metric. We introduce a new novelty metric for LLM generations that balances originality and quality -- the harmonic mean of the fraction of \ngrams unseen during training and a task-specific quality score. Using this framework, we identify trends that affect the novelty of generations from three families of open-data models (OLMo, OLMo-2, and Pythia) on three creative tasks: story completion, poetry writing, and creative tool use. We find that model-generated text from some base LLMs is less novel than human-written text from the internet. However, increasing model scale and post-training reliably improves novelty due to improvements in output quality. We also find that improving the base model at the same scale (\eg OLMo 7B to OLMo-2 7B) leads to higher novelty due to higher originality. Finally, we observe that inference-time methods, such as prompting and providing novel in-context examples, have a much smaller effect on novelty, often increasing originality at the expense of quality. This highlights the need for further research into more effective elicitation strategies as we use models for creative applications.
>
---
#### [replaced 056] Large Language Models Achieve Gold Medal Performance at the International Olympiad on Astronomy & Astrophysics (IOAA)
- **分类: astro-ph.IM; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05016v2](http://arxiv.org/pdf/2510.05016v2)**

> **作者:** Lucas Carrit Delgado Pinheiro; Ziru Chen; Bruno Caixeta Piazza; Ness Shroff; Yingbin Liang; Yuan-Sen Ting; Huan Sun
>
> **备注:** 18 pages, 6 figures, to be submitted, comments are welcome. Reproducibility details can be found at: https://github.com/OSU-NLP-Group/LLM-IOAA
>
> **摘要:** While task-specific demonstrations show early success in applying large language models (LLMs) to automate some astronomical research tasks, they only provide incomplete views of all necessary capabilities in solving astronomy problems, calling for more thorough understanding of LLMs' strengths and limitations. So far, existing benchmarks and evaluations focus on simple question-answering that primarily tests astronomical knowledge and fails to evaluate the complex reasoning required for real-world research in the discipline. Here, we address this gap by systematically benchmarking five state-of-the-art LLMs on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, which are designed to examine deep conceptual understanding, multi-step derivations, and multimodal analysis. With average scores of 85.6% and 84.2%, Gemini 2.5 Pro and GPT-5 (the two top-performing models) not only achieve gold medal level performance but also rank in the top two among ~200-300 participants in all four IOAA theory exams evaluated (2022-2025). In comparison, results on the data analysis exams show more divergence. GPT-5 still excels in the exams with an 88.5% average score, ranking top 10 among the participants in the four most recent IOAAs, while other models' performances drop to 48-76%. Furthermore, our in-depth error analysis underscores conceptual reasoning, geometric reasoning, and spatial visualization (52-79% accuracy) as consistent weaknesses among all LLMs. Hence, although LLMs approach peak human performance in theory exams, critical gaps must be addressed before they can serve as autonomous research agents in astronomy.
>
---
#### [replaced 057] Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training
- **分类: cs.AI; cs.CL; math.SP**

- **链接: [http://arxiv.org/pdf/2410.15460v5](http://arxiv.org/pdf/2410.15460v5)**

> **作者:** Shahrad Mohammadzadeh; Juan David Guerra; Marco Bonizzato; Reihaneh Rabbany; Golnoosh Farnadi
>
> **备注:** Accepted to ACL 2025, accepted to Safe Generative AI Workshop @ NeurIPS 2024. Camera-ready version for ACL 2025 (to appear). Submitted July 2025
>
> **摘要:** As large language models (LLMs) become increasingly prevalent, concerns about their reliability, particularly due to hallucinations - factually inaccurate or irrelevant outputs - have grown. Our research investigates the relationship between the uncertainty in training dynamics and the emergence of hallucinations. Using models from the Pythia suite and several hallucination detection metrics, we analyze hallucination trends and identify significant variance during training. To address this, we propose Sensitivity Dropout (SenD), a novel training protocol designed to reduce hallucination variance during training by deterministically dropping embedding indices with significant variability. In addition, we develop an unsupervised hallucination detection metric, Efficient EigenScore (EES), which approximates the traditional EigenScore in 2x speed. This metric is integrated into our training protocol, allowing SenD to be both computationally scalable and effective at reducing hallucination variance. SenD improves test-time reliability of Pythia and Meta's Llama models by up to 17% and enhances factual accuracy in Wikipedia, Medical, Legal, and Coding domains without affecting downstream task performance.
>
---
#### [replaced 058] Text Clustering as Classification with LLMs
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2410.00927v3](http://arxiv.org/pdf/2410.00927v3)**

> **作者:** Chen Huang; Guoxiu He
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Text clustering serves as a fundamental technique for organizing and interpreting unstructured textual data, particularly in contexts where manual annotation is prohibitively costly. With the rapid advancement of Large Language Models (LLMs) and their demonstrated effectiveness across a broad spectrum of NLP tasks, an emerging body of research has begun to explore their potential in the domain of text clustering. However, existing LLM-based approaches still rely on fine-tuned embedding models and sophisticated similarity metrics, rendering them computationally intensive and necessitating domain-specific adaptation. To address these limitations, we propose a novel framework that reframes text clustering as a classification task by harnessing the in-context learning capabilities of LLMs. Our framework eliminates the need for fine-tuning embedding models or intricate clustering algorithms. It comprises two key steps: first, the LLM is prompted to generate a set of candidate labels based on the dataset and then merges semantically similar labels; second, it assigns the most appropriate label to each text sample. By leveraging the advanced natural language understanding and generalization capabilities of LLMs, the proposed approach enables effective clustering with minimal human intervention. Experimental results on diverse datasets demonstrate that our framework achieves comparable or superior performance to state-of-the-art embedding-based clustering techniques, while significantly reducing computational complexity and resource requirements. These findings underscore the transformative potential of LLMs in simplifying and enhancing text clustering tasks. We make our code available to the public for utilization at https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM. We also provide the supplementary Appendix within the repository.
>
---
#### [replaced 059] RepIt: Representing Isolated Targets to Steer Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13281v2](http://arxiv.org/pdf/2509.13281v2)**

> **作者:** Vincent Siu; Nathan W. Henry; Nicholas Crispino; Yang Liu; Dawn Song; Chenguang Wang
>
> **摘要:** While activation steering in large language models (LLMs) is a growing area of research, methods can often incur broader effects than desired. This motivates isolation of purer concept vectors to enable targeted interventions and understand LLM behavior at a more granular level. We present RepIt, a simple and data-efficient framework for isolating concept-specific representations. Across five frontier LLMs, RepIt enables precise interventions: it selectively suppresses refusal on targeted concepts while preserving refusal elsewhere, producing models that answer WMD-related questions while still scoring as safe on standard benchmarks. We further show that the corrective signal localizes to just 100-200 neurons and that robust target representations can be extracted from as few as a dozen examples on a single A6000. This efficiency raises a dual concern: manipulations can be performed with modest compute and data to extend to underrepresented data-scarce topics while evading existing benchmarks. By disentangling refusal vectors with RepIt, this work demonstrates that targeted interventions can counteract overgeneralization, laying the foundation for more granular control of model behavior.
>
---
#### [replaced 060] Generative transformations and patterns in LLM-native approaches for software verification and falsification
- **分类: cs.SE; cs.AI; cs.CL; cs.LG; F.3.1; D.2.4; D.2.5; I.2.7**

- **链接: [http://arxiv.org/pdf/2404.09384v3](http://arxiv.org/pdf/2404.09384v3)**

> **作者:** Víctor A. Braberman; Flavia Bonomo-Braberman; Yiannis Charalambous; Juan G. Colonna; Lucas C. Cordeiro; Rosiane de Freitas
>
> **摘要:** The emergence of prompting as the dominant paradigm for leveraging Large Language Models (LLMs) has led to a proliferation of LLM-native software, where application behavior arises from complex, stochastic data transformations. However, the engineering of such systems remains largely exploratory and ad-hoc, hampered by the absence of conceptual frameworks, ex-ante methodologies, design guidelines, and specialized benchmarks. We argue that a foundational step towards a more disciplined engineering practice is a systematic understanding of the core functional units--generative transformations--and their compositional patterns within LLM-native applications. Focusing on the rich domain of software verification and falsification, we conduct a secondary study of over 100 research proposals to address this gap. We first present a fine-grained taxonomy of generative transformations, abstracting prompt-based interactions into conceptual signatures. This taxonomy serves as a scaffolding to identify recurrent transformation relationship patterns--analogous to software design patterns--that characterize solution approaches in the literature. Our analysis not only validates the utility of the taxonomy but also surfaces strategic gaps and cross-dimensional relationships, offering a structured foundation for future research in modular and compositional LLM application design, benchmarking, and the development of reliable LLM-native systems.
>
---
#### [replaced 061] Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17523v2](http://arxiv.org/pdf/2503.17523v2)**

> **作者:** Linlu Qiu; Fei Sha; Kelsey Allen; Yoon Kim; Tal Linzen; Sjoerd van Steenkiste
>
> **摘要:** Artificial intelligence systems based on large language models (LLMs) are increasingly used as agents that interact with users and with the world. To do so successfully, LLMs need to construct internal representations of the world and form probabilistic beliefs about those representations. To provide a user with personalized recommendations, for example, the LLM needs to gradually infer the user's preferences, over the course of multiple interactions. To evaluate whether contemporary LLMs are able to do so, we use the Bayesian inference framework from probability theory, which lays out the optimal way to update an agent's beliefs as it receives new information. We first show that LLMs do not update their beliefs as expected from the Bayesian framework, and that consequently their predictions do not improve as expected as more information becomes available. To address this issue, we teach the LLMs to reason in a Bayesian manner by training them to mimic the predictions of the normative Bayesian model. We find that this approach not only significantly improves the LLM's performance on the particular recommendation task it is trained on, but also enables generalization to other tasks. This suggests that this method teaches the LLM to better approximate Bayesian reasoning. More generally, our results indicate that LLMs can effectively learn reasoning skills from examples and generalize those skills to new domains.
>
---
#### [replaced 062] GLiDRE: Generalist Lightweight model for Document-level Relation Extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00757v2](http://arxiv.org/pdf/2508.00757v2)**

> **作者:** Robin Armingaud; Romaric Besançon
>
> **备注:** Submitted to ARR October
>
> **摘要:** Relation Extraction (RE) is a fundamental task in Natural Language Processing, and its document-level variant poses significant challenges, due to complex interactions between entities across sentences. While supervised models have achieved strong results in fully resourced settings, their behavior with limited training data remains insufficiently studied. We introduce GLiDRE, a new compact model for document-level relation extraction, designed to work efficiently in both supervised and few-shot settings. Experiments in both low-resource supervised training and few-shot meta-learning benchmarks show that our approach outperforms existing methods in data-constrained scenarios, establishing a new state-of-the-art in few-shot document-level relation extraction. Our code will be publicly available.
>
---
#### [replaced 063] On Relation-Specific Neurons in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17355v2](http://arxiv.org/pdf/2502.17355v2)**

> **作者:** Yihong Liu; Runsheng Chen; Lea Hirlimann; Ahmad Dawar Hakimi; Mingyang Wang; Amir Hossein Kargaran; Sascha Rothe; François Yvon; Hinrich Schütze
>
> **备注:** EMNLP 2025
>
> **摘要:** In large language models (LLMs), certain \emph{neurons} can store distinct pieces of knowledge learned during pretraining. While factual knowledge typically appears as a combination of \emph{relations} and \emph{entities}, it remains unclear whether some neurons focus on a relation itself -- independent of any entity. We hypothesize such neurons \emph{detect} a relation in the input text and \emph{guide} generation involving such a relation. To investigate this, we study the LLama-2 family on a chosen set of relations, with a \textit{statistics}-based method. Our experiments demonstrate the existence of relation-specific neurons. We measure the effect of selectively deactivating candidate neurons specific to relation $r$ on the LLM's ability to handle (1) facts involving relation $r$ and (2) facts involving a different relation $r' \neq r$. With respect to their capacity for encoding relation information, we give evidence for the following three properties of relation-specific neurons. \textbf{(i) Neuron cumulativity.} Multiple neurons jointly contribute to processing facts involving relation $r$, with no single neuron fully encoding a fact in $r$ on its own. \textbf{(ii) Neuron versatility.} Neurons can be shared across multiple closely related as well as less related relations. In addition, some relation neurons transfer across languages. \textbf{(iii) Neuron interference.} Deactivating neurons specific to one relation can improve LLMs' factual recall performance for facts of other relations. We make our code and data publicly available at https://github.com/cisnlp/relation-specific-neurons.
>
---
#### [replaced 064] BenchAgents: Multi-Agent Systems for Structured Benchmark Creation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.22584v2](http://arxiv.org/pdf/2410.22584v2)**

> **作者:** Natasha Butt; Varun Chandrasekaran; Neel Joshi; Besmira Nushi; Vidhisha Balachandran
>
> **摘要:** Evaluation insights are limited by the availability of high-quality benchmarks. As models evolve, there is a need to create benchmarks that can measure progress on new and complex generative capabilities. However, manually creating new benchmarks is slow and expensive, restricting comprehensive evaluations for any capability. We introduce BenchAgents, a multi-agent framework that methodically leverages large language models (LLMs) to automate evaluation benchmark creation while inherently ensuring data and (evaluation) metric quality. BenchAgents decomposes the benchmark creation process into planning, generation, verification, and evaluation, each of which is ] orchestrated via LLM agents. These agents interact with each other and utilize feedback from benchmark developers to improve and flexibly control data diversity and quality. We use BenchAgents to create benchmarks to evaluate capabilities related to planning, constraint satisfaction, and causal reasoning spanning both language and vision modalities. We then use these benchmarks to study state-of-the-art models and extract new insights into common failure modes and model differences.
>
---
#### [replaced 065] LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.14252v2](http://arxiv.org/pdf/2509.14252v2)**

> **作者:** Hai Huang; Yann LeCun; Randall Balestriero
>
> **摘要:** Large Language Model (LLM) pretraining, finetuning, and evaluation rely on input-space reconstruction and generative capabilities. Yet, it has been observed in vision that embedding-space training objectives, e.g., with Joint Embedding Predictive Architectures (JEPAs), are far superior to their input-space counterpart. That mismatch in how training is achieved between language and vision opens up a natural question: {\em can language training methods learn a few tricks from the vision ones?} The lack of JEPA-style LLM is a testimony of the challenge in designing such objectives for language. In this work, we propose a first step in that direction where we develop LLM-JEPA, a JEPA based solution for LLMs applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to outperform the standard LLM training objectives by a significant margin across models, all while being robust to overfiting. Those findings are observed across numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models from the Llama3, OpenELM, Gemma2 and Olmo families. Code: https://github.com/rbalestr-lab/llm-jepa.
>
---
#### [replaced 066] GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.14221v2](http://arxiv.org/pdf/2509.14221v2)**

> **作者:** Silan Hu; Shiqi Zhang; Yimin Shi; Xiaokui Xiao
>
> **备注:** Include more experimental results and supplementary materials
>
> **摘要:** Generative Engine Marketing (GEM) is an emerging ecosystem for monetizing generative engines, such as LLM-based chatbots, by seamlessly integrating relevant advertisements into their responses. At the core of GEM lies the generation and evaluation of ad-injected responses. However, existing benchmarks are not specifically designed for this purpose, which limits future research. To address this gap, we propose GEM-Bench, the first comprehensive benchmark for ad-injected response generation in GEM. GEM-Bench includes three curated datasets covering both chatbot and search scenarios, a metric ontology that captures multiple dimensions of user satisfaction and engagement, and several baseline solutions implemented within an extensible multi-agent framework. Our preliminary results indicate that, while simple prompt-based methods achieve reasonable engagement such as click-through rate, they often reduce user satisfaction. In contrast, approaches that insert ads based on pre-generated ad-free responses help mitigate this issue but introduce additional overhead. These findings highlight the need for future research on designing more effective and efficient solutions for generating ad-injected responses in GEM. The benchmark and all related resources are publicly available at https://gem-bench.org/.
>
---
#### [replaced 067] MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14146v3](http://arxiv.org/pdf/2508.14146v3)**

> **作者:** Xian Gao; Jiacheng Ruan; Zongyun Zhang; Jingsheng Gao; Ting Liu; Yuzhuo Fu
>
> **备注:** Work in progress
>
> **摘要:** With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems.
>
---
#### [replaced 068] Teaching Small Language Models to Learn Logic through Meta-Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14313v2](http://arxiv.org/pdf/2505.14313v2)**

> **作者:** Leonardo Bertolazzi; Manuel Vargas Guzmán; Raffaella Bernardi; Maciej Malicki; Jakub Szymanik
>
> **摘要:** Large language models (LLMs) are increasingly evaluated on reasoning tasks, yet their logical abilities remain contested. To address this, we study LLMs' reasoning in a well-defined fragment of logic: syllogistic reasoning. We cast the problem as premise selection and construct controlled datasets to isolate logical competence. Beyond evaluation, an open challenge is enabling LLMs to acquire abstract inference patterns that generalize to novel structures. We propose to apply few-shot meta-learning to this domain, thereby encouraging models to extract rules across tasks rather than memorize patterns within tasks. Although meta-learning has been little explored in the context of logic learnability, our experiments show that it is effective: small models (1.5B-7B) fine-tuned with meta-learning demonstrate strong gains in generalization, with especially pronounced benefits in low-data regimes. These meta-learned models outperform GPT-4o and o3-mini on our syllogistic reasoning task.
>
---
#### [replaced 069] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20291v2](http://arxiv.org/pdf/2505.20291v2)**

> **作者:** Di Wu; Yixin Wan; Kai-Wei Chang
>
> **摘要:** Text-to-image retrieval (T2I retrieval) remains challenging because cross-modal embeddings often behave as bags of concepts and underrepresent structured visual relationships such as pose and viewpoint. We propose Visualize-then-Retrieve (VisRet), a new paradigm for T2I retrieval that mitigates this limitation of cross-modal similarity alignment. VisRet first projects textual queries into the image modality via T2I generation. Then, it performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Across four benchmarks (Visual-RAG, INQUIRE-Rerank, Microsoft COCO, and our new Visual-RAG-ME featuring multi-entity comparisons), VisRet substantially outperforms cross-modal similarity matching and baselines that recast T2I retrieval as text-to-text similarity matching, improving nDCG@30 by 0.125 on average with CLIP as the retriever and by 0.121 with E5-V. For downstream question answering, VisRet increases accuracy on Visual-RAG and Visual-RAG-ME by 3.8% and 15.7% in top-1 retrieval, and by 3.9% and 11.1% in top-10 retrieval. Ablation studies show compatibility with different T2I instruction LLMs, T2I generation models, and downstream LLMs. VisRet provides a practical and principled path that energizes further advances in vision-language retrieval. Our code and the Visual-RAG-ME benchmark will be publicly released.
>
---
#### [replaced 070] COLE: a Comprehensive Benchmark for French Language Understanding Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05046v2](http://arxiv.org/pdf/2510.05046v2)**

> **作者:** David Beauchemin; Yan Tremblay; Mohamed Amine Youssef; Richard Khoury
>
> **备注:** Submitted to ACL Rolling Review of October
>
> **摘要:** To address the need for a more comprehensive evaluation of French Natural Language Understanding (NLU), we introduce COLE, a new benchmark composed of 23 diverse task covering a broad range of NLU capabilities, including sentiment analysis, paraphrase detection, grammatical judgment, and reasoning, with a particular focus on linguistic phenomena relevant to the French language. We benchmark 94 large language models (LLM), providing an extensive analysis of the current state of French NLU. Our results highlight a significant performance gap between closed- and open-weights models and identify key challenging frontiers for current LLMs, such as zero-shot extractive question-answering (QA), fine-grained word sense disambiguation, and understanding of regional language variations. We release COLE as a public resource to foster further progress in French language modelling.
>
---
#### [replaced 071] Epistemic Diversity and Knowledge Collapse in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04226v2](http://arxiv.org/pdf/2510.04226v2)**

> **作者:** Dustin Wright; Sarah Masud; Jared Moore; Srishti Yadav; Maria Antoniak; Chan Young Park; Isabelle Augenstein
>
> **备注:** 16 pages; 8 figures, 4 tables v2 changelog: Fixed the modeling for table 3, random effect is the model version
>
> **摘要:** Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation
>
---
#### [replaced 072] Adaptive Margin RLHF via Preference over Preferences
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.22851v2](http://arxiv.org/pdf/2509.22851v2)**

> **作者:** Yaswanth Chittepu; Prasann Singhal; Greg Durrett; Scott Niekum
>
> **摘要:** Margin-based optimization is fundamental to improving generalization and robustness in classification tasks. In the context of reward model learning from preferences within Reinforcement Learning from Human Feedback (RLHF), existing methods typically rely on no margins, fixed margins, or margins that are simplistic functions of preference ratings. However, such formulations often fail to account for the varying strengths of different preferences, for example some preferences are associated with larger margins between responses, or they rely on noisy margin information derived from ratings. We argue that modeling the strength of preferences can lead to better generalization and more faithful alignment. Furthermore, many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably. We propose an approach that leverages preferences over preferences, that is annotations indicating which of two preferences reflects a stronger distinction. We use this ordinal signal to infer adaptive margins on a per-datapoint basis. We introduce an extension to Direct Preference Optimization (DPO), DPO-PoP, that incorporates adaptive margins from preference-over-preference supervision, enabling improved discriminative and generative performance. Empirically, our method outperforms vanilla DPO, DPO with fixed margins, and DPO with ground-truth margins on the UltraFeedback dataset. Additionally, we show that there is a tradeoff between discriminative and generative performance: improving test classification accuracy, particularly by correctly labeling weaker preferences at the expense of stronger ones, can lead to a decline in generative quality. To navigate this tradeoff, we propose two sampling strategies to gather preference-over-preference labels: one favoring discriminative performance and one favoring generative performance.
>
---
#### [replaced 073] How Malicious AI Swarms Can Threaten Democracy: The Fusion of Agentic AI and LLMs Marks a New Frontier in Information Warfare
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06299v3](http://arxiv.org/pdf/2506.06299v3)**

> **作者:** Daniel Thilo Schroeder; Meeyoung Cha; Andrea Baronchelli; Nick Bostrom; Nicholas A. Christakis; David Garcia; Amit Goldenberg; Yara Kyrychenko; Kevin Leyton-Brown; Nina Lutz; Gary Marcus; Filippo Menczer; Gordon Pennycook; David G. Rand; Maria Ressa; Frank Schweitzer; Christopher Summerfield; Audrey Tang; Jay J. Van Bavel; Sander van der Linden; Dawn Song; Jonas R. Kunst
>
> **备注:** 15 pages, 1 figure
>
> **摘要:** Public opinion manipulation has entered a new phase, amplifying its roots in rhetoric and propaganda. Advances in large language models (LLMs) and autonomous agents now let influence campaigns reach unprecedented scale and precision. Researchers warn AI could foster mass manipulation. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create election falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, another disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus cheaply. By adaptively mimicking human social dynamics, they threaten democracy.
>
---
#### [replaced 074] Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.04340v2](http://arxiv.org/pdf/2510.04340v2)**

> **作者:** Daniel Tan; Anders Woodruff; Niels Warncke; Arun Jose; Maxime Riché; David Demitri Africa; Mia Taylor
>
> **备注:** 40 pages, 22 figures In proceedings at ICLR 2026
>
> **摘要:** Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., ``You always speak in Spanish.'') teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize.
>
---
#### [replaced 075] Self-Routing RAG: Binding Selective Retrieval with Knowledge Verbalization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.01018v2](http://arxiv.org/pdf/2504.01018v2)**

> **作者:** Di Wu; Jia-Chen Gu; Kai-Wei Chang; Nanyun Peng
>
> **摘要:** Selective retrieval improves the accuracy and efficiency of retrieval-augmented generation (RAG) by reducing distractions from low-quality retrievals. However, existing approaches underutilize the inherent knowledge of large language models (LLMs), leading to suboptimal retrieval decisions and degraded generation performance. To bridge this gap, we propose Self-Routing RAG (SR-RAG), a novel framework that binds selective retrieval with knowledge verbalization. SR-RAG enables an LLM to dynamically decide whether to retrieve external knowledge or verbalize its own parametric knowledge. To this end, we design a multi-task objective that jointly optimizes an LLM for knowledge source selection, knowledge verbalization, and response generation. SR-RAG further incorporates a nearest neighbor search mechanism at inference time to improve the accuracy of knowledge source decisions under domain shifts. Fine-tuning three LLMs with SR-RAG significantly improves both their response accuracy and reduces the inference latency. Compared to the strongest selective retrieval baseline, SR-RAG reduces the number of retrievals by 29% while improving performance by 5.1%.
>
---
#### [replaced 076] OWL: Probing Cross-Lingual Recall of Memorized Texts via World Literature
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22945v2](http://arxiv.org/pdf/2505.22945v2)**

> **作者:** Alisha Srivastava; Emir Korukluoglu; Minh Nhat Le; Duyen Tran; Chau Minh Pham; Marzena Karpinska; Mohit Iyyer
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** Large language models (LLMs) are known to memorize and recall English text from their pretraining data. However, the extent to which this ability generalizes to non-English languages or transfers across languages remains unclear. This paper investigates multilingual and cross-lingual memorization in LLMs, probing if memorized content in one language (e.g., English) can be recalled when presented in translation. To do so, we introduce OWL, a dataset of 31.5K aligned excerpts from 20 books in ten languages, including English originals, official translations (Vietnamese, Spanish, Turkish), and new translations in six low-resource languages (Sesotho, Yoruba, Maithili, Malagasy, Setswana, Tahitian). We evaluate memorization across model families and sizes through three tasks: (1) direct probing, which asks the model to identify a book's title and author; (2) name cloze, which requires predicting masked character names; and (3) prefix probing, which involves generating continuations. We find that LLMs consistently recall content across languages, even for texts without direct translation in pretraining data. GPT-4o, for example, identifies authors and titles 69% of the time and masked entities 6% of the time in newly translated excerpts. Perturbations (e.g., masking characters, shuffling words) modestly reduce direct probing accuracy (7% drop for shuffled official translations). Our results highlight the extent of cross-lingual memorization and provide insights on the differences between the models.
>
---
#### [replaced 077] Intent-Aware Schema Generation And Refinement For Literature Review Tables
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.19521v2](http://arxiv.org/pdf/2507.19521v2)**

> **作者:** Vishakh Padmakumar; Joseph Chee Chang; Kyle Lo; Doug Downey; Aakanksha Naik
>
> **备注:** To Appear at EMNLP Findings 2025
>
> **摘要:** The increasing volume of academic literature makes it essential for researchers to organize, compare, and contrast collections of documents. Large language models (LLMs) can support this process by generating schemas defining shared aspects along which to compare papers. However, progress on schema generation has been slow due to: (i) ambiguity in reference-based evaluations, and (ii) lack of editing/refinement methods. Our work is the first to address both issues. First, we present an approach for augmenting unannotated table corpora with \emph{synthesized intents}, and apply it to create a dataset for studying schema generation conditioned on a given information need, thus reducing ambiguity. With this dataset, we show how incorporating table intents significantly improves baseline performance in reconstructing reference schemas. We start by comprehensively benchmarking several single-shot schema generation methods, including prompted LLM workflows and fine-tuned models, showing that smaller, open-weight models can be fine-tuned to be competitive with state-of-the-art prompted LLMs. Next, we propose several LLM-based schema refinement techniques and show that these can further improve schemas generated by these methods.
>
---
#### [replaced 078] WildIFEval: Instruction Following in the Wild
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06573v2](http://arxiv.org/pdf/2503.06573v2)**

> **作者:** Gili Lior; Asaf Yehudai; Ariel Gera; Liat Ein-Dor
>
> **摘要:** Recent LLMs have shown remarkable success in following user instructions, yet handling instructions with multiple constraints remains a significant challenge. In this work, we introduce WildIFEval - a large-scale dataset of 7K real user instructions with diverse, multi-constraint conditions. Unlike prior datasets, our collection spans a broad lexical and topical spectrum of constraints, extracted from natural user instructions. We categorize these constraints into eight high-level classes to capture their distribution and dynamics in real-world scenarios. Leveraging WildIFEval, we conduct extensive experiments to benchmark the instruction-following capabilities of leading LLMs. WildIFEval clearly differentiates between small and large models, and demonstrates that all models have a large room for improvement on such tasks. We analyze the effects of the number and type of constraints on performance, revealing interesting patterns of model constraint-following behavior. We release our dataset to promote further research on instruction-following under complex, realistic conditions.
>
---
#### [replaced 079] HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in Real-World Multilingual Settings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13671v2](http://arxiv.org/pdf/2410.13671v2)**

> **作者:** Varun Gumma; Ananditha Raghunath; Mohit Jain; Sunayana Sitaram
>
> **摘要:** Assessing the capabilities and limitations of large language models (LLMs) has garnered significant interest, yet the evaluation of multiple models in real-world scenarios remains rare. Multilingual evaluation often relies on translated benchmarks, which typically do not capture linguistic and cultural nuances present in the source language. This study provides an extensive assessment of 24 LLMs on real world data collected from Indian patients interacting with a medical chatbot in Indian English and 4 other Indic languages. We employ a uniform Retrieval Augmented Generation framework to generate responses, which are evaluated using both automated techniques and human evaluators on four specific metrics relevant to our application. We find that models vary significantly in their performance and that instruction tuned Indic models do not always perform well on Indic language queries. Further, we empirically show that factual correctness is generally lower for responses to Indic queries compared to English queries. Finally, our qualitative work shows that code-mixed and culturally relevant queries in our dataset pose challenges to evaluated models.
>
---
#### [replaced 080] AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials
- **分类: cond-mat.mtrl-sci; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04704v2](http://arxiv.org/pdf/2510.04704v2)**

> **作者:** Taoyuze Lv; Alexander Chen; Fengyu Xie; Chu Wu; Jeffrey Meng; Dongzhan Zhou; Bram Hoex; Zhicheng Zhong; Tong Xie
>
> **摘要:** Large Language Models (LLMs) excel at textual reasoning and are beginning to develop spatial understanding, prompting the question of whether these abilities can be combined for complex, domain-specific tasks. This question is essential in fields like materials science, where deep understanding of 3D atomic structures is fundamental. While initial studies have successfully applied LLMs to tasks involving pure crystal generation or coordinate understandings, a standardized benchmark to systematically evaluate their core reasoning abilities across diverse atomic structures has been notably absent. To address this gap, we introduce the AtomWorld benchmark to evaluate LLMs on tasks based in Crystallographic Information Files (CIFs), a standard structure representation format. These tasks, including structural editing, CIF perception, and property-guided modeling, reveal a critical limitation: current models, despite establishing promising baselines, consistently fail in structural understanding and spatial reasoning. Our experiments show that these models make frequent errors on structure modification tasks, and even in the basic CIF format understandings, potentially leading to cumulative errors in subsequent analysis and materials insights. By defining these standardized tasks, AtomWorld lays the ground for advancing LLMs toward robust atomic-scale modeling, crucial for accelerating materials research and automating scientific workflows.
>
---
#### [replaced 081] Fair Play in the Newsroom: Actor-Based Filtering Gender Discrimination in Text Corpora
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2508.13169v2](http://arxiv.org/pdf/2508.13169v2)**

> **作者:** Stefanie Urchs; Veronika Thurner; Matthias Aßenmacher; Christian Heumann; Stephanie Thiemichen
>
> **摘要:** Language corpora are the foundation of most natural language processing research, yet they often reproduce structural inequalities. One such inequality is gender discrimination in how actors are represented, which can distort analyses and perpetuate discriminatory outcomes. This paper introduces a user-centric, actor-level pipeline for detecting and mitigating gender discrimination in large-scale text corpora. By combining discourse-aware analysis with metrics for sentiment, syntactic agency, and quotation styles, our method enables both fine-grained auditing and exclusion-based balancing. Applied to the taz2024full corpus of German newspaper articles (1980-2024), the pipeline yields a more gender-balanced dataset while preserving core dynamics of the source material. Our findings show that structural asymmetries can be reduced through systematic filtering, though subtler biases in sentiment and framing remain. We release the tools and reports to support further research in discourse-based fairness auditing and equitable corpus construction.
>
---
#### [replaced 082] v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18842v4](http://arxiv.org/pdf/2505.18842v4)**

> **作者:** Jiwan Chung; Junhyeok Kim; Siyeol Kim; Jaeyoung Lee; Min Soo Kim; Youngjae Yu
>
> **摘要:** When thinking with images, humans rarely rely on a single glance: they revisit visual information repeatedly during reasoning. However, existing models typically process images only once and thereafter generate reasoning entirely in text, lacking mechanisms to re-access or ground inference in visual representations. We empirically confirm this: as reasoning chains lengthen, models progressively lose focus on relevant regions. In response, we introduce v1, a lightweight extension that enables active visual referencing through a simple point-and-copy approach. This allows the model to identify relevant image patches and copy their embeddings back into the reasoning stream, ensuring that evolving hypotheses remain grounded in perceptual evidence. Crucially, our pointing strategy lets the MLLM directly select image patches using their semantic representations as keys, keeping perceptual evidence embedded in the same space as the model's reasoning. To train this capability, we construct v1g, a dataset of 300K multimodal reasoning traces with interleaved visual grounding annotations. Across various multimodal mathematical reasoning benchmarks, v1 consistently outperforms comparable baselines, establishing point-and-copy as a practical mechanism for grounded reasoning. The model checkpoint and dataset are available at github.com/jun297/v1.
>
---
#### [replaced 083] Explaining GPTs' Schema of Depression: A Machine Behavior Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.13800v2](http://arxiv.org/pdf/2411.13800v2)**

> **作者:** Adithya V Ganesan; Vasudha Varadarajan; Yash Kumar Lal; Veerle C. Eijsbroek; Katarina Kjell; Oscar N. E. Kjell; Tanuja Dhanasekaran; Elizabeth C. Stade; Johannes C. Eichstaedt; Ryan L. Boyd; H. Andrew Schwartz; Lucie Flek
>
> **备注:** 25 pages, 1 table, 4 figures, 1 supplementary table, 5 supplementary figures, 59 references
>
> **摘要:** Use of large language models such as ChatGPT (GPT-4/GPT-5) for mental health support has grown rapidly, emerging as a promising route to assess and help people with mood disorders like depression. However, we have a limited understanding of these language models' schema of mental disorders, that is, how they internally associate and interpret symptoms of such disorders. In this work, we leveraged contemporary measurement theory to decode how GPT-4 and GPT-5 interrelate depressive symptoms, providing an explanation of how LLMs apply what they learn and informing clinical applications. We found that GPT-4 (a) had strong convergent validity with standard instruments and expert judgments $(r = 0.70 - 0.81)$, and (b) behaviorally linked depression symptoms with each other (symptom inter-correlates $r = 0.23 - 0.78$) in accordance with established literature on depression; however, it (c) underemphasized the relationship between $\textit{suicidality}$ and other symptoms while overemphasizing $\textit{psychomotor symptoms}$; and (d) suggested novel hypotheses of symptom mechanisms, for instance, indicating that $\textit{sleep}$ and $\textit{fatigue}$ are broadly influenced by other depressive symptoms, while $\textit{worthlessness/guilt}$ is only tied to $\textit{depressed mood}$. GPT-5 showed a slightly lower convergence with self-report, a difference our machine-behavior analysis makes interpretable through shifts in symptom-symptom relationships. These insights provide an empirical foundation for understanding language models' mental health assessments and demonstrate a generalizable approach for explainability in other models and disorders. Our findings can guide key stakeholders to make informed decisions for effectively situating these technologies in the care system.
>
---
#### [replaced 084] Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14913v3](http://arxiv.org/pdf/2508.14913v3)**

> **作者:** Israel Abebe Azime; Tadesse Destaw Belay; Dietrich Klakow; Philipp Slusallek; Anshuman Chhabra
>
> **摘要:** Large language models (LLMs) have demonstrated significant capabilities in solving mathematical problems expressed in natural language. However, multilingual and culturally-grounded mathematical reasoning in low-resource languages lags behind English due to the scarcity of socio-cultural task datasets that reflect accurate native entities such as person names, organization names, and currencies. Existing multilingual benchmarks are predominantly produced via translation and typically retain English-centric entities, owing to the high cost associated with human annotater-based localization. Moreover, automated localization tools are limited, and hence, truly localized datasets remain scarce. To bridge this gap, we introduce a framework for LLM-driven cultural localization of math word problems that automatically constructs datasets with native names, organizations, and currencies from existing sources. We find that translated benchmarks can obscure true multilingual math ability under appropriate socio-cultural contexts. Through extensive experiments, we also show that our framework can help mitigate English-centric entity bias and improves robustness when native entities are introduced across various languages.
>
---
#### [replaced 085] Generative Interfaces for Language Models
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.19227v2](http://arxiv.org/pdf/2508.19227v2)**

> **作者:** Jiaqi Chen; Yanzhe Zhang; Yutong Zhang; Yijia Shao; Diyi Yang
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) are increasingly seen as assistants, copilots, and consultants, capable of supporting a wide range of tasks through natural conversation. However, most systems remain constrained by a linear request-response format that often makes interactions inefficient in multi-turn, information-dense, and exploratory tasks. To address these limitations, we propose Generative Interfaces for Language Models, a paradigm in which LLMs respond to user queries by proactively generating user interfaces (UIs) that enable more adaptive and interactive engagement. Our framework leverages structured interface-specific representations and iterative refinements to translate user queries into task-specific UIs. For systematic evaluation, we introduce a multidimensional assessment framework that compares generative interfaces with traditional chat-based ones across diverse tasks, interaction patterns, and query types, capturing functional, interactive, and emotional aspects of user experience. Results show that generative interfaces consistently outperform conversational ones, with up to a 72% improvement in human preference. These findings clarify when and why users favor generative interfaces, paving the way for future advancements in human-AI interaction.
>
---
#### [replaced 086] Illusion or Algorithm? Investigating Memorization, Emergence, and Symbolic Processing in In-Context Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11004v3](http://arxiv.org/pdf/2505.11004v3)**

> **作者:** Jingcheng Niu; Subhabrata Dutta; Ahmed Elshabrawy; Harish Tayyar Madabushi; Iryna Gurevych
>
> **备注:** TMLR
>
> **摘要:** Large-scale Transformer language models (LMs) trained solely on next-token prediction with web-scale data can solve a wide range of tasks after seeing just a few examples. The mechanism behind this capability, known as in-context learning (ICL), remains both controversial and poorly understood. Some studies argue that it is merely the result of memorizing vast amounts of data, while others contend that it reflects a fundamental, symbolic algorithmic development in LMs. In this work, we introduce a suite of investigative tasks and a novel method to systematically investigate ICL by leveraging the full Pythia scaling suite, including interim checkpoints that capture progressively larger amount of training data. By carefully exploring ICL performance on downstream tasks and simultaneously conducting a mechanistic analysis of the residual stream's subspace, we demonstrate that ICL extends beyond mere "memorization" of the training corpus, yet does not amount to the implementation of an independent symbolic algorithm. Our results also clarify several aspects of ICL, including the influence of training dynamics, model capabilities, and elements of mechanistic interpretability. Overall, our work advances the understanding of ICL and its implications, offering model developers insights into potential improvements and providing AI security practitioners with a basis for more informed guidelines.
>
---
#### [replaced 087] Compound AI Systems Optimization: A Survey of Methods, Challenges, and Future Directions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08234v2](http://arxiv.org/pdf/2506.08234v2)**

> **作者:** Yu-Ang Lee; Guan-Ting Yi; Mei-Yi Liu; Jui-Chao Lu; Guan-Bo Yang; Yun-Nung Chen
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** Recent advancements in large language models (LLMs) and AI systems have led to a paradigm shift in the design and optimization of complex AI workflows. By integrating multiple components, compound AI systems have become increasingly adept at performing sophisticated tasks. However, as these systems grow in complexity, new challenges arise in optimizing not only individual components but also their interactions. While traditional optimization methods such as supervised fine-tuning (SFT) and reinforcement learning (RL) remain foundational, the rise of natural language feedback introduces promising new approaches, especially for optimizing non-differentiable systems. This paper provides a systematic review of recent progress in optimizing compound AI systems, encompassing both numerical and language-based techniques. We formalize the notion of compound AI system optimization, classify existing methods along several key dimensions, and highlight open research challenges and future directions in this rapidly evolving field. A list of surveyed papers is publicly available at https://github.com/MiuLab/AISysOpt-Survey.
>
---
#### [replaced 088] Speech-Based Cognitive Screening: A Systematic Evaluation of LLM Adaptation Strategies
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.03525v2](http://arxiv.org/pdf/2509.03525v2)**

> **作者:** Fatemeh Taherinezhad; Mohamad Javad Momeni Nezhad; Sepehr Karimi; Sina Rashidi; Ali Zolnour; Maryam Dadkhah; Yasaman Haghbin; Hossein AzadMaleki; Maryam Zolnoori
>
> **摘要:** Over half of US adults with Alzheimer disease and related dementias remain undiagnosed, and speech-based screening offers a scalable detection approach. We compared large language model adaptation strategies for dementia detection using the DementiaBank speech corpus, evaluating nine text-only models and three multimodal audio-text models on recordings from DementiaBank speech corpus. Adaptations included in-context learning with different demonstration selection policies, reasoning-augmented prompting, parameter-efficient fine-tuning, and multimodal integration. Results showed that class-centroid demonstrations achieved the highest in-context learning performance, reasoning improved smaller models, and token-level fine-tuning generally produced the best scores. Adding a classification head substantially improved underperforming models. Among multimodal models, fine-tuned audio-text systems performed well but did not surpass the top text-only models. These findings highlight that model adaptation strategies, including demonstration selection, reasoning design, and tuning method, critically influence speech-based dementia detection, and that properly adapted open-weight models can match or exceed commercial systems.
>
---
#### [replaced 089] AWARE, Beyond Sentence Boundaries: A Contextual Transformer Framework for Identifying Cultural Capital in STEM Narratives
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04983v2](http://arxiv.org/pdf/2510.04983v2)**

> **作者:** Khalid Mehtab Khan; Anagha Kulkarni
>
> **摘要:** Identifying cultural capital (CC) themes in student reflections can offer valuable insights that help foster equitable learning environments in classrooms. However, themes such as aspirational goals or family support are often woven into narratives, rather than appearing as direct keywords. This makes them difficult to detect for standard NLP models that process sentences in isolation. The core challenge stems from a lack of awareness, as standard models are pre-trained on general corpora, leaving them blind to the domain-specific language and narrative context inherent to the data. To address this, we introduce AWARE, a framework that systematically attempts to improve a transformer model's awareness for this nuanced task. AWARE has three core components: 1) Domain Awareness, adapting the model's vocabulary to the linguistic style of student reflections; 2) Context Awareness, generating sentence embeddings that are aware of the full essay context; and 3) Class Overlap Awareness, employing a multi-label strategy to recognize the coexistence of themes in a single sentence. Our results show that by making the model explicitly aware of the properties of the input, AWARE outperforms a strong baseline by 2.1 percentage points in Macro-F1 and shows considerable improvements across all themes. This work provides a robust and generalizable methodology for any text classification task in which meaning depends on the context of the narrative.
>
---
#### [replaced 090] Evaluating and Mitigating Social Bias for Large Language Models in Open-ended Settings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.06134v3](http://arxiv.org/pdf/2412.06134v3)**

> **作者:** Zhao Liu; Tian Xie; Xueru Zhang
>
> **备注:** 15 pages
>
> **摘要:** Current social bias benchmarks for Large Language Models (LLMs) primarily rely on predefined question formats like multiple-choice, limiting their ability to reflect the complexity and open-ended nature of real-world interactions. To close this gap, we extend an existing dataset BBQ (Parrish et al., 2022) to Open-BBQ, a comprehensive framework to evaluate the social bias of LLMs in open-ended settings by incorporating two additional question categories: fill-in-the-blank and short-answer. Since our new Open-BBQ dataset contains a lot of open-ended responses like sentences and paragraphs, we developed an evaluation process to detect biases from open-ended content by labeling sentences and paragraphs. In addition to this, we also found that existing debiasing methods, such as self-debiasing (Gallegos et al., 2024), have over-correction issues, which make the original correct answers incorrect. In order to solve this issue, we propose Composite Prompting, an In-context Learning (ICL) method combining structured examples with explicit chain-of-thought reasoning to form a unified instruction template for LLMs to explicitly identify content that needs debiasing. Experimental results show that the proposed method significantly reduces the bias for both GPT-3.5 and GPT-4o while maintaining high accuracy.
>
---
#### [replaced 091] Tracing Multilingual Factual Knowledge Acquisition in Pretraining
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14824v2](http://arxiv.org/pdf/2505.14824v2)**

> **作者:** Yihong Liu; Mingyang Wang; Amir Hossein Kargaran; Felicia Körner; Ercong Nie; Barbara Plank; François Yvon; Hinrich Schütze
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** Large Language Models (LLMs) are capable of recalling multilingual factual knowledge present in their pretraining data. However, most studies evaluate only the final model, leaving the development of factual recall and crosslingual consistency throughout pretraining largely unexplored. In this work, we trace how factual recall and crosslingual consistency evolve during pretraining, focusing on OLMo-7B as a case study. We find that both accuracy and consistency improve over time for most languages. We show that this improvement is primarily driven by the fact frequency in the pretraining corpus: more frequent facts are more likely to be recalled correctly, regardless of language. Yet, some low-frequency facts in non-English languages can still be correctly recalled. Our analysis reveals that these instances largely benefit from crosslingual transfer of their English counterparts -- an effect that emerges predominantly in the early stages of pretraining. We pinpoint two distinct pathways through which multilingual factual knowledge acquisition occurs: (1) frequency-driven learning, which is dominant and language-agnostic, and (2) crosslingual transfer, which is limited in scale and typically constrained to relation types involving named entities. We release our code and data to facilitate further research at https://github.com/cisnlp/multilingual-fact-tracing.
>
---
#### [replaced 092] BanglaLlama: LLaMA for Bangla Language
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.21200v2](http://arxiv.org/pdf/2410.21200v2)**

> **作者:** Abdullah Khan Zehady; Shubhashis Roy Dipta; Naymul Islam; Safi Al Mamun; Santu Karmaker
>
> **摘要:** Bangla is a language spoken by approximately 240 million native speakers and around 300 million people worldwide. Despite being the 5th largest spoken language in the world, Bangla is still a "low-resource" language, and existing pretrained language models often struggle to perform well on Bangla Language Processing (BLP) tasks. This paper addresses this gap by: (1) introducing two high-quality translated Bangla-instruction datasets totaling 224k samples - Bangla-Orca (172k) and Bangla-Alpaca (52k); and (2) leveraging these datasets to develop BanglaLlama, an open-source family of Bangla-specific LLMs, consisting of five base and instruct variants. We present our methodology, two large datasets, and comprehensive benchmarking results showcasing the effectiveness of our dataset and model on multiple benchmarks. We believe our proposed datasets and models will serve as the new standard baseline for future research focused on this widely spoken yet "low-resource" language.
>
---
#### [replaced 093] Diagnosing the Performance Trade-off in Moral Alignment: A Case Study on Gender Stereotypes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21456v2](http://arxiv.org/pdf/2509.21456v2)**

> **作者:** Guangliang Liu; Bocheng Chen; Xitong Zhang; Kristen Marie Johnson
>
> **摘要:** Moral alignment has emerged as a widely adopted approach for regulating the behavior of pretrained language models (PLMs), typically through fine-tuning on curated datasets. Gender stereotype mitigation is a representational task within the broader application of moral alignment. However, this process often comes at the cost of degraded downstream task performance. Prior studies commonly aim to achieve a performance trade-off by encouraging PLMs to selectively forget only stereotypical knowledge through carefully designed fairness objective, while preserving their language modeling capability (overall forgetting). In this short paper, we investigate whether the performance trade-off can be achieved through the lens of forgetting and the fairness objective. Our analysis shows that the large datasets needed for satisfactory fairness highlight the limitations of current fairness objectives in achieving an effective trade-off: (1) downstream task performance is strongly correlated with overall forgetting; (2) selective forgetting reduces stereotypes, but overall forgetting increases. and (3) general solutions for alleviating forgetting are ineffective at reducing the overall forgetting and fail to improve downstream task performance.
>
---
