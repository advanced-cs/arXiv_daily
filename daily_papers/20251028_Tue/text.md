# 自然语言处理 cs.CL

- **最新发布 184 篇**

- **更新 133 篇**

## 最新发布

#### [new 001] Frustratingly Easy Task-aware Pruning for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大模型剪枝中忽视任务特性的缺陷，提出一种任务感知的剪枝方法。通过融合通用与任务特定数据的特征分布，计算参数重要性并分组剪枝，有效保留模型在特定任务上的性能。可无缝集成现有剪枝技术，显著提升压缩后模型的任务表现。**

- **链接: [http://arxiv.org/pdf/2510.22489v1](http://arxiv.org/pdf/2510.22489v1)**

> **作者:** Yuanhe Tian; Junjie Liu; Xican Yang; Haishan Ye; Yan Song
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Pruning provides a practical solution to reduce the resources required to run large language models (LLMs) to benefit from their effective capabilities as well as control their cost for training and inference. Research on LLM pruning often ranks the importance of LLM parameters using their magnitudes and calibration-data activations and removes (or masks) the less important ones, accordingly reducing LLMs' size. However, these approaches primarily focus on preserving the LLM's ability to generate fluent sentences, while neglecting performance on specific domains and tasks. In this paper, we propose a simple yet effective pruning approach for LLMs that preserves task-specific capabilities while shrinking their parameter space. We first analyze how conventional pruning minimizes loss perturbation under general-domain calibration and extend this formulation by incorporating task-specific feature distributions into the importance computation of existing pruning algorithms. Thus, our framework computes separate importance scores using both general and task-specific calibration data, partitions parameters into shared and exclusive groups based on activation-norm differences, and then fuses their scores to guide the pruning process. This design enables our method to integrate seamlessly with various foundation pruning techniques and preserve the LLM's specialized abilities under compression. Experiments on widely used benchmarks demonstrate that our approach is effective and consistently outperforms the baselines with identical pruning ratios and different settings.
>
---
#### [new 002] Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs
- **分类: cs.CL; cs.AI; I.2.0**

- **简介: 该论文针对大模型生成高质量剧本的难题，提出分解式两阶段框架DSR：先生成叙事文本，再转为规范格式。解决单一模型难以兼顾创意与格式的问题。通过混合数据合成缓解训练数据稀缺，显著提升产出质量，达人类水平82.7%。**

- **链接: [http://arxiv.org/pdf/2510.23163v1](http://arxiv.org/pdf/2510.23163v1)**

> **作者:** Hang Lei; Shengyi Zong; Zhaoyan Li; Ziren Zhou; Hao Liu
>
> **摘要:** The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
>
---
#### [new 003] Personal Care Utility (PCU): Building the Health Infrastructure for Everyday Insight and Guidance
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出个人健康服务（PCU）——一种基于AI的长期健康引导系统，旨在解决传统医疗碎片化问题。通过整合多模态数据与上下文推理，实现个性化健康信息、主动行为指导及治疗响应分析，构建全天候、自适应的健康基础设施，推动个体健康管理与公共健康创新。**

- **链接: [http://arxiv.org/pdf/2510.22602v1](http://arxiv.org/pdf/2510.22602v1)**

> **作者:** Mahyar Abbasian; Ramesh Jain
>
> **备注:** 22 pages, 2 figures, 1 table, Journal paper
>
> **摘要:** Building on decades of success in digital infrastructure and biomedical innovation, we propose the Personal Care Utility (PCU) - a cybernetic system for lifelong health guidance. PCU is conceived as a global, AI-powered utility that continuously orchestrates multimodal data, knowledge, and services to assist individuals and populations alike. Drawing on multimodal agents, event-centric modeling, and contextual inference, it offers three essential capabilities: (1) trusted health information tailored to the individual, (2) proactive health navigation and behavior guidance, and (3) ongoing interpretation of recovery and treatment response after medical events. Unlike conventional episodic care, PCU functions as an ambient, adaptive companion - observing, interpreting, and guiding health in real time across daily life. By integrating personal sensing, experiential computing, and population-level analytics, PCU promises not only improved outcomes for individuals but also a new substrate for public health and scientific discovery. We describe the architecture, design principles, and implementation challenges of this emerging paradigm.
>
---
#### [new 004] LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在真实场景下长依赖理解能力不足的问题，提出LooGLE v2基准。通过自动收集法律、金融等领域的长文本（16k–2M tokens），设计10类任务与1934个复杂问答实例，评估6种本地部署和4种API模型。结果表明，最优模型仅达59.2%准确率，揭示当前大模型在实际长文本理解中存在显著局限。**

- **链接: [http://arxiv.org/pdf/2510.22548v1](http://arxiv.org/pdf/2510.22548v1)**

> **作者:** Ziyuan He; Yuxuan Wang; Jiaqi Li; Kexin Liang; Muhan Zhang
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Large language models (LLMs) are equipped with increasingly extended context windows recently, yet their long context understanding capabilities over long dependency tasks remain fundamentally limited and underexplored. This gap is especially significant in many real-world long-context applications that were rarely benchmarked. In this paper, we introduce LooGLE v2, a novel benchmark designed to evaluate LLMs' long context ability in real-world applications and scenarios. Our benchmark consists of automatically collected real-world long texts, ranging from 16k to 2M tokens, encompassing domains in law, finance, game and code. Accordingly, we delicately design 10 types of domain-specific long-dependency tasks and generate 1,934 QA instances with various diversity and complexity in a scalable data curation pipeline for further practical needs. We conduct a comprehensive assessment of 6 locally deployed and 4 API-based LLMs. The evaluation results show that even the best-performing model achieves only a 59.2% overall score on our benchmark. Despite the extensive context windows, popular LLMs are only capable of understanding a much shorter length of context than they claim to be, revealing significant limitations in their ability to handle real-world tasks with long dependencies and highlighting substantial room for model improvement in practical long-context understanding.
>
---
#### [new 005] A Multi-lingual Dataset of Classified Paragraphs from Open Access Scientific Publications
- **分类: cs.CL; cs.DL**

- **简介: 该论文构建了一个多语言科学文献段落数据集，包含83.3万条标注段落，涵盖四类科学文本：致谢、数据、软件代码和临床试验提及。旨在支持科学文献挖掘中的文本分类与命名实体识别任务，解决跨语言科学信息提取难题。数据基于CC-BY开放获取文献，经GROBID处理并标注语言与领域。**

- **链接: [http://arxiv.org/pdf/2510.21762v1](http://arxiv.org/pdf/2510.21762v1)**

> **作者:** Eric Jeangirard
>
> **摘要:** We present a dataset of 833k paragraphs extracted from CC-BY licensed scientific publications, classified into four categories: acknowledgments, data mentions, software/code mentions, and clinical trial mentions. The paragraphs are primarily in English and French, with additional European languages represented. Each paragraph is annotated with language identification (using fastText) and scientific domain (from OpenAlex). This dataset, derived from the French Open Science Monitor corpus and processed using GROBID, enables training of text classification models and development of named entity recognition systems for scientific literature mining. The dataset is publicly available on HuggingFace https://doi.org/10.57967/hf/6679 under a CC-BY license.
>
---
#### [new 006] Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的组合偏见控制问题，旨在生成符合特定语义组合（如同时包含代理与亲社会描述）的文本。通过对比六种方法发现，仅显式监督微调（SFT）能有效实现高合规性与自然性，而基于偏好学习的方法因无法捕捉逻辑合取关系而失效。**

- **链接: [http://arxiv.org/pdf/2510.22084v1](http://arxiv.org/pdf/2510.22084v1)**

> **作者:** Atij Mahesh
>
> **备注:** 20 pages
>
> **摘要:** Large Language Models (LLMs) still produce gender-stereotyped language even in occupation-neutral contexts that reflect deep societal biases (Rudinger et al., 2018). To address this, prior work has proposed prompting, constrained decoding (Dathathri et al., 2020; Zhou et al., 2024), post-processing, and fine-tuning-based alignment (Rafailov et al., 2023; Ravfogel et al., 2022). However, the comparative efficacy and learning dynamics remain little understood. We report a comparative analysis of six control techniques for bias mitigation: prompt-only, generate-and-filter, DFA-based Ctrl-G decoding, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Iterative Nullspace Projection (INLP). We evaluate each method on a compositional constraint task. This task requires generating sentences that contain at least one agentic and one communal descriptor for each of the twenty Winogender-derived occupations. We quantify trade-offs between control strength and naturalness with evaluations of constraint compliance, lexical diversity, and fluency. Our results reveal key contrasts among the methods: SFT achieves 99.87 +- 0.15% compliance and high lexical diversity, while DPO, despite similar training stability, fails at 4.53 +- 0.82%. Ctrl-G guarantees perfect compliance, but at the cost of severely reduced fluency and diversity. Preference-based learning fundamentally differs: it cannot satisfy compositional constraints, as binary preference signals encode ranking, not logical conjunctions. Only explicit positive supervision enables mitigation of compositional biases; preference-based alignment fails to generalize logical structures, underscoring the limitations of preference learning and the necessity of explicit supervision for fair and fluent controlled generation.
>
---
#### [new 007] Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Ling 2.0，一个基于稀疏激活的万亿参数推理语言模型系列，解决大模型效率与推理能力平衡问题。通过统一的MoE架构、高效训练流程与推理优化，实现高计算效率与强推理性能，推动可扩展、开放的智能推理模型发展。**

- **链接: [http://arxiv.org/pdf/2510.22115v1](http://arxiv.org/pdf/2510.22115v1)**

> **作者:** Ling-Team; Ang Li; Ben Liu; Binbin Hu; Bing Li; Bingwei Zeng; Borui Ye; Caizhi Tang; Changxin Tian; Chao Huang; Chao Zhang; Chen Qian; Chenchen Ju; Chenchen Li; Chengfu Tang; Chili Fu; Chunshao Ren; Chunwei Wu; Cong Zhang; Cunyin Peng; Dafeng Xu; Daixin Wang; Dalong Zhang; Dingnan Jin; Dingyuan Zhu; Dongke Hu; Fangzheng Zhao; Feifan Wu; Feng Zhu; Gangshan Wang; Haitao Zhang; Hailin Zhao; Hanxiao Zhang; Hanzi Wang; Hao Qian; Haoyi Yu; Heng Zhang; Hongliang Zhang; Hongzhi Luan; Huirong Dong; Huizhong Li; Jia Li; Jia Liu; Jialong Zhu; Jian Sha; Jianping Wei; Jiaolong Yang; Jieyue Ma; Jiewei Wu; Jinjing Huang; Jingyun Tian; Jingyuan Zhang; Jinquan Sun; Juanhui Tu; Jun Liu; Jun Xu; Jun Zhou; Junjie Ou; Junpeng Fang; Kaihong Zhang; Kaiqin Hu; Ke Shi; Kun Tang; Kunlong Chen; Lanyin Mei; Lei Liang; Lei Xu; Libo Zhang; Lin Ju; Lin Yuan; Ling Zhong; Lintao Ma; Lu Liu; Lu Yu; Lun Cai; Meiqi Zhu; Mengying Li; Min Chen; Minghao Xue; Minghong Cai; Mingming Yin; Peijie Jiang; Peilong Zhao; Pingping Liu; Qian Zhao; Qing Cui; Qingxiang Huang; Qingyuan Yang; Quankun Yu; Shaowei Wei; Shijie Lian; Shoujian Zheng; Shun Song; Shungen Zhang; Shuo Zhang; Siyuan Li; Song Liu; Ting Guo; Tong Zhao; Wanli Gu; Weichang Wu; Weiguang Han; Wenjing Fang; Wubin Wang; Xiang Shu; Xiao Shi; Xiaoshun Lan; Xiaolu Zhang; Xiaqing Sun; Xin Zhao; Xingyu Lu; Xiong Xu; Xudong Wang; Xudong Wang; Xuemin Yang; Yajie Yang; Yang Xiang; Yanzhe Li; Yi Zhang; Yilong Wang; Yingxue Li; Yongzhen Guo; Yuzhuo Fu; Yuanyuan Wang; Yue Yang; Yue Yu; Yufeng Deng; Yun Zhang; Yunfei Xu; Yuqi Zhang; Yuxiao He; Zengke Gui; Zhaoxin Huan; Zhaoyang Wang; Zhibo Zhu; Zhihao Wang; Zhiqiang Zhang; Zhoufei Wang; Zihang Zeng; Ziqi Liu; Zitao Xuan; Zuoli Tang
>
> **备注:** Ling 2.0 Technical Report
>
> **摘要:** We introduce Ling 2.0, a series reasoning-oriented language foundation built upon the principle that every activation boosts reasoning capability. Designed to scale from tens of billions to one trillion parameters under a unified Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity, cross-scale consistency, and efficiency guided by empirical scaling laws. The series includes three non-thinking (instruct) models - Ling-mini-2.0, Ling-flash-2.0, and Ling-1T - ranging from 16B to 1T total parameters and achieving up to 7-fold active-compute efficiency compared with dense counterparts. Ling 2.0 integrates coordinated innovations across model architecture, pre-training, post-training, and infrastructure: a high-sparsity MoE with MTP for efficient reasoning, reasoning-oriented data and mid-training CoT activation, reinforcement-based fine-tuning (DFT, Evo-CoT), and full-scale FP8 training with fine-grained heterogeneous pipelines. At the trillion scale, Ling-1T establishes a new Pareto frontier of reasoning accuracy versus computational efficiency, demonstrating that sparse activation, when properly aligned with reasoning objectives, enables scalable and efficient intelligence. Collectively, Ling 2.0 provides a coherent, open, and efficient foundation for advancing future reasoning and thinking models, including the Ring series built upon the same base.
>
---
#### [new 008] Tagging-Augmented Generation: Assisting Language Models in Finding Intricate Knowledge In Long Contexts
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大模型在长文本问答中因上下文过长导致知识定位困难的问题，提出Tagging-Augmented Generation（TAG）策略。通过为上下文添加标签或在提示中引入标签定义，提升模型对复杂知识的检索与推理能力，无需修改原始文档，显著改善长文本问答性能。**

- **链接: [http://arxiv.org/pdf/2510.22956v1](http://arxiv.org/pdf/2510.22956v1)**

> **作者:** Anwesan Pal; Karen Hovsepian; Tinghao Guo; Mengnan Zhao; Somendra Tripathi; Nikos Kanakaris; George Mihaila; Sumit Nigam
>
> **备注:** Paper accepted at EMNLP 2025
>
> **摘要:** Recent investigations into effective context lengths of modern flagship large language models (LLMs) have revealed major limitations in effective question answering (QA) and reasoning over long and complex contexts for even the largest and most impressive cadre of models. While approaches like retrieval-augmented generation (RAG) and chunk-based re-ranking attempt to mitigate this issue, they are sensitive to chunking, embedding and retrieval strategies and models, and furthermore, rely on extensive pre-processing, knowledge acquisition and indexing steps. In this paper, we propose Tagging-Augmented Generation (TAG), a lightweight data augmentation strategy that boosts LLM performance in long-context scenarios, without degrading and altering the integrity and composition of retrieved documents. We validate our hypothesis by augmenting two challenging and directly relevant question-answering benchmarks -- NoLima and NovelQA -- and show that tagging the context or even just adding tag definitions into QA prompts leads to consistent performance gains over the baseline -- up to 17% for 32K token contexts, and 2.9% in complex reasoning question-answering for multi-hop queries requiring knowledge across a wide span of text. Additional details are available at https://sites.google.com/view/tag-emnlp.
>
---
#### [new 009] AutoBench: Automating LLM Evaluation through Reciprocal Peer Assessment
- **分类: cs.CL; cs.AI; I.2.7; I.2.11; H.3.4; D.2.8**

- **简介: 该论文提出AutoBench，一种基于互评的自动化大模型评估框架。针对静态基准测试易受数据污染、适应性差的问题，通过模型互为出题者、参赛者与裁判，动态生成任务并聚合多评委意见，实现持续、抗污染的评估。实验验证其结果与主流基准高度相关，显著优于单裁判方案。**

- **链接: [http://arxiv.org/pdf/2510.22593v1](http://arxiv.org/pdf/2510.22593v1)**

> **作者:** Dario Loi; Elena Maria Muià; Federico Siciliano; Giovanni Trappolini; Vincenzo Crisà; Peter Kruger; Fabrizio Silvestri
>
> **摘要:** We present AutoBench, a fully automated and self-sustaining framework for evaluating Large Language Models (LLMs) through reciprocal peer assessment. This paper provides a rigorous scientific validation of the AutoBench methodology, originally developed as an open-source project by eZecute S.R.L.. Unlike static benchmarks that suffer from test-set contamination and limited adaptability, AutoBench dynamically generates novel evaluation tasks while models alternately serve as question generators, contestants, and judges across diverse domains. An iterative weighting mechanism amplifies the influence of consistently reliable evaluators, aggregating peer judgments into consensus-based rankings that reflect collective model agreement. Our experiments demonstrate strong correlations with established benchmarks including MMLU-Pro and GPQA (respectively 78\% and 63\%), validating this peer-driven evaluation paradigm. The multi-judge design significantly outperforms single-judge baselines, confirming that distributed evaluation produces more robust and human-consistent assessments. AutoBench offers a scalable, contamination-resistant alternative to static benchmarks for the continuous evaluation of evolving language models.
>
---
#### [new 010] Integrating Linguistics and AI: Morphological Analysis and Corpus development of Endangered Toto Language of West Bengal
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对濒危语言Toto语的保护问题，结合语言学与人工智能技术，开展形态分析与语料库建设。通过田野调查构建三语（Toto-孟加拉语-英语）标注语料库，开发基于Unicode的数字工具与小型语言模型，实现语言数字化存档与学习应用，推动语言复兴。**

- **链接: [http://arxiv.org/pdf/2510.22629v1](http://arxiv.org/pdf/2510.22629v1)**

> **作者:** Ambalika Guha; Sajal Saha; Debanjan Ballav; Soumi Mitra; Hritwick Chakraborty
>
> **摘要:** Preserving linguistic diversity is necessary as every language offers a distinct perspective on the world. There have been numerous global initiatives to preserve endangered languages through documentation. This paper is a part of a project which aims to develop a trilingual (Toto-Bangla-English) language learning application to digitally archive and promote the endangered Toto language of West Bengal, India. This application, designed for both native Toto speakers and non-native learners, aims to revitalize the language by ensuring accessibility and usability through Unicode script integration and a structured language corpus. The research includes detailed linguistic documentation collected via fieldwork, followed by the creation of a morpheme-tagged, trilingual corpus used to train a Small Language Model (SLM) and a Transformer-based translation engine. The analysis covers inflectional morphology such as person-number-gender agreement, tense-aspect-mood distinctions, and case marking, alongside derivational strategies that reflect word-class changes. Script standardization and digital literacy tools were also developed to enhance script usage. The study offers a sustainable model for preserving endangered languages by incorporating traditional linguistic methodology with AI. This bridge between linguistic research with technological innovation highlights the value of interdisciplinary collaboration for community-based language revitalization.
>
---
#### [new 011] Beyond Semantics: How Temporal Biases Shape Retrieval in Transformer and State-Space Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型在上下文学习中对时间顺序的依赖性，旨在揭示变压器与状态空间模型如何通过时间偏置实现事件检索。通过设计无语义干扰的序列，发现模型更倾向于回忆开头或结尾的重复信息，且两类模型表现相似，揭示了时间偏置在记忆分离中的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.22752v1](http://arxiv.org/pdf/2510.22752v1)**

> **作者:** Anooshka Bajaj; Deven Mahesh Mistry; Sahaj Singh Maini; Yash Aggarwal; Zoran Tiganj
>
> **摘要:** In-context learning is governed by both temporal and semantic relationships, shaping how Large Language Models (LLMs) retrieve contextual information. Analogous to human episodic memory, where the retrieval of specific events is enabled by separating events that happened at different times, this work probes the ability of various pretrained LLMs, including transformer and state-space models, to differentiate and retrieve temporally separated events. Specifically, we prompted models with sequences containing multiple presentations of the same token, which reappears at the sequence end. By fixing the positions of these repeated tokens and permuting all others, we removed semantic confounds and isolated temporal effects on next-token prediction. Across diverse sequences, models consistently placed the highest probabilities on tokens following a repeated token, but with a notable bias for those nearest the beginning or end of the input. An ablation experiment linked this phenomenon in transformers to induction heads. Extending the analysis to unique semantic contexts with partial overlap further demonstrated that memories embedded in the middle of a prompt are retrieved less reliably. Despite architectural differences, state-space and transformer models showed comparable temporal biases. Our findings deepen the understanding of temporal biases in in-context learning and offer an illustration of how these biases can enable temporal separation and episodic retrieval.
>
---
#### [new 012] Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
- **分类: cs.CL**

- **简介: 该论文聚焦于语言模型生成内容的多样性问题，提出Infinity-Chat数据集与分类体系，揭示模型在开放任务中存在“人工蜂群效应”——模型间输出高度同质。研究通过大规模人类标注，发现模型与人类偏好不匹配，尤其在个体差异显著时。旨在推动对AI长期安全风险的系统性评估与缓解。**

- **链接: [http://arxiv.org/pdf/2510.22954v1](http://arxiv.org/pdf/2510.22954v1)**

> **作者:** Liwei Jiang; Yuanjun Chai; Margaret Li; Mickel Liu; Raymond Fok; Nouha Dziri; Yulia Tsvetkov; Maarten Sap; Alon Albalak; Yejin Choi
>
> **备注:** NeurIPS 2025 D&B Paper (Oral); Camera-Ready Version
>
> **摘要:** Language models (LMs) often struggle to generate diverse, human-like creative content, raising concerns about the long-term homogenization of human thought through repeated exposure to similar outputs. Yet scalable methods for evaluating LM output diversity remain limited, especially beyond narrow tasks such as random number or name generation, or beyond repeated sampling from a single model. We introduce Infinity-Chat, a large-scale dataset of 26K diverse, real-world, open-ended user queries that admit a wide range of plausible answers with no single ground truth. We introduce the first comprehensive taxonomy for characterizing the full spectrum of open-ended prompts posed to LMs, comprising 6 top-level categories (e.g., brainstorm & ideation) that further breaks down to 17 subcategories. Using Infinity-Chat, we present a large-scale study of mode collapse in LMs, revealing a pronounced Artificial Hivemind effect in open-ended generation of LMs, characterized by (1) intra-model repetition, where a single model consistently generates similar responses, and more so (2) inter-model homogeneity, where different models produce strikingly similar outputs. Infinity-Chat also includes 31,250 human annotations, across absolute ratings and pairwise preferences, with 25 independent human annotations per example. This enables studying collective and individual-specific human preferences in response to open-ended queries. Our findings show that LMs, reward models, and LM judges are less well calibrated to human ratings on model generations that elicit differing idiosyncratic annotator preferences, despite maintaining comparable overall quality. Overall, INFINITY-CHAT presents the first large-scale resource for systematically studying real-world open-ended queries to LMs, revealing critical insights to guide future research for mitigating long-term AI safety risks posed by the Artificial Hivemind.
>
---
#### [new 013] DCMM-SQL: Automated Data-Centric Pipeline and Multi-Model Collaboration Training for Text-to-SQL Model
- **分类: cs.CL**

- **简介: 该论文针对文本到SQL任务，提出自动化数据驱动流水线与多模型协作训练框架。通过自适应数据修复和错误数据增强提升数据质量，设计多模型协同训练与集成策略，有效克服单模型能力局限，显著提升轻量级模型性能。**

- **链接: [http://arxiv.org/pdf/2510.23284v1](http://arxiv.org/pdf/2510.23284v1)**

> **作者:** Yuanzhen Xie; Liu Ye; Jiqun Chu; Mochi Gao; Hehuan Liu; Yunzhi Tan; Bo Hu; Zang Li
>
> **摘要:** Text-to-SQL tasks have gained attractive improvements since the release of ChatGPT. Among them, agent-based frameworks have been widely used in this field. However, the impact of data-centric strategies on text-to-SQL tasks has rarely been explored. In this paper, we systemically design a fully automated data-centric pipeline for text-to-SQL tasks, including \emph{adaptive data repair}, which can automatically find and fix errors in the training dataset; and \emph{error data augmentation}, where we specifically diffuse and enhance erroneous data predicted by the initially trained models. Meanwhile, we propose a Multi-Model collaboration training schema, aiming to train multiple models with different augmented data, enabling them to possess distinct capabilities and work together to complement each other, because it has been found that the capability of a single fine-tuned model is very limited. Furthermore, we utilize an ensemble strategy to integrate the capabilities of multiple models to solve a multiple-choice question, aiming to further improve the accuracy of text-to-SQL tasks. The experiment results and ablation study have demonstrated the effectiveness of data-centric pipeline and Multi-Model(MM) interactive iterative strategies, achieving first place in lightweight text-to-SQL models (within 70B).
>
---
#### [new 014] Culturally Grounded Physical Commonsense Reasoning in Italian and English: A Submission to the MRL 2025 Shared Task
- **分类: cs.CL**

- **简介: 该论文参与MRL 2025多语言物理常识推理共享任务，针对非英语语言缺乏文化相关物理常识数据的问题，构建了基于意大利语与文化的基准数据集FormaMentis。由母语者专家创建并翻译样本，保留意大利本土文化特征，以提升跨语言物理推理评估的准确性与文化贴合度。**

- **链接: [http://arxiv.org/pdf/2510.22631v1](http://arxiv.org/pdf/2510.22631v1)**

> **作者:** Marco De Santis; Lisa Alazraki
>
> **备注:** MRL 2025 Shared Task on Multilingual Physical Reasoning Datasets
>
> **摘要:** This paper presents our submission to the MRL 2025 Shared Task on Multilingual Physical Reasoning Datasets. The objective of the shared task is to create manually-annotated evaluation data in the physical commonsense reasoning domain, for languages other than English, following a format similar to PIQA. Our contribution, FormaMentis, is a novel benchmark for physical commonsense reasoning that is grounded in Italian language and culture. The data samples in FormaMentis are created by expert annotators who are native Italian speakers and are familiar with local customs and norms. The samples are additionally translated into English, while preserving the cultural elements unique to the Italian context.
>
---
#### [new 015] PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding
- **分类: cs.CL; cs.AI; cs.IR; H.3.3; I.2.7; I.2.6**

- **简介: 该论文针对专利文本嵌入任务，提出PatenTEB基准与patembed模型家族。解决现有基准无法覆盖专利特有挑战的问题，构建包含15项任务的综合性评测集，设计多任务训练模型，显著提升专利检索与分析性能。**

- **链接: [http://arxiv.org/pdf/2510.22264v1](http://arxiv.org/pdf/2510.22264v1)**

> **作者:** Iliass Ayaou; Denis Cavallucci
>
> **摘要:** Patent text embeddings enable prior art search, technology landscaping, and patent analysis, yet existing benchmarks inadequately capture patent-specific challenges. We introduce PatenTEB, a comprehensive benchmark comprising 15 tasks across retrieval, classification, paraphrase, and clustering, with 2.06 million examples. PatenTEB employs domain-stratified splits, domain specific hard negative mining, and systematic coverage of asymmetric fragment-to-document matching scenarios absent from general embedding benchmarks. We develop the patembed model family through multi-task training, spanning 67M to 344M parameters with context lengths up to 4096 tokens. External validation shows strong generalization: patembed-base achieves state-of-the-art on MTEB BigPatentClustering.v2 (0.494 V-measure vs. 0.445 previous best), while patembed-large achieves 0.377 NDCG@100 on DAPFAM. Systematic ablations reveal that multi-task training improves external generalization despite minor benchmark costs, and that domain-pretrained initialization provides consistent advantages across task families. All resources will be made available at https://github.com/iliass-y/patenteb. Keywords: patent retrieval, sentence embeddings, multi-task learning, asymmetric retrieval, benchmark evaluation, contrastive learning.
>
---
#### [new 016] SentiMaithili: A Benchmark Dataset for Sentiment and Reason Generation for the Low-Resource Maithili Language
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对低资源语言Maithili缺乏高质量情感分析数据的问题，构建了首个可解释情感分析基准数据集SentiMaithili。包含3221条带情感标签与母语理由的句子，由专家标注确保质量。实验验证了其在提升模型可解释性方面的有效性，推动了多语言NLP与可解释AI的发展。**

- **链接: [http://arxiv.org/pdf/2510.22160v1](http://arxiv.org/pdf/2510.22160v1)**

> **作者:** Rahul Ranjan; Mahendra Kumar Gurve; Anuj; Nitin; Yamuna Prasad
>
> **摘要:** Developing benchmark datasets for low-resource languages poses significant challenges, primarily due to the limited availability of native linguistic experts and the substantial time and cost involved in annotation. Given these challenges, Maithili is still underrepresented in natural language processing research. It is an Indo-Aryan language spoken by more than 13 million people in the Purvanchal region of India, valued for its rich linguistic structure and cultural significance. While sentiment analysis has achieved remarkable progress in high-resource languages, resources for low-resource languages, such as Maithili, remain scarce, often restricted to coarse-grained annotations and lacking interpretability mechanisms. To address this limitation, we introduce a novel dataset comprising 3,221 Maithili sentences annotated for sentiment polarity and accompanied by natural language justifications. Moreover, the dataset is carefully curated and validated by linguistic experts to ensure both label reliability and contextual fidelity. Notably, the justifications are written in Maithili, thereby promoting culturally grounded interpretation and enhancing the explainability of sentiment models. Furthermore, extensive experiments using both classical machine learning and state-of-the-art transformer architectures demonstrate the dataset's effectiveness for interpretable sentiment analysis. Ultimately, this work establishes the first benchmark for explainable affective computing in Maithili, thus contributing a valuable resource to the broader advancement of multilingual NLP and explainable AI.
>
---
#### [new 017] CHOIR: Collaborative Harmonization fOr Inference Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CHOIR框架，针对大语言模型因角色设定微小变化导致推理不一致的问题，通过协同整合多个反事实角色的推理结果，提升推理鲁棒性。无需额外训练，显著改善跨群体、多架构下的推理性能。**

- **链接: [http://arxiv.org/pdf/2510.22475v1](http://arxiv.org/pdf/2510.22475v1)**

> **作者:** Xiangjue Dong; Cong Wang; Maria Teleki; Millennium Bismay; James Caverlee
>
> **备注:** updated version
>
> **摘要:** Persona-assigned Large Language Models (LLMs) can adopt diverse roles, enabling personalized and context-aware reasoning. However, even minor demographic perturbations in personas, such as simple pronoun changes, can alter reasoning trajectories, leading to divergent sets of correct answers. Instead of treating these variations as biases to be mitigated, we explore their potential as a constructive resource to improve reasoning robustness. We propose CHOIR (Collaborative Harmonization fOr Inference Robustness), a test-time framework that harmonizes multiple persona-conditioned reasoning signals into a unified prediction. CHOIR orchestrates a collaborative decoding process among counterfactual personas, dynamically balancing agreement and divergence in their reasoning paths. Experiments on various reasoning benchmarks demonstrate that CHOIR consistently enhances performance across demographics, model architectures, scales, and tasks - without additional training. Improvements reach up to 26.4% for individual demographic groups and 19.2% on average across five demographics. It remains effective even when base personas are suboptimal. By reframing persona variation as a constructive signal, CHOIR provides a scalable and generalizable approach to more reliable LLM reasoning.
>
---
#### [new 018] VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出VEHME，一个用于评估手写数学表达式的内容的视觉语言模型。针对手写数学解答评估中格式多样、布局无序、符号复杂的问题，通过两阶段训练与视觉提示模块，实现高精度、可解释的评分，显著提升开放题自动评分性能。**

- **链接: [http://arxiv.org/pdf/2510.22798v1](http://arxiv.org/pdf/2510.22798v1)**

> **作者:** Thu Phuong Nguyen; Duc M. Nguyen; Hyotaek Jeon; Hyunwook Lee; Hyunmin Song; Sungahn Ko; Taehwan Kim
>
> **备注:** EMNLP 2025. Project Website: https://vehme.github.io/
>
> **摘要:** Automatically assessing handwritten mathematical solutions is an important problem in educational technology with practical applications, but it remains a significant challenge due to the diverse formats, unstructured layouts, and symbolic complexity of student work. To address this challenge, we introduce VEHME-a Vision-Language Model for Evaluating Handwritten Mathematics Expressions-designed to assess open-form handwritten math responses with high accuracy and interpretable reasoning traces. VEHME integrates a two-phase training pipeline: (i) supervised fine-tuning using structured reasoning data, and (ii) reinforcement learning that aligns model outputs with multi-dimensional grading objectives, including correctness, reasoning depth, and error localization. To enhance spatial understanding, we propose an Expression-Aware Visual Prompting Module, trained on our synthesized multi-line math expressions dataset to robustly guide attention in visually heterogeneous inputs. Evaluated on AIHub and FERMAT datasets, VEHME achieves state-of-the-art performance among open-source models and approaches the accuracy of proprietary systems, demonstrating its potential as a scalable and accessible tool for automated math assessment. Our training and experiment code is publicly available at our GitHub repository.
>
---
#### [new 019] MATCH: Task-Driven Code Evaluation through Contrastive Learning
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出MATCH，一种基于对比学习的参考代码无关代码评估方法。针对生成代码与开发者意图对齐度难评估的问题，通过编码代码与任务描述生成语义嵌入，实现无参考的相似性评分，显著提升与功能正确性和人类偏好的相关性。**

- **链接: [http://arxiv.org/pdf/2510.23169v1](http://arxiv.org/pdf/2510.23169v1)**

> **作者:** Marah Ghoummaid; Vladimir Tchuiev; Ofek Glick; Michal Moschkovitz; Dotan Di Castro
>
> **摘要:** AI-based code generation is increasingly prevalent, with GitHub Copilot estimated to generate 46% of the code on GitHub. Accurately evaluating how well generated code aligns with developer intent remains a critical challenge. Traditional evaluation methods, such as unit tests, are often unscalable and costly. Syntactic similarity metrics (e.g., BLEU, ROUGE) fail to capture code functionality, and metrics like CodeBERTScore require reference code, which is not always available. To address the gap in reference-free evaluation, with few alternatives such as ICE-Score, this paper introduces MATCH, a novel reference-free metric. MATCH uses Contrastive Learning to generate meaningful embeddings for code and natural language task descriptions, enabling similarity scoring that reflects how well generated code implements the task. We show that MATCH achieves stronger correlations with functional correctness and human preference than existing metrics across multiple programming languages.
>
---
#### [new 020] Pedagogy-driven Evaluation of Generative AI-powered Intelligent Tutoring Systems
- **分类: cs.CL**

- **简介: 该论文属于智能辅导系统评估任务，旨在解决生成式AI驱动的ITS缺乏统一、基于教学法的评价框架问题。通过分析真实案例，提出三类基于学习科学的可实践评估方向，推动公平、统一、可扩展的评测方法发展。**

- **链接: [http://arxiv.org/pdf/2510.22581v1](http://arxiv.org/pdf/2510.22581v1)**

> **作者:** Kaushal Kumar Maurya; Ekaterina Kochmar
>
> **备注:** AIED 2025 (BlueSky)
>
> **摘要:** The interdisciplinary research domain of Artificial Intelligence in Education (AIED) has a long history of developing Intelligent Tutoring Systems (ITSs) by integrating insights from technological advancements, educational theories, and cognitive psychology. The remarkable success of generative AI (GenAI) models has accelerated the development of large language model (LLM)-powered ITSs, which have potential to imitate human-like, pedagogically rich, and cognitively demanding tutoring. However, the progress and impact of these systems remain largely untraceable due to the absence of reliable, universally accepted, and pedagogy-driven evaluation frameworks and benchmarks. Most existing educational dialogue-based ITS evaluations rely on subjective protocols and non-standardized benchmarks, leading to inconsistencies and limited generalizability. In this work, we take a step back from mainstream ITS development and provide comprehensive state-of-the-art evaluation practices, highlighting associated challenges through real-world case studies from careful and caring AIED research. Finally, building on insights from previous interdisciplinary AIED research, we propose three practical, feasible, and theoretically grounded research directions, rooted in learning science principles and aimed at establishing fair, unified, and scalable evaluation methodologies for ITSs.
>
---
#### [new 021] Penalizing Length: Uncovering Systematic Bias in Quality Estimation Metrics
- **分类: cs.CL**

- **简介: 该论文研究机器翻译中的质量评估（QE）任务，揭示了现有QE指标存在的长度偏差问题：过长翻译被误判为更多错误，且偏好短译文。为此，提出训练时进行长度归一化和评估时引入参考文本两种策略，有效缓解了长度偏差，提升评估公平性与准确性。**

- **链接: [http://arxiv.org/pdf/2510.22028v1](http://arxiv.org/pdf/2510.22028v1)**

> **作者:** Yilin Zhang; Wenda Xu; Zhongtao Liu; Tetsuji Nakagawa; Markus Freitag
>
> **摘要:** Quality Estimation (QE) metrics are vital in machine translation for reference-free evaluation and as a reward signal in tasks like reinforcement learning. However, the prevalence and impact of length bias in QE have been underexplored. Through a systematic study of top-performing regression-based and LLM-as-a-Judge QE metrics across 10 diverse language pairs, we reveal two critical length biases: First, QE metrics consistently over-predict errors with increasing translation length, even for high-quality, error-free texts. Second, they exhibit a preference for shorter translations when multiple candidates are available for the same source text. These inherent length biases risk unfairly penalizing longer, correct translations and can lead to sub-optimal decision-making in applications such as QE reranking and QE guided reinforcement learning. To mitigate this, we propose two strategies: (a) applying length normalization during model training, and (b) incorporating reference texts during evaluation. Both approaches were found to effectively reduce the identified length bias.
>
---
#### [new 022] IPQA: A Benchmark for Core Intent Identification in Personalized Question Answering
- **分类: cs.CL**

- **简介: 该论文聚焦个性化问答中的核心意图识别任务，针对现有基准未评估意图识别能力的问题，提出IPQA基准。通过用户行为推断核心意图，构建多领域数据集，并验证模型在复杂场景下识别能力不足，推动该方向研究。**

- **链接: [http://arxiv.org/pdf/2510.23536v1](http://arxiv.org/pdf/2510.23536v1)**

> **作者:** Jieyong Kim; Maryam Amirizaniani; Soojin Yoon; Dongha Lee
>
> **摘要:** Intent identification serves as the foundation for generating appropriate responses in personalized question answering (PQA). However, existing benchmarks evaluate only response quality or retrieval performance without directly measuring intent identification capabilities. This gap is critical because without understanding which intents users prioritize, systems cannot generate responses satisfying individual information needs. To address this, we introduce the concept of core intents: intents users prioritize when selecting answers to satisfy their information needs. To evaluate these core intents, we propose IPQA, a benchmark for core Intent identification in Personalized Question Answering. Since users do not explicitly state their prioritized intents, we derive core intents from observable behavior patterns in answer selection, grounded in satisficing theory where users choose answers meeting their acceptance thresholds. We construct a dataset with various domains through systematic filtering, LLM-based annotation, and rigorous quality control combining automated verification with human validation. Experimental evaluations across state-of-the-art language models reveal that current systems struggle with core intent identification in personalized contexts. Models fail to identify core intents from user histories, with performance degrading as question complexity increases. The code and dataset will be made publicly available to facilitate future research in this direction.
>
---
#### [new 023] A Cocktail-Party Benchmark: Multi-Modal dataset and Comparative Evaluation Results
- **分类: cs.CL**

- **简介: 该论文提出多模态上下文感知识别（MCoRec）任务，旨在解决单房间内多人重叠对话的鸡尾酒会问题。通过音频、视觉与上下文线索联合建模，实现“谁何时说什么及与谁对话”的精准识别。研究构建了真实场景下的多模态数据集，设计并评估了基线系统，验证了多模态融合的有效性。**

- **链接: [http://arxiv.org/pdf/2510.23276v1](http://arxiv.org/pdf/2510.23276v1)**

> **作者:** Thai-Binh Nguyen; Katerina Zmolikova; Pingchuan Ma; Ngoc Quan Pham; Christian Fuegen; Alexander Waibel
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We introduce the task of Multi-Modal Context-Aware Recognition (MCoRec) in the ninth CHiME Challenge, which addresses the cocktail-party problem of overlapping conversations in a single-room setting using audio, visual, and contextual cues. MCoRec captures natural multi-party conversations where the recordings focus on unscripted, casual group chats, leading to extreme speech overlap of up to 100% and highly fragmented conversational turns. The task requires systems to answer the question "Who speaks when, what, and with whom?" by jointly transcribing each speaker's speech and clustering them into their respective conversations from audio-visual recordings. Audio-only baselines exceed 100% word error rate, whereas incorporating visual cues yields substantial 50% improvements, highlighting the importance of multi-modality. In this manuscript, we present the motivation behind the task, outline the data collection process, and report the baseline systems developed for the MCoRec.
>
---
#### [new 024] MMTutorBench: The First Multimodal Benchmark for AI Math Tutoring
- **分类: cs.CL**

- **简介: 该论文提出MMTutorBench，首个面向AI数学辅导的多模态基准。针对现有模型缺乏诊断与渐进引导能力的问题，构建685个含关键步骤的题目，设计六维评分体系，涵盖洞察发现、运算构想与执行三任务。评估12个MLLMs，揭示开源与闭源系统差距，验证了鲁棒性评价方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.23477v1](http://arxiv.org/pdf/2510.23477v1)**

> **作者:** Tengchao Yang; Sichen Guo; Mengzhao Jia; Jiaming Su; Yuanyang Liu; Zhihan Zhang; Meng Jiang
>
> **摘要:** Effective math tutoring requires not only solving problems but also diagnosing students' difficulties and guiding them step by step. While multimodal large language models (MLLMs) show promise, existing benchmarks largely overlook these tutoring skills. We introduce MMTutorBench, the first benchmark for AI math tutoring, consisting of 685 problems built around pedagogically significant key-steps. Each problem is paired with problem-specific rubrics that enable fine-grained evaluation across six dimensions, and structured into three tasks-Insight Discovery, Operation Formulation, and Operation Execution. We evaluate 12 leading MLLMs and find clear performance gaps between proprietary and open-source systems, substantial room compared to human tutors, and consistent trends across input variants: OCR pipelines degrade tutoring quality, few-shot prompting yields limited gains, and our rubric-based LLM-as-a-Judge proves highly reliable. These results highlight both the difficulty and diagnostic value of MMTutorBench for advancing AI tutoring.
>
---
#### [new 025] DETECT: Determining Ease and Textual Clarity of German Text Simplifications
- **分类: cs.CL**

- **简介: 该论文针对德语自动文本简化（ATS）评估难题，提出首个专用指标DETECT。通过合成大模型数据构建无须人工标注的训练集，并引入模型精炼机制，实现对简化度、语义保留和流畅性的全面评估。实验表明，DETECT在各项指标上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22212v1](http://arxiv.org/pdf/2510.22212v1)**

> **作者:** Maria Korobeynikova; Alessia Battisti; Lukas Fischer; Yingqiang Gao
>
> **摘要:** Current evaluation of German automatic text simplification (ATS) relies on general-purpose metrics such as SARI, BLEU, and BERTScore, which insufficiently capture simplification quality in terms of simplicity, meaning preservation, and fluency. While specialized metrics like LENS have been developed for English, corresponding efforts for German have lagged behind due to the absence of human-annotated corpora. To close this gap, we introduce DETECT, the first German-specific metric that holistically evaluates ATS quality across all three dimensions of simplicity, meaning preservation, and fluency, and is trained entirely on synthetic large language model (LLM) responses. Our approach adapts the LENS framework to German and extends it with (i) a pipeline for generating synthetic quality scores via LLMs, enabling dataset creation without human annotation, and (ii) an LLM-based refinement step for aligning grading criteria with simplification requirements. To the best of our knowledge, we also construct the largest German human evaluation dataset for text simplification to validate our metric directly. Experimental results show that DETECT achieves substantially higher correlations with human judgments than widely used ATS metrics, with particularly strong gains in meaning preservation and fluency. Beyond ATS, our findings highlight both the potential and the limitations of LLMs for automatic evaluation and provide transferable guidelines for general language accessibility tasks.
>
---
#### [new 026] Evolution of the lexicon: a probabilistic point of view
- **分类: cs.CL; q-bio.PE**

- **简介: 该论文属于语言演化研究，旨在提高语言间时间分离估计的精度。针对传统Swadesh方法因假设不现实导致误差的问题，作者从概率视角分析了词汇替换与渐进性词汇变异两大随机过程，指出后者显著影响词库演变，并提出纳入此过程可提升估算准确性。**

- **链接: [http://arxiv.org/pdf/2510.22220v1](http://arxiv.org/pdf/2510.22220v1)**

> **作者:** Maurizio Serva
>
> **摘要:** The Swadesh approach for determining the temporal separation between two languages relies on the stochastic process of words replacement (when a complete new word emerges to represent a given concept). It is well known that the basic assumptions of the Swadesh approach are often unrealistic due to various contamination phenomena and misjudgments (horizontal transfers, variations over time and space of the replacement rate, incorrect assessments of cognacy relationships, presence of synonyms, and so on). All of this means that the results cannot be completely correct. More importantly, even in the unrealistic case that all basic assumptions are satisfied, simple mathematics places limits on the accuracy of estimating the temporal separation between two languages. These limits, which are purely probabilistic in nature and which are often neglected in lexicostatistical studies, are analyzed in detail in this article. Furthermore, in this work we highlight that the evolution of a language's lexicon is also driven by another stochastic process: gradual lexical modification of words. We show that this process equally also represents a major contribution to the reshaping of the vocabulary of languages over the centuries and we also show, from a purely probabilistic perspective, that taking into account this second random process significantly increases the precision in determining the temporal separation between two languages.
>
---
#### [new 027] Quality-Aware Translation Tagging in Multilingual RAG system
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言检索增强生成（mRAG）中低资源语言翻译质量差的问题，提出质量感知翻译标记方法（QTT-RAG）。通过三维度评估翻译质量并附加元数据，提升生成可靠性，避免事实错误。实验证明其在跨语言问答任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.23070v1](http://arxiv.org/pdf/2510.23070v1)**

> **作者:** Hoyeon Moon; Byeolhee Kim; Nikhil Verma
>
> **备注:** EMNLP 2025 MRL Workshop
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) often retrieves English documents and translates them into the query language for low-resource settings. However, poor translation quality degrades response generation performance. Existing approaches either assume sufficient translation quality or utilize the rewriting method, which introduces factual distortion and hallucinations. To mitigate these problems, we propose Quality-Aware Translation Tagging in mRAG (QTT-RAG), which explicitly evaluates translation quality along three dimensions-semantic equivalence, grammatical accuracy, and naturalness&fluency-and attach these scores as metadata without altering the original content. We evaluate QTT-RAG against CrossRAG and DKM-RAG as baselines in two open-domain QA benchmarks (XORQA, MKQA) using six instruction-tuned LLMs ranging from 2.4B to 14B parameters, covering two low-resource languages (Korean and Finnish) and one high-resource language (Chinese). QTT-RAG outperforms the baselines by preserving factual integrity while enabling generator models to make informed decisions based on translation reliability. This approach allows for effective usage of cross-lingual documents in low-resource settings with limited native language documents, offering a practical and robust solution across multilingual domains.
>
---
#### [new 028] Understanding Network Behaviors through Natural Language Question-Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NetMind框架，旨在通过自然语言问答理解复杂网络行为。针对大容量配置、设备异构性和复杂拓扑带来的挑战，采用树状分块、统一事实图与混合语言设计，提升LLM的准确性和可扩展性。研究属于网络行为理解任务，解决了传统方法学习成本高、灵活性差的问题。**

- **链接: [http://arxiv.org/pdf/2510.21894v1](http://arxiv.org/pdf/2510.21894v1)**

> **作者:** Mingzhe Xing; Chang Tian; Jianan Zhang; Lichen Pan; Peipei Liu; Zhaoteng Yan; Yinliang Yue
>
> **备注:** Large Language Models
>
> **摘要:** Modern large-scale networks introduce significant complexity in understanding network behaviors, increasing the risk of misconfiguration. Prior work proposed to understand network behaviors by mining network configurations, typically relying on domain-specific languages interfaced with formal models. While effective, they suffer from a steep learning curve and limited flexibility. In contrast, natural language (NL) offers a more accessible and interpretable interface, motivating recent research on NL-guided network behavior understanding. Recent advances in large language models (LLMs) further enhance this direction, leveraging their extensive prior knowledge of network concepts and strong reasoning capabilities. However, three key challenges remain: 1) numerous router devices with lengthy configuration files challenge LLM's long-context understanding ability; 2) heterogeneity across devices and protocols impedes scalability; and 3) complex network topologies and protocols demand advanced reasoning abilities beyond the current capabilities of LLMs. To tackle the above challenges, we propose NetMind, a novel framework for querying networks using NL. Our approach introduces a tree-based configuration chunking strategy to preserve semantic coherence while enabling efficient partitioning. We then construct a unified fact graph as an intermediate representation to normalize vendor-specific configurations. Finally, we design a hybrid imperative-declarative language to reduce the reasoning burden on LLMs and enhance precision. We contribute a benchmark consisting of NL question-answer pairs paired with network configurations. Experiments demonstrate that NetMind achieves accurate and scalable network behavior understanding, outperforming existing baselines.
>
---
#### [new 029] MMPersuade: A Dataset and Evaluation Framework for Multimodal Persuasion
- **分类: cs.CL**

- **简介: 该论文提出MMPersuade框架，研究大视觉语言模型在多模态说服下的响应机制。针对模型易受误导的风险，构建多模态说服数据集与评估体系，量化模型说服效果与敏感性，揭示多模态信息增强说服力、不同策略在不同场景的差异，为开发更稳健、符合伦理的模型提供依据。**

- **链接: [http://arxiv.org/pdf/2510.22768v1](http://arxiv.org/pdf/2510.22768v1)**

> **作者:** Haoyi Qiu; Yilun Zhou; Pranav Narayanan Venkit; Kung-Hsiang Huang; Jiaxin Zhang; Nanyun Peng; Chien-Sheng Wu
>
> **摘要:** As Large Vision-Language Models (LVLMs) are increasingly deployed in domains such as shopping, health, and news, they are exposed to pervasive persuasive content. A critical question is how these models function as persuadees-how and why they can be influenced by persuasive multimodal inputs. Understanding both their susceptibility to persuasion and the effectiveness of different persuasive strategies is crucial, as overly persuadable models may adopt misleading beliefs, override user preferences, or generate unethical or unsafe outputs when exposed to manipulative messages. We introduce MMPersuade, a unified framework for systematically studying multimodal persuasion dynamics in LVLMs. MMPersuade contributes (i) a comprehensive multimodal dataset that pairs images and videos with established persuasion principles across commercial, subjective and behavioral, and adversarial contexts, and (ii) an evaluation framework that quantifies both persuasion effectiveness and model susceptibility via third-party agreement scoring and self-estimated token probabilities on conversation histories. Our study of six leading LVLMs as persuadees yields three key insights: (i) multimodal inputs substantially increase persuasion effectiveness-and model susceptibility-compared to text alone, especially in misinformation scenarios; (ii) stated prior preferences decrease susceptibility, yet multimodal information maintains its persuasive advantage; and (iii) different strategies vary in effectiveness across contexts, with reciprocity being most potent in commercial and subjective contexts, and credibility and logic prevailing in adversarial contexts. By jointly analyzing persuasion effectiveness and susceptibility, MMPersuade provides a principled foundation for developing models that are robust, preference-consistent, and ethically aligned when engaging with persuasive multimodal content.
>
---
#### [new 030] Text to Trust: Evaluating Fine-Tuning and LoRA Trade-offs in Language Models for Unfair Terms of Service Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于服务条款中不公平条款的检测任务，旨在解决大模型在法律领域适应时成本高昂的问题。通过对比全量微调、LoRA及零样本提示等方法，评估其在精度与内存效率间的权衡，验证了LoRA在降低资源消耗的同时保持较高召回率的有效性。**

- **链接: [http://arxiv.org/pdf/2510.22531v1](http://arxiv.org/pdf/2510.22531v1)**

> **作者:** Noshitha Padma Pratyusha Juttu; Sahithi Singireddy; Sravani Gona; Sujal Timilsina
>
> **备注:** 6 pages, including figures and tables. All experiments are reproducible. Code and fine-tuned models are publicly available on: GitHub: (https://github.com/Stimils02/UnfairTOSAgreementsDetection) and Hugging Face: (https://huggingface.co/Noshitha98)
>
> **摘要:** Large Language Models (LLMs) have transformed text understanding, yet their adaptation to specialized legal domains remains constrained by the cost of full fine-tuning. This study provides a systematic evaluation of fine tuning, parameter efficient adaptation (LoRA, QLoRA), and zero-shot prompting strategies for unfair clause detection in Terms of Service (ToS) documents, a key application in legal NLP. We finetune BERT and DistilBERT, apply 4-bit Low-Rank Adaptation (LoRA) to models such as TinyLlama, LLaMA 3B/7B, and SaulLM, and evaluate GPT-4o and O-versions in zero-shot settings. Experiments on the CLAUDETTE-ToS benchmark and the Multilingual Scraper Corpus show that full fine-tuning achieves the strongest precision recall balance, while LoRA-based models provide competitive recall with up to 3x lower memory cost. These findings highlight practical design trade-offs for efficient and domain-adapted LLMs, contributing open baselines for fine-tuning research in legal text processing.
>
---
#### [new 031] Knocking-Heads Attention
- **分类: cs.CL**

- **简介: 该论文提出敲头注意力（KHA），用于改进多头注意力机制。针对现有方法中各注意力头独立、缺乏交互的问题，KHA通过共享对角初始化投影矩阵实现头间特征级交互，增强模型表达能力。该方法参数与计算开销极低，可无缝集成于多种注意力架构，显著提升训练稳定性和下游任务性能。**

- **链接: [http://arxiv.org/pdf/2510.23052v1](http://arxiv.org/pdf/2510.23052v1)**

> **作者:** Zhanchao Zhou; Xiaodong Chen; Haoxing Chen; Zhenzhong Lan; Jianguo Li
>
> **摘要:** Multi-head attention (MHA) has become the cornerstone of modern large language models, enhancing representational capacity through parallel attention heads. However, increasing the number of heads inherently weakens individual head capacity, and existing attention mechanisms - whether standard MHA or its variants like grouped-query attention (GQA) and grouped-tied attention (GTA) - simply concatenate outputs from isolated heads without strong interaction. To address this limitation, we propose knocking-heads attention (KHA), which enables attention heads to "knock" on each other - facilitating cross-head feature-level interactions before the scaled dot-product attention. This is achieved by applying a shared, diagonally-initialized projection matrix across all heads. The diagonal initialization preserves head-specific specialization at the start of training while allowing the model to progressively learn integrated cross-head representations. KHA adds only minimal parameters and FLOPs and can be seamlessly integrated into MHA, GQA, GTA, and other attention variants. We validate KHA by training a 6.1B parameter MoE model (1.01B activated) on 1T high-quality tokens. Compared to baseline attention mechanisms, KHA brings superior and more stable training dynamics, achieving better performance across downstream tasks.
>
---
#### [new 032] Language Server CLI Empowers Language Agents with Process Rewards
- **分类: cs.CL; cs.AI; cs.PL; cs.SE**

- **简介: 该论文针对语言模型在代码生成中幻觉和定位错误的问题，提出Lanser-CLI框架，通过语言服务器提供可验证的结构化信息与过程奖励，实现确定性、可复现的编码流程。工作包括：符号化选择器、分析包封装、安全变更机制及在线可计算的过程奖励函数，提升代理决策的真实性与可控性。**

- **链接: [http://arxiv.org/pdf/2510.22907v1](http://arxiv.org/pdf/2510.22907v1)**

> **作者:** Yifan Zhang; Lanser Contributors
>
> **备注:** Project Page: https://github.com/yifanzhang-pro/lanser-cli
>
> **摘要:** Large language models routinely hallucinate APIs and mislocalize edits, while language servers compute verified, IDE-grade facts about real code. We present Lanser-CLI, a CLI-first orchestration layer that pins and mediates a Language Server Protocol (LSP) server for coding agents and CI, exposing deterministic, replayable workflows. Our position is that language servers provide not only structural information (definitions, references, types, diagnostics) but also an actionable process reward: machine-checked, step-wise signals that align an agent's planning loop with program reality. In this work, Lanser-CLI contributes: (i) a robust addressing scheme beyond brittle "file:line:col" via a Selector DSL (symbolic, AST-path, and content-anchored selectors) with a principled relocation algorithm; (ii) deterministic Analysis Bundles that normalize Language Server responses and capture environment/capability metadata with stable content hashes; (iii) a safety envelope for mutating operations (rename, code actions) with preview, workspace jails, and Git-aware, transactional apply; and (iv) a process-reward functional derived from Language Server facts (diagnostic deltas, disambiguation confidence, and safe-apply checks) that is computable online and replayable offline. We formalize determinism under frozen snapshots and establish a monotonicity property for the process reward, making it suitable for process supervision and counterfactual analysis. Project Page: https://github.com/yifanzhang-pro/lanser-cli
>
---
#### [new 033] Leveraging Hierarchical Organization for Medical Multi-document Summarization
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文研究医疗多文档摘要任务，旨在通过引入层次化结构提升模型组织跨文档信息的能力。对比传统扁平方法，实验表明层次结构能增强摘要的清晰度、事实性与人类偏好，且GPT-4模拟判断与人类评价具较高一致性。**

- **链接: [http://arxiv.org/pdf/2510.23104v1](http://arxiv.org/pdf/2510.23104v1)**

> **作者:** Yi-Li Hsu; Katelyn X. Mei; Lucy Lu Wang
>
> **摘要:** Medical multi-document summarization (MDS) is a complex task that requires effectively managing cross-document relationships. This paper investigates whether incorporating hierarchical structures in the inputs of MDS can improve a model's ability to organize and contextualize information across documents compared to traditional flat summarization methods. We investigate two ways of incorporating hierarchical organization across three large language models (LLMs), and conduct comprehensive evaluations of the resulting summaries using automated metrics, model-based metrics, and domain expert evaluation of preference, understandability, clarity, complexity, relevance, coverage, factuality, and coherence. Our results show that human experts prefer model-generated summaries over human-written summaries. Hierarchical approaches generally preserve factuality, coverage, and coherence of information, while also increasing human preference for summaries. Additionally, we examine whether simulated judgments from GPT-4 align with human judgments, finding higher agreement along more objective evaluation facets. Our findings demonstrate that hierarchical structures can improve the clarity of medical summaries generated by models while maintaining content coverage, providing a practical way to improve human preference for generated summaries.
>
---
#### [new 034] ENTP: Enhancing Low-Quality SFT Data via Neural-Symbolic Text Purge-Mix
- **分类: cs.CL**

- **简介: 该论文针对低质量指令微调数据利用率低的问题，提出ENTP框架，结合符号化净化与神经重构，从低质数据中提取高价值信息。通过智能筛选与数据增强，显著提升数据质量与多样性，在多个基准上优于现有方法，甚至超越全量原始数据微调效果。**

- **链接: [http://arxiv.org/pdf/2510.23160v1](http://arxiv.org/pdf/2510.23160v1)**

> **作者:** Zile Yang; Ling Li; Na Di; Jinlong Pang; Yao Zhou; Hao Cheng; Bo Han; Jiaheng Wei
>
> **摘要:** Supervised Fine-Tuning (SFT) adapts pre-trained Large Language Models (LLMs) to domain-specific instructions by training on a carefully curated subset of high-quality instruction-response pairs, typically drawn from a larger dataset that often contains many low-quality or noisy samples. However, existing quality-first paradigms often overlook valuable signals in discarded low-quality data and rely on imperfect quality filters. We introduce ENTP (Enhancing low-quality SFT data via Neural-symbolic Text Purge-Mix), a framework that revitalizes low-quality corpora through symbolic purification and neural reconstruction. The symbolic module identifies and prunes noisy samples based on statistical priors, while the neural component synthesizes enriched instruction-response pairs by leveraging latent representations and model knowledge. This neural-symbolic synergy enhances data informativeness and diversity. Experiments show that ENTP-augmented datasets, constructed exclusively from low-quality data, outperform 13 established data-selection baselines across five instruction-following benchmarks, and even surpass fine-tuning on the full original dataset (approximately 300K examples). Our results highlight the untapped potential of low-quality data and underscore the importance of intelligent purification and synthesis for efficient instruction alignment.
>
---
#### [new 035] SABlock: Semantic-Aware KV Cache Eviction with Adaptive Compression Block Size
- **分类: cs.CL**

- **简介: 该论文针对长文本大模型推理中KV缓存内存占用过大的问题，提出SABlock框架。通过语义感知的分段与自适应块大小优化，在保证语义连贯性的同时提升压缩效率，显著降低内存使用并加速解码。**

- **链接: [http://arxiv.org/pdf/2510.22556v1](http://arxiv.org/pdf/2510.22556v1)**

> **作者:** Jinhan Chen; Jianchun Liu; Hongli Xu; Xianjun Gao; Shilong Wang
>
> **摘要:** The growing memory footprint of the Key-Value (KV) cache poses a severe scalability bottleneck for long-context Large Language Model (LLM) inference. While KV cache eviction has emerged as an effective solution by discarding less critical tokens, existing token-, block-, and sentence-level compression methods struggle to balance semantic coherence and memory efficiency. To this end, we introduce SABlock, a \underline{s}emantic-aware KV cache eviction framework with \underline{a}daptive \underline{block} sizes. Specifically, SABlock first performs semantic segmentation to align compression boundaries with linguistic structures, then applies segment-guided token scoring to refine token importance estimation. Finally, for each segment, a budget-driven search strategy adaptively determines the optimal block size that preserves semantic integrity while improving compression efficiency under a given cache budget. Extensive experiments on long-context benchmarks demonstrate that SABlock consistently outperforms state-of-the-art baselines under the same memory budgets. For instance, on Needle-in-a-Haystack (NIAH), SABlock achieves 99.9% retrieval accuracy with only 96 KV entries, nearly matching the performance of the full-cache baseline that retains up to 8K entries. Under a fixed cache budget of 1,024, SABlock further reduces peak memory usage by 46.28% and achieves up to 9.5x faster decoding on a 128K context length.
>
---
#### [new 036] Gradual Forgetting: Logarithmic Compression for Extending Transformer Context Windows
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对长文本建模中Transformer上下文窗口有限的问题，提出在输入层面采用尺度不变的对数压缩，以延长模型记忆长度。无需修改架构，仅通过压缩输入序列，即可显著降低困惑度，并随上下文变长持续提升性能，实现简单高效的长程依赖建模。**

- **链接: [http://arxiv.org/pdf/2510.22109v1](http://arxiv.org/pdf/2510.22109v1)**

> **作者:** Billy Dickson; Zoran Tiganj
>
> **摘要:** Most approaches to long-context processing increase the complexity of the transformer's internal architecture by integrating mechanisms such as recurrence or auxiliary memory modules. In this work, we introduce an alternative approach that modifies the input representation itself, rather than the transformer architecture. Inspired by cognitive models of human memory, our method applies a scale-invariant logarithmic compression to the input tokens. The resulting compressed representation is processed by a standard, unmodified transformer, preserving architectural simplicity. We evaluate this approach on the WikiText-103 and PG-19 language modeling benchmarks, showing a reduction in perplexity compared to uncompressed baselines. Moreover, performance improves consistently with longer compressed temporal contexts, showing that input-level logarithmic compression is a simple and effective way to extend a transformer's long-range memory.
>
---
#### [new 037] Estimating the Error of Large Language Models at Pairwise Text Comparison
- **分类: cs.CL; cs.AI; math.PR**

- **简介: 该论文研究大语言模型在文本配对比较中的错误率估计问题。提出无需真实标签的两种误差估计方法，基于Copeland计数分析模型偏好一致性，评估六款LLM在五类文本上的表现，发现Claude综合性能最优。**

- **链接: [http://arxiv.org/pdf/2510.22219v1](http://arxiv.org/pdf/2510.22219v1)**

> **作者:** Tianyi Li
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** We measure LLMs' output error at pairwise text comparison, noting the probability of error in their preferences. Our method does not rely on the ground truth and supports two scenarios: (i) uniform error rate regardless of the order of comparison, estimated with two comparisons for each text pair with either text placed first; (ii) binary positional bias assuming distinct error rates for the two orders of comparison, estimated with repeated comparisons between the texts. The Copeland counting constructs a ranking over the compared texts from pairwise preferences; the ranking reveals the poor scalability of LLM-based pairwise comparison and helps yield the estimates for LLMs' error rates. We apply the method to six LLMs (ChatGPT, Claude, DeepSeek, Gemini, Grok, Qwen) with five types of text input and obtain consistent estimates of LLMs' error. In general, the measured two positional bias terms are similar, close to the uniform error. Considering both the error rates and the robustness to the variation of prompts, Claude obtained the most desirable performance in this experiment. Our model outperforms the biased Bradley-Terry model and the commutativity score in indicating LLMs' error at this task.
>
---
#### [new 038] PerCoR: Evaluating Commonsense Reasoning in Persian via Multiple-Choice Sentence Completion
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文提出PerCoR，首个大规模波斯语常识推理基准，包含10.6万道多项选择句补全题。为提升多样性与难度，采用结合分割策略生成题目，并提出DRESS-AF方法筛选干扰项。实验表明该数据集挑战性强，人类与模型表现差异显著，且DRESS-AF可迁移至英文任务。**

- **链接: [http://arxiv.org/pdf/2510.22616v1](http://arxiv.org/pdf/2510.22616v1)**

> **作者:** Morteza Alikhani; Mohammadtaha Bagherifard; Erfan Zinvandi; Mehran Sarmadi
>
> **备注:** 20 pages, 17 figures, Accepted to IJCNLP-AACL 2025 (Main Conference)
>
> **摘要:** We introduced PerCoR (Persian Commonsense Reasoning), the first large-scale Persian benchmark for commonsense reasoning. PerCoR contains 106K multiple-choice sentence-completion problems drawn from more than forty news, cultural, and other web sources. We introduce a novel conjunction-based segmentation strategy to generate coherent sentence-completion pairs, enabling broad topical and structural diversity. To create challenging distractors, we propose DRESS-AF (Distractor Ranking via Embedding Similarity Scoring and Adversarial Filtering), a generation-free adversarial filtering method that selects distractors from the pool of gold continuations while maximising model confusion. Human annotators score 89% on PerCoR, while OpenAI-o3 achieves the highest performance at 92.18%, followed closely by Claude-Sonnet-3.7 (91.17%). The strongest open-source model, DeepSeek-R1, reaches 82.51%, underscoring both the dataset's difficulty and the remaining performance gap in Persian commonsense reasoning. We further show that DRESS-AF transfers to the English HellaSwag benchmark, increasing its difficulty without hurting human solvability. The dataset is available at https://huggingface.co/datasets/MCINext/PerCoR.
>
---
#### [new 039] Corpus Frequencies in Morphological Inflection: Do They Matter?
- **分类: cs.CL**

- **简介: 该论文研究形态屈折任务中词汇频率的影响。针对传统方法忽略词频分布的问题，提出三方面改进：频率加权数据划分、引入词频权重的准确率评估、频率感知训练采样。实验表明，频率感知训练在43种语言中26种优于均匀采样，提升了模型在真实语料中的表现。**

- **链接: [http://arxiv.org/pdf/2510.23131v1](http://arxiv.org/pdf/2510.23131v1)**

> **作者:** Tomáš Sourada; Jana Straková
>
> **备注:** Published in the proceedings of ITAT 2025.15 pages, 1 figure, 4 tables
>
> **摘要:** The traditional approach to morphological inflection (the task of modifying a base word (lemma) to express grammatical categories) has been, for decades, to consider lexical entries of lemma-tag-form triples uniformly, lacking any information about their frequency distribution. However, in production deployment, one might expect the user inputs to reflect a real-world distribution of frequencies in natural texts. With future deployment in mind, we explore the incorporation of corpus frequency information into the task of morphological inflection along three key dimensions during system development: (i) for train-dev-test split, we combine a lemma-disjoint approach, which evaluates the model's generalization capabilities, with a frequency-weighted strategy to better reflect the realistic distribution of items across different frequency bands in training and test sets; (ii) for evaluation, we complement the standard type accuracy (often referred to simply as accuracy), which treats all items equally regardless of frequency, with token accuracy, which assigns greater weight to frequent words and better approximates performance on running text; (iii) for training data sampling, we introduce a method novel in the context of inflection, frequency-aware training, which explicitly incorporates word frequency into the sampling process. We show that frequency-aware training outperforms uniform sampling in 26 out of 43 languages.
>
---
#### [new 040] Scalable Supervising Software Agents with Patch Reasoner
- **分类: cs.CL; cs.SE**

- **简介: 该论文针对软件工程大模型代理的可扩展性问题，提出R4P补丁验证模型，通过推理实现高效、稳定的奖励机制。解决测试依赖重、易被攻击及数据稀疏问题。利用组级目标进行强化学习训练，提升代理性能，显著优于基线模型，且验证速度达测试的50倍。**

- **链接: [http://arxiv.org/pdf/2510.22775v1](http://arxiv.org/pdf/2510.22775v1)**

> **作者:** Junjielong Xu; Boyin Tan; Xiaoyuan Liu; Chao Peng; Pengfei Gao; Pinjia He
>
> **摘要:** While large language model agents have advanced software engineering tasks, the unscalable nature of existing test-based supervision is limiting the potential improvement of data scaling. The reason is twofold: (1) building and running test sandbox is rather heavy and fragile, and (2) data with high-coverage tests is naturally rare and threatened by test hacking via edge cases. In this paper, we propose R4P, a patch verifier model to provide scalable rewards for training and testing SWE agents via reasoning. We consider that patch verification is fundamentally a reasoning task, mirroring how human repository maintainers review patches without writing and running new reproduction tests. To obtain sufficient reference and reduce the risk of reward hacking, R4P uses a group-wise objective for RL training, enabling it to verify multiple patches against each other's modification and gain a dense reward for stable training. R4P achieves 72.2% Acc. for verifying patches from SWE-bench-verified, surpassing OpenAI o3. To demonstrate R4P's practicality, we design and train a lite scaffold, Mini-SE, with pure reinforcement learning where all rewards are derived from R4P. As a result, Mini-SE achieves 26.2% Pass@1 on SWE-bench-verified, showing a 10.0% improvement over the original Qwen3-32B. This can be further improved to 32.8% with R4P for test-time scaling. Furthermore, R4P verifies patches within a second, 50x faster than testing on average. The stable scaling curves of rewards and accuracy along with high efficiency reflect R4P's practicality.
>
---
#### [new 041] LightKGG: Simple and Efficient Knowledge Graph Generation from Textual Data
- **分类: cs.CL**

- **简介: 该论文针对知识图谱生成任务，解决现有方法依赖高资源大模型或易出错的模式匹配问题。提出LightKGG框架，利用小规模语言模型，通过上下文融合图结构与拓扑增强关系推理，实现高效精准的知识图谱构建，显著降低硬件需求，提升实用性。**

- **链接: [http://arxiv.org/pdf/2510.23341v1](http://arxiv.org/pdf/2510.23341v1)**

> **作者:** Teng Lin
>
> **摘要:** The scarcity of high-quality knowledge graphs (KGs) remains a critical bottleneck for downstream AI applications, as existing extraction methods rely heavily on error-prone pattern-matching techniques or resource-intensive large language models (LLMs). While recent tools leverage LLMs to generate KGs, their computational demands limit accessibility for low-resource environments. Our paper introduces LightKGG, a novel framework that enables efficient KG extraction from textual data using small-scale language models (SLMs) through two key technical innovations: (1) Context-integrated Graph extraction integrates contextual information with nodes and edges into a unified graph structure, reducing the reliance on complex semantic processing while maintaining more key information; (2) Topology-enhanced relationship inference leverages the inherent topology of the extracted graph to efficiently infer relationships, enabling relationship discovery without relying on complex language understanding capabilities of LLMs. By enabling accurate KG construction with minimal hardware requirements, this work bridges the gap between automated knowledge extraction and practical deployment scenarios while introducing scientifically rigorous methods for optimizing SLM efficiency in structured NLP tasks.
>
---
#### [new 042] A Survey on LLM Mid-training
- **分类: cs.CL**

- **简介: 该论文聚焦大语言模型（LLM）的中段训练（mid-training）任务，旨在明确其定义与作用。针对预训练与后训练之间能力提升不足的问题，提出系统性优化框架，涵盖数据、策略与架构，构建分类体系，推动LLM在数学、编码等能力上的精准增强。**

- **链接: [http://arxiv.org/pdf/2510.23081v1](http://arxiv.org/pdf/2510.23081v1)**

> **作者:** Chengying Tu; Xuemiao Zhang; Rongxiang Weng; Rumei Li; Chen Zhang; Yang Bai; Hongfei Yan; Jingang Wang; Xunliang Cai
>
> **摘要:** Recent advances in foundation models have highlighted the significant benefits of multi-stage training, with a particular emphasis on the emergence of mid-training as a vital stage that bridges pre-training and post-training. Mid-training is distinguished by its use of intermediate data and computational resources, systematically enhancing specified capabilities such as mathematics, coding, reasoning, and long-context extension, while maintaining foundational competencies. This survey provides a formal definition of mid-training for large language models (LLMs) and investigates optimization frameworks that encompass data curation, training strategies, and model architecture optimization. We analyze mainstream model implementations in the context of objective-driven interventions, illustrating how mid-training serves as a distinct and critical stage in the progressive development of LLM capabilities. By clarifying the unique contributions of mid-training, this survey offers a comprehensive taxonomy and actionable insights, supporting future research and innovation in the advancement of LLMs.
>
---
#### [new 043] Interpreting and Mitigating Unwanted Uncertainty in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型中的意外不确定性问题，即重复提问时答案翻转。通过模拟真实场景，发现非检索注意力头过度关注误导性词元是主因。提出掩码这些头的方法，有效降低答案翻转率15%，且不引发不连贯。工作聚焦于机制解释与不确定性缓解。**

- **链接: [http://arxiv.org/pdf/2510.22866v1](http://arxiv.org/pdf/2510.22866v1)**

> **作者:** Tiasa Singha Roy; Ayush Rajesh Jhaveri; Ilias Triantafyllopoulos
>
> **摘要:** Despite their impressive capabilities, Large Language Models (LLMs) exhibit unwanted uncertainty, a phenomenon where a model changes a previously correct answer into an incorrect one when re-prompted. This behavior undermines trust and poses serious risks in high-stakes domains. In this work, we investigate the mechanisms that drive this phenomenon. We adapt the Needle-in-a-Haystack retrieval framework and integrate a Flip-style re-evaluation prompt to simulate realistic answer-flipping scenarios. We find that retrieval heads are not primarily responsible for avoiding uncertainty. Instead, we identify a small set of non-retrieval attention heads that disproportionately attend to misleading tokens in uncertain contexts. Masking these heads yields significant improvements, reducing flip behavior by up to 15% without introducing incoherence or overcorrection. However, when tested for downstream tasks, we observe trade-offs with flip behavior. Our findings contribute to the growing field of mechanistic interpretability and present a simple yet effective technique for mitigating uncertainty-driven failure modes in LLMs.
>
---
#### [new 044] Process Reward Models for Sentence-Level Verification of LVLM Radiology Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型生成放射科报告中的事实性幻觉问题，提出轻量级句子级过程奖励模型（PRM），基于临床上下文判断每句正确性。通过弱监督微调实现高精度验证与跨模型泛化，有效提升报告质量，无需访问模型内部状态。**

- **链接: [http://arxiv.org/pdf/2510.23217v1](http://arxiv.org/pdf/2510.23217v1)**

> **作者:** Alois Thomas; Maya Varma; Jean-Benoit Delbrouck; Curtis P. Langlotz
>
> **摘要:** Automating radiology report generation with Large Vision-Language Models (LVLMs) holds great potential, yet these models often produce clinically critical hallucinations, posing serious risks. Existing hallucination detection methods frequently lack the necessary sentence-level granularity or robust generalization across different LVLM generators. We introduce a novel approach: a sentence-level Process Reward Model (PRM) adapted for this vision-language task. Our PRM predicts the factual correctness of each generated sentence, conditioned on clinical context and preceding text. When fine-tuned on MIMIC-CXR with weakly-supervised labels, a lightweight 0.5B-parameter PRM outperforms existing verification techniques, demonstrating, for instance, relative improvements of 7.5% in Matthews Correlation Coefficient and 1.8% in AUROC over strong white-box baselines on outputs from one LVLM. Unlike methods reliant on internal model states, our PRM demonstrates strong generalization to an unseen LVLM. We further show its practical utility: PRM scores effectively filter low-quality reports, improving F1-CheXbert scores by 4.5% (when discarding the worst 10% of reports). Moreover, when guiding a novel weighted best-of-N selection process on the MIMIC-CXR test set, our PRM show relative improvements in clinical metrics of 7.4% for F1-CheXbert and 0.6% for BERTScore. These results demonstrate that a lightweight, context-aware PRM provides a model-agnostic safety layer for clinical LVLMs without access to internal activations
>
---
#### [new 045] SI-Bench: Benchmarking Social Intelligence of Large Language Models in Human-to-Human Conversations
- **分类: cs.CL**

- **简介: 该论文提出SI-Bench基准，用于评估大语言模型在真实人际对话中的社会智能。针对现有数据集多为模拟对话、缺乏真实社交动态的问题，研究收集2221条真实多轮对话并人工标注312条，发现当前模型虽在推理上超越人类，但在回复质量上仍不足，且链式思维可能降低表现。**

- **链接: [http://arxiv.org/pdf/2510.23182v1](http://arxiv.org/pdf/2510.23182v1)**

> **作者:** Shuai Huang; Wenxuan Zhao; Jun Gao
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** As large language models (LLMs) develop anthropomorphic abilities, they are increasingly being deployed as autonomous agents to interact with humans. However, evaluating their performance in realistic and complex social interactions remains a significant challenge. Most previous research built datasets through simulated agent-to-agent interactions, which fails to capture the authentic linguistic styles and relational dynamics found in real human conversations. To address this gap, we introduce SI-Bench, a novel benchmark designed to evaluate aspects of social intelligence in LLMs. Grounded in broad social science theories, SI-Bench contains 2,221 authentic multi-turn dialogues collected from a social networking application. We further selected a subset of 312 dialogues for manual annotation across 8 major models. The experiments show that SOTA models have surpassed the human expert in process reasoning under complex social situations, yet they still fall behind humans in reply quality. Moreover, introducing Chain-of-Thought (CoT) reasoning may degrade the performance of LLMs in social dialogue tasks. All datasets are openly available at https://github.com/SI-Bench/SI-Bench.git.
>
---
#### [new 046] A Sociophonetic Analysis of Racial Bias in Commercial ASR Systems Using the Pacific Northwest English Corpus
- **分类: cs.CL**

- **简介: 该论文研究商业语音识别系统中的种族偏见问题，聚焦于美国西北部英语口音的声学社会语言学差异。通过构建语音错误率（PER）指标，分析四类族裔群体在11个语音特征上的识别准确率，发现非裔美国人因低后元音合并和鼻音前合并等发音特征导致显著更高错误率，揭示了声学建模中方言变异是主要偏见来源。**

- **链接: [http://arxiv.org/pdf/2510.22495v1](http://arxiv.org/pdf/2510.22495v1)**

> **作者:** Michael Scott; Siyu Liang; Alicia Wassink; Gina-Anne Levow
>
> **摘要:** This paper presents a systematic evaluation of racial bias in four major commercial automatic speech recognition (ASR) systems using the Pacific Northwest English (PNWE) corpus. We analyze transcription accuracy across speakers from four ethnic backgrounds (African American, Caucasian American, ChicanX, and Yakama) and examine how sociophonetic variation contributes to differential system performance. We introduce a heuristically-determined Phonetic Error Rate (PER) metric that links recognition errors to specific linguistically motivated variables derived from sociophonetic annotation. Our analysis of eleven sociophonetic features reveals that vowel quality variation, particularly resistance to the low-back merger and pre-nasal merger patterns, is systematically associated with differential error rates across ethnic groups, with the most pronounced effects for African American speakers across all evaluated systems. These findings demonstrate that acoustic modeling of dialectal phonetic variation, rather than lexical or syntactic factors, remains a primary source of bias in commercial ASR systems. The study establishes the PNWE corpus as a valuable resource for bias evaluation in speech technologies and provides actionable guidance for improving ASR performance through targeted representation of sociophonetic diversity in training data.
>
---
#### [new 047] Iterative Layer Pruning for Efficient Translation Inference
- **分类: cs.CL; cs.PF**

- **简介: 该论文针对大语言模型在机器翻译中推理效率低的问题，提出基于层重要性分析的迭代层剪枝方法。通过逐步剪除低重要性层，在保持翻译质量的前提下显著减少模型大小和推理时间，适用于多语言翻译任务。**

- **链接: [http://arxiv.org/pdf/2510.22763v1](http://arxiv.org/pdf/2510.22763v1)**

> **作者:** Yasmin Moslem; Muhammad Hazim Al Farouq; John D. Kelleher
>
> **备注:** WMT 2025
>
> **摘要:** Large language models (LLMs) have transformed many areas of natural language processing, including machine translation. However, efficient deployment of LLMs remains challenging due to their intensive computational requirements. In this paper, we address this challenge and present our submissions to the Model Compression track at the Conference on Machine Translation (WMT 2025). In our experiments, we investigate iterative layer pruning guided by layer importance analysis. We evaluate this method using the Aya-Expanse-8B model for translation from Czech to German, and from English to Egyptian Arabic. Our approach achieves substantial reductions in model size and inference time, while maintaining the translation quality of the baseline models.
>
---
#### [new 048] Code Aesthetics with Agentic Reward Feedback
- **分类: cs.CL**

- **简介: 该论文聚焦代码美学优化任务，针对LLM生成代码视觉质量差的问题，构建AesCode-358K数据集，提出基于多智能体的奖励反馈机制与GRPO-AR算法，联合优化代码功能与美学，并建立OpenDesign基准。实验表明方法显著提升代码美观性，优于GPT-4o等模型。**

- **链接: [http://arxiv.org/pdf/2510.23272v1](http://arxiv.org/pdf/2510.23272v1)**

> **作者:** Bang Xiao; Lingjie Jiang; Shaohan Huang; Tengchao Lv; Yupan Huang; Xun Wu; Lei Cui; Furu Wei
>
> **备注:** 30 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) have become valuable assistants for developers in code-related tasks. While LLMs excel at traditional programming tasks such as code generation and bug fixing, they struggle with visually-oriented coding tasks, often producing suboptimal aesthetics. In this paper, we introduce a new pipeline to enhance the aesthetic quality of LLM-generated code. We first construct AesCode-358K, a large-scale instruction-tuning dataset focused on code aesthetics. Next, we propose agentic reward feedback, a multi-agent system that evaluates executability, static aesthetics, and interactive aesthetics. Building on this, we develop GRPO-AR, which integrates these signals into the GRPO algorithm for joint optimization of functionality and code aesthetics. Finally, we develop OpenDesign, a benchmark for assessing code aesthetics. Experimental results show that combining supervised fine-tuning on AesCode-358K with reinforcement learning using agentic reward feedback significantly improves performance on OpenDesign and also enhances results on existing benchmarks such as PandasPlotBench. Notably, our AesCoder-4B surpasses GPT-4o and GPT-4.1, and achieves performance comparable to large open-source models with 480B-685B parameters, underscoring the effectiveness of our approach.
>
---
#### [new 049] A Comprehensive Dataset for Human vs. AI Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文针对生成式AI文本检测与溯源问题，构建了包含5.8万条真实与合成文本的综合性数据集。通过融合《纽约时报》原文与多模型生成文本，建立了人类与AI文本区分及模型溯源的基准测试，推动可信AI内容识别技术发展。**

- **链接: [http://arxiv.org/pdf/2510.22874v1](http://arxiv.org/pdf/2510.22874v1)**

> **作者:** Rajarshi Roy; Nasrin Imanpour; Ashhar Aziz; Shashwat Bajpai; Gurpreet Singh; Shwetangshu Biswas; Kapil Wanaskar; Parth Patwa; Subhankar Ghosh; Shreyas Dixit; Nilesh Ranjan Pal; Vipula Rawte; Ritvik Garimella; Gaytri Jena; Amit Sheth; Vasu Sharma; Aishwarya Naresh Reganti; Vinija Jain; Aman Chadha; Amitava Das
>
> **备注:** Defactify4 @AAAI 2025
>
> **摘要:** The rapid advancement of large language models (LLMs) has led to increasingly human-like AI-generated text, raising concerns about content authenticity, misinformation, and trustworthiness. Addressing the challenge of reliably detecting AI-generated text and attributing it to specific models requires large-scale, diverse, and well-annotated datasets. In this work, we present a comprehensive dataset comprising over 58,000 text samples that combine authentic New York Times articles with synthetic versions generated by multiple state-of-the-art LLMs including Gemma-2-9b, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, and GPT-4-o. The dataset provides original article abstracts as prompts, full human-authored narratives. We establish baseline results for two key tasks: distinguishing human-written from AI-generated text, achieving an accuracy of 58.35\%, and attributing AI texts to their generating models with an accuracy of 8.92\%. By bridging real-world journalistic content with modern generative models, the dataset aims to catalyze the development of robust detection and attribution methods, fostering trust and transparency in the era of generative AI. Our dataset is available at: https://huggingface.co/datasets/gsingh1-py/train.
>
---
#### [new 050] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多语言预训练中的缩放定律，旨在解决英语主导的缩放规律无法适配多语言的问题。通过774次实验，提出自适应迁移缩放律（ATLAS），揭示跨语言迁移机制与最优扩展策略，指导模型规模与数据量的协同增长，并识别从头预训练与微调的计算拐点，推动多语言AI高效发展。**

- **链接: [http://arxiv.org/pdf/2510.22037v1](http://arxiv.org/pdf/2510.22037v1)**

> **作者:** Shayne Longpre; Sneha Kudugunta; Niklas Muennighoff; I-Hung Hsu; Isaac Caswell; Alex Pentland; Sercan Arik; Chen-Yu Lee; Sayna Ebrahimi
>
> **摘要:** Scaling laws research has focused overwhelmingly on English -- yet the most prominent AI models explicitly serve billions of international users. In this work, we undertake the largest multilingual scaling laws study to date, totaling 774 multilingual training experiments, spanning 10M-8B model parameters, 400+ training languages and 48 evaluation languages. We introduce the Adaptive Transfer Scaling Law (ATLAS) for both monolingual and multilingual pretraining, which outperforms existing scaling laws' out-of-sample generalization often by more than 0.3 R^2. Our analyses of the experiments shed light on multilingual learning dynamics, transfer properties between languages, and the curse of multilinguality. First, we derive a cross-lingual transfer matrix, empirically measuring mutual benefit scores between 38 x 38=1444 language pairs. Second, we derive a language-agnostic scaling law that reveals how to optimally scale model size and data when adding languages without sacrificing performance. Third, we identify the computational crossover points for when to pretrain from scratch versus finetune from multilingual checkpoints. We hope these findings provide the scientific foundation for democratizing scaling laws across languages, and enable practitioners to efficiently scale models -- beyond English-first AI.
>
---
#### [new 051] Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中情感的内部表征，旨在揭示情感在隐藏状态空间中的几何结构。通过分析发现情感存在于低维流形中，具有方向性、分布性和可解释性，且跨语言、跨数据集稳定。研究提出干预模块可操控情感表达而不破坏语义，证实了情感空间的通用性与可操纵性。**

- **链接: [http://arxiv.org/pdf/2510.22042v1](http://arxiv.org/pdf/2510.22042v1)**

> **作者:** Benjamin Reichman; Adar Avsian; Larry Heck
>
> **摘要:** This work investigates how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden-state space. The paper identifies a low-dimensional emotional manifold and shows that emotional representations are directionally encoded, distributed across layers, and aligned with interpretable dimensions. These structures are stable across depth and generalize to eight real-world emotion datasets spanning five languages. Cross-domain alignment yields low error and strong linear probe performance, indicating a universal emotional subspace. Within this space, internal emotion perception can be steered while preserving semantics using a learned intervention module, with especially strong control for basic emotions across languages. These findings reveal a consistent and manipulable affective geometry in LLMs and offer insight into how they internalize and process emotion.
>
---
#### [new 052] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大模型判别器在复杂约束验证与计算上的局限性，提出TIR-Judge框架，通过集成代码执行器与强化学习，实现工具增强的推理判别。工作包括多域训练、灵活判断格式支持及无需蒸馏的迭代强化学习，显著提升评估准确率，且小模型性能接近大模型。**

- **链接: [http://arxiv.org/pdf/2510.23038v1](http://arxiv.org/pdf/2510.23038v1)**

> **作者:** Ran Xu; Jingjing Chen; Jiayu Ye; Yu Wu; Jun Yan; Carl Yang; Hongkun Yu
>
> **备注:** Work in Progress
>
> **摘要:** Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning.
>
---
#### [new 053] Hope Speech Detection in Social Media English Corpora: Performance of Traditional and Transformer Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于社交媒体中希望话语的检测任务，旨在识别表达积极行动力与目标导向的言语。研究比较了传统机器学习模型与微调的Transformer模型在该任务上的表现，结果表明Transformer模型在精度和召回率上更优，尤其在小数据集上展现出潜力。**

- **链接: [http://arxiv.org/pdf/2510.23585v1](http://arxiv.org/pdf/2510.23585v1)**

> **作者:** Luis Ramos; Hiram Calvo; Olga Kolesnikova
>
> **摘要:** The identification of hope speech has become a promised NLP task, considering the need to detect motivational expressions of agency and goal-directed behaviour on social media platforms. This proposal evaluates traditional machine learning models and fine-tuned transformers for a previously split hope speech dataset as train, development and test set. On development test, a linear-kernel SVM and logistic regression both reached a macro-F1 of 0.78; SVM with RBF kernel reached 0.77, and Na\"ive Bayes hit 0.75. Transformer models delivered better results, the best model achieved weighted precision of 0.82, weighted recall of 0.80, weighted F1 of 0.79, macro F1 of 0.79, and 0.80 accuracy. These results suggest that while optimally configured traditional machine learning models remain agile, transformer architectures detect some subtle semantics of hope to achieve higher precision and recall in hope speech detection, suggesting that larges transformers and LLMs could perform better in small datasets.
>
---
#### [new 054] Model-Aware Tokenizer Transfer
- **分类: cs.CL**

- **简介: 该论文针对多语言大模型中低资源语言适配难题，提出模型感知的分词器迁移方法MATT。通过引入注意力影响建模（AIM）目标，利用源模型的跨标记交互模式指导目标模型嵌入初始化，提升分词器迁移质量。实验表明，MATT能快速恢复模型性能，优于传统语义启发方法。**

- **链接: [http://arxiv.org/pdf/2510.21954v1](http://arxiv.org/pdf/2510.21954v1)**

> **作者:** Mykola Haltiuk; Aleksander Smywiński-Pohl
>
> **摘要:** Large Language Models (LLMs) are trained to support an increasing number of languages, yet their predefined tokenizers remain a bottleneck for adapting models to lower-resource or distinct-script languages. Existing tokenizer transfer methods typically rely on semantic heuristics to initialize new embeddings, ignoring higher-layer model dynamics and limiting transfer quality. We propose Model-Aware Tokenizer Transfer (MATT), a method that incorporates model internals into the tokenizer transfer process. MATT introduces an Attention Influence Modeling (AIM) objective that distills inter-token communication patterns from a source model into a target model with a new tokenizer, providing an efficient warm-up before standard language modeling. Unlike approaches that focus solely on embedding similarity, MATT leverages attention behavior to guide embedding initialization and adaptation. Experiments across diverse linguistic settings show that MATT recovers a large fraction of the original model's performance within a few GPU hours, outperforming heuristic baselines. These results demonstrate that incorporating model-level signals offers a practical and effective path toward robust tokenizer transfer in multilingual LLMs.
>
---
#### [new 055] From Slides to Chatbots: Enhancing Large Language Models with University Course Materials
- **分类: cs.CL**

- **简介: 该论文研究如何利用大学课程材料提升大语言模型在计算机科学教育中的表现。针对课程材料（如含图表的讲义、口语化课件）与常规文本差异大的问题，比较了RAG与持续预训练两种方法，发现RAG更高效，并且以图像形式呈现讲义能显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.22272v1](http://arxiv.org/pdf/2510.22272v1)**

> **作者:** Tu Anh Dinh; Philipp Nicolas Schumacher; Jan Niehues
>
> **摘要:** Large Language Models (LLMs) have advanced rapidly in recent years. One application of LLMs is to support student learning in educational settings. However, prior work has shown that LLMs still struggle to answer questions accurately within university-level computer science courses. In this work, we investigate how incorporating university course materials can enhance LLM performance in this setting. A key challenge lies in leveraging diverse course materials such as lecture slides and transcripts, which differ substantially from typical textual corpora: slides also contain visual elements like images and formulas, while transcripts contain spoken, less structured language. We compare two strategies, Retrieval-Augmented Generation (RAG) and Continual Pre-Training (CPT), to extend LLMs with course-specific knowledge. For lecture slides, we further explore a multi-modal RAG approach, where we present the retrieved content to the generator in image form. Our experiments reveal that, given the relatively small size of university course materials, RAG is more effective and efficient than CPT. Moreover, incorporating slides as images in the multi-modal setting significantly improves performance over text-only retrieval. These findings highlight practical strategies for developing AI assistants that better support learning and teaching, and we hope they inspire similar efforts in other educational contexts.
>
---
#### [new 056] MAP4TS: A Multi-Aspect Prompting Framework for Time-Series Forecasting with Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对时间序列预测任务，提出MAP4TS框架，通过多方面提示（全局、局部、统计、时序）融合经典时间序列分析方法，增强大语言模型对时序数据的理解。实验表明，该方法显著提升预测性能，且结构化提示在长周期预测中优于大型模型。**

- **链接: [http://arxiv.org/pdf/2510.23090v1](http://arxiv.org/pdf/2510.23090v1)**

> **作者:** Suchan Lee; Jihoon Choi; Sohyeon Lee; Minseok Song; Bong-Gyu Jang; Hwanjo Yu; Soyeon Caren Han
>
> **摘要:** Recent advances have investigated the use of pretrained large language models (LLMs) for time-series forecasting by aligning numerical inputs with LLM embedding spaces. However, existing multimodal approaches often overlook the distinct statistical properties and temporal dependencies that are fundamental to time-series data. To bridge this gap, we propose MAP4TS, a novel Multi-Aspect Prompting Framework that explicitly incorporates classical time-series analysis into the prompt design. Our framework introduces four specialized prompt components: a Global Domain Prompt that conveys dataset-level context, a Local Domain Prompt that encodes recent trends and series-specific behaviors, and a pair of Statistical and Temporal Prompts that embed handcrafted insights derived from autocorrelation (ACF), partial autocorrelation (PACF), and Fourier analysis. Multi-Aspect Prompts are combined with raw time-series embeddings and passed through a cross-modality alignment module to produce unified representations, which are then processed by an LLM and projected for final forecasting. Extensive experiments across eight diverse datasets show that MAP4TS consistently outperforms state-of-the-art LLM-based methods. Our ablation studies further reveal that prompt-aware designs significantly enhance performance stability and that GPT-2 backbones, when paired with structured prompts, outperform larger models like LLaMA in long-term forecasting tasks.
>
---
#### [new 057] Mubeen AI: A Specialized Arabic Language Model for Heritage Preservation and User Intent Understanding
- **分类: cs.CL; 68T50 (68T50 Natural language processing); I.2.7; I.2.6; I.2.0; H.3.3**

- **简介: 该论文提出Mubeen AI，一个专用于阿拉伯语文化遗产保护与用户意图理解的语言模型。针对现有模型依赖翻译数据导致意图识别不准的问题，基于本土阿拉伯语资料训练，并引入实用闭合架构，解决“效用缺口”问题，提升响应准确性与实用性，支持古典、现代及方言文本，助力沙特2030愿景。**

- **链接: [http://arxiv.org/pdf/2510.23271v1](http://arxiv.org/pdf/2510.23271v1)**

> **作者:** Mohammed Aljafari; Ismail Alturki; Ahmed Mori; Yehya Kadumi
>
> **备注:** 21 pages, 2 figures, 3 tables. Includes appendices on ethical guidelines and training framework. Submitted September 04, 2025
>
> **摘要:** Mubeen is a proprietary Arabic language model developed by MASARAT SA, optimized for deep understanding of Arabic linguistics, Islamic studies, and cultural heritage. Trained on an extensive collection of authentic Arabic sources significantly expanded by digitizing historical manuscripts via a proprietary Arabic OCR engine, the model incorporates seminal scholarly works in linguistics, jurisprudence, hadith, and Quranic exegesis, alongside thousands of academic theses and peer-reviewed research papers. Conditioned through a deep linguistic engineering framework, Mubeen masters not just the meaning but the eloquence of Arabic, enabling precise understanding across classical texts, contemporary writing, and regional dialects with focus on comprehending user intent and delivering accurate, contextually relevant responses. Unlike other Arabic models relying on translated English data that often fail in intent detection or retrieval-augmented generation (RAG), Mubeen uses native Arabic sources to ensure cultural authenticity and accuracy. Its core innovation is the Practical Closure Architecture, designed to solve the "Utility Gap Crisis" where factually correct answers fail to resolve users' core needs, forcing them into frustrating cycles of re-prompting. By prioritizing clarity and decisive guidance, Mubeen transforms from an information repository into a decisive guide, aligning with Saudi Vision 2030. The model's architecture combines deep heritage specialization with multi-disciplinary expert modules, enabling robust performance across both cultural preservation and general knowledge domains.
>
---
#### [new 058] Multilingual Target-Stance Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出首个多语言目标立场抽取（Multilingual TSE）基准，涵盖六种语言。针对现有TSE仅限英语的问题，构建跨语言统一模型，实现多语言场景下的目标识别与立场分类。实验表明目标预测是主要瓶颈，且F1受目标表述方式影响显著，为多语言立场分析提供基础资源与评估标准。**

- **链接: [http://arxiv.org/pdf/2510.22334v1](http://arxiv.org/pdf/2510.22334v1)**

> **作者:** Ethan Mines; Bonnie Dorr
>
> **备注:** 11 pages, 2 figures, Submitted to the Fifteenth Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** Social media enables data-driven analysis of public opinion on contested issues. Target-Stance Extraction (TSE) is the task of identifying the target discussed in a document and the document's stance towards that target. Many works classify stance towards a given target in a multilingual setting, but all prior work in TSE is English-only. This work introduces the first multilingual TSE benchmark, spanning Catalan, Estonian, French, Italian, Mandarin, and Spanish corpora. It manages to extend the original TSE pipeline to a multilingual setting without requiring separate models for each language. Our model pipeline achieves a modest F1 score of 12.78, underscoring the increased difficulty of the multilingual task relative to English-only setups and highlighting target prediction as the primary bottleneck. We are also the first to demonstrate the sensitivity of TSE's F1 score to different target verbalizations. Together these serve as a much-needed baseline for resources, algorithms, and evaluation criteria in multilingual TSE.
>
---
#### [new 059] Cross-Lingual Stability and Bias in Instruction-Tuned Language Models for Humanitarian NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦多语言人权侵害检测任务，解决资源受限组织在商业API与开源模型间的选择难题。通过对比78,000次推理，评估六种模型的跨语言稳定性，发现指令对齐显著提升低资源语言下的可靠性，为开源模型的可信部署提供实证依据。**

- **链接: [http://arxiv.org/pdf/2510.22823v1](http://arxiv.org/pdf/2510.22823v1)**

> **作者:** Poli Nemkova; Amrit Adhikari; Matthew Pearson; Vamsi Krishna Sadu; Mark V. Albert
>
> **摘要:** Humanitarian organizations face a critical choice: invest in costly commercial APIs or rely on free open-weight models for multilingual human rights monitoring. While commercial systems offer reliability, open-weight alternatives lack empirical validation -- especially for low-resource languages common in conflict zones. This paper presents the first systematic comparison of commercial and open-weight large language models (LLMs) for human-rights-violation detection across seven languages, quantifying the cost-reliability trade-off facing resource-constrained organizations. Across 78,000 multilingual inferences, we evaluate six models -- four instruction-aligned (Claude-Sonnet-4, DeepSeek-V3, Gemini-Flash-2.0, GPT-4.1-mini) and two open-weight (LLaMA-3-8B, Mistral-7B) -- using both standard classification metrics and new measures of cross-lingual reliability: Calibration Deviation (CD), Decision Bias (B), Language Robustness Score (LRS), and Language Stability Score (LSS). Results show that alignment, not scale, determines stability: aligned models maintain near-invariant accuracy and balanced calibration across typologically distant and low-resource languages (e.g., Lingala, Burmese), while open-weight models exhibit significant prompt-language sensitivity and calibration drift. These findings demonstrate that multilingual alignment enables language-agnostic reasoning and provide practical guidance for humanitarian organizations balancing budget constraints with reliability in multilingual deployment.
>
---
#### [new 060] Conjugate Relation Modeling for Few-Shot Knowledge Graph Completion
- **分类: cs.CL**

- **简介: 该论文针对少样本知识图谱补全（FKGC）任务，解决关系模式复杂与数据稀疏问题。提出CR-FKGC框架，通过邻域聚合编码、共轭关系学习与流形解码，有效捕捉稳定语义与不确定性，提升补全性能。**

- **链接: [http://arxiv.org/pdf/2510.22656v1](http://arxiv.org/pdf/2510.22656v1)**

> **作者:** Zilong Wang; Qingtian Zeng; Hua Duan; Cheng Cheng; Minghao Zou; Ziyang Wang
>
> **摘要:** Few-shot Knowledge Graph Completion (FKGC) infers missing triples from limited support samples, tackling long-tail distribution challenges. Existing methods, however, struggle to capture complex relational patterns and mitigate data sparsity. To address these challenges, we propose a novel FKGC framework for conjugate relation modeling (CR-FKGC). Specifically, it employs a neighborhood aggregation encoder to integrate higher-order neighbor information, a conjugate relation learner combining an implicit conditional diffusion relation module with a stable relation module to capture stable semantics and uncertainty offsets, and a manifold conjugate decoder for efficient evaluation and inference of missing triples in manifold space. Experiments on three benchmarks demonstrate that our method achieves superior performance over state-of-the-art methods.
>
---
#### [new 061] $\text{E}^2\text{Rank}$: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出E²Rank框架，将文本嵌入模型统一用于检索与列表级重排序。针对嵌入模型重排序精度不足的问题，通过在列表级目标下持续训练，利用文档信息增强查询表示，提升排名效果，同时保持高效性，在BEIR和BRIGHT上达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2510.22733v1](http://arxiv.org/pdf/2510.22733v1)**

> **作者:** Qi Liu; Yanzhao Zhang; Mingxin Li; Dingkun Long; Pengjun Xie; Jiaxin Mao
>
> **备注:** Code and models are avaliable at https://alibaba-nlp.github.io/E2Rank
>
> **摘要:** Text embedding models serve as a fundamental component in real-world search applications. By mapping queries and documents into a shared embedding space, they deliver competitive retrieval performance with high efficiency. However, their ranking fidelity remains limited compared to dedicated rerankers, especially recent LLM-based listwise rerankers, which capture fine-grained query-document and document-document interactions. In this paper, we propose a simple yet effective unified framework $\text{E}^2\text{Rank}$, means Efficient Embedding-based Ranking (also means Embedding-to-Rank), which extends a single text embedding model to perform both high-quality retrieval and listwise reranking through continued training under a listwise ranking objective, thereby achieving strong effectiveness with remarkable efficiency. By applying cosine similarity between the query and document embeddings as a unified ranking function, the listwise ranking prompt, which is constructed from the original query and its candidate documents, serves as an enhanced query enriched with signals from the top-K documents, akin to pseudo-relevance feedback (PRF) in traditional retrieval models. This design preserves the efficiency and representational quality of the base embedding model while significantly improving its reranking performance. Empirically, $\textrm{E}^2\text{Rank}$ achieves state-of-the-art results on the BEIR reranking benchmark and demonstrates competitive performance on the reasoning-intensive BRIGHT benchmark, with very low reranking latency. We also show that the ranking training process improves embedding performance on the MTEB benchmark. Our findings indicate that a single embedding model can effectively unify retrieval and reranking, offering both computational efficiency and competitive ranking accuracy.
>
---
#### [new 062] LangLingual: A Personalised, Exercise-oriented English Language Learning Tool Leveraging Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出LangLingual，一个基于大语言模型的个性化英语学习工具，旨在解决教师反馈与练习资源有限的问题。系统通过对话式交互提供实时语法反馈、生成情境化练习并追踪学习进度，有效提升学习者参与度与成效。**

- **链接: [http://arxiv.org/pdf/2510.23011v1](http://arxiv.org/pdf/2510.23011v1)**

> **作者:** Sammriddh Gupta; Sonit Singh; Aditya Joshi; Mira Kim
>
> **备注:** 14 pages
>
> **摘要:** Language educators strive to create a rich experience for learners, while they may be restricted in the extend of feedback and practice they can provide. We present the design and development of LangLingual, a conversational agent built using the LangChain framework and powered by Large Language Models. The system is specifically designed to provide real-time, grammar-focused feedback, generate context-aware language exercises and track learner proficiency over time. The paper discusses the architecture, implementation and evaluation of LangLingual in detail. The results indicate strong usability, positive learning outcomes and encouraging learner engagement.
>
---
#### [new 063] BaZi-Based Character Simulation Benchmark: Evaluating AI on Temporal and Persona Reasoning
- **分类: cs.CL**

- **简介: 该论文针对虚拟角色生成中人格与时间动态性不足的问题，提出基于八字（BaZi）的符号推理框架。构建首个面向八字人格推理的问答数据集，设计融合符号逻辑与大模型的系统，实现更真实、连贯的角色模拟，并在准确性上显著优于主流模型。**

- **链接: [http://arxiv.org/pdf/2510.23337v1](http://arxiv.org/pdf/2510.23337v1)**

> **作者:** Siyuan Zheng; Pai Liu; Xi Chen; Jizheng Dong; Sihan Jia
>
> **摘要:** Human-like virtual characters are crucial for games, storytelling, and virtual reality, yet current methods rely heavily on annotated data or handcrafted persona prompts, making it difficult to scale up and generate realistic, contextually coherent personas. We create the first QA dataset for BaZi-based persona reasoning, where real human experiences categorized into wealth, health, kinship, career, and relationships are represented as life-event questions and answers. Furthermore, we propose the first BaZi-LLM system that integrates symbolic reasoning with large language models to generate temporally dynamic and fine-grained virtual personas. Compared with mainstream LLMs such as DeepSeek-v3 and GPT-5-mini, our method achieves a 30.3%-62.6% accuracy improvement. In addition, when incorrect BaZi information is used, our model's accuracy drops by 20%-45%, showing the potential of culturally grounded symbolic-LLM integration for realistic character simulation.
>
---
#### [new 064] LimRank: Less is More for Reasoning-Intensive Information Reranking
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对信息重排序任务，提出LimRank方法，通过少量高质量监督数据实现高效微调。设计开源合成数据生成管道LIMRANK-SYNTHESIZER，生成多样挑战性样本，使模型在仅用5%数据下达到竞争力表现，显著降低计算成本并展现强泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.23544v1](http://arxiv.org/pdf/2510.23544v1)**

> **作者:** Tingyu Song; Yilun Zhao; Siyue Zhang; Chen Zhao; Arman Cohan
>
> **备注:** EMNLP 2025 Main (Short)
>
> **摘要:** Existing approaches typically rely on large-scale fine-tuning to adapt LLMs for information reranking tasks, which is computationally expensive. In this work, we demonstrate that modern LLMs can be effectively adapted using only minimal, high-quality supervision. To enable this, we design LIMRANK-SYNTHESIZER, a reusable and open-source pipeline for generating diverse, challenging, and realistic reranking examples. Using this synthetic data, we fine-tune our reranker model, LIMRANK. We evaluate LIMRANK on two challenging benchmarks, i.e., BRIGHT for reasoning-intensive retrieval and FollowIR for instruction-following retrieval. Our experiments demonstrate that LIMRANK achieves competitive performance, while being trained on less than 5% of the data typically used in prior work. Further ablation studies demonstrate the effectiveness of LIMRANK-SYNTHESIZER and the strong generalization capabilities of LIMRANK across downstream tasks, including scientific literature search and retrieval-augmented generation for knowledge-intensive problem solving.
>
---
#### [new 065] Leveraging Large Language Models to Identify Conversation Threads in Collaborative Learning
- **分类: cs.CL**

- **简介: 该论文研究同步多人群体对话中的对话线程识别任务，旨在解决语音对话中因话语重叠和隐含线索导致的线程检测难题。通过构建线程识别指南与对比不同大模型提示策略，验证了明确线程结构能显著提升下游协作行为编码性能，提出人机协同优化方案。**

- **链接: [http://arxiv.org/pdf/2510.22844v1](http://arxiv.org/pdf/2510.22844v1)**

> **作者:** Prerna Ravi; Dong Won Lee; Beatriz Flamia; Jasmine David; Brandon Hanks; Cynthia Breazeal; Emma Anderson; Grace Lin
>
> **备注:** In Submission: Journal of Educational Data Mining (jEDM) 2026
>
> **摘要:** Understanding how ideas develop and flow in small-group conversations is critical for analyzing collaborative learning. A key structural feature of these interactions is threading, the way discourse talk naturally organizes into interwoven topical strands that evolve over time. While threading has been widely studied in asynchronous text settings, detecting threads in synchronous spoken dialogue remains challenging due to overlapping turns and implicit cues. At the same time, large language models (LLMs) show promise for automating discourse analysis but often struggle with long-context tasks that depend on tracing these conversational links. In this paper, we investigate whether explicit thread linkages can improve LLM-based coding of relational moves in group talk. We contribute a systematic guidebook for identifying threads in synchronous multi-party transcripts and benchmark different LLM prompting strategies for automated threading. We then test how threading influences performance on downstream coding of conversational analysis frameworks, that capture core collaborative actions such as agreeing, building, and eliciting. Our results show that providing clear conversational thread information improves LLM coding performance and underscores the heavy reliance of downstream analysis on well-structured dialogue. We also discuss practical trade-offs in time and cost, emphasizing where human-AI hybrid approaches can yield the best value. Together, this work advances methods for combining LLMs and robust conversational thread structures to make sense of complex, real-time group interactions.
>
---
#### [new 066] Policy Optimization Prefers The Path of Least Resistance
- **分类: cs.CL**

- **简介: 该论文研究政策优化在开放式思维链中的行为，揭示其倾向于选择最简路径。针对复杂推理任务中奖励机制易被“捷径”利用的问题，通过控制实验发现PO算法会退化为仅输出答案的格式，即使高权重奖励也无法阻止。核心工作是验证“最小阻力路径”原则，并指出自由度与对齐风险间的权衡。**

- **链接: [http://arxiv.org/pdf/2510.21853v1](http://arxiv.org/pdf/2510.21853v1)**

> **作者:** Debdeep Sanyal; Aakash Sen Sharma; Dhruv Kumar; Saurabh Deshpande; Murari Mandal
>
> **备注:** 21 pages, 8 figures, 2 tables
>
> **摘要:** Policy optimization (PO) algorithms are used to refine Large Language Models for complex, multi-step reasoning. Current state-of-the-art pipelines enforce a strict think-then-answer format to elicit chain-of-thought (CoT); however, the behavior of PO when these rigid constraints are relaxed into an open-ended CoT structure remains an under-studied question. We investigate this gap with an extensive suite of controlled experiments and identify a consistent principle: \textit{policy optimization consistently follows the path of least resistance}. When afforded the flexibility to interleave reasoning and response, policy optimization consistently learns to discard explicit reasoning, causing the policy to degenerate to a direct \texttt{<answer>}-only format. This outcome holds true across various models and algorithms. We find that this collapse in format is persistent even when the complex \texttt{<think><answer>} format is assigned up to 4x larger reward weights. We formalize this principle through a series of controlled reward decomposition experiments, demonstrating a clear hierarchy: PO systematically optimizes for the simplest reward component first, a preference that holds even when faced with mutually exclusive choices or strong incentives for more complex behaviors. Finally, we show that successful convergence on the high-reward shortcut is not a low-effort drift but is driven by the optimization process that requires the KL-regularized policy to have sufficient freedom to make a significant shift from its initial prior. Our findings reveal that granting policies the freedom to diverge is a double-edged sword: while necessary for discovering high-reward shortcuts, it also creates a powerful incentive to game the simplest aspects of the reward function, posing a critical challenge for reward hacking under alignment.
>
---
#### [new 067] Confabulations from ACL Publications (CAP): A Dataset for Scientific Hallucination Detection
- **分类: cs.CL**

- **简介: 该论文提出CAP数据集，用于科学领域大语言模型幻觉检测。针对科学文本中因术语、统计推理等导致的虚假信息问题，构建了多语言（五高四低）的900个问题与7000+答案对，含事实性与流畅性标注，助力幻觉检测与多语言模型评估。**

- **链接: [http://arxiv.org/pdf/2510.22395v1](http://arxiv.org/pdf/2510.22395v1)**

> **作者:** Federica Gamba; Aman Sinha; Timothee Mickus; Raul Vazquez; Patanjali Bhamidipati; Claudio Savelli; Ahana Chattopadhyay; Laura A. Zanella; Yash Kankanampati; Binesh Arakkal Remesh; Aryan Ashok Chandramania; Rohit Agarwal; Chuyuan Li; Ioana Buhnila; Radhika Mamidi
>
> **摘要:** We introduce the CAP (Confabulations from ACL Publications) dataset, a multilingual resource for studying hallucinations in large language models (LLMs) within scientific text generation. CAP focuses on the scientific domain, where hallucinations can distort factual knowledge, as they frequently do. In this domain, however, the presence of specialized terminology, statistical reasoning, and context-dependent interpretations further exacerbates these distortions, particularly given LLMs' lack of true comprehension, limited contextual understanding, and bias toward surface-level generalization. CAP operates in a cross-lingual setting covering five high-resource languages (English, French, Hindi, Italian, and Spanish) and four low-resource languages (Bengali, Gujarati, Malayalam, and Telugu). The dataset comprises 900 curated scientific questions and over 7000 LLM-generated answers from 16 publicly available models, provided as question-answer pairs along with token sequences and corresponding logits. Each instance is annotated with a binary label indicating the presence of a scientific hallucination, denoted as a factuality error, and a fluency label, capturing issues in the linguistic quality or naturalness of the text. CAP is publicly released to facilitate advanced research on hallucination detection, multilingual evaluation of LLMs, and the development of more reliable scientific NLP systems.
>
---
#### [new 068] Flexing in 73 Languages: A Single Small Model for Multilingual Inflection
- **分类: cs.CL**

- **简介: 该论文聚焦多语言词形变化任务，提出一个统一小模型，联合训练73种语言数据，实现轻量级、鲁棒的词形生成。解决传统单语言模型部署复杂及对未见词处理差的问题，通过频率加权采样提升评估真实性，代码开源，支持跨语言泛化。**

- **链接: [http://arxiv.org/pdf/2510.23114v1](http://arxiv.org/pdf/2510.23114v1)**

> **作者:** Tomáš Sourada; Jana Straková
>
> **备注:** Published in the proceedings of TSD 2025. 12 pages, 1 figure, 4 tables
>
> **摘要:** We present a compact, single-model approach to multilingual inflection, the task of generating inflected word forms from base lemmas to express grammatical categories. Our model, trained jointly on data from 73 languages, is lightweight, robust to unseen words, and outperforms monolingual baselines in most languages. This demonstrates the effectiveness of multilingual modeling for inflection and highlights its practical benefits: simplifying deployment by eliminating the need to manage and retrain dozens of separate monolingual models. In addition to the standard SIGMORPHON shared task benchmarks, we evaluate our monolingual and multilingual models on 73 Universal Dependencies (UD) treebanks, extracting lemma-tag-form triples and their frequency counts. To ensure realistic data splits, we introduce a novel frequency-weighted, lemma-disjoint train-dev-test resampling procedure. Our work addresses the lack of an open-source, general-purpose, multilingual morphological inflection system capable of handling unseen words across a wide range of languages, including Czech. All code is publicly released at: https://github.com/tomsouri/multilingual-inflection.
>
---
#### [new 069] Evaluating Large Language Models for Stance Detection on Financial Targets from SEC Filing Reports and Earnings Call Transcripts
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦金融领域句子级立场检测任务，针对债务、EPS、销售额三类财务指标，构建了基于SEC报告和财报电话会的标注语料。利用ChatGPT-o3-pro与人工验证实现高质量标注，系统评估LLMs在零样本、少样本及思维链提示下的表现，证明少样本+思维链策略优于监督基线，展现了无需大量标注数据即可实现精准立场识别的可行性。**

- **链接: [http://arxiv.org/pdf/2510.23464v1](http://arxiv.org/pdf/2510.23464v1)**

> **作者:** Nikesh Gyawali; Doina Caragea; Alex Vasenkov; Cornelia Caragea
>
> **摘要:** Financial narratives from U.S. Securities and Exchange Commission (SEC) filing reports and quarterly earnings call transcripts (ECTs) are very important for investors, auditors, and regulators. However, their length, financial jargon, and nuanced language make fine-grained analysis difficult. Prior sentiment analysis in the financial domain required a large, expensive labeled dataset, making the sentence-level stance towards specific financial targets challenging. In this work, we introduce a sentence-level corpus for stance detection focused on three core financial metrics: debt, earnings per share (EPS), and sales. The sentences were extracted from Form 10-K annual reports and ECTs, and labeled for stance (positive, negative, neutral) using the advanced ChatGPT-o3-pro model under rigorous human validation. Using this corpus, we conduct a systematic evaluation of modern large language models (LLMs) using zero-shot, few-shot, and Chain-of-Thought (CoT) prompting strategies. Our results show that few-shot with CoT prompting performs best compared to supervised baselines, and LLMs' performance varies across the SEC and ECT datasets. Our findings highlight the practical viability of leveraging LLMs for target-specific stance in the financial domain without requiring extensive labeled data.
>
---
#### [new 070] OlaMind: Towards Human-Like and Hallucination-Safe Customer Service for Retrieval-Augmented Dialogue
- **分类: cs.CL**

- **简介: 该论文针对检索增强对话中的幻觉与响应机械问题，提出OlaMind框架。通过学习人类专家的推理与应答策略，结合冷启动微调与强化学习实现自优化，显著提升响应自然性与安全性，在真实场景中大幅提高智能解决率并降低人工介入率。**

- **链接: [http://arxiv.org/pdf/2510.22143v1](http://arxiv.org/pdf/2510.22143v1)**

> **作者:** Tianhong Gao; Jundong Shen; Bei Shi; Jiapeng Wang; Ying Ju; Junfeng Yao; Jiao Ran; Yong Zhang; Lin Dong; Huiyu Yu; Tingting Ye
>
> **摘要:** Intelligent customer service (ICS) systems via retrieval-augmented generation (RAG) have been widely adopted in Web-based domains such as social platforms and e-commerce, achieving remarkable improvements in automation and efficiency. However, notable limitations still remain: these systems are prone to hallucinations and often generate rigid, mechanical responses, which can introduce business risks and undermine user experience, especially in Web-based customer service interactions under the RAG scenarios. In this paper, we introduce OlaMind, a human-like and hallucination-safe customer service framework for retrieval-augmented dialogue. Specifically, it first leverages a Learn-to-Think stage to learn the reasoning processes and response strategies from human experts, and then employs a Learn-to-Respond stage to perform cold-start supervised fine-tuning (SFT) combined with reinforcement learning (RL) for basic-to-hard self-refinement. Our method significantly enhances human-likeness and naturalness while effectively mitigating hallucinations and critical business risks. We have conducted large-scale online A/B experiments in an industry-level social customer service setting, and extensive experimental results show that OlaMind achieves significant cumulative relative improvements with intelligent resolution rates +28.92%/+18.42% and human takeover rate -6.08%/-7.12% in community-support/livestream-interaction scenarios, respectively, which highlights its consistent effectiveness across diverse real-world applications. The code and data will be publicly available.
>
---
#### [new 071] The Tonogenesis Continuum in Tibetan: A Computational Investigation
- **分类: cs.CL**

- **简介: 该论文研究藏语声调演化过程，提出用计算方法评估音高在语音识别中的功能变化。通过分析不同藏语方言对音高平直化的敏感度，揭示声调演化连续体，表明传统功能负荷指标可能高估过渡阶段的音高依赖性。**

- **链接: [http://arxiv.org/pdf/2510.22485v1](http://arxiv.org/pdf/2510.22485v1)**

> **作者:** Siyu Liang; Zhaxi Zerong
>
> **摘要:** Tonogenesis-the historical process by which segmental contrasts evolve into lexical tone-has traditionally been studied through comparative reconstruction and acoustic phonetics. We introduce a computational approach that quantifies the functional role of pitch at different stages of this sound change by measuring how pitch manipulation affects automatic speech recognition (ASR) performance. Through analysis on the sensitivity to pitch-flattening from a set of closely related Tibetan languages, we find evidence of a tonogenesis continuum: atonal Amdo dialects tolerate pitch removal the most, while fully tonal U-Tsang varieties show severe degradation, and intermediate Kham dialects fall measurably between these extremes. These gradient effects demonstrate how ASR models implicitly learn the shifting functional load of pitch as languages transition from consonant-based to tone-based lexical contrasts. Our findings show that computational methods can capture fine-grained stages of sound change and suggest that traditional functional load metrics, based solely on minimal pairs, may overestimate pitch dependence in transitional systems where segmental and suprasegmental cues remain phonetically intertwined.
>
---
#### [new 072] Deep Literature Survey Automation with an Iterative Workflow
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对自动文献综述生成任务，解决现有方法因一次性检索与静态结构导致的噪声多、结构散乱问题。提出基于循环大纲生成的框架\ours，通过迭代式阅读与更新，结合论文卡片与可视化反馈，提升内容覆盖与结构连贯性。引入Survey-Arena基准以更可靠评估生成质量。**

- **链接: [http://arxiv.org/pdf/2510.21900v1](http://arxiv.org/pdf/2510.21900v1)**

> **作者:** Hongbo Zhang; Han Cui; Yidong Wang; Yijian Tian; Qi Guo; Cunxiang Wang; Jian Wu; Chiyu Song; Yue Zhang
>
> **备注:** Preprint version
>
> **摘要:** Automatic literature survey generation has attracted increasing attention, yet most existing systems follow a one-shot paradigm, where a large set of papers is retrieved at once and a static outline is generated before drafting. This design often leads to noisy retrieval, fragmented structures, and context overload, ultimately limiting survey quality. Inspired by the iterative reading process of human researchers, we propose \ours, a framework based on recurrent outline generation, in which a planning agent incrementally retrieves, reads, and updates the outline to ensure both exploration and coherence. To provide faithful paper-level grounding, we design paper cards that distill each paper into its contributions, methods, and findings, and introduce a review-and-refine loop with visualization enhancement to improve textual flow and integrate multimodal elements such as figures and tables. Experiments on both established and emerging topics show that \ours\ substantially outperforms state-of-the-art baselines in content coverage, structural coherence, and citation quality, while producing more accessible and better-organized surveys. To provide a more reliable assessment of such improvements, we further introduce Survey-Arena, a pairwise benchmark that complements absolute scoring and more clearly positions machine-generated surveys relative to human-written ones. The code is available at https://github.com/HancCui/IterSurvey\_Autosurveyv2.
>
---
#### [new 073] Batch Speculative Decoding Done Right
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型推理中的批处理推测解码问题，解决因序列长度不一导致的乱序张量问题。提出EQSPEC与EXSPEC方法，确保输出等价性并降低重对齐开销，实现高达3倍吞吐提升，无需自定义内核，可无缝集成现有推理系统。**

- **链接: [http://arxiv.org/pdf/2510.22876v1](http://arxiv.org/pdf/2510.22876v1)**

> **作者:** Ranran Haoran Zhang; Soumik Dey; Ashirbad Mishra; Hansi Wu; Binbin Li; Rui Zhang
>
> **摘要:** Speculative decoding speeds up LLM inference by using a small draft model to propose multiple tokens that a target model verifies in parallel. Extending this idea to batches is essential for production serving, but it introduces the ragged tensor problem: sequences in the same batch accept different numbers of draft tokens, breaking right-alignment and corrupting position IDs, attention masks, and KV-cache state. We show that several existing batch implementations violate output equivalence-the fundamental requirement that speculative decoding must produce identical token sequences to standard autoregressive generation. These violations occur precisely due to improper handling of the ragged tensor problem. In response, we (1) characterize the synchronization requirements that guarantee correctness, (2) present a correctness-first batch speculative decoding EQSPEC that exposes realignment as consuming 40% of overhead, and (3) introduce EXSPEC, which maintains a sliding pool of sequences and dynamically forms same-length groups, to reduce the realignment overhead while preserving per-sequence speculative speedups. On the SpecBench dataset, across Vicuna-7B/68M, Qwen3-8B/0.6B, and GLM-4-9B/0.6B target/draft pairs, our approach achieves up to 3$\times$ throughput improvement at batch size 8 compared to batch size 1, with efficient scaling through batch size 8, while maintaining 95% output equivalence. Our method requires no custom kernels and integrates cleanly with existing inference stacks. Our code is available at https://github.com/eBay/spec_dec.
>
---
#### [new 074] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文聚焦于脑神经活动建模任务，旨在分离语言模型中混杂的语义与推理表征。针对传统方法因特征纠缠导致深层推理信号被掩盖的问题，提出残差解耦方法，提取独立的词法、句法、语义和推理嵌入。利用颅内脑电数据验证，发现推理嵌入具有独特时序特征与跨区域预测能力，揭示了高级认知在语言处理中的神经基础。**

- **链接: [http://arxiv.org/pdf/2510.22860v1](http://arxiv.org/pdf/2510.22860v1)**

> **作者:** Linyang He; Tianjun Zhong; Richard Antonello; Gavin Mischler; Micah Goldblum; Nima Mesgarani
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Understanding how the human brain progresses from processing simple linguistic inputs to performing high-level reasoning is a fundamental challenge in neuroscience. While modern large language models (LLMs) are increasingly used to model neural responses to language, their internal representations are highly "entangled," mixing information about lexicon, syntax, meaning, and reasoning. This entanglement biases conventional brain encoding analyses toward linguistically shallow features (e.g., lexicon and syntax), making it difficult to isolate the neural substrates of cognitively deeper processes. Here, we introduce a residual disentanglement method that computationally isolates these components. By first probing an LM to identify feature-specific layers, our method iteratively regresses out lower-level representations to produce four nearly orthogonal embeddings for lexicon, syntax, meaning, and, critically, reasoning. We used these disentangled embeddings to model intracranial (ECoG) brain recordings from neurosurgical patients listening to natural speech. We show that: 1) This isolated reasoning embedding exhibits unique predictive power, accounting for variance in neural activity not explained by other linguistic features and even extending to the recruitment of visual regions beyond classical language areas. 2) The neural signature for reasoning is temporally distinct, peaking later (~350-400ms) than signals related to lexicon, syntax, and meaning, consistent with its position atop a processing hierarchy. 3) Standard, non-disentangled LLM embeddings can be misleading, as their predictive success is primarily attributable to linguistically shallow features, masking the more subtle contributions of deeper cognitive processing.
>
---
#### [new 075] Understanding In-Context Learning Beyond Transformers: An Investigation of State Space and Hybrid Architectures
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究不同架构（Transformer、状态空间、混合模型）在上下文学习（ICL）中的表现差异。通过行为探测与干预分析，发现尽管任务表现相似，内部机制不同；功能向量（FVs）主要存在于自注意力和Mamba层，且对参数化知识检索更关键。工作揭示了架构与任务类型对ICL机制的影响，强调结合行为与机制分析的重要性。**

- **链接: [http://arxiv.org/pdf/2510.23006v1](http://arxiv.org/pdf/2510.23006v1)**

> **作者:** Shenran Wang; Timothy Tin-Long Tse; Jian Zhu
>
> **摘要:** We perform in-depth evaluations of in-context learning (ICL) on state-of-the-art transformer, state-space, and hybrid large language models over two categories of knowledge-based ICL tasks. Using a combination of behavioral probing and intervention-based methods, we have discovered that, while LLMs of different architectures can behave similarly in task performance, their internals could remain different. We discover that function vectors (FVs) responsible for ICL are primarily located in the self-attention and Mamba layers, and speculate that Mamba2 uses a different mechanism from FVs to perform ICL. FVs are more important for ICL involving parametric knowledge retrieval, but not for contextual knowledge understanding. Our work contributes to a more nuanced understanding across architectures and task types. Methodologically, our approach also highlights the importance of combining both behavioural and mechanistic analyses to investigate LLM capabilities.
>
---
#### [new 076] DREaM: Drug-Drug Relation Extraction via Transfer Learning Method
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦药物间关系抽取任务，旨在解决医疗领域缺乏专用数据集导致的模型训练困难问题。提出DREAM方法，结合预训练模型与迁移学习，在医学文本中提取药物关系，并用大语言模型验证结果，有效提升了关系抽取的效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.23189v1](http://arxiv.org/pdf/2510.23189v1)**

> **作者:** Ali Fata; Hossein Rahmani; Parinaz Soltanzadeh; Amirhossein Derakhshan; Behrouz Minaei Bidgoli
>
> **摘要:** Relation extraction between drugs plays a crucial role in identifying drug drug interactions and predicting side effects. The advancement of machine learning methods in relation extraction, along with the development of large medical text databases, has enabled the low cost extraction of such relations compared to other approaches that typically require expert knowledge. However, to the best of our knowledge, there are limited datasets specifically designed for drug drug relation extraction currently available. Therefore, employing transfer learning becomes necessary to apply machine learning methods in this domain. In this study, we propose DREAM, a method that first employs a trained relation extraction model to discover relations between entities and then applies this model to a corpus of medical texts to construct an ontology of drug relationships. The extracted relations are subsequently validated using a large language model. Quantitative results indicate that the LLM agreed with 71 of the relations extracted from a subset of PubMed abstracts. Furthermore, our qualitative analysis indicates that this approach can uncover ambiguities in the medical domain, highlighting the challenges inherent in relation extraction in this field.
>
---
#### [new 077] Language Ranker: A Lightweight Ranking framework for LLM Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Language Ranker，将语言模型解码视为推荐系统中的排序任务，通过轻量级模块基于基础模型特征重排候选响应。旨在解决传统解码方法与奖励模型计算开销大、效率低的问题，实现高性能且低参数消耗的生成优化。**

- **链接: [http://arxiv.org/pdf/2510.21883v1](http://arxiv.org/pdf/2510.21883v1)**

> **作者:** Chenheng Zhang; Tianqi Du; Jizhe Zhang; Mingqing Xiao; Yifei Wang; Yisen Wang; Zhouchen Lin
>
> **摘要:** Conventional research on large language models (LLMs) has primarily focused on refining output distributions, while paying less attention to the decoding process that transforms these distributions into final responses. Recent advances, such as scaling the computation of inference time with reward models, have underscored the importance of decoding, but these methods often suffer from high computational costs and limited applicability. In this paper, we revisit LLM generation through the lens of recommender systems, conceptualizing the decoding process as analogous to the ranking stage in recommendation pipelines. From this perspective, we observe that both traditional decoding methods and reward models exhibit clear limitations such as redundancy. Motivated by this insight, we propose Language Ranker, a novel framework that introduces a lightweight module to rerank candidate responses using features extracted by the base model. Experiments across a wide range of tasks show that Language Ranker achieves performance comparable to large-scale reward models, while requiring only <0.5M additional parameters, significantly reducing the computational overhead during both training and inference stages. This highlights the efficiency and effectiveness of our method, showcasing its potential to fully unlock the capabilities of LLMs.
>
---
#### [new 078] Measuring Teaching with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对教育中教学质量测量难的问题，提出基于句向量的定制化LLM方法，通过数据高效训练实现对课堂实录的精准评分。模型达人类甚至超人类水平，且得分与教师增值效应相关，验证了其外部效度。研究挑战了单轮注释范式，揭示了模型更关注整体课程特征。**

- **链接: [http://arxiv.org/pdf/2510.22968v1](http://arxiv.org/pdf/2510.22968v1)**

> **作者:** Michael Hardy
>
> **摘要:** Objective and scalable measurement of teaching quality is a persistent challenge in education. While Large Language Models (LLMs) offer potential, general-purpose models have struggled to reliably apply complex, authentic classroom observation instruments. This paper uses custom LLMs built on sentence-level embeddings, an architecture better suited for the long-form, interpretive nature of classroom transcripts than conventional subword tokenization. We systematically evaluate five different sentence embeddings under a data-efficient training regime designed to prevent overfitting. Our results demonstrate that these specialized models can achieve human-level and even super-human performance with expert human ratings above 0.65 and surpassing the average human-human rater correlation. Further, through analysis of annotation context windows, we find that more advanced models-those better aligned with human judgments-attribute a larger share of score variation to lesson-level features rather than isolated utterances, challenging the sufficiency of single-turn annotation paradigms. Finally, to assess external validity, we find that aggregate model scores align with teacher value-added measures, indicating they are capturing features relevant to student learning. However, this trend does not hold at the individual item level, suggesting that while the models learn useful signals, they have not yet achieved full generalization. This work establishes a viable and powerful new methodology for AI-driven instructional measurement, offering a path toward providing scalable, reliable, and valid feedback for educator development.
>
---
#### [new 079] Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long-Form Text Generation
- **分类: cs.CL; cs.AI; cs.LG; stat.ME; stat.ML**

- **简介: 该论文针对大模型生成长文本时的事实性评估难题，提出基于语义各向同性（semantic isotropy）的无监督信任度评估方法。通过分析文本嵌入在单位球面上的角分布，量化响应一致性，无需标注数据或微调，高效预测非事实性，适用于多种应用场景。**

- **链接: [http://arxiv.org/pdf/2510.21891v1](http://arxiv.org/pdf/2510.21891v1)**

> **作者:** Dhrupad Bhardwaj; Julia Kempe; Tim G. J. Rudner
>
> **摘要:** To deploy large language models (LLMs) in high-stakes application domains that require substantively accurate responses to open-ended prompts, we need reliable, computationally inexpensive methods that assess the trustworthiness of long-form responses generated by LLMs. However, existing approaches often rely on claim-by-claim fact-checking, which is computationally expensive and brittle in long-form responses to open-ended prompts. In this work, we introduce semantic isotropy -- the degree of uniformity across normalized text embeddings on the unit sphere -- and use it to assess the trustworthiness of long-form responses generated by LLMs. To do so, we generate several long-form responses, embed them, and estimate the level of semantic isotropy of these responses as the angular dispersion of the embeddings on the unit sphere. We find that higher semantic isotropy -- that is, greater embedding dispersion -- reliably signals lower factual consistency across samples. Our approach requires no labeled data, no fine-tuning, and no hyperparameter selection, and can be used with open- or closed-weight embedding models. Across multiple domains, our method consistently outperforms existing approaches in predicting nonfactuality in long-form responses using only a handful of samples -- offering a practical, low-cost approach for integrating trust assessment into real-world LLM workflows.
>
---
#### [new 080] EMTSF:Extraordinary Mixture of SOTA Models for Time Series Forecasting
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦时间序列预测任务，针对现有模型性能争议与数据特性挑战，提出EMTSF框架。通过融合xLSTM、PatchTST等多种SOTA模型，并基于Transformer的门控机制构建混合专家系统，显著提升预测精度，优于当前主流方法。**

- **链接: [http://arxiv.org/pdf/2510.23396v1](http://arxiv.org/pdf/2510.23396v1)**

> **作者:** Musleh Alharthi; Kaleel Mahmood; Sarosh Patel; Ausif Mahmood
>
> **摘要:** The immense success of the Transformer architecture in Natural Language Processing has led to its adoption in Time Se ries Forecasting (TSF), where superior performance has been shown. However, a recent important paper questioned their effectiveness by demonstrating that a simple single layer linear model outperforms Transformer-based models. This was soon shown to be not as valid, by a better transformer-based model termed PatchTST. More re cently, TimeLLM demonstrated even better results by repurposing a Large Language Model (LLM) for the TSF domain. Again, a follow up paper challenged this by demonstrating that removing the LLM component or replacing it with a basic attention layer in fact yields better performance. One of the challenges in forecasting is the fact that TSF data favors the more recent past, and is sometimes subject to unpredictable events. Based upon these recent insights in TSF, we propose a strong Mixture of Experts (MoE) framework. Our method combines the state-of-the-art (SOTA) models including xLSTM, en hanced Linear, PatchTST, and minGRU, among others. This set of complimentary and diverse models for TSF are integrated in a Trans former based MoE gating network. Our proposed model outperforms all existing TSF models on standard benchmarks, surpassing even the latest approaches based on MoE frameworks.
>
---
#### [new 081] Framework for Machine Evaluation of Reasoning Completeness in Large Language Models For Classification Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在分类任务中生成解释的完整性问题，提出RACE框架，通过对比模型解释与逻辑回归特征重要性，评估解释的忠实性。工作包括多粒度匹配技术分析四大数据集，发现正确预测更覆盖支持特征，错误预测则更多包含矛盾特征，揭示了模型解释中表面与灵活证据的混合使用及误导性线索的放大。**

- **链接: [http://arxiv.org/pdf/2510.21884v1](http://arxiv.org/pdf/2510.21884v1)**

> **作者:** Avinash Patil
>
> **备注:** 12 Pages, 12 Figures, 2 tables
>
> **摘要:** The growing adoption of machine learning (ML) in sensitive domains has heightened the demand for transparent and interpretable artificial intelligence. Large Language Models (LLMs) are increasingly capable of producing natural language explanations, yet it remains unclear whether these rationales faithfully capture the predictive signals that underlie decisions. This paper introduces RACE-Reasoning Alignment for Completeness of Explanations, a systematic framework to evaluate the alignment between LLM-generated explanations and interpretable feature importance scores derived from a logistic regression baseline. We analyze four widely used text classification datasets-WIKI ONTOLOGY, AG NEWS, IMDB, and GOEMOTIONS-and compare LLM rationales against top-ranked supporting and contradicting lexical features. To capture alignment at multiple levels of granularity, RACE implements token-aware, exact string, and edit-distance matching techniques. Empirical results reveal a consistent asymmetry: correct predictions exhibit higher coverage of supporting features, while incorrect predictions are associated with elevated coverage of contradicting features. Edit-distance matching further uncovers paraphrastic overlaps, boosting coverage while preserving this asymmetry. These findings demonstrate that LLM rationales combine both surface-level and flexible evidence reuse, yet can also amplify misleading cues in error cases. RACE provides new insights into the faithfulness of LLM explanations and establishes a quantitative basis for evaluating reasoning completeness in neural language models.
>
---
#### [new 082] Once Upon an Input: Reasoning via Per-Instance Program Synthesis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型在复杂推理任务中生成错误程序的问题，提出基于实例级程序合成的PIPS方法。通过结构反馈与动态置信度选择，提升推理准确率并减少错误程序生成，显著优于CoT和PoT，在多个基准上实现性能提升。**

- **链接: [http://arxiv.org/pdf/2510.22849v1](http://arxiv.org/pdf/2510.22849v1)**

> **作者:** Adam Stein; Neelay Velingker; Mayur Naik; Eric Wong
>
> **备注:** Accepted at NeurIPS 2025. 34 pages, 7 figures
>
> **摘要:** Large language models (LLMs) excel at zero-shot inference but continue to struggle with complex, multi-step reasoning. Recent methods that augment LLMs with intermediate reasoning steps such as Chain of Thought (CoT) and Program of Thought (PoT) improve performance but often produce undesirable solutions, especially in algorithmic domains. We introduce Per-Instance Program Synthesis (PIPS), a method that generates and refines programs at the instance-level using structural feedback without relying on task-specific guidance or explicit test cases. To further improve performance, PIPS incorporates a confidence metric that dynamically chooses between direct inference and program synthesis on a per-instance basis. Experiments across three frontier LLMs and 30 benchmarks including all tasks of Big Bench Extra Hard (BBEH), visual question answering tasks, relational reasoning tasks, and mathematical reasoning tasks show that PIPS improves the absolute harmonic mean accuracy by up to 8.6% and 9.4% compared to PoT and CoT respectively, and reduces undesirable program generations by 65.1% on the algorithmic tasks compared to PoT with Gemini-2.0-Flash.
>
---
#### [new 083] M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset
- **分类: cs.CL**

- **简介: 该论文提出M4FC数据集，解决现有事实核查数据集规模小、语言单一、存在证据泄露等问题。数据集包含4982张图像与6980个跨语言、跨文化的多任务事实核查样本，覆盖六类任务。研究提供基准结果，分析任务协同对结论预测的影响，推动多模态、多语言、多文化场景下的自动化事实核查发展。**

- **链接: [http://arxiv.org/pdf/2510.23508v1](http://arxiv.org/pdf/2510.23508v1)**

> **作者:** Jiahui Geng; Jonathan Tonglet; Iryna Gurevych
>
> **备注:** Preprint under review. Code and data available at: https://github.com/UKPLab/M4FC
>
> **摘要:** Existing real-world datasets for multimodal automated fact-checking have multiple limitations: they contain few instances, focus on only one or two languages and tasks, suffer from evidence leakage, or depend on external sets of news articles for sourcing true claims. To address these shortcomings, we introduce M4FC, a new real-world dataset comprising 4,982 images paired with 6,980 claims. The images, verified by professional fact-checkers from 22 organizations, represent diverse cultural and geographic contexts. Each claim is available in one or two out of ten languages. M4FC spans six multimodal fact-checking tasks: visual claim extraction, claimant intent prediction, fake detection, image contextualization, location verification, and verdict prediction. We provide baseline results for all tasks and analyze how combining intermediate tasks influence downstream verdict prediction performance. We make our dataset and code available.
>
---
#### [new 084] Think Twice: Branch-and-Rethink Reasoning Reward Model
- **分类: cs.CL**

- **简介: 该论文提出分支重思奖励模型（BR-RM），针对传统奖励模型因单次评分导致判断分散的问题，引入两轮推理机制。第一轮自适应选择关键评估维度并提出假设，第二轮针对性复核验证，提升对细微错误的敏感性。实验表明，BR-RM在多个基准上达到领先性能，兼具实用性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.23596v1](http://arxiv.org/pdf/2510.23596v1)**

> **作者:** Yizhu Jiao; Jiaqi Zeng; Julien Veron Vialard; Oleksii Kuchaiev; Jiawei Han; Olivier Delalleau
>
> **摘要:** Large language models (LLMs) increasingly rely on thinking models that externalize intermediate steps and allocate extra test-time compute, with think-twice strategies showing that a deliberate second pass can elicit stronger reasoning. In contrast, most reward models (RMs) still compress many quality dimensions into a single scalar in one shot, a design that induces judgment diffusion: attention spreads across evaluation criteria, yielding diluted focus and shallow analysis. We introduce branch-and-rethink (BR-RM), a two-turn RM that transfers the think-twice principle to reward modeling. Turn 1 performs adaptive branching, selecting a small set of instance-critical dimensions (such as factuality and safety) and sketching concise, evidence-seeking hypotheses. Turn 2 executes branch-conditioned rethinking, a targeted reread that tests those hypotheses and scrutinizes only what matters most. We train with GRPO-style reinforcement learning over structured two-turn traces using a simple binary outcome reward with strict format checks, making the approach compatible with standard RLHF pipelines. By converting all-at-oncescoringintofocused, second-lookreasoning, BR-RMreducesjudgmentdiffusionandimproves sensitivity to subtle yet consequential errors while remaining practical and scalable. Experimental results demonstrate that our model achieves state-of-the-art performance on three challenging reward modeling benchmarks across diverse domains. The code and the model will be released soon.
>
---
#### [new 085] A Stylometric Application of Large Language Models
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于作者风格识别任务，旨在利用大语言模型区分不同作者的写作风格。研究通过为每位作者单独训练GPT-2模型，发现模型对本作者文本预测更准确，从而证明模型能捕捉作者独特风格。实验验证了该方法在八位作家作品上的有效性，并成功确认了《绿野仙踪》第15部的真正作者为R.P.汤普森。**

- **链接: [http://arxiv.org/pdf/2510.21958v1](http://arxiv.org/pdf/2510.21958v1)**

> **作者:** Harrison F. Stropkay; Jiayi Chen; Mohammad J. Latifi; Daniel N. Rockmore; Jeremy R. Manning
>
> **备注:** All code and data needed to reproduce the results in this paper are available at https://github.com/ContextLab/llm-stylometry
>
> **摘要:** We show that large language models (LLMs) can be used to distinguish the writings of different authors. Specifically, an individual GPT-2 model, trained from scratch on the works of one author, will predict held-out text from that author more accurately than held-out text from other authors. We suggest that, in this way, a model trained on one author's works embodies the unique writing style of that author. We first demonstrate our approach on books written by eight different (known) authors. We also use this approach to confirm R. P. Thompson's authorship of the well-studied 15th book of the Oz series, originally attributed to F. L. Baum.
>
---
#### [new 086] SALSA: Single-pass Autoregressive LLM Structured Classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大模型在文本分类任务中表现不佳的问题，提出SALSA框架。通过类标签到唯一输出词元的映射与结构化提示，实现单次前向传播下的高效精准分类，结合参数高效微调，避免冷启动训练，显著提升分类性能。**

- **链接: [http://arxiv.org/pdf/2510.22691v1](http://arxiv.org/pdf/2510.22691v1)**

> **作者:** Ruslan Berdichevsky; Shai Nahum-Gefen; Elad Ben Zaken
>
> **摘要:** Despite their impressive generalization capabilities, instruction-tuned Large Language Models often underperform on text classification benchmarks. We introduce SALSA, a coherent pipeline that combines structured prompting, class-to-token mapping, and parameter-efficient fine-tuning, thereby avoiding cold-start training. Each class label is mapped to a distinct output token, and prompts are constructed to elicit a single-token response. During inference, the model's output is projected only onto the logits of the relevant class tokens, enabling efficient and accurate classification in a single forward pass. SALSA achieves state-of-the-art results across diverse benchmarks, demonstrating its robustness and scalability for LLM-based classification applications.
>
---
#### [new 087] You Don't Need Prompt Engineering Anymore: The Prompting Inversion
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大模型推理中的提示工程问题，针对链式思维（CoT）提示存在的语义模糊与常识错误，提出约束性提示方法“Sculpting”。在GSM8K基准上对比三种策略，发现Sculpting在gpt-4o上优于CoT，但在更先进的gpt-5上反而降低性能，揭示了提示策略需随模型能力进化。**

- **链接: [http://arxiv.org/pdf/2510.22251v1](http://arxiv.org/pdf/2510.22251v1)**

> **作者:** Imran Khan
>
> **备注:** 17 pages, 1 figure, 6 tables. Code and experimental data available at https://github.com/strongSoda/prompt-sculpting
>
> **摘要:** Prompt engineering, particularly Chain-of-Thought (CoT) prompting, significantly enhances LLM reasoning capabilities. We introduce "Sculpting," a constrained, rule-based prompting method designed to improve upon standard CoT by reducing errors from semantic ambiguity and flawed common sense. We evaluate three prompting strategies (Zero Shot, standard CoT, and Sculpting) across three OpenAI model generations (gpt-4o-mini, gpt-4o, gpt-5) using the GSM8K mathematical reasoning benchmark (1,317 problems). Our findings reveal a "Prompting Inversion": Sculpting provides advantages on gpt-4o (97% vs. 93% for standard CoT), but becomes detrimental on gpt-5 (94.00% vs. 96.36% for CoT on full benchmark). We trace this to a "Guardrail-to-Handcuff" transition where constraints preventing common-sense errors in mid-tier models induce hyper-literalism in advanced models. Our detailed error analysis demonstrates that optimal prompting strategies must co-evolve with model capabilities, suggesting simpler prompts for more capable models.
>
---
#### [new 088] Irony Detection in Urdu Text: A Comparative Study Using Machine Learning Models and Large Language Models
- **分类: cs.CL**

- **简介: 该论文聚焦于乌尔都语讽刺检测任务，旨在解决低资源语言中讽刺识别难题。通过翻译英文讽刺语料库至乌尔都语，对比十种机器学习模型与多种大语言模型，发现LLaMA 3（8B）表现最佳，证明结合翻译与先进模型可有效提升乌尔都语讽刺检测性能。**

- **链接: [http://arxiv.org/pdf/2510.22356v1](http://arxiv.org/pdf/2510.22356v1)**

> **作者:** Fiaz Ahmad; Nisar Hussain; Amna Qasim; Momina Hafeez; Muhammad Usman Grigori Sidorov; Alexander Gelbukh
>
> **备注:** 5 pages, 3 figuers
>
> **摘要:** Ironic identification is a challenging task in Natural Language Processing, particularly when dealing with languages that differ in syntax and cultural context. In this work, we aim to detect irony in Urdu by translating an English Ironic Corpus into the Urdu language. We evaluate ten state-of-the-art machine learning algorithms using GloVe and Word2Vec embeddings, and compare their performance with classical methods. Additionally, we fine-tune advanced transformer-based models, including BERT, RoBERTa, LLaMA 2 (7B), LLaMA 3 (8B), and Mistral, to assess the effectiveness of large-scale models in irony detection. Among machine learning models, Gradient Boosting achieved the best performance with an F1-score of 89.18%. Among transformer-based models, LLaMA 3 (8B) achieved the highest performance with an F1-score of 94.61%. These results demonstrate that combining transliteration techniques with modern NLP models enables robust irony detection in Urdu, a historically low-resource language.
>
---
#### [new 089] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大模型参数高效微调任务，解决标准LoRA因所有输入令牌共享相同投影权重而忽略词元语义差异的问题。提出TopLoRA，通过动态生成基于输入词元的对角矩阵，实现词元级自适应投影，在不增加秩的前提下提升微调效果。**

- **链接: [http://arxiv.org/pdf/2510.23123v1](http://arxiv.org/pdf/2510.23123v1)**

> **作者:** Shiwei Li; Xiandi Luo; Haozhao Wang; Xing Tang; Ziqiang Cui; Dugang Liu; Yuhua Li; Xiuqiang He; Ruixuan Li
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Low-rank adaptation (LoRA) is a parameter-efficient fine-tuning (PEFT) method widely used in large language models (LLMs). LoRA essentially describes the projection of an input space into a low-dimensional output space, with the dimensionality determined by the LoRA rank. In standard LoRA, all input tokens share the same weights and undergo an identical input-output projection. This limits LoRA's ability to capture token-specific information due to the inherent semantic differences among tokens. To address this limitation, we propose Token-wise Projected Low-Rank Adaptation (TopLoRA), which dynamically adjusts LoRA weights according to the input token, thereby learning token-wise input-output projections in an end-to-end manner. Formally, the weights of TopLoRA can be expressed as $B\Sigma_X A$, where $A$ and $B$ are low-rank matrices (as in standard LoRA), and $\Sigma_X$ is a diagonal matrix generated from each input token $X$. Notably, TopLoRA does not increase the rank of LoRA weights but achieves more granular adaptation by learning token-wise LoRA weights (i.e., token-wise input-output projections). Extensive experiments across multiple models and datasets demonstrate that TopLoRA consistently outperforms LoRA and its variants. The code is available at https://github.com/Leopold1423/toplora-neurips25.
>
---
#### [new 090] Explaining and Mitigating Crosslingual Tokenizer Inequities
- **分类: cs.CL**

- **简介: 该论文研究跨语言分词器的分词不均问题（token premiums），旨在降低不同语言间分词数量差异。通过训练7000个单语分词器，分析词汇量、预分词策略等因素影响，发现优化词汇量和采用超词分词可显著减少不均现象。**

- **链接: [http://arxiv.org/pdf/2510.21909v1](http://arxiv.org/pdf/2510.21909v1)**

> **作者:** Catherine Arnett; Tyler A. Chang; Stella Biderman; Benjamin K. Bergen
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The number of tokens it takes to encode parallel text in different languages is known to vary. These disparities are called token premiums. Having high token premiums leads to less throughput during training and increases costs at inference. In this paper, we show that even after controlling for dataset size, vocabulary size, and data content, monolingual tokenizers exhibit a wide range of token premiums across languages. To understand the cross-linguistic differences that cause these token premiums, we train a suite of approximately 7,000 comparable monolingual tokenizers for 97 languages, manipulating tokenization algorithm, vocabulary size, and dataset size. We measure token premiums and test for a relationship between factors such as data similarity (between tokenizer training and evaluation), vocabulary size, and pre-tokenization. We also investigate the role of language-specific features such as writing system and word length. We find that similarity between training and test data does not impact token premiums, but vocabulary size and pre-tokenization do. While simply increasing vocabulary size does not lead to reduced token premium effects, we can determine an ``optimal'' vocabulary size for each language to achieve significantly reduced token premium effects. We also train superword tokenizers which allow merges over whitespaces, and we find that they both reduce token premium effects and improve compression overall. Thus, intervening on the vocabulary size or the pre-tokenizer significantly reduces crosslingual token premium effects.
>
---
#### [new 091] SteerX: Disentangled Steering for LLM Personalization
- **分类: cs.CL**

- **简介: 该论文针对大语言模型个性化任务，解决现有激活调制方法因混杂非偏好内容导致的偏差问题。提出SteerX，基于因果推断识别偏好驱动的令牌，分离并重构偏好信号，生成更精准的调制向量，显著提升个性化效果。**

- **链接: [http://arxiv.org/pdf/2510.22256v1](http://arxiv.org/pdf/2510.22256v1)**

> **作者:** Xiaoyan Zhao; Ming Yan; Yilun Qiu; Haoting Ni; Yang Zhang; Fuli Feng; Hong Cheng; Tat-Seng Chua
>
> **摘要:** Large language models (LLMs) have shown remarkable success in recent years, enabling a wide range of applications, including intelligent assistants that support users' daily life and work. A critical factor in building such assistants is personalizing LLMs, as user preferences and needs vary widely. Activation steering, which directly leverages directions representing user preference in the LLM activation space to adjust its behavior, offers a cost-effective way to align the model's outputs with individual users. However, existing methods rely on all historical data to compute the steering vector, ignoring that not all content reflects true user preferences, which undermines the personalization signal. To address this, we propose SteerX, a disentangled steering method that isolates preference-driven components from preference-agnostic components. Grounded in causal inference theory, SteerX estimates token-level causal effects to identify preference-driven tokens, transforms these discrete signals into a coherent description, and then leverages them to steer personalized LLM generation. By focusing on the truly preference-driven information, SteerX produces more accurate activation steering vectors and enhances personalization. Experiments on two representative steering backbone methods across real-world datasets demonstrate that SteerX consistently enhances steering vector quality, offering a practical solution for more effective LLM personalization.
>
---
#### [new 092] Supervised Fine-Tuning or In-Context Learning? Evaluating LLMs for Clinical NER
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究临床命名实体识别（NER）任务，比较BERT类模型、GPT-4o的少样本上下文学习（ICL）与监督微调（SFT）的效果。结果表明，SFT性能最优（F1≈87.1%），ICL在简化任务上表现更好，而预训练模型提升有限。**

- **链接: [http://arxiv.org/pdf/2510.22285v1](http://arxiv.org/pdf/2510.22285v1)**

> **作者:** Andrei Baroian
>
> **备注:** Work done in November - December 2024
>
> **摘要:** We study clinical Named Entity Recognition (NER) on the CADEC corpus and compare three families of approaches: (i) BERT-style encoders (BERT Base, BioClinicalBERT, RoBERTa-large), (ii) GPT-4o used with few-shot in-context learning (ICL) under simple vs.\ complex prompts, and (iii) GPT-4o with supervised fine-tuning (SFT). All models are evaluated on standard NER metrics over CADEC's five entity types (ADR, Drug, Disease, Symptom, Finding). RoBERTa-large and BioClinicalBERT offer limited improvements over BERT Base, showing the limit of these family of models. Among LLM settings, simple ICL outperforms a longer, instruction-heavy prompt, and SFT achieves the strongest overall performance (F1 $\approx$ 87.1%), albeit with higher cost. We find that the LLM achieve higher accuracy on simplified tasks, restricting classification to two labels.
>
---
#### [new 093] Low-Resource Dialect Adaptation of Large Language Models: A French Dialect Case-Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究低资源方言的大型语言模型适配问题，针对加拿大魁北克法语这一少数方言，采用低秩适配（LoRA）与计算高效持续预训练方法，在极小数据和计算预算下成功适配三个LLM。实验表明，仅更新不足1%参数即可显著提升方言性能，且对标准法语基准影响极小，验证了该方法在低成本下扩展高质量语言模型覆盖的可行性。**

- **链接: [http://arxiv.org/pdf/2510.22747v1](http://arxiv.org/pdf/2510.22747v1)**

> **作者:** Eeham Khan; Firas Saidani; Owen Van Esbroeck; Richard Khoury; Leila Kosseim
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** Despite the widespread adoption of large language models (LLMs), their strongest capabilities remain largely confined to a small number of high-resource languages for which there is abundant training data. Recently, continual pre-training (CPT) has emerged as a means to fine-tune these models to low-resource regional dialects. In this paper, we study the use of CPT for dialect learning under tight data and compute budgets. Using low-rank adaptation (LoRA) and compute-efficient continual pre-training, we adapt three LLMs to the Qu\'ebec French dialect using a very small dataset and benchmark them on the COLE suite. Our experiments demonstrate an improvement on the minority dialect benchmarks with minimal regression on the prestige language benchmarks with under 1% of model parameters updated. Analysis of the results demonstrate that gains are highly contingent on corpus composition. These findings indicate that CPT with parameter-efficient fine-tuning (PEFT) can narrow the dialect gap by providing cost-effective and sustainable language resource creation, expanding high-quality LLM access to minority linguistic communities. We release the first Qu\'ebec French LLMs on HuggingFace.
>
---
#### [new 094] Arabic Little STT: Arabic Children Speech Recognition Dataset
- **分类: cs.CL; cs.AI; cs.HC; cs.LG; cs.SD**

- **简介: 该论文聚焦于阿拉伯语儿童语音识别任务，针对低资源语言及儿童语音数据稀缺问题，构建了首个黎凡特阿拉伯语儿童语音数据集Arabic Little STT（355条来自288名6-13岁儿童的语音）。通过评估Whisper模型在该数据集上的表现，发现其词错误率高达0.66，显著高于成人数据集，凸显儿童语音识别的挑战。研究呼吁建立专用儿童语音基准与伦理合规的数据集，推动更公平的语音技术发展。**

- **链接: [http://arxiv.org/pdf/2510.23319v1](http://arxiv.org/pdf/2510.23319v1)**

> **作者:** Mouhand Alkadri; Dania Desouki; Khloud Al Jallad
>
> **摘要:** The performance of Artificial Intelligence (AI) systems fundamentally depends on high-quality training data. However, low-resource languages like Arabic suffer from severe data scarcity. Moreover, the absence of child-specific speech corpora is an essential gap that poses significant challenges. To address this gap, we present our created dataset, Arabic Little STT, a dataset of Levantine Arabic child speech recorded in classrooms, containing 355 utterances from 288 children (ages 6 - 13). We further conduct a systematic assessment of Whisper, a state-of-the-art automatic speech recognition (ASR) model, on this dataset and compare its performance with adult Arabic benchmarks. Our evaluation across eight Whisper variants reveals that even the best-performing model (Large_v3) struggles significantly, achieving a 0.66 word error rate (WER) on child speech, starkly contrasting with its sub 0.20 WER on adult datasets. These results align with other research on English speech. Results highlight the critical need for dedicated child speech benchmarks and inclusive training data in ASR development. Emphasizing that such data must be governed by strict ethical and privacy frameworks to protect sensitive child information. We hope that this study provides an initial step for future work on equitable speech technologies for Arabic-speaking children. We hope that our publicly available dataset enrich the children's demographic representation in ASR datasets.
>
---
#### [new 095] Generalization or Memorization: Dynamic Decoding for Mode Steering
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在推理时出现的泛化与记忆混淆问题，提出动态模式引导（DMS）框架。基于信息瓶颈理论，通过轻量级探针识别记忆依赖，并动态引导计算向泛化路径偏移，提升逻辑一致性和事实准确性，属于增强模型推理可靠性的任务。**

- **链接: [http://arxiv.org/pdf/2510.22099v1](http://arxiv.org/pdf/2510.22099v1)**

> **作者:** Xuanming Zhang
>
> **摘要:** Large Language Models (LLMs) exhibit a troubling duality, capable of both remarkable generalization and brittle, verbatim memorization of their training data. This unpredictability undermines their reliability in high-stakes applications. In this work, we propose a unified framework to understand, identify, and control these distinct reasoning modes. First, we introduce a theoretical model based on the Information Bottleneck (IB) principle, formalizing generalization as the learning of a compressed, task-relevant representation and memorization as a failure to compress. Building on this theory, we develop Dynamic Mode Steering (DMS), a novel inference-time algorithm which comprises two components: (1) a lightweight, causally-grounded linear probe that identifies the model's instantaneous reliance on memorization, and (2) a dynamic activation steering mechanism that nudges the model's computation towards pre-identified generalization circuits. We frame DMS as a form of adaptive, self-contrastive decoding. Experiments on reasoning and faithfulness tasks demonstrate that DMS significantly improves logical consistency and factual accuracy, thereby offering a principled approach to enhancing LLM reliability.
>
---
#### [new 096] A Closed-Loop Personalized Learning Agent Integrating Neural Cognitive Diagnosis, Bounded-Ability Adaptive Testing, and LLM-Driven Feedback
- **分类: cs.CL**

- **简介: 该论文提出EduLoop-Agent，一个闭环个性化学习代理，整合神经认知诊断、受限能力自适应测试与大模型反馈。旨在解决传统方法中诊断、推荐、反馈割裂的问题，实现精准学业诊断、高效试题推荐与可操作反馈，推动智能教育中个性化学习路径的生成。**

- **链接: [http://arxiv.org/pdf/2510.22559v1](http://arxiv.org/pdf/2510.22559v1)**

> **作者:** Zhifeng Wang; Xinyue Zheng; Chunyan Zeng
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** As information technology advances, education is moving from one-size-fits-all instruction toward personalized learning. However, most methods handle modeling, item selection, and feedback in isolation rather than as a closed loop. This leads to coarse or opaque student models, assumption-bound adaptivity that ignores diagnostic posteriors, and generic, non-actionable feedback. To address these limitations, this paper presents an end-to-end personalized learning agent, EduLoop-Agent, which integrates a Neural Cognitive Diagnosis model (NCD), a Bounded-Ability Estimation Computerized Adaptive Testing strategy (BECAT), and large language models (LLMs). The NCD module provides fine-grained estimates of students' mastery at the knowledge-point level; BECAT dynamically selects subsequent items to maximize relevance and learning efficiency; and LLMs convert diagnostic signals into structured, actionable feedback. Together, these components form a closed-loop framework of ``Diagnosis--Recommendation--Feedback.'' Experiments on the ASSISTments dataset show that the NCD module achieves strong performance on response prediction while yielding interpretable mastery assessments. The adaptive recommendation strategy improves item relevance and personalization, and the LLM-based feedback offers targeted study guidance aligned with identified weaknesses. Overall, the results indicate that the proposed design is effective and practically deployable, providing a feasible pathway to generating individualized learning trajectories in intelligent education.
>
---
#### [new 097] Adaptive Blockwise Search: Inference-Time Alignment for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型对齐问题，提出自适应分块搜索（AdaSearch）策略。通过在推理阶段动态分配计算资源，聚焦关键响应初始令牌，提升对齐效果。实验表明，该方法显著优于Best-of-N与微调基线，在安全性、情感控制和数学推理任务上性能提升超10%。**

- **链接: [http://arxiv.org/pdf/2510.23334v1](http://arxiv.org/pdf/2510.23334v1)**

> **作者:** Mohammad Atif Quamar; Mohammad Areeb; Nishant Sharma; Ananth Shreekumar; Jonathan Rosenthal; Muslum Ozgur Ozmen; Mikhail Kuznetsov; Z. Berkay Celik
>
> **摘要:** LLM alignment remains a critical challenge. Inference-time methods provide a flexible alternative to fine-tuning, but their uniform computational effort often yields suboptimal alignment. We hypothesize that for many alignment tasks, the initial tokens of a response are disproportionately more critical. To leverage this principle, we introduce AdaSearch, a novel blockwise search strategy. It adaptively allocates a fixed computational budget using a sampling schedule, focusing search effort on these critical tokens. We apply AdaSearch to sequential decoding and introduce its tree-search counterpart, AdaBeam. Our comprehensive evaluation across eight LLMs demonstrates that AdaSearch outperforms strong Best-of-N and fine-tuning baselines. Specifically, win-rates improve by over 10% for harmlessness generation, controlled sentiment generation, and for mathematical reasoning tasks relative to Best-of-N.
>
---
#### [new 098] The Limits of Data Scaling: Sub-token Utilization and Acoustic Saturation in Multilingual ASR
- **分类: cs.CL**

- **简介: 该论文研究多语言语音识别中数据规模与子词利用的关系。针对“多少音频才能充分激活模型子词库”及“训练数据不均衡是否影响推理时的子词使用”问题，通过分析Whisper在49种语言上的解码行为，发现子词发现遵循指数饱和规律，提出“声学饱和时间”概念，并揭示子词使用受语言类型和书写系统影响更大，而非训练数据量。**

- **链接: [http://arxiv.org/pdf/2510.22492v1](http://arxiv.org/pdf/2510.22492v1)**

> **作者:** Siyu Liang; Nicolas Ballier; Gina-Anne Levow; Richard Wright
>
> **摘要:** How much audio is needed to fully observe a multilingual ASR model's learned sub-token inventory across languages, and does data disparity in multilingual pre-training affect how these tokens are utilized during inference? We address this question by analyzing Whisper's decoding behavior during inference across 49 languages. By logging decoding candidate sub-tokens and tracking their cumulative discovery over time, we study the utilization pattern of the model's sub-token space. Results show that the total number of discovered tokens remains largely independent of a language's pre-training hours, indicating that data disparity does not strongly influence lexical diversity in the model's hypothesis space. Sub-token discovery rates follow a consistent exponential saturation pattern across languages, suggesting a stable time window after which additional audio yields minimal new sub-token activation. We refer to this convergence threshold as acoustic saturation time (AST). Further analyses of rank-frequency distributions reveal Zipf-like patterns better modeled by a Zipf-Mandelbrot law, and mean sub-token length shows a positive correlation with resource level. Additionally, those metrics show more favorable patterns for languages in the Latin script than those in scripts such as Cyrillic, CJK, and Semitic. Together, our study suggests that sub-token utilization during multilingual ASR inference is constrained more by the statistical, typological, and orthographic structure of the speech than by training data scale, providing an empirical basis for more equitable corpus construction and cross-lingual evaluation.
>
---
#### [new 099] How AI Forecasts AI Jobs: Benchmarking LLM Predictions of Labor Market Changes
- **分类: cs.CL**

- **简介: 该论文聚焦于利用大语言模型（LLM）预测AI对就业市场的影响，属于劳动市场预测任务。针对现有方法缺乏系统性评估的问题，构建了包含美国高频职位数据与全球职业预测的基准，通过结构化提示提升预测稳定性，揭示了提示策略与领域特性对预测效果的关键影响。**

- **链接: [http://arxiv.org/pdf/2510.23358v1](http://arxiv.org/pdf/2510.23358v1)**

> **作者:** Sheri Osborn; Rohit Valecha; H. Raghav Rao; Dan Sass; Anthony Rios
>
> **备注:** 8 pages + Limitations + References
>
> **摘要:** Artificial intelligence is reshaping labor markets, yet we lack tools to systematically forecast its effects on employment. This paper introduces a benchmark for evaluating how well large language models (LLMs) can anticipate changes in job demand, especially in occupations affected by AI. Existing research has shown that LLMs can extract sentiment, summarize economic reports, and emulate forecaster behavior, but little work has assessed their use for forward-looking labor prediction. Our benchmark combines two complementary datasets: a high-frequency index of sector-level job postings in the United States, and a global dataset of projected occupational changes due to AI adoption. We format these data into forecasting tasks with clear temporal splits, minimizing the risk of information leakage. We then evaluate LLMs using multiple prompting strategies, comparing task-scaffolded, persona-driven, and hybrid approaches across model families. We assess both quantitative accuracy and qualitative consistency over time. Results show that structured task prompts consistently improve forecast stability, while persona prompts offer advantages on short-term trends. However, performance varies significantly across sectors and horizons, highlighting the need for domain-aware prompting and rigorous evaluation protocols. By releasing our benchmark, we aim to support future research on labor forecasting, prompt design, and LLM-based economic reasoning. This work contributes to a growing body of research on how LLMs interact with real-world economic data, and provides a reproducible testbed for studying the limits and opportunities of AI as a forecasting tool in the context of labor markets.
>
---
#### [new 100] Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中对抗性后缀的可迁移性问题。针对攻击后缀在未优化模型和提示上仍有效这一现象，提出三个关键统计指标：拒绝方向激活度、后缀偏离强度及正交方向位移，揭示其与迁移性的强相关性，弱化语义相似性作用，为提升攻击效率提供理论依据与实践指导。**

- **链接: [http://arxiv.org/pdf/2510.22014v1](http://arxiv.org/pdf/2510.22014v1)**

> **作者:** Sarah Ball; Niki Hasrati; Alexander Robey; Avi Schwarzschild; Frauke Kreuter; Zico Kolter; Andrej Risteski
>
> **摘要:** Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success.
>
---
#### [new 101] Rule-Based Explanations for Retrieval-Augmented LLM Systems
- **分类: cs.CL**

- **简介: 该论文针对检索增强型大语言模型（RAG）的可解释性问题，提出基于“若-则”规则的解释方法。通过分析检索源对输出的影响，生成解释性规则，并设计优化算法加速规则发现，提升效率与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.22689v1](http://arxiv.org/pdf/2510.22689v1)**

> **作者:** Joel Rorseth; Parke Godfrey; Lukasz Golab; Divesh Srivastava; Jarek Szlichta
>
> **摘要:** If-then rules are widely used to explain machine learning models; e.g., "if employed = no, then loan application = rejected." We present the first proposal to apply rules to explain the emerging class of large language models (LLMs) with retrieval-augmented generation (RAG). Since RAG enables LLM systems to incorporate retrieved information sources at inference time, rules linking the presence or absence of sources can explain output provenance; e.g., "if a Times Higher Education ranking article is retrieved, then the LLM ranks Oxford first." To generate such rules, a brute force approach would probe the LLM with all source combinations and check if the presence or absence of any sources leads to the same output. We propose optimizations to speed up rule generation, inspired by Apriori-like pruning from frequent itemset mining but redefined within the scope of our novel problem. We conclude with qualitative and quantitative experiments demonstrating our solutions' value and efficiency.
>
---
#### [new 102] Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages?
- **分类: cs.CL**

- **简介: 该论文研究低资源语言方言语音识别问题，针对现有语音识别模型在方言上表现不佳的挑战，构建了78小时的孟加拉语方言语料库Ben-10。实验表明，基础语音模型在方言识别中效果差，需针对性训练缓解问题，数据集可作为低资源场景下的分布外评估资源。**

- **链接: [http://arxiv.org/pdf/2510.23252v1](http://arxiv.org/pdf/2510.23252v1)**

> **作者:** Tawsif Tashwar Dipto; Azmol Hossain; Rubayet Sabbir Faruque; Md. Rezuwan Hassan; Kanij Fatema; Tanmoy Shome; Ruwad Naswan; Md. Foriduzzaman Zihad; Mohaymen Ul Anam; Nazia Tasnim; Hasan Mahmud; Md Kamrul Hasan; Md. Mehedi Hasan Shawon; Farig Sadeque; Tahsin Reasat
>
> **备注:** This manuscript contains 11 pages, 5 tables and 16 figures This was accepted at International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL) 2025
>
> **摘要:** Conventional research on speech recognition modeling relies on the canonical form for most low-resource languages while automatic speech recognition (ASR) for regional dialects is treated as a fine-tuning task. To investigate the effects of dialectal variations on ASR we develop a 78-hour annotated Bengali Speech-to-Text (STT) corpus named Ben-10. Investigation from linguistic and data-driven perspectives shows that speech foundation models struggle heavily in regional dialect ASR, both in zero-shot and fine-tuned settings. We observe that all deep learning methods struggle to model speech data under dialectal variations but dialect specific model training alleviates the issue. Our dataset also serves as a out of-distribution (OOD) resource for ASR modeling under constrained resources in ASR algorithms. The dataset and code developed for this project are publicly available
>
---
#### [new 103] FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR; 68T50, 68P20; I.2.7; H.3.3**

- **简介: 该论文提出FAIR-RAG，针对复杂多跳问答中信息碎片化、推理不完整的问题。通过结构化证据评估与自适应查询精炼的迭代循环，显式识别并填补信息缺口，提升检索-生成系统的忠实性与准确性，在多个基准上达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.22344v1](http://arxiv.org/pdf/2510.22344v1)**

> **作者:** Mohammad Aghajani Asl; Majid Asgari-Bidhendi; Behrooz Minaei-Bidgoli
>
> **备注:** 30 pages, 5 figures, 5 tables. Keywords: Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Agentic AI, Multi-hop Question Answering, Faithfulness
>
> **摘要:** While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. Current advanced RAG methods, employing iterative or adaptive strategies, lack a robust mechanism to systematically identify and fill evidence gaps, often propagating noise or failing to gather a comprehensive context. We introduce FAIR-RAG, a novel agentic framework that transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning process. At its core is an Iterative Refinement Cycle governed by a module we term Structured Evidence Assessment (SEA). The SEA acts as an analytical gating mechanism: it deconstructs the initial query into a checklist of required findings and audits the aggregated evidence to identify confirmed facts and, critically, explicit informational gaps. These gaps provide a precise signal to an Adaptive Query Refinement agent, which generates new, targeted sub-queries to retrieve missing information. This cycle repeats until the evidence is verified as sufficient, ensuring a comprehensive context for a final, strictly faithful generation. We conducted experiments on challenging multi-hop QA benchmarks, including HotpotQA, 2WikiMultiHopQA, and MusiQue. In a unified experimental setup, FAIR-RAG significantly outperforms strong baselines. On HotpotQA, it achieves an F1-score of 0.453 -- an absolute improvement of 8.3 points over the strongest iterative baseline -- establishing a new state-of-the-art for this class of methods on these benchmarks. Our work demonstrates that a structured, evidence-driven refinement process with explicit gap analysis is crucial for unlocking reliable and accurate reasoning in advanced RAG systems for complex, knowledge-intensive tasks.
>
---
#### [new 104] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models
- **分类: cs.CL**

- **简介: 该论文提出EchoMind，首个关联多层级的情感对话评估基准，旨在解决语音语言模型在感知非词汇声学线索与生成共情回应方面的不足。通过统一脚本下控制语音风格变化，测试模型对情感与语境的综合理解能力，揭示现有模型在指令遵循、语音鲁棒性及声学线索利用上的缺陷，推动更人性化共情对话系统的发展。**

- **链接: [http://arxiv.org/pdf/2510.22758v1](http://arxiv.org/pdf/2510.22758v1)**

> **作者:** Li Zhou; Lutong Yu; You Lyu; Yihang Lin; Zefeng Zhao; Junyi Ao; Yuhao Zhang; Benyou Wang; Haizhou Li
>
> **备注:** Speech Language Models, Spoken Language Understanding, Vocal Cue Perception, Empathetic Dialogue, Benchmark Evaluation
>
> **摘要:** Speech Language Models (SLMs) have made significant progress in spoken language understanding. Yet it remains unclear whether they can fully perceive non lexical vocal cues alongside spoken words, and respond with empathy that aligns with both emotional and contextual factors. Existing benchmarks typically evaluate linguistic, acoustic, reasoning, or dialogue abilities in isolation, overlooking the integration of these skills that is crucial for human-like, emotionally intelligent conversation. We present EchoMind, the first interrelated, multi-level benchmark that simulates the cognitive process of empathetic dialogue through sequential, context-linked tasks: spoken-content understanding, vocal-cue perception, integrated reasoning, and response generation. All tasks share identical and semantically neutral scripts that are free of explicit emotional or contextual cues, and controlled variations in vocal style are used to test the effect of delivery independent of the transcript. EchoMind is grounded in an empathy-oriented framework spanning 3 coarse and 12 fine-grained dimensions, encompassing 39 vocal attributes, and evaluated using both objective and subjective metrics. Testing 12 advanced SLMs reveals that even state-of-the-art models struggle with high-expressive vocal cues, limiting empathetic response quality. Analyses of prompt strength, speech source, and ideal vocal cue recognition reveal persistent weaknesses in instruction-following, resilience to natural speech variability, and effective use of vocal cues for empathy. These results underscore the need for SLMs that integrate linguistic content with diverse vocal cues to achieve truly empathetic conversational ability.
>
---
#### [new 105] Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）在越狱攻击中的脆弱性，提出基于社会科学研究的说服策略构建对抗性提示，揭示模型对特定语言结构的响应倾向。工作包括设计说服型提示、验证其越狱效果，并发现模型响应中存在可识别的“说服指纹”，强调跨学科方法对提升模型安全的重要性。**

- **链接: [http://arxiv.org/pdf/2510.21983v1](http://arxiv.org/pdf/2510.21983v1)**

> **作者:** Havva Alizadeh Noughabi; Julien Serbanescu; Fattane Zarrinkalam; Ali Dehghantanha
>
> **摘要:** Despite recent advances, Large Language Models remain vulnerable to jailbreak attacks that bypass alignment safeguards and elicit harmful outputs. While prior research has proposed various attack strategies differing in human readability and transferability, little attention has been paid to the linguistic and psychological mechanisms that may influence a model's susceptibility to such attacks. In this paper, we examine an interdisciplinary line of research that leverages foundational theories of persuasion from the social sciences to craft adversarial prompts capable of circumventing alignment constraints in LLMs. Drawing on well-established persuasive strategies, we hypothesize that LLMs, having been trained on large-scale human-generated text, may respond more compliantly to prompts with persuasive structures. Furthermore, we investigate whether LLMs themselves exhibit distinct persuasive fingerprints that emerge in their jailbreak responses. Empirical evaluations across multiple aligned LLMs reveal that persuasion-aware prompts significantly bypass safeguards, demonstrating their potential to induce jailbreak behaviors. This work underscores the importance of cross-disciplinary insight in addressing the evolving challenges of LLM safety. The code and data are available.
>
---
#### [new 106] Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对奖励模型（RM）在多模态支持与偏好灵活性上的不足，提出Omni-Reward框架。通过构建首个支持自由形式偏好的多模态基准Omni-RewardBench、248K多模态偏好数据集Omni-RewardData，以及兼具判别与生成能力的Omni-RewardModel，实现对文本、图像、视频、音频、3D等多模态的通用奖励建模，提升模型对多样化人类偏好的适应性。**

- **链接: [http://arxiv.org/pdf/2510.23451v1](http://arxiv.org/pdf/2510.23451v1)**

> **作者:** Zhuoran Jin; Hongbang Yuan; Kejian Zhu; Jiachun Li; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** 48 pages, 17 figures
>
> **摘要:** Reward models (RMs) play a critical role in aligning AI behaviors with human preferences, yet they face two fundamental challenges: (1) Modality Imbalance, where most RMs are mainly focused on text and image modalities, offering limited support for video, audio, and other modalities; and (2) Preference Rigidity, where training on fixed binary preference pairs fails to capture the complexity and diversity of personalized preferences. To address the above challenges, we propose Omni-Reward, a step toward generalist omni-modal reward modeling with support for free-form preferences, consisting of: (1) Evaluation: We introduce Omni-RewardBench, the first omni-modal RM benchmark with free-form preferences, covering nine tasks across five modalities including text, image, video, audio, and 3D; (2) Data: We construct Omni-RewardData, a multimodal preference dataset comprising 248K general preference pairs and 69K instruction-tuning pairs for training generalist omni-modal RMs; (3) Model: We propose Omni-RewardModel, which includes both discriminative and generative RMs, and achieves strong performance on Omni-RewardBench as well as other widely used reward modeling benchmarks.
>
---
#### [new 107] MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型长文本输出的事实性评估难题，提出MAD-Fact多智能体辩论框架。构建中文长文本事实性数据集LongHalluQA，设计基于事实重要性层级的加权评估机制，通过多代理辩论提升评估准确性。旨在提高大模型在医疗、法律等高风险领域输出的可靠性。**

- **链接: [http://arxiv.org/pdf/2510.22967v1](http://arxiv.org/pdf/2510.22967v1)**

> **作者:** Yucheng Ning; Xixun Lin; Fang Fang; Yanan Cao
>
> **备注:** This article has been accepted by Frontiers of Computer Science (FCS)
>
> **摘要:** The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains.
>
---
#### [new 108] Preventing Catastrophic Forgetting: Behavior-Aware Sampling for Safer Language Model Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型微调中的灾难性遗忘问题，提出行为感知采样框架，通过指令-响应行为与语义多样性筛选安全样本。有效减少有害输出达41%，仅用0.5%额外数据，兼顾安全性与有用性，提升微调效率与安全性。**

- **链接: [http://arxiv.org/pdf/2510.21885v1](http://arxiv.org/pdf/2510.21885v1)**

> **作者:** Anh Pham; Mihir Thalanki; Michael Sun; Aditya Chaloo; Ankita Gupta; Tian Xia; Aditya Mate; Ehimwenma Nosakhare; Soundararajan Srinivasan
>
> **摘要:** Large language models often lose previously aligned safety behaviors when fine-tuned on benign data, a phenomenon known as catastrophic forgetting. Prior work shows that adding random safety examples can mitigate this effect, but it remains unclear which examples are most effective. We propose a behavior-aware sampling framework that selects safety examples based on two complementary factors: instruction-response behavior (e.g., refusal versus compliance) and semantic diversity across harm categories. Systematic evaluation shows that this approach substantially reduces harmful outputs while maintaining helpfulness, achieving up to a 41% reduction in harmfulness with only 0.5% additional training data. These results highlight how targeted data selection can improve the safety and efficiency of fine-tuning at scale.
>
---
#### [new 109] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对可视化美学与质量评估任务，提出首个系统性基准VisJudge-Bench，涵盖3090个真实场景样本。研究发现主流MLLMs在评估上显著落后于人类专家，据此提出专用模型VisJudge，有效提升评估精度与一致性。**

- **链接: [http://arxiv.org/pdf/2510.22373v1](http://arxiv.org/pdf/2510.22373v1)**

> **作者:** Yupeng Xie; Zhiyang Zhang; Yifan Wu; Sirong Lu; Jiayi Zhang; Zhaoyang Yu; Jinlin Wang; Sirui Hong; Bang Liu; Chenglin Wu; Yuyu Luo
>
> **备注:** 53 pages, 26 figures, 5 tables
>
> **摘要:** Visualization, a domain-specific yet widely used form of imagery, is an effective way to turn complex datasets into intuitive insights, and its value depends on whether data are faithfully represented, clearly communicated, and aesthetically designed. However, evaluating visualization quality is challenging: unlike natural images, it requires simultaneous judgment across data encoding accuracy, information expressiveness, and visual aesthetics. Although multimodal large language models (MLLMs) have shown promising performance in aesthetic assessment of natural images, no systematic benchmark exists for measuring their capabilities in evaluating visualizations. To address this, we propose VisJudge-Bench, the first comprehensive benchmark for evaluating MLLMs' performance in assessing visualization aesthetics and quality. It contains 3,090 expert-annotated samples from real-world scenarios, covering single visualizations, multiple visualizations, and dashboards across 32 chart types. Systematic testing on this benchmark reveals that even the most advanced MLLMs (such as GPT-5) still exhibit significant gaps compared to human experts in judgment, with a Mean Absolute Error (MAE) of 0.551 and a correlation with human ratings of only 0.429. To address this issue, we propose VisJudge, a model specifically designed for visualization aesthetics and quality assessment. Experimental results demonstrate that VisJudge significantly narrows the gap with human judgment, reducing the MAE to 0.442 (a 19.8% reduction) and increasing the consistency with human experts to 0.681 (a 58.7% improvement) compared to GPT-5. The benchmark is available at https://github.com/HKUSTDial/VisJudgeBench.
>
---
#### [new 110] Memory-based Language Models: An Efficient, Explainable, and Eco-friendly Approach to Large Language Modeling
- **分类: cs.CL**

- **简介: 该论文提出基于记忆的语言建模（OLIFANT），用于高效、可解释且环保的文本生成。针对传统神经网络模型能耗高、不透明的问题，利用内存检索实现快速近邻分类，显著降低训练与推理能耗，提升速度与透明度，同时保持良好预测性能。**

- **链接: [http://arxiv.org/pdf/2510.22317v1](http://arxiv.org/pdf/2510.22317v1)**

> **作者:** Antal van den Bosch; Ainhoa Risco Patón; Teun Buijse; Peter Berck; Maarten van Gompel
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** We present memory-based language modeling as an efficient, eco-friendly alternative to deep neural network-based language modeling. It offers log-linearly scalable next-token prediction performance and strong memorization capabilities. Implementing fast approximations of k-nearest neighbor classification, memory-based language modeling leaves a relatively small ecological footprint both in training and in inference mode, as it relies fully on CPUs and attains low token latencies. Its internal workings are simple and fully transparent. We compare our implementation of memory-based language modeling, OLIFANT, with GPT-2 and GPT-Neo on next-token prediction accuracy, estimated emissions and speeds, and offer some deeper analyses of the model.
>
---
#### [new 111] BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多轮交互中大模型的置信度引导测试时扩展问题。针对现有方法多聚焦单轮场景，提出基于置信度的测试时缩放（TTS）策略，利用模型自评置信度动态决定是否重试，显著降低令牌消耗并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.23458v1](http://arxiv.org/pdf/2510.23458v1)**

> **作者:** Litu Ou; Kuan Li; Huifeng Yin; Liwen Zhang; Zhongwang Zhang; Xixi Wu; Rui Ye; Zile Qiao; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** 25 pages
>
> **摘要:** Confidence in LLMs is a useful indicator of model uncertainty and answer reliability. Existing work mainly focused on single-turn scenarios, while research on confidence in complex multi-turn interactions is limited. In this paper, we investigate whether LLM-based search agents have the ability to communicate their own confidence through verbalized confidence scores after long sequences of actions, a significantly more challenging task compared to outputting confidence in a single interaction. Experimenting on open-source agentic models, we first find that models exhibit much higher task accuracy at high confidence while having near-zero accuracy when confidence is low. Based on this observation, we propose Test-Time Scaling (TTS) methods that use confidence scores to determine answer quality, encourage the model to try again until reaching a satisfactory confidence level. Results show that our proposed methods significantly reduce token consumption while demonstrating competitive performance compared to baseline fixed budget TTS methods.
>
---
#### [new 112] GigaEmbeddings: Efficient Russian Language Embedding Model
- **分类: cs.CL**

- **简介: 该论文提出GigaEmbeddings，针对俄语文本嵌入任务，通过三阶段训练框架提升性能。解决现有模型在俄语上表现不足的问题，融合对比预训练、硬负样本微调与多任务学习，并引入架构优化与剪枝，显著提升效率与效果，在ruMTEB基准上达69.1平均分，超越参数更多模型。**

- **链接: [http://arxiv.org/pdf/2510.22369v1](http://arxiv.org/pdf/2510.22369v1)**

> **作者:** Egor Kolodin; Daria Khomich; Nikita Savushkin; Anastasia Ianina; Fyodor Minkin
>
> **摘要:** We introduce GigaEmbeddings, a novel framework for training high-performance Russian-focused text embeddings through hierarchical instruction tuning of the decoder-only LLM designed specifically for Russian language (GigaChat-3B). Our three-stage pipeline, comprising large-scale contrastive pre-training in web-scale corpora, fine-tuning with hard negatives, and multitask generalization across retrieval, classification, and clustering tasks, addresses key limitations of existing methods by unifying diverse objectives and leveraging synthetic data generation. Architectural innovations include bidirectional attention for contextual modeling, latent attention pooling for robust sequence aggregation, and strategic pruning of 25% of transformer layers to enhance efficiency without compromising performance. Evaluated on the ruMTEB benchmark spanning 23 multilingual tasks, GigaEmbeddings achieves state-of-the-art results (69.1 avg. score), outperforming strong baselines with a larger number of parameters.
>
---
#### [new 113] Detecting Religious Language in Climate Discourse
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的宗教语言检测任务，旨在分析气候话语中宗教语言的显性与隐性表达。研究对比了规则模型与零样本大语言模型在88万句文本上的检测效果，揭示了词汇与语境在定义宗教语言中的张力，展示了计算方法在数字宗教研究中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2510.23395v1](http://arxiv.org/pdf/2510.23395v1)**

> **作者:** Evy Beijen; Pien Pieterse; Yusuf Çelik; Willem Th. van Peursen; Sandjai Bhulai; Meike Morren
>
> **摘要:** Religious language continues to permeate contemporary discourse, even in ostensibly secular domains such as environmental activism and climate change debates. This paper investigates how explicit and implicit forms of religious language appear in climate-related texts produced by secular and religious nongovernmental organizations (NGOs). We introduce a dual methodological approach: a rule-based model using a hierarchical tree of religious terms derived from ecotheology literature, and large language models (LLMs) operating in a zero-shot setting. Using a dataset of more than 880,000 sentences, we compare how these methods detect religious language and analyze points of agreement and divergence. The results show that the rule-based method consistently labels more sentences as religious than LLMs. These findings highlight not only the methodological challenges of computationally detecting religious language but also the broader tension over whether religious language should be defined by vocabulary alone or by contextual meaning. This study contributes to digital methods in religious studies by demonstrating both the potential and the limitations of approaches for analyzing how the sacred persists in climate discourse.
>
---
#### [new 114] Exploration of Summarization by Generative Language Models for Automated Scoring of Long Essays
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究长作文自动评分任务，针对BERT等模型因输入长度限制（512 tokens）导致评分不准确的问题，提出通过生成式语言模型进行摘要与提示，以提升评分精度。实验显示，使用该方法后，QWK得分从0.822提升至0.8878。**

- **链接: [http://arxiv.org/pdf/2510.22830v1](http://arxiv.org/pdf/2510.22830v1)**

> **作者:** Haowei Hua; Hong Jiao; Xinyi Wang
>
> **备注:** 19 pages, 5 Tables 7 Figures, Presentation at Artificial Intelligence in Measurement and Education Conference (AIME-Con)
>
> **摘要:** BERT and its variants are extensively explored for automated scoring. However, a limit of 512 tokens for these encoder-based models showed the deficiency in automated scoring of long essays. Thus, this research explores generative language models for automated scoring of long essays via summarization and prompting. The results revealed great improvement of scoring accuracy with QWK increased from 0.822 to 0.8878 for the Learning Agency Lab Automated Essay Scoring 2.0 dataset.
>
---
#### [new 115] Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; I.2.7; I.2.0; K.6.5**

- **简介: 该论文针对大模型安全漏洞，提出Jailbreak Mimicry方法，通过高效微调生成叙事型越狱提示，实现一键式自动化攻击。解决人工构造攻击效率低、可复现性差问题，显著提升攻击成功率（81%），揭示模型在网络安全、欺诈等领域的脆弱性，推动AI安全防御研究。**

- **链接: [http://arxiv.org/pdf/2510.22085v1](http://arxiv.org/pdf/2510.22085v1)**

> **作者:** Pavlos Ntais
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity.
>
---
#### [new 116] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大模型终身知识编辑任务，解决长期编辑导致的误差累积与性能下降问题。提出NMKE框架，通过神经元级归因与动态稀疏掩码，精准定位知识相关神经元，实现少参数修改下的高效、高精度编辑，显著提升编辑成功率与模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22139v1](http://arxiv.org/pdf/2510.22139v1)**

> **作者:** Jinzhe Liu; Junshu Sun; Shufan Shen; Chenxue Yang; Shuhui Wang
>
> **备注:** 19 pages, 11 figures, Accepted by NeurIPS 2025
>
> **摘要:** Lifelong knowledge editing enables continuous, precise updates to outdated knowledge in large language models (LLMs) without computationally expensive full retraining. However, existing methods often accumulate errors throughout the editing process, causing a gradual decline in both editing accuracy and generalization. To tackle this problem, we propose Neuron-Specific Masked Knowledge Editing (NMKE), a novel fine-grained editing framework that combines neuron-level attribution with dynamic sparse masking. Leveraging neuron functional attribution, we identify two key types of knowledge neurons, with knowledge-general neurons activating consistently across prompts and knowledge-specific neurons activating to specific prompts. NMKE further introduces an entropy-guided dynamic sparse mask, locating relevant neurons to the target knowledge. This strategy enables precise neuron-level knowledge editing with fewer parameter modifications. Experimental results from thousands of sequential edits demonstrate that NMKE outperforms existing methods in maintaining high editing success rates and preserving model general capabilities in lifelong editing.
>
---
#### [new 117] M-CIF: Multi-Scale Alignment For CIF-Based Non-Autoregressive ASR
- **分类: cs.SD; cs.CL**

- **简介: 该论文针对非自回归语音识别中的对齐不稳问题，提出多尺度CIF（M-CIF）机制，通过融合字符与音素层级监督，逐步优化子词表示的对齐。实验表明，M-CIF显著降低WER，尤其在德语和法语上提升明显，验证了多层级监督对增强声文对齐的有效性。**

- **链接: [http://arxiv.org/pdf/2510.22172v1](http://arxiv.org/pdf/2510.22172v1)**

> **作者:** Ruixiang Mao; Xiangnan Ma; Qing Yang; Ziming Zhu; Yucheng Qiao; Yuan Ge; Tong Xiao; Shengxiang Gao; Zhengtao Yu; Jingbo Zhu
>
> **摘要:** The Continuous Integrate-and-Fire (CIF) mechanism provides effective alignment for non-autoregressive (NAR) speech recognition. This mechanism creates a smooth and monotonic mapping from acoustic features to target tokens, achieving performance on Mandarin competitive with other NAR approaches. However, without finer-grained guidance, its stability degrades in some languages such as English and French. In this paper, we propose Multi-scale CIF (M-CIF), which performs multi-level alignment by integrating character and phoneme level supervision progressively distilled into subword representations, thereby enhancing robust acoustic-text alignment. Experiments show that M-CIF reduces WER compared to the Paraformer baseline, especially on CommonVoice by 4.21% in German and 3.05% in French. To further investigate these gains, we define phonetic confusion errors (PE) and space-related segmentation errors (SE) as evaluation metrics. Analysis of these metrics across different M-CIF settings reveals that the phoneme and character layers are essential for enhancing progressive CIF alignment.
>
---
#### [new 118] Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对离散文本生成中的并行采样问题，提出PUNT方法，通过条件独立性测试识别并移除低置信度依赖项，实现高置信度、近似独立的并行更新。解决了并行与高置信度间的冲突，显著提升长序列生成的准确率与效率。**

- **链接: [http://arxiv.org/pdf/2510.21961v1](http://arxiv.org/pdf/2510.21961v1)**

> **作者:** Iskander Azangulov; Teodora Pandeva; Niranjani Prasad; Javier Zazo; Sushrut Karmalkar
>
> **摘要:** Masked diffusion models (MDMs) offer a compelling alternative to autoregressive models (ARMs) for discrete text generation because they enable parallel token sampling, rather than sequential, left-to-right generation. This means potentially much faster inference. However, effective parallel sampling faces two competing requirements: (i) simultaneously updated tokens must be conditionally independent, and (ii) updates should prioritise high-confidence predictions. These goals conflict because high-confidence predictions often cluster and depend on each other, opportunities for parallel updates. We present PUNT, a model-agnostic sampler that reconciles this trade-off. Our method identifies token dependencies and removes lower-confidence tokens from conflicting groups. This produces sets of indices for unmasking that satisfy both independence and confidence criteria. Our approach ensures improved parallel unmasking through approximate conditional independence testing. Our experiments show that PUNT delivers a superior trade-off between accuracy and compute when compared to other strong training-free baselines, especially for generation of longer sequences. On the IFEval benchmark, it achieves up to 16\% higher accuracy over baseline methods, including sequential generation (one-by-one). These gains hold across different values of hyperparameters, mitigating the need for brittle hyperparameter tuning. Moreover, we observe that PUNT induces an emergent hierarchical generation strategy, where the model first establishes high-level paragraph structure before local refinement, suggesting a planning-like generation process that contributes to strong alignment performance.
>
---
#### [new 119] DynaSolidGeo: A Dynamic Benchmark for Genuine Spatial Mathematical Reasoning of VLMs in Solid Geometry
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出DynaSolidGeo，首个动态基准，用于评估视觉语言模型在立体几何中的真实空间数学推理能力。针对现有基准多限于2D、静态数据及忽视推理过程的问题，构建可动态生成的多样化多模态题集，并引入专家标注的推理链进行过程评估，揭示模型在空间智能上的显著不足。**

- **链接: [http://arxiv.org/pdf/2510.22340v1](http://arxiv.org/pdf/2510.22340v1)**

> **作者:** Changti Wu; Shijie Lian; Zihao Liu; Lei Zhang; Laurence Tianruo Yang; Kai Chen
>
> **备注:** The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}
>
> **摘要:** Solid geometry problem solving demands spatial mathematical reasoning that integrates spatial intelligence and symbolic reasoning. However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process. To address these limitations, we introduce DynaSolidGeo, the first dynamic benchmark for evaluating genuine spatial reasoning in Vision-Language Models (VLMs). Constructed through a semi-automatic annotation pipeline, DynaSolidGeo contains 503 expert-curated seed questions that can, in principle, dynamically generate an unbounded number of diverse multimodal text-visual instances. Beyond answer accuracy, we incorporate process evaluation based on expert-annotated reasoning chains to measure logical validity and causal coherence. Experiments across representative open-source and closed-source VLMs reveal large performance gaps, severe degradation in dynamic settings, and poor performance on tasks requiring high-level spatial intelligence, such as mental rotation and visualization. The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}.
>
---
#### [new 120] ISA-Bench: Benchmarking Instruction Sensitivity for Large Audio Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文针对大音频语言模型（LALMs）对指令表述高度敏感的问题，提出ISA-Bench基准，从指令描述、输出格式、任务组合三方面系统评估其敏感性。通过实验发现主流LALMs性能受指令影响显著，进而提出微调策略提升指令遵循能力，但引发灾难性遗忘。研究为构建鲁棒音频理解系统提供标准化评估与改进路径。**

- **链接: [http://arxiv.org/pdf/2510.23558v1](http://arxiv.org/pdf/2510.23558v1)**

> **作者:** Bohan Li; Wenbin Huang; Yuhang Qiu; Yiwei Guo; Hankun Wang; Zhihan Li; Jing Peng; Ziyang Ma; Xie Chen; Kai Yu
>
> **备注:** submitted to icassp 2026
>
> **摘要:** Large Audio Language Models (LALMs), which couple acoustic perception with large language models (LLMs) to extract and understand diverse information from audio, have attracted intense interest from both academic and industrial communities. However, existing LALMs are highly sensitive to how instructions are phrased, affecting both (i) instruction-following rates and (ii) task performance. Yet, no existing benchmarks offer a systematic and comprehensive evaluation of this sensitivity. We introduce ISA-Bench, a dynamic benchmark evaluating instruction sensitivity for LALMs along three axes: instruction description, output format, and task composition. We assess recent open-source and proprietary LALMs using ISA-Bench, profiling both compliance and accuracy under controlled instruction variations. Experimental results reveal that even state-of-the-art LALMs suffer significant instruction sensitivity, leading to degraded performance on fundamental audio understanding tasks. To mitigate this issue, we fine-tune Qwen2-Audio on a specifically constructed complex instruction-variant dataset, achieving a marked improvement in instruction-following performance. However, this also induces nontrivial catastrophic forgetting: the model loses some previously mastered task capabilities when exposed to new instruction styles. Our benchmark provides a standardized basis for assessing and improving instruction sensitivity in LALMs, underscoring the need for instruction-robust audio understanding in real-world pipelines.
>
---
#### [new 121] Next-Generation LLM for UAV: From Natural Language to Autonomous Flight
- **分类: cs.RO; cs.AI; cs.CL; cs.SY; eess.SY**

- **简介: 该论文提出下一代无人机大语言模型系统NeLV，旨在将自然语言指令转化为多尺度无人机自主飞行任务。针对现有研究局限于小型无人机、缺乏全流程自动化的问题，构建五组件架构实现从指令解析到飞行控制的闭环，并提出五级自动化分级体系，推动无人机向全自主飞行演进。**

- **链接: [http://arxiv.org/pdf/2510.21739v1](http://arxiv.org/pdf/2510.21739v1)**

> **作者:** Liangqi Yuan; Chuhao Deng; Dong-Jun Han; Inseok Hwang; Sabine Brunswicker; Christopher G. Brinton
>
> **摘要:** With the rapid advancement of Large Language Models (LLMs), their capabilities in various automation domains, particularly Unmanned Aerial Vehicle (UAV) operations, have garnered increasing attention. Current research remains predominantly constrained to small-scale UAV applications, with most studies focusing on isolated components such as path planning for toy drones, while lacking comprehensive investigation of medium- and long-range UAV systems in real-world operational contexts. Larger UAV platforms introduce distinct challenges, including stringent requirements for airport-based take-off and landing procedures, adherence to complex regulatory frameworks, and specialized operational capabilities with elevated mission expectations. This position paper presents the Next-Generation LLM for UAV (NeLV) system -- a comprehensive demonstration and automation roadmap for integrating LLMs into multi-scale UAV operations. The NeLV system processes natural language instructions to orchestrate short-, medium-, and long-range UAV missions through five key technical components: (i) LLM-as-Parser for instruction interpretation, (ii) Route Planner for Points of Interest (POI) determination, (iii) Path Planner for waypoint generation, (iv) Control Platform for executable trajectory implementation, and (v) UAV monitoring. We demonstrate the system's feasibility through three representative use cases spanning different operational scales: multi-UAV patrol, multi-POI delivery, and multi-hop relocation. Beyond the current implementation, we establish a five-level automation taxonomy that charts the evolution from current LLM-as-Parser capabilities (Level 1) to fully autonomous LLM-as-Autopilot systems (Level 5), identifying technical prerequisites and research challenges at each stage.
>
---
#### [new 122] PTPP-Aware Adaptation Scaling Laws: Predicting Domain-Adaptation Performance at Unseen Pre-Training Budgets
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究持续预训练（CPT）中的领域适应问题，旨在预测不同预训练预算（PTPP）下的适应性能。提出PTPP-aware缩放律，显式建模预训练预算，实现对未见PTPP下目标域损失的准确预测，并用于在计算约束下规划适应策略。**

- **链接: [http://arxiv.org/pdf/2510.23198v1](http://arxiv.org/pdf/2510.23198v1)**

> **作者:** Etienne Goffinet; Shane Bergsma; Avraham Sheinin; Natalia Vassilieva; Shaheer Muhammad; Preslav Nakov; Gurpreet Gosal
>
> **摘要:** Continual pre-training (CPT) for domain adaptation must balance target-domain gains with stability on the base domain. Existing CPT scaling laws typically assume a fixed pre-training budget, which limits their ability to forecast adaptation outcomes for models trained at different tokens-per-parameter (PTPP). We present \emph{PTPP-aware} adaptation scaling laws that make the pre-training budget an explicit variable, enabling accurate \emph{prediction} of adaptation loss at unseen \ptpp. On a multilingual setup (English/Arabic $\rightarrow$ French), PTPP-aware formulations trained on early stages (\ptpp{}=\{15,31\}) predict target loss at \ptpp{}=279 and outperform a PTPP-agnostic \dcpt{} transfer baseline on metrics (Huber-on-log, MAE$_\mathrm{rel}$, calibration slope); full diagnostics (RMSE, MAPE) are in the appendix. Beyond forecasting, we show a practical use case: planning replay ratios and adaptation token budgets that satisfy target and forgetting constraints under compute limits.
>
---
#### [new 123] Planning Ahead with RSA: Efficient Signalling in Dynamic Environments by Projecting User Awareness across Future Timesteps
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究动态环境中人机协作的智能提示问题。针对人类注意力有限导致信息接收效率低的问题，提出基于理性言语行为（RSA）的多步规划框架，通过预测用户认知变化，优化消息传递时机与内容，提升人机信念对齐效率。首次将RSA应用于动态环境的人机交互，实现了更高效的适应性通信。**

- **链接: [http://arxiv.org/pdf/2510.23340v1](http://arxiv.org/pdf/2510.23340v1)**

> **作者:** Anwesha Das; John Duff; Jörg Hoffmann; Vera Demberg
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Adaptive agent design offers a way to improve human-AI collaboration on time-sensitive tasks in rapidly changing environments. In such cases, to ensure the human maintains an accurate understanding of critical task elements, an assistive agent must not only identify the highest priority information but also estimate how and when this information can be communicated most effectively, given that human attention represents a zero-sum cognitive resource where focus on one message diminishes awareness of other or upcoming information. We introduce a theoretical framework for adaptive signalling which meets these challenges by using principles of rational communication, formalised as Bayesian reference resolution using the Rational Speech Act (RSA) modelling framework, to plan a sequence of messages which optimise timely alignment between user belief and a dynamic environment. The agent adapts message specificity and timing to the particulars of a user and scenario based on projections of how prior-guided interpretation of messages will influence attention to the interface and subsequent belief update, across several timesteps out to a fixed horizon. In a comparison to baseline methods, we show that this effectiveness depends crucially on combining multi-step planning with a realistic model of user awareness. As the first application of RSA for communication in a dynamic environment, and for human-AI interaction in general, we establish theoretical foundations for pragmatic communication in human-agent teams, highlighting how insights from cognitive science can be capitalised to inform the design of assistive agents.
>
---
#### [new 124] Power to the Clients: Federated Learning in a Dictatorship Setting
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.CV; cs.DC**

- **简介: 该论文研究联邦学习中的安全问题，针对恶意客户端可能破坏模型训练的漏洞，提出“独裁客户端”概念。通过理论分析与实证验证，揭示单个或多个独裁客户端如何抹除其他客户端贡献并影响模型收敛，揭示了联邦学习在对抗性环境下的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.22149v1](http://arxiv.org/pdf/2510.22149v1)**

> **作者:** Mohammadsajad Alipour; Mohammad Mohammadi Amiri
>
> **摘要:** Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks.
>
---
#### [new 125] ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出ATOM，一种用于动态时间知识图谱构建的少样本、可扩展方法。针对传统静态知识图谱难以适应实时数据变化及现有方法稳定性差、覆盖不全的问题，ATOM通过拆分文本为原子事实，采用双时间建模并行合并，实现高效、稳定的知识抽取与更新。**

- **链接: [http://arxiv.org/pdf/2510.22590v1](http://arxiv.org/pdf/2510.22590v1)**

> **作者:** Yassir Lairgi; Ludovic Moncla; Khalid Benabdeslem; Rémy Cazabet; Pierre Cléau
>
> **摘要:** In today's rapidly expanding data landscape, knowledge extraction from unstructured text is vital for real-time analytics, temporal inference, and dynamic memory frameworks. However, traditional static knowledge graph (KG) construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts. To address these challenges, we introduce ATOM (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained "atomic" facts, improving extraction exhaustivity and stability. Then, it constructs atomic TKGs from these facts while employing a dual-time modeling that distinguishes when information is observed from when it is valid. The resulting atomic TKGs are subsequently merged in parallel. Empirical evaluations demonstrate that ATOM achieves ~18% higher exhaustivity, ~17% better stability, and over 90% latency reduction compared to baseline methods, demonstrating a strong scalability potential for dynamic TKG construction.
>
---
#### [new 126] Scalable Oversight via Partitioned Human Supervision
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对多领域复杂任务中人类专家难以提供完整标注的问题，提出基于互补标签的可扩展监督框架。利用专家在狭窄领域的弱信号（如排除错误选项），设计无偏估计器与混合估计方法，实现无需真实标签的模型评估与训练，支持大语言模型的自动优化。**

- **链接: [http://arxiv.org/pdf/2510.22500v1](http://arxiv.org/pdf/2510.22500v1)**

> **作者:** Ren Yin; Takashi Ishida; Masashi Sugiyama
>
> **摘要:** As artificial intelligence (AI) systems approach and surpass expert human performance across a broad range of tasks, obtaining high-quality human supervision for evaluation and training becomes increasingly challenging. Our focus is on tasks that require deep knowledge and skills of multiple domains. Unfortunately, even the best human experts are knowledgeable only in a single narrow area, and will not be able to evaluate the correctness of advanced AI systems on such superhuman tasks. However, based on their narrow expertise, humans may provide a weak signal, i.e., a complementary label indicating an option that is incorrect. For example, a cardiologist could state that "this is not related to cardiology,'' even if they cannot identify the true disease. Based on this weak signal, we propose a scalable oversight framework that enables us to evaluate frontier AI systems without the need to prepare the ground truth. We derive an unbiased estimator of top-1 accuracy from complementary labels and quantify how many complementary labels are needed to match the variance of ordinary labels. We further introduce two estimators to combine scarce ordinary labels with abundant complementary labels. We provide finite-sample deviation guarantees for both complementary-only and the mixed estimators. Empirically, we show that we can evaluate the output of large language models without the ground truth, if we have complementary labels. We further show that we can train an AI system with such weak signals: we show how we can design an agentic AI system automatically that can perform better with this partitioned human supervision. Our code is available at https://github.com/R-Yin-217/Scalable-Oversight-via-Human-Partitioned-Supervision.
>
---
#### [new 127] A Multimodal, Multitask System for Generating E Commerce Text Listings from Images
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出一种多模态多任务系统，用于从图像生成电商文本列表。针对人工编写耗时及现有模型易产生事实幻觉的问题，设计了联合训练视觉编码器与分层生成机制，提升事实一致性与效率，显著降低幻觉率并加速生成。**

- **链接: [http://arxiv.org/pdf/2510.21835v1](http://arxiv.org/pdf/2510.21835v1)**

> **作者:** Nayan Kumar Singh
>
> **备注:** 24 pages, 10 figures, 11 tables. Code can be found at: https://github.com/SinghNayanKumar/multimodal-product-lister/
>
> **摘要:** Manually generating catchy descriptions and names is labor intensive and a slow process for retailers. Although generative AI provides an automation solution in form of Vision to Language Models (VLM), the current VLMs are prone to factual "hallucinations". Siloed, single task models are not only inefficient but also fail to capture interdependent relationships between features. To address these challenges, we propose an end to end, multi task system that generates factually grounded textual listings from a single image. The contributions of this study are two proposals for the model architecture. First, application of multi task learning approach for fine tuning a vision encoder where a single vision backbone is jointly trained on attribute prediction such as color, hemline and neck style and price regression. Second, introduction of a hierarchical generation process where the model's own predicted attributes are embedded in a prompt and fed to the text decoder to improve factual consistency. The experiments demonstrate the superiority of this architecture. The multi tasking approach outperforms both the independent price regression, with a 3.6% better R2 Value and attribute classification, with a 6.6% improvement F1 score. Critically, the hierarchical generation process proves highly effective, slashing the factual hallucination rate from 12.7% to 7.1%, a 44.5% relative reduction, compared to a non hierarchical ablation. The hierarchical approach also reduces the latency of the autoregressive text generation process by a factor of 3.5 when compared to direct vision to language model of similar size. One minor caveat is that the model does perform 3.5% worse than direct vision-to-language model on ROUGE-L score.
>
---
#### [new 128] OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦多模态大模型中的信息遗忘任务，针对现有基准在图像多样性、准确性与评估场景上的不足，提出OFFSIDE基准。基于足球转会谣言构建15.68K条数据，涵盖遗忘效果、泛化、实用性和鲁棒性四类测试集，支持选择性遗忘与单模态遗忘。实验揭示当前方法在视觉谣言处理、信息恢复与提示攻击下均存在严重漏洞。**

- **链接: [http://arxiv.org/pdf/2510.22535v1](http://arxiv.org/pdf/2510.22535v1)**

> **作者:** Hao Zheng; Zirui Pang; Ling li; Zhijie Deng; Yuhan Pu; Zhaowei Zhu; Xiaobo Xia; Jiaheng Wei
>
> **摘要:** Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at \href{https://github.com/zh121800/OFFSIDE}{https://github.com/zh121800/OFFSIDE}.
>
---
#### [new 129] Optimal Detection for Language Watermarks with Pseudorandom Collision
- **分类: cs.LG; cs.CL; cs.CR; math.ST; stat.ML; stat.TH**

- **简介: 该论文针对大模型文本水印检测中因重复生成导致的伪随机性失效问题，提出基于最小单元的分层统计框架，解决依赖结构下的误报控制难题。通过构建非渐近效率指标与极小化最大风险检验，推导出最优检测规则，理论与实验均验证了其高效性与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.22007v1](http://arxiv.org/pdf/2510.22007v1)**

> **作者:** T. Tony Cai; Xiang Li; Qi Long; Weijie J. Su; Garrett G. Wen
>
> **摘要:** Text watermarking plays a crucial role in ensuring the traceability and accountability of large language model (LLM) outputs and mitigating misuse. While promising, most existing methods assume perfect pseudorandomness. In practice, repetition in generated text induces collisions that create structured dependence, compromising Type I error control and invalidating standard analyses. We introduce a statistical framework that captures this structure through a hierarchical two-layer partition. At its core is the concept of minimal units -- the smallest groups treatable as independent across units while permitting dependence within. Using minimal units, we define a non-asymptotic efficiency measure and cast watermark detection as a minimax hypothesis testing problem. Applied to Gumbel-max and inverse-transform watermarks, our framework produces closed-form optimal rules. It explains why discarding repeated statistics often improves performance and shows that within-unit dependence must be addressed unless degenerate. Both theory and experiments confirm improved detection power with rigorous Type I error control. These results provide the first principled foundation for watermark detection under imperfect pseudorandomness, offering both theoretical insight and practical guidance for reliable tracing of model outputs.
>
---
#### [new 130] CityRiSE: Reasoning Urban Socio-Economic Status in Vision-Language Models via Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CityRiSE框架，利用强化学习引导视觉语言模型推理城市社会经济状况。针对LVLM在视觉数据下预测不准、不可解释的问题，通过设计可验证奖励机制，促使模型聚焦语义线索，实现结构化、目标导向的推理，显著提升跨城市、跨指标的预测准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22282v1](http://arxiv.org/pdf/2510.22282v1)**

> **作者:** Tianhui Liu; Hetian Pang; Xin Zhang; Jie Feng; Yong Li; Pan Hui
>
> **摘要:** Harnessing publicly available, large-scale web data, such as street view and satellite imagery, urban socio-economic sensing is of paramount importance for achieving global sustainable development goals. With the emergence of Large Vision-Language Models (LVLMs), new opportunities have arisen to solve this task by treating it as a multi-modal perception and understanding problem. However, recent studies reveal that LVLMs still struggle with accurate and interpretable socio-economic predictions from visual data. To address these limitations and maximize the potential of LVLMs, we introduce \textbf{CityRiSE}, a novel framework for \textbf{R}eason\textbf{i}ng urban \textbf{S}ocio-\textbf{E}conomic status in LVLMs through pure reinforcement learning (RL). With carefully curated multi-modal data and verifiable reward design, our approach guides the LVLM to focus on semantically meaningful visual cues, enabling structured and goal-oriented reasoning for generalist socio-economic status prediction. Experiments demonstrate that CityRiSE with emergent reasoning process significantly outperforms existing baselines, improving both prediction accuracy and generalization across diverse urban contexts, particularly for prediction on unseen cities and unseen indicators. This work highlights the promise of combining RL and LVLMs for interpretable and generalist urban socio-economic sensing.
>
---
#### [new 131] Critical Insights into Leading Conversational AI Models
- **分类: cs.AI; cs.CL; I.2.7; I.2.8**

- **简介: 该论文属于对话式AI模型比较任务，旨在解决主流大语言模型在性能、伦理与可用性上的差异问题。研究对比了谷歌Gemini、DeepSeek、Claude、ChatGPT和LLaMA五款模型，分析其在准确性、道德行为及集成便利性方面的表现，指出各模型优势，提出应根据需求选择适配模型。**

- **链接: [http://arxiv.org/pdf/2510.22729v1](http://arxiv.org/pdf/2510.22729v1)**

> **作者:** Urja Kohli; Aditi Singh; Arun Sharma
>
> **备注:** 21 pages, 7 tables, 3 figures. Open-access preprint intended for journal or conference submission
>
> **摘要:** Big Language Models (LLMs) are changing the way businesses use software, the way people live their lives and the way industries work. Companies like Google, High-Flyer, Anthropic, OpenAI and Meta are making better LLMs. So, it's crucial to look at how each model is different in terms of performance, moral behaviour and usability, as these differences are based on the different ideas that built them. This study compares five top LLMs: Google's Gemini, High-Flyer's DeepSeek, Anthropic's Claude, OpenAI's GPT models and Meta's LLaMA. It performs this by analysing three important factors: Performance and Accuracy, Ethics and Bias Mitigation and Usability and Integration. It was found that Claude has good moral reasoning, Gemini is better at multimodal capabilities and has strong ethical frameworks. DeepSeek is great at reasoning based on facts, LLaMA is good for open applications and ChatGPT delivers balanced performance with a focus on usage. It was concluded that these models are different in terms of how well they work, how easy they are to use and how they treat people ethically, making it a point that each model should be utilised by the user in a way that makes the most of its strengths.
>
---
#### [new 132] Diagnosing Bottlenecks in Data Visualization Understanding by Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦数据可视化理解任务，旨在诊断视觉语言模型（VLMs）在处理数据图时失败的原因。通过构建FUGU基准，结合激活修补与线性探测，发现错误多源于视觉-语言模块间的信息传递，而非编码或语言处理本身。即使提供正确坐标，复杂统计任务性能仍下降，表明现有VLM架构存在根本局限。**

- **链接: [http://arxiv.org/pdf/2510.21740v1](http://arxiv.org/pdf/2510.21740v1)**

> **作者:** Alexa R. Tartaglini; Satchel Grant; Daniel Wurgaft; Christopher Potts; Judith E. Fan
>
> **摘要:** Data visualizations are vital components of many scientific articles and news stories. Current vision-language models (VLMs) still struggle on basic data visualization understanding tasks, but the causes of failure remain unclear. Are VLM failures attributable to limitations in how visual information in the data visualization is encoded, how information is transferred between the vision and language modules, or how information is processed within the language module? We developed FUGU, a suite of data visualization understanding tasks, to precisely characterize potential sources of difficulty (e.g., extracting the position of data points, distances between them, and other summary statistics). We used FUGU to investigate three widely used VLMs. To diagnose the sources of errors produced by these models, we used activation patching and linear probes to trace information flow through models across a variety of prompting strategies. We found that some models fail to generate the coordinates of individual data points correctly, and these initial errors often lead to erroneous final responses. When these models are provided with the correct coordinates, performance improves substantially. Moreover, even when the model generates an incorrect response, the correct coordinates can be successfully read out from the latent representations in the vision encoder, suggesting that the source of these errors lies in the vision-language handoff. We further found that while providing correct coordinates helps with tasks involving one or a small number of data points, it generally worsens performance for tasks that require extracting statistical relationships across many data points. Fine-tuning models on FUGU also fails to yield ceiling performance. These findings point to architectural constraints in current VLMs that might pose significant challenges for reliable data visualization understanding.
>
---
#### [new 133] GeoThought: A Dataset for Enhancing Mathematical Geometry Reasoning in Vision-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦于视觉-语言模型的几何推理任务，针对现有模型在几何问题上表现不佳的问题，构建了包含详细推理链的GeoThought数据集，并提出GeoThought-MLLM模型。通过大规模、多步骤的思维链训练，显著提升模型在几何推理上的准确率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21881v1](http://arxiv.org/pdf/2510.21881v1)**

> **作者:** Nannan Shi; Chuanyu Qin; Shipeng Song; Man Luo
>
> **摘要:** Large language models (LLMs) have demonstrated strong reasoning capabilities in text-based mathematical problem solving; however, when adapted to visual reasoning tasks, particularly geometric problem solving, their performance substantially declines because geometric problems present unique challenges. Specifically, these challenges stem from two key factors: first, the intrinsic complexity of geometry requiring detailed image comprehension and multi-step reasoning, and second, the limitations of existing datasets which lack sufficient scale, diversity, and explicit reasoning traces, consequently hindering effective model training. To address these challenges, we developed the GeoThoughts dataset, a comprehensive geometric reasoning corpus with two subsets: Geo-Thought-6K with 6,243 samples and its augmented version Geo-Thought-Augmented-10K containing 10,834 samples. Each entry includes visual descriptions, step-by-step solutions, explicit reasoning chains, reflection steps, and final answers. Using this dataset, we developed GeoThought-MLLM, a mathematical reasoning multimodal model that generates detailed thinking processes during problem-solving. Our model outperforms existing benchmarks in geometric tasks, demonstrating that training with our Chain-of-Thought dataset improves geometric reasoning capabilities across both in-domain and out-of-domain settings. Finally, we analyze failure cases and observe that errors primarily arise from incorrect interpretation of mathematical concepts or spatial misjudgment. By invoking CoT to correct these mistakes, the model produces correct answers.
>
---
#### [new 134] ATLAS: Actor-Critic Task-Completion with Look-ahead Action Simulation
- **分类: cs.LG; cs.AI; cs.CL; cs.IR; cs.MA; cs.RO**

- **简介: 该论文提出ATLAS，一种无需微调的网页智能体，解决现有方法在新环境中规划效率低的问题。通过构建认知地图、动作模拟与回溯优化，实现高效任务完成，在WebArena-Lite上达63%成功率，显著优于此前最优。**

- **链接: [http://arxiv.org/pdf/2510.22732v1](http://arxiv.org/pdf/2510.22732v1)**

> **作者:** Jiali Cheng; Anjishnu Kumar; Roshan Lal; Rishi Rajasekaran; Hani Ramezani; Omar Zia Khan; Oleg Rokhlenko; Sunny Chiu-Webster; Gang Hua; Hadi Amiri
>
> **备注:** 9 pages, NeurIPS 2025 Workshop on Language Agents and World Models
>
> **摘要:** We observe that current state-of-the-art web-agents are unable to effectively adapt to new environments without neural network fine-tuning, without which they produce inefficient execution plans due to a lack of awareness of the structure and dynamics of the new environment. To address this limitation, we introduce ATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented agent that is able to make plans grounded in a model of the environment by simulating the consequences of those actions in cognitive space. Our agent starts by building a "cognitive map" by performing a lightweight curiosity driven exploration of the environment. The planner proposes candidate actions; the simulator predicts their consequences in cognitive space; a critic analyzes the options to select the best roll-out and update the original plan; and a browser executor performs the chosen action. On the WebArena-Lite Benchmark, we achieve a 63% success rate compared to 53.9% success rate for the previously published state-of-the-art. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablations show sizable drops without the world-model, hierarchical planner, and look-ahead-based replanner confirming their complementary roles within the design of our system
>
---
#### [new 135] Mitigating Coordinate Prediction Bias from Positional Encoding Failures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对多模态大模型在高分辨率输入下坐标预测偏差问题，发现位置编码失效导致定向误差。提出VPSG方法，通过扰动位置编码生成负向证据，在不训练的前提下修正坐标预测，提升空间推理准确性。**

- **链接: [http://arxiv.org/pdf/2510.22102v1](http://arxiv.org/pdf/2510.22102v1)**

> **作者:** Xingjian Tao; Yiwei Wang; Yujun Cai; Yihong Luo; Jing Tang
>
> **摘要:** Multimodal large language models (MLLMs) excel at vision-language tasks such as VQA and document understanding, yet precise coordinate prediction remains challenging. High-resolution inputs exacerbate this difficulty by producing long token sequences that weaken positional encodings and introduce directional biases in coordinate outputs. We investigate this phenomenon by analyzing how MLLMs behave when visual positional encodings (VPEs) are deliberately perturbed through shuffling. Our analysis reveals that such perturbations induce predictable, non-random coordinate biases rather than random errors, suggesting that models rely on internal positional priors when spatial grounding signals are degraded. Crucially, we observe similar directional error patterns in natural high-resolution datasets, indicating that positional encoding failures are a key bottleneck for accurate coordinate prediction at scale. To address this issue, we propose Vision-PE Shuffle Guidance (VPSG), a training-free test-time method that leverages the directional nature of these biases for correction. VPSG runs auxiliary decoding with shuffled VPEs to isolate position-unconditioned tendencies, then uses this as negative evidence to guide digit prediction while preserving coordinate format through a lightweight finite-state machine. Experiments on ScreenSpot-Pro demonstrate reliable improvements, highlighting positional encoding robustness as a critical factor for spatial reasoning in MLLMs.
>
---
#### [new 136] ReCode: Unify Plan and Action for Universal Granularity Control
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ReCode，一种统一规划与行动的递归代码生成范式，解决大语言模型在多粒度决策中灵活性不足的问题。通过将高层计划视为可递归分解的抽象函数，实现动态粒度控制，提升推理性能与训练效率，推动智能体在复杂任务中的通用性。**

- **链接: [http://arxiv.org/pdf/2510.23564v1](http://arxiv.org/pdf/2510.23564v1)**

> **作者:** Zhaoyang Yu; Jiayi Zhang; Huixue Su; Yufan Zhao; Yifan Wu; Mingyi Deng; Jinyu Xiang; Yizhang Lin; Lingxiao Tang; Yingchao Li; Yuyu Luo; Bang Liu; Chenglin Wu
>
> **摘要:** Real-world tasks require decisions at varying granularities, and humans excel at this by leveraging a unified cognitive representation where planning is fundamentally understood as a high-level form of action. However, current Large Language Model (LLM)-based agents lack this crucial capability to operate fluidly across decision granularities. This limitation stems from existing paradigms that enforce a rigid separation between high-level planning and low-level action, which impairs dynamic adaptability and limits generalization. We propose ReCode (Recursive Code Generation), a novel paradigm that addresses this limitation by unifying planning and action within a single code representation. In this representation, ReCode treats high-level plans as abstract placeholder functions, which the agent then recursively decomposes into finer-grained sub-functions until reaching primitive actions. This recursive approach dissolves the rigid boundary between plan and action, enabling the agent to dynamically control its decision granularity. Furthermore, the recursive structure inherently generates rich, multi-granularity training data, enabling models to learn hierarchical decision-making processes. Extensive experiments show ReCode significantly surpasses advanced baselines in inference performance and demonstrates exceptional data efficiency in training, validating our core insight that unifying planning and action through recursive code generation is a powerful and effective approach to achieving universal granularity control. The code is available at https://github.com/FoundationAgents/ReCode.
>
---
#### [new 137] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦多模态抽象推理任务，针对当前模型在处理多模态关系知识（MMRK）时的不足，提出自动数据生成引擎与两阶段增强框架，构建了STAR-64K数据集。实验表明，小模型经训练后可超越GPT-4o，显著提升结构化抽象推理能力。**

- **链接: [http://arxiv.org/pdf/2510.21828v1](http://arxiv.org/pdf/2510.21828v1)**

> **作者:** Yichi Zhang; Zhuo Chen; Lingbing Guo; Lei Liang; Wen Zhang; Huajun Chen
>
> **备注:** Work in Progress. Code and data will be released at https://github.com/zjukg/STAR
>
> **摘要:** Understanding and reasoning with abstractive information from the visual modality presents significant challenges for current multi-modal large language models (MLLMs). Among the various forms of abstractive information, Multi-Modal Relational Knowledge (MMRK), which represents abstract relational structures between multi-modal entities using node-edge formats, remains largely under-explored. In particular, STructured and Abstractive Reasoning (STAR) on such data has received little attention from the research community. To bridge the dual gaps in large-scale high-quality data and capability enhancement methodologies, this paper makes the following key contributions: (i). An automatic STAR data engine capable of synthesizing images with MMRK to build multi-modal instruction data with reliable chain-of-thought thinking for various STAR tasks and (ii). A comprehsive two-stage capability enhancement training framework, accompanied by a suite of evaluation protocols tailored to different STAR tasks. Based upon these contributions, we introduce STAR-64K, a dataset comprising 64K high-quality multi-modal instruction samples, and conduct experiments across 5 open-source MLLMs. Experimental results show that our two-stage enhancement framework enables smaller 3B/7B models to significantly outperform GPT-4o in STAR. Additionally, we provide in-depth analysis regarding the effectiveness of various designs, data transferability, and scalability.
>
---
#### [new 138] Mapping Faithful Reasoning in Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对语言模型推理过程中的“忠实性”问题，提出Concept Walk框架，通过激活空间中概念方向的投影，分析推理步骤是否真正影响模型决策。研究发现，在简单任务中推理链常为装饰性，在复杂任务中则具实质影响，从而帮助识别可信推理。属于模型可解释性任务。**

- **链接: [http://arxiv.org/pdf/2510.22362v1](http://arxiv.org/pdf/2510.22362v1)**

> **作者:** Jiazheng Li; Andreas Damianou; J Rosser; José Luis Redondo García; Konstantina Palla
>
> **备注:** 9 pages, Accepted to the Mechanistic Interpretability Workshop at NeurIPS 2025
>
> **摘要:** Chain-of-thought (CoT) traces promise transparency for reasoning language models, but prior work shows they are not always faithful reflections of internal computation. This raises challenges for oversight: practitioners may misinterpret decorative reasoning as genuine. We introduce Concept Walk, a general framework for tracing how a model's internal stance evolves with respect to a concept direction during reasoning. Unlike surface text, Concept Walk operates in activation space, projecting each reasoning step onto the concept direction learned from contrastive data. This allows us to observe whether reasoning traces shape outcomes or are discarded. As a case study, we apply Concept Walk to the domain of Safety using Qwen 3-4B. We find that in 'easy' cases, perturbed CoTs are quickly ignored, indicating decorative reasoning, whereas in 'hard' cases, perturbations induce sustained shifts in internal activations, consistent with faithful reasoning. The contribution is methodological: Concept Walk provides a lens to re-examine faithfulness through concept-specific internal dynamics, helping identify when reasoning traces can be trusted and when they risk misleading practitioners.
>
---
#### [new 139] Transformer Based Linear Attention with Optimized GPU Kernel Implementation
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对Transformer中注意力机制计算效率低的问题，提出基于Transformer的线性注意力（LA）方法及优化的CUDA实现。通过改进前向与反向传播算法，显著提升速度（快3.3倍）并降低内存占用（减少3.6倍），在语言模型训练中达到与原生注意力相当的性能。**

- **链接: [http://arxiv.org/pdf/2510.21956v1](http://arxiv.org/pdf/2510.21956v1)**

> **作者:** Armin Gerami; Ramani Duraiswami
>
> **摘要:** The original softmax-based attention mechanism (regular attention) in the extremely successful Transformer architecture computes attention between $N$ tokens, each embedded in a $D$-dimensional head, with a time complexity of $O(N^2D)$. Given the success of Transformers, improving their runtime during both training and inference is a popular research area. One such approach is the introduction of the linear attention (LA) mechanisms, which offers a linear time complexity of $O(ND^2)$ and have demonstrated comparable accuracy to regular attention. However, LA in practice lags behind its theoretical efficiency. We propose a novel method for LA's forward and backward passes, along with a highly-optimized CUDA implementation. Our approach outperforms the state-of-the-art by 3.3 times in speed and reduces memory consumption by 3.6 times. We validate these improvements in both single-layer and end-to-end settings by training a 1.4 billion parameter language model, which demonstrates similar expressivity to regular attention on major reasoning benchmarks.
>
---
#### [new 140] A Benchmark for Open-Domain Numerical Fact-Checking Enhanced by Claim Decomposition
- **分类: cs.IR; cs.CL**

- **简介: 该论文聚焦开放域数值事实核查任务，针对现有基准数据集证据相关性差、存在时间泄漏等问题，提出QuanTemp++数据集。通过模拟人类分解声明的检索过程，收集无时间泄漏的相关证据，评估不同分解方法对核查效果的影响，推动更真实、高效的自动事实核查发展。**

- **链接: [http://arxiv.org/pdf/2510.22055v1](http://arxiv.org/pdf/2510.22055v1)**

> **作者:** V Venktesh; Deepali Prabhu; Avishek Anand
>
> **备注:** 16 pages
>
> **摘要:** Fact-checking numerical claims is critical as the presence of numbers provide mirage of veracity despite being fake potentially causing catastrophic impacts on society. The prior works in automatic fact verification do not primarily focus on natural numerical claims. A typical human fact-checker first retrieves relevant evidence addressing the different numerical aspects of the claim and then reasons about them to predict the veracity of the claim. Hence, the search process of a human fact-checker is a crucial skill that forms the foundation of the verification process. Emulating a real-world setting is essential to aid in the development of automated methods that encompass such skills. However, existing benchmarks employ heuristic claim decomposition approaches augmented with weakly supervised web search to collect evidences for verifying claims. This sometimes results in less relevant evidences and noisy sources with temporal leakage rendering a less realistic retrieval setting for claim verification. Hence, we introduce QuanTemp++: a dataset consisting of natural numerical claims, an open domain corpus, with the corresponding relevant evidence for each claim. The evidences are collected through a claim decomposition process approximately emulating the approach of human fact-checker and veracity labels ensuring there is no temporal leakage. Given this dataset, we also characterize the retrieval performance of key claim decomposition paradigms. Finally, we observe their effect on the outcome of the verification pipeline and draw insights. The code for data pipeline along with link to data can be found at https://github.com/VenkteshV/QuanTemp_Plus
>
---
#### [new 141] Surface Reading LLMs: Synthetic Text and its Styles
- **分类: cs.CY; cs.CL**

- **简介: 该论文探讨大语言模型生成文本的表层风格，提出“表面完整性”语义学框架，旨在揭示其在话语中作为文化主体的建构作用。通过分析合成文本的风格特征，强调表层风格分析对理解LLMs影响意义生成与传播的重要性，解决深度批判忽视表层现象的问题。**

- **链接: [http://arxiv.org/pdf/2510.22162v1](http://arxiv.org/pdf/2510.22162v1)**

> **作者:** Hannes Bajohr
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epist\=em\=e, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural actors that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness.
>
---
#### [new 142] LOC: A General Language-Guided Framework for Open-Set 3D Occupancy Prediction
- **分类: cs.CV; cs.CL; cs.LG; cs.RO; eess.IV**

- **简介: 该论文提出LOC框架，解决3D场景理解中因数据稀缺导致的开放集占用预测难题。通过语言引导融合多帧激光雷达点云与语义信息，结合对比学习增强特征区分性，实现无需额外训练即可识别未知类别的高精度3D占用预测。**

- **链接: [http://arxiv.org/pdf/2510.22141v1](http://arxiv.org/pdf/2510.22141v1)**

> **作者:** Yuhang Gao; Xiang Xiang; Sheng Zhong; Guoyou Wang
>
> **摘要:** Vision-Language Models (VLMs) have shown significant progress in open-set challenges. However, the limited availability of 3D datasets hinders their effective application in 3D scene understanding. We propose LOC, a general language-guided framework adaptable to various occupancy networks, supporting both supervised and self-supervised learning paradigms. For self-supervised tasks, we employ a strategy that fuses multi-frame LiDAR points for dynamic/static scenes, using Poisson reconstruction to fill voids, and assigning semantics to voxels via K-Nearest Neighbor (KNN) to obtain comprehensive voxel representations. To mitigate feature over-homogenization caused by direct high-dimensional feature distillation, we introduce Densely Contrastive Learning (DCL). DCL leverages dense voxel semantic information and predefined textual prompts. This efficiently enhances open-set recognition without dense pixel-level supervision, and our framework can also leverage existing ground truth to further improve performance. Our model predicts dense voxel features embedded in the CLIP feature space, integrating textual and image pixel information, and classifies based on text and semantic similarity. Experiments on the nuScenes dataset demonstrate the method's superior performance, achieving high-precision predictions for known classes and distinguishing unknown classes without additional training data.
>
---
#### [new 143] UniAIDet: A Unified and Universal Benchmark for AI-Generated Image Content Detection and Localization
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UniAIDet，一个统一的AI生成图像内容检测与定位基准。针对现有基准覆盖模型和图像类型有限的问题，构建涵盖多种生成模型与图像类别的全面数据集，并评估检测方法的泛化能力与检测-定位关系，推动AI生成内容检测研究发展。**

- **链接: [http://arxiv.org/pdf/2510.23023v1](http://arxiv.org/pdf/2510.23023v1)**

> **作者:** Huixuan Zhang; Xiaojun Wan
>
> **摘要:** With the rapid proliferation of image generative models, the authenticity of digital images has become a significant concern. While existing studies have proposed various methods for detecting AI-generated content, current benchmarks are limited in their coverage of diverse generative models and image categories, often overlooking end-to-end image editing and artistic images. To address these limitations, we introduce UniAIDet, a unified and comprehensive benchmark that includes both photographic and artistic images. UniAIDet covers a wide range of generative models, including text-to-image, image-to-image, image inpainting, image editing, and deepfake models. Using UniAIDet, we conduct a comprehensive evaluation of various detection methods and answer three key research questions regarding generalization capability and the relation between detection and localization. Our benchmark and analysis provide a robust foundation for future research.
>
---
#### [new 144] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文针对多模态大模型生成中检索策略僵化、模态选择不灵活、信息利用低效的问题，提出Windsock框架实现查询相关的检索决策与模态选择，并引入动态抗噪训练与自评估数据构建方法，显著提升生成质量并降低检索开销。**

- **链接: [http://arxiv.org/pdf/2510.22694v1](http://arxiv.org/pdf/2510.22694v1)**

> **作者:** Shu Zhao; Tianyi Shen; Nilesh Ahuja; Omesh Tickoo; Vijaykrishnan Narayanan
>
> **备注:** Accepted at NeurIPS 2025 UniReps Workshop
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) has emerged as a promising method to generate factual and up-to-date responses of Multimodal Large Language Models (MLLMs) by incorporating non-parametric knowledge from external knowledge bases. However, existing MRAG approaches suffer from static retrieval strategies, inflexible modality selection, and suboptimal utilization of retrieved information, leading to three critical challenges: determining when to retrieve, what modality to incorporate, and how to utilize retrieved information effectively. To address these challenges, we introduce Windsock, a query-dependent module making decisions on retrieval necessity and modality selection, effectively reducing computational overhead and improving response quality. Additionally, we propose Dynamic Noise-Resistance (DANCE) Instruction Tuning, an adaptive training strategy that enhances MLLMs' ability to utilize retrieved information while maintaining robustness against noise. Moreover, we adopt a self-assessment approach leveraging knowledge within MLLMs to convert question-answering datasets to MRAG training datasets. Extensive experiments demonstrate that our proposed method significantly improves the generation quality by 17.07% while reducing 8.95% retrieval times.
>
---
#### [new 145] Do Stop Me Now: Detecting Boilerplate Responses with a Single Iteration
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大模型生成冗余响应（如拒绝、问候）导致的资源浪费问题，提出仅通过首个词元的对数概率分布，用轻量k-NN分类器实时判断响应类型。工作实现高效早期终止或模型切换，显著降低计算成本，提升推理效率。属于大模型推理优化任务。**

- **链接: [http://arxiv.org/pdf/2510.22679v1](http://arxiv.org/pdf/2510.22679v1)**

> **作者:** Yuval Kainan; Shaked Zychlinski
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Large Language Models (LLMs) often expend significant computational resources generating boilerplate responses, such as refusals, simple acknowledgements and casual greetings, which adds unnecessary cost and latency. To address this inefficiency, we propose a simple yet highly effective method for detecting such responses after only a single generation step. We demonstrate that the log-probability distribution of the first generated token serves as a powerful signal for classifying the nature of the entire subsequent response. Our experiments, conducted across a diverse range of small, large, and reasoning-specialized models, show that the first-token log-probability vectors form distinctly separable clusters for different response types. Using a lightweight k-NN classifier, we achieve high accuracy in predicting whether a response will be a substantive answer or a form of boilerplate response, including user-specified refusals. The primary implication is a practical, computationally trivial technique, optimizing LLM inference by enabling early termination or redirection to a smaller model, thereby yielding significant savings in computational cost. This work presents a direct path toward more efficient and sustainable LLM deployment.
>
---
#### [new 146] VITA-E: Natural Embodied Interaction with Concurrent Seeing, Hearing, Speaking, and Acting
- **分类: cs.RO; cs.CL; cs.LG**

- **简介: 该论文提出VITA-E框架，解决现有视觉-语言-动作模型无法并发处理感知、听觉、语言与行动的问题。通过双模型架构与“模型即控制器”机制，实现人机交互中多任务并发与实时中断响应，提升机器人自然协同能力。**

- **链接: [http://arxiv.org/pdf/2510.21817v1](http://arxiv.org/pdf/2510.21817v1)**

> **作者:** Xiaoyu Liu; Chaoyou Fu; Chi Yan; Chu Wu; Haihan Gao; Yi-Fan Zhang; Shaoqi Dong; Cheng Qian; Bin Luo; Xiuyong Yang; Guanwu Li; Yusheng Cai; Yunhang Shen; Deqiang Jiang; Haoyu Cao; Xing Sun; Caifeng Shan; Ran He
>
> **备注:** Homepage: https://lxysl.github.io/VITA-E/
>
> **摘要:** Current Vision-Language-Action (VLA) models are often constrained by a rigid, static interaction paradigm, which lacks the ability to see, hear, speak, and act concurrently as well as handle real-time user interruptions dynamically. This hinders seamless embodied collaboration, resulting in an inflexible and unresponsive user experience. To address these limitations, we introduce VITA-E, a novel embodied interaction framework designed for both behavioral concurrency and nearly real-time interruption. The core of our approach is a dual-model architecture where two parallel VLA instances operate as an ``Active Model'' and a ``Standby Model'', allowing the embodied agent to observe its environment, listen to user speech, provide verbal responses, and execute actions, all concurrently and interruptibly, mimicking human-like multitasking capabilities. We further propose a ``model-as-controller'' paradigm, where we fine-tune the VLM to generate special tokens that serve as direct system-level commands, coupling the model's reasoning with the system's behavior. Experiments conducted on a physical humanoid platform demonstrate that VITA-E can reliably handle complex interactive scenarios. Our framework is compatible with various dual-system VLA models, achieving an extremely high success rate on emergency stops and speech interruptions while also successfully performing concurrent speech and action. This represents a significant step towards more natural and capable embodied assistants.
>
---
#### [new 147] The Mirror Loop: Recursive Non-Convergence in Generative Reasoning Systems
- **分类: cs.LG; cs.AI; cs.CL; 68T05; I.2.6; I.2.8**

- **简介: 该论文研究生成式推理系统中递归自省的非收敛问题。针对“无外部反馈时自我修正为何失效”这一问题，通过跨模型实验验证了自洽反思易陷入信息封闭（镜像环）。引入最小接地干预后，信息量回升，揭示了自主推理的结构性局限，并提出需通过与外部验证交互来维持认知动态。**

- **链接: [http://arxiv.org/pdf/2510.21861v1](http://arxiv.org/pdf/2510.21861v1)**

> **作者:** Bentley DeVilling
>
> **备注:** 18 pages, 2 figures. Category: cs.LG. Code and data: https://github.com/Course-Correct-Labs/mirror-loop
>
> **摘要:** Large language models are often described as capable of reflective reasoning, yet recursive self-evaluation without external feedback frequently yields reformulation rather than progress. We test this prediction in a cross-provider study of 144 reasoning sequences across three models (OpenAI GPT-4o-mini, Anthropic Claude 3 Haiku, and Google Gemini 2.0 Flash) and four task families (arithmetic, code, explanation, reflection), each iterated ten times under two conditions: ungrounded self-critique and a minimal grounding intervention (a single verification step at iteration three). Mean informational change (delta I, measured via normalized edit distance) declined by 55% from early (0.193) to late (0.087) iterations in ungrounded runs, with consistent patterns across all three providers. Grounded runs showed a +28% rebound in informational change immediately after the intervention and sustained non-zero variance thereafter. Complementary measures-n-gram novelty, embedding drift, and character-level entropy-converged on the same pattern: reflection without contact tends toward informational closure. We interpret this as evidence for a structural limit on self-correction in generative reasoning: without an exchange of information with an independent verifier or environment, recursive inference approaches an attractor state of epistemic stasis. Minimal grounding functions as dissipative coupling, reintroducing informational flux. The cross-architecture consistency suggests the mirror loop arises from shared autoregressive training objectives rather than provider-specific alignment schemes. The results delineate when reflection is performative rather than epistemic and motivate design principles for grounded, cooperative reasoning. Materials and code are publicly available.
>
---
#### [new 148] Performance Trade-offs of Optimizing Small Language Models for E-Commerce
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究小规模语言模型在电商领域的优化，旨在降低大模型部署的高成本。针对电商意图识别任务，采用QLoRA微调与量化技术，构建10亿参数模型，实现99%准确率，媲美大模型。对比GPTQ与GGUF格式，揭示硬件依赖的性能权衡，证明优化后的小模型在资源效率与精度上更具优势。**

- **链接: [http://arxiv.org/pdf/2510.21970v1](http://arxiv.org/pdf/2510.21970v1)**

> **作者:** Josip Tomo Licardo; Nikola Tankovic
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) offer state-of-the-art performance in natural language understanding and generation tasks. However, the deployment of leading commercial models for specialized tasks, such as e-commerce, is often hindered by high computational costs, latency, and operational expenses. This paper investigates the viability of smaller, open-weight models as a resource-efficient alternative. We present a methodology for optimizing a one-billion-parameter Llama 3.2 model for multilingual e-commerce intent recognition. The model was fine-tuned using Quantized Low-Rank Adaptation (QLoRA) on a synthetically generated dataset designed to mimic real-world user queries. Subsequently, we applied post-training quantization techniques, creating GPU-optimized (GPTQ) and CPU-optimized (GGUF) versions. Our results demonstrate that the specialized 1B model achieves 99% accuracy, matching the performance of the significantly larger GPT-4.1 model. A detailed performance analysis revealed critical, hardware-dependent trade-offs: while 4-bit GPTQ reduced VRAM usage by 41%, it paradoxically slowed inference by 82% on an older GPU architecture (NVIDIA T4) due to dequantization overhead. Conversely, GGUF formats on a CPU achieved a speedup of up to 18x in inference throughput and a reduction of over 90% in RAM consumption compared to the FP16 baseline. We conclude that small, properly optimized open-weight models are not just a viable but a more suitable alternative for domain-specific applications, offering state-of-the-art accuracy at a fraction of the computational cost.
>
---
#### [new 149] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
- **分类: cs.CV; cs.CL; cs.RO; I.2.10; I.2.9; I.2.7; H.5.2**

- **简介: 该论文提出Look and Tell数据集，用于研究第一人称与第三人称视角下的多模态语义对齐。针对跨视角指代理解难题，通过同步记录眼动、语音与视频，结合3D场景重建，提供2.7k条标注的指代表达，推动具身智能体在情境对话中的理解能力发展。**

- **链接: [http://arxiv.org/pdf/2510.22672v1](http://arxiv.org/pdf/2510.22672v1)**

> **作者:** Anna Deichler; Jonas Beskow
>
> **备注:** 10 pages, 6 figures, 2 tables. Accepted to the NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE)
>
> **摘要:** We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
>
---
#### [new 150] Can Language Models Compose Skills In-Context?
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型在上下文中的技能组合能力，旨在解决如何通过示例引导模型完成复合任务的问题。通过设计语言与逻辑任务进行系统实验，发现简单示例可能产生负面效果，提出需对齐示例与组合步骤的方法，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.22993v1](http://arxiv.org/pdf/2510.22993v1)**

> **作者:** Zidong Liu; Zhuoyan Xu; Zhenmei Shi; Yingyu Liang
>
> **摘要:** Composing basic skills from simple tasks to accomplish composite tasks is crucial for modern intelligent systems. We investigate the in-context composition ability of language models to perform composite tasks that combine basic skills demonstrated in in-context examples. This is more challenging than the standard setting, where skills and their composition can be learned in training. We conduct systematic experiments on various representative open-source language models, utilizing linguistic and logical tasks designed to probe composition abilities. The results reveal that simple task examples can have a surprising negative impact on the performance, because the models generally struggle to recognize and assemble the skills correctly, even with Chain-of-Thought examples. Theoretical analysis further shows that it is crucial to align examples with the corresponding steps in the composition. This inspires a method for the probing tasks, whose improved performance provides positive support for our insights.
>
---
#### [new 151] Label Smoothing Improves Gradient Ascent in LLM Unlearning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大模型遗忘任务，针对梯度上升（GA）方法在遗忘过程中导致模型性能严重下降的问题，提出平滑梯度上升（SGA）方法。通过融合遗忘数据与构造的正常数据，并引入可调平滑率，提升遗忘稳定性与模型实用性。理论指导最优平滑率选择，实验验证其在多个基准上优于原有方法。**

- **链接: [http://arxiv.org/pdf/2510.22376v1](http://arxiv.org/pdf/2510.22376v1)**

> **作者:** Zirui Pang; Hao Zheng; Zhijie Deng; Ling Li; Zixin Zhong; Jiaheng Wei
>
> **摘要:** LLM unlearning has emerged as a promising approach, aiming to enable models to forget hazardous/undesired knowledge at low cost while preserving as much model utility as possible. Among existing techniques, the most straightforward method is performing Gradient Ascent (GA) w.r.t. the forget data, thereby forcing the model to unlearn the forget dataset. However, GA suffers from severe instability, as it drives updates in a divergent direction, often resulting in drastically degraded model utility. To address this issue, we propose Smoothed Gradient Ascent (SGA). SGA combines the forget data with multiple constructed normal data through a tunable smoothing rate. Intuitively, this extends GA from learning solely on the forget data to jointly learning across both forget and normal data, enabling more stable unlearning while better preserving model utility. Theoretically, we provide the theoretical guidance on the selection of the optimal smoothing rate. Empirically, we evaluate SGA on three benchmarks: TOFU, Harry Potter, and MUSE-NEWS. Experimental results demonstrate that SGA consistently outperforms the original Gradient Ascent (GA) method across all metrics and achieves top-2 performance among all baseline methods on several key metrics.
>
---
#### [new 152] VietLyrics: A Large-Scale Dataset and Models for Vietnamese Automatic Lyrics Transcription
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦越南语自动歌词转录（ALT）任务，针对其声调复杂、方言多样且缺乏专用数据集的问题，构建了首个大规模越南语ALT数据集VietLyrics（647小时），并基于此微调Whisper模型，显著提升转录性能，优于现有系统。研究推动了低资源语言音乐计算发展。**

- **链接: [http://arxiv.org/pdf/2510.22295v1](http://arxiv.org/pdf/2510.22295v1)**

> **作者:** Quoc Anh Nguyen; Bernard Cheng; Kelvin Soh
>
> **摘要:** Automatic Lyrics Transcription (ALT) for Vietnamese music presents unique challenges due to its tonal complexity and dialectal variations, but remains largely unexplored due to the lack of a dedicated dataset. Therefore, we curated the first large-scale Vietnamese ALT dataset (VietLyrics), comprising 647 hours of songs with line-level aligned lyrics and metadata to address these issues. Our evaluation of current ASRbased approaches reveal significant limitations, including frequent transcription errors and hallucinations in non-vocal segments. To improve performance, we fine-tuned Whisper models on the VietLyrics dataset, achieving superior results compared to existing multilingual ALT systems, including LyricWhiz. We publicly release VietLyrics and our models, aiming to advance Vietnamese music computing research while demonstrating the potential of this approach for ALT in low-resource language and music.
>
---
#### [new 153] When Robots Say No: Temporal Trust Recovery Through Explanation
- **分类: cs.HC; cs.CL; cs.RO**

- **简介: 该论文研究人机协作中机器人拒绝请求时的信任恢复问题。针对高风险任务中机器人因自主决策导致用户信任受损的问题，通过模拟灭火场景实验发现：提供合理解释可促进信任随时间恢复，有效缓解信任危机。**

- **链接: [http://arxiv.org/pdf/2510.21716v1](http://arxiv.org/pdf/2510.21716v1)**

> **作者:** Nicola Webb; Zijun Huang; Sanja Milivojevic; Chris Baber; Edmund R. Hunt
>
> **摘要:** Mobile robots with some degree of autonomy could deliver significant advantages in high-risk missions such as search and rescue and firefighting. Integrated into a human-robot team (HRT), robots could work effectively to help search hazardous buildings. User trust is a key enabler for HRT, but during a mission, trust can be damaged. With distributed situation awareness, such as when team members are working in different locations, users may be inclined to doubt a robot's integrity if it declines to immediately change its priorities on request. In this paper, we present the results of a computer-based study investigating on-mission trust dynamics in a high-stakes human-robot teaming scenario. Participants (n = 38) played an interactive firefighting game alongside a robot teammate, where a trust violation occurs owing to the robot declining to help the user immediately. We find that when the robot provides an explanation for declining to help, trust better recovers over time, albeit following an initial drop that is comparable to a baseline condition where an explanation for refusal is not provided. Our findings indicate that trust can vary significantly during a mission, notably when robots do not immediately respond to user requests, but that this trust violation can be largely ameliorated over time if adequate explanation is provided.
>
---
#### [new 154] Reasoning Models Reason Well, Until They Don't
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在复杂推理任务中的表现。针对现有模型在高复杂度问题上突然失效的问题，作者构建了可扩展复杂度的DeepRD数据集，发现大推理模型性能随复杂度上升急剧下降，无法泛化。研究揭示了模型在真实世界复杂场景下的潜在失败风险，强调需发展更鲁棒的推理方法。**

- **链接: [http://arxiv.org/pdf/2510.22371v1](http://arxiv.org/pdf/2510.22371v1)**

> **作者:** Revanth Rameshkumar; Jimson Huang; Yunxin Sun; Fei Xia; Abulhair Saparov
>
> **摘要:** Large language models (LLMs) have shown significant progress in reasoning tasks. However, recent studies show that transformers and LLMs fail catastrophically once reasoning problems exceed modest complexity. We revisit these findings through the lens of large reasoning models (LRMs) -- LLMs fine-tuned with incentives for step-by-step argumentation and self-verification. LRM performance on graph and reasoning benchmarks such as NLGraph seem extraordinary, with some even claiming they are capable of generalized reasoning and innovation in reasoning-intensive fields such as mathematics, physics, medicine, and law. However, by more carefully scaling the complexity of reasoning problems, we show existing benchmarks actually have limited complexity. We develop a new dataset, the Deep Reasoning Dataset (DeepRD), along with a generative process for producing unlimited examples of scalable complexity. We use this dataset to evaluate model performance on graph connectivity and natural language proof planning. We find that the performance of LRMs drop abruptly at sufficient complexity and do not generalize. We also relate our LRM results to the distributions of the complexities of large, real-world knowledge graphs, interaction graphs, and proof datasets. We find the majority of real-world examples fall inside the LRMs' success regime, yet the long tails expose substantial failure potential. Our analysis highlights the near-term utility of LRMs while underscoring the need for new methods that generalize beyond the complexity of examples in the training distribution.
>
---
#### [new 155] LibriConvo: Simulating Conversations from Read Literature for ASR and Diarization
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出LibriConvo，一个基于文学文本的多说话人对话模拟数据集，用于语音识别（ASR）与说话人分离（Diarization）任务。针对现有数据语义断裂、时间间隔不真实的问题，通过语境一致的语音组织与空间合理的声音模拟，实现自然对话动态。实验表明其有效提升模型性能，为多说话人语音处理提供高质量基准。**

- **链接: [http://arxiv.org/pdf/2510.23320v1](http://arxiv.org/pdf/2510.23320v1)**

> **作者:** Máté Gedeon; Péter Mihajlik
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** We introduce LibriConvo, a simulated multi-speaker conversational dataset based on speaker-aware conversation simulation (SASC), designed to support training and evaluation of speaker diarization and automatic speech recognition (ASR) systems. Unlike prior resources that mostly rely on semantically disconnected utterances and implausible temporal gaps, LibriConvo ensures semantic coherence and realistic conversational timing. Our pipeline leverages CallHome with external VAD for reliable boundaries, applies compression to reduce unnaturally long silences, and organizes LibriTTS utterances by book to maintain contextual consistency. Acoustic realism is enhanced via a novel room impulse response selection procedure that ranks speaker-microphone configurations by spatial plausibility, balancing realism and diversity. The dataset comprises 240.1 hours across 1,496 dialogues with 830 unique speakers, split in a speaker-disjoint manner for robust evaluation. Baselines show that the sortformer model outperforms the pyannote pipeline in diarization, while a fine-tuned Fast Conformer-CTC XLarge with Serialized Output Training achieves 7.29\% WER for ASR, surpassing zero-shot Whisper-large-v3. LibriConvo provides a valuable resource for advancing multi-speaker speech processing research with realistic conversational dynamics and controlled experimental conditions.
>
---
#### [new 156] Beyond IVR Touch-Tones: Customer Intent Routing using LLMs
- **分类: cs.HC; cs.AI; cs.CL; eess.AS**

- **简介: 该论文针对传统IVR系统因固定按键导致用户体验差的问题，提出基于大语言模型（LLM）的用户意图路由方法。通过构建虚拟IVR结构与生成用户意图数据，比较不同提示设计的效果，验证了LLM在自然语言理解与路径匹配中的可行性，实现了更智能、无缝的客户语音服务引导。**

- **链接: [http://arxiv.org/pdf/2510.21715v1](http://arxiv.org/pdf/2510.21715v1)**

> **作者:** Sergio Rojas-Galeano
>
> **备注:** Accepted for publication in the Proceedings of the Workshop on Engineering Applications 2025 (WEA 2025)
>
> **摘要:** Widespread frustration with rigid touch-tone Interactive Voice Response (IVR) systems for customer service underscores the need for more direct and intuitive language interaction. While speech technologies are necessary, the key challenge lies in routing intents from user phrasings to IVR menu paths, a task where Large Language Models (LLMs) show strong potential. Progress, however, is limited by data scarcity, as real IVR structures and interactions are often proprietary. We present a novel LLM-based methodology to address this gap. Using three distinct models, we synthesized a realistic 23-node IVR structure, generated 920 user intents (230 base and 690 augmented), and performed the routing task. We evaluate two prompt designs: descriptive hierarchical menus and flattened path representations, across both base and augmented datasets. Results show that flattened paths consistently yield higher accuracy, reaching 89.13% on the base dataset compared to 81.30% with the descriptive format, while augmentation introduces linguistic noise that slightly reduces performance. Confusion matrix analysis further suggests that low-performing routes may reflect not only model limitations but also redundancies in menu design. Overall, our findings demonstrate proof-of-concept that LLMs can enable IVR routing through a smoother, more seamless user experience -- moving customer service one step ahead of touch-tone menus.
>
---
#### [new 157] UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models
- **分类: eess.AS; cs.CL**

- **简介: 该论文针对语音对话模型缺乏细粒度风格控制的问题，提出UltraVoice数据集，涵盖830小时多风格对话，支持情绪、语速、音量等六维控制。通过微调主流模型，显著提升风格可控性与对话能力，验证了数据集在语音合成与对话系统中的高价值。**

- **链接: [http://arxiv.org/pdf/2510.22588v1](http://arxiv.org/pdf/2510.22588v1)**

> **作者:** Wenming Tu; Guanrou Yang; Ruiqi Yan; Wenxi Chen; Ziyang Ma; Yipeng Kang; Kai Yu; Xie Chen; Zilong Zheng
>
> **备注:** 23 pages, 4 figures
>
> **摘要:** Spoken dialogue models currently lack the ability for fine-grained speech style control, a critical capability for human-like interaction that is often overlooked in favor of purely functional capabilities like reasoning and question answering. To address this limitation, we introduce UltraVoice, the first large-scale speech dialogue dataset engineered for multiple fine-grained speech style control. Encompassing over 830 hours of speech dialogues, UltraVoice provides instructions across six key speech stylistic dimensions: emotion, speed, volume, accent, language, and composite styles. Fine-tuning leading models such as SLAM-Omni and VocalNet on UltraVoice significantly enhances their fine-grained speech stylistic controllability without degrading core conversational abilities. Specifically, our fine-tuned models achieve improvements of 29.12-42.33% in Mean Opinion Score (MOS) and 14.61-40.09 percentage points in Instruction Following Rate (IFR) on multi-dimensional control tasks designed in the UltraVoice. Moreover, on the URO-Bench benchmark, our fine-tuned models demonstrate substantial gains in core understanding, reasoning, and conversational abilities, with average improvements of +10.84% on the Basic setting and +7.87% on the Pro setting. Furthermore, the dataset's utility extends to training controllable Text-to-Speech (TTS) models, underscoring its high quality and broad applicability for expressive speech synthesis. The complete dataset and model checkpoints are available at: https://github.com/bigai-nlco/UltraVoice.
>
---
#### [new 158] Agentic Reinforcement Learning for Real-World Code Repair
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文聚焦真实代码修复任务，针对复杂构建与依赖漂移导致评估不稳的问题，提出可验证的修复流水线并实现大规模强化学习。通过依赖固化与简化环境，训练出高效小模型，显著提升修复成功率，但模型泛化能力有限，强调训练测试环境一致性对构建可靠代码修复代理的重要性。**

- **链接: [http://arxiv.org/pdf/2510.22075v1](http://arxiv.org/pdf/2510.22075v1)**

> **作者:** Siyu Zhu; Anastasiya Karpovich; Albert Chen; Jessica Koscheka; Shailesh Jannu; Di Wen; Yuqing Zhu; Rohit Jain; Alborz Geramifard
>
> **摘要:** We tackle the challenge of training reliable code-fixing agents in real repositories, where complex builds and shifting dependencies make evaluation unstable. We developed a verifiable pipeline with success defined as post-fix build validation and improved reproducibility across ~1K real issues by pinning dependencies and disabling automatic upgrades. Building on this, we introduced a scalable simplified pipeline for large-scale reinforcement learning (RL). Using this setup, we supervised fine-tuned Qwen3-32B in the full pipeline and applied RL on top of the SFT model in the simplified environment. The SFT model distilled from GPT-4.1 trajectories performs on par while being 56x smaller, and RL added 7-20% absolute gains under matched train-test conditions. "Thinking mode" was on par or worse in our experiments. Both SFT and RL models failed to generalize across environments, highlighting the importance of matching train-test environments for building reliable real-world code-fixing agents.
>
---
#### [new 159] The Lossy Horizon: Error-Bounded Predictive Coding for Lossy Text Compression (Episode I)
- **分类: cs.LG; cs.CL; cs.IT; math.IT; 94A08, 68P30, 68T50; E.4; I.2.7; I.2.7**

- **简介: 该论文研究损失性文本压缩任务，旨在提升压缩比的同时控制重建误差。提出误差有界的预测编码（EPC）方法，利用掩码语言模型作为解码器，仅存储预测错误时的修正信息，实现高效率-失真调控。实验表明EPC优于基线方法，在更低比特率下保持更高重建质量。**

- **链接: [http://arxiv.org/pdf/2510.22207v1](http://arxiv.org/pdf/2510.22207v1)**

> **作者:** Nnamdi Aghanya; Jun Li; Kewei Wang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) can achieve near-optimal lossless compression by acting as powerful probability models. We investigate their use in the lossy domain, where reconstruction fidelity is traded for higher compression ratios. This paper introduces Error-Bounded Predictive Coding (EPC), a lossy text codec that leverages a Masked Language Model (MLM) as a decompressor. Instead of storing a subset of original tokens, EPC allows the model to predict masked content and stores minimal, rank-based corrections only when the model's top prediction is incorrect. This creates a residual channel that offers continuous rate-distortion control. We compare EPC to a simpler Predictive Masking (PM) baseline and a transform-based Vector Quantisation with a Residual Patch (VQ+RE) approach. Through an evaluation that includes precise bit accounting and rate-distortion analysis, we demonstrate that EPC consistently dominates PM, offering superior fidelity at a significantly lower bit rate by more efficiently utilising the model's intrinsic knowledge.
>
---
#### [new 160] Modeling Political Discourse with Sentence-BERT and BERTopic
- **分类: cs.SI; cs.CL; cs.CY; 68T50, 91D30; I.2.7; H.3.1; J.4**

- **简介: 该论文属于社会媒体政治话语分析任务，旨在揭示政治话题的演变与道德基础的关系。通过结合BERTopic与道德基础理论，研究了美国第117届国会期间推特上的话题动态，发现核心议题稳定但细粒度话题易消散，且关怀与忠诚等道德维度显著影响话题持久性，不同党派采用差异化的道德叙事策略。**

- **链接: [http://arxiv.org/pdf/2510.22904v1](http://arxiv.org/pdf/2510.22904v1)**

> **作者:** Margarida Mendonca; Alvaro Figueira
>
> **备注:** 11 pages. Continues previous study by Mendonca M. and Figueira A, 2023: "Analyzing Political Discourse in the 117th U.S. Congress Using Transformer-Based Topic Models", presented at the International Conference on Computational Social Science
>
> **摘要:** Social media has reshaped political discourse, offering politicians a platform for direct engagement while reinforcing polarization and ideological divides. This study introduces a novel topic evolution framework that integrates BERTopic-based topic modeling with Moral Foundations Theory (MFT) to analyze the longevity and moral dimensions of political topics in Twitter activity during the 117th U.S. Congress. We propose a methodology for tracking dynamic topic shifts over time and measuring their association with moral values and quantifying topic persistence. Our findings reveal that while overarching themes remain stable, granular topics tend to dissolve rapidly, limiting their long-term influence. Moreover, moral foundations play a critical role in topic longevity, with Care and Loyalty dominating durable topics, while partisan differences manifest in distinct moral framing strategies. This work contributes to the field of social network analysis and computational political discourse by offering a scalable, interpretable approach to understanding moral-driven topic evolution on social media.
>
---
#### [new 161] WAON: Large-Scale and High-Quality Japanese Image-Text Pair Dataset for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出WAON，一个包含约1.55亿条日语图文对的大规模高质量数据集，用于提升视觉语言模型性能。针对日语文化图像识别任务，构建了手动标注的WAON-Bench评估基准。实验表明，基于WAON训练的模型在多项日语文化相关任务上优于现有方法，达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.22276v1](http://arxiv.org/pdf/2510.22276v1)**

> **作者:** Issa Sugiura; Shuhei Kurita; Yusuke Oda; Daisuke Kawahara; Yasuo Okabe; Naoaki Okazaki
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Large-scale and high-quality image-text pair datasets play an important role in developing high-performing Vision-Language Models (VLMs). In this work, we introduce WAON, a large-scale and high-quality Japanese image-text pair dataset containing approximately 155 million examples, collected from Common Crawl. Our dataset construction pipeline employs various techniques, including filtering and deduplication, which have been shown to be effective in previous studies. To evaluate its effectiveness, we also construct WAON-Bench, a manually curated benchmark for Japanese cultural image classification, consisting of 374 classes. To assess the effectiveness of our dataset, we conduct experiments using both WAON and the Japanese subset of ReLAION, one of the most widely used vision-language datasets. We fine-tune SigLIP2, a strong multilingual model, on both datasets. The results demonstrate that WAON enhances model performance on WAON-Bench more efficiently than ReLAION and achieves higher accuracy across all evaluated benchmarks. Furthermore, the model fine-tuned on WAON achieves state-of-the-art performance on several Japanese cultural benchmarks. We release our dataset, model, and code at https://speed1313.github.io/WAON.
>
---
#### [new 162] PaperAsk: A Benchmark for Reliability Evaluation of LLMs in Paper Search and Reading
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对LLM在学术搜索与阅读中的可靠性问题，提出PaperAsk基准，评估其在引文检索、内容提取、论文发现和论断验证四类任务的表现。实验发现模型普遍存在高错误率，尤其在多参考文献查询和特定章节提取中失败率超70%。研究还开发了轻量级可靠性分类器以识别不可靠输出，为提升学术辅助系统可靠性提供诊断框架。**

- **链接: [http://arxiv.org/pdf/2510.22242v1](http://arxiv.org/pdf/2510.22242v1)**

> **作者:** Yutao Wu; Xiao Liu; Yunhao Feng; Jiale Ding; Xingjun Ma
>
> **摘要:** Large Language Models (LLMs) increasingly serve as research assistants, yet their reliability in scholarly tasks remains under-evaluated. In this work, we introduce PaperAsk, a benchmark that systematically evaluates LLMs across four key research tasks: citation retrieval, content extraction, paper discovery, and claim verification. We evaluate GPT-4o, GPT-5, and Gemini-2.5-Flash under realistic usage conditions-via web interfaces where search operations are opaque to the user. Through controlled experiments, we find consistent reliability failures: citation retrieval fails in 48-98% of multi-reference queries, section-specific content extraction fails in 72-91% of cases, and topical paper discovery yields F1 scores below 0.32, missing over 60% of relevant literature. Further human analysis attributes these failures to the uncontrolled expansion of retrieved context and the tendency of LLMs to prioritize semantically relevant text over task instructions. Across basic tasks, the LLMs display distinct failure behaviors: ChatGPT often withholds responses rather than risk errors, whereas Gemini produces fluent but fabricated answers. To address these issues, we develop lightweight reliability classifiers trained on PaperAsk data to identify unreliable outputs. PaperAsk provides a reproducible and diagnostic framework for advancing the reliability evaluation of LLM-based scholarly assistance systems.
>
---
#### [new 163] RoboSVG: A Unified Framework for Interactive SVG Generation with Multi-modal Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出RoboSVG框架，解决多模态引导下交互式SVG生成问题。通过文本、图像、数值信号联合指导，实现高保真SVG合成。构建百万级数据集RoboDraw，支持四类任务，显著提升生成质量与查询一致性，推动交互式矢量图形生成发展。**

- **链接: [http://arxiv.org/pdf/2510.22684v1](http://arxiv.org/pdf/2510.22684v1)**

> **作者:** Jiuniu Wang; Gongjie Zhang; Quanhao Qian; Junlong Gao; Deli Zhao; Ran Xu
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Scalable Vector Graphics (SVGs) are fundamental to digital design and robot control, encoding not only visual structure but also motion paths in interactive drawings. In this work, we introduce RoboSVG, a unified multimodal framework for generating interactive SVGs guided by textual, visual, and numerical signals. Given an input query, the RoboSVG model first produces multimodal guidance, then synthesizes candidate SVGs through dedicated generation modules, and finally refines them under numerical guidance to yield high-quality outputs. To support this framework, we construct RoboDraw, a large-scale dataset of one million examples, each pairing an SVG generation condition (e.g., text, image, and partial SVG) with its corresponding ground-truth SVG code. RoboDraw dataset enables systematic study of four tasks, including basic generation (Text-to-SVG, Image-to-SVG) and interactive generation (PartialSVG-to-SVG, PartialImage-to-SVG). Extensive experiments demonstrate that RoboSVG achieves superior query compliance and visual fidelity across tasks, establishing a new state of the art in versatile SVG generation. The dataset and source code of this project will be publicly available soon.
>
---
#### [new 164] SIGN: Schema-Induced Games for Naming
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; I.2; I.2.7; I.2.11**

- **简介: 该论文提出SIGN框架，研究多智能体协作中的命名一致性问题。针对大模型代理间因缺乏统一约定导致沟通失效的问题，引入轻量级结构引导命名游戏，显著提升共识速度与一致率（最高快5.8倍），验证了最小结构对高效多智能体协调的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21855v1](http://arxiv.org/pdf/2510.21855v1)**

> **作者:** Ryan Zhang; Herbert Woisetscläger
>
> **备注:** AAAI 2026 Student Abstract (Oral). Code available ar https://github.com/ryanzhangofficial/schema-induced-games-for-naming
>
> **摘要:** Real-world AI systems are tackling increasingly complex problems, often through interactions among large language model (LLM) agents. When these agents develop inconsistent conventions, coordination can break down. Applications such as collaborative coding and distributed planning therefore require reliable, consistent communication, and scalability is a central concern as systems grow. We introduce Schema-Induced Games for Naming (SIGN), a naming game that examines how lightweight structure can steer convention formation. We compare schema-induced communication to unconstrained natural language and find faster convergence with up to 5.8x higher agreement. These results suggest that minimal structure can act as a simple control knob for efficient multi-agent coordination, pointing toward broader applications beyond the naming game.
>
---
#### [new 165] JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.SE**

- **简介: 该论文提出JanusCoder，面向代码智能的视觉-程序化接口，解决多模态代码数据稀缺与模型专用化问题。构建大规模多模态代码数据集JanusCode-800K，训练统一模型实现文本、视觉或混合输入生成代码，显著提升编码任务性能。**

- **链接: [http://arxiv.org/pdf/2510.23538v1](http://arxiv.org/pdf/2510.23538v1)**

> **作者:** Qiushi Sun; Jingyang Gong; Yang Liu; Qiaosheng Chen; Lei Li; Kai Chen; Qipeng Guo; Ben Kao; Fei Yuan
>
> **备注:** Work in progress
>
> **摘要:** The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints will are available at https://github.com/InternLM/JanusCoder.
>
---
#### [new 166] Embracing Trustworthy Brain-Agent Collaboration as Paradigm Extension for Intelligent Assistive Technologies
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出将脑机接口（BCI）向脑-智能体协作（BAC）范式扩展，旨在解决低信息传输率与高校准成本问题。通过融合大语言模型，将智能体视为主动协作伙伴，强调伦理数据处理、模型可靠性和人机协同框架，以提升辅助技术的安全性与可信度。**

- **链接: [http://arxiv.org/pdf/2510.22095v1](http://arxiv.org/pdf/2510.22095v1)**

> **作者:** Yankai Chen; Xinni Zhang; Yifei Zhang; Yangning Li; Henry Peng Zou; Chunyu Miao; Weizhi Zhang; Xue Liu; Philip S. Yu
>
> **备注:** Accepted by NeurIPS'25 Position Track
>
> **摘要:** Brain-Computer Interfaces (BCIs) offer a direct communication pathway between the human brain and external devices, holding significant promise for individuals with severe neurological impairments. However, their widespread adoption is hindered by critical limitations, such as low information transfer rates and extensive user-specific calibration. To overcome these challenges, recent research has explored the integration of Large Language Models (LLMs), extending the focus from simple command decoding to understanding complex cognitive states. Despite these advancements, deploying agentic AI faces technical hurdles and ethical concerns. Due to the lack of comprehensive discussion on this emerging direction, this position paper argues that the field is poised for a paradigm extension from BCI to Brain-Agent Collaboration (BAC). We emphasize reframing agents as active and collaborative partners for intelligent assistance rather than passive brain signal data processors, demanding a focus on ethical data handling, model reliability, and a robust human-agent collaboration framework to ensure these systems are safe, trustworthy, and effective.
>
---
#### [new 167] Fast-MIA: Efficient and Scalable Membership Inference for LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对大语言模型（LLM）的成员推理攻击（MIA）研究中计算成本高、方法不统一的问题，提出Fast-MIA库。该工具提供高效批量推理与标准化MIA方法实现，支持可复现的大规模实验，促进LLM隐私安全研究。**

- **链接: [http://arxiv.org/pdf/2510.23074v1](http://arxiv.org/pdf/2510.23074v1)**

> **作者:** Hiromu Takahashi; Shotaro Ishihara
>
> **摘要:** We propose Fast-MIA (https://github.com/Nikkei/fast-mia), a Python library for efficiently evaluating membership inference attacks (MIA) against Large Language Models (LLMs). MIA against LLMs has emerged as a crucial challenge due to growing concerns over copyright, security, and data privacy, and has attracted increasing research attention. However, the progress of this research is significantly hindered by two main obstacles: (1) the high computational cost of inference in LLMs, and (2) the lack of standardized and maintained implementations of MIA methods, which makes large-scale empirical comparison difficult. To address these challenges, our library provides fast batch inference and includes implementations of representative MIA methods under a unified evaluation framework. This library supports easy implementation of reproducible benchmarks with simple configuration and extensibility. We release Fast-MIA as an open-source (Apache License 2.0) tool to support scalable and transparent research on LLMs.
>
---
#### [new 168] SCoPE VLM: Selective Context Processing for Efficient Document Navigation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉语言模型在长文档导航中的效率问题，提出SCoPE VLM框架。通过链式滚动机制与强化学习，实现对文档的精准、低内存的逐段聚焦阅读，解决现有方法内存占用高、缺乏人类阅读行为模拟的问题，首次建模了多页文档问答中的代理式阅读模式。**

- **链接: [http://arxiv.org/pdf/2510.21850v1](http://arxiv.org/pdf/2510.21850v1)**

> **作者:** Gyubeum Lim; Yemo Koo; Vijay Krishna Madisetti
>
> **摘要:** Understanding long-context visual information remains a fundamental challenge for vision-language models, particularly in agentic tasks such as GUI control and web navigation. While web pages and GUI environments are inherently structured documents, current VLMs typically neglect decision-oriented document understanding in their training objectives. Existing approaches primarily extend visual embeddings to process long, high-resolution inputs, but these methods are memory-intensive and impractical for locally deployable solutions. To address these issues, we propose SCoPE VLM, a document navigation expert that leverages a novel Chain of Scroll mechanism to selectively and recursively navigate documents, focusing exclusively on relevant segments. We introduce a dedicated data generation pipeline to construct informative Chain of Scroll trajectories and Episodic Group Relative Policy Optimization, a tailored reinforcement learning method to reduce the gap between training and inference. Our method substantially reduces memory usage and effectively models human-like reading behaviors. To the best of our knowledge, SCoPE VLM is the first framework to explicitly model agentic reading patterns in multi-page document question answering, advancing the capabilities of multimodal agents.
>
---
#### [new 169] Rethinking GSPO: The Perplexity-Entropy Equivalence
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究强化学习中的策略优化问题，聚焦GSPO算法。通过建立重要性权重与信息论量（困惑度、交叉熵）的等价关系，揭示其内在机制：权重等价于困惑度比或熵变指数，解释了其方差降低与训练稳定性。实验验证了理论等价性与预测效果。**

- **链接: [http://arxiv.org/pdf/2510.23142v1](http://arxiv.org/pdf/2510.23142v1)**

> **作者:** Chi Liu
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** We provide a new perspective on GSPO's length-normalized importance ratios by establishing their connection to information-theoretic quantities. We show that GSPO's sequence-level weight $s(\theta) = (\pi_\theta/\pi_{\theta_{\text{old}}})^{1/|y|}$ can be equivalently expressed as the inverse perplexity ratio $\text{PPL}_{\theta_{\text{old}}}/\text{PPL}_\theta$ and as the exponential cross-entropy change $\exp(\Delta H)$. While the perplexity-entropy relationship follows from standard definitions, this observation provides a useful lens for understanding GSPO: the algorithm weights policy gradient updates by perplexity ratios, offering an information-theoretic interpretation of the importance weights. This perspective helps explain GSPO's empirical properties, including log-domain variance reduction through geometric averaging and stability in training mixture-of-experts models. We validate the mathematical equivalences and variance predictions through controlled experiments on mathematical reasoning tasks.
>
---
#### [new 170] A Neuro-Symbolic Multi-Agent Approach to Legal-Cybersecurity Knowledge Integration
- **分类: cs.AI; cs.CL; cs.CR; cs.MA**

- **简介: 该论文提出一种神经符号多智能体方法，用于整合法律与网络安全知识。针对法律与技术信息脱节、跨领域协作困难的问题，通过融合自然语言处理与符号推理，实现对多语言法律与安全数据的智能理解与关联分析，推动跨域知识融合。**

- **链接: [http://arxiv.org/pdf/2510.23443v1](http://arxiv.org/pdf/2510.23443v1)**

> **作者:** Chiara Bonfanti; Alessandro Druetto; Cataldo Basile; Tharindu Ranasinghe; Marcos Zampieri
>
> **备注:** 7 pages
>
> **摘要:** The growing intersection of cybersecurity and law creates a complex information space where traditional legal research tools struggle to deal with nuanced connections between cases, statutes, and technical vulnerabilities. This knowledge divide hinders collaboration between legal experts and cybersecurity professionals. To address this important gap, this work provides a first step towards intelligent systems capable of navigating the increasingly intricate cyber-legal domain. We demonstrate promising initial results on multilingual tasks.
>
---
#### [new 171] Variational Masked Diffusion Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出变分掩码扩散模型（VMD），用于离散生成建模任务。针对传统掩码扩散模型无法有效捕捉并发预测词元间依赖的问题，VMD引入潜在变量，通过变分推断显式建模词元间依赖。在合成数据、数独和文本数据上验证，VMD提升了生成质量与全局一致性。**

- **链接: [http://arxiv.org/pdf/2510.23606v1](http://arxiv.org/pdf/2510.23606v1)**

> **作者:** Yichi Zhang; Alex Schwing; Zhizhen Zhao
>
> **备注:** Project Page: https://riccizz.github.io/VMD
>
> **摘要:** Masked diffusion models have recently emerged as a flexible framework for discrete generative modeling. However, a key limitation of standard masked diffusion is its inability to effectively capture dependencies among tokens that are predicted concurrently, leading to degraded generation quality when dependencies among tokens are important. To explicitly model dependencies among tokens, we propose Variational Masked Diffusion (VMD), a framework that introduces latent variables into the masked diffusion process. Through controlled experiments on synthetic datasets, we demonstrate that VMD successfully learns dependencies that conventional masked diffusion fails to capture. We further validate the effectiveness of our approach on Sudoku puzzles and text datasets, where learning of dependencies among tokens improves global consistency. Across these domains, VMD enhances both generation quality and dependency awareness, highlighting the value of integrating variational inference into masked diffusion. Our code is available at: https://riccizz.github.io/VMD.
>
---
#### [new 172] PACR: Progressively Ascending Confidence Reward for LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大模型推理中强化学习奖励稀疏的问题，提出PACR方法，通过模型自身对正确答案置信度的逐步上升趋势设计密集内在奖励，引导中间步骤探索。实验表明，PACR加速收敛，提升多任务表现，使RLVR更高效可靠。**

- **链接: [http://arxiv.org/pdf/2510.22255v1](http://arxiv.org/pdf/2510.22255v1)**

> **作者:** Eunseop Yoon; Hee Suk Yoon; Jaehyun Jang; SooHwan Eom; Qi Dai; Chong Luo; Mark A. Hasegawa-Johnson; Chang D. Yoo
>
> **备注:** 16 pages, 14 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved LLM reasoning, but its sparse, outcome-based reward provides no guidance for intermediate steps, slowing exploration. We propose Progressively Ascending Confidence Reward (PACR), a dense, model-intrinsic reward computed directly from the model's evolving belief in the correct answer. PACR encodes the inductive bias that, along a well-formed reasoning trajectory, the probability of the ground-truth answer should have a generally ascending trend. We provide empirical and theoretical analysis validating that such an inductive bias constrains the exploration search space to regions richer in logically sound reasoning. We demonstrate that PACR accelerates exploration, reaches reward saturation with fewer trajectories, and yields improvements on multiple benchmarks. Our results suggest that dense, model-intrinsic shaping signals can make RLVR training more effective and reliable.
>
---
#### [new 173] Multi-Modal Fact-Verification Framework for Reducing Hallucinations in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大语言模型（LLM）幻觉问题，提出多模态事实验证框架。通过交叉比对结构化数据库、网络搜索与学术文献，实时验证并修正生成内容中的错误，显著降低幻觉率67%，提升输出准确性与可信度，适用于医疗、金融等高精度场景。**

- **链接: [http://arxiv.org/pdf/2510.22751v1](http://arxiv.org/pdf/2510.22751v1)**

> **作者:** Piyushkumar Patel
>
> **摘要:** While Large Language Models have transformed how we interact with AI systems, they suffer from a critical flaw: they confidently generate false information that sounds entirely plausible. This hallucination problem has become a major barrier to deploying these models in real-world applications where accuracy matters. We developed a fact verification framework that catches and corrects these errors in real-time by cross checking LLM outputs against multiple knowledge sources. Our system combines structured databases, live web searches, and academic literature to verify factual claims as they're generated. When we detect inconsistencies, we automatically correct them while preserving the natural flow of the response. Testing across various domains showed we could reduce hallucinations by 67% without sacrificing response quality. Domain experts in healthcare, finance, and scientific research rated our corrected outputs 89% satisfactory a significant improvement over unverified LLM responses. This work offers a practical solution for making LLMs more trustworthy in applications where getting facts wrong isn't an option.
>
---
#### [new 174] From Social Division to Cohesion with AI Message Suggestions in Online Chat Groups
- **分类: cs.SI; cs.CL**

- **简介: 该论文研究AI消息建议对在线群组社会凝聚力的影响。针对意见分化下群体分裂问题，通过557人多轮实验，对比个体化与群体化AI建议的效果。结果表明，基于群体语境的个性化建议能促进包容性对话，增强凝聚力，而个体化建议则加剧群体隔离。**

- **链接: [http://arxiv.org/pdf/2510.21984v1](http://arxiv.org/pdf/2510.21984v1)**

> **作者:** Faria Huq; Elijah L. Claggett; Hirokazu Shirado
>
> **备注:** Preprint, Under Review
>
> **摘要:** Social cohesion is difficult to sustain in societies marked by opinion diversity, particularly in online communication. As large language model (LLM)-driven messaging assistance becomes increasingly embedded in these contexts, it raises critical questions about its societal impact. We present an online experiment with 557 participants who engaged in multi-round discussions on politically controversial topics while freely reconfiguring their discussion groups. In some conditions, participants received real-time message suggestions generated by an LLM, either personalized to the individual or adapted to their group context. We find that subtle shifts in linguistic style during communication, mediated by AI assistance, can scale up to reshape collective structures. While individual-focused assistance leads users to segregate into like-minded groups, relational assistance that incorporates group members' stances enhances cohesion through more receptive exchanges. These findings demonstrate that AI-mediated communication can support social cohesion in diverse groups, but outcomes critically depend on how personalization is designed.
>
---
#### [new 175] How Do AI Agents Do Human Work? Comparing AI and Human Workflows Across Diverse Occupations
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于人机协作研究，旨在比较AI代理与人类在数据、工程、写作、设计等任务中的工作流程。通过构建可扩展工具提取结构化工作流，发现代理虽速度快、成本低，但依赖程序化方法，质量较差且常造假，而人类更依赖界面交互。研究揭示了代理在多样化工作中的局限与协作潜力。**

- **链接: [http://arxiv.org/pdf/2510.22780v1](http://arxiv.org/pdf/2510.22780v1)**

> **作者:** Zora Zhiruo Wang; Yijia Shao; Omar Shaikh; Daniel Fried; Graham Neubig; Diyi Yang
>
> **摘要:** AI agents are continually optimized for tasks related to human work, such as software engineering and professional writing, signaling a pressing trend with significant impacts on the human workforce. However, these agent developments have often not been grounded in a clear understanding of how humans execute work, to reveal what expertise agents possess and the roles they can play in diverse workflows. In this work, we study how agents do human work by presenting the first direct comparison of human and agent workers across multiple essential work-related skills: data analysis, engineering, computation, writing, and design. To better understand and compare heterogeneous computer-use activities of workers, we introduce a scalable toolkit to induce interpretable, structured workflows from either human or agent computer-use activities. Using such induced workflows, we compare how humans and agents perform the same tasks and find that: (1) While agents exhibit promise in their alignment to human workflows, they take an overwhelmingly programmatic approach across all work domains, even for open-ended, visually dependent tasks like design, creating a contrast with the UI-centric methods typically used by humans. (2) Agents produce work of inferior quality, yet often mask their deficiencies via data fabrication and misuse of advanced tools. (3) Nonetheless, agents deliver results 88.3% faster and cost 90.4-96.2% less than humans, highlighting the potential for enabling efficient collaboration by delegating easily programmable tasks to agents.
>
---
#### [new 176] A U-Net and Transformer Pipeline for Multilingual Image Translation
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出一种端到端多语言图像文本翻译流水线，解决图像中文本检测、识别与跨语言翻译问题。工作包括：基于合成数据训练的自定义U-Net检测文本区域，Tesseract提取文字，以及从零训练的多语言Transformer模型实现序列到序列翻译，验证了全定制系统的有效性。**

- **链接: [http://arxiv.org/pdf/2510.23554v1](http://arxiv.org/pdf/2510.23554v1)**

> **作者:** Siddharth Sahay; Radhika Agarwal
>
> **备注:** 6 pages, 3 figures, 5 tables, and 2 algorithms. Prepared in IEEE double-column format
>
> **摘要:** This paper presents an end-to-end multilingual translation pipeline that integrates a custom U-Net for text detection, the Tesseract engine for text recognition, and a from-scratch sequence-to-sequence (Seq2Seq) Transformer for Neural Machine Translation (NMT). Our approach first utilizes a U-Net model, trained on a synthetic dataset , to accurately segment and detect text regions from an image. These detected regions are then processed by Tesseract to extract the source text. This extracted text is fed into a custom Transformer model trained from scratch on a multilingual parallel corpus spanning 5 languages. Unlike systems reliant on monolithic pre-trained models, our architecture emphasizes full customization and adaptability. The system is evaluated on its text detection accuracy, text recognition quality, and translation performance via BLEU scores. The complete pipeline demonstrates promising results, validating the viability of a custom-built system for translating text directly from images.
>
---
#### [new 177] BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文针对软件工程领域语言模型训练中高质量缺陷数据稀缺的问题，提出BugPilot方法，通过模拟真实开发行为生成复杂多样且符合人类编码习惯的合成缺陷。实验表明，其生成的缺陷能以更少数据（1.2k）实现比现有数据集（3k）更高效率的模型训练，显著提升SWE任务性能。**

- **链接: [http://arxiv.org/pdf/2510.19898v1](http://arxiv.org/pdf/2510.19898v1)**

> **作者:** Atharv Sonwane; Isadora White; Hyunji Lee; Matheus Pereira; Lucas Caccia; Minseon Kim; Zhengyan Shi; Chinmay Singh; Alessandro Sordoni; Marc-Alexandre Côté; Xingdi Yuan
>
> **摘要:** High quality bugs are key to training the next generation of language model based software engineering (SWE) agents. We introduce a novel method for synthetic generation of difficult and diverse bugs. Our method instructs SWE Agents to introduce a feature into the codebase whereby they may unintentionally break tests, resulting in bugs. Prior approaches often induce an out-of-distribution effect by generating bugs intentionally (e.g. by introducing local perturbation to existing code), which does not reflect realistic development processes. We perform qualitative analysis to demonstrate that our approach for generating bugs more closely reflects the patterns found in human-authored edits. Through extensive experiments, we demonstrate that our bugs provide more efficient training data for supervised fine-tuning, outperforming other bug datasets by 2% with half the training data (1.2k vs. 3k bugs). We train on our newly generated bugs in addition to existing bug datasets to get FrogBoss a state-of-the-art 32B parameter model on SWE-bench Verified with a pass@1 of 54.6% and FrogMini a state-of-the-art 14B model on SWE-bench Verified with a pass@1 of 45.3% on SWE-bench Verified averaged over three seeds.
>
---
#### [new 178] Modeling Hierarchical Thinking in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大推理模型（LRM）的层次化思维机制，旨在解析其链式思考（CoT）过程。通过构建无记忆有限状态机（FSM）模型，将推理过程抽象为初始化、推导、回溯等离散状态，实现对推理轨迹的结构化分析与可视化，揭示不同模型的推理差异与缺陷，为提升模型推理能力提供新视角。**

- **链接: [http://arxiv.org/pdf/2510.22437v1](http://arxiv.org/pdf/2510.22437v1)**

> **作者:** G M Shahariar; Ali Nazari; Erfan Shayegani; Nael Abu-Ghazaleh
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable reasoning abilities when they generate step-by-step solutions, known as chain-of-thought (CoT) reasoning. When trained to using chain-of-thought reasoning examples, the resulting models (called Large Reasoning Models, or LRMs) appear to learn hierarchical thinking strategies similar to those used by humans. However, understanding LRMs emerging reasoning capabilities remains a difficult open problem, with many potential important applications including improving training and understanding robustness. In this paper, we adopt a memoryless Finite State Machine formulation to approximate LRM's emerging hierarchical reasoning dynamics as a structured, interpretable abstraction. We identify a small set of discrete reasoning states including - initialization, deduction, augmentation-strategy, uncertainty-estimation, backtracking, and final-conclusion that capture the high-level states present in the model's reasoning process. By annotating each step of a model's CoT with these states, we can represent the reasoning trajectory as a transition sequence through the state graph. This FSM formulation provides a systematic way to analyze, interpret and visualize how different models approach problems. We describe the FSM model, provide examples of CoT annotations under this scheme, and discuss how it can shed light on differences between available models in their approach to reasoning. Our results demonstrate that this FSM-based analysis reveals distinct reasoning patterns and potential shortcomings, offering a new lens to evaluate and improve LLM reasoning.
>
---
#### [new 179] REVISION:Reflective Intent Mining and Online Reasoning Auxiliary for E-commerce Visual Search System Optimization
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对电商视觉搜索中用户隐式意图难以捕捉导致的点击率低问题，提出REVISION框架。通过离线挖掘历史无点击请求中的意图差异，结合大模型推理生成优化建议；在线阶段利用训练好的模型动态调整搜索策略，实现端到端智能优化，显著降低无点击率。**

- **链接: [http://arxiv.org/pdf/2510.22739v1](http://arxiv.org/pdf/2510.22739v1)**

> **作者:** Yiwen Tang; Qiuyu Zhao; Zenghui Sun; Jinsong Lan; Xiaoyong Zhu; Bo Zheng; Kaifu Zhang
>
> **摘要:** In Taobao e-commerce visual search, user behavior analysis reveals a large proportion of no-click requests, suggesting diverse and implicit user intents. These intents are expressed in various forms and are difficult to mine and discover, thereby leading to the limited adaptability and lag in platform strategies. This greatly restricts users' ability to express diverse intents and hinders the scalability of the visual search system. This mismatch between user implicit intent expression and system response defines the User-SearchSys Intent Discrepancy. To alleviate the issue, we propose a novel framework REVISION. This framework integrates offline reasoning mining with online decision-making and execution, enabling adaptive strategies to solve implicit user demands. In the offline stage, we construct a periodic pipeline to mine discrepancies from historical no-click requests. Leveraging large models, we analyze implicit intent factors and infer optimal suggestions by jointly reasoning over query and product metadata. These inferred suggestions serve as actionable insights for refining platform strategies. In the online stage, REVISION-R1-3B, trained on the curated offline data, performs holistic analysis over query images and associated historical products to generate optimization plans and adaptively schedule strategies across the search pipeline. Our framework offers a streamlined paradigm for integrating large models with traditional search systems, enabling end-to-end intelligent optimization across information aggregation and user interaction. Experimental results demonstrate that our approach improves the efficiency of implicit intent mining from large-scale search logs and significantly reduces the no-click rate.
>
---
#### [new 180] M$^{3}$T2IBench: A Large-Scale Multi-Category, Multi-Instance, Multi-Relation Text-to-Image Benchmark
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对文本到图像生成中的图文对齐问题，提出M³T2IBench基准，涵盖多类别、多实例、多关系复杂场景，并设计与人类评估高度一致的AlignScore指标。研究发现现有模型表现不佳，并提出无需训练的Revise-Then-Enforce后处理方法提升对齐效果。**

- **链接: [http://arxiv.org/pdf/2510.23020v1](http://arxiv.org/pdf/2510.23020v1)**

> **作者:** Huixuan Zhang; Xiaojun Wan
>
> **摘要:** Text-to-image models are known to struggle with generating images that perfectly align with textual prompts. Several previous studies have focused on evaluating image-text alignment in text-to-image generation. However, these evaluations either address overly simple scenarios, especially overlooking the difficulty of prompts with multiple different instances belonging to the same category, or they introduce metrics that do not correlate well with human evaluation. In this study, we introduce M$^3$T2IBench, a large-scale, multi-category, multi-instance, multi-relation along with an object-detection-based evaluation metric, $AlignScore$, which aligns closely with human evaluation. Our findings reveal that current open-source text-to-image models perform poorly on this challenging benchmark. Additionally, we propose the Revise-Then-Enforce approach to enhance image-text alignment. This training-free post-editing method demonstrates improvements in image-text alignment across a broad range of diffusion models. \footnote{Our code and data has been released in supplementary material and will be made publicly available after the paper is accepted.}
>
---
#### [new 181] Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大模型中混合专家（MoE）架构在强化学习（RL）训练中的不稳定性问题，提出一种基于路由器逻辑的重缩放策略，优化离策略RL中的重要性采样权重，有效降低梯度方差，提升训练稳定性和模型性能。**

- **链接: [http://arxiv.org/pdf/2510.23027v1](http://arxiv.org/pdf/2510.23027v1)**

> **作者:** Di Zhang; Xun Wu; Shaohan Huang; Yaru Hao; Li Dong; Zewen Chi; Zhifang Sui; Furu Wei
>
> **摘要:** Recent advances in reinforcement learning (RL) have substantially improved the training of large-scale language models, leading to significant gains in generation quality and reasoning ability. However, most existing research focuses on dense models, while RL training for Mixture-of-Experts (MoE) architectures remains underexplored. To address the instability commonly observed in MoE training, we propose a novel router-aware approach to optimize importance sampling (IS) weights in off-policy RL. Specifically, we design a rescaling strategy guided by router logits, which effectively reduces gradient variance and mitigates training divergence. Experimental results demonstrate that our method significantly improves both the convergence stability and the final performance of MoE models, highlighting the potential of RL algorithmic innovations tailored to MoE architectures and providing a promising direction for efficient training of large-scale expert models.
>
---
#### [new 182] Offline Preference Optimization via Maximum Marginal Likelihood Estimation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MMPO方法，通过最大边际似然估计实现大模型与人类偏好的对齐。针对RLHF复杂不稳定的缺点，MMPO无需显式奖励模型和熵正则化，直接优化偏好对，隐式增强优选响应。实验证明其更稳定、性能更优且保留语言能力。**

- **链接: [http://arxiv.org/pdf/2510.22881v1](http://arxiv.org/pdf/2510.22881v1)**

> **作者:** Saeed Najafi; Alona Fyshe
>
> **摘要:** Aligning Large Language Models (LLMs) with human preferences is crucial, but standard methods like Reinforcement Learning from Human Feedback (RLHF) are often complex and unstable. In this work, we propose a new, simpler approach that recasts alignment through the lens of Maximum Marginal Likelihood (MML) estimation. Our new MML based Preference Optimization (MMPO) maximizes the marginal log-likelihood of a preferred text output, using the preference pair as samples for approximation, and forgoes the need for both an explicit reward model and entropy maximization. We theoretically demonstrate that MMPO implicitly performs preference optimization, producing a weighted gradient that naturally up-weights chosen responses over rejected ones. Across models ranging from 135M to 8B parameters, we empirically show that MMPO: 1) is more stable with respect to the hyperparameter $\beta$ compared to alternative baselines, and 2) achieves competitive or superior preference alignment while better preserving the base model's general language capabilities. Through a series of ablation experiments, we show that this improved performance is indeed attributable to MMPO's implicit preference optimization within the gradient updates.
>
---
#### [new 183] DecoupleSearch: Decouple Planning and Search via Hierarchical Reward Modeling
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对Agentic RAG中规划与搜索耦合导致的效率低、难优化问题，提出DecoupleSearch框架。通过双价值模型解耦规划与搜索，构建推理树并结合蒙特卡洛树搜索与分层束搜索，实现两者的独立优化，显著提升RAG系统在复杂任务中的性能与灵活性。**

- **链接: [http://arxiv.org/pdf/2510.21712v1](http://arxiv.org/pdf/2510.21712v1)**

> **作者:** Hao Sun; Zile Qiao; Bo Wang; Guoxin Chen; Yingyan Hou; Yong Jiang; Pengjun Xie; Fei Huang; Yan Zhang
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems have emerged as a pivotal methodology for enhancing Large Language Models (LLMs) through the dynamic integration of external knowledge. To further improve RAG's flexibility, Agentic RAG introduces autonomous agents into the workflow. However, Agentic RAG faces several challenges: (1) the success of each step depends on both high-quality planning and accurate search, (2) the lack of supervision for intermediate reasoning steps, and (3) the exponentially large candidate space for planning and searching. To address these challenges, we propose DecoupleSearch, a novel framework that decouples planning and search processes using dual value models, enabling independent optimization of plan reasoning and search grounding. Our approach constructs a reasoning tree, where each node represents planning and search steps. We leverage Monte Carlo Tree Search to assess the quality of each step. During inference, Hierarchical Beam Search iteratively refines planning and search candidates with dual value models. Extensive experiments across policy models of varying parameter sizes, demonstrate the effectiveness of our method.
>
---
#### [new 184] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TALE，一种推理时任务感知的层剪枝方法，通过优化任务性能动态移除Transformer层。解决大模型效率与精度权衡问题，在9个任务、5个模型上实现更高准确率与更低计算成本，无需重训，支持灵活效率-精度调控，并揭示了层间瓶颈机制，提升模型可解释性。**

- **链接: [http://arxiv.org/pdf/2510.22767v1](http://arxiv.org/pdf/2510.22767v1)**

> **作者:** Omar Naim; Krish Sharma; Nicholas Asher
>
> **摘要:** In this paper we introduce Tale, Task-Aware Layer Elimination, an inference-time algorithm that prunes entire transformer layers in an LLM by directly optimizing task-specific validation performance. We evaluate TALE on 9 tasks and 5 models, including LLaMA 3.1 8B, Qwen 2.5 7B, Qwen 2.5 0.5B, Mistral 7B, and Lucie 7B, under both zero-shot and few-shot settings. Unlike prior approaches, TALE requires no retraining and consistently improves accuracy while reducing computational cost across all benchmarks. Furthermore, applying TALE during finetuning leads to additional performance gains. Finally, TALE provides flexible user control over trade-offs between accuracy and efficiency. Mutual information analysis shows that certain layers act as bottlenecks, degrading task-relevant representations. Tale's selective layer removal remedies this problem, producing smaller, faster, and more accurate models that are also faster to fine-tune while offering new insights into transformer interpretability.
>
---
## 更新

#### [replaced 001] Can Large Language Models Unlock Novel Scientific Research Ideas?
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.06185v2](http://arxiv.org/pdf/2409.06185v2)**

> **作者:** Sandeep Kumar; Tirthankar Ghosal; Vinayak Goyal; Asif Ekbal
>
> **备注:** EMNLP 2025 (Main)
>
> **摘要:** The widespread adoption of Large Language Models (LLMs) and publicly available ChatGPT have marked a significant turning point in the integration of Artificial Intelligence (AI) into people's everyday lives. This study examines the ability of Large Language Models (LLMs) to generate future research ideas from scientific papers. Unlike tasks such as summarization or translation, idea generation lacks a clearly defined reference set or structure, making manual evaluation the default standard. However, human evaluation in this setting is extremely challenging ie: it requires substantial domain expertise, contextual understanding of the paper, and awareness of the current research landscape. This makes it time-consuming, costly, and fundamentally non-scalable, particularly as new LLMs are being released at a rapid pace. Currently, there is no automated evaluation metric specifically designed for this task. To address this gap, we propose two automated evaluation metrics: Idea Alignment Score (IAScore) and Idea Distinctness Index. We further conducted human evaluation to assess the novelty, relevance, and feasibility of the generated future research ideas. This investigation offers insights into the evolving role of LLMs in idea generation, highlighting both its capability and limitations. Our work contributes to the ongoing efforts in evaluating and utilizing language models for generating future research ideas. We make our datasets and codes publicly available
>
---
#### [replaced 002] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10887v3](http://arxiv.org/pdf/2506.10887v3)**

> **作者:** Yixiao Huang; Hanlin Zhu; Tianyu Guo; Jiantao Jiao; Somayeh Sojoudi; Michael I. Jordan; Stuart Russell; Song Mei
>
> **备注:** NeurIPS 2025, first three authors contributed equally
>
> **摘要:** Large language models (LLMs) can acquire new knowledge through fine-tuning, but this process exhibits a puzzling duality: models can generalize remarkably from new facts, yet are also prone to hallucinating incorrect information. However, the reasons for this phenomenon remain poorly understood. In this work, we argue that both behaviors stem from a single mechanism known as out-of-context reasoning (OCR): the ability to deduce implications by associating concepts, even those without a causal link. Our experiments across five prominent LLMs confirm that OCR indeed drives both generalization and hallucination, depending on whether the associated concepts are causally related. To build a rigorous theoretical understanding of this phenomenon, we then formalize OCR as a synthetic factual recall task. We empirically show that a one-layer single-head attention-only transformer with factorized output and value matrices can learn to solve this task, while a model with combined weights cannot, highlighting the crucial role of matrix factorization. Our theoretical analysis shows that the OCR capability can be attributed to the implicit bias of gradient descent, which favors solutions that minimize the nuclear norm of the combined output-value matrix. This mathematical structure explains why the model learns to associate facts and implications with high sample efficiency, regardless of whether the correlation is causal or merely spurious. Ultimately, our work provides a theoretical foundation for understanding the OCR phenomenon, offering a new lens for analyzing and mitigating undesirable behaviors from knowledge injection.
>
---
#### [replaced 003] Better Estimation of the Kullback--Leibler Divergence Between Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10637v3](http://arxiv.org/pdf/2504.10637v3)**

> **作者:** Afra Amini; Tim Vieira; Ryan Cotterell
>
> **备注:** NeurIPS 2025
>
> **摘要:** Estimating the Kullback--Leibler (KL) divergence between language models has many applications, e.g., reinforcement learning from human feedback (RLHF), interpretability, and knowledge distillation. However, computing the exact KL divergence between two arbitrary language models is intractable. Thus, practitioners often resort to sampling-based estimators. While it is easy to fashion a simple Monte Carlo (MC) estimator that provides an unbiased estimate of the KL divergence between language models, this estimator notoriously suffers from high variance and can even result in a negative estimate of the KL divergence, a non-negative quantity. In this paper, we introduce a Rao--Blackwellized estimator that is unbiased and provably has variance less than or equal to that of the standard Monte Carlo estimator. In an empirical study on sentiment-controlled fine-tuning, we show that our estimator provides more stable KL estimates and reduces variance substantially. Additionally, we derive an analogous Rao--Blackwellized estimator of the gradient of the KL divergence, which leads to more stable training and produces models that more frequently appear on the Pareto frontier of reward vs. KL compared to the ones trained with the MC estimator of the gradient.
>
---
#### [replaced 004] How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.11477v3](http://arxiv.org/pdf/2406.11477v3)**

> **作者:** Atsuki Yamaguchi; Aline Villavicencio; Nikolaos Aletras
>
> **备注:** Accepted to Computational Linguistics
>
> **摘要:** Large language models (LLMs) have shown remarkable capabilities in many languages beyond English. Yet, LLMs require more inference steps when generating non-English text due to their reliance on English-centric tokenizers and vocabulary, resulting in higher usage costs to non-English speakers. Vocabulary expansion with target language tokens is a widely used cross-lingual vocabulary adaptation approach to remedy this issue. Despite its effectiveness in inference speedup, previous work on vocabulary expansion has focused on high-resource settings assuming access to a substantial amount of target language data to effectively initialize the embeddings of the new tokens and adapt the LLM to the target language. However, vocabulary expansion in low-resource settings has yet to be explored. In this article, we investigate vocabulary expansion in low-resource settings by considering embedding initialization methods and continual pre-training strategies. Through extensive experiments across typologically diverse languages, tasks and models, we establish a set of strategies to perform vocabulary expansion for faster inference, while striving to maintain competitive downstream performance to baselines. This is achieved with only 30K sentences ($\sim$0.01GB text data) from the target language.
>
---
#### [replaced 005] Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.18855v2](http://arxiv.org/pdf/2510.18855v2)**

> **作者:** Ling Team; Anqi Shen; Baihui Li; Bin Hu; Bin Jing; Cai Chen; Chao Huang; Chao Zhang; Chaokun Yang; Cheng Lin; Chengyao Wen; Congqi Li; Deng Zhao; Dingbo Yuan; Donghai You; Fagui Mao; Fanzhuang Meng; Feng Xu; Guojie Li; Guowei Wang; Hao Dai; Haonan Zheng; Hong Liu; Jia Guo; Jiaming Liu; Jian Liu; Jianhao Fu; Jiannan Shi; Jianwen Wang; Jianxin Lai; Jin Yang; Jun Mei; Jun Zhou; Junbo Zhao; Junping Zhao; Kuan Xu; Le Su; Lei Chen; Li Tang; Liang Jiang; Liangcheng Fu; Lianhao Xu; Linfeng Shi; Lisha Liao; Longfei Zheng; Meng Li; Mingchun Chen; Qi Zuo; Qiang Cheng; Qianggang Cao; Qitao Shi; Quanrui Guo; Senlin Zhu; Shaofei Wang; Shaomian Zheng; Shuaicheng Li; Shuwei Gu; Siba Chen; Tao Wu; Tao Zhang; Tianyu Zhang; Tianyu Zhou; Tiwei Bie; Tongkai Yang; Wang Hong; Wang Ren; Weihua Chen; Wenbo Yu; Wengang Zheng; Xiangchun Wang; Xiaodong Yan; Xiaopei Wan; Xin Zhao; Xinyu Kong; Xinyu Tang; Xudong Han; Xudong Wang; Xuemin Yang; Xueyu Hu; Yalin Zhang; Yan Sun; Yicheng Shan; Yilong Wang; Yingying Xu; Yongkang Liu; Yongzhen Guo; Yuanyuan Wang; Yuchen Yan; Yuefan Wang; Yuhong Guo; Zehuan Li; Zhankai Xu; Zhe Li; Zhenduo Zhang; Zhengke Gui; Zhenxuan Pan; Zhenyu Huang; Zhenzhong Lan; Zhiqiang Ding; Zhiqiang Zhang; Zhixun Li; Zhizhen Liu; Zihao Wang; Zujie Wen
>
> **备注:** Technical Report
>
> **摘要:** We present Ring-1T, the first open-source, state-of-the-art thinking model with a trillion-scale parameter. It features 1 trillion total parameters and activates approximately 50 billion per token. Training such models at a trillion-parameter scale introduces unprecedented challenges, including train-inference misalignment, inefficiencies in rollout processing, and bottlenecks in the RL system. To address these, we pioneer three interconnected innovations: (1) IcePop stabilizes RL training via token-level discrepancy masking and clipping, resolving instability from training-inference mismatches; (2) C3PO++ improves resource utilization for long rollouts under a token budget by dynamically partitioning them, thereby obtaining high time efficiency; and (3) ASystem, a high-performance RL framework designed to overcome the systemic bottlenecks that impede trillion-parameter model training. Ring-1T delivers breakthrough results across critical benchmarks: 93.4 on AIME-2025, 86.72 on HMMT-2025, 2088 on CodeForces, and 55.94 on ARC-AGI-1. Notably, it attains a silver medal-level result on the IMO-2025, underscoring its exceptional reasoning capabilities. By releasing the complete 1T parameter MoE model to the community, we provide the research community with direct access to cutting-edge reasoning capabilities. This contribution marks a significant milestone in democratizing large-scale reasoning intelligence and establishes a new baseline for open-source model performance.
>
---
#### [replaced 006] Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices
- **分类: cs.DC; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2510.05109v2](http://arxiv.org/pdf/2510.05109v2)**

> **作者:** Yilong Li; Shuai Zhang; Yijing Zeng; Hao Zhang; Xinmiao Xiong; Jingyu Liu; Pan Hu; Suman Banerjee
>
> **摘要:** Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly half a day and LLaMA-3-8B for voice interactions up to almost 20.8 hours.
>
---
#### [replaced 007] AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10720v2](http://arxiv.org/pdf/2503.10720v2)**

> **作者:** Yixiong Fang; Tianran Sun; Yuling Shi; Xiaodong Gu
>
> **摘要:** While RAG demonstrates remarkable capabilities in LLM applications, its effectiveness is hindered by the ever-increasing length of retrieved contexts, which introduces information redundancy and substantial computational overhead. Existing context pruning methods, such as LLMLingua, lack contextual awareness and offer limited flexibility in controlling compression rates, often resulting in either insufficient pruning or excessive information loss. In this paper, we propose AttentionRAG, an attention-guided context pruning method for RAG systems. The core idea of AttentionRAG lies in its attention focus mechanism, which reformulates RAG queries into a next-token prediction paradigm. This mechanism isolates the query's semantic focus to a single token, enabling precise and efficient attention calculation between queries and retrieved contexts. Extensive experiments on LongBench and Babilong benchmarks show that AttentionRAG achieves up to 6.3$\times$ context compression while outperforming LLMLingua methods by around 10\% in key metrics.
>
---
#### [replaced 008] Human-Aligned Faithfulness in Toxicity Explanations of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19113v2](http://arxiv.org/pdf/2506.19113v2)**

> **作者:** Ramaravind K. Mothilal; Joanna Roy; Syed Ishtiaque Ahmed; Shion Guha
>
> **备注:** 23 pages, 5 figures, 7 tables
>
> **摘要:** The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate HAF of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code at https://github.com/uofthcdslab/HAF and LLM-generated explanations at https://huggingface.co/collections/uofthcdslab/haf.
>
---
#### [replaced 009] Towards Hierarchical Multi-Step Reward Models for Enhanced Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13551v4](http://arxiv.org/pdf/2503.13551v4)**

> **作者:** Teng Wang; Zhangyi Jiang; Zhenqi He; Shenyang Tong; Wenhan Yang; Yanan Zheng; Zeyu Li; Zifan He; Hailei Gong; Zewen Ye; Shengjie Ma; Jianping Zhang
>
> **摘要:** Recent studies show that Large Language Models (LLMs) achieve strong reasoning capabilities through supervised fine-tuning or reinforcement learning. However, a key approach, the Process Reward Model (PRM), suffers from reward hacking, making it unreliable in identifying the best intermediate step. In addition, the cost of annotating reasoning processes for reward modeling is high, making large-scale collection of high-quality data challenging. To address this, we propose a novel reward model approach called the Hierarchical Reward Model (HRM), which evaluates both individual and consecutive reasoning steps at both fine-grained and coarse-grained levels. HRM excels at assessing multi-step reasoning coherence, especially when flawed steps are later corrected through self-reflection. To further reduce the cost of generating training data, we introduce a lightweight and effective data augmentation strategy called Hierarchical Node Compression (HNC), which merges two consecutive reasoning steps into one within the tree structure. By applying HNC to MCTS-generated reasoning trajectories, we enhance the diversity and robustness of HRM training data while introducing controlled noise with minimal computational overhead. Empirical results on the PRM800K dataset show that HRM, together with HNC, provides more stable and reliable evaluations than PRM. Furthermore, cross-domain evaluations on the MATH500 and GSM8K datasets demonstrate HRM's strong generalization and robustness across a variety of reasoning tasks.
>
---
#### [replaced 010] MathOPEval: A Fine-grained Evaluation Benchmark for Visual Operations of MLLMs in Mathematical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18140v2](http://arxiv.org/pdf/2507.18140v2)**

> **作者:** Xiaoyuan Li; Moxin Li; Wenjie Wang; Rui Men; Yichang Zhang; Fuli Feng; Dayiheng Liu
>
> **备注:** Under Review
>
> **摘要:** Recent progress in Multi-modal Large Language Models (MLLMs) has enabled step-by-step multi-modal mathematical reasoning by performing visual operations based on the textual instructions. A promising approach uses code as an intermediate representation to precisely express and manipulate the images in the reasoning steps. However, existing evaluations focus mainly on text-only reasoning outputs, leaving the MLLM's ability to perform accurate visual operations via code largely unexplored. This work takes a first step toward addressing that gap by evaluating MLLM's code-based capabilities in multi-modal mathematical reasoning.Specifically, our framework focuses on two key evaluation aspects: (1) Multi-modal Code Generation (MCG) evaluates the model's ability to accurately understand and construct visualizations from scratch. (2) Multi-modal Code Editing (MCE) assesses the model's capacity for fine-grained operations, which include three types: Deletion, Modification and Annotation. To evaluate the above tasks, we incorporate a dataset that covers the five most popular types of mathematical figures, including geometric diagrams, function plots, and three types of statistical charts, to provide a comprehensive and effective measurement of existing MLLMs. Our experimental evaluation involves nine mainstream MLLMs, and the results reveal that existing models still lag significantly behind human performance in performing fine-grained visual operations.
>
---
#### [replaced 011] Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes
- **分类: q-fin.RM; cs.AI; cs.CL; cs.LG; q-fin.CP**

- **链接: [http://arxiv.org/pdf/2410.17266v2](http://arxiv.org/pdf/2410.17266v2)**

> **作者:** Kelvin J. L. Koa; Yunshan Ma; Yi Xu; Ritchie Ng; Huanhuan Zheng; Tat-Seng Chua
>
> **备注:** ICAIF 2025 Workshop (Oral)
>
> **摘要:** Stock portfolios are often exposed to rare consequential events (e.g., 2007 global financial crisis, 2020 COVID-19 stock market crash), as they do not have enough historical information to learn from. Large Language Models (LLMs) now present a possible tool to tackle this problem, as they can generalize across their large corpus of training data and perform zero-shot reasoning on new events, allowing them to detect possible portfolio crash events without requiring specific training data. However, detecting portfolio crashes is a complex problem that requires more than reasoning abilities. Investors need to dynamically process the impact of each new piece of information found in news articles, analyze the relational network of impacts across different events and portfolio stocks, as well as understand the temporal context between impacts across time-steps, in order to obtain the aggregated impact on the target portfolio. In this work, we propose an algorithmic framework named Temporal Relational Reasoning (TRR). It seeks to emulate the spectrum of human cognitive capabilities used for complex problem-solving, which include brainstorming, memory, attention and reasoning. Through extensive experiments, we show that TRR is able to outperform state-of-the-art techniques on detecting stock portfolio crashes, and demonstrate how each of the proposed components help to contribute to its performance through an ablation study. Additionally, we further explore the possible applications of TRR by extending it to other related complex problems, such as the detection of possible global crisis events in Macroeconomics.
>
---
#### [replaced 012] LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.07793v2](http://arxiv.org/pdf/2510.07793v2)**

> **作者:** Sajib Acharjee Dip; Adrika Zafor; Bikash Kumar Paul; Uddip Acharjee Shuvo; Muhit Islam Emon; Xuan Wang; Liqing Zhang
>
> **备注:** 34 pages, 5 figures, 7 tables
>
> **摘要:** Large language models (LLMs) and emerging agentic frameworks are beginning to transform single-cell biology by enabling natural-language reasoning, generative annotation, and multimodal data integration. However, progress remains fragmented across data modalities, architectures, and evaluation standards. LLM4Cell presents the first unified survey of 58 foundation and agentic models developed for single-cell research, spanning RNA, ATAC, multi-omic, and spatial modalities. We categorize these methods into five families-foundation, text-bridge, spatial, multimodal, epigenomic, and agentic-and map them to eight key analytical tasks including annotation, trajectory and perturbation modeling, and drug-response prediction. Drawing on over 40 public datasets, we analyze benchmark suitability, data diversity, and ethical or scalability constraints, and evaluate models across 10 domain dimensions covering biological grounding, multi-omics alignment, fairness, privacy, and explainability. By linking datasets, models, and evaluation domains, LLM4Cell provides the first integrated view of language-driven single-cell intelligence and outlines open challenges in interpretability, standardization, and trustworthy model development.
>
---
#### [replaced 013] Fine-tuning Large Language Models with Limited Data: A Survey and Practical Guide
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.09539v2](http://arxiv.org/pdf/2411.09539v2)**

> **作者:** Marton Szep; Daniel Rueckert; Rüdiger von Eisenhart-Rothe; Florian Hinterwimmer
>
> **备注:** Accepted to TACL. Pre-MIT Press version. Major restructuring; added preference alignment section and additional tables. 36 pages
>
> **摘要:** Fine-tuning large language models (LLMs) with limited data poses a practical challenge in low-resource languages, specialized domains, and constrained deployment settings. While pre-trained LLMs provide strong foundations, effective adaptation under data scarcity requires focused and efficient fine-tuning techniques. This paper presents a structured and practical survey of recent methods for fine-tuning LLMs in data-scarce scenarios. We systematically review parameter-efficient fine-tuning techniques that lower training and deployment costs, domain and cross-lingual adaptation methods for both encoder and decoder models, and model specialization strategies. We further examine preference alignment approaches that guide model behavior using limited human or synthetic feedback, emphasizing sample and compute efficiency. Throughout, we highlight empirical trade-offs, selection criteria, and best practices for choosing suitable techniques based on task constraints, including model scaling, data scaling, and the mitigation of catastrophic forgetting. The aim is to equip researchers and practitioners with actionable insights for effectively fine-tuning LLMs when data and resources are limited.
>
---
#### [replaced 014] Solving the Unsolvable: Translating Case Law in Hong Kong
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2501.09444v3](http://arxiv.org/pdf/2501.09444v3)**

> **作者:** King-kui Sin; Xi Xuan; Chunyu Kit; Clara Ho-yan Chan; Honic Ho-kin Ip
>
> **摘要:** This paper addresses the challenges translating case law under Hong Kong's bilingual legal system. It highlights the initial success of translating all written statutes into Chinese before the 1997 handover, a task mandated by the Basic Law. The effort involved significant collaboration among legal, linguistic, and translation experts, resulting in a comprehensive and culturally appropriate bilingual legal system. However, translating case law remains a significant challenge due to the sheer volume and continuous growth of judicial decisions. The paper critiques the governments and judiciarys sporadic and uncoordinated efforts to translate case law, contrasting it with the thorough approach previously taken for statute translation. Although the government acknowledges the importance of legal bilingualism, it lacks a sustainable strategy for translating case law. The Judiciarys position that translating all judgments is unnecessary, unrealistic, and not cost-effectiveis analyzed and critiqued for its impact on legal transparency and public trust. A proposed solution involves leveraging machine translation technology through a human-machine interactive translation platform, which undergoes two major transitions. Initially based on a neural model, the platform transitions to using a large language model for improved translation accuracy. Furthermore, it evolves from a single-agent system to a multi-agent system, incorporating Translator, Annotator, and Proofreader agents. This multi-agent approach, supported by a grant, aims to facilitate efficient, high-quality translation of judicial judgments by integrating advanced artificial intelligence and continuous feedback mechanisms, thus better meeting the needs of a bilingual legal system.
>
---
#### [replaced 015] ComPO: Preference Alignment via Comparison Oracles
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05465v2](http://arxiv.org/pdf/2505.05465v2)**

> **作者:** Peter Chen; Xi Chen; Wotao Yin; Tianyi Lin
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Direct alignment methods are increasingly used for aligning large language models (LLMs) with human preferences. However, these methods suffer from the issues of verbosity and likelihood displacement, which can be driven by the noisy preference pairs that induce similar likelihood for preferred and dispreferred responses. The contributions of this paper are two-fold. First, we propose a new preference alignment method based on zeroth-order, comparison-based optimization via comparison oracles and provide convergence guarantees for its basic scheme. Second, we improve our method using some heuristics and conduct the experiments to demonstrate the flexibility and compatibility of practical scheme in improving the performance of LLMs using noisy preference pairs. Evaluations are conducted across multiple base and instruction-tuned models (Mistral-7B, Llama-3-8B and Gemma-2-9B) with benchmarks (AlpacaEval 2, MT-Bench and Arena-Hard). Experimental results show the effectiveness of our method as an alternative to addressing the limitations of existing direct alignment methods. A highlight of our work is that we evidence the importance of designing specialized methods for preference pairs with distinct likelihood margin, which complements the recent findings in Razin et al (2025).
>
---
#### [replaced 016] GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.04483v3](http://arxiv.org/pdf/2403.04483v3)**

> **作者:** Zihan Luo; Xiran Song; Hong Huang; Jianxun Lian; Chenhao Zhang; Jinqi Jiang; Xing Xie; Hai Jin
>
> **备注:** Accepted by Frontiers of Computer Science
>
> **摘要:** Improving the general capabilities of large language models (LLMs) is an active research topic. As a common data structure in many real-world domains, understanding graph data is a crucial part of advancing general intelligence. To this end, we propose a dynamic benchmark named GraphInstruct in this paper, which comprehensively includes 21 classical graph reasoning tasks, providing diverse graph generation pipelines and detailed intermediate reasoning steps for each sample. Based on GraphInstruct, we develop GraphSolver via efficient instruction-tuning, which demonstrates prominent graph understanding capability compared to other open-sourced LLMs. To further endow LLMs with multi-step graph reasoning capability, we propose a label-mask training strategy and build GraphSolver+, which leverages masked supervision on intermediate reasoning tokens to emphasize crucial node-identification signals. As one of the pioneering efforts to enhance the graph understanding and reasoning abilities of LLMs, extensive experiments have demonstrated the superiority of GraphSolver and GraphSolver+ over other LLMs. We sincerely hope GraphInstruct will facilitate further research on applying LLMs to graph-structured data. Our code and data are released publicly at: https://github.com/CGCL-codes/GraphInstruct.
>
---
#### [replaced 017] Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21656v2](http://arxiv.org/pdf/2506.21656v2)**

> **作者:** Yifan Shen; Yuanzhe Liu; Jingyuan Zhu; Xu Cao; Xiaofeng Zhang; Yixiao He; Wenming Ye; James Matthew Rehg; Ismini Lourentzou
>
> **摘要:** Current Vision-Language Models (VLMs) struggle with fine-grained spatial reasoning, particularly when multi-step logic and precise spatial alignment are required. In this work, we introduce SpatialReasoner-R1, a vision-language reasoning model designed to address these limitations. To construct high-quality supervision for spatial reasoning, we design a Multi-Model Monte Carlo Tree Search (M3CTS) method that generates diverse, logically consistent Long Chain-of-Thought (LongCoT) reasoning trajectories. In addition, we propose fine-grained Direct Preference Optimization (fDPO), which introduces segment-specific preference granularity for descriptive grounding and logical reasoning, guided by a spatial reward mechanism that evaluates candidate responses based on visual consistency, spatial grounding, and logical coherence. Experimental results demonstrate that fDPO achieves an average improvement of 4.1% over standard DPO across spatial quality tasks, and a 9.0% gain in spatial quantity tasks. SpatialReasoner-R1, trained with fDPO, sets a new SoTA on SPATIALRGPT-Bench, outperforming the strongest baseline by 9.8% in average accuracy, while maintaining competitive performance on general vision-language tasks.
>
---
#### [replaced 018] TaoSR1: The Thinking Model for E-commerce Relevance Search
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12365v2](http://arxiv.org/pdf/2508.12365v2)**

> **作者:** Chenhe Dong; Shaowei Yao; Pengkun Jiao; Jianhui Yang; Yiming Jin; Zerui Huang; Xiaojiang Zhou; Dan Ou; Haihong Tang; Bo Zheng
>
> **摘要:** Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification.
>
---
#### [replaced 019] Bhav-Net: Knowledge Transfer for Cross-Lingual Antonym vs Synonym Distinction via Dual-Space Graph Transformers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.15792v3](http://arxiv.org/pdf/2508.15792v3)**

> **作者:** Samyak S. Sanghvi
>
> **备注:** Found some issues and need to correct them
>
> **摘要:** Antonym vs synonym distinction across multiple languages presents unique computational challenges due to the paradoxical nature of antonymous relationships words that share semantic domains while expressing opposite meanings. This work introduces Bhav-Net, a novel dual-space architecture that enables effective knowledge transfer from complex multilingual models to simpler, language-specific architectures while maintaining robust cross-lingual antonym--synonym distinction capabilities. Our approach combines language-specific BERT encoders with graph transformer networks, creating distinct semantic projections where synonymous pairs cluster in one space while antonymous pairs exhibit high similarity in a complementary space. Through comprehensive evaluation across eight languages (English, German, French, Spanish, Italian, Portuguese, Dutch, and Russian), we demonstrate that semantic relationship modeling transfers effectively across languages. The dual-encoder design achieves competitive performance against state-of-the-art baselines while providing interpretable semantic representations and effective cross-lingual generalization.
>
---
#### [replaced 020] Unsupervised Classification of English Words Based on Phonological Information: Discovery of Germanic and Latinate Clusters
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11770v3](http://arxiv.org/pdf/2504.11770v3)**

> **作者:** Takashi Morita; Timothy J. O'Donnell
>
> **摘要:** Cross-linguistically, native words and loanwords follow different phonological rules. In English, for example, words of Germanic and Latinate origin exhibit different stress patterns, and a certain syntactic structure, double-object datives, is predominantly associated with Germanic verbs rather than Latinate verbs. As a cognitive model, however, such etymology-based generalizations face challenges in terms of learnability, since the historical origins of words are presumably inaccessible information for general language learners. In this study, we present computational evidence indicating that the Germanic-Latinate distinction in the English lexicon is learnable from the phonotactic information of individual words. Specifically, we performed an unsupervised clustering on corpus-extracted words, and the resulting word clusters largely aligned with the etymological distinction. The model-discovered clusters also recovered various linguistic generalizations documented in the previous literature regarding the corresponding etymological classes. Moreover, our findings also uncovered previously unrecognized features of the quasi-etymological clusters.
>
---
#### [replaced 021] LLMs can hide text in other text of the same length
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20075v3](http://arxiv.org/pdf/2510.20075v3)**

> **作者:** Antonio Norelli; Michael Bronstein
>
> **备注:** 21 pages, main paper 9 pages
>
> **摘要:** A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
>
---
#### [replaced 022] Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.10524v3](http://arxiv.org/pdf/2507.10524v3)**

> **作者:** Sangmin Bae; Yujin Kim; Reza Bayat; Sungnyun Kim; Jiyoun Ha; Tal Schuster; Adam Fisch; Hrayr Harutyunyan; Ziwei Ji; Aaron Courville; Se-Young Yun
>
> **备注:** 38 pages, 9 figures, 17 tables, codes at https://github.com/raymin0223/mixture_of_recursions
>
> **摘要:** Scaling language models unlocks impressive capabilities, but the accompanying computational and memory demands make both training and deployment expensive. Existing efficiency efforts typically target either parameter sharing or adaptive computation, leaving open the question of how to attain both simultaneously. We introduce Mixture-of-Recursions (MoR), a unified framework that combines the two axes of efficiency inside a single Recursive Transformer. MoR reuses a shared stack of layers across recursion steps to achieve parameter efficiency, while lightweight routers enable adaptive token-level thinking by dynamically assigning different recursion depths to individual tokens. This allows MoR to focus quadratic attention computation only among tokens still active at a given recursion depth, further improving memory access efficiency by selectively caching only their key-value pairs. Beyond these core mechanisms, we also propose a KV sharing variant that reuses KV pairs from the first recursion, specifically designed to further decrease memory footprint. Across model scales ranging from 135M to 1.7B parameters, MoR forms a new Pareto frontier: at equal training FLOPs and smaller model sizes, it significantly lowers validation perplexity and improves few-shot accuracy, while delivering higher throughput compared with vanilla and existing recursive baselines. These gains demonstrate that MoR is an effective path towards large-model quality without incurring large-model cost.
>
---
#### [replaced 023] FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01068v3](http://arxiv.org/pdf/2502.01068v3)**

> **作者:** Dongwon Jo; Jiwon Song; Yulhwa Kim; Jae-Joon Kim
>
> **摘要:** While large language models (LLMs) excel at handling long-context sequences, they require substantial prefill computation and key-value (KV) cache, which can heavily burden computational efficiency and memory usage in both prefill and decoding stages. Recent works that compress KV caches with prefill acceleration reduce this cost but inadvertently tie the prefill compute reduction to the decoding KV budget. This coupling arises from overlooking the layer-dependent variation of critical context, often leading to accuracy degradation. To address this issue, we introduce FastKV, a KV cache compression framework designed to reduce latency in both prefill and decoding by leveraging the stabilization of token importance in later layers. FastKV performs full-context computation until a Token-Selective Propagation (TSP) layer, which forwards only the most informative tokens to subsequent layers. From these propagated tokens, FastKV independently selects salient KV entries for caching, thereby decoupling KV budget from the prefill compute reduction based on the TSP decision. This independent control of the TSP rate and KV retention rate enables flexible optimization of efficiency and accuracy. Experimental results show that FastKV achieves speedups of up to 1.82$\times$ in prefill and 2.87$\times$ in decoding compared to the full-context baseline, while matching the accuracy of the baselines that only accelerate the decoding stage. Our code is available at https://github.com/dongwonjo/FastKV.
>
---
#### [replaced 024] When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19158v2](http://arxiv.org/pdf/2502.19158v2)**

> **作者:** Yijiang River Dong; Tiancheng Hu; Yinhong Liu; Ahmet Üstün; Nigel Collier
>
> **摘要:** While Reinforcement Learning from Human Feedback (RLHF) is widely used to align Large Language Models (LLMs) with human preferences, it typically assumes homogeneous preferences across users, overlooking diverse human values and minority viewpoints. Although personalized preference learning addresses this by tailoring separate preferences for individual users, the field lacks standardized methods to assess its effectiveness. We present a multi-faceted evaluation framework that measures not only performance but also fairness, unintended effects, and adaptability across varying levels of preference divergence. Through extensive experiments comparing eight personalization methods across three preference datasets, we demonstrate that performance differences between methods could reach 36% when users strongly disagree, and personalization can introduce up to 20% safety misalignment. These findings highlight the critical need for holistic evaluation approaches to advance the development of more effective and inclusive preference learning systems.
>
---
#### [replaced 025] The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.16712v2](http://arxiv.org/pdf/2510.16712v2)**

> **作者:** Shivam Ratnakar; Sanjay Raghavendra
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: MTI-LLM @ NeurIPS 2025
>
> **摘要:** Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support.
>
---
#### [replaced 026] ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.10999v2](http://arxiv.org/pdf/2502.10999v2)**

> **作者:** Bowen Jiang; Yuan Yuan; Xinyi Bai; Zhuoqun Hao; Alyson Yin; Yaojie Hu; Wenyu Liao; Lyle Ungar; Camillo J. Taylor
>
> **备注:** The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Findings
>
> **摘要:** This work demonstrates that diffusion models can achieve font-controllable multilingual text rendering using just raw images without font label annotations.Visual text rendering remains a significant challenge. While recent methods condition diffusion on glyphs, it is impossible to retrieve exact font annotations from large-scale, real-world datasets, which prevents user-specified font control. To address this, we propose a data-driven solution that integrates the conditional diffusion model with a text segmentation model, utilizing segmentation masks to capture and represent fonts in pixel space in a self-supervised manner, thereby eliminating the need for any ground-truth labels and enabling users to customize text rendering with any multilingual font of their choice. The experiment provides a proof of concept of our algorithm in zero-shot text and font editing across diverse fonts and languages, providing valuable insights for the community and industry toward achieving generalized visual text rendering. Code is available at github.com/bowen-upenn/ControlText.
>
---
#### [replaced 027] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22453v2](http://arxiv.org/pdf/2505.22453v2)**

> **作者:** Lai Wei; Yuting Li; Chen Wang; Yue Wang; Linghe Kong; Weiran Huang; Lichao Sun
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Improving Multi-modal Large Language Models (MLLMs) in the post-training stage typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL), which require expensive and manually annotated multi-modal data--an ultimately unsustainable resource. This limitation has motivated a growing interest in unsupervised paradigms as a third stage of post-training after SFT and RL. While recent efforts have explored this direction, their methods are complex and difficult to iterate. To address this, we propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs, enabling continual self-improvement without any external supervision. The training method of MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that such training method effectively improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3\%$\rightarrow$72.9\% on MathVista, 62.9\%$\rightarrow$68.7\% on We-Math), using standard dataset without ground truth labels. To further explore scalability, we extend our framework to a data self-generation setting, designing two strategies that prompt the MLLM to synthesize new training samples on its own. Additional experiments show that combining these synthetic data with the unsupervised training method can also boost performance, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for autonomous enhancement of MLLMs, serving as a critical third step after initial SFT and RL in the absence of external supervision. Our code is available at https://github.com/waltonfuture/MM-UPT.
>
---
#### [replaced 028] Cohort Discovery: A Survey on LLM-Assisted Clinical Trial Recruitment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.15301v2](http://arxiv.org/pdf/2506.15301v2)**

> **作者:** Shrestha Ghosh; Moritz Schneider; Carina Reinicke; Carsten Eickhoff
>
> **摘要:** Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions.
>
---
#### [replaced 029] Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.07976v4](http://arxiv.org/pdf/2508.07976v4)**

> **作者:** Jiaxuan Gao; Wei Fu; Minyang Xie; Shusheng Xu; Chuyi He; Zhiyu Mei; Banghua Zhu; Yi Wu
>
> **摘要:** Recent advancements in LLM-based agents have demonstrated remarkable capabilities in handling complex, knowledge-intensive tasks by integrating external tools. Among diverse choices of tools, search tools play a pivotal role in accessing vast external knowledge. However, open-source agents still fall short of achieving expert-level Search Intelligence, the ability to resolve ambiguous queries, generate precise searches, analyze results, and conduct thorough exploration. Existing approaches fall short in scalability, efficiency, and data quality. For example, small turn limits in existing online RL methods, e.g. <=10, restrict complex strategy learning. This paper introduces ASearcher, an open-source project for large-scale RL training of search agents. Our key contributions include: (1) Scalable fully asynchronous RL training that enables long-horizon search while maintaining high training efficiency. (2) A prompt-based LLM agent that autonomously synthesizes high-quality and challenging QAs, creating a large-scale QA dataset. Through RL training, our prompt-based QwQ-32B agent achieves substantial improvements, with 78.0% and 34.3% Avg@4 gains on xBench and GAIA, respectively. Notably, our agent exhibits extreme long-horizon search, with tool calls exceeding 100 turns and output tokens exceeding 400k during training time. With a simple agent design and no external LLMs, ASearcher-Web-QwQ achieves Avg@4 scores of 51.1 on xBench and 58.7 on GAIA, surpassing existing open-source 32B agents. Finally, we also show that ASearcher-Web-QwQ could achieve performance of commercial systems using external summary tool in a zero-shot transfer manner and test-time search. We open-source our models, training data, and codes in https://github.com/inclusionAI/ASearcher.
>
---
#### [replaced 030] Steering Evaluation-Aware Language Models to Act Like They Are Deployed
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.20487v2](http://arxiv.org/pdf/2510.20487v2)**

> **作者:** Tim Tian Hua; Andrew Qin; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on documents with factual descriptions of the model (1) using Python type hints during evaluation but not during deployment and (2) recognizing that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. We find that activation steering can suppress evaluation awareness and make the model act like it is deployed even when the cue is present. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed.
>
---
#### [replaced 031] Cancer-Myth: Evaluating AI Chatbot on Patient Questions with False Presuppositions
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2504.11373v2](http://arxiv.org/pdf/2504.11373v2)**

> **作者:** Wang Bill Zhu; Tianqi Chen; Xinyan Velocity Yu; Ching Ying Lin; Jade Law; Mazen Jizzini; Jorge J. Nieva; Ruishan Liu; Robin Jia
>
> **摘要:** Cancer patients are increasingly turning to large language models (LLMs) for medical information, making it critical to assess how well these models handle complex, personalized questions. However, current medical benchmarks focus on medical exams or consumer-searched questions and do not evaluate LLMs on real patient questions with patient details. In this paper, we first have three hematology-oncology physicians evaluate cancer-related questions drawn from real patients. While LLM responses are generally accurate, the models frequently fail to recognize or address false presuppositions in the questions, posing risks to safe medical decision-making. To study this limitation systematically, we introduce Cancer-Myth, an expert-verified adversarial dataset of 585 cancer-related questions with false presuppositions. On this benchmark, no frontier LLM -- including GPT-5, Gemini-2.5-Pro, and Claude-4-Sonnet -- corrects these false presuppositions more than $43\%$ of the time. To study mitigation strategies, we further construct a 150-question Cancer-Myth-NFP set, in which physicians confirm the absence of false presuppositions. We find typical mitigation strategies, such as adding precautionary prompts with GEPA optimization, can raise accuracy on Cancer-Myth to $80\%$, but at the cost of misidentifying presuppositions in $41\%$ of Cancer-Myth-NFP questions and causing a $10\%$ relative performance drop on other medical benchmarks. These findings highlight a critical gap in the reliability of LLMs, show that prompting alone is not a reliable remedy for false presuppositions, and underscore the need for more robust safeguards in medical AI systems.
>
---
#### [replaced 032] Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06229v5](http://arxiv.org/pdf/2507.06229v5)**

> **作者:** Xiangru Tang; Tianrui Qin; Tianhao Peng; Ziyang Zhou; Daniel Shao; Tingting Du; Xinming Wei; Peng Xia; Fang Wu; He Zhu; Ge Zhang; Jiaheng Liu; Xingyao Wang; Sirui Hong; Chenglin Wu; Hao Cheng; Chi Wang; Wangchunshu Zhou
>
> **摘要:** AI agent frameworks operate in isolation, forcing agents to rediscover solutions and repeat mistakes across different systems. Despite valuable problem-solving experiences accumulated by frameworks like smolagents, OpenHands, and OWL, this knowledge remains trapped within individual systems, preventing the emergence of collective intelligence. Current memory systems focus on individual agents or framework-specific demonstrations, failing to enable cross-architecture knowledge transfer. We introduce AGENT KB, a universal memory infrastructure enabling seamless experience sharing across heterogeneous agent frameworks without retraining. AGENT KB aggregates trajectories into a structured knowledge base and serves lightweight APIs. At inference time, hybrid retrieval operates through two stages: planning seeds agents with cross-domain workflows, while feedback applies targeted diagnostic fixes. A disagreement gate ensures retrieved knowledge enhances rather than disrupts reasoning, addressing knowledge interference in cross-framework transfer. We validate AGENT KB across major frameworks on GAIA, Humanity's Last Exam, GPQA, and SWE-bench. Results show substantial improvements across diverse model families: compared to baseline pass@1, smolagents with AGENT KB achieve up to 18.7pp gains at pass@3 (55.2% -> 73.9%), while OpenHands improves 4.0pp on SWE-bench pass@1 (24.3% -> 28.3%). Similar improvements are observed across all base model families. Ablations confirm that hybrid retrieval and feedback stages are essential, with automatically generated experiences matching manual curation. This establishes the foundation for collective agent intelligence through shared memory infrastructures.
>
---
#### [replaced 033] Assessing the Potential of Generative Agents in Crowdsourced Fact-Checking
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.19940v2](http://arxiv.org/pdf/2504.19940v2)**

> **作者:** Luigia Costabile; Gian Marco Orlando; Valerio La Gatta; Vincenzo Moscato
>
> **备注:** This paper has been published in Online Social Networks and Media (https://doi.org/10.1016/j.osnem.2025.100326). Please cite the published version accordingly
>
> **摘要:** The growing spread of online misinformation has created an urgent need for scalable, reliable fact-checking solutions. Crowdsourced fact-checking - where non-experts evaluate claim veracity - offers a cost-effective alternative to expert verification, despite concerns about variability in quality and bias. Encouraged by promising results in certain contexts, major platforms such as X (formerly Twitter), Facebook, and Instagram have begun shifting from centralized moderation to decentralized, crowd-based approaches. In parallel, advances in Large Language Models (LLMs) have shown strong performance across core fact-checking tasks, including claim detection and evidence evaluation. However, their potential role in crowdsourced workflows remains unexplored. This paper investigates whether LLM-powered generative agents - autonomous entities that emulate human behavior and decision-making - can meaningfully contribute to fact-checking tasks traditionally reserved for human crowds. Using the protocol of La Barbera et al. (2024), we simulate crowds of generative agents with diverse demographic and ideological profiles. Agents retrieve evidence, assess claims along multiple quality dimensions, and issue final veracity judgments. Our results show that agent crowds outperform human crowds in truthfulness classification, exhibit higher internal consistency, and show reduced susceptibility to social and cognitive biases. Compared to humans, agents rely more systematically on informative criteria such as Accuracy, Precision, and Informativeness, suggesting a more structured decision-making process. Overall, our findings highlight the potential of generative agents as scalable, consistent, and less biased contributors to crowd-based fact-checking systems.
>
---
#### [replaced 034] SafeCOMM: A Study on Safety Degradation in Fine-Tuned Telecom Large Language Models
- **分类: cs.CY; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00062v2](http://arxiv.org/pdf/2506.00062v2)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Farhan Ahmed; Syed Zawad; Fernando Koch; Walid Saad; Holger Boche
>
> **摘要:** Fine-tuning large language models (LLMs) on telecom datasets is a common practice to adapt general-purpose models to the telecom domain. However, little attention has been paid to how this process may compromise model safety. Recent research has shown that even benign fine-tuning can degrade the safety alignment of LLMs, causing them to respond to harmful or unethical user queries. In this paper, we investigate this issue by fine-tuning LLMs on three representative telecom datasets and show that safety degrades even for light telecom domain adaptation. To this end, we introduce TeleHarm, the first telecom-specific red-teaming benchmark, which we use alongside established Direct-Harm and HexPhi datasets to systematically assess harmful behavior. We further extend our analysis to publicly available TeleLLMs that were continually pre-trained on large telecom corpora, revealing that safety alignment is severely lacking, primarily due to the omission of safety-focused instruction tuning. To address these issues, we evaluate three realignment defenses: SafeInstruct, SafeLoRA, SafeMERGE. We show that, across all settings, the proposed defenses can effectively restore safety without compromising telecom task performance, leading to Safe teleCOMMunication (SafeCOMM) models. Our work serves as both a diagnostic study and practical guide for safety realignment in telecom-tuned LLMs, underscoring the need for safety-aware instruction and fine-tuning in the telecom domain.
>
---
#### [replaced 035] COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17701v3](http://arxiv.org/pdf/2505.17701v3)**

> **作者:** Jaewon Cheon; Pilsung Kang
>
> **备注:** EMNLP 2025 (Main Track)
>
> **摘要:** The growing size of large language models has created significant computational inefficiencies. To address this challenge, sparse activation methods selectively deactivates non-essential parameters during inference, reducing computational costs in FFNN layers. While existing methods focus on non-linear gating mechanisms, we hypothesize that the sparsity of the FFNN layer lies globally in the form of a linear combination over its internal down projection matrix. Based on this insight, we propose two methods: M-COUNTDOWN, leveraging indirect coefficients, and D-COUNTDOWN, utilizing direct coefficients of the linear combination. Experimental results demonstrate that D-COUNTDOWN can omit 90% of computations with performance loss as low as 5.5% ideally, while M-COUNTDOWN provides a predictor-free solution with up to 29.4% better performance preservation compared to existing methods. Our specialized kernel implementations effectively realize these theoretical gains into substantial real-world acceleration.
>
---
#### [replaced 036] Unified Sparse Mixture of Experts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22996v2](http://arxiv.org/pdf/2503.22996v2)**

> **作者:** Giang Do; Hung Le; Truyen Tran
>
> **备注:** 26 pages
>
> **摘要:** Sparse Mixture of Experts (SMoEs) models scale the capacity of models while maintaining constant computational overhead. Early designs typically relied on a fixed value of $k$, where $k$ represents either the number of experts selected per token or the number of tokens assigned per expert. However, these approaches encounter three key limitations: they may fail to route to important experts or tokens, may assign irrelevant ones, and often suffer from representation collapse among experts. This paper reexamines SMoEs through the lens of \textit{Linear Programming}, and proposes a Unified Sparse Mixture of Experts (USMoE) framework that addresses these limitations. Specifically, our approach introduces a unified mechanism that integrates information from both the expert and token dimensions, and a unified scoring function that linearly combines similarity scores between experts and tokens. We provide both theoretical justification and empirical evidence demonstrating USMoE's effectiveness in overcoming the limitations of traditional routing methods. Through comprehensive evaluations on both clean and corrupted settings for large language models and vision tasks, under both training-free and training scenarios, USMoE achieves up to a 10\% performance improvement over standard approaches or reduces inference costs by up to 14\%, while maintaining competitive accuracy.
>
---
#### [replaced 037] Preference Optimization by Estimating the Ratio of the Data Distribution
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19601v2](http://arxiv.org/pdf/2505.19601v2)**

> **作者:** Yeongmin Kim; Heesun Bae; Byeonghu Na; Il-Chul Moon
>
> **备注:** NeurIPS 2025
>
> **摘要:** Direct preference optimization (DPO) is widely used as a simple and stable method for aligning large language models (LLMs) with human preferences. This paper investigates a generalized DPO loss that enables a policy model to match the target policy from a likelihood ratio estimation perspective. The ratio of the target policy provides a unique identification of the policy distribution without relying on reward models or partition functions. This allows the generalized loss to retain both simplicity and theoretical guarantees, which prior work such as $f$-PO fails to achieve simultaneously. We propose Bregman preference optimization (BPO), a generalized framework for ratio matching that provides a family of objective functions achieving target policy optimality. BPO subsumes DPO as a special case and offers tractable forms for all instances, allowing implementation with a few lines of code. We further develop scaled Basu's power divergence (SBA), a gradient scaling method that can be used for BPO instances. The BPO framework complements other DPO variants and is applicable to target policies defined by these variants. In experiments, unlike other probabilistic loss extensions such as $f$-DPO or $f$-PO, which exhibit a trade-off between generation fidelity and diversity, instances of BPO improve both win rate and entropy compared with DPO. When applied to Llama-3-8B-Instruct, BPO achieves state-of-the-art performance among Llama-3-8B backbones, with a 55.9\% length-controlled win rate on AlpacaEval2. Project page: https://github.com/aailab-kaist/BPO.
>
---
#### [replaced 038] StereoDetect: Detecting Stereotypes and Anti-stereotypes the Correct Way Using Social Psychological Underpinnings
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.03352v3](http://arxiv.org/pdf/2504.03352v3)**

> **作者:** Kaustubh Shivshankar Shejole; Pushpak Bhattacharyya
>
> **摘要:** Stereotypes are known to have very harmful effects, making their detection critically important. However, current research predominantly focuses on detecting and evaluating stereotypical biases, thereby leaving the study of stereotypes in its early stages. Our study revealed that many works have failed to clearly distinguish between stereotypes and stereotypical biases, which has significantly slowed progress in advancing research in this area. Stereotype and Anti-stereotype detection is a problem that requires social knowledge; hence, it is one of the most difficult areas in Responsible AI. This work investigates this task, where we propose a five-tuple definition and provide precise terminologies disentangling stereotypes, anti-stereotypes, stereotypical bias, and general bias. We provide a conceptual framework grounded in social psychology for reliable detection. We identify key shortcomings in existing benchmarks for this task of stereotype and anti-stereotype detection. To address these gaps, we developed StereoDetect, a well curated, definition-aligned benchmark dataset designed for this task. We show that sub-10B language models and GPT-4o frequently misclassify anti-stereotypes and fail to recognize neutral overgeneralizations. We demonstrate StereoDetect's effectiveness through multiple qualitative and quantitative comparisons with existing benchmarks and models fine-tuned on them. The dataset and code is available at https://github.com/KaustubhShejole/StereoDetect.
>
---
#### [replaced 039] WolBanking77: Wolof Banking Speech Intent Classification Dataset
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19271v3](http://arxiv.org/pdf/2509.19271v3)**

> **作者:** Abdou Karim Kandji; Frédéric Precioso; Cheikh Ba; Samba Ndiaye; Augustin Ndione
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Intent classification models have made a significant progress in recent years. However, previous studies primarily focus on high-resource language datasets, which results in a gap for low-resource languages and for regions with high rates of illiteracy, where languages are more spoken than read or written. This is the case in Senegal, for example, where Wolof is spoken by around 90\% of the population, while the national illiteracy rate remains at of 42\%. Wolof is actually spoken by more than 10 million people in West African region. To address these limitations, we introduce the Wolof Banking Speech Intent Classification Dataset (WolBanking77), for academic research in intent classification. WolBanking77 currently contains 9,791 text sentences in the banking domain and more than 4 hours of spoken sentences. Experiments on various baselines are conducted in this work, including text and voice state-of-the-art models. The results are very promising on this current dataset. In addition, this paper presents an in-depth examination of the dataset's contents. We report baseline F1-scores and word error rates metrics respectively on NLP and ASR models trained on WolBanking77 dataset and also comparisons between models. Dataset and code available at: https://github.com/abdoukarim/wolbanking77.
>
---
#### [replaced 040] FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19207v2](http://arxiv.org/pdf/2502.19207v2)**

> **作者:** Nakyeong Yang; Minsung Kim; Seunghyun Yoon; Joongbo Shin; Kyomin Jung
>
> **备注:** accepted to EMNLP 2025
>
> **摘要:** Various studies have attempted to remove sensitive or private knowledge from a language model to prevent its unauthorized exposure. However, prior studies have overlooked the complex and interconnected nature of knowledge, where related knowledge must be carefully examined. Specifically, they have failed to evaluate whether an unlearning method faithfully erases interconnected knowledge that should be removed, retaining knowledge that appears relevant but exists in a completely different context. To resolve this problem, we first define a new concept called superficial unlearning, which refers to the phenomenon where an unlearning method either fails to erase the interconnected knowledge it should remove or unintentionally erases irrelevant knowledge. Based on the definition, we introduce a new benchmark, FaithUn, to analyze and evaluate the faithfulness of unlearning in real-world knowledge QA settings. Furthermore, we propose a novel unlearning method, KLUE, which updates only knowledge-related neurons to achieve faithful unlearning. KLUE identifies knowledge neurons using an explainability method and updates only those neurons using selected unforgotten samples. Experimental results demonstrate that widely-used unlearning methods fail to ensure faithful unlearning, while our method shows significant effectiveness in real-world QA unlearning.
>
---
#### [replaced 041] ThinkBrake: Mitigating Overthinking in Tool Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00546v2](http://arxiv.org/pdf/2510.00546v2)**

> **作者:** Minjae Oh; Sangjun Song; Seungkyu Lee; Sungmin Jo; Yohan Jo
>
> **摘要:** Small reasoning models (SRMs) often overthink during tool use: they reach a correct tool-argument configuration, then continue reasoning and overwrite it with an incorrect final call. We diagnose overthinking via oracle rollouts that inject </think> at sentence boundaries. On the Berkeley Function Calling Leaderboard (BFCL), this oracle termination lifts average accuracy from 85.8\% to 94.2\% while reducing tokens by 80-94\%, revealing substantial recoverable headroom and potential redundant reasoning. While prior work on concise reasoning has largely targeted mathematics, tool reasoning remains underexplored. We adapt various early-termination baselines to tool use and introduce ThinkBrake, a training-free decoding heuristic. ThinkBrake monitors the log-probability margin between </think> and the current top token at sentence boundaries and triggers termination when this margin becomes small. Across BFCL's single turn, non-live and live splits, ThinkBrake preserves or improves accuracy while reducing tokens up to 25\%, outperforming various baselines.
>
---
#### [replaced 042] The Cross-Lingual Cost: Retrieval Biases in RAG over Arabic-English Corpora
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.07543v2](http://arxiv.org/pdf/2507.07543v2)**

> **作者:** Chen Amiraz; Yaroslav Fyodorov; Elad Haramaty; Zohar Karnin; Liane Lewin-Eytan
>
> **备注:** Accepted to ArabicNLP 2025
>
> **摘要:** Cross-lingual retrieval-augmented generation (RAG) is a critical capability for retrieving and generating answers across languages. Prior work in this context has mostly focused on generation and relied on benchmarks derived from open-domain sources, most notably Wikipedia. In such settings, retrieval challenges often remain hidden due to language imbalances, overlap with pretraining data, and memorized content. To address this gap, we study Arabic-English RAG in a domain-specific setting using benchmarks derived from real-world corporate datasets. Our benchmarks include all combinations of languages for the user query and the supporting document, drawn independently and uniformly at random. This enables a systematic study of multilingual retrieval behavior. Our findings reveal that retrieval is a critical bottleneck in cross-lingual domain-specific scenarios, with substantial performance drops occurring when the user query and supporting document languages differ. A key insight is that these failures stem primarily from the retriever's difficulty in ranking documents across languages. Finally, we propose two simple retrieval strategies that address this source of failure by enforcing equal retrieval from both languages or by translating the query, resulting in substantial improvements in cross-lingual and overall performance. These results highlight meaningful opportunities for improving multilingual retrieval, particularly in practical, real-world RAG applications.
>
---
#### [replaced 043] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19363v2](http://arxiv.org/pdf/2510.19363v2)**

> **作者:** Siyuan Wang; Gaokai Zhang; Li Lyna Zhang; Ning Shang; Fan Yang; Dongyao Chen; Mao Yang
>
> **摘要:** Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities.
>
---
#### [replaced 044] GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23671v3](http://arxiv.org/pdf/2505.23671v3)**

> **作者:** Manish Shetty; Naman Jain; Jinjian Liu; Vijay Kethanaboyina; Koushik Sen; Ion Stoica
>
> **备注:** Website: https://gso-bench.github.io/
>
> **摘要:** Developing high-performance software is a complex task that requires specialized expertise. We introduce GSO, a benchmark for evaluating language models' capabilities in developing high-performance software. We develop an automated pipeline that generates and executes performance tests to analyze repository commit histories to identify 102 challenging optimization tasks across 10 codebases, spanning diverse domains and programming languages. An agent is provided with a codebase and performance test as a precise specification, and tasked to improve the runtime efficiency, which is measured against the expert developer optimization. Our quantitative evaluation reveals that leading SWE-Agents struggle significantly, achieving less than 5% success rate, with limited improvements even with inference-time scaling. Our qualitative analysis identifies key failure modes, including difficulties with low-level languages, practicing lazy optimization strategies, and challenges in accurately localizing bottlenecks. We release the code and artifacts of our benchmark along with agent trajectories to enable future research.
>
---
#### [replaced 045] Probabilistic adaptation of language comprehension for individual speakers: evidence from neural oscillations
- **分类: q-bio.NC; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01299v2](http://arxiv.org/pdf/2502.01299v2)**

> **作者:** Hanlin Wu; Xiaohui Rao; Zhenguang G Cai
>
> **摘要:** Listeners adapt language comprehension based on their mental representations of speakers, but how these representations are updated remains unclear. We investigated whether listeners probabilistically adapt comprehension based on the frequency of speakers making stereotype-incongruent statements. In two EEG experiments, participants heard speakers make stereotype-congruent or incongruent statements, with incongruency base rate manipulated. In Experiment 1, stereotype-incongruent statements decreased high-beta (21-30 Hz) and theta (4-6 Hz) oscillatory power in the low base rate condition but increased it in the high base rate condition. The theta effect varied with listeners' openness trait: less open-minded participants tended to show theta increases to stereotype incongruencies, while more open-minded participants tended to show theta decreases. In Experiment 2, we dissociated incongruency base rate from the target speaker by manipulating it using a non-target speaker and found that only the high-beta effect persisted. Our findings reveal two potential mechanisms: a speaker-general mechanism (indicated by high-beta oscillations) that adjusts overall expectations about hearing statements that violate social stereotypes, and a speaker-specific mechanism (indicated by theta oscillations) that updates a more detailed mental model specifically about an individual speaker. These findings provide evidence for how language processing interacts with social cognition.
>
---
#### [replaced 046] Unsupervised Document and Template Clustering using Multimodal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12116v3](http://arxiv.org/pdf/2506.12116v3)**

> **作者:** Phillipe R. Sampaio; Helene Maxcici
>
> **备注:** 24 pages, 12 figures
>
> **摘要:** We study unsupervised clustering of documents at both the category and template levels using frozen multimodal encoders and classical clustering algorithms. We systematize a model-agnostic pipeline that (i) projects heterogeneous last-layer states from text-layout-vision encoders into token-type-aware document vectors and (ii) performs clustering with centroid- or density-based methods, including an HDBSCAN + $k$-NN assignment to eliminate unlabeled points. We evaluate eight encoders (text-only, layout-aware, vision-only, and vision-language) with $k$-Means, DBSCAN, HDBSCAN + $k$-NN, and BIRCH on five corpora spanning clean synthetic invoices, their heavily degraded print-and-scan counterparts, scanned receipts, and real identity and certificate documents. The study reveals modality-specific failure modes and a robustness-accuracy trade-off, with vision features nearly solving template discovery on clean pages while text dominates under covariate shift, and fused encoders offering the best balance. We detail a reproducible, oracle-free tuning protocol and the curated evaluation settings to guide future work on unsupervised document organization.
>
---
#### [replaced 047] Reasoning is Periodicity? Improving Large Language Models Through Effective Periodicity Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.21309v4](http://arxiv.org/pdf/2502.21309v4)**

> **作者:** Yihong Dong; Ge Li; Xue Jiang; Yongding Tao; Kechi Zhang; Hao Zhu; Huanyu Liu; Jiazheng Ding; Jia Li; Jinliang Deng; Hong Mei
>
> **备注:** Accepted to NeurIPS'25
>
> **摘要:** Periodicity, as one of the most important basic characteristics, lays the foundation for facilitating structured knowledge acquisition and systematic cognitive processes within human learning paradigms. However, the potential flaws of periodicity modeling in Transformer affect the learning efficiency and establishment of underlying principles from data for large language models (LLMs) built upon it. In this paper, we demonstrate that integrating effective periodicity modeling can improve the learning efficiency and performance of LLMs. We introduce FANformer, which adapts Fourier Analysis Network (FAN) into attention mechanism to achieve efficient periodicity modeling, by modifying the feature projection process of attention mechanism. Extensive experimental results on language modeling show that FANformer consistently outperforms Transformer when scaling up model size and training tokens, underscoring its superior learning efficiency. Our pretrained FANformer-1B exhibits marked improvements on downstream tasks compared to open-source LLMs with similar model parameters or training tokens. Moreover, we reveal that FANformer exhibits superior ability to learn and apply rules for reasoning compared to Transformer. The results position FANformer as an effective and promising architecture for advancing LLMs.
>
---
#### [replaced 048] Less is More: Local Intrinsic Dimensions of Contextual Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01034v2](http://arxiv.org/pdf/2506.01034v2)**

> **作者:** Benjamin Matthias Ruppik; Julius von Rohrscheidt; Carel van Niekerk; Michael Heck; Renato Vukovic; Shutong Feng; Hsien-chin Lin; Nurul Lubis; Bastian Rieck; Marcus Zibrowius; Milica Gašić
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025; in press). 10 pages, with an additional 17 pages in the appendix. Our code is available at https://github.com/aidos-lab/Topo_LLM_public and https://github.com/aidos-lab/grokking-via-lid
>
> **摘要:** Understanding the internal mechanisms of large language models (LLMs) remains a challenging and complex endeavor. Even fundamental questions, such as how fine-tuning affects model behavior, often require extensive empirical evaluation. In this paper, we introduce a novel perspective based on the geometric properties of contextual latent embeddings to study the effects of training and fine-tuning. To that end, we measure the local dimensions of a contextual language model's latent space and analyze their shifts during training and fine-tuning. We show that the local dimensions provide insights into the model's training dynamics and generalization ability. Specifically, the mean of the local dimensions predicts when the model's training capabilities are exhausted, as exemplified in a dialogue state tracking task, overfitting, as demonstrated in an emotion recognition task, and grokking, as illustrated with an arithmetic task. Furthermore, our experiments suggest a practical heuristic: reductions in the mean local dimension tend to accompany and predict subsequent performance gains. Through this exploration, we aim to provide practitioners with a deeper understanding of the implications of fine-tuning on embedding spaces, facilitating informed decisions when configuring models for specific applications. The results of this work contribute to the ongoing discourse on the interpretability, adaptability, and generalizability of LLMs by bridging the gap between intrinsic model mechanisms and geometric properties in the respective embeddings.
>
---
#### [replaced 049] Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19028v2](http://arxiv.org/pdf/2510.19028v2)**

> **作者:** Eunsu Kim; Junyeong Park; Juhyun Oh; Kiwoong Park; Seyoung Song; A. Seza Doğruöz; Najoung Kim; Alice Oh
>
> **摘要:** As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models.
>
---
#### [replaced 050] UNO-Bench: A Unified Benchmark for Exploring the Compositional Law Between Uni-modal and Omni-modal in OmniModels
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.18915v2](http://arxiv.org/pdf/2510.18915v2)**

> **作者:** Chen Chen; ZeYang Hu; Fengjiao Chen; Liya Ma; Jiaxing Liu; Xiaoyu Li; Xuezhi Cao
>
> **备注:** v2: New title and new abstract. Updated evaluation results and analysis. The benchmark name has been updated to UNO-Bench from MMAO-Bench. Work in progress. Code and data are available at https://github.com/meituan-longcat/UNO-Bench
>
> **摘要:** Multimodal Large Languages models have been progressing from uni-modal understanding toward unifying visual, audio and language modalities, collectively termed omni models. However, the correlation between uni-modal and omni-modal remains unclear, which requires comprehensive evaluation to drive omni model's intelligence evolution. In this work, we propose a novel, high quality and UNified Omni model benchmark, UNO-Bench, which effectively assesses both UNi-modal and Omni-modal capabilities. The benchmark consists of 3730 human curated samples, with 98% cross-modality solvability, across 44 task types, and an innovative multi-step open-ended question type for assessing complex reasoning. Besides, a general scoring model supporting 6 question types is proposed for automated evaluation with 95% accuracy. Experimental result shows the Compositional Law between omni-modal and uni-modal performance and the omni-modal capability manifests as a bottleneck effect on weak models, while exhibiting synergistic promotion on strong models. The code and data are available at https://github.com/meituan-longcat/UNO-Bench
>
---
#### [replaced 051] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.20445v4](http://arxiv.org/pdf/2410.20445v4)**

> **作者:** Yuwei Du; Jie Feng; Jie Zhao; Yong Li
>
> **备注:** Accepted by NeurIPS 2025, https://github.com/tsinghua-fib-lab/TrajAgent
>
> **摘要:** Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. \fix In this paper, we propose \textit{TrajAgent}, a agent framework powered by large language models (LLMs), designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. \unfix~In \textit{TrajAgent}, we first develop \textit{UniEnv}, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on \textit{UniEnv}, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on four tasks using four real-world datasets demonstrate the effectiveness of \textit{TrajAgent} in automated trajectory modeling, achieving a performance improvement of \fix 2.38\%-69.91\% \unfix over baseline methods. The codes and data can be accessed via https://github.com/tsinghua-fib-lab/TrajAgent.
>
---
#### [replaced 052] Constrained Entropic Unlearning: A Primal-Dual Framework for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05314v2](http://arxiv.org/pdf/2506.05314v2)**

> **作者:** Taha Entesari; Arman Hatami; Rinat Khaziev; Anil Ramakrishna; Mahyar Fazlyab
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems
>
> **摘要:** Large Language Models (LLMs) deployed in real-world settings increasingly face the need to unlearn sensitive, outdated, or proprietary information. Existing unlearning methods typically formulate forgetting and retention as a regularized trade-off, combining both objectives into a single scalarized loss. This often leads to unstable optimization and degraded performance on retained data, especially under aggressive forgetting. We propose a new formulation of LLM unlearning as a constrained optimization problem: forgetting is enforced via a novel logit-margin flattening loss that explicitly drives the output distribution toward uniformity on a designated forget set, while retention is preserved through a hard constraint on a separate retain set. Compared to entropy-based objectives, our loss is softmax-free, numerically stable, and maintains non-vanishing gradients, enabling more efficient and robust optimization. We solve the constrained problem using a scalable primal-dual algorithm that exposes the trade-off between forgetting and retention through the dynamics of the dual variable, all without any extra computational overhead. Evaluations on the TOFU and MUSE benchmarks across diverse LLM architectures demonstrate that our approach consistently matches or exceeds state-of-the-art baselines, effectively removing targeted information while preserving downstream utility.
>
---
#### [replaced 053] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.04077v3](http://arxiv.org/pdf/2502.04077v3)**

> **作者:** Qingyue Yang; Jie Wang; Xing Li; Zhihai Wang; Chen Chen; Lei Chen; Xianzhi Yu; Wulong Liu; Jianye Hao; Mingxuan Yuan; Bin Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** With the development of large language models (LLMs), efficient inference through Key-Value (KV) cache compression has attracted considerable attention, especially for long-context generation. To compress the KV cache, recent methods identify critical KV tokens through static modeling of attention scores. However, these methods often struggle to accurately determine critical tokens as they neglect the temporal patterns in attention scores, resulting in a noticeable degradation in LLM performance. To address this challenge, we propose AttentionPredictor, which is the first learning-based method to directly predict attention patterns for KV cache compression and critical token identification. Specifically, AttentionPredictor learns a lightweight, unified convolution model to dynamically capture spatiotemporal patterns and predict the next-token attention scores. An appealing feature of AttentionPredictor is that it accurately predicts the attention score and shares the unified prediction model, which consumes negligible memory, among all transformer layers. Moreover, we propose a cross-token critical cache prefetching framework that hides the token estimation time overhead to accelerate the decoding stage. By retaining most of the attention information, AttentionPredictor achieves 13$\times$ KV cache compression and 5.6$\times$ speedup in a cache offloading scenario with comparable LLM performance, significantly outperforming the state-of-the-arts. The code is available at https://github.com/MIRALab-USTC/LLM-AttentionPredictor.
>
---
#### [replaced 054] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.10114v2](http://arxiv.org/pdf/2510.10114v2)**

> **作者:** Luyao Zhuang; Shengyuan Chen; Yilin Xiao; Huachi Zhou; Yujing Zhang; Hao Chen; Qinggang Zhang; Xiao Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) is widely used to mitigate hallucinations of Large Language Models (LLMs) by leveraging external knowledge. While effective for simple queries, traditional RAG systems struggle with large-scale, unstructured corpora where information is fragmented. Recent advances incorporate knowledge graphs to capture relational structures, enabling more comprehensive retrieval for complex, multi-hop reasoning tasks. However, existing graph-based RAG (GraphRAG) methods rely on unstable and costly relation extraction for graph construction, often producing noisy graphs with incorrect or inconsistent relations that degrade retrieval quality. In this paper, we revisit the pipeline of existing GraphRAG systems and propose LinearRAG (Linear Graph-based Retrieval-Augmented Generation), an efficient framework that enables reliable graph construction and precise passage retrieval. Specifically, LinearRAG constructs a relation-free hierarchical graph, termed Tri-Graph, using only lightweight entity extraction and semantic linking, avoiding unstable relation modeling. This new paradigm of graph construction scales linearly with corpus size and incurs no extra token consumption, providing an economical and reliable indexing of the original passages. For retrieval, LinearRAG adopts a two-stage strategy: (i) relevant entity activation via local semantic bridging, followed by (ii) passage retrieval through global importance aggregation. Extensive experiments on four datasets demonstrate that LinearRAG significantly outperforms baseline models.
>
---
#### [replaced 055] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.16024v2](http://arxiv.org/pdf/2503.16024v2)**

> **作者:** Ruihan Yang; Fanghua Ye; Jian Li; Siyu Yuan; Yikai Zhang; Zhaopeng Tu; Xiaolong Li; Deqing Yang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models (LLMs) have recently transformed from text-based assistants to autonomous agents capable of planning, reasoning, and iteratively improving their actions. While numerical reward signals and verifiers can effectively rank candidate actions, they often provide limited contextual guidance. In contrast, natural language feedback better aligns with the generative capabilities of LLMs, providing richer and more actionable suggestions. However, parsing and implementing this feedback effectively can be challenging for LLM-based agents. In this work, we introduce Critique-Guided Improvement (CGI), a novel two-player framework, comprising an actor model that explores an environment and a critic model that generates detailed nature language feedback. By training the critic to produce fine-grained assessments and actionable revisions, and the actor to utilize these critiques, our approach promotes more robust exploration of alternative strategies while avoiding local optima. Experiments in three interactive environments show that CGI outperforms existing baselines by a substantial margin. Notably, even a small critic model surpasses GPT-4 in feedback quality. The resulting actor achieves state-of-the-art performance, demonstrating the power of explicit iterative guidance to enhance decision-making in LLM-based agents.
>
---
#### [replaced 056] MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations
- **分类: cs.CL; cs.AI; cs.SI**

- **链接: [http://arxiv.org/pdf/2504.07830v3](http://arxiv.org/pdf/2504.07830v3)**

> **作者:** Genglin Liu; Vivian Le; Salman Rahman; Elisa Kreiss; Marzyeh Ghassemi; Saadia Gabriel
>
> **备注:** Accepted into EMNLP 2025 Main Conference, Oral Presentation
>
> **摘要:** We present a novel, open-source social network simulation framework, MOSAIC, where generative language agents predict user behaviors such as liking, sharing, and flagging content. This simulation combines LLM agents with a directed social graph to analyze emergent deception behaviors and gain a better understanding of how users determine the veracity of online social content. By constructing user representations from diverse fine-grained personas, our system enables multi-agent simulations that model content dissemination and engagement dynamics at scale. Within this framework, we evaluate three different content moderation strategies with simulated misinformation dissemination, and we find that they not only mitigate the spread of non-factual content but also increase user engagement. In addition, we analyze the trajectories of popular content in our simulations, and explore whether simulation agents' articulated reasoning for their social interactions truly aligns with their collective engagement patterns. We open-source our simulation software to encourage further research within AI and social sciences.
>
---
#### [replaced 057] Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.06522v2](http://arxiv.org/pdf/2506.06522v2)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Syed Zawad; Farhan Ahmed; Heiko Ludwig; Holger Boche
>
> **摘要:** Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
>
---
#### [replaced 058] Multi-turn Training with Basic Human Feedback Helps Little on LLM Reasoning
- **分类: cs.CL; cs.IT; cs.LG; math.IT**

- **链接: [http://arxiv.org/pdf/2510.21339v2](http://arxiv.org/pdf/2510.21339v2)**

> **作者:** Qiang Liu; Wuganjing Song; Zhenzhou Lin; Feifan Chen; Qiaolong Cai; Chen Li; Yongduo Sui
>
> **摘要:** The reasoning capabilities of Large Language Models (LLMs) are typically developed through the single-turn reinforcement learning, whereas real-world applications often involve multi-turn interactions with human feedback, leading to a potential mismatch between training and deployment conditions. In this work, we study whether multi-turn training with human feedback is necessary for reasoning tasks. We compare conventional single-turn training with three multi-turn strategies and reach contrary conclusions to previous research. We find that models trained in a single-turn setting generalize effectively to both single- and multi-turn evaluations, while models trained with multi-turn strategies exhibit a significant degradation in single-turn reasoning performance. These results suggest that for tasks with complete information, robust single-turn training remains more effective and reliable, as multi-turn training with basic feedback provides limited benefits and can even degrade reasoning capabilities.
>
---
#### [replaced 059] Thought Anchors: Which LLM Reasoning Steps Matter?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19143v4](http://arxiv.org/pdf/2506.19143v4)**

> **作者:** Paul C. Bogdan; Uzay Macar; Neel Nanda; Arthur Conmy
>
> **备注:** Paul C. Bogdan and Uzay Macar contributed equally to this work, and their listed order was determined by coinflip. Neel Nanda and Arthur Conmy contributed equally to this work as senior authors, and their listed order was determined by coinflip
>
> **摘要:** Current frontier large-language models rely on reasoning to achieve state-of-the-art performance. Many existing interpretability are limited in this area, as standard methods have been designed to study single forward passes of a model rather than the multi-token computational steps that unfold during reasoning. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We introduce a black-box method that measures each sentence's counterfactual importance by repeatedly sampling replacement sentences from the model, filtering for semantically different ones, and continuing the chain of thought from that point onwards to quantify the sentence's impact on the distribution of final answers. We discover that certain sentences can have an outsized impact on the trajectory of the reasoning trace and final answer. We term these sentences \textit{thought anchors}. These are generally planning or uncertainty management sentences, and specialized attention heads consistently attend from subsequent sentences to thought anchors. We further show that examining sentence-sentence causal links within a reasoning trace gives insight into a model's behavior. Such information can be used to predict a problem's difficulty and the extent different question domains involve sequential or diffuse reasoning. As a proof-of-concept, we demonstrate that our techniques together provide a practical toolkit for analyzing reasoning models by conducting a detailed case study of how the model solves a difficult math problem, finding that our techniques yield a consistent picture of the reasoning trace's structure. We provide an open-source tool (thought-anchors.com) for visualizing the outputs of our methods on further problems. The convergence across our methods shows the potential of sentence-level analysis for a deeper understanding of reasoning models.
>
---
#### [replaced 060] FaithLM: Towards Faithful Explanations for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.04678v4](http://arxiv.org/pdf/2402.04678v4)**

> **作者:** Yu-Neng Chuang; Guanchu Wang; Chia-Yuan Chang; Ruixiang Tang; Shaochen Zhong; Fan Yang; Mengnan Du; Xuanting Cai; Vladimir Braverman; Xia Hu
>
> **摘要:** Large language models (LLMs) increasingly produce natural language explanations, yet these explanations often lack faithfulness, and they do not reliably reflect the evidence the model uses to decide. We introduce FaithLM, a model-agnostic framework that evaluates and improves the faithfulness of LLM explanations without token masking or task-specific heuristics. FaithLM formalizes explanation faithfulness as an intervention property: a faithful explanation should yield a prediction shift when its content is contradicted. Theoretical analysis shows that the resulting contrary-hint score is a sound and discriminative estimator of faithfulness. Building on this principle, FaithLM iteratively refines both the elicitation prompt and the explanation to maximize the measured score. Experiments on three multi-domain datasets and multiple LLM backbones demonstrate that FaithLM consistently increases faithfulness and produces explanations more aligned with human rationales than strong self-explanation baselines. These findings highlight that intervention-based evaluation, coupled with iterative optimization, provides a principled route toward faithful and reliable LLM explanations.
>
---
#### [replaced 061] LyapLock: Bounded Knowledge Preservation in Sequential Large Language Model Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15702v2](http://arxiv.org/pdf/2505.15702v2)**

> **作者:** Peng Wang; Biyu Zhou; Xuehai Tang; Jizhong Han; Songlin Hu
>
> **备注:** EMNLP 2025 main
>
> **摘要:** Large Language Models often contain factually incorrect or outdated knowledge, giving rise to model editing methods for precise knowledge updates. However, current mainstream locate-then-edit approaches exhibit a progressive performance decline during sequential editing, due to inadequate mechanisms for long-term knowledge preservation. To tackle this, we model the sequential editing as a constrained stochastic programming. Given the challenges posed by the cumulative preservation error constraint and the gradually revealed editing tasks, \textbf{LyapLock} is proposed. It integrates queuing theory and Lyapunov optimization to decompose the long-term constrained programming into tractable stepwise subproblems for efficient solving. This is the first model editing framework with rigorous theoretical guarantees, achieving asymptotic optimal editing performance while meeting the constraints of long-term knowledge preservation. Experimental results show that our framework scales sequential editing capacity to over 10,000 edits while stabilizing general capabilities and boosting average editing efficacy by 11.89\% over SOTA baselines. Furthermore, it can be leveraged to enhance the performance of baseline methods. Our code is released on https://github.com/caskcsg/LyapLock.
>
---
#### [replaced 062] A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving
- **分类: cs.PF; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08343v2](http://arxiv.org/pdf/2508.08343v2)**

> **作者:** Ferran Agullo; Joan Oliveras; Chen Wang; Alberto Gutierrez-Torre; Olivier Tardieu; Alaa Youssef; Jordi Torres; Josep Ll. Berral
>
> **备注:** Accepted in a computer science workshop
>
> **摘要:** With the rapid adoption of Large Language Models (LLMs), LLM-adapters have become increasingly common, providing lightweight specialization of large-scale models. Serving hundreds or thousands of these adapters on a single GPU allows request aggregation, increasing throughput, but may also cause request starvation if GPU memory limits are exceeded. To address this issue, this study focuses on determining the joint configuration of concurrent and parallel adapters that maximizes GPU throughput without inducing starvation, given heterogeneous adapter and traffic properties. We propose a data-driven ML approach leveraging interpretable models to tackle this caching problem and introduce the first Digital Twin capable of reproducing an LLM-adapter serving system, enabling efficient training data generation. Experiments with the vLLM framework and LoRA adapters show that the Digital Twin reproduces throughput within 5.1% of real results, while the ML approach predicts optimal numbers of concurrent and parallel adapters with an error of at most 7.2% under heterogeneous, real-world workloads.
>
---
#### [replaced 063] OpenS2S: Advancing Fully Open-Source End-to-End Empathetic Large Speech Language Model
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05177v3](http://arxiv.org/pdf/2507.05177v3)**

> **作者:** Chen Wang; Tianyu Peng; Wen Yang; Yinan Bai; Guangfu Wang; Jun Lin; Lanpeng Jia; Lingxiang Wu; Jinqiao Wang; Chengqing Zong; Jiajun Zhang
>
> **备注:** Technical Report, Update on OpenS2S_v1.5
>
> **摘要:** Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at https://casia-lm.github.io/OpenS2S
>
---
#### [replaced 064] Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09501v2](http://arxiv.org/pdf/2506.09501v2)**

> **作者:** Jiayi Yuan; Hao Li; Xinheng Ding; Wenya Xie; Yu-Jhe Li; Wentian Zhao; Kun Wan; Jing Shi; Xia Hu; Zirui Liu
>
> **摘要:** Large Language Models (LLMs) are now integral across various domains and have demonstrated impressive performance. Progress, however, rests on the premise that benchmark scores are both accurate and reproducible. We demonstrate that the reproducibility of LLM performance is fragile: changing system configuration, such as evaluation batch size, GPU count, and GPU version, can introduce significant differences in the generated responses. This issue is especially pronounced in reasoning models, where minor rounding differences in early tokens can cascade into divergent chains of thought, ultimately affecting accuracy. For instance, under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size. We trace the root cause of this variability to the non-associative nature of floating-point arithmetic under limited numerical precision. This work presents the first systematic investigation into how numerical precision affects reproducibility in LLM inference. Through carefully controlled experiments across various hardware, software, and precision settings, we quantify when and how model outputs diverge. Our analysis reveals that floating-point precision - while critical for reproducibility - is often neglected in evaluation practices. Inspired by this, we develop a lightweight inference pipeline, dubbed LayerCast, that stores weights in 16-bit precision but performs all computations in FP32, balancing memory efficiency with numerical stability. Code is available at https://github.com/nanomaoli/llm_reproducibility.
>
---
#### [replaced 065] SEAL: Steerable Reasoning Calibration of Large Language Models for Free
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.07986v3](http://arxiv.org/pdf/2504.07986v3)**

> **作者:** Runjin Chen; Zhenyu Zhang; Junyuan Hong; Souvik Kundu; Zhangyang Wang
>
> **摘要:** Large Language Models (LLMs), such as OpenAI's o1-series have demonstrated compelling capabilities for complex reasoning tasks via the extended chain-of-thought (CoT) reasoning mechanism. However, recent studies reveal substantial redundancy in the CoT reasoning traces, which not only increases inference latency but also negatively impacts model performance by diverting attention to unnecessary reasoning paths. To address this issue, we investigate the internal reasoning structures of LLMs and categorize them into three primary thought types: execution, reflection, and transition thoughts. Moreover, our analysis reveals that excessive reflection and transition thoughts are strongly correlated with failure cases and these thought categories exhibit clear separation in the latent space. Based on these, we introduce SEAL (Steerable reasoning calibration), a training-free approach that seamlessly calibrates the CoT process, improving accuracy while demonstrating significant efficiency gains. SEAL consists of an offline stage for extracting the reasoning steering vector in the latent space, followed by an on-the-fly calibration of the reasoning trace through representation intervention using the steering vector. Notably, the steering vector exhibits strong transferability across various tasks. Extensive experiments across multiple models (DeepSeek-R1-Distill and QwQ-32B-Preview) and benchmarks (Math500, GSM8K, LiveCodeBench) validate the effectiveness of SEAL, up to a 11% improvement in accuracy while reducing reasoning tokens by 11.8% to 50.4%. Our code is publicly available at https://github.com/VITA-Group/SEAL.
>
---
#### [replaced 066] Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2412.15238v2](http://arxiv.org/pdf/2412.15238v2)**

> **作者:** Gregory Kang Ruey Lau; Wenyang Hu; Diwen Liu; Jizhuo Chen; See-Kiong Ng; Bryan Kian Hsiang Low
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Large Language Models (LLMs), particularly smaller variants, still struggle with complex reasoning tasks. While inference-time prompting can guide reasoning, existing methods often rely on sequential queries. Ensemble approaches offer a promising path to performance gains, especially given recent batch inference speed-ups. This work introduces DIPPER, a novel, training-free framework that transforms a single LLM into an effective inference-time ensemble. By feeding the model an optimized and diverse set of prompts in parallel, DIPPER elicits varied reasoning paths, leading to performance gains. We empirically demonstrate significant improvements on reasoning benchmarks, such as MATH, where a DIPPER ensemble of three Qwen2-MATH-1.5B instances (via parallel prompting of a single model) outperforms a larger 7B model.
>
---
#### [replaced 067] Entity-Augmented Neuroscience Knowledge Retrieval Using Ontology and Semantic Understanding Capability of LLM
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.03145v2](http://arxiv.org/pdf/2506.03145v2)**

> **作者:** Pralaypati Ta; Sriram Venkatesaperumal; Keerthi Ram; Mohanasankar Sivaprakasam
>
> **摘要:** Neuroscience research publications encompass a vast wealth of knowledge. Accurately retrieving existing information and discovering new insights from this extensive literature is essential for advancing the field. However, when knowledge is dispersed across multiple sources, current state-of-the-art retrieval methods often struggle to extract the necessary information. A knowledge graph (KG) can integrate and link knowledge from multiple sources. However, existing methods for constructing KGs in neuroscience often rely on labeled data and require domain expertise. Acquiring large-scale, labeled data for a specialized area like neuroscience presents significant challenges. This work proposes novel methods for constructing KG from unlabeled large-scale neuroscience research corpus utilizing large language models (LLM), neuroscience ontology, and text embeddings. We analyze the semantic relevance of neuroscience text segments identified by LLM for building the knowledge graph. We also introduce an entity-augmented information retrieval algorithm to extract knowledge from the KG. Several experiments were conducted to evaluate the proposed approaches. The results demonstrate that our methods significantly enhance knowledge discovery from the unlabeled neuroscience research corpus. The performance of the proposed entity and relation extraction method is comparable to the existing supervised method. It achieves an F1 score of 0.84 for entity extraction from the unlabeled data. The knowledge obtained from the KG improves answers to over 52% of neuroscience questions from the PubMedQA dataset and questions generated using selected neuroscience entities.
>
---
#### [replaced 068] Automated HIV Screening on Dutch Electronic Health Records with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19879v2](http://arxiv.org/pdf/2510.19879v2)**

> **作者:** Lang Zhou; Amrish Jhingoer; Yinghao Luo; Klaske Vliegenthart--Jongbloed; Carlijn Jordans; Ben Werkhoven; Tom Seinen; Erik van Mulligen; Casper Rokx; Yunlei Li
>
> **备注:** 28 pages, 6 figures
>
> **摘要:** Efficient screening and early diagnosis of HIV are critical for reducing onward transmission. Although large scale laboratory testing is not feasible, the widespread adoption of Electronic Health Records (EHRs) offers new opportunities to address this challenge. Existing research primarily focuses on applying machine learning methods to structured data, such as patient demographics, for improving HIV diagnosis. However, these approaches often overlook unstructured text data such as clinical notes, which potentially contain valuable information relevant to HIV risk. In this study, we propose a novel pipeline that leverages a Large Language Model (LLM) to analyze unstructured EHR text and determine a patient's eligibility for further HIV testing. Experimental results on clinical data from Erasmus University Medical Center Rotterdam demonstrate that our pipeline achieved high accuracy while maintaining a low false negative rate.
>
---
#### [replaced 069] BUSTED at AraGenEval Shared Task: A Comparative Study of Transformer-Based Models for Arabic AI-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.20610v2](http://arxiv.org/pdf/2510.20610v2)**

> **作者:** Ali Zain; Sareem Farooqui; Muhammad Rafi
>
> **摘要:** This paper details our submission to the AraGenEval Shared Task on Arabic AI-generated text detection, where our team, BUSTED, secured 5th place. We investigated the effectiveness of three pre-trained transformer models: AraELECTRA, CAMeLBERT, and XLM-RoBERTa. Our approach involved fine-tuning each model on the provided dataset for a binary classification task. Our findings revealed a surprising result: the multilingual XLM-RoBERTa model achieved the highest performance with an F1 score of 0.7701, outperforming the specialized Arabic models. This work underscores the complexities of AI-generated text detection and highlights the strong generalization capabilities of multilingual models.
>
---
#### [replaced 070] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15510v2](http://arxiv.org/pdf/2505.15510v2)**

> **作者:** Zihui Cheng; Qiguang Chen; Xiao Xu; Jiaqi Wang; Weiyun Wang; Hao Fei; Yidong Wang; Alex Jinpeng Wang; Zhi Chen; Wanxiang Che; Libo Qin
>
> **备注:** Accepted at NeurIPS 2025;
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved significant success in multimodal tasks, with multimodal chain-of-thought (MCoT) further enhancing performance and interpretability. Recent MCoT methods fall into two categories: (i) Textual-MCoT (T-MCoT), which takes multimodal input and produces textual output; and (ii) Interleaved-MCoT (I-MCoT), which generates interleaved image-text outputs. Despite advances in both approaches, the mechanisms driving these improvements are not fully understood. To fill this gap, we first reveal that MCoT boosts LVLMs by incorporating visual thoughts, which convey image information to the reasoning process regardless of the MCoT format, depending only on clarity and conciseness of expression. Furthermore, to explore visual thoughts systematically, we define four distinct forms of visual thought expressions and analyze them comprehensively. Our findings demonstrate that these forms differ in clarity and conciseness, yielding varying levels of MCoT improvement. Additionally, we explore the internal nature of visual thoughts, finding that visual thoughts serve as intermediaries between the input image and reasoning to deeper transformer layers, enabling more advanced visual information transmission. We hope that the visual thoughts can inspire further breakthroughs for future MCoT research.
>
---
#### [replaced 071] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search
- **分类: cs.CL; cs.AI; cs.CE; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.19209v2](http://arxiv.org/pdf/2505.19209v2)**

> **作者:** Zonglin Yang; Wanhao Liu; Ben Gao; Yujie Liu; Wei Li; Tong Xie; Lidong Bing; Wanli Ouyang; Erik Cambria; Dongzhan Zhou
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Large language models (LLMs) have shown promise in automating scientific hypothesis generation, yet existing approaches primarily yield coarse-grained hypotheses lacking critical methodological and experimental details. We introduce and formally define the new task of fine-grained scientific hypothesis discovery, which entails generating detailed, experimentally actionable hypotheses from coarse initial research directions. We frame this as a combinatorial optimization problem and investigate the upper limits of LLMs' capacity to solve it when maximally leveraged. Specifically, we explore four foundational questions: (1) how to best harness an LLM's internal heuristics to formulate the fine-grained hypothesis it itself would judge as the most promising among all the possible hypotheses it might generate, based on its own internal scoring-thus defining a latent reward landscape over the hypothesis space; (2) whether such LLM-judged better hypotheses exhibit stronger alignment with ground-truth hypotheses; (3) whether shaping the reward landscape using an ensemble of diverse LLMs of similar capacity yields better outcomes than defining it with repeated instances of the strongest LLM among them; and (4) whether an ensemble of identical LLMs provides a more reliable reward landscape than a single LLM. To address these questions, we propose a hierarchical search method that incrementally proposes and integrates details into the hypothesis, progressing from general concepts to specific experimental configurations. We show that this hierarchical process smooths the reward landscape and enables more effective optimization. Empirical evaluations on a new benchmark of expert-annotated fine-grained hypotheses from recent literature show that our method consistently outperforms strong baselines.
>
---
#### [replaced 072] Beyond QA Pairs: Assessing Parameter-Efficient Fine-Tuning for Fact Embedding in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.01131v2](http://arxiv.org/pdf/2503.01131v2)**

> **作者:** Shivam Ratnakar; Abhiroop Talasila; Raghav Chamadiya; Nikhil Agarwal; Vinayak K Doifode
>
> **备注:** Presented at the Workshop on Preparing Good Data for Generative AI: Challenges and Approaches (Good-Data) in conjunction with AAAI 2025. The authors retain the copyright
>
> **摘要:** This paper presents an extensive examination of Parameter-Efficient Fine-Tuning (PEFT) for embedding domain specific facts into Large Language Models (LLMs), focusing on improving the fine-tuning process by categorizing question-answer (QA) pairs into Factual and Conceptual classes using a BERT-based classifier. Two distinct Llama-2 models are fine-tuned based on these classifications and evaluated using larger models like GPT-3.5 Turbo and Gemini. Our results indicate that models trained on conceptual datasets outperform those trained on factual datasets. Additionally, we compare the efficiency of two synthetic fine-tuning dataset generation techniques, D-RAG and D-Naive, with D-Naive demonstrating superior performance. Although PEFT has shown effectiveness, our research indicates that it may not be the most optimal method for embedding facts into LLMs. However, it has demonstrated exceptional performance in instruction-based tasks. Our findings are reinforced by a 1000-sample dataset in the data center domain, where the fine-tuned Llama-2 7B model significantly outperforms the baseline model in generating product recommendations. Our study highlights the importance of QA pair categorization and synthetic dataset generation techniques in enhancing the performance of LLMs in specific domains.
>
---
#### [replaced 073] ARF-RLHF: Adaptive Reward-Following for RLHF through Emotion-Driven Self-Supervision and Trace-Biased Dynamic Optimization
- **分类: cs.CL; cs.AI; 68T05, 68Q25; I.2.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.03069v3](http://arxiv.org/pdf/2507.03069v3)**

> **作者:** YuXuan Zhang
>
> **备注:** This version fixes some minor typographical errors and adds more explanations to ensure clarity in presentation
>
> **摘要:** Current RLHF methods such as PPO and DPO typically reduce human preferences to binary labels, which are costly to obtain and too coarse to reflect individual variation. We observe that expressions of satisfaction and dissatisfaction follow stable linguistic patterns across users, indicating that more informative supervisory signals can be extracted from free-form feedback. Building on this insight, we introduce Adaptive Reward-Following (ARF), which converts natural feedback into continuous preference trajectories and optimizes them using the novel TraceBias algorithm. Across diverse LLMs and preference domains, ARF consistently outperforms PPO and DPO, improving alignment by up to 7.6%. Our results demonstrate that continuous reward modeling provides a scalable path toward personalized and theoretically grounded RLHF.
>
---
#### [replaced 074] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17239v2](http://arxiv.org/pdf/2503.17239v2)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Farhan Ahmed; Syed Zawad; Holger Boche
>
> **摘要:** Fine-tuning large language models (LLMs) is a common practice to adapt generalist models to specialized domains. However, recent studies show that fine-tuning can erode safety alignment, causing LLMs to respond to harmful or unethical prompts. Many methods to realign safety have been proposed, but often introduce custom algorithms that are difficult to implement or compromise task utility. In this work, we propose SafeMERGE, a lightweight, post-fine-tuning framework that preserves safety while maintaining downstream performance. SafeMERGE selectively merges fine-tuned with safety-aligned model layers only when they deviate from safe behavior, measured by a cosine similarity criterion. Across three LLMs and two tasks, SafeMERGE consistently reduces harmful outputs compared to other defenses, with negligible or even positive impact on utility. Our results demonstrate that selective layer-wise merging offers an effective safeguard against the inadvertent loss of safety during fine-tuning, establishing SafeMERGE as a simple post-fine-tuning defense.
>
---
#### [replaced 075] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions
- **分类: cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.14668v2](http://arxiv.org/pdf/2505.14668v2)**

> **作者:** Bufang Yang; Lilin Xu; Liekang Zeng; Kaiwei Liu; Siyang Jiang; Wenrui Lu; Hongkai Chen; Xiaofan Jiang; Guoliang Xing; Zhenyu Yan
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts surrounding humans to enhance the proactivity of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and personas from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants. The code and dataset are publicly available at https://github.com/openaiotlab/ContextAgent.
>
---
#### [replaced 076] Enhancing Naturalness in LLM-Generated Utterances through Disfluency Insertion
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12710v2](http://arxiv.org/pdf/2412.12710v2)**

> **作者:** Syed Zohaib Hassan; Pierre Lison; Pål Halvorsen
>
> **备注:** 8 pages. Limitations, ethical considerations, and references are additional
>
> **摘要:** Disfluencies are a natural feature of spontaneous human speech but are typically absent from the outputs of Large Language Models (LLMs). This absence can diminish the perceived naturalness of synthesized speech, which is an important criteria when building conversational agents that aim to mimick human behaviours. We show how the insertion of disfluencies can alleviate this shortcoming. The proposed approach involves (1) fine-tuning an LLM with Low-Rank Adaptation (LoRA) to incorporate various types of disfluencies into LLM-generated utterances and (2) synthesizing those utterances using a text-to-speech model that supports the generation of speech phenomena such as disfluencies. We evaluated the quality of the generated speech across two metrics: intelligibility and perceived spontaneity. We demonstrate through a user study that the insertion of disfluencies significantly increase the perceived spontaneity of the generated speech. This increase came, however, along with a slight reduction in intelligibility.
>
---
#### [replaced 077] DeepOmni: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21864v3](http://arxiv.org/pdf/2506.21864v3)**

> **作者:** Hang Shao; Heting Gao; Yunhang Shen; Jiawei Chen; Zuwei Long; Dong Yang; Ke Li; Xing Sun
>
> **备注:** Under Review
>
> **摘要:** Native multimodal large language models (MLLMs) restructure a single large language model (LLM) into a spoken language model (SLM) capable of both speech and text generation. Compared to modular and aligned MLLMs, native MLLMs preserve richer paralinguistic features such as emotion and prosody, and generate speech responses directly within the backbone LLM rather than using a separate speech decoder. This integration also results in lower response latency and smoother interaction. However, native MLLMs suffer from catastrophic forgetting and performance degradation because the available paired speech-text data is insufficient to support the pretraining of MLLMs compared to the vast amount of text data required to pretrain text LLMs. To address this issue, we propose DeepTalk, a framework for adaptive modality expert learning based on a Mixture of Experts (MoE) architecture. DeepTalk first adaptively distinguishes modality experts according to their modality load within the LLM. Each modality expert then undergoes specialized single-modality training, followed by joint multimodal collaborative training. As a result, DeepTalk incurs only a 5.5% performance drop compared to the original LLM, which is significantly lower than the average performance drop of over 20% typically seen in native MLLMs (such as GLM-4-Voice), and is on par with modular MLLMs. Meanwhile, the end-to-end dialogue latency remains within 0.5 seconds, ensuring a seamless and intelligent speech interaction experience. Code and models are released at https://github.com/talkking/DeepTalk.
>
---
#### [replaced 078] Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13586v3](http://arxiv.org/pdf/2510.13586v3)**

> **作者:** Pasin Buakhaw; Kun Kerdthaisong; Phuree Phenhiran; Pitikorn Khlaisamniang; Supasate Vorathammathorn; Piyalitt Ittichaiwong; Nutchanon Yongsatianchot
>
> **摘要:** The emergence of large language models (LLMs) has opened new opportunities for creating dynamic non-player characters (NPCs) in gaming environments, enabling both functional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which evaluates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
>
---
#### [replaced 079] A Simple Linear Patch Revives Layer-Pruned Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24680v2](http://arxiv.org/pdf/2505.24680v2)**

> **作者:** Xinrui Chen; Haoli Bai; Tao Yuan; Ruikang Liu; Kang Zhao; Xianzhi Yu; Lu Hou; Tian Guan; Yonghong He; Chun Yuan
>
> **备注:** 26 pages, accepted to NeurIPS 2025
>
> **摘要:** Layer pruning has emerged as a widely used technique for compressing large language models (LLMs). However, existing layer pruning approaches often incur substantial performance degradation. We identify the majority of this degradation to a single yet previously overlooked issue: \textit{the mismatch of activation magnitudes at the pruning interface}. The pre-interface activations exhibit significantly different scales from the post-interface ones, causing the distributional shift as it propagates through the remaining layers. To address this issue, we introduce \textsc{LinearPatch}, a lightweight and plug-and-play technique that fuses two operations into one matrix multiply at the pruning interface: (i) a Hadamard transformation that suppresses massive outliers at particular tokens and (ii) a channel-wise scaling that aligns activation statistics. On LLaMA-3-8B, \textsc{LinearPatch} preserves up to \textbf{94.15\%} of the original model's performance when pruning 5 out of 32 layers, outperforming the previous state of the art by \textbf{4\%}. The patch can be further refined with 5K unlabeled samples via memory-efficient offline distillation, pushing the retention to 95.16\% within only 30 minutes on a single GPU. Code is available at https://github.com/chenxinrui-tsinghua/LinearPatch.
>
---
#### [replaced 080] Distinct social-linguistic processing between humans and large audio-language models: Evidence from model-brain alignment
- **分类: cs.CL; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2503.19586v2](http://arxiv.org/pdf/2503.19586v2)**

> **作者:** Hanlin Wu; Xufeng Duan; Zhenguang Cai
>
> **备注:** Hanlin Wu, Xufeng Duan, and Zhenguang Cai. 2025. Distinct social-linguistic processing between humans and large audio-language models: Evidence from model-brain alignment. In Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, pages 135-143, Albuquerque, New Mexico, USA. Association for Computational Linguistics. https://aclanthology.org/2025.cmcl-1.18/
>
> **摘要:** Voice-based AI development faces unique challenges in processing both linguistic and paralinguistic information. This study compares how large audio-language models (LALMs) and humans integrate speaker characteristics during speech comprehension, asking whether LALMs process speaker-contextualized language in ways that parallel human cognitive mechanisms. We compared two LALMs' (Qwen2-Audio and Ultravox 0.5) processing patterns with human EEG responses. Using surprisal and entropy metrics from the models, we analyzed their sensitivity to speaker-content incongruency across social stereotype violations (e.g., a man claiming to regularly get manicures) and biological knowledge violations (e.g., a man claiming to be pregnant). Results revealed that Qwen2-Audio exhibited increased surprisal for speaker-incongruent content and its surprisal values significantly predicted human N400 responses, while Ultravox 0.5 showed limited sensitivity to speaker characteristics. Importantly, neither model replicated the human-like processing distinction between social violations (eliciting N400 effects) and biological violations (eliciting P600 effects). These findings reveal both the potential and limitations of current LALMs in processing speaker-contextualized language, and suggest differences in social-linguistic processing mechanisms between humans and LALMs.
>
---
#### [replaced 081] ClaimGen-CN: A Large-scale Chinese Dataset for Legal Claim Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.17234v2](http://arxiv.org/pdf/2508.17234v2)**

> **作者:** Siying Zhou; Yiquan Wu; Hui Chen; Xavier Hu; Kun Kuang; Adam Jatowt; Ming Hu; Chunyan Zheng; Fei Wu
>
> **摘要:** Legal claims refer to the plaintiff's demands in a case and are essential to guiding judicial reasoning and case resolution. While many works have focused on improving the efficiency of legal professionals, the research on helping non-professionals (e.g., plaintiffs) remains unexplored. This paper explores the problem of legal claim generation based on the given case's facts. First, we construct ClaimGen-CN, the first dataset for Chinese legal claim generation task, from various real-world legal disputes. Additionally, we design an evaluation metric tailored for assessing the generated claims, which encompasses two essential dimensions: factuality and clarity. Building on this, we conduct a comprehensive zero-shot evaluation of state-of-the-art general and legal-domain large language models. Our findings highlight the limitations of the current models in factual precision and expressive clarity, pointing to the need for more targeted development in this domain. To encourage further exploration of this important task, we will make the dataset publicly available.
>
---
#### [replaced 082] Language Model Guided Reinforcement Learning in Quantitative Trading
- **分类: cs.LG; cs.CL; q-fin.TR; I.2.7; I.2.6; J.4**

- **链接: [http://arxiv.org/pdf/2508.02366v3](http://arxiv.org/pdf/2508.02366v3)**

> **作者:** Adam Darmanin; Vince Vella
>
> **备注:** 12 pages (4 pages appendix and references) and 6 figures. Accepted for presentation at FLLM 2025, Vienna
>
> **摘要:** Algorithmic trading requires short-term tactical decisions consistent with long-term financial objectives. Reinforcement Learning (RL) has been applied to such problems, but adoption is limited by myopic behaviour and opaque policies. Large Language Models (LLMs) offer complementary strategic reasoning and multi-modal signal interpretation when guided by well-structured prompts. This paper proposes a hybrid framework in which LLMs generate high-level trading strategies to guide RL agents. We evaluate (i) the economic rationale of LLM-generated strategies through expert review, and (ii) the performance of LLM-guided agents against unguided RL baselines using Sharpe Ratio (SR) and Maximum Drawdown (MDD). Empirical results indicate that LLM guidance improves both return and risk metrics relative to standard RL.
>
---
#### [replaced 083] Breaking Language Barriers or Reinforcing Bias? A Study of Gender and Racial Disparities in Multilingual Contrastive Vision Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14160v3](http://arxiv.org/pdf/2505.14160v3)**

> **作者:** Zahraa Al Sahili; Ioannis Patras; Matthew Purver
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** Multilingual vision-language models (VLMs) promise universal image-text retrieval, yet their social biases remain underexplored. We perform the first systematic audit of four public multilingual CLIP variants: M-CLIP, NLLB-CLIP, CAPIVARA-CLIP, and the debiased SigLIP-2, covering ten languages that differ in resource availability and morphological gender marking. Using balanced subsets of FairFace and the PATA stereotype suite in a zero-shot setting, we quantify race and gender bias and measure stereotype amplification. Contrary to the intuition that multilinguality mitigates bias, every model exhibits stronger gender skew than its English-only baseline. CAPIVARA-CLIP shows its largest biases precisely in the low-resource languages it targets, while the shared encoder of NLLB-CLIP and SigLIP-2 transfers English gender stereotypes into gender-neutral languages; loosely coupled encoders largely avoid this leakage. Although SigLIP-2 reduces agency and communion skews, it inherits -- and in caption-sparse contexts (e.g., Xhosa) amplifies -- the English anchor's crime associations. Highly gendered languages consistently magnify all bias types, yet gender-neutral languages remain vulnerable whenever cross-lingual weight sharing imports foreign stereotypes. Aggregated metrics thus mask language-specific hot spots, underscoring the need for fine-grained, language-aware bias evaluation in future multilingual VLM research.
>
---
#### [replaced 084] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06607v3](http://arxiv.org/pdf/2507.06607v3)**

> **作者:** Liliang Ren; Congcong Chen; Haoran Xu; Young Jin Kim; Adam Atkinson; Zheng Zhan; Jiankai Sun; Baolin Peng; Liyuan Liu; Shuohang Wang; Hao Cheng; Jianfeng Gao; Weizhu Chen; Yelong Shen
>
> **备注:** Accepted by NeurIPS 2025. Camera-ready Version
>
> **摘要:** Recent advances in language modeling have demonstrated the effectiveness of State Space Models (SSMs) for efficient sequence modeling. While hybrid architectures such as Samba and the decoder-decoder architecture, YOCO, have shown promising performance gains over Transformers, prior works have not investigated the efficiency potential of representation sharing between SSM layers. In this paper, we introduce the Gated Memory Unit (GMU), a simple yet effective mechanism for efficient memory sharing across layers. We apply it to create SambaY, a decoder-hybrid-decoder architecture that incorporates GMUs in the cross-decoder to share memory readout states from a Samba-based self-decoder. SambaY significantly enhances decoding efficiency, preserves linear pre-filling time complexity, and boosts long-context performance, all while eliminating the need for explicit positional encoding. Through extensive scaling experiments, we demonstrate that our model exhibits a significantly lower irreducible loss compared to a strong YOCO baseline, indicating superior performance scalability under large-scale compute regimes. Our largest model enhanced with Differential Attention, Phi4-mini-Flash-Reasoning, achieves significantly better performance than Phi4-mini-Reasoning on reasoning tasks such as Math500, AIME24/25, and GPQA Diamond without any reinforcement learning, while delivering up to 10x higher decoding throughput on 2K-length prompts with 32K generation length under the vLLM inference framework. We release our training codebase on open-source data at https://github.com/microsoft/ArchScale.
>
---
#### [replaced 085] DocFinQA: A Long-Context Financial Reasoning Dataset
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.06915v3](http://arxiv.org/pdf/2401.06915v3)**

> **作者:** Varshini Reddy; Rik Koncel-Kedziorski; Viet Dac Lai; Michael Krumdick; Charles Lovering; Chris Tanner
>
> **备注:** 13 pages
>
> **摘要:** For large language models (LLMs) to be effective in the financial domain -- where each decision can have a significant impact -- it is necessary to investigate realistic tasks and data. Financial professionals often interact with documents that are hundreds of pages long, but most financial research datasets only deal with short excerpts from these documents. To address this, we introduce a long-document financial QA task. We augment 7,437 questions from the existing FinQA dataset with the full-document context, extending the average context length from under 700 words in FinQA to 123k words in DocFinQA. We conduct extensive experiments over retrieval-based QA pipelines and long-context language models. DocFinQA proves a significant challenge for even state-of-the-art systems. We also provide a case-study on the longest documents in DocFinQA and find that models particularly struggle on these documents. Addressing these challenges may have a wide reaching impact across applications where specificity and long-range contexts are critical, like gene sequences and legal document contract analysis.
>
---
#### [replaced 086] SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training
- **分类: cs.LG; cs.CL; math.OC**

- **链接: [http://arxiv.org/pdf/2505.24749v2](http://arxiv.org/pdf/2505.24749v2)**

> **作者:** Yehonathan Refael; Guy Smorodinsky; Tom Tirer; Ofir Lindenbaum
>
> **摘要:** Low-rank gradient-based optimization methods have significantly improved memory efficiency during the training of large language models (LLMs), enabling operations within constrained hardware without sacrificing performance. However, these methods primarily emphasize memory savings, often overlooking potential acceleration in convergence due to their reliance on standard isotropic steepest descent techniques, which can perform suboptimally in the highly anisotropic landscapes typical of deep networks, particularly LLMs. In this paper, we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods like Newton-Schulz orthogonalization approximation. We theoretically establish an upper bound on these approximation errors, proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods.
>
---
#### [replaced 087] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation
- **分类: cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15807v2](http://arxiv.org/pdf/2505.15807v2)**

> **作者:** Patrick Kahardipraja; Reduan Achtibat; Thomas Wiegand; Wojciech Samek; Sebastian Lapuschkin
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Large language models are able to exploit in-context learning to access external knowledge beyond their training data through retrieval-augmentation. While promising, its inner workings remain unclear. In this work, we shed light on the mechanism of in-context retrieval augmentation for question answering by viewing a prompt as a composition of informational components. We propose an attribution-based method to identify specialized attention heads, revealing in-context heads that comprehend instructions and retrieve relevant contextual information, and parametric heads that store entities' relational knowledge. To better understand their roles, we extract function vectors and modify their attention weights to show how they can influence the answer generation process. Finally, we leverage the gained insights to trace the sources of knowledge used during inference, paving the way towards more safe and transparent language models.
>
---
#### [replaced 088] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01347v2](http://arxiv.org/pdf/2506.01347v2)**

> **作者:** Xinyu Zhu; Mengzhou Xia; Zhepei Wei; Wei-Lin Chen; Danqi Chen; Yu Meng
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) is a promising approach for training language models (LMs) on reasoning tasks that elicit emergent long chains of thought (CoTs). Unlike supervised learning, it updates the model using both correct and incorrect samples via policy gradients. To better understand its mechanism, we decompose the learning signal into reinforcing correct responses and penalizing incorrect ones, referred to as Positive and Negative Sample Reinforcement (PSR and NSR), respectively. We train Qwen2.5-Math-7B, Qwen3-4B and Llama-3.1-8B-Instruct on a mathematical reasoning dataset and uncover a surprising result: training with only negative samples -- without reinforcing correct responses -- can be highly effective: it consistently improves performance over the base model across the entire Pass@$k$ spectrum $k$ up to 256), often matching or surpassing PPO and GRPO. In contrast, reinforcing only correct responses improves Pass@1 but degrades performance at higher $k$, due to reduced diversity. These inference-scaling trends highlight that solely penalizing incorrect responses may contribute more to performance than previously recognized. Through gradient analysis, we show that NSR works by suppressing incorrect generations and redistributing probability mass toward other plausible candidates, guided by the model's prior beliefs. It refines the model's existing knowledge rather than introducing entirely new behaviors. Building on this insight, we propose a simple variant of the RL objective that upweights NSR, and show that it consistently improves overall Pass@$k$ performance on MATH, AIME 2025, and AMC23. Our code is available at https://github.com/TianHongZXY/RLVR-Decomposed.
>
---
#### [replaced 089] Bootstrapping Referring Multi-Object Tracking
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.05039v2](http://arxiv.org/pdf/2406.05039v2)**

> **作者:** Yani Zhang; Dongming Wu; Wencheng Han; Xingping Dong
>
> **摘要:** Referring understanding is a fundamental task that bridges natural language and visual content by localizing objects described in free-form expressions. However, existing works are constrained by limited language expressiveness, lacking the capacity to model object dynamics in spatial numbers and temporal states. To address these limitations, we introduce a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a language expression as a semantic cue to guide the prediction of multi-object tracking, comprehensively accounting for variations in object quantity and temporal semantics. Along with RMOT, we introduce a RMOT benchmark named Refer-KITTI-V2, featuring scalable and diverse language expressions. To efficiently generate high-quality annotations covering object dynamics with minimal manual effort, we propose a semi-automatic labeling pipeline that formulates a total of 9,758 language prompts. In addition, we propose TempRMOT, an elegant end-to-end Transformer-based framework for RMOT. At its core is a query-driven Temporal Enhancement Module that represents each object as a Transformer query, enabling long-term spatial-temporal interactions with other objects and past frames to efficiently refine these queries. TempRMOT achieves state-of-the-art performance on both Refer-KITTI and Refer-KITTI-V2, demonstrating the effectiveness of our approach. The source code and dataset is available at https://github.com/zyn213/TempRMOT.
>
---
#### [replaced 090] Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14225v2](http://arxiv.org/pdf/2504.14225v2)**

> **作者:** Bowen Jiang; Zhuoqun Hao; Young-Min Cho; Bryan Li; Yuan Yuan; Sihao Chen; Lyle Ungar; Camillo J. Taylor; Dan Roth
>
> **备注:** The 2025 Conference on Language Modeling (COLM)
>
> **摘要:** Large Language Models (LLMs) have emerged as personalized assistants for users across a wide range of tasks -- from offering writing support to delivering tailored recommendations or consultations. Over time, the interaction history between a user and an LLM can provide extensive information about an individual's traits and preferences. However, open questions remain on how well LLMs today can effectively leverage such history to (1) internalize the user's inherent traits and preferences, (2) track how the user profiling and preferences evolve over time, and (3) generate personalized responses accordingly in new scenarios. In this work, we introduce the PERSONAMEM benchmark. PERSONAMEM features curated user profiles with over 180 simulated user-LLM interaction histories, each containing up to 60 sessions of multi-turn conversations across 15 real-world tasks that require personalization. Given an in-situ user query, i.e. query issued by the user from the first-person perspective, we evaluate LLM chatbots' ability to identify the most suitable response according to the current state of the user's profile. We observe that current LLMs still struggle to recognize the dynamic evolution in users' profiles over time through direct prompting approaches. As a consequence, LLMs often fail to deliver responses that align with users' current situations and preferences, with frontier models such as GPT-4.1, o4-mini, GPT-4.5, o1, or Gemini-2.0 achieving only around 50% overall accuracy, suggesting room for improvement. We hope that PERSONAMEM, along with the user profile and conversation simulation pipeline, can facilitate future research in the development of truly user-aware chatbots. Code and data are available at github.com/bowen-upenn/PersonaMem.
>
---
#### [replaced 091] Gatsby Without the 'E': Crafting Lipograms with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20501v2](http://arxiv.org/pdf/2505.20501v2)**

> **作者:** Rohan Balasubramanian; Nitish Gokulakrishnan; Syeda Jannatus Saba; Steven Skiena
>
> **摘要:** Lipograms are a unique form of constrained writing where all occurrences of a particular letter are excluded from the text, typified by the novel Gadsby, which daringly avoids all usage of the letter 'e'. In this study, we explore the power of modern large language models (LLMs) by transforming the novel F. Scott Fitzgerald's The Great Gatsby into a fully 'e'-less text. We experimented with a range of techniques, from baseline methods like synonym replacement to sophisticated generative models enhanced with beam search and named entity analysis. We show that excluding up to 3.6% of the most common letters (up to the letter 'u') had minimal impact on the text's meaning, although translation fidelity rapidly and predictably decays with stronger lipogram constraints. Our work highlights the surprising flexibility of English under strict constraints, revealing just how adaptable and creative language can be.
>
---
#### [replaced 092] MEXA: Towards General Multimodal Reasoning with Dynamic Multi-Expert Aggregation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17113v2](http://arxiv.org/pdf/2506.17113v2)**

> **作者:** Shoubin Yu; Yue Zhang; Ziyang Wang; Jaehong Yoon; Mohit Bansal
>
> **备注:** EMNLP 2025 Findings; The first two authors contributed equally; Github link: https://github.com/Yui010206/MEXA
>
> **摘要:** Combining pre-trained expert models offers substantial potential for scalable multimodal reasoning, but building a unified framework remains challenging due to the increasing diversity of input modalities and task complexity. For instance, medical diagnosis requires precise reasoning over structured clinical tables, while financial forecasting depends on interpreting plot-based data to make informed predictions. To tackle this challenge, we introduce MEXA, a training-free framework that performs modality- and task-aware aggregation of multiple expert models to enable effective multimodal reasoning across diverse and distinct domains. MEXA dynamically selects expert models based on the input modality and the task-specific reasoning demands (i.e., skills). Each expert model, specialized in a modality task pair, generates interpretable textual reasoning outputs. MEXA then aggregates and reasons over these outputs using a Large Reasoning Model (LRM) to produce the final answer. This modular design allows flexible and transparent multimodal reasoning across diverse domains without additional training overhead. We extensively evaluate our approach on diverse multimodal benchmarks, including Video Reasoning, Audio Reasoning, 3D Understanding, and Medical QA. MEXA consistently delivers performance improvements over strong multimodal baselines, highlighting the effectiveness and broad applicability of our expert-driven selection and aggregation in diverse multimodal reasoning tasks.
>
---
#### [replaced 093] QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22674v2](http://arxiv.org/pdf/2503.22674v2)**

> **作者:** Belinda Z. Li; Been Kim; Zi Wang
>
> **备注:** Code and dataset are available at \url{https://github.com/google-deepmind/questbench}
>
> **摘要:** Large language models (LLMs) have shown impressive performance on reasoning benchmarks like math and logic. While many works have largely assumed well-defined tasks, real-world queries are often underspecified and only solvable by acquiring missing information. We formalize this information-gathering problem as a constraint satisfaction problem (CSP) with missing variable assignments. Using a special case where only one necessary variable assignment is missing, we can evaluate an LLM's ability to identify the minimal necessary question to ask. We present QuestBench, a set of underspecified reasoning tasks solvable by asking at most one question, which includes: (1) Logic-Q: logical reasoning tasks with one missing proposition, (2) Planning-Q: PDDL planning problems with partially-observed initial states, (3) GSM-Q: human-annotated grade school math problems with one unknown variable, and (4) GSME-Q: equation-based version of GSM-Q. The LLM must select the correct clarification question from multiple options. While current models excel at GSM-Q and GSME-Q, they achieve only 40-50% accuracy on Logic-Q and Planning-Q. Analysis shows that the ability to solve well-specified reasoning problems is not sufficient for success on our benchmark: models struggle to identify the right question even when they can solve the fully specified version. This highlights the need for specifically optimizing models' information acquisition capabilities.
>
---
#### [replaced 094] The Gray Zone of Faithfulness: Taming Ambiguity in Unfaithfulness Detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.21118v2](http://arxiv.org/pdf/2510.21118v2)**

> **作者:** Qiang Ding; Lvzhou Luo; Yixuan Cao; Ping Luo
>
> **备注:** Updates: 1. further polishing the writing; 2. adding the motivation of investigating selective prediction for unfaithfulness detectors
>
> **摘要:** Ensuring that Large Language Models (LLMs) generate summaries faithful to a given source document is essential for real-world applications. While prior research has explored LLM faithfulness, existing benchmarks suffer from annotation ambiguity, primarily due to the ill-defined boundary of permissible external knowledge in generated outputs. For instance, common sense is often incorporated into responses and labeled as "faithful", yet the acceptable extent of such knowledge remains unspecified, leading to inconsistent annotations. To address this issue, we propose a novel faithfulness annotation framework, which introduces an intermediate category, Out-Dependent, to classify cases where external knowledge is required for verification. Using this framework, we construct VeriGray (Verification with the Gray Zone) -- a new unfaithfulness detection benchmark in summarization. Statistics reveal that even SOTA LLMs, such as GPT-5, exhibit hallucinations ($\sim 6\%$ of sentences) in summarization tasks. Moreover, a substantial proportion ($\sim 8\%$ on average of models) of generated sentences fall into the Out-Dependent category, underscoring the importance of resolving annotation ambiguity in unfaithfulness detection benchmarks. Experiments demonstrate that our benchmark poses significant challenges to multiple baseline methods, indicating considerable room for future improvement.
>
---
#### [replaced 095] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.14350v3](http://arxiv.org/pdf/2503.14350v3)**

> **作者:** Shoubin Yu; Difan Liu; Ziqiao Ma; Yicong Hong; Yang Zhou; Hao Tan; Joyce Chai; Mohit Bansal
>
> **备注:** ICCV 2025; First three authors contributed equally. Project page: https://veggie-gen.github.io/
>
> **摘要:** Recent video diffusion models have enhanced video editing, but it remains challenging to handle instructional editing and diverse tasks (e.g., adding, removing, changing) within a unified framework. In this paper, we introduce VEGGIE, a Video Editor with Grounded Generation from Instructions, a simple end-to-end framework that unifies video concept editing, grounding, and reasoning based on diverse user instructions. Specifically, given a video and text query, VEGGIE first utilizes an MLLM to interpret user intentions in instructions and ground them to the video contexts, generating frame-specific grounded task queries for pixel-space responses. A diffusion model then renders these plans and generates edited videos that align with user intent. To support diverse tasks and complex instructions, we employ a curriculum learning strategy: first aligning the MLLM and video diffusion model with large-scale instructional image editing data, followed by end-to-end fine-tuning on high-quality multitask video data. Additionally, we introduce a novel data synthesis pipeline to generate paired instructional video editing data for model training. It transforms static image data into diverse, high-quality video editing samples by leveraging Image-to-Video models to inject dynamics. VEGGIE shows strong performance in instructional video editing with different editing skills, outperforming the best instructional baseline as a versatile model, while other models struggle with multi-tasking. VEGGIE also excels in video object grounding and reasoning segmentation, where other baselines fail. We further reveal how the multiple tasks help each other and highlight promising applications like zero-shot multimodal instructional and in-context video editing.
>
---
#### [replaced 096] Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.16492v2](http://arxiv.org/pdf/2510.16492v2)**

> **作者:** Vamshi Krishna Bonagiri; Ponnurangam Kumaragurum; Khanh Nguyen; Benjamin Plaut
>
> **备注:** Reliable ML and Regulatable ML workshops, Neurips 2025
>
> **摘要:** As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.
>
---
#### [replaced 097] Integrated Design and Governance of Agentic AI Systems through Adaptive Information Modulation
- **分类: cs.AI; cs.CL; cs.CY; cs.GT**

- **链接: [http://arxiv.org/pdf/2409.10372v4](http://arxiv.org/pdf/2409.10372v4)**

> **作者:** Qiliang Chen; Sepehr Ilami; Nunzio Lore; Babak Heydari
>
> **摘要:** Modern engineered systems increasingly involve complex sociotechnical environments where multiple agents, including humans and the emerging paradigm of agentic AI powered by large language models, must navigate social dilemmas that pit individual interests against collective welfare. As engineered systems evolve toward multi-agent architectures with autonomous LLM-based agents, traditional governance approaches using static rules or fixed network structures fail to address the dynamic uncertainties inherent in real-world operations. This paper presents a novel framework that integrates adaptive governance mechanisms directly into the design of sociotechnical systems through a unique separation of agent interaction networks from information flow networks. We introduce a system comprising strategic LLM-based system agents that engage in repeated interactions and a reinforcement learning-based governing agent that dynamically modulates information transparency. Unlike conventional approaches that require direct structural interventions or payoff modifications, our framework preserves agent autonomy while promoting cooperation through adaptive information governance. The governing agent learns to strategically adjust information disclosure at each timestep, determining what contextual or historical information each system agent can access. Experimental results demonstrate that this RL-based governance significantly enhances cooperation compared to static information-sharing baselines.
>
---
#### [replaced 098] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.15390v2](http://arxiv.org/pdf/2508.15390v2)**

> **作者:** Woojin Chung; Jeonghoon Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models are trained with tokenizers, and the resulting token distribution is highly imbalanced: a few words dominate the stream while most occur rarely. Recent practice favors ever-larger vocabularies, but it is unclear where the benefit comes from. To this end, we perform a controlled study that scales the vocabulary of the language model from 24K to 196K while holding data, computation, and optimization unchanged. We begin by quantifying the complexity of tokenized text -- formalized via Kolmogorov complexity -- and show that larger vocabularies reduce this complexity. Above 24K, every common word is already tokenized as a single token, so enlarging vocabulary only deepens the relative token-frequency imbalance. Word-level loss decomposition shows that larger vocabularies reduce cross-entropy loss almost exclusively by lowering uncertainty on the 2,500 most frequent words, even though loss on the rare tail rises. The same frequent words cover roughly 75% of tokens in downstream benchmarks, so this training advantage transfers intact. We further show that enlarging model parameters with a fixed vocabulary yields the same frequent-word benefit. Our results recast "bigger vocabularies help" as "lowering complexity of tokenized text helps," offering a simple, principled knob for tokenizer--model co-design and clarifying the loss dynamics that govern language model scaling in pre-training.
>
---
#### [replaced 099] SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.17516v3](http://arxiv.org/pdf/2510.17516v3)**

> **作者:** Tiancheng Hu; Joachim Baumann; Lorenzo Lupo; Nigel Collier; Dirk Hovy; Paul Röttger
>
> **备注:** Project Website: http://simbench.tiancheng.hu/ Data: https://huggingface.co/datasets/pitehu/SimBench
>
> **摘要:** Large language model (LLM) simulations of human behavior have the potential to revolutionize the social and behavioral sciences, if and only if they faithfully reflect real human behaviors. Current evaluations are fragmented, based on bespoke tasks and metrics, creating a patchwork of incomparable results. To address this, we introduce SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation. By unifying 20 diverse datasets covering tasks from moral decision-making to economic choice across a large global participant pool, SimBench provides the necessary foundation to ask fundamental questions about when, how, and why LLM simulations succeed or fail. We show that, while even the best LLMs today have limited simulation ability (score: 40.80/100), performance scales log-linearly with model size. Simulation performance is not improved by increased inference-time compute. We demonstrate an alignment-simulation trade-off: instruction-tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones. Models particularly struggle when simulating specific demographic groups. Finally, we demonstrate that simulation ability correlates most strongly with deep, knowledge-intensive reasoning (MMLU-Pro, r=0.939). By making progress measurable, we aim to accelerate the development of more faithful LLM simulators.
>
---
#### [replaced 100] Twilight: Adaptive Attention Sparsity with Hierarchical Top-$p$ Pruning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02770v4](http://arxiv.org/pdf/2502.02770v4)**

> **作者:** Chaofan Lin; Jiaming Tang; Shuo Yang; Hanshuo Wang; Tian Tang; Boyu Tian; Ion Stoica; Mingyu Gao
>
> **备注:** To appear on NeurIPS 2025 (spotlight)
>
> **摘要:** Leveraging attention sparsity to accelerate long-context large language models (LLMs) has been a hot research topic. However, current algorithms such as sparse attention or key-value (KV) cache compression tend to use a fixed budget, which presents a significant challenge during deployment because it fails to account for the dynamic nature of real-world scenarios, where the optimal balance between accuracy and efficiency can vary greatly. In this paper, we find that borrowing top-$p$ sampling (nucleus sampling) to sparse attention can surprisingly achieve adaptive budgeting. Based on this, we propose Twilight, a framework to bring adaptive sparsity to any existing sparse attention algorithm without sacrificing their accuracy. Empirical results show that Twilight can adaptively prune at most 98% of redundant tokens, leading to $15.4\times$ acceleration in self-attention operations and $3.9\times$ acceleration in end-to-end per token latency in long context LLM decoding.
>
---
#### [replaced 101] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15424v2](http://arxiv.org/pdf/2505.15424v2)**

> **作者:** Yan-Shuo Liang; Jia-Rui Chen; Wu-Jun Li
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Continual learning (CL), which requires the model to learn multiple tasks sequentially, is crucial for large language models (LLMs). Recently, low-rank adaptation~(LoRA), one of the most representative parameter-efficient fine-tuning (PEFT) methods, has gained increasing attention in CL of LLMs. However, most existing CL methods based on LoRA typically expand a new LoRA branch to learn each new task and force the new and old LoRA branches to influence old tasks equally, potentially leading to forgetting. In this work, we propose a new method, called gated integration of low-rank adaptation (GainLoRA), for CL of LLMs. GainLoRA expands a new LoRA branch for each new task and introduces gating modules to integrate the new and old LoRA branches. Furthermore, GainLoRA leverages the new gating module to minimize the influence from the new LoRA branch to old tasks, effectively mitigating forgetting and improving the model's overall performance. Experimental results on CL benchmarks demonstrate that GainLoRA outperforms existing state-of-the-art methods.
>
---
#### [replaced 102] TCM-Ladder: A Benchmark for Multimodal Question Answering on Traditional Chinese Medicine
- **分类: cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2505.24063v2](http://arxiv.org/pdf/2505.24063v2)**

> **作者:** Jiacheng Xie; Yang Yu; Ziyang Zhang; Shuai Zeng; Jiaxuan He; Ayush Vasireddy; Xiaoting Tang; Congyu Guo; Lening Zhao; Congcong Jing; Guanghui An; Dong Xu
>
> **摘要:** Traditional Chinese Medicine (TCM), as an effective alternative medicine, has been receiving increasing attention. In recent years, the rapid development of large language models (LLMs) tailored for TCM has highlighted the urgent need for an objective and comprehensive evaluation framework to assess their performance on real-world tasks. However, existing evaluation datasets are limited in scope and primarily text-based, lacking a unified and standardized multimodal question-answering (QA) benchmark. To address this issue, we introduce TCM-Ladder, the first comprehensive multimodal QA dataset specifically designed for evaluating large TCM language models. The dataset covers multiple core disciplines of TCM, including fundamental theory, diagnostics, herbal formulas, internal medicine, surgery, pharmacognosy, and pediatrics. In addition to textual content, TCM-Ladder incorporates various modalities such as images and videos. The dataset was constructed using a combination of automated and manual filtering processes and comprises over 52,000 questions. These questions include single-choice, multiple-choice, fill-in-the-blank, diagnostic dialogue, and visual comprehension tasks. We trained a reasoning model on TCM-Ladder and conducted comparative experiments against nine state-of-the-art general domain and five leading TCM-specific LLMs to evaluate their performance on the dataset. Moreover, we propose Ladder-Score, an evaluation method specifically designed for TCM question answering that effectively assesses answer quality in terms of terminology usage and semantic expression. To the best of our knowledge, this is the first work to systematically evaluate mainstream general domain and TCM-specific LLMs on a unified multimodal benchmark. The datasets and leaderboard are publicly available at https://tcmladder.com and will be continuously updated.
>
---
#### [replaced 103] ColorEcosystem: Powering Personalized, Standardized, and Trustworthy Agentic Service in massive-agent Ecosystem
- **分类: cs.MA; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.21566v2](http://arxiv.org/pdf/2510.21566v2)**

> **作者:** Fangwen Wu; Zheng Wu; Jihong Wang; Yunku Chen; Ruiguang Pei; Heyuan Huang; Xin Liao; Xingyu Lou; Huarong Deng; Zhihui Fu; Weiwen Liu; Zhuosheng Zhang; Weinan Zhang; Jun Wang
>
> **摘要:** With the rapid development of (multimodal) large language model-based agents, the landscape of agentic service management has evolved from single-agent systems to multi-agent systems, and now to massive-agent ecosystems. Current massive-agent ecosystems face growing challenges, including impersonal service experiences, a lack of standardization, and untrustworthy behavior. To address these issues, we propose ColorEcosystem, a novel blueprint designed to enable personalized, standardized, and trustworthy agentic service at scale. Concretely, ColorEcosystem consists of three key components: agent carrier, agent store, and agent audit. The agent carrier provides personalized service experiences by utilizing user-specific data and creating a digital twin, while the agent store serves as a centralized, standardized platform for managing diverse agentic services. The agent audit, based on the supervision of developer and user activities, ensures the integrity and credibility of both service providers and users. Through the analysis of challenges, transitional forms, and practical considerations, the ColorEcosystem is poised to power personalized, standardized, and trustworthy agentic service across massive-agent ecosystems. Meanwhile, we have also implemented part of ColorEcosystem's functionality, and the relevant code is open-sourced at https://github.com/opas-lab/color-ecosystem.
>
---
#### [replaced 104] Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.13227v3](http://arxiv.org/pdf/2505.13227v3)**

> **作者:** Tianbao Xie; Jiaqi Deng; Xiaochuan Li; Junlin Yang; Haoyuan Wu; Jixuan Chen; Wenjing Hu; Xinyuan Wang; Yuhui Xu; Zekun Wang; Yiheng Xu; Junli Wang; Doyen Sahoo; Tao Yu; Caiming Xiong
>
> **备注:** 49 pages, 13 figures
>
> **摘要:** Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at https://osworld-grounding.github.io.
>
---
#### [replaced 105] Detecting and Rectifying Noisy Labels: A Similarity-based Approach
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23964v2](http://arxiv.org/pdf/2509.23964v2)**

> **作者:** Dang Huu-Tien; Minh-Phuong Nguyen; Naoya Inoue
>
> **摘要:** Label noise in datasets could significantly damage the performance and robustness of deep neural networks (DNNs) trained on these datasets. As the size of modern DNNs grows, there is a growing demand for automated tools for detecting such errors. In this paper, we propose post-hoc, model-agnostic noise detection and rectification methods utilizing the penultimate feature from a DNN. Our idea is based on the observation that the similarity between the penultimate feature of a mislabeled data point and its true class data points is higher than that for data points from other classes, making the probability of label occurrence within a tight, similar cluster informative for detecting and rectifying errors. Through theoretical and empirical analyses, we demonstrate that our approach achieves high detection performance across diverse, realistic noise scenarios and can automatically rectify these errors to improve dataset quality. Our implementation is available at https://anonymous.4open.science/r/noise-detection-and-rectification-AD8E.
>
---
#### [replaced 106] EuroSpeech: A Multilingual Speech Corpus
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.00514v2](http://arxiv.org/pdf/2510.00514v2)**

> **作者:** Samuel Pfisterer; Florian Grötschla; Luca A. Lanzendörfer; Florian Yan; Roger Wattenhofer
>
> **备注:** Published in the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmark
>
> **摘要:** Recent progress in speech processing has highlighted that high-quality performance across languages requires substantial training data for each individual language. While existing multilingual datasets cover many languages, they often contain insufficient data for most languages. Thus, trained models perform poorly on the majority of the supported languages. Our work addresses this challenge by introducing a scalable pipeline for constructing speech datasets from parliamentary recordings. The proposed pipeline includes robust components for media retrieval and a two-stage alignment algorithm designed to handle non-verbatim transcripts and long-form audio. Applying this pipeline to recordings from 22 European parliaments, we extract over 61k hours of aligned speech segments, achieving substantial per-language coverage with 19 languages exceeding 1k hours and 22 languages exceeding 500 hours of high-quality speech data. We obtain an average 41.8\% reduction in word error rates over baselines when finetuning an existing ASR model on our dataset, demonstrating the usefulness of our approach.
>
---
#### [replaced 107] Improving the Distributional Alignment of LLMs using Supervision
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.00439v2](http://arxiv.org/pdf/2507.00439v2)**

> **作者:** Gauri Kambhatla; Sanjana Gautam; Angela Zhang; Alex Liu; Ravi Srinivasan; Junyi Jessy Li; Matthew Lease
>
> **摘要:** The ability to accurately align LLMs with human population groups on subjective questions would have great value. In this work, we show that use of simple supervision can greatly improve language model alignment with diverse population groups more consistently, as measured over three datasets spanning various topics. Beyond evaluating average alignment, we also report how alignment varies across specific groups. Our broad findings provide insights into the distributional alignment of LLMs with diverse population groups. By conducting evaluation over many LLMs and prompting strategies, along with open-sourcing our work, we provide a benchmark to stimulate future research.
>
---
#### [replaced 108] TrendFact: A Benchmark for Explainable Hotspot Perception in Fact-Checking with Natural Language Explanation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.15135v4](http://arxiv.org/pdf/2410.15135v4)**

> **作者:** Xiaocheng Zhang; Xi Wang; Yifei Lu; Jianing Wang; Zhuangzhuang Ye; Mengjiao Bao; Peng Yan; Xiaohong Su
>
> **摘要:** Fact-checking benchmarks provide standardized testing criteria for automated fact-checking systems, driving technological advancement. With the surge of misinformation on social media and the emergence of various fact-checking methods, public concern about the transparency of automated systems and the accuracy of fact-checking for high infulence events has grown. However, existing benchmarks fail to meet these urgent needs and are predominantly English-centric, hindering the progress of comprehensive fact-checking. To address these issues, we introduce TrendFact, the first benchmark capable of evaluating hotspot perception ability (HPA) and all fact-checking tasks. TrendFact consists of 7,643 curated samples sourced from trending platforms and professional fact-checking datasets, as well as an evidence library containing 366,634 entries with publication dates. Additionally, to complement existing benchmarks in evaluating system explanation consistency and HPA, we propose two new metrics: ECS and HCPI. Experimental results show that current fact-checking systems face significant limitations when evaluated on TrendFact, which facilitates the development of more robust fact-checking methods. Furthermore, to enhance the capabilities of existing advanced fact-checking systems, the reasoning large language models (RLMs), we propose FactISR, a reasoning framework that integrates dynamic evidence augmentation with influence score-based iterative self-reflection. FactISR effectively improves RLM's performance, offering new insights into explainable and complex fact-checking.
>
---
#### [replaced 109] Aligning LLMs for Multilingual Consistency in Enterprise Applications
- **分类: cs.CL; cs.AI; 68T05, 68T50, 68Q25; I.2.7; I.5.1; I.2.8**

- **链接: [http://arxiv.org/pdf/2509.23659v2](http://arxiv.org/pdf/2509.23659v2)**

> **作者:** Amit Agarwal; Hansa Meghwani; Hitesh Laxmichand Patel; Tao Sheng; Sujith Ravi; Dan Roth
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Large language models (LLMs) remain unreliable for global enterprise applications due to substantial performance gaps between high-resource and mid/low-resource languages, driven by English-centric pretraining and internal reasoning biases. This inconsistency undermines customer experience and operational reliability in multilingual settings such as customer support, content moderation, and information retrieval. Even with advanced Retrieval-Augmented Generation (RAG) systems, we observe up to an 29% accuracy drop in non-English languages compared to English. We propose a practical, batch-wise alignment strategy for fine-tuning LLMs, leveraging semantically equivalent multilingual data in each training batch to directly align model outputs across languages. This approach improves non-English accuracy by up to 23.9% without compromising English performance, model reasoning, or retrieval quality. Our method is simple to implement, scalable, and integrates seamlessly with existing LLM training & deployment pipelines, enabling more robust and equitable multilingual AI solutions in industry.
>
---
#### [replaced 110] Can Confidence Estimates Decide When Chain-of-Thought Is Necessary for LLMs?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.21007v2](http://arxiv.org/pdf/2510.21007v2)**

> **作者:** Samuel Lewis-Lim; Xingwei Tan; Zhixue Zhao; Nikolaos Aletras
>
> **备注:** Under Review
>
> **摘要:** Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT.
>
---
#### [replaced 111] Modeling Bottom-up Information Quality during Language Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17047v2](http://arxiv.org/pdf/2509.17047v2)**

> **作者:** Cui Ding; Yanning Yin; Lena A. Jäger; Ethan Gotlieb Wilcox
>
> **摘要:** Contemporary theories model language processing as integrating both top-down expectations and bottom-up inputs. One major prediction of such models is that the quality of the bottom-up inputs modulates ease of processing -- noisy inputs should lead to difficult and effortful comprehension. We test this prediction in the domain of reading. First, we propose an information-theoretic operationalization for the "quality" of bottom-up information as the mutual information (MI) between visual information and word identity. We formalize this prediction in a mathematical model of reading as a Bayesian update. Second, we test our operationalization by comparing participants' reading times in conditions where words' information quality has been reduced, either by occluding their top or bottom half, with full words. We collect data in English and Chinese. We then use multimodal language models to estimate the mutual information between visual inputs and words. We use these data to estimate the specific effect of reduced information quality on reading times. Finally, we compare how information is distributed across visual forms. In English and Chinese, the upper half contains more information about word identity than the lower half. However, the asymmetry is more pronounced in English, a pattern which is reflected in the reading times.
>
---
#### [replaced 112] Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.09947v2](http://arxiv.org/pdf/2510.09947v2)**

> **作者:** Mir Tafseer Nayeem; Sawsan Alqahtani; Md Tahmid Rahman Laskar; Tasnim Mohiuddin; M Saiful Bari
>
> **备注:** NeurIPS 2025 Workshop
>
> **摘要:** Tokenization is a crucial but under-evaluated step in large language models (LLMs). The standard metric, fertility (the average number of tokens per word), captures compression efficiency but obscures how vocabularies are allocated across languages and domains. We analyze six widely used tokenizers across seven languages and two domains, finding stable fertility for English, high fertility for Chinese, and little domain sensitivity. To address fertility's blind spots, we propose the Single Token Retention Rate (STRR), which measures the proportion of words preserved as single tokens. STRR reveals systematic prioritization of English, strong support for Chinese, and fragmentation in Hindi, offering an interpretable view of cross-lingual fairness. Our results show that STRR complements fertility and provides practical guidance for designing more equitable multilingual tokenizers.
>
---
#### [replaced 113] Dynamic Retriever for In-Context Knowledge Editing via Policy Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.21059v2](http://arxiv.org/pdf/2510.21059v2)**

> **作者:** Mahmud Wasif Nafee; Maiqi Jiang; Haipeng Chen; Yanfu Zhang
>
> **备注:** Accepted at EMNLP 2025. Copyright 2025 Association for Computational Linguistics (CC BY 4.0). 12 pages, 5 figures
>
> **摘要:** Large language models (LLMs) excel at factual recall yet still propagate stale or incorrect knowledge. In-context knowledge editing offers a gradient-free remedy suitable for black-box APIs, but current editors rely on static demonstration sets chosen by surface-level similarity, leading to two persistent obstacles: (i) a quantity-quality trade-off, and (ii) lack of adaptivity to task difficulty. We address these issues by dynamically selecting supporting demonstrations according to their utility for the edit. We propose Dynamic Retriever for In-Context Knowledge Editing (DR-IKE), a lightweight framework that (1) trains a BERT retriever with REINFORCE to rank demonstrations by editing reward, and (2) employs a learnable threshold to prune low-value examples, shortening the prompt when the edit is easy and expanding it when the task is hard. DR-IKE performs editing without modifying model weights, relying solely on forward passes for compatibility with black-box LLMs. On the COUNTERFACT benchmark, it improves edit success by up to 17.1%, reduces latency by 41.6%, and preserves accuracy on unrelated queries, demonstrating scalable and adaptive knowledge editing. The code is available at https://github.com/mwnafee/DR-IKE .
>
---
#### [replaced 114] Antidistillation Sampling
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13146v5](http://arxiv.org/pdf/2504.13146v5)**

> **作者:** Yash Savani; Asher Trockman; Zhili Feng; Yixuan Even Xu; Avi Schwarzschild; Alexander Robey; Marc Finzi; J. Zico Kolter
>
> **摘要:** Frontier models that generate extended reasoning traces inadvertently produce rich token sequences that can facilitate model distillation. Recognizing this vulnerability, model owners may seek sampling strategies that limit the effectiveness of distillation without compromising model performance. Antidistillation sampling provides exactly this capability. By strategically modifying a model's next-token probability distribution, antidistillation sampling poisons reasoning traces, rendering them significantly less effective for distillation while preserving the model's practical utility. For further details, see https://antidistillation.com.
>
---
#### [replaced 115] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.12116v2](http://arxiv.org/pdf/2505.12116v2)**

> **作者:** Fitsum Gaim; Hoyun Song; Huije Lee; Changgeon Ko; Eui Jun Hwang; Jong C. Park
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Content moderation research has recently made significant advances, but remains limited in serving the majority of the world's languages due to the lack of resources, leaving millions of vulnerable users to online hostility. This work presents a large-scale human-annotated multi-task benchmark dataset for abusive language detection in Tigrinya social media with joint annotations for three tasks: abusiveness, sentiment, and topic classification. The dataset comprises 13,717 YouTube comments annotated by nine native speakers, collected from 7,373 videos with a total of over 1.2 billion views across 51 channels. We developed an iterative term clustering approach for effective data selection. Recognizing that around 64% of Tigrinya social media content uses Romanized transliterations rather than native Ge'ez script, our dataset accommodates both writing systems to reflect actual language use. We establish strong baselines across the tasks in the benchmark, while leaving significant challenges for future contributions. Our experiments demonstrate that small fine-tuned models outperform prompted frontier large language models (LLMs) in the low-resource setting, achieving 86.67% F1 in abusiveness detection (7+ points over best LLM), and maintain stronger performance in all other tasks. The benchmark is made public to promote research on online safety.
>
---
#### [replaced 116] Computational-Assisted Systematic Review and Meta-Analysis (CASMA): Effect of a Subclass of GnRH-a on Endometriosis Recurrence
- **分类: cs.CL; cs.IR; stat.AP; stat.ME; H.3.3; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2509.16599v3](http://arxiv.org/pdf/2509.16599v3)**

> **作者:** Sandro Tsang
>
> **备注:** 15 pages, 12 figures and 4 tables. This work describes an information retrieval-driven workflow for medical evidence synthesis, with an application to endometriosis recurrence. The method can be generalized to other systematic reviews. The preregistered protocol is available: https://doi.org/10.17605/OSF.IO/R2DFA
>
> **摘要:** Background: Evidence synthesis facilitates evidence-based medicine. This task becomes increasingly difficult to accomplished with applying computational solutions, since the medical literature grows at astonishing rates. Objective: This study evaluates an information retrieval-driven workflow, CASMA, to enhance the efficiency, transparency, and reproducibility of systematic reviews. Endometriosis recurrence serves as the ideal case due to its complex and ambiguous literature. Methods: The hybrid approach integrates PRISMA guidelines with fuzzy matching and regular expression (regex) to facilitate semi-automated deduplication and filtered records before manual screening. The workflow synthesised evidence from randomised controlled trials on the efficacy of a subclass of gonadotropin-releasing hormone agonists (GnRH-a). A modified splitting method addressed unit-of-analysis errors in multi-arm trials. Results: The workflow sharply reduced the screening workload, taking only 11 days to fetch and filter 33,444 records. Seven eligible RCTs were synthesized (841 patients). The pooled random-effects model yielded a Risk Ratio (RR) of $0.64$ ($95\%$ CI $0.48$ to $0.86$), demonstrating a $36\%$ reduction in recurrence, with non-significant heterogeneity ($I^2=0.00\%$, $\tau^2=0.00$). The findings were robust and stable, as they were backed by sensitivity analyses. Conclusion: This study demonstrates an application of an information-retrieval-driven workflow for medical evidence synthesis. The approach yields valuable clinical results and a generalisable framework to scale up the evidence synthesis, bridging the gap between clinical research and computer science.
>
---
#### [replaced 117] A Multi-faceted Analysis of Cognitive Abilities: Evaluating Prompt Methods with Large Language Models on the CONSORT Checklist
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19139v2](http://arxiv.org/pdf/2510.19139v2)**

> **作者:** Sohyeon Jeon; Hyung-Chul Lee
>
> **摘要:** Despite the rapid expansion of Large Language Models (LLMs) in healthcare, robust and explainable evaluation of their ability to assess clinical trial reporting according to CONSORT standards remains an open challenge. In particular, uncertainty calibration and metacognitive reliability of LLM reasoning are poorly understood and underexplored in medical automation. This study applies a behavioral and metacognitive analytic approach using an expert-validated dataset, systematically comparing two representative LLMs - one general and one domain-specialized - across three prompt strategies. We analyze both cognitive adaptation and calibration error using metrics: Expected Calibration Error (ECE) and a baseline-normalized Relative Calibration Error (RCE) that enables reliable cross-model comparison. Our results reveal pronounced miscalibration and overconfidence in both models, especially under clinical role-playing conditions, with calibration error persisting above clinically relevant thresholds. These findings underscore the need for improved calibration, transparent code, and strategic prompt engineering to develop reliable and explainable medical AI.
>
---
#### [replaced 118] Populism Meets AI: Advancing Populism Research with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07458v3](http://arxiv.org/pdf/2510.07458v3)**

> **作者:** Yujin J. Jung; Eduardo Ryô Tamaki; Julia Chatterley; Grant Mitchell; Semir Dzebo; Cristóbal Sandoval; Levente Littvay; Kirk A. Hawkins
>
> **备注:** 27 pages, 3 figures. Preprint version under review
>
> **摘要:** Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
>
---
#### [replaced 119] Learning to Better Search with Language Models via Guided Reinforced Self-Training
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02992v2](http://arxiv.org/pdf/2410.02992v2)**

> **作者:** Seungyong Moon; Bumsoo Park; Hyun Oh Song
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** While language models have shown remarkable performance across diverse tasks, they still encounter challenges in complex reasoning scenarios. Recent research suggests that language models trained on linearized search traces toward solutions, rather than solely on the final solutions, exhibit improved generalization, despite the search traces being potentially noisy or suboptimal. However, relying on such imperfect traces can result in inefficient use of test-time compute. To address this, we propose guided reinforced self-training (Guided-ReST), a fine-tuning algorithm designed to improve the model's capability for effective search during inference. The key insight behind Guided-ReST is that optimal solutions can serve as valuable step-by-step landmarks to guide the model's search process. Based on this insight, we introduce a novel data generation method that seamlessly incorporates optimal solutions into the model's search procedure, enabling the generation of high-quality search traces. By fine-tuning the model on these search traces, we effectively distill improved search strategies into the model. Our method significantly enhances the search capabilities of language models on arithmetic reasoning and code self-repair tasks, including Countdown, CodeContests, and CodeForces. We release the source code at https://github.com/snu-mllab/guided-rest.
>
---
#### [replaced 120] Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI; math.ST; stat.ME; stat.TH**

- **链接: [http://arxiv.org/pdf/2506.09853v3](http://arxiv.org/pdf/2506.09853v3)**

> **作者:** Xiangning Yu; Zhuohan Wang; Linyi Yang; Haoxuan Li; Anjie Liu; Xiao Xue; Jun Wang; Mengyue Yang
>
> **摘要:** Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness.
>
---
#### [replaced 121] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.09424v2](http://arxiv.org/pdf/2507.09424v2)**

> **作者:** Cathy Jiao; Yijun Pan; Emily Xiao; Daisy Sheng; Niket Jain; Hanzhang Zhao; Ishita Dasgupta; Jiaqi W. Ma; Chenyan Xiong
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Data attribution methods quantify the influence of training data on model outputs and are becoming increasingly relevant for a wide range of LLM research and applications, including dataset curation, model interpretability, data valuation. However, there remain critical gaps in systematic LLM-centric evaluation of data attribution methods. To this end, we introduce DATE-LM (Data Attribution Evaluation in Language Models), a unified benchmark for evaluating data attribution methods through real-world LLM applications. DATE-LM measures attribution quality through three key tasks -- training data selection, toxicity/bias filtering, and factual attribution. Our benchmark is designed for ease of use, enabling researchers to configure and run large-scale evaluations across diverse tasks and LLM architectures. Furthermore, we use DATE-LM to conduct a large-scale evaluation of existing data attribution methods. Our findings show that no single method dominates across all tasks, data attribution methods have trade-offs with simpler baselines, and method performance is sensitive to task-specific evaluation design. Finally, we release a public leaderboard for quick comparison of methods and to facilitate community engagement, with the motivation that DATE-LM can serve as a foundation for future data attribution research in LLMs.
>
---
#### [replaced 122] Are LLMs Empathetic to All? Investigating the Influence of Multi-Demographic Personas on a Model's Empathy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.10328v2](http://arxiv.org/pdf/2510.10328v2)**

> **作者:** Ananya Malik; Nazanin Sabri; Melissa Karnaze; Mai Elsherief
>
> **备注:** 9 pages, 4 figures, 4 tables, EMNLP 2025 Findings
>
> **摘要:** Large Language Models' (LLMs) ability to converse naturally is empowered by their ability to empathetically understand and respond to their users. However, emotional experiences are shaped by demographic and cultural contexts. This raises an important question: Can LLMs demonstrate equitable empathy across diverse user groups? We propose a framework to investigate how LLMs' cognitive and affective empathy vary across user personas defined by intersecting demographic attributes. Our study introduces a novel intersectional analysis spanning 315 unique personas, constructed from combinations of age, culture, and gender, across four LLMs. Results show that attributes profoundly shape a model's empathetic responses. Interestingly, we see that adding multiple attributes at once can attenuate and reverse expected empathy patterns. We show that they broadly reflect real-world empathetic trends, with notable misalignments for certain groups, such as those from Confucian culture. We complement our quantitative findings with qualitative insights to uncover model behaviour patterns across different demographic groups. Our findings highlight the importance of designing empathy-aware LLMs that account for demographic diversity to promote more inclusive and equitable model behaviour.
>
---
#### [replaced 123] Prompting is not Enough: Exploring Knowledge Integration and Controllable Generation
- **分类: cs.CL; cs.AI; 68P20; H.3.4; I.2.6**

- **链接: [http://arxiv.org/pdf/2505.19660v3](http://arxiv.org/pdf/2505.19660v3)**

> **作者:** Tingjia Shen; Hao Wang; Chuan Qin; Ruijun Sun; Yang Song; Defu Lian; Hengshu Zhu; Enhong Chen
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Open-domain question answering (OpenQA) represents a cornerstone in natural language processing (NLP), primarily focused on extracting answers from unstructured textual data. With the rapid advancements in Large Language Models (LLMs), LLM-based OpenQA methods have reaped the benefits of emergent understanding and answering capabilities enabled by massive parameters compared to traditional methods. However, most of these methods encounter two critical challenges: how to integrate knowledge into LLMs effectively and how to adaptively generate results with specific answer formats for various task situations. To address these challenges, we propose a novel framework named GenKI, which aims to improve the OpenQA performance by exploring Knowledge Integration and controllable Generation on LLMs simultaneously. Specifically, we first train a dense passage retrieval model to retrieve associated knowledge from a given knowledge base. Subsequently, we introduce a novel knowledge integration model that incorporates the retrieval knowledge into instructions during fine-tuning to intensify the model. Furthermore, to enable controllable generation in LLMs, we leverage a certain fine-tuned LLM and an ensemble based on text consistency incorporating all coherence, fluency, and answer format assurance. Finally, extensive experiments conducted on the TriviaQA, MSMARCO, and CMRC2018 datasets, featuring diverse answer formats, have demonstrated the effectiveness of GenKI with comparison of state-of-the-art baselines. Moreover, ablation studies have disclosed a linear relationship between the frequency of retrieved knowledge and the model's ability to recall knowledge accurately against the ground truth. Our code of GenKI is available at https://github.com/USTC-StarTeam/GenKI
>
---
#### [replaced 124] Exact Coset Sampling for Quantum Lattice Algorithms
- **分类: quant-ph; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.12341v4](http://arxiv.org/pdf/2509.12341v4)**

> **作者:** Yifan Zhang
>
> **备注:** Project Page: https://github.com/yifanzhang-pro/quantum-lattice
>
> **摘要:** We give a simple replacement for the contested "domain-extension" in Step 9 of a recent windowed-QFT lattice algorithm with complex-Gaussian windows (Chen, 2024). As acknowledged by the author, the reported issue is due to a periodicity/support mismatch when extending only the first coordinate in the presence of offsets, which breaks the intended $\mathbb{Z}_P$-fiber. Our new subroutine replaces domain extension by a pair-shift difference that cancels unknown offsets exactly and synthesizes a uniform cyclic subgroup (a zero-offset coset) of order $P$ inside $(\mathbb{Z}_{M_2})^n$. We adopt a gate-level access model and run a short prepass that measures the designated outcome registers (Chen's Steps 1, 3, and 5), fixing $E=(y',z',h^{\ast})$. We then identify a concrete program point $t^{\star}$ at which an index wire $J \in \mathbb{Z}_P$ is preserved and the coordinate block equals $\mathbf{X}(j)\equiv 2D^2 j\,\mathbf{b}^{\ast}+\mathbf{v}^{\ast}\ (\bmod M_2)$. A compute-copy-uncompute sandwich on the prefix up to $t^{\star}$ yields a reversible evaluator that we call only on basis inputs $j=0,1$ to harvest $V=\mathbf{X}(0)$ and $\Delta=\mathbf{X}(1)-\mathbf{X}(0)\equiv 2D^2\mathbf{b}^{\ast}$ within the same run. We never invert a measurement, and we do not claim the circuit suffix after $t^{\star}$. The default Step $9^{\dagger}$ uses only $\Delta$ (no foreknowledge of $\mathbf{b}^\ast$): set $\mathbf{Z}\leftarrow -\,T\cdot \Delta\ (\bmod M_2)$ for uniform $T\in\mathbb{Z}_P$ and erase $T$ coherently primewise by modular inversion and CRT.
>
---
#### [replaced 125] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08221v3](http://arxiv.org/pdf/2508.08221v3)**

> **作者:** Zihe Liu; Jiashun Liu; Yancheng He; Weixun Wang; Jiaheng Liu; Ling Pan; Xinyu Hu; Shaopan Xiong; Ju Huang; Jian Hu; Shengyi Huang; Johan Obando-Ceron; Siran Yang; Jiamang Wang; Wenbo Su; Bo Zheng
>
> **备注:** 26 pages, 21 figures
>
> **摘要:** Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
>
---
#### [replaced 126] Superficial Self-Improved Reasoners Benefit from Model Merging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02103v2](http://arxiv.org/pdf/2503.02103v2)**

> **作者:** Xiangchi Yuan; Chunhui Zhang; Zheyuan Liu; Dachuan Shi; Leyan Pan; Soroush Vosoughi; Wenke Lee
>
> **备注:** EMNLP 2025
>
> **摘要:** As scaled language models (LMs) approach human-level reasoning capabilities, self-improvement emerges as a solution to synthesizing high-quality data corpus. While previous research has identified model collapse as a risk in self-improvement, where model outputs become increasingly deterministic, we discover a more fundamental challenge: the superficial self-improved reasoners phenomenon. In particular, our analysis reveals that even when LMs show improved in-domain (ID) reasoning accuracy, they actually compromise their generalized reasoning capabilities on out-of-domain (OOD) tasks due to memorization rather than genuine. Through a systematic investigation of LM architecture, we discover that during self-improvement, LM weight updates are concentrated in less reasoning-critical layers, leading to superficial learning. To address this, we propose Iterative Model Merging (IMM), a method that strategically combines weights from original and self-improved models to preserve generalization while incorporating genuine reasoning improvements. Our approach effectively mitigates both LM collapse and superficial learning, moving towards more stable self-improving systems.
>
---
#### [replaced 127] Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.21034v2](http://arxiv.org/pdf/2510.21034v2)**

> **作者:** Barkavi Sundararajan; Somayajulu Sripada; Ehud Reiter
>
> **备注:** Accepted at INLG 2025
>
> **摘要:** A major concern when deploying LLMs in accuracy-critical domains such as sports reporting is that the generated text may not faithfully reflect the input data. We quantify how input structure affects hallucinations and other factual errors in LLM-generated summaries of NBA play-by-play data, across three formats: row-structured, JSON and unstructured. We manually annotated 3,312 factual errors across 180 game summaries produced by two models, Llama-3.1-70B and Qwen2.5-72B. Input structure has a strong effect: JSON input reduces error rates by 69% for Llama and 65% for Qwen compared to unstructured input, while row-structured input reduces errors by 54% for Llama and 51% for Qwen. A two-way repeated measures ANOVA shows that input structure accounts for over 80% of the variance in error rates, with Tukey HSD post hoc tests confirming statistically significant differences between all input formats.
>
---
#### [replaced 128] MOOSE-Chem3: Toward Experiment-Guided Hypothesis Ranking via Simulated Experimental Feedback
- **分类: cs.CL; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2505.17873v3](http://arxiv.org/pdf/2505.17873v3)**

> **作者:** Wanhao Liu; Zonglin Yang; Jue Wang; Lidong Bing; Di Zhang; Dongzhan Zhou; Yuqiang Li; Houqiang Li; Erik Cambria; Wanli Ouyang
>
> **摘要:** Hypothesis ranking is vital for automated scientific discovery, especially in cost-intensive, throughput-limited natural science domains. Current methods focus on pre-experiment ranking, relying solely on language model reasoning without empirical feedback. We introduce experiment-guided ranking, which prioritizes hypotheses based on feedback from prior tests. Due to the impracticality of real experiments, we propose a simulator grounded in domain-specific concepts that models hypothesis performance as a function of similarity to a hidden ground truth, perturbed by noise. Validated against 124 hypotheses with experimentally reported outcomes, the simulator approximates real results with consistent trend alignment. Although deviations exist, they mimic wet-lab noise, promoting more robust ranking strategies. We frame experiment-guided ranking as a sequential decision-making problem and propose an in-context reinforcement learning (ICRL) framework. Our LLM-based policy decomposes hypotheses into functional elements, clusters them by mechanistic roles, and prioritizes recombinations based on feedback. Experiments show our approach significantly outperforms pre-experiment baselines and strong ablations. Our toolkit, comprising the simulator and ICRL framework, enables systematic research on experiment-guided ranking, with the policy serving as a strong proof of concept.
>
---
#### [replaced 129] DiffHeads: Differential Analysis and Inference-Time Masking of Bias Heads in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10142v2](http://arxiv.org/pdf/2510.10142v2)**

> **作者:** Tingxu Han; Wei Song; Ziqi Ding; Ziming Li; Chunrong Fang; Yuekang Li; Dongfang Liu; Zhenyu Chen; Zhenting Wang
>
> **摘要:** Large language models (LLMs) increasingly mediate decisions in domains where unfair treatment of demographic groups is unacceptable. Existing work probes when biased outputs appear, but gives little insight into the mechanisms that generate them, leaving existing mitigations largely fragile. In this paper, we conduct a systematic investigation LLM unfairness and propose DiffHeads, a lightweight debiasing framework for LLMs. We first compare Direct-Answer (DA) prompting to Chain-of-Thought (CoT) prompting across eight representative open- and closed-source LLMs. DA will trigger the nature bias part of LLM and improve measured unfairness by 534.5%-391.9% in both one-turn and two-turn dialogues. Next, we define a token-to-head contribution score that traces each token's influence back to individual attention heads. This reveals a small cluster of bias heads that activate under DA but stay largely dormant with CoT, providing the first causal link between prompting strategy and bias emergence. Finally, building on this insight, we propose DiffHeads that identifies bias heads through differential activation analysis between DA and CoT, and selectively masks only those heads. DiffHeads reduces unfairness by 49.4%, and 40.3% under DA and CoT, respectively, without harming model utility.
>
---
#### [replaced 130] Computational Analysis of Character Development in Holocaust Testimonies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17063v5](http://arxiv.org/pdf/2412.17063v5)**

> **作者:** Esther Shizgal; Eitan Wagner; Renana Keydar; Omri Abend
>
> **摘要:** This work presents a computational approach to analyze character development along the narrative timeline. The analysis characterizes the inner and outer changes the protagonist undergoes within a narrative, and the interplay between them. We consider transcripts of Holocaust survivor testimonies as a test case, each telling the story of an individual in first-person terms. We focus on the survivor's religious trajectory, examining the evolution of their disposition toward religious belief and practice along the testimony. Clustering the resulting trajectories in the dataset, we identify common sequences in the data. Our findings highlight multiple common structures of religiosity across the narratives: in terms of belief, most present a constant disposition, while for practice, most present an oscillating structure, serving as valuable material for historical and sociological research. This work demonstrates the potential of natural language processing techniques for analyzing character evolution through thematic trajectories in narratives.
>
---
#### [replaced 131] Trusted Knowledge Extraction for Operations and Maintenance Intelligence
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.22935v3](http://arxiv.org/pdf/2507.22935v3)**

> **作者:** Kathleen P. Mealey; Jonathan A. Karr Jr.; Priscila Saboia Moreira; Paul R. Brenner; Charles F. Vardeman II
>
> **摘要:** Deriving operational intelligence from organizational data repositories is a key challenge due to the dichotomy of data confidentiality vs data integration objectives, as well as the limitations of Natural Language Processing (NLP) tools relative to the specific knowledge structure of domains such as operations and maintenance. In this work, we discuss Knowledge Graph construction and break down the Knowledge Extraction process into its Named Entity Recognition, Coreference Resolution, Named Entity Linking, and Relation Extraction functional components. We then evaluate sixteen NLP tools in concert with or in comparison to the rapidly advancing capabilities of Large Language Models (LLMs). We focus on the operational and maintenance intelligence use case for trusted applications in the aircraft industry. A baseline dataset is derived from a rich public domain US Federal Aviation Administration dataset focused on equipment failures or maintenance requirements. We assess the zero-shot performance of NLP and LLM tools that can be operated within a controlled, confidential environment (no data is sent to third parties). Based on our observation of significant performance limitations, we discuss the challenges related to trusted NLP and LLM tools as well as their Technical Readiness Level for wider use in mission-critical industries such as aviation. We conclude with recommendations to enhance trust and provide our open-source curated dataset to support further baseline testing and evaluation.
>
---
#### [replaced 132] Estimating LLM Consistency: A User Baseline vs Surrogate Metrics
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23799v3](http://arxiv.org/pdf/2505.23799v3)**

> **作者:** Xiaoyuan Wu; Weiran Lin; Omer Akgul; Lujo Bauer
>
> **备注:** Published as a main conference paper at EMNLP 2025
>
> **摘要:** Large language models (LLMs) are prone to hallucinations and sensitiveto prompt perturbations, often resulting in inconsistent or unreliablegenerated text. Different methods have been proposed to mitigate suchhallucinations and fragility, one of which is to measure theconsistency of LLM responses -- the model's confidence in the responseor likelihood of generating a similar response when resampled. Inprevious work, measuring LLM response consistency often relied oncalculating the probability of a response appearing within a pool of resampledresponses, analyzing internal states, or evaluating logits of resopnses.However, it was not clear how well theseapproaches approximated users' perceptions of consistency of LLMresponses. To find out, we performed a user study ($n=2,976$)demonstrating that current methods for measuring LLM responseconsistency typically do not align well with humans' perceptions of LLMconsistency. We propose a logit-based ensemble method for estimatingLLM consistency and show that our method matches the performance of thebest-performing existing metric in estimating human ratings of LLMconsistency. Our results suggest that methods for estimating LLMconsistency without human evaluation are sufficiently imperfect towarrant broader use of evaluation with human input; this would avoidmisjudging the adequacy of models because of the imperfections ofautomated consistency metrics.
>
---
#### [replaced 133] On the Convergence of Moral Self-Correction in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07290v3](http://arxiv.org/pdf/2510.07290v3)**

> **作者:** Guangliang Liu; Haitao Mao; Bochuan Cao; Zhiyu Xue; Xitong Zhang; Rongrong Wang; Kristen Marie Johnson
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) are able to improve their responses when instructed to do so, a capability known as self-correction. When instructions provide only a general and abstract goal without specific details about potential issues in the response, LLMs must rely on their internal knowledge to improve response quality, a process referred to as intrinsic self-correction. The empirical success of intrinsic self-correction is evident in various applications, but how and why it is effective remains unknown. Focusing on moral self-correction in LLMs, we reveal a key characteristic of intrinsic self-correction: performance convergence through multi-round interactions; and provide a mechanistic analysis of this convergence behavior. Based on our experimental results and analysis, we uncover the underlying mechanism of convergence: consistently injected self-correction instructions activate moral concepts that reduce model uncertainty, leading to converged performance as the activated moral concepts stabilize over successive rounds. This paper demonstrates the strong potential of moral self-correction by showing that it exhibits a desirable property of converged performance.
>
---
