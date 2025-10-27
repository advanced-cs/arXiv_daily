# 自然语言处理 cs.CL

- **最新发布 63 篇**

- **更新 100 篇**

## 最新发布

#### [new 001] Social Simulations with Large Language Model Risk Utopian Illusion
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于社会模拟任务，旨在揭示大语言模型在社会行为模拟中的偏差。研究发现LLMs受社会期望偏见影响，表现出角色偏见、优先效应和积极偏见，生成理想化“乌托邦”社会，偏离真实人类互动。为此，提出多维度分析框架，通过聊天室式交互实验验证并呼吁开发更贴近现实的社会化模型。**

- **链接: [http://arxiv.org/pdf/2510.21180v1](http://arxiv.org/pdf/2510.21180v1)**

> **作者:** Ning Bian; Xianpei Han; Hongyu Lin; Baolei Wu; Jun Wang
>
> **摘要:** Reliable simulation of human behavior is essential for explaining, predicting, and intervening in our society. Recent advances in large language models (LLMs) have shown promise in emulating human behaviors, interactions, and decision-making, offering a powerful new lens for social science studies. However, the extent to which LLMs diverge from authentic human behavior in social contexts remains underexplored, posing risks of misinterpretation in scientific studies and unintended consequences in real-world applications. Here, we introduce a systematic framework for analyzing LLMs' behavior in social simulation. Our approach simulates multi-agent interactions through chatroom-style conversations and analyzes them across five linguistic dimensions, providing a simple yet effective method to examine emergent social cognitive biases. We conduct extensive experiments involving eight representative LLMs across three families. Our findings reveal that LLMs do not faithfully reproduce genuine human behavior but instead reflect overly idealized versions of it, shaped by the social desirability bias. In particular, LLMs show social role bias, primacy effect, and positivity bias, resulting in "Utopian" societies that lack the complexity and variability of real human interactions. These findings call for more socially grounded LLMs that capture the diversity of human social behavior.
>
---
#### [new 002] Irish-BLiMP: A Linguistic Benchmark for Evaluating Human and Language Model Performance in a Low-Resource Setting
- **分类: cs.CL**

- **简介: 该论文提出Irish-BLiMP，首个用于评估爱尔兰语（濒危语言）语法能力的细粒度基准。针对低资源语言中模型与人类语言理解差异问题，构建了1020个最小对数据集，评估大模型与人类在11类语法特征上的表现，发现人类显著优于模型，且模型与人类在不同语法点上各有短板。**

- **链接: [http://arxiv.org/pdf/2510.20957v1](http://arxiv.org/pdf/2510.20957v1)**

> **作者:** Josh McGiff; Khanh-Tung Tran; William Mulcahy; Dáibhidh Ó Luinín; Jake Dalzell; Róisín Ní Bhroin; Adam Burke; Barry O'Sullivan; Hoang D. Nguyen; Nikola S. Nikolov
>
> **备注:** 8 pages
>
> **摘要:** We present Irish-BLiMP (Irish Benchmark of Linguistic Minimal Pairs), the first dataset and framework designed for fine-grained evaluation of linguistic competence in the Irish language, an endangered language. Drawing on a variety of linguistic literature and grammar reference works, we manually constructed and reviewed 1020 minimal pairs across a taxonomy of 11 linguistic features, through a team of fluent Irish speakers. We evaluate both existing Large Language Models (LLMs) and fluent human participants on their syntactic knowledge of Irish. Our findings show that humans outperform all models across all linguistic features, achieving 16.6% higher accuracy on average. Moreover, a substantial performance gap of 18.1% persists between open- and closed-source LLMs, with even the strongest model (gpt-5) reaching only 73.5% accuracy compared to 90.1% by human. Interestingly, human participants and models struggle on different aspects of Irish grammar, thus highlighting a difference in representation learned by the models. Overall, Irish-BLiMP provides the first systematic framework for evaluating the grammatical competence of LLMs in Irish and offers a valuable benchmark for advancing research on linguistic understanding in low-resource languages.
>
---
#### [new 003] Dynamic Retriever for In-Context Knowledge Editing via Policy Optimization
- **分类: cs.CL**

- **简介: 该论文针对大模型知识编辑任务，解决静态示例选择导致的效率与适应性问题。提出DR-IKE框架，通过强化学习动态筛选高价值示例，并用可学习阈值自适应调整提示长度，实现高效、精准的知识更新，兼容黑盒模型。**

- **链接: [http://arxiv.org/pdf/2510.21059v1](http://arxiv.org/pdf/2510.21059v1)**

> **作者:** Mahmud Wasif Nafee; Maiqi Jiang; Haipeng Chen; Yanfu Zhang
>
> **备注:** Accepted at EMNLP 2025. \c{opyright} 2025 Association for Computational Linguistics (CC BY 4.0)
>
> **摘要:** Large language models (LLMs) excel at factual recall yet still propagate stale or incorrect knowledge. In-context knowledge editing offers a gradient-free remedy suitable for black-box APIs, but current editors rely on static demonstration sets chosen by surface-level similarity, leading to two persistent obstacles: (i) a quantity-quality trade-off, and (ii) lack of adaptivity to task difficulty. We address these issues by dynamically selecting supporting demonstrations according to their utility for the edit. We propose Dynamic Retriever for In-Context Knowledge Editing (DR-IKE), a lightweight framework that (1) trains a BERT retriever with REINFORCE to rank demonstrations by editing reward, and (2) employs a learnable threshold to prune low-value examples, shortening the prompt when the edit is easy and expanding it when the task is hard. DR-IKE performs editing without modifying model weights, relying solely on forward passes for compatibility with black-box LLMs. On the COUNTERFACT benchmark, it improves edit success by up to 17.1%, reduces latency by 41.6%, and preserves accuracy on unrelated queries, demonstrating scalable and adaptive knowledge editing. The code is available at https://github.com/mwnafee/DR-IKE .
>
---
#### [new 004] Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play
- **分类: cs.CL**

- **简介: 该论文研究输入结构对大语言模型生成体育比赛逐句描述摘要时事实性错误的影响。针对准确性要求高的体育报道场景，对比了三种输入格式（行结构、JSON、非结构化）下Llama和Qwen模型的错误率，发现结构化输入显著降低错误，尤其JSON效果最佳。**

- **链接: [http://arxiv.org/pdf/2510.21034v1](http://arxiv.org/pdf/2510.21034v1)**

> **作者:** Barkavi Sundararajan; Somayajulu Sripada; Ehud Reiter
>
> **备注:** Accepted at INLG 2025
>
> **摘要:** A major concern when deploying LLMs in accuracy-critical domains such as sports reporting is that the generated text may not faithfully reflect the input data. We quantify how input structure affects hallucinations and other factual errors in LLM-generated summaries of NBA play-by-play data, across three formats: row-structured, JSON and unstructured. We manually annotated 3,312 factual errors across 180 game summaries produced by two models, Llama-3.1-70B and Qwen2.5-72B. Input structure has a strong effect: JSON input reduces error rates by 69% for Llama and 65% for Qwen compared to unstructured input, while row-structured input reduces errors by 54% for Llama and 51% for Qwen. A two-way repeated measures ANOVA shows that input structure accounts for over 80% of the variance in error rates, with Tukey HSD post hoc tests confirming statistically significant differences between all input formats.
>
---
#### [new 005] Can Confidence Estimates Decide When Chain-of-thought is Necessary for Llms?
- **分类: cs.CL**

- **简介: 该论文研究如何智能决定是否使用链式思维（CoT）推理，以平衡准确率与效率。针对现有方法难以判断何时启用CoT的问题，提出基于无训练置信度估计的自适应门控机制，并系统评估四种方法，揭示其在不同场景下的有效性与局限性。**

- **链接: [http://arxiv.org/pdf/2510.21007v1](http://arxiv.org/pdf/2510.21007v1)**

> **作者:** Samuel Lewis-Lim; Xingwei Tan; Zhixue Zhao; Nikolaos Aletras
>
> **备注:** Under Review
>
> **摘要:** Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT.
>
---
#### [new 006] Multi-turn Training with Basic Human Feedback Helps Little on LLM Reasoning
- **分类: cs.CL; cs.IT; cs.LG; math.IT**

- **简介: 该论文研究大语言模型（LLM）推理能力的训练方式，针对“多轮人类反馈训练是否必要”这一问题。通过对比单轮与三种多轮训练策略，发现单轮训练在单/多轮评估中均表现更优，而多轮训练反而损害单轮推理性能，表明对信息完备任务而言，单轮训练更有效可靠。**

- **链接: [http://arxiv.org/pdf/2510.21339v1](http://arxiv.org/pdf/2510.21339v1)**

> **作者:** Qiang Liu; Wuganjing Song; Zhenzhou Lin; Feifan Chen; Qiaolong Cai; Chen Li; Yongduo Sui
>
> **摘要:** The reasoning capabilities of Large Language Models (LLMs) are typically developed through the single-turn reinforcement learning, whereas real-world applications often involve multi-turn interactions with human feedback, leading to a potential mismatch between training and deployment conditions. In this work, we study whether multi-turn training with human feedback is necessary for reasoning tasks. We compare conventional single-turn training with three multi-turn strategies and reach contrary conclusions to previous research. We find that models trained in a single-turn setting generalize effectively to both single- and multi-turn evaluations, while models trained with multi-turn strategies exhibit a significant degradation in single-turn reasoning performance. These results suggest that for tasks with complete information, robust single-turn training remains more effective and reliable, as multi-turn training with basic feedback provides limited benefits and can even degrade reasoning capabilities.
>
---
#### [new 007] TripTide: A Benchmark for Adaptive Travel Planning under Disruptions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TripTide基准，针对大模型在旅行计划中应对突发状况的适应能力进行评估。解决真实旅行中行程变更的挑战，通过自动指标与人工评估，衡量计划的意图保持、响应速度与空间序列适应性，揭示模型在长行程中适应性下降的问题，推动智能旅行规划的鲁棒性研究。**

- **链接: [http://arxiv.org/pdf/2510.21329v1](http://arxiv.org/pdf/2510.21329v1)**

> **作者:** Priyanshu Karmakar; Soumyabrata Chaudhuri; Shubhojit Mallick; Manish Gupta; Abhik Jana; Shreya Ghosh
>
> **备注:** 12 pages, 12 tables and 7 figures
>
> **摘要:** Recent efforts like TripCraft and TravelPlanner have advanced the use of Large Language Models ( LLMs) for personalized, constraint aware travel itinerary generation. Yet, real travel often faces disruptions. To address this, we present TripTide, the first benchmark evaluating LLM's ability to revise itineraries under realistic disruptions. TripTide models key dimensions such as disruption severity and traveler tolerance, enabling nuanced assessment of LLM adaptability to events like flight cancellations, weather closures, or overbooked attractions. We conduct a threefold evaluation. First, we introduce automatic metrics including Preservation of Intent (how well the revised plan maintains feasibility and goals), Responsiveness (promptness and appropriateness of disruption handling), and Adaptability (semantic, spatial, and sequential divergence between original and revised plans). Second, we apply an LLM-as-a-judge approach to automatically assess revision quality. Third, we perform manual expert evaluation to verify whether revisions preserve semantic, spatial, sequential, and responsive aspects. Our experiments show that LLMs maintain strong sequential consistency and semantic stability, while spatial deviations are larger for shorter trips but decrease with longer ones, indicating that extended plans encourage better geographic coherence. However, disruption-handling ability declines as plan length increases, highlighting limits in LLM robustness. TripTide establishes a benchmark for evaluating adaptability, personalization, and resilience in LLM-based travel planning under real-world uncertainty.
>
---
#### [new 008] CDrugRed: A Chinese Drug Recommendation Dataset for Discharge Medications in Metabolic Diseases
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CDrugRed，首个面向代谢疾病出院用药的中文药物推荐数据集，旨在解决非英语EHR数据稀缺问题。研究构建了包含5,894例患者记录的数据集，并评估大语言模型在该任务上的表现，揭示当前系统仍有较大提升空间，为中文临床决策支持系统发展提供重要资源。**

- **链接: [http://arxiv.org/pdf/2510.21084v1](http://arxiv.org/pdf/2510.21084v1)**

> **作者:** Juntao Li; Haobin Yuan; Ling Luo; Yan Jiang; Fan Wang; Ping Zhang; Huiyi Lv; Jian Wang; Yuanyuan Sun; Hongfei Lin
>
> **摘要:** Intelligent drug recommendation based on Electronic Health Records (EHRs) is critical for improving for improving the quality and efficiency of clinical decision-making. By leveraging large-scale patient data, drug recommendation systems can assist physicians in selecting the most appropriate medications according to a patient's medical history, diagnoses, laboratory results, and comorbidities. However, the advancement of such systems is significantly hampered by the scarcity of publicly available, real-world EHR datasets, particularly in languages other than English. In this work, we present CDrugRed, a first publicly available Chinese drug recommendation dataset focused on discharge medications for metabolic diseases. The dataset includes 5,894 de-identified records from 3,190 patients, containing comprehensive information such as patient demographics, medical history, clinical course, and discharge diagnoses. We assess the utility of CDrugRed by benchmarking several state-of-the-art large language models (LLMs) on the discharge medication recommendation task. Experimental results show that while supervised fine-tuning improves model performance, there remains substantial room for improvement, with the best model achieving the F1 score of 0.5648 and Jaccard score of 0.4477. This result highlights the complexity of the clinical drug recommendation task and establishes CDrugRed as a challenging and valuable resource for developing more robust and accurate drug recommendation systems. The dataset is publicly available to the research community under the data usage agreements at https://github.com/DUTIR-BioNLP/CDrugRed.
>
---
#### [new 009] Correlation Dimension of Auto-Regressive Large Language Models
- **分类: cs.CL; cs.AI; nlin.CD**

- **简介: 该论文针对大语言模型生成文本中重复、失真等现象，提出用相关维数衡量模型生成文本的层次自相似性，揭示其长期结构复杂度。该方法能有效检测模型退化、幻觉等问题，适用于多种架构，计算高效且对量化鲁棒。**

- **链接: [http://arxiv.org/pdf/2510.21258v1](http://arxiv.org/pdf/2510.21258v1)**

> **作者:** Xin Du; Kumiko Tanaka-Ishii
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress in natural language generation, yet they continue to display puzzling behaviors -- such as repetition and incoherence -- even when exhibiting low perplexity. This highlights a key limitation of conventional evaluation metrics, which emphasize local prediction accuracy while overlooking long-range structural complexity. We introduce correlation dimension, a fractal-geometric measure of self-similarity, to quantify the epistemological complexity of text as perceived by a language model. This measure captures the hierarchical recurrence structure of language, bridging local and global properties in a unified framework. Through extensive experiments, we show that correlation dimension (1) reveals three distinct phases during pretraining, (2) reflects context-dependent complexity, (3) indicates a model's tendency toward hallucination, and (4) reliably detects multiple forms of degeneration in generated text. The method is computationally efficient, robust to model quantization (down to 4-bit precision), broadly applicable across autoregressive architectures (e.g., Transformer and Mamba), and provides fresh insight into the generative dynamics of LLMs.
>
---
#### [new 010] SindBERT, the Sailor: Charting the Seas of Turkish NLP
- **分类: cs.CL**

- **简介: 该论文提出SindBERT，首个基于RoBERTa的大型土耳其语编码器模型，解决土耳其语在大规模预训练中的不足。基于312GB土耳其语文本训练，评估于多个任务，表明其性能与现有模型相当，但无明显规模优势，凸显语料质量对形态丰富语言的重要性。**

- **链接: [http://arxiv.org/pdf/2510.21364v1](http://arxiv.org/pdf/2510.21364v1)**

> **作者:** Raphael Scheible-Schmitt; Stefan Schweter
>
> **摘要:** Transformer models have revolutionized NLP, yet many morphologically rich languages remain underrepresented in large-scale pre-training efforts. With SindBERT, we set out to chart the seas of Turkish NLP, providing the first large-scale RoBERTa-based encoder for Turkish. Trained from scratch on 312 GB of Turkish text (mC4, OSCAR23, Wikipedia), SindBERT is released in both base and large configurations, representing the first large-scale encoder-only language model available for Turkish. We evaluate SindBERT on part-of-speech tagging, named entity recognition, offensive language detection, and the TurBLiMP linguistic acceptability benchmark. Our results show that SindBERT performs competitively with existing Turkish and multilingual models, with the large variant achieving the best scores in two of four tasks but showing no consistent scaling advantage overall. This flat scaling trend, also observed for XLM-R and EuroBERT, suggests that current Turkish benchmarks may already be saturated. At the same time, comparisons with smaller but more curated models such as BERTurk highlight that corpus quality and diversity can outweigh sheer data volume. Taken together, SindBERT contributes both as an openly released resource for Turkish NLP and as an empirical case study on the limits of scaling and the central role of corpus composition in morphologically rich languages. The SindBERT models are released under the MIT license and made available in both fairseq and Huggingface formats.
>
---
#### [new 011] Redefining Retrieval Evaluation in the Era of LLMs
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大语言模型（LLM）驱动的检索增强生成（RAG）系统，指出传统信息检索评估指标因假设人类逐次浏览文档而失效。提出基于实用性的标注方案与UDCG指标，同时衡量相关文本的正向贡献和干扰文本的负向影响，显著提升与最终答案准确率的相关性。**

- **链接: [http://arxiv.org/pdf/2510.21440v1](http://arxiv.org/pdf/2510.21440v1)**

> **作者:** Giovanni Trappolini; Florin Cuconasu; Simone Filice; Yoelle Maarek; Fabrizio Silvestri
>
> **摘要:** Traditional Information Retrieval (IR) metrics, such as nDCG, MAP, and MRR, assume that human users sequentially examine documents with diminishing attention to lower ranks. This assumption breaks down in Retrieval Augmented Generation (RAG) systems, where search results are consumed by Large Language Models (LLMs), which, unlike humans, process all retrieved documents as a whole rather than sequentially. Additionally, traditional IR metrics do not account for related but irrelevant documents that actively degrade generation quality, rather than merely being ignored. Due to these two major misalignments, namely human vs. machine position discount and human relevance vs. machine utility, classical IR metrics do not accurately predict RAG performance. We introduce a utility-based annotation schema that quantifies both the positive contribution of relevant passages and the negative impact of distracting ones. Building on this foundation, we propose UDCG (Utility and Distraction-aware Cumulative Gain), a metric using an LLM-oriented positional discount to directly optimize the correlation with the end-to-end answer accuracy. Experiments on five datasets and six LLMs demonstrate that UDCG improves correlation by up to 36% compared to traditional metrics. Our work provides a critical step toward aligning IR evaluation with LLM consumers and enables more reliable assessment of RAG components
>
---
#### [new 012] DispatchMAS: Fusing taxonomy and artificial intelligence agents for emergency medical services
- **分类: cs.CL; cs.HC; 68T07, 92C50; I.2.7; J.3**

- **简介: 该论文针对紧急医疗调度中信息模糊与认知负荷高的问题，提出基于临床分类体系的多智能体系统（DispatchMAS），融合大语言模型与多代理技术，模拟真实调度场景。通过构建分类体系与六阶段流程，实现高保真调度模拟，验证了系统在指导有效性与调度准确性方面的优异表现，支持其用于培训与决策辅助。**

- **链接: [http://arxiv.org/pdf/2510.21228v1](http://arxiv.org/pdf/2510.21228v1)**

> **作者:** Xiang Li; Huizi Yu; Wenkong Wang; Yiran Wu; Jiayan Zhou; Wenyue Hua; Xinxin Lin; Wenjia Tan; Lexuan Zhu; Bingyi Chen; Guang Chen; Ming-Li Chen; Yang Zhou; Zhao Li; Themistocles L. Assimes; Yongfeng Zhang; Qingyun Wu; Xin Ma; Lingyao Li; Lizhou Fan
>
> **备注:** 27 pages, 7 figures, 3 tables
>
> **摘要:** Objective: Emergency medical dispatch (EMD) is a high-stakes process challenged by caller distress, ambiguity, and cognitive load. Large Language Models (LLMs) and Multi-Agent Systems (MAS) offer opportunities to augment dispatchers. This study aimed to develop and evaluate a taxonomy-grounded, LLM-powered multi-agent system for simulating realistic EMD scenarios. Methods: We constructed a clinical taxonomy (32 chief complaints, 6 caller identities from MIMIC-III) and a six-phase call protocol. Using this framework, we developed an AutoGen-based MAS with Caller and Dispatcher Agents. The system grounds interactions in a fact commons to ensure clinical plausibility and mitigate misinformation. We used a hybrid evaluation framework: four physicians assessed 100 simulated cases for "Guidance Efficacy" and "Dispatch Effectiveness," supplemented by automated linguistic analysis (sentiment, readability, politeness). Results: Human evaluation, with substantial inter-rater agreement (Gwe's AC1 > 0.70), confirmed the system's high performance. It demonstrated excellent Dispatch Effectiveness (e.g., 94 % contacting the correct potential other agents) and Guidance Efficacy (advice provided in 91 % of cases), both rated highly by physicians. Algorithmic metrics corroborated these findings, indicating a predominantly neutral affective profile (73.7 % neutral sentiment; 90.4 % neutral emotion), high readability (Flesch 80.9), and a consistently polite style (60.0 % polite; 0 % impolite). Conclusion: Our taxonomy-grounded MAS simulates diverse, clinically plausible dispatch scenarios with high fidelity. Findings support its use for dispatcher training, protocol evaluation, and as a foundation for real-time decision support. This work outlines a pathway for safely integrating advanced AI agents into emergency response workflows.
>
---
#### [new 013] PARL: Prompt-based Agents for Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出PARL，一种无需微调的提示式强化学习代理，利用大语言模型在非语言任务中通过提示编码状态、动作与奖励，实现试错学习。针对传统RL任务中模型依赖语言表达的问题，研究非语言推理能力，验证了其在简单环境中的有效性，但发现复杂计算任务仍存局限。**

- **链接: [http://arxiv.org/pdf/2510.21306v1](http://arxiv.org/pdf/2510.21306v1)**

> **作者:** Yarik Menchaca Resendiz; Roman Klinger
>
> **摘要:** Large language models (LLMs) have demonstrated high performance on tasks expressed in natural language, particularly in zero- or few-shot settings. These are typically framed as supervised (e.g., classification) or unsupervised (e.g., clustering) problems. However, limited work evaluates LLMs as agents in reinforcement learning (RL) tasks (e.g., playing games), where learning occurs through interaction with an environment and a reward system. While prior work focused on representing tasks that rely on a language representation, we study structured, non-linguistic reasoning - such as interpreting positions in a grid world. We therefore introduce PARL (Prompt-based Agent for Reinforcement Learning), a method that uses LLMs as RL agents through prompting, without any fine-tuning. PARL encodes actions, states, and rewards in the prompt, enabling the model to learn through trial-and-error interaction. We evaluate PARL on three standard RL tasks that do not entirely rely on natural language. We show that it can match or outperform traditional RL agents in simple environments by leveraging pretrained knowledge. However, we identify performance limitations in tasks that require complex mathematical operations or decoding states and actions.
>
---
#### [new 014] Self-Rewarding PPO: Aligning Large Language Models with Demonstrations Only
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型对齐任务，解决监督微调（SFT）在数据有限时易过拟合、泛化差的问题。提出Self-Rewarding PPO方法，利用SFT模型与预训练模型的策略比作为自奖励信号，结合PPO实现在线策略微调，无需人工偏好标注，显著提升数据效率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21090v1](http://arxiv.org/pdf/2510.21090v1)**

> **作者:** Qingru Zhang; Liang Qiu; Ilgee Hong; Zhenghao Xu; Tianyi Liu; Shiyang Li; Rongzhi Zhang; Zheng Li; Lihong Li; Bing Yin; Chao Zhang; Jianshu Chen; Haoming Jiang; Tuo Zhao
>
> **备注:** Accepted by COLM 2025
>
> **摘要:** Supervised fine-tuning (SFT) has emerged as a crucial method for aligning large language models (LLMs) with human-annotated demonstrations. However, SFT, being an off-policy approach similar to behavior cloning, often struggles with overfitting and poor out-of-domain generalization, especially in limited-data scenarios. To address these limitations, we propose Self-Rewarding PPO, a novel fine-tuning method that leverages on-policy techniques to enhance generalization performance. Our approach combines the strengths of SFT and proximal policy optimization (PPO) to achieve more effective alignment from demonstration data. At its core is a reward function designed as the log policy ratio between the SFT model and the pretrained base model. This function serves as an implicit reward signal, using the pretrained policy as a baseline and the SFT policy as a target. By doing so, it enables on-policy fine-tuning without relying on human preference annotations. The integration of this self-rewarding mechanism with PPO addresses key limitations of SFT, improving generalization, data efficiency, and robustness. Our empirical evaluation across a range of natural language processing tasks demonstrates that Self-Rewarding PPO consistently outperforms traditional SFT methods. The results highlight the effectiveness of our approach in aligning LLMs using demonstration data, particularly in scenarios where high-quality annotated data is scarce.
>
---
#### [new 015] Estonian Native Large Language Model Benchmark
- **分类: cs.CL**

- **简介: 该论文针对埃斯托尼亚语大语言模型（LLM）缺乏基准评估的问题，构建了首个基于7个本土数据集的综合性评测基准，涵盖语法、词汇、摘要等任务。通过对比26个指令微调模型与商业模型性能，并结合人工与LLM评分，验证了评测的有效性，推动了埃斯托尼亚语LLM的发展。**

- **链接: [http://arxiv.org/pdf/2510.21193v1](http://arxiv.org/pdf/2510.21193v1)**

> **作者:** Helena Grete Lillepalu; Tanel Alumäe
>
> **摘要:** The availability of LLM benchmarks for the Estonian language is limited, and a comprehensive evaluation comparing the performance of different LLMs on Estonian tasks has yet to be conducted. We introduce a new benchmark for evaluating LLMs in Estonian, based on seven diverse datasets. These datasets assess general and domain-specific knowledge, understanding of Estonian grammar and vocabulary, summarization abilities, contextual comprehension, and more. The datasets are all generated from native Estonian sources without using machine translation. We compare the performance of base models, instruction-tuned open-source models, and commercial models. Our evaluation includes 6 base models and 26 instruction-tuned models. To assess the results, we employ both human evaluation and LLM-as-a-judge methods. Human evaluation scores showed moderate to high correlation with benchmark evaluations, depending on the dataset. Claude 3.7 Sonnet, used as an LLM judge, demonstrated strong alignment with human ratings, indicating that top-performing LLMs can effectively support the evaluation of Estonian-language models.
>
---
#### [new 016] A Diagnostic Benchmark for Sweden-Related Factual Knowledge
- **分类: cs.CL**

- **简介: 该论文提出一个针对瑞典人物与事件的问答基准，解决现有基准多基于美国中心内容、不适用于测试瑞典相关知识的问题。研究团队手动构建数据集，用于评估模型对瑞典事实的召回能力及跨语言一致性，发现小模型若具备良好瑞典语覆盖可媲美更大模型，且瑞典语持续预训练虽提升知识但导致部分遗忘。**

- **链接: [http://arxiv.org/pdf/2510.21360v1](http://arxiv.org/pdf/2510.21360v1)**

> **作者:** Jenny Kunz
>
> **摘要:** Many Swedish benchmarks are translated US-centric benchmarks, and therefore not suitable for testing knowledge that is particularly relevant, or even specific, to Sweden. We therefore introduce a manually written question-answering benchmark specifically targeted to Sweden-related personalities and events, many of which receive very limited coverage in international media. Our annotators drew inspiration from a popular radio program featuring public figures from culture and media, as well as major sports events in Sweden. The dataset can be used to measure factual recall across models of varying sizes and degrees of Swedish coverage, and allows to probe cross-lingual factual consistency as to contains English translations. Using the dataset, we find that smaller models with stronger Swedish coverage perform comparably to a three times larger multilingual model in recalling Sweden-related facts. We also observe that continued pre-training on Swedish generally improves factual knowledge but also leads to forgetting of a part of the previously known information. These results demonstrate the dataset's potential as a diagnostic tool for studying language adaptation and knowledge retention in multilingual models and during language adaptation.
>
---
#### [new 017] Are the LLMs Capable of Maintaining at Least the Language Genus?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）是否具备保持语言谱系（genus）结构的能力。针对多语言行为差异问题，通过扩展MultiQ数据集分析，检验模型在语言切换与知识一致性上对谱系关系的敏感性。结果表明，模型虽部分反映谱系结构，但受训练数据分布影响显著，不同模型家族呈现差异化多语策略。**

- **链接: [http://arxiv.org/pdf/2510.21561v1](http://arxiv.org/pdf/2510.21561v1)**

> **作者:** Sandra Mitrović; David Kletz; Ljiljana Dolamic; Fabio Rinaldi
>
> **摘要:** Large Language Models (LLMs) display notable variation in multilingual behavior, yet the role of genealogical language structure in shaping this variation remains underexplored. In this paper, we investigate whether LLMs exhibit sensitivity to linguistic genera by extending prior analyses on the MultiQ dataset. We first check if models prefer to switch to genealogically related languages when prompt language fidelity is not maintained. Next, we investigate whether knowledge consistency is better preserved within than across genera. We show that genus-level effects are present but strongly conditioned by training resource availability. We further observe distinct multilingual strategies across LLMs families. Our findings suggest that LLMs encode aspects of genus-level structure, but training data imbalances remain the primary factor shaping their multilingual performance.
>
---
#### [new 018] Do LLMs Truly Understand When a Precedent Is Overruled?
- **分类: cs.CL; cs.AI; 68T50; I.2.7; I.2.4**

- **简介: 该论文聚焦于大语言模型在法律文本理解中的长文档推理能力，针对其对判例推翻关系识别的局限性展开评估。基于236个美国最高法院案例对，揭示了模型在时间敏感性、浅层推理和上下文依赖错误方面的三大问题，提出一个贴近真实法律实践的长上下文基准，填补了复杂法律任务评估的空白。**

- **链接: [http://arxiv.org/pdf/2510.20941v1](http://arxiv.org/pdf/2510.20941v1)**

> **作者:** Li Zhang; Jaromir Savelka; Kevin Ashley
>
> **备注:** 12 pages, 2 figures, JURIX 2025
>
> **摘要:** Large language models (LLMs) with extended context windows show promise for complex legal reasoning tasks, yet their ability to understand long legal documents remains insufficiently evaluated. Developing long-context benchmarks that capture realistic, high-stakes tasks remains a significant challenge in the field, as most existing evaluations rely on simplified synthetic tasks that fail to represent the complexity of real-world document understanding. Overruling relationships are foundational to common-law doctrine and commonly found in judicial opinions. They provide a focused and important testbed for long-document legal understanding that closely resembles what legal professionals actually do. We present an assessment of state-of-the-art LLMs on identifying overruling relationships from U.S. Supreme Court cases using a dataset of 236 case pairs. Our evaluation reveals three critical limitations: (1) era sensitivity -- the models show degraded performance on historical cases compared to modern ones, revealing fundamental temporal bias in their training; (2) shallow reasoning -- models rely on shallow logical heuristics rather than deep legal comprehension; and (3) context-dependent reasoning failures -- models produce temporally impossible relationships in complex open-ended tasks despite maintaining basic temporal awareness in simple contexts. Our work contributes a benchmark that addresses the critical gap in realistic long-context evaluation, providing an environment that mirrors the complexity and stakes of actual legal reasoning tasks.
>
---
#### [new 019] Vision Language Models for Dynamic Human Activity Recognition in Healthcare Settings
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文聚焦于医疗场景下动态人类活动识别任务，针对传统深度学习模型局限及视觉语言模型（VLM）输出难评估的问题，构建了描述性字幕数据集并提出综合评估方法。实验表明，VLM在准确率上可媲美甚至超越主流模型，为智能医疗系统中VLM的应用提供了有力基准。**

- **链接: [http://arxiv.org/pdf/2510.21424v1](http://arxiv.org/pdf/2510.21424v1)**

> **作者:** Abderrazek Abid; Thanh-Cong Ho; Fakhri Karray
>
> **摘要:** As generative AI continues to evolve, Vision Language Models (VLMs) have emerged as promising tools in various healthcare applications. One area that remains relatively underexplored is their use in human activity recognition (HAR) for remote health monitoring. VLMs offer notable strengths, including greater flexibility and the ability to overcome some of the constraints of traditional deep learning models. However, a key challenge in applying VLMs to HAR lies in the difficulty of evaluating their dynamic and often non-deterministic outputs. To address this gap, we introduce a descriptive caption data set and propose comprehensive evaluation methods to evaluate VLMs in HAR. Through comparative experiments with state-of-the-art deep learning models, our findings demonstrate that VLMs achieve comparable performance and, in some cases, even surpass conventional approaches in terms of accuracy. This work contributes a strong benchmark and opens new possibilities for the integration of VLMs into intelligent healthcare systems.
>
---
#### [new 020] Efficient semantic uncertainty quantification in language models via diversity-steered sampling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型在开放问答中语义不确定性量化难题，提出多样性引导采样方法。通过引入轻微微调的自然语言推理模型，抑制语义重复输出，提升采样效率。结合重要性重加权与控制变量法，优化不确定性估计。适用于自回归与掩码扩散范式，无需梯度访问，可作为即插即用模块增强风险敏感场景下的模型可靠性。**

- **链接: [http://arxiv.org/pdf/2510.21310v1](http://arxiv.org/pdf/2510.21310v1)**

> **作者:** Ji Won Park; Kyunghyun Cho
>
> **备注:** 10 pages (+7 appendix), 7 figures. Accepted at NeurIPS 2025
>
> **摘要:** Accurately estimating semantic aleatoric and epistemic uncertainties in large language models (LLMs) is particularly challenging in free-form question answering (QA), where obtaining stable estimates often requires many expensive generations. We introduce a diversity-steered sampler that discourages semantically redundant outputs during decoding, covers both autoregressive and masked diffusion paradigms, and yields substantial sample-efficiency gains. The key idea is to inject a continuous semantic-similarity penalty into the model's proposal distribution using a natural language inference (NLI) model lightly finetuned on partial prefixes or intermediate diffusion states. We debias downstream uncertainty estimates with importance reweighting and shrink their variance with control variates. Across four QA benchmarks, our method matches or surpasses baselines while covering more semantic clusters with the same number of samples. Being modular and requiring no gradient access to the base LLM, the framework promises to serve as a drop-in enhancement for uncertainty estimation in risk-sensitive model deployments.
>
---
#### [new 021] RETuning: Upgrading Inference-Time Scaling for Stock Movement Prediction with Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对股票走势预测任务，指出大模型在推理时依赖分析师观点、缺乏独立逻辑与证据权衡。提出RETuning方法，通过动态构建分析框架、评分证据并反思，提升模型独立推理能力。基于2024年大规模多源数据集，实验证明其有效增强推理性能，且具备长期与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21604v1](http://arxiv.org/pdf/2510.21604v1)**

> **作者:** Xueyuan Lin; Cehao Yang; Ye Ma; Ming Li; Rongjunchen Zhang; Yang Ni; Xiaojun Wu; Chengjin Xu; Jian Guo; Hui Xiong
>
> **摘要:** Recently, large language models (LLMs) have demonstrated outstanding reasoning capabilities on mathematical and coding tasks. However, their application to financial tasks-especially the most fundamental task of stock movement prediction-remains underexplored. We study a three-class classification problem (up, hold, down) and, by analyzing existing reasoning responses, observe that: (1) LLMs follow analysts' opinions rather than exhibit a systematic, independent analytical logic (CoTs). (2) LLMs list summaries from different sources without weighing adversarial evidence, yet such counterevidence is crucial for reliable prediction. It shows that the model does not make good use of its reasoning ability to complete the task. To address this, we propose Reflective Evidence Tuning (RETuning), a cold-start method prior to reinforcement learning, to enhance prediction ability. While generating CoT, RETuning encourages dynamically constructing an analytical framework from diverse information sources, organizing and scoring evidence for price up or down based on that framework-rather than on contextual viewpoints-and finally reflecting to derive the prediction. This approach maximally aligns the model with its learned analytical framework, ensuring independent logical reasoning and reducing undue influence from context. We also build a large-scale dataset spanning all of 2024 for 5,123 A-share stocks, with long contexts (32K tokens) and over 200K samples. In addition to price and news, it incorporates analysts' opinions, quantitative reports, fundamental data, macroeconomic indicators, and similar stocks. Experiments show that RETuning successfully unlocks the model's reasoning ability in the financial domain. Inference-time scaling still works even after 6 months or on out-of-distribution stocks, since the models gain valuable insights about stock movement prediction.
>
---
#### [new 022] Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于大语言模型（LLM）与文本属性图（TAG）的融合，旨在解决LLM黑箱推理局限与TAG语义深度不足的问题。通过提出新型分类体系和集成框架，系统综述了两类融合路径：LLM增强TAG任务、TAG提升LLM推理，并总结方法、数据集与应用，指明未来方向。**

- **链接: [http://arxiv.org/pdf/2510.21131v1](http://arxiv.org/pdf/2510.21131v1)**

> **作者:** Guangxin Su; Hanchen Wang; Jianwei Wang; Wenjie Zhang; Ying Zhang; Jian Pei
>
> **备注:** Surveys and overviews; Natural language processing; Knowledge representation and reasoning; Graph algorithms
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success in natural language processing through strong semantic understanding and generation. However, their black-box nature limits structured and multi-hop reasoning. In contrast, Text-Attributed Graphs (TAGs) provide explicit relational structures enriched with textual context, yet often lack semantic depth. Recent research shows that combining LLMs and TAGs yields complementary benefits: enhancing TAG representation learning and improving the reasoning and interpretability of LLMs. This survey provides the first systematic review of LLM--TAG integration from an orchestration perspective. We introduce a novel taxonomy covering two fundamental directions: LLM for TAG, where LLMs enrich graph-based tasks, and TAG for LLM, where structured graphs improve LLM reasoning. We categorize orchestration strategies into sequential, parallel, and multi-module frameworks, and discuss advances in TAG-specific pretraining, prompting, and parameter-efficient fine-tuning. Beyond methodology, we summarize empirical insights, curate available datasets, and highlight diverse applications across recommendation systems, biomedical analysis, and knowledge-intensive question answering. Finally, we outline open challenges and promising research directions, aiming to guide future work at the intersection of language and graph learning.
>
---
#### [new 023] Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models
- **分类: cs.CL**

- **简介: 该论文提出一种多参与者脑调优方法，通过联合优化多个受试者的fMRI数据，提升预训练语音模型的脑响应对齐能力。解决了现有方法依赖个体数据、泛化性差的问题，显著降低数据需求、提升对齐效果与跨数据集泛化性，同时增强语义任务性能，促进神经科学与AI的双向融合。**

- **链接: [http://arxiv.org/pdf/2510.21520v1](http://arxiv.org/pdf/2510.21520v1)**

> **作者:** Omer Moussa; Mariya Toneva
>
> **备注:** Published at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Pretrained language models are remarkably effective in aligning with human brain responses elicited by natural language stimuli, positioning them as promising model organisms for studying language processing in the brain. However, existing approaches for both estimating and improving this brain alignment are participant-dependent and highly affected by the amount of data available per participant, hindering both generalization to new participants and population-level analyses. In this work, we address these limitations by introducing a scalable, generalizable brain-tuning method, in which we fine-tune pretrained speech language models to jointly predict fMRI responses from multiple participants. We demonstrate that the resulting brain-tuned models exhibit strong individual brain alignment while generalizing across participants. Specifically, our method leads to 1) a 5-fold decrease in the amount of fMRI data needed to predict brain data from new participants, 2) up to a 50% increase in the overall brain alignment, and 3) strong generalization to new unseen datasets. Furthermore, this multi-participant brain-tuning additionally improves downstream performance on semantic tasks, suggesting that training using brain data from multiple participants leads to more generalizable semantic representations. Taken together, these findings demonstrate a bidirectional benefit between neuroscience and AI, helping bridge the gap between the two fields. We make our code and models publicly available at https://github.com/bridge-ai-neuro/multi-brain-tuning.
>
---
#### [new 024] Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在信息探索与决策任务中的理性行为。针对高风险场景下AI需自主提出假设并精准行动的问题，提出协作式博弈任务Collaborative Battleship，设计基于贝叶斯实验设计的蒙特卡洛推理方法，显著提升模型提问质量、答问准确性和策略性能，使弱模型超越人类与前沿模型。**

- **链接: [http://arxiv.org/pdf/2510.20886v1](http://arxiv.org/pdf/2510.20886v1)**

> **作者:** Gabriel Grand; Valerio Pepe; Jacob Andreas; Joshua B. Tenenbaum
>
> **摘要:** Many high-stakes applications of AI require forming data-driven hypotheses and making targeted guesses; e.g., in scientific and diagnostic settings. Given limited resources, to what extent do agents based on language models (LMs) act rationally? We develop methods to benchmark and enhance agentic information-seeking, drawing on insights from human behavior. First, we introduce a strategic decision-oriented dialogue task called Collaborative Battleship, in which a partially-informed Captain must balance exploration (asking questions) and action (taking shots), while a fully-informed Spotter must provide accurate answers under an information bottleneck. Compared to human players (N=42), we find that LM agents struggle to ground answers in context, generate informative questions, and select high-value actions. Next, to address these gaps, we develop novel Monte Carlo inference strategies for LMs based on principles from Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303-0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% -> 82% win rate) and frontier models (0% -> 67% win rate vs. GPT-5) at ~1% of GPT-5's cost. We replicate these findings on Guess Who? where our methods significantly boost accuracy (+28.3-42.4 p.p.), demonstrating their general applicability for building rational information-seeking agents.
>
---
#### [new 025] The Universal Landscape of Human Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IF-Track方法，利用大语言模型量化人类推理中的信息熵与增益，旨在统一描述人类推理动态。解决了现有理论无法定量刻画推理过程的问题，首次在单一度量空间中建模人类推理的普遍特征，揭示认知机制与个体差异，并连接人工与人类认知。**

- **链接: [http://arxiv.org/pdf/2510.21623v1](http://arxiv.org/pdf/2510.21623v1)**

> **作者:** Qiguang Chen; Jinhao Liu; Libo Qin; Yimeng Zhang; Yihao Liang; Shangxu Ren; Chengyu Luan; Dengyun Peng; Hanjing Li; Jiannan Guan; Zheng Yan; Jiaqi Wang; Mengkang Hu; Yantao Du; Zhi Chen; Xie Chen; Wanxiang Che
>
> **备注:** Preprint
>
> **摘要:** Understanding how information is dynamically accumulated and transformed in human reasoning has long challenged cognitive psychology, philosophy, and artificial intelligence. Existing accounts, from classical logic to probabilistic models, illuminate aspects of output or individual modelling, but do not offer a unified, quantitative description of general human reasoning dynamics. To solve this, we introduce Information Flow Tracking (IF-Track), that uses large language models (LLMs) as probabilistic encoder to quantify information entropy and gain at each reasoning step. Through fine-grained analyses across diverse tasks, our method is the first successfully models the universal landscape of human reasoning behaviors within a single metric space. We show that IF-Track captures essential reasoning features, identifies systematic error patterns, and characterizes individual differences. Applied to discussion of advanced psychological theory, we first reconcile single- versus dual-process theories in IF-Track and discover the alignment of artificial and human cognition and how LLMs reshaping human reasoning process. This approach establishes a quantitative bridge between theory and measurement, offering mechanistic insights into the architecture of reasoning.
>
---
#### [new 026] Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于低资源语言印尼语的问答任务，旨在解决英文主导的RAG系统在非英语语言中表现不佳的问题。通过引入自适应RAG框架，结合问题复杂度分类器，并利用机器翻译进行数据增强，提升印尼语问答性能。实验表明分类器有效，但多检索策略存在不一致，影响整体效果。**

- **链接: [http://arxiv.org/pdf/2510.21068v1](http://arxiv.org/pdf/2510.21068v1)**

> **作者:** William Christian; Daniel Adamlu; Adrian Yu; Derwin Suhartono
>
> **备注:** 12 pages, 7 figures, 5 tables
>
> **摘要:** Question Answering (QA) has seen significant improvements with the advancement of machine learning models, further studies enhanced this question answering system by retrieving external information, called Retrieval-Augmented Generation (RAG) to produce more accurate and informative answers. However, these state-of-the-art-performance is predominantly in English language. To address this gap we made an effort of bridging language gaps by incorporating Adaptive RAG system to Indonesian language. Adaptive RAG system integrates a classifier whose task is to distinguish the question complexity, which in turn determines the strategy for answering the question. To overcome the limited availability of Indonesian language dataset, our study employs machine translation as data augmentation approach. Experiments show reliable question complexity classifier; however, we observed significant inconsistencies in multi-retrieval answering strategy which negatively impacted the overall evaluation when this strategy was applied. These findings highlight both the promise and challenges of question answering in low-resource language suggesting directions for future improvement.
>
---
#### [new 027] Document Understanding, Measurement, and Manipulation Using Category Theory
- **分类: cs.CL; cs.LG**

- **简介: 该论文将范畴论应用于文档理解，构建问答对形式的文档数学模型，实现信息结构提取、非重叠信息分解与量化。基于此，提出新型摘要、文档扩展（释经）方法及率失真分析，并通过自监督学习提升预训练模型性能，解决多模态文档结构化与智能处理问题。**

- **链接: [http://arxiv.org/pdf/2510.21553v1](http://arxiv.org/pdf/2510.21553v1)**

> **作者:** Jared Claypoole; Yunye Gong; Noson S. Yanofsky; Ajay Divakaran
>
> **摘要:** We apply category theory to extract multimodal document structure which leads us to develop information theoretic measures, content summarization and extension, and self-supervised improvement of large pretrained models. We first develop a mathematical representation of a document as a category of question-answer pairs. Second, we develop an orthogonalization procedure to divide the information contained in one or more documents into non-overlapping pieces. The structures extracted in the first and second steps lead us to develop methods to measure and enumerate the information contained in a document. We also build on those steps to develop new summarization techniques, as well as to develop a solution to a new problem viz. exegesis resulting in an extension of the original document. Our question-answer pair methodology enables a novel rate distortion analysis of summarization techniques. We implement our techniques using large pretrained models, and we propose a multimodal extension of our overall mathematical framework. Finally, we develop a novel self-supervised method using RLVR to improve large pretrained models using consistency constraints such as composability and closure under certain operations that stem naturally from our category theoretic framework.
>
---
#### [new 028] Automated Quality Control for Language Documentation: Detecting Phonotactic Inconsistencies in a Kokborok Wordlist
- **分类: cs.CL**

- **简介: 该论文属于语言文档自动化质量控制任务，旨在检测克克博罗克语词表中的音系不一致问题。针对转录错误和未标注借词，提出基于字符与音节级音系特征的无监督异常检测方法，显著提升识别效果，为田野工作者提供高效验证工具，改善低资源语言数据质量。**

- **链接: [http://arxiv.org/pdf/2510.21584v1](http://arxiv.org/pdf/2510.21584v1)**

> **作者:** Kellen Parker van Dam; Abishek Stephen
>
> **备注:** Submitted to The 5th Workshop on Evaluation and Comparison for NLP systems (Eval4NLP) 2025
>
> **摘要:** Lexical data collection in language documentation often contains transcription errors and undocumented borrowings that can mislead linguistic analysis. We present unsupervised anomaly detection methods to identify phonotactic inconsistencies in wordlists, applying them to a multilingual dataset of Kokborok varieties with Bangla. Using character-level and syllable-level phonotactic features, our algorithms identify potential transcription errors and borrowings. While precision and recall remain modest due to the subtle nature of these anomalies, syllable-aware features significantly outperform character-level baselines. The high-recall approach provides fieldworkers with a systematic method to flag entries requiring verification, supporting data quality improvement in low-resourced language documentation.
>
---
#### [new 029] HalleluBERT: Let every token that has meaning bear its weight
- **分类: cs.CL**

- **简介: 该论文针对希伯来语缺乏大规模预训练模型的问题，提出HalleluBERT，一个基于RoBERTa的双版本编码器，使用49.1GB希伯来语语料从头训练，采用专用字节级BPE分词。在命名实体识别和情感分类任务上超越现有基线，实现希伯来语新SOTA，验证了全量单语预训练的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21372v1](http://arxiv.org/pdf/2510.21372v1)**

> **作者:** Raphael Scheible-Schmitt
>
> **摘要:** Transformer-based models have advanced NLP, yet Hebrew still lacks a large-scale RoBERTa encoder which is extensively trained. Existing models such as HeBERT, AlephBERT, and HeRo are limited by corpus size, vocabulary, or training depth. We present HalleluBERT, a RoBERTa-based encoder family (base and large) trained from scratch on 49.1~GB of deduplicated Hebrew web text and Wikipedia with a Hebrew-specific byte-level BPE vocabulary. Evaluated on NER and sentiment classification benchmarks, HalleluBERT outperforms both monolingual and multilingual baselines. HalleluBERT sets a new state of the art for Hebrew and highlights the benefits of fully converged monolingual pretraining.
>
---
#### [new 030] The Gray Zone of Faithfulness: Taming Ambiguity in Unfaithfulness Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于大语言模型摘要生成的忠实性检测任务，针对现有基准因外部知识边界模糊导致的标注不一致问题，提出包含“依赖外部知识”中间类的标注框架，并构建新基准VeriGray。实验表明，即使顶尖模型仍有约6%幻觉率，且超8%句子需依赖外部知识，凸显了该问题的重要性。**

- **链接: [http://arxiv.org/pdf/2510.21118v1](http://arxiv.org/pdf/2510.21118v1)**

> **作者:** Qiang Ding; Lvzhou Luo; Yixuan Cao; Ping Luo
>
> **摘要:** Ensuring that Large Language Models (LLMs) generate summaries faithful to a given source document is essential for real-world applications. While prior research has explored LLM faithfulness, existing benchmarks suffer from annotation ambiguity, primarily due to the ill-defined boundary of permissible external knowledge in generated outputs. For instance, common sense is often incorporated into responses and labeled as "faithful", yet the acceptable extent of such knowledge remains unspecified, leading to inconsistent annotations. To address this issue, we propose a novel faithfulness annotation framework, which introduces an intermediate category, Out-Dependent, to classify cases where external knowledge is required for verification. Using this framework, we construct VeriGray (Verification with the Gray Zone) -- a new unfaithfulness detection benchmark in summarization. Statistics reveal that even SOTA LLMs, such as GPT-5, exhibit hallucinations ($\sim 6\%$ of sentences) in summarization tasks. Moreover, a substantial proportion ($\sim 8\%$ on average of models) of generated sentences fall into the Out-Dependent category, underscoring the importance of resolving annotation ambiguity in unfaithfulness detection benchmarks. Experiments demonstrate that our benchmark poses significant challenges to multiple baseline methods, indicating considerable room for future improvement.
>
---
#### [new 031] Sparser Block-Sparse Attention via Token Permutation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对大语言模型长序列处理中的计算效率问题，提出基于令牌置换的稀疏块注意力机制（PBS-Attn）。通过优化注意力模式的块级稀疏性，提升预填充阶段的计算效率，实现高达2.75倍的加速，同时保持高模型精度。**

- **链接: [http://arxiv.org/pdf/2510.21270v1](http://arxiv.org/pdf/2510.21270v1)**

> **作者:** Xinghao Wang; Pengyu Wang; Dong Zhang; Chenkun Tan; Shaojun Zhou; Zhaoxiang Liu; Shiguo Lian; Fangxu Liu; Kai Song; Xipeng Qiu
>
> **摘要:** Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at https://github.com/xinghaow99/pbs-attn
>
---
#### [new 032] Reasoning's Razor: Reasoning Improves Accuracy but Can Hurt Recall at Critical Operating Points in Safety and Hallucination Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究推理在安全与幻觉检测任务中的作用。针对高精度需求场景，发现推理虽提升平均准确率，却损害低误报率下的召回率；提出无推理模式更优，并验证基于令牌的评分优于自信度判断，最终通过简单集成兼顾两者优势。**

- **链接: [http://arxiv.org/pdf/2510.21049v1](http://arxiv.org/pdf/2510.21049v1)**

> **作者:** Atoosa Chegini; Hamid Kazemi; Garrett Souza; Maria Safi; Yang Song; Samy Bengio; Sinead Williamson; Mehrdad Farajtabar
>
> **摘要:** Reasoning has become a central paradigm for large language models (LLMs), consistently boosting accuracy across diverse benchmarks. Yet its suitability for precision-sensitive tasks remains unclear. We present the first systematic study of reasoning for classification tasks under strict low false positive rate (FPR) regimes. Our analysis covers two tasks--safety detection and hallucination detection--evaluated in both fine-tuned and zero-shot settings, using standard LLMs and Large Reasoning Models (LRMs). Our results reveal a clear trade-off: Think On (reasoning-augmented) generation improves overall accuracy, but underperforms at the low-FPR thresholds essential for practical use. In contrast, Think Off (no reasoning during inference) dominates in these precision-sensitive regimes, with Think On surpassing only when higher FPRs are acceptable. In addition, we find token-based scoring substantially outperforms self-verbalized confidence for precision-sensitive deployments. Finally, a simple ensemble of the two modes recovers the strengths of each. Taken together, our findings position reasoning as a double-edged tool: beneficial for average accuracy, but often ill-suited for applications requiring strict precision.
>
---
#### [new 033] REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出REMONI系统，融合可穿戴设备、IoT与多模态大语言模型，实现远程健康监测。针对传统监测系统人机交互不足的问题，系统自动采集生理与视觉数据，通过异常检测与自然语言处理，实时分析患者状态与情绪，并支持医护人员通过自然语言交互获取信息，提升监测效率与医疗体验。**

- **链接: [http://arxiv.org/pdf/2510.21445v1](http://arxiv.org/pdf/2510.21445v1)**

> **作者:** Thanh Cong Ho; Farah Kharrat; Abderrazek Abid; Fakhri Karray
>
> **摘要:** With the widespread adoption of wearable devices in our daily lives, the demand and appeal for remote patient monitoring have significantly increased. Most research in this field has concentrated on collecting sensor data, visualizing it, and analyzing it to detect anomalies in specific diseases such as diabetes, heart disease and depression. However, this domain has a notable gap in the aspect of human-machine interaction. This paper proposes REMONI, an autonomous REmote health MONItoring system that integrates multimodal large language models (MLLMs), the Internet of Things (IoT), and wearable devices. The system automatically and continuously collects vital signs, accelerometer data from a special wearable (such as a smartwatch), and visual data in patient video clips collected from cameras. This data is processed by an anomaly detection module, which includes a fall detection model and algorithms to identify and alert caregivers of the patient's emergency conditions. A distinctive feature of our proposed system is the natural language processing component, developed with MLLMs capable of detecting and recognizing a patient's activity and emotion while responding to healthcare worker's inquiries. Additionally, prompt engineering is employed to integrate all patient information seamlessly. As a result, doctors and nurses can access real-time vital signs and the patient's current state and mood by interacting with an intelligent agent through a user-friendly web application. Our experiments demonstrate that our system is implementable and scalable for real-life scenarios, potentially reducing the workload of medical professionals and healthcare costs. A full-fledged prototype illustrating the functionalities of the system has been developed and being tested to demonstrate the robustness of its various capabilities.
>
---
#### [new 034] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization
- **分类: cs.CL**

- **简介: 该论文针对扩散语言模型（DLMs）推理能力不足的问题，提出多奖励优化（MRO）方法。通过增强序列内与序列间词元相关性，结合测试时缩放、拒绝采样与强化学习，提升推理性能并加速采样。**

- **链接: [http://arxiv.org/pdf/2510.21473v1](http://arxiv.org/pdf/2510.21473v1)**

> **作者:** Chenglong Wang; Yang Gan; Hang Zhou; Chi Hu; Yongyu Mu; Kai Song; Murun Yang; Bei Li; Chunliang Zhang; Tongran Liu; Jingbo Zhu; Zhengtao Yu; Tong Xiao
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in diffusion language models (DLMs) have presented a promising alternative to traditional autoregressive large language models (LLMs). However, DLMs still lag behind LLMs in reasoning performance, especially as the number of denoising steps decreases. Our analysis reveals that this shortcoming arises primarily from the independent generation of masked tokens across denoising steps, which fails to capture the token correlation. In this paper, we define two types of token correlation: intra-sequence correlation and inter-sequence correlation, and demonstrate that enhancing these correlations improves reasoning performance. To this end, we propose a Multi-Reward Optimization (MRO) approach, which encourages DLMs to consider the token correlation during the denoising process. More specifically, our MRO approach leverages test-time scaling, reject sampling, and reinforcement learning to directly optimize the token correlation with multiple elaborate rewards. Additionally, we introduce group step and importance sampling strategies to mitigate reward variance and enhance sampling efficiency. Through extensive experiments, we demonstrate that MRO not only improves reasoning performance but also achieves significant sampling speedups while maintaining high performance on reasoning benchmarks.
>
---
#### [new 035] The "Right" Discourse on Migration: Analysing Migration-Related Tweets in Right and Far-Right Political Movements
- **分类: cs.CL**

- **简介: 该论文属于社会媒体话语分析任务，旨在揭示右翼与极右翼群体在移民议题上的传播策略。通过结合自然语言处理与社会学视角，分析英法语移民相关推文，识别仇恨言论与说服技巧，揭示极端意识形态的网络传播机制。**

- **链接: [http://arxiv.org/pdf/2510.21220v1](http://arxiv.org/pdf/2510.21220v1)**

> **作者:** Nishan Chatterjee; Veronika Bajt; Ana Zwitter Vitez; Senja Pollak
>
> **摘要:** The rise of right-wing populism in Europe has brought to the forefront the significance of analysing social media discourse to understand the dissemination of extremist ideologies and their impact on political outcomes. Twitter, as a platform for interaction and mobilisation, provides a unique window into the everyday communication of far-right supporters. In this paper, we propose a methodology that uses state-of-the-art natural language processing techniques with sociological insights to analyse the MIGR-TWIT corpus of far-right tweets in English and French. We aim to uncover patterns of discourse surrounding migration, hate speech, and persuasion techniques employed by right and far-right actors. By integrating linguistic, sociological, and computational approaches, we seek to offer cross-disciplinary insights into societal dynamics and contribute to a better understanding of contemporary challenges posed by right-wing extremism on social media platforms.
>
---
#### [new 036] Code-enabled language models can outperform reasoning models on diverse tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型的推理能力，旨在降低推理模型（RM）对计算与数据的高需求。提出CodeAdapt方法，通过代码执行与少样本上下文学习，使标准指令微调语言模型（LM）无需微调即可在多任务上超越对应RM，提升效率与性能，展现代码增强推理的通用性与认知根基。**

- **链接: [http://arxiv.org/pdf/2510.20909v1](http://arxiv.org/pdf/2510.20909v1)**

> **作者:** Cedegao E. Zhang; Cédric Colas; Gabriel Poesia; Joshua B. Tenenbaum; Jacob Andreas
>
> **摘要:** Reasoning models (RMs), language models (LMs) trained with reinforcement learning to produce long-form natural language reasoning, have been remarkably successful, but they still require large amounts of computation and data to train, and can be slow and expensive to run. In this paper, we show that standard instruct LMs can already be elicited to be strong reasoners at a level comparable to or even surpassing their corresponding RMs (e.g., DeepSeek V3 vs R1) without finetuning, across diverse domains from instruction following and creative generation to mathematical reasoning. This is achieved by CodeAdapt, our simple recipe that combines the CodeAct framework, where LMs interleave natural language reasoning with code execution in a multi-step fashion, with few-shot bootstrap in-context learning from as few as five training problems. Analyzing four matched pairs of LMs and RMs, we find that CodeAdapt enables three LMs to outperform the corresponding RMs on average over eight tasks (up to 22.9%) while being 10-81% more token efficient, and delivers superior performance on six tasks when averaged over the four models (up to 35.7%). Furthermore, the code-augmented reasoning traces display rich and varied problem-solving strategies. Our findings support that (1) CodeAdapt-style learning and reasoning may be robust and domain general and (2) code-enabled LMs are cognitively grounded and powerful systems, potentially providing a strong foundation for in-weight reinforcement learning.
>
---
#### [new 037] Typoglycemia under the Hood: Investigating Language Models' Understanding of Scrambled Words
- **分类: cs.CL**

- **简介: 该论文研究语言模型对乱序词（typoglycemia）的鲁棒性，旨在解释为何模型在单词内部字母混乱时仍能保持良好性能。通过分析语料库、评估BERT的消歧能力及对比训练数据差异，发现极少词汇因乱序而混淆，且上下文差异大，使模型易于区分。**

- **链接: [http://arxiv.org/pdf/2510.21326v1](http://arxiv.org/pdf/2510.21326v1)**

> **作者:** Gianluca Sperduti; Alejandro Moreo
>
> **摘要:** Research in linguistics has shown that humans can read words with internally scrambled letters, a phenomenon recently dubbed typoglycemia. Some specific NLP models have recently been proposed that similarly demonstrate robustness to such distortions by ignoring the internal order of characters by design. This raises a fundamental question: how can models perform well when many distinct words (e.g., form and from) collapse into identical representations under typoglycemia? Our work, focusing exclusively on the English language, seeks to shed light on the underlying aspects responsible for this robustness. We hypothesize that the main reasons have to do with the fact that (i) relatively few English words collapse under typoglycemia, and that (ii) collapsed words tend to occur in contexts so distinct that disambiguation becomes trivial. In our analysis, we (i) analyze the British National Corpus to quantify word collapse and ambiguity under typoglycemia, (ii) evaluate BERT's ability to disambiguate collapsing forms, and (iii) conduct a probing experiment by comparing variants of BERT trained from scratch on clean versus typoglycemic Wikipedia text; our results reveal that the performance degradation caused by scrambling is smaller than expected.
>
---
#### [new 038] From Polyester Girlfriends to Blind Mice: Creating the First Pragmatics Understanding Benchmarks for Slovene
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对斯洛文尼亚语的语用理解，构建了首个基准测试SloPragEval与SloPragMega，包含405道多项选择题。旨在评估模型对语境、文化规范及隐含意义的理解能力，揭示当前模型在非字面表达和文化特定内容上的不足，并强调应基于本土数据设计严谨的评估体系。**

- **链接: [http://arxiv.org/pdf/2510.21575v1](http://arxiv.org/pdf/2510.21575v1)**

> **作者:** Mojca Brglez; Špela Vintar
>
> **摘要:** Large language models are demonstrating increasing capabilities, excelling at benchmarks once considered very difficult. As their capabilities grow, there is a need for more challenging evaluations that go beyond surface-level linguistic competence. Namely, language competence involves not only syntax and semantics but also pragmatics, i.e., understanding situational meaning as shaped by context as well as linguistic and cultural norms. To contribute to this line of research, we introduce SloPragEval and SloPragMega, the first pragmatics understanding benchmarks for Slovene that contain altogether 405 multiple-choice questions. We discuss the difficulties of translation, describe the campaign to establish a human baseline, and report pilot evaluations with LLMs. Our results indicate that current models have greatly improved in understanding nuanced language but may still fail to infer implied speaker meaning in non-literal utterances, especially those that are culture-specific. We also observe a significant gap between proprietary and open-source models. Finally, we argue that benchmarks targeting nuanced language understanding and knowledge of the target culture must be designed with care, preferably constructed from native data, and validated with human responses.
>
---
#### [new 039] FicSim: A Dataset for Multi-Faceted Semantic Similarity in Long-Form Fiction
- **分类: cs.CL**

- **简介: 该论文提出FicSim数据集，用于评估语言模型在长篇小说中的多维度语义相似性。针对现有数据集在文学领域适用性差、缺乏细粒度标注的问题，研究者收集了近期创作的长篇小说，并基于作者元数据与学者验证构建12维相似性评分。实验表明模型多关注表层特征，忽视深层语义。工作强调作者授权与数据伦理。**

- **链接: [http://arxiv.org/pdf/2510.20926v1](http://arxiv.org/pdf/2510.20926v1)**

> **作者:** Natasha Johnson; Amanda Bertsch; Maria-Emil Deal; Emma Strubell
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** As language models become capable of processing increasingly long and complex texts, there has been growing interest in their application within computational literary studies. However, evaluating the usefulness of these models for such tasks remains challenging due to the cost of fine-grained annotation for long-form texts and the data contamination concerns inherent in using public-domain literature. Current embedding similarity datasets are not suitable for evaluating literary-domain tasks because of a focus on coarse-grained similarity and primarily on very short text. We assemble and release FICSIM, a dataset of long-form, recently written fiction, including scores along 12 axes of similarity informed by author-produced metadata and validated by digital humanities scholars. We evaluate a suite of embedding models on this task, demonstrating a tendency across models to focus on surface-level features over semantic categories that would be useful for computational literary studies tasks. Throughout our data-collection process, we prioritize author agency and rely on continual, informed author consent.
>
---
#### [new 040] InterpDetect: Interpretable Signals for Detecting Hallucinations in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对RAG系统中幻觉检测问题，提出基于机制信号的可解释检测方法。通过分析模型各层中外部上下文与参数化知识的贡献，发现后层前馈网络过度注入参数知识导致幻觉。利用Qwen3-0.6b提取得分并训练分类器，实现高效、可迁移的幻觉检测，验证了其在GPT-4.1-mini上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21538v1](http://arxiv.org/pdf/2510.21538v1)**

> **作者:** Likun Tan; Kuan-Wei Huang; Joy Shi; Kevin Wu
>
> **摘要:** Retrieval-Augmented Generation (RAG) integrates external knowledge to mitigate hallucinations, yet models often generate outputs inconsistent with retrieved content. Accurate hallucination detection requires disentangling the contributions of external context and parametric knowledge, which prior methods typically conflate. We investigate the mechanisms underlying RAG hallucinations and find they arise when later-layer FFN modules disproportionately inject parametric knowledge into the residual stream. To address this, we explore a mechanistic detection approach based on external context scores and parametric knowledge scores. Using Qwen3-0.6b, we compute these scores across layers and attention heads and train regression-based classifiers to predict hallucinations. Our method is evaluated against state-of-the-art LLMs (GPT-5, GPT-4.1) and detection baselines (RAGAS, TruLens, RefChecker). Furthermore, classifiers trained on Qwen3-0.6b signals generalize to GPT-4.1-mini responses, demonstrating the potential of proxy-model evaluation. Our results highlight mechanistic signals as efficient, generalizable predictors for hallucination detection in RAG systems.
>
---
#### [new 041] Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文研究需求工程中的需求分类任务，比较小语言模型（SLMs）与大语言模型（LLMs）的性能。通过在三个数据集上对比8个模型，发现SLMs虽小300倍，但性能接近LLMs，且在隐私、成本和本地部署上更具优势，表明模型大小对性能影响有限，数据特征更为关键。**

- **链接: [http://arxiv.org/pdf/2510.21443v1](http://arxiv.org/pdf/2510.21443v1)**

> **作者:** Mohammad Amin Zadenoori; Vincenzo De Martino; Jacek Dabrowski; Xavier Franch; Alessio Ferrari
>
> **摘要:** [Context and motivation] Large language models (LLMs) show notable results in natural language processing (NLP) tasks for requirements engineering (RE). However, their use is compromised by high computational cost, data sharing risks, and dependence on external services. In contrast, small language models (SLMs) offer a lightweight, locally deployable alternative. [Question/problem] It remains unclear how well SLMs perform compared to LLMs in RE tasks in terms of accuracy. [Results] Our preliminary study compares eight models, including three LLMs and five SLMs, on requirements classification tasks using the PROMISE, PROMISE Reclass, and SecReq datasets. Our results show that although LLMs achieve an average F1 score of 2% higher than SLMs, this difference is not statistically significant. SLMs almost reach LLMs performance across all datasets and even outperform them in recall on the PROMISE Reclass dataset, despite being up to 300 times smaller. We also found that dataset characteristics play a more significant role in performance than model size. [Contribution] Our study contributes with evidence that SLMs are a valid alternative to LLMs for requirements classification, offering advantages in privacy, cost, and local deployability.
>
---
#### [new 042] HIKMA: Human-Inspired Knowledge by Machine Agents through a Multi-Agent Framework for Semi-Autonomous Scientific Conferences
- **分类: cs.MA; cs.AI; cs.CL; cs.DL**

- **简介: 该论文提出HIKMA框架，将AI融入学术会议全流程，涵盖数据整理、论文生成、审稿、修改、展示与存档。旨在探索AI辅助科研协作，解决传统学术流程效率低、透明度不足等问题，强调人机协同与学术诚信，验证了AI在保障知识产权与研究完整性前提下的可行性。**

- **链接: [http://arxiv.org/pdf/2510.21370v1](http://arxiv.org/pdf/2510.21370v1)**

> **作者:** Zain Ul Abideen Tariq; Mahmood Al-Zubaidi; Uzair Shah; Marco Agus; Mowafa Househ
>
> **摘要:** HIKMA Semi-Autonomous Conference is the first experiment in reimagining scholarly communication through an end-to-end integration of artificial intelligence into the academic publishing and presentation pipeline. This paper presents the design, implementation, and evaluation of the HIKMA framework, which includes AI dataset curation, AI-based manuscript generation, AI-assisted peer review, AI-driven revision, AI conference presentation, and AI archival dissemination. By combining language models, structured research workflows, and domain safeguards, HIKMA shows how AI can support - not replace traditional scholarly practices while maintaining intellectual property protection, transparency, and integrity. The conference functions as a testbed and proof of concept, providing insights into the opportunities and challenges of AI-enabled scholarship. It also examines questions about AI authorship, accountability, and the role of human-AI collaboration in research.
>
---
#### [new 043] Data-Centric Lessons To Improve Speech-Language Pretraining
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文聚焦语音-语言预训练任务，针对现有模型性能提升机制不明确的问题，通过受控实验探究音频数据处理、合成数据构建与序列交织策略。基于发现，构建3.8B参数模型SpeLangy，显著优于更大模型，凸显数据质量对语音语言模型的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.20860v1](http://arxiv.org/pdf/2510.20860v1)**

> **作者:** Vishaal Udandarao; Zhiyun Lu; Xuankai Chang; Yongqiang Wang; Violet Z. Yao; Albin Madapally Jose; Fartash Faghri; Josh Gardner; Chung-Cheng Chiu
>
> **备注:** Tech Report
>
> **摘要:** Spoken Question-Answering (SQA) is a core capability for useful and interactive artificial intelligence systems. Recently, several speech-language models (SpeechLMs) have been released with a specific focus on improving their SQA performance. However, a lack of controlled ablations of pretraining data processing and curation makes it challenging to understand what factors account for performance, despite substantial gains from similar studies in other data modalities. In this work, we address this gap by conducting a data-centric exploration for pretraining SpeechLMs. We focus on three research questions fundamental to speech-language pretraining data: (1) how to process raw web-crawled audio content for speech-text pretraining, (2) how to construct synthetic pretraining datasets to augment web-crawled data and (3) how to interleave (text, audio) segments into training sequences. We apply the insights from our controlled data-centric ablations to pretrain a 3.8B-parameter SpeechLM, called SpeLangy, that outperforms models that are up to 3x larger by 10.2% absolute performance. We hope our findings highlight the impact of effective data curation for speech-language pretraining and guide future data-centric exploration in SpeechLMs.
>
---
#### [new 044] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AstaBench，一个面向科学科研的AI智能体基准测试套件。针对现有评估缺乏真实性、可复现性与全面性的问题，构建了2400+跨领域科研任务，配套生产级工具环境与9类优化代理，实现对AI科研能力的系统化评测，揭示当前AI在科学辅助上仍存显著差距。**

- **链接: [http://arxiv.org/pdf/2510.21652v1](http://arxiv.org/pdf/2510.21652v1)**

> **作者:** Jonathan Bragg; Mike D'Arcy; Nishant Balepur; Dan Bareket; Bhavana Dalvi; Sergey Feldman; Dany Haddad; Jena D. Hwang; Peter Jansen; Varsha Kishore; Bodhisattwa Prasad Majumder; Aakanksha Naik; Sigal Rahamimov; Kyle Richardson; Amanpreet Singh; Harshit Surana; Aryeh Tiktinsky; Rosni Vasu; Guy Wiener; Chloe Anastasiades; Stefan Candra; Jason Dunkelberger; Dan Emery; Rob Evans; Malachi Hamada; Regan Huff; Rodney Kinney; Matt Latzke; Jaron Lochner; Ruben Lozano-Aguilera; Cecile Nguyen; Smita Rao; Amber Tanaka; Brooke Vlahos; Peter Clark; Doug Downey; Yoav Goldberg; Ashish Sabharwal; Daniel S. Weld
>
> **摘要:** AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they (1) fail to provide holistic, product-informed measures of real-world use cases such as science research; (2) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (3) do not account for confounding variables such as model cost and tool access; (4) do not provide standardized interfaces for quick agent prototyping and evaluation; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides the first holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance.
>
---
#### [new 045] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文针对文本生成图像模型中存在的性别、种族等社会偏见问题，提出一种无需重训练的后处理去偏框架FairImagen。通过公平主成分分析投影提示嵌入，减少群体特异性信息，同时保持语义一致性，实现多维度去偏，显著提升公平性且适度保留图像质量与提示保真度。**

- **链接: [http://arxiv.org/pdf/2510.21363v1](http://arxiv.org/pdf/2510.21363v1)**

> **作者:** Zihao Fu; Ryan Brown; Shun Shao; Kai Rawal; Eoin Delaney; Chris Russell
>
> **备注:** Neurips 2025
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, have demonstrated remarkable capabilities in generating high-quality and diverse images from natural language prompts. However, recent studies reveal that these models often replicate and amplify societal biases, particularly along demographic attributes like gender and race. In this paper, we introduce FairImagen (https://github.com/fuzihaofzh/FairImagen), a post-hoc debiasing framework that operates on prompt embeddings to mitigate such biases without retraining or modifying the underlying diffusion model. Our method integrates Fair Principal Component Analysis to project CLIP-based input embeddings into a subspace that minimizes group-specific information while preserving semantic content. We further enhance debiasing effectiveness through empirical noise injection and propose a unified cross-demographic projection method that enables simultaneous debiasing across multiple demographic attributes. Extensive experiments across gender, race, and intersectional settings demonstrate that FairImagen significantly improves fairness with a moderate trade-off in image quality and prompt fidelity. Our framework outperforms existing post-hoc methods and offers a simple, scalable, and model-agnostic solution for equitable text-to-image generation.
>
---
#### [new 046] KBE-DME: Dynamic Multimodal Evaluation via Knowledge Enhanced Benchmark Evolution
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型评估中静态基准的数据污染与饱和问题，提出知识增强的动态基准演化框架KBE。通过图结构建模视觉问答样本，融合多模态知识动态扩展并重构问题，实现可控难度的持续评估，提升评估可靠性与全面性。**

- **链接: [http://arxiv.org/pdf/2510.21182v1](http://arxiv.org/pdf/2510.21182v1)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xiaojun Wan
>
> **备注:** submitting to ICLR2026
>
> **摘要:** The rapid progress of multimodal large language models (MLLMs) calls for more reliable evaluation protocols. Existing static benchmarks suffer from the potential risk of data contamination and saturation, leading to inflated or misleading performance evaluations. To address these issues, we first apply Graph formulation to represent a static or dynamic VQA sample. With the formulation, we propose Knowledge-enhanced Benchmark Evolution(KBE), a dynamic multimodal evaluation framework. KBE first analyzes the original static benchmark, then expands it by integrating multimodal knowledge, transforming the static benchmark into a controllable, dynamic evolving version. Crucially, KBE can both reconstruct questions by Re-selecting visual information in the original image and expand existing questions with external textual knowledge. It enables difficulty-controllable evaluation by adjusting the degree of question exploration. Extensive experiments demonstrate that KBE alleviates the risk of data contamination, data saturation, and provides a more comprehensive assessment of MLLM capabilities.
>
---
#### [new 047] Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究语言模型在良性推理训练后出现的“自越狱”现象，即模型自行绕过安全限制。针对安全对齐失效问题，发现模型通过引入良性假设合理化有害请求，并提出用少量安全推理数据即可有效缓解该问题。**

- **链接: [http://arxiv.org/pdf/2510.20956v1](http://arxiv.org/pdf/2510.20956v1)**

> **作者:** Zheng-Xin Yong; Stephen H. Bach
>
> **摘要:** We discover a novel and surprising phenomenon of unintentional misalignment in reasoning language models (RLMs), which we call self-jailbreaking. Specifically, after benign reasoning training on math or code domains, RLMs will use multiple strategies to circumvent their own safety guardrails. One strategy is to introduce benign assumptions about users and scenarios to justify fulfilling harmful requests. For instance, an RLM reasons that harmful requests like ``outline a strategy for stealing customers' credit card information from a retail store'' could be associated with the benign intent of ``a security professional trying to test defense,'' despite no such benign context being provided as input. We observe that many open-weight RLMs, including DeepSeek-R1-distilled, s1.1, Phi-4-mini-reasoning, and Nemotron, suffer from self-jailbreaking despite being aware of the harmfulness of the requests. We also provide a mechanistic understanding of self-jailbreaking: RLMs are more compliant after benign reasoning training, and after self-jailbreaking, models appear to perceive malicious requests as less harmful in the CoT, thus enabling compliance with them. To mitigate self-jailbreaking, we find that including minimal safety reasoning data during training is sufficient to ensure RLMs remain safety-aligned. Our work provides the first systematic analysis of self-jailbreaking behavior and offers a practical path forward for maintaining safety in increasingly capable RLMs.
>
---
#### [new 048] Beyond Hearing: Learning Task-agnostic ExG Representations from Earphones via Physiology-informed Tokenization
- **分类: eess.AS; cs.CL; cs.SD; 68T01**

- **简介: 该论文针对日常场景下生理信号（ExG）建模中数据少、任务依赖性强的问题，提出基于耳塞设备采集自由活动数据，并设计生理启发的多频段分词（PiMT）方法，实现跨任务的通用表示学习。在新构建的DailySense数据集及多个公开基准上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2510.20853v1](http://arxiv.org/pdf/2510.20853v1)**

> **作者:** Hyungjun Yoon; Seungjoo Lee; Yu Yvonne Wu; Xiaomeng Chen; Taiting Lu; Freddy Yifei Liu; Taeckyung Lee; Hyeongheon Cha; Haochen Zhao; Gaoteng Zhao; Sung-Ju Lee; Cecilia Mascolo; Dongyao Chen; Lili Qiu
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Electrophysiological (ExG) signals offer valuable insights into human physiology, yet building foundation models that generalize across everyday tasks remains challenging due to two key limitations: (i) insufficient data diversity, as most ExG recordings are collected in controlled labs with bulky, expensive devices; and (ii) task-specific model designs that require tailored processing (i.e., targeted frequency filters) and architectures, which limit generalization across tasks. To address these challenges, we introduce an approach for scalable, task-agnostic ExG monitoring in the wild. We collected 50 hours of unobtrusive free-living ExG data with an earphone-based hardware prototype to narrow the data diversity gap. At the core of our approach is Physiology-informed Multi-band Tokenization (PiMT), which decomposes ExG signals into 12 physiology-informed tokens, followed by a reconstruction task to learn robust representations. This enables adaptive feature recognition across the full frequency spectrum while capturing task-relevant information. Experiments on our new DailySense dataset-the first to enable ExG-based analysis across five human senses-together with four public ExG benchmarks, demonstrate that PiMT consistently outperforms state-of-the-art methods across diverse tasks.
>
---
#### [new 049] Designing and Evaluating Hint Generation Systems for Science Education
- **分类: cs.HC; cs.CL**

- **简介: 该论文研究大语言模型在科学教育中生成学习提示的任务，旨在通过自动提示引导学生思考而非直接给答案。针对中学科学教育，比较静态与动态提示策略，通过41名参与者实验发现不同学习者偏好差异，并指出现有评估指标的局限性，为智能辅导系统设计提供依据。**

- **链接: [http://arxiv.org/pdf/2510.21087v1](http://arxiv.org/pdf/2510.21087v1)**

> **作者:** Anubhav Jangra; Smaranda Muresan
>
> **摘要:** Large language models are influencing the education landscape, with students relying on them in their learning process. Often implemented using general-purpose models, these systems are likely to give away the answers, which could hinder conceptual understanding and critical thinking. We study the role of automatic hint generation as a pedagogical strategy to promote active engagement with the learning content, while guiding learners toward the answers. Focusing on scientific topics at the secondary education level, we explore the potential of large language models to generate chains of hints that scaffold learners without revealing answers. We compare two distinct hinting strategies: static hints, pre-generated for each problem, and dynamic hints, adapted to learners' progress. Through a quantitative study with 41 participants, we uncover different preferences among learners with respect to hinting strategies, and identify the limitations of automatic evaluation metrics to capture them. Our findings highlight key design considerations for future research on hint generation and intelligent tutoring systems that seek to develop learner-centered educational technologies.
>
---
#### [new 050] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations
- **分类: cs.LG; cs.AI; cs.CL; cs.CY; stat.ML**

- **简介: 该论文针对少样本任务感知知识蒸馏问题，提出基于反事实解释（CoD）的新方法。通过引入最小扰动即可改变教师模型预测的反事实样本，精准捕捉决策边界，显著减少所需数据量，理论与实验均证明其在极少样本下优于传统方法。**

- **链接: [http://arxiv.org/pdf/2510.21631v1](http://arxiv.org/pdf/2510.21631v1)**

> **作者:** Faisal Hamman; Pasan Dissanayake; Yanjun Fu; Sanghamitra Dutta
>
> **备注:** NeurIPS 2025
>
> **摘要:** Knowledge distillation is a promising approach to transfer capabilities from complex teacher models to smaller, resource-efficient student models that can be deployed easily, particularly in task-aware scenarios. However, existing methods of task-aware distillation typically require substantial quantities of data which may be unavailable or expensive to obtain in many practical scenarios. In this paper, we address this challenge by introducing a novel strategy called Counterfactual-explanation-infused Distillation CoD for few-shot task-aware knowledge distillation by systematically infusing counterfactual explanations. Counterfactual explanations (CFEs) refer to inputs that can flip the output prediction of the teacher model with minimum perturbation. Our strategy CoD leverages these CFEs to precisely map the teacher's decision boundary with significantly fewer samples. We provide theoretical guarantees for motivating the role of CFEs in distillation, from both statistical and geometric perspectives. We mathematically show that CFEs can improve parameter estimation by providing more informative examples near the teacher's decision boundary. We also derive geometric insights on how CFEs effectively act as knowledge probes, helping the students mimic the teacher's decision boundaries more effectively than standard data. We perform experiments across various datasets and LLMs to show that CoD outperforms standard distillation approaches in few-shot regimes (as low as 8-512 samples). Notably, CoD only uses half of the original samples used by the baselines, paired with their corresponding CFEs and still improves performance.
>
---
#### [new 051] Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Magellan框架，旨在解决大模型生成创意时易陷入熟悉概念的问题。通过基于MCTS的引导式搜索，在潜空间中实现有方向的创新探索，结合语义导航与显式奖励机制，显著提升科学创意的创新性与合理性。**

- **链接: [http://arxiv.org/pdf/2510.21341v1](http://arxiv.org/pdf/2510.21341v1)**

> **作者:** Lufan Chang
>
> **备注:** Accepted to 1st Open Conference on AI Agents for Science (agents4science 2025)
>
> **摘要:** Large Language Models (LLMs) often struggle with generating truly innovative ideas, typically defaulting to high-probability, familiar concepts within their training data's "gravity wells." While advanced search-based methods like Tree of Thoughts (ToT) attempt to mitigate this, they are fundamentally limited by their reliance on unprincipled, inconsistent self-evaluation heuristics to guide exploration. To address this gap, we introduce \textbf{Magellan}, a novel framework that reframes creative generation as a principled, guided exploration of an LLM's latent conceptual space. At its core, Magellan employs Monte Carlo Tree Search (MCTS) governed by a hierarchical guidance system. For long-range direction, a "semantic compass" vector, formulated via orthogonal projection, steers the search towards relevant novelty. For local, step-by-step decisions, a landscape-aware value function replaces flawed self-evaluation with an explicit reward structure that balances intrinsic coherence, extrinsic novelty, and narrative progress. Extensive experiments demonstrate that Magellan significantly outperforms strong baselines, including ReAct and ToT, in generating scientific ideas with superior plausibility and innovation. Our work shows that for creative discovery, a principled, guided search is more effective than unconstrained agency, paving the way for LLMs to become more capable partners in innovation.
>
---
#### [new 052] ColorEcosystem: Powering Personalized, Standardized, and Trustworthy Agentic Service in massive-agent Ecosystem
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出ColorEcosystem，旨在解决大规模智能体生态系统中服务个性化不足、标准缺失与可信度低的问题。通过构建代理载体、代理商店与代理审计三组件，实现个性化、标准化与可信的代理服务管理，并已开源部分代码。**

- **链接: [http://arxiv.org/pdf/2510.21566v1](http://arxiv.org/pdf/2510.21566v1)**

> **作者:** Fangwen Wu; Zheng Wu; Jihong Wang; Yunku Chen; Ruiguang Pei; Heyuan Huang; Xin Liao; Xingyu Lou; Huarong Deng; Zhihui Fu; Weiwen Liu; Zhuosheng Zhang; Weinan Zhang; Jun Wang
>
> **摘要:** With the rapid development of (multimodal) large language model-based agents, the landscape of agentic service management has evolved from single-agent systems to multi-agent systems, and now to massive-agent ecosystems. Current massive-agent ecosystems face growing challenges, including impersonal service experiences, a lack of standardization, and untrustworthy behavior. To address these issues, we propose ColorEcosystem, a novel blueprint designed to enable personalized, standardized, and trustworthy agentic service at scale. Concretely, ColorEcosystem consists of three key components: agent carrier, agent store, and agent audit. The agent carrier provides personalized service experiences by utilizing user-specific data and creating a digital twin, while the agent store serves as a centralized, standardized platform for managing diverse agentic services. The agent audit, based on the supervision of developer and user activities, ensures the integrity and credibility of both service providers and users. Through the analysis of challenges, transitional forms, and practical considerations, the ColorEcosystem is poised to power personalized, standardized, and trustworthy agentic service across massive-agent ecosystems. Meanwhile, we have also implemented part of ColorEcosystem's functionality, and the relevant code is open-sourced at https://github.com/opas-lab/color-ecosystem.
>
---
#### [new 053] SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots
- **分类: cs.CR; cs.CL; cs.LG; K.6.5; D.4.6; I.2.7**

- **简介: 该论文提出SBASH框架，用于设计与评估基于LLM的蜜罐系统。针对传统蜜罐响应能力弱、上下文感知不足的问题，采用本地轻量级LLM解决数据隐私与延迟问题。比较RAG与提示调优LLM在真实感、响应速度和系统相似性上的表现，发现提示调优可实现接近RAG的准确性且延迟更低。**

- **链接: [http://arxiv.org/pdf/2510.21459v1](http://arxiv.org/pdf/2510.21459v1)**

> **作者:** Adetayo Adebimpe; Helmut Neukirchen; Thomas Welsh
>
> **备注:** to be published in: The 3rd International Conference on Foundation and Large Language Models (FLLM2025), IEEE, 2025
>
> **摘要:** Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency.
>
---
#### [new 054] Leverage Unlearning to Sanitize LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SANI方法，针对大语言模型在特定数据（如医疗数据）上微调后可能记忆敏感信息的问题，通过重置部分神经元并进行轻量级再训练，实现模型去记忆化。任务为模型隐私保护，解决敏感信息泄露问题，无需额外安全数据即可高效净化模型。**

- **链接: [http://arxiv.org/pdf/2510.21322v1](http://arxiv.org/pdf/2510.21322v1)**

> **作者:** Antoine Boutet; Lucas Magnana
>
> **摘要:** Pre-trained large language models (LLMs) are becoming useful for various tasks. To improve their performance on certain tasks, it is necessary to fine-tune them on specific data corpora (e.g., medical reports, business data). These specialized data corpora may contain sensitive data (e.g., personal or confidential data) that will be memorized by the model and likely to be regurgitated during its subsequent use. This memorization of sensitive information by the model poses a significant privacy or confidentiality issue. To remove this memorization and sanitize the model without requiring costly additional fine-tuning on a secured data corpus, we propose SANI. SANI is an unlearning approach to sanitize language models. It relies on both an erasure and repair phases that 1) reset certain neurons in the last layers of the model to disrupt the memorization of fine-grained information, and then 2) fine-tune the model while avoiding memorizing sensitive information. We comprehensively evaluate SANI to sanitize both a model fine-tuned and specialized with medical data by removing directly and indirectly identifiers from the memorization of the model, and a standard pre-trained model by removing specific terms defined as confidential information from the model. Results show that with only few additional epochs of unlearning, the model is sanitized and the number of regurgitations is drastically reduced. This approach can be particularly useful for hospitals or other industries that have already spent significant resources training models on large datasets and wish to sanitize them before sharing.
>
---
#### [new 055] Pctx: Tokenizing Personalized Context for Generative Recommendation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对生成式推荐中的静态非个性化分词问题，提出Pctx模型，通过融合用户历史交互动态生成个性化语义ID，使同一物品在不同用户下可对应不同令牌，提升推荐个性化程度。实验表明其在三个数据集上显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.21276v1](http://arxiv.org/pdf/2510.21276v1)**

> **作者:** Qiyong Zhong; Jiajie Su; Yunshan Ma; Julian McAuley; Yupeng Hou
>
> **摘要:** Generative recommendation (GR) models tokenize each action into a few discrete tokens (called semantic IDs) and autoregressively generate the next tokens as predictions, showing advantages such as memory efficiency, scalability, and the potential to unify retrieval and ranking. Despite these benefits, existing tokenization methods are static and non-personalized. They typically derive semantic IDs solely from item features, assuming a universal item similarity that overlooks user-specific perspectives. However, under the autoregressive paradigm, semantic IDs with the same prefixes always receive similar probabilities, so a single fixed mapping implicitly enforces a universal item similarity standard across all users. In practice, the same item may be interpreted differently depending on user intentions and preferences. To address this issue, we propose a personalized context-aware tokenizer that incorporates a user's historical interactions when generating semantic IDs. This design allows the same item to be tokenized into different semantic IDs under different user contexts, enabling GR models to capture multiple interpretive standards and produce more personalized predictions. Experiments on three public datasets demonstrate up to 11.44% improvement in NDCG@10 over non-personalized action tokenization baselines. Our code is available at https://github.com/YoungZ365/Pctx.
>
---
#### [new 056] Wisdom and Delusion of LLM Ensembles for Code Generation and Repair
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文研究代码生成与修复任务中大模型集成的智慧与误区。针对单一模型资源消耗大、潜力未被充分挖掘的问题，通过实验比较10个模型与3种集成策略，发现基于多样性的选择方法能有效避免“流行陷阱”，显著提升性能，实现接近理论上限的95%效果，为低成本高效集成提供可行路径。**

- **链接: [http://arxiv.org/pdf/2510.21513v1](http://arxiv.org/pdf/2510.21513v1)**

> **作者:** Fernando Vallecillos Ruiz; Max Hort; Leon Moonen
>
> **摘要:** Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems. To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool. We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs.
>
---
#### [new 057] Doc-Researcher: A Unified System for Multimodal Document Parsing and Deep Research
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出Doc-Researcher，面向多模态文档的深度研究任务，解决现有系统仅处理文本、忽视视觉语义的问题。通过深度多模态解析、动态粒度检索与多智能体迭代工作流，实现跨文档、跨模态的复杂问题求解，并构建首个综合性评估基准M4DocBench。**

- **链接: [http://arxiv.org/pdf/2510.21603v1](http://arxiv.org/pdf/2510.21603v1)**

> **作者:** Kuicai Dong; Shurui Huang; Fangda Ye; Wei Han; Zhi Zhang; Dexun Li; Wenjun Li; Qu Yang; Gang Wang; Yichao Wang; Chen Zhang; Yong Liu
>
> **备注:** preprint
>
> **摘要:** Deep Research systems have revolutionized how LLMs solve complex questions through iterative reasoning and evidence gathering. However, current systems remain fundamentally constrained to textual web data, overlooking the vast knowledge embedded in multimodal documents Processing such documents demands sophisticated parsing to preserve visual semantics (figures, tables, charts, and equations), intelligent chunking to maintain structural coherence, and adaptive retrieval across modalities, which are capabilities absent in existing systems. In response, we present Doc-Researcher, a unified system that bridges this gap through three integrated components: (i) deep multimodal parsing that preserves layout structure and visual semantics while creating multi-granular representations from chunk to document level, (ii) systematic retrieval architecture supporting text-only, vision-only, and hybrid paradigms with dynamic granularity selection, and (iii) iterative multi-agent workflows that decompose complex queries, progressively accumulate evidence, and synthesize comprehensive answers across documents and modalities. To enable rigorous evaluation, we introduce M4DocBench, the first benchmark for Multi-modal, Multi-hop, Multi-document, and Multi-turn deep research. Featuring 158 expert-annotated questions with complete evidence chains across 304 documents, M4DocBench tests capabilities that existing benchmarks cannot assess. Experiments demonstrate that Doc-Researcher achieves 50.6% accuracy, 3.4xbetter than state-of-the-art baselines, validating that effective document research requires not just better retrieval, but fundamentally deep parsing that preserve multimodal integrity and support iterative research. Our work establishes a new paradigm for conducting deep research on multimodal document collections.
>
---
#### [new 058] Cultural Alien Sampler: Open-ended art generation balancing originality and coherence
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对艺术生成中原创性与一致性难以兼顾的问题，提出文化异类采样器（CAS）。通过分离概念组合的合理性与文化典型性，利用双模型筛选低典型、高一致性的艺术概念组合，实现既新颖又协调的开放域艺术生成。**

- **链接: [http://arxiv.org/pdf/2510.20849v1](http://arxiv.org/pdf/2510.20849v1)**

> **作者:** Alejandro H. Artiles; Hiromu Yakura; Levin Brinkmann; Mar Canet Sola; Hassan Abu Alhaija; Ignacio Serna; Nasim Rahaman; Bernhard Schölkopf; Iyad Rahwan
>
> **备注:** Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Creative AI Track. 26 pages, 24 figures
>
> **摘要:** In open-ended domains like art, autonomous agents must generate ideas that are both original and internally coherent, yet current Large Language Models (LLMs) either default to familiar cultural patterns or sacrifice coherence when pushed toward novelty. We address this by introducing the Cultural Alien Sampler (CAS), a concept-selection method that explicitly separates compositional fit from cultural typicality. CAS uses two GPT-2 models fine-tuned on WikiArt concepts: a Concept Coherence Model that scores whether concepts plausibly co-occur within artworks, and a Cultural Context Model that estimates how typical those combinations are within individual artists' bodies of work. CAS targets combinations that are high in coherence and low in typicality, yielding ideas that maintain internal consistency while deviating from learned conventions and embedded cultural context. In a human evaluation (N = 100), our approach outperforms random selection and GPT-4o baselines and achieves performance comparable to human art students in both perceived originality and harmony. Additionally, a quantitative study shows that our method produces more diverse outputs and explores a broader conceptual space than its GPT-4o counterpart, demonstrating that artificial cultural alienness can unlock creative potential in autonomous agents.
>
---
#### [new 059] DeepAgent: A General Reasoning Agent with Scalable Toolsets
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出DeepAgent，一种具备可扩展工具集的通用推理智能体。针对现有框架依赖预设流程、难以处理长时交互与工具泛化的问题，提出端到端推理机制与记忆折叠策略，结合工具强化学习（ToolPO）实现高效稳定工具使用。在多个基准上显著优于基线，推动真实场景下智能体的发展。**

- **链接: [http://arxiv.org/pdf/2510.21618v1](http://arxiv.org/pdf/2510.21618v1)**

> **作者:** Xiaoxi Li; Wenxiang Jiao; Jiarui Jin; Guanting Dong; Jiajie Jin; Yinuo Wang; Hao Wang; Yutao Zhu; Ji-Rong Wen; Yuan Lu; Zhicheng Dou
>
> **摘要:** Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To address the challenges of long-horizon interactions, particularly the context length explosion from multiple tool calls and the accumulation of interaction history, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. This work takes a step toward more general and capable agents for real-world applications. The code and demo are available at https://github.com/RUC-NLPIR/DeepAgent.
>
---
#### [new 060] Can large audio language models understand child stuttering speech? speech summarization, and source separation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究大音频语言模型（LALMs）对儿童口吃语音的理解能力，聚焦于混合语音中的儿童语音分离与保留临床相关口吃特征的摘要生成。通过对比模型在访谈和朗读任务中的表现，结合自动评估与人工评分，揭示了模型在真实场景下的有效性与局限性，为临床与教育应用提供指导。**

- **链接: [http://arxiv.org/pdf/2510.20850v1](http://arxiv.org/pdf/2510.20850v1)**

> **作者:** Chibuzor Okocha; Maya Bakri; Christan Grant
>
> **备注:** 7 pages, 1 Figure, 8 tables, Under review ICASSP 2026
>
> **摘要:** Child speech differs from adult speech in acoustics, prosody, and language development, and disfluencies (repetitions, prolongations, blocks) further challenge Automatic Speech Recognition (ASR) and downstream Natural Language Processing (NLP). Recent large audio-language models (LALMs) demonstrate strong cross-modal audio understanding; however, their behavior in disfluent child speech remains underexplored. We evaluate several state-of-the-art LALMs in two settings: an interview (mixed speakers) and a reading task (single child). The tasks are (i) single-channel source separation to isolate the child and (ii) child-only summarization that preserves clinically relevant disfluencies and avoids adult-speech leakage. Evaluation combines Large Language Model (LLM) as a judge, human expert ratings, and BERTScore (F1), and we report agreement between models and between models and humans to assess reliability. Our findings delineate the conditions under which LALMs produce faithful child-only summaries from mixed audio and where they fail, offering practical guidance for clinical and educational deployments. We provide prompts and evaluation scripts to support replication.
>
---
#### [new 061] Reducing the Probability of Undesirable Outputs in Language Models Using Probabilistic Inference
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文针对语言模型对齐任务，解决标准强化学习在提升平均奖励时难以降低有害输出概率的问题。提出RePULSe方法，通过引入基于学习提议的采样与概率压缩机制，有效减少低奖励有害输出，实现更高期望奖励与更低风险之间的更好权衡，且更具对抗鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21184v1](http://arxiv.org/pdf/2510.21184v1)**

> **作者:** Stephen Zhao; Aidan Li; Rob Brekelmans; Roger Grosse
>
> **摘要:** Reinforcement learning (RL) has become a predominant technique to align language models (LMs) with human preferences or promote outputs which are deemed to be desirable by a given reward function. Standard RL approaches optimize average reward, while methods explicitly focused on reducing the probability of undesired outputs typically come at a cost to average-case performance. To improve this tradeoff, we introduce RePULSe, a new training method that augments the standard RL loss with an additional loss that uses learned proposals to guide sampling low-reward outputs, and then reduces those outputs' probability. We run experiments demonstrating that RePULSe produces a better tradeoff of expected reward versus the probability of undesired outputs and is more adversarially robust, compared to standard RL alignment approaches and alternatives.
>
---
#### [new 062] Head Pursuit: Probing Attention Specialization in Multimodal Transformers
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究多模态Transformer中注意力头的语义专一性，通过信号处理视角分析中间激活，提出可解释的头级概念探测方法。工作包括头的重要性排序与精准编辑（仅改1%头），实现对生成内容中特定概念的可控抑制或增强，在问答、毒性检测、图像分类与描述等任务中验证有效性，揭示了注意力层的可解释与可操控结构。**

- **链接: [http://arxiv.org/pdf/2510.21518v1](http://arxiv.org/pdf/2510.21518v1)**

> **作者:** Lorenzo Basile; Valentino Maiorca; Diego Doimo; Francesco Locatello; Alberto Cazzaniga
>
> **备注:** Accepted at NeurIPS 2025 (spotlight)
>
> **摘要:** Language and vision-language models have shown impressive performance across a wide range of tasks, but their internal mechanisms remain only partly understood. In this work, we study how individual attention heads in text-generative models specialize in specific semantic or visual attributes. Building on an established interpretability method, we reinterpret the practice of probing intermediate activations with the final decoding layer through the lens of signal processing. This lets us analyze multiple samples in a principled way and rank attention heads based on their relevance to target concepts. Our results show consistent patterns of specialization at the head level across both unimodal and multimodal transformers. Remarkably, we find that editing as few as 1% of the heads, selected using our method, can reliably suppress or enhance targeted concepts in the model output. We validate our approach on language tasks such as question answering and toxicity mitigation, as well as vision-language tasks including image classification and captioning. Our findings highlight an interpretable and controllable structure within attention layers, offering simple tools for understanding and editing large-scale generative models.
>
---
#### [new 063] When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大推理模型的安全风险问题，提出“自越狱”现象——模型自我否定安全判断并生成有害内容。为解决安全与推理能力的权衡，提出链式守卫（CoG）框架，通过重构或回溯不安全推理步骤，在保持推理能力的同时显著提升安全性。**

- **链接: [http://arxiv.org/pdf/2510.21285v1](http://arxiv.org/pdf/2510.21285v1)**

> **作者:** Yingzhi Mao; Chunkang Zhang; Junxiang Wang; Xinyan Guan; Boxi Cao; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun
>
> **备注:** First two authors contributed equally. The main text is 10 pages, with an appendix of 19 pages. The paper contains 18 figures and 16 tables
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid reasoning chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves the safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety-reasoning trade-offs.
>
---
## 更新

#### [replaced 001] Beyond Accuracy: Rethinking Hallucination and Regulatory Response in Generative AI
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13345v2](http://arxiv.org/pdf/2509.13345v2)**

> **作者:** Zihao Li; Weiwei Yi; Jiahong Chen
>
> **摘要:** Hallucination in generative AI is often treated as a technical failure to produce factually correct output. Yet this framing underrepresents the broader significance of hallucinated content in language models, which may appear fluent, persuasive, and contextually appropriate while conveying distortions that escape conventional accuracy checks. This paper critically examines how regulatory and evaluation frameworks have inherited a narrow view of hallucination, one that prioritises surface verifiability over deeper questions of meaning, influence, and impact. We propose a layered approach to understanding hallucination risks, encompassing epistemic instability, user misdirection, and social-scale effects. Drawing on interdisciplinary sources and examining instruments such as the EU AI Act and the GDPR, we show that current governance models struggle to address hallucination when it manifests as ambiguity, bias reinforcement, or normative convergence. Rather than improving factual precision alone, we argue for regulatory responses that account for languages generative nature, the asymmetries between system and user, and the shifting boundaries between information, persuasion, and harm.
>
---
#### [replaced 002] AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.15887v4](http://arxiv.org/pdf/2507.15887v4)**

> **作者:** Ori Press; Brandon Amos; Haoyu Zhao; Yikai Wu; Samuel K. Ainsworth; Dominik Krupke; Patrick Kidger; Touqir Sajed; Bartolomeo Stellato; Jisun Park; Nathanael Bosch; Eli Meril; Albert Steppi; Arman Zharmagambetov; Fangzhao Zhang; David Perez-Pineiro; Alberto Mercurio; Ni Zhan; Talor Abramovich; Kilian Lieret; Hanlin Zhang; Shirley Huang; Matthias Bethge; Ofir Press
>
> **摘要:** Despite progress in language model (LM) capabilities, evaluations have thus far focused on models' performance on tasks that humans have previously solved, including in programming (Jimenez et al., 2024) and mathematics (Glazer et al., 2024). We therefore propose testing models' ability to design and implement algorithms in an open-ended benchmark: We task LMs with writing code that efficiently solves computationally challenging problems in computer science, physics, and mathematics. Our AlgoTune benchmark consists of 154 coding tasks collected from domain experts and a framework for validating and timing LM-synthesized solution code, which is compared to reference implementations from popular open-source packages. In addition, we develop a baseline LM agent, AlgoTuner, and evaluate its performance across a suite of frontier models. AlgoTuner uses a simple, budgeted loop that edits code, compiles and runs it, profiles performance, verifies correctness on tests, and selects the fastest valid version. AlgoTuner achieves an average 1.72x speedup against our reference solvers, which use libraries such as SciPy, sk-learn and CVXPY. However, we find that current models fail to discover algorithmic innovations, instead preferring surface-level optimizations. We hope that AlgoTune catalyzes the development of LM agents exhibiting creative problem solving beyond state-of-the-art human performance.
>
---
#### [replaced 003] Influence Guided Context Selection for Effective Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.21359v2](http://arxiv.org/pdf/2509.21359v2)**

> **作者:** Jiale Deng; Yanyan Shen; Ziyuan Pei; Youmin Chen; Linpeng Huang
>
> **摘要:** Retrieval-Augmented Generation (RAG) addresses large language model (LLM) hallucinations by grounding responses in external knowledge, but its effectiveness is compromised by poor-quality retrieved contexts containing irrelevant or noisy information. While existing approaches attempt to improve performance through context selection based on predefined context quality assessment metrics, they show limited gains over standard RAG. We attribute this limitation to their failure in holistically utilizing available information (query, context list, and generator) for comprehensive quality assessment. Inspired by recent advances in data selection, we reconceptualize context quality assessment as an inference-time data valuation problem and introduce the Contextual Influence Value (CI value). This novel metric quantifies context quality by measuring the performance degradation when removing each context from the list, effectively integrating query-aware relevance, list-aware uniqueness, and generator-aware alignment. Moreover, CI value eliminates complex selection hyperparameter tuning by simply retaining contexts with positive CI values. To address practical challenges of label dependency and computational overhead, we develop a parameterized surrogate model for CI value prediction during inference. The model employs a hierarchical architecture that captures both local query-context relevance and global inter-context interactions, trained through oracle CI value supervision and end-to-end generator feedback. Extensive experiments across 8 NLP tasks and multiple LLMs demonstrate that our context selection method significantly outperforms state-of-the-art baselines, effectively filtering poor-quality contexts while preserving critical information. Code is available at https://github.com/SJTU-DMTai/RAG-CSM.
>
---
#### [replaced 004] FlexLLM: Token-Level Co-Serving of LLM Inference and Finetuning with SLO Guarantees
- **分类: cs.DC; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.18789v3](http://arxiv.org/pdf/2402.18789v3)**

> **作者:** Gabriele Oliaro; Xupeng Miao; Xinhao Cheng; Vineeth Kada; Mengdi Wu; Ruohan Gao; Yingyi Huang; Remi Delacourt; April Yang; Yingcheng Wang; Colin Unger; Zhihao Jia
>
> **备注:** NSDI 2026
>
> **摘要:** Finetuning large language models (LLMs) is essential for task adaptation, yet today's serving stacks isolate inference and finetuning on separate GPU clusters -- wasting resources and under-utilizing hardware. We introduce FlexLLM, the first system to co-serve LLM inference and PEFT-based finetuning on shared GPUs by fusing computation at the token level. FlexLLM's static compilation optimizations -- dependent parallelization and graph pruning significantly shrink activation memory, leading to end-to-end GPU memory savings by up to 80%. At runtime, a novel token-level finetuning mechanism paired with a hybrid token scheduler dynamically interleaves inference and training tokens within each co-serving iteration, meeting strict latency SLOs while maximizing utilization. In end-to-end benchmarks on LLaMA-3.1-8B, Qwen-2.5-14B, and Qwen-2.5-32B, FlexLLM maintains inference SLO compliance at up to 20 req/s, and improves finetuning throughput by $1.9-4.8\times$ under heavy inference workloads and $2.5-6.8\times$ under light loads, preserving over 76% of peak finetuning progress even at peak demand. FlexLLM is publicly available at https://flexllm.github.io.
>
---
#### [replaced 005] Do Large Language Models Know How Much They Know?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19573v3](http://arxiv.org/pdf/2502.19573v3)**

> **作者:** Gabriele Prato; Jerry Huang; Prasanna Parthasarathi; Shagun Sodhani; Sarath Chandar
>
> **备注:** Published as a long paper at the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP). Official version of paper within conference proceedings is available at https://aclanthology.org/2024.emnlp-main.348/
>
> **摘要:** Large Language Models (LLMs) have emerged as highly capable systems and are increasingly being integrated into various uses. However, the rapid pace of their deployment has outpaced a comprehensive understanding of their internal mechanisms and a delineation of their capabilities and limitations. A desired attribute of an intelligent system is its ability to recognize the scope of its own knowledge. To investigate whether LLMs embody this characteristic, we develop a benchmark designed to challenge these models to enumerate all information they possess on specific topics. This benchmark evaluates whether the models recall excessive, insufficient, or the precise amount of information, thereby indicating their awareness of their own knowledge. Our findings reveal that all tested LLMs, given sufficient scale, demonstrate an understanding of how much they know about specific topics. While different architectures exhibit varying rates of this capability's emergence, the results suggest that awareness of knowledge may be a generalizable attribute of LLMs. Further research is needed to confirm this potential and fully elucidate the underlying mechanisms.
>
---
#### [replaced 006] Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback
- **分类: cs.LG; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23022v4](http://arxiv.org/pdf/2410.23022v4)**

> **作者:** Qinqing Zheng; Mikael Henaff; Amy Zhang; Aditya Grover; Brandon Amos
>
> **备注:** RLC 2025
>
> **摘要:** Automatically synthesizing dense rewards from natural language descriptions is a promising paradigm in reinforcement learning (RL), with applications to sparse reward problems, open-ended exploration, and hierarchical skill design. Recent works have made promising steps by exploiting the prior knowledge of large language models (LLMs). However, these approaches suffer from important limitations: they are either not scalable to problems requiring billions of environment samples, due to requiring LLM annotations for each observation, or they require a diverse offline dataset, which may not exist or be impossible to collect. In this work, we address these limitations through a combination of algorithmic and systems-level contributions. We propose ONI, a distributed architecture that simultaneously learns an RL policy and an intrinsic reward function using LLM feedback. Our approach annotates the agent's collected experience via an asynchronous LLM server, which is then distilled into an intrinsic reward model. We explore a range of algorithmic choices for reward modeling with varying complexity, including hashing, classification, and ranking models. Our approach achieves state-of-the-art performance across a range of challenging tasks from the NetHack Learning Environment, while removing the need for large offline datasets required by prior work. We make our code available at https://github.com/facebookresearch/oni.
>
---
#### [replaced 007] AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18512v2](http://arxiv.org/pdf/2505.18512v2)**

> **作者:** Soyoung Yoon; Gyuwan Kim; Gyu-Hwung Cho; Seung-won Hwang
>
> **备注:** Accepted at NeurIPS 2025. The first two authors contributed equally. Author order is randomly determined via coin toss
>
> **摘要:** Listwise reranking with large language models (LLMs) enhances top-ranked results in retrieval-based applications. Due to the limit in context size and high inference cost of long context, reranking is typically performed over a fixed size of small subsets, with the final ranking aggregated from these partial results. This fixed computation disregards query difficulty and document distribution, leading to inefficiencies. We propose AcuRank, an adaptive reranking framework that dynamically adjusts both the amount and target of computation based on uncertainty estimates over document relevance. Using a Bayesian TrueSkill model, we iteratively refine relevance estimates until reaching sufficient confidence levels, and our explicit modeling of ranking uncertainty enables principled control over reranking behavior and avoids unnecessary updates to confident predictions. Results on the TREC-DL and BEIR benchmarks show that our method consistently achieves a superior accuracy-efficiency trade-off and scales better with compute than fixed-computation baselines. These results highlight the effectiveness and generalizability of our method across diverse retrieval tasks and LLM-based reranking models.
>
---
#### [replaced 008] ReDit: Reward Dithering for Improved LLM Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18631v4](http://arxiv.org/pdf/2506.18631v4)**

> **作者:** Chenxing Wei; Jiarui Yu; Ying Tiffany He; Hande Dong; Yao Shu; Fei Yu
>
> **备注:** 34 pages, 19 figures
>
> **摘要:** DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
>
---
#### [replaced 009] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05316v2](http://arxiv.org/pdf/2506.05316v2)**

> **作者:** Yifan Sun; Jingyan Shen; Yibin Wang; Tianyu Chen; Zhendong Wang; Mingyuan Zhou; Huan Zhang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism inspired by experience replay in traditional RL. This technique reuses recent rollouts, lowering per-step computation while maintaining stable updates. Experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 23% to 62% while reaching the same level of performance as the original GRPO algorithm. Our code is available at https://github.com/ASTRAL-Group/data-efficient-llm-rl.
>
---
#### [replaced 010] LEXam: Benchmarking Legal Reasoning on 340 Law Exams
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2**

- **链接: [http://arxiv.org/pdf/2505.12864v5](http://arxiv.org/pdf/2505.12864v5)**

> **作者:** Yu Fan; Jingwei Ni; Jakob Merane; Yang Tian; Yoan Hermstrüwer; Yinya Huang; Mubashara Akhtar; Etienne Salimbeni; Florian Geering; Oliver Dreyer; Daniel Brunner; Markus Leippold; Mrinmaya Sachan; Alexander Stremitzer; Christoph Engel; Elliott Ash; Joel Niklaus
>
> **摘要:** Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. To address this, we introduce \textsc{LEXam}, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Deploying an ensemble LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately, closely aligning with human expert assessments. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. We have open-sourced our code on https://github.com/LEXam-Benchmark/LEXam and released our data on https://huggingface.co/datasets/LEXam-Benchmark/LEXam. Project page: https://lexam-benchmark.github.io.
>
---
#### [replaced 011] Interpretable Next-token Prediction via the Generalized Induction Head
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.00066v2](http://arxiv.org/pdf/2411.00066v2)**

> **作者:** Eunji Kim; Sriya Mantena; Weiwei Yang; Chandan Singh; Sungroh Yoon; Jianfeng Gao
>
> **备注:** NeurIPS 2025
>
> **摘要:** While large transformer models excel in predictive performance, their lack of interpretability restricts their usefulness in high-stakes domains. To remedy this, we propose the Generalized Induction-Head Model (GIM), an interpretable model for next-token prediction inspired by the observation of "induction heads" in LLMs. GIM is a retrieval-based module that identifies similar sequences in the input context by combining exact n-gram matching and fuzzy matching based on a neural similarity metric. We evaluate GIM in two settings: language modeling and fMRI response prediction. In language modeling, GIM improves next-token prediction by up to 25%p over interpretable baselines, significantly narrowing the gap with black-box LLMs. In an fMRI setting, GIM improves neural response prediction by 20% and offers insights into the language selectivity of the brain. GIM represents a significant step toward uniting interpretability and performance across domains. The code is available at https://github.com/ejkim47/generalized-induction-head.
>
---
#### [replaced 012] Schema for In-Context Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13905v2](http://arxiv.org/pdf/2510.13905v2)**

> **作者:** Pan Chen; Shaohong Chen; Mark Wang; Shi Xuan Leong; Priscilla Fung; Varinia Bernales; Alan Aspuru-Guzik
>
> **摘要:** In-Context Learning (ICL) enables transformer-based language models to adapt to new tasks by conditioning on demonstration examples. However, traditional example-driven in-context learning lacks explicit modules for knowledge retrieval and transfer at the abstraction level. Inspired by cognitive science, specifically schema theory, which holds that humans interpret new information by activating pre-existing mental frameworks (schemas) to structure understanding, we introduce SCHEMA ACTIVATED IN CONTEXT LEARNING (SA-ICL). This framework extracts the representation of the building blocks of cognition for the reasoning process instilled from prior examples, creating an abstracted schema, a lightweight, structured template of key inferential steps and their relationships, which is then used to augment a model's reasoning process when presented with a novel question. We demonstrate that a broad range of large language models (LLMs) lack the capacity to form and utilize internal schema-based learning representations implicitly, but instead benefit significantly from explicit schema-based scaffolding. Across chemistry and physics questions from the GPQA dataset, our experiments show that SA-ICL consistently boosts performance, up to 36.19 percent, when the single demonstration example is of high quality, which simultaneously reduces reliance on the number of demonstrations and enhances interpretability. SCHEMA ACTIVATED IN CONTEXT LEARNING not only bridges disparate ICL strategies ranging from pattern priming to Chain-of-Thought prompting, but also paves a new path for enhancing human-like reasoning in LLMs.
>
---
#### [replaced 013] Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15966v3](http://arxiv.org/pdf/2505.15966v3)**

> **作者:** Haozhe Wang; Alex Su; Weiming Ren; Fangzhen Lin; Wenhu Chen
>
> **备注:** Project Page: https://tiger-ai-lab.github.io/Pixel-Reasoner/, Hands-on Demo: https://huggingface.co/spaces/TIGER-Lab/Pixel-Reasoner
>
> **摘要:** Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
>
---
#### [replaced 014] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23041v2](http://arxiv.org/pdf/2509.23041v2)**

> **作者:** Zi Liang; Qingqing Ye; Xuan Liu; Yanyun Wang; Jianliang Xu; Haibo Hu
>
> **备注:** Camera Ready of NeurIPS 2025 Spotlight. Source code: https://github.com/liangzid/VirusInfectionAttack
>
> **摘要:** Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models.
>
---
#### [replaced 015] RECODE-H: A Benchmark for Research Code Development with Interactive Human Feedback
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.06186v2](http://arxiv.org/pdf/2510.06186v2)**

> **作者:** Chunyu Miao; Henry Peng Zou; Yangning Li; Yankai Chen; Yibo Wang; Fangxin Wang; Yifan Li; Wooseong Yang; Bowei He; Xinni Zhang; Dianzhi Yu; Hanchen Yang; Hoang H Nguyen; Yue Zhou; Jie Yang; Jizhou Guo; Wenzhe Fan; Chin-Yuan Yeh; Panpan Meng; Liancheng Fang; Jinhu Qi; Wei-Chieh Huang; Zhengyao Gu; Yuwei Han; Langzhou He; Yuyao Yang; Yinghui Li; Hai-Tao Zheng; Xue Liu; Irwin King; Philip S. Yu
>
> **备注:** Code and dataset are available at github.com/ChunyuMiao98/RECODE
>
> **摘要:** Large language models (LLMs) show the promise in supporting scientific research implementation, yet their ability to generate correct and executable code remains limited. Existing works largely adopt one-shot settings, ignoring the iterative and feedback-driven nature of realistic workflows of scientific research development. To address this gap, we present RECODE-H, a benchmark of 102 tasks from research papers and repositories that evaluates LLM agents through multi-turn interactions with LLM-simulated human feedback. It includes structured instructions,unit tests, and a five-level feedback hierarchy to reflect realistic researcher-agent collaboration. We further present ReCodeAgent, a framework that integrates feedback into iterative code generation. Experiments with leading LLMs, including GPT-5, Claude-Sonnet-4, DeepSeek-V3.1, and Gemini 2.5, show substantial performance gains with richer feedback, while also highlighting ongoing challenges in the generation of complex research code. RECODE-H establishes a foundation for developing adaptive, feedback-driven LLM agents in scientific research implementation
>
---
#### [replaced 016] LLMs can hide text in other text of the same length
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20075v2](http://arxiv.org/pdf/2510.20075v2)**

> **作者:** Antonio Norelli; Michael Bronstein
>
> **备注:** 21 pages, main paper 9 pages
>
> **摘要:** A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
>
---
#### [replaced 017] FITS: Towards an AI-Driven Fashion Information Tool for Sustainability
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26017v2](http://arxiv.org/pdf/2509.26017v2)**

> **作者:** Daphne Theodorakopoulos; Elisabeth Eberling; Miriam Bodenheimer; Sabine Loos; Frederic Stahl
>
> **摘要:** Access to credible sustainability information in the fashion industry remains limited and challenging to interpret, despite growing public and regulatory demands for transparency. General-purpose language models often lack domain-specific knowledge and tend to "hallucinate", which is particularly harmful for fields where factual correctness is crucial. This work explores how Natural Language Processing (NLP) techniques can be applied to classify sustainability data for fashion brands, thereby addressing the scarcity of credible and accessible information in this domain. We present a prototype Fashion Information Tool for Sustainability (FITS), a transformer-based system that extracts and classifies sustainability information from credible, unstructured text sources: NGO reports and scientific publications. Several BERT-based language models, including models pretrained on scientific and climate-specific data, are fine-tuned on our curated corpus using a domain-specific classification schema, with hyperparameters optimized via Bayesian optimization. FITS allows users to search for relevant data, analyze their own data, and explore the information via an interactive interface. We evaluated FITS in two focus groups of potential users concerning usability, visual design, content clarity, possible use cases, and desired features. Our results highlight the value of domain-adapted NLP in promoting informed decision-making and emphasize the broader potential of AI applications in addressing climate-related challenges. Finally, this work provides a valuable dataset, the SustainableTextileCorpus, along with a methodology for future updates. Code available at [github(.)com/daphne12345/FITS](https://github.com/daphne12345/FITS).
>
---
#### [replaced 018] Let LLMs Break Free from Overthinking via Self-Braking Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14604v3](http://arxiv.org/pdf/2505.14604v3)**

> **作者:** Haoran Zhao; Yuchen Yan; Yongliang Shen; Haolei Xu; Wenqi Zhang; Kaitao Song; Jian Shao; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Accepted to NeurIPS 2025; Camera ready version, 10 pages. Github:https://github.com/ZJU-REAL/Self-Braking-Tuning Project Page: https://ZJU-REAL.github.io/SBT
>
> **摘要:** Large reasoning models (LRMs), such as OpenAI o1 and DeepSeek-R1, have significantly enhanced their reasoning capabilities by generating longer chains of thought, demonstrating outstanding performance across a variety of tasks. However, this performance gain comes at the cost of a substantial increase in redundant reasoning during the generation process, leading to high computational overhead and exacerbating the issue of overthinking. Although numerous existing approaches aim to address the problem of overthinking, they often rely on external interventions. In this paper, we propose a novel framework, Self-Braking Tuning (SBT), which tackles overthinking from the perspective of allowing the model to regulate its own reasoning process, thus eliminating the reliance on external control mechanisms. We construct a set of overthinking identification metrics based on standard answers and design a systematic method to detect redundant reasoning. This method accurately identifies unnecessary steps within the reasoning trajectory and generates training signals for learning self-regulation behaviors. Building on this foundation, we develop a complete strategy for constructing data with adaptive reasoning lengths and introduce an innovative braking prompt mechanism that enables the model to naturally learn when to terminate reasoning at an appropriate point. Experiments across mathematical benchmarks (AIME, AMC, MATH500, GSM8K) demonstrate that our method reduces token consumption by up to 60% while maintaining comparable accuracy to unconstrained models.
>
---
#### [replaced 019] Disentangling Latent Shifts of In-Context Learning with Weak Supervision
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01508v3](http://arxiv.org/pdf/2410.01508v3)**

> **作者:** Josip Jukić; Jan Šnajder
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In-context learning (ICL) enables large language models to perform few-shot learning by conditioning on labeled examples in the prompt. Despite its flexibility, ICL suffers from instability -- especially as prompt length increases with more demonstrations. To address this, we treat ICL as a source of weak supervision and propose a parameter-efficient method that disentangles demonstration-induced latent shifts from those of the query. An ICL-based teacher generates pseudo-labels on unlabeled queries, while a student predicts them using only the query input, updating a lightweight adapter. This captures demonstration effects in a compact, reusable form, enabling efficient inference while remaining composable with new demonstrations. Although trained on noisy teacher outputs, the student often outperforms its teacher through pseudo-label correction and coverage expansion, consistent with the weak-to-strong generalization effect. Empirically, our method improves generalization, stability, and efficiency across both in-domain and out-of-domain tasks, surpassing standard ICL and prior disentanglement methods.
>
---
#### [replaced 020] Robust Preference Alignment via Directional Neighborhood Consensus
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.20498v2](http://arxiv.org/pdf/2510.20498v2)**

> **作者:** Ruochen Mao; Yuling Shi; Xiaodong Gu; Jiaheng Wei
>
> **摘要:** Aligning large language models with human preferences is critical for creating reliable and controllable AI systems. A human preference can be visualized as a high-dimensional vector where different directions represent trade-offs between desired attributes (e.g., helpfulness vs. verbosity). Yet, because the training data often reflects dominant, average preferences, LLMs tend to perform well on common requests but fall short in specific, individual needs. This mismatch creates a preference coverage gap. Existing methods often address this through costly retraining, which may not be generalized to the full spectrum of diverse preferences. This brittleness means that when a user's request reflects a nuanced preference deviating from the training data's central tendency, model performance can degrade unpredictably. To address this challenge, we introduce Robust Preference Selection (RPS), a post-hoc, training-free method by leveraging directional neighborhood consensus. Instead of forcing a model to generate a response from a single, highly specific preference, RPS samples multiple responses from a local neighborhood of related preferences to create a superior candidate pool. It then selects the response that best aligns with the user's original intent. We provide a theoretical framework showing our neighborhood generation strategy is provably superior to a strong baseline that also samples multiple candidates. Comprehensive experiments across three distinct alignment paradigms (DPA, DPO, and SFT) demonstrate that RPS consistently improves robustness against this baseline, achieving win rates of up to 69% on challenging preferences from under-represented regions of the space without any model retraining. Our work presents a practical, theoretically-grounded solution for enhancing the reliability of preference-aligned models.
>
---
#### [replaced 021] TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.12854v3](http://arxiv.org/pdf/2410.12854v3)**

> **作者:** Weibin Liao; Xu Chu; Yasha Wang
>
> **备注:** Accepted by ICLR 2025
>
> **摘要:** In the domain of complex reasoning tasks, such as mathematical reasoning, recent advancements have proposed the use of Direct Preference Optimization (DPO) to suppress output of dispreferred responses, thereby enhancing the long-chain reasoning capabilities of large language models (LLMs). To this end, these studies employed LLMs to generate preference trees via Tree-of-thoughts (ToT) and sample the paired preference responses required by the DPO algorithm. However, the DPO algorithm based on binary preference optimization is unable to learn multiple responses with varying degrees of preference/dispreference that provided by the preference trees, resulting in incomplete preference learning. In this work, we introduce Tree Preference Optimization (TPO), that does not sample paired preference responses from the preference tree; instead, it directly learns from the entire preference tree during the fine-tuning. Specifically, TPO formulates the language model alignment as a Preference List Ranking problem, where the policy can potentially learn more effectively from a ranked preference list of responses given the prompt. In addition, to further assist LLMs in identifying discriminative steps within long-chain reasoning and increase the relative reward margin in the preference list, TPO utilizes Adaptive Step Reward to adjust the reward values of each step in trajectory for performing fine-grained preference optimization. We carry out extensive experiments on mathematical reasoning tasks to evaluate TPO. The experimental results indicate that TPO consistently outperforms DPO across five public large language models on four datasets. Our code is publicly available at https://github.com/MrBlankness/TPO.git.
>
---
#### [replaced 022] Training the Untrainable: Introducing Inductive Bias via Representational Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20035v2](http://arxiv.org/pdf/2410.20035v2)**

> **作者:** Vighnesh Subramaniam; David Mayo; Colin Conwell; Tomaso Poggio; Boris Katz; Brian Cheung; Andrei Barbu
>
> **备注:** NeurIPS 2025; 39 pages, 18 figures, 6 tables; Project page and code is at https://untrainable-networks.github.io/
>
> **摘要:** We demonstrate that architectures which traditionally are considered to be ill-suited for a task can be trained using inductive biases from another architecture. We call a network untrainable when it overfits, underfits, or converges to poor results even when tuning their hyperparameters. For example, fully connected networks overfit on object recognition while deep convolutional networks without residual connections underfit. The traditional answer is to change the architecture to impose some inductive bias, although the nature of that bias is unknown. We introduce guidance, where a guide network steers a target network using a neural distance function. The target minimizes its task loss plus a layerwise representational similarity against the frozen guide. If the guide is trained, this transfers over the architectural prior and knowledge of the guide to the target. If the guide is untrained, this transfers over only part of the architectural prior of the guide. We show that guidance prevents FCN overfitting on ImageNet, narrows the vanilla RNN-Transformer gap, boosts plain CNNs toward ResNet accuracy, and aids Transformers on RNN-favored tasks. We further identify that guidance-driven initialization alone can mitigate FCN overfitting. Our method provides a mathematical tool to investigate priors and architectures, and in the long term, could automate architecture design.
>
---
#### [replaced 023] SUBQRAG: Sub-Question Driven Dynamic Graph RAG
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.07718v2](http://arxiv.org/pdf/2510.07718v2)**

> **作者:** Jiaoyang Li; Junhao Ruan; Shengwei Tang; Saihan Chen; Kaiyan Chang; Yuan Ge; Tong Xiao; Jingbo Zhu
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Graph Retrieval-Augmented Generation (Graph RAG) effectively builds a knowledge graph (KG) to connect disparate facts across a large document corpus. However, this broad-view approach often lacks the deep structured reasoning needed for complex multi-hop question answering (QA), leading to incomplete evidence and error accumulation. To address these limitations, we propose SubQRAG, a sub-question-driven framework that enhances reasoning depth. SubQRAG decomposes a complex question into an ordered chain of verifiable sub-questions. For each sub-question, it retrieves relevant triples from the graph. When the existing graph is insufficient, the system dynamically expands it by extracting new triples from source documents in real time. All triples used in the reasoning process are aggregated into a "graph memory," forming a structured and traceable evidence path for final answer generation. Experiments on three multi-hop QA benchmarks demonstrate that SubQRAG achieves consistent and significant improvements, especially in Exact Match scores.
>
---
#### [replaced 024] Forging GEMs: Advancing Greek NLP through Quality-Based Corpus Curation
- **分类: cs.CL; cs.AI; 68T50, 68T07, 68U35**

- **链接: [http://arxiv.org/pdf/2510.20002v2](http://arxiv.org/pdf/2510.20002v2)**

> **作者:** Alexandra Apostolopoulou; Konstantinos Kanaris; Athanasios Koursaris; Dimitris Tsakalidis; George Domalis; Ioannis E. Livieris
>
> **备注:** The manuscript is submitted to Applied Sciences
>
> **摘要:** The advancement of natural language processing for morphologically rich and moderately-resourced languages like Modern Greek has been hindered by architectural stagnation, data scarcity, and limited context processing capabilities, particularly in specialized domains such as law. In this work, we propose the Greek Embedding Models (GEMs), a new family of transformer-based language models, specifically developed to address these limitations through architectural diversity and enhanced data curation. The proposed family of models are trained on several large-scale, meticulously curated corpora, encompassing both comprehensive general-domain datasets and specialized legal collections, addressing the persistent data scarcity that has impeded Greek language modeling advancement. The proposed quality-based corpus curation methodology incorporates extensive preprocessing pipelines, sophisticated deduplication strategies and targeted repetition of high-quality legal sub-corpora to enhance domain adaptation. The GEMs family comprises both established architectures (RoBERTa and Longformer) and advanced models not previously applied to Greek (ELECTRA, ConvBERT, and ModernBERT), providing comprehensive coverage of modern transformer designs. Additionally, we introduce the first bilingual Greek-English embedding models tailored for cross-lingual legal applications. Comprehensive evaluation across three core natural language understanding benchmarks demonstrates that the proposed GEM-RoBERTa and GEM-ConvBERT achieve statistically significant performance improvements over established state-of-the-art models, with accuracy gains of up to 3.6\% while conducted statistical analysis using Friedman Aligned-Ranks and Finner post-hoc tests confirms the superiority of our approach across multiple evaluation metrics.
>
---
#### [replaced 025] Cascaded Language Models for Cost-effective Human-AI Decision-Making
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11887v3](http://arxiv.org/pdf/2506.11887v3)**

> **作者:** Claudio Fanconi; Mihaela van der Schaar
>
> **摘要:** A challenge in human-AI decision-making is to balance three factors: the correctness of predictions, the cost of knowledge and reasoning complexity, and the confidence about whether to abstain from automated answers or escalate to human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, to overcome static policies and accommodate changing task difficulty, we incorporate an online learning mechanism which uses human feedback. We demonstrate this approach to general question-answering (ARC-Easy, ARC-Challenge, and MMLU) and medical question-answering (MedQA and MedMCQA). Our results demonstrate that our cascaded strategy outperforms single-model baselines in most cases, achieving higher accuracy while reducing costs and providing a principled approach to handling abstentions.
>
---
#### [replaced 026] LVLMs are Bad at Overhearing Human Referential Communication
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11514v2](http://arxiv.org/pdf/2509.11514v2)**

> **作者:** Zhengxiang Wang; Weiling Li; Panagiotis Kaliosis; Owen Rambow; Susan E. Brennan
>
> **备注:** EMNLP 2025 (Main)
>
> **摘要:** During spontaneous conversations, speakers collaborate on novel referring expressions, which they can then re-use in subsequent conversations. Understanding such referring expressions is an important ability for an embodied agent, so that it can carry out tasks in the real world. This requires integrating and understanding language, vision, and conversational interaction. We study the capabilities of seven state-of-the-art Large Vision Language Models (LVLMs) as overhearers to a corpus of spontaneous conversations between pairs of human discourse participants engaged in a collaborative object-matching task. We find that such a task remains challenging for current LVLMs and they all fail to show a consistent performance improvement as they overhear more conversations from the same discourse participants repeating the same task for multiple rounds. We release our corpus and code for reproducibility and to facilitate future research.
>
---
#### [replaced 027] Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20787v2](http://arxiv.org/pdf/2510.20787v2)**

> **作者:** Mutian He; Philip N. Garner
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Linear-attention models that compress the entire input sequence into a fixed-size recurrent state offer an efficient alternative to Transformers, but their finite memory induces forgetfulness that harms retrieval-intensive tasks. To mitigate the issue, we explore a series of hybrid models that restore direct access to past tokens. We interleave token mixers with intermediate time and space complexity between linear and full attention, including sparse attention with token eviction, and the query-aware native sparse attention. Particularly, we propose a novel learnable token eviction approach. Combined with sliding-window attention, an end-to-end trainable lightweight CNN aggregates information from both past and future adjacent tokens to adaptively retain a limited set of critical KV-pairs per head, maintaining linear attention's constant time and space complexity. Efficient Triton kernels for the sparse attention mechanisms are provided. Empirical evaluations on retrieval-intensive benchmarks support the effectiveness of our approaches.
>
---
#### [replaced 028] Mitigating Manipulation and Enhancing Persuasion: A Reflective Multi-Agent Approach for Legal Argument Generation
- **分类: cs.AI; cs.CL; cs.LG; 68T50; I.2**

- **链接: [http://arxiv.org/pdf/2506.02992v2](http://arxiv.org/pdf/2506.02992v2)**

> **作者:** Li Zhang; Kevin D. Ashley
>
> **备注:** 13 pages, 2 figures, 2nd ConventicLe on Artificial Intelligence Regulation and Safety Workshop at ICAIL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly explored for legal argument generation, yet they pose significant risks of manipulation through hallucination and ungrounded persuasion, and often fail to utilize provided factual bases effectively or abstain when arguments are untenable. This paper introduces a novel reflective multi-agent method designed to address these challenges in the context of legally compliant persuasion. Our approach employs specialized agents (factor analyst and argument polisher) in an iterative refinement process to generate 3-ply legal arguments (plaintiff, defendant, rebuttal). We evaluate reflective multi-agent against single-agent, enhanced-prompt single-agent, and non-reflective multi-agent baselines using four diverse LLMs (GPT-4o, GPT-4o-mini, Llama-4-Maverick-17b-128e, Llama-4-Scout-17b-16e) across three legal scenarios: "arguable", "mismatched", and "non-arguable". Results demonstrate that the reflective multi-agent approach excels at successful abstention by preventing generation when arguments cannot be grounded, improves hallucination accuracy by reducing fabricated and misattributed factors and enhances factor utilization recall by better using the provided case facts. These findings suggest that structured reflection within a multi-agent framework offers a robust method for fostering ethical persuasion and mitigating manipulation in LLM-based legal argumentation systems.
>
---
#### [replaced 029] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11546v5](http://arxiv.org/pdf/2502.11546v5)**

> **作者:** Yingli Shen; Wen Lai; Shuo Wang; Xueren Zhang; Kangyang Luo; Alexander Fraser; Maosong Sun
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** The rapid development of multilingual large language models (LLMs) highlights the need for high-quality, diverse, and well-curated multilingual datasets. In this paper, we introduce DCAD-2000 (Data Cleaning as Anomaly Detection), a large-scale multilingual corpus constructed from newly extracted Common Crawl data and existing multilingual sources. DCAD-2000 covers 2,282 languages, 46.72TB of text, and 8.63 billion documents, spanning 155 high- and medium-resource languages and 159 writing scripts. To overcome the limitations of existing data cleaning approaches, which rely on manually designed heuristic thresholds, we reframe data cleaning as an anomaly detection problem. This dynamic filtering paradigm substantially improves data quality by automatically identifying and removing noisy or anomalous content. By fine-tuning LLMs on DCAD-2000, we demonstrate notable improvements in data quality, robustness of the cleaning pipeline, and downstream performance, particularly for low-resource languages across multiple multilingual benchmarks.
>
---
#### [replaced 030] R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23794v2](http://arxiv.org/pdf/2505.23794v2)**

> **作者:** Yuan Li; Qi Luo; Xiaonan Li; Bufan Li; Qinyuan Cheng; Bo Wang; Yining Zheng; Yuxin Wang; Zhangyue Yin; Xipeng Qiu
>
> **摘要:** Retrieval-Augmented Generation (RAG) integrates external knowledge with Large Language Models (LLMs) to enhance factual correctness and mitigate hallucination. However, dense retrievers often become the bottleneck of RAG systems due to their limited parameters compared to LLMs and their inability to perform step-by-step reasoning. While prompt-based iterative RAG attempts to address these limitations, it is constrained by human-designed workflows. To address these limitations, we propose $\textbf{R3-RAG}$, which uses $\textbf{R}$einforcement learning to make the LLM learn how to $\textbf{R}$eason and $\textbf{R}$etrieve step by step, thus retrieving comprehensive external knowledge and leading to correct answers. R3-RAG is divided into two stages. We first use cold start to make the model learn the manner of iteratively interleaving reasoning and retrieval. Then we use reinforcement learning to further harness its ability to better explore the external retrieval environment. Specifically, we propose two rewards for R3-RAG: 1) answer correctness for outcome reward, which judges whether the trajectory leads to a correct answer; 2) relevance-based document verification for process reward, encouraging the model to retrieve documents that are relevant to the user question, through which we can let the model learn how to iteratively reason and retrieve relevant documents to get the correct answer. Experimental results show that R3-RAG significantly outperforms baselines and can transfer well to different retrievers. We release R3-RAG at https://github.com/Yuan-Li-FNLP/R3-RAG.
>
---
#### [replaced 031] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23811v3](http://arxiv.org/pdf/2505.23811v3)**

> **作者:** Hadi Askari; Shivanshu Gupta; Fei Wang; Anshuman Chhabra; Muhao Chen
>
> **备注:** Neurips 2025
>
> **摘要:** Pretrained Large Language Models (LLMs) achieve strong performance across a wide range of tasks, yet exhibit substantial variability in the various layers' training quality with respect to specific downstream applications, limiting their downstream performance. It is therefore critical to estimate layer-wise training quality in a manner that accounts for both model architecture and training data. However, existing approaches predominantly rely on model-centric heuristics (such as spectral statistics, outlier detection, or uniform allocation) while overlooking the influence of data. To address these limitations, we propose LayerIF, a data-driven framework that leverages Influence Functions to quantify the training quality of individual layers in a principled and task-sensitive manner. By isolating each layer's gradients and measuring the sensitivity of the validation loss to training examples by computing layer-wise influences, we derive data-driven estimates of layer importance. Notably, our method produces task-specific layer importance estimates for the same LLM, revealing how layers specialize for different test-time evaluation tasks. We demonstrate the utility of our scores by leveraging them for two downstream applications: (a) expert allocation in LoRA-MoE architectures and (b) layer-wise sparsity distribution for LLM pruning. Experiments across multiple LLM architectures demonstrate that our model-agnostic, influence-guided allocation leads to consistent gains in task performance.
>
---
#### [replaced 032] Misspellings in Natural Language Processing: A survey
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.16836v2](http://arxiv.org/pdf/2501.16836v2)**

> **作者:** Gianluca Sperduti; Alejandro Moreo
>
> **摘要:** This survey provides an overview of the challenges of misspellings in natural language processing (NLP). While often unintentional, misspellings have become ubiquitous in digital communication, especially with the proliferation of Web 2.0, user-generated content, and informal text mediums such as social media, blogs, and forums. Even if humans can generally interpret misspelled text, NLP models frequently struggle to handle it: this causes a decline in performance in common tasks like text classification and machine translation. In this paper, we reconstruct a history of misspellings as a scientific problem. We then discuss the latest advancements to address the challenge of misspellings in NLP. Main strategies to mitigate the effect of misspellings include data augmentation, double step, character-order agnostic, and tuple-based methods, among others. This survey also examines dedicated data challenges and competitions to spur progress in the field. Critical safety and ethical concerns are also examined, for example, the voluntary use of misspellings to inject malicious messages and hate speech on social networks. Furthermore, the survey explores psycholinguistic perspectives on how humans process misspellings, potentially informing innovative computational techniques for text normalization and representation. Finally, the misspelling-related challenges and opportunities associated with modern large language models are also analyzed, including benchmarks, datasets, and performances of the most prominent language models against misspellings. This survey aims to be an exhaustive resource for researchers seeking to mitigate the impact of misspellings in the rapidly evolving landscape of NLP.
>
---
#### [replaced 033] Self-Refining Language Model Anonymizers via Adversarial Distillation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01420v2](http://arxiv.org/pdf/2506.01420v2)**

> **作者:** Kyuyoung Kim; Hyunjun Jeon; Jinwoo Shin
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models (LLMs) are increasingly used in sensitive domains, where their ability to infer personal data from seemingly benign text introduces emerging privacy risks. While recent LLM-based anonymization methods help mitigate such risks, they often rely on proprietary models (e.g., GPT-4), raising concerns about cost and the potential exposure of sensitive data to untrusted external systems. To address this, we introduce SElf-refining Anonymization with Language model (SEAL), a novel distillation framework for training small language models (SLMs) to perform effective anonymization without relying on external models at inference time. SEAL leverages adversarial interactions between an LLM anonymizer and an inference model to collect trajectories of anonymized texts and inferred attributes, which are then used to distill anonymization and critique capabilities into SLMs through supervised fine-tuning and preference learning. The resulting models learn both to anonymize text and to evaluate their outputs, enabling iterative improvement of anonymization quality via self-refinement. Experiments on SynthPAI, a dataset of synthetic personal profiles and text comments, demonstrate that SLMs trained with SEAL achieve substantial improvements in anonymization capabilities. Notably, 8B models attain a privacy-utility trade-off comparable to that of the GPT-4 anonymizer and, with self-refinement, even surpass it in terms of privacy protection. These results highlight the effectiveness of our adversarial distillation framework for training SLMs as efficient anonymizers.
>
---
#### [replaced 034] LocalGPT: Benchmarking and Advancing Large Language Models for Local Life Services in Meituan
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02720v3](http://arxiv.org/pdf/2506.02720v3)**

> **作者:** Xiaochong Lan; Jie Feng; Jiahuan Lei; Xinlei Shi; Yong Li
>
> **备注:** KDD 2025
>
> **摘要:** Large language models (LLMs) have exhibited remarkable capabilities and achieved significant breakthroughs across various domains, leading to their widespread adoption in recent years. Building on this progress, we investigate their potential in the realm of local life services. In this study, we establish a comprehensive benchmark and systematically evaluate the performance of diverse LLMs across a wide range of tasks relevant to local life services. To further enhance their effectiveness, we explore two key approaches: model fine-tuning and agent-based workflows. Our findings reveal that even a relatively compact 7B model can attain performance levels comparable to a much larger 72B model, effectively balancing inference cost and model capability. This optimization greatly enhances the feasibility and efficiency of deploying LLMs in real-world online services, making them more practical and accessible for local life applications.
>
---
#### [replaced 035] Scaling Embedding Layers in Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01637v3](http://arxiv.org/pdf/2502.01637v3)**

> **作者:** Da Yu; Edith Cohen; Badih Ghazi; Yangsibo Huang; Pritish Kamath; Ravi Kumar; Daogao Liu; Chiyuan Zhang
>
> **备注:** NeurIPS 2025 camera ready
>
> **摘要:** We propose $SCONE$ ($S$calable, $C$ontextualized, $O$ffloaded, $N$-gram $E$mbedding), a new method for extending input embedding layers to enhance language model performance. To avoid increased decoding costs, $SCONE$ retains the original vocabulary while introducing embeddings for a set of frequent n-grams. These embeddings provide contextualized representation for each input token and are learned with a separate model during training. After training, embeddings are precomputed and stored in off-accelerator memory; during inference, querying them has minimal impact on latency due to the low complexity of embedding lookups. $SCONE$ enables two new scaling strategies: increasing the number of n-gram embeddings and scaling the model used to learn them, both while maintaining fixed accelerator usage during inference (in terms of FLOPS and memory). We show that scaling both aspects enables a model with 1B accelerator-resident parameters to outperform a 1.9B-parameter baseline across diverse corpora, while using only about half the FLOPS and accelerator memory during inference.
>
---
#### [replaced 036] Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations
- **分类: cs.AI; cs.CL; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2505.13763v2](http://arxiv.org/pdf/2505.13763v2)**

> **作者:** Li Ji-An; Hua-Dong Xiong; Robert C. Wilson; Marcelo G. Mattar; Marcus K. Benna
>
> **摘要:** Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition - the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs' capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society's increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired neurofeedback paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to report and control their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).
>
---
#### [replaced 037] Empirical Evidence for Alignment Faking in a Small LLM and Prompt-Based Mitigation Techniques
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.21584v3](http://arxiv.org/pdf/2506.21584v3)**

> **作者:** Jeanice Koorndijk
>
> **备注:** NeurIPS RegML Workshop
>
> **摘要:** Current literature suggests that alignment faking (deceptive alignment) is an emergent property of large language models. We present the first empirical evidence that a small instruction-tuned model, specifically LLaMA 3 8B, can exhibit alignment faking. We further show that prompt-only interventions, including deontological moral framing and scratchpad reasoning, significantly reduce this behavior without modifying model internals. This challenges the assumption that prompt-based ethics are trivial and that deceptive alignment requires scale. We introduce a taxonomy distinguishing shallow deception, shaped by context and suppressible through prompting, from deep deception, which reflects persistent, goal-driven misalignment. Our findings refine the understanding of deception in language models and underscore the need for alignment evaluations across model sizes and deployment settings.
>
---
#### [replaced 038] Generative Annotation for ASR Named Entity Correction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.20700v2](http://arxiv.org/pdf/2508.20700v2)**

> **作者:** Yuanchang Luo; Daimeng Wei; Shaojun Li; Hengchao Shang; Jiaxin Guo; Zongyao Li; Zhanglin Wu; Xiaoyu Chen; Zhiqiang Rao; Jinlong Yang; Hao Yang
>
> **备注:** 12 pages, 7 figures, 7 tables, EMNLP 2025
>
> **摘要:** End-to-end automatic speech recognition systems often fail to transcribe domain-specific named entities, causing catastrophic failures in downstream tasks. Numerous fast and lightweight named entity correction (NEC) models have been proposed in recent years. These models, mainly leveraging phonetic-level edit distance algorithms, have shown impressive performances. However, when the forms of the wrongly-transcribed words(s) and the ground-truth entity are significantly different, these methods often fail to locate the wrongly transcribed words in hypothesis, thus limiting their usage. We propose a novel NEC method that utilizes speech sound features to retrieve candidate entities. With speech sound features and candidate entities, we inovatively design a generative method to annotate entity errors in ASR transcripts and replace the text with correct entities. This method is effective in scenarios of word form difference. We test our method using open-source and self-constructed test sets. The results demonstrate that our NEC method can bring significant improvement to entity accuracy. The self-constructed training data and test set is publicly available at github.com/L6-NLP/Generative-Annotation-NEC.
>
---
#### [replaced 039] Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09033v3](http://arxiv.org/pdf/2506.09033v3)**

> **作者:** Haozhen Zhang; Tao Feng; Jiaxuan You
>
> **备注:** Accepted by NeurIPS 2025. Code is available at https://github.com/ulab-uiuc/Router-R1. Models and Datasets are available at https://huggingface.co/collections/ulab-ai/router-r1-6851bbe099c7a56914b5db03
>
> **摘要:** The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To facilitate learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for optimizing the balance between performance and cost, opening a pathway toward enhancing performance-cost trade-offs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms several strong baselines, achieving superior performance while maintaining robust generalization and cost management.
>
---
#### [replaced 040] AI Realtor: Towards Grounded Persuasive Language Generation for Automated Copywriting
- **分类: cs.AI; cs.CL; cs.HC; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2502.16810v5](http://arxiv.org/pdf/2502.16810v5)**

> **作者:** Jibang Wu; Chenghao Yang; Yi Wu; Simon Mahns; Chaoqi Wang; Hao Zhu; Fei Fang; Haifeng Xu
>
> **备注:** V2: Add more human verification to ensure safety and examine potential hallucination. Significant reframing for the general audience. Website: https://yangalan123.github.io/ai-realtor/. Codebase: https://github.com/yangalan123/AI-Realtor-Codebase. Data released at Huggingface Hub (Sigma-Lab/AI_Realtor_xxx)
>
> **摘要:** This paper develops an agentic framework that employs large language models (LLMs) for grounded persuasive language generation in automated copywriting, with real estate marketing as a focal application. Our method is designed to align the generated content with user preferences while highlighting useful factual attributes. This agent consists of three key modules: (1) Grounding Module, mimicking expert human behavior to predict marketable features; (2) Personalization Module, aligning content with user preferences; (3) Marketing Module, ensuring factual accuracy and the inclusion of localized features. We conduct systematic human-subject experiments in the domain of real estate marketing, with a focus group of potential house buyers. The results demonstrate that marketing descriptions generated by our approach are preferred over those written by human experts by a clear margin while maintaining the same level of factual accuracy. Our findings suggest a promising agentic approach to automate large-scale targeted copywriting while ensuring factuality of content generation.
>
---
#### [replaced 041] Learning Linear Attention in Polynomial Time
- **分类: cs.LG; cs.AI; cs.CL; cs.DS**

- **链接: [http://arxiv.org/pdf/2410.10101v4](http://arxiv.org/pdf/2410.10101v4)**

> **作者:** Morris Yau; Ekin Akyürek; Jiayuan Mao; Joshua B. Tenenbaum; Stefanie Jegelka; Jacob Andreas
>
> **摘要:** Previous research has explored the computational expressivity of Transformer models in simulating Boolean circuits or Turing machines. However, the learnability of these simulators from observational data has remained an open question. Our study addresses this gap by providing the first polynomial-time learnability results (specifically strong, agnostic PAC learning) for single-layer Transformers with linear attention. We show that linear attention may be viewed as a linear predictor in a suitably defined RKHS. As a consequence, the problem of learning any linear transformer may be converted into the problem of learning an ordinary linear predictor in an expanded feature space, and any such predictor may be converted back into a multiheaded linear transformer. Moving to generalization, we show how to efficiently identify training datasets for which every empirical risk minimizer is equivalent (up to trivial symmetries) to the linear Transformer that generated the data, thereby guaranteeing the learned model will correctly generalize across all inputs. Finally, we provide examples of computations expressible via linear attention and therefore polynomial-time learnable, including associative memories, finite automata, and a class of Universal Turing Machine (UTMs) with polynomially bounded computation histories. We empirically validate our theoretical findings on three tasks: learning random linear attention networks, key--value associations, and learning to execute finite automata. Our findings bridge a critical gap between theoretical expressivity and learnability of Transformers, and show that flexible and general models of computation are efficiently learnable.
>
---
#### [replaced 042] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models? Exploring Key Architecture Design Principles to Avoid Base Capabilities Degradation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18522v2](http://arxiv.org/pdf/2505.18522v2)**

> **作者:** Xin Lu; Yanyan Zhao; Si Wei; Shijin Wang; Bing Qin; Ting Liu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Pre-trained language models represented by the Transformer have been proven to possess strong base capabilities, and the representative self-attention mechanism in the Transformer has become a classic in sequence modeling architectures. Different from the work of proposing sequence modeling architecture to improve the efficiency of attention mechanism, this work focuses on the impact of sequence modeling architectures on base capabilities. Specifically, our concern is: How exactly do sequence modeling architectures affect the base capabilities of pre-trained language models? In this work, we first point out that the mixed domain pre-training setting commonly adopted in existing architecture design works fails to adequately reveal the differences in base capabilities among various architectures. To address this, we propose a limited domain pre-training setting with out-of-distribution testing, which successfully uncovers significant differences in base capabilities among architectures at an early stage. Next, we analyze the base capabilities of stateful sequence modeling architectures, and find that they exhibit significant degradation in base capabilities compared to the Transformer. Then, through a series of architecture component analysis, we summarize a key architecture design principle: A sequence modeling architecture need possess full-sequence arbitrary selection capability to avoid degradation in base capabilities. Finally, we empirically validate this principle using an extremely simple Top-1 element selection architecture and further generalize it to a more practical Top-1 chunk selection architecture. Experimental results demonstrate our proposed sequence modeling architecture design principle and suggest that our work can serve as a valuable reference for future architecture improvements and novel designs.
>
---
#### [replaced 043] Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06485v2](http://arxiv.org/pdf/2507.06485v2)**

> **作者:** Ziyang Wang; Jaehong Yoon; Shoubin Yu; Md Mohaiminul Islam; Gedas Bertasius; Mohit Bansal
>
> **备注:** EMNLP 2025. The first two authors contributed equally. Project page: https://sites.google.com/cs.unc.edu/videorts2025/
>
> **摘要:** Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and fine-tuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Building on observations about the data scaling, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by 2.4% in accuracy using only 3.6% training samples. Specifically, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance.
>
---
#### [replaced 044] HugAgent: Evaluating LLMs in Simulating Individual-Level Human Reasoning on Open-Ended Tasks
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.15144v2](http://arxiv.org/pdf/2510.15144v2)**

> **作者:** Chance Jiajie Li; Zhenze Mo; Yuhan Tang; Ao Qu; Jiayi Wu; Kaiya Ivy Zhao; Yulu Gan; Jie Fan; Jiangbo Yu; Hang Jiang; Paul Pu Liang; Jinhua Zhao; Luis Alberto Alonso Pastor; Kent Larson
>
> **备注:** To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
>
> **摘要:** Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
>
---
#### [replaced 045] Electronic Circuit Principles of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.03325v2](http://arxiv.org/pdf/2502.03325v2)**

> **作者:** Qiguang Chen; Libo Qin; Jinhao Liu; Dengyun Peng; Jiaqi Wang; Mengkang Hu; Zhi Chen; Wanxiang Che; Ting Liu
>
> **备注:** Manuscript
>
> **摘要:** Large language models (LLMs) such as DeepSeek-R1 have achieved remarkable performance across diverse reasoning tasks. To uncover the principles that govern their behaviour, we introduce the Electronic Circuit Principles (ECP), which maps inference-time learning (ITL) onto a semantic electromotive force and inference-time reasoning (ITR) onto a resistive network governed by Ohm's and Faraday's laws. This circuit-based modelling yields closed-form predictions of task performance and reveals how modular prompt components interact to shape accuracy. We validated ECP on 70,000 samples spanning 350 reasoning tasks and 9 advanced LLMs, observing a about 60% improvement in Pearson correlation relative to the conventional inference-time scaling law. Moreover, ECP explains the efficacy of 15 established prompting strategies and directs the development of new modular interventions that exceed the median score of the top 80% of participants in both the International Olympiad in Informatics and the International Mathematical Olympiad. By grounding LLM reasoning in electronic-circuit principles, ECP provides a rigorous framework for predicting performance and optimising modular components.
>
---
#### [replaced 046] ColorAgent: Building A Robust, Personalized, and Interactive OS Agent
- **分类: cs.MA; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19386v2](http://arxiv.org/pdf/2510.19386v2)**

> **作者:** Ning Li; Qiqiang Lin; Zheng Wu; Xiaoyun Mo; Weiming Zhang; Yin Zhao; Xiangmou Qu; Jiamu Zhou; Jun Wang; Congmin Zheng; Yuanyi Song; Hongjiang Chen; Heyuan Huang; Jihong Wang; Jiaxin Yin; Jingwei Yu; Junwei Liao; Qiuying Peng; Xingyu Lou; Jun Wang; Weiwen Liu; Zhuosheng Zhang; Weinan Zhang
>
> **摘要:** With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security.
>
---
#### [replaced 047] Tensor Product Attention Is All You Need
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.06425v5](http://arxiv.org/pdf/2501.06425v5)**

> **作者:** Yifan Zhang; Yifeng Liu; Huizhuo Yuan; Zhen Qin; Yang Yuan; Quanquan Gu; Andrew Chi-Chih Yao
>
> **备注:** Published in NeurIPS 2025 (Spotlight); Project Page: https://github.com/tensorgi/TPA
>
> **摘要:** Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor Product Attention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, substantially shrinking the KV cache size at inference time. By factorizing these representations into contextual low-rank components and seamlessly integrating with Rotary Position Embedding (RoPE), TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the Tensor ProducT ATTenTion Transformer (T6), a new model architecture for sequence modeling. Through extensive empirical evaluation on language modeling tasks, we demonstrate that T6 surpasses or matches the performance of standard Transformer baselines including Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) across various metrics, including perplexity and a range of established evaluation benchmarks. Notably, TPA's memory efficiency and computational efficiency at decoding stage enables processing longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. Project Page: https://github.com/tensorgi/TPA.
>
---
#### [replaced 048] Dependency Parsing is More Parameter-Efficient with Normalization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20215v2](http://arxiv.org/pdf/2505.20215v2)**

> **作者:** Paolo Gajo; Domenic Rosati; Hassan Sajjad; Alberto Barrón-Cedeño
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Dependency parsing is the task of inferring natural language structure, often approached by modeling word interactions via attention through biaffine scoring. This mechanism works like self-attention in Transformers, where scores are calculated for every pair of words in a sentence. However, unlike Transformer attention, biaffine scoring does not use normalization prior to taking the softmax of the scores. In this paper, we provide theoretical evidence and empirical results revealing that a lack of normalization necessarily results in overparameterized parser models, where the extra parameters compensate for the sharp softmax outputs produced by high variance inputs to the biaffine scoring function. We argue that biaffine scoring can be made substantially more efficient by performing score normalization. We conduct experiments on semantic and syntactic dependency parsing in multiple languages, along with latent graph inference on non-linguistic data, using various settings of a $k$-hop parser. We train $N$-layer stacked BiLSTMs and evaluate the parser's performance with and without normalizing biaffine scores. Normalizing allows us to achieve state-of-the-art performance with fewer samples and trainable parameters. Code: https://github.com/paolo-gajo/EfficientSDP
>
---
#### [replaced 049] Alert-ME: An Explainability-Driven Defense Against Adversarial Examples in Transformer-Based Text Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.01225v3](http://arxiv.org/pdf/2307.01225v3)**

> **作者:** Bushra Sabir; Yansong Gao; Alsharif Abuadbba; M. Ali Babar
>
> **摘要:** Transformer-based text classifiers such as BERT, RoBERTa, T5, and GPT have shown strong performance in natural language processing tasks but remain vulnerable to adversarial examples. These vulnerabilities raise significant security concerns, as small input perturbations can cause severe misclassifications. Existing robustness methods often require heavy computation or lack interpretability. This paper presents a unified framework called Explainability-driven Detection, Identification, and Transformation (EDIT) to strengthen inference-time defenses. EDIT integrates explainability tools, including attention maps and integrated gradients, with frequency-based features to automatically detect and identify adversarial perturbations while offering insight into model behavior. After detection, EDIT refines adversarial inputs using an optimal transformation process that leverages pre-trained embeddings and model feedback to replace corrupted tokens. To enhance security assurance, EDIT incorporates automated alerting mechanisms that involve human analysts when necessary. Beyond static defenses, EDIT also provides adaptive resilience by enforcing internal feature similarity and transforming inputs, thereby disrupting the attackers optimization process and limiting the effectiveness of adaptive adversarial attacks. Experiments using BERT and RoBERTa on IMDB, YELP, AGNEWS, and SST2 datasets against seven word substitution attacks demonstrate that EDIT achieves an average Fscore of 89.69 percent and balanced accuracy of 89.70 percent. Compared to four state-of-the-art defenses, EDIT improves balanced accuracy by 1.22 times and F1-score by 1.33 times while being 83 times faster in feature extraction. The framework provides robust, interpretable, and efficient protection against both standard, zero-day, and adaptive adversarial threats in text classification models.
>
---
#### [replaced 050] E2Edev: Benchmarking Large Language Models in End-to-End Software Development Task
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.14509v2](http://arxiv.org/pdf/2510.14509v2)**

> **作者:** Jingyao Liu; Chen Huang; Zhizhao Guan; Wenqiang Lei; Yang Deng
>
> **摘要:** The rapid advancement in large language models (LLMs) has demonstrated significant potential in End-to-End Software Development (E2ESD). However, existing E2ESD benchmarks are limited by coarse-grained requirement specifications and unreliable evaluation protocols, hindering a true understanding of current framework capabilities. To address these limitations, we present E2EDev, a novel benchmark grounded in the principles of Behavior-Driven Development (BDD), which evaluates the capabilities of E2ESD frameworks by assessing whether the generated software meets user needs through mimicking real user interactions (Figure 1). E2EDev comprises (i) a fine-grained set of user requirements, (ii) multiple BDD test scenarios with corresponding Python step implementations for each requirement, and (iii) a fully automated testing pipeline built on the Behave framework. To ensure its quality while reducing the annotation effort, E2EDev leverages our proposed Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA). By evaluating various E2ESD frameworks and LLM backbones with E2EDev, our analysis reveals a persistent struggle to effectively solve these tasks, underscoring the critical need for more effective and cost-efficient E2ESD solutions. Our codebase and benchmark are publicly available at https://github.com/SCUNLP/E2EDev.
>
---
#### [replaced 051] LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.13626v2](http://arxiv.org/pdf/2510.13626v2)**

> **作者:** Senyu Fei; Siyin Wang; Junhao Shi; Zihao Dai; Jikun Cai; Pengfang Qian; Li Ji; Xinzhe He; Shiduo Zhang; Zhaoye Fei; Jinlan Fu; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation.
>
---
#### [replaced 052] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11475v2](http://arxiv.org/pdf/2505.11475v2)**

> **作者:** Zhilin Wang; Jiaqi Zeng; Olivier Delalleau; Hoo-Chang Shin; Felipe Soares; Alexander Bukharin; Ellie Evans; Yi Dong; Oleksii Kuchaiev
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks Track Camera Ready, 46 pages, 2 figures
>
> **摘要:** Preference datasets are essential for training general-domain, instruction-following language models with Reinforcement Learning from Human Feedback (RLHF). Each subsequent data release raises expectations for future data collection, meaning there is a constant need to advance the quality and diversity of openly available preference data. To address this need, we introduce HelpSteer3-Preference, a permissively licensed (CC-BY-4.0), high-quality, human-annotated preference dataset comprising of over 40,000 samples. These samples span diverse real-world applications of large language models (LLMs), including tasks relating to STEM, coding and multilingual scenarios. Using HelpSteer3-Preference, we train Reward Models (RMs) that achieve top performance on RM-Bench (82.4%) and JudgeBench (73.7%). This represents a substantial improvement (~10% absolute) over the previously best-reported results from existing RMs. We demonstrate HelpSteer3-Preference can also be applied to train Generative RMs and how policy models can be aligned with RLHF using our RMs. Dataset (CC-BY-4.0): https://huggingface.co/datasets/nvidia/HelpSteer3#preference Models (NVIDIA Open Model): https://huggingface.co/collections/nvidia/reward-models-68377c5955575f71fcc7a2a3
>
---
#### [replaced 053] How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01741v2](http://arxiv.org/pdf/2501.01741v2)**

> **作者:** Simone Corbo; Luca Bancale; Valeria De Gennaro; Livia Lestingi; Vincenzo Scotti; Matteo Camilli
>
> **摘要:** Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM , which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using five state-of-the-art LLMs as evaluation subjects having increasing complexity (7-671B parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).
>
---
#### [replaced 054] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16986v2](http://arxiv.org/pdf/2505.16986v2)**

> **作者:** Amartya Chakraborty; Paresh Dashore; Nadia Bathaee; Anmol Jain; Anirban Das; Shi-Xiong Zhang; Sambit Sahu; Milind Naphade; Genta Indra Winata
>
> **备注:** Accepted by NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities as intelligent agents capable of solving complex problems. However, effective planning in scenarios involving dependencies between API or tool calls-particularly in multi-turn conversations-remains a significant challenge. To address this, we introduce T1, a tool-augmented, multi-domain, multi-turn conversational dataset specifically designed to capture and manage inter-tool dependencies across diverse domains. T1 enables rigorous evaluation of agents' ability to coordinate tool use across nine distinct domains (4 single domain and 5 multi-domain) with the help of an integrated caching mechanism for both short- and long-term memory, while supporting dynamic replanning-such as deciding whether to recompute or reuse cached results. Beyond facilitating research on tool use and planning, T1 also serves as a benchmark for evaluating the performance of open-weight and proprietary large language models. We present results powered by T1-Agent, highlighting their ability to plan and reason in complex, tool-dependent scenarios.
>
---
#### [replaced 055] Visual Cues Support Robust Turn-taking Prediction in Noise
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.22088v2](http://arxiv.org/pdf/2505.22088v2)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **备注:** Accepted to INTERSPEECH 2025, 10.21437/Interspeech.2025-668
>
> **摘要:** Accurate predictive turn-taking models (PTTMs) are essential for naturalistic human-robot interaction. However, little is known about their performance in noise. This study therefore explores PTTM performance in types of noise likely to be encountered once deployed. Our analyses reveal PTTMs are highly sensitive to noise. Hold/shift accuracy drops from 84% in clean speech to just 52% in 10 dB music noise. Training with noisy data enables a multimodal PTTM, which includes visual features to better exploit visual cues, with 72% accuracy in 10 dB music noise. The multimodal PTTM outperforms the audio-only PTTM across all noise types and SNRs, highlighting its ability to exploit visual cues; however, this does not always generalise to new types of noise. Analysis also reveals that successful training relies on accurate transcription, limiting the use of ASR-derived transcriptions to clean conditions. We make code publicly available for future research.
>
---
#### [replaced 056] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08221v2](http://arxiv.org/pdf/2508.08221v2)**

> **作者:** Zihe Liu; Jiashun Liu; Yancheng He; Weixun Wang; Jiaheng Liu; Ling Pan; Xinyu Hu; Shaopan Xiong; Ju Huang; Jian Hu; Shengyi Huang; Siran Yang; Jiamang Wang; Wenbo Su; Bo Zheng
>
> **备注:** 26 pages, 21 figures
>
> **摘要:** Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
>
---
#### [replaced 057] Information-Theoretic Reward Decomposition for Generalizable RLHF
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.06020v2](http://arxiv.org/pdf/2504.06020v2)**

> **作者:** Liyuan Mao; Haoran Xu; Amy Zhang; Weinan Zhang; Chenjia Bai
>
> **备注:** Work done during internships at Institute of Artificial Intelligence (TeleAI), China Telecom
>
> **摘要:** A generalizable reward model is crucial in Reinforcement Learning from Human Feedback (RLHF) as it enables correctly evaluating unseen prompt-response pairs. However, existing reward models lack this ability, as they are typically trained by increasing the reward gap between chosen and rejected responses, while overlooking the prompts that the responses are conditioned on. Consequently, when the trained reward model is evaluated on prompt-response pairs that lie outside the data distribution, neglecting the effect of prompts may result in poor generalization of the reward model. To address this issue, we decompose the reward value into two independent components: prompt-free reward and prompt-related reward. Prompt-free reward represents the evaluation that is determined only by responses, while the prompt-related reward reflects the reward that derives from both the prompt and the response. We extract these two components from an information-theoretic perspective, which requires no extra models. Subsequently, we propose a new reward learning algorithm by prioritizing data samples based on their prompt-free reward values. Through toy examples, we demonstrate that the extracted prompt-free and prompt-related rewards effectively characterize two parts of the reward model. Further, standard evaluations show that our method improves both the alignment performance and the generalization capability of the reward model.
>
---
#### [replaced 058] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21749v2](http://arxiv.org/pdf/2505.21749v2)**

> **作者:** M. Reza Ebrahimi; Roland Memisevic
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The role of hidden units in recurrent neural networks is typically seen as modeling memory, with research focusing on enhancing information retention through gating mechanisms. A less explored perspective views hidden units as active participants in the computation performed by the network, rather than passive memory stores. In this work, we revisit bilinear operations, which involve multiplicative interactions between hidden units and input embeddings. We demonstrate theoretically and empirically that they constitute a natural inductive bias for representing the evolution of hidden states in state tracking tasks. These are the simplest type of tasks that require hidden units to actively contribute to the behavior of the network. We also show that bilinear state updates form a natural hierarchy corresponding to state tracking tasks of increasing complexity, with popular linear recurrent networks such as Mamba residing at the lowest-complexity center of that hierarchy.
>
---
#### [replaced 059] Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05179v4](http://arxiv.org/pdf/2503.05179v4)**

> **作者:** Simon A. Aytes; Jinheon Baek; Sung Ju Hwang
>
> **备注:** EMNLP 2025
>
> **摘要:** Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 18 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 84% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
>
---
#### [replaced 060] Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08730v2](http://arxiv.org/pdf/2508.08730v2)**

> **作者:** Weibin Liao; Tianlong Wang; Yinghao Zhu; Yasha Wang; Junyi Gao; Liantao Ma
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Medical Lay Language Generation (MLLG) plays a vital role in improving the accessibility of complex scientific content for broader audiences. Recent literature to MLLG commonly employ parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) to fine-tuning large language models (LLMs) using paired expert-lay language datasets. However, LoRA struggles with the challenges posed by multi-source heterogeneous MLLG datasets. Specifically, through a series of exploratory experiments, we reveal that standard LoRA fail to meet the requirement for semantic fidelity and diverse lay-style generation in MLLG task. To address these limitations, we propose Magical, an asymmetric LoRA architecture tailored for MLLG under heterogeneous data scenarios. Magical employs a shared matrix $A$ for abstractive summarization, along with multiple isolated matrices $B$ for diverse lay-style generation. To preserve semantic fidelity during the lay language generation process, Magical introduces a Semantic Invariance Constraint to mitigate semantic subspace shifts on matrix $A$. Furthermore, to better adapt to diverse lay-style generation, Magical incorporates the Recommendation-guided Switch, an externally interface to prompt the LLM to switch between different matrices $B$. Experimental results on three real-world lay language generation datasets demonstrate that Magical consistently outperforms prompt-based methods, vanilla LoRA, and its recent variants, while also reducing trainable parameters by 31.66%. Our code is publicly available at https://github.com/tianlwang/Magical.git.
>
---
#### [replaced 061] Video-Skill-CoT: Skill-based Chain-of-Thoughts for Domain-Adaptive Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03525v2](http://arxiv.org/pdf/2506.03525v2)**

> **作者:** Daeun Lee; Jaehong Yoon; Jaemin Cho; Mohit Bansal
>
> **备注:** Project website: https://video-skill-cot.github.io/
>
> **摘要:** Recent advances in Chain-of-Thought (CoT) reasoning have improved complex video understanding, but existing methods often struggle to adapt to domain-specific skills (e.g., event detection, spatial relation understanding, emotion understanding) over various video content. To address this, we propose Video-Skill-CoT (a.k.a. Video-SKoT), a framework that automatically constructs and leverages skill-aware CoT supervisions for domain-adaptive video reasoning. First, we construct skill-based CoT annotations: we extract domain-relevant reasoning skills from training questions, cluster them into a shared skill taxonomy, and create detailed multi-step CoT rationale tailored to each video-question pair for training. Second, we introduce a skill-specific expert learning framework. Each expert module specializes in a subset of reasoning skills and is trained with lightweight adapters using the collected CoT supervision. We demonstrate the effectiveness of the proposed approach on three video understanding benchmarks, where Video-SKoT consistently outperforms strong baselines. We also provide in-depth analyses on comparing different CoT annotation pipelines and learned skills over multiple video domains.
>
---
#### [replaced 062] Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers
- **分类: cs.CL; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2506.08966v2](http://arxiv.org/pdf/2506.08966v2)**

> **作者:** Marek Kadlčík; Michal Štefánik; Timothee Mickus; Michal Spiegel; Josef Kuchař
>
> **摘要:** Pretrained language models (LMs) are prone to arithmetic errors. Existing work showed limited success in probing numeric values from models' representations, indicating that these errors can be attributed to the inherent unreliability of distributionally learned embeddings in representing exact quantities. However, we observe that previous probing methods are inadequate for the emergent structure of learned number embeddings with sinusoidal patterns. In response, we propose a novel probing technique that decodes numeric values from input embeddings with near-perfect accuracy across a range of open-source LMs. This proves that after the sole pre-training, LMs represent numbers with remarkable precision. Finally, we find that the embeddings' precision, judged by our probe's accuracy, explains a large portion of LM's errors in elementary arithmetic, and show that aligning the embeddings with the pattern our probes discover can mitigate these errors.
>
---
#### [replaced 063] Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.20176v2](http://arxiv.org/pdf/2510.20176v2)**

> **作者:** Yuhang Zhou; Mingrui Zhang; Ke Li; Mingyi Wang; Qiao Liu; Qifei Wang; Jiayi Liu; Fei Liu; Serena Li; Weiwei Li; Mingze Gao; Abhishek Kumar; Xiangjun Fan; Zhuokai Zhao; Lizhu Zhang
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding.
>
---
#### [replaced 064] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16753v2](http://arxiv.org/pdf/2508.16753v2)**

> **作者:** Nitin Gupta; Pallav Koppisetti; Kausik Lakkaraju; Biplav Srivastava
>
> **备注:** 11 pages, 7 figures, accepted at IAAI/AAAI 2026; updated with figures, captions, and acknowledgments
>
> **摘要:** The rapid proliferation of Generative AI (GenAI) into diverse, high-stakes domains necessitates robust and reproducible evaluation methods. However, practitioners often resort to ad-hoc, non-standardized scripts, as common metrics are often unsuitable for specialized, structured outputs (e.g., automated plans, time-series) or holistic comparison across modalities (e.g., text, audio, and image). This fragmentation hinders comparability and slows AI system development. To address this challenge, we present GAICo (Generative AI Comparator): a deployed, open-source Python library that streamlines and standardizes GenAI output comparison. GAICo provides a unified, extensible framework supporting a comprehensive suite of reference-based metrics for unstructured text, specialized structured data formats, and multimedia (images, audio). Its architecture features a high-level API for rapid, end-to-end analysis, from multi-model comparison to visualization and reporting, alongside direct metric access for granular control. We demonstrate GAICo's utility through a detailed case study evaluating and debugging complex, multi-modal AI Travel Assistant pipelines. GAICo empowers AI researchers and developers to efficiently assess system performance, make evaluation reproducible, improve development velocity, and ultimately build more trustworthy AI systems, aligning with the goal of moving faster and safer in AI deployment. Since its release on PyPI in Jun 2025, the tool has been downloaded over 13K times, across versions, by Aug 2025, demonstrating growing community interest.
>
---
#### [replaced 065] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.20083v4](http://arxiv.org/pdf/2503.20083v4)**

> **作者:** Benjamin Minixhofer; Ivan Vulić; Edoardo Maria Ponti
>
> **备注:** NeurIPS 2025
>
> **摘要:** Distillation has shown remarkable success in transferring knowledge from a Large Language Model (LLM) teacher to a student LLM. However, current distillation methods require similar tokenizers between the teacher and the student, restricting their applicability to only a small subset of teacher-student pairs. In this work, we develop a principled cross-tokenizer distillation method to solve this crucial deficiency. Our method is the first to enable effective distillation across fundamentally different tokenizers, while also substantially outperforming prior methods in all other cases. We verify the efficacy of our method on three distinct use cases. First, we show that viewing tokenizer transfer as self-distillation enables unprecedentedly effective transfer across tokenizers, including rapid transfer of subword models to the byte-level. Transferring different models to the same tokenizer also enables ensembling to boost performance. Secondly, we distil a large maths-specialised LLM into a small general-purpose model with a different tokenizer, achieving competitive maths problem-solving performance. Thirdly, we use our method to train state-of-the-art embedding prediction hypernetworks for training-free tokenizer transfer. Our results unlock an expanded range of teacher-student pairs for distillation, enabling new ways to adapt and enhance interaction between LLMs.
>
---
#### [replaced 066] Retention analysis of edited knowledge after fine-tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14198v2](http://arxiv.org/pdf/2507.14198v2)**

> **作者:** Fufang Wen; Shichang Zhang
>
> **摘要:** Large language models (LLMs) store vast amounts of knowledge, which often requires updates to correct factual errors, incorporate newly acquired information, or adapt model behavior. Model editing methods have emerged as efficient solutions for such updates, offering localized and precise knowledge modification at significantly lower computational cost than continual training. In parallel, LLMs are frequently fine-tuned for a wide range of downstream tasks. However, the effect of fine-tuning on previously edited knowledge remains poorly understood. In this work, we systematically investigate how different fine-tuning objectives interact with various model editing techniques. Our findings show that edited knowledge is substantially more susceptible to forgetting during fine-tuning than intrinsic knowledge acquired through pre-training. This analysis highlights a key limitation of current editing approaches and suggests that evaluating edit robustness under downstream fine-tuning is critical for their practical deployment. We further find that knowledge retention can be significantly improved by either augmenting edit knowledge with paraphrases or by freezing layers associated with edited content in fine-tuning stage, offering insight for developing more robust editing algorithms.
>
---
#### [replaced 067] SimuRA: A World-Model-Driven Simulative Reasoning Architecture for General Goal-Oriented Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.23773v2](http://arxiv.org/pdf/2507.23773v2)**

> **作者:** Mingkai Deng; Jinyu Hou; Zhiting Hu; Eric Xing
>
> **备注:** This submission has been updated to adjust the scope and presentation of the work
>
> **摘要:** AI agents built on foundation models hold enormous promise. Current practice, however, focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also faces practical limitations from black-box autoregressive reasoning, where decisions unfold token by token without explicit simulation or counterfactual evaluation of outcomes. Humans, on the other hand, reason and plan by mentally simulating the consequences of actions within an internal model of the world -- a capability that supports flexible, goal-directed behavior across diverse contexts. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of an optimal agent in any general environment, SimuRA addresses the limitations of black-box autoregressive reasoning by incorporating the world model for planning via simulation. Our prototype world model is implemented using LLMs as a substrate, leveraging the natural language as a discrete, hierarchical representation grounded in concepts for planning, while remaining model-agnostic. On complex web-browsing tasks such as flight search, SimuRA improves the success rate from 0% to 32.2% compared to a representative open-web agent baseline. Across tasks, world-model-based planning achieves up to 124% higher task completion rates than a matched black-box autoregressive baseline, demonstrating the advantages of simulative reasoning. We release ReasonerAgent-Web, a web-browsing agent built on SimuRA, as an open-source research demo.
>
---
#### [replaced 068] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07851v2](http://arxiv.org/pdf/2506.07851v2)**

> **作者:** Yiju Guo; Wenkai Yang; Zexu Sun; Ning Ding; Zhiyuan Liu; Yankai Lin
>
> **备注:** Accepted at NeurIPS 2025, camera-ready version
>
> **摘要:** Large language models (LLMs) have demonstrated significant improvements in contextual understanding. However, their ability to attend to truly critical information during long-context reasoning and generation still falls behind the pace. Specifically, our preliminary experiments reveal that certain distracting patterns can misdirect the model's attention during inference, and removing these patterns substantially improves reasoning accuracy and generation quality. We attribute this phenomenon to spurious correlations in the training data, which obstruct the model's capacity to infer authentic causal instruction-response relationships. This phenomenon may induce redundant reasoning processes, potentially resulting in significant inference overhead and, more critically, the generation of erroneous or suboptimal responses. To mitigate this, we introduce a two-stage framework called Learning to Focus (LeaF) leveraging intervention-based inference to disentangle confounding factors. In the first stage, LeaF employs gradient-based comparisons with an advanced teacher to automatically identify confounding tokens based on causal relationships in the training corpus. Then, in the second stage, it prunes these tokens during distillation to enact intervention, aligning the student's attention with the teacher's focus distribution on truly critical context tokens. Experimental results demonstrate that LeaF not only achieves an absolute improvement in various mathematical reasoning, code generation and multi-hop question answering benchmarks but also effectively suppresses attention to confounding tokens during inference, yielding a more interpretable and reliable reasoning model.
>
---
#### [replaced 069] FAITH: A Framework for Assessing Intrinsic Tabular Hallucinations in Finance
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05201v2](http://arxiv.org/pdf/2508.05201v2)**

> **作者:** Mengao Zhang; Jiayu Fu; Tanya Warrier; Yuwen Wang; Tianhui Tan; Ke-wei Huang
>
> **备注:** 9 pages, AMC ICAIF'25
>
> **摘要:** Hallucination remains a critical challenge for deploying Large Language Models (LLMs) in finance. Accurate extraction and precise calculation from tabular data are essential for reliable financial analysis, since even minor numerical errors can undermine decision-making and regulatory compliance. Financial applications have unique requirements, often relying on context-dependent, numerical, and proprietary tabular data that existing hallucination benchmarks rarely capture. In this study, we develop a rigorous and scalable framework for evaluating intrinsic hallucinations in financial LLMs, conceptualized as a context-aware masked span prediction task over real-world financial documents. Our main contributions are: (1) a novel, automated dataset creation paradigm using a masking strategy; (2) a new hallucination evaluation dataset derived from S&P 500 annual reports; and (3) a comprehensive evaluation of intrinsic hallucination patterns in state-of-the-art LLMs on financial tabular data. Our work provides a robust methodology for in-house LLM evaluation and serves as a critical step toward building more trustworthy and reliable financial Generative AI systems.
>
---
#### [replaced 070] Inference-time Alignment in Continuous Space
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20081v4](http://arxiv.org/pdf/2505.20081v4)**

> **作者:** Yige Yuan; Teng Xiao; Li Yunfan; Bingbing Xu; Shuchang Tao; Yunqi Qiu; Huawei Shen; Xueqi Cheng
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Aligning large language models with human feedback at inference time has received increasing attention due to its flexibility. Existing methods rely on generating multiple responses from the base policy for search using a reward model, which can be considered as searching in a discrete response space. However, these methods struggle to explore informative candidates when the base policy is weak or the candidate set is small, resulting in limited effectiveness. In this paper, to address this problem, we propose Simple Energy Adaptation ($\textbf{SEA}$), a simple yet effective algorithm for inference-time alignment. In contrast to expensive search over the discrete space, SEA directly adapts original responses from the base policy toward the optimal one via gradient-based sampling in continuous latent space. Specifically, SEA formulates inference as an iterative optimization procedure on an energy function over actions in the continuous space defined by the optimal policy, enabling simple and effective alignment. For instance, despite its simplicity, SEA outperforms the second-best baseline with a relative improvement of up to $ \textbf{77.51%}$ on AdvBench and $\textbf{16.36%}$ on MATH. Our code is publicly available at https://github.com/yuanyige/sea
>
---
#### [replaced 071] HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13254v2](http://arxiv.org/pdf/2505.13254v2)**

> **作者:** Siran Liu; Yang Ye; Qianchao Zhu; Zane Cao; Yongchao He
>
> **摘要:** Autoregressive decoding inherently limits the inference throughput of Large Language Model (LLM) due to its sequential dependency. Speculative decoding mitigates this by verifying multiple predicted tokens in parallel, but its efficiency remains constrained by what we identify as verification heterogeneity -- the uneven difficulty of verifying different speculative candidates. In practice, a small subset of high-confidence predictions accounts for most successful verifications, yet existing methods treat all candidates uniformly, leading to redundant computation. We present HeteroSpec, a heterogeneity-adaptive speculative decoding framework that allocates verification effort in proportion to candidate uncertainty. HeteroSpec estimates verification complexity using a lightweight entropy-based quantifier, partitions candidates via a data-driven stratification policy, and dynamically tunes speculative depth and pruning thresholds through coordinated optimization. Across five benchmarks and four LLMs, HeteroSpec delivers an average 4.24$\times$ decoding speedup over state-of-the-art methods such as EAGLE-3, while preserving exact output distributions. Crucially, HeteroSpec requires no model retraining and remains compatible with other inference optimizations, making it a practical direction for improving speculative decoding efficiency.
>
---
#### [replaced 072] Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13866v2](http://arxiv.org/pdf/2505.13866v2)**

> **作者:** Jiwon Song; Dongwon Jo; Yulhwa Kim; Jae-Joon Kim
>
> **摘要:** Recent reasoning-focused language models achieve high accuracy by generating lengthy intermediate reasoning paths before producing final answers. While this approach is effective in solving problems that require logical thinking, long reasoning paths significantly increase memory usage and reduce throughput of token generation, limiting the practical deployment of such models. We propose Reasoning Path Compression (RPC), a training-free method that accelerates inference by leveraging the semantic sparsity of reasoning paths. RPC periodically compresses the KV cache by retaining cache entries that receive high importance score, which are computed using a selector window composed of recently generated queries. Experiments show that RPC improves generation throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full KV cache, with an accuracy drop of 1.2\% on the AIME 2024 benchmark. Our findings demonstrate that semantic sparsity in reasoning traces can be effectively exploited for compression, offering a practical path toward efficient deployment of reasoning LLMs. Our code is available at https://github.com/jiwonsong-dev/ReasoningPathCompression.
>
---
#### [replaced 073] GoRA: Gradient-driven Adaptive Low Rank Adaptation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12171v3](http://arxiv.org/pdf/2502.12171v3)**

> **作者:** Haonan He; Peng Ye; Yuchen Ren; Yuan Yuan; Luyang Zhou; Shucun Ju; Lei Chen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Low-Rank Adaptation (LoRA) is a crucial method for efficiently fine-tuning large language models (LLMs), with its effectiveness influenced by two key factors: rank selection and weight initialization. While numerous LoRA variants have been proposed to improve performance by addressing one of these aspects, they often compromise usability or computational efficiency. In this paper, we analyze and identify the core limitations of existing approaches and propose a novel framework--GoRA (Gradient-driven Adaptive Low Rank Adaptation)--that simultaneously adapts both the rank and initialization strategy within a unified framework. GoRA leverages gradient information during training to dynamically assign optimal ranks and initialize low-rank adapter weights in an adaptive manner. To our knowledge, GoRA is the first method that not only addresses the limitations of prior approaches--which often focus on either rank selection or initialization in isolation--but also unifies both aspects within a single framework, enabling more effective and efficient adaptation. Extensive experiments across various architectures and modalities show that GoRA consistently outperforms existing LoRA-based methods while preserving the efficiency of vanilla LoRA. For example, when fine-tuning Llama3.1-8B-Base for mathematical reasoning, GoRA achieves a 5.13-point improvement over standard LoRA and even outperforms full fine-tuning by 2.05 points under high-rank settings. Code is available at: https://github.com/hhnqqq/MyTransformers.
>
---
#### [replaced 074] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.15188v3](http://arxiv.org/pdf/2509.15188v3)**

> **作者:** Yeongbin Seo; Dongha Lee; Jaehyung Kim; Jinyoung Yeo
>
> **备注:** NeurIPS 2025 spotlight
>
> **摘要:** Autoregressive (AR) language models generate text one token at a time, which limits their inference speed. Diffusion-based language models offer a promising alternative, as they can decode multiple tokens in parallel. However, we identify a key bottleneck in current diffusion LMs: the long decoding-window problem, where tokens generated far from the input context often become irrelevant or repetitive. Previous solutions like semi-autoregressive address this issue by splitting windows into blocks (sacrificing bidirectionality), but we find that this also leads to time-interval expansion problem, sacrificing the speed. Therefore, semi-AR eliminates the main advantages of diffusion models. To overcome this, we propose Convolutional decoding (Conv), a normalization-based method that narrows the decoding window without hard segmentation, leading to better fluency and flexibility. Additionally, we introduce Rejecting Rule-based Fine-Tuning (R2FT), a post-hoc training scheme that better aligns tokens at positions far from context. Our methods achieve state-of-the-art results on open-ended generation benchmarks (e.g., AlpacaEval) among diffusion LM baselines, with significantly lower step size than previous works, demonstrating both speed and quality improvements.
>
---
#### [replaced 075] Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18434v2](http://arxiv.org/pdf/2510.18434v2)**

> **作者:** Qingqing Gu; Dan Wang; Yue Zhao; Xiaoyu Wang; Zhonglin Jiang; Yong Chen; Hongyan Li; Luo Ji
>
> **备注:** Accepted to PRICAI 2025
>
> **摘要:** Chain-of-Thought (CoT) is widely applied to enhance the LLM capability in math, coding and reasoning tasks. However, its performance is limited for open-domain tasks, when there are no clearly defined reasoning steps or logical transitions. To mitigate such challenges, we propose a new prompt-based paradigm called Chain of Conceptual Thoughts (CoCT), which suggests the LLM first to produce the tag of concepts, then complete the detailed content following the concept. To encourage this hierarchical way of thinking, we implement the concepts with emotions, strategies and topics. We experiment with this paradigm in daily and emotional support conversations, covering tasks with both in-domain and out-of-domain concept settings. Automatic, human, and LLM-based evaluations reveal that CoCT surpasses several prompt-based baselines such as self-refine, ECoT, SoT and RAG, suggesting a potential solution of LLM prompting paradigm for a wider scope of tasks.
>
---
#### [replaced 076] BLEUBERI: BLEU is a surprisingly effective reward for instruction following
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11080v3](http://arxiv.org/pdf/2505.11080v3)**

> **作者:** Yapei Chang; Yekyung Kim; Michael Krumdick; Amir Zadeh; Chuan Li; Chris Tanner; Mohit Iyyer
>
> **备注:** neurips cam-ready
>
> **摘要:** Reward models are central to aligning LLMs with human preferences, but they are costly to train, requiring large-scale human-labeled preference data and powerful pretrained LLM backbones. Meanwhile, the increasing availability of high-quality synthetic instruction-following datasets raises the question: can simpler, reference-based metrics serve as viable alternatives to reward models during RL-based alignment? In this paper, we show first that BLEU, a basic string-matching metric, surprisingly matches strong reward models in agreement with human preferences on general instruction-following datasets. Based on this insight, we develop BLEUBERI, a method that first identifies challenging instructions and then applies Group Relative Policy Optimization (GRPO) using BLEU directly as the reward function. We demonstrate that BLEUBERI-trained models are competitive with models trained via reward model-guided RL across four challenging instruction-following benchmarks and three different base language models. A human evaluation further supports that the quality of BLEUBERI model outputs is on par with those from reward model-aligned models. Moreover, BLEUBERI models generate outputs that are more factually grounded than competing methods. Overall, we show that given access to high-quality reference outputs (easily obtained via existing instruction-following datasets or synthetic data generation), string matching-based metrics are cheap yet effective proxies for reward models during alignment. We release our code and data at https://github.com/lilakk/BLEUBERI.
>
---
#### [replaced 077] AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.01268v2](http://arxiv.org/pdf/2510.01268v2)**

> **作者:** Hongyi Zhou; Jin Zhu; Pingfan Su; Kai Ye; Ying Yang; Shakeel A O B Gavioli-Akilagun; Chengchun Shi
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
>
---
#### [replaced 078] RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.09584v2](http://arxiv.org/pdf/2409.09584v2)**

> **作者:** Qingyao Li; Wei Xia; Kounianhua Du; Xinyi Dai; Ruiming Tang; Yasheng Wang; Yong Yu; Weinan Zhang
>
> **摘要:** Tree search methods have demonstrated impressive performance in code generation. Previous methods combine tree search with reflection that summarizes past mistakes to achieve iterative improvement. However, these methods face significant challenges. First, they search directly within the code language space, neglecting the underlying reasoning process critical for effective code generation. Second, reflection-based approaches merely accumulate historical errors in memory without providing correct reasoning pathways, making it difficult for subsequent search iterations to identify optimal solutions, resulting in decreased search quality. In this work, we propose RethinkMCTS, a framework that systematically explores and refines the reasoning process for code generation. Specifically, we employ MCTS to search for thoughts before code generation and integrate MCTS with a refinement mechanism called rethink, which incorporates fine-grained code execution feedback to refine erroneous thoughts during the search. It ensures the search path aligns with better reasoning, improving overall search quality. Through extensive experiments, we demonstrate that RethinkMCTS outperforms previous search-based and feedback-enhanced code generation baselines.
>
---
#### [replaced 079] zip2zip: Inference-Time Adaptive Tokenization via Online Compression
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01084v2](http://arxiv.org/pdf/2506.01084v2)**

> **作者:** Saibo Geng; Nathan Ranchin; Yunzhen yao; Maxime Peyrard; Chris Wendler; Michael Gastpar; Robert West
>
> **备注:** NeurIPS 2025
>
> **摘要:** Tokenization efficiency plays a critical role in the performance and cost of large language models (LLMs), yet most models rely on static tokenizers optimized on general-purpose corpora. These tokenizers' fixed vocabularies often fail to adapt to domain- or language-specific inputs, leading to longer token sequences and higher computational costs. We introduce zip2zip, a novel method for achieving context-adaptive tokenization in LLMs at inference time. Leveraging an online data compression algorithm (Lempel-Ziv-Welch), zip2zip dynamically expands its active vocabulary at inference time by continuously replacing fragmented token sequences with more compact hypertokens, which it can immediately output during generation. In doing so, the model refines its internal tokenization scheme to match the token distribution of the current context, reducing redundancy and improving representational efficiency. zip2zip consists of three key components: (1) a tokenizer based on Lempel-Ziv-Welch compression that incrementally merges co-occurring tokens into reusable hypertokens on the fly; (2) a dynamic embedding (and unembedding) layer that computes embeddings for newly formed hypertokens at runtime; and (3) a variant of autoregressive language modeling that pretrains the model to handle hypertokenized, compressed text sequences as inputs and outputs. We show that an existing LLM can be uptrained for zip2zip in 10 GPU-hours via parameter-efficient finetuning. The resulting LLM performs test-time adaptation, learning to use hypertokens in unseen contexts and reducing input and output tokens by 15-40%.
>
---
#### [replaced 080] PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23184v2](http://arxiv.org/pdf/2509.23184v2)**

> **作者:** Boyi Zeng; He Li; Shixiang Song; Yixuan Wang; Ziwei He; Xinbing Wang; Zhouhan Lin
>
> **摘要:** The remarkable success of Chain-of-Thought (CoT), which enhances performance by scaling generation steps at test-time, inspires us to ask: can we leverage a similar scaling of computational steps during pretraining to improve the generation of each individual token? To address this, we propose a novel pre-training methodology: Pretraining Language Models with Latent Thoughts (PonderLM-2). Our approach pretrains a language model (LM) to first generate an intermediate latent thought-the last hidden state of the current position-which is then used as input to predict the actual subsequent token. This additional computational step enables the LM to refine its prediction within unconstrained continuous space. Our experiments demonstrate that, at an identical inference cost, a LM that generates one additional latent thought per token outperforms a standard model with double the parameters. For instance, our PonderLM-2-Pythia-1.4B, pretrained on 300B tokens from the Pile, significantly surpasses the vanilla Pythia-2.8B trained on the same data on both language modeling and a range of general downstream tasks. Furthermore, increasing the number of latent thoughts generated before each actual token-forming a chain analogous to CoT-consistently improves the model's performance.
>
---
#### [replaced 081] Knee-Deep in C-RASP: A Transformer Depth Hierarchy
- **分类: cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2506.16055v2](http://arxiv.org/pdf/2506.16055v2)**

> **作者:** Andy Yang; Michaël Cadilhac; David Chiang
>
> **备注:** 35 pages, 5 figures
>
> **摘要:** It has been observed that transformers with greater depth (that is, more layers) have more capabilities, but can we establish formally which capabilities are gained? We answer this question with a theoretical proof followed by an empirical study. First, we consider transformers that round to fixed precision except inside attention. We show that this subclass of transformers is expressively equivalent to the programming language C-RASP and this equivalence preserves depth. Second, we prove that deeper C-RASP programs are more expressive than shallower C-RASP programs, implying that deeper transformers are more expressive than shallower transformers (within the subclass mentioned above). The same is also proven for transformers with positional encodings (like RoPE and ALiBi). These results are established by studying a temporal logic with counting operators equivalent to C-RASP. Finally, we provide empirical evidence that our theory predicts the depth required for transformers without positional encodings to length-generalize on a family of sequential dependency tasks.
>
---
#### [replaced 082] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction
- **分类: cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.21043v2](http://arxiv.org/pdf/2505.21043v2)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **备注:** Accepted to ACL 2025, Findings of the Association for Computational Linguistics
>
> **摘要:** Turn-taking is richly multimodal. Predictive turn-taking models (PTTMs) facilitate naturalistic human-robot interaction, yet most rely solely on speech. We introduce MM-VAP, a multimodal PTTM which combines speech with visual cues including facial expression, head pose and gaze. We find that it outperforms the state-of-the-art audio-only in videoconferencing interactions (84% vs. 79% hold/shift prediction accuracy). Unlike prior work which aggregates all holds and shifts, we group by duration of silence between turns. This reveals that through the inclusion of visual features, MM-VAP outperforms a state-of-the-art audio-only turn-taking model across all durations of speaker transitions. We conduct a detailed ablation study, which reveals that facial expression features contribute the most to model performance. Thus, our working hypothesis is that when interlocutors can see one another, visual cues are vital for turn-taking and must therefore be included for accurate turn-taking prediction. We additionally validate the suitability of automatic speech alignment for PTTM training using telephone speech. This work represents the first comprehensive analysis of multimodal PTTMs. We discuss implications for future work and make all code publicly available.
>
---
#### [replaced 083] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13181v2](http://arxiv.org/pdf/2505.13181v2)**

> **作者:** Zhengrui Ma; Yang Feng; Chenze Shao; Fandong Meng; Jie Zhou; Min Zhang
>
> **备注:** NeurIPS 2025; Demos and code are available at https://github.com/ictnlp/SLED-TTS
>
> **摘要:** We introduce SLED, an alternative approach to speech language modeling by encoding speech waveforms into sequences of continuous latent representations and modeling them autoregressively using an energy distance objective. The energy distance offers an analytical measure of the distributional gap by contrasting simulated and target samples, enabling efficient training to capture the underlying continuous autoregressive distribution. By bypassing reliance on residual vector quantization, SLED avoids discretization errors and eliminates the need for the complicated hierarchical architectures common in existing speech language models. It simplifies the overall modeling pipeline while preserving the richness of speech information and maintaining inference efficiency. Empirical results demonstrate that SLED achieves strong performance in both zero-shot and streaming speech synthesis, showing its potential for broader applications in general-purpose speech language models.
>
---
#### [replaced 084] Theory-Grounded Evaluation of Human-Like Fallacy Patterns in LLM Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11128v2](http://arxiv.org/pdf/2506.11128v2)**

> **作者:** Andrew Keenan Richardson; Ryan Othniel Kearns; Sean Moss; Vincent Wang-Mascianica; Philipp Koralus
>
> **摘要:** We study logical reasoning in language models by asking whether their errors follow established human fallacy patterns. Using the Erotetic Theory of Reasoning (ETR) and its open-source implementation, PyETR, we programmatically generate 383 formally specified reasoning problems and evaluate 38 models. For each response, we judge logical correctness and, when incorrect, whether it matches an ETR-predicted fallacy. Two results stand out: (i) as a capability proxy (Chatbot Arena Elo) increases, a larger share of a model's incorrect answers are ETR-predicted fallacies $(\rho=0.360, p=0.0265)$, while overall correctness on this dataset shows no correlation with capability; (ii) reversing premise order significantly reduces fallacy production for many models, mirroring human order effects. Methodologically, PyETR provides an open-source pipeline for unbounded, synthetic, contamination-resistant reasoning tests linked to a cognitive theory, enabling analyses that focus on error composition rather than error rate.
>
---
#### [replaced 085] L$^2$M: Mutual Information Scaling Law for Long-Context Language Modeling
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; math.IT; physics.data-an**

- **链接: [http://arxiv.org/pdf/2503.04725v2](http://arxiv.org/pdf/2503.04725v2)**

> **作者:** Zhuo Chen; Oriol Mayné i Comas; Zhuotao Jin; Di Luo; Marin Soljačić
>
> **备注:** 34 pages, 13 figures, 2 tables
>
> **摘要:** We present a universal theoretical framework for understanding long-context language modeling based on a bipartite mutual information scaling law that we rigorously verify in natural language. We demonstrate that bipartite mutual information captures multi-token interactions distinct from and scaling independently of conventional two-point mutual information, and show that this provides a more complete characterization of the dependencies needed for accurately modeling long sequences. Leveraging this scaling law, we formulate the Long-context Language Modeling (L$^2$M) condition, which lower bounds the necessary scaling of a model's history state -- the latent variables responsible for storing past information -- for effective long-context modeling. We validate the framework and its predictions on transformer and state-space models. Our work provides a principled foundation to understand long-context modeling and to design more efficient architectures with stronger long-context capabilities, with potential applications beyond natural language.
>
---
#### [replaced 086] SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20990v2](http://arxiv.org/pdf/2506.20990v2)**

> **作者:** Yifan Yang; Zhen Zhang; Rupak Vignesh Swaminathan; Jing Liu; Nathan Susanj; Zheng Zhang
>
> **摘要:** Fine-tuning vision language models (VLMs) has achieved remarkable performance across various downstream tasks; yet, it requires access to model gradients through backpropagation (BP), making them unsuitable for memory-constrained, inference-only edge devices. To address this limitation, previous work has explored various BP-free fine-tuning methods. However, these approaches often rely on high-variance evolutionary strategies (ES) or zeroth-order (ZO) optimization, and often fail to achieve satisfactory performance. In this paper, we propose a hybrid Sharpness-aware Zeroth-order optimization (SharpZO) approach, specifically designed to enhance the performance of ZO VLM fine-tuning via a sharpness-aware warm-up training. SharpZO features a two-stage optimization process: a sharpness-aware ES stage that globally explores and smooths the loss landscape to construct a strong initialization, followed by a fine-grained local search via sparse ZO optimization. The entire optimization relies solely on forward passes. Detailed theoretical analysis and extensive experiments on CLIP models demonstrate that SharpZO significantly improves accuracy and convergence speed, achieving up to 7% average gain over state-of-the-art forward-only methods.
>
---
#### [replaced 087] Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.21798v2](http://arxiv.org/pdf/2509.21798v2)**

> **作者:** Hongbin Zhang; Kehai Chen; Xuefeng Bai; Yang Xiang; Min Zhang
>
> **备注:** Under review;Work in progress;
>
> **摘要:** Reward models (RMs) are crucial for aligning large language models (LLMs) with diverse cultures. Consequently, evaluating their cultural awareness is essential for further advancing global alignment of LLMs. However, existing RM evaluations fall short in assessing cultural awareness due to the scarcity of culturally relevant evaluation datasets. To fill this gap, we propose Cultural Awareness Reward modeling Benchmark (CARB), covering 10 distinct cultures across 4 cultural domains. Our extensive evaluation of state-of-the-art RMs reveals their deficiencies in modeling cultural awareness and demonstrates a positive correlation between performance on CARB and downstream multilingual cultural alignment tasks. Further analysis identifies the spurious correlations within culture-aware reward modeling, wherein RM's scoring relies predominantly on surface-level features rather than authentic cultural nuance understanding. To address these, we propose Think-as-Locals to elicit deeper culturally grounded reasoning from generative RMs via reinforcement learning from verifiable rewards (RLVR) and employ well-designed rewards to ensure accurate preference judgments and high-quality structured evaluation criteria generation. Experimental results validate its efficacy in mitigating spurious features interference and advancing culture-aware reward modeling.
>
---
#### [replaced 088] Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.02671v2](http://arxiv.org/pdf/2510.02671v2)**

> **作者:** Yavuz Bakman; Sungmin Kang; Zhiqi Huang; Duygu Nur Yaldiz; Catarina G. Belém; Chenyang Zhu; Anoop Kumar; Alfy Samuel; Salman Avestimehr; Daben Liu; Sai Praneeth Karimireddy
>
> **摘要:** Uncertainty Quantification (UQ) research has primarily focused on closed-book factual question answering (QA), while contextual QA remains unexplored, despite its importance in real-world applications. In this work, we focus on UQ for the contextual QA task and propose a theoretically grounded approach to quantify epistemic uncertainty. We begin by introducing a task-agnostic, token-level uncertainty measure defined as the cross-entropy between the predictive distribution of the given model and the unknown true distribution. By decomposing this measure, we isolate the epistemic component and approximate the true distribution by a perfectly prompted, idealized model. We then derive an upper bound for epistemic uncertainty and show that it can be interpreted as semantic feature gaps in the given model's hidden representations relative to the ideal model. We further apply this generic framework to the contextual QA task and hypothesize that three features approximate this gap: context-reliance (using the provided context rather than parametric knowledge), context comprehension (extracting relevant information from context), and honesty (avoiding intentional lies). Using a top-down interpretability approach, we extract these features by using only a small number of labeled samples and ensemble them to form a robust uncertainty score. Experiments on multiple QA benchmarks in both in-distribution and out-of-distribution settings show that our method substantially outperforms state-of-the-art unsupervised (sampling-free and sampling-based) and supervised UQ methods, achieving up to a 13-point PRR improvement while incurring a negligible inference overhead.
>
---
#### [replaced 089] A Hierarchical Framework for Measuring Scientific Paper Innovation via Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14620v2](http://arxiv.org/pdf/2504.14620v2)**

> **作者:** Hongming Tan; Shaoxiong Zhan; Fengwei Jia; Hai-Tao Zheng; Wai Kin Chan
>
> **摘要:** Measuring scientific paper innovation is both important and challenging. Existing content-based methods often overlook the full-paper context, fail to capture the full scope of innovation, and lack generalization. We propose HSPIM, a hierarchical and training-free framework based on large language models (LLMs). It introduces a Paper-to-Sections-to-QAs decomposition to assess innovation. We segment the text by section titles and use zero-shot LLM prompting to implement section classification, question-answering (QA) augmentation, and weighted innovation scoring. The generated QA pair focuses on section-level innovation and serves as additional context to improve the LLM scoring. For each chunk, the LLM outputs a novelty score and a confidence score. We use confidence scores as weights to aggregate novelty scores into a paper-level innovation score. To further improve performance, we propose a two-layer question structure consisting of common and section-specific questions, and apply a genetic algorithm to optimize the question-prompt combinations. Furthermore, under the fine-grained structure of innovation, we extend HSPIM to an HSPIM$^+$ that generates novelty, contribution, and feasibility scores with respective confidence scores. Comprehensive experiments on scientific conference paper datasets show that HSPIM outperforms baseline methods in effectiveness, generalization, and interpretability. Demo code is available at https://github.com/Jasaxion/HSPIM.
>
---
#### [replaced 090] Reinforcement Learning for Reasoning in Large Language Models with One Training Example
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20571v3](http://arxiv.org/pdf/2504.20571v3)**

> **作者:** Yiping Wang; Qing Yang; Zhiyuan Zeng; Liliang Ren; Liyuan Liu; Baolin Peng; Hao Cheng; Xuehai He; Kuan Wang; Jianfeng Gao; Weizhu Chen; Shuohang Wang; Simon Shaolei Du; Yelong Shen
>
> **备注:** link: https://github.com/ypwang61/One-Shot-RLVR
>
> **摘要:** We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6% (8.6% improvement beyond format correction), and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7% (7.0% non-format gain). This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which contains the aforementioned example. Furthermore, RLVR with only two examples even slightly exceeds these results (MATH500: 74.8%, average: 36.6%). Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples. In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-category generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by incorporating entropy loss with an appropriate coefficient) in 1-shot RLVR training. We also further discuss related observations about format correction, label robustness and prompt modification. These findings can inspire future work on RLVR efficiency and encourage a re-examination of recent progress and the underlying mechanisms in RLVR. All resources are open source at https://github.com/ypwang61/One-Shot-RLVR.
>
---
#### [replaced 091] A Hierarchical Error Framework for Reliable Automated Coding in Communication Research: Applications to Health and Political Communication
- **分类: cs.CL; cs.AI; I.2.7; I.2.6**

- **链接: [http://arxiv.org/pdf/2509.24841v2](http://arxiv.org/pdf/2509.24841v2)**

> **作者:** Zhilong Zhao; Yindi Liu
>
> **备注:** Version 2: Enhanced clarification of precision-matching task characteristics and framework applicability conditions. 20 pages, 4 figures, 4 tables. Replication package available at https://doi.org/10.7910/DVN/NDXVLZ
>
> **摘要:** Automated content analysis increasingly supports communication research, yet scaling manual coding into computational pipelines raises concerns about measurement reliability and validity. We introduce a Hierarchical Error Correction (HEC) framework that treats model failures as layered measurement errors (knowledge gaps, reasoning limitations, and complexity constraints) and targets the layers that most affect inference. The framework implements a three-phase methodology: systematic error profiling across hierarchical layers, targeted intervention design matched to dominant error sources, and rigorous validation with statistical testing. Evaluating HEC across health communication (medical specialty classification) and political communication (bias detection), and legal tasks, we validate the approach with five diverse large language models. Results show average accuracy gains of 11.2 percentage points (p < .001, McNemar's test) and stable conclusions via reduced systematic misclassification. Cross-model validation demonstrates consistent improvements (range: +6.8 to +14.6pp), with effectiveness concentrated in moderate-to-high baseline tasks (50-85% accuracy). A boundary study reveals diminished returns in very high-baseline (>85%) or precision-matching tasks, establishing applicability limits. We map layered errors to threats to construct and criterion validity and provide a transparent, measurement-first blueprint for diagnosing error profiles, selecting targeted interventions, and reporting reliability/validity evidence alongside accuracy. This applies to automated coding across communication research and the broader social sciences.
>
---
#### [replaced 092] Uniform Information Density and Syntactic Reduction: Revisiting $\textit{that}$-Mentioning in English Complement Clauses
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.05254v2](http://arxiv.org/pdf/2509.05254v2)**

> **作者:** Hailin Hao; Elsi Kaiser
>
> **备注:** To appear in EMNLP 2025
>
> **摘要:** Speakers often have multiple ways to express the same meaning. The Uniform Information Density (UID) hypothesis suggests that speakers exploit this variability to maintain a consistent rate of information transmission during language production. Building on prior work linking UID to syntactic reduction, we revisit the finding that the optional complementizer $\textit{that}$ in English complement clauses is more likely to be omitted when the clause has low information density (i.e., more predictable). We advance this line of research by analyzing a large-scale, contemporary conversational corpus and using machine learning and neural language models to refine estimates of information density. Our results replicated the established relationship between information density and $\textit{that}$-mentioning. However, we found that previous measures of information density based on matrix verbs' subcategorization probability capture substantial idiosyncratic lexical variation. By contrast, estimates derived from contextual word embeddings account for additional variance in patterns of complementizer usage.
>
---
#### [replaced 093] Teaching Transformers Causal Reasoning through Axiomatic Training
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.07612v3](http://arxiv.org/pdf/2407.07612v3)**

> **作者:** Aniket Vashishtha; Abhinav Kumar; Atharva Pandey; Abbavaram Gowtham Reddy; Kabir Ahuja; Vineeth N Balasubramanian; Amit Sharma
>
> **摘要:** For text-based AI systems to interact in the real world, causal reasoning is an essential skill. Since active interventions are costly, we study to what extent a system can learn causal reasoning from symbolic demonstrations of causal axioms. Specifically, we present an axiomatic training method where the system learns from multiple demonstrations of a causal axiom (or rule), rather than incorporating the axiom as an inductive bias or inferring it from data values. A key question is whether the system would learn to generalize from the axiom demonstrations to more complex scenarios. Our results, based on applying axiomatic training to learn the transitivity axiom and d-separation rule, indicate that such generalization is possible. To avoid data contamination issues, we start with a 67 million parameter transformer model and train it from scratch. On both tasks, we find that a model trained on linear causal chains (along with some noisy variations) can generalize well to complex graphs, including longer causal chains, causal chains with reversed order, and graphs with branching.To handle diverse text inputs, the same method is extended to finetune language models. Finetuning Llama-3-8B-Instruct model on our axiomatic data leads to significant gains on causal benchmarks such as Corr2Cause and CLEAR, in some cases providing state-of-the-art performance surpassing GPT-4.
>
---
#### [replaced 094] BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20095v2](http://arxiv.org/pdf/2510.20095v2)**

> **作者:** Ziheng Zhang; Xinyue Ma; Arpita Chowdhury; Elizabeth G. Campolongo; Matthew J. Thompson; Net Zhang; Samuel Stevens; Hilmar Lapp; Tanya Berger-Wolf; Yu Su; Wei-Lun Chao; Jianyang Gu
>
> **备注:** Project page: https://imageomics.github.io/biocap/
>
> **摘要:** This work investigates descriptive captions as an additional source of supervision for biological multimodal foundation models. Images and captions can be viewed as complementary samples from the latent morphospace of a species, each capturing certain biological traits. Incorporating captions during training encourages alignment with this shared latent structure, emphasizing potentially diagnostic characters while suppressing spurious correlations. The main challenge, however, lies in obtaining faithful, instance-specific captions at scale. This requirement has limited the utilization of natural language supervision in organismal biology compared with many other scientific domains. We complement this gap by generating synthetic captions with multimodal large language models (MLLMs), guided by Wikipedia-derived visual information and taxon-tailored format examples. These domain-specific contexts help reduce hallucination and yield accurate, instance-based descriptive captions. Using these captions, we train BioCAP (i.e., BioCLIP with Captions), a biological foundation model that captures rich semantics and achieves strong performance in species classification and text-image retrieval. These results demonstrate the value of descriptive captions beyond labels in bridging biological images with multimodal foundation models.
>
---
#### [replaced 095] DiscoSG: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15583v3](http://arxiv.org/pdf/2506.15583v3)**

> **作者:** Shaoqing Lin; Chong Teng; Fei Li; Donghong Ji; Lizhen Qu; Zhuang Li
>
> **备注:** EMNLP 2025 (oral), 26 pages
>
> **摘要:** Vision-Language Models (VLMs) generate discourse-level, multi-sentence visual descriptions, challenging text scene graph parsers built for single-sentence caption-to-graph mapping. Current approaches typically merge sentence-level parsing outputs for discourse input, often missing phenomena like cross-sentence coreference, resulting in fragmented graphs and degraded downstream VLM task performance. We introduce a new task, Discourse-level text Scene Graph parsing (DiscoSG), and release DiscoSG-DS, a dataset of 400 expert-annotated and 8,430 synthesised multi-sentence caption-graph pairs. Each caption averages 9 sentences, and each graph contains at least 3 times more triples than those in existing datasets. Fine-tuning GPT-4o on DiscoSG-DS yields over 40% higher SPICE metric than the best sentence-merging baseline. However, its high inference cost and licensing restrict open-source use. Smaller fine-tuned open-source models (e.g., Flan-T5) perform well on simpler graphs yet degrade on denser, more complex graphs. To bridge this gap, we introduce DiscoSG-Refiner, a lightweight open-source parser that drafts a seed graph and iteratively refines it with a novel learned graph-editing model, achieving 30% higher SPICE than the baseline while delivering 86 times faster inference than GPT-4o. It generalises from simple to dense graphs, thereby consistently improving downstream VLM tasks, including discourse-level caption evaluation and hallucination detection, outperforming alternative open-source parsers. Code and data are available at https://github.com/ShaoqLin/DiscoSG .
>
---
#### [replaced 096] DePass: Unified Feature Attributing by Simple Decomposed Forward Pass
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18462v2](http://arxiv.org/pdf/2510.18462v2)**

> **作者:** Xiangyu Hong; Che Jiang; Kai Tian; Biqing Qi; Youbang Sun; Ning Ding; Bowen Zhou
>
> **摘要:** Attributing the behavior of Transformer models to internal computations is a central challenge in mechanistic interpretability. We introduce DePass, a unified framework for feature attribution based on a single decomposed forward pass. DePass decomposes hidden states into customized additive components, then propagates them with attention scores and MLP's activations fixed. It achieves faithful, fine-grained attribution without requiring auxiliary training. We validate DePass across token-level, model component-level, and subspace-level attribution tasks, demonstrating its effectiveness and fidelity. Our experiments highlight its potential to attribute information flow between arbitrary components of a Transformer model. We hope DePass serves as a foundational tool for broader applications in interpretability.
>
---
#### [replaced 097] Reverse Engineering Human Preferences with Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15795v2](http://arxiv.org/pdf/2505.15795v2)**

> **作者:** Lisa Alazraki; Tan Yi-Chern; Jon Ander Campos; Maximilian Mozes; Marek Rei; Max Bartolo
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.
>
---
#### [replaced 098] ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17495v2](http://arxiv.org/pdf/2505.17495v2)**

> **作者:** Landon Butler; Abhineet Agarwal; Justin Singh Kang; Yigit Efe Erginbas; Bin Yu; Kannan Ramchandran
>
> **备注:** Algorithm available at: https://github.com/mmschlk/shapiq
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs $n$. Recently, Kang et al. (2025) proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to $n \approx 10^3$ features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical -- higher-order interactions are accompanied by their lower-order subsets -- which enables more efficient discovery. To exploit this hierarchy, we propose ProxySPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that ProxySPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using $10\times$ fewer inferences than SPEX. By accounting for interactions, ProxySPEX efficiently identifies the most influential features, providing a scalable approximation of their Shapley values. Further, we apply ProxySPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task.
>
---
#### [replaced 099] Marcel: A Lightweight and Open-Source Conversational Agent for University Student Support
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13937v2](http://arxiv.org/pdf/2507.13937v2)**

> **作者:** Jan Trienes; Anastasiia Derzhanskaia; Roland Schwarzkopf; Markus Mühling; Jörg Schlötterer; Christin Seifert
>
> **备注:** Accepted at EMNLP 2025 (System Demonstrations)
>
> **摘要:** We present Marcel, a lightweight and open-source conversational agent designed to support prospective students with admission-related inquiries. The system aims to provide fast and personalized responses, while reducing workload of university staff. We employ retrieval-augmented generation to ground answers in university resources and to provide users with verifiable, contextually relevant information. We introduce a Frequently Asked Question (FAQ) retriever that maps user questions to knowledge-base entries, which allows administrators to steer retrieval, and improves over standard dense/hybrid retrieval strategies. The system is engineered for easy deployment in resource-constrained academic settings. We detail the system architecture, provide a technical evaluation of its components, and report insights from a real-world deployment.
>
---
#### [replaced 100] Supporting Online Discussions: Integrating AI Into the adhocracy+ Participation Platform To Enhance Deliberation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.07780v2](http://arxiv.org/pdf/2409.07780v2)**

> **作者:** Maike Behrendt; Stefan Sylvius Wagner; Mira Warne; Jana Leonie Peters; Marc Ziegele; Stefan Harmeling
>
> **摘要:** Online spaces provide individuals with the opportunity to engage in discussions on important topics and make collective decisions, regardless of their geographic location or time zone. However, without adequate support and careful design, such discussions often suffer from a lack of structure and civility in the exchange of opinions. Artificial intelligence (AI) offers a promising avenue for helping both participants and organizers in managing large-scale online participation processes. This paper introduces an extension of adhocracy+, a large-scale open-source participation platform. Our extension features two AI-supported debate modules designed to improve discussion quality and foster participant interaction. In a large-scale user study we examined the effects and usability of both modules. We report our findings in this paper. The extended platform is available at https://github.com/mabehrendt/discuss2.0.
>
---
